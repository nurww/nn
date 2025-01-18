# trading_model.py

import sys
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
import pandas as pd
import json
import logging
from datetime import datetime
import time
import random
import torch.optim as optim
import asyncio
import redis.asyncio as aioredis

# Глобальный флаг для завершения программы
shutdown_flag = asyncio.Event()
REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'db': 0}

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from project_root.data.database_manager import execute_query

# Настройка логирования
logging.basicConfig(
    filename=f'../logs/real_binance_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

def denormalize(value):
    """
    Денормализует значение на основе диапазона.
    """
    min_value = 0
    max_value = 125000
    return value * (max_value - min_value) + min_value

def normalize(value, min_val = 0, max_val = 125000, precision=25):
    return round(value - min_val) / (max_val - min_val)

class TradingEnvironment:
    def __init__(self, agent):
        self.current_X = None
        self.agent = agent  # Устанавливаем агент
        self.state_size = 147
        self.current_step = 0  # Текущая позиция в данных
        self.leverage = 125
        self.positions = []
        self.trade_log = []
        self.prev_pnl = 0
        self.pnl = 0      # Текущий общий PnL
        self.current_price = 0  # Текущая рыночная цена
        self.start_balance = 0
        self.liquidation_price = 0
        self.initial_spot_balance=28
    
    def _set_X(self, current_X):
        self.current_X = current_X

    def reset(self, trade_balance):
        self.current_X = None
        self.initialize_balances(trade_balance=trade_balance)
        self.redistribute_balance_with_growth()
        self.current_step = 0
        self.leverage = 125
        self.positions = []
        self.trade_log = []
        self.prev_pnl = 0
        self.pnl = 0      # Текущий общий PnL
        self.current_price = 0  # Текущая рыночная цена
        self.liquidation_price = 0
        logging.info("Environment reset.")
        return self._get_state()

    def _get_state(self):
        # Рассчитываем текущий PnL по всем позициям
        if self.current_X is None:
            logging.warning("Current data (current_X) is not set.")
            return {
                "state_data": np.zeros(self.state_size, dtype=np.float32),
                "positions": [],
                "current_step": self.current_step,  # Добавляем current_step
                "trade_log": self.trade_log,  # Добавляем trade_log
                "current_pnl": 0.0,  # Добавляем текущий PnL
                "liquidation_price": 0.0
            }
        
        # current_pnl = sum(self._calculate_pnl(pos) for pos in self.positions)
        current_pnl = 0
        
        state = {
            "state_data": np.concatenate([
                np.array([
                    float(self.spot_balance),
                    float(self.futures_balance),
                    float(self.leverage),
                    float(len(self.positions)),
                    float(self.current_step - self.positions[0]["entry_step"] if self.positions else 0),
                    float(current_pnl),  # Добавляем текущий PnL
                    float(self.liquidation_price)  # Добавляем liquidation_price
                ], dtype=np.float32),
                np.array(self.current_X, dtype=np.float32)
            ]),
            "positions": self.positions,
            "current_step": self.current_step,  # Добавляем current_step
            "trade_log": self.trade_log,  # Добавляем trade_log
            "current_pnl": current_pnl,  # Добавляем текущий PnL
            "liquidation_price": self.liquidation_price  # Liquidation price
        }
        return state

    def step(self, action):
        reward = 0
        done = False

        # Обновление текущей цены
        self.current_price = self.current_X[0]  # Цена текущей свечи
        # Анализ трендов
        trends = self.analyze_trends()
        imbalance = self.current_X[3]

        # Штраф за ликвидацию через reward
        if self._check_liquidation():
            # logging.info(f"step() 'before' if self._check_liquidation(): - reward: {reward}")
            reward -= 10  # Штраф за ликвидацию
            # logging.info(f"step() 'after' if self._check_liquidation(): - reward: {reward}")
            return self._get_state(), reward, done
        
        if action[0] == 0:  # Open Long
            direction = self.determine_direction(trends, imbalance)
            if direction == "long":
                self._open_position("long")
            else:
                reward -= 0.1
        elif action[0] == 1:  # Open Short
            direction = self.determine_direction(trends, imbalance)
            if direction == "short":
                self._open_position("short")
            else:
                reward -= 0.1
        elif action[0] == 2:  # Close Positions
            # Логика удержания позиций
            for position in self.positions:
                if (position["direction"] == "long" and self.current_price >= position["take_profit"]) or \
                (position["direction"] == "short" and self.current_price <= position["take_profit"]):
                    total_pnl, closed, errors = self._close_positions()
                    # logging.info(f"step() Close Positions - reward: {reward}, total_pnl: {total_pnl}")
                    reward += total_pnl * 0.1
                elif (position["direction"] == "long" and self.current_price <= position["stop_loss"]) or \
                    (position["direction"] == "short" and self.current_price >= position["stop_loss"]):
                    total_pnl, closed, errors = self._close_positions()
                    # logging.info(f"step() Close Positions - reward: {reward}, total_pnl: {total_pnl}")
                    reward += total_pnl * 0.1
        elif action[0] == 3:  # Hold
            pass

        # Вознаграждение за общий баланс
        total_balance = self.spot_balance + self.futures_balance
        # logging.info(f"step() reward: {reward}, total_balance: {total_balance}, spot_balance: {self.spot_balance}, futures_balance: {self.futures_balance}")
        reward += (total_balance - self.initial_total_balance) * 0.05
        # logging.info(f"step() reward: {reward}, total_balance: {total_balance}, spot_balance: {self.spot_balance}, futures_balance: {self.futures_balance}")
        self.initial_total_balance = total_balance
        
        # Логируем состояние только каждые 250 шагов
        if self.current_step % 250 == 0:
            logging.info(f"Step: {self.current_step}, Total balance: {total_balance}, Spot balance: {self.spot_balance:.5f}, "
                        f"Futures balance: {self.futures_balance:.5f}, Positions: {len(self.positions)}")
        if self.positions:
            for i, position in enumerate(self.positions, start=1):
                self.adjust_stop_loss_take_profit(position)
                # logging.info(f"Position {i}: {position}")

        if self.current_step % 3000 == 0:
            trade_report = self.generate_trade_report()
            with pd.option_context('display.float_format', '{:.6f}'.format):  # Устанавливаем фиксированное количество знаков после запятой
                logging.info(f"Trade report: \n{trade_report}")
            # Очистка после логирования
            self.clean_closed_trades()

        # Переход на следующий шаг
        self.current_step += 1
        
        return self._get_state(), reward, done
    
    def clean_closed_trades(self):
        initial_count = len(self.trade_log)
        self.trade_log = [log for log in self.trade_log if log.get("action") != "close"]
        removed_count = initial_count - len(self.trade_log)
        logging.info(f"Removed {removed_count} closed trades from trade_log.")

    def initialize_balances(self, trade_balance):
        """
        Инициализирует начальные балансы на основе trade_balance и эталонного значения.
        """
        self.trade_balance = trade_balance  # Устанавливаем единичный trade_balance (например, 7)

        # Проверяем и устанавливаем спотовый и фьючерсный баланс
        if self.initial_spot_balance is not None:
            # Устанавливаем указанный спотовый баланс
            self.spot_balance = self.initial_spot_balance
            self.target_BB = (self.spot_balance / 4) * 10  # Эталонный уровень баланса
            self.futures_balance = trade_balance - self.spot_balance
        else:
            # Пропорции по умолчанию: 40% спот и 60% фьючерсы
            self.target_spot_balance = self.target_BB * 0.4
            self.target_futures_balance = self.target_BB * 0.6
            self.spot_balance = self.target_spot_balance
            self.futures_balance = self.target_BB - self.target_spot_balance

        # Сохраняем общий баланс и начальные данные
        self.initial_total_balance = self.spot_balance + self.futures_balance
        self.start_balance = self.trade_balance

        logging.info(
            f"Balances initialized: Spot: {self.spot_balance}, Futures: {self.futures_balance}, "
            f"Total: {self.target_BB} (Target trade_balance: {self.target_BB}, Unit trade_balance: {self.trade_balance})"
        )

    def redistribute_balance_with_growth(self):
        """
        Перераспределяет баланс в зависимости от текущего состояния с учетом уровней.
        """
        total_balance = self.spot_balance + self.futures_balance

        # Если общий баланс достиг уровня 1.75 * target_BB
        if total_balance >= self.target_BB * 1.75:
            growth_factor = 1.75
            logging.info(f"Balance growth reached {growth_factor * 100}% of the target level.")

            # Перераспределяем спот и фьючерсные балансы
            self.spot_balance = self.target_BB * 1.75 * 0.4  # Поддерживаем 40% на споте
            self.futures_balance = total_balance - self.spot_balance

            # Обновляем target_BB для следующего уровня
            self.target_BB = self.target_BB * growth_factor
            logging.info(f"Updated growth target_BB to {self.target_BB}.")

        # Если общий баланс упал ниже 0.75 уровня
        elif total_balance < self.target_BB * 0.5714:
            shrink_factor = 0.5714
            logging.info(f"Balance dropped below {shrink_factor * 100}% of the target level.")

            # Перераспределяем спот и фьючерсные балансы
            self.spot_balance = self.target_BB * 0.5714 * 0.4  # Поддерживаем 40% на споте
            self.futures_balance = total_balance - self.spot_balance

            # Обновляем target_BB для следующего уровня
            self.target_BB = self.target_BB * shrink_factor
            logging.info(f"Updated shrink target_BB to {self.target_BB}.")
        # logging.info(
        #     f"Balances after redistribution: Spot: {self.spot_balance:.2f}, "
        #     f"Futures: {self.futures_balance:.2f}, Total: {total_balance:.2f}"
        # )

    def get_trade_balance(self):
        """
        Возвращает сумму, доступную для торговли (50% от фьючерсного баланса).
        """
        return self.futures_balance / 2
    
    def retrospective_analysis(self, closed_position, future_data):
        """
        Анализ завершенной позиции.
        """
        entry_price = closed_position["entry_price"]
        direction = closed_position["direction"]
        position_size = closed_position["position_size"]
        self.liquidation_price = closed_position["liquidation_price"]

        if len(future_data) == 0:
            logging.warning("No future data available for retrospective analysis.")
            return None

        # Проверяем, могла ли сделка быть ликвидирована
        safe_range = (future_data >= self.liquidation_price) if direction == "long" else (future_data <= self.liquidation_price)
        if not safe_range.all():
            logging.warning("Position would have been liquidated. Retrospective analysis skipped.")
            return None

        # Ищем лучшую возможную цену для выхода
        if direction == "long":
            best_exit_price = future_data.max()
            best_pnl = (best_exit_price - entry_price) * position_size
            # logging.info(f"retrospective_analysis() - best_pnl: {best_pnl}, entry_price: {entry_price}, best_exit_price: {best_exit_price}, position_size: {position_size}")
        else:
            best_exit_price = future_data.min()
            best_pnl = (entry_price - best_exit_price) * position_size
            # logging.info(f"retrospective_analysis() - best_pnl: {best_pnl}, entry_price: {entry_price}, best_exit_price: {best_exit_price}, position_size: {position_size}")
        
        # Сравниваем лучший PnL с фактическим
        actual_pnl = closed_position["pnl"]
        if best_pnl > actual_pnl:
            reward = best_pnl - actual_pnl
            # logging.info(f"retrospective_analysis() 'if best_pnl > actual_pnl' - reward: {reward}, best_pnl: {best_pnl}, actual_pnl: {actual_pnl}")
            state = self._get_state()
            action = [2, []]
            # logging.info(f"Retrospective analysis: Missed opportunity to close at {best_exit_price:.5f}, "
            #             f"Best PnL: {best_pnl:.5f}, Actual PnL: {actual_pnl:.5f}")
            return state, action, reward
        # logging.info(f"retrospective_analysis() return None? actual_pnl: {actual_pnl}")
        return None

    def close_all_positions(self, client, symbol):
        """
        Закрывает все открытые позиции по указанному символу.
        """
        try:
            position_info = self.get_position_info(client, symbol)
            if not position_info or position_info["position_size"] == 0:
                logging.info("No open positions to close.")
                return None

            side = "SELL" if float(position_info["position_size"]) > 0 else "BUY"
            quantity = abs(position_info["position_size"])

            order = self.close_position(client, symbol, side, quantity)
            return order
        except BinanceAPIException as e:
            logging.error(f"Error closing position: {e}")
            return None

    def close_position(self, client, symbol, side, quantity, order_type="MARKET", price=None):
        """
        Закрывает позицию на фьючерсах Binance.
        """
        try:
            if order_type == "MARKET":
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                    reduceOnly=True  # Флаг, указывающий, что ордер только закрывает позицию
                )
            elif order_type == "LIMIT":
                if price is None:
                    raise ValueError("Price must be specified for LIMIT orders.")
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    quantity=quantity,
                    price=price,
                    timeInForce="GTC",
                    reduceOnly=True
                )
            return order
        except BinanceAPIException as e:
            logging.error(f"Error closing position: {e}")
            return None

    def close_all_positions(self):
        for position in self.positions:
            total_pnl = self._calculate_pnl(position)  # Фактический PnL
            position["pnl"] = total_pnl  # Добавляем PnL в позицию
            future_data = self.get_future_data(position["entry_step"])  # Данные после закрытия
            retrospective_experience = self.retrospective_analysis(position, future_data)
            if retrospective_experience:
                state, action, reward = retrospective_experience
                self.agent.store_experience(state, action, reward, self._get_state(), False)
        self.positions = []  # Очищаем активные позиции

    def get_position_info(client, symbol):
        """
        Получает информацию о позиции для указанного символа.
        Возвращает PnL, ликвидационную цену, размер позиции и другие данные.
        """
        try:
            positions = client.futures_position_information()
            for position in positions:
                if position['symbol'] == symbol:
                    return {
                        "pnl": float(position['unRealizedProfit']),
                        "liquidation_price": float(position['liquidationPrice']),
                        "entry_price": float(position['entryPrice']),
                        "position_size": float(position['positionAmt']),
                        "leverage": int(position['leverage']),
                        "margin_type": position['marginType']
                    }
            return None
        except Exception as e:
            logging.error(f"Error fetching position info: {e}")
            return None

    def analyze_trends(self):
        """
        Анализ трендов для 5-минутного и 1-минутного интервалов.
        """
        return {
            "5m_trend_diff": self.current_X[100],  # trend_price_difference для 5 минут
            "5m_trend_strength": self.current_X[102],  # trend_strength для 5 минут
            "1m_trend_diff": self.current_X[122],  # trend_price_difference для 1 минуты
            "1m_trend_strength": self.current_X[124],  # trend_strength для 1 минуты
            "atr": self.current_X[4],  # ATR
            "sma": self.current_X[5],  # SMA
            "mad": self.current_X[6],  # MAD
            "cci": self.current_X[7],  # CCI
        }

    def adjust_stop_loss_take_profit(self, position):
        """
        Улучшает stop_loss и take_profit позиции на основе предсказаний модели,
        с корректировкой на основе рыночных данных и относительных процентов.
        """
        # Анализ текущей цены
        current_price = denormalize(self.current_price)

        # Получение предсказаний модели
        _, predicted_stop_loss, predicted_take_profit = self.agent.predict(self._get_state())
        predicted_stop_loss = float(predicted_stop_loss)
        predicted_take_profit = float(predicted_take_profit)

        # Преобразование в абсолютные значения
        proposed_stop_loss = current_price * (1 - predicted_stop_loss)
        proposed_take_profit = current_price * (1 + predicted_take_profit)
        predicted_stop_loss = proposed_stop_loss
        predicted_take_profit = proposed_take_profit

        # Получение рыночных границ
        interval_low_5m = denormalize(self.current_X[98])  # Минимальная цена из интервала
        interval_high_5m = denormalize(self.current_X[99])  # Максимальная цена из интервала
        interval_low_1m = denormalize(self.current_X[120])  # Минимальная цена из интервала
        interval_high_1m = denormalize(self.current_X[121])  # Максимальная цена из интервала
        
        # Определяем допустимые диапазоны с отклонением в процентах
        # Рассчёт допустимых диапазонов (5m + 1m) с допуском
        tolerance = 0.003  # Допустимое отклонение (3%)
        adjusted_low = max(interval_low_5m, interval_low_1m) * (1 - tolerance)
        adjusted_high = min(interval_high_5m, interval_high_1m) * (1 + tolerance)

        # Проверяем границы ликвидационной цены
        liquidation_price = position["liquidation_price"]
        adjusted_low = max(adjusted_low, liquidation_price * 1.005)  # Не ниже ликвидации
        adjusted_high = min(adjusted_high, liquidation_price * 0.995)  # Не выше ликвидации

        # Корректируем значения для long позиции
        if position["direction"] == "long":
            if proposed_stop_loss < adjusted_low:
                proposed_stop_loss = adjusted_low
            if proposed_take_profit > adjusted_high:
                proposed_take_profit = adjusted_high

        # Корректируем значения для short позиции
        elif position["direction"] == "short":
            if proposed_stop_loss > adjusted_high:
                proposed_stop_loss = adjusted_high
            if proposed_take_profit < adjusted_low:
                proposed_take_profit = adjusted_low

        # Прямое присвоение значений stop_loss и take_profit
        position["stop_loss"] = proposed_stop_loss
        position["take_profit"] = proposed_take_profit
        position["predicted_stop_loss"] = predicted_stop_loss
        position["predicted_take_profit"] = predicted_take_profit

    def determine_direction(self, trends, imbalance):
        """
        Определяет направление сделки на основе данных интервалов, стакана и новых индикаторов.
        """
        if trends["5m_trend_strength"] > 0 and trends["1m_trend_strength"] > 0 and imbalance > 0 and trends["cci"] > 100:
            # Сильный тренд вверх, CCI подтверждает перекупленность
            return "long"
        elif trends["5m_trend_strength"] < 0 and trends["1m_trend_strength"] < 0 and imbalance < 0 and trends["cci"] < -100:
            # Сильный тренд вниз, CCI подтверждает перепроданность
            return "short"
        else:
            # ATR слишком низкий — рынок вялый
            if trends["atr"] < 0.01:
                return None  # Никакого направления, недостаточно сигналов

    def train_on_experience(self):
        """
        Обучает модель на основе сохраненного опыта.
        """
        for trade in self.trade_log:
            # Учитываем только закрытые сделки
            if trade.get('action') != 'close':
                logging.warning(f"Skipping trade without 'pnl': {trade}")
                continue

            try:
                # Проверяем наличие всех необходимых данных
                entry_price = trade.get("entry_price")
                exit_price = trade.get("exit_price")
                position_size = trade.get("position_size")
                actual_pnl = trade.get("pnl")
                entry_data = trade.get("entry_data", None)

                if not all([entry_price, exit_price, position_size, actual_pnl]):
                    logging.warning(f"Missing critical data in trade: {trade}")
                    continue

                # Рассчитываем максимальную возможную прибыль
                max_pnl = (exit_price - entry_price) * (position_size / self.current_price)

                # logging.info(
                #     f"train_on_experience - entry_price: {entry_price}, "
                #     f"exit_price: {exit_price}, max_pnl: {max_pnl}, actual_pnl: {actual_pnl}"
                # )

                # Рассчитываем вознаграждение
                reward = max_pnl - actual_pnl if max_pnl > actual_pnl else 0
                # logging.info(f"train_on_experience() reward: {reward}, max_pnl: {max_pnl}, actual_pnl: {actual_pnl}")
                # logging.info(f"train_on_experience max_pnl: {max_pnl}, actual_pnl: {actual_pnl}")

                # Используем последнее корректное состояние
                next_state = self._get_state()

                # Проверяем валидность `next_state`
                if "state_data" not in next_state or "positions" not in next_state:
                    logging.error(f"Invalid next_state to store: {next_state}")
                    continue

                # Сохраняем опыт
                self.agent.store_experience(entry_data, [2, []], reward, next_state, False)

            except KeyError as e:
                logging.error(f"Missing key in trade data: {e}. Trade: {trade}")
            except Exception as e:
                logging.error(f"Error processing trade: {e}")
    
    def get_account_balances(self, client):
        # Получение спотового баланса
        spot_balance = client.get_account()
        spot_usdt = next(item for item in spot_balance['balances'] if item['asset'] == 'USDT')['free']
        
        # Получение фьючерсного баланса
        futures_balance = client.futures_account_balance()
        futures_usdt = next(item for item in futures_balance if item['asset'] == 'USDT')['balance']
        
        return float(spot_usdt), float(futures_usdt)

    def open_market_position(client, symbol, side, quantity, leverage):
        """
        Открывает рыночную позицию на фьючерсах Binance.
        """
        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            return order
        except BinanceAPIException as e:
            logging.error(f"Error opening position: {e}")
            return None

    def open_position(self, client, symbol, side, quantity, leverage, order_type="MARKET", price=None):
        """
        Открывает позицию на фьючерсах Binance.
        
        :param client: Binance API клиент
        :param symbol: Строка, например "BTCUSDT"
        :param side: "BUY" или "SELL"
        :param quantity: Количество базового актива
        :param leverage: Плечо
        :param order_type: Тип ордера, например "MARKET" или "LIMIT"
        :param price: Цена (для лимитного ордера)
        """
        try:
            # Установка плеча
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            
            # Отправка ордера
            if order_type == "MARKET":
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity
                )
            elif order_type == "LIMIT":
                if price is None:
                    raise ValueError("Price must be specified for LIMIT orders.")
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    quantity=quantity,
                    price=price,
                    timeInForce="GTC"
                )
            return order
        except BinanceAPIException as e:
            logging.error(f"Error opening position: {e}")
            return None

    def _open_position(self, direction):
        # Проверка наличия активных позиций
        if len(self.positions) >= 1:  # Ограничение на одну активную позицию
            # logging.warning("Position not opened: An active position already exists.")
            return
        
        mid_price = denormalize(self.current_price)  # mid_price из стакана
        # Используем половину фьючерсного баланса для торговли
        trade_balance = self.get_trade_balance()
        position_size = trade_balance
        total_position_size = (self.leverage * position_size) / mid_price
        
        if self.futures_balance <= position_size / mid_price:
            return
        
        self.liquidation_price = self._calculate_liquidation_price(total_position_size, mid_price, direction)
        self.futures_balance -= position_size / mid_price

        position_id = len(self.positions) + 1

        # Добавляем позицию
        position = {
            "position_id": position_id,
            "entry_price": mid_price,
            "pos_s": position_size,
            "position_size": total_position_size,
            "direction": direction,
            "liquidation_price": self.liquidation_price,
            "entry_step": self.current_step,
            "stop_loss": None,
            "take_profit": None,
            "predicted_stop_loss": None,
            "predicted_take_profit": None,
        }
        
        self.adjust_stop_loss_take_profit(position)
        self.positions.append(position)

    def _close_positions(self):
        successfully_closed = []
        errors = []
        total_pnl = 0
        total_reward = 0

        # Устанавливаем комиссию как долю от размера позиции
        # commission_rate = 0.1  # 0.1% комиссии как на Binance
        # Устанавливаем комиссии
        maker_commission_rate = 0.0002  # 0.02%
        taker_commission_rate = 0.0004  # 0.04%
        funding_rate = 0.0001  # Пример ставки финансирования (нужно брать актуальное значение)

        for i, position in enumerate(self.positions):
            reward = 0  # Инициализация reward
            try:
                pnl = self._calculate_pnl(position)
                # position_size = position.get("position_size", 0) / denormalize(self.current_price)

                maker_commission = position.get("pos_s", 0) * maker_commission_rate
                taker_commission = position.get("pos_s", 0) * taker_commission_rate
                # Финансирование
                funding_fee = position.get("pos_s", 0) * funding_rate                
                # Общая комиссия
                total_commission = maker_commission + taker_commission + funding_fee
                # logging.info(f"Comission - total_commission: {total_commission}, maker_commission: {maker_commission}")
                # logging.info(f"taker_commission: {taker_commission}, funding_fee: {funding_fee}")

                # commission = position_size * total_commission
                pnl -= total_commission
                total_pnl += pnl
                self.futures_balance += total_pnl
                successfully_closed.append(i)
                # Обновляем PnL в позиции
                position["pnl"] = pnl
                # Рассчитываем награду/штраф на основе результата сделки
                reward = self._calculate_reward(position)
                total_reward += reward
            except Exception as e:
                errors.append((i, str(e)))
                logging.error(f"Error closing position {i}: {e}")
            finally:
                # Логирование закрытия сделки
                self.trade_log.append({
                    "action": "close",
                    "entry_step": position["entry_step"],
                    "step": self.current_step,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": denormalize(self.current_price),
                    "position_size": position["position_size"],
                    "liquidation_price": position["liquidation_price"],
                    "stop_loss": position["stop_loss"],
                    "take_profit": position["take_profit"],
                    "pnl": pnl,
                    "commission": total_commission,
                    "exit_spot_balance": self.spot_balance,
                    "exit_futures_balance": self.futures_balance,
                    "entry_data": self._get_state()
                })

        # Передача total_reward агенту
        if total_reward != 0:
            state = self._get_state()  # Получаем текущее состояние
            # logging.info(f"_close_positions() 'if total_reward != 0:' - total_reward: {total_reward}")
            self.agent.store_experience(state, [3, []], total_reward, state, True)  # Используем фиктивное действие [3, []]
        
        # Логируем окончательный баланс
        # logging.info(f"Final Futures balance after closing positions: {self.futures_balance:.5f} Spot {self.spot_balance:.5f}")
        if self.futures_balance <= 0:
            logging.warning("Futures balance reached zero. All positions liquidated.")

        # logging.info(f"_close_positions() 'before return' - total_pnl: {total_pnl}")
        self.positions = []  # Закрываем все позиции
        self.redistribute_balance_with_growth()
        return total_pnl, successfully_closed, errors

    def generate_trade_report(self):
        """
        Генерация отчета по сделкам на основе trade_log.
        """
        report_data = []

        for log in self.trade_log:
            if log["action"] == "close":
                report_data.append({
                    "Вход (шаг)": log.get("entry_step", "N/A"),
                    "Выход (шаг)": log.get("step", "N/A"),
                    "Цена входа": log.get("entry_price", "N/A"),
                    "Цена выхода": log.get("exit_price", "N/A"),
                    "Направление": log.get("direction", "N/A"),
                    "Прибыль/Убыток": log.get("pnl", "N/A"),
                    "Размер позиции": log.get("position_size", "N/A"),
                    "Ликвидация": log.get("liquidation_price", "N/A"),
                    "Комиссия": log.get("commission", "N/A"),
                    "Stop Loss": log.get("stop_loss", "N/A"),
                    "Take Profit": log.get("take_profit", "N/A"),
                    "Конечный баланс (Spot)": log.get("exit_spot_balance", "N/A"),
                    "Конечный баланс (Futures)": log.get("exit_futures_balance", "N/A"),
                    # "ATR при входе": self.current_X[log.get("entry_step", 0)][4],  # ATR при входе
                    # "SMA при входе": self.current_X[log.get("entry_step", 0)][5],  # SMA при входе
                    # "MAD при входе": self.current_X[log.get("entry_step", 0)][6],  # MAD при входе
                    # "CCI при входе": self.current_X[log.get("entry_step", 0)][7],  # CCI при входе
                })

        # Проверяем, есть ли данные для отчета
        if not report_data:
            logging.warning("Trade log is empty. No data for report.")
            return pd.DataFrame()  # Возвращаем пустой DataFrame, если лог пуст

        # Создаем DataFrame из данных
        trade_report = pd.DataFrame(report_data)

        # Добавляем метрику
        trade_report["Win"] = trade_report["Прибыль/Убыток"].apply(lambda x: 1 if x > 0 else 0)
        win_rate = trade_report["Win"].mean() if not trade_report.empty else 0
        profitability_ratio = trade_report["Прибыль/Убыток"].sum() / max(trade_report["Прибыль/Убыток"].abs().sum(), 1)

        logging.info(f"Current Last Step: {self.current_step}")
        logging.info(f"Total balance: {self.spot_balance + self.futures_balance}, Spot balance: {self.spot_balance}, Futures balance: {self.futures_balance}")
        # Логируем метрики
        logging.info(f"Win Rate: {win_rate:.5f}, Profitability Ratio: {profitability_ratio:.5f}")

        # Настройки pandas для отображения
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        return trade_report

    def _update_futures_balance_with_pnl(self):
        """
        Обновляет баланс фьючерсов на основе текущих PnL по всем позициям.
        """
        total_pnl = sum(self._calculate_pnl(position) for position in self.positions)
        self.futures_balance += total_pnl
        # logging.info(f"Updated Futures balance with PnL: {self.futures_balance:.5f}")

    def _transfer_balance(self, direction):
        transfer_amount = 0
        min_transfer_threshold = 5  # Минимальная сумма перевода

        if direction == "spot_to_futures":
            if self.spot_balance > min_transfer_threshold:
                transfer_amount = min(self.spot_balance * 0.1, self.spot_balance)
                self.spot_balance -= transfer_amount
                self.futures_balance += transfer_amount
            else:
                logging.info("Spot balance too low for transfer to futures.")
        elif direction == "futures_to_spot":
            min_required_margin = sum(pos["position_size"] / self.leverage for pos in self.positions)
            if self.futures_balance - min_required_margin > min_transfer_threshold:
                transfer_amount = min(self.futures_balance * 0.1, self.futures_balance - min_required_margin)
                self.futures_balance -= transfer_amount
                self.spot_balance += transfer_amount
            else:
                logging.info("Futures balance too low or insufficient margin for transfer to spot.")

        if transfer_amount > 0:
            logging.info(f"Transferred {transfer_amount:.5f} from {direction.replace('_', ' ')}. "
                        f"Spot balance: {self.spot_balance:.5f}, Futures balance: {self.futures_balance:.5f}")
        return transfer_amount

    def _calculate_liquidation_price(self, position_size, entry_price, direction, maintenance_margin_rate=0.05):
        # Средний процент отклонения для long и short
        correction_factor = 0.0039  # 0.39%
        # Рассчитываем Maintenance Margin
        maintenance_margin = position_size * maintenance_margin_rate
        # Рассчитываем цену ликвидации
        if direction == "long":
            liquidation_price = entry_price - ((self.futures_balance - maintenance_margin) / position_size)
            # Корректировка для long
            liquidation_price *= (1 + correction_factor)
        elif direction == "short":
            liquidation_price = entry_price + ((self.futures_balance - maintenance_margin) / position_size)
            # Корректировка для short
            liquidation_price *= (1 - correction_factor)
        else:
            raise ValueError("Direction must be 'long' or 'short'")

        return round(liquidation_price, 5)

    def _calculate_pnl(self, position):
        """
        Расчет PnL по позиции.
        """
        if position["direction"] == "long":
            # logging.info(f"_calculate_pnl() - current price: {denormalize(self.current_price)}, Entry Price: {position['entry_price']}, Position Size: {position['position_size']}")
            return (denormalize(self.current_price) - position["entry_price"]) * (position["position_size"] / self.current_price)
        elif position["direction"] == "short":
            # logging.info(f"_calculate_pnl() - current price: {denormalize(self.current_price)}, Entry Price: {position['entry_price']}, Position Size: {position['position_size']}")
            return (position["entry_price"] - denormalize(self.current_price)) * (position["position_size"] / self.current_price)
        return 0

    def calculate_volatility_reward(self, pnl, atr, trends):
        """
        Учитывает риски и волатильность в расчете награды.
        """
        reward = pnl  # Исходное значение награды
        # logging.info(f"calculate_volatility_reward() - 'begin' - reward: {reward}, pnl: {pnl}")
        # Учет волатильности через ATR
        if atr < 0.01:  # Низкая волатильность
            reward *= 1.2  # Бонус за прибыль в стабильных условиях
        elif atr > 0.05:  # Высокая волатильность
            reward *= 0.8  # Штраф за прибыль в условиях высокой волатильности

        # Учет MAD (Mean Absolute Deviation) из трендов
        if trends["mad"] > 0.02:  # Если MAD слишком высок
            reward -= 0.1  # Штраф

        # logging.info(f"calculate_volatility_reward() - 'finish' - reward: {reward}, pnl: {pnl}")
        return reward

    def calculate_hold_reward(self, position, current_step):
        """
        Рассчитывает награду за долгосрочное удержание позиции.
        """
        entry_step = position["entry_step"]
        pnl = position["pnl"]
        hold_duration = current_step - entry_step
        hold_reward = max(0, pnl * hold_duration * 0.01)  # Увеличиваем награду за длительное удержание
        
        # logging.info(f"calculate_hold_reward() - pnl: {pnl}, hold_duration: {hold_duration}, hold_reward: {hold_reward}")
        return hold_reward

    def calculate_liquidation_penalty(self, position, current_price, atr):
        """
        Рассчитывает штраф за близость к ликвидации.
        """
        self.liquidation_price = position["liquidation_price"]

        # Проверка на нулевой или слишком малый ATR
        if atr <= 1e-6:  # Используем небольшой порог вместо чисто нуля
            # logging.warning(f"calculate_liquidation_penalty() - ATR слишком мал или равен 0: {atr}")
            return 10  # Максимальный штраф, так как риск ликвидации трудно оценить

        time_to_liquidation = abs(current_price - self.liquidation_price) / atr

        if time_to_liquidation < 3:  # Если до ликвидации меньше 3 ATR
            penalty = 10 / time_to_liquidation  # Штраф обратно пропорционален времени
            # logging.info(f"calculate_liquidation_penalty() - liquidation_price: {self.liquidation_price}, time_to_liquidation: {time_to_liquidation}")
            return penalty

        return 0

    def calculate_small_loss_penalty(self, pnl):
        """
        Рассчитывает штраф за малые убытки.
        """
        if pnl >= 0:
            return 0
        return np.log(1 + abs(pnl)) * 0.5  # Нелинейный штраф

    def calculate_pnl_reward(self, position, trends, current_step, current_price, atr):
        """
        Итоговая функция расчета награды.
        """
        pnl = position["pnl"]
        
        # Основная награда за PnL
        max_pnl = (position["take_profit"] - position["entry_price"]) * position["position_size"]
        min_pnl = (position["stop_loss"] - position["entry_price"]) * position["position_size"]
        # logging.info(f"calculate_pnl_reward() - stop_loss: {position['stop_loss']}, entry_price: {position['entry_price']}, position_size: {position['position_size']}")
        # logging.info(f"calculate_pnl_reward() - pnl: {pnl}, max_pnl: {max_pnl}, min_pnl: {min_pnl}")

        if pnl > 0:
            reward = (pnl / max_pnl) * 2 if pnl <= max_pnl else pnl * 0.1
        else:
            reward = (pnl / min_pnl) * 0.5 if pnl >= min_pnl else pnl * 0.1

        # Учет волатильности
        reward = self.calculate_volatility_reward(reward, atr, trends)
        # logging.info(f"calculate_pnl_reward() - 1 - reward: {reward}")
        
        # Награда за удержание позиции
        reward += self.calculate_hold_reward(position, current_step)
        # logging.info(f"calculate_pnl_reward() - 2 - reward: {reward}")

        # Штраф за близость к ликвидации
        reward -= self.calculate_liquidation_penalty(position, current_price, atr)
        # logging.info(f"calculate_pnl_reward() - 3 - reward: {reward}")

        # Уменьшение штрафов за малые убытки
        reward -= self.calculate_small_loss_penalty(pnl)
        # logging.info(f"calculate_pnl_reward() - 4 - reward: {reward}")
        return reward

    def _calculate_reward(self, position):
        trends = self.analyze_trends()
        atr = trends["atr"]
        current_price = denormalize(self.current_price)
        
        return self.calculate_pnl_reward(position, trends, self.current_step, current_price, atr)

    reward_weights = {
        "pnl_weight": 0.5,
        "volatility_weight": 0.3,
        "hold_weight": 0.1,
        "liquidation_penalty_weight": 0.1,
        "loss_penalty_weight": 0.2
    }

    def _check_liquidation(self):
        """
        Проверяет ликвидацию всех открытых позиций.
        Возвращает True, если произошла ликвидация, иначе False.
        """
        liquidation_triggered = False
        # positions_to_remove = []

        for position in self.positions:
            # logging.info(f"Liquidation check: {position['liquidation_price']}")
            if position["direction"] == "long":
                if denormalize(self.current_price) <= position['liquidation_price']:
                    self.positions.remove(position)
                    liquidation_triggered = True
            elif position["direction"] == "short":
                if denormalize(self.current_price) >= position['liquidation_price']:
                    self.positions.remove(position)
                    liquidation_triggered = True
        
        # Если ликвидация произошла, сбрасываем фьючерсный баланс
        if liquidation_triggered:
            self.futures_balance = max(self.futures_balance, 0)  # Баланс не может быть меньше нуля
            # logging.info(f"Liquidation occurred at price {denormalize(self.current_price)}. Updated Futures balance: {self.futures_balance:.5f}")
        return liquidation_triggered

    def _log_state(self):
        # Логируем состояние только каждые 100 шагов
        if self.current_step % 1000 == 0:
            logging.info(f"Step: {self.current_step}, Spot balance: {self.spot_balance:.5f}, "
                        f"Futures balance: {self.futures_balance:.5f}, Positions: {len(self.positions)}")
            # if self.positions:
            #     for i, position in enumerate(self.positions, start=1):
            #         logging.info(f"Position {i}: {position}")
        # logging.info(f"Step: {self.current_step}, Spot balance: {self.spot_balance:.5f}, "
        #              f"Futures balance: {self.futures_balance:.5f}, Positions: {len(self.positions)}")
        # if self.positions:
        #     for i, position in enumerate(self.positions, start=1):
        #         logging.info(f"Position {i}: {position}")

class DQLAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # Параметры DQL
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Опыт для обучения
        self.memory = []
        self.batch_size = 64
        self.memory_capacity = 10000

        # Инициализация PnL
        self.pnl = 0

        # **Инициализация trade_log**
        self.trade_log = []  # Добавляем атрибут для хранения журнала сделок

        # Модель
        self.q_network = self._build_model().to("cuda")
        self.target_network = self._build_model().to("cuda")
        self.update_target_network()  # Синхронизируем веса в начале

        # Оптимизатор
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def _build_model(self):
        """
        Строит нейронную сеть для Q-функции.
        """
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size + 2)
        )

    def update_target_network(self):
        """
        Синхронизирует целевую сеть с основной.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state):
        """
        Выбирает действие на основе Q-values.
        """
        state_tensor = torch.tensor(state["state_data"], dtype=torch.float32).unsqueeze(0).to("cuda")
        with torch.no_grad():
            q_values = self.q_network(state_tensor)  # Получаем значения Q-функции для всех действий

        if np.random.rand() <= self.epsilon:  # Исследование
            action_index = random.choice(range(self.action_size))
        else:  # Эксплуатация
            action_index = torch.argmax(q_values[:self.action_size]).item()  # Максимальное значение Q-функции
        
        return [action_index]

    def store_experience(self, state, action, reward, next_state, done):
        if not isinstance(state, dict) or "state_data" not in state:
            raise ValueError(f"Invalid state to store: {state}")
        if not isinstance(next_state, dict) or "state_data" not in next_state:
            raise ValueError(f"Invalid next_state to store: {next_state}")
        # Рассчитываем дополнительную награду за удержание позиции, избегание ликвидации
        pnl_reward = state["current_pnl"] * 0.1  # Вес PnL
        liquidation_penalty = -1.0 if state["liquidation_price"] <= state["state_data"][3] else 0.0  # Штраф за ликвидацию

        adjusted_reward = reward + pnl_reward + liquidation_penalty

        # logging.info(f"store_experience() - reward: {reward}")
        self.memory.append((state, action, adjusted_reward, next_state, done))

    @staticmethod
    def convert_state_to_array(state):
        if isinstance(state, dict) and "state_data" in state:
            return state["state_data"]
        raise TypeError("State must be a dictionary containing 'state_data'.")

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Выбор случайной выборки
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Преобразование данных
        states = torch.stack([torch.tensor(state["state_data"], dtype=torch.float32).to("cuda") for state in states])
        actions = torch.tensor([action[0] for action in actions], dtype=torch.int64).unsqueeze(1).to("cuda")
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to("cuda")
        next_states = torch.stack([torch.tensor(next_state["state_data"], dtype=torch.float32).to("cuda") for next_state in next_states])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to("cuda")

        # Q-значения текущих состояний
        q_values = self.q_network(states).gather(1, actions)

        # Q-значения следующих состояний
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]

        # Расчет целевых значений (TD Target)
        targets = rewards + (1 - dones) * self.discount_factor * max_next_q_values

        # Потеря для Q-значений
        loss_q = nn.MSELoss()(q_values, targets)

        # Потери для stop_loss и take_profit
        stop_loss_predictions = self.q_network(states)[:, self.action_size]
        take_profit_predictions = self.q_network(states)[:, self.action_size + 1]

        true_stop_loss = torch.tensor(
            [state.get("stop_loss", 0.02) if isinstance(state, dict) else 0.02 for state in states],
            dtype=torch.float32
        ).to("cuda")

        true_take_profit = torch.tensor(
            [state.get("take_profit", 0.02) if isinstance(state, dict) else 0.02 for state in states],
            dtype=torch.float32
        ).to("cuda")

        loss_stop_loss = nn.MSELoss()(stop_loss_predictions, true_stop_loss)
        loss_take_profit = nn.MSELoss()(take_profit_predictions, true_take_profit)

        # Добавляем нормализацию потерь
        normalized_loss_q = loss_q / max(loss_q.item(), 1e-6)
        normalized_loss_stop_loss = loss_stop_loss / max(loss_stop_loss.item(), 1e-6)
        normalized_loss_take_profit = loss_take_profit / max(loss_take_profit.item(), 1e-6)

        # Общая потеря
        total_loss = (
            normalized_loss_q +
            0.5 * normalized_loss_stop_loss +
            0.5 * normalized_loss_take_profit
        )

        # Оптимизация
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def apply_rewards_and_penalties(self, reward, penalty, current_step, trade_log):
        """
        Применяет штрафы и бонусы к награде с учетом текущего шага и журнала сделок.
        """
        bonus = 0

        # Бонус за успешную сделку
        if trade_log and len(trade_log) > 0:
            last_trade = trade_log[-1]
            if last_trade.get("pnl") > 0:  # Если PnL положительный
                bonus += 1

        # Бонус за длительное удержание позиций
        if current_step > 10:
            recent_losses = [
                log['pnl'] for log in trade_log[-10:]
                if 'pnl' in log and log['pnl'] is not None and log['pnl'] < 0
            ]
            if len(recent_losses) < 5:  # Если меньше 50% убытков
                bonus += 1

        # Применение штрафов и бонусов
        reward -= penalty
        reward += bonus
        # logging.info(f"apply_rewards_and_penalties() - reward: {reward}, penalty: {penalty}, bonus: {bonus}")
        return reward

    def predict(self, state):
        """
        Предсказывает действия, stop_loss и take_profit.
        """
        state_tensor = torch.tensor(state["state_data"], dtype=torch.float32).unsqueeze(0).to("cuda")
        output = self.q_network(state_tensor).squeeze(0)

        action_values = output[:self.action_size]  # Значения Q-функции
        stop_loss = torch.sigmoid(output[self.action_size]).item()  # Нормализация для stop_loss
        take_profit = torch.sigmoid(output[self.action_size + 1]).item()  # Нормализация для take_profit

        # Применение штрафов и бонусов
        action_index = torch.argmax(action_values).item()
        # adjusted_q_values = self.apply_rewards_and_penalties(action_values[action_index].item(), penalty=0.1, bonus=0.2)

        # Добавляем вызов метода с необходимыми параметрами
        adjusted_q_values = self.apply_rewards_and_penalties(
            reward=action_values[action_index].item(),  # Используем выбранное действие
            penalty=0.1,  # Здесь можно динамически рассчитывать штраф
            current_step=state["current_step"],  # Добавьте текущий шаг в агент, если еще нет
            trade_log=state["trade_log"]  # Лог сделок
        )
        return adjusted_q_values, stop_loss, take_profit

    def initialize_binance_client(self):
        with open("acc_config.json", "r") as file:
            acc_config = json.load(file)

        # Укажите свои API ключи
        API_KEY = acc_config["API_KEY"]
        API_SECRET = acc_config["API_SECRET"]

        return Client(API_KEY, API_SECRET)

async def process_stream_data(agent, environment, redis_client):
    next_state = None
    reward = None
    done = None
    state = environment.reset(trade_balance=200)
    while not shutdown_flag.is_set():
        try:
            raw_data = await redis_client.lrange("final_order_book_stream", -1, -1)
            if raw_data:
                last_row = json.loads(raw_data[0])
                environment._set_X(last_row)
                action = agent.choose_action(state)
                # logging.info(f"before step - action: {action}")
                state, reward, done = environment.step(action)
                # logging.info(f"after step - action: {action}")
            await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error processing stream data: {e}")

async def main():
    # Путь к файлу сохраненной модели
    MODEL_PATH = "model/saved_trading_model_147_4_copy.pth"
    # Инициализация агента
    if os.path.exists(MODEL_PATH):
        logging.info("Загрузка сохраненной модели...")
        agent = DQLAgent(state_size=147, action_size=4)  # Создаем объект агента
        agent.q_network.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # Загружаем веса в основную сеть
        agent.update_target_network()  # Синхронизируем целевую сеть
        logging.info("Модель успешно загружена.")
    else:
        logging.info("Сохраненная модель не найдена. Создание новой модели...")
        agent = DQLAgent(state_size=147, action_size=4)  # Создаем нового агента

    
    agent = DQLAgent(state_size=147, action_size=4)
    environment = TradingEnvironment(agent)
    redis_client = await initialize_redis()
    try:
        await process_stream_data(agent, environment, redis_client)
    finally:
        await redis_client.aclose()  # Закрываем соединение с Redis

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        shutdown_flag.set()
        logging.info("Программа завершена пользователем (Ctrl + C).")