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

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from project_root.data.database_manager import execute_query

# Настройка логирования
logging.basicConfig(
    filename=f'../logs/hyperparameter_prediction_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def denormalize(value):
    """
    Денормализует значение на основе диапазона.
    """
    min_value = 0
    max_value = 115000
    return value * (max_value - min_value) + min_value

class TradingEnvironment:
    def __init__(self, X, agent):
        """
        Торговая среда для обучения модели.
        """
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("Input data X must be a 2D numpy array.")
        self.X = X
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
        self.initialize_balances(trade_balance=70)

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

    # Вызов проверки структуры данных X    
    def validate_X_structure(self):
        """
        Проверяет структуру self.X на корректность.
        """
        for i, row in enumerate(self.X):
            if not isinstance(row, (np.ndarray, list)):
                raise ValueError(f"Row {i} is not a list or numpy array: {row}")

            # Проверка вложенных массивов
            if any(isinstance(x, (np.ndarray, list)) for x in row):
                raise ValueError(f"Row {i} contains nested arrays: {row}")

            if not all(isinstance(x, (float, int)) for x in np.ravel(row)):
                raise ValueError(f"Row {i} contains non-numeric values: {row}")

        print("Validation passed! Structure of X is correct.")

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

    def analyze_trends(self):
        """
        Анализ трендов для 5-минутного и 1-минутного интервалов.
        """
        return {
            "5m_trend_diff": self.X[self.current_step][100],  # trend_price_difference для 5 минут
            "5m_trend_strength": self.X[self.current_step][102],  # trend_strength для 5 минут
            "1m_trend_diff": self.X[self.current_step][122],  # trend_price_difference для 1 минуты
            "1m_trend_strength": self.X[self.current_step][124],  # trend_strength для 1 минуты
            "atr": self.X[self.current_step][4],  # ATR
            "sma": self.X[self.current_step][5],  # SMA
            "mad": self.X[self.current_step][6],  # MAD
            "cci": self.X[self.current_step][7],  # CCI
        }

    def adjust_stop_loss_take_profit(self, position):
        """
        Улучшает stop_loss и take_profit позиции на основе предсказаний модели,
        с корректировкой на основе рыночных данных и относительных процентов.
        """
        # Анализ текущей цены
        current_price = denormalize(self.current_price)
        model_output = self.agent.predict(self._get_state())
        _, predicted_stop_loss, predicted_take_profit = model_output

        # Получение рыночных границ
        interval_low = denormalize(self.X[self.current_step][98])  # Минимальная цена из интервала
        interval_high = denormalize(self.X[self.current_step][99])  # Максимальная цена из интервала

        # Определяем допустимые диапазоны с отклонением в процентах
        tolerance = 0.03  # Допустимое отклонение (3%)
        adjusted_low = interval_low * (1 - tolerance)  # Нижняя граница с запасом
        adjusted_high = interval_high * (1 + tolerance)  # Верхняя граница с запасом

        # Предсказания модели
        proposed_stop_loss = current_price * (1 - predicted_stop_loss)
        proposed_take_profit = current_price * (1 + predicted_take_profit)

        # Логирование предсказаний
        # logging.info(
        #     f"Proposed stop_loss: {proposed_stop_loss}, Proposed take_profit: {proposed_take_profit}, "
        #     f"Adjusted_low: {adjusted_low}, Adjusted_high: {adjusted_high}"
        # )

        # Корректируем значения для long позиции
        if position["direction"] == "long":
            # Проверка stop_loss
            if proposed_stop_loss < adjusted_low:
                # logging.warning(f"Proposed stop_loss {proposed_stop_loss} is below adjusted_low {adjusted_low}. Adjusting.")
                proposed_stop_loss = adjusted_low
            # Проверка take_profit
            if proposed_take_profit > adjusted_high:
                # logging.warning(f"Proposed take_profit {proposed_take_profit} is above adjusted_high {adjusted_high}. Adjusting.")
                proposed_take_profit = adjusted_high

            # Применяем значения
            position["stop_loss"] = max(position["stop_loss"], proposed_stop_loss)
            position["take_profit"] = min(position["take_profit"], proposed_take_profit)  # Ограничиваем до adjusted_high

        # Корректируем значения для short позиции
        elif position["direction"] == "short":
            # Проверка stop_loss
            if proposed_stop_loss > adjusted_high:
                # logging.warning(f"Proposed stop_loss {proposed_stop_loss} is above adjusted_high {adjusted_high}. Adjusting.")
                proposed_stop_loss = adjusted_high
            # Проверка take_profit
            if proposed_take_profit < adjusted_low:
                # logging.warning(f"Proposed take_profit {proposed_take_profit} is below adjusted_low {adjusted_low}. Adjusting.")
                proposed_take_profit = adjusted_low

            # Применяем значения
            position["stop_loss"] = min(position["stop_loss"], proposed_stop_loss)
            position["take_profit"] = max(position["take_profit"], proposed_take_profit)  # Ограничиваем до adjusted_low

        # Логируем итоговые значения
        # logging.info(f"Final Adjusted stop_loss: {position['stop_loss']}, take_profit: {position['take_profit']}")

    def reset(self, trade_balance):
        """
        Сброс среды до начального состояния.
        """
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
        state = self._get_state()
        if not isinstance(state, dict) or "positions" not in state:
            raise ValueError(f"Invalid state after reset: {state}")
        # print(f"state state state: {state}")
        # print(f"shape shape shape: {state.shape}")
        return state

    def _get_state(self, force_last_state=False):
        """
        Возвращает текущее состояние среды. Если force_last_state=True,
        то возвращает последнее известное состояние, даже если current_step >= len(self.X).
        """
        # Рассчитываем текущий PnL по всем позициям
        current_pnl = sum(self._calculate_pnl(pos) for pos in self.positions)

        if self.current_step >= len(self.X):
            if force_last_state:
                # logging.warning("Current step exceeds data length. Returning last valid state.")
                last_valid_step = len(self.X) - 1  # Последний доступный шаг
                row_data = self.X[last_valid_step]
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
                        row_data
                    ]),
                    "positions": self.positions,
                    "current_step": self.current_step,  # Добавляем current_step
                    "trade_log": self.trade_log,  # Добавляем trade_log
                    "current_pnl": current_pnl,  # Добавляем текущий PnL
                    "liquidation_price": self.liquidation_price  # Liquidation price
                }
                return state
            else:
                logging.warning("Current step exceeds data length. Returning default state.")
                return {
                    "state_data": np.zeros(self.state_size, dtype=np.float32),
                    "positions": [],
                    "current_step": self.current_step,  # Добавляем current_step
                    "trade_log": self.trade_log,  # Добавляем trade_log
                    "current_pnl": 0.0,  # Добавляем текущий PnL
                    "liquidation_price": 0.0
                }

        row_data = self.X[self.current_step]
        state = {
            "state_data": np.concatenate([
                np.array([
                    float(self.spot_balance),
                    float(self.futures_balance),
                    float(self.leverage),
                    float(len(self.positions)),
                    float(self.current_step - self.positions[0]["entry_step"] if self.positions else 0),
                    float(current_pnl),  # Добавляем текущий PnL
                    float(self.liquidation_price)  # Liquidation price
                ], dtype=np.float32),
                row_data
            ]),
            "positions": self.positions,
            "current_step": self.current_step,  # Добавляем current_step
            "trade_log": self.trade_log,  # Добавляем trade_log
            "current_pnl": current_pnl,  # Добавляем текущий PnL
            "liquidation_price": self.liquidation_price
        }
        return state

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
                next_state = self._get_state(force_last_state=True)

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

    def step(self, action):
        reward = 0
        done = False
        
        # Проверяем, не превышает ли current_step размер данных
        if self.current_step >= len(self.X):
            # logging.warning("Current step exceeds data length. Ending the episode.")
            self._close_positions()
            self.train_on_experience()
            return self._get_state(), 0, True  # Завершаем эпизод

        # Обновление текущей цены
        self.current_price = self.X[self.current_step][0]  # Цена текущей свечи
        # Анализ трендов
        trends = self.analyze_trends()
        imbalance = self.X[self.current_step][3]  # bid_ask_imbalance
        
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
        
        # Логируем состояние только каждые 100 шагов
        if self.current_step % 1000 == 0:
            logging.info(f"Step: {self.current_step}, Total balance: {total_balance}, Spot balance: {self.spot_balance:.5f}, "
                        f"Futures balance: {self.futures_balance:.5f}, Positions: {len(self.positions)}")
        if self.positions:
            for i, position in enumerate(self.positions, start=1):
                self.adjust_stop_loss_take_profit(position)
                # logging.info(f"Position {i}: {position}")

        # Переход на следующий шаг
        self.current_step += 1
        
        return self._get_state(), reward, done

    def _open_position(self, direction):
        # Проверка наличия активных позиций
        if len(self.positions) >= 1:  # Ограничение на одну активную позицию
            # logging.warning("Position not opened: An active position already exists.")
            return
        
        mid_price = denormalize(self.current_price)  # mid_price из стакана
    
        # Расчет trade_balance
        # Используем половину фьючерсного баланса для торговли
        trade_balance = self.get_trade_balance()
        position_size = trade_balance
        total_position_size = (self.leverage * position_size) / mid_price
        
        if self.futures_balance <= position_size / mid_price:
            return
        
        # Предсказания от агента
        _, predicted_stop_loss, predicted_take_profit = self.agent.predict(self._get_state())
        predicted_stop_loss = float(predicted_stop_loss)
        predicted_take_profit = float(predicted_take_profit)

        stop_loss = mid_price * (1 - predicted_stop_loss)  # Преобразуем из нормализованных значений
        take_profit = mid_price * (1 + predicted_take_profit)
        
        self.liquidation_price = self._calculate_liquidation_price(total_position_size, direction)
        self.futures_balance -= position_size / mid_price

        position_id = len(self.positions) + 1

        # Добавляем позицию
        position = {
            "position_id": position_id,
            "entry_price": mid_price,
            "position_size": total_position_size,
            "direction": direction,
            "liquidation_price": self.liquidation_price,
            "entry_step": self.current_step,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        
        self.adjust_stop_loss_take_profit(position)
        self.positions.append(position)

        # logging.info(
        #     f"Opened {direction} - Entry price: {denormalize(self.current_price):.5f}, "
        #     f"Size: {total_position_size:.5f}, Liquidation price: {liquidation_price:.5f}, "
        # )

        # logging.info(
        #     # f"Confidence factor: {confidence_factor:.5f}, "
        #     f"Futures balance: {self.futures_balance:.5f}"
        # )

        # logging.info(
        #     f"Stop loss: {stop_loss:.5f}, Take profit: {take_profit:.5f}, "
        # )

    def _close_positions(self):
        successfully_closed = []
        errors = []
        total_pnl = 0
        total_reward = 0

        # Устанавливаем комиссию как долю от размера позиции
        commission_rate = 0.1  # 0.1% комиссии как на Binance

        for i, position in enumerate(self.positions):
            reward = 0  # Инициализация reward
            try:
                pnl = self._calculate_pnl(position)
                position_size = position.get("position_size", 0) / denormalize(self.current_price)
                commission = position_size * commission_rate
                pnl -= commission
                total_pnl += pnl

                self.futures_balance += total_pnl
                successfully_closed.append(i)

                # Обновляем PnL в позиции
                position["pnl"] = pnl
                # logging.info(f"_close_positions: {position["pnl"]} {pnl}")

                # Рассчитываем награду/штраф на основе результата сделки
                reward = self._calculate_reward(position)
                total_reward += reward
                # logging.info(f"_close_positions() reward: {reward}, pnl: {pnl}, total_pnl: {total_pnl}")

                # logging.info(
                #     f"Closed position - Current Price: {denormalize(self.current_price):.5f} - PnL: {pnl:.5f} (after commission: {commission:.5f}), "
                #     f"Updated Futures balance: {self.futures_balance:.5f}"
                # )

                # Ретроспективный анализ
                future_data_length = min(2100, len(self.X) - self.current_step)
                future_data = self.X[self.current_step:self.current_step + future_data_length, 0]  # Берём данные `mid_price` на 10 шагов вперёд
                future_data_denormalized = denormalize(future_data)
                # logging.info(f"future_data_denormalized: {future_data_denormalized}, type: {type(future_data_denormalized)}, dtype: {future_data_denormalized.dtype if isinstance(future_data_denormalized, np.ndarray) else 'Not an ndarray'}")

                if position["direction"] == "long":
                    cutoff_index = np.where(future_data_denormalized <= position["liquidation_price"])[0]
                elif position["direction"] == "short":
                    cutoff_index = np.where(future_data_denormalized >= position["liquidation_price"])[0]

                if len(cutoff_index) > 0:
                    future_data_denormalized = future_data_denormalized[:cutoff_index[0]]  # Обрезаем до первой цены, достигшей ликвидации
                # else:
                    # logging.warning("No cutoff point found. future_data_denormalized remains unchanged.")

                retrospective_experience = self.retrospective_analysis(position, future_data_denormalized)
                if retrospective_experience:
                    state, action, reward = retrospective_experience
                    # logging.info(f"_close_positions() 'if retrospective_experience' - reward: {reward}")
                    self.agent.store_experience(state, action, reward, self._get_state(), False)
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
                    "commission": commission,
                    "exit_spot_balance": self.spot_balance,
                    "exit_futures_balance": self.futures_balance,
                    "entry_data": self._get_state()
                })
                # logging.info(f"_close_positions Trade log entry: {self.trade_log}")


        # Учет дополнительной комиссии на общий PnL (опционально)
        # total_pnl -= abs(total_pnl) * commission_rate  # Если нужно еще одно снижение общего PnL
        # self.pnl = total_pnl

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
                    "ATR при входе": self.X[log.get("entry_step", 0)][4],  # ATR при входе
                    "SMA при входе": self.X[log.get("entry_step", 0)][5],  # SMA при входе
                    "MAD при входе": self.X[log.get("entry_step", 0)][6],  # MAD при входе
                    "CCI при входе": self.X[log.get("entry_step", 0)][7],  # CCI при входе
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

    def _calculate_liquidation_price(self, position_size, direction):
        """
        Рассчитывает уровень ликвидации для лонга и шорта.
        
        :param entry_price: float, цена входа
        :param position_size: float, размер позиции в долларах
        :param leverage: int, плечо
        :param futures_balance: float, текущий баланс фьючерсного кошелька
        :param maintenance_margin_rate: float, минимальная маржа (например, 0.005 для 0.5%)
        :param direction: str, направление сделки ("long" или "short")
        :return: float, уровень ликвидации
        """
        entry_price = denormalize(self.current_price)
        maintenance_margin_rate = 0.00255
        # Рассчитываем техническую маржу
        maintenance_margin = position_size * maintenance_margin_rate

        # Рассчитываем уровень ликвидации
        if direction == "long":
            self.liquidation_price = entry_price - (self.futures_balance - maintenance_margin) / position_size
        elif direction == "short":
            self.liquidation_price = entry_price + (self.futures_balance - maintenance_margin) / position_size
        else:
            raise ValueError("Direction must be 'long' or 'short'")
        
        # logging.info(f"_calculate_liquidation_price() - liquidation_price: {self.liquidation_price}, position_size: {position_size}")
        return round(self.liquidation_price, 5)  # Округляем до 5 знаков

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

def initialize_binance_client():
    with open("acc_config.json", "r") as file:
        acc_config = json.load(file)

    # Укажите свои API ключи
    API_KEY = acc_config["API_KEY"]
    API_SECRET = acc_config["API_SECRET"]

    return Client(API_KEY, API_SECRET)

# Модель GRU для данных стакана
class OrderBookGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(OrderBookGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)
        return self.fc(out[:, -1, :])

# Модель LSTM для интервалов
class IntervalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(IntervalLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        return self.fc(out[:, -1, :])

def fetch_orderbook_data() -> pd.DataFrame:
    logging.info(f"Fetching data for orderbook")
    # query = f"SELECT * FROM order_book_data order by id LIMIT 350000"
    # query = f"SELECT * FROM order_book_data order by id LIMIT 200000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 75000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 70000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 37500"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 27500"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 1000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id"

    query = f"SELECT * FROM order_book_data WHERE id >='1369323' and id <= '1812171' order by id"
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for orderbook")
    # else:
        # logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    return data

# Функция для загрузки обученной модели интервала
def load_interval_model(interval: str, params: dict, input_size) -> IntervalLSTMModel:
    model_path = f"../models/saved_models/interval_lstm_model_{interval}.pth"
    model = IntervalLSTMModel(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        output_size=4,
        dropout=params["dropout"]
    ).to("cuda")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

# Функция для загрузки обученной модели интервала
def load_orderbook_model(params: dict, input_size) -> OrderBookGRUModel:
    model_path = f"../models/saved_models/interval_lstm_model_orderbook.pth"
    model = OrderBookGRUModel(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        output_size=7,
        dropout=params["dropout"]
    ).to("cuda")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Функция для получения последних данных интервала
def fetch_interval_data(interval: str, sequence_length: int, last_open_time: datetime) -> pd.DataFrame:
    query = f"""
        SELECT * FROM binance_klines_normalized
        WHERE data_interval = '{interval}'
        AND open_time <= '{last_open_time}'
        ORDER BY open_time DESC
        LIMIT {sequence_length}
    """
    data = execute_query(query)
    if not data.empty:
        data = data.iloc[::-1]  # Реверсируем, чтобы данные были в хронологическом порядке
    return data

def fetch_small_interval_data(interval: str, last_timestamp: datetime, required_rows) -> pd.DataFrame:
    logging.info(f"Fetching small interval data for {interval} last_timestamp: {last_timestamp}")
    query = f"""SELECT open_time, open_price, high_price, low_price, close_price, close_time, volume
              FROM binance_klines WHERE `data_interval` = '{interval}'
              AND open_time <= '{last_timestamp}'
              order by open_time desc LIMIT {required_rows}"""
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for small interval {interval}")
    
    if not data.empty:
        data = data.iloc[::-1]  # Обратная сортировка по возрастанию времени

    return data

def calculate_trend_price_difference(data: pd.DataFrame) -> pd.Series:
    """Вычисляет тренд на основе разницы цен (close_price - open_price)."""
    return data["close_price_normalized"] - data["open_price_normalized"]

def calculate_trend_sma(data: pd.DataFrame, window: int = 5) -> pd.Series:
    """Вычисляет тренд на основе SMA."""
    return data["close_price_normalized"].rolling(window=window).mean()

def calculate_trend_strength(data: pd.DataFrame) -> pd.Series:
    """Вычисляет силу тренда как производную скользящего среднего."""
    sma = calculate_trend_sma(data, window=5)
    return sma.diff()  # Разница между текущим и предыдущим SMA

def prepare_interval_data(interval: str, sequence_length, first_open_time: datetime, last_open_time: datetime) -> pd.DataFrame:
    logging.info(f"Fetching data for interval: {interval}, first_open_time: {first_open_time} and last_open_time: {last_open_time}")

    first_open_time = adjust_interval_timestamp(interval, first_open_time)
    last_open_time = adjust_interval_timestamp(interval, last_open_time)
    trend_window = 5
    
    interval_sequence = calculate_rows_for_interval(interval, sequence_length, first_open_time, last_open_time)
    interval_sequence = interval_sequence + trend_window

    data = fetch_interval_data(interval, interval_sequence, last_open_time)
    window_data = get_active_window(interval)
    small_interval = get_small_interval(interval)

    required_rows = calculate_required_rows_for_small_interval(interval_sequence, interval)

    if small_interval is not None:
        small_data = fetch_small_interval_data(small_interval, last_open_time, required_rows)
        if small_data.empty:
            logging.warning(f"No small interval data available for interval {small_interval}. Proceeding without it.")
            small_data = None
    else:
        small_data = None

    if data.empty:
        logging.warning("No data available, skipping trial.")
        return float("inf")
    
    if small_data is not None:
        aggregated_small_data = aggregate_small_data(small_data, small_interval)
        # logging.info(f"Aggregated data: \n{aggregated_small_data}")
        normalized_small_data = normalize_small_data(aggregated_small_data, window_data)
        # logging.info(f"Normalized data: \n{normalized_small_data}")
        final_data = merge_large_and_small_data(data, normalized_small_data)
        final_data = final_data.drop(final_data.index[-1])
    else:
        final_data = data.copy()
    
    # Добавляем тренды
    final_data["trend_price_difference"] = calculate_trend_price_difference(final_data)
    final_data["trend_sma"] = calculate_trend_sma(final_data, trend_window)
    final_data["trend_strength"] = calculate_trend_strength(final_data)
    
    final_data = final_data.drop(final_data.index[:5])
    
    # print(final_data)

    return final_data

def get_intervals_predictions(first_open_time, last_open_time) -> pd.DataFrame:
    logging.info("Fetching synchronized predictions for intervals")
    
    # Загружаем параметры из JSON
    with open("../models/optimized_params.json", "r") as file:
        optimized_params = json.load(file)
    
    intervals = optimized_params.keys()  # Например: ["1d", "4h", "1h", "15m", "5m", "1m"]
    all_predictions = []

    for interval in intervals:
        # Загружаем модель
        params = optimized_params[interval]
        sequence_length = params["sequence_length"]

        interval_data = prepare_interval_data(interval, sequence_length, first_open_time, last_open_time)

        # filename = f"combined_data_{interval}.csv"
        # if os.path.exists(filename):
        #     add_null_row_to_csv(filename)
        # # Сохраняем DataFrame в CSV-файл
        # interval_data.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename))

        if interval == "1m":
            columns_to_drop = ["id", "open_time", "close_time", "data_interval", "window_id",
                           "next_open_time", "next_close_time", "small_open_time", "small_close_time",
                           "small_low_price", "small_high_price", "small_open_price", "small_close_price",
                           "small_volume",
                           "trend_price_difference", "trend_sma", "trend_strength"]
            timestamps = interval_data["open_time"]  # Используем `open_time` для интервала "1m"
        else:
            columns_to_drop = ["id", "open_time", "close_time", "data_interval", "window_id",
                           "next_open_time", "next_close_time", "small_open_time", "small_close_time",
                           "small_low_price", "small_high_price", "small_open_price", "small_close_price",
                           "small_volume", "open_price_normalized", "close_price_normalized",
                           "low_price_normalized", "high_price_normalized",
                           "trend_price_difference", "trend_sma", "trend_strength"]
            timestamps = interval_data["next_open_time"]  # Используем `next_open_time` для остальных интервалов
            timestamps = timestamps.sort_values().reset_index(drop=True)
            
        # Преобразуем данные для подачи в модель
        columns_to_drop = [col for col in columns_to_drop if col in interval_data.columns]
        features = interval_data.drop(columns=columns_to_drop).values.astype(np.float32)
        
        # Определяем input_size на основе features
        input_size = features.shape[1]

        # Загружаем модель
        model = load_interval_model(interval, params, input_size)

        total_rows = features.shape[0]
        # Формируем окна данных
        predictions = []
        
        for start_idx in range(0, total_rows - sequence_length + 1):
            # Формируем батч данных
            batch_features = features[start_idx:start_idx + sequence_length]
            input_data = torch.tensor(batch_features, dtype=torch.float32).unsqueeze(0).to("cuda")
            
            # Получаем прогноз
            with torch.no_grad():
                prediction = model(input_data).cpu().numpy().flatten()

            timestamp = timestamps.iloc[start_idx + sequence_length - 1]

            predictions.append({
                "timestamp": timestamp,  # Текущая временная метка
                "interval": interval,  # Интервал
                "prediction": prediction,
                # Добавляем тренды
                "trend_price_difference": interval_data["trend_price_difference"].iloc[start_idx + sequence_length - 1],
                "trend_sma": interval_data["trend_sma"].iloc[start_idx + sequence_length - 1],
                "trend_strength": interval_data["trend_strength"].iloc[start_idx + sequence_length - 1],
                # Добавляем индикаторы
                "open_price_normalized": interval_data["open_price_normalized"].iloc[start_idx + sequence_length - 1],
                "high_price_normalized": interval_data["high_price_normalized"].iloc[start_idx + sequence_length - 1],
                "low_price_normalized": interval_data["low_price_normalized"].iloc[start_idx + sequence_length - 1],
                "close_price_normalized": interval_data["close_price_normalized"].iloc[start_idx + sequence_length - 1],
                "volume_normalized": interval_data["volume_normalized"].iloc[start_idx + sequence_length - 1],
                "rsi_normalized": interval_data["rsi_normalized"].iloc[start_idx + sequence_length - 1],
                "macd_normalized": interval_data["macd_normalized"].iloc[start_idx + sequence_length - 1],
                "macd_signal_normalized": interval_data["macd_signal_normalized"].iloc[start_idx + sequence_length - 1],
                "macd_hist_normalized": interval_data["macd_hist_normalized"].iloc[start_idx + sequence_length - 1],
                "sma_20_normalized": interval_data["sma_20_normalized"].iloc[start_idx + sequence_length - 1],
                "ema_20_normalized": interval_data["ema_20_normalized"].iloc[start_idx + sequence_length - 1],
                "upper_bb_normalized": interval_data["upper_bb_normalized"].iloc[start_idx + sequence_length - 1],
                "middle_bb_normalized": interval_data["middle_bb_normalized"].iloc[start_idx + sequence_length - 1],
                "lower_bb_normalized": interval_data["lower_bb_normalized"].iloc[start_idx + sequence_length - 1],
                "obv_normalized": interval_data["obv_normalized"].iloc[start_idx + sequence_length - 1]
            })
        
        # Добавляем прогнозы для текущего интервала в общий список
        if predictions:
            all_predictions.append(pd.DataFrame(predictions))

    # Преобразуем результаты в DataFrame
    # Объединяем все прогнозы по интервалам в одну таблицу
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    else:
        return pd.DataFrame()

def aggregate_small_data(small_data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Агрегация данных для меньшего интервала, учитывая частоту, соответствующую заданному интервалу.
    """
    logging.info(f"Aggregating small interval data for interval: {interval}")

    # Определяем частоту группировки на основе входного интервала
    # Например, для "4h" агрегация будет по "1D", для "1h" — по "4h", и так далее
    freq_map = {
        "1d": "1D",
        "4h": "1D",
        "1h": "4h",
        "15m": "1h",
        "5m": "15min",
        "1m": "5min"
    }

    if interval not in freq_map:
        raise ValueError(f"Unsupported interval: {interval}")

    aggregation_freq = freq_map[interval]

    # Агрегация данных по частоте
    aggregated = small_data.resample(aggregation_freq, on='open_time').agg({
        'low_price': 'min',  # Минимальная цена
        'high_price': 'max',  # Максимальная цена
        'open_price': 'first',  # Первая цена
        'close_price': 'last',  # Последняя цена
        'volume': 'sum',  # Сумма объемов
        'open_time': 'first',  # Первая временная метка
        'close_time': 'last',  # Последняя временная метка
    })

    # Shift small interval data for прогнозирования следующего интервала
    aggregated['next_open_time'] = aggregated['open_time'].shift(-1)
    aggregated['next_close_time'] = aggregated['close_time'].shift(-1)
    aggregated['next_low_price'] = aggregated['low_price'].shift(-1)
    aggregated['next_high_price'] = aggregated['high_price'].shift(-1)
    aggregated['next_open_price'] = aggregated['open_price'].shift(-1)
    aggregated['next_close_price'] = aggregated['close_price'].shift(-1)
    aggregated['next_volume'] = aggregated['volume'].shift(-1)

    # Сбрасываем индекс и переименовываем его
    aggregated.reset_index(drop=True, inplace=True)

    logging.info(f"Aggregated small data: {aggregated.shape[0]} rows for interval {interval}.")
    
    return aggregated

def normalize_small_data(small_data: pd.DataFrame, window_data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Normalizing small interval data using active window.")
    
    # Извлекаем минимальные и максимальные значения из окна
    min_price = window_data['min_open_price'].values[0]
    max_price = window_data['max_open_price'].values[0]
    min_volume = window_data['min_volume'].values[0]
    max_volume = window_data['max_volume'].values[0]
    
    # Нормализуем данные
    small_data['low_price'] = normalize(small_data['low_price'], min_price, max_price)
    small_data['high_price'] = normalize(small_data['high_price'], min_price, max_price)
    small_data['open_price'] = normalize(small_data['open_price'], min_price, max_price)
    small_data['close_price'] = normalize(small_data['close_price'], min_price, max_price)
    small_data['volume'] = normalize(small_data['volume'], min_volume, max_volume)

    small_data['next_low_price'] = normalize(small_data['next_low_price'], min_price, max_price)
    small_data['next_high_price'] = normalize(small_data['next_high_price'], min_price, max_price)
    small_data['next_open_price'] = normalize(small_data['next_open_price'], min_price, max_price)
    small_data['next_close_price'] = normalize(small_data['next_close_price'], min_price, max_price)
    small_data['next_volume'] = normalize(small_data['next_volume'], min_volume, max_volume)
    
    logging.info("Normalization completed.")
    return small_data

def merge_large_and_small_data(data: pd.DataFrame, small_data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Merging large interval data with small interval features.")
    
    # Переименовываем столбцы в small_data для уникальности
    small_data.rename(columns={
        'open_time': 'small_open_time',
        'open_price': 'small_open_price',
        'high_price': 'small_high_price',
        'low_price': 'small_low_price',
        'close_price': 'small_close_price',
        'close_time': 'small_close_time',
        'volume': 'small_volume'
    }, inplace=True)

    # Объединяем по времени
    merged_data = pd.merge(
        data, 
        small_data, 
        left_on='open_time', 
        right_on='small_open_time', 
        how='left'
    )
    
    logging.info(f"Merged data shape: {merged_data.shape}")
    return merged_data

def get_small_interval(interval):
    interval_map = {
        "1d": "4h",
        "4h": "1h",
        "1h": "15m",
        "15m": "5m",
        "5m": "1m",
        "1m": None
    }

    return interval_map[interval]

def calculate_required_rows_for_small_interval(sequence_length: int, large_interval: str) -> int:
    """
    Рассчитывает количество строк для малого интервала на основе sequence_length и старшего интервала.
    """
    # Множители для пересчета
    MULTIPLIER = {
        ("1d", "4h"): 6,
        ("4h", "1h"): 4,
        ("1h", "15m"): 4,
        ("15m", "5m"): 3,
        ("5m", "1m"): 5
    }

    small_interval = get_small_interval(large_interval)  # Получаем младший интервал
    if not small_interval:
        return 0  # Если младший интервал отсутствует, данные не нужны
    
    multiplier = MULTIPLIER.get((large_interval, small_interval), 1)
    required_rows = sequence_length * multiplier + multiplier  # Количество строк для малого интервала
    return required_rows

def calculate_rows_for_interval(interval: str, sequence_length, first_open_time: datetime, last_open_time: datetime):
    """
    Подсчитывает количество строк для каждого интервала в заданном временном промежутке.
    :param first_open_time: Начальная временная метка.
    :param last_open_time: Конечная временная метка.
    :return: Словарь с количеством строк для каждого интервала.
    """
    # Длительность интервалов в секундах
    intervals = {
        "1d": 24 * 60 * 60,
        "4h": 4 * 60 * 60,
        "1h": 60 * 60,
        "15m": 15 * 60,
        "5m": 5 * 60,
        "1m": 60,
    }

    # Вычисляем разницу во времени в секундах
    total_seconds = (last_open_time - first_open_time).total_seconds()
    interval_sequence = int(total_seconds // intervals[interval]) + sequence_length
    if interval != '1m':
        interval_sequence = interval_sequence + 1

    return interval_sequence

def get_active_window(interval: str) -> pd.DataFrame:
    logging.info(f"Fetching window data for {interval}")
    query = f"""
        SELECT * FROM binance_normalization_windows 
        WHERE data_interval = '{interval}' AND is_active = 1
        ORDER BY end_time DESC LIMIT 1
    """
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for window interval {interval}")

    return data

def adjust_interval_timestamp(interval: str, timestamp: datetime) -> datetime:
    """
    Корректирует временную метку в зависимости от интервала.
    :param interval: Интервал данных (например, '1d', '1h').
    :param timestamp: Исходная временная метка.
    :return: Временная метка, скорректированная до последнего завершенного интервала.
    """
    if interval == '1d':
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == '4h':
        return timestamp.replace(minute=0, second=0, microsecond=0, hour=(timestamp.hour // 4) * 4)
    elif interval == '1h':
        return timestamp.replace(minute=0, second=0, microsecond=0)
    elif interval == '15m':
        return timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
    elif interval == '5m':
        return timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
    elif interval == '1m':
        return timestamp.replace(second=0, microsecond=0)
    return timestamp

def calculate_indicators(orderbook_data):
    orderbook_data["atr"] = orderbook_data["mid_price"].diff().abs().rolling(window=14).mean()
    orderbook_data["sma"] = orderbook_data["mid_price"].rolling(window=14).mean()
    orderbook_data["mad"] = orderbook_data["mid_price"].rolling(window=14).apply(lambda x: np.mean(np.abs(x - x.mean())))
    orderbook_data["cci"] = (orderbook_data["mid_price"] - orderbook_data["sma"]) / (0.015 * orderbook_data["mad"])
    return orderbook_data

def prepare_data_for_orderbook_non_overlapping(orderbook_data: pd.DataFrame) -> np.ndarray:
    """
    Подготавливает данные для обучения модели, исключая перекрывающиеся строки.
    :param orderbook_data: Данные стакана.
    :return: Массив X для обучения.
    """
    X = []
    # Устанавливаем опции печати для numpy
    # np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.14f}"})
    orderbook_data = calculate_indicators(orderbook_data).dropna().reset_index(drop=True)
    # Извлекаем временные рамки
    first_open_time = orderbook_data.iloc[0]["timestamp"]
    last_open_time = orderbook_data.iloc[-1]["timestamp"]
    # Получаем прогнозы интервалов
    intervals_predictions = get_intervals_predictions(first_open_time, last_open_time)

    # Обрабатываем данные стакана без перекрытия
    for i in range(len(orderbook_data)):
        # Логирование каждые 50000 строк
        if i % 50000 == 0:
            logging.info(f"Processed {i} rows out of {len(orderbook_data)}")
        # Извлекаем текущую строку стакана
        orderbook_row = orderbook_data.iloc[i].drop(["id", "timestamp"]).values.astype(np.float32)
        # Текущая временная метка
        current_timestamp = orderbook_data.iloc[i]["timestamp"]
        # Фильтруем прогнозы по интервалам, даты которых <= текущей временной метке
        filtered_predictions = intervals_predictions[intervals_predictions["timestamp"] <= current_timestamp]
        # Берем последние прогнозы для каждого интервала
        latest_predictions = filtered_predictions.groupby("interval").tail(1)

        # Объединяем данные `prediction` и индикаторы
        combined_features = []
        for _, row in latest_predictions.iterrows():
            # print(f"{row['prediction']}")
            # print(f"{row['trend_price_difference']} {row['trend_sma']} {row['trend_strength']}")
            combined_row = np.concatenate([row["prediction"],
                [row["trend_price_difference"], row["trend_sma"], row["trend_strength"],
                row["open_price_normalized"], row["high_price_normalized"], row["low_price_normalized"], row["close_price_normalized"],
                row["volume_normalized"], row["rsi_normalized"], row["macd_normalized"], 
                row["macd_signal_normalized"], row["macd_hist_normalized"], 
                row["sma_20_normalized"], row["ema_20_normalized"], row["upper_bb_normalized"],
                row["middle_bb_normalized"], row["lower_bb_normalized"], row["obv_normalized"]]])
            combined_features.append(combined_row)

        # Преобразуем список в двумерный массив и объединяем с текущей строкой стакана
        combined_features = np.array(combined_features).flatten()
        combined_data = np.hstack((orderbook_row, combined_features))
        # print("\n")
        # print(combined_data)
        # print("\n")

        # Добавляем результат в X
        X.append(combined_data)

    logging.info(f"Data preparation complete. Total rows processed: {len(orderbook_data)}")
    return np.array(X)

# Убедимся, что границы можно легко увидеть
def add_null_row_to_csv(filename):
    null_row = pd.DataFrame([["NULL"] * 14])  # 14 соответствует количеству колонок в DataFrame
    null_row.to_csv(filename, mode="a", index=False, header=False)

global cached_data

cached_data = None
def fetch_cached_data():
    global cached_data
    if cached_data is None:
        cached_data = fetch_orderbook_data()
    return cached_data

def objective(trial):
    # logging.info("Starting a new trial")
    
    # hidden_size = trial.suggest_int("hidden_size", 64, 256)
    # num_layers = trial.suggest_int("num_layers", 1, 3)
    # dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # sequence_length = trial.suggest_int("sequence_length", 30, 100)
    # batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    # logging.info(f"Trial parameters - hidden_size: {hidden_size}, num_layers: {num_layers}, dropout: {dropout}, learning_rate: {learning_rate}, sequence_length: {sequence_length}, batch_size: {batch_size}")
    
    # orderbook_data = fetch_orderbook_data()
    orderbook_data = fetch_cached_data()
    # logging.info(f"Orderbook data fetched: {len(orderbook_data)} rows")

    # # # TODO fetching data from db is too busy work
    # # # TODO it should be done one time for all trials to economy time of optimization process
    
    # # X = prepare_data_for_orderbook(orderbook_data, sequence_length)
    # TODO Cup data
    # 		"mid_price" "sum_bid_volume" "sum_ask_volume" "bid_ask_imbalance" "atr" "sma" "mad" "cci"
    # TODO Interval 1d data
    # 		"open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
    # 		"trend_price_difference" "trend_sma" "trend_strength"
    # 		"open_price_normalized" "high_price_normalized" "low_price_normalized" "close_price_normalized"
    # 		"volume_normalized" "rsi_normalized"
    # 		"macd_normalized" "macd_signal_normalized" "macd_hist_normalized" "sma_20_normalized" "ema_20_normalized"
    # 		"upper_bb_normalized" "middle_bb_normalized" "lower_bb_normalized" "obv_normalized"
    # TODO Interval 4h data
    # 		"open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
    # 		"trend_price_difference" "trend_sma" "trend_strength"
    # 		"open_price_normalized" "high_price_normalized" "low_price_normalized" "close_price_normalized"
    # 		"volume_normalized" "rsi_normalized"
    # 		"macd_normalized" "macd_signal_normalized" "macd_hist_normalized" "sma_20_normalized" "ema_20_normalized"
    # 		"upper_bb_normalized" "middle_bb_normalized" "lower_bb_normalized" "obv_normalized"
    # TODO Interval 1h data
    # 		"open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
    # 		"trend_price_difference" "trend_sma" "trend_strength"
    # 		"open_price_normalized" "high_price_normalized" "low_price_normalized" "close_price_normalized"
    # 		"volume_normalized" "rsi_normalized"
    # 		"macd_normalized" "macd_signal_normalized" "macd_hist_normalized" "sma_20_normalized" "ema_20_normalized"
    # 		"upper_bb_normalized" "middle_bb_normalized" "lower_bb_normalized" "obv_normalized"
    # TODO Interval 15m data
    # 		"open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
    # 		"trend_price_difference" "trend_sma" "trend_strength"
    # 		"open_price_normalized" "high_price_normalized" "low_price_normalized" "close_price_normalized"
    # 		"volume_normalized" "rsi_normalized"
    # 		"macd_normalized" "macd_signal_normalized" "macd_hist_normalized" "sma_20_normalized" "ema_20_normalized"
    # 		"upper_bb_normalized" "middle_bb_normalized" "lower_bb_normalized" "obv_normalized"
    # TODO Interval 5m data
    # 		"open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
    # 		"trend_price_difference" "trend_sma" "trend_strength"
    # 		"open_price_normalized" "high_price_normalized" "low_price_normalized" "close_price_normalized"
    # 		"volume_normalized" "rsi_normalized"
    # 		"macd_normalized" "macd_signal_normalized" "macd_hist_normalized" "sma_20_normalized" "ema_20_normalized"
    # 		"upper_bb_normalized" "middle_bb_normalized" "lower_bb_normalized" "obv_normalized"
    # TODO Interval 1m data
    # 		"open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
    # 		"trend_price_difference" "trend_sma" "trend_strength"
    # 		"open_price_normalized" "high_price_normalized" "low_price_normalized" "close_price_normalized"
    # 		"volume_normalized" "rsi_normalized"
    # 		"macd_normalized" "macd_signal_normalized" "macd_hist_normalized" "sma_20_normalized" "ema_20_normalized"
    # 		"upper_bb_normalized" "middle_bb_normalized" "lower_bb_normalized" "obv_normalized"

    # TODO  "atr" "sma" "mad" "cci"
    # TODO  4   5   6   7
    # TODO  "open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
    # TODO  "trend_price_difference" "trend_sma" "trend_strength"
    # TODO  Interval 5m data 96	97	98	99	100	101	102
    # TODO  Interval 1m data 118    119	120	121	122	123	124
    X = prepare_data_for_orderbook_non_overlapping(orderbook_data)
    # print("\n")
    # print(f"{X[0][96]} {X[0][97]} {X[0][98]} {X[0][99]}")
    # print(f"{X[0][100]} {X[0][101]} {X[0][102]}")
    # print("\n")
    # print(f"{X[0][118]} {X[0][119]} {X[0][120]} {X[0][121]}")
    # print(f"{X[0][122]} {X[0][123]} {X[0][124]}")
    # logging.info(f"Prepared data shape: {X.shape}")
    # print(f"Shape of X: {np.array(X).shape}")
    # print(f"Sample from X[0]: {X[0]}")
    # print(f"Sample from X[0]: {X[1]}")
    # print(f"Sample from X[0]: {X[2]}")
    # print(f"Sample from X[0]: {X[3]}")
    # print(f"Sample from X[0]: {X[3].shape}")

    # Путь к файлу сохраненной модели
    MODEL_PATH = "model/saved_trading_model_147_4.pth"
    # Инициализация агента
    if os.path.exists(MODEL_PATH):
        print("Загрузка сохраненной модели...")
        agent = DQLAgent(state_size=147, action_size=4)  # Создаем объект агента
        agent.q_network.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # Загружаем веса в основную сеть
        agent.update_target_network()  # Синхронизируем целевую сеть
        print("Модель успешно загружена.")
    else:
        print("Сохраненная модель не найдена. Создание новой модели...")
        agent = DQLAgent(state_size=147, action_size=4)  # Создаем нового агента
    
    # Инициализация среды
    env = TradingEnvironment(X, agent)

    episodes = 1000
    target_update_frequency = 2  # Как часто обновлять целевую сеть

    for episode in range(episodes):
        state = env.reset(trade_balance=70)
        total_reward = 0

        while True:
            if not isinstance(state, dict) or "positions" not in state:
                logging.error(f"Invalid state: {state}")
                raise ValueError("State must be a dictionary containing 'positions'.")

            action = agent.choose_action(state)

            # Совершаем шаг в среде
            next_state, reward, done = env.step(action)
            
            # Применяем систему штрафов и наград (если требуется)
            penalty = abs(reward) * 0.01 if reward < 0 else 0  # Штраф за отрицательную награду
            # Вызов с передачей позиций
            adjusted_reward = agent.apply_rewards_and_penalties(reward, penalty, env.current_step, env.trade_log)
            
            # Сохраняем опыт в памяти агента
            agent.store_experience(state, action, adjusted_reward, next_state, done)

            # Переходим к следующему состоянию
            state = next_state
            total_reward += adjusted_reward
            # logging.info(f"objective() - reward: {reward}, penalty: {penalty}, total_reward: {total_reward}, adjusted_reward: {adjusted_reward}")

            # Обучаем агента после каждого шага
            agent.learn()

            if done:
                env.redistribute_balance_with_growth()  # Перераспределение баланса после эпизода
                num_steps = env.current_step  # Текущее количество шагов в эпизоде
                average_reward = total_reward / num_steps if num_steps > 0 else 0  # Средняя награда за шаг
                print(f"Episode {episode + 1}: Total Reward = {total_reward}, Average Reward = {average_reward}")
                logging.info(f"Episode {episode + 1}: Total Reward = {total_reward}, Average Reward = {average_reward}")
                # Настройка форматирования чисел
                trade_report = env.generate_trade_report()
                with pd.option_context('display.float_format', '{:.6f}'.format):  # Устанавливаем фиксированное количество знаков после запятой
                    logging.info(f"Trade report: \n{trade_report}")
                # print(trade_report)
                break

        # Обновляем целевую сеть каждые `target_update_frequency` эпизодов
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()
            print(f"Target network updated at episode {episode + 1}")
            logging.info(f"Target network updated at episode {episode + 1}")
            env.close_all_positions()
            # Сохранение модели
            torch.save(agent.q_network.state_dict(), MODEL_PATH)
            print(f"Модель сохранена: {MODEL_PATH}")
            logging.info(f"Model saved at {MODEL_PATH}")

    return

def main():
    logging.info("Starting hyperparameter optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)  # 50 испытаний для оптимизации
    print("Лучшие гиперпараметры:", study.best_params)
    print("Лучшее значение потерь:", study.best_value)
    logging.info(f"Optimization completed - Best Params: {study.best_params}, Best Loss: {study.best_value:.14f}")

if __name__ == "__main__":
    main()