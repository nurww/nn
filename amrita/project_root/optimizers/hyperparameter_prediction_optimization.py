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
    def __init__(self, X, initial_spot_balance=50, initial_futures_balance=50):
        """
        Торговая среда для обучения модели.
        """
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("Input data X must be a 2D numpy array.")
        self.X = X
        self.current_step = 0  # Текущая позиция в данных
        self.state_size = 141
        self.spot_balance = initial_spot_balance
        self.futures_balance = initial_futures_balance
        self.prev_total_balance = initial_spot_balance + initial_futures_balance  # Инициализация
        self.leverage = 50
        self.positions = []
        self.trade_log = []
        self.prev_pnl = 0
        self.pnl = 0      # Текущий общий PnL
        self.current_price = 0  # Текущая рыночная цена

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

    def analyze_trends(self):
        """
        Анализ трендов для 5-минутного и 1-минутного интервалов.
        """
        return {
            "5m_trend_diff": self.X[self.current_step][96],  # trend_price_difference для 5 минут
            "5m_trend_strength": self.X[self.current_step][98],  # trend_strength для 5 минут
            "1m_trend_diff": self.X[self.current_step][118],  # trend_price_difference для 1 минуты
            "1m_trend_strength": self.X[self.current_step][120]  # trend_strength для 1 минуты
        }

    def adjust_stop_loss_take_profit(self, position):
        """
        Динамическая коррекция уровней stop-loss и take-profit.
        """
        trends = self.analyze_trends()
        position["stop_loss"] = min(self.X[self.current_step][94], self.X[self.current_step][116]) * 0.98  # Уменьшаем stop_loss
        position["take_profit"] = max(self.X[self.current_step][95], self.X[self.current_step][117]) * 1.02  # Увеличиваем take_profit

        # Дополнительная коррекция на основе краткосрочных трендов
        if trends["1m_trend_strength"] < 0:
            position["stop_loss"] *= 0.97  # Сильное уменьшение при негативном тренде
        if trends["1m_trend_strength"] > 0:
            position["take_profit"] *= 1.03  # Сильное увеличение при положительном тренде

    def reset(self):
        """
        Сброс среды до начального состояния.
        """
        self.current_step = 0
        self.spot_balance = 50
        self.futures_balance = 50
        self.prev_total_balance = 50 + 50  # Инициализация
        self.leverage = 50
        self.positions = []
        self.trade_log = []
        self.prev_pnl = 0
        self.pnl = 0      # Текущий общий PnL
        self.current_price = 0  # Текущая рыночная цена
        logging.info("Environment reset.")
        state = self._get_state()
        if not isinstance(state, dict) or "positions" not in state:
            raise ValueError(f"Invalid state after reset: {state}")
        # print(f"state state state: {state}")
        # print(f"shape shape shape: {state.shape}")
        return state

    def _get_state(self):
        if self.current_step >= len(self.X):
            logging.warning("Current step exceeds data length. Returning default state.")
            return {"state_data": np.zeros(self.state_size, dtype=np.float32), "positions": []}
        
        row_data = self.X[self.current_step]
        state = {
            "state_data": np.concatenate([
                np.array([
                    float(self.spot_balance),
                    float(self.futures_balance),
                    float(self.leverage),
                    float(len(self.positions)),
                    float(self.current_step - self.positions[0]["entry_step"] if self.positions else 0)  # Длительность удержания
                ], dtype=np.float32),
                row_data
            ]),
            "positions": self.positions
        }
        # print(f"Generated state: {state}")
        return state

    def step(self, action):
        reward = 0
        done = False

        # Проверка завершения эпизода
        if self.current_step >= len(self.X) or self.futures_balance <= 0:
            done = True
            print("Episode ended. No more steps or balance depleted.")
            return self._get_state(), reward, done
        
        # Обновление текущей цены
        self.current_price = self.X[self.current_step][0]  # Цена текущей свечи
        trends = self.analyze_trends()
        # logging.info(f"Current price: {self.current_price}")

        if action[0] == 0:  # Open Long
            if trends["5m_trend_strength"] > 0 and trends["1m_trend_strength"] > 0:
                self._open_position("long")
            else:
                reward -= 0.1  # Штраф за неправильное решение
        elif action[0] == 1:  # Open Short
            if trends["5m_trend_strength"] < 0 and trends["1m_trend_strength"] < 0:
                self._open_position("short")
            else:
                reward -= 0.1
        elif action[0] == 2:  # Close Positions
            total_pnl, closed, errors = self._close_positions()
            reward += total_pnl * 0.1
        elif action[0] == 3:  # Transfer from spot to futures
            # print("Transferring balance from Spot to Futures")
            self._transfer_balance("spot_to_futures")
            reward -= 1  # Штраф за перевод
        elif action[0] == 4:  # Transfer from futures to spot
            # print("Transferring balance from Futures to Spot")
            self._transfer_balance("futures_to_spot")
            reward -= 1  # Штраф за перевод
        elif action[0] == 5:  # Hold
            for position in self.positions:
                reward += self._calculate_pnl(position) * 0.005
        
        # Логика удержания позиций
        for position in self.positions:
            self.adjust_stop_loss_take_profit(position)
            position["hold_time"] = (self.current_step - position["entry_step"]) * 100
            if position["hold_time"] < 250:  # Минимальное время удержания
                continue  # Пропускаем закрытие позиции
            if position["direction"] == "long" and self.current_price <= position["stop_loss"]:
                self._close_positions()
            elif position["direction"] == "long" and self.current_price >= position["take_profit"]:
                self._close_positions()
            elif position["direction"] == "short" and self.current_price >= position["stop_loss"]:
                self._close_positions()
            elif position["direction"] == "short" and self.current_price <= position["take_profit"]:
                self._close_positions()
                
        # Вознаграждение за общий баланс
        total_balance = self.spot_balance + self.futures_balance
        reward += (total_balance - self.prev_total_balance) * 0.05
        self.prev_total_balance = total_balance
        
        # Штраф за ликвидацию через reward
        if self._check_liquidation():
            reward -= 10  # Штраф за ликвидацию
        
        # Логируем состояние после выполнения действия
        self._log_balances()
        
        # Переход на следующий шаг
        self.current_step += 1
        # print(f"Next state: {next_state[:10]}... (truncated), Reward: {reward}, Done: {done}")
        # logging.info(f"Step: {self.current_step}, Action: {action}, Reward: {reward}, Total Reward: {self.pnl}")
        # Логируем состояние только каждые 100 шагов
        if self.current_step % 100 == 0:
            logging.info(f"Step: {self.current_step}, Spot balance: {self.spot_balance:.2f}, "
                        f"Futures balance: {self.futures_balance:.2f}, Positions: {len(self.positions)}")
            if self.positions:
                for i, position in enumerate(self.positions, start=1):
                    logging.info(f"Position {i}: {position}")

        return self._get_state(), reward, done

    def _open_position(self, direction):
        # Проверка наличия активных позиций
        if len(self.positions) >= 1:  # Ограничение на одну активную позицию
            logging.warning("Position not opened: An active position already exists.")
            return
        
        # Максимальная доля от баланса, которую можно использовать
        max_risk_percent = 0.02  # 2% риска на сделку
        min_position_size = 10  # Минимальный размер позиции
        max_position_size = self.futures_balance * 0.8  # Максимум 80% от баланса

        # Уверенность модели (например, трендовая сила)
        trend_strength = self.X[self.current_step][98]  # Используем `trend_strength` из 5-минутного интервала
        confidence_factor = max(0.1, min(1.0, abs(trend_strength)))  # Нормализуем уверенность от 0.1 до 1.0

        # Рассчитываем базовый размер позиции
        base_position_size = self.futures_balance * max_risk_percent * confidence_factor

        # Устанавливаем окончательный размер позиции в пределах лимитов
        position_size = float(max(min(base_position_size, max_position_size), min_position_size))

        if self.futures_balance <= 0:
            logging.warning("Insufficient futures balance to open position.")
            return

        if position_size > self.futures_balance * self.leverage:
            position_size = self.futures_balance * self.leverage  # Ограничиваем плечом

        liquidation_price = self._calculate_liquidation_price(position_size, direction)

        # Установка stop-loss и take-profit
        stop_loss = self.X[self.current_step][94]  # low_price_normalized из 5-минутного интервала
        take_profit = self.X[self.current_step][95]  # high_price_normalized из 5-минутного интервала

        position_id = len(self.positions) + 1

        self.positions.append({
            "position_id": position_id,
            "entry_price": self.current_price,
            "position_size": position_size,
            "direction": direction,
            "liquidation_price": liquidation_price,
            "entry_step": self.current_step,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "hold_time": 0  # Инициализация времени удержания
        })

        self.trade_log.append({
            "action": "open",
            "step": self.current_step,
            "entry_price": self.current_price,
            "direction": direction
        })

        logging.info(
            f"Opened {direction} - Entry price: {self.current_price:.5f}, "
            f"Size: {position_size:.5f}, Liquidation price: {liquidation_price:.5f}, "
        )

        logging.info(
            f"Confidence factor: {confidence_factor:.5f}, "
            f"Futures balance: {self.futures_balance:.5f}"
        )

        logging.info(
            f"Stop loss: {stop_loss:.5f}, Take profit: {take_profit:.5f}, "
        )

    def _close_positions(self):
        successfully_closed = []
        errors = []
        total_pnl = 0
        remaining_positions = []

        # Устанавливаем комиссию как долю от размера позиции
        commission_rate = 0.1  # 0.1% комиссии как на Binance

        for i, position in enumerate(self.positions):
            try:
                # Рассчитываем PnL для позиции
                pnl = self._calculate_pnl(position)
                # Вычисляем комиссию
                position_size = position.get("position_size", 0)
                commission = position_size * commission_rate
                # Уменьшаем PnL на размер комиссии
                pnl -= commission

                # Обновляем фьючерсный баланс с учетом PnL
                total_pnl += pnl
                self.futures_balance += pnl
                successfully_closed.append(i)

                logging.info(
                    f"Closed position - PnL: {pnl:.5f} (after commission: {commission:.5f}), "
                    f"Updated Futures balance: {self.futures_balance:.5f}"
                )
            except Exception as e:
                errors.append((i, str(e)))
                logging.error(f"Error closing position {i}: {e}")
            finally:
                # Логирование закрытия сделки
                self.trade_log.append({
                    "action": "close",
                    "entry_step": position.get("entry_step", self.current_step - 1),  # Корректный шаг входа
                    "step": self.current_step,  # Текущий шаг как шаг закрытия
                    "entry_price": position.get("entry_price", "N/A"),
                    "exit_price": self.current_price,
                    "position_size": position_size,
                    "pnl": pnl,  # PnL уже учитывает комиссию
                    "commission": commission,
                    "exit_spot_balance": self.spot_balance,
                    "exit_futures_balance": self.futures_balance
                })

        # Учет дополнительной комиссии на общий PnL (опционально)
        total_pnl -= abs(total_pnl) * commission_rate  # Если нужно еще одно снижение общего PnL
        self.pnl = total_pnl

        # Обновляем оставшиеся позиции
        self.positions = remaining_positions

        return total_pnl, successfully_closed, errors

    def generate_trade_report(self):
        """
        Генерация отчета по сделкам на основе trade_log.
        """
        report_data = []

        # Проходим по каждому логу в trade_log
        for log in self.trade_log:
            if log["action"] == "close":
                # Собираем данные для отчета
                report_data.append({
                    "Вход (шаг)": log.get("entry_step", "N/A"),
                    "Выход (шаг)": log.get("step", "N/A"),
                    "Цена входа": log.get("entry_price", "N/A"),
                    "Цена выхода": log.get("exit_price", "N/A"),
                    "Прибыль/Убыток": log.get("pnl", "N/A"),
                    "Размер позиции": log.get("position_size", "N/A"),
                    "Комиссия": log.get("commission", "N/A"),
                    "Stop Loss": log.get("stop_loss", "N/A"),
                    "Take Profit": log.get("take_profit", "N/A"),
                    "Конечный баланс (Spot)": log.get("exit_spot_balance", "N/A"),
                    "Конечный баланс (Futures)": log.get("exit_futures_balance", "N/A"),
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

        logging.info(f"Spot balance: {self.spot_balance}, Futures balance: {self.futures_balance}")
        # Логируем метрики
        logging.info(f"Win Rate: {win_rate:.2f}, Profitability Ratio: {profitability_ratio:.2f}")

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
        # logging.info(f"Updated Futures balance with PnL: {self.futures_balance:.2f}")

    def _log_balances(self):
        # logging.info(f"Spot balance: {self.spot_balance:.2f}, Futures balance: {self.futures_balance:.2f}")
        # for position in self.positions:
            # logging.info(f"Position: {position}")
        return

    def _transfer_balance(self, direction):
        """
        Перемещает средства между spot и futures с учетом активных позиций и минимальной необходимости.
        """
        transfer_amount = 0
        min_transfer_threshold = 5  # Минимальная сумма перевода

        if direction == "spot_to_futures":
            # Убедимся, что перевод имеет смысл
            if self.spot_balance > min_transfer_threshold:
                transfer_amount = min(self.spot_balance * 0.1, self.spot_balance)
                self.spot_balance -= transfer_amount
                self.futures_balance += transfer_amount
            else:
                logging.info("Spot balance too low for transfer to futures.")
        elif direction == "futures_to_spot":
            # Убедимся, что есть достаточно маржи для открытых позиций
            min_required_margin = sum(pos["position_size"] / self.leverage for pos in self.positions)
            if self.futures_balance - min_required_margin > min_transfer_threshold:
                transfer_amount = min(self.futures_balance * 0.1, self.futures_balance - min_required_margin)
                self.futures_balance -= transfer_amount
                self.spot_balance += transfer_amount
            else:
                logging.info("Futures balance too low or insufficient margin for transfer to spot.")

        if transfer_amount > 0:
            logging.info(f"Transferred {transfer_amount:.2f} from {direction.replace('_', ' ')}. "
                        f"Spot balance: {self.spot_balance:.2f}, Futures balance: {self.futures_balance:.2f}")
        return transfer_amount

    def _calculate_liquidation_price(self, position_size, direction):
        """
        Рассчитывает ликвидационную цену для позиции.
        """
        entry_price = self.current_price
        leverage = self.leverage
        margin = self.futures_balance
        maintenance_margin_rate = 0.005  # 0.5% для фьючерсов (может варьироваться)
        maintenance_margin = position_size * entry_price * maintenance_margin_rate

        if direction == "long":
            # Формула для лонга
            liquidation_price = entry_price - (margin - maintenance_margin) / (position_size * leverage)
        elif direction == "short":
            # Формула для шорта
            liquidation_price = entry_price + (margin - maintenance_margin) / (position_size * leverage)
        else:
            raise ValueError("Invalid direction: must be 'long' or 'short'")

        return round(liquidation_price, 5)  # Округляем до 5 знаков

    def _calculate_reward(self, position):
        """
        Рассчитывает награду на основе PnL.
        """
        pnl = self._calculate_pnl(position)
        reward = pnl - abs(pnl) * 0.001  # Учет комиссии
        logging.debug(f"Calculating reward: PnL = {pnl}, Reward before commission: {reward}")
        
        if pnl > 0:
            return reward * 1.5  # Награда за прибыль
        else:
            return reward * 0.5  # Снижение награды за убыток

    def _calculate_pnl(self, position):
        """
        Расчет PnL по позиции.
        """
        if position["direction"] == "long":
            return (self.current_price - position["entry_price"]) * abs(position["position_size"])
        elif position["direction"] == "short":
            return (position["entry_price"] - self.current_price) * abs(position["position_size"])
        return 0

    def _check_liquidation(self):
        """
        Проверяет ликвидацию всех открытых позиций.
        Возвращает True, если произошла ликвидация, иначе False.
        """
        liquidation_triggered = False  # Флаг ликвидации
        total_pnl = 0  # Общий PnL от закрытых позиций
        remaining_positions = []  # Оставшиеся после ликвидации позиции

        for position in self.positions:
            if position["direction"] == "long" and self.current_price <= position["liquidation_price"]:
                logging.warning(f"Liquidation triggered for Long position at {self.current_price}")
                loss = position["position_size"] / self.leverage
                self.futures_balance -= loss
                total_pnl += self._calculate_pnl(position)  # Учитываем PnL от ликвидируемой позиции
                liquidation_triggered = True

            elif position["direction"] == "short" and self.current_price >= position["liquidation_price"]:
                logging.warning(f"Liquidation triggered for Short position at {self.current_price}")
                loss = position["position_size"] / self.leverage
                self.futures_balance -= loss
                total_pnl += self._calculate_pnl(position)  # Учитываем PnL от ликвидируемой позиции
                liquidation_triggered = True

            else:
                remaining_positions.append(position)  # Оставляем позиции, не затронутые ликвидацией

        # Обновляем оставшиеся позиции
        self.positions = remaining_positions

        # Если ликвидация произошла, обновляем балансы
        if liquidation_triggered:
            self.spot_balance += max(0, self.futures_balance)  # Переводим остаток в spot
            self.futures_balance = max(0, self.futures_balance)
            logging.info(f"Liquidation occurred. Total PnL: {total_pnl:.5f}, Remaining futures balance: {self.futures_balance:.2f}")
            return True

        return False

    def _log_state(self):
        # Логируем состояние только каждые 100 шагов
        if self.current_step % 100 == 0:
            logging.info(f"Step: {self.current_step}, Spot balance: {self.spot_balance:.2f}, "
                        f"Futures balance: {self.futures_balance:.2f}, Positions: {len(self.positions)}")
            if self.positions:
                for i, position in enumerate(self.positions, start=1):
                    logging.info(f"Position {i}: {position}")
        # logging.info(f"Step: {self.current_step}, Spot balance: {self.spot_balance:.2f}, "
        #              f"Futures balance: {self.futures_balance:.2f}, Positions: {len(self.positions)}")
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
            nn.Linear(self.hidden_size, self.action_size)
        )

    def update_target_network(self):
        """
        Синхронизирует целевую сеть с основной.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state):
        if not isinstance(state, dict) or "positions" not in state:
            raise ValueError("State must be a dictionary containing 'positions'.")
        if "positions" not in state or not isinstance(state["positions"], list):
            logging.error(f"Invalid 'positions' in state: {state.get('positions')}")
            return [5, []]  # Действие ожидания при некорректных данных

        action_index = 0
        positions_to_close = []

        if np.random.rand() <= self.epsilon:
            action_index = random.choice(range(self.action_size))  # Исследование
            if state["positions"]:
                positions_to_close = random.sample(range(len(state["positions"])), k=min(len(state["positions"]), 2))
        else:
            state_tensor = torch.tensor(state["state_data"], dtype=torch.float32).unsqueeze(0).to("cuda")  # Добавляем batch-измерение
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_index = torch.argmax(q_values).item()  # Эксплуатация
            if state["positions"]:
                positions_to_close = random.sample(range(len(state["positions"])), k=min(len(state["positions"]), 2))

        return [action_index, positions_to_close]

    def store_experience(self, state, action, reward, next_state, done):
        if not isinstance(state, dict) or "state_data" not in state:
            raise ValueError(f"Invalid state to store: {state}")
        if not isinstance(next_state, dict) or "state_data" not in next_state:
            raise ValueError(f"Invalid next_state to store: {next_state}")
        self.memory.append((state, action, reward, next_state, done))


    @staticmethod
    def convert_state_to_array(state):
        if isinstance(state, dict) and "state_data" in state:
            return state["state_data"]
        # print(f"Invalid state in `convert_state_to_array`: {state}")
        raise TypeError("State must be a dictionary containing 'state_data'.")

    def learn(self):
        """
        Обучение модели на основе опыта.
        """
        if len(self.memory) < self.batch_size:
            return  # Недостаточно данных для обучения

        # Сэмплируем случайные батчи
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([
            torch.tensor(DQLAgent.convert_state_to_array(state), dtype=torch.float32).to("cuda")
            for state in states
        ])

        # Преобразуем `actions` в тензор индексов
        actions = [action[0] for action in actions]  # Извлекаем индексы действий
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to("cuda")
        # print(f"Tensor shape after conversion: {actions.shape}")
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to("cuda")
        next_states = torch.stack([
            torch.tensor(DQLAgent.convert_state_to_array(state), dtype=torch.float32).to("cuda")
            for state in next_states
        ])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to("cuda")

        # Q-значения для текущих состояний
        q_values = self.q_network(states).gather(1, actions)

        # Q-значения для следующих состояний (по целевой сети)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]

        # Целевая функция (Bellman equation)
        targets = rewards + (1 - dones) * self.discount_factor * max_next_q_values

        # Loss
        loss = nn.MSELoss()(q_values, targets)

        # Обновление сети
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Уменьшаем epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def apply_rewards_and_penalties(self, reward, position_pnl, current_step, trade_log):
        penalty = 0
        bonus = 0

        # Штраф за отрицательный PnL
        if position_pnl < 0:
            penalty += abs(position_pnl) * 0.2  # Увеличенный штраф

        # Вознаграждение за положительный PnL
        if position_pnl > 0:
            bonus += position_pnl * 0.5  # Увеличенный бонус

        # Бонус за достижение take profit
        if any(log.get("pnl") is not None and log.get("pnl") > 0 for log in trade_log[-1:]):  # Последняя сделка с профитом
            bonus += 2

        # Штраф за частые убытки
        if current_step > 10:
            recent_losses = [log['pnl'] for log in trade_log[-10:] if 'pnl' in log and log['pnl'] < 0]
            if len(recent_losses) > 5:  # Более 50% убытков
                penalty += 1

        return reward - penalty + bonus

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
    # query = f"SELECT * FROM order_book_data order by id LIMIT 380000"
    # query = f"SELECT * FROM order_book_data order by id LIMIT 10000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 45000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 70000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 25000"
    query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 6000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 1000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id"
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

def prepare_data_for_orderbook_non_overlapping(orderbook_data: pd.DataFrame) -> np.ndarray:
    """
    Подготавливает данные для обучения модели, исключая перекрывающиеся строки.
    :param orderbook_data: Данные стакана.
    :return: Массив X для обучения.
    """
    X = []
    # Устанавливаем опции печати для numpy
    # np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.14f}"})
    orderbook_data["timestamp"] = pd.to_datetime(orderbook_data["timestamp"])

    # Извлекаем временные рамки
    first_open_time = orderbook_data.iloc[0]["timestamp"]
    last_open_time = orderbook_data.iloc[-1]["timestamp"]

    # Получаем прогнозы интервалов
    intervals_predictions = get_intervals_predictions(first_open_time, last_open_time)

    # Обрабатываем данные стакана без перекрытия
    for i in range(len(orderbook_data)):
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

    return np.array(X)

# Убедимся, что границы можно легко увидеть
def add_null_row_to_csv(filename):
    null_row = pd.DataFrame([["NULL"] * 14])  # 14 соответствует количеству колонок в DataFrame
    null_row.to_csv(filename, mode="a", index=False, header=False)

def prepare_targets(orderbook_data, sequence_length, target_column="mid_price"):
    """
    Формируем целевые метки (y) для данных.

    :param orderbook_data: Данные стакана.
    :param sequence_length: Длина последовательности для стакана.
    :param target_column: Столбец для определения целевых меток.
    :return: Массив целевых меток (y).
    """
    y = []

    for i in range(sequence_length, len(orderbook_data)):
        # Текущая цена mid_price
        current_price = orderbook_data.iloc[i][target_column]

        # Цена на следующий шаг (целевой прогноз)
        next_price = orderbook_data.iloc[i + 1][target_column] if i + 1 < len(orderbook_data) else current_price

        # Направление сделки
        if next_price > current_price:
            direction = 0  # long
        elif next_price < current_price:
            direction = 1  # short
        else:
            direction = 2  # hold

        min_confidence = 0
        max_confidence = 0.015
        
        # ex1_pr_1 = 0.826121
        # ex1_pr_2 = 0.822121
        # ex1_conf = abs(ex1_pr_1 - ex1_pr_2)
        # print(f"ex1_conf before: {ex1_conf}")
        # ex1_conf = (ex1_conf - min_confidence) / (max_confidence - min_confidence)
        # print(f"ex1_conf after: {ex1_conf}")
        # print()

        # Уверенность (разница в ценах)
        confidence = abs(next_price - current_price)
        # print(f"{confidence}")
        confidence = (confidence - min_confidence) / (max_confidence - min_confidence)
        # print(f"{confidence} {min_confidence} {max_confidence}")

        # Логирование для проверки данных
        logging.debug(f"Index {i}: current_price={current_price}, next_price={next_price}, "
                      f"direction={direction}, confidence={confidence}")

        # Добавляем направление и уверенность
        y.append([direction, confidence])
        
    # Преобразуем в Tensor
    if len(y) == 0:
        logging.warning("No targets generated. Check the input data and sequence length.")
    else:
        logging.info(f"Generated {len(y)} targets.")

    # Преобразуем в Tensor
    y = torch.tensor(y, dtype=torch.float32)
    return y

def log_predictions(model, X_train, y_train):
    model.eval()
    with torch.no_grad():
        # Логируем последние 5 строки тренировочного набора
        train_sample_data = X_train  # Последние 5 строки из X_train
        train_targets = y_train  # Последние 5 целевых значений из y_train

        train_sample_data_tensor = torch.tensor(train_sample_data, dtype=torch.float32).to("cuda")
        direction_pred, confidence_pred = model(train_sample_data_tensor)

        # Используем только последний временной шаг
        direction_pred = direction_pred[:, -1, :].cpu().numpy()  # Прогнозы для направления
        confidence_pred = confidence_pred[:, -1, :].cpu().numpy().flatten()  # Прогнозы для уверенности
        actual_direction = train_targets[:, 0].cpu().numpy()  # Целевые метки для направления
        actual_confidence = train_targets[:, 1].cpu().numpy()  # Целевые метки для уверенности

        # Определяем предсказанное направление
        predicted_direction = np.argmax(direction_pred, axis=1)  # Индекс максимального значения

        # Создаем DataFrame
        train_results_df = pd.DataFrame({
            "Predicted_direction_class_0": direction_pred[:, 0],
            "Predicted_direction_class_1": direction_pred[:, 1],
            "Predicted_direction_class_2": direction_pred[:, 2],
            "Predicted_direction": predicted_direction,  # Новая колонка с выбранным направлением
            "Actual_direction": actual_direction,
            "Predicted_confidence": confidence_pred,
            "Actual_confidence": actual_confidence,
        })

        # Настраиваем pandas для отображения чисел без научной нотации
        pd.set_option("display.float_format", "{:.10f}".format)
        pd.set_option("display.max_rows", None)  # Показывать все строки
        pd.set_option("display.max_columns", None)  # Показывать все столбцы
        pd.set_option("display.width", 1000)  # Увеличить ширину вывода

        # Логируем DataFrame
        # logging.info(f"Training predictions on entire training set:\n{train_results_df}")
        # print(train_results_df)

        # Сброс настроек pandas после вывода
        pd.reset_option("display.float_format")
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

import matplotlib.pyplot as plt

def plot_trades(data):
    plt.figure(figsize=(12, 6))

    # Основной график цен
    plt.plot(data['timestamp'], data['mid_price'], label='Mid Price', alpha=0.7)

    # Отметки сделок
    long_trades = data[data['direction'] == 0]
    short_trades = data[data['direction'] == 1]

    plt.scatter(long_trades['timestamp'], long_trades['entry_price'], color='green', label='Long Entry', marker='^', alpha=0.8)
    plt.scatter(long_trades['timestamp'], long_trades['exit_price'], color='lime', label='Long Exit', marker='v', alpha=0.8)
    
    plt.scatter(short_trades['timestamp'], short_trades['entry_price'], color='red', label='Short Entry', marker='^', alpha=0.8)
    plt.scatter(short_trades['timestamp'], short_trades['exit_price'], color='pink', label='Short Exit', marker='v', alpha=0.8)

    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('Trade Signals')
    plt.legend()
    plt.show()

def calculate_targets(data: pd.DataFrame, window_size: int = 15) -> pd.DataFrame:
    """
    Вычисление таргетов для модели, включая entry/exit цены, направления и потенциальную прибыль.
    
    :param data: DataFrame с ценами и объемами.
    :param window_size: Длительность окна для анализа (в минутах).
    :return: DataFrame с таргетами.
    """
    targets = []
    
    for i in range(len(data) - window_size):
        # Цены открытия сделки
        entry_price = data['open_price'].iloc[i]
        window_data = data.iloc[i + 1:i + 1 + window_size]
        
        # Long
        max_price = window_data['high_price'].max()
        potential_pnl_long = (max_price - entry_price) / entry_price
        max_negative_long = (entry_price - window_data['low_price'].min()) / entry_price
        
        # Short
        min_price = window_data['low_price'].min()
        potential_pnl_short = (entry_price - min_price) / entry_price
        max_negative_short = (window_data['high_price'].max() - entry_price) / entry_price

        # Направление сделки
        if potential_pnl_long > potential_pnl_short:
            direction = 0  # long
            exit_price = max_price
            potential_pnl = potential_pnl_long
            max_negative_diff = max_negative_long
        else:
            direction = 1  # short
            exit_price = min_price
            potential_pnl = potential_pnl_short
            max_negative_diff = max_negative_short

        # Время удержания сделки
        hold_time = window_data.index[window_data['high_price'].idxmax()] - data.index[i]

        # Сохраняем таргеты
        targets.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'potential_pnl': potential_pnl,
            'max_negative_diff': max_negative_diff,
            'hold_time': hold_time.total_seconds() / 60  # В минутах
        })
    
    return pd.DataFrame(targets)

global cached_data

cached_data = None
def fetch_cached_data():
    global cached_data
    if cached_data is None:
        cached_data = fetch_orderbook_data()
    return cached_data

def objective(trial):
    logging.info("Starting a new trial")
    
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    sequence_length = trial.suggest_int("sequence_length", 30, 100)
    batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    logging.info(f"Trial parameters - hidden_size: {hidden_size}, num_layers: {num_layers}, dropout: {dropout}, learning_rate: {learning_rate}, sequence_length: {sequence_length}, batch_size: {batch_size}")
    
    # orderbook_data = fetch_orderbook_data()
    orderbook_data = fetch_cached_data()
    # logging.info(f"Orderbook data fetched: {len(orderbook_data)} rows")

    # # # TODO fetching data from db is too busy work
    # # # TODO it should be done one time for all trials to economy time of optimization process
    # # X = prepare_data_for_orderbook(orderbook_data, sequence_length)
    # TODO Cup data
	# 		"mid_price" "sum_bid_volume" "sum_ask_volume" "bid_ask_imbalance"
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

    # TODO  "open_price_normalized" "close_price_normalized" "low_price_normalized" "high_price_normalized"
	# TODO  "trend_price_difference" "trend_sma" "trend_strength"
    # TODO  Interval 5m data 92	93	94	95	96	97	98
    # TODO  Interval 1m data 114	115	116	117	118	119	120
    X = prepare_data_for_orderbook_non_overlapping(orderbook_data)
    # print("\n")
    # print(f"{X[0][92]} {X[0][93]} {X[0][94]} {X[0][95]}")
    # print(f"{X[0][96]} {X[0][97]} {X[0][98]}")
    # print("\n")
    # print(f"{X[0][114]} {X[0][115]} {X[0][116]} {X[0][117]}")
    # print(f"{X[0][118]} {X[0][119]} {X[0][120]}")
    # logging.info(f"Prepared data shape: {X.shape}")
    # print(f"Shape of X: {np.array(X).shape}")
    # print(f"Sample from X[0]: {X[0]}")
    # print(f"Sample from X[0]: {X[1]}")
    # print(f"Sample from X[0]: {X[2]}")
    # print(f"Sample from X[0]: {X[3]}")
    # print(f"Sample from X[0]: {X[3].shape}")

    agent = DQLAgent(state_size=141, action_size=6)  # Пример с Deep Q-Learning
    env = TradingEnvironment(X)

    episodes = 1000
    target_update_frequency = 10  # Как часто обновлять целевую сеть

    for episode in range(episodes):
        state = env.reset()
        # print("Initial state:", state)
        total_reward = 0

        while True:
            # Выбор действия
            # print(f"statestatestate: {state}")
            # logging.info(f"Current state: {state}")
            if not isinstance(state, dict) or "positions" not in state:
                logging.error(f"Invalid state: {state}")
                raise ValueError("State must be a dictionary containing 'positions'.")

            action = agent.choose_action(state)

            # Совершаем шаг в среде
            next_state, reward, done = env.step(action)
            
            # Логирование балансов после каждого шага
            env._log_balances()

            # Применяем систему штрафов и наград (если требуется)
            penalty = abs(reward) * 0.01 if reward < 0 else 0  # Штраф за отрицательную награду
            # Вызов с передачей позиций
            adjusted_reward = agent.apply_rewards_and_penalties(reward, penalty, env.current_step, env.trade_log)

            # Сохраняем опыт в памяти агента
            agent.store_experience(state, action, adjusted_reward, next_state, done)

            # Переходим к следующему состоянию
            state = next_state
            total_reward += adjusted_reward

            # Обучаем агента после каждого шага
            agent.learn()

            if done:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
                trade_report = env.generate_trade_report()
                # print(trade_report)
                logging.info(f"Trade report: \n{trade_report}")
                logging.info(f"Episode {episode + 1}: Total Reward = {total_reward}")
                break

        # Обновляем целевую сеть каждые `target_update_frequency` эпизодов
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()
            print(f"Target network updated at episode {episode + 1}")
            logging.info(f"Target network updated at episode {episode + 1}")

    # # TEST 1
    # # Создадим тестовую позицию
    # test_position = {
    #     "direction": "long",  # Длинная позиция
    #     "entry_price": 0.798157,  # Цена входа
    #     "position_size": 1000.0,  # Размер позиции
    # }

    # # Установим текущую цену
    # current_price = 0.798045

    # # Рассчитаем PnL
    # def test_calculate_pnl(position, current_price):
    #     if position["direction"] == "long":
    #         return (current_price - position["entry_price"]) * position["position_size"]
    #     elif position["direction"] == "short":
    #         return (position["entry_price"] - current_price) * position["position_size"]
    #     return 0

    # print(denormalize(test_position["entry_price"]))
    # print(denormalize(current_price))
    # # Вычисляем PnL
    # pnl = test_calculate_pnl(test_position, current_price)
    # print(f"Рассчитанный PnL: {pnl:.10f}")
    # print("____________________________________________________")

    # # TEST 2
    # # Тестовые данные для среды
    # test_env = {
    #     "positions": [
    #         {"position_id": 1, "entry_price": 0.798157, "position_size": 1000.0, "direction": "long"},
    #     ],
    #     "futures_balance": 50.0,  # Начальный баланс
    #     "current_price": 0.798045,  # Текущая цена
    # }

    # # Функция закрытия позиций
    # def test_close_positions(env):
    #     total_pnl = 0
    #     remaining_positions = []

    #     for position in env["positions"]:
    #         pnl = test_calculate_pnl(position, env["current_price"])
    #         total_pnl += pnl
    #         print(f"Закрытие позиции ID {position['position_id']}: PnL = {pnl:.10f}")
        
    #     env["futures_balance"] += total_pnl  # Обновляем баланс
    #     env["positions"] = remaining_positions  # Все позиции закрыты
    #     return total_pnl, env["futures_balance"]

    # # Тестируем функцию
    # pnl, updated_balance = test_close_positions(test_env)
    # print(f"Итоговый PnL: {pnl:.10f}")
    # print(f"Обновленный баланс фьючерсов: {updated_balance:.10f}")
    # print("____________________________________________________")

    # # TEST 3
    # # Тестовые данные журнала сделок
    # trade_log = [
    #     {
    #         "action": "close",
    #         "step": 596,
    #         "position_id": 2,
    #         "entry_price": 0.798157,
    #         "exit_price": 0.798045,
    #         "pnl": -0.004750877,
    #         "exit_spot_balance": 40.00548,
    #         "exit_futures_balance": 61.964428,
    #     },
    # ]

    # # Генерация отчета
    # def test_generate_trade_report(trade_log):
    #     report_data = [
    #         {
    #             "Сделка №": log["position_id"],
    #             "Вход (шаг)": log["step"] - 1,
    #             "Выход (шаг)": log["step"],
    #             "Цена входа": log["entry_price"],
    #             "Цена выхода": log["exit_price"],
    #             "Прибыль/Убыток": log["pnl"],
    #             "Конечный баланс (Spot)": log["exit_spot_balance"],
    #             "Конечный баланс (Futures)": log["exit_futures_balance"],
    #         }
    #         for log in trade_log if log["action"] == "close"
    #     ]
    #     return pd.DataFrame(report_data)

    # # Проверяем функцию
    # report = test_generate_trade_report(trade_log)
    # print(report)

    # orderbook_data = fetch_orderbook_data()
    # logging.info(f"Orderbook data fetched: {len(orderbook_data)} rows")

    # # # TODO fetching data from db is too busy work
    # # # TODO it should be done one time for all trials to economy time of optimization process
    # X = prepare_data_for_orderbook(orderbook_data, sequence_length)
    # logging.info(f"Prepared data shape: {X.shape}")
    # # Формируем метки y
    # y = prepare_targets(orderbook_data, sequence_length)
    # logging.info(f"Prepared targets shape: {y.shape}")
    # print("Direction distribution:", np.unique(y[:, 0].cpu().numpy(), return_counts=True))
    # print("Confidence range:", np.min(y[:, 1].cpu().numpy()), np.max(y[:, 1].cpu().numpy()))

    # # Создаем датасет
    # # dataset = TradingDataset(X, y)
    # # Создаем DataLoader
    # # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # Пример использования
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # logging.info(
    #     f"Data split - X_train: {X_train.shape}, X_test: {X_test.shape}, "
    #     f"y_train: {y_train.shape}, y_test: {y_test.shape}"
    # )

    # # logging.info(
    # #     f"Data values - X_train: \n{X_train[:5]}, \nX_test: \n{X_test[:5]}\n, "
    # #     f"y_train: \n{y_train[:5]}, \ny_test: \n{y_test[:5]}"
    # # )

    # # torch.set_printoptions(edgeitems=torch.inf)
    # logging.info(
    #     f"\ny_train: \n{y_train}, \ny_test: \n{y_test}"
    # )

    # # Создаем объекты TradingDataset
    # train_dataset = TradingDataset(X_train, y_train)
    # test_dataset = TradingDataset(X_test, y_test)
    # logging.info(f"Datasets created - Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # # print("y_train sample:", y_train[:5])
    # # print("y_test sample:", y_test[:5])

    # # Создаем DataLoader для батчевой обработки данных
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # Проверка батчей данных
    # for batch_inputs, batch_targets in train_loader:
    #     logging.debug(f"Batch inputs shape: {batch_inputs.shape}, Batch targets shape: {batch_targets.shape}")
    #     break  # Проверяем только первую итерацию

    # input_size = X.shape[2]  # Число входных признаков
    # # output_size = 3
    # model = TradingModel(input_size, hidden_size).to("cuda")
    # logging.info(f"Model created with input_size: {input_size}, hidden_size: {hidden_size}")

    # # class_weights = torch.tensor([1.0, 1.0, len(y) / sum(y[:, 0] == 2)], dtype=torch.float32).to("cuda")
    # class_counts = torch.tensor([sum(y[:, 0] == 0), sum(y[:, 0] == 1), sum(y[:, 0] == 2)], dtype=torch.float32).to("cuda")
    # class_weights = 1.0 / class_counts  # Или другое обратное соотношение
    # criterion_direction = nn.CrossEntropyLoss(weight=class_weights)
    # # criterion_direction = nn.CrossEntropyLoss()
    # criterion_confidence = nn.MSELoss()
    # # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # logging.info("Loss functions and optimizer initialized")

    # train_model(model, train_loader, criterion_direction, criterion_confidence, optimizer, num_epochs=10)
    # logging.info("Model training completed")
    # evaluated = evaluate_model(model, test_loader, criterion_direction, criterion_confidence)
    # logging.info(f"Evaluation completed - Loss: {evaluated:.14f}")
    # print(evaluated)

    # log_predictions(model, X_train, y_train)
    # return evaluated
    return

    # # Устройство для обучения
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Инициализация модели и оптимизатора
    # meta_model = MetaModel(input_size=10, hidden_size=64, output_size=4).to(device)
    # optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)

    # # Подготовка данных
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # # Обучение модели
    # trained_model = train_meta_model_with_rewards(meta_model, train_loader, optimizer, epochs=10, device=device)

    return

    # Инициализация модели
    meta_model = load_meta_model()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Загрузка данных для обучения
    train_data, val_data = fetch_training_data(sequence_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Тренировка модели
    for epoch in range(10):  # Пример: 10 эпох
        train_loss = train_model(meta_model, train_loader, criterion, optimizer)
        val_loss = evaluate_model(meta_model, val_loader, criterion)
        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.14f}, Val Loss: {val_loss:.14f}")

    # Возвращаем значение функции потерь для валидации
    return val_loss

    # Инициализация клиентов и моделей
    binance_client = initialize_binance_client()
    meta_model = load_meta_model()

    decision_engine = DecisionEngine(meta_model)
    order_manager = OrderManager(binance_client)
    position_monitor = PositionMonitor(binance_client)

    while True:
        try:
            # Получить прогнозы от моделей интервалов и стакана
            interval_predictions = fetch_interval_predictions()
            orderbook_predictions = fetch_orderbook_predictions()

            # Получить текущие позиции
            positions = position_monitor.get_active_positions()

            # Принять решение
            decisions = decision_engine.make_decision(interval_predictions, orderbook_predictions, positions)

            # Выполнить действия
            for decision in decisions:
                if decision["action"] == "open":
                    order_manager.open_position(
                        decision["direction"],
                        decision["entry_price"],
                        decision["stop_loss"],
                        decision["take_profit"],
                        quantity=0.01
                    )
                elif decision["action"] == "close":
                    order_manager.close_position(decision["position_id"])
            
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            time.sleep(5)

def main():
    logging.info("Starting hyperparameter optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)  # 50 испытаний для оптимизации
    print("Лучшие гиперпараметры:", study.best_params)
    print("Лучшее значение потерь:", study.best_value)
    logging.info(f"Optimization completed - Best Params: {study.best_params}, Best Loss: {study.best_value:.14f}")

if __name__ == "__main__":
    main()