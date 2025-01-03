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

class TradingEnvironment:
    def __init__(self, X, initial_spot_balance=50, initial_futures_balance=50):
        """
        Торговая среда для обучения модели.
        """
        self.X = X  # Входные данные (интервалы и стакан)
        self.current_step = 0  # Текущая позиция в данных
        # self.state_size = X.shape[1]
        self.state_size = 140
        # print(f"self.state_size: {self.state_size}")
        # print(f"X.shape: {X.shape}")
        self.spot_balance = initial_spot_balance
        self.futures_balance = initial_futures_balance
        self.leverage = 10
        self.positions = []
        self.trade_log = []
        # Вызов проверки структуры данных X
        # self.validate_X_structure()   

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

    def reset(self):
        """
        Сброс среды до начального состояния.
        """
        self.current_step = 0
        self.spot_balance = 50
        self.futures_balance = 50
        self.positions = []
        self.trade_log = []
        logging.info("Environment reset.")
        state = self._get_state()
        # print(f"state state state: {state}")
        # print(f"shape shape shape: {state.shape}")
        return state

    def _get_state(self):
        if self.current_step >= len(self.X):
            logging.warning("Current step exceeds data length. Returning default state.")
            return np.zeros(self.state_size, dtype=np.float32)  # Или другое значение по умолчанию
        row_data = self.X[self.current_step]
        # Извлекаем текущую цену
        price_data = self.X[self.current_step][0]
        if isinstance(price_data, (np.ndarray, list)):
            if len(price_data) == 1:
                price_data = float(price_data[0])  # Извлекаем единственное значение
            else:
                raise ValueError(f"Expected price_data to have size 1, but got {len(price_data)}")

        return np.concatenate([
            np.array([
                float(self.spot_balance),
                float(self.futures_balance),
                float(self.leverage),
                float(len(self.positions))
            ], dtype=np.float32),
            row_data  # Вся строка данных
        ])

    def step(self, action):
        reward = 0
        done = False

        # Проверка завершения эпизода
        if self.current_step >= len(self.X) or self.futures_balance <= 0:
            done = True
            print("Episode ended. No more steps or balance depleted.")
            return self._get_state(), reward, done

        # Выполнение действия
        if action[0] == 0:  # Open Long
            # print("Opening Long position")
            self._open_position("long")
        elif action[0] == 1:  # Open Short
            # print("Opening Short position")
            self._open_position("short")
        elif action[0] == 2:  # Close Positions
            positions_to_close = action[1]
            total_pnl, closed, errors = self._close_positions(positions_to_close)
            # print(f"Closed positions: {closed}, Errors: {errors}")
            reward += total_pnl
        elif action[0] == 3:  # Transfer from spot to futures
            # print("Transferring balance from Spot to Futures")
            reward -= self._transfer_balance("spot_to_futures")
        elif action[0] == 4:  # Transfer from futures to spot
            # print("Transferring balance from Futures to Spot")
            reward -= self._transfer_balance("futures_to_spot")
        elif action[0] == 5:  # Ожидание
            # print("Waiting: no action taken this step")
            reward = 0  # Вознаграждение может быть нейтральным (0) или небольшим штрафом за упущенные возможности


        # Проверка ликвидации
        self._check_liquidation()

        # Переход на следующий шаг
        self.current_step += 1
        next_state = self._get_state()
        # print(f"Next state: {next_state[:10]}... (truncated), Reward: {reward}, Done: {done}")
        return next_state, reward, done

    def _open_position(self, direction):
        """
        Открытие позиции.
        """
        # Ограничение количества позиций
        if len(self.positions) >= 5:
            print("Cannot open new position: position limit reached.")
            return
        
        if self.futures_balance <= 0:
            logging.warning("Insufficient futures balance to open position.")
            return
        
        # Преобразуем данные в скаляр, если это массив
        entry_price = self.X[self.current_step][0]
        position_size = float(min(self.futures_balance * 0.5, self.futures_balance * self.leverage))
        liquidation_price = self._calculate_liquidation_price(entry_price, position_size, direction)

        self.positions.append({
            "entry_price": entry_price,
            "position_size": position_size,
            "direction": direction,
            "liquidation_price": liquidation_price
        })
        logging.info(f"Opened {direction} position: Entry price: {entry_price}, Size: {position_size}, "
                     f"Liquidation price: {liquidation_price}")

    def _close_positions(self, positions_to_close):
        total_pnl = 0
        successfully_closed = []
        errors = []

        current_price = self.X[self.current_step][0]  # Текущая цена
        remaining_positions = []

        for i, position in enumerate(self.positions):
            if i in positions_to_close:
                try:
                    pnl = self._calculate_pnl(position, current_price)
                    total_pnl += pnl
                    successfully_closed.append(i)
                except Exception as e:
                    errors.append((i, str(e)))
            else:
                remaining_positions.append(position)

        total_pnl -= abs(total_pnl) * 0.001  # Учет комиссии
        self.futures_balance += total_pnl
        self.positions = remaining_positions

        return total_pnl, successfully_closed, errors

    def _transfer_balance(self, direction):
        """
        Перемещение средств между кошельками.
        """
        transfer_amount = self.spot_balance * 0.1 if direction == "spot_to_futures" else self.futures_balance * 0.1
        if direction == "spot_to_futures" and self.spot_balance >= transfer_amount:
            self.spot_balance -= transfer_amount
            self.futures_balance += transfer_amount
        elif direction == "futures_to_spot" and self.futures_balance >= transfer_amount:
            self.futures_balance -= transfer_amount
            self.spot_balance += transfer_amount
        logging.info(f"Transferred {transfer_amount:.2f} from {direction.replace('_', ' ')}. "
                     f"Spot balance: {self.spot_balance:.2f}, Futures balance: {self.futures_balance:.2f}")
        return transfer_amount

    def _calculate_liquidation_price(self, entry_price, position_size, direction):
        max_loss = self.futures_balance / position_size
        entry_price = float(entry_price) if isinstance(entry_price, (np.ndarray, list)) else entry_price

        if direction == "long":
            return entry_price - max_loss
        elif direction == "short":
            return entry_price + max_loss
        return None

    def _calculate_pnl(self, position, current_price):
        """
        Расчет PnL по позиции.
        """
        if position["direction"] == "long":
            return (current_price - position["entry_price"]) * position["position_size"]
        elif position["direction"] == "short":
            return (position["entry_price"] - current_price) * abs(position["position_size"])
        return 0

    def _check_liquidation(self):
        for position in self.positions:
            current_price = self.X[self.current_step][0]
            current_price = float(current_price) if isinstance(current_price, (list, np.ndarray)) else current_price

            liquidation_price = position["liquidation_price"]
            # liquidation_price = float(liquidation_price) if isinstance(liquidation_price, (np.ndarray, list)) else liquidation_price
            
            # print(f"current_price: {current_price}, type: {type(current_price)}")
            # print(f"position['liquidation_price']: {position['liquidation_price']}, type: {type(position['liquidation_price'])}")
            if position["direction"] == "long" and current_price <= liquidation_price:
                self.futures_balance = 0
                self.positions = []
                logging.warning("Long position liquidated!")
            elif position["direction"] == "short" and current_price >= liquidation_price:
                self.futures_balance = 0
                self.positions = []
                logging.warning("Short position liquidated!")

    def _log_state(self):
        logging.info(f"Step: {self.current_step}, Spot balance: {self.spot_balance:.2f}, "
                     f"Futures balance: {self.futures_balance:.2f}, Positions: {len(self.positions)}")
        if self.positions:
            for i, position in enumerate(self.positions, start=1):
                logging.info(f"Position {i}: {position}")

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

        # Модель
        self.q_network = self._build_model()
        self.target_network = self._build_model()
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
        if np.random.rand() <= self.epsilon:
            action_index = random.choice(range(self.action_size))  # Исследование
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Добавляем batch-измерение
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_index = torch.argmax(q_values).item()  # Эксплуатация
        # Пример возвращения списка с дополнительными данными

        return [action_index, []]  # Второй элемент — пустой список, его нужно настроить по логике

    def store_experience(self, state, action, reward, next_state, done):
        """
        Сохраняет опыт для последующего обучения.
        """
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)  # Удаляем самый старый опыт
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """
        Обучение модели на основе опыта.
        """
        if len(self.memory) < self.batch_size:
            return  # Недостаточно данных для обучения

        # Сэмплируем случайные батчи
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Преобразуем в тензоры
        # print(f"Size of states: {len(states)}, Shape of each state: {[s.shape if isinstance(s, np.ndarray) else len(s) for s in states]}")
        # states = torch.tensor(states, dtype=torch.float32)
        states = torch.stack([torch.tensor(state, dtype=torch.float32) for state in states])
        # print(f"Tensor shape after conversion: {states.shape}")
        # print(f"Size of states: {len(actions)}, Shape of each state: {[s.shape if isinstance(s, np.ndarray) else len(s) for s in actions]}")
        # Преобразуем `actions` в тензор индексов
        actions = [action[0] for action in actions]  # Извлекаем индексы действий
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        # print(f"Tensor shape after conversion: {actions.shape}")
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        # next_states = torch.tensor(next_states, dtype=torch.float32)
        next_states = torch.stack([torch.tensor(state, dtype=torch.float32) for state in next_states])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

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

    def apply_rewards_and_penalties(self, reward, penalty):
        """
        Применяет систему наград и штрафов.
        """
        return reward - penalty

def initialize_binance_client():
    with open("acc_config.json", "r") as file:
        acc_config = json.load(file)

    # Укажите свои API ключи
    API_KEY = acc_config["API_KEY"]
    API_SECRET = acc_config["API_SECRET"]

    return Client(API_KEY, API_SECRET)

def train_model(model, train_loader, criterion_direction, criterion_confidence, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        # for inputs, targets in train_loader:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
            inputs, targets = inputs.to("cuda"), targets.to("cuda")

            # # Прогнозы модели
            direction_pred, confidence_pred = model(inputs)

            # Приведение к правильным формам
            direction_pred = direction_pred[:, -1, :]  # Последний временной шаг для классификации
            confidence_pred = confidence_pred[:, -1, :]  # Последний временной шаг для регрессии

            direction_target = targets[:, 0].long()  # Индексы классов (0: long, 1: short, 2: hold)
            confidence_target = targets[:, 1].unsqueeze(1)  # Уверенность как регрессия
            logging.info(f"Model created with input_size: \n{direction_target} \n{confidence_target}")

            # Расчет потерь
            loss_direction = criterion_direction(direction_pred, direction_target)
            loss_confidence = criterion_confidence(confidence_pred, confidence_target)
            loss = loss_direction + loss_confidence

            # np.set_printoptions(suppress=True, precision=10)
            # # Логи промежуточных значений
            # print(f"\nBatch {batch_idx + 1}")
            # print(f"Direction Predictions:\n{direction_pred.softmax(dim=1).detach().cpu().numpy()}")
            # # Логирование предсказаний направления
            # # predicted_classes = torch.argmax(direction_pred, dim=1)
            # # for i, (probs, pred_class, true_class) in enumerate(zip(direction_pred.tolist(), predicted_classes.tolist(), targets[:, 0].tolist())):
            # #     print(f"Sample {i + 1}: Probabilities: {probs}, Predicted Class: {pred_class}, True Class: {int(true_class)}")
            # print(f"Direction Targets:\n{direction_target.detach().cpu().numpy()}")
            # print(f"Confidence Predictions:\n{confidence_pred.detach().cpu().numpy()}")
            # print(f"Confidence Targets:\n{confidence_target.detach().cpu().numpy()}")
            # print(f"Loss (Direction): {loss_direction.item():.14f}")
            # print(f"Loss (Confidence): {loss_confidence.item():.14f}")
            # print(f"Total Loss: {loss.item():.14f}")
            # np.set_printoptions(suppress=False)

            optimizer.zero_grad()
            # Обновление градиентов
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.14f}")

def evaluate_model(model, test_loader, criterion_direction, criterion_confidence):
    """
    Оценка модели на тестовых данных.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")

            try:
                # Прогнозы модели
                direction_pred, confidence_pred = model(X_batch)

                direction_pred = direction_pred[:, -1, :]  # Последний временной шаг для классификации
                confidence_pred = confidence_pred[:, -1, :]  # Последний временной шаг для регрессии

                # Разделение таргетов
                direction_target = y_batch[:, 0].long()  # Индексы классов
                confidence_target = y_batch[:, 1].unsqueeze(1)  # Уверенность
                
                # Расчет потерь
                loss_direction = criterion_direction(direction_pred, direction_target)
                loss_confidence = criterion_confidence(confidence_pred, confidence_target)
                loss = loss_direction + loss_confidence

                total_loss += loss.item()

                # Логгирование форм данных
                # print(f"direction_pred.shape: {direction_pred.shape}, direction_target.shape: {direction_target.shape}")
                # print(f"confidence_pred.shape: {confidence_pred.shape}, confidence_target.shape: {confidence_target.shape}")
            except Exception as e:
                logging.error(f"Ошибка в evaluate_model: {e}")
                continue

    if len(test_loader) > 0:
        average_loss = total_loss / len(test_loader)
    else:
        logging.warning("test_loader пустой")
        average_loss = float('inf')

    print(f"Test Loss: {average_loss:.14f}")
    return average_loss

class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TradingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_direction = nn.Linear(hidden_size, 3)  # 3 класса: long, short, hold
        self.fc3_confidence = nn.Linear(hidden_size, 1)  # Уверенность

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        direction = self.softmax(self.fc3_direction(x))  # Выход для направления
        confidence = self.sigmoid(self.fc3_confidence(x))  # Выход для уверенности

        return direction, confidence

class TradingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.float32)
        self.y = y.clone().detach().to(torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 35000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 1000"
    # query = f"SELECT * FROM order_book_data WHERE id <= '686674' order by id LIMIT 3"
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

def prepare_data_for_orderbook(orderbook_data: pd.DataFrame, sequence_length: int) -> tuple:
    """
    Подготавливает данные для обучения модели, объединяя данные стакана и интервалов.
    :param orderbook_data: Данные стакана.
    :param intervals_predictions: Прогнозы моделей интервалов.
    :param sequence_length: Длина последовательности для стакана.
    :return: Кортеж (X, y) для обучения.
    """
    X = []
    orderbook_data["timestamp"] = pd.to_datetime(orderbook_data["timestamp"])

    first_open_time = orderbook_data.iloc[0]["timestamp"]  # Первый элемент
    last_open_time = orderbook_data.iloc[-1]["timestamp"]  # Последний элемент

    intervals_predictions = get_intervals_predictions(first_open_time, last_open_time)

    for i in range(len(orderbook_data) - sequence_length):
        # Извлечение последовательности стакана
        orderbook_sequence = orderbook_data.iloc[i:i + sequence_length].drop(columns=["id", "timestamp"]).values.astype(np.float32)
        # Последняя временная метка текущей последовательности стакана
        last_timestamp = orderbook_data.iloc[i + sequence_length - 1]["timestamp"]
        # Фильтруем прогнозы по интервалам, даты которых <= last_timestamp
        filtered_predictions = intervals_predictions[intervals_predictions["timestamp"] <= last_timestamp]
        # Если нужны только последние прогнозы по каждому интервалу
        latest_predictions = filtered_predictions.groupby("interval").tail(1)
        # Объединяем данные `prediction` и `trend` для каждого интервала
        combined_features = []
        for _, row in latest_predictions.iterrows():
            combined_row = np.concatenate([row["prediction"],
            [row["trend_price_difference"], row["trend_sma"], row["trend_strength"],
            # row["timestamp"], row['hour'], row['minute'], row['weekday'],
            row["open_price_normalized"], row["high_price_normalized"], row["low_price_normalized"], row["close_price_normalized"],
            row["volume_normalized"],
            row["rsi_normalized"], row["macd_normalized"], row["macd_signal_normalized"], row["macd_hist_normalized"],
            row["sma_20_normalized"], row["ema_20_normalized"],
            row["upper_bb_normalized"], row["middle_bb_normalized"], row["lower_bb_normalized"], row["obv_normalized"]]])
            combined_features.append(combined_row)
        # Преобразуем список в двумерный массив
        combined_features = np.array(combined_features).flatten()
        # Повторяем объединенные данные по количеству строк в `orderbook_sequence`
        repeated_predictions = np.tile(combined_features, (orderbook_sequence.shape[0], 1))
        try:
            # Объединяем массивы
            combined_data = np.hstack((orderbook_sequence, repeated_predictions))
        except ValueError as e:
            print(f"Ошибка при hstack: {e}")
        # print(f"combined_data: {combined_data.shape}")
        X.append(combined_data)

    return np.array(X)

def prepare_data_for_orderbook_non_overlapping(orderbook_data: pd.DataFrame) -> np.ndarray:
    """
    Подготавливает данные для обучения модели, исключая перекрывающиеся строки.
    :param orderbook_data: Данные стакана.
    :return: Массив X для обучения.
    """
    X = []
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
    logging.info(f"Orderbook data fetched: {len(orderbook_data)} rows")

    # # TODO fetching data from db is too busy work
    # # TODO it should be done one time for all trials to economy time of optimization process
    # X = prepare_data_for_orderbook(orderbook_data, sequence_length)
    X = prepare_data_for_orderbook_non_overlapping(orderbook_data)
    logging.info(f"Prepared data shape: {X.shape}")
    # print(f"Shape of X: {np.array(X).shape}")
    # print(f"Sample from X[0]: {X[0]}")
    # print(f"Sample from X[0]: {X[1]}")
    # print(f"Sample from X[0]: {X[2]}")
    # print(f"Sample from X[0]: {X[3]}")
    # print(f"Sample from X[0]: {X[3].shape}")

    agent = DQLAgent(state_size=140, action_size=6)  # Пример с Deep Q-Learning
    env = TradingEnvironment(X)

    episodes = 1000
    target_update_frequency = 10  # Как часто обновлять целевую сеть

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # Выбор действия
            # print(f"statestatestate: {state}")
            action = agent.choose_action(state)

            # Совершаем шаг в среде
            next_state, reward, done = env.step(action)

            # Применяем систему штрафов и наград (если требуется)
            penalty = 0  # Определите вашу логику штрафов
            adjusted_reward = agent.apply_rewards_and_penalties(reward, penalty)

            # Сохраняем опыт в памяти агента
            agent.store_experience(state, action, adjusted_reward, next_state, done)

            # Переходим к следующему состоянию
            state = next_state
            total_reward += adjusted_reward

            # Обучаем агента после каждого шага
            agent.learn()

            if done:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
                break

        # Обновляем целевую сеть каждые `target_update_frequency` эпизодов
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()
            print(f"Target network updated at episode {episode + 1}")

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