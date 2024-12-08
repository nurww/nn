# hyperparameter_cup_optimization.py

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import asyncio
from datetime import datetime, timedelta
from reward_system import calculate_reward
from amrita.project_root.data.database_manager import execute_query
import redis.asyncio as aioredis
import logging
import time
from datetime import datetime
import optuna

# Конфигурация Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Интервалы для моделей
INTERVALS = ["1d", "4h", "1h", "15m", "5m", "1m"]

# Настройка логирования
logging.basicConfig(
    filename=f'../logs/hyperparameter_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Класс для взаимодействия с Redis
class RedisManager:
    def __init__(self):
        self.redis_client = None
    
    async def connect(self):
        self.redis_client = await aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/{REDIS_CONFIG['db']}")

    async def get_last_timestamp(self, stream_key="order_book_stream"):
        last_entry = await self.redis_client.lindex(stream_key, -1)
        if last_entry:
            last_entry_data = json.loads(last_entry)
            return datetime.strptime(last_entry_data['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
        print("Нет данных в order_book_stream")
        return None

    async def set_interval_signal(self, interval, signal):
        await self.redis_client.set(f"interval_signal_{interval}", signal)

    async def fetch_data(self, redis_key: str, limit: int = 1500):
        data = await self.redis_client.lrange(redis_key, -limit, -1)
        return [json.loads(item) for item in data]

    async def close(self):
        await self.redis_client.close()

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

async def load_interval_model(interval):
    model_path = f"saved_models/interval_lstm_model_{interval}.pth"
    with open("optimized_params.json", "r") as file:
        params = json.load(file)[interval]
    model = IntervalLSTMModel(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        output_size=1,
        dropout=params["dropout"]
    ).to("cuda")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_data(orderbook_data: np.array, sequence_length: int):
    X, y = [], []
    for i in range(len(orderbook_data) - sequence_length):
        X.append(orderbook_data[i:i + sequence_length])
        y.append(orderbook_data[i + sequence_length]['price_bid'])
    return np.array(X), np.array(y)

def fetch_last_sequence_from_mysql(interval: str, sequence_length: int, end_time: datetime) -> pd.DataFrame:
    query = f"""
        SELECT * FROM binance_klines_normalized
        WHERE data_interval = '{interval}' AND open_time <= '{end_time}'
        ORDER BY open_time DESC
        LIMIT {sequence_length}
    """
    data = execute_query(query)
    if not data.empty:
        data = data.iloc[::-1]
    return data

async def predict_and_save_interval_signal(redis_manager, interval, model, sequence_length):
    last_timestamp = await redis_manager.get_last_timestamp()
    if last_timestamp is None:
        return

    interval_data = fetch_last_sequence_from_mysql(interval, sequence_length, last_timestamp)
    if interval_data.empty:
        print(f"Нет данных для интервала {interval} до времени {last_timestamp}")
        return

    interval_data_tensor = torch.tensor(interval_data.values).float().unsqueeze(0).to("cuda")
    prediction = model(interval_data_tensor)
    interval_signal = int(torch.sign(prediction.item()))
    await redis_manager.set_interval_signal(interval, interval_signal)

async def synchronize_and_train(redis_manager, orderbook_model, criterion, optimizer, train_loader, val_loader):
    interval_models = {interval: await load_interval_model(interval) for interval in INTERVALS}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(10):
        for interval, model in interval_models.items():
            await predict_and_save_interval_signal(redis_manager, interval, model, sequence_length=60)

        train_loss = train_model(orderbook_model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(orderbook_model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        await asyncio.sleep(60)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def fetch_interval_data(interval: str) -> pd.DataFrame:
    logging.info(f"Fetching interval data for {interval}")
    query = f"SELECT * FROM binance_klines_normalized WHERE `data_interval` = '{interval}' order by open_time"
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for interval {interval}")
    # else:
        # logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    return data

def fetch_orderbook_data() -> pd.DataFrame:
    logging.info(f"Fetching data for orderbook")
    query = f"SELECT * FROM order_book_data order by id"
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for orderbook")
    # else:
        # logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    return data

def objective(trial):
    logging.info("Starting a new trial")
    
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    sequence_length = trial.suggest_int("sequence_length", 30, 100)
    batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    orderbook_data = fetch_orderbook_data()
    predictions = get_intervals_predictions()

    X, y = prepare_data(orderbook_data, sequence_length)

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)
    input_size = X.shape[2]

    # Инициализация модели и оптимизатора
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orderbook_model = OrderBookGRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(orderbook_model.parameters(), lr=learning_rate)

    # await synchronize_and_train(redis_manager, orderbook_model, criterion, optimizer, train_loader, val_loader)

    return None

# Функция для загрузки обученной модели интервала
def load_interval_model(interval: str, params: dict) -> IntervalLSTMModel:
    model_path = f"saved_models/interval_lstm_model_{interval}.pth"
    model = IntervalLSTMModel(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        output_size=params["output_size"],
        dropout=params["dropout"]
    ).to("cuda")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Функция для получения последних данных интервала
def fetch_interval_data(interval: str, sequence_length: int, timestamp: datetime) -> pd.DataFrame:
    query = f"""
        SELECT * FROM binance_klines_normalized
        WHERE data_interval = '{interval}' AND open_time <= '{timestamp}'
        ORDER BY open_time DESC
        LIMIT {sequence_length}
    """
    data = execute_query(query)
    if not data.empty:
        data = data.iloc[::-1]  # Реверсируем, чтобы данные были в хронологическом порядке
    return data

# Функция для получения прогнозов интервалов
def get_intervals_predictions(timestamp: datetime) -> pd.DataFrame:
    logging.info("Fetching predictions for intervals")
    
    # Загружаем параметры из JSON
    with open("optimized_params.json", "r") as file:
        optimized_params = json.load(file)
    
    intervals = optimized_params.keys()  # Например: ["1m", "5m", "15m", "1h"]
    predictions = {}

    for interval in intervals:
        # Загружаем модель
        params = optimized_params[interval]
        model = load_interval_model(interval, params)
        
        # Загружаем последние данные для интервала
        sequence_length = params["sequence_length"]
        data = fetch_interval_data(interval, sequence_length, timestamp)
        
        if data.empty or len(data) < sequence_length:
            logging.warning(f"Not enough data for interval {interval}")
            continue
        
        # Преобразуем данные для подачи в модель
        features = data.drop(columns=["id", "open_time", "close_time", "data_interval"]).values
        input_data = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to("cuda")  # Добавляем размер батча
        
        # Получаем прогноз
        with torch.no_grad():
            prediction = model(input_data).cpu().numpy().flatten()  # Преобразуем в массив
            predictions[interval] = prediction

    # Преобразуем результаты в DataFrame
    results = pd.DataFrame(predictions, index=[timestamp])
    return results

def fetch_orderbook_data(limit: int = 1500) -> pd.DataFrame:
    logging.info("Fetching data for orderbook")
    query = f"SELECT * FROM order_book_data ORDER BY timestamp DESC LIMIT {limit}"
    data = execute_query(query)
    if not data.empty:
        data = data.iloc[::-1]  # Реверсируем, чтобы данные были в хронологическом порядке
    return data

def align_data(orderbook_data: pd.DataFrame, intervals_predictions: pd.DataFrame) -> pd.DataFrame:
    # Преобразуем временные метки в единый формат
    orderbook_data["timestamp"] = pd.to_datetime(orderbook_data["timestamp"])
    intervals_predictions.index = pd.to_datetime(intervals_predictions.index)

    # Объединяем по времени
    merged_data = pd.merge_asof(
        orderbook_data,
        intervals_predictions,
        left_on="timestamp",
        right_index=True,
        direction="backward"  # Берем ближайшие предсказания назад по времени
    )
    return merged_data

async def main():
    logging.info("Starting hyperparameter optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # 50 испытаний для оптимизации
    print("Лучшие гиперпараметры:", study.best_params)
    print("Лучшее значение потерь:", study.best_value)
    logging.info(f"Optimization completed - Best Params: {study.best_params}, Best Loss: {study.best_value:.4f}")

    # Шаг 1: Загрузить данные ордербука
    orderbook_data = fetch_orderbook_data()
    if orderbook_data.empty:
        logging.warning("No orderbook data available")
        return
    
    # Шаг 2: Получить прогнозы интервалов
    last_timestamp = orderbook_data["timestamp"].iloc[-1]
    intervals_predictions = get_intervals_predictions(last_timestamp)
    if intervals_predictions.empty:
        logging.warning("No interval predictions available")
        return

    # Шаг 3: Согласовать данные
    aligned_data = align_data(orderbook_data, intervals_predictions)

    # Шаг 4: Использовать согласованные данные для тренировки
    print(aligned_data.head())

if __name__ == "__main__":
    asyncio.run(main())
