# orderbook_model.py

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

# Конфигурация Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Интервалы для моделей
INTERVALS = ["5m", "15m", "1h", "4h"]

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

async def main():
    redis_manager = RedisManager()
    await redis_manager.connect()

    # Подготовка данных
    sequence_length = 60
    orderbook_data = await redis_manager.fetch_data("orderbook_data")
    X, y = prepare_data(orderbook_data, sequence_length)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=64)

    # Инициализация модели и оптимизатора
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orderbook_model = OrderBookGRUModel(input_size=5, hidden_size=128, num_layers=2, output_size=1, dropout=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(orderbook_model.parameters(), lr=0.001)

    await synchronize_and_train(redis_manager, orderbook_model, criterion, optimizer, train_loader, val_loader)
    await redis_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
