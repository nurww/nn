# interval_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from project_root.data.database_manager import execute_query
import os
import redis.asyncio as aioredis  # Асинхронная версия библиотеки redis-py

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Настройка асинхронного подключения к Redis
async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

# Функция для сохранения сигнала от модели интервалов в Redis
async def save_interval_signal(redis_client, interval_signal):
    """
    Сохраняет предсказание от модели интервалов в Redis.
    
    :param redis_client: клиент Redis
    :param interval_signal: сигнал от модели интервалов (например, +1 для восходящего тренда, -1 для нисходящего)
    """
    await redis_client.set("interval_signal", interval_signal)

# Модель LSTM для анализа интервалов
class IntervalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(IntervalLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Выход для последнего таймстепа
        return out

def fetch_interval_data(interval: str) -> pd.DataFrame:
    """Загрузка данных по интервалу из базы данных."""
    query = f"SELECT * FROM binance_klines_normalized WHERE interval = '{interval}'"
    data = execute_query(query)
    return data

def prepare_data(data: pd.DataFrame, target_column: str, sequence_length: int):
    """Подготовка данных для обучения модели."""
    features = data.drop(columns=[target_column]).values
    targets = data[target_column].values

    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    
    return np.array(X), np.array(y)

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

def save_model(model, filepath):
    """Сохраняет обученную модель на диск."""
    torch.save(model.state_dict(), filepath)
    print(f"Модель сохранена по пути: {filepath}")

def load_model(filepath, input_size, hidden_size, num_layers, output_size, dropout=0.2):
    """Загружает модель с диска."""
    model = IntervalLSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Модель загружена из файла: {filepath}")
    return model

def predict(model, data, device):
    """Выполняет предсказание на основе обученной модели."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32).to(device)
        prediction = model(data.unsqueeze(0))  # Добавляем размер батча
        return prediction.cpu().numpy()

# Основной процесс обучения и сохранения модели
async def main():
    interval = "1h"  # Пример интервала
    data = fetch_interval_data(interval)
    X, y = prepare_data(data, target_column="close_price_normalized", sequence_length=60)
    
    # Настройка загрузчиков данных
    batch_size = 64
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)
    
    # Настройки модели
    input_size = X.shape[2]
    model = IntervalLSTMModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.3).to("cuda")
    
    # Настройка оптимизатора и функции потерь
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение модели
    epochs = 10
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, "cuda")
        val_loss = evaluate_model(model, val_loader, criterion, "cuda")
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Сохранение модели
    model_filepath = "saved_models/interval_lstm_model.pth"
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    save_model(model, model_filepath)
    
    # Загрузка и тестовое предсказание
    loaded_model = load_model(model_filepath, input_size=input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.3)
    test_data = X_val[-1]  # Последняя последовательность валидационных данных
    prediction = predict(loaded_model, test_data, "cuda")
    print(f"Прогноз для последнего батча валидационных данных: {prediction}")

    redis_client = await initialize_redis()
    try:
        # Предположим, что `interval_signal` генерируется вашей моделью интервалов
        interval_signal = 1  # Например, восходящий тренд
        await save_interval_signal(redis_client, interval_signal)
    finally:
        await redis_client.close()

if __name__ == "__main__":
    main()
