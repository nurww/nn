# interval_model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from project_root.data.database_manager import execute_query
import os
import redis.asyncio as aioredis  # Асинхронная версия библиотеки redis-py
import json

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
# IntervalLSTMModel class in interval_model.py
class IntervalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(IntervalLSTMModel, self).__init__()
        self.num_layers = num_layers  # Сохраняем num_layers как атрибут экземпляра
        self.hidden_size = hidden_size  # Сохраняем hidden_size как атрибут экземпляра
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Используем self.num_layers и self.hidden_size
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
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
        y_batch = y_batch.view(-1, 1)  # Преобразуем размер y_batch в [batch_size, 1]
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
            y_batch = y_batch.view(-1, 1)  # Преобразуем размер y_batch в [batch_size, 1]
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


# Предположим, что у вас есть JSON файл "optimized_params.json" с оптимальными параметрами:
# {
#     "1h": {"hidden_size": 132, "num_layers": 3, "dropout": 0.434, "learning_rate": 1.1e-5, "sequence_length": 88, "batch_size": 34},
#     "15m": {"hidden_size": 144, "num_layers": 2, "dropout": 0.29, "learning_rate": 2.5e-5, "sequence_length": 64, "batch_size": 48}
# }
# Основной процесс обучения и сохранения модели
async def main():
    interval = "1h"  # Пример интервала

    # Загрузка оптимизированных гиперпараметров из JSON-файла
    with open("optimized_params.json", "r") as file:
        optimized_params = json.load(file)

    # Извлечение гиперпараметров для конкретного интервала
    params = optimized_params.get(interval)
    if not params:
        raise ValueError(f"No optimized parameters found for interval {interval}")

    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    learning_rate = params["learning_rate"]
    sequence_length = params["sequence_length"]
    batch_size = params["batch_size"]

    # Подготовка данных для выбранного интервала
    data = fetch_interval_data(interval)
    X, y = prepare_data(data, target_column="close_price_normalized", sequence_length=sequence_length)

    # Разделение на обучающую и тестовую выборки
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Настройка загрузчиков данных
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size
    )

    # Создание модели с использованием оптимальных гиперпараметров
    input_size = X.shape[2]
    model = IntervalLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to("cuda")
    
    # Настройка оптимизатора и функции потерь
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    epochs = 15
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, "cuda")
        val_loss = evaluate_model(model, val_loader, criterion, "cuda")
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Сохранение и предсказание, как в вашем коде
    model_filepath = f"saved_models/interval_lstm_model_{interval}.pth"
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    save_model(model, model_filepath)

    # Тестовое предсказание
    loaded_model = load_model(model_filepath, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout)
    test_data = X_val[-1]  # Последняя последовательность валидационных данных
    prediction = predict(loaded_model, test_data, "cuda")
    print(f"Прогноз для последнего батча валидационных данных: {prediction}")

if __name__ == "__main__":
    main()
