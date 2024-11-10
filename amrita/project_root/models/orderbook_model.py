# orderbook_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import redis
import os
import json
from reward_system import calculate_reward  # Импортируем функцию вознаграждения
import redis.asyncio as aioredis
import time
import traceback
import asyncio

# Конфигурация Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Подключение к Redis для получения сигнала интервалов
async def get_interval_signal(redis_client):
    interval_signal = await redis_client.get("interval_signal")
    return int(interval_signal) if interval_signal is not None else 0

def connect_to_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}/{REDIS_CONFIG['db']}")

# Подключение к Redis для получения сигнала интервалов
async def get_interval_signal(redis_client):
    interval_signal = await redis_client.get("interval_signal")
    return int(interval_signal) if interval_signal is not None else 0

# Определение модели GRU
class OrderBookGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(OrderBookGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])  # Выход для последнего таймстепа
        return out

# Получение данных стакана из Redis
def fetch_orderbook_data(redis_key: str, limit: int = 1500):
    redis_conn = connect_to_redis()
    data = redis_conn.lrange(redis_key, -limit, -1)
    orderbook = [json.loads(item) for item in data]
    return np.array(orderbook)

def prepare_data(orderbook_data: np.array, sequence_length: int):
    """Преобразует данные стакана в последовательности для обучения."""
    X, y = [], []
    for i in range(len(orderbook_data) - sequence_length):
        X.append(orderbook_data[i:i + sequence_length])
        # Пример: используем изменение цены bid как целевой признак
        y.append(orderbook_data[i + sequence_length]['price_bid'])
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

# Основной процесс обучения и оценки с системой поощрений
# Основной процесс обучения и оценки с системой поощрений
async def main():
    redis_key = "orderbook_data"
    sequence_length = 60
    input_size = 5  # Количество фичей (price_bid, quantity_bid, price_ask, quantity_ask, spread)
    hidden_size = 128
    num_layers = 2
    output_size = 1
    dropout = 0.3
    batch_size = 64

    # Подготовка данных
    orderbook_data = fetch_orderbook_data(redis_key)
    X, y = prepare_data(orderbook_data, sequence_length=sequence_length)
    
    # Разделение данных на обучающую и тестовую выборки
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Создание загрузчиков данных
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)
    
    # Настройки модели
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OrderBookGRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Подключение к Redis для получения сигнала интервалов
    redis_client = await connect_to_redis()

    # Обучение модели с системой поощрений
    epochs = 10
    for epoch in range(epochs):
        try:
            start_time = time.time()
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_model(model, val_loader, criterion, device)
            end_time = time.time()
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time Taken: {end_time - start_time:.2f} seconds")
            
            # Пример расчета вознаграждения на этапе валидации
            interval_signal = await get_interval_signal(redis_client)  # Получаем сигнал интервала
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                predictions = model(X_val_batch)

                for i in range(len(predictions)):
                    pred_value = predictions[i].item()
                    actual_value = y_val_batch[i].item()
                    interval_agreement = (interval_signal > 0 and pred_value > 0) or (interval_signal < 0 and pred_value < 0)
                    reward = calculate_reward(pred_value, actual_value)
                    if interval_agreement:
                        reward += 0.5
                    print(f"Prediction: {pred_value}, Actual: {actual_value}, Reward: {reward}, Interval Agreement: {interval_agreement}")
        except Exception as e:
            print(f"Error in epoch {epoch + 1}: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)  # Wait before retrying the next epoch

    # Сохранение модели
    model_filepath = "saved_models/orderbook_gru_model.pth"
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    save_model(model, model_filepath)

    await redis_client.close()

import torch
from models.orderbook_model import OrderBookGRUModel
from utils.model_utils import save_model, load_model

# Конфигурация теста
input_size = 5
hidden_size = 128
num_layers = 2
output_size = 1
sequence_length = 60
device = "cuda" if torch.cuda.is_available() else "cpu"

# Тестовая функция для модели
def test_orderbook_model():
    model = OrderBookGRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
    
    # Генерация случайных данных
    X_test = torch.randn(1, sequence_length, input_size).to(device)
    output = model(X_test)
    
    # Сохранение и загрузка модели
    save_model(model, "test_orderbook_model.pth")
    loaded_model = load_model(model, "test_orderbook_model.pth", device)
    
    # Предсказание с загруженной моделью
    prediction = loaded_model(X_test)
    print("Test prediction:", prediction.item())

if __name__ == "__main__":
    test_orderbook_model()