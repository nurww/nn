# retrospective_training.py

import redis
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime, timedelta

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Определение модели GRU (для разнообразия)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Используем выход последнего таймстепа
        return out

# Функция для извлечения прогнозов и фактических данных из Redis
def fetch_data_from_redis(redis_client, prediction_key, actual_key):
    predictions = redis_client.lrange(prediction_key, -1000, -1)
    actuals = redis_client.lrange(actual_key, -1000, -1)
    
    if predictions and actuals:
        predictions = [json.loads(item) for item in predictions]
        actuals = [json.loads(item) for item in actuals]
    
    return predictions, actuals

# Функция для подготовки данных к обучению
def prepare_data_for_training(predictions, actuals):
    # Создаем массивы для признаков и меток
    features = []
    targets = []

    for pred, actual in zip(predictions, actuals):
        features.append([
            pred['mid_price'],
            pred['sum_bid_volume'],
            pred['sum_ask_volume'],
            pred['imbalance']
        ])
        targets.append(actual['mid_price'])  # Пример: можно изменять на более сложные метрики

    # Преобразование в массивы NumPy
    X = np.array(features)
    y = np.array(targets)

    return X, y

# Функция для инкрементального дообучения модели
def retrain_model(model, X, y):
    # Преобразование данных в тензоры
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [samples, timesteps, features]
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [samples, 1]

    # Создание TensorDataset и DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Настройки для обучения
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Инкрементальное обучение
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Средняя ошибка после ретроспективного обучения: {loss.item():.4f}")

# Основная функция для ретроспективного обучения
def main():
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )

    # Извлечение данных
    predictions, actuals = fetch_data_from_redis(redis_client, 'model_predictions', 'actual_results')

    if not predictions or not actuals:
        print("Недостаточно данных для ретроспективного обучения.")
        return

    # Подготовка данных
    X, y = prepare_data_for_training(predictions, actuals)

    # Инициализация модели
    input_size = X.shape[1]
    hidden_size = 50
    num_layers = 2
    output_size = 1

    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load('lstm_model.pth'))  # Загрузка обученной модели

    # Ретроспективное обучение
    retrain_model(model, X, y)

    # Сохранение обновленной модели
    torch.save(model.state_dict(), 'updated_model.pth')
    print("Модель обновлена и сохранена!")

if __name__ == "__main__":
    main()
