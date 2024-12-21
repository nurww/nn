# interval_model.py

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
import redis.asyncio as aioredis  # Асинхронная версия библиотеки redis-py
import json
import logging
import time
from datetime import datetime

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)
from project_root.data.database_manager import execute_query

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Настройка логирования
logging.basicConfig(
    filename=f'../logs/model_interval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

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

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def normalize_small_data(small_data: pd.DataFrame, window_data: pd.DataFrame) -> pd.DataFrame:
    
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
    
    return small_data

def fetch_interval_data(interval: str) -> pd.DataFrame:
    logging.info(f"Fetching interval data for {interval}")
    query = f"SELECT * FROM binance_klines_normalized WHERE `data_interval` = '{interval}' order by open_time"
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for interval {interval}")
    # else:
        # logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    return data

def fetch_small_interval_data(interval: str) -> pd.DataFrame:
    logging.info(f"Fetching small interval data for {interval}")
    query = f"SELECT open_time, open_price, high_price, low_price, close_price, close_time, volume FROM binance_klines WHERE `data_interval` = '{interval}' order by open_time"
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for interval {interval}")
    # else:
        # logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    return data

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
    # else:
        # logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    return data

def merge_large_and_small_data(data: pd.DataFrame, small_data: pd.DataFrame) -> pd.DataFrame:
    """Объединение большого и меньшего интервалов."""
    small_data.rename(columns={
        'open_time': 'small_open_time',
        'open_price': 'small_open_price',
        'high_price': 'small_high_price',
        'low_price': 'small_low_price',
        'close_price': 'small_close_price',
        'close_time': 'small_close_time',
        'volume': 'small_volume'
    }, inplace=True)
    merged_data = pd.merge(data, small_data, left_on='open_time', right_on='small_open_time', how='left')
    return merged_data

def prepare_data(data: pd.DataFrame, target_columns: list, sequence_length: int):

    # Получаем копию данных, чтобы не повлиять на исходный DataFrame
    data = data.copy()

    # Определяем список колонок для удаления
    columns_to_drop = [
        "id", "open_time", "close_time", "data_interval", "window_id",
        "next_open_time", "next_close_time", "small_open_time", "small_close_time",
        "small_low_price", "small_high_price", "small_open_price", "small_close_price",
        "small_volume"
    ]

    # Убираем только существующие колонки
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns=columns_to_drop).values.astype(np.float32)

    X, y = [], []

    for i in range(len(features) - sequence_length):
        # Формируем последовательность признаков
        X_sequence = features[i:i + sequence_length]

        # Формируем последовательность целевых значений для предсказания
        y_target = data[target_columns].iloc[i + sequence_length].values.astype(np.float32)

        X.append(X_sequence)
        y.append(y_target)

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
    logging.info(f"Модель сохранена по пути: {filepath}")

def load_model(filepath, input_size, hidden_size, num_layers, output_size, dropout=0.2):
    """Загружает модель с диска."""
    model = IntervalLSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    print(f"Модель загружена из файла: {filepath}")
    logging.info(f"Модель загружена из файла: {filepath}")
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
def main():
    interval = "1m"  # Пример интервала
    small_interval = None

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

    target_columns = ["open_price_normalized", "close_price_normalized", 
                      "low_price_normalized", "high_price_normalized"]

    # Подготовка данных для выбранного интервала
    data = fetch_interval_data(interval)
    window_data = get_active_window(interval)

    if small_interval is not None:
        small_data = fetch_small_interval_data(small_interval)
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
    else:
        final_data = data.copy()

    X, y = prepare_data(final_data, target_columns=target_columns, sequence_length=sequence_length)

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
    output_size = len(target_columns)
    model = IntervalLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout).to("cuda")
    
    # Настройка оптимизатора и функции потерь
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    epochs = 15
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, "cuda")
        val_loss = evaluate_model(model, val_loader, criterion, "cuda")
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.14f}, Validation Loss: {val_loss:.14f}")
        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.14f}, Validation Loss: {val_loss:.14f}")

    # Сохранение и предсказание, как в вашем коде
    model_filepath = f"saved_models/interval_lstm_model_{interval}.pth"
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    save_model(model, model_filepath)

    # Тестовое предсказание
    loaded_model = load_model(model_filepath, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout)
    if len(X_val) == 0:
        logging.error("Validation set is empty. Skipping prediction.")
        return

    test_data = X_val[-1]  # Последняя последовательность валидационных данных
    prediction = predict(loaded_model, test_data, "cuda")
    print(f"Прогноз для последнего батча валидационных данных: {prediction}")
    logging.info(f"Прогноз для последнего батча валидационных данных: {prediction}")
    # denormalized_prediction = denormalize(prediction[0][0], min_value, max_value)
    # print(f"Денормализованный прогноз: {denormalized_prediction}")

if __name__ == "__main__":
    # asyncio.run(main())
    main()
