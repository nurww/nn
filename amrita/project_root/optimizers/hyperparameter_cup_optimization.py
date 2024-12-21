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
# from reward_system import calculate_reward
import redis.asyncio as aioredis
import logging
import time
from datetime import datetime
import optuna

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from project_root.data.database_manager import execute_query

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
    filename=f'../logs/hyperparameter_cup_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

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
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.14f}, Validation Loss: {val_loss:.14f}")

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

def fetch_orderbook_data() -> pd.DataFrame:
    logging.info(f"Fetching data for orderbook")
    # query = f"SELECT * FROM order_book_data order by id LIMIT 380000"
    query = f"SELECT * FROM order_book_data order by id LIMIT 300000"
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
    # else:
        # logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    
    if not data.empty:
        data = data.iloc[::-1]  # Обратная сортировка по возрастанию времени

    return data

def get_intervals_predictions(first_open_time, last_open_time) -> pd.DataFrame:
    logging.info("Fetching synchronized predictions for intervals")
    
    # Загружаем параметры из JSON
    with open("../models/optimized_params.json", "r") as file:
        optimized_params = json.load(file)
    
    intervals = optimized_params.keys()  # Например: ["1m", "5m", "15m", "1h"]
    all_predictions = []

    for interval in intervals:
        # Загружаем модель
        params = optimized_params[interval]
        sequence_length = params["sequence_length"]

        interval_data = prepare_interval_data(interval, sequence_length, first_open_time, last_open_time)

        if interval == "1m":
            columns_to_drop = ["id", "open_time", "close_time", "data_interval", "window_id",
                           "next_open_time", "next_close_time", "small_open_time", "small_close_time",
                           "small_low_price", "small_high_price", "small_open_price", "small_close_price",
                           "small_volume"]
            timestamps = interval_data["open_time"]  # Используем `open_time` для интервала "1m"
        else:
            columns_to_drop = ["id", "open_time", "close_time", "data_interval", "window_id",
                           "next_open_time", "next_close_time", "small_open_time", "small_close_time",
                           "small_low_price", "small_high_price", "small_open_price", "small_close_price",
                           "small_volume", "open_price_normalized", "close_price_normalized",
                           "low_price_normalized", "high_price_normalized"]
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
                "prediction": prediction
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

    # print(f"small_data before aggregation: {len(small_data)}")
    # print(f"small_data: {small_data}")

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
    # print(f"Aggregated small data for interval: {interval}")
    # print(aggregated[:5])
    # print(len(aggregated))
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
    freq_map = {
        "1d": "1D",
        "4h": "1D",
        "1h": "4h",
        "15m": "1h",
        "5m": "15min",
        "1m": "5min"
    }

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

def process_interval_pair(
    high_interval: str, low_interval: str, 
    high_sequence_length: int, low_sequence_length: int, 
    last_timestamp: datetime
) -> pd.DataFrame:
    """
    Обрабатывает пару интервалов: старший + младший.
    :param high_interval: Старший интервал (например, '1d').
    :param low_interval: Младший интервал (например, '4h').
    :param high_sequence_length: Длина последовательности для старшего интервала.
    :param low_sequence_length: Длина последовательности для младшего интервала.
    :param last_timestamp: Последний временной штамп для синхронизации.
    :return: DataFrame с объединенными данными и прогнозами.
    """
    # 1. Получаем нормализованные данные старшего интервала
    high_data_query = f"""
        SELECT * FROM binance_klines_normalized
        WHERE data_interval = '{high_interval}'
        AND open_time <= '{last_timestamp}'
        ORDER BY open_time DESC
        LIMIT {high_sequence_length};
    """
    high_data = execute_query(high_data_query)
    if high_data.empty:
        raise ValueError(f"No data found for high interval {high_interval}")
    high_data = high_data.iloc[::-1]  # Сортировка по времени

    # 2. Получаем ненормализованные данные младшего интервала
    low_data_query = f"""
        SELECT open_time, open_price, high_price, low_price, close_price, volume
        FROM binance_klines
        WHERE data_interval = '{low_interval}'
        AND open_time <= '{last_timestamp}'
        ORDER BY open_time DESC
        LIMIT {low_sequence_length};
    """
    low_data = execute_query(low_data_query)
    if low_data.empty:
        raise ValueError(f"No data found for low interval {low_interval}")
    low_data = low_data.iloc[::-1]  # Сортировка по времени

    # 3. Синхронизация данных
    combined_data = pd.merge_asof(
        low_data.sort_values("open_time"),
        high_data.sort_values("open_time"),
        on="open_time",
        direction="backward"
    )

    # 4. Прогон данных через модели
    combined_features = combined_data.drop(columns=["open_time"]).values.astype(np.float32)

    # Загружаем модели для интервалов
    high_model = load_interval_model(high_interval, params={"sequence_length": high_sequence_length}, input_size=combined_features.shape[1])
    low_model = load_interval_model(low_interval, params={"sequence_length": low_sequence_length}, input_size=combined_features.shape[1])

    # Прогноз для старшего интервала
    high_input = torch.tensor(combined_features[:high_sequence_length], dtype=torch.float32).unsqueeze(0).to("cuda")
    high_prediction = high_model(high_input).cpu().detach().numpy()

    # Прогноз для младшего интервала
    low_input = torch.tensor(combined_features[-low_sequence_length:], dtype=torch.float32).unsqueeze(0).to("cuda")
    low_prediction = low_model(low_input).cpu().detach().numpy()

    # Возвращаем объединенные данные и прогнозы
    combined_data["high_prediction"] = high_prediction.flatten()
    combined_data["low_prediction"] = low_prediction.flatten()
    return combined_data

# def process_all_intervals(last_timestamp: datetime) -> pd.DataFrame:
#     """
#     Обрабатывает все пары интервалов и объединяет результаты.
#     :param last_timestamp: Последний временной штамп.
#     :return: DataFrame с прогнозами для всех пар интервалов.
#     """
#     interval_pairs = [("1d", "4h"), ("4h", "1h"), ("1h", "15m"), ("15m", "5m"), ("5m", "1m")]
#     results = []

#     for high_interval, low_interval in interval_pairs:
#         logging.info(f"Processing interval pair: {high_interval} -> {low_interval}")
#         params = load_optimized_params()  # Загружаем параметры моделей
#         high_sequence_length = params[high_interval]["sequence_length"]
#         low_sequence_length = params[low_interval]["sequence_length"]

#         # Обработка пары интервалов
#         result = process_interval_pair(high_interval, low_interval, high_sequence_length, low_sequence_length, last_timestamp)
#         results.append(result)

#     # Объединение всех данных
#     all_data = pd.concat(results, ignore_index=True)
#     return all_data

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

def prepare_interval_data(interval: str, sequence_length, first_open_time: datetime, last_open_time: datetime) -> pd.DataFrame:
    logging.info(f"Fetching data for interval: {interval}, first_open_time: {first_open_time} and last_open_time: {last_open_time}")

    # print("Before adjusting:")
    # print(f"{interval} {first_open_time}")
    # print(f"{interval} {last_open_time}")
    # print("\n")
    first_open_time = adjust_interval_timestamp(interval, first_open_time)
    last_open_time = adjust_interval_timestamp(interval, last_open_time)
    # print("After adjusting:")
    # print(f"{interval} {first_open_time}")
    # print(f"{interval} {last_open_time}")
    # print("\n")
    interval_sequence = calculate_rows_for_interval(interval, sequence_length, first_open_time, last_open_time)
    # print(f"{interval} {interval_sequence} {sequence_length}")

    data = fetch_interval_data(interval, interval_sequence, last_open_time)
    window_data = get_active_window(interval)
    small_interval = get_small_interval(interval)

    required_rows = calculate_required_rows_for_small_interval(interval_sequence, interval)
    # print(f"required_rowsrequired_rowsrequired_rows: {required_rows}")

    if small_interval is not None:
        small_data = fetch_small_interval_data(small_interval, last_open_time, required_rows)
        if small_data.empty:
            logging.warning(f"No small interval data available for interval {small_interval}. Proceeding without it.")
            small_data = None
    else:
        small_data = None

    # print(f"Data: {data[:5]}")

    if data.empty:
        logging.warning("No data available, skipping trial.")
        return float("inf")
    
    if small_data is not None:
        aggregated_small_data = aggregate_small_data(small_data, small_interval)
    # logging.info(f"Aggregated data: \n{aggregated_small_data}")
        normalized_small_data = normalize_small_data(aggregated_small_data, window_data)
        # print(f"Small Data: {normalized_small_data[:5]}")
    # logging.info(f"Normalized data: \n{normalized_small_data}")
        final_data = merge_large_and_small_data(data, normalized_small_data)
        final_data = final_data.drop(final_data.index[-1])
    else:
        final_data = data.copy()
    
    # print(f"Interval data for interval: {interval}")
    # print(final_data[:5])
    # print(final_data[5:])
    # print(f"Final data {interval}: {final_data['open_time']}")
    # logging.info(f"Final data: {final_data[:5]}")
    # print(f"Len final_data: {len(final_data)}")

    # if print(normalized_small_data):
    #     print(normalized_small_data)
    # print(f"last_open_time: {last_open_time}")
    return final_data

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

def fetch_interval_sequence(interval: str, sequence_length: int, timestamp: datetime) -> pd.DataFrame:
    """
    Получение последовательности данных для указанного интервала.
    :param interval: Интервал данных (например, '1d', '1h').
    :param sequence_length: Длина последовательности.
    :param timestamp: Временная метка последней записи стакана.
    :return: Данные интервала за sequence_length записей.
    """
    adjusted_timestamp = adjust_interval_timestamp(interval, timestamp)
    query = f"""
        SELECT * FROM binance_klines_normalized
        WHERE data_interval = '{interval}' AND open_time <= '{adjusted_timestamp}'
        ORDER BY open_time DESC
        LIMIT {sequence_length}
    """
    data = execute_query(query)
    if not data.empty:
        data = data.iloc[::-1]  # Реверсируем, чтобы данные были в хронологическом порядке
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

def prepare_data(orderbook_data: pd.DataFrame, sequence_length: int) -> tuple:
    """
    Подготавливает данные для обучения модели, объединяя данные стакана и интервалов.
    :param orderbook_data: Данные стакана.
    :param intervals_predictions: Прогнозы моделей интервалов.
    :param sequence_length: Длина последовательности для стакана.
    :return: Кортеж (X, y) для обучения.
    """
    X, y = [], []
    orderbook_data["timestamp"] = pd.to_datetime(orderbook_data["timestamp"])

    first_open_time = orderbook_data.iloc[0]["timestamp"]  # Первый элемент
    last_open_time = orderbook_data.iloc[-1]["timestamp"]  # Последний элемент

    # print("Первый timestamp:", first_open_time)
    # print("Последний timestamp:", last_open_time)

    intervals_predictions = get_intervals_predictions(first_open_time, last_open_time)
    # print(f"intervals_predictions: \n{intervals_predictions}")
    # intervals_predictions.index = pd.to_datetime(intervals_predictions.index)

    for i in range(len(orderbook_data) - sequence_length):
        # Извлечение последовательности стакана
        orderbook_sequence = orderbook_data.iloc[i:i + sequence_length].drop(columns=["id", "timestamp"]).values.astype(np.float32)
        # last_timestamp = timestamps.iloc[sequence_length - 1]
        # orderbook_sequence = orderbook_sequence.drop(columns=["id", "timestamp"]).values.astype(np.float32)

        # Последняя временная метка текущей последовательности стакана
        last_timestamp = orderbook_data.iloc[i + sequence_length - 1]["timestamp"]
        # print(f"last_timestamp: {last_timestamp}")

        # Фильтруем прогнозы по интервалам, даты которых <= last_timestamp
        filtered_predictions = intervals_predictions[intervals_predictions["timestamp"] <= last_timestamp]

        # Если нужны только последние прогнозы по каждому интервалу
        latest_predictions = filtered_predictions.groupby("interval").tail(1)

        # Объединяем все прогнозы в одну строку
        combined_predictions = np.concatenate(latest_predictions["prediction"].values)

        # Преобразуем `combined_predictions` в двумерный массив
        combined_predictions = combined_predictions.reshape(1, -1)

        # Диагностика orderbook_sequence
        # print(f"orderbook_sequence shape: {orderbook_sequence.shape}")
        # print(f"orderbook_sequence sample:\n{orderbook_sequence[:5]}")

        # Диагностика combined_predictions
        # print(f"combined_predictions shape: {combined_predictions.shape}")
        # print(f"combined_predictions:\n{combined_predictions}")

        # Повторяем `combined_predictions` по количеству строк в `orderbook_sequence`
        repeated_predictions = np.tile(combined_predictions, (orderbook_sequence.shape[0], 1))

        # Добавляем прогнозы как новую строку к данным стакана
        # combined_data = np.hstack((orderbook_sequence, combined_predictions))
        # Проверка их совместимости для hstack
        try:
            # Объединяем массивы
            combined_data = np.hstack((orderbook_sequence, repeated_predictions))
            # print("Combined data shape (test):", combined_data.shape)
            # print(f"combined_predictions:\n{combined_predictions}")
            # print(f"orderbook_sequence sample:\n{orderbook_sequence[:5]}")
            # print(f"combined_data:\n{combined_data}")

            # logging.info(f"combined_predictions:\n{combined_predictions}")
            # logging.info(f"combined_predictions shape:\n{combined_predictions.shape}")
            # logging.info(f"orderbook_sequence sample:\n{orderbook_sequence[:5]}")
            # logging.info(f"orderbook_sequence sample shape:\n{orderbook_sequence.shape}")
            # logging.info(f"combined_data:\n{combined_data[:5]}")
            # logging.info(f"combined_data shape:\n{combined_data.shape}")
            # logging.info("\n")

            # np.savetxt("combined_predictions.txt", combined_predictions, delimiter=",")
            # np.savetxt("orderbook_sequence.txt", orderbook_sequence, delimiter=",")
            # np.savetxt("combined_data.txt", combined_data, delimiter=",")
            # print(intervals_predictions)

            # combined_predictions_str = "\n".join([",".join(map(str, row)) for row in combined_predictions])
            # orderbook_sequence_str = "\n".join([",".join(map(str, row)) for row in orderbook_sequence])
            # combined_data_str = "\n".join([",".join(map(str, row)) for row in combined_data])

            # # Открываем файлы в режиме "a" (append) для добавления новых данных
            # with open("combined_predictions.txt", "a") as f_pred:
            #     f_pred.write(combined_predictions_str + "\n")

            # with open("orderbook_sequence.txt", "a") as f_seq:
            #     f_seq.write(orderbook_sequence_str + "\n")

            # with open("combined_data.txt", "a") as f_data:
            #     f_data.write(combined_data_str + "\n")
        except ValueError as e:
            print(f"Ошибка при hstack: {e}")
        # print(f"combined_data: {combined_data}")
        # print(f"combined_data shape: {combined_data.shape}")
        
        # print("Shape orderbook_sequence:", orderbook_sequence.shape)
        # print("Shape intervals_predictions:", intervals_predictions.shape)

        # Извлечение прогнозируемых `low_price_normalized` и `high_price_normalized`
        # Последние два значения в combined_predictions (исходя из описания)
        normalized_low_price = combined_predictions[0, -2]
        normalized_high_price = combined_predictions[0, -1]

        # # Целевое значение: mid_price и volume для следующего шага
        target_mid_price = orderbook_data.iloc[i + sequence_length]["mid_price"]
        target_bid_volume = orderbook_data.iloc[i + sequence_length]["sum_bid_volume"]
        target_ask_volume = orderbook_data.iloc[i + sequence_length]["sum_ask_volume"]
        target_bid_ask_imbalance = orderbook_data.iloc[i + sequence_length]["bid_ask_imbalance"]
        # Таргет: волатильность (разница между max и min ценой)
        target_volatility = normalized_high_price - normalized_low_price

        # Добавляем в target оба значения
        target = [target_mid_price,
                target_bid_volume,
                target_ask_volume,
                target_bid_ask_imbalance,
                normalized_low_price,
                normalized_high_price,
                target_volatility]

        X.append(combined_data)
        y.append(target)
    return np.array(X), np.array(y)

def objective(trial):
    logging.info("Starting a new trial")
    
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    sequence_length = trial.suggest_int("sequence_length", 30, 100)
    batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    orderbook_data = fetch_orderbook_data()
    
    logging.info(f"Trial parameters - hidden_size: {hidden_size}, num_layers: {num_layers}, dropout: {dropout}, learning_rate: {learning_rate}, sequence_length: {sequence_length}, batch_size: {batch_size}")
    
    # prepare_data(orderbook_data, sequence_length)

    # combined_data = align_data(orderbook_data, intervals_predictions)

    # Шаг 4: Обучение модели с объединенными данными
    X, y = prepare_data(orderbook_data, sequence_length)
    logging.info(f"Prepared data for training. Features: {X.shape}, Targets: {y.shape}")
    # print(f"Prepared data for training. Features: {X.shape}, Targets: {y.shape}")

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)
    input_size = X.shape[2]
    output_size = 7

    # Инициализация модели и оптимизатора
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orderbook_model = OrderBookGRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(orderbook_model.parameters(), lr=learning_rate)

    epochs = 10
    batch_index = 0  # Счетчик батча

    for epoch in range(epochs):
        orderbook_model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            # Индексы текущего батча в тренировочных данных
            # batch_start = batch_index * batch_size
            # batch_end = batch_start + len(X_batch)

            # Основной цикл обучения
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            # y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            y_pred = orderbook_model(X_batch)
            loss = criterion(y_pred, y_batch)

            # if epoch == epochs - 1:  # Для последней эпохи
            #     logging.info(f"Last 5 rows from training set before batch prediction: {X_train[-5:]}")
                # logging.info(f"Last 5 targets from training set before batch prediction: {y_train[-5:]}")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Извлекаем оригинальные данные из X_train и y_train
            # original_X_batch = X_train[batch_start:batch_end, -1][-1]
            # original_y_batch = y_train[batch_start:batch_end][-1]
            # last_pred = y_pred[-1].item()  # Последнее предсказание из модели
            # last_target = y_batch[-1].item()  # Последняя цель
            # Логируем для проверки
            # logging.info(f"Batch index: {batch_index}")
            # logging.info(f"Original X_batch from X_train:{original_X_batch}")
            # logging.info(f"Original y_batch from y_train: {original_y_batch}")
            # logging.info(f"Model prediction for last sequence in batch: {last_pred}")
            # logging.info(f"Target for last sequence in batch: {last_target}")

            # Увеличиваем счетчик батча
            batch_index += 1



        # Сброс счетчика батча для следующей эпохи
        batch_index = 0
        logging.info(f"Epoch {epoch + 1} completed, Train Loss: {epoch_loss / len(train_loader):.14f}")


        # Пример прогноза после каждой эпохи
        orderbook_model.eval()
        with torch.no_grad():
            # Логируем последние 5 строки тренировочного набора
            train_sample_data = X_train[-5:]  # Последние 5 строк из X_train
            train_targets = y_train[-5:]  # Последние 5 целевых значений из y_train

            train_sample_data_tensor = torch.tensor(train_sample_data, dtype=torch.float32).to("cuda")
            train_predictions = orderbook_model(train_sample_data_tensor).cpu().numpy()

            # Логируем предсказания и целевые значения
            logging.info(f"Predictions on last 5 rows of training set: {train_predictions}")
            logging.info(f"Actual targets on last 5 rows of training set: {train_targets}")

            train_results_df = pd.DataFrame({
                "Predicted_mid_price": train_predictions[:, 0],
                "Actual_mid_price": train_targets[:, 0],
                "Predicted_sum_bid_volume": train_predictions[:, 1],
                "Actual_sum_bid_volume": train_targets[:, 1],
                "Predicted_sum_ask_volume": train_predictions[:, 2],
                "Actual_sum_ask_volume": train_targets[:, 2],
                "Predicted_bid_ask_imbalance": train_predictions[:, 3],
                "Actual_bid_ask_imbalance": train_targets[:, 3],
                "Predicted_normalized_low_price": train_predictions[:, 4],
                "Actual_normalized_low_price": train_targets[:, 4],
                "Predicted_normalized_high_price": train_predictions[:, 5],
                "Actual_normalized_high_price": train_targets[:, 5],
                "Predicted_target_volatility": train_predictions[:, 6],
                "Actual_target_volatility": train_targets[:, 6],
            })

            # Сохраняем DataFrame в CSV-файл
            # train_results_df.to_csv("train_results.csv", mode="a", index=False, header=not os.path.exists("train_results.csv"))

            filename = "train_results.csv"
            if os.path.exists(filename):
                add_null_row_to_csv(filename)

            # Сохраняем DataFrame в CSV-файл
            train_results_df.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename))
            # train_results_cup_data_str = "\n".join([",".join(map(str, row)) for row in train_results_df])

            # # Открываем файлы в режиме "a" (append) для добавления новых данных
            # with open("train_results_cup_data.txt", "a") as f_pred:
            #     f_pred.write(train_results_cup_data_str + "\n")

            logging.info(f"Results DataFrame for last 5 rows:\n{train_results_df}")
            # Логируем последние 5 строк из тренировочной выборки
            # train_sample_data = X_train[-5:]  # Последние 5 строк из X_train
            # train_times = times_train[-5:]  # Последние 5 временных меток из times_train
            # train_targets = y_train[-5:]  # Последние 5 целевых значений из y_train

            # batch_sample_data = X_batch[-5:]  # Последние 5 строк из X_train
            # batch_targets = y_batch[-5:]  # Последние 5 целевых значений из y_train

            # Если нужно предсказать на тренировочных данных (для проверки)
            # train_sample_data_tensor = torch.tensor(train_sample_data, dtype=torch.float32).to("cuda")
            # train_predictions = model(train_sample_data_tensor).cpu().numpy()

            # Денормализация предсказаний и фактических значений
            # denormalized_train_preds = [denormalize(pred[0], min_value, max_value) for pred in train_predictions]
            # denormalized_train_targets = [denormalize(value, min_value, max_value) for value in train_targets]

            # Формируем DataFrame для последних 5 строк тренировочной выборки
            # train_results_df = pd.DataFrame({
            #     "open_time": [time[0] for time in train_times],
            #     "close_time": [time[1] for time in train_times],
            #     # "batch_sample_data": [time[0] for time in batch_sample_data],
            #     # "batch_targets": [time[1] for time in batch_targets],
            #     "predicted_close": denormalized_train_preds,
            #     "actual_close": denormalized_train_targets
            # })
            # logging.info(f"Results DataFrame for Epoch {epoch + 1}:\n{train_results_df}")

    # Остальной код без изменений...

    orderbook_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            # y_batch = y_batch.view(-1, 1)
            y_pred = orderbook_model(X_batch)

            # Логируем предсказания и целевые значения для первых 5 примеров в батче
            # logging.info(f"Validation predictions: {y_pred.detach().cpu().numpy()[:5]}")
            # logging.info(f"Validation targets: {y_batch.detach().cpu().numpy()[:5]}")

            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Validation Loss: {avg_val_loss:.14f}")
    print(f"Validation Loss for trial: {avg_val_loss:.14f}")
    return avg_val_loss

# Убедимся, что границы можно легко увидеть
def add_null_row_to_csv(filename):
    null_row = pd.DataFrame([["NULL"] * 14])  # 14 соответствует количеству колонок в DataFrame
    null_row.to_csv(filename, mode="a", index=False, header=False)

async def main():
    logging.info("Starting hyperparameter optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=35)  # 50 испытаний для оптимизации
    print("Лучшие гиперпараметры:", study.best_params)
    print("Лучшее значение потерь:", study.best_value)
    logging.info(f"Optimization completed - Best Params: {study.best_params}, Best Loss: {study.best_value:.14f}")

if __name__ == "__main__":
    asyncio.run(main())
