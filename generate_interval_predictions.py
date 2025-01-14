# generate_interval_predictions.py

import sys
import os
import json
import logging
from datetime import datetime, timedelta
import torch
import numpy as np
import redis.asyncio as aioredis
import pandas as pd
import asyncio
import torch
import torch.nn as nn

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from amrita.project_root.data.database_manager import execute_query

# Настройки Redis
REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'db': 0}

# Настройка логирования
logging.basicConfig(
    filename='amrita/project_root/logs/interval_predictions.log',
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

async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

def load_interval_model(interval, params, input_size):
    """Загружает модель для указанного интервала."""
    model_path = f"amrita/project_root/models/saved_models/interval_lstm_model_{interval}.pth"
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

    # if interval not in freq_map:
    #     raise ValueError(f"Unsupported interval: {interval}")

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

    # Проверяем, получены ли данные после агрегации
    if aggregated.empty:
        logging.warning(f"Aggregation resulted in empty data for interval: {interval}. Skipping step.")
        return pd.DataFrame()
    
    # Shift small interval data for прогнозирования следующего интервала
    aggregated['next_open_time'] = aggregated['open_time'].shift(-1)
    aggregated['next_close_time'] = aggregated['close_time'].shift(-1)
    aggregated['next_low_price'] = aggregated['low_price'].shift(-1)
    aggregated['next_high_price'] = aggregated['high_price'].shift(-1)
    aggregated['next_open_price'] = aggregated['open_price'].shift(-1)
    aggregated['next_close_price'] = aggregated['close_price'].shift(-1)
    aggregated['next_volume'] = aggregated['volume'].shift(-1)
    columns_to_display = [
        "open_time", "next_open_time",
        "open_price_normalized", "next_open_price",
        "close_time", "next_close_time"
    ]
    # if all(col in aggregated.columns for col in columns_to_display):  # Проверяем наличие столбцов
    #     pd.set_option('display.max_rows', None)  # Показывать все строки
    #     pd.set_option('display.max_columns', None)  # Показывать все столбцы
    #     pd.set_option('display.expand_frame_repr', False)  # Не переносить DataFrame на несколько строк
    #     pd.set_option('display.max_colwidth', None)  # Показывать полные значения в ячейках
    #     print("Merged data preview (selected columns):")
    #     print(f"aggregate_small_data {len(aggregated)}")
    #     print(f"aggregate_small_data {aggregated[columns_to_display]}")
    
    # print(f"aggregate_small_data {aggregated.head(3)}")

    # Сбрасываем индекс и переименовываем его
    aggregated.reset_index(drop=True, inplace=True)

    logging.info(f"Aggregated small data: {aggregated.shape[0]} rows for interval {interval}.")
    return aggregated

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

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
    # merged_data = pd.merge_asof(
    #     data.sort_values('open_time'),
    #     small_data.sort_values('small_open_time'),
    #     left_on='open_time',
    #     right_on='small_open_time',
    #     direction='backward'
    # )
    
    print("start ______________________")
    # Перед объединением
    print("Large interval data range:", data["open_time"].min(), "-", data["open_time"].max())
    print("Small interval data range:", small_data["small_open_time"].min(), "-", small_data["small_open_time"].max())

    # После объединения
    # print("Merged data preview (all columns):")
    # print(merged_data.tail(5))

    # Отображаем только определенные столбцы
    columns_to_display = [
        "open_time", "next_open_time",
        "open_price_normalized", "next_open_price",
        "close_time", "next_close_time"
    ]
    # if all(col in merged_data.columns for col in columns_to_display):  # Проверяем наличие столбцов
    #     pd.set_option('display.max_rows', None)  # Показывать все строки
    #     pd.set_option('display.max_columns', None)  # Показывать все столбцы
    #     pd.set_option('display.expand_frame_repr', False)  # Не переносить DataFrame на несколько строк
    #     pd.set_option('display.max_colwidth', None)  # Показывать полные значения в ячейках
    #     print("Merged data preview (selected columns):")
    #     # print(merged_data[columns_to_display].head(5))
    #     print(merged_data[columns_to_display])
    # else:
    #     logging.warning("Some selected columns are missing in merged data.")
    print("end ______________________")
    logging.info(f"Merged data shape: {merged_data.shape}")
    return merged_data

def calculate_required_rows_for_small_interval(sequence_length: int, large_interval: str, small_interval: str) -> int:
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

    if not small_interval:
        return 0  # Если младший интервал отсутствует, данные не нужны
    
    multiplier = MULTIPLIER.get((large_interval, small_interval), 1)
    required_rows = sequence_length * multiplier + multiplier   # Количество строк для малого интервала
    return required_rows

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

def fetch_interval_data(interval: str, sequence_length: int) -> pd.DataFrame:
    query = f"""
        SELECT * FROM binance_klines_normalized
        WHERE data_interval = '{interval}'
        ORDER BY open_time DESC
        LIMIT {sequence_length}
    """
    data = execute_query(query)
    if not data.empty:
        data = data.iloc[::-1]  # Реверсируем, чтобы данные были в хронологическом порядке
    return data

def fetch_small_interval_data(interval: str, required_rows) -> pd.DataFrame:
    logging.info(f"Fetching small interval data for {interval}")
    query = f"""SELECT open_time, open_price, high_price, low_price, close_price, close_time, volume
              FROM binance_klines WHERE `data_interval` = '{interval}'
              order by open_time desc LIMIT {required_rows}"""
    
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for small interval {interval}")
    
    if not data.empty:
        data = data.iloc[::-1]  # Обратная сортировка по возрастанию времени

    return data

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

def prepare_interval_data(interval: str, sequence_length) -> pd.DataFrame:
    logging.info(f"Fetching data for interval: {interval}")

    trend_window = 5
    sequence_length = sequence_length + trend_window

    window_data = get_active_window(interval)
    small_interval = get_small_interval(interval)
    if small_interval is not None:
        sequence_length = sequence_length + 1
    data = fetch_interval_data(interval, sequence_length)
    
    required_rows = calculate_required_rows_for_small_interval(sequence_length, interval, small_interval)

    if small_interval is not None:
        small_data = fetch_small_interval_data(small_interval, required_rows)
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
        # if aggregated_small_data.empty:
        #     logging.info(f"Skipping further processing for interval: {interval} due to lack of data.")
        #     return pd.DataFrame() # Пропускаем текущий шаг
        normalized_small_data = normalize_small_data(aggregated_small_data, window_data)
        final_data = merge_large_and_small_data(data, normalized_small_data)
        final_data = final_data.drop(final_data.index[-1])
    else:
        final_data = data.copy()
    
    # Добавляем тренды
    final_data["trend_price_difference"] = calculate_trend_price_difference(final_data)
    final_data["trend_sma"] = calculate_trend_sma(final_data, trend_window)
    final_data["trend_strength"] = calculate_trend_strength(final_data)
    final_data = final_data.drop(final_data.index[:5])

    return final_data

async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

async def save_predictions_to_redis(redis_client, interval: str, predictions: pd.DataFrame):
    """Сохраняет прогнозы в Redis."""
    key = f"interval_predictions_{interval}"
    try:
        # Преобразуем DataFrame в JSON-строки
        predictions_json = [row.to_json() for _, row in predictions.iterrows()]
        
        # Используем pipeline для записи
        async with redis_client.pipeline() as pipe:
            for prediction_json in predictions_json:
                await pipe.rpush(key, prediction_json)
            # Ограничиваем список до max_records записей
            await pipe.ltrim(key, -1, -1)
            await pipe.execute()
        logging.info(f"Прогнозы для {interval} успешно сохранены в Redis.")
    except Exception as e:
        logging.error(f"Ошибка при записи прогнозов в Redis для {interval}: {e}")

async def get_intervals_predictions() -> pd.DataFrame:
    logging.info("Fetching synchronized predictions for intervals")
    
    # Загружаем параметры из JSON
    with open("amrita/project_root/models/optimized_params.json", "r") as file:
        optimized_params = json.load(file)
    
    intervals = optimized_params.keys()  # Например: ["1d", "4h", "1h", "15m", "5m", "1m"]
    all_predictions = []

    # Настройка расписания с временными окнами
    # schedule = {
    #     "1d": lambda now: now.hour % 4 == 0 and now.minute == 0 and now.second >= 0,    # Каждые 4 часа в начале часа
    #     "4h": lambda now: now.minute == 0 and now.second >= 0,                          # Каждый час в начале часа
    #     "1h": lambda now: now.minute % 15 == 0 and now.second >= 0,                     # Каждые 15 минут
    #     "15m": lambda now: now.minute % 5 == 0 and now.second >= 0,                     # Каждые 5 минут
    #     "5m": lambda now: now.second >= 0,                                              # Каждую минуту
    #     "1m": lambda now: now.second >= 0                                               # Каждую минуту
    # }

    redis_client = await initialize_redis()

    for interval in intervals:
        print(f"Interval: {interval}")
        now = datetime.utcnow()

        # # Проверяем, следует ли запускать прогнозы для текущего интервала
        # if interval in schedule and not schedule[interval](now):
        #     logging.info(f"Пропускаем прогнозы для интервала {interval} (не время для выполнения).")
        #     print(f"Пропускаем прогнозы для интервала {interval} (не время для выполнения).")
        #     continue

        # Загружаем модель

        # if interval != '15m':
        #     continue

        params = optimized_params[interval]
        sequence_length = params["sequence_length"]

        interval_data = prepare_interval_data(interval, sequence_length)
        # print(f"len(aggregated): {interval} {len(interval_data)}")
        # print(f"aggregated: {interval} {interval_data.head(5)}")
        if interval_data.empty:
            continue

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
            # print(f"Timestamp: {interval} {timestamp}")

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
            predictions_df = pd.DataFrame(predictions)
            await save_predictions_to_redis(redis_client, interval, predictions_df)
            all_predictions.append(predictions_df)
        
        await read_predictions_from_redis(redis_client, interval)

    await redis_client.aclose()

    # Чтение прогнозов из Redis
async def read_predictions_from_redis(redis_client, interval: str):
    """Считывает прогнозы из Redis."""
    key = f"interval_predictions_{interval}"
    try:
        predictions = await redis_client.lrange(key, 0, -1)
        if predictions:
            # Преобразуем JSON-строки в DataFrame
            data = [json.loads(prediction) for prediction in predictions]
            df = pd.DataFrame(data)
            print(f"Прогнозы для {interval}:\n{df}")
            logging.info(f"Прогнозы для {interval} успешно считаны из Redis.")
        else:
            print(f"Нет данных для {interval} в Redis.")
    except Exception as e:
        logging.error(f"Ошибка при чтении прогнозов из Redis для {interval}: {e}")

if __name__ == "__main__":
    print("\n")
    print("_____________# generate_interval_predictions.py")
    asyncio.run(get_intervals_predictions())
    print("\n")
    print("__________________________||||||||")