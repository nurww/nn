# hyperparameter_cup_optimization.py

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
import optuna

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from project_root.data.database_manager import execute_query

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
    
    if not data.empty:
        data = data.iloc[::-1]  # Обратная сортировка по возрастанию времени

    return data

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

def prepare_interval_data(interval: str, sequence_length, first_open_time: datetime, last_open_time: datetime) -> pd.DataFrame:
    logging.info(f"Fetching data for interval: {interval}, first_open_time: {first_open_time} and last_open_time: {last_open_time}")

    first_open_time = adjust_interval_timestamp(interval, first_open_time)
    last_open_time = adjust_interval_timestamp(interval, last_open_time)
    
    interval_sequence = calculate_rows_for_interval(interval, sequence_length, first_open_time, last_open_time)

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

    intervals_predictions = get_intervals_predictions(first_open_time, last_open_time)

    for i in range(len(orderbook_data) - sequence_length):
        # Извлечение последовательности стакана
        orderbook_sequence = orderbook_data.iloc[i:i + sequence_length].drop(columns=["id", "timestamp"]).values.astype(np.float32)
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
        # Повторяем `combined_predictions` по количеству строк в `orderbook_sequence`
        repeated_predictions = np.tile(combined_predictions, (orderbook_sequence.shape[0], 1))

        try:
            # Объединяем массивы
            combined_data = np.hstack((orderbook_sequence, repeated_predictions))

        except ValueError as e:
            print(f"Ошибка при hstack: {e}")

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
            # Основной цикл обучения
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            optimizer.zero_grad()
            y_pred = orderbook_model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

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
            logging.info(f"Results DataFrame for last 5 rows:\n{train_results_df}")

    orderbook_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            y_pred = orderbook_model(X_batch)
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

def main():
    logging.info("Starting hyperparameter optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=35)  # 50 испытаний для оптимизации
    print("Лучшие гиперпараметры:", study.best_params)
    print("Лучшее значение потерь:", study.best_value)
    logging.info(f"Optimization completed - Best Params: {study.best_params}, Best Loss: {study.best_value:.14f}")

if __name__ == "__main__":
    main()
