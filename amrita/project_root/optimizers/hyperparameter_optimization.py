# hyperparameter_optimization.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
import pandas as pd
import os
import sys
import logging
import time
from datetime import datetime

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from amrita.project_root.models.interval_model import IntervalLSTMModel
from amrita.project_root.data.database_manager import execute_query

# Настройка логирования
logging.basicConfig(
    filename=f'../logs/hyperparameter_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def wait_for_safe_time():
    """Ждет до тех пор, пока текущее время не достигнет безопасного интервала."""
    while True:
        now = datetime.now()
        # Ждем, пока секунды не достигнут или не превысят 44
        if now.second >= 44:
            print(f"Запуск в безопасное время: {now.strftime('%H:%M:%S')}")
            break
        # Пауза перед следующей проверкой, чтобы не загружать процессор
        time.sleep(0.5)

def denormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value

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
    logging.info("Merging large interval data with small interval features.")
    
    # Приводим данные из small_data к формату, сопоставимому с data
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

def prepare_data(data: pd.DataFrame, target_columns: list, sequence_length: int):
    logging.info(f"Preparing data with sequence length {sequence_length}")

    # Получаем копию данных, чтобы не повлиять на исходный DataFrame
    data = data.copy()

    # Определяем список колонок для удаления
    columns_to_drop = [
        "id", "open_time", "close_time", "data_interval", "window_id",
        "next_open_time", "next_close_time", "small_open_time", "small_close_time",
        "small_low_price", "small_high_price", "small_open_price", "small_close_price",
        "small_volume"
    ] + target_columns

    # Убираем только существующие колонки
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns=columns_to_drop).values.astype(np.float32)

    # Настраиваем вывод numpy, чтобы показывать все данные строки
    # np.set_printoptions(suppress=True, precision=8, threshold=np.inf, linewidth=np.inf)
    # logging.info(f"First 5 rows: \n{features[:5]}")
    # logging.info(f"Last 5 rows: \n{features[-5:]}")
    # # Восстанавливаем стандартные настройки вывода numpy
    # np.set_printoptions(suppress=False, precision=8, threshold=1000, linewidth=75)

    # Формируем X, y и временные метки
    X, y = [], []

    for i in range(len(features) - sequence_length):
        # Формируем последовательность признаков
        X_sequence = features[i:i + sequence_length]

        # Формируем последовательность целевых значений для предсказания
        y_target = data[target_columns].iloc[i + sequence_length].values.astype(np.float32)

        # Преобразуем y_targets в одномерный массив
        # y_targets = y_targets.flatten()

        X.append(X_sequence)
        y.append(y_target)

    logging.info(f"Prepared data - Sequence length: {sequence_length}, Total samples: {len(X)}")
    return np.array(X), np.array(y)

def objective(trial):
    # wait_for_safe_time()
    logging.info("Starting a new trial")
    
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    sequence_length = trial.suggest_int("sequence_length", 30, 100)
    batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    target_columns = ["open_price_normalized", "close_price_normalized", 
                      "low_price_normalized", "high_price_normalized"]

    logging.info(f"Trial parameters - hidden_size: {hidden_size}, num_layers: {num_layers}, dropout: {dropout}, learning_rate: {learning_rate}, sequence_length: {sequence_length}, batch_size: {batch_size}")

    interval = "1d"  # пример интервала
    small_interval = "4h"  # пример интервала

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

    # logging.info(f"Final data: \n{final_data}")

    # X, y, times, min_value, max_value = prepare_data(final_data, target_column="close_price_normalized", sequence_length=sequence_length)
    X, y = prepare_data(final_data, target_columns=target_columns, sequence_length=sequence_length)

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    # times_train, times_val = times[:train_size], times[train_size:]

    # Проверки индексов и данных
    logging.info(f"Train size: {train_size}, Validation size: {len(y_val)}")
    # logging.info(f"First 5 training targets: {y_train[:5]}")
    # logging.info(f"First 5 validation targets: {y_val[:5]}")
    # Логируем первые 5 временных меток из валидации
    # for i in range(5):
        # logging.info(f"Validation times[{i}]: Open: {times_val[i][0]}, Close: {times_val[i][1]}, y_val: {y_val[i]}")

    # Логи для данных перед разбиением
    # logging.info(f"First 5 rows of data before splitting: {data.head()}")
    # logging.info(f"Last 5 rows of data before splitting: {data.tail()}")

    # Логи для min/max из MySQL
    # mysql_min = data["close_price_normalized"].min()
    # mysql_max = data["close_price_normalized"].max()
    # logging.info(f"MySQL Min: {mysql_min}, MySQL Max: {mysql_max}")

    # Логи для min/max в коде
    # logging.info(f"Code Min: {min_value}, Code Max: {max_value}")

    # Логи для соответствия индексов
    # for i in range(5):
    #     logging.info(f"Index {i}: y_val={y_val[i]}, base_close={data.iloc[train_size + i]['close_price_normalized']}")

    # Тест нормализации и денормализации
    # test_value = 0.5
    # denormalized = denormalize(test_value, min_value, max_value)
    # renormalized = (denormalized - min_value) / (max_value - min_value)
    # logging.info(f"Test Value: {test_value}, Denormalized: {denormalized}, Renormalized: {renormalized}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)

    input_size = X.shape[2]
    output_size = len(target_columns)
    model = IntervalLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout).to("cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = 10
    batch_index = 0  # Счетчик батча

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            # Индексы текущего батча в тренировочных данных
            # batch_start = batch_index * batch_size
            # batch_end = batch_start + len(X_batch)

            # Основной цикл обучения
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            # y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            y_pred = model(X_batch)
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
        logging.info(f"Epoch {epoch + 1} completed, Train Loss: {epoch_loss / len(train_loader):.4f}")


        # Пример прогноза после каждой эпохи
        model.eval()
        with torch.no_grad():
            # Логируем последние 5 строки тренировочного набора
            train_sample_data = X_train[-5:]  # Последние 5 строк из X_train
            train_targets = y_train[-5:]  # Последние 5 целевых значений из y_train

            train_sample_data_tensor = torch.tensor(train_sample_data, dtype=torch.float32).to("cuda")
            train_predictions = model(train_sample_data_tensor).cpu().numpy()

            # Логируем предсказания и целевые значения
            logging.info(f"Predictions on last 5 rows of training set: {train_predictions}")
            logging.info(f"Actual targets on last 5 rows of training set: {train_targets}")

            train_results_df = pd.DataFrame({
                "Predicted_open": train_predictions[:, 0],
                "Actual_open": train_targets[:, 0],
                "Predicted_close": train_predictions[:, 1],
                "Actual_close": train_targets[:, 1],
                "Predicted_low": train_predictions[:, 2],
                "Actual_low": train_targets[:, 2],
                "Predicted_high": train_predictions[:, 3],
                "Actual_high": train_targets[:, 3],
            })

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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            # y_batch = y_batch.view(-1, 1)
            y_pred = model(X_batch)

            # Логируем предсказания и целевые значения для первых 5 примеров в батче
            # logging.info(f"Validation predictions: {y_pred.detach().cpu().numpy()[:5]}")
            # logging.info(f"Validation targets: {y_batch.detach().cpu().numpy()[:5]}")

            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Loss for trial: {avg_val_loss:.4f}")
    return avg_val_loss

def main():
    logging.info("Starting hyperparameter optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # 50 испытаний для оптимизации
    print("Лучшие гиперпараметры:", study.best_params)
    print("Лучшее значение потерь:", study.best_value)
    logging.info(f"Optimization completed - Best Params: {study.best_params}, Best Loss: {study.best_value:.4f}")

if __name__ == "__main__":
    main()
