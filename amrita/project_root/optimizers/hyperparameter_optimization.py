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

from project_root.models.interval_model import IntervalLSTMModel
from project_root.data.database_manager import execute_query

# Настройка логирования
logging.basicConfig(
    filename=f'../logs/hyperparameter_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def fetch_interval_data(interval: str) -> pd.DataFrame:
    logging.info(f"Fetching interval data for {interval}")
    query = f"SELECT * FROM binance_klines_normalized WHERE `data_interval` = '{interval}'"
    data = execute_query(query)
    if data.empty:
        logging.warning(f"No data found for interval {interval}")
    else:
        logging.info(f"Columns in fetched data: {data.columns.tolist()}")
    return data

def prepare_data(data: pd.DataFrame, target_column: str, sequence_length: int):
    logging.info(f"Preparing data with sequence length {sequence_length}")

    # Нормализация данных
    min_value = data[target_column].min()
    max_value = data[target_column].max()
    # data[target_column] = (data[target_column] - min_value) / (max_value - min_value)
    
    logging.info(f"Target column (normalized) min: {min_value}, max: {max_value}")
    logging.info(f"First 5 normalized values: {data[target_column].head().values}")
    logging.info(f"Last 5 normalized values: {data[target_column].tail().values}")

    # Формируем X, y и временные метки
    features = data.drop(columns=["open_time", "close_time", "data_interval", "window_id", target_column]).values.astype(np.float32)
    targets = data[target_column].values.astype(np.float32)
    open_times = data["open_time"].values
    close_times = data["close_time"].values

    X, y, times = [], [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])
        times.append((open_times[i + sequence_length], close_times[i + sequence_length]))

    logging.info(f"Prepared data - Sequence length: {sequence_length}, Total samples: {len(X)}")
    return np.array(X), np.array(y), np.array(times), min_value, max_value

def denormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value

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

def objective(trial):
    wait_for_safe_time()
    logging.info("Starting a new trial")
    
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5) if num_layers > 1 else 0.0
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    sequence_length = trial.suggest_int("sequence_length", 30, 100)
    batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    logging.info(f"Trial parameters - hidden_size: {hidden_size}, num_layers: {num_layers}, dropout: {dropout}, learning_rate: {learning_rate}, sequence_length: {sequence_length}, batch_size: {batch_size}")

    interval = "1d"  # пример интервала
    data = fetch_interval_data(interval)
    if data.empty:
        logging.warning("No data available, skipping trial.")
        return float("inf")

    X, y, times, min_value, max_value = prepare_data(data, target_column="close_price_normalized", sequence_length=sequence_length)

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    times_train, times_val = times[:train_size], times[train_size:]

    # Проверки индексов и данных
    logging.info(f"Train size: {train_size}, Validation size: {len(y_val)}")
    logging.info(f"First 5 training targets: {y_train[:5]}")
    logging.info(f"First 5 validation targets: {y_val[:5]}")
    # Логируем первые 5 временных меток из валидации
    for i in range(5):
        logging.info(f"Validation times[{i}]: Open: {times_val[i][0]}, Close: {times_val[i][1]}, y_val: {y_val[i]}")

    # Логи для данных перед разбиением
    logging.info(f"First 5 rows of data before splitting: {data.head()}")
    logging.info(f"Last 5 rows of data before splitting: {data.tail()}")

    # Логи для min/max из MySQL
    mysql_min = data["close_price_normalized"].min()
    mysql_max = data["close_price_normalized"].max()
    logging.info(f"MySQL Min: {mysql_min}, MySQL Max: {mysql_max}")

    # Логи для min/max в коде
    logging.info(f"Code Min: {min_value}, Code Max: {max_value}")

    # Логи для соответствия индексов
    for i in range(5):
        logging.info(f"Index {i}: y_val={y_val[i]}, base_close={data.iloc[train_size + i]['close_price_normalized']}")

    # Тест нормализации и денормализации
    test_value = 0.5
    denormalized = denormalize(test_value, min_value, max_value)
    renormalized = (denormalized - min_value) / (max_value - min_value)
    logging.info(f"Test Value: {test_value}, Denormalized: {denormalized}, Renormalized: {renormalized}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)

    input_size = X.shape[2]
    model = IntervalLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to("cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        logging.info(f"Epoch {epoch + 1} completed, Train Loss: {epoch_loss / len(train_loader):.4f}")

        # Пример прогноза после каждой эпохи
        model.eval()
        with torch.no_grad():
            sample_data = torch.tensor(X_val[-5:], dtype=torch.float32).to("cuda")
            predictions = model(sample_data).cpu().numpy()
            denormalized_preds = [denormalize(pred[0], min_value, max_value) for pred in predictions]
            denormalized_targets = [denormalize(value, min_value, max_value) for value in y_val[:5]]
            
            # Проверка нормализации
            for i, value in enumerate(y_val[:5]):
                denormalized_value = denormalize(value, min_value, max_value)
                base_value = data.iloc[train_size + i]["close_price_normalized"]
                logging.info(f"y_val[{i}]: {value}, Denormalized: {denormalized_value}, Base close_price_normalized: {base_value}")

            # Создаем DataFrame для вывода в лог
            results_df = pd.DataFrame({
                "open_time": [time[0] for time in times_val[:5]],
                "close_time": [time[1] for time in times_val[:5]],
                "predicted_close": [denormalize(pred[0], min_value, max_value) for pred in predictions],
                "actual_close": [denormalize(value, min_value, max_value) for value in y_val[:5]],
                "base_close": [data.iloc[train_size + i]["close_price_normalized"] for i in range(5)]
            })

            logging.info(f"Results DataFrame:\n{results_df}")


        logging.info(f"Epoch {epoch + 1}, Sample Predictions:\n{results_df}")

    # Остальной код без изменений...



    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            y_batch = y_batch.view(-1, 1)
            y_pred = model(X_batch)
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
