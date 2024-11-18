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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from amrita.project_root.models.interval_model import IntervalLSTMModel
from amrita.project_root.data.database_manager import execute_query

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
    min_value, max_value = data[target_column].min(), data[target_column].max()

    logging.info(f"Available columns: {data.columns.tolist()}")
    try:
        logging.info(f"Sample data from target column {target_column}: {data[target_column].head()}")
    except KeyError:
        logging.error(f"Column {target_column} not found in data.")
        raise KeyError(f"Column '{target_column}' not found in data.")

    columns_to_drop = ["open_time", "close_time", "data_interval", target_column]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns=columns_to_drop).values.astype(np.float32)
    targets = data[target_column].values.astype(np.float32)

    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])

    # Добавьте проверку данных для последних строк, чтобы убедиться в их корректности
    logging.info("Checking last 5 entries of the dataset for verification:")
    logging.info(data[["id", "open_time", "close_time", target_column]].tail(5))

    min_value = 0
    max_value = 1
    return np.array(X), np.array(y), min_value, max_value


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

    X, y, min_value, max_value = prepare_data(data, target_column="close_price_normalized", sequence_length=sequence_length)
    
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
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
            denormalized_targets = [denormalize(y_val[i], min_value, max_value) for i in range(-5, 0)]
            
            # Создаем DataFrame для вывода в лог и проверяем, что значения корректны
            results_df = pd.DataFrame({
            "id": data["id"].values[train_size:train_size+5],
            "open_time": data["open_time"].values[train_size:train_size+5],
            "close_time": data["close_time"].values[train_size:train_size+5],
            "predicted_close": [denormalize(pred, min_value, max_value) for pred in denormalized_preds],
            "actual_close": [denormalize(target, min_value, max_value) for target in y_val[:5]],
            "base_close": data["close_price_normalized"].values[train_size:train_size+5]
        })
        logging.info(f"Epoch {epoch + 1}, Sample Predictions:\n{results_df}")


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
