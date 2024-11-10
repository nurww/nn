# hyperparameter_optimization.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from models.interval_model import IntervalLSTMModel
from project_root.data.database_manager import execute_query
import pandas as pd

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

def objective(trial):
    # Определение гиперпараметров с использованием Optuna
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    sequence_length = trial.suggest_int("sequence_length", 30, 100)

    interval = "1h"  # пример интервала
    data = fetch_interval_data(interval)
    X, y = prepare_data(data, target_column="close_price_normalized", sequence_length=sequence_length)
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Настройка загрузчиков данных
    batch_size = 64
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)
    
    # Определение модели и настроек
    input_size = X.shape[2]
    model = IntervalLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to("cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Обучение модели
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Оценка модели
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # 50 испытаний для оптимизации
    print("Лучшие гиперпараметры:", study.best_params)
    print("Лучшее значение потерь:", study.best_value)

if __name__ == "__main__":
    main()
