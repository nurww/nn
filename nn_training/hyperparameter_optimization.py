# hyperparameter_optimization.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model_builder import build_model
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
import optuna
import os
import tqdm
import logging
import time
from sqlalchemy import create_engine

# Настройка логирования
logging.basicConfig(filename='hyperparam_intervals.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# Функция для загрузки данных из базы данных
def load_data_from_db(connection, table_name, interval):
    logging.info(f"Загрузка данных для интервала {interval}")
    query = f"""
        SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized,
               volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized,
               sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
        FROM {table_name}
        WHERE data_interval = '{interval}'
    """
    df = pd.read_sql(query, connection)
    if df.empty:
        logging.warning(f"Данные для интервала {interval} отсутствуют")
    return df.drop(columns=['close_price_normalized']).values, df['close_price_normalized'].values

# Функция для разделения данных на обучающую и тестовую выборки
def split_data(X, y, test_size=0.2):
    logging.info("Разделение данных на обучающую и тестовую выборки")
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Функция для обучения модели
def train_model(model, train_loader, optimizer, criterion, device, accumulation_steps):
    logging.info("Начало обучения модели")
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    start_time = time.time()
    with tqdm.tqdm(total=len(train_loader), desc='Training', unit='batch') as pbar:
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)

    end_time = time.time()
    logging.info(f"Средняя потеря за эпоху: {running_loss / len(train_loader)}")
    logging.info(f"Обучение завершено за {end_time - start_time:.2f} секунд")
    return running_loss / len(train_loader)

# Функция для оценки модели
def evaluate_model(model, test_loader, criterion, device):
    logging.info("Начало оценки модели")
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_loader), desc='Evaluating', unit='batch') as pbar:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                pbar.update(1)

    logging.info(f"Средняя потеря при оценке: {running_loss / len(test_loader)}")
    return running_loss / len(test_loader)

# Функция для загрузки данных из базы данных по разным интервалам
def load_data_from_db_for_intervals(engine, table_name, intervals):
    all_data = {}
    for interval in intervals:
        logging.info(f"Загрузка данных из базы для интервала {interval}")
        query = f"""
            SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized,
                   volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized,
                   sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
            FROM {table_name}
            WHERE data_interval = '{interval}'
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            logging.warning(f"Нет данных для интервала {interval}")
        else:
            logging.info(f"Загружено {len(df)} записей для интервала {interval}")
        X = df.drop(columns=['close_price_normalized']).values
        y = df['close_price_normalized'].values
        all_data[interval] = (X, y)
    return all_data

# Оптимизируемая функция с использованием SQLAlchemy
def objective(trial):
    logging.info("Запуск нового испытания")
    engine = create_engine('mysql+mysqlconnector://root:root@localhost/binance_data')

    try:
        # intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
        intervals = ['1d', '4h', '1h', '15m', '5m', '1m']
        all_data = load_data_from_db_for_intervals(engine, 'binance_klines_normalized', intervals)
        
        total_loss = 0.0

        for interval, (X, y) in all_data.items():
            if len(X) == 0 or len(y) == 0:
                logging.warning(f"Пропуск интервала {interval} из-за отсутствия данных")
                continue

            X_train, X_test, y_train, y_test = split_data(X, y)
            logging.info(f"Размеры обучающей выборки: {X_train.shape}, тестовой выборки: {X_test.shape}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

            model_type = trial.suggest_categorical('model_type', ['LSTM', 'Transformer'])
            logging.info(f"Выбран тип модели: {model_type}")

            if model_type == 'LSTM':
                hidden_size = trial.suggest_int('hidden_size', 50, 300)
                num_layers = trial.suggest_int('num_layers', 1, 3)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                model = build_model(
                    model_type='LSTM',
                    input_size=X_train.shape[1],
                    output_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate
                ).to(device)
                logging.info(f"Создана модель LSTM с параметрами: hidden_size={hidden_size}, num_layers={num_layers}, dropout_rate={dropout_rate}")
            else:
                num_heads = trial.suggest_int('num_heads', 2, 8)
    
                # Формируем список значений hidden_dim, которые делятся на num_heads без остатка
                valid_hidden_dims = [dim for dim in range(64, 365, 2) if dim % num_heads == 0]
                
                # Выбираем значение из допустимых
                hidden_dim = trial.suggest_categorical('hidden_dim', valid_hidden_dims)

                num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                model = build_model(
                    model_type='Transformer',
                    input_size=X_train.shape[1],
                    output_size=1,
                    num_heads=num_heads,
                    num_encoder_layers=num_encoder_layers,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    batch_first=True
                ).to(device)
                logging.info(f"Создана модель Transformer с параметрами: num_heads={num_heads}, hidden_dim={hidden_dim}, num_encoder_layers={num_encoder_layers}, dropout_rate={dropout_rate}")

            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 88, 128])
            accumulation_steps = trial.suggest_int('accumulation_steps', 1, 4)
            epochs = trial.suggest_int('epochs', 5, 20)
            logging.info(f"Гиперпараметры обучения: learning_rate={learning_rate}, batch_size={batch_size}, accumulation_steps={accumulation_steps}, epochs={epochs}")

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()

            for epoch in range(epochs):
                logging.info(f"Начало эпохи {epoch + 1}")
                epoch_loss = train_model(model, train_loader, optimizer, criterion, device, accumulation_steps)
                test_loss = evaluate_model(model, test_loader, criterion, device)
                logging.info(f"Потери на эпохе {epoch + 1}: {epoch_loss}, потери при оценке: {test_loss}")

                if test_loss < total_loss or total_loss == 0.0:
                    total_loss = test_loss

        return total_loss / len(all_data)

    finally:
        logging.info("Завершение испытания")

# Основная функция для запуска оптимизации
def main():
    logging.info("Начало оптимизации гиперпараметров")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # Запускаем оптимизацию
    logging.info("Оптимизация завершена")

    print("Лучшие гиперпараметры: ", study.best_params)
    print("Лучшее значение потерь: ", study.best_value)

if __name__ == '__main__':
    main()
