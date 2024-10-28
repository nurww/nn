#cross_validation.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model_builder import LSTMModel
from datetime import datetime
import os
import tqdm
from sklearn.model_selection import KFold
import mysql.connector
from mysql.connector import Error


# Функция для сохранения модели
def save_best_model(model, fold, epoch, loss):
    folder_name = 'saved_models_cross_validation'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model_path = os.path.join(folder_name, f"best_model_fold_{fold}_epoch_{epoch}_loss_{loss:.12f}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Модель для фолда {fold} сохранена как {model_path}")

# Функция для загрузки данных из базы данных вместо файла CSV
def load_data_from_db(connection, table_name, interval):
    query = f"""
        SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized,
               volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized,
               sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
        FROM {table_name}
        WHERE data_interval = '{interval}'
    """
    df = pd.read_sql(query, connection)
    
    X = df[['open_price_normalized', 'high_price_normalized', 'low_price_normalized', 'close_price_normalized', 
            'volume_normalized', 'rsi_normalized', 'macd_normalized', 'macd_signal_normalized', 'macd_hist_normalized', 
            'sma_20_normalized', 'ema_20_normalized', 'upper_bb_normalized', 'middle_bb_normalized', 'lower_bb_normalized', 'obv_normalized']].values
    y = df['close_price_normalized'].values  # Прогнозируем значение цены закрытия
    return X, y

# Функция обучения модели
def train_model(model, train_loader, optimizer, criterion, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

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

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

# Функция для оценки модели на тестовых данных
def evaluate_model(model, test_loader, criterion, device):
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

    avg_loss = running_loss / len(test_loader)
    print(f"Средняя потеря на тестовых данных: {avg_loss}")
    return avg_loss

# Функция для кросс-валидации
def cross_validate(connection, table_name, interval, n_splits=5, epochs=10, best_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Загрузка данных из базы данных
    X, y = load_data_from_db(connection, table_name, interval)
    
    kf = KFold(n_splits=n_splits)
    
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Преобразование данных в тензоры
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        # Создание DataLoader для тренировки и тестирования
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=best_params['batch_size'], shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=best_params['batch_size'])

        # Инициализация модели
        input_size = X_train.shape[1]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            output_size=1,
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = torch.nn.MSELoss()

        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = train_model(model, train_loader, optimizer, criterion, device, best_params['accumulation_steps'])
            print(f"Fold {fold}, Epoch {epoch + 1}: Loss = {epoch_loss}")

            # Оценка модели на тестовых данных
            test_loss = evaluate_model(model, test_loader, criterion, device)

            # Сохранение модели, если она лучшая на данный момент
            if test_loss < best_loss:
                best_loss = test_loss
                save_best_model(model, fold, epoch + 1, best_loss)

        fold += 1

    # Финальная проверка на глобальных данных (опционально)
    X_test_global, y_test_global = load_data_from_db(connection, table_name, interval)
    X_test_tensor_global = torch.tensor(X_test_global, dtype=torch.float32).to(device)
    y_test_tensor_global = torch.tensor(y_test_global, dtype=torch.float32).view(-1, 1).to(device)

    test_loader_global = DataLoader(TensorDataset(X_test_tensor_global, y_test_tensor_global), batch_size=best_params['batch_size'])

    # Финальная проверка на последней модели
    evaluate_model(model, test_loader_global, criterion, device)


if __name__ == '__main__':
    # Оптимальные параметры, найденные Optuna
    best_params = {
        'hidden_size': 153,
        'num_layers': 1,
        'dropout_rate': 0.3216,
        'batch_size': 88,
        'learning_rate': 0.00166,
        'accumulation_steps': 2
    }

    # Подключение к базе данных
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='binance_data',
            user='root',
            password='root'
        )
        
        # Запуск кросс-валидации
        cross_validate(connection, 'binance_klines_normalized', '1m', n_splits=5, epochs=10, best_params=best_params)
    
    except Error as e:
        print(f"Ошибка подключения к MySQL: {e}")
    
    finally:
        if connection.is_connected():
            connection.close()
            print("Соединение с MySQL закрыто")