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

# Функция для загрузки данных из базы данных
def load_data_from_db(connection, table_name, interval):
    query = f"""
        SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized,
               volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized,
               sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
        FROM {table_name}
        WHERE data_interval = '{interval}'
    """
    df = pd.read_sql(query, connection)
    X = df.drop(columns=['close_price_normalized']).values
    y = df['close_price_normalized'].values
    return X, y

# Функция для разделения данных на обучающую и тестовую выборки
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Функция для обучения модели
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

    return running_loss / len(train_loader)

# Функция для оценки модели
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

    return running_loss / len(test_loader)

# Оптимизируемая функция
def objective(trial):
    # Подключение к базе данных
    connection = mysql.connector.connect(
        host='localhost',
        database='binance_data',
        user='root',
        password='root'
    )

    try:
        # Загрузка данных
        X, y = load_data_from_db(connection, 'binance_klines_normalized', '1m')
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        # Преобразование данных в тензоры
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        # Определение модели для оптимизации
        model_type = trial.suggest_categorical('model_type', ['LSTM', 'Transformer'])

        if model_type == 'LSTM':
            # Определение гиперпараметров для LSTM
            hidden_size = trial.suggest_int('hidden_size', 50, 300)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        else:  # Для Transformer
            num_heads = trial.suggest_int('num_heads', 2, 8)
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
            hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        # Общие гиперпараметры для обеих моделей
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 88, 128])
        accumulation_steps = trial.suggest_int('accumulation_steps', 1, 4)

        # Оптимизация количества эпох
        epochs = trial.suggest_int('epochs', 5, 20)  # Оптимизация числа эпох от 5 до 20

        # Создание модели в зависимости от типа
        if model_type == 'LSTM':
            model = build_model(
                model_type='LSTM',
                input_size=X_train.shape[1],
                output_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            ).to(device)
        else:
            model = build_model(
                model_type='Transformer',
                input_size=X_train.shape[1],
                output_size=1,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate
            ).to(device)

        # Создаем DataLoader для обучения и тестирования
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        best_loss = float('inf')

        # Цикл обучения с оптимизируемым количеством эпох
        for epoch in range(epochs):
            epoch_loss = train_model(model, train_loader, optimizer, criterion, device, accumulation_steps)
            test_loss = evaluate_model(model, test_loader, criterion, device)

            if test_loss < best_loss:
                best_loss = test_loss

        return best_loss

    finally:
        if connection.is_connected():
            connection.close()

# Основная функция для запуска оптимизации
def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # Запускаем оптимизацию

    print("Лучшие гиперпараметры: ", study.best_params)
    print("Лучшее значение потерь: ", study.best_value)

if __name__ == '__main__':
    main()
