# multi_agent_trading.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from transformer_model import TransformerModel
from datetime import datetime
import os
import tqdm
from sklearn.model_selection import train_test_split
import mysql.connector
from mysql.connector import Error

# Функции из предыдущих скриптов
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

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

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
    return avg_loss

def save_best_model(model, agent_id, epoch, best_loss):
    folder_name = f'saved_models_agent_{agent_id}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model_path = os.path.join(folder_name, f"best_model_epoch_{epoch}_loss_{best_loss:.12f}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Модель для агента {agent_id} сохранена как {model_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Оптимальные параметры
    best_params = {
        'num_heads': 4,
        'num_encoder_layers': 3,
        'hidden_dim': 256,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'accumulation_steps': 2
    }

    # Определение агентов с разными интервалами
    agents = {
        'short_term': '1m',
        'medium_term': '5m',
        'long_term': '1h'
    }

    try:
        # Подключение к базе данных
        connection = mysql.connector.connect(
            host='localhost',
            database='binance_data',
            user='root',
            password='root'
        )

        for agent_id, interval in agents.items():
            print(f"Обучение агента {agent_id} с интервалом {interval}")

            # Загрузка данных из базы данных
            X, y = load_data_from_db(connection, 'binance_klines_normalized', interval)

            # Разделение данных на тренировочные и тестовые
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

            # Преобразование данных в тензоры
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

            # Датасеты и DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

            # Инициализация модели
            input_size = X_train.shape[1]
            model = TransformerModel(
                input_size=input_size,
                num_heads=best_params['num_heads'],
                num_encoder_layers=best_params['num_encoder_layers'],
                hidden_dim=best_params['hidden_dim'],
                output_size=1,
                dropout=best_params['dropout']
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
            criterion = custom_loss  # Используйте вашу кастомную функцию потерь

            epochs = 10
            best_loss = float('inf')

            for epoch in range(epochs):
                epoch_loss = train_model(model, train_loader, optimizer, criterion, device, best_params['accumulation_steps'])
                print(f"Agent {agent_id}, Epoch {epoch + 1}: Loss = {epoch_loss}")

                test_loss = evaluate_model(model, test_loader, criterion, device)
                print(f"Agent {agent_id}, Epoch {epoch + 1}: Test Loss = {test_loss}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    save_best_model(model, agent_id, epoch + 1, best_loss)

            print(f"Лучшее значение потери для агента {agent_id}: {best_loss}")

    except Error as e:
        print(f"Ошибка подключения к MySQL: {e}")

    finally:
        if connection.is_connected():
            connection.close()
            print("Соединение с MySQL закрыто")

if __name__ == '__main__':
    main()
