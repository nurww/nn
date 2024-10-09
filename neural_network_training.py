# neural_network_training.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model_builder import LSTMModel
from datetime import datetime
import os
import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import talib as ta

# Это нужно для того, чтобы оценить производительность модели на невидимых данных.
# Добавим функцию для разделения данных перед обучением нейросети
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Функция для сохранения модели
def save_best_model(model, epoch, best_loss):
    folder_name = 'saved_models_with_indicators'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model_path = os.path.join(folder_name, f"best_model_epoch_{epoch}_loss_{best_loss:.12f}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена как {model_path}")


# Теперь давай добавим кастомную функцию потерь для нейросети, которая будет учитывать прибыль и убытки.
# Это поможет сделать модель более "мотивационной".
def custom_loss(outputs, targets):
    mse_loss = torch.nn.functional.mse_loss(outputs, targets)
    loss_penalty = torch.mean(torch.relu(targets - outputs)) * 0.5
    reward = torch.mean(torch.relu(outputs - targets)) * 0.5
    return mse_loss + loss_penalty - reward

# Для отслеживания процесса обучения добавим логирование
def log_training(epoch, loss):
    with open('training_log.txt', 'a') as log_file:
        log_file.write(f'Epoch {epoch}, Loss: {loss}\n')

# Загрузка данных
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['Цена открытия', 'Максимум', 'Минимум', 'Цена закрытия', 'Объем', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_20', 'EMA_20', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'OBV']].values
    y = data['Цена закрытия'].values
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

# Функция для оценки модели на тестовых данных с прогресс-баром
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Загрузка данных
    X, y = load_data('data/train_data_with_indicators.csv')
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Преобразование данных в тензоры
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    # Оптимальные параметры
    best_params = {
        'hidden_size': 153,
        'num_layers': 1,
        'dropout_rate': 0.3216,
        'batch_size': 88,
        'learning_rate': 0.00166,
        'accumulation_steps': 2
    }

    # Датасеты и DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

    # Инициализация модели
    input_size = X_train.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], output_size=1, dropout_rate=best_params['dropout_rate']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    criterion = custom_loss  # Кастомная функция потерь

    epochs = 10
    best_loss = float('inf')

    for epoch in range(epochs):
        # Обучение
        epoch_loss = train_model(model, train_loader, optimizer, criterion, device, best_params['accumulation_steps'])
        log_training(epoch, epoch_loss)
        print(f"Loss at epoch {epoch + 1}: {epoch_loss}")

        # Проверка на тестовых данных
        test_loss = evaluate_model(model, test_loader, criterion, device)

        # Сохранение лучшей модели
        if test_loss < best_loss:
            best_loss = test_loss
            save_best_model(model, epoch, best_loss)

    print(f"Лучшее значение потери на тестовых данных: {best_loss}")

if __name__ == '__main__':
    main()