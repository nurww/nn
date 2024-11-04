# model_training.py

import asyncio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from nn_processor.nn_training.data_processing.fetch_data import fetch_data_from_redis
from nn_processor.nn_training.data_processing.process_data import process_data
from datetime import datetime

# Определение LSTM модели
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Используем выход последнего таймстепа
        return out

# Функция для обучения модели
def train_model(data):
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - start time")

    min_length = min(len(data['mid_prices']), len(data['moving_average']), len(data['bid_volumes']), 
                 len(data['ask_volumes']), len(data['imbalances']))

    # Обрезка массивов до минимальной длины
    mid_prices = data['mid_prices'][:min_length]
    moving_average = data['moving_average'][:min_length]
    bid_volumes = data['bid_volumes'][:min_length]
    ask_volumes = data['ask_volumes'][:min_length]
    imbalances = data['imbalances'][:min_length]

    # Объединение массивов
    X = np.column_stack([mid_prices, moving_average, bid_volumes, ask_volumes, imbalances])
    y = data['mid_prices']  # Пример меток (может быть изменено на нужные данные)

    # Преобразование данных для использования в LSTM
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [samples, timesteps, features]
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [samples, 1]

    # Создание TensorDataset и DataLoader
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Параметры модели
    input_size = X.shape[2]
    hidden_size = 50
    num_layers = 2
    output_size = 1

    # Инициализация модели, функции потерь и оптимизатора
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Оценка модели
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

    print(f'Средняя ошибка на тестовой выборке: {total_loss / len(test_loader):.4f}')
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - before model save time")

    # Сохранение модели
    torch.save(model.state_dict(), 'model/lstm_model.pth')
    print("Модель сохранена!")
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - end time")

# Запуск обучения
if __name__ == "__main__":
    data = asyncio.run(fetch_data_from_redis())  # Загрузка данных из Redis
    processed_data = process_data(data)
    train_model(processed_data)
