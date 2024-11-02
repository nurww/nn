# model_training.py

import asyncio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_processing.fetch_data import fetch_data_from_redis

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
    # Подготовка данных
    X = np.column_stack([
        data['mid_prices'],
        data['moving_average'],
        data['bid_volumes'],
        data['ask_volumes'],
        data['imbalances']
    ])
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
    num_epochs = 50
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

    # Сохранение модели
    torch.save(model.state_dict(), 'model/lstm_model.pth')
    print("Модель сохранена!")

# Запуск обучения
if __name__ == "__main__":
    data = asyncio.run(fetch_data_from_redis())  # Загрузка данных из Redis
    train_model(data)
