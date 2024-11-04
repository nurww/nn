import asyncio
import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import aiofiles
import redis.asyncio as redis
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
import logging

# Настройка логирования
logging.basicConfig(filename='hyperparams.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Определение модели GRU
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Утилита для проверки существования директории
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Функция загрузки данных из Redis
async def fetch_data(redis_key, batch_size=20000):
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )
    data = []
    start = -batch_size
    end = -1
    while True:
        batch = await redis_client.lrange(redis_key, start, end)
        if not batch:
            break
        data.extend(json.loads(item) for item in batch)
        start -= batch_size
        end -= batch_size
        logging.info(f"Загружен пакет данных: {len(batch)} записей.")
        if len(data) > 100000:
            break
    await redis_client.close()
    return data

# Проверка корректности данных
def validate_and_process_data(data):
    if not data or any(len(entry) == 0 for entry in data):
        logging.error("Найдены некорректные данные.")
        return None

    mid_prices = np.array([entry['mid_price'] for entry in data])
    bid_volumes = np.array([entry['sum_bid_volume'] for entry in data])
    ask_volumes = np.array([entry['sum_ask_volume'] for entry in data])
    imbalances = np.array([entry['imbalance'] for entry in data])
    moving_average = np.convolve(mid_prices, np.ones(10) / 10, mode='valid')

    min_length = min(len(mid_prices), len(moving_average), len(bid_volumes), len(ask_volumes), len(imbalances))
    processed_data = {
        "mid_prices": mid_prices[:min_length],
        "moving_average": moving_average[:min_length],
        "bid_volumes": bid_volumes[:min_length],
        "ask_volumes": ask_volumes[:min_length],
        "imbalances": imbalances[:min_length]
    }
    logging.info(f"Обработанные данные: {min_length} записей.")
    return processed_data

# Обучение модели
def train_model(model, data, optimizer, criterion, num_epochs=1, patience=5):
    X = np.column_stack([data['mid_prices'], data['moving_average'], data['bid_volumes'], data['ask_volumes'], data['imbalances']])
    y = data['mid_prices']
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            logging.info(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info("Ранняя остановка: модель перестала улучшаться.")
                return best_loss

    return best_loss

# Сохранение модели
async def async_save_model(model, path):
    torch.save(model.state_dict(), path)
    logging.info(f"Модель асинхронно сохранена в {path}")

# Основная функция подбора гиперпараметров
async def hyperparameter_tuning():
    input_size = 5
    output_size = 1
    hidden_sizes = [32, 50, 64]
    num_layers_list = [1, 2, 3]
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [16, 32, 64]

    best_loss = float('inf')
    best_model_path = 'gru_model.pth'
    ensure_directory_exists('hyperparams_models')

    for hidden_size in hidden_sizes:
        for num_layers in num_layers_list:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    model = GRUModel(input_size, hidden_size, num_layers, output_size)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

                    data = await fetch_data("normalized_order_book_stream", batch_size=1500)
                    if data:
                        processed_data = validate_and_process_data(data)
                        if processed_data:
                            loss = train_model(model, processed_data, optimizer, criterion)

                            model_path = f'hyperparams_models/gru_model_h{hidden_size}_l{num_layers}_lr{lr}_bs{batch_size}.pth'
                            torch.save(model.state_dict(), model_path)

                            if loss < best_loss:
                                best_loss = loss
                                torch.save(model.state_dict(), best_model_path)
                                logging.info(f"Лучшая модель обновлена: {model_path}")
    logging.info(f"Лучшая модель обновлена: {model_path}")

if __name__ == "__main__":
    asyncio.run(hyperparameter_tuning())
