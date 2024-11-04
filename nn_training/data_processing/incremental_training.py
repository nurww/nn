import signal
import sys
import redis.asyncio as redis
import json
import asyncio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import logging
from sklearn.metrics import mean_absolute_error, r2_score
import aiofiles
import time

# Настройка логирования
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Флаг для завершения программы безопасным способом
exit_requested = False

# Обработчик сигнала для безопасного завершения
def handle_exit_signal(signum, frame):
    global exit_requested
    exit_requested = True
    logging.info("Запрос на завершение программы получен. Завершение будет выполнено после текущего цикла.")

signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

# Определение модели GRU
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Функция для загрузки и анализа state_dict модели
def load_model_and_inspect(path):
    state_dict = torch.load(path)
    input_size = state_dict['gru.weight_ih_l0'].size(1)
    hidden_size = state_dict['gru.weight_hh_l0'].size(1)
    num_layers = len([key for key in state_dict if 'weight_hh' in key])
    output_size = state_dict['fc.weight'].size(0)
    
    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(state_dict)
    model.eval()
    
    logging.info(f"Model loaded with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, output_size={output_size}")
    return model

# Функция для обучения модели с ранней остановкой
def train_model(model, data, optimizer, criterion, num_epochs=1, patience=5, max_error_pct=3):
    X = np.column_stack([data['mid_prices'], data['moving_average'], data['bid_volumes'], data['ask_volumes'], data['imbalances']])
    y = data['mid_prices']

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

            # Проверка, если ошибка предсказания превышает максимальный процент
            with torch.no_grad():
                abs_diff = torch.abs(outputs - batch_y)
                percent_diff = (abs_diff / batch_y) * 100
                max_error_flag = (percent_diff > max_error_pct).sum().item() > 0

            if max_error_flag:
                logging.warning(f"Prediction exceeds {max_error_pct}% threshold.")

            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_without_improvement = 0
                torch.save(model.state_dict(), 'backup_gru_model.pth')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info("Ранняя остановка: модель перестала улучшаться.")
                return best_loss

    return best_loss

# Функция для кэшированной загрузки данных
async def fetch_with_cache(redis_key):
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )
    
    # Извлечение всех данных
    data = await redis_client.lrange(redis_key, 0, -1)
    data = [json.loads(item) for item in data]
    await redis_client.close()
    
    logging.info(f"Загружено {len(data)} записей из Redis.")
    logging.info(f"Пример данных для обучения: {data[0]}")
    return data

# Функция для проверки корректности данных
def validate_data(data):
    if not data or any(len(entry) == 0 for entry in data):
        logging.error("Найдены некорректные данные.")
        return False
    return True

# Функция для обработки данных
def process_data(data):
    if not validate_data(data):
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

# Функция для сохранения модели асинхронно
async def async_save_model(model, path):
    torch.save(model.state_dict(), path)
    logging.info(f"Модель асинхронно сохранена в {path}")

# Основная функция для запуска программы
async def main():
    model = load_model_and_inspect('gru_model.pth')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    global exit_requested
    while not exit_requested:
        data = await fetch_with_cache("normalized_order_book_stream")
        if data:
            processed_data = process_data(data)
            train_model(model, processed_data, optimizer, criterion, max_error_pct=3)
            await async_save_model(model, 'gru_model.pth')
    
    # await adaptive_pause(len(data))

# Запуск программы
if __name__ == "__main__":
    asyncio.run(main())
