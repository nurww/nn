# incremental_training.py

import redis.asyncio as redis
import json
import asyncio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

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
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Используем выход последнего таймстепа
        return out

# Функция для извлечения данных из Redis
async def fetch_data_from_redis(redis_key="normalized_order_book_stream"):
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )
    data = await redis_client.lrange(redis_key, -1500, -1)
    data = [json.loads(item) for item in data]
    await redis_client.close()
    return data

# Функция для обработки данных
def process_data(data):
    mid_prices = np.array([entry['mid_price'] for entry in data])
    bid_volumes = np.array([entry['sum_bid_volume'] for entry in data])
    ask_volumes = np.array([entry['sum_ask_volume'] for entry in data])
    imbalances = np.array([entry['imbalance'] for entry in data])
    moving_average = np.convolve(mid_prices, np.ones(10) / 10, mode='valid')

    min_length = min(len(mid_prices), len(moving_average), len(bid_volumes), len(ask_volumes), len(imbalances))
    return {
        "mid_prices": mid_prices[:min_length],
        "moving_average": moving_average[:min_length],
        "bid_volumes": bid_volumes[:min_length],
        "ask_volumes": ask_volumes[:min_length],
        "imbalances": imbalances[:min_length]
    }

# Функция для инкрементального обучения модели
def incremental_train(model, data, optimizer, criterion):
    model.train()
    X = np.column_stack([data['mid_prices'], data['moving_average'], data['bid_volumes'], data['ask_volumes'], data['imbalances']])
    y = data['mid_prices']

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item():.4f}')

# Основная функция для выполнения всех этапов
async def main():
    # Инициализация модели и оптимизатора
    input_size = 5  # Количество признаков
    hidden_size = 50
    num_layers = 2
    output_size = 1

    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    while True:
        data = await fetch_data_from_redis()
        processed_data = process_data(data)
        incremental_train(model, processed_data, optimizer, criterion)
        torch.save(model.state_dict(), 'gru_model.pth')
        print(f'Model updated at {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}')

        await asyncio.sleep(2)  # Интервал обновления в 2 секунды

# Запуск программы
if __name__ == "__main__":
    asyncio.run(main())


# import redis.asyncio as redis
# import json
# import asyncio
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from datetime import datetime

# # Конфигурация для Redis
# REDIS_CONFIG = {
#     'host': 'localhost',
#     'port': 6379,
#     'db': 0
# }

# # Определение LSTM модели
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # Используем выход последнего таймстепа
#         return out

# # Функция для извлечения данных из Redis
# async def fetch_data_from_redis(redis_key="normalized_order_book_stream"):
#     redis_client = redis.Redis(
#         host=REDIS_CONFIG['host'],
#         port=REDIS_CONFIG['port'],
#         db=REDIS_CONFIG['db'],
#         decode_responses=True
#     )
#     # Извлечение последних 1500 записей из Redis
#     data = await redis_client.lrange(redis_key, -1500, -1)
#     data = [json.loads(item) for item in data]  # Преобразование строки JSON в словарь
    
#     await redis_client.close()
#     return data

# # Функция для обработки данных
# def process_data(data):
#     # Преобразование данных в numpy массив
#     timestamps = [entry['timestamp'] for entry in data]
#     mid_prices = np.array([entry['mid_price'] for entry in data])
#     bid_volumes = np.array([entry['sum_bid_volume'] for entry in data])
#     ask_volumes = np.array([entry['sum_ask_volume'] for entry in data])
#     imbalances = np.array([entry['imbalance'] for entry in data])

#     # Расчет скользящей средней для mid_price
#     moving_average = np.convolve(mid_prices, np.ones(10) / 10, mode='valid')

#     # Возвращаем обработанные данные в виде словаря
#     return {
#         "timestamps": timestamps,
#         "mid_prices": mid_prices,
#         "moving_average": moving_average,
#         "bid_volumes": bid_volumes,
#         "ask_volumes": ask_volumes,
#         "imbalances": imbalances
#     }

# # Функция для обучения модели
# def train_model(data):
#     print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - start time")

#     min_length = min(len(data['mid_prices']), len(data['moving_average']), len(data['bid_volumes']),
#                      len(data['ask_volumes']), len(data['imbalances']))

#     # Обрезка массивов до минимальной длины
#     mid_prices = data['mid_prices'][:min_length]
#     moving_average = data['moving_average'][:min_length]
#     bid_volumes = data['bid_volumes'][:min_length]
#     ask_volumes = data['ask_volumes'][:min_length]
#     imbalances = data['imbalances'][:min_length]

#     # Объединение массивов
#     X = np.column_stack([mid_prices, moving_average, bid_volumes, ask_volumes, imbalances])
#     y = mid_prices  # Пример меток (может быть изменено на нужные данные)

#     # Преобразование данных для использования в LSTM
#     X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [samples, timesteps, features]
#     y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [samples, 1]

#     # Создание TensorDataset и DataLoader
#     dataset = TensorDataset(X, y)
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     # Параметры модели
#     input_size = X.shape[2]
#     hidden_size = 50
#     num_layers = 2
#     output_size = 1

#     # Инициализация модели, функции потерь и оптимизатора
#     model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Обучение модели
#     num_epochs = 1
#     for epoch in range(num_epochs):
#         model.train()
#         for batch_X, batch_y in train_loader:
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

#     # Оценка модели
#     model.eval()
#     with torch.no_grad():
#         total_loss = 0
#         for batch_X, batch_y in test_loader:
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             total_loss += loss.item()

#     print(f'Средняя ошибка на тестовой выборке: {total_loss / len(test_loader):.4f}')
#     print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - before model save time")

#     # Сохранение модели
#     torch.save(model.state_dict(), 'lstm_model.pth')
#     print("Модель сохранена!")
#     print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - end time")

# # Основная функция для выполнения всех этапов
# async def main():
#     data = await fetch_data_from_redis()  # Загрузка данных из Redis
#     processed_data = process_data(data)
#     train_model(processed_data)

# # Запуск программы
# if __name__ == "__main__":
#     asyncio.run(main())


# print(project_root)

# print("Current Working Directory:", os.getcwd())
# import aioredis
# import json
# import asyncio
# from config import REDIS_CONFIG

# async def fetch_data_from_redis(redis_key="normalized_order_book_stream"):
#     redis_client = await aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])
    
#     while True:
#         # Подготовка данных и обучение модели
#         data = await redis_client.lrange(redis_key, -1500, -1)
#         data = [json.loads(item) for item in data]  # Преобразование строки JSON в словарь
        
#         await redis_client.close()
#         # Задержка перед следующим циклом
#         await asyncio.sleep(0.1)  # Запуск цикла раз в минуту

#         return data

#         # Построение графика
#         # await plot_main()

#     # Извлечение последних 900 записей из Redis
#     # data = await redis_client.lrange(redis_key, -1500, -1)
#     # data = [json.loads(item) for item in data]  # Преобразование строки JSON в словарь
    
#     # await redis_client.close()
#     # return data

# # Запуск для проверки
# if __name__ == "__main__":
#     asyncio.run(fetch_data_from_redis())
