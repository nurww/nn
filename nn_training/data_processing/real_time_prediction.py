import redis.asyncio as redis
import json
import asyncio
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(filename='prediction.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        out = self.fc(out[:, -1, :])
        return out

# Функция для загрузки данных из Redis
async def fetch_latest_data_from_redis(redis_key="normalized_order_book_stream", num_entries=1500):
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )
    data = await redis_client.lrange(redis_key, -num_entries, -1)
    data = [json.loads(item) for item in data]
    await redis_client.close()
    return data

# Функция для обработки данных
def process_data_for_prediction(data):
    mid_prices = np.array([entry['mid_price'] for entry in data])
    bid_volumes = np.array([entry['sum_bid_volume'] for entry in data])
    ask_volumes = np.array([entry['sum_ask_volume'] for entry in data])
    imbalances = np.array([entry['imbalance'] for entry in data])
    moving_average = np.convolve(mid_prices, np.ones(10) / 10, mode='valid')

    min_length = min(len(mid_prices), len(moving_average), len(bid_volumes), len(ask_volumes), len(imbalances))
    X = np.column_stack([mid_prices[:min_length], moving_average[:min_length], bid_volumes[:min_length], ask_volumes[:min_length], imbalances[:min_length]])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # [1, timesteps, features]

def denormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value

# Загрузка модели и просмотр архитектуры
def load_model_and_inspect(path):
    state_dict = torch.load(path)

    # Просмотр структуры state_dict
    for param_tensor in state_dict:
        print(f"{param_tensor}: {state_dict[param_tensor].size()}")

    # Определение параметров модели
    input_size = state_dict['gru.weight_ih_l0'].size(1)  # Число признаков
    hidden_size = state_dict['gru.weight_hh_l0'].size(1)  # Размер скрытого слоя
    num_layers = len([key for key in state_dict if 'weight_hh' in key])  # Число слоев
    output_size = state_dict['fc.weight'].size(0)  # Размер выходного слоя

    print(f"Extracted Parameters - Input Size: {input_size}, Hidden Size: {hidden_size}, "
          f"Num Layers: {num_layers}, Output Size: {output_size}")

    # Создание модели с извлеченными параметрами
    model = GRUModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Функция для прогнозирования и записи результатов
async def predict_and_record(model, redis_client, num_entries=1500):
    while True:
        data = await fetch_latest_data_from_redis(num_entries=num_entries)
        if len(data) < 10:
            print("Недостаточно данных для предсказания.")
            await asyncio.sleep(0.1)
            continue

        input_data = process_data_for_prediction(data)
        with torch.no_grad():
            prediction = model(input_data)
            predicted_mid_price = prediction.item()
            actual_mid_price = data[-1]['mid_price']

            # Предположим, что вы знаете минимальные и максимальные значения
            min_mid_price = 40000  # Минимальное значение для mid_price
            max_mid_price = 85000  # Максимальное значение для mid_price

            predicted_mid_price_denorm = denormalize(predicted_mid_price, min_mid_price, max_mid_price)
            actual_mid_price_denorm = denormalize(actual_mid_price, min_mid_price, max_mid_price)

            # Расчет метрик после денормализации
            absolute_difference = abs(predicted_mid_price_denorm - actual_mid_price_denorm)
            percent_difference = (absolute_difference / actual_mid_price_denorm) * 100 if actual_mid_price_denorm != 0 else float('inf')
            
            # Запись в Redis
            result = {
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'expected': predicted_mid_price,
                'actual': actual_mid_price,
                'absolute_difference': absolute_difference,
                'percent_difference': percent_difference
            }

            # Логирование и вывод
            logging.info(f"Предсказание: {predicted_mid_price_denorm:.4f}, Реальный mid_price: {actual_mid_price_denorm:.4f}, Абсолютная разница: {absolute_difference:.4f}, % разницы: {percent_difference:.2f}%")
            print(f"{result['timestamp']} - Предсказанный mid_price: {predicted_mid_price_denorm:.4f}, Реальный mid_price: {actual_mid_price_denorm:.4f}, Абсолютная разница: {absolute_difference:.4f}, % разницы: {percent_difference:.2f}%")

            await redis_client.rpush('model_predictions', json.dumps(result))

        # Задержка 100 мс перед следующим прогнозом
        await asyncio.sleep(0.1)

# Загрузка модели и запуск предсказаний
async def main():
    # model = GRUModel(input_size, hidden_size, num_layers, output_size)
    # model.load_state_dict(torch.load('gru_model.pth', weights_only=True))

    # Загрузка и просмотр модели
    model = load_model_and_inspect('gru_model.pth')

    # Создание клиента Redis
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )

    # Запуск предсказаний
    await predict_and_record(model, redis_client)

    # Закрытие клиента Redis после завершения работы
    await redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())
