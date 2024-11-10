# orderbook_data.py

import asyncio
import websockets
import json
import redis.asyncio as aioredis  # Асинхронная версия библиотеки redis-py
from datetime import datetime
import numpy as np
import logging
import os

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

FIXED_BOUNDARIES = {
    "mid_price": {"min": 20000, "max": 90000},
    "sum_bid_volume": {"min": 0, "max": 15000},
    "sum_ask_volume": {"min": 0, "max": 15000},
    "imbalance": {"min": -1, "max": 1}
}

# Настройка логирования
logging.basicConfig(
    filename='../../logs/normalized_order_book_detailed.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Настройка асинхронного подключения к Redis
async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

# Функция для нормализации данных с фиксированными границами
def min_max_normalize(values, min_val, max_val):
    normalized = (values - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)  # Ограничение значений между 0 и 1
    return normalized

# Функция для сохранения данных в Redis
async def save_to_redis(redis_client, data):
    try:
        await redis_client.rpush("normalized_order_book_stream", json.dumps(data))
        await redis_client.ltrim("normalized_order_book_stream", -1500, -1)
    except Exception as e:
        logging.error(f"Ошибка при сохранении в Redis: {e}")

# Функция для извлечения сигнала от модели интервалов из Redis
async def get_interval_signal(redis_client):
    """
    Извлекает предсказание от модели интервалов из Redis.
    
    :param redis_client: клиент Redis
    :return: сигнал от модели интервалов
    """
    interval_signal = await redis_client.get("interval_signal")
    return int(interval_signal) if interval_signal is not None else 0  # Возвращаем 0, если сигнала нет

# Подготовка запроса для подписки на Binance Futures
def get_subscription_params():
    params = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@depth20@100ms"],  # Подписка на стакан для фьючерсного BTCUSDT с интервалом 100 мс
        "id": 1
    }
    return json.dumps(params)

# Обработка данных ордербука и нормализация
async def analyze_order_book(redis_client):
    endpoint = "wss://fstream.binance.com/ws/"  # URL для Binance Futures WebSocket
    while True:
        try:
            async with websockets.connect(endpoint) as websocket:
                await websocket.send(get_subscription_params())  # Подписка на канал стакана

                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)

                        # Обработка данных ордербука
                        bids = np.array(data['b'][:8], dtype=float)
                        asks = np.array(data['a'][:8], dtype=float)
                        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                        mid_price = (bids[0, 0] + asks[0, 0]) / 2
                        sum_bid_volume = np.sum(bids[:, 1])
                        sum_ask_volume = np.sum(asks[:, 1])
                        imbalance = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume) if (sum_bid_volume + sum_ask_volume) > 0 else 0

                        # Нормализация данных
                        normalized_data = {
                            "timestamp": timestamp,
                            "mid_price": min_max_normalize(mid_price, FIXED_BOUNDARIES["mid_price"]["min"], FIXED_BOUNDARIES["mid_price"]["max"]),
                            "sum_bid_volume": min_max_normalize(sum_bid_volume, FIXED_BOUNDARIES["sum_bid_volume"]["min"], FIXED_BOUNDARIES["sum_bid_volume"]["max"]),
                            "sum_ask_volume": min_max_normalize(sum_ask_volume, FIXED_BOUNDARIES["sum_ask_volume"]["min"], FIXED_BOUNDARIES["sum_ask_volume"]["max"]),
                            "imbalance": min_max_normalize(imbalance, FIXED_BOUNDARIES["imbalance"]["min"], FIXED_BOUNDARIES["imbalance"]["max"])
                        }

                        # Сохранение нормализованных данных в Redis
                        await save_to_redis(redis_client, normalized_data)
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Ошибка при обработке данных: {e}")
                        await asyncio.sleep(5)
                    except Exception as e:
                        logging.error(f"Неизвестная ошибка: {e}")
                        await asyncio.sleep(5)
        except (websockets.exceptions.ConnectionClosedError, asyncio.TimeoutError) as e:
            logging.error(f"Ошибка соединения WebSocket: {e}. Повторное подключение...")
            await asyncio.sleep(5)

# Функция для запуска скрипта бесконечно
async def main():
    redis_client = await initialize_redis()
    try:
        interval_signal = await get_interval_signal(redis_client)
        print("Received interval signal:", interval_signal)
        await analyze_order_book(redis_client)
    finally:
        await redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())
