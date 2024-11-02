# orderbook_data.py

import asyncio
import websockets
import json
import redis.asyncio as aioredis  # Асинхронная версия библиотеки redis-py
from datetime import datetime
import numpy as np
import logging
import time
import psutil  # Для мониторинга ресурсов
from config import REDIS_CONFIG, FIXED_BOUNDARIES

# Настройка логирования
logging.basicConfig(
    filename='logs/normalized_order_book_detailed.log',
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
        logging.info(f"Нормализованные данные записаны в Redis: {data}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении в Redis: {e}")

# Функция для мониторинга ресурсов
def log_resource_usage():
    memory = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent(interval=0.1)
    logging.info(f"Использование ресурсов: Память {memory}%, CPU {cpu}%")

# Обработка данных ордербука и нормализация
async def analyze_order_book(redis_client):
    symbol = "btcusdt"
    depth_url = f"wss://fstream.binance.com/ws/{symbol}@depth20@100ms"

    while True:
        try:
            async with websockets.connect(depth_url, ping_interval=None) as websocket:
                logging.info(f"Подключение к WebSocket для {symbol} установлено.")
                while True:
                    try:
                        start_time = time.time()
                        message = await websocket.recv()
                        data_received_time = time.time()

                        data = json.loads(message)
                        bids = np.array(data['b'][:8], dtype=float)
                        asks = np.array(data['a'][:8], dtype=float)
                        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # UTC время с миллисекундами

                        # Расчет mid_price и объемов с использованием numpy
                        mid_price = (bids[0, 0] + asks[0, 0]) / 2
                        sum_bid_volume = np.sum(bids[:, 1])
                        sum_ask_volume = np.sum(asks[:, 1])
                        imbalance = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume) if (sum_bid_volume + sum_ask_volume) > 0 else 0

                        # Логирование времени между этапами
                        processing_start_time = time.time()
                        time_diff_receiving_to_processing = processing_start_time - data_received_time
                        utc_time_received = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                        logging.info(f"Данные получены в {utc_time_received}. Разница между получением и обработкой: {time_diff_receiving_to_processing:.3f} сек")

                        # Нормализация данных с использованием numpy
                        normalized_data = {
                            "timestamp": timestamp,
                            "mid_price": min_max_normalize(mid_price, FIXED_BOUNDARIES["mid_price"]["min"], FIXED_BOUNDARIES["mid_price"]["max"]),
                            "sum_bid_volume": min_max_normalize(sum_bid_volume, FIXED_BOUNDARIES["sum_bid_volume"]["min"], FIXED_BOUNDARIES["sum_bid_volume"]["max"]),
                            "sum_ask_volume": min_max_normalize(sum_ask_volume, FIXED_BOUNDARIES["sum_ask_volume"]["min"], FIXED_BOUNDARIES["sum_ask_volume"]["max"]),
                            "imbalance": min_max_normalize(imbalance, FIXED_BOUNDARIES["imbalance"]["min"], FIXED_BOUNDARIES["imbalance"]["max"])
                        }

                        # Сохранение нормализованных данных в Redis
                        await save_to_redis(redis_client, normalized_data)

                        # Логирование времени завершения обработки
                        end_time = time.time()
                        time_diff_processing_to_end = end_time - processing_start_time
                        utc_time_processing_done = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                        logging.info(f"Обработка завершена в {utc_time_processing_done}. Время обработки: {time_diff_processing_to_end:.3f} сек")

                        # Логирование использования ресурсов
                        log_resource_usage()

                        await asyncio.sleep(0.1)
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Ошибка при обработке данных: {e}")
                    except Exception as e:
                        logging.error(f"Неизвестная ошибка: {e}")
        except (websockets.exceptions.ConnectionClosedError, asyncio.TimeoutError) as e:
            logging.error(f"Ошибка соединения WebSocket: {e}. Повторное подключение...")
            await asyncio.sleep(5)  # Ожидание перед повторным подключением

# Основной цикл для запуска обработки
async def main():
    redis_client = await initialize_redis()  # Подготовка Redis перед запуском
    await analyze_order_book(redis_client)
    await redis_client.close()

# Запуск обработки ордербука
if __name__ == "__main__":
    asyncio.run(main())
