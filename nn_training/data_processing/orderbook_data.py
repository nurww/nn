# orderbook_data.py

import asyncio
import websockets
import json
import redis.asyncio as aioredis
import numpy as np
import logging
from datetime import datetime, timedelta
import mysql.connector

# Конфигурация для Redis и MySQL
REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'db': 0}
MYSQL_CONFIG = {
    'host': 'localhost',
    'database': 'binance_data',
    'user': 'root',
    'password': 'root'
}

FIXED_BOUNDARIES = {
    "mid_price": {"min": 20000, "max": 100000},
    "sum_bid_volume": {"min": 0, "max": 20000},
    "sum_ask_volume": {"min": 0, "max": 20000},
    "imbalance": {"min": -1, "max": 1}
}

# Настройка логирования
logging.basicConfig(
    filename='../../logs/normalized_order_book_detailed.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

def min_max_normalize(value, min_val, max_val):
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)

async def save_to_redis(redis_client, data, key):
    try:
        await redis_client.rpush(key, json.dumps(data))
        await redis_client.ltrim(key, -1500, -1)
    except Exception as e:
        logging.error(f"Ошибка при сохранении в Redis: {e}")

async def save_to_mysql(mysql_conn, data_list):
    try:
        cursor = mysql_conn.cursor()
        query = """
            INSERT INTO order_book_data (timestamp, mid_price, sum_bid_volume, sum_ask_volume, bid_ask_imbalance)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.executemany(query, data_list)
        mysql_conn.commit()
        cursor.close()
        print(f"Записано {len(data_list)} записей в MySQL.")
    except mysql.connector.Error as e:
        logging.error(f"Ошибка при записи в MySQL: {e}")

# Функция для извлечения сигнала от модели интервалов из Redis
async def get_interval_signal(redis_client):
    interval_signal = await redis_client.get("interval_signal")
    return int(interval_signal) if interval_signal is not None else 0  # Возвращаем 0, если сигнала нет

async def process_message(data, redis_client, data_to_save, last_save_time, mysql_conn):
    """Обработка полученного сообщения и сохранение нормализованных данных."""
    try:
        bids = np.array(data['b'][:8], dtype=float)
        asks = np.array(data['a'][:8], dtype=float)
        timestamp = datetime.utcnow()

        mid_price = (bids[0, 0] + asks[0, 0]) / 2
        sum_bid_volume = np.sum(bids[:, 1])
        sum_ask_volume = np.sum(asks[:, 1])
        imbalance = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume) if (sum_bid_volume + sum_ask_volume) > 0 else 0

        normalized_data = {
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "mid_price": min_max_normalize(mid_price, FIXED_BOUNDARIES["mid_price"]["min"], FIXED_BOUNDARIES["mid_price"]["max"]),
            "sum_bid_volume": min_max_normalize(sum_bid_volume, FIXED_BOUNDARIES["sum_bid_volume"]["min"], FIXED_BOUNDARIES["sum_bid_volume"]["max"]),
            "sum_ask_volume": min_max_normalize(sum_ask_volume, FIXED_BOUNDARIES["sum_ask_volume"]["min"], FIXED_BOUNDARIES["sum_ask_volume"]["max"]),
            "imbalance": min_max_normalize(imbalance, FIXED_BOUNDARIES["imbalance"]["min"], FIXED_BOUNDARIES["imbalance"]["max"])
        }

        await save_to_redis(redis_client, normalized_data, "normalized_order_book_stream")
        await save_to_redis(redis_client, normalized_data, "order_book_10min_cache")

        # Сохраняем необработанные данные для последующей записи в MySQL
        data_to_save.append((timestamp, mid_price, sum_bid_volume, sum_ask_volume, imbalance))

        # Проверяем, прошло ли 10 минут для записи в MySQL
        if datetime.utcnow() - last_save_time >= timedelta(minutes=10):
            await save_to_mysql(mysql_conn, data_to_save)
            data_to_save.clear()
            last_save_time = datetime.utcnow()
    
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Ошибка при обработке данных: {e}")
    except Exception as e:
        logging.error(f"Неизвестная ошибка: {e}")

async def fetch_order_book_data(redis_client, mysql_conn):
    """Получение данных из WebSocket и обработка сообщений."""
    SYMBOL = "btcusdt"
    DEPTH_URL = f"wss://fstream.binance.com/ws/{SYMBOL}@depth20@100ms"
    data_to_save = []
    last_save_time = datetime.utcnow()

    while True:
        connection_start_time = datetime.utcnow()  # Время начала соединения

        try:
            async with websockets.connect(DEPTH_URL) as websocket:
                while True:
                    # Завершаем соединение через 25 минут
                    if datetime.utcnow() - connection_start_time >= timedelta(minutes=25):
                        logging.info("Переподключение WebSocket через 25 минут.")
                        break
                    
                    message = await websocket.recv()
                    data = json.loads(message)
                    await process_message(data, redis_client, data_to_save, last_save_time, mysql_conn)

        except (websockets.exceptions.ConnectionClosedError, asyncio.TimeoutError) as e:
            logging.error(f"Ошибка соединения WebSocket: {e}. Повторное подключение...")
            await asyncio.sleep(5)

async def main():
    redis_client = await initialize_redis()
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    try:
        interval_signal = await get_interval_signal(redis_client)
        print("Received interval signal:", interval_signal)
        await fetch_order_book_data(redis_client, mysql_conn)
    finally:
        await redis_client.close()
        mysql_conn.close()

if __name__ == "__main__":
    asyncio.run(main())
