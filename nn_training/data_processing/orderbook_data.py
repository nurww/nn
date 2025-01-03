# orderbook_data.py

import asyncio
import websockets
import json
import redis.asyncio as aioredis
import numpy as np
import logging
from datetime import datetime, timedelta
import mysql.connector
import time  # Добавьте импорт модуля time
from decimal import Decimal

# Конфигурация для Redis и MySQL
REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'db': 0}
MYSQL_CONFIG = {
    'host': 'localhost',
    'database': 'binance_data',
    'user': 'root',
    'password': 'root'
}

FIXED_BOUNDARIES = {
    "mid_price": {"min": 0, "max": 115000},
    "sum_bid_volume": {"min": 0, "max": 30000},
    "sum_ask_volume": {"min": 0, "max": 30000},
    "imbalance": {"min": -1, "max": 1}
}

# Настройка логирования
logging.basicConfig(
    filename='logs/normalized_order_book_detailed.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

def min_max_normalize(value, min_val, max_val, precision=25):
    normalized = (value - min_val) / (max_val - min_val)
    clipped = np.clip(normalized, 0, 1)
    return round(clipped, precision)


async def save_to_redis(redis_client, data, key):
    try:
        await redis_client.rpush(key, json.dumps(data))
        # Если ключ "normalized_order_book_stream", ограничиваем количество записей до 1500
        if key == "normalized_order_book_stream":
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

async def process_message(data, redis_client, mysql_conn):
    """Обработка полученного сообщения и сохранение нормализованных данных."""
    try:
        # Используем Decimal для повышения точности
        bids = np.array(data['b'][:8], dtype=object)
        asks = np.array(data['a'][:8], dtype=object)

        # Преобразуем значения в Decimal для повышения точности
        bids = np.array([[Decimal(bid[0]), Decimal(bid[1])] for bid in bids], dtype=object)
        asks = np.array([[Decimal(ask[0]), Decimal(ask[1])] for ask in asks], dtype=object)

        timestamp = datetime.utcnow()

        # Высокоточная средняя цена
        mid_price = (bids[0, 0] + asks[0, 0]) / Decimal(2)

        # Логирование для проверки
        # logging.info(f"bids (high precision): {bids}")
        # logging.info(f"asks (high precision): {asks}")
        # logging.info(f"mid_price (high precision): {mid_price}")

        # Оставляем остальную логику неизменной
        sum_bid_volume = np.sum([bid[1] for bid in bids])
        sum_ask_volume = np.sum([ask[1] for ask in asks])
        imbalance = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume) if (sum_bid_volume + sum_ask_volume) > 0 else Decimal(0)

        normalized_data = {
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "mid_price": min_max_normalize(float(mid_price), FIXED_BOUNDARIES["mid_price"]["min"], FIXED_BOUNDARIES["mid_price"]["max"]),
            "sum_bid_volume": min_max_normalize(float(sum_bid_volume), FIXED_BOUNDARIES["sum_bid_volume"]["min"], FIXED_BOUNDARIES["sum_bid_volume"]["max"]),
            "sum_ask_volume": min_max_normalize(float(sum_ask_volume), FIXED_BOUNDARIES["sum_ask_volume"]["min"], FIXED_BOUNDARIES["sum_ask_volume"]["max"]),
            "imbalance": min_max_normalize(float(imbalance), FIXED_BOUNDARIES["imbalance"]["min"], FIXED_BOUNDARIES["imbalance"]["max"])
        }


        # Сохраняем данные в Redis
        await save_to_redis(redis_client, normalized_data, "normalized_order_book_stream")
        await save_to_redis(redis_client, normalized_data, "order_book_10min_cache")

        # Замер времени перед началом обработки данных
        start_time = time.time()

        # Проверяем, достигли ли 10 минут (примерно 6000 записей, если данные поступают 5 раз в 1 сек.)
        cache_size = await redis_client.llen("order_book_10min_cache")
        if cache_size >= 6000:  # 10 минут данных
            # Извлекаем все данные из "order_book_10min_cache"
            raw_data = await redis_client.lrange("order_book_10min_cache", 0, -1)
            # Преобразуем их обратно в Python-объекты
            data_list = [json.loads(record) for record in raw_data]

            # Преобразуем для записи в MySQL
            mysql_data = [
                (
                    datetime.strptime(record["timestamp"], '%Y-%m-%d %H:%M:%S.%f'),
                    record["mid_price"],
                    record["sum_bid_volume"],
                    record["sum_ask_volume"],
                    record["imbalance"]
                )
                for record in data_list
            ]

            # Сохраняем данные в MySQL
            await save_to_mysql(mysql_conn, mysql_data)

            # Очищаем кеш в Redis после записи
            await redis_client.delete("order_book_10min_cache")

        # Замер времени после выполнения обработки
        elapsed_time = time.time() - start_time
        # logging.info(f"Время выполнения блока обработки: {elapsed_time:.4f} секунд.")
    
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Ошибка при обработке данных: {e}")
    except Exception as e:
        logging.error(f"Неизвестная ошибка: {e}")

async def fetch_order_book_data(redis_client, mysql_conn):
    SYMBOL = "btcusdt"
    DEPTH_URL = f"wss://fstream.binance.com/ws/{SYMBOL}@depth20@100ms"

    while True:
        connection_start_time = datetime.utcnow()  # Время начала соединения

        try:
            async with websockets.connect(DEPTH_URL) as websocket:
                while True:
                    # Проверяем, прошло ли 20 минут
                    if datetime.utcnow() - connection_start_time >= timedelta(minutes=20):
                        logging.info("Переключение на новый Redis клиент через 20 минут.")

                        total_runtime = datetime.utcnow() - connection_start_time
                        logging.info(f"Redis клиент работал {total_runtime}. Переключаемся на новый.")

                        # Создаем новый redis_client
                        next_redis_client = await initialize_redis()

                        # Закрываем старый redis_client
                        try:
                            await redis_client.close()
                        except Exception as e:
                            logging.error(f"Ошибка при закрытии старого Redis клиента: {e}")

                        # Переключаемся на новый redis_client
                        redis_client = next_redis_client
                        connection_start_time = datetime.utcnow()  # Перезапускаем таймер соединения
                        logging.info("Переключение на новый Redis клиент завершено.")
                        break  # Переподключение к WebSocket

                    message = await websocket.recv()
                    data = json.loads(message)
                    await process_message(data, redis_client, mysql_conn)

        except (websockets.exceptions.ConnectionClosedError, asyncio.TimeoutError) as e:
            logging.error(f"Ошибка соединения WebSocket: {e}. Повторное подключение...")
            await asyncio.sleep(5)
        except websockets.ConnectionClosed as e:
            logging.error(f"Соединение закрыто: {e}")
            break

async def main():
    redis_client = await initialize_redis()
    mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
    try:
        # interval_signal = await get_interval_signal(redis_client)
        # print("Received interval signal:", interval_signal)
        await fetch_order_book_data(redis_client, mysql_conn)
    finally:
        await redis_client.close()
        mysql_conn.close()

if __name__ == "__main__":
    asyncio.run(main())
