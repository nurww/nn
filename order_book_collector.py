# order_book_collector.py

import asyncio
import websockets
import json
import redis
import mysql.connector
from mysql.connector import pooling, Error
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Параметры для подключения к Redis и MySQL
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Настройка пула соединений MySQL
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'binance_data',
    'pool_name': 'mysql_pool',
    'pool_size': 5
}

try:
    mysql_pool = pooling.MySQLConnectionPool(**MYSQL_CONFIG)
except Error as e:
    logging.error(f"Ошибка при создании пула MySQL: {e}")
    exit(1)

# Функция для сохранения буфера в MySQL
def save_to_mysql(buffer):
    connection = mysql_pool.get_connection()
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO order_book_data (timestamp, mid_price, sum_bid_volume, sum_ask_volume, imbalance)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.executemany(query, buffer)
        connection.commit()
        logging.info(f"{len(buffer)} записей добавлено в MySQL")
    except Error as e:
        logging.error(f"Ошибка при записи в MySQL: {e}")
    finally:
        cursor.close()
        connection.close()

# Асинхронная функция для обработки ордербука и сохранения данных в Redis
async def analyze_order_book():
    symbol = "btcusdt"
    depth_url = f"wss://fstream.binance.com/ws/{symbol}@depth20@100ms"

    while True:
        try:
            async with websockets.connect(depth_url) as websocket_depth:
                buffer = []
                while True:
                    message_depth = await websocket_depth.recv()
                    data_depth = json.loads(message_depth)
                    bids = data_depth['b']
                    asks = data_depth['a']

                    # Расчет средней цены и дисбаланса объемов
                    mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
                    sum_bid_volume = sum(float(bid[1]) for bid in bids[:5])
                    sum_ask_volume = sum(float(ask[1]) for ask in asks[:5])
                    imbalance = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume)

                    # Текущее время
                    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

                    # Добавление в Redis и сохранение в буфер
                    redis_client.rpush("order_book_stream", json.dumps({
                        "timestamp": timestamp,
                        "mid_price": mid_price,
                        "sum_bid_volume": sum_bid_volume,
                        "sum_ask_volume": sum_ask_volume,
                        "imbalance": imbalance
                    }))

                    # Поддерживаем последние 600 записей в Redis
                    redis_client.ltrim("order_book_stream", -600, -1)
                    buffer.append((timestamp, mid_price, sum_bid_volume, sum_ask_volume, imbalance))

                    # Если буфер достиг 600 записей, отправляем в MySQL
                    if len(buffer) >= 600:
                        save_to_mysql(buffer)
                        buffer.clear()

                    await asyncio.sleep(0.1)
        except websockets.ConnectionClosedError as e:
            logging.warning(f"Соединение закрыто, повторная попытка через 5 секунд: {e}")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Ошибка при подключении к вебсокету: {e}")
            await asyncio.sleep(5)

# Запуск программы
if __name__ == "__main__":
    try:
        asyncio.run(analyze_order_book())
    except KeyboardInterrupt:
        logging.info("Программа остановлена пользователем")
