# orderbook_data.py

import asyncio
import websockets
import json
import redis
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import pooling
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Параметры для подключения к Redis и MySQL
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

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
except mysql.connector.Error as e:
    logging.error(f"Ошибка при создании пула MySQL: {e}")
    exit(1)

# Функция для сохранения данных в Redis
async def save_to_redis(data):
    redis_client.rpush("order_book_stream", json.dumps(data))
    # Оставляем только последние 600 записей
    redis_client.ltrim("order_book_stream", -600, -1)

# Функция для сохранения агрегированных данных в MySQL
def save_to_mysql_price_levels(aggregated_data):
    connection = mysql_pool.get_connection()
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO price_volume_levels (price, total_volume, side, avg_volume, activity_indicator, last_update)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                total_volume = VALUES(total_volume),
                avg_volume = VALUES(avg_volume),
                activity_indicator = VALUES(activity_indicator),
                last_update = VALUES(last_update)
        """
        cursor.executemany(query, aggregated_data)
        connection.commit()
        logging.info(f"{len(aggregated_data)} уровней цены обновлено в MySQL")
    except mysql.connector.Error as e:
        logging.error(f"Ошибка при записи в MySQL: {e}")
    finally:
        cursor.close()
        connection.close()

# Функция для получения данных ордербука и их обработки
async def analyze_order_book():
    symbol = "btcusdt"
    depth_url = f"wss://fstream.binance.com/ws/{symbol}@depth20@100ms"

    async with websockets.connect(depth_url) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            bids = data['b']
            asks = data['a']
            
            # Расчет параметров для каждого уровня цены
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            processed_data = []

            for side, levels in (("bid", bids[:8]), ("ask", asks[:8])):
                for price, volume in levels:
                    processed_data.append({
                        "timestamp": timestamp,
                        "price": float(price),
                        "volume": float(volume),
                        "side": side
                    })

            # Сохранение обработанных данных в Redis
            await save_to_redis(processed_data)

            # Задержка на 100 мс
            await asyncio.sleep(0.1)

# Функция для периодической агрегации данных из Redis и записи в MySQL
async def aggregate_and_save_to_mysql(interval=2):
    while True:
        # Получение всех данных из Redis
        order_book_data = [json.loads(item) for item in redis_client.lrange("order_book_stream", 0, -1)]
        
        # Агрегация данных для каждого уровня цены
        aggregated_data = {}
        
        for update in order_book_data:
            for entry in update:
                price = entry['price']
                side = entry['side']
                volume = entry['volume']
                
                if (price, side) not in aggregated_data:
                    aggregated_data[(price, side)] = {
                        "total_volume": 0,
                        "count": 0
                    }
                
                aggregated_data[(price, side)]['total_volume'] += volume
                aggregated_data[(price, side)]['count'] += 1

        # Подготовка данных для записи в MySQL
        mysql_data = []
        for (price, side), data in aggregated_data.items():
            avg_volume = data['total_volume'] / data['count']
            activity_indicator = data['count'] / len(order_book_data)
            last_update = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            mysql_data.append((price, data['total_volume'], side, avg_volume, activity_indicator, last_update))
        
        # Сохранение агрегированных данных в MySQL
        save_to_mysql_price_levels(mysql_data)

        # Задержка перед следующей агрегацией
        await asyncio.sleep(interval)

# Основная функция запуска всех задач
async def main():
    # Запускаем две задачи параллельно: сбор данных ордербука и агрегацию
    await asyncio.gather(
        analyze_order_book(),
        aggregate_and_save_to_mysql()
    )

# Запуск программы
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Программа остановлена пользователем")
