import mysql.connector
from mysql.connector import pooling
from datetime import datetime
import redis
import json
import time
import logging

# Настройка логирования
logging.basicConfig(
    filename='logs/orderbook_aggregator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Подключение к основному Redis
main_redis = redis.StrictRedis(host='localhost', port=6379, db=0)

# Пул соединений с MySQL
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'binance_data',
    'pool_name': 'mysql_pool',
    'pool_size': 5
}
mysql_pool = pooling.MySQLConnectionPool(**MYSQL_CONFIG)

# Функция для агрегации данных и записи в MySQL
def aggregate_and_save_to_mysql():
    # Получение данных из основного Redis
    order_book_data = [json.loads(item) for item in main_redis.lrange("order_book_stream", 0, -1)]
    
    # Словарь для накопления объемов по уровням цены
    aggregated_data = {}
    total_bid_volume = 0
    total_ask_volume = 0

    for update in order_book_data:
        for side_key, side in [("bids", "bid"), ("asks", "ask")]:  # Используем только "bid" и "ask"
            for price, volume in update[side_key][:8]:  # Используем 8 уровней
                price = float(price)
                volume = float(volume)
                key = (price, side)

                if key not in aggregated_data:
                    aggregated_data[key] = {"total_volume": 0, "count": 0}

                aggregated_data[key]["total_volume"] += volume
                aggregated_data[key]["count"] += 1

                # Суммируем объемы для расчета дисбаланса
                if side == "bid":
                    total_bid_volume += volume
                else:
                    total_ask_volume += volume

    # Рассчитываем imbalance
    if total_bid_volume + total_ask_volume > 0:
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    else:
        imbalance = 0

    # Подготовка данных для записи в MySQL
    mysql_data = []
    for (price, side), data in aggregated_data.items():
        avg_volume = data["total_volume"] / data["count"]
        activity_indicator = data["count"] / len(order_book_data)
        last_update = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        mysql_data.append((price, data["total_volume"], side, avg_volume, activity_indicator, last_update, imbalance))

    # Запись в MySQL
    connection = mysql_pool.get_connection()
    cursor = connection.cursor()
    query = """
        INSERT INTO price_volume_levels (price, total_volume, side, avg_volume, activity_indicator, last_update, imbalance)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            total_volume = VALUES(total_volume),
            avg_volume = VALUES(avg_volume),
            activity_indicator = VALUES(activity_indicator),
            last_update = VALUES(last_update),
            imbalance = VALUES(imbalance)
    """
    cursor.executemany(query, mysql_data)
    connection.commit()
    cursor.close()
    connection.close()

while True:
    logging.info("Начало агрегации данных.")
    aggregate_and_save_to_mysql()
    logging.info("Завершение агрегации данных.")
    time.sleep(2)
