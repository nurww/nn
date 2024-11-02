# redis_normalizer.py

import redis
import numpy as np
import time
from datetime import datetime
import json

# Настройка подключения к Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Функция для нормализации данных с фиксированными границами
def min_max_normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

# Задание фиксированных границ для различных полей
fixed_boundaries = {
    "mid_price": {"min": 40000, "max": 85000},
    "price": {"min": 40000, "max": 85000},
    "sum_bid_volume": {"min": 0, "max": 10000},
    "sum_ask_volume": {"min": 0, "max": 10000},
    "total_volume": {"min": 0, "max": 10000},
    "avg_volume": {"min": 0, "max": 1000},
    "activity_indicator": {"min": 0, "max": 0.2},
    "imbalance": {"min": -1, "max": 1}
}

# Основная функция обработки данных
def process_and_store_data():
    data_window = []  # Хранение нормализованных данных (15 минут)

    while True:
        # Получение данных из Redis
        raw_data = redis_client.lrange("order_book_stream", -1, -1)
        if not raw_data:
            time.sleep(0.1)
            continue

        # Преобразование строки в словарь
        latest_data = json.loads(raw_data[0])

        # Нормализация данных с учетом фиксированных границ
        normalized_data = {}
        for key, value in latest_data.items():
            if key in fixed_boundaries:
                normalized_data[key] = min_max_normalize(
                    value, 
                    fixed_boundaries[key]['min'], 
                    fixed_boundaries[key]['max']
                )
        
        # Добавление временной метки
        normalized_data['timestamp'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # Запись нормализованных данных в Redis
        redis_client.rpush("normalized_order_book_stream", json.dumps(normalized_data))

        # Ограничение на 900 записей (15 минут при 100 мс)
        redis_client.ltrim("normalized_order_book_stream", -900, -1)

        # Задержка для обновления каждые 100 мс
        time.sleep(0.1)

if __name__ == "__main__":
    process_and_store_data()
