# fetch_data.py

import redis.asyncio as redis
import json
import asyncio

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

async def fetch_data_from_redis(redis_key="normalized_order_book_stream"):
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )
    
    # Извлечение последних 900 записей из Redis
    data = await redis_client.lrange(redis_key, -1500, -1)
    data = [json.loads(item) for item in data]  # Преобразование строки JSON в словарь
    
    await redis_client.aclose()
    return data

# Запуск для проверки
if __name__ == "__main__":
    asyncio.run(fetch_data_from_redis())
