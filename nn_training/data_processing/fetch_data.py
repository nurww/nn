import aioredis
import json
import asyncio
from config import REDIS_CONFIG

async def fetch_data_from_redis(redis_key="normalized_order_book_stream"):
    redis_client = await aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])
    
    # Извлечение последних 900 записей из Redis
    data = await redis_client.lrange(redis_key, -1500, -1)
    data = [json.loads(item) for item in data]  # Преобразование строки JSON в словарь
    
    await redis_client.close()
    return data

# Запуск для проверки
if __name__ == "__main__":
    asyncio.run(fetch_data_from_redis())
