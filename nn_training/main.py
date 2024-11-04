# main.py

import asyncio
from data_processing.orderbook_data import analyze_order_book, initialize_redis  # Подключение функции для получения данных
from data_processing.fetch_data import fetch_data_from_redis
from data_processing.process_data import process_data
from model.model_training import train_model

async def main_loop():
    # Инициализация Redis
    redis_client = await initialize_redis()

    # Запуск сбора данных параллельно с обработкой и обучением
    task_collect_data = asyncio.create_task(analyze_order_book(redis_client))

    try:
        while True:
            # Подготовка данных и обучение модели
            data = await fetch_data_from_redis()
            processed_data = process_data(data)
            train_model(processed_data)

            # Задержка перед следующим циклом
            await asyncio.sleep(0.1)  # Запуск цикла раз в минуту
    except asyncio.CancelledError:
        print("Завершение основного цикла.")
    finally:
        task_collect_data.cancel()
        await redis_client.close()

# Запуск цикла
if __name__ == "__main__":
    asyncio.run(main_loop())
