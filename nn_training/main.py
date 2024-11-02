# main.py

import asyncio
from data_processing.fetch_data import fetch_data_from_redis
from data_processing.process_data import process_data
from model.model_training import train_model
from candlestick_volume_plot import main as plot_main

async def main_loop():
    while True:
        # Подготовка данных и обучение модели
        data = await fetch_data_from_redis()
        processed_data = process_data(data)
        train_model(processed_data)

        # Построение графика
        # await plot_main()

        # Задержка перед следующим циклом
        await asyncio.sleep(0.1)  # Запуск цикла раз в минуту

# Запуск цикла
if __name__ == "__main__":
    asyncio.run(main_loop())
