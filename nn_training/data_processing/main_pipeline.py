import asyncio
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    filename='main_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Импорты модулей для запуска
from real_time_prediction import run_real_time_prediction
from trading_decision_maker import run_trading_decision_maker
from retrospective_training import run_retrospective_training

async def main_pipeline():
    logging.info("Запуск основного процесса торговли и анализа...")
    
    try:
        while True:
            start_time = datetime.utcnow()
            logging.info(f"Запуск нового цикла обработки данных в {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Запуск модуля прогнозирования в реальном времени
            await run_real_time_prediction()

            # Запуск модуля для принятия торговых решений
            await run_trading_decision_maker()

            # Запуск модуля для ретроспективного обучения
            await run_retrospective_training()

            # Логирование завершения текущего цикла
            end_time = datetime.utcnow()
            cycle_duration = (end_time - start_time).total_seconds()
            logging.info(f"Цикл обработки завершен. Длительность: {cycle_duration:.2f} секунд.")

            # Задержка перед следующим запуском
            await asyncio.sleep(0.1)  # 100 мс
    except Exception as e:
        logging.error(f"Ошибка в основном процессе: {e}")

if __name__ == "__main__":
    asyncio.run(main_pipeline())
