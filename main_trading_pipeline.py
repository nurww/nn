# main_trading_pipeline.py

import logging
import time
import subprocess
from datetime import datetime, timedelta
import threading

# Настройка логирования
logging.basicConfig(
    filename='main_trading_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # Задаем кодировку UTF-8
)

# Флаг для остановки выполнения
should_stop = False

# Функция для управления остановкой по пользовательскому вводу
def stop_on_user_input():
    global should_stop
    while True:
        user_input = input("Введите 'safe' для безопасного завершения работы: ")
        if user_input.strip().lower() == 'safe':
            logging.info("Команда 'safe' получена. Остановка после завершения текущего цикла.")
            should_stop = True
            break

# Функция для запуска скрипта с тайм-аутом и ретраями
def run_script_with_retries_and_timing(script_name, max_retries=3, timeout=50):
    attempt = 1
    while attempt <= max_retries:
        start_time = time.time()
        logging.info(f"Попытка {attempt} для запуска {script_name}")
        try:
            process = subprocess.run(['python', script_name], timeout=timeout)
            elapsed_time = time.time() - start_time
            logging.info(f"Время выполнения {script_name}: {elapsed_time:.2f} секунд")

            if process.returncode == 0:
                logging.info(f"{script_name} завершен успешно.")
                return True
            else:
                logging.warning(f"Попытка {attempt} завершилась неудачей для {script_name}")
        except subprocess.TimeoutExpired:
            logging.error(f"{script_name} превысил время выполнения {timeout} секунд и был остановлен.")

        attempt += 1
        time.sleep(5)  # Ожидание перед повторной попыткой

    logging.error(f"{script_name} не удалось запустить после {max_retries} попыток.")
    return False

# Основной процесс обработки данных
def main_process():
    global should_stop

    # Запускаем поток для отслеживания пользовательского ввода
    threading.Thread(target=stop_on_user_input, daemon=True).start()

    while not should_stop:
        current_time = datetime.utcnow()
        
        # Рассчитываем момент 51-й секунды текущей или следующей минуты
        next_execution_time = current_time.replace(second=51, microsecond=0)
        if current_time.second >= 51:
            next_execution_time += timedelta(minutes=1)
        
        time_to_wait = (next_execution_time - datetime.utcnow()).total_seconds()
        logging.info(f"Ждем до 51-й секунды: {next_execution_time} (ожидание {time_to_wait:.2f} секунд)")
        time.sleep(time_to_wait)

        logging.info(f"Начало цикла обработки данных: {next_execution_time}")

        cycle_start_time = time.time()

        # Запускаем скрипты по очереди
        if run_script_with_retries_and_timing('binance_connection.py'):
            if run_script_with_retries_and_timing('indicators_for_periods.py'):
                if run_script_with_retries_and_timing('data_processor.py'):
                    logging.info(f"Цикл обработки данных за {next_execution_time} завершен успешно.")
                else:
                    logging.error(f"Ошибка при выполнении data_processor.py за {next_execution_time}")
            else:
                logging.error(f"Ошибка при выполнении indicators_for_periods.py за {next_execution_time}")
        else:
            logging.error(f"Ошибка при выполнении binance_connection.py за {next_execution_time}")

        # Проверяем, нужно ли завершить выполнение перед началом нового цикла
        if should_stop:
            logging.info("Процесс был остановлен пользователем после завершения текущего цикла.")
            break

        # Проверка времени выполнения цикла
        elapsed_cycle_time = time.time() - cycle_start_time
        max_cycle_duration = 60  # Максимальное время выполнения в секундах
        if elapsed_cycle_time > max_cycle_duration:
            logging.warning(f"Цикл превысил допустимое время {max_cycle_duration} секунд и следующий запуск пропущен.")
            continue

        logging.info(f"Ожидание следующего запуска...")

if __name__ == '__main__':
    logging.info("Запуск основного скрипта для управления торговыми процессами...")
    main_process()
