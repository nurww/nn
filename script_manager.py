# script_manager.py

import os
import subprocess
import time
import logging

# Создание папки logs, если ее нет
if not os.path.exists('logs'):
    os.makedirs('logs')

# Настройка логирования
logging.basicConfig(
    filename='logs/script_manager.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Остальной код остается без изменений
def start_script(script_name):
    logging.info(f"Запуск {script_name}")
    return subprocess.Popen(['python', script_name])

def manage_scripts():
    processes = {
        'orderbook_data': start_script('orderbook_data.py'),
        'orderbook_aggregator': start_script('orderbook_aggregator.py'),
        # 'realtime_inference': start_script('realtime_inference.py')  # Закомментировано для теста
    }

    try:
        while True:
            for name, process in processes.items():
                if process.poll() is not None:
                    logging.warning(f"{name} завершился. Перезапуск...")
                    processes[name] = start_script(f"{name}.py")
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Остановка всех процессов...")
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.wait()
        logging.info("Все процессы остановлены.")

if __name__ == "__main__":
    logging.info("Запуск script_manager")
    manage_scripts()
