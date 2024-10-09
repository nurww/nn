import subprocess
import threading
import time

data_written_event = threading.Event()

def run_binance_connection():
    print("Запуск скрипта binance_connection.py...")
    subprocess.run(['python', 'binance_connection.py'])
    print("Скрипт binance_connection.py завершил работу.")

def run_update_copies():
    print("Запуск скрипта update_copies.py...")
    subprocess.run(['python', 'update_copies.py'])
    print("Скрипт update_copies.py завершил работу.")

def run_indicators():
    print("Запуск скрипта indicators_for_periods.py...")
    subprocess.run(['python', 'indicators_for_periods.py'])
    print("Скрипт indicators_for_periods.py завершил работу.")

def run_data_processor():
    print("Запуск скрипта data_processor.py...")
    subprocess.run(['python', 'data_processor.py'])
    print("Скрипт data_processor.py завершил работу.")

def run_pipeline():
    while True:
        # Запускаем binance_connection
        run_binance_connection()
        
        # Проверяем, обновлены ли данные
        if data_written_event.is_set():
            # Если данные обновлены, запускаем остальные скрипты
            run_update_copies()
            run_indicators()
            run_data_processor()
            print("Цикл завершен, начинаем новый...")
        else:
            print("Данные не обновлены, ждем...")

        time.sleep(10)  # Небольшая пауза перед новым циклом

if __name__ == "__main__":
    run_pipeline()
