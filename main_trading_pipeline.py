import subprocess
import asyncio
import threading
import time

data_written_event = threading.Event()

# Переменная для контроля выполнения binance_connection.py
pause_binance_connection = threading.Event()

async def run_binance_connection():
    while True:
        print("Запуск получения данных с Binance...")
        await asyncio.create_subprocess_exec('python', 'binance_connection.py')
        
        # Ждем, пока данные обновлены и запущены другие скрипты
        await data_written_event.wait()  # Ждём события обновления данных
        print("Данные с Binance обновлены.")

        pause_binance_connection.clear()  # Ждем завершения других скриптов
        pause_binance_connection.wait()   # Когда другие скрипты завершены, продолжаем
        
        print("Возобновляем binance_connection.py...")

def run_update_copies():
    data_written_event.wait()  # Ждем события от binance_connection
    print("Запуск скрипта update_copies.py...")
    subprocess.run(['python', 'update_copies.py'])
    print("Копии файлов обновлены.")
    pause_binance_connection.set()  # Разрешаем binance_connection продолжить работу

def run_indicators():
    print("Запуск скрипта indicators_for_periods.py...")
    subprocess.run(['python', 'indicators_for_periods.py'])
    print("Индикаторы добавлены.")

def run_data_processor():
    print("Запуск скрипта data_processor.py...")
    subprocess.run(['python', 'data_processor.py'])
    print("Данные обработаны.")

async def run_pipeline():
    # Запускаем binance_connection в фоновом режиме
    threading.Thread(target=lambda: asyncio.run(run_binance_connection())).start()

    while True:
        # Как только данные обновлены, запускаем другие скрипты
        threading.Thread(target=run_update_copies).start()  # Копируем файлы
        threading.Thread(target=run_indicators).start()      # Добавляем индикаторы
        threading.Thread(target=run_data_processor).start()  # Обрабатываем данные
        
        time.sleep(10)  # Добавляем небольшую паузу перед следующим циклом

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_pipeline())
