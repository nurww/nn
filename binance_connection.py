import aiohttp
import asyncio
import os
import csv
from datetime import datetime
import sys
import time

data_written_event = asyncio.Event()

async def get_binance_data(session, symbol, interval, start_time, limit=1000):
    # URL для фьючерсных данных Binance
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'limit': limit
    }
    async with session.get(url, params=params) as response:
        return await response.json()

def safe_save_to_csv(filename, data):
    existing_rows = set()
    if os.path.exists(filename):
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                existing_rows.add(row[0])

    new_data_written = False  # Флаг для отслеживания новых данных
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if os.path.getsize(filename) == 0:
            writer.writerow(['Время открытия (UTC)', 'Цена открытия', 'Максимум', 'Минимум', 'Цена закрытия', 'Объем'])
        for entry in data:
            entry_time = datetime.utcfromtimestamp(entry[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            if entry_time not in existing_rows:
                writer.writerow([
                    entry_time,
                    entry[1],
                    entry[2],
                    entry[3],
                    entry[4],
                    entry[5]
                ])
                new_data_written = True
    if new_data_written:
        data_written_event.set()  # Сигнализируем, что данные обновлены
    else:
        data_written_event.clear()  # Если данных нет, сбрасываем событие

def get_last_timestamp(filename):
    if not os.path.isfile(filename):
        return None
    with open(filename, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        if len(lines) > 1:
            last_line = lines[-1]
            last_timestamp = last_line.split(',')[0]
            return int(datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    return None

def display_progress(progress_info):
    sys.stdout.write("\033[H\033[J")
    for interval, (year, total_loaded, total_to_load) in progress_info.items():
        total_to_load = max(total_to_load, total_loaded + 1)
        progress_percentage = int((total_loaded / total_to_load) * 100)
        bar_length = int((progress_percentage / 100) * 20)
        bar = 'I' * bar_length + '.' * (20 - bar_length)
        
        if total_to_load > 1000:
            total_to_load_display = f"{total_to_load // 1000}k"
        else:
            total_to_load_display = str(total_to_load)
        
        sys.stdout.write(f"{year}y {interval}:\t{bar} {total_loaded} / {total_to_load_display}\n")
    sys.stdout.flush()

async def continuous_data_fetch(symbol, intervals, year=2024, directory='periods_data'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    interval_mapping = {
        '1M': '1mo',
        '1m': '1min'
    }

    progress_info = {interval: (year, 0, 0) for interval in intervals}

    delay = 1  # Начальная задержка
    max_delay = 60  # Максимальная задержка между попытками переподключения

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                for interval in intervals:
                    filename = os.path.join(directory, f'{symbol}_{interval_mapping.get(interval, interval)}.csv')
                    last_timestamp = get_last_timestamp(filename)
                    total_loaded = 0

                    if last_timestamp is None:
                        start_time = int(datetime(year, 1, 1, 0, 0).timestamp() * 1000)
                    else:
                        start_time = last_timestamp + 1

                    while True:
                        data = await get_binance_data(session, symbol, interval, start_time)
                        if not data:
                            break

                        safe_save_to_csv(filename, data)
                        total_loaded += len(data)
                        last_timestamp = data[-1][0]
                        start_time = last_timestamp + 1

                        progress_info[interval] = (year, total_loaded, total_loaded + 1000)
                        # display_progress(progress_info)

                        print("В binance_connection.py что-то происходит ...")

                        await data_written_event.wait()
                        data_written_event.clear()

                        delay = 1  # Сброс задержки после успешного подключения
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Ошибка соединения: {e}. Переподключение через {delay} секунд...")
            time.sleep(delay)
            delay = min(max_delay, delay * 2)  # Экспоненциальное увеличение задержки
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            break

# Запуск программы
symbol = 'BTCUSDT'
# intervals = ['1M', '1d', '4h', '1h', '15m', '5m', '1m']
intervals = ['1m', '5m', '15m', '1h', '4h', '1d', '1M']
year = 2024

asyncio.run(continuous_data_fetch(symbol, intervals, year))
