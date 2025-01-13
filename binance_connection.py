# binance_connection.py

import argparse
import aiohttp
import asyncio
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timezone, timedelta

# Семафор для ограничения параллельных запросов
semaphore = asyncio.Semaphore(3)

# Подключение к базе данных MySQL
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='binance_data',
            user='root',
            password='root'
        )
        if connection.is_connected():
            print("Соединение с MySQL установлено")
        return connection
    except Error as e:
        print(f"Ошибка при подключении к MySQL: {e}")
        return None

# Форматирование и вывод данных в виде таблицы
def format_and_print_data(data):
    return 
    # print("open_time (UTC)        | open_time (UTC+5)     | open_price | high_price | low_price | close_price | volume")
    # print("-" * 90)
    # for entry in data:
    #     open_time_utc = datetime.fromtimestamp(entry[0] / 1000, tz=timezone.utc)
    #     open_time_utc5 = open_time_utc + timedelta(hours=5)
    #     print(f"{open_time_utc.strftime('%Y-%m-%d %H:%M:%S')} | "
    #           f"{open_time_utc5.strftime('%Y-%m-%d %H:%M:%S')} | "
    #           f"{entry[1]} | {entry[2]} | {entry[3]} | {entry[4]} | {entry[5]}")

# Получение данных с Binance
async def get_binance_data(session, symbol, interval, start_time, limit=1000):
    current_time = datetime.utcnow()
    next_execution_time = current_time.replace(second=5, microsecond=0)
    if current_time.second >= 5:
        next_execution_time += timedelta(minutes=0)

    time_to_wait = (next_execution_time - datetime.utcnow()).total_seconds()
    print(f"Ждем до 5-й секунды: {next_execution_time} (ожидание {time_to_wait:.2f} секунд)")
    await asyncio.sleep(time_to_wait)
    print(f"- {interval} запрос на Binance в {datetime.utcnow()}")

    # Определяем endTime для завершенного интервала
    if interval == "1m":
        end_time = int((current_time - timedelta(minutes=1) + timedelta(hours=5)).timestamp() * 1000)
    elif interval == "5m":
        end_time = int((current_time - timedelta(minutes=current_time.minute % 5 or 5) + timedelta(hours=5)).timestamp() * 1000)
    elif interval == "15m":
        end_time = int((current_time - timedelta(minutes=current_time.minute % 15 or 15) + timedelta(hours=5)).timestamp() * 1000)
    elif interval == "1h":
        end_time = int((current_time - timedelta(hours=1) + timedelta(hours=5)).timestamp() * 1000)
    elif interval == "4h":
        end_time = int((current_time - timedelta(hours=current_time.hour % 4 or 4) + timedelta(hours=5)).timestamp() * 1000)
    elif interval == "1d":
        end_time = int((current_time - timedelta(days=1) + timedelta(hours=5)).timestamp() * 1000)
    else:
        end_time = current_time.timestamp() * 1000
        # raise ValueError("Неподдерживаемый интервал")

    if start_time >= end_time:
        print(f"Пропускаем загрузку для {interval}, данные актуальны.")
    else:
        async with semaphore:
            url = 'https://fapi.binance.com/fapi/v1/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Ошибка при получении данных: {response.status}")

                data = await response.json()
                print(f"current_time: {current_time} | Полученные данные для {interval} (end_time: {datetime.utcfromtimestamp(end_time / 1000)}):")
                format_and_print_data(data)  # Выводим данные в виде таблицы
                return data

# Проверка на допустимое время для загрузки данных
def is_time_to_fetch(interval, force):
    if force:
        print(f"Принудительная загрузка данных для {interval}.")
        return True
    
    now = datetime.utcnow()

    if interval == '1d' and not (now.minute % 15 == 0 and now.second >= 5 and now.microsecond >= 0):
        print("Ещё рано для загрузки данных за 1 день. Ждем 00:05, 15:05, 30:05, 45:05 и т.д.")
        return False
    elif interval == '4h' and not (now.minute % 15 == 0 and now.second >= 5 and now.microsecond >= 0):
        print("Ещё рано для загрузки данных за 4 часа. Ждем 00:05, 15:05, 30:05, 45:05 и т.д.")
        return False
    elif interval == '1h' and not (now.minute % 15 == 0 and now.second >= 5 and now.microsecond >= 0):
        print("Ещё рано для загрузки данных за 1 час. Ждем 00:05, 15:05, 30:05, 45:05 и т.д.")
        return False
    elif interval == '15m' and not (now.minute % 15 == 0 and now.second >= 5 and now.microsecond >= 0):
        print("Ещё рано для загрузки данных за 15 минут. Ждем 00:05, 15:05, 30:05, 45:05 и т.д.")
        return False
    elif interval == '5m' and not (now.minute % 5 == 0 and now.second >= 5 and now.microsecond >= 0):
        print("Ещё рано для загрузки данных за 5 минут. Ждем 00:05, 05:05, 10:05 и т.д.")
        return False
    elif interval == '1m' and not (now.second >= 5 and now.microsecond >= 0):
        print("Ещё рано для загрузки данных за 1 минуту. Ждем 00:05 каждой минуты.")
        return False
    
    return True

# Запись данных в MySQL
def save_to_mysql(connection, data, data_interval):
    cursor = connection.cursor()
    sql_insert_query = """
    INSERT INTO binance_klines (open_time, open_price, high_price, low_price, close_price, volume, close_time, data_interval)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        values = []
        for entry in data:
            open_time = datetime.fromtimestamp(entry[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("SELECT COUNT(*) FROM binance_klines WHERE open_time = %s AND data_interval = %s", (open_time, data_interval))
            result = cursor.fetchone()
            
            if result[0] == 0:
                close_time = datetime.fromtimestamp(entry[6] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                values.append((open_time, entry[1], entry[2], entry[3], entry[4], entry[5], close_time, data_interval))
        
        if values:
            cursor.executemany(sql_insert_query, values)
            connection.commit()
            print(f"{cursor.rowcount} строк(и) добавлено для интервала {data_interval}")
        else:
            print("Данные уже присутствуют в базе, добавление пропущено")
    
    except Error as e:
        connection.rollback()
        print(f"Ошибка при записи данных в MySQL: {e}")
        raise e

# Получение последней метки времени
def get_last_timestamp_from_db(connection, data_interval):
    cursor = connection.cursor()
    cursor.execute(f"SELECT MAX(open_time) FROM binance_klines WHERE data_interval = '{data_interval}'")
    result = cursor.fetchone()
    if result[0] is None:
        return None
    return int(result[0].timestamp() * 1000)

# Постоянная загрузка данных с проверкой допустимого времени
async def fetch_and_store_data_with_timing(session, connection, symbol, interval, start_time, force):
    if is_time_to_fetch(interval, force):
        while True:
            data = await get_binance_data(session, symbol, interval, start_time)
            
            if not data:
                print(f"Данные для {interval} завершены на временной метке {start_time}.")
                break  # Если данных больше нет, выходим из цикла

            save_to_mysql(connection, data, interval)
            
            last_timestamp = data[-1][0]
            start_time = last_timestamp + 1

            if len(data) < 1000:
                break
    else:
        print(f"Пропускаем загрузку для интервала {interval}, время еще не наступило.")

# Постоянная загрузка данных
async def continuous_data_fetch(symbol, intervals, year, force):
    connection = connect_to_db()
    if connection is None:
        return
    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for interval in intervals:
                start_time = get_last_timestamp_from_db(connection, interval) or int(datetime(year, 9, 15).timestamp() * 1000)
                tasks.append(fetch_and_store_data_with_timing(session, connection, symbol, interval, start_time, force))
            await asyncio.gather(*tasks)
    finally:
        if connection.is_connected():
            connection.close()
            print("Соединение с MySQL закрыто")

# Парсинг аргументов командной строки
def parse_arguments():
    parser = argparse.ArgumentParser(description="Загрузка данных с Binance с интервалами")
    parser.add_argument("--force", action="store_true", help="Принудительная загрузка данных для всех интервалов")
    return parser.parse_args()

# Запуск программы
if __name__ == "__main__":
    args = parse_arguments()
    symbol = 'BTCUSDT'
    intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
    year = 2024
    print("\n")
    print("_____________# binance_connection.py")
    asyncio.run(continuous_data_fetch(symbol, intervals, year, args.force))
    print("\n")
    print("__________________________||||||||")
