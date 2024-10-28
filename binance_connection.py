import argparse
import aiohttp
import asyncio
import mysql.connector
from mysql.connector import Error
from tqdm import tqdm
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

# Получение данных с Binance
async def get_binance_data(session, symbol, interval, start_time, limit=1000):
    async with semaphore:
        url = 'https://fapi.binance.com/fapi/v1/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'limit': limit
        }
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Ошибка при получении данных: {response.status}")
            return await response.json()

# Проверка на допустимое время для загрузки данных
def is_time_to_fetch(interval, force):
    if force:
        print(f"Принудительная загрузка данных для {interval}.")
        return True
    
    now = datetime.utcnow()
    if interval == '1d' and not (now.hour == 23 and now.minute == 50 and now.second >= 50):
        print("Ещё рано для загрузки данных за 1 день. Ждем 23:59:57 UTC.")
        return False
    elif interval == '4h' and not (now.hour % 4 == 3 and now.minute == 59 and now.second >= 50):
        print("Ещё рано для загрузки данных за 4 часа. Ждем 03:59:50, 07:59:50 и т.д.")
        return False
    elif interval == '1h' and not (now.minute == 59 and now.second >= 50):
        print("Ещё рано для загрузки данных за 1 час. Ждем 59 минут и 50 секунд.")
        return False
    elif interval == '15m' and not (now.minute % 15 == 14 and now.second >= 50):
        print("Ещё рано для загрузки данных за 15 минут. Ждем 14 минут и 50 секунд в каждом 15-минутном интервале.")
        return False
    elif interval == '5m' and not (now.minute % 5 == 4 and now.second >= 50):
        print("Ещё рано для загрузки данных за 5 минут. Ждем 4 минуты и 50 секунд в каждом 5-минутном интервале.")
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

# Проверка на дубликаты
def check_for_duplicates(connection):
    cursor = connection.cursor()
    query = """
    SELECT open_time, data_interval, COUNT(*)
    FROM binance_klines
    GROUP BY open_time, data_interval
    HAVING COUNT(*) > 1;
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    if rows:
        print("Дубликаты найдены:")
        for row in rows:
            print(f"open_time: {row[0]}, interval: {row[1]}, количество: {row[2]}")
    else:
        print("Дубликатов не найдено.")

# Проверка на пропуски данных
def check_for_gaps(connection, data_interval):
    cursor = connection.cursor()
    cursor.execute(f"SELECT open_time FROM binance_klines WHERE data_interval = '{data_interval}' ORDER BY open_time")
    rows = cursor.fetchall()
    if len(rows) < 2:
        print(f"Недостаточно данных для проверки интервала {data_interval}")
        return
    expected_diff = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
    }
    previous_time = rows[0][0]
    for row in rows[1:]:
        current_time = row[0]
        if expected_diff[data_interval] is not None:
            if current_time - previous_time > expected_diff[data_interval]:
                print(f"Пропущенные данные между {previous_time} и {current_time} для интервала {data_interval}")
        previous_time = current_time

# Постоянная загрузка данных с проверкой допустимого времени
async def fetch_and_store_data_with_timing(session, connection, symbol, interval, start_time, force):
    if is_time_to_fetch(interval, force):
        data = await get_binance_data(session, symbol, interval, start_time)
        save_to_mysql(connection, data, interval)
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
                start_time = get_last_timestamp_from_db(connection, interval) or int(datetime(year, 1, 1).timestamp() * 1000)
                tasks.append(fetch_and_store_data_with_timing(session, connection, symbol, interval, start_time, force))
            await asyncio.gather(*tasks)
    finally:
        if connection.is_connected():
            for interval in intervals:
                check_for_duplicates(connection)
                check_for_gaps(connection, interval)
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
    asyncio.run(continuous_data_fetch(symbol, intervals, year, args.force))
