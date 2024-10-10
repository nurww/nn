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
            
            # Проверка на существование записи перед добавлением
            cursor.execute("SELECT COUNT(*) FROM binance_klines WHERE open_time = %s AND data_interval = %s", (open_time, data_interval))
            result = cursor.fetchone()
            
            if result[0] == 0:  # Если записи нет, добавляем ее
                close_time = datetime.fromtimestamp(entry[6] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                values.append((open_time, entry[1], entry[2], entry[3], entry[4], entry[5], close_time, data_interval))
        
        if values:  # Если есть что добавить
            cursor.executemany(sql_insert_query, values)
            connection.commit()  # Фиксируем транзакцию
            print(f"{cursor.rowcount} строк(и) добавлено")
        else:
            print("Данные уже присутствуют в базе, добавление пропущено")
    
    except Error as e:
        connection.rollback()  # Откат в случае ошибки
        print(f"Ошибка при записи данных в MySQL: {e}")
        raise e  # Прерываем программу в случае ошибки

# Получение последней метки времени
def get_last_timestamp_from_db(connection, data_interval):
    cursor = connection.cursor()
    cursor.execute(f"SELECT MAX(open_time) FROM binance_klines WHERE data_interval = '{data_interval}'")
    result = cursor.fetchone()
    if result[0] is None:
        return None
    return int(result[0].timestamp() * 1000)  # Преобразуем объект datetime напрямую в timestamp

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
        '1M': None  # Для месячного интервала особая логика
    }

    previous_time = rows[0][0]
    for row in rows[1:]:
        current_time = row[0]
        if expected_diff[data_interval] is not None:
            if current_time - previous_time > expected_diff[data_interval]:
                print(f"Пропущенные данные между {previous_time} и {current_time} для интервала {data_interval}")
        previous_time = current_time

# Постоянная загрузка данных
async def continuous_data_fetch(symbol, intervals, year=2024):
    connection = connect_to_db()
    if connection is None:
        return

    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for data_interval in intervals:
                tasks.append(fetch_and_store_data(session, connection, symbol, data_interval, year))
            await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
    finally:
        if connection.is_connected():
            # Проверка данных после загрузки
            for data_interval in intervals:
                check_for_duplicates(connection)
                check_for_gaps(connection, data_interval)
            connection.close()
            print("Соединение с MySQL закрыто")

async def fetch_and_store_data(session, connection, symbol, data_interval, year):
    last_timestamp = get_last_timestamp_from_db(connection, data_interval)
    total_loaded = 0

    if last_timestamp is None:
        start_time = int(datetime(year, 1, 1, 0, 0).timestamp() * 1000)
    else:
        start_time = last_timestamp + 1

    with tqdm(total=1000, desc=f"Загрузка данных для {data_interval}", unit="строк") as pbar:
        while True:
            try:
                data = await get_binance_data(session, symbol, data_interval, start_time)
                if not data:
                    break

                save_to_mysql(connection, data, data_interval)
                total_loaded += len(data)
                last_timestamp = data[-1][0]
                start_time = last_timestamp + 1

                pbar.update(len(data))

                if len(data) < 1000:
                    break
            except Exception as e:
                print(f"Ошибка при загрузке данных: {e}")
                break

# Запуск программы
symbol = 'BTCUSDT'
intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
year = 2024

asyncio.run(continuous_data_fetch(symbol, intervals, year))
