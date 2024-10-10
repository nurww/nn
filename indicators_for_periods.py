import aiomysql
import talib as ta
import time
import asyncio
import pandas as pd
import numpy as np

# Асинхронная функция для подключения к MySQL
async def connect_to_db():
    return await aiomysql.connect(
        host='localhost',
        db='binance_data',
        user='root',
        password='root'
    )

# Асинхронная функция для получения всех данных для расчета OBV
async def fetch_all_data_for_obv(interval):
    connection = await connect_to_db()
    try:
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            query = f"""
            SELECT open_time, close_price, volume FROM binance_klines
            WHERE data_interval = '{interval}'
            ORDER BY open_time
            """
            await cursor.execute(query)
            result = await cursor.fetchall()
        
        if not result:
            print(f"Нет данных для расчета OBV для интервала {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(result)
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.dropna(subset=['close_price', 'volume'])  # Удаление некорректных данных
        return df
    finally:
        connection.close()

# Асинхронная функция для получения данных из MySQL (без OBV)
async def fetch_data_from_mysql(interval):
    connection = await connect_to_db()
    try:
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            query = f"""
            SELECT * FROM binance_klines
            WHERE data_interval = '{interval}' AND (rsi IS NULL OR macd IS NULL OR sma_20 IS NULL OR ema_20 IS NULL)
            ORDER BY open_time
            """
            await cursor.execute(query)
            result = await cursor.fetchall()
        
        if not result:
            print(f"Нет новых данных для интервала {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(result)
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        df = df.dropna(subset=['close_price'])  # Удаление некорректных данных
        return df
    finally:
        connection.close()

# Функция для добавления индикаторов (кроме OBV)
def add_indicators_except_obv(df):
    df['RSI'] = ta.RSI(df['close_price'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close_price'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['SMA_20'] = ta.SMA(df['close_price'], timeperiod=20)
    df['EMA_20'] = ta.EMA(df['close_price'], timeperiod=20)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = ta.BBANDS(df['close_price'], timeperiod=20, nbdevup=2, nbdevdn=2)

    df = df.dropna(subset=['RSI', 'MACD', 'SMA_20', 'EMA_20', 'Upper_BB', 'Middle_BB', 'Lower_BB'])
    return df

# Функция для добавления OBV на основе всех данных
def add_obv(df, full_data_df):
    full_data_df['OBV'] = ta.OBV(full_data_df['close_price'], full_data_df['volume'])
    
    # Найдем пересечение по времени, чтобы правильно добавить OBV для текущих данных
    merged_df = pd.merge(df, full_data_df[['open_time', 'OBV']], on='open_time', how='left')
    return merged_df

# Асинхронная функция для массового обновления данных в MySQL
async def bulk_update_to_mysql(df, interval):
    if df.empty:
        print(f"Нет данных для обновления для интервала {interval}")
        return

    df = df.replace({np.nan: None})  # Замена NaN на None для MySQL
    connection = await connect_to_db()

    try:
        async with connection.cursor() as cursor:
            update_query = """
                UPDATE binance_klines
                SET rsi = %s, macd = %s, macd_signal = %s, macd_hist = %s, sma_20 = %s, ema_20 = %s, upper_bb = %s, middle_bb = %s, lower_bb = %s, obv = %s
                WHERE open_time = %s AND data_interval = %s
            """
            update_values = list(
                zip(
                    df['RSI'], df['MACD'], df['MACD_signal'], df['MACD_hist'],
                    df['SMA_20'], df['EMA_20'], df['Upper_BB'], df['Middle_BB'], df['Lower_BB'], df['OBV'],
                    df['open_time'], [interval] * len(df)
                )
            )

            await cursor.executemany(update_query, update_values)
            await connection.commit()

            print(f"Успешно обновлено {cursor.rowcount} строк для интервала {interval}.")
    finally:
        connection.close()

# Асинхронная обработка данных для каждого интервала
async def process_interval_async(interval):
    start_time = time.time()  # Логирование начала времени

    df = await fetch_data_from_mysql(interval)
    if df.empty:
        return

    # Получаем все данные для расчета OBV
    full_data_df = await fetch_all_data_for_obv(interval)

    # Добавляем индикаторы (кроме OBV)
    df = add_indicators_except_obv(df)

    # Добавляем OBV на основе всех данных
    df = add_obv(df, full_data_df)

    # Обновляем данные в MySQL
    await bulk_update_to_mysql(df, interval)

    end_time = time.time()  # Логирование окончания времени
    print(f"Время обработки интервала {interval}: {end_time - start_time} секунд")

# Асинхронная обработка всех интервалов
async def process_all_intervals_async():
    intervals_info = {
        '1m': 6,  
        '5m': 4,  
        '15m': 3,
        '1h': 2,
        '4h': 2,
        '1d': 1  
    }

    tasks = [process_interval_async(interval) for interval in intervals_info.keys()]
    await asyncio.gather(*tasks)

# Основная функция
async def main():
    start_time = time.time()
    await process_all_intervals_async()
    end_time = time.time()
    print(f"Общее время выполнения: {end_time - start_time} секунд")

# Запуск программы
if __name__ == '__main__':
    asyncio.run(main())
