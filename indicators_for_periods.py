# indicators_for_periods.py

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

# Функция для расчета Z-оценки
def z_score_normalization(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

# Функция для нормализации с учетом динамических минимумов и максимумов
def dynamic_min_max_normalization(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

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
        df = df.dropna(subset=['close_price', 'volume'])
        return df
    finally:
        connection.close()

async def fetch_data_from_mysql(interval):
    connection = await connect_to_db()
    try:
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            # Получаем последние 5000 строк для контекста расчета
            query_last_5000 = f"""
            SELECT * FROM binance_klines
            WHERE data_interval = '{interval}'
            ORDER BY open_time DESC
            LIMIT 5000
            """
            await cursor.execute(query_last_5000)
            result_last_5000 = await cursor.fetchall()

            # Преобразуем результаты в DataFrame
            df_last_5000 = pd.DataFrame(result_last_5000)

            # Получаем данные, у которых индикаторы еще не рассчитаны
            query_new_data = f"""
            SELECT * FROM binance_klines
            WHERE data_interval = '{interval}' AND rsi IS NULL
            ORDER BY open_time
            """
            await cursor.execute(query_new_data)
            result_new_data = await cursor.fetchall()

            df_new_data = pd.DataFrame(result_new_data)

        # Проверяем наличие данных
        if df_last_5000.empty and df_new_data.empty:
            print(f"Нет данных для обработки для интервала {interval}")
            return pd.DataFrame()

        # Объединяем последние 5000 строк с новыми данными
        df_combined = pd.concat([df_last_5000, df_new_data]).drop_duplicates(subset=['open_time'])

        # Преобразование значений в числовой формат
        df_combined['close_price'] = pd.to_numeric(df_combined['close_price'], errors='coerce')
        df_combined['open_price'] = pd.to_numeric(df_combined['open_price'], errors='coerce')
        df_combined['volume'] = pd.to_numeric(df_combined['volume'], errors='coerce')
        df_combined = df_combined.dropna(subset=['close_price', 'open_price', 'volume'])

        # Сортировка по времени
        df_combined = df_combined.sort_values(by='open_time').reset_index(drop=True)
        return df_combined
    finally:
        connection.close()

# Функция для добавления индикаторов на основе объединенных данных
def calculate_indicators_on_combined(df):
    # Сортировка данных по времени
    df = df.sort_values(by='open_time').reset_index(drop=True)
    
    # Рассчитываем индикаторы
    df['rsi'] = ta.RSI(df['close_price'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
        df['close_price'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['sma_20'] = ta.SMA(df['close_price'], timeperiod=20)
    df['ema_20'] = ta.EMA(df['close_price'], timeperiod=20)
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = ta.BBANDS(
        df['close_price'], timeperiod=20, nbdevup=2, nbdevdn=2)

    # Оставляем строки, где рассчитаны все индикаторы
    return df.dropna(subset=['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'ema_20', 'upper_bb', 'middle_bb', 'lower_bb'])

# Функция для добавления OBV на основе всех данных
def add_obv(df, full_data_df):
    full_data_df['OBV'] = ta.OBV(full_data_df['close_price'], full_data_df['volume'])
    merged_df = pd.merge(df, full_data_df[['open_time', 'OBV']], on='open_time', how='left')
    return merged_df


# Асинхронная функция для массового обновления данных в MySQL
async def bulk_update_to_mysql(df, interval):
    if df.empty:
        print(f"Нет данных для обновления для интервала {interval}")
        return

    df = df.replace({np.nan: None})

    connection = await connect_to_db()
    try:
        async with connection.cursor() as cursor:
            update_query = """
                UPDATE binance_klines
                SET rsi = %s, macd = %s, macd_signal = %s, macd_hist = %s, sma_20 = %s, ema_20 = %s, upper_bb = %s, middle_bb = %s, lower_bb = %s, OBV = %s
                WHERE open_time = %s AND data_interval = %s
            """
            update_values = list(
                zip(
                    df['rsi'], df['macd'], df['macd_signal'], df['macd_hist'],
                    df['sma_20'], df['ema_20'], df['upper_bb'], df['middle_bb'], df['lower_bb'], df['OBV'],
                    df['open_time'], [interval] * len(df)
                )
            )

            await cursor.executemany(update_query, update_values)
            await connection.commit()

            print(f"Успешно обновлено {cursor.rowcount} строк для интервала {interval}.")
    finally:
        connection.close()

# Объединяем все данные и рассчитываем индикаторы на них
async def process_interval_async(interval):
    start_time = time.time()

    # Загружаем новые данные и последние 5000 строк для контекста
    new_data_df = await fetch_data_from_mysql(interval)
    if new_data_df.empty:
        print(f"Нет данных для обработки для интервала {interval}")
        return

    # Рассчитываем индикаторы на всех новых данных
    df = calculate_indicators_on_combined(new_data_df)

    # Получаем все данные для расчета OBV, если это требуется
    full_data_df = await fetch_all_data_for_obv(interval)

    df = add_obv(df, full_data_df)

    # Обновляем данные в MySQL только для новых данных
    await bulk_update_to_mysql(df, interval)

    end_time = time.time()
    print(f"Время обработки интервала {interval}: {end_time - start_time} секунд")

# Асинхронная обработка всех интервалов
async def process_all_intervals_async():
    intervals_info = {'1m': 6, '5m': 4, '15m': 3, '1h': 2, '4h': 2, '1d': 1}
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
    print("\n")
    print("_____________# indicators_for_periods.py")
    asyncio.run(main())
    print("\n")
    print("__________________________||||||||")
