import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import time

# Подключение к базе данных
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

# Функция для получения всех данных для нормализации
def fetch_all_data_for_normalization(connection, interval):
    query = f"""
        SELECT open_price, high_price, low_price, close_price, volume, rsi, macd, macd_signal, macd_hist, sma_20, ema_20, upper_bb, middle_bb, lower_bb, obv
        FROM binance_klines
        WHERE data_interval = '{interval}'
        ORDER BY open_time;
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
    return pd.DataFrame(result)

# Функция для получения уже нормализованных данных
def fetch_existing_normalized_data(connection, interval):
    query = f"""
        SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized, volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized, sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
        FROM binance_klines_normalized
        WHERE data_interval = '{interval}'
        ORDER BY open_time;
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
    return pd.DataFrame(result)

# Функция для получения новых данных из оригинальной таблицы
def fetch_new_data(connection, interval):
    query = f"""
        SELECT open_time, open_price, high_price, low_price, close_price, volume, close_time, rsi, macd, macd_signal, macd_hist, sma_20, ema_20, upper_bb, middle_bb, lower_bb, obv
        FROM binance_klines
        WHERE data_interval = '{interval}'
        AND open_time NOT IN (SELECT open_time FROM binance_klines_normalized WHERE data_interval = '{interval}')
        ORDER BY open_time;
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
    df = pd.DataFrame(result)
    # Удаляем строки с отсутствующими данными в ключевых колонках
    df_cleaned = df.dropna(subset=['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'ema_20', 'upper_bb', 'middle_bb', 'lower_bb', 'obv'])  
    return df_cleaned

# Функция для нормализации данных с учетом полного диапазона
def normalize_data_with_full_range(df, full_data_df, columns):
    df_normalized = df.copy()
    for column in columns:
        min_value = full_data_df[column].min()
        max_value = full_data_df[column].max()
        print(f"Колонка {column}: min = {min_value}, max = {max_value}")  # Для отладки
        if max_value != min_value:  # Избегаем деления на ноль
            df_normalized[f'{column}_normalized'] = (df[column] - min_value) / (max_value - min_value)
        else:
            df_normalized[f'{column}_normalized'] = 0  # Или другое значение, если min == max
    return df_normalized

# Запись нормализованных данных в новую таблицу
def save_normalized_data_to_mysql(connection, normalized_df, interval):
    sql_insert_query = """
        INSERT INTO binance_klines_normalized (
            open_time, open_price_normalized, high_price_normalized, low_price_normalized,
            close_price_normalized, volume_normalized, close_time, rsi_normalized, macd_normalized,
            macd_signal_normalized, macd_hist_normalized, sma_20_normalized, ema_20_normalized,
            upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized, data_interval
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    cursor = connection.cursor()

    for _, row in normalized_df.iterrows():
        if row.isnull().any():
            continue  # Пропустить строки с отсутствующими данными
        values = (
            row['open_time'], row['open_price_normalized'], row['high_price_normalized'],
            row['low_price_normalized'], row['close_price_normalized'], row['volume_normalized'],
            row['close_time'], row['rsi_normalized'], row['macd_normalized'], row['macd_signal_normalized'],
            row['macd_hist_normalized'], row['sma_20_normalized'], row['ema_20_normalized'],
            row['upper_bb_normalized'], row['middle_bb_normalized'], row['lower_bb_normalized'],
            row['obv_normalized'], interval
        )

        cursor.execute(sql_insert_query, values)  # Вставка одной строки за раз
    
    connection.commit()
    cursor.close()

# Основной процесс нормализации
def process_normalization(interval):
    connection = connect_to_db()
    if connection is None:
        return

    try:
        start_time = time.time()

        # Получаем все данные для расчета минимальных и максимальных значений
        full_data = fetch_all_data_for_normalization(connection, interval)

        # Получаем существующие нормализованные данные
        existing_normalized_data = fetch_existing_normalized_data(connection, interval)

        # Добавляем нормализованные данные к исходным для расчета минимальных и максимальных значений
        full_data_with_existing = pd.concat([full_data, existing_normalized_data], axis=0)
        if full_data_with_existing.empty:
            print(f"Нет данных для интервала {interval}")
            return

        # Получаем новые данные для нормализации
        new_data = fetch_new_data(connection, interval)
        if new_data.empty:
            print(f"Нет новых данных для нормализации для интервала {interval}")
            return

        # Нормализуем новые данные с использованием минимальных и максимальных значений на основе полного набора данных
        normalized_data = normalize_data_with_full_range(new_data, full_data_with_existing, ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'ema_20', 'upper_bb', 'middle_bb', 'lower_bb', 'obv'])
        
        # Сохраняем нормализованные данные в базу данных
        save_normalized_data_to_mysql(connection, normalized_data, interval)

        end_time = time.time()
        print(f"Время обработки интервала {interval}: {end_time - start_time} секунд")
    
    finally:
        connection.close()

# Пример использования
if __name__ == '__main__':
    intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
    for interval in intervals:
        process_normalization(interval)
