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

# Функция для получения всех данных для нормализации (добавлена)
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

# Функция для получения минимальных и максимальных значений из таблицы stats
def fetch_min_max_from_stats(connection, interval):
    query = f"""
        SELECT * FROM binance_normalization_stats WHERE data_interval = '{interval}';
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
    return result

# Функция для обновления статистики минимумов и максимумов только на основе новых данных
def update_min_max_stats(connection, interval, new_data):
    if new_data.empty:
        print(f"Нет новых данных для обновления статистики для интервала {interval}")
        return

    # Получаем текущие минимальные и максимальные значения из таблицы stats
    current_min_max_values = fetch_min_max_from_stats(connection, interval)

    # Если текущие минимумы и максимумы не найдены, инициализируем их на основе новых данных
    if not current_min_max_values:
        min_max_values = {
            'min_open_price': new_data['open_price'].min(),
            'max_open_price': new_data['open_price'].max(),
            'min_high_price': new_data['high_price'].min(),
            'max_high_price': new_data['high_price'].max(),
            'min_low_price': new_data['low_price'].min(),
            'max_low_price': new_data['low_price'].max(),
            'min_close_price': new_data['close_price'].min(),
            'max_close_price': new_data['close_price'].max(),
            'min_volume': new_data['volume'].min(),
            'max_volume': new_data['volume'].max(),
            'min_rsi': new_data['rsi'].min(),
            'max_rsi': new_data['rsi'].max(),
            'min_macd': new_data['macd'].min(),
            'max_macd': new_data['macd'].max(),
            'min_macd_signal': new_data['macd_signal'].min(),
            'max_macd_signal': new_data['macd_signal'].max(),
            'min_macd_hist': new_data['macd_hist'].min(),
            'max_macd_hist': new_data['macd_hist'].max(),
            'min_sma_20': new_data['sma_20'].min(),
            'max_sma_20': new_data['sma_20'].max(),
            'min_ema_20': new_data['ema_20'].min(),
            'max_ema_20': new_data['ema_20'].max(),
            'min_upper_bb': new_data['upper_bb'].min(),
            'max_upper_bb': new_data['upper_bb'].max(),
            'min_middle_bb': new_data['middle_bb'].min(),
            'max_middle_bb': new_data['middle_bb'].max(),
            'min_lower_bb': new_data['lower_bb'].min(),
            'max_lower_bb': new_data['lower_bb'].max(),
            'min_obv': new_data['obv'].min(),
            'max_obv': new_data['obv'].max(),
        }
    else:
        # Сравниваем новые данные с текущими минимумами и максимумами
        min_max_values = {
            'min_open_price': min(new_data['open_price'].min(), current_min_max_values['min_open_price']),
            'max_open_price': max(new_data['open_price'].max(), current_min_max_values['max_open_price']),
            'min_high_price': min(new_data['high_price'].min(), current_min_max_values['min_high_price']),
            'max_high_price': max(new_data['high_price'].max(), current_min_max_values['max_high_price']),
            'min_low_price': min(new_data['low_price'].min(), current_min_max_values['min_low_price']),
            'max_low_price': max(new_data['low_price'].max(), current_min_max_values['max_low_price']),
            'min_close_price': min(new_data['close_price'].min(), current_min_max_values['min_close_price']),
            'max_close_price': max(new_data['close_price'].max(), current_min_max_values['max_close_price']),
            'min_volume': min(new_data['volume'].min(), current_min_max_values['min_volume']),
            'max_volume': max(new_data['volume'].max(), current_min_max_values['max_volume']),
            'min_rsi': min(new_data['rsi'].min(), current_min_max_values['min_rsi']),
            'max_rsi': max(new_data['rsi'].max(), current_min_max_values['max_rsi']),
            'min_macd': min(new_data['macd'].min(), current_min_max_values['min_macd']),
            'max_macd': max(new_data['macd'].max(), current_min_max_values['max_macd']),
            'min_macd_signal': min(new_data['macd_signal'].min(), current_min_max_values['min_macd_signal']),
            'max_macd_signal': max(new_data['macd_signal'].max(), current_min_max_values['max_macd_signal']),
            'min_macd_hist': min(new_data['macd_hist'].min(), current_min_max_values['min_macd_hist']),
            'max_macd_hist': max(new_data['macd_hist'].max(), current_min_max_values['max_macd_hist']),
            'min_sma_20': min(new_data['sma_20'].min(), current_min_max_values['min_sma_20']),
            'max_sma_20': max(new_data['sma_20'].max(), current_min_max_values['max_sma_20']),
            'min_ema_20': min(new_data['ema_20'].min(), current_min_max_values['min_ema_20']),
            'max_ema_20': max(new_data['ema_20'].max(), current_min_max_values['max_ema_20']),
            'min_upper_bb': min(new_data['upper_bb'].min(), current_min_max_values['min_upper_bb']),
            'max_upper_bb': max(new_data['upper_bb'].max(), current_min_max_values['max_upper_bb']),
            'min_middle_bb': min(new_data['middle_bb'].min(), current_min_max_values['min_middle_bb']),
            'max_middle_bb': max(new_data['middle_bb'].max(), current_min_max_values['max_middle_bb']),
            'min_lower_bb': min(new_data['lower_bb'].min(), current_min_max_values['min_lower_bb']),
            'max_lower_bb': max(new_data['lower_bb'].max(), current_min_max_values['max_lower_bb']),
            'min_obv': min(new_data['obv'].min(), current_min_max_values['min_obv']),
            'max_obv': max(new_data['obv'].max(), current_min_max_values['max_obv']),
        }

    # Обновляем статистику
    sql_update_query = """
        INSERT INTO binance_normalization_stats (
            data_interval, min_open_price, max_open_price, min_high_price, max_high_price,
            min_low_price, max_low_price, min_close_price, max_close_price, min_volume, max_volume,
            min_rsi, max_rsi, min_macd, max_macd, min_macd_signal, max_macd_signal, min_macd_hist,
            max_macd_hist, min_sma_20, max_sma_20, min_ema_20, max_ema_20, min_upper_bb, max_upper_bb,
            min_middle_bb, max_middle_bb, min_lower_bb, max_lower_bb, min_obv, max_obv
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            min_open_price = VALUES(min_open_price), max_open_price = VALUES(max_open_price),
            min_high_price = VALUES(min_high_price), max_high_price = VALUES(max_high_price),
            min_low_price = VALUES(min_low_price), max_low_price = VALUES(max_low_price),
            min_close_price = VALUES(min_close_price), max_close_price = VALUES(max_close_price),
            min_volume = VALUES(min_volume), max_volume = VALUES(max_volume),
            min_rsi = VALUES(min_rsi), max_rsi = VALUES(max_rsi), min_macd = VALUES(min_macd),
            max_macd = VALUES(max_macd), min_macd_signal = VALUES(min_macd_signal),
            max_macd_signal = VALUES(max_macd_signal), min_macd_hist = VALUES(min_macd_hist),
            max_macd_hist = VALUES(max_macd_hist), min_sma_20 = VALUES(min_sma_20),
            max_sma_20 = VALUES(max_sma_20), min_ema_20 = VALUES(min_ema_20), max_ema_20 = VALUES(max_ema_20),
            min_upper_bb = VALUES(min_upper_bb), max_upper_bb = VALUES(max_upper_bb),
            min_middle_bb = VALUES(min_middle_bb), max_middle_bb = VALUES(max_middle_bb),
            min_lower_bb = VALUES(min_lower_bb), max_lower_bb = VALUES(max_lower_bb),
            min_obv = VALUES(min_obv), max_obv = VALUES(max_obv)
    """
    
    cursor = connection.cursor()
    cursor.execute(sql_update_query, (interval, *min_max_values.values()))
    connection.commit()
    cursor.close()

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
    df_cleaned = df.dropna(subset=['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'ema_20', 'upper_bb', 'middle_bb', 'lower_bb', 'obv'])  
    return df_cleaned

# Функция для нормализации данных с использованием кэша минимумов и максимумов
def normalize_data_with_full_range(df, min_max_values, columns):
    df_normalized = df.copy()
    for column in columns:
        min_value = min_max_values[f'min_{column}']
        max_value = min_max_values[f'max_{column}']
        if max_value != min_value:  # Избегаем деления на ноль
            df_normalized[f'{column}_normalized'] = (df[column] - min_value) / (max_value - min_value)
        else:
            df_normalized[f'{column}_normalized'] = 0
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

# Основной процесс нормализации с использованием кэша минимумов и максимумов
def process_normalization_with_stats_cache(interval):
    connection = connect_to_db()
    if connection is None:
        return

    try:
        start_time = time.time()

        # Получаем новые данные для нормализации
        new_data = fetch_new_data(connection, interval)
        if new_data.empty:
            print(f"Нет новых данных для нормализации для интервала {interval}")
            return

        # Обновляем статистику минимумов и максимумов
        update_min_max_stats(connection, interval, new_data)

        # Получаем минимальные и максимальные значения из таблицы stats
        min_max_values = fetch_min_max_from_stats(connection, interval)
        if not min_max_values:
            print(f"Нет минимальных и максимальных значений для интервала {interval}, обновите статистику.")
            return

        # Нормализуем данные с использованием минимальных и максимальных значений
        normalized_data = normalize_data_with_full_range(new_data, min_max_values, ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'ema_20', 'upper_bb', 'middle_bb', 'lower_bb', 'obv'])
        
        # Сохраняем нормализованные данные в базу данных
        save_normalized_data_to_mysql(connection, normalized_data, interval)

        end_time = time.time()
        print(f"Время обработки интервала {interval}: {end_time - start_time} секунд")
    
    finally:
        connection.close()

# Функция для индексации таблицы
def optimize_table(connection):
    query = "OPTIMIZE TABLE binance_klines_normalized;"
    cursor = connection.cursor()
    cursor.execute(query)
    print("Оптимизация таблицы завершена")
    cursor.close()

# Пример использования
if __name__ == '__main__':
    intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
    # connection = connect_to_db()
    
    # # Оптимизация таблицы раз в день
    # optimize_table(connection)

    for interval in intervals:
        process_normalization_with_stats_cache(interval)
