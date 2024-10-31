# data_processor.py

import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import timedelta
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

# Функция для создания нового окна и деактивации старых
def create_new_window(connection, interval, min_max_values, start_time, end_time):
    deactivate_windows(connection, interval)
    
    # Задаем min_volume равным 0, чтобы исключить ошибочные создания окон
    min_max_values['min_volume'] = 0
    min_max_values['min_open_price'] = 0
    min_max_values['min_high_price'] = 0
    min_max_values['min_low_price'] = 0
    min_max_values['min_close_price'] = 0

    last_window = get_last_window(connection, interval)
    if last_window and start_time <= last_window['end_time']:
        start_time = last_window['end_time'] + timedelta(seconds=1)

    query = """
        INSERT INTO binance_normalization_windows (
            data_interval, start_time, end_time,
            min_open_price, max_open_price, min_high_price, max_high_price,
            min_low_price, max_low_price, min_close_price, max_close_price,
            min_volume, max_volume, min_rsi, max_rsi, min_macd, max_macd,
            min_macd_signal, max_macd_signal, min_macd_hist, max_macd_hist,
            min_sma_20, max_sma_20, min_ema_20, max_ema_20, min_upper_bb, max_upper_bb,
            min_middle_bb, max_middle_bb, min_lower_bb, max_lower_bb, min_obv, max_obv, is_active
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor = connection.cursor()
    cursor.execute(query, (
        interval, start_time, end_time,
        min_max_values['min_open_price'], min_max_values['max_open_price'],
        min_max_values['min_high_price'], min_max_values['max_high_price'],
        min_max_values['min_low_price'], min_max_values['max_low_price'],
        min_max_values['min_close_price'], min_max_values['max_close_price'],
        min_max_values['min_volume'], min_max_values['max_volume'],
        min_max_values['min_rsi'], min_max_values['max_rsi'],
        min_max_values['min_macd'], min_max_values['max_macd'],
        min_max_values['min_macd_signal'], min_max_values['max_macd_signal'],
        min_max_values['min_macd_hist'], min_max_values['max_macd_hist'],
        min_max_values['min_sma_20'], min_max_values['max_sma_20'],
        min_max_values['min_ema_20'], min_max_values['max_ema_20'],
        min_max_values['min_upper_bb'], min_max_values['max_upper_bb'],
        min_max_values['min_middle_bb'], min_max_values['max_middle_bb'],
        min_max_values['min_lower_bb'], min_max_values['max_lower_bb'],
        min_max_values['min_obv'], min_max_values['max_obv'],
        1
    ))
    connection.commit()
    cursor.execute("SELECT LAST_INSERT_ID();")
    window_id = cursor.fetchone()[0]
    cursor.close()
    print(f"Создано новое активное окно нормализации с ID {window_id} для интервала {interval}")
    return window_id

# Функция для деактивации окон
def deactivate_windows(connection, interval):
    query = """
        UPDATE binance_normalization_windows
        SET is_active = 0
        WHERE data_interval = %s AND is_active = 1
    """
    cursor = connection.cursor()
    cursor.execute(query, (interval,))
    connection.commit()
    cursor.close()

# Функция для получения последнего окна
def get_last_window(connection, interval):
    query = """
        SELECT * FROM binance_normalization_windows 
        WHERE data_interval = %s and is_active = 1
        ORDER BY end_time DESC LIMIT 1
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query, (interval,))
        result = cursor.fetchone()
    return result

# Функция для обновления временных границ текущего окна
def update_window_dates(connection, window_id, start_time, end_time):
    query = """
        UPDATE binance_normalization_windows
        SET start_time = LEAST(start_time, %s),
            end_time = GREATEST(end_time, %s)
        WHERE window_id = %s
    """
    cursor = connection.cursor()
    cursor.execute(query, (start_time, end_time, window_id))
    connection.commit()
    cursor.close()
    print(f"Обновлены временные границы окна с ID {window_id}.")

def update_min_max_stats_for_window(df):
    min_max_values = {
        'min_open_price': df['open_price'].min(),
        'max_open_price': df['open_price'].max(),
        'min_high_price': df['high_price'].min(),
        'max_high_price': df['high_price'].max(),
        'min_low_price': df['low_price'].min(),
        'max_low_price': df['low_price'].max(),
        'min_close_price': df['close_price'].min(),
        'max_close_price': df['close_price'].max(),
        'min_volume': df['volume'].min(),
        'max_volume': df['volume'].max(),
        'min_rsi': df['rsi'].min(),
        'max_rsi': df['rsi'].max(),
        'min_macd': df['macd'].min(),
        'max_macd': df['macd'].max(),
        'min_macd_signal': df['macd_signal'].min(),
        'max_macd_signal': df['macd_signal'].max(),
        'min_macd_hist': df['macd_hist'].min(),
        'max_macd_hist': df['macd_hist'].max(),
        'min_sma_20': df['sma_20'].min(),
        'max_sma_20': df['sma_20'].max(),
        'min_ema_20': df['ema_20'].min(),
        'max_ema_20': df['ema_20'].max(),
        'min_upper_bb': df['upper_bb'].min(),
        'max_upper_bb': df['upper_bb'].max(),
        'min_middle_bb': df['middle_bb'].min(),
        'max_middle_bb': df['middle_bb'].max(),
        'min_lower_bb': df['lower_bb'].min(),
        'max_lower_bb': df['lower_bb'].max(),
        'min_obv': df['obv'].min(),
        'max_obv': df['obv'].max(),
    }
    return min_max_values

# Функция для обновления min_max значений, обновляя только отличающиеся
def update_min_max_values(min_max_values, current_window):
    updated_min_max_values = {}

    for column in ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'rsi', 'macd', 
                   'macd_signal', 'macd_hist', 'sma_20', 'ema_20', 'upper_bb', 'middle_bb', 
                   'lower_bb', 'obv']:
        
        # Проверка для минимального значения
        min_key = f'min_{column}'
        if min_max_values[min_key] < current_window.get(min_key, 0):
            updated_min_max_values[min_key] = min_max_values[min_key]
        
        # Проверка для максимального значения
        max_key = f'max_{column}'
        if min_max_values[max_key] > current_window.get(max_key, float('-inf')):
            updated_min_max_values[max_key] = min_max_values[max_key]

    return updated_min_max_values

# Функция для проверки, нужно ли создать новое окно или обновить текущее
def check_and_update_window(connection, interval, df, current_window):
    # Обновляем статистику min/max по данным из df
    min_max_values = update_min_max_stats_for_window(df)
    
    # Если текущего окна нет, создаем новое
    if not current_window:
        print("Активного окна не найдено. Создание нового окна.")
        window_id = create_new_window(
            connection, interval, min_max_values,
            start_time=df['open_time'].min(),
            end_time=df['open_time'].max()
        )
        # Обновляем current_window после создания нового окна
        current_window = get_window_by_id(connection, window_id)

    # Проверка значений, выходящих за пределы текущих min/max, с учетом равенства
    new_extremes_found = any(
        (
            (min_max_values[f'min_{column}'] < current_window.get(f'min_{column}', 0) if column != 'volume' else False) or
            (min_max_values[f'max_{column}'] > current_window.get(f'max_{column}', float('-inf')))
        ) and not (
            min_max_values[f'min_{column}'] == current_window.get(f'min_{column}', 0) and
            min_max_values[f'max_{column}'] == current_window.get(f'max_{column}', float('-inf'))
        )
        for column in ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'rsi', 'macd', 
                    'macd_signal', 'macd_hist', 'sma_20', 'ema_20', 'upper_bb', 'middle_bb', 
                    'lower_bb', 'obv']
    )

    # Если найдены новые экстремумы, создаем новое окно
    if new_extremes_found:
        print("Найдено значение за пределами текущего окна. Создание нового окна.")

        # Обновляем только измененные значения
        updated_values = update_min_max_values(min_max_values, current_window)
                                           
        window_id = create_new_window(
            connection, interval, updated_values,
            start_time=df['open_time'].min(),
            end_time=df['open_time'].max()
        )
        current_window = get_window_by_id(connection, window_id)
    else:
        # Если значения в пределах текущих, обновляем даты границ окна
        print("Обновление временных границ текущего окна.")
        window_id = current_window['window_id']
        update_window_dates(connection, window_id, start_time=df['open_time'].min(), end_time=df['open_time'].max())
    
    return window_id

# Функция для получения окна по window_id
def get_window_by_id(connection, window_id):
    query = """
        SELECT * FROM binance_normalization_windows 
        WHERE window_id = %s 
        ORDER BY end_time DESC LIMIT 1
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query, (window_id,))
        result = cursor.fetchone()
    return result

# Функция для нормализации данных с использованием текущего окна
def normalize_data_with_window(df, min_max_values):
    df_normalized = df.copy()
    for column, (min_value, max_value) in min_max_values.items():
        if max_value != min_value:
            df_normalized[f'{column}_normalized'] = (df[column] - min_value) / (max_value - min_value)
        else:
            df_normalized[f'{column}_normalized'] = 0  # Избегаем деления на ноль
    return df_normalized

# Функция для нормализации данных с использованием информации об окнах
def normalize_data_with_windows(df, windows):
    df_normalized = pd.DataFrame()

    for window in windows:
        mask = (df['open_time'] >= window['start_time']) & (df['open_time'] <= window['end_time'])
        window_df = df[mask]
        if window_df.empty:
            continue

        min_max_values = {
            column: (window[f'min_{column}'], window[f'max_{column}']) for column in [
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'ema_20',
                'upper_bb', 'middle_bb', 'lower_bb', 'obv'
            ]
        }
        window_normalized_df = normalize_data_with_window(window_df, min_max_values)
        window_normalized_df['window_id'] = window['window_id']
        df_normalized = pd.concat([df_normalized, window_normalized_df])

    return df_normalized.reset_index(drop=True)

# Функция для получения времени последней нормализованной записи
def get_last_normalized_time(connection, interval):
    query = """
        SELECT MAX(open_time) AS last_time
        FROM binance_klines_normalized
        WHERE data_interval = %s
    """
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query, (interval,))
    result = cursor.fetchone()
    cursor.close()
    return result['last_time'] if result and result['last_time'] else None

# Функция для выборки только новых данных для нормализации
def fetch_new_data(connection, interval, last_normalized_time):
    query = f"""
        SELECT open_time, open_price, high_price, low_price, close_price, volume, close_time, rsi, macd, macd_signal, macd_hist, sma_20, ema_20, upper_bb, middle_bb, lower_bb, obv
        FROM binance_klines
        WHERE data_interval = %s AND rsi IS NOT NULL AND open_time > %s
        ORDER BY open_time;
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query, (interval, last_normalized_time))
        result = cursor.fetchall()
    df = pd.DataFrame(result)

    if df.empty:
        print("Нет новых данных для нормализации.")
        return df  # Возвращаем пустой DataFrame, чтобы избежать дальнейших ошибок

    df = df.sort_values(by='open_time').reset_index(drop=True)
    return df

# Функция для получения активного окна
def get_active_window(connection, interval):
    query = f"""
        SELECT * FROM binance_normalization_windows 
        WHERE data_interval = %s AND is_active = 1
        ORDER BY end_time DESC LIMIT 1
    """
    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query, (interval,))
        result = cursor.fetchone()
    return result

# Функция для получения информации об окнах нормализации
def fetch_window_info(connection, interval, window_id=None):
    """
    Получает информацию об окнах нормализации для заданного интервала.
    Если window_id передан, возвращает информацию только для конкретного окна.
    """
    if window_id:
        query = f"""
            SELECT * FROM binance_normalization_windows
            WHERE data_interval = '{interval}' AND window_id = {window_id}
            ORDER BY window_id DESC;
        """
    else:
        query = f"""
            SELECT * FROM binance_normalization_windows
            WHERE data_interval = '{interval}'
            ORDER BY window_id DESC;
        """

    with connection.cursor(dictionary=True) as cursor:
        cursor.execute(query)
        result = cursor.fetchall()

    return result

def save_normalized_data_to_mysql(connection, normalized_df, interval):
    sql_insert_query = """
        INSERT INTO binance_klines_normalized (
            open_time, open_price_normalized, high_price_normalized, low_price_normalized,
            close_price_normalized, volume_normalized, close_time, rsi_normalized, macd_normalized,
            macd_signal_normalized, macd_hist_normalized, sma_20_normalized, ema_20_normalized,
            upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized, data_interval, window_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            row['obv_normalized'], interval, row['window_id']
        )
        cursor.execute(sql_insert_query, values)
    
    connection.commit()
    cursor.close()

# Основная функция для нормализации данных с учётом окон
def process_normalization_with_windows(interval):
    connection = connect_to_db()
    if connection is None:
        return

    try:
        start_time = time.time()
        
        last_normalized_time = get_last_normalized_time(connection, interval) or '1970-01-01 00:00:00'
        df = fetch_new_data(connection, interval, last_normalized_time)
        if df.empty:
            print(f"Нет новых данных для нормализации для интервала {interval}")
            return

        current_window = get_active_window(connection, interval)
        window_id = check_and_update_window(connection, interval, df, current_window)

        windows = fetch_window_info(connection, interval, window_id)
        normalized_data = normalize_data_with_windows(df, windows)
        save_normalized_data_to_mysql(connection, normalized_data, interval)
        print(f"Время обработки интервала {interval}: {time.time() - start_time} секунд")
    finally:
        connection.close()

# Запуск основной функции для каждого интервала
def main():
    print("\n")
    print("_____________# data_processor.py")
    intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
    for interval in intervals:
        process_normalization_with_windows(interval)
    print("\n")
    print("__________________________||||||||")

if __name__ == '__main__':
    main()
