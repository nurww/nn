# custom_columns.py

import mysql.connector
import pandas as pd
import numpy as np

def connect_to_db():
    connection = mysql.connector.connect(
        host='localhost',
        database='binance_data',
        user='root',
        password='root'
    )
    return connection

# Функция для расчета процента объема за период
def calculate_volume_percentage(df, total_volume, period_days=365):
    # Рассчитываем средний объем за выбранный период
    average_volume = total_volume / period_days
    
    # Рассчитываем процент текущего объема к среднему
    df['volume_percentage'] = (df['volume'] / average_volume) * 100
    return df

# Пример расчета ожидаемого процента объема на основе меньшего интервала (например, 4 часа)
def calculate_intraday_volume_percentage(df, current_interval_volume, passed_intervals, total_intervals):
    # Рассчитываем процент объема для текущего времени в рамках дня
    df['intraday_volume_percentage'] = (current_interval_volume / df['volume']) * (passed_intervals / total_intervals) * 100
    return df

def calculate_custom_volume_percentage(connection, table_name, interval, days=365):
    # Запрос для получения данных по объему за предыдущие дни
    query = f"""
        SELECT open_time, volume
        FROM {table_name}
        WHERE data_interval = '{interval}'
        AND open_time >= NOW() - INTERVAL {days} DAY
    """
    df = pd.read_sql(query, connection)

    # Рассчитываем средний объем за последние {days} дней
    avg_volume = df['volume'].mean()

    # Получаем текущие данные
    query_current = f"""
        SELECT open_time, volume
        FROM {table_name}
        WHERE data_interval = '{interval}'
        ORDER BY open_time DESC
        LIMIT 1
    """
    current_data = pd.read_sql(query_current, connection)
    current_volume = current_data['volume'].values[0]

    # Рассчитываем процент текущего объема к среднему за период
    volume_percentage = (current_volume / avg_volume) * 100 if avg_volume != 0 else 0

    return volume_percentage

def add_custom_column(connection, table_name, interval):
    volume_percentage = calculate_custom_volume_percentage(connection, table_name, interval)

    # Добавляем кастомную колонку в таблицу
    cursor = connection.cursor()
    query = f"""
        ALTER TABLE {table_name}
        ADD COLUMN IF NOT EXISTS volume_percentage FLOAT
    """
    cursor.execute(query)
    
    # Обновляем данные с рассчитанным процентом объема
    query_update = f"""
        UPDATE {table_name}
        SET volume_percentage = {volume_percentage}
        WHERE open_time = (SELECT MAX(open_time) FROM {table_name} WHERE data_interval = '{interval}')
    """
    cursor.execute(query_update)
    connection.commit()
    cursor.close()

def main():
    connection = connect_to_db()
    if connection is None:
        return

    table_name = 'binance_klines_normalized'
    interval = '1d'  # Пример интервала, можно заменить на другой

    add_custom_column(connection, table_name, interval)
    print("Кастомная колонка добавлена и обновлена.")

    connection.close()

if __name__ == '__main__':
    main()
