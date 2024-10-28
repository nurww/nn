import mysql.connector
import pandas as pd
import numpy as np
import argparse

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
            print("Соединение с базой данных установлено")
        return connection
    except mysql.connector.Error as e:
        print(f"Ошибка подключения к базе данных: {e}")
        return None

# Загрузка данных из базы
def load_data_from_db(connection, table_name, interval):
    query = f"""
        SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized,
               volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized,
               sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
        FROM {table_name}
        WHERE data_interval = '{interval}'
    """
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        df = pd.DataFrame(result)

        # Если нет данных, возвращаем пустые значения
        if df.empty:
            print(f"Данные по интервалу {interval} отсутствуют.")
            return None, None

        # Преобразуем данные в массивы NumPy
        X = df[['open_price_normalized', 'high_price_normalized', 'low_price_normalized', 'close_price_normalized',
                'volume_normalized', 'rsi_normalized', 'macd_normalized', 'macd_signal_normalized', 'macd_hist_normalized',
                'sma_20_normalized', 'ema_20_normalized', 'upper_bb_normalized', 'middle_bb_normalized', 'lower_bb_normalized', 'obv_normalized']].values
        y = df['close_price_normalized'].values
        return X, y
    
    except mysql.connector.Error as e:
        print(f"Ошибка выполнения запроса: {e}")
        return None, None
    finally:
        cursor.close()

# Основная функция
def main():
    # Используем argparse для ввода параметров через командную строку
    parser = argparse.ArgumentParser(description="Загрузка данных из базы для нейросети.")
    parser.add_argument('--table', type=str, default='binance_klines_normalized', help="Название таблицы для загрузки данных.")
    parser.add_argument('--interval', type=str, default='1m', help="Интервал данных для загрузки (например, '1m', '5m', '1d').")
    
    args = parser.parse_args()

    connection = connect_to_db()
    if connection is None:
        return  # Завершить, если нет подключения

    X, y = load_data_from_db(connection, args.table, args.interval)
    
    if X is not None and y is not None:
        print("Данные загружены успешно")
        print("Размеры данных:", X.shape, y.shape)
    else:
        print("Данные не были загружены.")
    
    connection.close()

if __name__ == '__main__':
    main()
