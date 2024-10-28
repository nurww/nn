import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Функция для загрузки данных из базы данных
def load_order_book_data(connection, table_name, limit=1000):
    query = f"""
        SELECT open_time, price, amount_usdt, side
        FROM {table_name}
        ORDER BY open_time DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, connection)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values(by='open_time')  # Сортируем по времени для правильного формирования временных рядов
    return df

# Функция для формирования временных рядов
def create_time_series(df, window_size=10):
    """
    Создает временные ряды из данных ордербука.
    :param df: DataFrame с колонками 'price', 'amount_usdt' и 'side'.
    :param window_size: Количество временных шагов для создания временного ряда.
    :return: Наборы признаков X и целевых значений y.
    """
    X, y = [], []
    scaler = MinMaxScaler()

    # Нормализация данных
    df[['price', 'amount_usdt']] = scaler.fit_transform(df[['price', 'amount_usdt']])
    
    for i in range(len(df) - window_size):
        X.append(df[['price', 'amount_usdt']].iloc[i:i + window_size].values)
        # Целевое значение — средняя цена за следующий временной шаг
        y.append(df['price'].iloc[i + window_size])
    
    return np.array(X), np.array(y), scaler

# Функция для разделения данных на обучающую и тестовую выборки
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def main():
    # Подключение к базе данных
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='binance_data',
            user='root',
            password='root'
        )
        
        if connection.is_connected():
            print("Соединение с базой данных успешно.")
            
            # Загрузка данных
            df = load_order_book_data(connection, 'order_book_data', limit=10000)
            print(f"Загружено {len(df)} строк данных ордербука.")
            
            # Формирование временных рядов
            window_size = 10  # Определяем размер окна для временных рядов
            X, y, scaler = create_time_series(df, window_size)
            print(f"Сформировано {len(X)} временных рядов.")
            
            # Разделение данных
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
            print(f"Размер обучающей выборки: {len(X_train)}, тестовой выборки: {len(X_test)}.")
            
            # Сохранение подготовленных данных для дальнейшего использования
            np.save('data/X_train.npy', X_train)
            np.save('data/X_test.npy', X_test)
            np.save('data/y_train.npy', y_train)
            np.save('data/y_test.npy', y_test)
            print("Данные сохранены в папке 'data'.")

    except Error as e:
        print(f"Ошибка подключения к MySQL: {e}")

    finally:
        if connection.is_connected():
            connection.close()
            print("Соединение с MySQL закрыто.")

if __name__ == "__main__":
    main()
