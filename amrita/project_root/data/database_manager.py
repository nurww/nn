# database_manager.py

import pandas as pd
from sqlalchemy import create_engine
import os

# Настройки подключения к базе данных
DB_USER = "root"
DB_PASSWORD = "root"
DB_HOST = "localhost"
DB_NAME = "binance_data"

# Создаем подключение SQLAlchemy
def get_engine():
    """Создает и возвращает SQLAlchemy engine для подключения к базе данных."""
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    return create_engine(connection_string)

def execute_query(query: str) -> pd.DataFrame:
    """Выполняет SQL-запрос и возвращает результат."""
    engine = get_engine()
    try:
        result = pd.read_sql(query, engine)  # Теперь используем SQLAlchemy engine напрямую
        return result
    except Exception as e:
        print(f"Ошибка выполнения запроса: {e}")
        return pd.DataFrame()

def insert_data(table_name: str, data: pd.DataFrame) -> None:
    """Вставляет данные в указанную таблицу базы данных."""
    engine = get_engine()
    try:
        # Вставляем данные в таблицу
        data.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"Данные успешно вставлены в таблицу {table_name}")
    except Exception as e:
        print(f"Ошибка вставки данных: {e}")
