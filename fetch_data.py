import aiomysql
import pandas as pd

# Асинхронная функция для подключения к базе данных
async def connect_to_db():
    connection = await aiomysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='binance_data',
        cursorclass=aiomysql.DictCursor
    )
    return connection

# Асинхронная функция для получения данных из MySQL
async def fetch_data_from_mysql(interval):
    connection = await connect_to_db()
    try:
        async with connection.cursor() as cursor:
            query = f"""
                SELECT * FROM binance_klines
                WHERE data_interval = '{interval}'
                ORDER BY open_time;
            """
            await cursor.execute(query)
            result = await cursor.fetchall()
        
        if not result:
            print(f"Нет новых данных для интервала {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(result)
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        df = df.dropna(subset=['close_price'])
        return df
    finally:
        connection.close()
