# candlestick_volume_plot.py

import pandas as pd
import mplfinance as mpf
import json
import asyncio
import redis.asyncio as aioredis
from datetime import datetime

# Настройка подключения к Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Функция для извлечения данных из Redis
async def fetch_and_aggregate_data_from_redis():
    redis_client = aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])
    data = await redis_client.lrange("normalized_order_book_stream", -1500, -1)
    await redis_client.close()

    # Преобразование данных из JSON в DataFrame
    records = [json.loads(record) for record in data]
    df = pd.DataFrame(records)

    # Преобразование timestamp в индекс и агрегирование данных по минутам
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Аггрегация по минутам для создания OHLCV данных
    ohlc = df['mid_price'].resample('1T').ohlc()
    volume = (df['sum_bid_volume'] + df['sum_ask_volume']).resample('1T').sum()

    # Объединение данных OHLC и объема в один DataFrame
    ohlc['volume'] = volume
    ohlc.dropna(inplace=True)  # Удаление строк с отсутствующими значениями

    return ohlc

# Функция для построения свечного графика с объемами
def plot_candlestick_with_volume(ohlc):
    # Настройка стиля
    mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', gridstyle='-', mavcolors=['#FFA07A', '#20B2AA'])
    # Построение графика с объемами
    mpf.plot(ohlc, type='candle', style=mpf_style, volume=True, show_nontrading=False, 
             title='Minute Candlestick Chart with Volume', ylabel='Price', ylabel_lower='Volume', 
             datetime_format='%H:%M', xrotation=20)

# Основная функция для вызова и построения графика
async def main():
    ohlc_data = await fetch_and_aggregate_data_from_redis()
    plot_candlestick_with_volume(ohlc_data)

# Запуск основного процесса
if __name__ == "__main__":
    asyncio.run(main())
