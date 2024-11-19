import pandas as pd
from binance.client import Client
from amrita.project_root.data.database_manager import insert_data
import os

# Настройка клиента Binance API
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

def fetch_data_from_binance(interval: str, symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Получает данные по указанному интервалу с Binance API."""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 
        'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data = data.astype(float)
    return data

def save_data_to_db(data: pd.DataFrame, table_name: str) -> None:
    """Сохраняет данные в базу данных."""
    insert_data(table_name, data)
