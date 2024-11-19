import pandas as pd
from binance.client import Client
from amrita.project_root.data.database_manager import insert_data
import os

# Настройка клиента Binance API
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

def fetch_orderbook_data(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Получает данные стакана с Binance API."""
    orderbook = client.get_order_book(symbol=symbol, limit=limit)
    bids = pd.DataFrame(orderbook['bids'], columns=['price', 'quantity']).astype(float)
    asks = pd.DataFrame(orderbook['asks'], columns=['price', 'quantity']).astype(float)
    bids['side'] = 'bid'
    asks['side'] = 'ask'
    data = pd.concat([bids, asks])
    data['timestamp'] = pd.Timestamp.utcnow()
    return data

def save_orderbook_to_db(data: pd.DataFrame, table_name: str) -> None:
    """Сохраняет данные стакана в базу данных."""
    insert_data(table_name, data)
