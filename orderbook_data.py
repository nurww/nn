# orderbook_data.py — Загрузка данных стакана

import pandas as pd

def add_orderbook_data(df, orderbook_df):
    # Добавляем лучшие бид/аск цены и объемы в датафрейм с торговыми данными
    df['best_bid'] = orderbook_df['best_bid']
    df['best_ask'] = orderbook_df['best_ask']
    df['bid_volume'] = orderbook_df['bid_volume']
    df['ask_volume'] = orderbook_df['ask_volume']
    
    return df

def load_orderbook_data(connection, interval):
    query = f"""
        SELECT open_time, best_bid, best_ask, bid_volume, ask_volume
        FROM binance_orderbook
        WHERE data_interval = '{interval}'
    """
    orderbook_df = pd.read_sql(query, connection)
    return orderbook_df
