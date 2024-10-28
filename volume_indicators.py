import talib as ta

# Рассчитываем стандартный OBV
def add_obv(df):
    df['OBV'] = ta.OBV(df['close_price'], df['volume'])
    return df

# Кастомный индикатор объема (процент объема)
def calculate_volume_percentage(df, period='1d'):
    df['avg_volume'] = df['volume'].rolling(window=period, min_periods=1).mean()
    df['volume_percentage'] = (df['volume'] / df['avg_volume']) * 100
    df = df.dropna(subset=['volume_percentage'])
    return df
