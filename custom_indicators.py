# Кастомные индикаторы
def calculate_custom_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close_price'][i] > df['close_price'][i-1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close_price'][i] < df['close_price'][i-1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    df['custom_OBV'] = obv
    return df

# Индикатор процента объема
def calculate_volume_percentage(df, period='1d'):
    df['avg_volume'] = df['volume'].rolling(window=period, min_periods=1).mean()
    df['volume_percentage'] = (df['volume'] / df['avg_volume']) * 100
    df = df.dropna(subset=['volume_percentage'])
    return df
