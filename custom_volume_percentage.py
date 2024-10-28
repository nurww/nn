# custom_volume_percentage.py

def calculate_volume_percentage(df, period='1d'):
    # Рассчитываем средний объем за указанный период (например, день)
    df['avg_volume'] = df['volume'].rolling(window=period, min_periods=1).mean()

    # Рассчитываем процент текущего объема к среднему объему за указанный период
    df['volume_percentage'] = (df['volume'] / df['avg_volume']) * 100
    
    # Убираем строки с NaN значениями
    df = df.dropna(subset=['avg_volume', 'volume_percentage'])
    
    return df
