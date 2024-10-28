import talib as ta

# Индикаторы волатильности, например, Bollinger Bands
def add_volatility_indicators(df):
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = ta.BBANDS(df['close_price'], timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Убираем строки с NaN значениями
    df = df.dropna(subset=['Upper_BB', 'Middle_BB', 'Lower_BB'])
    
    return df
