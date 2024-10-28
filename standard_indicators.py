import talib as ta

def add_standard_indicators(df):
    df['RSI'] = ta.RSI(df['close_price'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close_price'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['SMA_20'] = ta.SMA(df['close_price'], timeperiod=20)
    df['EMA_20'] = ta.EMA(df['close_price'], timeperiod=20)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = ta.BBANDS(df['close_price'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df = df.dropna(subset=['RSI', 'MACD', 'SMA_20', 'EMA_20', 'Upper_BB', 'Middle_BB', 'Lower_BB'])
    return df
