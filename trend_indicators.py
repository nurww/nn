import talib as ta

def add_trend_indicators(df):
    # Индикатор RSI
    df['RSI'] = ta.RSI(df['close_price'], timeperiod=14)
    
    # Индикаторы MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close_price'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Скользящие средние
    df['SMA_20'] = ta.SMA(df['close_price'], timeperiod=20)
    df['EMA_20'] = ta.EMA(df['close_price'], timeperiod=20)
    
    # Удаляем строки с NaN значениями после расчетов
    df = df.dropna(subset=['RSI', 'MACD', 'SMA_20', 'EMA_20'])
    
    return df
