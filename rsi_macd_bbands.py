# rsi_macd_bbands.py

import talib as ta
import pandas as pd

# Функция для расчета RSI, MACD, и Bollinger Bands (BB)
def calculate_rsi_macd_bb(df):
    df['RSI'] = ta.RSI(df['close_price'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close_price'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = ta.BBANDS(df['close_price'], timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Убираем строки с NaN значениями
    df = df.dropna(subset=['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Upper_BB', 'Middle_BB', 'Lower_BB'])
    
    return df
