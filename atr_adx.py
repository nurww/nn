# atr_adx.py

import talib as ta
import pandas as pd

# Функция для расчета ATR и ADX
def calculate_atr_adx(df):
    df['ATR'] = ta.ATR(df['high_price'], df['low_price'], df['close_price'], timeperiod=14)
    df['ADX'] = ta.ADX(df['high_price'], df['low_price'], df['close_price'], timeperiod=14)
    
    # Убираем строки с NaN значениями
    df = df.dropna(subset=['ATR', 'ADX'])
    
    return df
