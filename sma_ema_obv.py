# sma_ema_obv.py

import talib as ta
import pandas as pd

# Функция для расчета SMA, EMA и OBV
def calculate_sma_ema_obv(df, full_data_df):
    df['SMA_20'] = ta.SMA(df['close_price'], timeperiod=20)
    df['EMA_20'] = ta.EMA(df['close_price'], timeperiod=20)
    
    # Рассчитываем OBV на основе всех данных
    full_data_df['OBV'] = ta.OBV(full_data_df['close_price'], full_data_df['volume'])
    
    # Объединяем df с данными OBV
    df = pd.merge(df, full_data_df[['open_time', 'OBV']], on='open_time', how='left')
    
    # Убираем строки с NaN значениями
    df = df.dropna(subset=['SMA_20', 'EMA_20', 'OBV'])
    
    return df
