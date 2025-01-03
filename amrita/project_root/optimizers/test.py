# trading_model.py

import sys
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
import pandas as pd
import json
import logging
from datetime import datetime
import time
import pandas as pd
import mplfinance as mpf

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from project_root.data.database_manager import execute_query

# Функция для получения последних данных интервала
def fetch_interval_data(interval: str, sequence_length: int, last_open_time: datetime) -> pd.DataFrame:
    query = f"""
        SELECT * FROM binance_klines
        WHERE data_interval = '{interval}'
        AND open_time <= '{last_open_time}'
        ORDER BY open_time DESC
        LIMIT {sequence_length}
    """
    data = execute_query(query)
    if not data.empty:
        data = data.iloc[::-1]  # Реверсируем, чтобы данные были в хронологическом порядке
    return data

def calculate_filtered_targets(data, window_size, rrr_threshold=2, min_potential_pnl=0.01, max_hold_time=15):
    """
    Генерация таргетов с учетом фильтров RRR, потенциальной прибыли и времени удержания.
    """
    print(data)
    
    data = data.iloc[::-1].reset_index(drop=True)
    targets = []
    for i in range(len(data) - window_size):
        entry_price = data['open_price'].iloc[i]
        window_data = data.iloc[i + 1:i + 1 + window_size]
        
        # Long
        max_price = window_data['high_price'].max()
        min_price = window_data['low_price'].min()
        # print(f"Max: {window_data['high_price'].max()}")
        # print(f"Min: {window_data['low_price'].min()}")
        potential_pnl_long = (max_price - entry_price) / entry_price
        max_negative_long = (entry_price - min_price) / entry_price
        rrr_long = potential_pnl_long / max_negative_long if max_negative_long > 0 else 0
        
        # Short
        potential_pnl_short = (entry_price - min_price) / entry_price
        max_negative_short = (max_price - entry_price) / entry_price
        rrr_short = potential_pnl_short / max_negative_short if max_negative_short > 0 else 0
        
        # Выбираем лучшее направление
        if potential_pnl_long > min_potential_pnl and rrr_long >= rrr_threshold:
            direction = 0  # long
            exit_price = max_price
            potential_pnl = potential_pnl_long
            max_negative_diff = max_negative_long
            # hold_time = (window_data.index[window_data['high_price'].idxmax()] - data.index[i]).total_seconds() / 60
        elif potential_pnl_short > min_potential_pnl and rrr_short >= rrr_threshold:
            direction = 1  # short
            exit_price = min_price
            potential_pnl = potential_pnl_short
            max_negative_diff = max_negative_short
            # hold_time = (window_data.index[window_data['low_price'].idxmin()] - data.index[i]).total_seconds() / 60
        else:
            print(f"{i} step, continue, no deal found")
            continue  # Если ни одно направление не проходит фильтры, пропускаем
        
        # Пропускаем сделки с длительностью больше max_hold_time
        # if hold_time > max_hold_time:
        #     print(f"{i} step, time limit")
        #     continue
        
        # Добавляем данные в "target"
        targets.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'potential_pnl': potential_pnl,
            'max_negative_diff': max_negative_diff,
            # 'hold_time': hold_time
        })
    print(targets)
    return pd.DataFrame(targets)

# Запуск программы
if __name__ == "__main__":
    interval = '1m'
    interval_sequence = 150
    last_open_time = "2024-11-26 00:00:00"
    data = fetch_interval_data(interval, interval_sequence, last_open_time)
    # data = data.iloc[::-1].reset_index(drop=True)
    results = calculate_filtered_targets(data, 15)

    # Преобразуем столбец open_time в datetime
    data['open_time'] = pd.to_datetime(data['open_time'], format='%d.%m.%Y %H:%M')

    # Установим индекс в формате datetime для работы с mplfinance
    data.set_index('open_time', inplace=True)

    # Переименуем столбцы в формат, который использует mplfinance
    data.rename(columns={
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'close_price': 'Close'
    }, inplace=True)

    # # Построим график свечей
    # mpf.plot(data, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', volume=False)

    # Преобразуем результаты в DataFrame
    results_df = pd.DataFrame(results)

    # Добавим точки сделок на график
    entry_markers = []
    exit_markers = []

    # Привязка сделок к временным меткам
    for idx, trade in results_df.iterrows():
        if trade['direction'] == 0:  # Long
            # Найдем временную метку для entry_price
            entry_time = data.index[(data['Open'] <= trade['entry_price']) & (data['High'] >= trade['entry_price'])]
            exit_time = data.index[(data['Low'] <= trade['exit_price']) & (data['High'] >= trade['exit_price'])]

            # Добавляем точки на график только если временные метки найдены
            if not entry_time.empty and not exit_time.empty:
                entry_markers.append(
                    mpf.make_addplot(
                        [trade['entry_price'] if idx == entry_time[0] else np.nan for idx in data.index],
                        type='scatter',
                        marker='^',
                        markersize=70,
                        color='green'
                    )
                )
                exit_markers.append(
                    mpf.make_addplot(
                        [trade['exit_price'] if idx == exit_time[0] else np.nan for idx in data.index],
                        type='scatter',
                        marker='v',
                        markersize=70,
                        color='lime'
                    )
                )
        elif trade['direction'] == 1:  # Short
            entry_time = data.index[(data['Open'] >= trade['entry_price']) & (data['Low'] <= trade['entry_price'])]
            exit_time = data.index[(data['Low'] <= trade['exit_price']) & (data['High'] >= trade['exit_price'])]

            if not entry_time.empty and not exit_time.empty:
                entry_markers.append(
                    mpf.make_addplot(
                        [trade['entry_price'] if idx == entry_time[0] else np.nan for idx in data.index],
                        type='scatter',
                        marker='v',
                        markersize=70,
                        color='red'
                    )
                )
                exit_markers.append(
                    mpf.make_addplot(
                        [trade['exit_price'] if idx == exit_time[0] else np.nan for idx in data.index],
                        type='scatter',
                        marker='^',
                        markersize=70,
                        color='pink'
                    )
                )

    # Построим график со сделками
    mpf.plot(data, type='candle', style='charles', title='Candlestick Chart with Trades',
            ylabel='Price', addplot=entry_markers + exit_markers)