import matplotlib.pyplot as plt
import pandas as pd
import talib as ta
import os

# Функция для добавления индикаторов
def add_indicators(input_file, output_file):
    # Загрузка данных
    df = pd.read_csv(input_file)

    # Убедитесь, что столбцы называются так, как ожидается для индикаторов
    # df['Date'] = pd.to_datetime(df['Время открытия (UTC)'])  # Приводим столбец времени к нужному формату
    # df.set_index('Date', inplace=True)

    # Рассчет индикаторов
    df['RSI'] = ta.RSI(df['Цена закрытия'], timeperiod=14)  # Индекс относительной силы (RSI)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Цена закрытия'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
    df['SMA_20'] = ta.SMA(df['Цена закрытия'], timeperiod=20)  # Простая скользящая средняя
    df['EMA_20'] = ta.EMA(df['Цена закрытия'], timeperiod=20)  # Экспоненциальная скользящая средняя
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = ta.BBANDS(df['Цена закрытия'], timeperiod=20, nbdevup=2, nbdevdn=2)  # Полосы Боллинджера
    df['OBV'] = ta.OBV(df['Цена закрытия'], df['Объем'])  # Индикатор балансового объема (OBV)

    # Сохранение данных с индикаторами
    df.to_csv(output_file, index=False)
    print(f"Индикаторы успешно добавлены и сохранены в {output_file}")

# Список файлов и их периодов
files = [
    ('copies_for_trading/BTCUSDT_1d.csv', 'periods_data_with_indicators/BTCUSDT_1d_with_indicators.csv'),
    ('copies_for_trading/BTCUSDT_1h.csv', 'periods_data_with_indicators/BTCUSDT_1h_with_indicators.csv'),
    ('copies_for_trading/BTCUSDT_1min.csv', 'periods_data_with_indicators/BTCUSDT_1min_with_indicators.csv'),
    ('copies_for_trading/BTCUSDT_1mo.csv', 'periods_data_with_indicators/BTCUSDT_1mo_with_indicators.csv'),
    ('copies_for_trading/BTCUSDT_4h.csv', 'periods_data_with_indicators/BTCUSDT_4h_with_indicators.csv'),
    ('copies_for_trading/BTCUSDT_5m.csv', 'periods_data_with_indicators/BTCUSDT_5m_with_indicators.csv'),
    ('copies_for_trading/BTCUSDT_15m.csv', 'periods_data_with_indicators/BTCUSDT_15m_with_indicators.csv')
]

# Применение индикаторов ко всем файлам
for input_file, output_file in files:
    if os.path.exists(input_file):
        add_indicators(input_file, output_file)

