import pandas as pd
import talib as ta
import os

# Функция для добавления индикаторов только к новым данным
def add_indicators(input_file, output_file):
    # Загрузка существующего файла с индикаторами, если он существует
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        last_existing_timestamp = df_existing['Время открытия (UTC)'].max()  # Последняя запись
    else:
        df_existing = pd.DataFrame()  # Если файл не существует, создаем пустой DataFrame
        last_existing_timestamp = None

    # Загрузка новых данных
    df_new = pd.read_csv(input_file)

    # Если есть уже существующие данные, фильтруем новые записи, которых нет в существующем файле
    if last_existing_timestamp:
        df_new = df_new[df_new['Время открытия (UTC)'] > last_existing_timestamp]

    # Если новых данных нет, завершаем выполнение
    if df_new.empty:
        print(f"Нет новых данных для файла {input_file}")
        return

    # Рассчет индикаторов для новых данных
    df_new['RSI'] = ta.RSI(df_new['Цена закрытия'], timeperiod=14)
    df_new['MACD'], df_new['MACD_signal'], df_new['MACD_hist'] = ta.MACD(df_new['Цена закрытия'], fastperiod=12, slowperiod=26, signalperiod=9)
    df_new['SMA_20'] = ta.SMA(df_new['Цена закрытия'], timeperiod=20)
    df_new['EMA_20'] = ta.EMA(df_new['Цена закрытия'], timeperiod=20)
    df_new['Upper_BB'], df_new['Middle_BB'], df_new['Lower_BB'] = ta.BBANDS(df_new['Цена закрытия'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df_new['OBV'] = ta.OBV(df_new['Цена закрытия'], df_new['Объем'])

    # Объединение существующих данных с новыми
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=['Время открытия (UTC)'], keep='last', inplace=True)

    # Проверяем, существует ли папка для сохранения output файла
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Сохранение объединенных данных с индикаторами
    df_combined.to_csv(output_file, index=False)
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
