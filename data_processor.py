import pandas as pd
import os
from tqdm import tqdm
import time
import numpy as np

# Маппинг для правильных названий файлов
interval_mapping = {
    '1M': '1mo',   # Месяц
    '1m': '1min',  # Минута
    '1d': '1d',    # День
    '4h': '4h',    # 4 часа
    '1h': '1h',    # 1 час
    '15m': '15m',  # 15 минут
    '5m': '5m'     # 5 минут
}

# Функция для загрузки уже существующих данных
def load_existing_data(output_filename):
    if os.path.exists(output_filename):
        return pd.read_csv(output_filename)
    return pd.DataFrame()

# Функция для разделения данных на тренировочные и тестовые выборки
def split_data_by_time(file_path, train_size=0.8):
    data = pd.read_csv(f'data/{file_path}')
    split_index = int(np.floor(len(data) * train_size))
    
    train_data = data[:split_index]  # Первые 80% данных для тренировки
    test_data = data[split_index:]   # Последние 20% данных для тестирования
    
    # Сохранение тренировочных и тестовых данных в файлы
    train_data.to_csv('data/train_data_with_indicators.csv', index=False)
    test_data.to_csv('data/test_data_with_indicators.csv', index=False)
    
    print('Тренировочные данные сохранены как train_data_with_indicators.csv')
    print('Тестовые данные сохранены как test_data_with_indicators.csv')
    
    return train_data, test_data

# Функция для загрузки данных из CSV файлов
def load_csv_data(folder_path, symbol, intervals):
    data = {}
    for interval in intervals:
        mapped_interval = interval_mapping.get(interval, interval)
        file_path = os.path.join(folder_path, f'{symbol}_{mapped_interval}_with_indicators.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Проверка на пропущенные значения
            if df.isnull().values.any():
                print(f'Пропущенные значения в {file_path}, удаление строк с пропущенными значениями...')
                df.dropna(inplace=True)
            data[interval] = df
        else:
            print(f'Файл {file_path} не найден.')
    return data

# Функция для объединения данных из разных временных интервалов
def merge_data(data_dict, existing_data):
    merged_data = pd.DataFrame()
    for interval, df in data_dict.items():
        df['Interval'] = interval  # Добавляем столбец для указания интервала
        merged_data = pd.concat([merged_data, df], axis=0)
    
    if not existing_data.empty:
        # Оставляем только новые данные, которых нет в уже существующих
        merged_data = merged_data[~merged_data['Время открытия (UTC)'].isin(existing_data['Время открытия (UTC)'])]
    
    return pd.concat([existing_data, merged_data]).sort_values(by='Время открытия (UTC)').reset_index(drop=True)

# Функция для нормализации данных (масштабирование)
def normalize_data(df, columns, existing_data):
    df_normalized = df.copy()
    for column in columns:
        if not existing_data.empty:
            min_value = min(existing_data[column].min(), df[column].min())
            max_value = max(existing_data[column].max(), df[column].max())
        else:
            min_value = df[column].min()
            max_value = df[column].max()
        df_normalized[column] = (df[column] - min_value) / (max_value - min_value)
    return df_normalized

# Функция для сохранения объединенных и нормализованных данных обратно в CSV
def save_merged_data(df, output_filename):
    output_path = os.path.join('data', output_filename)
    
    # Проверка на существование папки, создание если она отсутствует
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_path, index=False)
    print(f'Объединенные данные сохранены в {output_path}')

# Пример обработки файлов с прогресс-баром
def process_data(symbol, intervals, folder_path='periods_data_with_indicators', output_filename='merged_data_with_indicators.csv'):
    print('Загрузка данных...')
    existing_data = load_existing_data(f'data/{output_filename}')  # Загружаем уже существующие данные
    data_dict = load_csv_data(folder_path, symbol, intervals)
    
    print('Объединение данных...')
    merged_data = merge_data(data_dict, existing_data)
    
    print('Нормализация данных...')
    columns_to_normalize = ['Цена открытия', 'Максимум', 'Минимум', 'Цена закрытия', 'Объем', 'RSI', 'MACD', 'SMA_20', 'EMA_20', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'OBV']
    normalized_data = normalize_data(merged_data, columns_to_normalize, existing_data)
    
    print('Сохранение объединенных данных...')
    save_merged_data(normalized_data, output_filename)
    split_data_by_time(f'{output_filename}')

    with tqdm(total=len(intervals), desc="Обработка данных", unit="файлов") as pbar:
        for interval in intervals:
            time.sleep(1)  # Симуляция обработки каждого файла
            pbar.update(1)
        
        pbar.close()
    
    print('Процесс обработки данных завершен.')

# Пример использования
if __name__ == '__main__':
    symbol = 'BTCUSDT'
    intervals = ['1M', '1d', '4h', '1h', '15m', '5m', '1m']
    
    process_data(symbol, intervals)
