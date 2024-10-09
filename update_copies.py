import pandas as pd
import os

def update_copy(original_file, copy_file, last_timestamp_file):
    # Чтение последнего обработанного таймстампа
    if os.path.exists(last_timestamp_file):
        with open(last_timestamp_file, 'r') as f:
            last_timestamp = f.read().strip()
    else:
        last_timestamp = None

    # Загрузка оригинального файла
    df_original = pd.read_csv(original_file)

    # Фильтрация новых данных
    if last_timestamp:
        df_new = df_original[df_original['Время открытия (UTC)'] > last_timestamp]
    else:
        df_new = df_original

    if not df_new.empty:
        # Загрузка копии файла
        if os.path.exists(copy_file):
            df_copy = pd.read_csv(copy_file)
            # Удаление строк с дубликатами на всякий случай
            df_copy.drop_duplicates(subset=['Время открытия (UTC)'], keep='last', inplace=True)
        else:
            df_copy = pd.DataFrame()

        # Добавление новых данных к копии
        df_copy = pd.concat([df_copy, df_new], ignore_index=True)
        df_copy.drop_duplicates(subset=['Время открытия (UTC)'], keep='last', inplace=True)
        df_copy.to_csv(copy_file, index=False)

        # Обновление последнего таймстампа
        new_last_timestamp = df_new['Время открытия (UTC)'].max()
        with open(last_timestamp_file, 'w') as f:
            f.write(str(new_last_timestamp))

        print(f"Копия {copy_file} обновлена до {new_last_timestamp}")
    else:
        print(f"Нет новых данных для {original_file}")

def main():
    # Папки с исходными данными и копиями
    original_folder = 'periods_data'
    copy_folder = 'copies_for_trading'

    if not os.path.exists(copy_folder):
        os.makedirs(copy_folder)

    # Список файлов для копирования
    files = [
        'BTCUSDT_1d.csv',
        'BTCUSDT_1h.csv',
        'BTCUSDT_1min.csv',
        'BTCUSDT_1mo.csv',
        'BTCUSDT_4h.csv',
        'BTCUSDT_5m.csv',
        'BTCUSDT_15m.csv'
    ]

    for file in files:
        original_file = os.path.join(original_folder, file)
        copy_file = os.path.join(copy_folder, file)
        last_ts_file = os.path.join(copy_folder, f"{file}_last_timestamp.txt")
        update_copy(original_file, copy_file, last_ts_file)

if __name__ == "__main__":
    main()
