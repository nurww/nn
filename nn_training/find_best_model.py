import os
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model_builder import build_model  # Универсальная функция для сборки LSTM и Transformer моделей
import mysql.connector
from mysql.connector import Error

# Функция для загрузки модели с учетом типа
def load_best_model(model_type, model_path, input_size, **model_params):
    model = build_model(
        model_type=model_type, 
        input_size=input_size, 
        output_size=1, 
        **model_params
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

# Функция для тестирования модели
def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    return test_loss / len(test_loader)

# Функция для поиска лучшей модели в директории
def find_best_model(directory):
    best_loss = float('inf')
    best_model_path = None

    # Регулярное выражение для поиска значения loss в имени файла
    pattern = re.compile(r'_loss_(-?[0-9]+\.[0-9]+)\.pt')

    # Поиск всех файлов моделей в директории
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            loss = float(match.group(1))
            if loss < best_loss:
                best_loss = loss
                best_model_path = os.path.join(directory, filename)

    if best_model_path:
        print(f"Лучшая модель в {directory}: {best_model_path} с потерей {best_loss}")
    else:
        print(f"Не удалось найти модели в {directory}.")

    return best_model_path

# Функция для загрузки тестовых данных из базы данных MySQL
def load_test_data_from_db(connection, table_name, interval):
    query = f"""
        SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized,
               volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized,
               sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
        FROM {table_name}
        WHERE data_interval = '{interval}'
    """
    df = pd.read_sql(query, connection)
    X_test = df[['open_price_normalized', 'high_price_normalized', 'low_price_normalized', 'close_price_normalized',
                 'volume_normalized', 'rsi_normalized', 'macd_normalized', 'macd_signal_normalized', 'macd_hist_normalized',
                 'sma_20_normalized', 'ema_20_normalized', 'upper_bb_normalized', 'middle_bb_normalized', 'lower_bb_normalized',
                 'obv_normalized']].values
    y_test = df['close_price_normalized'].values
    return X_test, y_test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Подключаемся к базе данных MySQL
    connection = mysql.connector.connect(
        host='localhost',
        database='binance_data',
        user='root',
        password='root'
    )

    try:
        # Загрузка тестовых данных из MySQL
        X_test, y_test = load_test_data_from_db(connection, 'binance_klines_normalized', '1m')

        # Преобразуем данные в тензоры для PyTorch
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=88)

        # Параметры модели, которые были использованы при обучении
        input_size = X_test.shape[1]
        
        # Определяем параметры для LSTM и Transformer моделей
        lstm_params = {
            'hidden_size': 153,
            'num_layers': 1,
            'dropout_rate': 0.3216
        }

        transformer_params = {
            'num_heads': 4,
            'num_encoder_layers': 2,
            'hidden_dim': 256,
            'dropout_rate': 0.1
        }

        # Ищем лучшую модель в saved_models_cross_validation
        best_model_path_cv = find_best_model('saved_models_cross_validation')
        test_loss_cv = None
        if best_model_path_cv:
            model_cv = load_best_model(
                model_type='LSTM',  # Выбираем LSTM или Transformer
                model_path=best_model_path_cv, 
                input_size=input_size, 
                **lstm_params
            ).to(device)
            test_loss_cv = test_model(model_cv, test_loader, device)
            print(f"Тестовая потеря лучшей модели из cross-validation: {test_loss_cv}")

        # Ищем лучшую модель в saved_models_with_indicators
        best_model_path_indicators = find_best_model('saved_models_with_indicators')
        test_loss_indicators = None
        if best_model_path_indicators:
            model_indicators = load_best_model(
                model_type='Transformer',  # Выбираем LSTM или Transformer
                model_path=best_model_path_indicators, 
                input_size=input_size, 
                **transformer_params
            ).to(device)
            test_loss_indicators = test_model(model_indicators, test_loader, device)
            print(f"Тестовая потеря лучшей модели из saved_models_with_indicators: {test_loss_indicators}")
        else:
            print("Не удалось найти модели в saved_models_with_indicators.")
        
        # Выводим финальные результаты сравнения
        if test_loss_cv is not None or test_loss_indicators is not None:
            results = {
                'Model Type': ['Cross Validation', 'Indicators'],
                'Test Loss': [test_loss_cv, test_loss_indicators]
            }
            print(pd.DataFrame(results))
        else:
            print("Модели не найдены в обеих папках.")
    finally:
        # Закрываем соединение с базой данных
        if connection.is_connected():
            connection.close()
