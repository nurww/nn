import os
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model_builder import LSTMModel

def load_best_model(model_path, input_size, hidden_size, num_layers, dropout_rate):
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем тестовые данные
    test_data = pd.read_csv('data/test_data_with_indicators.csv')
    X_test = test_data[['Цена открытия', 'Максимум', 'Минимум', 'Цена закрытия', 'Объем', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_20', 'EMA_20', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'OBV']].values
    y_test = test_data['Цена закрытия'].values
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=88)

    # Параметры модели, которые были использованы при обучении
    input_size = X_test.shape[1]
    hidden_size = 153
    num_layers = 1
    dropout_rate = 0.3216

    # Ищем лучшую модель в saved_models_cross_validation
    best_model_path_cv = find_best_model('saved_models_cross_validation')
    test_loss_cv = None
    if best_model_path_cv:
        model_cv = load_best_model(best_model_path_cv, input_size, hidden_size, num_layers, dropout_rate).to(device)
        test_loss_cv = test_model(model_cv, test_loader, device)
        print(f"Тестовая потеря лучшей модели из cross-validation: {test_loss_cv}")

    # Ищем лучшую модель в saved_models_with_indicators
    best_model_path_indicators = find_best_model('saved_models_with_indicators')
    test_loss_indicators = None
    if best_model_path_indicators:
        model_indicators = load_best_model(best_model_path_indicators, input_size, hidden_size, num_layers, dropout_rate).to(device)
        test_loss_indicators = test_model(model_indicators, test_loader, device)
        print(f"Тестовая потеря лучшей модели из saved_models_with_indicators: {test_loss_indicators}")
    else:
        print("Не удалось найти модели в saved_models_with_indicators.")
    
    # # Выводим результат сравнения
    # if test_loss_cv is not None and test_loss_indicators is not None:
    #     print(f"Сравнение потерь моделей:\nCross-validation: {test_loss_cv}\nWith indicators: {test_loss_indicators}")
    # elif test_loss_cv is not None:
    #     print(f"Лучшая модель из cross-validation с потерей: {test_loss_cv}")
    # elif test_loss_indicators is not None:
    #     print(f"Лучшая модель из saved_models_with_indicators с потерей: {test_loss_indicators}")
    # else:
    #     print("Ни одна модель не прошла тестирование.")

    # Вывод финальных результатов
    if test_loss_cv is not None or test_loss_indicators is not None:
        results = {
            'Model Type': ['Cross Validation', 'Indicators'],
            'Test Loss': [test_loss_cv, test_loss_indicators]
        }
        print(pd.DataFrame(results))
    else:
        print("Модели не найдены в обеих папках.")
