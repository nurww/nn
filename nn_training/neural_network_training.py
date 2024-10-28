# neural_network_training.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from model_builder import build_model  # Импорт build_model из model_builder.py
import tqdm
import pandas as pd
from save_predictions import save_prediction_to_db  # Импортируем функцию для сохранения

# Кастомная функция потерь с поощрениями и штрафами
def custom_loss(outputs, targets):
    mse_loss = torch.nn.functional.mse_loss(outputs, targets)
    loss_penalty = torch.mean(torch.relu(targets - outputs)) * 0.5
    reward = torch.mean(torch.relu(outputs - targets)) * 0.5
    return mse_loss + loss_penalty - reward

# Функция для разделения данных
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Функция обучения модели
def train_model(model, train_loader, optimizer, criterion, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    with tqdm.tqdm(total=len(train_loader), desc='Training', unit='batch') as pbar:
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)

    return running_loss / len(train_loader)

# Функция оценки
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_loader), desc='Evaluating', unit='batch') as pbar:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                pbar.update(1)

                # Сохраняем предсказания
                all_predictions.extend(outputs.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    return avg_loss, np.array(all_predictions)

# Стандартное обучение
def train_standard(X_train, y_train, X_test, y_test, model_params, training_params, use_custom_loss=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Преобразуем данные в тензоры
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=training_params['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=training_params['batch_size'])
    
    # Модель
    model = build_model(model_params['model_type'], input_size=X_train.shape[1], output_size=1, **model_params).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    criterion = custom_loss if use_custom_loss else torch.nn.MSELoss()

    # Обучение
    for epoch in range(training_params['epochs']):
        train_loss = train_model(model, train_loader, optimizer, criterion, device, training_params['accumulation_steps'])
        test_loss, predictions = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{training_params['epochs']}, Train Loss: {train_loss}, Test Loss: {test_loss}")

        # Сохраняем предсказания в базу данных
        for prediction in predictions:
            trend_prediction = 1 if prediction > 0 else -1  # Пример: 1 - восходящий тренд, -1 - нисходящий тренд
            confidence = abs(prediction)  # Уверенность может быть абсолютным значением предсказания
            save_prediction_to_db(trend_prediction, confidence)
    
    return model

# Кросс-валидация
def cross_validate(X, y, model_params, training_params, n_splits=5, use_custom_loss=False):
    kf = KFold(n_splits=n_splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{n_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_standard(X_train, y_train, X_test, y_test, model_params, training_params, use_custom_loss)

# Функция для загрузки данных (из файла или базы данных)
def load_data(file_path='data/train_data_with_indicators.csv'):
    data = pd.read_csv(file_path)
    X = data[['Цена открытия', 'Максимум', 'Минимум', 'Цена закрытия', 'Объем', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_20', 'EMA_20', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'OBV']].values
    y = data['Цена закрытия'].values
    return X, y

def main():
    # Загружаем данные
    X, y = load_data()  # Загрузи данные с нужного пути

    # Параметры модели
    model_params = {
        'model_type': 'LSTM',  # или 'Transformer'
        'hidden_size': 153,
        'num_layers': 1,
        'dropout_rate': 0.3
    }

    # Параметры обучения
    training_params = {
        'batch_size': 88,
        'learning_rate': 0.001,
        'accumulation_steps': 2,
        'epochs': 10
    }

    # Стандартное обучение
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_standard(X_train, y_train, X_test, y_test, model_params, training_params, use_custom_loss=True)

    # Кросс-валидация
    cross_validate(X, y, model_params, training_params, use_custom_loss=True)

if __name__ == "__main__":
    main()
