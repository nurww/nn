import torch
import pandas as pd
import numpy as np
from model_builder import LSTMModel

# Загрузка модели
def load_best_model(model_path, input_size, hidden_size, num_layers, dropout_rate):
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path))
    return model

# Функция для симуляции торговли
def simulate_trading(model, test_data, initial_balance, commission_rate=0.001):
    balance = initial_balance
    positions = []
    model.eval()
    
    for i in range(1, len(test_data)):
        current_price = test_data.iloc[i - 1]['Цена закрытия']
        next_price = test_data.iloc[i]['Цена закрытия']
        
        # Предсказание будущей цены
        input_data = torch.tensor(test_data.iloc[i - 1].values, dtype=torch.float32).unsqueeze(0)
        predicted_price = model(input_data).item()

        # Решение об открытии позиции
        if predicted_price > current_price:
            # Открыть длинную позицию (Buy)
            position = {'type': 'buy', 'entry_price': current_price, 'exit_price': next_price}
            positions.append(position)
            balance -= current_price * (1 + commission_rate)
            balance += next_price * (1 - commission_rate)
        elif predicted_price < current_price:
            # Открыть короткую позицию (Sell)
            position = {'type': 'sell', 'entry_price': current_price, 'exit_price': next_price}
            positions.append(position)
            balance += current_price * (1 - commission_rate)
            balance -= next_price * (1 + commission_rate)
    
    return balance, positions

def reward_penalty_system(balance_before, balance_after):
    if balance_after > balance_before:
        return 0.05  # Вознаграждение за прибыльную сделку
    else:
        return -0.05  # Штраф за убыточную сделку

def simulate_trading(model, test_data, initial_balance, commission_rate=0.001):
    balance = initial_balance
    positions = []
    rewards = []
    model.eval()
    
    for i in range(1, len(test_data)):
        current_price = test_data.iloc[i - 1]['Цена закрытия']
        next_price = test_data.iloc[i]['Цена закрытия']
        
        # Предсказание будущей цены
        input_data = torch.tensor(test_data.iloc[i - 1].values, dtype=torch.float32).unsqueeze(0)
        predicted_price = model(input_data).item()

        balance_before = balance
        
        # Решение об открытии позиции
        if predicted_price > current_price:
            # Открыть длинную позицию (Buy)
            balance -= current_price * (1 + commission_rate)
            balance += next_price * (1 - commission_rate)
        elif predicted_price < current_price:
            # Открыть короткую позицию (Sell)
            balance += current_price * (1 - commission_rate)
            balance -= next_price * (1 + commission_rate)
        
        balance_after = balance
        
        # Применение системы поощрений и взысканий
        reward = reward_penalty_system(balance_before, balance_after)
        rewards.append(reward)
        positions.append({'type': 'buy' if predicted_price > current_price else 'sell', 'entry_price': current_price, 'exit_price': next_price, 'reward': reward})
    
    return balance, positions

# Основная функция для торговли
def main():
    initial_balance = 10000  # Начальный баланс
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка данных и модели
    test_data = pd.read_csv('../data/test_data.csv')
    best_model_path = 'saved_models_cross_validation/best_model_fold_4_epoch_9_loss_0.000047722531.pt'
    model = load_best_model(best_model_path, input_size=15, hidden_size=153, num_layers=1, dropout_rate=0.3216).to(device)
    
    # Симуляция торговли
    final_balance, trades = simulate_trading(model, test_data, initial_balance)
    
    # Вывод результатов
    print(f"Итоговый баланс: {final_balance}")
    print(f"Количество сделок: {len(trades)}")
    
    # Сохранение результатов
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('hypothetical_trades.csv', index=False)

if __name__ == "__main__":
    main()
