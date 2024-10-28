# multi_agent_decision.py

import torch
from transformer_model import TransformerModel
from stable_baselines3 import PPO

def load_model(agent_id):
    model = TransformerModel(...)  # Инициализируйте модель с правильными параметрами
    model.load_state_dict(torch.load(f'saved_models_agent_{agent_id}/best_model.pt'))
    model.eval()
    return model

def get_agent_signal(model, data):
    with torch.no_grad():
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        prediction = model(input_tensor).item()
        return prediction

def combine_signals(signals):
    # Пример простого усреднения сигналов
    return sum(signals) / len(signals)

def main():
    # Загрузите модели агентов
    agents = ['short_term', 'medium_term', 'long_term']
    models = {agent: load_model(agent) for agent in agents}

    # Получите текущие данные
    current_data = ...  # Получите текущие данные из базы или буфера

    # Получите сигналы от каждого агента
    signals = [get_agent_signal(model, current_data) for model in models.values()]

    # Комбинируйте сигналы
    final_signal = combine_signals(signals)

    # Примите торговое решение
    if final_signal > 0.5:
        action = 'Buy'
    elif final_signal < -0.5:
        action = 'Sell'
    else:
        action = 'Hold'

    print(f"Final Trading Action: {action}")

if __name__ == '__main__':
    main()
