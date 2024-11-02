from joblib import load
import numpy as np

def load_model(model_path):
    model = load(model_path)
    return model

def run_inference(processed_data, model):
    features = np.column_stack([
        processed_data['mid_prices'],
        processed_data['moving_average'],
        processed_data['bid_volumes'],
        processed_data['ask_volumes'],
        processed_data['imbalances']
    ])
    
    predictions = model.predict(features)
    return predictions

# Пример запуска предсказаний
if __name__ == "__main__":
    model = load_model('model/random_forest_model.joblib')
    # Загрузите и обработайте данные для теста, как в предыдущих скриптах
    processed_data = {
        "mid_prices": np.array([0.657, 0.658, 0.659]),
        "moving_average": np.array([0.6575, 0.658, 0.6585]),
        "bid_volumes": np.array([0.0015, 0.0014, 0.0016]),
        "ask_volumes": np.array([0.0003, 0.00035, 0.0004]),
        "imbalances": np.array([0.84, 0.83, 0.85])
    }
    results = run_inference(processed_data, model)
    print("Предсказания:", results)
