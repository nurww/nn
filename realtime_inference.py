import time
import json
import redis
import torch
import numpy as np
from datetime import datetime
from your_model import YourModel  # Импортируйте свою обученную модель

# Настройки Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=1)  # Второй Redis для хранения ордербука

# Путь к модели
MODEL_PATH = "models/your_model.pth"

# Загрузка обученной модели
model = YourModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Переводим модель в режим оценки (inference)

def get_latest_data_from_redis(redis_client, window_size=600):
    # Получаем последние 600 записей из Redis
    data = [json.loads(item) for item in redis_client.lrange("order_book_stream", -window_size, -1)]
    return data

def preprocess_data(data):
    # Преобразуем данные в формат numpy для модели
    # Предполагается, что каждая запись содержит 'bid', 'ask', и 'volume'
    features = [[entry['bid'], entry['ask'], entry['volume']] for entry in data]
    return np.array(features)

def run_inference(model, data):
    # Преобразуем данные в тензор
    data_tensor = torch.tensor(data, dtype=torch.float32)
    # Прогоняем через модель
    with torch.no_grad():
        output = model(data_tensor)
    return output

def log_prediction(prediction):
    # Логируем предсказание в файл
    with open("logs/realtime_inference.log", "a") as log_file:
        log_file.write(f"{datetime.utcnow()} - Prediction: {prediction}\n")

def main():
    while True:
        # Получаем последние 600 записей из Redis
        latest_data = get_latest_data_from_redis(redis_client)
        
        # Преобразуем в numpy-массив для подачи в модель
        processed_data = preprocess_data(latest_data)
        
        # Подаем данные модели и получаем предсказание
        prediction = run_inference(model, processed_data)
        
        # Логируем предсказание
        log_prediction(prediction)
        
        # Задержка в 100 мс
        time.sleep(0.1)

if __name__ == "__main__":
    main()
