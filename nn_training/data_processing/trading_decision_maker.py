# trading_decision_maker.py

import redis
import json
import numpy as np
from datetime import datetime

# Конфигурация для Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# Функция для извлечения прогнозов из Redis
def fetch_predictions_from_redis(redis_client, keys):
    predictions = {}
    for key in keys:
        data = redis_client.lrange(key, -1, -1)  # Получение последнего прогноза
        if data:
            predictions[key] = json.loads(data[0])
    return predictions

# Логика для принятия торговых решений
def make_trading_decision(predictions):
    # Извлечение прогнозов для анализа
    interval_prediction = predictions.get('interval_prediction')
    orderbook_prediction = predictions.get('orderbook_prediction')

    if not interval_prediction or not orderbook_prediction:
        print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - Недостаточно данных для принятия решения")
        return None

    # Пример стратегии принятия решений
    if interval_prediction['mid_price'] > orderbook_prediction['mid_price'] * 1.001:
        decision = 'BUY'  # Покупка, если ожидается рост цены
    elif interval_prediction['mid_price'] < orderbook_prediction['mid_price'] * 0.999:
        decision = 'SELL'  # Продажа, если ожидается падение цены
    else:
        decision = 'HOLD'  # Удержание позиции, если нет значительных изменений

    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - Принято решение: {decision}")
    return decision

# Функция для отправки решения в систему исполнения
def execute_trade(decision):
    if decision == 'BUY':
        print("Исполнение покупки...")
        # Реализуйте логику вызова API для покупки
    elif decision == 'SELL':
        print("Исполнение продажи...")
        # Реализуйте логику вызова API для продажи
    else:
        print("Удержание позиции, действия не требуются.")

# Основная функция для выполнения цикла принятия решений
def main():
    redis_client = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=True
    )

    prediction_keys = ['interval_prediction', 'orderbook_prediction']

    while True:
        predictions = fetch_predictions_from_redis(redis_client, prediction_keys)
        decision = make_trading_decision(predictions)
        if decision:
            execute_trade(decision)

        # Задержка 100 мс перед следующей итерацией
        time.sleep(0.1)

if __name__ == "__main__":
    main()
