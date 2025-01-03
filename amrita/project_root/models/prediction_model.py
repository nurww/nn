# trading_model.py

import sys
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException
import torch
import torch.nn as nn
import json
import logging
from datetime import datetime
import time

# Добавляем текущий путь к проекту в sys.path для корректного импорта
amrita = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(amrita)

from project_root.data.database_manager import execute_query

# Настройка логирования
logging.basicConfig(
    filename=f'../logs/model_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Инициализация модели прогнозирования.
        :param input_size: Размер входных данных (количество признаков).
        :param hidden_size: Размер скрытых слоев GRU.
        :param num_layers: Количество слоев GRU.
        :param output_size: Размер выходных данных (например, 4: направление, точка входа, SL, TP).
        :param dropout: Вероятность дропаут слоя.
        """
        super(PredictionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Прямой проход модели.
        :param x: Входные данные формы (batch_size, sequence_length, input_size).
        :return: Выходные данные формы (batch_size, output_size).
        """
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)  # GRU возвращает последовательности и последнее скрытое состояние
        return self.fc(out[:, -1, :])  # Прогноз только для последнего таймстепа

class DecisionEngine:
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model

    def make_decision(self, market_data, positions):
        """
        Принимает решение на основе прогнозной модели и текущих позиций.
        """
        # Получить прогноз
        prediction = self.prediction_model.predict(market_data)

        # Проверить активные позиции
        if not positions:
            # Если позиций нет, возможно, стоит открыть сделку
            if prediction['direction'] in ['long', 'short']:
                return {
                    "action": "open",
                    "direction": prediction['direction'],
                    "entry_price": prediction['entry_price'],
                    "stop_loss": prediction['stop_loss'],
                    "take_profit": prediction['take_profit']
                }
        else:
            # Если есть активные позиции, проверить условия закрытия
            active_position = positions[0]  # Предполагаем одну позицию
            if active_position['direction'] == 'long' and market_data['price'] <= active_position['stop_loss']:
                return {"action": "close", "reason": "stop_loss_hit"}
            if active_position['direction'] == 'short' and market_data['price'] >= active_position['stop_loss']:
                return {"action": "close", "reason": "stop_loss_hit"}
            # Обновление Stop Loss/Take Profit
            if prediction.get('update_sl_tp'):
                return {
                    "action": "update_sl_tp",
                    "stop_loss": prediction['stop_loss'],
                    "take_profit": prediction['take_profit']
                }
        return {"action": "hold"}

class OrderManager:
    def __init__(self, binance_client):
        self.client = binance_client

    def open_position(self, direction, entry_price, stop_loss, take_profit, quantity):
        """
        Открыть новую сделку на Binance.
        """
        try:
            order_type = "BUY" if direction == "long" else "SELL"
            order = self.client.futures_create_order(
                symbol="BTCUSDT",
                side=order_type,
                type="LIMIT",
                price=entry_price,
                quantity=quantity,
                timeInForce="GTC"
            )
            return {"status": "success", "order_id": order['orderId']}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close_position(self, position_id):
        """
        Закрыть активную сделку.
        """
        try:
            order = self.client.futures_cancel_order(
                symbol="BTCUSDT",
                orderId=position_id
            )
            return {"status": "success", "order_id": order['orderId']}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_balance(self):
        """
        Получить баланс фьючерсного счета.
        """
        return self.client.futures_account_balance()

class PositionMonitor:
    def __init__(self, binance_client):
        self.client = binance_client

    def get_active_positions(self):
        """
        Получить текущие активные сделки.
        """
        positions = self.client.futures_position_information()
        return [pos for pos in positions if float(pos['positionAmt']) != 0]

    def calculate_pnl(self, position):
        """
        Рассчитать прибыль/убыток для активной сделки.
        """
        entry_price = float(position['entryPrice'])
        mark_price = float(position['markPrice'])
        quantity = float(position['positionAmt'])
        return (mark_price - entry_price) * quantity if quantity > 0 else (entry_price - mark_price) * abs(quantity)

def initialize_binance_client():
    with open("acc_config.json", "r") as file:
        acc_config = json.load(file)

    # Укажите свои API ключи
    API_KEY = acc_config["API_KEY"]
    API_SECRET = acc_config["API_SECRET"]

    return Client(API_KEY, API_SECRET)

def load_prediction_model():
    model_path = "path/to/your/model.pth"
    model = PredictionModel()  # Замените на класс вашей модели
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Переводим модель в режим оценки
    return model

def fetch_market_data(binance_client):
    client = binance_client
    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_1MINUTE
    klines = client.get_klines(symbol=symbol, interval=interval, limit=1)  # Последняя свеча
    return {
        "price": float(klines[-1][4]),  # Закрытие последней свечи
        "high": float(klines[-1][2]),  # High
        "low": float(klines[-1][3]),  # Low
        "volume": float(klines[-1][5])  # Volume
    }
def predict_market(market_data, model):
    """
    Прогноз рыночного поведения на основе входных данных.
    :param market_data: Входные данные формы (sequence_length, input_size).
    :param model: Загруженная модель.
    :return: Прогноз [направление, точка входа, SL, TP].
    """
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(market_data, dtype=torch.float32).unsqueeze(0)  # Добавляем размер батча
        prediction = model(data_tensor)
        return prediction.squeeze(0).cpu().numpy()

def main():
    # Инициализация клиентов и моделей
    binance_client = initialize_binance_client()
    prediction_model = load_prediction_model()
    
    decision_engine = DecisionEngine(prediction_model)
    order_manager = OrderManager(binance_client)
    position_monitor = PositionMonitor(binance_client)

    # Основной цикл
    while True:
        # Получить рыночные данные
        market_data = fetch_market_data(binance_client)

        # Получить текущие позиции
        positions = position_monitor.get_active_positions()

        # Принять решение
        decision = decision_engine.make_decision(market_data, positions)

        # Выполнить действие
        if decision["action"] == "open":
            order_manager.open_position(
                decision["direction"],
                decision["entry_price"],
                decision["stop_loss"],
                decision["take_profit"],
                quantity=0.01  # Пример объема
            )
        elif decision["action"] == "close":
            order_manager.close_position(positions[0]['positionId'])
        elif decision["action"] == "update_sl_tp":
            # Обновить параметры SL/TP
            pass
        time.sleep(1)  # Задержка перед следующей итерацией
