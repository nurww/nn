# trading_model.py

import logging
import numpy as np
# from binance.client import Client  # Убедитесь, что у вас установлен binance-python
import torch

# Настройка логирования
logging.basicConfig(
    filename="trading_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

# Класс для торговой модели
class TradingModel:
    def __init__(self, interval_models, orderbook_model, client, risk_management):
        self.interval_models = interval_models  # Словарь моделей интервалов
        self.orderbook_model = orderbook_model  # Модель стакана
        self.client = client  # API клиент Binance
        self.risk_management = risk_management  # Компонент риск-менеджмента

    def get_predictions(self, interval_data, orderbook_data):
        # Получаем прогнозы интервалов
        interval_predictions = {}
        for interval, model in self.interval_models.items():
            interval_input = self.prepare_interval_input(interval_data[interval])
            interval_predictions[interval] = model.predict(interval_input)

        # Получаем прогнозы стакана
        orderbook_input = self.prepare_orderbook_input(orderbook_data)
        orderbook_prediction = self.orderbook_model.predict(orderbook_input)

        return interval_predictions, orderbook_prediction

    def prepare_interval_input(self, data):
        # Преобразование данных интервала для подачи в модель
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).to("cuda")

    def prepare_orderbook_input(self, data):
        # Преобразование данных стакана для подачи в модель
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).to("cuda")

    def decide_action(self, interval_predictions, orderbook_prediction):
        # Логика принятия решений
        # Используем волатильность, дисбаланс bid/ask и прогноз mid_price
        mid_price = orderbook_prediction[0]
        volatility = orderbook_prediction[-1]
        bid_ask_imbalance = orderbook_prediction[3]

        if bid_ask_imbalance > 0.6 and volatility > 0.02:
            action = "BUY"
        elif bid_ask_imbalance < -0.6 and volatility > 0.02:
            action = "SELL"
        else:
            action = "HOLD"

        return action, mid_price

    def execute_trade(self, action, mid_price):
        if action == "BUY":
            quantity = self.risk_management.calculate_position_size(mid_price)
            order = self.client.order_market_buy(
                symbol="BTCUSDT",
                quantity=quantity
            )
            logging.info(f"Executed BUY order: {order}")
        elif action == "SELL":
            quantity = self.risk_management.calculate_position_size(mid_price)
            order = self.client.order_market_sell(
                symbol="BTCUSDT",
                quantity=quantity
            )
            logging.info(f"Executed SELL order: {order}")
        else:
            logging.info("No action taken.")

    def run(self, interval_data, orderbook_data):
        interval_predictions, orderbook_prediction = self.get_predictions(interval_data, orderbook_data)
        action, mid_price = self.decide_action(interval_predictions, orderbook_prediction)
        self.execute_trade(action, mid_price)
