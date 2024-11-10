import os
from binance.client import Client
from .trade_logger import log_trade
import time

API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

client = Client(API_KEY, SECRET_KEY)

# Функция для отправки реального ордера
def place_real_order(symbol, side, quantity, order_type="MARKET"):
    try:
        order = client.order_market(
            symbol=symbol,
            side=side,
            quantity=quantity
        )
        log_trade(side, order["fills"][0]["price"], order["executedQty"], time.time())
        print(f"Order placed: {order}")
    except Exception as e:
        print(f"Error placing order: {e}")
