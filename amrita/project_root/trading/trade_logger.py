import logging

# Настройка логов
logging.basicConfig(
    filename='logs/trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_trade(action, price, balance, timestamp, pnl=None):
    trade_info = {
        "action": action,
        "price": price,
        "balance": balance,
        "time": timestamp
    }
    if pnl is not None:
        trade_info["pnl"] = pnl
    logging.info(f"Trade executed: {trade_info}")
    print(f"Trade logged: {trade_info}")
