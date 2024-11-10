from trading.trade_logger import log_trade

def test_logging():
    # Логирование тестового действия
    log_trade("BUY", 50000, 10000, "2023-01-01 12:00:00", pnl=100)
    print("Check logs/trading.log for the output.")

if __name__ == "__main__":
    test_logging()
