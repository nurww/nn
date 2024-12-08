# import asyncio
# import yaml
# from utils.model_utils import save_model, load_model
# from models.interval_model import IntervalLSTMModel
# from models.orderbook_model import OrderBookGRUModel
# from trading.commands import execute_command
# from utils.indicators import calculate_rsi, calculate_macd

# # Загрузка конфигурации
# def load_config():
#     with open("config/config.yaml", "r") as file:
#         return yaml.safe_load(file)

# # Основной запуск
# async def main():
#     config = load_config()
#     print("Loaded config:", config)
    
#     # Вывод доступных команд
#     print("Available commands: trial_trade, real_trade, exit")

#     while True:
#         command = input("--> ")
#         if command == "exit":
#             print("Exiting...")
#             break
#         elif command == "trial_trade":
#             await execute_command("trial_trade", config)
#         elif command == "real_trade":
#             symbol = input("Enter symbol (e.g., BTCUSDT): ")
#             side = input("Enter side (BUY/SELL): ")
#             quantity = float(input("Enter quantity: "))
#             execute_command("real_trade", symbol, side, quantity)
#         else:
#             print("Unknown command. Available commands: trial_trade, real_trade, exit")

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Импорт моделей
from interval_models import get_intervals_predictions
from orderbook_model import fetch_orderbook_data, predict_orderbook
from trading_model import trading_decision

async def update_interval_predictions():
    timestamp = datetime.utcnow()
    predictions = await get_intervals_predictions(timestamp)
    # Сохраняем или кэшируем прогнозы для модели стакана
    return predictions

async def update_orderbook_predictions(interval_predictions):
    orderbook_data = fetch_orderbook_data()
    orderbook_predictions = predict_orderbook(orderbook_data, interval_predictions)
    # Сохраняем прогнозы для торгующей модели
    return orderbook_predictions

async def execute_trading(orderbook_predictions):
    trading_decision(orderbook_predictions)

async def main():
    scheduler = AsyncIOScheduler()
    
    # Запускаем обновления моделей интервалов
    scheduler.add_job(update_interval_predictions, "interval", seconds=10, args=["1m"])
    scheduler.add_job(update_interval_predictions, "interval", minutes=1, args=["5m"])
    scheduler.add_job(update_interval_predictions, "interval", minutes=5, args=["15m"])
    scheduler.add_job(update_interval_predictions, "interval", minutes=15, args=["1h"])
    scheduler.add_job(update_interval_predictions, "interval", hours=1, args=["4h"])
    scheduler.add_job(update_interval_predictions, "interval", hours=4, args=["1d"])
    
    # Запускаем обновление модели стакана каждую секунду
    scheduler.add_job(update_orderbook_predictions, "interval", seconds=1)
    
    # Запускаем принятие решений торгующей модели каждую секунду
    scheduler.add_job(execute_trading, "interval", seconds=1)

    scheduler.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
