import asyncio
import yaml
from utils.model_utils import save_model, load_model
from models.interval_model import IntervalLSTMModel
from models.orderbook_model import OrderBookGRUModel
from trading.commands import execute_command
from utils.indicators import calculate_rsi, calculate_macd

# Загрузка конфигурации
def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

# Основной запуск
async def main():
    config = load_config()
    print("Loaded config:", config)
    
    # Вывод доступных команд
    print("Available commands: trial_trade, real_trade, exit")

    while True:
        command = input("--> ")
        if command == "exit":
            print("Exiting...")
            break
        elif command == "trial_trade":
            await execute_command("trial_trade", config)
        elif command == "real_trade":
            symbol = input("Enter symbol (e.g., BTCUSDT): ")
            side = input("Enter side (BUY/SELL): ")
            quantity = float(input("Enter quantity: "))
            execute_command("real_trade", symbol, side, quantity)
        else:
            print("Unknown command. Available commands: trial_trade, real_trade, exit")

if __name__ == "__main__":
    asyncio.run(main())
