import json
import time
import torch
from reward_system import calculate_reward
from .trade_logger import log_trade
import asyncio
from trading.commands import execute_command

THRESHOLD = 0.01  # Порог для открытия позиции
INITIAL_BALANCE = 10000  # Начальный баланс для гипотетической торговли

# Функция для генерации торгового сигнала
def generate_trade_signal(prediction, threshold=THRESHOLD):
    if prediction > threshold:
        return "BUY"
    elif prediction < -threshold:
        return "SELL"
    else:
        return "HOLD"

# Функция для расчета PnL
def calculate_pnl(trade_direction, entry_price, exit_price):
    if trade_direction == "BUY":
        return exit_price - entry_price
    elif trade_direction == "SELL":
        return entry_price - exit_price
    return 0

# Основная функция для симуляции торговли
async def simulate_trading(model, val_loader, redis_client, device):
    balance = INITIAL_BALANCE
    trade_log = []
    position = None
    entry_price = 0

    interval_signal = await redis_client.get("interval_signal")
    interval_signal = int(interval_signal) if interval_signal else 0  # Получение сигнала интервалов

    for X_val_batch, y_val_batch in val_loader:
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
        predictions = model(X_val_batch)

        for i in range(len(predictions)):
            pred_value = predictions[i].item()
            actual_price = y_val_batch[i].item()
            signal = generate_trade_signal(pred_value)

            if signal == "BUY" and position is None:
                position = "BUY"
                entry_price = actual_price
                log_trade("BUY", entry_price, balance, time.time())
                
            elif signal == "SELL" and position is None:
                position = "SELL"
                entry_price = actual_price
                log_trade("SELL", entry_price, balance, time.time())

            elif signal != "HOLD" and position is not None:
                pnl = calculate_pnl(position, entry_price, actual_price)
                balance += pnl
                log_trade("CLOSE", actual_price, balance, time.time(), pnl)
                position = None

            reward = calculate_reward(pred_value, actual_price)
            if (interval_signal > 0 and pred_value > 0) or (interval_signal < 0 and pred_value < 0):
                reward += 0.5

            trade_log.append({
                "prediction": pred_value,
                "actual": actual_price,
                "reward": reward,
                "signal": signal,
                "interval_signal": interval_signal
            })

    with open("logs/trade_log.json", "w") as file:
        json.dump(trade_log, file, indent=4)

    print(f"Final Balance after simulation: {balance}")
    return balance

async def test_trial_trade():
    # Пример вызова команды trial_trade
    await execute_command("trial_trade")

# Запуск теста
if __name__ == "__main__":
    asyncio.run(test_trial_trade())