from binance.client import Client
from binance.exceptions import BinanceAPIException
import json

with open("acc_config.json", "r") as file:
        acc_config = json.load(file)

# Укажите свои API ключи
API_KEY = acc_config["API_KEY"]
API_SECRET = acc_config["API_SECRET"]

# Инициализация клиента
client = Client(API_KEY, API_SECRET)

# Получение баланса кошелька
def get_wallet_balance():
    try:
        account_info = client.get_account()
        balances = account_info['balances']
        print("Your non-zero balances:")
        for asset in balances:
            free_balance = float(asset['free'])
            locked_balance = float(asset['locked'])
            if free_balance > 0 or locked_balance > 0:
                print(f"{asset['asset']} -> Free: {free_balance}, Locked: {locked_balance}")
    except BinanceAPIException as e:
        print(f"Error fetching wallet balance: {e}")

# Перевод средств между спотом и фьючерсами
def transfer_between_wallets(asset, amount, direction):
    """
    direction: 1 - from spot to futures
               2 - from futures to spot
    """
    try:
        result = client.futures_account_transfer(asset=asset, amount=amount, type=direction)
        print(f"Transfer successful: {result}")
    except BinanceAPIException as e:
        print(f"Error during transfer: {e}")

# Открытие фьючерсной позиции
def open_futures_position(symbol, side, quantity, price=None):
    """
    symbol: Trading pair (e.g., "BTCUSDT")
    side: "BUY" or "SELL"
    quantity: Quantity to trade
    price: Optional limit price
    """
    try:
        if price:
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                timeInForce="GTC",
                quantity=quantity,
                price=price
            )
        else:
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
        print(f"Order successful: {order}")
    except BinanceAPIException as e:
        print(f"Error creating order: {e}")

# Пример использования функций
if __name__ == "__main__":
    # Получение баланса
    print("Wallet balances:")
    get_wallet_balance()

    # # Пример перевода средств
    # print("\nTransferring BTC from spot to futures:")
    # transfer_between_wallets(asset="BTC", amount=0.00000002, direction=2)

    # # Открытие рыночной фьючерсной позиции
    # print("\nOpening futures position:")
    # open_futures_position(symbol="BTCUSDT", side="BUY", quantity=0.001)
