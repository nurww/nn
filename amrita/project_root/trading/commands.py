from trading.trial_trade import simulate_trading
from trading.real_trade import place_real_order

def execute_command(command, *args):
    if command == "trial_trade":
        return simulate_trading(*args)
    elif command == "real_trade":
        symbol, side, quantity = args
        return place_real_order(symbol, side, quantity)
    else:
        print("Unknown command")
