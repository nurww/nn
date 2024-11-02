# config.py

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}
FIXED_BOUNDARIES = {
    "mid_price": {"min": 40000, "max": 85000},
    "sum_bid_volume": {"min": 0, "max": 15000},
    "sum_ask_volume": {"min": 0, "max": 15000},
    "imbalance": {"min": -1, "max": 1}
}
