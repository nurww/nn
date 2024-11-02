import numpy as np

def process_data(data):
    # Преобразование данных в numpy массив
    timestamps = [entry['timestamp'] for entry in data]
    mid_prices = np.array([entry['mid_price'] for entry in data])
    bid_volumes = np.array([entry['sum_bid_volume'] for entry in data])
    ask_volumes = np.array([entry['sum_ask_volume'] for entry in data])
    imbalances = np.array([entry['imbalance'] for entry in data])

    # Расчет скользящей средней для mid_price (пример)
    moving_average = np.convolve(mid_prices, np.ones(10) / 10, mode='valid')
    
    return {
        "timestamps": timestamps,
        "mid_prices": mid_prices,
        "moving_average": moving_average,
        "bid_volumes": bid_volumes,
        "ask_volumes": ask_volumes,
        "imbalances": imbalances
    }

# Тест обработки данных
if __name__ == "__main__":
    data = [
        {"timestamp": "2024-11-02 01:40:22.332", "mid_price": 0.657, "sum_bid_volume": 0.0015, "sum_ask_volume": 0.0003, "imbalance": 0.84},
        # Добавьте больше примеров для теста
    ]
    processed_data = process_data(data)
    print(processed_data)
