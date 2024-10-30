import asyncio
import websockets
import json
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Параметры для подключения к вебсокету Binance
SYMBOL = "btcusdt"
DEPTH_URL = f"wss://fstream.binance.com/ws/{SYMBOL}@depth20@100ms"

async def fetch_order_book_data():
    async with websockets.connect(DEPTH_URL) as websocket:
        data_buffer = []
        
        for _ in range(20):  # 20 обновлений за 2 секунды (каждые 100 мс)
            message = await websocket.recv()
            data = json.loads(message)
            bids = data['b']
            asks = data['a']

            # Расчёт основных параметров
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            sum_bid_volume = sum(float(bid[1]) for bid in bids[:5])
            sum_ask_volume = sum(float(ask[1]) for ask in asks[:5])
            bid_ask_imbalance = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume)
            
            # Взвешенная средняя цена
            weighted_mid_price = (
                sum(float(bid[0]) * float(bid[1]) for bid in bids[:5]) +
                sum(float(ask[0]) * float(ask[1]) for ask in asks[:5])
            ) / (sum_bid_volume + sum_ask_volume)
            
            # Глубина объёма по заявкам на покупку и продажу
            sum_bid_depth_volume = sum(float(bid[1]) for bid in bids)
            sum_ask_depth_volume = sum(float(ask[1]) for ask in asks)

            # Добавление записи в буфер для логирования и дальнейшего анализа
            data_buffer.append({
                "timestamp": timestamp,
                "mid_price": mid_price,
                "sum_bid_volume": sum_bid_volume,
                "sum_ask_volume": sum_ask_volume,
                "weighted_mid_price": weighted_mid_price,
                "bid_ask_imbalance": bid_ask_imbalance,
                "sum_bid_depth_volume": sum_bid_depth_volume,
                "sum_ask_depth_volume": sum_ask_depth_volume
            })

            # Логирование данных
            logging.info(
                f"timestamp: {timestamp}, mid_price: {mid_price:.2f}, sum_bid_volume: {sum_bid_volume:.2f}, "
                f"sum_ask_volume: {sum_ask_volume:.2f}, weighted_mid_price: {weighted_mid_price:.2f}, "
                f"bid_ask_imbalance: {bid_ask_imbalance:.4f}, sum_bid_depth_volume: {sum_bid_depth_volume:.2f}, "
                f"sum_ask_depth_volume: {sum_ask_depth_volume:.2f}"
            )

            # Задержка на 100 мс для сбора данных с частотой 100 мс
            await asyncio.sleep(0.1)

        # Вывод данных в виде таблицы после завершения сбора
        print("\nСобранные данные за 2 секунды:\n")
        print("timestamp                | mid_price | sum_bid_volume | sum_ask_volume | weighted_mid_price | bid_ask_imbalance | sum_bid_depth_volume | sum_ask_depth_volume")
        print("-" * 130)
        for entry in data_buffer:
            print(
                f"{entry['timestamp']} | {entry['mid_price']:.2f}    | {entry['sum_bid_volume']:.2f}         | "
                f"{entry['sum_ask_volume']:.2f}         | {entry['weighted_mid_price']:.2f}            | "
                f"{entry['bid_ask_imbalance']:.4f}            | {entry['sum_bid_depth_volume']:.2f}            | "
                f"{entry['sum_ask_depth_volume']:.2f}"
            )

# Запуск функции
asyncio.run(fetch_order_book_data())
