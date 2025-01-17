# binance_cup_daemon.py

import asyncio
import websockets
import json
import mysql.connector
from datetime import datetime, timezone

# Буфер для хранения данных за последние 10 секунд
buffer = []

# Функция для сохранения данных в MySQL
def save_to_db(buffer, connection):
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO order_book_data (timestamp, mid_price, sum_bid_volume, sum_ask_volume)
            VALUES (%s, %s, %s, %s)
        """
        cursor.executemany(query, buffer)
        connection.commit()
        print(f"Сохранено {len(buffer)} записей в базу данных.")
    except mysql.connector.Error as e:
        print(f"Ошибка записи в MySQL: {e}")
    finally:
        cursor.close()

# Функция для расчета взвешенной средней цены и дисбаланса объемов
def calculate_weighted_mid_price(bids, asks):
    total_bid_weight = sum(float(bid[1]) for bid in bids[:5])
    total_ask_weight = sum(float(ask[1]) for ask in asks[:5])
    
    # Взвешенные средние цены
    weighted_bid_price = sum(float(bid[0]) * float(bid[1]) for bid in bids[:5]) / total_bid_weight
    weighted_ask_price = sum(float(ask[0]) * float(ask[1]) for ask in asks[:5]) / total_ask_weight

    # Взвешенная средняя цена
    weighted_mid_price = (weighted_bid_price + weighted_ask_price) / 2

    # Расчет дисбаланса между покупателями и продавцами
    imbalance = (total_bid_weight - total_ask_weight) / (total_bid_weight + total_ask_weight)
    
    return weighted_mid_price, imbalance

# Асинхронная функция для работы с ордербуком
async def analyze_order_book():
    symbol = "btcusdt"
    depth_url = f"wss://fstream.binance.com/ws/{symbol}@depth20@100ms"

    # Подключение к базе данных
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='binance_data',
            user='root',
            password='root'
        )
        print("Соединение с базой данных установлено.")
    except mysql.connector.Error as e:
        print(f"Ошибка подключения к MySQL: {e}")
        return

    async with websockets.connect(depth_url) as websocket_depth:
        while True:
            try:
                message_depth = await websocket_depth.recv()
                data_depth = json.loads(message_depth)

                bids = data_depth['b']
                asks = data_depth['a']

                # Используем улучшенный анализ
                weighted_mid_price, imbalance = calculate_weighted_mid_price(bids, asks)
                sum_bid_volume = sum(float(bid[1]) for bid in bids[:5])
                sum_ask_volume = sum(float(ask[1]) for ask in asks[:5])

                # Время записи в формате UTC
                timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

                # Добавляем данные в буфер
                buffer.append((timestamp, weighted_mid_price, sum_bid_volume, sum_ask_volume, imbalance))

                # Если буфер достиг 100 обновлений
                if len(buffer) >= 100:
                    save_to_db(buffer, connection)
                    buffer.clear()

                await asyncio.sleep(0.1)

            except websockets.exceptions.ConnectionClosed:
                print("Соединение закрыто, пытаемся переподключиться...")
                await asyncio.sleep(1)
                continue

            except Exception as e:
                print(f"Произошла ошибка: {e}")
                await asyncio.sleep(1)

    connection.close()

# Запуск программы
asyncio.get_event_loop().run_until_complete(analyze_order_book())
