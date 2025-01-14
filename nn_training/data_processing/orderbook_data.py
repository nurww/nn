# orderbook_data.py

import asyncio
import websockets
import json
import redis.asyncio as aioredis
import numpy as np
import logging
from datetime import datetime, timedelta
import time  # Добавьте импорт модуля time
from decimal import Decimal
import pandas as pd
import signal

# Глобальный флаг для завершения программы
shutdown_flag = asyncio.Event()
# Регистрируем обработчик сигналов
# signal.signal(signal.SIGINT, shutdown_handler)
# signal.signal(signal.SIGTERM, shutdown_handler)

# Конфигурация для Redis
REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'db': 0}

FIXED_BOUNDARIES = {
    "mid_price": {"min": 0, "max": 125000},
    "sum_bid_volume": {"min": 0, "max": 30000},
    "sum_ask_volume": {"min": 0, "max": 30000},
    "imbalance": {"min": -1, "max": 1}
}

INTERVALS = ["1d", "4h", "1h", "15m", "5m", "1m"]

# Настройка логирования
logging.basicConfig(
    filename='logs/normalized_order_book_detailed.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

async def initialize_redis():
    return aioredis.from_url(f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}", db=REDIS_CONFIG['db'])

def min_max_normalize(value, min_val, max_val, precision=25):
    normalized = (value - min_val) / (max_val - min_val)
    clipped = np.clip(normalized, 0, 1)
    return round(clipped, precision)


async def save_to_redis(redis_client, data, key):
    try:
        await redis_client.rpush(key, json.dumps(data))
        if key == "normalized_order_book_stream":
            await redis_client.ltrim(key, -50, -1)
    except Exception as e:
        logging.error(f"Ошибка при сохранении в Redis: {e}")

# Функция для извлечения сигнала от модели интервалов из Redis
async def get_interval_signal(redis_client):
    interval_signal = await redis_client.get("interval_signal")
    return int(interval_signal) if interval_signal is not None else 0  # Возвращаем 0, если сигнала нет

def calculate_indicators(orderbook_data):
    orderbook_data["atr"] = orderbook_data["mid_price"].diff().abs().rolling(window=14).mean()
    orderbook_data["sma"] = orderbook_data["mid_price"].rolling(window=14).mean()
    orderbook_data["mad"] = orderbook_data["mid_price"].rolling(window=14).apply(lambda x: np.mean(np.abs(x - x.mean())))
    orderbook_data["cci"] = (orderbook_data["mid_price"] - orderbook_data["sma"]) / (0.015 * orderbook_data["mad"])
    return orderbook_data

async def get_order_book_history(redis_client, count=50):
    raw_data = await redis_client.lrange("normalized_order_book_stream", -count, -1)
    if not raw_data:
        return pd.DataFrame()  # Если данных нет, возвращаем пустой DataFrame
    records = [json.loads(entry) for entry in raw_data]
    return pd.DataFrame(records)

def calculate_indicators_with_history(orderbook_history: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Рассчитывает индикаторы на основе истории стакана.
    
    :param orderbook_history: DataFrame с историей стакана.
    :param window: Размер окна для расчета индикаторов.
    :return: DataFrame с добавленными индикаторами.
    """
    if orderbook_history.empty or len(orderbook_history) < window:
        logging.warning("Недостаточно данных для расчета индикаторов.")
        return orderbook_history  # Возвращаем исходный DataFrame без изменений
    
    # Рассчитываем индикаторы
    orderbook_history["atr"] = orderbook_history["mid_price"].diff().abs().rolling(window=window).mean()
    orderbook_history["sma"] = orderbook_history["mid_price"].rolling(window=window).mean()
    orderbook_history["mad"] = orderbook_history["mid_price"].rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    orderbook_history["cci"] = (
        (orderbook_history["mid_price"] - orderbook_history["sma"]) /
        (0.015 * orderbook_history["mad"])
    )

    # Убираем строки, где индикаторы не могут быть рассчитаны
    orderbook_history = orderbook_history.dropna().reset_index(drop=True)

    logging.info(f"Индикаторы рассчитаны для {len(orderbook_history)} строк.")
    return orderbook_history

async def get_intervals_predictions(redis_client):
    """
    Получить прогнозы интервалов из Redis для заданного временного диапазона.
    """
    predictions = []
    intervals = ["1d", "4h", "1h", "15m", "5m", "1m"]

    for interval in intervals:
        key = f"interval_predictions_{interval}"
        data = await redis_client.lrange(key, -1, -1)  # Получить все данные из ключа
        if not data:
            continue
        
        # Преобразуем данные из Redis
        record = json.loads(data[0])
        for entry in data:
            record = json.loads(entry)
            predictions.append(record)


    # Преобразуем в DataFrame, если есть данные
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        return predictions_df

    # Возвращаем пустой DataFrame, если данных нет
    return pd.DataFrame()

async def process_and_store_indicators(redis_client):
    """Обрабатывает и добавляет индикаторы в нормализованные данные стакана."""
    # Получаем последние 50 записей из Redis
    raw_data = await redis_client.lrange("normalized_order_book_stream", -50, -1)

    if len(raw_data) < 14:
        logging.warning("Not enough data in Redis for indicator calculation.")
        return  # Недостаточно данных для расчета индикаторов

    # Преобразуем данные из JSON в DataFrame
    records = [json.loads(entry) for entry in raw_data]
    if not records:
        logging.warning("No valid records retrieved from Redis.")
        return

    orderbook_data = pd.DataFrame(records)
    if orderbook_data.empty:
        logging.warning("DataFrame is empty after loading data from Redis.")
        return

    # logging.info(f"Loaded {len(orderbook_data)} rows for indicator calculation.")

    # Рассчитываем индикаторы
    orderbook_data = calculate_indicators(orderbook_data).dropna().reset_index(drop=True)
    if orderbook_data.empty:
        logging.warning("All rows dropped after indicator calculation.")
        return

    # logging.info("Indicators calculated successfully.")

    # Добавляем последнюю строку с индикаторами в новый ключ Redis
    last_row = orderbook_data.iloc[-1].to_dict()
    await redis_client.rpush("normalized_order_book_stream_with_indicators", json.dumps(last_row))
    # logging.info("Last row with indicators saved to Redis.")

async def enrich_with_interval_predictions(redis_client):
    """Обогащает данные индикаторами интервалов."""
    raw_data = await redis_client.lrange("normalized_order_book_stream_with_indicators", -1, -1)
    if not raw_data:
        return  # Нет данных для обогащения

    last_row = json.loads(raw_data[0])
    
    # Список для сохранения значений в порядке
    enriched_values = list(last_row.values())  # Начинаем с текущих значений из last_row

    # Получаем прогнозы для всех интервалов
    for interval in INTERVALS:
        key = f"interval_predictions_{interval}"
        prediction_data = await redis_client.lrange(key, -1, -1)
        if prediction_data:
            prediction = json.loads(prediction_data[0])
            # Добавляем значения по порядку
            enriched_values.extend(prediction.get("prediction", []))  # Разворачиваем "prediction"
            enriched_values.append(prediction.get("trend_price_difference", 0))  # Одно значение
            enriched_values.append(prediction.get("trend_sma", 0))  # Одно значение
            enriched_values.append(prediction.get("trend_strength", 0))  # Одно значение
            enriched_values.append(prediction.get("open_price_normalized", 0))
            enriched_values.append(prediction.get("high_price_normalized", 0))
            enriched_values.append(prediction.get("low_price_normalized", 0))
            enriched_values.append(prediction.get("close_price_normalized", 0))
            enriched_values.append(prediction.get("volume_normalized", 0))
            enriched_values.append(prediction.get("rsi_normalized", 0))
            enriched_values.append(prediction.get("macd_normalized", 0))
            enriched_values.append(prediction.get("macd_signal_normalized", 0))
            enriched_values.append(prediction.get("macd_hist_normalized", 0))
            enriched_values.append(prediction.get("sma_20_normalized", 0))
            enriched_values.append(prediction.get("ema_20_normalized", 0))
            enriched_values.append(prediction.get("upper_bb_normalized", 0))
            enriched_values.append(prediction.get("middle_bb_normalized", 0))
            enriched_values.append(prediction.get("lower_bb_normalized", 0))
            enriched_values.append(prediction.get("obv_normalized", 0))

    # print(enriched_values)
    # print("\n")
    # Сохраняем только значения в новый ключ
    await redis_client.rpush("final_order_book_stream", json.dumps(enriched_values))

# TODO Do not DELETE
# async def enrich_with_interval_predictions(redis_client):
#     """Обогащает данные индикаторами интервалов."""
#     raw_data = await redis_client.lrange("normalized_order_book_stream_with_indicators", -1, -1)
#     if not raw_data:
#         return  # Нет данных для обогащения

#     last_row = json.loads(raw_data[0])
    
#     # Получаем прогнозы для всех интервалов
#     interval_predictions = {}
#     for interval in INTERVALS:
#         key = f"interval_predictions_{interval}"
#         prediction_data = await redis_client.lrange(key, -1, -1)
#         if prediction_data:
#             interval_predictions[interval] = json.loads(prediction_data[0])
    
#     # Добавляем прогнозы в последнюю строку
#     for interval, prediction in interval_predictions.items():
#         last_row[f"{interval}_prediction"] = prediction.get("prediction", [])
#         last_row[f"{interval}_trend_price_difference"] = prediction.get("trend_price_difference", [])
#         last_row[f"{interval}_trend_sma"] = prediction.get("trend_sma", [])
#         last_row[f"{interval}_trend_strength"] = prediction.get("trend_strength", [])
#         last_row[f"{interval}_open_price_normalized"] = prediction.get("open_price_normalized", [])
#         last_row[f"{interval}_high_price_normalized"] = prediction.get("high_price_normalized", [])
#         last_row[f"{interval}_low_price_normalized"] = prediction.get("low_price_normalized", [])
#         last_row[f"{interval}_close_price_normalized"] = prediction.get("close_price_normalized", [])
#         last_row[f"{interval}_volume_normalized"] = prediction.get("volume_normalized", [])
#         last_row[f"{interval}_rsi_normalized"] = prediction.get("rsi_normalized", [])
#         last_row[f"{interval}_macd_normalized"] = prediction.get("macd_normalized", [])
#         last_row[f"{interval}_macd_signal_normalized"] = prediction.get("macd_signal_normalized", [])
#         last_row[f"{interval}_macd_hist_normalized"] = prediction.get("macd_hist_normalized", [])
#         last_row[f"{interval}_sma_20_normalized"] = prediction.get("sma_20_normalized", [])
#         last_row[f"{interval}_ema_20_normalized"] = prediction.get("ema_20_normalized", [])
#         last_row[f"{interval}_upper_bb_normalized"] = prediction.get("upper_bb_normalized", [])
#         last_row[f"{interval}_middle_bb_normalized"] = prediction.get("middle_bb_normalized", [])
#         last_row[f"{interval}_lower_bb_normalized"] = prediction.get("lower_bb_normalized", [])
#         last_row[f"{interval}_obv_normalized"] = prediction.get("obv_normalized", [])

#     print(last_row)
#     print("\n")
#     # Сохраняем обогащенную строку в конечный ключ
#     await redis_client.rpush("final_order_book_stream", json.dumps(last_row))

async def process_order_book_and_store(data, redis_client):
    """
    Обработка сообщения с расчётом индикаторов и добавлением прогнозов.
    """
    try:
        # Используем Decimal для повышения точности
        bids = np.array(data['b'][:8], dtype=object)
        asks = np.array(data['a'][:8], dtype=object)

        # Преобразуем значения в Decimal для повышения точности
        bids = np.array([[Decimal(bid[0]), Decimal(bid[1])] for bid in bids], dtype=object)
        asks = np.array([[Decimal(ask[0]), Decimal(ask[1])] for ask in asks], dtype=object)

        timestamp = datetime.utcnow()

        # Основные показатели стакана
        mid_price = (bids[0, 0] + asks[0, 0]) / Decimal(2)
        sum_bid_volume = np.sum([bid[1] for bid in bids])
        sum_ask_volume = np.sum([ask[1] for ask in asks])
        imbalance = (sum_bid_volume - sum_ask_volume) / (sum_bid_volume + sum_ask_volume) if (sum_bid_volume + sum_ask_volume) > 0 else Decimal(0)

        # Нормализация данных
        normalized_data = {
            # "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "mid_price": min_max_normalize(float(mid_price), FIXED_BOUNDARIES["mid_price"]["min"], FIXED_BOUNDARIES["mid_price"]["max"]),
            "sum_bid_volume": min_max_normalize(float(sum_bid_volume), FIXED_BOUNDARIES["sum_bid_volume"]["min"], FIXED_BOUNDARIES["sum_bid_volume"]["max"]),
            "sum_ask_volume": min_max_normalize(float(sum_ask_volume), FIXED_BOUNDARIES["sum_ask_volume"]["min"], FIXED_BOUNDARIES["sum_ask_volume"]["max"]),
            "imbalance": min_max_normalize(float(imbalance), FIXED_BOUNDARIES["imbalance"]["min"], FIXED_BOUNDARIES["imbalance"]["max"])
        }
        # print(normalized_data)
        # print("\n")
        # Сохраняем данные в Redis
        await save_to_redis(redis_client, normalized_data, "normalized_order_book_stream")

    except Exception as e:
        logging.error(f"Ошибка при обработке данных с индикаторами: {e}")

async def check_safe_exit(redis_client):
    """Проверяет флаг safe_exit в Redis."""
    safe_exit = await redis_client.get("safe_exit")
    return safe_exit == b"1"

async def fetch_order_book_data(redis_client):
    SYMBOL = "btcusdt"
    DEPTH_URL = f"wss://fstream.binance.com/ws/{SYMBOL}@depth20@100ms"

    # while True:
    while not shutdown_flag.is_set():
        connection_start_time = datetime.utcnow()  # Время начала соединения

        try:
            async with websockets.connect(DEPTH_URL) as websocket:
                # while True:
                while not shutdown_flag.is_set():
                    # Проверяем команду safe
                    if await check_safe_exit(redis_client):
                        logging.info("Команда 'safe' получена. Завершаем безопасно...")
                        shutdown_flag.set()
                        break

                    # Проверяем, прошло ли 20 минут
                    if datetime.utcnow() - connection_start_time >= timedelta(minutes=20):
                        logging.info("Переключение на новый Redis клиент через 20 минут.")

                        total_runtime = datetime.utcnow() - connection_start_time
                        logging.info(f"Redis клиент работал {total_runtime}. Переключаемся на новый.")

                        # Создаем новый redis_client
                        next_redis_client = await initialize_redis()

                        # Закрываем старый redis_client
                        try:
                            await redis_client.close()
                        except Exception as e:
                            logging.error(f"Ошибка при закрытии старого Redis клиента: {e}")

                        # Переключаемся на новый redis_client
                        redis_client = next_redis_client
                        connection_start_time = datetime.utcnow()  # Перезапускаем таймер соединения
                        logging.info("Переключение на новый Redis клиент завершено.")
                        break  # Переподключение к WebSocket

                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    await process_order_book_and_store(data, redis_client)
                    await process_and_store_indicators(redis_client)  # Шаг 2
                    await enrich_with_interval_predictions(redis_client)  # Шаг 3

        except (websockets.exceptions.ConnectionClosedError, asyncio.TimeoutError) as e:
            logging.error(f"Ошибка соединения WebSocket: {e}. Повторное подключение...")
            await asyncio.sleep(5)
        except websockets.ConnectionClosed as e:
            logging.error(f"Соединение закрыто: {e}")
            break

async def main():
    redis_client = await initialize_redis()
    # Убедимся, что ключ safe_exit установлен в 0 при старте
    await redis_client.set("safe_exit", "0")
    try:
        await fetch_order_book_data(redis_client)
    finally:
        logging.info("Завершаем соединение с Redis...")
        await redis_client.close()
        

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Программа завершена пользователем (Ctrl + C).")
