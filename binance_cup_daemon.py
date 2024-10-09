import asyncio
import websockets
import json
import csv
from rich.console import Console
from rich.table import Table
from datetime import datetime
import os
import time
from rich.live import Live

# Функция форматирования чисел с добавлением K, M, B
def format_large_number(value):
    if value >= 1_000_000_000:  # Больше или равно миллиарду
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:  # Больше или равно миллиону
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:  # Больше или равно тысяче
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"

# Функция для записи данных в CSV
def save_to_csv(data, filename='data/BTCUSDT_1s.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Время открытия (UTC)', 'Тип сделки', 'Цена', 'Количество (USDT)', 'Кол-во в USDT'])
        writer.writerow(data)

# Асинхронная функция для работы с ордербуком
async def futures_order_book():
    console = Console()
    symbol = "btcusdt"
    depth_url = f"wss://fstream.binance.com/ws/{symbol}@depth20@100ms"
    
    async with websockets.connect(depth_url) as websocket_depth:
        with Live(console=console, refresh_per_second=10) as live:  # Настройка живого отображения
            while True:
                try:
                    # Получаем сообщение с ордеров (стакан)
                    message_depth = await websocket_depth.recv()
                    data_depth = json.loads(message_depth)

                    bids = data_depth['b']  # Покупка
                    asks = data_depth['a']  # Продажа

                    # Средняя цена между покупками и продажами
                    mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2

                    # Создаем таблицу для отображения стакана
                    table = Table(title="Order Book", show_header=True, header_style="bold magenta")
                    table.add_column("Тип", style="dim", width=12)
                    table.add_column("Цена (USDT)", justify="right")
                    table.add_column("Количество (USDT)", justify="right")
                    table.add_column("Кол-во в USDT", justify="right")

                    # Средняя цена между покупками и продажами
                    table.add_row("------", f"{mid_price:.1f}", "------", "------")

                    # Текущее время для записи
                    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

                    # Вывод заявок на продажу в обратном порядке
                    for ask in reversed(asks[:5]):
                        amount_in_usdt = float(ask[1]) * mid_price
                        table.add_row(
                            "sell", 
                            ask[0], 
                            ask[1], 
                            format_large_number(amount_in_usdt)
                        )
                        save_to_csv([current_time, "sell", ask[0], ask[1], format_large_number(amount_in_usdt)])

                    # Вывод заявок на покупку
                    for bid in bids[:5]:
                        amount_in_usdt = float(bid[1]) * mid_price
                        table.add_row(
                            "buy", 
                            bid[0], 
                            bid[1], 
                            format_large_number(amount_in_usdt)
                        )
                        save_to_csv([current_time, "buy", bid[0], bid[1], format_large_number(amount_in_usdt)])

                    # Обновляем таблицу без очистки экрана
                    live.update(table)

                    await asyncio.sleep(0.1)  # Обновляем данные каждые 100 ms

                except websockets.exceptions.ConnectionClosed:
                    console.print("Соединение закрыто, пытаемся переподключиться...")
                    time.sleep(0.1)  # Ждем перед переподключением

                except Exception as e:
                    console.print(f"Произошла ошибка: {e}")
                    break

# Запуск программы
asyncio.get_event_loop().run_until_complete(futures_order_book())
