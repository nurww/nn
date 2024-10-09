import asyncio
import websockets
import json
from rich.console import Console
from rich.table import Table

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

async def futures_order_book():
    console = Console()
    symbol = "btcusdt"
    depth_url = f"wss://fstream.binance.com/ws/{symbol}@depth20@100ms"

    async with websockets.connect(depth_url) as websocket_depth:
        while True:
            try:
                # Получаем сообщение с ордеров (стакан)
                message_depth = await websocket_depth.recv()
                data_depth = json.loads(message_depth)

                bids = data_depth['b']  # Покупка
                asks = data_depth['a']  # Продажа

                # Используем среднюю цену между лучшими предложениями покупки и продажи
                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2

                # Создаем таблицу для отображения стакана
                table = Table(title="Order Book", show_header=True, header_style="bold magenta")
                table.add_column("Тип", style="dim", width=12)
                table.add_column("Цена (USDT)", justify="right")
                table.add_column("Количество (USDT)", justify="right")
                table.add_column("Кол-во в USDT", justify="right")

                # Средняя цена между покупками и продажами
                table.add_row("------", f"{mid_price:.1f}", "------", "------")

                # Вывод заявок на продажу в обратном порядке
                sum_ask = 0
                for ask in reversed(asks[:5]):  # Обратный порядок
                    sum_ask += float(ask[1])
                    amount_in_usdt = float(ask[1]) * mid_price
                    table.add_row(
                        "Продажа", 
                        ask[0], 
                        ask[1], 
                        format_large_number(amount_in_usdt)
                    )

                # Вывод заявок на покупку (затем покупка)
                sum_bid = 0
                for bid in bids[:5]:
                    sum_bid += float(bid[1])
                    amount_in_usdt = float(bid[1]) * mid_price
                    table.add_row(
                        "Покупка", 
                        bid[0], 
                        bid[1], 
                        format_large_number(amount_in_usdt)
                    )

                # Очищаем консоль и выводим таблицу
                console.clear()
                console.print(table)

                await asyncio.sleep(0.1)  # Обновляем данные каждые 100 ms

            except websockets.exceptions.ConnectionClosed:
                console.print("Соединение закрыто, пытаемся переподключиться...")
                break

# Запуск программы
asyncio.get_event_loop().run_until_complete(futures_order_book())
