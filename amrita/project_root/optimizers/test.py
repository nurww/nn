# from decimal import Decimal

# def denormalize(value):
#     """
#     Денормализует значение на основе диапазона.
#     """
#     min_value = 0
#     max_value = 115000
#     return value * (max_value - min_value) + min_value

# def test(size):
#     print(f"Size: {size}")
#     # TEST 1
#     test_position = {
#         "direction": "long",  # Длинная позиция
#         "entry_price": Decimal("0.798157"),  # Цена входа
#         "position_size": Decimal(size),  # Размер позиции
#     }

#     current_price = Decimal("0.798045")
#     print(denormalize(current_price))
#     print(denormalize(test_position["entry_price"]))

#     def test_calculate_pnl(position, current_price):
#         """
#         Рассчитывает PnL без учета дополнительных корректировок.
#         """
#         if position["direction"] == "long":
#             pnl = (current_price - position["entry_price"]) * position["position_size"]
#         elif position["direction"] == "short":
#             pnl = (position["entry_price"] - current_price) * position["position_size"]
#         else:
#             pnl = Decimal("0")
#         return pnl

#     # Рассчитаем и выведем PnL
#     pnl = test_calculate_pnl(test_position, current_price)
#     print(f"Рассчитанный PnL: {pnl:.10f}")

#     # Сравнение с ожидаемым значением
#     actual_pnl = {
#         100: -0.01,
#         1000: -0.12,
#         10000: -1.39,
#         100000: -14.02,
#         1000000: -140.31,
#         10000000: -1403.22,
#     }.get(size, None)

#     if actual_pnl is not None:
#         print(f"Actual PnL: {actual_pnl:.10f}")

#     print("____________________________________________________")

# # Тесты
# for size in [100, 1000, 10000, 100000, 1000000, 10000000]:
#     test(size)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal

# def denormalize(value):
#     """
#     Денормализует значение на основе диапазона.
#     """
#     min_value = 0
#     max_value = 115000
#     return value * (max_value - min_value) + min_value

# def test(size):
#     print(f"Size: {size}")
#     # TEST 1
#     test_position = {
#         "direction": "long",  # Длинная позиция
#         "entry_price": Decimal("0.798157"),  # Цена входа
#         "position_size": Decimal(size),  # Размер позиции
#     }

#     current_price = Decimal("0.798045")

#     def test_calculate_pnl_with_adjustment(position, current_price, adjustment_threshold=10000, adjustment_factor=1.20):
#         """
#         Рассчитывает PnL с учетом корректировки для больших позиций.
#         """
#         # Рассчитаем PnL
#         if position["direction"] == "long":
#             pnl = (current_price - position["entry_price"]) * position["position_size"]
#         elif position["direction"] == "short":
#             pnl = (position["entry_price"] - current_price) * position["position_size"]
#         else:
#             pnl = Decimal("0")
        
#         # Если размер позиции превышает порог, добавляем поправочный коэффициент
#         if position["position_size"] > adjustment_threshold:
#             pnl *= Decimal(adjustment_factor)
        
#         return pnl

#     # Рассчитаем и выведем PnL
#     pnl = test_calculate_pnl_with_adjustment(test_position, current_price)
#     print(f"Рассчитанный PnL: {pnl:.10f}")

#     # Сравнение с ожидаемым значением
#     actual_pnl = {
#         100: -0.01,
#         1000: -0.12,
#         10000: -1.39,
#         100000: -14.02,
#         1000000: -140.31,
#         10000000: -1403.22,
#     }.get(size, None)

#     if actual_pnl is not None:
#         print(f"Actual PnL: {actual_pnl:.2f}")

#     print("____________________________________________________")

# # Тесты
# for size in [100, 1000, 10000, 100000, 1000000, 10000000]:
#     test(size)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal

# def calculate_initial_margin(quantity, entry_price, leverage):
#     """
#     Рассчитывает начальную маржу для USDⓈ-Margined Contracts.
#     Initial Margin = Quantity * Entry Price * IMR
#     IMR = 1 / leverage
#     """
#     imr = Decimal("1") / Decimal(leverage)
#     return quantity * entry_price * imr

# def calculate_pnl(quantity, entry_price, exit_price, position_side):
#     """
#     Рассчитывает PnL (Profit and Loss) для USDⓈ-Margined Contracts.
#     Long position: PnL = (Exit Price - Entry Price) * Quantity
#     Short position: PnL = (Entry Price - Exit Price) * Quantity
#     """
#     print(f"Quantity: {quantity} {entry_price} {exit_price}")
#     if position_side == "long":
#         return (exit_price - entry_price) * quantity
#     elif position_side == "short":
#         return (entry_price - exit_price) * quantity
#     else:
#         raise ValueError("Invalid position_side. Use 'long' or 'short'.")

# def calculate_roi(pnl, initial_margin):
#     """
#     Рассчитывает ROI (Return on Investment).
#     ROI% = PnL / Initial Margin
#     """
#     return pnl / initial_margin * 100

# # Пример теста
# def test_futures_calculations(size):
#     # Параметры сделки
#     quantity = Decimal(size)  # Размер позиции
#     entry_price = Decimal("91788.055000")  # Цена входа
#     exit_price = Decimal("91775.175000")  # Цена выхода
#     leverage = Decimal("50")  # Плечо
#     position_side = "long"  # Длинная позиция ("long" или "short")

#     # Расчеты
#     initial_margin = calculate_initial_margin(quantity, entry_price, leverage)
#     pnl = calculate_pnl(quantity, entry_price, exit_price, position_side)
#     roi = calculate_roi(pnl, initial_margin)

#     # Вывод результатов
#     print(f"Initial Margin: {initial_margin:.10f} USDT")
#     print(f"PnL: {pnl:.10f} USDT")
#     print(f"ROI: {roi:.2f}%")

# # Тестируем
# test_futures_calculations(10)
# print("____________________________________________________")
# test_futures_calculations(100)
# print("____________________________________________________")
# test_futures_calculations(1000)
# print("____________________________________________________")
# test_futures_calculations(10000)
# print("____________________________________________________")
# test_futures_calculations(100000)
# print("____________________________________________________")
# test_futures_calculations(1000000)
# print("____________________________________________________")


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal

# def denormalize(value):
#     """
#     Денормализует значение на основе диапазона.
#     """
#     min_value = 0
#     max_value = 115000
#     return value * (max_value - min_value) + min_value

# def test(size):
#     print(f"Size: {size}")
#     # TEST 1
#     test_position = {
#         "direction": "long",  # Длинная позиция
#         "entry_price": Decimal("0.798157"),  # Цена входа
#         "position_size": Decimal(size),  # Размер позиции
#     }

#     current_price = Decimal("0.798045")
#     print(denormalize(current_price))
#     print(denormalize(test_position["entry_price"]))

#     def test_calculate_pnl(position, current_price):
#         """
#         Рассчитывает PnL с учетом комиссии.
#         """
#         # Параметры комиссии
#         maker_fee_rate = Decimal("0.0002")  # Maker Fee (0.02%)
#         taker_fee_rate = Decimal("0.0005")  # Taker Fee (0.05%)
#         com = Decimal("0.1")

#         # Рассчитываем комиссию (используем position_size напрямую)
#         commission = position["position_size"] * taker_fee_rate
#         print(f"commission: {commission} {position['position_size']}")

#         # Рассчитываем PnL
#         if position["direction"] == "long":
#             print(((current_price - position["entry_price"]) * position["position_size"]))
#             print(((current_price - position["entry_price"]) * position["position_size"]) * com)
#             pnl = (current_price - position["entry_price"]) * position["position_size"] - commission
#         elif position["direction"] == "short":
#             pnl = (position["entry_price"] - current_price) * position["position_size"] - commission
#         else:
#             pnl = Decimal("0")
#         return pnl

#     # Рассчитаем и выведем PnL
#     pnl = test_calculate_pnl(test_position, current_price)
#     print(f"Рассчитанный PnL: {pnl:.10f}")

#     # Сравнение с ожидаемым значением
#     actual_pnl = {
#         100: -0.01,
#         1000: -0.12,
#         10000: -1.39,
#         100000: -14.02,
#         1000000: -140.31,
#         10000000: -1403.22,
#     }.get(size, None)

#     if actual_pnl is not None:
#         print(f"Actual PnL: {actual_pnl:.10f}")

#     print("____________________________________________________")

# # Тесты
# for size in [100, 1000, 10000, 100000, 1000000, 10000000]:
#     test(size)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal

# def denormalize(value):
#     """
#     Денормализует значение на основе диапазона.
#     """
#     min_value = 0
#     max_value = 115000
#     return value * (max_value - min_value) + min_value

# def test(size):
#     print(f"Size: {size}")
#     # TEST 1
#     test_position = {
#         "direction": "long",  # Длинная позиция
#         "entry_price": Decimal("8000"),  # Цена входа
#         "position_size": Decimal(size),  # Размер позиции
#     }

#     current_price = Decimal("18000")
#     print(denormalize(current_price))
#     print(denormalize(test_position["entry_price"]))

#     def test_calculate_pnl(position, current_price):
#         """
#         Рассчитывает PnL с учетом комиссии.
#         """
#         # Параметры комиссии
#         maker_fee_rate = Decimal("0.0002")  # Maker Fee (0.02%)
#         taker_fee_rate = Decimal("0.0005")  # Taker Fee (0.05%)
#         com = Decimal("0.1")

#         # Рассчитываем комиссию (используем position_size напрямую)
#         commission = position["position_size"] * taker_fee_rate
#         print(f"commission: {commission} {position['position_size']}")

#         # Рассчитываем PnL
#         if position["direction"] == "long":
#             # print(((current_price - position["entry_price"]) * position["position_size"]))
#             # print(((current_price - position["entry_price"]) * position["position_size"]) * com)
#             print(f"{current_price} {position['entry_price']} {position['position_size']}")
#             pnl = (current_price - position["entry_price"]) * position["position_size"]
#             print(Decimal("1250") / (current_price - position["entry_price"]))
#         elif position["direction"] == "short":
#             pnl = (position["entry_price"] - current_price) * position["position_size"]
#         else:
#             pnl = Decimal("0")
#         return pnl
    
#     # Рассчитаем и выведем PnL
#     pnl = test_calculate_pnl(test_position, current_price)
#     print(f"Рассчитанный PnL: {pnl:.10f}")

#     # Сравнение с ожидаемым значением
#     actual_pnl = {
#         100: -0.01,
#         1000: -0.12,
#         10000: -1.39,
#         100000: -14.02,
#         1000000: -140.31,
#         10000000: -1403.22,
#     }.get(size, None)

#     if actual_pnl is not None:
#         print(f"Actual PnL: {actual_pnl:.10f}")

#     print("____________________________________________________")

# # Тесты
# for size in [100, 1000, 10000, 100000, 1000000, 10000000]:
#     test(size)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal

# def calculate_pnl(entry_price, exit_price, quantity, leverage=20):
#     entry_price = Decimal(entry_price)
#     exit_price = Decimal(exit_price)
#     quantity = Decimal(quantity)
#     leverage = Decimal(leverage)

#     # Calculate Initial Margin
#     initial_margin = quantity / leverage

#     # Calculate PNL
#     pnl = quantity * (exit_price - entry_price) / entry_price

#     # Calculate ROI
#     roi = (pnl / initial_margin) * 100

#     return {
#         "Entry Price": entry_price,
#         "Exit Price": exit_price,
#         "Quantity": quantity,
#         "Initial Margin": round(initial_margin, 2),
#         "PNL": round(pnl, 2),
#         "ROI (%)": round(roi, 2)
#     }

# # Тестируем на ваших данных
# examples = [
#     {"entry_price": 8000, "exit_price": 8100, "quantity": 100},
#     {"entry_price": 18000, "exit_price": 18100, "quantity": 100},
#     {"entry_price": 28000, "exit_price": 28100, "quantity": 100},
#     {"entry_price": 38000, "exit_price": 38100, "quantity": 100},
#     {"entry_price": 48000, "exit_price": 48100, "quantity": 100},
#     {"entry_price": 58000, "exit_price": 58100, "quantity": 100},
#     {"entry_price": 68000, "exit_price": 68100, "quantity": 100},
#     {"entry_price": 78000, "exit_price": 78100, "quantity": 100},
#     {"entry_price": 80000, "exit_price": 80100, "quantity": 100},
# ]

# for example in examples:
#     result = calculate_pnl(**example)
#     print(result)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal, getcontext

# # Настройка точности Decimal
# getcontext().prec = 10

# def calculate_pnl(entry_price, exit_price, quantity, leverage=20):
#     entry_price = Decimal(entry_price)
#     exit_price = Decimal(exit_price)
#     quantity = Decimal(quantity)
#     leverage = Decimal(leverage)

#     # Calculate Initial Margin
#     initial_margin = quantity / leverage

#     # Calculate PNL
#     pnl = quantity * (exit_price - entry_price) / entry_price

#     # Calculate ROI
#     roi = (pnl / initial_margin) * 100 if initial_margin > 0 else Decimal("0")

#     return {
#         "Entry Price": entry_price,
#         "Exit Price": exit_price,
#         "Quantity": quantity,
#         "Initial Margin": round(initial_margin, 2),
#         "PNL": round(pnl, 2),
#         "ROI (%)": round(roi, 2),
#     }

# # Примеры из ваших данных
# examples = [
#     {"entry_price": 8000, "exit_price": 8100, "quantity": 100},
#     {"entry_price": 18000, "exit_price": 18100, "quantity": 100},
#     {"entry_price": 28000, "exit_price": 28100, "quantity": 100},
#     {"entry_price": 38000, "exit_price": 38100, "quantity": 100},
#     {"entry_price": 48000, "exit_price": 48100, "quantity": 100},
#     {"entry_price": 58000, "exit_price": 58100, "quantity": 100},
#     {"entry_price": 68000, "exit_price": 68100, "quantity": 100},
#     {"entry_price": 78000, "exit_price": 78100, "quantity": 100},
#     {"entry_price": 80000, "exit_price": 80100, "quantity": 100},
# ]

# # Расчет
# for example in examples:
#     result = calculate_pnl(**example)
#     print(result)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal, ROUND_DOWN

# def calculate_binance_metrics_corrected(entry_price, exit_price, quantity, leverage):
#     """
#     Рассчитывает Initial Margin, PNL и ROI с учетом верной интерпретации Quantity.
#     """
#     # Quantity интерпретируется как вложение с учетом плеча
#     initial_margin = (quantity / leverage).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     # PNL
#     pnl = ((exit_price - entry_price) * initial_margin * leverage / entry_price).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     # ROI
#     roi = (pnl / initial_margin * Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     return {
#         "Entry price": f"{entry_price:.2f} USDT",
#         "Exit price": f"{exit_price:.2f} USDT",
#         "Quantity": f"{quantity:.2f} USDT",
#         "Initial margin": f"{initial_margin:.2f} USDT",
#         "PNL": f"{pnl:.2f} USDT",
#         "ROI": f"{roi:.2f} %"
#     }

# # Примеры из калькулятора
# examples = [
#     {"entry_price": Decimal('8000'), "exit_price": Decimal('8100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('18000'), "exit_price": Decimal('18100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('28000'), "exit_price": Decimal('28100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('38000'), "exit_price": Decimal('38100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('48000'), "exit_price": Decimal('48100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('58000'), "exit_price": Decimal('58100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('68000'), "exit_price": Decimal('68100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('78000'), "exit_price": Decimal('78100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('80000'), "exit_price": Decimal('80100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
# ]

# # Расчёт
# results = [calculate_binance_metrics_corrected(**example) for example in examples]

# # Вывод
# for result in results:
#     print(result)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal, ROUND_DOWN

# def calculate_binance_metrics(entry_price, exit_price, quantity, leverage, position="long"):
#     """
#     Рассчитывает Initial Margin, PNL и ROI.
#     """
#     # Initial Margin
#     imr = Decimal('1') / leverage  # IMR = 1 / leverage
#     print(f"{imr} {quantity} {entry_price}")
#     print(f"{imr * (quantity * entry_price * imr)} {quantity} {entry_price}")
#     initial_margin = (quantity * entry_price * imr).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     quantity = Decimal("1.2")
#     # PNL
#     if position == "long":
#         pnl = ((exit_price - entry_price) * quantity).quantize(Decimal('0.01'), rounding=ROUND_DOWN)
#     elif position == "short":
#         pnl = ((entry_price - exit_price) * quantity).quantize(Decimal('0.01'), rounding=ROUND_DOWN)
#     else:
#         raise ValueError("Position must be 'long' or 'short'")

#     # ROI
#     roi = (pnl / initial_margin * Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     return {
#         "Entry price": f"{entry_price:.2f} USDT",
#         "Exit price": f"{exit_price:.2f} USDT",
#         "Quantity": f"{quantity:.2f} USDT",
#         "Initial margin": f"{initial_margin:.2f} USDT",
#         "PNL": f"{pnl:.2f} USDT",
#         "ROI": f"{roi:.2f} %",
#         "Leverage": f"{leverage}"
#     }

# # Примеры из калькулятора
# examples = [
#     {"entry_price": Decimal('8000'), "exit_price": Decimal('8100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('8000'), "exit_price": Decimal('8100'), "quantity": Decimal('100'), "leverage": Decimal('50')},
#     {"entry_price": Decimal('8000'), "exit_price": Decimal('8100'), "quantity": Decimal('100'), "leverage": Decimal('100')},
#     {"entry_price": Decimal('18000'), "exit_price": Decimal('18100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('28000'), "exit_price": Decimal('28100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('38000'), "exit_price": Decimal('38100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('78000'), "exit_price": Decimal('78100'), "quantity": Decimal('100'), "leverage": Decimal('30')},
#     {"entry_price": Decimal('80000'), "exit_price": Decimal('80100'), "quantity": Decimal('100'), "leverage": Decimal('80')},
# ]

# # Расчёт
# results = [calculate_binance_metrics(**example) for example in examples]

# # Вывод
# for result in results:
#     print(result)


# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")
# print("____________________________________________________")


# from decimal import Decimal, ROUND_DOWN

# def calculate_binance_metrics(entry_price, exit_price, quantity, leverage):
#     # Коэффициенты
#     k = Decimal('0.00012')  # Для Initial Margin
#     c = Decimal('0.00012')  # Для PnL

#     # Initial Margin
#     initial_margin = (k * quantity * entry_price / leverage).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     # PnL
#     pnl = (c * (exit_price - entry_price) * quantity).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     # ROI
#     roi = (pnl / initial_margin * Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_DOWN)

#     return {
#         "Entry price": f"{entry_price:.2f} USDT",
#         "Exit price": f"{exit_price:.2f} USDT",
#         "Quantity": f"{quantity:.2f} USDT",
#         "Initial margin": f"{initial_margin:.2f} USDT",
#         "PNL": f"{pnl:.2f} USDT",
#         "ROI": f"{roi:.2f} %",
#         "Leverage": f"{leverage}"
#     }

# # Примеры из калькулятора
# examples = [
#     {"entry_price": Decimal('8000'), "exit_price": Decimal('8100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('8000'), "exit_price": Decimal('8100'), "quantity": Decimal('100'), "leverage": Decimal('50')},
#     {"entry_price": Decimal('8000'), "exit_price": Decimal('8100'), "quantity": Decimal('100'), "leverage": Decimal('100')},
#     {"entry_price": Decimal('18000'), "exit_price": Decimal('18100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('18000'), "exit_price": Decimal('18100'), "quantity": Decimal('100'), "leverage": Decimal('50')},
#     {"entry_price": Decimal('18000'), "exit_price": Decimal('18100'), "quantity": Decimal('100'), "leverage": Decimal('100')},
#     {"entry_price": Decimal('28000'), "exit_price": Decimal('28100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('28000'), "exit_price": Decimal('28100'), "quantity": Decimal('100'), "leverage": Decimal('50')},
#     {"entry_price": Decimal('28000'), "exit_price": Decimal('28100'), "quantity": Decimal('100'), "leverage": Decimal('100')},
#     {"entry_price": Decimal('38000'), "exit_price": Decimal('38100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('48000'), "exit_price": Decimal('48100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('58000'), "exit_price": Decimal('58100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('68000'), "exit_price": Decimal('68100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('78000'), "exit_price": Decimal('78100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('78000'), "exit_price": Decimal('78100'), "quantity": Decimal('100'), "leverage": Decimal('30')},
#     {"entry_price": Decimal('78000'), "exit_price": Decimal('78100'), "quantity": Decimal('100'), "leverage": Decimal('80')},
#     {"entry_price": Decimal('80000'), "exit_price": Decimal('80100'), "quantity": Decimal('100'), "leverage": Decimal('20')},
#     {"entry_price": Decimal('80000'), "exit_price": Decimal('80100'), "quantity": Decimal('100'), "leverage": Decimal('30')},
#     {"entry_price": Decimal('80000'), "exit_price": Decimal('80100'), "quantity": Decimal('100'), "leverage": Decimal('80')},
# ]

# from decimal import Decimal

# # examples = [
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('100'), "leverage": Decimal('25')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('200'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('300'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('200'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('300'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('800'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('900'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('800'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('900'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('800'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('900'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('800'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('8000'), "exit_price": Decimal('8500'), "quantity": Decimal('900'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('100'), "leverage": Decimal('25')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('200'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('300'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('200'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('300'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('800'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('900'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('800'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('900'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('800'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('900'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('800'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('38000'), "exit_price": Decimal('38500'), "quantity": Decimal('900'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('100'), "leverage": Decimal('25')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('200'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('300'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('200'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('300'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('800'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('900'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('800'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('900'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('800'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('900'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('800'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('58000'), "exit_price": Decimal('58500'), "quantity": Decimal('900'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('100'), "leverage": Decimal('25')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('200'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('300'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('200'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('300'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('800'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('900'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('800'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('900'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('800'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('900'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('800'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('78000'), "exit_price": Decimal('78500'), "quantity": Decimal('900'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('100'), "leverage": Decimal('25')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('200'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('300'), "leverage": Decimal('35')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('200'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('300'), "leverage": Decimal('45')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('800'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('900'), "leverage": Decimal('65')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('800'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('900'), "leverage": Decimal('75')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('800'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('900'), "leverage": Decimal('97')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('800'), "leverage": Decimal('108')},
# #     {"entry_price": Decimal('98000'), "exit_price": Decimal('98500'), "quantity": Decimal('900'), "leverage": Decimal('108')}
# # ]

# # Расчёт
# results = [calculate_binance_metrics(**example) for example in examples]

# # Вывод
# for result in results:
#     print(result)


# entry_price = 91726.75
# maintenance_margin_rate = 0.005
# maintenance_margin = 1250 * maintenance_margin_rate
# futures_balance = 20

# liquidation_price = entry_price + (futures_balance - maintenance_margin) / (1250 / entry_price)
# print(liquidation_price)

# maintenance_margin = 1250 * 0.00255

# liquidation_price = entry_price + (futures_balance - maintenance_margin) / (1250 / entry_price)
# print(liquidation_price)

# entry_price = 90000
# maintenance_margin_rate = 0.005
# maintenance_margin = 1250 * maintenance_margin_rate

# liquidation_price = entry_price + (futures_balance - maintenance_margin) / (1250 / entry_price)
# print(liquidation_price)

# maintenance_margin = 1250 * 0.00255

# liquidation_price = entry_price + (futures_balance - maintenance_margin) / (1250 / entry_price)
# print(liquidation_price)

# entry_price = 100000
# maintenance_margin_rate = 0.005
# maintenance_margin = 1250 * maintenance_margin_rate

# liquidation_price = entry_price + (futures_balance - maintenance_margin) / (1250 / entry_price)
# print(liquidation_price)

# maintenance_margin = 1250 * 0.00255

# liquidation_price = entry_price + (futures_balance - maintenance_margin) / (1250 / entry_price)
# print(liquidation_price)


# print(1000 // 10)
# print(500 // 10)
# print(500 // 5)


# import logging


# class BalanceManager:
#     def __init__(self, spot_balance, futures_balance):
#         self.spot_balance = spot_balance
#         self.futures_balance = futures_balance
#         self.initial_total_balance = spot_balance + futures_balance
#         self.BB = max(self.initial_total_balance / 10, 1)  # Минимум 1
#         self.target_spot_balance = 4 * self.BB
#         self.target_futures_balance = self.initial_total_balance - self.target_spot_balance

#     def redistribute_balance(self, initial=False):
#         total_balance = self.spot_balance + self.futures_balance

#         if initial:
#             self.BB = max(total_balance / 10, 1)
#             self.target_spot_balance = 4 * self.BB
#             self.target_futures_balance = total_balance - self.target_spot_balance
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = total_balance - self.spot_balance
#             logging.info(
#                 f"Initial redistribution: Spot balance: {self.spot_balance:.5f}, "
#                 f"Futures balance: {self.futures_balance:.5f}, BB: {self.BB:.5f}, Total balance: {total_balance:.5f}"
#             )
#             return

#         # Условие перераспределения при достижении порога
#         if self.futures_balance >= 94.75:  # Заданное условие для перераспределения
#             # Увеличиваем спот до 175% от начального
#             self.spot_balance = round(self.spot_balance * 1.75, 5)
#             # Оставшееся назначаем на фьючерсы
#             self.futures_balance = total_balance - self.spot_balance

#             logging.info(
#                 f"Triggered redistribution: Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )
#         else:
#             logging.info(
#                 f"No redistribution needed. Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )


# def simulate_changes(manager, steps=10, increment=1):
#     logging.info("Starting simulation...")
#     for step in range(steps):
#         logging.info(f"Step {step + 1}:")
#         # Увеличиваем или уменьшаем фьючерсный баланс
#         manager.futures_balance += increment
#         manager.redistribute_balance()
#         logging.info(
#             f"After change: Spot balance: {manager.spot_balance:.5f}, "
#             f"Futures balance: {manager.futures_balance:.5f}, Total balance: {manager.spot_balance + manager.futures_balance:.5f}"
#         )


# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Инициализация начальных значений
# spot_balance = 28.0
# futures_balance = 89.75
# manager = BalanceManager(spot_balance, futures_balance)

# # Симуляция увеличения фьючерсного баланса
# logging.info("Simulating increase in futures balance...")
# simulate_changes(manager, steps=30, increment=1)



# import logging


# class BalanceManager:
#     def __init__(self, spot_balance, futures_balance):
#         self.spot_balance = spot_balance
#         self.futures_balance = futures_balance
#         self.initial_total_balance = spot_balance + futures_balance
#         self.BB = max(self.initial_total_balance / 10, 1)  # Минимум 1
#         self.target_spot_balance = 4 * self.BB
#         self.target_futures_balance = self.initial_total_balance - self.target_spot_balance

#     def redistribute_balance(self, initial=False):
#         total_balance = self.spot_balance + self.futures_balance

#         if initial:
#             # Инициализация начального перераспределения
#             self.BB = max(total_balance / 10, 1)
#             self.target_spot_balance = 4 * self.BB
#             self.target_futures_balance = total_balance - self.target_spot_balance
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = total_balance - self.spot_balance
#             logging.info(
#                 f"Initial redistribution: Spot balance: {self.spot_balance:.5f}, "
#                 f"Futures balance: {self.futures_balance:.5f}, BB: {self.BB:.5f}, Total balance: {total_balance:.5f}"
#             )
#             return

#         # Условие перераспределения при достижении порога
#         if self.futures_balance >= 94.75:  # Заданное условие для перераспределения
#             # Увеличиваем спот до 175% от текущего
#             new_spot_balance = round(self.spot_balance * 1.75, 5)

#             # Проверяем, чтобы новый спот не превышал общий баланс
#             if new_spot_balance > total_balance:
#                 new_spot_balance = round(total_balance * 0.75, 5)  # Ограничиваем до 75% от общего баланса

#             # Перераспределяем
#             self.spot_balance = new_spot_balance
#             self.futures_balance = round(total_balance - self.spot_balance, 5)

#             logging.info(
#                 f"Triggered redistribution: Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )
#         else:
#             logging.info(
#                 f"No redistribution needed. Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )


# def simulate_changes(manager, steps=10, increment=1):
#     logging.info("Starting simulation...")
#     for step in range(steps):
#         logging.info(f"Step {step + 1}:")
#         # Увеличиваем или уменьшаем фьючерсный баланс
#         manager.futures_balance += increment
#         manager.redistribute_balance()
#         logging.info(
#             f"After change: Spot balance: {manager.spot_balance:.5f}, "
#             f"Futures balance: {manager.futures_balance:.5f}, Total balance: {manager.spot_balance + manager.futures_balance:.5f}"
#         )


# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Инициализация начальных значений
# spot_balance = 28.0
# futures_balance = 42.0
# manager = BalanceManager(spot_balance, futures_balance)

# # Симуляция увеличения фьючерсного баланса
# logging.info("Simulating increase in futures balance...")
# simulate_changes(manager, steps=100, increment=3)







# import logging

# class BalanceManager:
#     def __init__(self, spot_balance, futures_balance):
#         self.spot_balance = spot_balance
#         self.futures_balance = futures_balance
#         self.initial_total_balance = spot_balance + futures_balance
#         self.BB = max(self.initial_total_balance / 10, 1)  # Минимум 1
#         self.update_targets()

#     def update_targets(self):
#         """
#         Обновляет целевые балансы на основе текущего общего баланса.
#         """
#         total_balance = self.spot_balance + self.futures_balance
#         self.BB = max(total_balance / 10, 1)
#         self.target_spot_balance = 4 * self.BB
#         self.target_futures_balance = total_balance - self.target_spot_balance

#     def redistribute_balance(self, initial=False):
#         """
#         Перераспределяет баланс между спотом и фьючерсами.
#         """
#         total_balance = self.spot_balance + self.futures_balance

#         if initial:
#             # Инициализация начального перераспределения
#             self.update_targets()
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = self.target_futures_balance
#             logging.info(
#                 f"Initial redistribution: Spot balance: {self.spot_balance:.5f}, "
#                 f"Futures balance: {self.futures_balance:.5f}, BB: {self.BB:.5f}, Total balance: {total_balance:.5f}"
#             )
#             return

#         # Проверка на необходимость перераспределения
#         if (
#             self.futures_balance > self.target_futures_balance * 1.1  # 10% превышение от целевого
#             or self.spot_balance < self.target_spot_balance * 0.9  # 10% недостача от целевого
#         ):
#             self.update_targets()
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = total_balance - self.spot_balance
#             logging.info(
#                 f"Triggered redistribution: Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )
#         else:
#             logging.info(
#                 f"No redistribution needed. Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )

# def simulate_changes(manager, steps=10, increment=1):
#     logging.info("Starting simulation...")
#     for step in range(steps):
#         logging.info(f"Step {step + 1}:")
#         # Увеличиваем или уменьшаем фьючерсный баланс
#         manager.futures_balance += increment
#         manager.redistribute_balance()
#         logging.info(
#             f"After change: Spot balance: {manager.spot_balance:.5f}, "
#             f"Futures balance: {manager.futures_balance:.5f}, Total balance: {manager.spot_balance + manager.futures_balance:.5f}"
#         )

# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Инициализация начальных значений
# spot_balance = 28.0
# futures_balance = 42.0
# manager = BalanceManager(spot_balance, futures_balance)

# # Симуляция увеличения фьючерсного баланса
# logging.info("Simulating increase in futures balance...")
# simulate_changes(manager, steps=100, increment=3)


# import logging
# from decimal import Decimal

# class BalanceManager:
#     def __init__(self, spot_balance, futures_balance, growth_multiplier=1.75):
#         self.spot_balance = Decimal(spot_balance)
#         self.futures_balance = Decimal(futures_balance)
#         self.growth_multiplier = Decimal(growth_multiplier)
#         self.initial_total_balance = self.spot_balance + self.futures_balance
#         self.BB = max(self.initial_total_balance / 10, 1)  # Минимум 1
#         self.update_targets()

#     def update_targets(self):
#         """
#         Обновляет целевые балансы на основе текущего общего баланса.
#         """
#         total_balance = self.spot_balance + self.futures_balance
#         self.BB = max(total_balance / 10, 1)
#         self.target_spot_balance = self.spot_balance * self.growth_multiplier
#         self.target_futures_balance = self.futures_balance * self.growth_multiplier

#     def redistribute_balance(self, initial=False):
#         """
#         Перераспределяет баланс между спотом и фьючерсами.
#         """
#         total_balance = self.spot_balance + self.futures_balance

#         if initial:
#             # Инициализация начального перераспределения
#             self.update_targets()
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = self.target_futures_balance
#             logging.info(
#                 f"Initial redistribution: Spot balance: {self.spot_balance:.5f}, "
#                 f"Futures balance: {self.futures_balance:.5f}, BB: {self.BB:.5f}, Total balance: {total_balance:.5f}"
#             )
#             return

#         # Проверка на необходимость перераспределения
#         if (
#             self.futures_balance > self.target_futures_balance * 1.1  # 10% превышение от целевого
#             or self.spot_balance < self.target_spot_balance * 0.9  # 10% недостача от целевого
#         ):
#             self.update_targets()
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = self.target_futures_balance
#             logging.info(
#                 f"Triggered redistribution: Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )
#         else:
#             logging.info(
#                 f"No redistribution needed. Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )

# def simulate_changes(manager, steps=10, increment=3):
#     """
#     Симулирует изменения фьючерсного баланса и перераспределение.
#     """
#     logging.info("Starting simulation...")
#     for step in range(steps):
#         logging.info(f"Step {step + 1}:")
#         # Увеличиваем или уменьшаем фьючерсный баланс
#         manager.futures_balance += Decimal(increment)
#         manager.redistribute_balance()
#         logging.info(
#             f"After change: Spot balance: {manager.spot_balance:.5f}, "
#             f"Futures balance: {manager.futures_balance:.5f}, Total balance: {manager.spot_balance + manager.futures_balance:.5f}"
#         )

# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Инициализация начальных значений
# spot_balance = 28.0
# futures_balance = 42.0
# manager = BalanceManager(spot_balance, futures_balance)

# # Симуляция увеличения фьючерсного баланса
# logging.info("Simulating increase in futures balance...")
# simulate_changes(manager, steps=100, increment=3)








# import logging
# from decimal import Decimal

# class BalanceManager:
#     def __init__(self, spot_balance, futures_balance, growth_multiplier=1.75):
#         self.spot_balance = Decimal(spot_balance)
#         self.futures_balance = Decimal(futures_balance)
#         self.growth_multiplier = Decimal(growth_multiplier)
#         self.initial_total_balance = self.spot_balance + self.futures_balance
#         self.BB = max(self.initial_total_balance / Decimal(10), Decimal(1))  # Минимум 1
#         self.update_targets()

#     def update_targets(self):
#         """
#         Обновляет целевые балансы на основе текущего общего баланса.
#         """
#         total_balance = self.spot_balance + self.futures_balance
#         self.BB = max(total_balance / Decimal(10), Decimal(1))
#         self.target_spot_balance = self.spot_balance * self.growth_multiplier
#         self.target_futures_balance = self.futures_balance * self.growth_multiplier

#     def redistribute_balance(self, initial=False):
#         """
#         Перераспределяет баланс между спотом и фьючерсами.
#         """
#         total_balance = self.spot_balance + self.futures_balance

#         if initial:
#             # Инициализация начального перераспределения
#             self.update_targets()
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = self.target_futures_balance
#             logging.info(
#                 f"Initial redistribution: Spot balance: {self.spot_balance:.5f}, "
#                 f"Futures balance: {self.futures_balance:.5f}, BB: {self.BB:.5f}, Total balance: {total_balance:.5f}"
#             )
#             return

#         # Проверка на необходимость перераспределения
#         if (
#             self.futures_balance > self.target_futures_balance * Decimal(1.1)  # 10% превышение от целевого
#             or self.spot_balance < self.target_spot_balance * Decimal(0.9)  # 10% недостача от целевого
#         ):
#             self.update_targets()
#             self.spot_balance = self.target_spot_balance
#             self.futures_balance = self.target_futures_balance
#             logging.info(
#                 f"Triggered redistribution: Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )
#         else:
#             logging.info(
#                 f"No redistribution needed. Spot: {self.spot_balance:.5f}, "
#                 f"Futures: {self.futures_balance:.5f}, Total balance: {total_balance:.5f}"
#             )

# def simulate_changes(manager, steps=10, increment=3):
#     """
#     Симулирует изменения фьючерсного баланса и перераспределение.
#     """
#     logging.info("Starting simulation...")
#     for step in range(steps):
#         logging.info(f"Step {step + 1}:")
#         # Увеличиваем или уменьшаем фьючерсный баланс
#         manager.futures_balance += Decimal(increment)
#         manager.redistribute_balance()
#         logging.info(
#             f"After change: Spot balance: {manager.spot_balance:.5f}, "
#             f"Futures balance: {manager.futures_balance:.5f}, Total balance: {manager.spot_balance + manager.futures_balance:.5f}"
#         )

# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Инициализация начальных значений
# spot_balance = 28.0
# futures_balance = 42.0
# manager = BalanceManager(spot_balance, futures_balance)

# # Симуляция увеличения фьючерсного баланса
# logging.info("Simulating increase in futures balance...")
# simulate_changes(manager, steps=100, increment=3)




import logging

class BalanceManager:
    def __init__(self, spot_balance, futures_balance, growth_factor=1.75):
        self.spot_balance = spot_balance
        self.futures_balance = futures_balance
        self.growth_factor = growth_factor
        self.current_target = self.spot_balance + self.futures_balance  # Initial total balance

    def redistribute_balance(self):
        """
        Redistributes the balance when the total balance exceeds the target.
        """
        total_balance = self.spot_balance + self.futures_balance

        if total_balance >= self.current_target * self.growth_factor:
            # Update the target total balance
            self.current_target = total_balance
            # Increase the spot balance by growth factor
            new_spot_balance = self.spot_balance * self.growth_factor
            # Calculate the new futures balance to maintain the total balance
            new_futures_balance = self.current_target - new_spot_balance

            # Update balances
            self.spot_balance = new_spot_balance
            self.futures_balance = new_futures_balance

            logging.info(
                f"Redistribution: Spot: {self.spot_balance:.2f}, "
                f"Futures: {self.futures_balance:.2f}, Total: {self.current_target:.2f}"
            )
        else:
            logging.info(
                f"No redistribution. Spot: {self.spot_balance:.2f}, "
                f"Futures: {self.futures_balance:.2f}, Total: {total_balance:.2f}"
            )

    def increment_futures(self, increment):
        """
        Increases the futures balance by a given increment.
        """
        self.futures_balance += increment

def simulate_changes(manager, steps, increment):
    """
    Simulates balance changes over a number of steps.
    """
    logging.info("Starting simulation...")
    for step in range(1, steps + 1):
        logging.info(f"Step {step}:")
        manager.increment_futures(increment)
        manager.redistribute_balance()
        logging.info(
            f"After change: Spot: {manager.spot_balance:.2f}, "
            f"Futures: {manager.futures_balance:.2f}, "
            f"Total: {manager.spot_balance + manager.futures_balance:.2f}"
        )


# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize manager with initial values
initial_spot = 28.0
initial_futures = 42.0
manager = BalanceManager(initial_spot, initial_futures)

# Run simulation
simulate_changes(manager, steps=100, increment=13)









def denormalize(value):
    """
    Денормализует значение на основе диапазона.
    """
    min_value = 0
    max_value = 115000
    return value * (max_value - min_value) + min_value

print(denormalize(0.808364))
print(denormalize(0.828695))

	
print(denormalize(0.79574))
print(denormalize(0.823198))






