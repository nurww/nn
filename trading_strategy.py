# trading_strategy.py

import time
import mysql.connector
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(
    filename='trading_strategy.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Функция для получения последнего прогноза из базы данных
def get_latest_prediction(connection):
    cursor = connection.cursor()
    query = "SELECT trend_prediction, confidence FROM trading_data ORDER BY id DESC LIMIT 1"
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    if result:
        return result[0], result[1]  # trend_prediction, confidence
    return None, None

# Функция для получения последних данных из ордербука
def get_latest_order_book_data(connection):
    cursor = connection.cursor()
    query = """
        SELECT mid_price, sum_bid_volume, sum_ask_volume 
        FROM order_book_data 
        ORDER BY timestamp DESC 
        LIMIT 1
    """
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    if result:
        return result[0], result[1], result[2]  # mid_price, sum_bid_volume, sum_ask_volume
    return None, None, None

# Пример логирования в стратегию
def execute_buy(connection, amount, price):
    logging.info(f"Выполняем покупку на {amount} по цене {price}")
    save_trading_action(connection, "buy", amount, price)

def execute_sell(connection, amount, price):
    logging.info(f"Выполняем продажу на {amount} по цене {price}")
    save_trading_action(connection, "sell", amount, price)

def hold_position():
    logging.info("Удерживаем текущую позицию")

def adapt_strategy_params(connection):
    query = "SELECT AVG(profit) FROM trading_actions"
    cursor = connection.cursor()
    cursor.execute(query)
    avg_profit = cursor.fetchone()[0]
    cursor.close()

    # Определение нового порога
    new_confidence_threshold = 0.65 if avg_profit is not None and avg_profit < 0 else 0.6

    # Обновляем значение в таблице strategy_params
    update_query = """
        INSERT INTO strategy_params (param_name, param_value)
        VALUES ('confidence_threshold', %s)
        ON DUPLICATE KEY UPDATE param_value = VALUES(param_value)
    """
    cursor = connection.cursor()
    cursor.execute(update_query, (new_confidence_threshold,))
    connection.commit()
    cursor.close()

    logging.info(f"Установлен новый порог уверенности: {new_confidence_threshold}")
    return new_confidence_threshold

# Функция для сохранения торгового действия в базу данных
def save_trading_action(connection, action, amount, price):
    cursor = connection.cursor()
    query = """
        INSERT INTO trading_actions (timestamp, action, amount, price)
        VALUES (%s, %s, %s, %s)
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(query, (timestamp, action, amount, price))
    connection.commit()
    cursor.close()
    print(f"Сохранено действие {action} на {amount} по цене {price} в базе данных")

def trading_decision(connection):
    # Получаем текущий порог уверенности
    confidence_threshold = adapt_strategy_params(connection)
    saved_confidence_threshold = get_confidence_threshold(connection)
    logging.info(f"Текущий порог уверенности: {confidence_threshold}, Сохраненный порог: {saved_confidence_threshold}")

    # Получаем последний прогноз
    trend_prediction, confidence = get_latest_prediction(connection)
    
    if trend_prediction is None:
        print("Нет доступных прогнозов для принятия решения.")
        return

    # Получаем текущую цену из ордербука
    cursor = connection.cursor()
    cursor.execute("SELECT mid_price FROM order_book_data ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    cursor.close()

    if not result:
        print("Не удалось получить данные из ордербука.")
        return

    current_price = result[0]
    amount = 0.01  # Пример объема сделки

    # Используем динамический порог уверенности для принятия решения
    if trend_prediction == 1 and confidence > confidence_threshold:
        execute_buy(connection, amount, current_price)
    elif trend_prediction == -1 and confidence > confidence_threshold:
        execute_sell(connection, amount, current_price)
    else:
        hold_position()

# Функция для загрузки порога уверенности из базы данных
def get_confidence_threshold(connection):
    cursor = connection.cursor()
    query = "SELECT param_value FROM strategy_params WHERE param_name = 'confidence_threshold' LIMIT 1"
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else 0.6  # Используем значение по умолчанию, если данных нет

# Запуск стратегии
def main():
    connection = mysql.connector.connect(
        host='localhost',
        database='binance_data',
        user='root',
        password='root'
    )

    try:
        while True:
            trading_decision(connection)
            time.sleep(10)  # Интервал времени для принятия решения (10 секунд)

    except KeyboardInterrupt:
        print("Торговая стратегия остановлена.")

    finally:
        if connection.is_connected():
            connection.close()

if __name__ == "__main__":
    main()
