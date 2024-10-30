import logging
import requests
import time
from datetime import datetime, timedelta, timezone

# Настройка логирования в файл с кодировкой UTF-8
logging.basicConfig(
    filename='binance_data_test.log',
    level=logging.INFO,
    format='%(message)s',  # Убираем временную метку логгера
    encoding='utf-8'
)

# Параметры запроса
url = 'https://fapi.binance.com/fapi/v1/klines'
symbol = 'BTCUSDT'  # Символ для запроса
interval = '1m'     # Интервал для запроса (1 минута)

def log_header():
    """Логирует заголовок с названиями колонок"""
    header = "open_time (UTC), open_time (UTC+5), open_price, high_price, low_price, close_price, volume"
    logging.info(header)
    print(header)

def format_data(data):
    """Форматирует данные свечи для логирования"""
    formatted_data = []
    for entry in data:
        open_time_utc = datetime.fromtimestamp(entry[0] / 1000, tz=timezone.utc)
        open_time_utc5 = open_time_utc + timedelta(hours=5)
        
        line = (
            f"{open_time_utc.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"{open_time_utc5.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"{entry[1]}, {entry[2]}, {entry[3]}, {entry[4]}, {entry[5]}"
        )
        formatted_data.append(line)
    
    return formatted_data

def fetch_data():
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 5  # Последние 5 записей
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Проверка на успешность запроса

        data = response.json()
        formatted_data = format_data(data)
        
        # Логируем данные с заголовком и отступом после завершения цикла
        log_header()
        for line in formatted_data:
            logging.info(line)
            print(line)
        logging.info("\n")  # Отступ для разделения циклов
        print("\n")
        
    except Exception as e:
        logging.error(f"Ошибка при получении данных: {e}")
        print(f"Ошибка при получении данных: {e}")

def wait_until_target_time():
    # Ждем до 57-й секунды текущей минуты
    target_time = datetime.utcnow().replace(second=57, microsecond=0)
    if datetime.utcnow().second >= 57:
        # Если уже прошло 57 секунд, ждем до следующей минуты
        target_time += timedelta(minutes=1)

    # Рассчитываем, сколько времени осталось до нужной секунды
    time_to_wait = (target_time - datetime.utcnow()).total_seconds()
    print(f"Ждем до 57-й секунды: {target_time} (ожидание {time_to_wait:.2f} секунд)")
    time.sleep(time_to_wait)

def main():
    while True:
        # Ждем до нужного времени (xx:xx:57)
        wait_until_target_time()

        # Начинаем запросы каждую секунду с xx:xx:57 до xx:xx+1:03
        end_time = datetime.utcnow().replace(second=3, microsecond=0) + timedelta(minutes=1)
        while datetime.utcnow() < end_time:
            fetch_data()
            time.sleep(1)  # Ждем 1 секунду перед следующим запросом
        
        # Пауза на 1 минуту перед началом следующего цикла
        time.sleep(60 - (datetime.utcnow().second % 60))  # Убедимся, что пауза равна ровно 1 минуте

if __name__ == "__main__":
    main()
