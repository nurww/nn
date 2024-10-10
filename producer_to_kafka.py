from kafka import KafkaProducer
import mysql.connector
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Настройка логирования
logging.basicConfig(filename='logs/producer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def decode_unicode_string(s):
    return s.encode().decode('unicode_escape')

# Подключение к MySQL
connection = mysql.connector.connect(
    host='localhost',
    database='binance_data',
    user='root',
    password='root'
)

# Инициализация Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),  # default=str для сериализации datetime
    acks='all',  # Подтверждение отправки
    linger_ms=10,  # Задержка для агрегации записей
    batch_size=32 * 1024  # Настройка размера партии
)

# Проверка соединения с MySQL
if connection.is_connected():
    logging.info(decode_unicode_string("Соединение с MySQL установлено"))
else:
    logging.error(decode_unicode_string("Не удалось установить соединение с MySQL"))
    sys.exit(1)

# Функция для получения новых данных
def fetch_new_data(data_interval):
    try:
        # Инициализируем соединение внутри функции для каждого потока
        connection = mysql.connector.connect(
            host='localhost',
            database='binance_data',
            user='root',
            password='root'
        )
        if connection.is_connected():
            logging.info(decode_unicode_string(f"Соединение с MySQL установлено для интервала {data_interval}"))
        else:
            logging.error(decode_unicode_string(f"Не удалось установить соединение с MySQL для интервала {data_interval}"))
            return []

        cursor = connection.cursor(dictionary=True)
        query = f"SELECT * FROM binance_klines WHERE data_interval = '{data_interval}' ORDER BY open_time"
        cursor.execute(query)
        data = cursor.fetchall()

        # Преобразуем все datetime поля в строки
        for record in data:
            for key, value in record.items():
                if isinstance(value, datetime):
                    record[key] = value.strftime('%Y-%m-%d %H:%M:%S')

        logging.info(decode_unicode_string(f"Найдено {len(data)} записей для интервала {data_interval}"))
        return data
    except mysql.connector.Error as e:
        logging.error(decode_unicode_string(f"Ошибка MySQL: {e}"))
        return []
    finally:
        if connection.is_connected():
            connection.close()
            logging.info(decode_unicode_string(f"Соединение с MySQL закрыто для интервала {data_interval}"))

# Асинхронная отправка данных в Kafka
def send_to_kafka_async(data_interval, batch_size=500):
    try:
        new_data = fetch_new_data(data_interval)
        if not new_data:
            logging.info(decode_unicode_string(f"Нет новых данных для интервала {data_interval}"))
            return
        
        batch = []
        for record in new_data:
            batch.append(record)
            if len(batch) >= batch_size:
                future = producer.send(f'BTCUSDT-{data_interval}', batch)
                future.add_callback(on_send_success)
                future.add_errback(on_send_error)
                logging.info(decode_unicode_string(f"Отправлено {len(batch)} сообщений в топик {data_interval}"))
                batch = []

        # Отправляем оставшиеся данные, если есть
        if batch:
            future = producer.send(f'BTCUSDT-{data_interval}', batch)
            future.add_callback(on_send_success)
            future.add_errback(on_send_error)
            logging.info(decode_unicode_string(f"Отправлено {len(batch)} сообщений в топик {data_interval}"))

        producer.flush()

    except Exception as e:
        logging.error(decode_unicode_string(f"Ошибка при отправке данных в Kafka: {e}"))

# Функции обратного вызова для отслеживания статуса отправки
def on_send_success(record_metadata):
    logging.info(decode_unicode_string(f"Сообщение успешно отправлено в {record_metadata.topic} на раздел {record_metadata.partition} с оффсетом {record_metadata.offset}"))

def on_send_error(excp):
    logging.error(decode_unicode_string(f"Ошибка при отправке сообщения: {excp}"))

# Параллельная отправка данных по интервалам
def send_all_intervals():
    intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
    with ThreadPoolExecutor(max_workers=len(intervals)) as executor:
        executor.map(send_to_kafka_async, intervals)

# Закрытие соединений
def close_connections():
    if connection.is_connected():
        connection.close()
        logging.info("Соединение с MySQL закрыто")
    
    producer.close()
    logging.info("Соединение с Kafka закрыто")

# Основная логика
try:
    send_all_intervals()
finally:
    close_connections()
