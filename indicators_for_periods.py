from kafka import KafkaConsumer
import dask.dataframe as dd
import talib as ta
import json
import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(filename='logs/consumer.log', level=logging.INFO)

# Инициализация Kafka Consumer для каждого интервала
consumer = KafkaConsumer(
    'BTCUSDT-1m',  # Можно заменить на другие интервал-топики
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Функция обработки данных с использованием Dask
def process_data(records):
    df = pd.DataFrame(records)
    ddf = dd.from_pandas(df, npartitions=4)  # Преобразуем в Dask DataFrame для параллельной обработки

    # Рассчитываем индикаторы параллельно
    ddf['RSI'] = ddf.map_partitions(lambda part: ta.RSI(part['close_price'], timeperiod=14))
    ddf['MACD'], ddf['MACD_signal'], ddf['MACD_hist'] = ddf.map_partitions(
        lambda part: ta.MACD(part['close_price'], fastperiod=12, slowperiod=26, signalperiod=9)
    )
    ddf['SMA_20'] = ddf.map_partitions(lambda part: ta.SMA(part['close_price'], timeperiod=20))

    # Собираем результат обратно в Pandas DataFrame
    result = ddf.compute()
    print(result)

# Забираем сообщения из Kafka и обрабатываем
for message in consumer:
    records = [message.value]
    logging.info(f"Получено сообщение: {records}")
    process_data(records)
