import mysql.connector

# Подключение к базе данных MySQL
def connect_to_db():
    connection = mysql.connector.connect(
        host='localhost',
        database='binance_data',
        user='root',   
        password='root'
    )
    return connection

# Функция для поиска дубликатов
def check_for_duplicates(connection):
    cursor = connection.cursor()

    query = """
    SELECT open_time, data_interval, COUNT(*)
    FROM binance_klines
    GROUP BY open_time, data_interval
    HAVING COUNT(*) > 1;
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()

    if rows:
        print("Дубликаты найдены:")
        for row in rows:
            print(f"open_time: {row[0]}, interval: {row[1]}, количество: {row[2]}")
    else:
        print("Дубликатов не найдено.")

# Подключаемся к базе данных
connection = connect_to_db()

# Проверяем на дубликаты
try:
    check_for_duplicates(connection)
finally:
    # Закрываем соединение с MySQL
    connection.close()
