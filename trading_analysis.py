import mysql.connector
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Функция для загрузки данных о торговых действиях
def load_trading_data(connection):
    query = "SELECT * FROM trading_actions"
    df = pd.read_sql(query, connection)
    return df

# Функция для анализа прибыльности
def analyze_profitability(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    
    # Добавляем колонку 'profit' для расчета прибыли по каждой сделке
    df['profit'] = df['amount'] * df['price'] * df['action'].apply(lambda x: 1 if x == 'sell' else -1)
    df['cumulative_profit'] = df['profit'].cumsum()

    print(f"Общая прибыль: {df['cumulative_profit'].iloc[-1]}")
    print(f"Количество сделок: {len(df)}")
    print(f"Средняя прибыль на сделку: {df['profit'].mean()}")

    return df

# Функция для сохранения графика
def plot_and_save(df, folder_name='results'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['cumulative_profit'], label='Cumulative Profit')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Profit')
    plt.title('Trading Cumulative Profit Over Time')
    plt.legend()
    plt.grid(True)

    # Сохраняем график в файл с текущей датой и временем
    file_path = os.path.join(folder_name, f"profit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"График сохранен в {file_path}")

def adapt_strategy_params(connection):
    # Пример изменения порога уверенности в зависимости от средней прибыли
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

    print(f"Установлен новый порог уверенности: {new_confidence_threshold}")
    return new_confidence_threshold

def main():
    connection = mysql.connector.connect(
        host='localhost',
        database='binance_data',
        user='root',
        password='root'
    )

    try:
        # Загружаем данные и анализируем их
        df = load_trading_data(connection)
        df = analyze_profitability(df)

        # Сохраняем график
        plot_and_save(df)

    except Exception as e:
        print(f"Ошибка: {e}")

    finally:
        if connection.is_connected():
            connection.close()

if __name__ == "__main__":
    main()
