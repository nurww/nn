# strategy_monitor.py

import os
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Функция для загрузки данных о сделках из базы данных
def load_trading_data(connection):
    query = "SELECT timestamp, action, amount, price FROM trading_actions"
    df = pd.read_sql(query, connection)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Функция для подсчета прибыли/убытка по сделкам
def calculate_pnl(df):
    pnl = 0.0
    position = 0.0  # Текущий объем позиции (положительное значение для длинных позиций, отрицательное для коротких)
    cash = 0.0  # Денежный баланс
    
    for _, row in df.iterrows():
        if row['action'] == 'buy':
            position += row['amount']
            cash -= row['amount'] * row['price']
        elif row['action'] == 'sell':
            position -= row['amount']
            cash += row['amount'] * row['price']
        
        # Рассчитываем текущий баланс как сумма кэша и стоимости позиции по текущей цене
        current_balance = cash + position * row['price']
        pnl = current_balance

    return pnl, current_balance

# Функция для визуализации результатов торговли
def plot_results(df, pnl, save_path):
    # Построим график цены и меток покупок/продаж
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['price'], label='Price', alpha=0.5)

    # Отображаем сделки на графике
    buys = df[df['action'] == 'buy']
    sells = df[df['action'] == 'sell']
    plt.scatter(buys['timestamp'], buys['price'], color='green', label='Buy', marker='^', alpha=0.6)
    plt.scatter(sells['timestamp'], sells['price'], color='red', label='Sell', marker='v', alpha=0.6)

    # Настройки графика
    plt.title(f'Trading Results - PnL: {pnl:.2f}')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig(save_path)
    plt.close()
    print(f"График сохранен в {save_path}")

def main():
    connection = mysql.connector.connect(
        host='localhost',
        database='binance_data',
        user='root',
        password='root'
    )

    try:
        # Создаем директорию для сохранения графиков, если она не существует
        os.makedirs('trading_charts', exist_ok=True)

        while True:
            # Загружаем данные о сделках
            df = load_trading_data(connection)

            if df.empty:
                print("Нет данных о сделках для анализа.")
            else:
                # Подсчитываем прибыль/убыток
                pnl, current_balance = calculate_pnl(df)
                print(f"Общая прибыль/убыток: {pnl:.2f}")
                print(f"Текущий баланс: {current_balance:.2f}")

                # Определяем путь для сохранения графика
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join('trading_charts', f"trading_results_{timestamp}.png")
                plot_results(df, pnl, save_path)

            # Сохраняем данные о сделках в CSV для дальнейшего анализа
            df.to_csv('trading_results.csv', index=False)
            print("Данные о сделках сохранены в trading_results.csv")

            # Пауза на час перед следующим построением графика
            time.sleep(3600)

    except KeyboardInterrupt:
        print("Мониторинг стратегии остановлен.")

    finally:
        if connection.is_connected():
            connection.close()

if __name__ == "__main__":
    main()
