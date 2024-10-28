# rl_trading.py

import gym
import torch
import pandas as pd
import mysql.connector
from mysql.connector import Error
from trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def load_data_from_db(connection, table_name, interval):
    query = f"""
        SELECT open_price_normalized, high_price_normalized, low_price_normalized, close_price_normalized,
               volume_normalized, rsi_normalized, macd_normalized, macd_signal_normalized, macd_hist_normalized,
               sma_20_normalized, ema_20_normalized, upper_bb_normalized, middle_bb_normalized, lower_bb_normalized, obv_normalized
        FROM {table_name}
        WHERE data_interval = '{interval}'
        ORDER BY open_time ASC
    """
    df = pd.read_sql(query, connection)
    # Можно добавить дополнительные индикаторы или обработку данных
    return df.values

def main():
    try:
        # Подключение к базе данных
        connection = mysql.connector.connect(
            host='localhost',
            database='binance_data',
            user='root',
            password='root'
        )

        # Загрузка данных
        data = load_data_from_db(connection, 'binance_klines_normalized', '1m')

        # Создание среды
        env = TradingEnv(data)
        env = DummyVecEnv([lambda: env])

        # Инициализация модели PPO
        model = PPO('MlpPolicy', env, verbose=1)

        # Обучение модели
        model.learn(total_timesteps=10000)

        # Сохранение модели
        model.save("ppo_trading_model")
        print("Модель сохранена")

    except Error as e:
        print(f"Ошибка подключения к MySQL: {e}")

    finally:
        if connection.is_connected():
            connection.close()
            print("Соединение с MySQL закрыто")

if __name__ == '__main__':
    main()
