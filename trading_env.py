# trading_env.py

import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=1000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.position = 0  # 0 - ничего не делаем, 1 - покупаем, -1 - продаем
        self.current_step = 0

        # Определяем пространство действий и наблюдений
        self.action_space = spaces.Discrete(3)  # 0 - ничего, 1 - покупка, 2 - продажа
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        current_price = self.data[self.current_step][3]  # Цена закрытия

        if action == 1:  # Покупка
            if self.position == 0:
                self.position = 1
                self.buy_price = current_price
        elif action == 2:  # Продажа
            if self.position == 1:
                profit = current_price - self.buy_price
                self.balance += profit
                self.net_worth += profit
                self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Награда
        reward = self.balance - self.initial_balance

        obs = self.data[self.current_step]
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Position: {self.position}')
