# agents.py

class Agent:
    def __init__(self, name):
        self.name = name
        self.total_reward = 0

    def make_decision(self, data_point):
        # Простая логика для примера: покупка/продажа на основе случайных данных
        return np.random.choice(['buy', 'sell'])

    def update_reward(self, reward):
        self.total_reward += reward
        print(f"Agent {self.name} received reward: {reward}. Total reward: {self.total_reward}")
