# reward_system.py

class RewardSystem:
    def __init__(self, risk_tolerance=0.02, reward_factor=1.0, penalty_factor=1.0):
        self.risk_tolerance = risk_tolerance
        self.reward_factor = reward_factor
        self.penalty_factor = penalty_factor

    def calculate_reward(self, agent, trade_result):
        """
        Вычисление вознаграждения или штрафа на основе результатов торговли агента.
        :param agent: Объект агента
        :param trade_result: Результаты торговли (прибыль или убыток)
        :return: Числовое значение вознаграждения или штрафа
        """
        profit = trade_result['profit']
        risk = trade_result['risk']
        duration = trade_result['duration']

        # Рассчитываем штраф за риск
        risk_penalty = max(0, (risk - self.risk_tolerance) * self.penalty_factor)

        # Рассчитываем поощрение за прибыль
        reward = max(0, profit * self.reward_factor)

        # Дополнительные штрафы или поощрения на основе длительности удержания позиции
        if duration > agent.optimal_hold_time:
            duration_penalty = (duration - agent.optimal_hold_time) * 0.1
            reward -= duration_penalty

        return reward - risk_penalty

# Пример использования системы вознаграждений для агента
class Agent:
    def __init__(self, name, optimal_hold_time=10):
        self.name = name
        self.optimal_hold_time = optimal_hold_time
        self.total_reward = 0

    def update_reward(self, reward):
        self.total_reward += reward
        print(f"Обновленное вознаграждение для {self.name}: {self.total_reward}")

# Пример торгового результата
trade_result = {
    'profit': 100,  # Прибыль в $
    'risk': 0.015,  # Риск, взятый агентом
    'duration': 12  # Время удержания позиции
}

# Инициализация агента и системы вознаграждений
agent = Agent(name="TrendAgent", optimal_hold_time=10)
reward_system = RewardSystem(risk_tolerance=0.02, reward_factor=1.5, penalty_factor=1.0)

# Вычисление вознаграждения
reward = reward_system.calculate_reward(agent, trade_result)
agent.update_reward(reward)
