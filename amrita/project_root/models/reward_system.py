# reward_system.py

def calculate_reward(prediction, actual_price, interval_signal=None):
    """
    Функция для расчета вознаграждения.
    - Положительное вознаграждение за точное предсказание (например, если направление совпадает с интервалом).
    - Отрицательное вознаграждение за ошибочное предсказание.
    
    :param prediction: Предсказанное значение
    :param actual_price: Фактическое значение
    :param interval_signal: Сигнал от модели интервалов (например, -1 или +1)
    :return: reward (числовое значение)
    """
    error = abs(prediction - actual_price)
    
    # Основное вознаграждение на основе точности
    reward = max(1 - error, -1)  # Чем меньше ошибка, тем выше награда

    # Дополнительное вознаграждение за совпадение с интервалом
    if interval_signal is not None:
        if (interval_signal > 0 and prediction > 0) or (interval_signal < 0 and prediction < 0):
            reward += 0.5  # Дополнительное вознаграждение за согласие с интервалом

    return reward

def test_reward_system():
    # Тестовые значения
    prediction = 1.05
    actual_price = 1.0
    interval_signal = 1  # восходящий тренд

    # Расчет вознаграждения
    reward = calculate_reward(prediction, actual_price, interval_signal)
    print("Reward:", reward)

if __name__ == "__main__":
    test_reward_system()