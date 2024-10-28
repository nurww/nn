# test_agents_with_rewards.py

from multi_agent_trading import agents, run_agent
from data_loader import load_data_from_db

# Убедимся, что соединение с базой данных и загрузка данных работает
connection = connect_to_db()

# Загрузка данных из базы данных
table_name = 'binance_klines_normalized'
interval = '1m'
X, y = load_data_from_db(connection, table_name, interval)

# Поскольку мы имитируем данные для торговых агентов, сгенерируем случайный рыночный набор
# Для демонстрации используем часть данных как 'market_data'
market_data = X

# Запускаем каждого агента с рыночными данными
for agent in agents:
    run_agent(agent, market_data)

# Закрытие соединения с базой данных
connection.close()
