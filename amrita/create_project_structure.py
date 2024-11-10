# create_project_structure.py

import os

# Определяем структуру проекта
project_structure = {
    "project_root": [
        "main.py",
        "README.md",
        "requirements.txt",
    ],
    "config": [
        "config.yaml",
        "logging.conf",
    ],
    "data": [
        "preprocess.py",
        "fetch_intervals.py",
        "fetch_orderbook.py",
        "database_manager.py",
        "sql_queries/interval_queries.sql",
    ],
    "models": [
        "__init__.py",
        "interval_model.py",
        "orderbook_model.py",
        "correlation_clustering.py",
        "genetic_algorithm.py",
        "reinforcement_learning.py",
    ],
    "optimizers": [
        "hyperparameter_optimization.py",
        "strategy_optimizer.py",
        "rl_agent_trainer.py",
    ],
    "trading": [
        "real_trade.py",
        "trial_trade.py",
        "interface.py",
        "trade_logger.py",
        "commands.py",
    ],
    "utils": [
        "logger.py",
        "data_splitter.py",
        "indicators.py",
        "model_utils.py",
        "orderbook_utils.py",
    ],
    "logs": [
        "training.log",
        "trading.log",
        "optimization.log",
    ],
}

# Функция для создания структуры файлов и директорий
def create_project_structure(base_dir, structure):
    for folder, files in structure.items():
        # Создаем путь для каждой папки
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        for file in files:
            file_path = os.path.join(base_dir, file) if "/" in file else os.path.join(folder_path, file)
            
            # Создаем директорию для подкаталогов
            if "/" in file:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Создаем пустой файл
            with open(file_path, "w") as f:
                pass

# Запуск создания структуры
base_directory = "project_root"  # Базовая директория для проекта
create_project_structure(base_directory, project_structure)

print(f"Структура проекта создана в директории: {base_directory}")
