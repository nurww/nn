import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Инициализация нормализатора
scaler = MinMaxScaler()

def normalize_data(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Нормализует указанные столбцы данных."""
    data[columns] = scaler.fit_transform(data[columns])
    return data

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропущенные значения предыдущими значениями."""
    return data.fillna(method='ffill').fillna(method='bfill')

def aggregate_data(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Агрегирует данные по указанному интервалу."""
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp').resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return data
