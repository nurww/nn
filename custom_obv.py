# custom_obv.py

def calculate_custom_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close_price'][i] > df['close_price'][i-1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close_price'][i] < df['close_price'][i-1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    df['custom_OBV'] = obv
    return df
