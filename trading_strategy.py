import pandas as pd

def generate_signal(df, short_window, long_window):
    df['short_ma'] = df['Close'].rolling(window=short_window).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window).mean()
    last_row = df.iloc[-1]
    previous_row = df.iloc[-2]
    if previous_row['short_ma'] <= previous_row['long_ma'] and last_row['short_ma'] > last_row['long_ma']:
        return 'BUY'
    if previous_row['short_ma'] >= previous_row['long_ma'] and last_row['short_ma'] < last_row['long_ma']:
        return 'SELL'
    return 'HOLD'
