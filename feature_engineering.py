import pandas as pd

def create_features(df):
    """
    Create features for the ML model.
    For this example, we'll just use price change and volatility.
    """
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    # Add macroeconomic features here
    # e.g., df['VIX'] = ...
    return df.dropna()
