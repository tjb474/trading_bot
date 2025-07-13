# ml/feature_engineering.py
import pandas as pd
from data.data_manager import load_dbn_to_df

def create_features(df):
    """
    Create features for the ML model from a historical price DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with lowercase 'open', 'high', 'low', 'close' columns.

    Returns:
        pd.DataFrame: DataFrame with added feature columns.
    """
    # Make a copy to avoid modifying the original DataFrame
    features_df = df.copy()
    
    # --- FIX ---
    # Use lowercase 'close' consistently
    features_df['returns'] = features_df['close'].pct_change()
    
    # Rolling volatility
    features_df['volatility'] = features_df['returns'].rolling(window=20).std()
    
    # Example of another feature: RSI
    delta = features_df['close'].diff()
    # --- END FIX ---
    
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features_df['rsi'] = 100 - (100 / (1 + rs))
    
    return features_df.dropna()