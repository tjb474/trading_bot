# ml/feature_engineering.py
import pandas as pd

def create_features(df, feature_list=None, **kwargs):
    """
    Create features for the ML model from a historical price DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with lowercase 'open', 'high', 'low', 'close' columns.
        feature_list (list): List of features to create (optional, defaults to basic features)
        **kwargs: Additional parameters for feature creation (volatility_window, rsi_window, etc.)

    Returns:
        pd.DataFrame: DataFrame with added feature columns.
    """
    # Make a copy to avoid modifying the original DataFrame
    features_df = df.copy()
    
    # Default feature list if none provided
    if feature_list is None:
        feature_list = ['returns', 'volatility', 'rsi']
    
    # Extract parameters with defaults
    volatility_window = kwargs.get('volatility_window', 20)
    rsi_window = kwargs.get('rsi_window', 14)
    atr_window = kwargs.get('atr_window', 14)
    
    # Add basic features that are commonly used
    if 'returns' in feature_list:
        features_df['returns'] = features_df['close'].pct_change()
    
    if 'volatility' in feature_list:
        features_df['volatility'] = features_df['returns'].rolling(window=volatility_window).std()
    
    if 'rsi' in feature_list:
        # RSI calculation
        delta = features_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
    
    return features_df.dropna()