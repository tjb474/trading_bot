# ml/feature_engineering.py
"""
Feature Engineering System
========================

This module implements a flexible and extensible feature engineering system using a registry pattern.
Features are registered with their dependencies and parameters, then created in the correct order.

Example Usage
------------
1. Register a new feature:
    @registry.register('my_feature', dependencies=['other_feature'], window=20)
    def add_my_feature(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        # Feature calculation logic
        return df

2. Configure features in config.yaml:
    features:
      feature_list: ['returns', 'volatility', 'my_feature']
      volatility_window: 20
      my_feature_window: 30

3. Use in code:
    df = create_features(df, feature_list=['returns', 'my_feature'])

How Dependencies Work
-------------------
Features declare their dependencies using the dependencies parameter in @register.
The system automatically resolves the dependency graph and calculates features
in the correct order. For example, since 'volatility' depends on 'returns',
'returns' will always be calculated first, even if not explicitly requested.

Adding New Features
-----------------
1. Create a new function that takes a DataFrame and returns a DataFrame
2. Decorate it with @registry.register
3. Specify any dependencies and default parameters
4. Add the feature name to feature_list in config.yaml

The system will handle:
- Dependency resolution
- Parameter management
- Feature ordering
- Error checking
"""

from typing import Callable, Dict, List, Optional
import pandas as pd
import numpy as np
from functools import wraps

class FeatureRegistry:
    """
    Registry for feature engineering functions that handles dependencies and parameters.
    
    The registry manages:
    - Feature function registration via decorators
    - Feature dependencies to ensure correct calculation order
    - Default and override parameters for each feature
    - Automatic dependency resolution and execution
    """
    
    def __init__(self):
        self._features: Dict[str, Callable] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._parameters: Dict[str, Dict] = {}
    
    def register(self, name: str, dependencies: Optional[List[str]] = None, **params):
        """
        Decorator to register a feature engineering function.
        
        Args:
            name: Name of the feature (must match config.yaml feature_list names)
            dependencies: List of other features this one depends on
            **params: Default parameters for the feature function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            self._features[name] = wrapper
            self._dependencies[name] = dependencies or []
            self._parameters[name] = params
            return wrapper
        return decorator
    
    def add_features(self, df: pd.DataFrame, feature_list: List[str], **params) -> pd.DataFrame:
        """
        Add requested features to the dataframe in the correct order based on dependencies.
        
        Args:
            df: Input DataFrame
            feature_list: List of features to add (from config.yaml)
            **params: Override default parameters for any feature
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Resolve dependencies and create execution order
        to_process = set(feature_list)
        processed = set()
        execution_order = []
        
        while to_process:
            # Find a feature whose dependencies are all satisfied
            for feature in to_process:
                if all(dep in processed for dep in self._dependencies[feature]):
                    execution_order.append(feature)
                    processed.add(feature)
                    to_process.remove(feature)
                    break
            else:
                missing_deps = {feat: [dep for dep in self._dependencies[feat] 
                                    if dep not in processed]
                              for feat in to_process}
                raise ValueError(f"Circular or missing dependency detected: {missing_deps}")
        
        # Apply features in order
        for feature in execution_order:
            if feature not in self._features:
                raise ValueError(f"Feature '{feature}' not found in registry")
            
            # Merge default and override parameters
            feature_params = self._parameters[feature].copy()
            feature_params.update({k: v for k, v in params.items() 
                                 if k in feature_params})
            
            print(f"Adding feature: {feature}")
            result_df = self._features[feature](result_df, **feature_params)
        
        return result_df

def create_features(df: pd.DataFrame, feature_list: Optional[List[str]] = None, **params) -> pd.DataFrame:
    """
    Create all requested features from the feature list.
    
    This is the main entry point for feature engineering. It takes a DataFrame and list
    of desired features, resolves dependencies, and returns a new DataFrame with all
    requested features added.
    
    Args:
        df: Input DataFrame with OHLC data
        feature_list: List of features to create (from config.yaml)
        **params: Parameters to override defaults for any feature
            Example: volatility_window=30, rsi_window=25
        
    Returns:
        DataFrame with requested features added
        
    Example:
        df = create_features(
            df, 
            feature_list=['returns', 'volatility', 'rsi'],
            volatility_window=30
        )
    """
    if feature_list is None:
        feature_list = ['returns', 'volatility', 'rsi']  # Default features
        
    return registry.add_features(df, feature_list, **params)

# Create global registry instance
registry = FeatureRegistry()

# Register basic features with docstrings explaining their purpose and parameters
@registry.register('returns')
def add_returns_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentage returns calculated from close prices.
    
    Formula: (close_t - close_t-1) / close_t-1
    """
    df['returns'] = df['close'].pct_change()
    return df

@registry.register('volatility', dependencies=['returns'], window=20)
def add_volatility_feature(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add rolling volatility calculated as standard deviation of returns.
    
    Args:
        window: Number of periods to calculate volatility over
    
    Dependencies:
        - returns: Requires the returns feature to be calculated first
    """
    df['volatility'] = df['returns'].rolling(window=window).std()
    return df

@registry.register('rsi', window=14)
def add_rsi_feature(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) using SMA of gains/losses.
    
    Args:
        window: Number of periods for RSI calculation
    
    Formula: 100 - (100 / (1 + RS))
    where RS = average gain / average loss over window periods
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

@registry.register('is_nr4')
def add_is_nr4_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Narrow Range 4 (NR4) signal. An NR4 day is when the current day's range
    is the smallest of the last 4 days.
    
    The feature is binary (0 or 1) and is shifted forward one day, so that
    we know at the start of each day whether the previous day was an NR4 day.
    
    Process:
    1. Resample minute data to daily OHLC
    2. Calculate daily ranges (high - low)
    3. Check if today's range is narrowest of last 4 days
    4. Shift signal forward one day (no lookahead bias)
    5. Map back to minute data
    """
    print("Calculating NR4 feature...")
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    daily_df['range'] = daily_df['high'] - daily_df['low']
    daily_df['min_range_4d'] = daily_df['range'].rolling(window=4).min()
    daily_df['is_nr4_day'] = np.where(daily_df['range'] == daily_df['min_range_4d'], 1, 0)
    daily_df['is_nr4_signal_for_today'] = daily_df['is_nr4_day'].shift(1)
    
    daily_signal = daily_df[['is_nr4_signal_for_today']].reindex(df.index, method='ffill').fillna(0)
    df_with_feature = df.join(daily_signal)
    df_with_feature.rename(columns={'is_nr4_signal_for_today': 'is_nr4'}, inplace=True)
    
    print(f"NR4 signal calculated. Found {int(df_with_feature['is_nr4'].sum() / 390)} potential NR4 trading days.")
    return df_with_feature

@registry.register('is_nr7')
def add_is_nr7_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Narrow Range 7 (NR7) signal. An NR7 day is when the current day's range
    is the smallest of the last 7 days.
    
    Similar to NR4 but uses a 7-day lookback. Often considered a stronger
    signal of price consolidation than NR4.
    
    See is_nr4_feature documentation for detailed process explanation.
    """
    print("Calculating NR7 feature...")
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    daily_df['range'] = daily_df['high'] - daily_df['low']
    daily_df['min_range_7d'] = daily_df['range'].rolling(window=7).min()
    daily_df['is_nr7_day'] = np.where(daily_df['range'] == daily_df['min_range_7d'], 1, 0)
    daily_df['is_nr7_signal_for_today'] = daily_df['is_nr7_day'].shift(1)
    
    daily_signal = daily_df[['is_nr7_signal_for_today']].reindex(df.index, method='ffill').fillna(0)
    df_with_feature = df.join(daily_signal)
    df_with_feature.rename(columns={'is_nr7_signal_for_today': 'is_nr7'}, inplace=True)
    
    print(f"NR7 signal calculated. Found {int(df_with_feature['is_nr7'].sum() / 390)} potential NR7 trading days.")
    return df_with_feature