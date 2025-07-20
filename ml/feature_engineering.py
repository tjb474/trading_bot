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
import logging

# Get the logger without configuring it - configuration will come from Config class
logger = logging.getLogger(__name__)

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
            
            logger.info(f"Adding feature: {feature}")
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

@registry.register('day_of_week')
def add_day_of_week_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add day of the week as categorical features (one-hot encoded).
    
    Creates binary features for each trading day of the week (avoiding dummy trap):
    - is_monday: 1 if Monday, 0 otherwise
    - is_tuesday: 1 if Tuesday, 0 otherwise  
    - is_wednesday: 1 if Wednesday, 0 otherwise
    - is_thursday: 1 if Thursday, 0 otherwise
    
    Note: Friday is omitted to avoid the dummy variable trap. When all other days 
    are 0, it implicitly represents Friday. This prevents perfect multicollinearity
    in linear models while preserving all information.
    
    This helps capture day-of-week effects in trading patterns, such as:
    - Monday reversals after weekend news
    - Friday profit-taking (captured when other days = 0)
    - Mid-week momentum patterns
    
    Returns:
        DataFrame with day-of-week categorical features added
    """
    logger.info("Adding day of week categorical features (avoiding dummy trap)...")
    
    df = df.copy()
    
    # Get day of week (0=Monday, 1=Tuesday, ..., 4=Friday)
    df['weekday'] = df.index.dayofweek
    
    # Create one-hot encoded features for each trading day (except Friday to avoid dummy trap)
    df['is_monday'] = (df['weekday'] == 0).astype(int)
    df['is_tuesday'] = (df['weekday'] == 1).astype(int) 
    df['is_wednesday'] = (df['weekday'] == 2).astype(int)
    df['is_thursday'] = (df['weekday'] == 3).astype(int)
    # Friday omitted: when all others are 0, it's implicitly Friday
    
    # Clean up temporary column
    df = df.drop('weekday', axis=1)
    
    # Count trading days for each day of week
    day_counts = {
        'Monday': df['is_monday'].sum(),
        'Tuesday': df['is_tuesday'].sum(),
        'Wednesday': df['is_wednesday'].sum(), 
        'Thursday': df['is_thursday'].sum(),
        'Friday': len(df[df.index.dayofweek == 4])  # Friday count (implicit)
    }
    
    # Convert to daily counts (assuming ~390 bars per day)
    daily_counts = {day: int(count / 390) if count > 0 else 0 
                   for day, count in day_counts.items()}
    
    logger.info(f"Day of week features added (Friday implicit). Trading days: {daily_counts}")
    
    return df

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
    logger.info("Calculating NR4 feature...")
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
    
    logger.debug("Daily NR4 data:\n%s", daily_df.head(50))

    df_with_feature = df.join(daily_signal)
    df_with_feature.rename(columns={'is_nr4_signal_for_today': 'is_nr4'}, inplace=True)
    
    nr4_days = int(df_with_feature['is_nr4'].sum() / 390)
    logger.info(f"NR4 signal calculated. Found {nr4_days} potential NR4 trading days.")
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
    logger.info("Calculating NR7 feature...")
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
    
    nr7_days = int(df_with_feature['is_nr7'].sum() / 390)
    logger.info(f"NR7 signal calculated. Found {nr7_days} potential NR7 trading days.")
    return df_with_feature

@registry.register('breakout_direction', range_start='09:30:00', range_end='10:15:00')
def add_breakout_direction_feature(df: pd.DataFrame, range_start: str = '09:30:00', range_end: str = '10:15:00') -> pd.DataFrame:
    """
    Add breakout direction feature that identifies Open Range Breakout direction in real-time.
    
    This feature can be calculated during live trading as soon as a breakout occurs.
    It assigns:
    - 1 for bullish breakouts (price breaks above opening range high)
    - -1 for bearish breakouts (price breaks below opening range low)
    - 0 for no breakout or during the opening range period
    
    Args:
        range_start: Start time of the opening range (e.g., "09:30:00")
        range_end: End time of the opening range (e.g., "10:15:00")
    
    Process:
    1. For each day, calculate the opening range high/low
    2. After the range ends, monitor for breakouts
    3. Set direction to 1/-1 immediately when breakout occurs
    4. Maintain the direction for the rest of the trading day
    5. Only the first breakout of the day counts (no reversals)
    
    Note: This feature is designed for real-time use - it doesn't look ahead
    and can be calculated bar-by-bar as new data arrives.
    """
    logger.info(f"Calculating breakout direction feature (range: {range_start} to {range_end})...")
    
    # Initialize the feature column
    df = df.copy()
    df['breakout_direction'] = 0
    
    # Convert time strings to time objects for comparison
    range_start_time = pd.to_datetime(range_start).time()
    range_end_time = pd.to_datetime(range_end).time()
    
    # Add time and date columns for processing
    df['time'] = pd.to_datetime(df.index).time
    df['date'] = pd.to_datetime(df.index).date
    
    # Process each trading day
    unique_dates = df['date'].unique()
    breakout_count = {'bullish': 0, 'bearish': 0}
    
    for date in unique_dates:
        day_mask = df['date'] == date
        day_data = df[day_mask].copy()
        
        # Get opening range data
        range_mask = (day_data['time'] >= range_start_time) & (day_data['time'] < range_end_time)
        range_data = day_data[range_mask]
        
        if len(range_data) == 0:
            continue
            
        # Calculate range bounds
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        
        # Get post-range data
        post_range_mask = day_data['time'] >= range_end_time
        post_range_indices = day_data[post_range_mask].index
        
        breakout_detected = False
        
        # Process each bar after range end chronologically
        for idx in post_range_indices:
            if breakout_detected:
                # Maintain the breakout direction for rest of day
                df.loc[idx, 'breakout_direction'] = current_direction
            else:
                # Check for breakout
                current_price = df.loc[idx, 'close']
                
                if current_price > range_high:
                    # Bullish breakout
                    current_direction = 1
                    df.loc[idx, 'breakout_direction'] = current_direction
                    breakout_detected = True
                    breakout_count['bullish'] += 1
                elif current_price < range_low:
                    # Bearish breakout
                    current_direction = -1
                    df.loc[idx, 'breakout_direction'] = current_direction
                    breakout_detected = True
                    breakout_count['bearish'] += 1
                # else: no breakout yet, stays 0
    
    # Clean up temporary columns
    df = df.drop(['time', 'date'], axis=1)
    
    # Debug: Show daily breakout signals for easier analysis
    daily_breakout = df[df['breakout_direction'] != 0].resample('D').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'breakout_direction': 'first'  # First (and only) breakout direction of the day
    }).dropna()
    logger.debug("Daily breakout signals:\n%s", daily_breakout.head(50))
    
    total_breakouts = breakout_count['bullish'] + breakout_count['bearish']
    logger.info(f"Breakout direction feature calculated. Found {total_breakouts} breakouts "
               f"({breakout_count['bullish']} bullish, {breakout_count['bearish']} bearish).")
    
    return df

@registry.register('or_size_absolute', range_start='09:30:00', range_end='10:15:00')
def add_or_size_absolute_feature(df: pd.DataFrame, range_start: str = '09:30:00', range_end: str = '10:15:00') -> pd.DataFrame:
    """
    Add Opening Range size in absolute price terms (OR High - OR Low).
    
    IMPORTANT: No lookahead bias - OR size is only available AFTER the range period ends.
    During the range formation (09:30-10:15), this feature is 0.
    After range ends, it shows the completed OR size.
    
    Args:
        range_start: Start time of the opening range (e.g., "09:30:00")
        range_end: End time of the opening range (e.g., "10:15:00")
    
    Returns:
        DataFrame with 'or_size_absolute' column added
    """
    logger.info(f"Calculating OR absolute size feature (range: {range_start} to {range_end})...")
    
    df = df.copy()
    df['or_size_absolute'] = 0.0
    
    # Convert time strings to time objects
    range_start_time = pd.to_datetime(range_start).time()
    range_end_time = pd.to_datetime(range_end).time()
    
    # Add time and date columns for processing
    df['time'] = pd.to_datetime(df.index).time
    df['date'] = pd.to_datetime(df.index).date
    
    # Process each trading day
    for date in df['date'].unique():
        day_mask = df['date'] == date
        day_data = df[day_mask]
        
        # Get opening range data
        range_mask = (day_data['time'] >= range_start_time) & (day_data['time'] < range_end_time)
        range_data = day_data[range_mask]
        
        if len(range_data) > 0:
            or_high = range_data['high'].max()
            or_low = range_data['low'].min()
            or_size = or_high - or_low
            
            # CRITICAL: Only fill AFTER the range period ends (no lookahead bias)
            post_range_mask = day_mask & (df['time'] >= range_end_time)
            df.loc[post_range_mask, 'or_size_absolute'] = or_size
    
    # Clean up temporary columns
    df = df.drop(['time', 'date'], axis=1)
    
    avg_or_size = df[df['or_size_absolute'] > 0]['or_size_absolute'].mean()
    logger.info(f"OR absolute size calculated. Average OR size: ${avg_or_size:.2f}")
    
    return df

@registry.register('or_size_normalized', dependencies=['or_size_absolute'], range_start='09:30:00', range_end='10:15:00', atr_window=14)
def add_or_size_normalized_feature(df: pd.DataFrame, range_start: str = '09:30:00', range_end: str = '10:15:00', atr_window: int = 14) -> pd.DataFrame:
    """
    Add Opening Range size normalized by Average True Range (OR Size / ATR).
    
    IMPORTANT: No lookahead bias - uses ATR from PREVIOUS days only.
    The ATR calculation excludes the current day to avoid lookahead bias.
    
    Args:
        range_start: Start time of the opening range (e.g., "09:30:00") 
        range_end: End time of the opening range (e.g., "10:15:00")
        atr_window: Window for ATR calculation (default: 14)
    
    Dependencies:
        - or_size_absolute: Requires absolute OR size to be calculated first
    
    Returns:
        DataFrame with 'or_size_normalized' column added
    """
    logger.info(f"Calculating OR normalized size feature (ATR window: {atr_window})...")
    
    df = df.copy()
    
    # Calculate daily OHLC first
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min', 
        'close': 'last',
        'or_size_absolute': 'first'  # Take first value since it's the same for the whole day
    }).dropna()
    
    # Calculate True Range components
    daily_df['tr1'] = daily_df['high'] - daily_df['low']
    daily_df['tr2'] = abs(daily_df['high'] - daily_df['close'].shift(1))
    daily_df['tr3'] = abs(daily_df['low'] - daily_df['close'].shift(1))
    daily_df['true_range'] = daily_df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # CRITICAL: Calculate ATR using PREVIOUS days only (shift forward to avoid lookahead)
    daily_df['atr'] = daily_df['true_range'].shift(1).rolling(window=atr_window).mean()
    
    # Calculate normalized OR size (using yesterday's ATR)
    daily_df['or_size_normalized'] = daily_df['or_size_absolute'] / daily_df['atr']
    
    # Map back to minute data
    normalized_signal = daily_df[['or_size_normalized']].reindex(df.index, method='ffill').fillna(0)
    df = df.join(normalized_signal)
    
    # Only calculate average for non-zero values
    valid_normalized = df[df['or_size_normalized'] > 0]['or_size_normalized']
    if len(valid_normalized) > 0:
        avg_normalized = valid_normalized.mean()
        logger.info(f"OR normalized size calculated. Average normalized OR size: {avg_normalized:.2f}")
    else:
        logger.info("OR normalized size calculated. No valid normalized values yet.")
    
    return df

@registry.register('or_volume', range_start='09:30:00', range_end='10:15:00')
def add_or_volume_feature(df: pd.DataFrame, range_start: str = '09:30:00', range_end: str = '10:15:00') -> pd.DataFrame:
    """
    Add Opening Range total volume feature.
    
    Total volume traded during the opening range formation. High volume can suggest
    stronger conviction behind the eventual breakout.
    
    Args:
        range_start: Start time of the opening range (e.g., "09:30:00")
        range_end: End time of the opening range (e.g., "10:15:00")
    
    Returns:
        DataFrame with 'or_volume' column added
    """
    logger.info(f"Calculating OR volume feature (range: {range_start} to {range_end})...")
    
    df = df.copy()
    df['or_volume'] = 0.0
    
    # Convert time strings to time objects
    range_start_time = pd.to_datetime(range_start).time()
    range_end_time = pd.to_datetime(range_end).time()
    
    # Add time and date columns for processing
    df['time'] = pd.to_datetime(df.index).time
    df['date'] = pd.to_datetime(df.index).date
    
    # Process each trading day
    for date in df['date'].unique():
        day_mask = df['date'] == date
        day_data = df[day_mask]
        
        # Get opening range data
        range_mask = (day_data['time'] >= range_start_time) & (day_data['time'] < range_end_time)
        range_data = day_data[range_mask]
        
        if len(range_data) > 0:
            or_total_volume = range_data['volume'].sum()
            
            # CRITICAL: Only populate AFTER the opening range ends (no lookahead bias)
            post_range_mask = day_data['time'] > range_end_time
            
            # Only set the feature for bars AFTER the opening range completes
            df.loc[day_data.index[post_range_mask], 'or_volume'] = or_total_volume
    
    # Clean up temporary columns
    df = df.drop(['time', 'date'], axis=1)
    
    valid_volume = df[df['or_volume'] > 0]['or_volume']
    if len(valid_volume) > 0:
        avg_or_volume = valid_volume.mean()
        logger.info(f"OR volume calculated. Average OR volume: {avg_or_volume:,.0f}")
    else:
        logger.info("OR volume calculated. No completed ranges found yet.")
    
    return df

@registry.register('or_volume_vs_average', dependencies=['or_volume'], range_start='09:30:00', range_end='10:15:00', lookback_days=20)
def add_or_volume_vs_average_feature(df: pd.DataFrame, range_start: str = '09:30:00', range_end: str = '10:15:00', lookback_days: int = 20) -> pd.DataFrame:
    """
    Add Opening Range volume relative to historical average (OR Volume / Average OR Volume).
    
    IMPORTANT: No lookahead bias - uses PREVIOUS days only for average calculation.
    The historical average excludes the current day to avoid lookahead bias.
    
    Args:
        range_start: Start time of the opening range (e.g., "09:30:00")
        range_end: End time of the opening range (e.g., "10:15:00") 
        lookback_days: Number of days to use for average calculation (default: 20)
    
    Dependencies:
        - or_volume: Requires OR volume to be calculated first
    
    Returns:
        DataFrame with 'or_volume_vs_average' column added
    """
    logger.info(f"Calculating OR volume vs average feature (lookback: {lookback_days} days)...")
    
    df = df.copy()
    
    # Get daily OR volumes
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'or_volume': 'first'  # Take first value since it's the same for the whole day
    }).dropna()
    
    # CRITICAL: Calculate rolling average using PREVIOUS days only (shift forward to avoid lookahead)
    daily_df['or_volume_avg'] = daily_df['or_volume'].shift(1).rolling(window=lookback_days).mean()
    
    # Calculate ratio (current OR volume vs historical average)
    daily_df['or_volume_vs_average'] = daily_df['or_volume'] / daily_df['or_volume_avg']
    
    # Map back to minute data
    volume_ratio_signal = daily_df[['or_volume_vs_average']].reindex(df.index, method='ffill').fillna(1.0)
    df = df.join(volume_ratio_signal)
    
    valid_ratios = df[df['or_volume_vs_average'] != 1.0]['or_volume_vs_average']
    if len(valid_ratios) > 0:
        avg_ratio = valid_ratios.mean()
        logger.info(f"OR volume vs average calculated. Average ratio: {avg_ratio:.2f}")
    else:
        logger.info("OR volume vs average calculated. No completed ranges found yet.")
    
    return df

@registry.register('or_midpoint_location', range_start='09:30:00', range_end='10:15:00')
def add_or_midpoint_location_feature(df: pd.DataFrame, range_start: str = '09:30:00', range_end: str = '10:15:00') -> pd.DataFrame:
    """
    Add Opening Range midpoint location relative to previous day's close.
    
    This feature shows where the initial balance area is forming relative to
    yesterday's settlement. Formula: (OR High + OR Low) / 2 - Previous Close
    
    Args:
        range_start: Start time of the opening range (e.g., "09:30:00")
        range_end: End time of the opening range (e.g., "10:15:00")
    
    Returns:
        DataFrame with 'or_midpoint_location' column added
    """
    logger.info(f"Calculating OR midpoint location feature (range: {range_start} to {range_end})...")
    
    df = df.copy()
    df['or_midpoint_location'] = 0.0
    
    # Convert time strings to time objects
    range_start_time = pd.to_datetime(range_start).time()
    range_end_time = pd.to_datetime(range_end).time()
    
    # Add time and date columns for processing
    df['time'] = pd.to_datetime(df.index).time
    df['date'] = pd.to_datetime(df.index).date
    
    # Get daily close prices for reference
    daily_df = df.resample('D').agg({
        'close': 'last'
    }).dropna()
    daily_df['prev_close'] = daily_df['close'].shift(1)
    
    # Process each trading day
    for date in df['date'].unique():
        day_mask = df['date'] == date
        day_data = df[day_mask]
        
        # Get opening range data
        range_mask = (day_data['time'] >= range_start_time) & (day_data['time'] < range_end_time)
        range_data = day_data[range_mask]
        
        if len(range_data) > 0:
            or_high = range_data['high'].max()
            or_low = range_data['low'].min()
            or_midpoint = (or_high + or_low) / 2
@registry.register('or_midpoint_location', range_start='09:30:00', range_end='10:15:00')
def add_or_midpoint_location_feature(df: pd.DataFrame, range_start: str = '09:30:00', range_end: str = '10:15:00') -> pd.DataFrame:
    """
    Add Opening Range midpoint location relative to previous day's close.
    
    IMPORTANT: No lookahead bias - midpoint is only available AFTER the opening range completes.
    The feature shows where the initial balance area formed relative to yesterday's settlement,
    but only becomes available at the end of the range period.
    
    Args:
        range_start: Start time of the opening range (e.g., "09:30:00")
        range_end: End time of the opening range (e.g., "10:15:00")
    
    Returns:
        DataFrame with 'or_midpoint_location' column added
    """
    logger.info(f"Calculating OR midpoint location feature (range: {range_start} to {range_end})...")
    
    df = df.copy()
    df['or_midpoint_location'] = 0.0
    
    # Convert time strings to time objects
    range_start_time = pd.to_datetime(range_start).time()
    range_end_time = pd.to_datetime(range_end).time()
    
    # Add time and date columns for processing
    df['time'] = pd.to_datetime(df.index).time
    df['date'] = pd.to_datetime(df.index).date
    
    # Get daily close prices for reference
    daily_df = df.resample('D').agg({
        'close': 'last'
    }).dropna()
    daily_df['prev_close'] = daily_df['close'].shift(1)
    
    # Process each trading day
    for date in df['date'].unique():
        day_mask = df['date'] == date
        day_data = df[day_mask]
        
        # Get opening range data
        range_mask = (day_data['time'] >= range_start_time) & (day_data['time'] < range_end_time)
        range_data = day_data[range_mask]
        
        if len(range_data) > 0:
            or_high = range_data['high'].max()
            or_low = range_data['low'].min()
            or_midpoint = (or_high + or_low) / 2
            
            # Get previous day's close
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            if date_str in daily_df.index:
                prev_close = daily_df.loc[date_str, 'prev_close']
                
                if pd.notna(prev_close):
                    midpoint_location = or_midpoint - prev_close
                    
                    # CRITICAL: Only populate AFTER the opening range ends (no lookahead bias)
                    post_range_mask = day_data['time'] > range_end_time
                    
                    # Only set the feature for bars AFTER the opening range completes
                    df.loc[day_data.index[post_range_mask], 'or_midpoint_location'] = midpoint_location
    
    # Clean up temporary columns
    df = df.drop(['time', 'date'], axis=1)
    
    valid_locations = df[df['or_midpoint_location'] != 0.0]['or_midpoint_location']
    if len(valid_locations) > 0:
        avg_midpoint_location = valid_locations.mean()
        logger.info(f"OR midpoint location calculated. Average location: ${avg_midpoint_location:.2f}")
    else:
        logger.info("OR midpoint location calculated. No completed ranges found yet.")
    
    return df