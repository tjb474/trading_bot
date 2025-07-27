# trading_bot/ml/feature_engineering.py

"""
Feature Engineering System
========================

This module implements a flexible and extensible feature engineering system using a registry pattern.
Features are registered with their dependencies and parameters, then created in the correct order.
"""

from typing import Callable, Dict, List, Optional
import pandas as pd
import numpy as np
from functools import wraps
import logging

# Get the logger
logger = logging.getLogger(__name__)

class FeatureRegistry:
    """
    Registry for feature engineering functions that handles dependencies and parameters.
    """

    def __init__(self):
        self._features: Dict[str, Callable] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._parameters: Dict[str, Dict] = {}

    def register(self, name: str, dependencies: Optional[List[str]] = None, **params):
        """
        Decorator to register a feature engineering function.
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._features[name] = wrapper
            self._dependencies[name] = dependencies or []
            self._parameters[name] = params
            logger.debug(f"Feature '{name}' registered with dependencies {dependencies or []}")
            return wrapper
        return decorator

    def add_features(self, df: pd.DataFrame, feature_list: List[str], **params) -> pd.DataFrame:
        """
        Add requested features to the dataframe in the correct order based on dependencies.
        """
        result_df = df.copy()
        
        all_features_to_process = set(feature_list)
        queue = list(feature_list)
        while queue:
            feature = queue.pop(0)
            if feature not in self._features:
                raise ValueError(f"Feature '{feature}' not found in registry")
            for dep in self._dependencies.get(feature, []):
                if dep not in all_features_to_process:
                    all_features_to_process.add(dep)
                    queue.append(dep)

        processed = set()
        execution_order = []
        
        while len(processed) < len(all_features_to_process):
            made_progress = False
            for feature in all_features_to_process:
                if feature not in processed and all(dep in processed for dep in self._dependencies.get(feature, [])):
                    execution_order.append(feature)
                    processed.add(feature)
                    made_progress = True
            
            if not made_progress:
                unresolved = all_features_to_process - processed
                raise ValueError(f"Circular or missing dependency detected. Could not resolve: {unresolved}")

        for feature_name in execution_order:
            feature_params = self._parameters[feature_name].copy()
            feature_params.update(params)
            
            logger.info(f"Adding feature: '{feature_name}'")
            result_df = self._features[feature_name](result_df, **feature_params)

        return result_df

# --- Global Instance and Main Function ---
registry = FeatureRegistry()

def create_features(df: pd.DataFrame, feature_list: Optional[List[str]] = None, **params) -> pd.DataFrame:
    if feature_list is None:
        feature_list = []
    
    logger.info(f"Request to create features: {feature_list}")
    return registry.add_features(df, feature_list, **params)


# --- Feature Implementations ---

@registry.register('returns')
def add_returns_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df['returns'] = df['close'].pct_change()
    return df

@registry.register('volatility', dependencies=['returns'], volatility_window=20)
def add_volatility_feature(df: pd.DataFrame, volatility_window: int, **kwargs) -> pd.DataFrame:
    df['volatility'] = df['returns'].rolling(window=volatility_window).std()
    return df

@registry.register('rsi', rsi_window=14)
def add_rsi_feature(df: pd.DataFrame, rsi_window: int, **kwargs) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

@registry.register('is_nr4')
def add_is_nr4_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    daily_df = df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
    daily_df['range'] = daily_df['high'] - daily_df['low']
    daily_df['is_nr4'] = (daily_df['range'] == daily_df['range'].rolling(window=4).min()).astype(int)
    df['is_nr4'] = df.index.normalize().map(daily_df['is_nr4'].shift(1))
    df['is_nr4'] = df['is_nr4'].ffill().fillna(0)
    return df

@registry.register('is_nr7')
def add_is_nr7_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    daily_df = df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
    daily_df['range'] = daily_df['high'] - daily_df['low']
    daily_df['is_nr7'] = (daily_df['range'] == daily_df['range'].rolling(window=7).min()).astype(int)
    df['is_nr7'] = df.index.normalize().map(daily_df['is_nr7'].shift(1))
    df['is_nr7'] = df['is_nr7'].ffill().fillna(0)
    return df

@registry.register('day_of_week')
def add_day_of_week_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    days = {0: 'is_monday', 1: 'is_tuesday', 2: 'is_wednesday', 3: 'is_thursday'}
    day_of_week = df.index.dayofweek
    for day_num, day_name in days.items():
        df[day_name] = (day_of_week == day_num).astype(int)
    return df
    
@registry.register('breakout_direction', range_start="09:30:00", range_end="10:15:00")
def add_breakout_direction_feature(df: pd.DataFrame, range_start: str, range_end: str, **kwargs) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['breakout_direction'] = 0
    
    def detect_breakout(day_data: pd.DataFrame) -> pd.DataFrame:
        try:
            # THE FIX: Get the timezone from the data itself.
            tz = day_data.index.tz
            
            day_str = day_data.index[0].strftime('%Y-%m-%d')
            
            # Create tz-naive timestamps from string
            start_ts_naive = pd.to_datetime(f"{day_str} {range_start}")
            end_ts_naive = pd.to_datetime(f"{day_str} {range_end}")

            # Localize the timestamps to match the DataFrame's timezone (or keep naive if tz is None)
            start_ts = start_ts_naive.tz_localize(tz)
            end_ts = end_ts_naive.tz_localize(tz)

            # Now the comparison will always be between objects of the same type.
            range_data = day_data.loc[start_ts:end_ts]
            if range_data.empty: return day_data
            
            or_high = range_data['high'].max()
            or_low = range_data['low'].min()
            
            post_range_data = day_data.loc[end_ts:]
            
            first_bull_break = (post_range_data['close'] > or_high).idxmax() if (post_range_data['close'] > or_high).any() else pd.NaT
            first_bear_break = (post_range_data['close'] < or_low).idxmax() if (post_range_data['close'] < or_low).any() else pd.NaT

            if pd.notna(first_bull_break) and (pd.isna(first_bear_break) or first_bull_break < first_bear_break):
                day_data.loc[first_bull_break:, 'breakout_direction'] = 1
            elif pd.notna(first_bear_break):
                day_data.loc[first_bear_break:, 'breakout_direction'] = -1
        except Exception as e:
            logger.error(f"Error calculating breakout for day {day_data.index[0].date()}: {e}")
        return day_data

    df_copy = df_copy.groupby(df_copy.index.date, group_keys=False).apply(detect_breakout)
    df['breakout_direction'] = df_copy['breakout_direction']
    return df

# --- Opening Range (OR) Features ---

def get_daily_or_metrics(df: pd.DataFrame, range_start: str, range_end: str, **kwargs) -> pd.DataFrame:
    if 'or_high' in df.columns:
        return df

    daily_metrics = {}
    tz = df.index.tz # Get timezone once
    for date, day_data in df.groupby(df.index.date):
        try:
            day_str = date.strftime('%Y-%m-%d')
            start_ts = pd.to_datetime(f"{day_str} {range_start}").tz_localize(tz)
            end_ts = pd.to_datetime(f"{day_str} {range_end}").tz_localize(tz)
            range_data = day_data.loc[start_ts:end_ts]
            
            if not range_data.empty:
                daily_metrics[date] = {
                    'or_high': range_data['high'].max(),
                    'or_low': range_data['low'].min(),
                    'or_volume': range_data['volume'].sum()
                }
        except Exception:
            pass
    
    daily_df = pd.DataFrame.from_dict(daily_metrics, orient='index')
    daily_df.index = pd.to_datetime(daily_df.index).tz_localize(tz)
    
    df['or_high'] = df.index.normalize().map(daily_df['or_high'])
    df['or_low'] = df.index.normalize().map(daily_df['or_low'])
    df['or_volume'] = df.index.normalize().map(daily_df['or_volume'])
    
    df[['or_high', 'or_low', 'or_volume']] = df[['or_high', 'or_low', 'or_volume']].ffill()
    return df

@registry.register('or_size_absolute', range_start="09:30:00", range_end="10:15:00")
def add_or_size_absolute_feature(df: pd.DataFrame, range_start: str, range_end: str, **kwargs) -> pd.DataFrame:
    df = get_daily_or_metrics(df, range_start=range_start, range_end=range_end)
    df['or_size_absolute'] = df['or_high'] - df['or_low']
    return df

@registry.register('daily_atr', atr_window=14)
def add_daily_atr_feature(df: pd.DataFrame, atr_window: int, **kwargs) -> pd.DataFrame:
    daily_df = df.resample('D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    high_low = daily_df['high'] - daily_df['low']
    high_close = np.abs(daily_df['high'] - daily_df['close'].shift())
    low_close = np.abs(daily_df['low'] - daily_df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    daily_df['daily_atr'] = tr.rolling(window=atr_window).mean()
    
    df['daily_atr'] = df.index.normalize().map(daily_df['daily_atr'])
    df['daily_atr'] = df['daily_atr'].ffill()
    return df

@registry.register('or_size_normalized', dependencies=['or_size_absolute', 'daily_atr'])
def add_or_size_normalized_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df['or_size_normalized'] = df['or_size_absolute'] / df['daily_atr']
    return df

@registry.register('or_volume', range_start="09:30:00", range_end="10:15:00")
def add_or_volume_feature(df: pd.DataFrame, range_start: str, range_end: str, **kwargs) -> pd.DataFrame:
    df = get_daily_or_metrics(df, range_start=range_start, range_end=range_end)
    return df

@registry.register('or_volume_vs_average', dependencies=['or_volume'], lookback_days=20)
def add_or_volume_vs_average_feature(df: pd.DataFrame, lookback_days: int, **kwargs) -> pd.DataFrame:
    daily_volume = df[['or_volume']].resample('D').first().dropna()
    avg_daily_or_volume = daily_volume['or_volume'].rolling(window=lookback_days).mean()
    
    df['avg_or_volume'] = df.index.normalize().map(avg_daily_or_volume)
    df['avg_or_volume'] = df['avg_or_volume'].ffill()
    
    df['or_volume_vs_average'] = df['or_volume'] / df['avg_or_volume']
    df.drop(columns=['avg_or_volume'], inplace=True, errors='ignore')
    return df

@registry.register('or_midpoint_location', dependencies=['or_size_absolute'])
def add_or_midpoint_location_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df['or_midpoint'] = (df['or_high'] + df['or_low']) / 2
    
    daily_hl = df.resample('D').agg({'high':'max', 'low':'min'}).dropna()
    daily_hl['prev_day_high'] = daily_hl['high'].shift(1)
    daily_hl['prev_day_low'] = daily_hl['low'].shift(1)
    daily_hl['prev_day_range'] = daily_hl['prev_day_high'] - daily_hl['prev_day_low']

    df['prev_day_low'] = df.index.normalize().map(daily_hl['prev_day_low'])
    df['prev_day_range'] = df.index.normalize().map(daily_hl['prev_day_range'])
    df[['prev_day_low', 'prev_day_range']] = df[['prev_day_low', 'prev_day_range']].ffill()
    
    df['or_midpoint_location'] = (df['or_midpoint'] - df['prev_day_low']) / df['prev_day_range']
    df.drop(columns=['or_high', 'or_low', 'or_midpoint', 'prev_day_low', 'prev_day_range'], inplace=True, errors='ignore')
    return df