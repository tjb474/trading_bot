import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .base_pipeline import BasePipeline
from ml.feature_engineering import create_features

# Get logger
logger = logging.getLogger(__name__)

def _generate_signals(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """
    Generate breakout signals (both bullish and bearish) from the opening range.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        start_time (str): Start time of the range window (e.g., "09:30:00")
        end_time (str): End time of the range window (e.g., "10:00:00")

    Returns:
        pd.DataFrame: DataFrame with breakout signals and their direction
    """
    signals = pd.DataFrame(index=df.index)
    
    # Convert time strings to pandas time objects
    signals['time'] = pd.to_datetime(df.index).time
    range_start = pd.to_datetime(start_time).time()
    range_end = pd.to_datetime(end_time).time()
    
    # Calculate daily ranges
    signals['date'] = pd.to_datetime(df.index).date
    unique_dates = signals['date'].unique()
    
    breakout_signals = []
    breakout_directions = []
    logger.info(f"Analyzing {len(unique_dates)} trading days for breakout signals...")
    
    for date in unique_dates:
        day_mask = signals['date'] == date
        range_mask = (signals['time'] >= range_start) & (signals['time'] < range_end)
        
        # Get data for the opening range on this day
        range_data = df[day_mask & range_mask]
        if len(range_data) == 0:
            continue
            
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        
        # Get data after the range on this day
        post_range_mask = day_mask & (signals['time'] >= range_end)
        post_range_data = df[post_range_mask]
        
        # Look for breakouts (both directions)
        if len(post_range_data) > 0:
            # Check for upward breakouts
            up_breakout_mask = post_range_data['close'] > range_high
            # Check for downward breakouts  
            down_breakout_mask = post_range_data['close'] < range_low
            
            up_breakout_times = post_range_data[up_breakout_mask].index
            down_breakout_times = post_range_data[down_breakout_mask].index
            
            # Find the first breakout (either direction)
            first_breakout_time = None
            breakout_direction = None
            
            if len(up_breakout_times) > 0 and len(down_breakout_times) > 0:
                # Both occurred - take the earliest
                if up_breakout_times[0] < down_breakout_times[0]:
                    first_breakout_time = up_breakout_times[0]
                    breakout_direction = 1  # Bullish breakout
                else:
                    first_breakout_time = down_breakout_times[0]
                    breakout_direction = -1  # Bearish breakout
            elif len(up_breakout_times) > 0:
                first_breakout_time = up_breakout_times[0]
                breakout_direction = 1  # Bullish breakout
            elif len(down_breakout_times) > 0:
                first_breakout_time = down_breakout_times[0]
                breakout_direction = -1  # Bearish breakout
            
            if first_breakout_time is not None:
                breakout_signals.append(first_breakout_time)
                breakout_directions.append(breakout_direction)
    
    signals = pd.DataFrame(index=breakout_signals)
    signals['direction'] = breakout_directions
    logger.info(f"Generated {len(signals)} breakout signals ({sum(d == 1 for d in breakout_directions)} bullish, {sum(d == -1 for d in breakout_directions)} bearish).")
    return signals


def _label_trades(signals: pd.DataFrame, df: pd.DataFrame, tp_mult: float, sl_mult: float) -> pd.DataFrame:
    """
    Label trades based on whether they hit take profit or stop loss first.
    Handles both bullish (long) and bearish (short) breakout trades.
    
    Args:
        signals (pd.DataFrame): DataFrame with signal timestamps as index and 'direction' column
        df (pd.DataFrame): Full OHLC DataFrame
        tp_mult (float): Take profit multiplier of the range size
        sl_mult (float): Stop loss multiplier of the range size
    
    Returns:
        pd.DataFrame: signals DataFrame with added 'target' column
    """
    logger.info("Labeling trades as a win or loss...")
    labels = []
    
    for signal_time in signals.index:
        # Get the day's data up to the signal
        signal_date = pd.to_datetime(signal_time).date()
        day_data = df[pd.to_datetime(df.index).date == signal_date]
        
        # Create range_end with the same timezone as the DataFrame index
        range_end = pd.to_datetime(signal_time.strftime("%Y-%m-%d ") + "10:00:00").tz_localize(df.index.tz)
        range_data = day_data[day_data.index <= range_end]
        
        # Calculate range size and price targets
        range_size = range_data['high'].max() - range_data['low'].min()
        entry_price = df.loc[signal_time, 'close']
        direction = signals.loc[signal_time, 'direction']
        
        if direction == 1:  # Bullish breakout (long trade)
            tp_price = entry_price + (range_size * tp_mult)
            sl_price = entry_price - (range_size * sl_mult)
        else:  # Bearish breakout (short trade)
            tp_price = entry_price - (range_size * tp_mult)
            sl_price = entry_price + (range_size * sl_mult)
        
        # Get future data for this trade
        future_data = df.loc[signal_time:].iloc[1:]  # Start from next bar
        if len(future_data) == 0:
            labels.append(0)  # No future data available
            continue
        
        # Check which price level was hit first based on trade direction
        if direction == 1:  # Long trade
            hit_tp = future_data['high'] >= tp_price
            hit_sl = future_data['low'] <= sl_price
        else:  # Short trade
            hit_tp = future_data['low'] <= tp_price
            hit_sl = future_data['high'] >= sl_price
        
        if not (hit_tp.any() or hit_sl.any()):
            labels.append(0)  # Neither target was hit
            continue
            
        if not hit_sl.any():
            labels.append(1)  # Only TP was hit
            continue
            
        if not hit_tp.any():
            labels.append(0)  # Only SL was hit
            continue
            
        # Both were hit - check which came first
        first_tp = hit_tp.idxmax()
        first_sl = hit_sl.idxmax()
        labels.append(1 if first_tp < first_sl else 0)
    
    signals['target'] = labels
    logger.info("\n--- Target Label Distribution ---")
    logger.info(signals['target'].value_counts(normalize=True).to_frame(name='Proportion').assign(Count=signals['target'].value_counts()))
    
    # Show breakdown by direction
    logger.info("\n--- Breakout Direction Distribution ---")
    direction_counts = signals['direction'].value_counts()
    logger.info(f"Bullish breakouts (direction=1): {direction_counts.get(1, 0)}")
    logger.info(f"Bearish breakouts (direction=-1): {direction_counts.get(-1, 0)}")
    
    return signals


class OpenRangeBreakoutPipeline(BasePipeline):
    """
    Pipeline for training the ML-enhanced Open Range Breakout model.
    """
    def __init__(self, config):
        super().__init__(config)
        self.params = self.config.get_strategy_config('ml_open_range_breakout')

    def run(self):
        logger.info("--- Running Training Pipeline for: ML Open Range Breakout ---")
        
        # 1. Load Data
        logger.info(f"Loading data from: {self.config.data_path}")
        full_df = self.data_manager.load_ohlc_data(str(self.config.data_path))
        if full_df.empty:
            logger.critical("Data could not be loaded. Aborting pipeline.")
            return

        # 2. Split Data
        split_ratio = self.config.trading_params['train_test_split_ratio']
        train_df, _ = self.data_manager.split_data(full_df, split_ratio)

        # 3. Add all required features using the registry
        feature_list = self.params['features']['feature_list']
        feature_params = {
            'volatility_window': self.params['features'].get('volatility_window', 20),
            'rsi_window': self.params['features'].get('rsi_window', 14)
        }
        train_df = create_features(train_df, feature_list, **feature_params)
        
        # 4. Generate and Label Signals
        range_start = self.params['range']['start']
        range_end = self.params['range']['end']
        signals_df = _generate_signals(train_df, range_start, range_end)
        
        tp_mult = self.params['risk']['take_profit_multiplier']
        sl_mult = self.params['risk']['stop_loss_multiplier']
        labeled_signals = _label_trades(signals_df, train_df, tp_mult, sl_mult)

        # 5. Join features with labeled signals
        model_data = labeled_signals.join(train_df[feature_list], how='inner').dropna()

        # 6. ML Training
        X = model_data[feature_list + ['direction']]  # Include direction as a feature
        y = model_data['target']
        
        if X.empty:
            logger.critical("No data available for training.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        logger.info(f"Training on {len(X_train)} signals, testing on {len(X_test)}.")
        model = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        logger.info("--- Model Evaluation on Hold-Out Test Set ---")
        logger.info("\n" + classification_report(y_test, model.predict(X_test)))
        
        # 7. Save Model
        model_path = self.config.get_model_path('ml_open_range_breakout')
        joblib.dump(model, model_path)
        logger.info(f"Model successfully trained and saved to: {model_path}")