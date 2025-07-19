import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .base_pipeline import BasePipeline
from ml.feature_engineering import create_features

def _generate_signals(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """
    Generate buy signals based on breakouts from the opening range.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        start_time (str): Start time of the range window (e.g., "09:30:00")
        end_time (str): End time of the range window (e.g., "10:00:00")

    Returns:
        pd.DataFrame: DataFrame with only the rows where breakout signals occurred
    """
    signals = pd.DataFrame(index=df.index)
    
    # Convert time strings to pandas time objects
    signals['time'] = pd.to_datetime(df.index).time
    range_start = pd.to_datetime(start_time).time()
    range_end = pd.to_datetime(end_time).time()
    
    # Calculate daily ranges
    signals['date'] = pd.to_datetime(df.index).date
    unique_dates = signals['date'].unique()
    
    buy_signals = []
    print(f"Analyzing {len(unique_dates)} trading days for breakout signals...")
    
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
        
        # Look for breakouts
        if len(post_range_data) > 0:
            breakout_mask = post_range_data['close'] > range_high
            if breakout_mask.any():
                # Get the first breakout of the day
                breakout_signal = post_range_data[breakout_mask].iloc[0]
                buy_signals.append(breakout_signal.name)
    
    signals = pd.DataFrame(index=buy_signals)
    print(f"Generated {len(signals)} breakout signals.")
    return signals


def _label_trades(signals: pd.DataFrame, df: pd.DataFrame, tp_mult: float, sl_mult: float) -> pd.DataFrame:
    """
    Label trades based on whether they hit take profit or stop loss first.
    
    Args:
        signals (pd.DataFrame): DataFrame with signal timestamps as index
        df (pd.DataFrame): Full OHLC DataFrame
        tp_mult (float): Take profit multiplier of the range size
        sl_mult (float): Stop loss multiplier of the range size
    
    Returns:
        pd.DataFrame: signals DataFrame with added 'target' column
    """
    print("Labeling trades...")
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
        tp_price = entry_price + (range_size * tp_mult)
        sl_price = entry_price - (range_size * sl_mult)
        
        # Get future data for this trade
        future_data = df.loc[signal_time:].iloc[1:]  # Start from next bar
        if len(future_data) == 0:
            labels.append(0)  # No future data available
            continue
            
        # Check which price level was hit first
        hit_tp = future_data['high'] >= tp_price
        hit_sl = future_data['low'] <= sl_price
        
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
    print("\n--- Target Label Distribution ---")
    print(signals['target'].value_counts(normalize=True))
    return signals


class OpenRangeBreakoutPipeline(BasePipeline):
    """
    Pipeline for training the ML-enhanced Open Range Breakout model.
    """
    def __init__(self):
        super().__init__()
        self.params = self.config.get_strategy_config('ml_open_range_breakout')

    def run(self):
        print("--- [INFO] Running Training Pipeline for: ML Open Range Breakout ---")
        
        # 1. Load Data
        print(f"[INFO] Loading data from: {self.config.data_path}")
        full_df = self.data_manager.load_ohlc_data(str(self.config.data_path))
        if full_df.empty:
            print("[CRITICAL] Data could not be loaded. Aborting pipeline.")
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
        X = model_data[feature_list]
        y = model_data['target']
        
        if X.empty:
            print("[CRITICAL] No data available for training.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print(f"\n[INFO] Training on {len(X_train)} signals, testing on {len(X_test)}.")
        model = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        print("\n--- [INFO] Model Evaluation on Hold-Out Test Set ---")
        print(classification_report(y_test, model.predict(X_test)))
        
        # 7. Save Model
        model_path = self.config.get_model_path('ml_open_range_breakout')
        joblib.dump(model, model_path)
        print(f"\n[INFO] Model successfully trained and saved to: {model_path}")