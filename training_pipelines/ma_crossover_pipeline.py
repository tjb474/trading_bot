import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .base_pipeline import BasePipeline

def _generate_signals(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Generate buy signals based on moving average crossovers.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        short_window (int): Short moving average window
        long_window (int): Long moving average window
        
    Returns:
        pd.DataFrame: DataFrame with only the rows where crossover signals occurred
    """
    # Calculate moving averages
    short_ma = df['close'].rolling(window=short_window).mean()
    long_ma = df['close'].rolling(window=long_window).mean()
    
    # Generate crossover signals
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    
    # Detect crossovers (short MA crosses above long MA)
    signals.loc[short_ma > long_ma, 'signal'] = 1
    
    # Only keep the crossover points
    signals['position_changed'] = signals['signal'].diff()
    buy_signals = signals[signals['position_changed'] > 0]
    
    print(f"Generated {len(buy_signals)} MA crossover signals.")
    return buy_signals


def _label_trades(signals: pd.DataFrame, df: pd.DataFrame, tp_mult: float, sl_mult: float) -> pd.DataFrame:
    """
    Label trades based on whether they hit take profit or stop loss first.
    
    Args:
        signals (pd.DataFrame): DataFrame with signal timestamps as index
        df (pd.DataFrame): Full OHLC DataFrame
        tp_mult (float): Take profit multiplier of ATR
        sl_mult (float): Stop loss multiplier of ATR
        
    Returns:
        pd.DataFrame: signals DataFrame with added 'target' column
    """
    print("Labeling trades...")
    labels = []
    
    # Calculate ATR for dynamic take profit and stop loss
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=14).mean()
    
    for signal_time in signals.index:
        entry_price = df.loc[signal_time, 'close']
        current_atr = atr.loc[signal_time]
        
        tp_price = entry_price + (current_atr * tp_mult)
        sl_price = entry_price - (current_atr * sl_mult)
        
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


class MovingAverageCrossoverPipeline(BasePipeline):
    """
    Pipeline for training the ML-enhanced Moving Average Crossover model.
    """
    def __init__(self):
        super().__init__()
        self.params = self.config.get_strategy_config('ml_moving_average_crossover')

    def run(self):
        print("--- [INFO] Running Training Pipeline for: ML Moving Average Crossover ---")
        
        # 1. Load Data
        print(f"[INFO] Loading data from: {self.config.data_path}")
        full_df = self.data_manager.load_ohlc_data(str(self.config.data_path))
        if full_df.empty:
            print("[CRITICAL] Data could not be loaded. Aborting pipeline.")
            return

        # 2. Split Data
        split_ratio = self.config.trading_params['train_test_split_ratio']
        train_df, _ = self.data_manager.split_data(full_df, split_ratio)
        
        # 3. Generate and Label Signals
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        signals_df = _generate_signals(train_df, short_window, long_window)
        
        tp_mult = self.params['risk']['take_profit_multiplier']
        sl_mult = self.params['risk']['stop_loss_multiplier']
        labeled_signals = _label_trades(signals_df, train_df, tp_mult, sl_mult)

        # 4. Create Features
        features_df = self.feature_engineering.create_features(train_df)
        model_data = labeled_signals.join(features_df, how='inner').dropna()

        # 5. ML Training
        X = model_data[['returns', 'volatility', 'rsi']]
        y = model_data['target']
        
        if X.empty:
            print("[CRITICAL] No data available for training.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print(f"\n[INFO] Training on {len(X_train)} signals, testing on {len(X_test)}.")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        print("\n--- [INFO] Model Evaluation on Hold-Out Test Set ---")
        print(classification_report(y_test, model.predict(X_test)))
        
        # 6. Save Model
        model_path = self.config.get_model_path('ml_moving_average_crossover')
        joblib.dump(model, model_path)
        print(f"\n[INFO] Model successfully trained and saved to: {model_path}")