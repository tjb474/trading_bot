# trading_bot/training_pipelines/open_range_breakout_pipeline.py

from pyexpat import model
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from .base_pipeline import BasePipeline
from ml.feature_engineering import create_features # <-- Import from new location
from common.data_manager import convert_to_eastern_time

# Get logger
logger = logging.getLogger(__name__)

# The helper functions _create_breakout_direction_feature, _generate_signals, and _label_trades can now be simplified or removed.
# For simplicity, we will keep _generate_signals and _label_trades but they will now operate on the pre-computed 'breakout_direction' feature.

def _generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate breakout signals from the pre-computed 'breakout_direction' feature."""
    logger.info("Extracting breakout signals from 'breakout_direction' feature...")
    
    # A signal is the first moment the direction changes from 0
    breakout_mask = (df['breakout_direction'] != 0) & (df['breakout_direction'].shift(1) == 0)
    
    if not breakout_mask.any():
        logger.warning("No breakout signals found in the data.")
        return pd.DataFrame()
    
    signals = df.loc[breakout_mask, ['breakout_direction']].copy()
    signals.rename(columns={'breakout_direction': 'direction'}, inplace=True)
    
    bullish_count = (signals['direction'] == 1).sum()
    bearish_count = (signals['direction'] == -1).sum()
    logger.info(f"Generated {len(signals)} breakout signals ({bullish_count} bullish, {bearish_count} bearish).")
    return signals


def _label_trades(signals: pd.DataFrame, df: pd.DataFrame, tp_mult: float, sl_mult: float, range_end_time: str) -> pd.DataFrame:
    """Label trades based on whether they hit take profit or stop loss first."""
    logger.info("Labeling trades as a win or loss...")
    labels = []
    
    for signal_time, row in signals.iterrows():
        day_data = df[df.index.date == signal_time.date()]
        range_end_ts = pd.to_datetime(f"{signal_time.date()} {range_end_time}")
        range_data = day_data.loc[:range_end_ts]
        
        range_size = range_data['high'].max() - range_data['low'].min()
        entry_price = df.loc[signal_time, 'close']
        direction = row['direction']
        
        if direction == 1:
            tp_price = entry_price + (range_size * tp_mult)
            sl_price = entry_price - (range_size * sl_mult)
        else:
            tp_price = entry_price - (range_size * tp_mult)
            sl_price = entry_price + (range_size * sl_mult)
        
        future_data = df.loc[signal_time:].iloc[1:]
        if future_data.empty:
            labels.append(0)
            continue
            
        hit_tp = future_data['high'] >= tp_price if direction == 1 else future_data['low'] <= tp_price
        hit_sl = future_data['low'] <= sl_price if direction == 1 else future_data['high'] >= sl_price
        
        first_tp = hit_tp.idxmax() if hit_tp.any() else pd.Timestamp.max
        first_sl = hit_sl.idxmax() if hit_sl.any() else pd.Timestamp.max
        
        if first_tp < first_sl:
            labels.append(1)
        else:
            labels.append(0)

    signals['target'] = labels
    logger.info(f"Target Label Distribution:\n{signals['target'].value_counts(normalize=True)}")
    return signals


class OpenRangeBreakoutPipeline(BasePipeline):
    """Pipeline for training the ML-enhanced Open Range Breakout model."""
    def __init__(self):
        super().__init__()
        self.params = self.config.get_strategy_config('ml_open_range_breakout')

    def run(self):
        logger.info("--- Running Training Pipeline for: ML Open Range Breakout ---")
        
        # 1. Load and Prepare Data
        full_df = self.data_manager.load_ohlc_data(str(self.config.data_path))
        full_df = convert_to_eastern_time(full_df)
        full_df.index = full_df.index.tz_localize(None)
        
        train_df, _ = self.data_manager.split_data(full_df, self.config.trading_params['train_test_split_ratio'])

        # 2. Create Features using the new system
        feature_config = self.params['features']
        feature_list = feature_config['feature_list']
        
        # Gather all parameters for features
        feature_params = {
            'volatility_window': feature_config.get('volatility_window'),
            'rsi_window': feature_config.get('rsi_window'),
            'range_start': self.params['range']['start'],
            'range_end': self.params['range']['end']
        }
        
        # The `create_features` function now handles dependency resolution and creation
        train_df_features = create_features(train_df, feature_list, **feature_params)
        
        # 3. Generate and Label Signals
        signals_df = _generate_signals(train_df_features)
        
        labeled_signals = _label_trades(
            signals_df, 
            train_df_features, 
            tp_mult=self.params['risk']['take_profit_multiplier'], 
            sl_mult=self.params['risk']['stop_loss_multiplier'],
            range_end_time=self.params['range']['end']
        )

        # 4. Prepare Data for Model Training
        expanded_feature_list = []
        for feature in feature_list:
            if feature == 'day_of_week':
                expanded_feature_list.extend(['is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday'])
            else:
                expanded_feature_list.append(feature)
        
        model_data = labeled_signals.join(train_df_features[expanded_feature_list], how='inner').dropna()

        # 5. ML Training
        X = model_data[expanded_feature_list]
        y = model_data['target']
        
        if X.empty:
            logger.critical("No data available for training after feature join. Aborting.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced')
        model.fit(X_train_res, y_train_res)
        
        logger.info(f"--- Model Evaluation ---\n{classification_report(y_test, model.predict(X_test))}")
        
        # 6. Save Model
        model_path = self.config.get_model_path('ml_open_range_breakout')
        joblib.dump(model, model_path)
        logger.info(f"Model successfully trained and saved to: {model_path}")