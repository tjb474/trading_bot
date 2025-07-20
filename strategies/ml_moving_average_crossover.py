import backtrader as bt
import numpy as np
import pandas as pd
import joblib
import datetime
from .base_strategy import BaseStrategy # Inherit from your base class

class MLMovingAverageCrossover(BaseStrategy):
    # 1. Define the parameters the strategy needs, with default values.
    #    backtrader will populate these from the config.
    params = (
        ('short_window', 40),
        ('long_window', 100),
        ('feature_vol_window', 20),
        ('feature_rsi_window', 14),
        ('model_file_path', None),
        ('probability_threshold', 0.60),
        ('feature_list', None),  # List of features to use
    )

    def __init__(self):
        super().__init__() # Call the base class initializer

        # 2. Use `self.p` (or self.params) to access parameters
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.short_window)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.long_window)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
        self.returns = bt.indicators.PercentChange(self.data.close, period=1)
        self.volatility = bt.indicators.StandardDeviation(self.returns, period=self.p.feature_vol_window)
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=self.p.feature_rsi_window, safediv=True)
        
        # Initialize daily feature cache for NR4/NR7 if needed
        self._daily_features_cache = {}
        self._last_calculated_date = None
        
        # 3. Load the model using the path from the parameters
        if not self.p.model_file_path:
             raise ValueError("Model file path not provided in strategy params")
        try:
            # The config object is not needed here anymore
            self.model = joblib.load(self.p.model_file_path)
            self.log(f"Successfully loaded model from {self.p.model_file_path}")
        except FileNotFoundError:
            self.log(f"CRITICAL: Model file not found at {self.p.model_file_path}")
            self.model = None

    def _calculate_daily_features(self):
        """
        Calculate daily NR4/NR7 features using the same logic as the pipeline.
        This matches the feature engineering pipeline's approach exactly.
        """
        current_date = self.data.datetime.date(0)
        
        # Only recalculate if we haven't done so today
        if current_date == self._last_calculated_date:
            return
            
        self._last_calculated_date = current_date
        
        # Get historical data up to current bar
        # We need at least 7 days of data for NR7 calculation
        min_bars_needed = 7 * 390  # Approximate bars per day
        
        if len(self.data) < min_bars_needed:
            self._daily_features_cache[current_date] = {'is_nr4': 0, 'is_nr7': 0}
            return
            
        # Create DataFrame from recent data
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        
        # Get data for the last 10 days to be safe
        lookback_bars = min(len(self.data), 10 * 390)
        
        for i in range(-lookback_bars + 1, 1):
            try:
                dates.append(self.data.datetime.datetime(-i))
                opens.append(self.data.open[-i])
                highs.append(self.data.high[-i])
                lows.append(self.data.low[-i])
                closes.append(self.data.close[-i])
            except IndexError:
                break
        
        if len(dates) == 0:
            self._daily_features_cache[current_date] = {'is_nr4': 0, 'is_nr7': 0}
            return
            
        # Create DataFrame and resample to daily - SAME AS PIPELINE
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes
        }, index=pd.DatetimeIndex(dates))
        
        # Resample to daily OHLC - SAME AS PIPELINE
        daily_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(daily_df) < 7:
            self._daily_features_cache[current_date] = {'is_nr4': 0, 'is_nr7': 0}
            return
            
        # Calculate NR4 - SAME AS PIPELINE
        daily_df['range'] = daily_df['high'] - daily_df['low']
        daily_df['min_range_4d'] = daily_df['range'].rolling(window=4).min()
        daily_df['is_nr4_day'] = np.where(daily_df['range'] == daily_df['min_range_4d'], 1, 0)
        daily_df['is_nr4_signal_for_today'] = daily_df['is_nr4_day'].shift(1)
        
        # Calculate NR7 - SAME AS PIPELINE  
        daily_df['min_range_7d'] = daily_df['range'].rolling(window=7).min()
        daily_df['is_nr7_day'] = np.where(daily_df['range'] == daily_df['min_range_7d'], 1, 0)
        daily_df['is_nr7_signal_for_today'] = daily_df['is_nr7_day'].shift(1)
        
        # Get today's signals - SAME AS PIPELINE
        today_data = daily_df[daily_df.index.date == current_date]
        
        if len(today_data) > 0:
            nr4_signal = today_data['is_nr4_signal_for_today'].iloc[-1]
            nr7_signal = today_data['is_nr7_signal_for_today'].iloc[-1]
            
            # Handle NaN values
            nr4_signal = 0 if pd.isna(nr4_signal) else int(nr4_signal)
            nr7_signal = 0 if pd.isna(nr7_signal) else int(nr7_signal)
        else:
            nr4_signal = 0
            nr7_signal = 0
            
        self._daily_features_cache[current_date] = {
            'is_nr4': nr4_signal,
            'is_nr7': nr7_signal
        }

    def is_nr4(self):
        """Get NR4 signal using the same calculation as the pipeline"""
        current_date = self.data.datetime.date(0)
        return self._daily_features_cache.get(current_date, {}).get('is_nr4', 0)

    def is_nr7(self):
        """Get NR7 signal using the same calculation as the pipeline"""
        current_date = self.data.datetime.date(0)
        return self._daily_features_cache.get(current_date, {}).get('is_nr7', 0)

    def next(self):
        # The logic here doesn't need to change, as it now uses self.p
        if len(self.data) < self.p.long_window or not self.model:
            return

        if not self.position:
            if self.crossover[0] > 0:
                # Calculate daily features if needed (for nr4/nr7)
                if self.p.feature_list and ('is_nr4' in self.p.feature_list or 'is_nr7' in self.p.feature_list):
                    self._calculate_daily_features()
                
                # Build feature vector dynamically based on feature_list
                features = []
                if self.p.feature_list:
                    for feature in self.p.feature_list:
                        if feature == 'returns':
                            features.append(self.returns[0])
                        elif feature == 'volatility':
                            features.append(self.volatility[0])
                        elif feature == 'rsi':
                            features.append(self.rsi[0])
                        elif feature == 'is_nr4':
                            features.append(self.is_nr4())
                        elif feature == 'is_nr7':
                            features.append(self.is_nr7())
                else:
                    # Default features if feature_list is not provided
                    features = [self.returns[0], self.volatility[0], self.rsi[0]]
                
                features = np.array(features).reshape(1, -1)
                prob = self.model.predict_proba(features)[0][1]
                self.log(f"BUY Signal. Close: {self.data.close[0]:.2f}, ML Prob: {prob:.2f}")
                if prob > self.p.probability_threshold:
                    self.log(f"ML Filter PASSED. Placing BUY order.")
                    self.buy()
        elif self.crossover[0] < 0:
            self.log(f"SELL Signal. Closing position at {self.data.close[0]:.2f}")
            self.close()