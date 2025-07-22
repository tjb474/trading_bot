import backtrader as bt
from .base_strategy import BaseStrategy
import datetime
import numpy as np
import pandas as pd
import joblib

class MLOpenRangeBreakout(BaseStrategy):
    params = (
        ('range_start', '09:30:00'),
        ('range_end', '10:15:00'),
        ('take_profit_multiplier', 1.0),
        ('stop_loss_multiplier', 0.5),
        ('volatility_window', 20),
        ('rsi_window', 14),
        ('atr_window', 14),
        ('lookback_days', 20),
        ('model_file_path', None),
        ('probability_threshold', 0.70),
        ('feature_list', None),  # Will be populated from config
        ('feature_data', None),  # DataFrame with all pre-calculated features
    )

    def __init__(self):
        super().__init__()
        # Convert string times to datetime.time objects
        self.p.range_start = datetime.datetime.strptime(self.p.range_start, '%H:%M:%S').time()
        self.p.range_end = datetime.datetime.strptime(self.p.range_end, '%H:%M:%S').time()
        
        # Create feature indicators
        self.returns = bt.indicators.PercentChange(self.data.close, period=1)
        self.volatility = bt.indicators.StandardDeviation(self.returns, period=self.p.volatility_window)
        self.rsi = bt.indicators.RSI_SMA(
            self.data.close,
            period=self.p.rsi_window,
            safediv=True
        )
        
        # Initialize daily feature cache for NR4/NR7 and day-of-week
        self._daily_features_cache = {}
        self._last_calculated_date = None
        
        # Daily state variables
        self.reset_daily_vars()
        
        # Load ML model
        try:
            self.model = joblib.load(self.p.model_file_path)
            print(f"Successfully loaded model from {self.p.model_file_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def reset_daily_vars(self):
        """Resets the state at the start of each new trading day."""
        self.opening_range_high = 0
        self.opening_range_low = float('inf')
        self.range_calculated_today = False
        self.trade_taken_today = False
        self.breakout_direction = 0  # Track breakout direction like pipeline

    def _calculate_daily_features(self):
        """
        Calculate daily features including NR4/NR7 and day-of-week.
        This matches the feature engineering pipeline's approach exactly.
        """
        current_date = self.data.datetime.date(0)
        
        # Only recalculate if we haven't done so today
        if current_date == self._last_calculated_date:
            return
            
        self._last_calculated_date = current_date
        
        # Calculate day of week features (avoiding dummy trap like pipeline)
        weekday = self.data.datetime.datetime(0).weekday()  # 0=Monday, 4=Friday
        day_features = {
            'is_monday': 1 if weekday == 0 else 0,
            'is_tuesday': 1 if weekday == 1 else 0,
            'is_wednesday': 1 if weekday == 2 else 0,
            'is_thursday': 1 if weekday == 3 else 0,
            # Friday omitted to avoid dummy trap (when all others are 0, it's Friday)
        }
        
        # Get historical data for NR calculations
        min_bars_needed = 7 * 390  # Approximate bars per day
        
        if len(self.data) < min_bars_needed:
            nr_features = {'is_nr4': 0, 'is_nr7': 0}
        else:
            nr_features = self._calculate_nr_features(current_date)
        
        # Combine all daily features
        self._daily_features_cache[current_date] = {**day_features, **nr_features}

    def _calculate_nr_features(self, current_date):
        """Calculate NR4/NR7 features using the same logic as the pipeline."""
    def _calculate_nr_features(self, current_date):
        """Calculate NR4/NR7 features using the same logic as the pipeline."""
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
            return {'is_nr4': 0, 'is_nr7': 0}
            
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
            return {'is_nr4': 0, 'is_nr7': 0}
            
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
            
        return {'is_nr4': nr4_signal, 'is_nr7': nr7_signal}

    def next(self):
        current_time = self.data.datetime.time()
        
        # --- Daily Reset Logic ---
        if self.data.datetime.date(0) != self.data.datetime.date(-1):
            self.reset_daily_vars()
            # Log timezone and trading day info for debugging
            current_dt = self.data.datetime.datetime(0)
            self.log(f"NEW TRADING DAY: {current_dt.date()} | Timezone: {current_dt.tzinfo} | Current time: {current_time}")
            self.log(f"OR window configured: {self.p.range_start} to {self.p.range_end}")

        # Calculate daily features once per day - MATCHES PIPELINE
        self._calculate_daily_features()

        # We can't trade if already in a position or if a trade was already taken today
        if self.position or self.trade_taken_today:
            return

        # --- Phase 1: Calculate Opening Range ---
        if current_time >= self.p.range_start and current_time < self.p.range_end:
            # Log the first few bars of range calculation for debugging
            if self.opening_range_high == 0:  # First bar in range
                self.log(f"OR Calculation STARTED at {current_time} - First bar: O={self.data.open[0]:.2f}, H={self.data.high[0]:.2f}, L={self.data.low[0]:.2f}, C={self.data.close[0]:.2f}")
            
            # Update range with current bar
            prev_high = self.opening_range_high
            prev_low = self.opening_range_low
            self.opening_range_high = max(self.data.high[0], self.opening_range_high)
            self.opening_range_low = min(self.data.low[0], self.opening_range_low)
            
            # Log when range values change significantly
            if abs(self.opening_range_high - prev_high) > 0.5 or abs(self.opening_range_low - prev_low) > 0.5:
                self.log(f"OR Update at {current_time}: H={self.opening_range_high:.2f} (was {prev_high:.2f}), L={self.opening_range_low:.2f} (was {prev_low:.2f})")
            
            return # Don't do anything else while in the range calculation window
        
        # --- Mark range as calculated once we are past the window ---
        if current_time >= self.p.range_end and not self.range_calculated_today:
            self.range_calculated_today = True
            
            # Log the end of range calculation
            self.log(f"OR Calculation ENDED at {current_time} (range was {self.p.range_start} to {self.p.range_end})")
            self.log(f"Current bar when range ended: O={self.data.open[0]:.2f}, H={self.data.high[0]:.2f}, L={self.data.low[0]:.2f}, C={self.data.close[0]:.2f}")
            
            if self.opening_range_high == 0 or self.opening_range_low == float('inf'):
                self.log("Opening range could not be calculated.")
                self.trade_taken_today = True # Prevent further trades today
                return
            
            # Calculate range size for additional context
            range_size = self.opening_range_high - self.opening_range_low
            range_midpoint = (self.opening_range_high + self.opening_range_low) / 2
            
            self.log(f"Opening Range Calculated: High={self.opening_range_high:.2f}, Low={self.opening_range_low:.2f}")
            self.log(f"OR Details: Size=${range_size:.2f}, Midpoint=${range_midpoint:.2f}, Current=${self.data.close[0]:.2f}")
            
            # Check if current price is already outside the range
            if self.data.close[0] > self.opening_range_high:
                self.log(f"WARNING: Price already above OR high when range calculation ended!")
            elif self.data.close[0] < self.opening_range_low:
                self.log(f"WARNING: Price already below OR low when range calculation ended!")

        # --- Phase 2: Breakout Detection (MATCHES PIPELINE LOGIC) ---
        if self.range_calculated_today and self.breakout_direction == 0:
            # Check for bullish breakout
            if self.data.close[0] > self.opening_range_high:
                self.breakout_direction = 1  # Bullish
                breakout_distance = self.data.close[0] - self.opening_range_high
                self.log(f"BULLISH breakout detected at {self.data.close[0]:.2f} (OR High: {self.opening_range_high:.2f}, Distance: +${breakout_distance:.2f})")
                self._handle_breakout_signal()
            # Check for bearish breakout
            elif self.data.close[0] < self.opening_range_low:
                self.breakout_direction = -1  # Bearish
                breakout_distance = self.opening_range_low - self.data.close[0]
                self.log(f"BEARISH breakout detected at {self.data.close[0]:.2f} (OR Low: {self.opening_range_low:.2f}, Distance: -${breakout_distance:.2f})")
                self._handle_breakout_signal()

    def _handle_breakout_signal(self):
        """Handle breakout signal using ML model - MATCHES PIPELINE LOGIC"""
        if not self.model:
            self.log("No model loaded, skipping ML filter")
            return
            
        # Get features for ML prediction - MATCHES PIPELINE
        features = self._get_feature_vector()
        
        if features is None:
            self.log("Could not get feature vector, skipping trade")
            return
            
        # Get ML probability
        prob = self.model.predict_proba(features)[0][1]
        self.log(f"ML probability for breakout success: {prob:.2f}")
        
        if prob > self.p.probability_threshold:
            self._execute_trade()
        else:
            self.log(f"ML Filter FAILED (prob={prob:.2f} < {self.p.probability_threshold:.2f}). Skipping trade.")
            self.trade_taken_today = True  # Prevent further signals today

    def _get_feature_vector(self):
        """Get feature vector for ML model - USES PRE-CALCULATED FEATURES FROM DATAFRAME"""
        try:
            # Get current timestamp
            current_datetime = self.data.datetime.datetime(0)
            
            # If we have the feature DataFrame, use it directly
            if self.p.feature_data is not None:
                try:
                    # Find the row with the current timestamp
                    self.log(f"Looking for timestamp: {current_datetime} (type: {type(current_datetime)})")
                    self.log(f"DataFrame index type: {type(self.p.feature_data.index[0])}")
                    self.log(f"Sample index values: {self.p.feature_data.index[:3].tolist()}")
                    
                    # Try exact match first
                    if current_datetime in self.p.feature_data.index:
                        current_row = self.p.feature_data.loc[current_datetime]
                    else:
                        # Try to find the closest timestamp (allowing for slight differences)
                        try:
                            # Convert current_datetime to pandas Timestamp with UTC timezone
                            current_ts = pd.Timestamp(current_datetime, tz='UTC')
                            if current_ts in self.p.feature_data.index:
                                current_row = self.p.feature_data.loc[current_ts]
                            else:
                                # Find the nearest timestamp
                                nearest_idx = self.p.feature_data.index.get_indexer([current_ts], method='nearest')[0]
                                if nearest_idx >= 0:
                                    current_row = self.p.feature_data.iloc[nearest_idx]
                                    self.log(f"Using nearest timestamp: {self.p.feature_data.index[nearest_idx]}")
                                else:
                                    raise KeyError("No suitable timestamp found")
                        except Exception as e:
                            self.log(f"Timestamp matching failed: {e}")
                            raise KeyError(f"Timestamp {current_datetime} not found")
                    
                    # Extract features in the same order as the training pipeline
                    feature_values = []
                    for feature in self.p.feature_list:
                        if feature == 'day_of_week':
                            # day_of_week creates 4 separate binary features
                            feature_values.extend([
                                current_row.get('is_monday', 0),
                                current_row.get('is_tuesday', 0),
                                current_row.get('is_wednesday', 0),
                                current_row.get('is_thursday', 0)
                            ])
                        else:
                            feature_values.append(current_row.get(feature, 0.0))
                    
                    feature_array = np.array(feature_values).reshape(1, -1)
                    self.log(f"Feature vector from DataFrame: shape={feature_array.shape}, count={len(feature_values)}")
                    return feature_array
                        
                except Exception as e:
                    self.log(f"Error accessing feature DataFrame: {e}")
            
            # Fallback to manual feature calculation
            self.log("Using fallback feature calculation")
            feature_values = []
            current_date = self.data.datetime.date(0)
            daily_features = self._daily_features_cache.get(current_date, {})
            
            for feature in self.p.feature_list:
                if feature == 'returns':
                    feature_values.append(self.returns[0])
                elif feature == 'volatility':
                    feature_values.append(self.volatility[0])
                elif feature == 'rsi':
                    feature_values.append(self.rsi[0])
                elif feature == 'breakout_direction':
                    feature_values.append(self.breakout_direction)
                elif feature == 'day_of_week':
                    feature_values.extend([
                        daily_features.get('is_monday', 0),
                        daily_features.get('is_tuesday', 0),
                        daily_features.get('is_wednesday', 0),
                        daily_features.get('is_thursday', 0)
                    ])
                elif feature in ['is_nr4', 'is_nr7']:
                    feature_values.append(daily_features.get(feature, 0))
                else:
                    # For OR features and others, use 0 as placeholder
                    feature_values.append(0.0)
            
            feature_array = np.array(feature_values).reshape(1, -1)
            self.log(f"Feature vector from fallback: shape={feature_array.shape}, count={len(feature_values)}")
            return feature_array
            
        except Exception as e:
            self.log(f"Error getting feature vector: {e}")
            return None

    def _execute_trade(self):
        """Execute the trade based on breakout direction - MATCHES PIPELINE LOGIC"""
        range_size = self.opening_range_high - self.opening_range_low
        entry_price = self.data.close[0]
        
        if self.breakout_direction == 1:  # Bullish breakout (long trade)
            tp_price = entry_price + (range_size * self.p.take_profit_multiplier)
            sl_price = entry_price - (range_size * self.p.stop_loss_multiplier)
            
            self.log(f"ML Filter PASSED. BUY BRACKET @ {entry_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
            
            # Use a bracket order to set take profit and stop loss automatically
            self.buy_bracket(
                price=entry_price,
                limitprice=tp_price,
                stopprice=sl_price
            )
            
        elif self.breakout_direction == -1:  # Bearish breakout (short trade)
            tp_price = entry_price - (range_size * self.p.take_profit_multiplier)
            sl_price = entry_price + (range_size * self.p.stop_loss_multiplier)
            
            self.log(f"ML Filter PASSED. SELL BRACKET @ {entry_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
            
            # Use a bracket order to set take profit and stop loss automatically
            self.sell_bracket(
                price=entry_price,
                limitprice=tp_price,
                stopprice=sl_price
            )
        
        self.trade_taken_today = True  # Only one trade per day

    def get_daily_feature(self, feature_name):
        """Get daily feature value - HELPER METHOD"""
        current_date = self.data.datetime.date(0)
        return self._daily_features_cache.get(current_date, {}).get(feature_name, 0)