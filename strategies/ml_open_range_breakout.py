import backtrader as bt
from .base_strategy import BaseStrategy
import datetime
import numpy as np
import joblib

class MLOpenRangeBreakout(BaseStrategy):
    params = (
        ('range_start', '09:30:00'),
        ('range_end', '10:00:00'),
        ('take_profit_multiplier', 1.0),
        ('stop_loss_multiplier', 0.5),
        ('feature_vol_window', 20),
        ('feature_rsi_window', 14),
        ('model_file_path', None),
        ('probability_threshold', 0.70),
        ('feature_list', None),  # Will be populated from config
    )

    def __init__(self):
        super().__init__()
        # Convert string times to datetime.time objects
        self.p.range_start = datetime.datetime.strptime(self.p.range_start, '%H:%M:%S').time()
        self.p.range_end = datetime.datetime.strptime(self.p.range_end, '%H:%M:%S').time()
        
        # Create feature indicators
        self.returns = bt.indicators.PercentChange(self.data.close, period=1)
        self.volatility = bt.indicators.StandardDeviation(self.returns, period=self.p.feature_vol_window)
        self.rsi = bt.indicators.RSI_SMA(
            self.data.close,
            period=self.p.feature_rsi_window,
            safediv=True
        )
        
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

    def next(self):
        current_time = self.data.datetime.time()
        
        # --- Daily Reset Logic ---
        if self.data.datetime.date(0) != self.data.datetime.date(-1):
            self.reset_daily_vars()

        # We can't trade if already in a position or if a trade was already taken today
        if self.position or self.trade_taken_today:
            return

        # --- Phase 1: Calculate Opening Range ---
        if current_time >= self.p.range_start and current_time < self.p.range_end:
            self.opening_range_high = max(self.data.high[0], self.opening_range_high)
            self.opening_range_low = min(self.data.low[0], self.opening_range_low)
            return # Don't do anything else while in the range calculation window
        
        # --- Mark range as calculated once we are past the window ---
        if current_time >= self.p.range_end and not self.range_calculated_today:
            self.range_calculated_today = True
            
            if self.opening_range_high == 0 or self.opening_range_low == float('inf'):
                self.log("Opening range could not be calculated.")
                self.trade_taken_today = True # Prevent further trades today
                return
            self.log(f"Opening Range Calculated: High={self.opening_range_high:.2f}, Low={self.opening_range_low:.2f}")

        # --- Phase 2: Trading Logic (after range is set) ---
        if self.range_calculated_today:
            # Entry condition: price breaks above the opening range high
            if self.data.close[0] > self.opening_range_high and self.model:
                # Get features for ML prediction
                features = []
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
                
                features = np.array(features).reshape(1, -1)
                prob = self.model.predict_proba(features)[0][1]
                self.log(f"Breakout detected. ML probability: {prob:.2f}")
                
                if prob > self.p.probability_threshold:
                    range_size = self.opening_range_high - self.opening_range_low
                    tp_price = self.data.close[0] + (range_size * self.p.take_profit_multiplier)
                    sl_price = self.data.close[0] - (range_size * self.p.stop_loss_multiplier)
                    
                    self.log(f"ML Filter PASSED. BUY BRACKET @ {self.data.close[0]:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
                    
                    # Use a bracket order to set take profit and stop loss automatically
                    self.buy_bracket(
                        price=self.data.close[0],
                        limitprice=tp_price,
                        stopprice=sl_price
                    )
                    self.trade_taken_today = True # Only one trade per day
                else:
                    self.log(f"ML Filter FAILED. Skipping trade.")

    def is_nr4(self):
        """Check if current bar's range is the narrowest in the last 4 bars"""
        ranges = []
        for i in range(4):
            ranges.append(self.data.high[-i] - self.data.low[-i])
        current_range = ranges[0]
        return int(all(current_range <= r for r in ranges[1:]))

    def is_nr7(self):
        """Check if current bar's range is the narrowest in the last 7 bars"""
        ranges = []
        for i in range(7):
            ranges.append(self.data.high[-i] - self.data.low[-i])
        current_range = ranges[0]
        return int(all(current_range <= r for r in ranges[1:]))