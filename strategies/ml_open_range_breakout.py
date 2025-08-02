# trading_bot\strategies\ml_open_range_breakout.py
import backtrader as bt
from .base_strategy import BaseStrategy # Assuming this is your base class
import datetime
import numpy as np
import pandas as pd
import joblib

class MLOpenRangeBreakout(BaseStrategy):
    params = (
        ('range_start', '09:30:00'),
        ('range_end', '10:15:00'),
        ('daily_close_time', '16:00:00'),
        ('take_profit_multiplier', 1.0),
        ('stop_loss_multiplier', 0.5),
        ('volatility_window', 20),
        ('rsi_window', 14),
        ('atr_window', 14),
        ('lookback_days', 20),
        ('model_file_path', None),
        ('probability_threshold', 0.70),
        ('use_ml_filter', True),
        ('feature_list', None),
        ('feature_data', None),
        ('entry_type', 'conservative'),  # 'aggressive' or 'conservative'
    )

    def __init__(self):
        super().__init__()
        # Convert string times to datetime.time objects
        self.p.range_start = datetime.datetime.strptime(self.p.range_start, '%H:%M:%S').time()
        self.p.range_end = datetime.datetime.strptime(self.p.range_end, '%H:%M:%S').time()
        self.p.daily_close_time = datetime.datetime.strptime(self.p.daily_close_time, '%H:%M:%S').time()

        # Create feature indicators
        self.returns = bt.indicators.PercentChange(self.data.close, period=1)
        self.volatility = bt.indicators.StandardDeviation(self.returns, period=self.p.volatility_window)
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=self.p.rsi_window, safediv=True)

        # State Variables
        self._last_date = None
        self._daily_features_cache = {}
        self.active_bracket_orders = None
        self.reset_daily_vars()

        # Load ML model
        self.model = None
        if self.p.use_ml_filter:
            try:
                self.model = joblib.load(self.p.model_file_path)
                self.log(f"Successfully loaded model from {self.p.model_file_path}")
            except Exception as e:
                self.log(f"Error loading model: {e}. ML filter will be skipped.")
                self.model = None
        else:
            self.log("Meta labeling (ML filter) is turned OFF by configuration.")

    def reset_daily_vars(self):
        """Resets the state at the start of each new trading day."""
        self.opening_range_high = 0.0
        self.opening_range_low = float('inf')
        self.range_calculated_today = False
        self.trade_taken_today = False
        self.breakout_direction = 0
        self.active_bracket_orders = None
        self.aggressive_breakout_detected = False  # Track aggressive breakout pending execution

    def next(self):
        current_dt = self.data.datetime.datetime(0)
        current_time = current_dt.time()
        current_date = current_dt.date()

        # --- Daily Reset Logic ---
        if self._last_date != current_date:
            self.log(f"[{current_time}] NEW TRADING DAY: {current_date}")
            self.reset_daily_vars()
            self._last_date = current_date

        # --- Position Management ---
        if self.position:
            if current_time >= self.p.daily_close_time:
                self.log(f"[{current_time}] Daily close time reached. Closing position.")

                # --- FIX: Treat self.active_bracket_orders as a LIST ---
                if self.active_bracket_orders:
                    self.log(f"[{current_time}] Canceling pending bracket orders (TP/SL) before EOD close.")
                    # The list contains [entry, limit, stop]. We cancel limit and stop.
                    self.broker.cancel(self.active_bracket_orders[1]) # Cancel TP
                    self.broker.cancel(self.active_bracket_orders[2]) # Cancel SL
                    self.active_bracket_orders = None

                self.close()
            return

        # --- Trade Entry Logic ---
        if self.trade_taken_today: return
        if current_time < self.p.range_start or current_time >= self.p.daily_close_time: return

        # Phase 1: Calculate Opening Range
        if current_time < self.p.range_end:
            if self.opening_range_high == 0.0: self.log(f"[{current_time}] OR Calculation STARTED.")
            self.opening_range_high = max(self.data.high[0], self.opening_range_high)
            self.opening_range_low = min(self.data.low[0], self.opening_range_low)
            return

        # Mark range as calculated
        if not self.range_calculated_today:
            self.range_calculated_today = True
            if self.opening_range_high == 0.0 or self.opening_range_low == float('inf'):
                self.log(f"[{current_time}] ERROR: Opening range not calculated. No trades today.")
                self.trade_taken_today = True
                return
            self.log(f"[{current_time}] OR Calculation ENDED. High={self.opening_range_high:.2f}, Low={self.opening_range_low:.2f}")

        # Phase 2: Detect Breakout and Handle Entry Types
        if self.breakout_direction == 0:
            if self.p.entry_type == 'aggressive':
                # Aggressive: Detect breakout when high/low breaches OR, enter on next bar
                current_high = self.data.high[0]
                current_low = self.data.low[0]
                
                if current_high > self.opening_range_high:
                    self.breakout_direction = 1
                    self.aggressive_breakout_detected = True
                    self.log(f"[{current_time}] AGGRESSIVE BULLISH breakout detected - high {current_high:.2f} > OR high {self.opening_range_high:.2f}")
                    self.log(f"[{current_time}] Will enter on next bar")
                elif current_low < self.opening_range_low:
                    self.breakout_direction = -1
                    self.aggressive_breakout_detected = True
                    self.log(f"[{current_time}] AGGRESSIVE BEARISH breakout detected - low {current_low:.2f} < OR low {self.opening_range_low:.2f}")
                    self.log(f"[{current_time}] Will enter on next bar")
                    
            else:  # conservative
                # Conservative: Detect breakout when candle close is outside OR, enter on next bar
                current_price = self.data.close[0]
                if current_price > self.opening_range_high:
                    self.breakout_direction = 1
                    self.aggressive_breakout_detected = True  # Reuse same flag for consistent behavior
                    self.log(f"[{current_time}] CONSERVATIVE BULLISH breakout detected - close {current_price:.2f} > OR high {self.opening_range_high:.2f}")
                    self.log(f"[{current_time}] Will enter on next bar")
                elif current_price < self.opening_range_low:
                    self.breakout_direction = -1
                    self.aggressive_breakout_detected = True  # Reuse same flag for consistent behavior
                    self.log(f"[{current_time}] CONSERVATIVE BEARISH breakout detected - close {current_price:.2f} < OR low {self.opening_range_low:.2f}")
                    self.log(f"[{current_time}] Will enter on next bar")
                    
        elif self.aggressive_breakout_detected and not self.trade_taken_today:
            # Execute the trade on the next bar after breakout detection (both aggressive and conservative)
            self.log(f"[{current_time}] Executing {self.p.entry_type.upper()} entry on next bar after breakout")
            self._handle_breakout_signal()

    def _handle_breakout_signal(self):
        self.trade_taken_today = True
        self._execute_trade() # Simplified for this fix

    def _execute_trade(self):
        current_time = self.data.datetime.time()
        
        # Both aggressive and conservative entries now execute on the bar after detection
        # For realistic backtesting, we use the current bar's open price (simulating next bar entry)
        entry_price = self.data.open[0]
        self.log(f"[{current_time}] Using {self.p.entry_type.upper()} entry at next bar open: {entry_price:.2f}")
            
        range_size = self.opening_range_high - self.opening_range_low
        if range_size <= 0:
            self.log(f"[{current_time}] Invalid range size ({range_size:.2f}). Skipping trade.")
            return

        # Store ORB data for trade tracking (needed for base strategy)
        self.set_orb_data(self.opening_range_high, self.opening_range_low)

        # self.active_bracket_orders will now correctly be a LIST of 3 orders
        if self.breakout_direction == 1:
            tp_price = entry_price + (range_size * self.p.take_profit_multiplier)
            sl_price = entry_price - (range_size * self.p.stop_loss_multiplier)
            
            # Store for trade tracking
            self.current_tp = tp_price
            self.current_sl = sl_price
            self.breakout_direction_label = 'BULLISH'
            
            self.log(f"[{current_time}] Submitting BUY BRACKET ({self.p.entry_type.upper()}): Entry={entry_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
            self.active_bracket_orders = self.buy_bracket(limitprice=tp_price, stopprice=sl_price)
        elif self.breakout_direction == -1:
            tp_price = entry_price - (range_size * self.p.take_profit_multiplier)
            sl_price = entry_price + (range_size * self.p.stop_loss_multiplier)
            
            # Store for trade tracking
            self.current_tp = tp_price
            self.current_sl = sl_price
            self.breakout_direction_label = 'BEARISH'
            
            self.log(f"[{current_time}] Submitting SELL BRACKET ({self.p.entry_type.upper()}): Entry={entry_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
            self.active_bracket_orders = self.sell_bracket(limitprice=tp_price, stopprice=sl_price)

    def notify_order(self, order):
        current_time = self.data.datetime.time()
        
        # Call base strategy's notify_order for proper trade tracking
        super().notify_order(order)
        
        if order.status in [order.Completed]:
            # --- FIX: Check if the completed order is IN the list of active orders ---
            if self.active_bracket_orders and order in self.active_bracket_orders:
                 # If a TP/SL is hit, the trade is over. Clear the stored orders.
                 is_tp_or_sl = (order == self.active_bracket_orders[1] or order == self.active_bracket_orders[2])
                 if is_tp_or_sl:
                    self.log(f"[{current_time}] A TP/SL bracket order was hit. Clearing active orders.")
                    self.active_bracket_orders = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # --- FIX: Check if the canceled order is IN the list ---
            if self.active_bracket_orders and order in self.active_bracket_orders:
                 self.log(f'[{current_time}] An active bracket order was canceled. Clearing.')
                 self.active_bracket_orders = None

    def notify_trade(self, trade):
        # Call base strategy's notify_trade for proper trade tracking
        super().notify_trade(trade)
        
        if not trade.isclosed:
            return
        current_time = self.data.datetime.time()
        self.active_bracket_orders = None # Final cleanup

    def _calculate_daily_features(self):
        current_date = self.data.datetime.date(0)
        if hasattr(self, '_last_feature_date') and self._last_feature_date == current_date:
            return
        self._last_feature_date = current_date
        weekday = self.data.datetime.datetime(0).weekday()
        day_features = {'is_monday': 1 if weekday == 0 else 0, 'is_tuesday': 1 if weekday == 1 else 0, 'is_wednesday': 1 if weekday == 2 else 0, 'is_thursday': 1 if weekday == 3 else 0}
        min_bars_needed = 7 * 390
        nr_features = self._calculate_nr_features(current_date) if len(self.data) >= min_bars_needed else {'is_nr4': 0, 'is_nr7': 0}
        self._daily_features_cache[current_date] = {**day_features, **nr_features}

    def _calculate_nr_features(self, current_date):
        dates, opens, highs, lows, closes = [], [], [], [], []
        lookback_bars = min(len(self.data), 10 * 390)
        for i in range(-lookback_bars + 1, 1):
            try:
                dates.append(self.data.datetime.datetime(-i))
                opens.append(self.data.open[-i])
                highs.append(self.data.high[-i])
                lows.append(self.data.low[-i])
                closes.append(self.data.close[-i])
            except IndexError: break
        if not dates: return {'is_nr4': 0, 'is_nr7': 0}
        df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes}, index=pd.DatetimeIndex(dates))
        daily_df = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily_df) < 7: return {'is_nr4': 0, 'is_nr7': 0}
        daily_df['range'] = daily_df['high'] - daily_df['low']
        daily_df['min_range_4d'] = daily_df['range'].rolling(window=4).min()
        daily_df['is_nr4_day'] = np.where(daily_df['range'] == daily_df['min_range_4d'], 1, 0)
        daily_df['is_nr4_signal_for_today'] = daily_df['is_nr4_day'].shift(1)
        daily_df['min_range_7d'] = daily_df['range'].rolling(window=7).min()
        daily_df['is_nr7_day'] = np.where(daily_df['range'] == daily_df['min_range_7d'], 1, 0)
        daily_df['is_nr7_signal_for_today'] = daily_df['is_nr7_day'].shift(1)
        today_data = daily_df[daily_df.index.date == current_date]
        if not today_data.empty:
            nr4_signal = 0 if pd.isna(today_data['is_nr4_signal_for_today'].iloc[-1]) else int(today_data['is_nr4_signal_for_today'].iloc[-1])
            nr7_signal = 0 if pd.isna(today_data['is_nr7_signal_for_today'].iloc[-1]) else int(today_data['is_nr7_signal_for_today'].iloc[-1])
        else: nr4_signal, nr7_signal = 0, 0
        return {'is_nr4': nr4_signal, 'is_nr7': nr7_signal}

    def _get_feature_vector(self):
        try:
            self._calculate_daily_features()
            current_date = self.data.datetime.date(0)
            daily_features = self._daily_features_cache.get(current_date, {})
            feature_values = []
            for feature in self.p.feature_list:
                if feature == 'returns': feature_values.append(self.returns[0])
                elif feature == 'volatility': feature_values.append(self.volatility[0])
                elif feature == 'rsi': feature_values.append(self.rsi[0])
                elif feature == 'breakout_direction': feature_values.append(self.breakout_direction)
                elif feature == 'day_of_week':
                    feature_values.extend([daily_features.get('is_monday', 0), daily_features.get('is_tuesday', 0), daily_features.get('is_wednesday', 0), daily_features.get('is_thursday', 0)])
                elif feature in ['is_nr4', 'is_nr7']: feature_values.append(daily_features.get(feature, 0))
                else: feature_values.append(0.0)
            return np.array(feature_values).reshape(1, -1)
        except Exception as e:
            self.log(f"Error getting feature vector: {e}")
            return None