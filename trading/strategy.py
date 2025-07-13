# trading/strategy.py

import backtrader as bt
import numpy as np
import joblib

class MLStrategy(bt.Strategy):
    params = (
        ('config', None),
    )

    def __init__(self):
        if not self.p.config:
            raise ValueError("Config not provided to strategy")
        cfg = self.p.config
        
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=cfg.SHORT_WINDOW)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=cfg.LONG_WINDOW)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
        self.returns = bt.indicators.PercentChange(self.data.close, period=1)
        self.volatility = bt.indicators.StandardDeviation(self.returns, period=cfg.FEATURE_VOL_WINDOW)
        
        # --- FIX: Make RSI calculation safe ---
        # The `safediv=True` parameter prevents division-by-zero errors.
        # If the average loss is zero, it will return `safehigh` (100.0) if there were gains,
        # or `safelow` (50.0) if there were no price changes.
        self.rsi = bt.indicators.RSI_SMA(
            self.data.close,
            period=cfg.FEATURE_RSI_WINDOW,
            safediv=True
        )
        # --- END FIX ---
        
        try:
            self.model = joblib.load(cfg.MODEL_FILE_PATH)
            print(f"Successfully loaded model from {cfg.MODEL_FILE_PATH}")
        except FileNotFoundError:
            print(f"CRITICAL: Model file not found at {cfg.MODEL_FILE_PATH}. Cannot run strategy.")
            self.model = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()}, {txt}')

    def next(self):
        cfg = self.p.config
        if len(self.data) < max(cfg.LONG_WINDOW, cfg.FEATURE_VOL_WINDOW) or not self.model:
            return

        if not self.position:
            if self.crossover[0] > 0:
                features = np.array([self.returns[0], self.volatility[0], self.rsi[0]]).reshape(1, -1)
                prob = self.model.predict_proba(features)[0][1]
                self.log(f"BUY Signal. Close: {self.data.close[0]:.2f}, ML Prob: {prob:.2f}")
                if prob > cfg.PROBABILITY_THRESHOLD:
                    self.log(f"ML Filter PASSED. Placing BUY order.")
                    self.buy()
        elif self.crossover[0] < 0:
            self.log(f"SELL Signal. Closing position at {self.data.close[0]:.2f}")
            self.close()