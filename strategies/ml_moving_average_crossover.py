import backtrader as bt
import numpy as np
import joblib
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

        # Add the parameter that the backtester is trying to pass in.
        # Give it a default value of None.
        # This tells backtrader: "It's okay to receive a keyword argument 
        # named feature_list. I know what it is. Please store its value in self.p.feature_list."
        ('feature_list', None),
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

    def next(self):
        # The logic here doesn't need to change, as it now uses self.p
        if len(self.data) < self.p.long_window or not self.model:
            return

        if not self.position:
            if self.crossover[0] > 0:
                features = np.array([self.returns[0], self.volatility[0], self.rsi[0]]).reshape(1, -1)
                prob = self.model.predict_proba(features)[0][1]
                self.log(f"BUY Signal. Close: {self.data.close[0]:.2f}, ML Prob: {prob:.2f}")
                if prob > self.p.probability_threshold:
                    self.log(f"ML Filter PASSED. Placing BUY order.")
                    self.buy()
        elif self.crossover[0] < 0:
            self.log(f"SELL Signal. Closing position at {self.data.close[0]:.2f}")
            self.close()