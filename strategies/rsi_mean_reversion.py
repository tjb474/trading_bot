import backtrader as bt
from .base_strategy import BaseStrategy

class RsiMeanReversion(BaseStrategy):
    params = (
        ('rsi_period', 14),
        ('oversold', 30),
        ('overbought', 70),
    )

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=self.params.rsi_period, safediv=True)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.rsi < self.params.oversold:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                self.order = self.buy()
                self.trade_count += 1
        else:
            if self.rsi > self.params.overbought:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()