import backtrader as bt

class BaseStrategy(bt.Strategy):
    """
    An abstract base class for all trading strategies.
    It defines the common interface for all strategies.
    """
    def __init__(self):
        """
        Common initialization for all strategies.
        """
        self.order = None
        self.trade_count = 0

    def next(self):
        """
        This method will be implemented by each concrete strategy.
        """
        raise NotImplementedError("The 'next' method must be implemented by the subclass.")

    def log(self, txt, dt=None):
        """
        Logging function for this strategy.
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')