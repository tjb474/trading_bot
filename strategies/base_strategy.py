# strategies/base_strategy.py
import backtrader as bt

class BaseStrategy(bt.Strategy):
    """..."""
    def __init__(self):
        """..."""
        self.order = None
        self.trade_count = 0

    def next(self):
        """..."""
        raise NotImplementedError("The 'next' method must be implemented by the subclass.")

    def log(self, txt, dt=None):
        """
        Improved logging function for strategies.
        """
        dt = dt or self.datas[0].datetime.date(0)
        # Get the simple name of the class (e.g., 'OpenRangeBreakout')
        strategy_name = self.__class__.__name__
        print(f'{dt.isoformat()} | {strategy_name} | {txt}')

    def notify_order(self, order):
        """Log order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            # No action needed for these statuses
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.getstatusname()}')

        # Reset order status
        self.order = None