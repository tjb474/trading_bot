# strategies/base_strategy.py
import backtrader as bt

class BaseStrategy(bt.Strategy):
    """Enhanced base strategy with trade tracking for visualization."""
    def __init__(self):
        """Initialize strategy with trade tracking."""
        self.order = None
        self.trade_count = 0
        self.trades_history = []  # Store trade information for visualization
        self.current_trade = None  # Track current open trade

    def next(self):
        """Must be implemented by subclass."""
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
        """Enhanced order notification with trade tracking."""
        if order.status in [order.Submitted, order.Accepted]:
            # No action needed for these statuses
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                # Could be opening long or closing short
                if self.current_trade is None:
                    # Opening long trade
                    self.current_trade = {
                        'entry_time': self.data.datetime.datetime(0),
                        'entry_price': order.executed.price,
                        'direction': 1,  # Long
                        'size': order.executed.size,
                        'commission': order.executed.comm
                    }
                else:
                    # Closing existing trade (must be short)
                    self.current_trade['exit_time'] = self.data.datetime.datetime(0)
                    self.current_trade['exit_price'] = order.executed.price
                    self.current_trade['commission'] += order.executed.comm
                    
                    # Calculate PnL for short trade
                    pnl = (self.current_trade['entry_price'] - order.executed.price) * abs(self.current_trade['size'])
                    self.current_trade['pnl'] = pnl - self.current_trade['commission']
                    
                    # Add to history and reset
                    self.trades_history.append(self.current_trade.copy())
                    self.current_trade = None
                    
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                # Could be opening short or closing long
                if self.current_trade is None:
                    # Opening short trade
                    self.current_trade = {
                        'entry_time': self.data.datetime.datetime(0),
                        'entry_price': order.executed.price,
                        'direction': -1,  # Short
                        'size': order.executed.size,
                        'commission': order.executed.comm
                    }
                else:
                    # Closing existing trade (must be long)
                    self.current_trade['exit_time'] = self.data.datetime.datetime(0)
                    self.current_trade['exit_price'] = order.executed.price
                    self.current_trade['commission'] += order.executed.comm
                    
                    # Calculate PnL for long trade
                    pnl = (order.executed.price - self.current_trade['entry_price']) * self.current_trade['size']
                    self.current_trade['pnl'] = pnl - self.current_trade['commission']
                    
                    # Add to history and reset
                    self.trades_history.append(self.current_trade.copy())
                    self.current_trade = None
            
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.getstatusname()}')

        # Reset order status
        self.order = None
        
    def notify_trade(self, trade):
        """Enhanced trade notification."""
        if not trade.isclosed:
            return
            
        self.log(f'TRADE CLOSED: PnL=${trade.pnl:.2f}, Commission=${trade.commission:.2f}')
        self.trade_count += 1
        
    def get_trades_history(self):
        """Return the complete trades history for visualization."""
        return self.trades_history.copy()