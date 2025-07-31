# strategies/base_strategy.py
import backtrader as bt
from typing import Optional, Dict, Any

class BaseStrategy(bt.Strategy):
    """Enhanced base strategy with trade tracking for visualization and reporting."""
    def __init__(self):
        """Initialize strategy with trade tracking."""
        self.order = None
        self.trade_count = 0
        self.trades_history = []  # Store trade information for visualization
        self.current_trade = None  # Track current open trade
        self.trade_reporter = None  # Will be set by backtester
        
        # Enhanced trade tracking for detailed reporting
        self.orb_data = {}  # Store opening range data for current day
        self.daily_reset_done = False

    def set_trade_reporter(self, reporter):
        """Set the trade reporter for detailed trade tracking."""
        self.trade_reporter = reporter
        
    def set_orb_data(self, orb_high: float, orb_low: float, date: str = None):
        """Set opening range breakout data for the current day."""
        current_date = date or self.data.datetime.date(0).isoformat()
        self.orb_data[current_date] = {
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_size': orb_high - orb_low
        }
        
    def get_current_orb_data(self) -> Dict:
        """Get ORB data for current date."""
        current_date = self.data.datetime.date(0).isoformat()
        return self.orb_data.get(current_date, {})

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
        """Enhanced order notification with trade tracking and reporting."""
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
                        'commission': order.executed.comm,
                        'take_profit': getattr(self, 'current_tp', None),
                        'stop_loss': getattr(self, 'current_sl', None),
                        'breakout_direction': getattr(self, 'breakout_direction', 'UNKNOWN'),
                        'ml_probability': getattr(self, 'last_ml_probability', None),
                        **self.get_current_orb_data()
                    }
                else:
                    # Closing existing trade (must be short)
                    self._finalize_trade(order, 'SHORT')
                    
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
                        'commission': order.executed.comm,
                        'take_profit': getattr(self, 'current_tp', None),
                        'stop_loss': getattr(self, 'current_sl', None),
                        'breakout_direction': getattr(self, 'breakout_direction', 'UNKNOWN'),
                        'ml_probability': getattr(self, 'last_ml_probability', None),
                        **self.get_current_orb_data()
                    }
                else:
                    # Closing existing trade (must be long)
                    self._finalize_trade(order, 'LONG')
            
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.getstatusname()}')

        # Reset order status
        self.order = None
        
    def _finalize_trade(self, order, closed_direction: str):
        """Finalize a completed trade and add to history/reporter."""
        self.current_trade['exit_time'] = self.data.datetime.datetime(0)
        self.current_trade['exit_price'] = order.executed.price
        self.current_trade['commission'] += order.executed.comm
        
        # Determine exit reason based on price levels
        exit_reason = self._determine_exit_reason(order.executed.price)
        self.current_trade['exit_reason'] = exit_reason
        
        # Calculate PnL
        if closed_direction == 'LONG':
            # Closing long trade with sell order
            pnl = (order.executed.price - self.current_trade['entry_price']) * self.current_trade['size']
        else:
            # Closing short trade with buy order
            pnl = (self.current_trade['entry_price'] - order.executed.price) * abs(self.current_trade['size'])
            
        self.current_trade['pnl'] = pnl - self.current_trade['commission']
        
        # Add additional metadata
        self.current_trade['additional_info'] = {
            'bars_in_trade': len(self) - getattr(self, 'entry_bar', 0),
            'closed_direction': closed_direction,
            'strategy_name': self.__class__.__name__
        }
        
        # Add to history
        self.trades_history.append(self.current_trade.copy())
        
        # Log trade summary for immediate feedback
        direction_str = "LONG" if self.current_trade['direction'] == 1 else "SHORT"
        pnl_str = f"+${self.current_trade['pnl']:.2f}" if self.current_trade['pnl'] > 0 else f"${self.current_trade['pnl']:.2f}"
        self.log(f"TRADE COMPLETE: {direction_str} | P&L: {pnl_str} | Exit: {self.current_trade['exit_reason']}")
        
        # Add to trade reporter if available
        if self.trade_reporter:
            self.trade_reporter.add_trade(self.current_trade.copy())
            
        # Reset current trade
        self.current_trade = None
        
    def _determine_exit_reason(self, exit_price: float) -> str:
        """Determine the reason for trade exit based on price levels."""
        if not self.current_trade:
            return 'UNKNOWN'
            
        tp = self.current_trade.get('take_profit')
        sl = self.current_trade.get('stop_loss')
        direction = self.current_trade.get('direction', 0)
        
        if direction == 1:  # Long trade
            if tp and abs(exit_price - tp) < abs(exit_price - sl if sl else float('inf')):
                return 'TP'
            elif sl and exit_price <= sl:
                return 'SL'
        elif direction == -1:  # Short trade
            if tp and abs(exit_price - tp) < abs(exit_price - sl if sl else float('inf')):
                return 'TP'
            elif sl and exit_price >= sl:
                return 'SL'
                
        return 'OTHER'  # Could be EOD, manual exit, etc.
        
    def notify_trade(self, trade):
        """Enhanced trade notification."""
        if not trade.isclosed:
            return
            
        self.log(f'TRADE CLOSED: PnL=${trade.pnl:.2f}, Commission=${trade.commission:.2f}')
        self.trade_count += 1
        
    def get_trades_history(self):
        """Return the complete trades history for visualization."""
        return self.trades_history.copy()