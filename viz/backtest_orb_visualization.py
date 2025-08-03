# viz/backtest_orb_visualization.py

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import logging
import numpy as np
from datetime import datetime, time as dt_time
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class BacktestORBVisualizer:
    """
    Enhanced visualization for Opening Range Breakout backtests.
    Shows ORB rectangles, breakout signals, and trade entry/exit points.
    """
    
    def __init__(self, 
                 range_start: str = "09:30:00", 
                 range_end: str = "10:30:00",
                 timeframe: str = "1min"):
        self.range_start = range_start
        self.range_end = range_end
        self.timeframe = timeframe
        self.trades_data = None  # Store trades data for summary calculations
        
    def create_enhanced_backtest_chart(self, 
                                     price_data: pd.DataFrame,
                                     trades_data: List[Dict] = None,
                                     start_date: str = None,
                                     end_date: str = None,
                                     save_path: str = None) -> None:
        """
        Create comprehensive ORB backtest visualization with:
        - OHLC candlestick chart
        - ORB rectangles for each trading day
        - Breakout signals
        - Trade entry/exit markers
        - Performance metrics overlay
        
        Args:
            price_data: DataFrame with OHLC data and features
            trades_data: List of trade dictionaries with entry/exit info
            start_date: Start date for visualization (YYYY-MM-DD)
            end_date: End date for visualization (YYYY-MM-DD)
            save_path: Optional path to save the chart
        """
        try:
            # Store trades data for summary calculations
            self.trades_data = trades_data
            
            # Filter data by date range if specified
            if start_date and end_date:
                plot_df = price_data.loc[start_date:end_date].copy()
                title_suffix = f"({start_date} to {end_date})"
            else:
                plot_df = price_data.copy()
                title_suffix = "(Full Dataset)"
            
            if plot_df.empty:
                logger.error("No data available for the specified date range")
                return
                
            # Resample to specified timeframe if needed
            if self.timeframe != "1min":
                plot_df = self._resample_data(plot_df)
                
            logger.info(f"Creating ORB visualization for {len(plot_df)} bars")
            
            # Calculate ORB rectangles and signals
            orb_rectangles = self._calculate_orb_rectangles(plot_df)
            breakout_signals = self._extract_breakout_signals(plot_df)
            
            # Process trades data
            trade_markers = self._process_trades_data(trades_data, plot_df) if trades_data else {}
            
            # Create the visualization
            self._create_plot(plot_df, orb_rectangles, breakout_signals, trade_markers, title_suffix, save_path)
            
        except Exception as e:
            logger.error(f"Error creating backtest visualization: {e}")
            raise
            
    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample OHLCV data to specified timeframe with proper market hour alignment."""
        # Standard OHLCV aggregation
        agg_dict = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add aggregation for other columns based on their data type
        for col in df.columns:
            if col not in agg_dict:
                # Check column data type
                if df[col].dtype in ['object', 'string']:
                    agg_dict[col] = 'first'  # Take first value for string columns
                elif col in ['breakout_direction', 'is_nr4', 'is_nr7', 'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday']:
                    agg_dict[col] = 'last'  # Keep last value for categorical features
                elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    agg_dict[col] = 'mean'  # Average for continuous numeric features
                else:
                    agg_dict[col] = 'last'  # Default to last value for unknown types
        
        # CRITICAL FIX: Use proper origin for market hour alignment
        # This ensures 5-minute candles align with market hours (9:30, 9:35, 9:40, 9:45, etc.)
        # instead of arbitrary 5-minute intervals that might include post-OR bars
        
        # Create market open origin with proper timezone matching
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            # Use a date from the data to ensure timezone compatibility
            sample_date = df.index[0].date()
            market_open_origin = pd.Timestamp.combine(sample_date, pd.Timestamp('09:30:00').time()).tz_localize(df.index.tz)
        else:
            market_open_origin = '09:30:00'
        
        return df.resample(
            self.timeframe, 
            origin=market_open_origin,  # Align with market open time (timezone-aware)
            closed='left',              # Left-closed intervals (9:30:00 to 9:34:59)
            label='left'                # Label with the left boundary (9:30 for 9:30-9:35 candle)
        ).agg(agg_dict).dropna()
    
    def _calculate_orb_rectangles(self, df: pd.DataFrame) -> List[Dict]:
        """Calculate ORB rectangle coordinates for each trading day."""
        rectangles = []
        
        # Group by trading day
        for date, day_data in df.groupby(df.index.date):
            try:
                # Create time range for this day
                range_start_time = datetime.combine(date, datetime.strptime(self.range_start, "%H:%M:%S").time())
                range_end_time = datetime.combine(date, datetime.strptime(self.range_end, "%H:%M:%S").time())
                
                # Localize if data has timezone
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    range_start_time = pd.Timestamp(range_start_time).tz_localize(df.index.tz)
                    range_end_time = pd.Timestamp(range_end_time).tz_localize(df.index.tz)
                else:
                    range_start_time = pd.Timestamp(range_start_time)
                    range_end_time = pd.Timestamp(range_end_time)
                
                # Get opening range data
                range_mask = (day_data.index >= range_start_time) & (day_data.index <= range_end_time)
                range_data = day_data[range_mask]
                
                if len(range_data) > 0:
                    or_high = range_data['high'].max()
                    or_low = range_data['low'].min()
                    
                    rectangles.append({
                        'date': date,
                        'start_time': range_start_time,
                        'end_time': range_end_time,
                        'high': or_high,
                        'low': or_low,
                        'range_size': or_high - or_low
                    })
                    
            except Exception as e:
                logger.warning(f"Could not calculate ORB for {date}: {e}")
                continue
                
        logger.info(f"Calculated {len(rectangles)} ORB rectangles")
        return rectangles
    
    def _extract_breakout_signals(self, df: pd.DataFrame) -> Dict:
        """Extract breakout signals from the data."""
        signals = {
            'bullish': pd.Series(index=df.index, dtype=float),
            'bearish': pd.Series(index=df.index, dtype=float)
        }
        
        if 'breakout_direction' in df.columns:
            # Find breakout moments (transition from 0 to 1 or -1)
            breakout_mask = (df['breakout_direction'] != 0) & (df['breakout_direction'].shift(1) == 0)
            breakout_points = df[breakout_mask]
            
            for idx in breakout_points.index:
                direction = df.loc[idx, 'breakout_direction']
                price = df.loc[idx, 'close']
                
                if direction == 1:  # Bullish
                    signals['bullish'].loc[idx] = price * 1.002  # Slightly above for visibility
                elif direction == -1:  # Bearish
                    signals['bearish'].loc[idx] = price * 0.998  # Slightly below for visibility
                    
        return signals
    
    def _process_trades_data(self, trades_data: List[Dict], df: pd.DataFrame) -> Dict:
        """Process trade data into visualization markers."""
        markers = {
            'entries_long': pd.Series(index=df.index, dtype=float),
            'entries_short': pd.Series(index=df.index, dtype=float), 
            'exits_profit': pd.Series(index=df.index, dtype=float),
            'exits_loss': pd.Series(index=df.index, dtype=float),
            'exits_eod': pd.Series(index=df.index, dtype=float)  # New EOD exit marker
        }
        
        # Store trades data for summary calculations and SL/TP lines
        self.trades_data = trades_data
        
        for trade in trades_data:
            try:
                # Parse trade timestamps
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time']) if trade.get('exit_time') else None
                
                # Match timezone if needed
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    if entry_time.tz is None:
                        entry_time = entry_time.tz_localize(df.index.tz)
                    if exit_time and exit_time.tz is None:
                        exit_time = exit_time.tz_localize(df.index.tz)
                
                # Find closest data points
                entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
                if entry_idx >= 0:
                    entry_idx = df.index[entry_idx]
                    entry_price = trade['entry_price']
                    
                    # Mark entry
                    if trade['direction'] > 0:  # Long
                        markers['entries_long'].loc[entry_idx] = entry_price * 0.995  # Below candle
                    else:  # Short
                        markers['entries_short'].loc[entry_idx] = entry_price * 1.005  # Above candle
                    
                    # Mark exit if available
                    if exit_time and trade.get('exit_price'):
                        exit_idx = df.index.get_indexer([exit_time], method='nearest')[0]
                        if exit_idx >= 0:
                            exit_idx = df.index[exit_idx]
                            exit_price = trade['exit_price']
                            
                            # Use actual exit_reason from trade data if available
                            exit_reason = trade.get('exit_reason', 'UNKNOWN')
                            
                            if exit_reason == 'TP':
                                # Take Profit exit
                                markers['exits_profit'].loc[exit_idx] = exit_price * 1.002
                            elif exit_reason == 'SL':
                                # Stop Loss exit  
                                markers['exits_loss'].loc[exit_idx] = exit_price * 0.998
                            elif exit_reason == 'OTHER':
                                # EOD or other manual close
                                markers['exits_eod'].loc[exit_idx] = exit_price * 1.001
                            else:
                                # Fallback to price-based determination for unknown exit reasons
                                is_profit = ((trade['direction'] > 0 and exit_price > entry_price) or 
                                           (trade['direction'] < 0 and exit_price < entry_price))
                                
                                if is_profit:
                                    markers['exits_profit'].loc[exit_idx] = exit_price * 1.002
                                else:
                                    markers['exits_loss'].loc[exit_idx] = exit_price * 0.998
                                
            except Exception as e:
                logger.warning(f"Could not process trade: {e}")
                continue
                
        return markers

    def _add_sl_tp_lines(self, ax, df: pd.DataFrame, trades_data: List[Dict]) -> None:
        """Add stop loss and take profit lines for each trade spanning only the trading day."""
        if not trades_data:
            return
            
        tp_line_added = False
        sl_line_added = False
            
        for trade in trades_data:
            try:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time']) if trade.get('exit_time') else None
                
                # Match timezone if needed
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    if entry_time.tz is None:
                        entry_time = entry_time.tz_localize(df.index.tz)
                    if exit_time and exit_time.tz is None:
                        exit_time = exit_time.tz_localize(df.index.tz)
                
                # Get the trading day
                trade_date = entry_time.date()
                
                # Define day boundaries (market hours)
                day_start = pd.Timestamp.combine(trade_date, pd.Timestamp("09:30:00").time())
                day_end = pd.Timestamp.combine(trade_date, pd.Timestamp("16:00:00").time())
                
                # Localize if data has timezone
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    day_start = day_start.tz_localize(df.index.tz)
                    day_end = day_end.tz_localize(df.index.tz)
                
                # Find start and end positions in the data
                entry_pos = df.index.get_indexer([entry_time], method='nearest')[0]
                if exit_time:
                    end_time = min(exit_time, day_end)
                    end_pos = df.index.get_indexer([end_time], method='nearest')[0]
                else:
                    end_pos = df.index.get_indexer([day_end], method='nearest')[0]
                
                if entry_pos >= 0 and end_pos >= 0 and end_pos > entry_pos:
                    # Get SL/TP levels
                    take_profit = trade.get('take_profit')
                    stop_loss = trade.get('stop_loss')
                    
                    # Draw take profit line (green)
                    if take_profit:
                        ax.plot([entry_pos, end_pos], [take_profit, take_profit], 
                               color='green', linestyle='--', linewidth=1.5, alpha=0.6,
                               label='Take Profit' if not tp_line_added else "")
                        tp_line_added = True
                    
                    # Draw stop loss line (red) 
                    if stop_loss:
                        ax.plot([entry_pos, end_pos], [stop_loss, stop_loss],
                               color='red', linestyle='--', linewidth=1.5, alpha=0.6,
                               label='Stop Loss' if not sl_line_added else "")
                        sl_line_added = True
                        
            except Exception as e:
                logger.warning(f"Could not add SL/TP lines for trade: {e}")
                continue

    def _add_entry_price_lines(self, ax, df: pd.DataFrame, trades_data: List[Dict]) -> None:
        """Add entry price lines for each trade spanning only the trading day."""
        if not trades_data:
            return
            
        entry_line_added = False
            
        for trade in trades_data:
            try:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time']) if trade.get('exit_time') else None
                entry_price = trade['entry_price']
                
                # Match timezone if needed
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    if entry_time.tz is None:
                        entry_time = entry_time.tz_localize(df.index.tz)
                    if exit_time and exit_time.tz is None:
                        exit_time = exit_time.tz_localize(df.index.tz)
                
                # Get the trading day
                trade_date = entry_time.date()
                
                # Define day boundaries (market hours)
                day_start = pd.Timestamp.combine(trade_date, pd.Timestamp("09:30:00").time())
                day_end = pd.Timestamp.combine(trade_date, pd.Timestamp("16:00:00").time())
                
                # Localize if data has timezone
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    day_start = day_start.tz_localize(df.index.tz)
                    day_end = day_end.tz_localize(df.index.tz)
                
                # Find start and end positions in the data
                entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
                if entry_idx < 0:
                    continue
                    
                if exit_time:
                    end_time = min(exit_time, day_end)
                    end_idx = df.index.get_indexer([end_time], method='nearest')[0]
                else:
                    end_idx = df.index.get_indexer([day_end], method='nearest')[0]
                
                if end_idx < 0 or end_idx <= entry_idx:
                    continue
                
                # Convert to mplfinance x-coordinates (sequential positions, not index values)
                entry_pos = entry_idx
                end_pos = end_idx
                
                # Draw entry price line (blue)
                ax.plot([entry_pos, end_pos], [entry_price, entry_price],
                       color='blue', linestyle='-', linewidth=2, alpha=0.7,
                       label='Entry Price' if not entry_line_added else "")
                entry_line_added = True
                        
            except Exception as e:
                logger.warning(f"Could not add entry price line for trade: {e}")
                continue
    
    def _create_plot(self, df: pd.DataFrame, orb_rectangles: List[Dict], 
                    breakout_signals: Dict, trade_markers: Dict, 
                    title_suffix: str, save_path: str = None) -> None:
        """Create the comprehensive visualization plot."""
        
        # Prepare addplot objects
        ap = []
        
        # Add breakout signals
        if breakout_signals['bullish'].notna().any():
            ap.append(mpf.make_addplot(breakout_signals['bullish'], type='scatter', 
                                     marker='^', markersize=60, color='blue', alpha=0.8))
        
        if breakout_signals['bearish'].notna().any():
            ap.append(mpf.make_addplot(breakout_signals['bearish'], type='scatter',
                                     marker='v', markersize=60, color='red', alpha=0.8))
        
        # Add trade markers
        if trade_markers:
            if trade_markers['entries_long'].notna().any():
                ap.append(mpf.make_addplot(trade_markers['entries_long'], type='scatter',
                                         marker='o', markersize=100, color='green', alpha=0.9))
            
            if trade_markers['entries_short'].notna().any():
                ap.append(mpf.make_addplot(trade_markers['entries_short'], type='scatter',
                                         marker='o', markersize=100, color='orange', alpha=0.9))
                                         
            if trade_markers['exits_profit'].notna().any():
                ap.append(mpf.make_addplot(trade_markers['exits_profit'], type='scatter',
                                         marker='X', markersize=80, color='darkgreen', alpha=0.9))
                                         
            if trade_markers['exits_loss'].notna().any():
                ap.append(mpf.make_addplot(trade_markers['exits_loss'], type='scatter',
                                         marker='X', markersize=80, color='darkred', alpha=0.9))
                                         
            if trade_markers['exits_eod'].notna().any():
                ap.append(mpf.make_addplot(trade_markers['exits_eod'], type='scatter',
                                         marker='D', markersize=60, color='orange', alpha=0.9))
        
        # Create title
        title = (f'Enhanced ORB Backtest Analysis {title_suffix}\n'
                f'Range: {self.range_start} to {self.range_end} | '
                f'Blue ↑ = Bullish Breakouts, Red ↓ = Bearish Breakouts\n'
                f'Green ● = Long Entry, Orange ● = Short Entry | '
                f'Green ✕ = TP Exit, Red ✕ = SL Exit, Orange ◆ = EOD Exit\n'
                f'Blue — = Entry Price, Green -- = Take Profit, Red -- = Stop Loss (per trading day)')
        
        # Create figure to add rectangles
        if ap:
            # When using addplot, don't use external axes to avoid mplfinance conflicts
            kwargs = {
                'type': 'candle',
                'style': 'charles',
                'title': title,
                'ylabel': 'Price ($)',
                'volume': True,
                'returnfig': True,
                'warn_too_much_data': 100000,
                'addplot': ap
            }
            fig, axes = mpf.plot(df, **kwargs)
        else:
            # When no addplot, use external axes to add rectangles manually
            fig, axes = plt.subplots(2, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [4, 1]})
            
            kwargs = {
                'type': 'candle',
                'style': 'charles',
                'title': title,
                'ylabel': 'Price ($)',
                'volume': axes[1],
                'ax': axes[0],
                'returnfig': True,
                'warn_too_much_data': 100000,
            }
            fig, axes = mpf.plot(df, **kwargs)
        
        # Add ORB rectangles
        self._add_orb_rectangles(axes[0], orb_rectangles, df.index)
        
        # Add SL/TP lines and entry price lines if we have trades data
        if hasattr(self, 'trades_data') and self.trades_data:
            self._add_sl_tp_lines(axes[0], df, self.trades_data)
            self._add_entry_price_lines(axes[0], df, self.trades_data)
            
        # Add EOD close time lines
        self._add_eod_close_lines(axes[0], df)
        
        # Add legend
        self._add_legend(axes[0])
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to: {save_path}")
        
        plt.show()
        
        # Print summary
        self._print_summary(orb_rectangles, breakout_signals, trade_markers)
    
    def _add_orb_rectangles(self, ax, rectangles: List[Dict], time_index: pd.DatetimeIndex) -> None:
        """Add ORB rectangles to the plot."""
        for rect in rectangles:
            try:
                # Convert times to numeric values for matplotlib
                start_num = pd.Timestamp(rect['start_time'])
                end_num = pd.Timestamp(rect['end_time'])
                
                # Find position in the time index
                start_pos = time_index.get_indexer([start_num], method='nearest')[0]
                end_pos = time_index.get_indexer([end_num], method='nearest')[0]
                
                if start_pos >= 0 and end_pos >= 0:
                    width = end_pos - start_pos
                    height = rect['high'] - rect['low']
                    
                    rectangle = Rectangle(
                        (start_pos, rect['low']),
                        width, height,
                        linewidth=2,
                        edgecolor='purple',
                        facecolor='purple',
                        alpha=0.15,
                        label='Opening Range' if rect == rectangles[0] else ""
                    )
                    ax.add_patch(rectangle)
                    
                    # Add range size annotation
                    mid_time = start_pos + width/2
                    mid_price = rect['low'] + height/2
                    ax.annotate(f"${rect['range_size']:.2f}", 
                              xy=(mid_time, mid_price),
                              fontsize=8, ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                              
            except Exception as e:
                logger.warning(f"Could not add rectangle for {rect['date']}: {e}")
                continue

    def _add_eod_close_lines(self, ax, df: pd.DataFrame) -> None:
        """Add vertical lines at 4:00 PM (EOD close time) for each trading day."""
        try:
            eod_line_added = False
            
            # Group by trading day
            for date, day_data in df.groupby(df.index.date):
                # Define EOD close time (4:00 PM)
                eod_time = pd.Timestamp.combine(date, pd.Timestamp("16:00:00").time())
                
                # Localize if data has timezone
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    eod_time = eod_time.tz_localize(df.index.tz)
                
                # Find the closest data point to 4:00 PM
                day_mask = day_data.index.date == date
                day_data_filtered = df[day_mask]
                
                if len(day_data_filtered) > 0:
                    # Find closest timestamp to 4:00 PM
                    closest_idx = day_data_filtered.index.get_indexer([eod_time], method='nearest')[0]
                    if closest_idx >= 0:
                        eod_position = day_data_filtered.index[closest_idx]
                        
                        # Convert to numeric position for plotting
                        numeric_position = df.index.get_loc(eod_position)
                        
                        # Draw vertical line at 4:00 PM
                        ax.axvline(x=numeric_position, color='gray', linestyle=':', alpha=0.6, linewidth=1,
                                  label='EOD Close (4:00 PM)' if not eod_line_added else "")
                        eod_line_added = True
                        
        except Exception as e:
            logger.warning(f"Could not add EOD close lines: {e}")
    
    def _add_legend(self, ax) -> None:
        """Add comprehensive legend to the plot."""
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='purple', alpha=0.15, edgecolor='purple', label='Opening Range'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Bullish Breakout'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Bearish Breakout'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='Long Entry'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='Short Entry'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkgreen', markersize=10, label='TP Exit'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred', markersize=10, label='SL Exit'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markersize=8, label='EOD Exit'),
            plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label='Entry Price'),
            plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Take Profit'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Stop Loss'),
            plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1, label='EOD Close (4:00 PM)'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    def _print_summary(self, rectangles: List[Dict], breakout_signals: Dict, trade_markers: Dict) -> None:
        """Print summary statistics."""
        logger.info(f"\n--- ORB Backtest Visualization Summary ---")
        logger.info(f"Total Trading Days: {len(rectangles)}")
        
        bullish_breakouts = breakout_signals['bullish'].notna().sum()
        bearish_breakouts = breakout_signals['bearish'].notna().sum()
        total_breakouts = bullish_breakouts + bearish_breakouts
        
        logger.info(f"Total Breakouts: {total_breakouts} ({bullish_breakouts} bullish, {bearish_breakouts} bearish)")
        
        if trade_markers:
            long_entries = trade_markers['entries_long'].notna().sum() 
            short_entries = trade_markers['entries_short'].notna().sum()
            profit_exits = trade_markers['exits_profit'].notna().sum()
            loss_exits = trade_markers['exits_loss'].notna().sum()
            eod_exits = trade_markers['exits_eod'].notna().sum()
            
            total_trades = long_entries + short_entries
            total_exits = profit_exits + loss_exits + eod_exits
            
            logger.info(f"Total Trades: {total_trades} ({long_entries} long, {short_entries} short)")
            logger.info(f"Total Exits: {total_exits} ({profit_exits} TP, {loss_exits} SL, {eod_exits} EOD)")
            
            # Exit breakdown percentages
            if total_exits > 0:
                tp_pct = profit_exits / total_exits * 100
                sl_pct = loss_exits / total_exits * 100
                eod_pct = eod_exits / total_exits * 100
                logger.info(f"Exit Breakdown: {tp_pct:.1f}% TP, {sl_pct:.1f}% SL, {eod_pct:.1f}% EOD")
            
            # Calculate win rate based on actual PnL (including commissions)
            # This matches the trade reporter's calculation method
            if hasattr(self, 'trades_data') and self.trades_data:
                profitable_trades = sum(1 for trade in self.trades_data if trade.get('pnl', 0) > 0)
                if total_trades > 0:
                    win_rate = profitable_trades / total_trades * 100
                    logger.info(f"Win Rate (by PnL): {win_rate:.1f}%")
            else:
                # Fallback to exit-based calculation if trade data not available
                if total_exits > 0:
                    win_rate = profit_exits / total_exits * 100
                    logger.info(f"Win Rate (by exit type): {win_rate:.1f}%")
        
        # Average range size
        if rectangles:
            avg_range = np.mean([r['range_size'] for r in rectangles])
            logger.info(f"Average Opening Range Size: ${avg_range:.2f}")


def create_backtest_orb_chart(price_data: pd.DataFrame,
                             trades_data: List[Dict] = None,
                             range_start: str = "09:30:00",
                             range_end: str = "10:30:00", 
                             start_date: str = None,
                             end_date: str = None,
                             timeframe: str = "1min",
                             save_path: str = None) -> None:
    """
    Convenience function to create ORB backtest visualization.
    
    Args:
        price_data: DataFrame with OHLC data and features
        trades_data: List of trade dictionaries
        range_start: Opening range start time (HH:MM:SS)
        range_end: Opening range end time (HH:MM:SS)
        start_date: Start date for chart (YYYY-MM-DD)
        end_date: End date for chart (YYYY-MM-DD)
        timeframe: Chart timeframe (1min, 5min, 15min, H, D)
        save_path: Path to save chart image
    """
    visualizer = BacktestORBVisualizer(range_start, range_end, timeframe)
    visualizer.create_enhanced_backtest_chart(
        price_data, trades_data, start_date, end_date, save_path
    )
