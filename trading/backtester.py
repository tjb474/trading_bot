# trading/backtester.py
import backtrader as bt
import pandas as pd
import logging
import os
from common.config import config
from strategies import get_strategy
from common.data_manager import convert_to_eastern_time  # Import the new function
from reports.trade_reporter import TradeReporter

class Backtester:
    """Backtester class that handles running trading strategy backtests."""
    
    def __init__(self, config, strategy_name: str = None):
        """Initialize backtester with configuration."""
        self.config = config
        self.strategy_name = strategy_name or config.active_strategy
        self.logger = logging.getLogger("Backtester")
        self.trade_reporter = None
        
    def run(self, test_data: pd.DataFrame):
        """Run a backtest with the given test data."""
        # --- Logging setup ---
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        
        # Suppress verbose matplotlib logging
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        # --- Data validation and Timezone alignment ---
        self.logger.info(f"Starting backtest with {len(test_data)} rows.")
        
        # REFACTORED: Use the centralized function to handle timezone conversion
        test_data = convert_to_eastern_time(test_data)
        
        # Now, create the naive version specifically for backtrader's feed
        test_data_for_bt = test_data.copy()
        test_data_for_bt.index = test_data_for_bt.index.tz_localize(None)
        self.logger.info(f"Converted to naive Eastern time for backtrader feed.")
        self.logger.info(f"Sample naive timestamps for feed: {test_data_for_bt.index[:3].tolist()}")
        
        data_feed = bt.feeds.PandasData(dataname=test_data_for_bt)
        
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        
        # --- Strategy Selection ---
        self.logger.info(f"Loading strategy: '{self.strategy_name}'")
        StrategyClass = get_strategy(self.strategy_name)
        
        # Get strategy configuration and prepare parameters
        strategy_config = self.config.get_strategy_config(self.strategy_name)
        
        # Flatten nested config into parameter dict
        strategy_params = {}
        
        # Add basic parameters that are directly in the strategy config
        for key, value in strategy_config.items():
            if not isinstance(value, dict):
                strategy_params[key] = value
                
        # Handle nested parameters (features, model, etc)
        if 'features' in strategy_config:
            for key, value in strategy_config['features'].items():
                strategy_params[key] = value
                
        if 'model' in strategy_config:
            for key, value in strategy_config['model'].items():
                if key == 'path':
                    strategy_params['model_file_path'] = str(self.config.get_model_path(self.strategy_name))
                else:
                    strategy_params[key] = value
                    
        if 'risk' in strategy_config:
            for key, value in strategy_config['risk'].items():
                strategy_params[key] = value
                
        if 'range' in strategy_config:
            for key, value in strategy_config['range'].items():
                strategy_params[f'range_{key}'] = value
        
        # Pass the feature data to the strategy so it can access pre-calculated features
        # Use the Eastern timezone version for accurate feature lookups
        strategy_params['feature_data'] = test_data
        
        # Initialize trade reporter
        self.trade_reporter = TradeReporter(
            strategy_name=self.strategy_name,
            config_data=self.config.yaml_data  # Access the yaml data directly
        )
        
        # Get trading params before using them
        trading_params = self.config.trading_params
        
        # Set backtest metadata
        backtest_metadata = {
            'strategy': self.strategy_name,
            'start_date': test_data.index[0].strftime('%Y-%m-%d'),
            'end_date': test_data.index[-1].strftime('%Y-%m-%d'),
            'total_bars': len(test_data),
            'initial_cash': trading_params['initial_cash'],
            'stake_size': trading_params['stake_size'],
            'commission_spread_points': trading_params['commission_spread_points']
        }
        self.trade_reporter.set_backtest_metadata(backtest_metadata)
        
        # Add strategy with its flattened parameters
        self.logger.info(f"Adding strategy with parameters: {strategy_params}")
        cerebro.addstrategy(StrategyClass, **strategy_params)
        
        # --- Broker Setup ---
        cerebro.broker.setcash(trading_params['initial_cash'])
        cerebro.addsizer(bt.sizers.FixedSize, stake=trading_params['stake_size'])
        
        # Calculate commission from spread points
        price_approx = test_data['close'].mean()
        commission = trading_params['commission_spread_points'] / price_approx
        cerebro.broker.setcommission(commission=commission)
        
        self.logger.info(f"Broker configured: Cash=${trading_params['initial_cash']:,.2f}, "
                    f"Stake={trading_params['stake_size']}, Commission={commission:.6f}")
        
        # --- Analyzers ---
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # --- Run Backtest ---
        self.logger.info(f'Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}')
        results = cerebro.run()
        
        # --- Process Results ---
        strat = results[0]
        
        # Now we need to manually process the trades from the strategy
        # The trades_history should contain all the trades we need
        if hasattr(strat, 'trades_history') and strat.trades_history:
            self.logger.info(f"Processing {len(strat.trades_history)} trades from strategy history")
            for trade in strat.trades_history:
                self.trade_reporter.add_trade(trade)
        else:
            self.logger.warning("No trades found in strategy history")
        
        analysis = strat.analyzers
        
        self.logger.info('\n--- Backtest Results ---')
        self.logger.info(f'Final Portfolio Value: {cerebro.broker.getvalue():,.2f}')

        # Get Sharpe Ratio
        sharpe_ratio = analysis.sharpe.get_analysis().get('sharperatio')
        if sharpe_ratio is not None:
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        else:
            self.logger.info("Sharpe Ratio: N/A (Not enough data or trades)")

        # Get Max Drawdown
        max_drawdown = analysis.drawdown.get_analysis().get('max', {}).get('drawdown')
        if max_drawdown is not None:
            self.logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        else:
            self.logger.info("Max Drawdown: N/A")

        # Get Total Return
        total_return = analysis.returns.get_analysis().get('rtot')
        if total_return is not None:
            self.logger.info(f"Total Return: {total_return * 100:.2f}%")
        else:
            self.logger.info("Total Return: N/A")

        # --- Generate Trade Report FIRST (before visualization) ---
        self.logger.info("\n--- Generating Trade Report ---")
        
        # Print detailed trade report to console
        self.trade_reporter.print_detailed_report(max_trades=50)
        
        # Export reports to files
        try:
            self.trade_reporter.export_to_csv()
            self.trade_reporter.export_to_html()
            self.trade_reporter.save_metadata()
            self.logger.info("Trade reports exported successfully")
        except Exception as e:
            self.logger.warning(f"Error exporting trade reports: {e}")

        self.logger.info("\nPlotting results...")
        
        # Debug: Show strategy name
        self.logger.info(f"Strategy name for visualization: '{self.strategy_name}'")
        
        # Create enhanced ORB visualization if it's an ORB strategy
        if 'open_range_breakout' in self.strategy_name.lower():
            self.logger.info("ORB strategy detected - creating enhanced visualization...")
            try:
                self._create_orb_visualization(test_data, strat)
                self.logger.info("Enhanced ORB visualization completed successfully!")
            except Exception as e:
                self.logger.error(f"Could not create ORB visualization: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Fallback to standard plot
                cerebro.plot(style='candlestick')
        else:
            self.logger.info("Non-ORB strategy detected - using standard plot")
            cerebro.plot(style='candlestick')
            
        return self.trade_reporter
            
    def _create_orb_visualization(self, price_data: pd.DataFrame, strategy_instance):
        """Create enhanced ORB visualization with rectangles and trade markers."""
        try:
            from viz.backtest_orb_visualization import create_backtest_orb_chart
            
            # Get strategy configuration for ORB parameters
            strategy_config = self.config.get_strategy_config(self.strategy_name)
            range_start = strategy_config.get('range', {}).get('start', '09:30:00')
            range_end = strategy_config.get('range', {}).get('end', '10:30:00')
            
            self.logger.info(f"ORB parameters: range_start={range_start}, range_end={range_end}")
            
            # Get trades history from strategy
            trades_data = []
            if hasattr(strategy_instance, 'get_trades_history'):
                trades_data = strategy_instance.get_trades_history()
                self.logger.info(f"Retrieved {len(trades_data)} trades from strategy")
            else:
                self.logger.warning("Strategy does not have get_trades_history method - no trade markers will be shown")
            
            # Create date range for visualization (last 30 days or backtest range)
            backtest_params = self.config.trading_params
            if backtest_params.get('backtest_start_date') and backtest_params.get('backtest_end_date'):
                start_date = backtest_params['backtest_start_date']
                end_date = backtest_params['backtest_end_date']
            else:
                # Use last 30 trading days
                end_date = price_data.index[-1].strftime('%Y-%m-%d')
                start_idx = max(0, len(price_data) - 7800)  # Approx 30 days of 1min data
                start_date = price_data.index[start_idx].strftime('%Y-%m-%d')
            
            self.logger.info(f"Creating ORB visualization from {start_date} to {end_date}")
            
            # Create reports directory for charts
            charts_dir = "reports/charts"
            os.makedirs(charts_dir, exist_ok=True)
            
            # Create the enhanced chart
            create_backtest_orb_chart(
                price_data=price_data,
                trades_data=trades_data,
                range_start=range_start,
                range_end=range_end,
                start_date=start_date,
                end_date=end_date,
                timeframe='1min',  # Use 1min to match trading logic granularity
                save_path=f"{charts_dir}/orb_backtest_results_{self.strategy_name}_{start_date}_to_{end_date}.png"
            )
            
        except ImportError as e:
            self.logger.error(f"ORB visualization dependencies not available: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error creating ORB visualization: {e}")
            raise