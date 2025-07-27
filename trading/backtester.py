# trading/backtester.py
import backtrader as bt
import pandas as pd
import logging
from common.config import config
from strategies import get_strategy
from common.data_manager import convert_to_eastern_time  # Import the new function

class Backtester:
    """Backtester class that handles running trading strategy backtests."""
    
    def __init__(self, config, strategy_name: str = None):
        """Initialize backtester with configuration."""
        self.config = config
        self.strategy_name = strategy_name or config.active_strategy
        self.logger = logging.getLogger("Backtester")
        
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
        
        # Add strategy with its flattened parameters
        self.logger.info(f"Adding strategy with parameters: {strategy_params}")
        cerebro.addstrategy(StrategyClass, **strategy_params)
        
        # --- Broker Setup ---
        trading_params = self.config.trading_params
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

        self.logger.info("\nPlotting results...")
        cerebro.plot(style='candlestick')