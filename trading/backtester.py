# trading/backtester.py
import backtrader as bt
import pandas as pd
import logging
from common.config import config
from strategies import get_strategy

def run_backtest(test_data: pd.DataFrame):
    # --- Logging setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("Backtester")
    
    # Data validation logging
    logger.info(f"Starting backtest with {len(test_data)} rows.")
    logger.info(f"Columns: {list(test_data.columns)}")
    logger.info(f"Data types:\n{test_data.dtypes}")
    nan_counts = test_data.isnull().sum()
    logger.info(f"NaN counts per column:\n{nan_counts}")
    logger.info(f"First 5 rows:\n{test_data.head()}\n")

    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=test_data)
    cerebro.adddata(data_feed)
    
    # --- Strategy Selection ---
    active_strategy_name = config.active_strategy
    logger.info(f"Loading strategy: '{active_strategy_name}'")
    StrategyClass = get_strategy(active_strategy_name)
    
    # Get strategy configuration and prepare parameters
    strategy_config = config.get_strategy_config()
    
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
                strategy_params['model_file_path'] = str(config.get_model_path())
            else:
                strategy_params[key] = value
                
    if 'risk' in strategy_config:
        for key, value in strategy_config['risk'].items():
            strategy_params[key] = value
            
    if 'range' in strategy_config:
        for key, value in strategy_config['range'].items():
            strategy_params[f'range_{key}'] = value
    
    # Add strategy with its flattened parameters
    logger.info(f"Adding strategy with parameters: {strategy_params}")
    cerebro.addstrategy(StrategyClass, **strategy_params)
    
    # --- Broker Setup ---
    trading_params = config.trading_params
    cerebro.broker.setcash(trading_params['initial_cash'])
    cerebro.addsizer(bt.sizers.FixedSize, stake=trading_params['stake_size'])
    
    # Calculate commission from spread points
    price_approx = test_data['close'].mean()
    commission = trading_params['commission_spread_points'] / price_approx
    cerebro.broker.setcommission(commission=commission)
    
    logger.info(f"Broker configured: Cash=${trading_params['initial_cash']:,.2f}, "
                f"Stake={trading_params['stake_size']}, Commission={commission:.6f}")
    
    # --- Analyzers ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # --- Run Backtest ---
    logger.info(f'Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}')
    results = cerebro.run()
    
    # --- Process Results ---
    strat = results[0]
    analysis = strat.analyzers
    
    logger.info('\n--- Backtest Results ---')
    logger.info(f'Final Portfolio Value: {cerebro.broker.getvalue():,.2f}')

    # Get Sharpe Ratio
    sharpe_ratio = analysis.sharpe.get_analysis().get('sharperatio')
    if sharpe_ratio is not None:
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        logger.info("Sharpe Ratio: N/A (Not enough data or trades)")

    # Get Max Drawdown
    max_drawdown = analysis.drawdown.get_analysis().get('max', {}).get('drawdown')
    if max_drawdown is not None:
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    else:
        logger.info("Max Drawdown: N/A")

    # Get Total Return
    total_return = analysis.returns.get_analysis().get('rtot')
    if total_return is not None:
        logger.info(f"Total Return: {total_return * 100:.2f}%")
    else:
        logger.info("Total Return: N/A")

    logger.info("\nPlotting results...")
    cerebro.plot(style='candlestick')