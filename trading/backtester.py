# trading/backtester.py

import backtrader as bt
import pandas as pd
import logging

from common.config import config # Grab the INSTANCE named 'config' from within the config.py module 
from strategies import get_strategy # Import the new factory function
# from .strategy import MLStrategy
from strategies.ml_moving_average_crossover import MLMovingAverageCrossover
# from strategies.rsi_mean_reversion import RsiMeanReversion
# from data.data_manager import load_dbn_to_df

def run_backtest(test_data: pd.DataFrame):
    # --- Logging setup ---
    logger = logging.getLogger("Backtester")
    logger.info("--- Starting Backtest ---")

    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=test_data)
    cerebro.adddata(data_feed)
    
    # 1. Get the active strategy name from the config
    active_strategy_name = config.active_strategy
    logger.info(f"Active strategy: {active_strategy_name}")

    # 2. Get the strategy class using the factory
    StrategyClass = get_strategy(active_strategy_name)
    
    # 3. Get the strategy-specific parameters
    strategy_params = config.get_strategy_params()
    # The model path needs to be absolute for the strategy to find it
    strategy_params['model_file_path'] = str(config.get_model_path())
    logger.info(f"Strategy params: {strategy_params}")
    
    # 4. Add the strategy and unpack its parameters using **
    cerebro.addstrategy(StrategyClass, **strategy_params)
    
    # 5. Use general config settings
    general_params = config.general
    cerebro.broker.setcash(general_params['initial_cash'])
    
    # Note: STAKE_SIZE is not in your YAML. It should be.
    # For now, let's add it to the general section of config.yaml
    cerebro.addsizer(bt.sizers.FixedSize, stake=general_params.get('stake_size', 1))
    
    price_approx = test_data['close'].mean()
    commission = general_params['commission_spread_points'] / price_approx
    cerebro.broker.setcommission(commission=commission)
    
    # --- The rest of the file is largely the same ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    
    strat = results[0]
    analysis = strat.analyzers
    
    # --- FIX: Robustly print analysis results ---
    print(f'\n--- Backtest Results ---')
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Get Sharpe Ratio safely
    sharpe_ratio = analysis.sharpe.get_analysis().get('sharperatio')
    if sharpe_ratio is not None:
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Sharpe Ratio: N/A (Not enough data or trades)")

    # Get Max Drawdown safely
    max_drawdown = analysis.drawdown.get_analysis().get('max', {}).get('drawdown')
    if max_drawdown is not None:
        print(f"Max Drawdown: {max_drawdown:.2f}%")
    else:
        print("Max Drawdown: N/A")

    # Get Total Return safely
    total_return = analysis.returns.get_analysis().get('rtot')
    if total_return is not None:
        print(f"Total Return: {total_return * 100:.2f}%")
    else:
        print("Total Return: N/A")
    # --- END OF FIX ---

    print("\nPlotting results...")
    cerebro.plot(style='candlestick')