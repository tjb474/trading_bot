# trading/backtester.py

import backtrader as bt
import pandas as pd
import logging
from .strategy import MLStrategy
from data.data_manager import load_dbn_to_df

def run_backtest(test_data: pd.DataFrame, config):
    # --- Logging setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("Backtester")
    logger.info(f"Starting backtest with {len(test_data)} rows.")
    logger.info(f"Columns: {list(test_data.columns)}")
    logger.info(f"Data types:\n{test_data.dtypes}")
    nan_counts = test_data.isnull().sum()
    logger.info(f"NaN counts per column:\n{nan_counts}")
    logger.info(f"First 5 rows:\n{test_data.head()}\n")
    # --- END OF Logging setup ---

    cerebro = bt.Cerebro()
    
    data_feed = bt.feeds.PandasData(dataname=test_data)
    cerebro.adddata(data_feed)
    
    cerebro.addstrategy(MLStrategy, config=config)
    
    cerebro.broker.setcash(config.INITIAL_CASH)
    cerebro.addsizer(bt.sizers.FixedSize, stake=config.STAKE_SIZE)
    
    price_approx = test_data['close'].mean()
    commission = config.COMMISSION_SPREAD_POINTS / price_approx
    cerebro.broker.setcommission(commission=commission)
    
    # --- Using a shorter name for the analyzer to make it easier to access ---
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