# main.py
"""
Trading Bot Main Entry Point
===========================

This module provides the main entry points for training models and running backtests.
It handles data loading, feature engineering, and pipeline execution based on the 
configuration in config.yaml.

Feature Engineering Configuration
-------------------------------
Features are configured in config.yaml under each strategy's config:

    strategies:
      my_strategy:
        features:
          feature_list: ['returns', 'volatility', 'rsi']  # Features to calculate
          volatility_window: 20  # Parameters for specific features
          rsi_window: 14
        
The system will:
1. Load the feature list from config
2. Resolve any feature dependencies
3. Apply features in the correct order with specified parameters

Adding New Features:
1. Create the feature function in ml/feature_engineering.py
2. Register it with the @registry.register decorator
3. Add it to feature_list in config.yaml
4. (Optional) Add any parameters to the features section
"""

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
from common.config import config
from common.data_manager import load_ohlc_data, split_data
from ml.feature_engineering import create_features
from training_pipelines.ma_crossover_pipeline import MovingAverageCrossoverPipeline
from training_pipelines.open_range_breakout_pipeline import OpenRangeBreakoutPipeline
from trading.backtester import Backtester

# Get logger
logger = logging.getLogger(__name__)

def run_training(pipeline_name: str):
    """Selects and runs the specified training pipeline."""
    logger.info(f"Mode: 'train', Pipeline: '{pipeline_name}'")
    try:
        if pipeline_name == 'ma_crossover':
            MovingAverageCrossoverPipeline(config).run()
        elif pipeline_name in ['orb', 'ml_open_range_breakout']:  # Support both names
            OpenRangeBreakoutPipeline(config).run()
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
    except Exception as e:
        logger.critical(str(e))
        return

def run_backtesting():
    """Loads data and runs the backtesting pipeline."""
    logger.info(f"Mode: 'backtest', Strategy: '{config.active_strategy}'")
    
    # 1. Load Data
    logger.info(f"Loading data from: {config.data_path}")
    full_df = load_ohlc_data(config.data_path)
    if full_df is None or full_df.empty:
        logger.critical("Data could not be loaded. Halting backtest.")
        return

    # 2. Feature Engineering based on strategy config
    strategy_config = config.get_strategy_config()
    if 'features' in strategy_config:
        logger.info("Adding features from config...")
        feature_config = strategy_config['features']
        
        # Get feature list and parameters
        feature_list = feature_config.get('feature_list', [])
        feature_params = {
            'volatility_window': feature_config.get('volatility_window', 20),
            'rsi_window': feature_config.get('rsi_window', 14),
            'atr_window': feature_config.get('atr_window', 14),
            'lookback_days': feature_config.get('lookback_days', 20)
        }
        
        # Add all features in one go using the registry
        full_df = create_features(full_df, feature_list, **feature_params)
    
    # 3. Splitting Data
    split_ratio = config.trading_params['train_test_split_ratio']
    _, test_df = split_data(full_df, split_ratio)

    bt = Backtester(config)
    bt.run(test_df)

def main():
    parser = argparse.ArgumentParser(description='Trading Bot CLI')
    parser.add_argument('mode', choices=['train', 'backtest'], help='Operation mode')
    
    # Add pipeline as a positional argument after mode when in train mode
    parser.add_argument('pipeline', nargs='?', help='Training pipeline to use (required for train mode)', 
                       choices=['ma_crossover', 'orb', 'ml_open_range_breakout'])
    
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.pipeline:
            parser.error("Pipeline argument is required for train mode")
        run_training(args.pipeline)
            
    elif args.mode == 'backtest':
        run_backtesting()

if __name__ == '__main__':
    main()
