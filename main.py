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
from common.config import config
from common import data_manager
from training_pipelines import get_training_pipeline
from ml.feature_engineering import create_features
from trading import backtester



def run_training(pipeline_name: str):
    """Selects and runs the specified training pipeline."""
    print(f"\n[INFO] Mode: 'train', Pipeline: '{pipeline_name}'")
    try:
        pipeline = get_training_pipeline(pipeline_name)
        pipeline.run()
    except ValueError as e:
        print(f"[CRITICAL] {e}")

def run_backtesting():
    """Loads data and runs the backtesting pipeline."""
    print(f"\n[INFO] Mode: 'backtest', Strategy: '{config.active_strategy}'")
    
    # 1. Load Data
    print(f"[INFO] Loading data from: {config.data_path}")
    full_df = data_manager.load_ohlc_data(str(config.data_path))
    if full_df.empty:
        print("[CRITICAL] Data could not be loaded. Halting backtest.")
        return

    # 2. Feature Engineering based on strategy config
    strategy_config = config.get_strategy_config()
    if 'features' in strategy_config:
        print("[INFO] Adding features from config...")
        feature_config = strategy_config['features']
        
        # Get feature list and parameters
        feature_list = feature_config.get('feature_list', [])
        feature_params = {
            'volatility_window': feature_config.get('volatility_window', 20),
            'rsi_window': feature_config.get('rsi_window', 14)
        }
        
        # Add all features in one go using the registry
        full_df = create_features(full_df, feature_list, **feature_params)
    
    # 3. Splitting Data
    split_ratio = config.trading_params['train_test_split_ratio']
    _, test_df = data_manager.split_data(full_df, split_ratio)

    backtester.run_backtest(test_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML-Enhanced Trading Bot")
    # Main command: 'train' or 'backtest'
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode')

    # Sub-parser for the 'train' command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('pipeline_name', help='The name of the training pipeline to run (e.g., ml_moving_average_crossover)')
    
    # Sub-parser for the 'backtest' command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest on a strategy')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(args.pipeline_name)
    elif args.mode == 'backtest':
        run_backtesting()
