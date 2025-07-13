# main.py

import argparse
from common.config import config
from common import data_manager
from ml.model import train_and_save_model
from trading import backtester

def run_training():
    """Loads data and runs the model training pipeline."""
    print("Mode: Training")
    full_df = data_manager.load_ohlc_data(str(config.DATA_FILE_PATH)) # Use str() for paths
    if full_df.empty:
        return
    
    # NOTE: You don't have TRAIN_TEST_SPLIT_RATIO in YAML. Add it to general config.
    split_ratio = config.general.get('train_test_split_ratio', 0.8)
    train_df, _ = data_manager.split_data(full_df, split_ratio)
    
    # Pass only what's needed: data, strategy params, and the config object itself
    strategy_params = config.get_strategy_params()
    train_and_save_model(train_df, strategy_params, config)
    
def run_backtesting():
    """Loads data and runs the backtesting pipeline."""
    print("Mode: Backtesting")
    full_df = data_manager.load_ohlc_data(str(config.DATA_FILE_PATH))
    if full_df.empty:
        return
        
    split_ratio = config.general.get('train_test_split_ratio', 0.8)
    _, test_df = data_manager.split_data(full_df, split_ratio)
    
    # The backtester can now get the config itself
    backtester.run_backtest(test_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML-Enhanced Trading Bot")
    parser.add_argument(
        'mode', 
        choices=['train', 'backtest'],
        help="The mode to run the application in: 'train' or 'backtest'."
    )
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training()
    elif args.mode == 'backtest':
        run_backtesting()