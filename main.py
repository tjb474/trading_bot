# main.py

import argparse
from common import config, data_manager
from ml.model import train_and_save_model
from trading import backtester

def run_training():
    """Loads data and runs the model training pipeline."""
    print("Mode: Training")
    full_df = data_manager.load_ohlc_data(config.DATA_FILE_PATH)
    if full_df.empty:
        return
    
    train_df, _ = data_manager.split_data(full_df, config.TRAIN_TEST_SPLIT_RATIO)
    train_and_save_model(train_df, config)

def run_backtesting():
    """Loads data and runs the backtesting pipeline."""
    print("Mode: Backtesting")
    full_df = data_manager.load_ohlc_data(config.DATA_FILE_PATH)
    if full_df.empty:
        return
        
    _, test_df = data_manager.split_data(full_df, config.TRAIN_TEST_SPLIT_RATIO)
    backtester.run_backtest(test_df, config)

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