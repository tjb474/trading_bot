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

import argparse
import logging
import pandas as pd
from common.config import config
from common.data_manager import load_ohlc_data, split_data, convert_to_eastern_time
from ml.feature_engineering import create_features
from training_pipelines.ma_crossover_pipeline import MovingAverageCrossoverPipeline
from training_pipelines.open_range_breakout_pipeline import OpenRangeBreakoutPipeline
from trading.backtester import Backtester

# Get logger
logger = logging.getLogger(__name__)

def run_training(pipeline_name: str):
    """Selects and runs the specified training pipeline."""
    # This function is now simplified, as the pipeline itself will call create_features
    logger.info(f"Mode: 'train', Pipeline: '{pipeline_name}'")
    try:
        if pipeline_name == 'ma_crossover':
            MovingAverageCrossoverPipeline().run()
        elif pipeline_name in ['orb', 'ml_open_range_breakout']:
            OpenRangeBreakoutPipeline().run()
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
    except Exception as e:
        logger.critical(f"An error occurred during training: {e}", exc_info=True)
        return

def run_backtesting(strategy_name: str = None):
    """Loads data and runs the backtesting pipeline with date filtering."""
    # Use provided strategy or fall back to config default
    active_strategy = strategy_name or config.active_strategy
    logger.info(f"Mode: 'backtest', Strategy: '{active_strategy}'")

    # 1. Load Data
    logger.info(f"Loading data from: {config.data_path}")
    full_df = load_ohlc_data(config.data_path)
    if full_df is None or full_df.empty:
        logger.critical("Data could not be loaded. Halting backtest.")
        return

    # 2. Apply date range filtering if specified in config
    trading_params = config.trading_params
    backtest_start = trading_params.get('backtest_start_date')
    backtest_end = trading_params.get('backtest_end_date')

    if backtest_start or backtest_end:
        logger.info(f"Applying date range filter: {backtest_start} to {backtest_end}")
        
        # Convert to timezone-aware index before filtering
        full_df = convert_to_eastern_time(full_df)

        if backtest_start:
            start_date = pd.to_datetime(backtest_start).tz_localize('US/Eastern')
            full_df = full_df[full_df.index >= start_date]
            logger.info(f"Data filtered from start date: {start_date}")

        if backtest_end:
            end_date = pd.to_datetime(backtest_end).tz_localize('US/Eastern')
            full_df = full_df[full_df.index <= end_date]
            logger.info(f"Data filtered to end date: {end_date}")

        if full_df.empty:
            logger.critical("No data remains after applying date filter. Halting backtest.")
            return
            
        logger.info(f"Data filtered to {len(full_df)} rows between {full_df.index.min()} and {full_df.index.max()}")

    # 3. Feature Engineering based on strategy config
    strategy_config = config.get_strategy_config(active_strategy)
    if 'features' in strategy_config and 'feature_list' in strategy_config['features']:
        logger.info("Adding features from config for backtesting...")

        # Ensure data is in Eastern Time before feature calculation
        # This call is safe to make even if already done for date filtering
        full_df = convert_to_eastern_time(full_df)

        feature_config = strategy_config['features']
        feature_list = feature_config.get('feature_list', [])

        # THE FIX: Gather ALL parameters from the feature config section automatically
        # This is more robust than manually listing them.
        feature_params = feature_config.copy()
        
        # Also add any other relevant parameters from other sections if needed (like 'range')
        if 'range' in strategy_config:
            feature_params.update(strategy_config['range'])
        
        # Remove the feature_list itself from the parameters dictionary
        feature_params.pop('feature_list', None)
        
        full_df = create_features(full_df, feature_list, **feature_params)

    # 4. Splitting Data (optional, but good practice to separate test set)
    # Check if we should use full date range or respect train/test split
    use_full_range = trading_params.get('use_full_date_range', False)
    strategy_config = config.get_strategy_config(active_strategy)
    uses_ml_filter = strategy_config.get('model', {}).get('use_ml_filter', False)
    
    if backtest_start or backtest_end:
        if use_full_range:
            # Use all the filtered data for backtesting when explicitly requested
            test_df = full_df
            logger.info(f"Using all filtered data for backtesting: {len(test_df)} rows")
            
            # LOUD WARNING when using ML and full range
            if uses_ml_filter:
                logger.warning("=" * 80)
                logger.warning("🚨 WARNING: BACKTESTING ON FULL DATE RANGE WITH ML MODEL ENABLED! 🚨")
                logger.warning("You are backtesting over the ENTIRE date range, which may include")
                logger.warning("data that was used to train the ML model. Results may be misleading!")
                logger.warning("Consider setting 'use_full_date_range: false' in config.yaml")
                logger.warning("to only backtest on the test portion of your date range.")
                logger.warning("=" * 80)
            else:
                logger.info("ℹ️  Using full date range for backtesting (ML filter disabled)")
        else:
            # Apply train/test split even with specific dates (recommended for ML)
            split_ratio = config.trading_params.get('train_test_split_ratio', 0.8)
            _, test_df = split_data(full_df, split_ratio)
            
            # Calculate actual date range being used
            actual_start = test_df.index.min().strftime('%Y-%m-%d')
            actual_end = test_df.index.max().strftime('%Y-%m-%d')
            
            logger.info(f"Applying train/test split to filtered data:")
            logger.info(f"  Requested range: {backtest_start} to {backtest_end}")
            logger.info(f"  Actual backtest range: {actual_start} to {actual_end}")
            logger.info(f"  Using {len(test_df)} rows (last {(1-split_ratio)*100:.0f}% of filtered data)")
            
            if uses_ml_filter:
                logger.info("✅ Good practice: Using test portion only with ML model enabled")
    else:
        # No specific dates provided - use train/test split on full dataset
        split_ratio = config.trading_params.get('train_test_split_ratio', 0.8)
        _, test_df = split_data(full_df, split_ratio)
        
        actual_start = test_df.index.min().strftime('%Y-%m-%d')
        actual_end = test_df.index.max().strftime('%Y-%m-%d')
        
        logger.info(f"No specific dates provided - using train/test split:")
        logger.info(f"  Backtest range: {actual_start} to {actual_end}")
        logger.info(f"  Using {len(test_df)} rows (last {(1-split_ratio)*100:.0f}% of full dataset)")

    # 5. Run backtest with the specified strategy
    bt = Backtester(config, active_strategy)
    bt.run(test_df)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description='Trading Bot CLI')
    parser.add_argument('mode', choices=['train', 'backtest'], help='Operation mode')

    # Add pipeline/strategy as a positional argument after mode
    parser.add_argument('strategy', nargs='?', help='Strategy/pipeline name (e.g., "ml_open_range_breakout")')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=config.logging['level'].upper(), format=config.logging['format'])

    if args.mode == 'train':
        if not args.strategy:
            parser.error("The 'train' mode requires a pipeline name argument (e.g., 'ml_open_range_breakout').")
        run_training(args.strategy)

    elif args.mode == 'backtest':
        # Strategy argument is optional for backtest; will use active_strategy from config if not provided
        run_backtesting(args.strategy)

if __name__ == '__main__':
    main()