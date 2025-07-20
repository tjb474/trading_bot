#!/usr/bin/env python3
"""
Open Range Breakout Visualization Demo

This script demonstrates how to visualize OHLC data with opening range bounds
for Open Range Breakout strategy analysis using the same data source as the training pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.plot_ohlc import plot_open_range_breakout
from common.config import config
import logging

# Configure logging using the same system as the main project
logger = logging.getLogger(__name__)

def main():
    """Demonstrate Open Range Breakout visualization using project configuration"""
    
    # Get data file path from config (same as training pipeline)
    DATA_FILE = str(config.data_path)
    
    # Get ORB strategy configuration
    orb_config = config.get_strategy_config('ml_open_range_breakout')
    RANGE_START = orb_config['range']['start']
    RANGE_END = orb_config['range']['end']
    
    # Demo date range (adjust as needed for your data)
    START_DATE = '2025-01-01'  # Adjust based on your data availability
    END_DATE = '2025-01-15'    # Two weeks for demonstration
    
    print("=" * 60)
    print("OPEN RANGE BREAKOUT VISUALIZATION DEMO")
    print("=" * 60)
    print(f"Using project configuration from config.yaml")
    print(f"Data File: {DATA_FILE}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Opening Range: {RANGE_START} to {RANGE_END}")
    print(f"Strategy Config: ml_open_range_breakout")
    print("=" * 60)
    
    try:
        # Create the visualization using project configuration
        plot_open_range_breakout(
            file_path=DATA_FILE,
            start_date=START_DATE,
            end_date=END_DATE,
            range_start_time=RANGE_START,
            range_end_time=RANGE_END,
            timeframe='1min'  # Keep minute-level detail
        )
        
        print("\n✓ Visualization complete!")
        print("\nLegend:")
        print("• Red dashed lines = Opening range high for each day")
        print("• Green dashed lines = Opening range low for each day") 
        print("• Blue triangles = Breakout signals (price breaks above range high)")
        print("• Candlesticks show minute-by-minute price action")
        
        print(f"\nStrategy Parameters from config.yaml:")
        print(f"• Opening Range: {RANGE_START} to {RANGE_END}")
        print(f"• Take Profit Multiplier: {orb_config['risk']['take_profit_multiplier']}")
        print(f"• Stop Loss Multiplier: {orb_config['risk']['stop_loss_multiplier']}")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file not found at {DATA_FILE}")
        print("Please check that your data file exists in the configured location.")
        print("You may need to run the data download pipeline first.")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()