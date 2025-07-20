# plot_ohlc.py

import pandas as pd
import mplfinance as mpf
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from ml.feature_engineering import create_features
from datetime import time as dt_time

# Get the logger without configuring it - configuration comes from Config class
logger = logging.getLogger(__name__)

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '1min', '5min', '15min', 'H', 'D')
        
    Returns:
        Resampled DataFrame with OHLCV data
    """
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled

def plot_ohlc_with_features(file_path, start_date=None, end_date=None, timeframe='1min'):
    """
    Loads OHLC data and plots it as a candlestick chart with NR4/NR7 markers.
    
    Args:
        file_path (str): The path to the CSV or DBN file
        start_date (str, optional): The start date for the plot slice (e.g., '2025-06-09')
        end_date (str, optional): The end date for the plot slice (e.g., '2025-06-11')
        timeframe (str, optional): Timeframe to display ('1min', '5min', '15min', 'H', 'D')
    """
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        # Load data
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.dbn':
            try:
                from databento import DBNStore
            except ImportError:
                logger.error("databento package is not installed. Please install it with 'pip install databento'")
                return
            store = DBNStore.from_file(file_path)
            df = store.to_df()
            logger.info("DBN file loaded and converted to DataFrame.")
        else:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info("CSV data loaded successfully.")

        # Sort and prepare data
        df = df.sort_index()

        # Calculate NR4 and NR7 features
        logger.info("Calculating NR4 and NR7 features...")
        df = create_features(df, feature_list=['is_nr4', 'is_nr7'])

        # Select plot range
        if start_date and end_date:
            plot_df = df.loc[start_date:end_date]
            title_date_range = f"({start_date} to {end_date})"
            logger.info(f"Slicing data for plotting from {start_date} to {end_date}...")
        else:
            plot_df = df.tail(1000)
            title_date_range = "(Last 1000 data points)"
            logger.info("No date range specified. Plotting the last 1000 data points...")

        if plot_df.empty:
            logger.error("\nNo data found in the specified date range.")
            logger.error(f"Please check that your data file '{file_path}' contains data between {start_date} and {end_date}.")
            return
            
        # Resample data to requested timeframe if different from 1min
        if timeframe != '1min':
            logger.info(f"Resampling data to {timeframe} timeframe...")
            plot_df = resample_ohlcv(plot_df, timeframe)

        # Create marker data for NR4 and NR7 signals
        nr4_markers = pd.Series(index=plot_df.index, dtype=float)
        nr7_markers = pd.Series(index=plot_df.index, dtype=float)
        
        # Process daily signals
        plot_count = 0
        for date, group in plot_df.groupby(plot_df.index.date):
            first_row = group.iloc[0]
            day_high = group['high'].max()
            day_range = day_high - group['low'].min()
            day_start = group.index[0]
            
            # Get signals from the original 1-minute data for this day
            orig_day_data = df[df.index.date == date]
            if len(orig_day_data) > 0 and orig_day_data['is_nr4'].iloc[0] == 1:
                plot_count += 1
                nr4_markers[day_start] = day_high + day_range * 0.01
            
            if len(orig_day_data) > 0 and orig_day_data['is_nr7'].iloc[0] == 1:
                plot_count += 1
                nr7_markers[day_start] = day_high + day_range * 0.02

        logger.info(f"Found {plot_count} days with NR4/NR7 signals in the selected date range")

        # Create addplot objects
        ap = []
        
        # Add NR4 markers if we found any
        if nr4_markers.notna().any():
            ap.append(mpf.make_addplot(nr4_markers, type='scatter', marker='^', 
                                     markersize=100, color='blue', label='NR4'))
        
        # Add NR7 markers if we found any
        if nr7_markers.notna().any():
            ap.append(mpf.make_addplot(nr7_markers, type='scatter', marker='v', 
                                     markersize=100, color='red', label='NR7'))

        # Plot configuration
        logger.info("Generating plot...")
        kwargs = {
            'type': 'candle',
            'style': 'charles',
            'title': f'\nSPY {timeframe} OHLC Data {title_date_range}\nBlue Triangle = NR4 Day, Red Triangle = NR7 Day',
            'ylabel': 'Price ($)',
            'volume': True,
            'mav': (40, 100),
            'figsize': (16, 9),
            'panel_ratios': (3, 1),
            'warn_too_much_data': 100000
        }
        
        # Only add the addplot parameter if we have markers to plot
        if ap:
            kwargs['addplot'] = ap
            
        mpf.plot(plot_df, **kwargs)

    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise  # Re-raise the exception to see the full traceback during development


def plot_ohlc_data(file_path, start_date=None, end_date=None):
    """Original plotting function without feature markers."""
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.dbn':
            try:
                from databento import DBNStore
            except ImportError:
                logger.error("databento package is not installed. Please install it with 'pip install databento'.")
                return
            store = DBNStore.from_file(file_path)
            df = store.to_df()
            logger.info("DBN file loaded and converted to DataFrame.")
        else:
            # 1. Load the data using pandas
            df = pd.read_csv(
                file_path,
                index_col=0,
                parse_dates=True
            )
            logger.info("CSV data loaded successfully.")

        # 2. Prepare the data for plotting
        # mplfinance requires specific column names: 'Open', 'High', 'Low', 'Close', 'Volume'
        # Your CSV already matches this format perfectly.
        # It's always good practice to ensure the data is sorted by date.
        df = df.sort_index()

        # 3. Select a subset of data to plot
        # Plotting millions of 1-minute bars is not feasible, so we'll slice the data.
        if start_date and end_date:
            plot_df = df.loc[start_date:end_date]
            title_date_range = f"({start_date} to {end_date})"
            logger.info(f"Slicing data for plotting from {start_date} to {end_date}...")
        else:
            # If no dates are provided, just plot the last 1000 bars as a sample.
            plot_df = df.tail(1000)
            title_date_range = "(Last 1000 data points)"
            logger.info("No date range specified. Plotting the last 1000 data points...")

        if plot_df.empty:
            logger.error("\nNo data found in the specified date range.")
            logger.error(f"Please check that your data file '{file_path}' contains data between {start_date} and {end_date}.")
            return

        # 4. Create the plot using mplfinance
        logger.info("Generating plot...")
        mpf.plot(
            plot_df,
            type='candle',         # Use 'candle' for candlestick chart. Other options: 'line', 'ohlc'.
            style='charles',       # A popular and clean style. Others: 'yahoo', 'nightclouds'.
            title=f'SPY 1-Minute OHLC Data {title_date_range}',
            ylabel='Price ($)',
            volume=True,           # Show a subplot with volume data.
            mav=(40, 100),         # Add 40 and 100-period moving averages, just like in your strategy.
            figratio=(16, 9),      # Make the plot wider.
            panel_ratios=(3, 1)    # Give more space to the price panel than the volume panel.
        )

    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def plot_open_range_breakout(file_path, start_date=None, end_date=None, 
                           range_start_time="09:30:00", range_end_time="10:15:00", 
                           timeframe='1min'):
    """
    Visualizes OHLC data with opening range bounds for Open Range Breakout analysis.
    
    This function creates a comprehensive visualization showing:
    - OHLC candlestick chart with volume
    - Opening range boundaries (upper and lower bounds) for each trading day
    - Both bullish AND bearish breakout signals using the same logic as training pipeline
    - Uses the breakout_direction feature for consistent signal detection
    - Summary statistics of breakout frequency and direction
    
    Args:
        file_path (str): Path to CSV or DBN file containing OHLC data
                        CSV format: columns should include 'open', 'high', 'low', 'close', 'volume'
                        DBN format: Databento binary format (requires databento package)
        start_date (str, optional): Start date for plot in 'YYYY-MM-DD' format (e.g., '2024-01-08')
                                   If None, shows last 2000 data points
        end_date (str, optional): End date for plot in 'YYYY-MM-DD' format (e.g., '2024-01-12')
                                 If None, shows last 2000 data points
        range_start_time (str): Opening range start time in 'HH:MM:SS' format (default: "09:30:00")
                               Typically market open time
        range_end_time (str): Opening range end time in 'HH:MM:SS' format (default: "10:15:00")
                             Defines the opening range period (e.g., first 45 minutes)
        timeframe (str): Data timeframe to display - '1min', '5min', '15min', '1H', 'D'
                        Note: Original data resolution should match or be higher than requested timeframe
    
    Returns:
        None: Displays matplotlib plot and prints analysis summary to console
    
    Visualization Elements:
        - Green/Red Candlesticks: Price action (green=bullish, red=bearish)
        - Red Dashed Lines: Opening range HIGH for each trading day
        - Green Dashed Lines: Opening range LOW for each trading day  
        - Blue Triangle Up: Bullish breakout signals (close above range high after range period)
        - Red Triangle Down: Bearish breakout signals (close below range low after range period)
        - Volume Bars: Trading volume subplot
        - Title: Shows date range, range times, and legend
    
    Console Output:
        - Loading progress and data validation messages
        - Count of trading days and opening ranges found
        - Count of bullish and bearish breakout signals detected
        - Breakout success rate percentage by direction
    
    Usage Examples:
        # Basic usage with date range
        plot_open_range_breakout('data/SPY_1min.csv', '2024-01-08', '2024-01-12')
        
        # Custom opening range (first 45 minutes - matches training pipeline)
        plot_open_range_breakout('data/SPY_1min.csv', '2024-01-08', '2024-01-12',
                               range_start_time="09:30:00", range_end_time="10:15:00")
        
        # Different timeframe (5-minute bars)
        plot_open_range_breakout('data/SPY_1min.csv', '2024-01-08', '2024-01-12',
                               timeframe='5min')
        
        # Auto-range (last 2000 bars)
        plot_open_range_breakout('data/SPY_1min.csv')
        
        # With DBN file
        plot_open_range_breakout('data/spy_ohlcv.dbn', '2024-01-08', '2024-01-12')
    
    Requirements:
        - pandas: Data manipulation
        - mplfinance: Candlestick charting
        - matplotlib: Plotting backend
        - databento (optional): For .dbn file support
        - ml.feature_engineering: For breakout_direction feature
    
    Notes:
        - Uses the same breakout detection logic as the training pipeline
        - Leverages the breakout_direction feature for consistent signal detection
        - Data should be in ascending chronological order
        - Function automatically filters for market hours based on range times
        - Only the first breakout per day is shown (no reversals)
        - Large datasets are automatically handled with warnings for performance
        - Opening ranges are calculated separately for each trading day
        - Weekend/holiday gaps are handled automatically
    
    Raises:
        FileNotFoundError: If the specified file_path does not exist
        ImportError: If databento package is required but not installed
        ValueError: If date formats are invalid or data is empty
        KeyError: If required OHLC columns are missing from data
    """
    logger.info(f"Creating Enhanced Open Range Breakout visualization from: {file_path}")
    
    try:
        # Load data
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.dbn':
            try:
                from databento import DBNStore
            except ImportError:
                logger.error("databento package is not installed. Please install it with 'pip install databento'")
                return
            store = DBNStore.from_file(file_path)
            df = store.to_df()
            logger.info("DBN file loaded and converted to DataFrame.")
        else:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info("CSV data loaded successfully.")

        # Ensure lowercase columns
        df.columns = [col.lower() for col in df.columns]
        df = df.sort_index()

        # Filter date range
        if start_date and end_date:
            plot_df = df.loc[start_date:end_date]
            title_date_range = f"({start_date} to {end_date})"
            logger.info(f"Plotting data from {start_date} to {end_date}...")
        else:
            plot_df = df.tail(2000)  # Show more data for ORB analysis
            title_date_range = "(Last 2000 bars)"
            logger.info("No date range specified. Plotting the last 2000 data points...")

        if plot_df.empty:
            logger.error("No data found in the specified date range.")
            return

        # Resample if needed
        if timeframe != '1min':
            logger.info(f"Resampling data to {timeframe}...")
            plot_df = resample_ohlcv(plot_df, timeframe)

        # Add breakout_direction feature using the same logic as training pipeline
        logger.info("Calculating breakout direction feature...")
        plot_df = create_features(plot_df, ['breakout_direction'], 
                                range_start=range_start_time, 
                                range_end=range_end_time)

        # Calculate opening ranges for visualization bounds
        logger.info("Calculating opening range bounds for visualization...")
        
        # Convert time strings to time objects
        range_start = pd.to_datetime(range_start_time).time()
        range_end = pd.to_datetime(range_end_time).time()
        
        # Create series to hold range bounds
        upper_bounds = pd.Series(index=plot_df.index, dtype=float)
        lower_bounds = pd.Series(index=plot_df.index, dtype=float)
        bullish_signals = pd.Series(index=plot_df.index, dtype=float)
        bearish_signals = pd.Series(index=plot_df.index, dtype=float)
        
        # Process each trading day for visualization bounds
        range_count = 0
        bullish_breakout_count = 0
        bearish_breakout_count = 0
        
        for date, day_data in plot_df.groupby(plot_df.index.date):
            # Get opening range data for this day
            day_times = pd.to_datetime(day_data.index).time
            range_mask = (day_times >= range_start) & (day_times < range_end)
            range_data = day_data[range_mask]
            
            if len(range_data) == 0:
                continue
                
            # Calculate range bounds
            range_high = range_data['high'].max()
            range_low = range_data['low'].min()
            range_count += 1
            
            # Fill bounds for the entire day
            day_mask = plot_df.index.date == date
            upper_bounds.loc[day_mask] = range_high
            lower_bounds.loc[day_mask] = range_low
        
        # Extract breakout signals from the breakout_direction feature
        logger.info("Extracting breakout signals from breakout_direction feature...")
        
        # Find breakout transitions (where direction changes from 0 to 1 or -1)
        breakout_mask = (plot_df['breakout_direction'] != 0) & (plot_df['breakout_direction'].shift(1) == 0)
        
        if breakout_mask.any():
            breakout_points = plot_df[breakout_mask].copy()
            
            # Separate bullish and bearish breakouts
            for idx in breakout_points.index:
                direction = plot_df.loc[idx, 'breakout_direction']
                price = plot_df.loc[idx, 'close']
                
                if direction == 1:  # Bullish breakout
                    bullish_signals.loc[idx] = price + (price * 0.002)  # Slightly above for visibility
                    bullish_breakout_count += 1
                elif direction == -1:  # Bearish breakout
                    bearish_signals.loc[idx] = price - (price * 0.002)  # Slightly below for visibility
                    bearish_breakout_count += 1

        total_breakouts = bullish_breakout_count + bearish_breakout_count
        logger.info(f"Found {range_count} trading days with opening ranges")
        logger.info(f"Detected {total_breakouts} breakout signals ({bullish_breakout_count} bullish, {bearish_breakout_count} bearish)")

        # Create addplot objects
        ap = []
        
        # Add upper bounds line
        if upper_bounds.notna().any():
            ap.append(mpf.make_addplot(upper_bounds, type='line', color='red', 
                                     width=2, linestyle='--', alpha=0.8))
        
        # Add lower bounds line  
        if lower_bounds.notna().any():
            ap.append(mpf.make_addplot(lower_bounds, type='line', color='green', 
                                     width=2, linestyle='--', alpha=0.8))
        
        # Add bullish breakout signals (blue triangles up)
        if bullish_signals.notna().any():
            ap.append(mpf.make_addplot(bullish_signals, type='scatter', marker='^', 
                                     markersize=80, color='blue', alpha=0.9))
        
        # Add bearish breakout signals (red triangles down)
        if bearish_signals.notna().any():
            ap.append(mpf.make_addplot(bearish_signals, type='scatter', marker='v', 
                                     markersize=80, color='red', alpha=0.9))
        
        # Add lower bounds line  
        if lower_bounds.notna().any():
            ap.append(mpf.make_addplot(lower_bounds, type='line', color='green', 
                                     width=2, linestyle='--', alpha=0.8))
        
        # Add bullish breakout signals (blue triangles up)
        if bullish_signals.notna().any():
            ap.append(mpf.make_addplot(bullish_signals, type='scatter', marker='^', 
                                     markersize=80, color='blue', alpha=0.9))
        
        # Add bearish breakout signals (red triangles down)
        if bearish_signals.notna().any():
            ap.append(mpf.make_addplot(bearish_signals, type='scatter', marker='v', 
                                     markersize=80, color='red', alpha=0.9))

        # Create the plot
        logger.info("Generating Enhanced Open Range Breakout visualization...")
        
        title = (f'Enhanced Open Range Breakout Analysis {title_date_range}\n'
                f'Range: {range_start_time} to {range_end_time} | '
                f'Red Dashed = Range High, Green Dashed = Range Low\n'
                f'Blue Triangles ↑ = Bullish Breakouts, Red Triangles ↓ = Bearish Breakouts')
        
        kwargs = {
            'type': 'candle',
            'style': 'charles',
            'title': title,
            'ylabel': 'Price ($)',
            'volume': True,
            'figsize': (20, 14),
            'panel_ratios': (4, 1),
            'warn_too_much_data': 100000,
        }
        
        if ap:
            kwargs['addplot'] = ap
            
        mpf.plot(plot_df, **kwargs)
        
        # Print enhanced summary statistics
        logger.info(f"\n--- Enhanced Open Range Breakout Analysis Summary ---")
        logger.info(f"Date Range: {title_date_range}")
        logger.info(f"Opening Range Time: {range_start_time} to {range_end_time}")
        logger.info(f"Total Trading Days: {range_count}")
        logger.info(f"Total Breakout Signals: {total_breakouts}")
        logger.info(f"  • Bullish Breakouts: {bullish_breakout_count}")
        logger.info(f"  • Bearish Breakouts: {bearish_breakout_count}")
        if range_count > 0:
            logger.info(f"Overall Breakout Rate: {total_breakouts/range_count:.1%}")
            if bullish_breakout_count > 0:
                logger.info(f"Bullish Breakout Rate: {bullish_breakout_count/range_count:.1%}")
            if bearish_breakout_count > 0:
                logger.info(f"Bearish Breakout Rate: {bearish_breakout_count/range_count:.1%}")
        logger.info(f"Integration: Using breakout_direction feature (same as training pipeline)")

    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == '__main__':
    # --- Configuration ---
    DBN_FILE = './data/spy_ohlcv_new.dbn'
    START_PLOT_DATE = '2025-06-09'
    END_PLOT_DATE = '2025-06-13'  # Extended to see more potential NR4/NR7 days

    # --- Run the plotting function ---
    # Example: Plot with daily bars
    plot_ohlc_with_features(DBN_FILE, start_date=START_PLOT_DATE, end_date=END_PLOT_DATE, timeframe='D')