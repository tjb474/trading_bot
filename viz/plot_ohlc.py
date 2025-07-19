# plot_ohlc.py

import pandas as pd
import mplfinance as mpf
import os
import numpy as np
from ml.feature_engineering import create_features

def plot_ohlc_with_features(file_path, start_date=None, end_date=None):
    """
    Loads OHLC data and plots it as a candlestick chart with NR4/NR7 markers.
    
    Args:
        file_path (str): The path to the CSV or DBN file
        start_date (str, optional): The start date for the plot slice (e.g., '2025-06-09')
        end_date (str, optional): The end date for the plot slice (e.g., '2025-06-11')
    """
    print(f"Attempting to load data from: {file_path}")
    try:
        # Load data
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.dbn':
            try:
                from databento import DBNStore
            except ImportError:
                print("Error: databento package is not installed. Please install it with 'pip install databento'")
                return
            store = DBNStore.from_file(file_path)
            df = store.to_df()
            print("DBN file loaded and converted to DataFrame.")
        else:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print("CSV data loaded successfully.")

        # Sort and prepare data
        df = df.sort_index()

        # Calculate NR4 and NR7 features
        print("Calculating NR4 and NR7 features...")
        df = create_features(df, feature_list=['is_nr4', 'is_nr7'])

        # Select plot range
        if start_date and end_date:
            plot_df = df.loc[start_date:end_date]
            title_date_range = f"({start_date} to {end_date})"
            print(f"Slicing data for plotting from {start_date} to {end_date}...")
        else:
            plot_df = df.tail(1000)
            title_date_range = "(Last 1000 data points)"
            print("No date range specified. Plotting the last 1000 data points...")

        if plot_df.empty:
            print("\nError: No data found in the specified date range.")
            print(f"Please check that your data file '{file_path}' contains data between {start_date} and {end_date}.")
            return

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
            
            # Place markers at the start of each signal day
            if first_row['is_nr4'] == 1:
                plot_count += 1
                nr4_markers[day_start] = day_high + day_range * 0.01
            
            if first_row['is_nr7'] == 1:
                plot_count += 1
                nr7_markers[day_start] = day_high + day_range * 0.02

        print(f"Found {plot_count} days with NR4/NR7 signals in the selected date range")

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
        print("Generating plot...")
        kwargs = {
            'type': 'candle',
            'style': 'charles',
            'title': f'\nSPY 1-Minute OHLC Data {title_date_range}\nBlue Triangle = NR4 Day, Red Triangle = NR7 Day',
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
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise  # Re-raise the exception to see the full traceback during development


def plot_ohlc_data(file_path, start_date=None, end_date=None):
    """Original plotting function without feature markers."""
    print(f"Attempting to load data from: {file_path}")
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.dbn':
            try:
                from databento import DBNStore
            except ImportError:
                print("Error: databento package is not installed. Please install it with 'pip install databento'.")
                return
            store = DBNStore.from_file(file_path)
            df = store.to_df()
            print("DBN file loaded and converted to DataFrame.")
        else:
            # 1. Load the data using pandas
            df = pd.read_csv(
                file_path,
                index_col=0,
                parse_dates=True
            )
            print("CSV data loaded successfully.")

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
            print(f"Slicing data for plotting from {start_date} to {end_date}...")
        else:
            # If no dates are provided, just plot the last 1000 bars as a sample.
            plot_df = df.tail(1000)
            title_date_range = "(Last 1000 data points)"
            print("No date range specified. Plotting the last 1000 data points...")

        if plot_df.empty:
            print("\nError: No data found in the specified date range.")
            print(f"Please check that your data file '{file_path}' contains data between {start_date} and {end_date}.")
            return

        # 4. Create the plot using mplfinance
        print("Generating plot...")
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
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    DBN_FILE = './data/spy_ohlcv_new.dbn'
    START_PLOT_DATE = '2025-06-09'
    END_PLOT_DATE = '2025-06-13'  # Extended to see more potential NR4/NR7 days

    # --- Run the plotting function ---
    plot_ohlc_with_features(DBN_FILE, start_date=START_PLOT_DATE, end_date=END_PLOT_DATE)