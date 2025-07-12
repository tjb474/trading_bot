# plot_ohlc.py

import pandas as pd
import mplfinance as mpf
import os

def plot_ohlc_data(file_path, start_date=None, end_date=None):
    """
    Loads OHLC data from a CSV or DBN file and plots it as a candlestick chart.
    Includes volume and the moving averages used in the strategy.

    Args:
        file_path (str): The path to the CSV or DBN file.
        start_date (str, optional): The start date for the plot slice (e.g., '2025-06-09').
        end_date (str, optional): The end date for the plot slice (e.g., '2025-06-11').
    """
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
    # IMPORTANT: Change these dates to view different parts of your data.
    # Let's plot the first two full days of data as an example.
    START_PLOT_DATE = '2025-06-09'
    END_PLOT_DATE = '2025-06-10'

    # --- Run the plotting function ---
    plot_ohlc_data(DBN_FILE, start_date=START_PLOT_DATE, end_date=END_PLOT_DATE)