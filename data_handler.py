import pandas as pd
from datetime import datetime, timedelta

def get_ig_historical_data(ig_service, epic, resolution='D', num_points=250):
    """
    Fetches historical price data from the IG API and correctly parses
    the pandas DataFrame returned by the trading-ig library.
    """
    # Calculate the date range for the request
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(num_points * 1.5))

    # Fetch the data
    response = ig_service.fetch_historical_prices_by_epic_and_date_range(
        epic=epic,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date
    )

    # --- THE CORRECT PARSING LOGIC ---
    # The 'prices' key already contains a DataFrame. Let's access it.
    prices_df = response['prices']

    # Create a new, clean DataFrame in the format backtrader needs.
    # We will average the bid and ask prices to get a single price point.
    df = pd.DataFrame()
    df['Open'] = (prices_df['bid']['Open'] + prices_df['ask']['Open']) / 2
    df['High'] = (prices_df['bid']['High'] + prices_df['ask']['High']) / 2
    df['Low'] = (prices_df['bid']['Low'] + prices_df['ask']['Low']) / 2
    df['Close'] = (prices_df['bid']['Close'] + prices_df['ask']['Close']) / 2
    df['Volume'] = prices_df['last']['Volume'] # Volume is a top-level column

    # Set the index name, which backtrader will use for the datetime axis
    df.index.name = 'Date'

    # The API might return slightly more data than requested, so trim it
    return df.tail(num_points)