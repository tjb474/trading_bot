import pandas as pd

def get_ig_historical_data(ig_service, epic, resolution='D', num_points=250):
    """
    Fetches historical price data from the IG API.
    Resolution can be 'D' (Day), 'H' (Hour), 'M' (Minute), etc.
    """
    response = ig_service.fetch_historical_prices_by_epic_and_num_points(
        epic=epic, 
        resolution=resolution, 
        num_points=num_points
    )
    prices = response['prices']
    df = pd.DataFrame({
        'Date': pd.to_datetime(prices['snapshotTime']),
        'Open': (prices['openPrice']['bid'] + prices['openPrice']['ask']) / 2,
        'High': (prices['highPrice']['bid'] + prices['highPrice']['ask']) / 2,
        'Low': (prices['lowPrice']['bid'] + prices['lowPrice']['ask']) / 2,
        'Close': (prices['closePrice']['bid'] + prices['closePrice']['ask']) / 2,
    })
    df.set_index('Date', inplace=True)
    return df
