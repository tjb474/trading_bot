# common/data_manager.py

import pandas as pd
from typing import Tuple

def load_ohlc_data(file_path: str) -> pd.DataFrame:
    """Loads OHLC data from a CSV file and prepares it."""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
        # --- DEFINITIVE FIX ---
        # Ensure ALL column names are ALWAYS lowercase.
        # No more renaming to 'Close'.
        df.columns = [col.lower() for col in df.columns]
        # --- END FIX ---
        print(f"Data loaded successfully from {file_path}")
        print(f"Data columns are: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()

# The split_data function is correct and does not need changes.
def split_data(df: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets."""
    split_index = int(len(df) * split_ratio)
    training_df = df.iloc[:split_index].copy() # Use .copy() to avoid SettingWithCopyWarning
    testing_df = df.iloc[split_index:].copy()  # Use .copy() to avoid SettingWithCopyWarning
    print(f"Data split. Training set: {len(training_df)} bars, Testing set: {len(testing_df)} bars.")
    return training_df, testing_df