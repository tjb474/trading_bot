# common/data_manager.py

import os
import pandas as pd
from typing import Tuple

def load_ohlc_data(file_path: str) -> pd.DataFrame:
    """Loads OHLC data from a CSV file and prepares it."""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
        # --- FIX ---
        # Ensure ALL column names are lowercase for consistency.
        # This is the only transformation we need.
        df.columns = [col.lower() for col in df.columns]
        # --- END FIX ---
        print(f"Data loaded successfully from {file_path}")
        print(f"Data columns are: {df.columns.tolist()}") # Add a debug print
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()

def split_data(df: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets."""
    split_index = int(len(df) * split_ratio)
    training_df = df.iloc[:split_index]
    testing_df = df.iloc[split_index:]
    print(f"Data split. Training set: {len(training_df)} bars, Testing set: {len(testing_df)} bars.")
    return training_df, testing_df

def load_dbn_to_df(dbn_path):
    """
    Loads a Databento .dbn file and returns a pandas DataFrame.
    """
    ext = os.path.splitext(dbn_path)[1].lower()
    if ext != '.dbn':
        raise ValueError(f"File {dbn_path} is not a .dbn file.")
    try:
        from databento import DBNStore
    except ImportError:
        raise ImportError("databento package is not installed. Please install it with 'pip install databento'.")
    store = DBNStore.from_file(dbn_path)
    df = store.to_df()
    return df