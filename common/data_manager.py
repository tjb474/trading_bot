# common/data_manager.py

import logging
import pandas as pd
import pytz
from typing import Tuple

logger = logging.getLogger(__name__)


def convert_to_eastern_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the DataFrame's index is converted to US/Eastern timezone.
    This is a prerequisite for both backtesting and training to ensure consistency.
    """
    df = df.copy() # Avoid SettingWithCopyWarning
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        logger.info(f"Data timezone before conversion: {df.index.tz}")
        eastern_tz = pytz.timezone('US/Eastern')
        
        if 'UTC' in str(df.index.tz):
            logger.info("Converting UTC data to Eastern timezone...")
            df.index = df.index.tz_convert(eastern_tz)
            logger.info(f"Data converted to: {df.index.tz}")
        elif str(df.index.tz) != str(eastern_tz):
            # Handle cases where it's already localized but not to Eastern
            logger.info(f"Converting {df.index.tz} data to Eastern timezone...")
            df.index = df.index.tz_convert(eastern_tz)
            logger.info(f"Data converted to: {df.index.tz}")
        else:
            logger.info("Data is already in US/Eastern timezone.")
    else:
        logger.warning("Data has no timezone information. Assuming it represents US/Eastern time and localizing.")
        eastern_tz = pytz.timezone('US/Eastern')
        df.index = df.index.tz_localize(eastern_tz)
    
    return df

def load_ohlc_data(file_path: str) -> pd.DataFrame:
    """
    Loads OHLC data from CSV or DBN files and ensures Eastern timezone.
    
    CRITICAL: All data must be in US/Eastern timezone for proper OR calculations.
    """
    print(f"🔍 DEBUG: load_ohlc_data called with file_path: {file_path}")
    print(f"🔍 DEBUG: File ends with .dbn: {file_path.endswith('.dbn')}")
    
    try:
        eastern_tz = pytz.timezone('US/Eastern')
        print(f"🔍 DEBUG: Created eastern_tz: {eastern_tz}")
        
        if file_path.endswith('.dbn'):
            # Handle DBN files with databento
            print(f"Loading DBN file: {file_path}")
            try:
                import databento as db
                spy_data = db.DBNStore.from_file(file_path)
                df = spy_data.to_df()
            except ImportError:
                raise ImportError("databento not installed. Please install: pip install databento")
            
            # FORCE conversion to Eastern timezone - OUTSIDE the try-except block
            print(f"Initial timezone: {df.index.tz}")
            if df.index.tz is not None:
                print(f"Converting timezone from {df.index.tz} to US/Eastern")
                df.index = df.index.tz_convert(eastern_tz)
            else:
                print("No timezone found, localizing as UTC then converting to Eastern")
                df.index = df.index.tz_localize('UTC').tz_convert(eastern_tz)
            
            print(f"Final timezone: {df.index.tz}")
            print(f"Sample timestamp: {df.index[0]}")
            
            # Verify the conversion worked
            if 'US/Eastern' not in str(df.index.tz):
                raise ValueError(f"Timezone conversion failed! Still have: {df.index.tz}")
                
        else:
            # Handle CSV files 
            print(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
            
            # Ensure Eastern timezone for CSV data
            if df.index.tz is None:
                print("CSV data has no timezone, assuming Eastern timezone")
                df.index = df.index.tz_localize(eastern_tz)
            elif df.index.tz != eastern_tz:
                print(f"Converting CSV timezone from {df.index.tz} to US/Eastern")
                df.index = df.index.tz_convert(eastern_tz)
        
        # Ensure ALL column names are ALWAYS lowercase
        df.columns = [col.lower() for col in df.columns]
        
        print(f"✅ Data loaded successfully from {file_path}")
        print(f"📊 Shape: {df.shape}")
        print(f"🕐 Timezone: {df.index.tz}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"📋 Columns: {df.columns.tolist()}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# The split_data function is correct and does not need changes.
def split_data(df: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets."""
    split_index = int(len(df) * split_ratio)
    training_df = df.iloc[:split_index].copy() # Use .copy() to avoid SettingWithCopyWarning
    testing_df = df.iloc[split_index:].copy()  # Use .copy() to avoid SettingWithCopyWarning
    print(f"Data split. Training set: {len(training_df)} bars, Testing set: {len(testing_df)} bars.")
    return training_df, testing_df