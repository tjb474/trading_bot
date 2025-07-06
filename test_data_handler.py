# test_data_handler.py

import config
from data_handler import get_ig_historical_data
from trading_ig import IGService
import pandas as pd

def check_data_handler():
    """
    A standalone test function to verify the data_handler works.
    """
    print("--- Starting Data Handler Test ---")
    
    # 1. Check if config variables are loaded correctly
    if not all([config.IG_USERNAME, config.IG_PASSWORD, config.IG_API_KEY]):
        print("🛑 ERROR: Missing IG credentials in your .env file or config.py.")
        print("Please ensure IG_USERNAME, IG_PASSWORD, and IG_API_KEY are set.")
        return

    print("✅ Config credentials loaded.")

    print(f"DEBUG: Username loaded as: '{config.IG_USERNAME}'") # Add this line
    print(f"DEBUG: Password loaded as: '{config.IG_PASSWORD}'") # And this one

    # Print non-sensitive config variables for debugging
    print(f"IG_ACC_TYPE: {config.IG_ACC_TYPE}")
    print(f"EPIC: {config.EPIC}")
    print(f"STAKE_SIZE: {config.STAKE_SIZE}")
    print(f"STOP_DISTANCE: {config.STOP_DISTANCE}")
    print(f"LIMIT_DISTANCE: {config.LIMIT_DISTANCE}")
    print(f"SHORT_WINDOW: {config.SHORT_WINDOW}")
    print(f"LONG_WINDOW: {config.LONG_WINDOW}")
    print(f"MODEL_PATH: {config.MODEL_PATH}")
    print(f"PROBABILITY_THRESHOLD: {config.PROBABILITY_THRESHOLD}")
    
    try:
        # 2. Connect to IG Service
        print(f"Attempting to connect to IG as '{config.IG_USERNAME}' on {config.IG_ACC_TYPE} account...")
        ig_service = IGService(
            config.IG_USERNAME, 
            config.IG_PASSWORD, 
            config.IG_API_KEY, 
            config.IG_ACC_TYPE
        )
        ig_service.create_session()
        print("✅ Successfully connected to IG.")

        # 3. Call the function we want to test
        epic_to_test = config.EPIC
        print(f"Fetching historical data for EPIC: {epic_to_test}...")
        
        df = get_ig_historical_data(ig_service, epic=epic_to_test, resolution='H', num_points=100)

        # 4. Validate the output
        if df.empty:
            print(f"🛑 ERROR: The function returned an empty DataFrame.")
            print(f"Check if the EPIC '{epic_to_test}' is correct.")
            return

        print("✅ Data received successfully!")
        print(f"Shape of the DataFrame: {df.shape}")
        print("\n--- First 5 Rows ---")
        print(df.head())
        print("\n--- Last 5 Rows ---")
        print(df.tail())
        
        # Check for expected columns
        expected_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in df.columns for col in expected_cols):
             print("\n✅ All expected columns (Open, High, Low, Close) are present.")
        else:
             print("\n🛑 ERROR: DataFrame is missing expected columns.")


    except Exception as e:
        print(f"🛑 AN ERROR OCCURRED: {e}")
        print("--- Troubleshooting ---")
        print("- Check your internet connection.")
        print("- Verify your IG_USERNAME, IG_PASSWORD, and IG_API_KEY are correct in .env")
        print("- Ensure your DEMO account is active.")

    print("\n--- Test Finished ---")


if __name__ == '__main__':
    check_data_handler()