# train_model_script.py
from trading_ig import IGService
import config
from data_handler import get_ig_historical_data
from ml_model import train_model

if __name__ == '__main__':
    print("Connecting to IG to fetch training data...")
    ig_service = IGService(config.IG_USERNAME, config.IG_PASSWORD, config.IG_API_KEY, config.IG_ACC_TYPE)
    ig_service.create_session()
    
    # Fetch 5 years of data for a decent training set
    data = get_ig_historical_data(ig_service, config.EPIC, resolution='D', num_points=5*252)
    
    if not data.empty:
        print(f"Successfully fetched {len(data)} data points for training.")
        train_model(data, config.MODEL_PATH)
    else:
        print("Failed to fetch training data.")