# common/config.py

import os
from dotenv import load_dotenv

# --- Base Directory ---
# This ensures that paths are correct regardless of where the script is run from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Path Configurations ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_DIR, 'SPY_1min_full.csv')
MODEL_FILE_PATH = os.path.join(DATA_DIR, 'profit_model.joblib')

# --- Data & Splitting Parameters ---
TRAIN_TEST_SPLIT_RATIO = 0.8

# --- Trading Strategy Parameters ---
SHORT_WINDOW = 40
LONG_WINDOW = 100

# --- ML Model Parameters ---
PROBABILITY_THRESHOLD = 0.60
FEATURE_LIST = ['returns', 'volatility', 'rsi']
FEATURE_VOL_WINDOW = 20
FEATURE_RSI_WINDOW = 14

# --- Backtester Parameters ---
INITIAL_CASH = 100000.0
STAKE_SIZE = 100
COMMISSION_SPREAD_POINTS = 0.01

# --- API Credentials (from .env) ---
IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")
IG_API_KEY = os.getenv("IG_API_KEY")
