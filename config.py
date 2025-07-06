
import os
from dotenv import load_dotenv

# Explicitly load .env from the current directory (where this file is located)
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# --- IG API Credentials (loaded from .env) ---
IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")
IG_API_KEY = os.getenv("IG_API_KEY")
IG_ACC_TYPE = os.getenv("IG_ACC_TYPE", "DEMO") # Defaults to DEMO if not set
# --- The rest of your config file continues below ---

# --- Trading Strategy Parameters ---
# Find the EPIC for the instrument you want to trade on the IG platform.
# This example is for the S&P 500 (Daily Funded Bet)
EPIC = 'IX.D.SPTRD.DAILY.IP'
STAKE_SIZE = 1 # £1 per point of movement
STOP_DISTANCE = 15 # Stop loss 15 points away from entry
LIMIT_DISTANCE = 30 # Take profit 30 points away

SHORT_WINDOW = 40
LONG_WINDOW = 100

# --- ML Model Parameters ---
MODEL_PATH = "profit_model.joblib"
PROBABILITY_THRESHOLD = 0.60
