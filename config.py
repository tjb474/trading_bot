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
