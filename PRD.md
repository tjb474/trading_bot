# PRD: ML-Enhanced Momentum Trading Bot

## 1. Overview & Goal

**Product Name**: ML-Gatekeeper Momentum Trader

**Goal**: To develop an automated trading system that executes a classic moving average momentum strategy. The entry signal from the strategy will be "gated" or "filtered" by a machine learning model. A trade is only executed if the technical signal is present AND the ML model predicts a sufficiently high probability of profit.

**Target User**: A developer with a strong ML background but new to algorithmic trading.

**Key Success Metrics**:

**Backtest Performance**: The combined strategy (Momentum + ML Filter) should outperform a simple momentum-only strategy in risk-adjusted returns (e.g., higher Sharpe Ratio).

**System Stability**: The system should run continuously without critical failures, with robust logging and error handling.

**Modularity**: The system should be designed so that the trading logic, ML model, and data sources can be easily swapped out.

----

## 2. Platform

----

## 3. System Architecture

We'll design a modular system that clearly separates concerns. The system will have two primary modes of operation: Backtesting and Live Trading.

**Core Components**:

**Data Handler**: Fetches and prepares market data (prices) and macroeconomic data.
**Feature Engineer**: Creates features for the ML model (e.g., volatility, RSI, macro indicators).

**ML Model**: A trained model that can be loaded to make predictions. It has two parts: train.py (offline) and predict.py (used by the bot).

**Strategy Logic**: Generates buy/sell signals based on moving averages.

**Execution Engine (Trader Bot)**: The main application loop. It gets signals from the strategy, validates them with the ML model, and places orders via the broker API.

**Backtesting Engine**: Simulates the strategy and execution on historical data to evaluate performance.

----

## 4. Base Template: Code Implementation (Python)

Let's lay out the file structure for our project.

```
trading_bot/
├── config.py                 # API keys and settings
├── data_handler.py           # Functions to get market/macro data
├── feature_engineering.py    # Functions to create model features
├── ml_model.py               # Code to train and predict with the ML model
├── trading_strategy.py       # The moving average logic
├── backtester.py             # Script to run backtests
└── trader.py                 # The main script for live/paper trading
```

**config.py**

Store your secrets and configurations here. Never commit this to Git.



```
# config.py

# --- IG API Credentials (from your Demo Account) ---
IG_USERNAME = "YOUR_IG_DEMO_USERNAME"
IG_PASSWORD = "YOUR_IG_DEMO_PASSWORD"
IG_API_KEY = "YOUR_IG_API_KEY"
IG_ACC_TYPE = "DEMO"  # Use "LIVE" for real money

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
```

**data_handler.py**

We'll use this for easy access to historical data for backtesting/training and Alpaca's API for live data.

```
# data_handler.py
import pandas as pd
# Note: The data from IG will be in a different format than yfinance.
# The 'Close' price is typically the midpoint of the bid/offer spread.

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
    
    # Convert the IG data into a familiar pandas DataFrame
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
```

**feature_engineering.py and ml_model.py**

This is your area of expertise. We'll create a simple placeholder.

The critical part is defining your target variable. A good start is a binary classification: "Will the price increase by X% within the next N days?"

```
# ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from data_handler import get_historical_data

# --- Feature Engineering (can be in its own file) ---
def create_features(df):
    """
    Create features for the ML model.
    For this example, we'll just use price change and volatility.
    """
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    # Add macroeconomic features here
    # e.g., df['VIX'] = ...
    return df.dropna()

# --- Target Variable Definition ---
def create_target(df, period=5, gain_threshold=0.02):
    """
    Create the target variable.
    Returns 1 if the price increases by 'gain_threshold' in the next 'period' days.
    """
    df['future_price'] = df['Close'].shift(-period)
    df['target'] = (df['future_price'] > df['Close'] * (1 + gain_threshold)).astype(int)
    return df.dropna()

# --- Model Training & Prediction ---
def train_model(symbol='SPY', start_date='2010-01-01', end_date='2022-12-31'):
    """
    A simple function to train and save the model.
    """
    print("Training model...")
    # 1. Get and prepare data
    data = get_historical_data(symbol, start_date, end_date)
    data = create_features(data)
    data = create_target(data)

    # 2. Define features (X) and target (y)
    features = ['returns', 'volatility'] # Add your features here
    X = data[features]
    y = data['target']

    # 3. Split data (ensure no lookahead bias)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # 6. Save model
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

def get_prediction(model, live_features):
    """
    Gets a probability prediction from the loaded model.
    live_features should be a DataFrame with one row of current features.
    """
    # The model expects a 2D array
    prediction_proba = model.predict_proba(live_features)
    # Return the probability of the positive class (class 1)
    return prediction_proba[0][1]

# You would run this once offline:
# if __name__ == '__main__':
#     train_model()
```


**trading_strategy.py**

This contains the pure trading logic, decoupled from execution.

```
# trading_strategy.py
import pandas as pd

def generate_signal(df, short_window, long_window):
    """
    Generates a trading signal based on a moving average crossover.
    - Returns 'BUY' if the short MA crosses above the long MA.
    - Returns 'SELL' to close position if short MA crosses below.
    - Returns 'HOLD' otherwise.
    """
    df['short_ma'] = df['Close'].rolling(window=short_window).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window).mean()
    
    # Get the last two rows to check for a crossover
    last_row = df.iloc[-1]
    previous_row = df.iloc[-2]

    # Buy signal: Short MA crosses above Long MA
    if previous_row['short_ma'] <= previous_row['long_ma'] and last_row['short_ma'] > last_row['long_ma']:
        return 'BUY'
    
    # Sell signal: Short MA crosses below Long MA
    if previous_row['short_ma'] >= previous_row['long_ma'] and last_row['short_ma'] < last_row['long_ma']:
        return 'SELL'
        
    return 'HOLD'
```

**trader.py (The Live/Paper Bot)**

This script ties everything together for live execution.

```
# trader.py
import time
import joblib
from trading_ig import IGService
import config
from data_handler import get_ig_historical_data
from trading_strategy import generate_signal
from ml_model import create_features, get_prediction

def run_trader():
    # 1. Setup IG Service
    ig_service = IGService(
        config.IG_USERNAME, 
        config.IG_PASSWORD, 
        config.IG_API_KEY, 
        config.IG_ACC_TYPE
    )
    ig_service.create_session()
    model = joblib.load(config.MODEL_PATH)
    print("Trader connected to IG. Awaiting signals...")

    while True:
        # 2. Check if we have an open position for this EPIC
        open_positions = ig_service.fetch_open_positions()
        position_for_epic = open_positions[open_positions['epic'] == config.EPIC]
        in_position = not position_for_epic.empty

        # 3. Get Data and Generate Signal
        data = get_ig_historical_data(ig_service, config.EPIC, resolution='D', num_points=config.LONG_WINDOW + 2)
        technical_signal = generate_signal(data, config.SHORT_WINDOW, config.LONG_WINDOW)
        print(f"Technical Signal: {technical_signal}")
        
        # 4. Execution Logic
        if technical_signal == 'BUY' and not in_position:
            print("Buy signal detected. Consulting ML model...")
            
            # (ML logic remains the same)
            live_features_df = create_features(data.tail(21))
            live_features = live_features_df[['returns', 'volatility']].iloc[[-1]]
            profit_prob = get_prediction(model, live_features)
            print(f"ML Model Profit Probability: {profit_prob:.2f}")

            if profit_prob > config.PROBABILITY_THRESHOLD:
                print("Placing long position.")
                ig_service.create_open_position(
                    epic=config.EPIC,
                    direction='BUY',
                    currency_code='GBP',
                    order_type='MARKET',
                    size=config.STAKE_SIZE,
                    force_open=True,
                    guaranteed_stop=False, # Can't use on market orders
                    stop_distance=config.STOP_DISTANCE,
                    limit_distance=config.LIMIT_DISTANCE
                )

        elif technical_signal == 'SELL' and in_position:
            # We only close if our position is a long ('BUY') position.
            if position_for_epic.iloc[0]['direction'] == 'BUY':
                print("Sell signal detected. Closing position.")
                deal_id = position_for_epic.iloc[0]['dealId']
                ig_service.close_position(deal_id=deal_id, direction='SELL', order_type='MARKET', size=config.STAKE_SIZE)
        
        else:
            print("Holding or no signal.")
            
        time.sleep(60 * 15) # Check every 15 minutes

if __name__ == '__main__':
    run_trader()

```

# 5. Setting Up Backtesting

For backtesting, backtrader is a robust and popular Python library. It handles a lot of the complexity of simulating a strategy.

**backtester.py**

This script will use backtrader to simulate our full strategy.

```
# backtester.py
import backtrader as bt
import pandas as pd
import joblib
import config
from data_handler import get_historical_data
from ml_model import create_features # Re-use the same feature logic

class MLStrategy(bt.Strategy):
    params = (
        ('short_window', config.SHORT_WINDOW),
        ('long_window', config.LONG_WINDOW),
        ('model_path', config.MODEL_PATH),
        ('prob_threshold', config.PROBABILITY_THRESHOLD),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.short_window
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.long_window
        )
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
        # Load the ML model
        self.model = joblib.load(self.p.model_path)
        
        # Prepare data for feature engineering
        # Convert backtrader data to a pandas DataFrame for our existing functions
        self.df = pd.DataFrame({
            'Open': self.data.open.get(size=len(self.data)),
            'High': self.data.high.get(size=len(self.data)),
            'Low': self.data.low.get(size=len(self.data)),
            'Close': self.data.close.get(size=len(self.data)),
            'Volume': self.data.volume.get(size=len(self.data)),
        }, index=self.data.datetime.get(size=len(self.data)))
        
        self.features_df = create_features(self.df)

    def next(self):
        # Check if we are in the market
        if not self.position:
            # Buy signal: short MA crosses above long MA
            if self.crossover > 0:
                print(f"{self.data.datetime.date(0)}: Buy signal detected. Consulting ML model...")
                
                # Get current date's features for the model
                current_date = self.data.datetime.date(0)
                try:
                    # Match date and get features. Use .loc for safety.
                    live_features = self.features_df.loc[current_date.strftime('%Y-%m-%d')][['returns', 'volatility']].values.reshape(1, -1)
                    
                    # Get prediction
                    profit_prob = self.model.predict_proba(live_features)[0][1]
                    print(f"ML Model Profit Probability: {profit_prob:.2f}")

                    if profit_prob > self.p.prob_threshold:
                        print("ML model approves. Placing BUY order.")
                        self.buy()
                    else:
                        print("ML model disapproves. No trade.")
                except KeyError:
                    # This can happen if features can't be calculated for a given day
                    print(f"Could not find features for {current_date}. Skipping.")

        # Sell signal: short MA crosses below long MA
        elif self.crossover < 0:
            print(f"{self.data.datetime.date(0)}: Sell signal. Closing position.")
            self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(MLStrategy)

    # Get data
    data = get_historical_data(config.SYMBOL, '2023-01-01', '2023-12-31')
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set starting cash
    cerebro.broker.setcash(100000.0)
    
    # Set commission
    cerebro.broker.setcommission(commission=0.001) # Simulate broker fees

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Run the backtest
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Print analysis
    strat = results[0]
    print('Sharpe Ratio:', strat.analyzers.sharpe_ratio.get_analysis()['sharperatio'])
    print('Max Drawdown:', strat.analyzers.drawdown.get_analysis()['max']['drawdown'])

    # Plot the results
    cerebro.plot()
```

# 6. Next Steps & Important Considerations

**Risk Management (CRITICAL)**:

**Position Sizing**: The template buys a fixed number of shares (qty=10). This is naive. You should calculate position size based on a fraction of your portfolio and the asset's volatility (e.g., risk 1% of your capital per trade).

**Stop-Losses**: The current strategy only exits on a crossover signal. You must implement a stop-loss to exit a losing trade before it becomes catastrophic. This can be a fixed percentage or based on indicators like ATR (Average True Range).

**Refine the ML Model**:

**Feature Engineering**: This is where you can get a huge edge. Add macroeconomic data (e.g., from FRED: interest rates, VIX, unemployment), sentiment data, or alternative data.

**Model Selection**: Try more advanced models like LGBMClassifier, XGBoost, or even a simple neural network.

**Overfitting**: This is the biggest danger. Be ruthless with your train/validation/test splits. Always split by time, never shuffle financial time-series data. Use walk-forward validation for a more realistic performance estimate.

**Deployment & Monitoring**:

**Cloud Server**: To run this 24/7, you'll need to deploy it on a cloud server (e.g., a small AWS EC2 or DigitalOcean droplet).

**Logging**: Log everything: signals, model probabilities, orders sent, errors from the API. Send these logs somewhere you can see them (e.g., Papertrail, AWS CloudWatch).

**Health Checks**: Set up a system to notify you if your bot crashes.

**Paper Trade Extensively**: Before you put a single dollar of real money on the line, run this bot on your Alpaca paper trading account for at least a month. The real world has frictions (slippage, latency, API outages) that a backtest cannot perfectly simulate.

# 7. Backtesting Methodology: The "Commission-as-Spread" Model

## 7.1. The Challenge & Solution

Standard backtesting tools like `backtrader` simulate buying/selling assets, which doesn't directly map to the stake-based nature of spread betting. To overcome this without building a custom backtester, we will adopt the **"Commission-as-Spread"** model.

This approach simulates the primary cost of spread betting—the broker's spread—as a transaction commission within the `backtrader` environment. It provides a fast, effective, and realistic way to evaluate the strategy's historical performance.

## 7.2. The Model Explained

The spread is the fee you pay to open and close a trade. We can express this fee as a percentage of the instrument's price and apply it as a commission in our simulation.

*   **Formula:** `Simulated Commission = Spread in Points / Instrument Price`

*   **Example:**
    *   **Instrument:** S&P 500, trading at a price of **4500.0**
    *   **Broker Spread:** **0.8 points**
    *   **Calculation:** `0.8 / 4500.0 = 0.000177`
    *   **Result:** We will use a commission of `0.0177%` for each transaction (open and close).

## 7.3. Implementation in `backtrader`

This model is implemented by setting the commission on the `cerebro` engine's broker before running the backtest.

```python
# In backtester.py

# --- Set up Cerebro engine ---
cerebro = bt.Cerebro()
# ... (add strategy and data)

# --- Simulate the spread as a commission ---
instrument_price_approx = 4500.0  # An average price for the backtest period
spread_points = 0.8
commission_value = spread_points / instrument_price_approx

cerebro.broker.setcommission(commission=commission_value)

# --- Run the backtest ---
print(f"Backtesting with simulated commission of: {commission_value:.5f}")
results = cerebro.run()

## 7.4. Limitations of this Model
This model provides a strong approximation for initial validation but does not natively account for secondary costs and market frictions, such as:

Overnight Financing: Daily charges for holding positions open.

Variable Spreads: Spreads can widen during volatile market conditions.

Slippage: The difference between the expected trade price and the actual execution price.
