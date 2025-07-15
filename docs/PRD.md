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

**Feature Engineer**: Creates features for the ML model (e.g., volatility, RSI, macro indicators).

**ML Model**: A trained model that can be loaded to make predictions. It has two parts: train.py (offline) and predict.py (used by the bot).

**Strategy Logic**: Generates buy/sell signals based on moving averages.

**Execution Engine (Trader Bot)**: The main application loop. It gets signals from the strategy, validates them with the ML model, and places orders via the broker API.

**Backtesting Engine**: Simulates the strategy and execution on historical data to evaluate performance.

----




**.env**

Environment variables file (not tracked by git). Stores your IG credentials and other secrets.

**.gitignore**

Should include venv/ and .env to prevent committing secrets and virtual environments.

**venv/**

Python virtual environment directory (ignored by git).




# 6. Next Steps & Important Considerations

## Core Meta-Labeling & ML Recommendations (from community best practices)

**Meta-Labeling Principle:**
Use machine learning to filter and improve an existing strategy's edge, not to create an edge from scratch. ML is best used as a supervised filter (meta-label) on top of a base strategy that already has positive expectancy.

**Meta-Labeling Workflow:**
1. **Run Your Primary Strategy:** Generate and log all trade signals, including entry/exit times and outcomes. Record every signal, even overlapping ones, to ensure the ML filter can learn from all possible trades.
2. **Label the Signals:** Assign a binary label (1 = profitable, 0 = not) to each trade, based on a profit threshold and/or time-to-profit. Binary classification is recommended for simplicity and effectiveness.
3. **Gather Features:** For each signal, collect features available at the time of entry (no future data). Use a diverse set: price, volume, volatility, macro, order book, and regime/state features.
4. **Train the Meta Model:** Use the features and labels to train a classifier (e.g., XGBoost, Random Forest, Logistic Regression). Train multiple models and use an ensemble for robustness. Calibrate model probabilities for real-world hit rates.
5. **Deploy:** In live trading, pass each new signal through the trained meta model. Only execute trades above a chosen confidence threshold.

**Feature Engineering Tips:**
- Use diverse, robust features (price, volume, volatility, macro, order book, regime/state).
- Prioritize features that remain relevant over time; test for non-stationarity.
- Use feature selection (RFECV, mutual information, model importance) to drop redundant/noisy features.
- Strictly avoid data leakage: all features must be available at signal time.

**Modeling & Validation:**
- Balance classes if labels are imbalanced.
- Use multiple model types and an ensemble meta-model for final predictions.
- Calibrate model outputs (Platt scaling, isotonic regression) for reliable probabilities.
- Use robust cross-validation: nested CV for IID trades, combinatorial purged CV for non-IID/time series.
- Always test out-of-sample and compare meta-labeled vs. raw strategy performance (Sharpe, drawdown, win rate, precision, etc.).

**Backtesting & Pitfalls:**
- Prevent overfitting: use robust CV, never shuffle time series, and test on truly unseen data.
- Prevent data leakage: ensure all features are strictly time-aligned, lagged if needed, and no future info is used.
- Avoid unstable or redundant features; use feature selection and stability checks.
- Ensure sample size is large enough (ideally 1000+ trades, 5000+ preferred).
- Always include slippage, fees, and realistic fills in backtests.

**Deployment & Monitoring:**
- Retrain models regularly (weekly/monthly or as new data accumulates).
- Log all signals, model probabilities, orders, and errors for monitoring and debugging.
- Set up health checks and alerts for bot failures.
- Paper trade extensively before going live to account for real-world frictions.

**General Advice:**
- Meta-labeling amplifies an existing edge; it cannot create one from random signals.
- Keep the pipeline modular and reusable for new strategies.
- Use a separate model per asset if needed, or include asset as a categorical feature.
- Use sockets or APIs to integrate Python ML models with trading platforms (e.g., Ninjatrader).

---

**Risk Management (CRITICAL):**

**Position Sizing:** The template buys a fixed number of shares (qty=10). This is naive. You should calculate position size based on a fraction of your portfolio and the asset's volatility (e.g., risk 1% of your capital per trade).

**Stop-Losses:** The current strategy only exits on a crossover signal. You must implement a stop-loss to exit a losing trade before it becomes catastrophic. This can be a fixed percentage or based on indicators like ATR (Average True Range).

**Refine the ML Model:**

**Feature Engineering:** This is where you can get a huge edge. Add macroeconomic data (e.g., from FRED: interest rates, VIX, unemployment), sentiment data, or alternative data.

**Model Selection:** Try more advanced models like LGBMClassifier, XGBoost, or even a simple neural network.

**Overfitting:** This is the biggest danger. Be ruthless with your train/validation/test splits. Always split by time, never shuffle financial time-series data. Use walk-forward validation for a more realistic performance estimate.

**Deployment & Monitoring:**

**Cloud Server:** To run this 24/7, you'll need to deploy it on a cloud server (e.g., a small AWS EC2 or DigitalOcean droplet).

**Logging:** Log everything: signals, model probabilities, orders sent, errors from the API. Send these logs somewhere you can see them (e.g., Papertrail, AWS CloudWatch).

**Health Checks:** Set up a system to notify you if your bot crashes.

**Paper Trade Extensively:** Before you put a single dollar of real money on the line, run this bot on your Alpaca paper trading account for at least a month. The real world has frictions (slippage, latency, API outages) that a backtest cannot perfectly simulate.

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
