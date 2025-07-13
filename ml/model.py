# ml/model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Note: You were importing from ml.feature_engineering. The correct relative import is .feature_engineering
from .feature_engineering import create_features 
from data.data_manager import load_dbn_to_df

def _generate_signals(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Generates a DataFrame of all potential BUY signals from the MA crossover."""
    signals = pd.DataFrame(index=df.index)
    # --- FIX: Assume 'close' is always present ---
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    # --- END FIX ---
    signals['signal'] = (df['short_ma'] > df['long_ma']).astype(int).diff()
    buy_signals = signals[signals['signal'] == 1.0].copy()
    print(f"Generated {len(buy_signals)} potential BUY signals.")
    return buy_signals

def _label_trades(signals: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Labels each trade based on the outcome of the reverse crossover exit."""
    print("Labeling trades...")
    labels = []
    # --- FIX: Assume 'close' is always present ---
    for signal_date in signals.index:
        entry_price = df.loc[signal_date]['close']
        future_data = df.loc[signal_date:].iloc[1:]
        sell_condition = (future_data['short_ma'] < future_data['long_ma'])
        exit_date = sell_condition.idxmax() if sell_condition.any() else None
        outcome = 0
        if exit_date and exit_date > signal_date:
            exit_price = future_data.loc[exit_date]['close']
            if exit_price > entry_price:
                outcome = 1
        elif not exit_date:
            last_price = future_data.iloc[-1]['close']
            if last_price > entry_price:
                outcome = 1
        labels.append(outcome)
    # --- END FIX ---
    
    signals['target'] = labels
    print("\n--- Target Label Distribution ---")
    print(signals['target'].value_counts(normalize=True))
    return signals

def train_and_save_model(training_data: pd.DataFrame, config):
    """Orchestrates the model training and saving pipeline."""
    print("--- Starting Meta-Model Training Pipeline ---")
    
    # The functions below now correctly modify the training_data DataFrame
    signals_df = _generate_signals(training_data, config.SHORT_WINDOW, config.LONG_WINDOW)
    labeled_signals = _label_trades(signals_df, training_data)
    
    # This call will now work because training_data has a 'close' column
    features_df = create_features(training_data) 
    model_data = labeled_signals.join(features_df, how='inner').dropna()
    
    X = model_data[config.FEATURE_LIST]
    y = model_data['target']
    
    if X.empty:
        print("CRITICAL ERROR: No data available for training.")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"\nTraining on {len(X_train)} signals, testing on {len(X_test)}.")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("\n--- Model Evaluation on Hold-Out Test Set ---")
    print(classification_report(y_test, model.predict(X_test)))
    
    joblib.dump(model, config.MODEL_FILE_PATH)
    print(f"\nModel successfully trained and saved to {config.MODEL_FILE_PATH}")