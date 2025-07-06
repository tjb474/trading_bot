import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from data_handler import get_ig_historical_data
from feature_engineering import create_features
import config

def create_target(df, period=5, gain_threshold=0.02):
    df['future_price'] = df['Close'].shift(-period)
    df['target'] = (df['future_price'] > df['Close'] * (1 + gain_threshold)).astype(int)
    return df.dropna()

def train_model(symbol='SPY', start_date='2010-01-01', end_date='2022-12-31'):
    print("Training model...")
    data = get_ig_historical_data(symbol, start_date, end_date)
    data = create_features(data)
    data = create_target(data)
    features = ['returns', 'volatility']
    X = data[features]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

def get_prediction(model, live_features):
    prediction_proba = model.predict_proba(live_features)
    return prediction_proba[0][1]
