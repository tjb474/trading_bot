# ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import config
# We will call the data handler from the training script, not here.

def create_features(df):
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    return df.dropna()

def create_target(df, period=5, gain_threshold=0.02):
    df['future_price'] = df['Close'].shift(-period)
    df['target'] = (df['future_price'] > df['Close'] * (1 + gain_threshold)).astype(int)
    return df.dropna()

def train_model(data, model_path):
    print("Training model...")
    data = create_features(data)
    data = create_target(data)

    features = ['returns', 'volatility']
    X = data[features]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if len(X_train) == 0:
        print("Not enough data to create a training set.")
        return

    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy on test set: {accuracy:.2f}")

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")