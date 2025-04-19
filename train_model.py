import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load Historical Data
# Specify the date format to ensure consistent parsing
data = pd.read_csv('NVDA_5min_historical_data.csv', parse_dates=['date'], date_format='%Y%m%d %H:%M:%S US/Eastern')
print("Data shape:", data.shape)
print("First few rows:\n", data.head())

# Step 2: Calculate Indicators and Features
def calculate_features(df):
    """Calculate technical indicators and features for the dataset."""
    df['sma20'] = df['close'].rolling(window=20).mean()  # 20-period Simple Moving Average
    df['std20'] = df['close'].rolling(window=20).std()   # 20-period Standard Deviation
    df['upper_band'] = df['sma20'] + 2 * df['std20']     # Upper Bollinger Band
    df['lower_band'] = df['sma20'] - 2 * df['std20']     # Lower Bollinger Band
    df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)  # 14-period ATR
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)  # 14-period RSI
    return df

data = calculate_features(data)

# Step 3: Generate Signals and Simulate Trades
def generate_signals_and_labels(df):
    """Generate trading signals and label them as profitable (1) or unprofitable (0)."""
    signals = []
    labels = []
    long_crossings = 0
    short_crossings = 0
    for i in range(20, len(df) - 1):  # Start after indicators are calculated
        close = df['close'].iloc[i]
        lower_band = df['lower_band'].iloc[i]
        upper_band = df['upper_band'].iloc[i]
        sma20 = df['sma20'].iloc[i]
        std20 = df['std20'].iloc[i]
        atr = df['atr'].iloc[i]
        rsi = df['rsi'].iloc[i]
        volume = df['volume'].iloc[i]
        mean_vol = df['volume'].iloc[i-5:i].mean()  # 5-period mean volume
        vol_z = (volume - mean_vol) / mean_vol if mean_vol != 0 else 0  # Volume Z-score

        if close < lower_band:
            long_crossings += 1
            if vol_z > 0.05:  # Long signal condition
                entry_price = close
                stop_loss = entry_price - 2.5 * atr
                take_profit = entry_price + 5 * atr
                future_prices = df['close'].iloc[i+1:]
                hit_tp = any(future_prices >= take_profit)
                hit_sl = any(future_prices <= stop_loss)
                if hit_tp and (not hit_sl or future_prices[future_prices >= take_profit].index[0] < 
                               future_prices[future_prices <= stop_loss].index[0]):
                    label = 1  # Profitable
                else:
                    label = 0  # Unprofitable
                deviation = (close - sma20) / std20
                atr_norm = atr / close
                signals.append([deviation, vol_z, atr_norm, rsi])
                labels.append(label)
                print(f"Long signal at index {i}: close={close}, lower_band={lower_band}, vol_z={vol_z}, label={label}")

        elif close > upper_band:
            short_crossings += 1
            if vol_z < -0.05:  # Short signal condition
                entry_price = close
                stop_loss = entry_price + 2.5 * atr
                take_profit = entry_price - 5 * atr
                future_prices = df['close'].iloc[i+1:]
                hit_tp = any(future_prices <= take_profit)
                hit_sl = any(future_prices >= stop_loss)
                if hit_tp and (not hit_sl or future_prices[future_prices <= take_profit].index[0] < 
                               future_prices[future_prices >= stop_loss].index[0]):
                    label = 1  # Profitable
                else:
                    label = 0  # Unprofitable
                deviation = (close - sma20) / std20
                atr_norm = atr / close
                signals.append([deviation, vol_z, atr_norm, rsi])
                labels.append(label)
                print(f"Short signal at index {i}: close={close}, upper_band={upper_band}, vol_z={vol_z}, label={label}")

    print(f"Price below lower band: {long_crossings} times")
    print(f"Price above upper band: {short_crossings} times")
    print(f"Total signals generated: {len(signals)}")
    return signals, labels

signals, labels = generate_signals_and_labels(data)

# Step 4: Prepare Data for Training
if len(signals) == 0:
    print("No signals generated. Check signal conditions or use a larger dataset.")
else:
    X = np.array(signals)  # Feature matrix
    y = np.array(labels)   # Label vector

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Step 6: Save the Trained Model
    joblib.dump(model, 'trading_signal_model.joblib')
    print("Model saved to 'trading_signal_model.joblib'")