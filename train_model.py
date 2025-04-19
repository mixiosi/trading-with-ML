import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import pandas.api.types # To check dtypes robustly
import traceback # For detailed error prints

# --- Configuration ---
# List of symbols to process
SYMBOLS = ['NVDA', 'SPY', 'RGTI', 'QBTS'] # Example: Added your symbols back
DATA_DIR = '.' # Directory where CSV files are located (e.g., './data/')
FILENAME_TEMPLATE = '{}_5min_historical_data.csv' # How your files are named

# Feature Calculation Parameters
SMA_WINDOW = 20
ATR_PERIOD = 14
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Signal Generation & Labeling Parameters
LOOKAHEAD_BARS = 12 # How many bars (e.g., 12 * 5min = 1 hour) to look ahead for TP/SL
SL_MULT = 2.0 # Stop Loss ATR multiplier
TP_MULT = 4.0 # Take Profit ATR multiplier
VOL_Z_LONG_THRESH = 0.05 # Volume Z-score threshold for long signals
VOL_Z_SHORT_THRESH = -0.05 # Volume Z-score threshold for short signals
MIN_SIGNALS_PER_SYMBOL = 10 # Minimum signals required from a symbol's data

# Model Training Parameters
TEST_SIZE = 0.2 # Proportion of data for testing (used for chronological split point)
RF_ESTIMATORS = 100
RANDOM_STATE = 42 # For reproducibility

# --- Feature Calculation ---
def calculate_features(df):
    """Calculate technical indicators and features for the dataset."""
    # Ensure input df columns are numeric before TALib calls
    # Although cleaned before, this is an extra safeguard if function is called elsewhere
    cols_to_check = ['high', 'low', 'close']
    for col in cols_to_check:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' inside calculate_features is not numeric. Attempting conversion.")
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=cols_to_check, inplace=True) # Drop rows if conversion failed

    if df.empty:
        print("Data empty after numeric check within calculate_features.")
        return df # Return empty df

    # Calculate indicators
    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']
    df['lower_band'] = df['sma'] - 2 * df['std']

    # Use .values to ensure numpy arrays for TALib
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values

    # Check if arrays are empty before passing to TALib
    if len(high_prices) == 0 or len(low_prices) == 0 or len(close_prices) == 0:
         print("Warning: Price arrays empty before TALib calculation.")
         # Assign NaN or handle appropriately
         df['atr'] = np.nan
         df['rsi'] = np.nan
         df['macd'] = np.nan
         df['macd_signal'] = np.nan
         df['macd_hist'] = np.nan
    else:
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
        df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist

    # Calculate Volume features
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    # Avoid division by zero robustly
    df['volume_z'] = (df['volume'] - df['volume_sma5']) / df['volume_sma5'].replace(0, 1e-10)

    # Add lagged features
    df['rsi_lag1'] = df['rsi'].shift(1)

    # Drop rows with NaN values resulting from indicator calculations
    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_dropped = initial_rows - len(df)
    # print(f"Dropped {rows_dropped} rows due to NaNs from indicators.")

    return df


# --- Signal Generation & Labeling ---
def generate_signals_and_labels(df, lookahead_bars, sl_mult, tp_mult, vol_z_long_thresh, vol_z_short_thresh):
    """Generate trading signals and label them as profitable (1) or unprofitable (0)."""
    signal_indices = []
    labels = []
    # entry_types = [] # Uncomment if needed later

    long_crossings = 0
    short_crossings = 0
    valid_indices = df.index

    # Stop 'lookahead_bars' before the end to allow future lookup
    if len(valid_indices) <= lookahead_bars:
        print("Warning: Not enough data points for lookahead period.")
        return [], []

    for idx_loc in range(len(valid_indices) - lookahead_bars):
        current_index = valid_indices[idx_loc] # Get the actual timestamp index
        current_data = df.loc[current_index]

        close = current_data['close']
        lower_band = current_data['lower_band']
        upper_band = current_data['upper_band']
        atr = current_data['atr']
        # volume = current_data['volume'] # Not directly used in condition
        # vol_sma = current_data['volume_sma5'] # Not directly used in condition
        vol_z = current_data['volume_z'] # Use pre-calculated Z-score

        # Ensure ATR is valid (not NaN or non-positive)
        if pd.isna(atr) or atr <= 1e-9:
            continue

        signal_generated = False
        entry_price = 0
        stop_loss = 0.0
        take_profit = 0.0
        label = 0
        entry_type = None

        # --- Long Signal ---
        # Ensure lower_band is not NaN before comparison
        if not pd.isna(lower_band) and close < lower_band and not pd.isna(vol_z) and vol_z > vol_z_long_thresh:
            long_crossings += 1
            entry_price = close
            stop_loss = entry_price - sl_mult * atr
            take_profit = entry_price + tp_mult * atr
            entry_type = 'long'
            signal_generated = True

        # --- Short Signal ---
        # Ensure upper_band is not NaN before comparison
        elif not pd.isna(upper_band) and close > upper_band and not pd.isna(vol_z) and vol_z < vol_z_short_thresh:
            short_crossings += 1
            entry_price = close
            stop_loss = entry_price + sl_mult * atr
            take_profit = entry_price - tp_mult * atr
            entry_type = 'short'
            signal_generated = True

        # --- Labeling (if signal was generated) ---
        if signal_generated:
            # Get future data using iloc for positional slicing relative to current loop position
            future_data_slice = df.iloc[idx_loc + 1 : idx_loc + 1 + lookahead_bars]
            future_lows = future_data_slice['low']
            future_highs = future_data_slice['high']

            hit_tp = False
            hit_sl = False
            # Use pd.NaT (Not a Time) for timestamp initialization
            first_tp_time = pd.NaT
            first_sl_time = pd.NaT

            try: # Add try-except around comparisons for pinpointing the 'str' vs 'float' error
                if entry_type == 'long':
                    # Check TP hit: Find indices where future highs meet/exceed TP
                    tp_hits = future_highs[future_highs >= take_profit] # Potential TypeError here
                    if not tp_hits.empty:
                        hit_tp = True
                        first_tp_time = tp_hits.index[0] # Get the timestamp index of the first hit

                    # Check SL hit: Find indices where future lows meet/fall below SL
                    sl_hits = future_lows[future_lows <= stop_loss] # Potential TypeError here
                    if not sl_hits.empty:
                        hit_sl = True
                        first_sl_time = sl_hits.index[0] # Get the timestamp index of the first hit

                elif entry_type == 'short':
                    # Check TP hit: Find indices where future lows meet/fall below TP
                    tp_hits = future_lows[future_lows <= take_profit] # Potential TypeError here
                    if not tp_hits.empty:
                        hit_tp = True
                        first_tp_time = tp_hits.index[0] # Get the timestamp index of the first hit

                    # Check SL hit: Find indices where future highs meet/exceed SL
                    sl_hits = future_highs[future_highs >= stop_loss] # Potential TypeError here
                    if not sl_hits.empty:
                        hit_sl = True
                        first_sl_time = sl_hits.index[0] # Get the timestamp index of the first hit

            except TypeError as e:
                 # This block specifically catches the comparison error we've been seeing
                 print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 print(f"!!! CRITICAL TypeError during TP/SL check at index {current_index} for {entry_type} signal!")
                 print(f"    Error: {e}")
                 print(f"    This usually means 'future_lows' or 'future_highs' still contain non-numeric strings.")
                 # Inspect the data that caused the error
                 print(f"    future_lows dtype: {future_lows.dtype}, sample values: {future_lows.head(3).tolist()}")
                 print(f"    future_highs dtype: {future_highs.dtype}, sample values: {future_highs.head(3).tolist()}")
                 print(f"    take_profit value: {take_profit} (type: {type(take_profit)})")
                 print(f"    stop_loss value: {stop_loss} (type: {type(stop_loss)})")
                 print(f"    Skipping this specific signal due to error.")
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                 # Skip this signal if error occurs during check - don't append index/label
                 continue # Skip to the next iteration of the main loop (next potential signal point)
            except Exception as e_generic:
                 print(f"\n--- Unexpected Error during TP/SL check for index {current_index} ---")
                 print(f"Error Type: {type(e_generic).__name__}")
                 print(f"Error Message: {e_generic}")
                 traceback.print_exc()
                 print(f"--- Skipping this specific signal ---")
                 continue # Skip to the next iteration

            # --- Revised Label Determination Logic (Using Timestamps only if both hit) ---
            label = 0 # Default to unprofitable
            if hit_tp and hit_sl:
                 # Both were hit, compare the timestamps
                 if pd.notna(first_tp_time) and pd.notna(first_sl_time): # Ensure timestamps are valid
                     if first_tp_time <= first_sl_time:
                         label = 1 # TP hit first or simultaneously
                     # else: label remains 0 (SL hit first)
                 elif pd.notna(first_tp_time): # Should not happen if both hit_tp/sl true, but safety check
                     label = 1
                 # else label remains 0
            elif hit_tp:
                 # Only TP was hit within the lookahead period
                 label = 1
            # elif hit_sl: # Only SL was hit -> label remains 0
            # else: # Neither TP nor SL was hit -> label remains 0

            # Append results if no error occurred during TP/SL check
            signal_indices.append(current_index) # Store the actual timestamp index
            labels.append(label)
            # entry_types.append(entry_type)

    # --- End Loop ---

    print(f"Price below lower band checks (potential longs): {long_crossings} times")
    print(f"Price above upper band checks (potential shorts): {short_crossings} times")
    print(f"Total valid signals generated and labeled: {len(signal_indices)}")

    return signal_indices, labels # Return indices and labels

# --- Main Execution ---
all_signal_features = []
all_labels = []

print("Starting data processing...")
for symbol in SYMBOLS:
    print(f"\n--- Processing Symbol: {symbol} ---")
    file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(symbol))

    if not os.path.exists(file_path):
        print(f"Warning: Data file not found for {symbol} at {file_path}. Skipping.")
        continue

    try:
        # Load data
        data = pd.read_csv(file_path, parse_dates=['date'], date_format='%Y%m%d %H:%M:%S US/Eastern')
        print(f"Initial loaded rows: {len(data)}")

        # --- Robust Data Cleaning Step ---
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

        # Check if essential columns exist
        cols_to_clean = [col for col in ohlcv_cols if col in data.columns]
        missing_cols = [col for col in ohlcv_cols if col not in data.columns]
        if missing_cols:
             print(f"CRITICAL WARNING: Essential columns missing in {symbol} data: {missing_cols}. Skipping symbol.")
             continue # Skip to next symbol if core data is missing

        # Apply pd.to_numeric to force conversion, turning errors into NaN
        for col in cols_to_clean:
            if data[col].isnull().all(): # Handle fully empty columns if they occur
                 print(f"Warning: Column '{col}' in {symbol} is entirely null before conversion.")
                 data[col] = pd.to_numeric(data[col], errors='coerce') # Still convert to float type
            else:
                data[col] = pd.to_numeric(data[col], errors='coerce')


        # Report and drop rows where any essential numeric data became NaN
        nan_rows_mask = data[cols_to_clean].isnull().any(axis=1)
        nan_rows_count = nan_rows_mask.sum()
        if nan_rows_count > 0:
             print(f"Found {nan_rows_count} rows with non-numeric or missing values in OHLCV columns for {symbol}. Dropping these rows.")
             # Optional: Inspect rows before dropping
             # print("Sample rows with NaNs being dropped:\n", data[nan_rows_mask].head())
             data.dropna(subset=cols_to_clean, inplace=True)
        print(f"Data shape after coercing OHLCV to numeric & dropna: {data.shape}")

        # --- Strict Type Verification ---
        if data.empty:
            print(f"Warning: Data for {symbol} is empty after initial cleaning. Skipping.")
            continue

        numeric_check_passed = True
        for col in cols_to_clean:
            # Use pandas API for robust type checking
            if not pd.api.types.is_numeric_dtype(data[col]):
                print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"CRITICAL ERROR: Column '{col}' in {symbol} is NOT numeric after cleaning!")
                print(f"   Its dtype is: {data[col].dtype}")
                # Try to find offending non-numeric values that weren't coerced to NaN
                try:
                    # Create a boolean mask where to_numeric fails but original value is not NaN
                    non_numeric_mask = pd.to_numeric(data[col], errors='coerce').isna() & data[col].notna()
                    if non_numeric_mask.any():
                        offending_values = data.loc[non_numeric_mask, col].unique()
                        print(f"   Sample offending string values found: {offending_values[:10]}")
                    else:
                        print(f"   Could not isolate specific offending string values (may indicate mixed types or other issue).")
                except Exception as inspect_err:
                    print(f"   (Could not inspect offending values: {inspect_err})")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                numeric_check_passed = False

        if not numeric_check_passed:
            print(f"Skipping symbol {symbol} due to persistent non-numeric data in critical columns.")
            continue # Skip to the next symbol

        # --- Set Index ---
        # Set date as index AFTER cleaning and verification
        try:
             data.set_index('date', inplace=True)
             if not isinstance(data.index, pd.DatetimeIndex):
                  print("Warning: Index is not a DatetimeIndex after setting. Check date parsing.")
                  # Potentially convert index if needed: data.index = pd.to_datetime(data.index)
        except KeyError:
             print("Error: 'date' column not found for setting index. Skipping symbol.")
             continue


        # --- Calculate Features ---
        data = calculate_features(data)
        print(f"Data shape after features & feature dropna: {data.shape}")

        if data.empty:
            print(f"Warning: No data remaining for {symbol} after feature calculation. Skipping.")
            continue

        # --- Generate Signals and Labels ---
        print("Generating signals and labels...")
        signal_indices, labels = generate_signals_and_labels(
            data,
            lookahead_bars=LOOKAHEAD_BARS,
            sl_mult=SL_MULT,
            tp_mult=TP_MULT,
            vol_z_long_thresh=VOL_Z_LONG_THRESH,
            vol_z_short_thresh=VOL_Z_SHORT_THRESH
        )

        # --- Collect Results for Model ---
        if len(signal_indices) < MIN_SIGNALS_PER_SYMBOL:
            print(f"Warning: Insufficient signals ({len(signal_indices)}) generated for {symbol} based on MIN_SIGNALS_PER_SYMBOL={MIN_SIGNALS_PER_SYMBOL}. Skipping symbol's signals.")
            continue

        # Select features *at the time of the signal* for the model
        # Define which calculated features to use for the ML model
        feature_columns = ['sma', 'std', 'atr', 'rsi', 'macd', 'macd_hist', 'volume_z', 'rsi_lag1']
        # Ensure the selected features exist in the dataframe columns after calculations
        valid_feature_columns = [col for col in feature_columns if col in data.columns]
        missing_model_features = [col for col in feature_columns if col not in data.columns]
        if missing_model_features:
            print(f"Warning: Some features specified for the model are not available in data columns for {symbol}: {missing_model_features}")
        if not valid_feature_columns:
            print(f"Error: No valid feature columns found for model input for {symbol}. Skipping.")
            continue

        # Extract features corresponding to the signal indices
        signal_features = data.loc[signal_indices, valid_feature_columns]

        # Add features to the combined list
        all_signal_features.append(signal_features)
        all_labels.extend(labels) # Use extend for the flat list of labels
        print(f"Successfully processed {symbol}. Added {len(signal_indices)} signals.")

    except Exception as e:
        # Catch any other unexpected errors during the processing of a symbol
        print(f"\n--- An unexpected error occurred processing symbol: {symbol} ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc() # Print full traceback for the error in this symbol's loop
        print(f"--- Skipping rest of processing for {symbol} ---")
        continue # Move to the next symbol

# --- Model Training and Evaluation (Outside the symbol loop) ---
print("\n--- Data Processing Complete ---")

# Check if any signals were collected across all symbols
if not all_signal_features:
    print("\nCRITICAL: No signals were collected from any symbol. Cannot train model.")
    print("Check data, parameters, signal logic, and previous error messages.")
else:
    # Combine features from all symbols into a single DataFrame
    print(f"Combining features from {len(all_signal_features)} dataframes...")
    X = pd.concat(all_signal_features)
    y = np.array(all_labels)

    print(f"\nTotal signals for training/testing: {len(y)}")
    if len(X) != len(y):
         # This should ideally not happen if appending logic is correct
         print(f"Error: Mismatch between number of feature sets ({len(X)}) and labels ({len(y)})!")
    elif len(y) == 0:
         print("Error: Zero labels collected. Cannot train.")
    elif len(np.unique(y)) < 2:
        print(f"Error: Only one class found in the labels ({np.unique(y)}). Cannot train classifier.")
        print(f"Class distribution: {np.bincount(y)}")
    else:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)} (Class 0, Class 1)")

        # --- Chronological Data Splitting ---
        # Ensure data is sorted by index if concatenating from different sources/times
        # If signal_indices were Timestamps, concat preserves order within each symbol,
        # but order between symbols depends on file processing order.
        # For true chronological split across symbols, you might need to sort X by index:
        # X.sort_index(inplace=True)
        # y = y[X.index] # Reorder y accordingly - requires storing signal indices with labels before concat
        # Simpler approach for now: Split based on concatenated order
        split_index = int(len(X) * (1 - TEST_SIZE))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        print(f"\nSplitting data chronologically (approx {TEST_SIZE*100:.0f}% test):")
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        if len(X_train) == 0 or len(X_test) == 0:
             print("Error: Training or testing set is empty after split. Check TEST_SIZE and data length.")
        else:
            # --- Train the Random Forest Classifier ---
            print("\nTraining Random Forest model...")
            # Add class_weight='balanced' to handle imbalance
            model = RandomForestClassifier(
                n_estimators=RF_ESTIMATORS,
                random_state=RANDOM_STATE,
                class_weight='balanced', # Helps with imbalanced classes
                n_jobs=-1 # Use all available CPU cores
            )
            model.fit(X_train, y_train)

            # --- Evaluate the Model ---
            print("\nEvaluating model on the test set...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for class 1

            print("\nAccuracy:", accuracy_score(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Unprofitable (0)', 'Profitable (1)']))

            # --- Feature Importances ---
            print("\nFeature Importances:")
            try:
                importances = model.feature_importances_
                feature_names = X_train.columns # Get feature names from the training dataframe
                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                print(feature_importance_df)
            except Exception as fi_e:
                print(f"Could not retrieve feature importances: {fi_e}")


            # --- Save the Trained Model and Features ---
            model_filename = 'trading_signal_model_multi.joblib'
            joblib.dump(model, model_filename)
            print(f"\nModel saved to '{model_filename}'")

            # Save the list of features used for training (important for prediction later)
            feature_list_filename = 'trading_model_features.list'
            try:
                 with open(feature_list_filename, 'w') as f:
                      for feature in feature_names:
                          f.write(f"{feature}\n")
                 print(f"Feature list saved to '{feature_list_filename}'")
            except Exception as fl_e:
                 print(f"Error saving feature list: {fl_e}")

print("\n--- Script Finished ---")