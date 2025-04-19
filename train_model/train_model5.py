# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
# --- NEW IMPORTS ---
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler # Added for feature scaling
# --- END NEW IMPORTS ---
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import pandas.api.types # To check dtypes robustly
import traceback # For detailed error prints

# --- Configuration ---
# Define symbols for training
SYMBOLS = ['AAPL', 'AMD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'QQQ', 'SPY', 'TSLA']
DATA_DIR = '.' # Directory where CSV files are located
FILENAME_TEMPLATE = '{}_5min_historical_data.csv' # Matches output of get_data.py

# --- Feature Calculation Parameters ---
SMA_WINDOW = 20
ATR_PERIOD = 14
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
# New Feature Params
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3 # Smoothing for %K before %D calculation
ROC_PERIOD = 10

# --- Labeling Parameters ---
# >>> MODIFIED: Changed labeling method <<<
LABELING_METHOD = 'significant_direction' # Options: 'direction', 'significant_direction', 'tp_sl'

# Parameters for 'direction' labeling (kept for reference)
LABEL_LOOKAHEAD_N_DIRECTION = 5

# >>> NEW: Parameters for 'significant_direction' labeling <<<
LABEL_LOOKAHEAD_N_SIG = 12      # How many bars into the future to check for significant move
LABEL_SIGNIFICANCE_THRESHOLD_PCT = 0.002 # e.g., 0.001 = 0.1% move required

# Parameters for 'tp_sl' labeling (kept for reference)
LOOKAHEAD_BARS_TP_SL = 12 # How many bars to look ahead for TP/SL
SL_MULT = 2.0 # Stop Loss ATR multiplier
TP_MULT = 4.0 # Take Profit ATR multiplier
VOL_Z_LONG_THRESH = 0.05 # Volume Z-score threshold for long signals
VOL_Z_SHORT_THRESH = -0.05 # Volume Z-score threshold for short signals

# --- Data Processing Parameters ---
MIN_ROWS_PER_SYMBOL = 100 # Minimum rows needed after processing a symbol
USE_FEATURE_SCALING = True # Flag to control feature scaling

# --- Model Training Parameters ---
TEST_SIZE = 0.2 # Proportion of data for testing (used for chronological split point)
RANDOM_STATE = 42 # For reproducibility
N_CV_SPLITS = 5 # Number of splits for TimeSeriesSplit cross-validation

# XGBoost GridSearch Parameters
XGB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    # 'subsample': [0.8],
    # 'colsample_bytree': [0.8],
}

# --- Feature Calculation Function (Unchanged) ---
def calculate_features(df):
    """Calculate technical indicators and features for the dataset."""
    print(f"Input shape for features: {df.shape}")
    cols_to_check = ['high', 'low', 'close', 'open', 'volume']
    for col in cols_to_check:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Converting column {col} to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
    print(f"Shape after dropping essential NaNs: {df.shape}")
    if df.empty: return df
    if not isinstance(df.index, pd.DatetimeIndex):
         print("CRITICAL WARNING: Index is NOT a DatetimeIndex before feature calculation!")
         try:
            df.index = pd.to_datetime(df.index)
            if not isinstance(df.index, pd.DatetimeIndex): raise ValueError("Failed conversion.")
            print("Index converted within calculate_features."); df.sort_index(inplace=True)
         except Exception as e:
            print(f"ERROR: Could not ensure DatetimeIndex: {e}. Time features fail."); df['hour']=np.nan; df['dayofweek']=np.nan
    else:
        print("Index confirmed as DatetimeIndex.")
        if not df.index.is_monotonic_increasing: print("Warning: DatetimeIndex not sorted."); df.sort_index(inplace=True)
    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']; df['lower_band'] = df['sma'] - 2 * df['std']
    high_prices=df['high'].values; low_prices=df['low'].values; close_prices=df['close'].values
    min_len_for_talib = max(SMA_WINDOW, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, STOCH_K, ROC_PERIOD) + 10
    if len(close_prices) >= min_len_for_talib:
        print("Calculating TA-Lib indicators...")
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
        df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
        df['macd']=macd; df['macd_signal']=macdsignal; df['macd_hist']=macdhist
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma'].replace(0, 1e-10)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=STOCH_K, slowk_period=STOCH_SMOOTH, slowk_matype=0, slowd_period=STOCH_D, slowd_matype=0)
        df['roc'] = talib.ROC(close_prices, timeperiod=ROC_PERIOD)
    else:
        print(f"Warning: Not enough data ({len(close_prices)}) for TA-Lib. Skipping.");
        for col in ['atr','rsi','macd','macd_signal','macd_hist','bb_width','stoch_k','stoch_d','roc']: df[col]=np.nan
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    df['volume_z'] = (df['volume'] - df['volume_sma5']) / df['volume_sma5'].replace(0, 1e-10)
    df['rsi_lag1'] = df['rsi'].shift(1)
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour; df['dayofweek'] = df.index.dayofweek; print("Time features created.")
    elif 'hour' not in df.columns:
        print("Warning: Cannot create time features."); df['hour']=np.nan; df['dayofweek']=np.nan
    initial_rows = len(df); df.dropna(inplace=True)
    print(f"Shape after feature calculation & final dropna: {df.shape}. Dropped {initial_rows - len(df)} rows.")
    return df

# --- Labeling Function 1: Simple Direction (Unchanged) ---
def create_direction_labels(df, n_bars):
    """Create labels based on future price direction."""
    if df.empty: return df
    print(f"Creating direction labels looking ahead {n_bars} bars...")
    df['future_close'] = df['close'].shift(-n_bars); df.dropna(subset=['future_close'], inplace=True)
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df
    df['label'] = (df['future_close'] > df['close']).astype(int)
    df.drop(columns=['future_close'], inplace=True)
    print(f"Created 'label' (direction). Shape: {df.shape}"); print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    return df

# --- >>> NEW LABELING FUNCTION: Significant Moves <<< ---
def create_significant_move_labels(df, n_bars, threshold_pct):
    """
    Create labels based on whether the price moves significantly up or down
    over the next n_bars.
    Label 1 = Significant UP move.
    Label 0 = Significant DOWN move.
    Insignificant moves are filtered out.
    """
    if df.empty: return df
    print(f"Creating labels for moves > {threshold_pct:.3%} over {n_bars} bars...")
    df['future_close'] = df['close'].shift(-n_bars)
    # Drop rows where future_close is NaN (the last n_bars)
    df.dropna(subset=['future_close'], inplace=True)
    if df.empty:
        print("DataFrame empty after dropping future_close NaNs.")
        return df

    # Calculate percentage change (handle potential division by zero for safety)
    price_change_pct = (df['future_close'] - df['close']) / df['close'].replace(0, 1e-10)

    # Define conditions for significant moves
    conditions = [
        price_change_pct > threshold_pct,      # Significant Up
        price_change_pct < -threshold_pct,     # Significant Down
    ]
    choices = [1, 0] # 1 = Up, 0 = Down

    # Apply conditions, default to -1 for insignificant moves
    df['label'] = np.select(conditions, choices, default=-1)

    # Filter out the insignificant moves (where label is -1)
    initial_count = len(df)
    df = df[df['label'] != -1].copy()
    filtered_count = initial_count - len(df)
    print(f"Filtered out {filtered_count} insignificant moves ({filtered_count/max(1, initial_count):.2%}).")

    # Drop the intermediate column
    df.drop(columns=['future_close'], inplace=True)

    print(f"Created 'label' (significant moves only). Shape after filtering: {df.shape}")
    if not df.empty:
        print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    else:
        print("Warning: DataFrame is empty after filtering for significant moves.")
    return df


# --- Labeling Function 3: TP/SL (Corrected Syntax) ---
def generate_signals_and_labels_tp_sl(df, lookahead_bars, sl_mult, tp_mult, vol_z_long_thresh, vol_z_short_thresh):
    """Generate trading signals and label them based on TP/SL criteria."""
    print("--- TP/SL LABELING FUNCTION (generate_signals_and_labels_tp_sl) ---")
    print("--- NOTE: This function is currently NOT ACTIVE if LABELING_METHOD != 'tp_sl' ---")
    signal_indices = []; labels = []; long_crossings = 0; short_crossings = 0
    if not isinstance(df.index, pd.DatetimeIndex): raise TypeError("Index must be DatetimeIndex")

    valid_indices = df.index
    # --- CORRECTED LINE BELOW ---
    if len(valid_indices) <= lookahead_bars:
        print(f"Warning: Not enough data ({len(valid_indices)} points) for TP/SL lookahead ({lookahead_bars}).")
        return [], []
    # --- END CORRECTION ---

    is_below_lower = df['close'] < df['lower_band']; is_above_upper = df['close'] > df['upper_band']
    is_vol_z_long = df['volume_z'] > vol_z_long_thresh; is_vol_z_short = df['volume_z'] < vol_z_short_thresh
    has_valid_atr = ~pd.isna(df['atr']) & (df['atr'] > 1e-9)
    for idx_loc in range(len(valid_indices) - lookahead_bars):
        current_index = valid_indices[idx_loc]; current_data = df.iloc[idx_loc]
        is_long_signal = (is_below_lower.iloc[idx_loc] and is_vol_z_long.iloc[idx_loc] and has_valid_atr.iloc[idx_loc])
        is_short_signal = (is_above_upper.iloc[idx_loc] and is_vol_z_short.iloc[idx_loc] and has_valid_atr.iloc[idx_loc])
        if not (is_long_signal or is_short_signal): continue
        entry_price = current_data['close']; atr = current_data['atr']; entry_type = 'long' if is_long_signal else 'short'
        if entry_type == 'long': long_crossings += 1; stop_loss = entry_price - sl_mult * atr; take_profit = entry_price + tp_mult * atr
        else: short_crossings += 1; stop_loss = entry_price + sl_mult * atr; take_profit = entry_price - tp_mult * atr
        future_data_slice = df.iloc[idx_loc + 1 : idx_loc + 1 + lookahead_bars]; future_lows = future_data_slice['low']; future_highs = future_data_slice['high']
        hit_tp = False; hit_sl = False; first_tp_time = pd.NaT; first_sl_time = pd.NaT
        try:
            if entry_type == 'long':
                tp_hits = future_highs[future_highs >= take_profit]; sl_hits = future_lows[future_lows <= stop_loss]
                if not tp_hits.empty: hit_tp = True; first_tp_time = tp_hits.index[0]
                if not sl_hits.empty: hit_sl = True; first_sl_time = sl_hits.index[0]
            elif entry_type == 'short':
                tp_hits = future_lows[future_lows <= take_profit]; sl_hits = future_highs[future_highs >= stop_loss]
                if not tp_hits.empty: hit_tp = True; first_tp_time = tp_hits.index[0]
                if not sl_hits.empty: hit_sl = True; first_sl_time = sl_hits.index[0]
        except Exception as e_generic: print(f"Error during TP/SL check {current_index}: {e_generic}"); traceback.print_exc(); continue
        label = 0
        if hit_tp and hit_sl:
            if pd.notna(first_tp_time) and pd.notna(first_sl_time):
                if first_tp_time <= first_sl_time: label = 1
        elif hit_tp: label = 1
        signal_indices.append(current_index); labels.append(label)
    print(f"Longs considered: {long_crossings}, Shorts considered: {short_crossings}, Total signals: {len(signal_indices)}")
    return signal_indices, labels


# --- Main Execution Block ---
if __name__ == "__main__":
    all_symbol_data_list = []
    base_feature_columns = [
        'sma', 'std', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'volume_z', 'rsi_lag1',
        'bb_width', 'stoch_k', 'stoch_d', 'roc', 'hour', 'dayofweek'
    ]
    print("Starting data processing...")
    for symbol in SYMBOLS:
        print(f"\n--- Processing Symbol: {symbol} ---")
        file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(symbol))
        if not os.path.exists(file_path): print(f"Warning: Data file not found: {file_path}. Skipping."); continue
        data = None
        try:
            # --- Load Data & Ensure DatetimeIndex ---
            print(f"Loading data for {symbol}..."); # Simplified logging
            # ... (Date loading/parsing logic remains the same) ...
            try:
                correct_date_format = '%Y-%m-%d %H:%M:%S'
                try:
                     data = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
                     if not isinstance(data.index, pd.DatetimeIndex):
                         print("Warning: Index parsing failed or didn't produce DatetimeIndex with read_csv. Attempting fallback...")
                         data = pd.read_csv(file_path)
                         if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
                     else: print("Index successfully parsed with read_csv."); data.sort_index(inplace=True)
                except (ValueError, TypeError, KeyError, pd.errors.ParserError) as e_read:
                    print(f"Info: read_csv parsing failed ('{e_read}'). Attempting fallback pd.to_datetime."); data = pd.read_csv(file_path)
                    if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
                if not isinstance(data.index, pd.DatetimeIndex):
                    print("Executing fallback: Parsing date column using pd.to_datetime.");
                    if 'date' not in data.columns: raise ValueError("Column 'date' not found for fallback parsing.")
                    data['date'] = pd.to_datetime(data['date'], format=correct_date_format, errors='coerce')
                    failed_count = data['date'].isnull().sum();
                    if failed_count > 0: print(f"Warning: {failed_count} dates failed parsing. Dropping.")
                    data.dropna(subset=['date'], inplace=True);
                    if data.empty: raise ValueError("No valid dates found after fallback parsing.")
                    data.set_index('date', inplace=True); data.sort_index(inplace=True)
                    if not isinstance(data.index, pd.DatetimeIndex): raise TypeError(f"CRITICAL: Failed to create DatetimeIndex.")
                    print("Index successfully parsed/set with fallback pd.to_datetime.")
                print(f"Loaded {len(data)} rows. Index type: {type(data.index)}, Is monotonic: {data.index.is_monotonic_increasing}")
                if data.empty: print("Error: Data empty after loading."); continue
            except Exception as e: print(f"CRITICAL Error loading dates for {symbol}: {e}"); traceback.print_exc(); continue

            # --- Clean OHLCV ---
            print("Cleaning OHLCV columns..."); # Simplified logging
            # ... (Cleaning logic remains the same) ...
            initial_rows = len(data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]): data[col] = pd.to_numeric(data[col], errors='coerce')
                else: print(f"Warning: Column {col} not found for {symbol}.")
            data.dropna(subset=[c for c in ohlcv_cols if c in data.columns], inplace=True)
            print(f"Dropped {initial_rows - len(data)} rows due to non-numeric/NaN OHLCV.")
            if data.empty: print(f"Error: Data empty after cleaning OHLCV for {symbol}."); continue

            # --- Calculate Features ---
            print("Calculating features...")
            data = calculate_features(data)
            if data.empty or len(data) < MIN_ROWS_PER_SYMBOL: print(f"Error or insufficient data ({len(data)} rows) after features for {symbol}. Skipping."); continue

            # <<< --- ADD SYMBOL COLUMN --- >>>
            data['symbol'] = symbol
            print(f"Added 'symbol' column.") # Simplified logging

            # --- Generate Labels ---
            print(f"Applying labeling method: '{LABELING_METHOD}'...")
            # >>> MODIFIED: Call appropriate labeling function <<<
            if LABELING_METHOD == 'direction':
                data = create_direction_labels(data, n_bars=LABEL_LOOKAHEAD_N_DIRECTION)
            elif LABELING_METHOD == 'significant_direction':
                 data = create_significant_move_labels(data,
                                                       n_bars=LABEL_LOOKAHEAD_N_SIG,
                                                       threshold_pct=LABEL_SIGNIFICANCE_THRESHOLD_PCT)
            elif LABELING_METHOD == 'tp_sl':
                # TP/SL Labeling logic remains separate for clarity if reactivated
                print("Generating TP/SL signals and labels...")
                signal_indices, labels = generate_signals_and_labels_tp_sl(data, LOOKAHEAD_BARS_TP_SL, SL_MULT, TP_MULT, VOL_Z_LONG_THRESH, VOL_Z_SHORT_THRESH)
                if len(signal_indices) < MIN_ROWS_PER_SYMBOL: print(f"Warning: Insufficient signals ({len(signal_indices)}) for {symbol}. Skipping."); continue
                missing_base_features = [f for f in base_feature_columns if f not in data.columns];
                if missing_base_features: print(f"FATAL: Base features missing for TP/SL: {missing_base_features}. Skipping."); continue
                try: signal_features = data.loc[signal_indices, base_feature_columns + ['symbol']].copy()
                except KeyError as ke: print(f"Error selecting signal features: {ke}"); continue
                signal_features['label'] = labels; initial_count = len(signal_features)
                signal_features.dropna(inplace=True); print(f"Dropped {initial_count - len(signal_features)} TP/SL rows due to NaNs.")
                if len(signal_features) < MIN_ROWS_PER_SYMBOL: print(f"Warning: Insufficient TP/SL data ({len(signal_features)}). Skipping."); continue
                # For TP/SL, we use signal_features DF, not the full 'data' DF
                current_symbol_data = signal_features
                all_symbol_data_list.append(current_symbol_data)
                print(f"Successfully processed {symbol} (TP/SL). Added {len(current_symbol_data)} rows.")
                continue # Skip to next symbol as data is already appended for TP/SL
            else:
                print(f"Error: Unknown LABELING_METHOD '{LABELING_METHOD}'. Skipping symbol."); continue

            # --- Post-labeling processing (for direction methods) ---
            if LABELING_METHOD in ['direction', 'significant_direction']:
                if data.empty or 'label' not in data.columns or data['label'].isnull().any():
                     print(f"Error: No valid labels or labels are all NaN for {symbol} after method '{LABELING_METHOD}'. Skipping.")
                     continue
                missing_base_features = [f for f in base_feature_columns if f not in data.columns]
                if missing_base_features: print(f"FATAL: Base features missing post-labeling: {missing_base_features}. Skipping."); continue
                cols_to_keep = base_feature_columns + ['label', 'symbol']
                current_symbol_data = data[cols_to_keep].copy()
                if current_symbol_data.isnull().values.any():
                    print(f"Warning: NaNs detected in final selection for {symbol}. Dropping."); initial_count = len(current_symbol_data)
                    current_symbol_data.dropna(inplace=True); print(f"Dropped {initial_count - len(current_symbol_data)} rows.")
                if len(current_symbol_data) < MIN_ROWS_PER_SYMBOL: print(f"Warning: Insufficient data ({len(current_symbol_data)}) after finalizing {LABELING_METHOD} labels. Skipping."); continue
                all_symbol_data_list.append(current_symbol_data)
                print(f"Successfully processed {symbol} ({LABELING_METHOD}). Added {len(current_symbol_data)} rows.")

        except Exception as e:
            print(f"\n--- Unexpected error processing symbol: {symbol} ---")
            print(f"Error Type: {type(e).__name__}, Message: {e}"); traceback.print_exc()
            print(f"--- Skipping rest of processing for {symbol} ---"); continue

    # --- Model Training and Evaluation ---
    print("\n--- Data Processing Complete ---")
    if not all_symbol_data_list: print("\nCRITICAL: No data collected. Cannot train."); exit()

    print(f"Combining data from {len(all_symbol_data_list)} symbols...")
    combined_data = pd.concat(all_symbol_data_list)
    print(f"Total rows before sorting: {len(combined_data)}")
    print("Sorting combined data by timestamp...")
    combined_data.sort_index(inplace=True)
    print(f"Combined data sorted. Index monotonic: {combined_data.index.is_monotonic_increasing}")
    print(f"Total rows after sorting: {len(combined_data)}")

    print("One-hot encoding 'symbol' column...")
    try:
        combined_data = pd.get_dummies(combined_data, columns=['symbol'], prefix='sym', drop_first=True)
        print(f"Data shape after one-hot encoding: {combined_data.shape}")
    except KeyError: print("Error: 'symbol' column not found. Skipping encoding.")

    final_feature_columns = base_feature_columns.copy()
    ohe_symbol_cols = [col for col in combined_data.columns if col.startswith('sym_')]
    final_feature_columns.extend(ohe_symbol_cols)
    print(f"Final feature columns ({len(final_feature_columns)}): {final_feature_columns}")

    if combined_data.isnull().values.any():
        print("CRITICAL Warning: NaNs detected before split."); nan_counts = combined_data.isnull().sum(); print("NaN counts:\n", nan_counts[nan_counts > 0])
        print("Dropping NaNs..."); combined_data.dropna(inplace=True); print(f"Rows remaining: {len(combined_data)}")
        if combined_data.empty: print("Error: Combined data empty after final NaN drop."); exit()

    print("Separating features (X) and labels (y)...")
    missing_final_features = [f for f in final_feature_columns if f not in combined_data.columns]
    if missing_final_features: print(f"FATAL Error: Missing final features: {missing_final_features}"); exit()
    if 'label' not in combined_data.columns: print("FATAL Error: 'label' column missing!"); exit()
    X = combined_data[final_feature_columns]; y = combined_data['label']
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    if len(X) == 0 or len(y) == 0: print("Error: X or y empty."); exit()
    if len(y.unique()) < 2: print(f"Error: Only one class found: {y.unique()}"); exit()

    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_index]; X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]; y_test = y.iloc[split_index:]
    print(f"\nSplitting data chronologically:")
    print(f"Training set: {X_train.index.min()} to {X_train.index.max()}, size: {len(X_train)}")
    print(f"Testing set : {X_test.index.min()} to {X_test.index.max()}, size: {len(X_test)}")
    if len(X_train) == 0 or len(X_test) == 0: print("Error: Empty train/test set."); exit()

    scaler = None
    if USE_FEATURE_SCALING:
        print("\nApplying StandardScaler..."); numeric_cols_to_scale = base_feature_columns
        print(f"Scaling columns: {numeric_cols_to_scale}")
        scaler = StandardScaler()
        # Use .loc to avoid SettingWithCopyWarning
        X_train.loc[:, numeric_cols_to_scale] = scaler.fit_transform(X_train[numeric_cols_to_scale])
        X_test.loc[:, numeric_cols_to_scale] = scaler.transform(X_test[numeric_cols_to_scale])
        print("StandardScaler applied.")

    print("\nSetting up GridSearchCV for XGBoost...")
    actual_n_splits = min(N_CV_SPLITS, len(X_train) // max(1, (len(X_train) // (N_CV_SPLITS + 1))))
    if actual_n_splits < 2: print(f"Warning: Training data small. Min splits=2 required."); actual_n_splits = max(2, actual_n_splits) if len(X_train) > 1 else 1
    print(f"Using TimeSeriesSplit with n_splits={actual_n_splits}")
    tscv = TimeSeriesSplit(n_splits=actual_n_splits)
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    xgb_model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=XGB_PARAM_GRID, scoring='accuracy', cv=tscv, n_jobs=-1, verbose=2)
    print(f"Starting GridSearchCV with {len(X_train)} samples...")
    best_model = None
    try:
        grid_search.fit(X_train, y_train)
        print("\nGridSearchCV finished.")
        print("Best parameters found:", grid_search.best_params_)
        print(f"Best CV score ({grid_search.scoring}): {grid_search.best_score_:.4f}")
        best_model = grid_search.best_estimator_
    except Exception as e:
         print(f"\nERROR during GridSearchCV.fit: {e}")
         print("Attempting fallback: train default model...")
         try:
             best_model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight)
             best_model.fit(X_train, y_train)
             print("Successfully trained default model as fallback.")
         except Exception as train_e: print(f"FATAL: Failed fallback training: {train_e}"); exit()
    if best_model is None: print("FATAL: No model trained."); exit()

    print("\nEvaluating model on test set...")
    y_pred = best_model.predict(X_test)
    print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # >>> MODIFIED: Updated target names for classification report <<<
    print("Test Set Classification Report:\n", classification_report(y_test, y_pred, target_names=['Significant Down (0)', 'Significant Up (1)'], zero_division=0))

    print("\nFeature Importances:")
    try:
        feature_names = X_train.columns; importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df.to_string())
    except Exception as fi_e: print(f"Could not get feature importances: {fi_e}")

    # >>> MODIFIED: Updated filenames <<<
    model_filename = 'trading_xgb_model_multisym_sigmove_tuned.joblib'
    joblib.dump(best_model, model_filename); print(f"\nModel saved to '{model_filename}'")
    feature_list_filename = 'trading_model_features_multisym_sigmove.list'
    try:
         current_features = list(X_train.columns)
         with open(feature_list_filename, 'w') as f:
              for feature in current_features: f.write(f"{feature}\n")
         print(f"Feature list ({len(current_features)}) saved to '{feature_list_filename}'")
    except Exception as fl_e: print(f"Error saving feature list: {fl_e}")
    if scaler:
        scaler_filename = 'trading_feature_scaler_multisym_sigmove.joblib'
        joblib.dump(scaler, scaler_filename); print(f"Scaler saved to '{scaler_filename}'")

    print("\n--- Script Finished ---")