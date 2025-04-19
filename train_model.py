# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np # Ensure numpy is imported
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import pandas.api.types # To check dtypes robustly
import traceback # For detailed error prints
from sklearn.model_selection import RandomizedSearchCV

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
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
ROC_PERIOD = 10
LAG_PERIODS = [1, 3, 5] # NEW: Define lag periods for features

# --- Labeling Parameters ---
LABELING_METHOD = 'significant_direction' # Options: 'direction', 'significant_direction', 'tp_sl'

# Parameters for 'direction' labeling
LABEL_LOOKAHEAD_N_DIRECTION = 12

# Parameters for 'significant_direction' labeling
# >>> EXPERIMENT HEAVILY WITH THESE <<<
LABEL_LOOKAHEAD_N_SIG = 12      # How many bars into the future (e.g., try 10, 12, 24)
LABEL_SIGNIFICANCE_THRESHOLD_PCT = 0.002 # Significance threshold (e.g., try 0.003, 0.005)

# Parameters for 'tp_sl' labeling
LOOKAHEAD_BARS_TP_SL = 12
SL_MULT = 2.0
TP_MULT = 4.0
VOL_Z_LONG_THRESH = 0.05
VOL_Z_SHORT_THRESH = -0.05

# --- Data Processing Parameters ---
MIN_ROWS_PER_SYMBOL = 100
USE_FEATURE_SCALING = True

# --- Model Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_SPLITS = 10 # Increased CV splits for potentially more robust evaluation

# XGBoost GridSearch Parameters (Smaller Grid)
XGB_PARAM_GRID = {
    'n_estimators': [100, 300, 500], # More trees
    'max_depth': [3, 5, 7],              # Deeper trees (adjust based on data size/complexity)
    'learning_rate': [0.05, 0.1],   # Different learning rates
    'subsample': [0.8, 1.0],    # Fraction of samples per tree
    'colsample_bytree': [0.7, 0.8, 1.0], # Fraction of features per tree
    'gamma': [0, 0.1, 0.5],            # Min loss reduction for split (regularization)
}

'''
# XGBoost GridSearch Parameters (Expanded Grid)
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500], # More trees
    'max_depth': [3, 5, 7],              # Deeper trees (adjust based on data size/complexity)
    'learning_rate': [0.01, 0.05, 0.1],   # Different learning rates
    'subsample': [0.7, 0.8, 0.9, 1.0],    # Fraction of samples per tree
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Fraction of features per tree
    'gamma': [0, 0.1, 0.5],            # Min loss reduction for split (regularization)
}
'''
# --- Define Base Feature Columns (Updated with New Features) ---
# NOTE: Original 'hour', 'dayofweek' are removed, replaced by cyclical versions
# NOTE: Lagged features are added
base_feature_columns = [
    'sma', 'std', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'volume_z',
    'rsi_lag1', 'bb_width', 'stoch_k', 'stoch_d', 'roc',
    # Cyclical time features
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    # New lagged features
    'return_1_lag1', 'return_1_lag3', 'return_1_lag5',
    'macd_hist_lag1', 'macd_hist_lag3', 'macd_hist_lag5',
    'stoch_k_lag1', 'stoch_k_lag3', 'stoch_k_lag5',
    'atr_lag1', 'atr_lag3', 'atr_lag5',
    'bb_width_lag1', 'bb_width_lag3', 'bb_width_lag5',
    'return_1' # Include the base return_1 as well
]
# Identify which of the base features need scaling (exclude cyclical, which are already scaled)
numeric_cols_to_scale = [
    col for col in base_feature_columns if col not in
    ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
]


# --- Feature Calculation Function (MODIFIED) ---
def calculate_features(df):
    """Calculate technical indicators and enhanced features for the dataset."""
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
            print(f"ERROR: Could not ensure DatetimeIndex: {e}. Time features fail.");
            # Assign NaN to time features if index conversion fails
            df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan
    else:
        print("Index confirmed as DatetimeIndex.")
        if not df.index.is_monotonic_increasing: print("Warning: DatetimeIndex not sorted."); df.sort_index(inplace=True)

    # Basic Features
    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']; df['lower_band'] = df['sma'] - 2 * df['std']

    # TA-Lib Indicators
    high_prices=df['high'].values; low_prices=df['low'].values; close_prices=df['close'].values
    min_len_for_talib = max(SMA_WINDOW, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, STOCH_K, ROC_PERIOD) + 10 # Adjusted min length slightly
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

    # Volume Feature
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    df['volume_z'] = (df['volume'] - df['volume_sma5']) / df['volume_sma5'].replace(0, 1e-10)

    # Initial Lag Feature
    df['rsi_lag1'] = df['rsi'].shift(1)

    # --- NEW: Cyclical Time Features ---
    if isinstance(df.index, pd.DatetimeIndex):
        print("Creating cyclical time features...")
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7.0)
        df.drop(columns=['hour', 'dayofweek'], inplace=True) # Drop original columns
        print("Cyclical time features created.")
    elif 'hour_sin' not in df.columns: # Ensure columns exist even if index fails
        print("Warning: Cannot create time features due to non-DatetimeIndex.")
        df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan

    # --- NEW: Calculate Returns and More Lagged Features ---
    print("Calculating returns and lagged features...")
    df['return_1'] = df['close'].pct_change(1) # Use 1-period return as a base feature

    features_to_lag = ['return_1', 'macd_hist', 'stoch_k', 'atr', 'bb_width'] # Add more features if desired
    new_lag_cols = []
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in LAG_PERIODS:
                col_name = f'{feature}_lag{lag}'
                df[col_name] = df[feature].shift(lag)
                new_lag_cols.append(col_name)
        else:
            print(f"Warning: Feature '{feature}' not found for lagging.")
    print(f"Added {len(new_lag_cols)} lagged feature columns.")

    initial_rows = len(df)
    # Drop NaNs resulting from rolling windows AND lagging
    df.dropna(inplace=True)
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

# --- Labeling Function 2: Significant Moves (Unchanged Function Logic) ---
def create_significant_move_labels(df, n_bars, threshold_pct):
    """
    Create labels based on whether the price moves significantly up or down
    over the next n_bars. Label 1 = Sig UP, Label 0 = Sig DOWN.
    Insignificant moves are filtered out.
    """
    if df.empty: return df
    print(f"Creating labels for moves > {threshold_pct:.3%} over {n_bars} bars...")
    df['future_close'] = df['close'].shift(-n_bars)
    df.dropna(subset=['future_close'], inplace=True) # Drop rows where future_close is NaN
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df

    price_change_pct = (df['future_close'] - df['close']) / df['close'].replace(0, 1e-10)
    conditions = [ price_change_pct > threshold_pct, price_change_pct < -threshold_pct ]
    choices = [1, 0] # 1 = Up, 0 = Down
    df['label'] = np.select(conditions, choices, default=-1) # Default -1 for insignificant

    initial_count = len(df)
    df = df[df['label'] != -1].copy() # Filter out insignificant moves
    filtered_count = initial_count - len(df)
    print(f"Filtered out {filtered_count} insignificant moves ({filtered_count/max(1, initial_count):.2%}).")

    df.drop(columns=['future_close'], inplace=True)
    print(f"Created 'label' (significant moves only). Shape after filtering: {df.shape}")
    if not df.empty:
        print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    else:
        print("Warning: DataFrame is empty after filtering for significant moves.")
    return df


# --- Labeling Function 3: TP/SL (Unchanged Function Logic) ---
def generate_signals_and_labels_tp_sl(df, lookahead_bars, sl_mult, tp_mult, vol_z_long_thresh, vol_z_short_thresh):
    """Generate trading signals and label them based on TP/SL criteria."""
    print("--- TP/SL LABELING FUNCTION (generate_signals_and_labels_tp_sl) ---")
    print("--- NOTE: This function is currently NOT ACTIVE if LABELING_METHOD != 'tp_sl' ---")
    signal_indices = []; labels = []; long_crossings = 0; short_crossings = 0
    if not isinstance(df.index, pd.DatetimeIndex): raise TypeError("Index must be DatetimeIndex")
    valid_indices = df.index
    if len(valid_indices) <= lookahead_bars:
        print(f"Warning: Not enough data ({len(valid_indices)} points) for TP/SL lookahead ({lookahead_bars}).")
        return [], []

    # Pre-calculate conditions for speed
    is_below_lower = df['close'] < df['lower_band']; is_above_upper = df['close'] > df['upper_band']
    is_vol_z_long = df['volume_z'] > vol_z_long_thresh; is_vol_z_short = df['volume_z'] < vol_z_short_thresh
    has_valid_atr = ~pd.isna(df['atr']) & (df['atr'] > 1e-9)

    # Iterate up to the point where a full lookahead is possible
    for idx_loc in range(len(valid_indices) - lookahead_bars):
        current_index = valid_indices[idx_loc]; current_data = df.iloc[idx_loc]

        # Check for signal conditions
        is_long_signal = (is_below_lower.iloc[idx_loc] and is_vol_z_long.iloc[idx_loc] and has_valid_atr.iloc[idx_loc])
        is_short_signal = (is_above_upper.iloc[idx_loc] and is_vol_z_short.iloc[idx_loc] and has_valid_atr.iloc[idx_loc])

        if not (is_long_signal or is_short_signal): continue # Skip if no signal

        entry_price = current_data['close']; atr = current_data['atr'];
        entry_type = 'long' if is_long_signal else 'short'

        # Calculate TP/SL levels
        if entry_type == 'long':
            long_crossings += 1; stop_loss = entry_price - sl_mult * atr; take_profit = entry_price + tp_mult * atr
        else: # Short signal
            short_crossings += 1; stop_loss = entry_price + sl_mult * atr; take_profit = entry_price - tp_mult * atr

        # Look ahead for TP/SL hits
        future_data_slice = df.iloc[idx_loc + 1 : idx_loc + 1 + lookahead_bars]
        future_lows = future_data_slice['low']; future_highs = future_data_slice['high']

        hit_tp = False; hit_sl = False; first_tp_time = pd.NaT; first_sl_time = pd.NaT

        try: # Find first time TP/SL are hit
            if entry_type == 'long':
                tp_hits = future_highs[future_highs >= take_profit]; sl_hits = future_lows[future_lows <= stop_loss]
                if not tp_hits.empty: hit_tp = True; first_tp_time = tp_hits.index[0]
                if not sl_hits.empty: hit_sl = True; first_sl_time = sl_hits.index[0]
            elif entry_type == 'short':
                tp_hits = future_lows[future_lows <= take_profit]; sl_hits = future_highs[future_highs >= stop_loss]
                if not tp_hits.empty: hit_tp = True; first_tp_time = tp_hits.index[0]
                if not sl_hits.empty: hit_sl = True; first_sl_time = sl_hits.index[0]
        except Exception as e_generic:
            print(f"Error during TP/SL check {current_index}: {e_generic}"); traceback.print_exc(); continue

        # Determine label based on which was hit first
        label = 0 # Default to loss (or no TP)
        if hit_tp and hit_sl: # Both hit
            if pd.notna(first_tp_time) and pd.notna(first_sl_time):
                if first_tp_time <= first_sl_time: label = 1 # TP hit first or at same time
        elif hit_tp: # Only TP hit
            label = 1

        signal_indices.append(current_index); labels.append(label)

    print(f"Longs considered: {long_crossings}, Shorts considered: {short_crossings}, Total signals: {len(signal_indices)}")
    return signal_indices, labels


# --- Main Execution Block ---
if __name__ == "__main__":
    all_symbol_data_list = []

    # `base_feature_columns` is now defined globally near config section
    print("Starting data processing...")
    for symbol in SYMBOLS:
        print(f"\n--- Processing Symbol: {symbol} ---")
        file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(symbol))
        if not os.path.exists(file_path): print(f"Warning: Data file not found: {file_path}. Skipping."); continue
        data = None
        try:
            # --- Load Data & Ensure DatetimeIndex ---
            print(f"Loading data for {symbol}...");
            try:
                correct_date_format = '%Y-%m-%d %H:%M:%S'
                try:
                     data = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
                     if not isinstance(data.index, pd.DatetimeIndex):
                         print("Warning: Index parsing failed or didn't produce DatetimeIndex with read_csv. Attempting fallback...")
                         data = pd.read_csv(file_path) # Reload without parsing index
                         if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
                     else: print("Index successfully parsed with read_csv."); data.sort_index(inplace=True)
                except (ValueError, TypeError, KeyError, pd.errors.ParserError) as e_read:
                    print(f"Info: read_csv parsing failed ('{e_read}'). Attempting fallback pd.to_datetime."); data = pd.read_csv(file_path)
                    if 'date' not in data.columns: raise ValueError("Column 'date' not found.")

                # Fallback parsing if needed
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
                if data.empty: print(f"Error: Data empty after loading for {symbol}."); continue
            except Exception as e: print(f"CRITICAL Error loading dates for {symbol}: {e}"); traceback.print_exc(); continue

            # --- Clean OHLCV ---
            print("Cleaning OHLCV columns...");
            initial_rows = len(data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]): data[col] = pd.to_numeric(data[col], errors='coerce')
                else: print(f"Warning: Column {col} not found for {symbol}.")
            data.dropna(subset=[c for c in ohlcv_cols if c in data.columns], inplace=True)
            print(f"Dropped {initial_rows - len(data)} rows due to non-numeric/NaN OHLCV.")
            if data.empty: print(f"Error: Data empty after cleaning OHLCV for {symbol}."); continue

            # --- Calculate Features (Uses Modified Function) ---
            print("Calculating features...")
            data = calculate_features(data) # This now creates cyclical time + more lags
            if data.empty or len(data) < MIN_ROWS_PER_SYMBOL:
                print(f"Error or insufficient data ({len(data)} rows) after features for {symbol}. Skipping."); continue

            # Add symbol column BEFORE labeling (needed for multi-symbol analysis later if desired)
            data['symbol'] = symbol
            print(f"Added 'symbol' column.")

            # --- Generate Labels ---
            print(f"Applying labeling method: '{LABELING_METHOD}'...")
            if LABELING_METHOD == 'direction':
                data = create_direction_labels(data, n_bars=LABEL_LOOKAHEAD_N_DIRECTION)
            elif LABELING_METHOD == 'significant_direction':
                 data = create_significant_move_labels(data,
                                                       n_bars=LABEL_LOOKAHEAD_N_SIG,
                                                       threshold_pct=LABEL_SIGNIFICANCE_THRESHOLD_PCT)
            elif LABELING_METHOD == 'tp_sl':
                print("Generating TP/SL signals and labels...")
                signal_indices, labels = generate_signals_and_labels_tp_sl(data, LOOKAHEAD_BARS_TP_SL, SL_MULT, TP_MULT, VOL_Z_LONG_THRESH, VOL_Z_SHORT_THRESH)
                if len(signal_indices) < MIN_ROWS_PER_SYMBOL: print(f"Warning: Insufficient signals ({len(signal_indices)}) for {symbol}. Skipping."); continue

                # Check if all *required* base features exist before selection
                missing_base_features = [f for f in base_feature_columns if f not in data.columns];
                if missing_base_features: print(f"FATAL: Base features missing for TP/SL: {missing_base_features}. Skipping."); continue

                try:
                    # Select only the rows corresponding to signals and the required columns
                    signal_features = data.loc[signal_indices, base_feature_columns + ['symbol']].copy()
                except KeyError as ke: print(f"Error selecting signal features: {ke}"); traceback.print_exc(); continue

                signal_features['label'] = labels; initial_count = len(signal_features)
                signal_features.dropna(inplace=True); # Drop rows if any features became NaN somehow
                print(f"Dropped {initial_count - len(signal_features)} TP/SL rows due to NaNs in selected features.")
                if len(signal_features) < MIN_ROWS_PER_SYMBOL: print(f"Warning: Insufficient TP/SL data ({len(signal_features)}). Skipping."); continue

                current_symbol_data = signal_features # This DF contains only labeled signal events
                all_symbol_data_list.append(current_symbol_data)
                print(f"Successfully processed {symbol} (TP/SL). Added {len(current_symbol_data)} rows.")
                continue # Skip to next symbol as data is already appended for TP/SL
            else:
                print(f"Error: Unknown LABELING_METHOD '{LABELING_METHOD}'. Skipping symbol."); continue

            # --- Post-labeling processing (for non-TP/SL methods) ---
            if LABELING_METHOD in ['direction', 'significant_direction']:
                if data.empty or 'label' not in data.columns or data['label'].isnull().all(): # Check if all labels are NaN
                     print(f"Error: No valid labels or labels are all NaN for {symbol} after method '{LABELING_METHOD}'. Skipping.")
                     continue

                # Check if all base features are present
                missing_base_features = [f for f in base_feature_columns if f not in data.columns]
                if missing_base_features: print(f"FATAL: Base features missing post-labeling: {missing_base_features}. Skipping."); continue

                # Select final columns for this symbol
                cols_to_keep = base_feature_columns + ['label', 'symbol']
                current_symbol_data = data[cols_to_keep].copy()

                # Final NaN check on the selected data
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
        combined_data = pd.get_dummies(combined_data, columns=['symbol'], prefix='sym', drop_first=False) # Keep all dummies for now
        print(f"Data shape after one-hot encoding: {combined_data.shape}")
    except KeyError: print("Error: 'symbol' column not found. Skipping encoding.")

    # Define final feature columns including OHE symbols
    final_feature_columns = base_feature_columns.copy() # Start with updated base features
    ohe_symbol_cols = [col for col in combined_data.columns if col.startswith('sym_')]
    final_feature_columns.extend(ohe_symbol_cols)
    # Remove duplicates if any base feature name clashes with OHE names (unlikely)
    final_feature_columns = sorted(list(set(final_feature_columns)))
    print(f"Final feature columns ({len(final_feature_columns)}): {final_feature_columns}")


    # Final check for NaNs before splitting
    if combined_data.isnull().values.any():
        print("CRITICAL Warning: NaNs detected before split."); nan_counts = combined_data.isnull().sum(); print("NaN counts:\n", nan_counts[nan_counts > 0])
        print("Dropping NaNs..."); combined_data.dropna(inplace=True); print(f"Rows remaining: {len(combined_data)}")
        if combined_data.empty: print("Error: Combined data empty after final NaN drop."); exit()

    print("Separating features (X) and labels (y)...")
    # Ensure all expected columns exist
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
        print("\nApplying StandardScaler...");
        # Use the globally defined `numeric_cols_to_scale` list
        # Ensure these columns actually exist in X_train/X_test before scaling
        cols_to_scale_present_train = [col for col in numeric_cols_to_scale if col in X_train.columns]
        cols_to_scale_present_test = [col for col in numeric_cols_to_scale if col in X_test.columns]

        if not cols_to_scale_present_train:
             print("Warning: No numeric columns identified for scaling in the training set.")
        else:
            print(f"Scaling columns ({len(cols_to_scale_present_train)}): {cols_to_scale_present_train}")
            scaler = StandardScaler()
            # Use .loc to avoid SettingWithCopyWarning and ensure alignment
            X_train.loc[:, cols_to_scale_present_train] = scaler.fit_transform(X_train[cols_to_scale_present_train])
            if cols_to_scale_present_test:
                 # Ensure test set columns match train set columns used for fitting
                 cols_to_transform = [col for col in cols_to_scale_present_train if col in cols_to_scale_present_test]
                 if cols_to_transform:
                      X_test.loc[:, cols_to_transform] = scaler.transform(X_test[cols_to_transform])
                 else:
                      print("Warning: No matching columns found in test set for scaling transformation.")
            else:
                 print("Warning: No numeric columns identified for scaling in the test set.")
            print("StandardScaler applied.")

    print("\nSetting up GridSearchCV for XGBoost...")
    # Ensure enough samples per split for TimeSeriesSplit
    min_samples_per_split = len(X_train) // (N_CV_SPLITS + 1)
    if min_samples_per_split < 1:
        print(f"Warning: Training data size ({len(X_train)}) too small for {N_CV_SPLITS} splits. Reducing splits.")
        actual_n_splits = max(2, len(X_train) // 2) # Attempt at least 2 splits if possible
    else:
        actual_n_splits = N_CV_SPLITS
    print(f"Using TimeSeriesSplit with n_splits={actual_n_splits}")
    # Set max_train_size to prevent leakage in TimeSeriesSplit if splits overlap too much implicitly
    tscv = TimeSeriesSplit(n_splits=actual_n_splits) # max_train_size=int(len(X_train)*0.8) # Optional: limit train size per fold

    # Inside the main execution block, before GridSearchCV

    # Calculate scale_pos_weight (as before)
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # --- MODIFIED HERE ---
    # In XGBClassifier instantiation:
    xgb_model = XGBClassifier(
        tree_method='hist',   # Use 'hist'
        device='cuda',        # Specify 'cuda' device
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )
    # --- END MODIFICATION ---

    # Use the expanded XGB_PARAM_GRID
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
        print("Attempting fallback: train default model with GPU...")
        try:
            # Use default XGBoost params but keep scale_pos_weight AND tree_method
            best_model = XGBClassifier(
                tree_method='gpu_hist', # <--- ADD HERE TOO
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight
            )
            best_model.fit(X_train, y_train)
            print("Successfully trained default model (GPU attempt) as fallback.")
        except Exception as train_e:
            print(f"GPU fallback training failed: {train_e}")
            print("Attempting CPU fallback training...")
            try: # Final fallback to CPU
                best_model = XGBClassifier(
                    # tree_method='hist', # Or leave as default auto/approx
                    random_state=RANDOM_STATE,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    scale_pos_weight=scale_pos_weight
                )
                best_model.fit(X_train, y_train)
                print("Successfully trained default model (CPU) as final fallback.")
            except Exception as cpu_train_e:
                print(f"FATAL: Failed CPU fallback training: {cpu_train_e}"); traceback.print_exc(); exit()

    if best_model is None: print("FATAL: No model trained."); exit()

    print("\nEvaluating model on test set...")
    y_pred = best_model.predict(X_test)
    print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Test Set Classification Report:\n", classification_report(y_test, y_pred, target_names=['Significant Down (0)', 'Significant Up (1)'], zero_division=0))

    print("\nFeature Importances:")
    try:
        feature_names = X_train.columns # Should align with features used in training
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df.to_string()) # Print all feature importances
    except Exception as fi_e: print(f"Could not get feature importances: {fi_e}")

    # --- Save Model, Features, Scaler (Updated Filenames) ---
    model_filename = 'trading_xgb_model_multisym_sigmove_tuned_v2.joblib'
    joblib.dump(best_model, model_filename); print(f"\nModel saved to '{model_filename}'")

    feature_list_filename = 'trading_model_features_multisym_sigmove_v2.list'
    try:
         # Save the columns from X_train, as these were used for training
         current_features = list(X_train.columns)
         with open(feature_list_filename, 'w') as f:
              for feature in current_features: f.write(f"{feature}\n")
         print(f"Feature list ({len(current_features)}) saved to '{feature_list_filename}'")
    except Exception as fl_e: print(f"Error saving feature list: {fl_e}")

    if scaler:
        scaler_filename = 'trading_feature_scaler_multisym_sigmove_v2.joblib'
        joblib.dump(scaler, scaler_filename); print(f"Scaler saved to '{scaler_filename}'")

    print("\n--- Script Finished ---")