# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np # Ensure numpy is imported
import talib
from xgboost import XGBClassifier
# Import necessary classes for manual CV loop
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score # <-- ADDED balanced_accuracy_score
import joblib
import os
import pandas.api.types # To check dtypes robustly
import traceback # For detailed error prints
# from sklearn.model_selection import RandomizedSearchCV # Keep commented unless switching
from datetime import datetime # <-- ADDED for timestamps
import xgboost as xgb # <--- ENSURE THIS IS IMPORTED AT THE TOP OF YOUR SCRIPT
# Removed functools.partial as it's no longer needed

# --- Configuration ---
# Define symbols for training
ALL_SYMBOLS = ['AAPL', 'AMD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'QQQ', 'SPY', 'TSLA']
MARKET_INDEX_SYMBOL = 'SPY' # Symbol to use for market context
SYMBOLS_TO_PREDICT = [s for s in ALL_SYMBOLS if s != MARKET_INDEX_SYMBOL] # Symbols we want predictions for

DATA_DIR = '.' # Directory where CSV files are located
FILENAME_TEMPLATE = '{}_1h_historical_data.csv' # Input files are hourly

# --- Feature Calculation Parameters (Interpret periods relative to 1-hour bars now) ---
SMA_WINDOW = 20 # Now 20 hours
ATR_PERIOD = 14 # Now 14 hours
RSI_PERIOD = 14 # Now 14 hours
MACD_FAST = 12 # Now 12 hours
MACD_SLOW = 26 # Now 26 hours
MACD_SIGNAL = 9 # Now 9 hours
STOCH_K = 14 # Now 14 hours
STOCH_D = 3 # Now 3 hours
STOCH_SMOOTH = 3
ROC_PERIOD = 10 # Now 10 hours
LAG_PERIODS = [1, 3, 5] # Now lags of 1, 3, 5 hours

# --- Labeling Parameters ---
LABELING_METHOD = 'significant_direction' # Options: 'direction', 'significant_direction', 'tp_sl'

# Parameters for 'direction' labeling
# LABEL_LOOKAHEAD_N_DIRECTION = 12 # Example: 12 hours if using 'direction'

# Parameters for 'significant_direction' labeling
LABEL_LOOKAHEAD_N_SIG = 6 # Lookahead 6 hours on 1-hour bars
LABEL_SIGNIFICANCE_THRESHOLD_PCT = 0.015 # 1.5% threshold

# Parameters for 'tp_sl' labeling (adjust lookahead if used)
# LOOKAHEAD_BARS_TP_SL = 12 # Example: 12 hours if using 'tp_sl'
SL_MULT = 2.0
TP_MULT = 4.0
VOL_Z_LONG_THRESH = 0.05
VOL_Z_SHORT_THRESH = -0.05

# --- Data Processing Parameters ---
MIN_ROWS_PER_SYMBOL = 50 # Reduced slightly for potentially less hourly data
USE_FEATURE_SCALING = True

# --- Model Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_SPLITS = 5 # Reduced splits as dataset size will decrease significantly
                # TimeSeriesSplit will automatically reduce further if needed

# XGBoost GridSearch Parameters (Using the smaller grid from before)
XGB_PARAM_GRID = {
    'n_estimators': [100, 200], # Reduced for faster testing, expand later
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
}

# --- Define Base Feature Columns (For individual stocks before adding context) ---
base_feature_columns = [
    'sma', 'std', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'volume_z',
    'rsi_lag1', 'bb_width', 'stoch_k', 'stoch_d', 'roc',
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    'return_1_lag1', 'return_1_lag3', 'return_1_lag5',
    'macd_hist_lag1', 'macd_hist_lag3', 'macd_hist_lag5',
    'stoch_k_lag1', 'stoch_k_lag3', 'stoch_k_lag5',
    'atr_lag1', 'atr_lag3', 'atr_lag5',
    'bb_width_lag1', 'bb_width_lag3', 'bb_width_lag5',
    'return_1'
]

# --- Define SPY Features to Use as Context ---
# These will be calculated on SPY data and then merged onto other symbols
spy_context_features_to_keep = [
    'return_1', 'rsi', 'atr', 'sma', 'volume_z', 'macd_hist' # Example set
]
# Generate the prefixed names for later use
spy_context_feature_names = [f"SPY_{col}" for col in spy_context_features_to_keep]


# --- Feature Calculation Function (Unchanged, operates on any input DF) ---
def calculate_features(df, symbol_name="Data"): # Added symbol_name for better logging
    """Calculate technical indicators and enhanced features for the dataset."""
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Input shape: {df.shape}") # Less verbose
    cols_to_check = ['high', 'low', 'close', 'open', 'volume']
    for col in cols_to_check:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            # print(f"Converting column {col} to numeric...") # Less verbose
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Shape after dropping essential NaNs: {df.shape}") # Less verbose
    if df.empty: return df

    if not isinstance(df.index, pd.DatetimeIndex):
         print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) CRITICAL WARNING: Index is NOT a DatetimeIndex before feature calculation!")
         try:
            df.index = pd.to_datetime(df.index)
            if not isinstance(df.index, pd.DatetimeIndex): raise ValueError("Failed conversion.")
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Index converted within calculate_features."); df.sort_index(inplace=True)
         except Exception as e:
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) ERROR: Could not ensure DatetimeIndex: {e}. Time features fail.");
            # Assign NaN to time features if index conversion fails
            df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan
    else:
        # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Index confirmed as DatetimeIndex.") # Less verbose
        if not df.index.is_monotonic_increasing: print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: DatetimeIndex not sorted."); df.sort_index(inplace=True)

    # Basic Features
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Calculating basic features (SMA, STD, Bands)...") # Less verbose
    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']; df['lower_band'] = df['sma'] - 2 * df['std']
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Finished basic features.") # Less verbose

    # TA-Lib Indicators
    # Use .copy() to avoid modifying underlying array if df columns are views
    high_prices=df['high'].copy().values; low_prices=df['low'].copy().values; close_prices=df['close'].copy().values
    min_len_for_talib = max(SMA_WINDOW, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, STOCH_K, ROC_PERIOD) + 10 # Adjusted min length slightly
    if len(close_prices) >= min_len_for_talib:
        # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Calculating TA-Lib indicators...") # Less verbose
        try:
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
            df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
            macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
            df['macd']=macd; df['macd_signal']=macdsignal; df['macd_hist']=macdhist
            # Ensure sma != 0 before dividing for bb_width
            sma_safe = df['sma'].replace(0, 1e-10) # Replace 0 with a tiny number
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / sma_safe

            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=STOCH_K, slowk_period=STOCH_SMOOTH, slowk_matype=0, slowd_period=STOCH_D, slowd_matype=0)
            df['roc'] = talib.ROC(close_prices, timeperiod=ROC_PERIOD)
        except Exception as talib_e:
             print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Error during TA-Lib calculations: {talib_e}. Assigning NaNs.")
             for col in ['atr','rsi','macd','macd_signal','macd_hist','bb_width','stoch_k','stoch_d','roc']: df[col]=np.nan

        # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Finished TA-Lib indicators.") # Less verbose
    else:
        print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Not enough data ({len(close_prices)}) for TA-Lib. Skipping.");
        for col in ['atr','rsi','macd','macd_signal','macd_hist','bb_width','stoch_k','stoch_d','roc']: df[col]=np.nan

    # Volume Feature
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Calculating volume features...") # Less verbose
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    # Ensure volume_sma5 != 0 before dividing for volume_z
    volume_sma5_safe = df['volume_sma5'].replace(0, 1e-10)
    df['volume_z'] = (df['volume'] - df['volume_sma5']) / volume_sma5_safe


    # Initial Lag Feature
    df['rsi_lag1'] = df['rsi'].shift(1)
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Finished volume features & initial lag.") # Less verbose

    # --- NEW: Cyclical Time Features ---
    if isinstance(df.index, pd.DatetimeIndex):
        # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Creating cyclical time features...") # Less verbose
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7.0)
        df.drop(columns=['hour', 'dayofweek'], inplace=True) # Drop original columns
        # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Cyclical time features created.") # Less verbose
    elif 'hour_sin' not in df.columns: # Ensure columns exist even if index fails
        print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Cannot create time features due to non-DatetimeIndex.")
        df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan

    # --- NEW: Calculate Returns and More Lagged Features ---
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Calculating returns and lagged features...") # Less verbose
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
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Feature '{feature}' not found for lagging.")
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Added {len(new_lag_cols)} lagged feature columns.") # Less verbose

    initial_rows = len(df)
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Dropping NaNs after feature calculation (initial rows: {initial_rows})...") # Less verbose
    # Drop rows where *any* column is NaN, which is standard after feature calc
    df.dropna(inplace=True)
    # print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Shape after feature calculation & final dropna: {df.shape}. Dropped {initial_rows - len(df)} rows.") # Less verbose
    return df

# --- Labeling Functions (Unchanged) ---
def create_direction_labels(df, n_bars):
    """Create labels based on future price direction."""
    if df.empty: return df
    print(f"{datetime.now()} - [DEBUG create_direction_labels] Creating direction labels looking ahead {n_bars} bars...")
    df['future_close'] = df['close'].shift(-n_bars); df.dropna(subset=['future_close'], inplace=True)
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df
    df['label'] = (df['future_close'] > df['close']).astype(int)
    df.drop(columns=['future_close'], inplace=True)
    print(f"{datetime.now()} - [DEBUG create_direction_labels] Created 'label' (direction). Shape: {df.shape}"); print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    return df

def create_significant_move_labels(df, n_bars, threshold_pct):
    """
    Create labels based on whether the price moves significantly up or down
    over the next n_bars. Label 1 = Sig UP, Label 0 = Sig DOWN.
    Insignificant moves are filtered out.
    """
    if df.empty: return df
    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Creating labels for moves > {threshold_pct:.3%} over {n_bars} bars...")
    # Ensure close is numeric before calculations
    if not pd.api.types.is_numeric_dtype(df['close']):
        print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Warning: 'close' column is not numeric. Attempting conversion.")
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if df.empty: print("DataFrame empty after close conversion/dropna."); return df

    df['future_close'] = df['close'].shift(-n_bars)
    df.dropna(subset=['future_close'], inplace=True) # Drop rows where future_close is NaN
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df

    # Use .copy() to avoid potential SettingWithCopyWarning if df is a slice
    close_safe = df['close'].copy().replace(0, 1e-10)
    price_change_pct = (df['future_close'] - close_safe) / close_safe

    conditions = [ price_change_pct > threshold_pct, price_change_pct < -threshold_pct ]
    choices = [1, 0] # 1 = Up, 0 = Down
    df['label'] = np.select(conditions, choices, default=-1) # Default -1 for insignificant

    initial_count = len(df)
    df = df[df['label'] != -1].copy() # Filter out insignificant moves, use .copy()
    filtered_count = initial_count - len(df)
    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Filtered out {filtered_count} insignificant moves ({filtered_count/max(1, initial_count):.2%}).")

    # Drop future_close only if it exists
    if 'future_close' in df.columns:
        df.drop(columns=['future_close'], inplace=True)

    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Created 'label' (significant moves only). Shape after filtering: {df.shape}")
    if not df.empty:
        print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    else:
        print("Warning: DataFrame is empty after filtering for significant moves.")
    return df


def generate_signals_and_labels_tp_sl(df, lookahead_bars, sl_mult, tp_mult, vol_z_long_thresh, vol_z_short_thresh):
    """Generate trading signals and label them based on TP/SL criteria."""
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] --- TP/SL LABELING FUNCTION START ---")
    signal_indices = []; labels = []; long_crossings = 0; short_crossings = 0
    if not isinstance(df.index, pd.DatetimeIndex): raise TypeError("Index must be DatetimeIndex")

    # Check required columns exist and are numeric
    required_cols = ['close', 'lower_band', 'upper_band', 'volume_z', 'atr', 'low', 'high']
    for col in required_cols:
        if col not in df.columns:
             print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] ERROR: Required column '{col}' missing. Cannot proceed.")
             return [], []
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Warning: Column '{col}' is not numeric. Attempting conversion.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop rows where conversion failed for essential columns
            if col in ['close', 'atr', 'low', 'high']: df.dropna(subset=[col], inplace=True)

    valid_indices = df.index
    if len(valid_indices) <= lookahead_bars:
        print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Warning: Not enough data ({len(valid_indices)} points) for TP/SL lookahead ({lookahead_bars}).")
        return [], []

    # Pre-calculate conditions for speed, handle potential NaNs from conversions
    df.dropna(subset=['close', 'lower_band', 'upper_band', 'volume_z', 'atr'], inplace=True) # Drop if core condition columns are NaN
    valid_indices = df.index # Update valid_indices after dropna
    if len(valid_indices) <= lookahead_bars: return [], [] # Check length again

    is_below_lower = df['close'] < df['lower_band']; is_above_upper = df['close'] > df['upper_band']
    is_vol_z_long = df['volume_z'] > vol_z_long_thresh; is_vol_z_short = df['volume_z'] < vol_z_short_thresh
    has_valid_atr = df['atr'] > 1e-9 # Already checked for NaN above

    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Starting loop for TP/SL checks ({len(valid_indices) - lookahead_bars} iterations)...")
    # Iterate up to the point where a full lookahead is possible
    # Use iloc for performance and to avoid index alignment issues
    df_values = df[['close', 'atr', 'low', 'high']].values # Get numpy array for faster access
    is_below_lower_values = is_below_lower.values
    is_above_upper_values = is_above_upper.values
    is_vol_z_long_values = is_vol_z_long.values
    is_vol_z_short_values = is_vol_z_short.values
    has_valid_atr_values = has_valid_atr.values

    for idx_loc in range(len(valid_indices) - lookahead_bars):
        current_index = valid_indices[idx_loc]
        current_close = df_values[idx_loc, 0]
        current_atr = df_values[idx_loc, 1]

        # Check for signal conditions using NumPy arrays
        is_long_signal = (is_below_lower_values[idx_loc] and is_vol_z_long_values[idx_loc] and has_valid_atr_values[idx_loc])
        is_short_signal = (is_above_upper_values[idx_loc] and is_vol_z_short_values[idx_loc] and has_valid_atr_values[idx_loc])

        if not (is_long_signal or is_short_signal): continue # Skip if no signal

        entry_price = current_close
        entry_type = 'long' if is_long_signal else 'short'

        # Calculate TP/SL levels
        if entry_type == 'long':
            long_crossings += 1; stop_loss = entry_price - sl_mult * current_atr; take_profit = entry_price + tp_mult * current_atr
        else: # Short signal
            short_crossings += 1; stop_loss = entry_price + sl_mult * current_atr; take_profit = entry_price - tp_mult * current_atr

        # Look ahead for TP/SL hits using NumPy slice
        future_slice = df_values[idx_loc + 1 : idx_loc + 1 + lookahead_bars, :]
        future_lows = future_slice[:, 2]
        future_highs = future_slice[:, 3]
        future_indices = valid_indices[idx_loc + 1 : idx_loc + 1 + lookahead_bars] # Keep track of index if needed

        hit_tp = False; hit_sl = False; first_tp_time_loc = -1; first_sl_time_loc = -1

        try:
            if entry_type == 'long':
                tp_hit_indices = np.where(future_highs >= take_profit)[0]
                sl_hit_indices = np.where(future_lows <= stop_loss)[0]
            else: # Short
                tp_hit_indices = np.where(future_lows <= take_profit)[0]
                sl_hit_indices = np.where(future_highs >= stop_loss)[0]

            if tp_hit_indices.size > 0: hit_tp = True; first_tp_time_loc = tp_hit_indices[0]
            if sl_hit_indices.size > 0: hit_sl = True; first_sl_time_loc = sl_hit_indices[0]

        except Exception as e_generic:
            print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Error during TP/SL check {current_index}: {e_generic}"); traceback.print_exc(); continue

        # Determine label based on which was hit first (using array index location)
        label = 0 # Default to loss (or no TP)
        if hit_tp and hit_sl: # Both hit
             if first_tp_time_loc <= first_sl_time_loc: label = 1 # TP hit first or at same time
        elif hit_tp: # Only TP hit
            label = 1

        signal_indices.append(current_index); labels.append(label)

    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Finished TP/SL loop.")
    print(f"Longs considered: {long_crossings}, Shorts considered: {short_crossings}, Total signals: {len(signal_indices)}")
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] --- TP/SL LABELING FUNCTION END ---")
    return signal_indices, labels


# --- Removed predict_gpu function ---

# --- >>> MODIFIED: Custom Cross-Validation Function <<< ---
def custom_cv_folds(X, y, base_model_params, current_iter_params, n_splits=5, random_state=42):
    """
    Custom cross-validation function using GPU-enabled model prediction.
    Uses TimeSeriesSplit for chronological data.
    Args:
        X (pd.DataFrame): Training features (should be pre-numeric).
        y (pd.Series): Training labels.
        base_model_params (dict): Base parameters for XGBClassifier (device, random_state etc).
        current_iter_params (dict): Parameters specific to this iteration from ParameterGrid.
        n_splits (int): Number of CV splits.
        random_state (int): Random state for model initialization (used in base_model_params).
    Returns:
        float: Mean cross-validation score (balanced accuracy).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []

    # Combine base params and iteration-specific params
    full_params = {**base_model_params, **current_iter_params}

    print(f"{datetime.now()} - [DEBUG custom_cv_folds] Starting CV for params: {current_iter_params}")

    for fold_idx, (train_index, val_index) in enumerate(tscv.split(X, y)):
        print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1}/{n_splits}")
        # Use .iloc with the main X (which is already numeric)
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        if len(X_train_fold) == 0 or len(X_val_fold) == 0:
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1} - ERROR: Empty train or validation set. Skipping fold.")
            continue # Skip this fold

        # --- Data already prepped before calling this function ---
        # X_train_fold and X_val_fold are slices of the pre-numeric X
        # No fold-specific median imputation needed here if X is globally cleaned

        try:
            # Train model for this fold
            model_fold = XGBClassifier(**full_params)
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Training model on {len(X_train_fold)} samples...")
            train_start = datetime.now()
            # Fit directly on the numeric fold data
            model_fold.fit(X_train_fold, y_train_fold)
            train_end = datetime.now()
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Training complete. Duration: {train_end - train_start}")

            # --- Make predictions directly using the model ---
            # The model (if device='cuda') should handle GPU prediction
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Predicting on validation set ({len(X_val_fold)} samples)...")
            predict_start = datetime.now()
            # Pass the numeric validation fold data directly
            y_pred_fold_proba = model_fold.predict_proba(X_val_fold)[:, 1]
            y_pred_fold = (y_pred_fold_proba > 0.5).astype(int) # Convert proba to class labels
            predict_end = datetime.now()
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Prediction complete. Duration: {predict_end - predict_start}")
            # --- End of Prediction Change ---

            # Calculate balanced accuracy
            fold_score = balanced_accuracy_score(y_val_fold, y_pred_fold)
            cv_results.append(fold_score)
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1} - Balanced Accuracy: {fold_score:.4f}")

        except Exception as e_fold:
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1} - ERROR during training or prediction: {e_fold}")
            traceback.print_exc()
            cv_results.append(-1.0) # Penalize parameters that cause errors

    # --- Calculate Mean Score ---
    if not cv_results: # Handle case where n_splits might be 0 or all folds skipped
        print(f"{datetime.now()} - [DEBUG custom_cv_folds] ERROR: No folds were executed for params: {current_iter_params}")
        return -1.0

    # Filter out failure scores (-1.0) before calculating mean
    valid_scores = [s for s in cv_results if s != -1.0]
    if not valid_scores: # Handle case where all folds failed
        print(f"{datetime.now()} - [DEBUG custom_cv_folds] ERROR: All folds failed for params: {current_iter_params}")
        mean_score = -1.0
    else:
        mean_score = np.mean(valid_scores)

    print(f"{datetime.now()} - [DEBUG custom_cv_folds] Finished CV for params {current_iter_params}. Mean Balanced Accuracy: {mean_score:.4f}")
    return mean_score


# --- Main Execution Block ---
if __name__ == "__main__":
    all_symbol_data_list = []
    spy_context_data = None # Initialize variable to hold processed SPY context features
    print(f"{datetime.now()} - [DEBUG Main] Script starting.")

    # --- Pre-process Market Index (SPY) Data ---
    print(f"\n{datetime.now()} - [DEBUG Main] --- Pre-processing Market Index: {MARKET_INDEX_SYMBOL} ---")
    spy_file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(MARKET_INDEX_SYMBOL))
    if not os.path.exists(spy_file_path):
        print(f"{datetime.now()} - [DEBUG Main] CRITICAL: Market index data file not found: {spy_file_path}. Cannot add context features. Exiting.")
        exit()

    try:
        print(f"{datetime.now()} - [DEBUG Main] Loading hourly data for {MARKET_INDEX_SYMBOL}...");
        load_start_time = datetime.now()
        # Use robust date loading logic (assuming it worked for hourly data previously)
        try:
            correct_date_format = '%Y-%m-%d %H:%M:%S' # Adapt if your format is different
            try:
                spy_data = pd.read_csv(spy_file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
            except ValueError: # Handle potential format mismatch
                 print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Info: read_csv parsing failed with format '{correct_date_format}'. Attempting without format.")
                 spy_data = pd.read_csv(spy_file_path, parse_dates=['date'], index_col='date')

            if not isinstance(spy_data.index, pd.DatetimeIndex):
                print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Warning: Index parsing failed. Attempting fallback...")
                spy_data = pd.read_csv(spy_file_path)
                if 'date' not in spy_data.columns: raise ValueError("Column 'date' not found.")
                # Try automatic parsing first, then specific format
                spy_data['date'] = pd.to_datetime(spy_data['date'], errors='coerce')
                if spy_data['date'].isnull().all(): # If auto fails, try specific format
                     print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Automatic date parsing failed, trying specific format '{correct_date_format}'.")
                     spy_data['date'] = pd.to_datetime(spy_data['date'], format=correct_date_format, errors='coerce')

                failed_count = spy_data['date'].isnull().sum()
                if failed_count > 0: print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Warning: {failed_count} dates failed parsing. Dropping.")
                spy_data.dropna(subset=['date'], inplace=True)
                if spy_data.empty: raise ValueError("No valid dates found after fallback parsing.")
                spy_data.set_index('date', inplace=True)

            spy_data.sort_index(inplace=True)
            load_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Loaded {len(spy_data)} rows ({MARKET_INDEX_SYMBOL}). Index type: {type(spy_data.index)}, Is monotonic: {spy_data.index.is_monotonic_increasing}. Load duration: {load_end_time - load_start_time}")
            if spy_data.empty: raise ValueError(f"Data empty after loading for {MARKET_INDEX_SYMBOL}.")

        except Exception as e:
             print(f"{datetime.now()} - [DEBUG Main] CRITICAL Error loading dates for {MARKET_INDEX_SYMBOL}: {e}"); traceback.print_exc(); exit() # Exit if SPY fails

        # Clean OHLCV
        print(f"{datetime.now()} - [DEBUG Main] Cleaning OHLCV columns for {MARKET_INDEX_SYMBOL}...");
        clean_start_time = datetime.now()
        initial_rows = len(spy_data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in spy_data.columns:
                if not pd.api.types.is_numeric_dtype(spy_data[col]): spy_data[col] = pd.to_numeric(spy_data[col], errors='coerce')
            else: print(f"{datetime.now()} - [DEBUG Main] Warning: Column {col} not found for {MARKET_INDEX_SYMBOL}.")
        spy_data.dropna(subset=[c for c in ohlcv_cols if c in spy_data.columns], inplace=True)
        clean_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_rows - len(spy_data)} rows due to non-numeric/NaN OHLCV ({MARKET_INDEX_SYMBOL}). Clean duration: {clean_end_time - clean_start_time}")
        if spy_data.empty: raise ValueError(f"Data empty after cleaning OHLCV for {MARKET_INDEX_SYMBOL}.")

        # Calculate Features for SPY
        print(f"{datetime.now()} - [DEBUG Main] Calculating features for {MARKET_INDEX_SYMBOL} (hourly data)...")
        feature_start_time = datetime.now()
        spy_features_df = calculate_features(spy_data.copy(), symbol_name=MARKET_INDEX_SYMBOL) # Use copy to be safe
        feature_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Finished calculating features for {MARKET_INDEX_SYMBOL}. Duration: {feature_end_time - feature_start_time}")
        if spy_features_df.empty or len(spy_features_df) < MIN_ROWS_PER_SYMBOL:
            raise ValueError(f"Insufficient data ({len(spy_features_df)} rows) after features for {MARKET_INDEX_SYMBOL}.")

        # Select and Rename Context Features
        missing_spy_context_cols = [col for col in spy_context_features_to_keep if col not in spy_features_df.columns]
        if missing_spy_context_cols:
            raise ValueError(f"Required SPY context features missing after calculation: {missing_spy_context_cols}")

        spy_context_data = spy_features_df[spy_context_features_to_keep].copy()
        spy_context_data.rename(columns={col: f"SPY_{col}" for col in spy_context_features_to_keep}, inplace=True)
        print(f"{datetime.now()} - [DEBUG Main] Selected and renamed {len(spy_context_data.columns)} SPY context features. Shape: {spy_context_data.shape}")
        # print(f"{datetime.now()} - [DEBUG Main] SPY Context Feature Columns: {spy_context_data.columns.tolist()}")

    except Exception as e:
        print(f"\n{datetime.now()} - [DEBUG Main] --- CRITICAL ERROR processing market index symbol: {MARKET_INDEX_SYMBOL} ---")
        print(f"Error Type: {type(e).__name__}, Message: {e}"); traceback.print_exc()
        print(f"{datetime.now()} - [DEBUG Main] --- Cannot proceed without market context. Exiting. ---"); exit()


    # --- Process Individual Symbols to Predict ---
    print(f"\n{datetime.now()} - [DEBUG Main] Starting data processing loop for {len(SYMBOLS_TO_PREDICT)} symbols to predict...")
    for symbol in SYMBOLS_TO_PREDICT: # Loop only through symbols we want to predict
        print(f"\n{datetime.now()} - [DEBUG Main] --- Processing Symbol: {symbol} ---")
        file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(symbol))
        if not os.path.exists(file_path): print(f"{datetime.now()} - [DEBUG Main] Warning: Data file not found: {file_path}. Skipping."); continue
        data = None
        try:
            # --- Load Data & Ensure DatetimeIndex ---
            print(f"{datetime.now()} - [DEBUG Main] Loading hourly data for {symbol}...");
            load_start_time = datetime.now()
            # Use robust date loading logic
            try:
                correct_date_format = '%Y-%m-%d %H:%M:%S' # Adapt if your format is different
                try:
                    data = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
                except ValueError:
                     print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Info: read_csv parsing failed with format '{correct_date_format}'. Attempting without format.")
                     data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

                if not isinstance(data.index, pd.DatetimeIndex):
                     print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Warning: Index parsing failed. Attempting fallback...")
                     data = pd.read_csv(file_path)
                     if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
                     data['date'] = pd.to_datetime(data['date'], errors='coerce')
                     if data['date'].isnull().all():
                          print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Automatic date parsing failed, trying specific format '{correct_date_format}'.")
                          data['date'] = pd.to_datetime(data['date'], format=correct_date_format, errors='coerce')

                     failed_count = data['date'].isnull().sum()
                     if failed_count > 0: print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Warning: {failed_count} dates failed parsing. Dropping.")
                     data.dropna(subset=['date'], inplace=True);
                     if data.empty: raise ValueError("No valid dates found after fallback parsing.")
                     data.set_index('date', inplace=True);

                data.sort_index(inplace=True)
                load_end_time = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Loaded {len(data)} rows ({symbol}). Index type: {type(data.index)}, Is monotonic: {data.index.is_monotonic_increasing}. Load duration: {load_end_time - load_start_time}")
                if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after loading for {symbol}."); continue
            except Exception as e: print(f"{datetime.now()} - [DEBUG Main] CRITICAL Error loading dates for {symbol}: {e}"); traceback.print_exc(); continue


            # --- Clean OHLCV ---
            print(f"{datetime.now()} - [DEBUG Main] Cleaning OHLCV columns for {symbol}...");
            clean_start_time = datetime.now()
            initial_rows = len(data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]): data[col] = pd.to_numeric(data[col], errors='coerce')
                else: print(f"{datetime.now()} - [DEBUG Main] Warning: Column {col} not found for {symbol}.")
            data.dropna(subset=[c for c in ohlcv_cols if c in data.columns], inplace=True)
            clean_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_rows - len(data)} rows due to non-numeric/NaN OHLCV ({symbol}). Clean duration: {clean_end_time - clean_start_time}")
            if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after cleaning OHLCV for {symbol}."); continue


            # --- Calculate Features (Operates on the hourly data now) ---
            print(f"{datetime.now()} - [DEBUG Main] Calculating features for {symbol} (hourly data)...")
            feature_start_time = datetime.now()
            data = calculate_features(data.copy(), symbol_name=symbol) # Use copy to be safe
            feature_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Finished calculating features for {symbol}. Shape: {data.shape}. Duration: {feature_end_time - feature_start_time}")
            if data.empty or len(data) < MIN_ROWS_PER_SYMBOL:
                print(f"{datetime.now()} - [DEBUG Main] Error or insufficient data ({len(data)} rows) after features for {symbol}. Skipping."); continue

            # --- >>> NEW: MERGE SPY CONTEXT FEATURES <<< ---
            if spy_context_data is not None:
                print(f"{datetime.now()} - [DEBUG Main] Merging SPY context features onto {symbol}...")
                merge_start = datetime.now()
                original_len = len(data)
                # Ensure both indices are DatetimeIndex before merging
                if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index)
                if not isinstance(spy_context_data.index, pd.DatetimeIndex): spy_context_data.index = pd.to_datetime(spy_context_data.index)
                # Use left merge to keep all rows from the symbol's data
                data = pd.merge(data, spy_context_data, left_index=True, right_index=True, how='left')
                merge_end = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Merge complete. Shape after merge: {data.shape}. Duration: {merge_end - merge_start}")

                # Check for NaNs introduced in SPY columns by merge and drop those rows
                spy_nan_mask = data[spy_context_feature_names].isnull().any(axis=1)
                if spy_nan_mask.any():
                    rows_to_drop = spy_nan_mask.sum()
                    print(f"{datetime.now()} - [DEBUG Main] Warning: {rows_to_drop} rows have missing SPY context after merge for {symbol}. Dropping these rows.")
                    data = data[~spy_nan_mask].copy() # Keep rows where SPY context is NOT NaN
                    print(f"{datetime.now()} - [DEBUG Main] Shape after dropping rows with missing SPY context: {data.shape}")

                if data.empty or len(data) < MIN_ROWS_PER_SYMBOL:
                     print(f"{datetime.now()} - [DEBUG Main] Error or insufficient data ({len(data)}) after merging SPY context for {symbol}. Skipping.")
                     continue
            else:
                 # This case should ideally not be reached if SPY processing is mandatory
                 print(f"{datetime.now()} - [DEBUG Main] CRITICAL Error: spy_context_data is None. Cannot proceed with merge for {symbol}. Exiting loop.")
                 continue # Or break, depending on desired behavior


            # Add symbol column BEFORE labeling
            data['symbol'] = symbol
            # print(f"{datetime.now()} - [DEBUG Main] Added 'symbol' column.") # Less verbose

            # --- Generate Labels ---
            print(f"{datetime.now()} - [DEBUG Main] Applying labeling method: '{LABELING_METHOD}' for {symbol} (hourly data)...")
            label_start_time = datetime.now()
            if LABELING_METHOD == 'direction':
                data = create_direction_labels(data, n_bars=LABEL_LOOKAHEAD_N_DIRECTION) # Ensure lookahead is appropriate for hourly
            elif LABELING_METHOD == 'significant_direction':
                 data = create_significant_move_labels(data,
                                                       n_bars=LABEL_LOOKAHEAD_N_SIG,
                                                       threshold_pct=LABEL_SIGNIFICANCE_THRESHOLD_PCT)
            elif LABELING_METHOD == 'tp_sl':
                print(f"{datetime.now()} - [DEBUG Main] Generating TP/SL signals and labels for {symbol} (hourly data)...")
                # Check if base features exist before calling TP/SL which relies on them
                missing_base_features_for_tpsl = [f for f in ['close', 'lower_band', 'upper_band', 'volume_z', 'atr'] if f not in data.columns]
                if missing_base_features_for_tpsl:
                     print(f"{datetime.now()} - [DEBUG Main] FATAL: Features needed for TP/SL ({missing_base_features_for_tpsl}) missing before signal generation for {symbol}. Skipping."); continue

                signal_indices, labels = generate_signals_and_labels_tp_sl(data, LOOKAHEAD_BARS_TP_SL, SL_MULT, TP_MULT, VOL_Z_LONG_THRESH, VOL_Z_SHORT_THRESH)
                label_end_time = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Finished generating TP/SL signals/labels for {symbol}. Duration: {label_end_time - label_start_time}")

                if not signal_indices: # Check if list is empty
                    print(f"{datetime.now()} - [DEBUG Main] Warning: No signals generated for {symbol} (TP/SL). Skipping."); continue
                if len(signal_indices) < MIN_ROWS_PER_SYMBOL:
                    print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient signals ({len(signal_indices)}) for {symbol}. Skipping."); continue

                # Select features *after* TP/SL generation, including new SPY features
                features_to_select_tpsl = base_feature_columns + spy_context_feature_names + ['symbol']
                # Ensure all selected features exist in the data *at the signal indices*
                data_at_signals = data.loc[signal_indices]
                missing_features_tpsl = [f for f in features_to_select_tpsl if f not in data_at_signals.columns]
                if missing_features_tpsl: print(f"{datetime.now()} - [DEBUG Main] FATAL: Features missing for TP/SL selection: {missing_features_tpsl}. Skipping."); continue

                try:
                    print(f"{datetime.now()} - [DEBUG Main] Selecting TP/SL signal features (including SPY context)...")
                    # Select only the rows corresponding to signals
                    signal_features = data_at_signals[features_to_select_tpsl].copy()
                except KeyError as ke: print(f"{datetime.now()} - [DEBUG Main] Error selecting signal features: {ke}"); traceback.print_exc(); continue

                signal_features['label'] = labels; initial_count = len(signal_features)
                # Drop rows if any feature (incl SPY) is NaN at signal time
                signal_features.dropna(inplace=True);
                print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_count - len(signal_features)} TP/SL rows due to NaNs in selected features.")
                if len(signal_features) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient TP/SL data ({len(signal_features)}). Skipping."); continue
                current_symbol_data = signal_features
                all_symbol_data_list.append(current_symbol_data)
                print(f"{datetime.now()} - [DEBUG Main] Successfully processed {symbol} (TP/SL). Added {len(current_symbol_data)} rows.")
                continue # Skip to next symbol

            else:
                print(f"{datetime.now()} - [DEBUG Main] Error: Unknown LABELING_METHOD '{LABELING_METHOD}'. Skipping symbol."); continue

            label_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Finished applying labeling method '{LABELING_METHOD}' for {symbol}. Duration: {label_end_time - label_start_time}")

            # --- Post-labeling processing (for non-TP/SL methods) ---
            if LABELING_METHOD in ['direction', 'significant_direction']:
                if data.empty or 'label' not in data.columns or data['label'].isnull().all():
                     print(f"{datetime.now()} - [DEBUG Main] Error: No valid labels or labels are all NaN for {symbol} after method '{LABELING_METHOD}'. Skipping.")
                     continue

                # Define columns to keep, including base features AND SPY context features
                cols_to_keep = base_feature_columns + spy_context_feature_names + ['label', 'symbol']
                # Verify columns exist before selection
                missing_final_features = [f for f in cols_to_keep if f not in data.columns]
                if missing_final_features: print(f"{datetime.now()} - [DEBUG Main] FATAL: Features missing post-labeling: {missing_final_features}. Skipping."); continue

                current_symbol_data = data[cols_to_keep].copy()
                # Check for NaNs in the final selected data *before* adding to list
                if current_symbol_data.isnull().values.any():
                    print(f"{datetime.now()} - [DEBUG Main] Warning: NaNs detected in final selection for {symbol}. Dropping rows with any NaN."); initial_count = len(current_symbol_data)
                    # Identify columns with NaNs for debugging
                    nan_cols = current_symbol_data.columns[current_symbol_data.isnull().any()].tolist()
                    print(f"{datetime.now()} - [DEBUG Main] Columns with NaNs causing drop: {nan_cols}")
                    current_symbol_data.dropna(inplace=True); print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_count - len(current_symbol_data)} rows.")

                if len(current_symbol_data) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient data ({len(current_symbol_data)}) after finalizing {LABELING_METHOD} labels. Skipping."); continue
                all_symbol_data_list.append(current_symbol_data)
                print(f"{datetime.now()} - [DEBUG Main] Successfully processed {symbol} ({LABELING_METHOD}). Added {len(current_symbol_data)} rows.")

        except Exception as e:
            print(f"\n{datetime.now()} - [DEBUG Main] --- Unexpected error processing symbol: {symbol} ---")
            print(f"Error Type: {type(e).__name__}, Message: {e}"); traceback.print_exc()
            print(f"{datetime.now()} - [DEBUG Main] --- Skipping rest of processing for {symbol} ---"); continue

    # --- Model Training and Evaluation ---
    print(f"\n{datetime.now()} - [DEBUG Main] --- Data Processing Complete ---")
    if not all_symbol_data_list: print(f"\n{datetime.now()} - [DEBUG Main] CRITICAL: No data collected. Cannot train."); exit()

    print(f"{datetime.now()} - [DEBUG Main] Combining data from {len(all_symbol_data_list)} symbols...")
    combine_start_time = datetime.now()
    combined_data = pd.concat(all_symbol_data_list)
    combine_end_time = datetime.now()
    print(f"{datetime.now()} - [DEBUG Main] Total rows before sorting: {len(combined_data)}. Combine duration: {combine_end_time - combine_start_time}")

    print(f"{datetime.now()} - [DEBUG Main] Sorting combined data by timestamp...")
    sort_start_time = datetime.now()
    combined_data.sort_index(inplace=True)
    sort_end_time = datetime.now()
    print(f"{datetime.now()} - [DEBUG Main] Combined data sorted. Index monotonic: {combined_data.index.is_monotonic_increasing}. Sort duration: {sort_end_time - sort_start_time}")
    print(f"{datetime.now()} - [DEBUG Main] Total rows after sorting: {len(combined_data)}")
    # print(f"{datetime.now()} - [DEBUG Main] Columns in combined_data: {combined_data.columns.tolist()}") # Verify SPY features are present

    print(f"{datetime.now()} - [DEBUG Main] One-hot encoding 'symbol' column...")
    ohe_start_time = datetime.now()
    try:
        combined_data = pd.get_dummies(combined_data, columns=['symbol'], prefix='sym', drop_first=False)
        ohe_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Data shape after one-hot encoding: {combined_data.shape}. OHE duration: {ohe_end_time - ohe_start_time}")

        # --- Convert OHE columns to integer type ---
        ohe_symbol_cols_post = [col for col in combined_data.columns if col.startswith('sym_')]
        if ohe_symbol_cols_post:
            combined_data[ohe_symbol_cols_post] = combined_data[ohe_symbol_cols_post].astype(int)
            print(f"{datetime.now()} - [DEBUG Main] Converted {len(ohe_symbol_cols_post)} OHE columns to int type.")
        else:
             print(f"{datetime.now()} - [DEBUG Main] Warning: No OHE columns found after get_dummies to convert type.")

    except KeyError: print("Error: 'symbol' column not found. Skipping encoding.")

    # --- Define Final Feature Columns (including base, SPY context, and OHE symbols) ---
    # Recalculate final_feature_columns based on actual columns present after OHE
    final_feature_columns = sorted([col for col in combined_data.columns if col != 'label'])

    print(f"{datetime.now()} - [DEBUG Main] Total final features identified: {len(final_feature_columns)}")
    # print(f"{datetime.now()} - [DEBUG Main] Final Feature Columns List: {final_feature_columns}") # Optional: Print full list

    # Check for NaNs again after potential drops during processing steps
    if combined_data.isnull().values.any():
        print(f"{datetime.now()} - [DEBUG Main] CRITICAL Warning: NaNs detected before split."); nan_counts = combined_data.isnull().sum(); print("NaN counts:\n", nan_counts[nan_counts > 0])
        print(f"{datetime.now()} - [DEBUG Main] Dropping rows with NaNs..."); combined_data.dropna(inplace=True); print(f"Rows remaining: {len(combined_data)}")
        if combined_data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Combined data empty after final NaN drop."); exit()

    print(f"{datetime.now()} - [DEBUG Main] Separating features (X) and labels (y)...")
    # Ensure 'label' is present before trying to access it
    if 'label' not in combined_data.columns: print(f"{datetime.now()} - [DEBUG Main] FATAL Error: 'label' column missing!"); exit()
    y = combined_data['label'].astype(int) # Ensure label is integer

    # X should contain all columns EXCEPT 'label'
    X = combined_data[final_feature_columns].copy() # Use the recalculated list based on actual columns

    missing_final_features_in_X = [f for f in final_feature_columns if f not in X.columns]
    if missing_final_features_in_X: print(f"{datetime.now()} - [DEBUG Main] FATAL Error: Missing final features IN X: {missing_final_features_in_X}"); exit()


    print(f"{datetime.now()} - [DEBUG Main] Feature matrix shape: {X.shape}")
    if len(X) == 0 or len(y) == 0: print("Error: X or y empty."); exit()
    if len(X) != len(y): print(f"CRITICAL Error: Length mismatch between X ({len(X)}) and y ({len(y)}) after NaN drop/separation."); exit()
    if len(y.unique()) < 2: print(f"Error: Only one class found: {y.unique()}"); exit()

    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train_orig = X.iloc[:split_index]; X_test_orig = X.iloc[split_index:] # Keep original before scaling
    y_train = y.iloc[:split_index]; y_test = y.iloc[split_index:]
    print(f"\n{datetime.now()} - [DEBUG Main] Splitting data chronologically:")
    print(f"Training set: {X_train_orig.index.min()} to {X_train_orig.index.max()}, size: {len(X_train_orig)}")
    print(f"Testing set : {X_test_orig.index.min()} to {X_test_orig.index.max()}, size: {len(X_test_orig)}")
    if len(X_train_orig) == 0 or len(X_test_orig) == 0: print("Error: Empty train/test set."); exit()

    # --- Define Numeric Columns for Scaling (including SPY features now) ---
    # OHE columns are typically NOT scaled
    numeric_cols_to_scale = [
        col for col in final_feature_columns if col not in
        ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'] and not col.startswith('sym_')
    ]
    print(f"{datetime.now()} - [DEBUG Main] Identified {len(numeric_cols_to_scale)} numeric columns for potential scaling (incl. SPY).")


    scaler = None
    X_train = X_train_orig.copy() # Work with copies for scaling
    X_test = X_test_orig.copy()

    if USE_FEATURE_SCALING:
        print(f"\n{datetime.now()} - [DEBUG Main] Applying StandardScaler...");
        scale_start_time = datetime.now()
        # Ensure columns exist in the actual train/test sets before scaling
        cols_to_scale_present_train = [col for col in numeric_cols_to_scale if col in X_train.columns]
        cols_to_scale_present_test = [col for col in numeric_cols_to_scale if col in X_test.columns]

        if not cols_to_scale_present_train:
             print(f"{datetime.now()} - [DEBUG Main] Warning: No numeric columns identified for scaling in the training set.")
        else:
            print(f"{datetime.now()} - [DEBUG Main] Scaling columns ({len(cols_to_scale_present_train)}): {cols_to_scale_present_train[:5]}...")
            scaler = StandardScaler()
            # Use .loc to avoid SettingWithCopyWarning
            # Ensure data is float before scaling
            X_train.loc[:, cols_to_scale_present_train] = scaler.fit_transform(X_train[cols_to_scale_present_train].astype(float))

            # Identify columns that were fit and are also present in the test set
            cols_to_transform_in_test = [col for col in cols_to_scale_present_train if col in cols_to_scale_present_test]

            if cols_to_transform_in_test:
                 print(f"{datetime.now()} - [DEBUG Main] Transforming {len(cols_to_transform_in_test)} columns in test set...")
                 # Ensure data is float before scaling
                 X_test.loc[:, cols_to_transform_in_test] = scaler.transform(X_test[cols_to_transform_in_test].astype(float))
            else:
                 print(f"{datetime.now()} - [DEBUG Main] Warning: No matching numeric columns found in test set for scaling transformation.")

            scale_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] StandardScaler applied. Duration: {scale_end_time - scale_start_time}")
    else:
        print(f"\n{datetime.now()} - [DEBUG Main] Skipping feature scaling (USE_FEATURE_SCALING is False).")


    # --- Manual Cross-Validation Loop (Replaces GridSearchCV) ---
    print(f"\n{datetime.now()} - [DEBUG Main] Setting up manual GridSearchCV for XGBoost (Manual CV Loop)...")

    # Check n_splits validity before loop
    min_samples_per_split = 2 # Minimum required by TimeSeriesSplit
    train_samples = len(X_train)
    if train_samples // (N_CV_SPLITS + 1) < min_samples_per_split:
        print(f"{datetime.now()} - [DEBUG Main] Warning: Training data size ({train_samples}) too small for {N_CV_SPLITS} splits with min {min_samples_per_split} samples per split. Reducing splits.")
        actual_n_splits = max(2, train_samples // min_samples_per_split -1) # Ensure at least 2 splits if possible
        if actual_n_splits < 2:
             print(f"{datetime.now()} - [DEBUG Main] ERROR: Cannot perform TimeSeriesSplit with less than 2 splits. Train samples: {train_samples}")
             exit() # Or handle differently
    else:
        actual_n_splits = N_CV_SPLITS
    print(f"{datetime.now()} - [DEBUG Main] Using TimeSeriesSplit with n_splits={actual_n_splits} in manual CV.")

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"{datetime.now()} - [DEBUG Main] Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # --- Define Base XGBoost Parameters (non-tunable) ---
    xgb_base_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss', # Still useful for internal model metrics if watched
        'scale_pos_weight': scale_pos_weight,
        'use_label_encoder': False, # Deprecated, set to False
        # 'enable_categorical': True # Consider adding if OHE columns cause issues, but int conversion might be enough
    }

    print(f"{datetime.now()} - [DEBUG Main] Checking XGBoost GPU support...")
    try:
        xgb_model_test_gpu = XGBClassifier(tree_method='hist', device='cuda') # Try to initialize with GPU
        print(f"{datetime.now()} - [DEBUG Main] XGBoost GPU support: Likely AVAILABLE.")
        del xgb_model_test_gpu # Clean up test object
    except Exception as e:
        print(f"{datetime.now()} - [DEBUG Main] XGBoost GPU support: NOT AVAILABLE. Error: {e}. Ensure XGBoost is compiled with CUDA support.")
        # Decide whether to exit or try CPU
        xgb_base_params['device'] = 'cpu' # Fallback to CPU?
        xgb_base_params['tree_method'] = 'hist' # Hist is generally good on CPU too
        print(f"{datetime.now()} - [DEBUG Main] Falling back to CPU ('device=cpu').")


    best_balanced_accuracy = -1.0 # Initialize with a value lower than any possible score
    best_params_overall = None
    all_cv_results_manual = {} # To store CV results for each param combination

    param_combinations = list(ParameterGrid(XGB_PARAM_GRID)) # Create list of param dicts
    total_combinations = len(param_combinations)
    print(f"{datetime.now()} - [DEBUG Main] Starting manual CV loop over {total_combinations} parameter combinations...")

    manual_gridsearch_start_time = datetime.now()

    # --- Prepare X_train for CV loop (ensure numeric, handle NaNs/Infs) ---
    # This is done once before the loop for efficiency
    print(f"{datetime.now()} - [DEBUG Main] Preparing numeric X_train for CV loop...")
    X_train_numeric_for_cv = X_train.apply(pd.to_numeric, errors='coerce')
    if X_train_numeric_for_cv.isnull().values.any():
         print(f"{datetime.now()} - [DEBUG Main] Warning: NaNs found in X_train before CV loop. Imputing with median.")
         cv_train_median = X_train_numeric_for_cv.median() # Calculate median once
         X_train_numeric_for_cv.fillna(cv_train_median, inplace=True)
    else:
        cv_train_median = None # No NaNs initially

    if not np.all(np.isfinite(X_train_numeric_for_cv)):
        print(f"{datetime.now()} - [DEBUG Main] WARNING: Non-finite values (inf) found in X_train before CV loop. Replacing with NaN and imputing.")
        X_train_numeric_for_cv.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Recalculate median if needed, or use previous one if it exists
        if cv_train_median is None: cv_train_median = X_train_numeric_for_cv.median()
        X_train_numeric_for_cv.fillna(cv_train_median, inplace=True) # Impute again
    print(f"{datetime.now()} - [DEBUG Main] Prepared numeric X_train (shape: {X_train_numeric_for_cv.shape}) for CV loop.")


    for i, params in enumerate(param_combinations): # Iterate through all parameter combinations
        print(f"\n{datetime.now()} - [DEBUG Main] --- Evaluating parameters ({i+1}/{total_combinations}): {params} ---")

        try: # Train and evaluate with current parameter set using the custom CV function
            # Pass the prepared numeric X_train to the custom function
            cv_score_mean = custom_cv_folds(
                X=X_train_numeric_for_cv,  # Use pre-processed numeric data
                y=y_train,
                base_model_params=xgb_base_params,
                current_iter_params=params,
                n_splits=actual_n_splits,
                random_state=RANDOM_STATE # Pass random state via base_params now
            )

            # Store mean CV score (use frozenset for dictionary key)
            param_key = tuple(sorted(params.items()))
            all_cv_results_manual[param_key] = cv_score_mean

            print(f"{datetime.now()} - [DEBUG Main] Mean CV Balanced Accuracy for params: {params} = {cv_score_mean:.4f}")

            # Check if current score is better than best (allowing for the initial -1)
            if cv_score_mean > best_balanced_accuracy:
                best_balanced_accuracy = cv_score_mean
                best_params_overall = params
                print(f"{datetime.now()} - [DEBUG Main] *** NEW BEST PARAMS FOUND ***: {best_params_overall} with Balanced Accuracy: {best_balanced_accuracy:.4f}")

        except Exception as e_cv: # Handle errors during CV for a parameter set
            print(f"{datetime.now()} - [DEBUG Main] ERROR during custom_cv_folds execution for params: {params}. Error: {e_cv}")
            traceback.print_exc()
            param_key = tuple(sorted(params.items()))
            all_cv_results_manual[param_key] = -1.0 # Mark as failed
            continue # Go to next parameter combination


    manual_gridsearch_end_time = datetime.now()
    print(f"\n{datetime.now()} - [DEBUG Main] Manual GridSearchCV (Manual Loop) finished. Duration: {manual_gridsearch_end_time - manual_gridsearch_start_time}")

    # --- Results of Manual CV ---
    if best_params_overall:
        print("Best parameters found:", best_params_overall)
        print(f"Best CV score (balanced_accuracy): {best_balanced_accuracy:.4f}")
    else:
        print(f"{datetime.now()} - [DEBUG Main] Warning: Manual GridSearchCV loop did not find any valid best parameters (possibly all failed or scores were <= -1.0). Check logs.")
        # Decide fallback: Use default params or exit?
        best_params_overall = {} # Use default params defined in xgb_base_params
        print(f"{datetime.now()} - [DEBUG Main] Proceeding with base XGBoost parameters as fallback.")


    # --- Train best model with best parameters on FULL training data ---
    print(f"\n{datetime.now()} - [DEBUG Main] Training final best model on full training data...")
    final_model_params = {**xgb_base_params, **best_params_overall} # Combine base and best tuned params
    best_model = XGBClassifier(**final_model_params)

    print(f"{datetime.now()} - [DEBUG Main] Final model parameters: {final_model_params}")

    # Use the same prepared numeric training data for the final fit
    print(f"{datetime.now()} - [DEBUG Main] Using pre-processed numeric X_train (shape: {X_train_numeric_for_cv.shape}) for final fit.")

    fit_start_time = datetime.now()
    try:
        best_model.fit(X_train_numeric_for_cv, y_train) # Train best model on prepared numeric training data
        fit_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Best model training finished. Duration: {fit_end_time - fit_start_time}")
    except Exception as final_fit_e:
        print(f"{datetime.now()} - [DEBUG Main] FATAL ERROR during final model training: {final_fit_e}")
        traceback.print_exc()
        best_model = None # Ensure no faulty model is used later


    # --- Evaluate best model on test set (using GPU prediction via Model) --- # <-- Changed comment slightly
    if best_model:
        print(f"\n{datetime.now()} - [DEBUG Main] Evaluating best model on test set...")

        # Ensure X_test is numeric for prediction
        numeric_X_test = X_test.apply(pd.to_numeric, errors='coerce')
        # Calculate median for imputation *once* using the final training data
        final_train_median = X_train_numeric_for_cv.median()

        if numeric_X_test.isnull().values.any():
            print(f"{datetime.now()} - [DEBUG Main] Warning: NaNs found in X_test before prediction. Imputing with training median.")
            numeric_X_test.fillna(final_train_median, inplace=True) # Impute using training median

        if not np.all(np.isfinite(numeric_X_test)):
            print(f"{datetime.now()} - [DEBUG Main] WARNING: Non-finite values (inf) found in X_test before prediction. Replacing with NaN and imputing.")
            numeric_X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
            numeric_X_test.fillna(final_train_median, inplace=True) # Impute using training median

        print(f"{datetime.now()} - [DEBUG Main] Predicting on test data (shape: {numeric_X_test.shape}) using the best model...")
        try:
            # --- REMOVED DMatrix Creation ---

            eval_start_time = datetime.now()
            # --- Predict directly on the DataFrame ---
            # The best_model (if device='cuda') will handle GPU usage
            y_pred_proba = best_model.predict_proba(numeric_X_test)[:, 1] # Get probability of class 1
            y_pred = (y_pred_proba > 0.5).astype(int) # Convert to class labels
            # --- End of Prediction Change ---
            eval_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Prediction finished. Duration: {eval_end_time - eval_start_time}")

            # --- Evaluation Metrics ---
            print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"Test Set Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}") # Also report balanced accuracy
            print("Test Set Classification Report:\n", classification_report(y_test, y_pred, target_names=['Significant Down (0)', 'Significant Up (1)'], zero_division=0))

            # --- Feature Importances ---
            print("\nFeature Importances:")
            try:
                # Feature names should correspond to the columns in X_train_numeric_for_cv (the data used for final fit)
                feature_names = X_train_numeric_for_cv.columns
                importances = best_model.feature_importances_
                if len(feature_names) != len(importances):
                     print(f"{datetime.now()} - [DEBUG Main] Warning: Mismatch between feature name count ({len(feature_names)}) and importance count ({len(importances)}). Using generic names.")
                     feature_names = [f'feature_{i}' for i in range(len(importances))]

                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                print(feature_importance_df.head(30).to_string()) # Print top 30
            except Exception as fi_e: print(f"Could not get feature importances: {fi_e}")

            # --- Save Model, Features, Scaler ---
            print(f"\n{datetime.now()} - [DEBUG Main] Saving artifacts...")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
            model_filename = f'trading_xgb_model_1h_sig_{timestamp_str}.joblib'
            feature_list_filename = f'trading_model_features_1h_sig_{timestamp_str}.list'
            scaler_filename = f'trading_feature_scaler_1h_sig_{timestamp_str}.joblib'

            joblib.dump(best_model, model_filename); print(f"Model saved to '{model_filename}'")

            try:
                 # Save the features used by the *final* model
                 current_features = list(X_train_numeric_for_cv.columns) # Use columns from the data fed to final fit
                 with open(feature_list_filename, 'w') as f:
                      for feature in current_features: f.write(f"{feature}\n")
                 print(f"Feature list ({len(current_features)}) saved to '{feature_list_filename}'")
            except Exception as fl_e: print(f"Error saving feature list: {fl_e}")

            if scaler: # Only save if scaling was used and successful
                joblib.dump(scaler, scaler_filename); print(f"Scaler saved to '{scaler_filename}'")

        except Exception as eval_e:
             print(f"{datetime.now()} - [DEBUG Main] ERROR during evaluation or saving: {eval_e}")
             traceback.print_exc()

    else:
        print(f"\n{datetime.now()} - [DEBUG Main] Skipping evaluation and saving as final model training failed.")

    print(f"\n{datetime.now()} - [DEBUG Main] --- Script Finished ---")