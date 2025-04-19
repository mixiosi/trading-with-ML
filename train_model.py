# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np # Ensure numpy is imported
import talib
from xgboost import XGBClassifier
# Import necessary classes for manual CV loop
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import joblib
import os
import pandas.api.types # To check dtypes robustly
import traceback # For detailed error prints
# from sklearn.model_selection import RandomizedSearchCV # Keep commented unless switching
from datetime import datetime # <-- ADDED for timestamps
import xgboost as xgb # <--- ENSURE THIS IS IMPORTED AT THE TOP OF YOUR SCRIPT

# --- NEW: Import SMOTE ---
from imblearn.over_sampling import SMOTE


# --- Configuration ---
# Define symbols for training
ALL_SYMBOLS = ['AAPL', 'AMD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'QQQ', 'SPY', 'TSLA']
MARKET_INDEX_SYMBOL = 'SPY' # Symbol to use for market context
SYMBOLS_TO_PREDICT = [s for s in ALL_SYMBOLS if s != MARKET_INDEX_SYMBOL] # Symbols we want predictions for

DATA_DIR = '.' # Directory where CSV files are located
FILENAME_TEMPLATE = '{}_1h_historical_data.csv' # Input files are hourly

# --- Feature Calculation Parameters (Interpret periods relative to 1-hour bars now) ---
# Consider experimenting with these values based on feature importance or domain knowledge
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
# Consider experimenting with these thresholds and lookahead periods
LABELING_METHOD = 'significant_direction' # Options: 'direction', 'significant_direction', 'tp_sl'

# Parameters for 'direction' labeling
# LABEL_LOOKAHEAD_N_DIRECTION = 12 # Example: 12 hours if using 'direction'

# Parameters for 'significant_direction' labeling
LABEL_LOOKAHEAD_N_SIG = 6 # Lookahead 6 hours on 1-hour bars. >>> Consider testing shorter (e.g., 4) or longer (e.g., 8, 10) horizons <<<
# --- Modified: Lowered threshold ---
LABEL_SIGNIFICANCE_THRESHOLD_PCT = 0.01 # <<< WAS 0.015. Try 0.0075 or 0.02 next if this doesn't improve things.

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
# --- Experiment with SMOTE ---
USE_SMOTE_IN_CV = True # Set to False to use 'scale_pos_weight' instead

# --- XGBoost GridSearch Parameters ---
# NOTE: This large grid increases training time. Consider reducing it.
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.5, 1.0],
    'min_child_weight': [1, 3, 5]
}

# --- Define Base Feature Columns ---
# >>> Consider reviewing feature importance <<<
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
# >>> Review feature importance <<<
spy_context_features_to_keep = [
    'return_1', 'rsi', 'atr', 'sma', 'volume_z', 'macd_hist'
]
spy_context_feature_names = [f"SPY_{col}" for col in spy_context_features_to_keep]


# --- Feature Calculation Function (Ensure identical to backtester) ---
def calculate_features(df, symbol_name="Data"):
    """Calculate technical indicators and enhanced features for the dataset."""
    # (Function content remains the same as before - No changes here)
    cols_to_check = ['high', 'low', 'close', 'open', 'volume']
    for col in cols_to_check:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
    if df.empty: return df

    if not isinstance(df.index, pd.DatetimeIndex):
         print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) CRITICAL WARNING: Index is NOT a DatetimeIndex before feature calculation!")
         try:
            # Attempt conversion using errors='coerce' to handle potential issues gracefully
            original_index_name = df.index.name # Store original name if it exists
            df.index = pd.to_datetime(df.index, errors='coerce')
            df.index.name = original_index_name # Restore original name
            # Check if conversion resulted in NaTs and handle them if needed (e.g., drop rows)
            if df.index.hasnans:
                 print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Index conversion resulted in NaT values. Dropping affected rows.")
                 df = df[~df.index.isna()] # Drop rows with NaT index
            if not isinstance(df.index, pd.DatetimeIndex) or df.empty: raise ValueError("Failed conversion or empty after NaT drop.")
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Index converted within calculate_features."); df.sort_index(inplace=True)
         except Exception as e:
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) ERROR: Could not ensure DatetimeIndex: {e}. Time features fail.");
            df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan
    else:
        if not df.index.is_monotonic_increasing: print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: DatetimeIndex not sorted."); df.sort_index(inplace=True)

    # --- SMA / Bollinger Bands ---
    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']; df['lower_band'] = df['sma'] - 2 * df['std']

    # --- TA-Lib Indicators ---
    # Prepare numpy arrays for TA-Lib (ensure they are float64)
    high_prices = df['high'].astype(np.float64).values
    low_prices = df['low'].astype(np.float64).values
    close_prices = df['close'].astype(np.float64).values
    min_len_for_talib = max(SMA_WINDOW, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, STOCH_K, ROC_PERIOD) + 10 # Add some buffer

    if len(close_prices) >= min_len_for_talib:
        try:
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
            df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
            macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
            df['macd']=macd; df['macd_signal']=macdsignal; df['macd_hist']=macdhist
            # Avoid division by zero in BB Width calculation
            sma_safe = df['sma'].replace(0, 1e-10)
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / sma_safe
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=STOCH_K, slowk_period=STOCH_SMOOTH, slowk_matype=0, slowd_period=STOCH_D, slowd_matype=0)
            df['roc'] = talib.ROC(close_prices, timeperiod=ROC_PERIOD)
        except Exception as talib_e:
             print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Error during TA-Lib calculations: {talib_e}. Assigning NaNs.")
             traceback.print_exc() # Print traceback for TA-Lib errors
             for col in ['atr','rsi','macd','macd_signal','macd_hist','bb_width','stoch_k','stoch_d','roc']: df[col]=np.nan
    else:
        print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Not enough data ({len(close_prices)}) for TA-Lib. Skipping TA-Lib features.");
        for col in ['atr','rsi','macd','macd_signal','macd_hist','bb_width','stoch_k','stoch_d','roc']: df[col]=np.nan

    # --- Volume Z-Score ---
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    volume_std5 = df['volume'].rolling(window=5).std().replace(0, 1e-10) # Use std dev instead of mean for denominator, handle 0 std
    df['volume_z'] = (df['volume'] - df['volume_sma5']) / volume_std5 # Z-score based on std dev

    # --- Lagged RSI ---
    df['rsi_lag1'] = df['rsi'].shift(1)

    # --- Time Features ---
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7.0)
        df.drop(columns=['hour', 'dayofweek'], inplace=True) # Drop intermediate columns
    elif 'hour_sin' not in df.columns: # Add NaNs if index failed AND they weren't added before
        print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Cannot create time features due to non-DatetimeIndex.")
        df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan

    # --- Returns ---
    df['return_1'] = df['close'].pct_change(1)

    # --- Lagged Features ---
    features_to_lag = ['return_1', 'macd_hist', 'stoch_k', 'atr', 'bb_width']
    new_lag_cols = []
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in LAG_PERIODS:
                col_name = f'{feature}_lag{lag}'
                df[col_name] = df[feature].shift(lag)
                new_lag_cols.append(col_name)
        else:
            # This warning might occur if TA-Lib failed earlier
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Feature '{feature}' not found for lagging (might be due to earlier TA-Lib error).")

    # --- Drop NaNs *introduced by calculations* within this function ---
    # Calculate initial NaNs from shifts/rolling windows
    max_lookback = max([SMA_WINDOW, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, STOCH_K, ROC_PERIOD] + LAG_PERIODS)
    # Keep initial dropna based on calculations within this function
    df.dropna(inplace=True)
    print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Features calculated. Shape after internal dropna: {df.shape}")
    return df


# --- Labeling Functions (Unchanged) ---
def create_direction_labels(df, n_bars):
    # (Function content remains the same as before)
    if df.empty: return df
    print(f"{datetime.now()} - [DEBUG create_direction_labels] Creating direction labels looking ahead {n_bars} bars...")
    df['future_close'] = df['close'].shift(-n_bars); df.dropna(subset=['future_close'], inplace=True)
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df
    df['label'] = (df['future_close'] > df['close']).astype(int)
    df.drop(columns=['future_close'], inplace=True)
    print(f"{datetime.now()} - [DEBUG create_direction_labels] Created 'label' (direction). Shape: {df.shape}"); print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    return df

def create_significant_move_labels(df, n_bars, threshold_pct):
    # (Function content remains the same as before)
    if df.empty: return df
    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Creating labels for moves > {threshold_pct:.3%} over {n_bars} bars...")
    if not pd.api.types.is_numeric_dtype(df['close']):
        print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Warning: 'close' column is not numeric. Attempting conversion.")
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if df.empty: print("DataFrame empty after close conversion/dropna."); return df

    # Calculate future close safely
    df['future_close'] = df['close'].shift(-n_bars)
    df.dropna(subset=['future_close'], inplace=True)
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df

    # Avoid division by zero or issues with zero prices
    close_safe = df['close'].copy()
    # Replace non-positive values with NaN before calculating change, then handle NaN results
    close_safe[close_safe <= 0] = np.nan
    price_change_pct = (df['future_close'] - close_safe) / close_safe
    price_change_pct.dropna(inplace=True) # Remove NaNs resulting from non-positive close prices

    # Re-align index after potential drops
    df = df.loc[price_change_pct.index].copy()
    if df.empty: print("DataFrame empty after handling non-positive close prices."); return df

    conditions = [ price_change_pct > threshold_pct, price_change_pct < -threshold_pct ]
    choices = [1, 0] # 1 = Significant Up, 0 = Significant Down
    # Use the calculated price_change_pct (which is aligned with df)
    df['label'] = np.select(conditions, choices, default=-1) # -1 = Insignificant move

    initial_count = len(df)
    # --- Filter out the insignificant moves (-1) ---
    df = df[df['label'] != -1].copy()
    filtered_count = initial_count - len(df)
    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Filtered out {filtered_count} insignificant moves ({filtered_count/max(1, initial_count):.2%}).")

    if 'future_close' in df.columns:
        df.drop(columns=['future_close'], inplace=True)

    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Created 'label' (significant moves only). Shape after filtering: {df.shape}")
    if not df.empty:
        # Print distribution AFTER filtering
        print(f"Label distribution (after filtering): {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    else:
        print("Warning: DataFrame is empty after filtering for significant moves.")
    return df


def generate_signals_and_labels_tp_sl(df, lookahead_bars, sl_mult, tp_mult, vol_z_long_thresh, vol_z_short_thresh):
    # (Function content remains the same as before)
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] --- TP/SL LABELING FUNCTION START ---")
    signal_indices = []; labels = []; long_crossings = 0; short_crossings = 0
    if not isinstance(df.index, pd.DatetimeIndex): raise TypeError("Index must be DatetimeIndex")

    required_cols = ['close', 'lower_band', 'upper_band', 'volume_z', 'atr', 'low', 'high']
    for col in required_cols:
        if col not in df.columns:
             print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] ERROR: Required column '{col}' missing. Cannot proceed.")
             return [], []
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Warning: Column '{col}' is not numeric. Attempting conversion.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop rows where essential numeric columns became NaN after conversion
            if col in ['close', 'atr', 'low', 'high']: df.dropna(subset=[col], inplace=True)

    # Ensure ATR is positive before proceeding
    df = df[df['atr'] > 1e-9].copy()

    valid_indices = df.index
    if len(valid_indices) <= lookahead_bars:
        print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Warning: Not enough data ({len(valid_indices)} points) for TP/SL lookahead ({lookahead_bars}) after initial cleaning/ATR filter.")
        return [], []

    # Drop other potential NaNs before loop
    df.dropna(subset=['close', 'lower_band', 'upper_band', 'volume_z', 'atr', 'high', 'low'], inplace=True)
    valid_indices = df.index
    if len(valid_indices) <= lookahead_bars: return [], [] # Check again after dropna

    # Conditions for signals
    is_below_lower = df['close'] < df['lower_band']; is_above_upper = df['close'] > df['upper_band']
    is_vol_z_long = df['volume_z'] > vol_z_long_thresh; is_vol_z_short = df['volume_z'] < vol_z_short_thresh
    # has_valid_atr already applied by filtering df

    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Starting loop for TP/SL checks ({len(valid_indices) - lookahead_bars} iterations)...")
    # Pre-fetch necessary columns as numpy arrays for speed
    close_vals = df['close'].values
    atr_vals = df['atr'].values
    low_vals = df['low'].values
    high_vals = df['high'].values
    is_below_lower_vals = is_below_lower.values
    is_above_upper_vals = is_above_upper.values
    is_vol_z_long_vals = is_vol_z_long.values
    is_vol_z_short_vals = is_vol_z_short.values

    for idx_loc in range(len(valid_indices) - lookahead_bars):
        current_index = valid_indices[idx_loc]
        current_close = close_vals[idx_loc]
        current_atr = atr_vals[idx_loc]

        # Determine if a signal occurs at this bar
        is_long_signal = (is_below_lower_vals[idx_loc] and is_vol_z_long_vals[idx_loc])
        is_short_signal = (is_above_upper_vals[idx_loc] and is_vol_z_short_vals[idx_loc])

        if not (is_long_signal or is_short_signal): continue # Skip if no signal

        entry_price = current_close
        entry_type = 'long' if is_long_signal else 'short'

        # Calculate SL/TP levels
        if entry_type == 'long':
            long_crossings += 1
            stop_loss = entry_price - sl_mult * current_atr
            take_profit = entry_price + tp_mult * current_atr
        else: # Short
            short_crossings += 1
            stop_loss = entry_price + sl_mult * current_atr
            take_profit = entry_price - tp_mult * current_atr

        # Look into the future bars
        future_lows = low_vals[idx_loc + 1 : idx_loc + 1 + lookahead_bars]
        future_highs = high_vals[idx_loc + 1 : idx_loc + 1 + lookahead_bars]

        hit_tp = False; hit_sl = False; first_tp_time_loc = -1; first_sl_time_loc = -1

        try:
            # Check for TP/SL hits in the future window
            if entry_type == 'long':
                tp_hit_indices = np.where(future_highs >= take_profit)[0]
                sl_hit_indices = np.where(future_lows <= stop_loss)[0]
            else: # Short
                tp_hit_indices = np.where(future_lows <= take_profit)[0]
                sl_hit_indices = np.where(future_highs >= stop_loss)[0]

            # Record if and when they were first hit
            if tp_hit_indices.size > 0: hit_tp = True; first_tp_time_loc = tp_hit_indices[0]
            if sl_hit_indices.size > 0: hit_sl = True; first_sl_time_loc = sl_hit_indices[0]

        except Exception as e_generic:
            print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Error during TP/SL check {current_index}: {e_generic}"); traceback.print_exc(); continue

        # Determine the label based on outcome
        label = 0 # Default label is Loss (hit SL or neither within lookahead)
        if hit_tp and hit_sl:
             # If both hit, label as win only if TP was hit first or on the same bar
             if first_tp_time_loc <= first_sl_time_loc: label = 1
        elif hit_tp:
             # If only TP was hit -> Win
            label = 1

        signal_indices.append(current_index); labels.append(label)

    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Finished TP/SL loop.")
    print(f"Long Signals Checked: {long_crossings}, Short Signals Checked: {short_crossings}")
    print(f"Total Labeled Events: {len(signal_indices)}, Wins (Label=1): {sum(1 for lbl in labels if lbl == 1)}")
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] --- TP/SL LABELING FUNCTION END ---")
    return signal_indices, labels


# --- Custom Cross-Validation Function (No changes needed here for the fix) ---
def custom_cv_folds(X, y, base_model_params, current_iter_params, n_splits=5, use_smote=False, random_state=42):
    """
    Custom cross-validation function using GPU-enabled model prediction.
    Uses TimeSeriesSplit for chronological data. Optionally uses SMOTE.
    Args:
        X (pd.DataFrame): Training features (should be pre-numeric).
        y (pd.Series): Training labels.
        base_model_params (dict): Base parameters for XGBClassifier (device, random_state etc).
        current_iter_params (dict): Parameters specific to this iteration from ParameterGrid.
        n_splits (int): Number of CV splits.
        use_smote (bool): Whether to apply SMOTE to the training fold.
        random_state (int): Random state for SMOTE and model initialization if not in base_params.
    Returns:
        float: Mean cross-validation score (balanced accuracy).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []

    # Combine base params and iteration-specific params
    full_params = {**base_model_params, **current_iter_params}
    if 'random_state' not in full_params:
        full_params['random_state'] = random_state

    smote_random_state = full_params.get('random_state', random_state)

    print(f"{datetime.now()} - [DEBUG custom_cv_folds] Starting CV for params: {current_iter_params} (SMOTE={use_smote})")

    for fold_idx, (train_index, val_index) in enumerate(tscv.split(X, y)):
        print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1}/{n_splits}")
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        if len(X_train_fold) < 5 or len(X_val_fold) == 0:
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1} - WARNING: Insufficient train ({len(X_train_fold)}) or validation ({len(X_val_fold)}) samples. Skipping fold.")
            continue

        X_train_fold_processed = X_train_fold.copy()
        y_train_fold_processed = y_train_fold.copy()

        try:
            # Apply SMOTE within the fold if enabled
            if use_smote:
                print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Applying SMOTE to training fold {fold_idx+1}...")
                print(f"{datetime.now()} - [DEBUG custom_cv_folds]       Shape before SMOTE: {X_train_fold_processed.shape}, Label dist: {dict(np.round(y_train_fold_processed.value_counts(normalize=True) * 100, 1))}")
                n_minority_samples = y_train_fold_processed.value_counts().min()
                # Ensure k_neighbors is valid (at least 1 and less than minority samples)
                k_neighbors = max(1, min(4, n_minority_samples - 1))
                if n_minority_samples <= k_neighbors: # Check if enough samples for chosen k
                    print(f"{datetime.now()} - [DEBUG custom_cv_folds]     WARNING: Cannot apply SMOTE for fold {fold_idx+1}, too few minority samples ({n_minority_samples}) for k_neighbors={k_neighbors}. Skipping SMOTE for this fold.")
                else:
                    try:
                        smote = SMOTE(random_state=smote_random_state, k_neighbors=k_neighbors)
                        X_train_fold_processed, y_train_fold_processed = smote.fit_resample(X_train_fold_processed, y_train_fold_processed)
                        print(f"{datetime.now()} - [DEBUG custom_cv_folds]       Shape after SMOTE: {X_train_fold_processed.shape}, Label dist: {dict(np.round(y_train_fold_processed.value_counts(normalize=True) * 100, 1))}")
                    except ValueError as smote_err:
                         # Catch other potential SMOTE errors
                         print(f"{datetime.now()} - [DEBUG custom_cv_folds]     WARNING: SMOTE failed for fold {fold_idx+1}: {smote_err}. Proceeding without resampling for this fold.")
                         X_train_fold_processed = X_train_fold.copy()
                         y_train_fold_processed = y_train_fold.copy()

            # Train model for this fold
            model_fold = XGBClassifier(**full_params)
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Training model on {len(X_train_fold_processed)} samples...")
            train_start = datetime.now()
            # Use original validation fold for eval_set, NOT SMOTE'd data
            eval_set = [(X_val_fold.astype(np.float32), y_val_fold)] # Ensure correct dtype for eval_set
            model_fold.fit(X_train_fold_processed.astype(np.float32), y_train_fold_processed, # Ensure correct dtype for training
                           eval_set=eval_set,
                           verbose=False) # Set verbose=True to see fold details
            train_end = datetime.now()
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Training complete. Duration: {train_end - train_start}")

            # Make predictions on the ORIGINAL validation set
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Predicting on validation set ({len(X_val_fold)} samples)...")
            predict_start = datetime.now()
            y_pred_fold_proba = model_fold.predict_proba(X_val_fold.astype(np.float32))[:, 1] # Ensure correct dtype
            y_pred_fold = (y_pred_fold_proba > 0.5).astype(int) # Standard 0.5 threshold for CV evaluation
            predict_end = datetime.now()
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]     Prediction complete. Duration: {predict_end - predict_start}")

            # Calculate balanced accuracy on the original validation labels
            fold_score = balanced_accuracy_score(y_val_fold, y_pred_fold)
            cv_results.append(fold_score)
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1} - Balanced Accuracy: {fold_score:.4f}")

        except Exception as e_fold:
            print(f"{datetime.now()} - [DEBUG custom_cv_folds]   Fold {fold_idx+1} - ERROR during training or prediction: {e_fold}")
            traceback.print_exc()
            cv_results.append(-1.0) # Penalize parameters that cause errors

    # Calculate Mean Score
    if not cv_results:
        print(f"{datetime.now()} - [DEBUG custom_cv_folds] ERROR: No folds were successfully executed for params: {current_iter_params}")
        return -1.0 # Indicate failure

    valid_scores = [s for s in cv_results if s != -1.0]
    if not valid_scores:
        print(f"{datetime.now()} - [DEBUG custom_cv_folds] ERROR: All folds failed for params: {current_iter_params}")
        mean_score = -1.0
    else:
        mean_score = np.mean(valid_scores)

    print(f"{datetime.now()} - [DEBUG custom_cv_folds] Finished CV for params {current_iter_params}. Mean Balanced Accuracy: {mean_score:.4f}")
    return mean_score


# --- Main Execution Block ---
if __name__ == "__main__":
    all_symbol_data_list = []
    spy_context_data = None
    print(f"{datetime.now()} - [DEBUG Main] Script starting.")

    # --- Pre-process Market Index (SPY) Data ---
    print(f"\n{datetime.now()} - [DEBUG Main] --- Pre-processing Market Index: {MARKET_INDEX_SYMBOL} ---")
    spy_file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(MARKET_INDEX_SYMBOL))
    if not os.path.exists(spy_file_path):
        print(f"{datetime.now()} - [DEBUG Main] CRITICAL: Market index data file not found: {spy_file_path}. Cannot add context features. Exiting.")
        exit()
    try:
        # --- SPY Loading ---
        print(f"{datetime.now()} - [DEBUG Main] Loading hourly data for {MARKET_INDEX_SYMBOL}...");
        load_start_time = datetime.now()
        try:
            correct_date_format = '%Y-%m-%d %H:%M:%S'
            try: spy_data = pd.read_csv(spy_file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
            except ValueError:
                 print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Info: read_csv parsing failed with format '{correct_date_format}'. Attempting without format.")
                 spy_data = pd.read_csv(spy_file_path, parse_dates=['date'], index_col='date')
            # --- Fallback and index validation ---
            if not isinstance(spy_data.index, pd.DatetimeIndex):
                print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Warning: Index parsing failed initially. Attempting fallback...")
                spy_data = pd.read_csv(spy_file_path) # Reload without index_col
                if 'date' not in spy_data.columns: raise ValueError("Column 'date' not found for fallback.")
                spy_data['date'] = pd.to_datetime(spy_data['date'], errors='coerce') # Try auto-parsing first
                if spy_data['date'].isnull().all(): # If auto failed, try specific format
                     print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Automatic date parsing failed, trying specific format '{correct_date_format}'.")
                     spy_data['date'] = pd.to_datetime(spy_data['date'], format=correct_date_format, errors='coerce')
                failed_count = spy_data['date'].isnull().sum()
                if failed_count > 0: print(f"{datetime.now()} - [DEBUG Main] ({MARKET_INDEX_SYMBOL}) Warning: {failed_count} dates failed parsing during fallback. Dropping.")
                spy_data.dropna(subset=['date'], inplace=True) # Drop rows where date parsing failed
                if spy_data.empty: raise ValueError("No valid dates found after fallback parsing.")
                spy_data.set_index('date', inplace=True) # Set the cleaned date column as index
            spy_data.sort_index(inplace=True) # Ensure chronological order
            load_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Loaded {len(spy_data)} rows ({MARKET_INDEX_SYMBOL}). Index type: {type(spy_data.index)}, Is monotonic: {spy_data.index.is_monotonic_increasing}. Load duration: {load_end_time - load_start_time}")
            if spy_data.empty: raise ValueError(f"Data empty after loading for {MARKET_INDEX_SYMBOL}.")
        except Exception as e: print(f"{datetime.now()} - [DEBUG Main] CRITICAL Error loading dates for {MARKET_INDEX_SYMBOL}: {e}"); traceback.print_exc(); exit()

        # --- SPY Cleaning OHLCV ---
        print(f"{datetime.now()} - [DEBUG Main] Cleaning OHLCV columns for {MARKET_INDEX_SYMBOL}...");
        clean_start_time = datetime.now()
        initial_rows = len(spy_data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in spy_data.columns:
                if not pd.api.types.is_numeric_dtype(spy_data[col]): spy_data[col] = pd.to_numeric(spy_data[col], errors='coerce')
            else: print(f"{datetime.now()} - [DEBUG Main] Warning: Column {col} not found for {MARKET_INDEX_SYMBOL}.")
        spy_data.dropna(subset=[c for c in ohlcv_cols if c in spy_data.columns], inplace=True) # Drop rows with NaN in present OHLCV cols
        clean_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_rows - len(spy_data)} rows due to non-numeric/NaN OHLCV ({MARKET_INDEX_SYMBOL}). Clean duration: {clean_end_time - clean_start_time}")
        if spy_data.empty: raise ValueError(f"Data empty after cleaning OHLCV for {MARKET_INDEX_SYMBOL}.")

        # --- SPY Feature Calculation ---
        print(f"{datetime.now()} - [DEBUG Main] Calculating features for {MARKET_INDEX_SYMBOL} (hourly data)...")
        feature_start_time = datetime.now()
        spy_features_df = calculate_features(spy_data.copy(), symbol_name=MARKET_INDEX_SYMBOL) # Pass copy to avoid modifying original
        feature_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Finished calculating features for {MARKET_INDEX_SYMBOL}. Duration: {feature_end_time - feature_start_time}")
        if spy_features_df.empty or len(spy_features_df) < MIN_ROWS_PER_SYMBOL: raise ValueError(f"Insufficient data ({len(spy_features_df)} rows) after features for {MARKET_INDEX_SYMBOL}.")

        # --- SPY Context Selection ---
        missing_spy_context_cols = [col for col in spy_context_features_to_keep if col not in spy_features_df.columns]
        if missing_spy_context_cols: raise ValueError(f"Required SPY context features missing after calculation: {missing_spy_context_cols}")
        spy_context_data = spy_features_df[spy_context_features_to_keep].copy()
        spy_context_data.rename(columns={col: f"SPY_{col}" for col in spy_context_features_to_keep}, inplace=True)
        print(f"{datetime.now()} - [DEBUG Main] Selected and renamed {len(spy_context_data.columns)} SPY context features. Shape: {spy_context_data.shape}")

    except Exception as e:
        print(f"\n{datetime.now()} - [DEBUG Main] --- CRITICAL ERROR processing market index symbol: {MARKET_INDEX_SYMBOL} ---")
        print(f"Error Type: {type(e).__name__}, Message: {e}"); traceback.print_exc()
        print(f"{datetime.now()} - [DEBUG Main] --- Cannot proceed without market context. Exiting. ---"); exit()


    # --- Process Individual Symbols to Predict ---
    print(f"\n{datetime.now()} - [DEBUG Main] Starting data processing loop for {len(SYMBOLS_TO_PREDICT)} symbols to predict...")
    for symbol in SYMBOLS_TO_PREDICT:
        print(f"\n{datetime.now()} - [DEBUG Main] --- Processing Symbol: {symbol} ---")
        file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(symbol))
        if not os.path.exists(file_path): print(f"{datetime.now()} - [DEBUG Main] Warning: Data file not found: {file_path}. Skipping."); continue
        data = None
        try:
            # --- Load target symbol data (using same logic as SPY) ---
            print(f"{datetime.now()} - [DEBUG Main] Loading hourly data for {symbol}...");
            load_start_time = datetime.now()
            try:
                correct_date_format = '%Y-%m-%d %H:%M:%S'
                try: data = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
                except ValueError:
                     print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Info: read_csv parsing failed with format '{correct_date_format}'. Attempting without format.")
                     data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
                # --- Fallback and index validation ---
                if not isinstance(data.index, pd.DatetimeIndex):
                    print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Warning: Index parsing failed initially. Attempting fallback...")
                    data = pd.read_csv(file_path)
                    if 'date' not in data.columns: raise ValueError("Column 'date' not found for fallback.")
                    data['date'] = pd.to_datetime(data['date'], errors='coerce')
                    if data['date'].isnull().all():
                        print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Automatic date parsing failed, trying specific format '{correct_date_format}'.")
                        data['date'] = pd.to_datetime(data['date'], format=correct_date_format, errors='coerce')
                    failed_count = data['date'].isnull().sum()
                    if failed_count > 0: print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Warning: {failed_count} dates failed parsing during fallback. Dropping.")
                    data.dropna(subset=['date'], inplace=True)
                    if data.empty: raise ValueError("No valid dates found after fallback parsing.")
                    data.set_index('date', inplace=True)
                data.sort_index(inplace=True)
                load_end_time = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Loaded {len(data)} rows ({symbol}). Index type: {type(data.index)}, Is monotonic: {data.index.is_monotonic_increasing}. Load duration: {load_end_time - load_start_time}")
                if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after loading for {symbol}."); continue
            except Exception as e: print(f"{datetime.now()} - [DEBUG Main] CRITICAL Error loading dates for {symbol}: {e}"); traceback.print_exc(); continue

            # --- Clean target symbol OHLCV ---
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

            # --- Calculate target symbol features ---
            print(f"{datetime.now()} - [DEBUG Main] Calculating features for {symbol} (hourly data)...")
            feature_start_time = datetime.now()
            data = calculate_features(data.copy(), symbol_name=symbol) # Pass copy
            feature_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Finished calculating features for {symbol}. Shape: {data.shape}. Duration: {feature_end_time - feature_start_time}")
            if data.empty or len(data) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Error or insufficient data ({len(data)} rows) after features for {symbol}. Skipping."); continue

            # --- Merge SPY Context Features ---
            if spy_context_data is not None:
                print(f"{datetime.now()} - [DEBUG Main] Merging SPY context features onto {symbol}...")
                merge_start = datetime.now()
                # Ensure indices are DatetimeIndex for merging
                if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index, errors='coerce')
                if not isinstance(spy_context_data.index, pd.DatetimeIndex): spy_context_data.index = pd.to_datetime(spy_context_data.index, errors='coerce')

                # --- CORRECTED LOGIC to remove rows with NaT index values ---
                data = data[~data.index.isna()]
                # No need to filter spy_context_data here again, but ensure it's valid before merge if concerned
                # spy_context_data = spy_context_data[~spy_context_data.index.isna()] # Already done after loading SPY if fallback happened
                if data.empty:
                     print(f"{datetime.now()} - [DEBUG Main] ({symbol}) Warning: Data empty after removing NaT index values before merge. Skipping symbol.")
                     continue
                # --- END CORRECTION ---

                # Perform the merge
                data = pd.merge(data, spy_context_data, left_index=True, right_index=True, how='left')
                merge_end = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Merge complete. Shape after merge: {data.shape}. Duration: {merge_end - merge_start}")

                # Check for NaNs introduced by the merge (e.g., non-matching timestamps in SPY data)
                spy_nan_mask = data[spy_context_feature_names].isnull().any(axis=1)
                if spy_nan_mask.any():
                    rows_to_drop = spy_nan_mask.sum()
                    print(f"{datetime.now()} - [DEBUG Main] Warning: {rows_to_drop} rows have missing SPY context after merge for {symbol} (likely timestamp mismatch). Dropping these rows.")
                    data = data[~spy_nan_mask].copy() # Keep rows where SPY context is NOT null
                    print(f"{datetime.now()} - [DEBUG Main] Shape after dropping rows with missing SPY context: {data.shape}")
                if data.empty or len(data) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Error or insufficient data ({len(data)}) after merging/cleaning SPY context for {symbol}. Skipping."); continue
            else:
                # This path should not be reached if SPY processing succeeded or exited
                print(f"{datetime.now()} - [DEBUG Main] CRITICAL Error: spy_context_data is None. Cannot proceed with merge for {symbol}."); continue

            # --- Add Symbol identifier BEFORE final processing ---
            data['symbol'] = symbol

            # --- Apply Labeling Method ---
            print(f"{datetime.now()} - [DEBUG Main] Applying labeling method: '{LABELING_METHOD}' for {symbol} (hourly data)...")
            label_start_time = datetime.now()
            if LABELING_METHOD == 'direction':
                # Ensure LABEL_LOOKAHEAD_N_DIRECTION is defined if using this
                if 'LABEL_LOOKAHEAD_N_DIRECTION' not in globals(): raise NameError("LABEL_LOOKAHEAD_N_DIRECTION not defined")
                data = create_direction_labels(data, n_bars=LABEL_LOOKAHEAD_N_DIRECTION)
            elif LABELING_METHOD == 'significant_direction':
                # Uses the modified threshold and configured lookahead
                data = create_significant_move_labels(data, n_bars=LABEL_LOOKAHEAD_N_SIG, threshold_pct=LABEL_SIGNIFICANCE_THRESHOLD_PCT)
            elif LABELING_METHOD == 'tp_sl':
                print(f"{datetime.now()} - [DEBUG Main] Generating TP/SL signals and labels for {symbol} (hourly data)...")
                 # Ensure LOOKAHEAD_BARS_TP_SL is defined if using this
                if 'LOOKAHEAD_BARS_TP_SL' not in globals(): raise NameError("LOOKAHEAD_BARS_TP_SL not defined")
                missing_base_features_for_tpsl = [f for f in ['close', 'lower_band', 'upper_band', 'volume_z', 'atr'] if f not in data.columns]
                if missing_base_features_for_tpsl: print(f"{datetime.now()} - [DEBUG Main] FATAL: Features needed for TP/SL ({missing_base_features_for_tpsl}) missing before signal generation for {symbol}. Skipping."); continue
                signal_indices, labels = generate_signals_and_labels_tp_sl(data, LOOKAHEAD_BARS_TP_SL, SL_MULT, TP_MULT, VOL_Z_LONG_THRESH, VOL_Z_SHORT_THRESH)
                label_end_time = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Finished generating TP/SL signals/labels for {symbol}. Duration: {label_end_time - label_start_time}")
                if not signal_indices: print(f"{datetime.now()} - [DEBUG Main] Warning: No signals generated for {symbol} (TP/SL). Skipping."); continue
                if len(signal_indices) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient signals ({len(signal_indices)}) for {symbol}. Skipping."); continue

                # Prepare data specifically for TP/SL signals
                features_to_select_tpsl = base_feature_columns + spy_context_feature_names + ['symbol']
                # Select data only at signal times
                data_at_signals = data.loc[signal_indices]
                # Check if all needed features exist *at signal times*
                missing_features_tpsl = [f for f in features_to_select_tpsl if f not in data_at_signals.columns]
                if missing_features_tpsl: print(f"{datetime.now()} - [DEBUG Main] FATAL: Features missing for TP/SL selection: {missing_features_tpsl}. Skipping."); continue
                try:
                    print(f"{datetime.now()} - [DEBUG Main] Selecting TP/SL signal features (including SPY context)...")
                    signal_features = data_at_signals[features_to_select_tpsl].copy()
                except KeyError as ke: print(f"{datetime.now()} - [DEBUG Main] Error selecting signal features: {ke}"); traceback.print_exc(); continue
                signal_features['label'] = labels; initial_count = len(signal_features)
                # Drop rows where any feature is NaN AT THE SIGNAL TIME
                signal_features.dropna(inplace=True);
                print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_count - len(signal_features)} TP/SL rows due to NaNs in selected features.")
                if len(signal_features) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient TP/SL data ({len(signal_features)}). Skipping."); continue
                current_symbol_data = signal_features
                all_symbol_data_list.append(current_symbol_data)
                print(f"{datetime.now()} - [DEBUG Main] Successfully processed {symbol} (TP/SL). Added {len(current_symbol_data)} rows.")
                # Continue to next symbol after handling TP/SL
                continue # Skip the common processing block below for TP/SL
            else:
                print(f"{datetime.now()} - [DEBUG Main] Error: Unknown LABELING_METHOD '{LABELING_METHOD}'. Skipping symbol."); continue

            label_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Finished applying labeling method '{LABELING_METHOD}' for {symbol}. Duration: {label_end_time - label_start_time}")

            # --- Final Data Selection & Cleaning for 'direction' or 'significant_direction' ---
            if LABELING_METHOD in ['direction', 'significant_direction']:
                if data.empty or 'label' not in data.columns or data['label'].isnull().all():
                    print(f"{datetime.now()} - [DEBUG Main] Error: No valid labels generated or remaining for {symbol} after method '{LABELING_METHOD}'. Skipping."); continue

                # Define columns to keep for the final combined dataset
                cols_to_keep = base_feature_columns + spy_context_feature_names + ['label', 'symbol']
                # Check if all required columns exist in the data before selection
                missing_cols_before_select = [f for f in cols_to_keep if f not in data.columns]
                if missing_cols_before_select:
                     print(f"{datetime.now()} - [DEBUG Main] FATAL: Columns missing just before final selection: {missing_cols_before_select}. Check feature calc/labeling. Skipping {symbol}."); continue

                # Select the final columns
                current_symbol_data = data[cols_to_keep].copy()

                # --- Final NaN check on the *selected* data ---
                if current_symbol_data.isnull().values.any():
                    initial_count = len(current_symbol_data)
                    nan_cols = current_symbol_data.columns[current_symbol_data.isnull().any()].tolist()
                    print(f"{datetime.now()} - [DEBUG Main] Warning: NaNs detected in final selection for {symbol}. Columns: {nan_cols}. Dropping rows with any NaN.")
                    current_symbol_data.dropna(inplace=True);
                    print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_count - len(current_symbol_data)} rows.")

                if len(current_symbol_data) < MIN_ROWS_PER_SYMBOL:
                     print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient data ({len(current_symbol_data)}) after finalizing {LABELING_METHOD} labels and NaN drop. Skipping {symbol}."); continue

                all_symbol_data_list.append(current_symbol_data)
                print(f"{datetime.now()} - [DEBUG Main] Successfully processed {symbol} ({LABELING_METHOD}). Added {len(current_symbol_data)} rows.")

        except Exception as e:
            # Catch unexpected errors during the processing of a single symbol
            print(f"\n{datetime.now()} - [DEBUG Main] --- Unexpected error processing symbol: {symbol} ---")
            print(f"Error Type: {type(e).__name__}, Message: {e}"); traceback.print_exc()
            print(f"{datetime.now()} - [DEBUG Main] --- Skipping rest of processing for {symbol} ---"); continue


    # --- Model Training and Evaluation ---
    print(f"\n{datetime.now()} - [DEBUG Main] --- Data Processing Complete ---")
    if not all_symbol_data_list:
        print(f"\n{datetime.now()} - [DEBUG Main] CRITICAL: No data collected from any symbol after processing loop. Cannot train model. Exiting."); exit()

    # --- Combine and Sort Data ---
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

    # --- One-Hot Encode Symbol ---
    print(f"{datetime.now()} - [DEBUG Main] One-hot encoding 'symbol' column...")
    ohe_start_time = datetime.now()
    try:
        # Ensure 'symbol' column exists before OHE
        if 'symbol' not in combined_data.columns:
            raise KeyError("'symbol' column not found in combined data. Check processing steps.")
        combined_data = pd.get_dummies(combined_data, columns=['symbol'], prefix='sym', drop_first=False)
        ohe_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Data shape after one-hot encoding: {combined_data.shape}. OHE duration: {ohe_end_time - ohe_start_time}")
        ohe_symbol_cols_post = [col for col in combined_data.columns if col.startswith('sym_')]
        if ohe_symbol_cols_post:
            combined_data[ohe_symbol_cols_post] = combined_data[ohe_symbol_cols_post].astype(np.uint8)
            print(f"{datetime.now()} - [DEBUG Main] Converted {len(ohe_symbol_cols_post)} OHE columns to uint8 type.")
        else: print(f"{datetime.now()} - [DEBUG Main] Warning: No OHE columns found after get_dummies. This is unexpected if symbols were processed.")
    except KeyError as e:
        print(f"{datetime.now()} - [DEBUG Main] Error during OHE: {e}. Skipping encoding.") # Should not happen if processing worked


    # --- Final Feature List and NaN Check ---
    final_feature_columns = sorted([col for col in combined_data.columns if col != 'label'])
    print(f"{datetime.now()} - [DEBUG Main] Total final features identified: {len(final_feature_columns)}")

    if combined_data.isnull().values.any():
        print(f"{datetime.now()} - [DEBUG Main] CRITICAL Warning: NaNs detected in combined data before split."); nan_counts = combined_data.isnull().sum(); print("NaN counts:\n", nan_counts[nan_counts > 0])
        print(f"{datetime.now()} - [DEBUG Main] Dropping rows with NaNs..."); combined_data.dropna(inplace=True); print(f"Rows remaining: {len(combined_data)}")
        if combined_data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Combined data empty after final NaN drop. Cannot train."); exit()

    # --- Separate Features (X) and Labels (y) ---
    print(f"{datetime.now()} - [DEBUG Main] Separating features (X) and labels (y)...")
    if 'label' not in combined_data.columns: print(f"{datetime.now()} - [DEBUG Main] FATAL Error: 'label' column missing in final combined data!"); exit()
    y = combined_data['label'].astype(int)
    X = combined_data[final_feature_columns].copy()

    # --- Sanity Checks ---
    missing_final_features_in_X = [f for f in final_feature_columns if f not in X.columns]
    if missing_final_features_in_X: print(f"{datetime.now()} - [DEBUG Main] FATAL Error: Missing final features IN X just before split: {missing_final_features_in_X}"); exit()
    print(f"{datetime.now()} - [DEBUG Main] Feature matrix shape: {X.shape}")
    if len(X) == 0 or len(y) == 0: print("Error: X or y empty before split."); exit()
    if len(X) != len(y): print(f"CRITICAL Error: Length mismatch between X ({len(X)}) and y ({len(y)}) before split."); exit()
    if len(y.unique()) < 2: print(f"Error: Only one class found in labels: {y.unique()}. Check labeling logic / data filtering."); exit()

    # --- Chronological Train/Test Split ---
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train_orig = X.iloc[:split_index]; X_test_orig = X.iloc[split_index:]
    y_train = y.iloc[:split_index]; y_test = y.iloc[split_index:]
    print(f"\n{datetime.now()} - [DEBUG Main] Splitting data chronologically:")
    print(f"Training set range: {X_train_orig.index.min()} to {X_train_orig.index.max()}, size: {len(X_train_orig)}")
    print(f"Testing set range : {X_test_orig.index.min()} to {X_test_orig.index.max()}, size: {len(X_test_orig)}")
    if len(X_train_orig) == 0 or len(X_test_orig) == 0: print("Error: Empty train/test set after split."); exit()

    # --- Feature Scaling ---
    numeric_cols_to_scale = [
        col for col in final_feature_columns if col not in
        ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'] and not col.startswith('sym_')
    ]
    print(f"{datetime.now()} - [DEBUG Main] Identified {len(numeric_cols_to_scale)} numeric columns for potential scaling.")

    scaler = None
    X_train = X_train_orig.copy()
    X_test = X_test_orig.copy()

    if USE_FEATURE_SCALING:
        print(f"\n{datetime.now()} - [DEBUG Main] Applying StandardScaler...");
        scale_start_time = datetime.now()
        cols_to_scale_present_train = [col for col in numeric_cols_to_scale if col in X_train.columns]
        cols_to_scale_present_test = [col for col in numeric_cols_to_scale if col in X_test.columns]

        if not cols_to_scale_present_train: print(f"{datetime.now()} - [DEBUG Main] Warning: No numeric columns identified for scaling in the training set.")
        else:
            print(f"{datetime.now()} - [DEBUG Main] Fitting scaler on {len(cols_to_scale_present_train)} training columns...")
            scaler = StandardScaler()
            # Ensure data is float before scaling - use .loc to avoid SettingWithCopyWarning
            X_train.loc[:, cols_to_scale_present_train] = scaler.fit_transform(X_train[cols_to_scale_present_train].astype(float))

            cols_to_transform_in_test = [col for col in cols_to_scale_present_train if col in cols_to_scale_present_test]
            if cols_to_transform_in_test:
                 print(f"{datetime.now()} - [DEBUG Main] Transforming {len(cols_to_transform_in_test)} columns in test set...")
                 X_test.loc[:, cols_to_transform_in_test] = scaler.transform(X_test[cols_to_transform_in_test].astype(float))
            else: print(f"{datetime.now()} - [DEBUG Main] Warning: No matching numeric columns found in test set for scaling transformation.")
            scale_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] StandardScaler applied. Duration: {scale_end_time - scale_start_time}")
    else: print(f"\n{datetime.now()} - [DEBUG Main] Skipping feature scaling (USE_FEATURE_SCALING is False).")


    # --- Manual Cross-Validation Loop ---
    print(f"\n{datetime.now()} - [DEBUG Main] Setting up manual GridSearchCV for XGBoost (Manual CV Loop)...")

    min_samples_per_split = 2 # Minimum required by TimeSeriesSplit
    train_samples = len(X_train)
    if train_samples // (N_CV_SPLITS + 1) < min_samples_per_split:
        actual_n_splits = max(2, train_samples // min_samples_per_split - 1)
        if actual_n_splits < 2:
             print(f"{datetime.now()} - [DEBUG Main] ERROR: Cannot perform TimeSeriesSplit with less than 2 splits. Train samples: {train_samples}. Not enough data."); exit()
        print(f"{datetime.now()} - [DEBUG Main] Warning: Training data size ({train_samples}) too small for {N_CV_SPLITS} splits. Reducing CV splits to {actual_n_splits}.")
    else:
        actual_n_splits = N_CV_SPLITS
    print(f"{datetime.now()} - [DEBUG Main] Using TimeSeriesSplit with n_splits={actual_n_splits} in manual CV.")


    # --- Define Base XGBoost Parameters ---
    xgb_base_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
    }

    # --- Handle Imbalance: scale_pos_weight OR SMOTE ---
    if not USE_SMOTE_IN_CV:
         scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
         xgb_base_params['scale_pos_weight'] = scale_pos_weight
         print(f"{datetime.now()} - [DEBUG Main] SMOTE disabled for CV. Using calculated scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        print(f"{datetime.now()} - [DEBUG Main] SMOTE enabled for CV. XGBoost 'scale_pos_weight' will NOT be used in CV folds.")


    # --- Check GPU Availability ---
    print(f"{datetime.now()} - [DEBUG Main] Checking XGBoost GPU support...")
    try:
        _ = xgb.XGBClassifier(device='cuda', tree_method='hist')
        print(f"{datetime.now()} - [DEBUG Main] XGBoost GPU support: Seems AVAILABLE ('device=cuda').")
    except Exception as e:
        print(f"{datetime.now()} - [DEBUG Main] XGBoost GPU support: NOT AVAILABLE. Error: {e}. Ensure XGBoost is compiled with CUDA support.")
        xgb_base_params['device'] = 'cpu'
        print(f"{datetime.now()} - [DEBUG Main] Falling back to CPU ('device=cpu').")


    best_balanced_accuracy = -1.0
    best_params_overall = None
    all_cv_results_manual = {}

    param_combinations = list(ParameterGrid(XGB_PARAM_GRID))
    total_combinations = len(param_combinations)
    print(f"{datetime.now()} - [DEBUG Main] Starting manual CV loop over {total_combinations} parameter combinations (USE_SMOTE_IN_CV={USE_SMOTE_IN_CV})...")

    manual_gridsearch_start_time = datetime.now()

    # --- Prepare X_train for CV loop (handle potential NaNs/Infs AFTER scaling) ---
    print(f"{datetime.now()} - [DEBUG Main] Preparing numeric X_train for CV loop...")
    X_train_numeric_for_cv = X_train.copy() # Start with the scaled (or unscaled) training data
    # Ensure all columns are numeric, coercing errors (should ideally be numeric already)
    for col in X_train_numeric_for_cv.columns:
         X_train_numeric_for_cv[col] = pd.to_numeric(X_train_numeric_for_cv[col], errors='coerce')

    # --- Impute NaNs ---
    cv_train_median = None
    if X_train_numeric_for_cv.isnull().values.any():
         print(f"{datetime.now()} - [DEBUG Main] Warning: NaNs found in X_train before CV loop (possibly from coercion). Imputing with median.")
         # Calculate median only on numeric columns that don't have all NaNs
         numeric_cols = X_train_numeric_for_cv.select_dtypes(include=np.number).columns
         cv_train_median = X_train_numeric_for_cv[numeric_cols].median()
         # Fill NaNs using the calculated median
         X_train_numeric_for_cv.fillna(cv_train_median, inplace=True)
         # Check if any NaNs remain (e.g., if median was NaN for a column of all NaNs)
         if X_train_numeric_for_cv.isnull().values.any():
             print(f"{datetime.now()} - [DEBUG Main] CRITICAL: NaNs still present after median imputation. Imputing remaining NaNs with 0.")
             X_train_numeric_for_cv.fillna(0, inplace=True)

    # --- Replace infinities ---
    if not np.all(np.isfinite(X_train_numeric_for_cv)):
        print(f"{datetime.now()} - [DEBUG Main] WARNING: Non-finite values (inf) found in X_train before CV loop. Replacing with NaN and imputing.")
        X_train_numeric_for_cv.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Re-impute using the previously calculated median if possible
        if cv_train_median is not None:
            X_train_numeric_for_cv.fillna(cv_train_median, inplace=True)
        # Final check for NaNs after inf replacement and imputation
        if X_train_numeric_for_cv.isnull().values.any():
             print(f"{datetime.now()} - [DEBUG Main] CRITICAL: Nans still present after inf->nan->median imputation. Imputing with 0.")
             X_train_numeric_for_cv.fillna(0, inplace=True)

    print(f"{datetime.now()} - [DEBUG Main] Prepared numeric X_train (shape: {X_train_numeric_for_cv.shape}) for CV loop. Final NaN check: {X_train_numeric_for_cv.isnull().sum().sum()}")


    # --- CV Loop Execution ---
    for i, params in enumerate(param_combinations):
        print(f"\n{datetime.now()} - [DEBUG Main] --- Evaluating parameters ({i+1}/{total_combinations}): {params} ---")
        try:
            cv_score_mean = custom_cv_folds(
                X=X_train_numeric_for_cv,
                y=y_train,
                base_model_params=xgb_base_params,
                current_iter_params=params,
                n_splits=actual_n_splits,
                use_smote=USE_SMOTE_IN_CV,
                random_state=RANDOM_STATE
            )

            param_key = tuple(sorted(params.items()))
            all_cv_results_manual[param_key] = cv_score_mean
            print(f"{datetime.now()} - [DEBUG Main] Mean CV Balanced Accuracy for params {params} = {cv_score_mean:.4f}")

            if cv_score_mean >= best_balanced_accuracy:
                if cv_score_mean > best_balanced_accuracy + 1e-6: print(f"{datetime.now()} - [DEBUG Main] *** NEW BEST PARAMS FOUND ***")
                elif best_params_overall is None or params.get('n_estimators', float('inf')) < best_params_overall.get('n_estimators', float('inf')):
                     print(f"{datetime.now()} - [DEBUG Main] *** Similar score, choosing simpler model ***")
                else: print(f"{datetime.now()} - [DEBUG Main] *** Similar score, keeping previous best params ***"); continue

                best_balanced_accuracy = cv_score_mean
                best_params_overall = params
                print(f"{datetime.now()} -   Best Params Now: {best_params_overall}")
                print(f"{datetime.now()} -   Best CV Balanced Accuracy: {best_balanced_accuracy:.4f}")

        except Exception as e_cv:
            print(f"{datetime.now()} - [DEBUG Main] ERROR during custom_cv_folds execution for params {params}. Error: {e_cv}")
            traceback.print_exc(); param_key = tuple(sorted(params.items())); all_cv_results_manual[param_key] = -1.0; continue

    manual_gridsearch_end_time = datetime.now()
    print(f"\n{datetime.now()} - [DEBUG Main] Manual GridSearchCV finished. Duration: {manual_gridsearch_end_time - manual_gridsearch_start_time}")

    if not best_params_overall:
        print(f"{datetime.now()} - [DEBUG Main] CRITICAL: Manual GridSearchCV did not find ANY valid best parameters. Check CV errors. Exiting."); exit()
    else:
        print("Best parameters found:", best_params_overall)
        print(f"Best CV score (balanced_accuracy): {best_balanced_accuracy:.4f}")


    # --- Train final best model ---
    print(f"\n{datetime.now()} - [DEBUG Main] Training final best model on full training data...")
    final_model_params = {**xgb_base_params, **best_params_overall}

    # --- Decide on final imbalance handling ---
    if 'scale_pos_weight' not in final_model_params and not USE_SMOTE_IN_CV:
         final_scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
         final_model_params['scale_pos_weight'] = final_scale_pos_weight
         print(f"{datetime.now()} - [DEBUG Main] Adding scale_pos_weight ({final_scale_pos_weight:.2f}) to final model (SMOTE not used in CV).")
    elif USE_SMOTE_IN_CV and 'scale_pos_weight' in final_model_params:
         print(f"{datetime.now()} - [DEBUG Main] Removing scale_pos_weight from final model (SMOTE used in CV).")
         del final_model_params['scale_pos_weight']

    # --- Use the cleaned/imputed numeric training data for final fit ---
    X_train_final_fit = X_train_numeric_for_cv
    y_train_final_fit = y_train
    # --- NOTE: SMOTE is NOT applied to the final fit by default. Add flag if needed ---
    # USE_SMOTE_ON_FINAL_TRAIN = False
    # if USE_SMOTE_ON_FINAL_TRAIN: ... apply SMOTE ...

    best_model = XGBClassifier(**final_model_params)
    print(f"{datetime.now()} - [DEBUG Main] Final model parameters for fit: {final_model_params}")
    print(f"{datetime.now()} - [DEBUG Main] Fitting final model on data shape: {X_train_final_fit.shape}")

    fit_start_time = datetime.now()
    try:
        # Ensure data is float32 for XGBoost efficiency
        best_model.fit(X_train_final_fit.astype(np.float32), y_train_final_fit)
        fit_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] Best model training finished. Duration: {fit_end_time - fit_start_time}")
    except Exception as final_fit_e: print(f"{datetime.now()} - [DEBUG Main] FATAL ERROR during final model training: {final_fit_e}"); traceback.print_exc(); best_model = None


    # --- Evaluate best model on test set ---
    if best_model:
        print(f"\n{datetime.now()} - [DEBUG Main] Evaluating best model on unseen test set...")

        # --- Prepare test set (scaling, NaN/inf handling) ---
        print(f"{datetime.now()} - [DEBUG Main] Preparing test set for evaluation...")
        X_test_numeric = X_test.copy()
        # Ensure numeric
        for col in X_test_numeric.columns:
            X_test_numeric[col] = pd.to_numeric(X_test_numeric[col], errors='coerce')

        # Use median calculated from the TRAINING set for imputation
        final_train_median = X_train_numeric_for_cv.median()

        if X_test_numeric.isnull().values.any():
            print(f"{datetime.now()} - [DEBUG Main] Warning: NaNs found in X_test before prediction. Imputing with TRAINING median.")
            X_test_numeric.fillna(final_train_median, inplace=True)
            if X_test_numeric.isnull().values.any():
                 print(f"{datetime.now()} - [DEBUG Main] CRITICAL: NaNs still present after test imputation. Imputing with 0.")
                 X_test_numeric.fillna(0, inplace=True)

        if not np.all(np.isfinite(X_test_numeric)):
            print(f"{datetime.now()} - [DEBUG Main] WARNING: Non-finite values (inf) found in X_test before prediction. Replacing with NaN and imputing with TRAINING median.")
            X_test_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_test_numeric.fillna(final_train_median, inplace=True)
            if X_test_numeric.isnull().values.any():
                 print(f"{datetime.now()} - [DEBUG Main] CRITICAL: NaNs still present after test inf->nan->median imputation. Imputing with 0.")
                 X_test_numeric.fillna(0, inplace=True)

        print(f"{datetime.now()} - [DEBUG Main] Test set prepared. Shape: {X_test_numeric.shape}. Final NaN check: {X_test_numeric.isnull().sum().sum()}")

        # --- Make Predictions & Evaluate ---
        print(f"{datetime.now()} - [DEBUG Main] Predicting on test data...")
        try:
            eval_start_time = datetime.now()
            # Ensure float32 for prediction
            y_pred_proba_test = best_model.predict_proba(X_test_numeric.astype(np.float32))[:, 1]
            y_pred_test = (y_pred_proba_test > 0.5).astype(int) # Standard 0.5 threshold
            eval_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Prediction finished. Duration: {eval_end_time - eval_start_time}")

            print("\n--- Test Set Evaluation Results ---")
            print(f"Test Set Accuracy (0.5 Threshold): {accuracy_score(y_test, y_pred_test):.4f}")
            print(f"Test Set Balanced Accuracy (0.5 Threshold): {balanced_accuracy_score(y_test, y_pred_test):.4f}")
            print("Test Set Classification Report (0.5 Threshold):\n",
                  classification_report(y_test, y_pred_test, target_names=['Significant Down (0)', 'Significant Up (1)'], zero_division=0))
            print("------------------------------------")

            # --- Feature Importances ---
            print("\nFeature Importances (Top 30):")
            try:
                feature_names = X_train_final_fit.columns # Use columns from final fit data
                importances = best_model.feature_importances_
                if len(feature_names) != len(importances):
                     print(f"{datetime.now()} - [DEBUG Main] Warning: Mismatch feature names ({len(feature_names)}) vs importances ({len(importances)}).")
                     feature_names = [f'feature_{i}' for i in range(len(importances))] # Generic names

                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                print(feature_importance_df.head(30).to_string(index=False))
            except Exception as fi_e: print(f"Could not get/display feature importances: {fi_e}")

            # --- Save Artifacts ---
            print(f"\n{datetime.now()} - [DEBUG Main] Saving artifacts...")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S") # Added seconds
            smote_tag = "_smoteCV" if USE_SMOTE_IN_CV else "_scalePosWeightCV"
            base_filename = f'trading_xgb_model_1h_sig_{timestamp_str}{smote_tag}'

            model_filename = f'{base_filename}.joblib'
            feature_list_filename = f'{base_filename}_features.list'
            scaler_filename = f'{base_filename}_scaler.joblib'

            joblib.dump(best_model, model_filename); print(f"Model saved to '{model_filename}'")
            try:
                 current_features = list(X_train_final_fit.columns) # Save features from final fit
                 with open(feature_list_filename, 'w') as f:
                      for feature in current_features: f.write(f"{feature}\n")
                 print(f"Feature list ({len(current_features)}) saved to '{feature_list_filename}'")
            except Exception as fl_e: print(f"Error saving feature list: {fl_e}")

            if scaler and USE_FEATURE_SCALING:
                joblib.dump(scaler, scaler_filename); print(f"Scaler saved to '{scaler_filename}'")
            else: print("Scaler not saved (not used or scaling disabled).")

        except Exception as eval_e: print(f"{datetime.now()} - [DEBUG Main] ERROR during evaluation or saving: {eval_e}"); traceback.print_exc()
    else: print(f"\n{datetime.now()} - [DEBUG Main] Skipping evaluation and saving as final model training failed.")

    print(f"\n{datetime.now()} - --- Training Script Finished ---")