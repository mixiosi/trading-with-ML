# train_model_revised.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV # Added GridSearchCV back
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import pandas.api.types
import traceback
from datetime import datetime
import shap
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

# --- Configuration ---
SYMBOLS = ['AAPL', 'AMD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'QQQ', 'SPY', 'TSLA']
INDEX_SYMBOL = 'SPY'
DATA_DIR = '.'
FILENAME_TEMPLATE = '{}_5min_historical_data.csv'

# --- Feature Calculation Parameters (5-min bars) ---
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
LAG_PERIODS = [1, 3, 5]
VOLUME_SMA_PERIOD = 20

# --- STRATEGY & LABELING PARAMETERS ---
LOOKAHEAD_BARS_TP_SL = 15
SL_MULT = 1.5
TP_MULT = 2.0 # Adjusted TP target

# Signal Thresholds (EXAMPLES - TUNE THESE)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_MID = 50
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80
VOLUME_Z_THRESH = 1.0

# --- Data Processing Parameters ---
MIN_ROWS_PER_SYMBOL = 100
MIN_SIGNAL_ROWS_PER_SYMBOL = 50
USE_FEATURE_SCALING = True

# --- Model Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_SPLITS = 5
N_RANDOM_SEARCH_ITER = 50

# XGBoost RandomizedSearch Parameter Distribution
XGB_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.25, 0.5, 1.0],
}

# Logistic Regression Parameters
LOGREG_PARAMS = { 'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear'] }

# --- Define Feature Columns ---
base_stock_feature_columns = [
    'sma', 'std', 'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'volume_z',
    'rsi_lag1', 'bb_width', 'stoch_k', 'stoch_d', 'roc',
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    'return_1_lag1', 'return_1_lag3', 'return_1_lag5',
    'macd_hist_lag1', 'macd_hist_lag3', 'macd_hist_lag5',
    'stoch_k_lag1', 'stoch_k_lag3', 'stoch_k_lag5',
    'atr_lag1', 'atr_lag3', 'atr_lag5',
    'bb_width_lag1', 'bb_width_lag3', 'bb_width_lag5',
    'return_1', 'sma_diff',
    'minutes_since_open', 'minutes_to_close' # Added market time
]
INDEX_CONTEXT_BASE_COLS = ['return_1', 'rsi', 'atr', 'sma_diff', 'volume_z']
RELATIVE_FEATURE_COLS = ['rel_return_1', 'rel_rsi', 'rel_volume_z']
base_feature_columns = base_stock_feature_columns + RELATIVE_FEATURE_COLS
SIGNAL_TYPE_COL = 'signal_type'


# --- Feature Calculation Function (MODIFIED: Added market time features, NO final dropna) ---
def calculate_features(df, index_features=None, market_open_hour=9, market_open_minute=30, market_close_hour=16, market_close_minute=0):
    """
    Calculate technical indicators, time features, lags, relative features,
    and market time features. Returns DataFrame potentially containing NaNs at the start.
    """
    required_input_cols = ['high', 'low', 'close', 'open', 'volume']
    if not all(col in df.columns for col in required_input_cols):
        missing_cols = [col for col in required_input_cols if col not in df.columns]
        print(f"[ERROR calculate_features] Missing required input columns: {missing_cols}. Returning empty DataFrame.")
        return pd.DataFrame()

    cols_to_check = ['high', 'low', 'close', 'open', 'volume']
    for col in cols_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=required_input_cols, inplace=True)
    if df.empty: return df

    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    if not is_datetime_index:
        print(f"{datetime.now()} - [WARN calculate_features] Index not DatetimeIndex. Attempting conversion.")
        try:
            df.index = pd.to_datetime(df.index); df.sort_index(inplace=True)
            if not isinstance(df.index, pd.DatetimeIndex): raise ValueError("Failed conversion.")
            is_datetime_index = True
        except Exception as e:
            print(f"{datetime.now()} - [ERROR calculate_features] Index conversion failed: {e}. Features will be NaN.")
            for tf in ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'minutes_since_open', 'minutes_to_close']: df[tf] = np.nan
            df['sma_diff'] = np.nan
            for rf in ['rel_return_1', 'rel_rsi', 'rel_volume_z']: df[rf] = np.nan
    if is_datetime_index and not df.index.is_monotonic_increasing:
        print(f"{datetime.now()} - [WARN calculate_features] Index not sorted. Sorting.")
        df.sort_index(inplace=True)

    # --- Calculate Basic Features ---
    df['sma'] = df['close'].rolling(window=SMA_WINDOW, min_periods=SMA_WINDOW // 2).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW, min_periods=SMA_WINDOW // 2).std()
    df['upper_band'] = df['sma'] + 2 * df['std']
    df['lower_band'] = df['sma'] - 2 * df['std']
    df['sma_diff'] = df['close'] - df['sma']

    # --- Calculate TA-Lib Indicators ---
    high_prices=df['high'].values; low_prices=df['low'].values; close_prices=df['close'].values
    if len(close_prices) >= ATR_PERIOD: df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
    else: df['atr'] = np.nan
    if len(close_prices) >= RSI_PERIOD: df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
    else: df['rsi'] = np.nan
    if len(close_prices) >= MACD_SLOW:
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
        df['macd']=macd; df['macd_signal']=macdsignal; df['macd_hist']=macdhist
    else: df['macd']=np.nan; df['macd_signal']=np.nan; df['macd_hist']=np.nan
    sma_safe = df['sma'].replace(0, 1e-10)
    df['bb_width'] = (df['upper_band'] - df['lower_band']) / sma_safe
    if len(close_prices) >= STOCH_K + STOCH_D - 1 :
         df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=STOCH_K, slowk_period=STOCH_SMOOTH, slowk_matype=0, slowd_period=STOCH_D, slowd_matype=0)
    else: df['stoch_k'], df['stoch_d'] = np.nan, np.nan
    if len(close_prices) >= ROC_PERIOD: df['roc'] = talib.ROC(close_prices, timeperiod=ROC_PERIOD)
    else: df['roc'] = np.nan

    # --- Calculate Volume Z-Score ---
    df['volume_sma'] = df['volume'].rolling(window=VOLUME_SMA_PERIOD, min_periods=VOLUME_SMA_PERIOD // 2).mean()
    df['volume_std'] = df['volume'].rolling(window=VOLUME_SMA_PERIOD, min_periods=VOLUME_SMA_PERIOD // 2).std()
    volume_std_safe = df['volume_std'].replace(0, 1e-10)
    df['volume_z'] = (df['volume'] - df['volume_sma']) / volume_std_safe
    df.drop(columns=['volume_sma', 'volume_std'], inplace=True, errors='ignore')

    # --- Calculate RSI Lag ---
    df['rsi_lag1'] = df['rsi'].shift(1)

    # --- Calculate Cyclical and Market Time Features ---
    if is_datetime_index:
        current_hour = df.index.hour
        current_minute = df.index.minute
        current_dayofweek = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * current_hour/24.0); df['hour_cos'] = np.cos(2 * np.pi * current_hour/24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * current_dayofweek/7.0); df['dayofweek_cos'] = np.cos(2 * np.pi * current_dayofweek/7.0)
        market_open_minutes = market_open_hour * 60 + market_open_minute
        market_close_minutes = market_close_hour * 60 + market_close_minute
        current_total_minutes = current_hour * 60 + current_minute
        df['minutes_since_open'] = current_total_minutes - market_open_minutes
        df['minutes_to_close'] = market_close_minutes - current_total_minutes
    elif 'hour_sin' not in df.columns:
        for tf in ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'minutes_since_open', 'minutes_to_close']: df[tf] = np.nan

    # --- Calculate Return Feature ---
    df['return_1'] = df['close'].pct_change(1)

    # --- Calculate Relative Strength / Context Features ---
    df['rel_return_1'] = np.nan; df['rel_rsi'] = np.nan; df['rel_volume_z'] = np.nan
    if index_features is not None and isinstance(index_features, pd.DataFrame) and not index_features.empty:
        required_idx_cols = INDEX_CONTEXT_BASE_COLS
        if all(col in index_features.columns for col in required_idx_cols):
            aligned_idx_features = index_features[required_idx_cols].reindex(df.index, method='ffill', limit=5)
            df['rel_return_1'] = df['return_1'].sub(aligned_idx_features['return_1'], fill_value=0)
            rsi_fill_val = df['rsi'].mean() if df['rsi'].notna().any() else 0
            df['rel_rsi'] = df['rsi'].sub(aligned_idx_features['rsi'], fill_value=rsi_fill_val)
            df['rel_volume_z'] = df['volume_z'].sub(aligned_idx_features['volume_z'], fill_value=0)
        else: print(f"[WARN calculate_features] Missing required base cols {required_idx_cols} in index_features for relative calc.")

    # --- Calculate Lagged Features ---
    features_to_lag = ['return_1', 'macd_hist', 'stoch_k', 'atr', 'bb_width', 'rel_return_1']
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in LAG_PERIODS:
                col_name = f'{feature}_lag{lag}'
                df[col_name] = df[feature].shift(lag)

    # ---> REMOVED FINAL DROPNA <---
    return df

# --- Signal Generation Function (MODIFIED: Combined Filters) ---
def generate_entry_signals(df):
    """
    Identifies potential entry signals based ONLY on Bollinger Band Mean Reversion,
    applying time, relative strength, and volume filters.
    """
    signals = pd.Series(0, index=df.index, dtype=int) # Default 0 = no signal

    # Ensure necessary columns exist for BB, Time, Relative RSI, Volume Z
    req_cols = ['close', 'lower_band', 'upper_band', 'volume_z',
                'minutes_since_open', 'rel_rsi'] # Added rel_rsi
    if not all(col in df.columns for col in req_cols):
        print(f"[WARN generate_entry_signals] Missing required columns for filtered BB signals: {[c for c in req_cols if c not in df.columns]}. Returning no signals.")
        return signals

    # --- Define Filters based on SHAP Analysis ---
    # 1. Time Filter: Avoid first hour
    time_filter = (df['minutes_since_open'] > 50)

    # 2. Relative Strength Filter (Example: Require positive relative RSI for longs, negative for shorts)
    #    Adjust threshold based on dependence plot if needed (e.g., rel_rsi > 5)
    rel_rsi_long_filter = (df['rel_rsi'].fillna(0) > 0)  # Stock stronger than SPY
    rel_rsi_short_filter = (df['rel_rsi'].fillna(0) < 0) # Stock weaker than SPY

    # 3. Volume Filter (Example: Avoid very low volume signals, require positive Z for longs?)
    #    Adjust thresholds based on dependence plot. Let's require non-negative Z for this example.
    volume_filter_long = (df['volume_z'].fillna(0) >= 0) # Avoid very low volume fades for longs
    volume_filter_short = (df['volume_z'].fillna(0) >= 0) # Maybe avoid low vol breakouts for shorts too? Or remove this filter for shorts? Test variations.
    # Alternative: volume_filter = (df['volume_z'].fillna(0) > -0.5) # Avoid extremely low volume

    # Combine all filters
    long_filters_combined = time_filter & rel_rsi_long_filter & volume_filter_long
    short_filters_combined = time_filter & rel_rsi_short_filter & volume_filter_short


    # --- Calculate Bollinger Band Reversion Signals ---
    bb_rev_long = (df['close'] < df['lower_band'])
    bb_rev_short = (df['close'] > df['upper_band'])

    # --- Assign ONLY signal types 7 and 8 IF they pass ALL combined filters ---
    signals.loc[bb_rev_long & long_filters_combined] = 7
    signals.loc[bb_rev_short & short_filters_combined] = 8

    # Optional: Print counts for debugging
    print(f"{datetime.now()} - [INFO generate_entry_signals] BB Signal counts (after ALL filters):\n{signals[signals > 0].value_counts()}")

    return signals


# --- Labeling Function based on Signal Outcome (Unchanged) ---
def label_signal_outcome_tp_sl(df, signal_indices, lookahead_bars, sl_mult, tp_mult):
    """Labels signals based on whether TP is hit before SL."""
    # ... (Function code remains the same as previous version) ...
    labels = pd.Series(np.nan, index=signal_indices)
    if signal_indices.empty or not isinstance(df.index, pd.DatetimeIndex): return labels.dropna().astype(int)
    signal_indices = signal_indices.intersection(df.index)
    if signal_indices.empty: return labels.dropna().astype(int)
    if 'atr' not in df.columns or 'signal_type' not in df.columns:
         print("[ERROR label_signal_outcome_tp_sl] Missing 'atr' or 'signal_type'. Cannot label.")
         return labels.dropna().astype(int)
    valid_signals = df.loc[signal_indices, ['atr', 'close', 'high', 'low', 'signal_type']].copy()
    valid_signals.dropna(subset=['atr', 'close', 'high', 'low', 'signal_type'], inplace=True)
    valid_signals = valid_signals[valid_signals['atr'] > 1e-9]
    if valid_signals.empty: print("[WARN label_signal_outcome_tp_sl] No valid signals found after filtering."); return labels.dropna().astype(int)
    print(f"{datetime.now()} - [INFO label_signal_outcome_tp_sl] Labeling {len(valid_signals)} signals...")
    df_len = len(df); df_index = df.index; df_lows = df['low'].values; df_highs = df['high'].values
    for idx in valid_signals.index:
        signal_data = valid_signals.loc[idx]; entry_price = signal_data['close']
        atr = signal_data['atr']; signal_type = int(signal_data['signal_type'])
        is_long = signal_type in [1, 3, 5, 7]
        if is_long: stop_loss = entry_price - sl_mult * atr; take_profit = entry_price + tp_mult * atr
        else: stop_loss = entry_price + sl_mult * atr; take_profit = entry_price - tp_mult * atr
        try: entry_loc = df_index.get_loc(idx)
        except KeyError: continue
        start_lookahead = entry_loc + 1; end_lookahead = min(entry_loc + 1 + lookahead_bars, df_len)
        if start_lookahead >= end_lookahead: continue
        future_lows_slice = df_lows[start_lookahead:end_lookahead]; future_highs_slice = df_highs[start_lookahead:end_lookahead]
        hit_tp = False; hit_sl = False; tp_idx = -1; sl_idx = -1
        try:
            if is_long:
                tp_indices = np.where(future_highs_slice >= take_profit)[0]; sl_indices = np.where(future_lows_slice <= stop_loss)[0]
            else:
                tp_indices = np.where(future_lows_slice <= take_profit)[0]; sl_indices = np.where(future_highs_slice >= stop_loss)[0]
            if tp_indices.size > 0: hit_tp = True; tp_idx = tp_indices[0]
            if sl_indices.size > 0: hit_sl = True; sl_idx = sl_indices[0]
            label = 0
            if hit_tp and hit_sl:
                if tp_idx <= sl_idx: label = 1
            elif hit_tp: label = 1
            labels.loc[idx] = label
        except Exception as e: print(f"[ERROR label_signal_outcome_tp_sl] Error in TP/SL check for {idx}: {e}"); labels.loc[idx] = np.nan
    labels.dropna(inplace=True); labels = labels.astype(int)
    print(f"{datetime.now()} - [INFO label_signal_outcome_tp_sl] Finished labeling. {len(labels)} signals labeled.")
    return labels


# --- Helper function to process index data (Unchanged from previous corrected version) ---
def process_index_data(index_symbol, data_dir, filename_template):
    """Loads 5-min index data, calculates ALL features on it, returns full features DF."""
    # ... (Function code remains the same as previous version) ...
    print(f"\n{datetime.now()} - [INFO process_index_data] --- Processing Index Symbol: {index_symbol} ---")
    file_path = os.path.join(data_dir, filename_template.format(index_symbol))
    if not os.path.exists(file_path): print(f"[ERROR process_index_data] Index data file not found: {file_path}"); return None
    index_data = None
    try:
        print(f"{datetime.now()} - [INFO process_index_data] Loading 5-minute data for {index_symbol}...");
        try:
            index_data = pd.read_csv(file_path, parse_dates=['date'], date_format='%Y-%m-%d %H:%M:%S', index_col='date')
            if not isinstance(index_data.index, pd.DatetimeIndex): raise ValueError("Not DatetimeIndex")
            index_data.sort_index(inplace=True)
        except Exception as e:
            print(f"[WARN process_index_data] Initial load failed: {e}. Fallback.")
            index_data = pd.read_csv(file_path)
            if 'date' not in index_data.columns: raise ValueError("'date' missing")
            index_data['date'] = pd.to_datetime(index_data['date'], errors='coerce'); index_data.dropna(subset=['date'], inplace=True)
            if index_data.empty: raise ValueError("No valid dates.")
            index_data.set_index('date', inplace=True); index_data.sort_index(inplace=True)
        print(f"{datetime.now()} - [INFO process_index_data] Loaded {len(index_data)} rows (5-min).")
        if index_data.empty: raise ValueError("Index data empty.")
        print(f"{datetime.now()} - [INFO process_index_data] Cleaning index OHLCV...");
        initial_rows = len(index_data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in index_data.columns: index_data[col] = pd.to_numeric(index_data[col], errors='coerce')
        index_data.dropna(subset=[c for c in ohlcv_cols if c in index_data.columns], inplace=True)
        if index_data.empty: raise ValueError("Index data empty after cleaning.")
        print(f"{datetime.now()} - [INFO process_index_data] Calculating index features (5-min data)...")
        index_features_df = calculate_features(index_data.copy(), index_features=None)
        if index_features_df is None or index_features_df.empty:
             print(f"[ERROR process_index_data] calculate_features returned empty or None for index.")
             return None
        print(f"{datetime.now()} - [INFO process_index_data] Finished processing index. Full features shape: {index_features_df.shape}")
        return index_features_df
    except Exception as e:
        print(f"\n[ERROR process_index_data] Error processing index {index_symbol}: {e}"); traceback.print_exc(); return None


# --- Main Execution Block ---
if __name__ == "__main__":
    all_symbol_signal_data = []
    print(f"{datetime.now()} - [INFO Main] Script starting. Revamped Logic: Multi-Signal TP/SL Labeling - V2.")

    # Pre-process Index Data
    index_features_df = process_index_data(INDEX_SYMBOL, DATA_DIR, FILENAME_TEMPLATE)
    if index_features_df is None or index_features_df.empty:
        print(f"{datetime.now()} - [ERROR Main] Failed to process index data or result is empty. Exiting."); exit()

    print(f"{datetime.now()} - [INFO Main] Starting data processing loop for {len(SYMBOLS)} symbols...")
    for symbol in SYMBOLS:
        if symbol == INDEX_SYMBOL:
            print(f"\n{datetime.now()} - [INFO Main] --- Skipping Index Symbol {symbol} ---")
            continue

        print(f"\n{datetime.now()} - [INFO Main] --- Processing Symbol: {symbol} ---")
        file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(symbol))
        if not os.path.exists(file_path): print(f"[WARN Main] Data file not found: {file_path}. Skipping."); continue
        data = None
        try:
            # Load 5-min data
            print(f"{datetime.now()} - [INFO Main] Loading 5-minute data for {symbol}...");
            try: # Simplified Loading
                data = pd.read_csv(file_path, parse_dates=['date'], date_format='%Y-%m-%d %H:%M:%S', index_col='date')
                if not isinstance(data.index, pd.DatetimeIndex): raise ValueError("Not DatetimeIndex")
                data.sort_index(inplace=True)
            except Exception as e:
                 print(f"[WARN Main] Initial load failed: {e}. Fallback.")
                 data = pd.read_csv(file_path)
                 if 'date' not in data.columns: raise ValueError("'date' missing")
                 data['date'] = pd.to_datetime(data['date'], errors='coerce'); data.dropna(subset=['date'], inplace=True)
                 if data.empty: raise ValueError("No valid dates.")
                 data.set_index('date', inplace=True); data.sort_index(inplace=True)
            print(f"{datetime.now()} - [INFO Main] Loaded {len(data)} rows (5-min).")
            if data.empty: print(f"[ERROR Main] Data empty after loading."); continue

            # Clean OHLCV
            print(f"{datetime.now()} - [INFO Main] Cleaning OHLCV columns...");
            initial_rows = len(data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                 if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(subset=[c for c in ohlcv_cols if c in data.columns], inplace=True)
            if data.empty: print(f"[ERROR Main] Data empty after cleaning."); continue
            if len(data) < MIN_ROWS_PER_SYMBOL:
                print(f"[WARN Main] Insufficient data ({len(data)}) after cleaning for {symbol}. Skipping."); continue

            # Calculate Stock + Relative + Time Features
            print(f"{datetime.now()} - [INFO Main] Calculating features for {symbol} (incl. relative & market time)...")
            data = calculate_features(data.copy(), index_features=index_features_df)
            print(f"{datetime.now()} - [INFO Main] Features calculated. Shape before final dropna: {data.shape}")
            if data.empty: print(f"[ERROR Main] Data empty immediately after feature calculation."); continue

            # ---> Drop rows with ANY NaNs AFTER all features are calculated <---
            initial_rows_dropna = len(data)
            data.dropna(inplace=True)
            rows_dropped_final = initial_rows_dropna - len(data)
            print(f"{datetime.now()} - [INFO Main] Dropped {rows_dropped_final} rows with NaNs. Final usable shape for signals/labeling: {data.shape}")
            if data.empty: print(f"[ERROR Main] Data empty after final feature dropna for {symbol}. Skipping."); continue
            if len(data) < MIN_ROWS_PER_SYMBOL:
                print(f"[WARN Main] Insufficient data ({len(data)}) after feature calculation/dropna for {symbol}. Skipping."); continue

            # Generate Entry Signals
            print(f"{datetime.now()} - [INFO Main] Generating entry signals for {symbol}...")
            data[SIGNAL_TYPE_COL] = generate_entry_signals(data)
            signals_found_indices = data.index[data[SIGNAL_TYPE_COL] > 0]
            print(f"{datetime.now()} - [INFO Main] Found {len(signals_found_indices)} potential signal events for {symbol}.")
            if signals_found_indices.empty: print(f"[INFO Main] No signals found for {symbol}. Skipping."); continue

            # Label Signal Outcomes
            print(f"{datetime.now()} - [INFO Main] Labeling signal outcomes (TP/SL) for {symbol}...")
            labels = label_signal_outcome_tp_sl(data, signals_found_indices, LOOKAHEAD_BARS_TP_SL, SL_MULT, TP_MULT)
            if labels.empty: print(f"[INFO Main] No signals could be labeled for {symbol}. Skipping."); continue

            # Prepare Data for Model
            model_data = data.loc[labels.index].copy()
            model_data['label'] = labels
            model_data['symbol'] = symbol

            # Define columns to keep
            cols_to_keep = base_feature_columns + [SIGNAL_TYPE_COL, 'label', 'symbol']
            missing_cols = [c for c in cols_to_keep if c not in model_data.columns]
            if missing_cols: print(f"[ERROR Main] Critical features missing before final selection: {missing_cols}. Skipping."); continue
            final_symbol_data = model_data[cols_to_keep].copy()

            # Final NaN check
            if final_symbol_data.isnull().values.any():
                print(f"[WARN Main] NaNs detected in final data. Dropping.");
                initial_count = len(final_symbol_data); final_symbol_data.dropna(inplace=True);
                print(f"Dropped {initial_count - len(final_symbol_data)} rows.")

            if len(final_symbol_data) < MIN_SIGNAL_ROWS_PER_SYMBOL:
                print(f"[WARN Main] Insufficient labeled signal data ({len(final_symbol_data)} < {MIN_SIGNAL_ROWS_PER_SYMBOL}). Skipping."); continue

            all_symbol_signal_data.append(final_symbol_data)
            print(f"{datetime.now()} - [INFO Main] Successfully processed {symbol}. Added {len(final_symbol_data)} labeled signal events.")

        except Exception as e:
            print(f"\n{datetime.now()} - [ERROR Main] --- Unexpected error processing symbol: {symbol} ---")
            print(f"Error Type: {type(e).__name__}, Message: {e}"); traceback.print_exc()
            print(f"{datetime.now()} - [INFO Main] --- Skipping rest of processing for {symbol} ---"); continue

    # --- Model Training and Evaluation ---
    print(f"\n{datetime.now()} - [INFO Main] --- Data Processing Complete ---")
    if not all_symbol_signal_data: print(f"[ERROR Main] No labeled signal data collected. Cannot train."); exit()

    print(f"{datetime.now()} - [INFO Main] Combining data from {len(all_symbol_signal_data)} symbols...")
    combined_data = pd.concat(all_symbol_signal_data)
    print(f"{datetime.now()} - [INFO Main] Total labeled signal events: {len(combined_data)}")

    print(f"{datetime.now()} - [INFO Main] Sorting combined data by timestamp...")
    combined_data.sort_index(inplace=True)
    print(f"{datetime.now()} - [INFO Main] Combined data sorted.")

    # One-Hot Encode Categorical Features
    print(f"{datetime.now()} - [INFO Main] One-hot encoding categorical features...")
    categorical_cols = ['symbol', SIGNAL_TYPE_COL]
    try:
        combined_data[SIGNAL_TYPE_COL] = combined_data[SIGNAL_TYPE_COL].astype(int).astype(str)
        combined_data = pd.get_dummies(combined_data, columns=categorical_cols, prefix=['sym', 'sig'], drop_first=False)
        print(f"{datetime.now()} - [INFO Main] OHE complete. Shape: {combined_data.shape}")
    except KeyError as e: print(f"[ERROR Main] Failed OHE - column missing? {e}"); exit()
    except Exception as e_ohe: print(f"[ERROR Main] Failed OHE: {e_ohe}"); exit()

    # Define final feature columns INCLUDING OHE versions
    final_feature_columns = base_feature_columns.copy() # Includes stock, relative, new time feats
    final_feature_columns = [col for col in final_feature_columns if col not in categorical_cols] # Remove originals
    ohe_cols = [col for col in combined_data.columns if col.startswith('sym_') or col.startswith('sig_')]
    final_feature_columns.extend(ohe_cols)
    final_feature_columns = sorted(list(set(final_feature_columns)))
    print(f"{datetime.now()} - [INFO Main] Final feature columns count: {len(final_feature_columns)}")

    # Final check for NaNs before splitting
    if combined_data.isnull().values.any():
        print(f"[WARN Main] NaNs detected before split."); nan_counts = combined_data.isnull().sum(); print("NaN counts:\n", nan_counts[nan_counts > 0])
        print("Dropping NaNs..."); combined_data.dropna(inplace=True); print(f"Rows remaining: {len(combined_data)}")
        if combined_data.empty: print("[ERROR Main] Combined data empty after final NaN drop."); exit()

    print(f"{datetime.now()} - [INFO Main] Separating features (X) and labels (y)...")
    missing_final_features = [f for f in final_feature_columns if f not in combined_data.columns]
    if missing_final_features: print(f"[ERROR Main] Missing final features: {missing_final_features}"); exit()
    if 'label' not in combined_data.columns: print("[ERROR Main] 'label' column missing!"); exit()
    X = combined_data[final_feature_columns]; y = combined_data['label']
    print(f"{datetime.now()} - [INFO Main] Feature matrix shape: {X.shape}")
    if len(X) == 0 or len(y) == 0: print("[ERROR Main] X or y empty."); exit()
    if len(y.unique()) < 2: print(f"[ERROR Main] Only one class found: {y.unique()}"); exit()
    print(f"Label distribution in combined data: {y.value_counts(normalize=True).to_dict()}")

    # Train/Test Split
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_index]; X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]; y_test = y.iloc[split_index:]
    print(f"\n{datetime.now()} - [INFO Main] Splitting data chronologically:")
    print(f"Training set size: {len(X_train)} ({X_train.index.min()} to {X_train.index.max()})")
    print(f"Testing set size : {len(X_test)} ({X_test.index.min()} to {X_test.index.max()})")
    if len(X_train) == 0 or len(X_test) == 0: print("[ERROR Main] Empty train/test set."); exit()

    # --->>> NEW: SAVE TEST DATA <<<---
    print(f"{datetime.now()} - [INFO Main] Saving unscaled test data...")
    try:
        # Save X_test BEFORE scaling, as SHAP needs unscaled data for interpretation
        X_test.to_pickle("X_test_unscaled.pkl")
        y_test.to_pickle("y_test.pkl")
        print("Test data saved successfully (X_test_unscaled.pkl, y_test.pkl).")
    except Exception as e_save:
        print(f"[ERROR Main] Failed to save test data: {e_save}")
    # --->>> END SAVE TEST DATA <<<---


    # Feature Scaling
    scaler = None
    if USE_FEATURE_SCALING:
        print(f"\n{datetime.now()} - [INFO Main] Applying StandardScaler...");
        # Identify numeric cols EXCLUDING OHE cols
        numeric_cols_to_scale_final = [
            col for col in X_train.columns if
            col in base_feature_columns and # Check against base list (incl relative, market time)
            not col.startswith('sym_') and not col.startswith('sig_')
        ]
        cols_to_scale_present_train = [col for col in numeric_cols_to_scale_final if col in X_train.columns]
        cols_to_scale_present_test = [col for col in numeric_cols_to_scale_final if col in X_test.columns]

        if not cols_to_scale_present_train: print(f"[WARN Main] No numeric (non-OHE) columns found for scaling.")
        else:
            print(f"{datetime.now()} - [INFO Main] Scaling {len(cols_to_scale_present_train)} numeric columns...")
            scaler = StandardScaler()
            X_train.loc[:, cols_to_scale_present_train] = scaler.fit_transform(X_train[cols_to_scale_present_train])
            if cols_to_scale_present_test:
                 cols_to_transform = [col for col in cols_to_scale_present_train if col in cols_to_scale_present_test]
                 if cols_to_transform: X_test.loc[:, cols_to_transform] = scaler.transform(X_test[cols_to_transform])
                 else: print(f"[WARN Main] No matching columns in test set for scaling.")
            else: print(f"[WARN Main] No numeric columns identified for scaling in test set.")
            print(f"{datetime.now()} - [INFO Main] StandardScaler applied.")

    # --- Model Training: XGBoost ---
    print(f"\n{datetime.now()} - [INFO Main] --- Training XGBoost Model (Optimizing for Precision) ---")
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"{datetime.now()} - [INFO Main] Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    min_samples_per_split = len(X_train) // (N_CV_SPLITS + 1)
    if min_samples_per_split < 2:
        print(f"[WARN Main] Training data size ({len(X_train)}) too small for {N_CV_SPLITS} splits. Reducing.")
        actual_n_splits = max(2, len(X_train) // 2) if len(X_train) >= 4 else 1
        if actual_n_splits < 2: print("[ERROR Main] Not enough data for even 2 CV splits."); exit()
    else: actual_n_splits = N_CV_SPLITS
    print(f"{datetime.now()} - [INFO Main] Using TimeSeriesSplit with n_splits={actual_n_splits}")
    tscv = TimeSeriesSplit(n_splits=actual_n_splits)
    xgb_model = XGBClassifier(
        tree_method='hist', device='cuda', random_state=RANDOM_STATE,
        eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight
    )
    # --- MODIFIED: Changed scoring to 'precision' ---
    xgb_search = RandomizedSearchCV(
        estimator=xgb_model, param_distributions=XGB_PARAM_DIST, n_iter=N_RANDOM_SEARCH_ITER,
        scoring='precision', cv=tscv, n_jobs=-1, verbose=1, random_state=RANDOM_STATE # Optimize for Precision
    )
    print(f"{datetime.now()} - [INFO Main] Starting RandomizedSearchCV for XGBoost ({N_RANDOM_SEARCH_ITER} iterations, scoring='precision')...")
    xgb_search_start_time = datetime.now()
    best_xgb_model = None
    try:
        xgb_search.fit(X_train, y_train)
        xgb_search_end_time = datetime.now()
        print(f"\n{datetime.now()} - [INFO Main] XGBoost RandomizedSearchCV finished. Duration: {xgb_search_end_time - xgb_search_start_time}")
        print("Best XGBoost parameters found (for precision):", xgb_search.best_params_)
        print(f"Best XGBoost CV score (precision): {xgb_search.best_score_:.4f}") # Note: score is now precision
        best_xgb_model = xgb_search.best_estimator_
    except Exception as e:
        xgb_search_end_time = datetime.now(); print(f"\n[ERROR Main] ERROR during XGB RandomizedSearchCV: {e}. Duration: {xgb_search_end_time - xgb_search_start_time}");

    # --- Model Evaluation: XGBoost ---
    if best_xgb_model:
        print(f"\n{datetime.now()} - [INFO Main] Evaluating XGBoost model (tuned for precision) on test set...")
        y_pred_xgb = best_xgb_model.predict(X_test)
        y_prob_xgb = best_xgb_model.predict_proba(X_test)[:, 1]
        print(f"\nXGBoost Test Set Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}") # Still show accuracy
        print("XGBoost Test Set Classification Report:\n", classification_report(y_test, y_pred_xgb, target_names=['SL Hit First (0)', 'TP Hit First (1)'], zero_division=0))
        print("\nXGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
        print(f"\nXGBoost Predicted Probability (Class 1) Stats: Min={np.min(y_prob_xgb):.3f}, Mean={np.mean(y_prob_xgb):.3f}, Max={np.max(y_prob_xgb):.3f}, Std={np.std(y_prob_xgb):.3f}")

        # --- SHAP Analysis for XGBoost ---
        print(f"\n{datetime.now()} - [INFO Main] Calculating SHAP values for XGBoost (on test set sample)...")
        try:
            explainer = shap.TreeExplainer(best_xgb_model)
            sample_size_shap = min(len(X_test), 1000)
            if sample_size_shap > 0:
                X_test_sample_shap = X_test.sample(sample_size_shap, random_state=RANDOM_STATE)
                try: shap_values = explainer.shap_values(X_test_sample_shap)
                except TypeError: print("[WARN SHAP] Retrying SHAP with .values"); shap_values = explainer.shap_values(X_test_sample_shap.values)
                if isinstance(shap_values, list) and len(shap_values) > 1: shap_values_for_plot = shap_values[1]
                else: shap_values_for_plot = shap_values
                print(f"{datetime.now()} - [INFO Main] Generating SHAP summary plot...")
                shap.summary_plot(shap_values_for_plot, X_test_sample_shap, plot_type="bar", max_display=30, show=False)
                if MATPLOTLIB_INSTALLED:
                    try:
                        plt.title(f"SHAP Importance (XGB Precision Tuned - TP Hit First) - {sample_size_shap} samples")
                        plt.savefig("shap_summary_xgb_tp_sl_prec_tuned.png", bbox_inches='tight'); plt.close() # New filename
                        print("SHAP summary plot saved to shap_summary_xgb_tp_sl_prec_tuned.png")
                    except Exception as plot_e: print(f"[ERROR Main] Error saving SHAP plot: {plot_e}")
                else: print("[WARN Main] Matplotlib not installed. Cannot save SHAP plot.")
            else: print("[WARN Main] Test set empty or too small for SHAP sampling.")
        except Exception as shap_e: print(f"[ERROR Main] Could not calculate or plot SHAP values: {shap_e}")

        # --- Save Best XGBoost Model ---
        xgb_model_filename = 'revamped_xgb_model_tp_sl_v2_prec_tuned.joblib' # New filename
        joblib.dump(best_xgb_model, xgb_model_filename); print(f"Best XGBoost model saved to '{xgb_model_filename}'")

    # --- Model Training: Logistic Regression ---
    print(f"\n{datetime.now()} - [INFO Main] --- Training Logistic Regression Model ---")
    logreg_model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000)
    # --- MODIFIED: Use GridSearchCV ---
    logreg_search = GridSearchCV(logreg_model, LOGREG_PARAMS, cv=tscv, scoring='accuracy', verbose=1, n_jobs=-1)
    print(f"{datetime.now()} - [INFO Main] Starting GridSearchCV for Logistic Regression...")
    logreg_search_start_time = datetime.now()
    best_logreg_model = None
    try:
        logreg_search.fit(X_train, y_train)
        logreg_search_end_time = datetime.now()
        print(f"\n{datetime.now()} - [INFO Main] Logistic Regression GridSearchCV finished. Duration: {logreg_search_end_time - logreg_search_start_time}")
        print("Best Logistic Regression parameters found:", logreg_search.best_params_)
        print(f"Best Logistic Regression CV score (accuracy): {logreg_search.best_score_:.4f}")
        best_logreg_model = logreg_search.best_estimator_
    except Exception as e: logreg_search_end_time = datetime.now(); print(f"\n[ERROR Main] ERROR during Logistic Regression GridSearchCV: {e}. Duration: {logreg_search_end_time - logreg_search_start_time}");

    # --- Model Evaluation: Logistic Regression ---
    if best_logreg_model:
        print(f"\n{datetime.now()} - [INFO Main] Evaluating Logistic Regression model on test set...")
        y_pred_logreg = best_logreg_model.predict(X_test)
        y_prob_logreg = best_logreg_model.predict_proba(X_test)[:, 1]
        print(f"\nLogistic Regression Test Set Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")
        print("Logistic Regression Test Set Classification Report:\n", classification_report(y_test, y_pred_logreg, target_names=['SL Hit First (0)', 'TP Hit First (1)'], zero_division=0))
        print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
        print(f"\nLogReg Predicted Probability (Class 1) Stats: Min={np.min(y_prob_logreg):.3f}, Mean={np.mean(y_prob_logreg):.3f}, Max={np.max(y_prob_logreg):.3f}, Std={np.std(y_prob_logreg):.3f}")
        logreg_model_filename = 'revamped_logreg_model_tp_sl_v2.joblib' # New filename
        joblib.dump(best_logreg_model, logreg_model_filename); print(f"Best Logistic Regression model saved to '{logreg_model_filename}'")

    # --- Save Final Features and Scaler ---
    feature_list_filename = 'revamped_model_features_tp_sl_v2.list' # New filename
    try:
         current_features = list(X_train.columns)
         with open(feature_list_filename, 'w') as f:
              for feature in current_features: f.write(f"{feature}\n")
         print(f"Final feature list ({len(current_features)}) saved to '{feature_list_filename}'")
    except Exception as fl_e: print(f"Error saving feature list: {fl_e}")
    if scaler:
        scaler_filename = 'revamped_feature_scaler_tp_sl_v2.joblib' # New filename
        joblib.dump(scaler, scaler_filename); print(f"Scaler saved to '{scaler_filename}'")

    print(f"\n{datetime.now()} - [INFO Main] --- Script Finished ---")