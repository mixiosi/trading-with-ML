# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import pandas.api.types
import traceback
from datetime import datetime

# --- Configuration ---
SYMBOLS = ['AAPL', 'AMD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'QQQ', 'SPY', 'TSLA']
INDEX_SYMBOL = 'SPY'  # Define the market index symbol
DATA_DIR = '.'
FILENAME_TEMPLATE = '{}_5min_historical_data.csv'

# --- Feature Calculation Parameters (Relative to 1-hour bars) ---
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

# --- Labeling Parameters (Relative to 1-hour bars) ---
LABELING_METHOD = 'significant_direction'
LABEL_LOOKAHEAD_N_SIG = 6 # 6 hours lookahead
LABEL_SIGNIFICANCE_THRESHOLD_PCT = 0.01 # 1% move

# (Other labeling method params kept for reference)
LABEL_LOOKAHEAD_N_DIRECTION = 12
LOOKAHEAD_BARS_TP_SL = 12
SL_MULT = 2.0; TP_MULT = 4.0
VOL_Z_LONG_THRESH = 0.05; VOL_Z_SHORT_THRESH = -0.05

# --- Data Processing Parameters ---
MIN_ROWS_PER_SYMBOL = 50
USE_FEATURE_SCALING = True

# --- Model Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_SPLITS = 5

# XGBoost GridSearch Parameters
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.5],
}

# --- Define Base Feature Columns (Features calculated for EACH stock) ---
base_stock_feature_columns = [
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

# --- Define Index Context Features (Calculated ONCE for the index) ---
INDEX_CONTEXT_FEATURE_COLS = ['return_1', 'rsi', 'atr', 'sma_diff'] # Base names
INDEX_CONTEXT_FEATURE_NAMES = [f"{INDEX_SYMBOL}_{col}" for col in INDEX_CONTEXT_FEATURE_COLS] # Renamed

# --- Combine feature lists ---
base_feature_columns = base_stock_feature_columns + INDEX_CONTEXT_FEATURE_NAMES

# --- Identify columns needing scaling ---
# Start with stock numeric features, then add index numeric features
numeric_cols_to_scale = [
    col for col in base_stock_feature_columns if col not in
    ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'] # Exclude cyclical
]
# Add numeric index features to scale list
numeric_index_features = [f"{INDEX_SYMBOL}_{col}" for col in ['return_1', 'rsi', 'atr', 'sma_diff']] # Define numeric index features
numeric_cols_to_scale.extend([col for col in numeric_index_features if col in INDEX_CONTEXT_FEATURE_NAMES])
# Remove duplicates if any
numeric_cols_to_scale = sorted(list(set(numeric_cols_to_scale)))

# --- Feature Calculation Function (Unchanged) ---
def calculate_features(df):
    """Calculate technical indicators and enhanced features for the dataset."""
    # (Function content remains the same as your previous version)
    # ... (Keep the existing function code here) ...
    # Ensure 'sma_diff' is calculated if needed for index features later
    if 'sma' in df.columns and 'close' in df.columns:
        df['sma_diff'] = df['close'] - df['sma']
    else:
         df['sma_diff'] = np.nan # Create column even if calculation fails

    cols_to_check = ['high', 'low', 'close', 'open', 'volume']
    for col in cols_to_check:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
    if df.empty: return df
    if not isinstance(df.index, pd.DatetimeIndex):
         print(f"{datetime.now()} - [DEBUG calculate_features] CRITICAL WARNING: Index is NOT a DatetimeIndex before feature calculation!")
         try:
            df.index = pd.to_datetime(df.index)
            if not isinstance(df.index, pd.DatetimeIndex): raise ValueError("Failed conversion.")
            print(f"{datetime.now()} - [DEBUG calculate_features] Index converted within calculate_features."); df.sort_index(inplace=True)
         except Exception as e:
            print(f"{datetime.now()} - [DEBUG calculate_features] ERROR: Could not ensure DatetimeIndex: {e}. Time features fail.");
            df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan
    else:
        if not df.index.is_monotonic_increasing: print(f"{datetime.now()} - [DEBUG calculate_features] Warning: DatetimeIndex not sorted."); df.sort_index(inplace=True)
    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']; df['lower_band'] = df['sma'] - 2 * df['std']
    high_prices=df['high'].values; low_prices=df['low'].values; close_prices=df['close'].values
    min_len_for_talib = max(SMA_WINDOW, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, STOCH_K, ROC_PERIOD) + 10
    if len(close_prices) >= min_len_for_talib:
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
        df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
        df['macd']=macd; df['macd_signal']=macdsignal; df['macd_hist']=macdhist
        sma_safe = df['sma'].replace(0, 1e-10)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / sma_safe
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=STOCH_K, slowk_period=STOCH_SMOOTH, slowk_matype=0, slowd_period=STOCH_D, slowd_matype=0)
        df['roc'] = talib.ROC(close_prices, timeperiod=ROC_PERIOD)
    else:
        print(f"{datetime.now()} - [DEBUG calculate_features] Warning: Not enough data ({len(close_prices)}) for TA-Lib. Skipping.");
        for col in ['atr','rsi','macd','macd_signal','macd_hist','bb_width','stoch_k','stoch_d','roc']: df[col]=np.nan
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    volume_sma5_safe = df['volume_sma5'].replace(0, 1e-10)
    df['volume_z'] = (df['volume'] - df['volume_sma5']) / volume_sma5_safe
    df['rsi_lag1'] = df['rsi'].shift(1)
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7.0)
        df.drop(columns=['hour', 'dayofweek'], inplace=True)
    elif 'hour_sin' not in df.columns:
        print(f"{datetime.now()} - [DEBUG calculate_features] Warning: Cannot create time features due to non-DatetimeIndex.")
        df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan
    df['return_1'] = df['close'].pct_change(1)
    # --- Calculate sma_diff again after sma is calculated ---
    if 'sma' in df.columns and 'close' in df.columns:
        df['sma_diff'] = df['close'] - df['sma']
    else:
        df['sma_diff'] = np.nan # Ensure column exists
    # Lagging features
    features_to_lag = ['return_1', 'macd_hist', 'stoch_k', 'atr', 'bb_width']
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in LAG_PERIODS:
                col_name = f'{feature}_lag{lag}'
                df[col_name] = df[feature].shift(lag)
        else:
            print(f"{datetime.now()} - [DEBUG calculate_features] Warning: Feature '{feature}' not found for lagging.")
    initial_rows = len(df)
    df.dropna(inplace=True)
    # print(f"{datetime.now()} - [DEBUG calculate_features] Shape after feature calculation & final dropna: {df.shape}. Dropped {initial_rows - len(df)} rows.")
    return df

# --- Labeling Functions (Unchanged) ---
def create_direction_labels(df, n_bars):
    # (Function content remains the same)
    # ...
    if df.empty: return df
    print(f"{datetime.now()} - [DEBUG create_direction_labels] Creating direction labels looking ahead {n_bars} bars...")
    df['future_close'] = df['close'].shift(-n_bars); df.dropna(subset=['future_close'], inplace=True)
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df
    df['label'] = (df['future_close'] > df['close']).astype(int)
    df.drop(columns=['future_close'], inplace=True)
    print(f"{datetime.now()} - [DEBUG create_direction_labels] Created 'label' (direction). Shape: {df.shape}"); print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    return df

def create_significant_move_labels(df, n_bars, threshold_pct):
    # (Function content remains the same)
    # ...
    if df.empty: return df
    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Creating labels for moves > {threshold_pct:.3%} over {n_bars} bars...")
    df['future_close'] = df['close'].shift(-n_bars)
    df.dropna(subset=['future_close'], inplace=True)
    if df.empty: print("DataFrame empty after dropping future_close NaNs."); return df
    price_change_pct = (df['future_close'] - df['close']) / df['close'].replace(0, 1e-10)
    conditions = [ price_change_pct > threshold_pct, price_change_pct < -threshold_pct ]
    choices = [1, 0]
    df['label'] = np.select(conditions, choices, default=-1)
    initial_count = len(df)
    df = df[df['label'] != -1].copy()
    filtered_count = initial_count - len(df)
    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Filtered out {filtered_count} insignificant moves ({filtered_count/max(1, initial_count):.2%}).")
    df.drop(columns=['future_close'], inplace=True)
    print(f"{datetime.now()} - [DEBUG create_significant_move_labels] Created 'label' (significant moves only). Shape after filtering: {df.shape}")
    if not df.empty:
        print(f"Label distribution: {df['label'].value_counts(normalize=True, dropna=False).to_dict()}")
    else:
        print("Warning: DataFrame is empty after filtering for significant moves.")
    return df

def generate_signals_and_labels_tp_sl(df, lookahead_bars, sl_mult, tp_mult, vol_z_long_thresh, vol_z_short_thresh):
    # (Function content remains the same)
    # ...
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] --- TP/SL LABELING FUNCTION START ---")
    signal_indices = []; labels = []; long_crossings = 0; short_crossings = 0
    if not isinstance(df.index, pd.DatetimeIndex): raise TypeError("Index must be DatetimeIndex")
    valid_indices = df.index
    if len(valid_indices) <= lookahead_bars:
        print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Warning: Not enough data ({len(valid_indices)} points) for TP/SL lookahead ({lookahead_bars}).")
        return [], []
    is_below_lower = df['close'] < df['lower_band']; is_above_upper = df['close'] > df['upper_band']
    is_vol_z_long = df['volume_z'] > vol_z_long_thresh; is_vol_z_short = df['volume_z'] < vol_z_short_thresh
    has_valid_atr = ~pd.isna(df['atr']) & (df['atr'] > 1e-9)
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Starting loop for TP/SL checks ({len(valid_indices) - lookahead_bars} iterations)...")
    for idx_loc in range(len(valid_indices) - lookahead_bars):
        current_index = valid_indices[idx_loc]; current_data = df.iloc[idx_loc]
        is_long_signal = (is_below_lower.iloc[idx_loc] and is_vol_z_long.iloc[idx_loc] and has_valid_atr.iloc[idx_loc])
        is_short_signal = (is_above_upper.iloc[idx_loc] and is_vol_z_short.iloc[idx_loc] and has_valid_atr.iloc[idx_loc])
        if not (is_long_signal or is_short_signal): continue
        entry_price = current_data['close']; atr = current_data['atr'];
        entry_type = 'long' if is_long_signal else 'short'
        if entry_type == 'long':
            long_crossings += 1; stop_loss = entry_price - sl_mult * atr; take_profit = entry_price + tp_mult * atr
        else:
            short_crossings += 1; stop_loss = entry_price + sl_mult * atr; take_profit = entry_price - tp_mult * atr
        future_data_slice = df.iloc[idx_loc + 1 : idx_loc + 1 + lookahead_bars]
        future_lows = future_data_slice['low']; future_highs = future_data_slice['high']
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
        except Exception as e_generic:
            print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Error during TP/SL check {current_index}: {e_generic}"); traceback.print_exc(); continue
        label = 0
        if hit_tp and hit_sl:
            if pd.notna(first_tp_time) and pd.notna(first_sl_time):
                if first_tp_time <= first_sl_time: label = 1
        elif hit_tp:
            label = 1
        signal_indices.append(current_index); labels.append(label)
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] Finished TP/SL loop.")
    print(f"Longs considered: {long_crossings}, Shorts considered: {short_crossings}, Total signals: {len(signal_indices)}")
    print(f"{datetime.now()} - [DEBUG generate_signals_and_labels_tp_sl] --- TP/SL LABELING FUNCTION END ---")
    return signal_indices, labels

# --- Helper function to load and process index data ---
def process_index_data(index_symbol, data_dir, filename_template):
    """Loads, resamples, and calculates features for the index symbol."""
    print(f"\n{datetime.now()} - [DEBUG process_index_data] --- Processing Index Symbol: {index_symbol} ---")
    file_path = os.path.join(data_dir, filename_template.format(index_symbol))
    if not os.path.exists(file_path):
        print(f"{datetime.now()} - [DEBUG process_index_data] CRITICAL ERROR: Index data file not found: {file_path}. Cannot proceed."); return None

    # --- Load Data (using same robust logic) ---
    index_data = None
    try:
        print(f"{datetime.now()} - [DEBUG process_index_data] Loading 5-minute data for {index_symbol}...");
        load_start_time = datetime.now()
        # (Keep the existing robust date loading logic here) ...
        try:
            correct_date_format = '%Y-%m-%d %H:%M:%S'
            try:
                 index_data = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
                 if not isinstance(index_data.index, pd.DatetimeIndex):
                     print(f"{datetime.now()} - [DEBUG process_index_data] Warning: Index parsing failed. Fallback...");
                     index_data = pd.read_csv(file_path)
                     if 'date' not in index_data.columns: raise ValueError("Column 'date' not found.")
                 else: print(f"{datetime.now()} - [DEBUG process_index_data] Index successfully parsed."); index_data.sort_index(inplace=True)
            except (ValueError, TypeError, KeyError, pd.errors.ParserError) as e_read:
                print(f"{datetime.now()} - [DEBUG process_index_data] Info: read_csv failed ('{e_read}'). Fallback..."); index_data = pd.read_csv(file_path)
                if 'date' not in index_data.columns: raise ValueError("Column 'date' not found.")
            if not isinstance(index_data.index, pd.DatetimeIndex):
                print(f"{datetime.now()} - [DEBUG process_index_data] Executing fallback parse...");
                if 'date' not in index_data.columns: raise ValueError("Column 'date' not found.")
                index_data['date'] = pd.to_datetime(index_data['date'], format=correct_date_format, errors='coerce')
                failed_count = index_data['date'].isnull().sum();
                if failed_count > 0: print(f"{datetime.now()} - [DEBUG process_index_data] Warning: {failed_count} dates failed parsing.")
                index_data.dropna(subset=['date'], inplace=True);
                if index_data.empty: raise ValueError("No valid dates.")
                index_data.set_index('date', inplace=True); index_data.sort_index(inplace=True)
                if not isinstance(index_data.index, pd.DatetimeIndex): raise TypeError("Failed DatetimeIndex.")
                print(f"{datetime.now()} - [DEBUG process_index_data] Index parsed/set with fallback.")
            load_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG process_index_data] Loaded {len(index_data)} rows (5-min). Duration: {load_end_time - load_start_time}")
            if index_data.empty: raise ValueError("Index data empty after loading.")
        except Exception as e: print(f"{datetime.now()} - [DEBUG process_index_data] CRITICAL Error loading index dates: {e}"); traceback.print_exc(); return None

        # --- Clean OHLCV ---
        print(f"{datetime.now()} - [DEBUG process_index_data] Cleaning index OHLCV...");
        initial_rows = len(index_data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in index_data.columns:
                if not pd.api.types.is_numeric_dtype(index_data[col]): index_data[col] = pd.to_numeric(index_data[col], errors='coerce')
            else: print(f"{datetime.now()} - [DEBUG process_index_data] Warning: Column {col} not found.")
        index_data.dropna(subset=[c for c in ohlcv_cols if c in index_data.columns], inplace=True)
        print(f"{datetime.now()} - [DEBUG process_index_data] Dropped {initial_rows - len(index_data)} rows.")
        if index_data.empty: raise ValueError("Index data empty after cleaning.")

        # --- Resample to Hourly ---
        print(f"{datetime.now()} - [DEBUG process_index_data] Resampling index to 1-hour bars...")
        aggregation_dict = {'open': 'first','high': 'max','low': 'min','close': 'last','volume': 'sum'}
        cols_to_agg = {k: v for k, v in aggregation_dict.items() if k in index_data.columns}
        if not cols_to_agg: raise ValueError("No OHLCV columns found to resample index.")
        if 'volume' in cols_to_agg and not pd.api.types.is_numeric_dtype(index_data['volume']):
            index_data['volume'] = pd.to_numeric(index_data['volume'], errors='coerce')
            index_data.dropna(subset=['volume'], inplace=True)
        index_data_hourly = index_data.resample('H').agg(cols_to_agg)
        index_data_hourly.dropna(inplace=True)
        print(f"{datetime.now()} - [DEBUG process_index_data] Resampling complete. Hourly shape: {index_data_hourly.shape}.")
        if index_data_hourly.empty: raise ValueError("Index data empty after resampling.")

        # --- Calculate Features ---
        print(f"{datetime.now()} - [DEBUG process_index_data] Calculating index features (hourly)...")
        index_features = calculate_features(index_data_hourly.copy()) # Use copy to avoid modifying original
        if index_features.empty: raise ValueError("Index features empty after calculation.")
        print(f"{datetime.now()} - [DEBUG process_index_data] Index features calculated.")

        # --- Select and Rename Context Features ---
        print(f"{datetime.now()} - [DEBUG process_index_data] Selecting and renaming context features...")
        context_features_to_select = {}
        missing_base_cols = []
        for col in INDEX_CONTEXT_FEATURE_COLS:
            if col in index_features.columns:
                context_features_to_select[f"{index_symbol}_{col}"] = index_features[col]
            else:
                missing_base_cols.append(col)
                print(f"{datetime.now()} - [DEBUG process_index_data] Warning: Base context feature '{col}' not found in calculated index features.")

        if not context_features_to_select:
             raise ValueError("No context features could be selected.")
        if missing_base_cols:
             print(f"{datetime.now()} - [DEBUG process_index_data] Warning: Missing base context features: {missing_base_cols}. Proceeding with available ones.")

        index_context_df = pd.DataFrame(context_features_to_select)
        print(f"{datetime.now()} - [DEBUG process_index_data] Finished processing index. Context features shape: {index_context_df.shape}")
        return index_context_df

    except Exception as e:
        print(f"\n{datetime.now()} - [DEBUG process_index_data] --- Error processing index symbol: {index_symbol} ---")
        print(f"Error Type: {type(e).__name__}, Message: {e}"); traceback.print_exc()
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    all_symbol_data_list = []
    print(f"{datetime.now()} - [DEBUG Main] Script starting.")

    # --- >>> NEW: Pre-process Index Data <<< ---
    index_context_features = process_index_data(INDEX_SYMBOL, DATA_DIR, FILENAME_TEMPLATE)
    if index_context_features is None:
        print(f"{datetime.now()} - [DEBUG Main] CRITICAL: Failed to process index data for {INDEX_SYMBOL}. Exiting.")
        exit()
    # --- >>> END: Pre-process Index Data <<< ---


    print(f"{datetime.now()} - [DEBUG Main] Starting data processing loop for {len(SYMBOLS)} symbols...")
    for symbol in SYMBOLS:
        # --- >>> NEW: Skip processing the index symbol itself in the main loop <<< ---
        if symbol == INDEX_SYMBOL:
            print(f"\n{datetime.now()} - [DEBUG Main] --- Skipping Index Symbol {symbol} in main loop ---")
            continue
        # --- >>> END SKIP <<< ---

        print(f"\n{datetime.now()} - [DEBUG Main] --- Processing Symbol: {symbol} ---")
        file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(symbol))
        if not os.path.exists(file_path): print(f"{datetime.now()} - [DEBUG Main] Warning: Data file not found: {file_path}. Skipping."); continue
        data = None
        try:
            # Load 5-min data (same as before)
            print(f"{datetime.now()} - [DEBUG Main] Loading 5-minute data for {symbol}...");
            load_start_time = datetime.now()
            # ... (Keep the existing robust date loading logic here) ...
            try:
                correct_date_format = '%Y-%m-%d %H:%M:%S'
                try:
                     data = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
                     if not isinstance(data.index, pd.DatetimeIndex):
                         print(f"{datetime.now()} - [DEBUG Main] Warning: Index parsing failed. Fallback...");
                         data = pd.read_csv(file_path)
                         if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
                     else: print(f"{datetime.now()} - [DEBUG Main] Index successfully parsed."); data.sort_index(inplace=True)
                except (ValueError, TypeError, KeyError, pd.errors.ParserError) as e_read:
                    print(f"{datetime.now()} - [DEBUG Main] Info: read_csv failed ('{e_read}'). Fallback..."); data = pd.read_csv(file_path)
                    if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
                if not isinstance(data.index, pd.DatetimeIndex):
                    print(f"{datetime.now()} - [DEBUG Main] Executing fallback parse...");
                    if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
                    data['date'] = pd.to_datetime(data['date'], format=correct_date_format, errors='coerce')
                    failed_count = data['date'].isnull().sum();
                    if failed_count > 0: print(f"{datetime.now()} - [DEBUG Main] Warning: {failed_count} dates failed parsing.")
                    data.dropna(subset=['date'], inplace=True);
                    if data.empty: raise ValueError("No valid dates.")
                    data.set_index('date', inplace=True); data.sort_index(inplace=True)
                    if not isinstance(data.index, pd.DatetimeIndex): raise TypeError("Failed DatetimeIndex.")
                    print(f"{datetime.now()} - [DEBUG Main] Index parsed/set with fallback.")
                load_end_time = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Loaded {len(data)} rows (5-min). Duration: {load_end_time - load_start_time}")
                if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after loading for {symbol}."); continue
            except Exception as e: print(f"{datetime.now()} - [DEBUG Main] CRITICAL Error loading dates for {symbol}: {e}"); traceback.print_exc(); continue

            # Clean OHLCV (same as before)
            print(f"{datetime.now()} - [DEBUG Main] Cleaning OHLCV columns...");
            clean_start_time = datetime.now()
            initial_rows = len(data); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]): data[col] = pd.to_numeric(data[col], errors='coerce')
                else: print(f"{datetime.now()} - [DEBUG Main] Warning: Column {col} not found.")
            data.dropna(subset=[c for c in ohlcv_cols if c in data.columns], inplace=True)
            clean_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_rows - len(data)} rows (5-min). Duration: {clean_end_time - clean_start_time}")
            if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after cleaning."); continue

            # Resample to Hourly (same as before)
            print(f"{datetime.now()} - [DEBUG Main] Resampling 5-min data to 1-hour bars for {symbol}...")
            resample_start_time = datetime.now()
            aggregation_dict = {'open': 'first','high': 'max','low': 'min','close': 'last','volume': 'sum'}
            cols_to_agg = {k: v for k, v in aggregation_dict.items() if k in data.columns}
            if not cols_to_agg:
                 print(f"{datetime.now()} - [DEBUG Main] Warning: No OHLCV columns found to resample. Skipping.")
            else:
                 if 'volume' in cols_to_agg and not pd.api.types.is_numeric_dtype(data['volume']):
                      data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
                      data.dropna(subset=['volume'], inplace=True)
                 data = data.resample('H').agg(cols_to_agg)
                 data.dropna(inplace=True)
                 resample_end_time = datetime.now()
                 print(f"{datetime.now()} - [DEBUG Main] Resampling complete. Hourly shape: {data.shape}. Duration: {resample_end_time - resample_start_time}")
                 if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after resampling."); continue

            # Calculate Stock Features (same as before)
            print(f"{datetime.now()} - [DEBUG Main] Calculating features for {symbol} (hourly data)...")
            feature_start_time = datetime.now()
            data = calculate_features(data)
            feature_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Finished calculating features. Duration: {feature_end_time - feature_start_time}")
            if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after features."); continue

            # --- >>> NEW: JOIN INDEX CONTEXT FEATURES <<< ---
            print(f"{datetime.now()} - [DEBUG Main] Joining index context features ({INDEX_SYMBOL})...")
            join_start_time = datetime.now()
            initial_rows_before_join = len(data)
            data = data.join(index_context_features, how='left')
            # Drop rows where index features couldn't be joined (e.g., missing index data for that hour)
            data.dropna(subset=INDEX_CONTEXT_FEATURE_NAMES, inplace=True)
            rows_dropped_join = initial_rows_before_join - len(data)
            join_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Join complete. Shape after join: {data.shape}. Dropped {rows_dropped_join} rows due to missing index context. Duration: {join_end_time - join_start_time}")
            if data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Data empty after joining index features."); continue
             # --- >>> END JOIN <<< ---

            if len(data) < MIN_ROWS_PER_SYMBOL: # Check row count after joining
                 print(f"{datetime.now()} - [DEBUG Main] Error or insufficient data ({len(data)} rows) after joining index features for {symbol}. Skipping."); continue

            # Add symbol column
            data['symbol'] = symbol

            # Generate Labels (same as before)
            print(f"{datetime.now()} - [DEBUG Main] Applying labeling method: '{LABELING_METHOD}' for {symbol}...")
            label_start_time = datetime.now()
            # ...(labeling logic remains the same)...
            if LABELING_METHOD == 'direction':
                data = create_direction_labels(data, n_bars=LABEL_LOOKAHEAD_N_DIRECTION)
            elif LABELING_METHOD == 'significant_direction':
                 data = create_significant_move_labels(data,
                                                       n_bars=LABEL_LOOKAHEAD_N_SIG,
                                                       threshold_pct=LABEL_SIGNIFICANCE_THRESHOLD_PCT)
            elif LABELING_METHOD == 'tp_sl':
                print(f"{datetime.now()} - [DEBUG Main] Generating TP/SL signals and labels for {symbol} (hourly data)...")
                signal_indices, labels = generate_signals_and_labels_tp_sl(data, LOOKAHEAD_BARS_TP_SL, SL_MULT, TP_MULT, VOL_Z_LONG_THRESH, VOL_Z_SHORT_THRESH)
                label_end_time = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Finished generating TP/SL signals/labels for {symbol}. Duration: {label_end_time - label_start_time}")
                if len(signal_indices) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient signals ({len(signal_indices)}) for {symbol}. Skipping."); continue
                # ---> NOTE: Base feature columns now include index features
                missing_base_features = [f for f in base_feature_columns if f not in data.columns];
                if missing_base_features: print(f"{datetime.now()} - [DEBUG Main] FATAL: Base features missing for TP/SL: {missing_base_features}. Skipping."); continue
                try:
                    print(f"{datetime.now()} - [DEBUG Main] Selecting TP/SL signal features...")
                    signal_features = data.loc[signal_indices, base_feature_columns + ['symbol']].copy() # Base features includes index context now
                except KeyError as ke: print(f"{datetime.now()} - [DEBUG Main] Error selecting signal features: {ke}"); traceback.print_exc(); continue
                signal_features['label'] = labels; initial_count = len(signal_features)
                signal_features.dropna(inplace=True);
                print(f"{datetime.now()} - [DEBUG Main] Dropped {initial_count - len(signal_features)} TP/SL rows due to NaNs in selected features.")
                if len(signal_features) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient TP/SL data ({len(signal_features)}). Skipping."); continue
                current_symbol_data = signal_features
                all_symbol_data_list.append(current_symbol_data)
                print(f"{datetime.now()} - [DEBUG Main] Successfully processed {symbol} (TP/SL). Added {len(current_symbol_data)} rows.")
                continue
            else:
                print(f"{datetime.now()} - [DEBUG Main] Error: Unknown LABELING_METHOD '{LABELING_METHOD}'. Skipping symbol."); continue
            label_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Finished applying labeling method '{LABELING_METHOD}'. Duration: {label_end_time - label_start_time}")


            # Post-labeling processing
            if LABELING_METHOD in ['direction', 'significant_direction']:
                if data.empty or 'label' not in data.columns or data['label'].isnull().all():
                     print(f"{datetime.now()} - [DEBUG Main] Error: No valid labels or labels are all NaN. Skipping.")
                     continue
                # ---> NOTE: Base feature columns now include index features
                missing_base_features = [f for f in base_feature_columns if f not in data.columns]
                if missing_base_features: print(f"{datetime.now()} - [DEBUG Main] FATAL: Base features missing post-labeling: {missing_base_features}. Skipping."); continue
                cols_to_keep = base_feature_columns + ['label', 'symbol']
                current_symbol_data = data[cols_to_keep].copy()
                if current_symbol_data.isnull().values.any():
                    print(f"{datetime.now()} - [DEBUG Main] Warning: NaNs detected in final selection. Dropping."); initial_count = len(current_symbol_data)
                    current_symbol_data.dropna(inplace=True); print(f"Dropped {initial_count - len(current_symbol_data)} rows.")
                if len(current_symbol_data) < MIN_ROWS_PER_SYMBOL: print(f"{datetime.now()} - [DEBUG Main] Warning: Insufficient data ({len(current_symbol_data)}) after finalizing labels. Skipping."); continue
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

    # Sort, OHE, Split, Scale (code remains same, uses the updated base_feature_columns list)
    print(f"{datetime.now()} - [DEBUG Main] Sorting combined data...")
    sort_start_time = datetime.now()
    combined_data.sort_index(inplace=True)
    sort_end_time = datetime.now()
    print(f"{datetime.now()} - [DEBUG Main] Sorting complete. Duration: {sort_end_time - sort_start_time}")
    print(f"{datetime.now()} - [DEBUG Main] Total rows after sorting: {len(combined_data)}")

    print(f"{datetime.now()} - [DEBUG Main] One-hot encoding 'symbol' column...")
    ohe_start_time = datetime.now()
    try:
        combined_data = pd.get_dummies(combined_data, columns=['symbol'], prefix='sym', drop_first=False)
        ohe_end_time = datetime.now()
        print(f"{datetime.now()} - [DEBUG Main] OHE complete. Shape: {combined_data.shape}. Duration: {ohe_end_time - ohe_start_time}")
    except KeyError: print("Error: 'symbol' column not found.")

    # ---> Use the UPDATED base_feature_columns which includes index features
    final_feature_columns = base_feature_columns.copy()
    ohe_symbol_cols = [col for col in combined_data.columns if col.startswith('sym_')]
    final_feature_columns.extend(ohe_symbol_cols)
    final_feature_columns = sorted(list(set(final_feature_columns)))
    print(f"{datetime.now()} - [DEBUG Main] Final feature columns count: {len(final_feature_columns)}")


    if combined_data.isnull().values.any():
        print(f"{datetime.now()} - [DEBUG Main] CRITICAL Warning: NaNs detected before split."); nan_counts = combined_data.isnull().sum(); print("NaN counts:\n", nan_counts[nan_counts > 0])
        print(f"{datetime.now()} - [DEBUG Main] Dropping NaNs..."); combined_data.dropna(inplace=True); print(f"Rows remaining: {len(combined_data)}")
        if combined_data.empty: print(f"{datetime.now()} - [DEBUG Main] Error: Combined data empty after final NaN drop."); exit()

    print(f"{datetime.now()} - [DEBUG Main] Separating features (X) and labels (y)...")
    missing_final_features = [f for f in final_feature_columns if f not in combined_data.columns]
    if missing_final_features: print(f"{datetime.now()} - [DEBUG Main] FATAL Error: Missing final features: {missing_final_features}"); exit()
    if 'label' not in combined_data.columns: print(f"{datetime.now()} - [DEBUG Main] FATAL Error: 'label' column missing!"); exit()
    X = combined_data[final_feature_columns]; y = combined_data['label']
    print(f"{datetime.now()} - [DEBUG Main] Feature matrix shape: {X.shape}")
    if len(X) == 0 or len(y) == 0: print("Error: X or y empty."); exit()
    if len(y.unique()) < 2: print(f"Error: Only one class found: {y.unique()}"); exit()

    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train = X.iloc[:split_index]; X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]; y_test = y.iloc[split_index:]
    print(f"\n{datetime.now()} - [DEBUG Main] Splitting data chronologically:")
    print(f"Training set: {X_train.index.min()} to {X_train.index.max()}, size: {len(X_train)}")
    print(f"Testing set : {X_test.index.min()} to {X_test.index.max()}, size: {len(X_test)}")
    if len(X_train) == 0 or len(X_test) == 0: print("Error: Empty train/test set."); exit()

    scaler = None
    if USE_FEATURE_SCALING:
        print(f"\n{datetime.now()} - [DEBUG Main] Applying StandardScaler...");
        scale_start_time = datetime.now()
        # ---> Use the UPDATED numeric_cols_to_scale list
        cols_to_scale_present_train = [col for col in numeric_cols_to_scale if col in X_train.columns]
        cols_to_scale_present_test = [col for col in numeric_cols_to_scale if col in X_test.columns]
        if not cols_to_scale_present_train:
             print(f"{datetime.now()} - [DEBUG Main] Warning: No numeric columns identified for scaling in train set.")
        else:
            print(f"{datetime.now()} - [DEBUG Main] Scaling columns ({len(cols_to_scale_present_train)}): {cols_to_scale_present_train[:5]}...")
            scaler = StandardScaler()
            X_train.loc[:, cols_to_scale_present_train] = scaler.fit_transform(X_train[cols_to_scale_present_train])
            if cols_to_scale_present_test:
                 cols_to_transform = [col for col in cols_to_scale_present_train if col in cols_to_scale_present_test]
                 if cols_to_transform:
                      X_test.loc[:, cols_to_transform] = scaler.transform(X_test[cols_to_transform])
                 else:
                      print(f"{datetime.now()} - [DEBUG Main] Warning: No matching columns in test set for scaling.")
            else:
                 print(f"{datetime.now()} - [DEBUG Main] Warning: No numeric columns identified for scaling in test set.")
            scale_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] StandardScaler applied. Duration: {scale_end_time - scale_start_time}")

    # GridSearchCV setup (same as before, uses updated N_CV_SPLITS)
    print(f"\n{datetime.now()} - [DEBUG Main] Setting up GridSearchCV...")
    min_samples_per_split = len(X_train) // (N_CV_SPLITS + 1)
    if min_samples_per_split < 1:
        print(f"{datetime.now()} - [DEBUG Main] Warning: Training data size ({len(X_train)}) too small for {N_CV_SPLITS} splits. Reducing.")
        actual_n_splits = max(2, len(X_train) // 2)
    else:
        actual_n_splits = N_CV_SPLITS
    print(f"{datetime.now()} - [DEBUG Main] Using TimeSeriesSplit with n_splits={actual_n_splits}")
    tscv = TimeSeriesSplit(n_splits=actual_n_splits)

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"{datetime.now()} - [DEBUG Main] Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    print(f"{datetime.now()} - [DEBUG Main] Instantiating XGBClassifier...")
    xgb_model = XGBClassifier(
        tree_method='hist', device='cuda', random_state=RANDOM_STATE,
        eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight
    )
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=XGB_PARAM_GRID, scoring='accuracy', cv=tscv, n_jobs=-1, verbose=2)

    print(f"{datetime.now()} - [DEBUG Main] Starting GridSearchCV with {len(X_train)} samples...")
    gridsearch_start_time = datetime.now()
    best_model = None
    try:
        # Fit, Evaluate, Save (code remains the same)
        # ... (Keep the existing try/except block for fitting/fallback) ...
        grid_search.fit(X_train, y_train)
        gridsearch_end_time = datetime.now()
        print(f"\n{datetime.now()} - [DEBUG Main] GridSearchCV finished. Duration: {gridsearch_end_time - gridsearch_start_time}")
        print("Best parameters found:", grid_search.best_params_)
        print(f"Best CV score ({grid_search.scoring}): {grid_search.best_score_:.4f}")
        best_model = grid_search.best_estimator_
    except Exception as e:
        gridsearch_end_time = datetime.now()
        print(f"\n{datetime.now()} - [DEBUG Main] ERROR during GridSearchCV.fit: {e}. Duration before error: {gridsearch_end_time - gridsearch_start_time}")
        print("Attempting fallback: train default model with GPU...")
        try:
            fallback_start_time = datetime.now()
            best_model = XGBClassifier(
                tree_method='hist', device='cuda', random_state=RANDOM_STATE,
                eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight
            )
            best_model.fit(X_train, y_train)
            fallback_end_time = datetime.now()
            print(f"{datetime.now()} - [DEBUG Main] Successfully trained default model (GPU attempt) as fallback. Duration: {fallback_end_time - fallback_start_time}")
        except Exception as train_e:
            print(f"{datetime.now()} - [DEBUG Main] GPU fallback training failed: {train_e}")
            print("Attempting CPU fallback training...")
            try:
                fallback_cpu_start_time = datetime.now()
                best_model = XGBClassifier(
                    random_state=RANDOM_STATE, eval_metric='logloss',
                    use_label_encoder=False, scale_pos_weight=scale_pos_weight
                )
                best_model.fit(X_train, y_train)
                fallback_cpu_end_time = datetime.now()
                print(f"{datetime.now()} - [DEBUG Main] Successfully trained default model (CPU) as final fallback. Duration: {fallback_cpu_end_time - fallback_cpu_start_time}")
            except Exception as cpu_train_e:
                print(f"{datetime.now()} - [DEBUG Main] FATAL: Failed CPU fallback training: {cpu_train_e}"); traceback.print_exc(); exit()

    if best_model is None: print(f"{datetime.now()} - [DEBUG Main] FATAL: No model trained."); exit()

    print(f"\n{datetime.now()} - [DEBUG Main] Evaluating model on test set...")
    eval_start_time = datetime.now()
    y_pred = best_model.predict(X_test)
    eval_end_time = datetime.now()
    print(f"{datetime.now()} - [DEBUG Main] Prediction finished. Duration: {eval_end_time - eval_start_time}")
    print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Test Set Classification Report:\n", classification_report(y_test, y_pred, target_names=['Significant Down (0)', 'Significant Up (1)'], zero_division=0))

    print("\nFeature Importances:")
    try:
        # ---> Features names now include index features
        feature_names = X_train.columns
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df.head(30).to_string()) # Print top 30 to see if index features appear
    except Exception as fi_e: print(f"Could not get feature importances: {fi_e}")

    # Save Model, Features, Scaler (consider renaming output files)
    print(f"\n{datetime.now()} - [DEBUG Main] Saving artifacts...")
    model_filename = 'trading_xgb_model_multisym_sigmove_tuned_v3_1H_context.joblib' # Example new name
    joblib.dump(best_model, model_filename); print(f"Model saved to '{model_filename}'")
    feature_list_filename = 'trading_model_features_multisym_sigmove_v3_1H_context.list' # Example new name
    try:
         current_features = list(X_train.columns)
         with open(feature_list_filename, 'w') as f:
              for feature in current_features: f.write(f"{feature}\n")
         print(f"Feature list ({len(current_features)}) saved to '{feature_list_filename}'")
    except Exception as fl_e: print(f"Error saving feature list: {fl_e}")
    if scaler:
        scaler_filename = 'trading_feature_scaler_multisym_sigmove_v3_1H_context.joblib' # Example new name
        joblib.dump(scaler, scaler_filename); print(f"Scaler saved to '{scaler_filename}'")

    print(f"\n{datetime.now()} - [DEBUG Main] --- Script Finished ---")