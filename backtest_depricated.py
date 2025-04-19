import pandas as pd
import numpy as np
import talib
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback
from datetime import datetime # For logging

# --- Configuration ---
SYMBOL_TO_BACKTEST = 'NVDA' # Symbol to backtest
MARKET_INDEX_SYMBOL = 'SPY' # For context features (must match training)
DATA_DIR = '.'
# --- >>> USE 1-HOUR DATA (Matches Training) <<< ---
FILENAME_TEMPLATE = '{}_1h_historical_data.csv'

# --- >>> LOAD ARTIFACTS FROM TRAINING <<< ---
# Make sure these filenames match EXACTLY what was saved by the training script
# If you used the SMOTE tag, include it here if applicable
smote_tag_in_filename = "" # Set to "_smote" if you saved SMOTE-trained artifacts
timestamp_str_from_training = "20250402_1345" # <<< --- CHANGE this to the timestamp of your saved model --- >>>

MODEL_FILE = f'trading_xgb_model_1h_sig{smote_tag_in_filename}_{timestamp_str_from_training}.joblib'
FEATURE_LIST_FILE = f'trading_model_features_1h_sig{smote_tag_in_filename}_{timestamp_str_from_training}.list'
SCALER_FILE = f'trading_feature_scaler_1h_sig{smote_tag_in_filename}_{timestamp_str_from_training}.joblib'
SCALER_USED_IN_TRAINING = True # Set to False if USE_FEATURE_SCALING was False during training

# Backtesting Parameters
INITIAL_CAPITAL = 30000.0
TRADE_SHARES = 10 # Number of shares per trade

# Cost Parameters
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.02 # Per share, applied on entry and exit

# Strategy & Risk Management Parameters
SL_MULT = 2.0 # Stop loss multiplier (x ATR)
TP_MULT = 4.0 # Take profit multiplier (x ATR)
TRADE_ON_SIGNAL_1 = True  # Enter LONG when model predicts 1 (Significant Up)
TRADE_ON_SIGNAL_0 = False # Enter SHORT when model predicts 0 (Significant Down) - Set to True to enable

# Feature Calculation Parameters (Should match trainer)
SMA_WINDOW = 20; ATR_PERIOD = 14; RSI_PERIOD = 14
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
STOCH_K = 14; STOCH_D = 3; STOCH_SMOOTH = 3
ROC_PERIOD = 10; LAG_PERIODS = [1, 3, 5]

PROBABILITY_THRESHOLD = 0.55

# --- Feature Calculation Function (Copied from Trainer) ---
def calculate_features(df, symbol_name="Data"):
    """Calculate technical indicators and enhanced features for the dataset."""
    print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Input shape: {df.shape}")
    cols_to_check = ['high', 'low', 'close', 'open', 'volume']
    for col in cols_to_check:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['high', 'low', 'close', 'volume'], inplace=True)
    if df.empty: return df

    if not isinstance(df.index, pd.DatetimeIndex):
         print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) CRITICAL WARNING: Index is NOT DatetimeIndex!")
         try:
            df.index = pd.to_datetime(df.index)
            if not isinstance(df.index, pd.DatetimeIndex): raise ValueError("Failed conversion.")
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Index converted."); df.sort_index(inplace=True)
         except Exception as e:
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) ERROR: Could not ensure DatetimeIndex: {e}. Time features fail.");
            df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan
    else:
        if not df.index.is_monotonic_increasing: df.sort_index(inplace=True)

    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']; df['lower_band'] = df['sma'] - 2 * df['std']

    high_prices=df['high'].copy().values; low_prices=df['low'].copy().values; close_prices=df['close'].copy().values
    min_len_for_talib = max(SMA_WINDOW, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, STOCH_K, ROC_PERIOD) + 10
    if len(close_prices) >= min_len_for_talib:
        try:
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
            df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
            macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
            df['macd']=macd; df['macd_signal']=macdsignal; df['macd_hist']=macdhist
            sma_safe = df['sma'].replace(0, 1e-10)
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / sma_safe
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=STOCH_K, slowk_period=STOCH_SMOOTH, slowk_matype=0, slowd_period=STOCH_D, slowd_matype=0)
            df['roc'] = talib.ROC(close_prices, timeperiod=ROC_PERIOD)
        except Exception as talib_e:
             print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Error during TA-Lib calculations: {talib_e}. Assigning NaNs.")
             for col in ['atr','rsi','macd','macd_signal','macd_hist','bb_width','stoch_k','stoch_d','roc']: df[col]=np.nan
    else:
        print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Not enough data ({len(close_prices)}) for TA-Lib. Skipping.");
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
        print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Cannot create time features due to non-DatetimeIndex.")
        df['hour_sin']=np.nan; df['hour_cos']=np.nan; df['dayofweek_sin']=np.nan; df['dayofweek_cos']=np.nan

    df['return_1'] = df['close'].pct_change(1)

    features_to_lag = ['return_1', 'macd_hist', 'stoch_k', 'atr', 'bb_width']
    new_lag_cols = []
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in LAG_PERIODS:
                col_name = f'{feature}_lag{lag}'
                df[col_name] = df[feature].shift(lag)
                new_lag_cols.append(col_name)
        else:
            print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Warning: Feature '{feature}' not found for lagging.")

    # --- Drop NaNs AFTER all calculations for this symbol ---
    # Don't drop here if merging SPY features later, drop after merge
    # df.dropna(inplace=True) # Moved dropna to prepare_backtest_data
    print(f"{datetime.now()} - [DEBUG calculate_features] ({symbol_name}) Features calculated. Shape before explicit dropna: {df.shape}")
    return df

# --- Data Loading Function ---
def load_data(symbol, filename_template, data_dir):
    """Loads and performs initial cleaning of data for a symbol."""
    print(f"{datetime.now()} - [DEBUG load_data] Loading data for {symbol}...")
    file_path = os.path.join(data_dir, filename_template.format(symbol))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        # Assuming 1h data uses '%Y-%m-%d %H:%M:%S' format
        correct_date_format = '%Y-%m-%d %H:%M:%S'
        try:
            df = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
        except ValueError:
            print(f"{datetime.now()} - [DEBUG load_data] ({symbol}) Format parse failed. Attempting without format.")
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

        if not isinstance(df.index, pd.DatetimeIndex):
             print(f"{datetime.now()} - [DEBUG load_data] ({symbol}) Index parsing failed. Attempting fallback.")
             df = pd.read_csv(file_path)
             if 'date' not in df.columns: raise ValueError("Column 'date' not found.")
             df['date'] = pd.to_datetime(df['date'], errors='coerce') # Try auto first
             if df['date'].isnull().all(): # Try specific format if auto fails
                df['date'] = pd.to_datetime(df['date'], format=correct_date_format, errors='coerce')
             df.dropna(subset=['date'], inplace=True)
             if df.empty: raise ValueError("No valid dates found after fallback parsing.")
             df.set_index('date', inplace=True)

        df.sort_index(inplace=True)
        print(f"{datetime.now()} - [DEBUG load_data] Loaded {len(df)} rows ({symbol}). Index type: {type(df.index)}")
        if df.empty: raise ValueError(f"Data empty after loading for {symbol}.")

        # Clean OHLCV
        print(f"{datetime.now()} - [DEBUG load_data] Cleaning OHLCV columns for {symbol}...");
        initial_rows = len(df); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]): df[col] = pd.to_numeric(df[col], errors='coerce')
            else: print(f"{datetime.now()} - [DEBUG load_data] Warning: Column {col} not found for {symbol}.")
        df.dropna(subset=[c for c in ohlcv_cols if c in df.columns], inplace=True)
        print(f"{datetime.now()} - [DEBUG load_data] Dropped {initial_rows - len(df)} rows due to non-numeric/NaN OHLCV ({symbol}).")
        if df.empty: raise ValueError(f"Data empty after cleaning OHLCV for {symbol}.")
        return df

    except Exception as e:
        print(f"{datetime.now()} - [DEBUG load_data] CRITICAL Error loading/cleaning data for {symbol}: {e}");
        traceback.print_exc()
        raise # Re-raise the exception


# --- >>> REVISED: Data Preparation Function (Replicates Training Pipeline but keeps necessary columns) <<< ---
def prepare_backtest_data(target_symbol, target_df, spy_df, expected_features, scaler=None):
    """Prepares the data for backtesting, mirroring the training pipeline
       while retaining columns needed for simulation (high, low, close, atr)."""
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Preparing data for {target_symbol}...")

    # 1. Calculate features for the target symbol
    target_df_featured = calculate_features(target_df.copy(), symbol_name=target_symbol)

    # 2. Prepare SPY context features
    spy_context_features_to_keep = [
        'return_1', 'rsi', 'atr', 'sma', 'volume_z', 'macd_hist' # Should match training
    ]
    spy_context_feature_names = [f"SPY_{col}" for col in spy_context_features_to_keep]
    missing_spy_features = [f for f in spy_context_features_to_keep if f not in spy_df.columns]
    if missing_spy_features:
        raise ValueError(f"Required SPY context features missing from spy_df: {missing_spy_features}")
    spy_context_data = spy_df[spy_context_features_to_keep].copy()
    spy_context_data.rename(columns={col: f"SPY_{col}" for col in spy_context_features_to_keep}, inplace=True)
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Prepared SPY context features: {spy_context_data.columns.tolist()}")

    # 3. Merge SPY context features
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Merging SPY context features onto {target_symbol}...")
    if not isinstance(target_df_featured.index, pd.DatetimeIndex): target_df_featured.index = pd.to_datetime(target_df_featured.index)
    if not isinstance(spy_context_data.index, pd.DatetimeIndex): spy_context_data.index = pd.to_datetime(spy_context_data.index)
    # --- Keep ALL columns from target_df_featured after merge ---
    merged_df = pd.merge(target_df_featured, spy_context_data, left_index=True, right_index=True, how='left')
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Shape after merge: {merged_df.shape}")

    # 4. Create One-Hot Encoded columns
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Creating OHE columns for {target_symbol}...")
    ohe_cols_expected = [f for f in expected_features if f.startswith('sym_')]
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Expected OHE cols: {ohe_cols_expected}")
    for col in ohe_cols_expected:
        if col not in merged_df.columns: # Add OHE column only if it doesn't exist
             merged_df[col] = 0
    target_ohe_col = f'sym_{target_symbol}'
    if target_ohe_col in merged_df.columns: # Check if the column exists before assignment
        merged_df[target_ohe_col] = 1
        print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Set {target_ohe_col} to 1.")
    else:
        # This case should ideally not happen if expected_features is correct
        print(f"{datetime.now()} - [DEBUG prepare_backtest_data] WARNING: OHE column for target '{target_ohe_col}' not found in DataFrame columns. OHE might be incorrect.")

    # 5. Handle NaNs (Crucial: Do this BEFORE scaling and final feature selection check)
    # Drop rows missing SPY context first, as these are definitely unusable
    initial_rows_spy_nan = len(merged_df)
    merged_df.dropna(subset=spy_context_feature_names, inplace=True)
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Dropped {initial_rows_spy_nan - len(merged_df)} rows due to missing SPY context.")
    if merged_df.empty: raise ValueError("Data empty after dropping SPY context NaNs.")

    # Now handle other NaNs that might exist in expected_features or required backtest columns
    # Identify all columns needed: model features + backtest columns
    cols_needed_for_model = expected_features
    cols_needed_for_backtest = ['high', 'low', 'close', 'atr']
    all_required_cols = list(set(cols_needed_for_model + cols_needed_for_backtest))

    initial_rows_final_nan = len(merged_df)
    # Drop rows where *any* required column is NaN
    merged_df.dropna(subset=all_required_cols, inplace=True)
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Dropped {initial_rows_final_nan - len(merged_df)} rows due to NaNs in other required columns.")
    if merged_df.empty: raise ValueError("Data empty after final NaN drop.")


    # 6. Apply Scaling (on the relevant columns within the merged_df)
    final_df = merged_df.copy() # Work on a copy
    if scaler:
        print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Applying scaler...")
        numeric_cols_to_scale = [
            col for col in expected_features if col not in # Scale only features expected by model
            ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'] and not col.startswith('sym_')
        ]
        # Ensure columns actually exist in the dataframe before trying to scale
        cols_to_scale_present = [col for col in numeric_cols_to_scale if col in final_df.columns]

        if cols_to_scale_present:
            print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Scaling {len(cols_to_scale_present)} columns...")
            # --- Scale IN PLACE on the final_df ---
            final_df.loc[:, cols_to_scale_present] = scaler.transform(final_df[cols_to_scale_present].astype(float))
            print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Scaling complete.")
        else:
            print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Warning: No numeric columns found/needed to scale based on expected features.")
    else:
        print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Scaler not provided or not used in training. Skipping scaling.")

    # 7. Final Check: Ensure all required columns still exist after all steps
    missing_model_features = [f for f in expected_features if f not in final_df.columns]
    if missing_model_features:
        raise ValueError(f"Model features missing after final preparation: {missing_model_features}")
    missing_backtest_cols = [f for f in ['high', 'low', 'close', 'atr'] if f not in final_df.columns]
    if missing_backtest_cols:
        raise ValueError(f"Backtest columns missing after final preparation: {missing_backtest_cols}")

    # --- No need to reorder here, run_backtest will select expected_features ---
    print(f"{datetime.now()} - [DEBUG prepare_backtest_data] Data preparation complete for {target_symbol}. Final shape: {final_df.shape}")
    # The final_df now contains ALL necessary columns (scaled model features + raw HLC/ATR etc.)
    return final_df


# --- Backtesting Logic (Uses Model Prediction for Entry) ---
# --- Backtesting Logic (Uses Model Prediction PROBABILITY for Entry) ---
def run_backtest(data_prepared, model, expected_features, capital, trade_shares,
                 sl_mult, tp_mult, commission_per_share, slippage_per_share,
                 trade_long=True, trade_short=False, probability_threshold=0.5): # Added probability_threshold
    """Runs the backtest loop using model prediction probabilities for entry."""
    cash = capital; position = 0; entry_price = 0.0
    stop_loss = 0.0; take_profit = 0.0
    equity = []; trades = []; entry_timestamp = pd.NaT

    # --- Get necessary price/indicator columns from the prepared data ---
    required_cols = ['close', 'high', 'low', 'atr']
    missing_req = [c for c in required_cols if c not in data_prepared.columns]
    if missing_req:
        raise ValueError(f"Required columns missing from prepared data for backtest loop: {missing_req}")

    print(f"\n{datetime.now()} - --- Running Backtest Simulation ---")
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Trade Size: {trade_shares} shares")
    print(f"Trading Logic: Long on Signal 1 >= {probability_threshold:.2f} = {trade_long}, Short on Signal 0 >= {probability_threshold:.2f} = {trade_short}") # Updated log
    print(f"Commission per Share: ${commission_per_share:.4f}")
    print(f"Slippage per Share: ${slippage_per_share:.4f}")
    print(f"SL/TP Multipliers (ATR): {sl_mult}/{tp_mult}")
    print(f"Iterating through {len(data_prepared)} bars...")

    for timestamp, row in data_prepared.iterrows():
        # --- Get prices and ATR for SL/TP logic ---
        current_price = row.get('close', np.nan)
        current_high = row.get('high', np.nan)
        current_low = row.get('low', np.nan)
        current_atr = row.get('atr', np.nan)

        if pd.isna(current_price) or pd.isna(current_high) or pd.isna(current_low) or pd.isna(current_atr):
             # print(f"{datetime.now()} - WARNING: Skipping bar {timestamp} due to missing price/ATR data.") # Can be verbose
             if equity: equity.append({'Timestamp': timestamp, 'Equity': equity[-1]['Equity']})
             else: equity.append({'Timestamp': timestamp, 'Equity': cash})
             continue

        # --- Exit Check (Based on SL/TP) ---
        exit_target_price = None; exit_reason = None
        if position > 0: # Long position
            if current_low <= stop_loss: exit_target_price = stop_loss; exit_reason = "Stop Loss (Long)"
            elif current_high >= take_profit: exit_target_price = take_profit; exit_reason = "Take Profit (Long)"
        elif position < 0: # Short position
            if current_high >= stop_loss: exit_target_price = stop_loss; exit_reason = "Stop Loss (Short)"
            elif current_low <= take_profit: exit_target_price = take_profit; exit_reason = "Take Profit (Short)"

        # --- Process Exit ---
        if exit_target_price is not None:
            commission_cost = commission_per_share * abs(position)
            profit_net = 0.0
            exit_shares = abs(position)

            if position > 0: # Closing Long
                effective_exit_price = exit_target_price - slippage_per_share
                profit_gross = (effective_exit_price - entry_price) * position
                profit_net = profit_gross - commission_cost
                cash += effective_exit_price * position
                cash -= commission_cost
                print(f"{timestamp} - EXIT LONG @ ~{effective_exit_price:.2f} ({exit_reason}). Profit: ${profit_net:.2f}")
            elif position < 0: # Closing Short
                effective_exit_price = exit_target_price + slippage_per_share
                profit_gross = (entry_price - effective_exit_price) * abs(position)
                profit_net = profit_gross - commission_cost
                cash += entry_price * abs(position)
                cash += profit_net
                print(f"{timestamp} - EXIT SHORT @ ~{effective_exit_price:.2f} ({exit_reason}). Profit: ${profit_net:.2f}")

            if pd.notna(entry_timestamp):
                trades.append({'entry_time': entry_timestamp, 'exit_time': timestamp, 'entry_price': entry_price,
                               'exit_price': effective_exit_price, 'position_type': 'long' if position > 0 else 'short',
                               'shares': exit_shares, 'profit': profit_net, 'exit_reason': exit_reason})
            else: print(f"Warning: Exit at {timestamp} without valid entry_timestamp.")

            position = 0; entry_price = 0.0; stop_loss = 0.0; take_profit = 0.0; entry_timestamp = pd.NaT

        # --- >>> Entry Check (Based on Model Prediction Probability) <<< ---
        if position == 0 and current_atr > 1e-9: # Only enter if flat and ATR is valid
            signal_type = None
            # Get the feature vector for the current row (already ordered and scaled)
            features = row[expected_features].values.reshape(1, -1)

            if np.isnan(features).any() or np.isinf(features).any():
                pass # Skip prediction if features are invalid
            else:
                # --- THIS IS THE MODIFIED PART --- START ---
                try:
                    # --- Predict PROBABILITY using the loaded model ---
                    pred_proba_all = model.predict_proba(features)[0]
                    prob_class_1 = pred_proba_all[1] # Probability of "Significant Up"
                    prob_class_0 = pred_proba_all[0] # Probability of "Significant Down"

                    # --- Determine signal based on prediction AND threshold ---
                    if prob_class_1 >= probability_threshold and trade_long:
                        signal_type = 'long'
                        # Optional: Log the probability that triggered the trade
                        print(f"{timestamp} - Potential LONG signal (Proba: {prob_class_1:.3f} >= {probability_threshold:.2f})")
                    elif prob_class_0 >= probability_threshold and trade_short:
                         signal_type = 'short'
                         # Optional: Log the probability that triggered the trade
                         print(f"{timestamp} - Potential SHORT signal (Proba Class 0: {prob_class_0:.3f} >= {probability_threshold:.2f})")
                    # else: No signal met the threshold

                except Exception as e:
                    print(f"{datetime.now()} - ERROR predicting probability at {timestamp}: {e}")
                    signal_type = None # Do not trade if prediction fails
                # --- THIS IS THE MODIFIED PART --- END ---

                # --- Process Entry ---
                if signal_type is not None:
                    commission_cost = commission_per_share * trade_shares
                    required_cash_for_trade = 0

                    if signal_type == 'long':
                        effective_entry_price = current_price + slippage_per_share
                        required_cash_for_trade = effective_entry_price * trade_shares + commission_cost
                        if cash >= required_cash_for_trade:
                            entry_price = effective_entry_price; entry_timestamp = timestamp
                            position = trade_shares
                            stop_loss = entry_price - sl_mult * current_atr
                            take_profit = entry_price + tp_mult * current_atr
                            cash -= required_cash_for_trade
                            # Log the original probability that led to the trade signal
                            print(f"{timestamp} - ENTER LONG @ ~{entry_price:.2f} (Proba: {prob_class_1:.3f}) SL: {stop_loss:.2f} TP: {take_profit:.2f}")
                        else:
                            print(f"{timestamp} - Signal LONG (Proba: {prob_class_1:.3f}), but insufficient capital (${cash:.2f} < ${required_cash_for_trade:.2f})")
                    elif signal_type == 'short':
                        effective_entry_price = current_price - slippage_per_share
                        required_cash_for_trade = commission_cost # Simplification for margin
                        if cash >= required_cash_for_trade:
                            entry_price = effective_entry_price; entry_timestamp = timestamp
                            position = -trade_shares
                            stop_loss = entry_price + sl_mult * current_atr
                            take_profit = entry_price - tp_mult * current_atr
                            cash -= commission_cost
                            # Log the original probability that led to the trade signal
                            print(f"{timestamp} - ENTER SHORT @ ~{entry_price:.2f} (Proba Class 0: {prob_class_0:.3f}) SL: {stop_loss:.2f} TP: {take_profit:.2f}")
                        else:
                             print(f"{timestamp} - Signal SHORT (Proba Class 0: {prob_class_0:.3f}), but insufficient capital for commission (${cash:.2f} < ${required_cash_for_trade:.2f})")


        # --- Update Equity ---
        current_value = cash
        if position > 0: # Holding long
            current_value += position * current_price
        elif position < 0: # Holding short
            unrealized_pnl = (entry_price - current_price) * abs(position)
            current_value += unrealized_pnl
        equity.append({'Timestamp': timestamp, 'Equity': current_value})

    if not equity: print("Warning: No equity points generated."); return pd.DataFrame(), []
    print(f"\n{datetime.now()} - Backtest finished. Final equity: ${equity[-1]['Equity']:,.2f}")
    equity_df = pd.DataFrame(equity).set_index('Timestamp')
    return equity_df, trades

# --- Performance Metrics Function (Identical to previous valid version) ---
def calculate_metrics(equity_df, trades, initial_capital):
    """Calculates performance metrics."""
    if equity_df.empty: print("Warning: Equity DataFrame empty."); return {"Status": "No data"}
    metrics = {}; final_equity = equity_df['Equity'].iloc[-1]
    metrics['Initial Capital'] = initial_capital; metrics['Final Equity'] = final_equity
    metrics['Total Return (%)'] = ((final_equity / initial_capital) - 1) * 100 if initial_capital != 0 else 0
    equity_df['Peak'] = equity_df['Equity'].cummax(); equity_df['Drawdown'] = equity_df['Equity'] - equity_df['Peak']
    # Avoid division by zero if peak equity is zero or NaN
    peak_safe = equity_df['Peak'].replace(0, np.nan)
    equity_df['Drawdown (%)'] = (equity_df['Drawdown'] / peak_safe).fillna(0) * 100
    metrics['Max Drawdown (%)'] = equity_df['Drawdown (%)'].min() if not equity_df['Drawdown (%)'].isnull().all() else 0
    metrics['Max Drawdown Abs'] = equity_df['Drawdown'].min() if not equity_df['Drawdown'].isnull().all() else 0
    num_trades = len(trades); metrics['Number of Trades'] = num_trades
    if num_trades > 0:
        trade_df = pd.DataFrame(trades); wins = trade_df[trade_df['profit'] > 0]; losses = trade_df[trade_df['profit'] <= 0]
        metrics['Number of Wins'] = len(wins); metrics['Number of Losses'] = len(losses)
        metrics['Win Rate (%)'] = (len(wins) / num_trades) * 100 if num_trades > 0 else 0
        metrics['Total Net Profit'] = trade_df['profit'].sum()
        metrics['Average Net Profit per Trade'] = trade_df['profit'].mean()
        metrics['Average Winning Trade (Net)'] = wins['profit'].mean() if len(wins) > 0 else 0
        metrics['Average Losing Trade (Net)'] = losses['profit'].mean() if len(losses) > 0 else 0
        total_won_net = wins['profit'].sum(); total_lost_net = abs(losses['profit'].sum())
        metrics['Profit Factor (Net)'] = total_won_net / total_lost_net if total_lost_net > 0 else np.inf
        metrics['Max Consecutive Wins'] = max_consecutive(trade_df['profit'] > 0)
        metrics['Max Consecutive Losses'] = max_consecutive(trade_df['profit'] <= 0)
    else:
        metrics['Number of Wins'] = 0; metrics['Number of Losses'] = 0; metrics['Win Rate (%)'] = 0
        metrics['Total Net Profit'] = 0; metrics['Average Net Profit per Trade'] = 0
        metrics['Average Winning Trade (Net)'] = 0; metrics['Average Losing Trade (Net)'] = 0
        metrics['Profit Factor (Net)'] = 0; metrics['Max Consecutive Wins'] = 0; metrics['Max Consecutive Losses'] = 0

    try: # Sharpe Ratio Calculation (annualized based on data frequency)
        returns = equity_df['Equity'].pct_change().dropna()
        if len(returns) > 1 and returns.std() != 0:
             # Try to infer frequency, default to daily if inference fails
            inferred_freq = pd.infer_freq(equity_df.index)
            if inferred_freq:
                periods_per_day = pd.Timedelta('1 day') / pd.tseries.frequencies.to_offset(inferred_freq)
            else: # Fallback if frequency cannot be inferred (e.g., irregular data)
                time_diff = equity_df.index.to_series().diff().median()
                if time_diff is not pd.NaT and time_diff.total_seconds() > 0:
                    periods_per_day = pd.Timedelta('1 day').total_seconds() / time_diff.total_seconds()
                else:
                    periods_per_day = 252 # Default assumption if calculation fails
                    print(f"{datetime.now()} - WARNING: Could not infer data frequency. Assuming daily for Sharpe Ratio.")

            trading_periods_per_year = periods_per_day * 252 # Assume 252 trading days
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(trading_periods_per_year)
            metrics['Approx Annual Sharpe Ratio'] = sharpe_ratio
        elif len(returns) <= 1: metrics['Approx Annual Sharpe Ratio'] = 0
        else: metrics['Approx Annual Sharpe Ratio'] = 'Std Dev is zero'
    except Exception as e: print(f"Error calculating Sharpe: {e}"); metrics['Approx Annual Sharpe Ratio'] = 'Error'

    return metrics

def max_consecutive(series):
    max_count = 0; current_count = 0
    for val in series:
        if val: current_count += 1
        else: max_count = max(max_count, current_count); current_count = 0
    return max(max_count, current_count)

# --- Plotting Function (Identical to previous valid version) ---
def plot_equity_curve(equity_df, symbol, cost_info=""):
    """Plots the equity curve."""
    if equity_df.empty or len(equity_df) < 2: print("Not enough data points to plot."); return
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(equity_df.index, equity_df['Equity'], label='Portfolio Value', color='blue', linewidth=1.5)
    max_dd_pct = equity_df['Drawdown (%)'].min()
    max_dd_loc = equity_df['Drawdown (%)'].idxmin() if not equity_df['Drawdown (%)'].isnull().all() and max_dd_pct < 0 else None # Only plot if drawdown exists
    if max_dd_loc: ax.scatter(max_dd_loc, equity_df.loc[max_dd_loc, 'Equity'], color='red', marker='v', s=100, zorder=5, label=f'Max Drawdown ({max_dd_pct:.1f}%)')
    ax.set_title(f'Equity Curve for {symbol} {cost_info}')
    ax.set_xlabel('Date'); ax.set_ylabel('Portfolio Value ($)')
    ax.yaxis.set_major_formatter('${x:,.0f}')
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    plt.xticks(rotation=30, ha='right'); plt.legend(); plt.tight_layout(); plt.grid(True)
    safe_cost_info = cost_info.replace('/','-').replace(':','').replace('(','').replace(')','').replace(',','').replace('$','')
    # Include model timestamp in plot filename
    plot_filename = f"{symbol}_equity_curve_{timestamp_str_from_training}{smote_tag_in_filename}_{safe_cost_info}.png"
    plt.savefig(plot_filename); print(f"Equity curve plot saved to {plot_filename}"); plt.close(fig) # Close the plot after saving


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"{datetime.now()} - --- Starting Backtest ---")
    print(f"Backtesting Symbol: {SYMBOL_TO_BACKTEST}")
    print(f"Using Model: {MODEL_FILE}")
    print(f"Using Feature List: {FEATURE_LIST_FILE}")
    print(f"Using Scaler: {SCALER_FILE if SCALER_USED_IN_TRAINING else 'None (Scaling Disabled in Training)'}")

    # --- Load Model, Features, Scaler ---
    print(f"{datetime.now()} - Loading artifacts...")
    if not os.path.exists(MODEL_FILE): print(f"Error: Model file not found: {MODEL_FILE}"); exit()
    if not os.path.exists(FEATURE_LIST_FILE): print(f"Error: Feature list file not found: {FEATURE_LIST_FILE}"); exit()
    scaler = None
    if SCALER_USED_IN_TRAINING:
        if not os.path.exists(SCALER_FILE): print(f"Error: Scaler file specified but not found: {SCALER_FILE}"); exit()
        try: scaler = joblib.load(SCALER_FILE)
        except Exception as e: print(f"Error loading scaler: {e}"); traceback.print_exc(); exit()

    try:
        model = joblib.load(MODEL_FILE)
        with open(FEATURE_LIST_FILE, 'r') as f: expected_features = [line.strip() for line in f if line.strip()]
        print(f"{datetime.now()} - Loaded model and {len(expected_features)} expected features.")
        if scaler: print(f"{datetime.now()} - Loaded scaler.")
    except Exception as e: print(f"Error loading model/features: {e}"); traceback.print_exc(); exit()

    # --- Load and Prepare Data ---
    try:
        # Load SPY data first for context
        spy_data_raw = load_data(MARKET_INDEX_SYMBOL, FILENAME_TEMPLATE, DATA_DIR)
        # Calculate features needed for context (subset of full feature calculation)
        spy_data_processed = calculate_features(spy_data_raw.copy(), symbol_name=MARKET_INDEX_SYMBOL)

        # Load target symbol data
        target_data_raw = load_data(SYMBOL_TO_BACKTEST, FILENAME_TEMPLATE, DATA_DIR)

        # Prepare the final data using the dedicated function
        data_final_prepared = prepare_backtest_data(
            target_symbol=SYMBOL_TO_BACKTEST,
            target_df=target_data_raw,
            spy_df=spy_data_processed,
            expected_features=expected_features,
            scaler=scaler if SCALER_USED_IN_TRAINING else None # Pass scaler only if used
        )

    except FileNotFoundError as e: print(e); exit()
    except ValueError as e: print(f"Data Preparation Error: {e}"); exit()
    except Exception as e: print(f"Unexpected error during data loading/preparation: {e}"); traceback.print_exc(); exit()

    if data_final_prepared.empty:
        print(f"{datetime.now()} - ERROR: Final prepared data is empty. Cannot run backtest."); exit()

    # --- Run Backtest ---
    equity_curve, trades = run_backtest(
        data_prepared=data_final_prepared,
        model=model,
        expected_features=expected_features,
        capital=INITIAL_CAPITAL,
        trade_shares=TRADE_SHARES,
        sl_mult=SL_MULT,
        tp_mult=TP_MULT,
        commission_per_share=COMMISSION_PER_SHARE,
        slippage_per_share=SLIPPAGE_PER_SHARE,
        trade_long=TRADE_ON_SIGNAL_1,
        trade_short=TRADE_ON_SIGNAL_0,
        probability_threshold=PROBABILITY_THRESHOLD # --- Pass the new parameter ---
    )

    # --- Calculate & Print Metrics ---
    print(f"\n{datetime.now()} - Calculating final performance metrics...")
    metrics = calculate_metrics(equity_curve, trades, INITIAL_CAPITAL)
    print("\n--- Backtest Results (Including Costs) ---")
    cost_details = f"(Comm: ${COMMISSION_PER_SHARE}/sh, Slip: ${SLIPPAGE_PER_SHARE}/sh)"
    print(cost_details)
    for key, value in metrics.items():
        if isinstance(value, (float, np.number)): print(f"{key}: {value:,.2f}")
        else: print(f"{key}: {value}")
    print("------------------------------------------")

    # --- Save Log & Plot ---
    if trades:
        try:
            trade_df_log = pd.DataFrame(trades)
            safe_cost_details = cost_details.replace('/','-').replace(':','').replace('(','').replace(')','').replace(',','').replace('$','')
            # Include model timestamp in log filename
            trade_log_filename = f"{SYMBOL_TO_BACKTEST}_backtest_trades_{timestamp_str_from_training}{smote_tag_in_filename}_{safe_cost_details}.csv"
            trade_df_log.to_csv(trade_log_filename)
            print(f"\nTrade log saved to {trade_log_filename}")
        except Exception as log_e: print(f"Error saving trade log: {log_e}")
    else:
        print("\nNo trades were executed during the backtest.")

    print(f"\n{datetime.now()} - Plotting final equity curve...")
    if not equity_curve.empty:
        plot_equity_curve(equity_curve, SYMBOL_TO_BACKTEST, cost_info=cost_details)
    else: print("Cannot plot equity curve as equity data is empty.")

    print(f"\n{datetime.now()} - --- Script Finished ---")