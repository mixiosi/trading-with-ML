import pandas as pd
import numpy as np
import talib
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback

# --- Configuration ---
SYMBOL = 'NVDA'  # Symbol to backtest
DATA_DIR = '.'
FILENAME_TEMPLATE = '{}_5min_historical_data.csv'
MODEL_FILE = 'trading_xgb_model_multisym_sigmove_tuned_v2.joblib'
FEATURE_LIST_FILE = 'trading_model_features_multisym_sigmove_v2.list'

# Backtesting Parameters
INITIAL_CAPITAL = 30000.0 # Match the value from the last successful run log
TRADE_SHARES = 10

# Cost Parameters
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.02

# Strategy Parameters
SMA_WINDOW = 20; ATR_PERIOD = 14; RSI_PERIOD = 14
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
SL_MULT = 2.0; TP_MULT = 4.0
VOL_Z_LONG_THRESH = 0.05; VOL_Z_SHORT_THRESH = -0.05

# --- Feature Calculation Function (Identical to trainer) ---
def calculate_features(df):
    """Calculate technical indicators and features for the dataset."""
    cols_to_check = ['high', 'low', 'close']
    for col in cols_to_check:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=cols_to_check, inplace=True)
    if df.empty: return df

    df['sma'] = df['close'].rolling(window=SMA_WINDOW).mean()
    df['std'] = df['close'].rolling(window=SMA_WINDOW).std()
    df['upper_band'] = df['sma'] + 2 * df['std']
    df['lower_band'] = df['sma'] - 2 * df['std']

    high_prices = df['high'].values; low_prices = df['low'].values; close_prices = df['close'].values

    if len(high_prices) == 0 or len(low_prices) == 0 or len(close_prices) == 0:
         df['atr'] = np.nan; df['rsi'] = np.nan; df['macd'] = np.nan
         df['macd_signal'] = np.nan; df['macd_hist'] = np.nan
    else:
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
        df['rsi'] = talib.RSI(close_prices, timeperiod=RSI_PERIOD)
        macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
        df['macd'] = macd; df['macd_signal'] = macdsignal; df['macd_hist'] = macdhist

    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    df['volume_z'] = (df['volume'] - df['volume_sma5']) / df['volume_sma5'].replace(0, 1e-10)
    df['rsi_lag1'] = df['rsi'].shift(1)

    df.dropna(inplace=True)
    return df

# --- Backtesting Logic with Costs (Identical to previous version) ---
def run_backtest(data, model, expected_features, capital, trade_shares,
                 sl_mult, tp_mult, vol_z_long, vol_z_short,
                 commission_per_share, slippage_per_share):
    """Runs the backtest loop, including commission and slippage."""
    cash = capital; position = 0; entry_price = 0.0
    stop_loss = 0.0; take_profit = 0.0
    equity = []; trades = []; entry_timestamp = pd.NaT

    if not isinstance(data.index, pd.DatetimeIndex): raise TypeError("Index must be DatetimeIndex.")

    print(f"\n--- Running Backtest Simulation ---"); print(f"Initial Capital: ${capital:,.2f}")
    print(f"Trade Size: {trade_shares} shares"); print(f"Commission per Share: ${commission_per_share:.4f}")
    print(f"Slippage per Share: ${slippage_per_share:.4f}"); print(f"SL/TP Multipliers: {sl_mult}/{tp_mult}")
    print(f"Iterating through {len(data)} bars...")

    for timestamp, row in data.iterrows():
        current_price = row['close']; current_high = row['high']
        current_low = row['low']; current_atr = row['atr']

        # Exit Check
        exit_target_price = None; exit_reason = None
        if position > 0:
            if current_low <= stop_loss: exit_target_price = stop_loss; exit_reason = "Stop Loss"
            elif current_high >= take_profit: exit_target_price = take_profit; exit_reason = "Take Profit"
        elif position < 0:
            if current_high >= stop_loss: exit_target_price = stop_loss; exit_reason = "Stop Loss"
            elif current_low <= take_profit: exit_target_price = take_profit; exit_reason = "Take Profit"

        # Process Exit
        if exit_target_price is not None:
            commission_cost = commission_per_share * abs(position); profit_net = 0.0
            if position > 0:
                effective_exit_price = exit_target_price - slippage_per_share
                profit_gross = (effective_exit_price - entry_price) * position
                profit_net = profit_gross - commission_cost
                cash += effective_exit_price * position - commission_cost
            elif position < 0:
                effective_exit_price = exit_target_price + slippage_per_share
                profit_gross = (entry_price - effective_exit_price) * abs(position)
                profit_net = profit_gross - commission_cost
                cash += entry_price * abs(position) + profit_net
            if pd.notna(entry_timestamp):
                trades.append({'entry_time': entry_timestamp, 'exit_time': timestamp, 'entry_price': entry_price,
                               'exit_price': effective_exit_price, 'position_type': 'long' if position > 0 else 'short',
                               'shares': abs(position), 'profit': profit_net, 'exit_reason': exit_reason})
            else: print(f"Warning: Exit at {timestamp} without valid entry_timestamp.")
            position = 0; entry_price = 0.0; stop_loss = 0.0; take_profit = 0.0; entry_timestamp = pd.NaT

        # Entry Check
        if position == 0:
            signal_type = None
            if not pd.isna(row['lower_band']) and row['close'] < row['lower_band'] and not pd.isna(row['volume_z']) and row['volume_z'] > vol_z_long: signal_type = 'long'
            elif not pd.isna(row['upper_band']) and row['close'] > row['upper_band'] and not pd.isna(row['volume_z']) and row['volume_z'] < vol_z_short: signal_type = 'short'
            if signal_type is not None and not pd.isna(current_atr) and current_atr > 1e-9:
                try: features = row[expected_features].values.reshape(1, -1)
                except KeyError as e: print(f"Error: Feature '{e}' missing at {timestamp}."); break
                except Exception as e: print(f"Error preparing features at {timestamp}: {e}"); continue
                if pd.isna(features).any(): continue
                try: prediction = model.predict(features)[0]
                except Exception as e: print(f"Error predicting at {timestamp}: {e}"); continue
                if prediction == 1:
                    commission_cost = commission_per_share * trade_shares
                    if signal_type == 'long':
                        effective_entry_price = current_price + slippage_per_share
                        required_cash = effective_entry_price * trade_shares + commission_cost
                    else:
                        effective_entry_price = current_price - slippage_per_share
                        required_cash = commission_cost
                    if cash >= required_cash:
                        entry_price = effective_entry_price; entry_timestamp = timestamp
                        if signal_type == 'long':
                            position = trade_shares; stop_loss = entry_price - sl_mult * current_atr
                            take_profit = entry_price + tp_mult * current_atr
                            cash -= (entry_price * position + commission_cost)
                        elif signal_type == 'short':
                            position = -trade_shares; stop_loss = entry_price + sl_mult * current_atr
                            take_profit = entry_price - tp_mult * current_atr
                            cash -= commission_cost

        # Update Equity
        current_value = cash
        if position > 0: current_value += position * current_price
        elif position < 0: current_value += (entry_price - current_price) * abs(position)
        equity.append({'Timestamp': timestamp, 'Equity': current_value})

    if not equity: print("Warning: No equity points generated."); return pd.DataFrame(), []
    print(f"\nBacktest finished. Final equity: ${equity[-1]['Equity']:,.2f}")
    equity_df = pd.DataFrame(equity).set_index('Timestamp')
    return equity_df, trades


# --- Performance Metrics Function (Identical) ---
def calculate_metrics(equity_df, trades, initial_capital):
    """Calculates performance metrics."""
    if equity_df.empty: print("Warning: Equity DataFrame empty."); return {"Status": "No data"}
    metrics = {}; final_equity = equity_df['Equity'].iloc[-1]
    metrics['Initial Capital'] = initial_capital; metrics['Final Equity'] = final_equity
    metrics['Total Return (%)'] = ((final_equity / initial_capital) - 1) * 100
    equity_df['Peak'] = equity_df['Equity'].cummax(); equity_df['Drawdown'] = equity_df['Equity'] - equity_df['Peak']
    equity_df['Drawdown (%)'] = (equity_df['Drawdown'] / equity_df['Peak'].replace(0, np.nan)) * 100
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
    try: # Sharpe Ratio
        returns = equity_df['Equity'].pct_change().dropna()
        if len(returns) > 1 and returns.std() != 0:
            time_diff = equity_df.index.to_series().diff().median()
            if time_diff is not pd.NaT:
                 periods_per_day = pd.Timedelta('1 day') / time_diff; trading_periods_per_year = periods_per_day * 252
                 sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(trading_periods_per_year)
                 metrics['Approx Annual Sharpe Ratio'] = sharpe_ratio
            else: metrics['Approx Annual Sharpe Ratio'] = 'Could not determine frequency'
        else: metrics['Approx Annual Sharpe Ratio'] = 0 if len(returns) <=1 else 'Std Dev is zero'
    except Exception as e: print(f"Error calculating Sharpe: {e}"); metrics['Approx Annual Sharpe Ratio'] = 'Error'
    return metrics

def max_consecutive(series):
    max_count = 0; current_count = 0
    for val in series:
        if val: current_count += 1
        else: max_count = max(max_count, current_count); current_count = 0
    return max(max_count, current_count)

# --- Plotting Function (Identical, includes sanitization fix) ---
def plot_equity_curve(equity_df, symbol, cost_info=""):
    """Plots the equity curve."""
    if equity_df.empty or len(equity_df) < 2: print("Not enough data points to plot."); return
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(equity_df.index, equity_df['Equity'], label='Portfolio Value', color='blue')
    max_dd_pct = equity_df['Drawdown (%)'].min()
    max_dd_loc = equity_df['Drawdown (%)'].idxmin() if not equity_df['Drawdown (%)'].isnull().all() else None
    if max_dd_loc: ax.scatter(max_dd_loc, equity_df.loc[max_dd_loc, 'Equity'], color='red', marker='v', s=100, zorder=5, label=f'Max Drawdown ({max_dd_pct:.1f}%)')
    ax.set_title(f'Equity Curve for {symbol} {cost_info}')
    ax.set_xlabel('Date'); ax.set_ylabel('Portfolio Value ($)')
    ax.yaxis.set_major_formatter('${x:,.0f}')
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    plt.xticks(rotation=30, ha='right'); plt.legend(); plt.tight_layout()
    safe_cost_info = cost_info.replace('/','-').replace(':','').replace('(','').replace(')','').replace(',','').replace('$','')
    plot_filename = f"{symbol}_equity_curve_{safe_cost_info}.png"
    plt.savefig(plot_filename); print(f"Equity curve plot saved to {plot_filename}"); plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Backtest ---")
    # Load Model/Features (Error handling included)
    print("Loading model and feature list...")
    if not os.path.exists(MODEL_FILE): print(f"Error: Model file not found: {MODEL_FILE}"); exit()
    if not os.path.exists(FEATURE_LIST_FILE): print(f"Error: Feature list file not found: {FEATURE_LIST_FILE}"); exit()
    try:
        model = joblib.load(MODEL_FILE);
        with open(FEATURE_LIST_FILE, 'r') as f: expected_features = [line.strip() for line in f if line.strip()]
        print(f"Loaded model and {len(expected_features)} expected features.")
    except Exception as e: print(f"Error loading model/features: {e}"); traceback.print_exc(); exit()

    # Load/Prepare Data (Includes date parsing fix)
    print(f"\nLoading data for {SYMBOL}...")
    file_path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(SYMBOL))
    if not os.path.exists(file_path): print(f"Error: Data file not found: {file_path}"); exit()
    try:
        # <<< START DATE PARSING FIX >>>
        correct_date_format = '%m/%d/%Y %H:%M' # Correct format based on screenshot
        try:
             data = pd.read_csv(file_path, parse_dates=['date'], date_format=correct_date_format, index_col='date')
             if not isinstance(data.index, pd.DatetimeIndex): raise ValueError("read_csv parsing failed or didn't produce DatetimeIndex")
             data.sort_index(inplace=True)
        except (ValueError, TypeError, KeyError):
             print("Fallback: Loading CSV then parsing date column using pd.to_datetime.")
             data = pd.read_csv(file_path)
             if 'date' not in data.columns: raise ValueError("Column 'date' not found.")
             data['date'] = pd.to_datetime(data['date'], format=correct_date_format, errors='coerce')
             data.dropna(subset=['date'], inplace=True); data.set_index('date', inplace=True); data.sort_index(inplace=True)
             if not isinstance(data.index, pd.DatetimeIndex): raise TypeError(f"Failed to create DatetimeIndex. Check format ('{correct_date_format}') and data.")
        print(f"Loaded {len(data)} rows. Index type: {type(data.index)}")
        if data.empty: print("Error: Data empty after loading."); exit()
        # <<< END DATE PARSING FIX >>>

        # OHLCV cleaning
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']; print("Cleaning OHLCV columns...")
        initial_rows = len(data)
        for col in ohlcv_cols:
            if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(subset=ohlcv_cols, inplace=True)
        print(f"Dropped {initial_rows - len(data)} rows due to non-numeric OHLCV.")
        if data.empty: print("Error: Data empty after cleaning OHLCV."); exit()
        # Feature calculation
        print("Calculating features...")
        data = calculate_features(data)
        print(f"Data shape after features: {data.shape}")
        if data.empty: print("Error: Data empty after features."); exit()
        # Verify features
        missing_features = [f for f in expected_features if f not in data.columns]
        if missing_features: print(f"Error: Expected features missing: {missing_features}"); exit()
    except ValueError as ve: # Catch specific date parsing value errors
         print(f"\nCRITICAL DATE PARSING ERROR: {ve}."); print(f"Attempted format: '{correct_date_format}'")
         print(f"Check CSV file ({file_path}):"); exit()
    except Exception as e: print(f"Error loading/preparing data: {e}"); traceback.print_exc(); exit()

    # Run Backtest
    equity_curve, trades = run_backtest(
        data, model, expected_features, INITIAL_CAPITAL, TRADE_SHARES,
        SL_MULT, TP_MULT, VOL_Z_LONG_THRESH, VOL_Z_SHORT_THRESH,
        COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE
    )

    # Calculate & Print Metrics
    print("\nCalculating final performance metrics...")
    metrics = calculate_metrics(equity_curve, trades, INITIAL_CAPITAL)
    print("\n--- Backtest Results (Including Costs) ---")
    cost_details = f"(Comm: ${COMMISSION_PER_SHARE}/sh, Slip: ${SLIPPAGE_PER_SHARE}/sh)"
    print(cost_details)
    for key, value in metrics.items():
        if isinstance(value, (float, np.number)): print(f"{key}: {value:,.2f}")
        else: print(f"{key}: {value}")
    print("------------------------------------------")

    # Save Log & Plot (Includes fix for incorrect "No trades" message)
    if trades:
        try:
            trade_df_log = pd.DataFrame(trades)
            safe_cost_details = cost_details.replace('/','-').replace(':','').replace('(','').replace(')','').replace(',','').replace('$','')
            trade_log_filename = f"{SYMBOL}_backtest_trades_{safe_cost_details}.csv"
            trade_df_log.to_csv(trade_log_filename)
            print(f"\nTrade log saved to {trade_log_filename}")
        except Exception as log_e: print(f"Error saving trade log: {log_e}")
    else: # Correctly handles case where trades list is empty
        print("\nNo trades were executed during the backtest.")

    print("\nPlotting final equity curve...")
    if not equity_curve.empty:
        plot_equity_curve(equity_curve, SYMBOL, cost_info=cost_details)
    else: print("Cannot plot equity curve as equity data is empty.") # Handles empty equity case

    print("\n--- Script Finished ---")