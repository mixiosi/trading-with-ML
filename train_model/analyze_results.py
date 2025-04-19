# analyze_results.py
import pandas as pd
import numpy as np
import joblib
import shap
import os
import traceback
from datetime import datetime
# --- ADDED: Ensure consistent imports ---
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler # Needed if re-scaling
# --- END ADDED ---

# Optional: Install matplotlib if you want to save SHAP plots
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

# --- Configuration (Should match the training script used for the artifacts) ---
SYMBOLS = ['AAPL', 'AMD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'QQQ', 'SPY', 'TSLA']
INDEX_SYMBOL = 'SPY'
DATA_DIR = '.'
FILENAME_TEMPLATE = '{}_5min_historical_data.csv'
TEST_SIZE = 0.2 # Must match the split used during training
RANDOM_STATE = 42 # For consistent sampling if needed

# --- Artifact Filenames (MODIFIED to match LATEST run - V2 Precision Tuned) ---
# Ensure these filenames match the output of your LATEST training run
MODEL_FILENAME = 'revamped_xgb_model_tp_sl_v2_prec_tuned.joblib'
SCALER_FILENAME = 'revamped_feature_scaler_tp_sl_v2.joblib'
FEATURE_LIST_FILENAME = 'revamped_model_features_tp_sl_v2.list'
# LOGREG_MODEL_FILENAME = 'revamped_logreg_model_tp_sl_v2.joblib' # Optional: If you want to load LogReg too

# --- Analysis Configuration ---
SHAP_SAMPLE_SIZE = 1000 # Number of test samples for SHAP (reduce if too slow)
PROBABILITY_THRESHOLDS = np.arange(0.5, 0.81, 0.05) # Thresholds to test (0.5, 0.55, ..., 0.8)
TOP_N_SHAP_FEATURES = 15 # Number of top features for dependence plots

# --- Load Features ---
print(f"{datetime.now()} - Loading feature list from: {FEATURE_LIST_FILENAME}")
expected_feature_columns = None # Initialize
try:
    with open(FEATURE_LIST_FILENAME, 'r') as f:
        expected_feature_columns = [line.strip() for line in f if line.strip()]
    if not expected_feature_columns:
         print(f"[ERROR] Feature list file '{FEATURE_LIST_FILENAME}' is empty or invalid.")
         exit()
    print(f"Loaded {len(expected_feature_columns)} expected features from list file.")
except FileNotFoundError:
    print(f"[ERROR] Feature list file not found: {FEATURE_LIST_FILENAME}")
    exit()
except Exception as e:
    print(f"[ERROR] Failed to load feature list: {e}"); exit()

# --- Function to reload and prepare test data ---
def get_test_data():
    """Loads previously saved test data."""
    print("Attempting to load saved test data (X_test_unscaled.pkl, y_test.pkl)...")
    try:
        X_test_unscaled = pd.read_pickle("X_test_unscaled.pkl")
        y_test = pd.read_pickle("y_test.pkl")
        print("Loaded saved test data.")
        return X_test_unscaled, y_test
    except FileNotFoundError:
        print("[ERROR] Cannot find saved test data (X_test_unscaled.pkl/y_test.pkl).")
        print("Please ensure you ran the corresponding training script that saves these files.")
        return None, None
    except Exception as e:
        print(f"[ERROR] Error loading saved test data: {e}")
        return None, None

if __name__ == "__main__":
    print(f"\n{datetime.now()} - --- Starting Analysis ---")

    # --- Load Model and Scaler ---
    print(f"{datetime.now()} - Loading model from: {MODEL_FILENAME}")
    try:
        best_xgb_model = joblib.load(MODEL_FILENAME)
    except Exception as e: print(f"[ERROR] Failed to load model: {e}"); exit()

    print(f"{datetime.now()} - Loading scaler from: {SCALER_FILENAME}")
    try:
        scaler = joblib.load(SCALER_FILENAME)
    except FileNotFoundError: print("[WARN] Scaler file not found."); scaler = None
    except Exception as e: print(f"[ERROR] Failed to load scaler: {e}"); exit()

    # --- Load Test Data ---
    print(f"{datetime.now()} - Loading test data...")
    X_test_unscaled, y_test = get_test_data()
    if X_test_unscaled is None or y_test is None:
        print(f"{datetime.now()} - Failed to get test data. Exiting analysis."); exit()
    if len(X_test_unscaled) == 0:
        print(f"{datetime.now()} - Test data is empty. Exiting analysis."); exit()

    print(f"Loaded test data: X_test shape {X_test_unscaled.shape}, y_test length {len(y_test)}")

    # --- Feature Consistency Check and Reordering ---
    actual_test_columns = list(X_test_unscaled.columns)
    final_feature_columns_ordered = expected_feature_columns # Assume list file order is correct

    if set(actual_test_columns) != set(final_feature_columns_ordered):
        print("\n[ERROR] FATAL: Feature set mismatch between loaded test data and feature list file!")
        missing_in_test = sorted(list(set(final_feature_columns_ordered) - set(actual_test_columns)))
        extra_in_test = sorted(list(set(actual_test_columns) - set(final_feature_columns_ordered)))
        print(f"  Features in list file but NOT in test data: {missing_in_test}")
        print(f"  Features in test data but NOT in list file: {extra_in_test}")
        print("Cannot proceed with analysis due to feature mismatch. Ensure artifacts match.")
        exit()
    else:
        # --- >>> FIX: Reorder loaded X_test columns <<< ---
        print("Feature set matches. Reordering test data columns to match feature list...")
        try:
            X_test_unscaled = X_test_unscaled[final_feature_columns_ordered]
            print("Test data columns reordered successfully.")
        except KeyError as e_reorder:
             print(f"[ERROR] Failed to reorder X_test columns: {e_reorder}. Mismatch likely exists.")
             exit()
        # --- >>> END FIX <<< ---


    # --- Scale Test Data (if scaler loaded) ---
    X_test_scaled = X_test_unscaled.copy() # Start with correctly ordered unscaled data
    if scaler:
        print(f"{datetime.now()} - Scaling test data...")
        try:
             # Use scaler's known features if available (preferred)
             if hasattr(scaler, 'n_features_in_') and hasattr(scaler, 'feature_names_in_'):
                  scaled_cols = scaler.feature_names_in_
                  # Check if the scaler's expected features match the *beginning* of our final feature list
                  # (assuming OHE columns are at the end and were not scaled)
                  num_scaled_expected = scaler.n_features_in_
                  cols_to_transform_from_scaler = list(scaled_cols)

                  # Verify these columns exist in the *correctly ordered* X_test
                  if all(c in X_test_scaled.columns for c in cols_to_transform_from_scaler):
                      print(f"Applying scaling to {len(cols_to_transform_from_scaler)} columns based on scaler info.")
                      X_test_scaled.loc[:, cols_to_transform_from_scaler] = scaler.transform(X_test_scaled[cols_to_transform_from_scaler])
                      print("Scaling applied using scaler's feature names.")
                  else:
                       print("[ERROR] Scaler feature names mismatch with test data columns even after reordering.")
                       missing_for_scaler = [c for c in cols_to_transform_from_scaler if c not in X_test_scaled.columns]
                       print(f"  Missing columns required by scaler: {missing_for_scaler}")
                       print("Cannot apply scaler reliably. Predictions might be inaccurate.")
                       # Decide whether to exit or proceed with unscaled
                       # Proceeding with unscaled for now, but analysis might be wrong
                       X_test_scaled = X_test_unscaled.copy()

             else:
                  # Fallback: Infer numeric columns (less reliable)
                  print("[WARN] Scaler lacks 'feature_names_in_'. Inferring numeric cols for scaling.")
                  numeric_cols_infer = [
                      col for col in final_feature_columns_ordered if # Use ordered list
                      not col.startswith('sym_') and not col.startswith('sig_') and
                      pd.api.types.is_numeric_dtype(X_test_scaled[col])
                  ]
                  cols_to_transform_infer = [c for c in numeric_cols_infer if c in X_test_scaled.columns]
                  if cols_to_transform_infer:
                    print(f"Applying scaling to {len(cols_to_transform_infer)} inferred numeric columns.")
                    X_test_scaled.loc[:, cols_to_transform_infer] = scaler.transform(X_test_scaled[cols_to_transform_infer])
                    print("Scaling applied using inferred columns.")
                  else:
                     print("[WARN] No inferred numeric columns found for scaling application.")

        except Exception as e:
             print(f"[ERROR] Error applying scaler: {e}. Predictions might be inaccurate.")
             X_test_scaled = X_test_unscaled.copy() # Revert
    else:
         print("[INFO] No scaler loaded. Using unscaled data for predictions.")


    # --- SHAP Analysis ---
    print(f"\n{datetime.now()} - --- SHAP Analysis ---")
    # Use X_test_unscaled (correctly ordered) for SHAP interpretation
    X_test_for_shap = X_test_unscaled

    try:
        print(f"Calculating SHAP values (using sample size: {SHAP_SAMPLE_SIZE})...")
        explainer = shap.TreeExplainer(best_xgb_model)
        actual_shap_sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_for_shap))

        if actual_shap_sample_size > 0:
            X_test_shap_sample = X_test_for_shap.sample(actual_shap_sample_size, random_state=RANDOM_STATE)
            try: shap_values = explainer.shap_values(X_test_shap_sample)
            except TypeError: print("[WARN SHAP] Retrying SHAP with .values"); shap_values = explainer.shap_values(X_test_shap_sample.values)

            if isinstance(shap_values, list) and len(shap_values) > 1: shap_values_for_plot = shap_values[1]
            else: shap_values_for_plot = shap_values

            # --- Generate Dependence Plots for Top N Features ---
            print(f"\nGenerating SHAP dependence plots for top {TOP_N_SHAP_FEATURES} features...")
            vals = np.abs(shap_values_for_plot).mean(0)
            # Ensure mapping columns correctly if X_test_shap_sample was used
            feature_importance = pd.DataFrame(list(zip(X_test_shap_sample.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            top_features = feature_importance['col_name'].head(TOP_N_SHAP_FEATURES).tolist()

            if not os.path.exists("shap_dependence_plots"): os.makedirs("shap_dependence_plots")

            for feature in top_features:
                try:
                    shap.dependence_plot(feature, shap_values_for_plot, X_test_shap_sample, interaction_index="auto", show=False)
                    if MATPLOTLIB_INSTALLED:
                        plt.title(f"SHAP Dependence Plot: {feature}")
                        filename = f"shap_dependence_plots/dependence_{feature.replace('/', '_').replace(':', '_')}.png"
                        plt.savefig(filename, bbox_inches='tight'); plt.close()
                        print(f"  - Saved dependence plot for: {feature}")
                    else: print(f"  - Processed dependence for: {feature} (matplotlib not installed)")
                except Exception as e_dep: print(f"[ERROR] Failed dependence plot for {feature}: {e_dep}")
        else: print("[WARN] Test set empty or too small for SHAP analysis.")
    except ImportError: print("[WARN] SHAP library not installed (`pip install shap`). Skipping SHAP analysis.")
    except Exception as e_shap: print(f"[ERROR] Error during SHAP analysis: {e_shap}"); traceback.print_exc()

    # --- Probability Threshold Analysis ---
    print(f"\n{datetime.now()} - --- Probability Threshold Analysis ---")
    try:
        print("Calculating predicted probabilities using SCALED data...")
        # Use X_test_scaled for prediction, ensure columns match final_feature_columns_ordered
        # This reordering might be redundant if X_test_scaled was created from reordered X_test_unscaled
        # but ensures safety if scaling failed.
        X_test_scaled_ordered = X_test_scaled[final_feature_columns_ordered]
        y_prob_xgb = best_xgb_model.predict_proba(X_test_scaled_ordered)[:, 1]

        results = []
        print("Threshold | Precision | Recall    | F1-Score  | TP    | FP    | FN     | Support (Pred 1)")
        print("----------|-----------|-----------|-----------|-------|-------|-------|-----------------")
        for threshold in PROBABILITY_THRESHOLDS:
            y_pred_thresh = (y_prob_xgb >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_thresh, labels=[1], average='binary', zero_division=0)
            # Handle potential (1,1) confusion matrix if threshold is too high/low
            cm = confusion_matrix(y_test, y_pred_thresh)
            if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1): # Only one class predicted/present
                 if y_test.iloc[0] == 0 and y_pred_thresh[0] == 0: tn, fp, fn, tp = cm[0,0],0,0,0 # All TN
                 elif y_test.iloc[0] == 1 and y_pred_thresh[0] == 1: tn, fp, fn, tp = 0,0,0,cm[0,0] # All TP
                 elif y_test.iloc[0] == 0 and y_pred_thresh[0] == 1: tn, fp, fn, tp = 0,cm[0,0],0,0 # All FP
                 else: tn, fp, fn, tp = 0,0,cm[0,0],0 # All FN
            else: tn, fp, fn, tp = 0,0,0,0 # Default / Error case


            support_pred_1 = tp + fp
            print(f"  {threshold:.2f}    |   {precision:.3f}   |   {recall:.3f}   |   {f1:.3f}   | {tp: 5d} | {fp: 5d} | {fn: 5d} | {support_pred_1: 15d}")
            results.append({'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'support_pred_1': support_pred_1})

        threshold_df = pd.DataFrame(results)
        # print("\nThreshold Analysis Summary:")
        # print(threshold_df.round(3)) # Optional: print full df

    except Exception as e_thresh:
        print(f"[ERROR] Error during threshold analysis: {e_thresh}"); traceback.print_exc()

    print(f"\n{datetime.now()} - --- Analysis Finished ---")