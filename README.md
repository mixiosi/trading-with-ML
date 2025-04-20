# ML Trading Signal Outcome Prediction

## Overview

This project attempts to use machine learning (XGBoost and Logistic Regression) to predict the outcome of technical trading signals derived from 5-minute historical price data for multiple stock symbols (and SPY as an index).

The initial goal was to predict general significant price direction, but this yielded poor results (~50-51% accuracy), indicating little predictive power over random chance.

## Current Approach & Status

The project has pivoted to predicting the **success or failure of specific entry signals**, framed as a Take Profit (TP) vs. Stop Loss (SL) outcome prediction.

*   **Labeling:** Signals are generated based on technical rules. Each signal is labeled '1' if a predefined ATR-based TP target is hit before the SL target within a set lookahead period (currently 75 mins / 15 bars), and '0' otherwise.
*   **Current Strategy Focus:** The latest runs focus specifically on **Bollinger Band Mean Reversion signals** (price crossing outside the bands), filtered by time-of-day and potentially other conditions based on SHAP analysis.
*   **Features:** The model uses a combination of:
    *   Standard technical indicators (SMA, RSI, MACD, Stochastics, ATR, BBands, ROC, lags).
    *   Cyclical time features (hour, day of week).
    *   Market time features (minutes since open, minutes to close).
    *   Relative strength features (stock indicators vs. SPY indicators).
    *   One-Hot Encoded symbol and signal type (when multiple signals were used).
*   **Modeling:** XGBoost (tuned with RandomizedSearchCV, often optimizing for `precision`) and Logistic Regression (tuned with GridSearchCV) are trained and compared.

## Key Findings & Current Results (Filtered BB Signal Run)

*   **Modest Accuracy:** The latest model (focused on filtered BB signals) achieved ~55% test set accuracy, indicating some predictive signal slightly better than random chance for this specific strategy.
*   **Low TP Precision:** The primary challenge remains: **Precision for predicting winning trades (TP Hits / Class 1) is low (~41-46%)**. This means the model is still incorrect more often than not when predicting a signal will be successful, making it likely unprofitable.
*   **Decent SL Prediction:** Both models generally show better precision/recall for predicting losing trades (SL Hits / Class 0).
*   **Time Dominance:** SHAP analysis consistently shows that **time-based features** (minutes since open, hour sin/cos, minutes to close, day of week sin/cos) have the **highest average impact** on the model's predictions for TP/SL outcomes. Momentum indicators (ROC, Stoch lags, RSI) are secondary.
*   **Filtering Impact:** Filtering signals based on SHAP insights (e.g., avoiding the market open, requiring specific relative strength) has **not yet significantly improved precision** for winning trades, though it impacts the dataset size and feature importance rankings.

## File Descriptions

1.  **`train_model_revised.py` (Main Training Script)**
    *   **Purpose:** Loads data, preprocesses, generates signals & labels, trains models, evaluates, and saves artifacts.
    *   **Workflow:**
        *   Loads 5-min data for `INDEX_SYMBOL` (SPY) and calculates a full set of features.
        *   Loops through `SYMBOLS`:
            *   Loads 5-min data.
            *   Cleans data.
            *   Calculates features (stock-specific, relative-to-index, time-based).
            *   Generates entry signals (currently configured for filtered BB Reversions).
            *   Labels signals based on TP/SL outcome (`label_signal_outcome_tp_sl`).
            *   Selects data rows corresponding to labeled signals.
        *   Combines signal data from all symbols.
        *   Sorts data chronologically.
        *   Performs One-Hot Encoding on `symbol` and `signal_type`.
        *   Splits data into Training and Testing sets chronologically.
        *   **Saves the unscaled test set (`X_test_unscaled.pkl`, `y_test.pkl`)**.
        *   Applies `StandardScaler` to numeric features in training/testing sets.
        *   Trains XGBoost using `RandomizedSearchCV` (currently optimizing for `precision`).
        *   Trains Logistic Regression using `GridSearchCV`.
        *   Evaluates both models on the test set (Accuracy, Classification Report, Confusion Matrix).
        *   Saves the best models, the scaler, and the list of final features used.

2.  **`analyze_results.py` (Post-Training Analysis Script)**
    *   **Purpose:** Loads artifacts from a training run and performs deeper analysis on the best XGBoost model.
    *   **Workflow:**
        *   Loads the saved XGBoost model (`.joblib`), feature scaler (`.joblib`), and feature list (`.list`).
        *   Loads the saved unscaled test data (`X_test_unscaled.pkl`, `y_test.pkl`).
        *   Checks consistency between the feature list and loaded test data columns, reordering test data columns if necessary.
        *   Applies the loaded scaler to the test data (using the scaler's known features).
        *   Performs **SHAP Analysis** on a sample of the unscaled test data:
            *   Calculates SHAP values.
            *   Generates and saves SHAP dependence plots for top features to visualize feature impacts.
        *   Performs **Probability Threshold Analysis**:
            *   Predicts probabilities on the scaled test data using the loaded XGBoost model.
            *   Calculates Precision, Recall, F1-score for Class 1 (TP Hits) at various probability thresholds (0.50, 0.55, ...).
            *   Prints a table showing the trade-off between confidence (threshold) and performance metrics.

## How to Run

1.  Ensure 5-minute data `.csv` files (named according to `FILENAME_TEMPLATE`) exist in the `DATA_DIR` for all symbols in `SYMBOLS` and for the `INDEX_SYMBOL`.
2.  Install required libraries (e.g., `pandas`, `numpy`, `scikit-learn`, `xgboost`, `talib`, `joblib`, `shap`, `matplotlib`). Consider using a `requirements.txt` file.
3.  Run the main training script: `python train_model_revised.py`
    *   This will generate model artifacts (`.joblib`), feature list (`.list`), scaler (`.joblib`), and test data (`.pkl`) files.
4.  Run the analysis script: `python analyze_results.py`
    *   This will load the artifacts and test data generated in step 3 and produce SHAP plots and the threshold analysis table.

## Next Steps / Future Work

*   Analyze the latest SHAP dependence plots (especially time and momentum features for the filtered BB signals).
*   Implement more targeted filters in `generate_entry_signals` based on SHAP insights (e.g., specific time windows combined with momentum conditions).
*   Test isolating other signal types (e.g., MACD crossovers) instead of Bollinger Bands.
*   Experiment further with TP/SL ratios (e.g., 1:1).
*   Explore more advanced feature engineering (interaction terms, different volatility/trend measures).
*   Consider alternative modeling approaches if current methods fail to yield sufficient precision.
