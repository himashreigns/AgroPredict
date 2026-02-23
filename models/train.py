"""
train.py
Trains a LightGBM regressor on the engineered Sri Lanka produce price dataset.
  - Time-based 70/15/15 train/val/test split
  - Early stopping on validation RMSE
  - Evaluates: RMSE, MAE, R², MAPE
  - Saves model + SHAP explainer + metrics JSON
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROC_PATH    = "data/processed/processed_prices.csv"
MODEL_PATH   = "models/lgbm_model.pkl"
METRICS_PATH = "models/metrics.json"
SHAP_PATH    = "models/shap_explainer.pkl"

# Feature columns used for training
FEATURE_COLS = [
    "year", "month", "week", "quarter", "day_of_year",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "commodity_enc", "category_enc", "market_enc",
    "price_type_enc", "province_enc", "season_enc",
    "monsoon", "is_festive", "inflation_index", "seasonal_factor",
    "price_lag_1w", "price_lag_4w", "price_lag_52w",
    "rolling_mean_4w", "rolling_std_4w", "rolling_mean_12w",
    "price_change_1w", "price_change_4w",
]
TARGET_COL = "price_lkr"


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def time_split(df, train_frac=0.70, val_frac=0.15):
    """Time-based split preserving chronological order."""
    dates = sorted(df["date"].unique())
    n = len(dates)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train_dates = dates[:t1]
    val_dates   = dates[t1:t2]
    test_dates  = dates[t2:]
    train = df[df["date"].isin(train_dates)]
    val   = df[df["date"].isin(val_dates)]
    test  = df[df["date"].isin(test_dates)]
    return train, val, test


def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv(PROC_PATH, parse_dates=["date"])
    print(f"  Shape: {df.shape}")

    train_df, val_df, test_df = time_split(df)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[TARGET_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].values

    # ── LightGBM ──────────────────────────────────────────────────────────
    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "num_leaves":       64,
        "learning_rate":    0.05,
        "n_estimators":     1000,
        "min_child_samples": 20,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    mape_ = mape(y_test, preds)

    metrics = {
        "RMSE":  round(rmse, 4),
        "MAE":   round(mae, 4),
        "R2":    round(r2, 4),
        "MAPE":  round(mape_, 4),
        "best_iteration": model.best_iteration_,
        "n_features": len(FEATURE_COLS),
        "train_size": len(train_df),
        "val_size":   len(val_df),
        "test_size":  len(test_df),
    }
    print("\n── Test Set Metrics ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    # ── Residual Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test[:500], preds[:500], alpha=0.4, s=10, color="#2563eb")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel("Actual Price (LKR)"); axes[0].set_ylabel("Predicted Price (LKR)")
    axes[0].set_title("Actual vs Predicted")
    residuals = y_test - preds
    axes[1].hist(residuals, bins=50, color="#10b981", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Residual (LKR)"); axes[1].set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig("assets/residuals.png", dpi=150)
    plt.close()
    print("  Residual plot → assets/residuals.png")

    # ── Feature Importance Plot ───────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1],
                   color="#6366f1")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top 15 Feature Importances — LightGBM")
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png", dpi=150)
    plt.close()
    print("  Feature importance → assets/feature_importance.png")

    # ── Save ──────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {METRICS_PATH}")

    # Save test data for explainer
    test_df.to_csv("data/processed/test_set.csv", index=False)
    # Save feature names
    joblib.dump(FEATURE_COLS, "models/feature_cols.pkl")

    return model, metrics


if __name__ == "__main__":
    train()
