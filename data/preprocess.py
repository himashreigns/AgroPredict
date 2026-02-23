"""
preprocess.py
Loads raw CSV and engineers features for LightGBM training:
  - Lag features: 1-week, 4-week, 52-week
  - Rolling statistics: 4-week rolling mean and std
  - Label encoding for categorical columns
  - Saves processed CSV + encoder dict
"""

import pandas as pd
import numpy as np
import os
import joblib

RAW_PATH  = "data/raw/srilanka_produce_prices.csv"
PROC_PATH = "data/processed/processed_prices.csv"
ENC_PATH  = "models/label_encoders.pkl"

CAT_COLS = ["commodity", "category", "market", "price_type", "province", "season"]


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["commodity", "market", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_lag_features(df):
    """Add lag and rolling features per commodity-market group."""
    print("  Adding lag & rolling features...")

    def group_lags(g):
        g = g.sort_values("date").copy()
        p = g["price_lkr"]
        g["price_lag_1w"]     = p.shift(1)
        g["price_lag_4w"]     = p.shift(4)
        g["price_lag_52w"]    = p.shift(52)
        g["rolling_mean_4w"]  = p.shift(1).rolling(4).mean()
        g["rolling_std_4w"]   = p.shift(1).rolling(4).std()
        g["rolling_mean_12w"] = p.shift(1).rolling(12).mean()
        g["price_change_1w"]  = p - p.shift(1)
        g["price_change_4w"]  = p - p.shift(4)
        return g

    df = df.groupby(["commodity", "market"], group_keys=False).apply(group_lags)
    df.dropna(subset=["price_lag_52w"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def encode_categoricals(df):
    """Label-encode categorical columns; save encoders for inference."""
    print("  Encoding categoricals...")
    encoders = {}
    for col in CAT_COLS:
        unique_vals = sorted(df[col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        df[col + "_enc"] = df[col].map(mapping)
        encoders[col] = mapping
    return df, encoders


def add_sinusoidal_time(df):
    """Encode cyclical month/week features as sin/cos."""
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]  = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["week"] / 52)
    return df


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading raw data...")
    df = load_raw()
    print(f"  Shape: {df.shape}")

    df = add_lag_features(df)
    df = add_sinusoidal_time(df)
    df, encoders = encode_categoricals(df)

    # Fill any remaining NaN in rolling cols with column median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    print(f"  Final shape after feature engineering: {df.shape}")
    df.to_csv(PROC_PATH, index=False)
    print(f"  Saved processed data → {PROC_PATH}")

    joblib.dump(encoders, ENC_PATH)
    print(f"  Saved encoders → {ENC_PATH}")
    return df, encoders


if __name__ == "__main__":
    main()
