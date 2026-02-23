"""
explain.py
Generates XAI artefacts using SHAP and Partial Dependence Plots (PDP).
  - SHAP summary beeswarm plot
  - SHAP bar plot (mean |SHAP|)
  - SHAP waterfall for a single prediction
  - PDP for month (seasonal effect)
  - PDP for inflation_index (crisis effect)
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

MODEL_PATH  = "models/lgbm_model.pkl"
FEAT_PATH   = "models/feature_cols.pkl"
TEST_PATH   = "data/processed/test_set.csv"
SHAP_PATH   = "models/shap_explainer.pkl"


def load_artifacts():
    model        = joblib.load(MODEL_PATH)
    feat_cols    = joblib.load(FEAT_PATH)
    test_df      = pd.read_csv(TEST_PATH, parse_dates=["date"])
    return model, feat_cols, test_df


def run_shap(model, X_sample, feat_cols):
    print("  Computing SHAP values (this may take ~30s)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_sample, feat_cols):
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feat_cols,
                      show=False, plot_size=None)
    plt.title("SHAP Feature Impact (beeswarm)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("assets/shap_summary.png", dpi=150)
    plt.close()
    print("  SHAP summary → assets/shap_summary.png")


def plot_shap_bar(shap_values, feat_cols):
    mean_shap = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feat_cols, "shap": mean_shap})
    imp_df = imp_df.sort_values("shap", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.35, len(imp_df)))
    ax.barh(imp_df["feature"][::-1], imp_df["shap"][::-1], color=colors)
    ax.set_xlabel("Mean |SHAP value| (LKR impact)")
    ax.set_title("Top 12 Features by Mean |SHAP|", fontsize=13)
    plt.tight_layout()
    plt.savefig("assets/shap_bar.png", dpi=150)
    plt.close()
    print("  SHAP bar → assets/shap_bar.png")


def plot_pdp(model, X_df, feat_cols, feature_name, n_points=30):
    """Manual PDP: vary one feature, average predictions."""
    X = X_df.values.copy()
    feat_idx = feat_cols.index(feature_name)
    vals = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), n_points)
    means, stds = [], []
    for v in vals:
        X_mod = X.copy()
        X_mod[:, feat_idx] = v
        preds = model.predict(X_mod)
        means.append(preds.mean())
        stds.append(preds.std())
    means, stds = np.array(means), np.array(stds)
    return vals, means, stds


def plot_pdps(model, X_sample_df, feat_cols):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (feat, label, color) in zip(axes, [
        ("month",            "Month of Year",     "#6366f1"),
        ("inflation_index",  "Inflation Index",   "#f59e0b"),
    ]):
        vals, means, stds = plot_pdp(model, X_sample_df, feat_cols, feat)
        ax.plot(vals, means, color=color, lw=2)
        ax.fill_between(vals, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_xlabel(label); ax.set_ylabel("Avg Predicted Price (LKR)")
        ax.set_title(f"PDP: {label}")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs {x:,.0f}"))
    plt.tight_layout()
    plt.savefig("assets/pdp.png", dpi=150)
    plt.close()
    print("  PDP plots → assets/pdp.png")


def main():
    os.makedirs("assets", exist_ok=True)
    print("Loading model and test data...")
    model, feat_cols, test_df = load_artifacts()

    X_sample_df = test_df[feat_cols].sample(500, random_state=42)
    X_sample    = X_sample_df.values

    explainer, shap_values = run_shap(model, X_sample, feat_cols)

    plot_shap_summary(shap_values, X_sample, feat_cols)
    plot_shap_bar(shap_values, feat_cols)
    plot_pdps(model, X_sample_df, feat_cols)

    # Save explainer for UI use
    joblib.dump(explainer, SHAP_PATH)
    print(f"  SHAP explainer saved → {SHAP_PATH}")
    print("\nAll explainability assets saved to assets/")


if __name__ == "__main__":
    main()
