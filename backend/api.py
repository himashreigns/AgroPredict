"""
backend/api.py — FastAPI backend for Sri Lanka Produce Price Predictor
Serves data for the React frontend and handles predictions + SHAP explanations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

# ─── Feature Display Names ───────────────────────────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "price_change_1w":  "Weekly Price Change (Rs)",
    "price_lag_1w":     "Last Week's Price (Rs)",
    "rolling_mean_4w":  "4-Week Moving Average (Rs)",
    "price_change_4w":  "Monthly Price Change (Rs)",
    "price_lag_4w":     "Price 4 Weeks Ago (Rs)",
    "rolling_std_4w":   "4-Week Price Volatility (Rs)",
    "price_lag_52w":    "Same Week Last Year Price (Rs)",
    "rolling_mean_12w": "3-Month Moving Average (Rs)",
    "day_of_year":      "Day of Year",
    "week_sin":         "Seasonal Cycle — Week (Sine)",
    "week_cos":         "Seasonal Cycle — Week (Cosine)",
    "week":             "Week Number",
    "month_sin":        "Seasonal Cycle — Month (Sine)",
    "month_cos":        "Seasonal Cycle — Month (Cosine)",
    "month":            "Month of Year",
    "quarter":          "Quarter",
    "year":             "Year",
    "commodity_enc":    "Commodity Type",
    "category_enc":     "Product Category",
    "market_enc":       "Market Location",
    "price_type_enc":   "Price Type (Wholesale/Retail)",
    "province_enc":     "Province",
    "season_enc":       "Agricultural Season",
    "monsoon":          "Monsoon Season Active",
    "is_festive":       "Festive Period",
    "inflation_index":  "Inflation Index",
    "seasonal_factor":  "Harvest Season Factor",
}

HARVEST_MONTHS = {
    "Tomato": [1,7], "Carrot": [2,8], "Cabbage": [1,7], "Beans": [12,6],
    "Brinjal": [3,9], "Snake Gourd": [6,11], "Pumpkin": [5,10],
    "Green Chilli": [4,10], "Lime": [4,10], "Leeks": [2,8], "Capsicum": [3,9],
    "Bitter Gourd": [6,12], "Mango": [4,5], "Papaya": [3,9], "Banana": [5,11],
    "Pineapple": [4,9], "Watermelon": [3,4], "Avocado": [7,8],
    "Passion Fruit": [6,12], "Guava": [8,10],
}

PROVINCE_MAP = {
    "Manning Market": "Western", "Dambulla": "Central",
    "Narahenpita": "Western", "Nuwara Eliya": "Central", "Colombo Local": "Western",
}

app = FastAPI(title="SL Produce Price API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load artifacts ───────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")

def load_all():
    raw_path  = os.path.join(BASE, "data", "raw", "srilanka_produce_prices.csv")
    proc_path = os.path.join(BASE, "data", "processed", "test_set.csv")
    model_path     = os.path.join(BASE, "models", "lgbm_model.pkl")
    encoder_path   = os.path.join(BASE, "models", "label_encoders.pkl")
    feat_path      = os.path.join(BASE, "models", "feature_cols.pkl")
    explainer_path = os.path.join(BASE, "models", "shap_explainer.pkl")
    metrics_path   = os.path.join(BASE, "models", "metrics.json")

    df        = pd.read_csv(raw_path, parse_dates=["date"])
    test_df   = pd.read_csv(proc_path, parse_dates=["date"])
    model     = joblib.load(model_path)
    encoders  = joblib.load(encoder_path)
    feat_cols = joblib.load(feat_path)
    explainer = joblib.load(explainer_path)
    with open(metrics_path) as f:
        metrics = json.load(f)
    return df, test_df, model, encoders, feat_cols, explainer, metrics

try:
    df, test_df, model, encoders, feat_cols, explainer, metrics = load_all()
    print("✓ All artifacts loaded.")
except Exception as e:
    print(f"✗ Load error: {e}")
    df = test_df = model = encoders = feat_cols = explainer = metrics = None

# ─── Pydantic schema ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    commodity: str
    market: str
    date: str         # YYYY-MM-DD
    price_type: str   # Wholesale | Retail
    inflation_index: float = 1.75

# ─── Helper: seasonal factor ─────────────────────────────────────────────────
def _seas_factor(commodity: str, month: int) -> float:
    peaks = HARVEST_MONTHS.get(commodity, [6])
    min_dist = min(abs(month - p) for p in peaks)
    return round(0.75 + (min_dist / 6.0) * 0.55, 3)

def _get_iterative_prediction(commodity: str, market: str, price_type: str, target_date: pd.Timestamp, inflation_index: float):
    """
    Core logic to get a prediction for a specific date (historical or future).
    If future, it recursively predicts from the last known data point up to target_date.
    """
    hist = df[(df["commodity"] == commodity) & (df["market"] == market)].sort_values("date")
    if len(hist) < 52:
        # Fallback for commodities with very little data
        mean_p = float(df[df["commodity"] == commodity]["price_lkr"].mean())
        std_p  = float(df[df["commodity"] == commodity]["price_lkr"].std())
        return mean_p, 0.0, None, None # price, base_val, sv, fv

    last_data_date = hist["date"].max()
    
    # helper for encoding
    def enc(col, val):
        return encoders[col].get(val, 0) if encoders else 0

    # Static category/province
    cat_rows = df[df["commodity"] == commodity]
    category = cat_rows["category"].iloc[0] if len(cat_rows) > 0 else "Vegetable"
    province = PROVINCE_MAP.get(market, "Western")

    # If the date is historical or at the edge, use simple lookup
    if target_date <= last_data_date:
        hist_before = hist[hist["date"] < target_date]
        # Use full history logic
        hist_seed = hist_before if len(hist_before) >= 52 else hist
        lag1  = float(hist_seed["price_lkr"].iloc[-1])
        lag4  = float(hist_seed["price_lkr"].iloc[-4])
        lag52 = float(hist_seed["price_lkr"].iloc[-52])
        rm4   = float(hist_seed["price_lkr"].iloc[-4:].mean())
        rs4   = float(hist_seed["price_lkr"].iloc[-4:].std())
        rm12  = float(hist_seed["price_lkr"].iloc[-12:].mean())
        pc1   = float(lag1 - hist_seed["price_lkr"].iloc[-2]) if len(hist_seed) > 1 else 0.0
        pc4   = float(lag1 - lag4)
        
        month, week, year, quarter, doy = target_date.month, int(target_date.isocalendar()[1]), target_date.year, (target_date.month - 1) // 3 + 1, target_date.dayofyear
        month_map = {10:"Maha",11:"Maha",12:"Maha",1:"Maha", 5:"Yala",6:"Yala",7:"Yala",8:"Yala"}
        season   = month_map.get(month, "Off-Season")
        monsoon  = 1 if month in [5,6,7,8,9,10,12] else 0
        is_fest  = 1 if (month == 4 and 14 <= week <= 16) else 0
        seas_fac = _seas_factor(commodity, month)

        fv = {
            "year": year, "month": month, "week": week, "quarter": quarter, "day_of_year": doy,
            "month_sin": float(np.sin(2*np.pi*month/12)), "month_cos": float(np.cos(2*np.pi*month/12)),
            "week_sin":  float(np.sin(2*np.pi*week/52)), "week_cos":  float(np.cos(2*np.pi*week/52)),
            "commodity_enc": enc("commodity", commodity), "category_enc": enc("category", category),
            "market_enc": enc("market", market), "price_type_enc": enc("price_type", price_type),
            "province_enc": enc("province", province), "season_enc": enc("season", season),
            "monsoon": monsoon, "is_festive": is_fest, "inflation_index": inflation_index, "seasonal_factor": seas_fac,
            "price_lag_1w": lag1, "price_lag_4w": lag4, "price_lag_52w": lag52,
            "rolling_mean_4w": rm4, "rolling_std_4w": rs4, "rolling_mean_12w": rm12,
            "price_change_1w": pc1, "price_change_4w": pc4,
        }
        X = pd.DataFrame([[fv[c] for c in feat_cols]], columns=feat_cols)
        pred = float(model.predict(X)[0])
        contrib = model.predict(X, pred_contrib=True)[0]
        return pred, float(contrib[-1]), contrib[:-1], fv

    # Else: target_date is in the future relative to the dataset
    # We must predict week-by-week from last_data_date to target_date
    recent_prices = list(hist["price_lkr"].iloc[-52:].values)
    current_date = last_data_date
    
    pred = 0.0
    base_val = 0.0
    sv = None
    fv = None

    while current_date < target_date:
        current_date += pd.Timedelta(weeks=1)
        month, week, year, quarter, doy = current_date.month, int(current_date.isocalendar()[1]), current_date.year, (current_date.month - 1) // 3 + 1, current_date.dayofyear
        
        month_map = {10:"Maha",11:"Maha",12:"Maha",1:"Maha", 5:"Yala",6:"Yala",7:"Yala",8:"Yala"}
        season  = month_map.get(month, "Off-Season")
        monsoon = 1 if month in [5,6,7,8,9,10,12] else 0
        is_fest = 1 if (month == 4 and 14 <= week <= 16) else 0
        seas_fac = _seas_factor(commodity, month)
        
        # Mildly grow inflation if desired, but for single predict we usually keep req.inflation
        # However, to match /forecast loop logic (step-based), we can use the requested inflation as base
        # If we are filling the gap between Dec 2025 and Target, we iterate.
        
        lag1  = recent_prices[-1]
        lag4  = recent_prices[-4]
        lag52 = recent_prices[-52]
        rm4   = float(np.mean(recent_prices[-4:]))
        rs4   = float(np.std(recent_prices[-4:]))
        rm12  = float(np.mean(recent_prices[-12:]))
        pc1   = lag1 - recent_prices[-2]
        pc4   = lag1 - lag4

        fv = {
            "year": year, "month": month, "week": week, "quarter": quarter, "day_of_year": doy,
            "month_sin": float(np.sin(2*np.pi*month/12)), "month_cos": float(np.cos(2*np.pi*month/12)),
            "week_sin":  float(np.sin(2*np.pi*week/52)), "week_cos":  float(np.cos(2*np.pi*week/52)),
            "commodity_enc": enc("commodity", commodity), "category_enc": enc("category", category),
            "market_enc": enc("market", market), "price_type_enc": enc("price_type", price_type),
            "province_enc": enc("province", province), "season_enc": enc("season", season),
            "monsoon": monsoon, "is_festive": is_fest, "inflation_index": inflation_index, "seasonal_factor": seas_fac,
            "price_lag_1w": lag1, "price_lag_4w": lag4, "price_lag_52w": lag52,
            "rolling_mean_4w": rm4, "rolling_std_4w": rs4, "rolling_mean_12w": rm12,
            "price_change_1w": pc1, "price_change_4w": pc4,
        }
        X = pd.DataFrame([[fv[c] for c in feat_cols]], columns=feat_cols)
        pred = float(model.predict(X)[0])
        
        # Update rolling window
        recent_prices.append(pred)
        recent_prices = recent_prices[-52:]
        
        # Only capture SHAP/Final FV for the LAST step (the one the user actually requested)
        if current_date >= target_date:
            contrib = model.predict(X, pred_contrib=True)[0]
            sv = contrib[:-1]
            base_val = float(contrib[-1])

    return pred, base_val, sv, fv

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/api/metadata")
def metadata():
    if df is None:
        raise HTTPException(503, "Data not loaded")
    return {
        "commodities": sorted(df["commodity"].unique().tolist()),
        "markets":     sorted(df["market"].unique().tolist()),
        "price_types": ["Wholesale", "Retail"],
        "date_range":  [str(df["date"].min().date()), str(df["date"].max().date())],
        "metrics":     metrics,
    }

@app.get("/api/dashboard/kpis")
def dashboard_kpis():
    if df is None:
        raise HTTPException(503, "Data not loaded")
    latest_date = df["date"].max()
    prev_date   = latest_date - pd.Timedelta(weeks=1)
    avg_now  = df[df["date"] == latest_date]["price_lkr"].mean()
    avg_prev = df[df["date"] == prev_date]["price_lkr"].mean()
    delta_pct = round((avg_now - avg_prev) / avg_prev * 100, 1) if avg_prev else 0
    most_exp  = df.groupby("commodity")["price_lkr"].mean().idxmax()
    most_vol  = df.groupby("commodity")["price_lkr"].std().idxmax()
    return {
        "avg_price": round(avg_now, 2),
        "avg_price_delta_pct": delta_pct,
        "most_expensive": most_exp,
        "most_volatile": most_vol,
        "total_records": len(df),
        "latest_date": str(latest_date.date()),
    }

@app.get("/api/dashboard/trends")
def dashboard_trends(
    commodities: str = "Tomato,Carrot,Beans,Green Chilli",
    markets: str = "",
    year_from: int = 2019,
    year_to: int = 2024,
):
    selected = [c.strip() for c in commodities.split(",") if c.strip()]
    fdf = df[df["commodity"].isin(selected) &
             df["year"].between(year_from, year_to)].copy()
    if markets:
        mlist = [m.strip() for m in markets.split(",") if m.strip()]
        fdf = fdf[fdf["market"].isin(mlist)]
    trend = (fdf.groupby(["date","commodity"])["price_lkr"]
               .mean().reset_index().sort_values("date"))
    trend["date"] = trend["date"].dt.strftime("%Y-%m-%d")
    return trend.to_dict(orient="records")

@app.get("/api/dashboard/market-comparison")
def market_comparison():
    if df is None:
        raise HTTPException(503)
    mkt = (df.groupby(["market","price_type"])["price_lkr"]
             .mean().reset_index()
             .rename(columns={"price_lkr": "avg_price"}))
    mkt["avg_price"] = mkt["avg_price"].round(2)
    return mkt.to_dict(orient="records")

@app.get("/api/dashboard/seasonal-heatmap")
def seasonal_heatmap():
    if df is None:
        raise HTTPException(503)
    heat = (df.groupby(["commodity","month"])["price_lkr"]
               .mean().reset_index()
               .rename(columns={"price_lkr": "avg_price"}))
    heat["avg_price"] = heat["avg_price"].round(2)
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    heat["month_name"] = heat["month"].map(month_map)
    return heat.to_dict(orient="records")

@app.get("/api/dashboard/yoy")
def year_over_year():
    if df is None:
        raise HTTPException(503)
    yoy = df.groupby("year")["price_lkr"].mean().reset_index().rename(
        columns={"price_lkr": "avg_price"})
    yoy["avg_price"] = yoy["avg_price"].round(2)
    return yoy.to_dict(orient="records")

@app.get("/api/dashboard/category-distribution")
def category_distribution():
    if df is None:
        raise HTTPException(503)
    # Return percentile stats per category
    result = []
    for cat, grp in df.groupby("category"):
        q = grp["price_lkr"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict()
        result.append({
            "category": cat,
            "min": round(grp["price_lkr"].min(), 2),
            "q10": q[0.10], "q25": q[0.25], "median": q[0.50],
            "q75": q[0.75], "q90": q[0.90],
            "max": round(grp["price_lkr"].max(), 2),
            "mean": round(grp["price_lkr"].mean(), 2),
        })
    return result

@app.post("/api/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
        
    pdate = pd.to_datetime(req.date)
    pred, base_val, sv, fv = _get_iterative_prediction(
        req.commodity, req.market, req.price_type, pdate, req.inflation_index
    )

    shap_contributions = []
    if sv is not None:
        for feat, val in zip(feat_cols, sv):
            shap_contributions.append({
                "feature": feat,
                "label": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "shap_value": round(float(val), 3),
                "feature_value": round(float(fv[feat]), 3),
            })
        shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Historical comparison
    month = pdate.month
    hist_month = df[
        (df["commodity"] == req.commodity) &
        (df["market"] == req.market) &
        (df["month"] == month)
    ]["price_lkr"]
    hist_avg = float(hist_month.mean()) if len(hist_month) > 0 else None
    hist_std = float(hist_month.std()) if len(hist_month) > 0 else None

    return {
        "predicted_price": round(pred, 2),
        "lower_bound": round(pred * 0.88, 2),
        "upper_bound": round(pred * 1.12, 2),
        "commodity": req.commodity,
        "market": req.market,
        "date": req.date,
        "price_type": req.price_type,
        "season": fv["season_enc"] if fv else "Unknown", 
        "is_festive": bool(fv["is_festive"]) if fv else False,
        "base_value": round(base_val, 2),
        "shap_contributions": shap_contributions[:14],
        "historical_avg": round(hist_avg, 2) if hist_avg else None,
        "historical_std": round(hist_std, 2) if hist_std else None,
    }

@app.get("/api/explain/feature-importance")
def feature_importance():
    if model is None:
        raise HTTPException(503)
    fi = model.feature_importances_
    result = []
    for feat, imp in zip(feat_cols, fi):
        result.append({
            "feature": feat,
            "label": FEATURE_DISPLAY_NAMES.get(feat, feat),
            "importance": round(float(imp), 4),
        })
    result.sort(key=lambda x: x["importance"], reverse=True)
    return result[:15]

@app.get("/api/explain/pdp")
def pdp(feature: str = "month", n_pts: int = 25):
    if model is None:
        raise HTTPException(503)
    if feature not in feat_cols:
        raise HTTPException(400, f"Feature '{feature}' not in model features")
    X_sample = test_df[feat_cols].sample(300, random_state=42).values
    feat_idx = feat_cols.index(feature)
    vals = np.linspace(X_sample[:, feat_idx].min(), X_sample[:, feat_idx].max(), n_pts)
    result = []
    for v in vals:
        Xc = X_sample.copy()
        Xc[:, feat_idx] = v
        preds = model.predict(Xc)
        result.append({
            "x": round(float(v), 4),
            "mean_price": round(float(preds.mean()), 2),
            "std_price":  round(float(preds.std()), 2),
        })
    return {
        "feature": feature,
        "label": FEATURE_DISPLAY_NAMES.get(feature, feature),
        "points": result,
    }

@app.get("/api/forecast")
def forecast(
    commodity: str = "Tomato",
    market: str = "Manning Market",
    price_type: str = "Wholesale",
    weeks: int = 12,
    inflation_index: float = 1.50,
):
    """
    Iterative multi-week forecast from TODAY forward.
    Ensures consistency by using the same logic as single predict.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    start_date = pd.Timestamp.today().normalize()
    results = []

    for step in range(weeks):
        pred_date = start_date + pd.Timedelta(weeks=step)
        
        # Inflation grows mildly over future weeks (0.5% per month)
        months_ahead = step / 4.33
        infl = min(inflation_index * (1 + 0.005 * months_ahead), 3.0)
        
        pred, _, _, fv = _get_iterative_prediction(
            commodity, market, price_type, pred_date, infl
        )

        results.append({
            "week": step + 1,
            "date": pred_date.strftime("%Y-%m-%d"),
            "predicted_price": round(pred, 2),
            "lower_bound": round(pred * 0.88, 2),
            "upper_bound": round(pred * 1.12, 2),
            "season": fv["season_enc"] if fv else "Unknown",
            "is_festive": bool(fv["is_festive"]) if fv else False,
            "inflation_used": round(infl, 3),
        })

    avg_fc = round(float(np.mean([r["predicted_price"] for r in results])), 2)
    min_fc = min(results, key=lambda r: r["predicted_price"])
    max_fc = max(results, key=lambda r: r["predicted_price"])

    return {
        "commodity": commodity,
        "market": market,
        "price_type": price_type,
        "weeks_ahead": weeks,
        "base_inflation": inflation_index,
        "avg_forecast": avg_fc,
        "min_week": {"date": min_fc["date"], "price": min_fc["predicted_price"]},
        "max_week": {"date": max_fc["date"], "price": max_fc["predicted_price"]},
        "forecasts": results,
    }


# ─── Serve React build ────────────────────────────────────────────────────────
frontend_dist = os.path.join(BASE, "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_react(full_path: str):
        index = os.path.join(frontend_dist, "index.html")
        return FileResponse(index)
