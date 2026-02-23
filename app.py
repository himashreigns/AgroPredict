"""
app.py  –  Sri Lanka Vegetable & Fruit Price Predictor
Two-page Streamlit application:
  Page 1: Dashboard  — historical trends, market heatmaps, commodity comparison
  Page 2: Predict & Explain — LightGBM prediction + SHAP waterfall
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 AgroPredict",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid rgba(99,102,241,0.3);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    backdrop-filter: blur(10px);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #818cf8; }
.metric-label { font-size: 0.85rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-delta { font-size: 0.9rem; margin-top: 4px; }
.delta-up { color: #f87171; }
.delta-down { color: #34d399; }

/* Section headers */
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #a5b4fc;
    margin: 24px 0 16px; padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.4);
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(6,182,212,0.1));
    border: 2px solid rgba(16,185,129,0.5);
    border-radius: 20px; padding: 30px; text-align: center;
    margin: 16px 0;
}
.pred-price { font-size: 3rem; font-weight: 800; color: #34d399; }
.pred-label { font-size: 1rem; color: #94a3b8; }
.pred-range { font-size: 1rem; color: #6ee7b7; margin-top: 6px; }

/* Nav buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; border: none; border-radius: 12px;
    padding: 10px 24px; font-weight: 600; width: 100%;
    transition: all 0.2s; font-size: 0.95rem;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }

/* Streamlit widgets */
.stSelectbox label, .stDateInput label, .stSlider label { color: #94a3b8 !important; }
div[data-baseweb="select"] { background: rgba(30,41,59,0.8) !important; }
.stTextInput input { background: rgba(30,41,59,0.8) !important; color: #e2e8f0 !important; }

/* plotly dark style wrapper */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    border-radius: 20px; padding: 28px 36px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
}
.hero-title { font-size: 1.9rem; font-weight: 800; color: white; margin: 0; }
.hero-sub { font-size: 0.95rem; color: rgba(255,255,255,0.85); margin-top: 4px; }

/* Alert box */
.info-box {
    background: rgba(99,102,241,0.12);
    border-left: 4px solid #6366f1;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
    color: #c7d2fe; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Data & Model Caching ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/raw/srilanka_produce_prices.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df

@st.cache_resource
def load_model():
    try:
        model      = joblib.load("models/lgbm_model.pkl")
        encoders   = joblib.load("models/label_encoders.pkl")
        feat_cols  = joblib.load("models/feature_cols.pkl")
        explainer  = joblib.load("models/shap_explainer.pkl")
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        return model, encoders, feat_cols, explainer, metrics
    except Exception:
        return None, None, None, None, None

# ── Plotly Theme ──────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(15,23,42,0)", plot_bgcolor="rgba(30,41,59,0.5)",
    font=dict(color="#cbd5e1", family="Inter"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(30,41,59,0.7)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
)
COLOR_SEQ = px.colors.qualitative.Bold

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 20px;">
        <div style="font-size:2.5rem;">🥦</div>
        <div style="font-size:1.1rem; font-weight:700; color:#a5b4fc;">AgroPredict</div>
        <div style="font-size:0.75rem; color:#64748b;">Powered by LightGBM + SHAP</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Predict & Explain"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#475569; padding: 8px 0;">
    <b style="color:#64748b;">Data Sources</b><br>
    • DOA / SHEP Agri InfoHub<br>
    • CBSL Economic Indicators<br>
    • HARTI Food Information Bulletins<br>
    • Dept. of Census & Statistics<br><br>
    <b style="color:#64748b;">Model</b><br>
    LightGBM Regressor<br>
    (Gradient Boosted Trees)<br><br>
    <b style="color:#64748b;">XAI</b><br>
    SHAP + Partial Dependence Plots
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">📊 AgroPredict — Price Dashboard</div>
            <div class="hero-sub">Historical weekly prices for 20 vegetables & fruits · 5 major markets · 2019–2024</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.error("Dataset not found. Please run `python data/generate_dataset.py` first.")
        st.stop()

    # ── KPI Cards ─────────────────────────────────────────────────────────
    latest_date = df["date"].max()
    prev_date   = latest_date - pd.Timedelta(weeks=1)
    latest_df   = df[df["date"] == latest_date]
    prev_df     = df[df["date"] == prev_date]

    avg_now  = latest_df["price_lkr"].mean()
    avg_prev = prev_df["price_lkr"].mean() if len(prev_df) > 0 else avg_now
    delta_pct = ((avg_now - avg_prev) / avg_prev * 100) if avg_prev else 0

    most_exp  = latest_df.groupby("commodity")["price_lkr"].mean().idxmax()
    most_vol  = df.groupby("commodity")["price_lkr"].std().idxmax()
    total_rec = f"{len(df):,}"

    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        (col1, "Avg Market Price", f"Rs {avg_now:,.0f}", f"{'▲' if delta_pct>0 else '▼'} {abs(delta_pct):.1f}% vs last week", delta_pct > 0),
        (col2, "Most Expensive", most_exp, "Latest week highest avg", False),
        (col3, "Most Volatile", most_vol, "Highest price std dev", False),
        (col4, "Total Records", total_rec, "Across 5 years · 5 markets", False),
    ]
    for col, label, val, sub, is_up in kpis:
        with col:
            delta_cls = "delta-up" if is_up else "delta-down"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-delta {delta_cls}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Filter Data</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_commodity = st.multiselect(
            "Commodity", sorted(df["commodity"].unique()),
            default=["Tomato", "Carrot", "Beans", "Green Chilli"]
        )
    with fc2:
        sel_market = st.multiselect(
            "Market", sorted(df["market"].unique()),
            default=list(df["market"].unique())
        )
    with fc3:
        sel_years = st.slider("Year Range", 2019, 2024, (2019, 2024))

    fdf = df.copy()
    if sel_commodity: fdf = fdf[fdf["commodity"].isin(sel_commodity)]
    if sel_market:    fdf = fdf[fdf["market"].isin(sel_market)]
    fdf = fdf[(fdf["year"] >= sel_years[0]) & (fdf["year"] <= sel_years[1])]

    # ── Price Trend Chart ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Price Trends Over Time</div>', unsafe_allow_html=True)
    trend_df = (fdf.groupby(["date", "commodity"])["price_lkr"]
                   .mean().reset_index()
                   .sort_values("date"))
    fig_trend = px.line(
        trend_df, x="date", y="price_lkr", color="commodity",
        labels={"price_lkr": "Price (LKR/kg)", "date": "Date", "commodity": "Commodity"},
        color_discrete_sequence=COLOR_SEQ,
        height=400,
    )
    fig_trend.update_traces(line_width=1.8)
    fig_trend.update_layout(**DARK_LAYOUT, title="Weekly Average Price per Commodity")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Market Comparison + Category Distribution ──────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">🏪 Average Price by Market</div>', unsafe_allow_html=True)
        mkt_df = fdf.groupby(["market", "price_type"])["price_lkr"].mean().reset_index()
        fig_mkt = px.bar(
            mkt_df, x="market", y="price_lkr", color="price_type",
            barmode="group",
            labels={"price_lkr": "Avg Price (LKR/kg)", "market": "Market", "price_type": "Type"},
            color_discrete_map={"Wholesale": "#6366f1", "Retail": "#f59e0b"},
            height=320,
        )
        fig_mkt.update_layout(**DARK_LAYOUT, title="Wholesale vs Retail by Market")
        st.plotly_chart(fig_mkt, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">📦 Price Distribution by Category</div>', unsafe_allow_html=True)
        fig_box = px.box(
            fdf, x="category", y="price_lkr", color="category",
            color_discrete_map={"Vegetable": "#10b981", "Fruit": "#f59e0b"},
            labels={"price_lkr": "Price (LKR/kg)", "category": "Category"},
            height=320,
        )
        fig_box.update_layout(**DARK_LAYOUT, title="Price Distribution: Vegetables vs Fruits")
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Seasonal Heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">🗓️ Seasonal Price Heatmap</div>', unsafe_allow_html=True)
    heat_df = (df[df["commodity"].isin(sel_commodity or df["commodity"].unique())]
               .groupby(["commodity", "month"])["price_lkr"].mean().reset_index())
    pivot = heat_df.pivot(index="commodity", columns="month", values="price_lkr")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[month_labels[i-1] for i in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        reversescale=True,
        text=np.round(pivot.values, 0),
        texttemplate="%{text:.0f}",
        textfont={"size": 9},
        colorbar=dict(title="LKR/kg", titlefont=dict(color="#94a3b8")),
    ))
    fig_heat.update_layout(
        **DARK_LAYOUT, height=400,
        title="Average Price by Commodity × Month (higher = more expensive, red = peak scarcity)",
        xaxis_title="Month", yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Top Commodities Bar ────────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-title">🏆 Top 10 Most Expensive (avg)</div>', unsafe_allow_html=True)
        top_df = (df.groupby("commodity")["price_lkr"].mean()
                    .sort_values(ascending=False).head(10).reset_index())
        fig_top = px.bar(
            top_df, x="price_lkr", y="commodity", orientation="h",
            color="price_lkr", color_continuous_scale="purples",
            labels={"price_lkr": "Avg Price (LKR/kg)", "commodity": ""},
            height=330,
        )
        fig_top.update_layout(**DARK_LAYOUT, showlegend=False,
                              title="Average Price 2019–2024")
        st.plotly_chart(fig_top, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">📅 Year-over-Year Avg Price</div>', unsafe_allow_html=True)
        yoy_df = df.groupby("year")["price_lkr"].mean().reset_index()
        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Bar(
            x=yoy_df["year"], y=yoy_df["price_lkr"],
            marker_color=["#6366f1","#8b5cf6","#a78bfa","#f87171","#fb923c","#fbbf24"],
            text=[f"Rs {v:,.0f}" for v in yoy_df["price_lkr"]],
            textposition="outside", textfont=dict(color="#e2e8f0"),
        ))
        fig_yoy.update_layout(
            **DARK_LAYOUT, height=330,
            title="Sri Lanka Inflation Impact on Produce Prices",
            xaxis_title="Year", yaxis_title="Avg Price (LKR/kg)",
        )
        fig_yoy.add_annotation(x=2022, y=yoy_df[yoy_df["year"]==2022]["price_lkr"].values[0],
            text="⚠️ 2022 Crisis", showarrow=True, arrowcolor="#f87171",
            font=dict(color="#f87171", size=11), ay=-40)
        st.plotly_chart(fig_yoy, use_container_width=True)

    # ── Raw Data Table ─────────────────────────────────────────────────────
    with st.expander("📋 View Raw Data Sample"):
        st.dataframe(
            fdf[["date","commodity","category","market","price_type","season","price_lkr"]]
              .sort_values("date", ascending=False).head(200),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICT & EXPLAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict & Explain":
    st.markdown("""
    <div class="hero-banner">
        <div>
            <div class="hero-title">🔮 Predict & Explain</div>
            <div class="hero-sub">LightGBM price prediction with SHAP explainability · Enter details below</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model, encoders, feat_cols, explainer, metrics = load_model()
    df = load_data()

    if model is None:
        st.error("Model not found. Please run `python models/train.py` first, then `python models/explain.py`.")
        st.stop()

    # ── Model Metrics Banner ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📐 Model Performance (Test Set)</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metric_info = [
        (m1, "RMSE", f"Rs {metrics.get('RMSE','-'):.2f}", "Root Mean Square Error"),
        (m2, "MAE",  f"Rs {metrics.get('MAE','-'):.2f}",  "Mean Absolute Error"),
        (m3, "R²",   f"{metrics.get('R2','-'):.4f}",       "Coefficient of Determination"),
        (m4, "MAPE", f"{metrics.get('MAPE','-'):.2f}%",    "Mean Abs Pct Error"),
    ]
    for col, label, val, desc in metric_info:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.6rem">{val}</div>
                <div class="metric-delta">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Input Panel ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎛️ Configure Prediction</div>', unsafe_allow_html=True)
    inp1, inp2, inp3 = st.columns(3)

    COMMODITY_LIST = sorted(encoders["commodity"].keys())
    MARKET_LIST    = sorted(encoders["market"].keys())

    with inp1:
        commodity  = st.selectbox("🥕 Commodity",  COMMODITY_LIST, index=COMMODITY_LIST.index("Tomato"))
        price_type = st.selectbox("🏷️ Price Type", ["Wholesale", "Retail"])
    with inp2:
        market     = st.selectbox("🏪 Market",  MARKET_LIST)
        pred_date  = st.date_input("📅 Date", value=pd.to_datetime("2024-06-15"))
    with inp3:
        inflation  = st.slider("📈 Inflation Index", 0.9, 2.5, 1.75, 0.05,
                               help="1.0 = 2019 baseline; 1.95 = 2022 crisis peak")
        st.markdown(f"""
        <div class="info-box">
            <b>Reference:</b><br>
            2019: 1.00 (baseline) · 2021: 1.15<br>
            2022 peak: ~1.95 · 2024: ~1.75
        </div>""", unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────────────────────────
    st.markdown("")
    predict_btn = st.button("⚡  Predict Price", use_container_width=False)

    if predict_btn:
        pdate   = pd.to_datetime(pred_date)
        month   = pdate.month
        week    = int(pdate.isocalendar()[1])
        year    = pdate.year
        quarter = (month - 1) // 3 + 1
        doy     = pdate.dayofyear

        season_map = {10:"Maha",11:"Maha",12:"Maha",1:"Maha",
                      5:"Yala",6:"Yala",7:"Yala",8:"Yala"}
        season   = season_map.get(month, "Off-Season")
        monsoon  = 1 if month in [5,6,7,8,9,10,12] else 0
        is_fest  = 1 if (month == 4 and 14 <= week <= 16) else 0

        # Pull category & province from data
        cat_row = df[df["commodity"] == commodity].iloc[0] if df is not None else None
        category = cat_row["category"]  if cat_row is not None else "Vegetable"
        province_map = {"Manning Market":"Western","Dambulla":"Central",
                        "Narahenpita":"Western","Nuwara Eliya":"Central","Colombo Local":"Western"}
        province = province_map.get(market, "Western")

        # Seasonal factor (approx)
        harvest_months = {
            "Tomato":[1,7],"Carrot":[2,8],"Cabbage":[1,7],"Beans":[12,6],
            "Brinjal":[3,9],"Snake Gourd":[6,11],"Pumpkin":[5,10],
            "Green Chilli":[4,10],"Lime":[4,10],"Leeks":[2,8],"Capsicum":[3,9],
            "Bitter Gourd":[6,12],"Mango":[4,5],"Papaya":[3,9],"Banana":[5,11],
            "Pineapple":[4,9],"Watermelon":[3,4],"Avocado":[7,8],
            "Passion Fruit":[6,12],"Guava":[8,10],
        }
        peaks = harvest_months.get(commodity, [6])
        min_dist = min(abs(month - p) for p in peaks)
        seas_fac = round(0.75 + (min_dist / 6.0) * 0.55, 3)

        # Encode categoricals
        def enc(col, val):
            return encoders[col].get(val, 0)

        # Estimate lags from historical data
        hist = df[(df["commodity"] == commodity) & (df["market"] == market)].sort_values("date")
        if len(hist) >= 52:
            lag1  = hist["price_lkr"].iloc[-1]
            lag4  = hist["price_lkr"].iloc[-4]
            lag52 = hist["price_lkr"].iloc[-52]
            rm4   = hist["price_lkr"].iloc[-4:].mean()
            rs4   = hist["price_lkr"].iloc[-4:].std()
            rm12  = hist["price_lkr"].iloc[-12:].mean()
            pc1   = lag1 - hist["price_lkr"].iloc[-2]
            pc4   = lag1 - lag4
        else:
            # Fallback: use commodity overall mean
            mean_p = df[df["commodity"] == commodity]["price_lkr"].mean()
            lag1 = lag4 = lag52 = rm4 = rm12 = mean_p
            rs4 = df[df["commodity"] == commodity]["price_lkr"].std()
            pc1 = pc4 = 0.0

        feature_vector = {
            "year": year, "month": month, "week": week, "quarter": quarter, "day_of_year": doy,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "week_sin":  np.sin(2 * np.pi * week / 52),
            "week_cos":  np.cos(2 * np.pi * week / 52),
            "commodity_enc":  enc("commodity", commodity),
            "category_enc":   enc("category", category),
            "market_enc":     enc("market", market),
            "price_type_enc": enc("price_type", price_type),
            "province_enc":   enc("province", province),
            "season_enc":     enc("season", season),
            "monsoon": monsoon, "is_festive": is_fest,
            "inflation_index": inflation, "seasonal_factor": seas_fac,
            "price_lag_1w": lag1, "price_lag_4w": lag4, "price_lag_52w": lag52,
            "rolling_mean_4w": rm4, "rolling_std_4w": rs4, "rolling_mean_12w": rm12,
            "price_change_1w": pc1, "price_change_4w": pc4,
        }

        X_input = np.array([[feature_vector[c] for c in feat_cols]])
        pred    = model.predict(X_input)[0]
        pred_lo = pred * 0.88
        pred_hi = pred * 1.12

        # ── Prediction Result ──────────────────────────────────────────────
        st.markdown('<div class="section-title">💰 Prediction Result</div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="pred-label">{commodity} · {market}</div>
                <div class="pred-price">Rs {pred:,.0f}</div>
                <div class="pred-label">per kg · {price_type}</div>
                <div class="pred-range">Range: Rs {pred_lo:,.0f} – Rs {pred_hi:,.0f}</div>
                <div style="margin-top:14px; font-size:0.8rem; color:#6ee7b7;">
                    {season} season · {'🎉 Festive period' if is_fest else '📅 Regular period'}
                </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            # Compare with historical same-month average
            if df is not None:
                hist_month = df[
                    (df["commodity"] == commodity) &
                    (df["market"] == market) &
                    (df["month"] == month)
                ]["price_lkr"]
                if len(hist_month) > 0:
                    hist_avg = hist_month.mean()
                    hist_std = hist_month.std()
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=pred,
                        delta={"reference": hist_avg, "valueformat": ".0f",
                               "prefix": "vs hist avg Rs "},
                        gauge={
                            "axis": {"range": [hist_avg - 2*hist_std, hist_avg + 2*hist_std],
                                     "tickcolor": "#94a3b8"},
                            "bar": {"color": "#6366f1"},
                            "steps": [
                                {"range": [hist_avg - 2*hist_std, hist_avg - hist_std], "color": "#1e3a5f"},
                                {"range": [hist_avg - hist_std, hist_avg + hist_std],   "color": "#1e4d3f"},
                                {"range": [hist_avg + hist_std, hist_avg + 2*hist_std], "color": "#4d1e1e"},
                            ],
                            "threshold": {"line": {"color": "#f87171", "width": 3},
                                          "thickness": 0.75, "value": hist_avg},
                        },
                        title={"text": f"Predicted vs Historical Avg (Month {month})",
                               "font": {"color": "#94a3b8"}},
                        number={"prefix": "Rs ", "font": {"color": "#34d399"}},
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1", family="Inter"),
                        height=280, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP Explanation ───────────────────────────────────────────────
        st.markdown('<div class="section-title">🔬 Why This Prediction? (SHAP Explanation)</div>',
                    unsafe_allow_html=True)
        try:
            shap_vals = explainer.shap_values(X_input)
            sv = shap_vals[0]
            base_val = explainer.expected_value

            # Build waterfall data
            sv_df = pd.DataFrame({
                "feature": feat_cols,
                "shap_value": sv,
            }).sort_values("shap_value", key=abs, ascending=False).head(12)

            colors = ["#f87171" if v > 0 else "#34d399" for v in sv_df["shap_value"]]
            fig_wf = go.Figure(go.Bar(
                x=sv_df["shap_value"], y=sv_df["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.1f}" for v in sv_df["shap_value"]],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig_wf.add_vline(x=0, line_color="#94a3b8", line_width=1)
            fig_wf.update_layout(
                **DARK_LAYOUT, height=400,
                title=f"SHAP Feature Contributions to Prediction (base ≈ Rs {base_val:.0f})",
                xaxis_title="SHAP Value (LKR impact on price)",
                yaxis_title="",
                xaxis=dict(**DARK_LAYOUT.get("xaxis", {})),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            🔴 <b>Red bars</b> = features that pushed the price <b>higher</b> than the baseline.<br>
            🟢 <b>Green bars</b> = features that pushed the price <b>lower</b>.<br>
            The prediction = base value + sum of all SHAP contributions.
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")

        # ── Global Explanations ────────────────────────────────────────────
        st.markdown('<div class="section-title">🌍 Global Model Explanations</div>',
                    unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🔬 SHAP Summary", "📉 Partial Dependence"])

        with tab1:
            if os.path.exists("assets/feature_importance.png"):
                st.image("assets/feature_importance.png", use_column_width=True)
            else:
                st.info("Run `python models/train.py` to generate feature importance plot.")

        with tab2:
            if os.path.exists("assets/shap_summary.png"):
                st.image("assets/shap_summary.png", use_column_width=True)
                st.markdown("""
                <div class="info-box">
                The SHAP beeswarm plot shows impact of each feature across 500 test samples.
                Feature values are colour-coded (red = high, blue = low).
                Wider spread = more variability in impact.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Run `python models/explain.py` to generate SHAP plots.")

        with tab3:
            if os.path.exists("assets/pdp.png"):
                st.image("assets/pdp.png", use_column_width=True)
                st.markdown("""
                <div class="info-box">
                <b>Partial Dependence Plots (PDP)</b> show the marginal effect of a single feature
                on the predicted price, averaging out all other features. Shaded band = ±1 std dev.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Run `python models/explain.py` to generate PDP plots.")

    # ── If no prediction yet, show prompts ────────────────────────────────
    else:
        st.markdown("""
        <div class="info-box" style="margin-top:20px;">
        👆 Fill in the commodity, market, date, and inflation index above, then click
        <b>⚡ Predict Price</b> to see the LightGBM prediction and SHAP explanation.
        </div>""", unsafe_allow_html=True)

        # Show algorithm explanation
        st.markdown('<div class="section-title">🤖 About the Algorithm: LightGBM</div>',
                    unsafe_allow_html=True)
        col_alg1, col_alg2 = st.columns(2)
        with col_alg1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Why LightGBM?</div>
                <div style="color:#e2e8f0; font-size:0.9rem; margin-top:8px;">
                <b>LightGBM</b> (Light Gradient Boosting Machine) is a gradient boosting framework
                using tree-based learning optimised for speed and memory efficiency.
                It uses <b>leaf-wise tree growth</b> (vs level-wise in XGBoost), achieving
                better accuracy for the same iterations.
                <br><br>
                <b>Why it fits this problem:</b><br>
                ✅ Handles mixed categorical + numerical features natively<br>
                ✅ Excellent performance on tabular time-series data<br>
                ✅ Fast training on 26,000+ records<br>
                ✅ Built-in early stopping prevents overfitting<br>
                ✅ Compatible with SHAP for explainability
                </div>
            </div>""", unsafe_allow_html=True)
        with col_alg2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">vs Standard Algorithms</div>
                <div style="color:#e2e8f0; font-size:0.9rem; margin-top:8px;">
                <table width="100%" style="border-collapse:collapse; font-size:0.85rem;">
                <tr style="color:#a5b4fc;"><th>Algorithm</th><th>Handles Non-linearity</th><th>Feature Interactions</th><th>Speed</th></tr>
                <tr><td>Linear Regression</td><td>❌</td><td>❌</td><td>⚡ Fast</td></tr>
                <tr><td>Decision Tree</td><td>✅</td><td>Partial</td><td>⚡ Fast</td></tr>
                <tr><td>k-NN</td><td>✅</td><td>❌</td><td>🐢 Slow</td></tr>
                <tr><td>SVM (RBF)</td><td>✅</td><td>Partial</td><td>🐢 Slow</td></tr>
                <tr style="color:#34d399;"><td><b>LightGBM ✓</b></td><td>✅✅</td><td>✅✅</td><td>⚡⚡ Very Fast</td></tr>
                </table>
                </div>
            </div>""", unsafe_allow_html=True)

        if os.path.exists("assets/shap_summary.png"):
            st.markdown('<div class="section-title">🔬 SHAP Feature Impacts (All Test Samples)</div>',
                        unsafe_allow_html=True)
            st.image("assets/shap_summary.png", use_column_width=True)
