"""
generate_dataset.py
comprehensive, realistic historical weekly price dataset
for vegetables and fruits at major Sri Lanka markets (2019–2025).

Grounded in:
  - DOA/HARTI published price ranges
  - Known Maha/Yala agricultural seasons
  - Sri Lanka 2021–2022 economic crisis inflation spike
  - Market-type differentials (wholesale vs retail)
  - Festive demand bumps (April Sinhala/Tamil New Year, December)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

np.random.seed(42)

# ── Base price ranges (LKR/kg) ──────────────────────────────────────────────
COMMODITIES = {
    # Vegetables
    "Tomato":        {"base": 80,  "vol": 30, "category": "Vegetable"},
    "Carrot":        {"base": 100, "vol": 35, "category": "Vegetable"},
    "Cabbage":       {"base": 60,  "vol": 20, "category": "Vegetable"},
    "Beans":         {"base": 120, "vol": 45, "category": "Vegetable"},
    "Brinjal":       {"base": 90,  "vol": 35, "category": "Vegetable"},
    "Snake Gourd":   {"base": 70,  "vol": 25, "category": "Vegetable"},
    "Pumpkin":       {"base": 55,  "vol": 20, "category": "Vegetable"},
    "Green Chilli":  {"base": 200, "vol": 80, "category": "Vegetable"},
    "Lime":          {"base": 180, "vol": 70, "category": "Vegetable"},
    "Leeks":         {"base": 110, "vol": 40, "category": "Vegetable"},
    "Capsicum":      {"base": 250, "vol": 90, "category": "Vegetable"},
    "Bitter Gourd":  {"base": 95,  "vol": 35, "category": "Vegetable"},
    # Fruits
    "Mango":         {"base": 150, "vol": 60, "category": "Fruit"},
    "Papaya":        {"base": 80,  "vol": 30, "category": "Fruit"},
    "Banana":        {"base": 70,  "vol": 25, "category": "Fruit"},
    "Pineapple":     {"base": 120, "vol": 40, "category": "Fruit"},
    "Watermelon":    {"base": 60,  "vol": 20, "category": "Fruit"},
    "Avocado":       {"base": 200, "vol": 70, "category": "Fruit"},
    "Passion Fruit": {"base": 280, "vol": 100,"category": "Fruit"},
    "Guava":         {"base": 90,  "vol": 35, "category": "Fruit"},
}

MARKETS = {
    "Manning Market": {"type": "Wholesale", "multiplier": 1.00, "province": "Western"},
    "Dambulla":       {"type": "Wholesale", "multiplier": 0.85, "province": "Central"},
    "Narahenpita":    {"type": "Retail",    "multiplier": 1.30, "province": "Western"},
    "Nuwara Eliya":   {"type": "Wholesale", "multiplier": 0.90, "province": "Central"},
    "Colombo Local":  {"type": "Retail",    "multiplier": 1.40, "province": "Western"},
}


def get_season(month):
    if month in [10, 11, 12, 1]:
        return "Maha"
    elif month in [5, 6, 7, 8]:
        return "Yala"
    else:
        return "Off-Season"


def get_monsoon(month):
    """South-West monsoon May-Sep; North-East monsoon Oct-Jan"""
    return 1 if month in [5, 6, 7, 8, 9, 10, 12] else 0


def get_festive_bump(month, week_of_year):
    if month == 4 and week_of_year in [14, 15, 16]:
        return 1.20  # Sinhala/Tamil New Year
    if (month == 12 and week_of_year >= 51) or (month == 1 and week_of_year <= 2):
        return 1.15  # Christmas / New Year
    if month == 2 and week_of_year in [6, 7]:
        return 1.05  # Valentine week
    return 1.0


def inflation_multiplier(year, month):
    """
    Mirrors Sri Lanka's actual inflation trajectory:
      2019 stable, 2020 COVID effect, 2021 supply shortages,
      2022 economic crisis peak (~70% food inflation), 2023 stabilising, 2024 recovering
    """
    base = {2019: 1.00, 2020: 1.06, 2021: 1.15, 2023: 1.60, 2024: 1.75, 2025: 1.80}
    if year in base:
        return base[year]
    if year == 2022:
        if month <= 2:
            return 1.20
        elif month <= 6:
            return 1.20 + (month - 2) * 0.12
        elif month <= 9:
            return 1.78
        else:
            return 1.65
    return 1.0


def seasonal_factor(month, commodity):
    """
    Harvest season → higher supply → lower price.
    Off-harvest → scarcity → higher price.
    """
    harvest_months = {
        "Tomato": [1, 7], "Carrot": [2, 8], "Cabbage": [1, 7],
        "Beans": [12, 6], "Brinjal": [3, 9], "Snake Gourd": [6, 11],
        "Pumpkin": [5, 10], "Green Chilli": [4, 10], "Lime": [4, 10],
        "Leeks": [2, 8], "Capsicum": [3, 9], "Bitter Gourd": [6, 12],
        "Mango": [4, 5], "Papaya": [3, 9], "Banana": [5, 11],
        "Pineapple": [4, 9], "Watermelon": [3, 4], "Avocado": [7, 8],
        "Passion Fruit": [6, 12], "Guava": [8, 10],
    }
    peaks = harvest_months.get(commodity, [6])
    min_dist = min(abs(month - p) for p in peaks)
    # 0.75 at harvest peak → 1.30 at max scarcity (6 months away)
    return 0.75 + (min_dist / 6.0) * 0.55


def generate_records():
    records = []
    dates = pd.date_range("2019-01-07", "2025-12-29", freq="W-MON")

    for date in dates:
        year  = date.year
        month = date.month
        week  = int(date.isocalendar()[1])

        season   = get_season(month)
        monsoon  = get_monsoon(month)
        festive  = get_festive_bump(month, week)
        inf_mult = inflation_multiplier(year, month)

        for commodity, cinfo in COMMODITIES.items():
            base = cinfo["base"]
            vol  = cinfo["vol"]
            cat  = cinfo["category"]
            seas = seasonal_factor(month, commodity)

            for market, minfo in MARKETS.items():
                mtype    = minfo["type"]
                mmult    = minfo["multiplier"]
                province = minfo["province"]

                noise = np.random.normal(0, vol * 0.35)
                price = base * seas * mmult * inf_mult * festive + noise
                price = max(price, base * 0.30)
                price = round(price, 2)

                records.append({
                    "date":             date.strftime("%Y-%m-%d"),
                    "year":             year,
                    "month":            month,
                    "week":             week,
                    "quarter":          (month - 1) // 3 + 1,
                    "day_of_year":      date.dayofyear,
                    "commodity":        commodity,
                    "category":         cat,
                    "market":           market,
                    "price_type":       mtype,
                    "province":         province,
                    "season":           season,
                    "monsoon":          monsoon,
                    "is_festive":       1 if festive > 1.0 else 0,
                    "inflation_index":  round(inf_mult, 3),
                    "seasonal_factor":  round(seas, 3),
                    "price_lkr":        price,
                })

    return pd.DataFrame(records)


def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Generating dataset...")
    df = generate_records()
    print(f"  Records : {len(df):,}")
    print(f"  Dates   : {df['date'].min()} → {df['date'].max()}")
    print(f"  Items   : {sorted(df['commodity'].unique())}")
    out = "data/raw/srilanka_produce_prices.csv"
    df.to_csv(out, index=False)
    print(f"  Saved   → {out}")
    return df


if __name__ == "__main__":
    main()
