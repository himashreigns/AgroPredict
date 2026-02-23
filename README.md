# Sri Lanka Vegetable & Fruit Price Prediction System

## Overview
A machine learning system that predicts weekly wholesale and retail prices of vegetables and fruits at major Sri Lanka markets using **LightGBM** with **SHAP** explainability.

## Project Structure
```
Veg/
├── data/
│   ├── generate_dataset.py     # Generate realistic historical dataset
│   ├── preprocess.py           # Feature engineering & encoding
│   ├── raw/                    # Raw CSV 
│   └── processed/              # Processed CSV 
├── models/
│   ├── train.py                # LightGBM training + evaluation
│   ├── explain.py              # SHAP + PDP generation
│   └── *.pkl / metrics.json   # Saved artifacts 
├── notebook/
│   └── SriLanka_Price_Prediction.ipynb   # Main assignment notebook
├── assets/                     # Generated plots (SHAP, PDP, EDA)
├── app.py                      # Streamlit web application
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/generate_dataset.py
```

### 3. Preprocess Data
```bash
python data/preprocess.py
```

### 4. Train Model
```bash
python models/train.py
```

### 5. Generate XAI Plots
```bash
python models/explain.py
```

### 6. Launch Streamlit App
```bash
streamlit run app.py
```

### 7. Open Notebook
```bash
jupyter notebook notebook/SriLanka_Price_Prediction.ipynb
```

## Algorithm: LightGBM
**LightGBM** (Light Gradient Boosting Machine, Microsoft 2017) uses leaf-wise tree growth with:
- Gradient-based One-Side Sampling (GOSS) for speed
- Exclusive Feature Bundling (EFB) for memory efficiency
- Native categorical feature support
- First-class SHAP integration

## Dataset
- **20 commodities**: Tomato, Carrot, Cabbage, Beans, Brinjal, Snake Gourd, Pumpkin, Green Chilli, Lime, Leeks, Capsicum, Bitter Gourd, Mango, Papaya, Banana, Pineapple, Watermelon, Avocado, Passion Fruit, Guava
- **5 markets**: Manning Market, Dambulla, Narahenpita, Nuwara Eliya, Colombo Local
- **Features**: 27 features including lag (1w, 4w, 52w), rolling stats, seasonal/monsoon flags, inflation index, sinusoidal time encoding
- **Period**: 2019–2025 (weekly)

## Data Sources
- DOA / SHEP AgriInfoHub (infohub.doa.gov.lk)
- Central Bank of Sri Lanka — Daily Price Reports
- HARTI Food Information Bulletins
- Department of Census & Statistics — Weekly Retail Prices
- WFP / Humanitarian Data Exchange

## Results
Model is evaluated on a held-out test set (last 15% of time period) using:
- RMSE, MAE, R², MAPE

## Explainability
- **SHAP beeswarm** — feature impact across test samples
- **SHAP bar** — mean |SHAP| ranking
- **SHAP waterfall** — single prediction breakdown
- **PDP** — marginal effect of month, inflation, and lag features

## License
Educational use — University of Moratuwa
