# Meme Stock Screener

**Duke MQM Capstone — Team 19 — Spring 2026**

A live meme stock screening tool powered by XGBoost (26% holdout precision, AUC 0.733).
Enter any stock tickers and get a ranked list of meme episode probabilities for the next 5 trading days.

---

## What it does

- Pulls live market data, Google Trends, and news for any ticker
- Computes 77 features the model was trained on
- Scores each stock with the trained XGBoost model
- Returns a ranked table with a bar chart and signal breakdown

---

## Files

```
meme-screener/
├── app.py                  # Main Streamlit app
├── predict_utils.py        # Data collection + feature engineering
├── requirements.txt        # Dependencies
├── README.md               # This file
└── models/
    ├── xgb_best.json       # Trained XGBoost model (seed=42)
    └── scaler.pkl          # StandardScaler fitted on training data
```

---

## Deploy to Streamlit Cloud (free)

1. Push this entire folder to a GitHub repository
2. Go to streamlit.io/cloud and sign in with GitHub
3. Click **New app**
4. Select your repository, set main file to `app.py`
5. Click **Deploy** — live URL in ~3 minutes

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Model details

| Metric | Value |
|--------|-------|
| Architecture | Standalone XGBoost |
| Holdout Precision | 26.1% |
| Holdout Recall | 13.6% |
| Holdout AUC | 0.733 |
| True Positives | 37 / 273 |
| Specificity | 98.3% |
| Seed | 42 (fully reproducible) |

**Why XGBoost over ensembles?**
The neural network collapsed on holdout stocks (val AUC 0.79 → holdout AUC 0.59), dragging all ensemble methods down. Standalone XGBoost achieved the highest holdout precision AND highest holdout AUC of all architectures tested.

---

## Disclaimer

This is a research tool. Not financial advice. At 26% precision, roughly 3 in 4 flagged stocks will NOT experience a meme episode. All alerts require human review.
