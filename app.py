"""
Meme Stock Screener — app.py
Duke MQM Capstone | Team 19 | Spring 2026

Two modes:
  1. AUTO SCREENER  — scans a built-in watchlist of 30 small caps automatically
  2. CUSTOM SEARCH  — type any tickers and get scores instantly
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import time
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Meme Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #0F1B36; }
  [data-testid="stSidebar"]          { background-color: #0A1628; }
  [data-testid="stHeader"]           { background-color: #0F1B36; }
  h1,h2,h3,h4,h5,p,label,div        { color: #FFFFFF; }
  .metric-card {
    background: #1A2C52; border-radius: 10px;
    padding: 18px; text-align: center; border: 1px solid #2A3C62;
    margin-bottom: 8px;
  }
  .metric-value { font-size: 2rem; font-weight: 900; color: #0EA5A0; }
  .metric-label { font-size: 0.8rem; color: #8FA3BF; margin-top: 4px; }
  .stButton > button {
    background: #0EA5A0; color: white; border: none;
    border-radius: 8px; font-weight: 700; width: 100%;
    padding: 0.6rem 1.5rem; font-size: 1rem;
  }
  .stButton > button:hover { background: #0C7E7A; }
  .disclaimer {
    background: #1A2C52; border-left: 4px solid #0EA5A0;
    padding: 10px 14px; border-radius: 4px;
    font-size: 0.78rem; color: #8FA3BF;
  }
</style>
""", unsafe_allow_html=True)


def find_models_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "models"),
        os.path.join(os.getcwd(), "models"),
        "/mount/src/meme-screener/models",
        "models",
    ]
    for path in candidates:
        if (os.path.isdir(path) and
                os.path.exists(os.path.join(path, "scaler.pkl")) and
                os.path.exists(os.path.join(path, "xgb_best.json"))):
            return path
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_models():
    import joblib
    import xgboost as xgb
    model_dir = find_models_dir()
    if model_dir is None:
        return None, None, "Model files not found."
    try:
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(model_dir, "xgb_best.json"))
        return scaler, xgb_model, None
    except Exception as e:
        return None, None, str(e)


FEATURE_COLS = [
    'Open','High','Low','Close','Volume',
    'Return','Price_Chg_1d','Price_Chg_5d','Price_Chg_20d',
    'Volatility_5d','Volatility_20d','Vol_Ratio',
    'Return_Mean_20d','Return_Std_20d','Return_Zscore',
    'Volume_Change','AvgVol_20d','Vol_vs_Avg','Turnover',
    'Vol_Mean_20d','Vol_Std_20d','Vol_Zscore',
    'Daily_Range','Intraday_Move',
    'news_count','meme_keyword_count','meme_keyword_ratio',
    'news_avg_20d','news_vs_avg',
    'search_index','Search_WoW','Search_vs_Avg4w',
    'Search_vs_Avg12w','Search_52w_high','Search_acceleration',
    'Market_Cap','Short_Pct','Short_Ratio','Beta',
    'Float_Shares','Avg_Volume',
    'Revenue','Net_Income','Gross_Profit','Operating_Income',
    'Total_Assets','Total_Equity','Current_Assets',
    'Current_Liabilities','Shares_Outstanding',
    'Current_Ratio','Is_Profitable',
    'Revenue_QoQ','NetIncome_QoQ','Assets_QoQ',
    'earnings_report',
    'Sector_Communication Services','Sector_Consumer Cyclical',
    'Sector_Healthcare','Sector_Industrials','Sector_Real Estate',
    'Sector_Technology',
    'Ind_Auto Manufacturers','Ind_Communication Equipment',
    'Ind_Computer Hardware','Ind_Consumer Electronics',
    'Ind_Entertainment','Ind_Furnishings, Fixtures & Appliances',
    'Ind_Healthcare Plans','Ind_Internet Content & Information',
    'Ind_Internet Retail','Ind_Luxury Goods',
    'Ind_Real Estate Services','Ind_Scientific & Technical Instruments',
    'Ind_Software - Infrastructure','Ind_Specialty Business Services',
    'Ind_Specialty Retail',
]

AUTO_WATCHLIST = [
    "GME","AMC","KOSS","BB","NOK","MVIS",
    "QUBT","FFAI","DUO","DJT","SPCE","NKLA","RIDE",
    "BYND","PLUG","OPEN","HOOD","SOFI","UWMC",
    "PLTR","NVAX","SNDL","CLOV","WKHS","EXPR",
]


def fetch_features(ticker):
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=150)
        stock = yf.Ticker(ticker)
        h     = stock.history(start=start, end=end, auto_adjust=False)
        if len(h) < 25:
            return None
        h = h.reset_index()
        h.columns = [c.strip() for c in h.columns]
        h = h.rename(columns={"Date": "date"})
        h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)
        h = h.sort_values("date").copy()

        h["Return"]          = h["Close"].pct_change()
        h["Price_Chg_1d"]    = h["Close"].pct_change(1)
        h["Price_Chg_5d"]    = h["Close"].pct_change(5)
        h["Price_Chg_20d"]   = h["Close"].pct_change(20)
        h["Volatility_5d"]   = h["Return"].rolling(5).std()
        h["Volatility_20d"]  = h["Return"].rolling(20).std()
        h["Vol_Ratio"]       = h["Volatility_5d"] / h["Volatility_20d"].replace(0, np.nan)
        h["Return_Mean_20d"] = h["Return"].rolling(20).mean()
        h["Return_Std_20d"]  = h["Return"].rolling(20).std()
        h["Return_Zscore"]   = (h["Return"] - h["Return_Mean_20d"]) / h["Return_Std_20d"].replace(0, np.nan)
        h["Volume_Change"]   = h["Volume"].pct_change()
        h["AvgVol_20d"]      = h["Volume"].rolling(20).mean()
        h["Vol_vs_Avg"]      = h["Volume"] / h["AvgVol_20d"].replace(0, np.nan)
        h["Turnover"]        = h["Volume"] * h["Close"]
        h["Vol_Mean_20d"]    = h["Volume"].rolling(20).mean()
        h["Vol_Std_20d"]     = h["Volume"].rolling(20).std()
        h["Vol_Zscore"]      = (h["Volume"] - h["Vol_Mean_20d"]) / h["Vol_Std_20d"].replace(0, np.nan)
        h["Daily_Range"]     = (h["High"] - h["Low"]) / h["Open"].replace(0, np.nan)
        h["Intraday_Move"]   = (h["Close"] - h["Open"]) / h["Open"].replace(0, np.nan)

        latest = h.iloc[-1]
        info   = stock.info

        def safe(k, d=0):
            v = info.get(k, d)
            return v if v is not None else d

        row = {}
        for c in ["Open","High","Low","Close","Volume","Return","Price_Chg_1d",
                  "Price_Chg_5d","Price_Chg_20d","Volatility_5d","Volatility_20d",
                  "Vol_Ratio","Return_Mean_20d","Return_Std_20d","Return_Zscore",
                  "Volume_Change","AvgVol_20d","Vol_vs_Avg","Turnover",
                  "Vol_Mean_20d","Vol_Std_20d","Vol_Zscore","Daily_Range","Intraday_Move"]:
            row[c] = float(latest.get(c, 0) or 0)

        row.update({
            "Market_Cap":          safe("marketCap"),
            "Short_Pct":           safe("shortPercentOfFloat"),
            "Short_Ratio":         safe("shortRatio"),
            "Beta":                safe("beta", 1.0),
            "Float_Shares":        safe("floatShares"),
            "Avg_Volume":          safe("averageVolume"),
            "Revenue":             safe("totalRevenue"),
            "Net_Income":          safe("netIncomeToCommon"),
            "Gross_Profit":        safe("grossProfits"),
            "Operating_Income":    safe("operatingCashflow"),
            "Total_Assets":        safe("totalAssets"),
            "Total_Equity":        safe("bookValue"),
            "Current_Assets":      safe("currentRatio",1)*safe("totalCurrentLiabilities",0),
            "Current_Liabilities": safe("totalCurrentLiabilities"),
            "Shares_Outstanding":  safe("sharesOutstanding"),
            "Current_Ratio":       safe("currentRatio", 1.0),
            "Is_Profitable":       1 if safe("netIncomeToCommon") > 0 else 0,
            "Revenue_QoQ": 0, "NetIncome_QoQ": 0, "Assets_QoQ": 0, "earnings_report": 0,
            "news_count": 0, "meme_keyword_count": 0, "meme_keyword_ratio": 0,
            "news_avg_20d": 0, "news_vs_avg": 0,
            "search_index": 0, "Search_WoW": 0, "Search_vs_Avg4w": 1,
            "Search_vs_Avg12w": 1, "Search_52w_high": 0, "Search_acceleration": 0,
        })

        for c in [c for c in FEATURE_COLS if c.startswith("Sector_") or c.startswith("Ind_")]:
            row[c] = 0.0
        sc = f"Sector_{safe('sector','')}"
        ic = f"Ind_{safe('industry','')}"
        if sc in FEATURE_COLS: row[sc] = 1.0
        if ic in FEATURE_COLS: row[ic] = 1.0

        # Google Trends
        try:
            from pytrends.request import TrendReq
            pt = TrendReq(hl="en-US", tz=360)
            pt.build_payload([ticker], timeframe="today 12-m")
            time.sleep(0.5)
            df_tr = pt.interest_over_time()
            if not df_tr.empty and ticker in df_tr.columns:
                vals = df_tr[ticker].values.astype(float)
                row["search_index"]        = float(vals[-1])
                row["Search_WoW"]          = (vals[-1]-vals[-2])/max(vals[-2],1) if len(vals)>=2 else 0
                row["Search_vs_Avg4w"]     = vals[-1]/max(np.mean(vals[-4:]),1) if len(vals)>=4 else 1
                row["Search_vs_Avg12w"]    = vals[-1]/max(np.mean(vals[-12:]),1) if len(vals)>=12 else 1
                row["Search_52w_high"]     = 1.0 if vals[-1] >= np.max(vals[-52:]) else 0.0
                row["Search_acceleration"] = vals[-1]-2*vals[-2]+vals[-3] if len(vals)>=3 else 0
        except Exception:
            pass

        row["_ticker"]     = ticker
        row["_close"]      = float(latest.get("Close", 0) or 0)
        row["_vol_vs_avg"] = float(row.get("Vol_vs_Avg", 0) or 0)
        row["_date"]       = str(latest.get("date", ""))[:10]
        return row
    except Exception:
        return None


def score_row(row, scaler, xgb_model):
    X = np.array([[row.get(f, 0) or 0 for f in FEATURE_COLS]], dtype=np.float32)
    for j in range(X.shape[1]):
        lo, hi = np.percentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)
    return float(xgb_model.predict_proba(scaler.transform(X))[0][1])


def get_flag(prob, threshold):
    if prob >= threshold:  return "🔴 HIGH"
    elif prob >= 0.15:     return "🟡 WATCH"
    else:                  return "🟢 LOW"


def top_signals(row):
    parts = []
    if row.get("Search_vs_Avg4w", 0) > 1.5:
        parts.append(f"Search {row['Search_vs_Avg4w']:.1f}x")
    if row.get("Vol_vs_Avg", 0) > 2.0:
        parts.append(f"Volume {row['Vol_vs_Avg']:.1f}x")
    if row.get("Daily_Range", 0) > 0.05:
        parts.append(f"Range {row['Daily_Range']*100:.1f}%")
    if row.get("Short_Pct", 0) > 0.15:
        parts.append(f"Short {row['Short_Pct']*100:.1f}%")
    return " | ".join(parts[:3]) if parts else "No strong signals"


def run_screen(tickers, scaler, xgb_model, threshold, pb=None, txt=None):
    results = []
    for i, t in enumerate(tickers):
        if txt:  txt.markdown(f"⏳ Processing **{t}** ({i+1}/{len(tickers)})...")
        if pb:   pb.progress(i / len(tickers))
        row = fetch_features(t.upper().strip())
        if row is None:
            continue
        prob = score_row(row, scaler, xgb_model)
        results.append({
            "Ticker":     row["_ticker"],
            "Date":       row["_date"],
            "Price ($)":  round(row["_close"], 2),
            "Vol/Avg":    f"{row['_vol_vs_avg']:.1f}x",
            "Meme Score": prob,
            "Flag":       get_flag(prob, threshold),
            "Search 4w":  round(row.get("Search_vs_Avg4w", 0), 2),
            "Short %":    f"{row.get('Short_Pct',0)*100:.1f}%",
            "Daily Range":f"{row.get('Daily_Range',0)*100:.1f}%",
            "Top Signals":top_signals(row),
            "_prob":      prob,
        })
    if pb:  pb.progress(1.0)
    if txt: txt.empty()
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("_prob", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df


def make_chart(df, threshold):
    import plotly.graph_objects as go
    colors = ["#EF4444" if p >= threshold else "#F0B429" if p >= 0.15 else "#22C55E"
              for p in df["_prob"]]
    fig = go.Figure(go.Bar(
        x=df["Ticker"], y=df["_prob"],
        marker_color=colors,
        text=[f"{p:.1%}" for p in df["_prob"]],
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#0EA5A0",
                  annotation_text=f"Threshold ({threshold:.0%})",
                  annotation_font_color="#0EA5A0")
    fig.update_layout(
        paper_bgcolor="#0F1B36", plot_bgcolor="#1A2C52",
        font=dict(color="white"),
        yaxis=dict(tickformat=".0%", gridcolor="#2A3C62",
                   range=[0, max(df["_prob"].max() * 1.3 + 0.03, 0.1)]),
        xaxis=dict(gridcolor="#2A3C62"),
        height=420, margin=dict(t=30, b=40, l=40, r=20),
        showlegend=False,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider("Alert threshold", 0.10, 0.50, 0.26, 0.01)
    st.markdown("---")
    st.markdown("""
### 📊 Model Info
**Standalone XGBoost**
Holdout Precision: **26.1%**
Holdout AUC: **0.733**
True Positives: **37 / 273**
Specificity: **98.3%**

*Duke MQM Capstone — Team 19 — Spring 2026*
""")
    st.markdown("---")
    st.markdown("""<div class="disclaimer">
⚠️ Research tool only. Not financial advice.
~1 in 4 flags is a real meme episode.
All alerts require human review.
</div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Meme Stock Screener")
st.markdown("Predict which stocks are most at risk of a meme episode in the next 5 trading days.")
st.markdown("---")

scaler, xgb_model, err = load_models()
if err:
    st.error(f"❌ Could not load model: {err}")
    st.info("Make sure `models/scaler.pkl` and `models/xgb_best.json` are in a `models/` folder committed to your GitHub repo.")
    st.stop()

tab1, tab2 = st.tabs(["🔄 Auto Screener", "🔍 Search Any Ticker"])

# ── TAB 1: AUTO SCREENER ──────────────────────────────────────────────────────
with tab1:
    st.markdown("### Automatic Watchlist Scan")
    st.markdown(f"Scans **{len(AUTO_WATCHLIST)} pre-loaded stocks** — classic meme stocks, high short-interest small caps, and recent meme candidates — and ranks them by meme probability.")

    extra_input = st.text_input("Add extra tickers (optional, comma-separated)",
                                placeholder="e.g. TSLA, RIVN, BBAI")
    run_auto = st.button("🚀 Run Auto Screener", key="auto")

    with st.expander("📋 View default watchlist"):
        st.write(", ".join(AUTO_WATCHLIST))

    if run_auto:
        tickers = list(AUTO_WATCHLIST)
        if extra_input.strip():
            extras = [t.strip().upper() for t in extra_input.split(",") if t.strip()]
            tickers = list(dict.fromkeys(tickers + extras))

        st.markdown("---")
        pb  = st.progress(0)
        txt = st.empty()
        df  = run_screen(tickers, scaler, xgb_model, threshold, pb, txt)

        if df.empty:
            st.error("Could not retrieve data. Check your internet connection.")
        else:
            high  = (df["Flag"] == "🔴 HIGH").sum()
            watch = (df["Flag"] == "🟡 WATCH").sum()
            low   = (df["Flag"] == "🟢 LOW").sum()
            top   = df.iloc[0]

            st.markdown("### 📊 Results")
            cols = st.columns(5)
            for col, val, label, color in [
                (cols[0], len(df),       "Screened",         "#0EA5A0"),
                (cols[1], high,          "🔴 HIGH Alerts",   "#EF4444"),
                (cols[2], watch,         "🟡 WATCH",         "#F0B429"),
                (cols[3], low,           "🟢 LOW Risk",      "#22C55E"),
                (cols[4], top["Ticker"], f"Top ({top['Meme Score']:.1%})", "#0EA5A0"),
            ]:
                col.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:{color}">{val}</div>
                  <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.plotly_chart(make_chart(df, threshold), use_container_width=True)

            st.markdown("### 📋 Full Ranked Table")
            show = df.drop(columns=["_prob"]).copy()
            show["Meme Score"] = show["Meme Score"].apply(lambda x: f"{x:.1%}")
            st.dataframe(show, use_container_width=True,
                         height=min(60 + len(show)*38, 650))

            st.markdown(f"### 🔎 Top Candidate: {top['Ticker']}")
            st.markdown(f"**Score:** {top['Meme Score']}  |  {top['Flag']}  |  {top['Date']}")
            st.markdown(f"**Key signals:** {top['Top Signals']}")
            st.markdown("---")
            st.markdown("""<div class="disclaimer">
⚠️ Research tool only. Not financial advice.
At 26% precision, ~3 in 4 flagged stocks will NOT experience a meme episode.
</div>""", unsafe_allow_html=True)

# ── TAB 2: CUSTOM SEARCH ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### Search Any Tickers")
    st.markdown("The model works on any publicly traded US stock — not just known meme stocks.")

    ticker_input = st.text_area("Tickers (comma or newline separated)",
                                placeholder="GME, AMC, QUBT, PLTR", height=80)
    run_custom = st.button("🔍 Score These Tickers", key="custom")

    if run_custom:
        if not ticker_input.strip():
            st.warning("Enter at least one ticker.")
        else:
            tickers = [t.strip().upper()
                       for t in ticker_input.replace("\n", ",").split(",")
                       if t.strip()][:20]
            st.markdown("---")
            pb  = st.progress(0)
            txt = st.empty()
            df  = run_screen(tickers, scaler, xgb_model, threshold, pb, txt)

            if df.empty:
                st.error("Could not retrieve data. Check tickers and try again.")
            else:
                st.plotly_chart(make_chart(df, threshold), use_container_width=True)
                show = df.drop(columns=["_prob"]).copy()
                show["Meme Score"] = show["Meme Score"].apply(lambda x: f"{x:.1%}")
                st.dataframe(show, use_container_width=True,
                             height=min(60 + len(show)*38, 500))
                st.markdown("---")
                st.markdown("""<div class="disclaimer">
⚠️ Research tool only. Not financial advice.
</div>""", unsafe_allow_html=True)
