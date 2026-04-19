"""
app.py — Meme Stock Screener
Duke MQM Capstone | Team 19 | Spring 2026

Streamlit web app that screens any list of stocks for meme episode risk
using a trained XGBoost model (26% holdout precision, AUC 0.733).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Meme Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1B36; }
    .stApp { background-color: #0F1B36; }
    
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #FFFFFF !important;
    }
    
    .metric-card {
        background: #1A2C52;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2A3C62;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0EA5A0 !important;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8FA3BF !important;
        margin-top: 4px;
    }

    .flag-high  { color: #EF4444 !important; font-weight: bold; }
    .flag-watch { color: #F0B429 !important; font-weight: bold; }
    .flag-low   { color: #22C55E !important; font-weight: bold; }

    .stButton > button {
        background-color: #0EA5A0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0C7E7A;
    }

    .disclaimer {
        background: #1A2C52;
        border-left: 4px solid #0EA5A0;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #8FA3BF !important;
    }
    
    div[data-testid="stDataFrame"] {
        background: #1A2C52;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached so they only load once) ───────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    scaler    = joblib.load(os.path.join(base, "models", "scaler.pkl"))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(base, "models", "xgb_best.json"))
    return scaler, xgb_model


# ── Default watchlist ─────────────────────────────────────────────
DEFAULT_WATCHLIST = [
    "GME", "AMC", "KOSS", "MVIS", "QUBT", "FFAI", "DUO",
    "BBBY", "CLOV", "BB", "NOK", "DJT", "NVDA", "PLTR",
    "SOFI", "HOOD", "UWMC", "SPCE", "NKLA", "RIDE",
]

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    threshold = st.slider(
        "Alert threshold",
        min_value=0.10,
        max_value=0.50,
        value=0.26,
        step=0.01,
        help="Stocks scoring above this are flagged HIGH. Model was optimised at 0.26 (26% precision)."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    **Model:** Standalone XGBoost  
    **Holdout Precision:** 26.1%  
    **Holdout AUC:** 0.733  
    **True Positives:** 37 / 273  
    **Specificity:** 98.3%  
    
    *Duke MQM Capstone — Team 19*  
    *Spring 2026*
    """)

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ This tool is an early warning screener, not a trading signal. 
    All alerts require human review. At 26% precision, roughly 1 in 4 
    flagged stocks will experience a meme episode.
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────
st.markdown("# 📈 Meme Stock Screener")
st.markdown("**Predict which stocks are most at risk of a meme episode in the next 5 trading days.**")
st.markdown("---")

# ── Input section ─────────────────────────────────────────────────
st.markdown("### Enter tickers to screen")

col1, col2 = st.columns([3, 1])

with col1:
    ticker_input = st.text_area(
        "Type tickers separated by commas, or use the default watchlist below",
        placeholder="e.g. GME, AMC, QUBT, MVIS, PLTR",
        height=80,
    )

with col2:
    use_default = st.checkbox("Use default watchlist", value=False)
    st.caption(f"{len(DEFAULT_WATCHLIST)} pre-loaded meme-susceptible small caps")

run_button = st.button("🔍 Run Screener", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────
if run_button:
    from predict_utils import score_tickers, FEATURE_COLS

    # Parse tickers
    if use_default:
        tickers = DEFAULT_WATCHLIST
    elif ticker_input.strip():
        tickers = [t.strip().upper() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
    else:
        st.warning("Please enter at least one ticker or check 'Use default watchlist'.")
        st.stop()

    if len(tickers) > 30:
        st.warning("Maximum 30 tickers per run to avoid rate limiting. Using first 30.")
        tickers = tickers[:30]

    st.markdown("---")
    st.markdown(f"### Screening {len(tickers)} stocks...")

    # Progress
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    results_placeholder = st.empty()

    ticker_results = []
    errors         = []

    scaler, xgb_model = load_models()

    for i, ticker in enumerate(tickers):
        progress_bar.progress((i) / len(tickers))
        status_text.markdown(f"⏳ Processing **{ticker}** ({i+1}/{len(tickers)})...")

        try:
            from predict_utils import build_feature_row, _get_top_signals, FEATURE_COLS
            row = build_feature_row(ticker)

            if row is None:
                errors.append(ticker)
                continue

            close      = row.pop('_close')
            vol_vs_avg = row.pop('_vol_vs_avg')
            date       = row.pop('_date')
            tkr        = row.pop('_ticker')

            X = np.array([[row.get(f, 0) for f in FEATURE_COLS]], dtype=np.float32)
            for j in range(X.shape[1]):
                q01 = np.percentile(X[:, j], 1)
                q99 = np.percentile(X[:, j], 99)
                X[:, j] = np.clip(X[:, j], q01, q99)

            X_scaled = scaler.transform(X)
            prob     = float(xgb_model.predict_proba(X_scaled)[0][1])

            if prob >= threshold:
                flag       = "🔴 HIGH"
                flag_color = "#EF4444"
            elif prob >= 0.15:
                flag       = "🟡 WATCH"
                flag_color = "#F0B429"
            else:
                flag       = "🟢 LOW"
                flag_color = "#22C55E"

            ticker_results.append({
                'Ticker':             tkr,
                'Date':               date,
                'Price ($)':          round(close, 2),
                'Vol / Avg':          f"{vol_vs_avg:.1f}x",
                'Meme Score':         prob,
                'Flag':               flag,
                'Top Signals':        _get_top_signals(row),
                'Search Spike':       round(row.get('Search_vs_Avg4w', 0), 2),
                'Volume Spike':       round(row.get('Vol_vs_Avg', 0), 2),
                'Short Interest':     f"{row.get('Short_Pct', 0)*100:.1f}%",
                'Daily Range':        f"{row.get('Daily_Range', 0)*100:.1f}%",
                'News vs Avg':        round(row.get('news_vs_avg', 0), 2),
                '_prob':              prob,
                '_flag_color':        flag_color,
            })

        except Exception as e:
            errors.append(f"{ticker} ({str(e)[:40]})")
            continue

    progress_bar.progress(1.0)
    status_text.empty()

    if not ticker_results:
        st.error("Could not retrieve data for any of the requested tickers. Please check the tickers and try again.")
        st.stop()

    # Sort by score
    ticker_results.sort(key=lambda x: x['_prob'], reverse=True)
    df_display = pd.DataFrame(ticker_results)

    # ── Summary metrics ───────────────────────────────────────────
    high_count  = sum(1 for r in ticker_results if "HIGH"  in r['Flag'])
    watch_count = sum(1 for r in ticker_results if "WATCH" in r['Flag'])
    low_count   = sum(1 for r in ticker_results if "LOW"   in r['Flag'])
    top_score   = ticker_results[0]['_prob'] if ticker_results else 0
    top_ticker  = ticker_results[0]['Ticker'] if ticker_results else "—"

    st.markdown("---")
    st.markdown("### 📊 Results Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(ticker_results)}</div>
            <div class="metric-label">Stocks Screened</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#EF4444 !important">{high_count}</div>
            <div class="metric-label">🔴 HIGH Alerts</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#F0B429 !important">{watch_count}</div>
            <div class="metric-label">🟡 WATCH</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#22C55E !important">{low_count}</div>
            <div class="metric-label">🟢 LOW Risk</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_ticker}</div>
            <div class="metric-label">Top Candidate ({top_score:.1%})</div>
        </div>""", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Meme Probability — All Stocks Ranked")

    colors = []
    for r in ticker_results:
        if r['_prob'] >= threshold:
            colors.append("#EF4444")
        elif r['_prob'] >= 0.15:
            colors.append("#F0B429")
        else:
            colors.append("#22C55E")

    fig = go.Figure(go.Bar(
        x=[r['Ticker'] for r in ticker_results],
        y=[r['_prob'] for r in ticker_results],
        marker_color=colors,
        text=[f"{r['_prob']:.1%}" for r in ticker_results],
        textposition='outside',
        textfont=dict(color='white', size=11),
    ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#0EA5A0",
        annotation_text=f"Alert threshold ({threshold:.0%})",
        annotation_font_color="#0EA5A0",
    )

    fig.update_layout(
        paper_bgcolor="#0F1B36",
        plot_bgcolor="#1A2C52",
        font=dict(color="white"),
        yaxis=dict(
            title="Meme Episode Probability",
            tickformat=".0%",
            gridcolor="#2A3C62",
            range=[0, max(r['_prob'] for r in ticker_results) * 1.25 + 0.05],
        ),
        xaxis=dict(title="", gridcolor="#2A3C62"),
        showlegend=False,
        height=420,
        margin=dict(t=30, b=40, l=40, r=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Ranked table ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Full Ranked Results")

    display_cols = ['Ticker', 'Date', 'Price ($)', 'Vol / Avg', 'Meme Score',
                    'Flag', 'Search Spike', 'Volume Spike', 'Short Interest',
                    'Daily Range', 'News vs Avg', 'Top Signals']

    df_table = df_display[display_cols].copy()
    df_table['Meme Score'] = df_table['Meme Score'].apply(lambda x: f"{x:.1%}")
    df_table.index = range(1, len(df_table) + 1)

    st.dataframe(
        df_table,
        use_container_width=True,
        height=min(50 + len(df_table) * 38, 600),
    )

    # ── Top stock deep dive ───────────────────────────────────────
    if ticker_results:
        st.markdown("---")
        top = ticker_results[0]
        st.markdown(f"### 🔎 Signal Deep Dive — {top['Ticker']}")
        st.markdown(f"**Meme Score: {top['_prob']:.1%}** | {top['Flag']} | As of {top['Date']}")

        signal_data = {
            'Google Search Spike (4w avg)': top['Search Spike'],
            'Volume vs 20d Avg':            top['Volume Spike'],
            'Short Interest (% float)':     float(str(top['Short Interest']).replace('%', '')) / 100,
            'Daily Intraday Range':         float(str(top['Daily Range']).replace('%', '')) / 100,
            'News vs 20d Avg':              top['News vs Avg'],
        }

        normal_benchmarks = {
            'Google Search Spike (4w avg)': 1.0,
            'Volume vs 20d Avg':            1.0,
            'Short Interest (% float)':     0.05,
            'Daily Intraday Range':         0.02,
            'News vs 20d Avg':              1.0,
        }

        sig_fig = go.Figure()
        sig_labels = list(signal_data.keys())
        sig_values = list(signal_data.values())
        benchmarks = [normal_benchmarks[k] for k in sig_labels]

        sig_fig.add_trace(go.Bar(
            name=top['Ticker'],
            x=sig_labels,
            y=sig_values,
            marker_color="#0EA5A0",
            text=[f"{v:.2f}" for v in sig_values],
            textposition='outside',
            textfont=dict(color='white'),
        ))

        sig_fig.add_trace(go.Scatter(
            name="Normal baseline",
            x=sig_labels,
            y=benchmarks,
            mode='markers',
            marker=dict(symbol='line-ew', size=20, color='#F0B429',
                        line=dict(width=3, color='#F0B429')),
        ))

        sig_fig.update_layout(
            paper_bgcolor="#0F1B36",
            plot_bgcolor="#1A2C52",
            font=dict(color="white"),
            yaxis=dict(gridcolor="#2A3C62"),
            xaxis=dict(gridcolor="#2A3C62"),
            legend=dict(bgcolor="#1A2C52", bordercolor="#2A3C62"),
            height=360,
            margin=dict(t=20, b=80, l=40, r=20),
            barmode='group',
        )

        st.plotly_chart(sig_fig, use_container_width=True)
        st.caption("Teal bars = current values. Yellow markers = normal baseline. "
                   "Bars significantly above baseline indicate elevated meme risk signals.")

    # ── Errors ────────────────────────────────────────────────────
    if errors:
        st.markdown("---")
        st.warning(f"Could not retrieve data for: {', '.join(errors)}")

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This screener is a research tool built for the Duke MQM Capstone project. 
    It is not financial advice. At 26% precision, approximately 3 in 4 flagged stocks will NOT experience a meme episode. 
    All alerts require human review before any action is taken. Past model performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Landing state ─────────────────────────────────────────────
    st.markdown("### How it works")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **1. Enter tickers**  
        Type any stock tickers you want to screen, or use the default watchlist of 20 known meme-susceptible small caps.
        """)
    with c2:
        st.markdown("""
        **2. Model scores each stock**  
        For each ticker, we pull live market data, Google Trends, and news, then compute 77 features and run them through the trained XGBoost model.
        """)
    with c3:
        st.markdown("""
        **3. Ranked output**  
        Stocks are ranked by meme episode probability with a signal breakdown showing what is driving each score.
        """)

    st.markdown("---")
    st.markdown("### Model performance (holdout set — 4 stocks never seen in training)")

    perf_data = {
        'Metric': ['Precision', 'Recall', 'True Positives Caught', 'Specificity', 'AUC'],
        'Value':  ['26.1%', '13.6%', '37 out of 273', '98.3%', '0.733'],
        'Meaning': [
            '1 in 4 flagged stocks actually goes meme',
            'Catches 14% of all real episodes',
            'Best balance of precision and recall',
            'Correctly ignores 98% of normal days',
            'Strong discriminative ability',
        ]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("👆 Enter tickers above and click **Run Screener** to get started.")
