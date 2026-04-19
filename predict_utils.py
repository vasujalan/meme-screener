"""
predict_utils.py
----------------
Data collection and feature engineering for the Meme Stock Screener.
Pulls live market data, news, and Google Trends for any ticker,
then computes all 77 features the XGBoost model expects.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── The exact 77 features the scaler and XGBoost were trained on ──
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Return', 'Price_Chg_1d', 'Price_Chg_5d', 'Price_Chg_20d',
    'Volatility_5d', 'Volatility_20d', 'Vol_Ratio',
    'Return_Mean_20d', 'Return_Std_20d', 'Return_Zscore',
    'Volume_Change', 'AvgVol_20d', 'Vol_vs_Avg', 'Turnover',
    'Vol_Mean_20d', 'Vol_Std_20d', 'Vol_Zscore',
    'Daily_Range', 'Intraday_Move',
    'news_count', 'meme_keyword_count', 'meme_keyword_ratio',
    'news_avg_20d', 'news_vs_avg',
    'search_index', 'Search_WoW', 'Search_vs_Avg4w',
    'Search_vs_Avg12w', 'Search_52w_high', 'Search_acceleration',
    'Market_Cap', 'Short_Pct', 'Short_Ratio', 'Beta',
    'Float_Shares', 'Avg_Volume',
    'Revenue', 'Net_Income', 'Gross_Profit', 'Operating_Income',
    'Total_Assets', 'Total_Equity', 'Current_Assets',
    'Current_Liabilities', 'Shares_Outstanding',
    'Current_Ratio', 'Is_Profitable',
    'Revenue_QoQ', 'NetIncome_QoQ', 'Assets_QoQ',
    'earnings_report',
    'Sector_Communication Services', 'Sector_Consumer Cyclical',
    'Sector_Healthcare', 'Sector_Industrials', 'Sector_Real Estate',
    'Sector_Technology',
    'Ind_Auto Manufacturers', 'Ind_Communication Equipment',
    'Ind_Computer Hardware', 'Ind_Consumer Electronics',
    'Ind_Entertainment', 'Ind_Furnishings, Fixtures & Appliances',
    'Ind_Healthcare Plans', 'Ind_Internet Content & Information',
    'Ind_Internet Retail', 'Ind_Luxury Goods',
    'Ind_Real Estate Services', 'Ind_Scientific & Technical Instruments',
    'Ind_Software - Infrastructure', 'Ind_Specialty Business Services',
    'Ind_Specialty Retail',
]

MEME_KEYWORDS = [
    'short squeeze', 'meme stock', 'meme', 'reddit', 'wallstreetbets',
    'wsb', 'retail investors', 'diamond hands', 'to the moon', 'apes',
    'yolo', 'hodl', 'roaring kitty', 'pump', 'squeeze', 'moon',
    'paper hands', 'stonk', 'short seller', 'heavily shorted',
    'gamma squeeze', 'fomo',
]

LOOKBACK = 10


def collect_market_data(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    """Download OHLCV and compute all price/volume features."""
    import yfinance as yf

    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end, auto_adjust=False)

    if len(hist) == 0:
        return pd.DataFrame()

    hist = hist.reset_index()
    hist.columns = [c.strip() for c in hist.columns]
    hist = hist.rename(columns={'Date': 'date'})
    hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)
    hist['Ticker'] = ticker

    h = hist.sort_values('date').copy()

    # Returns and price changes
    h['Return']         = h['Close'].pct_change()
    h['Price_Chg_1d']   = h['Close'].pct_change(1)
    h['Price_Chg_5d']   = h['Close'].pct_change(5)
    h['Price_Chg_20d']  = h['Close'].pct_change(20)

    # Volatility
    h['Volatility_5d']  = h['Return'].rolling(5).std()
    h['Volatility_20d'] = h['Return'].rolling(20).std()
    h['Vol_Ratio']      = h['Volatility_5d'] / h['Volatility_20d'].replace(0, np.nan)

    # Return stats
    h['Return_Mean_20d'] = h['Return'].rolling(20).mean()
    h['Return_Std_20d']  = h['Return'].rolling(20).std()
    h['Return_Zscore']   = (h['Return'] - h['Return_Mean_20d']) / h['Return_Std_20d'].replace(0, np.nan)

    # Volume
    h['Volume_Change'] = h['Volume'].pct_change()
    h['AvgVol_20d']    = h['Volume'].rolling(20).mean()
    h['Vol_vs_Avg']    = h['Volume'] / h['AvgVol_20d'].replace(0, np.nan)
    h['Turnover']      = h['Volume'] * h['Close']
    h['Vol_Mean_20d']  = h['Volume'].rolling(20).mean()
    h['Vol_Std_20d']   = h['Volume'].rolling(20).std()
    h['Vol_Zscore']    = (h['Volume'] - h['Vol_Mean_20d']) / h['Vol_Std_20d'].replace(0, np.nan)

    # Intraday
    h['Daily_Range']   = (h['High'] - h['Low']) / h['Open'].replace(0, np.nan)
    h['Intraday_Move'] = (h['Close'] - h['Open']) / h['Open'].replace(0, np.nan)

    return h


def collect_company_info(ticker: str) -> dict:
    """Get static company info from yfinance."""
    import yfinance as yf

    info = yf.Ticker(ticker).info

    def safe(key, default=0):
        val = info.get(key, default)
        return val if val is not None else default

    sector   = safe('sector', '')
    industry = safe('industry', '')

    return {
        'Market_Cap':         safe('marketCap'),
        'Short_Pct':          safe('shortPercentOfFloat'),
        'Short_Ratio':        safe('shortRatio'),
        'Beta':               safe('beta', 1.0),
        'Float_Shares':       safe('floatShares'),
        'Avg_Volume':         safe('averageVolume'),
        'Revenue':            safe('totalRevenue'),
        'Net_Income':         safe('netIncomeToCommon'),
        'Gross_Profit':       safe('grossProfits'),
        'Operating_Income':   safe('operatingCashflow'),
        'Total_Assets':       safe('totalAssets'),
        'Total_Equity':       safe('bookValue'),
        'Current_Assets':     safe('currentRatio', 1) * safe('totalCurrentLiabilities', 0),
        'Current_Liabilities':safe('totalCurrentLiabilities'),
        'Shares_Outstanding': safe('sharesOutstanding'),
        'Current_Ratio':      safe('currentRatio', 1.0),
        'Is_Profitable':      1 if safe('netIncomeToCommon') > 0 else 0,
        'Revenue_QoQ':        0,
        'NetIncome_QoQ':      0,
        'Assets_QoQ':         0,
        'earnings_report':    0,
        'sector':             sector,
        'industry':           industry,
    }


def collect_news(ticker: str, days: int = 30) -> pd.DataFrame:
    """Collect news article counts using GNews."""
    try:
        from gnews import GNews
        gn = GNews(language='en', country='US', max_results=100,
                   start_date=(datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d'),
                   end_date=datetime.today().strftime('%Y-%m-%d'))
        articles = gn.get_news(ticker)

        rows = []
        for a in articles:
            try:
                pub_date = pd.to_datetime(a.get('published date', ''), utc=True).tz_localize(None)
            except Exception:
                continue
            title = (a.get('title', '') or '').lower()
            kw_count = sum(1 for kw in MEME_KEYWORDS if kw in title)
            rows.append({'date': pub_date.normalize(), 'has_meme_kw': kw_count > 0, 'kw_count': kw_count})

        if not rows:
            return pd.DataFrame(columns=['date', 'news_count', 'meme_keyword_count', 'meme_keyword_ratio'])

        df = pd.DataFrame(rows)
        daily = df.groupby('date').agg(
            news_count=('has_meme_kw', 'count'),
            meme_keyword_count=('kw_count', 'sum')
        ).reset_index()
        daily['meme_keyword_ratio'] = daily['meme_keyword_count'] / daily['news_count'].replace(0, np.nan)
        daily['meme_keyword_ratio'] = daily['meme_keyword_ratio'].fillna(0)
        return daily

    except Exception:
        return pd.DataFrame(columns=['date', 'news_count', 'meme_keyword_count', 'meme_keyword_ratio'])


def collect_trends(ticker: str) -> pd.DataFrame:
    """Fetch Google Trends and compute search features."""
    try:
        from pytrends.request import TrendReq
        import time

        pt = TrendReq(hl='en-US', tz=360)
        pt.build_payload([ticker], timeframe='today 12-m')
        time.sleep(1)
        df = pt.interest_over_time()

        if df.empty or ticker not in df.columns:
            return pd.DataFrame()

        df = df[[ticker]].reset_index()
        df.columns = ['date', 'search_index']
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        df['Search_WoW']        = df['search_index'].pct_change(1)
        df['Search_vs_Avg4w']   = df['search_index'] / df['search_index'].rolling(4).mean().replace(0, np.nan)
        df['Search_vs_Avg12w']  = df['search_index'] / df['search_index'].rolling(12).mean().replace(0, np.nan)
        rolling_max             = df['search_index'].rolling(52).max()
        df['Search_52w_high']   = (df['search_index'] >= rolling_max).astype(float)
        df['Search_acceleration'] = df['Search_WoW'].diff(1)

        return df

    except Exception:
        return pd.DataFrame()


def build_feature_row(ticker: str, status_callback=None) -> dict:
    """
    Pull all data for a ticker and return a dict of the 77 features
    for the most recent available day.
    Returns None if data collection fails.
    """

    def update(msg):
        if status_callback:
            status_callback(msg)

    update(f"Downloading market data for {ticker}...")
    market = collect_market_data(ticker, lookback_days=120)
    if market.empty or len(market) < LOOKBACK + 5:
        return None

    update(f"Fetching company info for {ticker}...")
    info = collect_company_info(ticker)

    update(f"Collecting news for {ticker}...")
    news = collect_news(ticker, days=60)

    update(f"Fetching Google Trends for {ticker}...")
    trends = collect_trends(ticker)

    # Merge news onto market data
    market['date_only'] = market['date'].dt.normalize()
    if not news.empty:
        news['date'] = pd.to_datetime(news['date']).dt.normalize()
        market = market.merge(news, left_on='date_only', right_on='date', how='left', suffixes=('', '_news'))
    for col in ['news_count', 'meme_keyword_count', 'meme_keyword_ratio']:
        if col not in market.columns:
            market[col] = 0
    market[['news_count', 'meme_keyword_count', 'meme_keyword_ratio']] = \
        market[['news_count', 'meme_keyword_count', 'meme_keyword_ratio']].fillna(0)

    # Rolling news avg
    market['news_avg_20d'] = market['news_count'].rolling(20, min_periods=1).mean()
    market['news_vs_avg']  = market['news_count'] / market['news_avg_20d'].replace(0, np.nan)
    market['news_vs_avg']  = market['news_vs_avg'].fillna(0)

    # Merge trends — forward fill weekly to daily
    if not trends.empty:
        market = market.merge(
            trends[['date', 'search_index', 'Search_WoW', 'Search_vs_Avg4w',
                    'Search_vs_Avg12w', 'Search_52w_high', 'Search_acceleration']],
            left_on='date_only', right_on='date', how='left', suffixes=('', '_tr')
        )
        for col in ['search_index', 'Search_WoW', 'Search_vs_Avg4w',
                    'Search_vs_Avg12w', 'Search_52w_high', 'Search_acceleration']:
            if col in market.columns:
                market[col] = market[col].ffill()

    for col in ['search_index', 'Search_WoW', 'Search_vs_Avg4w',
                'Search_vs_Avg12w', 'Search_52w_high', 'Search_acceleration']:
        if col not in market.columns:
            market[col] = 0
    market[['search_index', 'Search_WoW', 'Search_vs_Avg4w', 'Search_vs_Avg12w',
            'Search_52w_high', 'Search_acceleration']] = \
        market[['search_index', 'Search_WoW', 'Search_vs_Avg4w', 'Search_vs_Avg12w',
                'Search_52w_high', 'Search_acceleration']].fillna(0)

    # Add company static info
    for k, v in info.items():
        if k not in ('sector', 'industry'):
            market[k] = v

    # Sector / Industry dummies — set all to 0 then flag the right one
    sector_cols   = [c for c in FEATURE_COLS if c.startswith('Sector_')]
    industry_cols = [c for c in FEATURE_COLS if c.startswith('Ind_')]
    for col in sector_cols + industry_cols:
        market[col] = 0.0

    sector_col = f"Sector_{info['sector']}"
    if sector_col in FEATURE_COLS:
        market[sector_col] = 1.0

    ind_col = f"Ind_{info['industry']}"
    if ind_col in FEATURE_COLS:
        market[ind_col] = 1.0

    # Take the most recent row
    market = market.sort_values('date').reset_index(drop=True)
    latest = market.iloc[-1]

    row = {}
    for col in FEATURE_COLS:
        val = latest.get(col, 0)
        row[col] = float(val) if pd.notna(val) else 0.0

    # Record current price and volume for display
    row['_close']      = float(latest.get('Close', 0))
    row['_vol_vs_avg'] = float(latest.get('Vol_vs_Avg', 0) or 0)
    row['_date']       = str(latest.get('date_only', latest.get('date', '')))[: 10]
    row['_ticker']     = ticker

    return row


def score_tickers(tickers: list, scaler, xgb_model, status_callback=None) -> pd.DataFrame:
    """
    Run the full pipeline for a list of tickers.
    Returns a DataFrame ranked by meme probability.
    """
    import joblib

    results = []

    for ticker in tickers:
        try:
            row = build_feature_row(ticker.upper().strip(), status_callback)
            if row is None:
                continue

            # Extract display fields
            close      = row.pop('_close')
            vol_vs_avg = row.pop('_vol_vs_avg')
            date       = row.pop('_date')
            tkr        = row.pop('_ticker')

            # Build feature vector in exact order
            X = np.array([[row.get(f, 0) for f in FEATURE_COLS]], dtype=np.float32)

            # Winsorize at 1st/99th (same as training)
            for i in range(X.shape[1]):
                q01, q99 = np.percentile(X[:, i], [1, 99])
                X[:, i]  = np.clip(X[:, i], q01, q99)

            # Scale
            X_scaled = scaler.transform(X)

            # Score
            prob = float(xgb_model.predict_proba(X_scaled)[0][1])

            # Flag
            if prob >= 0.30:
                flag = "🔴 HIGH"
            elif prob >= 0.15:
                flag = "🟡 WATCH"
            else:
                flag = "🟢 LOW"

            # Top signals — which features are most elevated
            top_signals = _get_top_signals(row)

            results.append({
                'Ticker':      tkr,
                'Date':        date,
                'Price':       round(close, 2),
                'Vol / 20d Avg': f"{vol_vs_avg:.1f}x",
                'Meme Probability': prob,
                'Signal Flag': flag,
                'Top Signals': top_signals,
                'Search Spike (4w)':  round(row.get('Search_vs_Avg4w', 0), 2),
                'Volume Spike':       round(row.get('Vol_vs_Avg', 0), 2),
                'Volatility (5d)':    round(row.get('Volatility_5d', 0), 4),
                'Short Interest %':   round(row.get('Short_Pct', 0) * 100, 1),
                'Daily Range':        round(row.get('Daily_Range', 0), 4),
                'News vs Avg':        round(row.get('news_vs_avg', 0), 2),
            })

        except Exception as e:
            if status_callback:
                status_callback(f"⚠️ Could not process {ticker}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values('Meme Probability', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # rank starts at 1
    return df


def _get_top_signals(row: dict) -> str:
    """Return a short string describing the top 3 elevated signals."""
    signals = []

    sv = row.get('Search_vs_Avg4w', 0)
    if sv > 1.5:
        signals.append(f"Search spike {sv:.1f}x avg")

    vv = row.get('Vol_vs_Avg', 0)
    if vv > 2.0:
        signals.append(f"Volume {vv:.1f}x avg")

    dr = row.get('Daily_Range', 0)
    if dr > 0.05:
        signals.append(f"Wide intraday range ({dr:.1%})")

    sp = row.get('Short_Pct', 0)
    if sp > 0.15:
        signals.append(f"Short interest {sp:.1%}")

    v5 = row.get('Volatility_5d', 0)
    if v5 > 0.05:
        signals.append(f"High 5d volatility ({v5:.3f})")

    nv = row.get('news_vs_avg', 0)
    if nv > 2.0:
        signals.append(f"News {nv:.1f}x avg")

    if not signals:
        signals = ["No strong signals detected"]

    return " | ".join(signals[:3])
