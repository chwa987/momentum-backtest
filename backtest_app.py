import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def get_series(data, column="Close"):
    try:
        if column not in data:
            return pd.Series(dtype=float)
        s = data[column]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)

def compute_indicators(price, idx_price, volume):
    if price.dropna().empty:
        return [np.nan]*6
    last_close = price.iloc[-1]
    gd200 = price.rolling(200).mean().iloc[-1] if len(price) >= 200 else np.nan
    gd130 = price.rolling(130).mean().iloc[-1] if len(price) >= 130 else np.nan
    abw200 = (last_close - gd200) / gd200 * 100 if np.isfinite(gd200) else np.nan
    abw130 = (last_close - gd130) / gd130 * 100 if np.isfinite(gd130) else np.nan
    if len(price) > 260 and np.isfinite(price.iloc[-260]):
        mom260 = (last_close / price.iloc[-260] - 1) * 100
        ret_12m = last_close / price.iloc[-260] - 1
    else:
        mom260, ret_12m = np.nan, np.nan
    if len(price) > 21 and np.isfinite(price.iloc[-21]) and np.isfinite(ret_12m):
        ret_1m = last_close / price.iloc[-21] - 1
        momjt = (ret_12m - ret_1m) * 100
    else:
        momjt = np.nan
    if not idx_price.dropna().empty and len(idx_price) > 260 and np.isfinite(ret_12m):
        idx_ret12m = idx_price.iloc[-1] / idx_price.iloc[-260] - 1
        rel_str = ((1 + ret_12m) / (1 + idx_ret12m) - 1) * 100 if np.isfinite(idx_ret12m) else np.nan
    else:
        rel_str = np.nan
    if not volume.dropna().empty and len(volume) > 50:
        vol50 = volume.rolling(50).mean().iloc[-1]
        vol_score = (volume.iloc[-1] / vol50) if np.isfinite(vol50) and vol50 != 0 else np.nan
    else:
        vol_score = np.nan
    return [abw200, abw130, mom260, momjt, rel_str, vol_score]

def momentum_backtest(tickers, start="2018-01-01", end=None, top_n=10):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    idx = yf.download("^GSPC", start=start, end=end, auto_adjust=True)
    idx_price = get_series(idx, "Close")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by="ticker")

    months = pd.date_range(start, end, freq="M")
    portfolio_values, benchmark_values, dates = [], [], []
    portfolio_value, benchmark_value = 100.0, 100.0

    for date in months:
        window_start = date - pd.Timedelta(days=400)
        scores = {}
        for ticker in tickers:
            try:
                df = data[ticker].loc[window_start:date]
                price = get_series(df, "Close")
                volume = get_series(df, "Volume")
                indicators = compute_indicators(price, idx_price.loc[window_start:date], volume)
                score = sum([ind for ind in indicators if np.isfinite(ind)])
                scores[ticker] = score
            except:
                continue

        if not scores:
            continue

        # ✅ Nur Top-N Aktien wählen
        top_tickers = sorted(scores, key=scores.get, reverse=True)[:top_n]

        ret = 0
        for ticker in top_tickers:
            try:
                df = data[ticker].loc[:date]
                price = get_series(df, "Close")
                if len(price) > 1:
                    ret += price.pct_change().iloc[-1]
            except:
                continue
        ret /= max(1, len(top_tickers))

        bench_ret = idx_price.pct_change().iloc[-1] if len(idx_price.loc[:date]) > 1 else 0
        portfolio_value *= (1 + ret)
        benchmark_value *= (1 + bench_ret)

        dates.append(date)
        portfolio_values.append(portfolio_value)
        benchmark_values.append(benchmark_value)

    results = pd.DataFrame({"Portfolio": portfolio_values, "Benchmark": benchmark_values}, index=dates)

    if results.empty:
        return results, {}

    cagr = (results["Portfolio"].iloc[-1] / results["Portfolio"].iloc[0]) ** (
        1 / ((results.index[-1] - results.index[0]).days / 365)) - 1
    vol = results["Portfolio"].pct_change().std() * np.sqrt(252)
    dd = ((results["Portfolio"] / results["Portfolio"].cummax()) - 1).min()
    sharpe = (cagr - 0.0) / vol if vol > 0 else np.nan

    stats = {
        "CAGR": round(cagr * 100, 2),
        "Volatilität": round(vol * 100, 2),
        "Max Drawdown": round(dd * 100, 2),
        "Sharpe-Ratio": round(sharpe, 2)
    }

    return results, stats

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📈 Momentum-Backtest (Top 10)")

uploaded_file = st.file_uploader("📂 Lade deine Champions-CSV hoch", type=["csv"])
tickers = []

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    if "Ticker" in df_csv.columns:
        tickers = df_csv["Ticker"].dropna().astype(str).tolist()
        st.success(f"{len(tickers)} Ticker aus CSV geladen.")
        st.write(df_csv.head())
else:
    tickers_input = st.text_area("Oder gib Ticker ein (kommasepariert)", "AAPL, MSFT, NVDA, AMZN, META, TSLA, NFLX, INTC, AMD, GOOGL")
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

if tickers and st.button("Backtest starten"):
    results, stats = momentum_backtest(tickers, start="2018-01-01", top_n=10)
    if not results.empty:
        st.line_chart(results)
        st.write("📊 Kennzahlen:")
        st.json(stats)
        st.download_button("📥 Ergebnisse als CSV", results.to_csv().encode("utf-8"), "backtest_results.csv", "text/csv")
    else:
        st.error("❌ Keine Ergebnisse – überprüfe deine Ticker oder CSV-Datei.")
