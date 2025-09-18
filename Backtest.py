import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def get_series(data, column="Close"):
    """Sichert, dass immer eine 1D-Serie zurückkommt."""
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
    """Berechnet die 6 Kriterien für eine Aktie."""
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

# ----------------------------
# Backtest
# ----------------------------
def momentum_backtest(tickers, start="2018-01-01", end=None, top_n=10):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # Benchmark laden
    idx = yf.download("^GSPC", start=start, end=end, auto_adjust=True)
    idx_price = get_series(idx, "Close")

    # Preisdaten für alle Ticker laden
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by="ticker")

    # Monatsende-Datenpunkte
    months = pd.date_range(start, end, freq="M")

    portfolio_values = []
    benchmark_values = []
    dates = []

    portfolio_value = 100.0
    benchmark_value = 100.0

    for date in months:
        window_start = date - pd.Timedelta(days=400)
        
        scores = {}
        for ticker in tickers:
            try:
                df = data[ticker].loc[window_start:date]
                price = get_series(df, "Close")
                volume = get_series(df, "Volume")

                indicators = compute_indicators(price, idx_price.loc[window_start:date], volume)
                score = 0
                for ind in indicators:
                    if np.isfinite(ind):
                        score += ind
                scores[ticker] = score
            except Exception:
                continue

        # Top N auswählen
        top_tickers = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # Portfolio performance
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

        # Benchmark performance
        if len(idx_price.loc[:date]) > 1:
            bench_ret = idx_price.pct_change().iloc[-1]
        else:
            bench_ret = 0

        portfolio_value *= (1 + ret)
        benchmark_value *= (1 + bench_ret)

        dates.append(date)
        portfolio_values.append(portfolio_value)
        benchmark_values.append(benchmark_value)

    # Equity-Kurven
    results = pd.DataFrame({"Portfolio": portfolio_values, "Benchmark": benchmark_values}, index=dates)

    # Kennzahlen
    cagr = (results["Portfolio"].iloc[-1] / results["Portfolio"].iloc[0]) ** (1/((results.index[-1]-results.index[0]).days/365)) - 1
    vol = results["Portfolio"].pct_change().std() * np.sqrt(252)
    dd = ((results["Portfolio"] / results["Portfolio"].cummax()) - 1).min()
    sharpe = (cagr - 0.0) / vol if vol > 0 else np.nan

    stats = {
        "CAGR": round(cagr*100,2),
        "Volatilität": round(vol*100,2),
        "Max Drawdown": round(dd*100,2),
        "Sharpe-Ratio": round(sharpe,2)
    }

    # Plot
    results.plot(title="Momentum Backtest (Top 10)")
    plt.show()

    return results, stats

# ----------------------------
# Beispiel: Backtest laufen lassen
# ----------------------------
if __name__ == "__main__":
    tickers = ["APP","LEU","XMTR","RHM.DE","KTOS","RKLB","MP","SOFI","LITE","LQDA"]
    results, stats = momentum_backtest(tickers, start="2018-01-01", top_n=10)
    print(stats)
