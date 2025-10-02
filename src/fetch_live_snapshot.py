# scripts/fetch_live_snapshot.py
import os
import pandas as pd
import yfinance as yf
import ta
import time
from datetime import datetime

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "today_snapshot.csv")

TICKERS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS",
    "AXISBANK.NS","TCS.NS","INFY.NS","MARUTI.NS","ADANIENT.NS","^NSEI"
]

# ⚠ Remove "1m" (not supported well by Yahoo for NSE)
TIMEFRAMES = {
    "5m": {"interval": "5m", "period": "5d"},
    "15m": {"interval": "15m", "period": "5d"},
    "1h": {"interval": "60m", "period": "60d"},
    "1d": {"interval": "1d", "period": "365d"}
}

def safe_last(series):
    """Return last value of Series or None."""
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return None

def compute_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    d = {}
    try:
        d["RSI_14"] = safe_last(ta.momentum.RSIIndicator(close, 14).rsi())
    except: d["RSI_14"] = None
    try:
        d["EMA_20"] = safe_last(ta.trend.EMAIndicator(close, 20).ema_indicator())
        d["EMA_50"] = safe_last(ta.trend.EMAIndicator(close, 50).ema_indicator())
        d["EMA_200"] = safe_last(ta.trend.EMAIndicator(close, 200).ema_indicator())
    except: pass
    try:
        macd = ta.trend.MACD(close)
        d["MACD"] = safe_last(macd.macd())
        d["MACD_Signal"] = safe_last(macd.macd_signal())
        d["MACD_Hist"] = safe_last(macd.macd_diff())
    except: pass
    try:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        d["BB_lower"] = safe_last(bb.bollinger_lband())
        d["BB_mid"] = safe_last(bb.bollinger_mavg())
        d["BB_upper"] = safe_last(bb.bollinger_hband())
    except: pass
    try:
        atr = ta.volatility.AverageTrueRange(high, low, close, 14)
        d["ATR_14"] = safe_last(atr.average_true_range())
    except: d["ATR_14"] = None

    d["return_1d"] = safe_last(close.pct_change())
    d["volatility_5"] = safe_last(close.pct_change().rolling(5).std())
    d["volume"] = safe_last(vol)

    return d

def fetch_for_ticker(ticker):
    stock_info = {"Ticker": ticker}
    for tf_name, params in TIMEFRAMES.items():
        try:
            data = yf.download(
                ticker, period=params["period"], interval=params["interval"],
                progress=False, threads=False, auto_adjust=True
            )
            if data.empty:
                print(f"⚠️ No data for {ticker} @ {tf_name}")
                continue

            data = data[["Open","High","Low","Close","Volume"]].dropna()
            latest = data.tail(1).iloc[0]

            stock_info[f"{tf_name}_Datetime"] = latest.name
            stock_info[f"{tf_name}_Open"] = float(latest["Open"])
            stock_info[f"{tf_name}_High"] = float(latest["High"])
            stock_info[f"{tf_name}_Low"] = float(latest["Low"])
            stock_info[f"{tf_name}_Close"] = float(latest["Close"])
            stock_info[f"{tf_name}_Volume"] = int(latest["Volume"])

            # Indicators
            inds = compute_indicators(data)
            for k, v in inds.items():
                stock_info[f"{k}_{tf_name}"] = v

        except Exception as e:
            print(f"⚠️ Indicator error for {ticker} @ {tf_name}: {e}")
            continue

        time.sleep(0.2)

    # Pick a main Close/Open/High/Low (prefer 15m > 5m > 1h > 1d)
    for prefer in ("15m","5m","1h","1d"):
        if f"{prefer}_Close" in stock_info:
            stock_info["Close"] = stock_info[f"{prefer}_Close"]
            stock_info["Open"] = stock_info[f"{prefer}_Open"]
            stock_info["High"] = stock_info[f"{prefer}_High"]
            stock_info["Low"] = stock_info[f"{prefer}_Low"]
            stock_info["Volume"] = stock_info[f"{prefer}_Volume"]
            break

    return stock_info

def fetch_and_build_snapshot(tickers=None):
    tickers = tickers or TICKERS
    rows = []
    start = datetime.now()
    for t in tickers:
        print(f"⏳ Fetching {t}...")
        rows.append(fetch_for_ticker(t))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"✅ Live snapshot saved → {OUT} (fetched {len(rows)} tickers in {(datetime.now()-start).seconds}s)")
    return df

if __name__ == "__main__":
    fetch_and_build_snapshot()
