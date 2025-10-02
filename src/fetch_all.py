# src/fetch_all.py
import os, time
import yfinance as yf
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = ["RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS",
           "AXISBANK.NS","SBIN.NS","ADANIENT.NS","MARUTI.NS","^NSEI"]

out_files = []
for t in TICKERS:
    print("Fetching", t)
    df = yf.download(t, period="5y", interval="1d", progress=False)
    if df.empty:
        print("Warning: no data for", t)
        continue
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df = df.reset_index()
    df['Ticker'] = t
    file = os.path.join(DATA_DIR, f"{t.replace('/','_')}_1d.csv")
    df.to_csv(file, index=False)
    out_files.append(file)
    time.sleep(1)  # polite pause to avoid rate limits

# concat
df_all = pd.concat([pd.read_csv(f, parse_dates=['Date']) for f in out_files], ignore_index=True)
df_all.rename(columns={'Date':'Datetime'}, inplace=True)
df_all.to_csv(os.path.join(DATA_DIR, "all_stocks_raw.csv"), index=False)
print("Saved all_stocks_raw.csv")
