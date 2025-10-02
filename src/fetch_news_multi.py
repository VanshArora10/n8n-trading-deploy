import os
import pandas as pd
import feedparser
from urllib.parse import quote
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

TICKER_TO_NAME = {
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "TCS.NS": "Tata Consultancy Services",
    "AXISBANK.NS": "Axis Bank",
    "SBIN.NS": "State Bank of India",
    "ADANIENT.NS": "Adani Enterprises",
    "MARUTI.NS": "Maruti Suzuki",
    "^NSEI": "Nifty 50"
}

all_rows = []

for ticker, name in TICKER_TO_NAME.items():
    query = quote(f"{name} India stock news")
    url = f"https://news.google.com/rss/search?q={query}"
    print(f"Fetching news for {ticker} ({name}) ...")

    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries:
        rows.append({
            "date": pd.to_datetime(entry.published, errors="coerce").date(),
            "title": entry.title,
            "link": entry.link,
            "Ticker": ticker
        })
    if rows:
        df = pd.DataFrame(rows).dropna().sort_values("date")
        out_file = os.path.join(DATA, f"news_{ticker.replace('/','_')}.csv")
        df.to_csv(out_file, index=False)
        all_rows.append(df)
    time.sleep(1)  # avoid hitting rate limits

if all_rows:
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(os.path.join(DATA, "all_news.csv"), index=False)
    print("✅ Saved all_news.csv")
else:
    print("⚠️ No news fetched")
