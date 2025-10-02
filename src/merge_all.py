import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
IND = os.path.join(ROOT, "data", "all_stocks_indicators.csv")
NEWS = os.path.join(ROOT, "data", "all_news_sentiment.csv")
OUT = os.path.join(ROOT, "data", "all_stocks_dataset.csv")

# Load indicators
df_ind = pd.read_csv(IND, parse_dates=["Datetime"])
df_ind["date"] = df_ind["Datetime"].dt.date  # pure date object

# Load news
df_news = pd.read_csv(NEWS, parse_dates=["date"])
df_news["date"] = df_news["date"].dt.date  # convert to date as well

# Merge
df = df_ind.merge(df_news, on=["Ticker", "date"], how="left")
df["sentiment"] = df["sentiment"].fillna(0.0)  # no news = neutral

df.to_csv(OUT, index=False)
print("✅ Saved all_stocks_dataset.csv")
