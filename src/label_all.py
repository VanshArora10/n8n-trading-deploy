import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN = os.path.join(ROOT, "data", "all_stocks_dataset.csv")
OUT = os.path.join(ROOT, "data", "all_stocks_labeled.csv")

df = pd.read_csv(IN, parse_dates=["Datetime"])
df = df.sort_values(["Ticker", "Datetime"]).reset_index(drop=True)

# Function to label per ticker
def label_group(g, up=0.02, down=-0.02, tp=0.03, sl=0.02):
    """
    up/down = thresholds for initial signal labeling
    tp = target % for take profit
    sl = stop loss % for stop loss
    """
    g = g.copy().reset_index(drop=True)
    g["future_close"] = g["Close"].shift(-1)
    g["future_return"] = g["future_close"] / g["Close"] - 1

    # Label raw signals
    def lab(x):
        if pd.isna(x):
            return None
        if x > up:
            return 1    # BUY
        elif x < down:
            return -1   # SELL
        else:
            return 0    # HOLD

    g["target"] = g["future_return"].apply(lab).astype("Int64")

    # Trade Plan Features
    g["entry_price"] = g["Close"].shift(-1)   # assume entry at next day close
    g["tp_price"] = g["entry_price"] * (1 + tp)
    g["sl_price"] = g["entry_price"] * (1 - sl)

    # Trade Outcome
    horizon = 3
    outcomes = []
    for i in range(len(g)):
        if pd.isna(g.iloc[i]["entry_price"]):
            outcomes.append(None)
            continue
        window = g.iloc[i+1:i+1+horizon]
        if len(window) == 0:
            outcomes.append(0)
            continue

        side = g.iloc[i]["target"]
        entry = g.iloc[i]["entry_price"]
        tp = g.iloc[i]["tp_price"]
        sl = g.iloc[i]["sl_price"]

        outcome = 0  # default HOLD/neutral
        if side == 1:  # BUY
            if (window["High"] >= tp).any():
                outcome = 1
            elif (window["Low"] <= sl).any():
                outcome = -1
        elif side == -1:  # SELL
            if (window["Low"] <= sl).any():
                outcome = 1
            elif (window["High"] >= tp).any():
                outcome = -1
        outcomes.append(outcome)

    g["trade_outcome"] = outcomes
    return g

# Apply per ticker
out = []
for t, g in df.groupby("Ticker"):
    out.append(label_group(g))

df2 = pd.concat(out, ignore_index=True)
df2 = df2.dropna(subset=["target"])
df2.to_csv(OUT, index=False)

print("✅ Saved labeled dataset:", OUT)
print(df2[["Datetime","Ticker","Close","target","entry_price","tp_price","sl_price","trade_outcome"]].head(15))
print("📊 Target Distribution:")
print(df2["target"].value_counts(normalize=True))
print("📊 Trade Outcome Distribution:")
print(df2["trade_outcome"].value_counts(normalize=True))
