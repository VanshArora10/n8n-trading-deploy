import os
import pandas as pd
import numpy as np
import joblib

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "all_stocks_labeled.csv")
MODEL_FILE = os.path.join(ROOT, "models", "general_model.pkl")
SCALER_FILE = os.path.join(ROOT, "models", "scaler.pkl")

# Load dataset
df = pd.read_csv(DATA, parse_dates=["Datetime"]).sort_values(["Datetime", "Ticker"])
df["date"] = df["Datetime"].dt.date
df["Ticker_original"] = df["Ticker"]

# Load model
info = joblib.load(MODEL_FILE)
model = info["model"]
num_feats = info["num_feats"]
ticker_dummies = info["ticker_dummies"]
scaler = joblib.load(SCALER_FILE)

# One-hot encode ticker
df = pd.get_dummies(df, columns=["Ticker"], prefix="T")
for col in ticker_dummies:
    if col not in df.columns:
        df[col] = 0

# Features
X_num = df[num_feats].astype(float).fillna(0)
X_scaled = scaler.transform(X_num)
X_dummies = df[ticker_dummies].values
X_final = np.hstack([X_scaled, X_dummies])

# Predictions
df["pred"] = model.predict(X_final)

# --- Backtest ---
initial_cash = 100000.0
cash = initial_cash
positions = {}   # ticker -> (shares, buy_price)
trade_log = []

# Process day by day
for day, group in df.groupby("date"):
    for _, row in group.iterrows():
        ticker = row["Ticker_original"]
        signal = row["pred"]

        # BUY
        if signal == 1 and ticker not in positions:
            allocation = cash * 0.1
            buy_price = row["Close"]
            shares = int(allocation // buy_price)

            if shares > 0 and cash >= shares * buy_price:
                positions[ticker] = (shares, buy_price)
                cash -= shares * buy_price
                trade_log.append(("BUY", ticker, row["Datetime"], buy_price, shares))

        # SELL
        elif signal == -1 and ticker in positions:
            shares, buy_price = positions.pop(ticker)
            sell_price = row["Open"]  # sell next day open ~ approximation
            proceeds = shares * sell_price
            cash += proceeds
            trade_log.append(("SELL", ticker, row["Datetime"], sell_price, shares))

# Final liquidation
for ticker, (shares, buy_price) in positions.items():
    last_close = df[df["Ticker_original"] == ticker].iloc[-1]["Close"]
    cash += shares * last_close
    trade_log.append(("SELL_END", ticker, df.iloc[-1]["Datetime"], last_close, shares))

final_value = cash
pnl = final_value - initial_cash

print("📊 Backtest Results")
print(f"Initial Cash: {initial_cash:.2f}")
print(f"Final Value: {final_value:.2f}")
print(f"Net P&L: {pnl:.2f}")
print(f"Total Trades: {len(trade_log)}")
print("Sample Trades:", trade_log[:10])
