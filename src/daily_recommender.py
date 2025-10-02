# src/daily_recommender.py
import os
import joblib
import pandas as pd
import numpy as np
import sys

# Import the live fetcher
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fetch_live_snapshot import fetch_and_build_snapshot

ROOT = os.path.join(os.path.dirname(__file__), "..")
SNAPSHOT_FILE = os.path.join(ROOT, "data", "today_snapshot.csv")
MODEL_SIGNAL = os.path.join(ROOT, "models", "signal_model.pkl")
MODEL_TRADE = os.path.join(ROOT, "models", "trade_outcome_model.pkl")
SCALER_FILE = os.path.join(ROOT, "models", "scaler.pkl")

OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Load models & scaler
# -----------------------
signal_info = joblib.load(MODEL_SIGNAL)
trade_info = joblib.load(MODEL_TRADE)
scaler = joblib.load(SCALER_FILE)

model_signal = signal_info["model"]
model_trade = trade_info["model"]
num_feats = signal_info["num_feats"]
ticker_dummies = signal_info["ticker_dummies"]
inverse_label_map = signal_info.get("label_map", {0: -1, 1: 0, 2: 1})

# -----------------------
# Parameters
# -----------------------
WIN_PROB_THRESHOLD = 0.35   # require win prob > 35%
CAPITAL = 10000
RISK_PER_TRADE = 0.05

# -----------------------
# Step 1: Fetch live snapshot
# -----------------------
print("⏳ Fetching live snapshot...")
df = fetch_and_build_snapshot()   # runs the live fetcher and saves CSV
print("✅ Using snapshot:", SNAPSHOT_FILE)

# -----------------------
# Helper: Build feature vector
# -----------------------
def build_feature_vector(row, prefix="15m_"):
    """Build feature vector using selected timeframe (default: 15m)."""
    x_num = []
    for f in num_feats:
        val = row.get(prefix + f, row.get("1d_" + f, 0.0))  # fallback daily if missing
        x_num.append(val)
    x_num = pd.DataFrame([x_num], columns=num_feats)
    x_num_scaled = scaler.transform(x_num)

    # ticker dummies
    x_dummy = np.zeros((1, len(ticker_dummies)))
    tcol = f"T_{row['Ticker']}"
    if tcol in ticker_dummies:
        x_dummy[0, ticker_dummies.index(tcol)] = 1.0

    return np.hstack([x_num_scaled, x_dummy])

# -----------------------
# Step 2: Run predictions
# -----------------------
recs = []
raw_recs = []

for _, row in df.iterrows():
    # --- Entry signals from 15m data ---
    X = build_feature_vector(row, prefix="15m_")

    sig_probs = model_signal.predict_proba(X)[0]
    sig_pred_idx = int(np.argmax(sig_probs))
    sig_label = inverse_label_map[sig_pred_idx]  # -1 SELL, 0 HOLD, 1 BUY

    # --- Outcome prediction ---
    trade_probs = model_trade.predict_proba(X)[0]
    win_prob = trade_probs[2] if len(trade_probs) == 3 else trade_probs[-1]

    side = "HOLD"
    if sig_label == 1:
        side = "BUY"
    elif sig_label == -1:
        side = "SELL"

    # Save raw info
    raw_recs.append({
        "Ticker": row["Ticker"],
        "Signal": side,
        "Signal_Probs": sig_probs,
        "Trade_Probs": trade_probs,
        "WinProb": win_prob
    })

    # --- Confirmation filter using 1h / 1d trend ---
    trend_ok = True
    if side == "BUY":
        trend_ok = (row.get("1h_EMA_50", 0) > row.get("1h_EMA_200", 0)) or \
                   (row.get("1d_EMA_50", 0) > row.get("1d_EMA_200", 0))
    elif side == "SELL":
        trend_ok = (row.get("1h_EMA_50", 0) < row.get("1h_EMA_200", 0)) or \
                   (row.get("1d_EMA_50", 0) < row.get("1d_EMA_200", 0))

    # Skip holds, low-confidence, or bad trend alignment
    if side == "HOLD" or win_prob < WIN_PROB_THRESHOLD or not trend_ok:
        continue

    # -----------------------
    # Entry / SL / TP
    # -----------------------
    entry = float(row.get("15m_Close", row.get("1d_Close", 0)))

    atr = row.get("15m_ATR_14", row.get("1d_ATR_14", entry * 0.01))
    if pd.isna(atr) or atr <= 0:
        atr = entry * 0.01

    if side == "BUY":
        sl = entry - 1.5 * atr
        tp = entry + 3.0 * atr
    else:  # SELL
        sl = entry + 1.5 * atr
        tp = entry - 3.0 * atr

    stop_distance = abs(entry - sl)
    if stop_distance <= 0 or pd.isna(stop_distance):
        continue

    # Position sizing
    risk_amount = CAPITAL * RISK_PER_TRADE
    shares = max(1, int(risk_amount // stop_distance))

    recs.append({
        "Stock": row["Ticker"],
        "Signal": side,
        "Entry": round(entry, 2),
        "StopLoss": round(sl, 2),
        "Target": round(tp, 2),
        "Shares": shares,
        "Confidence": f"{win_prob:.1%}"
    })

# -----------------------
# Step 3: Save outputs
# -----------------------
pd.DataFrame(raw_recs).to_csv(os.path.join(OUT_DIR, "trade_recs_raw.csv"), index=False)
print("✅ Saved raw predictions to output/trade_recs_raw.csv")

if not recs:
    print("\n⚠️ No trades passed filters today.")
else:
    pd.DataFrame(recs).to_csv(os.path.join(OUT_DIR, "trade_recs.csv"), index=False)
    print("\n📢 Trade Recommendations:\n")
    for r in recs:
        print(f"Stock: {r['Stock']}")
        print(f"Signal: {r['Signal']}")
        print(f"Entry: {r['Entry']} | Target: {r['Target']} | Stop Loss: {r['StopLoss']}")
        print(f"Shares: {r['Shares']} | Confidence: {r['Confidence']}")
        print("------")
