# src/daily_recommender_once.py
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# allow importing fetch_live_snapshot
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fetch_live_snapshot import fetch_and_build_snapshot

ROOT = os.path.join(os.path.dirname(__file__), "..")
SNAPSHOT_FILE = os.path.join(ROOT, "data", "today_snapshot.csv")
MODEL_SIGNAL = os.path.join(ROOT, "models", "signal_model.pkl")
MODEL_TRADE = os.path.join(ROOT, "models", "trade_outcome_model.pkl")
SCALER_FILE = os.path.join(ROOT, "models", "scaler.pkl")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Load models & scaler
signal_info = joblib.load(MODEL_SIGNAL)
trade_info = joblib.load(MODEL_TRADE)
scaler = joblib.load(SCALER_FILE)

model_signal = signal_info["model"]
model_trade = trade_info["model"]
num_feats = signal_info["num_feats"]
ticker_dummies = signal_info["ticker_dummies"]
inverse_label_map = signal_info.get("label_map", {0: -1, 1: 0, 2: 1})

# Parameters
WIN_PROB_THRESHOLD = 0.60
CAPITAL = 10000
RISK_PER_TRADE = 0.05

def build_feature_vector(row):
    x_num = pd.DataFrame([[row.get(f, 0.0) for f in num_feats]], columns=num_feats)
    x_num_scaled = scaler.transform(x_num)

    x_dummy = np.zeros((1, len(ticker_dummies)))
    tcol = f"T_{row['Ticker']}"
    if tcol in ticker_dummies:
        idx = ticker_dummies.index(tcol)
        x_dummy[0, idx] = 1.0
    return np.hstack([x_num_scaled, x_dummy])

def run_once():
    print("⏳ Fetching snapshot...")
    df = fetch_and_build_snapshot()
    if df is None or df.empty:
        print("⚠️ Snapshot empty.")
        return []

    recs = []
    for _, row in df.iterrows():
        X = build_feature_vector(row)
        sig_probs = model_signal.predict_proba(X)[0]
        sig_idx = int(np.argmax(sig_probs))
        sig_label = inverse_label_map[sig_idx]

        trade_probs = model_trade.predict_proba(X)[0]
        win_prob = float(trade_probs[2]) if len(trade_probs) == 3 else float(trade_probs[-1])

        side = "HOLD"
        if sig_label == 1:
            side = "BUY"
        elif sig_label == -1:
            side = "SELL"

        if side == "HOLD" or win_prob < WIN_PROB_THRESHOLD:
            continue

        entry = float(row.get("Close", 0.0))
        atr = row.get("ATR_14_15m", row.get("ATR_14_5m", entry * 0.01))
        if pd.isna(atr) or atr <= 0:
            atr = entry * 0.01

        if side == "BUY":
            sl = entry - 1.5 * atr
            tp = entry + 3.0 * atr
        else:
            sl = entry + 1.5 * atr
            tp = entry - 3.0 * atr

        stop_distance = abs(entry - sl)
        if stop_distance <= 0:
            continue

        risk_amount = CAPITAL * RISK_PER_TRADE
        shares = max(1, int(risk_amount // stop_distance))

        recs.append({
            "Stock": row["Ticker"],
            "Side": side,
            "Entry": round(entry, 2),
            "StopLoss": round(sl, 2),
            "Target": round(tp, 2),
            "Shares": shares,
            "Confidence": f"{win_prob:.1%}",
            "Timestamp": datetime.now().isoformat()
        })

    if recs:
        out_json = os.path.join(OUT_DIR, "trade_signals.json")
        pd.DataFrame(recs).to_json(out_json, orient="records", date_format="iso")
        print(f"✅ {len(recs)} signals saved to {out_json}")
    else:
        print("⚠️ No signals found this run.")

    return recs

if __name__ == "__main__":
    run_once()
