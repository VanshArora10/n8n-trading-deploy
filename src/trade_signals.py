# src/trade_signals.py
import os
import pandas as pd

# -----------------------
# Paths
# -----------------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
SNAPSHOT_FILE = os.path.join(ROOT, "data", "today_snapshot.csv")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Parameters
# -----------------------
CAPITAL = 10000
RISK_PER_TRADE = 0.02  # risk 2% per trade
ATR_MULT_SL = 1.5
ATR_MULT_TP = 3.0

# -----------------------
# Strategy Rules
# -----------------------
def generate_signal(row):
    """
    Simple strategy:
    - BUY if RSI_14_15m < 30 and Close > 5m_Open
    - SELL if RSI_14_15m > 70 and Close < 5m_Open
    - Otherwise HOLD
    """
    rsi = row.get("RSI_14_15m", None)
    close = row.get("Close", None)
    open_5m = row.get("5m_Open", None)

    if rsi is None or pd.isna(rsi):
        return "HOLD"

    if rsi < 30 and close > open_5m:
        return "BUY"
    elif rsi > 70 and close < open_5m:
        return "SELL"
    else:
        return "HOLD"

# -----------------------
# Run Evaluator
# -----------------------
def main():
    if not os.path.exists(SNAPSHOT_FILE):
        print(f"❌ No snapshot file found at {SNAPSHOT_FILE}")
        return

    df = pd.read_csv(SNAPSHOT_FILE)
    recs = []

    for _, row in df.iterrows():
        signal = generate_signal(row)

        if signal == "HOLD":
            continue

        entry = row["Close"]
        atr = row.get("ATR_14_15m", entry * 0.01)
        if pd.isna(atr) or atr <= 0:
            atr = entry * 0.01

        if signal == "BUY":
            stop = entry - ATR_MULT_SL * atr
            target = entry + ATR_MULT_TP * atr
        else:
            stop = entry + ATR_MULT_SL * atr
            target = entry - ATR_MULT_TP * atr

        # Position sizing
        stop_distance = abs(entry - stop)
        risk_amount = CAPITAL * RISK_PER_TRADE
        shares = max(1, int(risk_amount // stop_distance))

        recs.append({
            "Ticker": row["Ticker"],
            "Signal": signal,
            "Entry": round(entry, 2),
            "StopLoss": round(stop, 2),
            "Target": round(target, 2),
            "Shares": shares
        })

    if recs:
        out_path = os.path.join(OUT_DIR, "trade_signals.csv")
        pd.DataFrame(recs).to_csv(out_path, index=False)
        print(f"✅ Signals saved to {out_path}\n")
        for r in recs:
            print(f"📢 {r['Ticker']} | {r['Signal']} @ {r['Entry']} → TP {r['Target']} / SL {r['StopLoss']} | Qty {r['Shares']}")
    else:
        print("⚠️ No signals found today.")

if __name__ == "__main__":
    main()
