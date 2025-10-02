# src/daily_recommender_loop.py
import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime

# allow importing fetch_live_snapshot (scripts folder)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fetch_live_snapshot import fetch_and_build_snapshot

# news sentiment
from news_sentiment import get_news_sentiment

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

# runtime params
SLEEP_SECONDS = 300     # run every 5 minutes
WIN_PROB_THRESHOLD = 0.60
CAPITAL = 10000
RISK_PER_TRADE = 0.05

def build_feature_vector(row):
    # row is a pandas Series, features expected: names in num_feats (if missing -> 0)
    x_num = pd.DataFrame([[row.get(f, 0.0) for f in num_feats]], columns=num_feats)
    x_num_scaled = scaler.transform(x_num)

    x_dummy = np.zeros((1, len(ticker_dummies)))
    tcol = f"T_{row['Ticker']}"
    if tcol in ticker_dummies:
        idx = ticker_dummies.index(tcol)
        x_dummy[0, idx] = 1.0
    return np.hstack([x_num_scaled, x_dummy])

def process_snapshot_and_emit(snapshot_df):
    recs = []
    raw_recs = []
    for _, row in snapshot_df.iterrows():
        X = build_feature_vector(row)
        sig_probs = model_signal.predict_proba(X)[0]
        sig_idx = int(np.argmax(sig_probs))
        sig_label = inverse_label_map[sig_idx]   # -1,0,1
        trade_probs = model_trade.predict_proba(X)[0]
        win_prob = float(trade_probs[2]) if len(trade_probs) == 3 else float(trade_probs[-1])

        side = "HOLD"
        if sig_label == 1:
            side = "BUY"
        elif sig_label == -1:
            side = "SELL"

        # raw rec for logging
        raw_recs.append({
            "timestamp": datetime.now(),
            "Ticker": row["Ticker"],
            "Signal": side,
            "Signal_Probs": sig_probs.tolist(),
            "Trade_Probs": trade_probs.tolist(),
            "WinProb": win_prob
        })

        if side == "HOLD" or win_prob < WIN_PROB_THRESHOLD:
            continue

        # News check (optional)
        news_label, news_score = get_news_sentiment(row["Ticker"])
        # adjust win_prob slightly if news supports signal
        if (news_label == "Positive" and side == "BUY") or (news_label == "Negative" and side == "SELL"):
            win_prob = min(1.0, win_prob + 0.1 * (1 + news_score))
        elif (news_label == "Negative" and side == "BUY") or (news_label == "Positive" and side == "SELL"):
            win_prob = max(0.0, win_prob - 0.2 * (1 + abs(news_score)))

        # Entry/SL/TP using 15m ATR if available else fallback
        entry = float(row.get("Close") or row.get("15m_Close") or 0.0)
        atr = row.get("ATR_14_15m", row.get("ATR_14_5m", entry * 0.01))
        if pd.isna(atr) or atr <= 0:
            atr = max(0.01, entry * 0.01)

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

        rec = {
            "Stock": row["Ticker"],
            "Side": side,
            "Entry": round(entry,2),
            "StopLoss": round(sl,2),
            "Target": round(tp,2),
            "Shares": shares,
            "Confidence": f"{win_prob:.1%}",
            "News": news_label,
            "NewsScore": news_score,
            "SignalProbs": sig_probs.tolist(),
            "TradeProbs": trade_probs.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        recs.append(rec)

    # persist logs for n8n to pick up
    if raw_recs:
        pd.DataFrame(raw_recs).to_csv(os.path.join(OUT_DIR, f"raw_preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
    if recs:
        # save latest signals to a JSON that n8n can watch
        out_json = os.path.join(OUT_DIR, "live_trade_signals.json")
        pd.DataFrame(recs).to_json(out_json, orient="records", date_format="iso")
        # also append to a rolling CSV log
        csv_path = os.path.join(OUT_DIR, "live_trade_log.csv")
        dfnew = pd.DataFrame(recs)
        if os.path.exists(csv_path):
            dfnew.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            dfnew.to_csv(csv_path, index=False)
    return recs

def run_loop(start_time=(9,15), end_time=(15,30), sleep_seconds=SLEEP_SECONDS):
    start_dt = dtime(start_time[0], start_time[1])
    end_dt = dtime(end_time[0], end_time[1])
    print(f"Starting loop: monitoring between {start_dt} and {end_dt} every {sleep_seconds}s")
    while True:
        now = datetime.now()
        if now.time() < start_dt:
            # sleep until market open
            secs = (datetime.combine(now.date(), start_dt) - now).seconds
            print(f"Waiting {secs}s until market open at {start_dt}")
            time.sleep(min(secs, 300))
            continue
        if now.time() > end_dt:
            print("Market monitoring finished for today.")
            break

        # fetch fresh snapshot
        try:
            df = fetch_and_build_snapshot()
            if df is None or df.empty:
                print("Snapshot empty; sleeping and retrying.")
                time.sleep(sleep_seconds)
                continue
            recs = process_snapshot_and_emit(df)
            if recs:
                print(f"Found {len(recs)} candidate trades. Saved to output/live_trade_signals.json")
            else:
                print("No candidate trades at this tick.")
        except Exception as e:
            print("Error during cycle:", e)

        time.sleep(sleep_seconds)

if __name__ == "__main__":
    run_loop()
