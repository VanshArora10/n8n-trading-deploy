# src/simulate_backtest.py
import os
import joblib
import pandas as pd
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(ROOT, "data", "all_stocks_labeled.csv")
MODEL_SIGNAL = os.path.join(ROOT, "models", "signal_model.pkl")
MODEL_TRADE = os.path.join(ROOT, "models", "trade_outcome_model.pkl")
SCALER_FILE = os.path.join(ROOT, "models", "scaler.pkl")

# -----------------------
# Backtest parameters
# -----------------------
START_CAPITAL = 100000.0
ALLOCATION_PER_TRADE = 0.05
WIN_PROB_THRESHOLD = 0.60
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MAX_HORIZON_DAYS = 5
MIN_CASH_FOR_TRADE = 100.0
ALLOW_MULTIPLE_POSITIONS = True

# -----------------------
# Load models and data
# -----------------------
signal_info = joblib.load(MODEL_SIGNAL)
trade_info = joblib.load(MODEL_TRADE)
scaler = joblib.load(SCALER_FILE)
model_signal = signal_info["model"]
model_trade = trade_info["model"]
num_feats = signal_info["num_feats"]
ticker_dummies = signal_info["ticker_dummies"]
inverse_label_map = signal_info.get("label_map", {0: -1, 1: 0, 2: 1})

df = pd.read_csv(DATA_FILE, parse_dates=["Datetime"])
df = df.sort_values(["Datetime", "Ticker"]).reset_index(drop=True)

available_feats = [f for f in num_feats if f in df.columns]
df = df.dropna(subset=available_feats + ["Open", "High", "Low", "Close", "Volume"])

# -----------------------
# Simulation
# -----------------------
dates = sorted(df["Datetime"].unique())
cash = START_CAPITAL
equity = START_CAPITAL
trade_log = []
equity_curve = []

for current_date in dates:
    today_rows = df[df["Datetime"] == current_date]

    if today_rows.empty:
        continue

    # === Build batch features ===
    X_num = today_rows[available_feats].astype(float)
    X_num_scaled = scaler.transform(X_num)

    X_dummies = np.zeros((len(today_rows), len(ticker_dummies)))
    for i, t in enumerate(today_rows["Ticker"]):
        col = f"T_{t}"
        if col in ticker_dummies:
            idx = ticker_dummies.index(col)
            X_dummies[i, idx] = 1.0

    X_batch = np.hstack([X_num_scaled, X_dummies])

    # === Model predictions ===
    sig_probs_batch = model_signal.predict_proba(X_batch)
    trade_probs_batch = model_trade.predict_proba(X_batch)

    # === Loop through tickers of today ===
    for i, row in enumerate(today_rows.itertuples(index=False)):
        sig_probs = sig_probs_batch[i]
        sig_pred_idx = int(np.argmax(sig_probs))
        sig_label = inverse_label_map[sig_pred_idx]  # -1,0,1

        if sig_label == 0:
            side = "SELL"
        elif sig_label == 1:
            side = "HOLD"
        else:
            side = "BUY"

        if side == "HOLD":
            continue

        # trade outcome check
        trade_probs = trade_probs_batch[i]
        win_prob = float(trade_probs[-1]) if len(trade_probs) < 3 else float(trade_probs[2])
        if win_prob < WIN_PROB_THRESHOLD:
            continue

        # entry = next day's open
        future_rows = df[(df["Ticker"] == row.Ticker) & (df["Datetime"] > current_date)].sort_values("Datetime")
        if future_rows.empty:
            continue
        entry_row = future_rows.iloc[0]
        entry_price = float(entry_row["Open"]) if not pd.isna(entry_row["Open"]) else float(entry_row["Close"])
        atr = float(entry_row.get("ATR_14", entry_price * 0.01))
        if atr <= 0: atr = entry_price * 0.01

        if side == "BUY":
            sl_price = entry_price - SL_ATR_MULT * atr
            tp_price = entry_price + TP_ATR_MULT * atr
        else:
            sl_price = entry_price + SL_ATR_MULT * atr
            tp_price = entry_price - TP_ATR_MULT * atr

        alloc_amount = equity * ALLOCATION_PER_TRADE
        if alloc_amount < MIN_CASH_FOR_TRADE:
            continue
        shares = int(alloc_amount // entry_price)
        if shares <= 0:
            continue

        # monitor outcome
        outcome, exit_price, exit_date = None, None, None
        future_window = future_rows.iloc[:MAX_HORIZON_DAYS]
        for _, frow in future_window.iterrows():
            h, l, d = float(frow["High"]), float(frow["Low"]), frow["Datetime"]
            if side == "BUY":
                if h >= tp_price:
                    exit_price, exit_date, outcome = tp_price, d, "TP"; break
                if l <= sl_price:
                    exit_price, exit_date, outcome = sl_price, d, "SL"; break
            else:  # SELL
                if l <= tp_price:
                    exit_price, exit_date, outcome = tp_price, d, "TP"; break
                if h >= sl_price:
                    exit_price, exit_date, outcome = sl_price, d, "SL"; break
        if outcome is None:
            last_row = future_window.iloc[-1]
            exit_price, exit_date, outcome = float(last_row["Close"]), last_row["Datetime"], "EXIT"

        # pnl
        pnl = (exit_price - entry_price) * shares if side == "BUY" else (entry_price - exit_price) * shares
        cash += pnl
        equity += pnl

        trade_log.append({
            "entry_date": entry_row["Datetime"], "exit_date": exit_date,
            "ticker": row.Ticker, "side": side,
            "entry_price": entry_price, "exit_price": exit_price,
            "sl": sl_price, "tp": tp_price,
            "shares": shares, "pnl": pnl,
            "win_prob": win_prob, "outcome": outcome
        })

    equity_curve.append({"date": current_date, "equity": equity})

# -----------------------
# Results
# -----------------------
trades_df = pd.DataFrame(trade_log)
equity_df = pd.DataFrame(equity_curve).drop_duplicates("date").set_index("date").sort_index()

os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
if not trades_df.empty:
    trades_df.to_csv(os.path.join(ROOT, "output", "sim_trades.csv"), index=False)
    print("Saved trades to output/sim_trades.csv")

equity_df.to_csv(os.path.join(ROOT, "output", "equity_curve.csv"))
print("Saved equity curve to output/equity_curve.csv")

# Summary
if not trades_df.empty:
    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    win_rate = len(wins) / total_trades
    total_pnl = trades_df["pnl"].sum()
    avg_pnl = trades_df["pnl"].mean()
    equity_series = equity_df["equity"]
    drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax()
    max_dd = drawdown.min()

    print("\n===== BACKTEST SUMMARY =====")
    print(f"Start capital: {START_CAPITAL:.2f}")
    print(f"End equity: {equity:.2f}")
    print(f"Net PnL: {equity - START_CAPITAL:.2f}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average PnL per trade: {avg_pnl:.2f}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Max drawdown: {max_dd:.2%}")
else:
    print("No trades taken under current parameters.")
