import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_SIGNAL = os.path.join(ROOT, "models", "signal_model.pkl")
MODEL_TRADE_OUTCOME = os.path.join(ROOT, "models", "trade_outcome_model.pkl")
SCALER_FILE = os.path.join(ROOT, "models", "scaler.pkl")

# Load Models & Scaler
signal_info = joblib.load(MODEL_SIGNAL)
trade_info = joblib.load(MODEL_TRADE_OUTCOME)
scaler = joblib.load(SCALER_FILE)

model_signal = signal_info["model"]
model_trade = trade_info["model"]
num_feats = signal_info["num_feats"]
ticker_dummies = signal_info["ticker_dummies"]
inverse_label_map = {0: -1, 1: 0, 2: 1}

app = FastAPI()

class StockData(BaseModel):
    features: dict  # key:value of features like RSI_14, EMA_20 etc.
    ticker: str
    close_price: float
    atr_14: float

@app.post("/trade_plan")
def trade_plan(data: StockData):
    # 1. Prepare input features
    feat_dict = data.features

    # Convert to DataFrame for scaler
    X_num = pd.DataFrame([feat_dict])[num_feats].astype(float)

    # Scale numeric features
    X_scaled = scaler.transform(X_num)

    # Add ticker dummies
    tick_df = pd.DataFrame(columns=ticker_dummies)
    for col in ticker_dummies:
        tick_df.loc[0, col] = 1 if col == f"T_{data.ticker}" else 0

    # Final feature vector
    X_final = np.hstack([X_scaled, tick_df.values])

    # 2. Model 1 Prediction
    probs_sig = model_signal.predict_proba(X_final)[0]
    pred_sig = np.argmax(probs_sig)
    signal = inverse_label_map[pred_sig]

    # 3. If BUY or SELL → Model 2 Prediction
    trade_plan = {
        "signal": "HOLD",
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "win_probability": None
    }

    if signal in [-1, 1]:  # SELL or BUY
        # Model 2 Prediction
        probs_trade = model_trade.predict_proba(X_final)[0]
        win_prob = probs_trade[2] if len(probs_trade) == 3 else probs_trade[1]

        if win_prob >= 0.6:  # Only trade if high win probability
            entry = data.close_price
            atr = data.atr_14

            if signal == 1:  # BUY
                sl = entry - 1.5 * atr
                tp = entry + 3 * atr
            else:  # SELL
                sl = entry + 1.5 * atr
                tp = entry - 3 * atr

            trade_plan = {
                "signal": "BUY" if signal == 1 else "SELL",
                "entry_price": round(entry, 2),
                "stop_loss": round(sl, 2),
                "take_profit": round(tp, 2),
                "win_probability": round(win_prob, 2)
            }

    return trade_plan
