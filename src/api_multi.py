import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_FILE = os.path.join(ROOT, "models", "general_model.pkl")
SCALER_FILE = os.path.join(ROOT, "models", "scaler.pkl")

# Load model + scaler
info = joblib.load(MODEL_FILE)
model = info["model"]
num_feats = info["num_feats"]
ticker_dummies = info["ticker_dummies"]
scaler = joblib.load(SCALER_FILE)

# --- API Setup ---
app = FastAPI(title="Trading Prediction API")

# Input format
class StockData(BaseModel):
    ticker: str
    data: dict  # dictionary of features (RSI, EMA, etc.)


@app.post("/predict")
def predict(input_data: StockData):
    # Prepare input
    row = pd.DataFrame([input_data.data])

    # Ensure all numerical features exist
    for col in num_feats:
        if col not in row.columns:
            row[col] = 0.0

    # Scale numeric
    X_num = scaler.transform(row[num_feats].astype(float))

    # One-hot encode ticker
    ticker_row = {col: 0 for col in ticker_dummies}
    col_name = f"T_{input_data.ticker}"
    if col_name in ticker_row:
        ticker_row[col_name] = 1
    X_dummies = np.array([list(ticker_row.values())])

    # Final input
    X_final = np.hstack([X_num, X_dummies])

    # Predict
    pred = model.predict(X_final)[0]
    signal = {1: "BUY", -1: "SELL", 0: "HOLD"}[pred]

    return {"ticker": input_data.ticker, "prediction": signal}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
