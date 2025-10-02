import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN = os.path.join(ROOT, "data", "all_stocks_labeled.csv")

# Output models
MODEL_SIGNAL_OUT = os.path.join(ROOT, "models", "signal_model.pkl")
MODEL_TRADE_OUTCOME_OUT = os.path.join(ROOT, "models", "trade_outcome_model.pkl")
SCALER_OUT = os.path.join(ROOT, "models", "scaler.pkl")

# -----------------------
# 1. Load Dataset
# -----------------------
df = pd.read_csv(IN, parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

# Features
num_feats = [
    "RSI_14", "EMA_20", "EMA_50",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_lower", "BB_upper", "BB_mid",
    "Volume", "sentiment",
    "return_1d", "RSI_diff", "MACD_cross", "Volatility",
    "return_2d", "return_5d", "ATR_14", "BB_width",
    "candle_body", "upper_wick", "lower_wick", "EMA_ratio", "Close_to_EMA50",
    "Rel_Volume", "RSI_overbought", "RSI_oversold",
    "MACD_signal_cross", "Close_below_EMA20", "Close_below_EMA50",
    "RSI_strong_overbought", "MACD_bearish_cross",
    "neg_return_3d"
]

# Keep only available features
available_feats = [f for f in num_feats if f in df.columns]
df = df.dropna(subset=available_feats + ["target"])

# One-hot encode tickers
df = pd.get_dummies(df, columns=["Ticker"], prefix="T")

# -----------------------
# 2. Remap Target for Model 1 (Signal Prediction)
# -----------------------
# -1 → 0 (SELL), 0 → 1 (HOLD), 1 → 2 (BUY)
label_map = {-1: 0, 0: 1, 1: 2}
inverse_label_map = {0: -1, 1: 0, 2: 1}
df["target"] = df["target"].map(label_map)

# -----------------------
# 3. Split Train/Test
# -----------------------
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

# -----------------------
# 4. Train Model 1 (Signal prediction)
# -----------------------
X_train_sig = train_df[available_feats + list(train_df.filter(regex="^T_").columns)].astype(float)
y_train_sig = train_df["target"].astype(int)

X_test_sig = test_df[available_feats + list(test_df.filter(regex="^T_").columns)].astype(float)
y_test_sig = test_df["target"].astype(int)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[available_feats])
X_test_scaled = scaler.transform(test_df[available_feats])

X_train_sig_final = np.hstack([X_train_scaled, train_df.filter(regex="^T_").values])
X_test_sig_final = np.hstack([X_test_scaled, test_df.filter(regex="^T_").values])

# Balance with SMOTE
sm = SMOTE(random_state=42)
X_train_sig_final, y_train_sig = sm.fit_resample(X_train_sig_final, y_train_sig)

model_signal = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

print("\n🚀 Training Model 1 (Signal Prediction: BUY/SELL/HOLD)...")
model_signal.fit(X_train_sig_final, y_train_sig)

# Evaluate
preds_sig = model_signal.predict(X_test_sig_final)
print("\n📊 Classification Report (Signal Model):")
print(classification_report(y_test_sig, preds_sig))
print("⚡ Macro F1-score:", f1_score(y_test_sig, preds_sig, average="macro"))

# Save
joblib.dump(
    {"model": model_signal, "num_feats": available_feats,
     "ticker_dummies": list(df.filter(regex='^T_').columns),
     "label_map": inverse_label_map},
    MODEL_SIGNAL_OUT
)

# -----------------------
# 5. Train Model 2 (Trade Outcome prediction)
# -----------------------
# Filter rows where trade_outcome exists (only when entry_price set)
if "trade_outcome" in df.columns:
    trade_df = df.dropna(subset=["trade_outcome"])

    # Map trade_outcome to 0,1,2
    trade_map = {-1: 0, 0: 1, 1: 2}
    inverse_trade_map = {0: -1, 1: 0, 2: 1}
    trade_df["trade_outcome"] = trade_df["trade_outcome"].map(trade_map)

    X_trade = trade_df[available_feats + list(trade_df.filter(regex="^T_").columns)].astype(float)
    y_trade = trade_df["trade_outcome"].astype(int)

    split_idx2 = int(len(trade_df) * 0.8)
    X_train_to, X_test_to = X_trade.iloc[:split_idx2], X_trade.iloc[split_idx2:]
    y_train_to, y_test_to = y_trade.iloc[:split_idx2], y_trade.iloc[split_idx2:]

    # Scale
    X_train_to_scaled = scaler.fit_transform(X_train_to[available_feats])
    X_test_to_scaled = scaler.transform(X_test_to[available_feats])

    X_train_to_final = np.hstack([X_train_to_scaled, X_train_to.drop(columns=available_feats).values])
    X_test_to_final = np.hstack([X_test_to_scaled, X_test_to.drop(columns=available_feats).values])

    # Balance with SMOTE
    X_train_to_final, y_train_to = sm.fit_resample(X_train_to_final, y_train_to)

    model_trade_outcome = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )

    print("\n🚀 Training Model 2 (Trade Outcome Prediction: WIN/LOSS/NEUTRAL)...")
    model_trade_outcome.fit(X_train_to_final, y_train_to)

    # Evaluate
    preds_to = model_trade_outcome.predict(X_test_to_final)
    print("\n📊 Classification Report (Trade Outcome Model):")
    print(classification_report(y_test_to, preds_to))
    print("⚡ Macro F1-score:", f1_score(y_test_to, preds_to, average="macro"))

    # Save
    joblib.dump(
        {"model": model_trade_outcome, "num_feats": available_feats,
         "ticker_dummies": list(df.filter(regex='^T_').columns),
         "label_map": inverse_trade_map},
        MODEL_TRADE_OUTCOME_OUT
    )

# Save scaler
joblib.dump(scaler, SCALER_OUT)

print("\n✅ Models and scaler saved:")
print(" - Signal Model:", MODEL_SIGNAL_OUT)
print(" - Trade Outcome Model:", MODEL_TRADE_OUTCOME_OUT)
print(" - Scaler:", SCALER_OUT)
