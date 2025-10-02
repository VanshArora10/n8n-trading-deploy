# src/add_indicators_multi.py
import os
import pandas as pd
import pandas_ta as ta

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN = os.path.join(ROOT, "data", "all_stocks_raw.csv")
OUT = os.path.join(ROOT, "data", "all_stocks_indicators.csv")

df = pd.read_csv(IN, parse_dates=["Datetime"])
df = df.sort_values(["Ticker", "Datetime"]).reset_index(drop=True)

rows = []
for ticker, g in df.groupby("Ticker"):
    g = g.set_index("Datetime").copy()

    # 🔑 Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")

    # 📈 Indicators
    g["RSI_14"] = ta.rsi(g["Close"], length=14)
    g["EMA_20"] = ta.ema(g["Close"], length=20)
    g["EMA_50"] = ta.ema(g["Close"], length=50)

    macd = ta.macd(g["Close"])
    if macd is not None and not macd.empty:
        if "MACD_12_26_9" in macd.columns:
            g["MACD"] = macd["MACD_12_26_9"]
            g["MACD_Signal"] = macd["MACDs_12_26_9"]
            g["MACD_Hist"] = macd["MACDh_12_26_9"]
        else:
            g["MACD"] = macd.iloc[:, 0]
            g["MACD_Signal"] = macd.iloc[:, 1]
            g["MACD_Hist"] = macd.iloc[:, 2]

    bb = ta.bbands(g["Close"], length=20)
    if bb is not None and not bb.empty:
        g = g.join(bb)
        # 🧹 Rename Bollinger columns to clean names
        for c in list(g.columns):
            if "BBL" in c:
                g.rename(columns={c: "BB_lower"}, inplace=True)
            elif "BBU" in c:
                g.rename(columns={c: "BB_upper"}, inplace=True)
            elif "BBM" in c:
                g.rename(columns={c: "BB_mid"}, inplace=True)

    g = g.reset_index()
    g["Ticker"] = ticker
    rows.append(g)

df_out = pd.concat(rows, ignore_index=True)
df_out = df_out.dropna()  # drop rows without full indicator history
df_out.to_csv(OUT, index=False)
print("✅ Saved", OUT)
