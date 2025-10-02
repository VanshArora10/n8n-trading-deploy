# src/news_sentiment.py
import os
import requests
from typing import Dict, Tuple

# Simple keyword-based sentiment fallback (works offline)
POSITIVE = {"beats","profit","gain","upgrade","buy","bull","growth","beats expectations","surge","outperform"}
NEGATIVE = {"loss","miss","downgrade","sell","bear","fall","falls","decline","losses","fraud","investigation"}

def simple_headline_sentiment(headlines):
    """Return ('Positive'|'Negative'|'Neutral', score) where score [-1..1]"""
    score = 0.0
    for h in headlines:
        text = h.lower()
        for p in POSITIVE:
            if p in text:
                score += 1.0
        for n in NEGATIVE:
            if n in text:
                score -= 1.0
    if score > 0:
        return "Positive", min(1.0, score / (len(headlines) + 1))
    if score < 0:
        return "Negative", max(-1.0, score / (len(headlines) + 1))
    return "Neutral", 0.0

def fetch_headlines_gnews(query, max_results=5):
    """
    Optional: if you have a GNews or NewsAPI key, implement here.
    For now this function returns empty list (so fallback to simple logic).
    """
    return []

def get_news_sentiment(ticker: str) -> Tuple[str,float]:
    """
    Returns (label, score) where label in {'Positive','Neutral','Negative'}.
    This is a cheap fallback using keywords; you can replace with NewsAPI/GNews + a transformer sentiment later.
    """
    q = ticker.split(".")[0]  # simple query like RELIANCE
    headlines = fetch_headlines_gnews(q, max_results=6)
    if not headlines:
        # fallback: no headlines -> Neutral
        return "Neutral", 0.0
    return simple_headline_sentiment(headlines)

if __name__ == "__main__":
    print(get_news_sentiment("RELIANCE.NS"))
