import os
import time
import requests
import schedule
import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

load_dotenv()

# Load secrets
ALPACA_API_KEY = os.getenv("PKW2QGFBW74BYLLS48MS")
ALPACA_SECRET_KEY = os.getenv("V8K9NaWTpYdL9NNuQRG54e2EvvdTsXPBzrmUCVMI")
TELEGRAM_CHAT_ID = os.getenv(5079232641)
TELEGRAM_TOKEN = os.getenv("8405020655:AAHff_dafcrxkrLKLfQ1zjpYGKAudSM8H7w")
TICKER = "SPY"

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")

def get_data():
    bars = client.get_bars(TICKER, TimeFrame.Minute, limit=500).df
    bars = bars.tail(3)  # Last 3 candles
    bars['body'] = abs(bars['close'] - bars['open'])
    bars['range'] = bars['high'] - bars['low']
    return bars

def check_smc():
    df = get_data()

    if df.shape[0] < 3:
        return

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    bullish_engulfing = c2['close'] < c2['open'] and c3['close'] > c3['open'] and c3['close'] > c2['open'] and c3['open'] < c2['close']
    bearish_engulfing = c2['close'] > c2['open'] and c3['close'] < c3['open'] and c3['close'] < c2['open'] and c3['open'] > c2['close']

    if bullish_engulfing:
        send_telegram(f"Bullish SMC Detected on {TICKER}")
    elif bearish_engulfing:
        send_telegram(f"Bearish SMC Detected on {TICKER}")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram error: {e}")

schedule.every(5).minutes.do(check_smc)

print("Bot is running...")

while True:
    schedule.run_pending()
    time.sleep(1)
