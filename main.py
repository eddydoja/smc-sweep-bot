import os
import time
import requests
import schedule
import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

load_dotenv()

# Load secrets from environment variables (NOT raw values)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TICKER = "SPY"

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")

def get_data():
    bars = client.get_bars(TICKER, TimeFrame.Minute, limit=500).df
    bars = bars.tail(3)
    bars['body'] = abs(bars['close'] - bars['open'])
    bars['range'] = bars['high'] - bars['low']
    return bars

def check_smc():
    df = get_data()
    if df.shape[0] < 3:
        return

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    bullish = (
        c2['close'] < c2['open'] and
        c3['close'] > c3['open'] and
        c3['close'] > c2['open'] and
        c3['open'] < c2['close']
    )

    bearish = (
        c2['close'] > c2['open'] and
        c3['close'] < c3['open'] and
        c3['close'] < c2['open'] and
        c3['open'] > c2['close']
    )

    if bullish:
        send_telegram(f"Bullish SMC Detected on {TICKER}")
    elif bearish:
        send_telegram(f"Bearish SMC Detected on {TICKER}")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
    except Exception as e:
        print(f"Telegram error: {e}")

schedule.every(5).minutes.do(check_smc)

print("Bot is running...")

while True:
    schedule.run_pending()
    time.sleep(1)
