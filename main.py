import os
import time
import requests
import schedule
import pandas as pd
from datetime import datetime
import pytz
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

load_dotenv()

# Load environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TICKER = "SPY"

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")

def is_market_open_now():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    return now.weekday() < 5 and ((now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16))

def get_data():
    try:
        bars = client.get_bars(TICKER, TimeFrame.Minute, limit=3).df
        if bars.empty or 'open' not in bars.columns or 'close' not in bars.columns:
            print("Data fetch failed or missing expected columns.", flush=True)
            return pd.DataFrame()
        bars['body'] = abs(bars['close'] - bars['open'])
        bars['range'] = bars['high'] - bars['low']
        return bars
    except Exception as e:
        print(f"Error fetching data: {e}", flush=True)
        return pd.DataFrame()

def check_smc():
    if not is_market_open_now():
        print("Market is closed.", flush=True)
        return

    df = get_data()
    if df.shape[0] < 3:
        print("Not enough data to check SMC.", flush=True)
        return

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    bullish = c2['close'] < c2['open'] and c3['close'] > c3['open'] and c3['close'] > c2['open'] and c3['open'] < c2['close']
    bearish = c2['close'] > c2['open'] and c3['close'] < c3['open'] and c3['close'] < c2['open'] and c3['open'] > c2['close']

    if bullish:
        send_telegram(f"📈 Bullish SMC Detected on {TICKER}")
    elif bearish:
        send_telegram(f"📉 Bearish SMC Detected on {TICKER}")
    else:
        print("No signal detected.", flush=True)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.ok:
            print(f"Sent Telegram message: {message}", flush=True)
        else:
            print(f"[Telegram Error] {response.status_code} - {response.text}", flush=True)
    except Exception as e:
        print(f"[Telegram Exception] {e}", flush=True)

# Start scheduler
schedule.every(1).minutes.do(check_smc)

# Only send startup message once
try:
    send_telegram("✅ SMC Sweep Bot has started successfully.")
    print("Bot is running...", flush=True)
except Exception as e:
    print(f"[Startup Error] {e}", flush=True)

# Main loop
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as loop_error:
        print(f"[Loop Error] {loop_error}", flush=True)
        time.sleep(5)
