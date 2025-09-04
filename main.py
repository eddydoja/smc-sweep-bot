import os
import time
import requests
import schedule
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

TICKERS = [
    "SPY", "QQQ", "DIA",
    "DAX", "UKX",
    "EURUSD", "GBPUSD", "USDJPY",
    "AUDUSD", "USDCAD"
]

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")
open_positions = {}
last_signal_time = {}

SESSION_FILTER = {
    'FX': (3, 17),  # London 3am–5pm UTC
    'US': (13, 20)  # NY 9am–4pm EST (13–20 UTC approx)
}


def is_market_open_now():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    return now.weekday() < 5 and ((now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16))

def is_session_active(ticker):
    now_hour = datetime.now(pytz.UTC).hour
    if ticker in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]:
        start, end = SESSION_FILTER['FX']
    else:
        start, end = SESSION_FILTER['US']
    return start <= now_hour < end

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

def get_data(ticker, timeframe=TimeFrame.Minute, limit=100):
    try:
        bars = client.get_bars(ticker, timeframe, limit=min(limit, 50)).df
        if bars.empty or len(bars) < 30:
            raise ValueError("Insufficient data")
        bars['body'] = abs(bars['close'] - bars['open'])
        bars['range'] = bars['high'] - bars['low']
        bars['upper_wick'] = bars['high'] - bars[['open', 'close']].max(axis=1)
        bars['lower_wick'] = bars[['open', 'close']].min(axis=1) - bars['low']
        bars['is_doji'] = (bars['body'] / bars['range']) < 0.1
        bars['bullish_engulfing'] = (bars['close'].shift(1) < bars['open'].shift(1)) & (bars['close'] > bars['open']) & (bars['close'] > bars['open'].shift(1)) & (bars['open'] < bars['close'].shift(1))
        bars['bearish_engulfing'] = (bars['close'].shift(1) > bars['open'].shift(1)) & (bars['close'] < bars['open']) & (bars['close'] < bars['open'].shift(1)) & (bars['open'] > bars['close'].shift(1))
        bars['hammer'] = (bars['body'] / bars['range'] < 0.3) & (bars['lower_wick'] > bars['body'])
        bars['inverted_hammer'] = (bars['body'] / bars['range'] < 0.3) & (bars['upper_wick'] > bars['body'])
        bars['wick_rejection_up'] = bars['upper_wick'] > (bars['range'] * 0.6)
        bars['wick_rejection_down'] = bars['lower_wick'] > (bars['range'] * 0.6)
        bars['3bar_bullish'] = (bars['close'].shift(2) < bars['open'].shift(2)) & (bars['is_doji'].shift(1)) & (bars['close'] > bars['open'])
        bars['3bar_bearish'] = (bars['close'].shift(2) > bars['open'].shift(2)) & (bars['is_doji'].shift(1)) & (bars['close'] < bars['open'])
        bars['liquidity_sweep_high'] = bars['high'] > bars['high'].rolling(window=20).max().shift(1)
        bars['liquidity_sweep_low'] = bars['low'] < bars['low'].rolling(window=20).min().shift(1)
        return bars
    except Exception as e:
        send_telegram(f"⚠️ Data fetch failed for {ticker}: {e}")
        return pd.DataFrame()

def check_smc():
    if not is_market_open_now():
        return

    total_trades = sum(len(p) for p in open_positions.values())
    if total_trades >= 3:
        return

    for ticker in TICKERS:
        if not is_session_active(ticker):
            continue

        if ticker not in open_positions:
            open_positions[ticker] = []

        if open_positions[ticker]:
            continue

        last_signal = last_signal_time.get(ticker)
        if last_signal and datetime.now(pytz.UTC) - last_signal < timedelta(minutes=1):
            continue

        # (Signal generation logic continues unchanged...)

        last_signal_time[ticker] = datetime.now(pytz.UTC)

# (Other unchanged functions like detect_structure, etc., remain below)

def check_positions():
    for ticker, positions in open_positions.items():
        for pos in positions[:]:
            try:
                order = client.get_order(pos['id'])
                if order.filled_avg_price:
                    current_price = client.get_latest_trade(ticker).price
                    gain = (current_price - pos['entry']) / pos['entry'] * 100 if pos['side'] == 'long' else (pos['entry'] - current_price) / pos['entry'] * 100
                    if gain <= -1 or gain >= pos['tp']:
                        close_side = "sell" if pos['side'] == "long" else "buy"
                        client.submit_order(
                            symbol=ticker,
                            qty=pos['qty'],
                            side=close_side,
                            type="market",
                            time_in_force="gtc"
                        )
                        send_telegram(f"📤 Exited {pos['side'].upper()} {pos['qty']} {ticker} at gain/loss: {gain:.2f}%")
                        positions.remove(pos)
            except Exception as e:
                send_telegram(f"⚠️ Error checking {ticker} position: {e}")

schedule.every(30).seconds.do(check_smc)
schedule.every(30).seconds.do(check_positions)

try:
    send_telegram("✅ Multi-Asset SMC Bot started.")
    print("Bot is running...", flush=True)
except Exception as e:
    print(f"[Startup Error] {e}", flush=True)

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as loop_error:
        print(f"[Loop Error] {loop_error}", flush=True)
        time.sleep(5)
