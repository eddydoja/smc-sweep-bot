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
TICKER = "SPY"

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")
open_positions = []

def is_market_open_now():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    return now.weekday() < 5 and ((now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16))

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

def determine_qty(rating):
    return min(10, max(1, rating))

def check_positions():
    global open_positions
    current_time = datetime.now(pytz.timezone("US/Eastern"))
    to_close = []
    for pos in open_positions:
        try:
            order = client.get_order(pos['id'])
            if order.filled_avg_price:
                current_price = client.get_latest_trade(TICKER).price
                gain = (current_price - pos['entry']) / pos['entry'] * 100 if pos['side'] == 'buy' else (pos['entry'] - current_price) / pos['entry'] * 100
                if gain <= -2 or gain >= pos['tp']:
                    side = "sell" if pos['side'] == "buy" else "buy"
                    client.submit_order(
                        symbol=TICKER,
                        qty=pos['qty'],
                        side=side,
                        type="market",
                        time_in_force="gtc"
                    )
                    send_telegram(f"📤 Exited {pos['side'].upper()} {pos['qty']} {TICKER} at gain/loss: {gain:.2f}%")
                    to_close.append(pos)
        except Exception as e:
            print(f"Error checking positions: {e}", flush=True)

    open_positions = [pos for pos in open_positions if pos not in to_close]

def check_smc():
    if not is_market_open_now():
        print("Market is closed.", flush=True)
        return

    if len(open_positions) >= 3:
        print("Max open positions reached.", flush=True)
        return

    df = get_data()
    if df.shape[0] < 3:
        print("Not enough data to check SMC.", flush=True)
        return

    c2, c3 = df.iloc[-2], df.iloc[-1]

    bullish_engulfing = c2['close'] < c2['open'] and c3['close'] > c3['open'] and c3['close'] > c2['open'] and c3['open'] < c2['close']
    bearish_engulfing = c2['close'] > c2['open'] and c3['close'] < c3['open'] and c3['close'] < c2['open'] and c3['open'] > c2['close']

    if bullish_engulfing or bearish_engulfing:
        side = "buy" if bullish_engulfing else "sell"
        entry_price = float(c3['close'])
        sl = round(entry_price * 0.98, 2)
        tp = round(entry_price * (1.04 + (0.01 * bool(entry_price % 2 == 0))), 2)
        qty = determine_qty(rating=7)

        try:
            order = client.submit_order(
                symbol=TICKER,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc",
                order_class="bracket",
                stop_loss={"stop_price": str(sl)},
                take_profit={"limit_price": str(tp)}
            )
            open_positions.append({
                'id': order.id,
                'side': side,
                'entry': entry_price,
                'tp': 4,
                'qty': qty,
                'timestamp': datetime.now(pytz.timezone("US/Eastern"))
            })
            send_telegram(f"📥 Entered {side.upper()} {qty} {TICKER} @ {entry_price}\nTP: {tp} | SL: {sl}")
        except Exception as e:
            print(f"Order error: {e}", flush=True)
    else:
        print("No signal detected.", flush=True)

# Start scheduler
schedule.every(1).minutes.do(check_smc)
schedule.every(2).minutes.do(check_positions)

try:
    send_telegram("✅ SMC Sweep Bot has started successfully.")
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
