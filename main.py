import os
import time
import requests
import schedule
import pandas as pd
import random
from datetime import datetime
import pytz
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, TimeInForce, OrderSide

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TICKER = "SPY"
MAX_TRADES = 3
TRAIL_THRESHOLD = 0.03  # 3% initial TP trigger
TRAIL_DROP = 0.01       # 1% drop from peak triggers exit
STOP_LOSS_PCT = 0.02

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")
open_trades = {}

def send_telegram(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        print("[Telegram]", msg)
    except Exception as e:
        print("[Telegram Error]", e)

def is_market_open():
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    return now.weekday() < 5 and ((now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16))

def get_data():
    bars = client.get_bars(TICKER, TimeFrame.Minute, limit=3).df
    return bars if not bars.empty and 'close' in bars.columns and 'open' in bars.columns else pd.DataFrame()

def biased_tp(entry, direction):
    if direction == "buy":
        low, high = 1.03, 1.05
    else:
        low, high = 0.95, 0.97
    r = random.random() ** 0.5
    return round(entry * (low + (high - low) * r), 2)

def place_trade(direction, entry):
    if len(open_trades) >= MAX_TRADES:
        print("Max trades reached, skipping.")
        return

    sl = round(entry * (1 - STOP_LOSS_PCT), 2) if direction == "buy" else round(entry * (1 + STOP_LOSS_PCT),2)
    tp = biased_tp(entry, direction)

    order = client.submit_order(
        symbol=TICKER, qty=1, side=OrderSide.BUY if direction == "buy" else OrderSide.SELL,
        type="market", time_in_force=TimeInForce.GTC
    )
    open_trades[order.id] = {
        "entry": entry, "side": direction, "sl": sl, "tp": tp,
        "peak": entry, "time": datetime.now(pytz.timezone("US/Eastern"))
    }
    send_telegram(f"{direction.upper()} @ {entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

def manage_exits():
    if not open_trades: return
    price = client.get_latest_trade(TICKER).price

    for oid in list(open_trades):
        t = open_trades[oid]
        t["peak"] = max(t["peak"], price) if t["side"] == "buy" else min(t["peak"], price)
        drop_amount = t["peak"] * TRAIL_DROP
        if (t["side"] == "buy" and price <= t["peak"] - drop_amount) or (t["side"] == "sell" and price >= t["peak"] + drop_amount):
            client.submit_order(symbol=TICKER, qty=1,
                                side=OrderSide.SELL if t["side"] == "buy" else OrderSide.BUY,
                                type="market", time_in_force=TimeInForce.GTC)
            pnl = (price - t["entry"]) if t["side"] == "buy" else (t["entry"] - price)
            pct = pnl / t["entry"] * 100
            dur = datetime.now(pytz.timezone("US/Eastern")) - t["time"]
            send_telegram(f"EXIT {t['side'].upper()} @ {price:.2f} | P/L: {pct:.2f}% | Duration: {str(dur).split('.')[0]}")
            del open_trades[oid]

def force_exit_before_close():
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    if now.hour == 15 and now.minute >= 55 and open_trades:
        for oid in list(open_trades):
            price = client.get_latest_trade(TICKER).price
            side = OrderSide.SELL if open_trades[oid]["side"] == "buy" else OrderSide.BUY
            client.submit_order(symbol=TICKER, qty=1, side=side,
                                type="market", time_in_force=TimeInForce.GTC)
            send_telegram(f"Force exit {open_trades[oid]['side'].upper()} @ {price:.2f} at EOD")
            del open_trades[oid]

def check_smc():
    if not is_market_open(): return
    if len(open_trades) >= MAX_TRADES: return
    df = get_data()
    if df.shape[0] < 3: return

    c2, c3 = df.iloc[-2], df.iloc[-1]
    dirn = None
    if c2['close'] < c2['open'] and c3['close'] > c3['open'] and c3['close'] > c2['open']:
        dirn = "buy"
    elif c2['close'] > c2['open'] and c3['close'] < c3['open'] and c3['close'] < c2['open']:
        dirn = "sell"

    if dirn:
        place_trade(dirn, c3['close'])

schedule.every(1).minutes.do(check_smc)
schedule.every(1).minutes.do(manage_exits)
schedule.every().day.at("15:55").do(force_exit_before_close)

send_telegram("Bot started with smart trailing TP logic.")
print("Bot is running...")

while True:
    schedule.run_pending()
    time.sleep(1)
