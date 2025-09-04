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

def get_data(ticker):
    try:
        bars = client.get_bars(ticker, TimeFrame.Minute, limit=100).df
        if bars.empty:
            return pd.DataFrame()
        bars['body'] = abs(bars['close'] - bars['open'])
        bars['range'] = bars['high'] - bars['low']
        return bars
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}", flush=True)
        return pd.DataFrame()

def detect_structure(df):
    df = df.copy()
    df['is_hh'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    df['is_ll'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
    swing_highs = df[df['is_hh']]
    swing_lows = df[df['is_ll']]
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    latest_highs = swing_highs.iloc[-2:]
    latest_lows = swing_lows.iloc[-2:]

    trend = "neutral"
    bos = False
    if latest_highs.iloc[1]['high'] > latest_highs.iloc[0]['high'] and latest_lows.iloc[1]['low'] > latest_lows.iloc[0]['low']:
        trend = "bullish"
        bos = True
    elif latest_highs.iloc[1]['high'] < latest_highs.iloc[0]['high'] and latest_lows.iloc[1]['low'] < latest_lows.iloc[0]['low']:
        trend = "bearish"
        bos = True

    return {
        "trend": trend,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "bos": bos
    }

def detect_fvg(df):
    for i in range(2, len(df)):
        if df.iloc[i-2]['low'] > df.iloc[i]['high']:
            return {
                'start': df.iloc[i-2]['low'],
                'end': df.iloc[i]['high'],
                'index': i
            }
        if df.iloc[i]['low'] > df.iloc[i-2]['high']:
            return {
                'start': df.iloc[i]['low'],
                'end': df.iloc[i-2]['high'],
                'index': i
            }
    return None

def detect_order_block(df):
    for i in range(len(df)-3, 0, -1):
        if df.iloc[i]['close'] < df.iloc[i]['open'] and df.iloc[i+1]['close'] > df.iloc[i+1]['open']:
            return {
                'open': df.iloc[i]['open'],
                'close': df.iloc[i]['close']
            }
        if df.iloc[i]['close'] > df.iloc[i]['open'] and df.iloc[i+1]['close'] < df.iloc[i+1]['open']:
            return {
                'open': df.iloc[i]['open'],
                'close': df.iloc[i]['close']
            }
    return None

def determine_qty(rating):
    return min(10, max(1, rating))

def calculate_confidence_tp(df, side, entry_price):
    avg_range = df['range'][-20:].mean()
    confidence_multiplier = 2 if side == "buy" else 2.5
    tp_buffer = avg_range * confidence_multiplier
    if side == "buy":
        return round(entry_price + tp_buffer, 2)
    else:
        return round(entry_price - tp_buffer, 2)

def check_positions():
    for ticker, positions in open_positions.items():
        for pos in positions[:]:
            try:
                order = client.get_order(pos['id'])
                if order.filled_avg_price:
                    current_price = client.get_latest_trade(ticker).price
                    gain = (current_price - pos['entry']) / pos['entry'] * 100 if pos['side'] == 'long' else (pos['entry'] - current_price) / pos['entry'] * 100
                    if gain <= -1 or gain >= pos['tp']:  # SL = 1%, TP = dynamic %
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
                print(f"Error checking positions for {ticker}: {e}", flush=True)

def check_smc():
    if not is_market_open_now():
        print("Market is closed.", flush=True)
        return

    for ticker in TICKERS:
        if ticker not in open_positions:
            open_positions[ticker] = []

        if len(open_positions[ticker]) >= 1:
            continue

        df = get_data(ticker)
        if df.shape[0] < 100:
            continue

        structure = detect_structure(df)
        if not structure or not structure['bos']:
            continue

        fvg = detect_fvg(df)
        ob = detect_order_block(df)
        if not fvg or not ob:
            continue

        last_price = df.iloc[-1]['close']
        if structure['trend'] == "bullish" and last_price < fvg['start'] and last_price >= ob['close']:
            entry_signal = True
            side = "long"
        elif structure['trend'] == "bearish" and last_price > fvg['start'] and last_price <= ob['close']:
            entry_signal = True
            side = "short"
        else:
            entry_signal = False

        if not entry_signal:
            continue

        entry_price = float(last_price)
        sl = round(entry_price * 0.99, 2) if side == "long" else round(entry_price * 1.01, 2)
        tp = calculate_confidence_tp(df, side, entry_price)
        tp_pct = abs((tp - entry_price) / entry_price * 100)
        qty = determine_qty(rating=7)

        try:
            order = client.submit_order(
                symbol=ticker,
                qty=qty,
                side="buy" if side == "long" else "sell",
                type="market",
                time_in_force="gtc",
                order_class="bracket",
                stop_loss={"stop_price": str(sl)},
                take_profit={"limit_price": str(tp)}
            )
            open_positions[ticker].append({
                'id': order.id,
                'side': side,
                'entry': entry_price,
                'tp': tp_pct,
                'qty': qty,
                'timestamp': datetime.now(pytz.timezone("US/Eastern"))
            })
            send_telegram(f"📥 {ticker} SIGNAL: {side.upper()} {qty} @ {entry_price}\nTP: {tp} ({tp_pct:.1f}%) | SL: {sl} (1%)")
        except Exception as e:
            print(f"Order error on {ticker}: {e}", flush=True)

schedule.every(1).minutes.do(check_smc)
schedule.every(2).minutes.do(check_positions)

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
