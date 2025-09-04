# Existing imports and config
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
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# New scout trade config
SCOUT_TRADES_ENABLED = True
SCOUT_TRADE_COOLDOWN_MINUTES = 10
SCOUT_MIN_CONFIDENCE = 5
last_scout_signal_time = {}

# Toggle to execute trades vs. alert-only
TRADE_EXECUTION = True
HEARTBEAT_TELEGRAM = False  # keep heartbeats only on Render logs

# Map unsupported indices to ETFs Alpaca supports
TICKER_MAP = {
    "DAX": "EWG",  # iShares Germany ETF
    "UKX": "EWU"   # iShares UK ETF
}

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

from twelvedata import TDClient
twelve = TDClient(apikey=os.getenv("TWELVE_DATA_API_KEY"))

TWELVE_FX_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD"
}

# Dummy placeholder for check_smc to prevent NameError

def check_smc():
    pass

# Scout trade scanner

def scan_for_sweep_momentum_trades():
    if not SCOUT_TRADES_ENABLED:
        return

    total_trades = sum(len(p) for p in open_positions.values())
    if total_trades >= 10:
        return

    for ticker in TICKERS:
        if not is_session_active(ticker):
            continue

        now = datetime.now(pytz.UTC)
        last_time = last_scout_signal_time.get(ticker)
        if last_time and now - last_time < timedelta(minutes=SCOUT_TRADE_COOLDOWN_MINUTES):
            continue

        df = get_data(ticker, limit=10)
        if df.empty or df.shape[0] < 6:
            continue

        last6 = df.tail(6)
        sweep_low = last6.iloc[-2]['low'] < last6['low'].iloc[:-2].min()
        sweep_high = last6.iloc[-2]['high'] > last6['high'].iloc[:-2].max()
        candle = last6.iloc[-1]

        body = abs(candle['close'] - candle['open'])
        range_ = candle['high'] - candle['low']
        body_ratio = body / range_ if range_ > 0 else 0
        is_momentum = body_ratio > 0.6

        side = None
        if sweep_low and candle['close'] > candle['open'] and is_momentum:
            side = 'long'
        elif sweep_high and candle['close'] < candle['open'] and is_momentum:
            side = 'short'

        if not side:
            continue

        confidence = calculate_confidence(df, candle)
        if confidence < SCOUT_MIN_CONFIDENCE:
            continue

        entry = float(candle['close'])
        sl = round(entry * 0.995, 4) if side == 'long' else round(entry * 1.005, 4)
        tp = calculate_confidence_tp(df, side, entry)
        tp_pct = abs((tp - entry) / entry * 100)
        qty = determine_qty(confidence)

        try:
            if TRADE_EXECUTION:
                order = client.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='buy' if side == 'long' else 'sell',
                    type='market',
                    time_in_force='gtc',
                    order_class='bracket',
                    stop_loss={"stop_price": str(sl)},
                    take_profit={"limit_price": str(tp)}
                )
                order_id = order.id
            else:
                order_id = f"scout-{datetime.now(pytz.UTC).isoformat()}"

            open_positions.setdefault(ticker, []).append({
                'id': order_id,
                'side': side,
                'entry': entry,
                'tp': tp_pct,
                'qty': qty,
                'timestamp': now
            })
            last_scout_signal_time[ticker] = now
            send_telegram(f"⚡ SCOUT {ticker}: {side.upper()} {qty} @ {entry}\nTP: {tp} ({tp_pct:.1f}%) | SL: {sl} | Confidence: {confidence}/10")
        except Exception as e:
            send_telegram(f"❌ Scout order error on {ticker}: {e}")

def check_positions():
    for ticker, positions in open_positions.items():
        for pos in positions[:]:
            try:
                order = client.get_order(pos['id']) if TRADE_EXECUTION else None
                current_price = client.get_latest_trade(ticker).price
                gain = (
                    (current_price - pos['entry']) / pos['entry'] * 100
                    if pos['side'] == 'long'
                    else (pos['entry'] - current_price) / pos['entry'] * 100
                )
                if gain <= -1 or gain >= pos['tp']:
                    close_side = 'sell' if pos['side'] == 'long' else 'buy'
                    if TRADE_EXECUTION:
                        client.submit_order(
                            symbol=ticker,
                            qty=pos['qty'],
                            side=close_side,
                            type='market',
                            time_in_force='gtc'
                        )
                    send_telegram(
                        f"📤 Exited {pos['side'].upper()} {pos['qty']} {ticker} at gain/loss: {gain:.2f}%"
                    )
                    positions.remove(pos)
            except Exception as e:
                send_telegram(f"⚠️ Error checking {ticker} position: {e}")

schedule.every(30).seconds.do(check_smc)
schedule.every(30).seconds.do(check_positions)
schedule.every(30).seconds.do(scan_for_sweep_momentum_trades)
schedule.every(60).seconds.do(heartbeat)

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

