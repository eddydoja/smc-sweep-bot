import os
import itertools
import time
import requests
import schedule
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame
from twelvedata import TDClient

load_dotenv()

# Config & globals
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Scout strategy config
SCOUT_TRADES_ENABLED = True
SCOUT_TRADE_COOLDOWN_MINUTES = 10
SCOUT_MIN_CONFIDENCE = 5
last_scout_signal_time = {}

# Trade behavior toggles
TRADE_EXECUTION = True
HEARTBEAT_TELEGRAM = False

# Tickers mapping
TICKER_MAP = {"DAX": "EWG", "UKX": "EWU"}
TICKERS = [
    "SPY", "QQQ", "DIA", "DAX", "UKX",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"
]

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")
open_positions = {}
last_signal_time = {}

SESSION_FILTER = {'FX': (3, 17), 'US': (13, 20)}
twelve = TDClient(apikey=TWELVE_DATA_API_KEY)
TWELVE_FX_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD"
}
# Iterator for rotating FX pulls (reduce credits by ~80%)
fx_cycle = itertools.cycle(TWELVE_FX_SYMBOLS.keys())

# --- NEW: Pacific session guard (6:00–16:00 America/Los_Angeles) ---
PST_TZ = pytz.timezone("America/Los_Angeles")

def in_pst_trading_window(now: datetime | None = None) -> bool:
    """
    True only from 06:00 to 16:00 America/Los_Angeles, inclusive of 06:00, exclusive of 16:00.
    Applies every day; does not alter or remove any of your other session checks.
    """
    now = now.astimezone(PST_TZ) if now else datetime.now(PST_TZ)
    start = now.replace(hour=6, minute=0, second=0, microsecond=0)
    end   = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now < end
# --------------------------------------------------------------------

# Utility functions

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        resp = requests.post(url, json=payload)
        print(f"Sent Telegram message: {message}", flush=True) if resp.ok else print(f"[Telegram Error] {resp.status_code}", flush=True)
    except Exception as e:
        print(f"[Telegram Exception] {e}", flush=True)

def is_market_open_now():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    return now.weekday() < 5 and ((now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16))

def is_session_active(ticker):
    hour = datetime.now(pytz.UTC).hour
    return SESSION_FILTER['FX'][0] <= hour < SESSION_FILTER['FX'][1] if ticker in TWELVE_FX_SYMBOLS else SESSION_FILTER['US'][0] <= hour < SESSION_FILTER['US'][1]

def resolve_ticker(ticker):
    return TICKER_MAP.get(ticker, ticker)

def get_fx_close(ticker):
    symbol = TWELVE_FX_SYMBOLS.get(ticker)
    if not symbol:
        return None
    try:
        ts = twelve.time_series(symbol=symbol, interval="1min", outputsize=5).as_pandas()
        if ts.empty or len(ts) < 5:
            return None
        return float(ts['close'].iloc[-1]), float(ts['close'].iloc[0])
    except Exception as e:
        print(f"{ticker} FX fetch error: {e}", flush=True)
        return None
        
def pull_fx_prices_rotating():
    if not in_pst_trading_window():
        return
    ticker = next(fx_cycle)
    res = get_fx_close(ticker)
    if res:
        latest_close, close_5m_ago = res
        print(f"[FX Pull] {ticker}: {latest_close:.5f}", flush=True)
    else:
        print(f"{ticker}: data unavailable", flush=True)

def get_data(ticker, timeframe=TimeFrame.Minute, limit=100):
    try:
        bars = client.get_bars(resolve_ticker(ticker), timeframe, limit=min(limit, 1000)).df
        if bars.empty or len(bars) < 10:
            return pd.DataFrame()
        bars['body'] = (bars['close'] - bars['open']).abs()
        bars['range'] = bars['high'] - bars['low']
        bars['upper_wick'] = bars['high'] - bars[["open", "close"]].max(axis=1)
        bars['lower_wick'] = bars[["open", "close"]].min(axis=1) - bars['low']
        bars['is_doji'] = (bars['body'] / bars['range']).fillna(0) < 0.1
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
        atr = bars['range'].rolling(14).mean()
        bars['bull_momo'] = (bars['close'] > bars['open']) & (bars['body'] > atr * 0.8) & (bars['close'] > bars['high'].shift(1))
        bars['bear_momo'] = (bars['close'] < bars['open']) & (bars['body'] > atr * 0.8) & (bars['close'] < bars['low'].shift(1))
        return bars
    except Exception as e:
        print(f"⚠️ Data fetch failed for {ticker}: {e}", flush=True)
        return pd.DataFrame()

def detect_structure(df):
    df = df.copy()
    df['is_hh'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    df['is_ll'] = (df['low']  < df['low'].shift(1))  & (df['low']  < df['low'].shift(-1))
    swing_highs, swing_lows = df[df['is_hh']], df[df['is_ll']]
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # current vs previous swings (scalars)
    prev_sh = swing_highs.iloc[-2]
    curr_sh = swing_highs.iloc[-1]
    prev_sl = swing_lows.iloc[-2]
    curr_sl = swing_lows.iloc[-1]

    trend = 'neutral'; bos = False
    if (curr_sh['high'] > prev_sh['high']) and (curr_sl['low'] > prev_sl['low']):
        trend, bos = 'bullish', True
    elif (curr_sh['high'] < prev_sh['high']) and (curr_sl['low'] < prev_sl['low']):
        trend, bos = 'bearish', True

    return {
        'trend': trend,
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'bos': bos
    }

def detect_fvg(df):
    for i in range(2, len(df)):
        if df.iloc[i-2]['low'] > df.iloc[i]['high']:
            return {'start': df.iloc[i-2]['low'], 'end': df.iloc[i]['high'], 'index': i}
        if df.iloc[i]['low'] > df.iloc[i-2]['high']:
            return {'start': df.iloc[i]['low'], 'end': df.iloc[i-2]['high'], 'index': i}
    return None

def detect_order_block(df):
    for i in range(len(df)-3, -1, -1):
        if df.iloc[i]['close'] < df.iloc[i]['open'] and df.iloc[i+1]['close'] > df.iloc[i+1]['open']:
            return {'open': df.iloc[i]['open'], 'close': df.iloc[i]['close']}
        if df.iloc[i]['close'] > df.iloc[i]['open'] and df.iloc[i+1]['close'] < df.iloc[i+1]['open']:
            return {'open': df.iloc[i]['open'], 'close': df.iloc[i]['close']}
    return None

def calculate_confidence(df, last_candle):
    score = sum((
        2 if last_candle.get('bullish_engulfing') or last_candle.get('bearish_engulfing') else 0,
        1 if last_candle.get('hammer') or last_candle.get('inverted_hammer') else 0,
        2 if last_candle.get('3bar_bullish') or last_candle.get('3bar_bearish') else 0,
        1 if last_candle.get('is_doji') else 0,
        1 if last_candle.get('wick_rejection_up') or last_candle.get('wick_rejection_down') else 0,
        3 if last_candle.get('liquidity_sweep_high') or last_candle.get('liquidity_sweep_low') else 0,
        2 if last_candle.get('bull_momo') or last_candle.get('bear_momo') else 0
    ))
    return max(1, min(10, score))

def determine_qty(confidence):
    return min(10, max(1, confidence))

def calculate_confidence_tp(df, side, entry_price):
    avg_range = df['range'][-20:].mean()
    mult = 2.2  # default R multiple
    return round(entry_price + avg_range * mult, 2) if side == 'long' else round(entry_price - avg_range * mult, 2)

# SMC strategy
def check_smc():
    # --- NEW: limit signal searching to 6:00–16:00 PST ---
    if not in_pst_trading_window():
        return
    # -----------------------------------------------------
    if not is_market_open_now():
        return
    total = sum(len(v) for v in open_positions.values())
    if total >= 3:
        return
    for ticker in TICKERS:
        if not is_session_active(ticker):
            continue
        open_positions.setdefault(ticker, [])
        if open_positions[ticker]:
            continue
        last = last_signal_time.get(ticker)
        if last and datetime.now(pytz.UTC) - last < timedelta(seconds=30):
            continue
        df = get_data(ticker)
        if df.shape[0] < 50:
            continue
        ltf = detect_structure(df)
        if not ltf or not ltf['bos'] or ltf['trend']=='neutral':
            continue
        h1 = detect_structure(get_data(ticker, TimeFrame.Hour, limit=50))
        h4 = detect_structure(get_data(ticker, TimeFrame.Hour, limit=23))
        if not (h1 and h4 and (ltf['trend']==h1['trend'] or ltf['trend']==h4['trend'])):
            continue
        fvg, ob = detect_fvg(df), detect_order_block(df)
        if not fvg or not ob:
            continue
        last_c = df.iloc[-1]
        price = float(last_c['close'])
        in_bull = ltf['trend']=='bullish' and price < fvg['start'] and price >= ob['close']
        in_bear = ltf['trend']=='bearish' and price > fvg['start'] and price <= ob['close']
        cond_bull = in_bull and any(last_c[k] for k in ['bullish_engulfing','hammer','is_doji','wick_rejection_down','3bar_bullish','liquidity_sweep_low','bull_momo'])
        cond_bear = in_bear and any(last_c[k] for k in ['bearish_engulfing','inverted_hammer','is_doji','wick_rejection_up','3bar_bearish','liquidity_sweep_high','bear_momo'])
        side = 'long' if cond_bull else 'short' if cond_bear else None
        if not side:
            continue
        confidence = calculate_confidence(df, last_c)
        sl = round(price * 0.99,2) if side=='long' else round(price * 1.01,2)
        tp = calculate_confidence_tp(df, side, price)
        tp_pct = abs((tp-price)/price*100)
        qty = determine_qty(confidence)
        try:
            if TRADE_EXECUTION:
                order = client.submit_order(symbol=ticker, qty=qty, side=('buy' if side=='long' else 'sell'), type='market', time_in_force='gtc',
                                           order_class='bracket', stop_loss={"stop_price":str(sl)}, take_profit={"limit_price":str(tp)})
                order_id = order.id
            else:
                order_id = f"signal-{datetime.now(pytz.UTC).isoformat()}"
            open_positions[ticker].append({'id':order_id,'side':side,'entry':price,'tp':tp_pct,'qty':qty,'timestamp':datetime.now(pytz.UTC)})
            last_signal_time[ticker] = datetime.now(pytz.UTC)
            send_telegram(f"📥 {ticker} SIGNAL: {side.upper()} {qty} @ {price}\nTP: {tp} ({tp_pct:.1f}%) | SL: {sl} (1%) | Confidence: {confidence}/10")
        except Exception as e:
            send_telegram(f"❌ Order error on {ticker}: {e}")

# Scout strategy
def scan_for_sweep_momentum_trades():
    # --- NEW: limit scout signal searching to 6:00–16:00 PST ---
    if not in_pst_trading_window():
        return
    # -----------------------------------------------------------
    if not SCOUT_TRADES_ENABLED:
        return
    total = sum(len(v) for v in open_positions.values())
    if total >= 10:
        return
    for ticker in TICKERS:
        if not is_session_active(ticker):
            continue
        now = datetime.now(pytz.UTC)
        last = last_scout_signal_time.get(ticker)
        if last and now - last < timedelta(minutes=SCOUT_TRADE_COOLDOWN_MINUTES):
            continue
        df = get_data(ticker, limit=10)
        if df.shape[0] < 6:
            continue
        last6 = df.tail(6)
        sweep_low = last6.iloc[-2]['low'] < last6['low'].iloc[:-2].min()
        sweep_high = last6.iloc[-2]['high'] > last6['high'].iloc[:-2].max()
        c = last6.iloc[-1]
        body = abs(c['close']-c['open']); range_ = c['high']-c['low']
        if range_ <= 0: continue
        is_momentum = body / range_ > 0.6
        side = 'long' if sweep_low and c['close'] > c['open'] and is_momentum else 'short' if sweep_high and c['close'] < c['open'] and is_momentum else None
        if not side:
            continue
        confidence = calculate_confidence(df, c)
        if confidence < SCOUT_MIN_CONFIDENCE:
            continue
        entry = float(c['close'])
        sl = round(entry * 0.995,4) if side=='long' else round(entry * 1.005,4)
        tp = calculate_confidence_tp(df, side, entry)
        tp_pct = abs((tp-entry)/entry*100)
        qty = determine_qty(confidence)
        try:
            if TRADE_EXECUTION:
                order = client.submit_order(symbol=ticker, qty=qty, side=('buy' if side=='long' else 'sell'), type='market', time_in_force='gtc',
                                           order_class='bracket', stop_loss={"stop_price":str(sl)}, take_profit={"limit_price":str(tp)})
                order_id = order.id
            else:
                order_id = f"scout-{datetime.now(pytz.UTC).isoformat()}"
            open_positions.setdefault(ticker, []).append({'id':order_id, 'side':side, 'entry':entry, 'tp':tp_pct, 'qty':qty, 'timestamp':now})
            last_scout_signal_time[ticker] = now
            send_telegram(f"⚡ SCOUT {ticker}: {side.upper()} {qty} @ {entry}\nTP: {tp} ({tp_pct:.1f}%) | SL: {sl} | Confidence: {confidence}/10")
        except Exception as e:
            send_telegram(f"❌ Scout order error on {ticker}: {e}")

# Position management
def check_positions():
    for ticker, positions in open_positions.items():
        for pos in positions[:]:
            try:
                current_price = client.get_latest_trade(ticker).price
                gain = (current_price - pos['entry'])/pos['entry']*100 if pos['side']=='long' else (pos['entry'] - current_price)/pos['entry']*100
                if gain <= -1 or gain >= pos['tp']:
                    close = 'sell' if pos['side']=='long' else 'buy'
                    if TRADE_EXECUTION:
                        client.submit_order(symbol=ticker, qty=pos['qty'], side=close, type='market', time_in_force='gtc')
                    send_telegram(f"📤 Exited {pos['side'].upper()} {pos['qty']} {ticker} at gain/loss: {gain:.2f}%")
                    positions.remove(pos)
            except Exception as e:
                send_telegram(f"⚠️ Error checking {ticker} position: {e}")

# Heartbeat logging
def print_recent_price_action(ticker):
    try:
        if ticker in TWELVE_FX_SYMBOLS:
            res = get_fx_close(ticker)
            if res is None:
                print(f"{ticker}: Insufficient data", flush=True)
                return
            latest_close, close_5m_ago = res
        else:
            latest_trade = client.get_latest_trade(resolve_ticker(ticker))
            latest_close = latest_trade.price

            bars = client.get_bars(resolve_ticker(ticker), TimeFrame.Minute, limit=5).df
            if bars.empty or len(bars) < 5:
                print(f"{ticker}: Insufficient data", flush=True)
                return
            close_5m_ago = bars.iloc[0]['close']

        delta = latest_close - close_5m_ago
        pct_change = (delta / close_5m_ago) * 100

        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        price_fmt = f"{latest_close:.4f}" if latest_close < 10 else f"{latest_close:.2f}"
        delta_fmt = f"{delta:+.4f}" if abs(delta) < 1 else f"{delta:+.2f}"
        pct_fmt = f"{pct_change:+.2f}%"

        print(f"{ticker}: {price_fmt} {arrow} ({delta_fmt} / {pct_fmt} over 5m)", flush=True)

    except Exception as e:
        print(f"{ticker} price action error: {e}", flush=True)

def heartbeat():
    if not in_pst_trading_window():
        return
    total = sum(len(v) for v in open_positions.values())
    msg = f"⏱️ Heartbeat {datetime.now(pytz.UTC).strftime('%H:%M:%S')} UTC | open trades: {total} | tickers active: {len(TICKERS)}"
    print(msg, flush=True)

    # Only non-FX here to avoid extra FX pulls
    for t in TICKERS:
        if t in TWELVE_FX_SYMBOLS:
            continue
        print_recent_price_action(t)

    if HEARTBEAT_TELEGRAM:
        send_telegram(msg)

# Scheduler
schedule.every(30).seconds.do(check_smc)
schedule.every(30).seconds.do(check_positions)
schedule.every(30).seconds.do(scan_for_sweep_momentum_trades)
schedule.every(60).seconds.do(heartbeat)
schedule.every(45).seconds.do(pull_fx_prices_rotating)


# Launch
try:
    send_telegram("✅ Sniper Signal Bot started ✅")
    print("Bot is running...", flush=True)
except Exception as e:
    print(f"[Startup Error] {e}", flush=True)

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f"[Loop Error] {e}", flush=True)
        time.sleep(5)


