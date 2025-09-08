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

# =========================
# Config & globals
# =========================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Trade behavior toggles
TRADE_EXECUTION = True           # Equities/ETFs via Alpaca; FX is Telegram-only
HEARTBEAT_TELEGRAM = False

# Tickers mapping for ETFs representing indexes
TICKER_MAP = {"DAX": "EWG", "UKX": "EWU"}
TICKERS = [
    "SPY", "QQQ", "DIA", "DAX", "UKX",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"
]

client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")

# UTC hour filters for session gating
SESSION_FILTER = {'FX': (3, 23), 'US': (13, 20)}  # lets FX run 6:00–4:00 PT

twelve = TDClient(apikey=TWELVE_DATA_API_KEY)
TWELVE_FX_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD"
}

# Rotate lightweight FX pulls to save credits
fx_cycle = itertools.cycle(TWELVE_FX_SYMBOLS.keys())

# --- Pacific session guard (6:00–16:00 America/Los_Angeles) ---
PST_TZ = pytz.timezone("America/Los_Angeles")
def in_pst_trading_window(now: datetime | None = None) -> bool:
    now = now.astimezone(PST_TZ) if now else datetime.now(PST_TZ)
    start = now.replace(hour=6, minute=0, second=0, microsecond=0)
    end   = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now < end
# ----------------------------------------------------------------

# =========================
# Utilities
# =========================
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        resp = requests.post(url, json=payload)
        if resp.ok:
            print(f"Sent Telegram message: {message}", flush=True)
        else:
            print(f"[Telegram Error] {resp.status_code} {resp.text}", flush=True)
    except Exception as e:
        print(f"[Telegram Exception] {e}", flush=True)

def is_market_open_now():
    # US equities regular trading hours only
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    return now.weekday() < 5 and ((now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16))

def is_session_active(ticker: str) -> bool:
    hour = datetime.now(pytz.UTC).hour
    if ticker in TWELVE_FX_SYMBOLS:
        lo, hi = SESSION_FILTER['FX']
    else:
        lo, hi = SESSION_FILTER['US']
    return lo <= hour < hi

def resolve_ticker(ticker: str) -> str:
    return TICKER_MAP.get(ticker, ticker)

def get_fx_close(ticker: str):
    """Quick latest vs 5m-ago close for FX (for heartbeat/position checks)."""
    symbol = TWELVE_FX_SYMBOLS.get(ticker)
    if not symbol:
        return None
    try:
        ts = twelve.time_series(symbol=symbol, interval="1min", outputsize=5).as_pandas()
        if ts is None or ts.empty or len(ts) < 5:
            return None
        latest_close = float(ts['close'].iloc[0])   # newest-first
        close_5m_ago = float(ts['close'].iloc[-1])
        return latest_close, close_5m_ago
    except Exception as e:
        print(f"{ticker} FX fetch error: {e}", flush=True)
        return None

def pull_fx_prices_rotating():
    if not in_pst_trading_window():
        return
    ticker = next(fx_cycle)
    res = get_fx_close(ticker)
    if res:
        latest_close, _ = res
        print(f"[FX Pull] {ticker}: {latest_close:.5f}", flush=True)
    else:
        print(f"{ticker}: data unavailable", flush=True)

# =========================
# Data fetch + feature engineering (shared by all strategies)
# =========================
def get_data(ticker, timeframe=TimeFrame.Minute, limit=100):
    """
    Unified candle fetcher:
      - Equities/ETFs → Alpaca bars
      - FX → Twelve Data 1m series
    Returns a DataFrame with TJR-style engineered columns ready for signal logic.
    """
    try:
        if ticker in TWELVE_FX_SYMBOLS:
            symbol = TWELVE_FX_SYMBOLS[ticker]
            ts = twelve.time_series(symbol=symbol, interval="1min", outputsize=min(limit + 5, 500)).as_pandas()
            if ts is None or ts.empty or len(ts) < 10:
                return pd.DataFrame()
            df = ts.iloc[::-1].copy()  # ascending time
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        else:
            bars = client.get_bars(resolve_ticker(ticker), timeframe, limit=min(limit, 1000)).df
            if bars.empty or len(bars) < 10:
                return pd.DataFrame()
            df = bars.copy()

        # --- Feature engineering (TJR set) ---
        df['body'] = (df['close'] - df['open']).abs()
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[["open", "close"]].max(axis=1)
        df['lower_wick'] = df[["open", "close"]].min(axis=1) - df['low']
        df['is_doji'] = (df['body'] / df['range']).fillna(0) < 0.1

        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        )
        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        )
        df['hammer'] = (df['body'] / df['range'] < 0.3) & (df['lower_wick'] > df['body'])
        df['inverted_hammer'] = (df['body'] / df['range'] < 0.3) & (df['upper_wick'] > df['body'])
        df['wick_rejection_up'] = df['upper_wick'] > (df['range'] * 0.6)
        df['wick_rejection_down'] = df['lower_wick'] > (df['range'] * 0.6)

        df['3bar_bullish'] = (
            (df['close'].shift(2) < df['open'].shift(2)) &
            (df['is_doji'].shift(1)) &
            (df['close'] > df['open'])
        )
        df['3bar_bearish'] = (
            (df['close'].shift(2) > df['open'].shift(2)) &
            (df['is_doji'].shift(1)) &
            (df['close'] < df['open'])
        )

        df['liquidity_sweep_high'] = df['high'] > df['high'].rolling(window=20).max().shift(1)
        df['liquidity_sweep_low'] = df['low'] < df['low'].rolling(window=20).min().shift(1)

        atr = df['range'].rolling(14).mean()
        df['bull_momo'] = (df['close'] > df['open']) & (df['body'] > atr * 0.8) & (df['close'] > df['high'].shift(1))
        df['bear_momo'] = (df['close'] < df['open']) & (df['body'] > atr * 0.8) & (df['close'] < df['low'].shift(1))
        return df

    except Exception as e:
        print(f"⚠️ Data fetch failed for {ticker}: {e}", flush=True)
        return pd.DataFrame()

# =========================
# TJR components (structure/FVG/OB/confidence/TP)
# =========================
def detect_structure(df):
    df = df.copy()
    df['is_hh'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    df['is_ll'] = (df['low']  < df['low'].shift(1))  & (df['low']  < df['low'].shift(-1))
    swing_highs, swing_lows = df[df['is_hh']], df[df['is_ll']]
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    prev_sh, curr_sh = swing_highs.iloc[-2], swing_highs.iloc[-1]
    prev_sl, curr_sl = swing_lows.iloc[-2], swing_lows.iloc[-1]
    trend, bos = 'neutral', False
    if (curr_sh['high'] > prev_sh['high']) and (curr_sl['low'] > prev_sl['low']):
        trend, bos = 'bullish', True
    elif (curr_sh['high'] < prev_sh['high']) and (curr_sl['low'] < prev_sl['low']):
        trend, bos = 'bearish', True
    return {'trend': trend, 'swing_highs': swing_highs, 'swing_lows': swing_lows, 'bos': bos}

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
    precision = 5 if entry_price < 10 else 2
    return round(entry_price + avg_range * mult, precision) if side == 'long' else round(entry_price - avg_range * mult, precision)

# Latest price helper (equities via Alpaca; FX via Twelve Data)
def latest_price(ticker):
    if ticker in TWELVE_FX_SYMBOLS:
        res = get_fx_close(ticker)
        return res[0] if res else None
    else:
        return client.get_latest_trade(resolve_ticker(ticker)).price

# =========================
# Strategy Separation
# =========================
# Strategy labels
STRAT_TJR = "TJR"
STRAT_SCALP = "SCALP"

# Namespaced state
OPEN_POSITIONS = {
    STRAT_TJR: {},    # dict[ticker] -> list[positions]
    STRAT_SCALP: {},
}
LAST_SIGNAL_TIME = {
    STRAT_TJR: {},
    STRAT_SCALP: {},
}
LAST_SCOUT_SIGNAL_TIME = {       # TJR-only "scout" cooldowns
    STRAT_TJR: {}
}

# Optional cross-strategy symbol locks (prevents both entering same ticker simultaneously)
SYMBOL_LOCKS = {
    STRAT_TJR: set(),
    STRAT_SCALP: set(),
}

# Per-strategy risk caps
MAX_CONCURRENT = {
    STRAT_TJR: 3,
    STRAT_SCALP: 5,
}

# Helper funcs for namespaced state
def positions_count(strategy: str) -> int:
    return sum(len(v) for v in OPEN_POSITIONS[strategy].values())

def add_position(strategy: str, ticker: str, pos: dict):
    OPEN_POSITIONS[strategy].setdefault(ticker, []).append(pos)

def can_trade_symbol(strategy: str, ticker: str) -> bool:
    if OPEN_POSITIONS[strategy].get(ticker):  # already in that strategy
        return False
    # prevent cross-strategy simultaneous entries (optional)
    if ticker in SYMBOL_LOCKS[STRAT_TJR] or ticker in SYMBOL_LOCKS[STRAT_SCALP]:
        return False
    return True

def lock_symbol(strategy: str, ticker: str):
    SYMBOL_LOCKS[strategy].add(ticker)

def unlock_symbol(strategy: str, ticker: str):
    SYMBOL_LOCKS[strategy].discard(ticker)

def mark_signal_time(strategy: str, ticker: str):
    LAST_SIGNAL_TIME[strategy][ticker] = datetime.now(pytz.UTC)

def last_signal_within(strategy: str, ticker: str, delta: timedelta) -> bool:
    last = LAST_SIGNAL_TIME[strategy].get(ticker)
    return bool(last and datetime.now(pytz.UTC) - last < delta)

def last_scout_within(strategy: str, ticker: str, delta: timedelta) -> bool:
    last = LAST_SCOUT_SIGNAL_TIME[strategy].get(ticker)
    return bool(last and datetime.now(pytz.UTC) - last < delta)

def mark_scout_time(strategy: str, ticker: str):
    LAST_SCOUT_SIGNAL_TIME[strategy][ticker] = datetime.now(pytz.UTC)

def alpaca_place_order(symbol, side, qty, sl, tp, strategy):
    coid = f"{strategy}-{datetime.now(pytz.UTC).strftime('%Y%m%d-%H%M%S')}"
    order = client.submit_order(
        symbol=symbol, qty=qty, side=side,
        type='market', time_in_force='gtc',
        order_class='bracket',
        stop_loss={"stop_price": str(sl)},
        take_profit={"limit_price": str(tp)},
        client_order_id=coid
    )
    return order.id

# =========================
# TJR Strategy (SMC) — signals + optional scout
# =========================
SCOUT_TRADES_ENABLED = True
SCOUT_TRADE_COOLDOWN_MINUTES = 10
SCOUT_MIN_CONFIDENCE = 5

def check_smc():
    strategy = STRAT_TJR
    if not in_pst_trading_window():
        return
    if positions_count(strategy) >= MAX_CONCURRENT[strategy]:
        return

    for ticker in TICKERS:
        if not is_session_active(ticker):
            continue
        # Equities require US RTH; FX is allowed (within PST guard + session)
        if ticker not in TWELVE_FX_SYMBOLS and not is_market_open_now():
            continue
        if not can_trade_symbol(strategy, ticker):
            continue
        if last_signal_within(strategy, ticker, timedelta(seconds=30)):
            continue

        df = get_data(ticker)
        if df.shape[0] < 50:
            continue

        ltf = detect_structure(df)
        if not ltf or not ltf['bos'] or ltf['trend'] == 'neutral':
            continue

        # Higher-timeframe alignment (hourly + ~H4 via 23 hours)
        h1 = detect_structure(get_data(ticker, TimeFrame.Hour, limit=50))
        h4 = detect_structure(get_data(ticker, TimeFrame.Hour, limit=23))
        if not (h1 and h4 and (ltf['trend'] == h1['trend'] or ltf['trend'] == h4['trend'])):
            continue

        fvg, ob = detect_fvg(df), detect_order_block(df)
        if not fvg or not ob:
            continue

        last_c = df.iloc[-1]
        price = float(last_c['close'])

        in_bull = ltf['trend'] == 'bullish' and price < fvg['start'] and price >= ob['close']
        in_bear = ltf['trend'] == 'bearish' and price > fvg['start'] and price <= ob['close']

        cond_bull = in_bull and any(last_c[k] for k in [
            'bullish_engulfing', 'hammer', 'is_doji', 'wick_rejection_down', '3bar_bullish', 'liquidity_sweep_low', 'bull_momo'
        ])
        cond_bear = in_bear and any(last_c[k] for k in [
            'bearish_engulfing', 'inverted_hammer', 'is_doji', 'wick_rejection_up', '3bar_bearish', 'liquidity_sweep_high', 'bear_momo'
        ])
        side = 'long' if cond_bull else 'short' if cond_bear else None
        if not side:
            continue

        confidence = calculate_confidence(df, last_c)
        sl = round(price * 0.99, 5 if price < 10 else 2) if side == 'long' else round(price * 1.01, 5 if price < 10 else 2)
        tp = calculate_confidence_tp(df, side, price)
        tp_pct = abs((tp - price) / price * 100)
        qty = determine_qty(confidence)

        is_fx = ticker in TWELVE_FX_SYMBOLS
        try:
            if TRADE_EXECUTION and not is_fx:
                order_id = alpaca_place_order(
                    symbol=ticker,
                    side=('buy' if side == 'long' else 'sell'),
                    qty=qty, sl=sl, tp=tp, strategy=strategy
                )
            else:
                order_id = f"sig-{strategy}-{'fx' if is_fx else 'eq'}-{datetime.now(pytz.UTC).isoformat()}"

            add_position(strategy, ticker, {
                'id': order_id, 'strategy': strategy, 'side': side, 'entry': price,
                'tp': tp_pct, 'qty': qty, 'timestamp': datetime.now(pytz.UTC)
            })
            mark_signal_time(strategy, ticker)
            lock_symbol(strategy, ticker)

            label = "FX SIGNAL" if is_fx else "SIGNAL"
            send_telegram(
                f"📥 {label} [{strategy}] {ticker}: {side.upper()} {qty} @ {price}\n"
                f"TP: {tp} ({tp_pct:.1f}%) | SL: {sl} | Confidence: {confidence}/10"
            )
        except Exception as e:
            send_telegram(f"❌ Order error [{strategy}] on {ticker}: {e}")

def scan_for_sweep_momentum_trades():
    """TJR 'scout' side-car (unchanged logic, now namespaced and FX-aware)."""
    strategy = STRAT_TJR
    if not in_pst_trading_window():
        return
    if not SCOUT_TRADES_ENABLED:
        return
    if positions_count(strategy) >= 10:
        return

    for ticker in TICKERS:
        if not is_session_active(ticker):
            continue
        # Equities require US RTH; FX allowed
        if ticker not in TWELVE_FX_SYMBOLS and not is_market_open_now():
            continue
        if last_scout_within(strategy, ticker, timedelta(minutes=SCOUT_TRADE_COOLDOWN_MINUTES)):
            continue
        if not can_trade_symbol(strategy, ticker):
            continue

        df = get_data(ticker, limit=10)
        if df.shape[0] < 6:
            continue

        last6 = df.tail(6)
        sweep_low = last6.iloc[-2]['low'] < last6['low'].iloc[:-2].min()
        sweep_high = last6.iloc[-2]['high'] > last6['high'].iloc[:-2].max()
        c = last6.iloc[-1]
        body = abs(c['close'] - c['open']); range_ = c['high'] - c['low']
        if range_ <= 0:
            continue
        is_momentum = body / range_ > 0.6

        side = 'long' if (sweep_low and c['close'] > c['open'] and is_momentum) else \
               'short' if (sweep_high and c['close'] < c['open'] and is_momentum) else None
        if not side:
            continue

        confidence = calculate_confidence(df, c)
        if confidence < SCOUT_MIN_CONFIDENCE:
            continue

        entry = float(c['close'])
        sl = round(entry * 0.995, 5 if entry < 10 else 2) if side == 'long' else round(entry * 1.005, 5 if entry < 10 else 2)
        tp = calculate_confidence_tp(df, side, entry)
        tp_pct = abs((tp - entry) / entry * 100)
        qty = determine_qty(confidence)

        is_fx = ticker in TWELVE_FX_SYMBOLS
        try:
            if TRADE_EXECUTION and not is_fx:
                order_id = alpaca_place_order(
                    symbol=ticker,
                    side=('buy' if side == 'long' else 'sell'),
                    qty=qty, sl=sl, tp=tp, strategy=strategy
                )
            else:
                order_id = f"scout-{strategy}-{'fx' if is_fx else 'eq'}-{datetime.now(pytz.UTC).isoformat()}"

            add_position(strategy, ticker, {
                'id': order_id, 'strategy': strategy, 'side': side,
                'entry': entry, 'tp': tp_pct, 'qty': qty, 'timestamp': datetime.now(pytz.UTC)
            })
            mark_scout_time(strategy, ticker)
            lock_symbol(strategy, ticker)

            label = "SCOUT FX" if is_fx else "SCOUT"
            send_telegram(
                f"⚡ {label} [{strategy}] {ticker}: {side.upper()} {qty} @ {entry}\n"
                f"TP: {tp} ({tp_pct:.1f}%) | SL: {sl} | Confidence: {confidence}/10"
            )
        except Exception as e:
            send_telegram(f"❌ Scout order error [{strategy}] on {ticker}: {e}")

# =========================
# Independent Scalper Strategy
# =========================
SCALPER_COOLDOWN_MINUTES = 5
SCALPER_MIN_MOMO_BODY_RATIO = 0.7
SCALPER_ATR_MULT_TP = 1.2
SCALPER_SL_BPS = 60  # 0.60%

def check_scalper():
    strategy = STRAT_SCALP
    if not in_pst_trading_window():
        return
    if positions_count(strategy) >= MAX_CONCURRENT[strategy]:
        return

    for ticker in TICKERS:
        if not is_session_active(ticker):
            continue
        if ticker not in TWELVE_FX_SYMBOLS and not is_market_open_now():
            continue
        if last_signal_within(strategy, ticker, timedelta(minutes=SCALPER_COOLDOWN_MINUTES)):
            continue
        if not can_trade_symbol(strategy, ticker):
            continue

        df = get_data(ticker, limit=20)
        if df.shape[0] < 6:
            continue

        c1 = df.iloc[-2]; c2 = df.iloc[-1]

        def is_momo(c):
            rng = c['range']
            if rng <= 0:
                return False
            return (c['body'] / rng) >= SCALPER_MIN_MOMO_BODY_RATIO

        sweep_low = c1['low'] < df['low'][:-1].min()
        sweep_high = c1['high'] > df['high'][:-1].max()

        long_ok  = sweep_low and (c2['close'] > c2['open']) and is_momo(c2)
        short_ok = sweep_high and (c2['close'] < c2['open']) and is_momo(c2)
        if not (long_ok or short_ok):
            continue

        side = 'long' if long_ok else 'short'
        entry = float(c2['close'])

        if side == 'long':
            sl = round(entry * (1 - SCALPER_SL_BPS / 10000.0), 5 if entry < 10 else 2)
        else:
            sl = round(entry * (1 + SCALPER_SL_BPS / 10000.0), 5 if entry < 10 else 2)

        atr = df['range'].rolling(14).mean().iloc[-1]
        tp = entry + SCALPER_ATR_MULT_TP * atr if side == 'long' else entry - SCALPER_ATR_MULT_TP * atr
        tp = round(tp, 5 if entry < 10 else 2)
        tp_pct = abs((tp - entry) / entry * 100)

        confidence = calculate_confidence(df, c2)
        qty = determine_qty(confidence)

        is_fx = ticker in TWELVE_FX_SYMBOLS
        try:
            if TRADE_EXECUTION and not is_fx:
                order_id = alpaca_place_order(
                    symbol=ticker,
                    side=('buy' if side == 'long' else 'sell'),
                    qty=qty, sl=sl, tp=tp, strategy=strategy
                )
            else:
                order_id = f"sig-{strategy}-{'fx' if is_fx else 'eq'}-{datetime.now(pytz.UTC).isoformat()}"

            add_position(strategy, ticker, {
                'id': order_id, 'strategy': strategy, 'side': side,
                'entry': entry, 'tp': tp_pct, 'qty': qty, 'timestamp': datetime.now(pytz.UTC)
            })
            mark_signal_time(strategy, ticker)
            lock_symbol(strategy, ticker)

            label = "FX SCALP" if is_fx else "SCALP"
            send_telegram(
                f"⚡ {label} [{strategy}] {ticker}: {side.upper()} {qty} @ {entry}\n"
                f"TP: {tp} ({tp_pct:.1f}%) | SL: {sl} | Conf: {confidence}/10"
            )
        except Exception as e:
            send_telegram(f"❌ {strategy} order error on {ticker}: {e}")

# =========================
# Position management (strategy-aware)
# =========================
def check_positions():
    for strategy, book in OPEN_POSITIONS.items():
        for ticker, positions in list(book.items()):
            for pos in positions[:]:
                try:
                    current = latest_price(ticker)
                    if current is None:
                        continue

                    if pos['side'] == 'long':
                        gain = (current - pos['entry']) / pos['entry'] * 100
                        close_side = 'sell'
                    else:
                        gain = (pos['entry'] - current) / pos['entry'] * 100
                        close_side = 'buy'

                    if gain <= -1 or gain >= pos['tp']:
                        # Alpaca exit only for equities/ETFs
                        if TRADE_EXECUTION and ticker not in TWELVE_FX_SYMBOLS:
                            client.submit_order(symbol=ticker, qty=pos['qty'], side=close_side, type='market', time_in_force='gtc')

                        send_telegram(f"📤 Exit [{strategy}] {pos['side'].upper()} {pos['qty']} {ticker} PnL: {gain:.2f}%")
                        positions.remove(pos)
                        unlock_symbol(strategy, ticker)

                except Exception as e:
                    send_telegram(f"⚠️ Error checking [{strategy}] {ticker} position: {e}")

# =========================
# Heartbeat logging
# =========================
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
            latest_close = float(latest_trade.price)
            bars = client.get_bars(resolve_ticker(ticker), TimeFrame.Minute, limit=5).df
            if bars.empty or len(bars) < 5:
                print(f"{ticker}: Insufficient data", flush=True)
                return
            close_5m_ago = float(bars.iloc[0]['close'])

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
    total_tjr = positions_count(STRAT_TJR)
    total_scalp = positions_count(STRAT_SCALP)
    total = total_tjr + total_scalp

    msg = (
        f"⏱️ Heartbeat {datetime.now(pytz.UTC).strftime('%H:%M:%S')} UTC "
        f"| open trades: {total} (TJR {total_tjr} | SCALP {total_scalp}) "
        f"| tickers active: {len(TICKERS)}"
    )
    print(msg, flush=True)

    # Show 5m movement for ALL tickers (stocks + FX)
    for t in TICKERS:
        print_recent_price_action(t)

    if HEARTBEAT_TELEGRAM:
        send_telegram(msg)

# =========================
# Scheduler
# =========================
schedule.every(30).seconds.do(check_smc)                      # TJR core
schedule.every(30).seconds.do(scan_for_sweep_momentum_trades) # TJR scout
schedule.every(12).seconds.do(check_scalper)                  # Independent scalper
schedule.every(30).seconds.do(check_positions)
schedule.every(60).seconds.do(heartbeat)
schedule.every(45).seconds.do(pull_fx_prices_rotating)

# =========================
# Launch
# =========================
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

