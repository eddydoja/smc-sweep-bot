import os
HEARTBEAT_TELEGRAM = False # keep heartbeats only on Render logs


# Map unsupported indices to ETFs Alpaca supports
TICKER_MAP = {
"DAX": "EWG", # iShares Germany ETF
"UKX": "EWU" # iShares UK ETF
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
'FX': (3, 17), # London 3am–5pm UTC
'US': (13, 20) # NY 9am–4pm EST (13–20 UTC approx)
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


def resolve_ticker(ticker):
return TICKER_MAP.get(ticker, ticker)


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
"""Fetch bars safely; if thin data, return empty DF (no Telegram spam)."""
try:
resolved = resolve_ticker(ticker)
# request larger history window when needed
bars = client.get_bars(resolved, timeframe, limit=min(limit, 1000)).df
if bars.empty or len(bars) < 10: # lowered requirement from 30 → 10
return pd.DataFrame()
# Candle features
bars['body'] = (bars['close'] - bars['open']).abs()
bars['range'] = bars['high'] - bars['low']
bars['upper_wick'] = bars['high'] - bars[["open", "close"]].max(axis=1)
bars['lower_wick'] = bars[["open", "close"]].min(axis=1) - bars['low']
bars['is_doji'] = (bars['body'] / bars['range']).fillna(0) < 0.1
# Patterns
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
# Momentum candles (broader trigger)
atr = bars['range'].rolling(14).mean()
bars['bull_momo'] = (bars['close'] > bars['open']) & (bars['body'] > atr * 0.8) & (bars['close'] > bars['high'].shift(1))
bars['bear_momo'] = (bars['close'] < bars['open']) & (bars['body'] > atr * 0.8) & (bars['close'] < bars['low'].shift(1))
return bars
except Exception as e:
# Only print; do not Telegram every time
print(f"⚠️ Data fetch failed for {ticker}: {e}", flush=True)
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
        trend = "bullish"; bos = True
    elif latest_highs.iloc[1]['high'] < latest_highs.iloc[0]['high'] and latest_lows.iloc[1]['low'] < latest_lows.iloc[0]['low']:
        trend = "bearish"; bos = True

    return {"trend": trend, "swing_highs": swing_highs, "swing_lows": swing_lows, "bos": bos}

def detect_fvg(df):
    for i in range(2, len(df)):
        if df.iloc[i-2]['low'] > df.iloc[i]['high']:
            return {'start': df.iloc[i-2]['low'], 'end': df.iloc[i]['high'], 'index': i}
        if df.iloc[i]['low'] > df.iloc[i-2]['high']:
            return {'start': df.iloc[i]['low'], 'end': df.iloc[i-2]['high'], 'index': i}
    return None

def detect_order_block(df):
    for i in range(len(df)-3, 0, -1):
        # simple OB heuristic
        if df.iloc[i]['close'] < df.iloc[i]['open'] and df.iloc[i+1]['close'] > df.iloc[i+1]['open']:
            return {'open': df.iloc[i]['open'], 'close': df.iloc[i]['close']}
        if df.iloc[i]['close'] > df.iloc[i]['open'] and df.iloc[i+1]['close'] < df.iloc[i+1]['open']:
            return {'open': df.iloc[i]['open'], 'close': df.iloc[i]['close']}
    return None

def calculate_confidence(df, last_candle):
    score = 0
    if last_candle.get('bullish_engulfing', False) or last_candle.get('bearish_engulfing', False):
        score += 2
    if last_candle.get('hammer', False) or last_candle.get('inverted_hammer', False):
        score += 1
    if last_candle.get('3bar_bullish', False) or last_candle.get('3bar_bearish', False):
        score += 2
    if last_candle.get('is_doji', False):
        score += 1
    if last_candle.get('wick_rejection_down', False) or last_candle.get('wick_rejection_up', False):
        score += 1
    if last_candle.get('liquidity_sweep_high', False) or last_candle.get('liquidity_sweep_low', False):
        score += 3  # heavier weight
    if last_candle.get('bull_momo', False) or last_candle.get('bear_momo', False):
        score += 2
    return int(min(10, max(1, score)))

def determine_qty(confidence):
    return min(10, max(1, confidence))

def calculate_confidence_tp(df, side, entry_price):
    avg_range = df['range'][-20:].mean()
    # more conservative when momentum candle not present
    base_mult = 1.6
    mult = base_mult + 0.6  # default 2.2R-ish of avg range
    return round(entry_price + avg_range * mult, 2) if side == 'long' else round(entry_price - avg_range * mult, 2)

def check_positions():
    for ticker, positions in open_positions.items():
        for pos in positions[:]:
            try:
                order = client.get_order(pos['id'])
                if order.filled_avg_price:
                    current_price = client.get_latest_trade(ticker).price
                    gain = (current_price - pos['entry']) / pos['entry'] * 100 if pos['side'] == 'long' else (pos['entry'] - current_price) / pos['entry'] * 100
                    if gain <= -1 or gain >= pos['tp']:
                        close_side = 'sell' if pos['side'] == 'long' else 'buy'
                        if TRADE_EXECUTION:
                            client.submit_order(symbol=ticker, qty=pos['qty'], side=close_side, type='market', time_in_force='gtc')
                        send_telegram(f"📤 Exited {pos['side'].upper()} {pos['qty']} {ticker} at gain/loss: {gain:.2f}%")
                        positions.remove(pos)
            except Exception as e:
                send_telegram(f"⚠️ Error checking {ticker} position: {e}")

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
        # shorter cooldown to increase frequency
        if last_signal and datetime.now(pytz.UTC) - last_signal < timedelta(seconds=30):
            continue

        df = get_data(ticker)
        if df.shape[0] < 50:
            continue

        ltf = detect_structure(df)
        if not ltf or not ltf['bos'] or ltf['trend'] == 'neutral':
            continue

        # HTF alignment: relax to match EITHER H1 or H4
        h1_df = get_data(ticker, timeframe=TimeFrame.Hour, limit=50)
        h4_df = get_data(ticker, timeframe=TimeFrame.Hour, limit=23)  # proxy for H4 sample window
        h1 = detect_structure(h1_df) if not h1_df.empty else None
        h4 = detect_structure(h4_df) if not h4_df.empty else None
        if not h1 or not h4:
            continue
        if (ltf['trend'] != h1['trend']) and (ltf['trend'] != h4['trend']):
            continue

        fvg = detect_fvg(df); ob = detect_order_block(df)
        if not fvg or not ob:
            continue

        last = df.iloc[-1]
        price = float(last['close'])
        entry_signal = False; side = None

        # Zone check
        in_bull_zone = (ltf['trend'] == 'bullish') and (price < fvg['start']) and (price >= ob['close'])
        in_bear_zone = (ltf['trend'] == 'bearish') and (price > fvg['start']) and (price <= ob['close'])

        # Expanded triggers: include momentum candles
        if in_bull_zone and any([last['bullish_engulfing'], last['hammer'], last['is_doji'], last['wick_rejection_down'], last['3bar_bullish'], last['liquidity_sweep_low'], last['bull_momo']]):
            entry_signal = True; side = 'long'
        if in_bear_zone and any([last['bearish_engulfing'], last['inverted_hammer'], last['is_doji'], last['wick_rejection_up'], last['3bar_bearish'], last['liquidity_sweep_high'], last['bear_momo']]):
            entry_signal = True; side = 'short'

        if not entry_signal:
            continue

        confidence = calculate_confidence(df, last)
        entry = price
        sl = round(entry * 0.99, 2) if side == 'long' else round(entry * 1.01, 2)
        tp = calculate_confidence_tp(df, side, entry)
        tp_pct = abs((tp - entry) / entry * 100)
        qty = determine_qty(confidence)

        try:
            if TRADE_EXECUTION:
                order = client.submit_order(
                    symbol=ticker, qty=qty, side=('buy' if side == 'long' else 'sell'), type='market', time_in_force='gtc',
                    order_class='bracket', stop_loss={"stop_price": str(sl)}, take_profit={"limit_price": str(tp)}
                )
                order_id = order.id
            else:
                order_id = f"signal-{datetime.now(pytz.UTC).isoformat()}"
            open_positions[ticker].append({'id': order_id, 'side': side, 'entry': entry, 'tp': tp_pct, 'qty': qty, 'timestamp': datetime.now(pytz.UTC)})
            last_signal_time[ticker] = datetime.now(pytz.UTC)
            send_telegram(f"📥 {ticker} SIGNAL: {side.upper()} {qty} @ {entry}\nTP: {tp} ({tp_pct:.1f}%) | SL: {sl} (1%) | Confidence: {confidence}/10")
        except Exception as e:
            send_telegram(f"❌ Order error on {ticker}: {e}")

# Heartbeat for Render logs (and optional Telegram)

def heartbeat():
    total = sum(len(v) for v in open_positions.values())
    msg = f"⏱️ Heartbeat {datetime.now(pytz.UTC).strftime('%H:%M:%S')} UTC | open trades: {total} | tickers active: {len(TICKERS)}"
    print(msg, flush=True)
    if HEARTBEAT_TELEGRAM:
        send_telegram(msg)


def check_positions():
    for ticker, positions in open_positions.items():
        for pos in positions[:]:
            try:
                order = client.get_order(pos['id']) if TRADE_EXECUTION else None
                current_price = client.get_latest_trade(ticker).price
                gain = (current_price - pos['entry']) / pos['entry'] * 100 if pos['side'] == 'long' else (pos['entry'] - current_price) / pos['entry'] * 100
                if gain <= -1 or gain >= pos['tp']:
                    close_side = 'sell' if pos['side'] == 'long' else 'buy'
                    if TRADE_EXECUTION:
                        client.submit_order(symbol=ticker, qty=pos['qty'], side=close_side, type='market', time_in_force='gtc')
                    send_telegram(f"📤 Exited {pos['side'].upper()} {pos['qty']} {ticker} at gain/loss: {gain:.2f}%")
                    positions.remove(pos)
            except Exception as e:
                send_telegram(f"⚠️ Error checking {ticker} position: {e}")

# Scheduling
schedule.every(30).seconds.do(check_smc)
schedule.every(30).seconds.do(check_positions)
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

