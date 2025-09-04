from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import schedule
import time
import datetime
import requests

# === CONFIG ===
import os  # Make sure this line is included

ALPACA_API_KEY = os.getenv('PKW2QGFBW74BYLLS48MS')
ALPACA_SECRET_KEY = os.getenv('V8K9NaWTpYdL9NNuQRG54e2EvvdTsXPBzrmUCVMI')
TELEGRAM_TOKEN = os.getenv('8405020655:AAHff_dafcrxkrLKLfQ1zjpYGKAudSM8H7w')
TELEGRAM_CHAT_ID = os.getenv('5079232641')
TICKER = 'SPY'
SESSION_HOURS = {
    'asia': (0, 8),
    'london': (3, 8),
    'ny': (9, 12)
}

# === INIT ALPACA CLIENT ===
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# === TELEGRAM ALERT ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    requests.post(url, data=payload)

# === FETCH CANDLES ===
def get_candles(symbol=TICKER, tf_minutes=1, limit=500):
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(minutes=limit)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=end_time
    )
    bars = client.get_stock_bars(request_params).df
    df = bars[bars.index.get_level_values(0) == symbol].copy().reset_index()
    df.rename(columns={'timestamp': 'Datetime'}, inplace=True)
    df['hour'] = df['Datetime'].dt.hour
    return df

# === SMC LOGIC ===
def detect_bias(df):
    recent_highs = df['high'][-60:].max()
    recent_lows = df['low'][-60:].min()
    current = df['close'].iloc[-1]
    if current > recent_highs:
        return 'bullish'
    elif current < recent_lows:
        return 'bearish'
    return 'neutral'

def get_session_ranges(df):
    asia = df[df['hour'].between(*SESSION_HOURS['asia'])]
    london = df[df['hour'].between(*SESSION_HOURS['london'])]
    ny = df[df['hour'].between(*SESSION_HOURS['ny'])]
    return {
        'asia_high': asia['high'].max(),
        'asia_low': asia['low'].min(),
        'london_high': london['high'].max(),
        'london_low': london['low'].min(),
        'ny_high': ny['high'].max(),
        'ny_low': ny['low'].min()
    }

def detect_asia_sweep(price, asia_high, asia_low):
    if price > asia_high:
        return 'buy-side sweep of Asia'
    elif price < asia_low:
        return 'sell-side sweep of Asia'
    return None

def detect_bos(df):
    recent_high = df['high'].iloc[-5]
    current_high = df['high'].iloc[-1]
    return current_high > recent_high

def generate_signal(df):
    bias = detect_bias(df)
    session = get_session_ranges(df)
    price = df['close'].iloc[-1]
    sweep = detect_asia_sweep(price, session['asia_high'], session['asia_low'])
    bos = detect_bos(df)

    if bias == 'bullish' and sweep == 'sell-side sweep of Asia' and bos:
        entry = price
        sl = session['asia_low']
        tp = entry + (entry - sl) * 1.5
        return f"\U0001F7E2 LONG | Asia Sweep | Entry: {entry:.2f} | SL: {sl:.2f} | TP1: {tp:.2f}"

    if bias == 'bearish' and sweep == 'buy-side sweep of Asia' and bos:
        entry = price
        sl = session['asia_high']
        tp = entry - (sl - entry) * 1.5
        return f"\U0001F534 SHORT | Asia Sweep | Entry: {entry:.2f} | SL: {sl:.2f} | TP1: {tp:.2f}"

    return None

# === RUN BOT ===
def run_bot():
    df = get_candles()
    signal = generate_signal(df)
    if signal:
        print(signal)
        send_telegram_message(signal)

schedule.every(1).minutes.do(run_bot)

print("Alpaca SMC Sweep Bot Running...")
while True:
    schedule.run_pending()
    time.sleep(1)


