# app.py — Bot MVP + simple health webserver (para Render)
# NOTA: dejar DRY_RUN=True hasta probar en Testnet y revisar logs.

import os, time, threading, logging
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ---------------- Config desde ENV ----------------
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() in ("1","true","yes")
USE_TESTNET = os.environ.get("USE_TESTNET", "true").lower() in ("1","true","yes")
SYMBOL = os.environ.get("SYMBOL", "TRXUSDT")
PCT_OF_BALANCE = float(os.environ.get("PCT_OF_BALANCE", 0.02))  # 2% default
MIN_ORDER_USD = float(os.environ.get("MIN_ORDER_USD", 10.0))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", 0.03))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", 0.05))
INTERVAL = os.environ.get("INTERVAL", "15m")
SLEEP_SEC = int(os.environ.get("SLEEP_SEC", 300))  # 5 minutes

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger()

# ---------------- Client init ----------------
if not API_KEY or not API_SECRET:
    logger.error("No BINANCE_API_KEY / BINANCE_API_SECRET set in environment. Exiting.")
    raise SystemExit(1)

client = Client(API_KEY, API_SECRET)
if USE_TESTNET:
    # For python-binance, set API_URL to the Testnet base
    client.API_URL = 'https://testnet.binance.vision/api'
    logger.info("Using Binance TESTNET API")

# ---------------- Utilities ----------------
def fetch_klines(symbol, interval="15m", limit=500):
    kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    return df.set_index("open_time")

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df = df.copy()
    df["ma_short"] = df["close"].rolling(9).mean()
    df["ma_long"] = df["close"].rolling(21).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df.dropna(inplace=True)
    return df

def account_balance_usdt():
    try:
        bal = client.get_asset_balance(asset='USDT')
        if bal is None:
            return 0.0
        return float(bal.get('free',0)) + float(bal.get('locked',0))
    except Exception as e:
        logger.error("Error fetching balance: %s", e)
        return 0.0

# ---------------- Trading helpers ----------------
def market_buy_usdt(symbol, usdt_amount):
    logger.info("BUY request: %s USDT of %s (DRY_RUN=%s)", usdt_amount, symbol, DRY_RUN)
    if DRY_RUN:
        return {"status":"DRY_RUN", "usdt": usdt_amount}
    try:
        price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        qty = usdt_amount / price
        # NOTE: rounding should respect lot size for production
        order = client.create_order(symbol=symbol, side='BUY', type='MARKET', quantity=round(qty,6))
        logger.info("BUY Order executed: %s", order)
        return order
    except BinanceAPIException as e:
        logger.error("Buy failed: %s", e)
        return None

def market_sell_symbol(symbol):
    logger.info("SELL request for %s (DRY_RUN=%s)", symbol, DRY_RUN)
    asset = symbol.replace("USDT","")
    try:
        bal = client.get_asset_balance(asset=asset)
        qty = float(bal['free']) if bal else 0.0
        if qty <= 0:
            logger.info("No position to sell for %s", asset)
            return None
        if DRY_RUN:
            return {"status":"DRY_RUN", "qty": qty}
        order = client.create_order(symbol=symbol, side='SELL', type='MARKET', quantity=qty)
        logger.info("SELL Order executed: %s", order)
        return order
    except BinanceAPIException as e:
        logger.error("Sell failed: %s", e)
        return None

# ---------------- Strategy ----------------
def signal_from_df(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    buy = (prev["ma_short"] <= prev["ma_long"]) and (last["ma_short"] > last["ma_long"]) and (last["rsi"] < 70)
    sell = ((prev["ma_short"] >= prev["ma_long"]) and (last["ma_short"] < last["ma_long"])) or (last["rsi"] > 80)
    return "BUY" if buy else ("SELL" if sell else "HOLD")

# ---------------- Bot main loop ----------------
bot_state = {"last_signal": None, "last_run": None, "trades": []}

def bot_loop():
    logger.info("Bot loop started. DRY_RUN=%s, TESTNET=%s, SYMBOL=%s", DRY_RUN, USE_TESTNET, SYMBOL)
    initial_balance = account_balance_usdt()
    logger.info("Initial USDT balance: %.2f", initial_balance)
    while True:
        try:
            df = fetch_klines(SYMBOL, interval=INTERVAL, limit=500)
            df = compute_indicators(df)
            sig = signal_from_df(df)
            bot_state["last_signal"] = sig
            bot_state["last_run"] = datetime.utcnow().isoformat()
            logger.info("Signal: %s", sig)
            if sig == "BUY":
                balance = account_balance_usdt()
                usdt_to_use = max(MIN_ORDER_USD, balance * PCT_OF_BALANCE)
                if usdt_to_use > balance:
                    usdt_to_use = balance * 0.5
                if usdt_to_use < MIN_ORDER_USD:
                    logger.info("No enough balance for minimum order.")
                else:
                    r = market_buy_usdt(SYMBOL, usdt_to_use)
                    bot_state["trades"].append({"time": datetime.utcnow().isoformat(), "side":"BUY", "detail": str(r)})
            elif sig == "SELL":
                r = market_sell_symbol(SYMBOL)
                bot_state["trades"].append({"time": datetime.utcnow().isoformat(), "side":"SELL", "detail": str(r)})
            # simple daily drawdown check
            cur_bal = account_balance_usdt()
            if initial_balance > 0 and (initial_balance - cur_bal) / initial_balance > 0.05:
                logger.warning("Daily drawdown >5%%. Pausing 1 hour.")
                time.sleep(3600)
            time.sleep(SLEEP_SEC)
        except Exception as e:
            logger.exception("Bot loop error: %s", e)
            time.sleep(30)

# ---------------- Flask health app ----------------
app = Flask(__name__)

@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "last_run": bot_state.get("last_run"),
        "last_signal": bot_state.get("last_signal"),
        "trades_count": len(bot_state.get("trades", [])),
        "dry_run": DRY_RUN,
        "testnet": USE_TESTNET,
        "symbol": SYMBOL
    })

if __name__ == "__main__":
    # Start bot in background thread
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
