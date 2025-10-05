import os, time, threading, logging
from datetime import datetime
import pandas as pd
from flask import Flask, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ---------------- Config desde ENV ----------------
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes")
SYMBOL = os.environ.get("SYMBOL", "TRXUSDT")
PCT_OF_BALANCE = float(os.environ.get("PCT_OF_BALANCE", 0.02))
MIN_ORDER_USD = float(os.environ.get("MIN_ORDER_USD", 10.0))
INTERVAL = os.environ.get("INTERVAL", "15m")
SLEEP_SEC = int(os.environ.get("SLEEP_SEC", 300))

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger()

# ---------------- Binance Client (FORZAR TESTNET) ----------------
if not API_KEY or not API_SECRET:
    logger.error("❌ Faltan BINANCE_API_KEY / BINANCE_API_SECRET en variables de entorno.")
    raise SystemExit(1)

client = Client(API_KEY, API_SECRET)
# ⚠️ Forzar URL antes del primer ping:
client.API_URL = 'https://testnet.binance.vision/api'
logger.info("✅ Conectado a Binance TESTNET")

# ---------------- Funciones ----------------
def fetch_klines(symbol, interval="15m", limit=500):
    kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
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
        return float(bal.get('free', 0)) + float(bal.get('locked', 0))
    except Exception as e:
        logger.error("Error balance: %s", e)
        return 0.0

# ---------------- Estrategia ----------------
def signal_from_df(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    buy = (prev["ma_short"] <= prev["ma_long"]) and (last["ma_short"] > last["ma_long"]) and (last["rsi"] < 70)
    sell = ((prev["ma_short"] >= prev["ma_long"]) and (last["ma_short"] < last["ma_long"])) or (last["rsi"] > 80)
    return "BUY" if buy else ("SELL" if sell else "HOLD")

bot_state = {"last_signal": None, "last_run": None, "trades": []}

def bot_loop():
    logger.info("🤖 Bot iniciado en TESTNET (DRY_RUN=%s, SYMBOL=%s)", DRY_RUN, SYMBOL)
    while True:
        try:
            df = fetch_klines(SYMBOL, interval=INTERVAL, limit=500)
            df = compute_indicators(df)
            sig = signal_from_df(df)
            bot_state["last_signal"] = sig
            bot_state["last_run"] = datetime.utcnow().isoformat()
            logger.info("📊 Señal: %s", sig)
            time.sleep(SLEEP_SEC)
        except Exception as e:
            logger.exception("Error en loop: %s", e)
            time.sleep(30)

# ---------------- Flask Health ----------------
app = Flask(__name__)

@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "last_signal": bot_state.get("last_signal"),
        "last_run": bot_state.get("last_run"),
        "dry_run": DRY_RUN,
        "symbol": SYMBOL
    })

if __name__ == "__main__":
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
