import os
import time
import threading
import logging
import math # Necesario para la función de redondeo de cantidades
from datetime import datetime
import pandas as pd
from flask import Flask, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ---------------- Config desde ENV ----------------
# Las variables de entorno son cruciales para la seguridad.
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes") # Modo simulación
SYMBOL = os.environ.get("SYMBOL", "TRXUSDT") # Par a operar
PCT_OF_BALANCE = float(os.environ.get("PCT_OF_BALANCE", 0.02)) # Porcentaje de USDT a usar en cada compra (2%)
MIN_ORDER_USD = float(os.environ.get("MIN_ORDER_USD", 10.0)) # Mínimo de USD para una orden
INTERVAL = os.environ.get("INTERVAL", "15m") # Intervalo de las velas
SLEEP_SEC = int(os.environ.get("SLEEP_SEC", 300)) # Tiempo de espera entre cada ciclo (5 minutos)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger()

# ---------------- Binance Client (FORZAR TESTNET) ----------------
if not API_KEY or not API_SECRET:
    logger.error("❌ Faltan BINANCE_API_KEY / BINANCE_API_SECRET en variables de entorno.")
    raise SystemExit(1)

client = Client(API_KEY, API_SECRET)
# ⚠️ Forzar URL para conectar a Binance TESTNET
client.API_URL = 'https://testnet.binance.vision/api'
logger.info("✅ Conectado a Binance TESTNET")

# ---------------- Estado Global del Bot ----------------
bot_state = {
    "last_signal": None, 
    "last_run": None, 
    "trades": [], # Historial de transacciones
    "symbol_info": {}, # Guarda la precisión del símbolo
}

# ---------------- Funciones Auxiliares ----------------

def get_asset_balance(asset):
    """Obtiene el balance LIBRE de un activo específico (e.g., 'USDT' o 'TRX')."""
    try:
        bal = client.get_asset_balance(asset=asset)
        return float(bal.get('free', 0))
    except Exception as e:
        logger.error(f"Error al obtener balance de {asset}: {e}")
        return 0.0

def round_step_size(quantity, step_size):
    """
    Asegura que la cantidad (quantity) cumpla con el step_size de Binance 
    (la cantidad de decimales permitidos).
    """
    if step_size == 0:
        return 0.0
        
    # Calcular la precisión a partir del step_size (ej: 0.01 -> 2 decimales)
    precision = int(round(-math.log10(step_size)))
    
    # Redondear hacia abajo (floor) para asegurar que no se exceda el balance
    return math.floor(quantity * (10**precision)) / (10**precision)

def get_symbol_info(symbol):
    """Obtiene y almacena la información de filtros del símbolo (minQty, stepSize, etc)."""
    if symbol in bot_state["symbol_info"]:
        return bot_state["symbol_info"][symbol]

    try:
        info = client.get_symbol_info(symbol)
        filters = {f['filterType']: f for f in info['filters']}
        
        step_size = float(filters['LOT_SIZE']['stepSize'])
        min_qty = float(filters['LOT_SIZE']['minQty'])
        
        symbol_info = {
            'min_qty': min_qty,
            'step_size': step_size,
            'base_asset': info['baseAsset'],  # e.g., TRX
            'quote_asset': info['quoteAsset'], # e.g., USDT
        }
        bot_state["symbol_info"][symbol] = symbol_info
        logger.info(f"✅ Info Símbolo {symbol} cargada: Min Qty={min_qty}, Step Size={step_size}")
        return symbol_info

    except Exception as e:
        logger.error(f"❌ Error al cargar info del símbolo {symbol}: {e}")
        return None

# ---------------- Funciones de Análisis ----------------

def fetch_klines(symbol, interval="15m", limit=500):
    """Descarga los datos de velas (klines) de Binance y los convierte a DataFrame."""
    kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    return df.set_index("open_time")

def compute_rsi(series, period=14):
    """Calcula el índice de fuerza relativa (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    # Evitar división por cero
    with pd.option_context('mode.use_inf_as_na', True):
        rs = up.div(down.replace(0, pd.NA)).fillna(0)
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    """Calcula las Medias Móviles y el RSI."""
    df = df.copy()
    df["ma_short"] = df["close"].rolling(9).mean()
    df["ma_long"] = df["close"].rolling(21).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df.dropna(inplace=True)
    return df

def signal_from_df(df):
    """
    Genera la señal de trading:
    BUY: Cruce MA (corta > larga) + RSI < 70.
    SELL: Cruce MA (corta < larga) O RSI > 80.
    """
    if len(df) < 2:
        return "HOLD"
        
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Cruce MA: La MA corta pasa de estar por debajo a por encima de la larga.
    cruce_up = (prev["ma_short"] <= prev["ma_long"]) and (last["ma_short"] > last["ma_long"])
    # Cruce MA: La MA corta pasa de estar por encima a por debajo de la larga.
    cruce_down = (prev["ma_short"] >= prev["ma_long"]) and (last["ma_short"] < last["ma_long"])
    
    # Señal de Compra
    buy = cruce_up and (last["rsi"] < 70)
    
    # Señal de Venta (por cruce o sobrecompra)
    sell = cruce_down or (last["rsi"] > 80)
    
    return "BUY" if buy else ("SELL" if sell else "HOLD")

# ---------------- Lógica de Ejecución de Trading ----------------

def execute_trade(signal, current_price, symbol_info):
    """Ejecuta una orden de COMPRA o VENTA, respetando DRY_RUN y las reglas de Binance."""
    
    base_asset = symbol_info['base_asset']
    quote_asset = symbol_info['quote_asset']
    
    order_id = None
    order_status = "SIMULADO" if DRY_RUN else "FALLIDO"
    executed_qty = 0.0
    side = None
    
    # Asegurarse de no operar dos veces con la misma señal consecutiva (evitar spam)
    if signal == bot_state["last_signal"]:
        logger.info(f"⏳ Señal {signal} ya procesada en el ciclo anterior. Esperando...")
        return
        
    try:
        if signal == "BUY":
            side = Client.SIDE_BUY
            quote_balance = get_asset_balance(quote_asset) # Balance de USDT
            
            # Cantidad de USDT a usar (porcentaje del balance libre)
            usdt_to_spend = quote_balance * PCT_OF_BALANCE
            
            # 1. Chequeo de mínimos de USDT
            if usdt_to_spend < MIN_ORDER_USD:
                logger.warning(f"⚠️ Compra ignorada: Monto a gastar ({usdt_to_spend:.2f} {quote_asset}) es menor que MIN_ORDER_USD ({MIN_ORDER_USD}).")
                return
            
            # 2. Cálculo y redondeo de cantidad de activo base
            raw_qty = usdt_to_spend / current_price
            quantity = round_step_size(raw_qty, symbol_info['step_size'])
            
            # 3. Chequeo de mínimos de cantidad de activo base
            if quantity < symbol_info['min_qty']:
                 logger.warning(f"⚠️ Compra ignorada: Cantidad redondeada ({quantity}) es menor que Min Qty ({symbol_info['min_qty']}).")
                 return
                 
            logger.info(f"💰 BUY: Intentando comprar {quantity} {base_asset} (Inversión: {usdt_to_spend:.2f} {quote_asset})")

            if not DRY_RUN:
                order = client.create_order(
                    symbol=SYMBOL,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                order_id = order.get('orderId')
                executed_qty = float(order.get('executedQty', quantity))
                order_status = "EJECUTADO"
                logger.info(f"✅ Orden BUY {order_id} EJECUTADA. Cantidad: {executed_qty}")

        elif signal == "SELL":
            side = Client.SIDE_SELL
            base_balance = get_asset_balance(base_asset) # Balance de TRX
            
            # 1. Cálculo y redondeo de la cantidad a vender (usar todo el balance libre)
            quantity = round_step_size(base_balance, symbol_info['step_size'])
            
            # 2. Chequeo de mínimos de cantidad de activo base
            if quantity < symbol_info['min_qty']:
                logger.warning(f"⚠️ Venta ignorada: Cantidad disponible ({quantity}) es menor que Min Qty ({symbol_info['min_qty']}).")
                return

            # 3. Chequeo de mínimos por valor estimado en USDT
            estimated_usd_value = quantity * current_price
            if estimated_usd_value < MIN_ORDER_USD:
                logger.warning(f"⚠️ Venta ignorada: Valor estimado ({estimated_usd_value:.2f} {quote_asset}) es menor que MIN_ORDER_USD ({MIN_ORDER_USD}).")
                return
                
            logger.info(f"💸 SELL: Intentando vender {quantity} {base_asset} (Valor estimado: {estimated_usd_value:.2f} {quote_asset})")
            
            if not DRY_RUN:
                order = client.create_order(
                    symbol=SYMBOL,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                order_id = order.get('orderId')
                executed_qty = float(order.get('executedQty', quantity))
                order_status = "EJECUTADO"
                logger.info(f"✅ Orden SELL {order_id} EJECUTADA. Cantidad: {executed_qty}")
        
        # Actualizar la señal solo si se intentó una compra/venta (HOLD no actualiza)
        if side:
            bot_state["last_signal"] = signal

    except BinanceAPIException as e:
        logger.error(f"❌ Binance API Error al ejecutar {signal}: {e}")
        order_status = "ERROR_API"
        executed_qty = 0.0
    except Exception as e:
        logger.error(f"❌ Error desconocido al ejecutar {signal}: {e}")
        order_status = "ERROR_UNK"
        executed_qty = 0.0
    finally:
        # Registrar el trade (solo si fue BUY o SELL)
        if side:
            bot_state["trades"].append({
                "time": datetime.utcnow().isoformat(),
                "symbol": SYMBOL,
                "signal": signal,
                "price": current_price,
                # Si fue DRY_RUN, registramos la cantidad que intentamos usar
                "quantity": executed_qty if executed_qty > 0 else (quantity if DRY_RUN else 0.0), 
                "side": side,
                "status": order_status,
                "order_id": order_id,
                "dry_run": DRY_RUN
            })


# ---------------- Loop Principal del Bot ----------------
def bot_loop():
    """Bucle infinito que ejecuta la estrategia."""
    logger.info("🤖 Bot iniciado en TESTNET (DRY_RUN=%s, SYMBOL=%s)", DRY_RUN, SYMBOL)
    
    # 1. Cargar información del símbolo una sola vez
    symbol_info = get_symbol_info(SYMBOL)
    if not symbol_info:
        logger.error("❌ No se pudo cargar la información del símbolo. Deteniendo bot.")
        return

    # 2. Iniciar el bucle de trading
    while True:
        try:
            # a. Obtener y calcular indicadores
            df = fetch_klines(SYMBOL, interval=INTERVAL, limit=500)
            df = compute_indicators(df)
            
            # b. Obtener señal
            sig = signal_from_df(df)
            current_price = df.iloc[-1]["close"]
            
            # c. Ejecutar lógica de trade
            if sig in ["BUY", "SELL"]:
                execute_trade(sig, current_price, symbol_info)

            # d. Actualizar estado y logs para HOLD
            if sig == "HOLD":
                logger.info(f"⏸️ Señal: HOLD. Precio actual: {current_price}")
                
            bot_state["last_run"] = datetime.utcnow().isoformat()
            
            # e. Esperar
            time.sleep(SLEEP_SEC)
            
        except Exception as e:
            logger.exception("❌ Error en loop. Reintentando en 30 segundos: %s", e)
            time.sleep(30)

# ---------------- Flask Health/Status ----------------
app = Flask(__name__)

@app.route("/")
def health():
    """Muestra un resumen del estado del bot."""
    last_trade = bot_state["trades"][-1] if bot_state["trades"] else "N/A"
    return jsonify({
        "status": "ok",
        "symbol": SYMBOL,
        "dry_run": DRY_RUN,
        "interval": INTERVAL,
        "last_signal": bot_state.get("last_signal"),
        "last_run_utc": bot_state.get("last_run"),
        "last_trade": last_trade
    })

@app.route("/state")
def full_state():
    """Muestra el estado completo del bot, incluyendo el historial de trades."""
    # Obtener balances actuales para el reporte de estado
    info = bot_state["symbol_info"].get(SYMBOL, {})
    base_asset = info.get('base_asset', SYMBOL[:3])
    quote_asset = info.get('quote_asset', SYMBOL[3:])
    
    current_balances = {
        f"free_{base_asset}": get_asset_balance(base_asset),
        f"free_{quote_asset}": get_asset_balance(quote_asset),
    }

    return jsonify({
        "configuration": {
            "dry_run": DRY_RUN,
            "symbol": SYMBOL,
            "pct_of_balance": PCT_OF_BALANCE,
            "min_order_usd": MIN_ORDER_USD,
            "interval": INTERVAL,
            "sleep_sec": SLEEP_SEC
        },
        "current_state": {
            "last_signal": bot_state.get("last_signal"),
            "last_run_utc": bot_state.get("last_run"),
            "balances": current_balances
        },
        "trade_history": bot_state["trades"]
    })


if __name__ == "__main__":
    # Iniciar el bot de trading en un hilo separado
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    
    # Iniciar el servidor web (Flask)
    port = int(os.environ.get("PORT", 5000))
    logger.info("🌐 Servidor Flask iniciado en 0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port)
