# coding: utf-8
import os
import time
import threading
import logging
import json
import math
from datetime import datetime, timedelta

# Importaciones de Machine Learning
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import dump, load # Para serializar el modelo

# ---------------- CONFIGURACIÓN GLOBAL Y VARIABLES DE ENTORNO ----------------

# CLAVES CRÍTICAS (Deben estar en variables de entorno de Render)
API_KEY = os.environ.get('BINANCE_API_KEY', 'TU_API_KEY_AQUI')
API_SECRET = os.environ.get('BINANCE_API_SECRET', 'TU_SECRET_KEY_AQUI')

# CONFIGURACIÓN DEL BOT Y RIESGO
CYCLE_DELAY_SECONDS = int(os.environ.get('CYCLE_DELAY_SECONDS', 600)) 
FUTURES_TESTNET_URL = os.environ.get('FUTURES_TESTNET_URL', 'https://testnet.binancefuture.com')

# Control de Modo y Entorno
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() in ('1', 'true', 'yes')
USE_TESTNET = os.environ.get('USE_TESTNET', 'true').lower() in ('1', 'true', 'yes')

# Parámetros de Trading
SYMBOL_PAIRS = os.environ.get('SYMBOL_PAIRS', 'BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT').split(',')
INTERVAL = os.environ.get('INTERVAL', '15m')
LEVERAGE = int(os.environ.get('LEVERAGE', 5))
MIN_ORDER_USD = float(os.environ.get('MIN_ORDER_USD', 10.5))

# --- MEJORA DE RIESGO --- Límite de posiciones abiertas simultáneamente
MAX_OPEN_POSITIONS = int(os.environ.get('MAX_OPEN_POSITIONS', 2))

# Parámetros de Riesgo
TP_FACTOR = float(os.environ.get('TP_FACTOR', 0.045))
SL_FACTOR = float(os.environ.get('SL_FACTOR', 0.015))
TRAILING_STOP_PERCENT = float(os.environ.get('TRAILING_STOP_PERCENT', 0.012))
# El riesgo por operación ahora es dinámico, este es el valor base.
BASE_RISK_PER_TRADE = float(os.environ.get('BASE_RISK_PER_TRADE', 0.03)) 

# Parámetros de Machine Learning
MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', 0.60))
DAYS_FOR_TRAINING = int(os.environ.get('DAYS_FOR_TRAINING', 60))

# ---------------- ESTADO GLOBAL Y CLIENTES ----------------
APP_STATE = {
    'dry_run': DRY_RUN,
    'status': 'STARTING',
    'last_run_utc': None,
    'model_ml_ready': False,
    'balances': {'free_USDT': 1000.00 if DRY_RUN else 0.00, 'free_BNB': 0.00},
    'symbol_data': {},
    'open_positions': {},
    'symbol_precision': {},
}

CLIENT = None
PUBLIC_CLIENT = None
ML_MODEL = None
ML_SCALER = None

# ---------------- CONFIGURACIÓN INICIAL Y LOGGING ----------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger()

def initialize_client(api_key, api_secret):
    """Inicializa los clientes de Binance."""
    global CLIENT, PUBLIC_CLIENT
    if not api_key or not api_secret:
        logger.error("❌ ERROR CRÍTICO: Faltan BINANCE_API_KEY/SECRET en ENV.")
        raise SystemExit(1)

    PUBLIC_CLIENT = Client()
    CLIENT = Client(api_key, api_secret)
    CLIENT.base_url = 'https://api.binance.com/api'

    if USE_TESTNET:
        CLIENT.futures_base_url = FUTURES_TESTNET_URL
        logger.info(f"✅ Conectado a Binance TESTNET (SIMULACIÓN).")
    else:
        CLIENT.futures_base_url = 'https://fapi.binance.com'
        logger.info(f"✅ Conectado a Binance PRODUCCIÓN (DINERO REAL).")

    try:
        server_time = CLIENT.get_server_time()
        local_time = int(time.time() * 1000)
        CLIENT.timestamp_offset = server_time['serverTime'] - local_time
        logger.info(f"✅ Tiempo del servidor sincronizado. Offset: {CLIENT.timestamp_offset} ms.")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo sincronizar el tiempo del servidor: {e}")

def load_symbol_precision(symbol):
    """Obtiene y almacena la precisión de cantidad y precio de un símbolo."""
    global APP_STATE
    try:
        info = PUBLIC_CLIENT.futures_exchange_info()
        symbol_info = next(item for item in info['symbols'] if item['symbol'] == symbol)
        APP_STATE['symbol_precision'][symbol] = {
            'quantity_precision': symbol_info['quantityPrecision'],
            'price_precision': symbol_info['pricePrecision']
        }
        return True
    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO: No se pudo obtener la info de precisión para {symbol}: {e}")
        return False

# ----------------------------------------------------------------------------------
# FUNCIONES DE DATOS Y ML
# ----------------------------------------------------------------------------------

def get_binance_data(symbol, interval, lookback_days):
    """Descarga data histórica para entrenamiento ML."""
    try:
        start_str = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%d %b, %Y")
        klines = CLIENT.futures_historical_klines(symbol=symbol, interval=interval, start_str=start_str)
        if not klines:
            logger.error(f"❌ No se recibieron Klines para {symbol} en el intervalo {interval}.")
            return pd.DataFrame()

        data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"])
        data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
        data = data[["open_time", "open", "high", "low", "close"]]
        numeric_cols = ["open", "high", "low", "close"]
        data[numeric_cols] = data[numeric_cols].astype(float)
        data.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
        data.set_index('open_time', inplace=True)
        return data
    except Exception as e:
        logger.error(f"❌ Fallo al descargar data histórica de {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calcula todas las features de ML."""
    df_new = df.copy()
    # RSI
    delta = df_new['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_new['RSI'] = 100 - (100 / (1 + (gain / loss)))
    # MACD
    ema_fast = df_new['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df_new['close'].ewm(span=26, adjust=False).mean()
    df_new['MACD'] = ema_fast - ema_slow
    df_new['MACD_Signal'] = df_new['MACD'].ewm(span=9, adjust=False).mean()
    # EMA
    df_new['EMA_9'] = df_new['close'].ewm(span=9, adjust=False).mean()
    df_new['EMA_21'] = df_new['close'].ewm(span=21, adjust=False).mean()
    df_new['EMA_Diff'] = df_new['EMA_9'] - df_new['EMA_21']
    # ATR
    df_new['TR'] = np.maximum(df_new['high'] - df_new['low'], np.maximum(abs(df_new['high'] - df_new['close'].shift(1)), abs(df_new['low'] - df_new['close'].shift(1))))
    df_new['ATR'] = df_new['TR'].rolling(window=14).mean()
    # Bollinger Bands
    rolling_mean_20 = df_new['close'].rolling(window=20).mean()
    rolling_std_20 = df_new['close'].rolling(window=20).std()
    df_new['bb_upper'] = rolling_mean_20 + (rolling_std_20 * 2)
    df_new['bb_lower'] = rolling_mean_20 - (rolling_std_20 * 2)
    df_new['bb_width'] = (df_new['bb_upper'] - df_new['bb_lower']) / rolling_mean_20
    # Stochastic Oscillator
    low_14 = df_new['low'].rolling(window=14).min()
    high_14 = df_new['high'].rolling(window=14).max()
    df_new['stoch_k'] = 100 * ((df_new['close'] - low_14) / (high_14 - low_14))
    df_new['stoch_d'] = df_new['stoch_k'].rolling(window=3).mean()
    return df_new.dropna()

def create_target(df):
    future_close = df['close'].shift(-4)
    price_change = (future_close - df['close']) / df['close']
    df['Target'] = np.where(price_change > 0.01, 1, 0)
    return df.iloc[:-4].dropna()

def calculate_ml_features(df):
    return df[['RSI', 'MACD', 'MACD_Signal', 'EMA_Diff', 'ATR', 'bb_width', 'stoch_k', 'stoch_d']].dropna()

def initialize_ml_model(symbol):
    global ML_MODEL, ML_SCALER
    logger.warning("Modelo ML no encontrado en memoria. INICIANDO ENTRENAMIENTO...")
    try:
        df = get_binance_data(symbol, INTERVAL, DAYS_FOR_TRAINING)
        if df.empty:
            logger.error("❌ ERROR: Data histórica vacía. Entrenando fallido.")
            return
        df = calculate_indicators(df)
        df_target = create_target(df)
        X = calculate_ml_features(df_target)
        y = df_target['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test_scaled, y_test)
        ML_MODEL = model
        ML_SCALER = scaler
        logger.info(f"✅ Modelo ML cargado en memoria. Precisión (entrenamiento): {accuracy:.4f}")
    except Exception as e:
        logger.critical(f"❌ Error CRÍTICO durante el entrenamiento del modelo ML: {e}.")

# ----------------------------------------------------------------------------------
# FUNCIONES DE TRADING Y EJECUCIÓN
# ----------------------------------------------------------------------------------

def round_value(value, precision):
    if not isinstance(precision, int) or precision < 0: return value
    factor = 10**precision
    return math.floor(value * factor) / factor

def get_current_price(symbol):
    try:
        return float(PUBLIC_CLIENT.futures_symbol_ticker(symbol=symbol)['price'])
    except Exception: return None

def update_balances(is_simulation=True):
    global APP_STATE
    if is_simulation:
        if 'free_USDT' not in APP_STATE['balances']: APP_STATE['balances']['free_USDT'] = 1000.00
        logger.info(f"✅ Balances actualizados (SIM). USDT disponible: {APP_STATE['balances']['free_USDT']:.2f}")
        return True
    return False

def make_decision(symbol, df_data):
    global ML_MODEL, ML_SCALER
    if ML_MODEL and ML_SCALER:
        try:
            X_live = calculate_ml_features(df_data.tail(1))
            if X_live.empty: return ('HOLD', 0.50, 'ML')
            X_live_scaled = ML_SCALER.transform(X_live)
            proba = ML_MODEL.predict_proba(X_live_scaled)[0]
            prob_buy = proba[1]
            if prob_buy >= MODEL_CONFIDENCE_THRESHOLD: signal = 'BUY'
            elif prob_buy <= (1 - MODEL_CONFIDENCE_THRESHOLD): signal = 'SELL'
            else: signal = 'HOLD'
            return (signal, prob_buy, 'ML')
        except Exception as e:
            logger.error(f"❌ Error durante la predicción ML: {e}.")
    return ('HOLD', 0.5, 'Manual') # Fallback simple

def execute_order(symbol, side, confidence):
    global APP_STATE
    try:
        current_price = get_current_price(symbol)
        if not current_price:
            logger.warning(f"⚠️ {symbol} - No se pudo obtener precio. Orden cancelada.")
            return

        # --- MEJORA IA --- Lógica de Tamaño de Posición Dinámico
        actual_confidence = confidence if side == 'BUY' else 1 - confidence
        conf_score = (actual_confidence - MODEL_CONFIDENCE_THRESHOLD) / (1 - MODEL_CONFIDENCE_THRESHOLD) # Normaliza 0-1
        
        MIN_RISK, MAX_RISK = 0.015, 0.04 # Arriesgar entre 1.5% y 4%
        dynamic_risk = MIN_RISK + (MAX_RISK - MIN_RISK) * conf_score
        
        usdt_free = APP_STATE['balances']['free_USDT']
        usdt_margin = usdt_free * dynamic_risk
        
        if usdt_margin < MIN_ORDER_USD:
             logger.warning(f"⚠️ {symbol} - Margen ({usdt_margin:.2f}) < Mínimo ({MIN_ORDER_USD}). Orden cancelada.")
             return

        quantity = (usdt_margin * LEVERAGE) / current_price
        q_precision = APP_STATE['symbol_precision'].get(symbol, {}).get('quantity_precision', 0)
        quantity = round_value(quantity, q_precision)
        
        if quantity == 0:
            logger.warning(f"⚠️ {symbol} - Cantidad es 0 tras redondeo. Orden cancelada.")
            return

        p_precision = APP_STATE['symbol_precision'].get(symbol, {}).get('price_precision', 0)
        tp_price = round_value(current_price * (1 + TP_FACTOR if side == 'BUY' else 1 - TP_FACTOR), p_precision)
        sl_price = round_value(current_price * (1 - SL_FACTOR if side == 'BUY' else 1 + SL_FACTOR), p_precision)
        
        if APP_STATE['dry_run']:
            APP_STATE['balances']['free_USDT'] -= usdt_margin
            APP_STATE['open_positions'][symbol] = {
                'side': side, 'entry_price': current_price, 'quantity': quantity,
                'margin_used': usdt_margin, 'stop_loss': sl_price, 'take_profit': tp_price
            }
            logger.info(f"✅ SIM: Posición {side} abierta. Margen: {usdt_margin:.2f} (Riesgo Din: {dynamic_risk*100:.1f}%)")
        else:
            logger.warning("🚨 MODO REAL ACTIVO: Se enviaría orden de mercado.")
    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO al ejecutar orden de {symbol}: {e}")

def manage_positions(symbol, current_price):
    global APP_STATE
    position = APP_STATE['open_positions'][symbol]
    p_precision = APP_STATE['symbol_precision'].get(symbol, {}).get('price_precision', 0)

    # Lógica de Trailing Stop
    if position['side'] == 'BUY':
        new_sl = current_price * (1 - TRAILING_STOP_PERCENT)
        if new_sl > position['stop_loss']:
            position['stop_loss'] = round_value(new_sl, p_precision)
            logger.info(f"    📈 {symbol} - Trailing Stop (L) a: {position['stop_loss']:.4f}")
    else: # SELL
        new_sl = current_price * (1 + TRAILING_STOP_PERCENT)
        if new_sl < position['stop_loss']:
            position['stop_loss'] = round_value(new_sl, p_precision)
            logger.info(f"    📉 {symbol} - Trailing Stop (S) a: {position['stop_loss']:.4f}")

    # Lógica de Cierre
    pnl = (current_price - position['entry_price']) * position['quantity'] * (-1 if position['side'] == 'SELL' else 1)
    close_reason = None
    if position['side'] == 'BUY' and (current_price >= position['take_profit'] or current_price <= position['stop_loss']):
        close_reason = 'TP' if current_price >= position['take_profit'] else 'SL'
    elif position['side'] == 'SELL' and (current_price <= position['take_profit'] or current_price >= position['stop_loss']):
        close_reason = 'TP' if current_price <= position['take_profit'] else 'SL'
        
    if close_reason:
        final_balance = position['margin_used'] + pnl
        APP_STATE['balances']['free_USDT'] += final_balance
        del APP_STATE['open_positions'][symbol]
        logger.critical(f"🎉 {symbol} - CIERRE POR {close_reason} | PnL: {pnl:.2f} USDT | Nuevo Balance: {APP_STATE['balances']['free_USDT']:.2f}")
    else:
        logger.info(f"    ℹ️ {symbol} - {position['side']} abierto. SL: {position['stop_loss']:.4f} | TP: {position['take_profit']:.4f}")

# ----------------------------------------------------------------------------------
# CICLO PRINCIPAL Y ARRANQUE
# ----------------------------------------------------------------------------------

def run_trading_bot():
    global APP_STATE, ML_MODEL, ML_SCALER
    logger.info(f"Bot iniciado en modo: {'DRY_RUN' if APP_STATE['dry_run'] else 'REAL'}.")
    if not ML_MODEL: initialize_ml_model('BTCUSDT')
    
    while True:
        APP_STATE['status'] = 'RUNNING'
        start_time = time.time()
        update_balances(is_simulation=True)
        
        for symbol in SYMBOL_PAIRS:
            logger.info(f"⚙️ Procesando {symbol}...")
            try:
                # --- MEJORA DE RIESGO --- Chequeo de límite de posiciones
                if symbol not in APP_STATE['open_positions'] and len(APP_STATE['open_positions']) >= MAX_OPEN_POSITIONS:
                    logger.info(f"    ⚠️ Límite de {MAX_OPEN_POSITIONS} posiciones alcanzado. Saltando {symbol}.")
                    continue

                df_data = get_binance_data(symbol, INTERVAL, 2)
                if df_data.empty or len(df_data) < 30:
                    logger.warning(f"⚠️ {symbol} - Data insuficiente para indicadores.")
                    continue
                
                df_indicators = calculate_indicators(df_data)
                current_price = df_indicators.iloc[-1]['close']
                
                if symbol in APP_STATE['open_positions']:
                    manage_positions(symbol, current_price)
                    continue

                signal, confidence, mode = make_decision(symbol, df_indicators)
                display_confidence = confidence if signal == 'BUY' else (1 - confidence if signal == 'SELL' else confidence)
                
                if signal in ['BUY', 'SELL']:
                    logger.info(f"✨ {symbol} | Señal {mode}: {signal} (Conf: {display_confidence:.4f}) | Precio: {current_price:.4f}")
                    execute_order(symbol, signal, confidence)
                else:
                    logger.info(f"    ℹ️ {symbol} | Señal {mode}: HOLD (Conf: {confidence:.4f}) | Precio: {current_price:.4f}")

            except Exception as e:
                logger.error(f"❌ Error al procesar {symbol}: {e}")
        
        APP_STATE['last_run_utc'] = datetime.utcnow().isoformat()
        APP_STATE['status'] = 'SLEEPING'
        elapsed_time = time.time() - start_time
        sleep_for = max(0, CYCLE_DELAY_SECONDS - elapsed_time)
        logger.info(f"🟢 Ciclo completado. Abiertas: {len(APP_STATE['open_positions'])}. Durmiendo por {int(sleep_for)} segundos.")
        time.sleep(sleep_for)

def run_trading_bot_thread():
    try:
        initialize_client(API_KEY, API_SECRET)
        for symbol in SYMBOL_PAIRS:
            load_symbol_precision(symbol)
        threading.Thread(target=run_trading_bot, daemon=True).start()
    except Exception as e:
        logger.critical(f"❌ FALLO CRÍTICO EN ARRANQUE: {e}")

# ----------------- FLASK APP -----------------
from flask import Flask, jsonify
app = Flask(__name__)
run_trading_bot_thread()

@app.route('/state', methods=['GET'])
def get_state(): return jsonify(current_state=APP_STATE)

@app.route('/', methods=['GET'])
def home(): return jsonify(message="Trading Bot Activo. Accede a /state para ver el estado.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

