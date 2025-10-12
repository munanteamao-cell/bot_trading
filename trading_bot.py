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
# --- MEJORA ESTRATÉGICA --- Timeframe para la tendencia a largo plazo
LONG_TERM_INTERVAL = os.environ.get('LONG_TERM_INTERVAL', '4h')
LEVERAGE = int(os.environ.get('LEVERAGE', 5))
MIN_ORDER_USD = float(os.environ.get('MIN_ORDER_USD', 10.5))

# --- MEJORA DE RIESGO --- Límite de posiciones abiertas simultáneamente
MAX_OPEN_POSITIONS = int(os.environ.get('MAX_OPEN_POSITIONS', 2))

# Parámetros de Riesgo
TP_FACTOR = float(os.environ.get('TP_FACTOR', 0.045))
SL_FACTOR = float(os.environ.get('SL_FACTOR', 0.015))
TRAILING_STOP_PERCENT = float(os.environ.get('TRAILING_STOP_PERCENT', 0.012))
BASE_RISK_PER_TRADE = float(os.environ.get('BASE_RISK_PER_TRADE', 0.03)) 

# Parámetros de Machine Learning
MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', 0.60))
DAYS_FOR_TRAINING = int(os.environ.get('DAYS_FOR_TRAINING', 60))

# ---------------- ESTADO GLOBAL Y CLIENTES ----------------
APP_STATE = {
    'dry_run': DRY_RUN, 'status': 'STARTING', 'last_run_utc': None, 'model_ml_ready': False,
    'balances': {'free_USDT': 1000.00 if DRY_RUN else 0.00, 'free_BNB': 0.00},
    'symbol_data': {}, 'open_positions': {}, 'symbol_precision': {},
}

CLIENT, PUBLIC_CLIENT, ML_MODEL, ML_SCALER = None, None, None, None

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
    CLIENT.futures_base_url = FUTURES_TESTNET_URL if USE_TESTNET else 'https://fapi.binance.com'
    logger.info(f"✅ Conectado a Binance {'TESTNET (SIMULACIÓN)' if USE_TESTNET else 'PRODUCCIÓN (DINERO REAL)'}.")
    try:
        server_time = CLIENT.get_server_time()['serverTime']
        CLIENT.timestamp_offset = server_time - int(time.time() * 1000)
        logger.info(f"✅ Tiempo del servidor sincronizado. Offset: {CLIENT.timestamp_offset} ms.")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo sincronizar el tiempo del servidor: {e}")

def load_symbol_precision(symbol):
    """Obtiene y almacena la precisión de un símbolo."""
    try:
        info = PUBLIC_CLIENT.futures_exchange_info()
        s_info = next(i for i in info['symbols'] if i['symbol'] == symbol)
        APP_STATE['symbol_precision'][symbol] = {'quantity_precision': s_info['quantityPrecision'], 'price_precision': s_info['pricePrecision']}
        return True
    except Exception as e:
        logger.error(f"❌ No se pudo obtener la info de precisión para {symbol}: {e}")
        return False

# ----------------------------------------------------------------------------------
# FUNCIONES DE DATOS Y ML
# ----------------------------------------------------------------------------------

def get_binance_data(symbol, interval, lookback_days):
    """Descarga data histórica."""
    try:
        start_str = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%d %b, %Y")
        klines = CLIENT.futures_historical_klines(symbol=symbol, interval=interval, start_str=start_str)
        if not klines: return pd.DataFrame()
        data = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"])
        data = data[["open_time", "open", "high", "low", "close"]]
        data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
        data[["open", "high", "low", "close"]] = data[["open", "high", "low", "close"]].astype(float)
        return data.drop_duplicates(subset=['open_time']).set_index('open_time')
    except Exception as e:
        logger.error(f"❌ Fallo al descargar data de {symbol} ({interval}): {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calcula todas las features de ML."""
    df['RSI'] = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / -df['close'].diff().where(lambda x: x < 0, 0).rolling(14).mean())))
    ema_12, ema_26 = df['close'].ewm(span=12, adjust=False).mean(), df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['EMA_Diff'] = df['close'].ewm(span=9, adjust=False).mean() - df['close'].ewm(span=21, adjust=False).mean()
    df['ATR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))).rolling(14).mean()
    mean_20, std_20 = df['close'].rolling(20).mean(), df['close'].rolling(20).std()
    df['bb_width'] = ((mean_20 + std_20 * 2) - (mean_20 - std_20 * 2)) / mean_20
    low_14, high_14 = df['low'].rolling(14).min(), df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    return df.dropna()

def create_target(df):
    df['Target'] = np.where(df['close'].shift(-4) / df['close'] > 1.01, 1, 0)
    return df.iloc[:-4].dropna()

def calculate_ml_features(df):
    return df[['RSI', 'MACD', 'MACD_Signal', 'EMA_Diff', 'ATR', 'bb_width', 'stoch_k', 'stoch_d']].dropna()

def initialize_ml_model(symbol):
    global ML_MODEL, ML_SCALER
    logger.warning("Modelo ML no encontrado. INICIANDO ENTRENAMIENTO...")
    try:
        df = get_binance_data(symbol, INTERVAL, DAYS_FOR_TRAINING)
        if df.empty: return logger.error("❌ Data histórica vacía. Entrenando fallido.")
        df_target = create_target(calculate_indicators(df))
        X, y = calculate_ml_features(df_target), df_target['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        ML_SCALER = StandardScaler()
        X_train_scaled, X_test_scaled = ML_SCALER.fit_transform(X_train), ML_SCALER.transform(X_test)
        ML_MODEL = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
        ML_MODEL.fit(X_train_scaled, y_train)
        logger.info(f"✅ Modelo ML cargado. Precisión: {ML_MODEL.score(X_test_scaled, y_test):.4f}")
    except Exception as e:
        logger.critical(f"❌ Error CRÍTICO durante el entrenamiento ML: {e}.")

def check_long_term_trend(symbol):
    """Verifica la tendencia principal en un timeframe mayor."""
    df_long = get_binance_data(symbol, LONG_TERM_INTERVAL, 30) # 30 velas de 4h son 5 días
    if df_long.empty or len(df_long) < 50:
        logger.warning(f"⚠️ {symbol} - Data insuficiente en {LONG_TERM_INTERVAL} para definir tendencia.")
        return 'NEUTRAL'
    
    ema_50 = df_long['close'].ewm(span=50, adjust=False).mean()
    current_price = df_long['close'].iloc[-1]
    
    if current_price > ema_50.iloc[-1]: return 'UPTREND'
    elif current_price < ema_50.iloc[-1]: return 'DOWNTREND'
    return 'NEUTRAL'
    
# ----------------------------------------------------------------------------------
# FUNCIONES DE TRADING
# ----------------------------------------------------------------------------------

def round_value(v, p): return math.floor(v * 10**p) / 10**p if isinstance(p, int) and p >= 0 else v

def make_decision(df):
    if not (ML_MODEL and ML_SCALER): return 'HOLD', 0.5
    try:
        X_live = ML_SCALER.transform(calculate_ml_features(df.tail(1)))
        prob_buy = ML_MODEL.predict_proba(X_live)[0][1]
        if prob_buy >= MODEL_CONFIDENCE_THRESHOLD: return 'BUY', prob_buy
        if prob_buy <= (1 - MODEL_CONFIDENCE_THRESHOLD): return 'SELL', prob_buy
        return 'HOLD', prob_buy
    except Exception as e:
        logger.error(f"❌ Error en predicción ML: {e}.")
        return 'HOLD', 0.5

def execute_order(symbol, side, confidence):
    current_price = get_binance_data(symbol, INTERVAL, 1)['close'].iloc[-1]
    if not current_price: return logger.warning(f"⚠️ {symbol} - No se pudo obtener precio.")
    
    actual_conf = confidence if side == 'BUY' else 1 - confidence
    conf_score = (actual_conf - MODEL_CONFIDENCE_THRESHOLD) / (1 - MODEL_CONFIDENCE_THRESHOLD)
    dynamic_risk = 0.015 + (0.04 - 0.015) * conf_score
    usdt_margin = APP_STATE['balances']['free_USDT'] * dynamic_risk
    
    if usdt_margin < MIN_ORDER_USD: return logger.warning(f"⚠️ {symbol} - Margen ({usdt_margin:.2f}) < Mínimo.")
    
    prec = APP_STATE['symbol_precision'].get(symbol, {})
    quantity = round_value((usdt_margin * LEVERAGE) / current_price, prec.get('quantity_precision', 0))
    if quantity == 0: return logger.warning(f"⚠️ {symbol} - Cantidad es 0.")

    tp = round_value(current_price * (1 + TP_FACTOR if side == 'BUY' else 1 - TP_FACTOR), prec.get('price_precision', 0))
    sl = round_value(current_price * (1 - SL_FACTOR if side == 'BUY' else 1 + SL_FACTOR), prec.get('price_precision', 0))
    
    APP_STATE['balances']['free_USDT'] -= usdt_margin
    APP_STATE['open_positions'][symbol] = {'side': side, 'entry_price': current_price, 'quantity': quantity, 'margin_used': usdt_margin, 'stop_loss': sl, 'take_profit': tp}
    logger.info(f"✅ SIM: {side} abierto. Margen: {usdt_margin:.2f} (Riesgo: {dynamic_risk*100:.1f}%)")

def manage_positions(symbol, current_price):
    pos = APP_STATE['open_positions'][symbol]
    prec = APP_STATE['symbol_precision'].get(symbol, {}).get('price_precision', 0)
    
    new_sl = current_price * (1 - TRAILING_STOP_PERCENT if pos['side'] == 'BUY' else 1 + TRAILING_STOP_PERCENT)
    if (pos['side'] == 'BUY' and new_sl > pos['stop_loss']) or (pos['side'] == 'SELL' and new_sl < pos['stop_loss']):
        pos['stop_loss'] = round_value(new_sl, prec)
        logger.info(f"    {'📈' if pos['side'] == 'BUY' else '📉'} {symbol} - Trailing Stop a: {pos['stop_loss']:.4f}")

    pnl = (current_price - pos['entry_price']) * pos['quantity'] * (-1 if pos['side'] == 'SELL' else 1)
    close_reason = None
    if pos['side'] == 'BUY':
        if current_price >= pos['take_profit']: close_reason = 'TP'
        elif current_price <= pos['stop_loss']: close_reason = 'SL'
    elif pos['side'] == 'SELL':
        if current_price <= pos['take_profit']: close_reason = 'TP'
        elif current_price >= pos['stop_loss']: close_reason = 'SL'
        
    if close_reason:
        APP_STATE['balances']['free_USDT'] += pos['margin_used'] + pnl
        del APP_STATE['open_positions'][symbol]
        logger.critical(f"🎉 {symbol} - CIERRE POR {close_reason} | PnL: {pnl:.2f} USDT | Nuevo Balance: {APP_STATE['balances']['free_USDT']:.2f}")
    else:
        logger.info(f"    ℹ️ {symbol} - {pos['side']} abierto. SL: {pos['stop_loss']:.4f} | TP: {pos['take_profit']:.4f}")

# ----------------------------------------------------------------------------------
# CICLO PRINCIPAL
# ----------------------------------------------------------------------------------

def run_trading_bot():
    logger.info(f"Bot iniciado en modo: {'DRY_RUN' if DRY_RUN else 'REAL'}.")
    initialize_client(API_KEY, API_SECRET)
    for s in SYMBOL_PAIRS: load_symbol_precision(s)
    initialize_ml_model('BTCUSDT')
    
    while True:
        APP_STATE['status'] = 'RUNNING'
        start_time = time.time()
        
        if 'free_USDT' not in APP_STATE['balances']: APP_STATE['balances']['free_USDT'] = 1000.00
        logger.info(f"✅ Balances (SIM). USDT disponible: {APP_STATE['balances']['free_USDT']:.2f}")
        
        for symbol in SYMBOL_PAIRS:
            logger.info(f"⚙️ Procesando {symbol}...")
            try:
                if len(APP_STATE['open_positions']) >= MAX_OPEN_POSITIONS and symbol not in APP_STATE['open_positions']:
                    logger.info(f"    ⚠️ Límite de {MAX_OPEN_POSITIONS} posiciones alcanzado. Saltando.")
                    continue

                df_short = get_binance_data(symbol, INTERVAL, 2)
                if df_short.empty or len(df_short) < 30:
                    logger.warning(f"⚠️ {symbol} - Data {INTERVAL} insuficiente.")
                    continue
                
                df_indicators = calculate_indicators(df_short)
                current_price = df_indicators.iloc[-1]['close']
                
                if symbol in APP_STATE['open_positions']:
                    manage_positions(symbol, current_price)
                    continue

                long_term_trend = check_long_term_trend(symbol)
                signal, confidence = make_decision(df_indicators)
                display_conf = confidence if signal == 'BUY' else (1 - confidence if signal == 'SELL' else confidence)

                if (signal == 'BUY' and long_term_trend == 'UPTREND') or (signal == 'SELL' and long_term_trend == 'DOWNTREND'):
                    logger.info(f"✨ {symbol} | Señal: {signal} (Conf: {display_conf:.4f}) | Tendencia {LONG_TERM_INTERVAL}: {long_term_trend} | OK")
                    execute_order(symbol, signal, confidence)
                else:
                    logger.info(f"    ℹ️ {symbol} | Señal: {signal} (Conf: {display_conf:.4f}) | Tendencia {LONG_TERM_INTERVAL}: {long_term_trend} | SEÑAL FILTRADA")

            except Exception as e:
                logger.error(f"❌ Error al procesar {symbol}: {e}")
        
        APP_STATE['last_run_utc'] = datetime.utcnow().isoformat()
        APP_STATE['status'] = 'SLEEPING'
        sleep_for = max(0, CYCLE_DELAY_SECONDS - (time.time() - start_time))
        logger.info(f"🟢 Ciclo completado. Abiertas: {len(APP_STATE['open_positions'])}. Durmiendo {int(sleep_for)}s.")
        time.sleep(sleep_for)

# ----------------- FLASK APP Y ARRANQUE -----------------
from flask import Flask, jsonify
app = Flask(__name__)
threading.Thread(target=run_trading_bot, daemon=True).start()

@app.route('/state')
def get_state(): return jsonify(current_state=APP_STATE)

@app.route('/')
def home(): return jsonify(message="Trading Bot Activo. Accede a /state.")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

