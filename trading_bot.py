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
# NOTA: Estas claves son de Producción REAL y se usan SOLO para descargar data histórica (entrenamiento ML).
API_KEY = os.environ.get('BINANCE_API_KEY', 'TU_API_KEY_AQUI')
API_SECRET = os.environ.get('BINANCE_API_SECRET', 'TU_SECRET_KEY_AQUI')

# CONFIGURACIÓN DEL BOT Y RIESGO
CYCLE_DELAY_SECONDS = int(os.environ.get('CYCLE_DELAY_SECONDS', 1200)) # 20 minutos para reducir bloqueo de API
FUTURES_TESTNET_URL = os.environ.get('FUTURES_TESTNET_URL', 'https://testnet.binancefuture.com')

# Control de Modo y Entorno
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() in ('1', 'true', 'yes')
USE_TESTNET = os.environ.get('USE_TESTNET', 'true').lower() in ('1', 'true', 'yes') # ¡Usamos Testnet para simulación!

# Parámetros de Trading
SYMBOL_PAIRS = os.environ.get('SYMBOL_PAIRS', 'TRXUSDT,BTCUSDT,XRPUSDT').split(',') # Pares a vigilar
INTERVAL = os.environ.get('INTERVAL', '15m')
LEVERAGE = int(os.environ.get('LEVERAGE', 10))
MIN_ORDER_USD = float(os.environ.get('MIN_ORDER_USD', 10.5))

# Parámetros de Riesgo
TP_FACTOR = float(os.environ.get('TP_FACTOR', 0.045)) # Take Profit: 4.5% de movimiento (para R:R 3:1)
SL_FACTOR = float(os.environ.get('SL_FACTOR', 0.015)) # Stop Loss: 1.5% de movimiento
RISK_PER_TRADE = float(os.environ.get('RISK_PER_TRADE', 0.075)) # 7.5% del capital libre como margen

# Parámetros de Machine Learning
MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', 0.60)) # Min. 60% de confianza para ejecutar
DAYS_FOR_TRAINING = int(os.environ.get('DAYS_FOR_TRAINING', 60)) # AUMENTADO A 60 DÍAS PARA MEJOR ENTRENAMIENTO

# ---------------- ESTADO GLOBAL Y CLIENTES ----------------
APP_STATE = {
    'dry_run': DRY_RUN,
    'status': 'STARTING',
    'last_run_utc': None,
    'model_ml_ready': False,
    'balances': {'free_USDT': 1000.00 if DRY_RUN else 0.00, 'free_BNB': 0.00},
    'symbol_data': {}, # { 'BTCUSDT': { 'signal': 'HOLD', 'prob': 0.50 } }
    'open_positions': {}, # { 'BTCUSDT': { 'entry_price': 10000, 'margin_used': 50 } }
    'symbol_precision': {},
}

# Clientes de Binance (Inicialización global, las URLs se configuran después)
CLIENT = None
PUBLIC_CLIENT = None

# Variables de Estado de ML
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

    # Cliente público (no autenticado, para datos de mercado y reglas)
    PUBLIC_CLIENT = Client()

    # Cliente autenticado (para balances, órdenes y data histórica)
    CLIENT = Client(api_key, api_secret)
    CLIENT.base_url = 'https://api.binance.com/api' # URL de producción para data histórica

    # Configurar la URL de futuros para el trading simulado/real
    if USE_TESTNET:
        # Esto solo se usa para enviar órdenes simuladas/reales
        CLIENT.futures_base_url = FUTURES_TESTNET_URL
        logger.info(f"✅ Conectado a Binance TESTNET (SIMULACIÓN).")
    else:
        # Se asume que en modo REAL, se usa la URL estándar de producción para Futuros
        CLIENT.futures_base_url = 'https://fapi.binance.com'
        logger.info(f"✅ Conectado a Binance PRODUCCIÓN (DINERO REAL).")

    # Sincronización de tiempo (CRÍTICO para evitar error -1021)
    CLIENT.timestamp_offset = 0
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

        precision = {
            'quantity_precision': symbol_info['quantityPrecision'],
            'price_precision': symbol_info['pricePrecision']
        }
        APP_STATE['symbol_precision'][symbol] = precision
        return True
    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO: No se pudo obtener la info de precisión para {symbol}: {e}")
        return False

# ----------------------------------------------------------------------------------
# FUNCIONES DE DATOS Y ML
# ----------------------------------------------------------------------------------

def get_funding_rate(symbol):
    """Obtiene la última funding rate para un símbolo."""
    try:
        rate_info = PUBLIC_CLIENT.futures_funding_rate(symbol=symbol)
        return float(rate_info['lastFundingRate'])
    except Exception:
        return 0.0

def get_binance_data(symbol, interval, lookback_days):
    """
    Descarga data histórica para entrenamiento ML.
    Utiliza CLIENT (autenticado) para asegurar el acceso a data histórica.
    """
    try:
        # Calcular fecha de inicio de la descarga
        start_str = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%d %b, %Y")

        # Usar get_historical_klines para datos a largo plazo
        klines = CLIENT.futures_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str
        )
        if not klines:
            logger.error(f"❌ No se recibieron Klines para {symbol} en el intervalo {interval}.")
            return pd.DataFrame()

        data = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
        ])

        # 1. Convertir la columna de tiempo a formato datetime.
        data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')

        # 2. Seleccionar las columnas que nos interesan.
        data = data[["open_time", "open", "high", "low", "close"]]

        # 3. Convertir a float SOLO las columnas numéricas.
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            data[col] = data[col].astype(float)

        # CRÍTICO: Limpieza de duplicados de índice
        data.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
        data.set_index('open_time', inplace=True)

        return data
    except Exception as e:
        logger.error(f"❌ Fallo al descargar data histórica de {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calcula las features de ML: RSI, MACD, EMA, Volatilidad y nuevos indicadores."""
    df_new = df.copy()

    # 1. RSI
    delta = df_new['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_new['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD
    ema_fast = df_new['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df_new['close'].ewm(span=26, adjust=False).mean()
    df_new['MACD'] = ema_fast - ema_slow
    df_new['MACD_Signal'] = df_new['MACD'].ewm(span=9, adjust=False).mean()

    # 3. EMA Crossover
    df_new['EMA_9'] = df_new['close'].ewm(span=9, adjust=False).mean()
    df_new['EMA_21'] = df_new['close'].ewm(span=21, adjust=False).mean()
    df_new['EMA_Diff'] = df_new['EMA_9'] - df_new['EMA_21']

    # 4. Volatilidad (ATR Simple)
    df_new['TR'] = np.maximum(df_new['high'] - df_new['low'],
                                  np.maximum(abs(df_new['high'] - df_new['close'].shift(1)),
                                             abs(df_new['low'] - df_new['close'].shift(1))))
    df_new['ATR'] = df_new['TR'].rolling(window=14).mean()

    # --- INICIO DE NUEVOS INDICADORES ---
    # 5. Bandas de Bollinger
    rolling_mean_20 = df_new['close'].rolling(window=20).mean()
    rolling_std_20 = df_new['close'].rolling(window=20).std()
    df_new['bb_upper'] = rolling_mean_20 + (rolling_std_20 * 2)
    df_new['bb_lower'] = rolling_mean_20 - (rolling_std_20 * 2)
    df_new['bb_width'] = (df_new['bb_upper'] - df_new['bb_lower']) / rolling_mean_20 # Ancho de banda normalizado

    # 6. Oscilador Estocástico
    low_14 = df_new['low'].rolling(window=14).min()
    high_14 = df_new['high'].rolling(window=14).max()
    df_new['stoch_k'] = 100 * ((df_new['close'] - low_14) / (high_14 - low_14))
    df_new['stoch_d'] = df_new['stoch_k'].rolling(window=3).mean()
    # --- FIN DE NUEVOS INDICADORES ---

    return df_new.dropna()

def create_target(df):
    """
    Crea la variable objetivo (Target) para el ML:
    Target = 1 si el precio sube un 1% en las próximas 4 velas (1 hora), 0 en otro caso.
    """
    # 4 velas hacia adelante (4 * 15m = 1 hora)
    future_close = df['close'].shift(-4)
    price_change = (future_close - df['close']) / df['close']

    # Etiqueta: 1 si sube más de 1%, 0 si no.
    df['Target'] = np.where(price_change > 0.01, 1, 0)
    
    # Eliminar las últimas 4 filas ya que el Target será NaN
    return df.iloc[:-4].dropna()

def calculate_ml_features(df):
    """Prepara las features X para el modelo ML."""
    # Features a usar (incluyendo las nuevas):
    X = df[['RSI', 'MACD', 'MACD_Signal', 'EMA_Diff', 'ATR', 'bb_width', 'stoch_k', 'stoch_d']]
    
    # Agrega el funding rate (que se añade al df en el ciclo principal)
    if 'funding_rate' in df.columns:
        X['Funding_Rate'] = df['funding_rate']
    
    return X.dropna()

def initialize_ml_model(symbol):
    """
    Entrena el modelo XGBoost y lo guarda en memoria.
    Esta función se ejecuta al inicio del bot.
    """
    global ML_MODEL, ML_SCALER
    logger.warning("Modelo ML no encontrado en memoria. INICIANDO ENTRENAMIENTO...")
    
    try:
        # 1. Obtener datos históricos de PRODUCCIÓN
        logger.info(f"Buscando datos históricos de {symbol} por {DAYS_FOR_TRAINING} days ago UTC...")
        df = get_binance_data(symbol, INTERVAL, DAYS_FOR_TRAINING)
        if df.empty:
            logger.error("❌ ERROR: Data histórica vacía. Entrenando fallido.")
            return

        # 2. Calcular indicadores y Target
        df = calculate_indicators(df)
        df_target = create_target(df)
        
        # 3. Preparar Features (X) y Target (y)
        X = calculate_ml_features(df_target)
        y = df_target['Target']
        
        # CRÍTICO: Limpieza forzada del índice antes de entrenamiento
        duplicates_in_index = X.index.duplicated().sum()
        if duplicates_in_index > 0:
            X = X[~X.index.duplicated(keep='first')]
            y = y.loc[X.index] # Alinear y con el índice limpio de X
            logger.warning(f"FORZANDO LIMPIEZA: Se eliminaron {duplicates_in_index} índices duplicados del DataFrame de entrenamiento.")

        # 4. Escalado (Normalización de los datos)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Entrenar el modelo XGBoost con parámetros mejorados
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # 6. Evaluación
        accuracy = model.score(X_test_scaled, y_test)
        
        # 7. Guardar en memoria global
        ML_MODEL = model
        ML_SCALER = scaler
        
        logger.info(f"✅ Modelo ML cargado en memoria exitosamente. Precisión (entrenamiento): {accuracy:.4f}")
    
    except Exception as e:
        logger.critical(f"❌ Error CRÍTICO durante el entrenamiento del modelo ML: {e}. El bot usará la Lógica Manual.")


# ----------------------------------------------------------------------------------
# FUNCIONES DE TRADING Y EJECUCIÓN
# ----------------------------------------------------------------------------------

def round_quantity_by_precision(quantity, precision):
    """Redondea la cantidad a la precisión requerida por Binance."""
    if not isinstance(precision, int) or precision < 0:
        return quantity # Fallback
    
    factor = 10**precision
    return math.floor(quantity * factor) / factor

def round_price_by_precision(price, precision):
    """Redondea el precio a la precisión requerida por Binance."""
    if not isinstance(precision, int) or precision < 0:
        return price # Fallback
        
    return round(price, precision)

def get_current_price(symbol):
    """Obtiene el precio actual de un símbolo usando el cliente público."""
    try:
        ticker = PUBLIC_CLIENT.futures_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception:
        return None

def update_balances(is_simulation=False):
    """Actualiza el balance de USDT desde Binance o lo simula."""
    global APP_STATE
    
    if is_simulation:
        if 'free_USDT' not in APP_STATE['balances']:
            APP_STATE['balances']['free_USDT'] = 1000.00 # Balance inicial simulado
        
        # No se necesita más actualización en simulación
        logger.info(f"✅ Balances de la cuenta actualizados (SIMULACIÓN). USDT disponible: {APP_STATE['balances']['free_USDT']:.2f}")
        return True

    # Lógica para balance real (no implementada, pero se conectaría aquí)
    return False

def make_decision(symbol, df_data):
    """
    Genera la señal de trading usando el modelo ML o la lógica manual.
    Devuelve (signal, confidence/score).
    """
    global ML_MODEL, ML_SCALER

    # ---- 1. LÓGICA DE ML (si el modelo está cargado) ----
    if ML_MODEL and ML_SCALER:
        try:
            # Crear features del último punto de datos (sin Target)
            X_live = calculate_ml_features(df_data.tail(1))
            
            # Chequeo crítico: Si la data está incompleta
            if X_live.empty:
                 return ('HOLD', 0.50, 'ML')

            X_live_scaled = ML_SCALER.transform(X_live)
            
            # Obtener la probabilidad (0=DOWN, 1=UP)
            proba = ML_MODEL.predict_proba(X_live_scaled)[0]
            prob_buy = proba[1]
            
            # Decisión basada en la probabilidad de compra
            if prob_buy >= MODEL_CONFIDENCE_THRESHOLD:
                signal = 'BUY'
            elif prob_buy <= (1 - MODEL_CONFIDENCE_THRESHOLD): # Si prob_down >= 0.60
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return (signal, prob_buy, 'ML')
        except Exception as e:
            logger.error(f"❌ Error durante la predicción ML: {e}. Volviendo a Lógica Manual.")
            
    # ---- 2. LÓGICA MANUAL (FALLBACK) ----
    df_last = df_data.iloc[-1]
    score = 0
    
    # Criterios de Compra (Puntaje positivo)
    if df_last['RSI'] < 30: score += 1.5
    if df_last['MACD'] > df_last['MACD_Signal']: score += 1.0
    if df_last['EMA_9'] > df_last['EMA_21']: score += 0.5

    # Criterios de Venta (Puntaje negativo)
    if df_last['RSI'] > 70: score -= 1.5
    if df_last['MACD'] < df_last['MACD_Signal']: score -= 1.0
    if df_last['EMA_9'] < df_last['EMA_21']: score -= 0.5
    
    if score >= 2.0: signal = 'BUY'
    elif score <= -2.0: signal = 'SELL'
    else: signal = 'HOLD'
    
    return (signal, score, 'Manual')


def execute_order(symbol, side, confidence):
    """Abre una posición LONG/SHORT simulada o real."""
    global APP_STATE

    try:
        current_price = get_current_price(symbol)
        if current_price is None:
            logger.warning(f"⚠️ {symbol} - No se pudo obtener el precio actual. Orden cancelada.")
            return

        # 1. Determinar el tamaño del margen y la cantidad a operar
        usdt_free = APP_STATE['balances']['free_USDT']
        usdt_margin = usdt_free * RISK_PER_TRADE
        
        # Validar el margen
        if usdt_margin < MIN_ORDER_USD:
             logger.warning(f"⚠️ {symbol} - Margen ({usdt_margin:.2f} USDT) es menor al mínimo de orden {MIN_ORDER_USD}. Orden cancelada.")
             return
             
        # Margen Mínimo de Binance para un Valor Nocional de $10 USD (con 10x de apalancamiento) es $1 USD.
        if usdt_margin < 1.0:
            usdt_margin = 1.0 # Usamos el mínimo de margen requerido.

        # Valor nocional (tamaño total de la posición)
        notional_value = usdt_margin * LEVERAGE
        
        # Cantidad de la moneda (ej. cantidad de TRX)
        quantity = notional_value / current_price
        
        # 2. Redondeo de Precisión (CRÍTICO)
        precision = APP_STATE['symbol_precision'].get(symbol, {}).get('quantity_precision', 0)
        quantity = round_quantity_by_precision(quantity, precision)
        
        if quantity == 0:
            logger.warning(f"⚠️ {symbol} - La cantidad calculada es 0 después del redondeo. Orden cancelada.")
            return

        # 3. Calcular SL y TP (Se basan en el movimiento del precio, no en el margen)
        tp_move = TP_FACTOR * current_price
        sl_move = SL_FACTOR * current_price
        
        if side == 'BUY':
            tp_price = current_price + tp_move
            sl_price = current_price - sl_move
        else: # SELL (Short)
            tp_price = current_price - tp_move
            sl_price = current_price + sl_move
            
        # Redondeo de precios
        price_precision = APP_STATE['symbol_precision'].get(symbol, {}).get('price_precision', 0)
        tp_price = round_price_by_precision(tp_price, price_precision)
        sl_price = round_price_by_precision(sl_price, price_precision)
        
        # 4. Ejecución (Real o Simulada)
        if APP_STATE['dry_run']:
            # Simulación: Actualizar estado de simulación
            APP_STATE['balances']['free_USDT'] -= usdt_margin
            
            APP_STATE['open_positions'][symbol] = {
                'side': side,
                'entry_price': current_price,
                'entry_time': datetime.utcnow().isoformat(),
                'quantity': quantity,
                'margin_used': usdt_margin,
                'stop_loss': sl_price,
                'take_profit': tp_price
            }
            logger.info(f"✅ SIMULACIÓN: Posición {side} abierta exitosamente. Margen usado: {usdt_margin:.2f} USDT. Capital Libre Restante: {APP_STATE['balances']['free_USDT']:.2f}")

        else:
            # Lógica de Orden Real (Se descomentaría si DRY_RUN fuera false)
            # client.futures_create_order(...)
            logger.warning("🚨 MODO REAL ACTIVO: Se enviaría una orden de mercado real a Binance.")
        
    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO al ejecutar orden de {symbol}: {e}")

def manage_positions(symbol, current_price):
    """Gestiona posiciones abiertas, cerrando por SL/TP simulado o real."""
    global APP_STATE

    position = APP_STATE['open_positions'][symbol]
    
    # 1. Cálculo del PnL y cierre (simulación)
    if APP_STATE['dry_run']:
        
        # 1.1. Calcular Ganancia/Pérdida Simulado
        pnl = (current_price - position['entry_price']) * position['quantity']
        if position['side'] == 'SELL':
            pnl *= -1 # Se invierte para shorts

        # 1.2. Verificar SL/TP
        close_reason = None
        
        if position['side'] == 'BUY' and current_price >= position['take_profit']:
            close_reason = 'TAKE-PROFIT'
        elif position['side'] == 'BUY' and current_price <= position['stop_loss']:
            close_reason = 'STOP-LOSS'
        elif position['side'] == 'SELL' and current_price <= position['take_profit']:
            close_reason = 'TAKE-PROFIT'
        elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
            close_reason = 'STOP-LOSS'
            
        if close_reason:
            # 1.3. Ejecutar cierre y actualizar balances
            final_usdt_value = position['margin_used'] + pnl
            APP_STATE['balances']['free_USDT'] += final_usdt_value
            
            del APP_STATE['open_positions'][symbol]
            
            logger.critical(f"🎉 {symbol} - CIERRE POR {close_reason} | PnL Simulado: {pnl:.2f} USDT. Nuevo Balance: {APP_STATE['balances']['free_USDT']:.2f}")
            return
            
    # 2. Monitoreo Activo
    logger.info(f"    ℹ️ {symbol} - Posición {position['side']} abierta. SL: {position['stop_loss']:.4f} | TP: {position['take_profit']:.4f}")
    
# ----------------------------------------------------------------------------------
# CICLO PRINCIPAL Y ARRANQUE
# ----------------------------------------------------------------------------------

def run_trading_bot():
    """Bucle principal que se ejecuta continuamente en un thread."""
    global APP_STATE, ML_MODEL, ML_SCALER

    if APP_STATE['dry_run']:
        logger.info("Bot iniciado en modo: DRY_RUN.")
    else:
        logger.info("Bot iniciado en modo: REAL.")

    # 1. Inicialización de ML (solo se intenta una vez, luego se usa el resultado)
    if not ML_MODEL:
        initialize_ml_model('BTCUSDT') # Usamos BTCUSDT para el entrenamiento base

    # 2. Bucle principal
    while True:
        APP_STATE['status'] = 'RUNNING'
        start_time = time.time()
        
        # 2.1. Actualizar balances de simulación
        update_balances(is_simulation=True)

        # 2.2. Procesar cada símbolo
        for symbol in SYMBOL_PAIRS:
            
            logger.info(f"⚙️ Procesando {symbol}...")
            
            try:
                # Obtener data histórica (se usa el cliente público y la nueva lógica de estabilidad)
                df_data = get_binance_data(symbol, INTERVAL, 2) # Solo necesitamos 2 días para el cálculo de indicadores en vivo
                
                if df_data.empty or len(df_data) < 30: # Mínimo necesario para los nuevos indicadores
                    logger.warning(f"⚠️ {symbol} - Data insuficiente para indicadores.")
                    continue

                # 2.2.1. Calcular indicadores
                df_indicators = calculate_indicators(df_data)
                
                if df_indicators.empty:
                    logger.warning(f"⚠️ {symbol} - DataFrame vacío después de calcular indicadores.")
                    continue

                df_live = df_indicators.tail(1)
                
                current_price = df_live['close'].iloc[0]
                
                # 2.2.2. Gestionar posición abierta
                if symbol in APP_STATE['open_positions']:
                    manage_positions(symbol, current_price)
                    continue

                # 2.2.3. Generar señal (ML o Manual)
                signal, confidence, mode = make_decision(symbol, df_indicators)
                
                # 2.2.4. Ejecución
                if signal in ['BUY', 'SELL'] and confidence >= MODEL_CONFIDENCE_THRESHOLD:
                    logger.info(f"✨ {symbol} | Señal {mode}: {signal} (Conf: {confidence:.4f}) | Precio: {current_price:.4f}")
                    execute_order(symbol, signal, confidence)
                else:
                    logger.info(f"    ℹ️ {symbol} | Señal {mode}: HOLD (Conf: {confidence:.4f}) | Precio: {current_price:.4f}")

            except Exception as e:
                logger.error(f"❌ Error al procesar {symbol}: {e}")

        # 2.3. Finalizar ciclo y dormir
        APP_STATE['last_run_utc'] = datetime.utcnow().isoformat()
        APP_STATE['status'] = 'SLEEPING'
        
        elapsed_time = time.time() - start_time
        sleep_for = max(0, CYCLE_DELAY_SECONDS - elapsed_time)
        
        logger.info(f"🟢 Ciclo completado. Abiertas: {len(APP_STATE['open_positions'])}. Durmiendo por {int(sleep_for)} segundos.")
        time.sleep(sleep_for)


# ----------------------------------------------------------------------------------
# ARRANQUE Y FLASK
# ----------------------------------------------------------------------------------

def run_trading_bot_thread():
    """Inicializa el cliente de Binance y corre el thread de trading."""
    try:
        initialize_client(API_KEY, API_SECRET)

        # Cargar precisiones al inicio
        for symbol in SYMBOL_PAIRS:
            load_symbol_precision(symbol)

        # Iniciar el thread de trading
        trading_thread = threading.Thread(target=run_trading_bot, daemon=True)
        trading_thread.start()

    except Exception as e:
        logger.critical(f"❌ FALLO CRÍTICO EN ARRANQUE: {e}")
        # Si el cliente falla al iniciar, no se inicia el thread, pero Flask sigue activo

# ----------------- FLASK APP -----------------
from flask import Flask, jsonify
app = Flask(__name__)

# Ejecutar el setup del bot en un thread
run_trading_bot_thread()

@app.route('/state', methods=['GET'])
def get_state():
    """Ruta para ver el estado de trading y el historial."""
    state_data = {
        'status': APP_STATE['status'],
        'dry_run_mode': APP_STATE['dry_run'],
        'model_ml_ready': APP_STATE['model_ml_ready'],
        'last_run_utc': APP_STATE['last_run_utc'],
        'balances': APP_STATE['balances'],
        'symbol_data': APP_STATE['symbol_data'],
        'open_positions': APP_STATE['open_positions']
    }
    return jsonify(current_state=state_data)

@app.route('/', methods=['GET'])
def home():
    """Endpoint de bienvenida."""
    return jsonify(message="Trading Bot Activo. Accede a /state para ver el estado.")

if __name__ == '__main__':
    # Gunicorn ejecutará 'gunicorn trading_bot:app', usando la variable 'app'
    # Esta parte solo corre si ejecutas el archivo directamente (p.ej. python trading_bot.py)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

