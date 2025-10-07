# ----------------------------------------------------------------------------------
# CONFIGURACION E IMPORTACIONES
# ----------------------------------------------------------------------------------
import os
import time
import json
import logging
import math # Importar math para floor/ceil para redondear cantidad
from datetime import datetime, timedelta

# Librerías de Trading
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance.enums import *

# Librerías de Análisis y Machine Learning
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------------------------------
# CONFIGURACION GLOBAL
# ----------------------------------------------------------------------------------

# Claves de la API (Usará las variables de entorno de Render)
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
FUTURES_TESTNET_URL = os.environ.get('FUTURES_TESTNET_URL', 'https://testnet.binancefuture.com')

# Configuración del Bot
SYMBOL_PAIRS = ['TRXUSDT', 'XRPUSDT', 'BTCUSDT']  # Pares a vigilar
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK_PERIOD = "100 days ago UTC" # Periodo para la data histórica de entrenamiento
CYCLE_DELAY_SECONDS = 300 # 5 minutos

# --- Configuración de ML y Trading ---
MODEL_TARGET_CANDLES = 4 # Cuántas velas al futuro intentamos predecir (1 hora)
MODEL_CONFIDENCE_THRESHOLD = 0.55 # Probabilidad mínima para abrir una posición
RISK_PER_TRADE = 0.05 # 5% del capital libre
LEVERAGE = 10 # Apalancamiento fijo
STOP_LOSS_PCT = 0.005 # 0.5% de pérdida
TAKE_PROFIT_PCT = 0.015 # 1.5% de ganancia (Ratio 3:1)

# Estado global del bot y del modelo
APP_STATE = {
    'dry_run': True, # Modo de simulacion por defecto
    'balances': {'free_USDT': 1000.00, 'in_position_USDT': 0.0},
    'open_positions': {}, # {'TRXUSDT': {'entry_price': 0.35, 'side': 'LONG', 'quantity': 1000, 'sl': 0.348, 'tp': 0.355}}
    'model_ready': False,
    'last_run_utc': None,
    'symbol_data': {},
    'symbol_precision': {} # Almacenar precisiones para evitar el error 'stepSize'
}

# Variable para el modelo ML y el escalador, se cargan al inicio
ML_MODEL = None
SCALER = None

# Variable para el cliente de Binance
BINANCE_CLIENT = None

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------
# UTILIDADES Y CONEXIÓN
# ----------------------------------------------------------------------------------

def initialize_client():
    """Inicializa el cliente de Binance para Testnet o Producción."""
    global APP_STATE
    
    dry_run_env = os.environ.get('DRY_RUN', 'true').lower()
    APP_STATE['dry_run'] = (dry_run_env != 'false')
    
    if not API_KEY or not API_SECRET:
        APP_STATE['dry_run'] = True
        logger.warning("Claves API no encontradas. Bot iniciado en modo DRY_RUN forzado (Simulación).")
        
    try:
        # Nota: La API de Python utiliza 'base_url' en el constructor para Testnet,
        # pero para Testnet de FUTUROS, es mejor usar client.futures_base_url.
        client = Client(API_KEY, API_SECRET)
        
        if APP_STATE['dry_run']:
            client.futures_base_url = FUTURES_TESTNET_URL
            # Intentar configurar apalancamiento y modo de margen (solo para Testnet)
            try:
                # Configuramos un par de ejemplo para inicializar la conexión.
                client.futures_change_leverage(symbol='BTCUSDT', leverage=LEVERAGE)
                client.futures_change_margin_type(symbol='BTCUSDT', marginType='ISOLATED')
            except Exception as e:
                # El error APIError(code=-2015) es normal aquí si las claves son incorrectas/Testnet no está configurado
                logger.warning(f"No se pudo configurar apalancamiento/margen en Testnet: {e}")
                
            logger.info(f"✅ Conectado a Binance TESTNET (SIMULACIÓN).")
        else:
            logger.info("✅ Conectado a Binance PRODUCCIÓN (DINERO REAL).")
        
        logger.info(f"Bot iniciado en modo: {'DRY_RUN' if APP_STATE['dry_run'] else 'REAL'}.")
        logger.info(f"Vigilando: {', '.join(SYMBOL_PAIRS)}")
        return client
    
    except Exception as e:
        logger.critical(f"❌ Error al inicializar el cliente de Binance: {e}")
        logger.warning("Forzando modo DRY_RUN debido a error de conexión o credenciales.")
        APP_STATE['dry_run'] = True
        # Devuelve un cliente que probablemente fallará en cualquier llamada real, pero permite que el código siga.
        return Client(API_KEY, API_SECRET)

def load_symbol_precision(client):
    """Obtiene y almacena la precisión de cantidad y precio para cada símbolo."""
    global APP_STATE
    
    try:
        exchange_info = client.futures_exchange_info()
        for symbol_data in exchange_info['symbols']:
            if symbol_data['symbol'] in SYMBOL_PAIRS:
                precision = {
                    'price': symbol_data['pricePrecision'],
                    'quantity': symbol_data['quantityPrecision']
                }
                APP_STATE['symbol_precision'][symbol_data['symbol']] = precision
                
        if APP_STATE['symbol_precision']:
            logger.info(f"✅ Precisiones de {len(APP_STATE['symbol_precision'])} símbolos cargadas.")
            
    except Exception as e:
        logger.warning(f"Error al obtener info de exchange (precisiones): {e}. Usando precisión por defecto (3/4 decimales).")
        # Usar valores predeterminados de seguridad si la API falla
        for symbol in SYMBOL_PAIRS:
            APP_STATE['symbol_precision'][symbol] = {'price': 4, 'quantity': 3}


def get_funding_rate(client, symbol):
    """Obtiene la última tasa de funding rate para un símbolo."""
    try:
        # Se requiere la API de Producción para este endpoint, Testnet no lo tiene.
        # En Testnet, esto fallará, por lo que devolvemos 0.0
        rate_info = client.futures_funding_rate(symbol=symbol)
        return float(rate_info[0]['fundingRate']) if rate_info else 0.0
    except Exception as e:
        # logger.warning(f"Error al obtener Funding Rate para {symbol}: {e}. Asumiendo 0.0")
        return 0.0

def get_binance_data(client, symbol, interval, lookback):
    """Descarga velas históricas y las formatea como DataFrame, con reintentos."""
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            # Descarga la data histórica
            klines = client.futures_historical_klines(symbol, interval, lookback)
            
            # Formatea a DataFrame
            data = pd.DataFrame(klines, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Limpieza y preparación de columnas
            data['Close'] = pd.to_numeric(data['Close'])
            data['Open'] = pd.to_numeric(data['Open'])
            data['High'] = pd.to_numeric(data['High'])
            data['Low'] = pd.to_numeric(data['Low'])
            data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
            
            # --- CORRECCIÓN CLAVE: Eliminar filas duplicadas en el tiempo de apertura ---
            # Esto previene el error 'cannot reindex on an axis with duplicate labels'
            data.drop_duplicates(subset=['open_time'], keep='last', inplace=True)
            
            data.set_index('open_time', inplace=True)
            
            return data[['Open', 'High', 'Low', 'Low', 'Close']].iloc[:-1] # Excluye la vela actual incompleta
            
        except BinanceAPIException as e:
            # Manejar errores temporales de la API, especialmente comunes en Testnet (ej: 500)
            logger.warning(f"Error TEMPORAL (Intento {attempt + 1}/{MAX_RETRIES}) al obtener datos para {symbol}: {e}.")
            if attempt + 1 == MAX_RETRIES:
                logger.critical(f"Error CRÍTICO y persistente al obtener datos para {symbol} después de {MAX_RETRIES} intentos. Saltando el par.")
                return pd.DataFrame()
            
            wait_time = 2 ** attempt # Retroceso exponencial: 1s, 2s, 4s, 8s
            logger.warning(f"Reintentando en {wait_time} segundos.")
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Error inesperado al obtener datos para {symbol}: {e}")
            return pd.DataFrame()

def calculate_indicators(data, funding_rate):
    """Calcula indicadores técnicos (RSI, EMA, MACD, etc.)."""
    
    if data.empty:
        return None

    # RSI (Relative Strength Index) - Periodo 14
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))
    
    # EMA (Exponential Moving Average) - Periodo 50
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Hist'] = data['MACD'] - data['Signal']

    # Funding Rate (Se añade como una columna constante)
    data['FundingRate'] = funding_rate
    
    data.dropna(inplace=True)
    
    if data.empty:
        logger.warning("DataFrame vacío después de calcular indicadores y limpiar NaN.")
        return None
        
    return data

# ----------------------------------------------------------------------------------
# FUNCIONES DE MACHINE LEARNING
# ----------------------------------------------------------------------------------

def create_target_variable(df):
    """Crea la variable objetivo 'Target': 1 si el precio sube en las próximas N velas, 0 si baja/se mantiene."""
    
    df['FutureClose'] = df['Close'].shift(-MODEL_TARGET_CANDLES)
    df['Target'] = np.where(df['FutureClose'] > (df['Close'] * 1.00001), 1, 0)
    
    return df.dropna().drop(columns=['FutureClose'])

def calculate_ml_features(df):
    """Prepara las features X para el modelo ML."""
    
    if df is None or len(df) < 1:
        return None
        
    # Extraer la última fila (la que usaremos para predecir)
    features_row = df.iloc[-1].copy()
    
    # Asegurarse de que las columnas necesarias para X estén calculadas
    # Estas son las que se usan en initialize_ml_model
    df['Distancia_EMA50'] = (df['Close'] - df['EMA50']) / df['Close']
    df['Volatilidad'] = (df['High'] - df['Low']) / df['Close']
    
    # Features X para la última fila
    rsi_val = features_row['RSI']
    distance_to_ema = (features_row['Close'] - features_row['EMA50']) / features_row['Close']
    hist_val = features_row['Hist']
    funding_rate_val = features_row['FundingRate']
    volatility = (features_row['High'] - features_row['Low']) / features_row['Close']
    
    # Crear el array 2D de features (una fila, 5 columnas)
    X = np.array([[rsi_val, distance_to_ema, hist_val, funding_rate_val, volatility]])
    
    return X

def initialize_ml_model(client):
    """Entrena y carga el modelo de Regresión Logística en memoria."""
    global ML_MODEL, SCALER, APP_STATE

    APP_STATE['model_ready'] = False
    logger.warning("Modelo ML no encontrado en memoria. INICIANDO ENTRENAMIENTO...")
    
    symbol = 'BTCUSDT'
    logger.info(f"Buscando datos históricos de {symbol} por {LOOKBACK_PERIOD}...")
    
    # Usar get_binance_data, que ahora tiene reintentos
    data = get_binance_data(client, symbol, INTERVAL, LOOKBACK_PERIOD)
    
    if data.empty:
        logger.critical("❌ NO SE PUDO DESCARGAR DATA HISTÓRICA para entrenamiento. El bot usará la Lógica Manual.")
        return

    funding_rate = get_funding_rate(client, symbol) # Usar el funding rate más reciente
    df = calculate_indicators(data, funding_rate=funding_rate)
    
    if df is None or df.empty:
        logger.critical("❌ Data insuficiente para entrenamiento después de calcular indicadores.")
        return

    df_train = create_target_variable(df)
    
    feature_cols = ['RSI', 'Distancia_EMA50', 'Hist', 'FundingRate', 'Volatilidad']
    
    # Asegurar que estas columnas existan antes de usarlas como features
    df_train['Distancia_EMA50'] = (df_train['Close'] - df_train['EMA50']) / df_train['Close']
    df_train['Volatilidad'] = (df_train['High'] - df_train['Low']) / df_train['Close']
    
    X = df_train[feature_cols].values
    y = df_train['Target'].values

    if len(X) == 0:
        logger.critical("❌ No hay suficientes muestras de datos (X) para el entrenamiento.")
        return
    
    SCALER = StandardScaler()
    X_scaled = SCALER.fit_transform(X)
    
    ML_MODEL = LogisticRegression(solver='liblinear', random_state=42)
    ML_MODEL.fit(X_scaled, y)
    
    accuracy = ML_MODEL.score(X_scaled, y)
    
    APP_STATE['model_ready'] = True
    logger.info(f"✅ Modelo ML cargado en memoria exitosamente. Precisión (entrenamiento): {accuracy:.4f}")

# Función auxiliar para redondear una cantidad a la precisión requerida
def round_quantity_by_precision(quantity, precision):
    """Redondea la cantidad al número de decimales especificado."""
    if precision <= 0:
        return math.floor(quantity)
    # Redondear hacia abajo (floor) para asegurar que no excedamos el margen
    multiplier = 10 ** precision
    return math.floor(quantity * multiplier) / multiplier

def execute_order(client, symbol, signal, close_price, confidence):
    """Gestiona la apertura de una nueva posición si no hay una abierta."""
    global APP_STATE

    if symbol in APP_STATE['open_positions']:
        # Ya hay una posición abierta, no abrir más.
        return
    
    if APP_STATE['balances']['free_USDT'] < 10.0: # Mínimo 10 USDT para operar
        logger.warning(f"{symbol}: Capital libre insuficiente ({APP_STATE['balances']['free_USDT']:.2f} USDT). Saltando orden.")
        return

    if signal not in ['BUY', 'SELL']:
        return # Solo operamos con señales fuertes de compra/venta

    # Cálculo de la cantidad (Risk Management)
    # Capital a arriesgar (5% del capital libre)
    capital_to_use = APP_STATE['balances']['free_USDT'] * RISK_PER_TRADE
    
    # Cuánto USDT se compraría con el apalancamiento (valor nocional)
    nocional_value = capital_to_use * LEVERAGE
    
    # Cantidad (quantity) a comprar/vender (en unidades del activo base, ej: BTC, TRX)
    raw_quantity = nocional_value / close_price
    
    # Redondear la cantidad a la precisión necesaria
    # Usamos la precisión cargada o un valor por defecto seguro (3 decimales)
    qty_precision = APP_STATE['symbol_precision'].get(symbol, {}).get('quantity', 3)
    quantity = round_quantity_by_precision(raw_quantity, qty_precision)
    
    # Aseguramos que la cantidad no sea cero después del redondeo
    if quantity <= 0:
        logger.warning(f"{symbol}: Cantidad calculada ({raw_quantity:.8f}) se redondeó a cero. Saltando orden.")
        return

    # Definir Side y Stop/Take Profit
    if signal == 'BUY':
        side = SIDE_BUY
        sl_price = close_price * (1 - STOP_LOSS_PCT)
        tp_price = close_price * (1 + TAKE_PROFIT_PCT)
    else: # SELL (SHORT)
        side = SIDE_SELL
        sl_price = close_price * (1 + STOP_LOSS_PCT)
        tp_price = close_price * (1 - TAKE_PROFIT_PCT)

    # Redondear precios SL/TP a la precisión de precio
    price_precision = APP_STATE['symbol_precision'].get(symbol, {}).get('price', 4)
    sl_price = round(sl_price, price_precision)
    tp_price = round(tp_price, price_precision)
    
    logger.info(f"💰 {symbol}: Intentando {signal} (CONF: {confidence:.2f}) - {quantity:.{qty_precision}f} unidades. SL: {sl_price:.{price_precision}f} / TP: {tp_price:.{price_precision}f}")

    if APP_STATE['dry_run']:
        # SIMULACIÓN
        
        # Registrar posición abierta en el estado simulado
        APP_STATE['open_positions'][symbol] = {
            'entry_price': close_price,
            'side': signal,
            'quantity': quantity,
            'sl': sl_price,
            'tp': tp_price,
            'margin_used': capital_to_use,
            'entry_time': datetime.utcnow().isoformat()
        }
        APP_STATE['balances']['free_USDT'] -= capital_to_use
        APP_STATE['balances']['in_position_USDT'] += capital_to_use
        logger.info(f"✅ SIMULACIÓN: Posición {signal} abierta exitosamente. Margen usado: {capital_to_use:.2f} USDT. Capital Libre Restante: {APP_STATE['balances']['free_USDT']:.2f}")
        
    else:
        # EJECUCIÓN REAL (No implementado en esta fase por seguridad)
        logger.warning("🔴 EJECUCIÓN REAL: No implementada en esta fase por seguridad. Ejecutando DRY_RUN.")
        # Aquí iría la lógica real de client.futures_create_order()


def manage_positions(symbol, current_price):
    """Verifica SL/TP y cierra posiciones abiertas en el estado simulado."""
    global APP_STATE
    
    if symbol not in APP_STATE['open_positions']:
        return False
        
    position = APP_STATE['open_positions'][symbol]
    
    # Asumimos que la pérdida/ganancia se calcula sobre el margen invertido
    profit_loss = 0
    position_closed = False
    
    # 1. Comprobar Stop-Loss (SL)
    if position['side'] == 'BUY' and current_price <= position['sl']:
        # Si el precio cae al SL, se pierde el capital_to_use (RISK_PER_TRADE)
        profit_loss = -position['margin_used']
        logger.warning(f"🛑 {symbol}: ¡STOP-LOSS HIT! Precio actual ({current_price:.4f}) <= SL ({position['sl']:.4f}). Pérdida simulada: {profit_loss:.2f} USDT.")
        position_closed = True
    elif position['side'] == 'SELL' and current_price >= position['sl']:
        # Si el precio sube al SL (short), se pierde el capital_to_use
        profit_loss = -position['margin_used']
        logger.warning(f"🛑 {symbol}: ¡STOP-LOSS HIT! Precio actual ({current_price:.4f}) >= SL ({position['sl']:.4f}). Pérdida simulada: {profit_loss:.2f} USDT.")
        position_closed = True

    # 2. Comprobar Take-Profit (TP)
    elif position['side'] == 'BUY' and current_price >= position['tp']:
        # Si el precio sube al TP
        gain_pct = TAKE_PROFIT_PCT * LEVERAGE 
        profit_loss = position['margin_used'] * (gain_pct / RISK_PER_TRADE) * RISK_PER_TRADE # Ganancia real sobre el margen
        # Se necesita ajustar el cálculo de P/L de simulación, esto es una aproximación
        # P/L = (Cierre - Entrada) * Cantidad * Palanca
        # Para simulación simple, mantenemos la aproximación basada en el margen
        logger.info(f"🎉 {symbol}: ¡TAKE-PROFIT HIT! Precio actual ({current_price:.4f}) >= TP ({position['tp']:.4f}). Ganancia simulada: {profit_loss:.2f} USDT.")
        position_closed = True
    elif position['side'] == 'SELL' and current_price <= position['tp']:
        # Si el precio cae al TP (short)
        gain_pct = TAKE_PROFIT_PCT * LEVERAGE
        profit_loss = position['margin_used'] * (gain_pct / RISK_PER_TRADE) * RISK_PER_TRADE
        logger.info(f"🎉 {symbol}: ¡TAKE-PROFIT HIT! Precio actual ({current_price:.4f}) <= TP ({position['tp']:.4f}). Ganancia simulada: {profit_loss:.2f} USDT.")
        position_closed = True
        
    
    if position_closed:
        # Cerrar posición (Simulación)
        final_balance = position['margin_used'] + profit_loss
        APP_STATE['balances']['free_USDT'] += final_balance
        APP_STATE['balances']['in_position_USDT'] -= position['margin_used']
        
        # Eliminar posición del estado
        del APP_STATE['open_positions'][symbol]
        logger.info(f"✅ SIMULACIÓN: Posición de {symbol} cerrada. Nuevo capital libre: {APP_STATE['balances']['free_USDT']:.2f} USDT.")
        return True # Posición cerrada
        
    return False # Posición no cerrada

def make_decision(data, symbol, funding_rate):
    """Toma la decisión de trading usando el modelo ML o la lógica manual."""
    global APP_STATE, ML_MODEL, SCALER
    
    signal = 'HOLD'
    close_price = None
    confidence = 0.0

    # 1. LOGICA ML
    # Solo intentar si el modelo está marcado como listo
    if APP_STATE['model_ready'] and ML_MODEL is not None and SCALER is not None:
        try:
            df_with_indicators = calculate_indicators(data, funding_rate)
            
            if df_with_indicators is None:
                raise ValueError("Data frame de indicadores es None.")

            X_live = calculate_ml_features(df_with_indicators)
            
            if X_live is None:
                raise ValueError("No se pudieron calcular las features.")

            X_live_scaled = SCALER.transform(X_live)
            
            probabilities = ML_MODEL.predict_proba(X_live_scaled)[0]
            prob_buy = probabilities[1]
            
            if prob_buy >= MODEL_CONFIDENCE_THRESHOLD:
                signal = 'BUY'
            elif prob_buy <= (1 - MODEL_CONFIDENCE_THRESHOLD):
                signal = 'SELL' 
                
            close_price = df_with_indicators['Close'].iloc[-1]
            confidence = prob_buy
            
            log_message = f"{symbol} | Señal ML: {signal} (Prob: {prob_buy:.4f}) | Precio: {close_price:.4f} | Funding Rate: {funding_rate:.5f}"
            logger.info(log_message)
            
            APP_STATE['symbol_data'][symbol] = {
                'last_signal': signal,
                'last_price': close_price,
                'confidence': confidence,
                'funding_rate': funding_rate,
                'used_ml': True
            }

        except Exception as e:
            # Si el ML falla por cualquier razón, se desactiva y se usa el fallback manual.
            logger.error(f"❌ FALLO DE ML para {symbol}: {e}. Volviendo a la Lógica Manual.")
            APP_STATE['model_ready'] = False 
            # El error 'cannot reindex' debería ser manejado aquí, y el bot no debería caer.
            
    # 2. LOGICA MANUAL (Fallback si el ML no está listo o falló)
    if not APP_STATE['model_ready'] or close_price is None:
        
        df_with_indicators = calculate_indicators(data, funding_rate)

        if df_with_indicators is None:
            logger.critical(f"❌ {symbol}: No se obtuvieron datos suficientes, saltando análisis manual.")
            return 'HOLD', None, 0.0

        latest = df_with_indicators.iloc[-1]
        score = 0.0
        
        # Reglas Manuales (Puntuación)
        if latest['RSI'] < 30: score += 1.5
        elif latest['RSI'] > 70: score -= 1.5

        if latest['Hist'] > 0: score += 1.0
        elif latest['Hist'] < 0: score -= 1.0

        if latest['Close'] > latest['EMA50']: score += 0.5
        elif latest['Close'] < latest['EMA50']: score -= 0.5

        if funding_rate < -0.0001: score += 1.0
        elif funding_rate > 0.0001: score -= 1.0
        
        if score >= 2.0: signal = 'BUY'
        elif score <= -2.0: signal = 'SELL'
            
        close_price = latest['Close']
        confidence = abs(score) # Usamos el valor absoluto del puntaje como "confianza" para el riesgo
        
        log_message = f"{symbol} | Señal Manual: {signal} (Puntaje: {score:.1f}) | Precio: {close_price:.4f} | Funding Rate: {funding_rate:.5f}"
        logger.info(log_message)
        
        APP_STATE['symbol_data'][symbol] = {
            'last_signal': signal,
            'last_price': close_price,
            'confidence': confidence,
            'funding_rate': funding_rate,
            'used_ml': False
        }
    
    return signal, close_price, confidence

# ----------------------------------------------------------------------------------
# BUCLE PRINCIPAL DE EJECUCIÓN
# ----------------------------------------------------------------------------------

def run_trading_bot():
    """Bucle principal del bot de trading."""
    global BINANCE_CLIENT

    # 1. COMPROBAR CLIENTE (Debe estar inicializado)
    if BINANCE_CLIENT is None:
        logger.critical("Error: El cliente de Binance no fue inicializado correctamente al inicio. Terminando hilo de trading.")
        return

    while True:
        try:
            logger.info(f"--- Ciclo de trading iniciado. ---")

            # 2. INTENTO DE INICIALIZACIÓN DE ML (Se ejecuta si el modelo no está listo)
            # Esto permite que el modelo se entrene en el fondo sin bloquear el arranque web.
            if not APP_STATE['model_ready']:
                # El error de reindexación ocurre AQUI. Usamos un try/except en el bucle
                # principal para manejar errores que afectan el ciclo completo.
                try:
                    initialize_ml_model(BINANCE_CLIENT)
                except Exception as e:
                    logger.critical(f"Error CRÍTICO al intentar inicializar el modelo ML: {e}. El bot continuará usando solo la lógica manual.")
                    APP_STATE['model_ready'] = False
            
            logger.info(f"Balances de la cuenta ({'SIMULACIÓN' if APP_STATE['dry_run'] else 'REAL'}). USDT libre: {APP_STATE['balances']['free_USDT']:.2f}")

            # 3. Iterar sobre todos los pares
            for symbol in SYMBOL_PAIRS:
                
                # Envoltura de try/except para aislar errores por símbolo y no detener el bot
                try:
                    logger.info(f"Procesando {symbol}...")
                    data = get_binance_data(BINANCE_CLIENT, symbol, INTERVAL, LOOKBACK_PERIOD)
                    funding_rate = get_funding_rate(BINANCE_CLIENT, symbol)
                    
                    if data.empty:
                        logger.error(f"Error: No se pudieron obtener datos para {symbol}, saltando análisis.")
                        continue

                    # Tomar decisión (ML o Manual)
                    signal, close_price, confidence = make_decision(data, symbol, funding_rate)

                    if close_price is None:
                        continue # No data, no decision

                    # 4. GESTIÓN DE POSICIONES ABIERTAS (Stop Loss / Take Profit)
                    position_closed = manage_positions(symbol, close_price)
                    
                    # 5. EJECUCIÓN DE NUEVAS ÓRDENES (Solo si no se cerró una posición en este ciclo)
                    if not position_closed:
                        # Usamos un umbral de confianza/puntaje mínimo (se usa el mismo que MODEL_CONFIDENCE_THRESHOLD)
                        min_confidence = MODEL_CONFIDENCE_THRESHOLD if APP_STATE['model_ready'] else 2.0
                        
                        if confidence >= min_confidence:
                            # Si la señal es fuerte y no hay posición abierta, ejecutar orden
                            execute_order(BINANCE_CLIENT, symbol, signal, close_price, confidence)
                        elif symbol in APP_STATE['open_positions']:
                            # Si ya hay una posición, solo monitorear (no hay señal de reversión fuerte)
                            logger.info(f"{symbol}: Posición activa. Esperando SL/TP.")
                
                except BinanceAPIException as e:
                    # Captura el error 'stepSize' o 'LOT_SIZE' y permite que el bot continúe con el siguiente par.
                    logger.error(f"❌ Error de API de Binance al procesar {symbol} (Probablemente precisión/stepSize): {e}. El bot continúa con el siguiente par.")
                except Exception as e:
                    logger.error(f"❌ Error inesperado al procesar {symbol}: {e}. El bot continúa con el siguiente par.")

            
            logger.info(f"--- Ciclo completado. Abiertas: {len(APP_STATE['open_positions'])}. Durmiendo por {CYCLE_DELAY_SECONDS} segundos. ---")
            APP_STATE['last_run_utc'] = datetime.utcnow().isoformat()
            time.sleep(CYCLE_DELAY_SECONDS)

        except Exception as e:
            # Captura errores que no son específicos de un símbolo (ej: fallo de conexión general)
            # El error de reindexación será capturado por el try/except del initialize_ml_model
            logger.critical(f"Error CRÍTICO e inesperado en el bucle principal: {e}. Reiniciando en 30 segundos.")
            time.sleep(30)

# ----------------------------------------------------------------------------------
# ENDPOINTS WEB (Requerido por Render para mantener el bot activo)
# ----------------------------------------------------------------------------------
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/state', methods=['GET'])
def get_state():
    """Endpoint para obtener el estado actual del bot (usado por el monitor de ping)."""
    
    state_data = {
        'status': 'RUNNING',
        'dry_run_mode': APP_STATE['dry_run'],
        'model_ml_ready': APP_STATE['model_ready'],
        'last_run_utc': APP_STATE['last_run_utc'],
        'balances': {
            'free_USDT': APP_STATE['balances']['free_USDT'],
            'in_position_USDT': APP_STATE['balances']['in_position_USDT'],
        },
        'symbol_data': APP_STATE['symbol_data'],
        'open_positions': APP_STATE['open_positions'],
        'metrics': {
            'LEVERAGE': LEVERAGE,
            'STOP_LOSS_PCT': STOP_LOSS_PCT,
            'TAKE_PROFIT_PCT': TAKE_PROFIT_PCT,
            'RISK_PER_TRADE': RISK_PER_TRADE,
        }
    }
    return jsonify(current_state=state_data)

@app.route('/', methods=['GET'])
def home():
    """Endpoint de bienvenida."""
    return jsonify(message="Trading Bot Activo. Accede a /state para ver el estado.")

# ----------------------------------------------------------------------------------
# ARRANQUE Y SETUP INICIAL (Asegura que el thread de trading inicie en Gunicorn)
# ----------------------------------------------------------------------------------
import threading

# 1. Inicializar el cliente de Binance tan pronto como el módulo se cargue
BINANCE_CLIENT = initialize_client()

# 2. Cargar precisiones (depende del cliente)
if BINANCE_CLIENT and not APP_STATE['symbol_precision']:
    load_symbol_precision(BINANCE_CLIENT)

# 3. INICIAR EL THREAD DE TRADING
# Se inicia aquí, fuera del bloque if __name__ == '__main__', para que Gunicorn lo ejecute.
logger.info("⚙️ Iniciando Thread de Trading en segundo plano...")
trading_thread = threading.Thread(target=run_trading_bot)
trading_thread.daemon = True # Hace que el hilo se detenga cuando Gunicorn se detenga.
trading_thread.start()
    
# El proceso principal (Gunicorn) usará la variable 'app' para servir las peticiones web.
if __name__ == '__main__':
    # Esto solo se usa si ejecutas python trading_bot.py localmente, no en Render.
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
