# ----------------------------------------------------------------------------------
# CONFIGURACION E IMPORTACIONES
# ----------------------------------------------------------------------------------
import os
import time
import json
import logging
from datetime import datetime, timedelta

# Librerías de Trading
from binance.client import Client
from binance.exceptions import BinanceAPIException
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
MODEL_TARGET_CANDLES = 4 # Cuántas velas al futuro intentamos predecir (1 hora)
MODEL_CONFIDENCE_THRESHOLD = 0.55 # Probabilidad mínima para abrir una posición

# Estado global del bot y del modelo
APP_STATE = {
    'dry_run': True, # Modo de simulacion por defecto
    'balances': {'free_USDT': 1000.00, 'in_position_USDT': 0.0},
    'open_positions': {},
    'model_ready': False,
    'last_run_utc': None,
    'symbol_data': {}
}

# Variable para el modelo ML y el escalador, se cargan al inicio
ML_MODEL = None
SCALER = None

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------
# UTILIDADES Y CONEXIÓN
# ----------------------------------------------------------------------------------

def initialize_client():
    """Inicializa el cliente de Binance para Testnet o Producción."""
    global APP_STATE
    
    # 1. Determinar el modo de ejecución
    # Si la clave 'DRY_RUN' está presente en el entorno y es 'false' (o no está), asumimos REAL
    dry_run_env = os.environ.get('DRY_RUN', 'true').lower()
    APP_STATE['dry_run'] = (dry_run_env != 'false')
    
    # Si no hay claves, forzamos DRY_RUN para evitar errores
    if not API_KEY or not API_SECRET:
        APP_STATE['dry_run'] = True
        logger.warning("Claves API no encontradas. Bot iniciado en modo DRY_RUN forzado (Simulación).")
        
    try:
        # Inicializa el cliente principal con las claves de Producción (necesario para data histórica)
        client = Client(API_KEY, API_SECRET)
        
        if APP_STATE['dry_run']:
            # Para Testnet (simulación de órdenes de trading)
            client.futures_base_url = FUTURES_TESTNET_URL
            logger.info(f"✅ Conectado a Binance TESTNET (SIMULACIÓN). Estado de la cuenta: {client.get_account_status().get('data', {}).get('state')}")
        else:
            # Para Producción (órdenes de trading reales)
            logger.info("✅ Conectado a Binance PRODUCCIÓN (DINERO REAL). Estado de la cuenta: True")
        
        logger.info(f"Bot iniciado en modo: {'DRY_RUN' if APP_STATE['dry_run'] else 'REAL'}.")
        logger.info(f"Vigilando: {', '.join(SYMBOL_PAIRS)}")
        return client
    
    except Exception as e:
        logger.critical(f"❌ Error al inicializar el cliente de Binance: {e}")
        logger.warning("Forzando modo DRY_RUN debido a error de conexión o credenciales.")
        APP_STATE['dry_run'] = True
        return Client(API_KEY, API_SECRET) # Intenta inicializar de nuevo sin Testnet URL

# ----------------------------------------------------------------------------------
# FUNCIONES DE DATOS Y INDICADORES
# ----------------------------------------------------------------------------------

def get_funding_rate(client, symbol):
    """Obtiene la última tasa de funding rate para un símbolo."""
    try:
        # Se requiere la API de Producción para este endpoint, Testnet no lo tiene.
        rate_info = client.futures_funding_rate(symbol=symbol)
        return float(rate_info[0]['fundingRate']) if rate_info else 0.0
    except Exception as e:
        # Esto sucede en Testnet o si la clave de Prod es inválida. Asumimos 0.
        logger.warning(f"Error al obtener Funding Rate para {symbol}: {e}. Asumiendo 0.0")
        return 0.0

def get_binance_data(client, symbol, interval, lookback):
    """Descarga velas históricas y las formatea como DataFrame."""
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
        data.set_index('open_time', inplace=True)
        
        return data[['Open', 'High', 'Low', 'Close']].iloc[:-1] # Excluye la vela actual incompleta
    except BinanceAPIException as e:
        logger.error(f"Error de API al obtener datos para {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error inesperado al obtener datos para {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(data, funding_rate):
    """Calcula indicadores técnicos (RSI, EMA, MACD, etc.)."""
    
    # Asegurar que el DataFrame no esté vacío
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

    # Funding Rate (Se añade como una columna constante, ya que solo tenemos el valor actual)
    data['FundingRate'] = funding_rate
    
    # Limpiar NaN resultantes del cálculo de indicadores
    data.dropna(inplace=True)
    
    if data.empty:
        logger.warning("DataFrame vacío después de calcular indicadores y limpiar NaN.")
        return None
        
    return data

# ----------------------------------------------------------------------------------
# FUNCIONES DE MACHINE LEARNING (Fase 2)
# ----------------------------------------------------------------------------------

def create_target_variable(df):
    """Crea la variable objetivo 'Target': 1 si el precio sube en las próximas N velas, 0 si baja/se mantiene."""
    
    # Calcula el precio futuro N velas adelante
    df['FutureClose'] = df['Close'].shift(-MODEL_TARGET_CANDLES)
    
    # Target: 1 si el FutureClose es mayor que el Close actual, 0 en caso contrario.
    # Usaremos una pequeña tolerancia (0.001% de movimiento) para evitar ruido.
    df['Target'] = np.where(df['FutureClose'] > (df['Close'] * 1.00001), 1, 0)
    
    # Eliminar filas con NaN (las últimas N filas no tienen FutureClose)
    return df.dropna().drop(columns=['FutureClose'])

def calculate_ml_features(df):
    """Prepara las features X para el modelo ML."""
    
    # Si el DF está vacío o no tiene la longitud suficiente
    if df is None or len(df) < 1:
        return None
        
    # Última fila del DataFrame (datos de la vela actual)
    features_row = df.iloc[-1].copy()
    
    # 1. RSI (último valor)
    rsi_val = features_row['RSI']
    
    # 2. Distancia a EMA50 (Normalizada por el precio actual)
    # Valor positivo = precio por encima de la EMA
    distance_to_ema = (features_row['Close'] - features_row['EMA50']) / features_row['Close']
    
    # 3. MACD Histograma (último valor)
    hist_val = features_row['Hist']
    
    # 4. Tasa de Funding Rate (último valor)
    funding_rate_val = features_row['FundingRate']

    # 5. Volatilidad (Rango ATR simple normalizado)
    # Se utiliza el rango de la vela actual normalizado por el Close
    volatility = (features_row['High'] - features_row['Low']) / features_row['Close']
    
    # Crear el vector de features
    X = np.array([[rsi_val, distance_to_ema, hist_val, funding_rate_val, volatility]])
    
    return X
    # Continua del Bloque 1
# ----------------------------------------------------------------------------------
# FUNCIONES DE MACHINE LEARNING (Fase 2 - Continuación)
# ----------------------------------------------------------------------------------

def initialize_ml_model(client):
    """Entrena y carga el modelo de Regresión Logística en memoria."""
    global ML_MODEL, SCALER, APP_STATE

    APP_STATE['model_ready'] = False
    logger.warning("Modelo ML no encontrado en memoria. INICIANDO ENTRENAMIENTO...")
    
    # Usaremos BTCUSDT para un entrenamiento general más estable
    symbol = 'BTCUSDT'
    logger.info(f"Buscando datos históricos de {symbol} por {LOOKBACK_PERIOD}...")
    
    # 1. Obtener Data Histórica (Requiere claves de producción)
    # Usamos un periodo más largo para el entrenamiento
    data = get_binance_data(client, symbol, INTERVAL, LOOKBACK_PERIOD)
    
    if data.empty:
        logger.critical("❌ NO SE PUDO DESCARGAR DATA HISTÓRICA para entrenamiento. El bot usará el modo DRY_RUN con Lógica Manual.")
        return # Fallback a modo de predicción manual (Puntaje)

    # 2. Calcular Indicadores (Features de ML)
    # Nota: Aquí no podemos obtener el funding rate histórico fácilmente, así que usamos 0 para el training.
    # En el make_decision, usaremos el valor actual real.
    df = calculate_indicators(data, funding_rate=0.0)
    
    if df is None or df.empty:
        logger.critical("❌ Data insuficiente para entrenamiento después de calcular indicadores.")
        return

    # 3. Crear Target y Preparar Data
    df_train = create_target_variable(df)
    
    # Columnas que serán nuestras features X
    feature_cols = ['RSI', 'Distancia_EMA50', 'Hist', 'FundingRate', 'Volatilidad']
    
    # Recalcular las features X para todo el dataframe (no solo la última fila)
    # Re-calculamos Distancia_EMA50 y Volatilidad para el training set
    df_train['Distancia_EMA50'] = (df_train['Close'] - df_train['EMA50']) / df_train['Close']
    df_train['Volatilidad'] = (df_train['High'] - df_train['Low']) / df_train['Close']
    
    X = df_train[feature_cols].values
    y = df_train['Target'].values

    if len(X) == 0:
        logger.critical("❌ No hay suficientes muestras de datos (X) para el entrenamiento.")
        return
    
    # 4. Escalar y Entrenar
    SCALER = StandardScaler()
    X_scaled = SCALER.fit_transform(X)
    
    ML_MODEL = LogisticRegression(solver='liblinear', random_state=42)
    ML_MODEL.fit(X_scaled, y)
    
    # 5. Evaluación simple
    accuracy = ML_MODEL.score(X_scaled, y)
    
    APP_STATE['model_ready'] = True
    logger.info(f"✅ Modelo ML cargado en memoria exitosamente. Precisión (entrenamiento): {accuracy:.4f}")

# ----------------------------------------------------------------------------------
# LOGICA DE TRADING
# ----------------------------------------------------------------------------------

def make_decision(data, symbol, funding_rate):
    """Toma la decisión de trading usando el modelo ML o la lógica manual."""
    global APP_STATE
    
    # 1. LOGICA ML (Si está listo)
    if APP_STATE['model_ready']:
        try:
            # Re-calculamos las features X con la data real de la última vela
            df_with_indicators = calculate_indicators(data, funding_rate)
            
            # 1. Calcular las features para la predicción
            X_live = calculate_ml_features(df_with_indicators)
            
            if X_live is None or SCALER is None:
                raise ValueError("No se pudieron calcular las features o el escalador no está listo.")

            # 2. Escalar los datos EN VIVO
            X_live_scaled = SCALER.transform(X_live)
            
            # 3. Predecir la probabilidad de subida
            # predict_proba retorna [[Probabilidad_0 (baja), Probabilidad_1 (sube)]]
            probabilities = ML_MODEL.predict_proba(X_live_scaled)[0]
            prob_buy = probabilities[1]
            
            # 4. Decisión basada en el umbral
            signal = 'HOLD'
            if prob_buy >= MODEL_CONFIDENCE_THRESHOLD:
                signal = 'BUY'
            elif prob_buy <= (1 - MODEL_CONFIDENCE_THRESHOLD): # Umbral para la venta (bajada)
                signal = 'SELL' 
                
            last_close = df_with_indicators['Close'].iloc[-1]
            log_message = f"{symbol} | Señal: {signal} (Probabilidad Buy: {prob_buy:.4f}) | Precio: {last_close:.4f} | Funding Rate: {funding_rate:.5f}"
            logger.info(log_message)
            
            # Guardar el estado
            APP_STATE['symbol_data'][symbol] = {
                'last_signal': signal,
                'last_price': last_close,
                'confidence': prob_buy,
                'funding_rate': funding_rate,
                'used_ml': True
            }
            
            return signal, last_close, prob_buy

        except Exception as e:
            # Si el ML falla por cualquier motivo (ej: error en la data, escalador)
            logger.error(f"❌ FALLO DE ML para {symbol}: {e}. Volviendo a la Lógica Manual (Puntaje).")
            APP_STATE['model_ready'] = False # Forzar el re-entrenamiento en el siguiente ciclo si es necesario
            # Continuar con el código de Lógica Manual (Paso 2)
            pass

    # 2. LOGICA MANUAL (Fallback si el ML no está listo o falló)
    if not APP_STATE['model_ready']:
        
        # 1. Calcular Indicadores (incluye el Funding Rate actual)
        df_with_indicators = calculate_indicators(data, funding_rate)

        if df_with_indicators is None:
            logger.critical(f"❌ {symbol}: No se obtuvieron datos suficientes, saltando análisis manual.")
            return 'HOLD', None, 0.0

        latest = df_with_indicators.iloc[-1]
        score = 0.0
        signal = 'HOLD'
        
        # Reglas Manuales (Puntuación de la Fase 1)
        
        # Regla 1: RSI - Momentum
        if latest['RSI'] < 30: # Sobreventa
            score += 1.5 # Fuerte señal de compra
        elif latest['RSI'] > 70: # Sobrecompra
            score -= 1.5 # Fuerte señal de venta

        # Regla 2: Cruce de MACD (Hist > 0 para Buy, Hist < 0 para Sell)
        if latest['Hist'] > 0:
            score += 1.0
        elif latest['Hist'] < 0:
            score -= 1.0

        # Regla 3: Posición respecto a la EMA50 (Tendencia)
        if latest['Close'] > latest['EMA50']:
            score += 0.5
        elif latest['Close'] < latest['EMA50']:
            score -= 0.5

        # Regla 4: Funding Rate (Sesgo de la Tasa de Financiación)
        if funding_rate < -0.0001: # Tasa muy negativa (más gente shorteando -> Buy)
            score += 1.0
        elif funding_rate > 0.0001: # Tasa muy positiva (más gente comprando -> Sell)
            score -= 1.0
        
        # Decisión final
        if score >= 2.0:
            signal = 'BUY'
        elif score <= -2.0:
            signal = 'SELL'
            
        last_close = latest['Close']
        log_message = f"{symbol} | Señal: {signal} (Puntaje: {score:.1f}) | Precio: {last_close:.4f} | Funding Rate: {funding_rate:.5f}"
        logger.info(log_message)
        
        # Guardar el estado
        APP_STATE['symbol_data'][symbol] = {
            'last_signal': signal,
            'last_price': last_close,
            'confidence': score,
            'funding_rate': funding_rate,
            'used_ml': False
        }
        
        return signal, last_close, score

# ----------------------------------------------------------------------------------
# BUCLE PRINCIPAL DE EJECUCIÓN
# ----------------------------------------------------------------------------------

def run_trading_bot():
    """Bucle principal del bot de trading."""
    
    # 1. Inicialización de Conexión
    client = initialize_client()
    
    # 2. Inicialización de Machine Learning (Entrenamiento si no está cargado)
    if not APP_STATE['model_ready']:
        # Este paso usará las claves de Producción para descargar la data histórica
        initialize_ml_model(client)

    # El bucle del bot
    while True:
        try:
            logger.info(f"Ciclo de trading iniciado.")
            
            # Obtener balance actualizado (simulado)
            logger.info(f"Balances de la cuenta actualizados ({'SIMULACIÓN' if APP_STATE['dry_run'] else 'REAL'}). USDT disponible: {APP_STATE['balances']['free_USDT']:.2f}")

            # 3. Iterar sobre todos los pares
            for symbol in SYMBOL_PAIRS:
                
                # Obtener data y funding rate
                logger.info(f"Obteniendo datos de velas para {symbol} en intervalo {INTERVAL}...")
                data = get_binance_data(client, symbol, INTERVAL, LOOKBACK_PERIOD)
                funding_rate = get_funding_rate(client, symbol)
                
                if data.empty:
                    logger.error(f"Error INESPERADO en el ciclo de trading: No se pudieron obtener datos, saltando ciclo.")
                    continue

                # Tomar decisión (ML o Manual)
                signal, close_price, confidence = make_decision(data, symbol, funding_rate)

                # TODO: FASE 3: Aquí se implementará la lógica de gestión de riesgo y ejecución de órdenes

            logger.info(f"Ciclo completado para todos los pares. Volviendo a dormir por {CYCLE_DELAY_SECONDS} segundos.")
            APP_STATE['last_run_utc'] = datetime.utcnow().isoformat()
            time.sleep(CYCLE_DELAY_SECONDS)

        except BinanceAPIException as e:
            logger.error(f"Error de API de Binance: {e}. Esperando {CYCLE_DELAY_SECONDS} segundos.")
            time.sleep(CYCLE_DELAY_SECONDS)
        except Exception as e:
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
    
    # Formatear datos para el JSON
    state_data = {
        'status': 'RUNNING',
        'dry_run_mode': APP_STATE['dry_run'],
        'model_ml_ready': APP_STATE['model_ready'],
        'last_run_utc': APP_STATE['last_run_utc'],
        'balances': {
            'free_USDT': APP_STATE['balances']['free_USDT'],
        },
        'symbol_data': APP_STATE['symbol_data'],
        'open_positions': APP_STATE['open_positions'],
    }
    return jsonify(current_state=state_data)

@app.route('/', methods=['GET'])
def home():
    """Endpoint de bienvenida."""
    return jsonify(message="Trading Bot Activo. Accede a /state para ver el estado.")

# ----------------------------------------------------------------------------------
# ARRANQUE
# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    # El thread de trading inicia en segundo plano, la app web en primer plano
    import threading
    trading_thread = threading.Thread(target=run_trading_bot)
    trading_thread.daemon = True
    trading_thread.start()
    
    # Gunicorn ejecutará 'gunicorn trading_bot:app', usando la variable 'app'
    # Esta parte solo corre si ejecutas el archivo directamente (p.ej. python trading_bot.py)
    # En Render, esto es manejado por el Procfile (web: gunicorn trading_bot:app)
    # app.run(host='0.0.0.0', port=os.environ.get('PORT', 10000))

