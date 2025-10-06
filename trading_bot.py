import os
import time
import logging
import threading
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Librerías para Machine Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from concurrent.futures import ThreadPoolExecutor

# Flask para el servicio web
from flask import Flask, jsonify, request
from threading import Thread

# --- Configuración Inicial ---

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Claves de la API (obtenidas de variables de entorno)
# NOTA: Para entrenar el modelo (initialize_ml_model), se requiere la API de PRODUCCIÓN para datos históricos.
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

# URL de la API de Testnet (para operar de forma segura)
FUTURES_TESTNET_URL = os.environ.get('FUTURES_TESTNET_URL', 'https://testnet.binancefuture.com')

# --- Estado Global del Bot ---

app = Flask(__name__)

# Estado compartido para el ciclo de trading
state = {
    'is_running': False,
    'current_mode': 'DRY_RUN',
    'last_run_utc': None,
    'symbol_data': {}, # Datos técnicos y señales por par
    'balances': {'free_USDT': 0.0, 'total_USDT': 0.0},
    'ml_model': None, # El modelo de ML cargado en memoria
    'ml_scaler': None, # El scaler para normalizar datos
    'is_ml_ready': False,
    'monitored_symbols': ['TRXUSDT', 'XRPUSDT', 'BTCUSDT'],
    'interval': '5m',
    'limit_percent': 0.005, # % de margen por operación (0.5%)
    'initial_capital': 1000.00
}

# Parámetros de Trading
TRADE_INTERVAL_SECONDS = 300 # 5 minutos
MODEL_CONFIDENCE_THRESHOLD = 0.70 # Probabilidad mínima del modelo para BUY/SELL

# Inicialización del Cliente de Binance
try:
    # Usamos la URL de Testnet para el trading por seguridad (DRY_RUN)
    client = Client(API_KEY, API_SECRET, base_url=FUTURES_TESTNET_URL)
    state['is_running'] = True
    logging.info("🟢 Conectado a Binance PRODUCCIÓN (SIMULACIÓN). Estado de la cuenta: True")
    state['current_mode'] = 'DRY_RUN' 
except Exception as e:
    logging.error(f"🔴 Error al conectar con Binance: {e}")
    state['is_running'] = False


# --- Funciones de Utilidad de ML (Auto-Entrenamiento) ---

def calculate_ml_features(df):
    """Calcula indicadores técnicos y Funding Rate para usar como features de ML."""
    # 1. Indicadores Comunes
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()

    # 2. MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3. Volatilidad (ATR Simple: Rango Alto - Bajo)
    df['Volatilidad'] = df['High'] - df['Low']

    # 4. Target (Variable Objetivo): Sube/Baja en la siguiente vela
    # 1 si el precio sube en la siguiente vela, 0 si baja/se mantiene
    # Se usa shift(-1) para que la fila N tenga el resultado de la vela N+1
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # 5. Features Finales (seleccionadas para el modelo)
    features = df[['RSI', 'EMA20', 'EMA50', 'EMA100', 'MACD', 'Signal_Line', 'Volatilidad']].iloc[:-1] # Excluimos la última fila sin Target
    target = df['Target'].iloc[:-1]
    
    # Manejo de NaNs (rellenar con 0 o la media)
    features = features.fillna(features.mean())
    
    return features, target

def compute_rsi(data, window):
    """Función para calcular el RSI."""
    diff = data.diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_funding_rate(symbol):
    """Obtiene la Funding Rate actual de Binance (usando la API de producción)."""
    try:
        # **ATENCIÓN:** Usamos la API de PRODUCCIÓN para obtener la Funding Rate en vivo
        # Es por diseño para tener datos reales, aunque operemos en Testnet/DRY_RUN
        prod_client = Client(API_KEY, API_SECRET)
        rate_info = prod_client.futures_funding_rate(symbol=symbol)
        if rate_info:
            return float(rate_info[0]['fundingRate'])
        return 0.0
    except Exception as e:
        logging.error(f"🔴 Error al obtener Funding Rate para {symbol}: {e}")
        return 0.0


def initialize_ml_model(symbol):
    """
    Entrena un modelo de Regresión Logística y lo almacena en memoria.
    Esto se ejecuta solo una vez al inicio del bot.
    """
    global client
    logging.warning("Modelo ML no encontrado en memoria. INICIANDO ENTRENAMIENTO...")

    try:
        # Temporalmente, cambiamos el cliente a la URL de PRODUCCIÓN para la descarga de datos
        # Esto requiere las claves de PRODUCCIÓN en el entorno de Render
        training_client = Client(API_KEY, API_SECRET)
        
        logging.info(f"Buscando datos históricos de {symbol} por 100 days ago UTC...")
        
        # Descarga de datos
        klines = training_client.get_historical_klines(
            symbol,
            state['interval'],
            "100 days ago UTC"
        )
        
        # Convertir a DataFrame
        data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        data['Close'] = pd.to_numeric(data['Close'])
        data['High'] = pd.to_numeric(data['High'])
        data['Low'] = pd.to_numeric(data['Low'])

        # Calcular Features y Target
        features, target = calculate_ml_features(data)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)

        # Escalar/Normalizar Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar el Modelo (Regresión Logística)
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluar el modelo (opcional)
        accuracy = model.score(X_test_scaled, y_test)
        logging.info(f"🤖 Precisión del modelo de Regresión Logística para {symbol}: {accuracy:.4f}")

        # Guardar en el estado (memoria)
        state['ml_model'] = model
        state['ml_scaler'] = scaler
        state['is_ml_ready'] = True
        
        logging.info("✅ Modelo ML cargado en memoria exitosamente.")

    except BinanceAPIException as e:
        # Si las claves de Prod no funcionan para descargar (ej: error -1000), el bot no puede entrenar
        logging.critical(f"🔴 CRÍTICO: Error de API de Binance al intentar descargar data para ML. Verifica tus claves de PRODUCCIÓN. {e}")
    except Exception as e:
        logging.critical(f"🔴 CRÍTICO: Error inesperado durante el entrenamiento de ML: {e}")


# --- Funciones de Trading Principal ---

def get_binance_data(symbol, interval='5m', limit=500):
    """Obtiene datos de velas de Binance y calcula indicadores."""
    
    # 1. Obtener Klines
    try:
        # Obtener datos de velas (Close time, Open, High, Low, Close)
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        
        # Convertir a DataFrame
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        
        # 2. Obtener Funding Rate
        funding_rate = get_funding_rate(symbol)
        df['FundingRate'] = funding_rate
        
        return df
    except Exception as e:
        logging.error(f"🔴 Error al obtener datos de velas para {symbol}: {e}")
        return None

def calculate_ml_decision(df):
    """Usa el modelo de Regresión Logística para predecir la probabilidad de BUY/SELL."""
    
    if not state['is_ml_ready']:
        return "HOLD", 0.5 # Default si el modelo no está cargado
    
    # 1. Preparar el DataFrame para la predicción
    # Usamos la misma función de features, pero solo la última fila completa
    features, _ = calculate_ml_features(df)
    
    # Seleccionamos el último conjunto de features para predecir
    latest_features = features.iloc[[-1]]
    
    # 2. Normalizar las features (CRÍTICO: usar el mismo scaler del entrenamiento)
    latest_features_scaled = state['ml_scaler'].transform(latest_features)
    
    # 3. Predicción de Probabilidad
    # predict_proba retorna [[Prob_Clase_0 (SELL/HOLD), Prob_Clase_1 (BUY)]]
    probabilities = state['ml_model'].predict_proba(latest_features_scaled)[0]
    prob_buy = probabilities[1]
    
    # 4. Decisión Final basada en el umbral de confianza
    if prob_buy >= MODEL_CONFIDENCE_THRESHOLD:
        signal = "BUY"
    elif prob_buy <= (1 - MODEL_CONFIDENCE_THRESHOLD): # Umbral opuesto para SELL
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, prob_buy

def update_symbol_state(symbol, data):
    """Actualiza los datos técnicos, la señal y el precio actual."""
    
    if data is None or data.empty:
        logging.warning(f"⚠️ {symbol} | No se obtuvieron datos o el DF está vacío. Saltando análisis para este par.")
        return

    try:
        # Obtener señal de ML
        signal, prob_buy = calculate_ml_decision(data)
        
        # Precio de la última vela
        current_price = data['Close'].iloc[-1]
        
        # Funding Rate
        funding_rate = data['FundingRate'].iloc[-1]
        
        # Actualizar estado global
        state['symbol_data'][symbol] = {
            'last_signal': signal,
            'prob_buy': float(f'{prob_buy:.4f}'),
            'current_price': float(f'{current_price:.4f}'),
            'funding_rate': float(f'{funding_rate:.5f}'),
            'last_signal_time': datetime.utcnow().isoformat()
        }
        
        logging.info(f"📈 {symbol} | Señal: {signal} (Probabilidad Buy: {prob_buy:.4f}) | Precio: {current_price:.4f} | Funding Rate: {funding_rate:.5f}")

        # Ejecutar Trading (Solo si no es HOLD y estamos en DRY_RUN)
        if signal != "HOLD" and state['current_mode'] == 'DRY_RUN':
            execute_simulated_trade(symbol, signal, current_price)

    except Exception as e:
        logging.error(f"❌ Error al obtener datos o calcular indicadores para {symbol}: {e}")

def execute_simulated_trade(symbol, signal, price):
    """Ejecuta una orden simulada y actualiza el balance virtual."""
    
    # Calcular tamaño de la orden (usando el límite de capital)
    order_size_usd = state['initial_capital'] * state['limit_percent']
    quantity = order_size_usd / price

    if signal == "BUY":
        logging.info(f"💰 {symbol} - BUY (Simulado): Compraría {quantity:.2f} {symbol.replace('USDT', '')} a {price:.4f} USD. (Costo: {order_size_usd:.2f})")
        # Simular impacto en el balance (esto es muy simple y no maneja posiciones abiertas)
        state['balances']['free_USDT'] -= order_size_usd 
        state['balances']['total_USDT'] = state['balances']['free_USDT'] # Mantener simple por ahora
    elif signal == "SELL":
        logging.info(f"🔴 {symbol} - SELL (Simulado): Vendería {quantity:.2f} {symbol.replace('USDT', '')} a {price:.4f} USD. (Ganancia/Pérdida no calculada en simulación simple)")
        # Dejamos la gestión de riesgo más compleja para la Fase 3
        state['balances']['free_USDT'] += order_size_usd
        state['balances']['total_USDT'] = state['balances']['free_USDT']


def update_balances():
    """Actualiza el balance de la cuenta (Simulado)."""
    # En DRY_RUN o Testnet, solo inicializamos un balance ficticio
    if state['current_mode'] == 'DRY_RUN' or 'TESTNET' in client.base_url:
        if state['balances']['free_USDT'] == 0.0:
            state['balances']['free_USDT'] = state['initial_capital']
            state['balances']['total_USDT'] = state['initial_capital']
            logging.info(f"Balances iniciales cargados en el estado. USDT disponible: {state['balances']['free_USDT']:.2f}")
        else:
            logging.info(f"Balances de la cuenta actualizados (SIMULACIÓN). USDT disponible: {state['balances']['free_USDT']:.2f}")
    else:
        # Lógica para obtener el balance REAL (omitida por seguridad y enfoque en ML)
        logging.warning("El bot está en modo REAL pero la función de balance real no está implementada.")


def trading_cycle():
    """Bucle principal de ejecución del bot."""
    logging.info(f"Bot iniciado en modo: {state['current_mode']}.")
    logging.info(f"Vigilando: {', '.join(state['monitored_symbols'])}")
    
    # Inicialización del modelo ML
    if not state['is_ml_ready']:
        # Solo entrenamos con el primer símbolo como base
        initialize_ml_model(state['monitored_symbols'][0]) 

    update_balances()

    while state['is_running']:
        start_time = time.time()
        
        # Usamos ThreadPoolExecutor para obtener datos y calcular en paralelo
        with ThreadPoolExecutor(max_workers=len(state['monitored_symbols'])) as executor:
            future_to_symbol = {executor.submit(get_binance_data, symbol, state['interval'], 300): symbol for symbol in state['monitored_symbols']}
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    update_symbol_state(symbol, data)
                except Exception as exc:
                    logging.error(f"❌ Error al procesar {symbol}: {exc}")

        # Marcar la hora de ejecución
        state['last_run_utc'] = datetime.utcnow().isoformat()
        
        update_balances()

        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = max(0, TRADE_INTERVAL_SECONDS - elapsed_time)
        
        logging.info(f"Ciclo completado para todos los pares. Volviendo a dormir por {int(sleep_time)} segundos.")
        time.sleep(sleep_time)


# --- Flask Web Server (Para mantener vivo el bot) ---

@app.route('/state', methods=['GET'])
def get_state():
    """Devuelve el estado actual del bot como JSON."""
    return jsonify({
        'status': 'Running' if state['is_running'] else 'Stopped',
        'current_state': state
    })

def run_flask():
    """Ejecuta el servidor Flask."""
    # Usamos Threading si es necesario, pero gunicorn lo maneja bien
    logging.info("Servicio Flask iniciado.")
    # El puerto 8080 es el estándar de Render
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))


# --- Inicio del Bot ---

if __name__ == '__main__':
    # Inicializar y correr el ciclo de trading en un hilo separado
    trading_thread = Thread(target=trading_cycle)
    trading_thread.start()
    logging.info("Hilo de trading iniciado con éxito en segundo plano.")
    
    # Iniciar Flask en el hilo principal (lo requiere Gunicorn/Render)
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))

# Para Gunicorn
# NOTA: Gunicorn requiere que la aplicación Flask se llame 'app' en el nivel superior del módulo.
# Esto ya está hecho: app = Flask(__name__)
# El Procfile debe ser: web: gunicorn trading_bot:app
