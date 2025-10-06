import os
import time
import threading
import json
import pandas as pd
from flask import Flask, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException # Asegúrate de que esta importación esté presente
import numpy as np
# Importación necesaria para el cálculo avanzado de MACD y Bollinger.
# Si estás ejecutando esto localmente, puedes necesitar 'pip install ta'

# --- 1. CONFIGURACIÓN Y VARIABLES GLOBALES ---

# Cargar variables de entorno
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() == 'true'
# NUEVO: Se añade la variable para controlar explícitamente si se usa Testnet.
USE_TESTNET = os.environ.get('USE_TESTNET', 'false').lower() == 'true'


# Parámetros de la Estrategia (Recuperados de las variables de entorno)
# AHORA SE USA UNA LISTA DE SÍMBOLOS SEPARADOS POR COMAS
# VALOR ACTUALIZADO: TRXUSDT, XRPUSDT, BTCUSDT por defecto para hacer pruebas
SYMBOLS_LIST_STR = os.environ.get('SYMBOLS_LIST', 'TRXUSDT,XRPUSDT,BTCUSDT').replace(" ", "")
SYMBOLS_LIST = [s.strip() for s in SYMBOLS_LIST_STR.split(',') if s.strip()]

INTERVAL = os.environ.get('INTERVAL', '15m')
# 🚨 VALOR POR DEFECTO ACTUALIZADO A 0.5 (50%) PARA CUMPLIR EL MÍNIMO DE $10.5 CON $22 USD
PCT_OF_BALANCE = float(os.environ.get('PCT_OF_BALANCE', 0.5)) 
SLEEP_SEC = int(os.environ.get('SLEEP_SEC', 300))
MIN_ORDER_USD = float(os.environ.get('MIN_ORDER_USD', 10.5)) # Subido a 10.5 por seguridad
# Nuevo Umbral de Decisión: La señal debe tener al menos este puntaje para ejecutarse.
DECISION_THRESHOLD = 3 # Puntos de decisión requeridos para una señal

# NUEVA CONSTANTE: Máximo de reintentos para las llamadas al API de datos
MAX_RETRIES = 5 

# Variables de estado del bot
bot_state = {
    "configuration": {
        "dry_run": DRY_RUN,
        "use_testnet": USE_TESTNET, # <-- Añadido al estado
        "interval": INTERVAL,
        "min_order_usd": MIN_ORDER_USD,
        "pct_of_balance": PCT_OF_BALANCE,
        "sleep_sec": SLEEP_SEC,
        "symbols_list": SYMBOLS_LIST, # Lista de símbolos vigilados
        "decision_threshold": DECISION_THRESHOLD
    },
    "current_state": {
        # Balances generales, los balances de activos específicos se agregan en tiempo real
        "balances": {"free_USDT": 0, "free_BNB": 0}, 
        "asset_balances": {}, # Almacenará balances específicos de cada moneda (ej: 'TRX': 500)
        "last_run_utc": None,
        "symbol_data": {} # Almacenará la última señal y puntaje por cada símbolo
    },
    "trade_history": []
}

# Variable de control del hilo
bot_thread_running = False

# Inicializar Flask
app = Flask(__name__)

# --- 2. INICIALIZACIÓN DE BINANCE ---

if not API_KEY or not API_SECRET:
    print("❌ Faltan BINANCE_API_KEY / BINANCE_API_SECRET en variables de entorno.")
    exit()

try:
    # 🚨 CLIENTE AUTENTICADO (Para órdenes y balances)
    client = Client(API_KEY, API_SECRET)
    
    # 🚨 CLIENTE PÚBLICO (Solo para datos de mercado, NO requiere API Key)
    # Lo inicializamos sin credenciales. Lo usaremos para get_historical_klines
    public_client = Client("", "") 
    
    # Determinar el entorno de conexión
    if USE_TESTNET:
        connection_target = "TESTNET (SIMULACIÓN)"
        # Si USE_TESTNET es verdadero, forzamos la URL de Testnet en AMBOS clientes
        client.API_URL = 'https://testnet.binance.vision/api'
        public_client.API_URL = 'https://testnet.binance.vision/api'
    elif DRY_RUN: 
        connection_target = "PRODUCCIÓN (SIMULACIÓN)"
    else:
        connection_target = "PRODUCCIÓN (DINERO REAL)"

    # Sincronizar el tiempo del cliente con el servidor de Binance para el CLIENTE AUTENTICADO
    client.timestamp_offset = client.get_server_time()['serverTime'] - int(time.time() * 1000) 
    print("✅ Tiempo del servidor sincronizado.")
    
    # Verificar conexión (Esto fallará si las claves no coinciden con el entorno)
    info = client.get_account()
    print(f"✅ Conectado a Binance {connection_target}. Estado de la cuenta:", info['canTrade'])
    
except Exception as e:
    # Mensaje de error más detallado sobre la conexión
    print(f"❌ Error al conectar con Binance. Revise credenciales, entorno (real/testnet) y restricciones geográficas. {e}")
    exit()

# --- 3. FUNCIONES DE ESTRATEGIA (MODIFICADA PARA USAR public_client) ---

def get_data(symbol):
    """Obtiene datos de velas y calcula indicadores para un símbolo específico, con reintentos."""
    
    # Lógica de reintento para mejorar la resiliencia contra errores de conexión/servidor
    for attempt in range(MAX_RETRIES):
        try:
            print(f"📊 Obteniendo datos de velas para {symbol} en intervalo {INTERVAL}...")
            # 🚨 CAMBIO CLAVE: Usamos public_client para obtener klines. ESTO NO REQUIERE AUTENTICACIÓN.
            klines = public_client.get_historical_klines(symbol, INTERVAL, "500 ago UTC")
            
            # Si tiene éxito, procesar el DataFrame
            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 
                                              'volume', 'close_time', 'quote_asset_volume', 
                                              'number_of_trades', 'taker_buy_base_asset_volume', 
                                              'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = pd.to_numeric(df['close'])

            # --- CÁLCULO DE INDICADORES (MISMA LÓGICA DE IA PONDERADA) ---
            
            # 1. RSI (Índice de Fuerza Relativa)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # 2. EMAs 
            df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

            # 3. MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd_line'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()

            # 4. Bandas de Bollinger 
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['stddev'] = df['close'].rolling(window=20).std()
            df['bollinger_upper'] = df['sma20'] + (df['stddev'] * 2)
            df['bollinger_lower'] = df['sma20'] - (df['stddev'] * 2)

            return df # Éxito, retorna el DataFrame y sale del bucle
        
        except BinanceAPIException as e:
            # Si es un error de Binance, imprime el código de error.
            print(f"❌ Error CRÍTICO (Binance API Code {e.code}) al obtener datos para {symbol}: {e}. Saltando el par.")
            return None # Fallo final, devuelve None
            
        except Exception as e:
            wait_time = 2 ** attempt # Backoff exponencial (1s, 2s, 4s, 8s...)
            if attempt < MAX_RETRIES - 1:
                print(f"❌ Error TEMPORAL al obtener datos para {symbol}: {e}. Reintentando en {wait_time} segundos (Intento {attempt + 1}/{MAX_RETRIES}).")
                time.sleep(wait_time)
            else:
                print(f"❌ Error CRÍTICO y persistente al obtener datos para {symbol} después de {MAX_RETRIES} intentos: {e}. Saltando el par.")
                return None # Fallo final, devuelve None


def get_signal(df, symbol):
    """Genera la señal de trading para el símbolo usando lógica de puntaje."""
    if df is None or len(df) < 50: 
        bot_state["current_state"]["symbol_data"][symbol] = {"last_signal": "HOLD", "decision_score": 0}
        return "HOLD", 0

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # --- DATOS DE ENTRADA AL MOTOR DE DECISIÓN ---
    rsi = last_row['rsi']
    ema9 = last_row['ema9']
    ema21 = last_row['ema21']
    macd_line = last_row['macd_line']
    macd_signal = last_row['macd_signal']
    current_price = last_row['close']
    bollinger_upper = last_row['bollinger_upper']
    bollinger_lower = last_row['bollinger_lower']

    # --- LÓGICA DE PUNTAJE (SIMULACIÓN DE CLASIFICADOR) ---
    buy_score = 0
    sell_score = 0
    
    # 1. Criterio de RSI (Impulso)
    if rsi < 40:
        buy_score += 1
    elif rsi > 60:
        sell_score += 1

    # 2. Criterio de Crossover EMA (Tendencia)
    if ema9 > ema21 and prev_row['ema9'] <= prev_row['ema21']:
        buy_score += 2
    elif ema9 < ema21 and prev_row['ema9'] >= prev_row['ema21']:
        sell_score += 2
    
    # 3. Criterio de MACD (Momento)
    if macd_line > macd_signal and prev_row['macd_line'] <= prev_row['macd_signal']:
        buy_score += 1.5
    elif macd_line < macd_signal and prev_row['macd_line'] >= prev_row['macd_signal']:
        sell_score += 1.5

    # 4. Criterio de Bandas de Bollinger (Volatilidad y Extremo)
    if current_price < bollinger_lower:
        buy_score += 1
    elif current_price > bollinger_upper:
        sell_score += 1

    # --- EVALUACIÓN DE LA DECISIÓN ---
    
    decision_score = max(buy_score, sell_score)
    
    # print(f"⚙️ {symbol} - Análisis Ponderado: Puntaje Compra: {buy_score:.1f}, Puntaje Venta: {sell_score:.1f}")
    
    if buy_score >= DECISION_THRESHOLD and buy_score > sell_score:
        signal = "BUY"
    elif sell_score >= DECISION_THRESHOLD and sell_score > buy_score:
        signal = "SELL"
    else:
        signal = "HOLD"
        
    bot_state["current_state"]["symbol_data"][symbol] = {"last_signal": signal, "decision_score": decision_score}

    return signal, decision_score

# --- 4. FUNCIONES DE EJECUCIÓN (MODIFICADA PARA GESTIÓN DE CAPITAL) ---

def update_balances():
    """Actualiza los balances de USDT, BNB y de todos los activos vigilados."""
    try:
        # Se usa el cliente autenticado
        account_info = client.get_account() 
        balances = {asset['asset']: float(asset['free']) for asset in account_info['balances']}

        # 1. Actualizar saldos principales
        bot_state["current_state"]["balances"]["free_USDT"] = balances.get('USDT', 0.0)
        bot_state["current_state"]["balances"]["free_BNB"] = balances.get('BNB', 0.0)
        
        # 2. Actualizar saldos de activos base vigilados
        for symbol in SYMBOLS_LIST:
            base_asset = symbol.replace("USDT", "")
            bot_state["current_state"]["asset_balances"][base_asset] = balances.get(base_asset, 0.0)

    except Exception as e:
        # Si esta sección falla, CONFIRMA que la autenticación (claves) es el problema.
        print(f"❌ Error al actualizar balances (REQUIERE AUTENTICACIÓN): {e}")

def execute_order(symbol, signal, current_price):
    """Ejecuta una orden de COMPRA o VENTA si DRY_RUN es False para un símbolo específico."""
    
    base_asset = symbol.replace("USDT", "") 
    usdt_free_total = bot_state["current_state"]["balances"]["free_USDT"]
    base_free = bot_state["current_state"]["asset_balances"].get(base_asset, 0.0)

    # 🚨 GESTIÓN DE CAPITAL: Usa el porcentaje del saldo total de USDT
    
    if signal == "BUY":
        # Calcula el capital a gastar usando el porcentaje del saldo total de USDT
        usd_to_spend = usdt_free_total * PCT_OF_BALANCE
        
        # Esto asegura que no intentemos comprar más de lo que tenemos
        if usd_to_spend > usdt_free_total:
             usd_to_spend = usdt_free_total # Caso extremo, solo por seguridad
        
        # Asegura que la orden sea mayor que el mínimo de Binance (ej. $10.5)
        if usd_to_spend < MIN_ORDER_USD:
            print(f"⚠️ {symbol} - COMPRA: Saldo insuficiente o bajo para orden de {MIN_ORDER_USD} USD. (Disponible: {usdt_free_total:.2f})")
            return

        quantity = usd_to_spend / current_price
        
        try:
            # Se usa el cliente autenticado
            info = client.get_symbol_info(symbol=symbol) 
            step_size = float(info['filters'][2]['stepSize'])
            quantity = np.floor(quantity / step_size) * step_size # Redondeo de la cantidad
        except Exception as e:
            print(f"❌ Error al obtener info de símbolo {symbol}: {e}")
            return


        if DRY_RUN:
            # SIMULACIÓN DE ORDEN (Ajusta los balances en el estado local)
            print(f"💰 {symbol} - BUY (Simulado): Compraría {quantity:.2f} {base_asset} a {current_price:.4f} USD. (Costo: {usd_to_spend:.2f})")
            bot_state["current_state"]["balances"]["free_USDT"] -= usd_to_spend
            bot_state["current_state"]["asset_balances"][base_asset] = bot_state["current_state"]["asset_balances"].get(base_asset, 0.0) + quantity
        else:
            # ORDEN REAL DE BINANCE (Requiere cliente autenticado)
            try:
                print(f"💰 {symbol} - BUY (REAL): Enviando orden de mercado para comprar {quantity:.2f} {base_asset}...")
                order = client.create_order(symbol=symbol, side='BUY', type='MARKET', quantity=quantity)
                print(f"✅ {symbol} - Orden de COMPRA ejecutada. Status: {order['status']}")
                bot_state["trade_history"].append({"time": bot_state["current_state"]["last_run_utc"], "symbol": symbol, "type": "BUY", "quantity": quantity, "price": current_price, "status": order['status']})
            except Exception as e:
                 print(f"❌ {symbol} - FALLO AL EJECUTAR ORDEN REAL DE COMPRA (REQUIERE AUTENTICACIÓN): {e}")


    elif signal == "SELL":
        quantity = base_free
        
        try:
            # Se usa el cliente autenticado
            info = client.get_symbol_info(symbol=symbol) 
            step_size = float(info['filters'][2]['stepSize'])
            quantity = np.floor(quantity / step_size) * step_size
        except Exception as e:
            print(f"❌ Error al obtener info de símbolo {symbol}: {e}")
            return
        
        # Asegura que la cantidad a vender sea suficiente
        if (quantity * current_price) < MIN_ORDER_USD:
            print(f"⚠️ {symbol} - VENTA: Saldo de {base_asset} es muy bajo para vender. (Valor: {quantity * current_price:.2f} USD)")
            return
            
        if DRY_RUN:
            # SIMULACIÓN DE ORDEN
            revenue = quantity * current_price
            print(f"💸 {symbol} - SELL (Simulado): Vendería {quantity:.2f} {base_asset} a {current_price:.4f} USD. (Ingreso: {revenue:.2f})")
            bot_state["current_state"]["balances"]["free_USDT"] += revenue
            bot_state["current_state"]["asset_balances"][base_asset] = 0.0
        else:
            # ORDEN REAL DE BINANCE (Requiere cliente autenticado)
            try:
                print(f"💸 {symbol} - SELL (REAL): Enviando orden de mercado para vender {quantity:.2f} {base_asset}...")
                order = client.create_order(symbol=symbol, side='SELL', type='MARKET', quantity=quantity)
                print(f"✅ {symbol} - Orden de VENTA ejecutada. Status: {order['status']}")
                bot_state["trade_history"].append({"time": bot_state["current_state"]["last_run_utc"], "symbol": symbol, "type": "SELL", "quantity": quantity, "price": current_price, "status": order['status']})
            except Exception as e:
                 print(f"❌ {symbol} - FALLO AL EJECUTAR ORDEN REAL DE VENTA (REQUIERE AUTENTICACIÓN): {e}")


# --- 5. BUCLE PRINCIPAL DEL BOT (Thread) (MODIFICADO PARA MULTI-PAR) ---

def bot_loop():
    """El bucle infinito que corre en segundo plano, ahora iterando sobre múltiples símbolos."""
    global bot_thread_running
    bot_thread_running = True
    
    print(f"🤖 Bot iniciado en modo: {'DRY_RUN' if DRY_RUN else 'REAL'}.")
    print(f"✅ Vigilando: {', '.join(SYMBOLS_LIST)}")
    
    update_balances()
    print("✅ Balances iniciales cargados en el estado.")

    while True:
        try:
            bot_state["current_state"]["last_run_utc"] = pd.Timestamp.now(tz='UTC').isoformat()
            
            # 🚨 Bucle que procesa cada símbolo en la lista 🚨
            for symbol in SYMBOLS_LIST:
                # 1. Obtener datos (Usa public_client, NO requiere firma)
                df = get_data(symbol) 
                if df is None:
                    print(f"⚠️ {symbol}: No se obtuvieron datos, saltando análisis para este par.")
                    continue

                # 2. Generar señal
                signal, decision_score = get_signal(df, symbol)
                current_price = df.iloc[-1]['close']
                
                print(f"📊 {symbol} | Señal: {signal} (Puntaje: {decision_score:.1f}) | Precio: {current_price:.4f}")
                
                # 3. Ejecutar orden (Usa cliente, SÍ requiere firma)
                if signal != "HOLD":
                    execute_order(symbol, signal, current_price)
                    
            # Actualizamos todos los balances al final del ciclo (Usa cliente, SÍ requiere firma)
            update_balances()
            print(f"🟢 Ciclo completado para todos los pares. Volviendo a dormir por {SLEEP_SEC} segundos.")

        except BinanceAPIException as e:
            print(f"❌ ERROR DE BINANCE (API): {e}")
        except Exception as e:
            print(f"❌ ERROR INESPERADO en el ciclo de trading: {e}")
            
        time.sleep(SLEEP_SEC)

# --- 6. RUTAS FLASK (API) (SIN CAMBIOS) ---

@app.route('/')
def home():
    """Ruta principal para verificar que el servicio está activo."""
    return jsonify({
        "status": "ok",
        "message": f"Bot de Trading Activo. MODO: {'SIMULACIÓN' if DRY_RUN else 'REAL (RIESGO FINANCIERO)'}. Ver /state para detalles.",
        "dry_run": DRY_RUN
    })

@app.route('/state')
def get_state():
    """Ruta para obtener el estado actual del bot en formato JSON."""
    return jsonify(bot_state)

# --- 7. INICIO DEL SERVIDOR Y DEL THREAD (SIN CAMBIOS) ---

if not bot_thread_running:
    try:
        trading_thread = threading.Thread(target=bot_loop)
        trading_thread.start()
        print("🌐 Hilo de trading iniciado con éxito en segundo plano.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: No se pudo iniciar el hilo de trading: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
