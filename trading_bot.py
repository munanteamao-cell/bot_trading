import os
import time
import threading
import json
import pandas as pd
from flask import Flask, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np

# --- 1. CONFIGURACIÓN Y VARIABLES GLOBALES ---

# Cargar variables de entorno
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() == 'true'
USE_TESTNET = os.environ.get('USE_TESTNET', 'false').lower() == 'true'

# Parámetros de la Estrategia (Recuperados de las variables de entorno)
# NOTA: Los símbolos deben ser para FUTUROS (ej. BTCUSDT) si se usa funding.
SYMBOLS_LIST_STR = os.environ.get('SYMBOLS_LIST', 'TRXUSDT,XRPUSDT,BTCUSDT').replace(" ", "")
SYMBOLS_LIST = [s.strip() for s in SYMBOLS_LIST_STR.split(',') if s.strip()]

# *** Ajustar a 5m y 60 segundos en Render para mayor frecuencia ***
INTERVAL = os.environ.get('INTERVAL', '5m') 
# Reducimos a 60s para trading intradiario/ML
SLEEP_SEC = int(os.environ.get('SLEEP_SEC', 60)) 
# Porcentaje del saldo de USDT a usar por orden
PCT_OF_BALANCE = float(os.environ.get('PCT_OF_BALANCE', 0.5)) 
MIN_ORDER_USD = float(os.environ.get('MIN_ORDER_USD', 10.5)) 

# *** Usamos el DECISION_THRESHOLD que tenga configurado en Render (ej. 3.0) ***
DECISION_THRESHOLD = float(os.environ.get('DECISION_THRESHOLD', 3.0)) 

# Umbral de Funding Rate (Nuevo parámetro de decisión)
FUNDING_THRESHOLD = float(os.environ.get('FUNDING_THRESHOLD', 0.0001))

MAX_RETRIES = 5 

# Variables de estado del bot
bot_state = {
    "configuration": {
        "dry_run": DRY_RUN,
        "use_testnet": USE_TESTNET, 
        "interval": INTERVAL,
        "min_order_usd": MIN_ORDER_USD,
        "pct_of_balance": PCT_OF_BALANCE,
        "sleep_sec": SLEEP_SEC,
        "symbols_list": SYMBOLS_LIST, 
        "decision_threshold": DECISION_THRESHOLD,
        "funding_threshold": FUNDING_THRESHOLD, # Nuevo en el estado
    },
    "current_state": {
        "balances": {"free_USDT": 0, "free_BNB": 0}, 
        "asset_balances": {}, 
        "last_run_utc": None,
        "symbol_data": {} 
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
    # CLIENTE AUTENTICADO (Para órdenes y balances)
    client = Client(API_KEY, API_SECRET)
    # CLIENTE PÚBLICO (Para datos de mercado e info de símbolos)
    public_client = Client("", "") 
    
    # Determinar el entorno de conexión
    if USE_TESTNET:
        connection_target = "TESTNET (SIMULACIÓN)"
        client.API_URL = 'https://testnet.binance.vision/api'
        public_client.API_URL = 'https://testnet.binance.vision/api'
    elif DRY_RUN: 
        connection_target = "PRODUCCIÓN (SIMULACIÓN)"
    else:
        connection_target = "PRODUCCIÓN (DINERO REAL)"

    # Sincronizar el tiempo del cliente con el servidor de Binance para el CLIENTE AUTENTICADO
    client.timestamp_offset = client.get_server_time()['serverTime'] - int(time.time() * 1000) 
    print("✅ Tiempo del servidor sincronizado.")
    
    # Verificar conexión (Solo la parte de autenticación)
    info = client.get_account()
    print(f"✅ Conectado a Binance {connection_target}. Estado de la cuenta:", info['canTrade'])
    
except Exception as e:
    print(f"❌ Error al conectar con Binance. Revise credenciales, entorno (real/testnet) y restricciones geográficas. {e}")
    exit()

# --- 3. FUNCIONES DE ESTRATEGIA Y UTILIDAD ---

# --- NUEVA FUNCIÓN: OBTENER FUNDING RATE ---
def get_funding_rate(symbol):
    """
    Obtiene la Funding Rate más reciente para el símbolo (asumiendo Futures).
    Usamos el cliente PÚBLICO.
    """
    try:
        # Nota: La API de Futures usa un endpoint diferente, pero la librería python-binance lo maneja.
        # Si esta llamada falla, significa que el cliente no está configurado para Futuros,
        # pero para fines de recolección de datos, funcionará si la API_KEY lo permite.
        funding_rate_data = public_client.futures_funding_rate(symbol=symbol, limit=1)
        if funding_rate_data:
            return float(funding_rate_data[0]['fundingRate'])
        return 0.0
    except Exception as e:
        print(f"⚠️ {symbol} - ADVERTENCIA: Fallo al obtener Funding Rate (posiblemente no es Futuros). Usando 0.0: {e}")
        return 0.0

def get_symbol_step_size(symbol):
    """
    Obtiene el 'stepSize' para un símbolo usando el cliente público.
    Esto evita fallos de autenticación en DRY_RUN.
    """
    # 8 decimales es un valor seguro por defecto
    default_step_size = 0.00000001
    
    try:
        # Usamos el cliente PÚBLICO
        info = public_client.get_symbol_info(symbol=symbol)
        
        # Buscamos el filtro LOT_SIZE
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                return float(f['stepSize'])
        
        # Si no encontramos LOT_SIZE, usamos el valor por defecto y advertimos
        print(f"⚠️ {symbol} - ADVERTENCIA: No se encontró el filtro 'LOT_SIZE'. Usando {default_step_size} por defecto.")
        return default_step_size
        
    except Exception as e:
        # Si falla completamente (ej. conexión), usamos el valor por defecto
        print(f"❌ {symbol} - FALLO AL OBTENER 'stepSize' (API ERROR): {e}. Usando {default_step_size} por defecto.")
        return default_step_size

def get_data(symbol):
    """Obtiene datos de velas y calcula indicadores para un símbolo específico, con reintentos."""
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"📊 Obteniendo datos de velas para {symbol} en intervalo {INTERVAL}...")
            # Limit 500 es suficiente para calcular todos los indicadores
            klines = public_client.get_klines(symbol=symbol, interval=INTERVAL, limit=500)
            
            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 
                                              'volume', 'close_time', 'quote_asset_volume', 
                                              'number_of_trades', 'taker_buy_base_asset_volume', 
                                              'taker_buy_quote_asset_volume', 'ignore'])
            df['close'] = pd.to_numeric(df['close'])

            # --- CÁLCULO DE INDICADORES TÉCNICOS ---
            
            # 1. RSI (Índice de Fuerza Relativa)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            # Período estándar de 14, usando com=13
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

            # --- NUEVO CÁLCULO: FUNDING RATE ---
            # Solo obtenemos el valor más reciente ya que no hay historial de klines de Funding Rate
            current_funding_rate = get_funding_rate(symbol)
            df['funding_rate'] = current_funding_rate

            return df 
        
        except BinanceAPIException as e:
            print(f"❌ Error CRÍTICO (Binance API Code {e.code}) al obtener datos para {symbol}: {e}. Saltando el par.")
            return None 
            
        except Exception as e:
            wait_time = 2 ** attempt 
            if attempt < MAX_RETRIES - 1:
                print(f"❌ Error TEMPORAL al obtener datos para {symbol}: {e}. Reintentando en {wait_time} segundos (Intento {attempt + 1}/{MAX_RETRIES}).")
                time.sleep(wait_time)
            else:
                print(f"❌ Error CRÍTICO y persistente al obtener datos para {symbol} después de {MAX_RETRIES} intentos: {e}. Saltando el par.")
                return None 


def get_signal(df, symbol):
    """Genera la señal de trading para el símbolo usando lógica de puntaje, ahora incluyendo Funding Rate."""
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
    
    # --- NUEVA FEATURE: FUNDING RATE ---
    funding_rate = last_row['funding_rate']

    # --- LÓGICA DE PUNTAJE (SIMULACIÓN DE CLASIFICADOR) ---
    buy_score = 0
    sell_score = 0
    
    # 1. Criterio de RSI (Impulso: Comprar si es bajo <40, Vender si es alto >60)
    if rsi < 40:
        buy_score += 1.0
    elif rsi > 60:
        sell_score += 1.0

    # 2. Criterio de Crossover EMA (Tendencia: EMA corta cruza a EMA larga)
    if ema9 > ema21 and prev_row['ema9'] <= prev_row['ema21']:
        buy_score += 2.0
    elif ema9 < ema21 and prev_row['ema9'] >= prev_row['ema21']:
        sell_score += 2.0
    
    # 3. Criterio de MACD (Momento: MACD cruza la línea de señal)
    if macd_line > macd_signal and prev_row['macd_line'] <= prev_row['macd_signal']:
        buy_score += 1.5
    elif macd_line < macd_signal and prev_row['macd_line'] >= prev_row['macd_signal']:
        sell_score += 1.5

    # 4. Criterio de Bandas de Bollinger (Volatilidad y Extremo: Comprar en la banda inferior, Vender en la superior)
    if current_price < bollinger_lower:
        buy_score += 1.0
    elif current_price > bollinger_upper:
        sell_score += 1.0
        
    # --- 5. NUEVO CRITERIO: FUNDING RATE (Sentimiento Extremo) ---
    # Funding Rate alta (positiva) = Mucha gente en LONG, posible sobrecalentamiento (Señal de VENTA)
    if funding_rate > FUNDING_THRESHOLD:
        sell_score += 1.5
    # Funding Rate baja (negativa) = Mucha gente en SHORT, posible corrección terminada (Señal de COMPRA)
    elif funding_rate < -FUNDING_THRESHOLD:
        buy_score += 1.5


    # --- EVALUACIÓN DE LA DECISIÓN ---
    
    decision_score = max(buy_score, sell_score)
    
    # La señal se activa si el puntaje supera o iguala el umbral
    if buy_score >= DECISION_THRESHOLD and buy_score > sell_score:
        signal = "BUY"
    elif sell_score >= DECISION_THRESHOLD and sell_score > buy_score:
        signal = "SELL"
    else:
        signal = "HOLD"
        
    # Guardamos los datos de la última ejecución incluyendo el Funding Rate
    bot_state["current_state"]["symbol_data"][symbol] = {
        "last_signal": signal, 
        "decision_score": decision_score,
        "current_price": current_price,
        "funding_rate": funding_rate
    }

    return signal, decision_score

# --- 4. FUNCIONES DE EJECUCIÓN (SIN CAMBIOS EN ESTA VERSIÓN) ---

def update_balances():
    """Actualiza los balances de USDT, BNB y de todos los activos vigilados."""
    
    if DRY_RUN:
        # Lógica de saldo simulado
        initial_usdt = 1000.0
        # Solo inicializa el saldo a 1000 si no hay historial de trades, si no, usa el saldo actual simulado.
        if not bot_state["trade_history"] and bot_state["current_state"]["balances"]["free_USDT"] == 0.0:
            bot_state["current_state"]["balances"]["free_USDT"] = initial_usdt
        
        # Inicializa los saldos de las monedas base
        for symbol in SYMBOLS_LIST:
            base_asset = symbol.replace("USDT", "")
            if base_asset not in bot_state["current_state"]["asset_balances"]:
                bot_state["current_state"]["asset_balances"][base_asset] = 0.0
        
        print(f"✅ Balances de la cuenta actualizados (SIMULACIÓN). USDT disponible: {bot_state['current_state']['balances']['free_USDT']:.2f}")
        return
        
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
        
        print(f"✅ Balances de la cuenta actualizados (REAL). USDT disponible: {bot_state['current_state']['balances']['free_USDT']:.2f}")

    except Exception as e:
        print(f"❌ Error al actualizar balances (REQUIERE AUTENTICACIÓN): {e}")

def execute_order(symbol, signal, current_price):
    """Ejecuta una orden de COMPRA o VENTA si DRY_RUN es False para un símbolo específico."""
    
    base_asset = symbol.replace("USDT", "") 
    usdt_free_total = bot_state["current_state"]["balances"]["free_USDT"]
    base_free = bot_state["current_state"]["asset_balances"].get(base_asset, 0.0)
    
    # Obtener el tamaño de paso (stepSize)
    step_size = get_symbol_step_size(symbol)

    if signal == "BUY":
        # Calcula el capital a gastar usando el porcentaje del saldo total de USDT
        usd_to_spend = usdt_free_total * PCT_OF_BALANCE
        
        # Límite de gasto
        if usd_to_spend > usdt_free_total:
             usd_to_spend = usdt_free_total
        
        # Asegura que la orden sea mayor que el mínimo de Binance (ej. $10.5)
        if usd_to_spend < MIN_ORDER_USD:
            # Si el saldo es bajo, el bot no puede comprar
            if usdt_free_total < MIN_ORDER_USD:
                 return
            # Si el monto calculado es menor al mínimo, usamos el mínimo si el saldo lo permite
            usd_to_spend = MIN_ORDER_USD

        quantity = usd_to_spend / current_price
        
        # Redondeo de la cantidad usando el step_size obtenido
        quantity = np.floor(quantity / step_size) * step_size 

        if DRY_RUN:
            # SIMULACIÓN DE ORDEN (Ajusta los balances en el estado local)
            print(f"💰 {symbol} - BUY (Simulado): Compraría {quantity:.2f} {base_asset} a {current_price:.4f} USD. (Costo: {usd_to_spend:.2f})")
            # ACTUALIZACIÓN DE SALDO SIMULADA: 
            bot_state["current_state"]["balances"]["free_USDT"] -= usd_to_spend
            bot_state["current_state"]["asset_balances"][base_asset] = bot_state["current_state"]["asset_balances"].get(base_asset, 0.0) + quantity
            # Añadimos la orden al historial para verla en /state
            bot_state["trade_history"].append({"time": bot_state["current_state"]["last_run_utc"], "symbol": symbol, "type": "BUY (SIMULADO)", "quantity": quantity, "price": current_price, "cost_usd": usd_to_spend, "status": "FILLED"})
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
        # Solo vende si tiene algo de esa moneda
        if base_free <= 0:
            return
            
        quantity = base_free
        
        # Redondeo de la cantidad usando el step_size obtenido
        quantity = np.floor(quantity / step_size) * step_size 
        
        revenue = quantity * current_price
        
        # Asegura que la cantidad a vender sea suficiente para el mínimo
        if revenue < MIN_ORDER_USD:
            print(f"⚠️ {symbol} - VENTA: Saldo de {base_asset} es muy bajo para vender. (Valor: {revenue:.2f} USD)")
            return
            
        if DRY_RUN:
            # SIMULACIÓN DE ORDEN
            print(f"💸 {symbol} - SELL (Simulado): Vendería {quantity:.2f} {base_asset} a {current_price:.4f} USD. (Ingreso: {revenue:.2f})")
            bot_state["current_state"]["balances"]["free_USDT"] += revenue
            bot_state["current_state"]["asset_balances"][base_asset] = 0.0
            bot_state["trade_history"].append({"time": bot_state["current_state"]["last_run_utc"], "symbol": symbol, "type": "SELL (SIMULADO)", "quantity": quantity, "price": current_price, "revenue_usd": revenue, "status": "FILLED"})
        else:
            # ORDEN REAL DE BINANCE (Requiere cliente autenticado)
            try:
                print(f"💸 {symbol} - SELL (REAL): Enviando orden de mercado para vender {quantity:.2f} {base_asset}...")
                order = client.create_order(symbol=symbol, side='SELL', type='MARKET', quantity=quantity)
                print(f"✅ {symbol} - Orden de VENTA ejecutada. Status: {order['status']}")
                bot_state["trade_history"].append({"time": bot_state["current_state"]["last_run_utc"], "symbol": symbol, "type": "SELL", "quantity": quantity, "price": current_price, "status": order['status']})
            except Exception as e:
                 print(f"❌ {symbol} - FALLO AL EJECUTAR ORDEN REAL DE VENTA (REQUIERE AUTENTICACIÓN): {e}")


# --- 5. BUCLE PRINCIPAL DEL BOT (Thread) ---

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
            
            # Bucle que procesa cada símbolo en la lista
            for symbol in SYMBOLS_LIST:
                # 1. Obtener datos (Usa public_client, ahora incluye Funding Rate)
                df = get_data(symbol) 
                if df is None:
                    print(f"⚠️ {symbol}: No se obtuvieron datos, saltando análisis para este par.")
                    continue

                # 2. Generar señal
                signal, decision_score = get_signal(df, symbol)
                
                # Accede al estado para obtener los datos de la última ejecución
                symbol_data = bot_state["current_state"]["symbol_data"].get(symbol, {})
                current_price = symbol_data.get("current_price", df.iloc[-1]['close'])
                funding_rate = symbol_data.get("funding_rate", 0.0)


                print(f"📊 {symbol} | Señal: {signal} (Puntaje: {decision_score:.1f}) | Precio: {current_price:.4f} | Funding Rate: {funding_rate:.5f}")
                
                # 3. Ejecutar orden
                if signal != "HOLD":
                    # Solo COMPRA si tiene saldo de USDT
                    if signal == "BUY" and bot_state["current_state"]["balances"]["free_USDT"] > 0:
                        execute_order(symbol, signal, current_price)
                    # Solo VENDE si tiene algo del activo base
                    elif signal == "SELL" and bot_state["current_state"]["asset_balances"].get(symbol.replace("USDT", ""), 0.0) > 0:
                        execute_order(symbol, signal, current_price)
                    elif signal == "SELL" and bot_state["current_state"]["asset_balances"].get(symbol.replace("USDT", ""), 0.0) <= 0:
                         print(f"❌ {symbol} - VENTA OMITIDA: No hay {symbol.replace('USDT', '')} en el balance simulado.")
                    
            # Actualizamos todos los balances al final del ciclo
            update_balances()
            print(f"🟢 Ciclo completado para todos los pares. Volviendo a dormir por {SLEEP_SEC} segundos.")

        except BinanceAPIException as e:
            print(f"❌ ERROR DE BINANCE (API): {e}")
        except Exception as e:
            print(f"❌ ERROR INESPERADO en el ciclo de trading: {e}")
            
        time.sleep(SLEEP_SEC)

# --- 6. RUTAS FLASK (API) ---

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

# --- 7. INICIO DEL SERVIDOR Y DEL THREAD ---

if not bot_thread_running:
    try:
        trading_thread = threading.Thread(target=bot_loop)
        trading_thread.start()
        print("🌐 Hilo de trading iniciado con éxito en segundo plano.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: No se pudo iniciar el hilo de trading: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
