import os
import time
import threading
import json
import pandas as pd
from flask import Flask, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException

# --- 1. CONFIGURACIÓN Y VARIABLES GLOBALES ---

# Cargar variables de entorno (Render)
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
DRY_RUN = os.environ.get('DRY_RUN', 'true').lower() == 'true'

# Parámetros de la Estrategia (Recuperados de las variables de entorno)
SYMBOL = os.environ.get('SYMBOL', 'TRXUSDT')
INTERVAL = os.environ.get('INTERVAL', '15m')
PCT_OF_BALANCE = float(os.environ.get('PCT_OF_BALANCE', 0.02))
SLEEP_SEC = int(os.environ.get('SLEEP_SEC', 300))
MIN_ORDER_USD = float(os.environ.get('MIN_ORDER_USD', 10))

# Variables de estado del bot
bot_state = {
    "configuration": {
        "dry_run": DRY_RUN,
        "interval": INTERVAL,
        "min_order_usd": MIN_ORDER_USD,
        "pct_of_balance": PCT_OF_BALANCE,
        "sleep_sec": SLEEP_SEC,
        "symbol": SYMBOL,
    },
    "current_state": {
        "balances": {"free_TRX": 0, "free_USDT": 0},
        "last_run_utc": None,
        "last_signal": None
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
    # Conexión al cliente de Binance
    client = Client(API_KEY, API_SECRET)
    
    # Forzar el uso de Testnet
    client.API_URL = 'https://testnet.binance.vision/api'
    
    # Verificar conexión
    info = client.get_account()
    print("✅ Conectado a Binance TESTNET. Estado de la cuenta:", info['canTrade'])
    
except Exception as e:
    print(f"❌ Error al conectar con Binance Testnet. Revise credenciales y región: {e}")
    exit()

# --- 3. FUNCIONES DE ESTRATEGIA ---

def get_data():
    """Obtiene datos de velas y calcula indicadores (RSI y MA)."""
    try:
        # Obtener 500 velas (suficiente para RSI y MAs)
        print(f"📊 Obteniendo datos de velas para {SYMBOL} en intervalo {INTERVAL}...")
        klines = client.get_historical_klines(SYMBOL, INTERVAL, "500 ago UTC")
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 
                                          'volume', 'close_time', 'quote_asset_volume', 
                                          'number_of_trades', 'taker_buy_base_asset_volume', 
                                          'taker_buy_quote_asset_volume', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])

        # Cálculo de Indicadores
        # 1. RSI (Período 14, clásico)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 2. Medias Móviles (Estrategia 9/21 Ema Crossover)
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

        return df

    except Exception as e:
        print(f"❌ Error al obtener datos o calcular indicadores: {e}")
        return None

def get_signal(df):
    """Genera la señal de trading (BUY, SELL, HOLD)."""
    if df is None or len(df) < 21:
        return "HOLD"

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    rsi = last_row['rsi']
    ema9 = last_row['ema9']
    ema21 = last_row['ema21']
    prev_ema9 = prev_row['ema9']
    prev_ema21 = prev_row['ema21']
    current_price = last_row['close']

    # Condición de COMPRA (EMA crossover + Filtro RSI)
    buy_condition = (ema9 > ema21 and prev_ema9 <= prev_ema21) and (rsi < 70)

    # Condición de VENTA (EMA crossover + Filtro RSI)
    sell_condition = (ema9 < ema21 and prev_ema9 >= prev_ema21) and (rsi > 30)

    print(f"⚙️ Cálculo: Precio: {current_price:.4f}, RSI: {rsi:.2f}, EMA9: {ema9:.4f}, EMA21: {ema21:.4f}")

    if buy_condition:
        return "BUY"
    elif sell_condition:
        return "SELL"
    else:
        return "HOLD"

# --- 4. FUNCIONES DE EJECUCIÓN ---

def update_balances():
    """Actualiza los balances de USDT y del SYMBOL de la cuenta de Testnet."""
    try:
        base_asset = SYMBOL.replace("USDT", "") # Ejemplo: TRX
        
        usdt_balance = client.get_asset_balance(asset='USDT')
        base_balance = client.get_asset_balance(asset=base_asset)

        bot_state["current_state"]["balances"]["free_USDT"] = float(usdt_balance['free'])
        bot_state["current_state"]["balances"][f"free_{base_asset}"] = float(base_balance['free'])

    except Exception as e:
        print(f"❌ Error al actualizar balances: {e}")

def execute_order(signal, current_price):
    """Ejecuta una orden de COMPRA o VENTA si DRY_RUN es False."""
    
    # 1. Preparar la moneda base (ej. TRX)
    base_asset = SYMBOL.replace("USDT", "") 
    
    # 2. Obtener balances actualizados
    update_balances()
    usdt_free = bot_state["current_state"]["balances"]["free_USDT"]
    base_free = bot_state["current_state"]["balances"][f"free_{base_asset}"]

    if signal == "BUY":
        # Calcular la cantidad a comprar
        usd_to_spend = usdt_free * PCT_OF_BALANCE
        if usd_to_spend < MIN_ORDER_USD:
            print(f"⚠️ COMPRA: Saldo insuficiente o bajo para orden de {MIN_ORDER_USD} USD.")
            return

        quantity = usd_to_spend / current_price
        
        # Obtener reglas de redondeo de Binance
        info = client.get_symbol_info(symbol=SYMBOL)
        step_size = float(info['filters'][2]['stepSize'])
        quantity = round(quantity / step_size) * step_size

        if DRY_RUN:
            print(f"💰 BUY (Simulado): Compraría {quantity:.2f} {base_asset} a {current_price:.4f} USD.")
            # Simular actualización del balance para el estado
            bot_state["current_state"]["balances"]["free_USDT"] -= usd_to_spend
            bot_state["current_state"]["balances"][f"free_{base_asset}"] += quantity
        else:
            print(f"💰 BUY (Real): Enviando orden para comprar {quantity:.2f} {base_asset}...")
            # Aquí se ejecutaría la orden de compra real:
            # order = client.create_order(symbol=SYMBOL, side='BUY', type='MARKET', quantity=quantity)
            # bot_state["trade_history"].append({"time": bot_state["current_state"]["last_run_utc"], "type": "BUY", "quantity": quantity, "price": current_price, "status": "executed"})


    elif signal == "SELL":
        # Calcular la cantidad a vender (se vende todo lo disponible)
        quantity = base_free
        
        # Obtener reglas de redondeo de Binance
        info = client.get_symbol_info(symbol=SYMBOL)
        step_size = float(info['filters'][2]['stepSize'])
        quantity = round(quantity / step_size) * step_size
        
        if (quantity * current_price) < MIN_ORDER_USD:
            print(f"⚠️ VENTA: Saldo de {base_asset} es muy bajo para vender. (Valor: {quantity * current_price:.2f} USD)")
            return
            
        if DRY_RUN:
            print(f"💸 SELL (Simulado): Vendería {quantity:.2f} {base_asset} a {current_price:.4f} USD.")
            # Simular actualización del balance para el estado
            bot_state["current_state"]["balances"]["free_USDT"] += quantity * current_price
            bot_state["current_state"]["balances"][f"free_{base_asset}"] = 0.0
        else:
            print(f"💸 SELL (Real): Enviando orden para vender {quantity:.2f} {base_asset}...")
            # Aquí se ejecutaría la orden de venta real:
            # order = client.create_order(symbol=SYMBOL, side='SELL', type='MARKET', quantity=quantity)
            # bot_state["trade_history"].append({"time": bot_state["current_state"]["last_run_utc"], "type": "SELL", "quantity": quantity, "price": current_price, "status": "executed"})


# --- 5. BUCLE PRINCIPAL DEL BOT (Thread) ---

def bot_loop():
    """El bucle infinito que corre en segundo plano."""
    global bot_thread_running
    bot_thread_running = True
    
    print(f"🤖 Bot iniciado en TESTNET (DRY_RUN={DRY_RUN}, SYMBOL={SYMBOL}).")
    
    while True:
        # A. MANEJO DE ERRORES: CRÍTICO para Render
        try:
            # 1. Obtener y procesar datos
            df = get_data()
            if df is None:
                raise Exception("No se pudieron obtener datos, saltando ciclo.")

            # 2. Generar señal
            signal = get_signal(df)
            current_price = df.iloc[-1]['close']
            
            # 3. Actualizar estado global
            bot_state["current_state"]["last_signal"] = signal
            bot_state["current_state"]["last_run_utc"] = pd.Timestamp.now(tz='UTC').isoformat()
            
            # 4. Ejecutar si la señal no es HOLD
            if signal != "HOLD":
                execute_order(signal, current_price)
            else:
                update_balances() # Siempre actualizar balances aunque sea HOLD
                
            print(f"📊 Señal: {signal}. Volviendo a dormir por {SLEEP_SEC} segundos.")

        # B. Capturar CUALQUIER error y continuar el ciclo
        except BinanceAPIException as e:
            print(f"❌ ERROR DE BINANCE (API): {e}")
        except Exception as e:
            # Esto atrapa errores de Pandas, Redondeo o cualquier fallo inesperado
            print(f"❌ ERROR INESPERADO en el ciclo de trading: {e}")
            
        # 5. Esperar
        time.sleep(SLEEP_SEC)

# --- 6. RUTAS FLASK (API) ---

@app.route('/')
def home():
    """Ruta principal para verificar que el servicio está activo."""
    return jsonify({
        "status": "ok",
        "message": "Bot de Trading Activo. Ver /state para detalles.",
        "dry_run": DRY_RUN
    })

@app.route('/state')
def get_state():
    """Ruta para obtener el estado actual del bot en formato JSON."""
    return jsonify(bot_state)

# --- 7. INICIO DEL SERVIDOR Y DEL THREAD ---

if __name__ == '__main__':
    # Iniciar el hilo de trading en segundo plano una única vez
    if not bot_thread_running:
        trading_thread = threading.Thread(target=bot_loop)
        trading_thread.start()
        print("🌐 Hilo de trading iniciado.")
        
    # El servidor Gunicorn (desde el Procfile) se encargará de ejecutar Flask.
    # Esta línea se ejecuta en local, Gunicorn la ignora al ser el punto de entrada.
    # app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
