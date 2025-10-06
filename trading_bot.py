import os
import time
import threading
import json
import pandas as pd
from flask import Flask, jsonify
from binance.client import Client
from binance.exceptions import BinanceAPIException
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
# VALOR ACTUALIZADO: TRXUSDT y XRPUSDT por defecto
SYMBOLS_LIST_STR = os.environ.get('SYMBOLS_LIST', 'TRXUSDT,XRPUSDT').replace(" ", "")
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
    # Conexión al cliente de Binance
    client = Client(API_KEY, API_SECRET)
    
    # Determinar el entorno de conexión
    if USE_TESTNET:
        connection_target = "TESTNET (SIMULACIÓN)"
        # Si USE_TESTNET es verdadero, forzamos la URL de Testnet
        client.API_URL = 'https://testnet.binance.vision/api'
    elif DRY_RUN: 
        connection_target = "PRODUCCIÓN (SIMULACIÓN)"
        # Si es DRY_RUN pero no es Testnet, usamos el API de producción solo para leer datos.
    else:
        connection_target = "PRODUCCIÓN (DINERO REAL)"
    
    # Verificar conexión (Esto fallará si las claves no coinciden con el entorno)
    info = client.get_account()
    print(f"✅ Conectado a Binance {connection_target}. Estado de la cuenta:", info['canTrade'])
    
except Exception as e:
    # Mensaje de error más detallado sobre la conexión
    print(f"❌ Error al conectar con Binance. Revise credenciales, entorno (real/testnet) y restricciones geográficas. {e}")
    exit()

# --- 3. FUNCIONES DE ESTRATEGIA (MODIFICADA PARA ACEPTAR SYMBOL) ---

def get_data(symbol):
    """Obtiene datos de velas y calcula indicadores para un símbolo específico, con reintentos."""
    
    # Lógica de reintento para mejorar la resiliencia contra errores de conexión/servidor
    for attempt in range(MAX_RETRIES):
        try:
            print(f"📊 Obteniendo datos de velas para {symbol} en intervalo {INTERVAL}...")
            # Intento de obtener datos de velas
            klines = client.get_historical_klines(symbol, INTERVAL, "500 ago UTC")
            
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
            
        except Exception as e:
            wait_time = 2 ** attempt # Backoff exponencial (1s, 2s, 4s, 8s...)
            if attempt < MAX_RETRIES - 1:
                print(f"❌ Error TEMPORAL al obtener datos para {symbol}: {e}. Reintentando en {wait_time} segundos (Intento {attempt + 1}/{MAX_RETRIES}).")
                time.sleep(wait_time)
            else:
                print(f"❌ Error CRÍTICO y persistente al obtener datos para {symbol} después de {MAX_RETRIES} intentos: {e}. Saltando el par.")
                return None # Fallo final, devuelve None


def get_signal(df, symbol):
# ... (rest of the function is the same)
