import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data(ttl=86400, show_spinner=False)
def obtener_historico_cache(ticker, periodo="2y"):
    """
    Descarga y cachea los datos históricos para evitar llamadas innecesarias a la API.
    """
    import yfinance as yf
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return None
        return data
    except Exception:
        return None

def calcular_indicadores_historicos(df_slice):
    """
    Calcula indicadores técnicos usando ÚNICAMENTE los datos hasta la fecha de corte.
    Evita estrictamente Look-Ahead Bias.
    """
    if len(df_slice) < 50:
        return {}
    
    close = df_slice['Close'].values
    precio_actual = close[-1]
    
    # Medias Móviles
    ma20 = np.mean(close[-20:]) if len(close) >= 20 else precio_actual
    ma50 = np.mean(close[-50:]) if len(close) >= 50 else precio_actual
    ma200 = np.mean(close[-200:]) if len(close) >= 200 else precio_actual
    
    # RSI (14 periodos)
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1e-6
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    
    # Volatilidad (20 días)
    returns = np.diff(np.log(close[-21:])) if len(close) >= 21 else np.array([0])
    volatilidad = np.std(returns) * np.sqrt(252) * 100
    
    return {
        "precio": precio_actual,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi,
        "volatilidad": volatilidad
    }

def simular_market_ai_score_historico(tec_data, es_metal=False):
    """
    Simula la puntuación y dirección de MARKET AI en un punto pasado
    sin acceder a datos del futuro.
    """
    if not tec_data:
        return 50.0, "🟡 NEUTRAL", 50
    
    score = 50.0
    
    # Lógica de señales pasadas
    precio = tec_data["precio"]
    if precio > tec_data["ma20"]: score += 10
    else: score -= 10
    
    if tec_data["ma20"] > tec_data["ma50"]: score += 15
    else: score -= 15
    
    if precio > tec_data["ma200"]: score += 10
    else: score -= 10
    
    if tec_data["rsi"] < 30: score += 10 (Sobrevendido)
    elif tec_data["rsi"] > 70: score -= 10 (Sobrecomprado)
    
    score = float(np.clip(score, 0.0, 100.0))
    
    if score >= 60:
        direccion = "🟢 ALCISTA"
        confianza = int(np.clip(50 + (score - 60) * 1.2, 50, 95))
    elif score <= 40:
        direccion = "🔴 BAJISTA"
        confianza = int(np.clip(50 + (40 - score) * 1.2, 50, 95))
    else:
        direccion = "🟡 NEUTRAL"
        confianza = 50
        
    return score, direccion, confianza

def ejecutar_backtest_engine(ticker, horizonte_dias, periodo_meses, es_metal=False):
    """
    Motor principal de backtesting iterativo.
    """
    dias_totales = periodo_meses * 30
    periodo_yf = "2y" if dias_totales <= 365 else "5y"
    
    df = obtener_historico_cache(ticker, periodo=periodo_yf)
    if df is None or len(df) < (100 + horizonte_dias):
        return None
    
    # Filtrar fechas para dejar margen del horizonte al final
    df = df.sort_index()
    total_barras = len(df)
    inicio_idx = max(200, total_barras - dias_totales - horizonte_dias)
    fin_idx = total_barras - horizonte_dias
    
    if inicio_idx >= fin_idx:
        return None
    
    registros = []
    paso = max(1, horizonte_dias // 2) # Frecuencia de muestreo
    
    for i in range(inicio_idx, fin_idx, paso):
        df_slice = df.iloc[:i+1] # Estricta ventana histórica
        
        fecha_senal = df_slice.index[-1].strftime("%Y-%m-%d")
        precio_inicial = df_slice['Close'].iloc[-1]
        
        # Datos del futuro para evaluación POSTERIOR
        precio_final = df['Close'].iloc[i + horizonte_dias]
        rentabilidad = ((precio_final - precio_inicial) / precio_inicial) * 100.0
        
        tec_data = calcular_indicadores_historicos(df_slice)
        score, direccion, confianza = simular_market_ai_score_historico(tec_data, es_metal)
        
        # Evaluación del acierto
        if direccion == "🟢 ALCISTA":
            resultado = "✅ Acierto" if rentabilidad > 0.5 else "❌ Fallo"
        elif direccion == "🔴 BAJISTA":
            resultado = "✅ Acierto" if rentabilidad < -0.5 else "❌ Fallo"
        else:
            resultado = "✅ Acierto" if abs(rentabilidad) <= 2.0 else "⚪ Neutral"
            
        registros.append({
            "fecha": fecha_senal,
            "ticker": ticker,
            "score": round(score, 1),
            "direccion": direccion,
            "confianza": confianza,
            "precio_inicial": round(precio_inicial, 2),
            "precio_final": round(precio_final, 2),
            "rentabilidad": round(rentabilidad, 2),
            "resultado": resultado,
            "horizonte": f"{horizonte_dias}d",
            "ma20_gt_ma50": tec_data.get("ma20", 0) > tec_data.get("ma50", 0),
            "precio_gt_ma200": precio_inicial > tec_data.get("ma200", 0)
        })
        
    df_res = pd.DataFrame(registros)
    
    # Rentabilidad Buy & Hold del periodo evaluado
    p_inicio_periodo = df['Close'].iloc[inicio_idx]
    p_fin_periodo = df['Close'].iloc[fin_idx]
    ret_buy_hold = ((p_fin_periodo - p_inicio_periodo) / p_inicio_periodo) * 100.0
    
    return df_res, round(ret_buy_hold, 2)
