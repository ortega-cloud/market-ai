import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400, show_spinner=False)
def obtener_historico_cache(ticker, periodo="5y"):
    """Descarga y cachea los datos históricos usando yfinance."""
    import yfinance as yf
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data is None or data.empty or len(data) < 50:
            return None
        return data
    except Exception:
        return None

def calcular_indicadores_historicos(df_slice):
    """Calcula indicadores técnicos únicamente con datos pasados (sin Look-Ahead Bias)."""
    if len(df_slice) < 20:
        return {}
    
    close = df_slice['Close'].values
    precio_actual = float(close[-1])
    
    # Medias Móviles adaptables al tamaño disponible
    ma20 = float(np.mean(close[-20:]))
    ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else ma20
    ma200 = float(np.mean(close[-200:])) if len(close) >= 200 else ma50
    
    # RSI (14 periodos)
    delta = np.diff(close)
    if len(delta) >= 14:
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = float(100 - (100 / (1 + rs)))
    else:
        rsi = 50.0
    
    return {
        "precio": precio_actual,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi
    }

def simular_market_ai_score_historico(tec_data, es_metal=False):
    """Simula el score predictivo pasado."""
    if not tec_data:
        return 50.0, "🟡 NEUTRAL", 50
    
    score = 50.0
    precio = tec_data["precio"]
    
    if precio > tec_data["ma20"]: score += 10
    else: score -= 10
    
    if tec_data["ma20"] > tec_data["ma50"]: score += 15
    else: score -= 15
    
    if precio > tec_data["ma200"]: score += 10
    else: score -= 10
    
    if tec_data["rsi"] < 30: score += 10
    elif tec_data["rsi"] > 70: score -= 10
    
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

def ejecutar_backtest_engine(ticker, periodo_meses, horizonte_dias, es_metal=False):
    """Motor principal de backtesting ajustado y flexible."""
    df = obtener_historico_cache(ticker, periodo="5y")
    
    if df is None or len(df) < 50:
        return None
    
    df = df.sort_index()
    total_barras = len(df)
    
    # Cálculo flexible de barras
    barras_solicitadas = int((periodo_meses / 12) * 252)
    horizonte_dias = int(horizonte_dias)
    
    fin_idx = total_barras - horizonte_dias
    if fin_idx <= 20:
        return None
        
    inicio_idx = max(20, fin_idx - barras_solicitadas)
    if inicio_idx >= fin_idx:
        inicio_idx = max(20, fin_idx - 50)
    
    registros = []
    paso = max(1, horizonte_dias // 2)
    
    for i in range(inicio_idx, fin_idx, paso):
        df_slice = df.iloc[:i+1]
        
        fecha_senal = df_slice.index[-1].strftime("%Y-%m-%d")
        precio_inicial = float(df_slice['Close'].iloc[-1])
        precio_final = float(df['Close'].iloc[i + horizonte_dias])
        
        rentabilidad = ((precio_final - precio_inicial) / precio_inicial) * 100.0
        
        tec_data = calcular_indicadores_historicos(df_slice)
        score, direccion, confianza = simular_market_ai_score_historico(tec_data, es_metal)
        
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
        
    if not registros:
        return None
        
    df_res = pd.DataFrame(registros)
    
    p_inicio_periodo = float(df['Close'].iloc[inicio_idx])
    p_fin_periodo = float(df['Close'].iloc[fin_idx])
    ret_buy_hold = ((p_fin_periodo - p_inicio_periodo) / p_inicio_periodo) * 100.0
    
    return df_res, round(ret_buy_hold, 2)
