import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400, show_spinner=False)
@st.cache_data(ttl=86400, show_spinner=False)
def obtener_historico_cache(ticker, periodo="5y"):
    """
    Descarga y cachea los datos históricos usando yfinance.
    Soporta errores devolviendo un mensaje explícito en lugar de fallar en silencio.
    """
    import yfinance as yf
    try:
        # CORRECCIÓN: Se usa period=periodo para coincidir con el nombre del argumento
        data = yf.Ticker(ticker).history(period=periodo)
        if data is None or data.empty:
            return None, f"No se encontraron datos para el ticker '{ticker}'."
        if len(data) < 20:
            return None, f"Datos insuficientes para '{ticker}': solo hay {len(data)} sesiones disponibles (se requieren al menos 20)."
        return data, None
    except Exception as e:
        return None, f"Error descargando histórico para {ticker}: {str(e)}"

def calcular_indicadores_historicos(df_slice):
    """
    Calcula indicadores técnicos usando ÚNICAMENTE datos pasados (evita Look-Ahead Bias).
    Permite degradación elegante si no hay 200 sesiones disponibles.
    """
    len_slice = len(df_slice)
    if len_slice < 5:
        return {}
    
    close = df_slice['Close'].values
    precio_actual = float(close[-1])
    
    # Medias Móviles adaptables al histórico disponible
    ma20 = float(np.mean(close[-20:])) if len_slice >= 20 else precio_actual
    ma50 = float(np.mean(close[-50:])) if len_slice >= 50 else ma20
    ma200 = float(np.mean(close[-200:])) if len_slice >= 200 else ma50
    
    # RSI (14 periodos)
    if len_slice >= 15:
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = float(100 - (100 / (1 + rs)))
    else:
        rsi = 50.0
        
    # Momentum (10 días)
    if len_slice >= 11:
        momentum = float(((close[-1] - close[-11]) / close[-11]) * 100.0)
    else:
        momentum = 0.0
    
    return {
        "precio": precio_actual,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi,
        "momentum": momentum,
        "tiene_ma200": len_slice >= 200
    }

def simular_market_ai_score_historico(tec_data, es_metal=False):
    """
    Calcula la puntuación (0-100), dirección y confianza del modelo
    con datos disponibles hasta la fecha de corte.
    """
    if not tec_data:
        return 50.0, "🟡 NEUTRAL", 50
    
    score = 50.0
    precio = tec_data["precio"]
    
    # Componentes del Score
    if precio > tec_data["ma20"]: score += 10
    else: score -= 10
    
    if tec_data["ma20"] > tec_data["ma50"]: score += 15
    else: score -= 15
    
    if tec_data["tiene_ma200"]:
        if precio > tec_data["ma200"]: score += 10
        else: score -= 10
        
    if tec_data["rsi"] < 30: score += 10 # Sobrevendido -> Oportunidad
    elif tec_data["rsi"] > 70: score -= 10 # Sobrecomprado
    
    if tec_data["momentum"] > 0: score += 5
    else: score -= 5
    
    score = float(np.clip(score, 0.0, 100.0))
    
    if score >= 60:
        direccion = "🟢 ALCISTA"
        confianza = int(np.clip(50 + (score - 60) * 1.15, 50, 95))
    elif score <= 40:
        direccion = "🔴 BAJISTA"
        confianza = int(np.clip(50 + (40 - score) * 1.15, 50, 95))
    else:
        direccion = "🟡 NEUTRAL"
        confianza = 50
        
    return score, direccion, confianza

def calcular_metricas_backtest(df_res, ret_buy_hold):
    """Calcula el conjunto completo de métricas cuantitativas y agrupaciones."""
    if df_res is None or df_res.empty:
        return {}
        
    tot = len(df_res)
    aciertos = len(df_res[df_res["resultado"] == "✅ Acierto"])
    fallos = len(df_res[df_res["resultado"] == "❌ Fallo"])
    tasa_acierto = (aciertos / tot) * 100.0 if tot > 0 else 0.0
    
    ret_media = float(df_res["rentabilidad"].mean())
    mejor_res = float(df_res["rentabilidad"].max())
    peor_res = float(df_res["rentabilidad"].min())
    ret_acum = float(df_res["rentabilidad"].sum())
    
    # Desglose por dirección
    alcistas = df_res[df_res["direccion"] == "🟢 ALCISTA"]
    bajistas = df_res[df_res["direccion"] == "🔴 BAJISTA"]
    neutrales = df_res[df_res["direccion"] == "🟡 NEUTRAL"]
    
    prec_alcista = (len(alcistas[alcistas["resultado"] == "✅ Acierto"]) / len(alcistas) * 100) if len(alcistas) > 0 else 0.0
    prec_bajista = (len(bajistas[bajistas["resultado"] == "✅ Acierto"]) / len(bajistas) * 100) if len(bajistas) > 0 else 0.0
    prec_neutral = (len(neutrales[neutrales["resultado"] == "✅ Acierto"]) / len(neutrales) * 100) if len(neutrales) > 0 else 0.0

    # Agrupación por Rango de Confianza
    bins_conf = [49, 59, 69, 79, 89, 100]
    labels_conf = ["50-59%", "60-69%", "70-79%", "80-89%", "90-100%"]
    df_res['rango_conf'] = pd.cut(df_res['confianza'], bins=bins_conf, labels=labels_conf)
    
    tabla_confianza = df_res.groupby('rango_conf', observed=False).agg(
        Predicciones=('resultado', 'count'),
        Aciertos=('resultado', lambda x: (x == "✅ Acierto").sum()),
        Tasa_Acierto=('resultado', lambda x: f"{((x == '✅ Acierto').sum()/len(x)*100):.1f}%" if len(x)>0 else "0%"),
        Rentabilidad_Media=('rentabilidad', lambda x: f"{x.mean():+.2f}%" if len(x)>0 else "0.00%")
    )

    # Agrupación por Rango de Score
    bins_score = [-1, 39, 54, 69, 84, 100]
    labels_score = ["0-39", "40-54", "55-69", "70-84", "85-100"]
    df_res['rango_score'] = pd.cut(df_res['score'], bins=bins_score, labels=labels_score)
    
    tabla_score = df_res.groupby('rango_score', observed=False).agg(
        Predicciones=('resultado', 'count'),
        Aciertos=('resultado', lambda x: (x == "✅ Acierto").sum()),
        Tasa_Acierto=('resultado', lambda x: f"{((x == '✅ Acierto').sum()/len(x)*100):.1f}%" if len(x)>0 else "0%"),
        Rentabilidad_Media=('rentabilidad', lambda x: f"{x.mean():+.2f}%" if len(x)>0 else "0.00%")
    )

    # Agrupación por Horizonte
    tabla_horizonte = df_res.groupby('horizonte', observed=False).agg(
        Predicciones=('resultado', 'count'),
        Aciertos=('resultado', lambda x: (x == "✅ Acierto").sum()),
        Tasa_Acierto=('resultado', lambda x: f"{((x == '✅ Acierto').sum()/len(x)*100):.1f}%" if len(x)>0 else "0%"),
        Rentabilidad_Media=('rentabilidad', lambda x: f"{x.mean():+.2f}%" if len(x)>0 else "0.00%")
    )

    return {
        "numero_predicciones": tot,
        "numero_aciertos": aciertos,
        "numero_fallos": fallos,
        "tasa_acierto": round(tasa_acierto, 1),
        "rentabilidad_media": round(ret_media, 2),
        "mejor_resultado": round(mejor_res, 2),
        "peor_resultado": round(peor_res, 2),
        "retorno_total": round(ret_acum, 2),
        "retorno_buy_hold": round(ret_buy_hold, 2),
        "precision_alcistas": round(prec_alcista, 1),
        "precision_bajistas": round(prec_bajista, 1),
        "precision_neutrales": round(prec_neutral, 1),
        "tabla_confianza": tabla_confianza,
        "tabla_score": tabla_score,
        "tabla_horizonte": tabla_horizonte
    }

def ejecutar_backtest_engine(ticker, periodo_meses, horizonte_dias, es_metal=False):
    """
    Ejecuta el backtest garantizando compatibilidad doble:
    1. Si se llama directamente, devuelve el DataFrame y el retorno Buy & Hold (manteniendo compatibilidad previa).
    2. Incorpora la estructura extendida con todas las métricas en caso de ser solicitada.
    """
    df, err_msg = obtener_historico_cache(ticker, periodo="5y")
    if df is None:
        return None, err_msg
    
    df = df.sort_index()
    total_barras = len(df)
    horizonte_dias = int(horizonte_dias)
    
    fin_idx = total_barras - horizonte_dias
    if fin_idx <= 10:
        return None, f"El horizonte de {horizonte_dias} días es demasiado amplio para los datos históricos disponibles de {ticker}."
        
    barras_solicitadas = int((periodo_meses / 12) * 252)
    inicio_idx = max(10, fin_idx - barras_solicitadas)
    
    if inicio_idx >= fin_idx:
        inicio_idx = max(5, fin_idx - 20)
    
    registros = []
    paso = max(1, horizonte_dias // 2) # Avance de muestreo dinámico
    
    for i in range(inicio_idx, fin_idx, paso):
        df_slice = df.iloc[:i+1] # Estricta ventana temporal histórica sin datos futuros
        
        fecha_senal = df_slice.index[-1].strftime("%Y-%m-%d")
        precio_inicial = float(df_slice['Close'].iloc[-1])
        precio_final = float(df['Close'].iloc[i + horizonte_dias])
        
        rentabilidad = ((precio_final - precio_inicial) / precio_inicial) * 100.0
        
        tec_data = calcular_indicadores_historicos(df_slice)
        score, direccion, confianza = simular_market_ai_score_historico(tec_data, es_metal)
        
        # Evaluación flexible del resultado
        if direccion == "🟢 ALCISTA":
            resultado = "✅ Acierto" if rentabilidad > 0.0 else "❌ Fallo"
        elif direccion == "🔴 BAJISTA":
            resultado = "✅ Acierto" if rentabilidad < 0.0 else "❌ Fallo"
        else:
            # Neutral: Acertado si se mantiene en una banda de oscilación de +/- 2.5%
            resultado = "✅ Acierto" if abs(rentabilidad) <= 2.5 else "⚪ Neutral"
            
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
        return None, "No se pudieron calcular predicciones en la ventana especificada."
        
    df_res = pd.DataFrame(registros)
    
    p_inicio_periodo = float(df['Close'].iloc[inicio_idx])
    p_fin_periodo = float(df['Close'].iloc[fin_idx])
    ret_buy_hold = ((p_fin_periodo - p_inicio_periodo) / p_inicio_periodo) * 100.0
    
    # Calcular estructura de métricas completas
    metricas = calcular_metricas_backtest(df_res, ret_buy_hold)
    
    resultado_completo = {
        "resultados": df_res,
        "metricas": metricas,
        "buy_hold": round(ret_buy_hold, 2)
    }
    
    # Retorno tuple para total compatibilidad con la interfaz existente
    return (df_res, round(ret_buy_hold, 2)), resultado_completo
