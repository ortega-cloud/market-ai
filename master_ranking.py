import pandas as pd
import numpy as np
import datetime
import streamlit as st

def calcular_oportunidad_global(item: dict) -> float:
    """
    Calcula una puntuación de Oportunidad Global entre 0 y 100 sin favorecer
    retornos masivos pero con baja confianza.
    """
    score = float(item.get("score", 50))
    potencial = float(item.get("potencial", 0))  # En porcentaje (-100 a +Inf)
    confianza = float(item.get("confianza", 50))  # 0 a 100
    
    # Normalización del riesgo a un factor de penalización (0.0 a 1.0)
    riesgo_map = {"Bajo": 0.1, "Medio": 0.3, "Alto": 0.6, "Muy alto": 0.9}
    riesgo_str = item.get("riesgo", "Medio")
    riesgo_penalization = riesgo_map.get(riesgo_str, 0.3)
    
    # 1. Ponderación por Confianza (Penalización no lineal si la confianza es baja)
    conf_factor = (confianza / 100.0) ** 1.5
    
    # 2. Score Ajusado por Confianza
    score_ajustado = score * conf_factor
    
    # 3. Potencial Ponderado (Log de atenuación para rendimientos extremos)
    direccion = item.get("direccion", "🟢 Alcista")
    if "Bajista" in direccion:
        # En señales bajistas, un potencial negativo alto es deseable para el trade bajista
        potencial_magnitud = max(0.0, -potencial)
    else:
        potencial_magnitud = max(0.0, potencial)
        
    # Cap de impacto de potencial para evitar que +500% distorsione el algoritmo
    potencial_score = min(30.0, np.log1p(potencial_magnitud) * 8.0) * conf_factor
    
    # 4. Ajuste por Riesgo
    factor_riesgo = 1.0 - (riesgo_penalization * 0.35)
    
    # Cálculo Final (Máximo 100)
    oportunidad_global = (score_ajustado * 0.6 + potencial_score * 0.4) * factor_riesgo
    return float(np.clip(oportunidad_global, 0.0, 100.0))

def validar_calidad_datos(item: dict) -> bool:
    """Filtra oportunidades con datos esenciales incompletos."""
    campos_requeridos = ["activo", "precio", "score", "confianza", "potencial"]
    for campo in campos_requeridos:
        val = item.get(campo)
        if val is None or pd.isna(val):
            return False
    if item.get("confianza", 0) < 30.0:  # Mínimo 30% de confianza para entrar en rankings
        return False
    return True

def generar_explicacion(item: dict) -> str:
    """Genera explicaciones basadas exclusivamente en variables presentes."""
    razones = []
    if item.get("score", 0) >= 80:
        razones.append("alto MARKET AI Score general")
    if abs(item.get("potencial", 0)) > 15:
        razones.append(f"potencial estimado del {item.get('potencial'):+.1f}%")
    if item.get("confianza", 0) >= 75:
        razones.append("alta convergencia de señales y fuentes sólidas")
    
    riesgo = item.get("riesgo", "Medio")
    if riesgo in ["Alto", "Muy alto"]:
        riesgo_txt = f" El riesgo es {riesgo.lower()}, exigiendo un control estricto de stop loss."
    else:
        riesgo_txt = f" Presenta un nivel de riesgo {riesgo.lower()}."

    if razones:
        base = f"MARKET AI detecta una oportunidad destacada impulsada por {', '.join(razones)}."
    else:
        base = "MARKET AI identifica una alineación técnica y fundamental neutra-positiva."
        
    return base + riesgo_txt

@st.cache_data(ttl=900, show_spinner=False)
def ejecutar_escaneo_master_ranking(_escanner_sp500_fn, _escanner_metales_fn):
    """
    Reutiliza las funciones de escaneo existentes para evitar peticiones duplicadas.
    Cacheado por 15 minutos (900 segundos).
    """
    # Ejecutar escáneres existentes
    res_acciones = _escanner_sp500_fn()
    res_metales = _escanner_metales_fn()
    
    universo = []
    
    for a in res_acciones:
        a["tipo"] = "📈 Acción"
        if validar_calidad_datos(a):
            a["oportunidad_global"] = calcular_oportunidad_global(a)
            universo.append(a)
            
    for m in res_metales:
        m["tipo"] = "🥇 Metal/Futuro"
        if validar_calidad_datos(m):
            m["oportunidad_global"] = calcular_oportunidad_global(m)
            universo.append(m)
            
    # Ordenar por Oportunidad Global descendente
    universo.sort(key=lambda x: x["oportunidad_global"], reverse=True)
    return universo

    # Llamada al prediction engine dentro del escáner
    pred_res = ejecutar_prediction_engine(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas, res_noticias, es_metal)

    # Añadir a las columnas de la tabla final del ranking:
    registro_item = {
    'activo': ticker,
    'score': score_global,
    'Predicción': pred_res['direccion'],
    'Confianza': f"{pred_res['confianza']}%",
    'Horizonte': pred_res['horizonte_principal'],
    # ... conserva exactamente el resto de tus campos originales ...
}
