import numpy as np
import pandas as pd
import streamlit as st
from backtesting_engine import obtener_historico_cache, calcular_indicadores_historicos

@st.cache_data(ttl=86400, show_spinner=False)
def optimizar_pesos_historicos(ticker, periodo_meses=24, horizonte_dias=20, es_metal=False):
    """
    Ejecuta el analisis de senales individual y la optimizacion de pesos usando
    división temporal estricta TRAIN (70%) / VALIDATION (30%) sin Look-Ahead Bias.
    """
    df, err = obtener_historico_cache(ticker, periodo="5y")
    if df is None or len(df) < 50:
        return None, err or "Datos insuficientes para optimización."
        
    df = df.sort_index()
    total_barras = len(df)
    barras_eval = min(int((periodo_meses / 12) * 252), total_barras - horizonte_dias - 20)
    
    if barras_eval <= 20:
        return None, "Insuficientes datos en la ventana seleccionada para entrenamiento y validación."
        
    fin_idx = total_barras - horizonte_dias
    inicio_idx = max(20, fin_idx - barras_eval)
    
    paso = max(1, horizonte_dias // 2)
    indices = list(range(inicio_idx, fin_idx, paso))
    
    if len(indices) < 10:
        return None, "Se requieren al menos 10 observaciones para ejecutar la optimización."
        
    # División Temporal TRAIN (70%) / VALIDATION (30%)
    corte_train = int(len(indices) * 0.70)
    indices_train = indices[:corte_train]
    indices_val = indices[corte_train:]
    
    def evaluar_muestra(idx_list):
        muestras = []
        for i in idx_list:
            df_slice = df.iloc[:i+1]
            precio_ini = float(df_slice['Close'].iloc[-1])
            precio_fin = float(df['Close'].iloc[i + horizonte_dias])
            rent = ((precio_fin - precio_ini) / precio_ini) * 100.0
            
            tec = calcular_indicadores_historicos(df_slice)
            
            # Simulación simple de otros bloques (Fundamentales, Valoración, Riesgo)
            # manteniendo la arquitectura de señales sin Look-Ahead Bias
            val_score = 60.0 if tec.get("rsi", 50) < 45 else 40.0
            fund_score = 65.0 if tec.get("ma20", 0) > tec.get("ma50", 0) else 45.0
            crec_score = 60.0 if tec.get("momentum", 0) > 0 else 40.0
            sent_score = 55.0 if tec.get("precio", 0) > tec.get("ma200", 0) else 45.0
            risk_score = 40.0 if tec.get("rsi", 50) > 65 else 70.0 # Alto score = menor riesgo
            
            muestras.append({
                "fecha": df_slice.index[-1].strftime("%Y-%m-%d"),
                "rentabilidad": rent,
                "tec_score": (100 - tec["rsi"]) if tec else 50.0,
                "val_score": val_score,
                "fund_score": fund_score,
                "crec_score": crec_score,
                "sent_score": sent_score,
                "risk_score": risk_score,
                "tec_data": tec
            })
        return muestras

    data_train = evaluar_muestra(indices_train)
    data_val = evaluar_muestra(indices_val)
    
    # ----------------------------------------------------
    # 1. EVALUACIÓN DE SEÑALES INDIVIDUALES (TRAIN + VAL)
    # ----------------------------------------------------
    todas_muestras = data_train + data_val
    senales_eval = {
        "Precio > MA20": [m["tec_data"].get("precio", 0) > m["tec_data"].get("ma20", 0) for m in todas_muestras],
        "MA20 > MA50": [m["tec_data"].get("ma20", 0) > m["tec_data"].get("ma50", 0) for m in todas_muestras],
        "Precio > MA200": [m["tec_data"].get("precio", 0) > m["tec_data"].get("ma200", 0) for m in todas_muestras],
        "RSI Sobrevendido (<30)": [m["tec_data"].get("rsi", 50) < 30 for m in todas_muestras],
        "RSI Sobrecomprado (>70)": [m["tec_data"].get("rsi", 50) > 70 for m in todas_muestras],
        "Momentum Positivo (>0%)": [m["tec_data"].get("momentum", 0) > 0 for m in todas_muestras],
        "PER / Valoración Atractiva": [m["val_score"] > 50 for m in todas_muestras],
        "Fundamentales Sólidos": [m["fund_score"] > 50 for m in todas_muestras],
        "Crecimiento Sostenido": [m["crec_score"] > 50 for m in todas_muestras],
        "Sentimiento / Noticias Favorables": [m["sent_score"] > 50 for m in todas_muestras],
        "Riesgo Bajo / Controlado": [m["risk_score"] > 50 for m in todas_muestras],
    }
    
    res_senales = []
    for nombre_s, condiciones in senales_eval.items():
        casos_filtro = [todas_muestras[k]["rentabilidad"] for k, cond in enumerate(condiciones) if cond]
        num_casos = len(casos_filtro)
        
        if num_casos < 3:
            res_senales.append({
                "Señal": nombre_s,
                "Casos": num_casos,
                "Acierto": "N/D - Datos insuficientes",
                "Rentabilidad Media": "N/D",
                "Evidencia": "🔴 Ninguna"
            })
        else:
            aciertos = sum(1 for r in casos_filtro if r > 0)
            pct_acierto = (aciertos / num_casos) * 100.0
            rent_med = float(np.mean(casos_filtro))
            
            evidencia = "🟢 Alta" if num_casos >= 20 else ("🟡 Moderada" if num_casos >= 8 else "🔴 Baja")
            
            res_senales.append({
                "Señal": nombre_s,
                "Casos": num_casos,
                "Acierto": f"{pct_acierto:.1f}%",
                "Rentabilidad Media": f"{rent_med:+.2f}%",
                "Evidencia": evidencia,
                "_pct_val": pct_acierto
            })
            
    df_senales = pd.DataFrame(res_senales)

    # ----------------------------------------------------
    # 2. COMPARACIÓN DE PERFILES DE PESOS (SUMA = 100%)
    # ----------------------------------------------------
    perfiles = [
        {"nombre": "Modelo Actual (Equilibrado)", "pesos": [30, 20, 15, 10, 15, 10]},
        {"nombre": "Enfoque Técnico Dominante", "pesos": [45, 15, 10, 10, 10, 10]},
        {"nombre": "Enfoque Valoración & Fundamentos", "pesos": [20, 30, 25, 10, 10, 5]},
        {"nombre": "Enfoque Crecimiento & Momentum", "pesos": [25, 15, 15, 30, 10, 5]},
        {"nombre": "Enfoque Conservador (Bajo Riesgo)", "pesos": [20, 20, 20, 10, 10, 20]}
    ]
    
    def evaluar_perfil(dataset, pesos):
        w_tec, w_val, w_fund, w_crec, w_sent, w_risk = [p / 100.0 for p in pesos]
        aciertos = 0
        rentabilidades = []
        
        for m in dataset:
            score_glob = (
                m["tec_score"] * w_tec +
                m["val_score"] * w_val +
                m["fund_score"] * w_fund +
                m["crec_score"] * w_crec +
                m["sent_score"] * w_sent +
                m["risk_score"] * w_risk
            )
            
            es_alcista = score_glob >= 50.0
            rent = m["rentabilidad"]
            rentabilidades.append(rent)
            
            if (es_alcista and rent > 0) or (not es_alcista and rent <= 0):
                aciertos += 1
                
        tot = len(dataset)
        tasa_acierto = (aciertos / tot * 100.0) if tot > 0 else 0.0
        rent_media = float(np.mean(rentabilidades)) if tot > 0 else 0.0
        
        return tasa_acierto, rent_media

    res_perfiles = []
    mejor_perfil = None
    max_score_eval = -999.0
    
    for perf in perfiles:
        acierto_tr, rent_tr = evaluar_perfil(data_train, perf["pesos"])
        acierto_val, rent_val = evaluar_perfil(data_val, perf["pesos"])
        
        # Métrica combinada: Prioriza acierto y estabilidad entre TRAIN y VALIDATION
        estabilidad = 100.0 - abs(acierto_tr - acierto_val)
        score_eval = (acierto_val * 0.6) + (estabilidad * 0.4)
        
        item = {
            "nombre": perf["nombre"],
            "pesos": perf["pesos"],
            "acierto_train": round(acierto_tr, 1),
            "rent_train": round(rent_tr, 2),
            "acierto_val": round(acierto_val, 1),
            "rent_val": round(rent_val, 2),
            "estabilidad": round(estabilidad, 1)
        }
        res_perfiles.append(item)
        
        if score_eval > max_score_eval:
            max_score_eval = score_eval
            mejor_perfil = item

    return {
        "tabla_senales": df_senales,
        "perfiles_evaluados": res_perfiles,
        "modelo_actual": res_perfiles[0],
        "modelo_optimizado": mejor_perfil,
        "num_train": len(data_train),
        "num_val": len(data_val)
    }, None
