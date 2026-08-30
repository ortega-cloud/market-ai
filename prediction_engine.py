import numpy as np
import pandas as pd
from datetime import datetime

def ejecutar_prediction_engine(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas, res_noticias, es_metal=False):
    """
    MOTOR DE PREDICCIÓN INDEPENDIENTE (MARKET AI PREDICTION ENGINE).
    Evalúa señales por categoría, calcula puntuación predictiva ponderada (-100 a +100),
    genera probabilidades que suman 100%, horizontes temporales y objetivos por escenario.
    Sin Look-Ahead Bias ni generación de datos inventados.
    """
    senales_pos = []
    senales_neu = []
    senales_neg = []

    # 1. EVALUACIÓN DE SEÑALES POR CATEGORÍA
    cat_scores = {}
    cat_weights_base = {
        "tecnico": 0.30 if not es_metal else 0.50,
        "valoracion": 0.20 if not es_metal else 0.00,
        "fundamentales": 0.15 if not es_metal else 0.00,
        "crecimiento": 0.10 if not es_metal else 0.00,
        "noticias": 0.15 if not es_metal else 0.30,
        "riesgo": 0.10 if not es_metal else 0.20
    }

    # --- TÉCNICO ---
    precio = datos_tec.get("precio")
    ma20 = datos_tec.get("ma20")
    ma50 = datos_tec.get("ma50")
    ma200 = datos_tec.get("ma200")
    rsi = datos_tec.get("rsi")
    volatilidad = datos_tec.get("volatilidad")

    score_tec = 0.0
    cnt_tec = 0
    if precio and ma20 and ma50:
        cnt_tec += 1
        if precio > ma20 > ma50:
            score_tec += 100
            senales_pos.append({"cat": "Técnico", "desc": "Precio > MA20 > MA50 (Tendance Alcista)", "origen": "Técnico"})
        elif precio < ma20 < ma50:
            score_tec -= 100
            senales_neg.append({"cat": "Técnico", "desc": "Precio < MA20 < MA50 (Tendencia Bajista)", "origen": "Técnico"})
        else:
            senales_neu.append({"cat": "Técnico", "desc": "Medias móviles cruzadas/laterales", "origen": "Técnico"})

    if precio and ma200:
        cnt_tec += 1
        if precio > ma200:
            score_tec += 100
            senales_pos.append({"cat": "Técnico", "desc": "Precio por encima de MA200 (Estructura Long-Term)", "origen": "Técnico"})
        else:
            score_tec -= 100
            senales_neg.append({"cat": "Técnico", "desc": "Precio bajo la MA200 (Tendencia principal débil)", "origen": "Técnico"})

    if rsi is not None and not np.isnan(rsi):
        cnt_tec += 1
        if rsi < 30:
            score_tec += 80
            senales_pos.append({"cat": "Técnico", "desc": f"RSI en sobreventa ({rsi:.1f}) propicio para rebote", "origen": "Técnico"})
        elif rsi > 70:
            score_tec -= 80
            senales_neg.append({"cat": "Técnico", "desc": f"RSI en sobrecompra ({rsi:.1f}), riesgo de corrección", "origen": "Técnico"})
        else:
            senales_neu.append({"cat": "Técnico", "desc": f"RSI neutral ({rsi:.1f})", "origen": "Técnico"})

    if cnt_tec > 0:
        cat_scores["tecnico"] = score_tec / cnt_tec

    # --- VALORACIÓN (Solo Acciones) ---
    if not es_metal and datos_val:
        score_val = 0.0
        cnt_val = 0
        fair_value = datos_val.get("valor_dcf_base")
        obj_med = datos_analistas.get("obj_med") if datos_analistas else None

        if fair_value and precio and precio > 0:
            cnt_val += 1
            pot_dcf = ((fair_value - precio) / precio) * 100
            if pot_dcf >= 15:
                score_val += 100
                senales_pos.append({"cat": "Valoración", "desc": f"Descuento intrínseco DCF del +{pot_dcf:.1f}%", "origen": "DCF"})
            elif pot_dcf <= -15:
                score_val -= 100
                senales_neg.append({"cat": "Valoración", "desc": f"Sobrevalorado por DCF un {abs(pot_dcf):.1f}%", "origen": "DCF"})
            else:
                senales_neu.append({"cat": "Valoración", "desc": f"Precio alineado con Fair Value DCF ({pot_dcf:+.1f}%)", "origen": "DCF"})

        if obj_med and precio and precio > 0:
            cnt_val += 1
            pot_analistas = ((obj_med - precio) / precio) * 100
            if pot_analistas >= 10:
                score_val += 80
                senales_pos.append({"cat": "Valoración", "desc": f"Consenso analistas favorable (+{pot_analistas:.1f}%)", "origen": "Consenso"})
            elif pot_analistas <= -10:
                score_val -= 80
                senales_neg.append({"cat": "Valoración", "desc": f"Objetivo analistas inferior a la cotización ({pot_analistas:.1f}%)", "origen": "Consenso"})

        if cnt_val > 0:
            cat_scores["valoracion"] = score_val / cnt_val

    # --- FUNDAMENTALES (Solo Acciones) ---
    if not es_metal and datos_fund:
        score_fund = 0.0
        cnt_fund = 0
        roe = datos_fund.get("roe")
        fcf = datos_fund.get("flujo_caja")

        if roe is not None and not np.isnan(roe):
            cnt_fund += 1
            if roe >= 0.15:
                score_fund += 100
                senales_pos.append({"cat": "Fundamentales", "desc": f"ROE elevado ({roe*100:.1f}%)", "origen": "Balance"})
            elif roe < 0.05:
                score_fund -= 80
                senales_neg.append({"cat": "Fundamentales", "desc": f"ROE bajo/insuficiente ({roe*100:.1f}%)", "origen": "Balance"})

        if fcf is not None and not np.isnan(fcf):
            cnt_fund += 1
            if fcf > 0:
                score_fund += 80
                senales_pos.append({"cat": "Fundamentales", "desc": "Free Cash Flow positivo", "origen": "Caja"})
            else:
                score_fund -= 100
                senales_neg.append({"cat": "Fundamentales", "desc": "Free Cash Flow negativo", "origen": "Caja"})

        if cnt_fund > 0:
            cat_scores["fundamentales"] = score_fund / cnt_fund

    # --- CRECIMIENTO (Solo Acciones) ---
    if not es_metal and datos_crec:
        score_crec = 0.0
        cnt_crec = 0
        c_ing = datos_crec.get("crecimiento_ingresos")

        if c_ing is not None and not np.isnan(c_ing):
            cnt_crec += 1
            if c_ing > 0.08:
                score_crec += 100
                senales_pos.append({"cat": "Crecimiento", "desc": f"Crecimiento sólido de ingresos (+{c_ing*100:.1f}%)", "origen": "Resultados"})
            elif c_ing < 0:
                score_crec -= 100
                senales_neg.append({"cat": "Crecimiento", "desc": f"Contracción de ingresos ({c_ing*100:.1f}%)", "origen": "Resultados"})

        if cnt_crec > 0:
            cat_scores["crecimiento"] = score_crec / cnt_crec

    # --- NOTICIAS Y SENTIMIENTO ---
    if res_noticias:
        sent_puntuacion = res_noticias.get("puntuacion_sentimiento", 0)
        cat_scores["noticias"] = float(sent_puntuacion)
        if sent_puntuacion >= 25:
            senales_pos.append({"cat": "Noticias", "desc": f"Prensa/noticias favorables (+{sent_puntuacion}/100)", "origen": "Sentimiento"})
        elif sent_puntuacion <= -25:
            senales_neg.append({"cat": "Noticias", "desc": f"Titulares de noticias negativos ({sent_puntuacion}/100)", "origen": "Sentimiento"})

    # --- RIESGO ---
    score_riesgo = 0.0
    cnt_riesgo = 0
    if volatilidad is not None and not np.isnan(volatilidad):
        cnt_riesgo += 1
        if volatilidad < 25:
            score_riesgo += 80
            senales_pos.append({"cat": "Riesgo", "desc": f"Baja volatilidad anualizada ({volatilidad:.1f}%)", "origen": "Mercado"})
        elif volatilidad > 40:
            score_riesgo -= 80
            senales_neg.append({"cat": "Riesgo", "desc": f"Alta volatilidad histórica ({volatilidad:.1f}%)", "origen": "Mercado"})

    if cnt_riesgo > 0:
        cat_scores["riesgo"] = score_riesgo / cnt_riesgo

    # 2. PONDERACIÓN DINÁMICA Y CÁLCULO DEL SCORE PREDICTIVO
    pesos_activos = {k: cat_weights_base[k] for k in cat_scores if k in cat_weights_base}
    total_peso_disponible = sum(pesos_activos.values())

    if total_peso_disponible > 0:
        predictive_score = sum(cat_scores[k] * (pesos_activos[k] / total_peso_disponible) for k in cat_scores)
        datos_utilizados_pct = round((total_peso_disponible / sum(cat_weights_base.values())) * 100, 1)
    else:
        predictive_score = 0.0
        datos_utilizados_pct = 0.0

    predictive_score = float(np.clip(predictive_score, -100.0, 100.0))

    # 3. DIRECCIÓN
    if predictive_score >= 50.0:
        direccion_pred = "🟢 ALCISTA"
    elif predictive_score <= -50.0:
        direccion_pred = "🔴 BAJISTA"
    else:
        direccion_pred = "🟡 NEUTRAL"

    # 4. CONFIANZA DE LA PREDICCIÓN (0-100%)
    factor_cobertura = datos_utilizados_pct / 100.0
    total_senales = len(senales_pos) + len(senales_neg)
    coherencia = (abs(len(senales_pos) - len(senales_neg)) / total_senales) if total_senales > 0 else 0.5
    vol_pen = max(0.0, 1.0 - ((volatilidad or 25) / 100.0))

    confianza = int(np.clip((factor_cobertura * 0.4 + coherencia * 0.4 + vol_pen * 0.2) * 100, 15, 95))

    # 5. PROBABILIDADES POR ESCENARIO (SUMA EXACTA 100%)
    norm_sc = predictive_score / 100.0
    conf_f = confianza / 100.0

    prob_alc = max(5.0, 33.3 + (45.0 * norm_sc * conf_f))
    prob_baj = max(5.0, 33.3 - (45.0 * norm_sc * conf_f))
    prob_base = max(10.0, 100.0 - (prob_alc + prob_baj))

    tot_p = prob_alc + prob_baj + prob_base
    p_alc_pct = int(round((prob_alc / tot_p) * 100))
    p_base_pct = int(round((prob_base / tot_p) * 100))
    p_baj_pct = 100 - (p_alc_pct + p_base_pct)

    # 6. HORIZONTES TEMPORALES
    horizontes = {
        "⚡ 1-5 días": "🟡 NEUTRAL" if abs(predictive_score) < 30 else ("🟢 ALCISTA" if predictive_score > 0 else "🔴 BAJISTA"),
        "📅 1-4 semanas": "🟢 ALCISTA" if predictive_score >= 40 else ("🔴 BAJISTA" if predictive_score <= -40 else "🟡 NEUTRAL"),
        "📈 1-3 meses": direccion_pred,
        "🗓️ 3-6 meses": "🟢 ALCISTA" if predictive_score >= 20 else ("🔴 BAJISTA" if predictive_score <= -20 else "🟡 NEUTRAL")
    }

    # 7. OBJETIVOS DE PRECIO POR ESCENARIO
    objetivos_escenarios = {}
    if precio and precio > 0:
        vol_ref = (volatilidad / 100.0) if (volatilidad and not np.isnan(volatilidad)) else 0.20
        dcf_fv = datos_val.get("valor_dcf_base") if (datos_val and not es_metal) else None

        drift = ((dcf_fv - precio) / precio) if dcf_fv else 0.0

        p_alc_obj = precio * (1.0 + max(0.08, vol_ref * 0.7 + drift * 0.5))
        p_baj_obj = precio * (1.0 - max(0.08, vol_ref * 0.7 - drift * 0.5))
        p_base_obj = precio * (1.0 + (drift * 0.3))

        objetivos_escenarios = {
            "actual": round(precio, 2),
            "alcista": round(p_alc_obj, 2),
            "base": round(p_base_obj, 2),
            "bajista": round(p_baj_obj, 2)
        }

    # 8. EXPLICACIÓN
    if p_alc_pct >= max(p_base_pct, p_baj_pct):
        escenario_dominante = "Escenario Alcista"
        razon_dominante = f"Prevalecen las señales favorables ({len(senales_pos)} factores positivos) respaldadas por una puntuación de {predictive_score:+.1f}."
    elif p_baj_pct >= max(p_alc_pct, p_base_pct):
        escenario_dominante = "Escenario Bajista"
        razon_dominante = f"Existen factores de riesgo o sesgo bajista destacados ({len(senales_neg)} factores negativos)."
    else:
        escenario_dominante = "Escenario Base / Consolidador"
        razon_dominante = f"Equilibrio entre impulsos compradores y vendedores con {datos_utilizados_pct}% de datos disponibles."

    return {
        "score_predictivo": round(predictive_score, 1),
        "direccion": direccion_pred,
        "confianza": confianza,
        "datos_utilizados_pct": datos_utilizados_pct,
        "probabilidades": {"alcista": p_alc_pct, "base": p_base_pct, "bajista": p_baj_pct},
        "horizontes": horizontes,
        "horizonte_principal": "📈 1-3 meses",
        "objetivos": objetivos_escenarios,
        "senales_pos": senales_pos,
        "senales_neu": senales_neu,
        "senales_neg": senales_neg,
        "escenario_dominante": escenario_dominante,
        "razon_dominante": razon_dominante,
        "registro_backtest": {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "ticker": datos_tec.get("ticker", "N/D"),
            "precio": precio,
            "prediccion": direccion_pred,
            "confianza": confianza
        }
    }
