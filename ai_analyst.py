import pandas as pd
import numpy as np

def determinar_calidad_datos(datos):
    """
    Evalúa la disponibilidad y completitud de los datos para asignar un nivel de calidad:
    Alta, Media o Baja.
    """
    puntos = 0
    total_posible = 6
    detalles_faltantes = []

    # 1. Precio e Histórico
    if datos.get("precio_actual") and float(datos.get("precio_actual", 0)) > 0:
        puntos += 1
    else:
        detalles_faltantes.append("Precio actual / histórico")

    # 2. DCF / Fair Value
    if datos.get("fair_value_dcf") and float(datos.get("fair_value_dcf", 0)) > 0:
        puntos += 1
    else:
        detalles_faltantes.append("Modelo Valuation DCF")

    # 3. Fundamentales
    fund = datos.get("fundamentales", {})
    if isinstance(fund, dict) and len(fund) > 0 and (fund.get("roe") is not None or fund.get("per") is not None):
        puntos += 1
    else:
        detalles_faltantes.append("Métricas Fundamentales (PER, ROE, Deuda)")

    # 4. Analistas
    analistas = datos.get("analistas", {})
    if isinstance(analistas, dict) and analistas.get("precio_objetivo"):
        puntos += 1
    else:
        detalles_faltantes.append("Consenso de Analistas")

    # 5. Machine Learning
    ml = datos.get("ml_predicciones", {})
    if isinstance(ml, dict) and (ml.get("5D") or ml.get("20D") or ml.get("60D")):
        puntos += 1
    else:
        detalles_faltantes.append("Predicciones Machine Learning")

    # 6. Score MARKET AI
    if datos.get("score_mai") is not None or datos.get("score_historico") is not None:
        puntos += 1
    else:
        detalles_faltantes.append("MARKET AI Score")

    ratio = puntos / total_posible
    if ratio >= 0.8:
        nivel = "Alta"
    elif ratio >= 0.5:
        nivel = "Media"
    else:
        nivel = "Baja"

    mensaje_advertencia = None
    if detalles_faltantes:
        mensaje_advertencia = f"Análisis limitado por falta de datos en: {', '.join(detalles_faltantes)}."

    return {
        "nivel": nivel,
        "puntos": puntos,
        "total": total_posible,
        "detalles_faltantes": detalles_faltantes,
        "mensaje_advertencia": mensaje_advertencia
    }


def evaluar_horizonte_ml(ml):
    """
    Evalúa la coherencia y divergencia entre horizontes (5D, 20D, 60D, 120D).
    """
    if not isinstance(ml, dict) or not ml:
        return {"coherencia": "DESCONOCIDO", "resumen": "Sin datos de predicción ML para evaluar horizontes."}

    p_5d = ml.get("5D", "N/D")
    p_20d = ml.get("20D", "N/D")
    p_60d = ml.get("60D", "N/D")
    p_120d = ml.get("120D", "N/D")

    bulls = sum(1 for p in [p_5d, p_20d, p_60d, p_120d] if p in ["BULLISH", "COMPRA", "COMPRA FUERTE"])
    bears = sum(1 for p in [p_5d, p_20d, p_60d, p_120d] if p in ["BEARISH", "VENTA", "VENTA FUERTE"])

    if bulls >= 3:
        coherencia = "ALCISTA_CONSOLIDADO"
        resumen = "Alineación alcista clara en la mayoría de horizontes temporales (Corto, Medio y Largo Plazo)."
    elif bears >= 3:
        coherencia = "BAJISTA_CONSOLIDADO"
        resumen = "Alineación bajista clara en la mayoría de horizontes temporales."
    elif p_5d in ["BEARISH", "VENTA"] and p_60d in ["BULLISH", "COMPRA"]:
        coherencia = "DIVERGENCIA_CORTO_MEDIO"
        resumen = "Divergencia: Presión o consolidación a corto plazo (5D), pero tendencia favorable a medio/largo plazo (60D/120D)."
    elif p_5d in ["BULLISH", "COMPRA"] and p_60d in ["BEARISH", "VENTA"]:
        coherencia = "DIVERGENCIA_IMPULSO_CORTO"
        resumen = "Divergencia: Impulso alcista de corto plazo (5D), pero estructura de medio/largo plazo debilitada."
    else:
        coherencia = "MIXTO"
        resumen = "Señales mixtas entre horizontes temporales, sugiriendo fase de transición o rango lateral."

    return {
        "coherencia": coherencia,
        "resumen": resumen,
        "horizontes": {
            "Corto Plazo (5D)": p_5d,
            "Medio Plazo (20D/60D)": f"20D: {p_20d} | 60D: {p_60d}",
            "Largo Plazo (120D)": p_120d
        }
    }


def detectar_conflictos(datos, score, senal, ml):
    """
    Detecta contradicciones explícitas entre los diferentes subsistemas de MARKET AI.
    """
    conflictos = []
    
    # ML vs Algoritmo Principal / Señal
    ml_pred = ml.get("20D", ml.get("60D", ml.get("prediccion_principal", "N/D"))) if isinstance(ml, dict) else "N/D"
    if ml_pred in ["BEARISH", "VENTA"] and "COMPRA" in senal:
        conflictos.append({
            "titulo": "⚠️ CONFLICTO ENTRE ML Y ALGORITMO PRINCIPAL",
            "explicacion": f"El algoritmo asigna una señal favorable ({senal}), pero el modelo ML anticipa dinámica bajista ({ml_pred}).",
            "implicacion": "El activo muestra buen valor fundamental pero podría sufrir presión vendedora técnica a corto plazo."
        })
    elif ml_pred in ["BULLISH", "COMPRA"] and "VENTA" in senal:
        conflictos.append({
            "titulo": "⚠️ CONFLICTO ENTRE ML Y ALGORITMO PRINCIPAL",
            "explicacion": f"El modelo ML detecta momentum alcista ({ml_pred}), mientras que el algoritmo indica cautela/venta ({senal}).",
            "implicacion": "Posible rally especulativo de corto plazo sobre un activo con debilidades de fondo."
        })

    # DCF vs Mercado / Precio Actual
    potencial_dcf = float(datos.get("potencial_dcf", 0) or 0)
    if potencial_dcf < -25.0 and "COMPRA" in senal:
        conflictos.append({
            "titulo": "⚠️ VALORACIÓN DCF EN CONFLICTO CON LA SEÑAL",
            "explicacion": f"El modelo DCF indica sobrevaloración severa ({potencial_dcf:.1f}%), pero la señal técnica/momentum es de {senal}.",
            "implicacion": "Riesgo de corrección fundamental por múltiplos exigentes."
        })
    elif potencial_dcf > 40.0 and "VENTA" in senal:
        conflictos.append({
            "titulo": "⚠️ POTENCIAL DCF ALTO CON SEÑAL DE VENTA",
            "explicacion": f"Existe un elevado margen de seguridad según DCF (+{potencial_dcf:.1f}%), pero la tendencia/señal es de {senal}.",
            "implicacion": "Posible 'trampa de valor' (Value Trap) donde el mercado descuenta deterioro operativo continuo."
        })

    # Consenso de Analistas vs MARKET AI
    analistas = datos.get("analistas", {})
    if isinstance(analistas, dict):
        pot_analistas = float(analistas.get("potencial", 0) or 0)
        if pot_analistas < -10 and score >= 65:
            conflictos.append({
                "titulo": "⚠️ CAUTELA EN CONSENSO DE ANALISTAS",
                "explicacion": f"MARKET AI otorga un Score elevado ({score}), mientras que el objetivo de analistas implica caída ({pot_analistas:.1f}%).",
                "implicacion": "Divergencia entre los indicadores cuantitativos internos y las revisiones de Wall Street."
            })

    return conflictos


def generar_analisis_ai(datos):
    """
    Función principal del módulo AI Analyst.
    Recibe un diccionario con los datos agregados existentes del activo y genera
    una interpretación estructurada basada en reglas multivariable.
    """
    if not isinstance(datos, dict):
        datos = {}

    # Extraer variables con valores por defecto seguros
    score = float(datos.get("score_mai", datos.get("score_historico", 50.0)) or 50.0)
    precio = float(datos.get("precio_actual", datos.get("precio", 0.0)) or 0.0)
    fair_value = float(datos.get("fair_value_dcf", datos.get("dcf", 0.0)) or 0.0)
    potencial_dcf = float(datos.get("potencial_dcf", 0.0) or 0.0)
    senal = str(datos.get("senal", datos.get("signal", "MANTENER")))
    confianza = float(datos.get("confianza", 50.0) or 50.0)
    riesgo = str(datos.get("riesgo", "MEDIO")).upper()

    fund = datos.get("fundamentales", {}) if isinstance(datos.get("fundamentales"), dict) else {}
    crec = datos.get("crecimiento", {}) if isinstance(datos.get("crecimiento"), dict) else {}
    ml = datos.get("ml_predicciones", {}) if isinstance(datos.get("ml_predicciones"), dict) else {}
    analistas = datos.get("analistas", {}) if isinstance(datos.get("analistas"), dict) else {}

    # 1. EVALUAR CALIDAD DE DATOS
    calidad = determinar_calidad_datos(datos)

    # 2. EVALUAR HORIZONTES DE MACHINE LEARNING
    horizonte_info = evaluar_horizonte_ml(ml)

    # 3. DETECTAR CONFLICTOS
    conflictos = detectar_conflictos(datos, score, senal, ml)

    # 4. DETERMINAR CONCLUSIÓN GLOBAL
    # Matriz multivariable: Score + DCF + ML + Conflictos
    if score >= 72 and potencial_dcf >= 5.0 and len(conflictos) == 0:
        conclusion = "🟢 Muy favorable"
        resumen_conclusion = "El activo presenta una sólida combinación de valoración atractiva, indicadores técnicos favorables y respaldo de modelos ML sin contradicciones relevantes."
    elif score >= 58 and potencial_dcf >= -10.0:
        conclusion = "🟢 Favorable"
        resumen_conclusion = "Perfil positivo general. El activo muestra fortalezas en su Score y prospectiva, aunque con matices moderados en valoración o momentum."
    elif score <= 38 or potencial_dcf <= -25.0 or (score < 48 and riesgo == "ALTO"):
        conclusion = "🔴 Muy desfavorable"
        resumen_conclusion = "Deterioro significativo en múltiples métricas. Alta penalización por valoración excesiva, debilidad técnica o riesgo elevado."
    elif score <= 48 or "VENTA" in senal:
        conclusion = "🟠 Desfavorable"
        resumen_conclusion = "Predominio de factores de cautela. La estructura actual sugiere mantener distancia o considerar toma de beneficios/reducción."
    else:
        conclusion = "🟡 Neutral"
        resumen_conclusion = "Equilibrio entre factores positivos y negativos. El activo cotiza cerca de su valor razonable o muestra señales mixtas entre horizontes."

    # 5. PUNTOS POSITIVOS (3-5 factores reales)
    puntos_positivos = []

    if potencial_dcf >= 15.0:
        puntos_positivos.append(f"Potencial DCF atractivo (+{potencial_dcf:.1f}% al Fair Value de ${fair_value:.2f}).")
    elif potencial_dcf > 0:
        puntos_positivos.append(f"Cotiza por debajo de su Fair Value DCF razonable (${fair_value:.2f}).")

    if score >= 65:
        puntos_positivos.append(f"MARKET AI Score elevado ({score:.1f}/100), reflejando salud técnica y cuantitativa.")

    if horizonte_info["coherencia"] == "ALCISTA_CONSOLIDADO":
        puntos_positivos.append("Modelos Machine Learning alineados al alza en horizontes de corto, medio y largo plazo.")
    elif ml.get("60D") in ["BULLISH", "COMPRA"]:
        puntos_positivos.append("Predicción ML favorable a medio plazo (60D).")

    roe = float(fund.get("roe", 0) or 0)
    if roe >= 15.0:
        puntos_positivos.append(f"Alta rentabilidad sobre el capital (ROE de {roe:.1f}%).")

    fcf = float(fund.get("free_cash_flow", 0) or 0)
    if fcf > 0:
        puntos_positivos.append("Generación de Free Cash Flow positiva y saludable.")

    rev_growth = float(crec.get("ingresos", crec.get("revenue_growth", 0)) or 0)
    if rev_growth >= 10.0:
        puntos_positivos.append(f"Crecimiento dinámico de ingresos (+{rev_growth:.1f}% interanual).")

    pot_analistas = float(analistas.get("potencial", 0) or 0)
    if pot_analistas >= 12.0:
        puntos_positivos.append(f"Consenso de analistas con margen alcista positivo (+{pot_analistas:.1f}%).")

    if not puntos_positivos:
        puntos_positivos.append("El activo mantiene estabilidad básica en sus métricas principales sin destacar negativamente.")

    # 6. PUNTOS NEGATIVOS (3-5 problemas reales)
    puntos_negativos = []

    if potencial_dcf <= -15.0:
        puntos_negativos.append(f"Sobrevaloración por DCF ({potencial_dcf:.1f}% sobre su Fair Value estimado de ${fair_value:.2f}).")

    if score <= 45:
        puntos_negativos.append(f"MARKET AI Score débil ({score:.1f}/100), señalando fragilidad cuantitativa.")

    if horizonte_info["coherencia"] == "BAJISTA_CONSOLIDADO":
        puntos_negativos.append("Predicciones Machine Learning bajistas consolidadas en múltiples horizontes.")
    elif ml.get("5D") in ["BEARISH", "VENTA"]:
        puntos_negativos.append("Perspectiva de corto plazo (5D) bajo presión vendedora según el modelo ML.")

    per = float(fund.get("per", 0) or 0)
    if per > 35.0:
        puntos_negativos.append(f"Múltiplo PER elevado ({per:.1f}x), requiriendo un crecimiento sostenido exigente.")

    deuda_ebitda = float(fund.get("deuda_ebitda", fund.get("debt_to_equity", 0)) or 0)
    if deuda_ebitda > 3.0 or fund.get("nivel_deuda") == "ALTO":
        puntos_negativos.append("Nivel de endeudamiento financiero elevado o superior a la media.")

    if not puntos_negativos:
        puntos_negativos.append("No se detectan debilidades severas en los datos financieros o técnicos evaluados.")

    # 7. PRINCIPALES RIESGOS (Segmentados y específicos)
    riesgos = []

    if potencial_dcf < -10.0 or per > 30.0:
        riesgos.append({
            "tipo": "Riesgo de Valoración",
            "descripcion": "El precio cotiza con prima sobre métricas intrínsecas; vulnerable a compresiones de múltiplo."
        })

    if riesgo == "ALTO" or float(datos.get("volatilidad", 0) or 0) > 30.0:
        riesgos.append({
            "tipo": "Riesgo Técnico / Volatilidad",
            "descripcion": "Alta volatilidad histórica de precios con oscilaciones pronunciadas."
        })

    if fund.get("nivel_deuda") == "ALTO" or deuda_ebitda > 2.5:
        riesgos.append({
            "tipo": "Riesgo Financiero",
            "descripcion": "Estructura de capital apalancada expuesta a variaciones en tipos de interés."
        })

    if crec.get("ingresos") is not None and float(crec.get("ingresos", 0) or 0) < 0:
        riesgos.append({
            "tipo": "Riesgo de Crecimiento",
            "descripcion": "Contracción en la tasa de crecimiento de ventas/ingresos."
        })

    if not riesgos:
        riesgos.append({
            "tipo": "Riesgo General de Mercado",
            "descripcion": "Exposición estándar a condiciones macroeconómicas y fluctuaciones generales del sector."
        })

    # 8. CATALIZADORES
    catalizadores = []

    if potencial_dcf > 0:
        catalizadores.append("Convergencia progresiva del precio de mercado hacia su Fair Value DCF.")

    if ml.get("5D") in ["BEARISH", "NEUTRAL"] and ml.get("60D") in ["BULLISH"]:
        catalizadores.append("Superación de la consolidación de corto plazo acelerando el impulso a 60 días.")

    if analistas.get("precio_objetivo") and float(analistas.get("precio_objetivo", 0)) > precio:
        catalizadores.append("Revisiones al alza en las estimaciones y precios objetivo de analistas de Wall Street.")

    catalizadores.append("Presentación de resultados trimestrales que superen las expectativas en margen y beneficio.")
    catalizadores.append("Mejora en las métricas de eficiencia operativa y expansión del Free Cash Flow.")

    # 9. CONDICIONES DE CAMBIO DE OPINIÓN
    condiciones_cambio = []

    if "COMPRA" in senal or score >= 55:
        target_favorable_max = fair_value * 1.15 if fair_value > 0 else precio * 1.25
        condiciones_cambio.append(
            f"Una subida del precio por encima de aprox. ${target_favorable_max:.2f} agotaría el margen de seguridad DCF, reduciendo la calificación."
        )
        condiciones_cambio.append(
            "Un deterioro en las predicciones ML a 20D/60D pasando a BEARISH obligaría a reducir la recomendación a MANTENER o VENTA."
        )
    else:
        target_descuento = fair_value * 0.85 if fair_value > 0 else precio * 0.80
        condiciones_cambio.append(
            f"Una corrección del precio hacia el nivel de ${target_descuento:.2f} reconstruiría un margen de seguridad suficiente para elevar la recomendación."
        )
        condiciones_cambio.append(
            "Una mejora conjunta del Score superando los 60 puntos y un giro alcista en ML cambiarían la visión a FAVORABLE."
        )

    return {
        "conclusion": conclusion,
        "resumen_conclusion": resumen_conclusion,
        "puntos_positivos": puntos_positivos[:5],
        "puntos_negativos": puntos_negativos[:5],
        "riesgos": riesgos[:4],
        "catalizadores": catalizadores[:4],
        "conflictos": conflictos,
        "condiciones_cambio": condiciones_cambio,
        "horizonte_info": horizonte_info,
        "calidad_datos": calidad
    }
