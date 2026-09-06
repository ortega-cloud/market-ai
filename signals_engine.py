import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

SIGNALS_FILE = os.path.join("models", "signals_history.json")


def _cargar_historial_senales():
    if os.path.exists(SIGNALS_FILE):
        try:
            with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _guardar_historial_senales(data):
    os.makedirs(os.path.dirname(SIGNALS_FILE), exist_ok=True)
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluar_confirmacion_ml(pred_hibrida_horizonte):
    """
    Evalúa la congruencia de ML para un horizonte específico.
    Retorna: '🟢 CONFIRMA', '🟡 NEUTRO', o '🔴 CONTRADICE'
    """
    if not pred_hibrida_horizonte or "error" in pred_hibrida_horizonte:
        return "🟡 NEUTRO", "Sin Datos ML"

    dir_score = pred_hibrida_horizonte.get("dir_score", "NEUTRAL")
    dir_ml = pred_hibrida_horizonte.get("dir_ml", "NEUTRAL")
    conf_ml = pred_hibrida_horizonte.get("confianza_ml", 50.0)

    if dir_score == dir_ml and dir_ml != "NEUTRAL":
        return "🟢 CONFIRMA", f"{dir_ml} ({conf_ml:.1f}%)"
    elif (dir_score == "BULLISH" and dir_ml == "BEARISH") or (dir_score == "BEARISH" and dir_ml == "BULLISH"):
        return "🔴 CONTRADICE", f"{dir_ml} ({conf_ml:.1f}%)"
    else:
        return "🟡 NEUTRO", f"{dir_ml} ({conf_ml:.1f}%)"


def calcular_confianza_senal(score_mai, pred_hibridas, dcf_val, precio_actual, datos_completos=True):
    """
    Calcula una puntuación de confianza (0 a 100) basada en la consistencia de los inputs.
    """
    puntos = 50.0  # Base neutral

    # 1. Fuerza del Score Principal (hasta +/- 15 pts)
    dist_score = abs(score_mai - 50.0)
    puntos += (dist_score / 50.0) * 15.0

    # 2. Consistencia entre los 4 horizontes de ML (hasta +/- 20 pts)
    confirmaciones = 0
    contradicciones = 0
    for hk, ph in pred_hibridas.items():
        estado_ml, _ = evaluar_confirmacion_ml(ph)
        if estado_ml == "🟢 CONFIRMA":
            confirmaciones += 1
        elif estado_ml == "🔴 CONTRADICE":
            contradicciones += 1

    puntos += (confirmaciones * 5.0) - (contradicciones * 5.0)

    # 3. Alineación con Valoración DCF (hasta +/- 10 pts)
    if dcf_val and precio_actual and precio_actual > 0:
        upside = ((dcf_val - precio_actual) / precio_actual) * 100.0
        if score_mai >= 60 and upside > 10:
            puntos += 10.0
        elif score_mai <= 40 and upside < -10:
            puntos += 10.0
        elif (score_mai >= 60 and upside < -15) or (score_mai <= 40 and upside > 15):
            puntos -= 10.0

    # 4. Castigo por calidad o faltante de datos
    if not datos_completos:
        puntos -= 15.0

    return float(np.clip(round(puntos, 1), 10.0, 98.0))


def generar_senal_multimetrica(score_mai, pred_hibridas, dcf_val, precio_actual, rsi=None, beta=None):
    """
    Sintetiza todas las dimensiones para determinar la Señal de Inversión y los motivos clave.
    """
    motivos = []
    
    # Evaluar componente fundamental / valoración
    upside_dcf = None
    if dcf_val and precio_actual and precio_actual > 0:
        upside_dcf = ((dcf_val - precio_actual) / precio_actual) * 100.0

    # Evaluar consenso ML (Horizontes clave 20D y 60D)
    ml_20d = pred_hibridas.get("20D", {})
    ml_60d = pred_hibridas.get("60D", {})
    dir_ml_20d = ml_20d.get("dir_ml", "NEUTRAL")
    dir_ml_60d = ml_60d.get("dir_ml", "NEUTRAL")

    # Regla 1: COMPRA FUERTE
    if score_mai >= 70 and (dir_ml_20d == "BULLISH" or dir_ml_60d == "BULLISH") and (upside_dcf is None or upside_dcf > -5.0):
        tipo_senal = "🟢 COMPRA FUERTE"
        motivos.append(f"MARKET AI Score sobresaliente ({score_mai:.1f}/100).")
        motivos.append(f"Modelos ML respaldan tendencia alcista a medio plazo (20D: {dir_ml_20d}, 60D: {dir_ml_60d}).")
        if upside_dcf and upside_dcf > 10:
            motivos.append(f"Valoración DCF atractiva con margen de seguridad del {upside_dcf:.1f}%.")

    # Regla 2: COMPRA
    elif score_mai >= 58 and (dir_ml_20d != "BEARISH") and (upside_dcf is None or upside_dcf > -15.0):
        tipo_senal = "🟢 COMPRA"
        motivos.append(f"Score técnico y fundamental sólido ({score_mai:.1f}/100).")
        motivos.append("Sesgo predictivo híbrido mayoritariamente alcista.")
        if rsi and rsi < 65:
            motivos.append(f"RSI en zona templada ({rsi:.1f}), sin sobrecompra extrema.")

    # Regla 3: VENTA
    elif score_mai <= 38 or (score_mai <= 45 and dir_ml_20d == "BEARISH" and dir_ml_60d == "BEARISH"):
        tipo_senal = "🔴 VENTA"
        motivos.append(f"Deterioro marcado en el Score Global ({score_mai:.1f}/100).")
        motivos.append(f"Proyección negativa de Machine Learning en horizontes clave.")
        if upside_dcf and upside_dcf < -10:
            motivos.append(f"Cotización sobrevalorada frente al Fair Value DCF (${dcf_val:.2f}).")

    # Regla 4: VIGILAR
    elif (score_mai >= 52 and (upside_dcf and upside_dcf < -15)) or (score_mai <= 48 and (upside_dcf and upside_dcf > 20)) or (rsi and (rsi > 72 or rsi < 30)):
        tipo_senal = "🟠 VIGILAR"
        motivos.append("Conflicto entre la valoración intrínseca y el impulso técnico del precio.")
        if rsi and rsi > 72:
            motivos.append(f"Alerta técnica: RSI sobrecomprado ({rsi:.1f}). Posible corrección.")
        elif rsi and rsi < 30:
            motivos.append(f"Alerta técnica: RSI sobrevendido ({rsi:.1f}). Buscar confirmación de suelo.")

    # Regla 5: MANTENER
    else:
        tipo_senal = "🟡 MANTENER"
        motivos.append(f"Score en zona neutral ({score_mai:.1f}/100).")
        motivos.append("Señales contrapuestas o falta de catalizadores claros a corto plazo.")

    confianza = calcular_confianza_senal(score_mai, pred_hibridas, dcf_val, precio_actual)

    return {
        "tipo_senal": tipo_senal,
        "confianza": confianza,
        "motivos": motivos
    }


def procesar_alertas_y_guardar(ticker, precio_actual, score_mai, pred_hibridas, dcf_val, target_analistas, dict_senal):
    """
    Guarda el registro en el historial y genera alertas únicamente si existe un cambio de estado real.
    """
    historial = _cargar_historial_senales()

    # Buscar la última señal registrada para este ticker
    registros_ticker = [r for r in historial if r.get("ticker") == ticker]
    ultima_registro = registros_ticker[-1] if registros_ticker else None

    alertas_generadas = []
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")

    senal_actual = dict_senal["tipo_senal"]
    confianza_actual = dict_senal["confianza"]

    if ultima_registro:
        senal_previa = ultima_registro.get("tipo_senal")
        confianza_previa = ultima_registro.get("confianza", 50.0)
        score_previo = ultima_registro.get("score_mai", 50.0)

        # 1. Alerta de Cambio de Señal
        if senal_previa != senal_actual:
            alertas_generadas.append({
                "tipo": "🚨 CAMBIO DE SEÑAL",
                "mensaje": f"Cambio de tendencia detectado: De {senal_previa} ➔ {senal_actual}."
            })

        # 2. Alerta de Mejora / Deterioro de Score
        if score_mai >= score_previo + 8.0:
            alertas_generadas.append({
                "tipo": "🚀 MEJORA DE SCORE",
                "mensaje": f"Incremento notable del Market AI Score (+{score_mai - score_previo:.1f} pts)."
            })
        elif score_mai <= score_previo - 8.0:
            alertas_generadas.append({
                "tipo": "📉 DETERIORO DE SCORE",
                "mensaje": f"Caída relevante en la puntuación técnica/fundamental (-{score_previo - score_mai:.1f} pts)."
            })

        # 3. Alerta de Cambio Importante en ML (20D)
        ml_prev = ultima_registro.get("dir_ml_20d", "NEUTRAL")
        ml_curr = pred_hibridas.get("20D", {}).get("dir_ml", "NEUTRAL")
        if ml_prev != ml_curr and ml_curr != "NEUTRAL":
            alertas_generadas.append({
                "tipo": "🤖 CAMBIO EN ML",
                "mensaje": f"El modelo ML (20D) ha cambiado su proyección a: {ml_curr}."
            })
    else:
        # Primer registro histórico para el ticker
        alertas_generadas.append({
            "tipo": "ℹ️ INICIALIZACIÓN",
            "mensaje": f"Primer registro de seguimiento generado en modo {senal_actual}."
        })

    # Guardar en JSON (evitando duplicar si se analiza múltiples veces el mismo día)
    ya_guardado_hoy = [r for r in registros_ticker if r.get("fecha", "").startswith(fecha_hoy)]
    if not ya_guardado_hoy:
        nuevo_registro = {
            "ticker": ticker,
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precio": float(precio_actual),
            "tipo_senal": senal_actual,
            "confianza": confianza_actual,
            "score_mai": float(score_mai),
            "dir_ml_20d": pred_hibridas.get("20D", {}).get("dir_ml", "NEUTRAL"),
            "dcf": float(dcf_val) if dcf_val else None,
            "target_analistas": float(target_analistas) if target_analistas else None,
            "motivos": dict_senal["motivos"]
        }
        historial.append(nuevo_registro)
        _guardar_historial_senales(historial)

    return alertas_generadas


def simular_senal_historica(df_row, score_historico):
    """
    Función auxiliar para adaptar la generación de señales durante el Backtesting
    sin incurrir en Look-Ahead Bias.
    """
    rsi = df_row.get("RSI_14", 50.0)
    
    if score_historico >= 68:
        return "COMPRA FUERTE"
    elif score_historico >= 56:
        return "COMPRA"
    elif score_historico <= 38:
        return "VENTA"
    elif rsi > 70 or rsi < 30:
        return "VIGILAR"
    else:
        return "MANTENER"
