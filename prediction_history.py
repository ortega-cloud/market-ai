import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from hybrid_engine import calcular_prediccion_hibrida
from ml_model import HORIZONTES_CONFIG

PREDICTION_FILE = os.path.join("models", "prediction_history.json")


def _cargar_predicciones():
    if os.path.exists(PREDICTION_FILE):
        try:
            with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _guardar_predicciones(data):
    os.makedirs(os.path.dirname(PREDICTION_FILE), exist_ok=True)
    with open(PREDICTION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def registrar_prediccion_actual(ticker, precio_actual, predicciones_hibridas_dict, score_mai, modelos_activos=None):
    """
    Registra en tiempo real cada predicción generada por MARKET AI para posterior evaluación.
    """
    historial = _cargar_predicciones()
    fecha_ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for h_key, datos_h in predicciones_hibridas_dict.items():
        if "error" in datos_h:
            continue

        pred_id = f"{ticker}_{h_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Evitar duplicar una predicción exacta del mismo ticker/horizonte realizada el mismo día
        misma_fecha = [p for p in historial if p.get("ticker") == ticker and p.get("horizonte") == h_key and p.get("fecha_hora", "").startswith(fecha_ahora.split(" ")[0])]
        if misma_fecha:
            continue

        ver_modelo = modelos_activos.get(h_key, "v001") if modelos_activos else "v001"

        registro = {
            "id": pred_id,
            "ticker": ticker,
            "fecha_hora": fecha_ahora,
            "precio_inicial": float(precio_actual),
            "horizonte": h_key,
            "score_mai": float(score_mai),
            "dir_mai": datos_h.get("dir_score", "NEUTRAL"),
            "dir_ml": datos_h.get("dir_ml", "NEUTRAL"),
            "dir_hybrid": datos_h.get("direccion_hibrida", "NEUTRAL"),
            "confianza_ml": float(datos_h.get("confianza_ml", 0.0)),
            "confianza_hybrid": float(datos_h.get("confianza_hibrida", 0.0)),
            "version_ml": ver_modelo,
            "estado": "PENDIENTE",
            "resultado": None,
            "precio_final": None,
            "variacion_pct": None,
            "fecha_evaluacion": None
        }
        historial.append(registro)

    _guardar_predicciones(historial)


def evaluar_predicciones_pendientes():
    """
    Obtiene los precios reales posteriores al vencimiento del horizonte y determina si acertó o falló.
    """
    historial = _cargar_predicciones()
    if not historial:
        return 0, 0

    dias_horizonte_map = {"5D": 5, "20D": 20, "60D": 60, "120D": 120}
    actualizados = 0

    tickers_pendientes = list(set([p["ticker"] for p in historial if p["estado"] == "PENDIENTE"]))
    datos_precios = {}

    for t in tickers_pendientes:
        try:
            df_hist = yf.Ticker(t).history(period="1y")
            if not df_hist.empty:
                df_hist.index = pd.to_datetime(df_hist.index).tz_localize(None)
                datos_precios[t] = df_hist
        except Exception:
            continue

    for item in historial:
        if item["estado"] != "PENDIENTE":
            continue

        ticker = item["ticker"]
        if ticker not in datos_precios:
            continue

        fecha_p = datetime.strptime(item["fecha_hora"], "%Y-%m-%d %H:%M:%S")
        dias_req = dias_horizonte_map.get(item["horizonte"], 5)
        fecha_eval_min = fecha_p + timedelta(days=dias_req)

        df_t = datos_precios[ticker]
        df_post = df_t[df_t.index >= fecha_eval_min]

        if not df_post.empty:
            precio_final = float(df_post.iloc[0]["Close"])
            precio_inic = float(item["precio_inicial"])

            var_pct = ((precio_final - precio_inic) / precio_inic) * 100.0

            dir_real = "BULLISH" if var_pct > 0.5 else ("BEARISH" if var_pct < -0.5 else "NEUTRAL")

            acierto_mai = (item["dir_mai"] == dir_real) or (item["dir_mai"] == "BULLISH" and var_pct > 0) or (item["dir_mai"] == "BEARISH" and var_pct < 0)
            acierto_ml = (item["dir_ml"] == dir_real) or (item["dir_ml"] == "BULLISH" and var_pct > 0) or (item["dir_ml"] == "BEARISH" and var_pct < 0)
            acierto_hybrid = (item["dir_hybrid"] == dir_real) or (item["dir_hybrid"] == "BULLISH" and var_pct > 0) or (item["dir_hybrid"] == "BEARISH" and var_pct < 0)

            item["estado"] = "EVALUADO"
            item["precio_final"] = round(precio_final, 2)
            item["variacion_pct"] = round(var_pct, 2)
            item["dir_real"] = dir_real
            item["fecha_evaluacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            item["resultado"] = {
                "acierto_mai": acierto_mai,
                "acierto_ml": acierto_ml,
                "acierto_hybrid": acierto_hybrid
            }
            actualizados += 1

    if actualizados > 0:
        _guardar_predicciones(historial)

    evaluados = [p for p in historial if p["estado"] == "EVALUADO"]
    return len(evaluados), actualizados


def generar_estadisticas_aprendizaje():
    """Genera el desglose comparativo de aciertos, errores y combinaciones de pesos."""
    historial = _cargar_predicciones()
    evaluados = [p for p in historial if p["estado"] == "EVALUADO"]

    if not evaluados:
        return {
            "total_registrados": len(historial),
            "total_evaluados": 0,
            "suficientes_datos": False
        }

    res_h = {}
    for hk in ["5D", "20D", "60D", "120D"]:
        items_hk = [p for p in evaluados if p["horizonte"] == hk]
        if items_hk:
            tot = len(items_hk)
            a_mai = sum(1 for p in items_hk if p["resultado"]["acierto_mai"])
            a_ml = sum(1 for p in items_hk if p["resultado"]["acierto_ml"])
            a_hy = sum(1 for p in items_hk if p["resultado"]["acierto_hybrid"])

            res_h[hk] = {
                "total": tot,
                "pct_mai": round((a_mai / tot) * 100.0, 1),
                "pct_ml": round((a_ml / tot) * 100.0, 1),
                "pct_hybrid": round((a_hy / tot) * 100.0, 1),
                "error_medio_pct": round(float(np.mean([abs(p["variacion_pct"]) for p in items_hk])), 2)
            }
        else:
            res_h[hk] = None

    # Simulación experimental de combinación óptima de pesos
    pesos_test = [(50, 50), (55, 45), (60, 40), (65, 35), (70, 30), (75, 25)]
    mejor_peso = "60/40"
    max_aciertos = -1

    for p_mai, p_ml in pesos_test:
        hits = 0
        for item in evaluados:
            mai_score = item["score_mai"]
            p_score_b = mai_score / 100.0
            p_ml_b = 0.8 if item["dir_ml"] == "BULLISH" else (0.2 if item["dir_ml"] == "BEARISH" else 0.5)

            p_comb = (p_score_b * (p_mai / 100.0)) + (p_ml_b * (p_ml / 100.0))
            dir_comb = "BULLISH" if p_comb > 0.52 else ("BEARISH" if p_comb < 0.48 else "NEUTRAL")

            if (dir_comb == item["dir_real"]) or (dir_comb == "BULLISH" and item["variacion_pct"] > 0) or (dir_comb == "BEARISH" and item["variacion_pct"] < 0):
                hits += 1

        if hits > max_aciertos:
            max_aciertos = hits
            mejor_peso = f"{p_mai}/{p_ml}"

    # Tasa general acumulada de HYBRID
    a_hy_tot = sum(1 for p in evaluados if p["resultado"]["acierto_hybrid"])
    pct_hy_global = round((a_hy_tot / len(evaluados)) * 100.0, 1)

    a_mai_tot = sum(1 for p in evaluados if p["resultado"]["acierto_mai"])
    pct_mai_global = round((a_mai_tot / len(evaluados)) * 100.0, 1)

    a_ml_tot = sum(1 for p in evaluados if p["resultado"]["acierto_ml"])
    pct_ml_global = round((a_ml_tot / len(evaluados)) * 100.0, 1)

    return {
        "total_registrados": len(historial),
        "total_evaluados": len(evaluados),
        "suficientes_datos": True,
        "pct_global_hybrid": pct_hy_global,
        "pct_global_mai": pct_mai_global,
        "pct_global_ml": pct_ml_global,
        "por_horizonte": res_h,
        "mejor_combinacion_pesos": mejor_peso
    }
