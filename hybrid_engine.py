import numpy as np
import pandas as pd
from ml_model import HORIZONTES_CONFIG, predecir_multihorizonte_actual, entrenar_modelo_horizonte
from ml_dataset_engine import generar_ml_dataset, clasificar_target
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def score_a_probabilidades(score):
    """
    Convierte el MARKET AI Score (0-100) en probabilidades (Bullish, Neutral, Bearish).
    """
    score = max(0.0, min(100.0, float(score)))
    
    if score >= 60:
        p_bullish = 0.5 + (score - 60) * (0.45 / 40.0)
        p_bearish = (100 - score) * (0.2 / 40.0)
        p_neutral = 1.0 - p_bullish - p_bearish
    elif score <= 40:
        p_bearish = 0.5 + (40 - score) * (0.45 / 40.0)
        p_bullish = score * (0.2 / 40.0)
        p_neutral = 1.0 - p_bullish - p_bearish
    else:
        p_neutral = 0.5 + (50 - abs(score - 50)) * (0.3 / 10.0)
        p_bullish = (score / 100.0) * (1.0 - p_neutral)
        p_bearish = 1.0 - p_neutral - p_bullish

    total = p_bullish + p_neutral + p_bearish
    return {
        "BULLISH": (p_bullish / total) * 100.0,
        "NEUTRAL": (p_neutral / total) * 100.0,
        "BEARISH": (p_bearish / total) * 100.0
    }


def calcular_prediccion_hibrida(score_market_ai, ml_proba_map, peso_market_ai=0.60, peso_ml=0.40):
    """
    Combina las probabilidades derivadas del Score con las del ML mediante
    ponderación lineal (60/40), detecta conflictos/acuerdos y calcula la confianza final.
    """
    if not ml_proba_map:
        p_score = score_a_probabilidades(score_market_ai)
        dir_s = max(p_score, key=p_score.get)
        return {
            "direccion_hibrida": dir_s,
            "confianza_hibrida": round(p_score[dir_s], 1),
            "score_hibrido": round(score_market_ai, 1),
            "conflicto": False,
            "mensaje_conflicto": None,
            "probas_combinadas": p_score
        }

    p_score = score_a_probabilidades(score_market_ai)
    p_ml = {cls: ml_proba_map.get(cls, 0.0) * 100.0 if ml_proba_map.get(cls, 0.0) <= 1.0 else ml_proba_map.get(cls, 0.0)
            for cls in ["BULLISH", "NEUTRAL", "BEARISH"]}

    # Combinación ponderada (60% Score / 40% ML)
    p_hibrida = {}
    for cls in ["BULLISH", "NEUTRAL", "BEARISH"]:
        p_hibrida[cls] = (p_score[cls] * peso_market_ai) + (p_ml[cls] * peso_ml)

    dir_score = max(p_score, key=p_score.get)
    dir_ml = max(p_ml, key=p_ml.get)
    dir_final = max(p_hibrida, key=p_hibrida.get)

    confianza_base = p_hibrida[dir_final]

    # Gestión de Conflictos y Acuerdos
    es_conflicto = (dir_score in ["BULLISH", "BEARISH"]) and (dir_ml in ["BULLISH", "BEARISH"]) and (dir_score != dir_ml)
    es_acuerdo = (dir_score == dir_ml) and (dir_score in ["BULLISH", "BEARISH"])

    mensaje_conflicto = None

    if es_conflicto:
        confianza_final = confianza_base * 0.75  # Penalización del 25%
        mensaje_conflicto = "⚠️ CONFLICTO DE SEÑALES: El algoritmo técnico/fundamental y el modelo ML presentan señales opuestas."
    elif es_acuerdo:
        confianza_final = confianza_base * 1.15  # Bonificación del 15%
    else:
        confianza_final = confianza_base

    # Límite absoluto de seguridad en la confianza (máximo 95%)
    confianza_final = min(95.0, round(confianza_final, 1))

    # Cálculo del Score Híbrido equivalente (0-100)
    score_hibrido = (p_hibrida["BULLISH"] * 100.0 + p_hibrida["NEUTRAL"] * 50.0) / 100.0

    return {
        "direccion_hibrida": dir_final,
        "confianza_hibrida": confianza_final,
        "score_hibrido": round(score_hibrido, 1),
        "conflicto": es_conflicto,
        "mensaje_conflicto": mensaje_conflicto,
        "probas_combinadas": {k: round(v, 1) for k, v in p_hibrida.items()},
        "dir_score": dir_score,
        "dir_ml": dir_ml
    }


def evaluar_backtest_hibrido(ticker="AAPL", periodo="5y", es_metal=False):
    """
    Ejecuta una evaluación sobre la muestra de VALIDACIÓN/TEST (30% Out-of-Sample)
    comparando MARKET AI, ML, HYBRID y BUY & HOLD para todos los horizontes.
    """
    df_ml, err = generar_ml_dataset(ticker=ticker, periodo=periodo, es_metal=es_metal)
    if df_ml is None or df_ml.empty:
        return None, f"Error generando dataset para backtest: {err}"

    resultados_comparativa = {}

    for h_key, cfg in HORIZONTES_CONFIG.items():
        target_col = cfg["target_ret_col"]
        
        df_h = df_ml.copy()
        df_h["Target_Class"] = df_h[target_col].apply(clasificar_target)
        df_h = df_h.dropna(subset=["Target_Class"]).copy()
        
        if len(df_h) < 60:
            resultados_comparativa[h_key] = {"error": "Insuficientes muestras para validación"}
            continue

        split_idx = int(len(df_h) * 0.70)
        df_test = df_h.iloc[split_idx:].copy()

        res_ml, err_m = entrenar_modelo_horizonte(df_ml, target_col, cfg["model_file"])
        if res_ml is None:
            resultados_comparativa[h_key] = {"error": err_m}
            continue

       # Obtener o cargar el modelo entrenado
        clf = None
        if isinstance(res_ml.get("model"), RandomForestClassifier):
            clf = res_ml["model"]
        elif isinstance(res_ml.get("_model_obj"), RandomForestClassifier):
            clf = res_ml["_model_obj"]
        elif os.path.exists(cfg["model_file"]):
            # Carga de respaldo directa desde el archivo joblib
            saved_data = joblib.load(cfg["model_file"])
            clf = saved_data.get("model")

        if clf is None or not hasattr(clf, "predict"):
            resultados_comparativa[h_key] = {"error": "Error: Objeto de modelo ML no válido para inferencia."}
            continue

        features = res_ml.get("features_utilizadas", [])
        X_test = df_test[features].fillna(0)

        # Inferencia segura de predicciones y probabilidades
        ml_preds = clf.predict(X_test)
        ml_probas = clf.predict_proba(X_test)

        for idx, (_, row) in enumerate(df_test.iterrows()):
            ret_real = row[target_col]
            score_mai = row.get("Market_AI_Score", 50.0)

            # 1. MARKET AI
            dir_mai = "BULLISH" if score_mai >= 60 else ("BEARISH" if score_mai <= 40 else "NEUTRAL")
            
            # 2. ML
            dir_ml = ml_preds[idx]
            proba_map_idx = dict(zip(clf.classes_, ml_probas[idx]))

            # 3. HYBRID
            res_hib = calcular_prediccion_hibrida(score_mai, proba_map_idx)
            dir_hib = res_hib["direccion_hibrida"]

            metrics["MARKET_AI"].append((dir_mai, ret_real))
            metrics["ML"].append((dir_ml, ret_real))
            metrics["HYBRID"].append((dir_hib, ret_real))
            metrics["BUY_HOLD"].append(("BULLISH", ret_real))

        # Cálculo de desempeño Out-of-Sample por estrategia
        res_h = {}
        for est_name, datos in metrics.items():
            total_senales = len(datos)
            senales_activas = [d for d in datos if d[0] in ["BULLISH", "BEARISH"]]
            
            if len(senales_activas) > 0:
                hits = sum(1 for d, r in senales_activas if (d == "BULLISH" and r > 0) or (d == "BEARISH" and r < 0))
                acierto = (hits / len(senales_activas)) * 100.0
            else:
                acierto = 0.0

            rets_estrategia = [r if d == "BULLISH" else (-r if d == "BEARISH" else 0.0) for d, r in datos]
            rent_acum = sum(rets_estrategia)
            rent_media = np.mean(rets_estrategia) if rets_estrategia else 0.0

            res_h[est_name] = {
                "acierto": round(acierto, 1),
                "rent_media": round(rent_media, 2),
                "rent_acum": round(rent_acum, 2),
                "senales_activas": len(senales_activas),
                "total_evaluadas": total_senales
            }

        resultados_comparativa[h_key] = res_h

    return resultados_comparativa, None
