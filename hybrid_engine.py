import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml_dataset_engine import generar_ml_dataset, clasificar_target
from ml_model import HORIZONTES_CONFIG, entrenar_modelo_horizonte


def evaluar_estrategia_retornos(tuples_signal_ret):
    """
    Calcula el rendimiento, número de operaciones y hit rate dado un listado de (señal, retorno_real).
    Señal puede ser: 'BULLISH', 'BEARISH', 'NEUTRAL' o 1, -1, 0.
    """
    if not tuples_signal_ret:
        return {"rent_total": 0.0, "hit_rate": 0.0, "num_senales": 0}

    rets = []
    hits = 0
    total_ops = 0

    for sig, ret in tuples_signal_ret:
        if sig in ["BULLISH", 1]:
            factor = 1.0
        elif sig in ["BEARISH", -1]:
            factor = -1.0
        else:
            factor = 0.0

        if factor != 0.0:
            total_ops += 1
            ret_est = float(ret) * factor
            rets.append(ret_est)
            if ret_est > 0:
                hits += 1

    rent_total = float(np.sum(rets)) if rets else 0.0
    hit_rate = (hits / total_ops * 100.0) if total_ops > 0 else 0.0

    return {
        "rent_total": rent_total,
        "hit_rate": hit_rate,
        "num_senales": total_ops
    }


def evaluar_backtest_hibrido(ticker="AAPL", periodo="5y", es_metal=False):
    """
    Ejecuta una evaluación comparativa entre Técnico, Sentimiento, ML y el Ensamble Market AI
    para todos los horizontes configurados.
    """
    df_ml, err = generar_ml_dataset(ticker=ticker, periodo=periodo, es_metal=es_metal)
    if df_ml is None or df_ml.empty:
        return None, f"Error al generar el dataset para el motor híbrido: {err or ''}"

    resultados_comparativa = {}

    for h_key, cfg in HORIZONTES_CONFIG.items():
        res_ml, err_h = entrenar_modelo_horizonte(
            df_ml=df_ml,
            target_col=cfg["target_ret_col"],
            model_path=cfg["model_file"]
        )

        if res_ml is None:
            resultados_comparativa[h_key] = {"error": err_h}
            continue

        # Reconstruir df_h con clases válidas para el test
        df_h = df_ml.copy()
        df_h["Target_Class"] = df_h[cfg["target_ret_col"]].apply(clasificar_target)
        df_h = df_h.dropna(subset=["Target_Class"]).copy()

        split_idx = int(len(df_h) * 0.70)
        df_test = df_h.iloc[split_idx:].copy()

        if len(df_test) == 0:
            resultados_comparativa[h_key] = {"error": "Sin datos suficientes para el conjunto de prueba de backtest."}
            continue

        # Obtener el modelo entrenado
        clf = None
        if isinstance(res_ml.get("model"), RandomForestClassifier):
            clf = res_ml["model"]
        elif isinstance(res_ml.get("_model_obj"), RandomForestClassifier):
            clf = res_ml["_model_obj"]
        elif os.path.exists(cfg["model_file"]):
            saved_data = joblib.load(cfg["model_file"])
            clf = saved_data.get("model")

        if clf is None or not hasattr(clf, "predict"):
            resultados_comparativa[h_key] = {"error": "Error: Objeto de modelo ML no disponible para inferencia."}
            continue

        features = res_ml.get("features_utilizadas", [])
        X_test = df_test[features].fillna(0)

        # Predicciones ML
        ml_preds = clf.predict(X_test)

        # DICCIONARIO DE MÉTRICAS (AQUÍ SE DEFINE 'metrics' EXP LÍCITAMENTE)
        metrics = {
            "TECNICO": [],
            "SENTIMIENTO": [],
            "ML": [],
            "MARKET_AI": []
        }

        # Bucle de acumulación sobre las muestras de prueba
        for idx in range(len(df_test)):
            row = df_test.iloc[idx]
            ret_real = row.get(cfg["target_ret_col"], 0.0)

            # Simulación / obtención de señales por capa
            dir_tec = row.get("Signal_Tecnico", "NEUTRAL")
            dir_sent = row.get("Signal_Sentimiento", "NEUTRAL")
            dir_ml = ml_preds[idx] if idx < len(ml_preds) else "NEUTRAL"

            # Voto del ensamble (Market AI)
            votos = [dir_tec, dir_sent, dir_ml]
            bulls = votos.count("BULLISH")
            bears = votos.count("BEARISH")

            if bulls > bears:
                dir_mai = "BULLISH"
            elif bears > bulls:
                dir_mai = "BEARISH"
            else:
                dir_mai = "NEUTRAL"

            # Acumular tuplas para evaluación
            metrics["TECNICO"].append((dir_tec, ret_real))
            metrics["SENTIMIENTO"].append((dir_sent, ret_real))
            metrics["ML"].append((dir_ml, ret_real))
            metrics["MARKET_AI"].append((dir_mai, ret_real))

        # Calcular resultados consolidados por horizonte
        resultados_comparativa[h_key] = {
            "TECNICO": evaluar_estrategia_retornos(metrics["TECNICO"]),
            "SENTIMIENTO": evaluar_estrategia_retornos(metrics["SENTIMIENTO"]),
            "ML": evaluar_estrategia_retornos(metrics["ML"]),
            "MARKET_AI": evaluar_estrategia_retornos(metrics["MARKET_AI"]),
            "test_samples": len(df_test)
        }

    return resultados_comparativa, None
