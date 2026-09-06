import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ml_dataset_engine import generar_ml_dataset, clasificar_target
from ml_model import HORIZONTES_CONFIG, entrenar_modelo_horizonte

MODEL_DIR = "models"
METADATA_FILE = os.path.join(MODEL_DIR, "training_metadata.json")
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.json")


def _cargar_json(path, default=None):
    if default is None:
        default = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _guardar_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def obtener_siguiente_version(horizonte_key):
    """Obtiene la siguiente versión numerada para un horizonte (ej: v001, v002)."""
    history = _cargar_json(HISTORY_FILE, default=[])
    versiones = [
        int(item["version"].replace("v", "")) 
        for item in history 
        if item.get("horizonte") == horizonte_key and "version" in item
    ]
    siguiente = max(versiones, default=0) + 1
    return f"v{siguiente:03d}"


def evaluar_criterio_aceptacion(metricas_viejas, metricas_nuevas):
    """
    Regla de seguridad: Compara modelo anterior vs nuevo.
    Prioriza F1-Score (60%) y Hit Rate en señales activas (40%).
    """
    if not metricas_viejas:
        return True, "Primer entrenamiento inicial (Modelo Aceptado)."

    f1_viejo = metricas_viejas.get("f1_score", 0.0)
    f1_nuevo = metricas_nuevas.get("f1_score", 0.0)
    
    hit_viejo = metricas_viejas.get("hit_rate", 0.0)
    hit_nuevo = metricas_nuevas.get("hit_rate", 0.0)

    score_viejo = (f1_viejo * 0.60) + ((hit_viejo / 100.0) * 0.40)
    score_nuevo = (f1_nuevo * 0.60) + ((hit_nuevo / 100.0) * 0.40)

    # Si el rendimiento nuevo cae más de un 2% relativo, se rechaza
    if score_nuevo >= (score_viejo * 0.98):
        return True, f"Modelo nuevo aceptado (Score: {score_nuevo:.3f} >= Anterior: {score_viejo:.3f})."
    else:
        return False, f"Modelo anterior conservado porque el nuevo no mejora la validación (Score Nuevo: {score_nuevo:.3f} < Anterior: {score_viejo:.3f})."


def ejecutar_reentrenamiento_completo(ticker="AAPL", periodo="5y", es_metal=False, forzar=False):
    """
    Ejecuta el pipeline completo de reentrenamiento, validación Out-of-Sample,
    versionado y actualización de modelos activos.
    """
    metadata = _cargar_json(METADATA_FILE, default={})
    
    # 1. Verificación de Frecuencia (Mínimo 7 días excepto si es forzado)
    ultima_fecha_str = metadata.get("ultima_fecha_entrenamiento")
    if ultima_fecha_str and not forzar:
        ultima_fecha = datetime.strptime(ultima_fecha_str, "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - ultima_fecha) < timedelta(days=7):
            return False, "Reentrenamiento no necesario: Ya se ejecutó dentro de los últimos 7 días.", metadata

    # 2. Cargar Dataset Actualizado (Evitando Look-Ahead Bias)
    df_ml, err = generar_ml_dataset(ticker=ticker, periodo=periodo, es_metal=es_metal)
    if df_ml is None or df_ml.empty:
        return False, f"Error al generar/actualizar dataset: {err or ''}", metadata

    # Limpieza, orden cronológico estricto y deduplicación
    df_ml = df_ml.drop_duplicates(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

    fecha_max_datos = str(df_ml["Fecha"].max())
    registros_totales = len(df_ml)

    # Comprobar si hay datos realmente nuevos
    if metadata.get("fecha_max_datos") == fecha_max_datos and not forzar:
        return False, "No se detectaron registros o datos históricos nuevos.", metadata

    modelos_activos = metadata.get("modelos_activos", {})
    metricas_anteriores = metadata.get("metricas_ultimas", {})
    history = _cargar_json(HISTORY_FILE, default=[])

    nuevas_metricas_dict = {}
    resumen_ejecucion = []

    for h_key, cfg in HORIZONTES_CONFIG.items():
        ver_str = obtener_siguiente_version(h_key)
        v_model_path = os.path.join(MODEL_DIR, f"market_ai_{h_key.lower()}_{ver_str}.pkl")

        # Reentrenamiento en el dataset
        res_h, err_h = entrenar_modelo_horizonte(df_ml, cfg["target_ret_col"], v_model_path)

        if res_h is None:
            resumen_ejecucion.append(f"{h_key}: ❌ Falló entrenamiento ({err_h})")
            continue

        met_viejas = metricas_anteriores.get(h_key, {})
        aceptado, razon = evaluar_criterio_aceptacion(met_viejas, res_h)

        log_entry = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "horizonte": h_key,
            "version": ver_str,
            "registros": registros_totales,
            "accuracy": round(res_h["accuracy"], 4),
            "f1_score": round(res_h["f1_score"], 4),
            "precision": round(res_h["precision"], 4),
            "recall": round(res_h["recall"], 4),
            "hit_rate": round(res_h["hit_rate"], 2),
            "rentabilidad_acum": round(res_h["rent_total"], 2),
            "aceptado": aceptado,
            "razon": razon
        }
        history.append(log_entry)

        if aceptado:
            # Sobrescribir archivo de producción activo
            joblib.dump({
                "model": res_h["_model_obj"],
                "features": res_h["features_utilizadas"],
                "classes": res_h["classes"]
            }, cfg["model_file"])

            modelos_activos[h_key] = ver_str
            nuevas_metricas_dict[h_key] = {
                "accuracy": res_h["accuracy"],
                "f1_score": res_h["f1_score"],
                "precision": res_h["precision"],
                "recall": res_h["recall"],
                "hit_rate": res_h["hit_rate"],
                "rent_total": res_h["rent_total"]
            }
            resumen_ejecucion.append(f"{h_key} ({ver_str}): 🟢 ACTIVO — {razon}")
        else:
            # Se conserva la versión vieja en producción
            nuevas_metricas_dict[h_key] = met_viejas
            resumen_ejecucion.append(f"{h_key} ({ver_str}): 🟠 CONSERVADO ANTERIOR — {razon}")

    # Guardar metadata actualizada
    metadata_actualizada = {
        "ultima_fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fecha_max_datos": fecha_max_datos,
        "registros_totales": registros_totales,
        "modelos_activos": modelos_activos,
        "metricas_ultimas": nuevas_metricas_dict
    }

    _guardar_json(METADATA_FILE, metadata_actualizada)
    _guardar_json(HISTORY_FILE, history)

    msg_final = " | ".join(resumen_ejecucion)
    return True, msg_final, metadata_actualizada


def comprobar_reentrenamiento_automatico(ticker="AAPL", periodo="5y", es_metal=False):
    """Función de verificación no bloqueante llamada al iniciar la app."""
    metadata = _cargar_json(METADATA_FILE, default={})
    ultima_fecha_str = metadata.get("ultima_fecha_entrenamiento")

    if ultima_fecha_str:
        ultima_fecha = datetime.strptime(ultima_fecha_str, "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - ultima_fecha) < timedelta(days=7):
            return False, "Dentro del periodo de vigencia (7 días)."

    return ejecutar_reentrenamiento_completo(ticker=ticker, periodo=periodo, es_metal=es_metal, forzar=False)
