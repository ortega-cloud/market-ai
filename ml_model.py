import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml_dataset_engine import generar_ml_dataset, clasificar_target

MODEL_DIR = "models"

HORIZONTES_CONFIG = {
    "5D": {
        "target_ret_col": "Target_Ret_1D_5D",
        "model_file": os.path.join(MODEL_DIR, "market_ai_5d.pkl"),
        "label": "5 Días (Corto Plazo)"
    },
    "20D": {
        "target_ret_col": "Target_Ret_1W_4W",
        "model_file": os.path.join(MODEL_DIR, "market_ai_20d.pkl"),
        "label": "20 Días (Medio Plazo - 1 Mes)"
    },
    "60D": {
        "target_ret_col": "Target_Ret_1M_3M",
        "model_file": os.path.join(MODEL_DIR, "market_ai_60d.pkl"),
        "label": "60 Días (Medio-Largo Plazo - 3 Meses)"
    },
    "120D": {
        "target_ret_col": "Target_Ret_3M_6M",
        "model_file": os.path.join(MODEL_DIR, "market_ai_120d.pkl"),
        "label": "120 Días (Largo Plazo - 6 Meses)"
    }
}


def entrenar_modelo_horizonte(df_ml, target_col, model_path):
    """
    Entrena un RandomForestClassifier para un horizonte específico usando
    división temporal 70% Train / 30% Test e imputación segura.
    """
    df_h = df_ml.copy()
    
    # Crear la etiqueta categórica explícita si no existe en el dataset
    df_h["Target_Class"] = df_h[target_col].apply(clasificar_target)
    df_h = df_h.dropna(subset=["Target_Class"]).copy()
    
    if len(df_h) < 60:
        return None, f"Datos insuficientes para entrenar este horizonte ({len(df_h)} muestras válidas, mínimo 60)."

    cols_excluir = [
        "Fecha", "Ticker", "Mercado", "Direccion_Senal", "Confianza",
        "Target_Ret_1D_5D", "Target_Ret_1W_4W", "Target_Ret_1M_3M", "Target_Ret_3M_6M",
        "Target_Class_1W_4W", "Target_Class"
    ]
    
    candidates = [c for c in df_h.columns if c not in cols_excluir]
    valid_features = [c for c in candidates if df_h[c].dropna().nunique() > 1]
    
    if not valid_features:
        return None, "Sin features válidas con suficiente variabilidad."

    X = df_h[valid_features].copy()
    y = df_h["Target_Class"].copy()

    # Imputación segura por mediana (evita look-ahead bias)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    # División Temporal Estricta (70% Train / 30% Test)
    split_idx = int(len(df_h) * 0.70)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        return None, "Error en la división temporal de muestras."

    # Entrenar RandomForest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluación en Muestra TEST (Out-of-Sample)
    y_pred = clf.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # Importancia de Variables
    fi_df = pd.DataFrame({
        "Feature": valid_features,
        "Importancia (%)": np.round(clf.feature_importances_ * 100.0, 2)
    }).sort_values(by="Importancia (%)", ascending=False).reset_index(drop=True)

    # Comparación de Estrategias y Tasa de Acierto sobre TEST
    df_test = df_h.iloc[split_idx:].copy()
    df_test["ML_Pred"] = y_pred
    
    rets = df_test[target_col].fillna(0)
    signals = df_test["ML_Pred"].map({"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}).fillna(0)
    
    # Rentabilidad acumulada y media por operación
    rets_estrategia = rets * signals
    rent_total = float(rets_estrategia.sum())
    rent_media = float(rets_estrategia.mean()) if len(rets_estrategia) > 0 else 0.0
    
    # Hit rate de las señales activas (Bullish/Bearish)
    senales_activas = df_test[df_test["ML_Pred"].isin(["BULLISH", "BEARISH"])]
    if len(senales_activas) > 0:
        hits = sum(
            (row["ML_Pred"] == "BULLISH" and row[target_col] > 0) or
            (row["ML_Pred"] == "BEARISH" and row[target_col] < 0)
            for _, row in senales_activas.iterrows()
        )
        hit_rate = (hits / len(senales_activas)) * 100.0
    else:
        hit_rate = 0.0

    # Guardar Artefacto en Disco Local
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": clf,
        "features": valid_features,
        "classes": clf.classes_
    }, model_path)

    # Inferencia en el registro actual (último día)
    x_latest = X.iloc[[-1]]
    latest_pred = clf.predict(x_latest)[0]
    latest_probas = clf.predict_proba(x_latest)[0]
    proba_map = dict(zip(clf.classes_, latest_probas))
    confianza = float(np.max(latest_probas) * 100.0)

   return {
        "model": clf,
        "_model_obj": clf,  # Servirá para compatibilidad directa con hybrid_engine.py
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classes": clf.classes_,
        "feature_importance": fi_df,
        "latest_pred": latest_pred,
        "confianza": confianza,
        "proba_map": proba_map,
        "rent_total": rent_total,
        "rent_media": rent_media,
        "hit_rate": hit_rate,
        "num_senales": len(senales_activas),
        "features_utilizadas": valid_features
    }, None


def entrenar_modelos_multihorizonte(ticker="AAPL", periodo="5y", es_metal=False):
    """
    Genera el dataset e independientemente entrena y evalúa los modelos para
    los horizontes: 5D, 20D, 60D y 120D.
    """
    df_ml, err = generar_ml_dataset(ticker=ticker, periodo=periodo, es_metal=es_metal)
    if df_ml is None or df_ml.empty:
        return None, f"Error al generar el dataset base: {err or ''}"

    resultados = {}
    
    for horiz_key, cfg in HORIZONTES_CONFIG.items():
        res, err_h = entrenar_modelo_horizonte(
            df_ml=df_ml,
            target_col=cfg["target_ret_col"],
            model_path=cfg["model_file"]
        )
        if res is None:
            resultados[horiz_key] = {"error": err_h}
        else:
            resultados[horiz_key] = res
            
    return resultados, None


def predecir_multihorizonte_actual(ticker="AAPL", es_metal=False):
    """
    Carga los modelos .pkl guardados en disco y genera una predicción
    para los 4 horizontes con los datos actuales.
    """
    df_ml, err = generar_ml_dataset(ticker=ticker, periodo="1y", es_metal=es_metal)
    if df_ml is None or df_ml.empty:
        return None, f"No se obtuvieron datos actuales para predicción: {err or ''}"

    predicciones = {}
    
    for horiz_key, cfg in HORIZONTES_CONFIG.items():
        model_path = cfg["model_file"]
        if not os.path.exists(model_path):
            predicciones[horiz_key] = {"error": "N/D — Modelo no entrenado localmente"}
            continue
            
        try:
            data = joblib.load(model_path)
            clf = data["model"]
            features = data["features"]
            
            x_latest = df_ml[features].iloc[[-1]].fillna(0)
            
            pred = clf.predict(x_latest)[0]
            probas = clf.predict_proba(x_latest)[0]
            proba_map = dict(zip(clf.classes_, probas))
            confianza = float(np.max(probas) * 100.0)
            
            predicciones[horiz_key] = {
                "label": cfg["label"],
                "direccion": pred,
                "confianza": confianza,
                "proba_map": proba_map
            }
        except Exception as e:
            predicciones[horiz_key] = {"error": f"Error al inferir: {str(e)}"}
            
    return predicciones, None
