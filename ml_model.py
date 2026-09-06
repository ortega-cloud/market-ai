import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml_dataset_engine import generar_ml_dataset

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "market_ai_model.pkl")


def entrenar_modelo_ml(ticker="AAPL", periodo="5y", es_metal=False):
    """
    Entrena un modelo RandomForestClassifier con división temporal (70% Train, 30% Test),
    imputación segura y exportación a disco local.
    """
    df_ml, err = generar_ml_dataset(ticker=ticker, periodo=periodo, es_metal=es_metal)
    
    if df_ml is None or df_ml.empty:
        return None, f"No se pudo cargar el dataset para entrenar: {err or ''}"

    df_ml = df_ml.dropna(subset=["Target_Class_1W_4W"]).copy()
    
    if len(df_ml) < 60:
        return None, f"Insuficientes muestras con Target etiquetado (mínimo 60, disponibles: {len(df_ml)})."

    cols_excluir = [
        "Fecha", "Ticker", "Mercado", 
        "Target_Ret_1D_5D", "Target_Ret_1W_4W", "Target_Ret_1M_3M", "Target_Ret_3M_6M", 
        "Target_Class_1W_4W"
    ]
    
    candidates_features = [c for c in df_ml.columns if c not in cols_excluir]
    
    # 1. Filtro de features válidas (se descartan las totalmente vacías o con varianza 0)
    valid_features = []
    for c in candidates_features:
        if df_ml[c].dropna().nunique() > 1:
            valid_features.append(c)
            
    if not valid_features:
        return None, "No hay suficiente variabilidad en las features para entrenar un modelo."

    # 2. Imputación segura por mediana histórica (evitando look-ahead bias)
    X = df_ml[valid_features].copy()
    y = df_ml["Target_Class_1W_4W"].copy()

    # Imputación por mediana de la feature
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    # 3. División Temporal Estricta Cronológica (70% Train, 30% Test)
    split_idx = int(len(df_ml) * 0.70)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        return None, "Error en la división temporal de Train/Test."

    # 4. Entrenamiento del modelo RandomForest
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=42, 
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 5. Evaluación de Desempeño
    y_pred_test = clf.predict(X_test)
    y_proba_test = clf.predict_proba(X_test)
    
    labels = np.unique(np.concatenate([y_train, y_test]))
    
    acc = float(accuracy_score(y_test, y_pred_test))
    prec = float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred_test, labels=clf.classes_)

    # 6. Importancia de Variables (Feature Importance)
    fi_df = pd.DataFrame({
        "Feature": valid_features,
        "Importancia (%)": np.round(clf.feature_importances_ * 100.0, 2)
    }).sort_values(by="Importancia (%)", ascending=False).reset_index(drop=True)

    # 7. Inferencia sobre el registro más reciente (Hoy)
    x_latest = X.iloc[[-1]]
    latest_pred = clf.predict(x_latest)[0]
    latest_probas = clf.predict_proba(x_latest)[0]
    
    proba_map = dict(zip(clf.classes_, latest_probas))
    confianza_ml = float(np.max(latest_probas) * 100.0)

    # 8. Comparativa de Estrategias sobre la muestra TEST
    df_test_full = df_ml.iloc[split_idx:].copy()
    df_test_full["ML_Pred"] = y_pred_test
    
    # Rentabilidad acumulada
    rets = df_test_full["Target_Ret_1W_4W"].fillna(0)
    
    # Strategy ML
    signal_ml = df_test_full["ML_Pred"].map({"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}).fillna(0)
    ret_ml = (rets * signal_ml).sum()

    # Strategy Algoritmo Actual
    signal_act = df_test_full["Direccion_Senal"].map({"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}).fillna(0)
    ret_act = (rets * signal_act).sum()

    # Strategy Buy & Hold
    ret_bh = rets.sum()

    # 9. Guardar Artefacto Localmente (models/market_ai_model.pkl)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": clf,
        "features": valid_features,
        "classes": clf.classes_
    }, MODEL_FILE)

    res_summary = {
        "ticker": ticker,
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
        "confianza_ml": confianza_ml,
        "proba_map": proba_map,
        "comp_ret_ml": ret_ml,
        "comp_ret_actual": ret_act,
        "comp_ret_bh": ret_bh,
        "features_utilizadas": valid_features
    }

    return res_summary, None


def predecir_ml_actual(ticker="AAPL", es_metal=False):
    """
    Función de inferencia ligera que reutiliza el modelo guardado en pkl
    para realizar predicciones con los datos actuales.
    """
    if not os.path.exists(MODEL_FILE):
        return None, "No existe ningún modelo ML guardado. Entrena uno en la interfaz primero."

    try:
        data = joblib.load(MODEL_FILE)
        clf = data["model"]
        features = data["features"]
        
        df_ml, err = generar_ml_dataset(ticker=ticker, periodo="1y", es_metal=es_metal)
        if df_ml is None or df_ml.empty:
            return None, f"No se obtuvieron datos para inferencia: {err or ''}"

        x_latest = df_ml[features].iloc[[-1]].fillna(0)
        
        pred = clf.predict(x_latest)[0]
        probas = clf.predict_proba(x_latest)[0]
        proba_map = dict(zip(clf.classes_, probas))
        confianza = float(np.max(probas) * 100.0)

        return {
            "direccion": pred,
            "confianza": confianza,
            "proba_map": proba_map
        }, None
    except Exception as e:
        return None, f"Error al cargar/ejecutar el modelo ML guardado: {str(e)}"
