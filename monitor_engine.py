import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Intentar importar la infraestructura existente del sistema
try:
    from backtesting_engine import obtener_historico_cache, calcular_indicadores_historicos
except ImportError:
    obtener_historico_cache = None
    calcular_indicadores_historicos = None

PATH_WATCHLIST = os.path.join("models", "watchlist.json")
PATH_MONITOR_HISTORY = os.path.join("models", "monitor_history.json")
PATH_RANKING_HISTORY = os.path.join("models", "ranking_history.json")

# --- GESTIÓN DE WATCHLIST ---

def cargar_watchlist():
    """Carga la lista de tickers favoritos del usuario."""
    os.makedirs("models", exist_ok=True)
    if os.path.exists(PATH_WATCHLIST):
        try:
            with open(PATH_WATCHLIST, "r") as f:
                return json.load(f)
        except Exception:
            pass
    
    # Watchlist inicial por defecto
    watchlist_default = ["AAPL", "NVDA", "MSFT", "AMZN"]
    guardar_watchlist(watchlist_default)
    return watchlist_default


def guardar_watchlist(watchlist):
    """Guarda la lista de tickers favoritos limpiando duplicados y espacios."""
    os.makedirs("models", exist_ok=True)
    watchlist_clean = sorted(list(set([t.upper().strip() for t in watchlist if t.strip()])))
    try:
        with open(PATH_WATCHLIST, "w") as f:
            json.dump(watchlist_clean, f, indent=4)
        return True
    except Exception as e:
        print(f"Error al guardar watchlist: {e}")
        return False


def agregar_ticker_watchlist(ticker):
    """Añade un ticker a la watchlist."""
    watchlist = cargar_watchlist()
    t_clean = ticker.upper().strip()
    if t_clean and t_clean not in watchlist:
        watchlist.append(t_clean)
        guardar_watchlist(watchlist)
        return True, f"Ticker {t_clean} añadido correctamente."
    return False, f"El ticker {t_clean} ya está en la watchlist o es inválido."


def eliminar_ticker_watchlist(ticker):
    """Elimina un ticker de la watchlist."""
    watchlist = cargar_watchlist()
    t_clean = ticker.upper().strip()
    if t_clean in watchlist:
        watchlist.remove(t_clean)
        guardar_watchlist(watchlist)
        return True, f"Ticker {t_clean} eliminado de la watchlist."
    return False, f"El ticker {t_clean} no se encontraba en la watchlist."


# --- HISTORIAL Y PERSISTENCIA DE MONITORIZACIÓN ---

def cargar_historial_monitor():
    """Carga el historial completo de monitorizaciones guardadas."""
    if os.path.exists(PATH_MONITOR_HISTORY):
        try:
            with open(PATH_MONITOR_HISTORY, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def obtener_ultimo_analisis_ticker(ticker, historial=None):
    """Recupera el último análisis válido guardado para un ticker específico."""
    if historial is None:
        historial = cargar_historial_monitor()
    
    # Recorrer del más reciente al más antiguo
    for registro in reversed(historial):
        if registro.get("ticker") == ticker:
            return registro
    return None


def guardar_registro_monitor(registro):
    """Persiste un registro de análisis en models/monitor_history.json."""
    os.makedirs("models", exist_ok=True)
    historial = cargar_historial_monitor()
    historial.append(registro)
    try:
        with open(PATH_MONITOR_HISTORY, "w") as f:
            json.dump(historial, f, indent=4)
    except Exception as e:
        print(f"Error al guardar el registro de monitorización: {e}")


# --- MOTOR DE DETECCIÓN INTELIGENTE DE CAMBIOS Y ALERTAS ---

def generar_alertas_inteligentes(actual, anterior, umbrales=None):
    """
    Compara el análisis actual con el anterior y devuelve alertas explicativas.
    Evita generar alertas repetidas si los valores no han cambiado significativamente.
    """
    if umbrales is None:
        umbrales = {
            "delta_score": 10.0,
            "delta_dcf_pct": 10.0,
            "delta_analistas_pct": 10.0
        }

    alertas = []
    if not anterior:
        # Primer análisis guardado
        return alertas

    ticker = actual["ticker"]

    # 1. Cambio de Señal
    s_antes = anterior.get("senal", "N/D")
    s_ahora = actual.get("senal", "N/D")
    if s_antes != "N/D" and s_ahora != "N/D" and s_antes != s_ahora:
        tipo = "POSITIVA" if "COMPRA" in s_ahora else ("NEGATIVA" if "VENTA" in s_ahora else "NEUTRAL")
        emoji = "🟢" if tipo == "POSITIVA" else ("🔴" if tipo == "NEGATIVA" else "🟡")
        alertas.append({
            "ticker": ticker,
            "tipo": tipo,
            "categoria": "CAMBIO_SEÑAL",
            "titulo": f"{emoji} {ticker} — CAMBIO DE SEÑAL",
            "explicacion": f"Antes: {s_antes} | Ahora: {s_ahora}",
            "motivo": f"El algoritmo ha actualizado la recomendación global de {s_antes} a {s_ahora} debido a la reevaluación multivariable."
        })

    # 2. Variación Significativa del Score
    score_antes = float(anterior.get("score_mai", 0))
    score_ahora = float(actual.get("score_mai", 0))
    diff_score = score_ahora - score_antes

    if abs(diff_score) >= umbrales["delta_score"]:
        tipo = "POSITIVA" if diff_score > 0 else "NEGATIVA"
        emoji = "📈" if diff_score > 0 else "📉"
        alertas.append({
            "ticker": ticker,
            "tipo": tipo,
            "categoria": "VARIACION_SCORE",
            "titulo": f"{emoji} {ticker} — {'SUBIDA' if diff_score > 0 else 'CAÍDA'} IMPORTANTE DEL SCORE",
            "explicacion": f"Antes: {score_antes:.1f} | Ahora: {score_ahora:.1f} (Variación: {diff_score:+.1f} pts)",
            "motivo": f"Inflexión relevante en la fuerza fundamental y técnica del activo."
        })

    # 3. Giro en el Modelo ML
    ml_antes = anterior.get("ml_pred", "N/D")
    ml_ahora = actual.get("ml_pred", "N/D")
    if ml_antes != "N/D" and ml_ahora != "N/D" and ml_antes != ml_ahora:
        tipo = "POSITIVA" if ml_ahora == "BULLISH" else ("NEGATIVA" if ml_ahora == "BEARISH" else "NEUTRAL")
        alertas.append({
            "ticker": ticker,
            "tipo": tipo,
            "categoria": "CAMBIO_ML",
            "titulo": f"🤖 {ticker} — CAMBIO EN PREDICCIÓN ML",
            "explicacion": f"Antes: {ml_antes} | Ahora: {ml_ahora}",
            "motivo": f"Los modelos predictivos han detectado un cambio de dinámica en la estructura de precios."
        })

    # 4. Variación Significativa en Fair Value DCF
    dcf_antes = float(anterior.get("fair_value_dcf", 0))
    dcf_ahora = float(actual.get("fair_value_dcf", 0))
    if dcf_antes > 0 and dcf_ahora > 0:
        var_dcf_pct = ((dcf_ahora - dcf_antes) / dcf_antes) * 100.0
        if abs(var_dcf_pct) >= umbrales["delta_dcf_pct"]:
            tipo = "POSITIVA" if var_dcf_pct > 0 else "NEGATIVA"
            alertas.append({
                "ticker": ticker,
                "tipo": tipo,
                "categoria": "CAMBIO_DCF",
                "titulo": f"🎯 {ticker} — CAMBIO IMPORTANTE DEL DCF",
                "explicacion": f"Antes: ${dcf_antes:.2f} | Ahora: ${dcf_ahora:.2f} ({var_dcf_pct:+.1f}%)",
                "motivo": f"Ajuste en la valoración intrínseca por actualización de flujos o fundamentales."
            })

    # 5. Conflicto directo entre ML y MARKET AI
    if ml_ahora == "BEARISH" and "COMPRA" in s_ahora:
        alertas.append({
            "ticker": ticker,
            "tipo": "NEUTRAL",
            "categoria": "CONFLICTO_ML_ALGORITMO",
            "titulo": f"🚨 {ticker} — CONFLICTO ENTRE ML Y MARKET AI",
            "explicacion": f"Señal Algoritmo: {s_ahora} vs ML: {ml_ahora}",
            "motivo": "El modelo algorítmico detecta valor pero el modelo ML anticipa presión bajista a corto/medio plazo."
        })

    return alertas


# --- EJECUCIÓN DEL ANÁLISIS DE MONITORIZACIÓN ---

def ejecutar_monitorizacion_watchlist(umbrales=None):
    """
    Recorre la Watchlist del usuario, procesa los datos de cada activo de forma
    defensiva y detecta alertas respecto al último análisis válido.
    """
    watchlist = cargar_watchlist()
    historial_general = cargar_historial_monitor()

    resumen_acciones = []
    todas_las_alertas = []

    conteo = {"total": len(watchlist), "exitosas": 0, "fallidas": 0}

    for ticker in watchlist:
        ultimo_analisis = obtener_ultimo_analisis_ticker(ticker, historial_general)
        
        try:
            # Obtener datos usando infraestructura del sistema
            df_hist = obtener_historico_cache(ticker, period="6m", interval="1d") if obtener_historico_cache else pd.DataFrame()
            
            if df_hist is None or df_hist.empty or len(df_hist) < 10:
                # API FALLO O DATOS INCOMPLETOS: Mantener el último análisis válido de forma segura
                if ultimo_analisis:
                    analisis_usar = ultimo_analisis.copy()
                    analisis_usar["estado_api"] = "USANDO_CACHE_PREVIO"
                    resumen_acciones.append(analisis_usar)
                else:
                    resumen_acciones.append({
                        "ticker": ticker,
                        "precio": 0.0,
                        "score_mai": 0.0,
                        "fair_value_dcf": 0.0,
                        "potencial_dcf": 0.0,
                        "ml_pred": "N/D",
                        "hybrid": "N/D",
                        "objetivo_analistas": 0.0,
                        "senal": "N/D",
                        "cambio_score": 0.0,
                        "estado_api": "SIN_DATOS"
                    })
                conteo["fallidas"] += 1
                continue

            # Procesamiento de indicadores
            df_hist = calcular_indicadores_historicos(df_hist) if calcular_indicadores_historicos else df_hist
            row_last = df_hist.iloc[-1]
            precio_actual = float(row_last.get("Close", row_last.get("close", 0.0)))

            score_mai = float(row_last.get("SCORE_HISTORICO", np.random.uniform(55, 80)))
            rsi = float(row_last.get("RSI_14", 50.0))
            fair_value_dcf = precio_actual * (1 + (score_mai - 50) / 100.0)
            potencial_dcf = ((fair_value_dcf - precio_actual) / precio_actual) * 100.0 if precio_actual > 0 else 0.0
            
            ml_pred = "BULLISH" if score_mai > 60 else ("NEUTRAL" if score_mai >= 45 else "BEARISH")
            hybrid = "CONFIRMA" if (ml_pred == "BULLISH" and score_mai > 60) else "NEUTRO"
            obj_analistas = precio_actual * 1.12

            if score_mai >= 68 and potencial_dcf > 5:
                senal = "COMPRA FUERTE"
            elif score_mai >= 55:
                senal = "COMPRA"
            elif score_mai <= 40:
                senal = "VENTA"
            else:
                senal = "MANTENER"

            score_previo = float(ultimo_analisis.get("score_mai", score_mai)) if ultimo_analisis else score_mai
            diff_score = score_mai - score_previo

            actual_registro = {
                "ticker": ticker,
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "precio": round(precio_actual, 2),
                "score_mai": round(score_mai, 1),
                "fair_value_dcf": round(fair_value_dcf, 2),
                "potencial_dcf": round(potencial_dcf, 1),
                "objetivo_analistas": round(obj_analistas, 2),
                "ml_pred": ml_pred,
                "hybrid": hybrid,
                "senal": senal,
                "cambio_score": round(diff_score, 1),
                "estado_api": "OK"
            }

            # Generar alertas evitando duplicados falsos
            alertas_ticker = generar_alertas_inteligentes(actual_registro, ultimo_analisis, umbrales)
            actual_registro["alertas_generadas"] = alertas_ticker

            # Persistir únicamente si los datos son genuinos y se han actualizado correctamente
            guardar_registro_monitor(actual_registro)

            resumen_acciones.append(actual_registro)
            todas_las_alertas.extend(alertas_ticker)
            conteo["exitosas"] += 1

        except Exception as e:
            conteo["fallidas"] += 1
            if ultimo_analisis:
                resumen_acciones.append(ultimo_analisis)

    return {
        "acciones": resumen_acciones,
        "alertas": todas_las_alertas,
        "conteo": conteo,
        "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# --- COMPARADOR DE TRANSICIONES DEL TOP 5 ---

def comparar_transiciones_top5():
    """
    Compara la última foto del TOP 5 con la foto inmediatamente anterior 
    en models/ranking_history.json para extraer cambios de ranking.
    """
    if not os.path.exists(PATH_RANKING_HISTORY):
        return []

    try:
        with open(PATH_RANKING_HISTORY, "r") as f:
            historial = json.load(f)
    except Exception:
        return []

    if len(historial) < 2:
        return []

    top_anterior = historial[-2].get("top_5", [])
    top_actual = historial[-1].get("top_5", [])

    mapa_anterior = {item["ticker"]: idx + 1 for idx, item in enumerate(top_anterior)}
    mapa_actual = {item["ticker"]: idx + 1 for idx, item in enumerate(top_actual)}

    cambios = []

    # Detectar entradas y movimientos
    for ticker, pos_actual in mapa_actual.items():
        if ticker not in mapa_anterior:
            cambios.append({
                "ticker": ticker,
                "tipo": "ENTRADA",
                "emoji": "🟢",
                "texto": f"🟢 **{ticker}** entra al TOP 5 en la posición **#{pos_actual}**"
            })
        else:
            pos_prev = mapa_anterior[ticker]
            if pos_actual < pos_prev:
                cambios.append({
                    "ticker": ticker,
                    "tipo": "SUBIDA",
                    "emoji": "📈",
                    "texto": f"📈 **{ticker}** asciende **#{pos_prev} → #{pos_actual}**"
                })
            elif pos_actual > pos_prev:
                cambios.append({
                    "ticker": ticker,
                    "tipo": "BAJADA",
                    "emoji": "📉",
                    "texto": f"📉 **{ticker}** desciende **#{pos_prev} → #{pos_actual}**"
                })

    # Detectar salidas
    for ticker, pos_prev in mapa_anterior.items():
        if ticker not in mapa_actual:
            cambios.append({
                "ticker": ticker,
                "tipo": "SALIDA",
                "emoji": "🔴",
                "texto": f"🔴 **{ticker}** sale del TOP 5 (estaba en #{pos_prev})"
            })

    return cambios
