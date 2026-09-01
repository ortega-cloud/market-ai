import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

def obtener_historico_wf_safe(ticker, periodo_preferido="5y"):
    """
    Descarga datos históricos utilizando yf.download() con reintentos exponenciales
    y soporte completo para Futuros (GC=F, SI=F, etc.) y Acciones.
    """
    periodos_fallback = [periodo_preferido, "5y", "2y", "1y"]
    periodos_fallback = list(dict.fromkeys(periodos_fallback))
    
    ultimo_error = ""
    
    for p in periodos_fallback:
        for intento in range(3):
            try:
                # yf.download es más estable para tickers de futuros (GC=F, CL=F, etc.)
                data = yf.download(ticker, period=p, progress=False, auto_adjust=True)
                
                # Manejar multi-index en columnas si yf.download devuelve nivele de columnas
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if data is not None and not data.empty and len(data) >= 60:
                    # Verificar que la columna Close existe
                    if 'Close' in data.columns:
                        return data.dropna(subset=['Close']), None
            except Exception as e:
                ultimo_error = str(e)
                time.sleep(1.5 * (intento + 1))
                
    return None, f"Error descargando histórico para {ticker}: {ultimo_error or 'Respuesta vacía o límite de peticiones alcanzado.'}"


def calcular_mdd(series_returns):
    """Calcula el Maximum Drawdown (%) a partir de una serie de retornos porcentuales."""
    if len(series_returns) == 0:
        return 0.0
    cum_returns = (1 + series_returns / 100.0).cumprod()
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    return float(dd.min() * 100.0)


def calcular_sharpe(series_returns, rf=0.0):
    """Calcula el Sharpe Ratio anualizado a partir de retornos por operación."""
    if len(series_returns) < 2:
        return 0.0
    std = series_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    mean_ret = series_returns.mean()
    return float((mean_ret - rf) / std * np.sqrt(252 / 20))


@st.cache_data(ttl=86400, show_spinner=False)
def ejecutar_walk_forward_engine(ticker, ventana_train_años=2, ventana_val_meses=6, horizonte_dias=20, es_metal=False):
    """
    Ejecuta el análisis Walk-Forward con ventanas deslizantes Out-of-Sample sin Look-Ahead Bias,
    soporta futuros y acciones con gestión de rate-limiting.
    """
    df, err = obtener_historico_wf_safe(ticker, periodo_preferido="5y" if not es_metal else "2y")
    
    if df is None:
        return None, f"N/D - Datos insuficientes para Walk-Forward. {err or ''}"

    df = df.sort_index()
    total_barras = len(df)
    
    # Configuración de ventanas adaptativas
    barras_train = int(ventana_train_años * 252)
    barras_val = int((ventana_val_meses / 12) * 252)
    paso = max(1, horizonte_dias)
    
    # Adaptar requerimientos de ventanas si el dataset es más corto
    if total_barras < (barras_train + barras_val + horizonte_dias):
        barras_train = max(60, int(total_barras * 0.50))
        barras_val = max(20, int(total_barras * 0.25))

    min_requerido = barras_train + barras_val + horizonte_dias
    if total_barras < min_requerido or total_barras < 60:
        return None, f"N/D - Se requieren al menos {min_requerido} registros históricos para esta configuración (registros disponibles: {total_barras})."

    # Generar ventanas temporales deslizantes TRAIN -> VALIDATION
    ventanas = []
    inicio_train = 0
    
    while (inicio_train + barras_train + barras_val + horizonte_dias) <= total_barras:
        fin_train = inicio_train + barras_train
        fin_val = fin_train + barras_val
        
        ventanas.append({
            "train_idx": (inicio_train, fin_train),
            "val_idx": (fin_train, fin_val),
            "label": f"{df.index[fin_train].strftime('%Y-%m')} a {df.index[min(fin_val-1, total_barras-1)].strftime('%Y-%m')}"
        })
        inicio_train += barras_val

    if not ventanas:
        return None, "N/D - No se pudieron generar ventanas Walk-Forward suficientes con la cantidad de datos obtenida."

    pesos_actual = [30, 20, 15, 10, 15, 10]
    pesos_opt = [45, 15, 10, 10, 10, 10]

    eval_ventanas = []
    predicciones_val_actual = []
    predicciones_val_opt = []
    
    for v in ventanas:
        val_start, val_end = v["val_idx"]
        indices_val = list(range(val_start, min(val_end, total_barras - horizonte_dias), paso))
        
        preds_v_act = []
        preds_v_opt = []
        
        for idx in indices_val:
            df_slice = df.iloc[:idx+1]
            p_ini = float(df_slice['Close'].iloc[-1])
            p_fin = float(df['Close'].iloc[idx + horizonte_dias])
            rent = ((p_fin - p_ini) / p_ini) * 100.0
            
            # Cálculo de indicadores técnicos sobre el slice temporal
            close_s = df_slice['Close']
            precio_act = float(close_s.iloc[-1])
            ma20 = float(close_s.tail(20).mean()) if len(close_s) >= 20 else precio_act
            ma50 = float(close_s.tail(50).mean()) if len(close_s) >= 50 else precio_act
            ma200 = float(close_s.tail(200).mean()) if len(close_s) >= 200 else precio_act
            
            # RSI
            delta = close_s.diff()
            gain = (delta.where(delta > 0, 0)).tail(14).mean()
            loss = (-delta.where(delta < 0, 0)).tail(14).mean()
            rs = gain / loss if loss != 0 else 1.0
            rsi = 100.0 - (100.0 / (1.0 + rs)) if not np.isnan(rs) else 50.0

            momentum = ((precio_act - float(close_s.iloc[-min(10, len(close_s))])) / float(close_s.iloc[-min(10, len(close_s))])) * 100.0

            tec_score = (100.0 - rsi)
            val_score = 60.0 if rsi < 45 else 40.0
            fund_score = 65.0 if ma20 > ma50 else 45.0
            crec_score = 60.0 if momentum > 0 else 40.0
            sent_score = 55.0 if precio_act > ma200 else 45.0
            risk_score = 40.0 if rsi > 65 else 70.0
            
            def calc_score(w):
                return (
                    tec_score * (w[0]/100) + val_score * (w[1]/100) +
                    fund_score * (w[2]/100) + crec_score * (w[3]/100) +
                    sent_score * (w[4]/100) + risk_score * (w[5]/100)
                )

            sc_act = calc_score(pesos_actual)
            sc_opt = calc_score(pesos_opt)
            
            acierto_act = (sc_act >= 50 and rent > 0) or (sc_act < 50 and rent <= 0)
            acierto_opt = (sc_opt >= 50 and rent > 0) or (sc_opt < 50 and rent <= 0)
            
            confianza = "Alta" if abs(sc_act - 50) > 15 else ("Moderada" if abs(sc_act - 50) > 5 else "Baja")
            
            p_act = {"fecha": df_slice.index[-1].strftime("%Y-%m-%d"), "rent": rent, "acierto": acierto_act, "score": sc_act, "confianza": confianza}
            p_opt = {"fecha": df_slice.index[-1].strftime("%Y-%m-%d"), "rent": rent, "acierto": acierto_opt, "score": sc_opt, "confianza": confianza}
            
            preds_v_act.append(p_act)
            preds_v_opt.append(p_opt)
            predicciones_val_actual.append(p_act)
            predicciones_val_opt.append(p_opt)

        if preds_v_act:
            hr_act = (sum(1 for x in preds_v_act if x["acierto"]) / len(preds_v_act)) * 100.0
            hr_opt = (sum(1 for x in preds_v_opt if x["acierto"]) / len(preds_v_opt)) * 100.0
            eval_ventanas.append({
                "Periodo Out-of-Sample": v["label"],
                "Hit Rate Actual (%)": round(hr_act, 1),
                "Hit Rate Optimizado (%)": round(hr_opt, 1),
                "Mejora (%)": round(hr_opt - hr_act, 1)
            })

    def compilar_metricas(preds):
        if not preds:
            return {}
        df_p = pd.DataFrame(preds)
        rets = df_p["rent"]
        hits = df_p["acierto"]
        
        conf_summary = df_p.groupby("confianza").agg(
            Casos=("rent", "count"),
            HitRate=("acierto", lambda x: round((x.sum()/len(x))*100, 1)),
            RentMedia=("rent", lambda x: round(x.mean(), 2))
        ).to_dict(orient="index")
        
        return {
            "total_predicciones": len(df_p),
            "hit_rate": round((hits.sum() / len(hits)) * 100.0, 1),
            "avg_return": round(float(rets.mean()), 2),
            "median_return": round(float(rets.median()), 2),
            "best_return": round(float(rets.max()), 2),
            "worst_return": round(float(rets.min()), 2),
            "sharpe": round(calcular_sharpe(rets), 2),
            "mdd": round(calcular_mdd(rets), 2),
            "ret_acum": round(float(rets.sum()), 2),
            "confianza_summary": conf_summary
        }

    res_act = compilar_metricas(predicciones_val_actual)
    res_opt = compilar_metricas(predicciones_val_opt)

    val_start_global = ventanas[0]["val_idx"][0]
    p_ini_bh = float(df['Close'].iloc[val_start_global])
    p_fin_bh = float(df['Close'].iloc[min(ventanas[-1]["val_idx"][1], total_barras-1)])
    buy_hold_total = round(((p_fin_bh - p_ini_bh) / p_ini_bh) * 100.0, 2)

    return {
        "resumen_actual": res_act,
        "resumen_optimizado": res_opt,
        "buy_hold_total": buy_hold_total,
        "tabla_ventanas": pd.DataFrame(eval_ventanas),
        "df_preds_act": pd.DataFrame(predicciones_val_actual),
        "df_preds_opt": pd.DataFrame(predicciones_val_opt),
        "num_ventanas": len(ventanas)
    }, None
