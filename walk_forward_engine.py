import numpy as np
import pandas as pd
import streamlit as st
from backtesting_engine import obtener_historico_cache, calcular_indicadores_historicos

def calcular_mdd(series_returns):
    """Calcula el Maximum Drawdown (%) a partir de una serie de retornos porcentuales."""
    if len(series_returns) == 0:
        return 0.0
    cum_returns = (1 + series_returns / 100.0).cumprod()
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    return float(dd.min() * 100.0)

def calcular_sharpe(series_returns, rf=0.0):
    """Calcula el Sharpe Ratio anualizado a partir de retornos por operacion."""
    if len(series_returns) < 2:
        return 0.0
    std = series_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    mean_ret = series_returns.mean()
    return float((mean_ret - rf) / std * np.sqrt(252 / 20))

@st.cache_data(ttl=86400, show_spinner=False)
def ejecutar_walk_forward_engine(ticker, ventana_train_años=3, ventana_val_meses=12, horizonte_dias=20, es_metal=False):
    """
    Ejecuta el analisis Walk-Forward con ventanas deslizantes estricta out-of-sample sin Look-Ahead Bias.
    """
    df, err = obtener_historico_cache(ticker, periodo="10y" if not es_metal else "5y")
    if df is None or len(df) < 500:
        return None, f"N/D - Datos insuficientes para Walk-Forward (se requieren al menos 500 registros historicos). {err or ''}"

    df = df.sort_index()
    barras_train = int(ventana_train_años * 252)
    barras_val = int((ventana_val_meses / 12) * 252)
    paso = max(1, horizonte_dias)
    total_barras = len(df)
    
    min_requerido = barras_train + barras_val + horizonte_dias
    if total_barras < min_requerido:
        return None, f"N/D - Se requieren al menos {min_requerido} barras para esta configuracion (disponibles: {total_barras})."

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
        return None, "N/D - No se pudieron generar ventanas Walk-Forward suficientes con la configuracion seleccionada."

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
            
            tec = calcular_indicadores_historicos(df_slice)
            
            tec_score = (100 - tec["rsi"]) if tec else 50.0
            val_score = 60.0 if tec.get("rsi", 50) < 45 else 40.0
            fund_score = 65.0 if tec.get("ma20", 0) > tec.get("ma50", 0) else 45.0
            crec_score = 60.0 if tec.get("momentum", 0) > 0 else 40.0
            sent_score = 55.0 if tec.get("precio", 0) > tec.get("ma200", 0) else 45.0
            risk_score = 40.0 if tec.get("rsi", 50) > 65 else 70.0
            
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
            
            # Niveles de Confianza y Score
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
        
        # Desglose por Confianza
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
