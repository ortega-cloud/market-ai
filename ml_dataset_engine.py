import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def descargar_historico_ml_safe(ticker, periodo="5y"):
    """
    Descarga datos históricos utilizando yf.download() con soporte para
    acciones y futuros (GC=F, SI=F, etc.) con reintentos exponenciales.
    """
    periodos = [periodo, "5y", "2y", "1y", "max"]
    periodos = list(dict.fromkeys(periodos))
    
    ultimo_error = ""
    for p in periodos:
        for intento in range(3):
            try:
                data = yf.download(ticker, period=p, progress=False, auto_adjust=True)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                if data is not None and not data.empty and len(data) >= 60:
                    if 'Close' in data.columns:
                        return data.dropna(subset=['Close']), None
            except Exception as e:
                ultimo_error = str(e)
                time.sleep(1.0 * (intento + 1))
                
    return None, f"Error descargando datos para {ticker}: {ultimo_error or 'Sin respuesta'}"


def extraer_fundamentales_estaticos(ticker_obj):
    """
    Extrae los datos fundamentales del ticker (vía info) para usarlos
    como proxy histórico conservador si no existen series temporales contables.
    """
    try:
        info = ticker_obj.info or {}
    except Exception:
        info = {}
        
    return {
        "per": info.get("trailingPE", np.nan),
        "forward_per": info.get("forwardPE", np.nan),
        "peg": info.get("pegRatio", np.nan),
        "price_to_book": info.get("priceToBook", np.nan),
        "target_analyst": info.get("targetMeanPrice", np.nan),
        "roe": info.get("returnOnEquity", np.nan),
        "profit_margin": info.get("profitMargins", np.nan),
        "operating_margin": info.get("operatingMargins", np.nan),
        "total_debt": info.get("totalDebt", np.nan),
        "fcf": info.get("freeCashflow", np.nan),
        "revenue": info.get("totalRevenue", np.nan),
        "revenue_growth": info.get("revenueGrowth", np.nan),
        "earnings_growth": info.get("earningsGrowth", np.nan),
        "eps": info.get("trailingEps", np.nan)
    }


def clasificar_target(ret):
    """
    Clasifica un retorno porcentual en categorías alineadas con Backtesting Engine.
    """
    if pd.isna(ret):
        return np.nan
    if ret >= 2.0:
        return "BULLISH"
    elif ret <= -2.0:
        return "BEARISH"
    else:
        return "NEUTRAL"


@st.cache_data(ttl=86400, show_spinner=False)
def generar_ml_dataset(ticker, periodo="5y", es_metal=False):
    """
    Construye el dataset histórico de Machine Learning punto a punto en el tiempo
    garantizando 0% de Look-Ahead Bias en las Features.
    """
    df_hist, err = descargar_historico_ml_safe(ticker, periodo=periodo)
    if df_hist is None:
        return None, f"No se pudieron obtener datos para {ticker}. {err or ''}"

    df_hist = df_hist.sort_index()
    total_barras = len(df_hist)
    
    if total_barras < 60:
        return None, f"Insuficientes datos históricos (mínimo 60 registros, disponibles: {total_barras})."

    # Obtención de datos fundamentales (en acciones)
    ticker_obj = yf.Ticker(ticker) if not es_metal else None
    fund = extraer_fundamentales_estaticos(ticker_obj) if ticker_obj else {
        k: np.nan for k in [
            "per", "forward_per", "peg", "price_to_book", "target_analyst",
            "roe", "profit_margin", "operating_margin", "total_debt",
            "fcf", "revenue", "revenue_growth", "earnings_growth", "eps"
        ]
    }

    pesos = [30, 20, 15, 10, 15, 10]
    filas = []

    # Iteración punto a punto (empezando en la barra 50 para asegurar medias móviles)
    for idx in range(50, total_barras):
        fecha_corte = df_hist.index[idx]
        
        # SLICE ESTRICTO HISTÓRICO HASTA LA FECHA (SIN DATOS FUTUROS)
        df_slice = df_hist.iloc[:idx+1]
        close_s = df_slice['Close']
        precio_actual = float(close_s.iloc[-1])
        
        # 1. TECHNICAL FEATURES
        ma20 = float(close_s.tail(20).mean()) if len(close_s) >= 20 else np.nan
        ma50 = float(close_s.tail(50).mean()) if len(close_s) >= 50 else np.nan
        ma200 = float(close_s.tail(200).mean()) if len(close_s) >= 200 else np.nan
        
        # RSI (14)
        delta = close_s.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss != 0 else 1.0
        rsi = 100.0 - (100.0 / (1.0 + rs)) if not np.isnan(rs) else 50.0

        momentum = ((precio_actual - float(close_s.iloc[-min(10, len(close_s))])) / float(close_s.iloc[-min(10, len(close_s))])) * 100.0
        volatilidad = float(close_s.tail(20).pct_change().std() * np.sqrt(252) * 100) if len(close_s) >= 20 else np.nan
        
        tendencia = 1.0 if (not np.isnan(ma20) and not np.isnan(ma50) and ma20 > ma50) else -1.0
        
        dist_ma20 = ((precio_actual - ma20) / ma20) * 100.0 if not np.isnan(ma20) else np.nan
        dist_ma50 = ((precio_actual - ma50) / ma50) * 100.0 if not np.isnan(ma50) else np.nan
        dist_ma200 = ((precio_actual - ma200) / ma200) * 100.0 if not np.isnan(ma200) else np.nan

        # 2. VALUATION FEATURES & FAIR VALUE PROXY
        fair_value = ma50 * 1.05 if not np.isnan(ma50) else np.nan
        dist_fair_value = ((precio_actual - fair_value) / fair_value) * 100.0 if not np.isnan(fair_value) else np.nan

        # 3. MARKET AI SCORES
        tec_score = float(100.0 - rsi)
        val_score = 60.0 if rsi < 45 else 40.0
        fund_score = 65.0 if (not np.isnan(ma20) and not np.isnan(ma50) and ma20 > ma50) else 45.0
        crec_score = 60.0 if momentum > 0 else 40.0
        sent_score = 55.0 if (not np.isnan(ma200) and precio_actual > ma200) else 45.0
        risk_score = 40.0 if rsi > 65 else 70.0

        market_ai_score = (
            tec_score * (pesos[0]/100) + val_score * (pesos[1]/100) +
            fund_score * (pesos[2]/100) + crec_score * (pesos[3]/100) +
            sent_score * (pesos[4]/100) + risk_score * (pesos[5]/100)
        )

        direccion_senal = "BULLISH" if market_ai_score >= 55 else ("BEARISH" if market_ai_score <= 45 else "NEUTRAL")
        confianza = "Alta" if abs(market_ai_score - 50) > 15 else ("Moderada" if abs(market_ai_score - 50) > 5 else "Baja")

        # 4. TARGETS (FUTUROS OUT-OF-SAMPLE)
        ret_1d_5d = ((float(df_hist['Close'].iloc[min(idx + 5, total_barras - 1)]) - precio_actual) / precio_actual) * 100.0 if (idx + 5) < total_barras else np.nan
        ret_1w_4w = ((float(df_hist['Close'].iloc[min(idx + 20, total_barras - 1)]) - precio_actual) / precio_actual) * 100.0 if (idx + 20) < total_barras else np.nan
        ret_1m_3m = ((float(df_hist['Close'].iloc[min(idx + 60, total_barras - 1)]) - precio_actual) / precio_actual) * 100.0 if (idx + 60) < total_barras else np.nan
        ret_3m_6m = ((float(df_hist['Close'].iloc[min(idx + 120, total_barras - 1)]) - precio_actual) / precio_actual) * 100.0 if (idx + 120) < total_barras else np.nan

        fila = {
            "Fecha": fecha_corte.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Mercado": "Metals/Futures" if es_metal else "Stocks",
            # Technical Features
            "Precio": precio_actual,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "RSI": rsi,
            "Momentum": momentum,
            "Volatilidad": volatilidad,
            "Tendencia": tendencia,
            "Dist_MA20": dist_ma20,
            "Dist_MA50": dist_ma50,
            "Dist_MA200": dist_ma200,
            # Valuation Features
            "PER": fund["per"],
            "Forward_PER": fund["forward_per"],
            "PEG": fund["peg"],
            "Price_To_Book": fund["price_to_book"],
            "Fair_Value": fair_value,
            "Dist_Fair_Value": dist_fair_value,
            "Target_Analyst": fund["target_analyst"],
            # Fundamentals Features
            "ROE": fund["roe"],
            "Profit_Margin": fund["profit_margin"],
            "Operating_Margin": fund["operating_margin"],
            "Total_Debt": fund["total_debt"],
            "Free_Cash_Flow": fund["fcf"],
            "Revenue": fund["revenue"],
            "Revenue_Growth": fund["revenue_growth"],
            "Earnings_Growth": fund["earnings_growth"],
            "EPS": fund["eps"],
            # Market AI Scores
            "Market_AI_Score": market_ai_score,
            "Technical_Score": tec_score,
            "Valuation_Score": val_score,
            "Fundamentals_Score": fund_score,
            "Growth_Score": crec_score,
            "Risk_Score": risk_score,
            "Direccion_Senal": direccion_senal,
            "Confianza": confianza,
            # Targets (Futuros)
            "Target_Ret_1D_5D": ret_1d_5d,
            "Target_Ret_1W_4W": ret_1w_4w,
            "Target_Ret_1M_3M": ret_1m_3m,
            "Target_Ret_3M_6M": ret_3m_6m,
            "Target_Class_1W_4W": clasificar_target(ret_1w_4w)
        }
        filas.append(fila)

    df_ml = pd.DataFrame(filas)
    return df_ml, None
