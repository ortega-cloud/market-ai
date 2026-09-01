import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import os
import re
import time
# Importar la función del prediction engine desde tu archivo externo
try:
    from prediction_engine import ejecutar_prediction_engine
except ImportError:
    # Función de respaldo si el archivo prediction_engine.py no está presente
    def ejecutar_prediction_engine(*args, **kwargs):
        return {
            "direccion": "🟡 NEUTRAL",
            "confianza": 50,
            "datos_utilizados_pct": 100,
            "score_predictivo": 0.0,
            "probabilidades": {"alcista": 33, "base": 34, "bajista": 33},
            "objetivos": None,
            "horizontes": {"Corto Plazo": "Neutral", "Medio Plazo": "Neutral"},
            "escenario_dominante": "Base",
            "razon_dominante": "Sin datos predictivos suficientes.",
            "senales_pos": [],
            "senales_neg": []
        }
from backtesting_engine import ejecutar_backtest_engine

# ==========================================
# SECCIÓN: MARKET AI BACKTESTING
# ==========================================
st.divider()
st.header("📊 BACKTESTING Y MEDICIÓN DE PRECISIÓN MARKET AI")
st.caption("Comprueba cómo habrían funcionado históricamente las señales de MARKET AI.")

# 1. Controles y Selección
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    tipo_activo_bt = st.radio("Tipo de Activo", ["📈 Acciones", "🥇 Metales/Futuros"], horizontal=True)

with col_b2:
    if tipo_activo_bt == "📈 Acciones":
        ticker_bt = st.text_input("Ticker", value="AAPL").upper()
        es_metal_bt = False
    else:
        ticker_bt = st.selectbox("Metal / Futuro", ["GC=F", "SI=F", "HG=F", "CL=F"])
        es_metal_bt = True

with col_b3:
    periodo_meses = st.selectbox("Periodo Histórico Evaluado", [3, 6, 12, 24], index=2, format_func=lambda x: f"{x} Meses")

with col_b4:
    horizonte_dias = st.selectbox("Horizonte de Predicción", [1, 5, 20, 60, 120], index=2, format_func=lambda x: f"{x} Días")
        
if st.button("🔎 EJECUTAR BACKTEST", use_container_width=True, key="btn_backtest_unique"):
    with st.spinner("Evaluando rendimiento histórico sin Look-Ahead Bias..."):
        res_bt = ejecutar_backtest_engine(
            ticker=ticker_bt,
            periodo_meses=periodo_meses,
            horizonte_dias=horizonte_dias,
            es_metal=es_metal_bt
        )
        
        # Unpack seguro según lo que devuelva la función
        df_bt = None
        ret_buy_hold = 0.0
        err_msg = "No se pudieron obtener datos."
        
        if isinstance(res_bt, tuple):
            if isinstance(res_bt[0], tuple):
                # Si vino empaquetado como ((df, ret), full_dict)
                df_bt, ret_buy_hold = res_bt[0]
            elif isinstance(res_bt[0], pd.DataFrame):
                # Si vino empaquetado como (df, ret)
                df_bt, ret_buy_hold = res_bt[0], res_bt[1]
            else:
                df_bt, err_msg = res_bt[0], str(res_bt[1])
        elif isinstance(res_bt, dict):
            # Si devolvió un diccionario directo
            df_bt = res_bt.get("resultados")
            ret_buy_hold = res_bt.get("buy_hold", 0.0)

        # Validación sin romper por AttributeError
        if df_bt is None or not isinstance(df_bt, pd.DataFrame) or df_bt.empty:
            st.warning(f"⚠️ {err_msg if isinstance(err_msg, str) else 'N/D - Datos insuficientes'}")
        else:
            # 1. Métricas Principales
            tot_pred = len(df_bt)
            aciertos = len(df_bt[df_bt["resultado"] == "✅ Acierto"])
            tasa_acierto = (aciertos / tot_pred) * 100.0 if tot_pred > 0 else 0.0
            ret_media = float(df_bt["rentabilidad"].mean())
            mejor_res = float(df_bt["rentabilidad"].max())
            peor_res = float(df_bt["rentabilidad"].min())
            ret_acum = float(df_bt["rentabilidad"].sum())

            st.subheader("🎯 Resumen de Rendimiento")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("📊 Predicciones", tot_pred)
            m2.metric("🎯 Tasa de Acierto", f"{tasa_acierto:.1f}%")
            m3.metric("📈 Rentabilidad Media", f"{ret_media:+.2f}%")
            m4.metric("🚀 Mejor Operación", f"{mejor_res:+.2f}%")
            m5.metric("📉 Peor Operación", f"{peor_res:+.2f}%")
            
            # 2. Comparativa MARKET AI vs BUY & HOLD
            st.subheader("⚔️ MARKET AI vs BUY & HOLD")
            c_comp1, c_comp2 = st.columns(2)
            c_comp1.metric("Estrategia MARKET AI (Acumulada)", f"{ret_acum:+.2f}%")
            c_comp2.metric(f"📊 Buy & Hold ({ticker_bt})", f"{ret_buy_hold:+.2f}%")
            
            # 3. Evolución Acumulada
            st.subheader("📈 Evolución Acumulada MARKET AI")
            df_bt["ret_acum_mai"] = df_bt["rentabilidad"].cumsum()
            st.line_chart(df_bt.set_index("fecha")[["ret_acum_mai"]])

            # 4. Registros Detallados
            st.subheader("📋 Resultados históricos")
            st.dataframe(
                df_bt[["fecha", "ticker", "score", "direccion", "confianza", "precio_inicial", "precio_final", "rentabilidad", "resultado"]],
                use_container_width=True
            )

# Análisis de sentimiento con TextBlob
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

# Importar el módulo DCF existente (INTACTO)
try:
    import dcf
except ImportError:
    dcf = None

# Módulo de funciones externas
try:
    from master_ranking import ejecutar_escaneo_master_ranking, generar_explicacion
except ImportError:
    def generar_explicacion(item):
        return f"El activo presenta un Score de {item.get('score', 0)}/100 y una proyección {item.get('direccion', 'neutral')}."

st.set_page_config(page_title="MARKET AI - Análisis y Escáner de Acciones", layout="wide", page_icon="📈")

# =========================================================
# CONFIGURACIÓN Y API KEYS
# =========================================================
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# =========================================================
# UNIVERSO DE EMPRESAS DE RESPALDO Y OBTENCIÓN DE S&P 500
# =========================================================

RESPALDO_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "LLY", "AVGO",
    "JPM", "WMT", "UNH", "V", "PG", "MA", "ORCL", "HD", "HD", "JNJ", "COST", "ABBV",
    "BAC", "KO", "NFLX", "CRM", "CVX", "MRK", "AMD", "PEP", "TMO", "LIN", "WFC",
    "ADBE", "MCD", "DIS", "PM", "CSCO", "ABT", "GE", "INTU", "CAT", "TXN", "AMAT",
    "VZ", "AXP", "IBM", "QCOM", "PFE", "COP"
]

@st.cache_data(ttl=86400)
def obtener_universo_sp500():
    """
    Obtiene la lista de tickers del S&P 500 desde Wikipedia.
    Si falla la petición o el scraping, retorna la lista de respaldo.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df_sp = tables[0]
        if "Symbol" in df_sp.columns:
            tickers = df_sp["Symbol"].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
            if len(tickers) > 100:
                return tickers
    except Exception:
        pass
    return RESPALDO_SP500

# =========================================================
# MOTORES ESPECIALIZADOS
# =========================================================

def escaneo_metales_motor():
    """Motor de análisis rápido para metales y materias primas."""
    metales = {
        "GC=F": "Oro",
        "SI=F": "Plata",
        "HG=F": "Cobre",
        "PL=F": "Platino",
        "PA=F": "Paladio"
    }
    
    resultados = []
    for ticker_symbol, nombre in metales.items():
        try:
            t = yf.Ticker(ticker_symbol)
            hist = t.history(period="1y")
            if hist.empty:
                continue
            
            precio = float(hist['Close'].iloc[-1])
            var_pct = float(((precio - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100)
            
            # Cálculo de RSI sencillo
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            
            score = 50.0
            if rsi < 35: score += 25
            elif rsi > 70: score -= 15
            if var_pct > 0: score += 15
            
            direccion = "🟢 Alcista" if score >= 60 else ("🔴 Bajista" if score <= 40 else "🟡 Neutral")
            
            resultados.append({
                'activo': f"{nombre} ({ticker_symbol})",
                'tipo': '🥇 Metal/Futuro',
                'score': round(score, 1),
                'precio': round(precio, 2),
                'potencial': round(var_pct, 2),
                'confianza': 75,
                'riesgo': 'Bajo' if rsi < 60 else 'Alto',
                'direccion': direccion,
                'horizonte': '1-4 semanas',
                'oportunidad_global': score * 0.7 + 75 * 0.3
            })
        except Exception:
            continue
            
    return resultados

# =========================================================
# FUNCIONES AUXILIARES Y OBTENCIÓN DE DATOS
# =========================================================

@st.cache_data(ttl=3600)
def obtener_info_accion(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except Exception:
        return None

@st.cache_data(ttl=1800)
def obtener_historico(ticker_symbol, periodo="1y"):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=periodo)
        if df.empty:
            return None
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def obtener_objetivos_analistas(ticker_symbol):
    objetivos = {}
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if "targetMeanPrice" in info and info["targetMeanPrice"] is not None:
            objetivos["mean"] = info.get("targetMeanPrice")
            objetivos["high"] = info.get("targetHighPrice")
            objetivos["low"] = info.get("targetLowPrice")
            objetivos["median"] = info.get("targetMedianPrice")
            objetivos["consensus"] = info.get("recommendationKey", "N/D").upper()
            objetivos["num_analistas"] = info.get("numberOfAnalystOpinions")
            return objetivos
    except Exception:
        pass

    if FINNHUB_API_KEY:
        try:
            url = f"https://finnhub.io/api/v1/stock/price-target?symbol={ticker_symbol}&token={FINNHUB_API_KEY}"
            res = requests.get(url).json()
            if res and "targetMean" in res:
                objetivos["mean"] = res.get("targetMean")
                objetivos["high"] = res.get("targetHigh")
                objetivos["low"] = res.get("targetLow")
                objetivos["median"] = res.get("targetMedian")
                return objetivos
        except Exception:
            pass

    return objetivos

@st.cache_data(ttl=3600)
def obtener_recomendaciones_analistas(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        rec = ticker.recommendations
        if rec is not None and not rec.empty:
            return rec
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def obtener_estimaciones_eps(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        eps_est = ticker.earnings_estimate
        if eps_est is not None and not eps_est.empty:
            return eps_est
    except Exception:
        pass
    return None

# =========================================================
# MOTOR DE NOTICIAS Y SENTIMIENTO MARKET AI
# =========================================================

@st.cache_data(ttl=1800)
def analizar_noticias(ticker_symbol):
    noticias_procesadas = []
    catalizadores_set = set()
    riesgos_set = set()

    KW_CATALIZADORES = {
        'earnings': 'Publicación o superación de resultados empresariales',
        'profit': 'Mejora en las previsiones de beneficios',
        'revenue': 'Incremento destacado en ventas/ingresos',
        'launch': 'Lanzamiento de nuevos productos/servicios',
        'contract': 'Firma de contratos relevantes',
        'acquisition': 'Estrategia de adquisición / M&A',
        'buyback': 'Anuncio o programa de recompra de acciones',
        'dividend': 'Incremento o pago de dividendos',
        'partnership': 'Alianzas estratégicas o expansiones',
        'approval': 'Aprobación regulatoria o patente concedida',
        'upgrade': 'Mejora de recomendación por firmas de inversión'
    }

    KW_RIESGOS = {
        'downgrade': 'Rebaja de recomendación o precio objetivo',
        'loss': 'Pérdidas operativas o caída en márgenes',
        'lawsuit': 'Litigios, demandas o investigaciones legales',
        'investigation': 'Escrutinio o investigación regulatoria',
        'debt': 'Preocupaciones vinculadas a la deuda',
        'layoff': 'Anuncio de reestructuraciones o despidos',
        'decline': 'Contracción de la demanda o cuota de mercado',
        'delay': 'Retrasos en producción o cadena de suministro',
        'cut': 'Recorte de previsiones financieras (Guidance)',
        'sanction': 'Tensiones geopolíticas o sanciones'
    }

    try:
        ticker = yf.Ticker(ticker_symbol)
        raw_news = ticker.news
        if not raw_news:
            return {
                "noticias": [],
                "sentimiento_general": "🟡 NEUTRAL",
                "puntuacion_sentimiento": 0,
                "noticias_positivas": 0,
                "noticias_negativas": 0,
                "catalizadores": [],
                "riesgos": [],
                "resumen": "No hay información reciente suficiente."
            }

        scores_polaridad = []
        pos_count = 0
        neg_count = 0

        for item in raw_news[:10]:
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}

            titulo = item.get("title") or content.get("title") or "Sin título"
            fuente = item.get("publisher") or content.get("provider", {}).get("displayName") or "Fuente Financiera"
            url = item.get("link") or content.get("canonicalUrl", {}).get("url") or "#"

            pub_date = item.get("providerPublishTime") or content.get("pubDate")
            if isinstance(pub_date, (int, float)):
                fecha_str = datetime.fromtimestamp(pub_date).strftime("%d/%m/%Y %H:%M")
            else:
                fecha_str = "Reciente"

            resumen = item.get("summary") or content.get("summary") or titulo
            texto_completo = f"{titulo}. {resumen}"

            if TextBlob is not None:
                blob = TextBlob(texto_completo)
                polarity = blob.sentiment.polarity
            else:
                polarity = 0.0
                words = re.findall(r'\w+', texto_completo.lower())
                pos_w = sum(1 for w in words if w in ['beat', 'growth', 'up', 'high', 'gain', 'buy', 'positive'])
                neg_w = sum(1 for w in words if w in ['miss', 'fall', 'down', 'low', 'drop', 'sell', 'negative'])
                if pos_w + neg_w > 0:
                    polarity = (pos_w - neg_w) / (pos_w + neg_w)

            scores_polaridad.append(polarity)

            if polarity > 0.1:
                sentimiento_str = "🟢 Positivo"
                pos_count += 1
            elif polarity < -0.1:
                sentimiento_str = "🔴 Negativo"
                neg_count += 1
            else:
                sentimiento_str = "🟡 Neutral"

            abs_pol = abs(polarity)
            if abs_pol > 0.35 or "breaking" in titulo.lower():
                importancia_str = "🔥 Alta"
            elif abs_pol > 0.15:
                importancia_str = "🟠 Media"
            else:
                importancia_str = "⚪ Baja"

            texto_lc = texto_completo.lower()
            for kw, desc in KW_CATALIZADORES.items():
                if kw in texto_lc:
                    catalizadores_set.add(desc)
            for kw, desc in KW_RIESGOS.items():
                if kw in texto_lc:
                    riesgos_set.add(desc)

            noticias_procesadas.append({
                "titulo": titulo,
                "fuente": fuente,
                "fecha": fecha_str,
                "url": url,
                "resumen": resumen,
                "sentimiento": sentimiento_str,
                "importancia": importancia_str
            })

        if scores_polaridad:
            avg_pol = float(np.mean(scores_polaridad))
            puntuacion = int(avg_pol * 100)
            puntuacion = max(-100, min(100, puntuacion))
        else:
            puntuacion = 0

        if puntuacion >= 25:
            sent_gen = "🟢 POSITIVO"
        elif puntuacion <= -25:
            sent_gen = "🔴 NEGATIVO"
        else:
            sent_gen = "🟡 NEUTRAL"

        resumen_gen = f"Se han analizado {len(noticias_procesadas)} noticias recientes de fuentes financieras. El volumen presenta {pos_count} noticias de sesgo favorable, {neg_count} desfavorables y {len(noticias_procesadas) - pos_count - neg_count} de corte neutral."

        return {
            "noticias": noticias_procesadas,
            "sentimiento_general": sent_gen,
            "puntuacion_sentimiento": puntuacion,
            "noticias_positivas": pos_count,
            "noticias_negativas": neg_count,
            "catalizadores": list(catalizadores_set),
            "riesgos": list(riesgos_set),
            "resumen": resumen_gen
        }

    except Exception:
        return {
            "noticias": [],
            "sentimiento_general": "🟡 NEUTRAL",
            "puntuacion_sentimiento": 0,
            "noticias_positivas": 0,
            "noticias_negativas": 0,
            "catalizadores": [],
            "riesgos": [],
            "resumen": "⚠️ No se ha podido obtener información de noticias en este momento."
        }

# =========================================================
# MOTOR CENTRAL DE DIAGNÓSTICO MARKET AI (SCORE GLOBAL)
# =========================================================

def calcular_market_ai_score(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas):
    pts_tec, pts_val, pts_fund, pts_crec, pts_riesgo = 0.0, 0.0, 0.0, 0.0, 0.0
    pos, neg, riesgos, catalizadores = [], [], [], []

    precio = datos_tec.get("precio")
    ma20 = datos_tec.get("ma20")
    ma50 = datos_tec.get("ma50")
    ma200 = datos_tec.get("ma200")
    rsi = datos_tec.get("rsi")
    momentum = datos_tec.get("variacion")

    tendencia_str = "Neutral / Indeterminada"

    if precio is not None and ma20 is not None and ma50 is not None:
        if precio > ma20 and ma20 > ma50:
            pts_tec += 8.0
            tendencia_str = "Alcista en corto y medio plazo"
            pos.append("Estructura alcista definida por encima de sus medias rápidas (MA20 y MA50).")
        elif precio < ma20 and ma20 < ma50:
            tendencia_str = "Bajista en corto y medio plazo"
            neg.append("Cotización presionada por debajo de sus medias de 20 y 50 periodos.")
        else:
            pts_tec += 4.0

    if precio is not None and ma200 is not None:
        if precio > ma200:
            pts_tec += 7.0
            if "Alcista" not in tendencia_str:
                tendencia_str = "Alcista de largo plazo"
            pos.append("Mantiene soporte estructural clave sobre la media móvil de largo plazo (MA200).")
        else:
            neg.append("El precio cotiza por debajo de la media móvil de 200 periodos (tendencia principal débil).")

    if rsi is not None and not np.isnan(rsi):
        if 40 <= rsi <= 65:
            pts_tec += 6.0
            pos.append(f"RSI equilibrado ({rsi:.1f}), mostrando dinamismo sin sobrecompra.")
        elif rsi < 30:
            pts_tec += 4.0
            pos.append(f"RSI en sobreventa técnica ({rsi:.1f}), propicio para un rebote.")
        elif rsi > 70:
            pts_tec += 1.0
            neg.append(f"RSI en zona de sobrecompra ({rsi:.1f}), riesgo elevado de consolidación.")
        else:
            pts_tec += 3.0

    if momentum is not None and not np.isnan(momentum):
        if momentum > 3.0:
            pts_tec += 4.0
            pos.append(f"Impulso comprador reciente (+{momentum:.2f}% en la última sesión).")
        elif momentum < -3.0:
            neg.append(f"Inercia vendedora en la cotización ({momentum:.2f}% en la última sesión).")
        else:
            pts_tec += 2.0

    pts_tec = min(25.0, max(0.0, pts_tec))

    fair_value = datos_val.get("valor_dcf_base")
    obj_med = datos_analistas.get("obj_med")
    pe = datos_val.get("pe")
    forward_pe = datos_val.get("forward_pe")
    peg = datos_val.get("peg")
    pb = datos_val.get("price_to_book")

    potencial_dcf = ((fair_value - precio) / precio * 100.0) if (fair_value is not None and precio and precio > 0) else None
    potencial_analistas = ((obj_med - precio) / precio * 100.0) if (obj_med is not None and precio and precio > 0) else None

    if potencial_dcf is not None:
        if potencial_dcf >= 20.0:
            pts_val += 9.0
            pos.append(f"Descuento relevante frente a su valor intrínseco DCF (Potencial +{potencial_dcf:.1f}%).")
            catalizadores.append("Descuento significativo respecto al modelo de flujos de caja descontados (DCF).")
        elif potencial_dcf >= -10.0:
            pts_val += 5.0
            pos.append("Precio cotizando en un rango de valoración razonable según el modelo DCF.")
        else:
            neg.append(f"Sobrevaloración estimada del {abs(potencial_dcf):.1f}% respecto a su DCF.")
    else:
        pts_val += 4.0

    if potencial_analistas is not None:
        if potencial_analistas >= 15.0:
            pts_val += 4.0
            pos.append(f"El consenso de analistas otorga un potencial alcista medio del +{potencial_analistas:.1f}%.")
            catalizadores.append(f"Objetivo medio de analistas situado en ${obj_med:,.2f}.")
        elif potencial_analistas >= 0:
            pts_val += 2.0
        else:
            pts_val += 2.0

    mult_pe = forward_pe if forward_pe is not None else pe
    if mult_pe is not None and mult_pe > 0:
        if mult_pe < 15.0:
            pts_val += 5.0
            pos.append(f"Múltiplo de beneficios atractivo (PER {mult_pe:.1f}x).")
        elif mult_pe < 25.0:
            pts_val += 3.0
        else:
            neg.append(f"Múltiplo PER exigente ({mult_pe:.1f}x).")

    if peg is not None and peg > 0:
        if peg < 1.0:
            pts_val += 4.0
            pos.append(f"PEG de {peg:.2f} (< 1.0), indicando infravaloración ajustada por crecimiento.")
            catalizadores.append("Relación precio/crecimiento (PEG) en niveles muy competitivos.")
        elif peg < 2.0:
            pts_val += 2.0

    if pb is not None and pb > 0 and pb < 3.0:
        pts_val += 3.0

    pts_val = min(25.0, max(0.0, pts_val))

    if potencial_dcf is not None:
        if potencial_dcf >= 15.0:
            est_val = "🟢 INFRAVALORADA"
        elif potencial_dcf <= -15.0:
            est_val = "🔴 SOBREVALORADA"
        else:
            est_val = "🟡 RAZONABLEMENTE VALORADA"
    elif potencial_analistas is not None:
        if potencial_analistas >= 15.0:
            est_val = "🟢 INFRAVALORADA"
        elif potencial_analistas <= -10.0:
            est_val = "🔴 SOBREVALORADA"
        else:
            est_val = "🟡 RAZONABLEMENTE VALORADA"
    else:
        est_val = "🟡 RAZONABLEMENTE VALORADA"

    roe = datos_fund.get("roe")
    margen = datos_fund.get("margen")
    deuda = datos_fund.get("deuda")
    fcf = datos_fund.get("flujo_caja")

    fund_eval_pts = 0

    if roe is not None and not np.isnan(roe):
        if roe >= 0.15:
            pts_fund += 7.0
            fund_eval_pts += 2
            pos.append(f"Excelente retribución sobre el capital propio (ROE {roe*100:.1f}%).")
        elif roe >= 0.08:
            pts_fund += 4.0
            fund_eval_pts += 1
        else:
            neg.append(f"Rentabilidad sobre fondos propios modesta (ROE {roe*100:.1f}%).")

    if margen is not None and not np.isnan(margen):
        if margen >= 0.15:
            pts_fund += 7.0
            fund_eval_pts += 2
            pos.append(f"Sólida eficiencia operativa con margen neto del {margen*100:.1f}%.")
        elif margen >= 0.05:
            pts_fund += 4.0
            fund_eval_pts += 1

    if deuda is not None and not np.isnan(deuda):
        if deuda < 80.0:
            pts_fund += 6.0
            fund_eval_pts += 2
            pos.append("Estructura de balance con endeudamiento bajo y controlado.")
        elif deuda > 150.0:
            neg.append(f"Apalancamiento elevado (Deuda/Patrimonio {deuda:.1f}%).")
        else:
            pts_fund += 3.0
            fund_eval_pts += 1

    if fcf is not None and not np.isnan(fcf):
        if fcf > 0:
            pts_fund += 5.0
            fund_eval_pts += 1
            pos.append("Generación de caja libre (Free Cash Flow) en terreno positivo.")
        else:
            neg.append("Free Cash Flow negativo (quema operativa de caja).")

    pts_fund = min(25.0, max(0.0, pts_fund))

    if fund_eval_pts >= 5:
        calidad_fund = "Fuertes"
    elif fund_eval_pts >= 3:
        calidad_fund = "Normales"
    else:
        calidad_fund = "Débiles"

    crec_ing = datos_crec.get("crecimiento_ingresos")
    crec_ben = datos_crec.get("crecimiento_beneficios")
    div_yield = datos_crec.get("dividend_yield")

    crec_eval_pts = 0

    if crec_ing is not None and not np.isnan(crec_ing):
        if crec_ing >= 0.10:
            pts_crec += 6.0
            crec_eval_pts += 2
            pos.append(f"Crecimiento sólido en la facturación (+{crec_ing*100:.1f}%).")
        elif crec_ing > 0:
            pts_crec += 3.0
            crec_eval_pts += 1
        else:
            neg.append("Ingresos brutos en fase de contracción.")

    if crec_ben is not None and not np.isnan(crec_ben):
        if crec_ben >= 0.10:
            pts_crec += 6.0
            crec_eval_pts += 2
            pos.append(f"Fuerte impulso de ganancias (+{crec_ben*100:.1f}% en beneficio neto).")
            catalizadores.append("Expansión acelerada de los beneficios de la empresa.")
        elif crec_ben > 0:
            pts_crec += 3.0
            crec_eval_pts += 1
        else:
            neg.append("Retroceso en la generación de beneficio neto.")

    if div_yield is not None and not np.isnan(div_yield) and div_yield > 0:
        pts_crec += 3.0
        pos.append(f"Política de retribución con rentabilidad por dividendo del {div_yield*100:.2f}%.")

    pts_crec = min(15.0, max(0.0, pts_crec))

    if crec_eval_pts >= 3:
        nivel_crec = "Crecimiento fuerte"
    elif crec_eval_pts >= 1:
        nivel_crec = "Crecimiento moderado"
    elif (crec_ing is not None and crec_ing < 0) or (crec_ben is not None and crec_ben < 0):
        nivel_crec = "Contracción"
    else:
        nivel_crec = "Estancamiento"

    vol = datos_tec.get("volatilidad")

    if vol is not None and not np.isnan(vol):
        if vol < 22.0:
            pts_riesgo += 4.0
            pos.append("Baja volatilidad histórica, otorgando mayor previsibilidad al activo.")
        elif vol < 38.0:
            pts_riesgo += 2.0
        else:
            neg.append(f"Elevada volatilidad anualizada ({vol:.1f}%).")
            riesgos.append("Alta volatilidad en la cotización diaria.")

    if deuda is not None and not np.isnan(deuda):
        if deuda < 100.0:
            pts_riesgo += 3.0
        elif deuda > 180.0:
            riesgos.append("Riesgo financiero latente por elevado nivel de endeudamiento.")

    if rsi is not None and not np.isnan(rsi):
        if 30 <= rsi <= 70:
            pts_riesgo += 3.0
        else:
            riesgos.append(f"Tensión técnica en el indicador RSI ({rsi:.1f}).")

    pts_riesgo = min(10.0, max(0.0, pts_riesgo))

    score_total = min(100.0, max(0.0, pts_tec + pts_val + pts_fund + pts_crec + pts_riesgo))

    if score_total >= 85.0:
        categoria_str = "🟢 OPORTUNIDAD MUY INTERESANTE"
    elif score_total >= 70.0:
        categoria_str = "🟢 OPORTUNIDAD INTERESANTE"
    elif score_total >= 55.0:
        categoria_str = "🟡 NEUTRAL"
    elif score_total >= 40.0:
        categoria_str = "🟠 RIESGO ELEVADO"
    else:
        categoria_str = "🔴 POCO ATRACTIVA"

    return {
        "score_total": round(score_total, 1),
        "categoria": categoria_str,
        "desglose": {
            "tecnico": round(pts_tec, 1),
            "valoracion": round(pts_val, 1),
            "fundamentales": round(pts_fund, 1),
            "crecimiento": round(pts_crec, 1),
            "riesgo": round(pts_riesgo, 1)
        },
        "diagnostico": {
            "tendencia": tendencia_str,
            "estado_valoracion": est_val,
            "calidad_fundamentales": calidad_fund,
            "nivel_crecimiento": nivel_crec,
            "positivos": pos if pos else ["Indicadores técnicos y fundamentales dentro de promedios normativos."],
            "negativos": neg if neg else ["No se detectan debilidades severas con los datos disponibles."],
            "riesgos": riesgos if riesgos else ["Sin factores de riesgo crítico específicos detectados."],
            "catalizadores": catalizadores if catalizadores else ["Publicaciones trimestrales de resultados corporativos."],
            "precio": precio,
            "fair_value": fair_value,
            "target_analistas": obj_med,
            "potencial_dcf": potencial_dcf,
            "potencial_analistas": potencial_analistas
        }
    }

# =========================================================
# MOTOR DE DIAGNÓSTICO Y PREDICCIÓN (HOLÍSTICO)
# =========================================================

def generar_diagnostico_market_ai(res_score, datos_tec, datos_val, datos_fund, datos_crec, datos_analistas, res_noticias):
    signals_pos = []
    signals_neg = []

    score = res_score["score_total"]
    precio = datos_tec.get("precio")
    rsi = datos_tec.get("rsi")
    ma20 = datos_tec.get("ma20")
    ma200 = datos_tec.get("ma200")

    fair_value = datos_val.get("valor_dcf_base")
    obj_med = datos_analistas.get("obj_med")
    pe = datos_val.get("pe")

    roe = datos_fund.get("roe")
    deuda = datos_fund.get("deuda")

    crec_ing = datos_crec.get("crecimiento_ingresos")
    sent_score = res_noticias.get("puntuacion_sentimiento", 0) if res_noticias else 0

    # DCF
    pot_dcf = None
    if fair_value and precio and precio > 0:
        pot_dcf = ((fair_value - precio) / precio) * 100
        if pot_dcf >= 15.0:
            signals_pos.append(f"Valoración DCF con margen de seguridad (+{pot_dcf:.1f}% de potencial).")
        elif pot_dcf <= -15.0:
            signals_neg.append(f"Sobrevaloración respecto al modelo DCF ({pot_dcf:.1f}% de desviación).")

    # Analistas
    pot_analistas = None
    if obj_med and precio and precio > 0:
        pot_analistas = ((obj_med - precio) / precio) * 100
        if pot_analistas >= 10.0:
            signals_pos.append(f"Consenso de analistas favorable (Objetivo medio +{pot_analistas:.1f}%).")
        elif pot_analistas <= -10.0:
            signals_neg.append(f"Precio actual por encima del objetivo medio de analistas ({pot_analistas:.1f}%).")

    # Técnico
    if precio and ma200:
        if precio > ma200:
            signals_pos.append("Precio por encima de la media móvil estructural (MA200).")
        else:
            signals_neg.append("Cotización por debajo de la media móvil de 200 días.")

    if rsi is not None and not np.isnan(rsi):
        if rsi < 35:
            signals_pos.append(f"RSI en zona de sobreventa ({rsi:.1f}), propicio para rebotes.")
        elif rsi > 70:
            signals_neg.append(f"RSI en sobrecompra técnica ({rsi:.1f}), posible consolidación.")

    # Fundamentales
    if roe and not np.isnan(roe) and roe > 0.15:
        signals_pos.append(f"Alta rentabilidad sobre capital (ROE {roe*100:.1f}%).")
    if pe and pe > 30:
        signals_neg.append(f"Múltiplo de beneficios exigente (PER {pe:.1f}x).")
    if crec_ing and not np.isnan(crec_ing) and crec_ing < 0:
        signals_neg.append(f"Contracción reciente en los ingresos ({crec_ing*100:.1f}%).")

    # Sentimiento
    if sent_score >= 25:
        signals_pos.append(f"Sentimiento de noticias favorable (+{sent_score}/100).")
    elif sent_score <= -25:
        signals_neg.append(f"Sentimiento de noticias desfavorable ({sent_score}/100).")

    net_signals = len(signals_pos) - len(signals_neg)

    if score >= 68 and net_signals >= 1:
        direccion = "🟢 PROBABLEMENTE ALCISTA"
        veredicto = "🟢 COMPRA / SESGO ALCISTA"
        sesgo = "ALCISTA"
    elif score <= 42 or net_signals <= -2:
        direccion = "🔴 PROBABLEMENTE BAJISTA"
        veredicto = "🔴 VENTA / SESGO BAJISTA"
        sesgo = "BAJISTA"
    else:
        direccion = "🟡 NEUTRAL / INCIERTA"
        veredicto = "🟡 MANTENER / ESPERAR"
        sesgo = "NEUTRAL"

    # Confianza
    data_points = 0
    max_data_points = 6
    if datos_tec.get("precio") is not None: data_points += 1
    if fair_value is not None: data_points += 1
    if obj_med is not None: data_points += 1
    if roe is not None: data_points += 1
    if crec_ing is not None: data_points += 1
    if res_noticias and len(res_noticias.get("noticias", [])) > 0: data_points += 1

    base_completeness = (data_points / max_data_points) * 60.0

    total_sig = len(signals_pos) + len(signals_neg)
    if total_sig > 0:
        alignment = abs(len(signals_pos) - len(signals_neg)) / total_sig
        coherence_score = alignment * 40.0
    else:
        coherence_score = 20.0

    confianza = int(min(95, max(25, base_completeness + coherence_score)))

    # Horizonte
    short_term_weight = 0
    medium_term_weight = 0
    long_term_weight = 0

    if rsi and (rsi > 70 or rsi < 35): short_term_weight += 2
    if abs(sent_score) >= 30: short_term_weight += 2

    if obj_med is not None: medium_term_weight += 2
    if ma20 and precio and (precio > ma20): medium_term_weight += 1

    if fair_value is not None: long_term_weight += 3
    if roe and roe > 0.12: long_term_weight += 2

    if short_term_weight >= max(medium_term_weight, long_term_weight):
        horizonte = "1-5 días"
    elif medium_term_weight >= long_term_weight:
        horizonte = "1-4 semanas"
    elif long_term_weight > 3:
        horizonte = "1-6 meses"
    else:
        horizonte = "6-24 meses"

    # Calidad de la Oportunidad
    if score >= 80 and confianza >= 75:
        calidad = "🔥 EXCELENTE"
    elif score >= 68 and confianza >= 60:
        calidad = "🟢 BUENA"
    elif score <= 40 and confianza >= 60:
        calidad = "🟢 BUENA (SEÑAL BAJISTA)"
    elif score >= 50:
        calidad = "🟡 MODERADA"
    else:
        calidad = "🔴 DÉBIL"

    partes = []
    partes.append(f"{datos_tec.get('ticker', 'El activo')} presenta un Score de {score:.1f}/100 y confianza del {confianza}%.")
    if pot_dcf is not None:
        partes.append(f"Potencial DCF: {pot_dcf:+.1f}%.")
    if signals_pos:
        partes.append(f"Destaca: {signals_pos[0]}")
    explicacion = " ".join(partes)

    return {
        "direccion": direccion,
        "sesgo": sesgo,
        "confianza": confianza,
        "horizonte": horizonte,
        "veredicto": veredicto,
        "calidad": calidad,
        "explicacion": explicacion,
        "signals_pos": signals_pos if signals_pos else ["Sin factores fuertemente alcistas."],
        "signals_neg": signals_neg if signals_neg else ["Sin factores fuertemente bajistas."],
        "pot_dcf": pot_dcf,
        "pot_analistas": pot_analistas
    }

# =========================================================
# MOTOR DEL ESCÁNER AUTOMÁTICO DE OPORTUNIDADES
# =========================================================

@st.cache_data(ttl=1800)
def ejecutar_filtro_rapido(universe_tickers):
    """
    Fase 1: Filtro ultrarrápido sobre la lista completa de tickers.
    Descarta empresas sin cotización, ilíquidas o corruptas.
    Retorna un ranking preliminar para seleccionar las mejores ~25 candidatas.
    """
    candidatas_score = []

    for ticker in universe_tickers:
        try:
            t = yf.Ticker(ticker)
            fast_info = t.fast_info

            precio = fast_info.last_price
            market_cap = fast_info.market_cap

            if precio is None or np.isnan(precio) or precio <= 0:
                continue
            if market_cap is None or market_cap < 1_000_000_000: # Excluir micro caps sin liquidez
                continue

            info = t.info
            pe = info.get("trailingPE") or info.get("forwardPE")
            revenue_growth = info.get("revenueGrowth")
            earnings_growth = info.get("earningsGrowth")

            # Puntuación preliminar rápida de interés
            quick_score = 50.0
            if revenue_growth and revenue_growth > 0.08: quick_score += 15
            if earnings_growth and earnings_growth > 0.10: quick_score += 15
            if pe and 0 < pe < 22: quick_score += 10
            elif pe and pe > 60: quick_score -= 10

            candidatas_score.append({
                "ticker": ticker,
                "quick_score": quick_score,
                "precio": precio,
                "market_cap": market_cap
            })
        except Exception:
            continue

    # Ordenar por quick_score de mayor a menor y tomar los top 25
    candidatas_score.sort(key=lambda x: x["quick_score"], reverse=True)
    candidatas = [c["ticker"] for c in candidatas_score[:25]]
    return candidatas

def ejecutar_analisis_completo_ticker(ticker_symbol):
    """
    Ejecuta el pipeline completo idéntico al análisis individual para 1 ticker.
    """
    info = obtener_info_accion(ticker_symbol)
    if not info:
        return None

    precio = info.get("currentPrice") or info.get("regularMarketPrice")
    df_hist = obtener_historico(ticker_symbol)

    if (precio is None) and df_hist is not None and not df_hist.empty:
        precio = float(df_hist['Close'].iloc[-1])

    if precio is None:
        return None

    ma20 = float(df_hist['MA20'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'MA20' in df_hist and not pd.isna(df_hist['MA20'].iloc[-1])) else None
    ma50 = float(df_hist['MA50'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'MA50' in df_hist and not pd.isna(df_hist['MA50'].iloc[-1])) else None
    ma200 = float(df_hist['MA200'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'MA200' in df_hist and not pd.isna(df_hist['MA200'].iloc[-1])) else None
    rsi = float(df_hist['RSI'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'RSI' in df_hist and not pd.isna(df_hist['RSI'].iloc[-1])) else None

    variacion = None
    volatilidad = None
    if df_hist is not None and len(df_hist) > 1:
        variacion = float(((df_hist['Close'].iloc[-1] - df_hist['Close'].iloc[-2]) / df_hist['Close'].iloc[-2]) * 100)
        returns = df_hist['Close'].pct_change().dropna()
        volatilidad = float(returns.std() * np.sqrt(252) * 100)

    pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    peg = info.get("pegRatio")
    price_to_book = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    margen = info.get("profitMargins")
    margen_operativo = info.get("operatingMargins")
    deuda = info.get("debtToEquity")
    flujo_caja = info.get("freeCashflow")
    ingresos = info.get("totalRevenue")
    beneficio = info.get("netIncomeToCommon")
    eps = info.get("trailingEps")
    crecimiento_ingresos = info.get("revenueGrowth")
    crecimiento_beneficios = info.get("earningsGrowth")

    objetivos = obtener_objetivos_analistas(ticker_symbol)
    obj_med = objetivos.get("mean")

    valor_dcf_base = None
    if dcf is not None:
        try:
            fcf = info.get("freeCashflow")
            d_dcf = info.get("totalDebt") or 0.0
            c_dcf = info.get("totalCash") or info.get("cashAndCashEquivalents") or 0.0
            a_dcf = info.get("sharesOutstanding")
            if fcf and fcf > 0 and a_dcf and a_dcf > 0:
                esc = dcf.calcular_escenarios_dcf(float(fcf), float(d_dcf), float(c_dcf), float(a_dcf))
                if esc and "base" in esc:
                    c_base = esc["base"].get("valor_por_accion") if isinstance(esc["base"], dict) else esc["base"]
                    if c_base and c_base > 0 and np.isfinite(c_base):
                        valor_dcf_base = float(c_base)
        except Exception:
            pass

    datos_tec = {"precio": precio, "ma20": ma20, "ma50": ma50, "ma200": ma200, "rsi": rsi, "volatilidad": volatilidad, "variacion": variacion, "ticker": ticker_symbol}
    datos_val = {"valor_dcf_base": valor_dcf_base, "pe": pe, "forward_pe": forward_pe, "peg": peg, "price_to_book": price_to_book}
    datos_fund = {"roe": roe, "margen": margen, "margen_operativo": margen_operativo, "deuda": deuda, "flujo_caja": flujo_caja, "ingresos": ingresos, "beneficio": beneficio, "eps": eps}
    datos_crec = {"crecimiento_ingresos": crecimiento_ingresos, "crecimiento_beneficios": crecimiento_beneficios, "dividend_yield": info.get("dividendYield")}
    datos_analistas = {"obj_med": obj_med}

    res_noticias = analizar_noticias(ticker_symbol)

    res_score = calcular_market_ai_score(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas)
    diag_pred = generar_diagnostico_market_ai(res_score, datos_tec, datos_val, datos_fund, datos_crec, datos_analistas, res_noticias)

    return {
        "ticker": ticker_symbol,
        "activo": ticker_symbol,
        "tipo": "📈 Acción",
        "nombre": info.get("shortName") or info.get("longName") or ticker_symbol,
        "precio": precio,
        "score": res_score["score_total"],
        "fair_value": valor_dcf_base,
        "potencial": diag_pred["pot_dcf"] if diag_pred["pot_dcf"] is not None else 0.0,
        "potencial_dcf": diag_pred["pot_dcf"],
        "direccion": diag_pred["direccion"],
        "sesgo": diag_pred["sesgo"],
        "confianza": diag_pred["confianza"],
        "horizonte": diag_pred["horizonte"],
        "calidad": diag_pred["calidad"],
        "explicacion": diag_pred["explicacion"],
        "veredicto": diag_pred["veredicto"],
        "riesgo": "Medio",
        "oportunidad_global": res_score["score_total"] * 0.7 + diag_pred["confianza"] * 0.3
    }
    
# ==========================================
# PARTE C: PREDICCIÓN MARKET AI (AUTOCORREGIDA)
# ==========================================
# Mapeo inteligente de variables para evitar NameError
d_tec = globals().get('datos_tec_in', globals().get('datos_tec', {}))
d_val = globals().get('datos_val_in', globals().get('datos_val', {}))
d_fund = globals().get('datos_fund_in', globals().get('datos_fund', {}))
d_crec = globals().get('datos_crec_in', globals().get('datos_crec', {}))
d_analistas = globals().get('datos_analistas_in', globals().get('datos_analistas', {}))
d_noticias = globals().get('res_noticias', {})

# 1. Calculamos la predicción
pred_res = ejecutar_prediction_engine(
    d_tec, 
    d_val, 
    d_fund, 
    d_crec, 
    d_analistas, 
    d_noticias
)

# 2. Mostramos los datos en pantalla
st.divider()
st.header("🔮 PREDICCIÓN MARKET AI")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dirección Probable", pred_res["direccion"])
c2.metric("Confianza", f"{pred_res['confianza']}%")
c3.metric("Datos Utilizados", f"{pred_res['datos_utilizados_pct']}%")
c4.metric("Score Predictivo", f"{pred_res['score_predictivo']:+.1f}")

st.subheader("📊 Probabilidades por Escenario")
pc1, pc2, pc3 = st.columns(3)
pc1.metric("🟢 Alcista", f"{pred_res['probabilidades']['alcista']}%")
pc2.metric("🟡 Base", f"{pred_res['probabilidades']['base']}%")
pc3.metric("🔴 Bajista", f"{pred_res['probabilidades']['bajista']}%")

if pred_res.get("objetivos"):
    st.subheader("🎯 Objetivos de Precio")
    po1, po2, po3, po4 = st.columns(4)
    po1.metric("Precio Actual", f"${pred_res['objetivos']['actual']}")
    po2.metric("Objetivo Alcista", f"${pred_res['objetivos']['alcista']}")
    po3.metric("Escenario Base", f"${pred_res['objetivos']['base']}")
    po4.metric("Objetivo Bajista", f"${pred_res['objetivos']['bajista']}")

st.subheader("⏱️ Horizontes Temporales")
h_cols = st.columns(4)
for i, (h_nombre, h_dir) in enumerate(pred_res.get("horizontes", {}).items()):
    with h_cols[i]:
        st.caption(h_nombre)
        st.write(f"**{h_dir}**")

st.subheader("🧠 Razonamiento del Modelo")
st.info(f"**{pred_res['escenario_dominante']}:** {pred_res['razon_dominante']}")

col_pos, col_neg = st.columns(2)
with col_pos:
    st.markdown("### 🟢 Señales Alcistas")
    if pred_res["senales_pos"]:
        for s in pred_res["senales_pos"]:
            st.write(f"• **[{s['cat']}]** {s['desc']}")
    else:
        st.caption("No hay señales alcistas destacadas.")

with col_neg:
    st.markdown("### 🔴 Señales Bajistas")
    if pred_res["senales_neg"]:
        for s in pred_res["senales_neg"]:
            st.write(f"• **[{s['cat']}]** {s['desc']}")
    else:
        st.caption("No hay señales bajistas destacadas.")

# =========================================================
# 🌎 MARKET AI MASTER RANKING (SECCIÓN PRINCIPAL)
# =========================================================

st.title("🌎 MARKET AI MASTER RANKING")
st.caption("Encuentra las oportunidades más interesantes detectadas por MARKET AI en el mercado.")

if st.button("🔎 BUSCAR MEJORES OPORTUNIDADES"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🔄 Analizando mercado (Acciones S&P 500 y Metales)...")
    progress_bar.progress(10)

    resultados_globales = []

    # 1. Escaneo S&P 500
    universo_sp500 = obtener_universo_sp500()
    candidatas_sp500 = ejecutar_filtro_rapido(universo_sp500)
    
    for idx, ticker in enumerate(candidatas_sp500):
        res = ejecutar_analisis_completo_ticker(ticker)
        if res:
            resultados_globales.append(res)
        progress_bar.progress(10 + int((idx + 1) / len(candidatas_sp500) * 60))

    # 2. Escaneo Metales
    status_text.text("🔄 Analizando futuros de metales...")
    res_metales = escaneo_metales_motor()
    resultados_globales.extend(res_metales)

    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    st.success("✅ Análisis completado.")

    resultados_globales.sort(key=lambda x: x.get('oportunidad_global', 0), reverse=True)

    num_acciones = sum(1 for x in resultados_globales if x.get('tipo') == '📈 Acción')
    num_metales = sum(1 for x in resultados_globales if x.get('tipo') == '🥇 Metal/Futuro')

    st.write(f"**Acciones analizadas:** {num_acciones} | **Metales analizados:** {num_metales} | **Oportunidades válidas:** {len(resultados_globales)}")
    st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if resultados_globales:
        # 1. MEJOR OPORTUNIDAD ACTUAL
        top1 = resultados_globales[0]
        st.markdown("---")
        st.subheader("🥇 MEJOR OPORTUNIDAD DETECTADA AHORA")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Activo", f"{top1.get('activo', top1.get('ticker'))} ({top1['tipo']})")
        col2.metric("Score", f"{top1['score']}/100")
        col3.metric("Dirección", top1['direccion'])
        col4.metric("Precio", f"${top1['precio']:.2f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Potencial", f"{top1['potencial']:+.2f}%")
        col6.metric("Confianza", f"{top1['confianza']}%")
        col7.metric("Horizonte", top1.get('horizonte', '📅 Medio plazo'))

        st.write("**¿Por qué?**")
        st.info(generar_explicacion(top1))

        # 2. CATEGORÍAS DESTACADAS
        st.markdown("---")
        st.subheader("📌 Mejor Oportunidad por Categoría")

        alcistas = [x for x in resultados_globales if "Alcista" in x.get('direccion', '')]
        bajistas = [x for x in resultados_globales if "Bajista" in x.get('direccion', '')]
        metales_list = [x for x in resultados_globales if x.get('tipo') == "🥇 Metal/Futuro"]

        c1, c2, c3 = st.columns(3)
        if alcistas:
            c1.success(f"📈 **Mejor Acción Alcista:** {alcistas[0].get('activo', alcistas[0].get('ticker'))} ({alcistas[0]['oportunidad_global']:.1f} Pts)")
        if bajistas:
            c2.error(f"🔻 **Mejor Acción Bajista:** {bajistas[0].get('activo', bajistas[0].get('ticker'))} ({bajistas[0]['oportunidad_global']:.1f} Pts)")
        if metales_list:
            c3.warning(f"🥇 **Mejor Metal:** {metales_list[0]['activo']} ({metales_list[0]['oportunidad_global']:.1f} Pts)")

        # 3. TOP 10 GLOBAL
        st.markdown("---")
        st.subheader("🏆 TOP 10 MARKET AI")

        top10 = resultados_globales[:10]
        df_top10 = pd.DataFrame(top10)[[
            'activo', 'tipo', 'score', 'precio', 'potencial', 
            'confianza', 'riesgo', 'direccion', 'horizonte', 'oportunidad_global'
        ]]
        df_top10.columns = ['Activo', 'Tipo', 'Score', 'Precio', 'Potencial (%)', 'Confianza (%)', 'Riesgo', 'Dirección', 'Horizonte', 'Oportunidad Global']
        st.dataframe(df_top10, use_container_width=True)

        # 4. TOP 5 ALCISTAS Y BAJISTAS
        col_alc, col_baj = st.columns(2)
        with col_alc:
            st.subheader("🟢 TOP 5 OPORTUNIDADES ALCISTAS")
            df_alc = pd.DataFrame(alcistas[:5])[['activo', 'tipo', 'score', 'potencial', 'confianza', 'riesgo']] if alcistas else pd.DataFrame()
            st.dataframe(df_alc, use_container_width=True)

        with col_baj:
            st.subheader("🔴 TOP 5 OPORTUNIDADES BAJISTAS")
            st.caption("Nota: Muestra señales de caída, sobrevaloración, debilidad técnica o deterioro fundamental.")
            df_baj = pd.DataFrame(bajistas[:5])[['activo', 'tipo', 'score', 'potencial', 'confianza', 'riesgo']] if bajistas else pd.DataFrame()
            st.dataframe(df_baj, use_container_width=True)

# Divisor para separar el Ranking Global del resto de tus herramientas existentes
st.markdown("---")

# =========================================================
# INTERFAZ DE USUARIO EN STREAMLIT: ANÁLISIS INDIVIDUAL
# =========================================================

st.title("📈 MARKET AI - Terminal de Análisis y Escáner")

# --- SECCIÓN 1: ESCÁNER AUTOMÁTICO DE MERCADO ---
st.divider()
st.header("🔎 ESCÁNER DE MERCADO")
st.write("Analiza automáticamente empresas del S&P 500 para encontrar las oportunidades alcistas y bajistas más interesantes.")

if st.button("🔎 ESCANEAR S&P 500", type="primary"):
    start_time = time.time()
    universo = obtener_universo_sp500()
    total_universo = len(universo)

    status_box = st.empty()
    progress_bar = st.progress(0)

    status_box.info(f"Fase 1/2: Ejecutando filtro rápido sobre {total_universo} empresas del S&P 500...")
    candidatas = ejecutar_filtro_rapido(universo)

    total_candidatas = len(candidatas)
    status_box.info(f"Fase 2/2: Analizando en profundidad {total_candidatas} empresas seleccionadas...")

    resultados_escaner = []
    errores_count = 0

    for idx, ticker in enumerate(candidatas):
        progress_bar.progress((idx + 1) / total_candidatas)
        status_box.text(f"Analizando candidata {idx+1}/{total_candidatas}: {ticker}...")

        res_comp = ejecutar_analisis_completo_ticker(ticker)
        if res_comp:
            resultados_escaner.append(res_comp)
        else:
            errores_count += 1

    elapsed_time = round(time.time() - start_time, 1)
    status_box.success(f"✅ ESCANEO COMPLETADO en {elapsed_time}s | Empresas analizadas: {total_universo} | Candidatas analizadas: {len(resultados_escaner)} ({errores_count} con datos insuficientes).")
    progress_bar.empty()

    if resultados_escaner:
        df_res = pd.DataFrame(resultados_escaner)

        # Algoritmo de Ranking Global (Score + Confianza + Coherencia)
        df_res["rank_global"] = df_res["score"] * 0.7 + df_res["confianza"] * 0.3
        df_res = df_res.sort_values(by="rank_global", ascending=False).reset_index(drop=True)

        st.subheader("🏆 TOP 5 MARKET AI (GLOBAL)")
        for idx, row in df_res.head(5).iterrows():
            medalla = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][idx]
            fv_str = f"${row['fair_value']:,.2f}" if row['fair_value'] else "N/D"
            pot_str = f"{row['potencial_dcf']:+.1f}%" if row['potencial_dcf'] is not None else "N/D"

            with st.expander(f"{medalla} {row['ticker']} - {row['nombre']} | Score: {row['score']:.1f}/100 | {row['direccion']}"):
                st.write(f"**Calidad:** {row['calidad']} | **Confianza:** {row['confianza']}% | **Horizonte:** {row['horizonte']}")
                st.write(f"**Precio Actual:** ${row['precio']:,.2f} | **Fair Value DCF:** {fv_str} | **Potencial:** {pot_str}")
                st.caption(f"**Principal motivo:** {row['explicacion']}")

        # TOP ALCISTAS & BAJISTAS
        st.divider()
        col_alc, col_baj = st.columns(2)

        with col_alc:
            st.subheader("🚀 TOP 5 OPORTUNIDADES ALCISTAS")
            df_alc = df_res[df_res["sesgo"] == "ALCISTA"].sort_values(by="score", ascending=False).head(5)
            if not df_alc.empty:
                for _, r in df_alc.iterrows():
                    st.write(f"🟢 **{r['ticker']}** ({r['nombre']}) — Score: **{r['score']:.1f}** | Confianza: {r['confianza']}%")
                    st.caption(f"{r['explicacion']}")
                    st.write("---")
            else:
                st.info("No se detectaron oportunidades alcistas claras en el grupo.")

        with col_baj:
            st.subheader("🔻 TOP 5 OPORTUNIDADES BAJISTAS")
            df_baj = df_res[df_res["sesgo"] == "BAJISTA"].sort_values(by="score", ascending=True).head(5)
            if not df_baj.empty:
                for _, r in df_baj.iterrows():
                    st.write(f"🔴 **{r['ticker']}** ({r['nombre']}) — Score: **{r['score']:.1f}** | Confianza: {r['confianza']}%")
                    st.caption(f"{r['explicacion']}")
                    st.write("---")
            else:
                st.info("No se detectaron oportunidades bajistas severas en el grupo.")

        # TABLA GENERAL
        st.subheader("📋 TABLA GENERAL DE CANDIDATAS ANALIZADAS")

        df_tabla = df_res[[
            "ticker", "nombre", "score", "direccion", "confianza", "precio", "fair_value", "potencial_dcf", "horizonte", "calidad"
        ]].copy()

        df_tabla.columns = ["Ticker", "Empresa", "Score", "Dirección", "Confianza", "Precio", "Fair Value", "Potencial DCF (%)", "Horizonte", "Calidad"]
        st.dataframe(df_tabla, use_container_width=True)

# --- SECCIÓN 2: ANÁLISIS INDIVIDUAL DE EMPRESA ---
st.divider()
st.header("🔍 ANÁLISIS INDIVIDUAL DE EMPRESA")

st.markdown("### ⚡ Accesos Rápidos")
b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)

if "ticker_seleccionado" not in st.session_state:
    st.session_state.ticker_seleccionado = "AAPL"

with b_col1:
    if st.button("🇺🇸 S&P 500 (^GSPC)"):
        st.session_state.ticker_seleccionado = "^GSPC"
with b_col2:
    if st.button("🍏 Apple (AAPL)"):
        st.session_state.ticker_seleccionado = "AAPL"
with b_col3:
    if st.button("💻 Microsoft (MSFT)"):
        st.session_state.ticker_seleccionado = "MSFT"
with b_col4:
    if st.button("🚀 Nvidia (NVDA)"):
        st.session_state.ticker_seleccionado = "NVDA"
with b_col5:
    if st.button("🔍 Google (GOOGL)"):
        st.session_state.ticker_seleccionado = "GOOGL"

st.sidebar.header("🔍 Búsqueda Individual")
ticker_input = st.sidebar.text_input(
    "Símbolo Ticker (ej: AAPL, MSFT, NVDA, ^GSPC):", 
    value=st.session_state.ticker_seleccionado
).upper().strip()

if ticker_input:
    info = obtener_info_accion(ticker_input)

    if info:
        nombre = info.get("longName", ticker_input)
        sector = info.get("sector", "N/D")
        industria = info.get("industry", "N/D")

        st.subheader(f"{nombre} ({ticker_input})")
        st.caption(f"**Sector:** {sector} | **Industria:** {industria}")

        df_hist = obtener_historico(ticker_input)

        precio_analisis = info.get("currentPrice") or info.get("regularMarketPrice")
        if (precio_analisis is None) and df_hist is not None and not df_hist.empty:
            precio_analisis = float(df_hist['Close'].iloc[-1])

        ma20 = float(df_hist['MA20'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'MA20' in df_hist and not pd.isna(df_hist['MA20'].iloc[-1])) else None
        ma50 = float(df_hist['MA50'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'MA50' in df_hist and not pd.isna(df_hist['MA50'].iloc[-1])) else None
        ma200 = float(df_hist['MA200'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'MA200' in df_hist and not pd.isna(df_hist['MA200'].iloc[-1])) else None
        rsi = float(df_hist['RSI'].iloc[-1]) if (df_hist is not None and not df_hist.empty and 'RSI' in df_hist and not pd.isna(df_hist['RSI'].iloc[-1])) else None

        if df_hist is not None and len(df_hist) > 1:
            variacion = float(((df_hist['Close'].iloc[-1] - df_hist['Close'].iloc[-2]) / df_hist['Close'].iloc[-2]) * 100)
            returns = df_hist['Close'].pct_change().dropna()
            volatilidad = float(returns.std() * np.sqrt(252) * 100)
        else:
            variacion = None
            volatilidad = None

        pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        peg = info.get("pegRatio")
        price_to_book = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        margen = info.get("profitMargins")
        margen_operativo = info.get("operatingMargins")
        deuda = info.get("debtToEquity")
        flujo_caja = info.get("freeCashflow")
        ingresos = info.get("totalRevenue")
        beneficio = info.get("netIncomeToCommon")
        eps = info.get("trailingEps")
        crecimiento_ingresos = info.get("revenueGrowth")
        crecimiento_beneficios = info.get("earningsGrowth")

        objetivos = obtener_objetivos_analistas(ticker_input)
        obj_med = objetivos.get("mean")

        escenarios_dcf = None
        valor_dcf_base = None

        if dcf is not None:
            try:
                free_cash_flow = info.get("freeCashflow")
                deuda_dcf = info.get("totalDebt") or 0.0
                caja_dcf = info.get("totalCash") or info.get("cashAndCashEquivalents") or 0.0
                acciones_dcf = info.get("sharesOutstanding")

                if free_cash_flow and free_cash_flow > 0 and acciones_dcf and acciones_dcf > 0:
                    escenarios_dcf = dcf.calcular_escenarios_dcf(
                        free_cash_flow=float(free_cash_flow),
                        deuda=float(deuda_dcf),
                        caja=float(caja_dcf),
                        acciones=float(acciones_dcf)
                    )

                if escenarios_dcf and isinstance(escenarios_dcf, dict) and "base" in escenarios_dcf:
                    esc_base = escenarios_dcf["base"]
                    candidato_base = esc_base.get("valor_por_accion") if isinstance(esc_base, dict) else esc_base
                    if (candidato_base is not None and isinstance(candidato_base, (int, float)) and np.isfinite(candidato_base) and candidato_base > 0):
                        valor_dcf_base = float(candidato_base)
            except Exception:
                pass

        datos_tec_in = {
            "precio": precio_analisis, "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "rsi": rsi, "volatilidad": volatilidad, "variacion": variacion, "ticker": ticker_input
        }
        datos_val_in = {
            "valor_dcf_base": valor_dcf_base, "pe": pe, "forward_pe": forward_pe,
            "peg": peg, "price_to_book": price_to_book
        }
        datos_fund_in = {
            "roe": roe, "margen": margen, "margen_operativo": margen_operativo, "deuda": deuda,
            "flujo_caja": flujo_caja, "ingresos": ingresos, "beneficio": beneficio, "eps": eps
        }
        datos_crec_in = {
            "crecimiento_ingresos": crecimiento_ingresos,
            "crecimiento_beneficios": crecimiento_beneficios,
            "dividend_yield": info.get("dividendYield")
        }
        datos_analistas_in = {
            "obj_med": obj_med
        }

        res_noticias = analizar_noticias(ticker_input)
        res_market_ai = calcular_market_ai_score(datos_tec_in, datos_val_in, datos_fund_in, datos_crec_in, datos_analistas_in)
        diag_pred = generar_diagnostico_market_ai(res_market_ai, datos_tec_in, datos_val_in, datos_fund_in, datos_crec_in, datos_analistas_in, res_noticias)

        diag = res_market_ai["diagnostico"]

        # --- Gráfico Histórico ---
        st.header("📈 Gráfico de Cotización")
        if df_hist is not None and not df_hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_hist.index,
                open=df_hist['Open'],
                high=df_hist['High'],
                low=df_hist['Low'],
                close=df_hist['Close'],
                name='Precio'
            ))
            if 'MA20' in df_hist:
                fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA20'], line=dict(color='orange', width=1.5), name='MA20'))
            if 'MA50' in df_hist:
                fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA50'], line=dict(color='blue', width=1.5), name='MA50'))
            if 'MA200' in df_hist:
                fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA200'], line=dict(color='red', width=1.5), name='MA200'))
            fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para renderear el gráfico.")

        # --- Sección Fundamentales ---
        st.header("📊 Fundamentales")
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        f_c1.metric("PER (Trailing)", f"{pe:.2f}x" if pe else "N/D")
        f_c2.metric("PER Futuro", f"{forward_pe:.2f}x" if forward_pe else "N/D")
        f_c3.metric("ROE", f"{roe*100:.2f}%" if roe else "N/D")
        f_c4.metric("Margen Neto", f"{margen*100:.2f}%" if margen else "N/D")

        f_c5, f_c6, f_c7, f_c8 = st.columns(4)
        f_c5.metric("Price / Book", f"{price_to_book:.2f}x" if price_to_book else "N/D")
        f_c6.metric("Deuda / Patrimonio", f"{deuda:.2f}%" if deuda else "N/D")
        f_c7.metric("Free Cash Flow", f"${flujo_caja:,.0f}" if flujo_caja else "N/D")
        f_c8.metric("EPS", f"${eps:.2f}" if eps else "N/D")

        # --- Sección Crecimiento ---
        st.header("🚀 Crecimiento")
        g_c1, g_c2, g_c3 = st.columns(3)
        g_c1.metric("Crecimiento Ingresos", f"{crecimiento_ingresos*100:+.2f}%" if crecimiento_ingresos else "N/D")
        g_c2.metric("Crecimiento Beneficios", f"{crecimiento_beneficios*100:+.2f}%" if crecimiento_beneficios else "N/D")
        g_c3.metric("Dividend Yield", f"{info.get('dividendYield')*100:.2f}%" if info.get('dividendYield') else "N/D")

        # --- Sección DCF ---
        if escenarios_dcf and isinstance(escenarios_dcf, dict):
            st.header("🎯 DCF / Valoración")
            dcf_cols = st.columns(len(escenarios_dcf))
            idx = 0
            for nombre_esc, datos_esc in escenarios_dcf.items():
                with dcf_cols[idx]:
                    st.subheader(f"Caso {str(nombre_esc).capitalize()}")
                    val_accion = datos_esc.get('valor_por_accion') if isinstance(datos_esc, dict) else datos_esc

                    if val_accion is not None and isinstance(val_accion, (int, float)) and np.isfinite(val_accion) and val_accion > 0:
                        st.metric("Fair Value", f"${val_accion:,.2f}")
                        pot = ((val_accion - precio_analisis) / precio_analisis) * 100 if precio_analisis else None
                        st.metric("Potencial", f"{pot:+.1f}%" if pot is not None else "N/D")
                    else:
                        st.metric("Fair Value", "N/D")
                        st.metric("Potencial", "N/D")
                    idx += 1

        # --- Sección Analistas ---
        st.header("🧑‍💼 Analistas")
        a_c1, a_c2, a_c3, a_c4 = st.columns(4)
        a_c1.metric("Objetivo Mínimo", f"${objetivos.get('low'):,.2f}" if objetivos.get('low') else "N/D")
        a_c2.metric("Objetivo Medio", f"${objetivos.get('mean'):,.2f}" if objetivos.get('mean') else "N/D")
        a_c3.metric("Objetivo Máximo", f"${objetivos.get('high'):,.2f}" if objetivos.get('high') else "N/D")
        a_c4.metric("Mediana", f"${objetivos.get('median'):,.2f}" if objetivos.get('median') else "N/D")

        st.subheader("Consenso de Analistas")
        if "consensus" in objetivos:
            st.info(f"**Recomendación de Consenso:** {objetivos.get('consensus')} (Basado en {objetivos.get('num_analistas', 'N/D')} opiniones)")

        st.subheader("Cambios Recientes de Analistas")
        rec_df = obtener_recomendaciones_analistas(ticker_input)
        if rec_df is not None and not rec_df.empty:
            st.dataframe(rec_df.tail(10), use_container_width=True)
        else:
            st.write("No hay registros recientes de revisiones disponibles.")

        # --- Sección Estimaciones ---
        st.header("🔮 Estimaciones")
        eps_df = obtener_estimaciones_eps(ticker_input)
        if eps_df is not None and not eps_df.empty:
            st.dataframe(eps_df, use_container_width=True)
        else:
            st.write("No hay estimaciones de EPS disponibles.")

        # --- Sección Noticias y Sentimiento ---
        st.divider()
        st.header("📰 Noticias y Sentimiento")

        ns_col1, ns_col2 = st.columns(2)
        with ns_col1:
            st.metric("Sentimiento General", res_noticias["sentimiento_general"])
        with ns_col2:
            st.metric("Puntuación de Sentimiento", f"{res_noticias['puntuacion_sentimiento']} / 100")

        st.write(f"*{res_noticias['resumen']}*")

        listado_noticias = res_noticias.get("noticias", [])
        if listado_noticias:
            for item in listado_noticias[:5]:
                n_col1, n_col2 = st.columns([4, 1])
                with n_col1:
                    if item['url'] != "#":
                        st.markdown(f"**[{item['titulo']}]({item['url']})**")
                    else:
                        st.markdown(f"**{item['titulo']}**")
                    st.caption(f"Fuente: {item['fuente']} | Fecha: {item['fecha']}")
                with n_col2:
                    st.write(f"{item['sentimiento']} ({item['importancia']})")
                st.write("---")
        else:
            st.info("No hay noticias recientes suficientes.")

        # --- Sección MARKET AI SCORE ---
        st.divider()
        st.header("🎯 MARKET AI SCORE")

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.metric("SCORE GLOBAL", f"{res_market_ai['score_total']:.1f} / 100")
            st.markdown(f"### {res_market_ai['categoria']}")

        with col_s2:
            st.write("**Desglose de la Puntuación:**")
            st.write(
                f"📈 **Técnico:** {res_market_ai['desglose']['tecnico']:.0f}/25 &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"💰 **Valoración:** {res_market_ai['desglose']['valoracion']:.0f}/25 &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"📊 **Fundamentales:** {res_market_ai['desglose']['fundamentales']:.0f}/25 &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"🚀 **Crecimiento:** {res_market_ai['desglose']['crecimiento']:.0f}/15 &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"⚠️ **Riesgo:** {res_market_ai['desglose']['riesgo']:.0f}/10"
            )
            st.progress(res_market_ai['score_total'] / 100)

        # --- SECCIÓN FINAL: DIAGNÓSTICO FINAL DE MARKET AI ---
        st.divider()
        st.header("🧠 DIAGNÓSTICO FINAL DE MARKET AI")

        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Dirección Probable", diag_pred["direccion"])
        p_col2.metric("Confianza del Diagnóstico", f"{diag_pred['confianza']}%")
        p_col3.metric("Horizonte Temporal", diag_pred["horizonte"])

        st.info(f"**Síntesis:** {diag_pred['explicacion']}")

        st.subheader("💵 Precio y Valoración Comparativa")
        v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)
        v_col1.metric("Precio Actual", f"${diag['precio']:,.2f}" if diag['precio'] else "N/D")
        v_col2.metric("Fair Value DCF", f"${diag['fair_value']:,.2f}" if diag['fair_value'] else "N/D")
        v_col3.metric("Potencial DCF", f"{diag['potencial_dcf']:+.1f}%" if diag['potencial_dcf'] is not None else "N/D")
        v_col4.metric("Objetivo Analistas", f"${diag['target_analistas']:,.2f}" if diag['target_analistas'] else "N/D")
        v_col5.metric("Potencial Analistas", f"{diag['potencial_analistas']:+.1f}%" if diag['potencial_analistas'] is not None else "N/D")

        st.subheader("⚖️ SEÑALES A FAVOR Y EN CONTRA")
        sig_col1, sig_col2 = st.columns(2)
        with sig_col1:
            st.write("🟢 **A FAVOR**")
            for item in diag_pred["signals_pos"]:
                st.write(f"- {item}")
        with sig_col2:
            st.write("🔴 **EN CONTRA**")
            for item in diag_pred["signals_neg"]:
                st.write(f"- {item}")

        cr_col1, cr_col2 = st.columns(2)
        with cr_col1:
            st.subheader("🚀 PRÓXIMOS CATALIZADORES")
            cat_list = set(diag["catalizadores"] + res_noticias["catalizadores"])
            if cat_list:
                for c_item in cat_list:
                    st.write(f"- {c_item}")
            else:
                st.write("- No hay acontecimientos disparadores detectados.")

        with cr_col2:
            st.subheader("⚠️ PRINCIPALES RIESGOS")
            r_list = set(diag["riesgos"] + res_noticias["riesgos"])
            if r_list:
                for r_item in r_list:
                    st.write(f"- {r_item}")
            else:
                st.write("- Sin factores de riesgo crítico detectados.")

        st.subheader("🎯 VEREDICTO MARKET AI")
        st.markdown(f"## {diag_pred['veredicto']}")

        st.caption("⚠️ **Aviso de Responsabilidad:** Este diagnóstico es una estimación algorítmica basada en los datos disponibles y no garantiza el comportamiento futuro del mercado.")

    else:
        st.error("No se pudo cargar la información para el ticker introducido. Verifica el símbolo.")
        
   # ==========================================
# SECCIÓN: MARKET AI BACKTESTING
# ==========================================
from backtesting_engine import ejecutar_backtest_engine

st.divider()
st.header("📊 BACKTESTING Y MEDICIÓN DE PRECISIÓN MARKET AI")
st.caption("Comprueba cómo habrían funcionado históricamente las señales de MARKET AI.")

# 1. Controles de Selección
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
  col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    tipo_activo_bt = st.radio("Tipo de Activo", ["📈 Acciones", "🥇 Metales/Futuros"], horizontal=True, key="rad_tipo_activo_bt_unique")

with col_b2:
    if tipo_activo_bt == "📈 Acciones":
        ticker_bt = st.text_input("Ticker Backtest", value="AAPL", key="txt_ticker_bt_unique").upper()
        es_metal_bt = False
    else:
        ticker_bt = st.selectbox("Metal / Futuro Backtest", ["GC=F", "SI=F", "HG=F", "CL=F"], key="sb_metal_bt_unique")
        es_metal_bt = True

with col_b3:
    periodo_meses = st.selectbox("Periodo Histórico Evaluado", [3, 6, 12, 24], index=2, format_func=lambda x: f"{x} Meses", key="sb_periodo_meses_bt_unique")

with col_b4:
    horizonte_dias = st.selectbox("Horizonte de Predicción", [1, 5, 20, 60, 120], index=2, format_func=lambda x: f"{x} Días", key="sb_horizonte_dias_bt_unique")
# 2. Botón de Ejecución con parámetros explicito
if st.button("🔎 EJECUTAR BACKTEST", use_container_width=True):
    with st.spinner("Evaluando rendimiento histórico sin Look-Ahead Bias..."):
        res_bt = ejecutar_backtest_engine(
            ticker=ticker_bt,
            periodo_meses=periodo_meses,
            horizonte_dias=horizonte_dias,
            es_metal=es_metal_bt
        )
        
        if res_bt is None:
            st.warning("N/D - Datos históricos insuficientes para el periodo y horizonte seleccionados.")
        else:
            df_bt, ret_buy_hold = res_bt
            
            if df_bt.empty:
                st.warning("N/D - No se pudieron generar muestras con los parámetros seleccionados.")
            else:
                # Métricas Globales
                tot_pred = len(df_bt)
                aciertos = len(df_bt[df_bt["resultado"] == "✅ Acierto"])
                tasa_acierto = (aciertos / tot_pred) * 100 if tot_pred > 0 else 0
                ret_media = df_bt["rentabilidad"].mean()
                mejor_res = df_bt["rentabilidad"].max()
                peor_res = df_bt["rentabilidad"].min()
                ret_acum_mai = df_bt["rentabilidad"].sum()

                st.subheader("🎯 Resumen de Rendimiento")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Predicciones", tot_pred)
                m2.metric("Tasa de Acierto", f"{tasa_acierto:.1f}%")
                m3.metric("Rentabilidad Media", f"{ret_media:+.2f}%")
                m4.metric("Mejor Resultado", f"{mejor_res:+.2f}%")
                m5.metric("Peor Resultado", f"{peor_res:+.2f}%")
                
                # Comparativa MARKET AI vs BUY & HOLD
                st.subheader("⚔️ MARKET AI vs BUY & HOLD")
                c_comp1, c_comp2 = st.columns(2)
                c_comp1.metric("Estrategia MARKET AI (Acumulada)", f"{ret_acum_mai:+.2f}%")
                c_comp2.metric(f"Buy & Hold ({ticker_bt})", f"{ret_buy_hold:+.2f}%")
                
                # Desglose de Precisión
                st.subheader("📊 Desglose de Precisión")
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("**Precisión por Rango de Confianza**")
                    bins_conf = [49, 59, 69, 79, 89, 100]
                    labels_conf = ["50-59%", "60-69%", "70-79%", "80-89%", "90-100%"]
                    df_bt['rango_conf'] = pd.cut(df_bt['confianza'], bins=bins_conf, labels=labels_conf)
                    
                    conf_stats = df_bt.groupby('rango_conf', observed=False).agg(
                        Predicciones=('resultado', 'count'),
                        Aciertos=('resultado', lambda x: (x == "✅ Acierto").sum()),
                        Tasa_Acierto=('resultado', lambda x: f"{((x == '✅ Acierto').sum()/len(x)*100):.1f}%" if len(x)>0 else "0%")
                    )
                    st.dataframe(conf_stats, use_container_width=True)
                    
                with col_d2:
                    st.markdown("**Precisión por Rango de Score**")
                    bins_score = [-1, 39, 54, 69, 84, 100]
                    labels_score = ["0-39", "40-54", "55-69", "70-84", "85-100"]
                    df_bt['rango_score'] = pd.cut(df_bt['score'], bins=bins_score, labels=labels_score)
                    
                    score_stats = df_bt.groupby('rango_score', observed=False).agg(
                        Predicciones=('resultado', 'count'),
                        Aciertos=('resultado', lambda x: (x == "✅ Acierto").sum()),
                        Rentabilidad_Media=('rentabilidad', lambda x: f"{x.mean():+.2f}%" if len(x)>0 else "0.00%")
                    )
                    st.dataframe(score_stats, use_container_width=True)

                # Gráficos
                st.subheader("📈 Evolución Acumulada de la Estrategia")
                df_bt["ret_acum"] = df_bt["rentabilidad"].cumsum()
                st.line_chart(df_bt.set_index("fecha")["ret_acum"])

                # Análisis de Señales
                st.subheader("🧠 ¿Qué señales funcionan mejor?")
                ma_match = df_bt[df_bt["ma20_gt_ma50"] & (df_bt["resultado"] == "✅ Acierto")]
                pct_ma_match = (len(ma_match) / tot_pred) * 100 if tot_pred > 0 else 0
                
                st.info(
                    f"Históricamente, la condición **MA20 > MA50** estuvo asociada con un acierto directo "
                    f"en el **{pct_ma_match:.1f}%** de todas las predicciones evaluadas para {ticker_bt}."
                )

                # Tabla Registro
                st.subheader("📋 Registro de Predicciones Históricas")
                st.dataframe(
                    df_bt[["fecha", "ticker", "score", "direccion", "confianza", "precio_inicial", "precio_final", "rentabilidad", "resultado"]],
                    use_container_width=True
                )

# ==========================================
# SECCIÓN: MARKET AI MODEL OPTIMIZER
# ==========================================
from model_optimizer import optimizar_pesos_historicos

st.divider()
st.header("🧠 OPTIMIZACIÓN Y CALIBRACIÓN DEL MODELO MARKET AI")
st.caption("Evalúa señales históricas y valida configuraciones sin Overfitting ni Look-Ahead Bias.")

col_opt1, col_opt2, col_opt3 = st.columns(3)

with col_opt1:
    tipo_activo_opt = st.radio("Mercado", ["📈 Acciones", "🥇 Metales/Futuros"], key="rad_opt_market")
    
with col_opt2:
    if tipo_activo_opt == "📈 Acciones":
        ticker_opt = st.text_input("Ticker Optimización", value="AAPL", key="txt_opt_ticker").upper()
        es_metal_opt = False
    else:
        ticker_opt = st.selectbox("Metal / Futuro", ["GC=F", "SI=F", "HG=F", "CL=F"], key="sb_opt_metal")
        es_metal_opt = True

with col_opt3:
    periodo_opt = st.selectbox("Ventana de Datos", [12, 24, 36], index=1, format_func=lambda x: f"{x} Meses", key="sb_opt_period")

if st.button("🚀 EJECUTAR OPTIMIZACIÓN DE MODELO", use_container_width=True, key="btn_run_optimizer"):
    with st.spinner("Analizando independencia de señales y validando subconjuntos TRAIN / VALIDATION..."):
        res_opt, err_opt = optimizar_pesos_historicos(
            ticker=ticker_opt,
            periodo_meses=periodo_opt,
            horizonte_dias=20,
            es_metal=es_metal_opt
        )
        
        if res_opt is None:
            st.warning(f"⚠️ {err_opt}")
        else:
            mod_act = res_opt["modelo_actual"]
            mod_opt = res_opt["modelo_optimizado"]
            
            # 1. Comparativa Modelo Actual vs Optimizado
            st.subheader("⚖️ Configuración Actualmente Activa vs Recomendada")
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.info("📌 **CONFIGURACIÓN ACTUALMENTE ACTIVA**")
                p_act = mod_act["pesos"]
                st.write(f"- **Técnico:** {p_act[0]}% | **Valoración:** {p_act[1]}%")
                st.write(f"- **Fundamentales:** {p_act[2]}% | **Crecimiento:** {p_act[3]}%")
                st.write(f"- **Noticias/Sentimiento:** {p_act[4]}% | **Riesgo:** {p_act[5]}%")
                st.markdown(f"**Tasa Acierto (TRAIN / VAL):** {mod_act['acierto_train']}% / **{mod_act['acierto_val']}%**")
                st.markdown(f"**Rent. Media (VAL):** {mod_act['rent_val']:+.2f}%")
                
            with col_m2:
                st.success("🔬 **MARKET AI RECOMIENDA ESTA CONFIGURACIÓN**")
                p_opt = mod_opt["pesos"]
                st.write(f"- **Técnico:** {p_opt[0]}% | **Valoración:** {p_opt[1]}%")
                st.write(f"- **Fundamentales:** {p_opt[2]}% | **Crecimiento:** {p_opt[3]}%")
                st.write(f"- **Noticias/Sentimiento:** {p_opt[4]}% | **Riesgo:** {p_opt[5]}%")
                st.markdown(f"**Tasa Acierto (TRAIN / VAL):** {mod_opt['acierto_train']}% / **{mod_opt['acierto_val']}%**")
                st.markdown(f"**Rent. Media (VAL):** {mod_opt['rent_val']:+.2f}%")
                
            st.warning("⚠️ **Nota de Control de Overfitting:** La recomendación no modifica automáticamente los pesos del sistema en producción. Requiere validación por el usuario.")

            # 2. División de Datos
            st.markdown(f"📊 **Subconjuntos Evaluados:** TRAIN ({res_opt['num_train']} muestras) | VALIDATION ({res_opt['num_val']} muestras)")

            # 3. Comparación Visual TRAIN vs VALIDATION
            st.subheader("📈 Comparativa de Precisión por Perfil de Pesos")
            df_perf_plot = pd.DataFrame(res_opt["perfiles_evaluados"])
            st.bar_chart(df_perf_plot.set_index("nombre")[["acierto_train", "acierto_val"]])

            # 4. Ranking de Señales Individuales
            st.subheader("🎯 Ranking de Fiabilidad de Señales Individuales")
            df_senales_disp = res_opt["tabla_senales"].drop(columns=["_pct_val"], errors="ignore")
            st.dataframe(df_senales_disp, use_container_width=True)
