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

st.set_page_config(page_title="MARKET AI - Análisis, Escáner y Predicción", layout="wide", page_icon="📈")

# =========================================================
# CONFIGURACIÓN Y API KEYS
# =========================================================
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# =========================================================
# UNIVERSO DE EMPRESAS DE RESPALDO Y OBTENCIÓN DE S&P 500
# =========================================================

RESPALDO_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "LLY", "AVGO",
    "JPM", "WMT", "UNH", "V", "PG", "MA", "ORCL", "HD", "JNJ", "COST", "ABBV",
    "BAC", "KO", "NFLX", "CRM", "CVX", "MRK", "AMD", "PEP", "TMO", "LIN", "WFC",
    "ADBE", "MCD", "DIS", "PM", "CSCO", "ABT", "GE", "INTU", "CAT", "TXN", "AMAT",
    "VZ", "AXP", "IBM", "QCOM", "PFE", "COP"
]

@st.cache_data(ttl=86400)
def obtener_universo_sp500():
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

        resumen_gen = f"Se han analizado {len(noticias_procesadas)} noticias recientes. El volumen presenta {pos_count} noticias favorables, {neg_count} desfavorables y {len(noticias_procesadas) - pos_count - neg_count} neutrales."

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
# 🔮 NEW MARKET AI PREDICTION ENGINE
# =========================================================

def ejecutar_prediction_engine(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas, res_noticias, es_metal=False):
    """
    MOTOR DE PREDICCIÓN INDEPENDIENTE.
    Evalúa señales por categoría, calcula puntuación predictiva ponderada (-100 a +100),
    genera probabilidades que suman 100%, horizontes temporales y objetivos por escenario.
    """
    senales_pos = []
    senales_neu = []
    senales_neg = []

    # 1. EVALUACIÓN Y CATEGORIZACIÓN DE SEÑALES
    cat_scores = {}
    cat_weights_base = {
        "tecnico": 0.30 if not es_metal else 0.50,
        "valoracion": 0.20 if not es_metal else 0.00,
        "fundamentales": 0.15 if not es_metal else 0.00,
        "crecimiento": 0.10 if not es_metal else 0.00,
        "noticias": 0.15 if not es_metal else 0.30,
        "riesgo": 0.10 if not es_metal else 0.20
    }

    # --- TÉCNICO ---
    precio = datos_tec.get("precio")
    ma20 = datos_tec.get("ma20")
    ma50 = datos_tec.get("ma50")
    ma200 = datos_tec.get("ma200")
    rsi = datos_tec.get("rsi")
    volatilidad = datos_tec.get("volatilidad")

    score_tec = 0.0
    cnt_tec = 0
    if precio and ma20 and ma50:
        cnt_tec += 1
        if precio > ma20 > ma50:
            score_tec += 100
            senales_pos.append({"cat": "Técnico", "desc": "Cruce alcista: Precio > MA20 > MA50", "origen": "Técnico"})
        elif precio < ma20 < ma50:
            score_tec -= 100
            senales_neg.append({"cat": "Técnico", "desc": "Estructura bajista: Precio < MA20 < MA50", "origen": "Técnico"})
        else:
            senales_neu.append({"cat": "Técnico", "desc": "Media móvil MA20/MA50 lateral", "origen": "Técnico"})

    if precio and ma200:
        cnt_tec += 1
        if precio > ma200:
            score_tec += 100
            senales_pos.append({"cat": "Técnico", "desc": "Precio por encima de la tendencia MA200", "origen": "Técnico"})
        else:
            score_tec -= 100
            senales_neg.append({"cat": "Técnico", "desc": "Precio cotizando bajo la MA200", "origen": "Técnico"})

    if rsi is not None and not np.isnan(rsi):
        cnt_tec += 1
        if rsi < 30:
            score_tec += 80
            senales_pos.append({"cat": "Técnico", "desc": f"RSI en sobreventa ({rsi:.1f}) propicio para rebotar", "origen": "Técnico"})
        elif rsi > 70:
            score_tec -= 80
            senales_neg.append({"cat": "Técnico", "desc": f"RSI en sobrecompra ({rsi:.1f}), posible recorte", "origen": "Técnico"})
        else:
            senales_neu.append({"cat": "Técnico", "desc": f"RSI neutral ({rsi:.1f})", "origen": "Técnico"})

    if cnt_tec > 0:
        cat_scores["tecnico"] = score_tec / cnt_tec

    # --- VALORACIÓN (Solo Acciones) ---
    if not es_metal:
        score_val = 0.0
        cnt_val = 0
        fair_value = datos_val.get("valor_dcf_base")
        obj_med = datos_analistas.get("obj_med")

        if fair_value and precio and precio > 0:
            cnt_val += 1
            pot_dcf = ((fair_value - precio) / precio) * 100
            if pot_dcf >= 15:
                score_val += 100
                senales_pos.append({"cat": "Valoración", "desc": f"Descuento intrínseco DCF del +{pot_dcf:.1f}%", "origen": "DCF"})
            elif pot_dcf <= -15:
                score_val -= 100
                senales_neg.append({"cat": "Valoración", "desc": f"Sobrevaloración respecto al DCF del {abs(pot_dcf):.1f}%", "origen": "DCF"})
            else:
                senales_neu.append({"cat": "Valoración", "desc": f"Precio alineado con Fair Value DCF ({pot_dcf:+.1f}%)", "origen": "DCF"})

        if obj_med and precio and precio > 0:
            cnt_val += 1
            pot_analistas = ((obj_med - precio) / precio) * 100
            if pot_analistas >= 10:
                score_val += 80
                senales_pos.append({"cat": "Valoración", "desc": f"Objetivo de analistas atractivo (+{pot_analistas:.1f}%)", "origen": "Consenso"})
            elif pot_analistas <= -10:
                score_val -= 80
                senales_neg.append({"cat": "Valoración", "desc": f"Precio sobre el objetivo medio de analistas ({pot_analistas:.1f}%)", "origen": "Consenso"})

        if cnt_val > 0:
            cat_scores["valoracion"] = score_val / cnt_val

    # --- FUNDAMENTALES (Solo Acciones) ---
    if not es_metal:
        score_fund = 0.0
        cnt_fund = 0
        roe = datos_fund.get("roe")
        deuda = datos_fund.get("deuda")
        fcf = datos_fund.get("flujo_caja")

        if roe is not None and not np.isnan(roe):
            cnt_fund += 1
            if roe >= 0.15:
                score_fund += 100
                senales_pos.append({"cat": "Fundamentales", "desc": f"Excelente rentabilidad sobre capital (ROE {roe*100:.1f}%)", "origen": "Balance"})
            elif roe < 0.05:
                score_fund -= 80
                senales_neg.append({"cat": "Fundamentales", "desc": f"ROE débil ({roe*100:.1f}%)", "origen": "Balance"})

        if fcf is not None and not np.isnan(fcf):
            cnt_fund += 1
            if fcf > 0:
                score_fund += 80
                senales_pos.append({"cat": "Fundamentales", "desc": "Generación sólida de caja libre (FCF Positivo)", "origen": "Caja"})
            else:
                score_fund -= 100
                senales_neg.append({"cat": "Fundamentales", "desc": "Free Cash Flow negativo (Quema de caja)", "origen": "Caja"})

        if cnt_fund > 0:
            cat_scores["fundamentales"] = score_fund / cnt_fund

    # --- CRECIMIENTO (Solo Acciones) ---
    if not es_metal:
        score_crec = 0.0
        cnt_crec = 0
        c_ing = datos_crec.get("crecimiento_ingresos")
        c_ben = datos_crec.get("crecimiento_beneficios")

        if c_ing is not None and not np.isnan(c_ing):
            cnt_crec += 1
            if c_ing > 0.08:
                score_crec += 100
                senales_pos.append({"cat": "Crecimiento", "desc": f"Sólido crecimiento de ingresos (+{c_ing*100:.1f}%)", "origen": "Resultados"})
            elif c_ing < 0:
                score_crec -= 100
                senales_neg.append({"cat": "Crecimiento", "desc": f"Contracción en las ventas ({c_ing*100:.1f}%)", "origen": "Resultados"})

        if cnt_crec > 0:
            cat_scores["crecimiento"] = score_crec / cnt_crec

    # --- NOTICIAS Y SENTIMIENTO ---
    if res_noticias:
        sent_puntuacion = res_noticias.get("puntuacion_sentimiento", 0)
        cat_scores["noticias"] = float(sent_puntuacion)
        if sent_puntuacion >= 25:
            senales_pos.append({"cat": "Noticias", "desc": f"Flujo de noticias favorable (+{sent_puntuacion}/100)", "origen": "Sentimiento"})
        elif sent_puntuacion <= -25:
            senales_neg.append({"cat": "Noticias", "desc": f"Sesgo mediático desfavorables ({sent_puntuacion}/100)", "origen": "Sentimiento"})
        else:
            senales_neu.append({"cat": "Noticias", "desc": "Prensa neutral sin titulares extremos", "origen": "Sentimiento"})

    # --- RIESGO ---
    score_riesgo = 0.0
    cnt_riesgo = 0
    if volatilidad is not None and not np.isnan(volatilidad):
        cnt_riesgo += 1
        if volatilidad < 25:
            score_riesgo += 80
            senales_pos.append({"cat": "Riesgo", "desc": f"Baja volatilidad histórica ({volatilidad:.1f}%)", "origen": "Mercado"})
        elif volatilidad > 40:
            score_riesgo -= 80
            senales_neg.append({"cat": "Riesgo", "desc": f"Volatilidad anualizada elevada ({volatilidad:.1f}%)", "origen": "Mercado"})

    if cnt_riesgo > 0:
        cat_scores["riesgo"] = score_riesgo / cnt_riesgo

    # 2. PONDERACIÓN NORMALIZADA Y PUNTUACIÓN PREDICTIVA (-100 A +100)
    pesos_activos = {k: cat_weights_base[k] for k in cat_scores if k in cat_weights_base}
    total_peso_disponible = sum(pesos_activos.values())

    if total_peso_disponible > 0:
        predictive_score = sum(cat_scores[k] * (pesos_activos[k] / total_peso_disponible) for k in cat_scores)
        datos_utilizados_pct = round((total_peso_disponible / sum(cat_weights_base.values())) * 100, 1)
    else:
        predictive_score = 0.0
        datos_utilizados_pct = 0.0

    predictive_score = float(np.clip(predictive_score, -100.0, 100.0))

    # 3. DIRECCIÓN PROBABLE
    if predictive_score >= 50.0:
        direccion_pred = "🟢 ALCISTA"
        sesgo_pred = "ALCISTA"
    elif predictive_score <= -50.0:
        direccion_pred = "🔴 BAJISTA"
        sesgo_pred = "BAJISTA"
    else:
        direccion_pred = "🟡 NEUTRAL"
        sesgo_pred = "NEUTRAL"

    # 4. CONFIANZA DE LA PREDICCIÓN (0-100%)
    factor_cobertura = datos_utilizados_pct / 100.0
    total_senales = len(senales_pos) + len(senales_neg)
    if total_senales > 0:
        coherencia = abs(len(senales_pos) - len(senales_neg)) / total_senales
    else:
        coherencia = 0.5

    vol_pen = max(0.0, 1.0 - ((volatilidad or 25) / 100.0))
    confianza = int(np.clip((factor_cobertura * 0.4 + coherencia * 0.4 + vol_pen * 0.2) * 100, 15, 95))

    # 5. PROBABILIDADES POR ESCENARIO (SUMA EXACTA DE 100%)
    norm_sc = predictive_score / 100.0
    conf_f = confianza / 100.0

    prob_alc = max(5.0, 33.3 + (45.0 * norm_sc * conf_f))
    prob_baj = max(5.0, 33.3 - (45.0 * norm_sc * conf_f))
    prob_base = max(10.0, 100.0 - (prob_alc + prob_baj))

    tot_p = prob_alc + prob_baj + prob_base
    p_alc_pct = int(round((prob_alc / tot_p) * 100))
    p_base_pct = int(round((prob_base / tot_p) * 100))
    p_baj_pct = 100 - (p_alc_pct + p_base_pct)

    # 6. HORIZONTES TEMPORALES
    horizontes = {
        "⚡ 1-5 días": "🟡 NEUTRAL" if abs(predictive_score) < 30 else ("🟢 ALCISTA" if predictive_score > 0 else "🔴 BAJISTA"),
        "📅 1-4 semanas": "🟢 ALCISTA" if predictive_score >= 40 else ("🔴 BAJISTA" if predictive_score <= -40 else "🟡 NEUTRAL"),
        "📈 1-3 meses": direccion_pred,
        "🗓️ 3-6 meses": "🟢 ALCISTA" if predictive_score >= 20 else ("🔴 BAJISTA" if predictive_score <= -20 else "🟡 NEUTRAL")
    }

    # 7. OBJETIVOS DE PRECIO POR ESCENARIO
    objetivos_escenarios = {}
    if precio and precio > 0:
        vol_ref = (volatilidad / 100.0) if (volatilidad and not np.isnan(volatilidad)) else 0.20
        dcf_fv = datos_val.get("valor_dcf_base") if not es_metal else None

        drift = ((dcf_fv - precio) / precio) if dcf_fv else 0.0

        p_alc_obj = precio * (1.0 + max(0.08, vol_ref * 0.7 + drift * 0.5))
        p_baj_obj = precio * (1.0 - max(0.08, vol_ref * 0.7 - drift * 0.5))
        p_base_obj = precio * (1.0 + (drift * 0.3))

        objetivos_escenarios = {
            "actual": round(precio, 2),
            "alcista": round(p_alc_obj, 2),
            "base": round(p_base_obj, 2),
            "bajista": round(p_baj_obj, 2)
        }

    # 8. SÍNTESIS EXPLICATIVA
    if p_alc_pct >= max(p_base_pct, p_baj_pct):
        escenario_dominante = "Escenario Alcista"
        razon_dominante = f"Prevalecen las señales favorables ({len(senales_pos)} factores positivos) respaldadas por una puntuación de {predictive_score:+.1f}."
    elif p_baj_pct >= max(p_alc_pct, p_base_pct):
        escenario_dominante = "Escenario Bajista"
        razon_dominante = f"Existen riesgos técnicos o fundamentales destacados ({len(senales_neg)} factores negativos) que presionan la cotización."
    else:
        escenario_dominante = "Escenario Base / Consolidador"
        razon_dominante = f"Equilibrio de fuerzas sin un catalizador unidireccional dominante con {datos_utilizados_pct}% de datos disponibles."

    return {
        "score_predictivo": round(predictive_score, 1),
        "direccion": direccion_pred,
        "sesgo": sesgo_pred,
        "confianza": confianza,
        "datos_utilizados_pct": datos_utilizados_pct,
        "probabilidades": {"alcista": p_alc_pct, "base": p_base_pct, "bajista": p_baj_pct},
        "horizontes": horizontes,
        "horizonte_principal": "📈 1-3 meses",
        "objetivos": objetivos_escenarios,
        "senales_pos": senales_pos,
        "senales_neu": senales_neu,
        "senales_neg": senales_neg,
        "escenario_dominante": escenario_dominante,
        "razon_dominante": razon_dominante,
        "registro_backtest": {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "ticker": datos_tec.get("ticker", "N/D"),
            "precio": precio,
            "prediccion": direccion_pred,
            "confianza": confianza
        }
    }

# =========================================================
# MOTOR DE METALES Y FUTUROS
# =========================================================

def escaneo_metales_motor():
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

            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])

            returns = hist['Close'].pct_change().dropna()
            volatilidad = float(returns.std() * np.sqrt(252) * 100)

            ma20 = float(hist['Close'].rolling(20).mean().iloc[-1])
            ma50 = float(hist['Close'].rolling(50).mean().iloc[-1])
            ma200 = float(hist['Close'].rolling(200).mean().iloc[-1])

            datos_tec = {
                "precio": precio, "ma20": ma20, "ma50": ma50, "ma200": ma200,
                "rsi": rsi, "volatilidad": volatilidad, "variacion": var_pct, "ticker": ticker_symbol
            }

            res_noticias = analizar_noticias(ticker_symbol)
            pred_res = ejecutar_prediction_engine(datos_tec, {}, {}, {}, {}, res_noticias, es_metal=True)

            score = pred_res["score_predictivo"] + 50.0 # Mapeo a escala 0-100

            resultados.append({
                'activo': f"{nombre} ({ticker_symbol})",
                'ticker': ticker_symbol,
                'tipo': '🥇 Metal/Futuro',
                'score': round(score, 1),
                'precio': round(precio, 2),
                'potencial': round(var_pct, 2),
                'confianza': pred_res["confianza"],
                'riesgo': 'Bajo' if rsi < 60 else 'Alto',
                'direccion': pred_res["direccion"],
                'prediccion': pred_res["direccion"],
                'horizonte': pred_res["horizonte_principal"],
                'oportunidad_global': score * 0.7 + pred_res["confianza"] * 0.3
            })
        except Exception:
            continue

    return resultados

# =========================================================
# MOTOR DEL ESCÁNER AUTOMÁTICO DE OPORTUNIDADES
# =========================================================

@st.cache_data(ttl=1800)
def ejecutar_filtro_rapido(universe_tickers):
    candidatas_score = []

    for ticker in universe_tickers:
        try:
            t = yf.Ticker(ticker)
            fast_info = t.fast_info

            precio = fast_info.last_price
            market_cap = fast_info.market_cap

            if precio is None or np.isnan(precio) or precio <= 0:
                continue
            if market_cap is None or market_cap < 1_000_000_000:
                continue

            info = t.info
            pe = info.get("trailingPE") or info.get("forwardPE")
            revenue_growth = info.get("revenueGrowth")
            earnings_growth = info.get("earningsGrowth")

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

    candidatas_score.sort(key=lambda x: x["quick_score"], reverse=True)
    candidatas = [c["ticker"] for c in candidatas_score[:25]]
    return candidatas

def ejecutar_analisis_completo_ticker(ticker_symbol):
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
    
    # Integración con el nuevo motor de predicción
    pred_res = ejecutar_prediction_engine(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas, res_noticias, es_metal=False)

    return {
        "ticker": ticker_symbol,
        "activo": ticker_symbol,
        "tipo": "📈 Acción",
        "nombre": info.get("shortName") or info.get("longName") or ticker_symbol,
        "precio": precio,
        "score": res_score["score_total"],
        "fair_value": valor_dcf_base,
        "potencial": ((valor_dcf_base - precio) / precio * 100) if (valor_dcf_base and precio) else 0.0,
        "potencial_dcf": ((valor_dcf_base - precio) / precio * 100) if (valor_dcf_base and precio) else None,
        "direccion": pred_res["direccion"],
        "prediccion": pred_res["direccion"],
        "sesgo": pred_res["sesgo"],
        "confianza": pred_res["confianza"],
        "horizonte": pred_res["horizonte_principal"],
        "calidad": "Excelente" if res_score["score_total"] >= 75 else "Moderada",
        "explicacion": pred_res["razon_dominante"],
        "veredicto": pred_res["direccion"],
        "riesgo": "Alto" if (volatilidad and volatilidad > 35) else "Medio",
        "oportunidad_global": res_score["score_total"] * 0.7 + pred_res["confianza"] * 0.3,
        "prediction_details": pred_res
    }

# =========================================================
# 🌎 MARKET AI MASTER RANKING
# =========================================================

st.title("🌎 MARKET AI MASTER RANKING")
st.caption("Ranking global de oportunidades integrando el MARKET AI SCORE y la predicción cuantitativa.")

if st.button("🔎 BUSCAR MEJORES OPORTUNIDADES"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🔄 Analizando mercado (Acciones S&P 500 y Metales)...")
    progress_bar.progress(10)

    resultados_globales = []

    universo_sp500 = obtener_universo_sp500()
    candidatas_sp500 = ejecutar_filtro_rapido(universo_sp500)

    for idx, ticker in enumerate(candidatas_sp500):
        res = ejecutar_analisis_completo_ticker(ticker)
        if res:
            resultados_globales.append(res)
        progress_bar.progress(10 + int((idx + 1) / len(candidatas_sp500) * 60))

    status_text.text("🔄 Analizando futuros de metales...")
    res_metales = escaneo_metales_motor()
    resultados_globales.extend(res_metales)

    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    st.success("✅ Análisis completado.")

    resultados_globales.sort(key=lambda x: x.get('oportunidad_global', 0), reverse=True)

    if resultados_globales:
        top1 = resultados_globales[0]
        st.markdown("---")
        st.subheader("🥇 MEJOR OPORTUNIDAD DETECTADA AHORA")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Activo", f"{top1.get('activo', top1.get('ticker'))} ({top1['tipo']})")
        col2.metric("Score", f"{top1['score']}/100")
        col3.metric("Predicción", top1.get('prediccion', top1['direccion']))
        col4.metric("Precio", f"${top1['precio']:.2f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Potencial", f"{top1['potencial']:+.2f}%")
        col6.metric("Confianza", f"{top1['confianza']}%")
        col7.metric("Horizonte", top1.get('horizonte', '📅 Medio plazo'))

        st.info(f"**Motivo Principal:** {top1.get('explicacion', '')}")

        st.markdown("---")
        st.subheader("🏆 TOP MARKET AI MASTER RANKING")

        df_top = pd.DataFrame(resultados_globales)[[
            'activo', 'tipo', 'score', 'prediccion', 'confianza', 
            'potencial', 'riesgo', 'horizonte', 'precio'
        ]]
        df_top.columns = ['Activo', 'Tipo', 'Score', 'Predicción', 'Confianza (%)', 'Potencial (%)', 'Riesgo', 'Horizonte', 'Precio ($)']
        st.dataframe(df_top, use_container_width=True)

st.markdown("---")

# =========================================================
# ANÁLISIS INDIVIDUAL & PREDICCIÓN DETALLADA
# =========================================================

st.title("📈 MARKET AI - Terminal de Análisis y Predicción")

st.sidebar.header("🔍 Búsqueda Individual")
if "ticker_seleccionado" not in st.session_state:
    st.session_state.ticker_seleccionado = "AAPL"

ticker_input = st.sidebar.text_input(
    "Símbolo Ticker (ej: AAPL, MSFT, NVDA, GC=F):", 
    value=st.session_state.ticker_seleccionado
).upper().strip()

if ticker_input:
    info = obtener_info_accion(ticker_input)
    df_hist = obtener_historico(ticker_input)

    es_metal = "=" in ticker_input or ticker_input.endswith("=F")

    if info or df_hist is not None:
        nombre = info.get("longName", ticker_input) if info else ticker_input
        st.subheader(f"{nombre} ({ticker_input})")

        precio_analisis = info.get("currentPrice") or info.get("regularMarketPrice") if info else None
        if (precio_analisis is None) and df_hist is not None and not df_hist.empty:
            precio_analisis = float(df_hist['Close'].iloc[-1])

        ma20 = float(df_hist['MA20'].iloc[-1]) if (df_hist is not None and 'MA20' in df_hist and not pd.isna(df_hist['MA20'].iloc[-1])) else None
        ma50 = float(df_hist['MA50'].iloc[-1]) if (df_hist is not None and 'MA50' in df_hist and not pd.isna(df_hist['MA50'].iloc[-1])) else None
        ma200 = float(df_hist['MA200'].iloc[-1]) if (df_hist is not None and 'MA200' in df_hist and not pd.isna(df_hist['MA200'].iloc[-1])) else None
        rsi = float(df_hist['RSI'].iloc[-1]) if (df_hist is not None and 'RSI' in df_hist and not pd.isna(df_hist['RSI'].iloc[-1])) else None

        variacion, volatilidad = None, None
        if df_hist is not None and len(df_hist) > 1:
            variacion = float(((df_hist['Close'].iloc[-1] - df_hist['Close'].iloc[-2]) / df_hist['Close'].iloc[-2]) * 100)
            returns = df_hist['Close'].pct_change().dropna()
            volatilidad = float(returns.std() * np.sqrt(252) * 100)

        datos_tec = {"precio": precio_analisis, "ma20": ma20, "ma50": ma50, "ma200": ma200, "rsi": rsi, "volatilidad": volatilidad, "variacion": variacion, "ticker": ticker_input}
        datos_val = {"valor_dcf_base": None, "pe": info.get("trailingPE") if info else None}
        datos_fund = {"roe": info.get("returnOnEquity") if info else None, "deuda": info.get("debtToEquity") if info else None, "flujo_caja": info.get("freeCashflow") if info else None}
        datos_crec = {"crecimiento_ingresos": info.get("revenueGrowth") if info else None}
        datos_analistas = {"obj_med": info.get("targetMeanPrice") if info else None}

        res_noticias = analizar_noticias(ticker_input)
        pred_res = ejecutar_prediction_engine(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas, res_noticias, es_metal=es_metal)

        # --- SECCIÓN GRÁFICO HISTÓRICO Y ESCENARIOS ---
        st.header("📈 Gráfico con Escenarios Predictivos")
        if df_hist is not None and not df_hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_hist.index, open=df_hist['Open'], high=df_hist['High'],
                low=df_hist['Low'], close=df_hist['Close'], name='Histórico'
            ))

            # Proyección Visual de Escenarios Futuros
            objs = pred_res.get("objetivos", {})
            if objs and "alcista" in objs:
                last_date = df_hist.index[-1]
                future_date = last_date + timedelta(days=60)

                fig.add_trace(go.Scatter(
                    x=[last_date, future_date], y=[precio_analisis, objs["alcista"]],
                    mode="lines+markers", line=dict(color="green", dash="dash"), name=f"Alcista (${objs['alcista']})"
                ))
                fig.add_trace(go.Scatter(
                    x=[last_date, future_date], y=[precio_analisis, objs["base"]],
                    mode="lines+markers", line=dict(color="orange", dash="dash"), name=f"Base (${objs['base']})"
                ))
                fig.add_trace(go.Scatter(
                    x=[last_date, future_date], y=[precio_analisis, objs["bajista"]],
                    mode="lines+markers", line=dict(color="red", dash="dash"), name=f"Bajista (${objs['bajista']})"
                ))

            fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # --- 🔮 MARKET AI PREDICTION ENGINE DISPLAY ---
        st.divider()
        st.header("🔮 MARKET AI PREDICTION ENGINE")

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Dirección Probable", pred_res["direccion"])
        p_col2.metric("Confianza de Predicción", f"{pred_res['confianza']}%")
        p_col3.metric("Datos Utilizados", f"{pred_res['datos_utilizados_pct']}%")
        p_col4.metric("Score Predictivo", f"{pred_res['score_predictivo']:+.1f}")

        st.subheader("📊 Probabilidades por Escenario")
        pr_c1, pr_c2, pr_c3 = st.columns(3)
        pr_c1.metric("🟢 Escenario Alcista", f"{pred_res['probabilidades']['alcista']}%")
        pr_c2.metric("🟡 Escenario Base", f"{pred_res['probabilidades']['base']}%")
        pr_c3.metric("🔴 Escenario Bajista", f"{pred_res['probabilidades']['bajista']}%")

        st.subheader("⏱️ Análisis por Horizontes Temporales")
        h_cols = st.columns(4)
        idx_h = 0
        for h_nombre, h_dir in pred_res["horizontes"].items():
            with h_cols[idx_h]:
                st.caption(h_nombre)
                st.write(f"**{h_dir}**")
            idx_h += 1

        st.subheader("🧠 ¿Por qué MARKET AI piensa esto?")
        st.write(f"**Síntesis:** {pred_res['razon_dominante']}")

        sig_col1, sig_col2 = st.columns(2)
        with sig_col1:
            st.write("🟢 **Señales Alcistas Detectadas:**")
            for s in pred_res["senales_pos"]:
                st.write(f"- [{s['cat']}] {s['desc']}")
            if not pred_res["senales_pos"]:
                st.caption("No se detectan señales alcistas claras.")

        with sig_col2:
            st.write("🔴 **Señales Bajistas Detectadas:**")
            for s in pred_res["senales_neg"]:
                st.write(f"- [{s['cat']}] {s['desc']}")
            if not pred_res["senales_neg"]:
                st.caption("No se detectan señales bajistas críticas.")
