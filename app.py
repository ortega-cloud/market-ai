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

st.set_page_config(page_title="MARKET AI - Análisis de Acciones", layout="wide", page_icon="📈")

# =========================================================
# CONFIGURACIÓN Y API KEYS
# =========================================================
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# =========================================================
# FUNCIONES AUXILIARES Y OBTENCIÓN DE DATOS
# =========================================================

@st.cache_data(ttl=3600)
def obtener_info_accion(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except Exception as e:
        st.error(f"Error al obtener datos de Yahoo Finance: {e}")
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
# MOTOR DE NOTICIAS Y SENTIMIENTO MARKET AI (INDEPENDIENTE)
# =========================================================

@st.cache_data(ttl=1800)
def analizar_noticias(ticker_symbol):
    """
    Obtiene noticias reales de Yahoo Finance y realiza un análisis de
    sentimiento, detección de catalizadores y riesgos.
    """
    noticias_procesadas = []
    catalizadores_set = set()
    riesgos_set = set()
    
    # Palabras clave para la detección de Catalizadores y Riesgos
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
            # Extracción segura de la estructura de yfinance
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
            
            titulo = item.get("title") or content.get("title") or "Sin título"
            fuente = item.get("publisher") or content.get("provider", {}).get("displayName") or "Fuente Financiera"
            
            url = item.get("link") or content.get("canonicalUrl", {}).get("url") or "#"
            
            # Fecha
            pub_date = item.get("providerPublishTime") or content.get("pubDate")
            if isinstance(pub_date, (int, float)):
                fecha_str = datetime.fromtimestamp(pub_date).strftime("%d/%m/%Y %H:%M")
            else:
                fecha_str = "Reciente"

            resumen = item.get("summary") or content.get("summary") or titulo

            # Cálculo de polaridad del título y resumen
            texto_completo = f"{titulo}. {resumen}"
            
            if TextBlob is not None:
                blob = TextBlob(texto_completo)
                polarity = blob.sentiment.polarity  # Valor entre -1.0 y 1.0
            else:
                # Fallback de análisis por vocabulario básico si TextBlob no está disponible
                polarity = 0.0
                words = re.findall(r'\w+', texto_completo.lower())
                pos_w = sum(1 for w in words if w in ['beat', 'growth', 'up', 'high', 'gain', 'buy', 'positive'])
                neg_w = sum(1 for w in words if w in ['miss', 'fall', 'down', 'low', 'drop', 'sell', 'negative'])
                if pos_w + neg_w > 0:
                    polarity = (pos_w - neg_w) / (pos_w + neg_w)

            scores_polaridad.append(polarity)

            # Clasificación del Sentimiento
            if polarity > 0.1:
                sentimiento_str = "🟢 Positivo"
                pos_count += 1
            elif polarity < -0.1:
                sentimiento_str = "🔴 Negativo"
                neg_count += 1
            else:
                sentimiento_str = "🟡 Neutral"

            # Clasificación de Importancia
            longitud = len(texto_completo)
            abs_pol = abs(polarity)
            if abs_pol > 0.35 or "breaking" in titulo.lower():
                importancia_str = "🔥 Alta"
            elif abs_pol > 0.15:
                importancia_str = "🟠 Media"
            else:
                importancia_str = "⚪ Baja"

            # Detección de Catalizadores y Riesgos
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

        # Sentimiento General Normalizado (-100 a +100)
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
# MOTOR CENTRAL DE DIAGNÓSTICO MARKET AI
# =========================================================

def calcular_market_ai_score(datos_tec, datos_val, datos_fund, datos_crec, datos_analistas):
    pts_tec, pts_val, pts_fund, pts_crec, pts_riesgo = 0.0, 0.0, 0.0, 0.0, 0.0
    pos, neg, riesgos, catalizadores = [], [], [], []

    # 1. ANÁLISIS TÉCNICO (MÁX 25 PTS)
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

    # 2. VALORACIÓN (MÁX 25 PTS)
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

    # 3. FUNDAMENTALES (MÁX 25 PTS)
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

    # 4. CRECIMIENTO (MÁX 15 PTS)
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

    # 5. RIESGO (MÁX 10 PTS)
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

    # SCORE FINAL
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
# INTERFAZ DE USUARIO EN STREAMLIT
# =========================================================

st.title("📈 MARKET AI - Terminal de Análisis Financiero")

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

st.sidebar.header("🔍 Búsqueda de Activo")
ticker_input = st.sidebar.text_input(
    "Símbolo Ticker (ej: AAPL, MSFT, NVDA, ^GSPC):", 
    value=st.session_state.ticker_seleccionado
).upper().strip()

if ticker_input:
    info = obtener_info_accion(ticker_input)
    
    if info:
        ticker_obj = yf.Ticker(ticker_input)
        
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

        # Integración con dcf.py
        escenarios_dcf = None
        valor_dcf_base = None

        if dcf is not None:
            try:
                free_cash_flow = info.get("freeCashflow")
                deuda_dcf = info.get("totalDebt") or 0.0
                
                caja_dcf = info.get("totalCash")
                if caja_dcf is None:
                    caja_dcf = info.get("cashAndCashEquivalents") or 0.0
                
                acciones_dcf = info.get("sharesOutstanding")

                faltantes = []
                if free_cash_flow is None or free_cash_flow <= 0:
                    faltantes.append("Free Cash Flow (FCF) no disponible o <= 0")
                if acciones_dcf is None or acciones_dcf <= 0:
                    faltantes.append("Acciones en circulación (sharesOutstanding) no disponibles o <= 0")

                if not faltantes:
                    escenarios_dcf = dcf.calcular_escenarios_dcf(
                        free_cash_flow=float(free_cash_flow),
                        deuda=float(deuda_dcf),
                        caja=float(caja_dcf),
                        acciones=float(acciones_dcf)
                    )

                    if escenarios_dcf and isinstance(escenarios_dcf, dict) and "base" in escenarios_dcf:
                        esc_base = escenarios_dcf["base"]
                        candidato_base = None
                        if isinstance(esc_base, dict):
                            candidato_base = esc_base.get("valor_por_accion")
                        elif isinstance(esc_base, (int, float)):
                            candidato_base = esc_base

                        if (candidato_base is not None 
                                and isinstance(candidato_base, (int, float)) 
                                and np.isfinite(candidato_base) 
                                and candidato_base > 0):
                            valor_dcf_base = float(candidato_base)
                else:
                    st.warning(f"⚠️ **DCF no disponible para {ticker_input}:** " + " | ".join(faltantes))

            except Exception as e:
                st.error(f"⚠️ **Error en el cálculo del DCF:** {str(e)}")
                escenarios_dcf = None
                valor_dcf_base = None

        datos_tec_in = {
            "precio": precio_analisis, "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "rsi": rsi, "volatilidad": volatilidad, "variacion": variacion
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

        res_market_ai = calcular_market_ai_score(datos_tec_in, datos_val_in, datos_fund_in, datos_crec_in, datos_analistas_in)
        diag = res_market_ai["diagnostico"]

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

        st.subheader("🧠 DIAGNÓSTICO DE MARKET AI")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Precio Actual", f"${diag['precio']:,.2f}" if diag['precio'] else "N/D")
        m_col2.metric("Fair Value DCF", f"${diag['fair_value']:,.2f}" if diag['fair_value'] else "N/D")
        m_col3.metric("Objetivo Analistas", f"${diag['target_analistas']:,.2f}" if diag['target_analistas'] else "N/D")
        m_col4.metric("Potencial DCF", f"{diag['potencial_dcf']:+.1f}%" if diag['potencial_dcf'] is not None else "N/D")
        m_col5.metric("Potencial Analistas", f"{diag['potencial_analistas']:+.1f}%" if diag['potencial_analistas'] is not None else "N/D")

        st.info(f"**Tendencia:** {diag['tendencia']} &nbsp;|&nbsp; **Valoración:** {diag['estado_valoracion']} &nbsp;|&nbsp; **Fundamentales:** {diag['calidad_fundamentales']} &nbsp;|&nbsp; **Crecimiento:** {diag['nivel_crecimiento']}")

        st.write(f"MARKET AI evalúa la compañía con una puntuación global de **{res_market_ai['score_total']:.1f}/100** ({res_market_ai['categoria']}), reflejando un perfil con **{diag['calidad_fundamentales'].lower()}** fundamentales y una orientación de **{diag['estado_valoracion'].lower()}**.")

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.write("🟢 **PUNTOS POSITIVOS**")
            for item in diag["positivos"]:
                st.write(f"- {item}")
                
            st.write("🚀 **CATALIZADORES**")
            for item in diag["catalizadores"]:
                st.write(f"- {item}")

        with diag_col2:
            st.write("🔴 **PUNTOS NEGATIVOS**")
            for item in diag["negativos"]:
                st.write(f"- {item}")

            st.write("⚠️ **RIESGOS**")
            for item in diag["riesgos"]:
                st.write(f"- {item}")

        st.divider()

        # Gráfico Histórico
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

        # Sección Fundamentales
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

        # Sección Crecimiento
        st.header("🚀 Crecimiento")
        g_c1, g_c2, g_c3 = st.columns(3)
        g_c1.metric("Crecimiento Ingresos", f"{crecimiento_ingresos*100:+.2f}%" if crecimiento_ingresos else "N/D")
        g_c2.metric("Crecimiento Beneficios", f"{crecimiento_beneficios*100:+.2f}%" if crecimiento_beneficios else "N/D")
        g_c3.metric("Dividend Yield", f"{info.get('dividendYield')*100:.2f}%" if info.get('dividendYield') else "N/D")

        # Sección DCF
        if escenarios_dcf and isinstance(escenarios_dcf, dict):
            st.header("🧮 Valoración DCF (Descuento de Flujos de Caja)")
            dcf_cols = st.columns(len(escenarios_dcf))
            idx = 0
            for nombre_esc, datos_esc in escenarios_dcf.items():
                with dcf_cols[idx]:
                    st.subheader(f"Caso {str(nombre_esc).capitalize()}")
                    
                    val_accion = None
                    if isinstance(datos_esc, dict):
                        val_accion = datos_esc.get('valor_por_accion')
                    elif isinstance(datos_esc, (int, float)):
                        val_accion = datos_esc

                    if val_accion is not None and isinstance(val_accion, (int, float)) and np.isfinite(val_accion) and val_accion > 0:
                        st.metric("Fair Value", f"${val_accion:,.2f}")
                        pot = ((val_accion - precio_analisis) / precio_analisis) * 100 if precio_analisis else None
                        st.metric("Potencial", f"{pot:+.1f}%" if pot is not None else "N/D")
                    else:
                        st.metric("Fair Value", "N/D")
                        st.metric("Potencial", "N/D")
                idx += 1

        # Sección Analistas
        st.header("🎯 Analistas")
        a_c1, a_c2, a_c3, a_c4 = st.columns(4)
        a_c1.metric("Objetivo Mínimo", f"${objetivos.get('low'):,.2f}" if objetivos.get('low') else "N/D")
        a_c2.metric("Objetivo Medio", f"${objetivos.get('mean'):,.2f}" if objetivos.get('mean') else "N/D")
        a_c3.metric("Objetivo Máximo", f"${objetivos.get('high'):,.2f}" if objetivos.get('high') else "N/D")
        a_c4.metric("Mediana", f"${objetivos.get('median'):,.2f}" if objetivos.get('median') else "N/D")

        st.subheader("🧑‍💼 Consenso de Analistas")
        if "consensus" in objetivos:
            st.info(f"**Recomendación de Consenso:** {objetivos.get('consensus')} (Basado en {objetivos.get('num_analistas', 'N/D')} opiniones)")

        st.subheader("🔄 Cambios Recientes de Analistas")
        rec_df = obtener_recomendaciones_analistas(ticker_input)
        if rec_df is not None and not rec_df.empty:
            st.dataframe(rec_df.tail(10), use_container_width=True)
        else:
            st.write("No hay registros recientes de revisiones disponibles.")

        st.subheader("🔮 Estimaciones de Analistas")
        eps_df = obtener_estimaciones_eps(ticker_input)
        if eps_df is not None and not eps_df.empty:
            st.dataframe(eps_df, use_container_width=True)
        else:
            st.write("No hay estimaciones de EPS disponibles.")

        # =========================================================
        # NUEVA SECCIÓN: NOTICIAS Y SENTIMIENTO MARKET AI
        # =========================================================
        st.divider()
        st.header("📰 NOTICIAS Y SENTIMIENTO MARKET AI")
        
        data_noticias = analizar_noticias(ticker_input)
        
        # 1. Metricas Principales de Sentimiento
        ns_col1, ns_col2 = st.columns(2)
        with ns_col1:
            st.metric("Sentimiento General", data_noticias["sentimiento_general"])
        with ns_col2:
            st.metric("Puntuación de Sentimiento", f"{data_noticias['puntuacion_sentimiento']} / 100")
            
        st.write(f"*{data_noticias['resumen']}*")
        st.write("")

        # 2. Catalizadores y Riesgos Detectados en Noticias
        cat_col, ries_col = st.columns(2)
        with cat_col:
            st.subheader("🚀 CATALIZADORES")
            if data_noticias["catalizadores"]:
                for cat in data_noticias["catalizadores"]:
                    st.write(f"- {cat}")
            else:
                st.write("- No hay información reciente suficiente para determinar catalizadores específicos.")

        with ries_col:
            st.subheader("⚠️ RIESGOS DETECTADOS")
            if data_noticias["riesgos"]:
                for rsg in data_noticias["riesgos"]:
                    st.write(f"- {rsg}")
            else:
                st.write("- No hay información reciente suficiente para determinar riesgos específicos.")

        st.divider()

        # 3. Listado de Noticias Recientes
        st.subheader("📰 NOTICIAS RECIENTES")
        
        listado_noticias = data_noticias.get("noticias", [])
        if listado_noticias:
            for item in listado_noticias:
                n_col1, n_col2 = st.columns([4, 1])
                with n_col1:
                    if item['url'] != "#":
                        st.markdown(f"### [{item['titulo']}]({item['url']})")
                    else:
                        st.markdown(f"### {item['titulo']}")
                    
                    st.caption(f"**Fuente:** {item['fuente']} | **Fecha:** {item['fecha']}")
                    st.write(item['resumen'])
                
                with n_col2:
                    st.write(f"**Sentimiento:** {item['sentimiento']}")
                    st.write(f"**Importancia:** {item['importancia']}")
                
                st.write("---")
        else:
            st.info("No hay información reciente suficiente.")

    else:
        st.error("No se pudo cargar la información para el ticker introducido. Verifica el símbolo.")
