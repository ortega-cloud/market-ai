import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

from dcf import (
    calcular_escenarios_dcf,
    diagnosticar_valoracion
)

# =========================================================
# CONFIGURACIÓN DE PÁGINA Y SESIÓN YFINANCE (EVITAR BLOQUEOS)
# =========================================================

st.set_page_config(
    page_title="MARKET AI",
    page_icon="📈",
    layout="wide"
)

# Configurar headers para que Yahoo Finance no bloquee las peticiones
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

st.title("🤖 MARKET AI")
st.caption("Sistema experimental de análisis técnico, fundamentales, valoración, analistas y mercado.")

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def limpiar_numero(valor):
    try:
        if valor is None or pd.isna(valor):
            return None
        return float(valor)
    except Exception:
        return None

def obtener_valor(diccionario, claves):
    if not isinstance(diccionario, dict):
        return None
    for clave in claves:
        valor = diccionario.get(clave)
        if valor is not None:
            return valor
    return None

# =========================================================
# ALPHA VANTAGE
# =========================================================

@st.cache_data(ttl=900)
def obtener_precio_alpha_vantage(ticker):
    try:
        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    except Exception:
        return None, "No se ha encontrado ALPHA_VANTAGE_API_KEY en los Secrets de Streamlit."

    try:
        url = "https://www.alphavantage.co/query"
        parametros = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": api_key
        }
        respuesta = requests.get(url, params=parametros, timeout=15)
        respuesta.raise_for_status()
        datos = respuesta.json()

        if "Note" in datos:
            return None, "Alpha Vantage ha alcanzado el límite de peticiones."
        if "Information" in datos:
            return None, datos["Information"]
        if "Error Message" in datos:
            return None, "El ticker no es válido."

        quote = datos.get("Global Quote", {})
        precio = quote.get("05. price")
        fecha = quote.get("07. latest trading day")

        if not precio:
            return None, "Alpha Vantage no ha devuelto el precio."

        return {"precio": float(precio), "fecha": fecha}, None

    except Exception as error:
        return None, str(error)

# =========================================================
# HISTÓRICOS Y DATOS ROBUSTOS
# =========================================================

@st.cache_data(ttl=900)
def obtener_precios(ticker, periodo):
    try:
        datos = yf.download(ticker, period=periodo, auto_adjust=False, progress=False, session=session)
        if datos is None or datos.empty:
            return pd.DataFrame()

        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.get_level_values(0)

        columnas = ["Open", "High", "Low", "Close", "Volume"]
        disponibles = [columna for columna in columnas if columna in datos.columns]

        datos = datos[disponibles].copy()
        return datos.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_info(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.info
        if datos and isinstance(datos, dict) and len(datos) > 5:
            return datos
    except Exception:
        pass
    
    # Fallback usando fast_info si .info viene vacio por bloqueo
    try:
        empresa = yf.Ticker(ticker, session=session)
        fast = empresa.fast_info
        return {
            "symbol": ticker,
            "longName": ticker,
            "lastPrice": fast.get("lastPrice"),
            "marketCap": fast.get("marketCap"),
            "sharesOutstanding": fast.get("shares"),
            "yearHigh": fast.get("yearHigh"),
            "yearLow": fast.get("yearLow")
        }
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def obtener_objetivos_analistas(ticker, info_dict=None):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.analyst_price_targets
        if datos is not None:
            if hasattr(datos, "to_dict"):
                datos = datos.to_dict()
            if isinstance(datos, dict) and any(v is not None for v in datos.values()):
                return datos
    except Exception:
        pass

    info = info_dict or obtener_info(ticker)
    return {
        "low": info.get("targetLowPrice"),
        "mean": info.get("targetMeanPrice"),
        "median": info.get("targetMedianPrice"),
        "high": info.get("targetHighPrice")
    }

@st.cache_data(ttl=3600)
def obtener_recomendaciones(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.recommendations_summary
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_historial_recomendaciones(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.recommendations
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_upgrades_downgrades(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.upgrades_downgrades
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_estimaciones_eps(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.earnings_estimate
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_estimaciones_ingresos(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.revenue_estimate
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_revisiones_eps(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.eps_revisions
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_tendencia_eps(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.eps_trend
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def obtener_crecimiento_estimado(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        datos = empresa.growth_estimates
        return datos if datos is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def obtener_noticias(ticker):
    try:
        empresa = yf.Ticker(ticker, session=session)
        noticias = empresa.news
        return noticias if noticias is not None else []
    except Exception:
        return []

# =========================================================
# INDICADORES TÉCNICOS Y SCORES
# =========================================================

def calcular_rsi(precios, periodo=14):
    delta = precios.diff()
    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)
    media_ganancias = ganancias.rolling(periodo).mean()
    media_perdidas = perdidas.rolling(periodo).mean()
    rs = media_ganancias / media_perdidas.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calcular_score_tecnico(precio, ma20, ma50, ma200, rsi):
    score = 0
    razones = []
    if pd.notna(ma20):
        if precio > ma20: score += 5; razones.append("El precio está por encima de la MA20.")
        else: razones.append("El precio está por debajo de la MA20.")
    if pd.notna(ma20) and pd.notna(ma50):
        if ma20 > ma50: score += 5; razones.append("La MA20 está por encima de la MA50.")
        else: razones.append("La MA20 está por debajo de la MA50.")
    if pd.notna(ma50) and pd.notna(ma200):
        if ma50 > ma200: score += 7; razones.append("La MA50 está por encima de la MA200.")
        else: razones.append("La MA50 está por debajo de la MA200.")
    if pd.notna(rsi):
        if rsi < 30: score += 6; razones.append("El RSI indica posible sobreventa.")
        elif rsi <= 65: score += 8; razones.append("El RSI se encuentra en una zona equilibrada.")
        elif rsi <= 70: score += 5; razones.append("El RSI se aproxima a sobrecompra.")
        else: score += 2; razones.append("El RSI indica posible sobrecompra.")
    return min(score, 25), razones

def calcular_score_valoracion(pe, forward_pe, peg, price_to_book):
    score = 0
    razones = []
    if pe is not None:
        if pe < 15: score += 8; razones.append("PER relativamente bajo.")
        elif pe < 25: score += 6; razones.append("PER en zona intermedia.")
        elif pe < 40: score += 3; razones.append("PER relativamente elevado.")
        else: score += 1; razones.append("PER muy elevado.")
    if forward_pe is not None:
        if forward_pe < 15: score += 7
        elif forward_pe < 25: score += 5
        elif forward_pe < 40: score += 2
    if peg is not None:
        if peg < 1: score += 6
        elif peg < 2: score += 4
        else: score += 1
    if price_to_book is not None:
        if price_to_book < 2: score += 4
        elif price_to_book < 5: score += 2
        else: score += 1
    return min(score, 25), razones

def calcular_score_fundamentales(roe, margen, deuda, flujo_caja):
    score = 0
    razones = []
    if roe is not None:
        if roe > 0.20: score += 8; razones.append("ROE elevado.")
        elif roe > 0.10: score += 5; razones.append("ROE razonable.")
        else: score += 2; razones.append("ROE relativamente bajo.")
    if margen is not None:
        if margen > 0.20: score += 7; razones.append("Margen de beneficios elevado.")
        elif margen > 0.10: score += 5; razones.append("Margen de beneficios razonable.")
        else: score += 2; razones.append("Margen de beneficios reducido.")
    if deuda is not None:
        if deuda < 50: score += 5; razones.append("Deuda relativamente baja.")
        elif deuda < 100: score += 3; razones.append("Deuda moderada.")
        else: score += 1; razones.append("Deuda elevada.")
    if flujo_caja is not None:
        if flujo_caja > 0: score += 5; razones.append("Free Cash Flow positivo.")
        else: razones.append("Free Cash Flow negativo.")
    return min(score, 25), razones

def calcular_score_crecimiento(crecimiento_ingresos, crecimiento_beneficios):
    score = 0
    razones = []
    if crecimiento_ingresos is not None:
        if crecimiento_ingresos > 0.15: score += 7
        elif crecimiento_ingresos > 0.05: score += 5
        elif crecimiento_ingresos > 0: score += 2
        else: razones.append("Los ingresos están decreciendo.")
    if crecimiento_beneficios is not None:
        if crecimiento_beneficios > 0.15: score += 8
        elif crecimiento_beneficios > 0.05: score += 5
        elif crecimiento_beneficios > 0: score += 2
        else: razones.append("Los beneficios están decreciendo.")
    return min(score, 15), razones

def calcular_score_riesgo(volatilidad, deuda):
    score = 0
    razones = []
    if volatilidad < 20: score += 6
    elif volatilidad < 35: score += 4
    elif volatilidad < 50: score += 2
    else: score += 1; razones.append("Volatilidad muy elevada.")
    if deuda is not None:
        if deuda < 50: score += 4
        elif deuda < 100: score += 2
        else: score += 1
    else: score += 2
    return min(score, 10), razones

# =========================================================
# VALORACIÓN COMBINADA & PREDICTIVO
# =========================================================

def calcular_fair_value_combinado(precio_actual, valor_dcf, objetivo_analistas, eps, crecimiento_beneficios):
    valores = []
    pesos = []
    if valor_dcf is not None and valor_dcf > 0:
        valores.append(float(valor_dcf))
        pesos.append(0.50)
    if objetivo_analistas is not None and objetivo_analistas > 0:
        valores.append(float(objetivo_analistas))
        pesos.append(0.25)
    if eps is not None and eps > 0:
        crecimiento_pct = float(crecimiento_beneficios) * 100 if crecimiento_beneficios is not None else 5.0
        crecimiento_pct = max(-5.0, min(25.0, crecimiento_pct))
        pe_razonable = max(10.0, min(30.0, 15.0 + (0.50 * crecimiento_pct)))
        valor_multiplo = eps * pe_razonable
        if valor_multiplo > 0:
            valores.append(float(valor_multiplo))
            pesos.append(0.25)
    if not valores:
        return None
    return sum(v * p for v, p in zip(valores, pesos)) / sum(pesos)

def diagnosticar_fair_value(precio_actual, fair_value):
    if precio_actual is None or fair_value is None or precio_actual <= 0:
        return "⚪ SIN DATOS", None
    potencial = ((fair_value - precio_actual) / precio_actual) * 100
    if potencial >= 30: return "🟢 MUY INFRAVALORADA", potencial
    if potencial >= 15: return "🟢 INFRAVALORADA", potencial
    if potencial >= -10: return "🟡 VALORACIÓN RAZONABLE", potencial
    if potencial >= -25: return "🟠 SOBREVALORADA", potencial
    return "🔴 MUY SOBREVALORADA", potencial

def calcular_diagnostico_predictivo(precio, fair_value, score_total, score_tecnico, score_valoracion, score_fundamentales, score_crecimiento, score_riesgo, rsi, ma20, ma50, ma200, objetivo_analistas=None):
    señales, riesgos = [], []
    potencial_fair = None
    if precio is not None and fair_value is not None and precio > 0:
        potencial_fair = ((fair_value - precio) / precio) * 100
        if potencial_fair >= 20: señales.append("La acción presenta un descuento importante respecto al Fair Value.")
        elif potencial_fair >= 10: señales.append("La acción presenta un descuento moderado respecto al Fair Value.")
        elif potencial_fair <= -20: riesgos.append("El precio está muy por encima del Fair Value estimado.")
        elif potencial_fair <= -10: riesgos.append("El precio está por encima del Fair Value estimado.")

    tendencia = "LATERAL"
    if ma20 is not None and ma50 is not None and ma200 is not None:
        if precio > ma20 and ma20 > ma50 and ma50 > ma200:
            tendencia = "ALCISTA"
            señales.append("La estructura de medias móviles confirma una tendencia alcista.")
        elif precio < ma20 and ma20 < ma50 and ma50 < ma200:
            tendencia = "BAJISTA"
            riesgos.append("Las medias móviles muestran una estructura bajista.")

    situacion_rsi = "NEUTRAL"
    if rsi is not None:
        if rsi >= 75: situacion_rsi = "SOBRECOMPRA"; riesgos.append("El RSI indica una situación de sobrecompra.")
        elif rsi <= 30: situacion_rsi = "SOBREVENTA"; señales.append("El RSI indica una situación de sobreventa.")
        elif 45 <= rsi <= 70: situacion_rsi = "SALUDABLE"

    if score_fundamentales >= 20: señales.append("Los fundamentales son sólidos.")
    elif score_fundamentales < 12: riesgos.append("Los fundamentales presentan varias señales débiles.")
    if score_crecimiento >= 12: señales.append("El crecimiento de ingresos y beneficios es favorable.")
    elif score_crecimiento < 7: riesgos.append("El crecimiento de la empresa es limitado o débil.")

    potencial_analistas = ((objetivo_analistas - precio) / precio) * 100 if objetivo_analistas and precio else None

    puntos_alcistas = (3 if tendencia == "ALCISTA" else 0) + (3 if potencial_fair and potencial_fair >= 15 else 0) + (2 if score_fundamentales >= 20 else 0)
    puntos_bajistas = (3 if tendencia == "BAJISTA" else 0) + (3 if potencial_fair and potencial_fair <= -15 else 0) + (2 if score_fundamentales < 12 else 0)

    if puntos_alcistas >= puntos_bajistas + 3: direccion = "🟢 ALCISTA"
    elif puntos_bajistas >= puntos_alcistas + 3: direccion = "🔴 BAJISTA"
    else: direccion = "🟡 LATERAL / INCIERTA"

    if score_total >= 85: señal = "🟢 COMPRA"
    elif score_total >= 70: señal = "🟢 COMPRA MODERADA"
    elif score_total >= 55: señal = "🟡 MANTENER"
    elif score_total >= 40: señal = "🟠 ESPERAR"
    else: señal = "🔴 EVITAR"

    return {
        "direccion": direccion, "tendencia": tendencia, "rsi": situacion_rsi,
        "horizonte": "3–12 meses", "señal": señal, "potencial_fair": potencial_fair,
        "potencial_analistas": potencial_analistas, "señales": señales, "riesgos": riesgos,
        "puntos_alcistas": puntos_alcistas, "puntos_bajistas": puntos_bajistas
    }

# =========================================================
# INTERFAZ (SIDEBAR)
# =========================================================

st.sidebar.header("⚙️ Configuración")
ticker = st.sidebar.text_input("Símbolo", value="NVDA").upper().strip()
periodo = st.sidebar.selectbox("Periodo", ["6mo", "1y", "2y", "5y", "10y"], index=1)

st.sidebar.divider()
st.sidebar.subheader("🎯 Precio para el análisis")
tipo_precio = st.sidebar.radio("Fuente", ["Precio automático", "Precio personalizado"])
precio_personalizado = st.sidebar.number_input("Precio de entrada", min_value=0.01, value=100.00, step=1.00) if tipo_precio == "Precio personalizado" else None

analizar = st.sidebar.button("📊 ANALIZAR", type="primary")

# =========================================================
# ANÁLISIS DE TICKER
# =========================================================

if analizar:
    if not ticker:
        st.error("Debes introducir un ticker.")
        st.stop()

    with st.spinner("Obteniendo datos del mercado..."):
        precio_alpha, error_alpha = obtener_precio_alpha_vantage(ticker)
        datos = obtener_precios(ticker, periodo)
        info = obtener_info(ticker)
        objetivos = obtener_objetivos_analistas(ticker, info_dict=info)
        recomendaciones = obtener_recomendaciones(ticker)
        upgrades = obtener_upgrades_downgrades(ticker)
        eps_estimaciones = obtener_estimaciones_eps(ticker)
        ingresos_estimaciones = obtener_estimaciones_ingresos(ticker)
        revisiones_eps = obtener_revisiones_eps(ticker)
        tendencia_eps = obtener_tendencia_eps(ticker)
        crecimiento_estimado = obtener_crecimiento_estimado(ticker)
        noticias = obtener_noticias(ticker)

    if datos.empty:
        st.error(f"No se han encontrado datos históricos para {ticker}.")
        st.stop()

    precio_mercado = precio_alpha["precio"] if precio_alpha else None
    precio_historico = float(datos["Close"].iloc[-1])
    precio_anterior = float(datos["Close"].iloc[-2]) if len(datos) > 1 else precio_historico
    variacion = ((precio_historico - precio_anterior) / precio_anterior) * 100
    precio_analisis = precio_personalizado if tipo_precio == "Precio personalizado" else (precio_mercado or precio_historico)

    # Indicadores Técnicos
    datos["MA20"] = datos["Close"].rolling(20).mean()
    datos["MA50"] = datos["Close"].rolling(50).mean()
    datos["MA200"] = datos["Close"].rolling(200).mean()
    ma20, ma50, ma200 = datos["MA20"].iloc[-1], datos["MA50"].iloc[-1], datos["MA200"].iloc[-1]
    datos["RSI"] = calcular_rsi(datos["Close"])
    rsi = float(datos["RSI"].iloc[-1])
    volatilidad = datos["Close"].pct_change().std() * np.sqrt(252) * 100

    # Extraer variables con Múltiples Respaldos
    empresa_obj = yf.Ticker(ticker, session=session)
    fast_info = getattr(empresa_obj, "fast_info", {})

    nombre = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector", "N/D")
    industria = info.get("industry", "N/D")
    
    pe = limpiar_numero(info.get("trailingPE"))
    forward_pe = limpiar_numero(info.get("forwardPE"))
    peg = limpiar_numero(info.get("pegRatio"))
    price_to_book = limpiar_numero(info.get("priceToBook"))
    roe = limpiar_numero(info.get("returnOnEquity"))
    margen = limpiar_numero(info.get("profitMargins"))
    margen_operativo = limpiar_numero(info.get("operatingMargins"))
    deuda = limpiar_numero(info.get("debtToEquity"))
    deuda_total = limpiar_numero(info.get("totalDebt"))
    caja_total = limpiar_numero(info.get("totalCash"))
    
    acciones_en_circulacion = limpiar_numero(info.get("sharesOutstanding")) or limpiar_numero(fast_info.get("shares"))
    ingresos = limpiar_numero(info.get("totalRevenue"))
    beneficio = limpiar_numero(info.get("netIncomeToCommon"))
    eps = limpiar_numero(info.get("trailingEps"))
    crecimiento_ingresos = limpiar_numero(info.get("revenueGrowth"))
    crecimiento_beneficios = limpiar_numero(info.get("earningsGrowth"))
    dividend_yield = limpiar_numero(info.get("dividendYield"))

    # Extraer Free Cash Flow directo o desde el Estado Financiero (Cashflow)
    flujo_caja = limpiar_numero(info.get("freeCashflow"))
    if flujo_caja is None:
        try:
            cf_df = empresa_obj.cashflow
            if cf_df is not None and not cf_df.empty:
                if "Free Cash Flow" in cf_df.index:
                    flujo_caja = float(cf_df.loc["Free Cash Flow"].iloc[0])
                elif "Operating Cash Flow" in cf_df.index and "Capital Expenditure" in cf_df.index:
                    flujo_caja = float(cf_df.loc["Operating Cash Flow"].iloc[0]) + float(cf_df.loc["Capital Expenditure"].iloc[0])
        except Exception:
            flujo_caja = None

    # Scores
    score_tecnico, razones_tecnico = calcular_score_tecnico(precio_historico, ma20, ma50, ma200, rsi)
    score_valoracion, razones_valoracion = calcular_score_valoracion(pe, forward_pe, peg, price_to_book)
    score_fundamentales, razones_fundamentales = calcular_score_fundamentales(roe, margen, deuda, flujo_caja)
    score_crecimiento, razones_crecimiento = calcular_score_crecimiento(crecimiento_ingresos, crecimiento_beneficios)
    score_riesgo, razones_riesgo = calcular_score_riesgo(volatilidad, deuda)
    score_total = score_tecnico + score_valoracion + score_fundamentales + score_crecimiento + score_riesgo

    # Renderizado UI
    st.success(f"✅ {nombre}")
    st.write(f"**Sector:** {sector}  |  **Industria:** {industria}")

    st.divider()
    st.header("💵 Situación de mercado")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precio", f"${precio_historico:,.2f}", f"{variacion:+.2f}%")
    with c2: st.metric("Máximo periodo", f"${datos['High'].max():,.2f}")
    with c3: st.metric("Mínimo periodo", f"${datos['Low'].min():,.2f}")
    with c4: st.metric("RSI", f"{rsi:.1f}")

    st.divider()
    st.header("🎯 MARKET AI SCORE")
    st.metric("Puntuación Total", f"{score_total}/100")
    st.progress(score_total / 100)

    st.divider()
    st.header("📊 Fundamentales")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("PER", f"{pe:.2f}" if pe else "N/D")
    with c2: st.metric("PER futuro", f"{forward_pe:.2f}" if forward_pe else "N/D")
    with c3: st.metric("PEG", f"{peg:.2f}" if peg else "N/D")
    with c4: st.metric("Price/Book", f"{price_to_book:.2f}" if price_to_book else "N/D")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ROE", f"{roe * 100:.2f}%" if roe else "N/D")
    with c2: st.metric("Margen beneficio", f"{margen * 100:.2f}%" if margen else "N/D")
    with c3: st.metric("Margen operativo", f"{margen_operativo * 100:.2f}%" if margen_operativo else "N/D")
    with c4: st.metric("Deuda/Patrimonio", f"{deuda:.1f}" if deuda else "N/D")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Free Cash Flow", f"${flujo_caja / 1e9:.2f} B" if flujo_caja else "N/D")
    with c2: st.metric("Ingresos", f"${ingresos / 1e9:.2f} B" if ingresos else "N/D")
    with c3: st.metric("Beneficio neto", f"${beneficio / 1e9:.2f} B" if beneficio else "N/D")
    with c4: st.metric("EPS", f"${eps:.2f}" if eps else "N/D")

    st.divider()
    st.header("🚀 Crecimiento")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Crecimiento ingresos", f"{crecimiento_ingresos * 100:.2f}%" if crecimiento_ingresos else "N/D")
    with c2: st.metric("Crecimiento beneficios", f"{crecimiento_beneficios * 100:.2f}%" if crecimiento_beneficios else "N/D")
    with c3: st.metric("Dividend Yield", f"{dividend_yield * 100:.2f}%" if dividend_yield else "N/D")

    st.divider()
    st.header("🎯 Analistas")
    objetivo_bajo = limpiar_numero(obtener_valor(objetivos, ["low"]))
    objetivo_medio = limpiar_numero(obtener_valor(objetivos, ["mean"]))
    objetivo_mediano = limpiar_numero(obtener_valor(objetivos, ["median"]))
    objetivo_alto = limpiar_numero(obtener_valor(objetivos, ["high"]))

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Objetivo bajo", f"${objetivo_bajo:,.2f}" if objetivo_bajo else "N/D")
    with c2: st.metric("Objetivo medio", f"${objetivo_medio:,.2f}" if objetivo_medio else "N/D")
    with c3: st.metric("Mediana", f"${objetivo_mediano:,.2f}" if objetivo_mediano else "N/D")
    with c4: st.metric("Objetivo alto", f"${objetivo_alto:,.2f}" if objetivo_alto else "N/D")

    # Consenso Buy / Sell
    st.subheader("🧑‍💼 Consenso de analistas")
    if not recomendaciones.empty:
        st.dataframe(recomendaciones, use_container_width=True)
    else:
        st.info("No hay resumen de recomendaciones disponible.")

    # Upgrades / Downgrades
    st.subheader("🔄 Cambios recientes de analistas")
    if not upgrades.empty:
        st.dataframe(upgrades.tail(10).reset_index(), use_container_width=True, hide_index=True)
    else:
        st.info("No hay cambios recientes disponibles.")

    # Estimaciones
    st.divider()
    st.header("🔮 Estimaciones de analistas")
    if not eps_estimaciones.empty:
        st.subheader("EPS Estimados")
        st.dataframe(eps_estimaciones, use_container_width=True)
    if not ingresos_estimaciones.empty:
        st.subheader("Ingresos Estimados")
        st.dataframe(ingresos_estimaciones, use_container_width=True)

    # DCF y Fair Value
    st.divider()
    st.header("💎 Valoración DCF")
    valor_base = None
    if flujo_caja and flujo_caja > 0 and acciones_en_circulacion:
        escenarios_dcf = calcular_escenarios_dcf(
            free_cash_flow=flujo_caja,
            deuda=deuda_total or 0,
            caja=caja_total or 0,
            acciones=acciones_en_circulacion
        )
        dcf_base = escenarios_dcf.get("base")
        valor_base = dcf_base.get("valor_por_accion") if dcf_base else None

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🔴 Pesimista", f"${escenarios_dcf['pesimista']['valor_por_accion']:,.2f}")
        with c2: st.metric("🟡 Base", f"${valor_base:,.2f}")
        with c3: st.metric("🟢 Optimista", f"${escenarios_dcf['optimista']['valor_por_accion']:,.2f}")
    else:
        st.info("No hay suficiente Free Cash Flow o acciones en circulación para calcular el DCF de forma fiable.")

    st.divider()
    st.header("💠 MARKET AI FAIR VALUE")
    fair_value = calcular_fair_value_combinado(precio_analisis, valor_base, objetivo_medio, eps, crecimiento_beneficios)
    estado_fair, potencial_fair = diagnosticar_fair_value(precio_analisis, fair_value)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Precio actual", f"${precio_analisis:,.2f}")
    with c2: st.metric("Fair Value", f"${fair_value:,.2f}" if fair_value else "N/D")
    with c3: st.metric("Potencial", f"{potencial_fair:+.2f}%" if potencial_fair else "N/D")
    st.write(f"### {estado_fair}")

    # Predictivo
    st.divider()
    st.header("🔮 Diagnóstico predictivo MARKET AI")
    diagnostico = calcular_diagnostico_predictivo(
        precio_analisis, fair_value, score_total, score_tecnico,
        score_valoracion, score_fundamentales, score_crecimiento, score_riesgo,
        rsi, ma20, ma50, ma200, objetivo_medio
    )
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Dirección Probable", diagnostico["direccion"])
    with col2: st.metric("Señal", diagnostico["señal"])
    with col3: st.metric("Horizonte", diagnostico["horizonte"])

    # Gráfico
    st.divider()
    st.header("📈 Gráfico")
    figura = go.Figure()
    figura.add_trace(go.Candlestick(x=datos.index, open=datos["Open"], high=datos["High"], low=datos["Low"], close=datos["Close"], name=ticker))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA20"], name="MA20"))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA50"], name="MA50"))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA200"], name="MA200"))
    figura.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(figura, use_container_width=True)
