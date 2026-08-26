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
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="MARKET AI",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# TÍTULO
# =========================================================

st.title("🤖 MARKET AI")

st.caption(
    "Sistema experimental de análisis técnico, fundamentales, "
    "valoración, analistas y mercado."
)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def limpiar_numero(valor):
    try:
        if valor is None:
            return None
        if pd.isna(valor):
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
# HISTÓRICOS Y DATOS GENERALES
# =========================================================

@st.cache_data(ttl=900)
def obtener_precios(ticker, periodo):
    try:
        datos = yf.download(ticker, period=periodo, auto_adjust=False, progress=False)
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
        empresa = yf.Ticker(ticker)
        datos = empresa.info
        return datos if datos is not None else {}
    except Exception:
        return {}

# =========================================================
# OBJETIVOS DE ANALISTAS Y ESTIMACIONES (CON FALLBACKS)
# =========================================================

@st.cache_data(ttl=3600)
def obtener_objetivos_analistas(ticker, info_dict=None):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.analyst_price_targets
        if datos is not None:
            if hasattr(datos, "to_dict"):
                datos = datos.to_dict()
            if isinstance(datos, dict) and any(v is not None for v in datos.values()):
                return datos
    except Exception:
        pass

    # FALLBACK: Extraer directamente de info si la API secundaria falla
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
        empresa = yf.Ticker(ticker)
        datos = empresa.recommendations_summary
        if datos is not None and not datos.empty:
            return datos
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_upgrades_downgrades(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.upgrades_downgrades
        if datos is not None and not datos.empty:
            return datos
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_estimaciones_eps(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.earnings_estimate
        if datos is not None and not datos.empty:
            return datos
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_estimaciones_ingresos(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.revenue_estimate
        if datos is not None and not datos.empty:
            return datos
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_revisiones_eps(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.eps_revisions
        if datos is not None and not datos.empty:
            return datos
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_tendencia_eps(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.eps_trend
        if datos is not None and not datos.empty:
            return datos
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_crecimiento_estimado(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.growth_estimates
        if datos is not None and not datos.empty:
            return datos
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=900)
def obtener_noticias(ticker):
    try:
        empresa = yf.Ticker(ticker)
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
        if precio > ma20:
            score += 5
            razones.append("El precio está por encima de la MA20.")
        else:
            razones.append("El precio está por debajo de la MA20.")

    if pd.notna(ma20) and pd.notna(ma50):
        if ma20 > ma50:
            score += 5
            razones.append("La MA20 está por encima de la MA50.")
        else:
            razones.append("La MA20 está por debajo de la MA50.")

    if pd.notna(ma50) and pd.notna(ma200):
        if ma50 > ma200:
            score += 7
            razones.append("La MA50 está por encima de la MA200.")
        else:
            razones.append("La MA50 está por debajo de la MA200.")

    if pd.notna(rsi):
        if rsi < 30:
            score += 6
            razones.append("El RSI indica posible sobreventa.")
        elif rsi <= 65:
            score += 8
            razones.append("El RSI se encuentra en una zona equilibrada.")
        elif rsi <= 70:
            score += 5
            razones.append("El RSI se aproxima a sobrecompra.")
        else:
            score += 2
            razones.append("El RSI indica posible sobrecompra.")

    return min(score, 25), razones


def calcular_score_valoracion(pe, forward_pe, peg, price_to_book):
    score = 0
    razones = []

    if pe is not None:
        if pe < 15: score += 8
        elif pe < 25: score += 6
        elif pe < 40: score += 3
        else: score += 1

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
        if roe > 0.20: score += 8
        elif roe > 0.10: score += 5
        else: score += 2

    if margen is not None:
        if margen > 0.20: score += 7
        elif margen > 0.10: score += 5
        else: score += 2

    if deuda is not None:
        if deuda < 50: score += 5
        elif deuda < 100: score += 3
        else: score += 1

    if flujo_caja is not None:
        if flujo_caja > 0: score += 5
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
# FAIR VALUE Y PREDICTOR
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

    peso_total = sum(pesos)
    return sum(v * p for v, p in zip(valores, pesos)) / peso_total


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
        if rsi >= 75:
            situacion_rsi = "SOBRECOMPRA"
            riesgos.append("El RSI indica una situación de sobrecompra.")
        elif rsi <= 30:
            situacion_rsi = "SOBREVENTA"
            señales.append("El RSI indica una situación de sobreventa.")
        elif 45 <= rsi <= 70:
            situacion_rsi = "SALUDABLE"

    if score_fundamentales >= 20: señales.append("Los fundamentales son sólidos.")
    elif score_fundamentales < 12: riesgos.append("Los fundamentales presentan varias señales débiles.")

    if score_crecimiento >= 12: señales.append("El crecimiento de ingresos y beneficios es favorable.")
    elif score_crecimiento < 7: riesgos.append("El crecimiento de la empresa es limitado o débil.")

    potencial_analistas = None
    if objetivo_analistas is not None and precio is not None and precio > 0:
        potencial_analistas = ((objetivo_analistas - precio) / precio) * 100

    puntos_alcistas = (3 if tendencia == "ALCISTA" else 0) + (3 if potencial_fair and potencial_fair >= 15 else 0) + (2 if score_fundamentales >= 20 else 0)
    puntos_bajistas = (3 if tendencia == "BAJISTA" else 0) + (3 if potencial_fair and potencial_fair <= -15 else 0) + (2 if score_fundamentales < 12 else 0)

    if puntos_alcistas >= puntos_bajistas + 2: direccion = "🟢 ALCISTA"
    elif puntos_bajistas >= puntos_alcistas + 2: direccion = "🔴 BAJISTA"
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
# INTERFAZ (SIDEBAR Y ACCIÓN)
# =========================================================

st.sidebar.header("⚙️ Configuración")
ticker = st.sidebar.text_input("Símbolo", value="NVDA").upper().strip()
periodo = st.sidebar.selectbox("Periodo", ["6mo", "1y", "2y", "5y", "10y"], index=1)

tipo_precio = st.sidebar.radio("Fuente", ["Precio automático", "Precio personalizado"])
precio_personalizado = st.sidebar.number_input("Precio de entrada", min_value=0.01, value=100.00, step=1.00) if tipo_precio == "Precio personalizado" else None

analizar = st.sidebar.button("📊 ANALIZAR", type="primary")

if analizar:
    if not ticker:
        st.error("Debes introducir un ticker.")
        st.stop()

    with st.spinner("Obteniendo datos..."):
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
        st.error(f"No se han encontrado precios históricos para {ticker}.")
        st.stop()

    precio_mercado = precio_alpha["precio"] if precio_alpha else None
    precio_historico = float(datos["Close"].iloc[-1])
    precio_analisis = precio_personalizado if tipo_precio == "Precio personalizado" else (precio_mercado or precio_historico)

    # Indicadores Técnicos
    datos["MA20"] = datos["Close"].rolling(20).mean()
    datos["MA50"] = datos["Close"].rolling(50).mean()
    datos["MA200"] = datos["Close"].rolling(200).mean()
    ma20, ma50, ma200 = datos["MA20"].iloc[-1], datos["MA50"].iloc[-1], datos["MA200"].iloc[-1]

    datos["RSI"] = calcular_rsi(datos["Close"])
    rsi = float(datos["RSI"].iloc[-1])
    volatilidad = datos["Close"].pct_change().std() * np.sqrt(252) * 100

    # Extraer variables con Fallbacks para FCF y Métricas
    nombre = info.get("longName", ticker)
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
    acciones_en_circulacion = limpiar_numero(info.get("sharesOutstanding"))
    ingresos = limpiar_numero(info.get("totalRevenue"))
    beneficio = limpiar_numero(info.get("netIncomeToCommon"))
    eps = limpiar_numero(info.get("trailingEps"))
    crecimiento_ingresos = limpiar_numero(info.get("revenueGrowth"))
    crecimiento_beneficios = limpiar_numero(info.get("earningsGrowth"))
    dividend_yield = limpiar_numero(info.get("dividendYield"))

    # Extraer Free Cash Flow directamente de Cashflow si falta en info
    flujo_caja = limpiar_numero(info.get("freeCashflow"))
    if flujo_caja is None:
        try:
            cf_df = yf.Ticker(ticker).cashflow
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
    st.header("🎯 MARKET AI SCORE")
    st.metric("Puntuación Total", f"{score_total}/100")
    st.progress(score_total / 100)

    # Fundamentales
    st.divider()
    st.header("📊 Fundamentales")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("PER", f"{pe:.2f}" if pe else "N/D")
    with c2: st.metric("PER futuro", f"{forward_pe:.2f}" if forward_pe else "N/D")
    with c3: st.metric("PEG", f"{peg:.2f}" if peg else "N/D")
    with c4: st.metric("Price/Book", f"{price_to_book:.2f}" if price_to_book else "N/D")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ROE", f"{roe * 100:.2f}%" if roe else "N/D")
    with c2: st.metric("Margen Beneficio", f"{margen * 100:.2f}%" if margen else "N/D")
    with c3: st.metric("Margen Operativo", f"{margen_operativo * 100:.2f}%" if margen_operativo else "N/D")
    with c4: st.metric("Deuda/Patrimonio", f"{deuda:.1f}" if deuda else "N/D")

    # Analistas
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

    # Estimaciones
    st.divider()
    st.header("🔮 Estimaciones de analistas")
    if not eps_estimaciones.empty:
        st.subheader("EPS Estimados")
        st.dataframe(eps_estimaciones, use_container_width=True)
    elif not ingresos_estimaciones.empty:
        st.subheader("Ingresos Estimados")
        st.dataframe(ingresos_estimaciones, use_container_width=True)
    else:
        st.info("No hay tablas avanzadas de estimaciones disponibles directamente para este ticker en Yahoo Finance.")

    # DCF y Fair Value
    st.divider()
    st.header("💎 Valoración DCF y Fair Value")
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

    fair_value = calcular_fair_value_combinado(precio_analisis, valor_base, objetivo_medio, eps, crecimiento_beneficios)
    estado_fair, potencial_fair = diagnosticar_fair_value(precio_analisis, fair_value)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Precio Actual", f"${precio_analisis:,.2f}")
    with c2: st.metric("DCF Base", f"${valor_base:,.2f}" if valor_base else "N/D")
    with c3: st.metric("Fair Value Combinado", f"${fair_value:,.2f}" if fair_value else "N/D")

    st.subheader(f"Diagnóstico: {estado_fair}")

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
    st.header("📈 Gráfico de Precios")
    fig = go.Figure(data=[go.Candlestick(x=datos.index, open=datos["Open"], high=datos["High"], low=datos["Low"], close=datos["Close"])])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
