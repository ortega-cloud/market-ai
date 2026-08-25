import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests


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
    "Sistema experimental de análisis de mercados, "
    "valoración, fundamentales y análisis técnico."
)


# =========================================================
# FUNCIONES GENERALES
# =========================================================

def limpiar_numero(valor):
    """
    Convierte valores numéricos de Yahoo Finance
    en números utilizables.
    """

    try:
        if valor is None:
            return None

        if pd.isna(valor):
            return None

        return float(valor)

    except Exception:
        return None


# =========================================================
# ALPHA VANTAGE - PRECIO
# =========================================================

@st.cache_data(ttl=900)
def obtener_precio_alpha_vantage(ticker):

    try:

        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]

    except Exception:

        return None, (
            "No se ha encontrado ALPHA_VANTAGE_API_KEY "
            "en los Secrets de Streamlit."
        )

    try:

        url = "https://www.alphavantage.co/query"

        parametros = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": api_key
        }

        respuesta = requests.get(
            url,
            params=parametros,
            timeout=10
        )

        respuesta.raise_for_status()

        datos = respuesta.json()

        if "Note" in datos:

            return None, (
                "Alpha Vantage ha alcanzado el límite "
                "de peticiones gratuitas."
            )

        if "Information" in datos:

            return None, datos["Information"]

        if "Error Message" in datos:

            return None, "El símbolo introducido no es válido."

        quote = datos.get(
            "Global Quote",
            {}
        )

        precio = quote.get(
            "05. price"
        )

        fecha = quote.get(
            "07. latest trading day"
        )

        if not precio:

            return None, (
                "Alpha Vantage no ha devuelto un precio."
            )

        return {
            "precio": float(precio),
            "fecha": fecha,
            "fuente": "Alpha Vantage"
        }, None

    except Exception as error:

        return None, str(error)


# =========================================================
# YAHOO FINANCE - DATOS HISTÓRICOS
# =========================================================

@st.cache_data(ttl=900)
def obtener_precios(ticker, periodo):

    try:

        datos = yf.download(
            ticker,
            period=periodo,
            auto_adjust=False,
            progress=False
        )

        if datos is None:
            return pd.DataFrame()

        if datos.empty:
            return pd.DataFrame()

        # Algunas versiones de yfinance devuelven
        # columnas MultiIndex.

        if isinstance(datos.columns, pd.MultiIndex):

            datos.columns = datos.columns.get_level_values(0)

        columnas_necesarias = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        columnas_disponibles = [
            columna
            for columna in columnas_necesarias
            if columna in datos.columns
        ]

        datos = datos[columnas_disponibles].copy()

        return datos.dropna(
            subset=["Close"]
        )

    except Exception:

        return pd.DataFrame()


# =========================================================
# YAHOO FINANCE - FUNDAMENTALES
# =========================================================

@st.cache_data(ttl=3600)
def obtener_fundamentales(ticker):

    try:

        empresa = yf.Ticker(ticker)

        info = empresa.info

        if info is None:
            return {}

        return info

    except Exception:

        return {}


# =========================================================
# RSI
# =========================================================

def calcular_rsi(precios, periodo=14):

    delta = precios.diff()

    ganancias = delta.clip(
        lower=0
    )

    perdidas = -delta.clip(
        upper=0
    )

    media_ganancias = ganancias.rolling(
        periodo
    ).mean()

    media_perdidas = perdidas.rolling(
        periodo
    ).mean()

    rs = (
        media_ganancias
        / media_perdidas.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# =========================================================
# SCORE TÉCNICO
# =========================================================

def calcular_score_tecnico(
    precio,
    ma20,
    ma50,
    ma200,
    rsi
):

    score = 0

    razones = []

    # Precio frente a MA20

    if pd.notna(ma20):

        if precio > ma20:

            score += 5

            razones.append(
                "El precio está por encima de la MA20."
            )

        else:

            razones.append(
                "El precio está por debajo de la MA20."
            )

    # MA20 frente a MA50

    if pd.notna(ma20) and pd.notna(ma50):

        if ma20 > ma50:

            score += 5

            razones.append(
                "La MA20 está por encima de la MA50."
            )

        else:

            razones.append(
                "La MA20 está por debajo de la MA50."
            )

    # MA50 frente a MA200

    if pd.notna(ma50) and pd.notna(ma200):

        if ma50 > ma200:

            score += 7

            razones.append(
                "La MA50 está por encima de la MA200."
            )

        else:

            razones.append(
                "La MA50 está por debajo de la MA200."
            )

    # RSI

    if pd.notna(rsi):

        if 40 <= rsi <= 65:

            score += 8

            razones.append(
                "El RSI se encuentra en una zona relativamente equilibrada."
            )

        elif rsi < 30:

            score += 6

            razones.append(
                "El RSI indica posible sobreventa."
            )

        elif rsi > 70:

            score += 2

            razones.append(
                "El RSI indica posible sobrecompra."
            )

        else:

            score += 4

            razones.append(
                "El RSI muestra una situación intermedia."
            )

    return min(score, 25), razones


# =========================================================
# SCORE VALORACIÓN
# =========================================================

def calcular_score_valoracion(
    pe,
    forward_pe,
    peg,
    price_to_book
):

    score = 0

    razones = []

    # PER

    if pe is not None:

        if pe < 15:

            score += 8

            razones.append(
                "El PER actual es relativamente bajo."
            )

        elif pe < 25:

            score += 6

            razones.append(
                "El PER actual se encuentra en una zona intermedia."
            )

        elif pe < 40:

            score += 3

            razones.append(
                "El PER actual es relativamente elevado."
            )

        else:

            score += 1

            razones.append(
                "El PER actual es muy elevado."
            )

    else:

        razones.append(
            "No hay PER disponible."
        )

    # Forward PER

    if forward_pe is not None:

        if forward_pe < 15:

            score += 7

            razones.append(
                "El PER futuro es relativamente atractivo."
            )

        elif forward_pe < 25:

            score += 5

            razones.append(
                "El PER futuro se encuentra en una zona intermedia."
            )

        elif forward_pe < 40:

            score += 2

            razones.append(
                "El PER futuro es elevado."
            )

    # PEG

    if peg is not None:

        if peg < 1:

            score += 6

            razones.append(
                "El PEG puede indicar una valoración atractiva."
            )

        elif peg < 2:

            score += 4

            razones.append(
                "El PEG se encuentra en una zona intermedia."
            )

        else:

            score += 1

            razones.append(
                "El PEG es elevado."
            )

    # Price to Book

    if price_to_book is not None:

        if price_to_book < 2:

            score += 4

        elif price_to_book < 5:

            score += 2

        else:

            score += 1

    return min(score, 25), razones


# =========================================================
# SCORE FUNDAMENTALES
# =========================================================

def calcular_score_fundamentales(
    roe,
    margen,
    deuda,
    flujo_caja
):

    score = 0

    razones = []

    # ROE

    if roe is not None:

        if roe > 0.20:

            score += 8

            razones.append(
                "El ROE es elevado."
            )

        elif roe > 0.10:

            score += 5

            razones.append(
                "El ROE es razonable."
            )

        else:

            score += 2

            razones.append(
                "El ROE es relativamente bajo."
            )

    # Margen

    if margen is not None:

        if margen > 0.20:

            score += 7

            razones.append(
                "El margen de beneficios es elevado."
            )

        elif margen > 0.10:

            score += 5

            razones.append(
                "El margen de beneficios es razonable."
            )

        else:

            score += 2

            razones.append(
                "El margen de beneficios es reducido."
            )

    # Deuda

    if deuda is not None:

        if deuda < 50:

            score += 5

            razones.append(
                "El nivel de deuda respecto al capital es reducido."
            )

        elif deuda < 100:

            score += 3

            razones.append(
                "El nivel de deuda es moderado."
            )

        else:

            score += 1

            razones.append(
                "El nivel de deuda es elevado."
            )

    # Flujo de caja

    if flujo_caja is not None:

        if flujo_caja > 0:

            score += 5

            razones.append(
                "La empresa presenta flujo de caja libre positivo."
            )

        else:

            razones.append(
                "El flujo de caja libre no es positivo."
            )

    return min(score, 25), razones


# =========================================================
# SCORE CRECIMIENTO
# =========================================================

def calcular_score_crecimiento(
    crecimiento_ingresos,
    crecimiento_beneficios
):

    score = 0

    razones = []

    if crecimiento_ingresos is not None:

        if crecimiento_ingresos > 0.15:

            score += 7

            razones.append(
                "Los ingresos presentan un crecimiento elevado."
            )

        elif crecimiento_ingresos > 0.05:

            score += 5

            razones.append(
                "Los ingresos presentan crecimiento moderado."
            )

        elif crecimiento_ingresos > 0:

            score += 2

            razones.append(
                "Los ingresos crecen ligeramente."
            )

        else:

            razones.append(
                "Los ingresos están decreciendo."
            )

    if crecimiento_beneficios is not None:

        if crecimiento_beneficios > 0.15:

            score += 8

            razones.append(
                "Los beneficios presentan un crecimiento elevado."
            )

        elif crecimiento_beneficios > 0.05:

            score += 5

            razones.append(
                "Los beneficios presentan crecimiento moderado."
            )

        elif crecimiento_beneficios > 0:

            score += 2

            razones.append(
                "Los beneficios presentan crecimiento positivo."
            )

        else:

            razones.append(
                "Los beneficios están decreciendo."
            )

    return min(score, 15), razones


# =========================================================
# SCORE RIESGO
# =========================================================

def calcular_score_riesgo(
    volatilidad,
    deuda
):

    score = 0

    razones = []

    # Volatilidad

    if volatilidad < 20:

        score += 6

        razones.append(
            "La volatilidad histórica es relativamente baja."
        )

    elif volatilidad < 35:

        score += 4

        razones.append(
            "La volatilidad histórica es moderada."
        )

    elif volatilidad < 50:

        score += 2

        razones.append(
            "La volatilidad histórica es elevada."
        )

    else:

        score += 1

        razones.append(
            "La volatilidad histórica es muy elevada."
        )

    # Deuda

    if deuda is not None:

        if deuda < 50:

            score += 4

        elif deuda < 100:

            score += 2

        else:

            score += 1

    else:

        score += 2

    return min(score, 10), razones


# =========================================================
# INTERFAZ
# =========================================================

st.sidebar.header(
    "⚙️ Configuración"
)

ticker = st.sidebar.text_input(
    "Símbolo de la empresa",
    value="AAPL"
).upper().strip()


periodo = st.sidebar.selectbox(
    "Periodo del gráfico",
    [
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y"
    ],
    index=1
)


st.sidebar.divider()

st.sidebar.subheader(
    "🎯 Precio para el análisis"
)

tipo_precio = st.sidebar.radio(
    "Selecciona el precio",
    [
        "Precio automático",
        "Precio personalizado"
    ]
)


precio_personalizado = None


if tipo_precio == "Precio personalizado":

    precio_personalizado = st.sidebar.number_input(
        "Precio de entrada",
        min_value=0.01,
        value=100.00,
        step=1.00
    )


analizar = st.sidebar.button(
    "📊 Analizar mercado",
    type="primary"
)


# =========================================================
# INFORMACIÓN INICIAL
# =========================================================

st.info(
    "Introduce un ticker en el panel izquierdo "
    "y pulsa «Analizar mercado»."
)


# =========================================================
# ANÁLISIS PRINCIPAL
# =========================================================

if analizar:

    if not ticker:

        st.error(
            "Debes introducir un ticker."
        )

        st.stop()


    # =====================================================
    # PRECIO ALPHA VANTAGE
    # =====================================================

    with st.spinner(
        "Obteniendo precio de mercado..."
    ):

        precio_alpha, error_alpha = (
            obtener_precio_alpha_vantage(
                ticker
            )
        )


    # =====================================================
    # PRECIO DE MERCADO
    # =====================================================

    if precio_alpha:

        precio_mercado = precio_alpha["precio"]

        fecha_precio = precio_alpha["fecha"]

        st.info(
            f"💵 **Precio de mercado:** "
            f"${precio_mercado:,.2f}  |  "
            f"**Fuente:** Alpha Vantage  |  "
            f"**Última sesión:** {fecha_precio}"
        )

    else:

        precio_mercado = None

        st.warning(
            "⚠️ No se pudo obtener el precio mediante "
            "Alpha Vantage."
        )

        st.caption(
            f"Detalle: {error_alpha}"
        )


    # =====================================================
    # PRECIO UTILIZADO
    # =====================================================

    if (
        tipo_precio == "Precio personalizado"
        and precio_personalizado is not None
    ):

        precio_analisis = float(
            precio_personalizado
        )

        st.success(
            f"🎯 MARKET AI utilizará "
            f"**${precio_analisis:,.2f}** "
            f"como precio para el análisis."
        )

    elif precio_mercado is not None:

        precio_analisis = float(
            precio_mercado
        )

        st.success(
            f"🎯 MARKET AI utilizará el precio "
            f"actual de **${precio_analisis:,.2f}**."
        )

    else:

        precio_analisis = None


    # =====================================================
    # DATOS HISTÓRICOS
    # =====================================================

    with st.spinner(
        "Obteniendo datos históricos..."
    ):

        datos = obtener_precios(
            ticker,
            periodo
        )


    # =====================================================
    # FUNDAMENTALES
    # =====================================================

    with st.spinner(
        "Obteniendo fundamentales..."
    ):

        fundamentales = obtener_fundamentales(
            ticker
        )


    if datos.empty:

        st.error(
            "❌ No se han encontrado datos históricos "
            f"para {ticker}."
        )

        st.stop()


    # =====================================================
    # PRECIO HISTÓRICO
    # =====================================================

    try:

        precio_historico = float(
            datos["Close"].iloc[-1]
        )

        precio_anterior = float(
            datos["Close"].iloc[-2]
        )

    except Exception:

        st.error(
            "No se ha podido interpretar el precio histórico."
        )

        st.stop()


    variacion = (
        (
            precio_historico
            - precio_anterior
        )
        / precio_anterior
    ) * 100


    maximo = float(
        datos["High"].max()
    )

    minimo = float(
        datos["Low"].min()
    )


    # =====================================================
    # MEDIAS MÓVILES
    # =====================================================

    datos["MA20"] = (
        datos["Close"]
        .rolling(20)
        .mean()
    )

    datos["MA50"] = (
        datos["Close"]
        .rolling(50)
        .mean()
    )

    datos["MA200"] = (
        datos["Close"]
        .rolling(200)
        .mean()
    )


    ma20 = datos["MA20"].iloc[-1]

    ma50 = datos["MA50"].iloc[-1]

    ma200 = datos["MA200"].iloc[-1]


    # =====================================================
    # RSI
    # =====================================================

    datos["RSI"] = calcular_rsi(
        datos["Close"]
    )

    rsi = float(
        datos["RSI"].iloc[-1]
    )


    # =====================================================
    # VOLATILIDAD
    # =====================================================

    retornos = (
        datos["Close"]
        .pct_change()
    )

    volatilidad = (
        retornos.std()
        * np.sqrt(252)
        * 100
    )


    # =====================================================
    # FUNDAMENTALES
    # =====================================================

    nombre = fundamentales.get(
        "longName",
        ticker
    )

    sector = fundamentales.get(
        "sector",
        "N/D"
    )

    pe = limpiar_numero(
        fundamentales.get(
            "trailingPE"
        )
    )

    forward_pe = limpiar_numero(
        fundamentales.get(
            "forwardPE"
        )
    )

    peg = limpiar_numero(
        fundamentales.get(
            "pegRatio"
        )
    )

    price_to_book = limpiar_numero(
        fundamentales.get(
            "priceToBook"
        )
    )

    roe = limpiar_numero(
        fundamentales.get(
            "returnOnEquity"
        )
    )

    margen = limpiar_numero(
        fundamentales.get(
            "profitMargins"
        )
    )

    deuda = limpiar_numero(
        fundamentales.get(
            "debtToEquity"
        )
    )

    flujo_caja = limpiar_numero(
        fundamentales.get(
            "freeCashflow"
        )
    )

    crecimiento_ingresos = limpiar_numero(
        fundamentales.get(
            "revenueGrowth"
        )
    )

    crecimiento_beneficios = limpiar_numero(
        fundamentales.get(
            "earningsGrowth"
        )
    )

    target_price = limpiar_numero(
        fundamentales.get(
            "targetMeanPrice"
        )
    )


    # =====================================================
    # SCORES
    # =====================================================

    score_tecnico, razones_tecnico = (
        calcular_score_tecnico(
            precio_historico,
            ma20,
            ma50,
            ma200,
            rsi
        )
    )


    score_valoracion, razones_valoracion = (
        calcular_score_valoracion(
            pe,
            forward_pe,
            peg,
            price_to_book
        )
    )


    score_fundamentales, razones_fundamentales = (
        calcular_score_fundamentales(
            roe,
            margen,
            deuda,
            flujo_caja
        )
    )


    score_crecimiento, razones_crecimiento = (
        calcular_score_crecimiento(
            crecimiento_ingresos,
            crecimiento_beneficios
        )
    )


    score_riesgo, razones_riesgo = (
        calcular_score_riesgo(
            volatilidad,
            deuda
        )
    )


    score_total = (
        score_tecnico
        + score_valoracion
        + score_fundamentales
        + score_crecimiento
        + score_riesgo
    )


    # =====================================================
    # TENDENCIA
    # =====================================================

    if score_tecnico >= 20:

        tendencia = "🟢 FUERTE"

    elif score_tecnico >= 13:

        tendencia = "🟡 MODERADA"

    else:

        tendencia = "🔴 DÉBIL"


    # =====================================================
    # RESULTADO
    # =====================================================

    st.success(
        f"✅ Análisis completado: {nombre}"
    )

    st.write(
        f"**Sector:** {sector}"
    )


    # =====================================================
    # MÉTRICAS DE MERCADO
    # =====================================================

    st.divider()

    st.header(
        "💵 Mercado"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Precio histórico",
            f"${precio_historico:,.2f}",
            f"{variacion:+.2f}%"
        )


    with col2:

        if precio_analisis is not None:

            diferencia_precio = (
                (
                    precio_analisis
                    - precio_historico
                )
                / precio_historico
            ) * 100

            st.metric(
                "Precio analizado",
                f"${precio_analisis:,.2f}",
                f"{diferencia_precio:+.2f}%"
            )

        else:

            st.metric(
                "Precio analizado",
                "N/D"
            )


    with col3:

        st.metric(
            "Máximo del periodo",
            f"${maximo:,.2f}"
        )


    with col4:

        st.metric(
            "Mínimo del periodo",
            f"${minimo:,.2f}"
        )


    # =====================================================
    # MARKET AI SCORE
    # =====================================================

    st.divider()

    st.header(
        "🎯 MARKET AI SCORE"
    )


    if score_total >= 85:

        estado = (
            "🟢 OPORTUNIDAD MUY INTERESANTE"
        )

    elif score_total >= 70:

        estado = (
            "🟢 OPORTUNIDAD INTERESANTE"
        )

    elif score_total >= 55:

        estado = (
            "🟡 NEUTRAL"
        )

    elif score_total >= 40:

        estado = (
            "🟠 RIESGO ELEVADO"
        )

    else:

        estado = (
            "🔴 POCO ATRACTIVA"
        )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Puntuación",
            f"{score_total}/100"
        )


    with col2:

        st.subheader(
            estado
        )


    st.progress(
        score_total / 100
    )


    # =====================================================
    # DESGLOSE
    # =====================================================

    st.subheader(
        "📊 Desglose de la puntuación"
    )


    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "📈 Técnica",
            f"{score_tecnico}/25"
        )


    with col2:

        st.metric(
            "💰 Valoración",
            f"{score_valoracion}/25"
        )


    with col3:

        st.metric(
            "📊 Fundamentales",
            f"{score_fundamentales}/25"
        )


    with col4:

        st.metric(
            "🚀 Crecimiento",
            f"{score_crecimiento}/15"
        )


    with col5:

        st.metric(
            "⚠️ Riesgo",
            f"{score_riesgo}/10"
        )


    # =====================================================
    # DIAGNÓSTICO
    # =====================================================

    st.divider()

    st.header(
        "🧠 Diagnóstico de MARKET AI"
    )


    if score_total >= 70:

        st.write(
            f"**{ticker} presenta actualmente "
            f"una configuración relativamente favorable**, "
            f"con una puntuación de "
            f"**{score_total}/100**."
        )

    elif score_total >= 55:

        st.write(
            f"**{ticker} presenta una situación mixta**, "
            f"con una puntuación de "
            f"**{score_total}/100**."
        )

    else:

        st.write(
            f"**{ticker} presenta actualmente "
            f"varias señales desfavorables**, "
            f"con una puntuación de "
            f"**{score_total}/100**."
        )


    st.write(
        f"**Tendencia técnica:** {tendencia}"
    )


    # =====================================================
    # RAZONES
    # =====================================================

    st.subheader(
        "🔎 ¿Por qué obtiene esta puntuación?"
    )


    todas_las_razones = (
        razones_tecnico
        + razones_valoracion
        + razones_fundamentales
        + razones_crecimiento
        + razones_riesgo
    )


    for razon in todas_las_razones:

        st.write(
            f"• {razon}"
        )


    # =====================================================
    # FUNDAMENTALES
    # =====================================================

    st.divider()

    st.header(
        "📊 Fundamentales"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "PER",
            (
                f"{pe:.2f}"
                if pe is not None
                else "N/D"
            )
        )


    with col2:

        st.metric(
            "PER futuro",
            (
                f"{forward_pe:.2f}"
                if forward_pe is not None
                else "N/D"
            )
        )


    with col3:

        st.metric(
            "PEG",
            (
                f"{peg:.2f}"
                if peg is not None
                else "N/D"
            )
        )


    with col4:

        st.metric(
            "Precio/Valor contable",
            (
                f"{price_to_book:.2f}"
                if price_to_book is not None
                else "N/D"
            )
        )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "ROE",
            (
                f"{roe * 100:.2f}%"
                if roe is not None
                else "N/D"
            )
        )


    with col2:

        st.metric(
            "Margen",
            (
                f"{margen * 100:.2f}%"
                if margen is not None
                else "N/D"
            )
        )


    with col3:

        st.metric(
            "Crecimiento ingresos",
            (
                f"{crecimiento_ingresos * 100:.2f}%"
                if crecimiento_ingresos is not None
                else "N/D"
            )
        )


    with col4:

        st.metric(
            "Crecimiento beneficios",
            (
                f"{crecimiento_beneficios * 100:.2f}%"
                if crecimiento_beneficios is not None
                else "N/D"
            )
        )


    # =====================================================
    # PRECIO OBJETIVO DE ANALISTAS
    # =====================================================

    st.divider()

    st.header(
        "🎯 Precio objetivo de analistas"
    )


    if target_price is not None:

        precio_referencia = (
            precio_analisis
            if precio_analisis is not None
            else precio_historico
        )

        potencial = (
            (
                target_price
                - precio_referencia
            )
            / precio_referencia
        ) * 100


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Objetivo medio",
                f"${target_price:,.2f}"
            )


        with col2:

            st.metric(
                "Potencial",
                f"{potencial:+.2f}%"
            )

    else:

        st.info(
            "El precio objetivo de analistas "
            "no está disponible."
        )


    # =====================================================
    # GRÁFICO
    # =====================================================

    st.divider()

    st.header(
        "📈 Gráfico"
    )


    figura = go.Figure()


    figura.add_trace(
        go.Candlestick(
            x=datos.index,
            open=datos["Open"],
            high=datos["High"],
            low=datos["Low"],
            close=datos["Close"],
            name=ticker
        )
    )


    figura.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["MA20"],
            name="MA20"
        )
    )


    figura.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["MA50"],
            name="MA50"
        )
    )


    figura.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["MA200"],
            name="MA200"
        )
    )


    figura.update_layout(
        xaxis_rangeslider_visible=False,
        height=600
    )


    st.plotly_chart(
        figura,
        use_container_width=True
    )


    # =====================================================
    # RSI
    # =====================================================

    st.divider()

    st.header(
        "📉 RSI"
    )


    figura_rsi = go.Figure()


    figura_rsi.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos["RSI"],
            name="RSI"
        )
    )


    figura_rsi.add_hline(
        y=70,
        line_dash="dash"
    )


    figura_rsi.add_hline(
        y=30,
        line_dash="dash"
    )


    figura_rsi.update_layout(
        height=350,
        yaxis_title="RSI"
    )


    st.plotly_chart(
        figura_rsi,
        use_container_width=True
    )


    # =====================================================
    # VOLATILIDAD
    # =====================================================

    st.divider()

    st.header(
        "⚠️ Riesgo y volatilidad"
    )


    st.metric(
        "Volatilidad anualizada",
        f"{volatilidad:.2f}%"
    )


    # =====================================================
    # PRECIO PERSONALIZADO
    # =====================================================

    if (
        tipo_precio == "Precio personalizado"
        and precio_personalizado is not None
        and precio_mercado is not None
    ):

        st.divider()

        st.header(
            "🎯 Análisis del precio personalizado"
        )


        diferencia = (
            (
                precio_personalizado
                - precio_mercado
            )
            / precio_mercado
        ) * 100


        st.write(
            f"Has indicado un precio de entrada de "
            f"**${precio_personalizado:,.2f}**."
        )


        st.write(
            f"El precio de mercado obtenido es "
            f"**${precio_mercado:,.2f}**."
        )


        if precio_personalizado < precio_mercado:

            st.success(
                f"El precio personalizado está "
                f"{abs(diferencia):.2f}% por debajo "
                f"del precio actual."
            )

        elif precio_personalizado > precio_mercado:

            st.warning(
                f"El precio personalizado está "
                f"{diferencia:.2f}% por encima "
                f"del precio actual."
            )

        else:

            st.info(
                "El precio personalizado coincide "
                "con el precio actual."
            )


    # =====================================================
    # AVISO
    # =====================================================

    st.divider()

    st.warning(
        "⚠️ MARKET AI es actualmente un modelo "
        "experimental. La puntuación y los datos "
        "no constituyen asesoramiento financiero "
        "y no deben utilizarse por sí solos para "
        "tomar decisiones de inversión."
    )
