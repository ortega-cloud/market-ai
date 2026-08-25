import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="MARKET AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MARKET AI")
st.subheader("Sistema inteligente de análisis de mercados")

st.divider()
# =========================================================
# ALPHA VANTAGE - PRECIO ACTUAL
# =========================================================

@st.cache_data(ttl=900)
def obtener_precio_alpha_vantage(ticker):

    try:

        api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]

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

        datos = respuesta.json()

        # Comprobar errores de API

        if "Note" in datos:

            return None, "Límite de peticiones alcanzado"

        if "Information" in datos:

            return None, datos["Information"]

        if "Error Message" in datos:

            return None, "Símbolo no válido"

        # Obtener cotización

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

            return None, "Precio no disponible"

        return {
            "precio": float(precio),
            "fecha": fecha,
            "fuente": "Alpha Vantage"
        }, None

    except Exception as error:

        return None, str(error)

# =========================================================
# OBTENER PRECIOS
# =========================================================

@st.cache_data(ttl=900)
def obtener_precios(ticker, periodo):

    datos = yf.download(
        ticker,
        period=periodo,
        progress=False,
        auto_adjust=False
    )

    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)

    return datos


# =========================================================
# OBTENER FUNDAMENTALES
# =========================================================

@st.cache_data(ttl=3600)
def obtener_fundamentales(ticker):

    try:

        empresa = yf.Ticker(ticker)

        return empresa.get_info()

    except Exception:

        return {}


# =========================================================
# RSI
# =========================================================

def calcular_rsi(precios, periodo=14):

    diferencia = precios.diff()

    ganancias = diferencia.clip(lower=0)

    perdidas = -diferencia.clip(upper=0)

    media_ganancias = ganancias.rolling(periodo).mean()

    media_perdidas = perdidas.rolling(periodo).mean()

    rs = media_ganancias / media_perdidas

    return 100 - (100 / (1 + rs))


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

    # Precio vs MA20
    if precio > ma20:

        score += 5

        razones.append(
            "Precio por encima de MA20"
        )

    else:

        razones.append(
            "Precio por debajo de MA20"
        )


    # Precio vs MA50
    if precio > ma50:

        score += 5

        razones.append(
            "Precio por encima de MA50"
        )

    else:

        razones.append(
            "Precio por debajo de MA50"
        )


    # Precio vs MA200
    if not pd.isna(ma200):

        if precio > ma200:

            score += 7

            razones.append(
                "Precio por encima de MA200"
            )

        else:

            razones.append(
                "Precio por debajo de MA200"
            )

    else:

        score += 3


    # RSI
    if 40 <= rsi <= 65:

        score += 8

        razones.append(
            "RSI en zona saludable"
        )

    elif 30 <= rsi < 40:

        score += 6

        razones.append(
            "RSI bajo"
        )

    elif 65 < rsi < 70:

        score += 5

        razones.append(
            "RSI elevado"
        )

    elif rsi <= 30:

        score += 4

        razones.append(
            "RSI en sobreventa"
        )

    else:

        razones.append(
            "RSI en sobrecompra"
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

        if 0 < pe < 15:

            score += 8

            razones.append(
                "PER relativamente bajo"
            )

        elif 15 <= pe < 25:

            score += 6

            razones.append(
                "PER moderado"
            )

        elif 25 <= pe < 35:

            score += 3

            razones.append(
                "PER elevado"
            )

        elif pe >= 35:

            razones.append(
                "PER muy elevado"
            )

    else:

        razones.append(
            "PER no disponible"
        )


    # Forward PER
    if forward_pe is not None:

        if 0 < forward_pe < 15:

            score += 7

        elif 15 <= forward_pe < 25:

            score += 5

        elif 25 <= forward_pe < 35:

            score += 3


    # PEG
    if peg is not None:

        if 0 < peg < 1:

            score += 6

            razones.append(
                "PEG atractivo"
            )

        elif 1 <= peg < 2:

            score += 4

            razones.append(
                "PEG moderado"
            )

        elif peg >= 2:

            score += 1

            razones.append(
                "PEG elevado"
            )


    # Price / Book
    if price_to_book is not None:

        if 0 < price_to_book < 3:

            score += 4

        elif price_to_book >= 6:

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

        if roe >= 0.20:

            score += 8

            razones.append(
                "ROE fuerte"
            )

        elif roe >= 0.10:

            score += 5

            razones.append(
                "ROE aceptable"
            )

        elif roe < 0:

            razones.append(
                "ROE negativo"
            )


    # Margen
    if margen is not None:

        if margen >= 0.20:

            score += 7

            razones.append(
                "Margen de beneficio fuerte"
            )

        elif margen >= 0.10:

            score += 5

            razones.append(
                "Margen de beneficio saludable"
            )

        elif margen < 0:

            razones.append(
                "Margen negativo"
            )


    # Deuda
    if deuda is not None:

        if deuda < 50:

            score += 6

            razones.append(
                "Nivel de deuda bajo"
            )

        elif deuda < 100:

            score += 4

        elif deuda > 200:

            score += 1

            razones.append(
                "Nivel de deuda elevado"
            )


    # Flujo de caja
    if flujo_caja is not None:

        if flujo_caja > 0:

            score += 4

            razones.append(
                "Flujo de caja libre positivo"
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

        if crecimiento_ingresos >= 0.20:

            score += 8

            razones.append(
                "Crecimiento de ingresos fuerte"
            )

        elif crecimiento_ingresos >= 0.10:

            score += 6

        elif crecimiento_ingresos > 0:

            score += 3

        else:

            razones.append(
                "Ingresos en decrecimiento"
            )


    if crecimiento_beneficios is not None:

        if crecimiento_beneficios >= 0.20:

            score += 7

            razones.append(
                "Crecimiento de beneficios fuerte"
            )

        elif crecimiento_beneficios >= 0.10:

            score += 5

        elif crecimiento_beneficios > 0:

            score += 3

        else:

            razones.append(
                "Beneficios en decrecimiento"
            )


    return min(score, 15), razones


# =========================================================
# SCORE RIESGO
# =========================================================

def calcular_score_riesgo(
    volatilidad,
    deuda
):

    score = 10
    razones = []


    # Volatilidad
    if volatilidad > 50:

        score -= 5

        razones.append(
            "Volatilidad muy elevada"
        )

    elif volatilidad > 30:

        score -= 3

        razones.append(
            "Volatilidad elevada"
        )

    elif volatilidad > 20:

        score -= 1


    # Deuda
    if deuda is not None:

        if deuda > 200:

            score -= 3

        elif deuda > 100:

            score -= 1


    return max(score, 0), razones


# =========================================================
# ANALIZAR
# =========================================================

st.header("🔍 Analizar un activo")


ticker = st.text_input(
    "Introduce el símbolo",
    value="AAPL",
    placeholder="AAPL, NVDA, MSFT, TSLA..."
).upper().strip()


periodo = st.selectbox(
    "Periodo",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)


if st.button("📊 Analizar mercado"):

    if not ticker:

        st.warning(
            "Introduce un símbolo."
        )

        st.stop()


with st.spinner(
    f"Analizando {ticker}..."
):

    # =================================================
    # PRECIO ALPHA VANTAGE
    # =================================================

    precio_alpha, error_alpha = (
        obtener_precio_alpha_vantage(ticker)
    )


    # =================================================
    # PRECIO DE MERCADO
    # =================================================

    if precio_alpha:

        precio_mercado = precio_alpha["precio"]

        fecha_precio = precio_alpha["fecha"]

        st.info(
            f"💵 Precio de mercado: "
            f"${precio_mercado:,.2f}  |  "
            f"Fuente: Alpha Vantage  |  "
            f"Última sesión: {fecha_precio}"
        )

    else:

        precio_mercado = None

        st.warning(
            f"No se pudo obtener el precio de "
            f"Alpha Vantage: {error_alpha}"
        )


    try:
           
            # =================================================
            # DATOS
            # =================================================

            datos = obtener_precios(
                ticker,
                periodo
            )

            fundamentales = obtener_fundamentales(
                ticker
            )


            if datos.empty:

                st.error(
                    "No se han encontrado datos."
                )

                st.stop()


            # =================================================
            # MERCADO
            # =================================================

            precio = float(
                datos["Close"].iloc[-1]
            )

            precio_anterior = float(
                datos["Close"].iloc[-2]
            )

            variacion = (
                (precio - precio_anterior)
                / precio_anterior
            ) * 100


            maximo = float(
                datos["High"].max()
            )

            minimo = float(
                datos["Low"].min()
            )


            # =================================================
            # MEDIAS
            # =================================================

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


            # =================================================
            # RSI
            # =================================================

            datos["RSI"] = calcular_rsi(
                datos["Close"]
            )

            rsi = float(
                datos["RSI"].iloc[-1]
            )


            # =================================================
            # VOLATILIDAD
            # =================================================

            retornos = datos["Close"].pct_change()

            volatilidad = (
                retornos.std()
                * np.sqrt(252)
                * 100
            )


            # =================================================
            # FUNDAMENTALES
            # =================================================

            nombre = fundamentales.get(
                "longName",
                ticker
            )

            sector = fundamentales.get(
                "sector",
                "N/D"
            )

            pe = fundamentales.get(
                "trailingPE"
            )

            forward_pe = fundamentales.get(
                "forwardPE"
            )

            peg = fundamentales.get(
                "pegRatio"
            )

            price_to_book = fundamentales.get(
                "priceToBook"
            )

            roe = fundamentales.get(
                "returnOnEquity"
            )

            margen = fundamentales.get(
                "profitMargins"
            )

            deuda = fundamentales.get(
                "debtToEquity"
            )

            flujo_caja = fundamentales.get(
                "freeCashflow"
            )

            crecimiento_ingresos = fundamentales.get(
                "revenueGrowth"
            )

            crecimiento_beneficios = fundamentales.get(
                "earningsGrowth"
            )

            target_price = fundamentales.get(
                "targetMeanPrice"
            )


            # =================================================
            # SCORES
            # =================================================

            score_tecnico, razones_tecnico = (
                calcular_score_tecnico(
                    precio,
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


            # =================================================
            # TENDENCIA
            # =================================================

            if score_tecnico >= 20:

                tendencia = "🟢 FUERTE"

            elif score_tecnico >= 13:

                tendencia = "🟡 MODERADA"

            else:

                tendencia = "🔴 DÉBIL"


            # =================================================
            # RESULTADO
            # =================================================

            st.success(
                f"Análisis completado: {nombre}"
            )

            st.write(
                f"**Sector:** {sector}"
            )


            # =================================================
            # SCORE PRINCIPAL
            # =================================================

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


            # =================================================
            # DESGLOSE
            # =================================================

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


            # =================================================
            # DIAGNÓSTICO
            # =================================================

            st.divider()

            st.header(
                "🧠 Diagnóstico de MARKET AI"
            )


            if score_total >= 70:

                st.write(
                    f"**{ticker} presenta actualmente "
                    f"una configuración relativamente "
                    f"favorable**, con una puntuación de "
                    f"**{score_total}/100**."
                )

            elif score_total >= 55:

                st.write(
                    f"**{ticker} presenta una situación "
                    f"mixta**, con una puntuación de "
                    f"**{score_total}/100**."
                )

            else:

                st.write(
                    f"**{ticker} presenta actualmente "
                    f"varias señales desfavorables**, "
                    f"con una puntuación de "
                    f"**{score_total}/100**."
                )


            # =================================================
            # RAZONES
            # =================================================

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


            # =================================================
            # PRECIO OBJETIVO
            # =================================================

            st.divider()

            st.header(
                "🎯 Precio objetivo"
            )


            if target_price:

                potencial = (
                    (target_price - precio)
                    / precio
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
                    "Precio objetivo no disponible."
                )


            # =================================================
            # GRÁFICO
            # =================================================

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


            st.warning(
                "⚠️ El MARKET AI SCORE es actualmente "
                "un modelo experimental. No debe utilizarse "
                "por sí solo para tomar decisiones de inversión."
            )


        except Exception as error:

            st.error(
                "No se ha podido completar el análisis."
            )

            st.caption(
                f"Detalle técnico: {error}"
            )


# =========================================================
# TOP 5
# =========================================================

st.divider()

st.header(
    "🏆 TOP 5 OPORTUNIDADES"
)

col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric("🥇 #1", "—")

with col2:
    st.metric("🥈 #2", "—")

with col3:
    st.metric("🥉 #3", "—")

with col4:
    st.metric("4️⃣ #4", "—")

with col5:
    st.metric("5️⃣ #5", "—")


st.divider()

st.caption(
    "MARKET AI — Proyecto experimental de análisis financiero."
)
