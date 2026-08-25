import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np

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
# FUNCIÓN: OBTENER DATOS DE PRECIOS
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
# FUNCIÓN: OBTENER FUNDAMENTALES
# =========================================================

@st.cache_data(ttl=3600)
def obtener_fundamentales(ticker):

    try:

        empresa = yf.Ticker(ticker)

        datos = empresa.get_info()

        return datos

    except Exception:

        return {}


# =========================================================
# FUNCIÓN: CALCULAR RSI
# =========================================================

def calcular_rsi(precios, periodo=14):

    diferencia = precios.diff()

    ganancias = diferencia.clip(lower=0)

    perdidas = -diferencia.clip(upper=0)

    media_ganancias = ganancias.rolling(periodo).mean()

    media_perdidas = perdidas.rolling(periodo).mean()

    rs = media_ganancias / media_perdidas

    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================================================
# SELECCIÓN
# =========================================================

st.header("🔍 Analizar un activo")

ticker = st.text_input(
    "Introduce el símbolo del activo",
    value="AAPL",
    placeholder="Ejemplo: AAPL, NVDA, MSFT, TSLA..."
).upper().strip()


periodo = st.selectbox(
    "Periodo del gráfico",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)


# =========================================================
# BOTÓN ANALIZAR
# =========================================================

if st.button("📊 Analizar mercado"):

    if not ticker:

        st.warning("Introduce un símbolo.")

        st.stop()


    with st.spinner(
        f"Analizando {ticker}..."
    ):

        try:

            # =================================================
            # PRECIOS
            # =================================================

            datos = obtener_precios(
                ticker,
                periodo
            )


            if datos.empty:

                st.error(
                    "No se han encontrado datos para este activo."
                )

                st.stop()


            # =================================================
            # PRECIOS BÁSICOS
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


            volumen = float(
                datos["Volume"].iloc[-1]
            )


            # =================================================
            # MEDIAS MÓVILES
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

            retornos = (
                datos["Close"]
                .pct_change()
            )


            volatilidad = (
                retornos.std()
                * np.sqrt(252)
                * 100
            )


            # =================================================
            # TENDENCIA
            # =================================================

            puntos_tendencia = 0


            if precio > ma20:

                puntos_tendencia += 1


            if precio > ma50:

                puntos_tendencia += 1


            if not pd.isna(ma200):

                if precio > ma200:

                    puntos_tendencia += 1


            if puntos_tendencia >= 3:

                tendencia = "🟢 ALCISTA"


            elif puntos_tendencia == 2:

                tendencia = "🟡 NEUTRAL-ALCISTA"


            elif puntos_tendencia == 1:

                tendencia = "🟠 NEUTRAL-BAJISTA"


            else:

                tendencia = "🔴 BAJISTA"


            # =================================================
            # FUNDAMENTALES
            # =================================================

            fundamentales = obtener_fundamentales(
                ticker
            )


            nombre = fundamentales.get(
                "longName",
                ticker
            )


            sector = fundamentales.get(
                "sector",
                "No disponible"
            )


            industria = fundamentales.get(
                "industry",
                "No disponible"
            )


            market_cap = fundamentales.get(
                "marketCap"
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


            profit_margin = fundamentales.get(
                "profitMargins"
            )


            operating_margin = fundamentales.get(
                "operatingMargins"
            )


            revenue_growth = fundamentales.get(
                "revenueGrowth"
            )


            earnings_growth = fundamentales.get(
                "earningsGrowth"
            )


            debt_to_equity = fundamentales.get(
                "debtToEquity"
            )


            return_on_equity = fundamentales.get(
                "returnOnEquity"
            )


            free_cash_flow = fundamentales.get(
                "freeCashflow"
            )


            target_price = fundamentales.get(
                "targetMeanPrice"
            )


            # =================================================
            # CABECERA
            # =================================================

            st.success(
                f"Análisis completado: {nombre}"
            )


            st.write(
                f"**Sector:** {sector}  |  "
                f"**Industria:** {industria}"
            )


            # =================================================
            # MERCADO
            # =================================================

            st.divider()

            st.header("📊 Mercado")


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Precio",
                    f"${precio:,.2f}",
                    f"{variacion:+.2f}%"
                )


            with col2:

                st.metric(
                    "Máximo",
                    f"${maximo:,.2f}"
                )


            with col3:

                st.metric(
                    "Mínimo",
                    f"${minimo:,.2f}"
                )


            with col4:

                st.metric(
                    "Volatilidad",
                    f"{volatilidad:.2f}%"
                )


            # =================================================
            # GRÁFICO
            # =================================================

            st.divider()

            st.header("📈 Análisis técnico")


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
                    name="Media 20"
                )
            )


            figura.add_trace(
                go.Scatter(
                    x=datos.index,
                    y=datos["MA50"],
                    name="Media 50"
                )
            )


            figura.add_trace(
                go.Scatter(
                    x=datos.index,
                    y=datos["MA200"],
                    name="Media 200"
                )
            )


            figura.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Precio",
                xaxis_rangeslider_visible=False,
                height=600
            )


            st.plotly_chart(
                figura,
                use_container_width=True
            )


            # =================================================
            # INDICADORES
            # =================================================

            st.subheader(
                "🧠 Indicadores técnicos"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Tendencia",
                    tendencia
                )


            with col2:

                st.metric(
                    "RSI",
                    f"{rsi:.2f}"
                )


            with col3:

                st.metric(
                    "Volatilidad",
                    f"{volatilidad:.2f}%"
                )


            if rsi >= 70:

                st.warning(
                    "RSI elevado: posible zona de sobrecompra."
                )


            elif rsi <= 30:

                st.success(
                    "RSI bajo: posible zona de sobreventa."
                )


            else:

                st.info(
                    "RSI en zona intermedia."
                )


            # =================================================
            # FUNDAMENTALES
            # =================================================

            st.divider()

            st.header("💰 Fundamentales")


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                if market_cap:

                    st.metric(
                        "Capitalización",
                        f"${market_cap / 1e9:.2f} B"
                    )

                else:

                    st.metric(
                        "Capitalización",
                        "N/D"
                    )


            with col2:

                if pe:

                    st.metric(
                        "PER",
                        f"{pe:.2f}"
                    )

                else:

                    st.metric(
                        "PER",
                        "N/D"
                    )


            with col3:

                if forward_pe:

                    st.metric(
                        "PER futuro",
                        f"{forward_pe:.2f}"
                    )

                else:

                    st.metric(
                        "PER futuro",
                        "N/D"
                    )


            with col4:

                if price_to_book:

                    st.metric(
                        "Precio/Valor contable",
                        f"{price_to_book:.2f}"
                    )

                else:

                    st.metric(
                        "Precio/Valor contable",
                        "N/D"
                    )


            # =================================================
            # CRECIMIENTO Y RENTABILIDAD
            # =================================================

            st.subheader(
                "📋 Crecimiento y rentabilidad"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                if revenue_growth is not None:

                    st.metric(
                        "Crecimiento ingresos",
                        f"{revenue_growth * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "Crecimiento ingresos",
                        "N/D"
                    )


            with col2:

                if earnings_growth is not None:

                    st.metric(
                        "Crecimiento beneficios",
                        f"{earnings_growth * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "Crecimiento beneficios",
                        "N/D"
                    )


            with col3:

                if profit_margin is not None:

                    st.metric(
                        "Margen beneficio",
                        f"{profit_margin * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "Margen beneficio",
                        "N/D"
                    )


            with col4:

                if return_on_equity is not None:

                    st.metric(
                        "ROE",
                        f"{return_on_equity * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "ROE",
                        "N/D"
                    )


            # =================================================
            # DEUDA Y FLUJO DE CAJA
            # =================================================

            st.subheader(
                "🏦 Deuda y flujo de caja"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                if debt_to_equity is not None:

                    st.metric(
                        "Deuda / Capital",
                        f"{debt_to_equity:.2f}"
                    )

                else:

                    st.metric(
                        "Deuda / Capital",
                        "N/D"
                    )


            with col2:

                if operating_margin is not None:

                    st.metric(
                        "Margen operativo",
                        f"{operating_margin * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "Margen operativo",
                        "N/D"
                    )


            with col3:

                if free_cash_flow is not None:

                    st.metric(
                        "Flujo de caja libre",
                        f"${free_cash_flow / 1e9:.2f} B"
                    )

                else:

                    st.metric(
                        "Flujo de caja libre",
                        "N/D"
                    )


            # =================================================
            # PRECIO OBJETIVO
            # =================================================

            st.divider()

            st.header("🎯 Precio objetivo")


            if target_price:

                diferencia_objetivo = (
                    (target_price - precio)
                    / precio
                ) * 100


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Objetivo medio analistas",
                        f"${target_price:,.2f}"
                    )


                with col2:

                    st.metric(
                        "Potencial",
                        f"{diferencia_objetivo:+.2f}%"
                    )


            else:

                st.info(
                    "No hay precio objetivo disponible."
                )


            # =================================================
            # VALORACIÓN PRELIMINAR
            # =================================================

            st.divider()

            st.header(
                "🔎 Valoración preliminar"
            )


            positivas = 0
            negativas = 0


            if pe:

                if pe < 20:

                    positivas += 1

                elif pe > 35:

                    negativas += 1


            if earnings_growth is not None:

                if earnings_growth > 0.10:

                    positivas += 1

                elif earnings_growth < 0:

                    negativas += 1


            if return_on_equity is not None:

                if return_on_equity > 0.15:

                    positivas += 1

                elif return_on_equity < 0:

                    negativas += 1


            if debt_to_equity is not None:

                if debt_to_equity < 100:

                    positivas += 1

                elif debt_to_equity > 200:

                    negativas += 1


            if target_price:

                if target_price > precio * 1.10:

                    positivas += 1

                elif target_price < precio * 0.90:

                    negativas += 1


            if positivas > negativas:

                valoracion = (
                    "🟢 FUNDAMENTALES FAVORABLES"
                )


            elif negativas > positivas:

                valoracion = (
                    "🔴 FUNDAMENTALES DESFAVORABLES"
                )


            else:

                valoracion = (
                    "🟡 FUNDAMENTALES MIXTOS"
                )


            st.subheader(
                valoracion
            )


            st.write(
                f"Señales positivas: **{positivas}**"
            )


            st.write(
                f"Señales negativas: **{negativas}**"
            )


            st.warning(
                "⚠️ Esta valoración es experimental "
                "y todavía no constituye una recomendación "
                "de inversión."
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

    st.metric(
        "🥇 #1",
        "—"
    )


with col2:

    st.metric(
        "🥈 #2",
        "—"
    )


with col3:

    st.metric(
        "🥉 #3",
        "—"
    )


with col4:

    st.metric(
        "4️⃣ #4",
        "—"
    )


with col5:

    st.metric(
        "5️⃣ #5",
        "—"
    )


st.divider()


st.caption(
    "MARKET AI — Proyecto experimental de análisis financiero."
)
