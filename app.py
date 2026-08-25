import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="MARKET AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MARKET AI")
st.subheader("Sistema inteligente de análisis de mercados")

st.divider()

st.header("🔍 Analizar un activo")

ticker = st.text_input(
    "Introduce el símbolo del activo",
    value="AAPL",
    placeholder="Ejemplo: AAPL, NVDA, MSFT, TSLA..."
).upper()

periodo = st.selectbox(
    "Periodo del gráfico",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

if st.button("📊 Analizar mercado"):

    with st.spinner("Analizando el mercado..."):

        try:

            activo = yf.Ticker(ticker)
            datos = activo.history(period=periodo)

            if datos.empty:

                st.error(
                    "No se han encontrado datos para este activo."
                )

            else:

                # --------------------------------
                # DATOS BÁSICOS
                # --------------------------------

                precio = datos["Close"].iloc[-1]
                precio_anterior = datos["Close"].iloc[-2]

                variacion = (
                    (precio - precio_anterior)
                    / precio_anterior
                ) * 100

                maximo = datos["High"].max()
                minimo = datos["Low"].min()

                volumen = datos["Volume"].iloc[-1]

                # --------------------------------
                # MEDIAS MÓVILES
                # --------------------------------

                datos["MA20"] = datos["Close"].rolling(20).mean()
                datos["MA50"] = datos["Close"].rolling(50).mean()
                datos["MA200"] = datos["Close"].rolling(200).mean()

                # --------------------------------
                # RSI
                # --------------------------------

                diferencia = datos["Close"].diff()

                ganancias = diferencia.where(
                    diferencia > 0, 0
                )

                perdidas = -diferencia.where(
                    diferencia < 0, 0
                )

                media_ganancias = ganancias.rolling(14).mean()
                media_perdidas = perdidas.rolling(14).mean()

                rs = (
                    media_ganancias /
                    media_perdidas
                )

                datos["RSI"] = 100 - (
                    100 / (1 + rs)
                )

                rsi_actual = datos["RSI"].iloc[-1]

                # --------------------------------
                # VOLATILIDAD
                # --------------------------------

                retornos = datos["Close"].pct_change()

                volatilidad = (
                    retornos.std() *
                    np.sqrt(252) *
                    100
                )

                # --------------------------------
                # TENDENCIA
                # --------------------------------

                ma20 = datos["MA20"].iloc[-1]
                ma50 = datos["MA50"].iloc[-1]
                ma200 = datos["MA200"].iloc[-1]

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

                # --------------------------------
                # DATOS PRINCIPALES
                # --------------------------------

                st.success(
                    f"Análisis completado: {ticker}"
                )

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
                        "Volatilidad anual",
                        f"{volatilidad:.2f}%"
                    )

                # --------------------------------
                # GRÁFICO
                # --------------------------------

                st.divider()

                st.subheader(
                    "📈 Gráfico técnico"
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

                # --------------------------------
                # INDICADORES
                # --------------------------------

                st.divider()

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
                        f"{rsi_actual:.2f}"
                    )

                with col3:

                    st.metric(
                        "Volatilidad",
                        f"{volatilidad:.2f}%"
                    )

                # --------------------------------
                # INTERPRETACIÓN RSI
                # --------------------------------

                if rsi_actual >= 70:

                    rsi_diagnostico = (
                        "⚠️ RSI elevado: el activo podría "
                        "estar sobrecomprado."
                    )

                elif rsi_actual <= 30:

                    rsi_diagnostico = (
                        "🟢 RSI bajo: el activo podría "
                        "estar sobrevendido."
                    )

                else:

                    rsi_diagnostico = (
                        "🟡 RSI en una zona intermedia."
                    )

                st.info(rsi_diagnostico)

                # --------------------------------
                # DIAGNÓSTICO TÉCNICO
                # --------------------------------

                st.divider()

                st.subheader(
                    "🔎 Diagnóstico técnico"
                )

                if puntos_tendencia >= 3:

                    diagnostico = (
                        "La estructura actual presenta "
                        "una tendencia predominantemente alcista. "
                        "El precio se encuentra por encima de "
                        "las principales medias móviles."
                    )

                elif puntos_tendencia == 2:

                    diagnostico = (
                        "La estructura presenta señales "
                        "moderadamente alcistas, aunque "
                        "no todas las medias móviles confirman "
                        "la tendencia."
                    )

                elif puntos_tendencia == 1:

                    diagnostico = (
                        "La estructura presenta señales "
                        "moderadamente bajistas."
                    )

                else:

                    diagnostico = (
                        "La estructura presenta una "
                        "tendencia predominantemente bajista."
                    )

                st.write(diagnostico)

                st.warning(
                    "⚠️ Este análisis es experimental. "
                    "Todavía no constituye una recomendación "
                    "de inversión."
                )

        except Exception as error:

            st.error(
                f"Ha ocurrido un error: {error}"
            )

st.divider()

st.header("🏆 TOP 5 OPORTUNIDADES")

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
