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

    with st.spinner("Analizando mercado y fundamentales..."):

        try:

            activo = yf.Ticker(ticker)

            datos = activo.history(period=periodo)

            info = activo.info

            if datos.empty:
                st.error(
                    "No se han encontrado datos para este activo."
                )
                st.stop()

            # ==========================================
            # DATOS DE MERCADO
            # ==========================================

            precio = datos["Close"].iloc[-1]
            precio_anterior = datos["Close"].iloc[-2]

            variacion = (
                (precio - precio_anterior)
                / precio_anterior
            ) * 100

            maximo = datos["High"].max()
            minimo = datos["Low"].min()

            volumen = datos["Volume"].iloc[-1]

            # ==========================================
            # MEDIAS MÓVILES
            # ==========================================

            datos["MA20"] = datos["Close"].rolling(20).mean()
            datos["MA50"] = datos["Close"].rolling(50).mean()
            datos["MA200"] = datos["Close"].rolling(200).mean()

            ma20 = datos["MA20"].iloc[-1]
            ma50 = datos["MA50"].iloc[-1]
            ma200 = datos["MA200"].iloc[-1]

            # ==========================================
            # RSI
            # ==========================================

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

            rsi = datos["RSI"].iloc[-1]

            # ==========================================
            # VOLATILIDAD
            # ==========================================

            retornos = datos["Close"].pct_change()

            volatilidad = (
                retornos.std()
                * np.sqrt(252)
                * 100
            )

            # ==========================================
            # TENDENCIA
            # ==========================================

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

            # ==========================================
            # FUNDAMENTALES
            # ==========================================

            nombre = info.get(
                "longName",
                ticker
            )

            sector = info.get(
                "sector",
                "No disponible"
            )

            industria = info.get(
                "industry",
                "No disponible"
            )

            market_cap = info.get(
                "marketCap"
            )

            pe = info.get(
                "trailingPE"
            )

            forward_pe = info.get(
                "forwardPE"
            )

            peg = info.get(
                "pegRatio"
            )

            price_to_book = info.get(
                "priceToBook"
            )

            profit_margin = info.get(
                "profitMargins"
            )

            operating_margin = info.get(
                "operatingMargins"
            )

            revenue_growth = info.get(
                "revenueGrowth"
            )

            earnings_growth = info.get(
                "earningsGrowth"
            )

            debt_to_equity = info.get(
                "debtToEquity"
            )

            return_on_equity = info.get(
                "returnOnEquity"
            )

            free_cash_flow = info.get(
                "freeCashflow"
            )

            target_price = info.get(
                "targetMeanPrice"
            )

            # ==========================================
            # CABECERA
            # ==========================================

            st.success(
                f"Análisis completado: {nombre}"
            )

            st.write(
                f"**Sector:** {sector}  |  "
                f"**Industria:** {industria}"
            )

            # ==========================================
            # MERCADO
            # ==========================================

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
                    "Máximo periodo",
                    f"${maximo:,.2f}"
                )

            with col3:
                st.metric(
                    "Mínimo periodo",
                    f"${minimo:,.2f}"
                )

            with col4:
                st.metric(
                    "Volatilidad",
                    f"{volatilidad:.2f}%"
                )

            # ==========================================
            # GRÁFICO
            # ==========================================

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

            # ==========================================
            # INDICADORES
            # ==========================================

            st.subheader("🧠 Indicadores")

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
                    "El RSI está elevado. "
                    "El activo podría encontrarse "
                    "en una zona de sobrecompra."
                )

            elif rsi <= 30:

                st.success(
                    "El RSI está bajo. "
                    "El activo podría encontrarse "
                    "en una zona de sobreventa."
                )

            else:

                st.info(
                    "El RSI se encuentra en una zona intermedia."
                )

            # ==========================================
            # FUNDAMENTALES
            # ==========================================

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

            st.subheader("📋 Crecimiento y rentabilidad")

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

            st.subheader("🏦 Deuda y flujo de caja")

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

                if free_cash_flow:

                    st.metric(
                        "Flujo de caja libre",
                        f"${free_cash_flow / 1e9:.2f} B"
                    )

                else:

                    st.metric(
                        "Flujo de caja libre",
                        "N/D"
                    )

            # ==========================================
            # PRECIO OBJETIVO DE ANALISTAS
            # ==========================================

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
                        "Potencial estimado",
                        f"{diferencia_objetivo:+.2f}%"
                    )

            else:

                st.info(
                    "No hay un precio objetivo disponible."
                )

            # ==========================================
            # PRIMERA VALORACIÓN
            # ==========================================

            st.divider()

            st.header("🔎 Primera valoración")

            señales_positivas = 0
            señales_negativas = 0

            # PER

            if pe:

                if pe < 20:
                    señales_positivas += 1

                elif pe > 35:
                    señales_negativas += 1

            # Crecimiento

            if earnings_growth:

                if earnings_growth > 0.10:
                    señales_positivas += 1

                elif earnings_growth < 0:
                    señales_negativas += 1

            # ROE

            if return_on_equity:

                if return_on_equity > 0.15:
                    señales_positivas += 1

                elif return_on_equity < 0:
                    señales_negativas += 1

            # Deuda

            if debt_to_equity:

                if debt_to_equity < 100:
                    señales_positivas += 1

                elif debt_to_equity > 200:
                    señales_negativas += 1

            # Precio objetivo

            if target_price:

                if target_price > precio * 1.10:
                    señales_positivas += 1

                elif target_price < precio * 0.90:
                    señales_negativas += 1

            if señales_positivas > señales_negativas:

                valoracion = "🟢 FUNDAMENTALES FAVORABLES"

            elif señales_negativas > señales_positivas:

                valoracion = "🔴 FUNDAMENTALES DESFAVORABLES"

            else:

                valoracion = "🟡 FUNDAMENTALES MIXTOS"

            st.subheader(valoracion)

            st.write(
                f"Señales positivas detectadas: "
                f"**{señales_positivas}**"
            )

            st.write(
                f"Señales negativas detectadas: "
                f"**{señales_negativas}**"
            )

            st.warning(
                "⚠️ Esta valoración es experimental. "
                "Todavía no representa una recomendación "
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
