import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

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
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

if st.button("📊 Analizar mercado"):

    with st.spinner("Obteniendo datos del mercado..."):

        try:
            activo = yf.Ticker(ticker)
            datos = activo.history(period=periodo)

            if datos.empty:
                st.error(
                    "No se han encontrado datos para este activo. "
                    "Comprueba el símbolo."
                )
            else:

                precio = datos["Close"].iloc[-1]
                precio_anterior = datos["Close"].iloc[-2]

                variacion = (
                    (precio - precio_anterior)
                    / precio_anterior
                ) * 100

                maximo = datos["High"].max()
                minimo = datos["Low"].min()
                volumen = datos["Volume"].iloc[-1]

                st.success(
                    f"Datos encontrados para {ticker}"
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
                        "Volumen",
                        f"{volumen:,.0f}"
                    )

                st.divider()

                st.subheader("📈 Evolución del precio")

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

                figura.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Precio",
                    xaxis_rangeslider_visible=False,
                    height=550
                )

                st.plotly_chart(
                    figura,
                    use_container_width=True
                )

                st.divider()

                st.subheader("📋 Primer diagnóstico")

                if precio > datos["Close"].mean():
                    st.info(
                        "El precio actual se encuentra por encima "
                        "de la media del periodo seleccionado."
                    )
                else:
                    st.info(
                        "El precio actual se encuentra por debajo "
                        "de la media del periodo seleccionado."
                    )

                st.warning(
                    "⚠️ Este diagnóstico es únicamente una primera "
                    "lectura del mercado. Todavía no constituye una "
                    "recomendación de inversión."
                )

        except Exception as error:
            st.error(
                f"Ha ocurrido un error al obtener los datos: {error}"
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
