import streamlit as st

st.set_page_config(
    page_title="MARKET AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MARKET AI")
st.subheader("Sistema inteligente de análisis de mercados")

st.write(
    "Analiza acciones, metales y mercados para detectar "
    "posibles oportunidades de inversión."
)

st.divider()

st.header("🔍 Analizar un activo")

ticker = st.text_input(
    "Introduce el símbolo de la empresa o activo",
    placeholder="Ejemplo: AAPL, NVDA, MSFT, GOLD..."
)

if st.button("Analizar"):
    if ticker:
        st.success(f"Activo seleccionado: {ticker.upper()}")
        st.info(
            "El motor de análisis se añadirá en las siguientes fases."
        )
    else:
        st.warning("Introduce primero un activo.")

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
