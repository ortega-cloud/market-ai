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
        return None, (
            "No se ha encontrado "
            "ALPHA_VANTAGE_API_KEY "
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
            timeout=15
        )
        respuesta.raise_for_status()
        datos = respuesta.json()

        if "Note" in datos:
            return None, (
                "Alpha Vantage ha alcanzado "
                "el límite de peticiones."
            )

        if "Information" in datos:
            return None, datos["Information"]

        if "Error Message" in datos:
            return None, (
                "El ticker no es válido."
            )

        quote = datos.get("Global Quote", {})
        precio = quote.get("05. price")
        fecha = quote.get("07. latest trading day")

        if not precio:
            return None, (
                "Alpha Vantage no ha devuelto "
                "el precio."
            )

        return {
            "precio": float(precio),
            "fecha": fecha
        }, None

    except Exception as error:
        return None, str(error)

# =========================================================
# HISTÓRICOS
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

# =========================================================
# INFORMACIÓN GENERAL
# =========================================================

@st.cache_data(ttl=3600)
def obtener_info(ticker):
    try:
        empresa = yf.Ticker(ticker)
        return empresa.info or {}
    except Exception:
        return {}

# =========================================================
# OBJETIVOS DE ANALISTAS
# =========================================================

@st.cache_data(ttl=3600)
def obtener_objetivos_analistas(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.analyst_price_targets
        if datos is None:
            return {}
        if hasattr(datos, "to_dict"):
            datos = datos.to_dict()
        return datos
    except Exception:
        try:
            empresa = yf.Ticker(ticker)
            datos = empresa.get_analyst_price_targets()
            if datos is None:
                return {}
            if hasattr(datos, "to_dict"):
                datos = datos.to_dict()
            return datos
        except Exception:
            return {}

# =========================================================
# RECOMENDACIONES
# =========================================================

@st.cache_data(ttl=3600)
def obtener_recomendaciones(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.recommendations_summary
        if datos is None:
            datos = pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_historial_recomendaciones(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.recommendations
        if datos is None:
            return pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_upgrades_downgrades(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.upgrades_downgrades
        if datos is None:
            return pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()

# =========================================================
# ESTIMACIONES EPS
# =========================================================

@st.cache_data(ttl=3600)
def obtener_estimaciones_eps(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.earnings_estimate
        if datos is None:
            return pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_estimaciones_ingresos(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.revenue_estimate
        if datos is None:
            return pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_revisiones_eps(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.eps_revisions
        if datos is None:
            return pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_tendencia_eps(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.eps_trend
        if datos is None:
            return pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def obtener_crecimiento_estimado(ticker):
    try:
        empresa = yf.Ticker(ticker)
        datos = empresa.growth_estimates
        if datos is None:
            return pd.DataFrame()
        return datos
    except Exception:
        return pd.DataFrame()

# =========================================================
# NOTICIAS
# =========================================================

@st.cache_data(ttl=900)
def obtener_noticias(ticker):
    try:
        empresa = yf.Ticker(ticker)
        noticias = empresa.news
        if noticias is None:
            return []
        return noticias
    except Exception:
        return []

# =========================================================
# RSI
# =========================================================

def calcular_rsi(precios, periodo=14):
    delta = precios.diff()
    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)

    media_ganancias = ganancias.rolling(periodo).mean()
    media_perdidas = perdidas.rolling(periodo).mean()

    rs = media_ganancias / media_perdidas.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# =========================================================
# SCORES
# =========================================================

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
        if pe < 15:
            score += 8
            razones.append("PER relativamente bajo.")
        elif pe < 25:
            score += 6
            razones.append("PER en zona intermedia.")
        elif pe < 40:
            score += 3
            razones.append("PER relativamente elevado.")
        else:
            score += 1
            razones.append("PER muy elevado.")

    if forward_pe is not None:
        if forward_pe < 15:
            score += 7
        elif forward_pe < 25:
            score += 5
        elif forward_pe < 40:
            score += 2

    if peg is not None:
        if peg < 1:
            score += 6
        elif peg < 2:
            score += 4
        else:
            score += 1

    if price_to_book is not None:
        if price_to_book < 2:
            score += 4
        elif price_to_book < 5:
            score += 2
        else:
            score += 1

    return min(score, 25), razones


def calcular_score_fundamentales(roe, margen, deuda, flujo_caja):
    score = 0
    razones = []

    if roe is not None:
        if roe > 0.20:
            score += 8
            razones.append("ROE elevado.")
        elif roe > 0.10:
            score += 5
            razones.append("ROE razonable.")
        else:
            score += 2
            razones.append("ROE relativamente bajo.")

    if margen is not None:
        if margen > 0.20:
            score += 7
            razones.append("Margen de beneficios elevado.")
        elif margen > 0.10:
            score += 5
            razones.append("Margen de beneficios razonable.")
        else:
            score += 2
            razones.append("Margen de beneficios reducido.")

    if deuda is not None:
        if deuda < 50:
            score += 5
            razones.append("Deuda relativamente baja.")
        elif deuda < 100:
            score += 3
            razones.append("Deuda moderada.")
        else:
            score += 1
            razones.append("Deuda elevada.")

    if flujo_caja is not None:
        if flujo_caja > 0:
            score += 5
            razones.append("Free Cash Flow positivo.")
        else:
            razones.append("Free Cash Flow negativo.")

    return min(score, 25), razones


def calcular_score_crecimiento(crecimiento_ingresos, crecimiento_beneficios):
    score = 0
    razones = []

    if crecimiento_ingresos is not None:
        if crecimiento_ingresos > 0.15:
            score += 7
        elif crecimiento_ingresos > 0.05:
            score += 5
        elif crecimiento_ingresos > 0:
            score += 2
        else:
            razones.append("Los ingresos están decreciendo.")

    if crecimiento_beneficios is not None:
        if crecimiento_beneficios > 0.15:
            score += 8
        elif crecimiento_beneficios > 0.05:
            score += 5
        elif crecimiento_beneficios > 0:
            score += 2
        else:
            razones.append("Los beneficios están decreciendo.")

    return min(score, 15), razones


def calcular_score_riesgo(volatilidad, deuda):
    score = 0
    razones = []

    if volatilidad < 20:
        score += 6
    elif volatilidad < 35:
        score += 4
    elif volatilidad < 50:
        score += 2
    else:
        score += 1
        razones.append("Volatilidad muy elevada.")

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
# FAIR VALUE COMBINADO Y RANKING
# =========================================================

def calcular_fair_value_combinado(
    precio_actual,
    valor_dcf,
    objetivo_analistas,
    eps,
    crecimiento_beneficios
):
    valores = []
    pesos = []

    if valor_dcf is not None and valor_dcf > 0:
        valores.append(float(valor_dcf))
        pesos.append(0.50)

    if objetivo_analistas is not None and objetivo_analistas > 0:
        valores.append(float(objetivo_analistas))
        pesos.append(0.25)

    if eps is not None and eps > 0:
        if crecimiento_beneficios is not None:
            crecimiento_pct = float(crecimiento_beneficios) * 100
        else:
            crecimiento_pct = 5.0

        crecimiento_pct = max(-5.0, min(25.0, crecimiento_pct))
        pe_razonable = max(10.0, min(30.0, 15.0 + (0.50 * crecimiento_pct)))
        valor_multiplo = eps * pe_razonable

        if valor_multiplo > 0:
            valores.append(float(valor_multiplo))
            pesos.append(0.25)

    if not valores:
        return None

    peso_total = sum(pesos)
    return sum(valor * peso for valor, peso in zip(valores, pesos)) / peso_total


def diagnosticar_fair_value(precio_actual, fair_value):
    if precio_actual is None or fair_value is None or precio_actual <= 0:
        return "⚪ SIN DATOS", None

    potencial = ((fair_value - precio_actual) / precio_actual) * 100

    if potencial >= 30:
        return "🟢 MUY INFRAVALORADA", potencial
    if potencial >= 15:
        return "🟢 INFRAVALORADA", potencial
    if potencial >= -10:
        return "🟡 VALORACIÓN RAZONABLE", potencial
    if potencial >= -25:
        return "🟠 SOBREVALORADA", potencial

    return "🔴 MUY SOBREVALORADA", potencial


SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "JPM", "LLY",
    "V", "MA", "XOM", "WMT", "COST", "NFLX", "ORCL", "AMD", "ADBE", "CRM",
    "BAC", "KO", "PEP", "CSCO", "QCOM", "TXN", "GE", "CAT", "GS", "MS"
]


@st.cache_data(ttl=86400)
def obtener_componentes_sp500():
    try:
        tablas = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        simbolos = (
            tablas[0]["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        return [simbolo for simbolo in simbolos if simbolo and simbolo != "nan"]
    except Exception:
        return SP500_FALLBACK


def obtener_close_serie(datos):
    if isinstance(datos, pd.DataFrame) and "Close" in datos.columns:
        return datos["Close"].dropna()
    return pd.Series(dtype=float)


@st.cache_data(ttl=1800)
def escanear_sp500(candidatos_finales=20):
    tickers = obtener_componentes_sp500()

    try:
        precios = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True
        )
    except Exception:
        return pd.DataFrame()

    candidatos = []

    for simbolo in tickers:
        try:
            if isinstance(precios.columns, pd.MultiIndex):
                niveles = precios.columns.get_level_values(0)
                if simbolo not in niveles:
                    continue
                datos = precios[simbolo]
            else:
                datos = precios

            close = obtener_close_serie(datos)
            if len(close) < 60:
                continue

            precio = float(close.iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50
            rsi = float(calcular_rsi(close).iloc[-1])

            score_tecnico = 0
            if precio > ma20: score_tecnico += 25
            if precio > ma50: score_tecnico += 25
            if precio > ma200: score_tecnico += 25
            if 45 <= rsi <= 70: score_tecnico += 15
            elif 35 <= rsi < 45 or 70 < rsi <= 78: score_tecnico += 8

            candidatos.append((simbolo, precio, rsi, score_tecnico))
        except Exception:
            continue

    candidatos = sorted(candidatos, key=lambda x: x[3], reverse=True)[:candidatos_finales]
    resultados = []

    for simbolo, precio, rsi, score_tecnico in candidatos:
        try:
            info = obtener_info(simbolo)
            fcf = limpiar_numero(info.get("freeCashflow"))
            deuda_total = limpiar_numero(info.get("totalDebt"))
            caja_total = limpiar_numero(info.get("totalCash"))
            acciones = limpiar_numero(info.get("sharesOutstanding"))
            eps = limpiar_numero(info.get("trailingEps"))
            crecimiento = limpiar_numero(info.get("earningsGrowth"))
            objetivo = limpiar_numero(info.get("targetMeanPrice"))
            pe = limpiar_numero(info.get("trailingPE"))

            valor_dcf = None
            if fcf and fcf > 0 and acciones and acciones > 0:
                escenarios = calcular_escenarios_dcf(
                    fcf,
                    deuda=deuda_total or 0,
                    caja=caja_total or 0,
                    acciones=acciones
                )
                dcf_base = escenarios.get("base")
                if dcf_base:
                    valor_dcf = limpiar_numero(dcf_base.get("valor_por_accion"))

            fair_value = calcular_fair_value_combinado(
                precio, valor_dcf, objetivo, eps, crecimiento
            )
            estado, potencial = diagnosticar_fair_value(precio, fair_value)

            score = float(score_tecnico)
            if potencial is not None:
                score += max(-20, min(35, potencial * 0.50))
            if pe is not None:
                if pe < 20: score += 8
                elif pe < 30: score += 4
                elif pe > 50: score -= 5

            resultados.append({
                "Ticker": simbolo,
                "Empresa": info.get("shortName", simbolo),
                "Precio": precio,
                "Fair Value": fair_value,
                "Potencial %": potencial,
                "Diagnóstico": estado,
                "MARKET AI": round(max(0, min(100, score)), 1),
                "DCF": valor_dcf,
                "Objetivo analistas": objetivo,
                "P/E": pe,
                "RSI": rsi
            })
        except Exception:
            continue

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados).sort_values("MARKET AI", ascending=False).reset_index(drop=True)

# =========================================================
# DIAGNÓSTICO PREDICTIVO MARKET AI
# =========================================================

def calcular_diagnostico_predictivo(
    precio,
    fair_value,
    score_total,
    score_tecnico,
    score_valoracion,
    score_fundamentales,
    score_crecimiento,
    score_riesgo,
    rsi,
    ma20,
    ma50,
    ma200,
    objetivo_analistas=None
):

    señales = []
    riesgos = []

    # =====================================================
    # VALORACIÓN
    # =====================================================

    potencial_fair = None

    if (
        precio is not None
        and fair_value is not None
        and precio > 0
    ):

        potencial_fair = (
            (fair_value - precio)
            / precio
        ) * 100

        if potencial_fair >= 20:
            señales.append(
                "La acción presenta un descuento importante "
                "respecto al Fair Value."
            )
        elif potencial_fair >= 10:
            señales.append(
                "La acción presenta un descuento moderado "
                "respecto al Fair Value."
            )
        elif potencial_fair <= -20:
            riesgos.append(
                "El precio está muy por encima "
                "del Fair Value estimado."
            )
        elif potencial_fair <= -10:
            riesgos.append(
                "El precio está por encima "
                "del Fair Value estimado."
            )

    # =====================================================
    # TENDENCIA
    # =====================================================

    tendencia = "LATERAL"

    if (
        ma20 is not None
        and ma50 is not None
        and ma200 is not None
    ):
        if (
            precio > ma20
            and ma20 > ma50
            and ma50 > ma200
        ):
            tendencia = "ALCISTA"
            señales.append(
                "La estructura de medias móviles "
                "confirma una tendencia alcista."
            )
        elif (
            precio < ma20
            and ma20 < ma50
            and ma50 < ma200
        ):
            tendencia = "BAJISTA"
            riesgos.append(
                "Las medias móviles muestran "
                "una estructura bajista."
            )
        else:
            tendencia = "LATERAL"
            señales.append(
                "Las medias móviles no muestran "
                "una tendencia claramente definida."
            )

    # =====================================================
    # RSI
    # =====================================================

    situacion_rsi = "NEUTRAL"

    if rsi is not None:
        if rsi >= 75:
            situacion_rsi = "SOBRECOMPRA"
            riesgos.append(
                "El RSI indica una situación "
                "de sobrecompra."
            )
        elif rsi <= 30:
            situacion_rsi = "SOBREVENTA"
            señales.append(
                "El RSI indica una situación "
                "de sobreventa."
            )
        elif 45 <= rsi <= 70:
            situacion_rsi = "SALUDABLE"
            señales.append(
                "El RSI se encuentra en una zona "
                "relativamente saludable."
            )

    # =====================================================
    # FUNDAMENTALES
    # =====================================================

    if score_fundamentales >= 20:
        señales.append("Los fundamentales son sólidos.")
    elif score_fundamentales < 12:
        riesgos.append("Los fundamentales presentan varias señales débiles.")

    # =====================================================
    # CRECIMIENTO
    # =====================================================

    if score_crecimiento >= 12:
        señales.append(
            "El crecimiento de ingresos y beneficios "
            "es favorable."
        )
    elif score_crecimiento < 7:
        riesgos.append(
            "El crecimiento de la empresa "
            "es limitado o débil."
        )

    # =====================================================
    # RIESGO
    # =====================================================

    if score_riesgo <= 4:
        riesgos.append(
            "El nivel de riesgo estimado "
            "es relativamente elevado."
        )
    elif score_riesgo >= 8:
        señales.append("El perfil de riesgo es relativamente favorable.")

    # =====================================================
    # OBJETIVO DE ANALISTAS
    # =====================================================

    potencial_analistas = None

    if (
        objetivo_analistas is not None
        and precio is not None
        and precio > 0
    ):
        potencial_analistas = (
            (objetivo_analistas - precio) / precio
        ) * 100

        if potencial_analistas >= 15:
            señales.append(
                "El objetivo medio de los analistas "
                "implica un potencial relevante."
            )
        elif potencial_analistas <= -10:
            riesgos.append(
                "El objetivo medio de los analistas "
                "está por debajo del precio actual."
            )

    # =====================================================
    # DIRECCIÓN PROBABLE
    # =====================================================

    puntos_alcistas = 0
    puntos_bajistas = 0

    if tendencia == "ALCISTA":
        puntos_alcistas += 3
    elif tendencia == "BAJISTA":
        puntos_bajistas += 3

    if potencial_fair is not None:
        if potencial_fair >= 15:
            puntos_alcistas += 3
        elif potencial_fair >= 5:
            puntos_alcistas += 1
        elif potencial_fair <= -15:
            puntos_bajistas += 3
        elif potencial_fair <= -5:
            puntos_bajistas += 1

    if score_fundamentales >= 20:
        puntos_alcistas += 2
    elif score_fundamentales < 12:
        puntos_bajistas += 2

    if score_crecimiento >= 12:
        puntos_alcistas += 2
    elif score_crecimiento < 7:
        puntos_bajistas += 1

    if rsi is not None:
        if rsi <= 30:
            puntos_alcistas += 1
        elif rsi >= 75:
            puntos_bajistas += 1

    if potencial_analistas is not None:
        if potencial_analistas >= 15:
            puntos_alcistas += 2
        elif potencial_analistas <= -10:
            puntos_bajistas += 2

    if puntos_alcistas >= puntos_bajistas + 3:
        direccion = "🟢 ALCISTA"
    elif puntos_bajistas >= puntos_alcistas + 3:
        direccion = "🔴 BAJISTA"
    else:
        direccion = "🟡 LATERAL / INCIERTA"

    # =====================================================
    # HORIZONTE
    # =====================================================

    if (
        potencial_fair is not None
        and potencial_fair >= 20
        and tendencia == "ALCISTA"
    ):
        horizonte = "3–9 meses"
    elif (
        potencial_fair is not None
        and potencial_fair >= 10
    ):
        horizonte = "3–12 meses"
    elif (
        tendencia == "ALCISTA"
        and score_total >= 70
    ):
        horizonte = "1–6 meses"
    elif tendencia == "BAJISTA":
        horizonte = "1–6 meses"
    else:
        horizonte = "1–12 meses"

    # =====================================================
    # SEÑAL FINAL
    # =====================================================

    if score_total >= 85:
        señal = "🟢 COMPRA"
    elif score_total >= 70:
        señal = "🟢 COMPRA MODERADA"
    elif score_total >= 55:
        señal = "🟡 MANTENER"
    elif score_total >= 40:
        señal = "🟠 ESPERAR"
    else:
        señal = "🔴 EVITAR"

    # Ajuste por valoración extrema
    if (
        potencial_fair is not None
        and potencial_fair <= -25
        and señal in ["🟢 COMPRA", "🟢 COMPRA MODERADA"]
    ):
        señal = "🟡 MANTENER"

    return {
        "direccion": direccion,
        "tendencia": tendencia,
        "rsi": situacion_rsi,
        "horizonte": horizonte,
        "señal": señal,
        "potencial_fair": potencial_fair,
        "potencial_analistas": potencial_analistas,
        "señales": señales,
        "riesgos": riesgos,
        "puntos_alcistas": puntos_alcistas,
        "puntos_bajistas": puntos_bajistas
    }

# =========================================================
# INTERFAZ (SIDEBAR)
# =========================================================

st.sidebar.header("⚙️ Configuración")

ticker = st.sidebar.text_input("Símbolo", value="AAPL").upper().strip()
periodo = st.sidebar.selectbox("Periodo", ["6mo", "1y", "2y", "5y", "10y"], index=1)

st.sidebar.divider()
st.sidebar.subheader("🎯 Precio para el análisis")

tipo_precio = st.sidebar.radio("Fuente", ["Precio automático", "Precio personalizado"])
precio_personalizado = None

if tipo_precio == "Precio personalizado":
    precio_personalizado = st.sidebar.number_input(
        "Precio de entrada", min_value=0.01, value=100.00, step=1.00
    )

analizar = st.sidebar.button("📊 ANALIZAR", type="primary")
ranking_sp500 = st.sidebar.button("🏆 TOP 5 S&P 500")

st.info("Introduce un ticker y pulsa ANALIZAR.")

# =========================================================
# RANKING AUTOMÁTICO S&P 500
# =========================================================

if ranking_sp500:
    st.divider()
    st.header("🏆 TOP 5 — S&P 500")
    st.caption(
        "MARKET AI realiza una primera criba técnica y posteriormente analiza "
        "valoración, DCF, fundamentales y objetivos de analistas."
    )

    with st.spinner("Escaneando el S&P 500... La primera ejecución puede tardar unos minutos."):
        ranking = escanear_sp500(candidatos_finales=20)

    if ranking.empty:
        st.error(
            "No se ha podido completar el escaneo. Puede deberse a un límite temporal "
            "de la fuente de datos."
        )
    else:
        st.success(f"Se han analizado {len(ranking)} candidatos finales.")
        st.subheader("🥇 Las 5 mejores oportunidades")

        top5 = ranking.head(5)
        for posicion, (_, fila) in enumerate(top5.iterrows(), start=1):
            st.markdown(f"### {posicion}. {fila['Ticker']} — {fila['Empresa']}")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Precio", f"${fila['Precio']:,.2f}")
            with c2:
                st.metric(
                    "Fair Value",
                    f"${fila['Fair Value']:,.2f}" if pd.notna(fila["Fair Value"]) else "N/D"
                )
            with c3:
                st.metric(
                    "Potencial",
                    f"{fila['Potencial %']:+.1f}%" if pd.notna(fila["Potencial %"]) else "N/D"
                )
            with c4:
                st.metric("MARKET AI", f"{fila['MARKET AI']:.1f}/100")

            st.write(f"**Diagnóstico:** {fila['Diagnóstico']}  |  **RSI:** {fila['RSI']:.1f}")
            st.divider()

        st.subheader("📊 Ranking completo")
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.warning("⚠️ Este ranking es experimental y no garantiza rentabilidad.")

# =========================================================
# ANÁLISIS DE TICKER
# =========================================================

if analizar:
    if not ticker:
        st.error("Debes introducir un ticker.")
        st.stop()

    # Precios
    with st.spinner("Obteniendo precio..."):
        precio_alpha, error_alpha = obtener_precio_alpha_vantage(ticker)

    if precio_alpha:
        precio_mercado = precio_alpha["precio"]
        fecha_precio = precio_alpha["fecha"]
        st.info(f"💵 Precio: **${precio_mercado:,.2f}** | Alpha Vantage | {fecha_precio}")
    else:
        precio_mercado = None
        st.warning("Alpha Vantage no ha proporcionado el precio.")
        st.caption(str(error_alpha))

    if tipo_precio == "Precio personalizado" and precio_personalizado is not None:
        precio_analisis = float(precio_personalizado)
    elif precio_mercado is not None:
        precio_analisis = float(precio_mercado)
    else:
        precio_analisis = None

    # Extracción de Datos
    with st.spinner("Analizando mercado..."):
        datos = obtener_precios(ticker, periodo)
        info = obtener_info(ticker)
        objetivos = obtener_objetivos_analistas(ticker)
        recomendaciones = obtener_recomendaciones(ticker)
        historial_recomendaciones = obtener_historial_recomendaciones(ticker)
        upgrades = obtener_upgrades_downgrades(ticker)
        eps_estimaciones = obtener_estimaciones_eps(ticker)
        ingresos_estimaciones = obtener_estimaciones_ingresos(ticker)
        revisiones_eps = obtener_revisiones_eps(ticker)
        tendencia_eps = obtener_tendencia_eps(ticker)
        crecimiento_estimado = obtener_crecimiento_estimado(ticker)
        noticias = obtener_noticias(ticker)

    if datos.empty:
        st.error(f"No se han encontrado datos para {ticker}.")
        st.stop()

    # Cálculo técnico
    precio_historico = float(datos["Close"].iloc[-1])
    precio_anterior = float(datos["Close"].iloc[-2])
    variacion = ((precio_historico - precio_anterior) / precio_anterior) * 100
    maximo = float(datos["High"].max())
    minimo = float(datos["Low"].min())

    datos["MA20"] = datos["Close"].rolling(20).mean()
    datos["MA50"] = datos["Close"].rolling(50).mean()
    datos["MA200"] = datos["Close"].rolling(200).mean()

    ma20 = datos["MA20"].iloc[-1]
    ma50 = datos["MA50"].iloc[-1]
    ma200 = datos["MA200"].iloc[-1]

    datos["RSI"] = calcular_rsi(datos["Close"])
    rsi = float(datos["RSI"].iloc[-1])

    retornos = datos["Close"].pct_change()
    volatilidad = retornos.std() * np.sqrt(252) * 100

    # Fundamentales
    nombre = info.get("longName", ticker)
    sector = info.get("sector", "N/D")
    industria = info.get("industry", "N/D")
    capitalizacion = limpiar_numero(info.get("marketCap"))
    pe = limpiar_numero(info.get("trailingPE"))
    forward_pe = limpiar_numero(info.get("forwardPE"))
    peg = limpiar_numero(info.get("pegRatio"))
    price_to_book = limpiar_numero(info.get("priceToBook"))
    roe = limpiar_numero(info.get("returnOnEquity"))
    margen = limpiar_numero(info.get("profitMargins"))
    margen_operativo = limpiar_numero(info.get("operatingMargins"))
    deuda = limpiar_numero(info.get("debtToEquity"))
    flujo_caja = limpiar_numero(info.get("freeCashflow"))
    deuda_total = limpiar_numero(info.get("totalDebt"))
    caja_total = limpiar_numero(info.get("totalCash"))
    acciones_en_circulacion = limpiar_numero(info.get("sharesOutstanding"))
    ingresos = limpiar_numero(info.get("totalRevenue"))
    beneficio = limpiar_numero(info.get("netIncomeToCommon"))
    eps = limpiar_numero(info.get("trailingEps"))
    crecimiento_ingresos = limpiar_numero(info.get("revenueGrowth"))
    crecimiento_beneficios = limpiar_numero(info.get("earningsGrowth"))
    dividend_yield = limpiar_numero(info.get("dividendYield"))

    # Scores
    score_tecnico, razones_tecnico = calcular_score_tecnico(precio_historico, ma20, ma50, ma200, rsi)
    score_valoracion, razones_valoracion = calcular_score_valoracion(pe, forward_pe, peg, price_to_book)
    score_fundamentales, razones_fundamentales = calcular_score_fundamentales(roe, margen, deuda, flujo_caja)
    score_crecimiento, razones_crecimiento = calcular_score_crecimiento(crecimiento_ingresos, crecimiento_beneficios)
    score_riesgo, razones_riesgo = calcular_score_riesgo(volatilidad, deuda)

    score_total = score_tecnico + score_valoracion + score_fundamentales + score_crecimiento + score_riesgo

    # Renderizado: Cabecera y Mercado
    st.success(f"✅ {nombre}")
    st.write(f"**Sector:** {sector}  |  **Industria:** {industria}")

    st.divider()
    st.header("💵 Situación de mercado")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Precio", f"${precio_historico:,.2f}", f"{variacion:+.2f}%")
    with c2:
        st.metric("Máximo periodo", f"${maximo:,.2f}")
    with c3:
        st.metric("Mínimo periodo", f"${minimo:,.2f}")
    with c4:
        st.metric("RSI", f"{rsi:.1f}")

    # Renderizado: MARKET AI Score
    st.divider()
    st.header("🎯 MARKET AI SCORE")

    if score_total >= 85: estado_score = "🟢 OPORTUNIDAD MUY INTERESANTE"
    elif score_total >= 70: estado_score = "🟢 OPORTUNIDAD INTERESANTE"
    elif score_total >= 55: estado_score = "🟡 NEUTRAL"
    elif score_total >= 40: estado_score = "🟠 RIESGO ELEVADO"
    else: estado_score = "🔴 POCO ATRACTIVA"

    c1, c2 = st.columns(2)
    with c1: st.metric("Puntuación", f"{score_total}/100")
    with c2: st.subheader(estado_score)

    st.progress(score_total / 100)

    st.subheader("📊 Desglose")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Técnica", f"{score_tecnico}/25")
    with c2: st.metric("Valoración", f"{score_valoracion}/25")
    with c3: st.metric("Fundamentales", f"{score_fundamentales}/25")
    with c4: st.metric("Crecimiento", f"{score_crecimiento}/15")
    with c5: st.metric("Riesgo", f"{score_riesgo}/10")

    # Renderizado: Fundamentales
    st.divider()
    st.header("📊 Fundamentales")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("PER", f"{pe:.2f}" if pe is not None else "N/D")
    with c2: st.metric("PER futuro", f"{forward_pe:.2f}" if forward_pe is not None else "N/D")
    with c3: st.metric("PEG", f"{peg:.2f}" if peg is not None else "N/D")
    with c4: st.metric("Price/Book", f"{price_to_book:.2f}" if price_to_book is not None else "N/D")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ROE", f"{roe * 100:.2f}%" if roe is not None else "N/D")
    with c2: st.metric("Margen beneficio", f"{margen * 100:.2f}%" if margen is not None else "N/D")
    with c3: st.metric("Margen operativo", f"{margen_operativo * 100:.2f}%" if margen_operativo is not None else "N/D")
    with c4: st.metric("Deuda/Patrimonio", f"{deuda:.1f}" if deuda is not None else "N/D")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Free Cash Flow", f"${flujo_caja / 1e9:.2f} B" if flujo_caja is not None else "N/D")
    with c2: st.metric("Ingresos", f"${ingresos / 1e9:.2f} B" if ingresos is not None else "N/D")
    with c3: st.metric("Beneficio neto", f"${beneficio / 1e9:.2f} B" if beneficio is not None else "N/D")
    with c4: st.metric("EPS", f"${eps:.2f}" if eps is not None else "N/D")

    # Renderizado: Crecimiento
    st.divider()
    st.header("🚀 Crecimiento")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Crecimiento ingresos", f"{crecimiento_ingresos * 100:.2f}%" if crecimiento_ingresos is not None else "N/D")
    with c2: st.metric("Crecimiento beneficios", f"{crecimiento_beneficios * 100:.2f}%" if crecimiento_beneficios is not None else "N/D")
    with c3: st.metric("Dividend Yield", f"{dividend_yield * 100:.2f}%" if dividend_yield is not None else "N/D")

    # Renderizado: Analistas
    st.divider()
    st.header("🎯 Analistas")

    objetivo_bajo = limpiar_numero(obtener_valor(objetivos, ["low"]))
    objetivo_medio = limpiar_numero(obtener_valor(objetivos, ["mean"]))
    objetivo_mediano = limpiar_numero(obtener_valor(objetivos, ["median"]))
    objetivo_alto = limpiar_numero(obtener_valor(objetivos, ["high"]))

    if objetivo_medio is not None:
        precio_referencia = precio_analisis if precio_analisis is not None else precio_historico
        potencial_analistas = ((objetivo_medio - precio_referencia) / precio_referencia) * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Objetivo bajo", f"${objetivo_bajo:,.2f}" if objetivo_bajo is not None else "N/D")
        with c2: st.metric("Objetivo medio", f"${objetivo_medio:,.2f}")
        with c3: st.metric("Mediana", f"${objetivo_mediano:,.2f}" if objetivo_mediano is not None else "N/D")
        with c4: st.metric("Objetivo alto", f"${objetivo_alto:,.2f}" if objetivo_alto is not None else "N/D")

        if potencial_analistas > 15:
            st.success(f"📈 Los analistas ven un potencial medio de **{potencial_analistas:+.2f}%**.")
        elif potencial_analistas > 0:
            st.info(f"📈 Los analistas ven un potencial medio de **{potencial_analistas:+.2f}%**.")
        else:
            st.warning(f"📉 El objetivo medio está {abs(potencial_analistas):.2f}% por debajo del precio de referencia.")
    else:
        st.info("No se han encontrado objetivos de analistas para este ticker.")

    # Consenso Buy / Sell
    st.subheader("🧑‍💼 Consenso de analistas")
    if isinstance(recomendaciones, pd.DataFrame) and not recomendaciones.empty:
        tabla = recomendaciones.copy()
        columnas = ["strongBuy", "buy", "hold", "sell", "strongSell"]
        disponibles = [columna for columna in columnas if columna in tabla.columns]

        if disponibles:
            fila = tabla.iloc[-1]
            c1, c2, c3, c4, c5 = st.columns(5)
            valores = [
                ("🟢 Strong Buy", "strongBuy"),
                ("🟢 Buy", "buy"),
                ("🟡 Hold", "hold"),
                ("🔴 Sell", "sell"),
                ("🔴 Strong Sell", "strongSell")
            ]
            for columna_ui, (nombre_ui, clave) in zip([c1, c2, c3, c4, c5], valores):
                with columna_ui:
                    val = int(fila[clave]) if clave in tabla.columns and pd.notna(fila[clave]) else 0
                    st.metric(nombre_ui, val)

            strong_buy = int(fila.get("strongBuy", 0))
            buy = int(fila.get("buy", 0))
            hold = int(fila.get("hold", 0))
            sell = int(fila.get("sell", 0))
            strong_sell = int(fila.get("strongSell", 0))
            total_analistas = strong_buy + buy + hold + sell + strong_sell

            if total_analistas > 0:
                consenso = (strong_buy * 5 + buy * 4 + hold * 3 + sell * 2 + strong_sell * 1) / total_analistas
                if consenso >= 4.2: texto_consenso = "🟢 CONSENSO MUY POSITIVO"
                elif consenso >= 3.5: texto_consenso = "🟢 CONSENSO POSITIVO"
                elif consenso >= 2.7: texto_consenso = "🟡 CONSENSO NEUTRAL"
                elif consenso >= 2: texto_consenso = "🟠 CONSENSO NEGATIVO"
                else: texto_consenso = "🔴 CONSENSO MUY NEGATIVO"
                st.write(f"**{texto_consenso}** — {total_analistas} analistas.")
    else:
        st.info("No hay resumen de recomendaciones disponible.")

    # Upgrades / Downgrades
    st.subheader("🔄 Cambios recientes de analistas")
    if isinstance(upgrades, pd.DataFrame) and not upgrades.empty:
        st.dataframe(upgrades.tail(10).reset_index(), use_container_width=True, hide_index=True)
    else:
        st.info("No hay cambios recientes disponibles.")

    # Estimaciones
    st.divider()
    st.header("🔮 Estimaciones de analistas")
    if isinstance(eps_estimaciones, pd.DataFrame) and not eps_estimaciones.empty:
        st.dataframe(eps_estimaciones, use_container_width=True)
    else:
        st.info("No hay estimaciones EPS disponibles.")

    if isinstance(ingresos_estimaciones, pd.DataFrame) and not ingresos_estimaciones.empty:
        st.subheader("💰 Estimaciones de ingresos")
        st.dataframe(ingresos_estimaciones, use_container_width=True)

    if isinstance(revisiones_eps, pd.DataFrame) and not revisiones_eps.empty:
        st.subheader("📐 Revisiones de EPS")
        st.dataframe(revisiones_eps, use_container_width=True)

    if isinstance(tendencia_eps, pd.DataFrame) and not tendencia_eps.empty:
        st.subheader("📈 Tendencia de EPS")
        st.dataframe(tendencia_eps, use_container_width=True)

    if isinstance(crecimiento_estimado, pd.DataFrame) and not crecimiento_estimado.empty:
        st.subheader("🚀 Crecimiento estimado")
        st.dataframe(crecimiento_estimado, use_container_width=True)

    # Valoración DCF
    st.divider()
    st.header("💎 Valoración DCF")
    st.caption("Estimación experimental del valor razonable mediante un modelo de Discounted Cash Flow (DCF).")

    valor_base = None
    if flujo_caja and flujo_caja > 0 and acciones_en_circulacion and acciones_en_circulacion > 0:
        escenarios_dcf = calcular_escenarios_dcf(
            free_cash_flow=flujo_caja,
            deuda=deuda_total if deuda_total is not None else 0,
            caja=caja_total if caja_total is not None else 0,
            acciones=acciones_en_circulacion,
        )

        dcf_pesimista = escenarios_dcf.get("pesimista")
        dcf_base = escenarios_dcf.get("base")
        dcf_optimista = escenarios_dcf.get("optimista")

        valor_pesimista = dcf_pesimista.get("valor_por_accion") if dcf_pesimista else None
        valor_base = dcf_base.get("valor_por_accion") if dcf_base else None
        valor_optimista = dcf_optimista.get("valor_por_accion") if dcf_optimista else None

        if precio_analisis is not None and valor_base is not None:
            estado_dcf, potencial_dcf, explicacion_dcf = diagnosticar_valoracion(precio_analisis, valor_base)
        else:
            estado_dcf = "⚪ SIN DATOS"
            potencial_dcf = None
            explicacion_dcf = "No hay precio actual o valor DCF base suficiente para realizar la comparación."

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🔴 Pesimista", f"${valor_pesimista:,.2f}" if valor_pesimista else "N/D")
        with c2: st.metric("🟡 Base", f"${valor_base:,.2f}" if valor_base else "N/D")
        with c3: st.metric("🟢 Optimista", f"${valor_optimista:,.2f}" if valor_optimista else "N/D")

        st.subheader("🧠 Diagnóstico de valoración")
        st.write(f"### {estado_dcf}")
        st.write(explicacion_dcf)

        if potencial_dcf is not None:
            st.metric("Potencial estimado hasta el valor DCF base", f"{potencial_dcf:+.2f}%")

        with st.expander("🔎 Ver parámetros utilizados por el DCF"):
            st.write(f"**Free Cash Flow:** ${flujo_caja / 1e9:.2f} B")
            st.write(f"**Deuda:** ${(deuda_total or 0) / 1e9:.2f} B")
            st.write(f"**Caja:** ${(caja_total or 0) / 1e9:.2f} B")
            st.write(f"**Acciones en circulación:** {acciones_en_circulacion / 1e9:.2f} B")
            st.write("**Horizonte:** 5 años")
            st.write("**Base:** crecimiento inicial 8%, crecimiento terminal 3% y tasa de descuento 10%.")
    else:
        st.info("No hay suficiente Free Cash Flow o acciones en circulación para calcular el DCF de forma fiable.")

    # Fair Value Combinado
    st.divider()
    st.header("💠 MARKET AI FAIR VALUE")
    st.caption("Estimación combinada de DCF, objetivo de analistas y valoración por beneficios.")

    fair_value = calcular_fair_value_combinado(
        precio_analisis, valor_base, objetivo_medio, eps, crecimiento_beneficios
    )
    estado_fair, potencial_fair = diagnosticar_fair_value(precio_analisis, fair_value)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Precio actual", f"${precio_analisis:,.2f}" if isinstance(precio_analisis, (int, float)) else "N/D")
    with c2:
        st.metric("Fair Value", f"${fair_value:,.2f}" if isinstance(fair_value, (int, float)) else "N/D")
    with c3:
        st.metric("Potencial", f"{potencial_fair:+.2f}%" if isinstance(potencial_fair, (int, float)) else "N/D")

    st.write(f"### {estado_fair}")
    st.info("El Fair Value es una estimación experimental y no constituye asesoramiento financiero.")

    # =====================================================
    # DIAGNÓSTICO PREDICTIVO (BLOQUE 2 INTEGRADO)
    # =====================================================

    st.divider()
    st.header("🔮 Diagnóstico predictivo MARKET AI")

    diagnostico = calcular_diagnostico_predictivo(
        precio_analisis,
        fair_value,
        score_total,
        score_tecnico,
        score_valoracion,
        score_fundamentales,
        score_crecimiento,
        score_riesgo,
        rsi,
        ma20,
        ma50,
        ma200,
        objetivo_medio
    )

    # RESUMEN PRINCIPAL
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dirección probable", diagnostico["direccion"])
    with col2:
        st.metric("Señal", diagnostico["señal"])
    with col3:
        st.metric("Horizonte", diagnostico["horizonte"])
    with col4:
        potencial = diagnostico["potencial_fair"]
        st.metric(
            "Potencial Fair Value",
            f"{potencial:+.2f}%" if potencial is not None else "N/D"
        )

    # DIAGNÓSTICO EN LENGUAJE NATURAL
    st.subheader("🧠 ¿Qué está viendo MARKET AI?")
    direccion_texto = diagnostico["direccion"]
    tendencia_texto = diagnostico["tendencia"]
    rsi_texto = diagnostico["rsi"]
    señal_texto = diagnostico["señal"]
    horizonte_texto = diagnostico["horizonte"]

    if potencial is not None:
        potencial_texto = f"{potencial:+.1f}%"
    else:
        potencial_texto = "no disponible"

    st.write(f"""
**{ticker}** presenta actualmente una dirección probable **{direccion_texto}**.

La tendencia técnica es **{tendencia_texto}** y el RSI se encuentra en una situación **{rsi_texto}**.

El Fair Value estima un potencial de **{potencial_texto}** respecto al precio actual.

Teniendo en cuenta el conjunto de factores, MARKET AI establece actualmente una señal de **{señal_texto}**.

El horizonte orientativo del modelo es de **{horizonte_texto}**.
""")

    # SEÑALES POSITIVAS
    if diagnostico["señales"]:
        st.subheader("🟢 Factores favorables")
        for razon in diagnostico["señales"]:
            st.write(f"✓ {razon}")

    # RIESGOS
    if diagnostico["riesgos"]:
        st.subheader("⚠️ Riesgos detectados")
        for riesgo in diagnostico["riesgos"]:
            st.write(f"• {riesgo}")

    # FUERZA DE LA SEÑAL
    st.subheader("⚖️ Balance de señales")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Presión alcista", diagnostico["puntos_alcistas"])
    with c2:
        st.metric("Presión bajista", diagnostico["puntos_bajistas"])

    # Diagnóstico General Estático
    st.divider()
    st.header("🧠 Diagnóstico de MARKET AI")

    if score_total >= 70:
        st.write(f"**{ticker} presenta actualmente una configuración relativamente favorable**, con una puntuación de **{score_total}/100**.")
    elif score_total >= 55:
        st.write(f"**{ticker} presenta una situación mixta**, con una puntuación de **{score_total}/100**.")
    else:
        st.write(f"**{ticker} presenta actualmente varias señales desfavorables**, con una puntuación de **{score_total}/100**.")

    razones = razones_tecnico + razones_valoracion + razones_fundamentales + razones_crecimiento + razones_riesgo
    for razon in razones:
        st.write(f"• {razon}")

    # Gráficos
    st.divider()
    st.header("📈 Gráfico")

    figura = go.Figure()
    figura.add_trace(go.Candlestick(
        x=datos.index, open=datos["Open"], high=datos["High"], low=datos["Low"], close=datos["Close"], name=ticker
    ))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA20"], name="MA20"))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA50"], name="MA50"))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA200"], name="MA200"))
    figura.update_layout(xaxis_rangeslider_visible=False, height=600)

    st.plotly_chart(figura, use_container_width=True)

    st.header("📉 RSI")
    figura_rsi = go.Figure()
    figura_rsi.add_trace(go.Scatter(x=datos.index, y=datos["RSI"], name="RSI"))
    figura_rsi.add_hline(y=70, line_dash="dash")
    figura_rsi.add_hline(y=30, line_dash="dash")
    figura_rsi.update_layout(height=350)

    st.plotly_chart(figura_rsi, use_container_width=True)

    # Noticias
    st.divider()
    st.header("📰 Noticias recientes")

    if noticias:
        contador = 0
        for noticia in noticias:
            if contador >= 8: break
            try:
                contenido = noticia.get("content", noticia)
                titulo = contenido.get("title", "Sin título")
                resumen = contenido.get("summary", "")
                enlace = contenido.get("canonicalUrl", {})
                url = enlace.get("url", "") if isinstance(enlace, dict) else str(enlace)

                st.markdown(f"### {titulo}")
                if resumen: st.write(resumen)
                if url: st.markdown(f"[Leer noticia]({url})")
                contador += 1
            except Exception:
                continue
    else:
        st.info("No hay noticias disponibles.")

    # Comparación Precio Personalizado
    if tipo_precio == "Precio personalizado" and precio_personalizado is not None and precio_mercado is not None:
        st.divider()
        st.header("🎯 Precio personalizado")
        diferencia = ((precio_personalizado - precio_mercado) / precio_mercado) * 100

        if precio_personalizado < precio_mercado:
            st.success(f"El precio indicado está {abs(diferencia):.2f}% por debajo del mercado.")
        elif precio_personalizado > precio_mercado:
            st.warning(f"El precio indicado está {diferencia:.2f}% por encima del mercado.")
        else:
            st.info("El precio indicado coincide con el precio actual.")

    # Aviso Legal
    st.divider()
    st.warning(
        "⚠️ MARKET AI es un modelo experimental. Los datos, puntuaciones y estimaciones "
        "no constituyen asesoramiento financiero y no deben utilizarse por sí solos "
        "para tomar decisiones de inversión."
    )
