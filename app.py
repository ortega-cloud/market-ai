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

st.title("🤖 MARKET AI")
st.caption("Sistema experimental con arquitectura multifuente de resiliencia de datos.")

# =========================================================
# FUNCIONES AUXILIARES DE LIMPIEZA
# =========================================================

def limpiar_numero(valor):
    try:
        if valor is None or pd.isna(valor) or valor == "N/A" or valor == "":
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

@st.cache_data(ttl=3600)
def obtener_precio_alpha_vantage(ticker, api_key="DEMO"):
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

        if "Note" in datos or "Information" in datos or "Error Message" in datos:
            return None, "Límite de API alcanzado o ticker inválido."

        precio = datos.get("Global Quote", {}).get("05. price")
        fecha = datos.get("Global Quote", {}).get("07. latest trading day")

        if not precio:
            return None, "Alpha Vantage no ha devuelto el precio."

        return {"precio": float(precio), "fecha": fecha}, None
    except Exception as error:
        return None, str(error)

# =========================================================
# EXTRACCIÓN ROBUSTA MULTIFUENTE (YAHOO DIRECTO + YFINANCE)
# =========================================================

@st.cache_data(ttl=3600)
def obtener_info(ticker):
    info = {}
    empresa = yf.Ticker(ticker)

    # 1. FUENTE PRINCIPAL: yfinance standard
    try:
        if isinstance(empresa.info, dict):
            info.update(empresa.info)
    except Exception:
        pass

    try:
        fast = empresa.fast_info
        if fast:
            mapa_fast = {
                "currentPrice": "last_price",
                "marketCap": "market_cap",
                "sharesOutstanding": "shares"
            }
            for destino, origen in mapa_fast.items():
                if info.get(destino) is None:
                    valor = getattr(fast, origen, None)
                    if valor is not None: info[destino] = valor
    except Exception:
        pass

    # 2. FUENTE SECUNDARIA: Yahoo Finance API Directa (Endpoints v10)
    campos_criticos = ["trailingPE", "forwardPE", "pegRatio", "priceToBook", "targetMeanPrice", "returnOnEquity", "profitMargins", "freeCashflow"]
    faltan_datos = any(info.get(k) is None for k in campos_criticos)

    if faltan_datos:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=summaryDetail,financialData,defaultKeyStatistics"
            r = requests.get(url, headers=headers, timeout=5)
            
            if r.status_code == 200:
                resultado = r.json().get("quoteSummary", {}).get("result", [])
                if resultado:
                    datos_crudos = resultado[0]
                    for modulo in ["summaryDetail", "financialData", "defaultKeyStatistics"]:
                        data_mod = datos_crudos.get(modulo, {})
                        for key, val in data_mod.items():
                            if info.get(key) is None and isinstance(val, dict) and "raw" in val:
                                info[key] = val["raw"]
        except Exception:
            pass

    # 3. EXTRACCIÓN DE ESTADOS FINANCIEROS YFINANCE
    try:
        income = empresa.get_income_stmt(freq="yearly")
        if isinstance(income, pd.DataFrame) and not income.empty:
            col = income.columns[0]
            if info.get("totalRevenue") is None and "TotalRevenue" in income.index: info["totalRevenue"] = limpiar_numero(income.loc["TotalRevenue", col])
            if info.get("netIncomeToCommon") is None and "NetIncome" in income.index: info["netIncomeToCommon"] = limpiar_numero(income.loc["NetIncome", col])
            if info.get("operatingIncome") is None and "OperatingIncome" in income.index: info["operatingIncome"] = limpiar_numero(income.loc["OperatingIncome", col])
    except Exception:
        pass

    try:
        balance = empresa.get_balance_sheet(freq="yearly")
        if isinstance(balance, pd.DataFrame) and not balance.empty:
            col = balance.columns[0]
            if info.get("totalDebt") is None and "TotalDebt" in balance.index: info["totalDebt"] = limpiar_numero(balance.loc["TotalDebt", col])
            if info.get("totalCash") is None and "CashAndCashEquivalents" in balance.index: info["totalCash"] = limpiar_numero(balance.loc["CashAndCashEquivalents", col])
            if info.get("stockholdersEquity") is None and "StockholdersEquity" in balance.index: info["stockholdersEquity"] = limpiar_numero(balance.loc["StockholdersEquity", col])
    except Exception:
        pass

    try:
        cashflow = empresa.get_cash_flow(freq="yearly")
        if isinstance(cashflow, pd.DataFrame) and not cashflow.empty:
            col = cashflow.columns[0]
            if info.get("freeCashflow") is None and "FreeCashFlow" in cashflow.index:
                info["freeCashflow"] = limpiar_numero(cashflow.loc["FreeCashFlow", col])
            elif info.get("freeCashflow") is None and "OperatingCashFlow" in cashflow.index:
                fcf = limpiar_numero(cashflow.loc["OperatingCashFlow", col])
                capex = limpiar_numero(cashflow.loc.get("CapitalExpenditure", col)) or 0
                info["freeCashflow"] = fcf + capex
    except Exception:
        pass

    # 4. VALORACIÓN Y RATIOS CON FALLBACKS OFICIALES
    pe = limpiar_numero(info.get("trailingPE"))
    forward_pe = limpiar_numero(info.get("forwardPE"))
    peg = limpiar_numero(info.get("pegRatio"))
    price_to_book = limpiar_numero(info.get("priceToBook"))

    try:
        valuation = empresa.get_valuation_measures(freq="trailing", periods=0)
        if isinstance(valuation, pd.DataFrame) and not valuation.empty:
            col = "Current" if "Current" in valuation.columns else valuation.columns[0]
            if pe is None and "P/E" in valuation.index: pe = limpiar_numero(valuation.loc["P/E", col])
            if price_to_book is None and "P/B" in valuation.index: price_to_book = limpiar_numero(valuation.loc["P/B", col])
    except Exception:
        pass

    # Fallback PEG Derivado
    if peg is None:
        try:
            growth = limpiar_numero(info.get("earningsGrowth"))
            if pe is not None and growth is not None and growth > 0:
                peg = pe / (growth * 100)
        except Exception:
            pass

    info["trailingPE"] = pe
    info["forwardPE"] = forward_pe
    info["pegRatio"] = peg
    info["priceToBook"] = price_to_book

    # Cálculos derivados de seguridad si faltan ratios
    eps = limpiar_numero(info.get("trailingEps"))
    beneficio = limpiar_numero(info.get("netIncomeToCommon"))
    ingresos = limpiar_numero(info.get("totalRevenue"))
    acciones = limpiar_numero(info.get("sharesOutstanding"))
    equity = limpiar_numero(info.get("stockholdersEquity"))

    if eps is None and beneficio is not None and acciones and acciones > 0: info["trailingEps"] = beneficio / acciones
    if info.get("profitMargins") is None and beneficio is not None and ingresos and ingresos > 0: info["profitMargins"] = beneficio / ingresos
    if info.get("returnOnEquity") is None and beneficio is not None and equity and equity > 0: info["returnOnEquity"] = beneficio / equity

    return info

# =========================================================
# OBJETIVOS DE ANALISTAS MULTIFUENTE (YFINANCE + FINNHUB)
# =========================================================

@st.cache_data(ttl=3600)
def obtener_objetivos_analistas(ticker, api_key_finnhub=None):
    resultado = {}
    
    try:
        empresa = yf.Ticker(ticker)
        for metodo in ["get_analyst_price_targets", "analyst_price_targets"]:
            try:
                datos = getattr(empresa, metodo, None)
                if callable(datos): datos = datos()
                if isinstance(datos, dict):
                    for k, v in datos.items():
                        v = limpiar_numero(v)
                        if v is not None: resultado[k] = v
                    if any(k in resultado for k in ["mean", "median", "low", "high"]):
                        return resultado
            except Exception:
                pass
    except Exception:
        pass

    info = obtener_info(ticker)
    mapa = {"low": "targetLowPrice", "mean": "targetMeanPrice", "median": "targetMedianPrice", "high": "targetHighPrice"}
    for destino, origen in mapa.items():
        if destino not in resultado:
            valor = limpiar_numero(info.get(origen))
            if valor is not None: resultado[destino] = valor

    # Fallback API Finnhub si Yahoo no ha entregado nada y hay Key configurada
    if not resultado.get("mean") and api_key_finnhub:
        try:
            url = f"https://finnhub.io/api/v1/stock/price-target?symbol={ticker}&token={api_key_finnhub}"
            res = requests.get(url, timeout=5).json()
            if res.get("targetMean"):
                resultado["mean"] = limpiar_numero(res.get("targetMean"))
                resultado["high"] = limpiar_numero(res.get("targetHigh"))
                resultado["low"] = limpiar_numero(res.get("targetLow"))
                resultado["median"] = limpiar_numero(res.get("targetMedian"))
        except Exception:
            pass

    return resultado

# =========================================================
# DATOS HISTÓRICOS Y TÉCNICOS
# =========================================================

@st.cache_data(ttl=900)
def obtener_precios(ticker, periodo):
    try:
        datos = yf.download(ticker, period=periodo, auto_adjust=False, progress=False)
        if isinstance(datos, pd.DataFrame) and not datos.empty:
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)
            columnas = ["Open", "High", "Low", "Close", "Volume"]
            disponibles = [col for col in columnas if col in datos.columns]
            return datos[disponibles].dropna(subset=["Close"])
    except Exception:
        pass
    return pd.DataFrame()

def calcular_rsi(precios, periodo=14):
    delta = precios.diff()
    ganancias = delta.clip(lower=0)
    perdidas = -delta.clip(upper=0)
    media_ganancias = ganancias.rolling(periodo).mean()
    media_perdidas = perdidas.rolling(periodo).mean()
    rs = media_ganancias / media_perdidas.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# =========================================================
# RECOMENDACIONES Y ESTIMACIONES PROTEGIDAS
# =========================================================

@st.cache_data(ttl=3600)
def obtener_datos_pandas_seguro(ticker, metodos):
    try:
        empresa = yf.Ticker(ticker)
        for metodo in metodos:
            try:
                datos = getattr(empresa, metodo, None)
                if callable(datos): datos = datos()
                if isinstance(datos, pd.DataFrame) and not datos.empty:
                    return datos
            except Exception:
                pass
    except Exception:
        pass
    return pd.DataFrame()

def obtener_recomendaciones(ticker): return obtener_datos_pandas_seguro(ticker, ["get_recommendations_summary", "recommendations_summary", "get_recommendations", "recommendations"])
def obtener_upgrades_downgrades(ticker): return obtener_datos_pandas_seguro(ticker, ["get_upgrades_downgrades", "upgrades_downgrades"])
def obtener_estimaciones_eps(ticker): return obtener_datos_pandas_seguro(ticker, ["get_earnings_estimate", "earnings_estimate"])
def obtener_estimaciones_ingresos(ticker): return obtener_datos_pandas_seguro(ticker, ["get_revenue_estimate", "revenue_estimate"])
def obtener_revisiones_eps(ticker): return obtener_datos_pandas_seguro(ticker, ["get_eps_revisions", "eps_revisions"])
def obtener_tendencia_eps(ticker): return obtener_datos_pandas_seguro(ticker, ["get_eps_trend", "eps_trend"])
def obtener_crecimiento_estimado(ticker): return obtener_datos_pandas_seguro(ticker, ["get_growth_estimates", "growth_estimates"])

@st.cache_data(ttl=900)
def obtener_noticias(ticker):
    try:
        empresa = yf.Ticker(ticker)
        noticias = empresa.news
        return noticias if noticias else []
    except Exception:
        return []

# =========================================================
# SCORES DE EVALUACIÓN
# =========================================================

def calcular_score_tecnico(precio, ma20, ma50, ma200, rsi):
    score = 0
    razones = []
    if pd.notna(ma20) and precio is not None:
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
        elif rsi <= 65: score += 8; razones.append("El RSI se encuentra en zona equilibrada.")
        elif rsi <= 70: score += 5; razones.append("El RSI se aproxima a sobrecompra.")
        else: score += 2; razones.append("El RSI indica posible sobrecompra.")
    return min(score, 25), razones

def calcular_score_valoracion(pe, forward_pe, peg, price_to_book):
    score = 0
    razones = []
    if pe is not None:
        if pe < 15: score += 8; razones.append("PER bajo.")
        elif pe < 25: score += 6; razones.append("PER intermedio.")
        elif pe < 40: score += 3; razones.append("PER elevado.")
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
        else: score += 2; razones.append("ROE bajo.")
    if margen is not None:
        if margen > 0.20: score += 7; razones.append("Margen elevado.")
        elif margen > 0.10: score += 5; razones.append("Margen razonable.")
        else: score += 2; razones.append("Margen reducido.")
    if deuda is not None:
        if deuda < 50: score += 5; razones.append("Deuda baja.")
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
        else: razones.append("Ingresos decreciendo.")
    if crecimiento_beneficios is not None:
        if crecimiento_beneficios > 0.15: score += 8
        elif crecimiento_beneficios > 0.05: score += 5
        elif crecimiento_beneficios > 0: score += 2
        else: razones.append("Beneficios decreciendo.")
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
# FAIR VALUE COMBINADO
# =========================================================

def calcular_fair_value_combinado(precio_actual, valor_dcf, objetivo_analistas, eps, crecimiento_beneficios):
    valores, pesos = [], []
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
    
    if not valores: return None
    return sum(v * p for v, p in zip(valores, pesos)) / sum(pesos)

def diagnosticar_fair_value(precio_actual, fair_value):
    if precio_actual is None or fair_value is None or precio_actual <= 0: return "⚪ SIN DATOS", None
    potencial = ((fair_value - precio_actual) / precio_actual) * 100
    if potencial >= 30: return "🟢 MUY INFRAVALORADA", potencial
    if potencial >= 15: return "🟢 INFRAVALORADA", potencial
    if potencial >= -10: return "🟡 VALORACIÓN RAZONABLE", potencial
    if potencial >= -25: return "🟠 SOBREVALORADA", potencial
    return "🔴 MUY SOBREVALORADA", potencial

# =========================================================
# S&P 500 ESCÁNER
# =========================================================
SP500_FALLBACK = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "JPM", "LLY", "V", "MA", "XOM", "WMT", "COST"]

@st.cache_data(ttl=86400)
def obtener_componentes_sp500():
    try:
        tablas = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tablas[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip().tolist()
    except Exception: return SP500_FALLBACK

@st.cache_data(ttl=1800)
def escanear_sp500(candidatos_finales=20):
    tickers = obtener_componentes_sp500()
    try:
        precios = yf.download(tickers=tickers, period="1y", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
    except Exception: return pd.DataFrame()

    candidatos = []
    for simbolo in tickers:
        try:
            if isinstance(precios.columns, pd.MultiIndex) and simbolo not in precios.columns.get_level_values(0): continue
            datos = precios[simbolo] if isinstance(precios.columns, pd.MultiIndex) else precios
            if "Close" not in datos.columns: continue
            close = datos["Close"].dropna()
            if len(close) < 60: continue

            precio = float(close.iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50
            rsi = float(calcular_rsi(close).iloc[-1])

            score = 0
            if precio > ma20: score += 25
            if precio > ma50: score += 25
            if precio > ma200: score += 25
            if 45 <= rsi <= 70: score += 15
            elif 35 <= rsi < 45 or 70 < rsi <= 78: score += 8
            candidatos.append((simbolo, precio, rsi, score))
        except Exception: continue

    candidatos = sorted(candidatos, key=lambda x: x[3], reverse=True)[:candidatos_finales]
    resultados = []
    for simbolo, precio, rsi, score_tecnico in candidatos:
        try:
            info = obtener_info(simbolo)
            fcf = limpiar_numero(info.get("freeCashflow"))
            deuda = limpiar_numero(info.get("totalDebt"))
            caja = limpiar_numero(info.get("totalCash"))
            acciones = limpiar_numero(info.get("sharesOutstanding"))
            eps = limpiar_numero(info.get("trailingEps"))
            crecimiento = limpiar_numero(info.get("earningsGrowth"))
            objetivo = limpiar_numero(info.get("targetMeanPrice"))
            pe = limpiar_numero(info.get("trailingPE"))

            valor_dcf = None
            if fcf and fcf > 0 and acciones and acciones > 0:
                escenarios = calcular_escenarios_dcf(fcf, deuda=deuda or 0, caja=caja or 0, acciones=acciones)
                if escenarios.get("base"): valor_dcf = limpiar_numero(escenarios["base"].get("valor_por_accion"))

            fair_value = calcular_fair_value_combinado(precio, valor_dcf, objetivo, eps, crecimiento)
            estado, potencial = diagnosticar_fair_value(precio, fair_value)

            score_final = float(score_tecnico)
            if potencial is not None: score_final += max(-20, min(35, potencial * 0.50))
            if pe is not None:
                if pe < 20: score_final += 8
                elif pe < 30: score_final += 4
                elif pe > 50: score_final -= 5

            resultados.append({
                "Ticker": simbolo, "Empresa": info.get("shortName", simbolo), "Precio": precio,
                "Fair Value": fair_value, "Potencial %": potencial, "Diagnóstico": estado,
                "MARKET AI": round(max(0, min(100, score_final)), 1), "DCF": valor_dcf,
                "Objetivo analistas": objetivo, "P/E": pe, "RSI": rsi
            })
        except Exception: continue

    return pd.DataFrame(resultados).sort_values("MARKET AI", ascending=False).reset_index(drop=True) if resultados else pd.DataFrame()

# =========================================================
# INTERFAZ (SIDEBAR)
# =========================================================

st.sidebar.header("⚙️ Configuración")
ticker = st.sidebar.text_input("Símbolo", value="AAPL").upper().strip()
periodo = st.sidebar.selectbox("Periodo", ["6mo", "1y", "2y", "5y", "10y"], index=1)
api_key_finnhub = st.sidebar.text_input("API Finnhub (Opcional, Fallback)", type="password")

st.sidebar.divider()
st.sidebar.subheader("🎯 Precio para el análisis")
tipo_precio = st.sidebar.radio("Fuente", ["Precio automático", "Precio personalizado"])
precio_personalizado = st.sidebar.number_input("Precio de entrada", min_value=0.01, value=100.00, step=1.00) if tipo_precio == "Precio personalizado" else None

analizar = st.sidebar.button("📊 ANALIZAR", type="primary")
ranking_sp500 = st.sidebar.button("🏆 TOP 5 S&P 500")

# =========================================================
# RANKING S&P 500
# =========================================================

if ranking_sp500:
    st.divider()
    st.header("🏆 TOP 5 — S&P 500")
    with st.spinner("Escaneando el S&P 500..."):
        ranking = escanear_sp500(candidatos_finales=20)
    if ranking.empty:
        st.error("No se ha podido completar el escaneo (límite temporal de fuente).")
    else:
        st.success(f"Analizados {len(ranking)} candidatos.")
        for pos, (_, fila) in enumerate(ranking.head(5).iterrows(), start=1):
            st.markdown(f"### {pos}. {fila['Ticker']} — {fila['Empresa']}")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Precio", f"${fila['Precio']:,.2f}")
            with c2: st.metric("Fair Value", f"${fila['Fair Value']:,.2f}" if pd.notna(fila["Fair Value"]) else "N/D")
            with c3: st.metric("Potencial", f"{fila['Potencial %']:+.1f}%" if pd.notna(fila["Potencial %"]) else "N/D")
            with c4: st.metric("MARKET AI", f"{fila['MARKET AI']:.1f}/100")
            st.write(f"**Diagnóstico:** {fila['Diagnóstico']}  |  **RSI:** {fila['RSI']:.1f}")
            st.divider()
        st.dataframe(ranking, use_container_width=True, hide_index=True)

# =========================================================
# ANÁLISIS PRINCIPAL
# =========================================================

if analizar:
    if not ticker: st.error("Debes introducir un ticker."); st.stop()

    with st.spinner("Extrayendo datos multifuente seguros..."):
        precio_alpha, error_alpha = obtener_precio_alpha_vantage(ticker)
        datos = obtener_precios(ticker, periodo)
        info = obtener_info(ticker)
        objetivos = obtener_objetivos_analistas(ticker, api_key_finnhub)
        recomendaciones = obtener_recomendaciones(ticker)
        upgrades = obtener_upgrades_downgrades(ticker)
        eps_estimaciones = obtener_estimaciones_eps(ticker)
        ingresos_estimaciones = obtener_estimaciones_ingresos(ticker)
        noticias = obtener_noticias(ticker)

    if datos.empty:
        st.error(f"No hay datos históricos para {ticker}. Verifica el símbolo.")
        st.stop()

    precio_mercado = precio_alpha["precio"] if precio_alpha else None
    if precio_alpha: st.info(f"💵 Precio: **${precio_mercado:,.2f}** | Alpha Vantage | {precio_alpha['fecha']}")
    else: st.warning(f"⚠️ Respaldo a Yahoo Finance activado. ({error_alpha})")

    precio_historico = float(datos["Close"].iloc[-1])
    precio_analisis = float(precio_personalizado) if tipo_precio == "Precio personalizado" and precio_personalizado else (precio_mercado or precio_historico)
    
    variacion = ((precio_historico - (float(datos["Close"].iloc[-2]) if len(datos) > 1 else precio_historico)) / (float(datos["Close"].iloc[-2]) if len(datos) > 1 else precio_historico)) * 100

    datos["MA20"], datos["MA50"], datos["MA200"] = datos["Close"].rolling(20).mean(), datos["Close"].rolling(50).mean(), datos["Close"].rolling(200).mean()
    ma20, ma50, ma200 = datos["MA20"].iloc[-1], datos["MA50"].iloc[-1], datos["MA200"].iloc[-1]
    datos["RSI"] = calcular_rsi(datos["Close"])
    rsi = float(datos["RSI"].iloc[-1])
    volatilidad = datos["Close"].pct_change().std() * np.sqrt(252) * 100

    # Variables de Info
    pe, forward_pe, peg, price_to_book = info.get("trailingPE"), info.get("forwardPE"), info.get("pegRatio"), info.get("priceToBook")
    roe, margen, deuda = info.get("returnOnEquity"), info.get("profitMargins"), info.get("debtToEquity")
    flujo_caja, deuda_total, caja_total = info.get("freeCashflow"), info.get("totalDebt"), info.get("totalCash")
    acciones = info.get("sharesOutstanding")
    ingresos, beneficio, eps = info.get("totalRevenue"), info.get("netIncomeToCommon"), info.get("trailingEps")
    crecimiento_ingresos, crecimiento_beneficios = info.get("revenueGrowth"), info.get("earningsGrowth")

    sc_tec, rz_tec = calcular_score_tecnico(precio_historico, ma20, ma50, ma200, rsi)
    sc_val, rz_val = calcular_score_valoracion(pe, forward_pe, peg, price_to_book)
    sc_fun, rz_fun = calcular_score_fundamentales(roe, margen, deuda, flujo_caja)
    sc_cre, rz_cre = calcular_score_crecimiento(crecimiento_ingresos, crecimiento_beneficios)
    sc_rie, rz_rie = calcular_score_riesgo(volatilidad, deuda)
    score_total = sc_tec + sc_val + sc_fun + sc_cre + sc_rie

    st.success(f"✅ {info.get('longName', ticker)}")
    st.write(f"**Sector:** {info.get('sector', 'N/D')} | **Industria:** {info.get('industry', 'N/D')}")

    st.divider(); st.header("💵 Mercado")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precio", f"${precio_historico:,.2f}", f"{variacion:+.2f}%")
    with c2: st.metric("Máximo", f"${float(datos['High'].max()):,.2f}")
    with c3: st.metric("Mínimo", f"${float(datos['Low'].min()):,.2f}")
    with c4: st.metric("RSI", f"{rsi:.1f}")

    st.divider(); st.header("🎯 SCORE")
    st.progress(score_total / 100)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Técnica", f"{sc_tec}/25")
    with c2: st.metric("Valoración", f"{sc_val}/25")
    with c3: st.metric("Fundamentales", f"{sc_fun}/25")
    with c4: st.metric("Crecimiento", f"{sc_cre}/15")
    with c5: st.metric("Riesgo", f"{sc_rie}/10")

    st.divider(); st.header("📊 Fundamentales")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("PER", f"{pe:.2f}" if pe is not None else "N/D")
    with c2: st.metric("PER futuro", f"{forward_pe:.2f}" if forward_pe is not None else "N/D")
    with c3: st.metric("PEG", f"{peg:.2f}" if peg is not None else "N/D")
    with c4: st.metric("Price/Book", f"{price_to_book:.2f}" if price_to_book is not None else "N/D")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ROE", f"{roe * 100:.2f}%" if roe is not None else "N/D")
    with c2: st.metric("Margen", f"{margen * 100:.2f}%" if margen is not None else "N/D")
    with c3: st.metric("Crecimiento Ingresos", f"{crecimiento_ingresos * 100:.2f}%" if crecimiento_ingresos is not None else "N/D")
    with c4: st.metric("Crecimiento Beneficios", f"{crecimiento_beneficios * 100:.2f}%" if crecimiento_beneficios is not None else "N/D")

    st.divider(); st.header("🎯 Analistas")
    obj_bajo = objetivos.get("low")
    obj_med = objetivos.get("mean")
    obj_alto = objetivos.get("high")

    if obj_med is not None:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Objetivo bajo", f"${obj_bajo:,.2f}" if obj_bajo is not None else "N/D")
        with c2: st.metric("Objetivo medio", f"${obj_med:,.2f}")
        with c3: st.metric("Objetivo alto", f"${obj_alto:,.2f}" if obj_alto is not None else "N/D")
    else: st.info("No hay objetivos de analistas disponibles.")

    if isinstance(recomendaciones, pd.DataFrame) and not recomendaciones.empty:
        st.subheader("Consenso")
        fila = recomendaciones.iloc[-1]
        cols = st.columns(5)
        for i, (nombre, clave) in enumerate([("Strong Buy", "strongBuy"), ("Buy", "buy"), ("Hold", "hold"), ("Sell", "sell"), ("Strong Sell", "strongSell")]):
            with cols[i]: st.metric(nombre, int(fila.get(clave, 0)) if clave in fila else 0)

    st.divider(); st.header("💎 DCF & Fair Value")
    valor_base = None
    if flujo_caja and acciones:
        dcf = calcular_escenarios_dcf(flujo_caja, deuda_total or 0, caja_total or 0, acciones)
        valor_base = dcf.get("base", {}).get("valor_por_accion")
        st.metric("DCF Base Estimado", f"${valor_base:,.2f}" if valor_base else "N/D")
    else: st.info("Datos insuficientes para DCF.")

    fair_value = calcular_fair_value_combinado(precio_analisis, valor_base, obj_med, eps, crecimiento_beneficios)
    estado_fair, potencial_fair = diagnosticar_fair_value(precio_analisis, fair_value)
    
    st.subheader("Fair Value Combinado")
    c1, c2 = st.columns(2)
    with c1: st.metric("Valor", f"${fair_value:,.2f}" if fair_value else "N/D")
    with c2: st.metric("Potencial", f"{potencial_fair:+.2f}%" if potencial_fair else "N/D")
    st.write(f"### {estado_fair}")

    st.divider(); st.header("📈 Gráfico")
    figura = go.Figure()
    figura.add_trace(go.Candlestick(x=datos.index, open=datos["Open"], high=datos["High"], low=datos["Low"], close=datos["Close"], name=ticker))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA50"], name="MA50"))
    figura.add_trace(go.Scatter(x=datos.index, y=datos["MA200"], name="MA200"))
    figura.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(figura, use_container_width=True)

    if noticias:
        st.divider(); st.header("📰 Noticias")
        for noticia in noticias[:5]:
            t = noticia.get("content", noticia).get("title", "Sin título")
            u = noticia.get("content", noticia).get("canonicalUrl", {}).get("url", "")
            if t and u: st.markdown(f"- [{t}]({u})")
