import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# Intentar importar dcf de tu módulo local si existe, o usar lógica interna de fallback
try:
    from dcf import calcular_escenarios_dcf, diagnosticar_valoracion
except ImportError:
    def calcular_escenarios_dcf(fcf, deuda=0, caja=0, acciones=1, tasa_descuento=0.09, tasa_terminal=0.025):
        if not fcf or fcf <= 0 or not acciones or acciones <= 0:
            return {}
        escenarios = {}
        tasas = {"conservador": 0.03, "base": 0.08, "optimista": 0.14}
        for nombre, g in tasas.items():
            fcf_futuro = [fcf * ((1 + g) ** i) for i in range(1, 6)]
            v_presente = [f / ((1 + tasa_descuento) ** i) for i, f in enumerate(fcf_futuro, start=1)]
            val_terminal = (fcf_futuro[-1] * (1 + tasa_terminal)) / (tasa_descuento - tasa_terminal)
            val_terminal_pres = val_terminal / ((1 + tasa_descuento) ** 5)
            ev = sum(v_presente) + val_terminal_pres
            eq_val = ev + caja - deuda
            valor_accion = eq_val / acciones
            escenarios[nombre] = {
                "crecimiento_pct": g * 100,
                "valor_empresa": ev,
                "valor_equity": eq_val,
                "valor_por_accion": valor_accion
            }
        return escenarios

    def diagnosticar_valoracion(precio, fair_value):
        if not precio or not fair_value: return "⚪ SIN DATOS", 0
        potencial = ((fair_value - precio) / precio) * 100
        if potencial >= 30: return "🟢 MUY INFRAVALORADA", potencial
        if potencial >= 15: return "🟢 INFRAVALORADA", potencial
        if potencial >= -10: return "🟡 VALORACIÓN RAZONABLE", potencial
        if potencial >= -25: return "🟠 SOBREVALORADA", potencial
        return "🔴 MUY SOBREVALORADA", potencial

# =========================================================
# CONFIGURACIÓN GENERAL DE LA APP
# =========================================================

st.set_page_config(
    page_title="MARKET AI",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 MARKET AI")
st.caption("Sistema de Análisis Financiero Multifuente con Resiliencia de Datos.")

# =========================================================
# FUNCIONES AUXILIARES DE LIMPIEZA Y SEGURIDAD
# =========================================================

def limpiar_numero(valor):
    try:
        if valor is None or pd.isna(valor) or valor == "N/A" or valor == "":
            return None
        return float(valor)
    except Exception:
        return None

# =========================================================
# API ALPHA VANTAGE (RESPALDO DE PRECIO)
# =========================================================

@st.cache_data(ttl=3600)
def obtener_precio_alpha_vantage(ticker, api_key="DEMO"):
    try:
        url = "https://www.alphavantage.co/query"
        parametros = {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": api_key}
        respuesta = requests.get(url, params=parametros, timeout=10)
        datos = respuesta.json()
        precio = datos.get("Global Quote", {}).get("05. price")
        fecha = datos.get("Global Quote", {}).get("07. latest trading day")
        if precio:
            return {"precio": float(precio), "fecha": fecha}, None
        return None, "Sin datos de precio en Alpha Vantage"
    except Exception as e:
        return None, str(e)

# =========================================================
# EXTRACCIÓN ROBUSTA MULTIFUENTE (YAHOO DIRECTO + YFINANCE)
# =========================================================

@st.cache_data(ttl=3600)
def obtener_info(ticker):
    info = {}
    empresa = yf.Ticker(ticker)

    # 1. Fuente estándar yfinance
    try:
        if isinstance(empresa.info, dict):
            info.update(empresa.info)
    except Exception:
        pass

    try:
        fast = empresa.fast_info
        if fast:
            for destino, origen in [("currentPrice", "last_price"), ("marketCap", "market_cap"), ("sharesOutstanding", "shares")]:
                if info.get(destino) is None:
                    v = getattr(fast, origen, None)
                    if v is not None: info[destino] = v
    except Exception:
        pass

    # 2. Respaldo directo a Endpoints v10 de Yahoo Query si faltan datos
    campos_criticos = ["trailingPE", "forwardPE", "pegRatio", "priceToBook", "targetMeanPrice", "returnOnEquity", "profitMargins", "freeCashflow"]
    if any(info.get(k) is None for k in campos_criticos):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=summaryDetail,financialData,defaultKeyStatistics"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                res = r.json().get("quoteSummary", {}).get("result", [])
                if res:
                    for mod in ["summaryDetail", "financialData", "defaultKeyStatistics"]:
                        for k, v in res[0].get(mod, {}).items():
                            if info.get(k) is None and isinstance(v, dict) and "raw" in v:
                                info[k] = v["raw"]
        except Exception:
            pass

    # 3. Respaldo desde Estados Financieros
    try:
        income = empresa.get_income_stmt(freq="yearly")
        if isinstance(income, pd.DataFrame) and not income.empty:
            col = income.columns[0]
            if info.get("totalRevenue") is None and "TotalRevenue" in income.index: info["totalRevenue"] = limpiar_numero(income.loc["TotalRevenue", col])
            if info.get("netIncomeToCommon") is None and "NetIncome" in income.index: info["netIncomeToCommon"] = limpiar_numero(income.loc["NetIncome", col])
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
                fcf = limpiar_numero(cashflow.loc["OperatingCashFlow", col]) or 0
                capex = limpiar_numero(cashflow.loc.get("CapitalExpenditure", col)) or 0
                info["freeCashflow"] = fcf + capex
    except Exception:
        pass

    # 4. Cálculo derivado seguro para PEG / PER
    pe = limpiar_numero(info.get("trailingPE"))
    growth = limpiar_numero(info.get("earningsGrowth"))
    if info.get("pegRatio") is None and pe and growth and growth > 0:
        info["pegRatio"] = pe / (growth * 100)

    return info

# =========================================================
# ESTIMACIONES Y OBJETIVOS DE ANALISTAS (YFINANCE + FINNHUB)
# =========================================================

@st.cache_data(ttl=3600)
def obtener_objetivos_analistas(ticker, api_key_finnhub=None):
    resultado = {}
    try:
        empresa = yf.Ticker(ticker)
        for metodo in ["get_analyst_price_targets", "analyst_price_targets"]:
            datos = getattr(empresa, metodo, None)
            if callable(datos): datos = datos()
            if isinstance(datos, dict):
                for k, v in datos.items():
                    val = limpiar_numero(v)
                    if val is not None: resultado[k] = val
                if any(k in resultado for k in ["mean", "low", "high"]):
                    return resultado
    except Exception:
        pass

    info = obtener_info(ticker)
    for destino, origen in [("low", "targetLowPrice"), ("mean", "targetMeanPrice"), ("median", "targetMedianPrice"), ("high", "targetHighPrice")]:
        if destino not in resultado:
            val = limpiar_numero(info.get(origen))
            if val is not None: resultado[destino] = val

    if not resultado.get("mean") and api_key_finnhub:
        try:
            url = f"https://finnhub.io/api/v1/stock/price-target?symbol={ticker}&token={api_key_finnhub}"
            res = requests.get(url, timeout=5).json()
            if res.get("targetMean"):
                resultado = {
                    "mean": limpiar_numero(res.get("targetMean")),
                    "high": limpiar_numero(res.get("targetHigh")),
                    "low": limpiar_numero(res.get("targetLow")),
                    "median": limpiar_numero(res.get("targetMedian"))
                }
        except Exception:
            pass

    return resultado

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
def obtener_precios(ticker, periodo):
    try:
        datos = yf.download(ticker, period=periodo, auto_adjust=False, progress=False)
        if isinstance(datos, pd.DataFrame) and not datos.empty:
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)
            return datos[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
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

@st.cache_data(ttl=900)
def obtener_noticias(ticker):
    try:
        empresa = yf.Ticker(ticker)
        return empresa.news or []
    except Exception:
        return []

# =========================================================
# ALGORITMO DE SCORES
# =========================================================

def calcular_score_tecnico(precio, ma20, ma50, ma200, rsi):
    score, razones = 0, []
    if pd.notna(ma20) and precio:
        if precio > ma20: score += 5; razones.append("Precio por encima de MA20.")
        else: razones.append("Precio por debajo de MA20.")
    if pd.notna(ma20) and pd.notna(ma50):
        if ma20 > ma50: score += 5; razones.append("MA20 por encima de MA50.")
        else: razones.append("MA20 por debajo de MA50.")
    if pd.notna(ma50) and pd.notna(ma200):
        if ma50 > ma200: score += 7; razones.append("MA50 por encima de MA200.")
        else: razones.append("MA50 por debajo de MA200.")
    if pd.notna(rsi):
        if rsi < 30: score += 6; razones.append("RSI en sobreventa.")
        elif rsi <= 65: score += 8; razones.append("RSI en zona equilibrada.")
        elif rsi <= 70: score += 5; razones.append("RSI cercano a sobrecompra.")
        else: score += 2; razones.append("RSI en sobrecompra.")
    return min(score, 25), razones

def calcular_score_valoracion(pe, forward_pe, peg, price_to_book):
    score, razones = 0, []
    if pe is not None:
        if pe < 15: score += 8; razones.append("PER atractivo (<15).")
        elif pe < 25: score += 6; razones.append("PER moderado (15-25).")
        elif pe < 40: score += 3; razones.append("PER elevado (25-40).")
        else: score += 1; razones.append("PER muy elevado (>40).")
    if forward_pe is not None:
        if forward_pe < 15: score += 7
        elif forward_pe < 25: score += 5
        elif forward_pe < 40: score += 2
    if peg is not None:
        if peg < 1: score += 6; razones.append("PEG bajo (<1.0).")
        elif peg < 2: score += 4; razones.append("PEG razonable.")
        else: score += 1
    if price_to_book is not None:
        if price_to_book < 2: score += 4
        elif price_to_book < 5: score += 2
        else: score += 1
    return min(score, 25), razones

def calcular_score_fundamentales(roe, margen, deuda, flujo_caja):
    score, razones = 0, []
    if roe is not None:
        if roe > 0.20: score += 8; razones.append("ROE excelente (>20%).")
        elif roe > 0.10: score += 5; razones.append("ROE saludable.")
        else: score += 2
    if margen is not None:
        if margen > 0.20: score += 7; razones.append("Margen de beneficio alto (>20%).")
        elif margen > 0.10: score += 5; razones.append("Margen correcto.")
        else: score += 2
    if deuda is not None:
        if deuda < 50: score += 5; razones.append("Deuda muy baja.")
        elif deuda < 100: score += 3
        else: score += 1; razones.append("Deuda alta.")
    if flujo_caja is not None:
        if flujo_caja > 0: score += 5; razones.append("Free Cash Flow positivo.")
        else: razones.append("Free Cash Flow negativo.")
    return min(score, 25), razones

def calcular_score_crecimiento(crecimiento_ingresos, crecimiento_beneficios):
    score, razones = 0, []
    if crecimiento_ingresos is not None:
        if crecimiento_ingresos > 0.15: score += 7
        elif crecimiento_ingresos > 0.05: score += 5
        elif crecimiento_ingresos > 0: score += 2
        else: razones.append("Ingresos en retroceso.")
    if crecimiento_beneficios is not None:
        if crecimiento_beneficios > 0.15: score += 8
        elif crecimiento_beneficios > 0.05: score += 5
        elif crecimiento_beneficios > 0: score += 2
        else: razones.append("Beneficios en retroceso.")
    return min(score, 15), razones

def calcular_score_riesgo(volatilidad, deuda):
    score, razones = 0, []
    if volatilidad < 20: score += 6
    elif volatilidad < 35: score += 4
    elif volatilidad < 50: score += 2
    else: score += 1; razones.append("Alta volatilidad anualizada.")
    if deuda is not None:
        if deuda < 50: score += 4
        elif deuda < 100: score += 2
        else: score += 1
    else: score += 2
    return min(score, 10), razones

def calcular_fair_value_combinado(precio_actual, valor_dcf, objetivo_analistas, eps, crecimiento_beneficios):
    valores, pesos = [], []
    if valor_dcf and valor_dcf > 0:
        valores.append(float(valor_dcf)); pesos.append(0.50)
    if objetivo_analistas and objetivo_analistas > 0:
        valores.append(float(objetivo_analistas)); pesos.append(0.25)
    if eps and eps > 0:
        g = float(crecimiento_beneficios) * 100 if crecimiento_beneficios else 5.0
        g = max(-5.0, min(25.0, g))
        pe_mult = max(10.0, min(30.0, 15.0 + (0.50 * g)))
        val_m = eps * pe_mult
        if val_m > 0: valores.append(float(val_m)); pesos.append(0.25)
    if not valores: return None
    return sum(v * p for v, p in zip(valores, pesos)) / sum(pesos)

# =========================================================
# ESCÁNER S&P 500
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
            ma20, ma50 = float(close.rolling(20).mean().iloc[-1]), float(close.rolling(50).mean().iloc[-1])
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
            fcf, deuda, caja, acciones = limpiar_numero(info.get("freeCashflow")), limpiar_numero(info.get("totalDebt")), limpiar_numero(info.get("totalCash")), limpiar_numero(info.get("sharesOutstanding"))
            eps, crecimiento, objetivo, pe = limpiar_numero(info.get("trailingEps")), limpiar_numero(info.get("earningsGrowth")), limpiar_numero(info.get("targetMeanPrice")), limpiar_numero(info.get("trailingPE"))

            valor_dcf = None
            if fcf and fcf > 0 and acciones:
                esc = calcular_escenarios_dcf(fcf, deuda=deuda or 0, caja=caja or 0, acciones=acciones)
                if esc.get("base"): valor_dcf = limpiar_numero(esc["base"].get("valor_por_accion"))

            fair_value = calcular_fair_value_combinado(precio, valor_dcf, objetivo, eps, crecimiento)
            estado, potencial = diagnosticar_valoracion(precio, fair_value)

            score_final = float(score_tecnico)
            if potencial is not None: score_final += max(-20, min(35, potencial * 0.50))
            if pe:
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
api_key_finnhub = st.sidebar.text_input("API Finnhub (Respaldo opcional)", type="password")

st.sidebar.divider()
st.sidebar.subheader("🎯 Precio para el análisis")
tipo_precio = st.sidebar.radio("Fuente de Precio", ["Precio automático", "Precio personalizado"])
precio_personalizado = st.sidebar.number_input("Precio de entrada ($)", min_value=0.01, value=100.00, step=1.00) if tipo_precio == "Precio personalizado" else None

analizar = st.sidebar.button("📊 ANALIZAR ACCIÓN", type="primary")
ranking_sp500 = st.sidebar.button("🏆 TOP 5 S&P 500")

# =========================================================
# VISTA: RANKING S&P 500
# =========================================================

if ranking_sp500:
    st.divider()
    st.header("🏆 TOP 5 — S&P 500")
    with st.spinner("Escaneando las mejores oportunidades del S&P 500..."):
        ranking = escanear_sp500(candidatos_finales=20)
    if ranking.empty:
        st.error("No se ha podido procesar el escaneo en este momento.")
    else:
        st.success(f"Escaneo finalizado con éxito ({len(ranking)} empresas evaluadas).")
        for pos, (_, fila) in enumerate(ranking.head(5).iterrows(), start=1):
            st.markdown(f"### {pos}. {fila['Ticker']} — {fila['Empresa']}")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Precio", f"${fila['Precio']:,.2f}")
            with c2: st.metric("Fair Value", f"${fila['Fair Value']:,.2f}" if pd.notna(fila["Fair Value"]) else "N/D")
            with c3: st.metric("Potencial", f"{fila['Potencial %']:+.1f}%" if pd.notna(fila["Potencial %"]) else "N/D")
            with c4: st.metric("MARKET AI Score", f"{fila['MARKET AI']:.1f}/100")
            st.write(f"**Diagnóstico:** {fila['Diagnóstico']} | **RSI:** {fila['RSI']:.1f} | **PER:** {fila['P/E'] if pd.notna(fila['P/E']) else 'N/D'}")
            st.divider()
        
        st.subheader("Tabla Escaneada")
        st.dataframe(ranking, use_container_width=True, hide_index=True)

# =========================================================
# VISTA: ANÁLISIS PRINCIPAL DE TICKER
# =========================================================

if analizar:
    if not ticker:
        st.error("Por favor introduce un símbolo válido.")
        st.stop()

    with st.spinner("Extrayendo datos de múltiples fuentes..."):
        precio_alpha, error_alpha = obtener_precio_alpha_vantage(ticker)
        datos = obtener_precios(ticker, periodo)
        info = obtener_info(ticker)
        objetivos = obtener_objetivos_analistas(ticker, api_key_finnhub)
        recomendaciones = obtener_recomendaciones(ticker)
        upgrades = obtener_upgrades_downgrades(ticker)
        eps_estimaciones = obtener_estimaciones_eps(ticker)
        ingresos_estimaciones = obtener_estimaciones_ingresos(ticker)
        revisiones_eps = obtener_revisiones_eps(ticker)
        tendencia_eps = obtener_tendencia_eps(ticker)
        crecimiento_est = obtener_crecimiento_estimado(ticker)
        noticias = obtener_noticias(ticker)

    if datos.empty:
        st.error(f"No fue posible encontrar información técnica para **{ticker}**.")
        st.stop()

    # Precios
    precio_mercado = precio_alpha["precio"] if precio_alpha else None
    if precio_alpha:
        st.info(f"💵 Precio Alpha Vantage: **${precio_mercado:,.2f}** | Fecha: {precio_alpha['fecha']}")
    elif error_alpha:
        st.warning(f"⚠️ Usando precio de Yahoo Finance (Alpha Vantage: {error_alpha})")

    precio_historico = float(datos["Close"].iloc[-1])
    precio_analisis = float(precio_personalizado) if tipo_precio == "Precio personalizado" and precio_personalizado else (precio_mercado or precio_historico)
    
    variacion = ((precio_historico - (float(datos["Close"].iloc[-2]) if len(datos) > 1 else precio_historico)) / (float(datos["Close"].iloc[-2]) if len(datos) > 1 else precio_historico)) * 100

    # Indicadores
    datos["MA20"] = datos["Close"].rolling(20).mean()
    datos["MA50"] = datos["Close"].rolling(50).mean()
    datos["MA200"] = datos["Close"].rolling(200).mean()
    ma20, ma50, ma200 = datos["MA20"].iloc[-1], datos["MA50"].iloc[-1], datos["MA200"].iloc[-1]
    datos["RSI"] = calcular_rsi(datos["Close"])
    rsi = float(datos["RSI"].iloc[-1])
    volatilidad = datos["Close"].pct_change().std() * np.sqrt(252) * 100

    # Ratios
    pe, forward_pe, peg, price_to_book = info.get("trailingPE"), info.get("forwardPE"), info.get("pegRatio"), info.get("priceToBook")
    roe, margen, deuda = info.get("returnOnEquity"), info.get("profitMargins"), info.get("debtToEquity")
    flujo_caja, deuda_total, caja_total = info.get("freeCashflow"), info.get("totalDebt"), info.get("totalCash")
    acciones = info.get("sharesOutstanding")
    ingresos, beneficio, eps = info.get("totalRevenue"), info.get("netIncomeToCommon"), info.get("trailingEps")
    crecimiento_ingresos, crecimiento_beneficios = info.get("revenueGrowth"), info.get("earningsGrowth")

    # Scores
    sc_tec, rz_tec = calcular_score_tecnico(precio_historico, ma20, ma50, ma200, rsi)
    sc_val, rz_val = calcular_score_valoracion(pe, forward_pe, peg, price_to_book)
    sc_fun, rz_fun = calcular_score_fundamentales(roe, margen, deuda, flujo_caja)
    sc_cre, rz_cre = calcular_score_crecimiento(crecimiento_ingresos, crecimiento_beneficios)
    sc_rie, rz_rie = calcular_score_riesgo(volatilidad, deuda)
    score_total = sc_tec + sc_val + sc_fun + sc_cre + sc_rie

    # Cabecera
    st.success(f"✅ **{info.get('longName', ticker)}** ({ticker})")
    st.write(f"**Sector:** {info.get('sector', 'N/D')} | **Industria:** {info.get('industry', 'N/D')} | **País:** {info.get('country', 'N/D')}")

    # Resumen Mercado
    st.divider()
    st.header("💵 RESUMEN DE MERCADO")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precio Actual", f"${precio_historico:,.2f}", f"{variacion:+.2f}%")
    with c2: st.metric("Máximo Periodo", f"${float(datos['High'].max()):,.2f}")
    with c3: st.metric("Mínimo Periodo", f"${float(datos['Low'].min()):,.2f}")
    with c4: st.metric("RSI (14)", f"{rsi:.1f}")

    # Score General
    st.divider()
    st.header("🎯 MARKET AI SCORE")
    st.progress(score_total / 100)
    st.subheader(f"Puntuación Global: **{score_total:.1f} / 100**")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Técnico", f"{sc_tec}/25")
    with c2: st.metric("Valoración", f"{sc_val}/25")
    with c3: st.metric("Fundamentales", f"{sc_fun}/25")
    with c4: st.metric("Crecimiento", f"{sc_cre}/15")
    with c5: st.metric("Riesgo", f"{sc_rie}/10")

    with st.expander("🔍 Ver detalles del diagnóstico del Score"):
        for r in rz_tec + rz_val + rz_fun + rz_cre + rz_rie:
            st.write(f"- {r}")

    # Fundamental y Valoración
    st.divider()
    st.header("📊 FUNDAMENTALES Y VALORACIÓN")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("PER (Trailing)", f"{pe:.2f}" if pe is not None else "N/D")
    with c2: st.metric("PER (Forward)", f"{forward_pe:.2f}" if forward_pe is not None else "N/D")
    with c3: st.metric("PEG Ratio", f"{peg:.2f}" if peg is not None else "N/D")
    with c4: st.metric("Price / Book", f"{price_to_book:.2f}" if price_to_book is not None else "N/D")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ROE", f"{roe * 100:.2f}%" if roe is not None else "N/D")
    with c2: st.metric("Margen Neto", f"{margen * 100:.2f}%" if margen is not None else "N/D")
    with c3: st.metric("Crec. Ingresos", f"{crecimiento_ingresos * 100:.2f}%" if crecimiento_ingresos is not None else "N/D")
    with c4: st.metric("Crec. Beneficios", f"{crecimiento_beneficios * 100:.2f}%" if crecimiento_beneficios is not None else "N/D")

    # Escenarios DCF Modelo Completo
    st.divider()
    st.header("💎 MODELO DE VALORACIÓN DCF (3 ESCENARIOS)")
    escenarios_dcf = {}
    if flujo_caja and flujo_caja > 0 and acciones:
        escenarios_dcf = calcular_escenarios_dcf(flujo_caja, deuda=deuda_total or 0, caja=caja_total or 0, acciones=acciones)
        cols_dcf = st.columns(3)
        nombres = [("Conservador", "conservador"), ("Caso Base", "base"), ("Optimista", "optimista")]
        for idx, (titulo, key) in enumerate(nombres):
            if key in escenarios_dcf:
                e = escenarios_dcf[key]
                v_acc = e.get("valor_por_accion")
                with cols_dcf[idx]:
                    st.subheader(f"Escenario {titulo}")
                    st.metric("Fair Value DCF", f"${v_acc:,.2f}" if v_acc else "N/D")
                    st.write(f"- Crecimiento estimado: **{e.get('crecimiento_pct', 0):.1f}%**")
                    st.write(f"- Enterprise Value: **${e.get('valor_empresa', 0)/1e9:,.2f}B**")
    else:
        st.info("No hay suficientes datos de Free Cash Flow o Acciones para calcular el modelo DCF.")

    valor_dcf_base = escenarios_dcf.get("base", {}).get("valor_por_accion") if escenarios_dcf else None
    obj_med = objetivos.get("mean")

    fair_value = calcular_fair_value_combinado(precio_analisis, valor_dcf_base, obj_med, eps, crecimiento_beneficios)
    estado_fair, potencial_fair = diagnosticar_valoracion(precio_analisis, fair_value)

    st.subheader("Fair Value Ponderado Combinado")
    c1, c2 = st.columns(2)
    with c1: st.metric("Fair Value", f"${fair_value:,.2f}" if fair_value else "N/D")
    with c2: st.metric("Potencial Estimado", f"{potencial_fair:+.2f}%" if potencial_fair is not None else "N/D")
    st.markdown(f"### Diagnóstico Final: **{estado_fair}**")

    # Consenso y Estimaciones
    st.divider()
    st.header("🎯 OPINIÓN DE ANALISTAS Y ESTIMACIONES")
    obj_bajo, obj_alto = objetivos.get("low"), objetivos.get("high")
    if obj_med:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Objetivo Mínimo", f"${obj_bajo:,.2f}" if obj_bajo else "N/D")
        with c2: st.metric("Objetivo Promedio", f"${obj_med:,.2f}")
        with c3: st.metric("Objetivo Máximo", f"${obj_alto:,.2f}" if obj_alto else "N/D")

    t1, t2, t3, t4 = st.tabs(["📊 Recomendaciones", "📈 Estimaciones EPS", "💵 Estimaciones Ingresos", "🔄 Revisiones y Cambios"])
    
    with t1:
        if isinstance(recomendaciones, pd.DataFrame) and not recomendaciones.empty:
            st.dataframe(recomendaciones, use_container_width=True)
        else: st.write("No hay resumen de recomendaciones disponible.")

    with t2:
        if isinstance(eps_estimaciones, pd.DataFrame) and not eps_estimaciones.empty:
            st.dataframe(eps_estimaciones, use_container_width=True)
        else: st.write("No hay datos de estimaciones de EPS disponibles.")

    with t3:
        if isinstance(ingresos_estimaciones, pd.DataFrame) and not ingresos_estimaciones.empty:
            st.dataframe(ingresos_estimaciones, use_container_width=True)
        else: st.write("No hay datos de estimaciones de ingresos disponibles.")

    with t4:
        if isinstance(upgrades, pd.DataFrame) and not upgrades.empty:
            st.subheader("Últimos Cambios de Recomendación (Upgrades / Downgrades)")
            st.dataframe(upgrades.head(10), use_container_width=True)
        if isinstance(revisiones_eps, pd.DataFrame) and not revisiones_eps.empty:
            st.subheader("Revisiones de EPS")
            st.dataframe(revisiones_eps, use_container_width=True)

    # Gráfico Técnico
    st.divider()
    st.header("📈 ANÁLISIS TÉCNICO VIRTUAL")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=datos.index, open=datos["Open"], high=datos["High"], low=datos["Low"], close=datos["Close"], name=ticker))
    fig.add_trace(go.Scatter(x=datos.index, y=datos["MA20"], name="MA20", line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=datos.index, y=datos["MA50"], name="MA50", line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=datos.index, y=datos["MA200"], name="MA200", line=dict(color='red', width=2)))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, title=f"Precio e Indicadores ({ticker})")
    st.plotly_chart(fig, use_container_width=True)

    # Noticias
    if noticias:
        st.divider()
        st.header("📰 ÚLTIMAS NOTICIAS")
        for noticia in noticias[:5]:
            # Compatibilidad con formato antiguo y nuevo de dicts de noticias de Yahoo
            item = noticia.get("content", noticia)
            titulo = item.get("title", "Noticia sin título")
            link = item.get("canonicalUrl", {}).get("url") or noticia.get("link") or ""
            publisher = item.get("provider", {}).get("displayName") or noticia.get("publisher") or "Fuente externa"
            if titulo:
                if link:
                    st.markdown(f"- **[{titulo}]({link})** *({publisher})*")
                else:
                    st.markdown(f"- **{titulo}** *({publisher})*")
