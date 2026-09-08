import pandas as pd
import numpy as np
from datetime import datetime

# Intentar importar el simulador de señales si existe en el proyecto
try:
    from signals_engine import simular_senal_historica
except ImportError:
    def simular_senal_historica(df_row, score_historico):
        rsi = df_row.get("RSI_14", 50.0)
        if score_historico >= 68:
            return "COMPRA FUERTE"
        elif score_historico >= 56:
            return "COMPRA"
        elif score_historico <= 38:
            return "VENTA"
        elif rsi > 70 or rsi < 30:
            return "VIGILAR"
        else:
            return "MANTENER"


def ejecutar_backtest_engine(
    df_historico, 
    capital_inicial=10000.0, 
    comision_pct=0.1, 
    stop_loss_pct=5.0, 
    take_profit_pct=10.0
):
    """
    Motor completo de Backtesting para simulación de estrategias históricas.
    Garantiza cero Look-Ahead Bias utilizando únicamente datos pasados.
    """
    if df_historico is None or not isinstance(df_historico, pd.DataFrame) or df_historico.empty:
        return {
            "error": "No hay datos suficientes para ejecutar el backtest.",
            "capital_final": capital_inicial,
            "retorno_total_pct": 0.0,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "total_operaciones": 0,
            "operaciones": [],
            "curva_capital": []
        }

    df = df_historico.copy()

    # Normalizar nombres de columnas de precios
    col_close = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
    if not col_close:
        return {"error": "El DataFrame no contiene la columna de precio 'Close'."}

    posicion = 0  # 0: Sin posición, 1: Comprado (Long)
    precio_entrada = 0.0
    fecha_entrada = None
    capital = float(capital_inicial)
    max_capital = capital
    max_drawdown = 0.0
    
    operaciones = []
    curva_capital = []

    # Iterar sobre cada vela histórica (sin anticipar precios futuros)
    for i in range(len(df)):
        row = df.iloc[i]
        precio_actual = float(row[col_close])
        fecha_actual = str(row.name) if hasattr(row, 'name') else f"Día {i+1}"
        score_t = float(row.get("SCORE_HISTORICO", 50.0))

        # 1. Obtener la señal técnica/predictiva simulada
        senal_i = simular_senal_historica(row, score_t)

        # 2. Control de posiciones activas (Gestión de Riesgo)
        if posicion == 1:
            retorno_unrealized = ((precio_actual - precio_entrada) / precio_entrada) * 100.0

            # Evaluar Stop Loss o Take Profit
            alcanzo_stop = stop_loss_pct is not None and retorno_unrealized <= -abs(stop_loss_pct)
            alcanzo_tp = take_profit_pct is not None and retorno_unrealized >= abs(take_profit_pct)
            es_venta_senal = senal_i in ["VENTA", "VENTA FUERTE"]

            if alcanzó_stop or alcanzó_tp or es_venta_senal:
                motivo_salida = "Stop Loss" if alcanzó_stop else ("Take Profit" if alcanzó_tp else "Señal Venta")
                
                # Ejecutar Venta
                posicion = 0
                comision_salida = precio_actual * (comision_pct / 100.0)
                precio_neto_salida = precio_actual - comision_salida
                
                retorno_realizado_pct = ((precio_neto_salida - precio_entrada) / precio_entrada) * 100.0
                capital = capital * (1 + (retorno_realizado_pct / 100.0))

                operaciones.append({
                    "tipo": "VENTA",
                    "motivo": motivo_salida,
                    "fecha_entrada": fecha_entrada,
                    "fecha_salida": fecha_actual,
                    "precio_entrada": round(precio_entrada, 2),
                    "precio_salida": round(precio_actual, 2),
                    "retorno_pct": round(retorno_realizado_pct, 2),
                    "capital_resultante": round(capital, 2)
                })

        # 3. Evaluar nueva Entrada (Compra)
        elif posicion == 0:
            if senal_i in ["COMPRA FUERTE", "COMPRA"]:
                posicion = 1
                comision_entrada = precio_actual * (comision_pct / 100.0)
                precio_entrada = precio_actual + comision_entrada
                fecha_entrada = fecha_actual

        # 4. Cálculo de Max Drawdown y seguimiento de capital
        if capital > max_capital:
            max_capital = capital
        drawdown_actual = ((max_capital - capital) / max_capital) * 100.0
        if drawdown_actual > max_drawdown:
            max_drawdown = drawdown_actual

        curva_capital.append({
            "fecha": fecha_actual,
            "capital": round(capital, 2)
        })

    # Métricas consolidadas
    tot_ops = len(operaciones)
    ops_ganadoras = [op for op in operaciones if op.get("retorno_pct", 0) > 0]
    ops_perdedoras = [op for op in operaciones if op.get("retorno_pct", 0) <= 0]
    
    win_rate = (len(ops_ganadoras) / tot_ops * 100.0) if tot_ops > 0 else 0.0

    sum_ganancias = sum([op["retorno_pct"] for op in ops_ganadoras])
    sum_perdidas = abs(sum([op["retorno_pct"] for op in ops_perdedoras]))
    profit_factor = (sum_ganancias / sum_perdidas) if sum_perdidas > 0 else (sum_ganancias if sum_ganancias > 0 else 1.0)

    retorno_total = ((capital - capital_inicial) / capital_inicial) * 100.0

    return {
        "capital_inicial": capital_inicial,
        "capital_final": round(capital, 2),
        "retorno_total_pct": round(retorno_total, 2),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2),
        "total_operaciones": tot_ops,
        "operaciones_ganadoras": len(ops_ganadoras),
        "operaciones_perdedoras": len(ops_perdedoras),
        "operaciones": operaciones,
        "curva_capital": curva_capital
    }
