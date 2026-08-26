import math


def calcular_dcf(
    free_cash_flow,
    crecimiento_inicial=0.10,
    crecimiento_largo_plazo=0.03,
    tasa_descuento=0.10,
    años=5,
    deuda=0.0,
    caja=0.0,
    acciones=1.0,
):
    """Calcula un DCF sencillo y devuelve valor empresarial y valor por acción."""
    try:
        fcf = float(free_cash_flow)
        deuda = float(deuda or 0)
        caja = float(caja or 0)
        acciones = float(acciones)
        crecimiento_inicial = float(crecimiento_inicial)
        crecimiento_largo_plazo = float(crecimiento_largo_plazo)
        tasa_descuento = float(tasa_descuento)
        años = int(años)
    except (TypeError, ValueError):
        return None

    if fcf <= 0 or acciones <= 0 or años < 1:
        return None

    if tasa_descuento <= crecimiento_largo_plazo:
        return None

    flujos = []
    fcf_actual = fcf

    for año in range(1, años + 1):
        if años == 1:
            crecimiento = crecimiento_largo_plazo
        else:
            progreso = (año - 1) / (años - 1)
            crecimiento = crecimiento_inicial + (
                crecimiento_largo_plazo - crecimiento_inicial
            ) * progreso

        fcf_actual *= 1 + crecimiento
        valor_presente = fcf_actual / ((1 + tasa_descuento) ** año)

        flujos.append(
            {
                "año": año,
                "crecimiento": crecimiento,
                "fcf": fcf_actual,
                "valor_presente": valor_presente,
            }
        )

    valor_presente_flujos = sum(
        flujo["valor_presente"] for flujo in flujos
    )

    fcf_terminal = fcf_actual * (1 + crecimiento_largo_plazo)
    valor_terminal = fcf_terminal / (
        tasa_descuento - crecimiento_largo_plazo
    )
    valor_presente_terminal = valor_terminal / (
        (1 + tasa_descuento) ** años
    )

    enterprise_value = (
        valor_presente_flujos + valor_presente_terminal
    )
    equity_value = enterprise_value - deuda + caja
    valor_por_accion = equity_value / acciones

    if not math.isfinite(valor_por_accion):
        return None

    return {
        "flujos": flujos,
        "valor_presente_flujos": valor_presente_flujos,
        "valor_terminal": valor_terminal,
        "valor_presente_terminal": valor_presente_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "valor_por_accion": valor_por_accion,
    }


def calcular_escenarios_dcf(
    free_cash_flow,
    deuda=0.0,
    caja=0.0,
    acciones=1.0,
):
    """Devuelve tres escenarios DCF: pesimista, base y optimista."""
    return {
        "pesimista": calcular_dcf(
            free_cash_flow,
            crecimiento_inicial=0.04,
            crecimiento_largo_plazo=0.02,
            tasa_descuento=0.12,
            años=5,
            deuda=deuda,
            caja=caja,
            acciones=acciones,
        ),
        "base": calcular_dcf(
            free_cash_flow,
            crecimiento_inicial=0.08,
            crecimiento_largo_plazo=0.03,
            tasa_descuento=0.10,
            años=5,
            deuda=deuda,
            caja=caja,
            acciones=acciones,
        ),
        "optimista": calcular_dcf(
            free_cash_flow,
            crecimiento_inicial=0.14,
            crecimiento_largo_plazo=0.04,
            tasa_descuento=0.09,
            años=5,
            deuda=deuda,
            caja=caja,
            acciones=acciones,
        ),
    }


def diagnosticar_valoracion(precio_actual, valor_razonable):
    """Compara el precio con el valor razonable estimado."""
    try:
        precio_actual = float(precio_actual)
        valor_razonable = float(valor_razonable)
    except (TypeError, ValueError):
        return "⚪ SIN DATOS", None, "No hay datos suficientes para comparar el precio."

    if precio_actual <= 0:
        return "⚪ SIN DATOS", None, "El precio actual no es válido."

    potencial = (valor_razonable - precio_actual) / precio_actual * 100

    if potencial >= 30:
        estado = "🟢 MUY INFRAVALORADA"
        explicacion = "El modelo estima un valor considerablemente superior al precio actual."
    elif potencial >= 15:
        estado = "🟢 INFRAVALORADA"
        explicacion = "El modelo estima un valor superior al precio actual."
    elif potencial >= -10:
        estado = "🟡 VALORACIÓN RAZONABLE"
        explicacion = "El precio está relativamente cerca del valor estimado."
    elif potencial >= -25:
        estado = "🟠 SOBREVALORADA"
        explicacion = "El precio actual está por encima del valor estimado."
    else:
        estado = "🔴 MUY SOBREVALORADA"
        explicacion = "El precio actual está considerablemente por encima del valor estimado."

    return estado, potencial, explicacion
