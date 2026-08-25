import numpy as np


def calcular_dcf(
    free_cash_flow,
    crecimiento_inicial=0.10,
    crecimiento_largo_plazo=0.03,
    tasa_descuento=0.10,
    años=5,
    deuda=0,
    caja=0,
    acciones=1
):

    if free_cash_flow is None:
        return None

    if free_cash_flow <= 0:
        return None

    if acciones is None or acciones <= 0:
        return None

    flujos = []

    fcf = float(free_cash_flow)

    for año in range(1, años + 1):

        crecimiento = (
            crecimiento_inicial
            - (
                (
                    crecimiento_inicial
                    - crecimiento_largo_plazo
                )
                * (año - 1)
                / max(años - 1, 1)
            )
        )

        fcf = fcf * (1 + crecimiento)

        flujos.append(
            {
                "año": año,
                "crecimiento": crecimiento,
                "fcf": fcf
            }
        )


    valor_presente_flujos = 0

    for flujo in flujos:

        año = flujo["año"]

        fcf_año = flujo["fcf"]

        valor_presente = (
            fcf_año
            / (
                (1 + tasa_descuento)
                ** año
            )
        )

        valor_presente_flujos += (
            valor_presente
        )


    fcf_final = flujos[-1]["fcf"]

    fcf_terminal = (
        fcf_final
        * (
            1
            + crecimiento_largo_plazo
        )
    )


    valor_terminal = (
        fcf_terminal
        / (
            tasa_descuento
            - crecimiento_largo_plazo
        )
    )


    valor_presente_terminal = (
        valor_terminal
        / (
            (1 + tasa_descuento)
            ** años
        )
    )


    enterprise_value = (
        valor_presente_flujos
        + valor_presente_terminal
    )


    equity_value = (
        enterprise_value
        - deuda
        + caja
    )


    valor_por_accion = (
        equity_value
        / acciones
    )


    return {
        "flujos": flujos,
        "valor_presente_flujos": (
            valor_presente_flujos
        ),
        "valor_terminal": (
            valor_terminal
        ),
        "valor_presente_terminal": (
            valor_presente_terminal
        ),
        "enterprise_value": (
            enterprise_value
        ),
        "equity_value": (
            equity_value
        ),
        "valor_por_accion": (
            valor_por_accion
        )
    }


def calcular_escenarios_dcf(
    free_cash_flow,
    deuda=0,
    caja=0,
    acciones=1
):

    escenario_pesimista = calcular_dcf(
        free_cash_flow=free_cash_flow,
        crecimiento_inicial=0.05,
        crecimiento_largo_plazo=0.02,
        tasa_descuento=0.12,
        años=5,
        deuda=deuda,
        caja=caja,
        acciones=acciones
    )


    escenario_base = calcular_dcf(
        free_cash_flow=free_cash_flow,
        crecimiento_inicial=0.10,
        crecimiento_largo_plazo=0.03,
        tasa_descuento=0.10,
        años=5,
        deuda=deuda,
        caja=caja,
        acciones=acciones
    )


    escenario_optimista = calcular_dcf(
        free_cash_flow=free_cash_flow,
        crecimiento_inicial=0.15,
        crecimiento_largo_plazo=0.04,
        tasa_descuento=0.09,
        años=5,
        deuda=deuda,
        caja=caja,
        acciones=acciones
    )


    return {
        "pesimista": escenario_pesimista,
        "base": escenario_base,
        "optimista": escenario_optimista
    }


def diagnosticar_valoracion(
    precio_actual,
    valor_razonable
):

    if precio_actual is None:
        return (
            "⚪ SIN DATOS",
            None,
            "No hay precio actual disponible."
        )


    if valor_razonable is None:
        return (
            "⚪ SIN DATOS",
            None,
            "No se ha podido calcular el valor razonable."
        )


    potencial = (
        (
            valor_razonable
            - precio_actual
        )
        / precio_actual
    ) * 100


    if potencial >= 30:

        estado = (
            "🟢 MUY INFRAVALORADA"
        )

        explicacion = (
            "El modelo estima un valor "
            "considerablemente superior "
            "al precio actual."
        )


    elif potencial >= 15:

        estado = (
            "🟢 INFRAVALORADA"
        )

        explicacion = (
            "El modelo estima un valor "
            "superior al precio actual."
        )


    elif potencial >= -10:

        estado = (
            "🟡 VALORACIÓN RAZONABLE"
        )

        explicacion = (
            "El precio se encuentra "
            "relativamente cerca del valor "
            "estimado."
        )


    elif potencial >= -25:

        estado = (
            "🟠 SOBREVALORADA"
        )

        explicacion = (
            "El precio actual está por encima "
            "del valor estimado."
        )


    else:

        estado = (
            "🔴 MUY SOBREVALORADA"
        )

        explicacion = (
            "El precio actual está "
            "considerablemente por encima "
            "del valor estimado."
        )


    return (
        estado,
        potencial,
        explicacion
    )
