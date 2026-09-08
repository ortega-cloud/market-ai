import numpy as np

def calcular_consenso(senal_market_ai, prediccion_ml, signal_hybrid, dcf_upside, signal_analyst, signal_tecnico):
    votos_alcistas = 0
    votos_bajistas = 0
    total_modelos = 0

    modelos = [
        ('MARKET_AI', senal_market_ai),
        ('ML', prediccion_ml),
        ('HYBRID', signal_hybrid),
        ('DCF', 'COMPRA' if dcf_upside and dcf_upside > 0.15 else ('VENTA' if dcf_upside and dcf_upside < -0.10 else 'NEUTRAL')),
        ('ANALISTAS', signal_analyst),
        ('TECNICO', signal_tecnico)
    ]

    for nombre, sig in modelos:
        if sig and sig != 'N/D':
            total_modelos += 1
            if sig in ['COMPRA FUERTE', 'COMPRA', 'ALCISTA']:
                votos_alcistas += 1
            elif sig in ['VENTA FUERTE', 'VENTA', 'BAJISTA']:
                votos_bajistas += 1

    if total_modelos == 0:
        return "N/D", 0.0

    pct_alcista = votos_alcistas / total_modelos
    pct_bajista = votos_bajistas / total_modelos

    if pct_alcista >= 0.8:
        consenso = "🟢 ALTO CONSENSO ALCISTA"
    elif pct_alcista >= 0.55:
        consenso = "🟢 CONSENSO ALCISTA"
    elif pct_bajista >= 0.8:
        consenso = "🔴 ALTO CONSENSO BAJISTA"
    elif pct_bajista >= 0.55:
        consenso = "🔴 CONSENSO BAJISTA"
    else:
        consenso = "🟡 SEÑALES MIXTAS"

    return consenso, pct_alcista

def determinar_horizonte(ml_5d, ml_20d, ml_60d, ml_120d):
    horizontes = {
        'CORTO (5D)': ml_5d,
        'MEDIO (20D)': ml_20d,
        'MEDIO (60D)': ml_60d,
        'LARGO (120D)': ml_120d
    }
    
    validados = {k: v for k, v in horizontes.items() if v is not None and v != 'N/D'}
    if not validados:
        return "N/D"

    # Selecciona el horizonte con mayor certidumbre/consistencia de dirección
    return max(validados, key=lambda k: abs(validados[k].get('probabilidad', 0.5) - 0.5))

def calcular_confianza(datos_disponibles, discrepancia_ml_score, dcf_valido, analistas_validos, riesgo_alto):
    confianza = 100.0

    # Penalización por datos faltantes (10% por cada módulo N/D)
    confianza -= (1.0 - datos_disponibles) * 40.0

    # Penalización por discrepancia ML vs Market AI Score
    if discrepancia_ml_score:
        confianza -= 20.0

    # Penalización por ausencia de DCF o Analistas
    if not dcf_valido:
        confianza -= 10.0
    if not analistas_validos:
        confianza -= 10.0

    # Penalización por riesgo elevado
    if riesgo_alto:
        confianza -= 15.0

    return max(0.0, min(100.0, confianza))

def generar_decision_market_ai(datos_ticker):
    """
    Función central que integra Fundamentales, Valoración, DCF, Analistas,
    Técnico, Score, ML (5D, 20D, 60D, 120D), Hybrid y Riesgo.
    """
    score = datos_ticker.get('score', 'N/D')
    ml_preds = datos_ticker.get('ml_preds', {})
    dcf = datos_ticker.get('dcf', None)
    analistas = datos_ticker.get('analistas', None)
    tecnico = datos_ticker.get('tecnico', {})
    riesgo = datos_ticker.get('riesgo', {})

    # 1. Conteo de disponibilidad de datos
    modulos = [score, ml_preds, dcf, analistas, tecnico, riesgo]
    disponibles = sum(1 for m in modulos if m is not None and m != 'N/D')
    cobertura_datos = disponibles / len(modulos)

    # 2. Horizonte recomendado
    horizonte = determinar_horizonte(
        ml_preds.get('5D'), ml_preds.get('20D'), 
        ml_preds.get('60D'), ml_preds.get('120D')
    )

    # 3. Consenso entre modelos
    dcf_upside = dcf.get('upside') if dcf else None
    analyst_sig = analistas.get('recomendacion') if analistas else 'N/D'
    tech_sig = tecnico.get('senal') if tecnico else 'N/D'
    ml_sig = ml_preds.get('general_signal', 'N/D')
    
    consenso, pct_alcista = calcular_consenso(
        score.get('senal') if score != 'N/D' else 'N/D',
        ml_sig,
        datos_ticker.get('hybrid_signal', 'N/D'),
        dcf_upside,
        analyst_sig,
        tech_sig
    )

    # 4. Determinación de la señal final
    if pct_alcista >= 0.75:
        senal_final = "COMPRA FUERTE"
    elif pct_alcista >= 0.55:
        senal_final = "COMPRA"
    elif pct_alcista <= 0.25:
        senal_final = "VENTA FUERTE"
    elif pct_alcista <= 0.45:
        senal_final = "VENTA"
    else:
        senal_final = "MANTENER"

    # 5. Cálculo de Confianza
    discrepancia = (score != 'N/D' and score.get('senal') != ml_sig)
    riesgo_elevado = riesgo.get('nivel') == 'ALTO' if riesgo else False
    
    confianza = calcular_confianza(
        cobertura_datos,
        discrepancia,
        dcf is not None,
        analistas is not None,
        riesgo_elevado
    )

    # 6. Extracción de motivos principales
    motivos = []
    if score != 'N/D': motivos.append(f"Score global de {score.get('puntuacion', 'N/D')}/100")
    if dcf_upside: motivos.append(f"Potencial DCF estimado: {dcf_upside*100:.1f}%")
    if ml_sig != 'N/D': motivos.append(f"Tendencia ML agregada: {ml_sig}")

    return {
        "senal_final": senal_final,
        "confianza": round(confianza, 1),
        "horizonte": horizonte,
        "score": score.get('puntuacion') if isinstance(score, dict) else 'N/D',
        "valoracion": dcf.get('valor_intrinseco') if dcf else 'N/D',
        "tendencia": tech_sig,
        "prediccion_ml": ml_sig,
        "riesgo": riesgo.get('nivel', 'N/D') if riesgo else 'N/D',
        "consenso": consenso,
        "motivos": motivos
    }
