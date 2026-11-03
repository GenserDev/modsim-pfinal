from dataclasses import dataclass

import numpy as np

from src.modelo import COMPUESTOS, degradacion, tiempo_verde, tiempos_verdes_estrategia

MAX_PARADAS_EXTRA = 4


@dataclass
class Escenarios:
    ruido: np.ndarray
    uniformes_sc: np.ndarray
    indices_duracion: np.ndarray
    uniformes_pits: np.ndarray
    semilla: int


def generar_escenarios(N, replicas, semilla):
    rng = np.random.default_rng(semilla)
    return Escenarios(
        ruido=rng.standard_normal((replicas, N)),
        uniformes_sc=rng.random((replicas, N)),
        indices_duracion=rng.random((replicas, N)),
        uniformes_pits=rng.random((replicas, MAX_PARADAS_EXTRA)),
        semilla=semilla,
    )


def mascara_safety_car(parametros, escenarios):
    replicas, N = escenarios.uniformes_sc.shape
    duraciones = np.array(parametros.duraciones_sc)
    mascara = np.zeros((replicas, N), dtype=bool)
    restantes = np.zeros(replicas, dtype=int)
    for l in range(N):
        inicia = (restantes == 0) & (escenarios.uniformes_sc[:, l] < parametros.p_sc)
        duracion = duraciones[(escenarios.indices_duracion[:, l] * len(duraciones)).astype(int)]
        restantes = np.where(inicia, duracion, restantes)
        mascara[:, l] = restantes > 0
        restantes = np.maximum(restantes - 1, 0)
    return mascara


def perdidas_parada(parametros, escenarios):
    perdidas = np.sort(parametros.perdidas_pits)
    indices = (escenarios.uniformes_pits * len(perdidas)).astype(int)
    return perdidas[indices]


def simular(estrategia, parametros, escenarios, con_ruido=True, mascara_sc=None, por_vuelta=False):
    if mascara_sc is None:
        mascara_sc = mascara_safety_car(parametros, escenarios)
    if estrategia.umbral_sc is not None:
        return _simular_reactiva(estrategia, parametros, escenarios, con_ruido, mascara_sc, por_vuelta)
    verdes = tiempos_verdes_estrategia(estrategia, parametros)
    vueltas = np.where(mascara_sc, parametros.factor_vuelta_sc * verdes, verdes)
    if con_ruido:
        vueltas = vueltas + parametros.sigma * escenarios.ruido
    perdidas = perdidas_parada(parametros, escenarios)
    for i, p in enumerate(estrategia.paradas):
        factor = np.where(mascara_sc[:, p - 1], parametros.factor_parada_sc, 1.0)
        vueltas[:, p - 1] += factor * perdidas[:, i]
    return vueltas if por_vuelta else vueltas.sum(axis=1)


def _simular_reactiva(estrategia, parametros, escenarios, con_ruido, mascara_sc, por_vuelta):
    replicas, N = mascara_sc.shape
    planeadas = list(estrategia.paradas)
    compuestos = list(estrategia.compuestos)
    edades_plan = estrategia.edades(parametros)
    indice_sc = COMPUESTOS.index(estrategia.compuesto_sc)
    perdidas = perdidas_parada(parametros, escenarios)

    tabla_k = np.array([[tiempo_verde(parametros, c, a, 2) - parametros.f * (N - 2)
                         for a in range(0, 80)] for c in COMPUESTOS])
    vida = np.array([parametros.vida_max[c] for c in COMPUESTOS])
    plan_compuestos = np.array([COMPUESTOS.index(c) for c in compuestos])
    limites = np.array(planeadas + [N])

    compuesto = np.full(replicas, plan_compuestos[0])
    edad = np.full(replicas, edades_plan[0])
    siguiente = np.zeros(replicas, dtype=int)
    paradas_hechas = np.zeros(replicas, dtype=int)
    vueltas = np.zeros((replicas, N))
    filas = np.arange(replicas)

    for l in range(1, N + 1):
        edad = edad + 1
        verde = tabla_k[compuesto, np.minimum(edad, 79)] + parametros.f * (N - l)
        if l == 1:
            verde = verde + parametros.extra_vuelta_1
        sc = mascara_sc[:, l - 1]
        tiempo = np.where(sc, parametros.factor_vuelta_sc * verde, verde)
        if con_ruido:
            tiempo = tiempo + parametros.sigma * escenarios.ruido[:, l - 1]

        quedan_planeadas = siguiente < len(planeadas)
        parada_planeada = quedan_planeadas & (limites[np.minimum(siguiente, len(planeadas))] == l)
        dispara = sc & (edad >= estrategia.umbral_sc) & ~parada_planeada & (l <= N - 1)
        siguiente_comp = plan_compuestos[np.minimum(siguiente + 1, len(compuestos) - 1)]
        fin_stint_adelantado = limites[np.minimum(siguiente + 1, len(planeadas))]
        edad_siguiente = np.array(edades_plan)[np.minimum(siguiente + 1, len(compuestos) - 1)]
        adelanta = dispara & quedan_planeadas & (edad_siguiente + fin_stint_adelantado - l <= vida[siguiente_comp])
        agrega = (dispara & ~quedan_planeadas & (parametros.edad_inicial[estrategia.compuesto_sc] + N - l <= vida[indice_sc])
                  & (paradas_hechas < MAX_PARADAS_EXTRA))
        para = parada_planeada | adelanta | agrega

        indice_perdida = np.minimum(paradas_hechas, MAX_PARADAS_EXTRA - 1)
        factor = np.where(sc, parametros.factor_parada_sc, 1.0)
        tiempo = tiempo + np.where(para, factor * perdidas[filas, indice_perdida], 0.0)
        vueltas[:, l - 1] = tiempo

        nuevo_planeado = parada_planeada | adelanta
        compuesto = np.where(nuevo_planeado, siguiente_comp, np.where(agrega, indice_sc, compuesto))
        edad = np.where(nuevo_planeado, edad_siguiente, np.where(agrega, parametros.edad_inicial[estrategia.compuesto_sc], edad))
        siguiente = siguiente + nuevo_planeado
        paradas_hechas = paradas_hechas + para
    return vueltas if por_vuelta else vueltas.sum(axis=1)


def resumen(totales, referencia=None):
    media = float(np.mean(totales))
    de = float(np.std(totales, ddof=1))
    mitad = 1.96 * de / np.sqrt(len(totales))
    datos = {"media_s": media, "de_s": de, "ic95_inf_s": float(media - mitad), "ic95_sup_s": float(media + mitad), "n": len(totales)}
    if referencia is not None:
        diferencia = totales - referencia
        mitad_dif = 1.96 * np.std(diferencia, ddof=1) / np.sqrt(len(diferencia))
        datos.update({"diferencia_media_s": float(diferencia.mean()),
                      "diferencia_ic95_inf_s": float(diferencia.mean() - mitad_dif),
                      "diferencia_ic95_sup_s": float(diferencia.mean() + mitad_dif),
                      "prob_mejor_que_referencia": float(np.mean(totales < referencia))})
    return datos
