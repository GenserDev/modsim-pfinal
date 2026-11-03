import numpy as np
import pytest

from src.ga import AlgoritmoGenetico, Evaluador, fuerza_bruta
from src.modelo import (COMPUESTOS, Estrategia, cargar_parametros, largos_stints, plan_por_vuelta, variar,
                        verificar_restricciones)
from src.simulacion import generar_escenarios, mascara_safety_car, simular

EDADES_GA = {"SOFT": 3, "MEDIUM": 0, "HARD": 0}
BASE = Estrategia((22, 49, 55), ("SOFT", "MEDIUM", "SOFT", "SOFT"), (0, 0, 3, 3))


@pytest.fixture
def parametros():
    return cargar_parametros(edad_inicial=EDADES_GA)


@pytest.fixture
def escenarios(parametros):
    return generar_escenarios(parametros.N, 100, 123)


@pytest.mark.parametrize("estrategia", [
    BASE,
    Estrategia((30,), ("MEDIUM", "HARD")),
    Estrategia((15, 40), ("SOFT", "MEDIUM", "HARD")),
])
def test_conservacion_de_vueltas(parametros, estrategia):
    assert sum(largos_stints(estrategia, parametros.N)) == parametros.N
    compuestos, edades = plan_por_vuelta(estrategia, parametros)
    assert len(compuestos) == len(edades) == parametros.N


def test_degradacion_cero_empata_estrategias_de_una_parada(parametros, escenarios):
    planos = variar(parametros, k={c: 0.0 for c in COMPUESTOS}, k2={c: 0.0 for c in COMPUESTOS},
                    delta={c: 0.0 for c in COMPUESTOS}, edad_inicial={c: 0 for c in COMPUESTOS}, p_sc=0.0)
    totales = [simular(Estrategia((l,), ("MEDIUM", "HARD")), planos, escenarios).mean() for l in range(16, 35)]
    totales += [simular(Estrategia((l,), ("SOFT", "HARD")), planos, escenarios).mean() for l in range(16, 25)]
    assert np.ptp(totales) < 1e-9


def test_determinismo_sin_ruido(parametros):
    sin_ruido = variar(parametros, sigma=0.0, p_sc=0.0, perdidas_pits=np.array([23.0]))
    a = simular(BASE, sin_ruido, generar_escenarios(sin_ruido.N, 50, 1))
    b = simular(BASE, sin_ruido, generar_escenarios(sin_ruido.N, 50, 999))
    assert np.ptp(a) < 1e-9 and np.allclose(a, b)


def test_misma_semilla_mismo_resultado(parametros):
    a = simular(BASE, parametros, generar_escenarios(parametros.N, 200, 5))
    b = simular(BASE, parametros, generar_escenarios(parametros.N, 200, 5))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("estrategia, motivo", [
    (Estrategia((30,), ("MEDIUM", "MEDIUM")), "compuestos"),
    (Estrategia((), ("HARD",)), "número de paradas"),
    (Estrategia((10, 20, 30, 40), ("SOFT", "MEDIUM", "HARD", "SOFT", "MEDIUM")), "número de paradas"),
    (Estrategia((40, 20), ("SOFT", "MEDIUM", "HARD")), "crecientes"),
    (Estrategia((20, 20), ("SOFT", "MEDIUM", "HARD")), "crecientes"),
    (Estrategia((0,), ("SOFT", "HARD")), "fuera"),
    (Estrategia((66,), ("SOFT", "HARD")), "fuera"),
    (Estrategia((40,), ("SOFT", "HARD")), "vida máxima"),
    (Estrategia((30,), ("SOFT", "INTERMEDIATE")), "inválidos"),
])
def test_verificador_rechaza_invalidas(parametros, estrategia, motivo):
    valida, motivos = verificar_restricciones(estrategia, parametros)
    assert not valida
    assert any(motivo in m for m in motivos)


def test_verificador_acepta_base(parametros):
    assert verificar_restricciones(BASE, parametros) == (True, [])


def test_mascara_sc_extremos(parametros, escenarios):
    assert not mascara_safety_car(variar(parametros, p_sc=0.0), escenarios).any()
    assert mascara_safety_car(variar(parametros, p_sc=1.0), escenarios).all()


def test_reactiva_sin_disparo_igual_a_fija(parametros, escenarios):
    fija = Estrategia((20, 45), ("SOFT", "MEDIUM", "HARD"))
    reactiva = Estrategia((20, 45), ("SOFT", "MEDIUM", "HARD"), umbral_sc=999)
    con_sc = variar(parametros, p_sc=0.05)
    assert np.allclose(simular(fija, con_sc, escenarios), simular(reactiva, con_sc, escenarios))


def test_sc_hace_vueltas_mas_lentas(parametros, escenarios):
    sin_sc = simular(BASE, variar(parametros, p_sc=0.0), escenarios).mean()
    con_sc = simular(BASE, variar(parametros, p_sc=0.02), escenarios).mean()
    assert con_sc > sin_sc


def test_ga_iguala_fuerza_bruta(parametros):
    evaluador = Evaluador(parametros, generar_escenarios(parametros.N, 50, 3))
    optimo, valor, _ = fuerza_bruta(evaluador, max_paradas=2)
    mejor = AlgoritmoGenetico(Evaluador(parametros, evaluador.escenarios), np.random.default_rng(1),
                              poblacion=100, max_paradas=2).ejecutar()
    assert verificar_restricciones(mejor.estrategia(), parametros)[0]
    assert mejor.aptitud == pytest.approx(valor)
