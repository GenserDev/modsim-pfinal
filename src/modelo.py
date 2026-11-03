import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

COMPUESTOS = ("SOFT", "MEDIUM", "HARD")
RAIZ = Path(__file__).resolve().parent.parent


@dataclass
class Parametros:
    N: int
    t_base: float
    delta: dict
    k: dict
    k2: dict
    f: float
    sigma: float
    extra_vuelta_1: float
    perdidas_pits: np.ndarray
    p_sc: float
    duraciones_sc: list
    factor_vuelta_sc: float
    factor_parada_sc: float
    vida_max: dict
    edad_inicial: dict = field(default_factory=lambda: {c: 0 for c in COMPUESTOS})
    factor_degradacion: float = 1.0


def cargar_parametros(ruta=None, edad_inicial=None):
    datos = json.loads(Path(ruta or RAIZ / "results" / "parametros.json").read_text())
    return Parametros(
        N=datos["N"]["valor"],
        t_base=datos["t_base"]["valor"],
        delta={c: datos["delta"][c]["valor"] for c in COMPUESTOS},
        k={c: datos["k"][c]["valor"] for c in COMPUESTOS},
        k2={c: datos["k2"][c]["valor"] for c in COMPUESTOS},
        f=datos["f"]["valor"],
        sigma=datos["sigma"]["valor"],
        extra_vuelta_1=datos["extra_vuelta_1"]["valor"],
        perdidas_pits=np.array(datos["perdidas_pits"]["valores"]),
        p_sc=datos["p_sc"]["valor"],
        duraciones_sc=datos["duraciones_sc"]["valores"],
        factor_vuelta_sc=datos["factor_vuelta_sc"]["valor"],
        factor_parada_sc=datos["factor_parada_sc"]["valor"],
        vida_max=datos["vida_max"]["valores"],
        edad_inicial=edad_inicial or {c: 0 for c in COMPUESTOS},
    )


def variar(parametros, **cambios):
    return replace(parametros, **cambios)


@dataclass
class Estrategia:
    paradas: tuple
    compuestos: tuple
    edades_iniciales: tuple = None
    umbral_sc: int = None
    compuesto_sc: str = "SOFT"

    def edades(self, parametros):
        if self.edades_iniciales is not None:
            return tuple(self.edades_iniciales)
        return tuple(0 if i == 0 else parametros.edad_inicial[c] for i, c in enumerate(self.compuestos))

    def __str__(self):
        texto = "-".join(c[0] for c in self.compuestos) + " @ " + ",".join(str(p) for p in self.paradas)
        if self.umbral_sc is not None:
            texto += f" | SC si edad>={self.umbral_sc} ({self.compuesto_sc[0]})"
        return texto


def degradacion(parametros, compuesto, edad):
    k, k2 = parametros.k[compuesto], parametros.k2[compuesto]
    edad = np.asarray(edad, dtype=float)
    if k2 < 0:
        # Con k2 negativo la parábola bajaría después del vértice, así que la degradación se congela ahí
        edad = np.minimum(edad, -k / (2 * k2))
    return parametros.factor_degradacion * (k * edad + k2 * edad ** 2)


def tiempo_verde(parametros, compuesto, edad, vuelta):
    vuelta = np.asarray(vuelta)
    tiempo = (parametros.t_base + parametros.delta[compuesto] + degradacion(parametros, compuesto, edad)
              + parametros.f * (parametros.N - vuelta))
    return tiempo + np.where(vuelta == 1, parametros.extra_vuelta_1, 0.0)


def largos_stints(estrategia, N):
    limites = (0,) + tuple(estrategia.paradas) + (N,)
    return [limites[i + 1] - limites[i] for i in range(len(limites) - 1)]


def verificar_restricciones(estrategia, parametros, devolver_violacion=False):
    paradas, compuestos, N = list(estrategia.paradas), list(estrategia.compuestos), parametros.N
    motivos, violacion = [], 0.0
    if not 1 <= len(paradas) <= 3:
        motivos.append("número de paradas fuera de [1, 3]")
        violacion += abs(len(paradas) - min(max(len(paradas), 1), 3)) * N
    if len(compuestos) != len(paradas) + 1 or any(c not in COMPUESTOS for c in compuestos):
        motivos.append("compuestos inválidos")
        return (False, motivos, float("inf")) if devolver_violacion else (False, motivos)
    if len(set(compuestos)) < 2:
        motivos.append("menos de 2 compuestos de seco distintos")
        violacion += N
    fuera = sum(max(0, 1 - p) + max(0, p - (N - 1)) for p in paradas)
    if fuera:
        motivos.append("vuelta de parada fuera de [1, N-1]")
        violacion += fuera
    no_crecientes = sum(max(0, paradas[i] - paradas[i + 1] + 1) for i in range(len(paradas) - 1))
    if no_crecientes:
        motivos.append("vueltas de parada no estrictamente crecientes")
        violacion += no_crecientes
    if not fuera and not no_crecientes:
        for largo, c, edad in zip(largos_stints(estrategia, N), compuestos, estrategia.edades(parametros)):
            exceso = edad + largo - parametros.vida_max[c]
            if exceso > 0:
                motivos.append(f"stint de {c} con {edad + largo} vueltas supera la vida máxima {parametros.vida_max[c]}")
                violacion += exceso
    valida = not motivos
    return (valida, motivos, violacion) if devolver_violacion else (valida, motivos)


def plan_por_vuelta(estrategia, parametros):
    N = parametros.N
    compuesto_vuelta, edad_vuelta = [], []
    edades = estrategia.edades(parametros)
    for i, (largo, c) in enumerate(zip(largos_stints(estrategia, N), estrategia.compuestos)):
        compuesto_vuelta += [c] * largo
        edad_vuelta += list(range(edades[i] + 1, edades[i] + largo + 1))
    return compuesto_vuelta, np.array(edad_vuelta)


def tiempos_verdes_estrategia(estrategia, parametros):
    compuestos, edades = plan_por_vuelta(estrategia, parametros)
    vueltas = np.arange(1, parametros.N + 1)
    return np.array([float(tiempo_verde(parametros, c, a, l)) for c, a, l in zip(compuestos, edades, vueltas)])
