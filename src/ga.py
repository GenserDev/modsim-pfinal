import itertools
from dataclasses import dataclass

import numpy as np

from src.modelo import COMPUESTOS, Estrategia, verificar_restricciones
from src.simulacion import mascara_safety_car, simular

MAX_PARADAS = 3
UMBRAL_MIN, UMBRAL_MAX = 5, 30


@dataclass
class Individuo:
    n_paradas: int
    vueltas: list
    compuestos: list
    umbral_sc: int = None
    compuesto_sc: str = "SOFT"
    aptitud: float = None
    violacion: float = None
    factible: bool = None

    def estrategia(self):
        return Estrategia(tuple(self.vueltas[:self.n_paradas]), tuple(self.compuestos[:self.n_paradas + 1]),
                          umbral_sc=self.umbral_sc, compuesto_sc=self.compuesto_sc)

    def clave(self):
        return (self.n_paradas, tuple(self.vueltas[:self.n_paradas]), tuple(self.compuestos[:self.n_paradas + 1]),
                self.umbral_sc, self.compuesto_sc if self.umbral_sc is not None else None)

    def copia(self):
        return Individuo(self.n_paradas, list(self.vueltas), list(self.compuestos), self.umbral_sc,
                         self.compuesto_sc, self.aptitud, self.violacion, self.factible)


class Evaluador:
    def __init__(self, parametros, escenarios, lambda_riesgo=0.0):
        self.parametros = parametros
        self.escenarios = escenarios
        self.lambda_riesgo = lambda_riesgo
        self.mascara = mascara_safety_car(parametros, escenarios)
        self.cache = {}
        self.evaluaciones = 0

    def objetivo(self, estrategia):
        totales = simular(estrategia, self.parametros, self.escenarios, mascara_sc=self.mascara)
        return float(totales.mean() + self.lambda_riesgo * totales.std(ddof=1))

    def evaluar(self, individuo):
        clave = individuo.clave()
        if clave not in self.cache:
            estrategia = individuo.estrategia()
            factible, _, violacion = verificar_restricciones(estrategia, self.parametros, devolver_violacion=True)
            aptitud = self.objetivo(estrategia) if factible else None
            self.cache[clave] = (factible, violacion, aptitud)
            self.evaluaciones += 1
        individuo.factible, individuo.violacion, individuo.aptitud = self.cache[clave]
        return individuo


def mejor_segun_deb(a, b):
    if a.factible and not b.factible:
        return a
    if b.factible and not a.factible:
        return b
    if a.factible and b.factible:
        return a if a.aptitud <= b.aptitud else b
    return a if a.violacion <= b.violacion else b


def orden_deb(individuo):
    return (0, individuo.aptitud, 0) if individuo.factible else (1, 0, individuo.violacion)


class AlgoritmoGenetico:
    def __init__(self, evaluador, rng, poblacion=60, generaciones=150, paciencia=20, torneo=3,
                 prob_cruce=0.9, prob_mutacion=0.3, elites=2, max_paradas=MAX_PARADAS,
                 movimiento_parada=3, reactiva=False):
        self.ev = evaluador
        self.rng = rng
        self.N = evaluador.parametros.N
        self.tam = poblacion
        self.generaciones = generaciones
        self.paciencia = paciencia
        self.torneo = torneo
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.elites = elites
        self.max_paradas = max_paradas
        self.movimiento = movimiento_parada
        self.reactiva = reactiva
        self.historial = []

    def aleatorio(self):
        n = int(self.rng.integers(1, self.max_paradas + 1))
        vueltas = sorted(self.rng.choice(np.arange(1, self.N), size=MAX_PARADAS, replace=False).tolist())
        activas = sorted(vueltas[:n])
        vueltas = activas + vueltas[n:]
        compuestos = [str(self.rng.choice(COMPUESTOS)) for _ in range(MAX_PARADAS + 1)]
        umbral = int(self.rng.integers(UMBRAL_MIN, UMBRAL_MAX + 1)) if self.reactiva else None
        return Individuo(n, vueltas, compuestos, umbral, str(self.rng.choice(COMPUESTOS)))

    def seleccionar(self, poblacion):
        indices = self.rng.choice(len(poblacion), size=self.torneo, replace=False)
        ganador = poblacion[indices[0]]
        for i in indices[1:]:
            ganador = mejor_segun_deb(ganador, poblacion[i])
        return ganador

    def cruzar(self, a, b):
        hijo1, hijo2 = a.copia(), b.copia()
        if self.rng.random() < self.prob_cruce:
            if self.rng.random() < 0.5:
                hijo1.n_paradas, hijo2.n_paradas = b.n_paradas, a.n_paradas
            for i in range(MAX_PARADAS):
                if self.rng.random() < 0.5:
                    hijo1.vueltas[i], hijo2.vueltas[i] = b.vueltas[i], a.vueltas[i]
            for i in range(MAX_PARADAS + 1):
                if self.rng.random() < 0.5:
                    hijo1.compuestos[i], hijo2.compuestos[i] = b.compuestos[i], a.compuestos[i]
            if self.reactiva and self.rng.random() < 0.5:
                hijo1.umbral_sc, hijo2.umbral_sc = b.umbral_sc, a.umbral_sc
                hijo1.compuesto_sc, hijo2.compuesto_sc = b.compuesto_sc, a.compuesto_sc
        return hijo1, hijo2

    def mutar(self, ind):
        if self.rng.random() >= self.prob_mutacion:
            return ind
        opciones = ["mover", "compuesto", "agregar_quitar"] + (["umbral"] if self.reactiva else [])
        tipo = self.rng.choice(opciones)
        if tipo == "mover":
            i = int(self.rng.integers(ind.n_paradas))
            paso = int(self.rng.integers(1, self.movimiento + 1)) * (1 if self.rng.random() < 0.5 else -1)
            ind.vueltas[i] = int(np.clip(ind.vueltas[i] + paso, 1, self.N - 1))
        elif tipo == "compuesto":
            i = int(self.rng.integers(ind.n_paradas + 1))
            ind.compuestos[i] = str(self.rng.choice([c for c in COMPUESTOS if c != ind.compuestos[i]]))
        elif tipo == "agregar_quitar":
            activas = ind.vueltas[:ind.n_paradas]
            puede_agregar = ind.n_paradas < self.max_paradas
            if puede_agregar and (ind.n_paradas == 1 or self.rng.random() < 0.5):
                activas = sorted(activas + [int(self.rng.integers(1, self.N))])
                posicion = int(self.rng.integers(ind.n_paradas + 2))
                ind.compuestos.insert(posicion, str(self.rng.choice(COMPUESTOS)))
                ind.compuestos = ind.compuestos[:MAX_PARADAS + 1]
                ind.n_paradas += 1
            elif ind.n_paradas > 1:
                quitar = int(self.rng.integers(ind.n_paradas))
                activas.pop(quitar)
                ind.compuestos.pop(int(self.rng.integers(ind.n_paradas + 1)))
                ind.compuestos.append(str(self.rng.choice(COMPUESTOS)))
                ind.n_paradas -= 1
            ind.vueltas = activas + ind.vueltas[len(activas):][:MAX_PARADAS - len(activas)]
            while len(ind.vueltas) < MAX_PARADAS:
                ind.vueltas.append(int(self.rng.integers(1, self.N)))
        else:
            if self.rng.random() < 0.7:
                ind.umbral_sc = int(np.clip(ind.umbral_sc + self.rng.integers(-3, 4), UMBRAL_MIN, UMBRAL_MAX))
            else:
                ind.compuesto_sc = str(self.rng.choice(COMPUESTOS))
        ind.aptitud = ind.violacion = ind.factible = None
        return ind

    def ejecutar(self, semillas_iniciales=()):
        poblacion = [self.ev.evaluar(i.copia()) for i in semillas_iniciales]
        poblacion += [self.ev.evaluar(self.aleatorio()) for _ in range(self.tam - len(poblacion))]
        mejor = min(poblacion, key=orden_deb).copia()
        sin_mejora = 0
        for generacion in range(self.generaciones):
            poblacion.sort(key=orden_deb)
            nueva = [p.copia() for p in poblacion[:self.elites]]
            while len(nueva) < self.tam:
                h1, h2 = self.cruzar(self.seleccionar(poblacion), self.seleccionar(poblacion))
                nueva += [self.ev.evaluar(self.mutar(h1)), self.ev.evaluar(self.mutar(h2))]
            poblacion = nueva[:self.tam]
            candidato = min(poblacion, key=orden_deb)
            if orden_deb(candidato) < orden_deb(mejor) and (not mejor.factible or candidato.aptitud < mejor.aptitud - 1e-9):
                mejor, sin_mejora = candidato.copia(), 0
            else:
                sin_mejora += 1
            factibles = [p.aptitud for p in poblacion if p.factible]
            self.historial.append({
                "generacion": generacion + 1,
                "mejor_s": mejor.aptitud if mejor.factible else None,
                "media_factibles_s": float(np.mean(factibles)) if factibles else None,
                "fraccion_factible": len(factibles) / len(poblacion),
                "evaluaciones": self.ev.evaluaciones,
            })
            if sin_mejora >= self.paciencia:
                break
        return mejor


def fuerza_bruta(evaluador, max_paradas=2):
    N = evaluador.parametros.N
    mejor, mejor_valor, evaluadas = None, np.inf, 0
    for n in range(1, max_paradas + 1):
        for vueltas in itertools.combinations(range(1, N), n):
            for compuestos in itertools.product(COMPUESTOS, repeat=n + 1):
                estrategia = Estrategia(vueltas, compuestos)
                if not verificar_restricciones(estrategia, evaluador.parametros)[0]:
                    continue
                valor = evaluador.objetivo(estrategia)
                evaluadas += 1
                if valor < mejor_valor:
                    mejor, mejor_valor = estrategia, valor
    return mejor, mejor_valor, evaluadas
