import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibracion import cargar_config
from src.ga import AlgoritmoGenetico, Evaluador, fuerza_bruta
from src.modelo import Estrategia, cargar_parametros, largos_stints, variar, verificar_restricciones
from src.simulacion import generar_escenarios, mascara_safety_car, resumen, simular

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "results"
SEMILLA_SELECCION = 5000


def estrategia_base(cfg):
    b = cfg["estrategia_base"]
    return Estrategia(tuple(b["paradas"]), tuple(b["compuestos"]), tuple(b["edades_iniciales"]))


def mascara_observada(parametros, replicas, cfg):
    neutral = pd.read_csv(RAIZ / "data" / "processed" / "neutralizaciones.csv")
    sc = neutral[(neutral["anio"] == cfg["anio_validacion"]) & (neutral["tipo"] == "SC")]
    mascara = np.zeros((replicas, parametros.N), dtype=bool)
    for _, p in sc.iterrows():
        mascara[:, int(p["vuelta_inicio"]) - 1:int(p["vuelta_fin"])] = True
    return mascara


def validar(parametros, cfg, base):
    vueltas = pd.read_csv(RAIZ / "data" / "processed" / "vueltas.csv")
    resultados = pd.read_csv(RAIZ / "data" / "processed" / "resultados.csv")
    reales = vueltas[(vueltas["anio"] == cfg["anio_validacion"]) & (vueltas["piloto"] == cfg["piloto_validacion"])]
    reales = reales.sort_values("vuelta").reset_index(drop=True)
    total_oficial = resultados[(resultados["anio"] == cfg["anio_validacion"])
                               & (resultados["piloto"] == cfg["piloto_validacion"])]["tiempo_total_s"].iloc[0]

    mediana_pits = float(np.median(parametros.perdidas_pits))
    deterministico = variar(parametros, sigma=0.0, perdidas_pits=np.array([mediana_pits]))
    escenario = generar_escenarios(parametros.N, 1, 0)
    simuladas = simular(base, deterministico, escenario, mascara_sc=mascara_observada(parametros, 1, cfg),
                        por_vuelta=True)[0]

    tabla = reales[["vuelta", "tiempo_s", "compuesto", "edad_llanta", "entrada_pits", "salida_pits", "neutralizada"]].copy()
    tabla["simulado_s"] = simuladas
    tabla["error_s"] = tabla["simulado_s"] - tabla["tiempo_s"]
    limpias = ~tabla["entrada_pits"] & ~tabla["salida_pits"] & ~tabla["neutralizada"] & (tabla["vuelta"] > 1)
    tabla["limpia"] = limpias
    tabla.to_csv(RESULTADOS / "validacion_vueltas.csv", index=False)

    replicas = cfg["simulacion"]["replicas"]
    escenarios = generar_escenarios(parametros.N, replicas, cfg["simulacion"]["semilla"])
    estocastico = simular(base, parametros, escenarios, mascara_sc=mascara_observada(parametros, replicas, cfg))
    percentil_real = float(np.mean(estocastico <= total_oficial))

    validacion = {
        "fuente": "OpenF1 (vueltas) y Jolpica-F1 (tiempo oficial)",
        "carrera": f"GP de España {cfg['anio_validacion']}, {cfg['piloto_validacion']}, {parametros.N} vueltas",
        "vueltas_limpias": int(limpias.sum()),
        "mae_vuelta_limpia_s": float(tabla.loc[limpias, "error_s"].abs().mean()),
        "sesgo_vuelta_limpia_s": float(tabla.loc[limpias, "error_s"].mean()),
        "mae_todas_s": float(tabla["error_s"].abs().mean()),
        "rmse_vuelta_limpia_s": float(np.sqrt((tabla.loc[limpias, "error_s"] ** 2).mean())),
        "suma_vueltas_real_s": float(tabla["tiempo_s"].sum()),
        "total_oficial_s": float(total_oficial),
        "total_simulado_deterministico_s": float(simuladas.sum()),
        "error_total_s": float(simuladas.sum() - total_oficial),
        "error_total_pct": float((simuladas.sum() - total_oficial) / total_oficial * 100),
        "error_bloques": {
            "stint_" + str(i + 1): float(tabla.loc[limpias & (tabla["vuelta"].between(ini, fin)), "error_s"].mean())
            for i, (ini, fin) in enumerate(zip((1,) + tuple(p + 1 for p in base.paradas),
                                               tuple(base.paradas) + (parametros.N,)))
        },
        "error_periodo_sc_s": float(tabla.loc[tabla["neutralizada"], "error_s"].sum()),
        "diagnostico_sc": diagnostico_sc(tabla, parametros),
        "simulacion_sc_observado": resumen(estocastico),
        "percentil_del_real_en_simulacion": percentil_real,
    }
    return validacion


def diagnostico_sc(tabla, parametros):
    # La última vuelta marcada es la de relanzamiento, que ya corre en verde
    relanzamiento = tabla.loc[tabla["neutralizada"], "vuelta"].max()
    sc = tabla[tabla["neutralizada"] & ~tabla["entrada_pits"] & ~tabla["salida_pits"] & (tabla["vuelta"] != relanzamiento)]
    verde_equivalente = sc["simulado_s"] / parametros.factor_vuelta_sc
    pits = tabla[tabla["neutralizada"] & (tabla["entrada_pits"] | tabla["salida_pits"])]
    return {
        "vueltas_sc_sin_pits": sc["vuelta"].tolist(),
        "factor_observado_lider": float((sc["tiempo_s"] / verde_equivalente).mean()),
        "factor_calibrado_2021": parametros.factor_vuelta_sc,
        "error_vueltas_sc_sin_pits_s": float(sc["error_s"].sum()),
        "error_vueltas_pits_bajo_sc_s": float(pits["error_s"].sum()),
        "nota": "Diagnóstico solamente. No se usa para calibrar porque 2025 es el año de validación",
    }


def correr_ga(parametros, cfg, corridas, reactiva=False, max_paradas=None, etiqueta=""):
    g = cfg["ga"]
    resultados, historiales, candidatas = [], [], {}
    for i in range(corridas):
        escenarios = generar_escenarios(parametros.N, g["replicas_fitness"], 1000 + i)
        evaluador = Evaluador(parametros, escenarios, g["lambda_riesgo"])
        ga = AlgoritmoGenetico(evaluador, np.random.default_rng(g["semilla"] + i), poblacion=g["poblacion"],
                               generaciones=g["generaciones"], paciencia=g["paciencia"], torneo=g["torneo"],
                               prob_cruce=g["prob_cruce"], prob_mutacion=g["prob_mutacion"], elites=g["elites"],
                               max_paradas=max_paradas or g["max_paradas"], movimiento_parada=g["movimiento_parada"],
                               reactiva=reactiva)
        inicio = time.time()
        mejor = ga.ejecutar()
        estrategia = mejor.estrategia()
        resultados.append({"condicion": etiqueta, "corrida": i, "semilla_ga": g["semilla"] + i,
                           "semilla_fitness": 1000 + i, "estrategia": str(estrategia),
                           "aptitud_s": mejor.aptitud, "generaciones": len(ga.historial),
                           "evaluaciones": evaluador.evaluaciones, "segundos": time.time() - inicio,
                           "factible": verificar_restricciones(estrategia, parametros)[0]})
        candidatas[str(estrategia)] = estrategia
        for h in ga.historial:
            historiales.append({"condicion": etiqueta, "corrida": i, **h})
    seleccion = Evaluador(parametros, generar_escenarios(parametros.N, cfg["simulacion"]["replicas"], SEMILLA_SELECCION))
    valores = {texto: seleccion.objetivo(e) for texto, e in candidatas.items()}
    tabla = pd.DataFrame(resultados)
    tabla["valor_seleccion_s"] = tabla["estrategia"].map(valores)
    return candidatas[min(valores, key=valores.get)], tabla, pd.DataFrame(historiales)


def evaluar_condicion(nombre, estrategia, parametros, escenarios, referencia=None, mascara=None):
    totales = simular(estrategia, parametros, escenarios, mascara_sc=mascara)
    fila = {"condicion": nombre, "estrategia": str(estrategia),
            "factible": verificar_restricciones(estrategia, parametros)[0], **resumen(totales, referencia)}
    return fila, totales


def mapa_calor(parametros, escenarios, compuestos):
    evaluador = Evaluador(parametros, escenarios)
    N = parametros.N
    filas = []
    for l1 in range(1, N - 1):
        for l2 in range(l1 + 1, N):
            e = Estrategia((l1, l2), compuestos)
            valido = verificar_restricciones(e, parametros)[0]
            filas.append({"parada_1": l1, "parada_2": l2, "factible": valido,
                          "tiempo_esperado_s": evaluador.objetivo(e) if valido else np.nan})
    return pd.DataFrame(filas)


def principal():
    cfg = cargar_config()
    RESULTADOS.mkdir(exist_ok=True)
    nominal = cargar_parametros(edad_inicial=cfg["edad_inicial_ga"])
    base = estrategia_base(cfg)
    planificada = Estrategia(base.paradas[:-1], base.compuestos[:-1], base.edades_iniciales[:-1])
    replicas, semilla = cfg["simulacion"]["replicas"], cfg["simulacion"]["semilla"]
    corridas = cfg["ga"]["corridas_independientes"]
    exp = cfg["experimentos"]

    validacion = validar(nominal, cfg, base)
    (RESULTADOS / "validacion.json").write_text(json.dumps(validacion, indent=2, ensure_ascii=False))
    print("validacion", {k: v for k, v in validacion.items() if not isinstance(v, dict)})

    evaluador_fb = Evaluador(nominal, generar_escenarios(nominal.N, cfg["ga"]["replicas_fitness"], 1000))
    inicio = time.time()
    optimo_fb, valor_fb, n_fb = fuerza_bruta(evaluador_fb, max_paradas=2)
    _, tabla_ga2, _ = correr_ga(nominal, cfg, corridas, max_paradas=2, etiqueta="GA_1_2_paradas")
    tabla_ga2_misma = []
    for i in range(corridas):
        ga = AlgoritmoGenetico(Evaluador(nominal, evaluador_fb.escenarios), np.random.default_rng(cfg["ga"]["semilla"] + i),
                               poblacion=cfg["ga"]["poblacion"], max_paradas=2, paciencia=cfg["ga"]["paciencia"],
                               prob_mutacion=cfg["ga"]["prob_mutacion"])
        tabla_ga2_misma.append(ga.ejecutar().aptitud)
    aciertos = int(np.sum(np.isclose(tabla_ga2_misma, valor_fb)))
    verificacion = {"optimo_fuerza_bruta": str(optimo_fb), "valor_s": valor_fb, "estrategias_evaluadas": n_fb,
                    "segundos": time.time() - inicio, "semilla_escenarios": 1000,
                    "ga_aciertos": aciertos, "ga_corridas": corridas,
                    "ga_peor_diferencia_s": float(np.max(np.array(tabla_ga2_misma) - valor_fb))}
    (RESULTADOS / "verificacion_fuerza_bruta.json").write_text(json.dumps(verificacion, indent=2, ensure_ascii=False))
    print("fuerza bruta", verificacion)

    condiciones = {
        "nominal": nominal,
        "p_sc_x2": variar(nominal, p_sc=nominal.p_sc * exp["factor_sc"]),
        "degradacion_+30%": variar(nominal, factor_degradacion=exp["factor_degradacion"]),
    }
    estrategias_ga, tablas, historiales = {}, [], []
    for nombre, params in condiciones.items():
        for reactiva in (False, True):
            etiqueta = f"{nombre}_{'reactiva' if reactiva else 'fija'}"
            mejor, tabla, historial = correr_ga(params, cfg, corridas, reactiva=reactiva, etiqueta=etiqueta)
            estrategias_ga[etiqueta] = mejor
            tablas.append(tabla)
            historiales.append(historial)
            print(etiqueta, mejor, Counter(tabla["estrategia"]).most_common(3))
    pd.concat(tablas).to_csv(RESULTADOS / "ga_corridas.csv", index=False)
    pd.concat(historiales).to_csv(RESULTADOS / "ga_convergencia.csv", index=False)

    escenarios = generar_escenarios(nominal.N, replicas, semilla)
    filas, totales = [], {}
    for nombre, params in condiciones.items():
        fila, t_base = evaluar_condicion("base_real_2025", base, params, escenarios)
        fila["parametros"] = nombre
        filas.append(fila)
        totales[(nombre, "base_real_2025")] = t_base
        evaluadas = {"base_planificada_2025": planificada, "ga_fija": estrategias_ga[f"{nombre}_fija"],
                     "ga_reactiva": estrategias_ga[f"{nombre}_reactiva"]}
        if nombre != "nominal":
            evaluadas["ga_fija_nominal"] = estrategias_ga["nominal_fija"]
            evaluadas["ga_reactiva_nominal"] = estrategias_ga["nominal_reactiva"]
        for etiqueta, estrategia in evaluadas.items():
            fila, t = evaluar_condicion(etiqueta, estrategia, params, escenarios, referencia=t_base)
            fila["parametros"] = nombre
            filas.append(fila)
            totales[(nombre, etiqueta)] = t

    for factor in (0.3, 0.7):
        params = variar(nominal, factor_parada_sc=factor)
        _, t_base = evaluar_condicion("base_real_2025", base, params, escenarios)
        for etiqueta in ("ga_fija", "ga_reactiva"):
            clave = "nominal_fija" if etiqueta == "ga_fija" else "nominal_reactiva"
            fila, _ = evaluar_condicion(etiqueta, estrategias_ga[clave], params, escenarios, referencia=t_base)
            fila["parametros"] = f"factor_parada_sc_{factor}"
            filas.append(fila)

    mascara = mascara_observada(nominal, replicas, cfg)
    _, t_base_obs = evaluar_condicion("base_real_2025", base, nominal, escenarios, mascara=mascara)
    comparacion_real = [{"condicion": "base_real_2025", "estrategia": str(base), **resumen(t_base_obs)}]
    fila, _ = evaluar_condicion("base_planificada_2025", planificada, nominal, escenarios, referencia=t_base_obs,
                                mascara=mascara)
    comparacion_real.append(fila)
    for etiqueta in ("nominal_fija", "nominal_reactiva"):
        fila, _ = evaluar_condicion(f"ga_{etiqueta.split('_')[1]}", estrategias_ga[etiqueta], nominal, escenarios,
                                    referencia=t_base_obs, mascara=mascara)
        comparacion_real.append(fila)
    pd.DataFrame(comparacion_real).to_csv(RESULTADOS / "comparacion_sc_observado.csv", index=False)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "experimentos.csv", index=False)
    pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in totales.items()}).to_csv(RESULTADOS / "totales_por_replica.csv", index=False)

    optimo_2, valor_2, _ = fuerza_bruta(evaluador_fb, max_paradas=2, min_paradas=2)
    verificacion["optimo_2_paradas"] = str(optimo_2)
    verificacion["valor_2_paradas_s"] = valor_2
    (RESULTADOS / "verificacion_fuerza_bruta.json").write_text(json.dumps(verificacion, indent=2, ensure_ascii=False))
    compuestos_mapa = optimo_2.compuestos
    mapa_calor(nominal, evaluador_fb.escenarios, compuestos_mapa).assign(
        compuestos="-".join(compuestos_mapa)).to_csv(RESULTADOS / "mapa_calor.csv", index=False)

    semillas = {"evaluacion_final": semilla, "replicas_evaluacion": replicas,
                "ga_semillas_rng": [cfg["ga"]["semilla"] + i for i in range(corridas)],
                "ga_semillas_fitness": [1000 + i for i in range(corridas)],
                "replicas_fitness": cfg["ga"]["replicas_fitness"], "fuerza_bruta_y_mapa": 1000,
                "seleccion_entre_corridas": SEMILLA_SELECCION,
                "validacion_deterministica": 0}
    (RESULTADOS / "semillas.json").write_text(json.dumps(semillas, indent=2))
    (RESULTADOS / "estrategias_ga.json").write_text(json.dumps(
        {k: {"texto": str(v), "paradas": list(v.paradas), "compuestos": list(v.compuestos),
             "umbral_sc": v.umbral_sc, "compuesto_sc": v.compuesto_sc,
             "stints": largos_stints(v, nominal.N)} for k, v in estrategias_ga.items()}, indent=2))
    print(tabla.to_string())


if __name__ == "__main__":
    principal()
