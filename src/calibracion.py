import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

RAIZ = Path(__file__).resolve().parent.parent
PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"


def cargar_config():
    return yaml.safe_load((RAIZ / "config.yaml").read_text())


def cargar_datos():
    return {
        "vueltas": pd.read_csv(PROCESADOS / "vueltas.csv"),
        "neutralizaciones": pd.read_csv(PROCESADOS / "neutralizaciones.csv"),
        "resultados": pd.read_csv(PROCESADOS / "resultados.csv"),
        "clasificacion": pd.read_csv(PROCESADOS / "clasificacion.csv"),
    }


def vueltas_limpias(vueltas, anios, umbral, compuestos):
    v = vueltas[vueltas["anio"].isin(anios)].copy()
    v = v[v["tiempo_s"].notna() & v["compuesto"].isin(compuestos) & v["edad_llanta"].notna()]
    v = v[(v["vuelta"] > 1) & ~v["entrada_pits"] & ~v["salida_pits"] & ~v["neutralizada"]]
    mejor = v.groupby("anio")["tiempo_s"].transform("min")
    return v[v["tiempo_s"] < umbral * mejor]


def matriz_diseno(v, compuestos, referencia, n_vueltas, efectos):
    columnas = {}
    for efecto in efectos:
        columnas[f"alfa_{efecto}"] = ((v["piloto"] + "_" + v["anio"].astype(str)) == efecto).astype(float)
    for c in compuestos:
        es_c = (v["compuesto"] == c).astype(float)
        if c != referencia:
            columnas[f"delta_{c}"] = es_c
        columnas[f"k_{c}"] = es_c * v["edad_llanta"]
        columnas[f"k2_{c}"] = es_c * v["edad_llanta"] ** 2
    columnas["f"] = n_vueltas - v["vuelta"]
    return pd.DataFrame(columnas, index=v.index)


def ajustar_modelo(limpias, compuestos, referencia, n_vueltas):
    efectos = sorted((limpias["piloto"] + "_" + limpias["anio"].astype(str)).unique())
    X = matriz_diseno(limpias, compuestos, referencia, n_vueltas, efectos)
    y = limpias["tiempo_s"].values
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    residuos = y - X.values @ beta
    gl = len(y) - X.shape[1]
    sigma2 = residuos @ residuos / gl
    cov = sigma2 * np.linalg.pinv(X.values.T @ X.values)
    errores = np.sqrt(np.diag(cov))
    coef = pd.DataFrame({"valor": beta, "error_estandar": errores}, index=X.columns)
    r2 = 1 - residuos.var() / y.var()
    return coef, residuos, r2


def perdida_pits(vueltas, coef, anios, compuestos, referencia, n_vueltas):
    v = vueltas[vueltas["anio"].isin(anios) & vueltas["compuesto"].isin(compuestos)].copy()
    v["efecto"] = v["piloto"] + "_" + v["anio"].astype(str)
    v = v[v["efecto"].isin([i[5:] for i in coef.index if i.startswith("alfa_")])]
    v["prediccion"] = predecir(v, coef, referencia, n_vueltas)
    perdidas = []
    for (anio, piloto), g in v.groupby(["anio", "piloto"]):
        g = g.set_index("vuelta")
        for vuelta in g.index[g["entrada_pits"]]:
            if vuelta + 1 not in g.index or not g.loc[vuelta + 1, "salida_pits"]:
                continue
            if g.loc[[vuelta, vuelta + 1], "neutralizada"].any():
                continue
            extra = (g.loc[vuelta, "tiempo_s"] - g.loc[vuelta, "prediccion"]
                     + g.loc[vuelta + 1, "tiempo_s"] - g.loc[vuelta + 1, "prediccion"])
            perdidas.append({"anio": anio, "piloto": piloto, "vuelta": vuelta, "perdida_s": extra})
    tabla = pd.DataFrame(perdidas)
    q1, q3 = tabla["perdida_s"].quantile([0.25, 0.75])
    # Se descartan paradas con problemas (por ejemplo, cambio de alerón) porque no representan una parada normal
    return tabla[tabla["perdida_s"] <= q3 + 3 * (q3 - q1)]


def predecir(v, coef, referencia, n_vueltas):
    c = coef["valor"]
    efecto = "alfa_" + v["piloto"] + "_" + v["anio"].astype(str)
    base = efecto.map(c).fillna(np.nan)
    delta = v["compuesto"].map(lambda x: 0.0 if x == referencia else c.get(f"delta_{x}", np.nan))
    k = v["compuesto"].map(lambda x: c.get(f"k_{x}", np.nan))
    k2 = v["compuesto"].map(lambda x: c.get(f"k2_{x}", np.nan))
    a = v["edad_llanta"]
    return base + delta + k * a + k2 * a ** 2 + c["f"] * (n_vueltas - v["vuelta"])


def efectos_safety_car(vueltas, neutralizaciones, anios):
    neutral = neutralizaciones[neutralizaciones["anio"].isin(anios) & (neutralizaciones["tipo"] == "SC")]
    filas_ratio, filas_parada = [], []
    for _, p in neutral.iterrows():
        v = vueltas[vueltas["anio"] == p["anio"]]
        verdes = v[~v["neutralizada"] & ~v["entrada_pits"] & ~v["salida_pits"] & (v["vuelta"] > 1)]
        referencia = verdes.groupby("piloto")["tiempo_s"].median()
        bajo_sc = v[v["vuelta"].between(p["vuelta_inicio"], p["vuelta_fin"])]
        normales = bajo_sc[~bajo_sc["entrada_pits"] & ~bajo_sc["salida_pits"]]
        ratios = normales["tiempo_s"] / normales["piloto"].map(referencia)
        filas_ratio.extend(ratios.dropna().tolist())
        mediana_sc = normales.groupby("vuelta")["tiempo_s"].median()
        for _, entrada in bajo_sc[bajo_sc["entrada_pits"]].iterrows():
            salida = v[(v["piloto"] == entrada["piloto"]) & (v["vuelta"] == entrada["vuelta"] + 1)]
            if salida.empty or entrada["vuelta"] + 1 not in mediana_sc.index:
                continue
            extra = (entrada["tiempo_s"] - mediana_sc[entrada["vuelta"]]
                     + salida["tiempo_s"].iloc[0] - mediana_sc[entrada["vuelta"] + 1])
            filas_parada.append({"anio": p["anio"], "piloto": entrada["piloto"], "vuelta": entrada["vuelta"],
                                 "perdida_s": extra})
    return np.array(filas_ratio), pd.DataFrame(filas_parada)


def probabilidad_safety_car(neutralizaciones, anios, n_vueltas):
    neutral = neutralizaciones[neutralizaciones["anio"].isin(anios) & (neutralizaciones["tipo"] == "SC")]
    despliegues = len(neutral)
    vueltas_riesgo = len(anios) * n_vueltas
    p = despliegues / vueltas_riesgo
    bajo = stats.beta.ppf(0.025, despliegues, vueltas_riesgo - despliegues + 1) if despliegues else 0.0
    alto = stats.beta.ppf(0.975, despliegues + 1, vueltas_riesgo - despliegues)
    duraciones = (neutral["vuelta_fin"] - neutral["vuelta_inicio"] + 1).astype(int).tolist()
    return p, (bajo, alto), despliegues, vueltas_riesgo, duraciones


def ritmo_base_validacion(coef, clasificacion, resultados, anios, anio_val, piloto_val):
    filas = []
    for anio in anios:
        ganador = resultados[(resultados["anio"] == anio) & (resultados["posicion"] == 1)]["piloto"].iloc[0]
        qualy = clasificacion[(clasificacion["anio"] == anio) & (clasificacion["piloto"] == ganador)]["mejor_qualy_s"].iloc[0]
        alfa = coef.loc[f"alfa_{ganador}_{anio}", "valor"]
        filas.append({"anio": anio, "ganador": ganador, "alfa_s": alfa, "qualy_s": qualy, "diferencia_s": alfa - qualy})
    tabla = pd.DataFrame(filas)
    qualy_val = clasificacion[(clasificacion["anio"] == anio_val) & (clasificacion["piloto"] == piloto_val)]["mejor_qualy_s"].iloc[0]
    return qualy_val + tabla["diferencia_s"].mean(), qualy_val, tabla


def extra_primera_vuelta(vueltas, coef, resultados, anios, referencia, n_vueltas):
    filas = []
    for anio in anios:
        ganador = resultados[(resultados["anio"] == anio) & (resultados["posicion"] == 1)]["piloto"].iloc[0]
        v = vueltas[(vueltas["anio"] == anio) & (vueltas["piloto"] == ganador) & (vueltas["vuelta"] == 1)]
        filas.append(float(v["tiempo_s"].iloc[0] - predecir(v, coef, referencia, n_vueltas).iloc[0]))
    return float(np.mean(filas)), filas


def calibrar():
    cfg = cargar_config()
    d = cargar_datos()
    n = cfg["vueltas_carrera"]
    cal = cfg["calibracion"]
    compuestos, ref = cal["compuestos"], cal["compuesto_referencia"]
    anios = cfg["anios_calibracion"]

    limpias = vueltas_limpias(d["vueltas"], anios, cal["umbral_vuelta_rapida"], compuestos)
    coef, residuos, r2 = ajustar_modelo(limpias, compuestos, ref, n)
    perdidas = perdida_pits(d["vueltas"], coef, anios, compuestos, ref, n)
    ratios_sc, perdidas_sc = efectos_safety_car(d["vueltas"], d["neutralizaciones"], cfg["anios_safety_car"])
    p_sc, ic_sc, despliegues, vueltas_riesgo, duraciones = probabilidad_safety_car(
        d["neutralizaciones"], cfg["anios_safety_car"], n)
    t_base, qualy_val, tabla_ritmo = ritmo_base_validacion(
        coef, d["clasificacion"], d["resultados"], anios, cfg["anio_validacion"], cfg["piloto_validacion"])
    extra_v1, extras_v1 = extra_primera_vuelta(d["vueltas"], coef, d["resultados"], anios, ref, n)

    vida_max = (d["vueltas"][d["vueltas"]["anio"].isin(anios) & d["vueltas"]["compuesto"].isin(compuestos)]
                .groupby("compuesto")["edad_llanta"].max().astype(int).to_dict())
    c = coef["valor"]
    mediana_perdida = float(perdidas["perdida_s"].median())

    parametros = {
        "N": {"valor": n, "unidad": "vueltas", "fuente": "Jolpica-F1, resultados 2022 a 2025"},
        "t_base": {"valor": float(t_base), "unidad": "s",
                   "fuente": f"Mejor vuelta de qualy {cfg['anio_validacion']} de {cfg['piloto_validacion']} ({qualy_val} s) más la diferencia media ritmo de carrera menos qualy de los ganadores {anios}"},
        "delta": {comp: {"valor": float(c[f"delta_{comp}"]) if comp != ref else 0.0, "unidad": "s",
                         "error_estandar": float(coef.loc[f"delta_{comp}", "error_estandar"]) if comp != ref else 0.0}
                  for comp in compuestos},
        "k": {comp: {"valor": float(c[f"k_{comp}"]), "unidad": "s/vuelta",
                     "error_estandar": float(coef.loc[f"k_{comp}", "error_estandar"])} for comp in compuestos},
        "k2": {comp: {"valor": float(c[f"k2_{comp}"]), "unidad": "s/vuelta²",
                      "error_estandar": float(coef.loc[f"k2_{comp}", "error_estandar"])} for comp in compuestos},
        "f": {"valor": float(c["f"]), "unidad": "s/vuelta restante",
              "error_estandar": float(coef.loc["f", "error_estandar"])},
        "sigma": {"valor": float(residuos.std(ddof=coef.shape[0])), "unidad": "s",
                  "fuente": "Desviación estándar de los residuos de la regresión"},
        "extra_vuelta_1": {"valor": extra_v1, "unidad": "s",
                           "fuente": "Vuelta 1 real menos predicción del modelo, ganadores de calibración"},
        "perdidas_pits": {"valores": perdidas["perdida_s"].round(3).tolist(), "unidad": "s",
                          "mediana": mediana_perdida, "n": int(len(perdidas)),
                          "fuente": "Vuelta de entrada más vuelta de salida menos su predicción en verde"},
        "p_sc": {"valor": p_sc, "ic95": list(ic_sc), "unidad": "probabilidad de despliegue por vuelta",
                 "despliegues": despliegues, "vueltas_en_riesgo": vueltas_riesgo,
                 "fuente": f"Neutralizaciones SC en Barcelona {cfg['anios_safety_car']}"},
        "duraciones_sc": {"valores": duraciones, "unidad": "vueltas"},
        "factor_vuelta_sc": {"valor": float(np.mean(ratios_sc)), "n": int(len(ratios_sc)), "unidad": "adimensional",
                             "fuente": "Media de la vuelta bajo SC dividida entre la mediana en verde del mismo piloto"},
        "factor_parada_sc": {"valor": cal["factor_parada_sc_supuesto"], "unidad": "adimensional",
                             "fuente": "Supuesto del equipo. Las paradas bajo SC observadas dan pérdidas por vuelta no interpretables por el reagrupamiento",
                             "observaciones_descartadas_s": perdidas_sc["perdida_s"].round(3).tolist()},
        "vida_max": {"valores": vida_max, "unidad": "vueltas", "fuente": f"Máxima edad observada {anios}"},
        "regresion": {"r2": float(r2), "vueltas_limpias": int(len(limpias)), "parametros": int(coef.shape[0])},
    }
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "parametros.json").write_text(json.dumps(parametros, indent=2, ensure_ascii=False))
    coef.to_csv(RESULTADOS / "coeficientes_regresion.csv")
    tabla_ritmo.to_csv(RESULTADOS / "ritmo_base.csv", index=False)
    perdidas.to_csv(RESULTADOS / "perdidas_pits.csv", index=False)
    perdidas_sc.to_csv(RESULTADOS / "perdidas_pits_sc.csv", index=False)
    return parametros


if __name__ == "__main__":
    p = calibrar()
    resumen = {k: v for k, v in p.items() if k not in ("perdidas_pits",)}
    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    print("perdida pits mediana", p["perdidas_pits"]["mediana"], "n", p["perdidas_pits"]["n"])
