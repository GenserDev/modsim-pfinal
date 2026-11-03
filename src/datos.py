import json
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "data" / "cache"
PROCESADOS = RAIZ / "data" / "processed"
OPENF1 = "https://api.openf1.org/v1"
JOLPICA = "https://api.jolpi.ca/ergast/f1"
ESTADOS_NEUTRALIZADOS = ("4", "5", "6", "7")


def _get_json(url, nombre_cache):
    ruta = CACHE / "json" / f"{nombre_cache}.json"
    if ruta.exists():
        return json.loads(ruta.read_text())
    for intento in range(5):
        respuesta = requests.get(url, timeout=60)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json.dumps(datos))
            time.sleep(0.5)
            return datos
        time.sleep(2 ** (intento + 1))
    raise RuntimeError(f"No se pudo descargar {url}: {respuesta.status_code}")


def ronda_espana(anio):
    datos = _get_json(f"{JOLPICA}/{anio}.json?limit=100", f"jolpica_calendario_{anio}")
    for carrera in datos["MRData"]["RaceTable"]["Races"]:
        if carrera["Circuit"]["circuitId"] == "catalunya":
            return int(carrera["round"])
    raise ValueError(f"No hay GP en Barcelona en {anio}")


def resultados(anio):
    ronda = ronda_espana(anio)
    datos = _get_json(f"{JOLPICA}/{anio}/{ronda}/results.json?limit=100", f"jolpica_resultados_{anio}")
    filas = []
    for r in datos["MRData"]["RaceTable"]["Races"][0]["Results"]:
        filas.append({
            "anio": anio,
            "numero": int(r["number"]),
            "piloto": r["Driver"]["code"],
            "posicion": int(r["position"]),
            "vueltas": int(r["laps"]),
            "estado": r["status"],
            "tiempo_total_s": int(r["Time"]["millis"]) / 1000 if "Time" in r else None,
        })
    return pd.DataFrame(filas)


def clasificacion(anio):
    ronda = ronda_espana(anio)
    datos = _get_json(f"{JOLPICA}/{anio}/{ronda}/qualifying.json?limit=100", f"jolpica_qualy_{anio}")
    filas = []
    for q in datos["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]:
        tiempos = [_a_segundos(q[k]) for k in ("Q1", "Q2", "Q3") if q.get(k)]
        tiempos = [t for t in tiempos if t is not None]
        filas.append({"anio": anio, "piloto": q["Driver"]["code"], "posicion_qualy": int(q["position"]),
                      "mejor_qualy_s": min(tiempos) if tiempos else None})
    return pd.DataFrame(filas)


def paradas_jolpica(anio):
    ronda = ronda_espana(anio)
    datos = _get_json(f"{JOLPICA}/{anio}/{ronda}/pitstops.json?limit=200", f"jolpica_paradas_{anio}")
    filas = []
    for p in datos["MRData"]["RaceTable"]["Races"][0]["PitStops"]:
        filas.append({"anio": anio, "piloto_id": p["driverId"], "vuelta": int(p["lap"]),
                      "parada": int(p["stop"]), "duracion_pitlane_s": _a_segundos(p["duration"])})
    return pd.DataFrame(filas)


def _a_segundos(texto):
    try:
        if ":" in texto:
            minutos, segundos = texto.split(":")
            return int(minutos) * 60 + float(segundos)
        return float(texto)
    except (ValueError, AttributeError):
        return None


def _periodos_desde_estados(estados_por_vuelta):
    periodos = []
    actual = None
    for vuelta, tipo in estados_por_vuelta:
        if tipo and actual and actual["tipo"] == tipo and vuelta == actual["vuelta_fin"] + 1:
            actual["vuelta_fin"] = vuelta
        elif tipo:
            actual = {"tipo": tipo, "vuelta_inicio": vuelta, "vuelta_fin": vuelta}
            periodos.append(actual)
        else:
            actual = None
    return periodos


def _tipo_neutralizacion(estado):
    estado = str(estado)
    if "4" in estado:
        return "SC"
    if "6" in estado or "7" in estado:
        return "VSC"
    if "5" in estado:
        return "BANDERA_ROJA"
    return None


def vueltas_fastf1(anio):
    import fastf1

    warnings.filterwarnings("ignore")
    fastf1.set_log_level("ERROR")
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    sesion = fastf1.get_session(anio, "Spanish Grand Prix", "R")
    sesion.load(telemetry=False, weather=False, messages=False)
    if len(sesion.laps) == 0:
        raise RuntimeError(f"FastF1 no devolvió vueltas para {anio}")
    v = sesion.laps
    tabla = pd.DataFrame({
        "anio": anio,
        "piloto": v["Driver"].values,
        "vuelta": v["LapNumber"].astype(int).values,
        "tiempo_s": v["LapTime"].dt.total_seconds().values,
        "compuesto": v["Compound"].values,
        "edad_llanta": v["TyreLife"].values,
        "stint": v["Stint"].values,
        "entrada_pits": v["PitInTime"].notna().values,
        "salida_pits": v["PitOutTime"].notna().values,
        "estado_pista": v["TrackStatus"].astype(str).values,
        "fuente": "FastF1 (espejo público)",
    })
    tabla.loc[tabla["vuelta"] == 1, "salida_pits"] = False
    lider = tabla.sort_values("vuelta").groupby("vuelta")["estado_pista"].agg(lambda s: "".join(s))
    estados = [(int(vuelta), _tipo_neutralizacion(est)) for vuelta, est in lider.items()]
    return tabla, _periodos_desde_estados(estados)


def _sesion_openf1(anio):
    sesiones = _get_json(f"{OPENF1}/sessions?year={anio}&country_name=Spain&session_name=Race",
                         f"openf1_sesion_{anio}")
    return sesiones[0]["session_key"]


def periodos_openf1(clave, anio):
    mensajes = _get_json(f"{OPENF1}/race_control?session_key={clave}", f"openf1_control_{anio}")
    periodos = []
    abierto = {}
    for m in sorted(mensajes, key=lambda x: x["date"]):
        texto = (m.get("message") or "").upper()
        vuelta = m.get("lap_number")
        if vuelta is None:
            continue
        tipo = "VSC" if "VIRTUAL" in texto else "SC" if "SAFETY CAR" in texto else None
        if m.get("category") == "Flag" and m.get("flag") == "RED":
            periodos.append({"tipo": "BANDERA_ROJA", "vuelta_inicio": vuelta, "vuelta_fin": vuelta})
        if tipo is None or m.get("category") != "SafetyCar":
            continue
        if "DEPLOYED" in texto and tipo not in abierto:
            abierto[tipo] = vuelta
        elif ("IN THIS LAP" in texto or "ENDING" in texto) and tipo in abierto:
            periodos.append({"tipo": tipo, "vuelta_inicio": abierto.pop(tipo), "vuelta_fin": vuelta})
    return periodos


def vueltas_openf1(anio):
    clave = _sesion_openf1(anio)
    vueltas = pd.DataFrame(_get_json(f"{OPENF1}/laps?session_key={clave}", f"openf1_vueltas_{anio}"))
    stints = pd.DataFrame(_get_json(f"{OPENF1}/stints?session_key={clave}", f"openf1_stints_{anio}"))
    pits = pd.DataFrame(_get_json(f"{OPENF1}/pit?session_key={clave}", f"openf1_pits_{anio}"))
    pilotos = pd.DataFrame(_get_json(f"{OPENF1}/drivers?session_key={clave}", f"openf1_pilotos_{anio}"))
    codigos = dict(zip(pilotos["driver_number"], pilotos["name_acronym"]))
    periodos = periodos_openf1(clave, anio)

    filas = []
    for _, vuelta in vueltas.iterrows():
        numero, n = vuelta["driver_number"], int(vuelta["lap_number"])
        stint = stints[(stints["driver_number"] == numero) & (stints["lap_start"] <= n) & (stints["lap_end"] >= n)]
        if len(stint):
            s = stint.iloc[0]
            compuesto, numero_stint = s["compound"], s["stint_number"]
            edad = s["tyre_age_at_start"] + n - s["lap_start"] + 1
        else:
            compuesto, numero_stint, edad = None, None, None
        entrada = bool(((pits["driver_number"] == numero) & (pits["lap_number"] == n)).any()) if len(pits) else False
        filas.append({
            "anio": anio,
            "piloto": codigos.get(numero, str(numero)),
            "vuelta": n,
            "tiempo_s": vuelta["lap_duration"],
            "compuesto": compuesto,
            "edad_llanta": edad,
            "stint": numero_stint,
            "entrada_pits": entrada,
            "salida_pits": bool(vuelta["is_pit_out_lap"]) and n > 1,
            "estado_pista": "",
            "fuente": "OpenF1",
        })
    tabla = pd.DataFrame(filas)
    neutralizadas = {v: p["tipo"] for p in periodos for v in range(p["vuelta_inicio"], p["vuelta_fin"] + 1)}
    tabla["estado_pista"] = tabla["vuelta"].map(neutralizadas).fillna("1")
    return tabla, periodos


def marcar_neutralizadas(tabla, periodos):
    vueltas_neutralizadas = set()
    for p in periodos:
        vueltas_neutralizadas.update(range(p["vuelta_inicio"], p["vuelta_fin"] + 2))
    tabla["neutralizada"] = tabla["vuelta"].isin(vueltas_neutralizadas)
    return tabla


def descargar_todo(anios_carrera=(2021, 2022, 2023, 2024, 2025), anios_sc=(2021, 2022, 2023, 2024)):
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    todas_vueltas, todos_periodos, todos_resultados, todas_qualy, todas_paradas = [], [], [], [], []
    for anio in sorted(set(anios_carrera) | set(anios_sc)):
        tabla, periodos = vueltas_fastf1(anio) if anio <= 2022 else vueltas_openf1(anio)
        tabla = marcar_neutralizadas(tabla, periodos)
        print(f"{anio}: {len(tabla)} vueltas, neutralizaciones {periodos}")
        for p in periodos:
            todos_periodos.append({"anio": anio, **p, "fuente": tabla["fuente"].iloc[0]})
        if not any(p for p in periodos):
            todos_periodos.append({"anio": anio, "tipo": "NINGUNA", "vuelta_inicio": None,
                                   "vuelta_fin": None, "fuente": tabla["fuente"].iloc[0]})
        if anio in anios_carrera:
            todas_vueltas.append(tabla)
            todos_resultados.append(resultados(anio))
            todas_qualy.append(clasificacion(anio))
            todas_paradas.append(paradas_jolpica(anio))

    pd.concat(todas_vueltas).to_csv(PROCESADOS / "vueltas.csv", index=False)
    pd.DataFrame(todos_periodos).to_csv(PROCESADOS / "neutralizaciones.csv", index=False)
    pd.concat(todos_resultados).to_csv(PROCESADOS / "resultados.csv", index=False)
    pd.concat(todas_qualy).to_csv(PROCESADOS / "clasificacion.csv", index=False)
    pd.concat(todas_paradas).to_csv(PROCESADOS / "paradas_jolpica.csv", index=False)


if __name__ == "__main__":
    descargar_todo()
