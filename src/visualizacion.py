import json
from pathlib import Path

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "results"
FIGURAS = RESULTADOS / "figuras"

SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
REJILLA = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLOR_COMPUESTO = {"SOFT": SERIES[0], "MEDIUM": SERIES[1], "HARD": SERIES[2]}
NOMBRE_COMPUESTO = {"SOFT": "Blando (SOFT)", "MEDIUM": "Medio (MEDIUM)", "HARD": "Duro (HARD)"}

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_SECUNDARIO, "xtick.color": TEXTO_SECUNDARIO,
    "ytick.color": TEXTO_SECUNDARIO, "text.color": TEXTO, "axes.titlecolor": TEXTO, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 9.5,
})


def guardar(figura, nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURAS / nombre, dpi=200, bbox_inches="tight")
    plt.close(figura)


def grafica_validacion():
    v = pd.read_csv(RESULTADOS / "validacion_vueltas.csv")
    datos = json.loads((RESULTADOS / "validacion.json").read_text())
    figura, eje = plt.subplots(figsize=(10, 4.8))
    sc = v[v["neutralizada"]]
    eje.axvspan(sc["vuelta"].min() - 0.5, sc["vuelta"].max() + 0.5, color=REJILLA, alpha=0.7, lw=0)
    eje.text(sc["vuelta"].mean(), 81.8, "Safety Car\n(observado)", ha="center", va="top", color=TEXTO_SECUNDARIO, fontsize=8.5)
    for compuesto, g in v.groupby("compuesto"):
        eje.scatter(g["vuelta"], g["tiempo_s"], s=26, color=COLOR_COMPUESTO[compuesto], edgecolor=SUPERFICIE,
                    linewidth=1, zorder=3, label=f"Real, {NOMBRE_COMPUESTO[compuesto]}")
    eje.plot(v["vuelta"], v["simulado_s"], color=TEXTO, lw=1.6, zorder=2, label="Simulado (modelo base)")
    eje.set_ylim(75, 82)
    eje.set_xlim(0, 67)
    eje.set_xlabel("Vuelta")
    eje.set_ylabel("Tiempo de vuelta (s)")
    eje.set_title(f"Tiempo por vuelta real vs simulado, Piastri, GP de España 2025\n"
                  f"MAE en {datos['vueltas_limpias']} vueltas limpias = {datos['mae_vuelta_limpia_s']:.2f} s. "
                  f"Vueltas de pits y bajo SC quedan fuera de escala", loc="left")
    eje.legend(loc="lower left", ncol=2)
    guardar(figura, "1_vueltas_real_vs_simulado.png")


def grafica_boxplot():
    totales = pd.read_csv(RESULTADOS / "totales_por_replica.csv")
    nombres = {"base_real_2025": "Base real\n(3 paradas)", "base_planificada_2025": "Base sin\nparada SC",
               "ga_fija": "GA\nfija", "ga_reactiva": "GA\nreactiva"}
    grupos = {"nominal": "Nominal", "p_sc_x2": "p_SC × 2", "degradacion_+30%": "Degradación +30 %"}
    figura, ejes = plt.subplots(1, 3, figsize=(12.5, 4.8), sharey=True)
    for eje, (clave, titulo) in zip(ejes, grupos.items()):
        columnas = [f"{clave}|{n}" for n in nombres]
        datos = [totales[c].values for c in columnas]
        caja = eje.boxplot(datos, widths=0.55, patch_artist=True, showfliers=True,
                           medianprops={"color": TEXTO, "lw": 1.6},
                           flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": TEXTO_SECUNDARIO,
                                       "markeredgecolor": "none", "alpha": 0.5},
                           whiskerprops={"color": TEXTO_SECUNDARIO}, capprops={"color": TEXTO_SECUNDARIO})
        for parche, color in zip(caja["boxes"], SERIES):
            parche.set_facecolor(color)
            parche.set_alpha(0.85)
            parche.set_edgecolor(SUPERFICIE)
        for i, d in enumerate(datos, start=1):
            eje.scatter(i, d.mean(), marker="D", s=22, color=SUPERFICIE, edgecolor=TEXTO, zorder=4)
            eje.text(i, np.percentile(d, 99.5) + 8, f"{d.mean():.0f}", ha="center", fontsize=8, color=TEXTO)
        eje.set_xticks(range(1, len(nombres) + 1), list(nombres.values()))
        eje.set_title(titulo, loc="left")
    ejes[0].set_ylabel("Tiempo total de carrera (s)")
    figura.suptitle("Tiempo total simulado por condición (1000 réplicas, rombo = media)", x=0.01, ha="left",
                    fontweight="bold", fontsize=12)
    figura.tight_layout()
    guardar(figura, "2_boxplot_condiciones.png")


def grafica_convergencia():
    h = pd.read_csv(RESULTADOS / "ga_convergencia.csv")
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for eje, condicion, titulo in zip(ejes, ("nominal_fija", "nominal_reactiva"), ("GA fija", "GA reactiva al SC")):
        g = h[h["condicion"] == condicion]
        activas = g.groupby("generacion")["corrida"].nunique()
        g = g[g["generacion"] <= activas[activas >= 15].index.max()]
        for corrida, gc in g.groupby("corrida"):
            eje.plot(gc["generacion"], gc["mejor_s"], color=SERIES[0], alpha=0.25, lw=1)
        mediana = g.groupby("generacion")["mejor_s"].median()
        media_pob = g.groupby("generacion")["media_factibles_s"].median()
        eje.plot(mediana.index, mediana.values, color=TEXTO, lw=2, label="Mejor individuo (mediana de 30 corridas)")
        eje.plot(media_pob.index, media_pob.values, color=SERIES[1], lw=2, label="Media de factibles (mediana)")
        eje.plot([], [], color=SERIES[0], alpha=0.5, lw=1, label="Mejor individuo, cada corrida")
        eje.set_title(titulo, loc="left")
        eje.set_xlabel("Generación (hasta que quedan menos de 15 corridas activas)")
    ejes[0].set_ylabel("E[T] estimado (s)")
    minimo = h[h["condicion"].str.startswith("nominal")]["mejor_s"].min()
    ejes[0].set_ylim(minimo - 5, minimo + 60)
    ejes[1].legend(loc="upper right")
    figura.suptitle("Convergencia del algoritmo genético, condición nominal (30 corridas, 200 réplicas por evaluación)",
                    x=0.01, ha="left", fontweight="bold", fontsize=12)
    figura.tight_layout()
    guardar(figura, "3_convergencia_ga.png")


def grafica_mapa_calor():
    m = pd.read_csv(RESULTADOS / "mapa_calor.csv")
    compuestos = m["compuestos"].iloc[0]
    tabla = m.pivot(index="parada_2", columns="parada_1", values="tiempo_esperado_s")
    minimo = np.nanmin(tabla.values)
    mapa = LinearSegmentedColormap.from_list("azul", ["#0d3a78", "#2a78d6", "#a9cdf4", "#f2f7fd"])
    mapa.set_bad(SUPERFICIE)
    figura, eje = plt.subplots(figsize=(7.5, 6))
    imagen = eje.pcolormesh(tabla.columns, tabla.index, tabla.values - minimo, cmap=mapa, vmin=0, vmax=40,
                            shading="nearest")
    fila = m.loc[m["tiempo_esperado_s"].idxmin()]
    eje.scatter(fila["parada_1"], fila["parada_2"], marker="x", s=60, color=SERIES[1], lw=2,
                label=f"Óptimo: vueltas {int(fila['parada_1'])} y {int(fila['parada_2'])}")
    barra = figura.colorbar(imagen, ax=eje, extend="max")
    barra.set_label("E[T] − mínimo (s)")
    eje.set_xlabel("Vuelta de la parada 1")
    eje.set_ylabel("Vuelta de la parada 2")
    eje.grid(False)
    eje.set_title(f"Tiempo esperado por vueltas de parada, 2 paradas {compuestos}\n"
                  f"(blanco = infactible por vida máxima de la llanta)", loc="left")
    eje.legend(loc="lower right")
    guardar(figura, "4_mapa_calor_paradas.png")


def grafica_sensibilidad():
    e = pd.read_csv(RESULTADOS / "experimentos.csv")
    e = e[e["condicion"].isin(["ga_fija", "ga_reactiva"])].copy()
    orden = ["nominal", "p_sc_x2", "degradacion_+30%", "factor_parada_sc_0.3", "factor_parada_sc_0.7"]
    etiquetas = ["Nominal", "p_SC × 2", "Degradación +30 %", "Pérdida en pits bajo SC × 0.3",
                 "Pérdida en pits bajo SC × 0.7"]
    figura, eje = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(orden))
    for i, (condicion, nombre) in enumerate((("ga_fija", "GA fija"), ("ga_reactiva", "GA reactiva al SC"))):
        g = e[e["condicion"] == condicion].set_index("parametros").reindex(orden)
        posiciones = y + (i - 0.5) * 0.32
        error = [g["diferencia_media_s"] - g["diferencia_ic95_inf_s"], g["diferencia_ic95_sup_s"] - g["diferencia_media_s"]]
        eje.barh(posiciones, g["diferencia_media_s"], height=0.28, color=SERIES[i + 2], label=nombre)
        eje.errorbar(g["diferencia_media_s"], posiciones, xerr=error, fmt="none", ecolor=TEXTO, lw=1, capsize=2)
        for p, valor in zip(posiciones, g["diferencia_media_s"]):
            eje.text(valor - 0.6, p, f"{valor:.1f}", va="center", ha="right", fontsize=8, color=TEXTO)
    eje.axvline(0, color=TEXTO_SECUNDARIO, lw=1)
    eje.set_yticks(y, etiquetas)
    eje.invert_yaxis()
    eje.set_xlabel("Diferencia de E[T] frente a la base real 2025 (s, negativo = más rápido)")
    eje.set_title("Sensibilidad de la mejora (media e IC 95 %, 1000 réplicas pareadas)", loc="left")
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    guardar(figura, "5_sensibilidad.png")


def diagrama_sistema():
    figura, eje = plt.subplots(figsize=(11, 5.6))
    eje.set_xlim(0, 11)
    eje.set_ylim(0, 6)
    eje.axis("off")
    cajas = {
        "datos": (0.2, 3.9, "Datos reales\nOpenF1, espejo FastF1,\nJolpica-F1 (2021 a 2025)"),
        "calibracion": (0.2, 1.4, "Calibración\nregresión en vueltas limpias,\npérdida en pits, p_SC"),
        "modelo": (3.9, 3.9, "Modelo por vuelta\nt = t_base + δc + kc·a + k2c·a²\n+ f·(N−l) + ε"),
        "eventos": (3.9, 1.4, "Eventos aleatorios\nSC ~ Bernoulli(p_SC) por vuelta,\npérdida en pits empírica"),
        "simulacion": (7.5, 3.9, "Simulación Monte Carlo\nR réplicas con números\naleatorios comunes"),
        "ga": (7.5, 1.4, "Algoritmo genético\nreglas de Deb, torneo,\ncruce uniforme, elitismo"),
    }
    for clave, (x, y, texto) in cajas.items():
        color = SERIES[1] if clave == "ga" else SERIES[0] if clave in ("modelo", "simulacion", "eventos") else TEXTO_SECUNDARIO
        eje.add_patch(matplotlib.patches.FancyBboxPatch((x, y), 3.1, 1.5, boxstyle="round,pad=0.04,rounding_size=0.12",
                                                        facecolor=SUPERFICIE, edgecolor=color, lw=2))
        eje.text(x + 1.55, y + 0.75, texto, ha="center", va="center", fontsize=9, color=TEXTO)
    flechas = [((1.75, 3.9), (1.75, 2.9), "parámetros", "left", 1.85),
               ((3.3, 2.15), (3.9, 4.4), "", "left", 0), ((3.3, 2.15), (3.9, 2.15), "", "left", 0),
               ((7.0, 4.65), (7.5, 4.65), "", "left", 0), ((7.0, 2.15), (7.5, 4.2), "", "left", 0),
               ((8.85, 2.9), (8.85, 3.9), "estrategia", "right", 8.75),
               ((9.45, 3.9), (9.45, 2.9), "E[T] estimado", "left", 9.55)]
    for inicio, fin, etiqueta, alineacion, x_texto in flechas:
        eje.annotate("", xy=fin, xytext=inicio, arrowprops={"arrowstyle": "-|>", "color": TEXTO_SECUNDARIO, "lw": 1.3})
        if etiqueta:
            eje.text(x_texto, 3.4, etiqueta, fontsize=8.5, ha=alineacion, va="center", color=TEXTO_SECUNDARIO)
    eje.text(0.2, 0.55, "Entradas: estrategia (paradas, vueltas, compuestos, umbral SC).   Salidas: tiempo total T (s), "
             "E[T], desviación estándar, P(T < base).\nRestricciones: 1 a 3 paradas, ≥ 2 compuestos de seco, vueltas "
             "crecientes en [1, N−1], stint ≤ vida máxima observada.", fontsize=8.5, color=TEXTO_SECUNDARIO)
    eje.set_title("Diagrama del sistema: datos, modelo, simulación y optimización", loc="left")
    guardar(figura, "0_diagrama_sistema.png")


def generar_todas():
    diagrama_sistema()
    grafica_validacion()
    grafica_boxplot()
    grafica_convergencia()
    grafica_mapa_calor()
    grafica_sensibilidad()


if __name__ == "__main__":
    generar_todas()
