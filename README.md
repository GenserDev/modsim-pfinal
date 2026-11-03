# Estrategia de paradas en pits en el GP de España con Monte Carlo y algoritmo genético

Proyecto 2 de CC3074 Modelización y Simulación, Universidad del Valle de Guatemala, ciclo 2 de 2026.

Buscamos la estrategia de pits (número de paradas, vueltas y compuestos) que minimiza el tiempo total esperado de carrera en Barcelona. El modelo por vuelta se calibró con datos públicos de 2022 a 2024 y se validó fuera de muestra contra la carrera real de 2025. Un simulador Monte Carlo con Safety Car aleatorio evalúa cada estrategia y un algoritmo genético con restricciones, implementado por el equipo, la optimiza y la compara contra la estrategia real del ganador.

## Cómo correr el proyecto

**1. Clonar el repositorio y entrar a la carpeta**

```bash
git clone https://github.com/GenserDev/modsim-pfinal.git && cd modsim-pfinal
```

**2. Crear un entorno virtual con Python 3.12 y activarlo**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**3. Instalar las dependencias**

```bash
pip install -r requirements.txt
```

**4. Correr las pruebas**

```bash
pytest -q
```

**5. Ejecutar todo el flujo**

```bash
python main.py
```

Descarga los datos, calibra el modelo, corre los experimentos y genera las gráficas. Las tablas quedan en `results/` y las figuras en `results/figuras/`.

## Notas

**Datos sin costo.** Las vueltas de 2023 a 2025 vienen de la API pública OpenF1. Las de 2021 y 2022 vienen del espejo público de FastF1 y los resultados, la clasificación y las paradas, de Jolpica-F1. Ninguna fuente pide cuenta. Los CSV procesados ya están en `data/processed`, así que `python main.py --sin-descarga` corre todo sin internet.

**Tiempo de ejecución.** La descarga tarda unos minutos la primera vez y después queda en `data/cache`. Los experimentos completos (30 corridas del GA por condición) tardan unos 5 minutos en una laptop.

**Carpeta de ejecución.** Los comandos deben correrse desde la raíz del repositorio.

**Módulos.** `src/datos.py` descarga, `src/calibracion.py` estima parámetros, `src/modelo.py` define ecuaciones y restricciones, `src/simulacion.py` es el motor Monte Carlo, `src/ga.py` es el algoritmo genético, `src/experimentos.py` corre las condiciones y `src/visualizacion.py` hace las gráficas. Los parámetros configurables están en `config.yaml`.
