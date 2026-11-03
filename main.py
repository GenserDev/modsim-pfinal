import argparse

from src import calibracion, datos, experimentos, visualizacion


def main():
    parser = argparse.ArgumentParser(description="Estrategia de pits en el GP de España con Monte Carlo y GA")
    parser.add_argument("--sin-descarga", action="store_true", help="usa los CSV ya guardados en data/processed")
    args = parser.parse_args()
    if not args.sin_descarga:
        datos.descargar_todo()
    calibracion.calibrar()
    experimentos.principal()
    visualizacion.generar_todas()
    print("Listo. Resultados en results/ y gráficas en results/figuras/")


if __name__ == "__main__":
    main()
