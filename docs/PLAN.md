# Plan del Proyecto 2

CC3074 Modelización y Simulación, UVG, ciclo 2 de 2026.

## Estrategia

**Tema.** Optimización de la estrategia de paradas en pits de F1 con simulación Monte Carlo (técnica de clase) y un algoritmo genético con restricciones (técnica adicional).

**Pregunta de investigación.** ¿Qué estrategia (número de paradas, vueltas y compuestos) minimiza el tiempo total esperado de carrera y cuánto mejora frente a la estrategia real?

**Métricas.** Principal: tiempo total de carrera en segundos, a minimizar. Secundarias: desviación estándar del tiempo total y probabilidad de superar a la base.

**Escenario.** GP de España (Barcelona), reglamento 2022 a 2025. No se usa 2026 porque cambiaron los autos. Se calibra con 2022, 2023 y 2024 y se valida fuera de muestra con 2025. La base es la estrategia real del ganador de 2025, verificada con los datos. Quedan fuera tráfico, adelantamientos, lluvia e incidentes propios.

**Modelo por vuelta.** `t = t_base + δ_c + k_c·a + k2_c·a² + f·(N−l) + ε`, con ε~N(0,σ²). Pérdida en pits con distribución empírica del circuito. Safety Car como Bernoulli por vuelta con p_SC histórica 2018 a 2024. Vuelta bajo SC más lenta y parada más barata, estimadas de los datos si es posible. Calibración por regresión sobre vueltas limpias.

**Datos.** FastF1 con caché en `data/cache` (ignorada por git). Jolpica-F1 para duración de paradas. Pirelli para compuestos. Reglamento Deportivo FIA para la regla de 2 compuestos. CSV procesados en `data/processed` con script de descarga.

**Algoritmo genético.** Implementación propia. Cromosoma `[n_paradas, vueltas…, compuestos…]`. Objetivo E[T] con R réplicas y números aleatorios comunes (variante E[T] + λ·σ). Restricciones con las reglas de Deb (2000). Torneo, cruce uniforme, mutaciones (mover parada, cambiar compuesto, agregar o quitar parada) y elitismo. Termina en G generaciones o 20 sin mejora. Verificación contra fuerza bruta en 1 y 2 paradas. Extensión reactiva al SC con umbral en el cromosoma.

**Experimentos.** Base, GA, sensibilidad p_SC ×2, sensibilidad degradación +30 %, y Monza si alcanza el tiempo. Al menos 30 repeticiones por condición (1000 si es barato), semillas en `results/`. Media, desviación estándar e IC 95 %.

**Validación.** Reproducir la estrategia real 2025: MAE por vuelta y error del tiempo total. Tabla real, base y mejorado con cambio absoluto y porcentual. Separar mejora simulada de mejora real. Pruebas con pytest.

## Checklist

### Parte 1. Problema
- [ ] Contexto, actores, entradas y salidas
- [ ] Pregunta de investigación concreta
- [ ] Métrica principal con unidades y sentido (min)
- [ ] Alcance, restricciones, supuestos y exclusiones
- [ ] Al menos una fuente verificable del escenario

### Parte 2. Investigación técnica (GA con restricciones)
- [ ] Objetivo, representación, parámetros, procedimiento y terminación
- [ ] Al menos 3 fuentes técnicas (1 académica o documentación)
- [ ] Justificación frente a alternativas
- [ ] Al menos 2 limitaciones
- [ ] Qué hace la biblioteca y qué hace el equipo

### Parte 3. Modelo y simulación
- [ ] Diagrama del sistema
- [ ] Variables de estado, parámetros, entradas, salidas, restricciones y distribuciones
- [ ] Reglas, fórmulas o pseudocódigo
- [ ] Procedencia, valor y unidad de cada parámetro
- [ ] Simulación funcional y parametrizable (`config.yaml`), con modelo, simulación, técnica y visualización separados
- [ ] Reproducción desde entorno limpio documentada

### Parte 4. Mejora
- [ ] Función objetivo y restricciones definidas formalmente
- [ ] Integración GA y simulación explicada
- [ ] Hiperparámetros con valor y justificación
- [ ] Solución candidata verificada por código (`verificar_restricciones()`)
- [ ] Alternativa base sin mejora conservada

### Parte 5. Experimentación y validación
- [ ] Al menos 3 condiciones: base, mejorada, sensibilidad
- [ ] Entradas, parámetros, resultados y métricas registrados por condición
- [ ] ≥30 repeticiones, semillas registradas, media y dispersión
- [ ] Propiedad interna probada (pytest: conservación, degradación 0, determinismo, restricciones)
- [ ] Comparación con datos de referencia (MAE por vuelta y error del total)
- [ ] Prueba de sensibilidad explicada
- [ ] Incertidumbre, limitaciones y riesgo de extrapolación

### Parte 6. Comparación con la vida real
- [ ] Fuente, fecha, período, tamaño de muestra y condiciones
- [ ] Tabla real, base y mejorado con mismas métricas y unidades
- [ ] Correspondencia base vs real explicada antes de la mejora
- [ ] Cambio absoluto y porcentual respecto de la base
- [ ] Mejora simulada separada de mejora real

### Parte 7. Resultados y conclusiones
- [ ] Al menos 3 visualizaciones (comportamiento, comparación, variabilidad o sensibilidad)
- [ ] Cada gráfica con título, ejes, unidades, leyenda y explicación
- [ ] Tablas numéricas
- [ ] Qué mejoró, cuánto, bajo qué condiciones, confiabilidad y limitaciones
- [ ] Conclusión y recomendación concreta

### Gráficas planeadas
- [ ] Tiempo por vuelta real vs simulado por compuesto
- [ ] Boxplot del tiempo total por condición
- [ ] Convergencia del GA
- [ ] Mapa de calor de E[T] por vuelta de parada 1 y parada 2

### Entregables
- [ ] Informe técnico PDF (portada, problema, objetivos, fuentes, técnica, modelo, implementación, diseño experimental, validación, comparación real, resultados, limitaciones, conclusiones, referencias, contribuciones, declaración de uso de IA)
- [ ] Repositorio con código, README reproducible, `requirements.txt` con versiones fijas, datos o script de descarga
- [ ] Evidencia experimental en `results/` (tablas, parámetros, semillas, gráficas, métricas)
- [ ] Diapositivas y demostración (máximo 15 minutos)
- [ ] Registro de contribuciones

### Presentación (15 min)
- [ ] 2 min problema, objetivo y fuente
- [ ] 2 min modelo, variables, restricciones y supuestos
- [ ] 3 min técnica e integración
- [ ] 3 min demostración
- [ ] 3 min resultados, validación y comparación real
- [ ] 2 min conclusiones y preguntas
- [ ] Ejecución preparada de respaldo
- [ ] Participación de todos los integrantes

### Lista de verificación final del PDF
- [ ] Problema real con métrica principal y fuentes verificables
- [ ] Técnica adicional investigada e implementada, con justificación y limitaciones
- [ ] Modelo con variables, reglas, supuestos, restricciones y unidades
- [ ] Simulación y mejora se ejecutan desde el README
- [ ] Base, mejorado y sensibilidad documentados
- [ ] Resultados, incertidumbre y errores verificados y reproducibles
- [ ] Referencia real documentada y datos reales diferenciados de sintéticos
- [ ] Gráficas, tablas y conclusiones permiten verificar la mejora
- [ ] Informe, repositorio, evidencias, presentación y contribuciones entregados
- [ ] Demostración y defensa caben en 15 minutos
