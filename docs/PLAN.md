# Plan del Proyecto 2

CC3074 Modelización y Simulación, UVG, ciclo 2 de 2026.

## Estrategia

**Tema.** Optimización de la estrategia de paradas en pits de F1 con simulación Monte Carlo (técnica de clase) y un algoritmo genético con restricciones (técnica adicional).

**Pregunta de investigación.** ¿Qué estrategia (número de paradas, vueltas y compuestos) minimiza el tiempo total esperado de carrera y cuánto mejora frente a la estrategia real?

**Métricas.** Principal: tiempo total de carrera en segundos, a minimizar. Secundarias: desviación estándar del tiempo total y probabilidad de superar a la base.

**Escenario.** GP de España (Barcelona), reglamento 2022 a 2025. No se usa 2026 porque cambiaron los autos. Se calibra con 2022, 2023 y 2024 y se valida fuera de muestra con 2025. La base es la estrategia real del ganador de 2025, verificada con los datos. Quedan fuera tráfico, adelantamientos, lluvia e incidentes propios.

**Modelo por vuelta.** `t = t_base + δ_c + k_c·a + k2_c·a² + f·(N−l) + ε`, con ε~N(0,σ²). Pérdida en pits con distribución empírica del circuito. Safety Car como Bernoulli por vuelta con p_SC histórica 2018 a 2024. Vuelta bajo SC más lenta y parada más barata, estimadas de los datos si es posible. Calibración por regresión sobre vueltas limpias.

**Datos.** Solo fuentes gratuitas. El servidor oficial de F1 exige suscripción F1TV desde FastF1 3.9, así que no se usa. 2022 y años anteriores salen del espejo público de FastF1. 2023, 2024 y 2025 salen de la API pública OpenF1 (`/laps`, `/stints`, `/pit`, `/race_control`). Jolpica-F1 da resultados y paradas. Pirelli da los compuestos y el Reglamento Deportivo FIA la regla de 2 compuestos. La caché local va en `data/cache` (ignorada por git) y los CSV procesados en `data/processed`, con script de descarga. OpenF1 no trae `pick_quicklaps` ni TrackStatus por vuelta, así que el filtro de vueltas limpias es propio: se excluyen la vuelta 1, las vueltas de entrada y salida de pits, las vueltas neutralizadas según `race_control` y los atípicos.

**Algoritmo genético.** Implementación propia. Cromosoma `[n_paradas, vueltas…, compuestos…]`. Objetivo E[T] con R réplicas y números aleatorios comunes (variante E[T] + λ·σ). Restricciones con las reglas de Deb (2000). Torneo, cruce uniforme, mutaciones (mover parada, cambiar compuesto, agregar o quitar parada) y elitismo. Termina en G generaciones o 20 sin mejora. Verificación contra fuerza bruta en 1 y 2 paradas. Extensión reactiva al SC con umbral en el cromosoma.

**Experimentos.** Base, GA, sensibilidad p_SC ×2, sensibilidad degradación +30 %, y Monza si alcanza el tiempo. Al menos 30 repeticiones por condición (1000 si es barato), semillas en `results/`. Media, desviación estándar e IC 95 %.

**Validación.** Reproducir la estrategia real 2025: MAE por vuelta y error del tiempo total. Tabla real, base y mejorado con cambio absoluto y porcentual. Separar mejora simulada de mejora real. Pruebas con pytest.

## Checklist

### Parte 1. Problema
- [ ] Contexto, actores, entradas y salidas
- [x] Pregunta de investigación concreta
- [x] Métrica principal con unidades y sentido (min)
- [ ] Alcance, restricciones, supuestos y exclusiones
- [ ] Al menos una fuente verificable del escenario

### Parte 2. Investigación técnica (GA con restricciones)
- [x] Objetivo, representación, parámetros, procedimiento y terminación
- [ ] Al menos 3 fuentes técnicas (1 académica o documentación)
- [ ] Justificación frente a alternativas
- [ ] Al menos 2 limitaciones
- [x] Qué hace la biblioteca y qué hace el equipo

### Parte 3. Modelo y simulación
- [x] Diagrama del sistema
- [ ] Variables de estado, parámetros, entradas, salidas, restricciones y distribuciones
- [ ] Reglas, fórmulas o pseudocódigo
- [ ] Procedencia, valor y unidad de cada parámetro
- [x] Simulación funcional y parametrizable (`config.yaml`), con modelo, simulación, técnica y visualización separados
- [x] Reproducción desde entorno limpio documentada

### Parte 4. Mejora
- [x] Función objetivo y restricciones definidas formalmente
- [x] Integración GA y simulación explicada
- [ ] Hiperparámetros con valor y justificación
- [x] Solución candidata verificada por código (`verificar_restricciones()`)
- [x] Alternativa base sin mejora conservada

### Parte 5. Experimentación y validación
- [x] Al menos 3 condiciones: base, mejorada, sensibilidad
- [x] Entradas, parámetros, resultados y métricas registrados por condición
- [x] ≥30 repeticiones, semillas registradas, media y dispersión
- [x] Propiedad interna probada (pytest: conservación, degradación 0, determinismo, restricciones)
- [x] Comparación con datos de referencia (MAE por vuelta y error del total)
- [x] Prueba de sensibilidad explicada
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
- [x] Tiempo por vuelta real vs simulado por compuesto
- [x] Boxplot del tiempo total por condición
- [x] Convergencia del GA
- [x] Mapa de calor de E[T] por vuelta de parada 1 y parada 2

### Entregables
- [ ] Informe técnico PDF (portada, problema, objetivos, fuentes, técnica, modelo, implementación, diseño experimental, validación, comparación real, resultados, limitaciones, conclusiones, referencias, contribuciones, declaración de uso de IA)
- [x] Repositorio con código, README reproducible, `requirements.txt` con versiones fijas, datos o script de descarga
- [x] Evidencia experimental en `results/` (tablas, parámetros, semillas, gráficas, métricas)
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
- [x] Simulación y mejora se ejecutan desde el README
- [x] Base, mejorado y sensibilidad documentados
- [x] Resultados, incertidumbre y errores verificados y reproducibles
- [ ] Referencia real documentada y datos reales diferenciados de sintéticos
- [ ] Gráficas, tablas y conclusiones permiten verificar la mejora
- [ ] Informe, repositorio, evidencias, presentación y contribuciones entregados
- [ ] Demostración y defensa caben en 15 minutos

## Cambios respecto del plan original

**p_SC con 2021 a 2024 en lugar de 2018 a 2024.** El espejo de FastF1 no tiene 2018, 2019 ni 2020, y OpenF1 empieza en 2023. Hubo 1 SC en 264 vueltas.

**Factor de parada bajo SC como supuesto (0.5) con sensibilidad 0.3 y 0.7.** Las paradas bajo SC observadas dan pérdidas por vuelta no interpretables por el reagrupamiento del pelotón.

**Segunda base "planificada" (S-M-S en 22 y 49).** La tercera parada real fue una reacción al SC de la vuelta 55, así que la base real se castiga en réplicas sin SC.

**Monza no se hizo.** Era opcional y no hay tiempo de calibrar otro circuito con el mismo cuidado.
