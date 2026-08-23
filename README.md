# Tennis Strategy Recommender

TFM de Omar Errandi para construir un recomendador explicable de estrategias de tenis a partir de datos profesionales a nivel de punto.

## Estado

Proyecto en fase de inicialización.

## Primer hito

Demostrar empíricamente la viabilidad del Match Charting Project:

1. Descargar los datos oficiales.
2. Integrar puntos y partidos.
3. Analizar calidad y cobertura.
4. Implementar el primer parser de secuencias.
5. Evaluar segundo saque y longitud del intercambio.

Consulta `PROJECT_CONTEXT.md` para el alcance completo y `AGENTS.md` para las reglas de trabajo.

## Analisis reproducible de cobertura

El analisis de cobertura utiliza `data/processed/points_enriched.parquet` y
agrega partidos y puntos por jugador, superficie y ano:

```powershell
python -m src.analysis.coverage
```

Artefactos versionables generados:

- `reports/coverage_summary.json`
- `reports/tables/coverage_by_player_surface_year.csv`
- `reports/tables/coverage_by_surface_year.csv`

Figuras generadas localmente y mantenidas fuera de Git:

- `reports/figures/coverage_by_surface_year.png`
- `reports/figures/coverage_distribution.png`

Los cortes de 1, 3, 5 y 10 partidos son exclusivamente descriptivos; no son
umbrales aprobados para el futuro modelo.

## Próximo paso

Comprobar el entorno local de Windows antes de fijar dependencias:

```powershell
python --version
git --version
code --version
conda --version
```
