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

## Documentación de datos

La procedencia y la notación oficial de las secuencias de servicio se documentan
en [`docs/data/serve_sequence_grammar_sources.md`](docs/data/serve_sequence_grammar_sources.md).

### Auditoría observable de secuencias de servicio

La auditoría describe presencia, longitudes y caracteres de `first_serve` y
`second_serve` sin aplicar una gramática ni un parser:

```powershell
python -m src.analysis.serve_sequence_audit
```

Artefactos versionables generados:

- `reports/serve_sequence_audit_summary.json`
- `reports/tables/serve_sequence_audit_by_group.csv`
- `reports/tables/serve_sequence_character_inventory.csv`
- `reports/tables/serve_sequence_examples.csv`

### Análisis descriptivo de dirección del segundo servicio

El análisis utiliza las secuencias sustantivas de `second_serve` para describir
dirección, superficie, periodo y resultado observado del punto:

```powershell
python -m src.analysis.second_serve_direction_analysis
```

Artefactos versionables generados:

- `reports/second_serve_direction_summary.json`
- `reports/tables/second_serve_direction_by_group.csv`
- `reports/tables/second_serve_direction_coverage.csv`

El resultado es exclusivamente descriptivo y no estima efectos causales. No
incluye longitud del intercambio ni lado de servicio, y sus cortes de cobertura
no son umbrales aprobados para features o modelos.

### Baseline descriptivo de segundo servicio

El baseline incluye los 481.190 puntos con segundo saque sustantivo y prefijo
inicial reconocido en direccion `4`, `5` o `6`. Los warnings, el residuo y el
rally posterior no excluyen el punto; los outcomes conservadores son solo un
diagnostico.

```powershell
python -m src.analysis.second_serve_direction_baseline
```

Artefactos versionables generados:

- `reports/second_serve_direction_baseline_summary.json`
- `reports/tables/second_serve_direction_baseline_by_group.csv`

Este analisis no genera recomendaciones, no estima causalidad y no utiliza rally
length, resto, golpes, train/test split ni features historicas.

### Modelo ajustado de direccion del segundo servicio

La especificacion asociativa prevista usa `wide`, `Hard`, `to_2009` y un
servidor de referencia determinista. No se ajusta en la implementacion actual:
la estimacion preventiva de memoria excede el limite seguro antes de construir
la matriz densa.

```powershell
python -m src.analysis.second_serve_direction_adjusted
```

La especificacion deja previstos errores estandar robustos agrupados por
`match_id` y genera:

- `reports/second_serve_direction_adjusted_summary.json`
- `reports/tables/second_serve_direction_adjusted_coefficients.csv`

No hay OR, IC ni p-values publicados. Una futura alternativa requeriria diseno
disperso o logistica condicional; no esta implementada. El analisis no es causal
ni produce recomendaciones tacticas.

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
