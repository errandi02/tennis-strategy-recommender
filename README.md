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

La especificacion asociativa usa `wide`, `Hard`, `to_2009` y un servidor de
referencia determinista. La limitacion de memoria del diseno denso se resolvio
con CSR y el nucleo disperso se valido sinteticamente frente a `statsmodels`.
El ajuste real realizo dos intentos: el segundo reutilizo la solucion del
primero mediante *warm start*. El resultado permanece `not_available` por no
convergencia y una norma de gradiente superior a la tolerancia; el segundo
intento termino con perdida de precision. Por ello no se publican OR, IC,
p-values ni contraste Wald. Esto no implica ausencia de asociacion: esta
especificacion con 1.002 efectos fijos de servidor no proporciona inferencia
suficientemente fiable con el metodo actual y no genera recomendacion tactica.

```powershell
python -m src.analysis.second_serve_direction_adjusted
```

La especificacion deja previstos errores estandar robustos agrupados por
`match_id` y genera:

- `reports/second_serve_direction_adjusted_summary.json`
- `reports/tables/second_serve_direction_adjusted_coefficients.csv`

No hay OR, IC ni p-values publicados. El analisis no es causal.
ni produce recomendaciones tacticas.

### Modelo pooled inferencial de direccion del segundo servicio

El analisis pooled estima la asociacion poblacional de la direccion del segundo
saque, ajustada por superficie y periodo:

```text
server_won_point ~ direction + surface + derived_period
```

Usa `wide`, `Hard` y `to_2009` como referencias y errores estandar robustos
agrupados por `match_id`. Sus resultados son asociativos, no causales, no
generan recomendaciones tacticas y pueden conservar confusion residual por
diferencias entre servidores, que no se incluyen como efectos fijos.

```powershell
python -m src.analysis.second_serve_direction_pooled_adjusted
```

Artefactos versionables generados:

- `reports/second_serve_direction_pooled_adjusted_summary.json`
- `reports/tables/second_serve_direction_pooled_adjusted_coefficients.csv`

### Robustez y heterogeneidad del segundo servicio

El análisis de heterogeneidad usa la misma población del modelo pooled y ajusta
por separado interacciones preespecificadas de dirección con superficie y con
periodo. Los márgenes estandarizados y contrastes `body`/`T` frente a `wide`
usan covarianza robusta agrupada por partido y corrección Holm dentro de cada
familia de seis contrastes. Es un análisis exploratorio de asociación, no una
recomendación táctica ni una estimación causal. El ajuste se limita a las
covariables preespecificadas y puede conservar confusión residual por servidor;
la dependencia se agrupa por partido, no por servidor. Los Wald no significativos
no demuestran homogeneidad, y el signo, la significación estadística y la
relevancia práctica deben interpretarse por separado.

```powershell
python -m src.analysis.second_serve_direction_heterogeneity
```

Artefactos versionables generados:

- `reports/second_serve_direction_heterogeneity_summary.json`
- `reports/tables/second_serve_direction_heterogeneity_contrasts.csv`
- `reports/tables/second_serve_direction_heterogeneity_margins.csv`

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
