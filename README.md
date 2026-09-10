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

### Viabilidad descriptiva de dirección del primer saque (P02)

P02 describe la dirección inicial reconocida del primer saque (`wide`, `body`
y `T`) hasta el 2023-12-31, con el test 2024--2026 sellado y excluido antes de
la lectura analítica. Incluye también los primeros saques que terminan en
falta: la dirección representa el intento, mientras que `server_won_point`
representa el resultado final del punto. Los outcomes inmediatos son solo
diagnósticos del parser actual.

La publicación actual describe 1.035.760 puntos de desarrollo de 5.993
partidos y 870 servidores; 1.019.891 puntos tienen dirección inicial reconocida.

```powershell
python -m src.analysis.first_serve_direction_feasibility
```

Artefactos versionables generados:

- `reports/first_serve_direction_feasibility_summary.json`
- `reports/tables/first_serve_direction_feasibility_by_direction.csv`
- `reports/tables/first_serve_direction_feasibility_outcomes.csv`
- `reports/tables/first_serve_direction_feasibility_coverage.csv`

La cobertura por servidor es retrospectiva y constituye solo un límite superior
estructural: no hay perfiles históricos, evidencia conjunta servidor-rival,
selección de thresholds, scoring, causalidad ni recomendaciones.

### Viabilidad descriptiva de intención anotada de saque y volea (P03)

P03 cuenta exclusivamente el marcador literal `+` cuando comienza exactamente
al final del prefijo de servicio. Es una intención anotada: no prueba que el
servidor llegara a la red ni que realizara una volea. La unidad es el intento
de saque; el segundo intento solo se construye cuando `second_serve` contiene
texto sustantivo. El test 2024--2026 queda sellado antes del parser.

La publicación describe 1.426.863 intentos de desarrollo (1.035.760 primeros
y 391.103 segundos). En 130.981 se observa `+` inmediato; 130.963 de ellos
tienen dirección conocida y son `positive_tagged`, mientras que 18 con
dirección `unknown` permanecen fuera del comparador. El parser actual deja los
rallies no etiquetados como `unknown`, por lo que no hay un grupo
`not_explicitly_tagged` observable y la comparación de outcomes no está
disponible. No se comparan etiquetas positivas contra `unknown` o censura.

```powershell
python -m src.analysis.serve_and_volley_descriptive_feasibility
```

Artefactos versionables generados:

- `reports/serve_and_volley_descriptive_feasibility_summary.json`
- `reports/tables/serve_and_volley_descriptive_feasibility_by_state.csv`
- `reports/tables/serve_and_volley_descriptive_feasibility_by_group.csv`
- `reports/tables/serve_and_volley_descriptive_feasibility_outcomes.csv`

Es un estudio descriptivo observacional: no fija thresholds, no hace scoring,
modelos ni recomendaciones, y no establece efectos causales.

### Viabilidad descriptiva de dirección lateral del primer resto (P04)

P04 observa únicamente la dirección lateral documentada del primer evento de
resto localizado inmediatamente tras un saque entrado. Los códigos `1`, `2` y
`3` indican, respectivamente, lado de derecha de un diestro/revés de un zurdo,
zona central y lado de revés de un diestro/derecha de un zurdo; `0` indica
dirección lateral desconocida. No significan cruzado, paralelo, izquierda o
derecha universal, ni las direcciones de saque wide/body/T. La profundidad
`7/8/9/0` se conserva como diagnóstico separado.

La unidad es el intento de saque sustantivo; el segundo solo existe cuando
`second_serve` contiene texto sustantivo. En desarrollo hasta 2023 se publican
1.426.863 intentos, de los que 830.471 contienen dirección lateral `1/2/3`
(cobertura end-to-end aproximada del 58,20 %); las tres categorías aparecen.
Los outcomes son tasas observacionales del punto
ganado por el restador, con intervalos Wilson, solo entre esas direcciones
observadas: no incluyen unknown ni censura.

```powershell
python -m src.analysis.return_direction_descriptive_feasibility
```

Artefactos agregados versionables:

- `reports/return_direction_descriptive_feasibility_summary.json`
- `reports/tables/return_direction_descriptive_feasibility_by_state.csv`
- `reports/tables/return_direction_descriptive_feasibility_by_direction.csv`
- `reports/tables/return_direction_descriptive_feasibility_by_group.csv`
- `reports/tables/return_direction_descriptive_feasibility_outcomes.csv`

El test 2024--2026 permanece sellado: sus 1.531 partidos se excluyen antes de
construir intentos o llamar al extractor. Es un estudio descriptivo
observacional, condicionado a retornos documentados; no selecciona thresholds,
perfiles, modelos, scoring ni recomendaciones, y no permite afirmaciones
causales.

### Viabilidad descriptiva de profundidad documentada del primer resto (P05)

P05 mide únicamente la profundidad documentada del primer resto: `7` dentro de
los cuadros de saque; `8` tras la línea de saque, más cerca de ella que de la
línea de fondo; `9` más cerca de la línea de fondo que de la línea de saque; y
`0` profundidad desconocida. No emplea etiquetas espaciales alternativas.

En desarrollo hasta 2023 publica 1.035.760 puntos de 5.993 partidos y 870
servidores: 1.426.863 intentos (1.035.760 primeros y 391.103 segundos). Hay
601.246 profundidades observadas `7`/`8`/`9`, una cobertura end-to-end del
42,14 %. Los outcomes son tasas descriptivas condicionadas exclusivamente a
esa profundidad observada, con intervalos Wilson; pueden sufrir sesgo de
observabilidad y censura y no implican causalidad.

```powershell
python -m src.analysis.return_depth_descriptive_feasibility
```

Artefactos agregados versionables:

- `reports/return_depth_descriptive_feasibility_summary.json`
- `reports/tables/return_depth_descriptive_feasibility_by_state.csv`
- `reports/tables/return_depth_descriptive_feasibility_by_depth.csv`
- `reports/tables/return_depth_descriptive_feasibility_by_group.csv`
- `reports/tables/return_depth_descriptive_feasibility_outcomes.csv`

El test 2024--2026 permanece sellado: 1.531 partidos se excluyen antes de
construir intentos; no se seleccionan thresholds, perfiles, modelos, scoring
ni recomendaciones.

### Viabilidad descriptiva del tipo documentado del primer resto (P06)

P06 conserva únicamente el código literal del primer golpe de resto documentado:
17 códigos (`f`, `b`, `r`, `s`, `v`, `z`, `o`, `p`, `u`, `y`, `l`, `m`, `h`,
`i`, `j`, `k`, `t`) como resultado primario, y nueve familias preespecificadas
como agregado secundario. `q` permanece como tipo desconocido no comparable;
`t` no representa una técnica homogénea.

En desarrollo hasta 2023 publica 1.035.760 puntos de 5.993 partidos y 870
servidores: 1.426.863 intentos (1.035.760 primeros y 391.103 segundos). Hay
881.717 tipos documentados, con cobertura end-to-end del 61,79 %. Los outcomes
son tasas descriptivas condicionadas a tipo documentado, con intervalos Wilson;
pueden sufrir sesgo de observabilidad y censura y no implican causalidad.

```powershell
$performanceLog = Join-Path (Split-Path (Get-Location) -Parent) 'p06-performance.json'
python -m src.analysis.return_shot_type_descriptive_feasibility --performance-log $performanceLog
```

Artefactos agregados versionables:

- `reports/return_shot_type_descriptive_feasibility_summary.json`
- `reports/tables/return_shot_type_descriptive_feasibility_by_state.csv`
- `reports/tables/return_shot_type_descriptive_feasibility_by_type.csv`
- `reports/tables/return_shot_type_descriptive_feasibility_by_group.csv`
- `reports/tables/return_shot_type_descriptive_feasibility_outcomes.csv`

El test 2024--2026 permanece sellado: 1.531 partidos se excluyen antes de
construir intentos o llamar al extractor. P06 no fija thresholds, perfiles,
modelos, scoring ni recomendaciones.

### Resultado terminal documentado inmediatamente del primer resto (P07)

P07 atribuye solo un ganador, error forzado o error no forzado que pertenezca
al primer resto local y cierre completamente la secuencia. No busca terminales
posteriores; la ausencia de terminal no prueba que el rally continuase, y el
outcome no interviene en la clasificacion. En desarrollo hasta 2023 publica
1.426.863 intentos (1.035.760 primeros y 391.103 segundos), con 74.178
terminales elegibles (5,20 %): 17.955 winners, 16.550 forced errors y 39.673
unforced errors (`n` 14.541, `w` 8.063, `d` 15.158, `x` 1.480, `e` 10 y `!`
421). Los segundos intentos se diagnostican como 382.143 con primera falta
documentada y 8.960 sin ella; no se infiere ni corrige una falta previa.

```powershell
python -m src.analysis.return_terminal_descriptive_feasibility
```

Artefactos agregados versionables:

- `reports/return_terminal_descriptive_feasibility_summary.json`
- `reports/tables/return_terminal_descriptive_feasibility_by_state.csv`
- `reports/tables/return_terminal_descriptive_feasibility_by_terminal.csv`
- `reports/tables/return_terminal_descriptive_feasibility_by_group.csv`
- `reports/tables/return_terminal_descriptive_feasibility_consistency.csv`

La comprobacion terminal--outcome conserva tres discordancias como auditoria de
anotacion, sin corregirlas ni interpretarlas como eficacia tactica. El test
2024--2026 permanece sellado; P07 es descriptivo, no causal, y no genera
recomendaciones. La publicacion final se reparo una unica vez solo sobre los
artefactos para restaurar el orden canonico de superficies; no hubo una tercera
ejecucion real.

### Aproximacion documentada inmediatamente tras el primer resto (P08)

P08 identifica exclusivamente el marcador literal `+` situado inmediatamente
despues del tipo conocido del primer resto. Es distinto de P03: aquel marcador
describe la intencion anotada del servidor tras el saque, mientras que P08
pertenece al restador. En desarrollo hasta 2023 se examinaron 1.426.863 intentos
(1.035.760 primeros y 391.103 segundos): 12.434 contienen P08 documentado
(0,8714 %), 885.892 quedan como `unknown_initial_return` y 528.537 estan
censurados. No se observo ningun negativo P08 inequivoco, por lo que el estado
es `available_descriptive_not_comparable` y los outcomes no estan disponibles.

```powershell
python -m src.analysis.return_approach_descriptive_feasibility --performance-log <ruta-externa>
```

Artefactos agregados versionables:

- `reports/return_approach_descriptive_feasibility_summary.json`
- `reports/tables/return_approach_descriptive_feasibility_by_state.csv`
- `reports/tables/return_approach_descriptive_feasibility_by_group.csv`
- `reports/tables/return_approach_descriptive_feasibility_marker_context.csv`
- `reports/tables/return_approach_descriptive_feasibility_outcomes.csv`

El test 2024--2026 permanece sellado (1.531 partidos excluidos). P08 es una
etiqueta documental y observacional, no causal: no demuestra llegada a red,
volea fisica ni efectividad, y no genera una recomendacion tactica.

### Perfil documental compuesto del primer resto (P09)

P09 combina exclusivamente perfiles completos del primer resto: tipo de golpe
documentado x direccion lateral `1/2/3` x profundidad `7/8/9`. El significado
documental original de direccion y profundidad se conserva; no se renombran
como cruzado/paralelo ni corto/profundo. El catalogo cerrado contiene 153
perfiles (17 x 3 x 3).

En desarrollo hasta 2023 se analizaron 1.426.863 intentos: 1.035.760 primeros
y 391.103 segundos. Hay 601.246 perfiles completos (aprox. 42,14 %), de los
que 79 perfiles aparecen al menos una vez, 74 tienen cero observaciones y 19
son raros, definidos descriptivamente como 1--9 intentos. Por tipo predominan
`b` (271.819), `f` (221.735), `s` (80.891) y `r` (24.593); los otros trece
codigos suman 2.208. Las direcciones `1/2/3` suman respectivamente
103.527/312.084/185.635 intentos y las profundidades `7/8/9`,
159.482/293.400/148.364.

```powershell
python -m src.analysis.return_profile_descriptive_feasibility --performance-log <ruta-externa>
```

Artefactos agregados versionables:

- `reports/return_profile_descriptive_feasibility_summary.json`
- `reports/tables/return_profile_descriptive_feasibility_by_state.csv`
- `reports/tables/return_profile_descriptive_feasibility_by_profile.csv`
- `reports/tables/return_profile_descriptive_feasibility_by_group.csv`
- `reports/tables/return_profile_descriptive_feasibility_outcomes.csv`

Los outcomes son asociaciones condicionadas a los 601.246 perfiles
completamente documentados (286.028 puntos ganados por el restador y 315.218
por el servidor), con intervalos Wilson descriptivos. La observabilidad puede
introducir seleccion y censura, y los perfiles raros producen intervalos muy
amplios. El test 2024--2026 sigue sellado con 1.531 partidos excluidos. P09 no
establece causalidad ni una estrategia optima, y no selecciona thresholds,
smoothing, scoring, rankings o recomendaciones automaticas.

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

### Perfiles historicos descriptivos sin fuga temporal

El constructor genera en memoria 15.048 snapshots con unidad
`target_match_id x target_player`: exactamente dos participantes por cada uno
de los 7.524 partidos. Usa solo historia con fecha estrictamente anterior. Los
partidos del mismo jugador en la misma fecha se consideran simultaneos, sin
ordenarlos por hora, ronda, torneo o identificador. Resume por separado los
conteos brutos del jugador objetivo al saque y del oponente al resto, tanto
globales como en la superficie objetivo.

```powershell
python -m src.analysis.historical_profiles
```

Artefactos versionables generados:

- `reports/historical_profiles_summary.json`
- `reports/tables/historical_profiles_coverage.csv`

Los snapshots completos de `target_match_id x target_player` no se publican
como artefacto en esta fase. Los cortes de cobertura son exclusivamente
descriptivos y no constituyen umbrales aprobados para features o modelos. No
se aplican tasas, suavizado, fallback, ventanas historicas, head-to-head,
modelos ni recomendaciones. La identidad textual exacta se considera valida
solo para el Parquet procesado fijado. La eleccion de ventanas, suavizado,
umbrales y validacion rolling-origin permanece pendiente para fases posteriores.

### Variables comparables jugador-rival por direccion

El constructor descriptivo amplía cada snapshot histórico a `wide`, `body` y
`T`, tanto globalmente como en la superficie objetivo. Su unidad en memoria es
`target_match_id x target_player x direction x scope` (90.288 filas). Todas las
tasas utilizan el outcome común `P(server wins point)` y se publican como
proporciones brutas acompañadas por IC95% Wilson. El baseline poblacional usa
exclusivamente días anteriores al partido objetivo, igual que los historiales
del servidor target y del rival-restador. La tabla completa de 90.288 filas
permanece en memoria y no se versiona.

```powershell
python -m src.analysis.player_opponent_direction_features
```

Artefactos versionables generados:

- `reports/player_opponent_direction_features_summary.json`
- `reports/tables/player_opponent_direction_features_coverage.csv`
- `reports/tables/player_opponent_direction_features_stability.csv`

Se conservan estados explícitos de ausencia o presencia de evidencia. Los
cortes de cobertura son solo descriptivos: no filtran filas ni seleccionan un
umbral. No se aplican smoothing, fallback, scoring, modelos o recomendaciones,
y las comparaciones son asociaciones históricas, no efectos causales.

### Protocolo de validacion cronologica

El constructor lee exclusivamente `match_id`, `date`, `player_1`, `player_2` y
`surface` del Parquet procesado y lo reduce inmediatamente a una fila por
partido. Asigna fechas civiles completas a train (hasta 2019), validation
(2020--2023) y test final sellado (2024--2026-05-21). Todos los partidos de una
fecha se evalúan contra el estado anterior y solo después actualizan la historia;
no se usa orden intradía. La validación principal usa cuatro folds rolling-origin
expansivos para 2020--2023, con definición de features, política de evidencia,
scoring y parámetros fijados antes de cada evaluación.

La sensibilidad frozen mantiene features, baselines y parámetros fijados al
2023-12-31 durante los mismos 1.531 partidos de test y no actualiza el estado con
partidos del propio test. El test permanece sellado, registra cero ejecuciones de
evaluación y no interviene en la selección de thresholds, ventanas, smoothing,
fallback, scoring, hiperparámetros o definiciones de features; se reserva para un
único uso final. La fuente termina el 2026-05-21, por lo que 2026 se marca como
año parcial sin afirmar cobertura histórica exhaustiva para otros años.

```powershell
python -m src.analysis.chronological_validation
```

Artefactos agregados versionables:

- `reports/chronological_validation_summary.json`
- `reports/tables/chronological_validation_by_year.csv`
- `reports/tables/chronological_validation_folds.csv`

No se publica una tabla partido a partido. El cold start aqui documentado solo
indica si existe algun partido en una fecha estrictamente anterior; no mide
evidencia direccional suficiente y no selecciona thresholds, modelos ni
recomendaciones.

### Validacion cronologica de politicas de evidencia

La evaluacion descriptiva compara en los folds rolling-origin 2020--2023 el
producto cartesiano fijo de 25/50/100 puntos, 3/5/10 partidos y los scopes
`global_only`, `surface_only` y `surface_then_global`: exactamente 27
politicas. Una orientacion-direccion solo es elegible cuando el jugador al
saque y el rival al resto superan conjuntamente ambos minimos en el mismo
scope. El fallback de superficie a global es atomico para los dos roles.

El Parquet completo se lee una sola vez para reconciliar la fuente de 7.524
partidos. Las cardinalidades upstream son 15.048 snapshots y 90.288 filas de
features, pero este analisis excluye los 1.531 targets de test antes de llamar
a sus constructores: solo construye 5.993 partidos target hasta 2023, 11.986
snapshots y 71.916 filas de features. Por tanto, se construyen y evaluan cero
targets o features de test.

```powershell
python -m src.analysis.evidence_policy_validation
```

Artefactos agregados versionables:

- `reports/evidence_policy_validation_summary.json`
- `reports/tables/evidence_policy_validation_candidates.csv`
- `reports/tables/evidence_policy_validation_by_fold.csv`
- `reports/tables/evidence_policy_validation_pareto.csv`

La shortlist conserva todas las politicas no dominadas al maximizar la peor
cobertura de partidos completos y minimizar los peores p90 de amplitud Wilson
y cambio historico. No selecciona automaticamente una ganadora. El test
2024--2026 permanece sellado con cero evaluaciones y no interviene en esta
comparacion. No se entrena ningun modelo ni se crean scoring, recomendaciones
o afirmaciones causales.

#### Seleccion metodologica de la politica de evidencia

La regla congelada sobre validacion 2020--2023 selecciona
`p050_m05_surface_then_global`: exige conjuntamente al servidor y al oponente
al menos 50 puntos y 5 partidos por direccion. Usa historia de la superficie
solo cuando ambos roles superan los minimos; en caso contrario aplica fallback
global conjunto y se abstiene si tampoco existe evidencia global suficiente.
La candidata supera los limites preespecificados de peor cobertura completa
(`>= 0.50`), peor p90 Wilson (`<= 0.25`) y peor p90 de cambio historico
(`<= 0.08`). Estos thresholds son reglas operativas, no suficiencia universal.

```powershell
python -m src.analysis.evidence_policy_selection
```

Artefactos agregados versionables:

- `reports/evidence_policy_selection_summary.json`
- `reports/tables/evidence_policy_selection_comparison.csv`

La decision consume exclusivamente los artefactos agregados del commit
`88fafe7`; no lee el Parquet ni reconstruye perfiles. El test 2024--2026 sigue
sellado y no interviene. La politica controla disponibilidad de evidencia: no
demuestra causalidad, no entrena un modelo y no genera recomendaciones tacticas
automaticas. La shortlist Pareto completa anterior se conserva como contexto.
El control `p050_m05_global_only` se compara siempre de forma descriptiva, pero
solo se activa como fallback si falla la candidata primaria; en el resultado
publicado la primaria pasa y el fallback no se evalua ni se activa.

#### Baseline explicable de scoring por direccion

El baseline descriptivo aplica la politica seleccionada
`p050_m05_surface_then_global` a las features historicas sin fuga y evalua los
folds 2020--2023. Para cada direccion elegible combina, con pesos iguales, la
tasa historica del servidor y la tasa historica permitida por el rival al resto
alrededor de la tasa poblacional: `score = population + 0.5 * (server -
population) + 0.5 * (opponent_allowed - population)`. Compara cuatro scorers:
poblacion, solo servidor, solo rival y el combinado igualitario.

```powershell
python -m src.analysis.explainable_direction_scoring
```

Artefactos agregados versionables:

- `reports/explainable_direction_scoring_summary.json`
- `reports/tables/explainable_direction_scoring_by_fold.csv`
- `reports/tables/explainable_direction_scoring_by_direction.csv`
- `reports/tables/explainable_direction_scoring_calibration.csv`
- `reports/tables/explainable_direction_scoring_ranking.csv`

El baseline queda validado solo si satisface los criterios congelados de Brier,
log-loss, calibracion, rango finito de probabilidades y estabilidad por fold.
Se abstiene cuando no hay evidencia conjunta suficiente para las tres
direcciones y ambos roles. El test 2024--2026 permanece sellado y recibe cero
evaluaciones. Este score es descriptivo: no establece causalidad, no constituye
una recomendacion tactica final y no sustituye la evaluacion posterior del
protocolo sellado. La mejora observada frente al scorer poblacional es modesta
(Δ Brier aproximado -0,000359; Δ log-loss aproximado -0,000715): no demuestra
relevancia tactica ni garantiza generalizacion al test sellado, y no genera
recomendaciones automaticas.

#### Motor explicable de recomendaciones por dirección

El motor transforma el score descriptivo congelado en un orden reproducible de
`wide`, `body` y `T`, con la misma fórmula `0.5 * server_selected_rate + 0.5 *
opponent_allowed_rate` y la política `p050_m05_surface_then_global`. La política
upstream selecciona evidencia por dirección: exige conjuntamente al servidor y
al rival-restador al menos 50 puntos y 5 partidos, priorizando superficie y
aplicando fallback global conjunto entre ambos roles cuando procede.

Como barrera conservadora propia del motor, solo se publica un ranking completo
cuando las tres direcciones puntuables comparten el mismo scope (`surface` o
`global`). Los scopes mixtos se abstienen, sin recalcular scores ni publicar un
ranking parcial; esta decisión puede reducir cobertura y no afirma que ambos
scopes sean estadísticamente incompatibles. El orden es por score descendente y
los empates se resuelven de forma determinista `wide`, `body`, `T`, únicamente
para reproducibilidad. Las explicaciones agregan componentes, evidencia y
códigos deterministas; la tabla individual permanece solo en memoria.

En la ejecución publicada hay 2.295 de 3.610 orientaciones disponibles
(aprox. 63,57 %): 1.844 con scope común de superficie y 451 con scope global
común. Se abstienen 1.068 por evidencia insuficiente y 247 por scopes
incompatibles; exigir un scope común reduce deliberadamente la cobertura y las
orientaciones con scope mixto no producen ranking. El resultado es descriptivo:
no implica causalidad, optimalidad táctica ni generalización al test sellado.

```powershell
python -m src.analysis.explainable_direction_recommender
```

Artefactos agregados versionables:

- `reports/explainable_direction_recommender_summary.json`
- `reports/tables/explainable_direction_recommender_coverage.csv`
- `reports/tables/explainable_direction_recommender_rankings.csv`
- `reports/tables/explainable_direction_recommender_explanations.csv`

La ejecución publicada cubre los folds 2020--2023 y mantiene sellado el test
2024--2026: se excluyen 1.531 targets antes de snapshots y features, y registra
cero recomendaciones, evaluaciones o selección metodológica sobre el test. Los
rankings son asociaciones históricas descriptivas; no demuestran causalidad,
optimalidad táctica ni generalización al test sellado.

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
