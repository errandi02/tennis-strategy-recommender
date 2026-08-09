# Contexto del proyecto: recomendador de estrategias de tenis

## 1. Propósito

Este repositorio contiene el Trabajo Fin de Máster de Omar Errandi para el Máster en Big Data, Data Science e Inteligencia Artificial de la UCM.

El objetivo es desarrollar una solución software que analice datos históricos de tenis profesional a nivel de punto y recomiende estrategias tácticas para un enfrentamiento entre un jugador y un rival, teniendo en cuenta la superficie y la evidencia disponible.

La salida prevista será un perfil comparativo y un **Top 3 de estrategias**, acompañado de métricas, tamaño de muestra, nivel de confianza y una explicación comprensible.

## 2. Decisiones aprobadas

- Fuente principal: Match Charting Project de Tennis Abstract.
- Circuito inicial: masculino.
- Unidad principal de análisis y modelado: el punto.
- La información de partido se integrará como contexto y para agrupaciones y validaciones.
- Salida del producto: recomendaciones previas a un partido, no recomendaciones en tiempo real.
- Enfoque: asociaciones históricas y efectividad observacional; no se afirmarán efectos causales.
- MVP productivizado: pipeline reproducible, modelo serializado, FastAPI, Streamlit, Docker, logging y pruebas básicas.
- Fuera del MVP: vídeo, tiempo real, Kubernetes, reentrenamiento automático, monitorización avanzada y chatbot.

## 3. Fuente y volumen

Repositorio oficial:

`https://github.com/JeffSackmann/tennis_MatchChartingProject`

Cifras verificadas al definir el proyecto:

- 7.566 partidos masculinos.
- 1.284.276 registros a nivel de punto.
- 14 columnas originales en los archivos de puntos.
- Objetivo aproximado de 30 a 60 variables derivadas.

Archivos principales:

- `charting-m-matches.csv`
- `charting-m-points-to-2009.csv`
- `charting-m-points-2010s.csv`
- `charting-m-points-2020s.csv`
- `charting-m-stats-*.csv`
- `data_dictionary.txt`
- `MatchChart 0.3.2.xlsm`

Licencia de la fuente: Creative Commons BY-NC-SA 4.0. La memoria y el repositorio deberán atribuir claramente la procedencia.

## 4. Información disponible por punto

Columnas originales:

`match_id`, `Pt`, `Set1`, `Set2`, `Gm1`, `Gm2`, `Pts`, `Gm#`, `TbSet`, `Svr`, `1st`, `2nd`, `Notes`, `PtWinner`.

Las secuencias de `1st` y `2nd` codifican el servicio y los golpes posteriores. El proyecto deberá implementar un parser probado que transforme esas cadenas en variables utilizables.

Grupos de variables previstos:

- Contexto: jugadores, fecha, torneo, ronda, superficie, set, juego y marcador.
- Servicio: primero o segundo, dirección, ace, doble falta y saque no devuelto.
- Resto: tipo, dirección y profundidad cuando esté disponible.
- Intercambio: longitud, tipos de golpe y direcciones.
- Finalización: ganador, error forzado y error no forzado.
- Red: subida a la red y saque-volea.
- Historial: rendimiento previo por jugador, superficie y patrón.
- Resultado: ganador del punto y resultado desde la perspectiva del jugador analizado.

## 5. Catálogo táctico inicial

Prioridad alta:

1. Atacar el segundo saque.
2. Servir al revés o a la derecha.
3. Abrir la pista con saque exterior.
4. Buscar intercambios cortos.
5. Alargar los intercambios.
6. Subir a la red.
7. Saque y volea.

Prioridad media, condicionada a la cobertura real:

8. Atacar el revés del rival.
9. Evitar la derecha del rival.
10. Restar con profundidad.
11. Modificar la agresividad en puntos de rotura.

No se priorizarán estrategias basadas en altura o efecto porque la codificación no ofrece esa información de manera suficientemente uniforme.

## 6. Diseño analítico

El dataset de modelado tendrá una fila por punto. La etiqueta inicial será si el jugador analizado ganó el punto.

Es obligatorio separar:

- Variables conocidas antes del punto: jugadores, superficie, marcador, servidor, historia previa y patrón que se evalúa.
- Variables observadas durante o después del punto: longitud final, ganador, errores y resultado.

Las variables posteriores no podrán utilizarse como entradas de una predicción previa. Sí podrán emplearse para construir etiquetas, medir patrones y realizar análisis descriptivo.

El sistema combinará:

1. Historial directo jugador-rival, si existe y tiene muestra suficiente.
2. Perfil individual del jugador y del rival.
3. Rendimiento frente a perfiles o estilos comparables.

El baseline será un motor de reglas explicables. El modelo posterior estimará o priorizará el rendimiento de patrones tácticos y se comparará con dicho baseline.

## 7. Validación y prevención de fugas

Nunca se realizará una división aleatoria de puntos que permita que el mismo partido aparezca en entrenamiento y prueba.

Reglas obligatorias:

- Dividir por partidos completos.
- Preferir una separación cronológica.
- Calcular estadísticas históricas usando exclusivamente partidos anteriores al partido evaluado.
- Ajustar transformaciones solo con el conjunto de entrenamiento.
- Mantener un conjunto de prueba final sin utilizar durante las decisiones de modelado.
- Registrar tamaño de muestra e incertidumbre de cada estrategia.

## 8. Limitaciones que deben declararse

- Cobertura desigual entre jugadores, superficies y periodos.
- Datos anotados manualmente por voluntarios.
- Variables opcionales con valores ausentes.
- Pocos enfrentamientos directos para determinadas parejas.
- Posible sesgo de selección en patrones como las subidas a la red.
- Las recomendaciones serán observacionales, no garantías causales.

## 9. Arquitectura de productivización

Flujo previsto:

`Datos MCP -> pipeline de procesamiento -> variables y perfiles -> reglas/modelo -> artefacto versionado -> API FastAPI -> aplicación Streamlit`

Requisitos del MVP:

- Código definitivo en módulos Python; notebooks reservados para exploración.
- Configuración reproducible.
- Validación de entradas.
- Serialización y versionado del modelo y transformaciones.
- API de inferencia.
- Interfaz web.
- Contenedorización.
- Logging de predicciones y errores.
- Pruebas del parser, pipeline y API.
- Documentación para ejecutar la solución.

## 10. Hitos

### Hito 1: viabilidad empírica

- Descargar y registrar los datos oficiales.
- Unir puntos con metadatos de partido.
- Validar claves, duplicados, ausencias y secuencias no interpretables.
- Medir cobertura por jugador, superficie y año.
- Implementar la primera versión del parser.
- Probar dos patrones: segundo saque y longitud del intercambio.

### Hito 2: dataset analítico

- Completar el parser.
- Construir variables derivadas.
- Crear perfiles históricos sin fuga temporal.
- Definir umbrales mínimos de cobertura.

### Hito 3: baseline y modelo

- Implementar reglas explicables.
- Entrenar modelos candidatos interpretables.
- Validar cronológicamente por partidos.
- Comparar métricas, estabilidad e incertidumbre.

### Hito 4: recomendador

- Convertir resultados en estrategias priorizadas.
- Generar evidencias y niveles de confianza.
- Validar casos de estudio.

### Hito 5: productivización

- Serializar el pipeline y el modelo.
- Implementar FastAPI y Streamlit.
- Añadir Docker, logging, pruebas y documentación.

### Hito 6: memoria y entrega

- Consolidar experimentos, resultados y limitaciones.
- Añadir atribución y licencia.
- Preparar demostración reproducible.

## 11. Forma de trabajo con Codex

- Avanzar en bloques pequeños y verificables.
- Explicar el objetivo antes de cada bloque.
- No inventar resultados: ejecutar y registrar cada comprobación.
- No avanzar al modelado hasta cerrar la viabilidad de los datos.
- Mantener trazabilidad entre código, experimento, resultado y sección de la memoria.
- Proteger los datos originales: nunca modificarlos manualmente.
- Pedir confirmación a Omar ante decisiones que cambien el alcance académico.

## 12. Siguiente acción

Comprobar en Windows las versiones de Python, Git, VS Code y Conda. Después se fijará la versión de Python, se creará el entorno aislado y se inicializará Git antes de descargar los datos.
