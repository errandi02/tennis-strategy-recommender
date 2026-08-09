# Instrucciones para Codex

Lee `PROJECT_CONTEXT.md` antes de proponer o implementar cambios.

## Reglas del proyecto

- Trabaja paso a paso y verifica cada bloque antes de continuar.
- La unidad principal del análisis es el punto; los metadatos del partido aportan contexto.
- Evita cualquier fuga de información entre puntos o partidos.
- Toda división de datos debe agrupar partidos completos y, cuando corresponda, respetar el tiempo.
- Las estadísticas históricas solo pueden utilizar información anterior al partido evaluado.
- Separa notebooks exploratorios del código reutilizable en `src/`.
- No modifiques archivos de datos originales.
- Añade pruebas para el parser y para cualquier transformación crítica.
- No presentes asociaciones observacionales como relaciones causales.
- No amplíes el MVP con vídeo, tiempo real, Kubernetes, LLM o reentrenamiento automático sin aprobación explícita.
- Documenta comandos, decisiones, resultados y limitaciones que puedan reutilizarse en la memoria del TFM.

## Calidad mínima antes de cerrar una tarea

1. El código relevante se ejecuta.
2. Las pruebas relacionadas pasan.
3. No se han introducido fugas temporales o entre partidos.
4. La documentación refleja el cambio.
5. Se informa de cualquier supuesto, dato ausente o limitación.
