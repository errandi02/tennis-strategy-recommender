# Fuentes de la notación de secuencias de servicio

## 1. Propósito y alcance

Este documento registra la notación oficial disponible para los campos `1st`,
`2nd` y `Notes` del workbook del Match Charting Project. **Interpretación del
proyecto:** estos nombres se normalizan como `1st` → `first_serve`, `2nd` →
`second_serve` y `Notes` → `notes`. En adelante se utilizan los nombres
procesados. El objetivo es preservar la procedencia, separar hechos de
interpretaciones y enumerar las cuestiones que deberán resolverse antes de
implementar el parser.

No es una gramática ejecutable, no define expresiones regulares y no implica
que exista actualmente un parser en el proyecto.

### Jerarquía de evidencia

Las etiquetas siguientes se usan de forma constante:

- **Fuente oficial:** afirmación explícita de `MatchChart 0.3.2.xlsm`, con
  referencia de hoja y celda.
- **Interpretación del proyecto:** conclusión necesaria para diseñar la futura
  auditoría o el parser, pero no formulada literalmente por el workbook.
- **Observación empírica:** patrón medido en los datos del proyecto. Este
  documento no incorpora todavía resultados empíricos de las secuencias.
- **Decisión pendiente:** criterio que aún no ha sido aprobado.

## 2. Procedencia reproducible

- **Fuente oficial:** repositorio Match Charting Project de Jeff Sackmann:
  <https://github.com/JeffSackmann/tennis_MatchChartingProject>.
- **Fuente oficial:** commit fijado:
  `2c59eef194967e688b69e73df344184a06322cd8`.
- **Fuente oficial:** URL exacta del workbook:
  <https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/2c59eef194967e688b69e73df344184a06322cd8/MatchChart%200.3.2.xlsm>.
- Nombre: `MatchChart 0.3.2.xlsm`.
- Tamaño verificado: `482.316 bytes`.
- SHA-256 verificado:
  `46E2349EEE512296A86170449F6E463A6BE91BE9261A0C7B6B5D5A25C006729F`.
- **Fuente oficial:** versión 0.3.2, fechada el 15 de enero de 2019
  (`Version info!A32:C33`).
- Ruta local ignorada por Git:
  `data/raw/match_charting_project/MatchChart 0.3.2.xlsm`.

El workbook local se conserva como fuente de consulta y no debe añadirse a
Git.

## 3. Principios de `first_serve`, `second_serve` y `notes`

- **Fuente oficial:** si entra el primer saque, el saque y todo el rally se
  anotan juntos en `first_serve`; `second_serve` queda vacío
  (`Instructions!A18:A19`, `Instructions!A32`).
- **Fuente oficial:** si falla el primer saque, `first_serve` contiene solo ese
  saque y su fallo; el segundo saque y el rally se anotan juntos en
  `second_serve` (`Instructions!A33`).
- **Fuente oficial:** `Notes` es un campo libre, normalmente vacío y sin
  formato preestablecido; se pide evitar comas (`Instructions!A19`,
  `Instructions!A218:A220`).
- **Interpretación del proyecto:** la ausencia de `second_serve` no equivale
  automáticamente a un dato ausente o defectuoso. Es el estado esperado cuando
  el primer saque entra.

## 4. Vocabulario oficial

Todos los significados siguientes son contextuales. Un mismo carácter puede
tener otra función fuera de las secuencias o en otra posición.

### 4.1. Direcciones del saque

| Código | Significado | Evidencia |
|---|---|---|
| `4` | Saque abierto | `Instructions!A46:A47` |
| `5` | Saque al cuerpo | `Instructions!A46:A47` |
| `6` | Saque a la T | `Instructions!A46:A47` |
| `0` | Dirección desconocida | `Instructions!A48` |

Las direcciones se aplican igual desde los lados de iguales y ventaja.

### 4.2. Tipos de golpe

| Código | Significado | Evidencia |
|---|---|---|
| `f` / `b` | Derecha / revés, excluidos slices y golpes especiales | `Instructions!A96:A97` |
| `r` / `s` | Slice de derecha / revés | `Instructions!A99:A100` |
| `v` / `z` | Volea de derecha / revés | `Instructions!A102:A104` |
| `o` / `p` | Remate estándar / remate de revés | `Instructions!A106:A107` |
| `u` / `y` | Dejada de derecha / revés | `Instructions!A109:A110` |
| `l` / `m` | Globo de derecha / revés | `Instructions!A112:A113` |
| `h` / `i` | Media volea de derecha / revés | `Instructions!A115:A116` |
| `j` / `k` | Volea liftada de derecha / revés | `Instructions!A118:A119` |
| `t` | Golpe especial, incluido trick shot o tweener | `Instructions!A121` |
| `q` | Tipo de golpe desconocido | `Instructions!A123` |

### 4.3. Direcciones del golpe

| Código | Significado | Evidencia |
|---|---|---|
| `1` | Lado de derecha de un diestro / revés de un zurdo | `Instructions!A125:A126` |
| `2` | Zona central | `Instructions!A125:A127` |
| `3` | Lado de revés de un diestro / derecha de un zurdo | `Instructions!A125:A128` |
| `0` | Dirección desconocida | `Instructions!A138` |

**Fuente oficial:** la dirección representa la zona en la que la pelota cruza
o habría cruzado la línea de fondo rival, no necesariamente donde bota
(`Instructions!A130:A136`). La dirección del golpe es opcional
(`Instructions!A140:A141`).

### 4.4. Profundidad del resto

| Código | Significado | Evidencia |
|---|---|---|
| `7` | Dentro de los cuadros de saque | `Instructions!A172:A173` |
| `8` | Tras la línea de saque, más cerca de ella que de la línea de fondo | `Instructions!A172:A173` |
| `9` | Más cerca de la línea de fondo que de la línea de saque | `Instructions!A172:A173` |
| `0` | Profundidad desconocida | `Instructions!A179` |

**Fuente oficial:** la profundidad se añade al resto después del tipo y la
dirección. Es opcional y también puede omitirse (`Instructions!A175:A180`).

### 4.5. Fallos de saque y errores

| Código | Significado contextual | Evidencia |
|---|---|---|
| `n` | Red | `Instructions!A50:A51`, `A156` |
| `w` | Ancho | `Instructions!A50:A52`, `A156` |
| `d` | Largo | `Instructions!A50:A53`, `A156` |
| `x` | Ancho y largo | `Instructions!A50:A54`, `A156` |
| `g` | Falta de pie en el saque | `Instructions!A55` |
| `e` | Tipo de fallo o error desconocido | `Instructions!A56`, `A156` |
| `!` | Saque o golpe enmarcado | `Instructions!A57`, `A156` |

`g` se define para el saque y no aparece en la lista de errores de rally de
`Instructions!A156`.

### 4.6. Terminadores

| Código | Significado contextual | Evidencia |
|---|---|---|
| `*` | Ace tras el saque; ganador tras un golpe | `Instructions!A78`, `A152:A153` |
| `#` | Saque no retornable; error forzado tras un golpe | `Instructions!A79:A80`, `A157:A163` |
| `@` | Error no forzado | `Instructions!A81`, `A157:A162` |
| `C` | Punto detenido por un challenge o revisión de marca incorrectos | `Instructions!A213:A216` |

### 4.7. Modificadores

| Código | Significado contextual | Evidencia |
|---|---|---|
| `c` | Let de saque; puede repetirse | `Instructions!A61` |
| `+` | Intento de saque-red tras un saque; aproximación tras otro golpe | `Instructions!A70:A72`, `A187:A190` |
| `-` | Golpe normalmente de fondo ejecutado cerca de la red | `Instructions!A192:A194` |
| `=` | Golpe normalmente de red ejecutado desde el fondo | `Instructions!A192:A194` |
| `;` | Golpe que toca la cinta | `Instructions!A196` |
| `^` | Stop volley o drop volley | `Instructions!A198` |

### 4.8. Códigos especiales de punto

| Código | Significado contextual | Evidencia |
|---|---|---|
| `V` | El servidor pierde el primer saque por violación de tiempo | `Instructions!A59` |
| `S` | Punto no observado concedido al servidor | `Instructions!A204` |
| `R` | Punto no observado concedido al restador | `Instructions!A204` |
| `P` | Penalización de punto contra el servidor | `Instructions!A207:A208` |
| `Q` | Penalización de punto contra el restador | `Instructions!A207:A208` |
| `C` | Challenge incorrecto que detiene el rally | `Instructions!A213:A216` |

## 5. Reglas humanas de composición

**Fuente oficial:** cada golpe, incluido el saque, se codifica
(`Instructions!A28`). Los números expresan principalmente dirección o
profundidad; las letras, tipos de golpe o error; ciertos símbolos añaden
modificadores o desenlaces (`Instructions!A29:A30`).

La documentación humana respalda únicamente este esquema conceptual:

1. La celda comienza con el saque correspondiente, precedido opcionalmente por
   uno o varios lets `c`.
2. El saque emplea una dirección `4`, `5`, `6` o `0` cuando se conoce o se
   registra.
3. Un primer saque fallado termina en su descripción de fallo y el punto
   continúa en `second_serve`.
4. Tras el saque que entra se concatena la sucesión de golpes del rally.
5. Cada golpe indica su tipo; la dirección es opcional.
6. El primer golpe tras el saque puede añadir profundidad de resto, también
   opcional.
7. Los símbolos de ganador, error forzado, error no forzado o challenge cierran
   los ejemplos correspondientes.
8. Un error no forzado exige tipo de golpe, tipo de error y `@`; la dirección
   es opcional (`Instructions!A159:A162`).
9. Un error forzado exige como mínimo tipo de golpe y `#`; puede incorporar más
   información (`Instructions!A163`).

No se adopta aquí una precedencia formal ni se deducen combinaciones válidas a
partir de estas descripciones.

## 6. Selección de ejemplos oficiales

| Ejemplo | Interpretación ofrecida por la fuente | Procedencia |
|---|---|---|
| `6f27b1*` | Primer saque dentro y rally completo en `first_serve` | Imagen; `Instructions!A35` |
| `5d` / `4s39b3b1w@` | Primer saque fallado; segundo saque y rally en `second_serve` | Imagen; `Instructions!A36` |
| `5*` | Ace al cuerpo | `Instructions!A78` |
| `6f#` | Error forzado del resto | `Instructions!A80` |
| `6f2d@` | Error no forzado del resto | `Instructions!A81` |
| `f3*` | Ganador de derecha | `Instructions!A153` |
| `f37` | Resto de derecha con dirección 3 y profundidad 7 | `Instructions!A177` |
| `b+2` | Aproximación de revés al centro | `Instructions!A187` |
| `4+b27v1*` | Intento de saque-red terminado con volea ganadora | `Instructions!A189:A190` |
| `6b29C` | Punto detenido por challenge incorrecto | `Instructions!A216` |

La imagen de `Instructions`, anclada aproximadamente en A39:A43, representa
visualmente los dos primeros ejemplos. No se versiona separadamente.

## 7. Casos especiales y valores desconocidos

- **Fuente oficial:** `V` registra la pérdida del primer saque por violación de
  tiempo; `P` y `Q`, penalizaciones; `S` y `R`, puntos no observados; y `C`, un
  challenge incorrecto.
- **Fuente oficial:** `0` representa dirección o profundidad desconocida según
  contexto; `q`, tipo de golpe desconocido; y `e`, tipo de fallo o error
  desconocido (`Instructions!A48`, `A56`, `A123`, `A138`, `A179`, `A240:A241`).
- **Fuente oficial:** `Notes` sirve como campo libre para circunstancias que no
  encajan en la secuencia, como interrupciones, atención médica, lluvia,
  coaching o advertencias (`Instructions!A218:A222`).
- **Interpretación del proyecto:** `S` y `V` colisionan con códigos externos a
  las secuencias usados para formatos de partido en `Instructions!A9:A16`.
  Deben separarse por campo y contexto.

## 8. Ambigüedades e inconsistencias pendientes

Las veinte cuestiones siguientes permanecen abiertas:

1. **Decisión pendiente:** no existe una tokenización formal ni delimitadores
   explícitos entre golpes.
2. **Decisión pendiente:** no se define la precedencia entre modificadores,
   dirección, profundidad, error y terminador.
3. **Decisión pendiente:** no se enumeran exhaustivamente las combinaciones
   permitidas o prohibidas.
4. **Decisión pendiente:** no se establece si varios modificadores pueden
   coexistir ni su orden cuando coinciden.
5. **Fuente oficial:** `*` significa ace o ganador según su posición.
6. **Fuente oficial:** `#` significa saque no retornable o error
   forzado según su posición.
7. **Fuente oficial:** `n`, `w`, `d`, `x`, `e` y `!` pueden describir
   un fallo de saque o un error de golpe.
8. **Fuente oficial:** `0` puede significar dirección desconocida o
   profundidad desconocida.
9. **Fuente oficial:** `+` significa saque-red tras un saque o
   aproximación tras otro golpe.
10. **Interpretación del proyecto:** `S` y `V` también son códigos de formato de
    partido fuera de las secuencias.
11. **Decisión pendiente:** no se formaliza cómo identificar y tokenizar el
    primer golpe del rally como resto.
12. **Decisión pendiente:** la profundidad solo corresponde al resto, pero no
    se especifica cómo resolver secuencias numéricamente ambiguas.
13. **Decisión pendiente:** se permite repetir `c`, pero no se fija un máximo ni
    se enumeran sus combinaciones con otras marcas de saque.
14. **Decisión pendiente:** no se describe de forma completa la sintaxis de una
    doble falta.
15. **Interpretación del proyecto:** la fuente deja ambiguo que la dirección del
    golpe sea opcional mientras recomienda la dirección de los fallos de saque,
    sin aclarar siempre cuándo es un requisito normativo.
16. **Decisión pendiente:** `C` está respaldado por un ejemplo, pero no se
    documentan todas sus posibles combinaciones con otros terminadores.
17. **Interpretación del proyecto:** existe una inconsistencia en
    `Instructions!A71`: presenta `4+w` como un saque abierto que termina largo,
    aunque `d` se define como largo y `w` como ancho.
18. **Interpretación del proyecto:** existe una inconsistencia porque
    `Version info!B9` denomina `;` de forma que parece referirse a un let,
    mientras `Instructions!A196` lo define como contacto de un golpe con la
    cinta.
19. **Decisión pendiente:** la fuente mezcla recomendaciones prácticas con
    reglas normativas sin marcarlas siempre de forma diferenciada.
20. **Decisión pendiente:** no hay una regla general de mayúsculas y minúsculas;
    solo se observan usos concretos en el vocabulario y los ejemplos.

## 9. Licencia y cesión

**Fuente oficial:** `License_Assignment!B1:B5` declara que la plantilla se
ofrece bajo CC BY-NC-SA 4.0 dentro de una licencia limitada vinculada a su uso
para contribuir a Tennis Abstract y al Match Charting Project.

**Fuente oficial:** `License_Assignment!B7:B15` contiene además una cesión
amplia de los derechos que el contribuyente pudiera tener sobre las
aportaciones enviadas mediante la plantilla, junto con facultades de uso,
reproducción, distribución y adaptación por Jeff Sackmann/Tennis Abstract.

**Interpretación del proyecto:** este texto no permite concluir por sí solo que
todo el repositorio o todos los datasets publicados estén cubiertos por esa
misma licencia. Este apartado es un resumen informativo de la hoja, no
asesoramiento jurídico.

## 10. Elementos auxiliares del workbook

- `Version info!A1:C33` documenta la evolución hasta 0.3.2 y confirma la
  incorporación histórica de varios códigos, pero no define otra gramática.
- La hoja oculta `Tables` contiene principalmente transiciones de marcadores y
  nombres de sets; no contiene un diccionario alternativo de golpes.
- El único nombre definido es `SetNames = Tables!$N$1:$N$6`.
- La única validación de datos encontrada es `MatchStats!C1`, una lista basada
  en `SetNames` que permite valores vacíos.
- `Instructions` contiene una imagen PNG de 427 × 81 píxeles con los ejemplos
  `6f27b1*` y `5d` / `4s39b3b1w@`.
- `Instructions!A168` contiene el texto
  `5f2f1f1v2n@ = body serve, followed by the rally described above.`. Excel lo
  convirtió automáticamente en un falso enlace `mailto:` debido a `@`; no es
  una referencia intencionada.

## 11. Decisiones necesarias antes del parser

Antes de implementar un parser deberán resolverse o aislarse explícitamente:

- el modelo de tokenización y la precedencia;
- el tratamiento de combinaciones no documentadas;
- la sintaxis de dobles faltas;
- las colisiones y caracteres contextuales;
- la separación entre secuencias válidas, especiales, incompletas y ambiguas;
- el tratamiento de inconsistencias entre reglas y ejemplos;
- qué recomendaciones del anotador deben convertirse, si procede, en reglas de
  validación.

**Decisión pendiente:** una alternativa futura es un parser tolerante que
reconozca fragmentos respaldados y conserve el resto; otra es un parser estricto
que rechace toda secuencia fuera de una gramática aprobada. No se adopta todavía
ninguna de las dos.

**Interpretación del proyecto:** cualquiera que sea la alternativa elegida, la
primera implementación deberá conservar al menos:

1. la secuencia original sin modificar;
2. los tokens reconocidos;
3. el residuo no interpretado;
4. las advertencias y decisiones aplicadas.

Esto permitirá auditar el parser y evitar que una interpretación parcial se
presente como información completa.
