"""P22: adaptador productivo puro -- DataFrame YA CARGADO -> estructuras P21.

Este modulo NO hace I/O: recibe un ``pandas.DataFrame`` como argumento
(el llamador ya lo leyo, real o sintetico) y lo transforma en
``development_observations`` / ``test_observations`` (P21) y en
``P02ObservedAttempt`` (labels reales P02, para desarrollo y para
test). Nunca lee Parquet/CSV, nunca abre el snapshot y nunca ejecuta
P10 (``run_tactical_recommender_pipeline`` /
``compute_tactical_pipeline_result``).

Auditoria previa (evidencia exacta, no supuesta) y correccion tras
revision:

- Esquema real minimo: ``tactical_recommender_pipeline.SOURCE_COLUMNS``
  = ``match_id, point_number, date, surface, server, point_winner,
  player_1, player_2, first_serve, second_serve``.
  ``validate_source_points`` (publica) se reutiliza sin cambios.
- ``seal_development_points`` (publica) separa incondicionalmente
  ``development = source[date<=2023-12-31]`` /
  ``sealed = source[date>=2024-01-01]``: se reutiliza sin cambios.
- CORREGIDO (bloqueo de revision): la version anterior de este modulo
  MIRRORS-eaba ``_point_seeds`` y el cuerpo de
  ``construct_tactical_attempt_records`` para el periodo de test --
  duplicacion no aceptada. Ahora ``tactical_recommender_pipeline.py``
  expone ``construct_tactical_attempt_records_for_test_partition``
  (nueva, publica, simetrica), que comparte la MISMA implementacion
  interna (``_convert_points_to_attempt_records``, privada) con
  ``construct_tactical_attempt_records`` (desarrollo). Cada wrapper
  impone su PROPIO rango de fechas cerrado antes de delegar: ninguno
  de los dos, por separado, permite procesar la particion del otro; no
  existe una tercera funcion generica que acepte ambos rangos. Este
  adaptador ya NO define ninguna copia de ``_point_seeds``,
  ``_sequence_presence`` ni ``_documents_service_fault``: son
  exclusivamente responsabilidad de P10.
- Labels P02: derivados reutilizando sin cambios
  ``src.analysis.first_serve_direction_classification.
  parse_and_classify_first_serve_direction`` (unico precedente real),
  tanto para desarrollo (necesario para la baseline poblacional P02,
  decision humana congelada -- ver ``final_sealed_evaluation_
  orchestrator.py``) como para test. Sin equivalente para P04/P05/P06:
  este adaptador NO fabrica esos labels.
- Cardinalidades exactas: ``chronological_validation.EXPECTED_SPLIT_
  MATCHES`` (train=4188, validation=1805, test=1531, autoridad
  independiente); orientaciones test = 1531*2 = 3062
  (``final_sealed_evaluation.EXPECTED_TEST_ORIENTATIONS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from src.analysis.chronological_validation import (
    EXPECTED_SPLIT_MATCHES,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
)
from src.analysis.final_sealed_evaluation import EXPECTED_TEST_ORIENTATIONS
from src.analysis.final_sealed_evaluation_boundary import SealedTestMatchRow
from src.analysis.first_serve_direction_classification import (
    parse_and_classify_first_serve_direction,
)
from src.analysis.tactical_recommender_pipeline import (
    DEVELOPMENT_CUTOFF,
    PRODUCTION_ENCODER_POLICY,
    TacticalAttemptRecord,
    TacticalPipelineContractError,
    build_tactical_historical_observations,
    construct_tactical_attempt_records,
    construct_tactical_attempt_records_for_test_partition,
    seal_development_points,
    validate_source_points,
)
from src.recommender.tactical_feature_encoder import (
    TacticalFeatureSchema,
    build_tactical_feature_schema,
)
from src.recommender.tactical_history_profiles import TacticalHistoricalObservation


FINAL_SEALED_EVALUATION_ADAPTER_CONTRACT_VERSION: Final = "1.0.0"

EXPECTED_TRAIN_MATCHES: Final = EXPECTED_SPLIT_MATCHES["train"]
EXPECTED_VALIDATION_MATCHES: Final = EXPECTED_SPLIT_MATCHES["validation"]
EXPECTED_TEST_MATCHES: Final = EXPECTED_SPLIT_MATCHES["test"]


class FinalSealedEvaluationAdapterError(RuntimeError):
    """Rechazo cerrado del adaptador P22 (mensaje del catalogo, sin fugas)."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class P02ObservedAttempt:
    """Label real observado P02 (unico patron con precedente de scoring).

    ``category`` es la DIRECCION observada del primer saque (via
    ``parse_and_classify_first_serve_direction``); ``server_won_point``
    es el OUTCOME real del punto, un campo distinto e independiente
    del ``TacticalAttemptRecord`` de origen -- nunca se derivan el uno
    del otro. No incluye ninguna puntuacion predicha: eso pertenece al
    orquestador (P22, paso 3), que la combina con el resultado de
    evaluacion por target."""

    target_match_id: str
    player: str
    category: str
    server_won_point: bool


@dataclass(frozen=True, slots=True)
class AdaptedSealedEvaluationInput:
    contract_version: str
    development_observations: tuple[TacticalHistoricalObservation, ...]
    test_observations: tuple[TacticalHistoricalObservation, ...]
    test_match_rows: tuple[SealedTestMatchRow, ...]
    test_p02_observed_attempts: tuple[P02ObservedAttempt, ...]
    development_p02_observed_attempts: tuple[P02ObservedAttempt, ...]
    schema: TacticalFeatureSchema
    train_matches: int
    validation_matches: int
    test_matches: int
    test_orientations: int


def _fail(message: str) -> FinalSealedEvaluationAdapterError:
    return FinalSealedEvaluationAdapterError(message)


def _validate_three_way_split(development: pd.DataFrame) -> tuple[int, int]:
    """Reconcilia train/validation DENTRO de development (ya sellado test)."""
    if bool(development.date.gt(DEVELOPMENT_CUTOFF).any()):
        raise _fail("development contiene fechas de test tras el sellado.")
    train_rows = development.loc[development.date.le(TRAIN_END)]
    validation_rows = development.loc[
        development.date.between(VALIDATION_START, VALIDATION_END)
    ]
    if len(train_rows) + len(validation_rows) != len(development):
        raise _fail("development contiene fechas fuera de train/validation.")
    train_matches = int(train_rows.match_id.nunique())
    validation_matches = int(validation_rows.match_id.nunique())
    if train_matches != EXPECTED_TRAIN_MATCHES:
        raise _fail("Cardinalidad de partidos de train fuera de contrato.")
    if validation_matches != EXPECTED_VALIDATION_MATCHES:
        raise _fail("Cardinalidad de partidos de validacion fuera de contrato.")
    return train_matches, validation_matches


def _derive_test_match_rows(sealed: pd.DataFrame) -> tuple[SealedTestMatchRow, ...]:
    matches = (
        sealed.loc[:, ["match_id", "date", "player_1", "player_2", "surface"]]
        .drop_duplicates("match_id", keep="first")
        .sort_values(["date", "match_id"], kind="stable")
    )
    rows = tuple(
        SealedTestMatchRow(
            item.match_id,
            item.date.date() if hasattr(item.date, "date") else item.date,
            item.player_1,
            item.player_2,
            item.surface,
        )
        for item in matches.itertuples(index=False)
    )
    if len(rows) != EXPECTED_TEST_MATCHES:
        raise _fail("Cardinalidad de partidos de test fuera de contrato.")
    return rows


def _derive_p02_observed_attempts(
    records: tuple[TacticalAttemptRecord, ...],
) -> tuple[P02ObservedAttempt, ...]:
    """Reutiliza integra la clasificacion P02 ya demostrada; sin label
    cuando la direccion no es elegible (nunca se fabrica un valor).
    ``category`` (direccion) y ``server_won_point`` (outcome) se leen
    de fuentes independientes del mismo ``record`` -- la clasificacion
    NUNCA determina el outcome, y el outcome NUNCA determina la
    categoria: ver pruebas de no-intercambio en el orquestador."""
    observed: list[P02ObservedAttempt] = []
    for record in records:
        if record.serve_number != 1:
            continue
        classification = parse_and_classify_first_serve_direction(
            record.sequence_text, record.serve_number
        )
        if not classification.eligible_for_direction_signal:
            continue
        if classification.direction_description is None:
            continue
        observed.append(
            P02ObservedAttempt(
                record.match_id,
                record.server_player,
                classification.direction_description,
                record.server_won_point,
            )
        )
    return tuple(observed)


def adapt_real_points_dataframe(
    points: pd.DataFrame,
    *,
    schema: TacticalFeatureSchema | None = None,
) -> AdaptedSealedEvaluationInput:
    """Unica frontera productiva pura: DataFrame ya cargado -> P21.

    No hace I/O: ``points`` debe ser un DataFrame que el llamador ya
    leyo (real o sintetico). No muta ``points``: todas las particiones
    se derivan con ``.copy(deep=True)`` internamente (``seal_development
    _points`` ya lo garantiza; este modulo no reutiliza sus salidas por
    referencia mutable en ningun punto)."""
    if not isinstance(points, pd.DataFrame):
        raise _fail("points debe ser pandas.DataFrame real.")
    selected_schema = (
        build_tactical_feature_schema(PRODUCTION_ENCODER_POLICY) if schema is None else schema
    )
    original_shape = points.shape
    try:
        validated = validate_source_points(points)
    except TacticalPipelineContractError as error:
        raise _fail(str(error)) from None
    if points.shape != original_shape:
        raise _fail("points fue mutado durante la validacion.")
    development, _population, _seal = seal_development_points(validated)
    sealed = validated.loc[validated.date.ge(TEST_START)].copy(deep=True)
    if bool(sealed.date.gt(TEST_END).any()):
        raise _fail("La fuente contiene fechas posteriores al test congelado.")
    train_matches, validation_matches = _validate_three_way_split(development)

    try:
        dev_batch = construct_tactical_attempt_records(development, selected_schema)
    except TacticalPipelineContractError as error:
        raise _fail(str(error)) from None
    development_observations = build_tactical_historical_observations(dev_batch.records)
    development_p02_observed = _derive_p02_observed_attempts(dev_batch.records)

    try:
        test_batch = construct_tactical_attempt_records_for_test_partition(
            sealed, selected_schema
        )
    except TacticalPipelineContractError as error:
        raise _fail(str(error)) from None
    test_observations = build_tactical_historical_observations(test_batch.records)
    test_match_rows = _derive_test_match_rows(sealed)
    test_p02_observed = _derive_p02_observed_attempts(test_batch.records)

    test_orientations = len(test_match_rows) * 2
    if test_orientations != EXPECTED_TEST_ORIENTATIONS:
        raise _fail("Orientaciones de test fuera del contrato congelado P20.")

    return AdaptedSealedEvaluationInput(
        FINAL_SEALED_EVALUATION_ADAPTER_CONTRACT_VERSION,
        development_observations,
        test_observations,
        test_match_rows,
        test_p02_observed,
        development_p02_observed,
        selected_schema,
        train_matches,
        validation_matches,
        len(test_match_rows),
        test_orientations,
    )


__all__ = (
    "EXPECTED_TEST_MATCHES",
    "EXPECTED_TRAIN_MATCHES",
    "EXPECTED_VALIDATION_MATCHES",
    "FINAL_SEALED_EVALUATION_ADAPTER_CONTRACT_VERSION",
    "AdaptedSealedEvaluationInput",
    "FinalSealedEvaluationAdapterError",
    "P02ObservedAttempt",
    "adapt_real_points_dataframe",
)
