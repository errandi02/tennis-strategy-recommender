"""P21: frontera segura y evaluador compute-only del test sellado
2024-01-01..2026-05-21, sobre datos EXPLICITOS o sintéticos unicamente.

Este modulo NO lee Parquet/CSV, NO abre el snapshot privado, NO
ejecuta P10 (``run_tactical_recommender_pipeline`` /
``compute_tactical_pipeline_result``) y NO consulta variables de
entorno. Todas sus funciones son puras: reciben observaciones y
partidos ya construidos (reales o sinteticos) por el llamador y
devuelven estructuras deterministas. La UNICA puerta de autorizacion
del proyecto sigue viviendo en
``src.analysis.final_sealed_evaluation.REAL_TEST_EVALUATION_AUTHORIZED``
/ ``run_real_test_evaluation``; este modulo no define una segunda
puerta porque no tiene ninguna capacidad de I/O que autorizar.

Auditoria previa contra la fuente (evidencia de codigo, no asumida):

- ``build_validation_targets`` (``tactical_recommender_pipeline.py``)
  construye 2 orientaciones por partido de validacion y usa
  ``TacticalPipelineTarget``, cuyo ``validate_tactical_pipeline_target``
  EXIGE ``VALIDATION_START.date() <= as_of_date <= DEVELOPMENT_CUTOFF.
  date()`` (2020-01-01..2023-12-31): esa clase rechaza estructuralmente
  cualquier fecha de test y por tanto NO es reutilizable para P21 sin
  modificar P10. Por eso P21 define su propio ``SealedTestTarget``
  (misma forma canonica, rango propio 2024-2026), sin tocar
  ``tactical_recommender_pipeline.py``.
- El test se excluye actualmente en
  ``seal_development_points`` (misma fuente): separa
  incondicionalmente ``development = source[date <= 2023-12-31]`` de
  ``sealed = source[date >= 2024-01-01]``; ``sealed`` nunca llega a
  ``construct_tactical_attempt_records`` ni a la construccion de
  observaciones. ``compute_tactical_pipeline_result`` tampoco acepta un
  parametro ``targets`` (siempre delega en la reconstruccion de
  validacion). Por eso P21 NO reutiliza esa frontera de archivo: opera
  directamente sobre los mismos primitivos PUBLICOS de nivel inferior
  que ya usa en produccion el servicio P12 en vivo
  (``build_tactical_matchup_evidence`` + ``prioritize_tactical_matchup``,
  en ``src/recommender/``), sin necesitar ningun cambio en
  ``src/analysis/tactical_recommender_pipeline.py``.
- ``build_tactical_matchup_evidence`` (``tactical_matchup_evidence.py``,
  linea ~1066) ya filtra
  ``excluded_date = observations[effective_date >= query.as_of_date]``
  / ``before_cut = observations[effective_date < query.as_of_date]``:
  la regla ``strictly_before_as_of_date`` (misma constante contractual,
  linea ~72) es la fuente autoritativa de ``calendar_day_rule`` --
  ningun punto del mismo dia o posterior puede influir. P21 aplica la
  MISMA regla de forma explicita y aislada en
  ``visible_observations_for_target`` para poder demostrarla con tests
  puros, ademas de heredarla implicitamente de
  ``build_tactical_matchup_evidence``.
- Labels de scoring P02: el unico precedente de codigo es
  ``explainable_direction_scoring.py`` (``server_won_point`` renombrado
  ``outcome``, ``EPSILON = 1e-15`` para el log-loss). No existe
  equivalente para P04/P05/P06: P21 no inventa uno.
- Metricas y denominadores congelados: ``final_sealed_evaluation.py``
  (P20) -- ``PRIMARY_METRICS_ALL_PATTERNS``, ``PRIMARY_METRICS_P02_ONLY``,
  ``METRIC_DENOMINATOR_RULES``. P21 los importa sin modificarlos.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, fields
from datetime import date
from types import MappingProxyType
from typing import Final

from src.analysis.chronological_validation import ALLOWED_SURFACES
from src.analysis.final_sealed_evaluation import (
    MINIMUM_DISTINCT_MATCHES,
    MINIMUM_LABELED_ACTIVATIONS,
    PATTERNS,
    REAL_TEST_EVALUATION_AUTHORIZED,
    REAL_TEST_EVALUATION_BLOCK_REASON,
    REQUESTED_TOP_K,
    TEST_END_DATE,
    TEST_START_DATE,
    run_real_test_evaluation,
)
from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    TacticalFeatureSchema,
    feature_schema_fingerprint,
)
from src.recommender.tactical_history_profiles import TacticalHistoricalObservation
from src.recommender.tactical_matchup_evidence import (
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupContractError,
    TacticalMatchupEvidence,
    TacticalMatchupQuery,
    build_tactical_matchup_evidence,
)
from src.recommender.tactical_prioritization import (
    TacticalCandidateState,
    TacticalPrioritizationContractError,
    TacticalPrioritizationResult,
    TacticalPrioritizationState,
    prioritize_tactical_matchup,
)


FINAL_SEALED_EVALUATION_BOUNDARY_CONTRACT_VERSION: Final = "1.0.0"

FROZEN_HISTORY_CUTOFF: Final = date(2023, 12, 31)
TEST_RANGE_START: Final = date.fromisoformat(TEST_START_DATE)
TEST_RANGE_END: Final = date.fromisoformat(TEST_END_DATE)

ROLLING_PROTOCOL: Final = "rolling_origin"
FROZEN_PROTOCOL: Final = "frozen"
PROTOCOLS: Final = (ROLLING_PROTOCOL, FROZEN_PROTOCOL)
ORIENTATIONS: Final = ("player_1_vs_player_2", "player_2_vs_player_1")

_LOG_LOSS_EPSILON: Final = 1e-15


class FinalSealedEvaluationBoundaryError(RuntimeError):
    """Rechazo cerrado de la frontera P21 (mensaje del catalogo, sin fugas)."""

    __slots__ = ()


# --------------------------------------------------------------------- #
# 1. Targets de test: API simetrica y cerrada, no la de validacion       #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SealedTestMatchRow:
    """Fila minima de un partido de test YA separado por el llamador."""

    match_id: str
    date: date
    player_1: str
    player_2: str
    surface: str

    def __post_init__(self) -> None:
        _validate_match_row(self)


def _validate_match_row(row: SealedTestMatchRow) -> None:
    for field_name in ("match_id", "player_1", "player_2", "surface"):
        value = getattr(row, field_name)
        if type(value) is not str or not value or value != value.strip():
            raise FinalSealedEvaluationBoundaryError(
                "Fila de partido de test fuera de contrato textual."
            )
    if type(row.date) is not date:
        raise FinalSealedEvaluationBoundaryError(
            "Fecha de partido de test debe ser date real."
        )
    if row.player_1 == row.player_2:
        raise FinalSealedEvaluationBoundaryError(
            "player_1 y player_2 deben ser distintos."
        )
    if row.surface not in ALLOWED_SURFACES:
        raise FinalSealedEvaluationBoundaryError(
            "Superficie de partido de test fuera de contrato."
        )


@dataclass(frozen=True, slots=True)
class SealedTestTarget:
    """Orientacion target del test (P21), rango propio 2024-2026.

    Deliberadamente NO es ``TacticalPipelineTarget``: esa clase valida
    ``VALIDATION_START..DEVELOPMENT_CUTOFF`` en su ``__post_init__`` y
    rechazaria cualquier fecha de test. Generalizarla debilitaria el
    contrato de validacion existente; esta clase separada lo evita.
    """

    target_match_id: str
    player: str
    opponent: str
    as_of_date: date
    orientation: str
    surface: str

    def __post_init__(self) -> None:
        _validate_sealed_test_target(self)


def _validate_sealed_test_target(target: SealedTestTarget) -> None:
    for field_name in ("target_match_id", "player", "opponent", "surface"):
        value = getattr(target, field_name)
        if type(value) is not str or not value or value != value.strip():
            raise FinalSealedEvaluationBoundaryError(
                "Target de test fuera de contrato textual."
            )
    if type(target.as_of_date) is not date:
        raise FinalSealedEvaluationBoundaryError(
            "as_of_date de target de test debe ser date real."
        )
    if not (TEST_RANGE_START <= target.as_of_date <= TEST_RANGE_END):
        raise FinalSealedEvaluationBoundaryError(
            "as_of_date de target de test fuera del rango sellado."
        )
    if target.player == target.opponent:
        raise FinalSealedEvaluationBoundaryError(
            "player y opponent deben ser distintos."
        )
    if target.orientation not in ORIENTATIONS:
        raise FinalSealedEvaluationBoundaryError(
            "orientation de target de test fuera de contrato."
        )
    if target.surface not in ALLOWED_SURFACES:
        raise FinalSealedEvaluationBoundaryError(
            "Superficie de target de test fuera de contrato."
        )


def build_test_targets(
    matches: tuple[SealedTestMatchRow, ...],
    *,
    enforce_frozen_cardinalities: bool = False,
) -> tuple[SealedTestTarget, ...]:
    """Construye exactamente dos orientaciones por partido de test.

    Acepta EXCLUSIVAMENTE una particion de test ya separada por el
    llamador (nunca deriva la particion desde una fuente mixta: no hay
    I/O ni acceso a train/validation aqui). Rechaza cualquier fila
    fuera de ``[2024-01-01, 2026-05-21]``, duplicados de ``match_id`` y
    orden no determinista; el resultado esta ordenado por
    ``(date, match_id, orientation)`` para reproducibilidad exacta
    independientemente del orden de entrada.
    """
    if type(matches) is not tuple or not matches:
        raise FinalSealedEvaluationBoundaryError(
            "matches debe ser tuple no vacia de SealedTestMatchRow."
        )
    for row in matches:
        if type(row) is not SealedTestMatchRow:
            raise FinalSealedEvaluationBoundaryError(
                "Cada fila debe ser SealedTestMatchRow exacto."
            )
        if not (TEST_RANGE_START <= row.date <= TEST_RANGE_END):
            raise FinalSealedEvaluationBoundaryError(
                "Fila fuera del rango sellado del test (train/validation/futuro)."
            )
    match_ids = tuple(row.match_id for row in matches)
    if len(set(match_ids)) != len(match_ids):
        raise FinalSealedEvaluationBoundaryError("match_id duplicado en matches.")
    ordered_rows = tuple(sorted(matches, key=lambda row: (row.date, row.match_id)))
    targets: list[SealedTestTarget] = []
    for row in ordered_rows:
        targets.append(
            SealedTestTarget(
                row.match_id, row.player_1, row.player_2, row.date,
                "player_1_vs_player_2", row.surface,
            )
        )
        targets.append(
            SealedTestTarget(
                row.match_id, row.player_2, row.player_1, row.date,
                "player_2_vs_player_1", row.surface,
            )
        )
    result = tuple(
        sorted(targets, key=lambda item: (item.as_of_date, item.target_match_id, item.orientation))
    )
    keys = tuple((item.target_match_id, item.orientation) for item in result)
    if len(keys) != len(set(keys)) or len(result) != 2 * len(matches):
        raise FinalSealedEvaluationBoundaryError(
            "Orientaciones target de test no son exhaustivas y unicas."
        )
    if enforce_frozen_cardinalities:
        from src.analysis.final_sealed_evaluation import (
            EXPECTED_TEST_MATCHES,
            EXPECTED_TEST_ORIENTATIONS,
        )

        if len(matches) != EXPECTED_TEST_MATCHES or len(result) != EXPECTED_TEST_ORIENTATIONS:
            raise FinalSealedEvaluationBoundaryError(
                "Cardinalidades del test congeladas (P20) no reconcilian."
            )
    return result


# --------------------------------------------------------------------- #
# 2. Visibilidad temporal (calendar_day_rule / frozen) -- funciones puras #
# --------------------------------------------------------------------- #


def visible_observations_for_target(
    target: SealedTestTarget,
    observations: tuple[TacticalHistoricalObservation, ...],
    *,
    protocol: str,
) -> tuple[TacticalHistoricalObservation, ...]:
    """Regla temporal explicita y aislada (probable por si sola en tests).

    ``frozen``: solo observaciones con ``effective_date <=
    2023-12-31``, sin importar la fecha real del target (nunca se
    actualiza con nada del periodo de test).
    ``rolling_origin``: solo observaciones con ``effective_date <
    target.as_of_date`` (misma regla ``strictly_before_as_of_date`` que
    ``build_tactical_matchup_evidence`` aplica internamente): ningun
    partido del mismo dia o posterior puede influir.
    """
    if protocol not in PROTOCOLS:
        raise FinalSealedEvaluationBoundaryError("Protocolo de test fuera de contrato.")
    if protocol == FROZEN_PROTOCOL:
        return tuple(
            item for item in observations if item.effective_date <= FROZEN_HISTORY_CUTOFF
        )
    return tuple(
        item for item in observations if item.effective_date < target.as_of_date
    )


# --------------------------------------------------------------------- #
# 3. Evaluador compute-only por target y por poblacion                   #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SealedTestTargetEvaluation:
    target: SealedTestTarget
    protocol: str
    evidence: TacticalMatchupEvidence
    prioritization: TacticalPrioritizationResult
    visible_observation_count: int
    visible_test_observation_count: int
    visible_max_effective_date: date | None


@dataclass(frozen=True, slots=True)
class SealedTestTargetEvaluationError:
    """Resultado de error cerrado: cuenta aparte, fuera de denominadores
    de performance (``METRIC_DENOMINATOR_RULES
    ["incompatible_or_upstream_error_results"]``)."""

    target: SealedTestTarget
    protocol: str
    reason_code: str


def _build_query(
    target: SealedTestTarget, schema: TacticalFeatureSchema
) -> TacticalMatchupQuery:
    return TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        target.player,
        target.opponent,
        target.as_of_date,
        TacticalEncodingPolicy.COMPONENT_ONLY,
        feature_schema_fingerprint(schema),
        (1, 2),
        None,
        MINIMUM_LABELED_ACTIVATIONS,
        MINIMUM_DISTINCT_MATCHES,
        PATTERNS,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )


def evaluate_single_test_target(
    target: SealedTestTarget,
    observations: tuple[TacticalHistoricalObservation, ...],
    schema: TacticalFeatureSchema,
    *,
    protocol: str,
    test_match_ids: frozenset[str] = frozenset(),
) -> SealedTestTargetEvaluation | SealedTestTargetEvaluationError:
    """Un target -> evidencia + priorizacion, reutilizando SOLO los
    primitivos publicos de P11-B (sin tocar P10)."""
    try:
        visible = visible_observations_for_target(target, observations, protocol=protocol)
        query = _build_query(target, schema)
        evidence = build_tactical_matchup_evidence(visible, query, schema)
        prioritization = prioritize_tactical_matchup(evidence, top_k=REQUESTED_TOP_K)
    except (TacticalMatchupContractError, TacticalPrioritizationContractError):
        return SealedTestTargetEvaluationError(
            target, protocol, "upstream_contract_violation"
        )
    visible_test_count = sum(1 for item in visible if item.match_id in test_match_ids)
    max_date = max((item.effective_date for item in visible), default=None)
    return SealedTestTargetEvaluation(
        target, protocol, evidence, prioritization,
        len(visible), visible_test_count, max_date,
    )


@dataclass(frozen=True, slots=True)
class SealedTestEvaluationResult:
    contract_version: str
    protocol: str
    targets_total: int
    evaluations: tuple[SealedTestTargetEvaluation | SealedTestTargetEvaluationError, ...]


def _validate_target_population(targets: tuple[SealedTestTarget, ...]) -> None:
    if type(targets) is not tuple or not targets:
        raise FinalSealedEvaluationBoundaryError("targets debe ser tuple no vacia.")
    keys = tuple((item.target_match_id, item.orientation) for item in targets)
    if len(keys) != len(set(keys)):
        raise FinalSealedEvaluationBoundaryError("Target de test duplicado.")
    by_match: dict[str, int] = Counter(item.target_match_id for item in targets)
    if any(count != 2 for count in by_match.values()):
        raise FinalSealedEvaluationBoundaryError(
            "Cada partido de test debe tener exactamente dos orientaciones."
        )


def _validate_no_temporal_leakage(
    evaluations: tuple[SealedTestTargetEvaluation | SealedTestTargetEvaluationError, ...],
    *,
    protocol: str,
) -> None:
    for item in evaluations:
        if isinstance(item, SealedTestTargetEvaluationError):
            continue
        if item.visible_max_effective_date is not None:
            if item.visible_max_effective_date >= item.target.as_of_date:
                raise FinalSealedEvaluationBoundaryError(
                    "Fuga temporal: historia visible en o despues del target."
                )
            if protocol == FROZEN_PROTOCOL and item.visible_max_effective_date > FROZEN_HISTORY_CUTOFF:
                raise FinalSealedEvaluationBoundaryError(
                    "Fuga temporal: frozen actualizo historia mas alla de 2023-12-31."
                )
        if protocol == FROZEN_PROTOCOL and item.visible_test_observation_count != 0:
            raise FinalSealedEvaluationBoundaryError(
                "Fuga temporal: frozen uso observaciones del periodo de test."
            )


def evaluate_sealed_test_population(
    targets: tuple[SealedTestTarget, ...],
    development_observations: tuple[TacticalHistoricalObservation, ...],
    test_observations: tuple[TacticalHistoricalObservation, ...],
    schema: TacticalFeatureSchema,
    *,
    protocol: str,
) -> SealedTestEvaluationResult:
    """Evaluador compute-only de poblacion completa (sin I/O).

    ``rolling_origin`` ve ``development_observations`` mas
    ``test_observations`` (la regla de fecha estricta en
    ``visible_observations_for_target`` restringe correctamente cada
    target sin necesitar actualizar nada de forma incremental).
    ``frozen`` ve EXCLUSIVAMENTE ``development_observations``: el
    parametro ``test_observations`` se recibe solo para el conteo de
    auditoria (``visible_test_observation_count``), nunca se anade al
    pool visible.
    """
    if protocol not in PROTOCOLS:
        raise FinalSealedEvaluationBoundaryError("Protocolo de test fuera de contrato.")
    _validate_target_population(targets)
    if type(development_observations) is not tuple or type(test_observations) is not tuple:
        raise FinalSealedEvaluationBoundaryError(
            "Las observaciones deben ser tuple exacto."
        )
    development_match_ids = frozenset(item.match_id for item in development_observations)
    test_match_ids = frozenset(item.match_id for item in test_observations)
    if development_match_ids & test_match_ids:
        # Revision de diseno P21: sin esta comprobacion, un llamador que
        # suministrase por error el mismo partido en ambas particiones
        # duplicaria su influencia (doble conteo silencioso) sin que
        # ninguna otra validacion lo detectara.
        raise FinalSealedEvaluationBoundaryError(
            "development_observations y test_observations comparten un match_id: "
            "particion no disjunta."
        )
    pool = (
        development_observations
        if protocol == FROZEN_PROTOCOL
        else development_observations + test_observations
    )
    evaluations = tuple(
        evaluate_single_test_target(
            target, pool, schema, protocol=protocol, test_match_ids=test_match_ids
        )
        for target in targets
    )
    if len(evaluations) != len(targets):
        raise FinalSealedEvaluationBoundaryError(
            "Perdida de targets durante la evaluacion de poblacion."
        )
    _validate_no_temporal_leakage(evaluations, protocol=protocol)
    return SealedTestEvaluationResult(
        FINAL_SEALED_EVALUATION_BOUNDARY_CONTRACT_VERSION,
        protocol,
        len(targets),
        evaluations,
    )


def assert_protocols_share_population(
    rolling_result: SealedTestEvaluationResult, frozen_result: SealedTestEvaluationResult
) -> None:
    """Prueba que rolling y frozen evaluan EXACTAMENTE la misma poblacion."""
    if rolling_result.protocol != ROLLING_PROTOCOL or frozen_result.protocol != FROZEN_PROTOCOL:
        raise FinalSealedEvaluationBoundaryError("Orden de protocolos inesperado.")
    rolling_keys = {
        (item.target.target_match_id, item.target.orientation)
        for item in rolling_result.evaluations
    }
    frozen_keys = {
        (item.target.target_match_id, item.target.orientation)
        for item in frozen_result.evaluations
    }
    if rolling_keys != frozen_keys:
        raise FinalSealedEvaluationBoundaryError(
            "rolling y frozen no comparten exactamente la misma poblacion."
        )


# --------------------------------------------------------------------- #
# 4. Metricas compute-only congeladas por P20                            #
# --------------------------------------------------------------------- #


def _successful_evaluations(
    result: SealedTestEvaluationResult,
) -> tuple[SealedTestTargetEvaluation, ...]:
    return tuple(
        item for item in result.evaluations if isinstance(item, SealedTestTargetEvaluation)
    )


def aggregate_error_bucket(result: SealedTestEvaluationResult) -> dict[str, object]:
    errors = tuple(
        item for item in result.evaluations if isinstance(item, SealedTestTargetEvaluationError)
    )
    return {
        "denominator": len(result.evaluations),
        "error_count": len(errors),
        "error_rate": len(errors) / len(result.evaluations),
        "reason_codes": dict(Counter(item.reason_code for item in errors)),
    }


def aggregate_orientation_coverage_by_status(
    result: SealedTestEvaluationResult,
) -> dict[str, object]:
    successful = _successful_evaluations(result)
    counts = {state.value: 0 for state in TacticalPrioritizationState}
    for item in successful:
        counts[item.prioritization.state.value] += 1
    return {
        "denominator": len(result.evaluations),
        "counted": len(successful),
        "excluded_as_error": len(result.evaluations) - len(successful),
        "by_status": counts,
    }


def aggregate_pattern_coverage_scored_vs_abstained(
    result: SealedTestEvaluationResult,
) -> dict[str, dict[str, object]]:
    per_pattern: dict[str, dict[str, int]] = {
        pattern: {"total_candidates": 0, "scored": 0, "abstained": 0} for pattern in PATTERNS
    }
    for item in _successful_evaluations(result):
        for ranking in item.prioritization.rankings:
            bucket = per_pattern[ranking.pattern_id]
            bucket["total_candidates"] += len(ranking.candidates)
            bucket["scored"] += ranking.scored_candidate_count
            bucket["abstained"] += ranking.abstained_candidate_count
    return {
        pattern: {
            **values,
            "scored_proportion": (
                None if values["total_candidates"] == 0
                else values["scored"] / values["total_candidates"]
            ),
        }
        for pattern, values in per_pattern.items()
    }


def _year_bucket(target_year: int) -> str:
    return "2026_partial" if target_year == 2026 else str(target_year)


def aggregate_reconciliation_by_year(
    result: SealedTestEvaluationResult,
) -> dict[str, dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    for item in _successful_evaluations(result):
        key = _year_bucket(item.target.as_of_date.year)
        bucket = buckets.setdefault(
            key, {"orientations": 0, "matches": set(), "by_status": Counter()}
        )
        bucket["orientations"] += 1
        bucket["matches"].add(item.target.target_match_id)
        bucket["by_status"][item.prioritization.state.value] += 1
    return {
        key: {
            "orientations": value["orientations"],
            "matches": len(value["matches"]),
            "by_status": dict(value["by_status"]),
        }
        for key, value in sorted(buckets.items())
    }


def aggregate_wilson_interval_width_distribution(
    result: SealedTestEvaluationResult,
) -> dict[str, object]:
    widths = [
        candidate.combined_width
        for item in _successful_evaluations(result)
        for ranking in item.prioritization.rankings
        for candidate in ranking.candidates
        if candidate.state is TacticalCandidateState.SCORED
        and candidate.combined_width is not None
    ]
    if not widths:
        return {"denominator": 0, "min": None, "max": None, "mean": None}
    return {
        "denominator": len(widths),
        "min": min(widths),
        "max": max(widths),
        "mean": sum(widths) / len(widths),
    }


def aggregate_rank_and_tie_distribution(
    result: SealedTestEvaluationResult,
) -> dict[str, object]:
    rank_counts: Counter[int] = Counter()
    tie_group_sizes: Counter[int] = Counter()
    boundary_tie_expanded_count = 0
    for item in _successful_evaluations(result):
        for ranking in item.prioritization.rankings:
            if ranking.boundary_tie_expanded:
                boundary_tie_expanded_count += 1
            groups: Counter[int] = Counter()
            for candidate in ranking.candidates:
                if candidate.rank_position is not None:
                    rank_counts[candidate.rank_position] += 1
                if candidate.tie_group is not None:
                    groups[candidate.tie_group] += 1
            for size in groups.values():
                tie_group_sizes[size] += 1
    return {
        "rank_position_counts": dict(sorted(rank_counts.items())),
        "tie_group_size_counts": dict(sorted(tie_group_sizes.items())),
        "boundary_tie_expanded_rankings": boundary_tie_expanded_count,
    }


def aggregate_abstention_reason_codes(
    result: SealedTestEvaluationResult,
) -> dict[str, int]:
    codes: Counter[str] = Counter()
    for item in _successful_evaluations(result):
        for ranking in item.prioritization.rankings:
            for candidate in ranking.candidates:
                if candidate.state is not TacticalCandidateState.SCORED:
                    codes.update(candidate.reason_codes)
    return dict(sorted(codes.items()))


def aggregate_coverage_by_surface_and_period(
    result: SealedTestEvaluationResult,
) -> dict[str, dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    for item in _successful_evaluations(result):
        period = _year_bucket(item.target.as_of_date.year)
        key = f"{item.target.surface}:{period}"
        bucket = buckets.setdefault(key, {"orientations": 0, "matches": set()})
        bucket["orientations"] += 1
        bucket["matches"].add(item.target.target_match_id)
    return {
        key: {"orientations": value["orientations"], "matches": len(value["matches"])}
        for key, value in sorted(buckets.items())
    }


# --------------------------------------------------------------------- #
# 5. Metricas exclusivas de P02: Brier, log-loss, baseline poblacional   #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class P02LabeledAttempt:
    """Cierre minimo y explicito del unico label real disponible para
    scoring (``server_won_point``, precedente exclusivo de
    ``explainable_direction_scoring.py``). ``predicted_score`` es la
    puntuacion P10 (``[0,1]``) ya calculada ANTES de observar el punto;
    ``observed_server_won_point`` es el resultado real de ese intento
    especifico dentro del propio partido de test."""

    target_match_id: str
    player: str
    category: str
    predicted_score: float
    observed_server_won_point: bool

    def __post_init__(self) -> None:
        if type(self.predicted_score) is not float or not 0.0 <= self.predicted_score <= 1.0:
            raise FinalSealedEvaluationBoundaryError(
                "predicted_score de P02 debe ser float en [0,1]."
            )
        if type(self.observed_server_won_point) is not bool:
            raise FinalSealedEvaluationBoundaryError(
                "observed_server_won_point debe ser bool real."
            )


def _clip_probability(value: float) -> float:
    return min(max(value, _LOG_LOSS_EPSILON), 1.0 - _LOG_LOSS_EPSILON)


def compute_p02_brier_and_log_loss(
    attempts: tuple[P02LabeledAttempt, ...],
) -> dict[str, object]:
    """Brier/log-loss P02 exclusivos; ``None`` (no ``0``) sin labels."""
    if type(attempts) is not tuple:
        raise FinalSealedEvaluationBoundaryError("attempts debe ser tuple exacto.")
    if not attempts:
        return {"denominator": 0, "brier_score": None, "log_loss": None}
    squared_errors = []
    log_losses = []
    for attempt in attempts:
        outcome = 1.0 if attempt.observed_server_won_point else 0.0
        probability = _clip_probability(attempt.predicted_score)
        squared_errors.append((attempt.predicted_score - outcome) ** 2)
        log_losses.append(
            -(outcome * math.log(probability) + (1.0 - outcome) * math.log(1.0 - probability))
        )
    return {
        "denominator": len(attempts),
        "brier_score": sum(squared_errors) / len(squared_errors),
        "log_loss": sum(log_losses) / len(log_losses),
    }


def compute_p02_population_baseline_comparison(
    attempts: tuple[P02LabeledAttempt, ...],
    population_rate: float,
) -> dict[str, object]:
    """Delta contra el baseline poblacional constante (mismo precedente
    ``population_only`` de ``explainable_direction_scoring.py``); usa
    EXACTAMENTE la misma poblacion que ``compute_p02_brier_and_log_loss``
    para que el delta sea justo."""
    if type(population_rate) is not float or not 0.0 <= population_rate <= 1.0:
        raise FinalSealedEvaluationBoundaryError(
            "population_rate debe ser float en [0,1]."
        )
    model_metrics = compute_p02_brier_and_log_loss(attempts)
    if not attempts:
        return {
            "denominator": 0,
            "delta_brier_vs_population": None,
            "delta_log_loss_vs_population": None,
        }
    baseline_attempts = tuple(
        P02LabeledAttempt(
            item.target_match_id, item.player, item.category,
            population_rate, item.observed_server_won_point,
        )
        for item in attempts
    )
    baseline_metrics = compute_p02_brier_and_log_loss(baseline_attempts)
    return {
        "denominator": model_metrics["denominator"],
        "model_brier_score": model_metrics["brier_score"],
        "population_brier_score": baseline_metrics["brier_score"],
        "delta_brier_vs_population": (
            model_metrics["brier_score"] - baseline_metrics["brier_score"]
        ),
        "model_log_loss": model_metrics["log_loss"],
        "population_log_loss": baseline_metrics["log_loss"],
        "delta_log_loss_vs_population": (
            model_metrics["log_loss"] - baseline_metrics["log_loss"]
        ),
    }


__all__ = (
    "FINAL_SEALED_EVALUATION_BOUNDARY_CONTRACT_VERSION",
    "FROZEN_HISTORY_CUTOFF",
    "FROZEN_PROTOCOL",
    "ORIENTATIONS",
    "PROTOCOLS",
    "ROLLING_PROTOCOL",
    "TEST_RANGE_END",
    "TEST_RANGE_START",
    "FinalSealedEvaluationBoundaryError",
    "P02LabeledAttempt",
    "REAL_TEST_EVALUATION_AUTHORIZED",
    "REAL_TEST_EVALUATION_BLOCK_REASON",
    "SealedTestEvaluationResult",
    "SealedTestMatchRow",
    "SealedTestTarget",
    "SealedTestTargetEvaluation",
    "SealedTestTargetEvaluationError",
    "aggregate_abstention_reason_codes",
    "aggregate_coverage_by_surface_and_period",
    "aggregate_error_bucket",
    "aggregate_orientation_coverage_by_status",
    "aggregate_pattern_coverage_scored_vs_abstained",
    "aggregate_rank_and_tie_distribution",
    "aggregate_reconciliation_by_year",
    "aggregate_wilson_interval_width_distribution",
    "assert_protocols_share_population",
    "build_test_targets",
    "compute_p02_brier_and_log_loss",
    "compute_p02_population_baseline_comparison",
    "evaluate_sealed_test_population",
    "evaluate_single_test_target",
    "run_real_test_evaluation",
    "visible_observations_for_target",
)
