"""Generador offline P14: frontera pura de resultados a snapshot P13.

Transforma un iterable de ``TacticalPrioritizationResult`` target-level
ya calculados en el snapshot privado P13, con metricas agregadas,
reconciliaciones y diagnostico cerrado. No ejecuta P10, no lee datos,
no abre Parquet/CSV, no accede a ``data/`` ni a ``reports/``, no usa
red, no usa pickle y no instancia nada al importar.

Auditoria contractual (evidencia en el codigo P10):

- Los 3.610 resultados target-level existen en la etapa
  ``evidence_and_prioritization`` de
  ``run_tactical_recommender_pipeline`` (bucle de targets), donde cada
  orientacion invoca ``prioritize_tactical_matchup``; se materializan en
  la tupla publica ``TacticalPipelineResult.target_results``
  (``TacticalTargetResult`` con ``.prioritization`` y metadatos de
  target: ``target_match_id``, ``fold``, ``orientation``).
- P10 los conserva en memoria; los descarta al persistir: los cinco
  artefactos son agregados sin identidades (``_safe_summary`` y
  estructuras por patron/fold/disponibilidad).
- La integracion real futura no exige modificar P10: basta un hook/sink
  explicito en una ejecucion offline autorizada que proyecte cada
  ``TacticalTargetResult`` a ``P10OfflineSnapshotRecord``. El snapshot
  real NO puede reconstruirse desde los cinco
  artefactos persistidos (solo agregados); P14 nunca reconstruye ni
  re-extrae: consume resultados preconstruidos.
- Un ``TacticalPrioritizationResult`` no porta ``target_match_id``. Por
  ello el modo ``p10_offline`` exige el record anterior y reconcilia por
  identidad de partido; el modo ``generic`` acepta resultados desnudos
  y no afirma reconciliacion del universo P10.

Semantica de fallos (opcion A): el generador lanza errores contractuales
cerrados (stage + reason code) sin devolver snapshot parcial. El unico
estado de terminacion no-``available`` es ``empty_input`` (modo generic),
que devuelve el snapshot P13 vacio validado. La persistencia real se
delega integramente a P13 (``persist``/``verify``); P14 no implementa
otro formato ni otra politica de atomicidad.

El consumo del iterable esta acotado a ``MAX_ENTRIES + 1``
(``itertools.islice``); complejidad O(N). El resultado publico no porta
identidades, fechas individuales, match_id, rutas ni payloads: solo
agregados cerrados y el fingerprint P13. El snapshot es privado y
contiene sus claves contractuales.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from hashlib import sha256
from itertools import islice
import json
import re
from typing import Final
from types import MappingProxyType

from src.recommender import persisted_tactical_recommendation_provider as p13
from src.recommender.tactical_feature_encoder import TacticalEncodingPolicy
from src.recommender.tactical_prioritization import (
    TacticalPrioritizationResult,
    TacticalPrioritizationState,
    TacticalScoringPolicy,
    default_tactical_scoring_policy,
    tactical_prioritization_result_fingerprint,
    validate_tactical_prioritization_result,
)


GENERATION_CONTRACT_NAME: Final = "tactical_recommendation_snapshot_generation"
GENERATION_SCHEMA_VERSION: Final = "1.0.0"
GENERATION_FINGERPRINT_DOMAIN: Final = (
    b"tennis-tactical-recommendation-snapshot-generation\x00"
)

SEALED_TEST_FIRST_DAY: Final = date(2024, 1, 1)
_CIVIL_ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_P10_FOLDS: Final = frozenset(
    {"validation_2020", "validation_2021", "validation_2022", "validation_2023"}
)
_P10_ORIENTATIONS: Final = frozenset(
    {"player_1_vs_player_2", "player_2_vs_player_1"}
)

SNAPSHOT_GENERATION_MODE_GENERIC: Final = "generic"
SNAPSHOT_GENERATION_MODE_P10_OFFLINE: Final = "p10_offline"
_SNAPSHOT_GENERATION_MODES: Final = frozenset(
    {SNAPSHOT_GENERATION_MODE_GENERIC, SNAPSHOT_GENERATION_MODE_P10_OFFLINE}
)

SNAPSHOT_GENERATION_STATE_AVAILABLE: Final = "available"
SNAPSHOT_GENERATION_STATE_EMPTY_INPUT: Final = "empty_input"
_SNAPSHOT_GENERATION_STATES: Final = frozenset(
    {SNAPSHOT_GENERATION_STATE_AVAILABLE, SNAPSHOT_GENERATION_STATE_EMPTY_INPUT}
)

_GENERATION_STAGE_INPUT: Final = "input"
_GENERATION_STAGE_UPSTREAM: Final = "upstream"
_GENERATION_STAGE_POLICY: Final = "policy"
_GENERATION_STAGE_TEMPORAL: Final = "temporal"
_GENERATION_STAGE_KEY: Final = "key"
_GENERATION_STAGE_LIMIT: Final = "limit"
_GENERATION_STAGE_UNIVERSE: Final = "universe"
_GENERATION_STAGE_BUILD: Final = "build"
GENERATION_STAGES: Final = frozenset(
    {
        _GENERATION_STAGE_INPUT,
        _GENERATION_STAGE_UPSTREAM,
        _GENERATION_STAGE_POLICY,
        _GENERATION_STAGE_TEMPORAL,
        _GENERATION_STAGE_KEY,
        _GENERATION_STAGE_LIMIT,
        _GENERATION_STAGE_UNIVERSE,
        _GENERATION_STAGE_BUILD,
    }
)

_STATE_REASON_EMPTY_INPUT: Final = "empty_input"
GENERATION_REASON_CODES: Final = frozenset(
    {
        _STATE_REASON_EMPTY_INPUT,
        "element_not_tactical_prioritization_result",
        "element_not_p10_offline_snapshot_record",
        "upstream_result_invalid",
        "scoring_policy_mismatch",
        "matchup_query_policy_mismatch",
        "ranking_policy_mismatch",
        "as_of_date_not_civil",
        "as_of_date_in_sealed_test_range",
        "player_equals_opponent",
        "duplicate_key_exact",
        "duplicate_key_conflicting",
        "entry_limit_exceeded",
        "universe_entry_count_mismatch",
        "universe_reciprocal_orientation_missing",
        "universe_reciprocal_pair_count_mismatch",
        "universe_fold_distribution_mismatch",
        "p13_snapshot_build_failed",
        "canonical_serialization_mismatch",
        "p13_verification_mismatch",
    }
)

_ERROR_MESSAGES: Final = MappingProxyType(
    {
        "element_not_tactical_prioritization_result": (
            "Input fuera de contrato: elemento no es TacticalPrioritizationResult exacto."
        ),
        "element_not_p10_offline_snapshot_record": (
            "Input fuera de contrato: elemento no es P10OfflineSnapshotRecord exacto."
        ),
        "upstream_result_invalid": (
            "Input fuera de contrato: resultado upstream invalido."
        ),
        "scoring_policy_mismatch": (
            "Politica fuera de contrato: scoring policy no coincidente."
        ),
        "matchup_query_policy_mismatch": (
            "Politica fuera de contrato: query de matchup no coincidente."
        ),
        "ranking_policy_mismatch": (
            "Politica fuera de contrato: ranking no coincidente."
        ),
        "as_of_date_not_civil": (
            "Temporal fuera de contrato: fecha as_of no civil ISO exacta."
        ),
        "as_of_date_in_sealed_test_range": (
            "Temporal fuera de contrato: fecha as_of dentro del test sellado."
        ),
        "player_equals_opponent": (
            "Clave fuera de contrato: player igual a opponent."
        ),
        "duplicate_key_exact": "Clave fuera de contrato: duplicado exacto.",
        "duplicate_key_conflicting": "Clave fuera de contrato: duplicado conflictivo.",
        "entry_limit_exceeded": "Limite fuera de contrato: MAX_ENTRIES excedido.",
        "universe_entry_count_mismatch": (
            "Universo P10 fuera de contrato: cantidad de entradas."
        ),
        "universe_reciprocal_orientation_missing": (
            "Universo P10 fuera de contrato: orientacion reciproca ausente."
        ),
        "universe_reciprocal_pair_count_mismatch": (
            "Universo P10 fuera de contrato: cantidad de pares reciprocos."
        ),
        "universe_fold_distribution_mismatch": (
            "Universo P10 fuera de contrato: distribucion por fold."
        ),
        "p13_snapshot_build_failed": "Construccion P13 fallida sin snapshot parcial.",
        "canonical_serialization_mismatch": "Serializacion canonica no determinista.",
        "p13_verification_mismatch": "Verificacion P13 tras persistencia fallida.",
    }
)


class TacticalRecommendationSnapshotGenerationError(RuntimeError):
    """Error contractual P14 cerrado (stage + reason code)."""

    __slots__ = ("stage", "reason_code")

    def __init__(self, stage: str, reason_code: str) -> None:
        if stage not in GENERATION_STAGES:
            stage = _GENERATION_STAGE_BUILD
        if reason_code not in GENERATION_REASON_CODES:
            reason_code = "p13_snapshot_build_failed"
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "reason_code", reason_code)
        super().__init__(_ERROR_MESSAGES[reason_code])


class SnapshotGenerationInputError(TacticalRecommendationSnapshotGenerationError):
    """Elemento de input fuera de contrato."""


class SnapshotGenerationUpstreamError(TacticalRecommendationSnapshotGenerationError):
    """Resultado upstream P10 invalido."""


class SnapshotGenerationPolicyError(TacticalRecommendationSnapshotGenerationError):
    """Politica congelada no coincidente."""


class SnapshotGenerationTemporalError(TacticalRecommendationSnapshotGenerationError):
    """Frontera temporal sellada violada."""


class SnapshotGenerationKeyError(TacticalRecommendationSnapshotGenerationError):
    """Clave privada fuera de contrato (clash o duplicado)."""


class SnapshotGenerationLimitError(TacticalRecommendationSnapshotGenerationError):
    """Limite MAX_ENTRIES excedido."""


class SnapshotGenerationUniverseError(TacticalRecommendationSnapshotGenerationError):
    """Universo P10 no reconciliado."""


class SnapshotGenerationBuildError(TacticalRecommendationSnapshotGenerationError):
    """Construccion/verificacion P13 fallida."""


def _fail(error_cls, stage: str, reason_code: str):
    return error_cls(stage, reason_code)


def _civil_as_of_date(value: object) -> date:
    if type(value) is not str or _CIVIL_ISO_DATE.fullmatch(value) is None:
        raise _fail(
            SnapshotGenerationTemporalError,
            _GENERATION_STAGE_TEMPORAL,
            "as_of_date_not_civil",
        )
    parsed: date | None = None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        pass
    if parsed is None:
        raise _fail(
            SnapshotGenerationTemporalError,
            _GENERATION_STAGE_TEMPORAL,
            "as_of_date_not_civil",
        )
    return parsed


@dataclass(frozen=True)
class P10OfflineSnapshotRecord:
    """Proyeccion minima, privada y sin dependencia de importacion de P10."""

    target_match_id: str
    fold: str
    orientation: str
    prioritization: TacticalPrioritizationResult

    def __post_init__(self) -> None:
        if (
            type(self.target_match_id) is not str
            or not self.target_match_id
            or self.target_match_id != self.target_match_id.strip()
            or len(self.target_match_id) > 512
            or any(ord(character) < 0x20 for character in self.target_match_id)
            or type(self.fold) is not str
            or self.fold not in _P10_FOLDS
            or type(self.orientation) is not str
            or self.orientation not in _P10_ORIENTATIONS
            or type(self.prioritization) is not TacticalPrioritizationResult
        ):
            raise TypeError("Record P10 offline fuera del contrato exacto.")
        invalid = False
        try:
            validate_tactical_prioritization_result(self.prioritization)
        except Exception:
            invalid = True
        if invalid:
            raise TypeError("Record P10 offline fuera del contrato exacto.")
        as_of = _civil_as_of_date(self.prioritization.matchup_query.as_of_date)
        if as_of >= SEALED_TEST_FIRST_DAY or self.fold != f"validation_{as_of.year}":
            raise TypeError("Record P10 offline fuera del contrato temporal.")


@dataclass(frozen=True)
class SnapshotGenerationPolicy:
    """Politica congelada de generacion (generico o modo P10 offline)."""

    mode: str
    scoring_policy: TacticalScoringPolicy
    expected_encoder_policy: TacticalEncodingPolicy
    expected_scope_strategy: str
    expected_requested_patterns: tuple[str, ...]
    expected_serve_numbers: tuple[int, ...]
    expected_minimum_labeled_attempts: int
    expected_minimum_matches: int
    expected_window_days: int | None
    expected_top_k: int
    required_entries: int | None
    required_reciprocal_pairs: int | None
    expected_fold_orientations: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        validate_snapshot_generation_policy(self)


def validate_snapshot_generation_policy(policy: object) -> None:
    if type(policy) is not SnapshotGenerationPolicy:
        raise TypeError("La politica debe ser SnapshotGenerationPolicy exacta.")
    if type(policy.mode) is not str or policy.mode not in _SNAPSHOT_GENERATION_MODES:
        raise ValueError("El modo de generacion esta fuera del contrato cerrado.")
    if type(policy.scoring_policy) is not TacticalScoringPolicy:
        raise ValueError("La scoring policy de referencia no es del contrato P10.")
    if (
        type(policy.expected_encoder_policy) is not TacticalEncodingPolicy
        or type(policy.expected_scope_strategy) is not str
        or policy.expected_scope_strategy != "global_only"
        or type(policy.expected_requested_patterns) is not tuple
        or any(
            type(item) is not str for item in policy.expected_requested_patterns
        )
        or type(policy.expected_serve_numbers) is not tuple
        or not policy.expected_serve_numbers
        or any(
            type(item) is not int or isinstance(item, bool)
            for item in policy.expected_serve_numbers
        )
        or type(policy.expected_minimum_labeled_attempts) is not int
        or isinstance(policy.expected_minimum_labeled_attempts, bool)
        or type(policy.expected_minimum_matches) is not int
        or isinstance(policy.expected_minimum_matches, bool)
        or policy.expected_minimum_labeled_attempts < 0
        or policy.expected_minimum_matches < 0
        or (
            policy.expected_window_days is not None
            and (
                type(policy.expected_window_days) is not int
                or isinstance(policy.expected_window_days, bool)
            )
        )
        or type(policy.expected_top_k) is not int
        or isinstance(policy.expected_top_k, bool)
        or policy.expected_top_k < 1
    ):
        raise ValueError("Expectativas de la politica fuera del contrato exacto.")
    is_p10 = policy.mode == SNAPSHOT_GENERATION_MODE_P10_OFFLINE
    if not is_p10 and (
        policy.required_entries is not None
        or policy.required_reciprocal_pairs is not None
    ):
        raise ValueError(
            "El modo generico no admite universo requerido (campos ignorables prohibidos)."
        )
    if policy.required_entries is None:
        entries_ok = not is_p10
    else:
        entries_ok = (
            type(policy.required_entries) is int
            and not isinstance(policy.required_entries, bool)
            and 0 < policy.required_entries <= p13.MAX_ENTRIES
        )
    if policy.required_reciprocal_pairs is None:
        pairs_ok = not is_p10
    else:
        pairs_ok = (
            type(policy.required_reciprocal_pairs) is int
            and not isinstance(policy.required_reciprocal_pairs, bool)
            and policy.required_reciprocal_pairs > 0
        )
    folds = policy.expected_fold_orientations
    folds_ok = type(folds) is tuple
    if folds_ok:
        years = set()
        for item in folds:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or isinstance(item[0], bool)
                or type(item[1]) is not int
                or isinstance(item[1], bool)
                or item[1] < 0
            ):
                folds_ok = False
                break
            years.add(item[0])
        if folds_ok and is_p10 and years != {2020, 2021, 2022, 2023}:
            folds_ok = False
        if folds_ok and not is_p10 and folds:
            folds_ok = False
    if not (entries_ok and pairs_ok and folds_ok):
        raise ValueError("Universo de la politica fuera del contrato cerrado.")
    if is_p10:
        if (
            policy.required_entries != 3610
            or policy.required_reciprocal_pairs != 1805
            or tuple(folds)
            != ((2020, 336), (2021, 742), (2022, 1292), (2023, 1240))
            or sum(count for _, count in folds) != 3610
        ):
            raise ValueError("El universo P10 offline es exacto y cerrado.")
        if (
            policy.scoring_policy != default_tactical_scoring_policy()
            or policy.expected_encoder_policy
            is not TacticalEncodingPolicy.COMPONENT_ONLY
            or tuple(policy.expected_requested_patterns)
            != ("P02", "P04", "P05", "P06")
            or tuple(policy.expected_serve_numbers) != (1, 2)
            or policy.expected_minimum_labeled_attempts != 50
            or policy.expected_minimum_matches != 5
            or policy.expected_window_days is not None
            or policy.expected_top_k != 3
        ):
            raise ValueError("La politica P10 offline es exacta y congelada.")


GENERIC_SNAPSHOT_GENERATION_POLICY: Final = SnapshotGenerationPolicy(
    mode=SNAPSHOT_GENERATION_MODE_GENERIC,
    scoring_policy=default_tactical_scoring_policy(),
    expected_encoder_policy=TacticalEncodingPolicy.COMPONENT_ONLY,
    expected_scope_strategy="global_only",
    expected_requested_patterns=("P02", "P04", "P05", "P06"),
    expected_serve_numbers=(1, 2),
    expected_minimum_labeled_attempts=50,
    expected_minimum_matches=5,
    expected_window_days=None,
    expected_top_k=3,
    required_entries=None,
    required_reciprocal_pairs=None,
    expected_fold_orientations=(),
)

P10_OFFLINE_SNAPSHOT_GENERATION_POLICY: Final = SnapshotGenerationPolicy(
    mode=SNAPSHOT_GENERATION_MODE_P10_OFFLINE,
    scoring_policy=default_tactical_scoring_policy(),
    expected_encoder_policy=TacticalEncodingPolicy.COMPONENT_ONLY,
    expected_scope_strategy="global_only",
    expected_requested_patterns=("P02", "P04", "P05", "P06"),
    expected_serve_numbers=(1, 2),
    expected_minimum_labeled_attempts=50,
    expected_minimum_matches=5,
    expected_window_days=None,
    expected_top_k=3,
    required_entries=3610,
    required_reciprocal_pairs=1805,
    expected_fold_orientations=((2020, 336), (2021, 742), (2022, 1292), (2023, 1240)),
)


def _scoring_policy_payload(policy: object) -> dict[str, object]:
    return {
        "contract_version": policy.contract_version,
        "name": policy.name,
        "version": policy.version,
        "executor_weight": policy.executor_weight,
        "opponent_allowed_weight": policy.opponent_allowed_weight,
        "formula": policy.formula,
        "allowed_scope": policy.allowed_scope,
        "uncertainty_method": policy.uncertainty_method,
        "abstention_rules": list(policy.abstention_rules),
        "comparability_rule": policy.comparability_rule,
        "interpretation": policy.interpretation,
        "reconciliations": [[k, v] for k, v in policy.reconciliations],
    }


def _policy_payload(policy: SnapshotGenerationPolicy) -> dict[str, object]:
    return {
        "mode": policy.mode,
        "scoring_policy": _scoring_policy_payload(policy.scoring_policy),
        "expected_encoder_policy": policy.expected_encoder_policy.value,
        "expected_scope_strategy": policy.expected_scope_strategy,
        "expected_requested_patterns": list(policy.expected_requested_patterns),
        "expected_serve_numbers": list(policy.expected_serve_numbers),
        "expected_minimum_labeled_attempts": policy.expected_minimum_labeled_attempts,
        "expected_minimum_matches": policy.expected_minimum_matches,
        "expected_window_days": policy.expected_window_days,
        "expected_top_k": policy.expected_top_k,
        "required_entries": policy.required_entries,
        "required_reciprocal_pairs": policy.required_reciprocal_pairs,
        "expected_fold_orientations": [
            [year, count] for year, count in policy.expected_fold_orientations
        ],
    }


@dataclass(frozen=True)
class SnapshotGenerationDiagnostics:
    """Instrumentacion cerrada y determinista (sin duraciones)."""

    elements_inspected: int
    results_validated: int
    entries_built: int
    builder_calls: int
    serializations: int
    persistences: int
    verifications: int
    canonical_bytes: int
    estimated_universe_bytes: int


@dataclass(frozen=True)
class TacticalRecommendationSnapshotGenerationResult:
    """Resultado agregado, seguro y reconstructivamente validable."""

    contract: str
    schema_version: str
    state: str
    policy: SnapshotGenerationPolicy
    inputs_received: int
    entries_generated: int
    available_entries: int
    partially_available_entries: int
    not_available_entries: int
    fold_orientations: tuple[tuple[int, int], ...]
    snapshot_fingerprint: str
    serialized_bytes: int
    reason_codes: tuple[str, ...]
    diagnostics: SnapshotGenerationDiagnostics
    snapshot: p13.PersistedTacticalRecommendationSnapshot

    def __post_init__(self) -> None:
        validate_tactical_recommendation_snapshot_generation_result(self)


def _check_policy_compatibility(
    result: TacticalPrioritizationResult, policy: SnapshotGenerationPolicy
) -> None:
    if result.policy != policy.scoring_policy:
        raise _fail(
            SnapshotGenerationPolicyError,
            _GENERATION_STAGE_POLICY,
            "scoring_policy_mismatch",
        )
    summary = result.matchup_query
    if (
        summary.encoder_policy is not policy.expected_encoder_policy
        or summary.scope_strategy != policy.expected_scope_strategy
        or tuple(summary.requested_patterns)
        != tuple(policy.expected_requested_patterns)
        or tuple(summary.serve_numbers) != tuple(policy.expected_serve_numbers)
        or summary.minimum_labeled_attempts
        != policy.expected_minimum_labeled_attempts
        or summary.minimum_matches != policy.expected_minimum_matches
        or summary.window_days != policy.expected_window_days
    ):
        raise _fail(
            SnapshotGenerationPolicyError,
            _GENERATION_STAGE_POLICY,
            "matchup_query_policy_mismatch",
        )
    pattern_order = tuple(ranking.pattern_id for ranking in result.rankings)
    for ranking in result.rankings:
        if (
            ranking.encoder_policy is not policy.expected_encoder_policy
            or ranking.requested_top_k != policy.expected_top_k
        ):
            raise _fail(
                SnapshotGenerationPolicyError,
                _GENERATION_STAGE_POLICY,
                "ranking_policy_mismatch",
            )
    if pattern_order != tuple(policy.expected_requested_patterns):
        raise _fail(
            SnapshotGenerationPolicyError,
            _GENERATION_STAGE_POLICY,
            "ranking_policy_mismatch",
        )


def _reconcile_p10_universe(
    records: tuple[P10OfflineSnapshotRecord, ...], policy: SnapshotGenerationPolicy
) -> tuple[tuple[int, int], ...]:
    by_match: dict[str, list[P10OfflineSnapshotRecord]] = {}
    for record in records:
        by_match.setdefault(record.target_match_id, []).append(record)
    for rows in by_match.values():
        if len(rows) != 2:
            raise _fail(
                SnapshotGenerationUniverseError,
                _GENERATION_STAGE_UNIVERSE,
                "universe_reciprocal_orientation_missing",
            )
        first, second = rows
        first_query = first.prioritization.matchup_query
        second_query = second.prioritization.matchup_query
        if (
            {first.orientation, second.orientation} != _P10_ORIENTATIONS
            or first.fold != second.fold
            or first_query.as_of_date != second_query.as_of_date
            or first_query.player != second_query.opponent
            or first_query.opponent != second_query.player
        ):
            raise _fail(
                SnapshotGenerationUniverseError,
                _GENERATION_STAGE_UNIVERSE,
                "universe_reciprocal_orientation_missing",
            )
    if len(by_match) != policy.required_reciprocal_pairs:
        raise _fail(
            SnapshotGenerationUniverseError,
            _GENERATION_STAGE_UNIVERSE,
            "universe_reciprocal_pair_count_mismatch",
        )
    fold_counts: dict[int, int] = {}
    for record in records:
        year = int(record.fold.removeprefix("validation_"))
        fold_counts[year] = fold_counts.get(year, 0) + 1
    expected = tuple(sorted(policy.expected_fold_orientations))
    if tuple(sorted(fold_counts.items())) != expected:
        raise _fail(
            SnapshotGenerationUniverseError,
            _GENERATION_STAGE_UNIVERSE,
            "universe_fold_distribution_mismatch",
        )
    return expected


def generate_tactical_recommendation_snapshot(
    results: object,
    *,
    policy: SnapshotGenerationPolicy = GENERIC_SNAPSHOT_GENERATION_POLICY,
) -> TacticalRecommendationSnapshotGenerationResult:
    """Genera el snapshot P13 validado con metricas cerradas (O(N)).

    Fallos: lanza ``TacticalRecommendationSnapshotGenerationError``
    (stage + reason code cerrados) sin devolver snapshot parcial.
    Exitos: ``available`` o ``empty_input`` (solo generico, con el
    snapshot P13 vacio validado).
    """
    validate_snapshot_generation_policy(policy)
    if isinstance(results, (str, bytes, bytearray, dict)):
        raise TypeError("results debe ser un iterable de resultados validados.")
    try:
        iterator = iter(results)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            "results debe ser un iterable de resultados validados."
        ) from None

    limit = p13.MAX_ENTRIES
    materialized: list[TacticalPrioritizationResult] = []
    p10_records: list[P10OfflineSnapshotRecord] = []
    seen: dict[tuple[str, str, date], str] = {}
    available_count = 0
    partial_count = 0
    unavailable_count = 0
    year_counts: dict[int, int] = {}

    iteration_failed = False
    try:
        bounded_items = islice(iterator, limit + 1)
        for item in bounded_items:
            if policy.mode == SNAPSHOT_GENERATION_MODE_P10_OFFLINE:
                if type(item) is not P10OfflineSnapshotRecord:
                    raise _fail(
                        SnapshotGenerationInputError,
                        _GENERATION_STAGE_INPUT,
                        "element_not_p10_offline_snapshot_record",
                    )
                p10_records.append(item)
                result = item.prioritization
            else:
                if type(item) is not TacticalPrioritizationResult:
                    raise _fail(
                        SnapshotGenerationInputError,
                        _GENERATION_STAGE_INPUT,
                        "element_not_tactical_prioritization_result",
                    )
                result = item
            if type(result) is not TacticalPrioritizationResult:
                raise _fail(
                    SnapshotGenerationInputError,
                    _GENERATION_STAGE_INPUT,
                    "element_not_tactical_prioritization_result",
                )
            upstream_invalid = False
            try:
                validate_tactical_prioritization_result(result)
            except Exception:
                upstream_invalid = True
            if upstream_invalid:
                raise _fail(
                    SnapshotGenerationUpstreamError,
                    _GENERATION_STAGE_UPSTREAM,
                    "upstream_result_invalid",
                )
            _check_policy_compatibility(result, policy)
            summary = result.matchup_query
            player = summary.player
            opponent = summary.opponent
            if player == opponent:
                raise _fail(
                    SnapshotGenerationKeyError,
                    _GENERATION_STAGE_KEY,
                    "player_equals_opponent",
                )
            as_of = _civil_as_of_date(summary.as_of_date)
            if as_of >= SEALED_TEST_FIRST_DAY:
                raise _fail(
                    SnapshotGenerationTemporalError,
                    _GENERATION_STAGE_TEMPORAL,
                    "as_of_date_in_sealed_test_range",
                )
            key = (player, opponent, as_of)
            fingerprint_failed = False
            try:
                upstream_fingerprint = tactical_prioritization_result_fingerprint(result)
            except Exception:
                fingerprint_failed = True
                upstream_fingerprint = ""
            if fingerprint_failed:
                raise _fail(
                    SnapshotGenerationUpstreamError,
                    _GENERATION_STAGE_UPSTREAM,
                    "upstream_result_invalid",
                )
            previous = seen.get(key)
            if previous is not None:
                reason = (
                    "duplicate_key_exact"
                    if previous == upstream_fingerprint
                    else "duplicate_key_conflicting"
                )
                raise _fail(SnapshotGenerationKeyError, _GENERATION_STAGE_KEY, reason)
            seen[key] = upstream_fingerprint
            materialized.append(result)
            if result.state is TacticalPrioritizationState.AVAILABLE:
                available_count += 1
            elif result.state is TacticalPrioritizationState.PARTIALLY_AVAILABLE:
                partial_count += 1
            else:
                unavailable_count += 1
            year_counts[as_of.year] = year_counts.get(as_of.year, 0) + 1
    except TacticalRecommendationSnapshotGenerationError:
        raise
    except Exception:
        iteration_failed = True
    if iteration_failed:
        raise _fail(
            SnapshotGenerationInputError,
            _GENERATION_STAGE_INPUT,
            "element_not_tactical_prioritization_result",
        )

    if len(materialized) > limit:
        raise _fail(
            SnapshotGenerationLimitError,
            _GENERATION_STAGE_LIMIT,
            "entry_limit_exceeded",
        )

    inputs_received = len(materialized)
    if policy.mode == SNAPSHOT_GENERATION_MODE_P10_OFFLINE:
        if inputs_received != policy.required_entries:
            raise _fail(
                SnapshotGenerationUniverseError,
                _GENERATION_STAGE_UNIVERSE,
                "universe_entry_count_mismatch",
            )
        fold_orientations = _reconcile_p10_universe(tuple(p10_records), policy)
    else:
        fold_orientations = tuple(sorted(year_counts.items()))

    build_failed = False
    try:
        if inputs_received == 0:
            snapshot = p13.build_persisted_tactical_recommendation_snapshot(())
        else:
            snapshot = p13.build_persisted_tactical_recommendation_snapshot(
                tuple(materialized)
            )
        serialized = p13.serialize_persisted_tactical_recommendation_snapshot(snapshot)
    except Exception:
        build_failed = True
    if build_failed:
        raise _fail(
            SnapshotGenerationBuildError,
            _GENERATION_STAGE_BUILD,
            "p13_snapshot_build_failed",
        ) from None

    if policy.mode == SNAPSHOT_GENERATION_MODE_P10_OFFLINE:
        estimated_universe_bytes = (
            -(-len(serialized) // inputs_received) * policy.required_entries
        )
    else:
        estimated_universe_bytes = 0

    state = (
        SNAPSHOT_GENERATION_STATE_EMPTY_INPUT
        if inputs_received == 0
        else SNAPSHOT_GENERATION_STATE_AVAILABLE
    )
    reason_codes = (_STATE_REASON_EMPTY_INPUT,) if inputs_received == 0 else ()

    diagnostics = SnapshotGenerationDiagnostics(
        elements_inspected=inputs_received,
        results_validated=inputs_received,
        entries_built=inputs_received,
        builder_calls=1,
        serializations=1,
        persistences=0,
        verifications=0,
        canonical_bytes=len(serialized),
        estimated_universe_bytes=estimated_universe_bytes,
    )
    return TacticalRecommendationSnapshotGenerationResult(
        contract=GENERATION_CONTRACT_NAME,
        schema_version=GENERATION_SCHEMA_VERSION,
        state=state,
        policy=policy,
        inputs_received=inputs_received,
        entries_generated=inputs_received,
        available_entries=available_count,
        partially_available_entries=partial_count,
        not_available_entries=unavailable_count,
        fold_orientations=fold_orientations,
        snapshot_fingerprint=snapshot.fingerprint,
        serialized_bytes=len(serialized),
        reason_codes=reason_codes,
        diagnostics=diagnostics,
        snapshot=snapshot,
    )


def generate_and_persist_tactical_recommendation_snapshot(
    results: object,
    destination_path: object,
    *,
    policy: SnapshotGenerationPolicy = GENERIC_SNAPSHOT_GENERATION_POLICY,
) -> TacticalRecommendationSnapshotGenerationResult:
    """Genera, valida, serializa dos veces, persiste (P13) y verifica (P13).

    La persistencia y la verificacion se delegan a P13; sus errores
    internos se propagan sin snapshot parcial. No se implementa aqui
    ningun formato ni ninguna politica de atomicidad.
    """
    result = generate_tactical_recommendation_snapshot(results, policy=policy)
    first = p13.serialize_persisted_tactical_recommendation_snapshot(result.snapshot)
    second = p13.serialize_persisted_tactical_recommendation_snapshot(result.snapshot)
    if first != second:
        raise _fail(
            SnapshotGenerationBuildError,
            _GENERATION_STAGE_BUILD,
            "canonical_serialization_mismatch",
        )
    p13.persist_persisted_tactical_recommendation_snapshot(
        result.snapshot, destination_path
    )
    verified = p13.verify_persisted_tactical_recommendation_snapshot(destination_path)
    if verified != result.snapshot.fingerprint:
        raise _fail(
            SnapshotGenerationBuildError,
            _GENERATION_STAGE_BUILD,
            "p13_verification_mismatch",
        )
    diagnostics = SnapshotGenerationDiagnostics(
        elements_inspected=result.diagnostics.elements_inspected,
        results_validated=result.diagnostics.results_validated,
        entries_built=result.diagnostics.entries_built,
        builder_calls=result.diagnostics.builder_calls,
        serializations=result.diagnostics.serializations + 2,
        persistences=1,
        verifications=1,
        canonical_bytes=result.diagnostics.canonical_bytes,
        estimated_universe_bytes=result.diagnostics.estimated_universe_bytes,
    )
    return TacticalRecommendationSnapshotGenerationResult(
        contract=GENERATION_CONTRACT_NAME,
        schema_version=GENERATION_SCHEMA_VERSION,
        state=result.state,
        policy=result.policy,
        inputs_received=result.inputs_received,
        entries_generated=result.entries_generated,
        available_entries=result.available_entries,
        partially_available_entries=result.partially_available_entries,
        not_available_entries=result.not_available_entries,
        fold_orientations=result.fold_orientations,
        snapshot_fingerprint=result.snapshot_fingerprint,
        serialized_bytes=result.serialized_bytes,
        reason_codes=result.reason_codes,
        diagnostics=diagnostics,
        snapshot=result.snapshot,
    )


def _generation_payload(
    result: TacticalRecommendationSnapshotGenerationResult,
) -> dict[str, object]:
    return {
        "contract": result.contract,
        "schema_version": result.schema_version,
        "state": result.state,
        "policy": _policy_payload(result.policy),
        "inputs_received": result.inputs_received,
        "entries_generated": result.entries_generated,
        "available_entries": result.available_entries,
        "partially_available_entries": result.partially_available_entries,
        "not_available_entries": result.not_available_entries,
        "fold_orientations": [
            [year, count] for year, count in result.fold_orientations
        ],
        "snapshot_fingerprint": result.snapshot_fingerprint,
        "serialized_bytes": result.serialized_bytes,
        "reason_codes": list(result.reason_codes),
        "diagnostics": {
            "elements_inspected": result.diagnostics.elements_inspected,
            "results_validated": result.diagnostics.results_validated,
            "entries_built": result.diagnostics.entries_built,
            "builder_calls": result.diagnostics.builder_calls,
            "serializations": result.diagnostics.serializations,
            "persistences": result.diagnostics.persistences,
            "verifications": result.diagnostics.verifications,
            "canonical_bytes": result.diagnostics.canonical_bytes,
            "estimated_universe_bytes": result.diagnostics.estimated_universe_bytes,
        },
    }


def _generation_canonical_bytes(result: object) -> bytes:
    payload = _generation_payload(result)  # type: ignore[arg-type]
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_snapshot_generation_json(
    result: TacticalRecommendationSnapshotGenerationResult,
) -> bytes:
    """JSON canonico cerrado del resultado (compacto, ordenado, UTF-8)."""
    validate_tactical_recommendation_snapshot_generation_result(result)
    return _generation_canonical_bytes(result)


def snapshot_generation_fingerprint(
    result: TacticalRecommendationSnapshotGenerationResult,
) -> str:
    """SHA-256 mayuscula con dominio P14 sobre el payload canónico.

    El payload no porta un campo de fingerprint propio (nada que
    excluir) e incluye el fingerprint P13; no recalcula una
    representacion alternativa del contenido del snapshot.
    """
    first = canonical_snapshot_generation_json(result)
    second = canonical_snapshot_generation_json(result)
    if first != second:
        raise _fail(
            SnapshotGenerationBuildError,
            _GENERATION_STAGE_BUILD,
            "canonical_serialization_mismatch",
        )
    return sha256(GENERATION_FINGERPRINT_DOMAIN + first).hexdigest().upper()


def _check_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} debe ser un entero no negativo exacto.")


def validate_tactical_recommendation_snapshot_generation_result(
    result: object,
) -> None:
    """Validacion reconstructiva completa del resultado P14."""
    if type(result) is not TacticalRecommendationSnapshotGenerationResult:
        raise TypeError(
            "El resultado debe ser TacticalRecommendationSnapshotGenerationResult exacto."
        )
    if (
        result.contract != GENERATION_CONTRACT_NAME
        or result.schema_version != GENERATION_SCHEMA_VERSION
        or result.state not in _SNAPSHOT_GENERATION_STATES
    ):
        raise ValueError("Metadatos contractuales fuera del cierre P14.")
    validate_snapshot_generation_policy(result.policy)
    for name in (
        "inputs_received",
        "entries_generated",
        "available_entries",
        "partially_available_entries",
        "not_available_entries",
        "serialized_bytes",
    ):
        _check_nonnegative_int(getattr(result, name), name)
    if type(result.fold_orientations) is not tuple or any(
        (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not int
            or isinstance(item[0], bool)
            or type(item[1]) is not int
            or isinstance(item[1], bool)
            or item[1] < 0
        )
        for item in result.fold_orientations
    ):
        raise TypeError("fold_orientations fuera del contrato exacto.")
    if type(result.reason_codes) is not tuple or any(
        type(item) is not str or item not in GENERATION_REASON_CODES
        for item in result.reason_codes
    ):
        raise TypeError("reason_codes fuera del contrato cerrado.")
    diagnostics = result.diagnostics
    if type(diagnostics) is not SnapshotGenerationDiagnostics:
        raise TypeError("diagnostics fuera del contrato exacto.")
    for field_info in fields(SnapshotGenerationDiagnostics):
        _check_nonnegative_int(getattr(diagnostics, field_info.name), field_info.name)
    p13.validate_persisted_tactical_recommendation_snapshot(result.snapshot)
    snapshot = result.snapshot
    snapshot_entry_count = len(snapshot.entries)
    if (
        result.entries_generated != snapshot_entry_count
        or result.inputs_received != snapshot_entry_count
        or result.serialized_bytes
        != len(
            p13.serialize_persisted_tactical_recommendation_snapshot(snapshot)
        )
        or result.snapshot_fingerprint != snapshot.fingerprint
    ):
        raise ValueError("Conteos P14 inconsistentes con el snapshot P13.")
    available = 0
    partial = 0
    unavailable = 0
    year_counts: dict[int, int] = {}
    for entry in snapshot.entries:
        if entry.result.state is TacticalPrioritizationState.AVAILABLE:
            available += 1
        elif entry.result.state is TacticalPrioritizationState.PARTIALLY_AVAILABLE:
            partial += 1
        else:
            unavailable += 1
        year = entry.key.as_of_date.year
        if entry.key.as_of_date >= SEALED_TEST_FIRST_DAY:
            raise ValueError("El snapshot contiene fechas del test sellado.")
        year_counts[year] = year_counts.get(year, 0) + 1
    if (
        result.available_entries != available
        or result.partially_available_entries != partial
        or result.not_available_entries != unavailable
        or tuple(sorted(year_counts.items())) != tuple(result.fold_orientations)
    ):
        raise ValueError("Particiones agregadas P14 inconsistentes.")
    if snapshot_entry_count == 0:
        if (
            result.state != SNAPSHOT_GENERATION_STATE_EMPTY_INPUT
            or result.policy.mode == SNAPSHOT_GENERATION_MODE_P10_OFFLINE
            or result.reason_codes != (_STATE_REASON_EMPTY_INPUT,)
        ):
            raise ValueError("Estado empty_input fuera del contrato cerrado.")
    else:
        if (
            result.state != SNAPSHOT_GENERATION_STATE_AVAILABLE
            or result.reason_codes != ()
        ):
            raise ValueError("Estado available fuera del contrato cerrado.")
    if (
        diagnostics.elements_inspected != result.inputs_received
        or diagnostics.results_validated != result.inputs_received
        or diagnostics.entries_built != result.entries_generated
        or diagnostics.builder_calls != 1
        or diagnostics.serializations < 1
        or diagnostics.canonical_bytes != result.serialized_bytes
    ):
        raise ValueError("Instrumentacion P14 inconsistente.")
    if result.policy.mode == SNAPSHOT_GENERATION_MODE_P10_OFFLINE:
        expected_estimate = -(-result.serialized_bytes // result.entries_generated) * (
            result.policy.required_entries
        )
        if diagnostics.estimated_universe_bytes != expected_estimate:
            raise ValueError("Estimacion de universo P14 inconsistente.")
    elif diagnostics.estimated_universe_bytes != 0:
        raise ValueError("Estimacion de universo P14 inconsistente.")
    payload_text = _generation_canonical_bytes(result).decode("utf-8")
    for entry in snapshot.entries:
        if entry.key.player in payload_text or entry.key.opponent in payload_text:
            raise ValueError("El resultado P14 fuga identidades privadas.")


__all__ = (
    "GENERATION_CONTRACT_NAME",
    "GENERATION_FINGERPRINT_DOMAIN",
    "GENERATION_REASON_CODES",
    "GENERATION_SCHEMA_VERSION",
    "GENERATION_STAGES",
    "GENERIC_SNAPSHOT_GENERATION_POLICY",
    "P10_OFFLINE_SNAPSHOT_GENERATION_POLICY",
    "P10OfflineSnapshotRecord",
    "SEALED_TEST_FIRST_DAY",
    "SNAPSHOT_GENERATION_MODE_GENERIC",
    "SNAPSHOT_GENERATION_MODE_P10_OFFLINE",
    "SNAPSHOT_GENERATION_STATE_AVAILABLE",
    "SNAPSHOT_GENERATION_STATE_EMPTY_INPUT",
    "SnapshotGenerationBuildError",
    "SnapshotGenerationDiagnostics",
    "SnapshotGenerationInputError",
    "SnapshotGenerationKeyError",
    "SnapshotGenerationLimitError",
    "SnapshotGenerationPolicy",
    "SnapshotGenerationPolicyError",
    "SnapshotGenerationTemporalError",
    "SnapshotGenerationUniverseError",
    "SnapshotGenerationUpstreamError",
    "TacticalRecommendationSnapshotGenerationError",
    "TacticalRecommendationSnapshotGenerationResult",
    "canonical_snapshot_generation_json",
    "generate_and_persist_tactical_recommendation_snapshot",
    "generate_tactical_recommendation_snapshot",
    "snapshot_generation_fingerprint",
    "validate_snapshot_generation_policy",
    "validate_tactical_recommendation_snapshot_generation_result",
)
