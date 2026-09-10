"""Evidencia tactica historica emparejada, pura y leakage-safe.

La observacion upstream representa un unico intento orientado hacia ``player``.
Su ``role`` permite derivar servidor y restador sin duplicar el intento. Este
modulo empareja evidencia descriptiva del ejecutor y del oponente expuesto; no
produce prioridades, puntuaciones ni decisiones tacticas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Final

from src.recommender.tactical_feature_encoder import (
    P02_FEATURE_NAMES,
    P04_FEATURE_NAMES,
    P05_FEATURE_NAMES,
    P06_FEATURE_NAMES,
    P09_FEATURE_NAMES,
    TacticalEncodingPolicy,
    TacticalFeatureSchema,
    build_tactical_feature_schema,
    feature_schema_fingerprint,
    validate_tactical_feature_schema,
)
from src.recommender.tactical_history_profiles import (
    HISTORY_CONTRACT_VERSION,
    TacticalHistoricalObservation,
    TacticalLabelAvailability,
    TacticalOutcomeLabel,
    TacticalPlayerRole,
    validate_tactical_historical_observation,
    wilson_interval as _history_wilson_interval,
)


MATCHUP_EVIDENCE_CONTRACT_VERSION: Final = "1.0.0"
MATCHUP_QUERY_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-matchup-query\x00"
CATEGORY_EVIDENCE_FINGERPRINT_DOMAIN: Final = (
    b"tennis-tactical-category-evidence\x00"
)
MATCHUP_CATEGORY_FINGERPRINT_DOMAIN: Final = (
    b"tennis-tactical-matchup-category-evidence\x00"
)
MATCHUP_EVIDENCE_FINGERPRINT_DOMAIN: Final = (
    b"tennis-tactical-matchup-evidence\x00"
)
PATTERN_ORDER: Final = ("P02", "P04", "P05", "P06", "P09")
PATTERN_FEATURES: Final = (
    ("P02", P02_FEATURE_NAMES),
    ("P04", P04_FEATURE_NAMES),
    ("P05", P05_FEATURE_NAMES),
    ("P06", P06_FEATURE_NAMES),
    ("P09", P09_FEATURE_NAMES),
)
PATTERN_PREFIXES: Final = (
    ("P02", "p02.first_serve_direction."),
    ("P04", "p04.return_direction."),
    ("P05", "p05.return_depth."),
    ("P06", "p06.return_shot_type."),
    ("P09", "p09.return_profile."),
)
EVIDENCE_PROVENANCE: Final = (
    ("source", "tactical_historical_observations_in_memory"),
    ("history_contract_version", HISTORY_CONTRACT_VERSION),
    ("temporal_policy", "strictly_before_as_of_date"),
    ("scope_policy", "global_only"),
)
CATEGORY_RECONCILIATIONS: Final = (
    ("counts_ordered", True),
    ("labels_separated", True),
    ("wilson_reconciled", True),
    ("distinct_matches_internal_only", True),
)
PAIR_RECONCILIATIONS: Final = (
    ("category_identity_reconciled", True),
    ("perspectives_opposed", True),
    ("scope_reconciled", True),
    ("joint_state_reconciled", True),
)
RESULT_RECONCILIATIONS: Final = (
    ("input_counts_exhaustive", True),
    ("attempt_keys_unique", True),
    ("strict_temporal_cut_reconciled", True),
    ("pattern_catalog_reconciled", True),
    ("category_order_reconciled", True),
    ("perspectives_reconciled", True),
    ("no_combined_redundant_total", True),
    ("public_output_contains_aggregates_only", True),
)

_SAFE_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_SHA256 = re.compile(r"^[0-9A-F]{64}$")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_PUBLIC_KEYS: Final = frozenset(
    {
        "match_id",
        "point_number",
        "sequence_text",
        "raw_sequence",
        "tokens",
        "spans",
        "warnings",
        "residual_text",
        "point_winner",
        "effective_date",
        "individual_outcome",
        "timestamp",
        "path",
    }
)
_FORBIDDEN_TERMS: Final = ("evaluation", "scoring", "recommendation")


class TacticalMatchupContractError(ValueError):
    """Error cerrado que no revela claves ni observaciones individuales."""


class TacticalEvidencePerspective(str, Enum):
    EXECUTOR = "executor"
    OPPONENT_ALLOWED = "opponent_allowed"


class TacticalEvidenceScopeStrategy(str, Enum):
    GLOBAL_ONLY = "global_only"


class TacticalEvidenceScope(str, Enum):
    GLOBAL = "global"


class TacticalCategoryEvidenceState(str, Enum):
    AVAILABLE = "available"
    INSUFFICIENT_LABELED_ATTEMPTS = "insufficient_labeled_attempts"
    INSUFFICIENT_MATCHES = "insufficient_matches"
    NO_OBSERVED_CATEGORY = "no_observed_category"
    NOT_APPLICABLE = "not_applicable"
    NOT_AVAILABLE = "not_available"


class TacticalMatchupCategoryState(str, Enum):
    COMPARABLE = "comparable"
    EXECUTOR_INSUFFICIENT = "executor_insufficient"
    OPPONENT_INSUFFICIENT = "opponent_insufficient"
    BOTH_INSUFFICIENT = "both_insufficient"
    NOT_APPLICABLE = "not_applicable"
    NOT_AVAILABLE = "not_available"


class TacticalMatchupEvidenceState(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class TacticalPatternOwnership:
    pattern_id: str
    documented_construct: str
    executor_role: TacticalPlayerRole
    exposed_role: TacticalPlayerRole
    tactical_success_label: TacticalOutcomeLabel

    def __post_init__(self) -> None:
        _validate_pattern_ownership(self)


def _validate_pattern_ownership(item: TacticalPatternOwnership) -> None:
    if type(item) is not TacticalPatternOwnership:
        raise TypeError("pattern ownership debe usar el tipo contractual exacto.")
    expected = {
        "P02": (
            "documented_first_serve_direction",
            TacticalPlayerRole.SERVER,
            TacticalPlayerRole.RETURNER,
            TacticalOutcomeLabel.SERVER_WON_POINT,
        ),
        "P04": (
            "documented_initial_return_lateral_direction",
            TacticalPlayerRole.RETURNER,
            TacticalPlayerRole.SERVER,
            TacticalOutcomeLabel.RETURNER_WON_POINT,
        ),
        "P05": (
            "documented_initial_return_depth",
            TacticalPlayerRole.RETURNER,
            TacticalPlayerRole.SERVER,
            TacticalOutcomeLabel.RETURNER_WON_POINT,
        ),
        "P06": (
            "documented_initial_return_shot_type",
            TacticalPlayerRole.RETURNER,
            TacticalPlayerRole.SERVER,
            TacticalOutcomeLabel.RETURNER_WON_POINT,
        ),
        "P09": (
            "documented_initial_return_profile",
            TacticalPlayerRole.RETURNER,
            TacticalPlayerRole.SERVER,
            TacticalOutcomeLabel.RETURNER_WON_POINT,
        ),
    }
    if type(item.pattern_id) is not str or item.pattern_id not in expected:
        raise TacticalMatchupContractError("pattern_id fuera del catalogo cerrado.")
    if (
        item.documented_construct,
        item.executor_role,
        item.exposed_role,
        item.tactical_success_label,
    ) != expected[item.pattern_id]:
        raise TacticalMatchupContractError(
            "La propiedad del patron no coincide con el contrato."
        )
    if type(item.documented_construct) is not str:
        raise TacticalMatchupContractError("documented_construct debe ser str exacto.")
    if type(item.executor_role) is not TacticalPlayerRole:
        raise TacticalMatchupContractError("executor_role fuera del dominio.")
    if type(item.exposed_role) is not TacticalPlayerRole:
        raise TacticalMatchupContractError("exposed_role fuera del dominio.")
    if type(item.tactical_success_label) is not TacticalOutcomeLabel:
        raise TacticalMatchupContractError("tactical_success_label fuera del dominio.")


PATTERN_OWNERSHIP_CATALOG: Final = (
    TacticalPatternOwnership(
        "P02",
        "documented_first_serve_direction",
        TacticalPlayerRole.SERVER,
        TacticalPlayerRole.RETURNER,
        TacticalOutcomeLabel.SERVER_WON_POINT,
    ),
    TacticalPatternOwnership(
        "P04",
        "documented_initial_return_lateral_direction",
        TacticalPlayerRole.RETURNER,
        TacticalPlayerRole.SERVER,
        TacticalOutcomeLabel.RETURNER_WON_POINT,
    ),
    TacticalPatternOwnership(
        "P05",
        "documented_initial_return_depth",
        TacticalPlayerRole.RETURNER,
        TacticalPlayerRole.SERVER,
        TacticalOutcomeLabel.RETURNER_WON_POINT,
    ),
    TacticalPatternOwnership(
        "P06",
        "documented_initial_return_shot_type",
        TacticalPlayerRole.RETURNER,
        TacticalPlayerRole.SERVER,
        TacticalOutcomeLabel.RETURNER_WON_POINT,
    ),
    TacticalPatternOwnership(
        "P09",
        "documented_initial_return_profile",
        TacticalPlayerRole.RETURNER,
        TacticalPlayerRole.SERVER,
        TacticalOutcomeLabel.RETURNER_WON_POINT,
    ),
)
@dataclass(frozen=True)
class TacticalMatchupQuery:
    contract_version: str
    player: str
    opponent: str
    as_of_date: date
    policy: TacticalEncodingPolicy
    schema_fingerprint: str
    serve_numbers: tuple[int, ...]
    window_days: int | None
    minimum_labeled_attempts: int
    minimum_matches: int
    requested_patterns: tuple[str, ...]
    scope_strategy: TacticalEvidenceScopeStrategy

    def __post_init__(self) -> None:
        validate_tactical_matchup_query(self)


@dataclass(frozen=True)
class TacticalCategoryEvidence:
    contract_version: str
    query: TacticalMatchupQuery
    query_fingerprint: str
    policy: TacticalEncodingPolicy
    schema_fingerprint: str
    pattern_id: str
    feature_name: str
    category: str
    perspective: TacticalEvidencePerspective
    scope_used: TacticalEvidenceScope
    executor_role: TacticalPlayerRole
    serve_numbers: tuple[int, ...]
    minimum_labeled_attempts: int
    minimum_matches: int
    relevant_attempt_count: int
    applicable_attempts: int
    observed_pattern_attempts: int
    category_activations: int
    labeled_activations: int
    tactical_successes: int
    tactical_failures: int
    tactical_success_rate: float | None
    wilson_lower: float | None
    wilson_upper: float | None
    distinct_match_count: int
    evidence_state: TacticalCategoryEvidenceState
    reason_codes: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...]
    reconciliations: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        validate_tactical_category_evidence(self)


@dataclass(frozen=True)
class TacticalMatchupCategoryEvidence:
    contract_version: str
    query_fingerprint: str
    policy: TacticalEncodingPolicy
    schema_fingerprint: str
    pattern_id: str
    feature_name: str
    category: str
    executor_evidence: TacticalCategoryEvidence
    opponent_allowed_evidence: TacticalCategoryEvidence
    matchup_state: TacticalMatchupCategoryState
    reason_codes: tuple[str, ...]
    reconciliations: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        validate_tactical_matchup_category_evidence(self)


@dataclass(frozen=True)
class TacticalMatchupEvidence:
    contract_version: str
    query: TacticalMatchupQuery
    policy: TacticalEncodingPolicy
    schema_fingerprint: str
    evidence_state: TacticalMatchupEvidenceState
    reason_codes: tuple[str, ...]
    categories: tuple[TacticalMatchupCategoryEvidence, ...]
    catalog_category_count: int
    candidate_category_count: int
    comparable_category_count: int
    insufficient_category_count: int
    state_counts: tuple[tuple[str, int], ...]
    total_observations_received: int
    unique_attempt_count: int
    evidence_eligible_attempt_count: int
    excluded_date_count: int
    excluded_window_count: int
    excluded_player_count: int
    excluded_serve_number_count: int
    excluded_pattern_policy_count: int
    redundant_representation: bool
    reconciliations: tuple[tuple[str, bool], ...]
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        validate_tactical_matchup_evidence(self)


def _strict_text(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _SAFE_TEXT.fullmatch(value) is None
    ):
        raise TacticalMatchupContractError(
            f"{field} debe ser str exacto, no vacio y sin espacios externos."
        )
    return value


def _strict_date(value: object, field: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{field} debe ser datetime.date exacta sin hora.")
    return value


def _strict_count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TacticalMatchupContractError(
            f"{field} debe ser int real no negativo."
        )
    return value


def _validate_version(value: object) -> None:
    if type(value) is not str or value != MATCHUP_EVIDENCE_CONTRACT_VERSION:
        raise TacticalMatchupContractError("Version contractual invalida.")


def _validate_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TacticalMatchupContractError(f"{field} debe ser SHA-256 mayusculo.")
    return value


def _validate_optional_float(value: object, field: str) -> None:
    if value is not None and (type(value) is not float or not isfinite(value)):
        raise TacticalMatchupContractError(f"{field} debe ser float finito o None.")


def tactical_wilson_interval(
    successes: int, trials: int, *, feature_name: str
) -> tuple[float | None, float | None, float | None]:
    """Aplica el contrato Wilson upstream sin exponer datos individuales."""
    try:
        return _history_wilson_interval(
            successes, trials, feature_name=feature_name
        )
    except (TypeError, ValueError) as exc:
        if (
            type(successes) is int
            and type(trials) is int
            and type(feature_name) is str
            and re.fullmatch(r"[A-Za-z0-9_.|]+", feature_name) is not None
        ):
            raise TacticalMatchupContractError(
                "Wilson no disponible para los conteos agregados "
                f"feature={feature_name};successes={successes};trials={trials}."
            ) from None
        raise TacticalMatchupContractError(
            "Parametros Wilson fuera del contrato exacto."
        ) from exc


def _validate_reason_codes(value: object, expected: tuple[str, ...]) -> None:
    if (
        type(value) is not tuple
        or any(type(reason) is not str for reason in value)
        or value != expected
    ):
        raise TacticalMatchupContractError("Reason codes no reconcilian.")


def _validate_pairs(
    value: object,
    expected: tuple[tuple[str, object], ...],
    *,
    value_type: type,
    field: str,
) -> None:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not value_type
        for item in value
    ):
        raise TacticalMatchupContractError(f"{field} debe ser profundamente inmutable.")
    if value != expected:
        raise TacticalMatchupContractError(f"{field} no reconcilia.")


def _ownership(pattern_id: str) -> TacticalPatternOwnership:
    for item in PATTERN_OWNERSHIP_CATALOG:
        if item.pattern_id == pattern_id:
            return item
    raise TacticalMatchupContractError("pattern_id fuera del catalogo cerrado.")


def _policy_patterns(policy: TacticalEncodingPolicy) -> tuple[str, ...]:
    return build_tactical_feature_schema(policy).encoded_pattern_order


def validate_tactical_matchup_query(query: TacticalMatchupQuery) -> None:
    if type(query) is not TacticalMatchupQuery:
        raise TypeError("query debe ser TacticalMatchupQuery exacta.")
    _validate_version(query.contract_version)
    player = _strict_text(query.player, "player")
    opponent = _strict_text(query.opponent, "opponent")
    if player == opponent:
        raise TacticalMatchupContractError("player y opponent deben ser distintos.")
    as_of = _strict_date(query.as_of_date, "as_of_date")
    if type(query.policy) is not TacticalEncodingPolicy:
        raise TypeError("policy debe ser TacticalEncodingPolicy exacta.")
    schema = build_tactical_feature_schema(query.policy)
    expected_schema = feature_schema_fingerprint(schema)
    if (
        _validate_sha256(query.schema_fingerprint, "schema_fingerprint")
        != expected_schema
    ):
        raise TacticalMatchupContractError("schema_fingerprint no reconcilia.")
    if (
        type(query.serve_numbers) is not tuple
        or any(type(item) is not int for item in query.serve_numbers)
        or query.serve_numbers not in {(1,), (2,), (1, 2)}
    ):
        raise TacticalMatchupContractError(
            "serve_numbers debe ser (1,), (2,) o (1, 2)."
        )
    if query.window_days is not None:
        window = _strict_count(query.window_days, "window_days")
        if window == 0:
            raise TacticalMatchupContractError("window_days debe ser positivo.")
        if window > (as_of - date.min).days:
            raise TacticalMatchupContractError("window_days excede el calendario.")
    _strict_count(query.minimum_labeled_attempts, "minimum_labeled_attempts")
    _strict_count(query.minimum_matches, "minimum_matches")
    if type(query.requested_patterns) is not tuple or not query.requested_patterns:
        raise TacticalMatchupContractError(
            "requested_patterns debe ser tuple no vacia."
        )
    if any(type(item) is not str or item not in PATTERN_ORDER for item in query.requested_patterns):
        raise TacticalMatchupContractError("requested_patterns contiene un patron invalido.")
    expected_order = tuple(
        item for item in PATTERN_ORDER if item in query.requested_patterns
    )
    if query.requested_patterns != expected_order:
        raise TacticalMatchupContractError(
            "requested_patterns debe ser unico y usar orden canonico."
        )
    allowed = _policy_patterns(query.policy)
    if any(pattern not in allowed for pattern in query.requested_patterns):
        raise TacticalMatchupContractError(
            "requested_patterns no es compatible con la policy."
        )
    if type(query.scope_strategy) is not TacticalEvidenceScopeStrategy:
        raise TypeError("scope_strategy fuera del dominio cerrado.")
    if query.scope_strategy is not TacticalEvidenceScopeStrategy.GLOBAL_ONLY:
        raise TacticalMatchupContractError("Solo global_only esta implementado.")


def _feature_catalog(
    schema: TacticalFeatureSchema, requested_patterns: tuple[str, ...]
) -> tuple[tuple[str, str, str], ...]:
    prefixes = dict(PATTERN_PREFIXES)
    output = []
    for pattern_id in requested_patterns:
        prefix = prefixes[pattern_id]
        names = tuple(
            name
            for name in schema.tactical_feature_names
            if name.startswith(prefix)
        )
        if not names:
            raise TacticalMatchupContractError(
                "El schema no contiene el catalogo solicitado."
            )
        output.extend((pattern_id, name, name[len(prefix) :]) for name in names)
    return tuple(output)


def _derived_actors(
    observation: TacticalHistoricalObservation,
) -> tuple[str, str]:
    if observation.role is TacticalPlayerRole.SERVER:
        return observation.player, observation.opponent
    return observation.opponent, observation.player


def _pattern_actors(
    observation: TacticalHistoricalObservation, pattern_id: str
) -> tuple[str, str]:
    server, returner = _derived_actors(observation)
    if _ownership(pattern_id).executor_role is TacticalPlayerRole.SERVER:
        return server, returner
    return returner, server


def _validate_observations(
    observations: object,
) -> tuple[TacticalHistoricalObservation, ...]:
    if type(observations) is not tuple:
        raise TypeError("observations debe ser tuple inmutable.")
    keys: set[tuple[str, int, int]] = set()
    for observation in observations:
        validate_tactical_historical_observation(observation)
        key = (observation.match_id, observation.point_number, observation.serve_number)
        if key in keys:
            raise TacticalMatchupContractError(
                "Existe una clave de intento duplicada."
            )
        keys.add(key)
    return tuple(
        sorted(
            observations,
            key=lambda item: (
                item.effective_date,
                item.match_id,
                item.point_number,
                item.serve_number,
            ),
        )
    )


def _evidence_state(
    *,
    structurally_applicable: bool,
    relevant_attempt_count: int,
    observed_count: int,
    active_count: int,
    labeled_count: int,
    match_count: int,
    minimum_labeled_attempts: int,
    minimum_matches: int,
) -> tuple[TacticalCategoryEvidenceState, tuple[str, ...]]:
    if not structurally_applicable:
        return TacticalCategoryEvidenceState.NOT_APPLICABLE, (
            "pattern_not_applicable_to_requested_serves",
        )
    if relevant_attempt_count == 0:
        return TacticalCategoryEvidenceState.NOT_AVAILABLE, (
            "no_relevant_historical_attempts",
        )
    if observed_count == 0 or active_count == 0:
        return TacticalCategoryEvidenceState.NO_OBSERVED_CATEGORY, (
            "category_not_observed",
        )
    if labeled_count < minimum_labeled_attempts:
        return TacticalCategoryEvidenceState.INSUFFICIENT_LABELED_ATTEMPTS, (
            "minimum_labeled_attempts_not_met",
        )
    if match_count < minimum_matches:
        return TacticalCategoryEvidenceState.INSUFFICIENT_MATCHES, (
            "minimum_matches_not_met",
        )
    return TacticalCategoryEvidenceState.AVAILABLE, ("evidence_minimums_met",)


def _build_category_evidence(
    *,
    observations: tuple[TacticalHistoricalObservation, ...],
    query: TacticalMatchupQuery,
    schema: TacticalFeatureSchema,
    pattern_id: str,
    feature_name: str,
    category: str,
    perspective: TacticalEvidencePerspective,
) -> TacticalCategoryEvidence:
    ownership = _ownership(pattern_id)
    if perspective is TacticalEvidencePerspective.EXECUTOR:
        relevant = tuple(
            item
            for item in observations
            if _pattern_actors(item, pattern_id)[0] == query.player
        )
    else:
        relevant = tuple(
            item
            for item in observations
            if _pattern_actors(item, pattern_id)[1] == query.opponent
        )
    index = dict(schema.name_to_index)
    applicable_name = f"mask.{pattern_id.lower()}.applicable"
    observed_name = f"mask.{pattern_id.lower()}.observed"
    applicable_count = sum(
        item.feature_vector.values[index[applicable_name]] for item in relevant
    )
    observed_count = sum(
        item.feature_vector.values[index[observed_name]] for item in relevant
    )
    active = tuple(
        item for item in relevant if item.feature_vector.values[index[feature_name]] == 1
    )
    labeled = tuple(
        item
        for item in active
        if item.label_availability is TacticalLabelAvailability.AVAILABLE
    )
    successes = sum(item.label is ownership.tactical_success_label for item in labeled)
    failures = len(labeled) - successes
    rate, lower, upper = tactical_wilson_interval(
        successes, len(labeled), feature_name=feature_name
    )
    match_count = len({item.match_id for item in active})
    structurally_applicable = not (
        pattern_id == "P02" and 1 not in query.serve_numbers
    )
    state, reasons = _evidence_state(
        structurally_applicable=structurally_applicable,
        relevant_attempt_count=len(relevant),
        observed_count=observed_count,
        active_count=len(active),
        labeled_count=len(labeled),
        match_count=match_count,
        minimum_labeled_attempts=query.minimum_labeled_attempts,
        minimum_matches=query.minimum_matches,
    )
    return TacticalCategoryEvidence(
        contract_version=MATCHUP_EVIDENCE_CONTRACT_VERSION,
        query=query,
        query_fingerprint=tactical_matchup_query_fingerprint(query),
        policy=query.policy,
        schema_fingerprint=query.schema_fingerprint,
        pattern_id=pattern_id,
        feature_name=feature_name,
        category=category,
        perspective=perspective,
        scope_used=TacticalEvidenceScope.GLOBAL,
        executor_role=ownership.executor_role,
        serve_numbers=query.serve_numbers,
        minimum_labeled_attempts=query.minimum_labeled_attempts,
        minimum_matches=query.minimum_matches,
        relevant_attempt_count=len(relevant),
        applicable_attempts=applicable_count,
        observed_pattern_attempts=observed_count,
        category_activations=len(active),
        labeled_activations=len(labeled),
        tactical_successes=successes,
        tactical_failures=failures,
        tactical_success_rate=rate,
        wilson_lower=lower,
        wilson_upper=upper,
        distinct_match_count=match_count,
        evidence_state=state,
        reason_codes=reasons,
        provenance=EVIDENCE_PROVENANCE,
        reconciliations=CATEGORY_RECONCILIATIONS,
    )


def _expected_category_state(
    evidence: TacticalCategoryEvidence,
) -> tuple[TacticalCategoryEvidenceState, tuple[str, ...]]:
    structurally_applicable = evidence.pattern_id != "P02" or 1 in evidence.serve_numbers
    return _evidence_state(
        structurally_applicable=structurally_applicable,
        relevant_attempt_count=evidence.relevant_attempt_count,
        observed_count=evidence.observed_pattern_attempts,
        active_count=evidence.category_activations,
        labeled_count=evidence.labeled_activations,
        match_count=evidence.distinct_match_count,
        minimum_labeled_attempts=evidence.minimum_labeled_attempts,
        minimum_matches=evidence.minimum_matches,
    )


def validate_tactical_category_evidence(evidence: TacticalCategoryEvidence) -> None:
    if type(evidence) is not TacticalCategoryEvidence:
        raise TypeError("evidence debe ser TacticalCategoryEvidence exacta.")
    _validate_version(evidence.contract_version)
    validate_tactical_matchup_query(evidence.query)
    if (
        _validate_sha256(evidence.query_fingerprint, "query_fingerprint")
        != tactical_matchup_query_fingerprint(evidence.query)
    ):
        raise TacticalMatchupContractError("query_fingerprint no reconcilia.")
    if type(evidence.policy) is not TacticalEncodingPolicy:
        raise TypeError("policy de evidence fuera del dominio.")
    schema = build_tactical_feature_schema(evidence.policy)
    if (
        _validate_sha256(evidence.schema_fingerprint, "schema_fingerprint")
        != feature_schema_fingerprint(schema)
    ):
        raise TacticalMatchupContractError("schema_fingerprint no reconcilia.")
    if (
        evidence.policy is not evidence.query.policy
        or evidence.schema_fingerprint != evidence.query.schema_fingerprint
        or evidence.serve_numbers != evidence.query.serve_numbers
        or evidence.minimum_labeled_attempts
        != evidence.query.minimum_labeled_attempts
        or evidence.minimum_matches != evidence.query.minimum_matches
        or evidence.pattern_id not in evidence.query.requested_patterns
    ):
        raise TacticalMatchupContractError(
            "Evidence no reconcilia con los parametros de su query."
        )
    ownership = _ownership(evidence.pattern_id)
    catalog = dict(PATTERN_FEATURES)[evidence.pattern_id]
    if type(evidence.feature_name) is not str or evidence.feature_name not in catalog:
        raise TacticalMatchupContractError("feature_name fuera del catalogo del patron.")
    prefix = dict(PATTERN_PREFIXES)[evidence.pattern_id]
    expected_category = evidence.feature_name[len(prefix) :]
    if type(evidence.category) is not str or evidence.category != expected_category:
        raise TacticalMatchupContractError("category no reconcilia con feature_name.")
    if type(evidence.perspective) is not TacticalEvidencePerspective:
        raise TypeError("perspective fuera del dominio.")
    if type(evidence.scope_used) is not TacticalEvidenceScope or evidence.scope_used is not TacticalEvidenceScope.GLOBAL:
        raise TacticalMatchupContractError("scope_used debe ser global.")
    if type(evidence.executor_role) is not TacticalPlayerRole or evidence.executor_role is not ownership.executor_role:
        raise TacticalMatchupContractError("executor_role no reconcilia con el patron.")
    if (
        type(evidence.serve_numbers) is not tuple
        or any(type(item) is not int for item in evidence.serve_numbers)
        or evidence.serve_numbers not in {(1,), (2,), (1, 2)}
    ):
        raise TacticalMatchupContractError("serve_numbers de evidence invalido.")
    for field in (
        "minimum_labeled_attempts",
        "minimum_matches",
        "relevant_attempt_count",
        "applicable_attempts",
        "observed_pattern_attempts",
        "category_activations",
        "labeled_activations",
        "tactical_successes",
        "tactical_failures",
        "distinct_match_count",
    ):
        _strict_count(getattr(evidence, field), field)
    if not (
        evidence.category_activations
        <= evidence.observed_pattern_attempts
        <= evidence.applicable_attempts
        <= evidence.relevant_attempt_count
    ):
        raise TacticalMatchupContractError("Denominadores de categoria fuera de orden.")
    if evidence.labeled_activations > evidence.category_activations:
        raise TacticalMatchupContractError("Labels superan activaciones.")
    if evidence.tactical_successes + evidence.tactical_failures != evidence.labeled_activations:
        raise TacticalMatchupContractError("Successes y failures no reconcilian trials.")
    if (
        evidence.distinct_match_count > evidence.category_activations
        or (evidence.category_activations == 0) != (evidence.distinct_match_count == 0)
    ):
        raise TacticalMatchupContractError(
            "Partidos distintos no reconcilian con activaciones."
        )
    if evidence.pattern_id == "P02" and 1 not in evidence.serve_numbers and any(
        (
            evidence.applicable_attempts,
            evidence.observed_pattern_attempts,
            evidence.category_activations,
            evidence.labeled_activations,
            evidence.tactical_successes,
            evidence.tactical_failures,
            evidence.distinct_match_count,
        )
    ):
        raise TacticalMatchupContractError(
            "P02 no puede publicar evidencia de segundo saque."
        )
    for field in ("tactical_success_rate", "wilson_lower", "wilson_upper"):
        _validate_optional_float(getattr(evidence, field), field)
    rate, lower, upper = tactical_wilson_interval(
        evidence.tactical_successes,
        evidence.labeled_activations,
        feature_name=evidence.feature_name,
    )
    if (
        evidence.tactical_success_rate != rate
        or evidence.wilson_lower != lower
        or evidence.wilson_upper != upper
    ):
        raise TacticalMatchupContractError("Tasa o Wilson no reconcilia.")
    if type(evidence.evidence_state) is not TacticalCategoryEvidenceState:
        raise TypeError("evidence_state fuera del dominio.")
    expected_state, expected_reasons = _expected_category_state(evidence)
    if evidence.evidence_state is not expected_state:
        raise TacticalMatchupContractError("evidence_state no reconcilia.")
    _validate_reason_codes(evidence.reason_codes, expected_reasons)
    _validate_pairs(
        evidence.provenance,
        EVIDENCE_PROVENANCE,
        value_type=str,
        field="provenance",
    )
    _validate_pairs(
        evidence.reconciliations,
        CATEGORY_RECONCILIATIONS,
        value_type=bool,
        field="reconciliations",
    )


def _joint_state(
    executor: TacticalCategoryEvidence,
    opponent: TacticalCategoryEvidence,
) -> tuple[TacticalMatchupCategoryState, tuple[str, ...]]:
    executor_available = executor.evidence_state is TacticalCategoryEvidenceState.AVAILABLE
    opponent_available = opponent.evidence_state is TacticalCategoryEvidenceState.AVAILABLE
    if executor_available and opponent_available:
        return TacticalMatchupCategoryState.COMPARABLE, ("both_perspectives_available",)
    if (
        executor.evidence_state is TacticalCategoryEvidenceState.NOT_APPLICABLE
        and opponent.evidence_state is TacticalCategoryEvidenceState.NOT_APPLICABLE
    ):
        return TacticalMatchupCategoryState.NOT_APPLICABLE, (
            "pattern_not_applicable_to_requested_serves",
        )
    if (
        executor.evidence_state is TacticalCategoryEvidenceState.NOT_AVAILABLE
        and opponent.evidence_state is TacticalCategoryEvidenceState.NOT_AVAILABLE
    ):
        return TacticalMatchupCategoryState.NOT_AVAILABLE, (
            "both_perspectives_not_available",
        )
    if executor_available:
        return TacticalMatchupCategoryState.OPPONENT_INSUFFICIENT, (
            "opponent_allowed_evidence_insufficient",
        )
    if opponent_available:
        return TacticalMatchupCategoryState.EXECUTOR_INSUFFICIENT, (
            "executor_evidence_insufficient",
        )
    return TacticalMatchupCategoryState.BOTH_INSUFFICIENT, (
        "both_perspectives_insufficient",
    )


def _pair_evidence(
    executor: TacticalCategoryEvidence,
    opponent: TacticalCategoryEvidence,
) -> TacticalMatchupCategoryEvidence:
    state, reasons = _joint_state(executor, opponent)
    return TacticalMatchupCategoryEvidence(
        contract_version=MATCHUP_EVIDENCE_CONTRACT_VERSION,
        query_fingerprint=executor.query_fingerprint,
        policy=executor.policy,
        schema_fingerprint=executor.schema_fingerprint,
        pattern_id=executor.pattern_id,
        feature_name=executor.feature_name,
        category=executor.category,
        executor_evidence=executor,
        opponent_allowed_evidence=opponent,
        matchup_state=state,
        reason_codes=reasons,
        reconciliations=PAIR_RECONCILIATIONS,
    )


def validate_tactical_matchup_category_evidence(
    evidence: TacticalMatchupCategoryEvidence,
) -> None:
    if type(evidence) is not TacticalMatchupCategoryEvidence:
        raise TypeError("evidence debe ser TacticalMatchupCategoryEvidence exacta.")
    _validate_version(evidence.contract_version)
    _validate_sha256(evidence.query_fingerprint, "query_fingerprint")
    if type(evidence.policy) is not TacticalEncodingPolicy:
        raise TypeError("policy de pareja fuera del dominio.")
    schema = build_tactical_feature_schema(evidence.policy)
    if (
        _validate_sha256(evidence.schema_fingerprint, "schema_fingerprint")
        != feature_schema_fingerprint(schema)
    ):
        raise TacticalMatchupContractError("schema_fingerprint de pareja invalido.")
    if (
        type(evidence.pattern_id) is not str
        or type(evidence.feature_name) is not str
        or type(evidence.category) is not str
    ):
        raise TacticalMatchupContractError(
            "La identidad de categoria debe usar strings exactos."
        )
    validate_tactical_category_evidence(evidence.executor_evidence)
    validate_tactical_category_evidence(evidence.opponent_allowed_evidence)
    executor = evidence.executor_evidence
    opponent = evidence.opponent_allowed_evidence
    identity = (
        evidence.query_fingerprint,
        evidence.policy,
        evidence.schema_fingerprint,
        evidence.pattern_id,
        evidence.feature_name,
        evidence.category,
    )
    if identity != (
        executor.query_fingerprint,
        executor.policy,
        executor.schema_fingerprint,
        executor.pattern_id,
        executor.feature_name,
        executor.category,
    ) or identity != (
        opponent.query_fingerprint,
        opponent.policy,
        opponent.schema_fingerprint,
        opponent.pattern_id,
        opponent.feature_name,
        opponent.category,
    ):
        raise TacticalMatchupContractError("Identidad de categoria no reconcilia.")
    if executor.perspective is not TacticalEvidencePerspective.EXECUTOR:
        raise TacticalMatchupContractError("Falta la perspectiva executor.")
    if opponent.perspective is not TacticalEvidencePerspective.OPPONENT_ALLOWED:
        raise TacticalMatchupContractError("Falta opponent_allowed.")
    if executor.scope_used is not opponent.scope_used:
        raise TacticalMatchupContractError("Las perspectivas no comparten scope.")
    if executor.query != opponent.query:
        raise TacticalMatchupContractError("Las perspectivas no comparten query exacta.")
    state, reasons = _joint_state(executor, opponent)
    if type(evidence.matchup_state) is not TacticalMatchupCategoryState or evidence.matchup_state is not state:
        raise TacticalMatchupContractError("matchup_state no reconcilia.")
    _validate_reason_codes(evidence.reason_codes, reasons)
    _validate_pairs(
        evidence.reconciliations,
        PAIR_RECONCILIATIONS,
        value_type=bool,
        field="reconciliations",
    )


def _result_state(
    categories: tuple[TacticalMatchupCategoryEvidence, ...],
) -> tuple[TacticalMatchupEvidenceState, tuple[str, ...], int, int]:
    candidates = tuple(
        item
        for item in categories
        if item.executor_evidence.category_activations > 0
        or item.opponent_allowed_evidence.category_activations > 0
    )
    comparable = sum(
        item.matchup_state is TacticalMatchupCategoryState.COMPARABLE
        for item in candidates
    )
    if candidates and comparable == len(candidates):
        return (
            TacticalMatchupEvidenceState.AVAILABLE,
            ("all_observed_candidate_categories_comparable",),
            len(candidates),
            comparable,
        )
    if comparable:
        return (
            TacticalMatchupEvidenceState.PARTIALLY_AVAILABLE,
            ("some_observed_candidate_categories_comparable",),
            len(candidates),
            comparable,
        )
    return (
        TacticalMatchupEvidenceState.NOT_AVAILABLE,
        ("no_observed_candidate_category_comparable",),
        len(candidates),
        0,
    )


def _state_counts(
    categories: tuple[TacticalMatchupCategoryEvidence, ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            state.value,
            sum(item.matchup_state is state for item in categories),
        )
        for state in TacticalMatchupCategoryState
    )


def build_tactical_matchup_evidence(
    observations: tuple[TacticalHistoricalObservation, ...],
    query: TacticalMatchupQuery,
    schema: TacticalFeatureSchema,
) -> TacticalMatchupEvidence:
    """Construye evidencia global usando solo intentos anteriores al corte."""
    ordered = _validate_observations(observations)
    validate_tactical_matchup_query(query)
    if type(schema) is not TacticalFeatureSchema:
        raise TypeError("schema debe ser TacticalFeatureSchema exacto.")
    validate_tactical_feature_schema(schema)
    schema_fingerprint = feature_schema_fingerprint(schema)
    if query.policy is not schema.policy or query.schema_fingerprint != schema_fingerprint:
        raise TacticalMatchupContractError("Query y schema no coinciden.")
    for observation in ordered:
        if (
            observation.feature_vector.policy is not schema.policy
            or observation.feature_vector.schema_fingerprint != schema_fingerprint
        ):
            raise TacticalMatchupContractError(
                "Todas las observaciones deben compartir schema y policy."
            )

    excluded_date = tuple(item for item in ordered if item.effective_date >= query.as_of_date)
    before_cut = tuple(item for item in ordered if item.effective_date < query.as_of_date)
    excluded_window: tuple[TacticalHistoricalObservation, ...] = ()
    if query.window_days is not None:
        lower = query.as_of_date - timedelta(days=query.window_days)
        excluded_window = tuple(item for item in before_cut if item.effective_date < lower)
        before_cut = tuple(item for item in before_cut if item.effective_date >= lower)
    excluded_serve = tuple(
        item for item in before_cut if item.serve_number not in query.serve_numbers
    )
    after_serve = tuple(
        item for item in before_cut if item.serve_number in query.serve_numbers
    )

    def involves_query(item: TacticalHistoricalObservation) -> bool:
        return any(
            _pattern_actors(item, pattern_id)[0] == query.player
            or _pattern_actors(item, pattern_id)[1] == query.opponent
            for pattern_id in query.requested_patterns
        )

    excluded_player = tuple(item for item in after_serve if not involves_query(item))
    after_player = tuple(item for item in after_serve if involves_query(item))

    def relevant_pattern_role(item: TacticalHistoricalObservation) -> bool:
        return any(
            item.feature_vector.values[
                dict(schema.name_to_index)[f"mask.{pattern_id.lower()}.applicable"]
            ]
            for pattern_id in query.requested_patterns
        )

    excluded_pattern = tuple(
        item for item in after_player if not relevant_pattern_role(item)
    )
    eligible = tuple(item for item in after_player if relevant_pattern_role(item))

    categories = []
    for pattern_id, feature_name, category in _feature_catalog(
        schema, query.requested_patterns
    ):
        executor = _build_category_evidence(
            observations=eligible,
            query=query,
            schema=schema,
            pattern_id=pattern_id,
            feature_name=feature_name,
            category=category,
            perspective=TacticalEvidencePerspective.EXECUTOR,
        )
        opponent = _build_category_evidence(
            observations=eligible,
            query=query,
            schema=schema,
            pattern_id=pattern_id,
            feature_name=feature_name,
            category=category,
            perspective=TacticalEvidencePerspective.OPPONENT_ALLOWED,
        )
        categories.append(_pair_evidence(executor, opponent))
    category_tuple = tuple(categories)
    state, reasons, candidate_count, comparable_count = _result_state(category_tuple)
    return TacticalMatchupEvidence(
        contract_version=MATCHUP_EVIDENCE_CONTRACT_VERSION,
        query=query,
        policy=query.policy,
        schema_fingerprint=query.schema_fingerprint,
        evidence_state=state,
        reason_codes=reasons,
        categories=category_tuple,
        catalog_category_count=len(category_tuple),
        candidate_category_count=candidate_count,
        comparable_category_count=comparable_count,
        insufficient_category_count=candidate_count - comparable_count,
        state_counts=_state_counts(category_tuple),
        total_observations_received=len(ordered),
        unique_attempt_count=len(ordered),
        evidence_eligible_attempt_count=len(eligible),
        excluded_date_count=len(excluded_date),
        excluded_window_count=len(excluded_window),
        excluded_player_count=len(excluded_player),
        excluded_serve_number_count=len(excluded_serve),
        excluded_pattern_policy_count=len(excluded_pattern),
        redundant_representation=schema.redundant_representation,
        reconciliations=RESULT_RECONCILIATIONS,
        provenance=EVIDENCE_PROVENANCE,
    )


def validate_tactical_matchup_evidence(evidence: TacticalMatchupEvidence) -> None:
    if type(evidence) is not TacticalMatchupEvidence:
        raise TypeError("evidence debe ser TacticalMatchupEvidence exacta.")
    _validate_version(evidence.contract_version)
    validate_tactical_matchup_query(evidence.query)
    if type(evidence.policy) is not TacticalEncodingPolicy or evidence.policy is not evidence.query.policy:
        raise TacticalMatchupContractError("policy del resultado no reconcilia.")
    if (
        _validate_sha256(evidence.schema_fingerprint, "schema_fingerprint")
        != evidence.query.schema_fingerprint
    ):
        raise TacticalMatchupContractError("schema_fingerprint del resultado invalido.")
    schema = build_tactical_feature_schema(evidence.policy)
    expected_catalog = _feature_catalog(schema, evidence.query.requested_patterns)
    if type(evidence.categories) is not tuple:
        raise TacticalMatchupContractError("categories debe ser tuple inmutable.")
    for category in evidence.categories:
        validate_tactical_matchup_category_evidence(category)
        if (
            category.query_fingerprint
            != tactical_matchup_query_fingerprint(evidence.query)
            or category.executor_evidence.query != evidence.query
            or category.opponent_allowed_evidence.query != evidence.query
        ):
            raise TacticalMatchupContractError("Categoria vinculada a otra query.")
    actual_catalog = tuple(
        (item.pattern_id, item.feature_name, item.category)
        for item in evidence.categories
    )
    if actual_catalog != expected_catalog:
        raise TacticalMatchupContractError("Catalogo u orden de categorias invalido.")
    for field in (
        "catalog_category_count",
        "candidate_category_count",
        "comparable_category_count",
        "insufficient_category_count",
        "total_observations_received",
        "unique_attempt_count",
        "evidence_eligible_attempt_count",
        "excluded_date_count",
        "excluded_window_count",
        "excluded_player_count",
        "excluded_serve_number_count",
        "excluded_pattern_policy_count",
    ):
        _strict_count(getattr(evidence, field), field)
    if evidence.catalog_category_count != len(evidence.categories):
        raise TacticalMatchupContractError("catalog_category_count no reconcilia.")
    if evidence.unique_attempt_count != evidence.total_observations_received:
        raise TacticalMatchupContractError("unique_attempt_count no reconcilia.")
    if evidence.total_observations_received != (
        evidence.evidence_eligible_attempt_count
        + evidence.excluded_date_count
        + evidence.excluded_window_count
        + evidence.excluded_player_count
        + evidence.excluded_serve_number_count
        + evidence.excluded_pattern_policy_count
    ):
        raise TacticalMatchupContractError("Conteos de exclusion no son exhaustivos.")
    state, reasons, candidates, comparable = _result_state(evidence.categories)
    if type(evidence.evidence_state) is not TacticalMatchupEvidenceState or evidence.evidence_state is not state:
        raise TacticalMatchupContractError("evidence_state global no reconcilia.")
    _validate_reason_codes(evidence.reason_codes, reasons)
    if (
        evidence.candidate_category_count != candidates
        or evidence.comparable_category_count != comparable
        or evidence.insufficient_category_count != candidates - comparable
    ):
        raise TacticalMatchupContractError("Conteos de comparabilidad no reconcilian.")
    expected_counts = _state_counts(evidence.categories)
    _validate_pairs(evidence.state_counts, expected_counts, value_type=int, field="state_counts")
    if type(evidence.redundant_representation) is not bool or evidence.redundant_representation != schema.redundant_representation:
        raise TacticalMatchupContractError("redundant_representation no reconcilia.")
    _validate_pairs(
        evidence.reconciliations,
        RESULT_RECONCILIATIONS,
        value_type=bool,
        field="reconciliations",
    )
    _validate_pairs(
        evidence.provenance,
        EVIDENCE_PROVENANCE,
        value_type=str,
        field="provenance",
    )


def _query_structure(query: TacticalMatchupQuery) -> dict[str, object]:
    validate_tactical_matchup_query(query)
    return {
        "as_of_date": query.as_of_date.isoformat(),
        "contract_version": query.contract_version,
        "minimum_labeled_attempts": query.minimum_labeled_attempts,
        "minimum_matches": query.minimum_matches,
        "opponent": query.opponent,
        "player": query.player,
        "policy": query.policy.value,
        "requested_patterns": list(query.requested_patterns),
        "schema_fingerprint": query.schema_fingerprint,
        "scope_strategy": query.scope_strategy.value,
        "serve_numbers": list(query.serve_numbers),
        "window_days": query.window_days,
    }


def _category_structure(evidence: TacticalCategoryEvidence) -> dict[str, object]:
    validate_tactical_category_evidence(evidence)
    return {
        "applicable_attempts": evidence.applicable_attempts,
        "category": evidence.category,
        "category_activations": evidence.category_activations,
        "contract_version": evidence.contract_version,
        "distinct_match_count": evidence.distinct_match_count,
        "evidence_state": evidence.evidence_state.value,
        "executor_role": evidence.executor_role.value,
        "feature_name": evidence.feature_name,
        "labeled_activations": evidence.labeled_activations,
        "minimum_labeled_attempts": evidence.minimum_labeled_attempts,
        "minimum_matches": evidence.minimum_matches,
        "observed_pattern_attempts": evidence.observed_pattern_attempts,
        "pattern_id": evidence.pattern_id,
        "perspective": evidence.perspective.value,
        "policy": evidence.policy.value,
        "provenance": {key: value for key, value in evidence.provenance},
        "query_fingerprint": evidence.query_fingerprint,
        "reason_codes": list(evidence.reason_codes),
        "reconciliations": {key: value for key, value in evidence.reconciliations},
        "relevant_attempt_count": evidence.relevant_attempt_count,
        "schema_fingerprint": evidence.schema_fingerprint,
        "scope_used": evidence.scope_used.value,
        "serve_numbers": list(evidence.serve_numbers),
        "tactical_failures": evidence.tactical_failures,
        "tactical_success_rate": evidence.tactical_success_rate,
        "tactical_successes": evidence.tactical_successes,
        "wilson_lower": evidence.wilson_lower,
        "wilson_upper": evidence.wilson_upper,
    }


def _pair_structure(evidence: TacticalMatchupCategoryEvidence) -> dict[str, object]:
    validate_tactical_matchup_category_evidence(evidence)
    return {
        "category": evidence.category,
        "contract_version": evidence.contract_version,
        "executor_evidence": _category_structure(evidence.executor_evidence),
        "feature_name": evidence.feature_name,
        "matchup_state": evidence.matchup_state.value,
        "opponent_allowed_evidence": _category_structure(
            evidence.opponent_allowed_evidence
        ),
        "pattern_id": evidence.pattern_id,
        "policy": evidence.policy.value,
        "query_fingerprint": evidence.query_fingerprint,
        "reason_codes": list(evidence.reason_codes),
        "reconciliations": {key: value for key, value in evidence.reconciliations},
        "schema_fingerprint": evidence.schema_fingerprint,
    }


def _result_structure(evidence: TacticalMatchupEvidence) -> dict[str, object]:
    validate_tactical_matchup_evidence(evidence)
    return {
        "candidate_category_count": evidence.candidate_category_count,
        "catalog_category_count": evidence.catalog_category_count,
        "categories": [_pair_structure(item) for item in evidence.categories],
        "comparable_category_count": evidence.comparable_category_count,
        "contract_version": evidence.contract_version,
        "evidence_eligible_attempt_count": evidence.evidence_eligible_attempt_count,
        "evidence_state": evidence.evidence_state.value,
        "excluded_date_count": evidence.excluded_date_count,
        "excluded_pattern_policy_count": evidence.excluded_pattern_policy_count,
        "excluded_player_count": evidence.excluded_player_count,
        "excluded_serve_number_count": evidence.excluded_serve_number_count,
        "excluded_window_count": evidence.excluded_window_count,
        "insufficient_category_count": evidence.insufficient_category_count,
        "policy": evidence.policy.value,
        "provenance": {key: value for key, value in evidence.provenance},
        "query": _query_structure(evidence.query),
        "reason_codes": list(evidence.reason_codes),
        "reconciliations": {key: value for key, value in evidence.reconciliations},
        "redundant_representation": evidence.redundant_representation,
        "schema_fingerprint": evidence.schema_fingerprint,
        "state_counts": {key: value for key, value in evidence.state_counts},
        "total_observations_received": evidence.total_observations_received,
        "unique_attempt_count": evidence.unique_attempt_count,
    }


def _validate_public_tree(value: object, *, key: str | None = None) -> None:
    if key is not None:
        lowered = key.lower()
        if (
            lowered in _FORBIDDEN_PUBLIC_KEYS
            or lowered.startswith("test_")
            or any(term in lowered for term in _FORBIDDEN_TERMS)
        ):
            raise TacticalMatchupContractError(
                "La salida publica contiene una clave no autorizada."
            )
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise TacticalMatchupContractError("La salida contiene un valor no finito.")
        return
    if type(value) is str:
        lowered = value.lower()
        if (
            "file://" in lowered
            or _WINDOWS_PATH.match(value)
            or value.startswith(("/", "\\\\"))
            or "../" in value
            or "..\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise TacticalMatchupContractError(
                "La salida contiene texto operativo no autorizado."
            )
        return
    if type(value) is list:
        for item in value:
            _validate_public_tree(item)
        return
    if type(value) is dict:
        for child_key, child_value in value.items():
            if type(child_key) is not str:
                raise TacticalMatchupContractError("Las claves JSON deben ser strings.")
            _validate_public_tree(child_value, key=child_key)
        return
    raise TacticalMatchupContractError("La salida contiene un objeto no contractual.")


def _canonical_bytes(structure: dict[str, object]) -> bytes:
    _validate_public_tree(structure)
    return json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_tactical_matchup_query_json(query: TacticalMatchupQuery) -> bytes:
    return _canonical_bytes(_query_structure(query))


def canonical_tactical_category_evidence_json(
    evidence: TacticalCategoryEvidence,
) -> bytes:
    return _canonical_bytes(_category_structure(evidence))


def canonical_tactical_matchup_category_evidence_json(
    evidence: TacticalMatchupCategoryEvidence,
) -> bytes:
    return _canonical_bytes(_pair_structure(evidence))


def canonical_tactical_matchup_evidence_json(
    evidence: TacticalMatchupEvidence,
) -> bytes:
    return _canonical_bytes(_result_structure(evidence))


def tactical_matchup_query_fingerprint(query: TacticalMatchupQuery) -> str:
    return sha256(
        MATCHUP_QUERY_FINGERPRINT_DOMAIN
        + canonical_tactical_matchup_query_json(query)
    ).hexdigest().upper()


def tactical_category_evidence_fingerprint(
    evidence: TacticalCategoryEvidence,
) -> str:
    return sha256(
        CATEGORY_EVIDENCE_FINGERPRINT_DOMAIN
        + canonical_tactical_category_evidence_json(evidence)
    ).hexdigest().upper()


def tactical_matchup_category_evidence_fingerprint(
    evidence: TacticalMatchupCategoryEvidence,
) -> str:
    return sha256(
        MATCHUP_CATEGORY_FINGERPRINT_DOMAIN
        + canonical_tactical_matchup_category_evidence_json(evidence)
    ).hexdigest().upper()


def tactical_matchup_evidence_fingerprint(
    evidence: TacticalMatchupEvidence,
) -> str:
    return sha256(
        MATCHUP_EVIDENCE_FINGERPRINT_DOMAIN
        + canonical_tactical_matchup_evidence_json(evidence)
    ).hexdigest().upper()


__all__ = [
    "CATEGORY_EVIDENCE_FINGERPRINT_DOMAIN",
    "EVIDENCE_PROVENANCE",
    "MATCHUP_CATEGORY_FINGERPRINT_DOMAIN",
    "MATCHUP_EVIDENCE_CONTRACT_VERSION",
    "MATCHUP_EVIDENCE_FINGERPRINT_DOMAIN",
    "MATCHUP_QUERY_FINGERPRINT_DOMAIN",
    "PATTERN_OWNERSHIP_CATALOG",
    "TacticalCategoryEvidence",
    "TacticalCategoryEvidenceState",
    "TacticalEvidencePerspective",
    "TacticalEvidenceScope",
    "TacticalEvidenceScopeStrategy",
    "TacticalMatchupCategoryEvidence",
    "TacticalMatchupCategoryState",
    "TacticalMatchupContractError",
    "TacticalMatchupEvidence",
    "TacticalMatchupEvidenceState",
    "TacticalMatchupQuery",
    "TacticalPatternOwnership",
    "build_tactical_matchup_evidence",
    "canonical_tactical_category_evidence_json",
    "canonical_tactical_matchup_category_evidence_json",
    "canonical_tactical_matchup_evidence_json",
    "canonical_tactical_matchup_query_json",
    "tactical_category_evidence_fingerprint",
    "tactical_matchup_category_evidence_fingerprint",
    "tactical_matchup_evidence_fingerprint",
    "tactical_matchup_query_fingerprint",
    "tactical_wilson_interval",
    "validate_tactical_category_evidence",
    "validate_tactical_matchup_category_evidence",
    "validate_tactical_matchup_evidence",
    "validate_tactical_matchup_query",
]
