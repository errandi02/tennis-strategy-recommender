"""Perfiles tacticos historicos sinteticos, temporales y leakage-safe.

El modulo agrega ``TacticalFeatureVector`` ya validados. No ejecuta parser,
extractores o adaptadores, no hace E/S y no ajusta modelos. Las claves de las
observaciones son internas: perfiles y batches publican solo agregados.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from hashlib import sha256
import json
from math import isfinite, sqrt
import re
from typing import Final

from src.recommender.tactical_feature_encoder import (
    CONTEXT_FEATURE_NAMES,
    FEATURE_CONTRACT_VERSION,
    MASK_FEATURE_NAMES,
    P02_FEATURE_NAMES,
    P04_FEATURE_NAMES,
    P05_FEATURE_NAMES,
    P06_FEATURE_NAMES,
    P09_FEATURE_NAMES,
    TacticalEncodingPolicy,
    TacticalFeatureSchema,
    TacticalFeatureVector,
    build_tactical_feature_schema,
    feature_schema_fingerprint,
    feature_vector_fingerprint,
    validate_tactical_feature_schema,
    validate_tactical_feature_vector,
)


HISTORY_CONTRACT_VERSION: Final = "1.0.0"
WILSON_Z_95: Final = 1.959963984540054
NUMERIC_TOLERANCE: Final = 1e-15
OBSERVATION_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-history-observation\x00"
QUERY_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-history-query\x00"
PROFILE_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-player-profile\x00"
PROFILE_BATCH_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-profile-batch\x00"

PATTERN_ORDER: Final = ("P02", "P04", "P05", "P06", "P09")
SERVER_PATTERNS: Final = frozenset({"P02"})
RETURNER_PATTERNS: Final = frozenset({"P04", "P05", "P06", "P09"})
OBSERVATION_PROVENANCE: Final = (
    ("source", "synthetic_tactical_feature_vector"),
    ("feature_contract_version", FEATURE_CONTRACT_VERSION),
)
PROFILE_PROVENANCE: Final = (
    ("source", "tactical_feature_vectors_in_memory"),
    ("feature_contract_version", FEATURE_CONTRACT_VERSION),
    ("leakage_policy", "strictly_before_as_of_date"),
)
PROFILE_RECONCILIATIONS: Final = (
    ("input_counts_exhaustive", True),
    ("strict_temporal_cut_reconciled", True),
    ("feature_order_reconciled", True),
    ("roles_separated", True),
    ("labels_separated_from_features", True),
    ("no_combined_redundant_total", True),
)
_CONTRACTUAL_FEATURE_NAMES: Final = (
    P02_FEATURE_NAMES
    + P04_FEATURE_NAMES
    + P05_FEATURE_NAMES
    + P06_FEATURE_NAMES
    + P09_FEATURE_NAMES
    + MASK_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
)
_SAFE_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SAFE_FEATURE = re.compile(r"^[A-Za-z0-9_.|]+$")
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
        "server_won_point",
        "returner_won_point",
        "timestamp",
        "path",
    }
)
_FORBIDDEN_TERMS: Final = ("evaluation", "scoring", "recommendation")


class TacticalHistoryContractError(ValueError):
    """Error cerrado sin claves de observacion ni datos individuales."""


class TacticalPlayerRole(str, Enum):
    SERVER = "server"
    RETURNER = "returner"


class TacticalOutcomeLabel(str, Enum):
    SERVER_WON_POINT = "server_won_point"
    RETURNER_WON_POINT = "returner_won_point"


class TacticalLabelAvailability(str, Enum):
    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"


class TacticalFeatureGroup(str, Enum):
    TACTICAL = "tactical"
    MASK = "mask"
    CONTEXT = "context"


class TacticalProfileAvailability(str, Enum):
    AVAILABLE = "available"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class TacticalHistoricalObservation:
    contract_version: str
    match_id: str
    point_number: int
    serve_number: int
    effective_date: date
    player: str
    opponent: str
    role: TacticalPlayerRole
    feature_vector: TacticalFeatureVector
    label: TacticalOutcomeLabel | None
    label_availability: TacticalLabelAvailability
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        validate_tactical_historical_observation(self)


@dataclass(frozen=True)
class TacticalProfileQuery:
    contract_version: str
    player: str
    role: TacticalPlayerRole
    as_of_date: date
    policy: TacticalEncodingPolicy
    schema_fingerprint: str
    serve_numbers: tuple[int, ...]
    opponent_filter: str | None
    window_days: int | None
    minimum_history: int

    def __post_init__(self) -> None:
        validate_tactical_profile_query(self)


@dataclass(frozen=True)
class TacticalFeatureAggregate:
    contract_version: str
    feature_name: str
    feature_group: TacticalFeatureGroup
    pattern_id: str | None
    active_count: int
    applicable_count: int
    observed_count: int
    frequency_over_applicable: float | None
    frequency_over_observed: float | None
    labeled_active_count: int
    actor_successes: int
    actor_success_rate: float | None
    wilson_lower: float | None
    wilson_upper: float | None
    outcome_unavailable_reason: str | None

    def __post_init__(self) -> None:
        validate_tactical_feature_aggregate(self)


@dataclass(frozen=True)
class TacticalPlayerProfile:
    contract_version: str
    query: TacticalProfileQuery
    development_end_date: date | None
    redundant_representation: bool
    total_observations_received: int
    temporally_eligible_count: int
    profile_eligible_count: int
    excluded_sealed_period_count: int
    excluded_future_or_cutoff_count: int
    excluded_before_window_count: int
    excluded_player_count: int
    excluded_role_count: int
    excluded_context_filter_count: int
    first_serve_attempts: int
    second_serve_attempts: int
    labeled_observation_count: int
    unlabeled_observation_count: int
    aggregates: tuple[TacticalFeatureAggregate, ...]
    availability_state: TacticalProfileAvailability
    reason_codes: tuple[str, ...]
    reconciliations: tuple[tuple[str, bool], ...]
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        validate_tactical_player_profile(self)


@dataclass(frozen=True)
class TacticalProfileBatch:
    contract_version: str
    schema_fingerprint: str
    policy: TacticalEncodingPolicy
    development_end_date: date | None
    profiles: tuple[TacticalPlayerProfile, ...]
    profile_count: int
    reconciliations: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        validate_tactical_profile_batch(self)


def _strict_date(value: object, field: str) -> date:
    if type(value) is not date:
        raise TypeError(f"{field} debe ser datetime.date exacta sin hora.")
    return value


def _strict_text(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _SAFE_TEXT.fullmatch(value) is None
    ):
        raise TacticalHistoryContractError(
            f"{field} debe ser texto no vacio, estable y sin espacios externos."
        )
    return value


def _strict_count(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positivo" if positive else "no negativo"
        raise TacticalHistoryContractError(f"{field} debe ser int real {qualifier}.")
    return value


def _validate_provenance(
    value: object, expected: tuple[tuple[str, str], ...], owner: str
) -> None:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in value
    ):
        raise TacticalHistoryContractError(
            f"La procedencia de {owner} debe ser profundamente inmutable."
        )
    if value != expected:
        raise TacticalHistoryContractError(
            f"La procedencia de {owner} no coincide con el contrato cerrado."
        )


def _validate_reconciliations(
    value: object, expected: tuple[tuple[str, bool], ...], owner: str
) -> None:
    if type(value) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not bool
        for item in value
    ):
        raise TacticalHistoryContractError(
            f"Las reconciliaciones de {owner} deben ser profundamente inmutables."
        )
    if value != expected:
        raise TacticalHistoryContractError(
            f"Las reconciliaciones de {owner} no coinciden con el contrato."
        )


def _validate_optional_float(value: object, field: str) -> None:
    if value is not None and (type(value) is not float or not isfinite(value)):
        raise TacticalHistoryContractError(
            f"{field} debe ser float finito o None."
        )


def validate_tactical_historical_observation(
    observation: TacticalHistoricalObservation,
) -> None:
    if type(observation) is not TacticalHistoricalObservation:
        raise TypeError("observation debe ser TacticalHistoricalObservation exacta.")
    if (
        type(observation.contract_version) is not str
        or observation.contract_version != HISTORY_CONTRACT_VERSION
    ):
        raise TacticalHistoryContractError("Version contractual de observacion invalida.")
    match_id = _strict_text(observation.match_id, "match_id interno")
    if _OPAQUE_ID.fullmatch(match_id) is None:
        raise TacticalHistoryContractError("match_id interno debe ser opaco y seguro.")
    _strict_count(observation.point_number, "point_number interno")
    if type(observation.serve_number) is not int or observation.serve_number not in (1, 2):
        raise TacticalHistoryContractError("serve_number debe ser int real 1/2.")
    _strict_date(observation.effective_date, "effective_date")
    player = _strict_text(observation.player, "player")
    opponent = _strict_text(observation.opponent, "opponent")
    if player == opponent:
        raise TacticalHistoryContractError("player y opponent deben ser distintos.")
    if type(observation.role) is not TacticalPlayerRole:
        raise TypeError("role debe ser TacticalPlayerRole exacto.")
    if type(observation.feature_vector) is not TacticalFeatureVector:
        raise TypeError("feature_vector debe ser TacticalFeatureVector exacto.")
    validate_tactical_feature_vector(observation.feature_vector)
    if observation.feature_vector.serve_number != observation.serve_number:
        raise TacticalHistoryContractError(
            "serve_number no coincide con el vector tactico."
        )
    if type(observation.label_availability) is not TacticalLabelAvailability:
        raise TypeError("label_availability fuera del dominio cerrado.")
    if observation.label_availability is TacticalLabelAvailability.AVAILABLE:
        if type(observation.label) is not TacticalOutcomeLabel:
            raise TacticalHistoryContractError(
                "Un label disponible exige TacticalOutcomeLabel exacto."
            )
    elif observation.label is not None:
        raise TacticalHistoryContractError(
            "Un label no disponible debe conservar valor None."
        )
    _validate_provenance(observation.provenance, OBSERVATION_PROVENANCE, "observacion")


def validate_tactical_profile_query(query: TacticalProfileQuery) -> None:
    if type(query) is not TacticalProfileQuery:
        raise TypeError("query debe ser TacticalProfileQuery exacta.")
    if (
        type(query.contract_version) is not str
        or query.contract_version != HISTORY_CONTRACT_VERSION
    ):
        raise TacticalHistoryContractError("Version contractual de query invalida.")
    _strict_text(query.player, "player objetivo")
    if type(query.role) is not TacticalPlayerRole:
        raise TypeError("role de query debe ser TacticalPlayerRole exacto.")
    as_of = _strict_date(query.as_of_date, "as_of_date")
    if type(query.policy) is not TacticalEncodingPolicy:
        raise TypeError("policy debe ser TacticalEncodingPolicy exacta.")
    schema = build_tactical_feature_schema(query.policy)
    expected_fingerprint = feature_schema_fingerprint(schema)
    if (
        type(query.schema_fingerprint) is not str
        or _SHA256.fullmatch(query.schema_fingerprint) is None
        or query.schema_fingerprint != expected_fingerprint
    ):
        raise TacticalHistoryContractError("schema_fingerprint de query invalido.")
    if (
        type(query.serve_numbers) is not tuple
        or any(type(value) is not int for value in query.serve_numbers)
        or query.serve_numbers not in {(1,), (2,), (1, 2)}
    ):
        raise TacticalHistoryContractError(
            "serve_numbers debe ser (1,), (2,) o (1, 2)."
        )
    if query.opponent_filter is not None:
        opponent = _strict_text(query.opponent_filter, "opponent_filter")
        if opponent == query.player:
            raise TacticalHistoryContractError(
                "opponent_filter debe ser distinto del jugador objetivo."
            )
    if query.window_days is not None:
        window = _strict_count(query.window_days, "window_days", positive=True)
        if window > (as_of - date.min).days:
            raise TacticalHistoryContractError("window_days excede el calendario valido.")
    _strict_count(query.minimum_history, "minimum_history", positive=True)


def validate_development_queries(
    queries: tuple[TacticalProfileQuery, ...], development_end_date: date
) -> None:
    """Garantiza que las queries no atraviesan el fin de desarrollo configurado."""
    if type(queries) is not tuple or not queries:
        raise TacticalHistoryContractError("queries debe ser tuple no vacia.")
    boundary = _strict_date(development_end_date, "development_end_date")
    for query in queries:
        validate_tactical_profile_query(query)
        if query.as_of_date > boundary:
            raise TacticalHistoryContractError(
                "Una query excede el periodo de desarrollo autorizado."
            )


def wilson_interval(
    successes: int, trials: int, *, feature_name: str
) -> tuple[float | None, float | None, float | None]:
    """Devuelve tasa e intervalo Wilson 95 % con bordes teoricos exactos."""
    if type(successes) is not int or successes < 0:
        raise TacticalHistoryContractError("successes debe ser int real no negativo.")
    if type(trials) is not int or trials < 0:
        raise TacticalHistoryContractError("trials debe ser int real no negativo.")
    if successes > trials:
        raise TacticalHistoryContractError("successes no puede superar trials.")
    if type(feature_name) is not str or _SAFE_FEATURE.fullmatch(feature_name) is None:
        raise TacticalHistoryContractError("feature_name no es contractual y seguro.")
    if trials == 0:
        return None, None, None
    rate = successes / trials
    z2 = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z2 / trials
    centre = (rate + z2 / (2.0 * trials)) / denominator
    margin = (
        WILSON_Z_95
        * sqrt(rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    lower = centre - margin
    upper = centre + margin
    if successes == 0:
        lower = 0.0
    elif -NUMERIC_TOLERANCE <= lower < 0.0:
        lower = 0.0
    if successes == trials:
        upper = 1.0
    elif 1.0 < upper <= 1.0 + NUMERIC_TOLERANCE:
        upper = 1.0
    if not all(isfinite(value) for value in (rate, lower, upper)):
        raise TacticalHistoryContractError(
            "Wilson produjo valores no finitos para conteos agregados."
        )
    if not (
        0.0 <= lower <= upper <= 1.0
        and lower - NUMERIC_TOLERANCE <= rate <= upper + NUMERIC_TOLERANCE
    ):
        raise TacticalHistoryContractError(
            "Wilson fuera de rango para la feature agregada."
        )
    return rate, lower, upper


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _pattern_for_feature(feature_name: str) -> str | None:
    lowered = feature_name.lower()
    for pattern_id in PATTERN_ORDER:
        if lowered.startswith((f"{pattern_id.lower()}.", f"mask.{pattern_id.lower()}.")):
            return pattern_id
    return None


def _group_for_feature(feature_name: str) -> TacticalFeatureGroup:
    if feature_name.startswith("mask."):
        return TacticalFeatureGroup.MASK
    if feature_name.startswith("context."):
        return TacticalFeatureGroup.CONTEXT
    return TacticalFeatureGroup.TACTICAL


def _role_allows_pattern(role: TacticalPlayerRole, pattern_id: str | None) -> bool:
    if pattern_id is None:
        return True
    if role is TacticalPlayerRole.SERVER:
        return pattern_id in SERVER_PATTERNS
    return pattern_id in RETURNER_PATTERNS


def _actor_success(observation: TacticalHistoricalObservation) -> bool:
    if observation.label_availability is TacticalLabelAvailability.NOT_AVAILABLE:
        raise TacticalHistoryContractError("No puede evaluarse un label no disponible.")
    expected = (
        TacticalOutcomeLabel.SERVER_WON_POINT
        if observation.role is TacticalPlayerRole.SERVER
        else TacticalOutcomeLabel.RETURNER_WON_POINT
    )
    return observation.label is expected


def _make_aggregate(
    feature_name: str,
    schema: TacticalFeatureSchema,
    observations: tuple[TacticalHistoricalObservation, ...],
) -> TacticalFeatureAggregate:
    feature_group = _group_for_feature(feature_name)
    pattern_id = _pattern_for_feature(feature_name)
    index = dict(schema.name_to_index)
    role = observations[0].role if observations else None
    allowed = role is None or _role_allows_pattern(role, pattern_id)

    effective_values = tuple(
        observation.feature_vector.values[index[feature_name]] if allowed else 0
        for observation in observations
    )
    active_count = sum(effective_values)
    if feature_group is TacticalFeatureGroup.TACTICAL:
        if pattern_id is None:
            raise TacticalHistoryContractError("Feature tactica sin patron.")
        applicable_name = f"mask.{pattern_id.lower()}.applicable"
        observed_name = f"mask.{pattern_id.lower()}.observed"
        applicable_count = (
            sum(
                observation.feature_vector.values[index[applicable_name]]
                for observation in observations
            )
            if allowed
            else 0
        )
        observed_count = (
            sum(
                observation.feature_vector.values[index[observed_name]]
                for observation in observations
            )
            if allowed
            else 0
        )
    else:
        applicable_count = len(observations)
        observed_count = len(observations)

    active_observations = tuple(
        observation
        for observation, active in zip(observations, effective_values)
        if active == 1
    )
    labeled = tuple(
        observation
        for observation in active_observations
        if observation.label_availability is TacticalLabelAvailability.AVAILABLE
    )
    successes = sum(_actor_success(observation) for observation in labeled)
    rate, lower, upper = wilson_interval(
        successes, len(labeled), feature_name=feature_name
    )
    return TacticalFeatureAggregate(
        contract_version=HISTORY_CONTRACT_VERSION,
        feature_name=feature_name,
        feature_group=feature_group,
        pattern_id=pattern_id,
        active_count=active_count,
        applicable_count=applicable_count,
        observed_count=observed_count,
        frequency_over_applicable=_optional_rate(active_count, applicable_count),
        frequency_over_observed=_optional_rate(active_count, observed_count),
        labeled_active_count=len(labeled),
        actor_successes=successes,
        actor_success_rate=rate,
        wilson_lower=lower,
        wilson_upper=upper,
        outcome_unavailable_reason=(
            "no_labeled_activations" if not labeled else None
        ),
    )


def validate_tactical_feature_aggregate(
    aggregate: TacticalFeatureAggregate,
) -> None:
    if type(aggregate) is not TacticalFeatureAggregate:
        raise TypeError("aggregate debe ser TacticalFeatureAggregate exacto.")
    if (
        type(aggregate.contract_version) is not str
        or aggregate.contract_version != HISTORY_CONTRACT_VERSION
    ):
        raise TacticalHistoryContractError("Version contractual del agregado invalida.")
    if type(aggregate.feature_name) is not str or _SAFE_FEATURE.fullmatch(aggregate.feature_name) is None:
        raise TacticalHistoryContractError("feature_name agregado invalido.")
    if aggregate.feature_name not in _CONTRACTUAL_FEATURE_NAMES:
        raise TacticalHistoryContractError(
            "feature_name no pertenece al catalogo contractual."
        )
    expected_group = _group_for_feature(aggregate.feature_name)
    if type(aggregate.feature_group) is not TacticalFeatureGroup or aggregate.feature_group is not expected_group:
        raise TacticalHistoryContractError("feature_group no reconcilia con el nombre.")
    expected_pattern = _pattern_for_feature(aggregate.feature_name)
    if (
        aggregate.pattern_id is not None
        and type(aggregate.pattern_id) is not str
    ) or aggregate.pattern_id != expected_pattern:
        raise TacticalHistoryContractError("pattern_id no reconcilia con la feature.")
    for field in (
        "active_count",
        "applicable_count",
        "observed_count",
        "labeled_active_count",
        "actor_successes",
    ):
        _strict_count(getattr(aggregate, field), field)
    if not (
        0
        <= aggregate.actor_successes
        <= aggregate.labeled_active_count
        <= aggregate.active_count
        <= aggregate.observed_count
        <= aggregate.applicable_count
    ):
        raise TacticalHistoryContractError("Conteos del agregado fuera de orden.")
    expected_applicable_rate = _optional_rate(
        aggregate.active_count, aggregate.applicable_count
    )
    expected_observed_rate = _optional_rate(
        aggregate.active_count, aggregate.observed_count
    )
    for field in (
        "frequency_over_applicable",
        "frequency_over_observed",
        "actor_success_rate",
        "wilson_lower",
        "wilson_upper",
    ):
        _validate_optional_float(getattr(aggregate, field), field)
    for name, actual, expected in (
        ("frequency_over_applicable", aggregate.frequency_over_applicable, expected_applicable_rate),
        ("frequency_over_observed", aggregate.frequency_over_observed, expected_observed_rate),
    ):
        if actual != expected or (actual is not None and not isfinite(actual)):
            raise TacticalHistoryContractError(f"{name} no reconcilia.")
    rate, lower, upper = wilson_interval(
        aggregate.actor_successes,
        aggregate.labeled_active_count,
        feature_name=aggregate.feature_name,
    )
    if (
        aggregate.actor_success_rate != rate
        or aggregate.wilson_lower != lower
        or aggregate.wilson_upper != upper
    ):
        raise TacticalHistoryContractError("Outcome o Wilson no reconcilia.")
    expected_reason = (
        "no_labeled_activations" if aggregate.labeled_active_count == 0 else None
    )
    if (
        aggregate.outcome_unavailable_reason is not None
        and type(aggregate.outcome_unavailable_reason) is not str
    ) or aggregate.outcome_unavailable_reason != expected_reason:
        raise TacticalHistoryContractError(
            "Razon de outcome no disponible incompatible con trials."
        )


def _validate_observation_collection(
    observations: object,
) -> tuple[TacticalHistoricalObservation, ...]:
    if type(observations) is not tuple:
        raise TypeError("observations debe ser tuple inmutable.")
    keys: set[tuple[str, int, int]] = set()
    for observation in observations:
        validate_tactical_historical_observation(observation)
        key = (observation.match_id, observation.point_number, observation.serve_number)
        if key in keys:
            raise TacticalHistoryContractError(
                "Existe una clave de observacion duplicada."
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


def _query_sort_key(query: TacticalProfileQuery) -> tuple[object, ...]:
    return (
        query.player,
        query.role.value,
        query.as_of_date,
        query.serve_numbers,
        "" if query.opponent_filter is None else query.opponent_filter,
        -1 if query.window_days is None else query.window_days,
        query.minimum_history,
    )


def _profile_state(
    eligible_count: int, minimum_history: int
) -> tuple[TacticalProfileAvailability, tuple[str, ...]]:
    if eligible_count == 0:
        return TacticalProfileAvailability.NOT_AVAILABLE, ("no_eligible_history",)
    if eligible_count < minimum_history:
        return TacticalProfileAvailability.INSUFFICIENT_HISTORY, (
            "minimum_history_not_met",
        )
    return TacticalProfileAvailability.AVAILABLE, ("minimum_history_met",)


def _build_profile_from_validated(
    observations: tuple[TacticalHistoricalObservation, ...],
    query: TacticalProfileQuery,
    schema: TacticalFeatureSchema,
    development_end_date: date | None,
) -> TacticalPlayerProfile:
    sealed: list[TacticalHistoricalObservation] = []
    temporal_candidates: list[TacticalHistoricalObservation] = []
    for observation in observations:
        if development_end_date is not None and observation.effective_date > development_end_date:
            sealed.append(observation)
        else:
            temporal_candidates.append(observation)
    future = [
        observation
        for observation in temporal_candidates
        if observation.effective_date >= query.as_of_date
    ]
    before_cut = [
        observation
        for observation in temporal_candidates
        if observation.effective_date < query.as_of_date
    ]
    before_window: list[TacticalHistoricalObservation] = []
    if query.window_days is not None:
        lower = query.as_of_date - timedelta(days=query.window_days)
        before_window = [item for item in before_cut if item.effective_date < lower]
        before_cut = [item for item in before_cut if item.effective_date >= lower]
    temporally_eligible = tuple(before_cut)
    wrong_player = [item for item in before_cut if item.player != query.player]
    after_player = [item for item in before_cut if item.player == query.player]
    wrong_role = [item for item in after_player if item.role is not query.role]
    after_role = [item for item in after_player if item.role is query.role]
    wrong_filter = [
        item
        for item in after_role
        if item.serve_number not in query.serve_numbers
        or (
            query.opponent_filter is not None
            and item.opponent != query.opponent_filter
        )
    ]
    eligible = tuple(
        item
        for item in after_role
        if item.serve_number in query.serve_numbers
        and (
            query.opponent_filter is None
            or item.opponent == query.opponent_filter
        )
    )
    aggregates = tuple(
        _make_aggregate(feature_name, schema, eligible)
        for feature_name in schema.feature_names
    )
    labeled_count = sum(
        item.label_availability is TacticalLabelAvailability.AVAILABLE
        for item in eligible
    )
    state, reasons = _profile_state(len(eligible), query.minimum_history)
    return TacticalPlayerProfile(
        contract_version=HISTORY_CONTRACT_VERSION,
        query=query,
        development_end_date=development_end_date,
        redundant_representation=schema.redundant_representation,
        total_observations_received=len(observations),
        temporally_eligible_count=len(temporally_eligible),
        profile_eligible_count=len(eligible),
        excluded_sealed_period_count=len(sealed),
        excluded_future_or_cutoff_count=len(future),
        excluded_before_window_count=len(before_window),
        excluded_player_count=len(wrong_player),
        excluded_role_count=len(wrong_role),
        excluded_context_filter_count=len(wrong_filter),
        first_serve_attempts=sum(item.serve_number == 1 for item in eligible),
        second_serve_attempts=sum(item.serve_number == 2 for item in eligible),
        labeled_observation_count=labeled_count,
        unlabeled_observation_count=len(eligible) - labeled_count,
        aggregates=aggregates,
        availability_state=state,
        reason_codes=reasons,
        reconciliations=PROFILE_RECONCILIATIONS,
        provenance=PROFILE_PROVENANCE,
    )


def build_tactical_player_profile(
    observations: tuple[TacticalHistoricalObservation, ...],
    query: TacticalProfileQuery,
    schema: TacticalFeatureSchema,
    *,
    development_end_date: date | None = None,
) -> TacticalPlayerProfile:
    """Construye un perfil usando solo observaciones estrictamente anteriores."""
    ordered = _validate_observation_collection(observations)
    validate_tactical_profile_query(query)
    if type(schema) is not TacticalFeatureSchema:
        raise TypeError("schema debe ser TacticalFeatureSchema exacto.")
    validate_tactical_feature_schema(schema)
    if query.policy is not schema.policy or query.schema_fingerprint != feature_schema_fingerprint(schema):
        raise TacticalHistoryContractError("Query y schema no coinciden.")
    if development_end_date is not None:
        validate_development_queries((query,), development_end_date)
    for observation in ordered:
        if (
            observation.feature_vector.policy is not schema.policy
            or observation.feature_vector.schema_fingerprint
            != feature_schema_fingerprint(schema)
        ):
            raise TacticalHistoryContractError(
                "Todas las observaciones deben compartir schema y policy."
            )
    return _build_profile_from_validated(
        ordered, query, schema, development_end_date
    )


def _aggregate_by_name(
    profile: TacticalPlayerProfile,
) -> dict[str, TacticalFeatureAggregate]:
    return {aggregate.feature_name: aggregate for aggregate in profile.aggregates}


def validate_tactical_player_profile(profile: TacticalPlayerProfile) -> None:
    if type(profile) is not TacticalPlayerProfile:
        raise TypeError("profile debe ser TacticalPlayerProfile exacto.")
    if (
        type(profile.contract_version) is not str
        or profile.contract_version != HISTORY_CONTRACT_VERSION
    ):
        raise TacticalHistoryContractError("Version contractual del perfil invalida.")
    validate_tactical_profile_query(profile.query)
    schema = build_tactical_feature_schema(profile.query.policy)
    if profile.development_end_date is not None:
        _strict_date(profile.development_end_date, "development_end_date")
        validate_development_queries((profile.query,), profile.development_end_date)
    if type(profile.redundant_representation) is not bool or (
        profile.redundant_representation != schema.redundant_representation
    ):
        raise TacticalHistoryContractError("Flag de redundancia no reconciliado.")
    count_fields = (
        "total_observations_received",
        "temporally_eligible_count",
        "profile_eligible_count",
        "excluded_sealed_period_count",
        "excluded_future_or_cutoff_count",
        "excluded_before_window_count",
        "excluded_player_count",
        "excluded_role_count",
        "excluded_context_filter_count",
        "first_serve_attempts",
        "second_serve_attempts",
        "labeled_observation_count",
        "unlabeled_observation_count",
    )
    for field in count_fields:
        _strict_count(getattr(profile, field), field)
    if profile.temporally_eligible_count != (
        profile.total_observations_received
        - profile.excluded_sealed_period_count
        - profile.excluded_future_or_cutoff_count
        - profile.excluded_before_window_count
    ):
        raise TacticalHistoryContractError("Conteos temporales no son exhaustivos.")
    if profile.profile_eligible_count != (
        profile.temporally_eligible_count
        - profile.excluded_player_count
        - profile.excluded_role_count
        - profile.excluded_context_filter_count
    ):
        raise TacticalHistoryContractError("Conteos de filtros no son exhaustivos.")
    if profile.first_serve_attempts + profile.second_serve_attempts != profile.profile_eligible_count:
        raise TacticalHistoryContractError("Intentos por numero de saque no reconcilian.")
    if profile.labeled_observation_count + profile.unlabeled_observation_count != profile.profile_eligible_count:
        raise TacticalHistoryContractError("Labels disponibles y ausentes no reconcilian.")
    if type(profile.aggregates) is not tuple or len(profile.aggregates) != schema.feature_count:
        raise TacticalHistoryContractError("Agregados ausentes o con cardinalidad invalida.")
    for aggregate in profile.aggregates:
        validate_tactical_feature_aggregate(aggregate)
        if aggregate.applicable_count > profile.profile_eligible_count:
            raise TacticalHistoryContractError("Denominador supera intentos elegibles.")
    if tuple(item.feature_name for item in profile.aggregates) != schema.feature_names:
        raise TacticalHistoryContractError("Agregados fuera del orden del schema.")
    by_name = _aggregate_by_name(profile)
    blocks = (
        ("P02", P02_FEATURE_NAMES),
        ("P04", P04_FEATURE_NAMES),
        ("P05", P05_FEATURE_NAMES),
        ("P06", P06_FEATURE_NAMES),
        ("P09", P09_FEATURE_NAMES),
    )
    allowed_patterns = (
        SERVER_PATTERNS
        if profile.query.role is TacticalPlayerRole.SERVER
        else RETURNER_PATTERNS
    )
    for pattern_id, feature_names in blocks:
        mask_applicable = by_name[f"mask.{pattern_id.lower()}.applicable"].active_count
        mask_observed = by_name[f"mask.{pattern_id.lower()}.observed"].active_count
        if pattern_id not in allowed_patterns and (mask_applicable or mask_observed):
            raise TacticalHistoryContractError("Un rol contiene mascaras de otro actor.")
        present_features = tuple(name for name in feature_names if name in by_name)
        for name in present_features:
            aggregate = by_name[name]
            if aggregate.applicable_count != mask_applicable or aggregate.observed_count != mask_observed:
                raise TacticalHistoryContractError(
                    "Denominadores tacticos no reconcilian con mascaras."
                )
        if present_features and sum(by_name[name].active_count for name in present_features) != mask_observed:
            raise TacticalHistoryContractError(
                "Categorias tacticas no agotan observaciones del patron."
            )
    for name in (*MASK_FEATURE_NAMES, *schema.context_feature_names):
        aggregate = by_name[name]
        if (
            aggregate.applicable_count != profile.profile_eligible_count
            or aggregate.observed_count != profile.profile_eligible_count
        ):
            raise TacticalHistoryContractError(
                "Mascara o contexto usa denominador incorrecto."
            )
    expected_state, expected_reasons = _profile_state(
        profile.profile_eligible_count, profile.query.minimum_history
    )
    if type(profile.availability_state) is not TacticalProfileAvailability or profile.availability_state is not expected_state:
        raise TacticalHistoryContractError("Estado del perfil no reconciliado.")
    if (
        type(profile.reason_codes) is not tuple
        or any(type(reason) is not str for reason in profile.reason_codes)
        or profile.reason_codes != expected_reasons
    ):
        raise TacticalHistoryContractError("Reason codes del perfil no reconciliados.")
    _validate_reconciliations(
        profile.reconciliations, PROFILE_RECONCILIATIONS, "perfil"
    )
    _validate_provenance(profile.provenance, PROFILE_PROVENANCE, "perfil")


def build_tactical_profile_batch(
    observations: tuple[TacticalHistoricalObservation, ...],
    queries: tuple[TacticalProfileQuery, ...],
    schema: TacticalFeatureSchema,
    *,
    development_end_date: date | None = None,
) -> TacticalProfileBatch:
    """Construye perfiles canonicos desde una unica coleccion ya validada."""
    ordered_observations = _validate_observation_collection(observations)
    if type(queries) is not tuple or not queries:
        raise TacticalHistoryContractError("queries debe ser tuple no vacia.")
    if type(schema) is not TacticalFeatureSchema:
        raise TypeError("schema debe ser TacticalFeatureSchema exacto.")
    validate_tactical_feature_schema(schema)
    seen: set[TacticalProfileQuery] = set()
    for query in queries:
        validate_tactical_profile_query(query)
        if query in seen:
            raise TacticalHistoryContractError("Existe una query duplicada.")
        seen.add(query)
        if query.policy is not schema.policy or query.schema_fingerprint != feature_schema_fingerprint(schema):
            raise TacticalHistoryContractError("Todas las queries deben compartir schema y policy.")
    if development_end_date is not None:
        validate_development_queries(queries, development_end_date)
    for observation in ordered_observations:
        if (
            observation.feature_vector.policy is not schema.policy
            or observation.feature_vector.schema_fingerprint
            != feature_schema_fingerprint(schema)
        ):
            raise TacticalHistoryContractError(
                "Todas las observaciones deben compartir schema y policy."
            )
    profiles = tuple(
        _build_profile_from_validated(
            ordered_observations, query, schema, development_end_date
        )
        for query in sorted(queries, key=_query_sort_key)
    )
    return TacticalProfileBatch(
        contract_version=HISTORY_CONTRACT_VERSION,
        schema_fingerprint=feature_schema_fingerprint(schema),
        policy=schema.policy,
        development_end_date=development_end_date,
        profiles=profiles,
        profile_count=len(profiles),
        reconciliations=(
            ("profile_count_reconciled", True),
            ("canonical_query_order_reconciled", True),
            ("shared_schema_reconciled", True),
            ("development_period_reconciled", True),
        ),
    )


def validate_tactical_profile_batch(batch: TacticalProfileBatch) -> None:
    if type(batch) is not TacticalProfileBatch:
        raise TypeError("batch debe ser TacticalProfileBatch exacto.")
    if (
        type(batch.contract_version) is not str
        or batch.contract_version != HISTORY_CONTRACT_VERSION
    ):
        raise TacticalHistoryContractError("Version contractual del batch invalida.")
    if type(batch.policy) is not TacticalEncodingPolicy:
        raise TypeError("policy del batch invalida.")
    schema = build_tactical_feature_schema(batch.policy)
    if (
        type(batch.schema_fingerprint) is not str
        or _SHA256.fullmatch(batch.schema_fingerprint) is None
        or batch.schema_fingerprint != feature_schema_fingerprint(schema)
    ):
        raise TacticalHistoryContractError("Fingerprint del batch invalido.")
    if batch.development_end_date is not None:
        _strict_date(batch.development_end_date, "development_end_date")
    if type(batch.profiles) is not tuple or not batch.profiles:
        raise TacticalHistoryContractError("profiles debe ser tuple no vacia.")
    for profile in batch.profiles:
        validate_tactical_player_profile(profile)
        if (
            profile.query.policy is not batch.policy
            or profile.query.schema_fingerprint != batch.schema_fingerprint
            or profile.development_end_date != batch.development_end_date
        ):
            raise TacticalHistoryContractError("Perfil incompatible dentro del batch.")
    queries = tuple(profile.query for profile in batch.profiles)
    if len(set(queries)) != len(queries):
        raise TacticalHistoryContractError("El batch contiene queries duplicadas.")
    if queries != tuple(sorted(queries, key=_query_sort_key)):
        raise TacticalHistoryContractError("Perfiles fuera de orden canonico.")
    if type(batch.profile_count) is not int or batch.profile_count != len(batch.profiles):
        raise TacticalHistoryContractError("profile_count no reconcilia.")
    expected_reconciliations = (
        ("profile_count_reconciled", True),
        ("canonical_query_order_reconciled", True),
        ("shared_schema_reconciled", True),
        ("development_period_reconciled", True),
    )
    _validate_reconciliations(
        batch.reconciliations, expected_reconciliations, "batch"
    )


def _date_text(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _query_structure(query: TacticalProfileQuery) -> dict[str, object]:
    validate_tactical_profile_query(query)
    return {
        "as_of_date": query.as_of_date.isoformat(),
        "contract_version": query.contract_version,
        "minimum_history": query.minimum_history,
        "opponent_filter": query.opponent_filter,
        "player": query.player,
        "policy": query.policy.value,
        "role": query.role.value,
        "schema_fingerprint": query.schema_fingerprint,
        "serve_numbers": list(query.serve_numbers),
        "window_days": query.window_days,
    }


def _aggregate_structure(aggregate: TacticalFeatureAggregate) -> dict[str, object]:
    validate_tactical_feature_aggregate(aggregate)
    return {
        "active_count": aggregate.active_count,
        "actor_success_rate": aggregate.actor_success_rate,
        "actor_successes": aggregate.actor_successes,
        "applicable_count": aggregate.applicable_count,
        "contract_version": aggregate.contract_version,
        "feature_group": aggregate.feature_group.value,
        "feature_name": aggregate.feature_name,
        "frequency_over_applicable": aggregate.frequency_over_applicable,
        "frequency_over_observed": aggregate.frequency_over_observed,
        "labeled_active_count": aggregate.labeled_active_count,
        "observed_count": aggregate.observed_count,
        "outcome_unavailable_reason": aggregate.outcome_unavailable_reason,
        "pattern_id": aggregate.pattern_id,
        "wilson_lower": aggregate.wilson_lower,
        "wilson_upper": aggregate.wilson_upper,
    }


def _profile_structure(profile: TacticalPlayerProfile) -> dict[str, object]:
    validate_tactical_player_profile(profile)
    return {
        "aggregates": [_aggregate_structure(item) for item in profile.aggregates],
        "availability_state": profile.availability_state.value,
        "contract_version": profile.contract_version,
        "development_end_date": _date_text(profile.development_end_date),
        "excluded_before_window_count": profile.excluded_before_window_count,
        "excluded_context_filter_count": profile.excluded_context_filter_count,
        "excluded_future_or_cutoff_count": profile.excluded_future_or_cutoff_count,
        "excluded_player_count": profile.excluded_player_count,
        "excluded_role_count": profile.excluded_role_count,
        "excluded_sealed_period_count": profile.excluded_sealed_period_count,
        "first_serve_attempts": profile.first_serve_attempts,
        "labeled_observation_count": profile.labeled_observation_count,
        "profile_eligible_count": profile.profile_eligible_count,
        "provenance": {key: value for key, value in profile.provenance},
        "query": _query_structure(profile.query),
        "reason_codes": list(profile.reason_codes),
        "reconciliations": {key: value for key, value in profile.reconciliations},
        "redundant_representation": profile.redundant_representation,
        "second_serve_attempts": profile.second_serve_attempts,
        "temporally_eligible_count": profile.temporally_eligible_count,
        "total_observations_received": profile.total_observations_received,
        "unlabeled_observation_count": profile.unlabeled_observation_count,
    }


def _batch_structure(batch: TacticalProfileBatch) -> dict[str, object]:
    validate_tactical_profile_batch(batch)
    return {
        "contract_version": batch.contract_version,
        "development_end_date": _date_text(batch.development_end_date),
        "policy": batch.policy.value,
        "profile_count": batch.profile_count,
        "profiles": [_profile_structure(profile) for profile in batch.profiles],
        "reconciliations": {key: value for key, value in batch.reconciliations},
        "schema_fingerprint": batch.schema_fingerprint,
    }


def _observation_internal_structure(
    observation: TacticalHistoricalObservation,
) -> dict[str, object]:
    validate_tactical_historical_observation(observation)
    return {
        "contract_version": observation.contract_version,
        "effective_date": observation.effective_date.isoformat(),
        "feature_vector_fingerprint": feature_vector_fingerprint(
            observation.feature_vector
        ),
        "label": None if observation.label is None else observation.label.value,
        "label_availability": observation.label_availability.value,
        "match_id": observation.match_id,
        "opponent": observation.opponent,
        "player": observation.player,
        "point_number": observation.point_number,
        "provenance": {key: value for key, value in observation.provenance},
        "role": observation.role.value,
        "serve_number": observation.serve_number,
        "visibility": "internal_only",
    }


def _validate_json_tree(
    value: object, *, key: str | None = None, internal_observation: bool = False
) -> None:
    if key is not None:
        lowered = key.lower()
        if (
            not internal_observation
            and (
                lowered in _FORBIDDEN_PUBLIC_KEYS
                or lowered.startswith("test_")
                or any(term in lowered for term in _FORBIDDEN_TERMS)
            )
        ):
            raise TacticalHistoryContractError(
                "La salida publica contiene una clave no autorizada."
            )
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise TacticalHistoryContractError("La salida contiene un valor no finito.")
        return
    if type(value) is str:
        lowered_value = value.lower()
        if (
            "file://" in lowered_value
            or _WINDOWS_PATH.match(value)
            or value.startswith(("/", "\\\\"))
            or "../" in value
            or "..\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise TacticalHistoryContractError(
                "La salida contiene texto operativo no autorizado."
            )
        return
    if type(value) is list:
        for item in value:
            _validate_json_tree(item, internal_observation=internal_observation)
        return
    if type(value) is dict:
        for child_key, child_value in value.items():
            if type(child_key) is not str:
                raise TacticalHistoryContractError("Las claves JSON deben ser strings.")
            _validate_json_tree(
                child_value,
                key=child_key,
                internal_observation=internal_observation,
            )
        return
    raise TacticalHistoryContractError("La salida contiene un objeto no contractual.")


def _canonical_bytes(
    structure: dict[str, object], *, internal_observation: bool = False
) -> bytes:
    _validate_json_tree(structure, internal_observation=internal_observation)
    return json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_historical_observation_json(
    observation: TacticalHistoricalObservation,
) -> bytes:
    """Serializacion explicitamente interna; nunca se incluye en perfiles."""
    return _canonical_bytes(
        _observation_internal_structure(observation), internal_observation=True
    )


def canonical_profile_query_json(query: TacticalProfileQuery) -> bytes:
    return _canonical_bytes(_query_structure(query))


def canonical_player_profile_json(profile: TacticalPlayerProfile) -> bytes:
    return _canonical_bytes(_profile_structure(profile))


def canonical_profile_batch_json(batch: TacticalProfileBatch) -> bytes:
    return _canonical_bytes(_batch_structure(batch))


def historical_observation_fingerprint(
    observation: TacticalHistoricalObservation,
) -> str:
    return sha256(
        OBSERVATION_FINGERPRINT_DOMAIN
        + canonical_historical_observation_json(observation)
    ).hexdigest().upper()


def profile_query_fingerprint(query: TacticalProfileQuery) -> str:
    return sha256(
        QUERY_FINGERPRINT_DOMAIN + canonical_profile_query_json(query)
    ).hexdigest().upper()


def tactical_player_profile_fingerprint(profile: TacticalPlayerProfile) -> str:
    return sha256(
        PROFILE_FINGERPRINT_DOMAIN + canonical_player_profile_json(profile)
    ).hexdigest().upper()


def tactical_profile_batch_fingerprint(batch: TacticalProfileBatch) -> str:
    return sha256(
        PROFILE_BATCH_FINGERPRINT_DOMAIN + canonical_profile_batch_json(batch)
    ).hexdigest().upper()


__all__ = [
    "HISTORY_CONTRACT_VERSION",
    "NUMERIC_TOLERANCE",
    "OBSERVATION_FINGERPRINT_DOMAIN",
    "OBSERVATION_PROVENANCE",
    "PROFILE_BATCH_FINGERPRINT_DOMAIN",
    "PROFILE_FINGERPRINT_DOMAIN",
    "PROFILE_PROVENANCE",
    "QUERY_FINGERPRINT_DOMAIN",
    "TacticalFeatureAggregate",
    "TacticalFeatureGroup",
    "TacticalHistoricalObservation",
    "TacticalHistoryContractError",
    "TacticalLabelAvailability",
    "TacticalOutcomeLabel",
    "TacticalPlayerProfile",
    "TacticalPlayerRole",
    "TacticalProfileAvailability",
    "TacticalProfileBatch",
    "TacticalProfileQuery",
    "build_tactical_player_profile",
    "build_tactical_profile_batch",
    "canonical_historical_observation_json",
    "canonical_player_profile_json",
    "canonical_profile_batch_json",
    "canonical_profile_query_json",
    "historical_observation_fingerprint",
    "profile_query_fingerprint",
    "tactical_player_profile_fingerprint",
    "tactical_profile_batch_fingerprint",
    "validate_development_queries",
    "validate_tactical_feature_aggregate",
    "validate_tactical_historical_observation",
    "validate_tactical_player_profile",
    "validate_tactical_profile_batch",
    "validate_tactical_profile_query",
    "wilson_interval",
]
