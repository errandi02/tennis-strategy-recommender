"""Contrato publico de recomendaciones tacticas explicables multipatron.

Proyeccion presentacional, determinista e inmutable de un
``TacticalPrioritizationResult`` ya calculado y validado. No recalcula
evidencia, scores ni historia, no lee datos, no expone identidades de
jugador, rival, partido, query ni upstream fingerprints, y no produce un
ranking global entre patrones.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
import re
from types import MappingProxyType
from typing import Final

from src.recommender.tactical_feature_encoder import (
    P02_FEATURE_NAMES,
    P04_FEATURE_NAMES,
    P05_FEATURE_NAMES,
    P06_FEATURE_NAMES,
)
from src.recommender.tactical_matchup_evidence import (
    TacticalCategoryEvidenceState,
    TacticalEvidenceScope,
    tactical_wilson_interval,
)
from src.recommender.tactical_prioritization import (
    SCORE_FORMULA,
    UNCERTAINTY_METHOD,
    TacticalCandidateState,
    TacticalEncodingPolicy,
    TacticalPrioritizationResult,
    default_tactical_scoring_policy,
    explain_tactical_candidate,
    validate_tactical_prioritization_result,
)


PUBLIC_CONTRACT_VERSION: Final = "1.0.0"
METHODOLOGY: Final = "descriptive_observational"
COMBINATION: Final = "equal_weight_executor_opponent"
EXECUTOR_WEIGHT: Final = 0.5
OPPONENT_WEIGHT: Final = 0.5
ENCODER_POLICY: Final = "component_only"
EVIDENCE_SCOPE: Final = "global_only"
MINIMUM_LABELED_ACTIVATIONS: Final = 50
MINIMUM_DISTINCT_MATCHES: Final = 5
REQUESTED_TOP_K: Final = 3
RANKING_SCOPE: Final = "independent_within_pattern"
GLOBAL_CROSS_PATTERN_RANKING: Final = False
PUBLIC_FINGERPRINT_DOMAIN: Final = b"tennis-public-tactical-recommendation\x00"
PUBLIC_LIMITATIONS: Final = (
    "observational_not_causal",
    "historical_execution_not_prescriptive",
    "uncertainty_envelope_not_joint_confidence_interval",
    "coverage_and_annotation_may_bias_results",
    "no_global_cross_pattern_ranking",
    "sealed_test_not_evaluated",
)

PUBLIC_PATTERN_ORDER: Final = ("P02", "P04", "P05", "P06")
_PUBLIC_PATTERN_CONTRACT: Final = MappingProxyType(
    {
        "P02": ("first_serve_direction", "server"),
        "P04": ("initial_return_direction", "returner"),
        "P05": ("initial_return_depth", "returner"),
        "P06": ("initial_return_shot_type", "returner"),
    }
)
_PUBLIC_FEATURES: Final = MappingProxyType(
    {
        "P02": P02_FEATURE_NAMES,
        "P04": P04_FEATURE_NAMES,
        "P05": P05_FEATURE_NAMES,
        "P06": P06_FEATURE_NAMES,
    }
)
_PUBLIC_CATALOG: Final = MappingProxyType(
    {
        pattern_id: tuple(
            feature.split(".", 2)[2] for feature in features
        )
        for pattern_id, features in _PUBLIC_FEATURES.items()
    }
)
_PUBLIC_CATEGORY_TO_FEATURE: Final = MappingProxyType(
    {
        pattern_id: MappingProxyType(
            {
                feature.split(".", 2)[2]: feature
                for feature in features
            }
        )
        for pattern_id, features in _PUBLIC_FEATURES.items()
    }
)
_FORBIDDEN_CATEGORIES: Final = frozenset({"0", "q", "unknown", "censored"})
_CARD_STATUS_REASONS: Final = MappingProxyType(
    {
        "available": ("all_categories_scored",),
        "partially_available": ("some_categories_scored",),
        "not_available": ("no_category_scored",),
    }
)
_RESPONSE_STATUS_REASONS: Final = MappingProxyType(
    {
        "available": ("all_requested_patterns_have_scored_candidates",),
        "partially_available": ("some_requested_patterns_have_scored_candidates",),
        "not_available": ("no_requested_pattern_has_scored_candidates",),
    }
)
_UPSTREAM_OPTION_STATE: Final = MappingProxyType(
    {
        "ranked": TacticalCandidateState.SCORED,
        "abstained_insufficient_evidence": TacticalCandidateState.INSUFFICIENT_EVIDENCE,
        "not_applicable": TacticalCandidateState.NOT_APPLICABLE,
        "not_available": TacticalCandidateState.NOT_AVAILABLE,
    }
)
_OPTION_REASON_DOMAIN: Final = MappingProxyType(
    {
        "ranked": (
            ("both_perspectives_available", "scored_equal_weight_evidence"),
        ),
        "abstained_insufficient_evidence": (
            ("executor_evidence_insufficient",),
            ("opponent_allowed_evidence_insufficient",),
            ("both_perspectives_insufficient",),
        ),
        "not_applicable": (
            ("pattern_not_applicable_to_requested_serves",),
        ),
        "not_available": (
            ("both_perspectives_not_available",),
        ),
    }
)
CARD_RECONCILIATIONS: Final = MappingProxyType(
    {
        "catalog_complete": True,
        "options_in_catalog_order": True,
        "ranked_and_abstained_partition": True,
        "rank_positions_reconciled": True,
        "tie_expansions_preserved": True,
        "counts_reconciled": True,
    }
)
RESPONSE_RECONCILIATIONS: Final = MappingProxyType(
    {
        "policy_frozen": True,
        "four_scoreable_patterns_ordered": True,
        "no_cross_pattern_ranking": True,
        "status_reconciled": True,
        "identity_and_upstream_fingerprints_redacted": True,
        "descriptive_not_causal": True,
    }
)

_SHA256 = re.compile(r"^[0-9A-F]{64}$")
_SAFE_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_PUBLIC_KEYS: Final = frozenset(
    {
        "match_id",
        "match_ids",
        "match",
        "point",
        "point_number",
        "point_numbers",
        "player",
        "players",
        "player_name",
        "opponent",
        "opponents",
        "opponent_name",
        "server_player",
        "returner_player",
        "serve_number",
        "serve_numbers",
        "sequence",
        "sequence_text",
        "raw_sequence",
        "token",
        "tokens",
        "span",
        "spans",
        "first_serve",
        "second_serve",
        "date",
        "dates",
        "as_of",
        "as_of_date",
        "target_date",
        "effective_date",
        "timestamp",
        "timestamps",
        "path",
        "paths",
        "individual_outcome",
        "point_winner",
        "outcome",
        "outcomes",
        "query_fingerprint",
        "schema_fingerprint",
        "source_evidence_fingerprint",
        "candidate_fingerprint",
        "scoring_policy_fingerprint",
        "evidence_fingerprint",
        "test",
        "tests",
        "sealed",
        "evaluation",
        "evaluations",
    }
)
_FORBIDDEN_TEXT_FRAGMENTS: Final = ("file://", "../", "..\\")
_WILSON_RECOMPUTE_FEATURE: Final = "p02.first_serve_direction.4"


class TacticalRecommendationContractError(ValueError):
    """Fallo cerrado del contrato publico de recomendaciones tacticas."""


class TacticalRecommendationStatus(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    NOT_AVAILABLE = "not_available"


class TacticalPatternCardStatus(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    NOT_AVAILABLE = "not_available"


class TacticalOptionStatus(str, Enum):
    RANKED = "ranked"
    ABSTAINED_INSUFFICIENT_EVIDENCE = "abstained_insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"
    NOT_AVAILABLE = "not_available"


_OPTION_STATUS_VALUES = tuple(item.value for item in TacticalOptionStatus)
_CARD_STATUS_VALUES = tuple(item.value for item in TacticalPatternCardStatus)
_RESPONSE_STATUS_VALUES = tuple(item.value for item in TacticalRecommendationStatus)


def _integrity_check() -> None:
    if tuple(_PUBLIC_CATALOG) != PUBLIC_PATTERN_ORDER:
        raise RuntimeError("El catalogo publico no coincide con el orden contractual.")
    for pattern_id in PUBLIC_PATTERN_ORDER:
        categories = _PUBLIC_CATALOG[pattern_id]
        if len(set(categories)) != len(categories):
            raise RuntimeError("El catalogo publico contiene categorias duplicadas.")
        if any(type(category) is not str for category in categories):
            raise RuntimeError("El catalogo publico contiene categorias no textuales.")
        if set(categories) & _FORBIDDEN_CATEGORIES:
            raise RuntimeError("El catalogo publico contiene categorias no tacticas.")
        if set(_PUBLIC_CATEGORY_TO_FEATURE[pattern_id]) != set(categories):
            raise RuntimeError("El mapa categoria-feature no reconcilia.")


_integrity_check()


@dataclass(frozen=True)
class PublicEvidenceComponent:
    perspective: str
    scope: str
    evidence_state: str
    labeled_activations: int
    successes: int
    failures: int
    distinct_matches: int
    success_rate: float | None
    wilson_lower: float | None
    wilson_upper: float | None


@dataclass(frozen=True)
class TacticalRecommendationOption:
    pattern_id: str
    category: str
    tactical_opportunity: str
    actor: str
    status: str
    reason_codes: tuple[str, ...]
    executor_evidence: PublicEvidenceComponent
    opponent_allowed_evidence: PublicEvidenceComponent
    score_formula: str
    score: float | None
    descriptive_uncertainty_envelope: tuple[float, float] | None
    rank_position: int | None
    tie_group: int | None
    canonical_explanation: str

    def __post_init__(self) -> None:
        validate_tactical_recommendation_option(self)


@dataclass(frozen=True)
class TacticalPatternCard:
    pattern_id: str
    actor: str
    tactical_opportunity: str
    status: str
    status_reason_codes: tuple[str, ...]
    categories: tuple[str, ...]
    options: tuple[TacticalRecommendationOption, ...]
    ranked_options: tuple[TacticalRecommendationOption, ...]
    top_options: tuple[TacticalRecommendationOption, ...]
    abstained_options: tuple[TacticalRecommendationOption, ...]
    requested_top_k: int
    effective_top_k: int
    tie_expanded: bool
    tie_group_count: int
    total_options: int
    scored_options: int
    abstained_options_count: int
    reconciliations: MappingProxyType

    def __post_init__(self) -> None:
        validate_tactical_pattern_card(self)


@dataclass(frozen=True)
class TacticalRecommendationResponse:
    contract_version: str
    status: str
    status_reason_codes: tuple[str, ...]
    methodology: str
    combination: str
    score_formula: str
    uncertainty_method: str
    executor_weight: float
    opponent_weight: float
    encoder_policy: str
    evidence_scope: str
    minimum_labeled_activations: int
    minimum_distinct_matches: int
    requested_top_k: int
    ranking_scope: str
    global_cross_pattern_ranking: bool
    cards: tuple[TacticalPatternCard, ...]
    limitations: tuple[str, ...]
    reconciliations: MappingProxyType
    fingerprint: str

    def __post_init__(self) -> None:
        validate_public_tactical_recommendation(self)


def _strict_text(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _SAFE_TEXT.fullmatch(value) is None
    ):
        raise TacticalRecommendationContractError(
            f"{field} debe ser str exacto, no vacio y sin espacios externos."
        )
    return value


def _strict_count(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise TacticalRecommendationContractError(
            f"{field} debe ser int real {'positivo' if positive else 'no negativo'}."
        )
    return value


def _unit(value: object, field: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise TacticalRecommendationContractError(
            f"{field} debe ser float finito en [0, 1]."
        )
    return value


def _optional_unit(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _unit(value, field)


def _strict_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise TacticalRecommendationContractError(f"{field} debe ser bool real.")
    return value


def _sha256_text(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TacticalRecommendationContractError(
            f"{field} debe ser SHA-256 mayuscula hexagonal."
        )
    return value


def _feature_name(pattern_id: str, category: str) -> str:
    try:
        return _PUBLIC_CATEGORY_TO_FEATURE[pattern_id][category]
    except (KeyError, TypeError):
        raise TacticalRecommendationContractError(
            "Categoria fuera del catalogo cerrado del patron."
        ) from None


def _option_state_from_evidence(
    executor: PublicEvidenceComponent,
    opponent: PublicEvidenceComponent,
) -> tuple[str, tuple[str, ...]]:
    available = TacticalCategoryEvidenceState.AVAILABLE.value
    not_applicable = TacticalCategoryEvidenceState.NOT_APPLICABLE.value
    not_available = TacticalCategoryEvidenceState.NOT_AVAILABLE.value
    executor_available = executor.evidence_state == available
    opponent_available = opponent.evidence_state == available
    if executor_available and opponent_available:
        return (
            TacticalOptionStatus.RANKED.value,
            ("both_perspectives_available", "scored_equal_weight_evidence"),
        )
    if executor.evidence_state == not_applicable and opponent.evidence_state == not_applicable:
        return (
            TacticalOptionStatus.NOT_APPLICABLE.value,
            ("pattern_not_applicable_to_requested_serves",),
        )
    if executor.evidence_state == not_available and opponent.evidence_state == not_available:
        return (
            TacticalOptionStatus.NOT_AVAILABLE.value,
            ("both_perspectives_not_available",),
        )
    if executor_available:
        return (
            TacticalOptionStatus.ABSTAINED_INSUFFICIENT_EVIDENCE.value,
            ("opponent_allowed_evidence_insufficient",),
        )
    if opponent_available:
        return (
            TacticalOptionStatus.ABSTAINED_INSUFFICIENT_EVIDENCE.value,
            ("executor_evidence_insufficient",),
        )
    return (
        TacticalOptionStatus.ABSTAINED_INSUFFICIENT_EVIDENCE.value,
        ("both_perspectives_insufficient",),
    )


def _public_rank_key(pattern_id: str, option: TacticalRecommendationOption) -> tuple[object, ...]:
    if (
        option.score is None
        or option.descriptive_uncertainty_envelope is None
    ):
        raise TacticalRecommendationContractError("Solo se ordenan opciones ranked.")
    lower, upper = option.descriptive_uncertainty_envelope
    width = upper - lower
    floor = min(
        option.executor_evidence.labeled_activations,
        option.opponent_allowed_evidence.labeled_activations,
    )
    return (
        -option.score,
        -lower,
        width,
        -floor,
        _feature_name(pattern_id, option.category),
    )


def _public_tie_key(pattern_id: str, option: TacticalRecommendationOption) -> tuple[object, ...]:
    if (
        option.score is None
        or option.descriptive_uncertainty_envelope is None
    ):
        raise TacticalRecommendationContractError("Solo se agrupan opciones ranked.")
    lower, upper = option.descriptive_uncertainty_envelope
    width = upper - lower
    floor = min(
        option.executor_evidence.labeled_activations,
        option.opponent_allowed_evidence.labeled_activations,
    )
    return (option.score, lower, width, floor)


def _expected_explanation_text(
    pattern_id: str, category: str, upstream_state: TacticalCandidateState
) -> str:
    if upstream_state is TacticalCandidateState.SCORED:
        return (
            f"Ranking descriptivo para {pattern_id}:{category}; "
            "combina por igual dos tasas historicas comparables y no implica causalidad."
        )
    return (
        f"Sin score descriptivo para {pattern_id}:{category}; "
        f"estado {upstream_state.value} y sin imputacion."
    )


def validate_public_evidence_component(component: PublicEvidenceComponent) -> None:
    if type(component) is not PublicEvidenceComponent:
        raise TypeError("component debe ser PublicEvidenceComponent exacto.")
    if (
        type(component.perspective) is not str
        or component.perspective not in {"executor", "opponent_allowed"}
    ):
        raise TacticalRecommendationContractError("Perspectiva fuera del dominio.")
    if (
        type(component.scope) is not str
        or component.scope != TacticalEvidenceScope.GLOBAL.value
    ):
        raise TacticalRecommendationContractError("Scope debe ser global.")
    if (
        type(component.evidence_state) is not str
        or component.evidence_state
        not in tuple(item.value for item in TacticalCategoryEvidenceState)
    ):
        raise TacticalRecommendationContractError("Estado de evidencia upstream invalido.")
    for field in ("labeled_activations", "successes", "failures", "distinct_matches"):
        _strict_count(getattr(component, field), field)
    if component.successes + component.failures != component.labeled_activations:
        raise TacticalRecommendationContractError(
            "Successes y failures no reconcilian las activations."
        )
    if component.distinct_matches > component.labeled_activations:
        raise TacticalRecommendationContractError(
            "Partidos distintos superan las activations etiquetadas."
        )
    values = (component.success_rate, component.wilson_lower, component.wilson_upper)
    for field, value in zip(
        ("success_rate", "wilson_lower", "wilson_upper"), values
    ):
        _optional_unit(value, field)
    if component.labeled_activations == 0:
        if any(value is not None for value in values):
            raise TacticalRecommendationContractError(
                "Sin activations no puede haber tasa ni Wilson."
            )
    else:
        if any(value is None for value in values):
            raise TacticalRecommendationContractError(
                "Con activations deben existir tasa y Wilson."
            )
        if component.success_rate != component.successes / component.labeled_activations:
            raise TacticalRecommendationContractError("La tasa no reconcilia.")
        rate, lower, upper = tactical_wilson_interval(
            component.successes,
            component.labeled_activations,
            feature_name=_WILSON_RECOMPUTE_FEATURE,
        )
        if (component.success_rate, component.wilson_lower, component.wilson_upper) != (
            rate,
            lower,
            upper,
        ):
            raise TacticalRecommendationContractError(
                "Tasa o Wilson no reconcilian con el upstream."
            )


def validate_tactical_recommendation_option(
    option: TacticalRecommendationOption,
) -> None:
    if type(option) is not TacticalRecommendationOption:
        raise TypeError("option debe ser TacticalRecommendationOption exacta.")
    if (
        type(option.pattern_id) is not str
        or option.pattern_id not in PUBLIC_PATTERN_ORDER
    ):
        raise TacticalRecommendationContractError(
            "pattern_id fuera del catalogo publico cerrado."
        )
    opportunity, actor = _PUBLIC_PATTERN_CONTRACT[option.pattern_id]
    if option.category not in _PUBLIC_CATALOG[option.pattern_id] or option.category in _FORBIDDEN_CATEGORIES:
        raise TacticalRecommendationContractError("Categoria fuera del catalogo cerrado.")
    if (
        type(option.tactical_opportunity) is not str
        or option.tactical_opportunity != opportunity
    ):
        raise TacticalRecommendationContractError("Oportunidad del patron invalida.")
    if type(option.actor) is not str or option.actor != actor:
        raise TacticalRecommendationContractError("Actor del patron invalido.")
    if type(option.status) is not str or option.status not in _OPTION_STATUS_VALUES:
        raise TacticalRecommendationContractError("Estado de opcion invalido.")
    if (
        type(option.reason_codes) is not tuple
        or any(type(item) is not str for item in option.reason_codes)
        or option.reason_codes not in _OPTION_REASON_DOMAIN[option.status]
    ):
        raise TacticalRecommendationContractError("Reason codes no reconcilian.")
    validate_public_evidence_component(option.executor_evidence)
    validate_public_evidence_component(option.opponent_allowed_evidence)
    if option.executor_evidence.perspective != "executor":
        raise TacticalRecommendationContractError("Falta la perspectiva executor.")
    if option.opponent_allowed_evidence.perspective != "opponent_allowed":
        raise TacticalRecommendationContractError("Falta la perspectiva opponent_allowed.")
    expected_status, expected_reasons = _option_state_from_evidence(
        option.executor_evidence, option.opponent_allowed_evidence
    )
    if option.status != expected_status or option.reason_codes != expected_reasons:
        raise TacticalRecommendationContractError(
            "Estado de opcion no reconcilia con las dos evidencias."
        )
    if option.status == TacticalOptionStatus.RANKED.value:
        if (
            option.score is None
            or option.descriptive_uncertainty_envelope is None
            or option.rank_position is None
            or option.tie_group is None
        ):
            raise TacticalRecommendationContractError("Opcion ranked incompleta.")
        _unit(option.score, "score")
        envelope = option.descriptive_uncertainty_envelope
        if type(envelope) is not tuple or len(envelope) != 2:
            raise TacticalRecommendationContractError(
                "Envolvente descriptiva debe ser tuple de dos bounds."
            )
        executor = option.executor_evidence
        opponent = option.opponent_allowed_evidence
        expected_score = (
            0.5 * executor.success_rate + 0.5 * opponent.success_rate
        )
        expected_lower = (
            0.5 * executor.wilson_lower + 0.5 * opponent.wilson_lower
        )
        expected_upper = (
            0.5 * executor.wilson_upper + 0.5 * opponent.wilson_upper
        )
        if option.score != expected_score:
            raise TacticalRecommendationContractError(
                "El score no reconcilia con la combinacion 50/50."
            )
        if envelope[0] != expected_lower or envelope[1] != expected_upper:
            raise TacticalRecommendationContractError(
                "La envolvente descriptiva no reconcilia con los Wilson 50/50."
            )
        if not envelope[0] <= option.score <= envelope[1]:
            raise TacticalRecommendationContractError(
                "La envolvente descriptiva no contiene el score."
            )
        _strict_count(option.rank_position, "rank_position", positive=True)
        _strict_count(option.tie_group, "tie_group", positive=True)
    else:
        if (
            option.score is not None
            or option.descriptive_uncertainty_envelope is not None
            or option.rank_position is not None
            or option.tie_group is not None
        ):
            raise TacticalRecommendationContractError(
                "Abstencion no puede conservar score, envolvente ni ranking."
            )
    if option.score_formula != SCORE_FORMULA:
        raise TacticalRecommendationContractError("Formula fuera del contrato.")
    upstream_state = _UPSTREAM_OPTION_STATE[option.status]
    _strict_text(option.canonical_explanation, "canonical_explanation")
    if (
        option.canonical_explanation
        != _expected_explanation_text(option.pattern_id, option.category, upstream_state)
    ):
        raise TacticalRecommendationContractError(
            "Explicacion canonica no reconcilia."
        )


def validate_tactical_pattern_card(card: TacticalPatternCard) -> None:
    if type(card) is not TacticalPatternCard:
        raise TypeError("card debe ser TacticalPatternCard exacta.")
    if type(card.pattern_id) is not str or card.pattern_id not in PUBLIC_PATTERN_ORDER:
        raise TacticalRecommendationContractError(
            "pattern_id fuera del catalogo publico cerrado."
        )
    opportunity, actor = _PUBLIC_PATTERN_CONTRACT[card.pattern_id]
    if (
        type(card.tactical_opportunity) is not str
        or card.tactical_opportunity != opportunity
    ):
        raise TacticalRecommendationContractError("Oportunidad del patron invalida.")
    if type(card.actor) is not str or card.actor != actor:
        raise TacticalRecommendationContractError("Actor del patron invalido.")
    if (
        type(card.status) is not str
        or card.status not in _CARD_STATUS_VALUES
    ):
        raise TacticalRecommendationContractError("Estado de tarjeta invalido.")
    if (
        type(card.status_reason_codes) is not tuple
        or any(type(item) is not str for item in card.status_reason_codes)
        or card.status_reason_codes != _CARD_STATUS_REASONS[card.status]
    ):
        raise TacticalRecommendationContractError("Reason codes de tarjeta no reconcilian.")
    catalog = _PUBLIC_CATALOG[card.pattern_id]
    if type(card.categories) is not tuple or card.categories != catalog:
        raise TacticalRecommendationContractError(
            "El catalogo de categorias debe ser completo y en orden contractual."
        )
    if type(card.options) is not tuple or len(card.options) != len(catalog):
        raise TacticalRecommendationContractError(
            "La tarjeta debe proyectar una opcion por categoria."
        )
    for position, option in enumerate(card.options):
        validate_tactical_recommendation_option(option)
        if option.pattern_id != card.pattern_id:
            raise TacticalRecommendationContractError("Opcion de otro patron.")
        if option.category != catalog[position]:
            raise TacticalRecommendationContractError(
                "Las opciones no conservan el orden contractual del catalogo."
            )
    for field in ("ranked_options", "top_options", "abstained_options"):
        if type(getattr(card, field)) is not tuple:
            raise TacticalRecommendationContractError(
                f"{field} debe ser tuple inmutable."
            )
    ranked = tuple(
        option
        for option in card.options
        if option.status == TacticalOptionStatus.RANKED.value
    )
    abstained = tuple(
        option
        for option in card.options
        if option.status != TacticalOptionStatus.RANKED.value
    )
    ranked_categories = {item.category for item in card.ranked_options}
    abstained_categories = {item.category for item in card.abstained_options}
    option_categories = {item.category for item in card.options}
    if not (
        not ranked_categories & abstained_categories
        and ranked_categories | abstained_categories == option_categories
    ):
        raise TacticalRecommendationContractError(
            "Ranked y abstenciones no particionan las opciones."
        )
    expected_ranked = tuple(sorted(ranked, key=lambda item: _public_rank_key(card.pattern_id, item)))
    if (
        len(card.ranked_options) != len(expected_ranked)
        or [item.category for item in card.ranked_options]
        != [item.category for item in expected_ranked]
    ):
        raise TacticalRecommendationContractError("Orden del ranking invalido.")
    tie_group = 0
    previous: tuple[object, ...] | None = None
    for position, item in enumerate(expected_ranked, start=1):
        current = _public_tie_key(card.pattern_id, item)
        if current != previous:
            tie_group += 1
            previous = current
        if item.rank_position != position or item.tie_group != tie_group:
            raise TacticalRecommendationContractError("Posicion o empate invalido.")
    if not (
        type(card.tie_group_count) is int
        and card.tie_group_count >= 0
        and card.tie_group_count == tie_group
    ):
        raise TacticalRecommendationContractError("Conteo de empates no reconcilia.")
    expected_abstained = tuple(
        sorted(abstained, key=lambda item: _feature_name(card.pattern_id, item.category))
    )
    if (
        len(card.abstained_options) != len(expected_abstained)
        or [item.category for item in card.abstained_options]
        != [item.category for item in expected_abstained]
    ):
        raise TacticalRecommendationContractError("Orden de abstenciones invalido.")
    if not (
        type(card.requested_top_k) is int and card.requested_top_k == REQUESTED_TOP_K
    ):
        raise TacticalRecommendationContractError("top_k fuera del contrato congelado.")
    scored_count = len(expected_ranked)
    if scored_count == 0:
        expected_top: tuple[TacticalRecommendationOption, ...] = ()
    elif scored_count <= card.requested_top_k:
        expected_top = expected_ranked
    else:
        boundary = expected_ranked[card.requested_top_k - 1].tie_group
        expected_top = tuple(
            item for item in expected_ranked if item.tie_group <= boundary
        )
    if (
        len(card.top_options) != len(expected_top)
        or [item.category for item in card.top_options]
        != [item.category for item in expected_top]
    ):
        raise TacticalRecommendationContractError("Top-k no reconcilia.")
    if (
        type(card.effective_top_k) is not int
        or card.effective_top_k != len(expected_top)
    ):
        raise TacticalRecommendationContractError("effective_top_k no reconcilia.")
    expected_expanded = len(expected_top) > min(card.requested_top_k, scored_count)
    if (
        type(card.tie_expanded) is not bool
        or card.tie_expanded != expected_expanded
    ):
        raise TacticalRecommendationContractError("Expansion de empate invalida.")
    for field, expected_value in (
        ("total_options", len(catalog)),
        ("scored_options", scored_count),
        ("abstained_options_count", len(abstained)),
    ):
        value = getattr(card, field)
        if type(value) is not int or value != expected_value:
            raise TacticalRecommendationContractError("Conteos de tarjeta no reconcilian.")
    if scored_count == 0:
        expected_status = TacticalPatternCardStatus.NOT_AVAILABLE.value
    elif not abstained:
        expected_status = TacticalPatternCardStatus.AVAILABLE.value
    else:
        expected_status = TacticalPatternCardStatus.PARTIALLY_AVAILABLE.value
    if card.status != expected_status:
        raise TacticalRecommendationContractError(
            "Estado de tarjeta no reconcilia con las opciones."
        )
    if (
        type(card.reconciliations) is not MappingProxyType
        or dict(card.reconciliations) != dict(CARD_RECONCILIATIONS)
    ):
        raise TacticalRecommendationContractError("Reconciliaciones de tarjeta invalidas.")


def _require_frozen_policy(source: TacticalPrioritizationResult) -> None:
    if source.policy != default_tactical_scoring_policy():
        raise TacticalRecommendationContractError(
            "La politica de scoring no coincide con la politica congelada."
        )
    query = source.matchup_query
    if query.encoder_policy is not TacticalEncodingPolicy.COMPONENT_ONLY:
        raise TacticalRecommendationContractError(
            "encoder policy debe ser component_only."
        )
    if query.requested_patterns != PUBLIC_PATTERN_ORDER:
        raise TacticalRecommendationContractError(
            "Los patrones solicitados no coinciden con el contrato congelado."
        )
    if query.minimum_labeled_attempts != MINIMUM_LABELED_ACTIVATIONS:
        raise TacticalRecommendationContractError(
            "minimum labeled activations fuera del contrato congelado."
        )
    if query.minimum_matches != MINIMUM_DISTINCT_MATCHES:
        raise TacticalRecommendationContractError(
            "minimum distinct matches fuera del contrato congelado."
        )
    for ranking in source.rankings:
        if ranking.requested_top_k != REQUESTED_TOP_K:
            raise TacticalRecommendationContractError(
                "top_k fuera del contrato congelado."
            )


def validate_public_tactical_recommendation(
    response: TacticalRecommendationResponse,
    *,
    source: TacticalPrioritizationResult | None = None,
) -> None:
    if type(response) is not TacticalRecommendationResponse:
        raise TypeError(
            "response debe ser TacticalRecommendationResponse exacta."
        )
    if (
        type(response.contract_version) is not str
        or response.contract_version != PUBLIC_CONTRACT_VERSION
    ):
        raise TacticalRecommendationContractError("contract_version invalida.")
    if (
        type(response.status) is not str
        or response.status not in _RESPONSE_STATUS_VALUES
    ):
        raise TacticalRecommendationContractError("Estado global invalido.")
    if (
        type(response.status_reason_codes) is not tuple
        or any(type(item) is not str for item in response.status_reason_codes)
        or response.status_reason_codes != _RESPONSE_STATUS_REASONS[response.status]
    ):
        raise TacticalRecommendationContractError(
            "Reason codes globales no reconcilian."
        )
    if response.methodology != METHODOLOGY:
        raise TacticalRecommendationContractError("Metodologia fuera del contrato.")
    if response.combination != COMBINATION:
        raise TacticalRecommendationContractError("Combinacion fuera del contrato.")
    if response.score_formula != SCORE_FORMULA:
        raise TacticalRecommendationContractError("Formula fuera del contrato.")
    if response.uncertainty_method != UNCERTAINTY_METHOD:
        raise TacticalRecommendationContractError("Incertidumbre fuera del contrato.")
    for field in ("executor_weight", "opponent_weight"):
        value = getattr(response, field)
        if type(value) is not float or value != 0.5:
            raise TacticalRecommendationContractError(
                f"{field} debe ser 0.5 exacto."
            )
    if response.encoder_policy != ENCODER_POLICY:
        raise TacticalRecommendationContractError("encoder policy fuera del contrato.")
    if response.evidence_scope != EVIDENCE_SCOPE:
        raise TacticalRecommendationContractError("evidence scope fuera del contrato.")
    if (
        type(response.minimum_labeled_activations) is not int
        or response.minimum_labeled_activations != MINIMUM_LABELED_ACTIVATIONS
    ):
        raise TacticalRecommendationContractError("Mínimo de activations invalido.")
    if (
        type(response.minimum_distinct_matches) is not int
        or response.minimum_distinct_matches != MINIMUM_DISTINCT_MATCHES
    ):
        raise TacticalRecommendationContractError("Mínimo de partidos invalido.")
    if (
        type(response.requested_top_k) is not int
        or response.requested_top_k != REQUESTED_TOP_K
    ):
        raise TacticalRecommendationContractError("top_k fuera del contrato.")
    if response.ranking_scope != RANKING_SCOPE:
        raise TacticalRecommendationContractError("ranking scope fuera del contrato.")
    if (
        type(response.global_cross_pattern_ranking) is not bool
        or response.global_cross_pattern_ranking is not False
    ):
        raise TacticalRecommendationContractError(
            "El contrato publico no admite ranking global."
        )
    if type(response.cards) is not tuple or len(response.cards) != len(PUBLIC_PATTERN_ORDER):
        raise TacticalRecommendationContractError(
            "La respuesta debe contener exactamente cuatro tarjetas."
        )
    if tuple(item.pattern_id for item in response.cards) != PUBLIC_PATTERN_ORDER:
        raise TacticalRecommendationContractError(
            "Las tarjetas deben existir una por patron y en orden contractual."
        )
    for card in response.cards:
        validate_tactical_pattern_card(card)
    scored_patterns = sum(1 for card in response.cards if card.scored_options > 0)
    if scored_patterns == len(response.cards):
        expected_status = TacticalRecommendationStatus.AVAILABLE.value
        expected_reasons = ("all_requested_patterns_have_scored_candidates",)
    elif scored_patterns:
        expected_status = TacticalRecommendationStatus.PARTIALLY_AVAILABLE.value
        expected_reasons = ("some_requested_patterns_have_scored_candidates",)
    else:
        expected_status = TacticalRecommendationStatus.NOT_AVAILABLE.value
        expected_reasons = ("no_requested_pattern_has_scored_candidates",)
    if response.status != expected_status or response.status_reason_codes != expected_reasons:
        raise TacticalRecommendationContractError(
            "El estado global no reconcilia con las tarjetas."
        )
    if (
        type(response.limitations) is not tuple
        or response.limitations != PUBLIC_LIMITATIONS
    ):
        raise TacticalRecommendationContractError("Limitaciones fuera del catalogo.")
    if (
        type(response.reconciliations) is not MappingProxyType
        or dict(response.reconciliations) != dict(RESPONSE_RECONCILIATIONS)
    ):
        raise TacticalRecommendationContractError("Reconciliaciones globales invalidas.")
    _sha256_text(response.fingerprint, "fingerprint")
    if (
        response.fingerprint
        != _public_fingerprint_of(response.status, response.status_reason_codes, response.cards)
    ):
        raise TacticalRecommendationContractError("Fingerprint no reconcilia.")
    _validate_public_tree(_response_structure(response))
    if source is not None:
        if type(source) is not TacticalPrioritizationResult:
            raise TypeError("source debe ser TacticalPrioritizationResult exacto.")
        validate_tactical_prioritization_result(source)
        _require_frozen_policy(source)
        expected_cards = tuple(_public_card(item) for item in source.rankings)
        expected_status = source.state.value
        expected_reasons = source.reason_codes
        expected = _response_core_structure(
            expected_status, expected_reasons, expected_cards
        )
        expected["fingerprint"] = _public_fingerprint_of(
            expected_status, expected_reasons, expected_cards
        )
        if _response_structure(response) != expected:
            raise TacticalRecommendationContractError(
                "La respuesta no reconstruye canonicamente desde su source."
            )


def _validate_public_tree(value: object, *, key: str | None = None) -> None:
    if key is not None:
        lowered = key.lower()
        if (
            lowered in _FORBIDDEN_PUBLIC_KEYS
            or lowered.startswith(("test_", "sealed_"))
            or "evaluation" in lowered
        ):
            raise TacticalRecommendationContractError("Clave publica no autorizada.")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise TacticalRecommendationContractError("Valor no finito.")
        return
    if type(value) is str:
        lowered = value.lower()
        if (
            any(fragment in lowered for fragment in _FORBIDDEN_TEXT_FRAGMENTS)
            or _WINDOWS_PATH.match(value) is not None
            or value.startswith(("/", "\\\\"))
            or any(ord(character) < 32 for character in value)
        ):
            raise TacticalRecommendationContractError("Texto operativo no autorizado.")
        return
    if type(value) is list:
        for item in value:
            _validate_public_tree(item)
        return
    if type(value) is dict:
        for child_key, child_value in value.items():
            if type(child_key) is not str:
                raise TacticalRecommendationContractError("Clave JSON no textual.")
            _validate_public_tree(child_value, key=child_key)
        return
    raise TacticalRecommendationContractError("Objeto publico no contractual.")


def _canonical_bytes(structure: dict[str, object]) -> bytes:
    _validate_public_tree(structure)
    return json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _component_structure(component: PublicEvidenceComponent) -> dict[str, object]:
    return {
        "distinct_matches": component.distinct_matches,
        "evidence_state": component.evidence_state,
        "failures": component.failures,
        "labeled_activations": component.labeled_activations,
        "perspective": component.perspective,
        "scope": component.scope,
        "success_rate": component.success_rate,
        "successes": component.successes,
        "wilson_lower": component.wilson_lower,
        "wilson_upper": component.wilson_upper,
    }


def _option_structure(option: TacticalRecommendationOption) -> dict[str, object]:
    return {
        "actor": option.actor,
        "canonical_explanation": option.canonical_explanation,
        "category": option.category,
        "descriptive_uncertainty_envelope": (
            None
            if option.descriptive_uncertainty_envelope is None
            else list(option.descriptive_uncertainty_envelope)
        ),
        "executor_evidence": _component_structure(option.executor_evidence),
        "opponent_allowed_evidence": _component_structure(option.opponent_allowed_evidence),
        "pattern_id": option.pattern_id,
        "reason_codes": list(option.reason_codes),
        "rank_position": option.rank_position,
        "score": option.score,
        "score_formula": option.score_formula,
        "status": option.status,
        "tactical_opportunity": option.tactical_opportunity,
        "tie_group": option.tie_group,
    }


def _card_structure(card: TacticalPatternCard) -> dict[str, object]:
    return {
        "abstained_options": [item.category for item in card.abstained_options],
        "abstained_options_count": card.abstained_options_count,
        "actor": card.actor,
        "categories": list(card.categories),
        "effective_top_k": card.effective_top_k,
        "options": [_option_structure(item) for item in card.options],
        "pattern_id": card.pattern_id,
        "ranked_options": [item.category for item in card.ranked_options],
        "reconciliations": dict(card.reconciliations),
        "requested_top_k": card.requested_top_k,
        "scored_options": card.scored_options,
        "status": card.status,
        "status_reason_codes": list(card.status_reason_codes),
        "tactical_opportunity": card.tactical_opportunity,
        "tie_expanded": card.tie_expanded,
        "tie_group_count": card.tie_group_count,
        "top_options": [item.category for item in card.top_options],
        "total_options": card.total_options,
    }


def _response_core_structure(
    status: str,
    status_reason_codes: tuple[str, ...],
    cards: tuple[TacticalPatternCard, ...],
) -> dict[str, object]:
    return {
        "cards": [_card_structure(item) for item in cards],
        "combination": COMBINATION,
        "contract_version": PUBLIC_CONTRACT_VERSION,
        "encoder_policy": ENCODER_POLICY,
        "evidence_scope": EVIDENCE_SCOPE,
        "executor_weight": EXECUTOR_WEIGHT,
        "global_cross_pattern_ranking": GLOBAL_CROSS_PATTERN_RANKING,
        "limitations": list(PUBLIC_LIMITATIONS),
        "methodology": METHODOLOGY,
        "minimum_distinct_matches": MINIMUM_DISTINCT_MATCHES,
        "minimum_labeled_activations": MINIMUM_LABELED_ACTIVATIONS,
        "opponent_weight": OPPONENT_WEIGHT,
        "ranking_scope": RANKING_SCOPE,
        "reconciliations": dict(RESPONSE_RECONCILIATIONS),
        "requested_top_k": REQUESTED_TOP_K,
        "score_formula": SCORE_FORMULA,
        "status": status,
        "status_reason_codes": list(status_reason_codes),
        "uncertainty_method": UNCERTAINTY_METHOD,
    }


def _response_structure(response: TacticalRecommendationResponse) -> dict[str, object]:
    structure = _response_core_structure(
        response.status, response.status_reason_codes, response.cards
    )
    structure["fingerprint"] = response.fingerprint
    return structure


def _public_fingerprint_of(
    status: str,
    status_reason_codes: tuple[str, ...],
    cards: tuple[TacticalPatternCard, ...],
) -> str:
    return sha256(
        PUBLIC_FINGERPRINT_DOMAIN
        + _canonical_bytes(_response_core_structure(status, status_reason_codes, cards))
    ).hexdigest().upper()


def _public_component(summary: object) -> PublicEvidenceComponent:
    return PublicEvidenceComponent(
        perspective=summary.perspective,
        scope=summary.scope,
        evidence_state=summary.evidence_state,
        labeled_activations=summary.labeled_attempts,
        successes=summary.successes,
        failures=summary.failures,
        distinct_matches=summary.distinct_matches,
        success_rate=summary.success_rate,
        wilson_lower=summary.wilson_lower,
        wilson_upper=summary.wilson_upper,
    )


def _public_option(candidate: object) -> TacticalRecommendationOption:
    explanation = explain_tactical_candidate(candidate)
    state = explanation.state
    return TacticalRecommendationOption(
        pattern_id=explanation.pattern_id,
        category=explanation.category,
        tactical_opportunity=explanation.tactical_opportunity,
        actor=explanation.actor,
        status=_UPSTREAM_PUBLIC_STATE[state],
        reason_codes=explanation.reason_codes,
        executor_evidence=_public_component(candidate.executor_evidence),
        opponent_allowed_evidence=_public_component(candidate.opponent_allowed_evidence),
        score_formula=explanation.formula,
        score=explanation.combined_rate,
        descriptive_uncertainty_envelope=(
            (explanation.combined_lower, explanation.combined_upper)
            if explanation.combined_rate is not None
            else None
        ),
        rank_position=candidate.rank_position,
        tie_group=candidate.tie_group,
        canonical_explanation=explanation.canonical_text,
    )


_UPSTREAM_PUBLIC_STATE: Final = MappingProxyType(
    {
        TacticalCandidateState.SCORED: TacticalOptionStatus.RANKED.value,
        TacticalCandidateState.INSUFFICIENT_EVIDENCE: (
            TacticalOptionStatus.ABSTAINED_INSUFFICIENT_EVIDENCE.value
        ),
        TacticalCandidateState.NOT_APPLICABLE: TacticalOptionStatus.NOT_APPLICABLE.value,
        TacticalCandidateState.NOT_AVAILABLE: TacticalOptionStatus.NOT_AVAILABLE.value,
    }
)


def _public_card(ranking: object) -> TacticalPatternCard:
    by_category = {
        item.category: _public_option(item) for item in ranking.candidates
    }
    catalog = _PUBLIC_CATALOG[ranking.pattern_id]
    options = tuple(by_category[item] for item in catalog)
    ranked_options = tuple(
        by_category[item.category] for item in ranking.candidates if item.rank_eligible
    )
    top_options = tuple(by_category[item.category] for item in ranking.top_candidates)
    abstained_options = tuple(
        by_category[item.category]
        for item in ranking.candidates
        if not item.rank_eligible
    )
    return TacticalPatternCard(
        pattern_id=ranking.pattern_id,
        actor=_PUBLIC_PATTERN_CONTRACT[ranking.pattern_id][1],
        tactical_opportunity=ranking.tactical_opportunity,
        status=ranking.state.value,
        status_reason_codes=ranking.reason_codes,
        categories=catalog,
        options=options,
        ranked_options=ranked_options,
        top_options=top_options,
        abstained_options=abstained_options,
        requested_top_k=ranking.requested_top_k,
        effective_top_k=ranking.effective_top_k,
        tie_expanded=ranking.boundary_tie_expanded,
        tie_group_count=ranking.tie_group_count,
        total_options=len(options),
        scored_options=ranking.scored_candidate_count,
        abstained_options_count=ranking.abstained_candidate_count,
        reconciliations=CARD_RECONCILIATIONS,
    )


def build_public_tactical_recommendation(
    prioritization: TacticalPrioritizationResult,
) -> TacticalRecommendationResponse:
    """Proyecta un resultado priorizado validado en una respuesta publico inmutable."""
    if type(prioritization) is not TacticalPrioritizationResult:
        raise TypeError("prioritization debe ser TacticalPrioritizationResult exacto.")
    validate_tactical_prioritization_result(prioritization)
    _require_frozen_policy(prioritization)
    cards = tuple(_public_card(item) for item in prioritization.rankings)
    return TacticalRecommendationResponse(
        contract_version=PUBLIC_CONTRACT_VERSION,
        status=prioritization.state.value,
        status_reason_codes=prioritization.reason_codes,
        methodology=METHODOLOGY,
        combination=COMBINATION,
        score_formula=SCORE_FORMULA,
        uncertainty_method=UNCERTAINTY_METHOD,
        executor_weight=EXECUTOR_WEIGHT,
        opponent_weight=OPPONENT_WEIGHT,
        encoder_policy=ENCODER_POLICY,
        evidence_scope=EVIDENCE_SCOPE,
        minimum_labeled_activations=MINIMUM_LABELED_ACTIVATIONS,
        minimum_distinct_matches=MINIMUM_DISTINCT_MATCHES,
        requested_top_k=REQUESTED_TOP_K,
        ranking_scope=RANKING_SCOPE,
        global_cross_pattern_ranking=GLOBAL_CROSS_PATTERN_RANKING,
        cards=cards,
        limitations=PUBLIC_LIMITATIONS,
        reconciliations=RESPONSE_RECONCILIATIONS,
        fingerprint=_public_fingerprint_of(
            prioritization.state.value, prioritization.reason_codes, cards
        ),
    )


def canonical_tactical_recommendation_json(
    response: TacticalRecommendationResponse,
) -> bytes:
    if type(response) is not TacticalRecommendationResponse:
        raise TypeError("response debe ser TacticalRecommendationResponse exacta.")
    validate_public_tactical_recommendation(response)
    return _canonical_bytes(_response_structure(response))


def tactical_recommendation_fingerprint(response: TacticalRecommendationResponse) -> str:
    if type(response) is not TacticalRecommendationResponse:
        raise TypeError("response debe ser TacticalRecommendationResponse exacta.")
    validate_public_tactical_recommendation(response)
    expected = _public_fingerprint_of(
        response.status, response.status_reason_codes, response.cards
    )
    if response.fingerprint != expected:
        raise TacticalRecommendationContractError("Fingerprint no reconcilia.")
    return expected


__all__ = (
    "PUBLIC_CONTRACT_VERSION",
    "PUBLIC_FINGERPRINT_DOMAIN",
    "PUBLIC_LIMITATIONS",
    "PUBLIC_PATTERN_ORDER",
    "PublicEvidenceComponent",
    "TacticalOptionStatus",
    "TacticalPatternCard",
    "TacticalPatternCardStatus",
    "TacticalRecommendationContractError",
    "TacticalRecommendationOption",
    "TacticalRecommendationResponse",
    "TacticalRecommendationStatus",
    "build_public_tactical_recommendation",
    "canonical_tactical_recommendation_json",
    "tactical_recommendation_fingerprint",
    "validate_public_evidence_component",
    "validate_public_tactical_recommendation",
    "validate_tactical_pattern_card",
    "validate_tactical_recommendation_option",
)
