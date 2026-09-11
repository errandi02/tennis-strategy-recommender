"""Priorizacion tactica descriptiva desde evidencia agregada emparejada.

El modulo es puro: no lee datos, no ajusta modelos y no produce un ranking
global entre oportunidades incompatibles. Cada score combina, con pesos fijos,
la tasa historica del ejecutor y la tasa permitida por el oponente.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
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
)
from src.recommender.tactical_matchup_evidence import (
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    TacticalCategoryEvidence,
    TacticalCategoryEvidenceState,
    TacticalEvidencePerspective,
    TacticalEvidenceScope,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupCategoryEvidence,
    TacticalMatchupCategoryState,
    TacticalMatchupEvidence,
    TacticalMatchupQuery,
    tactical_matchup_category_evidence_fingerprint,
    tactical_matchup_evidence_fingerprint,
    tactical_matchup_query_fingerprint,
    tactical_wilson_interval,
    validate_tactical_matchup_category_evidence,
    validate_tactical_matchup_evidence,
)


PRIORITIZATION_CONTRACT_VERSION: Final = "1.0.0"
SCORING_POLICY_NAME: Final = "equal_weight_descriptive_matchup"
SCORING_POLICY_VERSION: Final = "1.0.0"
SCORE_FORMULA: Final = (
    "combined_rate = 0.5 * executor_success_rate + "
    "0.5 * opponent_allowed_success_rate"
)
UNCERTAINTY_METHOD: Final = (
    "descriptive_uncertainty_envelope_from_two_wilson_intervals"
)
RANK_ORDER_RULE: Final = (
    "combined_rate_desc|combined_lower_desc|combined_width_asc|"
    "evidence_floor_desc|canonical_category_order"
)

PATTERN_ORDER: Final = ("P02", "P04", "P05", "P06", "P09")
PATTERN_CONTRACT: Final = (
    ("P02", "first_serve_direction", "server"),
    ("P04", "initial_return_direction", "returner"),
    ("P05", "initial_return_depth", "returner"),
    ("P06", "initial_return_shot_type", "returner"),
    ("P09", "initial_return_profile", "returner"),
)
PATTERN_FEATURES: Final = (
    ("P02", P02_FEATURE_NAMES),
    ("P04", P04_FEATURE_NAMES),
    ("P05", P05_FEATURE_NAMES),
    ("P06", P06_FEATURE_NAMES),
    ("P09", P09_FEATURE_NAMES),
)
POLICY_PATTERN_CONTRACT: Final = (
    (TacticalEncodingPolicy.COMPONENT_ONLY, ("P02", "P04", "P05", "P06")),
    (TacticalEncodingPolicy.PROFILE_ONLY, ("P02", "P09")),
    (
        TacticalEncodingPolicy.COMPONENTS_AND_PROFILE,
        ("P02", "P04", "P05", "P06", "P09"),
    ),
)

POLICY_RECONCILIATIONS: Final = (
    ("fixed_equal_weights", True),
    ("formula_reconciled", True),
    ("no_fitting_or_smoothing", True),
)
CANDIDATE_RECONCILIATIONS: Final = (
    ("source_identity_reconciled", True),
    ("perspectives_reconciled", True),
    ("score_reconciled", True),
    ("uncertainty_envelope_reconciled", True),
    ("no_partial_score", True),
)
RANKING_RECONCILIATIONS: Final = (
    ("single_pattern_only", True),
    ("candidate_partition_reconciled", True),
    ("canonical_order_reconciled", True),
    ("tie_groups_reconciled", True),
    ("top_k_reconciled", True),
)
RESULT_RECONCILIATIONS: Final = (
    ("requested_patterns_reconciled", True),
    ("rankings_separated_by_pattern", True),
    ("candidate_counts_reconciled", True),
    ("global_state_reconciled", True),
    ("no_cross_pattern_ranking", True),
)
PROVENANCE: Final = (
    ("source", "tactical_matchup_evidence_in_memory"),
    ("scope", "global_only"),
    ("selection", "fixed_descriptive_policy"),
)
BASE_LIMITATIONS: Final = (
    "historical_descriptive_association_not_causal",
    "descriptive_uncertainty_envelope_not_joint_confidence_interval",
    "no_cross_pattern_global_ranking",
    "no_population_baseline_imputed",
)
REDUNDANT_LIMITATION: Final = (
    "redundant_representation_audit_or_interaction_research_only"
)

POLICY_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-scoring-policy\x00"
CANDIDATE_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-scored-candidate\x00"
RANKING_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-pattern-ranking\x00"
EXPLANATION_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-candidate-explanation\x00"
RESULT_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-prioritization-result\x00"

_SHA256 = re.compile(r"^[0-9A-F]{64}$")
_SAFE_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_KEYS: Final = frozenset(
    {
        "match_id",
        "point_number",
        "sequence",
        "sequence_text",
        "raw_sequence",
        "token",
        "tokens",
        "span",
        "spans",
        "point_winner",
        "individual_outcome",
        "match_ids",
        "point_numbers",
        "path",
        "timestamp",
        "test",
        "evaluation",
    }
)
_FORBIDDEN_TEXT_FRAGMENTS: Final = (
    "file://",
    "../",
    "..\\",
)
_CAUSAL_CLAIMS: Final = (
    "causara",
    "causará",
    "garantiza",
    "debilidad demostrada",
    "mejor opcion",
    "mejor opción",
)


class TacticalPrioritizationContractError(ValueError):
    """Fallo cerrado del contrato sintetico de priorizacion."""


class TacticalCandidateState(str, Enum):
    SCORED = "scored"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"
    NOT_AVAILABLE = "not_available"


class TacticalRankingState(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    NOT_AVAILABLE = "not_available"


class TacticalPrioritizationState(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class TacticalScoringPolicy:
    contract_version: str
    name: str
    version: str
    executor_weight: float
    opponent_allowed_weight: float
    formula: str
    allowed_scope: str
    uncertainty_method: str
    abstention_rules: tuple[str, ...]
    comparability_rule: str
    interpretation: str
    reconciliations: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        validate_tactical_scoring_policy(self)


@dataclass(frozen=True)
class TacticalEvidenceSummary:
    perspective: str
    evidence_state: str
    scope: str
    labeled_attempts: int
    successes: int
    failures: int
    success_rate: float | None
    wilson_lower: float | None
    wilson_upper: float | None
    distinct_matches: int


@dataclass(frozen=True)
class TacticalScoredCandidate:
    contract_version: str
    pattern_id: str
    feature_name: str
    category: str
    encoder_policy: TacticalEncodingPolicy
    schema_fingerprint: str
    query_fingerprint: str
    tactical_opportunity: str
    actor: str
    state: TacticalCandidateState
    reason_codes: tuple[str, ...]
    executor_evidence: TacticalEvidenceSummary
    opponent_allowed_evidence: TacticalEvidenceSummary
    combined_rate: float | None
    combined_lower: float | None
    combined_upper: float | None
    combined_width: float | None
    evidence_floor: int | None
    match_floor: int | None
    disagreement: float | None
    uncertainty_label: str
    score_formula: str
    comparable: bool
    rank_eligible: bool
    rank_position: int | None
    tie_group: int | None
    source_category_fingerprint: str
    provenance: tuple[tuple[str, str], ...]
    reconciliations: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        validate_tactical_scored_candidate(self)


@dataclass(frozen=True)
class TacticalPatternRanking:
    contract_version: str
    pattern_id: str
    tactical_opportunity: str
    scoring_policy_fingerprint: str
    encoder_policy: TacticalEncodingPolicy
    candidates: tuple[TacticalScoredCandidate, ...]
    top_candidates: tuple[TacticalScoredCandidate, ...]
    scored_candidate_count: int
    abstained_candidate_count: int
    requested_top_k: int
    effective_top_k: int
    tie_group_count: int
    boundary_tie_expanded: bool
    state: TacticalRankingState
    reason_codes: tuple[str, ...]
    order_rule: str
    redundant_representation: bool
    contractual_purpose: str
    provenance: tuple[tuple[str, str], ...]
    reconciliations: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        validate_tactical_pattern_ranking(self)


@dataclass(frozen=True)
class TacticalCandidateExplanation:
    contract_version: str
    candidate_fingerprint: str
    pattern_id: str
    category: str
    tactical_opportunity: str
    actor: str
    state: TacticalCandidateState
    reason_codes: tuple[str, ...]
    executor_rate: float | None
    opponent_allowed_rate: float | None
    formula: str
    combined_rate: float | None
    combined_lower: float | None
    combined_upper: float | None
    evidence_floor: int | None
    scope: str
    uncertainty_label: str
    interpretation: str
    canonical_text: str

    def __post_init__(self) -> None:
        validate_tactical_candidate_explanation(self)


@dataclass(frozen=True)
class TacticalMatchupQuerySummary:
    player: str
    opponent: str
    as_of_date: str
    serve_numbers: tuple[int, ...]
    requested_patterns: tuple[str, ...]
    encoder_policy: TacticalEncodingPolicy
    scope_strategy: str
    minimum_labeled_attempts: int
    minimum_matches: int
    window_days: int | None
    query_fingerprint: str


@dataclass(frozen=True)
class TacticalPrioritizationResult:
    contract_version: str
    policy: TacticalScoringPolicy
    matchup_query: TacticalMatchupQuerySummary
    schema_fingerprint: str
    source_evidence_fingerprint: str
    state: TacticalPrioritizationState
    reason_codes: tuple[str, ...]
    rankings: tuple[TacticalPatternRanking, ...]
    requested_pattern_count: int
    available_pattern_count: int
    partial_or_unavailable_pattern_count: int
    total_candidate_count: int
    scored_candidate_count: int
    abstained_candidate_count: int
    redundant_representation: bool
    limitations: tuple[str, ...]
    reconciliations: tuple[tuple[str, bool], ...]
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        validate_tactical_prioritization_result(self)


def default_tactical_scoring_policy() -> TacticalScoringPolicy:
    return TacticalScoringPolicy(
        contract_version=PRIORITIZATION_CONTRACT_VERSION,
        name=SCORING_POLICY_NAME,
        version=SCORING_POLICY_VERSION,
        executor_weight=0.5,
        opponent_allowed_weight=0.5,
        formula=SCORE_FORMULA,
        allowed_scope="global",
        uncertainty_method=UNCERTAINTY_METHOD,
        abstention_rules=(
            "both_perspectives_must_be_available",
            "rates_and_wilson_bounds_must_be_present",
            "no_imputation_or_partial_score",
        ),
        comparability_rule=(
            "same_pattern_feature_category_policy_schema_query_cut_and_scope"
        ),
        interpretation="descriptive_historical_association_not_causal",
        reconciliations=POLICY_RECONCILIATIONS,
    )


def _strict_text(value: object, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _SAFE_TEXT.fullmatch(value) is None
    ):
        raise TacticalPrioritizationContractError(
            f"{field} debe ser str exacto, no vacio y sin espacios externos."
        )
    return value


def _strict_count(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise TacticalPrioritizationContractError(
            f"{field} debe ser int real {'positivo' if positive else 'no negativo'}."
        )
    return value


def _finite_unit(value: object, field: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise TacticalPrioritizationContractError(
            f"{field} debe ser float finito en [0, 1]."
        )
    return value


def _optional_unit(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _finite_unit(value, field)


def _sha256_text(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TacticalPrioritizationContractError(
            f"{field} debe ser SHA-256 mayusculo."
        )
    return value


def _pairs(
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
        raise TacticalPrioritizationContractError(
            f"{field} debe ser profundamente inmutable."
        )
    if value != expected:
        raise TacticalPrioritizationContractError(f"{field} no reconcilia.")


def _reason_codes(value: object, expected: tuple[str, ...]) -> None:
    if (
        type(value) is not tuple
        or any(type(item) is not str for item in value)
        or value != expected
    ):
        raise TacticalPrioritizationContractError("reason_codes no reconcilia.")


def _pattern_contract(pattern_id: str) -> tuple[str, str]:
    for pattern, opportunity, actor in PATTERN_CONTRACT:
        if pattern == pattern_id:
            return opportunity, actor
    raise TacticalPrioritizationContractError("pattern_id fuera del catalogo cerrado.")


def _allowed_patterns(policy: TacticalEncodingPolicy) -> tuple[str, ...]:
    for candidate, patterns in POLICY_PATTERN_CONTRACT:
        if policy is candidate:
            return patterns
    raise TacticalPrioritizationContractError("encoder policy fuera del dominio.")


def validate_tactical_scoring_policy(policy: TacticalScoringPolicy) -> None:
    if type(policy) is not TacticalScoringPolicy:
        raise TypeError("policy debe ser TacticalScoringPolicy exacta.")
    if policy.contract_version != PRIORITIZATION_CONTRACT_VERSION:
        raise TacticalPrioritizationContractError("contract_version invalida.")
    if policy.name != SCORING_POLICY_NAME or policy.version != SCORING_POLICY_VERSION:
        raise TacticalPrioritizationContractError("Identidad de policy invalida.")
    if type(policy.executor_weight) is not float or policy.executor_weight != 0.5:
        raise TacticalPrioritizationContractError("executor_weight debe ser 0.5 exacto.")
    if (
        type(policy.opponent_allowed_weight) is not float
        or policy.opponent_allowed_weight != 0.5
    ):
        raise TacticalPrioritizationContractError(
            "opponent_allowed_weight debe ser 0.5 exacto."
        )
    if policy.formula != SCORE_FORMULA:
        raise TacticalPrioritizationContractError("formula fuera del contrato.")
    if policy.allowed_scope != "global" or policy.uncertainty_method != UNCERTAINTY_METHOD:
        raise TacticalPrioritizationContractError("Scope o incertidumbre invalidos.")
    expected_abstention = (
        "both_perspectives_must_be_available",
        "rates_and_wilson_bounds_must_be_present",
        "no_imputation_or_partial_score",
    )
    if policy.abstention_rules != expected_abstention:
        raise TacticalPrioritizationContractError("Reglas de abstencion invalidas.")
    if policy.comparability_rule != (
        "same_pattern_feature_category_policy_schema_query_cut_and_scope"
    ):
        raise TacticalPrioritizationContractError("Comparabilidad invalida.")
    if policy.interpretation != "descriptive_historical_association_not_causal":
        raise TacticalPrioritizationContractError("Interpretacion invalida.")
    _pairs(
        policy.reconciliations,
        POLICY_RECONCILIATIONS,
        value_type=bool,
        field="reconciliations",
    )


def _summary(evidence: TacticalCategoryEvidence) -> TacticalEvidenceSummary:
    return TacticalEvidenceSummary(
        perspective=evidence.perspective.value,
        evidence_state=evidence.evidence_state.value,
        scope=evidence.scope_used.value,
        labeled_attempts=evidence.labeled_activations,
        successes=evidence.tactical_successes,
        failures=evidence.tactical_failures,
        success_rate=evidence.tactical_success_rate,
        wilson_lower=evidence.wilson_lower,
        wilson_upper=evidence.wilson_upper,
        distinct_matches=evidence.distinct_match_count,
    )


def _validate_summary(
    summary: TacticalEvidenceSummary,
    perspective: str,
    feature_name: str,
) -> None:
    if type(summary) is not TacticalEvidenceSummary:
        raise TypeError("Evidence summary debe usar el tipo contractual exacto.")
    if summary.perspective != perspective:
        raise TacticalPrioritizationContractError("Perspectiva resumida invalida.")
    if summary.evidence_state not in tuple(item.value for item in TacticalCategoryEvidenceState):
        raise TacticalPrioritizationContractError("Estado upstream invalido.")
    if summary.scope != TacticalEvidenceScope.GLOBAL.value:
        raise TacticalPrioritizationContractError("Scope resumido debe ser global.")
    for field in ("labeled_attempts", "successes", "failures", "distinct_matches"):
        _strict_count(getattr(summary, field), field)
    if summary.successes + summary.failures != summary.labeled_attempts:
        raise TacticalPrioritizationContractError("Conteos del resumen no reconcilian.")
    values = (summary.success_rate, summary.wilson_lower, summary.wilson_upper)
    for field, value in zip(("success_rate", "wilson_lower", "wilson_upper"), values):
        _optional_unit(value, field)
    if summary.labeled_attempts == 0:
        if any(value is not None for value in values):
            raise TacticalPrioritizationContractError("Sin trials no puede haber tasa o Wilson.")
    else:
        if any(value is None for value in values):
            raise TacticalPrioritizationContractError("Con trials deben existir tasa y Wilson.")
        expected_rate = summary.successes / summary.labeled_attempts
        if summary.success_rate != expected_rate:
            raise TacticalPrioritizationContractError("La tasa resumida no reconcilia.")
        if not summary.wilson_lower <= summary.success_rate <= summary.wilson_upper:
            raise TacticalPrioritizationContractError("Wilson resumido no contiene la tasa.")
    rate, lower, upper = tactical_wilson_interval(
        summary.successes,
        summary.labeled_attempts,
        feature_name=feature_name,
    )
    if (summary.success_rate, summary.wilson_lower, summary.wilson_upper) != (
        rate,
        lower,
        upper,
    ):
        raise TacticalPrioritizationContractError(
            "Tasa o Wilson resumido no reconcilia."
        )
    if summary.distinct_matches > summary.labeled_attempts:
        raise TacticalPrioritizationContractError("Partidos superan trials etiquetados.")


def _candidate_state(
    source: TacticalMatchupCategoryEvidence,
) -> tuple[TacticalCandidateState, tuple[str, ...]]:
    mapping = {
        TacticalMatchupCategoryState.COMPARABLE: (
            TacticalCandidateState.SCORED,
            ("both_perspectives_available", "scored_equal_weight_evidence"),
        ),
        TacticalMatchupCategoryState.EXECUTOR_INSUFFICIENT: (
            TacticalCandidateState.INSUFFICIENT_EVIDENCE,
            ("executor_evidence_insufficient",),
        ),
        TacticalMatchupCategoryState.OPPONENT_INSUFFICIENT: (
            TacticalCandidateState.INSUFFICIENT_EVIDENCE,
            ("opponent_allowed_evidence_insufficient",),
        ),
        TacticalMatchupCategoryState.BOTH_INSUFFICIENT: (
            TacticalCandidateState.INSUFFICIENT_EVIDENCE,
            ("both_perspectives_insufficient",),
        ),
        TacticalMatchupCategoryState.NOT_APPLICABLE: (
            TacticalCandidateState.NOT_APPLICABLE,
            ("pattern_not_applicable_to_requested_serves",),
        ),
        TacticalMatchupCategoryState.NOT_AVAILABLE: (
            TacticalCandidateState.NOT_AVAILABLE,
            ("both_perspectives_not_available",),
        ),
    }
    return mapping[source.matchup_state]


def _candidate_state_from_summaries(
    executor: TacticalEvidenceSummary,
    opponent: TacticalEvidenceSummary,
) -> tuple[TacticalCandidateState, tuple[str, ...]]:
    available = TacticalCategoryEvidenceState.AVAILABLE.value
    not_applicable = TacticalCategoryEvidenceState.NOT_APPLICABLE.value
    not_available = TacticalCategoryEvidenceState.NOT_AVAILABLE.value
    executor_available = executor.evidence_state == available
    opponent_available = opponent.evidence_state == available
    if executor_available and opponent_available:
        return TacticalCandidateState.SCORED, (
            "both_perspectives_available",
            "scored_equal_weight_evidence",
        )
    if executor.evidence_state == not_applicable and opponent.evidence_state == not_applicable:
        return TacticalCandidateState.NOT_APPLICABLE, (
            "pattern_not_applicable_to_requested_serves",
        )
    if executor.evidence_state == not_available and opponent.evidence_state == not_available:
        return TacticalCandidateState.NOT_AVAILABLE, (
            "both_perspectives_not_available",
        )
    if executor_available:
        return TacticalCandidateState.INSUFFICIENT_EVIDENCE, (
            "opponent_allowed_evidence_insufficient",
        )
    if opponent_available:
        return TacticalCandidateState.INSUFFICIENT_EVIDENCE, (
            "executor_evidence_insufficient",
        )
    return TacticalCandidateState.INSUFFICIENT_EVIDENCE, (
        "both_perspectives_insufficient",
    )


def score_tactical_candidate(
    source: TacticalMatchupCategoryEvidence,
    policy: TacticalScoringPolicy,
) -> TacticalScoredCandidate:
    """Puntua una categoria solo cuando ambas perspectivas son comparables."""
    validate_tactical_matchup_category_evidence(source)
    validate_tactical_scoring_policy(policy)
    opportunity, actor = _pattern_contract(source.pattern_id)
    state, reasons = _candidate_state(source)
    executor = _summary(source.executor_evidence)
    opponent = _summary(source.opponent_allowed_evidence)
    scored = state is TacticalCandidateState.SCORED
    if scored:
        values = (
            executor.success_rate,
            opponent.success_rate,
            executor.wilson_lower,
            executor.wilson_upper,
            opponent.wilson_lower,
            opponent.wilson_upper,
        )
        if any(value is None for value in values):
            raise TacticalPrioritizationContractError(
                "Una categoria comparable carece de tasa o Wilson."
            )
        executor_rate, opponent_rate, executor_lower, executor_upper, opponent_lower, opponent_upper = values
        combined_rate = 0.5 * executor_rate + 0.5 * opponent_rate
        combined_lower = 0.5 * executor_lower + 0.5 * opponent_lower
        combined_upper = 0.5 * executor_upper + 0.5 * opponent_upper
        combined_width = combined_upper - combined_lower
        evidence_floor = min(executor.labeled_attempts, opponent.labeled_attempts)
        match_floor = min(executor.distinct_matches, opponent.distinct_matches)
        disagreement = abs(executor_rate - opponent_rate)
    else:
        combined_rate = combined_lower = combined_upper = combined_width = None
        evidence_floor = match_floor = disagreement = None
    return TacticalScoredCandidate(
        contract_version=PRIORITIZATION_CONTRACT_VERSION,
        pattern_id=source.pattern_id,
        feature_name=source.feature_name,
        category=source.category,
        encoder_policy=source.policy,
        schema_fingerprint=source.schema_fingerprint,
        query_fingerprint=source.query_fingerprint,
        tactical_opportunity=opportunity,
        actor=actor,
        state=state,
        reason_codes=reasons,
        executor_evidence=executor,
        opponent_allowed_evidence=opponent,
        combined_rate=combined_rate,
        combined_lower=combined_lower,
        combined_upper=combined_upper,
        combined_width=combined_width,
        evidence_floor=evidence_floor,
        match_floor=match_floor,
        disagreement=disagreement,
        uncertainty_label=UNCERTAINTY_METHOD,
        score_formula=policy.formula,
        comparable=scored,
        rank_eligible=scored,
        rank_position=None,
        tie_group=None,
        source_category_fingerprint=tactical_matchup_category_evidence_fingerprint(source),
        provenance=PROVENANCE,
        reconciliations=CANDIDATE_RECONCILIATIONS,
    )


def validate_tactical_scored_candidate(candidate: TacticalScoredCandidate) -> None:
    if type(candidate) is not TacticalScoredCandidate:
        raise TypeError("candidate debe ser TacticalScoredCandidate exacto.")
    if candidate.contract_version != PRIORITIZATION_CONTRACT_VERSION:
        raise TacticalPrioritizationContractError("Version de candidate invalida.")
    opportunity, actor = _pattern_contract(candidate.pattern_id)
    if candidate.tactical_opportunity != opportunity or candidate.actor != actor:
        raise TacticalPrioritizationContractError("Oportunidad o actor no reconcilia.")
    catalog = dict(PATTERN_FEATURES)[candidate.pattern_id]
    if candidate.feature_name not in catalog:
        raise TacticalPrioritizationContractError(
            "feature_name fuera del catalogo cerrado."
        )
    prefix = {
        "P02": "p02.first_serve_direction.",
        "P04": "p04.return_direction.",
        "P05": "p05.return_depth.",
        "P06": "p06.return_shot_type.",
        "P09": "p09.return_profile.",
    }[candidate.pattern_id]
    if (
        type(candidate.feature_name) is not str
        or not candidate.feature_name.startswith(prefix)
        or candidate.category != candidate.feature_name[len(prefix) :]
        or not candidate.category
        or candidate.category in {"0", "q", "unknown", "censored"}
    ):
        raise TacticalPrioritizationContractError("Identidad de categoria invalida.")
    if type(candidate.encoder_policy) is not TacticalEncodingPolicy:
        raise TypeError("encoder_policy fuera del dominio.")
    if candidate.pattern_id not in _allowed_patterns(candidate.encoder_policy):
        raise TacticalPrioritizationContractError(
            "Candidate incompatible con encoder policy."
        )
    _sha256_text(candidate.schema_fingerprint, "schema_fingerprint")
    _sha256_text(candidate.query_fingerprint, "query_fingerprint")
    if type(candidate.state) is not TacticalCandidateState:
        raise TypeError("state fuera del dominio.")
    _validate_summary(
        candidate.executor_evidence,
        TacticalEvidencePerspective.EXECUTOR.value,
        candidate.feature_name,
    )
    _validate_summary(
        candidate.opponent_allowed_evidence,
        TacticalEvidencePerspective.OPPONENT_ALLOWED.value,
        candidate.feature_name,
    )
    if candidate.uncertainty_label != UNCERTAINTY_METHOD or candidate.score_formula != SCORE_FORMULA:
        raise TacticalPrioritizationContractError("Formula o etiqueta de incertidumbre invalida.")
    _sha256_text(candidate.source_category_fingerprint, "source_category_fingerprint")
    expected_state, expected_reasons = _candidate_state_from_summaries(
        candidate.executor_evidence,
        candidate.opponent_allowed_evidence,
    )
    if candidate.state is not expected_state:
        raise TacticalPrioritizationContractError(
            "State no reconcilia con las dos evidencias upstream."
        )
    scored = candidate.state is TacticalCandidateState.SCORED
    _reason_codes(candidate.reason_codes, expected_reasons)
    numeric = (
        candidate.combined_rate,
        candidate.combined_lower,
        candidate.combined_upper,
        candidate.combined_width,
        candidate.evidence_floor,
        candidate.match_floor,
        candidate.disagreement,
    )
    if scored:
        if any(value is None for value in numeric):
            raise TacticalPrioritizationContractError("Candidate scored incompleto.")
        er = candidate.executor_evidence.success_rate
        op = candidate.opponent_allowed_evidence.success_rate
        el = candidate.executor_evidence.wilson_lower
        eu = candidate.executor_evidence.wilson_upper
        ol = candidate.opponent_allowed_evidence.wilson_lower
        ou = candidate.opponent_allowed_evidence.wilson_upper
        if any(value is None for value in (er, op, el, eu, ol, ou)):
            raise TacticalPrioritizationContractError("Fuente scored incompleta.")
        expected = (
            0.5 * er + 0.5 * op,
            0.5 * el + 0.5 * ol,
            0.5 * eu + 0.5 * ou,
            (0.5 * eu + 0.5 * ou) - (0.5 * el + 0.5 * ol),
            min(
                candidate.executor_evidence.labeled_attempts,
                candidate.opponent_allowed_evidence.labeled_attempts,
            ),
            min(
                candidate.executor_evidence.distinct_matches,
                candidate.opponent_allowed_evidence.distinct_matches,
            ),
            abs(er - op),
        )
        if numeric != expected:
            raise TacticalPrioritizationContractError("Metricas de score no reconcilian.")
        for field in ("combined_rate", "combined_lower", "combined_upper", "combined_width", "disagreement"):
            _finite_unit(getattr(candidate, field), field)
        if not candidate.combined_lower <= candidate.combined_rate <= candidate.combined_upper:
            raise TacticalPrioritizationContractError("Envolvente no contiene el score.")
        _strict_count(candidate.evidence_floor, "evidence_floor")
        _strict_count(candidate.match_floor, "match_floor")
        if type(candidate.comparable) is not bool or not candidate.comparable:
            raise TacticalPrioritizationContractError("Candidate scored debe ser comparable.")
        if type(candidate.rank_eligible) is not bool or not candidate.rank_eligible:
            raise TacticalPrioritizationContractError("Candidate scored debe ser rank eligible.")
    else:
        if any(value is not None for value in numeric):
            raise TacticalPrioritizationContractError("Abstencion no puede conservar score.")
        if type(candidate.comparable) is not bool or candidate.comparable:
            raise TacticalPrioritizationContractError("Abstencion no es comparable.")
        if type(candidate.rank_eligible) is not bool or candidate.rank_eligible:
            raise TacticalPrioritizationContractError("Abstencion no es rank eligible.")
    if (candidate.rank_position is None) != (candidate.tie_group is None):
        raise TacticalPrioritizationContractError("Rank y tie_group deben coexistir.")
    if candidate.rank_position is not None:
        if not scored:
            raise TacticalPrioritizationContractError("Abstencion no puede tener ranking.")
        _strict_count(candidate.rank_position, "rank_position", positive=True)
        _strict_count(candidate.tie_group, "tie_group", positive=True)
    _pairs(candidate.provenance, PROVENANCE, value_type=str, field="provenance")
    _pairs(
        candidate.reconciliations,
        CANDIDATE_RECONCILIATIONS,
        value_type=bool,
        field="reconciliations",
    )


def _rank_key(candidate: TacticalScoredCandidate) -> tuple[object, ...]:
    if not candidate.rank_eligible:
        raise TacticalPrioritizationContractError("Solo se ordenan candidates elegibles.")
    return (
        -candidate.combined_rate,
        -candidate.combined_lower,
        candidate.combined_width,
        -candidate.evidence_floor,
        candidate.feature_name,
    )


def _tie_key(candidate: TacticalScoredCandidate) -> tuple[object, ...]:
    return (
        candidate.combined_rate,
        candidate.combined_lower,
        candidate.combined_width,
        candidate.evidence_floor,
    )


def _ranking_state(scored: int, abstained: int) -> tuple[TacticalRankingState, tuple[str, ...]]:
    if scored and abstained:
        return TacticalRankingState.PARTIALLY_AVAILABLE, ("some_categories_scored",)
    if scored:
        return TacticalRankingState.AVAILABLE, ("all_categories_scored",)
    return TacticalRankingState.NOT_AVAILABLE, ("no_category_scored",)


def rank_tactical_pattern(
    candidates: tuple[TacticalScoredCandidate, ...],
    *,
    pattern_id: str,
    policy: TacticalScoringPolicy,
    encoder_policy: TacticalEncodingPolicy,
    top_k: int,
) -> TacticalPatternRanking:
    """Ordena un unico patron e incluye entero el empate que cruza top-k."""
    validate_tactical_scoring_policy(policy)
    opportunity, _ = _pattern_contract(pattern_id)
    if type(encoder_policy) is not TacticalEncodingPolicy:
        raise TypeError("encoder_policy debe ser TacticalEncodingPolicy exacta.")
    if pattern_id not in _allowed_patterns(encoder_policy):
        raise TacticalPrioritizationContractError("Patron incompatible con encoder policy.")
    if type(candidates) is not tuple or not candidates:
        raise TacticalPrioritizationContractError("candidates debe ser tuple no vacia.")
    if type(top_k) is not int or top_k <= 0:
        raise TacticalPrioritizationContractError("top_k debe ser int real positivo.")
    for candidate in candidates:
        validate_tactical_scored_candidate(candidate)
        if candidate.pattern_id != pattern_id:
            raise TacticalPrioritizationContractError("Ranking mezcla patrones.")
        if candidate.rank_position is not None:
            raise TacticalPrioritizationContractError("Candidate ya rankeado.")
    identities = tuple((item.feature_name, item.category) for item in candidates)
    if len(set(identities)) != len(identities):
        raise TacticalPrioritizationContractError("Candidate duplicado.")
    expected_features = dict(PATTERN_FEATURES)[pattern_id]
    if frozenset(item.feature_name for item in candidates) != frozenset(expected_features):
        raise TacticalPrioritizationContractError(
            "El ranking debe recibir el catalogo completo del patron."
        )
    query_identities = {
        (item.encoder_policy, item.schema_fingerprint, item.query_fingerprint)
        for item in candidates
    }
    expected_identity = (
        encoder_policy,
        candidates[0].schema_fingerprint,
        candidates[0].query_fingerprint,
    )
    if query_identities != {expected_identity}:
        raise TacticalPrioritizationContractError(
            "Candidates no comparten encoder policy, schema y query."
        )
    if top_k > len(candidates):
        raise TacticalPrioritizationContractError("top_k supera el catalogo del patron.")
    eligible = sorted((item for item in candidates if item.rank_eligible), key=_rank_key)
    ranked: list[TacticalScoredCandidate] = []
    tie_group = 0
    previous_tie: tuple[object, ...] | None = None
    for position, item in enumerate(eligible, start=1):
        current_tie = _tie_key(item)
        if current_tie != previous_tie:
            tie_group += 1
            previous_tie = current_tie
        ranked.append(replace(item, rank_position=position, tie_group=tie_group))
    abstained = sorted(
        (item for item in candidates if not item.rank_eligible),
        key=lambda item: item.feature_name,
    )
    if not ranked:
        top: tuple[TacticalScoredCandidate, ...] = ()
    elif len(ranked) <= top_k:
        top = tuple(ranked)
    else:
        boundary_group = ranked[top_k - 1].tie_group
        top = tuple(item for item in ranked if item.tie_group <= boundary_group)
    state, reasons = _ranking_state(len(ranked), len(abstained))
    redundant = encoder_policy is TacticalEncodingPolicy.COMPONENTS_AND_PROFILE
    purpose = (
        "audit_or_interaction_research_only"
        if redundant
        else "descriptive_pattern_prioritization"
    )
    return TacticalPatternRanking(
        contract_version=PRIORITIZATION_CONTRACT_VERSION,
        pattern_id=pattern_id,
        tactical_opportunity=opportunity,
        scoring_policy_fingerprint=tactical_scoring_policy_fingerprint(policy),
        encoder_policy=encoder_policy,
        candidates=tuple(ranked) + tuple(abstained),
        top_candidates=top,
        scored_candidate_count=len(ranked),
        abstained_candidate_count=len(abstained),
        requested_top_k=top_k,
        effective_top_k=len(top),
        tie_group_count=tie_group,
        boundary_tie_expanded=len(top) > min(top_k, len(ranked)),
        state=state,
        reason_codes=reasons,
        order_rule=RANK_ORDER_RULE,
        redundant_representation=redundant,
        contractual_purpose=purpose,
        provenance=PROVENANCE,
        reconciliations=RANKING_RECONCILIATIONS,
    )


def validate_tactical_pattern_ranking(ranking: TacticalPatternRanking) -> None:
    if type(ranking) is not TacticalPatternRanking:
        raise TypeError("ranking debe ser TacticalPatternRanking exacto.")
    if ranking.contract_version != PRIORITIZATION_CONTRACT_VERSION:
        raise TacticalPrioritizationContractError("Version de ranking invalida.")
    opportunity, _ = _pattern_contract(ranking.pattern_id)
    if ranking.tactical_opportunity != opportunity:
        raise TacticalPrioritizationContractError("Oportunidad del ranking invalida.")
    _sha256_text(ranking.scoring_policy_fingerprint, "scoring_policy_fingerprint")
    if type(ranking.encoder_policy) is not TacticalEncodingPolicy:
        raise TypeError("encoder_policy fuera del dominio.")
    if ranking.pattern_id not in _allowed_patterns(ranking.encoder_policy):
        raise TacticalPrioritizationContractError("Ranking incompatible con policy.")
    if type(ranking.candidates) is not tuple or not ranking.candidates:
        raise TacticalPrioritizationContractError("Ranking sin catalogo de candidates.")
    if type(ranking.top_candidates) is not tuple:
        raise TacticalPrioritizationContractError("top_candidates debe ser tuple.")
    for item in ranking.candidates:
        validate_tactical_scored_candidate(item)
        if item.pattern_id != ranking.pattern_id:
            raise TacticalPrioritizationContractError("Ranking mezcla patrones.")
    identities = tuple((item.feature_name, item.category) for item in ranking.candidates)
    if len(set(identities)) != len(identities):
        raise TacticalPrioritizationContractError("Ranking contiene duplicados.")
    if frozenset(item.feature_name for item in ranking.candidates) != frozenset(
        dict(PATTERN_FEATURES)[ranking.pattern_id]
    ):
        raise TacticalPrioritizationContractError(
            "El ranking no conserva el catalogo completo del patron."
        )
    query_identities = {
        (item.encoder_policy, item.schema_fingerprint, item.query_fingerprint)
        for item in ranking.candidates
    }
    if len(query_identities) != 1 or next(iter(query_identities))[0] is not ranking.encoder_policy:
        raise TacticalPrioritizationContractError(
            "Candidates del ranking no comparten policy, schema y query."
        )
    scored = tuple(item for item in ranking.candidates if item.rank_eligible)
    abstained = tuple(item for item in ranking.candidates if not item.rank_eligible)
    expected_ranked = tuple(sorted(scored, key=_rank_key))
    if scored != expected_ranked:
        raise TacticalPrioritizationContractError("Orden del ranking invalido.")
    expected_tie = 0
    previous: tuple[object, ...] | None = None
    for position, item in enumerate(scored, start=1):
        current = _tie_key(item)
        if current != previous:
            expected_tie += 1
            previous = current
        if item.rank_position != position or item.tie_group != expected_tie:
            raise TacticalPrioritizationContractError("Posicion o empate invalido.")
    if abstained != tuple(sorted(abstained, key=lambda item: item.feature_name)):
        raise TacticalPrioritizationContractError("Orden de abstenciones invalido.")
    for field in (
        "scored_candidate_count",
        "abstained_candidate_count",
        "requested_top_k",
        "effective_top_k",
        "tie_group_count",
    ):
        _strict_count(getattr(ranking, field), field, positive=field == "requested_top_k")
    if ranking.requested_top_k > len(ranking.candidates):
        raise TacticalPrioritizationContractError("top_k supera el catalogo.")
    if (
        ranking.scored_candidate_count != len(scored)
        or ranking.abstained_candidate_count != len(abstained)
        or ranking.tie_group_count != expected_tie
    ):
        raise TacticalPrioritizationContractError("Conteos del ranking no reconcilian.")
    if not scored:
        expected_top: tuple[TacticalScoredCandidate, ...] = ()
    elif len(scored) <= ranking.requested_top_k:
        expected_top = scored
    else:
        boundary = scored[ranking.requested_top_k - 1].tie_group
        expected_top = tuple(item for item in scored if item.tie_group <= boundary)
    if ranking.top_candidates != expected_top or ranking.effective_top_k != len(expected_top):
        raise TacticalPrioritizationContractError("Top-k no reconcilia.")
    expected_expanded = len(expected_top) > min(ranking.requested_top_k, len(scored))
    if type(ranking.boundary_tie_expanded) is not bool or ranking.boundary_tie_expanded != expected_expanded:
        raise TacticalPrioritizationContractError("Expansion de empate invalida.")
    state, reasons = _ranking_state(len(scored), len(abstained))
    if type(ranking.state) is not TacticalRankingState or ranking.state is not state:
        raise TacticalPrioritizationContractError("Estado del ranking invalido.")
    _reason_codes(ranking.reason_codes, reasons)
    if ranking.order_rule != RANK_ORDER_RULE:
        raise TacticalPrioritizationContractError("Regla de orden invalida.")
    redundant = ranking.encoder_policy is TacticalEncodingPolicy.COMPONENTS_AND_PROFILE
    purpose = "audit_or_interaction_research_only" if redundant else "descriptive_pattern_prioritization"
    if (
        type(ranking.redundant_representation) is not bool
        or ranking.redundant_representation != redundant
        or ranking.contractual_purpose != purpose
    ):
        raise TacticalPrioritizationContractError("Contrato de redundancia invalido.")
    _pairs(ranking.provenance, PROVENANCE, value_type=str, field="provenance")
    _pairs(
        ranking.reconciliations,
        RANKING_RECONCILIATIONS,
        value_type=bool,
        field="reconciliations",
    )


def explain_tactical_candidate(candidate: TacticalScoredCandidate) -> TacticalCandidateExplanation:
    validate_tactical_scored_candidate(candidate)
    if candidate.state is TacticalCandidateState.SCORED:
        text = (
            f"Ranking descriptivo para {candidate.pattern_id}:{candidate.category}; "
            "combina por igual dos tasas historicas comparables y no implica causalidad."
        )
    else:
        text = (
            f"Sin score descriptivo para {candidate.pattern_id}:{candidate.category}; "
            f"estado {candidate.state.value} y sin imputacion."
        )
    return TacticalCandidateExplanation(
        contract_version=PRIORITIZATION_CONTRACT_VERSION,
        candidate_fingerprint=tactical_scored_candidate_fingerprint(candidate),
        pattern_id=candidate.pattern_id,
        category=candidate.category,
        tactical_opportunity=candidate.tactical_opportunity,
        actor=candidate.actor,
        state=candidate.state,
        reason_codes=candidate.reason_codes,
        executor_rate=candidate.executor_evidence.success_rate if candidate.rank_eligible else None,
        opponent_allowed_rate=(
            candidate.opponent_allowed_evidence.success_rate if candidate.rank_eligible else None
        ),
        formula=candidate.score_formula,
        combined_rate=candidate.combined_rate,
        combined_lower=candidate.combined_lower,
        combined_upper=candidate.combined_upper,
        evidence_floor=candidate.evidence_floor,
        scope="global",
        uncertainty_label=candidate.uncertainty_label,
        interpretation="descriptive_historical_association_not_causal",
        canonical_text=text,
    )


def validate_tactical_candidate_explanation(
    explanation: TacticalCandidateExplanation,
) -> None:
    if type(explanation) is not TacticalCandidateExplanation:
        raise TypeError("explanation debe usar el tipo contractual exacto.")
    if explanation.contract_version != PRIORITIZATION_CONTRACT_VERSION:
        raise TacticalPrioritizationContractError("Version de explanation invalida.")
    _sha256_text(explanation.candidate_fingerprint, "candidate_fingerprint")
    opportunity, actor = _pattern_contract(explanation.pattern_id)
    if explanation.tactical_opportunity != opportunity or explanation.actor != actor:
        raise TacticalPrioritizationContractError("Identidad de explanation invalida.")
    if type(explanation.state) is not TacticalCandidateState:
        raise TypeError("Estado de explanation invalido.")
    expected_reason_domains = {
        TacticalCandidateState.SCORED: {
            ("both_perspectives_available", "scored_equal_weight_evidence")
        },
        TacticalCandidateState.INSUFFICIENT_EVIDENCE: {
            ("executor_evidence_insufficient",),
            ("opponent_allowed_evidence_insufficient",),
            ("both_perspectives_insufficient",),
        },
        TacticalCandidateState.NOT_APPLICABLE: {
            ("pattern_not_applicable_to_requested_serves",)
        },
        TacticalCandidateState.NOT_AVAILABLE: {
            ("both_perspectives_not_available",)
        },
    }
    if explanation.reason_codes not in expected_reason_domains[explanation.state]:
        raise TacticalPrioritizationContractError("Reasons de explanation invalidos.")
    if explanation.formula != SCORE_FORMULA or explanation.scope != "global":
        raise TacticalPrioritizationContractError("Formula o scope de explanation invalido.")
    if explanation.uncertainty_label != UNCERTAINTY_METHOD:
        raise TacticalPrioritizationContractError("Incertidumbre de explanation invalida.")
    if explanation.interpretation != "descriptive_historical_association_not_causal":
        raise TacticalPrioritizationContractError("Interpretacion de explanation invalida.")
    scored = explanation.state is TacticalCandidateState.SCORED
    values = (
        explanation.executor_rate,
        explanation.opponent_allowed_rate,
        explanation.combined_rate,
        explanation.combined_lower,
        explanation.combined_upper,
        explanation.evidence_floor,
    )
    if scored:
        if any(value is None for value in values):
            raise TacticalPrioritizationContractError("Explanation scored incompleta.")
        for field in (
            "executor_rate",
            "opponent_allowed_rate",
            "combined_rate",
            "combined_lower",
            "combined_upper",
        ):
            _finite_unit(getattr(explanation, field), field)
        _strict_count(explanation.evidence_floor, "evidence_floor")
        expected_rate = 0.5 * explanation.executor_rate + 0.5 * explanation.opponent_allowed_rate
        if explanation.combined_rate != expected_rate:
            raise TacticalPrioritizationContractError("Score explicado no reconcilia.")
        if not explanation.combined_lower <= explanation.combined_rate <= explanation.combined_upper:
            raise TacticalPrioritizationContractError("Envolvente explicada invalida.")
    elif any(value is not None for value in values):
        raise TacticalPrioritizationContractError("Abstencion explicada contiene score.")
    _strict_text(explanation.category, "category")
    valid_categories = {
        feature.split(".", 3)[-1]
        for feature in dict(PATTERN_FEATURES)[explanation.pattern_id]
    }
    if explanation.category not in valid_categories:
        raise TacticalPrioritizationContractError("Categoria de explanation invalida.")
    text = _strict_text(explanation.canonical_text, "canonical_text")
    if scored:
        expected_text = (
            f"Ranking descriptivo para {explanation.pattern_id}:{explanation.category}; "
            "combina por igual dos tasas historicas comparables y no implica causalidad."
        )
    else:
        expected_text = (
            f"Sin score descriptivo para {explanation.pattern_id}:{explanation.category}; "
            f"estado {explanation.state.value} y sin imputacion."
        )
    if text != expected_text:
        raise TacticalPrioritizationContractError(
            "Texto canonico de explanation no reconcilia."
        )
    lowered = text.lower()
    if any(claim in lowered for claim in _CAUSAL_CLAIMS):
        raise TacticalPrioritizationContractError("La explicacion contiene lenguaje no permitido.")


def _query_summary(evidence: TacticalMatchupEvidence) -> TacticalMatchupQuerySummary:
    query = evidence.query
    return TacticalMatchupQuerySummary(
        player=query.player,
        opponent=query.opponent,
        as_of_date=query.as_of_date.isoformat(),
        serve_numbers=query.serve_numbers,
        requested_patterns=query.requested_patterns,
        encoder_policy=query.policy,
        scope_strategy=query.scope_strategy.value,
        minimum_labeled_attempts=query.minimum_labeled_attempts,
        minimum_matches=query.minimum_matches,
        window_days=query.window_days,
        query_fingerprint=tactical_matchup_query_fingerprint(query),
    )


def _validate_query_summary(summary: TacticalMatchupQuerySummary) -> None:
    if type(summary) is not TacticalMatchupQuerySummary:
        raise TypeError("matchup_query debe usar el tipo contractual exacto.")
    _strict_text(summary.player, "player")
    _strict_text(summary.opponent, "opponent")
    if summary.player == summary.opponent:
        raise TacticalPrioritizationContractError("Player y opponent deben ser distintos.")
    if type(summary.as_of_date) is not str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", summary.as_of_date):
        raise TacticalPrioritizationContractError("as_of_date invalida.")
    try:
        parsed_date = date.fromisoformat(summary.as_of_date)
    except ValueError:
        raise TacticalPrioritizationContractError("as_of_date invalida.") from None
    if parsed_date.isoformat() != summary.as_of_date:
        raise TacticalPrioritizationContractError("as_of_date no es canonica.")
    if (
        type(summary.serve_numbers) is not tuple
        or any(type(item) is not int for item in summary.serve_numbers)
        or summary.serve_numbers not in {(1,), (2,), (1, 2)}
    ):
        raise TacticalPrioritizationContractError("serve_numbers invalidos.")
    if type(summary.encoder_policy) is not TacticalEncodingPolicy:
        raise TypeError("encoder_policy invalida.")
    allowed = _allowed_patterns(summary.encoder_policy)
    if (
        type(summary.requested_patterns) is not tuple
        or not summary.requested_patterns
        or summary.requested_patterns != tuple(p for p in PATTERN_ORDER if p in summary.requested_patterns)
        or any(p not in allowed for p in summary.requested_patterns)
    ):
        raise TacticalPrioritizationContractError("requested_patterns invalidos.")
    if summary.scope_strategy != "global_only":
        raise TacticalPrioritizationContractError("scope_strategy invalido.")
    _strict_count(summary.minimum_labeled_attempts, "minimum_labeled_attempts")
    _strict_count(summary.minimum_matches, "minimum_matches")
    if summary.window_days is not None:
        _strict_count(summary.window_days, "window_days", positive=True)
    _sha256_text(summary.query_fingerprint, "query_fingerprint")


def prioritize_tactical_matchup(
    evidence: TacticalMatchupEvidence,
    *,
    policy: TacticalScoringPolicy | None = None,
    top_k: int = 3,
) -> TacticalPrioritizationResult:
    """Construye rankings independientes desde un unico objeto evidence."""
    validate_tactical_matchup_evidence(evidence)
    selected_policy = default_tactical_scoring_policy() if policy is None else policy
    validate_tactical_scoring_policy(selected_policy)
    if type(top_k) is not int or top_k <= 0:
        raise TacticalPrioritizationContractError("top_k debe ser int real positivo.")
    rankings = []
    for pattern_id in evidence.query.requested_patterns:
        sources = tuple(item for item in evidence.categories if item.pattern_id == pattern_id)
        if not sources:
            raise TacticalPrioritizationContractError("Patron solicitado sin catalogo.")
        if top_k > len(sources):
            raise TacticalPrioritizationContractError("top_k supera categorias del patron.")
        candidates = tuple(score_tactical_candidate(item, selected_policy) for item in sources)
        rankings.append(
            rank_tactical_pattern(
                candidates,
                pattern_id=pattern_id,
                policy=selected_policy,
                encoder_policy=evidence.policy,
                top_k=top_k,
            )
        )
    ranking_tuple = tuple(rankings)
    available = sum(item.scored_candidate_count > 0 for item in ranking_tuple)
    if available == len(ranking_tuple):
        state = TacticalPrioritizationState.AVAILABLE
        reasons = ("all_requested_patterns_have_scored_candidates",)
    elif available:
        state = TacticalPrioritizationState.PARTIALLY_AVAILABLE
        reasons = ("some_requested_patterns_have_scored_candidates",)
    else:
        state = TacticalPrioritizationState.NOT_AVAILABLE
        reasons = ("no_requested_pattern_has_scored_candidates",)
    total = sum(len(item.candidates) for item in ranking_tuple)
    scored = sum(item.scored_candidate_count for item in ranking_tuple)
    limitations = BASE_LIMITATIONS + ((REDUNDANT_LIMITATION,) if evidence.redundant_representation else ())
    return TacticalPrioritizationResult(
        contract_version=PRIORITIZATION_CONTRACT_VERSION,
        policy=selected_policy,
        matchup_query=_query_summary(evidence),
        schema_fingerprint=evidence.schema_fingerprint,
        source_evidence_fingerprint=tactical_matchup_evidence_fingerprint(evidence),
        state=state,
        reason_codes=reasons,
        rankings=ranking_tuple,
        requested_pattern_count=len(ranking_tuple),
        available_pattern_count=available,
        partial_or_unavailable_pattern_count=len(ranking_tuple) - available,
        total_candidate_count=total,
        scored_candidate_count=scored,
        abstained_candidate_count=total - scored,
        redundant_representation=evidence.redundant_representation,
        limitations=limitations,
        reconciliations=RESULT_RECONCILIATIONS,
        provenance=PROVENANCE,
    )


build_tactical_prioritization = prioritize_tactical_matchup


def validate_tactical_prioritization_result(result: TacticalPrioritizationResult) -> None:
    if type(result) is not TacticalPrioritizationResult:
        raise TypeError("result debe ser TacticalPrioritizationResult exacto.")
    if result.contract_version != PRIORITIZATION_CONTRACT_VERSION:
        raise TacticalPrioritizationContractError("Version de result invalida.")
    validate_tactical_scoring_policy(result.policy)
    _validate_query_summary(result.matchup_query)
    _sha256_text(result.schema_fingerprint, "schema_fingerprint")
    _sha256_text(result.source_evidence_fingerprint, "source_evidence_fingerprint")
    reconstructed_query = TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        result.matchup_query.player,
        result.matchup_query.opponent,
        date.fromisoformat(result.matchup_query.as_of_date),
        result.matchup_query.encoder_policy,
        result.schema_fingerprint,
        result.matchup_query.serve_numbers,
        result.matchup_query.window_days,
        result.matchup_query.minimum_labeled_attempts,
        result.matchup_query.minimum_matches,
        result.matchup_query.requested_patterns,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )
    if (
        tactical_matchup_query_fingerprint(reconstructed_query)
        != result.matchup_query.query_fingerprint
    ):
        raise TacticalPrioritizationContractError(
            "Fingerprint de matchup query no reconcilia."
        )
    if type(result.rankings) is not tuple or not result.rankings:
        raise TacticalPrioritizationContractError("Result sin rankings.")
    for ranking in result.rankings:
        validate_tactical_pattern_ranking(ranking)
        if ranking.encoder_policy is not result.matchup_query.encoder_policy:
            raise TacticalPrioritizationContractError("Encoder policy no reconcilia.")
        if ranking.scoring_policy_fingerprint != tactical_scoring_policy_fingerprint(result.policy):
            raise TacticalPrioritizationContractError("Scoring policy no reconcilia.")
        for candidate in ranking.candidates:
            if (
                candidate.schema_fingerprint != result.schema_fingerprint
                or candidate.query_fingerprint != result.matchup_query.query_fingerprint
            ):
                raise TacticalPrioritizationContractError(
                    "Candidate no reconcilia con schema y query globales."
                )
    actual_patterns = tuple(item.pattern_id for item in result.rankings)
    if actual_patterns != result.matchup_query.requested_patterns:
        raise TacticalPrioritizationContractError("Orden o patrones de result invalidos.")
    for field in (
        "requested_pattern_count",
        "available_pattern_count",
        "partial_or_unavailable_pattern_count",
        "total_candidate_count",
        "scored_candidate_count",
        "abstained_candidate_count",
    ):
        _strict_count(getattr(result, field), field)
    available = sum(item.scored_candidate_count > 0 for item in result.rankings)
    total = sum(len(item.candidates) for item in result.rankings)
    scored = sum(item.scored_candidate_count for item in result.rankings)
    expected_counts = (
        len(result.rankings),
        available,
        len(result.rankings) - available,
        total,
        scored,
        total - scored,
    )
    actual_counts = (
        result.requested_pattern_count,
        result.available_pattern_count,
        result.partial_or_unavailable_pattern_count,
        result.total_candidate_count,
        result.scored_candidate_count,
        result.abstained_candidate_count,
    )
    if actual_counts != expected_counts:
        raise TacticalPrioritizationContractError("Conteos globales no reconcilian.")
    if available == len(result.rankings):
        state = TacticalPrioritizationState.AVAILABLE
        reasons = ("all_requested_patterns_have_scored_candidates",)
    elif available:
        state = TacticalPrioritizationState.PARTIALLY_AVAILABLE
        reasons = ("some_requested_patterns_have_scored_candidates",)
    else:
        state = TacticalPrioritizationState.NOT_AVAILABLE
        reasons = ("no_requested_pattern_has_scored_candidates",)
    if type(result.state) is not TacticalPrioritizationState or result.state is not state:
        raise TacticalPrioritizationContractError("Estado global no reconcilia.")
    _reason_codes(result.reason_codes, reasons)
    redundant = result.matchup_query.encoder_policy is TacticalEncodingPolicy.COMPONENTS_AND_PROFILE
    if type(result.redundant_representation) is not bool or result.redundant_representation != redundant:
        raise TacticalPrioritizationContractError("Redundancia global no reconcilia.")
    expected_limitations = BASE_LIMITATIONS + ((REDUNDANT_LIMITATION,) if redundant else ())
    if type(result.limitations) is not tuple or result.limitations != expected_limitations:
        raise TacticalPrioritizationContractError("Limitaciones no reconcilian.")
    _pairs(result.reconciliations, RESULT_RECONCILIATIONS, value_type=bool, field="reconciliations")
    _pairs(result.provenance, PROVENANCE, value_type=str, field="provenance")


def _summary_structure(summary: TacticalEvidenceSummary) -> dict[str, object]:
    return {
        "distinct_matches": summary.distinct_matches,
        "evidence_state": summary.evidence_state,
        "failures": summary.failures,
        "labeled_attempts": summary.labeled_attempts,
        "perspective": summary.perspective,
        "scope": summary.scope,
        "success_rate": summary.success_rate,
        "successes": summary.successes,
        "wilson_lower": summary.wilson_lower,
        "wilson_upper": summary.wilson_upper,
    }


def _policy_structure(policy: TacticalScoringPolicy) -> dict[str, object]:
    validate_tactical_scoring_policy(policy)
    return {
        "abstention_rules": list(policy.abstention_rules),
        "allowed_scope": policy.allowed_scope,
        "comparability_rule": policy.comparability_rule,
        "contract_version": policy.contract_version,
        "executor_weight": policy.executor_weight,
        "formula": policy.formula,
        "interpretation": policy.interpretation,
        "name": policy.name,
        "opponent_allowed_weight": policy.opponent_allowed_weight,
        "reconciliations": dict(policy.reconciliations),
        "uncertainty_method": policy.uncertainty_method,
        "version": policy.version,
    }


def _candidate_structure(candidate: TacticalScoredCandidate) -> dict[str, object]:
    validate_tactical_scored_candidate(candidate)
    return {
        "actor": candidate.actor,
        "category": candidate.category,
        "combined_lower": candidate.combined_lower,
        "combined_rate": candidate.combined_rate,
        "combined_upper": candidate.combined_upper,
        "combined_width": candidate.combined_width,
        "comparable": candidate.comparable,
        "contract_version": candidate.contract_version,
        "disagreement": candidate.disagreement,
        "evidence_floor": candidate.evidence_floor,
        "encoder_policy": candidate.encoder_policy.value,
        "executor_evidence": _summary_structure(candidate.executor_evidence),
        "feature_name": candidate.feature_name,
        "match_floor": candidate.match_floor,
        "opponent_allowed_evidence": _summary_structure(candidate.opponent_allowed_evidence),
        "pattern_id": candidate.pattern_id,
        "provenance": dict(candidate.provenance),
        "rank_eligible": candidate.rank_eligible,
        "rank_position": candidate.rank_position,
        "query_fingerprint": candidate.query_fingerprint,
        "reason_codes": list(candidate.reason_codes),
        "reconciliations": dict(candidate.reconciliations),
        "score_formula": candidate.score_formula,
        "schema_fingerprint": candidate.schema_fingerprint,
        "source_category_fingerprint": candidate.source_category_fingerprint,
        "state": candidate.state.value,
        "tactical_opportunity": candidate.tactical_opportunity,
        "tie_group": candidate.tie_group,
        "uncertainty_label": candidate.uncertainty_label,
    }


def _ranking_structure(ranking: TacticalPatternRanking) -> dict[str, object]:
    validate_tactical_pattern_ranking(ranking)
    return {
        "abstained_candidate_count": ranking.abstained_candidate_count,
        "boundary_tie_expanded": ranking.boundary_tie_expanded,
        "candidates": [_candidate_structure(item) for item in ranking.candidates],
        "contract_version": ranking.contract_version,
        "contractual_purpose": ranking.contractual_purpose,
        "effective_top_k": ranking.effective_top_k,
        "encoder_policy": ranking.encoder_policy.value,
        "order_rule": ranking.order_rule,
        "pattern_id": ranking.pattern_id,
        "provenance": dict(ranking.provenance),
        "reason_codes": list(ranking.reason_codes),
        "reconciliations": dict(ranking.reconciliations),
        "redundant_representation": ranking.redundant_representation,
        "requested_top_k": ranking.requested_top_k,
        "scored_candidate_count": ranking.scored_candidate_count,
        "scoring_policy_fingerprint": ranking.scoring_policy_fingerprint,
        "state": ranking.state.value,
        "tactical_opportunity": ranking.tactical_opportunity,
        "tie_group_count": ranking.tie_group_count,
        "top_candidates": [_candidate_structure(item) for item in ranking.top_candidates],
    }


def _explanation_structure(explanation: TacticalCandidateExplanation) -> dict[str, object]:
    validate_tactical_candidate_explanation(explanation)
    return {
        "actor": explanation.actor,
        "candidate_fingerprint": explanation.candidate_fingerprint,
        "canonical_text": explanation.canonical_text,
        "category": explanation.category,
        "combined_lower": explanation.combined_lower,
        "combined_rate": explanation.combined_rate,
        "combined_upper": explanation.combined_upper,
        "contract_version": explanation.contract_version,
        "evidence_floor": explanation.evidence_floor,
        "executor_rate": explanation.executor_rate,
        "formula": explanation.formula,
        "interpretation": explanation.interpretation,
        "opponent_allowed_rate": explanation.opponent_allowed_rate,
        "pattern_id": explanation.pattern_id,
        "reason_codes": list(explanation.reason_codes),
        "scope": explanation.scope,
        "state": explanation.state.value,
        "tactical_opportunity": explanation.tactical_opportunity,
        "uncertainty_label": explanation.uncertainty_label,
    }


def _query_summary_structure(summary: TacticalMatchupQuerySummary) -> dict[str, object]:
    _validate_query_summary(summary)
    return {
        "as_of_date": summary.as_of_date,
        "encoder_policy": summary.encoder_policy.value,
        "minimum_labeled_attempts": summary.minimum_labeled_attempts,
        "minimum_matches": summary.minimum_matches,
        "opponent": summary.opponent,
        "player": summary.player,
        "query_fingerprint": summary.query_fingerprint,
        "requested_patterns": list(summary.requested_patterns),
        "scope_strategy": summary.scope_strategy,
        "serve_numbers": list(summary.serve_numbers),
        "window_days": summary.window_days,
    }


def _result_structure(result: TacticalPrioritizationResult) -> dict[str, object]:
    validate_tactical_prioritization_result(result)
    return {
        "abstained_candidate_count": result.abstained_candidate_count,
        "available_pattern_count": result.available_pattern_count,
        "contract_version": result.contract_version,
        "limitations": list(result.limitations),
        "matchup_query": _query_summary_structure(result.matchup_query),
        "partial_or_unavailable_pattern_count": result.partial_or_unavailable_pattern_count,
        "policy": _policy_structure(result.policy),
        "provenance": dict(result.provenance),
        "rankings": [_ranking_structure(item) for item in result.rankings],
        "reason_codes": list(result.reason_codes),
        "reconciliations": dict(result.reconciliations),
        "redundant_representation": result.redundant_representation,
        "requested_pattern_count": result.requested_pattern_count,
        "schema_fingerprint": result.schema_fingerprint,
        "scored_candidate_count": result.scored_candidate_count,
        "source_evidence_fingerprint": result.source_evidence_fingerprint,
        "state": result.state.value,
        "total_candidate_count": result.total_candidate_count,
    }


def _validate_public_tree(value: object, *, key: str | None = None) -> None:
    if key is not None:
        lowered = key.lower()
        if lowered in _FORBIDDEN_KEYS or lowered.startswith("test_") or "evaluation" in lowered:
            raise TacticalPrioritizationContractError("Clave publica no autorizada.")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise TacticalPrioritizationContractError("Valor no finito.")
        return
    if type(value) is str:
        lowered = value.lower()
        if (
            any(fragment in lowered for fragment in _FORBIDDEN_TEXT_FRAGMENTS)
            or _WINDOWS_PATH.match(value)
            or value.startswith(("/", "\\\\"))
            or any(ord(character) < 32 for character in value)
        ):
            raise TacticalPrioritizationContractError("Texto operativo no autorizado.")
        return
    if type(value) is list:
        for item in value:
            _validate_public_tree(item)
        return
    if type(value) is dict:
        for child_key, child_value in value.items():
            if type(child_key) is not str:
                raise TacticalPrioritizationContractError("Clave JSON no textual.")
            _validate_public_tree(child_value, key=child_key)
        return
    raise TacticalPrioritizationContractError("Objeto publico no contractual.")


def _canonical_bytes(structure: dict[str, object]) -> bytes:
    _validate_public_tree(structure)
    return json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_tactical_scoring_policy_json(policy: TacticalScoringPolicy) -> bytes:
    return _canonical_bytes(_policy_structure(policy))


def canonical_tactical_scored_candidate_json(candidate: TacticalScoredCandidate) -> bytes:
    return _canonical_bytes(_candidate_structure(candidate))


def canonical_tactical_pattern_ranking_json(ranking: TacticalPatternRanking) -> bytes:
    return _canonical_bytes(_ranking_structure(ranking))


def canonical_tactical_candidate_explanation_json(
    explanation: TacticalCandidateExplanation,
) -> bytes:
    return _canonical_bytes(_explanation_structure(explanation))


def canonical_tactical_prioritization_result_json(
    result: TacticalPrioritizationResult,
) -> bytes:
    return _canonical_bytes(_result_structure(result))


def tactical_scoring_policy_fingerprint(policy: TacticalScoringPolicy) -> str:
    return sha256(POLICY_FINGERPRINT_DOMAIN + canonical_tactical_scoring_policy_json(policy)).hexdigest().upper()


def tactical_scored_candidate_fingerprint(candidate: TacticalScoredCandidate) -> str:
    return sha256(CANDIDATE_FINGERPRINT_DOMAIN + canonical_tactical_scored_candidate_json(candidate)).hexdigest().upper()


def tactical_pattern_ranking_fingerprint(ranking: TacticalPatternRanking) -> str:
    return sha256(RANKING_FINGERPRINT_DOMAIN + canonical_tactical_pattern_ranking_json(ranking)).hexdigest().upper()


def tactical_candidate_explanation_fingerprint(explanation: TacticalCandidateExplanation) -> str:
    return sha256(EXPLANATION_FINGERPRINT_DOMAIN + canonical_tactical_candidate_explanation_json(explanation)).hexdigest().upper()


def tactical_prioritization_result_fingerprint(result: TacticalPrioritizationResult) -> str:
    return sha256(RESULT_FINGERPRINT_DOMAIN + canonical_tactical_prioritization_result_json(result)).hexdigest().upper()


__all__ = (
    "BASE_LIMITATIONS",
    "CANDIDATE_FINGERPRINT_DOMAIN",
    "EXPLANATION_FINGERPRINT_DOMAIN",
    "PATTERN_CONTRACT",
    "PATTERN_ORDER",
    "POLICY_FINGERPRINT_DOMAIN",
    "PRIORITIZATION_CONTRACT_VERSION",
    "RANKING_FINGERPRINT_DOMAIN",
    "RANK_ORDER_RULE",
    "RESULT_FINGERPRINT_DOMAIN",
    "SCORE_FORMULA",
    "SCORING_POLICY_NAME",
    "SCORING_POLICY_VERSION",
    "TacticalCandidateExplanation",
    "TacticalCandidateState",
    "TacticalEvidenceSummary",
    "TacticalMatchupQuerySummary",
    "TacticalPatternRanking",
    "TacticalPrioritizationContractError",
    "TacticalPrioritizationResult",
    "TacticalPrioritizationState",
    "TacticalRankingState",
    "TacticalScoredCandidate",
    "TacticalScoringPolicy",
    "UNCERTAINTY_METHOD",
    "build_tactical_prioritization",
    "canonical_tactical_candidate_explanation_json",
    "canonical_tactical_pattern_ranking_json",
    "canonical_tactical_prioritization_result_json",
    "canonical_tactical_scored_candidate_json",
    "canonical_tactical_scoring_policy_json",
    "default_tactical_scoring_policy",
    "explain_tactical_candidate",
    "prioritize_tactical_matchup",
    "rank_tactical_pattern",
    "score_tactical_candidate",
    "tactical_candidate_explanation_fingerprint",
    "tactical_pattern_ranking_fingerprint",
    "tactical_prioritization_result_fingerprint",
    "tactical_scored_candidate_fingerprint",
    "tactical_scoring_policy_fingerprint",
    "validate_tactical_candidate_explanation",
    "validate_tactical_pattern_ranking",
    "validate_tactical_prioritization_result",
    "validate_tactical_scored_candidate",
    "validate_tactical_scoring_policy",
)
