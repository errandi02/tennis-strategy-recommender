"""Integracion end-to-end completamente sintetica del recomendador tactico P11."""

from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import date, timedelta
from hashlib import sha256
from functools import lru_cache
import json
from math import sqrt
import sys
from types import MappingProxyType

import pytest

import src.recommender.tactical_signal_orchestrator as orchestrator
from src.recommender.tactical_feature_encoder import (
    SCHEMA_FINGERPRINT_DOMAIN,
    VECTOR_FINGERPRINT_DOMAIN,
    TacticalEncodingPolicy,
    TacticalFeatureContractError,
    TacticalFeatureSchema,
    TacticalFeatureVector,
    build_tactical_feature_schema,
    encode_tactical_attempt,
    feature_schema_fingerprint,
    feature_vector_fingerprint,
    validate_tactical_feature_schema,
    validate_tactical_feature_vector,
)
from src.recommender.tactical_history_profiles import (
    HISTORY_CONTRACT_VERSION,
    OBSERVATION_FINGERPRINT_DOMAIN,
    OBSERVATION_PROVENANCE,
    TacticalHistoricalObservation,
    TacticalLabelAvailability,
    TacticalOutcomeLabel,
    TacticalPlayerRole,
    historical_observation_fingerprint,
    validate_tactical_historical_observation,
)
from src.recommender.tactical_matchup_evidence import (
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    MATCHUP_EVIDENCE_FINGERPRINT_DOMAIN,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupEvidence,
    TacticalMatchupContractError,
    TacticalMatchupQuery,
    build_tactical_matchup_evidence,
    tactical_matchup_evidence_fingerprint,
    validate_tactical_matchup_evidence,
)
from src.recommender.tactical_prioritization import (
    RESULT_FINGERPRINT_DOMAIN,
    TacticalPrioritizationResult,
    TacticalPrioritizationContractError,
    prioritize_tactical_matchup,
    tactical_prioritization_result_fingerprint,
    validate_tactical_prioritization_result,
)
from src.recommender.tactical_recommendation_contract import (
    PUBLIC_FINGERPRINT_DOMAIN,
    PublicEvidenceComponent,
    TacticalRecommendationContractError,
    TacticalRecommendationResponse,
    build_public_tactical_recommendation,
    canonical_tactical_recommendation_json,
    tactical_recommendation_fingerprint,
    validate_public_tactical_recommendation,
)
from src.recommender.tactical_signal_contract import (
    TacticalSignalContractError,
    TacticalSignalBundle,
    tactical_signal_bundle_fingerprint,
    validate_tactical_signal_bundle,
)
from src.recommender.tactical_signal_orchestrator import (
    ATTEMPT_FINGERPRINT_DOMAIN,
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalExtraction,
    AttemptSignalRequest,
    attempt_extraction_fingerprint,
    extract_tactical_signals_for_attempt,
    validate_attempt_signal_extraction,
)


SCORING_PATTERNS = ("P02", "P04", "P05", "P06")
SHOT_TYPES = tuple("fbrsvzopuylmhijkt")
SERVE_DIRECTIONS = (("4", "wide"), ("5", "body"), ("6", "T"))
AS_OF_DATE = date(2021, 1, 1)
SIGNAL_BUNDLE_FINGERPRINT_DOMAIN = b"tennis-tactical-signal-bundle\x00"


@dataclass(frozen=True)
class SyntheticAttempt:
    match_id: str
    point_number: int
    effective_date: date
    server: str
    returner: str
    sequence: str
    serve_number: int
    previous_attempt_was_fault: bool
    server_won_point: bool


@dataclass(frozen=True)
class MaterializedHistory:
    requests: tuple[AttemptSignalRequest, ...]
    extractions: tuple[AttemptSignalExtraction, ...]
    vectors: tuple[TacticalFeatureVector, ...]
    observations: tuple[TacticalHistoricalObservation, ...]


@dataclass(frozen=True)
class EndToEndResult:
    schema: TacticalFeatureSchema
    history: MaterializedHistory
    evidence: TacticalMatchupEvidence
    prioritization: TacticalPrioritizationResult
    response: TacticalRecommendationResponse


def _attempt(
    *,
    match_id: str,
    point_number: int,
    effective_date: date,
    server: str,
    returner: str,
    sequence: str,
    serve_number: int = 1,
    previous_fault: bool = False,
    success: bool = True,
) -> SyntheticAttempt:
    return SyntheticAttempt(
        match_id,
        point_number,
        effective_date,
        server,
        returner,
        sequence,
        serve_number,
        previous_fault,
        success,
    )


def _materialize(
    attempts: tuple[SyntheticAttempt, ...],
    schema: TacticalFeatureSchema,
) -> MaterializedHistory:
    requests: list[AttemptSignalRequest] = []
    extractions: list[AttemptSignalExtraction] = []
    vectors: list[TacticalFeatureVector] = []
    observations: list[TacticalHistoricalObservation] = []
    for attempt in attempts:
        request = AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION,
            attempt.sequence,
            attempt.serve_number,
            attempt.previous_attempt_was_fault,
        )
        extraction = extract_tactical_signals_for_attempt(request)
        vector = encode_tactical_attempt(extraction, schema)
        observation = TacticalHistoricalObservation(
            HISTORY_CONTRACT_VERSION,
            attempt.match_id,
            attempt.point_number,
            attempt.serve_number,
            attempt.effective_date,
            attempt.server,
            attempt.returner,
            TacticalPlayerRole.SERVER,
            vector,
            (
                TacticalOutcomeLabel.SERVER_WON_POINT
                if attempt.server_won_point
                else TacticalOutcomeLabel.RETURNER_WON_POINT
            ),
            TacticalLabelAvailability.AVAILABLE,
            OBSERVATION_PROVENANCE,
        )
        requests.append(request)
        extractions.append(extraction)
        vectors.append(vector)
        observations.append(observation)
    return MaterializedHistory(
        tuple(requests), tuple(extractions), tuple(vectors), tuple(observations)
    )


def _query(
    schema: TacticalFeatureSchema,
    *,
    player: str = "Alice",
    opponent: str = "Bob",
    as_of_date: date = AS_OF_DATE,
    window_days: int | None = None,
    minimum_labeled_attempts: int = 50,
    minimum_matches: int = 5,
    requested_patterns: tuple[str, ...] = SCORING_PATTERNS,
) -> TacticalMatchupQuery:
    return TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        player,
        opponent,
        as_of_date,
        TacticalEncodingPolicy.COMPONENT_ONLY,
        feature_schema_fingerprint(schema),
        (1, 2),
        window_days,
        minimum_labeled_attempts,
        minimum_matches,
        requested_patterns,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )


def _run_chain(
    attempts: tuple[SyntheticAttempt, ...],
    *,
    player: str = "Alice",
    opponent: str = "Bob",
) -> EndToEndResult:
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    history = _materialize(attempts, schema)
    evidence = build_tactical_matchup_evidence(
        history.observations,
        _query(schema, player=player, opponent=opponent),
        schema,
    )
    prioritization = prioritize_tactical_matchup(evidence, top_k=3)
    response = build_public_tactical_recommendation(prioritization)
    return EndToEndResult(schema, history, evidence, prioritization, response)


@lru_cache(maxsize=2)
def _available_result(variant: int = 0) -> EndToEndResult:
    player, opponent, _, _, _ = _identity_tuple(variant)
    return _run_chain(
        _sufficient_history(variant), player=player, opponent=opponent
    )


@lru_cache(maxsize=1)
def _partial_result() -> EndToEndResult:
    return _run_chain(_p02_only_history())


@lru_cache(maxsize=1)
def _unavailable_result() -> EndToEndResult:
    return _run_chain(_unknown_and_censored_history())


def _identity_tuple(
    variant: int,
) -> tuple[str, str, str, str, str]:
    if variant == 0:
        return "Alice", "Bob", "Neutral", "h", ""
    return "Irene", "Jules", "Synthetic", "z", "c"


def _sufficient_history(variant: int = 0) -> tuple[SyntheticAttempt, ...]:
    player, opponent, neutral, namespace, let_prefix = _identity_tuple(variant)
    attempts: list[SyntheticAttempt] = []
    point = 1000 * variant + 1
    base_date = date(2020 - variant, 1, 2)

    for perspective in ("executor", "opponent"):
        for serve_code, category in SERVE_DIRECTIONS:
            for index in range(50):
                server = player if perspective == "executor" else f"{neutral}P02{category}"
                returner = f"{neutral}P02{category}" if perspective == "executor" else opponent
                attempts.append(
                    _attempt(
                        match_id=f"{namespace}-p02-{perspective}-{category}-{index // 10}",
                        point_number=point,
                        effective_date=base_date + timedelta(days=index % 20),
                        server=server,
                        returner=returner,
                        sequence=f"{let_prefix}{serve_code}#",
                        success=index % 2 == 0,
                    )
                )
                point += 1

    for perspective in ("executor", "opponent"):
        for shot_index, shot_type in enumerate(SHOT_TYPES):
            lateral = str(shot_index % 3 + 1)
            depth = str(shot_index % 3 + 7)
            for index in range(50):
                server = f"{neutral}P06{shot_type}" if perspective == "executor" else opponent
                returner = player if perspective == "executor" else f"{neutral}P06{shot_type}"
                second_serve = index % 2 == 1
                attempts.append(
                    _attempt(
                        match_id=f"{namespace}-p06-{perspective}-{shot_type}-{index // 10}",
                        point_number=point,
                        effective_date=base_date + timedelta(days=index % 20),
                        server=server,
                        returner=returner,
                        sequence=f"{let_prefix}4{shot_type}{lateral}{depth}",
                        serve_number=2 if second_serve else 1,
                        previous_fault=second_serve and index % 4 == 1,
                        success=index % 2 != 0,
                    )
                )
                point += 1
    return tuple(attempts)


def _p02_only_history() -> tuple[SyntheticAttempt, ...]:
    attempts: list[SyntheticAttempt] = []
    point = 1
    for perspective in ("executor", "opponent"):
        for serve_code, category in SERVE_DIRECTIONS:
            for index in range(50):
                attempts.append(
                    _attempt(
                        match_id=f"partial-{perspective}-{category}-{index // 10}",
                        point_number=point,
                        effective_date=date(2020, 3, 1) + timedelta(days=index % 20),
                        server="Alice" if perspective == "executor" else f"N{category}",
                        returner=f"N{category}" if perspective == "executor" else "Bob",
                        sequence=f"{serve_code}#",
                        success=index % 2 == 0,
                    )
                )
                point += 1
    return tuple(attempts)


def _unknown_and_censored_history() -> tuple[SyntheticAttempt, ...]:
    actor_pairs = (
        ("Alice", "N1"),
        ("N2", "Bob"),
        ("N3", "Alice"),
        ("Bob", "N4"),
    )
    sequences = ("0q00", "4", "4#", "4n")
    return tuple(
        _attempt(
            match_id=f"unavailable-{index}",
            point_number=index + 1,
            effective_date=date(2020, 5, index + 1),
            server=server,
            returner=returner,
            sequence=sequences[index],
            serve_number=2 if index == 3 else 1,
            previous_fault=index == 3,
            success=index % 2 == 0,
        )
        for index, (server, returner) in enumerate(actor_pairs)
    )


def _manual_wilson(successes: int, trials: int) -> tuple[float, float, float]:
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return rate, max(0.0, center - margin), min(1.0, center + margin)


def _card(response: TacticalRecommendationResponse, pattern_id: str):
    return next(card for card in response.cards if card.pattern_id == pattern_id)


def _assert_boundary_contracts(result: EndToEndResult) -> None:
    validate_tactical_feature_schema(result.schema)
    assert result.schema.policy is TacticalEncodingPolicy.COMPONENT_ONLY
    assert len(result.history.requests) == len(result.history.observations)
    keys = {
        (item.match_id, item.point_number, item.serve_number)
        for item in result.history.observations
    }
    assert len(keys) == len(result.history.observations)
    for extraction, vector, observation in zip(
        result.history.extractions,
        result.history.vectors,
        result.history.observations,
        strict=True,
    ):
        validate_attempt_signal_extraction(extraction)
        validate_tactical_signal_bundle(extraction.bundle)
        validate_tactical_feature_vector(vector)
        validate_tactical_historical_observation(observation)
    validate_tactical_matchup_evidence(result.evidence)
    validate_tactical_prioritization_result(result.prioritization)
    validate_public_tactical_recommendation(result.response)
    assert feature_schema_fingerprint(result.schema) == result.evidence.schema_fingerprint
    assert tactical_matchup_evidence_fingerprint(result.evidence)
    assert tactical_prioritization_result_fingerprint(result.prioritization)
    assert tactical_recommendation_fingerprint(result.response) == result.response.fingerprint
    assert result.evidence.unique_attempt_count == len(result.history.observations)
    assert sum(value for _, value in result.evidence.state_counts) == result.evidence.catalog_category_count
    assert (
        result.evidence.evidence_eligible_attempt_count
        + result.evidence.excluded_date_count
        + result.evidence.excluded_window_count
        + result.evidence.excluded_player_count
        + result.evidence.excluded_serve_number_count
        + result.evidence.excluded_pattern_policy_count
        == result.evidence.total_observations_received
    )


def _assert_evidence_arithmetic(result: EndToEndResult) -> None:
    for pair in result.evidence.categories:
        for component in (pair.executor_evidence, pair.opponent_allowed_evidence):
            assert component.labeled_activations == (
                component.tactical_successes + component.tactical_failures
            )
            assert component.category_activations <= component.observed_pattern_attempts
            assert component.observed_pattern_attempts <= component.applicable_attempts
            assert component.applicable_attempts <= component.relevant_attempt_count
            if component.labeled_activations:
                rate, lower, upper = _manual_wilson(
                    component.tactical_successes, component.labeled_activations
                )
                assert component.tactical_success_rate == pytest.approx(rate, abs=1e-15)
                assert component.wilson_lower == pytest.approx(lower, abs=1e-15)
                assert component.wilson_upper == pytest.approx(upper, abs=1e-15)
            else:
                assert component.tactical_success_rate is None
                assert component.wilson_lower is None
                assert component.wilson_upper is None


def _assert_public_projections(response: TacticalRecommendationResponse) -> None:
    assert tuple(card.pattern_id for card in response.cards) == SCORING_PATTERNS
    for card in response.cards:
        assert card.total_options == len(card.options)
        assert card.scored_options + card.abstained_options_count == card.total_options
        ranked_from_options = {
            option.category: option
            for option in card.options
            if option.rank_position is not None
        }
        abstained_from_options = {
            option.category: option
            for option in card.options
            if option.rank_position is None
        }
        assert {item.category: item for item in card.ranked_options} == ranked_from_options
        assert {item.category: item for item in card.abstained_options} == abstained_from_options
        if len(card.ranked_options) <= card.requested_top_k:
            expected_top = card.ranked_options
        elif card.ranked_options:
            boundary_group = card.ranked_options[card.requested_top_k - 1].tie_group
            expected_top = tuple(
                item for item in card.ranked_options if item.tie_group <= boundary_group
            )
        else:
            expected_top = ()
        assert card.top_options == expected_top
        for option in card.options:
            for component in (option.executor_evidence, option.opponent_allowed_evidence):
                assert component.labeled_activations == component.successes + component.failures
            if option.score is not None:
                expected = 0.5 * option.executor_evidence.success_rate + 0.5 * option.opponent_allowed_evidence.success_rate
                assert option.score == pytest.approx(expected, abs=1e-15)


def _assert_private_values_absent(
    response: TacticalRecommendationResponse,
    forbidden: tuple[str, ...],
) -> None:
    payload = canonical_tactical_recommendation_json(response)
    text = payload.decode("utf-8")
    for value in forbidden:
        assert value not in text
    for path_marker in ("C:\\", "/Users/", "file://", "<repository>"):
        assert path_marker not in text
    tree = json.loads(payload)
    forbidden_keys = {
        "match_id",
        "point_number",
        "sequence_text",
        "as_of_date",
        "effective_date",
        "span",
        "warnings",
        "residue",
        "label",
        "source_evidence_fingerprint",
        "schema_fingerprint",
        "query_fingerprint",
    }

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(tree)


def test_available_chain_reaches_public_contract_with_frozen_policy():
    result = _available_result()
    _assert_boundary_contracts(result)
    _assert_evidence_arithmetic(result)
    _assert_public_projections(result.response)

    assert result.response.status == "available"
    assert result.response.encoder_policy == "component_only"
    assert result.response.evidence_scope == "global_only"
    assert result.response.minimum_labeled_activations == 50
    assert result.response.minimum_distinct_matches == 5
    assert result.response.executor_weight == 0.5
    assert result.response.opponent_weight == 0.5
    assert result.response.requested_top_k == 3
    assert result.response.ranking_scope == "independent_within_pattern"
    assert result.response.global_cross_pattern_ranking is False
    assert result.response.methodology == "descriptive_observational"
    assert result.response.combination == "equal_weight_executor_opponent"
    assert result.response.score_formula == (
        "combined_rate = 0.5 * executor_success_rate + "
        "0.5 * opponent_allowed_success_rate"
    )
    assert result.prioritization.policy.abstention_rules == (
        "both_perspectives_must_be_available",
        "rates_and_wilson_bounds_must_be_present",
        "no_imputation_or_partial_score",
    )
    assert dict(result.prioritization.policy.reconciliations)["no_fitting_or_smoothing"] is True
    assert all(card.status == "available" for card in result.response.cards)
    assert all(option.score == pytest.approx(0.5) for card in result.response.cards for option in card.options)
    assert _card(result.response, "P02").actor == "server"
    assert all(
        _card(result.response, pattern_id).actor == "returner"
        for pattern_id in ("P04", "P05", "P06")
    )
    assert all(
        component.scope == "global"
        for card in result.response.cards
        for option in card.options
        for component in (option.executor_evidence, option.opponent_allowed_evidence)
    )
    for option in _card(result.response, "P06").options:
        assert option.executor_evidence.labeled_activations == 50
        assert option.opponent_allowed_evidence.labeled_activations == 50
        assert option.executor_evidence.distinct_matches == 5
        assert option.opponent_allowed_evidence.distinct_matches == 5


def test_partial_chain_scores_only_p02_and_preserves_explicit_abstentions():
    result = _partial_result()
    _assert_boundary_contracts(result)
    _assert_evidence_arithmetic(result)
    _assert_public_projections(result.response)

    assert result.response.status == "partially_available"
    assert _card(result.response, "P02").status == "available"
    for pattern_id in ("P04", "P05", "P06"):
        card = _card(result.response, pattern_id)
        assert card.status == "not_available"
        assert card.scored_options == 0
        assert card.top_options == ()
        assert {item.category: item for item in card.abstained_options} == {
            item.category: item for item in card.options
        }
        assert all(option.score is None for option in card.options)


def test_not_available_chain_never_fabricates_a_score():
    result = _unavailable_result()
    _assert_boundary_contracts(result)
    _assert_evidence_arithmetic(result)
    _assert_public_projections(result.response)

    assert result.response.status == "not_available"
    assert result.response.status_reason_codes == ("no_requested_pattern_has_scored_candidates",)
    assert all(card.status == "not_available" for card in result.response.cards)
    assert all(card.top_options == () for card in result.response.cards)
    assert all(
        option.score is None and option.rank_position is None
        for card in result.response.cards
        for option in card.options
    )


def test_boundary_tie_is_complete_and_canonically_ordered():
    result = _available_result()
    p06 = _card(result.response, "P06")
    assert len(p06.ranked_options) == len(SHOT_TYPES) == 17
    assert len(p06.top_options) == 17
    assert p06.tie_expanded is True
    assert p06.effective_top_k == 17
    assert {item.tie_group for item in p06.top_options} == {1}
    assert tuple(item.category for item in p06.ranked_options) == tuple(sorted(SHOT_TYPES))


def test_equal_weight_score_is_not_replaced_by_volume_weighting():
    base = _available_result()
    extra_attempts = tuple(
        _attempt(
            match_id=f"weight-wide-{index // 10}",
            point_number=5000 + index,
            effective_date=date(2020, 6, 1) + timedelta(days=index % 20),
            server="Alice",
            returner="WeightNeutral",
            sequence="4#",
            success=True,
        )
        for index in range(50)
    )
    extra = _materialize(extra_attempts, base.schema)
    evidence = build_tactical_matchup_evidence(
        base.history.observations + extra.observations,
        _query(base.schema),
        base.schema,
    )
    response = build_public_tactical_recommendation(
        prioritize_tactical_matchup(evidence, top_k=3)
    )
    wide = next(item for item in _card(response, "P02").options if item.category == "4")
    assert wide.executor_evidence.success_rate == pytest.approx(0.75)
    assert wide.opponent_allowed_evidence.success_rate == pytest.approx(0.5)
    assert wide.score == pytest.approx(0.625)
    assert wide.score != pytest.approx(100 / 150)


def test_unknown_zero_q_absence_and_censoring_are_never_negative_evidence():
    result = _unavailable_result()
    active_counts = {
        pattern: sum(
            vector.values[dict(result.schema.name_to_index)[name]]
            for vector in result.history.vectors
            for name in result.schema.tactical_feature_names
            if name.startswith(f"{pattern.lower()}.")
        )
        for pattern in SCORING_PATTERNS
    }
    assert active_counts == {"P02": 2, "P04": 0, "P05": 0, "P06": 0}
    for pair in result.evidence.categories:
        if pair.pattern_id == "P02":
            continue
        for component in (pair.executor_evidence, pair.opponent_allowed_evidence):
            assert component.category_activations == 0
            assert component.labeled_activations == 0
            assert component.tactical_failures == 0


def test_first_and_second_serve_contract_and_p09_internal_only():
    attempts = (
        _attempt(match_id="serve-1", point_number=1, effective_date=date(2020, 1, 1), server="Alice", returner="Bob", sequence="4f17"),
        _attempt(match_id="serve-2", point_number=2, effective_date=date(2020, 1, 2), server="Bob", returner="Alice", sequence="4f17", serve_number=2, previous_fault=True),
        _attempt(match_id="serve-3", point_number=3, effective_date=date(2020, 1, 3), server="Bob", returner="Alice", sequence="4f17", serve_number=2, previous_fault=False),
    )
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    history = _materialize(attempts, schema)
    first, after_fault, without_fault = history.extractions
    assert "P02" in first.applicable_patterns
    assert "P02" not in after_fault.applicable_patterns
    assert "P02" not in without_fault.applicable_patterns
    assert after_fault.previous_attempt_was_fault is True
    assert without_fault.previous_attempt_was_fault is False
    index = dict(schema.name_to_index)
    assert history.vectors[1].values[index["context.previous_attempt_was_fault"]] == 1
    assert history.vectors[2].values[index["context.previous_attempt_was_fault"]] == 0
    for vector in history.vectors[1:]:
        assert vector.values[index["mask.p02.applicable"]] == 0
        assert all(
            vector.values[index[name]] == 0
            for name in schema.tactical_feature_names
            if name.startswith("p02.")
        )
    assert all(
        next(item for item in extraction.adaptations if item.requested_pattern_id == "P09").signal is not None
        for extraction in history.extractions
    )
    assert "P09" not in schema.encoded_pattern_order
    assert not any(name.startswith("p09.") for name in schema.tactical_feature_names)

    result = _available_result()
    assert not any(card.pattern_id in {"P03", "P07", "P08", "P09"} for card in result.response.cards)
    payload = canonical_tactical_recommendation_json(result.response)
    assert all(f'"{pattern}"'.encode() not in payload for pattern in ("P03", "P07", "P08", "P09"))


def test_temporal_cut_is_strict_and_window_lower_bound_is_inclusive():
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    cutoff = date(2020, 2, 1)
    attempts = tuple(
        _attempt(
            match_id=f"time-{index}",
            point_number=index,
            effective_date=when,
            server="Alice",
            returner="N",
            sequence="4#",
        )
        for index, when in enumerate(
            (date(2020, 1, 1), date(2020, 1, 2), cutoff, date(2020, 2, 2)),
            start=1,
        )
    )
    history = _materialize(attempts, schema)
    evidence = build_tactical_matchup_evidence(
        history.observations,
        _query(
            schema,
            as_of_date=cutoff,
            window_days=30,
            minimum_labeled_attempts=1,
            minimum_matches=1,
            requested_patterns=("P02",),
        ),
        schema,
    )
    validate_tactical_matchup_evidence(evidence)
    assert evidence.total_observations_received == 4
    assert evidence.excluded_date_count == 2
    assert evidence.excluded_window_count == 1
    assert evidence.evidence_eligible_attempt_count == 1
    wide = next(item for item in evidence.categories if item.category == "4")
    assert wide.executor_evidence.labeled_activations == 1


def test_actor_ownership_and_alternating_servers_do_not_swap_labels():
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    attempts = (
        _attempt(match_id="actor-1", point_number=1, effective_date=date(2020, 1, 1), server="Alice", returner="N1", sequence="4#", success=True),
        _attempt(match_id="actor-2", point_number=2, effective_date=date(2020, 1, 2), server="N2", returner="Bob", sequence="4#", success=True),
        _attempt(match_id="actor-3", point_number=3, effective_date=date(2020, 1, 3), server="N3", returner="Alice", sequence="4f17", success=False),
        _attempt(match_id="actor-4", point_number=4, effective_date=date(2020, 1, 4), server="Bob", returner="N4", sequence="4f17", success=False),
    )
    history = _materialize(attempts, schema)
    assert all(item.role is TacticalPlayerRole.SERVER for item in history.observations)
    evidence = build_tactical_matchup_evidence(
        history.observations,
        _query(schema, minimum_labeled_attempts=1, minimum_matches=1),
        schema,
    )
    p02 = next(item for item in evidence.categories if item.pattern_id == "P02" and item.category == "4")
    p04 = next(item for item in evidence.categories if item.pattern_id == "P04" and item.category == "1")
    assert p02.executor_evidence.tactical_successes == 1
    assert p02.opponent_allowed_evidence.tactical_successes == 1
    assert p04.executor_evidence.tactical_successes == 1
    assert p04.opponent_allowed_evidence.tactical_successes == 1


def test_outcome_is_added_only_at_history_boundary_and_never_changes_features():
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    base = _attempt(match_id="label-a", point_number=1, effective_date=date(2020, 1, 1), server="Alice", returner="Bob", sequence="4f17", success=True)
    flipped = replace(base, match_id="label-b", server_won_point=False)
    left = _materialize((base,), schema)
    right = _materialize((flipped,), schema)
    assert left.requests[0] == right.requests[0]
    assert left.extractions[0] == right.extractions[0]
    assert left.vectors[0] == right.vectors[0]
    assert left.observations[0].label is TacticalOutcomeLabel.SERVER_WON_POINT
    assert right.observations[0].label is TacticalOutcomeLabel.RETURNER_WON_POINT


def test_semantically_equivalent_private_histories_have_identical_public_payload():
    left = _available_result(0)
    right = _available_result(1)
    left_json = canonical_tactical_recommendation_json(left.response)
    right_json = canonical_tactical_recommendation_json(right.response)
    assert left.response == right.response
    assert left_json == right_json
    assert left.response.fingerprint == right.response.fingerprint
    _assert_private_values_absent(
        left.response,
        ("Alice", "Bob", "h-p02", "4f17", "2020-01-02"),
    )
    _assert_private_values_absent(
        right.response,
        ("Irene", "Jules", "z-p02", "c4f17", "2019-01-02"),
    )


def test_input_order_and_repeated_builds_are_byte_deterministic():
    forward = _available_result()
    query = _query(forward.schema)

    def rebuild(observations):
        evidence = build_tactical_matchup_evidence(observations, query, forward.schema)
        prioritization = prioritize_tactical_matchup(evidence, top_k=3)
        return build_public_tactical_recommendation(prioritization)

    reverse = rebuild(tuple(reversed(forward.history.observations)))
    repeated = rebuild(forward.history.observations)
    expected = canonical_tactical_recommendation_json(forward.response)
    assert canonical_tactical_recommendation_json(reverse) == expected
    assert canonical_tactical_recommendation_json(repeated) == expected
    assert reverse.fingerprint == forward.response.fingerprint
    assert repeated.fingerprint == forward.response.fingerprint


def test_parser_is_called_once_per_attempt_and_downstream_layers_do_not_reparse(monkeypatch):
    original = orchestrator.parse_sequence
    calls: list[tuple[str, int]] = []

    def spy(sequence_text, serve_number):
        calls.append((sequence_text, serve_number))
        return original(sequence_text, serve_number)

    monkeypatch.setattr(orchestrator, "parse_sequence", spy)
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    attempts = (
        _attempt(match_id="parse-1", point_number=1, effective_date=date(2020, 1, 1), server="Alice", returner="Bob", sequence="4f17"),
        _attempt(match_id="parse-2", point_number=2, effective_date=date(2020, 1, 2), server="Bob", returner="Alice", sequence="4f17", serve_number=2, previous_fault=True),
    )
    history = _materialize(attempts, schema)
    evidence = build_tactical_matchup_evidence(
        history.observations,
        _query(schema, minimum_labeled_attempts=1, minimum_matches=1),
        schema,
    )
    prioritize_tactical_matchup(evidence, top_k=3)
    assert calls == [("4f17", 1), ("4f17", 2)]


def test_main_chain_performs_no_io_and_never_imports_p10(monkeypatch):
    forbidden_module = "src.analysis.tactical_recommender_pipeline"
    sys.modules.pop(forbidden_module, None)

    def forbidden_open(*args, **kwargs):
        raise AssertionError(f"E/S inesperada: {args!r} {kwargs!r}")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    result = _run_chain(_p02_only_history())
    assert result.response.status == "partially_available"
    assert forbidden_module not in sys.modules


def test_public_objects_are_deeply_immutable_and_semantic_mutations_fail():
    result = _available_result()
    response = result.response
    with pytest.raises(FrozenInstanceError):
        response.status = "not_available"  # type: ignore[misc]
    assert type(response.cards) is tuple
    assert type(response.reconciliations) is MappingProxyType
    with pytest.raises(TypeError):
        response.reconciliations["all"] = False  # type: ignore[index]
    with pytest.raises(TacticalRecommendationContractError):
        replace(response, fingerprint="0" * 64)
    with pytest.raises(TacticalRecommendationContractError):
        replace(response, cards=tuple(reversed(response.cards)))
    first_card = response.cards[0]
    with pytest.raises(TacticalRecommendationContractError):
        replace(first_card, top_options=())
    with pytest.raises(TypeError):
        first_card.reconciliations["all"] = False  # type: ignore[index]


def test_semantic_mutations_are_rejected_at_signal_vector_evidence_and_ranking_boundaries():
    result = _available_result()
    extraction = next(
        item for item in result.history.extractions if len(item.bundle.signals) > 1
    )
    with pytest.raises(TacticalSignalContractError):
        replace(extraction.bundle, signals=tuple(reversed(extraction.bundle.signals)))
    vector = result.history.vectors[0]
    with pytest.raises(TacticalFeatureContractError):
        replace(vector, values=(0,) * len(vector.values))
    with pytest.raises(TacticalMatchupContractError):
        replace(
            result.evidence,
            total_observations_received=result.evidence.total_observations_received + 1,
        )
    with pytest.raises(TacticalPrioritizationContractError):
        replace(
            result.prioritization,
            scored_candidate_count=result.prioritization.scored_candidate_count + 1,
        )


@pytest.mark.parametrize("bad_serve", [True, False, 1.0, 2.0, "1", "2", None])
def test_coercible_or_non_integer_serve_numbers_are_rejected(bad_serve):
    with pytest.raises((TypeError, ValueError)):
        extract_tactical_signals_for_attempt(
            AttemptSignalRequest(
                ORCHESTRATOR_CONTRACT_VERSION,
                "4f17",
                bad_serve,  # type: ignore[arg-type]
                False,
            )
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_public_numeric_values_are_rejected(bad_value):
    with pytest.raises(TacticalRecommendationContractError):
        PublicEvidenceComponent(
            "executor",
            "global",
            "available",
            50,
            25,
            25,
            5,
            bad_value,
            0.3,
            0.7,
        )


@pytest.mark.parametrize("bad_count", [True, False, 50.0, "50", None])
def test_public_counts_reject_bool_float_string_and_null(bad_count):
    with pytest.raises((TypeError, TacticalRecommendationContractError)):
        PublicEvidenceComponent(
            "executor",
            "global",
            "available",
            bad_count,  # type: ignore[arg-type]
            25,
            25,
            5,
            0.5,
            0.3,
            0.7,
        )


def test_fingerprint_domains_are_closed_and_distinct_across_every_boundary():
    domains = (
        ATTEMPT_FINGERPRINT_DOMAIN,
        SIGNAL_BUNDLE_FINGERPRINT_DOMAIN,
        SCHEMA_FINGERPRINT_DOMAIN,
        VECTOR_FINGERPRINT_DOMAIN,
        OBSERVATION_FINGERPRINT_DOMAIN,
        MATCHUP_EVIDENCE_FINGERPRINT_DOMAIN,
        RESULT_FINGERPRINT_DOMAIN,
        PUBLIC_FINGERPRINT_DOMAIN,
    )
    assert len(domains) == len(set(domains))
    result = _partial_result()
    extraction = result.history.extractions[0]
    vector = result.history.vectors[0]
    observation = result.history.observations[0]
    fingerprints = (
        attempt_extraction_fingerprint(extraction),
        tactical_signal_bundle_fingerprint(extraction.bundle),
        feature_schema_fingerprint(result.schema),
        feature_vector_fingerprint(vector),
        historical_observation_fingerprint(observation),
        tactical_matchup_evidence_fingerprint(result.evidence),
        tactical_prioritization_result_fingerprint(result.prioritization),
        tactical_recommendation_fingerprint(result.response),
    )
    assert all(len(value) == 64 and value == value.upper() for value in fingerprints)
    assert len(fingerprints) == len(set(fingerprints))
    assert sha256(canonical_tactical_recommendation_json(result.response)).hexdigest().upper() != result.response.fingerprint
