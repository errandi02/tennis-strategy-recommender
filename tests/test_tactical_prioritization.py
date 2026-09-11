"""Pruebas sinteticas adversariales de priorizacion tactica descriptiva."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.recommender.tactical_prioritization as prioritization
from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
    encode_tactical_attempt,
    feature_schema_fingerprint,
)
from src.recommender.tactical_history_profiles import (
    HISTORY_CONTRACT_VERSION,
    OBSERVATION_PROVENANCE,
    TacticalHistoricalObservation,
    TacticalLabelAvailability,
    TacticalOutcomeLabel,
    TacticalPlayerRole,
)
from src.recommender.tactical_matchup_evidence import (
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupQuery,
    build_tactical_matchup_evidence,
)
from src.recommender.tactical_prioritization import (
    CANDIDATE_FINGERPRINT_DOMAIN,
    EXPLANATION_FINGERPRINT_DOMAIN,
    POLICY_FINGERPRINT_DOMAIN,
    RANKING_FINGERPRINT_DOMAIN,
    RESULT_FINGERPRINT_DOMAIN,
    SCORE_FORMULA,
    UNCERTAINTY_METHOD,
    TacticalCandidateState,
    TacticalPatternRanking,
    TacticalPrioritizationContractError,
    TacticalPrioritizationState,
    TacticalRankingState,
    TacticalScoredCandidate,
    build_tactical_prioritization,
    canonical_tactical_candidate_explanation_json,
    canonical_tactical_pattern_ranking_json,
    canonical_tactical_prioritization_result_json,
    canonical_tactical_scored_candidate_json,
    canonical_tactical_scoring_policy_json,
    default_tactical_scoring_policy,
    explain_tactical_candidate,
    rank_tactical_pattern,
    score_tactical_candidate,
    tactical_candidate_explanation_fingerprint,
    tactical_pattern_ranking_fingerprint,
    tactical_prioritization_result_fingerprint,
    tactical_scored_candidate_fingerprint,
    tactical_scoring_policy_fingerprint,
    validate_tactical_candidate_explanation,
    validate_tactical_pattern_ranking,
    validate_tactical_prioritization_result,
    validate_tactical_scored_candidate,
    validate_tactical_scoring_policy,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
)


MODULE_PATH = Path(prioritization.__file__)


def _schema(policy=TacticalEncodingPolicy.COMPONENT_ONLY):
    return build_tactical_feature_schema(policy)


def _vector(sequence="4f17", policy=TacticalEncodingPolicy.COMPONENT_ONLY):
    extraction = extract_tactical_signals_for_attempt(
        AttemptSignalRequest(ORCHESTRATOR_CONTRACT_VERSION, sequence, 1, False)
    )
    return encode_tactical_attempt(extraction, _schema(policy))


def _observation(
    number,
    *,
    player,
    opponent,
    label,
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
    sequence="4f17",
):
    return TacticalHistoricalObservation(
        HISTORY_CONTRACT_VERSION,
        f"m{number:03d}",
        number,
        1,
        date(2020, 1, min(number, 28)),
        player,
        opponent,
        TacticalPlayerRole.SERVER,
        _vector(sequence, policy),
        label,
        TacticalLabelAvailability.AVAILABLE,
        OBSERVATION_PROVENANCE,
    )


def _query(policy, patterns, minimum=1):
    schema = _schema(policy)
    return TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        "Alice",
        "Bob",
        date(2020, 2, 1),
        policy,
        feature_schema_fingerprint(schema),
        (1,),
        None,
        minimum,
        minimum,
        patterns,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )


def _evidence(
    *,
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
    patterns=("P02",),
    executor=(True, False),
    opponent=(True, True),
    minimum=1,
):
    observations = []
    number = 1
    for won in executor:
        if "P02" in patterns:
            player, rival = "Alice", "Carol"
            label = TacticalOutcomeLabel.SERVER_WON_POINT if won else TacticalOutcomeLabel.RETURNER_WON_POINT
        else:
            player, rival = "Carol", "Alice"
            label = TacticalOutcomeLabel.RETURNER_WON_POINT if won else TacticalOutcomeLabel.SERVER_WON_POINT
        observations.append(
            _observation(number, player=player, opponent=rival, label=label, policy=policy)
        )
        number += 1
    for won in opponent:
        if "P02" in patterns:
            player, rival = "Carol", "Bob"
            label = TacticalOutcomeLabel.SERVER_WON_POINT if won else TacticalOutcomeLabel.RETURNER_WON_POINT
        else:
            player, rival = "Bob", "Carol"
            label = TacticalOutcomeLabel.RETURNER_WON_POINT if won else TacticalOutcomeLabel.SERVER_WON_POINT
        observations.append(
            _observation(number, player=player, opponent=rival, label=label, policy=policy)
        )
        number += 1
    query = _query(policy, patterns, minimum)
    return build_tactical_matchup_evidence(tuple(observations), query, _schema(policy))


def _source(evidence, pattern="P02", category=None):
    choices = [item for item in evidence.categories if item.pattern_id == pattern]
    if category is None:
        return next(item for item in choices if item.matchup_state.value == "comparable")
    return next(item for item in choices if item.category == category)


def _candidate(**kwargs):
    evidence = _evidence(**kwargs)
    return score_tactical_candidate(_source(evidence, kwargs.get("patterns", ("P02",))[0]), default_tactical_scoring_policy())


def test_policy_is_exact_versioned_immutable_and_50_50():
    policy = default_tactical_scoring_policy()
    assert policy.name == "equal_weight_descriptive_matchup"
    assert policy.version == "1.0.0"
    assert (policy.executor_weight, policy.opponent_allowed_weight) == (0.5, 0.5)
    assert policy.formula == SCORE_FORMULA
    assert policy.allowed_scope == "global"
    assert policy.uncertainty_method == UNCERTAINTY_METHOD
    with pytest.raises(FrozenInstanceError):
        policy.executor_weight = 0.7


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", "2.0.0"),
        ("name", "adaptive"),
        ("version", "2.0.0"),
        ("executor_weight", 1),
        ("executor_weight", 0.6),
        ("opponent_allowed_weight", 0.4),
        ("formula", "weighted by trials"),
        ("allowed_scope", "surface"),
        ("uncertainty_method", "joint_confidence_interval"),
        ("abstention_rules", ["mutable"]),
        ("interpretation", "causal"),
    ],
)
def test_policy_rejects_mutations_and_coercions(field, value):
    policy = default_tactical_scoring_policy()
    with pytest.raises((TacticalPrioritizationContractError, TypeError)):
        replace(policy, **{field: value})


def test_policy_serialization_and_domain_separated_fingerprint_are_exact():
    policy = default_tactical_scoring_policy()
    payload = canonical_tactical_scoring_policy_json(policy)
    assert json.loads(payload)["formula"] == SCORE_FORMULA
    assert b"NaN" not in payload and b"Infinity" not in payload
    assert tactical_scoring_policy_fingerprint(policy) == sha256(POLICY_FINGERPRINT_DOMAIN + payload).hexdigest().upper()


@pytest.mark.parametrize(
    ("executor", "opponent", "expected"),
    [
        ((False,), (False,), 0.0),
        ((True,), (True,), 1.0),
        ((True, False), (True, True), 0.75),
        ((True, True), (False, False), 0.5),
        ((False, False), (True, True), 0.5),
        ((True, False), (True, False), 0.5),
    ],
)
def test_scoring_exact_rates_bounds_floors_and_disagreement(executor, opponent, expected):
    candidate = _candidate(executor=executor, opponent=opponent)
    assert candidate.state is TacticalCandidateState.SCORED
    assert candidate.combined_rate == expected
    assert candidate.combined_lower == 0.5 * candidate.executor_evidence.wilson_lower + 0.5 * candidate.opponent_allowed_evidence.wilson_lower
    assert candidate.combined_upper == 0.5 * candidate.executor_evidence.wilson_upper + 0.5 * candidate.opponent_allowed_evidence.wilson_upper
    assert candidate.combined_width == candidate.combined_upper - candidate.combined_lower
    assert candidate.evidence_floor == min(len(executor), len(opponent))
    assert candidate.match_floor == min(len(executor), len(opponent))
    assert candidate.disagreement == abs(sum(executor) / len(executor) - sum(opponent) / len(opponent))
    assert 0 <= candidate.combined_lower <= candidate.combined_rate <= candidate.combined_upper <= 1


@pytest.mark.parametrize(
    ("executor", "opponent", "minimum", "reason"),
    [
        ((True,), (True, True), 2, "executor_evidence_insufficient"),
        ((True, True), (True,), 2, "opponent_allowed_evidence_insufficient"),
        ((True,), (True,), 2, "both_perspectives_insufficient"),
        ((), (), 1, "both_perspectives_not_available"),
    ],
)
def test_insufficient_or_missing_evidence_never_produces_partial_score(executor, opponent, minimum, reason):
    evidence = _evidence(executor=executor, opponent=opponent, minimum=minimum)
    source = next(item for item in evidence.categories if item.category == "4")
    candidate = score_tactical_candidate(source, default_tactical_scoring_policy())
    assert candidate.state is not TacticalCandidateState.SCORED
    assert candidate.reason_codes == (reason,)
    assert not candidate.comparable and not candidate.rank_eligible
    for field in ("combined_rate", "combined_lower", "combined_upper", "combined_width", "evidence_floor", "match_floor", "disagreement"):
        assert getattr(candidate, field) is None


def test_not_applicable_second_serve_p02_abstains():
    schema = _schema()
    query = replace(_query(TacticalEncodingPolicy.COMPONENT_ONLY, ("P02",)), serve_numbers=(2,))
    evidence = build_tactical_matchup_evidence((), query, schema)
    candidate = score_tactical_candidate(evidence.categories[0], default_tactical_scoring_policy())
    assert candidate.state is TacticalCandidateState.NOT_APPLICABLE
    assert candidate.combined_rate is None


@pytest.mark.parametrize(
    ("policy", "patterns", "pattern", "opportunity", "actor"),
    [
        (TacticalEncodingPolicy.COMPONENT_ONLY, ("P02",), "P02", "first_serve_direction", "server"),
        (TacticalEncodingPolicy.COMPONENT_ONLY, ("P04",), "P04", "initial_return_direction", "returner"),
        (TacticalEncodingPolicy.COMPONENT_ONLY, ("P05",), "P05", "initial_return_depth", "returner"),
        (TacticalEncodingPolicy.COMPONENT_ONLY, ("P06",), "P06", "initial_return_shot_type", "returner"),
        (TacticalEncodingPolicy.PROFILE_ONLY, ("P09",), "P09", "initial_return_profile", "returner"),
    ],
)
def test_closed_patterns_have_exact_opportunity_actor_and_success_perspective(policy, patterns, pattern, opportunity, actor):
    evidence = _evidence(policy=policy, patterns=patterns)
    candidate = score_tactical_candidate(_source(evidence, pattern), default_tactical_scoring_policy())
    assert candidate.pattern_id == pattern
    assert candidate.tactical_opportunity == opportunity
    assert candidate.actor == actor


@pytest.mark.parametrize("pattern", ["P03", "P07", "P08", "P10"])
def test_patterns_outside_scoring_catalog_are_rejected(pattern):
    candidate = _candidate()
    with pytest.raises(TacticalPrioritizationContractError):
        replace(candidate, pattern_id=pattern)


def test_component_profile_and_combined_policies_enforce_exact_pattern_sets():
    component = _evidence(patterns=("P02", "P04", "P05", "P06"))
    assert tuple(r.pattern_id for r in build_tactical_prioritization(component).rankings) == ("P02", "P04", "P05", "P06")
    profile = _evidence(policy=TacticalEncodingPolicy.PROFILE_ONLY, patterns=("P02", "P09"))
    assert tuple(r.pattern_id for r in build_tactical_prioritization(profile).rankings) == ("P02", "P09")
    combined = _evidence(policy=TacticalEncodingPolicy.COMPONENTS_AND_PROFILE, patterns=("P02", "P04", "P05", "P06", "P09"))
    result = build_tactical_prioritization(combined)
    assert result.redundant_representation is True
    assert all(r.redundant_representation for r in result.rankings)
    assert all(r.contractual_purpose == "audit_or_interaction_research_only" for r in result.rankings)
    assert not hasattr(result, "global_ranking")


def test_incompatible_pattern_request_fails_upstream_contract_not_silent_abstention():
    with pytest.raises(Exception, match="compatible"):
        _query(TacticalEncodingPolicy.COMPONENT_ONLY, ("P09",))


def _clone_category(category):
    base = _candidate()
    return replace(
        base,
        feature_name=f"p02.first_serve_direction.{category}",
        category=category,
        source_category_fingerprint="A"*64,
    )


def test_ranking_uses_all_four_numeric_criteria_then_canonical_feature():
    # Independent lightweight records expose the literal tuple used by ordering.
    item = SimpleNamespace(
        rank_eligible=True,
        combined_rate=0.8,
        combined_lower=0.6,
        combined_width=0.2,
        evidence_floor=50,
        feature_name="p02.first_serve_direction.4",
    )
    assert prioritization._rank_key(item) == (-0.8, -0.6, 0.2, -50, "p02.first_serve_direction.4")


def test_rank_key_orders_each_contractual_criterion_independently():
    def row(rate, lower, width, floor, feature):
        return SimpleNamespace(
            rank_eligible=True,
            combined_rate=rate,
            combined_lower=lower,
            combined_width=width,
            evidence_floor=floor,
            feature_name=feature,
        )
    cases = (
        (row(0.8, 0.1, 0.9, 1, "z"), row(0.7, 0.9, 0.1, 999, "a")),
        (row(0.5, 0.4, 0.9, 1, "z"), row(0.5, 0.3, 0.1, 999, "a")),
        (row(0.5, 0.3, 0.2, 1, "z"), row(0.5, 0.3, 0.3, 999, "a")),
        (row(0.5, 0.3, 0.2, 10, "z"), row(0.5, 0.3, 0.2, 9, "a")),
        (row(0.5, 0.3, 0.2, 10, "a"), row(0.5, 0.3, 0.2, 10, "z")),
    )
    for preferred, other in cases:
        assert sorted((other, preferred), key=prioritization._rank_key)[0] is preferred


def test_ranking_is_permutation_invariant_and_canonical_tie_break_is_not_preference():
    a = _clone_category("4")
    b = _clone_category("5")
    c = replace(
        _candidate(executor=(False,), opponent=(False,)),
        feature_name="p02.first_serve_direction.6",
        category="6",
        source_category_fingerprint="B" * 64,
    )
    kwargs = dict(pattern_id="P02", policy=default_tactical_scoring_policy(), encoder_policy=TacticalEncodingPolicy.COMPONENT_ONLY, top_k=1)
    left = rank_tactical_pattern((a, b, c), **kwargs)
    right = rank_tactical_pattern((c, b, a), **kwargs)
    assert canonical_tactical_pattern_ranking_json(left) == canonical_tactical_pattern_ranking_json(right)
    assert left.effective_top_k == 2
    assert left.boundary_tie_expanded is True
    assert [item.tie_group for item in left.top_candidates] == [1, 1]


@pytest.mark.parametrize("top_k", [0, -1, True, 1.0, "1"])
def test_top_k_rejects_non_positive_and_coercible_values(top_k):
    with pytest.raises(TacticalPrioritizationContractError):
        build_tactical_prioritization(_evidence(), top_k=top_k)


def test_top_k_may_exceed_scored_candidates_but_not_pattern_catalog():
    evidence = _evidence()
    result = build_tactical_prioritization(evidence, top_k=3)
    ranking = result.rankings[0]
    assert ranking.scored_candidate_count == 1
    assert ranking.effective_top_k == 1
    with pytest.raises(TacticalPrioritizationContractError):
        build_tactical_prioritization(evidence, top_k=4)


def test_empty_scored_ranking_and_counts_are_reconciled():
    result = build_tactical_prioritization(_evidence(executor=(), opponent=()))
    ranking = result.rankings[0]
    assert ranking.state is TacticalRankingState.NOT_AVAILABLE
    assert ranking.top_candidates == ()
    assert ranking.scored_candidate_count == 0
    assert ranking.abstained_candidate_count == 3


def test_ranking_rejects_missing_extra_and_cross_query_candidates():
    result = build_tactical_prioritization(_evidence())
    ranking = result.rankings[0]
    unranked = tuple(replace(item, rank_position=None, tie_group=None) for item in ranking.candidates)
    kwargs = dict(
        pattern_id="P02",
        policy=result.policy,
        encoder_policy=TacticalEncodingPolicy.COMPONENT_ONLY,
        top_k=1,
    )
    with pytest.raises(TacticalPrioritizationContractError, match="catalogo completo"):
        rank_tactical_pattern(unranked[:-1], **kwargs)
    with pytest.raises(TacticalPrioritizationContractError, match="duplicado"):
        rank_tactical_pattern(unranked + (unranked[0],), **kwargs)
    altered = replace(unranked[0], query_fingerprint="F" * 64)
    with pytest.raises(TacticalPrioritizationContractError, match="query"):
        rank_tactical_pattern((altered,) + unranked[1:], **kwargs)


def test_result_available_partial_and_not_available_states():
    available = build_tactical_prioritization(_evidence())
    assert available.state is TacticalPrioritizationState.AVAILABLE
    unavailable = build_tactical_prioritization(_evidence(executor=(), opponent=()))
    assert unavailable.state is TacticalPrioritizationState.NOT_AVAILABLE
    mixed_evidence = _evidence(patterns=("P02", "P04"))
    # The fixture orients history for P02, leaving P04 without comparable evidence.
    partial = build_tactical_prioritization(mixed_evidence)
    assert partial.state is TacticalPrioritizationState.PARTIALLY_AVAILABLE
    assert partial.available_pattern_count == 1
    assert partial.partial_or_unavailable_pattern_count == 1


def test_result_order_counts_limitations_and_no_cross_pattern_total():
    evidence = _evidence(patterns=("P02", "P04", "P05", "P06"))
    result = build_tactical_prioritization(evidence)
    assert tuple(item.pattern_id for item in result.rankings) == ("P02", "P04", "P05", "P06")
    assert result.total_candidate_count == sum(len(item.candidates) for item in result.rankings)
    assert result.scored_candidate_count + result.abstained_candidate_count == result.total_candidate_count
    assert "no_cross_pattern_global_ranking" in result.limitations
    assert "no_population_baseline_imputed" in result.limitations


def test_explanation_exact_scored_and_abstained_contract_without_causal_claims():
    scored = _candidate()
    explanation = explain_tactical_candidate(scored)
    assert explanation.executor_rate == scored.executor_evidence.success_rate
    assert explanation.opponent_allowed_rate == scored.opponent_allowed_evidence.success_rate
    assert explanation.combined_rate == scored.combined_rate
    assert "no implica causalidad" in explanation.canonical_text
    assert "garantiza" not in explanation.canonical_text.lower()
    abstained_source = _source(_evidence(executor=(), opponent=()), category="4")
    abstained = score_tactical_candidate(abstained_source, default_tactical_scoring_policy())
    abstained_explanation = explain_tactical_candidate(abstained)
    assert abstained_explanation.combined_rate is None
    assert "sin imputacion" in abstained_explanation.canonical_text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "5"),
        ("canonical_text", "garantiza una mejor opcion"),
        ("combined_rate", 0.1),
        ("executor_rate", 0.1),
        ("formula", "other"),
        ("scope", "surface"),
        ("interpretation", "causal"),
        ("state", TacticalCandidateState.NOT_AVAILABLE),
        ("reason_codes", ("other",)),
        ("evidence_floor", True),
    ],
)
def test_explanation_validator_rejects_semantic_mutations(field, value):
    explanation = explain_tactical_candidate(_candidate())
    with pytest.raises((TacticalPrioritizationContractError, TypeError)):
        replace(explanation, **{field: value})


@pytest.mark.parametrize(
    ("perspective", "field", "value"),
    [
        ("executor", "success_rate", 0.1),
        ("executor", "wilson_lower", 0.1),
        ("executor", "evidence_state", "not_available"),
        ("executor", "successes", 0),
        ("opponent", "wilson_upper", 0.9),
        ("opponent", "distinct_matches", 99),
    ],
)
def test_candidate_reconstructs_upstream_summary_semantics(perspective, field, value):
    candidate = _candidate()
    summary_field = "executor_evidence" if perspective == "executor" else "opponent_allowed_evidence"
    mutated_summary = replace(getattr(candidate, summary_field), **{field: value})
    with pytest.raises(TacticalPrioritizationContractError):
        replace(candidate, **{summary_field: mutated_summary})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("combined_rate", 0.1),
        ("combined_lower", 0.1),
        ("combined_upper", 0.9),
        ("combined_width", 0.1),
        ("evidence_floor", 99),
        ("match_floor", 99),
        ("disagreement", 0.1),
        ("state", TacticalCandidateState.NOT_AVAILABLE),
        ("reason_codes", ("both_perspectives_available",)),
        ("rank_eligible", False),
        ("comparable", False),
        ("actor", "returner"),
        ("tactical_opportunity", "other"),
        ("category", "unknown"),
        ("score_formula", "other"),
        ("provenance", (("source", "other"),)),
        ("reconciliations", (("score_reconciled", False),)),
    ],
)
def test_candidate_validator_rejects_semantic_mutations(field, value):
    with pytest.raises((TacticalPrioritizationContractError, TypeError)):
        replace(_candidate(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_top_k", 4),
        ("effective_top_k", 0),
        ("scored_candidate_count", 2),
        ("abstained_candidate_count", 0),
        ("tie_group_count", 9),
        ("boundary_tie_expanded", True),
        ("state", TacticalRankingState.NOT_AVAILABLE),
        ("reason_codes", ("other",)),
        ("order_rule", "other"),
        ("contractual_purpose", "production"),
    ],
)
def test_ranking_validator_rejects_mutations(field, value):
    ranking = build_tactical_prioritization(_evidence()).rankings[0]
    with pytest.raises((TacticalPrioritizationContractError, TypeError)):
        replace(ranking, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_pattern_count", 2),
        ("available_pattern_count", 0),
        ("partial_or_unavailable_pattern_count", 1),
        ("total_candidate_count", 2),
        ("scored_candidate_count", 0),
        ("abstained_candidate_count", 0),
        ("state", TacticalPrioritizationState.NOT_AVAILABLE),
        ("reason_codes", ("other",)),
        ("redundant_representation", True),
        ("limitations", ("causal",)),
        ("reconciliations", (("global_state_reconciled", False),)),
        ("provenance", (("source", "other"),)),
    ],
)
def test_result_validator_rejects_mutations(field, value):
    result = build_tactical_prioritization(_evidence())
    with pytest.raises((TacticalPrioritizationContractError, TypeError)):
        replace(result, **{field: value})


def test_result_reconstructs_query_fingerprint_and_rejects_invalid_calendar_date():
    result = build_tactical_prioritization(_evidence())
    with pytest.raises(TacticalPrioritizationContractError, match="Fingerprint"):
        replace(result, matchup_query=replace(result.matchup_query, query_fingerprint="A" * 64))
    with pytest.raises(TacticalPrioritizationContractError, match="as_of_date"):
        replace(result, matchup_query=replace(result.matchup_query, as_of_date="2020-02-31"))


def test_serialization_is_canonical_deterministic_and_domain_separated():
    result = build_tactical_prioritization(_evidence())
    candidate = result.rankings[0].candidates[0]
    ranking = result.rankings[0]
    explanation = explain_tactical_candidate(candidate)
    pairs = (
        (canonical_tactical_scored_candidate_json(candidate), CANDIDATE_FINGERPRINT_DOMAIN, tactical_scored_candidate_fingerprint(candidate)),
        (canonical_tactical_pattern_ranking_json(ranking), RANKING_FINGERPRINT_DOMAIN, tactical_pattern_ranking_fingerprint(ranking)),
        (canonical_tactical_candidate_explanation_json(explanation), EXPLANATION_FINGERPRINT_DOMAIN, tactical_candidate_explanation_fingerprint(explanation)),
        (canonical_tactical_prioritization_result_json(result), RESULT_FINGERPRINT_DOMAIN, tactical_prioritization_result_fingerprint(result)),
    )
    for payload, domain, fingerprint in pairs:
        assert fingerprint == sha256(domain + payload).hexdigest().upper()
        assert payload == payload.decode("utf-8").encode("utf-8")
        assert b"NaN" not in payload and b"Infinity" not in payload
    assert len({fingerprint for _, _, fingerprint in pairs}) == len(pairs)


def test_fingerprints_change_with_rate_category_order_top_k_and_reason():
    result = build_tactical_prioritization(_evidence())
    ranking = result.rankings[0]
    candidate = ranking.candidates[0]
    assert tactical_scored_candidate_fingerprint(candidate) != tactical_scored_candidate_fingerprint(replace(candidate, rank_position=2, tie_group=2))
    assert tactical_pattern_ranking_fingerprint(ranking) != tactical_pattern_ranking_fingerprint(rank_tactical_pattern(tuple(replace(item, rank_position=None, tie_group=None) for item in ranking.candidates), pattern_id="P02", policy=result.policy, encoder_policy=TacticalEncodingPolicy.COMPONENT_ONLY, top_k=2))
    unavailable = build_tactical_prioritization(_evidence(executor=(), opponent=()))
    assert tactical_prioritization_result_fingerprint(result) != tactical_prioritization_result_fingerprint(unavailable)


def test_dataclasses_and_nested_collections_are_deeply_immutable():
    result = build_tactical_prioritization(_evidence())
    objects = (result.policy, result.matchup_query, result.rankings[0], result.rankings[0].candidates[0], explain_tactical_candidate(result.rankings[0].candidates[0]))
    for item in objects:
        with pytest.raises(FrozenInstanceError):
            setattr(item, fields(item)[0].name, None)
    assert type(result.rankings) is tuple
    assert type(result.limitations) is tuple
    assert type(result.rankings[0].candidates) is tuple
    assert type(result.rankings[0].candidates[0].provenance) is tuple


@pytest.mark.parametrize(
    "malicious",
    [
        (("match_id", "m1"),),
        (("point_number", "1"),),
        (("sequence_text", "4f17"),),
        (("individual_outcome", "won"),),
        (("path", "C:\\secret\\x"),),
        (("source", "file://secret"),),
        (("source", "../secret"),),
        (("timestamp", "2026-01-01T00:00:00"),),
        (("test_status", "opened"),),
        (("evaluation", "done"),),
    ],
)
def test_public_tree_rejects_sensitive_provenance(malicious):
    candidate = _candidate()
    # Construction itself rejects all but structurally valid fixed provenance;
    # exercise the recursive guard directly for every forbidden family.
    with pytest.raises(TacticalPrioritizationContractError):
        prioritization._validate_public_tree(dict(malicious))
    with pytest.raises(TacticalPrioritizationContractError):
        replace(candidate, provenance=malicious)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected(value):
    with pytest.raises(TacticalPrioritizationContractError):
        replace(_candidate(), combined_rate=value)
    with pytest.raises(TacticalPrioritizationContractError):
        prioritization._validate_public_tree({"value": value})


def test_mutable_and_custom_public_objects_are_rejected():
    class Custom:
        pass
    with pytest.raises(TacticalPrioritizationContractError):
        prioritization._validate_public_tree({"value": ("mutable-or-custom",)})
    with pytest.raises(TacticalPrioritizationContractError):
        prioritization._validate_public_tree({"value": Custom()})


def test_serialized_result_contains_query_players_but_no_individual_history_or_global_ranking():
    payload = canonical_tactical_prioritization_result_json(build_tactical_prioritization(_evidence()))
    text = payload.decode("utf-8")
    assert '"player":"Alice"' in text and '"opponent":"Bob"' in text
    for forbidden in ("match_id", "point_number", "sequence_text", "point_winner", '"global_ranking":', "timestamp", "data/", "reports/"):
        assert forbidden not in text


def test_module_architecture_is_pure_dependency_free_and_has_no_productive_asserts():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not ({"pandas", "numpy", "scipy", "sklearn"} & imports)
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not ({"open", "print", "input", "read_csv", "read_parquet"} & calls)
    assert "__main__" not in source
    assert "data/" not in source and "reports/" not in source
    assert "parse_sequence" not in source
    assert "explainable_direction_recommender" not in source


def test_module_has_no_mutable_global_collections():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            assert not isinstance(value, (ast.List, ast.Dict, ast.Set))
