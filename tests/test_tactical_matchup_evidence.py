"""Pruebas sinteticas adversariales de evidencia tactica jugador-rival."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path

import pytest

import src.recommender.tactical_matchup_evidence as matchup
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
    CATEGORY_EVIDENCE_FINGERPRINT_DOMAIN,
    MATCHUP_CATEGORY_FINGERPRINT_DOMAIN,
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    MATCHUP_EVIDENCE_FINGERPRINT_DOMAIN,
    MATCHUP_QUERY_FINGERPRINT_DOMAIN,
    PATTERN_OWNERSHIP_CATALOG,
    TacticalCategoryEvidence,
    TacticalCategoryEvidenceState,
    TacticalEvidencePerspective,
    TacticalEvidenceScope,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupCategoryEvidence,
    TacticalMatchupCategoryState,
    TacticalMatchupContractError,
    TacticalMatchupEvidence,
    TacticalMatchupEvidenceState,
    TacticalMatchupQuery,
    build_tactical_matchup_evidence,
    canonical_tactical_category_evidence_json,
    canonical_tactical_matchup_category_evidence_json,
    canonical_tactical_matchup_evidence_json,
    canonical_tactical_matchup_query_json,
    tactical_category_evidence_fingerprint,
    tactical_matchup_category_evidence_fingerprint,
    tactical_matchup_evidence_fingerprint,
    tactical_matchup_query_fingerprint,
    tactical_wilson_interval,
    validate_tactical_category_evidence,
    validate_tactical_matchup_category_evidence,
    validate_tactical_matchup_evidence,
    validate_tactical_matchup_query,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
)


MODULE_PATH = Path(matchup.__file__)
_DEFAULT_LABEL = object()


def _schema(policy=TacticalEncodingPolicy.COMPONENT_ONLY):
    return build_tactical_feature_schema(policy)


def _vector(
    sequence="4f17",
    serve_number=1,
    previous_fault=False,
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
):
    extraction = extract_tactical_signals_for_attempt(
        AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION,
            sequence,
            serve_number,
            previous_fault,
        )
    )
    return encode_tactical_attempt(extraction, _schema(policy))


def _observation(
    *,
    match_id="m001",
    point_number=1,
    effective_date=date(2020, 1, 1),
    player="Alice",
    opponent="Bob",
    role=TacticalPlayerRole.SERVER,
    sequence="4f17",
    serve_number=1,
    previous_fault=False,
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
    label=TacticalOutcomeLabel.SERVER_WON_POINT,
):
    return TacticalHistoricalObservation(
        HISTORY_CONTRACT_VERSION,
        match_id,
        point_number,
        serve_number,
        effective_date,
        player,
        opponent,
        role,
        _vector(sequence, serve_number, previous_fault, policy),
        label,
        (
            TacticalLabelAvailability.AVAILABLE
            if label is not None
            else TacticalLabelAvailability.NOT_AVAILABLE
        ),
        OBSERVATION_PROVENANCE,
    )


def _query(
    *,
    player="Alice",
    opponent="Bob",
    as_of_date=date(2020, 2, 1),
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
    serve_numbers=(1, 2),
    window_days=None,
    minimum_labeled_attempts=1,
    minimum_matches=1,
    requested_patterns=("P02",),
    scope_strategy=TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
):
    schema = _schema(policy)
    return TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        player,
        opponent,
        as_of_date,
        policy,
        feature_schema_fingerprint(schema),
        serve_numbers,
        window_days,
        minimum_labeled_attempts,
        minimum_matches,
        requested_patterns,
        scope_strategy,
    )


def _build(observations, query=None):
    if query is None:
        query = _query()
    return build_tactical_matchup_evidence(
        tuple(observations), query, _schema(query.policy)
    )


def _pair(result, feature_name):
    return next(item for item in result.categories if item.feature_name == feature_name)


def _p02_history(*, executor=True, opponent_allowed=True, label=_DEFAULT_LABEL):
    selected_label = (
        TacticalOutcomeLabel.SERVER_WON_POINT
        if label is _DEFAULT_LABEL
        else label
    )
    observations = []
    if executor:
        observations.append(
            _observation(
                match_id="executor-match",
                player="Alice",
                opponent="Dora",
                label=selected_label,
            )
        )
    if opponent_allowed:
        observations.append(
            _observation(
                match_id="opponent-match",
                player="Carol",
                opponent="Bob",
                label=selected_label,
            )
        )
    return tuple(observations)


def _return_history(
    *,
    executor=True,
    opponent_allowed=True,
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
    label=TacticalOutcomeLabel.RETURNER_WON_POINT,
):
    observations = []
    if executor:
        observations.append(
            _observation(
                match_id="return-executor",
                player="Dora",
                opponent="Alice",
                role=TacticalPlayerRole.SERVER,
                policy=policy,
                label=label,
            )
        )
    if opponent_allowed:
        observations.append(
            _observation(
                match_id="return-opponent",
                player="Bob",
                opponent="Carol",
                role=TacticalPlayerRole.SERVER,
                policy=policy,
                label=label,
            )
        )
    return tuple(observations)


def test_pattern_ownership_catalog_is_exact_immutable_and_closed():
    assert type(PATTERN_OWNERSHIP_CATALOG) is tuple
    assert tuple(item.pattern_id for item in PATTERN_OWNERSHIP_CATALOG) == (
        "P02", "P04", "P05", "P06", "P09"
    )
    expected = {
        "P02": (TacticalPlayerRole.SERVER, TacticalPlayerRole.RETURNER, TacticalOutcomeLabel.SERVER_WON_POINT),
        "P04": (TacticalPlayerRole.RETURNER, TacticalPlayerRole.SERVER, TacticalOutcomeLabel.RETURNER_WON_POINT),
        "P05": (TacticalPlayerRole.RETURNER, TacticalPlayerRole.SERVER, TacticalOutcomeLabel.RETURNER_WON_POINT),
        "P06": (TacticalPlayerRole.RETURNER, TacticalPlayerRole.SERVER, TacticalOutcomeLabel.RETURNER_WON_POINT),
        "P09": (TacticalPlayerRole.RETURNER, TacticalPlayerRole.SERVER, TacticalOutcomeLabel.RETURNER_WON_POINT),
    }
    for item in PATTERN_OWNERSHIP_CATALOG:
        assert (item.executor_role, item.exposed_role, item.tactical_success_label) == expected[item.pattern_id]
        with pytest.raises(FrozenInstanceError):
            item.pattern_id = "P03"
    with pytest.raises(TacticalMatchupContractError, match="catalogo"):
        matchup.TacticalPatternOwnership(
            "P03",
            "invented",
            TacticalPlayerRole.SERVER,
            TacticalPlayerRole.RETURNER,
            TacticalOutcomeLabel.SERVER_WON_POINT,
        )


@pytest.mark.parametrize("pattern", ["P03", "P07", "P08", "P10"])
def test_query_rejects_patterns_outside_closed_catalog(pattern):
    query = _query()
    object.__setattr__(query, "requested_patterns", (pattern,))
    with pytest.raises(TacticalMatchupContractError, match="patron invalido"):
        validate_tactical_matchup_query(query)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("contract_version", "0", "Version"),
        ("player", "", "player"),
        ("player", " Alice", "player"),
        ("opponent", "Bob ", "opponent"),
        ("as_of_date", datetime(2020, 1, 1), "date"),
        ("as_of_date", "2020-01-01", "date"),
        ("policy", "component_only", "policy"),
        ("schema_fingerprint", "A" * 64, "reconcilia"),
        ("serve_numbers", [1, 2], "serve_numbers"),
        ("serve_numbers", (True,), "serve_numbers"),
        ("serve_numbers", (2, 1), "serve_numbers"),
        ("window_days", 0, "positivo"),
        ("window_days", True, "int real"),
        ("window_days", 1.0, "int real"),
        ("minimum_labeled_attempts", True, "int real"),
        ("minimum_labeled_attempts", -1, "no negativo"),
        ("minimum_matches", 1.0, "int real"),
        ("requested_patterns", ["P02"], "tuple"),
        ("requested_patterns", ("P04", "P02"), "orden canonico"),
        ("requested_patterns", ("P02", "P02"), "orden canonico"),
        ("scope_strategy", "global_only", "scope_strategy"),
    ],
)
def test_query_rejects_invalid_exact_domains(field, value, error):
    query = _query()
    object.__setattr__(query, field, value)
    with pytest.raises((TypeError, TacticalMatchupContractError), match=error):
        validate_tactical_matchup_query(query)


def test_query_rejects_equal_players_and_window_calendar_overflow():
    with pytest.raises(TacticalMatchupContractError, match="distintos"):
        _query(opponent="Alice")
    with pytest.raises(TacticalMatchupContractError, match="calendario"):
        _query(as_of_date=date.min, window_days=1)


@pytest.mark.parametrize(
    "policy,valid_patterns,invalid_pattern",
    [
        (TacticalEncodingPolicy.COMPONENT_ONLY, ("P02", "P04", "P05", "P06"), "P09"),
        (TacticalEncodingPolicy.PROFILE_ONLY, ("P02", "P09"), "P04"),
        (TacticalEncodingPolicy.COMPONENTS_AND_PROFILE, ("P02", "P04", "P05", "P06", "P09"), None),
    ],
)
def test_policy_pattern_compatibility_is_explicit(policy, valid_patterns, invalid_pattern):
    validate_tactical_matchup_query(_query(policy=policy, requested_patterns=valid_patterns))
    if invalid_pattern is not None:
        with pytest.raises(TacticalMatchupContractError, match="compatible"):
            _query(policy=policy, requested_patterns=(invalid_pattern,))


def test_upstream_observation_orientation_derives_both_actors_without_duplication():
    server_oriented = _observation(player="Alice", opponent="Bob", role=TacticalPlayerRole.SERVER)
    returner_oriented = _observation(player="Bob", opponent="Alice", role=TacticalPlayerRole.RETURNER, match_id="m002")
    left = _build((server_oriented,))
    right = _build((returner_oriented,))
    left_pair = _pair(left, "p02.first_serve_direction.4")
    right_pair = _pair(right, "p02.first_serve_direction.4")
    assert left_pair.executor_evidence.category_activations == right_pair.executor_evidence.category_activations == 1
    assert left_pair.opponent_allowed_evidence.category_activations == right_pair.opponent_allowed_evidence.category_activations == 1
    assert left.unique_attempt_count == right.unique_attempt_count == 1


def test_duplicate_attempt_exact_or_conflicting_is_rejected_without_id_disclosure():
    first = _observation(match_id="private-match")
    for second in (
        first,
        _observation(match_id="private-match", sequence="5b28"),
    ):
        with pytest.raises(TacticalMatchupContractError) as captured:
            _build((first, second))
        assert "duplicada" in str(captured.value)
        assert "private-match" not in str(captured.value)


def test_mutable_observation_collection_and_invalid_irrelevant_row_are_rejected():
    with pytest.raises(TypeError, match="tuple"):
        build_tactical_matchup_evidence([], _query(), _schema())
    irrelevant = _observation(player="Carol", opponent="Dora")
    object.__setattr__(irrelevant, "point_number", True)
    with pytest.raises(Exception, match="point_number"):
        _build((irrelevant,))


def test_p02_executor_and_opponent_allowed_use_correct_people_and_same_category():
    result = _build(_p02_history())
    pair = _pair(result, "p02.first_serve_direction.4")
    assert pair.category == "4"
    assert pair.executor_evidence.perspective is TacticalEvidencePerspective.EXECUTOR
    assert pair.opponent_allowed_evidence.perspective is TacticalEvidencePerspective.OPPONENT_ALLOWED
    assert pair.executor_evidence.executor_role is TacticalPlayerRole.SERVER
    assert pair.executor_evidence.category_activations == 1
    assert pair.opponent_allowed_evidence.category_activations == 1
    assert pair.matchup_state is TacticalMatchupCategoryState.COMPARABLE
    assert result.evidence_state is TacticalMatchupEvidenceState.AVAILABLE


@pytest.mark.parametrize(
    "pattern,feature",
    [
        ("P04", "p04.return_direction.1"),
        ("P05", "p05.return_depth.7"),
        ("P06", "p06.return_shot_type.f"),
    ],
)
def test_return_patterns_executor_returner_and_exposed_server(pattern, feature):
    query = _query(requested_patterns=(pattern,))
    result = _build(_return_history(), query)
    pair = _pair(result, feature)
    assert pair.executor_evidence.executor_role is TacticalPlayerRole.RETURNER
    assert pair.executor_evidence.category_activations == 1
    assert pair.opponent_allowed_evidence.category_activations == 1
    assert pair.executor_evidence.tactical_successes == 1
    assert pair.opponent_allowed_evidence.tactical_successes == 1
    assert pair.matchup_state is TacticalMatchupCategoryState.COMPARABLE


def test_p09_preserves_full_profile_category_and_returner_success():
    policy = TacticalEncodingPolicy.PROFILE_ONLY
    query = _query(policy=policy, requested_patterns=("P09",))
    result = _build(_return_history(policy=policy), query)
    pair = _pair(result, "p09.return_profile.f|1|7")
    assert pair.category == "f|1|7"
    assert pair.executor_evidence.executor_role is TacticalPlayerRole.RETURNER
    assert pair.executor_evidence.tactical_success_rate == 1.0
    assert pair.opponent_allowed_evidence.tactical_success_rate == 1.0


def test_opponent_executing_action_is_not_confused_with_opponent_exposure():
    observations = (
        _observation(match_id="bob-serves", player="Bob", opponent="Dora"),
        _observation(match_id="carol-serves", player="Carol", opponent="Bob"),
    )
    result = _build(observations)
    opponent = _pair(result, "p02.first_serve_direction.4").opponent_allowed_evidence
    assert opponent.category_activations == 1
    assert opponent.distinct_match_count == 1


def test_tactical_outcome_is_always_from_executor_pattern_perspective():
    p02 = _build(
        _p02_history(label=TacticalOutcomeLabel.RETURNER_WON_POINT)
    )
    p02_pair = _pair(p02, "p02.first_serve_direction.4")
    assert p02_pair.executor_evidence.tactical_successes == 0
    assert p02_pair.executor_evidence.tactical_failures == 1
    returns = _build(
        _return_history(label=TacticalOutcomeLabel.SERVER_WON_POINT),
        _query(requested_patterns=("P04",)),
    )
    return_pair = _pair(returns, "p04.return_direction.1")
    assert return_pair.executor_evidence.tactical_successes == 0
    assert return_pair.opponent_allowed_evidence.tactical_failures == 1


def test_unlabeled_activations_are_not_failures():
    result = _build(_p02_history(label=None))
    pair = _pair(result, "p02.first_serve_direction.4")
    for evidence in (pair.executor_evidence, pair.opponent_allowed_evidence):
        assert evidence.category_activations == 1
        assert evidence.relevant_attempt_count == 1
        assert evidence.labeled_activations == 0
        assert evidence.tactical_successes == evidence.tactical_failures == 0
        assert evidence.tactical_success_rate is None
        assert evidence.wilson_lower is evidence.wilson_upper is None


@pytest.mark.parametrize(
    "successes,trials",
    [(0, 1), (0, 3), (3, 3), (1, 3), (2, 3)],
)
def test_wilson_matches_independent_formula_and_exact_boundaries(successes, trials):
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    margin = z * (
        rate * (1 - rate) / trials + z * z / (4 * trials * trials)
    ) ** 0.5 / denominator
    expected_lower = 0.0 if successes == 0 else centre - margin
    expected_upper = 1.0 if successes == trials else centre + margin
    actual = tactical_wilson_interval(
        successes, trials, feature_name="p02.first_serve_direction.4"
    )
    assert actual == pytest.approx((rate, expected_lower, expected_upper), abs=1e-15)
    assert 0 <= actual[1] <= actual[0] <= actual[2] <= 1


def test_wilson_zero_trials_and_invalid_domains_fail_safely():
    assert tactical_wilson_interval(0, 0, feature_name="safe.feature") == (
        None, None, None
    )
    for successes, trials, feature in (
        (True, 1, "safe.feature"),
        (1.0, 1, "safe.feature"),
        ("1", 1, "safe.feature"),
        (-1, 1, "safe.feature"),
        (2, 1, "safe.feature"),
        (0, False, "safe.feature"),
        (0, 1, "unsafe/path"),
    ):
        with pytest.raises(TacticalMatchupContractError) as captured:
            tactical_wilson_interval(successes, trials, feature_name=feature)
        assert "match" not in str(captured.value).lower()


def test_unknown_or_censored_pattern_is_applicable_but_not_observed_category():
    observations = (
        _observation(match_id="a", player="Dora", opponent="Alice", sequence="6"),
        _observation(match_id="b", player="Bob", opponent="Carol", sequence="6"),
    )
    result = _build(observations, _query(requested_patterns=("P04",)))
    pair = _pair(result, "p04.return_direction.1")
    assert pair.executor_evidence.applicable_attempts == 1
    assert pair.executor_evidence.observed_pattern_attempts == 0
    assert pair.executor_evidence.category_activations == 0
    assert pair.executor_evidence.evidence_state is TacticalCategoryEvidenceState.NO_OBSERVED_CATEGORY
    assert result.candidate_category_count == 0
    assert result.evidence_state is TacticalMatchupEvidenceState.NOT_AVAILABLE


def test_p02_is_not_applicable_when_query_allows_only_second_serves():
    query = _query(serve_numbers=(2,), requested_patterns=("P02",))
    observations = (
        _observation(match_id="a", player="Alice", opponent="Dora", serve_number=2, previous_fault=True),
        _observation(match_id="b", player="Carol", opponent="Bob", serve_number=2, previous_fault=True),
    )
    result = _build(observations, query)
    assert result.excluded_pattern_policy_count == 2
    assert all(item.matchup_state is TacticalMatchupCategoryState.NOT_APPLICABLE for item in result.categories)
    evidence = result.categories[0].executor_evidence
    object.__setattr__(evidence, "applicable_attempts", 1)
    with pytest.raises(TacticalMatchupContractError):
        validate_tactical_category_evidence(evidence)


def test_combined_query_distinguishes_relevant_second_serve_from_missing_p02_history():
    query = _query(requested_patterns=("P02", "P04"), serve_numbers=(1, 2))
    observations = (
        _observation(
            match_id="a",
            player="Alice",
            opponent="Dora",
            serve_number=2,
            previous_fault=True,
        ),
        _observation(
            match_id="b",
            player="Carol",
            opponent="Bob",
            serve_number=2,
            previous_fault=True,
        ),
    )
    result = _build(observations, query)
    p02 = _pair(result, "p02.first_serve_direction.4")
    assert p02.executor_evidence.relevant_attempt_count == 1
    assert p02.executor_evidence.applicable_attempts == 0
    assert p02.executor_evidence.evidence_state is TacticalCategoryEvidenceState.NO_OBSERVED_CATEGORY
    validate_tactical_matchup_evidence(result)


def test_strict_cut_window_and_exclusion_counts_are_exact():
    query = _query(window_days=10, serve_numbers=(1,))
    observations = (
        _observation(match_id="eligible", effective_date=date(2020, 1, 22), player="Alice", opponent="Dora"),
        _observation(match_id="too-old", effective_date=date(2020, 1, 21), player="Alice", opponent="Dora"),
        _observation(match_id="cut", effective_date=date(2020, 2, 1), player="Alice", opponent="Dora"),
        _observation(match_id="future", effective_date=date(2020, 2, 2), player="Alice", opponent="Dora"),
        _observation(match_id="wrong-serve", effective_date=date(2020, 1, 25), player="Alice", opponent="Dora", serve_number=2, previous_fault=True),
        _observation(match_id="wrong-players", effective_date=date(2020, 1, 25), player="Carol", opponent="Dora"),
    )
    result = _build(observations, query)
    assert result.total_observations_received == result.unique_attempt_count == 6
    assert result.evidence_eligible_attempt_count == 1
    assert result.excluded_date_count == 2
    assert result.excluded_window_count == 1
    assert result.excluded_serve_number_count == 1
    assert result.excluded_player_count == 1
    assert result.excluded_pattern_policy_count == 0


def test_future_rows_and_labels_never_reach_category_accumulator(monkeypatch):
    query = _query()
    observations = _p02_history() + (
        _observation(
            match_id="future-secret",
            effective_date=query.as_of_date,
            player="Alice",
            opponent="Bob",
            sequence="5b28",
            label=TacticalOutcomeLabel.RETURNER_WON_POINT,
        ),
    )
    original = matchup._build_category_evidence
    seen = []

    def guarded(**kwargs):
        received = kwargs["observations"]
        seen.extend(item.match_id for item in received)
        assert all(item.effective_date < query.as_of_date for item in received)
        assert all(item.match_id != "future-secret" for item in received)
        return original(**kwargs)

    monkeypatch.setattr(matchup, "_build_category_evidence", guarded)
    result = _build(observations, query)
    assert result.excluded_date_count == 1
    assert "future-secret" not in seen


def test_permutation_and_same_day_attempts_produce_identical_bytes():
    observations = _p02_history() + (
        _observation(match_id="same-day", point_number=2, player="Alice", opponent="Eve", sequence="5b28"),
    )
    left = _build(observations)
    right = _build(tuple(reversed(observations)))
    assert left == right
    assert canonical_tactical_matchup_evidence_json(left) == canonical_tactical_matchup_evidence_json(right)
    assert tactical_matchup_evidence_fingerprint(left) == tactical_matchup_evidence_fingerprint(right)


@pytest.mark.parametrize(
    "executor,opponent,state",
    [
        (True, True, TacticalMatchupCategoryState.COMPARABLE),
        (True, False, TacticalMatchupCategoryState.OPPONENT_INSUFFICIENT),
        (False, True, TacticalMatchupCategoryState.EXECUTOR_INSUFFICIENT),
        (False, False, TacticalMatchupCategoryState.NOT_AVAILABLE),
    ],
)
def test_joint_states_for_each_perspective(executor, opponent, state):
    result = _build(_p02_history(executor=executor, opponent_allowed=opponent))
    pair = _pair(result, "p02.first_serve_direction.4")
    assert pair.matchup_state is state


def test_both_insufficient_when_both_have_observations_below_minimums():
    query = _query(minimum_labeled_attempts=2, minimum_matches=2)
    result = _build(_p02_history(), query)
    pair = _pair(result, "p02.first_serve_direction.4")
    assert pair.executor_evidence.evidence_state is TacticalCategoryEvidenceState.INSUFFICIENT_LABELED_ATTEMPTS
    assert pair.opponent_allowed_evidence.evidence_state is TacticalCategoryEvidenceState.INSUFFICIENT_LABELED_ATTEMPTS
    assert pair.matchup_state is TacticalMatchupCategoryState.BOTH_INSUFFICIENT


def test_minimum_matches_is_based_on_distinct_active_matches():
    observations = (
        _observation(match_id="same", point_number=1, player="Alice", opponent="Dora"),
        _observation(match_id="same", point_number=2, player="Alice", opponent="Eve"),
        _observation(match_id="opp", player="Carol", opponent="Bob"),
    )
    result = _build(observations, _query(minimum_matches=2))
    executor = _pair(result, "p02.first_serve_direction.4").executor_evidence
    assert executor.category_activations == 2
    assert executor.distinct_match_count == 1
    assert executor.evidence_state is TacticalCategoryEvidenceState.INSUFFICIENT_MATCHES


def test_global_status_uses_only_observed_candidate_union_not_zero_catalog_rows():
    result = _build(_p02_history())
    assert result.catalog_category_count == 3
    assert result.candidate_category_count == 1
    assert result.comparable_category_count == 1
    assert result.insufficient_category_count == 0
    assert result.evidence_state is TacticalMatchupEvidenceState.AVAILABLE
    assert sum(value for _, value in result.state_counts) == 3


def test_global_status_partial_and_not_available():
    observations = _p02_history() + (
        _observation(match_id="body-only-executor", player="Alice", opponent="Eve", sequence="5b28"),
    )
    partial = _build(observations)
    assert partial.candidate_category_count == 2
    assert partial.comparable_category_count == 1
    assert partial.evidence_state is TacticalMatchupEvidenceState.PARTIALLY_AVAILABLE
    unavailable = _build(_p02_history(executor=True, opponent_allowed=False))
    assert unavailable.comparable_category_count == 0
    assert unavailable.evidence_state is TacticalMatchupEvidenceState.NOT_AVAILABLE


@pytest.mark.parametrize(
    "policy,patterns,expected_catalog,redundant",
    [
        (TacticalEncodingPolicy.COMPONENT_ONLY, ("P02", "P04", "P05", "P06"), 26, False),
        (TacticalEncodingPolicy.PROFILE_ONLY, ("P02", "P09"), 156, False),
        (TacticalEncodingPolicy.COMPONENTS_AND_PROFILE, ("P02", "P04", "P05", "P06", "P09"), 179, True),
    ],
)
def test_all_policies_derive_full_catalog_from_schema(policy, patterns, expected_catalog, redundant):
    query = _query(policy=policy, requested_patterns=patterns)
    result = _build((), query)
    assert result.catalog_category_count == expected_catalog
    assert tuple(item.feature_name for item in result.categories) == tuple(
        name for name in _schema(policy).tactical_feature_names
        if name.split(".", 1)[0].upper() in patterns
    )
    assert result.redundant_representation is redundant
    assert not hasattr(result, "combined_tactical_total")


def test_components_and_profile_remain_separate_blocks_without_combined_total():
    policy = TacticalEncodingPolicy.COMPONENTS_AND_PROFILE
    query = _query(
        policy=policy,
        requested_patterns=("P04", "P05", "P06", "P09"),
    )
    result = _build(_return_history(policy=policy), query)
    active = {
        item.pattern_id: item.executor_evidence.category_activations
        for item in result.categories
        if item.executor_evidence.category_activations
    }
    assert active == {"P04": 1, "P05": 1, "P06": 1, "P09": 1}
    assert result.redundant_representation is True


def test_global_only_scope_is_fixed_and_never_selected_by_outcome():
    won = _build(_p02_history(label=TacticalOutcomeLabel.SERVER_WON_POINT))
    lost = _build(_p02_history(label=TacticalOutcomeLabel.RETURNER_WON_POINT))
    for result in (won, lost):
        for pair in result.categories:
            assert pair.executor_evidence.scope_used is TacticalEvidenceScope.GLOBAL
            assert pair.opponent_allowed_evidence.scope_used is TacticalEvidenceScope.GLOBAL
    assert won.query.scope_strategy is lost.query.scope_strategy is TacticalEvidenceScopeStrategy.GLOBAL_ONLY


def test_category_validator_rejects_counts_rates_matches_state_and_metadata_mutations():
    evidence = _pair(_build(_p02_history()), "p02.first_serve_direction.4").executor_evidence
    mutations = (
        ("contract_version", "0"),
        ("query", _query(opponent="Eve")),
        ("query_fingerprint", "A" * 64),
        ("policy", TacticalEncodingPolicy.PROFILE_ONLY),
        ("schema_fingerprint", "A" * 64),
        ("pattern_id", "P04"),
        ("feature_name", "p02.first_serve_direction.5"),
        ("category", "5"),
        ("scope_used", "global"),
        ("executor_role", TacticalPlayerRole.RETURNER),
        ("serve_numbers", (True,)),
        ("relevant_attempt_count", 0),
        ("applicable_attempts", 0),
        ("observed_pattern_attempts", 0),
        ("category_activations", 0),
        ("labeled_activations", 0),
        ("tactical_successes", 0),
        ("tactical_failures", 1),
        ("tactical_success_rate", True),
        ("wilson_lower", 0),
        ("wilson_upper", float("nan")),
        ("distinct_match_count", 0),
        ("evidence_state", TacticalCategoryEvidenceState.NOT_AVAILABLE),
        ("reason_codes", ("wrong",)),
        ("provenance", (("source", "wrong"),)),
        ("reconciliations", (("counts_ordered", 1),)),
    )
    for field, value in mutations:
        changed = object.__new__(TacticalCategoryEvidence)
        for item in fields(TacticalCategoryEvidence):
            object.__setattr__(changed, item.name, getattr(evidence, item.name))
        object.__setattr__(changed, field, value)
        with pytest.raises((TypeError, TacticalMatchupContractError)):
            validate_tactical_category_evidence(changed)


def test_pair_validator_rejects_identity_perspective_state_and_reconciliation_mutations():
    pair = _pair(_build(_p02_history()), "p02.first_serve_direction.4")
    mutations = (
        ("pattern_id", "P04"),
        ("feature_name", "p02.first_serve_direction.5"),
        ("category", "5"),
        ("query_fingerprint", "A" * 64),
        ("matchup_state", TacticalMatchupCategoryState.BOTH_INSUFFICIENT),
        ("reason_codes", ("wrong",)),
        ("reconciliations", (("category_identity_reconciled", 1),)),
        ("executor_evidence", pair.opponent_allowed_evidence),
        ("opponent_allowed_evidence", pair.executor_evidence),
    )
    for field, value in mutations:
        changed = object.__new__(TacticalMatchupCategoryEvidence)
        for item in fields(TacticalMatchupCategoryEvidence):
            object.__setattr__(changed, item.name, getattr(pair, item.name))
        object.__setattr__(changed, field, value)
        with pytest.raises((TypeError, TacticalMatchupContractError)):
            validate_tactical_matchup_category_evidence(changed)


def test_result_validator_rejects_counts_order_duplicates_state_and_contract_mutations():
    result = _build(_p02_history())
    mutations = (
        ("contract_version", "0"),
        ("policy", TacticalEncodingPolicy.PROFILE_ONLY),
        ("schema_fingerprint", "A" * 64),
        ("categories", tuple(reversed(result.categories))),
        ("categories", (result.categories[0],) + result.categories[:-1]),
        ("catalog_category_count", 4),
        ("candidate_category_count", 2),
        ("comparable_category_count", 0),
        ("insufficient_category_count", 1),
        ("evidence_state", TacticalMatchupEvidenceState.NOT_AVAILABLE),
        ("reason_codes", ("wrong",)),
        ("unique_attempt_count", 99),
        ("excluded_date_count", 1),
        ("state_counts", (("comparable", True),)),
        ("redundant_representation", True),
        ("reconciliations", (("input_counts_exhaustive", 1),)),
        ("provenance", (("source", "wrong"),)),
    )
    for field, value in mutations:
        changed = object.__new__(TacticalMatchupEvidence)
        for item in fields(TacticalMatchupEvidence):
            object.__setattr__(changed, item.name, getattr(result, item.name))
        object.__setattr__(changed, field, value)
        with pytest.raises((TypeError, TacticalMatchupContractError)):
            validate_tactical_matchup_evidence(changed)


def test_dataclasses_and_nested_collections_are_immutable():
    result = _build(_p02_history())
    with pytest.raises(FrozenInstanceError):
        result.evidence_state = TacticalMatchupEvidenceState.NOT_AVAILABLE
    with pytest.raises(FrozenInstanceError):
        result.query.player = "Mallory"
    assert type(result.categories) is tuple
    assert type(result.state_counts) is tuple
    assert all(type(item.reason_codes) is tuple for item in result.categories)


def test_canonical_serialization_and_four_fingerprint_domains_are_exact():
    query = _query()
    result = _build(_p02_history(), query)
    pair = _pair(result, "p02.first_serve_direction.4")
    category = pair.executor_evidence
    payloads = (
        (canonical_tactical_matchup_query_json(query), MATCHUP_QUERY_FINGERPRINT_DOMAIN, tactical_matchup_query_fingerprint(query)),
        (canonical_tactical_category_evidence_json(category), CATEGORY_EVIDENCE_FINGERPRINT_DOMAIN, tactical_category_evidence_fingerprint(category)),
        (canonical_tactical_matchup_category_evidence_json(pair), MATCHUP_CATEGORY_FINGERPRINT_DOMAIN, tactical_matchup_category_evidence_fingerprint(pair)),
        (canonical_tactical_matchup_evidence_json(result), MATCHUP_EVIDENCE_FINGERPRINT_DOMAIN, tactical_matchup_evidence_fingerprint(result)),
    )
    assert len({fingerprint for _, _, fingerprint in payloads}) == 4
    for payload, domain, fingerprint in payloads:
        assert fingerprint == sha256(domain + payload).hexdigest().upper()
        assert json.dumps(json.loads(payload), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") == payload


def test_fingerprints_change_with_query_and_evidence_contract():
    baseline = _build(_p02_history())
    variants = (
        _build(_p02_history(), _query(player="Carol")),
        _build(_p02_history(), _query(opponent="Dora")),
        _build(_p02_history(), _query(as_of_date=date(2020, 3, 1))),
        _build(_p02_history(), _query(window_days=20)),
        _build(_p02_history(), _query(minimum_labeled_attempts=2)),
        _build(_p02_history(), _query(minimum_matches=2)),
    )
    fingerprints = {tactical_matchup_evidence_fingerprint(baseline)}
    fingerprints.update(tactical_matchup_evidence_fingerprint(item) for item in variants)
    assert len(fingerprints) == 1 + len(variants)


def test_public_json_exposes_only_declared_players_and_aggregates():
    observations = (
        _observation(match_id="secret-one", point_number=77, player="Alice", opponent="HiddenOpponent"),
        _observation(match_id="secret-two", point_number=88, player="HiddenExecutor", opponent="Bob"),
    )
    text = canonical_tactical_matchup_evidence_json(_build(observations)).decode("utf-8")
    assert '"player":"Alice"' in text and '"opponent":"Bob"' in text
    for forbidden in ("secret-one", "secret-two", "HiddenOpponent", "HiddenExecutor", "match_id", "point_number", "effective_date", "4f17"):
        assert forbidden not in text


def test_changing_internal_attempt_ids_does_not_change_public_result():
    left = _build(_p02_history())
    renamed = tuple(
        _observation(
            match_id=f"renamed-{index}",
            point_number=99 + index,
            player=item.player,
            opponent=item.opponent,
            role=item.role,
            label=item.label,
        )
        for index, item in enumerate(_p02_history())
    )
    right = _build(renamed)
    assert canonical_tactical_matchup_evidence_json(left) == canonical_tactical_matchup_evidence_json(right)


def test_query_fingerprint_rejects_paths_and_public_json_rejects_nonfinite():
    query = _query(player="C:\\private")
    with pytest.raises(TacticalMatchupContractError, match="operativo"):
        canonical_tactical_matchup_query_json(query)
    result = _build(_p02_history())
    evidence = result.categories[0].executor_evidence
    object.__setattr__(evidence, "tactical_success_rate", float("inf"))
    with pytest.raises(TacticalMatchupContractError):
        canonical_tactical_matchup_evidence_json(result)


def test_schema_mismatch_and_mixed_observation_policies_fail_closed():
    query = _query(policy=TacticalEncodingPolicy.PROFILE_ONLY, requested_patterns=("P02",))
    with pytest.raises(TacticalMatchupContractError, match="Query y schema"):
        build_tactical_matchup_evidence((), query, _schema())
    with pytest.raises(TacticalMatchupContractError, match="observaciones"):
        _build((_observation(),), query)


def test_module_architecture_has_no_io_models_or_productive_asserts():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"pandas", "numpy", "sklearn"})
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called.isdisjoint({"open", "read_csv", "read_parquet", "to_csv", "to_parquet", "parse_sequence"})
    assert not any(name.startswith(("classify_", "adapt_", "extract_")) for name in called)
    assert "if __name__" not in source
    assert "data/" not in source and "reports/" not in source
    assert not any(isinstance(node, (ast.With, ast.AsyncWith)) for node in ast.walk(tree))


def test_module_has_no_mutable_global_collections():
    mutable = []
    for name, value in vars(matchup).items():
        if name.startswith("__") or callable(value) or isinstance(value, type):
            continue
        if isinstance(value, (list, dict, set, bytearray)):
            mutable.append(name)
    assert mutable == []
