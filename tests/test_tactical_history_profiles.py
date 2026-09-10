"""Pruebas sinteticas de perfiles tacticos historicos leakage-safe."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path

import pytest

import src.recommender.tactical_history_profiles as history
from src.recommender.tactical_feature_encoder import (
    FEATURE_CONTRACT_VERSION,
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
    encode_tactical_attempt,
    feature_schema_fingerprint,
)
from src.recommender.tactical_history_profiles import (
    HISTORY_CONTRACT_VERSION,
    NUMERIC_TOLERANCE,
    OBSERVATION_FINGERPRINT_DOMAIN,
    OBSERVATION_PROVENANCE,
    PROFILE_BATCH_FINGERPRINT_DOMAIN,
    PROFILE_FINGERPRINT_DOMAIN,
    QUERY_FINGERPRINT_DOMAIN,
    TacticalHistoricalObservation,
    TacticalHistoryContractError,
    TacticalLabelAvailability,
    TacticalOutcomeLabel,
    TacticalPlayerRole,
    TacticalProfileAvailability,
    TacticalProfileBatch,
    TacticalProfileQuery,
    build_tactical_player_profile,
    build_tactical_profile_batch,
    canonical_historical_observation_json,
    canonical_player_profile_json,
    canonical_profile_batch_json,
    canonical_profile_query_json,
    historical_observation_fingerprint,
    profile_query_fingerprint,
    tactical_player_profile_fingerprint,
    tactical_profile_batch_fingerprint,
    validate_development_queries,
    validate_tactical_feature_aggregate,
    validate_tactical_historical_observation,
    validate_tactical_player_profile,
    validate_tactical_profile_batch,
    validate_tactical_profile_query,
    wilson_interval,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
)


MODULE_PATH = Path(history.__file__)
POLICIES = tuple(TacticalEncodingPolicy)


def _schema(policy: TacticalEncodingPolicy = TacticalEncodingPolicy.COMPONENT_ONLY):
    return build_tactical_feature_schema(policy)


def _vector(
    sequence: str = "4f17",
    serve_number: int = 1,
    previous_fault: bool = False,
    policy: TacticalEncodingPolicy = TacticalEncodingPolicy.COMPONENT_ONLY,
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
    match_id: str = "match-001",
    point_number: int = 1,
    effective_date: date = date(2020, 1, 1),
    player: str = "Alice",
    opponent: str = "Bob",
    role: TacticalPlayerRole = TacticalPlayerRole.SERVER,
    sequence: str = "4f17",
    serve_number: int = 1,
    previous_fault: bool = False,
    policy: TacticalEncodingPolicy = TacticalEncodingPolicy.COMPONENT_ONLY,
    label: TacticalOutcomeLabel | None = TacticalOutcomeLabel.SERVER_WON_POINT,
):
    availability = (
        TacticalLabelAvailability.AVAILABLE
        if label is not None
        else TacticalLabelAvailability.NOT_AVAILABLE
    )
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
        availability,
        OBSERVATION_PROVENANCE,
    )


def _query(
    *,
    player: str = "Alice",
    role: TacticalPlayerRole = TacticalPlayerRole.SERVER,
    as_of_date: date = date(2020, 2, 1),
    policy: TacticalEncodingPolicy = TacticalEncodingPolicy.COMPONENT_ONLY,
    serve_numbers: tuple[int, ...] = (1, 2),
    opponent_filter: str | None = None,
    window_days: int | None = None,
    minimum_history: int = 1,
):
    schema = _schema(policy)
    return TacticalProfileQuery(
        HISTORY_CONTRACT_VERSION,
        player,
        role,
        as_of_date,
        policy,
        feature_schema_fingerprint(schema),
        serve_numbers,
        opponent_filter,
        window_days,
        minimum_history,
    )


def _profile(observations, query=None, *, development_end_date=None):
    if query is None:
        query = _query()
    return build_tactical_player_profile(
        tuple(observations),
        query,
        _schema(query.policy),
        development_end_date=development_end_date,
    )


def _aggregate(profile, feature_name):
    return next(item for item in profile.aggregates if item.feature_name == feature_name)


def _manual_wilson(successes: int, trials: int):
    if trials == 0:
        return None, None, None
    z = 1.959963984540054
    rate = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    centre = (rate + z2 / (2 * trials)) / denominator
    margin = z * ((rate * (1 - rate) / trials + z2 / (4 * trials**2)) ** 0.5) / denominator
    lower = 0.0 if successes == 0 else centre - margin
    upper = 1.0 if successes == trials else centre + margin
    return rate, lower, upper


def test_public_dataclasses_have_exact_fields_and_are_frozen():
    assert tuple(field.name for field in fields(TacticalHistoricalObservation)) == (
        "contract_version", "match_id", "point_number", "serve_number",
        "effective_date", "player", "opponent", "role", "feature_vector",
        "label", "label_availability", "provenance",
    )
    assert tuple(field.name for field in fields(TacticalProfileQuery)) == (
        "contract_version", "player", "role", "as_of_date", "policy",
        "schema_fingerprint", "serve_numbers", "opponent_filter", "window_days",
        "minimum_history",
    )
    observation = _observation()
    with pytest.raises(FrozenInstanceError):
        observation.player = "Mallory"
    with pytest.raises(FrozenInstanceError):
        _query().window_days = 7


@pytest.mark.parametrize("serve_number,previous_fault", [(1, False), (2, True)])
@pytest.mark.parametrize("role", tuple(TacticalPlayerRole))
@pytest.mark.parametrize("label", [None, *tuple(TacticalOutcomeLabel)])
def test_valid_observations_cover_serves_roles_and_label_availability(
    serve_number, previous_fault, role, label
):
    observation = _observation(
        serve_number=serve_number,
        previous_fault=previous_fault,
        role=role,
        label=label,
    )
    validate_tactical_historical_observation(observation)
    assert observation.feature_vector.serve_number == serve_number
    assert observation.label_availability is (
        TacticalLabelAvailability.NOT_AVAILABLE
        if label is None
        else TacticalLabelAvailability.AVAILABLE
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("contract_version", "0", "Version"),
        ("match_id", "", "match_id"),
        ("match_id", "unsafe/id", "opaco"),
        ("point_number", True, "point_number"),
        ("point_number", 1.0, "point_number"),
        ("point_number", -1, "point_number"),
        ("serve_number", True, "serve_number"),
        ("serve_number", 1.0, "serve_number"),
        ("serve_number", "1", "serve_number"),
        ("effective_date", datetime(2020, 1, 1), "date"),
        ("effective_date", "2020-01-01", "date"),
        ("player", "", "player"),
        ("player", " Alice", "player"),
        ("opponent", "Bob ", "opponent"),
        ("role", "server", "role"),
        ("provenance", list(OBSERVATION_PROVENANCE), "procedencia"),
    ],
)
def test_observation_rejects_invalid_exact_contract(field, value, error):
    observation = _observation()
    object.__setattr__(observation, field, value)
    with pytest.raises((TypeError, TacticalHistoryContractError), match=error):
        validate_tactical_historical_observation(observation)


def test_observation_rejects_equal_players_and_incoherent_labels():
    observation = _observation()
    object.__setattr__(observation, "opponent", "Alice")
    with pytest.raises(TacticalHistoryContractError, match="distintos"):
        validate_tactical_historical_observation(observation)
    observation = _observation()
    object.__setattr__(observation, "label_availability", TacticalLabelAvailability.NOT_AVAILABLE)
    with pytest.raises(TacticalHistoryContractError, match="None"):
        validate_tactical_historical_observation(observation)
    observation = _observation()
    object.__setattr__(observation, "label_availability", "available")
    with pytest.raises(TypeError, match="label_availability"):
        validate_tactical_historical_observation(observation)
    observation = _observation()
    object.__setattr__(observation, "label", "server_won_point")
    with pytest.raises(TacticalHistoryContractError, match="TacticalOutcomeLabel"):
        validate_tactical_historical_observation(observation)
    observation = _observation(label=None)
    object.__setattr__(observation, "label", TacticalOutcomeLabel.SERVER_WON_POINT)
    with pytest.raises(TacticalHistoryContractError, match="None"):
        validate_tactical_historical_observation(observation)


def test_observation_rejects_manipulated_vector_and_serve_mismatch():
    observation = _observation()
    vector = observation.feature_vector
    object.__setattr__(vector, "values", tuple(0 for _ in vector.values))
    with pytest.raises(Exception):
        validate_tactical_historical_observation(observation)
    observation = _observation()
    object.__setattr__(observation, "serve_number", 2)
    with pytest.raises(TacticalHistoryContractError, match="no coincide"):
        validate_tactical_historical_observation(observation)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("contract_version", "0", "Version"),
        ("player", "", "player"),
        ("role", "server", "role"),
        ("as_of_date", datetime(2020, 1, 1), "date"),
        ("as_of_date", "2020-01-01", "date"),
        ("policy", "component_only", "policy"),
        ("schema_fingerprint", "A" * 64, "fingerprint"),
        ("serve_numbers", [1, 2], "serve_numbers"),
        ("serve_numbers", (2, 1), "serve_numbers"),
        ("serve_numbers", (True,), "serve_numbers"),
        ("opponent_filter", " ", "opponent_filter"),
        ("window_days", True, "window_days"),
        ("window_days", 0, "window_days"),
        ("window_days", 1.0, "window_days"),
        ("minimum_history", True, "minimum_history"),
        ("minimum_history", 0, "minimum_history"),
    ],
)
def test_query_rejects_invalid_exact_contract(field, value, error):
    query = _query()
    object.__setattr__(query, field, value)
    with pytest.raises((TypeError, TacticalHistoryContractError), match=error):
        validate_tactical_profile_query(query)


def test_query_rejects_player_as_opponent_and_calendar_overflow():
    with pytest.raises(TacticalHistoryContractError, match="distinto"):
        _query(opponent_filter="Alice")
    with pytest.raises(TacticalHistoryContractError, match="calendario"):
        _query(as_of_date=date.min, window_days=1)


def test_duplicate_observation_is_rejected_without_identifier_disclosure():
    observation = _observation(match_id="secret-opaque-id")
    with pytest.raises(TacticalHistoryContractError) as captured:
        _profile((observation, observation))
    assert "duplicada" in str(captured.value)
    assert "secret-opaque-id" not in str(captured.value)


def test_mutable_observation_collection_is_rejected():
    with pytest.raises(TypeError, match="tuple"):
        build_tactical_player_profile([_observation()], _query(), _schema())


def test_all_observations_are_validated_before_filtering():
    irrelevant = _observation(player="Carol", opponent="Dan")
    object.__setattr__(irrelevant, "point_number", True)
    with pytest.raises(TacticalHistoryContractError, match="point_number"):
        _profile((irrelevant,))


def test_schema_and_policy_must_match_query_and_every_observation():
    query = _query(policy=TacticalEncodingPolicy.PROFILE_ONLY)
    with pytest.raises(TacticalHistoryContractError, match="Query y schema"):
        build_tactical_player_profile((), query, _schema())
    with pytest.raises(TacticalHistoryContractError, match="observaciones"):
        build_tactical_player_profile((_observation(),), query, _schema(query.policy))


def test_strict_temporal_cut_and_counters_are_exhaustive():
    observations = (
        _observation(match_id="m-before", effective_date=date(2020, 1, 30)),
        _observation(match_id="m-equal", effective_date=date(2020, 2, 1)),
        _observation(match_id="m-after", effective_date=date(2020, 2, 2)),
    )
    profile = _profile(observations)
    assert profile.total_observations_received == 3
    assert profile.temporally_eligible_count == 1
    assert profile.profile_eligible_count == 1
    assert profile.excluded_future_or_cutoff_count == 2
    assert profile.excluded_before_window_count == 0


def test_window_lower_bound_is_included_and_previous_day_excluded():
    query = _query(as_of_date=date(2020, 2, 1), window_days=10)
    observations = (
        _observation(match_id="m-lower", effective_date=date(2020, 1, 22)),
        _observation(match_id="m-too-old", effective_date=date(2020, 1, 21)),
        _observation(match_id="m-cut", effective_date=date(2020, 2, 1)),
    )
    profile = _profile(observations, query)
    assert profile.profile_eligible_count == 1
    assert profile.excluded_before_window_count == 1
    assert profile.excluded_future_or_cutoff_count == 1


def test_temporal_result_is_independent_of_input_order_and_same_day_ties():
    observations = (
        _observation(match_id="m-z", point_number=2, effective_date=date(2020, 1, 1), sequence="5b28"),
        _observation(match_id="m-a", point_number=1, effective_date=date(2020, 1, 1), sequence="4f17"),
        _observation(match_id="m-b", point_number=3, effective_date=date(2020, 1, 2), sequence="6r39"),
    )
    left = _profile(observations)
    right = _profile(tuple(reversed(observations)))
    assert left == right
    assert canonical_player_profile_json(left) == canonical_player_profile_json(right)
    assert tactical_player_profile_fingerprint(left) == tactical_player_profile_fingerprint(right)


def test_future_observations_never_change_any_aggregate_or_outcome():
    past = _observation(match_id="m-past", effective_date=date(2020, 1, 1))
    future = _observation(
        match_id="m-future",
        effective_date=date(2021, 1, 1),
        sequence="5b28",
        label=TacticalOutcomeLabel.RETURNER_WON_POINT,
    )
    base = _profile((past,))
    with_future = _profile((past, future))
    assert base.aggregates == with_future.aggregates
    assert base.labeled_observation_count == with_future.labeled_observation_count == 1
    assert with_future.excluded_future_or_cutoff_count == 1


def test_only_fully_eligible_observations_reach_the_accumulator(monkeypatch):
    query = _query(serve_numbers=(1,), opponent_filter="Bob")
    observations = (
        _observation(match_id="m-ok", effective_date=date(2020, 1, 1)),
        _observation(match_id="m-cut", effective_date=query.as_of_date),
        _observation(match_id="m-player", player="Carol", opponent="Bob"),
        _observation(match_id="m-role", role=TacticalPlayerRole.RETURNER),
        _observation(match_id="m-serve", serve_number=2, previous_fault=True),
        _observation(match_id="m-rival", opponent="Dan"),
    )
    original = history._make_aggregate
    received = []

    def guarded(feature_name, schema, eligible):
        received.append(eligible)
        assert all(item.effective_date < query.as_of_date for item in eligible)
        assert all(item.player == "Alice" for item in eligible)
        assert all(item.role is TacticalPlayerRole.SERVER for item in eligible)
        assert all(item.serve_number == 1 and item.opponent == "Bob" for item in eligible)
        return original(feature_name, schema, eligible)

    monkeypatch.setattr(history, "_make_aggregate", guarded)
    profile = _profile(observations, query)
    assert profile.profile_eligible_count == 1
    assert len(received) == _schema().feature_count
    assert all(tuple(item.match_id for item in batch) == ("m-ok",) for batch in received)


def test_development_seal_excludes_later_observations_before_aggregation():
    query = _query(as_of_date=date(2020, 1, 5))
    observations = (
        _observation(match_id="m-dev", effective_date=date(2020, 1, 4)),
        _observation(match_id="m-sealed", effective_date=date(2020, 1, 6), sequence="5b28"),
    )
    profile = _profile(observations, query, development_end_date=date(2020, 1, 5))
    assert profile.excluded_sealed_period_count == 1
    assert profile.profile_eligible_count == 1
    assert _aggregate(profile, "p02.first_serve_direction.5").active_count == 0


def test_development_query_boundary_is_inclusive_but_later_query_rejected():
    boundary = date(2023, 12, 31)
    validate_development_queries((_query(as_of_date=boundary),), boundary)
    with pytest.raises(TacticalHistoryContractError, match="excede"):
        validate_development_queries((_query(as_of_date=date(2024, 1, 1)),), boundary)
    with pytest.raises(TacticalHistoryContractError, match="no vacia"):
        validate_development_queries((), boundary)


def test_filter_counters_separate_player_role_serve_and_opponent():
    observations = (
        _observation(match_id="m-ok", opponent="Bob"),
        _observation(match_id="m-player", player="Carol", opponent="Bob"),
        _observation(match_id="m-role", role=TacticalPlayerRole.RETURNER),
        _observation(match_id="m-serve", serve_number=2, previous_fault=True),
        _observation(match_id="m-opponent", opponent="Dan"),
    )
    query = _query(serve_numbers=(1,), opponent_filter="Bob")
    profile = _profile(observations, query)
    assert profile.temporally_eligible_count == 5
    assert profile.excluded_player_count == 1
    assert profile.excluded_role_count == 1
    assert profile.excluded_context_filter_count == 2
    assert profile.profile_eligible_count == 1


def test_player_occurring_only_as_opponent_never_enters_profile():
    profile = _profile((_observation(player="Carol", opponent="Alice"),))
    assert profile.profile_eligible_count == 0
    assert profile.excluded_player_count == 1


def test_server_profile_uses_only_p02_and_never_return_patterns():
    profile = _profile((_observation(sequence="4f17"),))
    assert _aggregate(profile, "p02.first_serve_direction.4").active_count == 1
    for name in ("p04.return_direction.1", "p05.return_depth.7", "p06.return_shot_type.f"):
        aggregate = _aggregate(profile, name)
        assert (aggregate.active_count, aggregate.applicable_count, aggregate.observed_count) == (0, 0, 0)
    for pattern in ("p04", "p05", "p06", "p09"):
        assert _aggregate(profile, f"mask.{pattern}.observed").active_count == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_returner_profile_excludes_p02_and_uses_only_policy_authorized_return_features(policy):
    query = _query(role=TacticalPlayerRole.RETURNER, policy=policy)
    profile = _profile((_observation(role=TacticalPlayerRole.RETURNER, policy=policy),), query)
    assert _aggregate(profile, "mask.p02.applicable").active_count == 0
    assert _aggregate(profile, "mask.p02.observed").active_count == 0
    if "p04.return_direction.1" in {item.feature_name for item in profile.aggregates}:
        assert _aggregate(profile, "p04.return_direction.1").active_count == 1
    if "p09.return_profile.f|1|7" in {item.feature_name for item in profile.aggregates}:
        assert _aggregate(profile, "p09.return_profile.f|1|7").active_count == 1


def test_mixed_roles_do_not_contaminate_denominators():
    observations = (
        _observation(match_id="m-server", role=TacticalPlayerRole.SERVER),
        _observation(match_id="m-return", point_number=2, role=TacticalPlayerRole.RETURNER),
    )
    server = _profile(observations, _query(role=TacticalPlayerRole.SERVER))
    returner = _profile(observations, _query(role=TacticalPlayerRole.RETURNER))
    assert server.profile_eligible_count == returner.profile_eligible_count == 1
    assert server.excluded_role_count == returner.excluded_role_count == 1
    assert _aggregate(server, "p02.first_serve_direction.4").applicable_count == 1
    assert _aggregate(returner, "p04.return_direction.1").applicable_count == 1


def test_manual_server_aggregates_labels_rates_and_denominators():
    observations = (
        _observation(match_id="m1", sequence="4f17", label=TacticalOutcomeLabel.SERVER_WON_POINT),
        _observation(match_id="m2", sequence="4f17", label=TacticalOutcomeLabel.RETURNER_WON_POINT),
        _observation(match_id="m3", sequence="5b28", label=None),
        _observation(match_id="m4", sequence="6", label=TacticalOutcomeLabel.SERVER_WON_POINT),
        _observation(match_id="m5", serve_number=2, previous_fault=True, label=TacticalOutcomeLabel.SERVER_WON_POINT),
    )
    profile = _profile(observations)
    wide = _aggregate(profile, "p02.first_serve_direction.4")
    body = _aggregate(profile, "p02.first_serve_direction.5")
    tee = _aggregate(profile, "p02.first_serve_direction.6")
    assert profile.profile_eligible_count == 5
    assert profile.first_serve_attempts == 4
    assert profile.second_serve_attempts == 1
    assert profile.labeled_observation_count == 4
    assert profile.unlabeled_observation_count == 1
    assert (wide.active_count, wide.applicable_count, wide.observed_count) == (2, 4, 4)
    assert wide.frequency_over_applicable == wide.frequency_over_observed == 0.5
    assert (wide.labeled_active_count, wide.actor_successes, wide.actor_success_rate) == (2, 1, 0.5)
    assert (body.active_count, body.labeled_active_count, body.outcome_unavailable_reason) == (1, 0, "no_labeled_activations")
    assert (tee.active_count, tee.actor_successes) == (1, 1)
    assert _aggregate(profile, "mask.p02.applicable").active_count == 4
    assert _aggregate(profile, "mask.p02.observed").active_count == 4


def test_returner_success_semantics_are_role_specific():
    observations = (
        _observation(match_id="m1", role=TacticalPlayerRole.RETURNER, label=TacticalOutcomeLabel.RETURNER_WON_POINT),
        _observation(match_id="m2", role=TacticalPlayerRole.RETURNER, label=TacticalOutcomeLabel.SERVER_WON_POINT),
    )
    profile = _profile(observations, _query(role=TacticalPlayerRole.RETURNER))
    aggregate = _aggregate(profile, "p04.return_direction.1")
    assert aggregate.labeled_active_count == 2
    assert aggregate.actor_successes == 1
    assert aggregate.actor_success_rate == 0.5


def test_label_changes_outcome_only_and_never_feature_eligibility_or_activation():
    won = _profile((_observation(label=TacticalOutcomeLabel.SERVER_WON_POINT),))
    lost = _profile((_observation(label=TacticalOutcomeLabel.RETURNER_WON_POINT),))
    unlabeled = _profile((_observation(label=None),))
    won_counts = tuple(
        (item.feature_name, item.active_count, item.applicable_count, item.observed_count)
        for item in won.aggregates
    )
    assert won_counts == tuple(
        (item.feature_name, item.active_count, item.applicable_count, item.observed_count)
        for item in lost.aggregates
    )
    assert won_counts == tuple(
        (item.feature_name, item.active_count, item.applicable_count, item.observed_count)
        for item in unlabeled.aggregates
    )
    feature = "p02.first_serve_direction.4"
    assert _aggregate(won, feature).actor_successes == 1
    assert _aggregate(lost, feature).actor_successes == 0
    assert _aggregate(unlabeled, feature).labeled_active_count == 0


def test_unobserved_return_signals_count_as_applicable_not_observed():
    query = _query(role=TacticalPlayerRole.RETURNER)
    profile = _profile(
        (_observation(role=TacticalPlayerRole.RETURNER, sequence="6"),), query
    )
    for name in (
        "p04.return_direction.1",
        "p05.return_depth.7",
        "p06.return_shot_type.f",
    ):
        aggregate = _aggregate(profile, name)
        assert (aggregate.active_count, aggregate.applicable_count, aggregate.observed_count) == (0, 1, 0)
        assert aggregate.frequency_over_applicable == 0.0
        assert aggregate.frequency_over_observed is None


def test_p02_second_serve_is_not_applicable_and_has_no_denominator():
    profile = _profile(
        (_observation(serve_number=2, previous_fault=True, sequence="4f17"),)
    )
    for name in (
        "p02.first_serve_direction.4",
        "p02.first_serve_direction.5",
        "p02.first_serve_direction.6",
    ):
        aggregate = _aggregate(profile, name)
        assert (aggregate.active_count, aggregate.applicable_count, aggregate.observed_count) == (0, 0, 0)
        assert aggregate.frequency_over_applicable is None
    assert _aggregate(profile, "mask.p02.applicable").active_count == 0


def test_pattern_categories_exhaust_observed_masks_and_do_not_double_count():
    query = _query(role=TacticalPlayerRole.RETURNER)
    observations = tuple(
        _observation(match_id=f"m{i}", point_number=i, role=TacticalPlayerRole.RETURNER, sequence=sequence)
        for i, sequence in enumerate(("4f17", "5b28", "6r39", "6"), start=1)
    )
    profile = _profile(observations, query)
    p04 = tuple(item for item in profile.aggregates if item.feature_name.startswith("p04."))
    p05 = tuple(item for item in profile.aggregates if item.feature_name.startswith("p05."))
    p06 = tuple(item for item in profile.aggregates if item.feature_name.startswith("p06."))
    assert sum(item.active_count for item in p04) == _aggregate(profile, "mask.p04.observed").active_count == 3
    assert sum(item.active_count for item in p05) == _aggregate(profile, "mask.p05.observed").active_count == 3
    assert sum(item.active_count for item in p06) == _aggregate(profile, "mask.p06.observed").active_count == 3
    assert all(item.applicable_count == 4 for item in (*p04, *p05, *p06))


def test_context_aggregates_track_serve_and_previous_fault_separately():
    observations = (
        _observation(match_id="m1", serve_number=1),
        _observation(match_id="m2", serve_number=2, previous_fault=True),
        _observation(match_id="m3", serve_number=2, previous_fault=True),
    )
    profile = _profile(observations)
    assert _aggregate(profile, "context.serve_number.1").active_count == 1
    assert _aggregate(profile, "context.serve_number.2").active_count == 2
    assert _aggregate(profile, "context.previous_attempt_was_fault").active_count == 2
    for name in ("context.serve_number.1", "context.serve_number.2", "context.previous_attempt_was_fault"):
        item = _aggregate(profile, name)
        assert item.applicable_count == item.observed_count == 3


@pytest.mark.parametrize("successes,trials", [(0, 1), (0, 3), (0, 20), (1, 1), (3, 3), (20, 20), (1, 3), (2, 3)])
def test_wilson_matches_independent_calculation_and_exact_boundaries(successes, trials):
    actual = wilson_interval(successes, trials, feature_name="p02.first_serve_direction.4")
    expected = _manual_wilson(successes, trials)
    assert actual == pytest.approx(expected, abs=1e-15)
    rate, lower, upper = actual
    assert 0 <= lower <= rate <= upper <= 1
    if successes == 0:
        assert lower == 0.0
    if successes == trials:
        assert upper == 1.0


def test_wilson_zero_trials_has_explicit_absence_in_aggregate():
    assert wilson_interval(0, 0, feature_name="safe.feature") == (None, None, None)
    aggregate = _aggregate(_profile((), _query()), "p02.first_serve_direction.4")
    assert aggregate.labeled_active_count == 0
    assert aggregate.actor_success_rate is None
    assert aggregate.wilson_lower is aggregate.wilson_upper is None
    assert aggregate.outcome_unavailable_reason == "no_labeled_activations"


@pytest.mark.parametrize("successes,trials", [(1, 4), (2, 7), (9, 11)])
def test_wilson_is_symmetric_for_complementary_successes(successes, trials):
    rate, lower, upper = wilson_interval(successes, trials, feature_name="safe.feature")
    complement_rate, complement_lower, complement_upper = wilson_interval(
        trials - successes, trials, feature_name="safe.feature"
    )
    assert rate == pytest.approx(1 - complement_rate, abs=1e-15)
    assert lower == pytest.approx(1 - complement_upper, abs=1e-15)
    assert upper == pytest.approx(1 - complement_lower, abs=1e-15)


@pytest.mark.parametrize(
    "successes,trials,feature_name",
    [(-1, 1, "safe"), (2, 1, "safe"), (True, 1, "safe"), (1.0, 1, "safe"), ("1", 1, "safe"), (0, True, "safe"), (0, 1.0, "safe"), (0, 1, "unsafe/name")],
)
def test_wilson_rejects_invalid_domains(successes, trials, feature_name):
    with pytest.raises(TacticalHistoryContractError):
        wilson_interval(successes, trials, feature_name=feature_name)


@pytest.mark.parametrize("policy", POLICIES)
def test_all_encoder_policies_preserve_exact_schema_order_masks_and_redundancy(policy):
    query = _query(role=TacticalPlayerRole.RETURNER, policy=policy)
    profile = _profile((_observation(role=TacticalPlayerRole.RETURNER, policy=policy),), query)
    schema = _schema(policy)
    assert tuple(item.feature_name for item in profile.aggregates) == schema.feature_names
    assert profile.redundant_representation is schema.redundant_representation
    for pattern in ("p02", "p04", "p05", "p06", "p09"):
        assert f"mask.{pattern}.applicable" in schema.mask_feature_names
        assert f"mask.{pattern}.observed" in schema.mask_feature_names
    assert not hasattr(profile, "combined_tactical_total")


def test_components_and_profile_keeps_overlapping_blocks_separate():
    policy = TacticalEncodingPolicy.COMPONENTS_AND_PROFILE
    query = _query(role=TacticalPlayerRole.RETURNER, policy=policy)
    profile = _profile((_observation(role=TacticalPlayerRole.RETURNER, policy=policy),), query)
    assert profile.redundant_representation is True
    assert _aggregate(profile, "p04.return_direction.1").active_count == 1
    assert _aggregate(profile, "p05.return_depth.7").active_count == 1
    assert _aggregate(profile, "p06.return_shot_type.f").active_count == 1
    assert _aggregate(profile, "p09.return_profile.f|1|7").active_count == 1
    assert _aggregate(profile, "mask.p09.observed").active_count == 1


@pytest.mark.parametrize(
    "count,minimum,state,reasons",
    [
        (0, 1, TacticalProfileAvailability.NOT_AVAILABLE, ("no_eligible_history",)),
        (1, 2, TacticalProfileAvailability.INSUFFICIENT_HISTORY, ("minimum_history_not_met",)),
        (1, 1, TacticalProfileAvailability.AVAILABLE, ("minimum_history_met",)),
    ],
)
def test_profile_availability_states_are_exhaustive(count, minimum, state, reasons):
    observations = tuple(_observation(match_id=f"m{i}", point_number=i) for i in range(count))
    profile = _profile(observations, _query(minimum_history=minimum))
    assert profile.availability_state is state
    assert profile.reason_codes == reasons
    assert len(profile.aggregates) == _schema().feature_count


def test_validators_reject_representative_resigned_mutations():
    profile = _profile((_observation(),))
    aggregate = _aggregate(profile, "p02.first_serve_direction.4")
    mutations = (
        (aggregate, "active_count", 2, validate_tactical_feature_aggregate),
        (aggregate, "frequency_over_applicable", 0.25, validate_tactical_feature_aggregate),
        (aggregate, "wilson_lower", 0.5, validate_tactical_feature_aggregate),
        (profile, "availability_state", TacticalProfileAvailability.NOT_AVAILABLE, validate_tactical_player_profile),
        (profile, "reason_codes", ("no_eligible_history",), validate_tactical_player_profile),
        (profile, "profile_eligible_count", 2, validate_tactical_player_profile),
        (profile, "reconciliations", (("input_counts_exhaustive", False),), validate_tactical_player_profile),
        (profile, "provenance", (("source", "changed"),), validate_tactical_player_profile),
    )
    for obj, field, value, validator in mutations:
        object.__setattr__(obj, field, value)
        with pytest.raises(TacticalHistoryContractError):
            validator(obj)
        object.__setattr__(obj, field, getattr(_profile((_observation(),)) if obj is profile else _aggregate(_profile((_observation(),)), "p02.first_serve_direction.4"), field))


def test_standalone_aggregate_rejects_unknown_but_well_formed_feature_name():
    aggregate = _aggregate(_profile((_observation(),)), "p02.first_serve_direction.4")
    object.__setattr__(aggregate, "feature_name", "p99.safe.feature")
    object.__setattr__(aggregate, "pattern_id", None)
    with pytest.raises(TacticalHistoryContractError, match="catalogo"):
        validate_tactical_feature_aggregate(aggregate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frequency_over_applicable", True),
        ("frequency_over_observed", 1),
        ("actor_success_rate", True),
        ("wilson_lower", 0),
        ("wilson_upper", float("inf")),
    ],
)
def test_aggregate_optional_numbers_reject_bool_int_and_nonfinite(field, value):
    aggregate = _aggregate(_profile((_observation(),)), "p02.first_serve_direction.4")
    object.__setattr__(aggregate, field, value)
    with pytest.raises(TacticalHistoryContractError, match="float finito"):
        validate_tactical_feature_aggregate(aggregate)


def test_reconciliations_reject_integer_one_as_boolean_true():
    profile = _profile((_observation(),))
    manipulated = tuple((key, 1) for key, _ in profile.reconciliations)
    object.__setattr__(profile, "reconciliations", manipulated)
    with pytest.raises(TacticalHistoryContractError, match="inmutables"):
        validate_tactical_player_profile(profile)


def test_contract_versions_require_exact_strings():
    class StringLike(str):
        pass

    observation = _observation()
    object.__setattr__(observation, "contract_version", StringLike(HISTORY_CONTRACT_VERSION))
    with pytest.raises(TacticalHistoryContractError, match="Version"):
        validate_tactical_historical_observation(observation)
    query = _query()
    object.__setattr__(query, "contract_version", StringLike(HISTORY_CONTRACT_VERSION))
    with pytest.raises(TacticalHistoryContractError, match="Version"):
        validate_tactical_profile_query(query)


def test_profile_rejects_aggregate_order_and_duplicates():
    profile = _profile((_observation(),))
    original = profile.aggregates
    object.__setattr__(profile, "aggregates", tuple(reversed(original)))
    with pytest.raises(TacticalHistoryContractError, match="orden"):
        validate_tactical_player_profile(profile)
    object.__setattr__(profile, "aggregates", (original[0],) + original[:-1])
    with pytest.raises(TacticalHistoryContractError):
        validate_tactical_player_profile(profile)


def test_batch_one_or_many_queries_is_canonical_and_deterministic():
    observations = (
        _observation(match_id="m1", player="Alice", opponent="Bob"),
        _observation(match_id="m2", player="Carol", opponent="Dan"),
    )
    queries = (
        _query(player="Carol", as_of_date=date(2020, 3, 1)),
        _query(player="Alice", as_of_date=date(2020, 2, 1)),
        _query(player="Alice", as_of_date=date(2020, 3, 1)),
    )
    left = build_tactical_profile_batch(observations, queries, _schema())
    right = build_tactical_profile_batch(tuple(reversed(observations)), tuple(reversed(queries)), _schema())
    assert left == right
    assert left.profile_count == 3
    assert tuple((p.query.player, p.query.as_of_date) for p in left.profiles) == (
        ("Alice", date(2020, 2, 1)),
        ("Alice", date(2020, 3, 1)),
        ("Carol", date(2020, 3, 1)),
    )
    assert tactical_profile_batch_fingerprint(left) == tactical_profile_batch_fingerprint(right)
    single = build_tactical_profile_batch(observations, (queries[0],), _schema())
    assert single.profile_count == 1


def test_batch_supports_roles_dates_windows_and_development_seal():
    observations = (
        _observation(match_id="m1", role=TacticalPlayerRole.SERVER),
        _observation(match_id="m2", role=TacticalPlayerRole.RETURNER),
    )
    queries = (
        _query(role=TacticalPlayerRole.SERVER, window_days=20),
        _query(role=TacticalPlayerRole.RETURNER, as_of_date=date(2020, 3, 1)),
    )
    batch = build_tactical_profile_batch(observations, queries, _schema(), development_end_date=date(2020, 3, 1))
    validate_tactical_profile_batch(batch)
    assert {profile.query.role for profile in batch.profiles} == set(TacticalPlayerRole)


def test_batch_rejects_empty_mutable_duplicate_and_mixed_queries():
    with pytest.raises(TacticalHistoryContractError, match="no vacia"):
        build_tactical_profile_batch((), (), _schema())
    with pytest.raises(TacticalHistoryContractError, match="no vacia"):
        build_tactical_profile_batch((), [], _schema())
    query = _query()
    with pytest.raises(TacticalHistoryContractError, match="duplicada"):
        build_tactical_profile_batch((), (query, query), _schema())
    other = _query(policy=TacticalEncodingPolicy.PROFILE_ONLY)
    with pytest.raises(TacticalHistoryContractError, match="schema"):
        build_tactical_profile_batch((), (query, other), _schema())


def test_batch_validator_rejects_count_order_schema_and_reconciliation_mutations():
    queries = (_query(player="Alice"), _query(player="Carol"))
    batch = build_tactical_profile_batch((), queries, _schema())
    mutations = (
        ("profile_count", 3),
        ("profiles", tuple(reversed(batch.profiles))),
        ("schema_fingerprint", "A" * 64),
        ("reconciliations", (("profile_count_reconciled", False),)),
    )
    for field, value in mutations:
        changed = object.__new__(TacticalProfileBatch)
        for dataclass_field in fields(TacticalProfileBatch):
            object.__setattr__(changed, dataclass_field.name, getattr(batch, dataclass_field.name))
        object.__setattr__(changed, field, value)
        with pytest.raises(TacticalHistoryContractError):
            validate_tactical_profile_batch(changed)


def test_canonical_json_and_fingerprints_use_separate_domains():
    observation = _observation()
    query = _query()
    profile = _profile((observation,), query)
    batch = build_tactical_profile_batch((observation,), (query,), _schema())
    payloads = (
        (canonical_historical_observation_json(observation), OBSERVATION_FINGERPRINT_DOMAIN, historical_observation_fingerprint(observation)),
        (canonical_profile_query_json(query), QUERY_FINGERPRINT_DOMAIN, profile_query_fingerprint(query)),
        (canonical_player_profile_json(profile), PROFILE_FINGERPRINT_DOMAIN, tactical_player_profile_fingerprint(profile)),
        (canonical_profile_batch_json(batch), PROFILE_BATCH_FINGERPRINT_DOMAIN, tactical_profile_batch_fingerprint(batch)),
    )
    assert len({fingerprint for _, _, fingerprint in payloads}) == 4
    for payload, domain, fingerprint in payloads:
        assert fingerprint == sha256(domain + payload).hexdigest().upper()
        assert len(fingerprint) == 64 and fingerprint == fingerprint.upper()
        assert json.dumps(json.loads(payload), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") == payload


def test_public_serialization_contains_aggregates_but_no_individual_keys_or_dates():
    profile = _profile((_observation(match_id="secret-match", point_number=99),))
    payload = canonical_player_profile_json(profile)
    text = payload.decode("utf-8")
    structure = json.loads(payload)
    assert structure["profile_eligible_count"] == 1
    assert "secret-match" not in text
    assert '"match_id"' not in text
    assert '"point_number"' not in text
    assert '"effective_date"' not in text
    assert "4f17" not in text
    assert "outcomes" not in text
    assert not isinstance(structure.get("effective_dates"), list)


def test_internal_observation_serialization_is_explicitly_marked_and_never_nested():
    observation = _observation(match_id="opaque-1")
    internal = json.loads(canonical_historical_observation_json(observation))
    assert internal["visibility"] == "internal_only"
    assert internal["match_id"] == "opaque-1"
    public = canonical_player_profile_json(_profile((observation,))).decode("utf-8")
    assert "internal_only" not in public
    assert "opaque-1" not in public


def test_changing_query_contract_or_valid_aggregate_changes_fingerprint():
    observation = _observation()
    profile = _profile((observation,))
    later = _profile((observation,), _query(as_of_date=date(2020, 3, 1)))
    returner = _profile((_observation(role=TacticalPlayerRole.RETURNER),), _query(role=TacticalPlayerRole.RETURNER))
    assert tactical_player_profile_fingerprint(profile) != tactical_player_profile_fingerprint(later)
    assert tactical_player_profile_fingerprint(profile) != tactical_player_profile_fingerprint(returner)
    assert profile_query_fingerprint(_query(window_days=None)) != profile_query_fingerprint(_query(window_days=30))
    assert profile_query_fingerprint(_query(serve_numbers=(1,))) != profile_query_fingerprint(_query(serve_numbers=(2,)))
    assert profile_query_fingerprint(_query(opponent_filter=None)) != profile_query_fingerprint(_query(opponent_filter="Bob"))


def test_observation_ids_change_only_internal_fingerprint_not_public_profile():
    left_observation = _observation(match_id="opaque-a", point_number=1)
    right_observation = _observation(match_id="opaque-b", point_number=88)
    left = _profile((left_observation,))
    right = _profile((right_observation,))
    assert historical_observation_fingerprint(left_observation) != historical_observation_fingerprint(right_observation)
    assert canonical_player_profile_json(left) == canonical_player_profile_json(right)
    assert tactical_player_profile_fingerprint(left) == tactical_player_profile_fingerprint(right)


def test_profile_public_security_rejects_paths_and_nonfinite_values():
    profile = _profile((_observation(),))
    object.__setattr__(profile, "provenance", (("source", "C:\\secret\\file"),))
    with pytest.raises(TacticalHistoryContractError):
        canonical_player_profile_json(profile)
    aggregate = _aggregate(_profile((_observation(),)), "p02.first_serve_direction.4")
    object.__setattr__(aggregate, "frequency_over_applicable", float("nan"))
    with pytest.raises(TacticalHistoryContractError):
        validate_tactical_feature_aggregate(aggregate)


def test_module_has_no_forbidden_architecture_or_productive_asserts():
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
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint({"open", "parse_sequence", "read_csv", "read_parquet", "to_csv", "to_parquet"})
    assert not any(isinstance(node, (ast.With, ast.AsyncWith)) for node in ast.walk(tree))
    assert "if __name__" not in source
    assert "data/" not in source and "reports/" not in source
    assert not any(name.startswith(("classify_", "adapt_", "extract_")) for name in called_names)


def test_public_module_constants_are_deeply_immutable():
    mutable_globals = []
    for name, value in vars(history).items():
        if name.startswith("__") or callable(value) or isinstance(value, type):
            continue
        if isinstance(value, (list, dict, set, bytearray)):
            mutable_globals.append(name)
    assert mutable_globals == []


def test_no_existing_files_are_part_of_the_implementation_scope():
    assert MODULE_PATH.as_posix().endswith("src/recommender/tactical_history_profiles.py")
    assert HISTORY_CONTRACT_VERSION == "1.0.0"
    assert FEATURE_CONTRACT_VERSION == "1.0.0"
    assert NUMERIC_TOLERANCE == 1e-15
