"""Pruebas sinteticas del pipeline tactico leakage-safe."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import inspect
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import src.analysis.tactical_recommender_pipeline as pipeline
import src.recommender.tactical_matchup_evidence as matchup_evidence
from src.analysis.tactical_recommender_pipeline import (
    DEVELOPMENT_CUTOFF,
    DIAGNOSTICS_COLUMNS,
    EXPECTED_FOLD_MATCHES,
    EXPECTED_TARGET_ORIENTATIONS,
    EXPECTED_VALIDATION_MATCHES,
    FINGERPRINT_CONTRACT_VERSION,
    PIPELINE_CONTRACT_VERSION,
    POPULATION_COLUMNS,
    POPULATION_ROW_ORDER,
    PRODUCTION_PATTERNS,
    RANKINGS_COLUMNS,
    SOURCE_COLUMNS,
    TEST_ZERO_FIELDS,
    VALIDATION_FOLDS,
    AVAILABILITY_COLUMNS,
    TacticalPipelineConfig,
    TacticalPipelineArtifactPaths,
    TacticalPipelineContractError,
    TacticalPipelineExecutionError,
    TacticalPipelineState,
    TacticalPipelineTarget,
    build_tactical_historical_observations,
    build_not_available_publication,
    build_validation_targets,
    construct_tactical_attempt_records,
    default_artifact_paths,
    default_pipeline_config,
    performance_log_payload,
    prepare_tactical_pipeline_payloads,
    publish_tactical_pipeline_artifacts,
    run_tactical_recommender_pipeline,
    seal_development_points,
    serialize_tactical_pipeline_artifacts,
    serialize_tactical_pipeline_unavailable_artifacts,
    tactical_pipeline_result_fingerprint,
    validate_upstream_ancestry,
    validate_source_points,
    validate_tactical_attempt_batch,
    validate_tactical_pipeline_result,
    verify_persisted_artifacts,
    verify_persisted_artifacts_only,
)
from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
    encode_tactical_attempt,
)
from src.recommender.tactical_history_profiles import (
    TacticalOutcomeLabel,
    TacticalPlayerRole,
)
from src.recommender.tactical_matchup_evidence import (
    build_tactical_matchup_evidence,
)
from src.recommender.tactical_prioritization import (
    TacticalPrioritizationState,
    prioritize_tactical_matchup,
)
from src.recommender.tactical_signal_orchestrator import (
    extract_tactical_signals_for_attempt,
)


MODULE_PATH = Path(pipeline.__file__)


def _points() -> pd.DataFrame:
    rows = [
        ("m1", 1, "2020-01-01", "Hard", 1, 1, "Alice", "Dora", "4f17", None),
        ("m2", 1, "2020-01-02", "Hard", 1, 1, "Carol", "Bob", "4f17", None),
        ("m3", 1, "2020-01-03", "Clay", 1, 2, "Carol", "Alice", "4f17", "6b28"),
        ("m4", 1, "2020-01-04", "Grass", 2, 1, "Carol", "Bob", "4f17", None),
        ("m5", 1, "2023-12-31", "Hard", 1, 2, "Alice", "Dora", "4n", "6f17"),
        ("mt", 1, "2024-01-01", "Hard", 1, 1, "TestA", "TestB", "5b28", None),
    ]
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _config(
    *,
    policy: TacticalEncodingPolicy = TacticalEncodingPolicy.COMPONENT_ONLY,
    patterns: tuple[str, ...] | None = None,
    serves: tuple[int, ...] = (1, 2),
    window_days: int | None = None,
    minimum: int = 1,
    top_k: int = 3,
    redundant: bool = False,
) -> TacticalPipelineConfig:
    if patterns is None:
        patterns = (
            ("P02", "P09")
            if policy is TacticalEncodingPolicy.PROFILE_ONLY
            else ("P02", "P04", "P05", "P06")
        )
    return TacticalPipelineConfig(
        PIPELINE_CONTRACT_VERSION,
        policy,
        patterns,
        serves,
        window_days,
        minimum,
        minimum,
        top_k,
        DEVELOPMENT_CUTOFF.date(),
        None,
        redundant,
    )


def _target(
    player: str = "Alice",
    opponent: str = "Bob",
    when: date = date(2020, 2, 1),
    match_id: str = "target-match",
    orientation: str = "player_1_vs_player_2",
) -> TacticalPipelineTarget:
    return TacticalPipelineTarget(
        match_id,
        player,
        opponent,
        when,
        f"validation_{when.year}",
        orientation,
    )


@lru_cache(maxsize=1)
def _base_result():
    return run_tactical_recommender_pipeline(
        _points(),
        (_target(),),
        _config(),
    )


def _result(**kwargs):
    if not kwargs:
        return _base_result()
    return run_tactical_recommender_pipeline(
        _points(), (_target(),), _config(), **kwargs
    )


@lru_cache(maxsize=1)
def _publication_result():
    return run_tactical_recommender_pipeline(
        _points(), (_target(),), default_pipeline_config()
    )


def test_valid_source_is_copied_sorted_and_not_mutated():
    original = _points().iloc[::-1].reset_index(drop=True)
    before = original.copy(deep=True)
    actual = validate_source_points(original)
    pd.testing.assert_frame_equal(original, before)
    assert actual is not original
    assert list(actual.match_id) == ["m1", "m2", "m3", "m4", "m5", "mt"]
    assert pd.api.types.is_datetime64_any_dtype(actual.date)


@pytest.mark.parametrize("change", ["missing", "additional", "reordered"])
def test_source_rejects_non_exact_schema(change):
    points = _points()
    if change == "missing":
        points = points.drop(columns="surface")
    elif change == "additional":
        points["extra"] = 1
    else:
        points = points.loc[:, list(reversed(points.columns))]
    with pytest.raises(TacticalPipelineContractError, match="Schema"):
        validate_source_points(points)


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("match_id", "", "match_id"),
        ("match_id", "m 1", "match_id"),
        ("point_number", True, "point_number"),
        ("point_number", 1.0, "point_number"),
        ("point_number", "1", "point_number"),
        ("date", "01/02/2020", "date"),
        ("date", None, "date"),
        ("surface", "Carpet", "surface"),
        ("surface", None, "surface"),
        ("server", True, "server"),
        ("server", 1.0, "server"),
        ("server", "1", "server"),
        ("server", 3, "server"),
        ("point_winner", False, "point_winner"),
        ("point_winner", 2.0, "point_winner"),
        ("point_winner", "2", "point_winner"),
        ("player_1", 1, "player_1"),
        ("player_2", " Bob", "player_2"),
        ("first_serve", None, "first_serve"),
        ("first_serve", " ", "first_serve"),
        ("second_serve", 6, "second_serve"),
    ],
)
def test_source_rejects_invalid_domains(column, value, error):
    points = _points()
    points[column] = points[column].astype(object)
    points.loc[0, column] = value
    with pytest.raises((TacticalPipelineContractError, TypeError), match=error):
        validate_source_points(points)


def test_source_rejects_duplicate_point_key():
    points = pd.concat([_points(), _points().iloc[[0]]], ignore_index=True)
    with pytest.raises(TacticalPipelineContractError, match="duplicada"):
        validate_source_points(points)


def test_source_rejects_equal_players():
    points = _points()
    points.loc[0, "player_2"] = "Alice"
    with pytest.raises(TacticalPipelineContractError, match="distintos"):
        validate_source_points(points)


@pytest.mark.parametrize("field,value", [("date", "2020-01-02"), ("surface", "Clay"), ("player_2", "Erin")])
def test_source_rejects_inconsistent_match_metadata(field, value):
    points = _points().iloc[[0]].copy()
    extra = points.iloc[[0]].copy()
    extra.loc[:, "point_number"] = 2
    extra.loc[:, field] = value
    points = pd.concat([points, extra], ignore_index=True)
    with pytest.raises(TacticalPipelineContractError, match="Metadatos"):
        validate_source_points(points)


def test_expected_source_cardinality_is_closed():
    with pytest.raises(TacticalPipelineContractError, match="Cardinalidad"):
        validate_source_points(_points(), expected_rows=1_280_408)


def test_source_rejects_empty_frame_with_exact_schema():
    with pytest.raises(TacticalPipelineContractError, match="vacia"):
        validate_source_points(pd.DataFrame(columns=SOURCE_COLUMNS))


def test_temporal_seal_includes_cutoff_excludes_test_and_has_nine_zeroes():
    source = validate_source_points(_points())
    development, population, seal = seal_development_points(source)
    assert list(development.match_id) == ["m1", "m2", "m3", "m4", "m5"]
    assert population.development_point_rows == 5
    assert population.excluded_test_point_rows == 1
    assert population.excluded_test_matches == 1
    assert seal.test_status == "sealed"
    assert seal.used_for_method_selection is False
    assert seal.counters == tuple((name, 0) for name in TEST_ZERO_FIELDS)


def test_temporal_seal_rejects_match_crossing_cutoff():
    source = validate_source_points(_points())
    crossing = source.iloc[[0, -1]].copy()
    crossing.loc[:, "match_id"] = "shared"
    crossing.loc[:, "point_number"] = [1, 2]
    crossing.loc[:, "surface"] = "Hard"
    crossing.loc[:, "player_1"] = "A"
    crossing.loc[:, "player_2"] = "B"
    with pytest.raises(TacticalPipelineContractError, match="cruza"):
        seal_development_points(crossing)


def test_test_rows_never_reach_orchestrator_encoder_evidence_or_prioritizer():
    calls = {"orchestrator": [], "encoder": 0, "evidence": 0, "prioritizer": 0}

    def orchestrator(request):
        calls["orchestrator"].append(request.sequence_text)
        return extract_tactical_signals_for_attempt(request)

    def encoder(extraction, schema):
        calls["encoder"] += 1
        return encode_tactical_attempt(extraction, schema)

    def evidence(history, query, schema):
        calls["evidence"] += 1
        assert all(item.effective_date <= DEVELOPMENT_CUTOFF.date() for item in history)
        assert all(item.player not in {"TestA", "TestB"} for item in history)
        return build_tactical_matchup_evidence(history, query, schema)

    def prioritizer(value, *, top_k):
        calls["prioritizer"] += 1
        return prioritize_tactical_matchup(value, top_k=top_k)

    result = _result(
        orchestrator=orchestrator,
        encoder=encoder,
        reference_evidence_builder_for_tests=evidence,
        prioritizer=prioritizer,
    )
    assert "5b28" not in calls["orchestrator"]
    assert calls["encoder"] == result.cache_metrics.unique_keys
    assert calls["evidence"] == calls["prioritizer"] == 1


def test_attempts_include_every_first_and_only_substantive_seconds():
    development, _, _ = seal_development_points(validate_source_points(_points()))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    assert batch.first_attempts == 5
    assert batch.second_attempts == 2
    assert len(batch.records) == 7
    assert batch.second_with_documented_first_fault == 1
    assert batch.second_without_documented_first_fault == 1
    assert [(item.match_id, item.serve_number) for item in batch.records if item.match_id == "m5"] == [("m5", 1), ("m5", 2)]
    assert next(item for item in batch.records if item.match_id == "m3" and item.serve_number == 2).previous_attempt_was_fault is False
    assert next(item for item in batch.records if item.match_id == "m5" and item.serve_number == 2).previous_attempt_was_fault is True


@pytest.mark.parametrize("second", [None, "", "   "])
def test_null_empty_and_whitespace_second_serve_do_not_create_attempt(second):
    points = _points().iloc[[0]].copy()
    points["second_serve"] = pd.Series([second], dtype=object)
    development = validate_source_points(points)
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    assert batch.first_attempts == 1
    assert batch.second_attempts == 0


def test_attempt_keys_and_order_are_deterministic():
    development, _, _ = seal_development_points(validate_source_points(_points().sample(frac=1, random_state=7)))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    keys = [(item.effective_date, item.match_id, item.point_number, item.serve_number) for item in batch.records]
    assert keys == sorted(keys)
    assert len(keys) == len(set((b, c, d) for _, b, c, d in keys))


def test_cache_key_calls_and_object_reuse_are_exact():
    development, _, _ = seal_development_points(validate_source_points(_points()))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    orchestrator_calls = []
    encoder_calls = []

    def orchestrator(request):
        orchestrator_calls.append((request.sequence_text, request.serve_number, request.previous_attempt_was_fault))
        return extract_tactical_signals_for_attempt(request)

    def encoder(extraction, selected_schema):
        encoder_calls.append(id(extraction))
        return encode_tactical_attempt(extraction, selected_schema)

    batch = construct_tactical_attempt_records(
        development, schema, orchestrator=orchestrator, encoder=encoder
    )
    expected_keys = {
        (item.sequence_text, item.serve_number, item.previous_attempt_was_fault)
        for item in batch.records
    }
    assert set(orchestrator_calls) == expected_keys
    assert len(orchestrator_calls) == len(encoder_calls) == len(expected_keys)
    assert batch.cache_metrics.cache_misses == len(expected_keys)
    assert batch.cache_metrics.cache_hits == len(batch.records) - len(expected_keys)
    repeated = [item for item in batch.records if item.sequence_text == "4f17" and item.serve_number == 1]
    assert len({id(item.extraction) for item in repeated}) == 1
    assert len({id(item.feature_vector) for item in repeated}) == 1


def test_cache_validator_rejects_equal_but_non_reused_objects():
    development, _, _ = seal_development_points(validate_source_points(_points()))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    repeated = [item for item in batch.records if item.sequence_text == "4f17" and item.serve_number == 1]
    victim = repeated[-1]
    detached = replace(
        victim,
        extraction=replace(victim.extraction),
        feature_vector=replace(victim.feature_vector),
    )
    records = tuple(detached if item is victim else item for item in batch.records)
    with pytest.raises(TacticalPipelineContractError, match="reutilizados"):
        validate_tactical_attempt_batch(
            replace(batch, records=records),
            development_point_rows=5,
            schema=schema,
        )


def test_cache_distinguishes_text_serve_number_and_fault_context():
    points = _points().iloc[[0, 2, 4]].copy()
    points.loc[points.match_id == "m3", "second_serve"] = "6f17"
    development = validate_source_points(points)
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    keys = {
        (item.sequence_text, item.serve_number, item.previous_attempt_was_fault)
        for item in batch.records
    }
    assert ("4f17", 1, False) in keys
    assert ("6f17", 2, False) in keys
    assert ("6f17", 2, True) in keys
    assert ("4n", 1, False) in keys


def test_contextual_double_fault_is_preserved_on_second_attempt():
    points = _points().iloc[[0]].copy()
    points.loc[:, "first_serve"] = "4n"
    points.loc[:, "second_serve"] = "6n"
    development = validate_source_points(points)
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    second = next(item for item in batch.records if item.serve_number == 2)
    assert second.previous_attempt_was_fault is True
    assert any(
        "censored_double_fault" in item.reason_codes
        for item in second.extraction.adaptations
    )


def test_attempt_batch_validator_rejects_mutated_context_and_duplicate_key():
    development, _, _ = seal_development_points(validate_source_points(_points()))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    second = next(item for item in batch.records if item.match_id == "m5" and item.serve_number == 2)
    bad_context = replace(second, previous_attempt_was_fault=False)
    bad_records = tuple(bad_context if item is second else item for item in batch.records)
    with pytest.raises(TacticalPipelineContractError):
        validate_tactical_attempt_batch(replace(batch, records=bad_records), development_point_rows=5, schema=schema)
    with pytest.raises(TacticalPipelineContractError, match="duplicada"):
        validate_tactical_attempt_batch(replace(batch, records=batch.records + (batch.records[0],)), development_point_rows=5, schema=schema)


def test_attempt_validator_rejects_valid_extraction_paired_with_another_vector():
    development, _, _ = seal_development_points(validate_source_points(_points()))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    batch = construct_tactical_attempt_records(development, schema)
    wide = next(item for item in batch.records if item.sequence_text == "4f17" and item.serve_number == 1)
    fault = next(item for item in batch.records if item.sequence_text == "4n")
    with pytest.raises(TacticalPipelineContractError, match="reconcilia"):
        pipeline.validate_tactical_attempt_record(
            replace(wide, extraction=fault.extraction), schema
        )


def test_observations_have_neutral_server_orientation_and_correct_labels():
    result = _result()
    assert len(result.observations) == len(result.attempts) == 7
    for attempt, observation in zip(result.attempts, result.observations):
        assert observation.player == attempt.server_player
        assert observation.opponent == attempt.returner_player
        assert observation.role is TacticalPlayerRole.SERVER
        assert observation.feature_vector is attempt.feature_vector
        assert observation.label is (
            TacticalOutcomeLabel.SERVER_WON_POINT
            if attempt.server_won_point
            else TacticalOutcomeLabel.RETURNER_WON_POINT
        )
    m4 = next(item for item in result.attempts if item.match_id == "m4")
    assert (m4.server_player, m4.returner_player, m4.server_won_point) == ("Bob", "Carol", False)


def test_matchup_evidence_attributes_p02_to_server_and_return_patterns_to_returner():
    evidence = _result().target_results[0].evidence
    expected = {
        "p02.first_serve_direction.4": (1, 1),
        "p04.return_direction.1": (1, 1),
        "p05.return_depth.7": (1, 1),
        "p06.return_shot_type.f": (1, 1),
    }
    for feature, activations in expected.items():
        category = next(item for item in evidence.categories if item.feature_name == feature)
        assert (
            category.executor_evidence.labeled_activations,
            category.opponent_allowed_evidence.labeled_activations,
        ) == activations


def test_feature_vector_is_independent_of_outcome_label():
    points_a = _points()
    points_b = _points()
    points_b.loc[points_b.match_id == "m1", "point_winner"] = 2
    dev_a, _, _ = seal_development_points(validate_source_points(points_a))
    dev_b, _, _ = seal_development_points(validate_source_points(points_b))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    attempts_a = construct_tactical_attempt_records(dev_a, schema).records
    attempts_b = construct_tactical_attempt_records(dev_b, schema).records
    vector_a = next(item.feature_vector for item in attempts_a if item.match_id == "m1")
    vector_b = next(item.feature_vector for item in attempts_b if item.match_id == "m1")
    assert vector_a == vector_b
    assert next(item.server_won_point for item in attempts_a if item.match_id == "m1") is True
    assert next(item.server_won_point for item in attempts_b if item.match_id == "m1") is False


def test_same_day_and_future_are_excluded_by_exclusive_date_cut():
    result = run_tactical_recommender_pipeline(
        _points(), (_target(when=date(2020, 1, 3)),), _config()
    )
    target = result.target_results[0]
    assert target.history_observation_count == 2
    assert target.evidence.total_observations_received == 2
    assert all(item.effective_date < date(2020, 1, 3) for item in result.observations[:2])


def test_window_uses_inclusive_lower_and_exclusive_upper_bounds():
    result = run_tactical_recommender_pipeline(
        _points(),
        (_target(when=date(2020, 1, 4)),),
        _config(window_days=1),
    )
    assert result.target_results[0].history_observation_count == 2
    assert result.target_results[0].evidence.total_observations_received == 2


def test_permuted_source_order_and_same_day_outcome_do_not_change_target_result():
    target = _target(when=date(2020, 1, 3))
    first = run_tactical_recommender_pipeline(_points(), (target,), _config())
    changed = _points().sample(frac=1, random_state=11).reset_index(drop=True)
    changed.loc[changed.match_id == "m3", "point_winner"] = 1
    second = run_tactical_recommender_pipeline(changed, (target,), _config())
    assert first.target_results[0].prioritization == second.target_results[0].prioritization
    assert tactical_pipeline_result_fingerprint(first) == tactical_pipeline_result_fingerprint(second)


def test_multiple_targets_are_sorted_and_history_is_built_once():
    targets = (
        _target("Alice", "Bob", date(2021, 1, 1), "target-2021"),
        _target("Bob", "Alice", date(2020, 1, 4), "target-2020"),
    )
    result = run_tactical_recommender_pipeline(_points(), targets, _config())
    assert [item.target.as_of_date for item in result.target_results] == sorted(item.as_of_date for item in targets)
    assert result.diagnostics.full_observation_rebuilds == 1
    assert result.diagnostics.target_history_slices == 2


def test_available_partial_and_not_available_global_states():
    available = _result()
    partial = run_tactical_recommender_pipeline(
        _points(),
        (_target(), _target("Nobody", "Unknown", date(2020, 2, 2), "target-other")),
        _config(),
    )
    unavailable = run_tactical_recommender_pipeline(
        _points(), (_target(),), _config(minimum=100)
    )
    assert available.state is TacticalPipelineState.AVAILABLE
    assert partial.state is TacticalPipelineState.PARTIALLY_AVAILABLE
    assert unavailable.state is TacticalPipelineState.NOT_AVAILABLE
    assert (partial.targets_available + partial.targets_partial, partial.targets_not_available) == (1, 1)


def test_one_target_with_only_some_pattern_rankings_is_globally_available():
    result = run_tactical_recommender_pipeline(
        _points(),
        (_target(when=date(2020, 1, 3)),),
        _config(),
    )
    assert result.target_results[0].state is TacticalPrioritizationState.PARTIALLY_AVAILABLE
    assert result.state is TacticalPipelineState.AVAILABLE
    assert result.reason_codes == ("all_targets_have_some_prioritization",)


def test_rankings_are_separate_by_pattern_with_no_global_top_three():
    result = _result()
    prioritization = result.target_results[0].prioritization
    assert tuple(item.pattern_id for item in prioritization.rankings) == ("P02", "P04", "P05", "P06")
    assert all(len(item.top_candidates) <= 3 for item in prioritization.rankings)
    payload_names = [name for name, _ in prepare_tactical_pipeline_payloads(_publication_result())]
    assert payload_names == ["summary", "population", "availability", "rankings", "diagnostics"]


def test_profile_only_is_supported_synthetically_without_component_profile_mix():
    result = run_tactical_recommender_pipeline(
        _points(),
        (_target(),),
        _config(
            policy=TacticalEncodingPolicy.PROFILE_ONLY,
            patterns=("P02", "P09"),
        ),
    )
    assert result.schema.policy is TacticalEncodingPolicy.PROFILE_ONLY
    assert tuple(item.pattern_id for item in result.target_results[0].prioritization.rankings) == ("P02", "P09")


def test_redundant_policy_requires_explicit_audit_flag():
    with pytest.raises(TacticalPipelineContractError, match="auditoria redundante"):
        _config(
            policy=TacticalEncodingPolicy.COMPONENTS_AND_PROFILE,
            patterns=("P02", "P04", "P05", "P06", "P09"),
        )
    audit = _config(
        policy=TacticalEncodingPolicy.COMPONENTS_AND_PROFILE,
        patterns=("P02", "P04", "P05", "P06", "P09"),
        redundant=True,
    )
    assert audit.allow_redundant_audit is True
    assert default_pipeline_config().encoder_policy is TacticalEncodingPolicy.COMPONENT_ONLY


def test_result_counts_reconciliations_and_timings_are_complete():
    result = _result()
    assert result.population.attempt_count == 7
    assert result.population.observation_count == 7
    assert result.population.second_with_documented_first_fault == 1
    assert result.population.second_without_documented_first_fault == 1
    assert result.targets_requested == result.targets_processed == 1
    assert result.total_candidates == result.scored_candidates + result.abstained_candidates
    assert all(value is True for _, value in result.reconciliations)
    assert tuple(name for name, _ in result.diagnostics.stage_seconds) == pipeline.STAGE_ORDER
    assert all(value >= 0 for _, value in result.diagnostics.stage_seconds)
    validate_tactical_pipeline_result(result)


def test_payloads_are_deterministic_aggregate_safe_and_valid_json():
    result = _publication_result()
    first = prepare_tactical_pipeline_payloads(result)
    second = prepare_tactical_pipeline_payloads(result)
    assert first == second
    serialized = b"".join(payload for _, payload in first)
    for forbidden in (b"4f17", b"match_id", b"point_number", b"TestA", b"Alice"):
        assert forbidden not in serialized
    assert isinstance(json.loads(first[0][1].decode("utf-8")), dict)
    for _, payload in first:
        assert b"NaN" not in payload and b"Infinity" not in payload


def test_fingerprint_is_deterministic_and_excludes_timings():
    result = _result()
    changed_diagnostics = replace(
        result.diagnostics,
        stage_seconds=tuple((name, value + 10.0) for name, value in result.diagnostics.stage_seconds),
    )
    changed = replace(result, diagnostics=changed_diagnostics)
    validate_tactical_pipeline_result(changed)
    assert tactical_pipeline_result_fingerprint(result) == tactical_pipeline_result_fingerprint(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: replace(value, state=TacticalPipelineState.NOT_AVAILABLE),
        lambda value: replace(value, targets_processed=0),
        lambda value: replace(value, total_candidates=value.total_candidates + 1),
        lambda value: replace(value, reconciliations=(("source_population_exhaustive", False),)),
        lambda value: replace(value, population=replace(value.population, observation_count=0)),
        lambda value: replace(value, cache_metrics=replace(value.cache_metrics, cache_hits=0)),
        lambda value: replace(value, test_seal=replace(value.test_seal, used_for_method_selection=True)),
        lambda value: replace(value, diagnostics=replace(value.diagnostics, stage_seconds=(("bad", 0.0),))),
    ],
)
def test_reconstructive_validator_rejects_mutated_result_contract(mutation):
    with pytest.raises((TacticalPipelineContractError, TypeError)):
        validate_tactical_pipeline_result(mutation(_result()))


def test_dataclasses_are_frozen():
    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.state = TacticalPipelineState.NOT_AVAILABLE
    with pytest.raises(FrozenInstanceError):
        result.attempts[0].surface = "Clay"


def test_failures_preserve_exact_stage_and_sanitize_sensitive_message():
    def broken(_request):
        raise ValueError("m1 Alice 4f17 C:\\secret")

    with pytest.raises(TacticalPipelineExecutionError) as captured:
        _result(orchestrator=broken)
    assert captured.value.stage == "extraction_and_encoding"
    assert captured.value.error_type == "ValueError"
    assert str(captured.value) == "extraction_and_encoding_failed:ValueError"
    assert all(secret not in str(captured.value) for secret in ("m1", "Alice", "4f17", "secret"))


def test_controlled_analysis_failure_preserves_partial_stage_timings():
    timings = {}

    def broken(_request):
        raise ValueError("sensitive synthetic detail")

    with pytest.raises(TacticalPipelineExecutionError) as captured:
        run_tactical_recommender_pipeline(
            _points(),
            (_target(),),
            _config(),
            orchestrator=broken,
            stage_timing_sink=timings,
        )
    assert captured.value.stage == "extraction_and_encoding"
    assert tuple(timings) == (
        "source_validation",
        "temporal_seal",
        "extraction_and_encoding",
    )
    assert all(type(value) is float and value >= 0.0 for value in timings.values())


def test_source_failure_is_stage_aware_when_using_pipeline():
    points = _points().drop(columns="surface")
    with pytest.raises(TacticalPipelineExecutionError) as captured:
        run_tactical_recommender_pipeline(points, (_target(),), _config())
    assert captured.value.stage == "source_validation"
    assert captured.value.error_type == "TacticalPipelineContractError"


@pytest.mark.parametrize(
    "dependency,stage",
    [("evidence", "target_evidence"), ("prioritizer", "prioritization")],
)
def test_downstream_failures_preserve_stage_without_sensitive_details(dependency, stage):
    def broken(*_args, **_kwargs):
        raise RuntimeError("Alice m1 4f17")

    kwargs = {
        "reference_evidence_builder_for_tests"
        if dependency == "evidence"
        else "prioritizer": broken
    }
    with pytest.raises(TacticalPipelineExecutionError) as captured:
        _result(**kwargs)
    assert captured.value.stage == stage
    assert str(captured.value) == f"{stage}_failed:RuntimeError"
    assert all(value not in str(captured.value) for value in ("Alice", "m1", "4f17"))


def test_cli_is_blocked_before_reading_or_writing(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", False)
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: called.append((args, kwargs)))
    with pytest.raises(SystemExit, match=pipeline.REAL_EXECUTION_BLOCK_REASON):
        pipeline.main(())
    assert pipeline.REAL_EXECUTION_AUTHORIZED is False
    assert called == []
    assert list(tmp_path.iterdir()) == []


def test_real_execution_entry_point_requires_manual_reauthorization_before_any_io(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", False)
    calls = []
    paths = _artifact_paths(tmp_path / "artifacts")
    with pytest.raises(SystemExit, match=pipeline.REAL_EXECUTION_BLOCK_REASON):
        pipeline.execute_authorized_real_pipeline(
            pipeline.POINTS_PATH,
            tmp_path / "performance.json",
            artifact_paths=paths,
            source_reader=lambda _path: calls.append("read"),
            upstream_validator=lambda: calls.append("upstream"),
            publisher=lambda _result, _paths: calls.append("publish"),
        )
    assert calls == []
    assert not (tmp_path / "performance.json").exists()
    assert not any(path.exists() for path in pipeline._artifact_paths_tuple(paths))
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_cli_rejects_performance_log_inside_repository_before_read(monkeypatch):
    called = []
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: called.append((args, kwargs)))
    with pytest.raises(TacticalPipelineContractError, match="fuera_del_repositorio"):
        pipeline.main(("--performance-log", str(pipeline.ROOT / "bad.json")))
    assert called == []


def test_cli_rejects_alternative_source_before_read(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: called.append((args, kwargs)))
    with pytest.raises(TacticalPipelineContractError, match="source_no_autorizada"):
        pipeline.main(("--source", str(tmp_path / "other.parquet")))
    assert called == []


def test_authorized_main_returns_exit_code_zero_without_bypass_or_real_io(
    monkeypatch, tmp_path
):
    result = _publication_result()
    calls = []
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        pipeline,
        "execute_authorized_real_pipeline",
        lambda source, performance: calls.append((source, performance)) or result,
    )
    performance = tmp_path / "performance.json"
    assert pipeline.main(("--performance-log", str(performance))) is None
    assert calls == [(pipeline.POINTS_PATH, performance)]


def test_authorized_main_returns_exit_code_one_for_controlled_not_available(
    monkeypatch, tmp_path
):
    base = _publication_result()
    unavailable = build_not_available_publication(
        TacticalPipelineExecutionError("target_evidence", "SyntheticFailure"),
        base.test_seal,
    )
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        pipeline,
        "execute_authorized_real_pipeline",
        lambda _source, _performance: unavailable,
    )
    with pytest.raises(SystemExit) as captured:
        pipeline.main(("--performance-log", str(tmp_path / "performance.json")))
    assert captured.value.code == 1


def test_authorization_is_one_explicit_constant_and_execution_has_no_retry_loop():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "REAL_EXECUTION_AUTHORIZED"
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant)
    assert assignments[0].value.value is False
    assert pipeline.REAL_EXECUTION_AUTHORIZATION_REASON == (
        "real_execution_completed_no_further_execution_authorized"
    )
    assert pipeline.REAL_EXECUTION_ATTEMPTS == 2
    assert pipeline.FIRST_EXECUTION_STATUS == "interrupted_for_performance_diagnosis"
    assert pipeline.SECOND_EXECUTION_STATUS == "completed"
    assert pipeline.HISTORICAL_EXECUTION_STATUS == "second_execution_completed"
    assert pipeline.NO_ARTIFACTS_PUBLISHED is False
    assert pipeline.AUTOMATIC_RETRY is False
    assert pipeline.FURTHER_REAL_EXECUTION_AUTHORIZED is False
    assert (
        pipeline.REAL_EXECUTION_BLOCK_REASON
        == "real_execution_completed_no_further_execution_authorized"
    )
    execution_tree = ast.parse(inspect.getsource(pipeline.execute_authorized_real_pipeline))
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(execution_tree))
    called_names = [
        node.func.id
        for node in ast.walk(execution_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert called_names.count("source_reader") == 0
    assert called_names.count("selected_source_reader") == 1


def test_real_configuration_rejects_the_reference_evidence_route():
    with pytest.raises(TacticalPipelineContractError, match="evidencia indexada"):
        run_tactical_recommender_pipeline(
            _points(),
            (_target(),),
            replace(
                default_pipeline_config(),
                expected_source_rows=pipeline.EXPECTED_SOURCE_ROWS,
            ),
            reference_evidence_builder_for_tests=lambda history, query, schema: (
                build_tactical_matchup_evidence(history, query, schema)
            ),
        )


def test_architecture_has_one_parquet_read_and_no_csv_iterrows_or_productive_asserts():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    read_parquet = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pd"
        and node.func.attr == "read_parquet"
    ]
    assert len(read_parquet) == 1
    analytical_source = inspect.getsource(pipeline.run_tactical_recommender_pipeline)
    source_reader = inspect.getsource(pipeline._read_source_points_once)
    assert "read_csv" not in analytical_source
    assert "read_csv" not in source_reader
    assert ".iterrows(" not in source
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    assert not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, (ast.Dict, ast.List, ast.Set))
        for node in tree.body
    )


def test_architecture_seals_before_constructing_and_builds_observations_once():
    source = inspect.getsource(run_tactical_recommender_pipeline)
    indexed_selection = inspect.getsource(pipeline._select_target_history)
    assert source.index("seal_development_points") < source.index("construct_tactical_attempt_records")
    assert source.count("construct_tactical_attempt_records") == 1
    assert source.count("build_tactical_historical_observations") == 1
    assert source.index("build_tactical_historical_observations") < source.index("build_validation_targets")
    assert "_build_history_index" in source
    assert "bisect_left" in indexed_selection
    assert "observations[:" not in source


def test_only_authorized_files_exist_for_this_phase():
    assert MODULE_PATH.name == "tactical_recommender_pipeline.py"
    assert Path(__file__).name == "test_tactical_recommender_pipeline.py"


def _artifact_paths(root: Path) -> TacticalPipelineArtifactPaths:
    return TacticalPipelineArtifactPaths(
        root / "summary.json",
        root / "population.csv",
        root / "availability.csv",
        root / "rankings.csv",
        root / "diagnostics.csv",
    )


def _read_serialized_csv(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(payload), keep_default_na=False)


def _resign_summary(summary: dict, csv_payloads: tuple[tuple[str, bytes], ...]) -> bytes:
    summary["artifact_manifest"] = {
        name: {"bytes": len(payload), "sha256": sha256(payload).hexdigest().upper()}
        for name, payload in csv_payloads
    }
    unsigned = dict(summary)
    unsigned.pop("publication_fingerprint", None)
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    summary["publication_fingerprint"] = sha256(
        pipeline.ARTIFACT_FINGERPRINT_DOMAIN + canonical
    ).hexdigest().upper()
    return (
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def test_source_reader_is_the_single_exact_projected_parquet_call():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pd"
        and node.func.attr == "read_parquet"
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0]) == "pd.read_parquet(source_path, columns=SOURCE_COLUMNS)"


def test_default_policy_is_the_frozen_transferred_conservative_baseline():
    config = default_pipeline_config()
    assert pipeline.ANALYSIS_PURPOSE == "sealed_descriptive_pipeline_feasibility"
    assert pipeline.DESCRIPTIVE_POLICY_NAME == "transferred_conservative_baseline"
    assert config.encoder_policy is TacticalEncodingPolicy.COMPONENT_ONLY
    assert config.requested_patterns == PRODUCTION_PATTERNS == ("P02", "P04", "P05", "P06")
    assert config.serve_numbers == (1, 2)
    assert config.window_days is None
    assert (config.minimum_labeled_attempts, config.minimum_matches, config.top_k) == (50, 5, 3)
    assert config.allow_redundant_audit is False


def test_validation_target_builder_emits_two_canonical_orientations_per_match():
    source = validate_source_points(_points())
    development, _, _ = seal_development_points(source)
    targets = build_validation_targets(development)
    assert len(targets) == 10
    assert [(item.target_match_id, item.orientation) for item in targets[:2]] == [
        ("m1", "player_1_vs_player_2"),
        ("m1", "player_2_vs_player_1"),
    ]
    by_match = {}
    for target in targets:
        by_match.setdefault(target.target_match_id, []).append(target)
    for pair in by_match.values():
        assert len(pair) == 2
        assert pair[0].as_of_date == pair[1].as_of_date
        assert (pair[0].player, pair[0].opponent) == (pair[1].opponent, pair[1].player)
    keys = [(item.as_of_date, item.target_match_id, item.orientation) for item in targets]
    assert keys == sorted(keys)


def test_frozen_validation_fold_cardinalities_are_enforced_from_synthetic_matches():
    rows = []
    counter = 0
    for fold, match_count in EXPECTED_FOLD_MATCHES:
        year = int(fold[-4:])
        for index in range(match_count):
            counter += 1
            rows.append(
                (
                    f"synthetic-{counter:04d}",
                    1,
                    pd.Timestamp(year=year, month=1 + index % 12, day=1 + index % 27),
                    "Hard",
                    1,
                    1,
                    f"Player-{counter}-A",
                    f"Player-{counter}-B",
                    "4f17",
                    None,
                )
            )
    development = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    targets = build_validation_targets(development, enforce_frozen_cardinalities=True)
    assert len({item.target_match_id for item in targets}) == EXPECTED_VALIDATION_MATCHES
    assert len(targets) == EXPECTED_TARGET_ORIENTATIONS
    assert tuple(
        (fold, len({item.target_match_id for item in targets if item.fold == fold}))
        for fold in VALIDATION_FOLDS
    ) == EXPECTED_FOLD_MATCHES


def test_target_builder_rejects_frozen_cardinality_drift():
    development, _, _ = seal_development_points(validate_source_points(_points()))
    with pytest.raises(TacticalPipelineContractError, match="Cardinalidades target"):
        build_validation_targets(development, enforce_frozen_cardinalities=True)


def test_history_lookup_uses_actor_date_streams_not_full_history_cross_product():
    template = _base_result().observations[0]
    observations = []
    for index in range(500):
        relevant = index < 10
        observations.append(
            replace(
                template,
                match_id=f"history-{index}",
                effective_date=date(2019, 1, 1),
                player="Alice" if relevant else f"Server-{index}",
                opponent="Bob" if relevant else f"Returner-{index}",
            )
        )
    index = pipeline._build_history_index(tuple(observations))
    examined = lookups = 0
    target_count = 100
    for number in range(target_count):
        history, current_lookups, current_examined = pipeline._select_target_history(
            index,
            _target(match_id=f"target-{number}"),
            default_pipeline_config(),
        )
        assert len(history) == 10
        examined += current_examined
        lookups += current_lookups
    assert lookups == 4 * target_count
    assert examined == 20 * target_count
    assert examined < len(observations) * target_count


def _prepared_observations(
    *, relevant: int, unrelated: int = 0
) -> tuple:
    template = _base_result().observations[0]
    rows = []
    for number in range(relevant):
        rows.append(
            replace(
                template,
                match_id=f"relevant-{number}",
                point_number=number + 1,
                effective_date=date(2018, 1, 1 + number % 20),
                player="Alice" if number % 2 == 0 else "Bob",
                opponent="Bob" if number % 2 == 0 else "Alice",
            )
        )
    for number in range(unrelated):
        rows.append(
            replace(
                template,
                match_id=f"unrelated-{number}",
                point_number=number + 1,
                effective_date=date(2018, 1, 1 + number % 20),
                player=f"OtherServer{number}",
                opponent=f"OtherReturner{number}",
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.effective_date,
                item.match_id,
                item.point_number,
                item.serve_number,
            ),
        )
    )


def _indexed_and_reference_evidence(observations, target, config):
    schema = build_tactical_feature_schema(config.encoder_policy)
    counters = dict.fromkeys(pipeline.OPERATION_COUNTER_FIELDS, 0)
    prepared = pipeline._build_prepared_evidence_index(
        observations,
        config,
        schema,
        operation_counters=counters,
    )
    indexed, lookups, examined = pipeline._build_indexed_matchup_evidence(
        prepared,
        target,
        config,
        schema,
        operation_counters=counters,
    )
    historical = pipeline._build_history_index(observations)
    selected, expected_lookups, expected_examined = pipeline._select_target_history(
        historical,
        target,
        config,
    )
    reference = build_tactical_matchup_evidence(selected, indexed.query, schema)
    return indexed, reference, counters, (lookups, examined), (
        expected_lookups,
        expected_examined,
    )


@pytest.mark.parametrize(
    "config",
    [
        _config(),
        _config(serves=(2,)),
        _config(window_days=20),
    ],
    ids=("both_serves", "second_serve_only", "bounded_window"),
)
def test_indexed_evidence_is_exactly_equivalent_to_reference_for_all_catalogs(config):
    development, _, _ = seal_development_points(validate_source_points(_points()))
    schema = build_tactical_feature_schema(config.encoder_policy)
    batch = construct_tactical_attempt_records(development, schema)
    observations = build_tactical_historical_observations(batch.records)
    actual, expected, counters, actual_cost, expected_cost = (
        _indexed_and_reference_evidence(observations, _target(), config)
    )
    assert actual == expected
    assert actual_cost == expected_cost
    assert len(actual.categories) == 26
    assert tuple((item.pattern_id, item.category) for item in actual.categories) == (
        ("P02", "4"), ("P02", "5"), ("P02", "6"),
        ("P04", "1"), ("P04", "2"), ("P04", "3"),
        ("P05", "7"), ("P05", "8"), ("P05", "9"),
        *(("P06", code) for code in "fbrsvzopuylmhijkt"),
    )
    assert counters["actor_relevant_observations_inspected"] == 0
    assert counters["history_materializations"] == 0
    if config.serve_numbers == (2,):
        p02 = tuple(item for item in actual.categories if item.pattern_id == "P02")
        assert all(
            item.matchup_state.value == "not_applicable" for item in p02
        )


@pytest.mark.parametrize("random_state", [None, 19], ids=("canonical", "permuted"))
def test_optimized_pipeline_artifacts_are_byte_identical_to_reference_across_input_orders(
    random_state,
):
    points = (
        _points()
        if random_state is None
        else _points().sample(frac=1, random_state=random_state).reset_index(drop=True)
    )
    target = (_target(),)
    config = default_pipeline_config()
    optimized = run_tactical_recommender_pipeline(points, target, config)
    reference = run_tactical_recommender_pipeline(
        points,
        target,
        config,
        reference_evidence_builder_for_tests=lambda history, query, schema: build_tactical_matchup_evidence(
            history, query, schema
        ),
    )
    assert optimized.target_results == reference.target_results
    assert optimized.population == reference.population
    assert optimized.cache_metrics == reference.cache_metrics
    assert serialize_tactical_pipeline_artifacts(optimized) == (
        serialize_tactical_pipeline_artifacts(reference)
    )


def test_indexed_queries_ignore_unconsulted_actor_history_by_operation_count():
    target = _target(when=date(2020, 1, 1))
    config = _config()
    small = _indexed_and_reference_evidence(
        _prepared_observations(relevant=12), target, config
    )
    large = _indexed_and_reference_evidence(
        _prepared_observations(relevant=12, unrelated=400), target, config
    )
    assert small[0] == large[0]
    assert small[2]["stream_queries"] == large[2]["stream_queries"]
    assert small[2]["category_iterations"] == large[2]["category_iterations"] == 26
    assert small[2]["actor_relevant_observations_inspected"] == 0
    assert large[2]["actor_relevant_observations_inspected"] == 0
    assert small[2]["history_materializations"] == large[2]["history_materializations"] == 0
    assert small[2]["history_index_observations"] == 12
    assert large[2]["history_index_observations"] == 412


def test_indexed_target_growth_is_linear_in_targets_not_global_history():
    config = _config()
    observations = _prepared_observations(relevant=20, unrelated=200)
    schema = build_tactical_feature_schema(config.encoder_policy)
    counters = dict.fromkeys(pipeline.OPERATION_COUNTER_FIELDS, 0)
    prepared = pipeline._build_prepared_evidence_index(
        observations, config, schema, operation_counters=counters
    )
    totals = []
    for number in range(30):
        target = _target(
            when=date(2020, 1, 1),
            match_id=f"shared-cut-{number}",
            orientation=(
                "player_1_vs_player_2"
                if number % 2 == 0
                else "player_2_vs_player_1"
            ),
        )
        evidence, _, _ = pipeline._build_indexed_matchup_evidence(
            prepared,
            target,
            config,
            schema,
            operation_counters=counters,
        )
        totals.append(evidence.total_observations_received)
    assert totals == [20] * 30
    assert counters["query_constructions"] == 30
    assert counters["evidence_constructions"] == 30
    assert counters["category_iterations"] == 26 * 30
    assert counters["actor_relevant_observations_inspected"] == 0
    assert counters["history_materializations"] == 0
    assert counters["history_index_observations"] == len(observations)


def test_successive_date_cuts_use_bisect_and_preserve_immutable_index():
    config = _config()
    observations = tuple(
        replace(item, effective_date=date(2020, 1, 1) + timedelta(days=number))
        for number, item in enumerate(_prepared_observations(relevant=20))
    )
    schema = build_tactical_feature_schema(config.encoder_policy)
    counters = dict.fromkeys(pipeline.OPERATION_COUNTER_FIELDS, 0)
    prepared = pipeline._build_prepared_evidence_index(
        observations, config, schema, operation_counters=counters
    )
    sample_series = next(iter(prepared.all_by_server.values()))
    original_dates = sample_series.dates
    original_cumulative = sample_series.cumulative
    totals = []
    for day in range(2, 12):
        evidence, _, _ = pipeline._build_indexed_matchup_evidence(
            prepared,
            _target(when=date(2020, 1, day), match_id=f"successive-{day}"),
            config,
            schema,
            operation_counters=counters,
        )
        totals.append(evidence.total_observations_received)
    assert totals == list(range(1, 11))
    assert sample_series.dates is original_dates
    assert sample_series.cumulative is original_cumulative
    assert counters["actor_relevant_observations_inspected"] == 0


def test_reference_path_exposes_old_target_times_actor_history_cost():
    observations = _prepared_observations(relevant=20, unrelated=200)
    config = _config()
    target = _target(when=date(2020, 1, 1))
    indexed, reference, counters, _, reference_cost = _indexed_and_reference_evidence(
        observations, target, config
    )
    assert indexed == reference
    assert reference_cost == (4, 40)
    assert counters["actor_relevant_observations_inspected"] == 0
    historical = pipeline._build_history_index(observations)
    old_examined = old_materializations = 0
    for number in range(20):
        history, _, examined = pipeline._select_target_history(
            historical,
            _target(match_id=f"reference-{number}"),
            config,
        )
        old_examined += examined
        old_materializations += 1
        assert len(history) == 20
    assert old_examined == 800
    assert old_materializations == 20
    assert counters["category_iterations"] == 26


def test_reference_builder_scans_eligible_history_twice_per_catalog_category(
    monkeypatch,
):
    observations = _prepared_observations(relevant=20)
    target = _target(when=date(2020, 1, 1))
    config = _config()
    schema = build_tactical_feature_schema(config.encoder_policy)
    prepared = pipeline._build_prepared_evidence_index(observations, config, schema)
    indexed, _, _ = pipeline._build_indexed_matchup_evidence(
        prepared, target, config, schema
    )
    historical = pipeline._build_history_index(observations)
    history, _, _ = pipeline._select_target_history(historical, target, config)
    calls = {"categories": 0, "observation_inputs": 0}
    original = matchup_evidence._build_category_evidence

    def counted(**kwargs):
        calls["categories"] += 1
        calls["observation_inputs"] += len(kwargs["observations"])
        return original(**kwargs)

    monkeypatch.setattr(matchup_evidence, "_build_category_evidence", counted)
    reference = matchup_evidence.build_tactical_matchup_evidence(
        history, indexed.query, schema
    )
    assert indexed == reference
    assert calls == {"categories": 52, "observation_inputs": 1_040}


def test_analytical_target_loop_never_serializes_or_fingerprints_artifacts(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("artifact serialization inside analytical target loop")

    monkeypatch.setattr(pipeline, "serialize_tactical_pipeline_artifacts", forbidden)
    monkeypatch.setattr(pipeline, "tactical_pipeline_result_fingerprint", forbidden)
    counters = {}
    result = run_tactical_recommender_pipeline(
        _points(),
        (_target(),),
        _config(),
        operation_counters=counters,
    )
    assert result.targets_processed == 1
    assert counters["artifact_serializations_or_fingerprints_in_target_loops"] == 0
    assert counters["history_materializations"] == 0
    assert counters["expensive_reconstructive_validations"] == 0


def test_two_targets_do_not_share_evidence_when_any_query_dimension_differs():
    config = _config()
    observations = _prepared_observations(relevant=20)
    schema = build_tactical_feature_schema(config.encoder_policy)
    prepared = pipeline._build_prepared_evidence_index(observations, config, schema)
    first, _, _ = pipeline._build_indexed_matchup_evidence(
        prepared, _target(), config, schema
    )
    second, _, _ = pipeline._build_indexed_matchup_evidence(
        prepared,
        _target(player="Bob", opponent="Alice", orientation="player_2_vs_player_1"),
        config,
        schema,
    )
    assert first is not second
    assert first.query != second.query
    assert first.categories is not second.categories


def test_identical_query_semantics_reuse_only_immutable_evidence_and_prioritization():
    counters = {}
    targets = (
        _target(match_id="same-query-a"),
        _target(match_id="same-query-b"),
    )
    result = run_tactical_recommender_pipeline(
        _points(),
        targets,
        _config(),
        operation_counters=counters,
    )
    first, second = result.target_results
    assert first.target != second.target
    assert first.evidence is second.evidence
    assert first.prioritization is second.prioritization
    assert counters["targets_processed"] == 2
    assert counters["query_constructions"] == 1
    assert counters["evidence_constructions"] == 1
    assert counters["prioritizations"] == 1
    assert counters["history_materializations"] == 0


def test_target_memoization_key_contains_every_evidence_and_ranking_dimension():
    target = _target()
    config = _config()
    schema_fingerprint = pipeline.feature_schema_fingerprint(
        build_tactical_feature_schema(config.encoder_policy)
    )
    variants = (
        (target, config, schema_fingerprint),
        (_target(player="Bob", opponent="Alice"), config, schema_fingerprint),
        (_target(when=date(2020, 2, 2)), config, schema_fingerprint),
        (target, _config(serves=(1,)), schema_fingerprint),
        (target, _config(window_days=30), schema_fingerprint),
        (target, _config(minimum=2), schema_fingerprint),
        (target, _config(patterns=("P02",)), schema_fingerprint),
        (target, _config(top_k=2), schema_fingerprint),
        (target, config, "A" * 64),
    )
    keys = tuple(
        pipeline._target_semantic_cache_key(*variant) for variant in variants
    )
    assert len(keys) == len(set(keys))


def test_indexed_route_preserves_insufficiency_and_top_k_boundary_ties():
    rows = []
    for number, code in enumerate(("f", "b", "r", "s"), start=1):
        rows.append(
            (
                f"tie-{number}",
                1,
                f"2019-01-{number:02d}",
                "Hard",
                1,
                2,
                "Bob",
                "Alice",
                f"6{code}17",
                None,
            )
        )
    points = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    tied = run_tactical_recommender_pipeline(
        points,
        (_target(),),
        _config(patterns=("P06",), minimum=1, top_k=3),
    )
    ranking = tied.target_results[0].prioritization.rankings[0]
    assert ranking.pattern_id == "P06"
    assert ranking.boundary_tie_expanded is True
    assert len(ranking.top_candidates) == 4
    insufficient = run_tactical_recommender_pipeline(
        points,
        (_target(),),
        _config(patterns=("P06",), minimum=5, top_k=3),
    )
    assert insufficient.target_results[0].state is TacticalPrioritizationState.NOT_AVAILABLE


def test_pipeline_reports_zero_full_history_scans_and_zero_leakage_audits():
    result = _publication_result()
    assert result.diagnostics.full_history_scans_per_target == 0
    assert result.diagnostics.history_index_entries == result.population.observation_count
    assert result.diagnostics.history_stream_lookups == 4 * result.targets_processed
    assert all(getattr(result.leakage_audit, field) == 0 for field in result.leakage_audit.__dataclass_fields__)
    assert result.test_seal.counters == tuple((field, 0) for field in TEST_ZERO_FIELDS)


def test_five_artifact_names_paths_schemas_and_row_orders_are_exact(tmp_path):
    payloads = prepare_tactical_pipeline_payloads(_publication_result())
    assert tuple(name for name, _ in payloads) == pipeline.ARTIFACT_ORDER
    frames = tuple(_read_serialized_csv(payload) for _, payload in payloads[1:])
    assert tuple(frames[0].columns) == POPULATION_COLUMNS
    assert tuple(frames[1].columns) == AVAILABILITY_COLUMNS
    assert tuple(frames[2].columns) == RANKINGS_COLUMNS
    assert tuple(frames[3].columns) == DIAGNOSTICS_COLUMNS
    assert tuple(frames[0].metric) == POPULATION_ROW_ORDER
    assert tuple(frames[3].check) == pipeline.DIAGNOSTIC_CHECK_ORDER
    assert not any(column.startswith("Unnamed") for frame in frames for column in frame.columns)
    paths = default_artifact_paths(tmp_path)
    assert tuple(path.relative_to(tmp_path).as_posix() for path in pipeline._artifact_paths_tuple(paths)) == (
        "reports/tactical_recommender_pipeline_summary.json",
        "reports/tables/tactical_recommender_pipeline_population.csv",
        "reports/tables/tactical_recommender_pipeline_availability.csv",
        "reports/tables/tactical_recommender_pipeline_rankings.csv",
        "reports/tables/tactical_recommender_pipeline_diagnostics.csv",
    )


def test_serialized_summary_freezes_method_target_seal_manifest_and_fingerprint():
    payloads = prepare_tactical_pipeline_payloads(_publication_result())
    summary = json.loads(payloads[0][1])
    assert summary["analysis_purpose"] == "sealed_descriptive_pipeline_feasibility"
    assert summary["descriptive_policy"] == "transferred_conservative_baseline"
    assert (summary["encoder_policy"], summary["scope"]) == ("component_only", "global_only")
    assert (summary["minimum_labeled_attempts"], summary["minimum_matches"], summary["top_k_per_pattern"]) == (50, 5, 3)
    assert summary["patterns"] == list(PRODUCTION_PATTERNS)
    assert summary["targets"]["unit"] == "target_match_x_target_player_orientation"
    assert summary["test_seal"]["status"] == "sealed"
    assert summary["test_seal"]["used_for_method_selection"] is False
    assert summary["test_seal"]["excluded_test_point_rows"] == 1
    assert summary["test_seal"]["excluded_test_matches"] == 1
    assert summary["test_seal"]["counters"] == {field: 0 for field in TEST_ZERO_FIELDS}
    assert summary["fingerprint_contract_version"] == FINGERPRINT_CONTRACT_VERSION
    assert set(summary["artifact_manifest"]) == set(pipeline.ARTIFACT_ORDER[1:])
    assert len(summary["publication_fingerprint"]) == 64
    serialized = b"".join(payload for _, payload in payloads)
    assert b"NaN" not in serialized and b"Infinity" not in serialized and b"\r" not in serialized


def test_publish_round_trip_reopens_and_validates_all_five_artifacts(tmp_path):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    publish_tactical_pipeline_artifacts(result, paths)
    verify_persisted_artifacts(paths, result)
    expected = serialize_tactical_pipeline_artifacts(result)
    assert tuple(path.read_bytes() for path in pipeline._artifact_paths_tuple(paths)) == expected
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


def test_default_publication_reuses_validated_payloads_without_reconstructing_result(
    monkeypatch, tmp_path
):
    result = _publication_result()
    calls = 0
    original = pipeline.validate_tactical_pipeline_result

    def counted(value, **kwargs):
        nonlocal calls
        calls += 1
        return original(value, **kwargs)

    monkeypatch.setattr(pipeline, "validate_tactical_pipeline_result", counted)
    publish_tactical_pipeline_artifacts(result, _artifact_paths(tmp_path))
    assert calls == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("analysis_status", "unknown"),
        ("analysis_purpose", "validation"),
        ("descriptive_policy", "optimal"),
        ("minimum_labeled_attempts", 49),
        ("minimum_matches", 4),
        ("top_k_per_pattern", 2),
        ("scope", "surface"),
        ("patterns", ["P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]),
    ],
)
def test_resigned_summary_contract_mutations_are_rejected(tmp_path, field, value):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    publish_tactical_pipeline_artifacts(result, paths)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    summary[field] = value
    csv_payloads = tuple(
        zip(pipeline.ARTIFACT_ORDER[1:], (path.read_bytes() for path in pipeline._artifact_paths_tuple(paths)[1:]))
    )
    paths.summary.write_bytes(_resign_summary(summary, csv_payloads))
    with pytest.raises(TacticalPipelineContractError):
        verify_persisted_artifacts(paths, result)


@pytest.mark.parametrize("artifact_index", [1, 2, 3, 4])
def test_resigned_csv_mutations_are_rejected(tmp_path, artifact_index):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    publish_tactical_pipeline_artifacts(result, paths)
    path_values = pipeline._artifact_paths_tuple(paths)
    victim = path_values[artifact_index]
    payload = victim.read_bytes().replace(b"reconciled", b"manipulated", 1)
    victim.write_bytes(payload)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    csv_payloads = tuple(
        zip(pipeline.ARTIFACT_ORDER[1:], (path.read_bytes() for path in path_values[1:]))
    )
    paths.summary.write_bytes(_resign_summary(summary, csv_payloads))
    with pytest.raises(TacticalPipelineContractError):
        verify_persisted_artifacts(paths, result)


@pytest.mark.parametrize("failure_index", range(5))
@pytest.mark.parametrize("with_previous", [False, True])
def test_replace_failure_rolls_back_every_destination_and_cleans_staging(
    tmp_path, failure_index, with_previous
):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    path_values = pipeline._artifact_paths_tuple(paths)
    prior = tuple(f"prior-{index}".encode() for index in range(5))
    if with_previous:
        for path, payload in zip(path_values, prior):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    calls = 0

    def fail_at(source, destination):
        nonlocal calls
        if calls == failure_index:
            raise OSError("synthetic replace failure")
        calls += 1
        os.replace(source, destination)

    with pytest.raises(TacticalPipelineExecutionError, match="publication_failed"):
        publish_tactical_pipeline_artifacts(result, paths, replace_operation=fail_at)
    if with_previous:
        assert tuple(path.read_bytes() for path in path_values) == prior
    else:
        assert not any(path.exists() for path in path_values)
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


@pytest.mark.parametrize("failure_index", range(5))
def test_temporary_write_failure_leaves_no_partial_artifacts(tmp_path, failure_index):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    calls = 0

    def fail_at(path, payload):
        nonlocal calls
        if calls == failure_index:
            raise OSError("synthetic staging failure")
        calls += 1
        path.write_bytes(payload)

    with pytest.raises(TacticalPipelineExecutionError, match="publication_failed"):
        publish_tactical_pipeline_artifacts(result, paths, payload_writer=fail_at)
    assert not any(path.exists() for path in pipeline._artifact_paths_tuple(paths))
    assert not tuple(tmp_path.rglob("*.tmp"))
    assert not tuple(tmp_path.rglob("*.bak"))


def test_post_publish_verification_failure_restores_previous_bytes(tmp_path):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    path_values = pipeline._artifact_paths_tuple(paths)
    prior = tuple(f"old-{index}".encode() for index in range(5))
    for path, payload in zip(path_values, prior):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    calls = 0

    def fail_final(selected_paths, selected_result):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TacticalPipelineContractError("synthetic final verification failure")
        verify_persisted_artifacts(selected_paths, selected_result)

    with pytest.raises(TacticalPipelineExecutionError, match="publication_failed"):
        publish_tactical_pipeline_artifacts(result, paths, verifier=fail_final)
    assert tuple(path.read_bytes() for path in path_values) == prior
    assert not tuple(tmp_path.rglob("*.tmp"))
    assert not tuple(tmp_path.rglob("*.bak"))


def test_serialization_failure_happens_before_any_staging(tmp_path):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)

    def broken(_result):
        raise ValueError("individual details must not escape")

    with pytest.raises(TacticalPipelineExecutionError) as captured:
        publish_tactical_pipeline_artifacts(result, paths, serializer=broken)
    assert captured.value.stage == "serialization"
    assert captured.value.sanitized_message == "serialization_failed:ValueError"
    assert not tmp_path.exists() or not tuple(tmp_path.rglob("*"))


def test_not_available_publication_is_sealed_deterministic_and_header_only(tmp_path):
    base = _publication_result()
    error = TacticalPipelineExecutionError("target_evidence", "SyntheticError")
    unavailable = build_not_available_publication(error, base.test_seal, base.population)
    payloads = serialize_tactical_pipeline_unavailable_artifacts(unavailable)
    assert payloads == serialize_tactical_pipeline_unavailable_artifacts(unavailable)
    summary = json.loads(payloads[0])
    assert summary["analysis_status"] == "not_available"
    assert summary["reason_codes"] == ["analysis_failed_closed", "target_evidence_failed"]
    assert summary["failure"] == {
        "message": "target_evidence_failed:SyntheticError",
        "stage": "target_evidence",
        "type": "SyntheticError",
    }
    assert summary["test_seal"]["status"] == "sealed"
    assert summary["test_seal"]["excluded_test_point_rows"] == 1
    assert summary["test_seal"]["excluded_test_matches"] == 1
    assert summary["test_seal"]["counters"] == {field: 0 for field in TEST_ZERO_FIELDS}
    for payload, columns in zip(
        payloads[1:],
        (POPULATION_COLUMNS, AVAILABILITY_COLUMNS, RANKINGS_COLUMNS, DIAGNOSTICS_COLUMNS),
    ):
        frame = _read_serialized_csv(payload)
        assert frame.empty
        assert tuple(frame.columns) == columns
    paths = _artifact_paths(tmp_path)
    publish_tactical_pipeline_artifacts(unavailable, paths)
    verify_persisted_artifacts(paths, unavailable)


def test_publication_failure_does_not_replace_complete_result_with_not_available(tmp_path):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    prior = tuple(f"complete-before-{index}".encode() for index in range(5))
    for path, payload in zip(pipeline._artifact_paths_tuple(paths), prior):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def fail(*_args):
        raise OSError("publication failure")

    with pytest.raises(TacticalPipelineExecutionError) as captured:
        publish_tactical_pipeline_artifacts(result, paths, replace_operation=fail)
    assert captured.value.stage == "publication"
    assert tuple(path.read_bytes() for path in pipeline._artifact_paths_tuple(paths)) == prior


def test_performance_log_is_external_non_contractual_and_contains_only_aggregates():
    result = _publication_result()
    payload = performance_log_payload(result, 1.25)
    parsed = json.loads(payload)
    assert parsed["analysis_name"] == "tactical_recommender_pipeline"
    assert parsed["duration_seconds"] == 1.25
    assert set(parsed["stage_seconds"]) == set(pipeline.STAGE_ORDER)
    assert set(parsed["counts"]) == {"points", "attempts", "targets"}
    assert all(token not in payload for token in (b"match_id", b"point_number", b"Alice", b"4f17"))
    assert b"duration_seconds" not in prepare_tactical_pipeline_payloads(result)[0][1]


def test_upstream_validation_uses_ancestor_and_clean_path_rules(tmp_path):
    for _, relative in pipeline.UPSTREAM_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
    for _, relative, fingerprint in pipeline.UPSTREAM_ARTIFACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"publication_fingerprint": fingerprint}), encoding="utf-8"
        )
    calls = []

    def git_reader(root, *arguments):
        calls.append((root, arguments))
        return ""

    validate_upstream_ancestry(tmp_path, git_reader=git_reader)
    assert len(calls) == 2 * len(pipeline.UPSTREAM_PATHS) + len(pipeline.UPSTREAM_ARTIFACTS)
    module_calls = calls[: 2 * len(pipeline.UPSTREAM_PATHS)]
    assert all(
        arguments[:2] == ("merge-base", "--is-ancestor")
        for _, arguments in module_calls[::2]
    )
    assert all(arguments[:2] == ("status", "--short") for _, arguments in module_calls[1::2])
    assert all(
        arguments[:2] == ("status", "--short")
        for _, arguments in calls[2 * len(pipeline.UPSTREAM_PATHS) :]
    )


def test_upstream_validation_fails_closed_for_dirty_or_non_ancestor_source(tmp_path):
    for _, relative in pipeline.UPSTREAM_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
    for _, relative, fingerprint in pipeline.UPSTREAM_ARTIFACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"publication_fingerprint": fingerprint}), encoding="utf-8"
        )

    def dirty(_root, *arguments):
        return " M source" if arguments[:2] == ("status", "--short") else ""

    with pytest.raises(TacticalPipelineContractError, match="modificada"):
        validate_upstream_ancestry(tmp_path, git_reader=dirty)


def test_upstream_validation_rejects_a_fingerprint_substitution(tmp_path):
    for _, relative in pipeline.UPSTREAM_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
    for index, (_, relative, fingerprint) in enumerate(pipeline.UPSTREAM_ARTIFACTS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"publication_fingerprint": "0" * 64 if index == 0 else fingerprint}
            ),
            encoding="utf-8",
        )

    with pytest.raises(TacticalPipelineContractError, match="incoherente"):
        validate_upstream_ancestry(tmp_path, git_reader=lambda *_args: "")


def test_future_execution_has_one_blocked_entry_point_and_uses_derived_targets(
    monkeypatch, tmp_path
):
    result = _publication_result()
    calls = {"read": 0, "upstream": 0, "run": [], "publish": []}

    def reader(path):
        calls["read"] += 1
        assert path == pipeline.POINTS_PATH
        return _points()

    def upstream():
        calls["upstream"] += 1

    def run(
        points,
        targets,
        config,
        *,
        stage_timing_sink,
        operation_counters,
        progress_callback,
    ):
        calls["run"].append(
            (
                points,
                targets,
                config,
                stage_timing_sink,
                operation_counters,
                progress_callback,
            )
        )
        return result

    def publish(value, paths):
        calls["publish"].append((value, paths))

    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(pipeline, "run_tactical_recommender_pipeline", run)
    performance = tmp_path / "performance.json"
    artifact_paths = _artifact_paths(tmp_path / "artifacts")
    actual = pipeline.execute_authorized_real_pipeline(
        pipeline.POINTS_PATH,
        performance,
        artifact_paths=artifact_paths,
        source_reader=reader,
        upstream_validator=upstream,
        publisher=publish,
    )
    assert actual is result
    assert calls["read"] == calls["upstream"] == 1
    assert len(calls["run"]) == len(calls["publish"]) == 1
    assert calls["run"][0][1] is None
    assert calls["run"][0][2].expected_source_rows == pipeline.EXPECTED_SOURCE_ROWS
    assert calls["run"][0][3] == {}
    assert calls["run"][0][4] == {}
    assert callable(calls["run"][0][5])
    performance_payload = json.loads(performance.read_bytes())
    assert performance_payload["execution_status"] == "completed"
    assert performance_payload["current_stage"] == "publication"


def test_future_analysis_failure_publishes_only_sealed_not_available(
    monkeypatch, tmp_path
):
    published = []
    points = _points()
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(pipeline, "EXPECTED_SOURCE_ROWS", len(points))
    monkeypatch.setattr(pipeline, "EXPECTED_SOURCE_MATCHES", points.match_id.nunique())
    monkeypatch.setattr(pipeline, "EXPECTED_FIRST_DATE", pd.Timestamp("2020-01-01"))
    monkeypatch.setattr(pipeline, "EXPECTED_LAST_DATE", pd.Timestamp("2024-01-01"))

    def fail_after_seal(
        _points,
        _targets,
        _config,
        *,
        stage_timing_sink,
        operation_counters,
        progress_callback,
    ):
        stage_timing_sink["source_validation"] = 0.25
        stage_timing_sink["temporal_seal"] = 0.05
        stage_timing_sink["extraction_and_encoding"] = 0.75
        operation_counters.update(dict.fromkeys(pipeline.OPERATION_COUNTER_FIELDS, 0))
        progress_callback(
            {
                "execution_status": "running",
                "current_stage": "attempt_extraction_and_encoding",
                "completed_stages": ["source_validation", "temporal_sealing"],
                "stage_seconds": {
                    "source_validation": 0.25,
                    "temporal_sealing": 0.05,
                },
                "targets_total": 0,
                "targets_processed": 0,
                "targets_remaining": 0,
                "progress_fraction": 0.0,
                "attempts_processed": 0,
                "unique_cache_entries": 0,
                "observations_indexed": 0,
                "operation_counters": operation_counters,
            }
        )
        raise TacticalPipelineExecutionError("extraction_and_encoding", "SyntheticFailure")

    monkeypatch.setattr(pipeline, "run_tactical_recommender_pipeline", fail_after_seal)
    performance = tmp_path / "failure-performance.json"
    result = pipeline.execute_authorized_real_pipeline(
        pipeline.POINTS_PATH,
        performance,
        artifact_paths=_artifact_paths(tmp_path / "artifacts"),
        source_reader=lambda _path: points,
        upstream_validator=lambda: None,
        publisher=lambda value, paths: published.append((value, paths)),
    )
    assert result.state == "not_available"
    assert result.test_seal.test_status == "sealed"
    assert result.test_seal.counters == tuple((field, 0) for field in TEST_ZERO_FIELDS)
    assert result.population is None
    assert len(published) == 1 and published[0][0] is result
    log = json.loads(performance.read_bytes())
    assert log["execution_status"] == "not_available"
    assert log["failure"]["stage"] == "attempt_extraction_and_encoding"
    assert log["failure"]["type"] == "SyntheticFailure"
    assert log["stage_seconds"] == {
        "source_validation": 0.25,
        "temporal_sealing": 0.05,
    }


def test_future_publication_failure_is_logged_and_never_falls_back(
    monkeypatch, tmp_path
):
    result = _publication_result()
    calls = []
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        pipeline,
        "run_tactical_recommender_pipeline",
        lambda _points, _targets, _config, **_kwargs: result,
    )

    def fail_publication(value, _paths):
        calls.append(value)
        raise TacticalPipelineExecutionError("publication", "SyntheticFailure")

    performance = tmp_path / "publication-performance.json"
    with pytest.raises(TacticalPipelineExecutionError) as captured:
        pipeline.execute_authorized_real_pipeline(
            pipeline.POINTS_PATH,
            performance,
            artifact_paths=_artifact_paths(tmp_path / "artifacts"),
            source_reader=lambda _path: _points(),
            upstream_validator=lambda: None,
            publisher=fail_publication,
        )
    assert captured.value.stage == "publication"
    assert calls == [result]
    log = json.loads(performance.read_bytes())
    assert log["execution_status"] == "publication_failed"
    assert log["failure"]["stage"] == "publication"


def test_productive_route_builds_one_prepared_index_and_never_uses_reference_history(
    monkeypatch,
):
    calls = {"prepared_indexes": 0}
    original = pipeline._build_prepared_evidence_index

    def counted(*args, **kwargs):
        calls["prepared_indexes"] += 1
        return original(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("La ruta productiva alcanzo la referencia historica.")

    monkeypatch.setattr(pipeline, "_build_prepared_evidence_index", counted)
    monkeypatch.setattr(pipeline, "_select_target_history", forbidden)
    counters = {}
    result = run_tactical_recommender_pipeline(
        _points(), (_target(),), _config(), operation_counters=counters
    )
    assert calls == {"prepared_indexes": 1}
    assert result.prepared_evidence_index.observation_count == len(result.observations)
    assert counters == {
        "targets_processed": 1,
        "stream_queries": 152,
        "actor_relevant_observations_inspected": 0,
        "query_constructions": 1,
        "evidence_constructions": 1,
        "prioritizations": 1,
        "expensive_reconstructive_validations": 0,
        "artifact_serializations_or_fingerprints_in_target_loops": 0,
        "category_iterations": 26,
        "history_materializations": 0,
        "history_index_observations": len(result.observations),
    }


def test_productive_target_loop_has_no_index_rebuild_retry_or_serialization():
    tree = ast.parse(inspect.getsource(run_tactical_recommender_pipeline))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("_build_prepared_evidence_index") == 1
    assert "serialize_tactical_pipeline_artifacts" not in calls
    assert "tactical_pipeline_result_fingerprint" not in calls
    loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
    target_loop = next(
        node
        for node in loops
        if isinstance(node, ast.For) and ast.unparse(node.target) == "target"
        and ast.unparse(node.iter) == "ordered_targets"
    )
    target_calls = {
        node.func.id
        for node in ast.walk(target_loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_prepared_evidence_index" not in target_calls
    assert "_select_target_history" in target_calls  # rama de referencia solo sintetica
    assert "serialize_tactical_pipeline_artifacts" not in target_calls
    assert not any(isinstance(node, ast.While) for node in ast.walk(tree))


def test_progress_is_emitted_at_fifty_targets_and_final_target():
    targets = tuple(
        _target(match_id=f"progress-{number:03d}") for number in range(51)
    )
    snapshots = []
    counters = {}
    result = run_tactical_recommender_pipeline(
        _points(),
        targets,
        _config(),
        operation_counters=counters,
        progress_callback=lambda payload: snapshots.append(payload),
    )
    evidence_progress = [
        item["targets_processed"]
        for item in snapshots
        if item["current_stage"] == "evidence_and_prioritization"
    ]
    assert pipeline.PERFORMANCE_PROGRESS_TARGET_INTERVAL == 50
    assert evidence_progress == [0, 50, 51]
    assert result.targets_processed == counters["targets_processed"] == 51
    assert counters["query_constructions"] == counters["evidence_constructions"] == 1
    assert counters["prioritizations"] == 1
    assert counters["history_materializations"] == 0


def test_progress_contract_closes_every_analytical_stage_in_order():
    snapshots = []
    run_tactical_recommender_pipeline(
        _points(),
        (_target(),),
        _config(),
        progress_callback=lambda payload: snapshots.append(payload),
    )
    for stage in pipeline.PERFORMANCE_STAGE_ORDER[:-1]:
        assert any(
            item["current_stage"] == stage
            and stage in item["completed_stages"]
            for item in snapshots
        )
    final = snapshots[-1]
    assert final["current_stage"] == "aggregation_and_validation"
    assert final["completed_stages"] == list(pipeline.PERFORMANCE_STAGE_ORDER[:-1])
    assert final["targets_processed"] == final["targets_total"] == 1


def _synthetic_progress_snapshot(stage, *, processed=0, total=0):
    stage_index = pipeline.PERFORMANCE_STAGE_ORDER.index(stage)
    return {
        "execution_status": "running",
        "current_stage": stage,
        "completed_stages": list(pipeline.PERFORMANCE_STAGE_ORDER[:stage_index]),
        "stage_seconds": {
            item: 0.01 for item in pipeline.PERFORMANCE_STAGE_ORDER[:stage_index]
        },
        "targets_total": total,
        "targets_processed": processed,
        "targets_remaining": total - processed,
        "progress_fraction": 0.0 if total == 0 else processed / total,
        "attempts_processed": 3,
        "unique_cache_entries": 2,
        "observations_indexed": 3,
        "operation_counters": dict.fromkeys(pipeline.OPERATION_COUNTER_FIELDS, 0),
    }


@pytest.mark.parametrize(
    "stage",
    pipeline.PERFORMANCE_STAGE_ORDER[:-1],
)
def test_keyboard_interrupt_preserves_each_analytical_stage_without_publication(
    monkeypatch, tmp_path, stage
):
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    paths = _artifact_paths(tmp_path / "artifacts")
    performance = tmp_path / f"interrupt-{stage}.json"

    def reader(_path):
        if stage == "source_validation":
            raise KeyboardInterrupt
        return _points()

    def interrupted_run(_points, _targets, _config, **kwargs):
        processed = 37 if stage == "evidence_and_prioritization" else 0
        total = 100 if stage == "evidence_and_prioritization" else 0
        kwargs["progress_callback"](
            _synthetic_progress_snapshot(stage, processed=processed, total=total)
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run_tactical_recommender_pipeline", interrupted_run)
    with pytest.raises(KeyboardInterrupt):
        pipeline.execute_authorized_real_pipeline(
            pipeline.POINTS_PATH,
            performance,
            artifact_paths=paths,
            source_reader=reader,
            upstream_validator=lambda: None,
            publisher=lambda *_args: pytest.fail("No debe publicar tras Ctrl+C"),
        )
    payload = json.loads(performance.read_bytes())
    assert payload["execution_status"] == "interrupted"
    assert payload["reason_code"] == "manual_interrupt"
    assert payload["current_stage"] == stage
    if stage == "evidence_and_prioritization":
        assert (payload["targets_processed"], payload["targets_remaining"]) == (37, 63)
    assert not any(path.exists() for path in pipeline._artifact_paths_tuple(paths))
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_keyboard_interrupt_during_atomic_publication_rolls_back_and_cleans(tmp_path):
    result = _publication_result()
    paths = _artifact_paths(tmp_path)
    destinations = pipeline._artifact_paths_tuple(paths)
    prior = tuple(f"prior-{number}".encode() for number in range(5))
    for destination, payload in zip(destinations, prior):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    calls = 0

    def interrupt_third(source, destination):
        nonlocal calls
        if calls == 2:
            raise KeyboardInterrupt
        calls += 1
        os.replace(source, destination)

    with pytest.raises(KeyboardInterrupt):
        publish_tactical_pipeline_artifacts(
            result, paths, replace_operation=interrupt_third
        )
    assert tuple(path.read_bytes() for path in destinations) == prior
    assert not tuple(tmp_path.rglob("*.tmp"))
    assert not tuple(tmp_path.rglob("*.bak"))


def test_keyboard_interrupt_at_publication_is_logged_without_not_available(
    monkeypatch, tmp_path
):
    result = _publication_result()
    paths = _artifact_paths(tmp_path / "artifacts")
    performance = tmp_path / "publication-interrupt.json"
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        pipeline,
        "run_tactical_recommender_pipeline",
        lambda _points, _targets, _config, **_kwargs: result,
    )

    def interrupted_publication(_result, _paths):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pipeline.execute_authorized_real_pipeline(
            pipeline.POINTS_PATH,
            performance,
            artifact_paths=paths,
            source_reader=lambda _path: _points(),
            upstream_validator=lambda: None,
            publisher=interrupted_publication,
        )
    payload = json.loads(performance.read_bytes())
    assert payload["execution_status"] == "interrupted"
    assert payload["reason_code"] == "manual_interrupt"
    assert payload["current_stage"] == "publication"
    assert "failure" not in payload
    assert not any(path.exists() for path in pipeline._artifact_paths_tuple(paths))


def test_prepared_index_is_single_immutable_aggregate_representation():
    result = _publication_result()
    prepared = result.prepared_evidence_index
    assert prepared.observation_count == len(result.observations)
    for mapping in (
        prepared.all_by_server,
        prepared.all_by_returner,
        prepared.all_by_pair,
        prepared.eligible_by_server,
        prepared.eligible_by_returner,
        prepared.eligible_by_pair,
    ):
        assert type(mapping).__name__ == "mappingproxy"
        for series in mapping.values():
            assert type(series.dates) is tuple
            assert type(series.cumulative) is tuple
    assert not hasattr(prepared, "observations")


def test_incremental_performance_log_is_atomic_compact_and_aggregate_only(tmp_path):
    path = tmp_path / "performance.json"
    log = pipeline._IncrementalPerformanceLog(path)
    log.start()
    initial = json.loads(path.read_bytes())
    assert initial["execution_status"] == "running_preflight"
    assert initial["current_stage"] == "running_preflight"
    log(_synthetic_progress_snapshot("evidence_and_prioritization", processed=50, total=100))
    payload = path.read_bytes()
    parsed = json.loads(payload)
    assert parsed["targets_processed"] == parsed["targets_remaining"] == 50
    assert parsed["progress_fraction"] == 0.5
    assert b"NaN" not in payload and b"Infinity" not in payload
    assert all(
        token not in payload
        for token in (b"match_id", b"point_number", b"Alice", b"4f17", b"C:\\")
    )
    assert not tuple(tmp_path.glob("*.tmp"))
    assert payload.endswith(b"\n") and b"\n" not in payload[:-1]


def test_performance_log_exists_before_source_reader_is_called(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    performance = tmp_path / "preflight.json"

    def reader(_path):
        payload = json.loads(performance.read_bytes())
        assert payload["execution_status"] == "running"
        assert payload["current_stage"] == "source_validation"
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pipeline.execute_authorized_real_pipeline(
            pipeline.POINTS_PATH,
            performance,
            artifact_paths=_artifact_paths(tmp_path / "artifacts"),
            source_reader=reader,
            upstream_validator=lambda: None,
        )


def test_main_translates_keyboard_interrupt_to_exit_code_130(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        pipeline,
        "execute_authorized_real_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(SystemExit) as captured:
        pipeline.main(("--performance-log", str(tmp_path / "performance.json")))
    assert captured.value.code == 130


def test_prevalidated_serialization_is_byte_identical_and_skips_second_deep_validation(
    monkeypatch,
):
    result = _publication_result()
    calls = {"deep_validation": 0, "target_reconstructions": 0}
    original_validator = pipeline.validate_tactical_pipeline_result
    original_reconstruction = pipeline._build_indexed_matchup_evidence

    def counted_validator(value, **kwargs):
        calls["deep_validation"] += 1
        return original_validator(value, **kwargs)

    def counted_reconstruction(*args, **kwargs):
        calls["target_reconstructions"] += 1
        return original_reconstruction(*args, **kwargs)

    monkeypatch.setattr(pipeline, "validate_tactical_pipeline_result", counted_validator)
    monkeypatch.setattr(pipeline, "_build_indexed_matchup_evidence", counted_reconstruction)
    reference = serialize_tactical_pipeline_artifacts(result)
    reference_counts = dict(calls)
    optimized = pipeline._serialize_prevalidated_tactical_pipeline_artifacts(result)

    assert optimized == reference
    assert reference_counts == {
        "deep_validation": 1,
        "target_reconstructions": len(result.target_results),
    }
    assert calls == reference_counts


def test_publication_reuses_prevalidated_payloads_and_keeps_aggregate_checks(
    monkeypatch, tmp_path
):
    result = _publication_result()
    reference = serialize_tactical_pipeline_artifacts(result)
    aggregate_checks = 0
    original_aggregate_check = pipeline._verify_summary_contract

    def forbidden_deep_validation(*_args, **_kwargs):
        raise AssertionError("La publicacion reconstruyo el resultado analitico.")

    def counted_aggregate_check(*args, **kwargs):
        nonlocal aggregate_checks
        aggregate_checks += 1
        return original_aggregate_check(*args, **kwargs)

    monkeypatch.setattr(
        pipeline, "validate_tactical_pipeline_result", forbidden_deep_validation
    )
    monkeypatch.setattr(
        pipeline, "_verify_summary_contract", counted_aggregate_check
    )
    paths = _artifact_paths(tmp_path)
    publish_tactical_pipeline_artifacts(
        result,
        paths,
        prepared_payloads=pipeline._serialize_prevalidated_tactical_pipeline_artifacts(
            result
        ),
    )

    assert tuple(path.read_bytes() for path in pipeline._artifact_paths_tuple(paths)) == reference
    assert aggregate_checks == 2  # staging y destino final


def test_authorized_entry_point_uses_prevalidated_publication_path(
    monkeypatch, tmp_path
):
    result = _publication_result()
    deep_calls = 0

    def forbidden_deep_validation(*_args, **_kwargs):
        nonlocal deep_calls
        deep_calls += 1
        raise AssertionError("La publicacion repitio la validacion reconstructiva.")

    monkeypatch.setattr(pipeline, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        pipeline,
        "run_tactical_recommender_pipeline",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        pipeline, "validate_tactical_pipeline_result", forbidden_deep_validation
    )
    paths = _artifact_paths(tmp_path / "artifacts")

    actual = pipeline.execute_authorized_real_pipeline(
        pipeline.POINTS_PATH,
        tmp_path / "performance.json",
        artifact_paths=paths,
        source_reader=lambda _path: _points(),
        upstream_validator=lambda: None,
    )

    assert actual is result
    assert deep_calls == 0
    assert tuple(path.read_bytes() for path in pipeline._artifact_paths_tuple(paths)) == pipeline._serialize_prevalidated_tactical_pipeline_artifacts(result)


def _copy_published_p10_artifacts(destination: Path) -> TacticalPipelineArtifactPaths:
    source_paths = pipeline._artifact_paths_tuple(default_artifact_paths())
    copied = _artifact_paths(destination)
    for source, target in zip(source_paths, pipeline._artifact_paths_tuple(copied)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return copied


def _resign_persisted_p10(paths: TacticalPipelineArtifactPaths) -> None:
    path_values = pipeline._artifact_paths_tuple(paths)
    summary = json.loads(paths.summary.read_bytes())
    csv_payloads = tuple(zip(pipeline.ARTIFACT_ORDER[1:], (path.read_bytes() for path in path_values[1:])))
    paths.summary.write_bytes(_resign_summary(summary, csv_payloads))


@pytest.mark.integration
def test_published_p10_artifacts_are_frozen_and_reconciled_without_source_data(
    monkeypatch,
):
    paths = default_artifact_paths()
    if not all(path.is_file() for path in pipeline._artifact_paths_tuple(paths)):
        pytest.skip("Los cinco artefactos agregados P10 no existen localmente.")
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("La integracion artifact-only leyo Parquet."),
    )

    summary = verify_persisted_artifacts_only(paths)

    assert summary["analysis_status"] == "partially_available"
    assert summary["publication_fingerprint"] == pipeline.PUBLISHED_PUBLICATION_FINGERPRINT
    assert summary["population"] == {
        "attempt_count": 1_426_863,
        "development_matches": 5_993,
        "development_point_rows": 1_035_760,
        "excluded_test_matches": 1_531,
        "excluded_test_point_rows": 244_648,
        "first_attempt_count": 1_035_760,
        "observation_count": 1_426_863,
        "second_attempt_count": 391_103,
        "second_with_documented_first_fault": 382_143,
        "second_without_documented_first_fault": 8_960,
        "source_matches": 7_524,
        "source_point_rows": 1_280_408,
    }
    assert summary["targets"]["matches"] == 1_805
    assert summary["targets"]["orientations"] == 3_610
    assert summary["targets"]["folds"] == [
        {"fold": "validation_2020", "matches": 168, "orientations": 336},
        {"fold": "validation_2021", "matches": 371, "orientations": 742},
        {"fold": "validation_2022", "matches": 646, "orientations": 1_292},
        {"fold": "validation_2023", "matches": 620, "orientations": 1_240},
    ]
    assert summary["availability"] == {
        "available": 2_682,
        "partially_available": 116,
        "not_available": 812,
    }
    assert summary["scoring"] == {
        "candidates": 93_860,
        "scored": 33_944,
        "abstained": 59_916,
        "ranking_scope": "separate_by_pattern",
    }
    for payload, (_, size, digest) in zip(
        (path.read_bytes() for path in pipeline._artifact_paths_tuple(paths)),
        pipeline.PUBLISHED_ARTIFACT_CONTRACT,
    ):
        assert len(payload) == size
        assert sha256(payload).hexdigest().upper() == digest
    reports = pipeline.ROOT / "reports"
    assert not tuple(reports.rglob("*tactical_recommender_pipeline*.tmp"))
    assert not tuple(reports.rglob("*tactical_recommender_pipeline*.bak"))


@pytest.mark.integration
@pytest.mark.parametrize(
    "field,mutation",
    [
        ("minimum_labeled_attempts", 49),
        ("encoder_policy", "profile_only"),
        ("test_seal", "counter"),
        ("targets", "fold"),
    ],
)
def test_artifact_only_validator_rejects_resigned_summary_mutations(
    tmp_path, field, mutation
):
    source = default_artifact_paths()
    if not all(path.is_file() for path in pipeline._artifact_paths_tuple(source)):
        pytest.skip("Los cinco artefactos agregados P10 no existen localmente.")
    paths = _copy_published_p10_artifacts(tmp_path)
    summary = json.loads(paths.summary.read_bytes())
    if field == "test_seal":
        summary["test_seal"]["counters"]["test_rows_used"] = 1
    elif field == "targets":
        summary["targets"]["folds"][0]["orientations"] += 2
    else:
        summary[field] = mutation
    csv_payloads = tuple(
        zip(
            pipeline.ARTIFACT_ORDER[1:],
            (path.read_bytes() for path in pipeline._artifact_paths_tuple(paths)[1:]),
        )
    )
    paths.summary.write_bytes(_resign_summary(summary, csv_payloads))

    with pytest.raises(TacticalPipelineContractError):
        verify_persisted_artifacts_only(paths, enforce_frozen_bytes=False)


@pytest.mark.integration
@pytest.mark.parametrize(
    "artifact_index,column,value",
    [
        (1, "count", 1_280_409),
        (2, "coverage", 1.5),
        (3, "mean_score", "1.5"),
        (3, "rank_1", 9_999),
        (4, "observed", "manipulated"),
    ],
)
def test_artifact_only_validator_rejects_resigned_csv_semantic_mutations(
    tmp_path, artifact_index, column, value
):
    source = default_artifact_paths()
    if not all(path.is_file() for path in pipeline._artifact_paths_tuple(source)):
        pytest.skip("Los cinco artefactos agregados P10 no existen localmente.")
    paths = _copy_published_p10_artifacts(tmp_path)
    path_values = pipeline._artifact_paths_tuple(paths)
    columns = (
        POPULATION_COLUMNS,
        AVAILABILITY_COLUMNS,
        RANKINGS_COLUMNS,
        DIAGNOSTICS_COLUMNS,
    )[artifact_index - 1]
    frame = pipeline._read_csv_payload(path_values[artifact_index].read_bytes(), columns)
    frame.loc[0, column] = value
    path_values[artifact_index].write_bytes(pipeline._csv_bytes(frame, columns))
    _resign_persisted_p10(paths)

    with pytest.raises(TacticalPipelineContractError):
        verify_persisted_artifacts_only(paths, enforce_frozen_bytes=False)


@pytest.mark.integration
@pytest.mark.parametrize("artifact_index", [1, 2, 3, 4])
@pytest.mark.parametrize("mutation", ["reordered", "missing", "duplicated", "additional"])
def test_artifact_only_validator_rejects_csv_row_set_mutations(
    tmp_path, artifact_index, mutation
):
    source = default_artifact_paths()
    if not all(path.is_file() for path in pipeline._artifact_paths_tuple(source)):
        pytest.skip("Los cinco artefactos agregados P10 no existen localmente.")
    paths = _copy_published_p10_artifacts(tmp_path)
    path_values = pipeline._artifact_paths_tuple(paths)
    columns = (
        POPULATION_COLUMNS,
        AVAILABILITY_COLUMNS,
        RANKINGS_COLUMNS,
        DIAGNOSTICS_COLUMNS,
    )[artifact_index - 1]
    frame = pipeline._read_csv_payload(path_values[artifact_index].read_bytes(), columns)
    if mutation == "reordered":
        frame = pd.concat([frame.iloc[[1]], frame.iloc[[0]], frame.iloc[2:]], ignore_index=True)
    elif mutation == "missing":
        frame = frame.iloc[:-1].copy()
    elif mutation == "duplicated":
        frame = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    else:
        extra = frame.iloc[[-1]].copy()
        key_column = ("metric", "aggregation_level", "pattern_id", "check")[artifact_index - 1]
        extra.loc[:, key_column] = "unexpected"
        frame = pd.concat([frame, extra], ignore_index=True)
    path_values[artifact_index].write_bytes(pipeline._csv_bytes(frame, columns))
    _resign_persisted_p10(paths)

    with pytest.raises(TacticalPipelineContractError):
        verify_persisted_artifacts_only(paths, enforce_frozen_bytes=False)
