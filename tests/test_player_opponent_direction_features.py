from dataclasses import replace
import ast
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.analysis.player_opponent_direction_features as feature_module
from src.analysis.historical_profiles import SOURCE_COLUMNS
from src.analysis.player_opponent_direction_features import (
    COVERAGE_COLUMNS,
    COVERAGE_PATH,
    DIRECTIONS,
    FEATURE_COLUMNS,
    PAIR_STATES,
    PerformanceRecorder,
    STABILITY_COLUMNS,
    STABILITY_PATH,
    SUMMARY_PATH,
    FeatureValidationError,
    analyze_player_opponent_direction_features,
    serialize_artifacts,
    validate_feature_table,
    validate_performance_log_path,
    validate_result,
    validate_serialized_payloads,
    wilson_interval,
    write_artifacts,
)


def synthetic_points() -> pd.DataFrame:
    rows = [
        ["m1", 1, "2009-01-01", "Hard", 1, 1, "A", "B", "4*"],
        ["m1", 2, "2009-01-01", "Hard", 2, 1, "A", "B", "5#"],
        ["m2", 1, "2009-01-01", "Hard", 1, 2, "A", "C", "6n"],
        ["m2", 2, "2009-01-01", "Hard", 2, 2, "A", "C", "4*f"],
        ["m3", 1, "2009-01-02", "Hard", 1, 1, "A", "B", "5#"],
        ["m3", 2, "2009-01-02", "Hard", 2, 2, "A", "B", "6n"],
        ["m4", 1, "2009-01-02", "Clay", 1, 1, "A", "D", "4*"],
        ["m4", 2, "2009-01-02", "Clay", 2, 1, "A", "D", "5#"],
        ["m5", 1, "2010-01-03", "Clay", 1, 2, "A", "D", "6n"],
        ["m5", 2, "2010-01-03", "Clay", 2, 2, "A", "D", "4*"],
        ["m5", 3, "2010-01-03", "Clay", 1, 1, "A", "D", "0*"],
        ["m5", 4, "2010-01-03", "Clay", 1, 1, "A", "D", None],
        ["m6", 1, "2022-05-01", "Grass", 1, 1, "E", "F", "4*"],
    ]
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def scaled_synthetic_points(scale: int) -> pd.DataFrame:
    base = synthetic_points()
    frames = []
    for index in range(scale):
        frame = base.copy(deep=True)
        suffix = f"_b{index:02d}"
        frame["match_id"] = frame["match_id"].astype(str) + suffix
        frame["player_1"] = frame["player_1"].astype(str) + suffix
        frame["player_2"] = frame["player_2"].astype(str) + suffix
        frame["date"] = (
            pd.to_datetime(frame["date"]) + pd.to_timedelta(index * 3000, unit="D")
        ).astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def synthetic_result():
    return analyze_player_opponent_direction_features(synthetic_points())


def feature_row(result, match_id, player, direction, scope):
    return result.features[
        result.features["target_match_id"].eq(match_id)
        & result.features["target_player"].eq(player)
        & result.features["direction"].eq(direction)
        & result.features["scope"].eq(scope)
    ].iloc[0]


def snapshot_stub(result):
    return result.features[["target_match_id", "target_player"]].drop_duplicates().reset_index(drop=True)


def test_feature_cardinality_key_schema_and_deterministic_order(synthetic_result):
    features = synthetic_result.features
    assert features.columns.tolist() == FEATURE_COLUMNS
    assert len(features) == 6 * 2 * 3 * 2
    assert not features.duplicated(["target_match_id", "target_player", "direction", "scope"]).any()
    expected_tail = [(direction, scope) for direction in DIRECTIONS for scope in ("global", "surface")]
    assert list(features.iloc[:6][["direction", "scope"]].itertuples(index=False, name=None)) == expected_tail
    assert features.groupby(["target_match_id", "direction", "scope"]).size().eq(2).all()


def test_common_outcome_roles_numerators_denominators_and_complement_are_manual(synthetic_result):
    row = feature_row(synthetic_result, "m3", "A", "wide", "global")
    assert (row.server_direction_matches, row.server_direction_points, row.server_direction_wins) == (1, 1, 1)
    assert row.server_direction_raw_rate == 1.0
    assert (row.opponent_direction_matches, row.opponent_direction_points) == (1, 1)
    assert (row.opponent_allowed_server_wins, row.opponent_return_wins) == (1, 0)
    assert row.opponent_allowed_server_raw_rate == 1.0
    assert (row.population_direction_points, row.population_direction_server_wins) == (2, 2)
    assert row.population_direction_raw_rate == 1.0
    assert row.server_opponent_gap == 0.0


def test_server_two_and_direction_specific_matches_and_dates(synthetic_result):
    b = feature_row(synthetic_result, "m3", "B", "body", "global")
    assert (b.server_direction_matches, b.server_direction_points, b.server_direction_wins) == (1, 1, 0)
    assert b.server_direction_first_history_date == pd.Timestamp("2009-01-01")
    assert b.server_direction_last_history_date == pd.Timestamp("2009-01-01")
    assert (b.opponent_direction_points, b.opponent_allowed_server_wins) == (1, 0)
    assert b.opponent_direction_first_history_date == pd.Timestamp("2009-01-01")


def test_direction_matches_counts_a_match_once_with_multiple_points():
    repeated = pd.concat(
        [
            synthetic_points(),
            pd.DataFrame(
                [["m1", 3, "2009-01-01", "Hard", 1, 2, "A", "B", "4*"]],
                columns=SOURCE_COLUMNS,
            ),
        ],
        ignore_index=True,
    )
    result = analyze_player_opponent_direction_features(repeated)
    row = feature_row(result, "m3", "A", "wide", "global")
    assert row.server_direction_matches == 1
    assert row.server_direction_points == 2
    assert row.server_direction_wins == 1
    assert row.server_direction_first_history_date == pd.Timestamp("2009-01-01")
    assert row.server_direction_last_history_date == pd.Timestamp("2009-01-01")


def test_daily_blocking_time_components_and_next_day_population_baseline(synthetic_result):
    same_day = feature_row(synthetic_result, "m4", "A", "wide", "surface")
    assert same_day.server_direction_points == 0
    assert same_day.population_direction_points == 0
    next_day = feature_row(synthetic_result, "m5", "A", "wide", "surface")
    assert next_day.server_direction_points == 1
    assert next_day.population_direction_points == 1
    assert next_day.population_direction_first_history_date == pd.Timestamp("2009-01-02")

    timed = synthetic_points().copy()
    timed.loc[timed["match_id"].eq("m1"), "date"] = "2009-01-01 10:00"
    timed.loc[timed["match_id"].eq("m2"), "date"] = "2009-01-01 18:00:00"
    result = analyze_player_opponent_direction_features(timed)
    pdt.assert_frame_equal(result.features, synthetic_result.features, check_exact=True)


def test_row_order_future_addition_and_future_mutation_do_not_change_past(synthetic_result):
    shuffled = synthetic_points().sample(frac=1, random_state=17).reset_index(drop=True)
    pdt.assert_frame_equal(
        analyze_player_opponent_direction_features(shuffled).features,
        synthetic_result.features,
        check_exact=True,
    )
    future = pd.DataFrame([["m7", 1, "2025-01-01", "Hard", 1, 1, "A", "Z", "5#"]], columns=SOURCE_COLUMNS)
    expanded = analyze_player_opponent_direction_features(pd.concat([synthetic_points(), future], ignore_index=True))
    existing = expanded.features[expanded.features["target_match_id"].isin([f"m{i}" for i in range(1, 7)])]
    pdt.assert_frame_equal(existing.reset_index(drop=True), synthetic_result.features, check_exact=True)
    changed = pd.concat([synthetic_points(), future], ignore_index=True)
    changed.loc[changed["match_id"].eq("m7"), ["server", "point_winner", "second_serve"]] = [2, 2, "6n"]
    changed_result = analyze_player_opponent_direction_features(changed)
    changed_past = changed_result.features[changed_result.features["target_match_id"].isin([f"m{i}" for i in range(1, 7)])]
    pdt.assert_frame_equal(changed_past.reset_index(drop=True), synthetic_result.features, check_exact=True)


def test_zero_history_global_surface_separation_shares_and_states(synthetic_result):
    first = feature_row(synthetic_result, "m1", "A", "wide", "global")
    assert first.server_direction_points == 0
    assert first.server_direction_raw_rate is None or pd.isna(first.server_direction_raw_rate)
    assert first.server_direction_wilson_low is None or pd.isna(first.server_direction_wilson_low)
    assert first.server_direction_share is None or pd.isna(first.server_direction_share)
    assert first.pair_evidence_state == "neither_observed"
    global_row = feature_row(synthetic_result, "m4", "A", "wide", "global")
    surface_row = feature_row(synthetic_result, "m4", "A", "wide", "surface")
    assert global_row.server_direction_points == 1
    assert surface_row.server_direction_points == 0
    assert global_row.server_evidence_state == "observed"
    assert surface_row.server_evidence_state == "no_history"
    m5 = synthetic_result.features[
        synthetic_result.features["target_match_id"].eq("m5")
        & synthetic_result.features["target_player"].eq("A")
        & synthetic_result.features["scope"].eq("global")
    ]
    assert m5.server_direction_share.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("wins", "points", "expected"),
    [(0, 0, (None, None)), (0, 10, (0.0, 0.2775327998628892)), (5, 10, (0.236593090512564, 0.7634069094874361)), (10, 10, (0.7224672001371107, 1.0))],
)
def test_wilson_matches_independent_numeric_references(wins, points, expected):
    actual = wilson_interval(wins, points)
    if expected[0] is None:
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, abs=1e-15)


def test_vectorized_wilson_matches_independent_scalar_references():
    wins = pd.Series([0, 1, 5, 10, 500], dtype="int64")
    points = pd.Series([0, 1, 10, 10, 1000], dtype="int64")
    low, high = wilson_interval(wins, points)
    expected_low = [np.nan, 0.20654931437723745, 0.236593090512564, 0.7224672001371107, 0.4690696003681043]
    expected_high = [np.nan, 1.0, 0.7634069094874361, 1.0, 0.5309303996318957]
    np.testing.assert_allclose(low, expected_low, rtol=0, atol=1e-15, equal_nan=True)
    np.testing.assert_allclose(high, expected_high, rtol=0, atol=1e-15, equal_nan=True)


def test_evidence_states_gaps_and_algebraic_identity(synthetic_result):
    assert set(synthetic_result.features.server_evidence_state) == {"no_history", "observed"}
    assert set(synthetic_result.features.opponent_evidence_state) == {"no_history", "observed"}
    assert set(synthetic_result.features.pair_evidence_state).issubset(PAIR_STATES)
    summary_states = synthetic_result.summary["evidence_states"]
    for row in summary_states:
        assert sum(row[state] for state in PAIR_STATES) == row["total"]
        assert sum(row[f"{state}_share"] for state in PAIR_STATES) == pytest.approx(1.0)
    comparable = synthetic_result.features.dropna(subset=[
        "server_excess_vs_population", "opponent_allowed_excess_vs_population", "server_opponent_gap"
    ])
    error = (
        comparable.server_opponent_gap
        - comparable.server_excess_vs_population
        + comparable.opponent_allowed_excess_vs_population
    ).abs().max()
    assert error <= 1e-15


def test_coverage_both_complete_matches_and_thresholds_are_exact(synthetic_result):
    coverage = synthetic_result.coverage
    assert coverage.columns.tolist() == COVERAGE_COLUMNS
    assert len(coverage) == 6 * 3 * 2 * 6 * 7
    all_rows = coverage[
        coverage.scope.eq("both_global") & coverage.direction.eq("wide")
        & coverage.metric.eq("direction_points") & coverage.threshold.eq(0)
        & coverage.stratum_type.eq("overall")
    ].iloc[0]
    assert (all_rows.eligible_rows, all_rows.total_rows, all_rows.complete_matches, all_rows.total_matches) == (12, 12, 6, 6)
    observed = coverage[
        coverage.scope.eq("both_global") & coverage.direction.eq("wide")
        & coverage.metric.eq("direction_matches") & coverage.threshold.eq(1)
        & coverage.stratum_type.eq("overall")
    ].iloc[0]
    assert (observed.eligible_rows, observed.total_rows, observed.complete_matches, observed.covered_players) == (2, 12, 0, 1)


def test_stability_quantiles_extremes_and_undefined_correlation_are_exact(synthetic_result):
    stability = synthetic_result.stability
    assert stability.columns.tolist() == STABILITY_COLUMNS
    assert len(stability) == 2 * 2 * 3 * 6 * 7
    row = stability[
        stability.role.eq("server") & stability.scope.eq("global")
        & stability.direction.eq("wide") & stability.threshold_points.eq(0)
        & stability.stratum_type.eq("overall")
    ].iloc[0]
    assert row.observed_rows == 3
    assert (row.minimum_points, row.median_points, row.points_q1, row.points_q3) == (1, 1.0, 1.0, 1.5)
    assert (row.median_raw_rate, row.one_rate_rows, row.one_rate_share) == (1.0, 3, 1.0)
    comparison = synthetic_result.summary["global_surface_comparison"]
    assert len(comparison) == 2 * 3 * 7
    constant = next(item for item in comparison if item["role"] == "server" and item["direction"] == "wide" and item["target_surface"] == "Overall" and item["derived_period"] == "Overall")
    assert constant["correlation"] is None


def test_temporal_validation_has_21_rules_and_deterministic_examples(synthetic_result):
    diagnostics = synthetic_result.summary["temporal_diagnostics"]
    assert len(diagnostics["rows"]) == 21
    assert diagnostics["total_violations"] == 0
    assert diagnostics["examples"] == []
    broken = synthetic_result.features.copy(deep=True)
    index = broken["server_direction_points"].gt(0).idxmax()
    broken.loc[index, "server_direction_last_history_date"] = broken.loc[index, "target_date"]
    with pytest.raises(FeatureValidationError) as error:
        validate_feature_table(broken, snapshot_stub(synthetic_result))
    assert "last_on_or_after_target" in {row["reason_code"] for row in error.value.diagnostics if row["violations"]}
    assert error.value.examples[0]["target_match_id"] == broken.loc[index, "target_match_id"]


@pytest.mark.parametrize("prefix", ["server", "opponent", "population"])
@pytest.mark.parametrize(
    "reason_code",
    [
        "zero_first_present", "zero_last_present", "positive_first_missing",
        "positive_last_missing", "first_after_last", "last_on_or_after_target",
        "first_on_or_after_target",
    ],
)
def test_each_temporal_reason_code_is_detected_independently(synthetic_result, prefix, reason_code):
    columns = {
        "server": (
            "server_direction_points", "server_direction_first_history_date",
            "server_direction_last_history_date",
        ),
        "opponent": (
            "opponent_direction_points", "opponent_direction_first_history_date",
            "opponent_direction_last_history_date",
        ),
        "population": (
            "population_direction_points", "population_direction_first_history_date",
            "population_direction_last_history_date",
        ),
    }
    points_column, first_column, last_column = columns[prefix]
    broken = synthetic_result.features.copy(deep=True)
    if reason_code.startswith("zero_"):
        index = broken[points_column].eq(0).idxmax()
    else:
        index = broken[points_column].gt(0).idxmax()
    target = broken.loc[index, "target_date"]
    if reason_code == "zero_first_present":
        broken.loc[index, first_column] = target - pd.Timedelta(days=1)
    elif reason_code == "zero_last_present":
        broken.loc[index, last_column] = target - pd.Timedelta(days=1)
    elif reason_code == "positive_first_missing":
        broken.loc[index, first_column] = pd.NaT
    elif reason_code == "positive_last_missing":
        broken.loc[index, last_column] = pd.NaT
    elif reason_code == "first_after_last":
        broken.loc[index, first_column] = target - pd.Timedelta(days=1)
        broken.loc[index, last_column] = target - pd.Timedelta(days=2)
    elif reason_code == "last_on_or_after_target":
        broken.loc[index, last_column] = target
    else:
        broken.loc[index, first_column] = target
    with pytest.raises(FeatureValidationError) as error:
        validate_feature_table(broken, snapshot_stub(synthetic_result))
    detected = {
        row["reason_code"]
        for row in error.value.diagnostics
        if row["prefix"] == prefix and row["violations"]
    }
    assert reason_code in detected
    matching_examples = [
        row for row in error.value.examples
        if row["prefix"] == prefix and row["reason_code"] == reason_code
    ]
    assert matching_examples
    assert matching_examples[0]["target_match_id"] == broken.loc[index, "target_match_id"]


@pytest.mark.parametrize("column", ["server", "point_winner"])
@pytest.mark.parametrize("bad_value", [True, False, "1", "2", 1.0, 2.0])
def test_coercible_server_and_winner_domains_are_rejected(column, bad_value):
    points = synthetic_points()
    points[column] = pd.Series([bad_value] * len(points), dtype="object")
    with pytest.raises(ValueError, match=f"Dominio invalido en {column}"):
        analyze_player_opponent_direction_features(points)


def test_mixed_dates_are_explicit_and_invalid_or_null_dates_are_rejected(synthetic_result):
    mixed = synthetic_points().copy()
    mixed["date"] = mixed["date"].astype("object")
    mixed.loc[mixed.match_id.eq("m1"), "date"] = "2009-01-01 10:00:00"
    mixed.loc[mixed.match_id.eq("m2"), "date"] = pd.Timestamp("2009-01-01 18:00")
    pdt.assert_frame_equal(analyze_player_opponent_direction_features(mixed).features, synthetic_result.features, check_exact=True)
    for value in (None, "not-a-date"):
        broken = synthetic_points(); broken.loc[0, "date"] = value
        with pytest.raises(ValueError, match="date contiene fechas"):
            analyze_player_opponent_direction_features(broken)


def test_complete_summary_and_csv_manipulation_are_rejected(synthetic_result):
    broken_summary = json.loads(json.dumps(synthetic_result.summary))
    broken_summary["feature_table_contract"]["rows"] -= 1
    broken = replace(synthetic_result, summary=broken_summary, publication_fingerprint="")
    with pytest.raises(ValueError, match="resumen completo"):
        validate_result(broken, snapshot_stub(synthetic_result))
    summary_payload, coverage_payload, stability_payload = serialize_artifacts(synthetic_result)
    manipulated = coverage_payload.replace(b",12,12,", b",11,12,", 1)
    with pytest.raises(ValueError, match="coverage"):
        validate_serialized_payloads(synthetic_result, summary_payload, manipulated, stability_payload)


def test_post_validation_mutation_is_rejected(tmp_path, synthetic_result):
    mutated = replace(synthetic_result, coverage=synthetic_result.coverage.copy(deep=True))
    mutated.coverage.loc[0, "eligible_rows"] += 1
    with pytest.raises(ValueError, match="modificado"):
        write_artifacts(mutated, tmp_path / "summary.json", tmp_path / "coverage.csv", tmp_path / "stability.csv")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure_position", [2, 3])
def test_atomic_publication_rolls_back_second_and_third_replacement(tmp_path, monkeypatch, synthetic_result, failure_position):
    paths = [tmp_path / "summary.json", tmp_path / "coverage.csv", tmp_path / "stability.csv"]
    originals = [b"old summary\n", b"old coverage\n", b"old stability\n"]
    for path, payload in zip(paths, originals):
        path.write_bytes(payload)
    real_replace = feature_module.os.replace
    calls = 0
    failed = False

    def fail_once(source, target):
        nonlocal calls, failed
        calls += 1
        if calls == failure_position and not failed:
            failed = True
            raise OSError(f"failure at replacement {failure_position}")
        return real_replace(source, target)

    monkeypatch.setattr(feature_module.os, "replace", fail_once)
    with pytest.raises(OSError, match="failure at replacement"):
        write_artifacts(synthetic_result, *paths)
    assert [path.read_bytes() for path in paths] == originals
    assert not list(tmp_path.glob("*.tmp"))


def test_serialization_is_deterministic_index_free_and_contains_no_inference_or_recommendation(tmp_path, synthetic_result):
    first = serialize_artifacts(synthetic_result)
    second = serialize_artifacts(synthetic_result)
    assert first == second
    for payload, columns in zip(first[1:], (COVERAGE_COLUMNS, STABILITY_COLUMNS)):
        loaded = pd.read_csv(pd.io.common.BytesIO(payload))
        assert loaded.columns.tolist() == columns
        assert not any(column.startswith("Unnamed") for column in loaded.columns)
    text = first[0].decode("utf-8").lower()
    for forbidden in ('"p_value"', '"odds_ratio"', '"recommendations"', '"selected_threshold"'):
        assert forbidden not in text
    json.loads(first[0], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_performance_instrumentation_is_optional_structured_and_hash_neutral(tmp_path):
    default = analyze_player_opponent_direction_features(synthetic_points())
    assert default.performance_metrics == ()
    log_path = tmp_path / "performance.json"
    callback_records = []
    recorder = PerformanceRecorder(enabled=True, callback=callback_records.append, log_path=log_path)
    measured = analyze_player_opponent_direction_features(synthetic_points(), recorder=recorder)
    assert measured.publication_fingerprint == default.publication_fingerprint
    phases = [row["phase"] for row in measured.performance_metrics]
    assert phases == [
        "prepare_eligible_points", "build_base_snapshots",
        "build_directional_daily_aggregates", "build_population_baseline",
        "expand_directions_scopes", "calculate_rates_wilson_gaps",
        "temporal_validation", "calculate_coverage", "calculate_stability",
        "global_surface_comparison", "build_summary", "validate_result",
    ]
    assert callback_records == list(measured.performance_metrics)
    assert all(row["duration_seconds"] >= 0 and "rows" in row for row in callback_records)
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["last_completed_phase"] == "validate_result"
    assert "timestamp" not in log_path.read_text(encoding="utf-8").lower()

    publication_recorder = PerformanceRecorder(enabled=True)
    write_artifacts(
        measured,
        tmp_path / "summary.json",
        tmp_path / "coverage.csv",
        tmp_path / "stability.csv",
        recorder=publication_recorder,
    )
    assert [row["phase"] for row in publication_recorder.records] == [
        "serialize_payloads", "publish_artifacts"
    ]


def test_performance_log_must_be_outside_repository(tmp_path):
    validate_performance_log_path(None)
    validate_performance_log_path(tmp_path / "performance.json")
    with pytest.raises(ValueError, match="fuera del repositorio"):
        validate_performance_log_path(feature_module.ROOT / "reports" / "performance.json")


def test_performance_structure_uses_one_snapshot_build_and_bounded_aggregations():
    module_tree = ast.parse(Path(feature_module.__file__).read_text(encoding="utf-8"))
    snapshot_calls = [
        node for node in ast.walk(module_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "build_historical_snapshots"
    ]
    assert len(snapshot_calls) == 1
    coverage_source = inspect.getsource(feature_module.build_coverage)
    stability_source = inspect.getsource(feature_module.build_stability)
    assert "iterrows" not in coverage_source + stability_source
    assert "itertuples" not in coverage_source + stability_source
    assert coverage_source.count(".groupby(") <= 4
    assert stability_source.count(".groupby(") == 1
    expansion_source = inspect.getsource(feature_module.build_direction_features)
    target_loop = expansion_source.split("for target in daily_targets.itertuples", 1)[1].split(
        "for event in server_by_date", 1
    )[0]
    assert ".loc[" not in target_loop
    assert "features[" not in target_loop


def test_scaled_benchmark_preserves_results_without_quadratic_growth():
    expected_fingerprints = {
        1: "787cc3168ecb690bb15c5b3e55bb70c3b49dfe0bb8b4f067398dc4abc21cfa8f",
        4: "565cc67c1a7e8b50a84e80bdf8a83332b2c9a0e723024f51f7bb9633b675957e",
        16: "c5f434fe673e7b5b64493b91ce120e2cb2f69752c4082549694743ebe80f1354",
    }
    durations = []
    for scale in (1, 4, 16):
        started = time.perf_counter()
        result = analyze_player_opponent_direction_features(scaled_synthetic_points(scale))
        durations.append(time.perf_counter() - started)
        assert result.publication_fingerprint == expected_fingerprints[scale]
        assert len(result.features) == scale * 6 * 2 * 3 * 2
    # Four times more data must stay comfortably below the 16x signature of
    # quadratic work; the broad bound avoids sensitivity to ordinary CPU noise.
    assert durations[1] / durations[0] < 8
    assert durations[2] / durations[1] < 8


@pytest.mark.integration
def test_real_artifacts_exact_population_contract_and_no_leakage():
    parquet = Path("data/processed/points_enriched.parquet")
    if not parquet.exists() or not all(path.exists() for path in (SUMMARY_PATH, COVERAGE_PATH, STABILITY_PATH)):
        pytest.skip("Requiere Parquet y artefactos locales.")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    coverage = pd.read_csv(COVERAGE_PATH)
    stability = pd.read_csv(STABILITY_PATH)
    assert summary["source"]["point_rows"] == 1_280_408
    assert (summary["source"]["matches"], summary["source"]["players"]) == (7_524, 1_002)
    population = summary["eligible_history_population"]
    assert (population["eligible_points"], population["server_wins"]) == (481_190, 245_683)
    assert (population["wide_points"], population["wide_server_wins"]) == (176_757, 91_440)
    assert (population["body_points"], population["body_server_wins"]) == (165_841, 84_299)
    assert (population["t_points"], population["t_server_wins"]) == (138_592, 69_944)
    assert population["excluded_direction_0"] == 148
    assert population["excluded_without_recognized_direction"] == 170
    assert summary["feature_table_contract"]["rows"] == 90_288
    assert summary["reconciliations"]["unique_feature_keys"] == 90_288
    assert summary["reconciliations"]["two_orientations_per_match"] is True
    assert summary["temporal_diagnostics"]["total_violations"] == 0
    assert len(summary["temporal_diagnostics"]["rows"]) == 21
    assert coverage.columns.tolist() == COVERAGE_COLUMNS and len(coverage) == 1_512
    assert stability.columns.tolist() == STABILITY_COLUMNS and len(stability) == 504
