import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

import src.analysis.historical_profiles as historical_profiles
from src.analysis.historical_profiles import (
    COVERAGE_COLUMNS,
    COVERAGE_PATH,
    SNAPSHOT_COLUMNS,
    SOURCE_COLUMNS,
    SUMMARY_PATH,
    HistoricalProfileResult,
    analyze_historical_profiles,
    build_coverage_table,
    prepare_historical_population,
    validate_published_contract,
    validate_snapshots,
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


@pytest.fixture(scope="module")
def synthetic_result():
    return analyze_historical_profiles(synthetic_points())


def _snapshot(result, match_id: str, player: str) -> pd.Series:
    return result.snapshots[
        result.snapshots["target_match_id"].eq(match_id)
        & result.snapshots["target_player"].eq(player)
    ].iloc[0]


def test_role_identity_wins_directions_and_distinct_matches_are_exact(synthetic_result):
    a_m3 = _snapshot(synthetic_result, "m3", "A")
    assert a_m3["opponent"] == "B"
    assert a_m3["server_global_matches"] == 2
    assert a_m3["server_global_points"] == 2
    assert a_m3["server_global_wins"] == 1
    assert a_m3["server_global_wide_points"] == 1
    assert a_m3["server_global_wide_wins"] == 1
    assert a_m3["server_global_body_points"] == 0
    assert a_m3["server_global_t_points"] == 1
    assert a_m3["server_global_t_wins"] == 0
    assert a_m3["opponent_return_global_matches"] == 1
    assert a_m3["opponent_return_global_points"] == 1
    assert a_m3["opponent_return_global_server_wins"] == 1
    assert a_m3["opponent_return_global_return_wins"] == 0

    b_m3 = _snapshot(synthetic_result, "m3", "B")
    assert b_m3["server_global_points"] == 1
    assert b_m3["server_global_body_points"] == 1
    assert b_m3["server_global_wins"] == 0
    assert b_m3["opponent_return_global_points"] == 2
    assert b_m3["opponent_return_global_server_wins"] == 1
    assert b_m3["opponent_return_global_return_wins"] == 1


def test_same_date_is_simultaneous_and_next_date_updates_all_daily_matches(synthetic_result):
    a_m3 = _snapshot(synthetic_result, "m3", "A")
    a_m4 = _snapshot(synthetic_result, "m4", "A")
    for column in [c for c in SNAPSHOT_COLUMNS if c.startswith("server_global_")]:
        assert a_m3[column] == a_m4[column]
    assert a_m4["server_surface_points"] == 0
    assert pd.isna(a_m4["server_surface_first_history_date"])

    a_m5 = _snapshot(synthetic_result, "m5", "A")
    assert a_m5["server_global_matches"] == 4
    assert a_m5["server_global_points"] == 4
    assert a_m5["server_surface_matches"] == 1
    assert a_m5["server_surface_points"] == 1
    assert a_m5["server_global_last_history_date"] == pd.Timestamp("2009-01-02")
    assert a_m5["server_global_last_history_date"] < a_m5["target_date"]


def test_time_components_never_create_intraday_ordering(synthetic_result):
    timed = synthetic_points().copy()
    timed["date"] = pd.to_datetime(timed["date"])
    timed.loc[timed["match_id"].eq("m1"), "date"] = pd.Timestamp("2009-01-01 10:00:00")
    timed.loc[timed["match_id"].eq("m2"), "date"] = pd.Timestamp("2009-01-01 18:00:00")
    result = analyze_historical_profiles(timed)
    pdt.assert_frame_equal(result.snapshots, synthetic_result.snapshots, check_exact=True)


def test_target_match_never_self_references_and_new_players_are_zero(synthetic_result):
    first_day = synthetic_result.snapshots[
        synthetic_result.snapshots["target_date"].eq(pd.Timestamp("2009-01-01"))
    ]
    count_columns = [
        column
        for column in SNAPSHOT_COLUMNS
        if column not in {"target_match_id", "target_date", "target_player", "opponent", "target_surface"}
        and not column.endswith("history_date")
    ]
    assert first_day[count_columns].to_numpy().sum() == 0
    new_player = _snapshot(synthetic_result, "m6", "E")
    assert new_player[count_columns].to_numpy().sum() == 0


def test_new_players_and_absent_allowed_strata_produce_finite_zero_coverage():
    one_match = synthetic_points().iloc[[0]].copy()
    result = analyze_historical_profiles(one_match)
    assert len(result.snapshots) == 2
    absent = result.coverage[
        result.coverage["stratum_type"].eq("surface")
        & result.coverage["stratum"].isin(["Clay", "Grass"])
    ]
    assert absent["total_snapshots"].eq(0).all()
    assert absent["total_matches"].eq(0).all()
    assert absent["coverage_rate"].eq(0.0).all()
    assert absent["complete_match_rate"].eq(0.0).all()


def test_two_unique_snapshots_per_match_and_shared_point_join(synthetic_result):
    snapshots = synthetic_result.snapshots
    assert len(snapshots) == 12
    assert snapshots.columns.tolist() == SNAPSHOT_COLUMNS
    assert not snapshots.duplicated(["target_match_id", "target_player"]).any()
    assert snapshots.groupby("target_match_id").size().eq(2).all()

    points = synthetic_points()
    point_roles = pd.concat(
        [
            points.assign(target_player=points["player_1"]),
            points.assign(target_player=points["player_2"]),
        ],
        ignore_index=True,
    )
    joined = point_roles.merge(
        snapshots,
        left_on=["match_id", "target_player"],
        right_on=["target_match_id", "target_player"],
        validate="many_to_one",
    )
    assert len(joined) == 2 * len(points)
    historical_columns = [
        column for column in SNAPSHOT_COLUMNS if column not in {"target_match_id", "target_player"}
    ]
    for column in historical_columns:
        assert joined.groupby(["match_id", "target_player"])[column].nunique(dropna=False).le(1).all()


def test_row_shuffle_does_not_change_snapshots_or_coverage(synthetic_result):
    shuffled = synthetic_points().sample(frac=1, random_state=732).reset_index(drop=True)
    result = analyze_historical_profiles(shuffled)
    pdt.assert_frame_equal(result.snapshots, synthetic_result.snapshots, check_exact=True)
    pdt.assert_frame_equal(result.coverage, synthetic_result.coverage, check_exact=True)
    assert result.summary == synthetic_result.summary


def test_future_addition_does_not_change_existing_snapshots(synthetic_result):
    future = pd.DataFrame(
        [["m7", 1, "2025-01-01", "Hard", 1, 1, "A", "Z", "5#"]],
        columns=SOURCE_COLUMNS,
    )
    expanded = analyze_historical_profiles(pd.concat([synthetic_points(), future], ignore_index=True))
    existing = expanded.snapshots[expanded.snapshots["target_match_id"].isin([f"m{i}" for i in range(1, 7)])]
    pdt.assert_frame_equal(existing.reset_index(drop=True), synthetic_result.snapshots, check_exact=True)


def test_future_mutation_does_not_change_past_snapshots(synthetic_result):
    changed = synthetic_points().copy()
    mask = changed["match_id"].eq("m5") & changed["point_number"].eq(1)
    # A and D already appear in earlier snapshots, so this exercises a future
    # mutation of players with genuine prior history.
    changed.loc[mask, ["server", "point_winner", "second_serve"]] = [2, 1, "4*"]
    mutated = analyze_historical_profiles(changed)
    past_ids = ["m1", "m2", "m3", "m4"]
    past = synthetic_result.snapshots[synthetic_result.snapshots["target_match_id"].isin(past_ids)]
    mutated_past = mutated.snapshots[mutated.snapshots["target_match_id"].isin(past_ids)]
    pdt.assert_frame_equal(mutated_past.reset_index(drop=True), past.reset_index(drop=True), check_exact=True)


def test_missing_direction_is_excluded_but_match_snapshot_is_retained(synthetic_result):
    audit = synthetic_result.summary["eligible_history_population"]
    assert audit["eligible_points"] == 11
    assert audit["excluded_direction_0"] == 1
    assert audit["excluded_without_recognized_direction"] == 0
    assert synthetic_result.snapshots["target_match_id"].nunique() == 6


def test_coverage_both_complete_matches_and_thresholds_are_exact(synthetic_result):
    coverage = synthetic_result.coverage
    assert coverage.columns.tolist() == COVERAGE_COLUMNS
    assert len(coverage) == 546
    key = coverage[
        coverage["scope"].eq("both_global")
        & coverage["metric"].eq("historical_matches")
        & coverage["threshold"].eq(1)
        & coverage["stratum_type"].eq("overall")
    ].iloc[0]
    # Both orientations of m3 and m5 satisfy both historical roles.
    assert key["eligible_snapshots"] == 4
    assert key["total_snapshots"] == 12
    assert key["complete_matches"] == 2
    assert key["total_matches"] == 6
    assert key["coverage_rate"] == pytest.approx(1 / 3)
    assert key["complete_match_rate"] == pytest.approx(1 / 3)
    assert key["covered_players"] == 3


def test_summary_and_coverage_manipulation_are_rejected(synthetic_result):
    frame, targets, eligible, audit = prepare_historical_population(synthetic_points())
    broken_coverage = synthetic_result.coverage.copy(deep=True)
    broken_coverage.loc[0, "eligible_snapshots"] += 1
    broken = HistoricalProfileResult(
        snapshots=synthetic_result.snapshots,
        coverage=broken_coverage,
        summary=synthetic_result.summary,
    )
    with pytest.raises(AssertionError):
        validate_published_contract(broken, targets, eligible)

    broken_summary = json.loads(json.dumps(synthetic_result.summary))
    broken_summary["snapshot_population"]["snapshots"] -= 1
    broken = HistoricalProfileResult(
        snapshots=synthetic_result.snapshots,
        coverage=synthetic_result.coverage,
        summary=broken_summary,
    )
    with pytest.raises(ValueError, match="snapshots"):
        validate_published_contract(broken, targets, eligible)

    broken_summary = json.loads(json.dumps(synthetic_result.summary))
    broken_summary["eligible_history_population"]["server_wins"] += 1
    broken = HistoricalProfileResult(
        snapshots=synthetic_result.snapshots,
        coverage=synthetic_result.coverage,
        summary=broken_summary,
    )
    with pytest.raises(ValueError, match="resumen completo"):
        validate_published_contract(
            broken,
            targets,
            eligible,
            frame=frame,
            audit=audit,
        )


def test_serialization_is_deterministic_without_index_or_nonfinite_values(tmp_path, synthetic_result):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_artifacts(synthetic_result, first / "summary.json", first / "coverage.csv")
    write_artifacts(synthetic_result, second / "summary.json", second / "coverage.csv")
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    assert (first / "coverage.csv").read_bytes() == (second / "coverage.csv").read_bytes()
    loaded = pd.read_csv(first / "coverage.csv")
    assert loaded.columns.tolist() == COVERAGE_COLUMNS
    assert not any(column.startswith("Unnamed") for column in loaded.columns)
    json.loads((first / "summary.json").read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def test_publication_rejects_post_validation_mutation(tmp_path, synthetic_result):
    mutated = HistoricalProfileResult(
        snapshots=synthetic_result.snapshots.copy(deep=True),
        coverage=synthetic_result.coverage.copy(deep=True),
        summary=json.loads(json.dumps(synthetic_result.summary)),
        publication_fingerprint=synthetic_result.publication_fingerprint,
    )
    mutated.coverage.loc[0, "eligible_snapshots"] += 1
    with pytest.raises(ValueError, match="modificado"):
        write_artifacts(mutated, tmp_path / "summary.json", tmp_path / "coverage.csv")
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "coverage.csv").exists()


def test_publication_rolls_back_both_artifacts_if_second_replace_fails(
    tmp_path,
    monkeypatch,
    synthetic_result,
):
    summary_path = tmp_path / "summary.json"
    coverage_path = tmp_path / "coverage.csv"
    original_summary = b"previous summary\n"
    original_coverage = b"previous coverage\n"
    summary_path.write_bytes(original_summary)
    coverage_path.write_bytes(original_coverage)
    real_replace = historical_profiles.os.replace
    failed = False

    def fail_coverage_once(source, target):
        nonlocal failed
        if Path(target) == coverage_path and not failed:
            failed = True
            raise OSError("synthetic coverage replacement failure")
        return real_replace(source, target)

    monkeypatch.setattr(historical_profiles.os, "replace", fail_coverage_once)
    with pytest.raises(OSError, match="synthetic coverage"):
        write_artifacts(synthetic_result, summary_path, coverage_path)
    assert summary_path.read_bytes() == original_summary
    assert coverage_path.read_bytes() == original_coverage
    assert not list(tmp_path.glob("*.tmp"))


def test_parser_cache_uses_sequence_and_second_serve_number():
    calls = []

    def recording_parser(sequence, serve_number):
        calls.append((sequence, serve_number))
        return historical_profiles.parse_sequence(sequence, serve_number=serve_number)

    repeated = pd.concat(
        [synthetic_points(), synthetic_points().iloc[[0]]],
        ignore_index=True,
    )
    repeated["point_number"] = repeated.groupby("match_id").cumcount() + 1
    _, _, _, audit = prepare_historical_population(repeated, parser=recording_parser)
    assert calls
    assert all(serve_number == 2 for _, serve_number in calls)
    assert len(calls) == len(set(calls))
    assert audit["parser_cache_key"] == "(sequence_text, serve_number)"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("server", True),
        ("server", False),
        ("server", "1"),
        ("server", "2"),
        ("server", 1.0),
        ("server", 2.0),
        ("server", 3),
        ("point_winner", True),
        ("point_winner", False),
        ("point_winner", "1"),
        ("point_winner", "2"),
        ("point_winner", 1.0),
        ("point_winner", 2.0),
        ("point_winner", 0),
    ],
)
def test_coercible_server_and_winner_domains_are_rejected(column, value):
    points = synthetic_points()
    points[column] = pd.Series([value] * len(points), dtype="object")
    with pytest.raises(ValueError, match=f"Dominio invalido en {column}"):
        analyze_historical_profiles(points)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda df: df.assign(surface="Carpet"), "Superficies no permitidas"),
        (lambda df: df.assign(surface=None), "Superficies no permitidas"),
        (lambda df: df.assign(date="not-a-date"), "date contiene fechas"),
        (lambda df: df.assign(player_2=df["player_1"]), "deben ser distintos"),
    ],
)
def test_other_invalid_source_domains_are_rejected_independently(mutation, message):
    with pytest.raises(ValueError, match=message):
        analyze_historical_profiles(mutation(synthetic_points()))


def test_duplicate_point_key_and_inconsistent_match_metadata_are_rejected():
    duplicate = pd.concat([synthetic_points(), synthetic_points().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="no es unica"):
        analyze_historical_profiles(duplicate)

    inconsistent = synthetic_points().copy()
    inconsistent.loc[1, "surface"] = "Clay"
    with pytest.raises(ValueError, match="Metadatos no constantes"):
        analyze_historical_profiles(inconsistent)


def test_snapshot_reconciliation_detects_temporal_and_count_manipulation(synthetic_result):
    _, targets, _, _ = prepare_historical_population(synthetic_points())
    temporal = synthetic_result.snapshots.copy(deep=True)
    row = temporal["server_global_points"].gt(0).idxmax()
    temporal.loc[row, "server_global_last_history_date"] = temporal.loc[row, "target_date"]
    with pytest.raises(ValueError, match="Fuga temporal"):
        validate_snapshots(temporal, targets)

    counts = synthetic_result.snapshots.copy(deep=True)
    counts.loc[row, "server_global_wide_points"] += 1
    with pytest.raises(ValueError, match="puntos por direccion"):
        validate_snapshots(counts, targets)


@pytest.mark.integration
def test_real_artifacts_and_source_contract():
    points_path = Path("data/processed/points_enriched.parquet")
    if not points_path.exists() or not SUMMARY_PATH.exists() or not COVERAGE_PATH.exists():
        pytest.skip("Requiere Parquet y artefactos historicos locales.")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    coverage = pd.read_csv(COVERAGE_PATH)
    assert summary["source"] == {
        "columns_used": SOURCE_COLUMNS,
        "date_range": {"first": "1960-05-29", "last": "2026-05-21"},
        "matches": 7524,
        "path": "data/processed/points_enriched.parquet",
        "point_rows": 1280408,
    }
    population = summary["eligible_history_population"]
    assert population["eligible_points"] == 481190
    assert population["eligible_matches"] == 7524
    assert population["eligible_servers"] == 1002
    assert population["eligible_returners"] == 1002
    assert population["server_wins"] == 245683
    assert (population["wide_points"], population["wide_server_wins"]) == (176757, 91440)
    assert (population["body_points"], population["body_server_wins"]) == (165841, 84299)
    assert (population["t_points"], population["t_server_wins"]) == (138592, 69944)
    assert population["excluded_direction_0"] == 148
    assert population["excluded_without_recognized_direction"] == 170
    assert summary["snapshot_population"]["snapshots"] == 15048
    assert summary["snapshot_population"]["target_matches"] == 7524
    assert len(coverage) == 546
    assert coverage.columns.tolist() == COVERAGE_COLUMNS

    benchmarks = {
        ("both_global", "historical_matches", 1): 0.88264221158958,
        ("both_surface", "historical_matches", 1): 0.8027644869750132,
        ("both_global", "historical_matches", 5): 0.6887293992557151,
        ("both_surface", "historical_matches", 5): 0.5128920786815524,
        ("both_global", "historical_points", 100): 0.738237639553429,
        ("both_surface", "historical_points", 100): 0.5835991493886231,
    }
    overall = coverage[coverage["stratum_type"].eq("overall")]
    for key, expected in benchmarks.items():
        scope, metric, threshold = key
        row = overall[
            overall["scope"].eq(scope)
            & overall["metric"].eq(metric)
            & overall["threshold"].eq(threshold)
        ].iloc[0]
        assert row["coverage_rate"] == pytest.approx(expected, abs=1e-15)
