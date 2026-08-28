import copy
import inspect
import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

import src.analysis.chronological_validation as module
from src.analysis.chronological_validation import (
    ALLOWED_SURFACES,
    BY_YEAR_COLUMNS,
    ChronologicalValidationResult,
    EXPECTED_SPLIT_MATCHES,
    FOLD_COLUMNS,
    SOURCE_COLUMNS,
    add_basic_cold_start,
    add_splits,
    assign_split,
    build_by_year,
    build_rolling_folds,
    build_summary,
    read_and_build,
    serialize_artifacts,
    validate_and_reduce_matches,
    validate_result,
    validate_serialized_artifacts,
    write_artifacts,
)


def synthetic_points() -> pd.DataFrame:
    matches = [
        ("m0", "2019-12-31", "A", "B", "Hard"),
        ("m1", "2020-01-01", "A", "C", "Clay"),
        ("m2", "2020-01-01", "C", "D", "Grass"),
        ("m3", "2020-01-02", "C", "A", "Hard"),
        ("m4", "2021-05-01", "A", "D", "Clay"),
        ("m5", "2022-06-01", "A", "E", "Grass"),
        ("m6", "2023-07-01", "E", "B", "Hard"),
        ("m7", "2024-01-01", "F", "A", "Clay"),
    ]
    rows = []
    for match in matches:
        rows.extend([dict(zip(SOURCE_COLUMNS, match)), dict(zip(SOURCE_COLUMNS, match))])
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


@pytest.fixture
def synthetic_result() -> ChronologicalValidationResult:
    points = synthetic_points()
    matches = add_splits(validate_and_reduce_matches(points))
    by_year = build_by_year(matches)
    folds = build_rolling_folds(matches)
    summary = build_summary(len(points), matches, by_year, folds)
    result = ChronologicalValidationResult(len(points), matches, by_year, folds, summary)
    validate_result(result)
    return result


def test_validate_and_reduce_matches_is_non_mutating_and_deterministic():
    points = synthetic_points().sample(frac=1, random_state=4).reset_index(drop=True)
    original = points.copy(deep=True)
    matches = validate_and_reduce_matches(points, expected_source_rows=16, expected_matches=8)
    pdt.assert_frame_equal(points, original)
    assert matches.columns.tolist() == SOURCE_COLUMNS
    assert matches.match_id.tolist() == ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]
    assert pd.api.types.is_datetime64_any_dtype(matches.date)
    assert matches.match_id.is_unique


@pytest.mark.parametrize("column", ["date", "player_1", "player_2", "surface"])
def test_inconsistent_metadata_within_match_is_rejected(column):
    points = synthetic_points()
    index = points.index[points.match_id.eq("m0")][-1]
    replacements = {
        "date": "2019-12-30", "player_1": "X", "player_2": "Y", "surface": "Clay",
    }
    points.loc[index, column] = replacements[column]
    with pytest.raises(ValueError, match="Metadatos inconsistentes"):
        validate_and_reduce_matches(points)


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("match_id", "", "vacias"),
        ("match_id", " m0", "espacios"),
        ("player_1", 1, "solo cadenas"),
        ("player_2", True, "solo cadenas"),
        ("player_1", "A ", "espacios"),
        ("surface", 1.0, "solo cadenas"),
    ],
)
def test_invalid_exact_text_domains_are_rejected(column, value, error):
    points = synthetic_points()
    points[column] = points[column].astype(object)
    points.loc[0, column] = value
    with pytest.raises((TypeError, ValueError), match=error):
        validate_and_reduce_matches(points)


def test_equal_players_are_rejected():
    points = synthetic_points()
    points.loc[points.match_id.eq("m0"), "player_2"] = "A"
    with pytest.raises(ValueError, match="deben ser distintos"):
        validate_and_reduce_matches(points)


@pytest.mark.parametrize("surface", [None, "Carpet", "hard", " Hard"])
def test_invalid_surfaces_are_rejected(surface):
    points = synthetic_points()
    points.loc[points.match_id.eq("m0"), "surface"] = surface
    with pytest.raises((TypeError, ValueError)):
        validate_and_reduce_matches(points)


@pytest.mark.parametrize("date_value", ["01/02/2020", "2020-02-30", "20200101", 20200101])
def test_invalid_or_ambiguous_dates_are_rejected(date_value):
    points = synthetic_points()
    points["date"] = points["date"].astype(object)
    points.loc[:, "date"] = date_value
    with pytest.raises((TypeError, ValueError), match="date"):
        validate_and_reduce_matches(points)


def test_mixed_date_representations_are_rejected():
    points = synthetic_points()
    points["date"] = points["date"].astype(object)
    points.loc[0, "date"] = pd.Timestamp("2019-12-31")
    with pytest.raises(ValueError, match="mezcla"):
        validate_and_reduce_matches(points)


def test_time_components_are_normalized_to_the_same_calendar_day():
    points = synthetic_points()
    points["date"] = pd.to_datetime(points["date"])
    points.loc[points.match_id.eq("m1"), "date"] = pd.Timestamp("2020-01-01 12:00:00")
    points.loc[points.match_id.eq("m2"), "date"] = pd.Timestamp("2020-01-01 23:59:59")
    matches = validate_and_reduce_matches(points)
    assert matches.loc[matches.match_id.isin(["m1", "m2"]), "date"].nunique() == 1
    cold = add_basic_cold_start(add_splits(matches)).set_index("match_id")
    assert cold.loc["m2", "basic_cold_start_state"] == "neither_prior"


def test_source_and_match_count_contracts_are_explicit():
    points = synthetic_points()
    with pytest.raises(ValueError, match="Filas fuente inesperadas"):
        validate_and_reduce_matches(points, expected_source_rows=17)
    with pytest.raises(ValueError, match="Partidos inesperados"):
        validate_and_reduce_matches(points, expected_matches=9)


def test_split_boundaries_are_pure_and_complete():
    dates = pd.Series(pd.to_datetime([
        "1960-05-29", "2019-12-31", "2020-01-01", "2023-12-31",
        "2024-01-01", "2026-05-21",
    ]))
    original = dates.copy(deep=True)
    actual = assign_split(dates)
    assert actual.tolist() == ["train", "train", "validation", "validation", "test", "test"]
    pdt.assert_series_equal(dates, original)
    with pytest.raises(ValueError, match="posteriores"):
        assign_split(pd.Series(pd.to_datetime(["2026-05-22"])))


def test_same_date_is_indivisible_and_strictly_prior_for_cold_start():
    matches = add_splits(validate_and_reduce_matches(synthetic_points()))
    assert matches.groupby("date").split.nunique().eq(1).all()
    cold = add_basic_cold_start(matches).set_index("match_id")
    assert cold.loc["m0", "basic_cold_start_state"] == "neither_prior"
    assert cold.loc["m1", "basic_cold_start_state"] == "one_prior"
    assert cold.loc["m2", "basic_cold_start_state"] == "neither_prior"
    assert cold.loc["m3", "basic_cold_start_state"] == "both_prior"


def test_row_order_does_not_change_matches_annual_folds_or_cold_start():
    points = synthetic_points()
    shuffled = points.sample(frac=1, random_state=19).reset_index(drop=True)
    matches_a = add_splits(validate_and_reduce_matches(points))
    matches_b = add_splits(validate_and_reduce_matches(shuffled))
    pdt.assert_frame_equal(matches_a, matches_b, check_exact=True)
    pdt.assert_frame_equal(build_by_year(matches_a), build_by_year(matches_b), check_exact=True)
    pdt.assert_frame_equal(build_rolling_folds(matches_a), build_rolling_folds(matches_b), check_exact=True)


def test_metadata_constancy_is_checked_before_deduplication():
    source = inspect.getsource(validate_and_reduce_matches)
    assert source.index('.nunique(dropna=False)') < source.index('.drop_duplicates("match_id"')


def test_annual_aggregates_are_exact_and_exhaustive(synthetic_result):
    annual = synthetic_result.by_year
    assert annual.columns.tolist() == BY_YEAR_COLUMNS
    row = annual.loc[annual.year.eq(2020)].iloc[0]
    assert row.split == "validation" and not bool(row.is_partial_year)
    assert (
        row.matches, row.players, row.new_players_at_year_start,
        row.previously_seen_players_at_year_start,
    ) == (3, 3, 2, 1)
    assert (
        row.both_players_prior_matches,
        row.one_player_prior_matches,
        row.neither_player_prior_matches,
    ) == (1, 1, 1)
    assert row.both_players_prior_rate == pytest.approx(1 / 3)
    assert (row.hard_matches, row.clay_matches, row.grass_matches) == (1, 1, 1)
    assert annual.matches.sum() == 8


def test_rolling_folds_are_expanding_and_use_known_calendar_years(synthetic_result):
    folds = synthetic_result.folds
    assert folds.columns.tolist() == FOLD_COLUMNS
    assert folds.fold_id.tolist() == [
        "validation_2020", "validation_2021", "validation_2022", "validation_2023",
        "final_test_rolling", "final_test_frozen",
    ]
    validation = folds.loc[~folds.is_final_test]
    assert validation.train_matches.tolist() == [1, 4, 5, 6]
    assert validation.evaluation_matches.tolist() == [3, 1, 1, 1]
    first = folds.iloc[0]
    assert (first.evaluation_players, first.unseen_players_at_evaluation_start) == (3, 2)
    assert (
        first.both_players_prior_matches,
        first.one_player_prior_matches,
        first.neither_player_prior_matches,
    ) == (1, 1, 1)
    assert validation.train_matches.is_monotonic_increasing
    final = folds.loc[folds.is_final_test]
    assert final.evaluation_matches.tolist() == [1, 1]
    assert final.protocol.tolist() == ["rolling_origin", "frozen"]
    assert final.history_update_policy.tolist() == [
        "update_after_complete_date", "frozen_at_2023-12-31"
    ]
    assert final.test_usage_policy.eq("sealed_single_final_use").all()


def test_summary_freezes_test_and_contains_no_analytical_outputs(synthetic_result):
    summary = synthetic_result.summary
    assert summary["frozen_sensitivity_contract"] == {
        "freeze_date": "2023-12-31",
        "evaluation_start": "2024-01-01",
        "evaluation_end": "2026-05-21",
        "test_matches": 1,
        "features_baselines_and_parameters_remain_fixed": True,
        "test_matches_do_not_update_later_test_predictions": True,
        "same_test_population_as_principal": True,
        "performance_metrics_executed": False,
    }
    protected = summary["protected_test_contract"]
    assert protected["test_status"] == "sealed"
    assert protected["test_used_for_method_selection"] is False
    assert protected["test_evaluation_runs"] == 0
    assert (protected["test_start"], protected["test_end"], protected["test_matches"]) == (
        "2024-01-01", "2026-05-21", 1,
    )
    assert set(protected["excluded_from_selection"]) == {
        "thresholds", "windows", "smoothing", "fallback", "scoring",
        "hyperparameters", "feature_definitions",
    }
    assert summary["match_table_contract"]["published_match_level_table"] is False
    text = json.dumps(summary).lower()
    assert "recommendation" in text
    assert summary["reconciliations"]["no_features_outcomes_models_or_recommendations"] is True
    assert summary["next_allowed_step"]["test_remains_sealed"] is True
    assert summary["next_allowed_step"]["scoring_models_or_test_evaluation_authorized"] is False


def test_serialization_is_deterministic_utf8_and_has_no_csv_index(synthetic_result):
    first = serialize_artifacts(synthetic_result)
    second = serialize_artifacts(synthetic_result)
    assert first == second
    validate_serialized_artifacts(synthetic_result, *first)
    assert first[0].decode("utf-8").endswith("\n")
    for payload, columns in zip(first[1:], [BY_YEAR_COLUMNS, FOLD_COLUMNS]):
        loaded = pd.read_csv(pd.io.common.BytesIO(payload))
        assert loaded.columns.tolist() == columns
        assert not any(column.startswith("Unnamed") for column in loaded.columns)


@pytest.mark.parametrize(
    "section",
    [
        "source_contract", "match_table_contract", "date_contract", "identity_contract",
        "surface_contract", "split_contract", "split_population",
        "rolling_origin_contract", "rolling_folds", "frozen_sensitivity_contract",
        "protected_test_contract", "cold_start_contract", "yearly_contract",
        "folds_table_contract", "reconciliations", "methodological_limits",
        "next_allowed_step", "publication_fingerprint",
    ],
)
def test_any_summary_contract_mutation_is_rejected(section, synthetic_result):
    summary = copy.deepcopy(synthetic_result.summary)
    value = summary[section]
    if isinstance(value, dict):
        value["tampered"] = True
    elif isinstance(value, list):
        value.append("tampered")
    else:
        summary[section] = "tampered"
    broken = ChronologicalValidationResult(
        synthetic_result.source_rows,
        synthetic_result.matches.copy(deep=True),
        synthetic_result.by_year.copy(deep=True),
        synthetic_result.folds.copy(deep=True),
        summary,
    )
    with pytest.raises(ValueError, match="resumen"):
        validate_result(broken)


@pytest.mark.parametrize("table", ["by_year", "folds"])
def test_any_published_table_mutation_is_rejected(table, synthetic_result):
    annual = synthetic_result.by_year.copy(deep=True)
    folds = synthetic_result.folds.copy(deep=True)
    if table == "by_year":
        annual.loc[0, "matches"] += 1
    else:
        folds.loc[0, "evaluation_matches"] += 1
    broken = ChronologicalValidationResult(
        synthetic_result.source_rows,
        synthetic_result.matches.copy(deep=True), annual, folds,
        copy.deepcopy(synthetic_result.summary),
    )
    with pytest.raises((AssertionError, ValueError)):
        validate_result(broken)


def test_serialized_json_and_csv_manipulation_is_rejected(synthetic_result):
    payloads = list(serialize_artifacts(synthetic_result))
    summary = json.loads(payloads[0])
    summary["protected_test_contract"]["test_status"] = "used"
    payloads[0] = (json.dumps(summary) + "\n").encode()
    with pytest.raises((AssertionError, ValueError)):
        validate_serialized_artifacts(synthetic_result, *payloads)
    payloads = list(serialize_artifacts(synthetic_result))
    annual = pd.read_csv(pd.io.common.BytesIO(payloads[1]))
    annual.loc[0, "matches"] += 1
    payloads[1] = annual.to_csv(index=False, lineterminator="\n").encode()
    with pytest.raises((AssertionError, ValueError)):
        validate_serialized_artifacts(synthetic_result, *payloads)


@pytest.mark.parametrize("failed_replace", [2, 3])
def test_atomic_rollback_restores_all_artifacts_on_later_replace(
    failed_replace, tmp_path, synthetic_result, monkeypatch,
):
    paths = [tmp_path / "summary.json", tmp_path / "year.csv", tmp_path / "folds.csv"]
    originals = [b"old summary\n", b"old year\n", b"old folds\n"]
    for path, payload in zip(paths, originals):
        path.write_bytes(payload)
    real_replace = module.os.replace
    calls = 0

    def fail_once(source, destination):
        nonlocal calls
        calls += 1
        if calls == failed_replace:
            raise OSError("replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_once)
    with pytest.raises(OSError, match="replace failure"):
        write_artifacts(synthetic_result, *paths)
    assert [path.read_bytes() for path in paths] == originals
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_artifacts_writes_only_three_aggregates(tmp_path, synthetic_result):
    paths = [tmp_path / "summary.json", tmp_path / "year.csv", tmp_path / "folds.csv"]
    write_artifacts(synthetic_result, *paths)
    assert all(path.exists() for path in paths)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["folds.csv", "summary.json", "year.csv"]
    assert json.loads(paths[0].read_text(encoding="utf-8"))["folds_table_contract"]["only_aggregates_published"]


def test_read_and_build_reads_parquet_once_with_only_authorized_columns(monkeypatch, tmp_path):
    calls = []
    sentinel = object()

    def fake_read(path, *, columns):
        calls.append((Path(path), columns))
        return synthetic_points()

    monkeypatch.setattr(module.pd, "read_parquet", fake_read)
    monkeypatch.setattr(module, "build_analysis", lambda points: sentinel)
    assert read_and_build(tmp_path / "points.parquet") is sentinel
    assert calls == [(tmp_path / "points.parquet", SOURCE_COLUMNS)]


def test_allowed_surface_catalog_is_exact():
    assert ALLOWED_SURFACES == ("Hard", "Clay", "Grass")


@pytest.mark.integration
def test_published_real_chronological_contract_and_populations():
    paths = (module.SUMMARY_PATH, module.BY_YEAR_PATH, module.FOLDS_PATH)
    if not all(path.exists() for path in paths):
        pytest.skip("No existen los artefactos cronologicos locales.")
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    by_year = pd.read_csv(paths[1])
    folds = pd.read_csv(paths[2])
    assert summary["source_contract"]["source_point_rows"] == 1_280_408
    assert summary["source_contract"]["matches_after_immediate_reduction"] == 7_524
    assert summary["source_contract"]["players"] == 1_002
    assert summary["source_contract"]["date_range"] == {
        "first": "1960-05-29", "last": "2026-05-21",
    }
    population = {row["split"]: row for row in summary["split_population"]}
    assert {split: row["matches"] for split, row in population.items()} == EXPECTED_SPLIT_MATCHES
    assert population["train"]["last_observed_date"] == "2019-11-24"
    assert population["train"]["surface_matches"] == {"hard": 2_666, "clay": 987, "grass": 535}
    assert population["validation"]["surface_matches"] == {"hard": 1_183, "clay": 452, "grass": 170}
    assert population["test"]["surface_matches"] == {"hard": 974, "clay": 433, "grass": 124}
    validation = folds.loc[~folds.is_final_test]
    assert validation.train_matches.tolist() == [4_188, 4_356, 4_727, 5_373]
    assert validation.evaluation_matches.tolist() == [168, 371, 646, 620]
    assert validation.grass_matches.tolist() == [0, 28, 66, 76]
    final = folds.loc[folds.is_final_test]
    assert final.evaluation_matches.tolist() == [1_531, 1_531]
    assert final[["hard_matches", "clay_matches", "grass_matches"]].values.tolist() == [
        [974, 433, 124], [974, 433, 124],
    ]
    assert by_year.year.is_unique and len(by_year) == 59
    assert by_year.loc[by_year.year.eq(2026), "is_partial_year"].item() is True
    recent_columns = [
        "matches", "players", "new_players_at_year_start",
        "previously_seen_players_at_year_start", "both_players_prior_matches",
        "one_player_prior_matches", "neither_player_prior_matches",
        "hard_matches", "clay_matches", "grass_matches",
    ]
    expected_recent = {
        2018: [259, 122, 28, 94, 236, 18, 5, 158, 68, 33],
        2019: [606, 234, 80, 154, 535, 62, 9, 384, 161, 61],
        2020: [168, 113, 15, 98, 155, 11, 2, 144, 24, 0],
        2021: [371, 166, 51, 115, 327, 37, 7, 249, 94, 28],
        2022: [646, 220, 67, 153, 589, 47, 10, 403, 177, 66],
        2023: [620, 178, 28, 150, 594, 24, 2, 387, 157, 76],
        2024: [768, 259, 81, 178, 700, 55, 13, 472, 225, 71],
        2025: [579, 209, 40, 169, 544, 30, 5, 361, 165, 53],
        2026: [184, 111, 11, 100, 176, 5, 3, 141, 43, 0],
    }
    observed_recent = by_year.set_index("year").loc[list(expected_recent), recent_columns]
    assert observed_recent.values.tolist() == list(expected_recent.values())
    assert summary["protected_test_contract"] == {
        **summary["protected_test_contract"],
        "test_status": "sealed",
        "test_used_for_method_selection": False,
        "test_evaluation_runs": 0,
        "test_start": "2024-01-01",
        "test_end": "2026-05-21",
        "test_matches": 1_531,
    }
    assert summary["reconciliations"]["single_split_per_calendar_date"] is True
