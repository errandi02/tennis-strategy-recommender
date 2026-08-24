import json

import pandas as pd
import pandas.testing as pdt
import pytest

from src.analysis.second_serve_direction_analysis import (
    COVERAGE_COLUMNS,
    GROUP_COLUMNS,
    PARSER_VERSION,
    POINTS_FILE,
    analyze_second_serve_directions,
    classify_presence,
    derive_period,
    prepare_points,
    validate_reconciliations,
    write_artifacts,
)
from src.parsing.serve_sequence import parse_sequence


def synthetic_points():
    return pd.DataFrame(
        [
            ["m1", 1, "2009-12-31", "Hard", 1, 1, "A", "B", None],
            ["m1", 2, "2009-12-31", "Hard", 2, 1, "A", "B", ""],
            ["m1", 3, "2009-12-31", "Hard", 1, 1, "A", "B", "  "],
            ["m1", 4, "2009-12-31", "Hard", 1, 1, "A", "B", "4*"],
            ["m1", 5, "2009-12-31", "Hard", 2, 2, "A", "B", "4#"],
            ["m2", 1, "2015-06-01", "Clay", 2, 1, "A", "C", "5n"],
            ["m2", 2, "2015-06-01", "Clay", 1, 1, "A", "C", "6f3*"],
            ["m3", 1, "2022-01-01", "Grass", 2, 2, "D", "E", "0"],
            ["m3", 2, "2022-01-01", "Grass", 1, 2, "D", "E", "R"],
            ["m4", 1, "2022-02-01", "Hard", 1, 1, "A", "D", "4*"],
        ],
        columns=["match_id", "point_number", "date", "surface", "server", "point_winner", "player_1", "player_2", "second_serve"],
    )


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        prepare_points(synthetic_points().drop(columns="surface"))


def test_duplicate_key_is_rejected():
    points = pd.concat([synthetic_points(), synthetic_points().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="no es unica"):
        prepare_points(points)


def test_null_key_is_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "match_id"] = None
    with pytest.raises(ValueError, match="claves de punto"):
        prepare_points(points)


@pytest.mark.parametrize(("column", "value"), [("server", 0), ("point_winner", 3)])
def test_invalid_player_index_domains_are_rejected(column, value):
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, column] = value
    with pytest.raises(ValueError, match=f"Dominio invalido en {column}"):
        prepare_points(points)


@pytest.mark.parametrize("value", [True, False, "1", "2", 1.0, 2.0])
@pytest.mark.parametrize("column", ["server", "point_winner"])
def test_coercible_player_index_domains_are_rejected(column, value):
    points = synthetic_points().iloc[[3]].copy()
    points[column] = pd.Series([value], index=points.index, dtype="object")
    with pytest.raises(ValueError, match=f"Dominio invalido en {column}"):
        prepare_points(points)


def test_invalid_surface_is_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "surface"] = "Carpet"
    with pytest.raises(ValueError, match="Carpet"):
        prepare_points(points)


def test_invalid_date_is_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="fechas nulas o no validas"):
        prepare_points(points)


def test_equal_players_are_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "player_2"] = "A"
    with pytest.raises(ValueError, match="deben ser distintos"):
        prepare_points(points)


def test_null_player_is_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "player_1"] = None
    with pytest.raises(ValueError, match="jugadores no pueden contener nulos"):
        prepare_points(points)


def test_presence_states_are_exact():
    assert [classify_presence(x) for x in [None, "", "\t ", " 4 "]] == ["null", "empty", "whitespace_only", "substantive"]
    summary, _, _, _ = analyze_second_serve_directions(synthetic_points())
    states = summary["population"]["presence_states"]
    assert {key: states[key]["numerator"] for key in states} == {"null": 1, "empty": 1, "whitespace_only": 1, "substantive": 7}


def test_derivations_and_period_boundaries_are_exact():
    frame, _ = prepare_points(synthetic_points())
    assert frame.loc[0, ["server_player", "returner_player", "server_won_point"]].tolist() == ["A", "B", True]
    assert frame.loc[1, ["server_player", "returner_player", "server_won_point"]].tolist() == ["B", "A", False]
    assert [derive_period(year) for year in [2009, 2010, 2019, 2020]] == ["to_2009", "2010s", "2010s", "2020s"]


def test_parser_is_only_called_once_per_distinct_substantive_sequence():
    calls = []
    def spy(value, serve_number):
        calls.append((value, serve_number))
        return parse_sequence(value, serve_number)
    prepare_points(synthetic_points(), parser=spy)
    substantive = synthetic_points()["second_serve"].dropna()
    substantive = substantive[(substantive != "") & ~substantive.str.isspace()]
    assert sorted(calls) == sorted((value, 2) for value in substantive.unique())


def test_prefix_extraction_and_actionability_cover_all_structures():
    _, prepared = prepare_points(synthetic_points())
    data = prepared["substantive"].set_index("second_serve")
    assert data.loc["4*", "structure_type"].iloc[0] == "ServeAce"
    assert data.loc["4#", "structure_type"] == "ServeUnreturned"
    assert data.loc["5n", "structure_type"] == "ServeFault"
    assert data.loc["6f3*", "structure_type"] == "ServicePrefix"
    assert pd.isna(data.loc["R", "direction_code"])
    assert bool(data.loc["0", "tactically_actionable"]) is False
    assert data.loc["0", "direction_label"] == "unknown"


def test_conservative_outcomes_require_complete_clean_structure():
    _, prepared = prepare_points(synthetic_points())
    data = prepared["substantive"].set_index("second_serve")
    assert data.loc["4*", "conservative_serve_outcome"].tolist() == ["ace", "ace"]
    assert data.loc["4#", "conservative_serve_outcome"] == "unreturned"
    assert data.loc["5n", "conservative_serve_outcome"] == "fault"
    assert data.loc["6f3*", "conservative_serve_outcome"] == "not_assigned"


def test_conservative_reconciliation_preserves_contradiction():
    points = synthetic_points()
    points.loc[points["second_serve"].eq("5n"), "point_winner"] = 2
    summary, _, _, _ = analyze_second_serve_directions(points)
    fault = next(row for row in summary["conservative_outcome_reconciliation"] if row["outcome"] == "fault")
    assert fault["contradictions"] == 1
    assert fault["examples"][0]["second_serve"] == "5n"


def test_conservative_outcomes_reconcile_when_observed_result_agrees():
    summary, _, _, _ = analyze_second_serve_directions(synthetic_points())
    assert summary["conservative_outcome_counts"] == {
        "not_assigned": 3,
        "fault": 1,
        "ace": 2,
        "unreturned": 1,
    }
    rows = {row["outcome"]: row for row in summary["conservative_outcome_reconciliation"]}
    assert rows["ace"]["contradictions"] == 0
    assert rows["unreturned"]["contradictions"] == 0
    assert rows["fault"]["contradictions"] == 0


def test_group_and_coverage_aggregations_are_exact_without_double_counting():
    summary, groups, coverage, substantive = analyze_second_serve_directions(synthetic_points())
    overall = groups[groups["dimension"].eq("overall")].set_index("direction_code")
    assert overall["direction_points"].to_dict() == {"4": 3, "5": 1, "6": 1, "0": 1}
    assert overall.loc["4", "group_second_serve_points"] == 7
    assert overall.loc["4", "direction_proportion"] == pytest.approx(3 / 7)
    assert overall.loc["4", "matches"] == 2
    assert overall.loc["4", "conservative_aces"] == 2
    assert len(substantive) == 7
    assert coverage["points"].sum() == 6
    assert not coverage.duplicated(["server_player", "surface", "derived_period", "direction_code"]).any()
    assert summary["parser_coverage"]["without_direction"] == 1


def test_distribution_uses_linear_percentiles_and_nontrivial_denominators():
    summary, _, _, _ = analyze_second_serve_directions(synthetic_points())
    distribution = summary["coverage_distribution"]["all_directions"]
    assert distribution["percentile_method"] == "linear"
    assert distribution["points_per_combination"]["p25"] == pytest.approx(1.0)
    cut = distribution["descriptive_point_cuts"]["at_least_10_points"]
    assert cut == {"numerator": 0, "denominator": 6, "proportion": 0.0, "represented_servers": 0}


def test_output_order_and_schemas_are_deterministic():
    outputs = analyze_second_serve_directions(synthetic_points())
    _, groups, coverage, _ = outputs
    assert groups.columns.tolist() == GROUP_COLUMNS
    assert coverage.columns.tolist() == COVERAGE_COLUMNS
    assert groups[groups["dimension"].eq("overall")]["direction_code"].tolist() == ["4", "5", "6", "0"]
    shuffled = synthetic_points().sample(frac=1, random_state=7)
    second = analyze_second_serve_directions(shuffled)
    pdt.assert_frame_equal(groups, second[1])
    pdt.assert_frame_equal(coverage, second[2])


def test_artifacts_are_utf8_deterministic_and_without_csv_index(tmp_path):
    summary, groups, coverage, _ = analyze_second_serve_directions(synthetic_points())
    write_artifacts(summary, groups, coverage, tmp_path, tmp_path / "tables")
    first = {p.name: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    write_artifacts(summary, groups, coverage, tmp_path, tmp_path / "tables")
    second = {p.name: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert first == second
    for filename, columns in [("second_serve_direction_by_group.csv", GROUP_COLUMNS), ("second_serve_direction_coverage.csv", COVERAGE_COLUMNS)]:
        loaded = pd.read_csv(tmp_path / "tables" / filename)
        assert loaded.columns.tolist() == columns
        assert not any(column.startswith("Unnamed") for column in loaded.columns)
    loaded_json = json.loads((tmp_path / "second_serve_direction_summary.json").read_text(encoding="utf-8"))
    assert "timestamp" not in json.dumps(loaded_json).lower()
    flags = loaded_json["methodological_flags"]
    assert flags["causal_effect_estimated"] is False
    assert flags["model_feature_approved"] is False
    assert flags["model_thresholds_approved"] is False


def test_reconciliation_rejects_manipulated_tables():
    frame, prepared = prepare_points(synthetic_points())
    substantive = prepared["substantive"]
    summary, groups, coverage, _ = analyze_second_serve_directions(synthetic_points())
    broken = coverage.copy()
    broken.loc[0, "points"] += 1
    with pytest.raises(ValueError, match="cobertura: valores publicados"):
        validate_reconciliations(frame, substantive, groups, broken, summary)


def test_reconciliation_rejects_wrong_presence_state():
    frame, prepared = prepare_points(synthetic_points())
    summary, groups, coverage, _ = analyze_second_serve_directions(synthetic_points())
    frame.loc[0, "second_serve_presence_state"] = "substantive"
    with pytest.raises(ValueError, match="estados de presencia"):
        validate_reconciliations(frame, prepared["substantive"], groups, coverage, summary)


def test_reconciliation_rejects_missing_additional_and_duplicate_group_rows():
    frame, prepared = prepare_points(synthetic_points())
    summary, groups, coverage, _ = analyze_second_serve_directions(synthetic_points())
    with pytest.raises(ValueError, match="grupos: claves no reconcilian"):
        validate_reconciliations(
            frame,
            prepared["substantive"],
            groups.drop(index=groups.index[0]),
            coverage,
            summary,
        )
    additional = pd.concat([groups, groups.iloc[[0]]], ignore_index=True)
    additional.loc[additional.index[-1], "surface"] = "Extra"
    with pytest.raises(ValueError, match="grupos: claves no reconcilian"):
        validate_reconciliations(frame, prepared["substantive"], additional, coverage, summary)
    duplicate = pd.concat([groups, groups.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="grupos: clave no unica"):
        validate_reconciliations(frame, prepared["substantive"], duplicate, coverage, summary)


def test_reconciliation_rejects_manipulated_group_values_even_if_ratios_match():
    frame, prepared = prepare_points(synthetic_points())
    summary, groups, coverage, _ = analyze_second_serve_directions(synthetic_points())
    broken = groups.copy()
    broken.loc[0, "group_second_serve_points"] += 10
    broken.loc[0, "direction_proportion"] = (
        broken.loc[0, "direction_points"] / broken.loc[0, "group_second_serve_points"]
    )
    with pytest.raises(ValueError, match="grupos: valores publicados"):
        validate_reconciliations(frame, prepared["substantive"], broken, coverage, summary)
    broken = groups.copy()
    broken.loc[0, ["matches", "servers", "server_wins", "conservative_aces"]] += 1
    broken.loc[0, "server_win_rate"] = (
        broken.loc[0, "server_wins"] / broken.loc[0, "direction_points"]
    )
    with pytest.raises(ValueError, match="grupos: valores publicados"):
        validate_reconciliations(frame, prepared["substantive"], broken, coverage, summary)


def test_reconciliation_rejects_manipulated_coverage_values():
    frame, prepared = prepare_points(synthetic_points())
    summary, groups, coverage, _ = analyze_second_serve_directions(synthetic_points())
    broken = coverage.copy()
    broken.loc[0, "matches"] += 1
    broken.loc[0, "server_win_rate"] = 0.25
    with pytest.raises(ValueError, match="cobertura: valores publicados"):
        validate_reconciliations(frame, prepared["substantive"], groups, broken, summary)
    broken = coverage.copy()
    broken.loc[0, "first_date"] = pd.Timestamp("1999-01-01")
    with pytest.raises(ValueError, match="cobertura: valores publicados"):
        validate_reconciliations(frame, prepared["substantive"], groups, broken, summary)


def test_reconciliation_rejects_manipulated_summary_outcomes_and_contradictions():
    frame, prepared = prepare_points(synthetic_points())
    summary, groups, coverage, _ = analyze_second_serve_directions(synthetic_points())
    broken = json.loads(json.dumps(summary))
    broken["conservative_outcome_counts"]["ace"] += 1
    broken["conservative_outcome_counts"]["not_assigned"] -= 1
    with pytest.raises(ValueError, match="outcomes conservadores"):
        validate_reconciliations(frame, prepared["substantive"], groups, coverage, broken)
    broken = json.loads(json.dumps(summary))
    broken["conservative_outcome_reconciliation"][0]["contradictions"] = 1
    with pytest.raises(ValueError, match="contradicciones publicadas"):
        validate_reconciliations(frame, prepared["substantive"], groups, coverage, broken)


def test_reconciliation_rejects_bad_contradiction_example():
    points = synthetic_points()
    points.loc[points["second_serve"].eq("5n"), "point_winner"] = 2
    frame, prepared = prepare_points(points)
    summary, groups, coverage, _ = analyze_second_serve_directions(points)
    broken = json.loads(json.dumps(summary))
    fault = next(
        row for row in broken["conservative_outcome_reconciliation"]
        if row["outcome"] == "fault"
    )
    fault["examples"][0]["point_number"] = 999
    with pytest.raises(ValueError, match="contradicciones publicadas"):
        validate_reconciliations(frame, prepared["substantive"], groups, coverage, broken)


@pytest.mark.integration
def test_real_analysis_matches_inspected_metrics():
    if not POINTS_FILE.exists():
        pytest.skip("No esta disponible points_enriched.parquet.")
    points = pd.read_parquet(POINTS_FILE, columns=["match_id", "point_number", "date", "surface", "server", "point_winner", "player_1", "player_2", "second_serve"])
    summary, groups, coverage, substantive = analyze_second_serve_directions(points)
    assert summary["source"]["point_rows"] == 1_280_408
    states = summary["population"]["presence_states"]
    assert [states[x]["numerator"] for x in ["null", "empty", "whitespace_only", "substantive"]] == [798_865, 0, 35, 481_508]
    assert len(substantive) == 481_508
    assert summary["parser_coverage"]["recognized_direction"] == 481_338
    assert summary["parser_coverage"]["without_direction"] == 170
    overall = groups[groups["dimension"].eq("overall")].set_index("direction_code")
    assert overall["direction_points"].to_dict() == {"4": 176_757, "5": 165_841, "6": 138_592, "0": 148}
    assert coverage["points"].sum() == 481_338
    assert summary["parser"]["version"] == PARSER_VERSION == "0.2.0"
    reconciliations = {row["outcome"]: row for row in summary["conservative_outcome_reconciliation"]}
    assert (reconciliations["ace"]["rows"], reconciliations["ace"]["contradictions"]) == (3_517, 0)
    assert (reconciliations["unreturned"]["rows"], reconciliations["unreturned"]["contradictions"]) == (3_246, 0)
    assert (reconciliations["fault"]["rows"], reconciliations["fault"]["contradictions"]) == (42_619, 8)
    assert summary["conservative_outcome_counts"] == {
        "not_assigned": 432_126,
        "fault": 42_619,
        "ace": 3_517,
        "unreturned": 3_246,
    }
    assert summary["coverage_distribution"]["all_directions"]["combinations"] == 6_303
    assert summary["coverage_distribution"]["tactically_actionable_directions"]["combinations"] == 6_210
    assert summary["annual_fragmentation"]["combinations"] == 15_330
