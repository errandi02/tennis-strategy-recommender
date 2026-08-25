import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.second_serve_direction_baseline import (
    GROUP_COLUMNS,
    POINTS_FILE,
    SOURCE_COLUMNS,
    analyze_second_serve_direction_baseline,
    build_analytic_population,
    validate_reconciliations,
    write_artifacts,
)
from src.analysis.second_serve_direction_analysis import prepare_points


def synthetic_points() -> pd.DataFrame:
    rows = [
        ["m1", 1, "2009-01-01", "Hard", 1, 1, "A", "B", "4*"],
        ["m1", 2, "2009-01-01", "Hard", 1, 1, "A", "B", "4*"],
        ["m1", 3, "2009-01-01", "Hard", 2, 1, "A", "B", "5#"],
        ["m1", 4, "2009-01-01", "Hard", 2, 1, "A", "B", "6n"],
        ["m2", 1, "2015-06-01", "Clay", 1, 1, "A", "C", "4*"],
        ["m2", 2, "2015-06-01", "Clay", 1, 2, "A", "C", "6n"],
        ["m2", 3, "2015-06-01", "Clay", 2, 2, "A", "C", "5#"],
        ["m3", 1, "2022-02-01", "Grass", 2, 2, "D", "E", "4*"],
        ["m3", 2, "2022-02-01", "Grass", 2, 1, "D", "E", "6n"],
        ["m4", 1, "2022-03-01", "Hard", 1, 1, "F", "G", "4*"],
        ["m4", 2, "2022-03-01", "Hard", 1, 2, "F", "G", "6n"],
        ["m4", 3, "2022-03-01", "Hard", 2, 2, "F", "G", "5#"],
        ["m5", 1, "2022-04-01", "Hard", 1, 1, "H", "I", "0*"],
        ["m5", 2, "2022-04-01", "Hard", 1, 1, "H", "I", "4*f"],
        ["m5", 3, "2022-04-01", "Hard", 1, 1, "H", "I", "4s3*"],
        ["m5", 4, "2022-04-01", "Hard", 1, 1, "H", "I", None],
        ["m5", 5, "2022-04-01", "Hard", 1, 1, "H", "I", "  "],
    ]
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def test_population_includes_recognized_prefixes_even_with_rally_residue_or_warnings():
    analytic, audit = build_analytic_population(synthetic_points())
    assert len(analytic) == 14
    assert audit["source_rows"] == 17
    assert audit["eligible_rows"] == 14
    assert audit["excluded_direction_0"] == 1
    assert set(analytic["direction"].astype(str)) == {"wide", "body", "T"}
    assert "0" not in set(analytic["direction"].astype(str))
    assert audit["excluded_without_recognized_direction"] == 0
    assert analytic[["match_id", "point_number"]].drop_duplicates().shape[0] == 14


def test_domain_rejections_are_inherited_from_source_validation():
    for value in [True, False, "1", "2", 1.0, 2.0]:
        points = synthetic_points().iloc[[0]].copy()
        points["server"] = pd.Series([value], index=points.index, dtype="object")
        with pytest.raises(ValueError, match="Dominio invalido en server"):
            prepare_points(points)


def test_descriptive_group_aggregations_are_exact():
    summary, groups, _, _ = analyze_second_serve_direction_baseline(synthetic_points())
    overall = summary["descriptive"]["overall"]
    assert overall == {
        "points": 14,
        "matches": 5,
        "servers": 7,
        "server_wins": 9,
        "server_win_rate": pytest.approx(9 / 14),
    }
    direction = groups[groups["dimension"].eq("direction")].set_index("direction")
    assert direction["points"].to_dict() == {"wide": 7, "body": 3, "T": 4}
    assert direction["server_wins"].to_dict() == {"wide": 7, "body": 2, "T": 0}
    assert direction["matches"].to_dict() == {"wide": 5, "body": 3, "T": 4}
    assert direction["servers"].to_dict() == {"wide": 4, "body": 3, "T": 4}
    assert direction.loc["wide", "server_win_rate"] == pytest.approx(1.0)
    assert direction.loc["T", "server_win_rate"] == pytest.approx(0.0)


def test_surface_period_group_rows_are_exact_and_ordered():
    _, groups, _, _ = analyze_second_serve_direction_baseline(synthetic_points())
    assert groups.columns.tolist() == GROUP_COLUMNS
    assert groups[groups["dimension"].eq("direction")]["direction"].tolist() == [
        "wide",
        "body",
        "T",
    ]
    hard_wide = groups[
        (groups["dimension"] == "direction_surface")
        & (groups["direction"] == "wide")
        & (groups["surface"] == "Hard")
    ].iloc[0]
    assert hard_wide["points"] == 5
    assert hard_wide["matches"] == 3
    period_t = groups[
        (groups["dimension"] == "direction_period")
        & (groups["direction"] == "T")
        & (groups["derived_period"] == "2020s")
    ].iloc[0]
    assert period_t["points"] == 2


def test_coverage_percentiles_and_cuts_are_exact():
    summary, _, coverage, _ = analyze_second_serve_direction_baseline(synthetic_points())
    assert len(coverage) == 12
    assert sorted(coverage["points"].tolist()) == [1] * 10 + [2, 2]
    assert sorted(coverage["matches"].tolist()) == [1] * 12
    distribution = summary["coverage"]
    assert distribution["points_per_combination"]["p25"] == pytest.approx(1.0)
    assert distribution["points_per_combination"]["p50"] == pytest.approx(1.0)
    assert distribution["points_per_combination"]["p75"] == pytest.approx(1.0)
    assert distribution["points_per_combination"]["p90"] == pytest.approx(1.9)
    assert distribution["points_per_combination"]["p95"] == pytest.approx(2.0)
    assert distribution["points_per_combination"]["p99"] == pytest.approx(2.0)
    assert distribution["descriptive_point_cuts"]["at_least_10_points"] == {
        "numerator": 0,
        "denominator": 12,
        "proportion": 0.0,
    }


def test_adjusted_analysis_is_not_available_without_statistical_support():
    summary, _, _, _ = analyze_second_serve_direction_baseline(synthetic_points())
    adjusted = summary["adjusted_analysis"]
    assert adjusted["status"] == "not_available"
    assert adjusted["formula"] == (
        "server_won_point ~ direction + surface + derived_period + server_player"
    )
    assert adjusted["variables"] == [
        "direction",
        "surface",
        "derived_period",
        "server_player",
    ]


def test_summary_has_no_nan_inf_and_no_causal_claims():
    summary, _, _, _ = analyze_second_serve_direction_baseline(synthetic_points())
    serialized = json.dumps(summary, allow_nan=False)
    forbidden = ["mejor", "deberia", "recomendamos", "optima"]
    assert not any(word in serialized.lower() for word in forbidden)


def test_reconciliation_rejects_manipulated_groups_and_coverage():
    summary, groups, coverage, analytic = analyze_second_serve_direction_baseline(synthetic_points())
    broken = groups.copy()
    broken.loc[0, "points"] += 1
    broken.loc[0, "server_win_rate"] = broken.loc[0, "server_wins"] / broken.loc[0, "points"]
    with pytest.raises(ValueError, match="grupos: valores"):
        validate_reconciliations(analytic, broken, coverage, summary)
    missing = groups.drop(index=groups.index[0])
    with pytest.raises(ValueError, match="grupos: claves"):
        validate_reconciliations(analytic, missing, coverage, summary)
    duplicate = pd.concat([groups, groups.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="grupos: clave no unica"):
        validate_reconciliations(analytic, duplicate, coverage, summary)
    broken_coverage = coverage.copy()
    broken_coverage.loc[0, "matches"] += 1
    with pytest.raises(ValueError, match="cobertura: valores"):
        validate_reconciliations(analytic, groups, broken_coverage, summary)


@pytest.mark.parametrize("column", ["points", "server_wins", "server_win_rate"])
def test_reconciliation_rejects_group_metric_manipulations(column):
    summary, groups, coverage, analytic = analyze_second_serve_direction_baseline(synthetic_points())
    broken = groups.copy()
    broken.loc[0, column] = np.inf if column == "server_win_rate" else broken.loc[0, column] + 1
    with pytest.raises(ValueError):
        validate_reconciliations(analytic, broken, coverage, summary)


def test_reconciliation_rejects_invalid_direction_nan_and_infinite():
    summary, groups, coverage, analytic = analyze_second_serve_direction_baseline(synthetic_points())
    invalid = analytic.copy()
    invalid["direction"] = invalid["direction"].astype("object")
    invalid.loc[0, "direction"] = "unknown"
    with pytest.raises(ValueError, match="Direccion"):
        validate_reconciliations(invalid, groups, coverage, summary)
    broken = groups.copy()
    broken.loc[0, "server_win_rate"] = np.nan
    with pytest.raises(ValueError):
        validate_reconciliations(analytic, broken, coverage, summary)


def test_reconciliation_rejects_manipulated_summary():
    summary, groups, coverage, analytic = analyze_second_serve_direction_baseline(synthetic_points())
    broken = json.loads(json.dumps(summary))
    broken["population"]["analytic_points"] += 1
    with pytest.raises(ValueError, match="summary: poblacion"):
        validate_reconciliations(analytic, groups, coverage, broken)


def test_artifact_writing_is_deterministic_and_index_free(tmp_path):
    summary, groups, _, _ = analyze_second_serve_direction_baseline(synthetic_points())
    write_artifacts(summary, groups, tmp_path, tmp_path / "tables")
    first = {path.name: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    write_artifacts(summary, groups, tmp_path, tmp_path / "tables")
    second = {path.name: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert first == second
    loaded = pd.read_csv(tmp_path / "tables" / "second_serve_direction_baseline_by_group.csv")
    assert loaded.columns.tolist() == GROUP_COLUMNS
    assert not any(column.startswith("Unnamed") for column in loaded.columns)
    json.loads((tmp_path / "second_serve_direction_baseline_summary.json").read_text(encoding="utf-8"))


@pytest.mark.integration
def test_real_baseline_main_metrics():
    if not POINTS_FILE.exists():
        pytest.skip("No esta disponible points_enriched.parquet.")
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    summary, groups, coverage, analytic = analyze_second_serve_direction_baseline(points)
    assert summary["population"]["analytic_points"] == 481_190
    assert summary["descriptive"]["overall"]["server_wins"] == 245_683
    assert groups[groups["dimension"].eq("direction")]["points"].sum() == 481_190
    assert coverage["points"].sum() == 481_190
    assert set(analytic["direction"].astype(str)) == {"wide", "body", "T"}
