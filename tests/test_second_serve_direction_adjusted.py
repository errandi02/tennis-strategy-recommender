import json
import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.analysis.second_serve_direction_adjusted import (
    COEFFICIENT_COLUMNS,
    MODEL_FORMULA,
    POINTS_FILE,
    REFERENCE_DIRECTION,
    SOURCE_COLUMNS,
    analyze_second_serve_direction_adjusted,
    validate_reconciliations,
    write_artifacts,
)
from src.analysis.second_serve_direction_baseline import (
    analyze_second_serve_direction_baseline,
)
from src.analysis.second_serve_direction_analysis import prepare_points


def synthetic_points() -> pd.DataFrame:
    rows = [
        ["m1", 1, "2009-01-01", "Hard", 1, 1, "A", "B", "4*"],
        ["m1", 2, "2009-01-01", "Hard", 1, 1, "A", "B", "5#"],
        ["m1", 3, "2009-01-01", "Hard", 2, 1, "A", "B", "6n"],
        ["m2", 1, "2015-01-01", "Clay", 1, 1, "A", "C", "4*"],
        ["m2", 2, "2015-01-01", "Clay", 2, 2, "A", "C", "5#"],
        ["m2", 3, "2015-01-01", "Clay", 1, 2, "A", "C", "6n"],
        ["m3", 1, "2022-01-01", "Grass", 2, 2, "D", "E", "4*"],
        ["m3", 2, "2022-01-01", "Grass", 2, 1, "D", "E", "6n"],
        ["m3", 3, "2022-01-01", "Grass", 1, 1, "D", "E", "0*"],
        ["m3", 4, "2022-01-01", "Grass", 1, 1, "D", "E", "4s3*"],
        ["m3", 5, "2022-01-01", "Grass", 1, 1, "D", "E", "4*f"],
        ["m3", 6, "2022-01-01", "Grass", 1, 1, "D", "E", None],
    ]
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def fittable_synthetic_points() -> pd.DataFrame:
    rows = []
    directions = ["4*", "5#", "6n"]
    surfaces = ["Hard", "Clay", "Grass"]
    dates = ["2009-01-01", "2015-01-01", "2022-01-01"]
    players = [("A", "B"), ("C", "D"), ("E", "F")]
    index = 0
    for direction in directions:
        for surface in surfaces:
            for date in dates:
                for player_1, player_2 in players:
                    for won in (True, False):
                        rows.append([
                            f"fit{index}", 1, date, surface,
                            1, 1 if won else 2, player_1, player_2, direction,
                        ])
                        index += 1
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def test_population_contract_excludes_non_baseline_rows():
    summary, _, analytic = analyze_second_serve_direction_adjusted(synthetic_points())
    assert summary["analytical_points"] == 10
    assert summary["population_by_direction"] == {"wide": 5, "body": 2, "T": 3}
    assert set(analytic["direction"].astype(str)) == {"wide", "body", "T"}
    assert analytic["server_won_point"].isin([True, False]).all()


@pytest.mark.parametrize("value", [True, False, "1", "2", 1.0, 2.0])
def test_source_domain_rejections_are_strict(value):
    points = synthetic_points().iloc[[0]].copy()
    points["server"] = pd.Series([value], index=points.index, dtype="object")
    with pytest.raises(ValueError, match="Dominio invalido en server"):
        prepare_points(points)


def test_model_contract_without_requiring_statsmodels():
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(synthetic_points())
    assert summary["reference_direction"] == REFERENCE_DIRECTION == "wide"
    assert summary["model_formula"] == MODEL_FORMULA.format(server_reference="A")
    assert summary["standard_error_method"] == "cluster_robust_by_match_id"
    assert coefficients.columns.tolist() == COEFFICIENT_COLUMNS
    if summary["model_status"] == "available":
        assert set(coefficients.loc[coefficients["variable"].eq("direction"), "level"]) == {"body", "T"}
        assert summary["global_direction_test"]["status"] in {"available", "not_available"}
    else:
        assert summary["global_direction_test"]["status"] == "not_available"
        assert len(coefficients) == 0


def test_memory_guard_returns_complete_not_available_contract(monkeypatch):
    monkeypatch.setattr("src.analysis.second_serve_direction_adjusted.MEMORY_LIMIT_BYTES", 1)
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(fittable_synthetic_points())
    assert summary["model_status"] == "not_available"
    assert summary["global_direction_test"]["reason"] == "estimated_dense_design_exceeds_memory_limit"
    assert summary["planned_model_rows"] == 162
    assert summary["estimated_dense_design_bytes"] > summary["memory_limit_bytes"]
    assert coefficients.empty


def test_summary_contains_no_nan_or_infinity_and_no_recommendation_language():
    summary, _, _ = analyze_second_serve_direction_adjusted(synthetic_points())
    serialized = json.dumps(summary, allow_nan=False)
    forbidden = ["deberia sacar", "recomendamos", "estrategia optima"]
    assert not any(text in serialized.lower() for text in forbidden)


def test_reconciliation_rejects_population_formula_reference_and_global_test_changes():
    baseline_summary, _, _, analytic = analyze_second_serve_direction_baseline(synthetic_points())
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(synthetic_points())
    adjusted = {
        "model_status": summary["model_status"],
        "formula": summary["model_formula"],
        "reason": summary["global_direction_test"]["reason"],
        "estimator": summary["estimator"],
        "standard_error_method": summary["standard_error_method"],
    }
    broken = json.loads(json.dumps(summary))
    broken["analytical_points"] += 1
    with pytest.raises(ValueError, match="analytical_points"):
        validate_reconciliations(baseline_summary, analytic, adjusted, coefficients, broken)
    broken = json.loads(json.dumps(summary))
    broken["reference_direction"] = "body"
    with pytest.raises(ValueError, match="reference_direction"):
        validate_reconciliations(baseline_summary, analytic, adjusted, coefficients, broken)
    broken = json.loads(json.dumps(summary))
    broken["model_formula"] = "y ~ x"
    with pytest.raises(ValueError, match="model_formula"):
        validate_reconciliations(baseline_summary, analytic, adjusted, coefficients, broken)
    broken = json.loads(json.dumps(summary))
    broken["global_direction_test"] = {"status": "available"}
    with pytest.raises(ValueError, match="global_direction_test"):
        validate_reconciliations(baseline_summary, analytic, adjusted, coefficients, broken)


def test_reconciliation_rejects_coefficients_when_model_is_not_available():
    baseline_summary, _, _, analytic = analyze_second_serve_direction_baseline(synthetic_points())
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(synthetic_points())
    adjusted = {
        "model_status": "not_available",
        "reason": summary["global_direction_test"]["reason"],
        "estimator": summary["estimator"],
        "standard_error_method": summary["standard_error_method"],
    }
    broken = pd.DataFrame(
        [["term", "direction", "body", "wide", 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 2.0]],
        columns=COEFFICIENT_COLUMNS,
    )
    with pytest.raises(ValueError, match="No debe haber coeficientes"):
        validate_reconciliations(baseline_summary, analytic, adjusted, broken, summary)
    broken = pd.DataFrame(
        [["term", "direction", "body", "wide", np.inf, 1.0, 1.0, 0.0, 0.5, 0.5, 2.0]],
        columns=COEFFICIENT_COLUMNS,
    )
    with pytest.raises(ValueError, match="infinitos"):
        validate_reconciliations(baseline_summary, analytic, adjusted, broken, summary)


def test_reconciliation_rejects_pathological_analytic_data():
    baseline_summary, _, _, analytic = analyze_second_serve_direction_baseline(synthetic_points())
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(synthetic_points())
    adjusted = {
        "model_status": summary["model_status"],
        "reason": summary["global_direction_test"]["reason"],
        "estimator": summary["estimator"],
        "standard_error_method": summary["standard_error_method"],
    }
    broken = analytic.iloc[0:0].copy()
    with pytest.raises(ValueError, match="baseline"):
        validate_reconciliations(baseline_summary, broken, adjusted, coefficients, summary)
    broken = analytic.copy()
    broken["direction"] = broken["direction"].astype("object")
    broken.loc[0, "direction"] = "unknown"
    with pytest.raises(ValueError, match="direccion"):
        validate_reconciliations(baseline_summary, broken, adjusted, coefficients, summary)
    broken = analytic.copy()
    broken["server_won_point"] = broken["server_won_point"].astype("object")
    broken.loc[0, "server_won_point"] = "yes"
    with pytest.raises(ValueError, match="binaria"):
        validate_reconciliations(baseline_summary, broken, adjusted, coefficients, summary)


def test_artifact_writing_is_deterministic_and_index_free(tmp_path):
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(synthetic_points())
    write_artifacts(summary, coefficients, tmp_path, tmp_path / "tables")
    first = {path.name: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    write_artifacts(summary, coefficients, tmp_path, tmp_path / "tables")
    second = {path.name: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert first == second
    loaded = pd.read_csv(tmp_path / "tables" / "second_serve_direction_adjusted_coefficients.csv")
    assert loaded.columns.tolist() == COEFFICIENT_COLUMNS
    assert not any(column.startswith("Unnamed") for column in loaded.columns)


@pytest.mark.integration
def test_real_adjusted_analysis_contract_and_population():
    if not POINTS_FILE.exists() or not importlib.util.find_spec("pyarrow"):
        pytest.skip("No estan disponibles points_enriched.parquet y un motor Parquet local.")
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    summary, coefficients, analytic = analyze_second_serve_direction_adjusted(points)
    assert summary["analytical_points"] == 481_190
    assert summary["matches"] == 7_524
    assert summary["servers"] == 1_002
    assert summary["excluded_direction_zero"] == 148
    assert summary["excluded_without_recognized_direction"] == 170
    assert summary["population_by_direction"] == {
        "wide": 176_757,
        "body": 165_841,
        "T": 138_592,
    }
    assert summary["reference_direction"] == "wide"
    assert summary["model_formula"] == MODEL_FORMULA.format(server_reference="Aaron Krickstein")
    assert summary["model_status"] == "not_available"
    assert summary["global_direction_test"]["reason"] == "estimated_dense_design_exceeds_memory_limit"
    assert summary["estimated_dense_design_bytes"] == 3_880_316_160
    assert summary["memory_limit_bytes"] == 536_870_912
    assert len(coefficients) == 0
    assert len(analytic) == 481_190
    assert int(analytic["server_won_point"].sum()) == 245_683
