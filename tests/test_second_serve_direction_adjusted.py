import json
import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.analysis.second_serve_direction_adjusted import (
    COEFFICIENT_COLUMNS,
    MODEL_FORMULA,
    POINTS_FILE,
    SUMMARY_PATH,
    COEFFICIENTS_PATH,
    REFERENCE_DIRECTION,
    _coefficient_row,
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
    assert summary["global_direction_test"]["reason"] == "estimated_sparse_peak_exceeds_memory_limit"
    assert summary["planned_model_rows"] == 162
    assert summary["estimated_dense_design_bytes"] > summary["memory_limit_bytes"]
    assert coefficients.empty


def test_csr_term_metadata_identifies_direction_levels_without_patsy_syntax():
    body = _coefficient_row(
        {"term": "direction[body]", "variable": "direction", "level": "body", "reference_level": "wide", "column_index": 1},
        0.1, 0.2, 0.5, 0.6, -0.3, 0.5,
    )
    tee = _coefficient_row(
        {"term": "direction[T]", "variable": "direction", "level": "T", "reference_level": "wide", "column_index": 2},
        0.2, 0.2, 1.0, 0.3, -0.2, 0.6,
    )
    assert [(row["variable"], row["level"], row["reference_level"]) for row in (body, tee)] == [
        ("direction", "body", "wide"), ("direction", "T", "wide"),
    ]
    with pytest.raises(ValueError, match="Variable de termino desconocida"):
        _coefficient_row(
            {"term": "direction[T]", "variable": "unknown", "level": "T", "reference_level": "wide", "column_index": 2},
            0.2, 0.2, 1.0, 0.3, -0.2, 0.6,
        )


def test_summary_contains_no_nan_or_infinity_and_no_recommendation_language():
    summary, _, _ = analyze_second_serve_direction_adjusted(synthetic_points())
    serialized = json.dumps(summary, allow_nan=False)
    forbidden = ["deberia sacar", "recomendamos", "estrategia optima"]
    assert not any(text in serialized.lower() for text in forbidden)


def test_non_publishable_fit_stops_after_exactly_two_attempts(monkeypatch):
    calls = []
    builder_calls = []
    from src.analysis.sparse_logistic import build_design as sparse_build_design
    def capture_builder(*args, **kwargs):
        design, mapping = sparse_build_design(*args, **kwargs)
        assert sparse.isspmatrix_csr(design)
        builder_calls.append((design.shape, mapping))
        return design, mapping
    def fake_fit(design, outcome, **kwargs):
        calls.append(kwargs)
        diagnostics = {
            "reason_codes": ["optimizer_not_converged", "gradient_above_tolerance"],
            "reason": "optimizer_not_converged", "converged": False,
            "optimizer_success": False, "optimizer": "BFGS", "tolerance": 1e-8,
            "max_iterations": kwargs["max_iterations"],
            "initial_point": "previous_solution" if "initial_coefficients" in kwargs else "zeros",
            "used_warm_start": "initial_coefficients" in kwargs,
            "optimizer_status": 1, "optimizer_message": "not converged", "iterations": 1,
            "objective_value": 2.0, "log_likelihood": -2.0, "gradient_norm": 2.0,
            "max_abs_coefficient": 1.0, "minimum_probability": 0.2,
            "maximum_probability": 0.8, "saturated_probability_count": 0,
            "hessian_status": "invertible", "covariance_status": "conventional_valid",
            "finite_coefficients": True, "finite_probabilities": True, "publishable": False,
        }
        return {"beta": np.zeros(design.shape[1]), "publishable": False, "diagnostics": diagnostics}
    monkeypatch.setattr("src.analysis.second_serve_direction_adjusted.fit_mle", fake_fit)
    monkeypatch.setattr("src.analysis.second_serve_direction_adjusted.build_design", capture_builder)
    monkeypatch.setattr("src.analysis.second_serve_direction_adjusted.cluster_covariance", lambda *args: (_ for _ in ()).throw(AssertionError("No debe calcularse covarianza cluster")))
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(fittable_synthetic_points())
    assert [call["max_iterations"] for call in calls] == [200, 800]
    assert len(builder_calls) == 1
    assert "initial_coefficients" not in calls[0] and "initial_coefficients" in calls[1]
    assert summary["model_status"] == "not_available"
    assert summary["fit_diagnostics"]["reason_codes"] == ["optimizer_not_converged", "gradient_above_tolerance"]
    assert summary["total_iterations"] == 2
    assert summary["cluster_covariance_status"] == "not_calculated_after_rejection"
    assert summary["wald_status"] == "not_calculated_after_rejection"
    assert coefficients.empty


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
def test_real_adjusted_artifacts_contract_and_population(tmp_path):
    if not POINTS_FILE.exists() or not importlib.util.find_spec("pyarrow") or not SUMMARY_PATH.exists() or not COEFFICIENTS_PATH.exists():
        pytest.skip("No estan disponibles los Parquet y artefactos locales necesarios.")
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    baseline_summary, _, _, analytic = analyze_second_serve_direction_baseline(points)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    coefficients = pd.read_csv(COEFFICIENTS_PATH)
    assert summary["analytical_points"] == 481_190
    assert summary["matches"] == 7_524
    assert summary["clusters"] == 7_524
    assert summary["servers"] == 1_002
    assert summary["server_wins"] == 245_683
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
    assert summary["fit_diagnostics"]["reason_codes"] == ["optimizer_not_converged", "gradient_above_tolerance"]
    assert summary["cluster_covariance_status"] == "not_calculated_after_rejection"
    assert summary["wald_status"] == "not_calculated_after_rejection"
    attempts = summary["optimization_attempts"]
    assert len(attempts) == 2 and attempts[0]["initial_point"] == "zeros" and attempts[1]["initial_point"] == "previous_solution"
    assert attempts[0]["iterations"] == 200 and attempts[1]["iterations"] == 388
    assert summary["total_iterations"] == 588
    assert attempts[1]["log_likelihood"] > attempts[0]["log_likelihood"]
    assert attempts[1]["gradient_norm"] < attempts[0]["gradient_norm"]
    assert len(coefficients) == 0
    assert len(analytic) == 481_190
    assert int(analytic["server_won_point"].sum()) == 245_683
    assert "beta" not in json.dumps(summary, allow_nan=False).lower()
    assert baseline_summary["population"]["analytic_points"] == summary["analytical_points"]
    write_artifacts(summary, coefficients, tmp_path / "first", tmp_path / "first" / "tables")
    write_artifacts(summary, coefficients, tmp_path / "second", tmp_path / "second" / "tables")
    assert (tmp_path / "first" / SUMMARY_PATH.name).read_bytes() == (tmp_path / "second" / SUMMARY_PATH.name).read_bytes()
    assert (tmp_path / "first" / "tables" / COEFFICIENTS_PATH.name).read_bytes() == (tmp_path / "second" / "tables" / COEFFICIENTS_PATH.name).read_bytes()
