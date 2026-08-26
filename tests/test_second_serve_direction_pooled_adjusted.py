import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.analysis.second_serve_direction_pooled_adjusted import (
    CLUSTER_VARIABLE,
    COEFFICIENT_COLUMNS,
    COEFFICIENTS_PATH,
    DESIGN_COLUMNS,
    MODEL_FORMULA,
    POINTS_FILE,
    REFERENCES,
    SOURCE_COLUMNS,
    SUMMARY_PATH,
    _fit_diagnostics,
    analyze_second_serve_direction_pooled_adjusted,
    build_design,
    fit_pooled_model,
    validate_analytic_population,
    validate_reconciliations,
    write_artifacts,
)
from src.analysis.second_serve_direction_baseline import build_analytic_population


def synthetic_analytic() -> pd.DataFrame:
    rows = []
    number = 0
    for direction, direction_effect in (("wide", 0), ("body", 1), ("T", -1)):
        for surface, surface_effect in (("Hard", 0), ("Clay", 1), ("Grass", -1)):
            for period, period_effect in (("to_2009", 0), ("2010s", 1), ("2020s", -1)):
                for outcome in (0, 1, 0, 1):
                    rows.append({
                        "match_id": f"m{number}", "point_number": 1,
                        "server_player": f"p{number % 4}", "surface": surface,
                        "derived_period": period, "direction": direction,
                        "server_won_point": bool((outcome + direction_effect + surface_effect + period_effect) % 2),
                    })
                    number += 1
    return pd.DataFrame(rows)


def test_exact_seven_column_design_and_references():
    analytic = synthetic_analytic()
    design = build_design(analytic)
    assert design.columns.tolist() == list(DESIGN_COLUMNS)
    assert REFERENCES == {"direction": "wide", "surface": "Hard", "derived_period": "to_2009"}
    first = design.iloc[0].tolist()
    assert first == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    body_clay_2010 = design[(design["direction[body]"] == 1) & (design["surface[Clay]"] == 1) & (design["derived_period[2010s]"] == 1)].iloc[0]
    assert body_clay_2010.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("column,value", [("direction", "0"), ("surface", "Carpet"), ("derived_period", "future")])
def test_invalid_or_missing_categories_are_rejected(column, value):
    analytic = synthetic_analytic()
    analytic.loc[0, column] = value
    with pytest.raises(ValueError, match="Categorias"):
        build_design(analytic)
    analytic = synthetic_analytic().query("direction != 'T'")
    with pytest.raises(ValueError, match="Categorias"):
        build_design(analytic)


def test_invalid_outcome_clusters_and_rank_are_rejected():
    analytic = synthetic_analytic()
    analytic["server_won_point"] = analytic["server_won_point"].astype(object)
    analytic.loc[0, "server_won_point"] = "yes"
    with pytest.raises(ValueError, match="binario"):
        validate_analytic_population(analytic)
    analytic = synthetic_analytic()
    analytic["match_id"] = "one"
    analytic["point_number"] = np.arange(1, len(analytic) + 1)
    with pytest.raises(ValueError, match="clusters"):
        validate_analytic_population(analytic)
    analytic = synthetic_analytic()
    analytic["server_won_point"] = True
    with pytest.raises(ValueError, match="constante"):
        validate_analytic_population(analytic)
    analytic = synthetic_analytic()
    analytic["surface"] = np.where(analytic["direction"].eq("wide"), "Hard", "Clay")
    analytic.loc[analytic["direction"].eq("T"), "surface"] = "Grass"
    with pytest.raises(ValueError, match="rango"):
        build_design(analytic)


def test_cluster_covariance_matches_independent_sandwich_calculation():
    analytic = synthetic_analytic()
    design = build_design(analytic)
    outcome = analytic["server_won_point"].astype(int).to_numpy()
    clusters = analytic["match_id"].to_numpy()
    result = sm.GLM(outcome, design, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    probability = result.predict(design).to_numpy()
    weights = probability * (1 - probability)
    matrix = design.to_numpy()
    bread = np.linalg.inv(matrix.T @ (matrix * weights[:, None]))
    unique, inverse = np.unique(clusters, return_inverse=True)
    scores = np.vstack([(matrix[inverse == group].T @ (outcome[inverse == group] - probability[inverse == group])) for group in range(len(unique))])
    correction = len(unique) / (len(unique) - 1) * (len(outcome) - 1) / (len(outcome) - matrix.shape[1])
    expected = correction * bread @ (scores.T @ scores) @ bread
    np.testing.assert_allclose(result.cov_params(), expected, rtol=1e-10, atol=1e-10)
    assert result.cov_type == "cluster"


def test_fit_contract_and_csv_are_deterministic(tmp_path):
    adjusted, coefficients = fit_pooled_model(synthetic_analytic())
    assert adjusted["model_status"] == "available"
    assert coefficients.columns.tolist() == COEFFICIENT_COLUMNS
    assert coefficients["term"].tolist() == list(DESIGN_COLUMNS)
    assert np.allclose(coefficients["odds_ratio"], np.exp(coefficients["coefficient"]))
    assert coefficients["ci_low"].gt(0).all() and coefficients["ci_high"].ge(coefficients["ci_low"]).all()
    assert adjusted["global_direction_test"]["df"] == 2


def test_invalid_fit_diagnostics_are_not_publishable():
    analytic = synthetic_analytic()
    design = build_design(analytic)
    outcome = analytic["server_won_point"].astype(int).to_numpy()
    result = sm.GLM(outcome, design, family=sm.families.Binomial()).fit()
    diagnostics = _fit_diagnostics(result, design, outcome, analytic["match_id"])
    assert not diagnostics["publishable"]
    assert "covariance_not_cluster" in diagnostics["reason_codes"]


@pytest.mark.parametrize("field", ["parameters", "probabilities", "covariance"])
def test_non_finite_fit_components_are_not_publishable(field):
    analytic = synthetic_analytic()
    design = build_design(analytic)
    class FakeResult:
        cov_type = "cluster"
        cov_kwds = {"use_correction": True}
        converged = True
        params = np.zeros(len(DESIGN_COLUMNS))
        def predict(self, value): return np.repeat(0.5, len(value))
        def cov_params(self): return np.eye(len(DESIGN_COLUMNS))
    fake = FakeResult()
    if field == "parameters": fake.params[0] = np.nan
    elif field == "probabilities": fake.predict = lambda value: np.repeat(np.nan, len(value))
    else: fake.cov_params = lambda: np.eye(len(DESIGN_COLUMNS)) * np.nan
    diagnostics = _fit_diagnostics(fake, design, analytic["server_won_point"].astype(int).to_numpy(), analytic["match_id"])
    assert not diagnostics["publishable"]


def test_not_available_does_not_publish_inference(monkeypatch):
    analytic = synthetic_analytic()
    class FailingGLM:
        def fit(self, **kwargs):
            raise RuntimeError("synthetic failure")
    monkeypatch.setattr("src.analysis.second_serve_direction_pooled_adjusted.sm.GLM", lambda *args, **kwargs: FailingGLM())
    adjusted, coefficients = fit_pooled_model(analytic)
    assert adjusted["model_status"] == "not_available"
    assert coefficients.empty
    assert adjusted["reason_codes"] == ["fit_failed_RuntimeError"]


def test_separation_is_not_publishable():
    analytic = synthetic_analytic()
    analytic["server_won_point"] = analytic["direction"].ne("wide")
    adjusted, coefficients = fit_pooled_model(analytic)
    assert adjusted["model_status"] == "not_available"
    assert coefficients.empty


def test_reconciliations_reject_manipulated_coefficients_and_population():
    analytic = synthetic_analytic()
    adjusted, coefficients = fit_pooled_model(analytic)
    audit = {"excluded_direction_0": 0, "excluded_without_recognized_direction": 0}
    from src.analysis.second_serve_direction_pooled_adjusted import build_summary
    summary = build_summary(analytic, audit, adjusted, coefficients)
    broken = coefficients.copy(); broken.loc[0, "odds_ratio"] = 0
    with pytest.raises(ValueError, match="OR"):
        validate_reconciliations(analytic, audit, adjusted, broken, summary)
    broken = coefficients.copy(); broken.loc[0, "ci_low"] = -1
    with pytest.raises(ValueError, match="intervalos"):
        validate_reconciliations(analytic, audit, adjusted, broken, summary)
    broken = coefficients.copy(); broken.loc[0, "p_value"] = 2
    with pytest.raises(ValueError, match="p-values"):
        validate_reconciliations(analytic, audit, adjusted, broken, summary)
    broken_adjusted = dict(adjusted); broken_adjusted["global_direction_test"] = {"status": "available", "df": 1, "statistic": 1.0, "p_value": 0.5}
    with pytest.raises(ValueError, match="Wald"):
        validate_reconciliations(analytic, audit, broken_adjusted, coefficients, summary)
    broken_summary = json.loads(json.dumps(summary)); broken_summary["population"]["server_wins"] += 1
    with pytest.raises(ValueError, match="poblacion"):
        validate_reconciliations(analytic, audit, adjusted, coefficients, broken_summary)
    for field, value in (("formula", "y ~ x"), ("standard_error_method", "nonrobust"), ("model_status", "not_available")):
        broken_summary = json.loads(json.dumps(summary)); broken_summary[field] = value
        with pytest.raises(ValueError, match="Campos contractuales"):
            validate_reconciliations(analytic, audit, adjusted, coefficients, broken_summary)
    broken_adjusted = dict(adjusted); broken_adjusted["model_diagnostics"] = dict(adjusted["model_diagnostics"], cov_type="nonrobust")
    with pytest.raises(ValueError, match="covarianza cluster"):
        validate_reconciliations(analytic, audit, broken_adjusted, coefficients)


def test_write_artifacts_has_no_index(tmp_path):
    adjusted, coefficients = fit_pooled_model(synthetic_analytic())
    from src.analysis.second_serve_direction_pooled_adjusted import build_summary
    summary = build_summary(synthetic_analytic(), {"excluded_direction_0": 0, "excluded_without_recognized_direction": 0}, adjusted, coefficients)
    write_artifacts(summary, coefficients, tmp_path, tmp_path / "tables")
    loaded = pd.read_csv(tmp_path / "tables" / COEFFICIENTS_PATH.name)
    assert loaded.columns.tolist() == COEFFICIENT_COLUMNS
    assert not any(column.startswith("Unnamed") for column in loaded.columns)


@pytest.mark.integration
def test_real_pooled_contract_and_artifacts_without_refitting():
    if not POINTS_FILE.exists() or not SUMMARY_PATH.exists() or not COEFFICIENTS_PATH.exists() or not importlib.util.find_spec("pyarrow"):
        pytest.skip("No estan disponibles Parquet y artefactos locales para integracion.")
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    analytic, audit = build_analytic_population(points)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    coefficients = pd.read_csv(COEFFICIENTS_PATH)
    assert len(analytic) == 481_190
    assert analytic["match_id"].nunique() == 7_524
    assert analytic["server_player"].nunique() == 1_002
    assert int(analytic["server_won_point"].sum()) == 245_683
    assert audit["excluded_direction_0"] == 148
    assert audit["excluded_without_recognized_direction"] == 170
    assert summary["population"]["by_direction"] == {"wide": {"points": 176_757, "server_wins": 91_440}, "body": {"points": 165_841, "server_wins": 84_299}, "T": {"points": 138_592, "server_wins": 69_944}}
    assert summary["formula"] == MODEL_FORMULA and summary["categorical_references"] == REFERENCES
    assert summary["standard_error_method"] == "cluster_robust_by_match_id"
    if summary["model_status"] == "available":
        assert summary["model_diagnostics"]["cov_type"] == "cluster"
        assert len(coefficients) == len(DESIGN_COLUMNS)
    else:
        assert coefficients.empty
