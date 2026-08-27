import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.analysis.second_serve_direction_heterogeneity import (
    CLUSTER_VARIABLE,
    CONTRAST_COLUMNS,
    CONTRASTS_PATH,
    MARGIN_COLUMNS,
    MARGINS_PATH,
    MODEL_SPECS,
    POINTS_FILE,
    REFERENCES,
    SOURCE_COLUMNS,
    SUMMARY_PATH,
    _counterfactual_design,
    _diagnostics,
    _margin,
    _wald,
    analyze_second_serve_direction_heterogeneity,
    build_design,
    build_summary,
    fit_model,
    holm_adjust,
    margins_and_contrasts,
    stability_summary,
    validate_population,
    validate_reconciliations,
    write_artifacts,
)
from src.analysis.second_serve_direction_baseline import build_analytic_population


def synthetic_analytic() -> pd.DataFrame:
    """Poblacion completa, no separada y con clusters repetidos."""
    generator = np.random.default_rng(20260826)
    rows, index = [], 0
    for direction, direction_effect in (("wide", 0.0), ("body", 0.18), ("T", -0.08)):
        for surface, surface_effect in (("Hard", 0.0), ("Clay", 0.05), ("Grass", -0.04)):
            for period, period_effect in (("to_2009", 0.0), ("2010s", 0.03), ("2020s", -0.02)):
                for replicate in range(18):
                    probability = 0.47 + direction_effect + surface_effect + period_effect
                    rows.append({
                        "match_id": f"match_{index // 6:03d}", "point_number": index % 6 + 1,
                        "server_player": f"player_{index % 7}", "surface": surface,
                        "derived_period": period, "direction": direction,
                        "server_won_point": int(generator.random() < probability),
                    })
                    index += 1
    return pd.DataFrame(rows)


def available_fits(analytic: pd.DataFrame):
    statuses, results = {}, {}
    for model_name in MODEL_SPECS:
        statuses[model_name], results[model_name], _ = fit_model(analytic, model_name)
        assert statuses[model_name]["status"] == "available"
    return statuses, results


def test_designs_are_explicit_and_interactions_have_expected_columns():
    analytic = synthetic_analytic()
    additive = build_design(analytic, "additive")
    surface = build_design(analytic, "surface_interaction")
    period = build_design(analytic, "period_interaction")
    assert additive.columns.tolist() == [
        "Intercept", "direction[body]", "direction[T]", "surface[Clay]", "surface[Grass]",
        "derived_period[2010s]", "derived_period[2020s]",
    ]
    assert surface.columns.tolist()[-4:] == [
        "direction[body]:surface[Clay]", "direction[body]:surface[Grass]",
        "direction[T]:surface[Clay]", "direction[T]:surface[Grass]",
    ]
    assert period.columns.tolist()[-4:] == [
        "direction[body]:derived_period[2010s]", "direction[body]:derived_period[2020s]",
        "direction[T]:derived_period[2010s]", "direction[T]:derived_period[2020s]",
    ]
    row = surface[(surface["direction[body]"] == 1) & (surface["surface[Clay]"] == 1)].iloc[0]
    assert row["direction[body]:surface[Clay]"] == 1.0
    assert row["direction[body]:surface[Grass]"] == 0.0
    assert REFERENCES == {"direction": "wide", "surface": "Hard", "derived_period": "to_2009"}
    assert [MODEL_SPECS[name]["formula"] for name in MODEL_SPECS] == [
        "server_won_point ~ direction + surface + derived_period",
        "server_won_point ~ direction * surface + derived_period",
        "server_won_point ~ direction * derived_period + surface",
    ]


@pytest.mark.parametrize("column,value", [("direction", "unknown"), ("surface", "Carpet"), ("derived_period", "future")])
def test_invalid_or_missing_categories_are_rejected(column, value):
    analytic = synthetic_analytic()
    analytic.loc[0, column] = value
    with pytest.raises(ValueError, match="Categorias"):
        build_design(analytic, "additive")
    analytic = synthetic_analytic().query("direction != 'T'")
    with pytest.raises(ValueError, match="Categorias"):
        validate_population(analytic)


def test_invalid_population_rank_and_constant_outcome_are_rejected():
    analytic = synthetic_analytic()
    analytic["server_won_point"] = 1
    with pytest.raises(ValueError, match="constante"):
        validate_population(analytic)
    analytic = synthetic_analytic()
    analytic["match_id"] = "one"
    analytic["point_number"] = np.arange(1, len(analytic) + 1)
    with pytest.raises(ValueError, match="Clusters"):
        validate_population(analytic)
    analytic = synthetic_analytic()
    analytic["surface"] = np.where(analytic["direction"].eq("wide"), "Hard", "Clay")
    analytic.loc[analytic["direction"].eq("T"), "surface"] = "Grass"
    with pytest.raises(ValueError, match="rango"):
        build_design(analytic, "surface_interaction")


def test_cluster_covariance_and_wald_follow_statsmodels_contract():
    analytic = synthetic_analytic()
    status, result, design = fit_model(analytic, "surface_interaction")
    assert status["status"] == "available"
    direct = sm.GLM(analytic["server_won_point"], design, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": analytic[CLUSTER_VARIABLE].to_numpy()},
    )
    np.testing.assert_allclose(result.cov_params(), direct.cov_params(), rtol=1e-11, atol=1e-11)
    test = _wald(result, [
        "direction[body]:surface[Clay]", "direction[T]:surface[Clay]",
        "direction[body]:surface[Grass]", "direction[T]:surface[Grass]",
    ])
    assert test["df"] == 4 and 0 <= test["p_value"] <= 1


def test_standardized_margin_gradient_and_delta_contrast_are_numeric():
    analytic = synthetic_analytic()
    _, result, _ = fit_model(analytic, "surface_interaction")
    probability, gradient, standard_error = _margin(result, analytic, "surface_interaction", "body", "surface", "Clay")
    design = _counterfactual_design(analytic, "surface_interaction", "body", "surface", "Clay").to_numpy()
    beta = np.asarray(result.params, dtype=float)
    numerical = []
    for column in range(len(beta)):
        shifted = beta.copy(); shifted[column] += 1e-6
        numerical.append(((1 / (1 + np.exp(-(design @ shifted)))).mean() - probability) / 1e-6)
    np.testing.assert_allclose(gradient, numerical, rtol=3e-5, atol=3e-6)
    assert 0 < probability < 1 and standard_error > 0
    margins, contrasts = margins_and_contrasts(result, analytic, "surface_interaction", "surface")
    clay = contrasts.query("stratum == 'Clay' and contrast == 'body_minus_wide'").iloc[0]
    values = margins.query("stratum == 'Clay'").set_index("direction")["adjusted_probability"]
    assert clay["difference"] == pytest.approx(values["body"] - values["wide"])
    assert clay["standard_error"] > 0


def test_holm_is_monotone_and_uses_the_six_contrast_family():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.002, 0.90, 0.20])
    assert adjusted == pytest.approx([0.05, 0.12, 0.12, 0.012, 0.90, 0.40])
    analytic = synthetic_analytic()
    _, result, _ = fit_model(analytic, "period_interaction")
    _, contrasts = margins_and_contrasts(result, analytic, "period_interaction", "period")
    assert len(contrasts) == 6
    assert contrasts["p_value_holm"].ge(contrasts["p_value_raw"]).all()
    assert set(contrasts["holm_family"]) == {"period"}


def test_stability_summary_uses_stratified_contrasts_without_causal_claims():
    analytic = synthetic_analytic()
    statuses, results = available_fits(analytic)
    global_tests = {
        "surface": _wald(results["surface_interaction"], list(build_design(analytic, "surface_interaction").columns[-4:])),
        "period": _wald(results["period_interaction"], list(build_design(analytic, "period_interaction").columns[-4:])),
    }
    _, surface = margins_and_contrasts(results["surface_interaction"], analytic, "surface_interaction", "surface")
    _, period = margins_and_contrasts(results["period_interaction"], analytic, "period_interaction", "period")
    summary = stability_summary(pd.concat([surface, period], ignore_index=True), global_tests)
    assert set(summary) == {"surface", "period"}
    assert len(summary["surface"]["coverage"]) == 3
    assert set(summary["period"]["contrasts"]) == {"body_minus_wide", "T_minus_wide"}
    changed = pd.concat([surface, period], ignore_index=True)
    body_surface = (changed["stratum_type"] == "surface") & (changed["contrast"] == "body_minus_wide")
    changed.loc[body_surface, "difference"] = -0.01
    target = changed.index[body_surface][0]
    changed.loc[target, "difference"] = 0.01
    assert stability_summary(changed, global_tests)["surface"]["contrasts"]["body_minus_wide"]["sign_changes_between_strata"]


def test_full_analysis_uses_canonical_design_order_for_wald(monkeypatch):
    analytic = synthetic_analytic()
    monkeypatch.setattr(
        "src.analysis.second_serve_direction_heterogeneity.build_analytic_population",
        lambda points: (analytic, {"excluded_direction_0": 0, "excluded_without_recognized_direction": 0}),
    )
    summary, margins, contrasts, returned = analyze_second_serve_direction_heterogeneity(pd.DataFrame())
    assert returned is analytic
    assert summary["global_interaction_tests"]["surface"]["terms"] == list(build_design(analytic, "surface_interaction").columns[-4:])
    assert summary["global_interaction_tests"]["period"]["terms"] == list(build_design(analytic, "period_interaction").columns[-4:])
    assert len(margins) == 18 and len(contrasts) == 12


def test_failed_model_never_publishes_partial_inference(monkeypatch):
    analytic = synthetic_analytic()
    class FailingFit:
        def fit(self, **kwargs):
            raise RuntimeError("synthetic failure")
    monkeypatch.setattr("src.analysis.second_serve_direction_heterogeneity.sm.GLM", lambda *args, **kwargs: FailingFit())
    status, result, _ = fit_model(analytic, "additive")
    assert status["status"] == "not_available"
    assert result is None and status["reason_codes"] == ["fit_failed_RuntimeError"]


def test_one_unavailable_model_suppresses_all_inference(monkeypatch):
    analytic = synthetic_analytic()
    real_fit = fit_model
    monkeypatch.setattr(
        "src.analysis.second_serve_direction_heterogeneity.build_analytic_population",
        lambda points: (analytic, {"excluded_direction_0": 0, "excluded_without_recognized_direction": 0}),
    )
    def controlled_fit(frame, model_name):
        if model_name == "additive":
            return {"status": "not_available", "reason_codes": ["synthetic_failure"], "diagnostics": {"publishable": False}}, None, build_design(frame, model_name)
        return real_fit(frame, model_name)
    monkeypatch.setattr("src.analysis.second_serve_direction_heterogeneity.fit_model", controlled_fit)
    summary, margins, contrasts, _ = analyze_second_serve_direction_heterogeneity(pd.DataFrame())
    assert margins.empty and contrasts.empty
    assert summary["stability_summary"] == {"status": "not_available"}
    assert all(test["status"] == "not_available" for test in summary["global_interaction_tests"].values())
    assert not summary["reconciliations"]["published_values_reconciled"]


@pytest.mark.parametrize("component", ["parameters", "probabilities", "covariance"])
def test_nonfinite_or_invalid_fit_diagnostics_are_not_publishable(component):
    analytic = synthetic_analytic()
    design = build_design(analytic, "additive")
    class FakeResult:
        cov_type = "cluster"
        cov_kwds = {"use_correction": True}
        converged = True
        params = np.zeros(design.shape[1])
        def predict(self, value): return np.repeat(0.5, len(value))
        def cov_params(self): return np.eye(design.shape[1])
    fake = FakeResult()
    if component == "parameters":
        fake.params[0] = np.nan
    elif component == "probabilities":
        fake.predict = lambda value: np.repeat(np.inf, len(value))
    else:
        fake.cov_params = lambda: np.diag([-1.0] + [1.0] * (design.shape[1] - 1))
    diagnostics = _diagnostics(fake, design, analytic["server_won_point"].to_numpy(), analytic["match_id"])
    assert not diagnostics["publishable"]


def test_perfect_separation_is_not_publishable():
    analytic = synthetic_analytic()
    analytic["server_won_point"] = analytic["direction"].ne("wide").astype(int)
    status, result, _ = fit_model(analytic, "additive")
    assert status["status"] == "not_available"
    assert result is None


def test_reconciliations_reject_missing_duplicate_and_manipulated_rows():
    analytic = synthetic_analytic()
    statuses, results = available_fits(analytic)
    global_tests = {
        "surface": _wald(results["surface_interaction"], list(build_design(analytic, "surface_interaction").columns[-4:])),
        "period": _wald(results["period_interaction"], list(build_design(analytic, "period_interaction").columns[-4:])),
    }
    surface_margins, surface_contrasts = margins_and_contrasts(results["surface_interaction"], analytic, "surface_interaction", "surface")
    period_margins, period_contrasts = margins_and_contrasts(results["period_interaction"], analytic, "period_interaction", "period")
    margins = pd.concat([surface_margins, period_margins], ignore_index=True)
    contrasts = pd.concat([surface_contrasts, period_contrasts], ignore_index=True)
    audit = {"excluded_direction_0": 0, "excluded_without_recognized_direction": 0}
    summary = build_summary(analytic, audit, statuses, global_tests, margins, contrasts, results)
    assert validate_reconciliations(analytic, audit, statuses, global_tests, margins, contrasts, summary, results)["published_values_reconciled"]
    with pytest.raises(ValueError, match="Claves de margenes"):
        validate_reconciliations(analytic, audit, statuses, global_tests, margins.iloc[1:], contrasts, summary, results)
    duplicate = pd.concat([contrasts, contrasts.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Contrastes invalidos"):
        validate_reconciliations(analytic, audit, statuses, global_tests, margins, duplicate, summary, results)
    broken = contrasts.copy(); broken.loc[0, "difference"] += 0.1
    with pytest.raises(ValueError, match="Valores de contrastes"):
        validate_reconciliations(analytic, audit, statuses, global_tests, margins, broken, summary, results)
    broken = contrasts.copy(); broken.loc[0, "p_value_holm"] += 0.01
    with pytest.raises(ValueError, match="Valores de contrastes"):
        validate_reconciliations(analytic, audit, statuses, global_tests, margins, broken, summary, results)
    broken_tests = json.loads(json.dumps(global_tests)); broken_tests["surface"]["statistic"] += 1
    with pytest.raises(ValueError, match="Wald global"):
        validate_reconciliations(analytic, audit, statuses, broken_tests, margins, contrasts, summary, results)
    broken_summary = json.loads(json.dumps(summary)); broken_summary["population"]["server_wins"] += 1
    with pytest.raises(ValueError, match="Resumen"):
        validate_reconciliations(analytic, audit, statuses, global_tests, margins, contrasts, broken_summary, results)


def test_artifacts_are_deterministic_and_csvs_have_no_index(tmp_path):
    analytic = synthetic_analytic()
    statuses, results = available_fits(analytic)
    global_tests = {
        "surface": _wald(results["surface_interaction"], list(build_design(analytic, "surface_interaction").columns[-4:])),
        "period": _wald(results["period_interaction"], list(build_design(analytic, "period_interaction").columns[-4:])),
    }
    surface_margins, surface_contrasts = margins_and_contrasts(results["surface_interaction"], analytic, "surface_interaction", "surface")
    period_margins, period_contrasts = margins_and_contrasts(results["period_interaction"], analytic, "period_interaction", "period")
    margins = pd.concat([surface_margins, period_margins], ignore_index=True)
    contrasts = pd.concat([surface_contrasts, period_contrasts], ignore_index=True)
    summary = build_summary(analytic, {"excluded_direction_0": 0, "excluded_without_recognized_direction": 0}, statuses, global_tests, margins, contrasts, results)
    write_artifacts(summary, margins, contrasts, tmp_path, tmp_path / "tables")
    assert list(pd.read_csv(tmp_path / "tables" / MARGINS_PATH.name).columns) == MARGIN_COLUMNS
    assert list(pd.read_csv(tmp_path / "tables" / CONTRASTS_PATH.name).columns) == CONTRAST_COLUMNS
    assert not any(name.startswith("Unnamed") for name in pd.read_csv(tmp_path / "tables" / MARGINS_PATH.name).columns)
    assert json.loads((tmp_path / SUMMARY_PATH.name).read_text(encoding="utf-8"))["analysis_name"] == "second_serve_direction_heterogeneity"


@pytest.mark.integration
def test_real_heterogeneity_artifacts_without_refitting():
    if not POINTS_FILE.exists() or not SUMMARY_PATH.exists() or not MARGINS_PATH.exists() or not CONTRASTS_PATH.exists() or not importlib.util.find_spec("pyarrow"):
        pytest.skip("No estan disponibles Parquet y artefactos locales para integracion.")
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    analytic, audit = build_analytic_population(points)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    margins, contrasts = pd.read_csv(MARGINS_PATH), pd.read_csv(CONTRASTS_PATH)
    assert len(analytic) == 481_190 and analytic["match_id"].nunique() == 7_524
    assert analytic["server_player"].nunique() == 1_002 and int(analytic["server_won_point"].sum()) == 245_683
    assert audit["excluded_direction_0"] == 148 and audit["excluded_without_recognized_direction"] == 170
    assert summary["population"]["by_direction"] == {"wide": {"points": 176_757, "server_wins": 91_440}, "body": {"points": 165_841, "server_wins": 84_299}, "T": {"points": 138_592, "server_wins": 69_944}}
    assert set(summary["model_statuses"]) == set(MODEL_SPECS)
    if all(value["status"] == "available" for value in summary["model_statuses"].values()):
        assert len(margins) == 18 and len(contrasts) == 12
        assert set(contrasts["holm_family"]) == {"surface", "period"}
