"""Robustez y heterogeneidad exploratoria del GLM pooled de segundo saque."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.analysis.second_serve_direction_analysis import (
    PERIOD_ORDER, POINTS_FILE, REPORTS_DIR, SOURCE_COLUMNS, SURFACE_ORDER, TABLES_DIR,
)
from src.analysis.second_serve_direction_baseline import DIRECTION_ORDER, build_analytic_population

ANALYSIS_NAME = "second_serve_direction_heterogeneity"
VERSION = 1
REFERENCES = {"direction": "wide", "surface": "Hard", "derived_period": "to_2009"}
CLUSTER_VARIABLE = "match_id"
COEFFICIENT_LIMIT = 30.0
SATURATION_TOLERANCE = 1e-12
SUMMARY_PATH = REPORTS_DIR / "second_serve_direction_heterogeneity_summary.json"
CONTRASTS_PATH = TABLES_DIR / "second_serve_direction_heterogeneity_contrasts.csv"
MARGINS_PATH = TABLES_DIR / "second_serve_direction_heterogeneity_margins.csv"

MODEL_SPECS = {
    "additive": {
        "formula": "server_won_point ~ direction + surface + derived_period",
        "interaction": None,
    },
    "surface_interaction": {
        "formula": "server_won_point ~ direction * surface + derived_period",
        "interaction": "surface",
    },
    "period_interaction": {
        "formula": "server_won_point ~ direction * derived_period + surface",
        "interaction": "period",
    },
}
STRATUM_COLUMNS = {
    "surface": ("surface", SURFACE_ORDER),
    "period": ("derived_period", PERIOD_ORDER),
}
MARGIN_COLUMNS = [
    "model", "stratum_type", "stratum", "direction", "analytical_points", "matches",
    "adjusted_probability", "standard_error", "ci_low", "ci_high",
]
CONTRAST_COLUMNS = [
    "model", "stratum_type", "stratum", "contrast", "analytical_points", "matches",
    "difference", "standard_error", "ci_low", "ci_high", "z", "p_value_raw",
    "p_value_holm", "holm_family",
]


def _finite_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("El resumen contiene NaN o infinito.")
    if isinstance(value, dict):
        for item in value.values(): _finite_json(item)
    if isinstance(value, list):
        for item in value: _finite_json(item)


def validate_population(analytic: pd.DataFrame) -> None:
    required = {"match_id", "point_number", "server_player", "surface", "derived_period", "direction", "server_won_point"}
    if required - set(analytic.columns) or analytic.empty or analytic.duplicated(["match_id", "point_number"]).any():
        raise ValueError("Poblacion analitica invalida.")
    if not analytic["server_won_point"].isin([True, False, 0, 1]).all() or analytic["server_won_point"].nunique() != 2:
        raise ValueError("Outcome binario invalido o constante.")
    for column, levels in (("direction", DIRECTION_ORDER), ("surface", SURFACE_ORDER), ("derived_period", PERIOD_ORDER)):
        if set(analytic[column].astype(str)) != set(levels):
            raise ValueError(f"Categorias invalidas o ausentes en {column}.")
    if analytic[CLUSTER_VARIABLE].nunique() < 2:
        raise ValueError("Clusters insuficientes.")


def _base_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "Intercept": np.ones(len(frame)),
        "direction[body]": frame["direction"].astype(str).eq("body").to_numpy(float),
        "direction[T]": frame["direction"].astype(str).eq("T").to_numpy(float),
        "surface[Clay]": frame["surface"].astype(str).eq("Clay").to_numpy(float),
        "surface[Grass]": frame["surface"].astype(str).eq("Grass").to_numpy(float),
        "derived_period[2010s]": frame["derived_period"].astype(str).eq("2010s").to_numpy(float),
        "derived_period[2020s]": frame["derived_period"].astype(str).eq("2020s").to_numpy(float),
    }


def _build_design_from_validated_frame(analytic: pd.DataFrame, model_name: str, *, check_rank: bool = True) -> pd.DataFrame:
    if model_name not in MODEL_SPECS: raise ValueError("Modelo desconocido.")
    columns = _base_columns(analytic)
    if model_name == "surface_interaction":
        for direction in ("body", "T"):
            for surface in ("Clay", "Grass"):
                columns[f"direction[{direction}]:surface[{surface}]"] = columns[f"direction[{direction}]"] * columns[f"surface[{surface}]"]
    elif model_name == "period_interaction":
        for direction in ("body", "T"):
            for period in ("2010s", "2020s"):
                columns[f"direction[{direction}]:derived_period[{period}]"] = columns[f"direction[{direction}]"] * columns[f"derived_period[{period}]"]
    design = pd.DataFrame(columns, index=analytic.index)
    if check_rank and np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Diseno sin rango completo.")
    return design


def build_design(analytic: pd.DataFrame, model_name: str) -> pd.DataFrame:
    validate_population(analytic)
    return _build_design_from_validated_frame(analytic, model_name)


def _diagnostics(result: Any, design: pd.DataFrame, outcome: np.ndarray, clusters: pd.Series) -> dict[str, Any]:
    params, probabilities, covariance = np.asarray(result.params), np.asarray(result.predict(design)), np.asarray(result.cov_params())
    cov_kwds = getattr(result, "cov_kwds", {})
    reasons = []
    if getattr(result, "cov_type", None) != "cluster": reasons.append("covariance_not_cluster")
    if not cov_kwds.get("use_correction", False): reasons.append("cluster_correction_not_registered")
    if not getattr(result, "converged", False): reasons.append("not_converged")
    if not np.isfinite(params).all(): reasons.append("non_finite_parameters")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)): reasons.append("invalid_probabilities")
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T) or np.any(np.diag(covariance) < 0): reasons.append("invalid_covariance")
    saturated = int(np.count_nonzero((probabilities <= SATURATION_TOLERANCE) | (probabilities >= 1 - SATURATION_TOLERANCE)))
    if saturated: reasons.append("saturated_probabilities")
    if np.max(np.abs(params)) > COEFFICIENT_LIMIT: reasons.append("extreme_coefficients")
    return {
        "reason_codes": reasons, "publishable": not reasons, "converged": bool(result.converged),
        "cov_type": getattr(result, "cov_type", None), "cluster_finite_sample_correction": bool(cov_kwds.get("use_correction", False)),
        "clusters": int(clusters.nunique()), "design_rank": int(np.linalg.matrix_rank(design.to_numpy())),
        "design_columns": int(design.shape[1]), "parameters_finite": bool(np.isfinite(params).all()),
        "probabilities_finite": bool(np.isfinite(probabilities).all()), "covariance_finite": bool(np.isfinite(covariance).all()),
        "covariance_symmetric": bool(np.allclose(covariance, covariance.T)),
        "covariance_nonnegative_diagonal": bool(np.all(np.diag(covariance) >= 0)),
        "minimum_probability": float(probabilities.min()), "maximum_probability": float(probabilities.max()),
        "saturated_probability_count": saturated, "max_abs_coefficient": float(np.abs(params).max()),
    }


def fit_model(analytic: pd.DataFrame, model_name: str) -> tuple[dict[str, Any], Any | None, pd.DataFrame]:
    design = build_design(analytic, model_name)
    try:
        result = sm.GLM(analytic["server_won_point"].astype(int).to_numpy(), design, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": analytic[CLUSTER_VARIABLE].to_numpy()})
        diagnostics = _diagnostics(result, design, analytic["server_won_point"].to_numpy(), analytic[CLUSTER_VARIABLE])
    except Exception as exc:
        return {"status": "not_available", "reason_codes": [f"fit_failed_{type(exc).__name__}"], "diagnostics": {"publishable": False, "exception_type": type(exc).__name__}}, None, design
    if not diagnostics["publishable"]:
        return {"status": "not_available", "reason_codes": diagnostics["reason_codes"], "diagnostics": diagnostics}, None, design
    return {"status": "available", "reason_codes": [], "diagnostics": diagnostics, "log_likelihood": float(result.llf), "aic": float(result.aic), "n_observations": int(result.nobs)}, result, design


def _wald(result: Any, terms: list[str]) -> dict[str, Any]:
    indices = [list(result.params.index).index(term) for term in terms]
    beta = np.asarray(result.params)[indices]
    covariance = np.asarray(result.cov_params())[np.ix_(indices, indices)]
    if np.linalg.matrix_rank(covariance) != len(indices): raise ValueError("Wald no identificable.")
    statistic = float(beta @ np.linalg.solve(covariance, beta))
    return {"status": "available", "statistic": statistic, "df": len(indices), "p_value": float(stats.chi2.sf(statistic, len(indices))), "terms": terms}


def _counterfactual_design(analytic: pd.DataFrame, model_name: str, direction: str, stratum_type: str, stratum: str) -> pd.DataFrame:
    column, _ = STRATUM_COLUMNS[stratum_type]
    frame = analytic.copy()
    frame["direction"] = direction
    frame[column] = stratum
    # El contrafactual conserva los metadatos observados pero, por definicion,
    # fija una categoria; no debe exigir que sigan presentes las tres direcciones.
    return _build_design_from_validated_frame(frame, model_name, check_rank=False)


def _margin(result: Any, analytic: pd.DataFrame, model_name: str, direction: str, stratum_type: str, stratum: str) -> tuple[float, np.ndarray, float]:
    design = _counterfactual_design(analytic, model_name, direction, stratum_type, stratum)
    beta, covariance = np.asarray(result.params), np.asarray(result.cov_params())
    probability = 1 / (1 + np.exp(-np.asarray(design) @ beta))
    gradient = np.asarray(design).T @ (probability * (1 - probability)) / len(design)
    standard_error = float(np.sqrt(gradient @ covariance @ gradient))
    return float(probability.mean()), gradient, standard_error


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index])); adjusted[index] = running
    return adjusted.tolist()


def margins_and_contrasts(result: Any, analytic: pd.DataFrame, model_name: str, stratum_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    column, strata = STRATUM_COLUMNS[stratum_type]
    margin_rows, contrast_rows = [], []
    covariance = np.asarray(result.cov_params())
    for stratum in strata:
        observed = analytic[analytic[column].astype(str).eq(stratum)]
        values = {}
        for direction in DIRECTION_ORDER:
            probability, gradient, standard_error = _margin(result, analytic, model_name, direction, stratum_type, stratum)
            values[direction] = (probability, gradient)
            margin_rows.append({"model": model_name, "stratum_type": stratum_type, "stratum": stratum, "direction": direction, "analytical_points": int(len(observed)), "matches": int(observed["match_id"].nunique()), "adjusted_probability": probability, "standard_error": standard_error, "ci_low": probability - 1.959963984540054 * standard_error, "ci_high": probability + 1.959963984540054 * standard_error})
        for direction in ("body", "T"):
            difference = values[direction][0] - values["wide"][0]
            gradient = values[direction][1] - values["wide"][1]
            standard_error = float(np.sqrt(gradient @ covariance @ gradient))
            z = difference / standard_error
            contrast_rows.append({"model": model_name, "stratum_type": stratum_type, "stratum": stratum, "contrast": f"{direction}_minus_wide", "analytical_points": int(len(observed)), "matches": int(observed["match_id"].nunique()), "difference": difference, "standard_error": standard_error, "ci_low": difference - 1.959963984540054 * standard_error, "ci_high": difference + 1.959963984540054 * standard_error, "z": z, "p_value_raw": float(2 * stats.norm.sf(abs(z))), "holm_family": stratum_type})
    contrasts = pd.DataFrame(contrast_rows, columns=CONTRAST_COLUMNS)
    contrasts["p_value_holm"] = holm_adjust(contrasts["p_value_raw"].tolist())
    return pd.DataFrame(margin_rows, columns=MARGIN_COLUMNS), contrasts[CONTRAST_COLUMNS]


def _population(analytic: pd.DataFrame) -> dict[str, Any]:
    by_direction = analytic.groupby("direction", observed=True)["server_won_point"].agg(["size", "sum"])
    return {"analytical_points": int(len(analytic)), "matches": int(analytic["match_id"].nunique()), "clusters": int(analytic["match_id"].nunique()), "servers": int(analytic["server_player"].nunique()), "server_wins": int(analytic["server_won_point"].sum()), "by_direction": {d: {"points": int(by_direction.loc[d, "size"]), "server_wins": int(by_direction.loc[d, "sum"])} for d in DIRECTION_ORDER}}


def stability_summary(contrasts: pd.DataFrame, global_tests: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for family in ("surface", "period"):
        rows = contrasts[contrasts["stratum_type"].eq(family)]
        item = {}
        for contrast in ("body_minus_wide", "T_minus_wide"):
            values = rows.loc[rows["contrast"].eq(contrast), "difference"]
            signs = set(np.sign(values[values.ne(0)]))
            item[contrast] = {"sign_changes_between_strata": len(signs) > 1, "difference_min": float(values.min()), "difference_max": float(values.max())}
        output[family] = {"global_interaction_test": global_tests[family], "coverage": rows[["stratum", "analytical_points", "matches"]].drop_duplicates().to_dict("records"), "contrasts": item}
    return output


def build_summary(analytic: pd.DataFrame, audit: dict[str, Any], statuses: dict[str, Any], global_tests: dict[str, Any], margins: pd.DataFrame, contrasts: pd.DataFrame, results: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {"analysis_name": ANALYSIS_NAME, "version": VERSION, "source": "data/processed/points_enriched.parquet", "population": _population(analytic), "exclusions": {"direction_0": audit["excluded_direction_0"], "without_recognized_direction": audit["excluded_without_recognized_direction"]}, "model_contracts": {name: {"formula": spec["formula"], "references": REFERENCES} for name, spec in MODEL_SPECS.items()}, "model_statuses": statuses, "global_interaction_tests": global_tests, "multiple_testing": {"method": "Holm", "families": {"surface": 6, "period": 6}}, "publication_contract": "all_three_models_must_be_available", "methodological_limits": ["Asociacion poblacional, no causal.", "El ajuste se limita a direccion, superficie, periodo y sus interacciones preespecificadas.", "Posible confusion residual por servidor.", "Estratos definidos a priori; heterogeneidad exploratoria.", "Correccion Holm dentro de cada familia de seis contrastes.", "Dependencia agrupada por partido, no por servidor.", "Un Wald no significativo no demuestra homogeneidad.", "El signo, la significacion estadistica y la relevancia practica son dimensiones distintas.", "No constituye recomendacion tactica personalizada."], "margins_contract": {"columns": MARGIN_COLUMNS, "rows": int(len(margins)), "standardization": "Distribucion observada de las demas covariables."}, "contrasts_contract": {"columns": CONTRAST_COLUMNS, "rows": int(len(contrasts)), "scale": "probability", "delta_method": True}}
    summary["stability_summary"] = stability_summary(contrasts, global_tests) if len(contrasts) == 12 else {"status": "not_available"}
    summary["reconciliations"] = validate_reconciliations(analytic, audit, statuses, global_tests, margins, contrasts, summary, results)
    _finite_json(summary); return summary


def _compare_published_table(actual: pd.DataFrame, expected: pd.DataFrame, key_columns: list[str], label: str) -> None:
    """Comprueba claves y valores sin depender de sumas agregadas."""
    if actual.columns.tolist() != expected.columns.tolist():
        raise ValueError(f"Esquema de {label} no reconciliado.")
    if actual.duplicated(key_columns).any():
        raise ValueError(f"Claves duplicadas en {label}.")
    actual = actual.sort_values(key_columns).reset_index(drop=True)
    expected = expected.sort_values(key_columns).reset_index(drop=True)
    if not actual[key_columns].equals(expected[key_columns]):
        raise ValueError(f"Claves de {label} no reconciliadas.")
    for column in actual.columns:
        if column in key_columns:
            continue
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(actual[column].to_numpy(float), expected[column].to_numpy(float), rtol=1e-10, atol=1e-12, equal_nan=False):
                raise ValueError(f"Valores de {label} no reconciliados: {column}.")
        elif not actual[column].equals(expected[column]):
            raise ValueError(f"Valores de {label} no reconciliados: {column}.")


def validate_reconciliations(analytic: pd.DataFrame, audit: dict[str, Any], statuses: dict[str, Any], global_tests: dict[str, Any], margins: pd.DataFrame, contrasts: pd.DataFrame, summary: dict[str, Any] | None = None, results: dict[str, Any] | None = None) -> dict[str, bool]:
    population = _population(analytic)
    all_models_available = all(statuses[name]["status"] == "available" for name in MODEL_SPECS)
    if len(analytic) > 1_000 and population["analytical_points"] != 481_190: raise ValueError("Poblacion real no reconciliada.")
    if any(statuses[name]["diagnostics"].get("clusters") != population["clusters"] for name in statuses if statuses[name]["status"] == "available"): raise ValueError("Clusters no reconciliados.")
    for family, expected_model, expected_df in (("surface", "surface_interaction", 4), ("period", "period_interaction", 4)):
        test = global_tests[family]
        if all_models_available and statuses[expected_model]["status"] == "available":
            if test.get("status") != "available" or test.get("df") != expected_df or not 0 <= test.get("p_value", np.nan) <= 1: raise ValueError("Wald global invalido.")
    if not margins.empty and (margins["adjusted_probability"].lt(0).any() or margins["adjusted_probability"].gt(1).any() or margins["ci_low"].isna().any()): raise ValueError("Margenes invalidos.")
    if not contrasts.empty:
        if contrasts.duplicated(["model", "stratum_type", "stratum", "contrast"]).any() or not contrasts["p_value_raw"].between(0, 1).all() or not contrasts["p_value_holm"].between(0, 1).all() or contrasts["p_value_holm"].lt(contrasts["p_value_raw"]).any(): raise ValueError("Contrastes invalidos.")
    if not all_models_available and (not margins.empty or not contrasts.empty or any(test.get("status") == "available" for test in global_tests.values())):
        raise ValueError("No se permite inferencia parcial si algun modelo no esta disponible.")
    expected_margin_keys, expected_contrast_keys = set(), set()
    for family, model_name in (("surface", "surface_interaction"), ("period", "period_interaction")):
        if not all_models_available or statuses[model_name]["status"] != "available":
            continue
        _, strata = STRATUM_COLUMNS[family]
        expected_margin_keys.update((model_name, family, stratum, direction) for stratum in strata for direction in DIRECTION_ORDER)
        expected_contrast_keys.update((model_name, family, stratum, contrast) for stratum in strata for contrast in ("body_minus_wide", "T_minus_wide"))
    actual_margin_keys = set(map(tuple, margins[["model", "stratum_type", "stratum", "direction"]].to_numpy())) if not margins.empty else set()
    actual_contrast_keys = set(map(tuple, contrasts[["model", "stratum_type", "stratum", "contrast"]].to_numpy())) if not contrasts.empty else set()
    if actual_margin_keys != expected_margin_keys or margins.duplicated(["model", "stratum_type", "stratum", "direction"]).any():
        raise ValueError("Claves de margenes no reconciliadas.")
    if actual_contrast_keys != expected_contrast_keys:
        raise ValueError("Claves de contrastes no reconciliadas.")
    for family, (column, strata) in STRATUM_COLUMNS.items():
        for stratum in strata:
            observed = analytic[analytic[column].astype(str).eq(stratum)]
            for table in (margins, contrasts):
                rows = table[(table["stratum_type"] == family) & (table["stratum"] == stratum)]
                if not rows.empty and (not rows["analytical_points"].eq(len(observed)).all() or not rows["matches"].eq(observed["match_id"].nunique()).all()):
                    raise ValueError("Cobertura de estrato no reconciliada.")
    if results is not None and all_models_available:
        expected_margins, expected_contrasts = [], []
        for family, model_name in (("surface", "surface_interaction"), ("period", "period_interaction")):
            result = results.get(model_name)
            if result is not None:
                expected_wald = _wald(result, [term for term in build_design(analytic, model_name).columns if ":" in term])
                if global_tests[family] != expected_wald:
                    raise ValueError("Wald global no reconciliado.")
                expected_margin, expected_contrast = margins_and_contrasts(result, analytic, model_name, family)
                expected_margins.append(expected_margin); expected_contrasts.append(expected_contrast)
        expected_margin_table = pd.concat(expected_margins, ignore_index=True) if expected_margins else pd.DataFrame(columns=MARGIN_COLUMNS)
        expected_contrast_table = pd.concat(expected_contrasts, ignore_index=True) if expected_contrasts else pd.DataFrame(columns=CONTRAST_COLUMNS)
        _compare_published_table(margins, expected_margin_table, ["model", "stratum_type", "stratum", "direction"], "margenes")
        _compare_published_table(contrasts, expected_contrast_table, ["model", "stratum_type", "stratum", "contrast"], "contrastes")
    if summary is not None:
        if summary["population"] != population or summary["exclusions"] != {"direction_0": audit["excluded_direction_0"], "without_recognized_direction": audit["excluded_without_recognized_direction"]}: raise ValueError("Resumen no reconciliado.")
        _finite_json(summary)
    return {"common_population": True, "clusters_reconciled": True, "margins_valid": True, "contrasts_valid": True, "summary_contract_reconciled": summary is not None, "published_values_reconciled": results is not None and all_models_available}


def analyze_second_serve_direction_heterogeneity(points: pd.DataFrame):
    analytic, audit = build_analytic_population(points)
    statuses, results, designs = {}, {}, {}
    for name in MODEL_SPECS:
        statuses[name], results[name], designs[name] = fit_model(analytic, name)
    global_tests = {"surface": {"status": "not_available", "reason": "model_not_publishable"}, "period": {"status": "not_available", "reason": "model_not_publishable"}}
    margins, contrasts = [], []
    all_models_available = all(statuses[name]["status"] == "available" for name in MODEL_SPECS)
    if all_models_available:
        surface_terms = [term for term in designs["surface_interaction"].columns if ":" in term]
        global_tests["surface"] = _wald(results["surface_interaction"], surface_terms)
        m, c = margins_and_contrasts(results["surface_interaction"], analytic, "surface_interaction", "surface"); margins.append(m); contrasts.append(c)
        period_terms = [term for term in designs["period_interaction"].columns if ":" in term]
        global_tests["period"] = _wald(results["period_interaction"], period_terms)
        m, c = margins_and_contrasts(results["period_interaction"], analytic, "period_interaction", "period"); margins.append(m); contrasts.append(c)
    margins_table = pd.concat(margins, ignore_index=True) if margins else pd.DataFrame(columns=MARGIN_COLUMNS)
    contrasts_table = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame(columns=CONTRAST_COLUMNS)
    margins_table = margins_table.sort_values(["stratum_type", "stratum", "direction"], key=lambda x: x.map({**{v:i for i,v in enumerate(SURFACE_ORDER)}, **{v:i for i,v in enumerate(PERIOD_ORDER)}, **{v:i for i,v in enumerate(DIRECTION_ORDER)}}).fillna(0) if x.name in {"stratum", "direction"} else x).reset_index(drop=True)
    contrasts_table = contrasts_table.sort_values(["stratum_type", "stratum", "contrast"]).reset_index(drop=True)
    return build_summary(analytic, audit, statuses, global_tests, margins_table, contrasts_table, results), margins_table, contrasts_table, analytic


def write_artifacts(summary: dict[str, Any], margins: pd.DataFrame, contrasts: pd.DataFrame, reports_dir: Path = REPORTS_DIR, tables_dir: Path = TABLES_DIR) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True); tables_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / SUMMARY_PATH.name).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    margins.fillna("").to_csv(tables_dir / MARGINS_PATH.name, index=False, encoding="utf-8", lineterminator="\n")
    contrasts.fillna("").to_csv(tables_dir / CONTRASTS_PATH.name, index=False, encoding="utf-8", lineterminator="\n")


def main() -> None:
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    summary, margins, contrasts, _ = analyze_second_serve_direction_heterogeneity(points)
    write_artifacts(summary, margins, contrasts)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()
