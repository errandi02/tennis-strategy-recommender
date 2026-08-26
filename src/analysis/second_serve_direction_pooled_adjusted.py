"""GLM pooled asociativo para la direccion del segundo saque."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.analysis.second_serve_direction_analysis import (
    PERIOD_ORDER,
    POINTS_FILE,
    REPORTS_DIR,
    SOURCE_COLUMNS,
    SURFACE_ORDER,
    TABLES_DIR,
)
from src.analysis.second_serve_direction_baseline import (
    DIRECTION_ORDER,
    build_analytic_population,
)


ANALYSIS_NAME = "second_serve_direction_pooled_adjusted"
ANALYSIS_VERSION = 1
MODEL_FORMULA = "server_won_point ~ direction + surface + derived_period"
DESIGN_COLUMNS = (
    "Intercept",
    "direction[body]",
    "direction[T]",
    "surface[Clay]",
    "surface[Grass]",
    "derived_period[2010s]",
    "derived_period[2020s]",
)
REFERENCES = {
    "direction": "wide",
    "surface": "Hard",
    "derived_period": "to_2009",
}
CLUSTER_VARIABLE = "match_id"
STANDARD_ERROR_METHOD = "cluster_robust_by_match_id"
COEFFICIENT_ABSOLUTE_LIMIT = 30.0
SATURATION_TOLERANCE = 1e-12
SUMMARY_PATH = REPORTS_DIR / "second_serve_direction_pooled_adjusted_summary.json"
COEFFICIENTS_PATH = TABLES_DIR / "second_serve_direction_pooled_adjusted_coefficients.csv"

COEFFICIENT_COLUMNS = [
    "term", "variable", "level", "reference_level", "coefficient", "odds_ratio",
    "std_error", "statistic", "p_value", "ci_low", "ci_high",
]
REQUIRED_ANALYTIC_COLUMNS = frozenset({
    "match_id", "point_number", "server_player", "surface", "derived_period",
    "direction", "server_won_point",
})


def _assert_json_finite(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Valor no finito en {path}.")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")


def _empty_coefficients() -> pd.DataFrame:
    return pd.DataFrame(columns=COEFFICIENT_COLUMNS)


def validate_analytic_population(analytic: pd.DataFrame) -> None:
    missing = REQUIRED_ANALYTIC_COLUMNS - set(analytic.columns)
    if missing:
        raise ValueError(f"Faltan columnas analiticas: {sorted(missing)}")
    if analytic.empty or analytic.duplicated(["match_id", "point_number"]).any():
        raise ValueError("La poblacion analitica debe ser no vacia y tener clave unica.")
    if not analytic["server_won_point"].isin([True, False, 0, 1]).all():
        raise ValueError("server_won_point debe ser binario.")
    for column, categories in (
        ("direction", DIRECTION_ORDER),
        ("surface", SURFACE_ORDER),
        ("derived_period", PERIOD_ORDER),
    ):
        observed = set(analytic[column].astype(str))
        if observed != set(categories):
            raise ValueError(f"Categorias invalidas o ausentes en {column}: {sorted(observed)}")
    if analytic["match_id"].isna().any() or analytic["server_player"].isna().any():
        raise ValueError("match_id y server_player no pueden ser nulos.")
    if analytic["match_id"].nunique() < 2:
        raise ValueError("Se requieren al menos dos clusters.")
    if analytic["server_won_point"].nunique() != 2:
        raise ValueError("El outcome no puede ser constante.")


def build_design(analytic: pd.DataFrame) -> pd.DataFrame:
    """Construye las siete columnas deterministas del GLM pooled."""
    validate_analytic_population(analytic)
    design = pd.DataFrame({
        "Intercept": np.ones(len(analytic), dtype=float),
        "direction[body]": analytic["direction"].astype(str).eq("body").astype(float),
        "direction[T]": analytic["direction"].astype(str).eq("T").astype(float),
        "surface[Clay]": analytic["surface"].astype(str).eq("Clay").astype(float),
        "surface[Grass]": analytic["surface"].astype(str).eq("Grass").astype(float),
        "derived_period[2010s]": analytic["derived_period"].astype(str).eq("2010s").astype(float),
        "derived_period[2020s]": analytic["derived_period"].astype(str).eq("2020s").astype(float),
    }, index=analytic.index)
    if design.columns.tolist() != list(DESIGN_COLUMNS):
        raise ValueError("El orden del diseno no reconcilia.")
    if np.linalg.matrix_rank(design.to_numpy()) != len(DESIGN_COLUMNS):
        raise ValueError("El diseno no tiene rango completo.")
    return design


def _fit_diagnostics(result: Any, design: pd.DataFrame, outcome: np.ndarray, clusters: pd.Series) -> dict[str, Any]:
    parameters = np.asarray(result.params, dtype=float)
    probabilities = np.asarray(result.predict(design), dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    reasons: list[str] = []
    if getattr(result, "cov_type", None) != "cluster":
        reasons.append("covariance_not_cluster")
    cov_kwds = getattr(result, "cov_kwds", {})
    if not cov_kwds.get("use_correction", False):
        reasons.append("cluster_finite_sample_correction_not_registered")
    if not bool(getattr(result, "converged", False)):
        reasons.append("not_converged")
    if not np.isfinite(parameters).all():
        reasons.append("non_finite_parameters")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        reasons.append("invalid_probabilities")
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12) or np.any(np.diag(covariance) < 0):
        reasons.append("invalid_covariance")
    if np.max(np.abs(parameters)) > COEFFICIENT_ABSOLUTE_LIMIT:
        reasons.append("extreme_coefficients")
    saturated = int(np.count_nonzero((probabilities <= SATURATION_TOLERANCE) | (probabilities >= 1 - SATURATION_TOLERANCE)))
    if saturated:
        reasons.append("saturated_probabilities")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        reasons.append("rank_deficient_design")
    if len(np.unique(outcome)) != 2:
        reasons.append("constant_outcome")
    if clusters.nunique() < 2:
        reasons.append("insufficient_clusters")
    return {
        "reason_codes": reasons,
        "converged": bool(getattr(result, "converged", False)),
        "cov_type": getattr(result, "cov_type", None),
        "cluster_finite_sample_correction": bool(cov_kwds.get("use_correction", False)),
        "parameters_finite": bool(np.isfinite(parameters).all()),
        "probabilities_finite": bool(np.isfinite(probabilities).all()),
        "minimum_probability": float(np.min(probabilities)),
        "maximum_probability": float(np.max(probabilities)),
        "saturated_probability_count": saturated,
        "covariance_finite": bool(np.isfinite(covariance).all()),
        "covariance_symmetric": bool(np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)),
        "covariance_nonnegative_diagonal": bool(np.all(np.diag(covariance) >= 0)),
        "design_rank": int(np.linalg.matrix_rank(design.to_numpy())),
        "design_columns": int(design.shape[1]),
        "max_abs_coefficient": float(np.max(np.abs(parameters))),
        "clusters": int(clusters.nunique()),
        "publishable": not reasons,
    }


def _term_metadata(term: str) -> tuple[str, str, str]:
    metadata = {
        "Intercept": ("intercept", "", ""),
        "direction[body]": ("direction", "body", "wide"),
        "direction[T]": ("direction", "T", "wide"),
        "surface[Clay]": ("surface", "Clay", "Hard"),
        "surface[Grass]": ("surface", "Grass", "Hard"),
        "derived_period[2010s]": ("derived_period", "2010s", "to_2009"),
        "derived_period[2020s]": ("derived_period", "2020s", "to_2009"),
    }
    if term not in metadata:
        raise ValueError(f"Termino no reconocido: {term}")
    return metadata[term]


def _coefficients(result: Any) -> pd.DataFrame:
    confidence = np.asarray(result.conf_int(alpha=0.05), dtype=float)
    rows = []
    for index, term in enumerate(DESIGN_COLUMNS):
        variable, level, reference = _term_metadata(term)
        coefficient = float(result.params[term])
        rows.append({
            "term": term, "variable": variable, "level": level,
            "reference_level": reference, "coefficient": coefficient,
            "odds_ratio": float(np.exp(coefficient)), "std_error": float(result.bse[term]),
            "statistic": float(result.tvalues[term]), "p_value": float(result.pvalues[term]),
            "ci_low": float(np.exp(confidence[index, 0])),
            "ci_high": float(np.exp(confidence[index, 1])),
        })
    return pd.DataFrame(rows, columns=COEFFICIENT_COLUMNS)


def _global_direction_test(result: Any) -> dict[str, Any]:
    covariance = np.asarray(result.cov_params(), dtype=float)
    parameters = np.asarray(result.params, dtype=float)
    indices = (DESIGN_COLUMNS.index("direction[body]"), DESIGN_COLUMNS.index("direction[T]"))
    selected = parameters[list(indices)]
    selected_covariance = covariance[np.ix_(indices, indices)]
    if np.linalg.matrix_rank(selected_covariance) != 2:
        raise ValueError("La covarianza del contraste Wald no es invertible.")
    statistic = float(selected @ np.linalg.solve(selected_covariance, selected))
    return {
        "status": "available", "hypothesis": "body_vs_wide = 0 and T_vs_wide = 0",
        "statistic": statistic, "df": 2, "p_value": float(stats.chi2.sf(statistic, 2)),
    }


def fit_pooled_model(analytic: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    design = build_design(analytic)
    outcome = analytic["server_won_point"].astype(int).to_numpy()
    clusters = analytic[CLUSTER_VARIABLE]
    try:
        result = sm.GLM(outcome, design, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": clusters.to_numpy()},
        )
        diagnostics = _fit_diagnostics(result, design, outcome, clusters)
        if not diagnostics["publishable"]:
            return {
                "model_status": "not_available", "reason_codes": diagnostics["reason_codes"],
                "model_diagnostics": diagnostics,
            }, _empty_coefficients()
        coefficients = _coefficients(result)
        global_test = _global_direction_test(result)
    except Exception as exc:
        return {
            "model_status": "not_available", "reason_codes": [f"fit_failed_{type(exc).__name__}"],
            "model_diagnostics": {"exception_type": type(exc).__name__, "publishable": False},
        }, _empty_coefficients()
    return {
        "model_status": "available", "reason_codes": [], "model_diagnostics": diagnostics,
        "n_observations": int(result.nobs), "log_likelihood": float(result.llf),
        "deviance": float(result.deviance), "null_deviance": float(result.null_deviance),
        "pseudo_r_squared": float(1 - result.llf / result.llnull),
        "pseudo_r_squared_definition": "McFadden: 1 - log_likelihood / null_log_likelihood",
        "aic": float(result.aic), "global_direction_test": global_test,
    }, coefficients


def _population(analytic: pd.DataFrame) -> dict[str, Any]:
    direction = analytic.groupby("direction", observed=True)["server_won_point"].agg(["size", "sum"])
    return {
        "analytical_points": int(len(analytic)), "matches": int(analytic["match_id"].nunique()),
        "servers": int(analytic["server_player"].nunique()),
        "server_wins": int(analytic["server_won_point"].sum()),
        "by_direction": {
            value: {"points": int(direction.loc[value, "size"]), "server_wins": int(direction.loc[value, "sum"])}
            for value in DIRECTION_ORDER
        },
    }


def validate_reconciliations(analytic: pd.DataFrame, audit: dict[str, Any], adjusted: dict[str, Any], coefficients: pd.DataFrame, summary: dict[str, Any] | None = None) -> dict[str, bool]:
    population = _population(analytic)
    if population["analytical_points"] != 481_190 and len(analytic) > 1_000:
        raise ValueError("La poblacion real no reconcilia el contrato aprobado.")
    if adjusted["model_status"] == "available":
        diagnostics = adjusted["model_diagnostics"]
        if not diagnostics.get("publishable") or diagnostics.get("cov_type") != "cluster" or not diagnostics.get("cluster_finite_sample_correction"):
            raise ValueError("La inferencia publicada no usa covarianza cluster valida.")
        if diagnostics.get("clusters") != population["matches"]:
            raise ValueError("Los clusters del diagnostico no reconcilian.")
        if len(coefficients) != len(DESIGN_COLUMNS) or coefficients["term"].tolist() != list(DESIGN_COLUMNS):
            raise ValueError("El contrato de coeficientes no reconcilia.")
        if not np.allclose(coefficients["odds_ratio"], np.exp(coefficients["coefficient"])):
            raise ValueError("OR no reconcilia con coeficientes.")
        if not coefficients["p_value"].between(0, 1).all() or not coefficients["ci_low"].gt(0).all() or not coefficients["ci_high"].ge(coefficients["ci_low"]).all():
            raise ValueError("p-values o intervalos fuera de rango.")
        global_test = adjusted["global_direction_test"]
        if global_test["status"] != "available":
            raise ValueError("Falta contraste Wald publicado.")
        if global_test.get("df") != 2 or not np.isfinite(global_test.get("statistic", np.nan)) or not 0 <= global_test.get("p_value", np.nan) <= 1:
            raise ValueError("Contraste Wald no valido.")
    elif not coefficients.empty:
        raise ValueError("No se deben publicar coeficientes si el modelo no esta disponible.")
    if summary is not None:
        if summary["population"] != population or summary["exclusions"] != {
            "direction_0": audit["excluded_direction_0"],
            "without_recognized_direction": audit["excluded_without_recognized_direction"],
        }:
            raise ValueError("La poblacion o exclusiones del resumen no reconcilian.")
        if summary["coefficients_contract"]["coefficient_rows"] != len(coefficients):
            raise ValueError("El numero de coeficientes no reconcilia.")
        expected_contract = {
            "csv_columns": COEFFICIENT_COLUMNS,
            "coefficient_rows": int(len(coefficients)),
            "odds_ratio": "exp(coefficient)",
            "confidence_interval": "95% odds-ratio interval",
        }
        if summary["coefficients_contract"] != expected_contract:
            raise ValueError("El contrato CSV del resumen no reconcilia.")
        expected_global_test = adjusted.get("global_direction_test", {"status": "not_available", "reason": "model_not_publishable"})
        expected_summary_fields = {
            "analysis_name": ANALYSIS_NAME,
            "analysis_version": ANALYSIS_VERSION,
            "source": "data/processed/points_enriched.parquet",
            "model_status": adjusted["model_status"],
            "reason_codes": adjusted["reason_codes"],
            "formula": MODEL_FORMULA,
            "estimator": "statsmodels.GLM Binomial",
            "design_columns": list(DESIGN_COLUMNS),
            "categorical_references": REFERENCES,
            "standard_error_method": STANDARD_ERROR_METHOD,
            "cluster_variable": CLUSTER_VARIABLE,
            "clusters": population["matches"],
            "convergence": adjusted["model_diagnostics"].get("converged"),
            "model_diagnostics": adjusted["model_diagnostics"],
            "global_direction_test": expected_global_test,
        }
        if any(summary.get(key) != value for key, value in expected_summary_fields.items()):
            raise ValueError("Campos contractuales del resumen no reconcilian.")
        if summary["global_direction_test"] != adjusted.get("global_direction_test", {"status": "not_available", "reason": "model_not_publishable"}):
            raise ValueError("El contraste Wald del resumen no reconcilia.")
        for key in ("n_observations", "log_likelihood", "deviance", "null_deviance", "pseudo_r_squared", "pseudo_r_squared_definition", "aic"):
            if key in adjusted and summary.get(key) != adjusted[key]:
                raise ValueError(f"La metrica {key} del resumen no reconcilia.")
        _assert_json_finite(summary)
    return {
        "population_reconciled": True, "wins_reconciled": True,
        "coefficients_contract_reconciled": True, "summary_contract_reconciled": summary is not None,
    }


def build_summary(analytic: pd.DataFrame, audit: dict[str, Any], adjusted: dict[str, Any], coefficients: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "source": "data/processed/points_enriched.parquet", "population": _population(analytic),
        "exclusions": {"direction_0": audit["excluded_direction_0"], "without_recognized_direction": audit["excluded_without_recognized_direction"]},
        "model_status": adjusted["model_status"], "reason_codes": adjusted["reason_codes"],
        "formula": MODEL_FORMULA, "estimator": "statsmodels.GLM Binomial",
        "design_columns": list(DESIGN_COLUMNS), "categorical_references": REFERENCES,
        "standard_error_method": STANDARD_ERROR_METHOD, "cluster_variable": CLUSTER_VARIABLE,
        "clusters": int(analytic["match_id"].nunique()),
        "convergence": adjusted["model_diagnostics"].get("converged"),
        "model_diagnostics": adjusted["model_diagnostics"],
        "global_direction_test": adjusted.get("global_direction_test", {"status": "not_available", "reason": "model_not_publishable"}),
        "coefficients_contract": {"csv_columns": COEFFICIENT_COLUMNS, "coefficient_rows": int(len(coefficients)), "odds_ratio": "exp(coefficient)", "confidence_interval": "95% odds-ratio interval"},
        "methodological_limits": [
            "Asociacion poblacional ajustada solo por superficie y periodo.",
            "Puede existir confusion residual por diferencias entre servidores.",
            "Los errores estandar se agrupan por match_id; la dependencia entre partidos de un mismo servidor no se modela explicitamente.",
            "Analisis no causal y sin recomendacion tactica.",
        ],
    }
    for key in ("n_observations", "log_likelihood", "deviance", "null_deviance", "pseudo_r_squared", "pseudo_r_squared_definition", "aic"):
        if key in adjusted:
            summary[key] = adjusted[key]
    summary["reconciliations"] = validate_reconciliations(analytic, audit, adjusted, coefficients, summary)
    _assert_json_finite(summary)
    return summary


def analyze_second_serve_direction_pooled_adjusted(points: pd.DataFrame):
    analytic, audit = build_analytic_population(points)
    adjusted, coefficients = fit_pooled_model(analytic)
    return build_summary(analytic, audit, adjusted, coefficients), coefficients, analytic


def write_artifacts(summary: dict[str, Any], coefficients: pd.DataFrame, reports_dir: Path = REPORTS_DIR, tables_dir: Path = TABLES_DIR) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / SUMMARY_PATH.name).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coefficients.to_csv(tables_dir / COEFFICIENTS_PATH.name, index=False, encoding="utf-8", lineterminator="\n")


def main() -> None:
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    summary, coefficients, _ = analyze_second_serve_direction_pooled_adjusted(points)
    write_artifacts(summary, coefficients)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
