"""Modelo ajustado para la direccion del segundo servicio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.sparse_logistic import build_design, cluster_covariance, fit_mle, inference, wald_test

from src.analysis.second_serve_direction_baseline import (
    DIRECTION_ORDER,
    POINTS_FILE,
    REPORTS_DIR,
    SOURCE_COLUMNS,
    TABLES_DIR,
    analyze_second_serve_direction_baseline,
)


ANALYSIS_NAME = "second_serve_direction_adjusted"
ANALYSIS_VERSION = 1
REFERENCE_DIRECTION = "wide"
MODEL_FORMULA = (
    'server_won_point ~ C(direction, Treatment(reference="wide")) '
    '+ C(surface, Treatment(reference="Hard")) '
    '+ C(derived_period, Treatment(reference="to_2009")) '
    '+ C(server_player, Treatment(reference="{server_reference}"))'
)
STANDARD_ERROR_METHOD = "cluster_robust_by_match_id"
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
BYTES_PER_DENSE_ELEMENT = 8
INITIAL_MAX_ITERATIONS = 200
CONTINUATION_MAX_ITERATIONS = 800
SUMMARY_PATH = REPORTS_DIR / "second_serve_direction_adjusted_summary.json"
COEFFICIENTS_PATH = TABLES_DIR / "second_serve_direction_adjusted_coefficients.csv"

COEFFICIENT_COLUMNS = [
    "term",
    "variable",
    "level",
    "reference_level",
    "coefficient",
    "odds_ratio",
    "std_error",
    "statistic",
    "p_value",
    "ci_low",
    "ci_high",
]


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
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")
        return


def _coefficient_row(term_metadata: dict[str, Any], coefficient: float, std_error: float, statistic: float,
                     p_value: float, ci_low: float, ci_high: float) -> dict[str, Any]:
    required = {"term", "variable", "level", "reference_level", "column_index"}
    if not required.issubset(term_metadata):
        raise ValueError("Metadatos canonicos de termino incompletos.")
    term = term_metadata["term"]
    variable = term_metadata["variable"]
    level = term_metadata["level"]
    reference = term_metadata["reference_level"]
    if variable not in {"intercept", "direction", "surface", "derived_period", "server_player"}:
        raise ValueError(f"Variable de termino desconocida: {variable}")
    return {
        "term": term,
        "variable": variable,
        "level": level,
        "reference_level": "" if reference is None else reference,
        "coefficient": float(coefficient),
        "odds_ratio": float(np.exp(coefficient)),
        "std_error": float(std_error),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "ci_low": float(np.exp(ci_low)),
        "ci_high": float(np.exp(ci_high)),
    }


def _attempt_summary(attempt: int, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Diagnostico serializable, sin coeficientes, de un intento BFGS."""
    keys = (
        "initial_point", "used_warm_start", "optimizer", "tolerance", "max_iterations",
        "converged", "optimizer_success", "optimizer_status", "optimizer_message",
        "iterations", "objective_value", "log_likelihood", "gradient_norm",
        "max_abs_coefficient", "minimum_probability", "maximum_probability",
        "saturated_probability_count", "hessian_status", "covariance_status",
        "finite_coefficients", "finite_probabilities", "publishable", "reason", "reason_codes",
    )
    return {"attempt": attempt, **{key: diagnostics[key] for key in keys}}


def fit_adjusted_model(analytic: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    model_data = analytic.copy()
    server_reference = sorted(model_data["server_player"].unique())[0]
    formula = MODEL_FORMULA.format(server_reference=server_reference)
    # Intercept + two non-reference levels for direction, surface and period,
    # plus one indicator for each non-reference server.
    expected_columns = 1 + 2 + 2 + 2 + (model_data["server_player"].nunique() - 1)
    expected_rows = int(len(model_data))
    estimated_dense_bytes = expected_rows * expected_columns * BYTES_PER_DENSE_ELEMENT
    required_categories = {
        "direction": ("wide", "body", "T"),
        "surface": ("Hard", "Clay", "Grass"),
        "derived_period": ("to_2009", "2010s", "2020s"),
        "server_player": tuple(sorted(model_data["server_player"].unique())),
    }
    references = {"direction": "wide", "surface": "Hard", "derived_period": "to_2009", "server_player": server_reference}
    design, term_mapping = build_design(model_data, required_categories, references)
    csr_bytes = int(design.data.nbytes + design.indices.nbytes + design.indptr.nbytes)
    hessian_bytes = int(design.shape[1] ** 2 * BYTES_PER_DENSE_ELEMENT)
    peak_bytes = csr_bytes + 3 * hessian_bytes + int(model_data["match_id"].nunique()) * design.shape[1] * BYTES_PER_DENSE_ELEMENT
    if peak_bytes > MEMORY_LIMIT_BYTES:
        return (
            {
                "model_status": "not_available",
                "reason": "estimated_sparse_peak_exceeds_memory_limit",
                "formula": formula,
                "reference_direction": REFERENCE_DIRECTION,
                "server_reference": server_reference,
                "estimator": "sparse logistic MLE (planned, not fitted)",
                "standard_error_method": STANDARD_ERROR_METHOD,
                "n_observations": expected_rows,
                "planned_model_rows": expected_rows,
                "planned_model_columns": int(expected_columns),
                "bytes_per_dense_element": BYTES_PER_DENSE_ELEMENT,
                "estimated_dense_design_bytes": int(estimated_dense_bytes), "sparse_shape": list(design.shape), "sparse_nnz": int(design.nnz), "sparse_bytes": csr_bytes, "hessian_bytes": hessian_bytes, "estimated_peak_bytes": peak_bytes, "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            },
            pd.DataFrame(columns=COEFFICIENT_COLUMNS),
        )
    try:
        outcome = model_data["server_won_point"].astype(int).to_numpy()
        first_fit = fit_mle(design, outcome, max_iterations=INITIAL_MAX_ITERATIONS)
        second_fit = fit_mle(
            design,
            outcome,
            max_iterations=CONTINUATION_MAX_ITERATIONS,
            initial_coefficients=first_fit["beta"],
        )
        fit = second_fit
        optimization_attempts = [
            _attempt_summary(1, first_fit["diagnostics"]),
            _attempt_summary(2, second_fit["diagnostics"]),
        ]
        total_iterations = sum(
            attempt["iterations"] for attempt in optimization_attempts
            if attempt["iterations"] is not None
        )
        if total_iterations > INITIAL_MAX_ITERATIONS + CONTINUATION_MAX_ITERATIONS:
            raise ValueError("Se supero el presupuesto total de iteraciones.")
        if not fit["publishable"]:
            return ({"model_status":"not_available","reason":"sparse_fit_not_publishable","fit_diagnostics":fit["diagnostics"],"optimization_attempts":optimization_attempts,"total_iterations":total_iterations,"cluster_covariance_status":"not_calculated_after_rejection","wald_status":"not_calculated_after_rejection","inference_status":"not_calculated_after_rejection","formula":formula,"reference_direction":REFERENCE_DIRECTION,"server_reference":server_reference,"estimator":"sparse unpenalized logistic MLE","standard_error_method":STANDARD_ERROR_METHOD,"n_observations":expected_rows,"sparse_shape":list(design.shape),"sparse_nnz":int(design.nnz),"sparse_bytes":csr_bytes,"hessian_bytes":hessian_bytes,"estimated_peak_bytes":peak_bytes,"cluster_count":int(model_data["match_id"].nunique())}, pd.DataFrame(columns=COEFFICIENT_COLUMNS))
        robust = cluster_covariance(design, outcome, fit["beta"], model_data["match_id"].to_numpy())
        details = inference(fit["beta"], robust)
        rows = [_coefficient_row(metadata, fit["beta"][metadata["column_index"]], details["std_error"][metadata["column_index"]], details["statistic"][metadata["column_index"]], details["p_value"][metadata["column_index"]], details["ci_low"][metadata["column_index"]], details["ci_high"][metadata["column_index"]]) for metadata in term_mapping.metadata]
        coefficients = pd.DataFrame(rows, columns=COEFFICIENT_COLUMNS)
        global_test = wald_test(fit["beta"], robust, (term_mapping["direction[body]"], term_mapping["direction[T]"]))
        return ({"model_status":"available","formula":formula,"reference_direction":REFERENCE_DIRECTION,"server_reference":server_reference,"estimator":"sparse unpenalized logistic MLE","standard_error_method":STANDARD_ERROR_METHOD,"n_observations":expected_rows,"iterations":fit["iterations"],"total_iterations":total_iterations,"optimization_attempts":optimization_attempts,"gradient_norm":fit["gradient_norm"],"log_likelihood":fit["log_likelihood"],"sparse_shape":list(design.shape),"sparse_nnz":int(design.nnz),"sparse_bytes":csr_bytes,"hessian_bytes":hessian_bytes,"estimated_peak_bytes":peak_bytes,"cluster_count":int(model_data["match_id"].nunique()),"global_direction_test":{"status":"available",**global_test}}, coefficients)
    except Exception as exc:
        return ({"model_status":"not_available","reason":f"sparse_fit_failed: {type(exc).__name__}: {exc}","formula":formula,"reference_direction":REFERENCE_DIRECTION,"server_reference":server_reference,"estimator":"sparse unpenalized logistic MLE","standard_error_method":STANDARD_ERROR_METHOD,"n_observations":expected_rows,"sparse_shape":list(design.shape),"sparse_nnz":int(design.nnz),"sparse_bytes":csr_bytes,"hessian_bytes":hessian_bytes,"estimated_peak_bytes":peak_bytes,"cluster_count":int(model_data["match_id"].nunique())}, pd.DataFrame(columns=COEFFICIENT_COLUMNS))


def build_summary(
    baseline_summary: dict[str, Any],
    analytic: pd.DataFrame,
    adjusted: dict[str, Any],
    coefficients: pd.DataFrame,
) -> dict[str, Any]:
    direction_counts = analytic["direction"].astype(str).value_counts()
    summary = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "source": "data/processed/points_enriched.parquet",
        "source_rows": int(baseline_summary["population"]["audit"]["source_rows"]),
        "substantive_second_serve_points": int(
            baseline_summary["population"]["audit"]["substantive_second_serve_rows"]
        ),
        "analytical_points": int(len(analytic)),
        "matches": int(analytic["match_id"].nunique()),
        "clusters": int(analytic["match_id"].nunique()),
        "servers": int(analytic["server_player"].nunique()),
        "server_wins": int(analytic["server_won_point"].sum()),
        "excluded_direction_zero": int(
            baseline_summary["population"]["audit"]["excluded_direction_0"]
        ),
        "excluded_without_recognized_direction": int(
            baseline_summary["population"]["audit"]["excluded_without_recognized_direction"]
        ),
        "reference_direction": REFERENCE_DIRECTION,
        "model_formula": adjusted["formula"],
        "estimator": adjusted["estimator"],
        "standard_error_method": adjusted["standard_error_method"],
        "model_status": adjusted["model_status"],
        "global_direction_test": adjusted.get("global_direction_test")
        or {"status": "not_available", "reason": adjusted["reason"]},
        "coefficients_contract": {
            "csv_columns": COEFFICIENT_COLUMNS,
            "direction_terms": ["body vs wide", "T vs wide"],
            "odds_ratio": "exp(coefficient)",
            "confidence_interval": "95% odds-ratio interval",
            "coefficient_rows": int(len(coefficients)),
        },
        "population_by_direction": {
            direction: int(direction_counts.get(direction, 0))
            for direction in DIRECTION_ORDER
        },
        "categorical_inputs": {
            "direction": {
                "categories": list(DIRECTION_ORDER),
                "reference": REFERENCE_DIRECTION,
            },
            "surface": {
                "categories": sorted(analytic["surface"].unique().tolist()),
                "reference": "Hard",
            },
            "derived_period": {
                "categories": sorted(analytic["derived_period"].unique().tolist()),
                "reference": "to_2009",
            },
            "server_player": {
                "categories": int(analytic["server_player"].nunique()),
                "reference": adjusted.get("server_reference"),
                "servers_with_fewer_than_10_points": int(
                    (analytic.groupby("server_player", observed=True).size() < 10).sum()
                ),
            },
        },
        "methodological_limits": [
            "Analisis asociativo, no causal.",
            "No produce recomendaciones tacticas.",
            "No usa rally length, resto, golpes, train/test split ni features historicas.",
            "Los errores estandar se agruparan por match_id solo cuando el modelo este disponible.",
        ],
    }
    for metric in ("n_observations", "planned_model_rows", "planned_model_columns", "bytes_per_dense_element", "estimated_dense_design_bytes", "memory_limit_bytes", "log_likelihood"):
        if metric in adjusted and adjusted[metric] is not None:
            summary[metric] = adjusted[metric]
    if "fit_diagnostics" in adjusted:
        summary["fit_diagnostics"] = adjusted["fit_diagnostics"]
    for metric in (
        "optimization_attempts", "total_iterations", "cluster_covariance_status",
        "wald_status", "inference_status",
    ):
        if metric in adjusted:
            summary[metric] = adjusted[metric]
    summary["reconciliations"] = validate_reconciliations(
        baseline_summary,
        analytic,
        adjusted,
        coefficients,
        summary,
    )
    _assert_json_finite(summary)
    return summary


def validate_reconciliations(
    baseline_summary: dict[str, Any],
    analytic: pd.DataFrame,
    adjusted: dict[str, Any],
    coefficients: pd.DataFrame,
    summary: dict[str, Any] | None = None,
) -> dict[str, bool]:
    expected_points = int(baseline_summary["population"]["analytic_points"])
    if len(analytic) != expected_points:
        raise ValueError("La poblacion ajustada no reconcilia con el baseline.")
    if not analytic["server_won_point"].isin([True, False]).all():
        raise ValueError("server_won_point debe ser binaria.")
    if set(analytic["direction"].astype(str)) - set(DIRECTION_ORDER):
        raise ValueError("La direccion ajustada no pertenece al catalogo.")
    if coefficients.isna().any().any():
        raise ValueError("Los coeficientes contienen NaN.")
    numeric = coefficients.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Los coeficientes contienen infinitos.")
    if adjusted["model_status"] == "available":
        if int(adjusted["n_observations"]) != len(analytic):
            raise ValueError("n_observations no reconcilia.")
        direction_terms = coefficients[coefficients["variable"].eq("direction")]
        if set(direction_terms["level"]) != {"body", "T"}:
            raise ValueError("Faltan coeficientes de direccion.")
        if not np.allclose(direction_terms["odds_ratio"], np.exp(direction_terms["coefficient"])):
            raise ValueError("odds_ratio no reconcilia con coefficient.")
        if not direction_terms["p_value"].between(0, 1).all():
            raise ValueError("p_value fuera de rango.")
    else:
        if len(coefficients) != 0:
            raise ValueError("No debe haber coeficientes si el modelo no esta disponible.")
        if adjusted.get("reason") == "sparse_fit_not_publishable":
            diagnostics = adjusted.get("fit_diagnostics")
            required_diagnostics = {
                "reason_codes", "reason", "converged", "optimizer_success", "optimizer_status",
                "optimizer_message", "iterations", "objective_value", "log_likelihood",
                "gradient_norm", "max_abs_coefficient", "minimum_probability",
                "maximum_probability", "saturated_probability_count", "hessian_status",
                "covariance_status", "finite_coefficients", "finite_probabilities", "publishable",
                "initial_point", "used_warm_start", "optimizer", "tolerance", "max_iterations",
            }
            if not isinstance(diagnostics, dict) or not required_diagnostics.issubset(diagnostics):
                raise ValueError("fit_diagnostics incompleto.")
            if diagnostics["converged"] != diagnostics["optimizer_success"] or diagnostics["publishable"]:
                raise ValueError("fit_diagnostics no reconcilia.")
            attempts = adjusted.get("optimization_attempts")
            if not isinstance(attempts, list) or len(attempts) != 2:
                raise ValueError("optimization_attempts incompleto.")
            if attempts[0]["initial_point"] != "zeros" or attempts[1]["initial_point"] != "previous_solution":
                raise ValueError("optimization_attempts no conserva el warm start.")
            if attempts[0]["max_iterations"] != INITIAL_MAX_ITERATIONS or attempts[1]["max_iterations"] != CONTINUATION_MAX_ITERATIONS:
                raise ValueError("optimization_attempts no conserva el presupuesto.")
            if attempts[0]["optimizer"] != attempts[1]["optimizer"] or attempts[0]["tolerance"] != attempts[1]["tolerance"]:
                raise ValueError("optimization_attempts no conserva el contrato del optimizador.")
            if adjusted.get("total_iterations", -1) > INITIAL_MAX_ITERATIONS + CONTINUATION_MAX_ITERATIONS:
                raise ValueError("optimization_attempts supera el presupuesto total.")
            if adjusted.get("cluster_covariance_status") != "not_calculated_after_rejection" or adjusted.get("wald_status") != "not_calculated_after_rejection" or adjusted.get("inference_status") != "not_calculated_after_rejection":
                raise ValueError("El rechazo no conserva el estado de inferencia.")
    if summary is not None:
        if summary["analytical_points"] != len(analytic):
            raise ValueError("summary analytical_points no reconcilia.")
        if summary["matches"] != analytic["match_id"].nunique() or summary["clusters"] != analytic["match_id"].nunique():
            raise ValueError("summary matches/clusters no reconcilia.")
        if summary["servers"] != analytic["server_player"].nunique() or summary["server_wins"] != int(analytic["server_won_point"].sum()):
            raise ValueError("summary servers/server_wins no reconcilia.")
        if summary["reference_direction"] != REFERENCE_DIRECTION:
            raise ValueError("summary reference_direction no reconcilia.")
        if summary["model_formula"] != adjusted["formula"]:
            raise ValueError("summary model_formula no reconcilia.")
        if summary["model_status"] != adjusted["model_status"]:
            raise ValueError("summary model_status no reconcilia.")
        if summary["coefficients_contract"]["coefficient_rows"] != len(coefficients):
            raise ValueError("summary coefficients_contract no reconcilia.")
        expected_global_test = adjusted.get("global_direction_test") or {
            "status": "not_available", "reason": adjusted["reason"]
        }
        if summary["global_direction_test"] != expected_global_test:
            raise ValueError("summary global_direction_test no reconcilia.")
        if adjusted.get("fit_diagnostics") is not None and summary.get("fit_diagnostics") != adjusted["fit_diagnostics"]:
            raise ValueError("summary fit_diagnostics no reconcilia.")
        if adjusted.get("optimization_attempts") is not None and summary.get("optimization_attempts") != adjusted["optimization_attempts"]:
            raise ValueError("summary optimization_attempts no reconcilia.")
    return {
        "population_matches_baseline": True,
        "binary_outcome": True,
        "known_directions_only": True,
        "coefficient_values_finite": True,
        "summary_contract_reconciled": summary is not None,
    }


def analyze_second_serve_direction_adjusted(points: pd.DataFrame):
    baseline_summary, _, _, analytic = analyze_second_serve_direction_baseline(points)
    adjusted, coefficients = fit_adjusted_model(analytic)
    summary = build_summary(baseline_summary, analytic, adjusted, coefficients)
    return summary, coefficients, analytic


def write_artifacts(
    summary: dict[str, Any],
    coefficients: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "second_serve_direction_adjusted_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coefficients.to_csv(
        tables_dir / "second_serve_direction_adjusted_coefficients.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def main() -> None:
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    summary, coefficients, _ = analyze_second_serve_direction_adjusted(points)
    write_artifacts(summary, coefficients)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
