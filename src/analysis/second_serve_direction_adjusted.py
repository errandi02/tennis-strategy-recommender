"""Modelo ajustado para la direccion del segundo servicio."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _statsmodels_available() -> bool:
    return importlib.util.find_spec("statsmodels") is not None


def _coefficient_row(term: str, coefficient: float, std_error: float, statistic: float,
                     p_value: float, ci_low: float, ci_high: float) -> dict[str, Any]:
    variable = "other"
    level = term
    reference = ""
    if "direction" in term and "[T.body]" in term:
        variable, level, reference = "direction", "body", REFERENCE_DIRECTION
    elif "direction" in term and "[T.T]" in term:
        variable, level, reference = "direction", "T", REFERENCE_DIRECTION
    elif "surface" in term:
        variable, level, reference = "surface", term.rsplit("[T.", 1)[-1].rstrip("]"), ""
    elif "derived_period" in term:
        variable, level, reference = "derived_period", term.rsplit("[T.", 1)[-1].rstrip("]"), ""
    elif "server_player" in term:
        variable, level, reference = "server_player", term.rsplit("[T.", 1)[-1].rstrip("]"), ""
    return {
        "term": term,
        "variable": variable,
        "level": level,
        "reference_level": reference,
        "coefficient": float(coefficient),
        "odds_ratio": float(np.exp(coefficient)),
        "std_error": float(std_error),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "ci_low": float(np.exp(ci_low)),
        "ci_high": float(np.exp(ci_high)),
    }


def fit_adjusted_model(analytic: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if not _statsmodels_available():
        return (
            {
                "model_status": "not_available",
                "reason": "statsmodels no esta instalado en el entorno actual; se declaro en requirements.txt y no se instalan dependencias durante la ejecucion.",
                "formula": MODEL_FORMULA,
                "reference_direction": REFERENCE_DIRECTION,
                "estimator": "statsmodels Logit",
                "standard_error_method": STANDARD_ERROR_METHOD,
            },
            pd.DataFrame(columns=COEFFICIENT_COLUMNS),
        )
    model_data = analytic.copy()
    server_reference = sorted(model_data["server_player"].unique())[0]
    formula = MODEL_FORMULA.format(server_reference=server_reference)
    # Intercept + two non-reference levels for direction, surface and period,
    # plus one indicator for each non-reference server.
    expected_columns = 1 + 2 + 2 + 2 + (model_data["server_player"].nunique() - 1)
    expected_rows = int(len(model_data))
    estimated_dense_bytes = expected_rows * expected_columns * BYTES_PER_DENSE_ELEMENT
    if estimated_dense_bytes > MEMORY_LIMIT_BYTES:
        return (
            {
                "model_status": "not_available",
                "reason": "estimated_dense_design_exceeds_memory_limit",
                "formula": formula,
                "reference_direction": REFERENCE_DIRECTION,
                "server_reference": server_reference,
                "estimator": "GLM binomial (planned, not fitted)",
                "standard_error_method": STANDARD_ERROR_METHOD,
                "n_observations": expected_rows,
                "planned_model_rows": expected_rows,
                "planned_model_columns": int(expected_columns),
                "bytes_per_dense_element": BYTES_PER_DENSE_ELEMENT,
                "estimated_dense_design_bytes": int(estimated_dense_bytes),
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            },
            pd.DataFrame(columns=COEFFICIENT_COLUMNS),
        )
    model_data["direction"] = pd.Categorical(
        model_data["direction"].astype(str),
        categories=DIRECTION_ORDER,
        ordered=False,
    )
    model_data["surface"] = pd.Categorical(model_data["surface"], categories=["Hard", "Clay", "Grass"])
    model_data["derived_period"] = pd.Categorical(model_data["derived_period"], categories=["to_2009", "2010s", "2020s"])
    model_data["server_player"] = pd.Categorical(model_data["server_player"], categories=sorted(model_data["server_player"].unique()))
    grouped = model_data.groupby(["match_id", "server_player", "surface", "derived_period", "direction"], observed=True, as_index=False).agg(successes=("server_won_point", "sum"), points=("server_won_point", "size"))
    grouped["failures"] = grouped["points"] - grouped["successes"]
    estimated_dense_bytes = int(len(model_data) * (len(grouped["server_player"].unique()) + 8) * 8)
    estimated_grouped_design_bytes = int(len(grouped) * (len(grouped["server_player"].unique()) + 8) * 8)
    if estimated_grouped_design_bytes > 128 * 1024 * 1024:
        return ({"model_status": "not_available", "reason": "La matriz de diseno agrupada supera el limite de memoria de 128 MiB; no se estiman inferencias no fiables.", "formula": formula, "reference_direction": REFERENCE_DIRECTION, "server_reference": server_reference, "estimator": "statsmodels GLM Binomial grouped", "standard_error_method": STANDARD_ERROR_METHOD, "n_observations": int(len(analytic)), "model_rows": int(len(grouped)), "estimated_dense_design_bytes": estimated_dense_bytes, "estimated_grouped_design_bytes": estimated_grouped_design_bytes}, pd.DataFrame(columns=COEFFICIENT_COLUMNS))
    try:
        import patsy
        import statsmodels.api as sm
        design = patsy.dmatrix(formula.split(" ~ ", 1)[1], grouped, return_type="dataframe")
        fitted = sm.GLM(grouped[["successes", "failures"]].astype(float).to_numpy(), design.astype(float).to_numpy(), family=sm.families.Binomial()).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": grouped["match_id"]},
        )
    except Exception as exc:
        return (
            {
                "model_status": "not_available",
                "reason": f"El ajuste no pudo estimarse de forma fiable: {type(exc).__name__}: {exc}",
                "formula": formula,
                "reference_direction": REFERENCE_DIRECTION,
                "estimator": "statsmodels Logit",
                "standard_error_method": STANDARD_ERROR_METHOD,
            },
            pd.DataFrame(columns=COEFFICIENT_COLUMNS),
        )
    conf = fitted.conf_int(alpha=0.05)
    rows = [
        _coefficient_row(
            term,
            fitted.params[term],
            fitted.bse[term],
            fitted.tvalues[term],
            fitted.pvalues[term],
            conf.loc[term, 0],
            conf.loc[term, 1],
        )
        for term in fitted.params.index
    ]
    coefficients = pd.DataFrame(rows, columns=COEFFICIENT_COLUMNS)
    coefficients["_direction_order"] = coefficients["level"].map({"body": 0, "T": 1})
    coefficients["_variable_order"] = coefficients["variable"].map(
        {"direction": 0, "surface": 1, "derived_period": 2, "server_player": 3, "other": 4}
    )
    coefficients = coefficients.sort_values(
        ["_variable_order", "_direction_order", "term"],
        na_position="last",
    ).drop(columns=["_variable_order", "_direction_order"]).reset_index(drop=True)
    numeric_coefficients = coefficients.select_dtypes(include=[np.number])
    if not np.isfinite(numeric_coefficients.to_numpy(dtype=float)).all():
        return (
            {
                "model_status": "not_available",
                "reason": "El ajuste produjo coeficientes, intervalos o errores no finitos; no se publican resultados no fiables.",
                "formula": MODEL_FORMULA,
                "reference_direction": REFERENCE_DIRECTION,
                "estimator": "statsmodels Logit",
                "standard_error_method": STANDARD_ERROR_METHOD,
            },
            pd.DataFrame(columns=COEFFICIENT_COLUMNS),
        )
    global_test = _direction_global_test(fitted)
    return (
        {
            "model_status": "available",
            "formula": formula,
            "reference_direction": REFERENCE_DIRECTION,
            "estimator": "statsmodels Logit",
            "standard_error_method": STANDARD_ERROR_METHOD,
            "n_observations": int(len(analytic)),
            "model_rows": int(len(grouped)),
            "estimated_dense_design_bytes": estimated_dense_bytes,
            "estimated_grouped_design_bytes": estimated_grouped_design_bytes,
            "server_reference": server_reference,
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "global_direction_test": global_test,
        },
        coefficients,
    )


def _direction_global_test(fitted: Any) -> dict[str, Any]:
    terms = list(fitted.params.index)
    direction_terms = [
        term for term in terms
        if "C(direction, Treatment(reference=\"wide\"))" in term
    ]
    if len(direction_terms) != 2:
        return {
            "status": "not_available",
            "reason": "No se encontraron exactamente los terminos body y T frente a wide.",
        }
    matrix = np.zeros((2, len(terms)))
    for row, term in enumerate(direction_terms):
        matrix[row, terms.index(term)] = 1.0
    try:
        test = fitted.wald_test(matrix, scalar=True)
    except Exception as exc:
        return {
            "status": "not_available",
            "reason": f"No se pudo calcular el contraste global: {type(exc).__name__}: {exc}",
        }
    return {
        "status": "available",
        "hypothesis": "body_vs_wide = 0 and T_vs_wide = 0",
        "statistic": float(test.statistic),
        "df": int(test.df_denom) if hasattr(test, "df_denom") else len(direction_terms),
        "p_value": float(test.pvalue),
    }


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
        "servers": int(analytic["server_player"].nunique()),
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
    for metric in ("n_observations", "planned_model_rows", "planned_model_columns", "bytes_per_dense_element", "estimated_dense_design_bytes", "memory_limit_bytes", "model_rows", "estimated_grouped_design_bytes", "pseudo_r_squared", "log_likelihood", "aic", "bic"):
        if metric in adjusted and adjusted[metric] is not None:
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
    if summary is not None:
        if summary["analytical_points"] != len(analytic):
            raise ValueError("summary analytical_points no reconcilia.")
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
