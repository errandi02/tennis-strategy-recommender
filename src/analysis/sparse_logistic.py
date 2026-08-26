"""Logistica binaria no penalizada sobre disenos CSR."""

from __future__ import annotations

import numpy as np
from scipy import optimize, sparse, stats
from dataclasses import dataclass

GRADIENT_TOLERANCE = 1e-6
COEFFICIENT_ABSOLUTE_LIMIT = 30.0
SATURATION_TOLERANCE = 1e-12
MAX_DENSE_HESSIAN_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class SparseFitNotPublishableError(RuntimeError):
    diagnostics: dict

    def __str__(self) -> str:
        return "sparse_fit_not_publishable"


class TermMapping(dict):
    """Mapeo termino-columna con metadatos canonicos del diseno CSR."""

    def __init__(self, *args, metadata: tuple[dict, ...], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata = metadata


def build_design(frame, required_categories: dict[str, tuple[str, ...]], references: dict[str, str]):
    """Construye CSR determinista: intercepto y dummies no referencia."""
    terms = ["Intercept"]
    metadata = [{
        "term": "Intercept", "variable": "intercept", "level": None,
        "reference_level": None, "column_index": 0,
    }]
    rows = [np.arange(len(frame), dtype=np.int64)]
    cols = [np.zeros(len(frame), dtype=np.int32)]
    for name, categories in required_categories.items():
        if len(categories) != len(set(categories)) or references[name] not in categories:
            raise ValueError("Categorias o referencia duplicada/invalida.")
        observed = set(frame[name])
        if not observed.issubset(set(categories)) or references[name] not in observed:
            raise ValueError("Categoria desconocida o referencia ausente.")
        for category in categories:
            if category != references[name]:
                terms.append(f"{name}[{category}]")
                metadata.append({
                    "term": terms[-1], "variable": name, "level": category,
                    "reference_level": references[name], "column_index": len(terms) - 1,
                })
                matching = np.flatnonzero(frame[name].to_numpy() == category)
                rows.append(matching)
                cols.append(np.full(len(matching), len(terms) - 1, dtype=np.int32))
    if len(terms) != len(set(terms)):
        raise ValueError("Mapping de columnas duplicado.")
    matrix = sparse.coo_matrix((np.ones(sum(len(item) for item in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(len(frame), len(terms))).tocsr()
    return matrix, TermMapping(
        {term: index for index, term in enumerate(terms)}, metadata=tuple(metadata)
    )


def _validate(design: sparse.csr_matrix, outcome: np.ndarray, clusters: np.ndarray | None = None) -> None:
    if not sparse.isspmatrix_csr(design):
        raise TypeError("design debe ser una matriz CSR.")
    if design.shape[0] != len(outcome) or not np.isfinite(design.data).all():
        raise ValueError("Diseno u outcome no valido.")
    if not np.isin(outcome, [0, 1]).all() or np.unique(outcome).size != 2:
        raise ValueError("outcome debe ser binario y tener variacion.")
    if design.shape[1] ** 2 * 8 > MAX_DENSE_HESSIAN_BYTES:
        raise MemoryError("Hessiano denso supera el limite seguro.")
    if np.linalg.matrix_rank((design.T @ design).toarray()) != design.shape[1]:
        raise ValueError("Hessiano singular o columnas no identificadas.")
    if clusters is not None and (len(clusters) != len(outcome) or np.unique(clusters).size < 2):
        raise ValueError("Se requieren al menos dos clusters completos.")


def log_likelihood(beta: np.ndarray, design: sparse.csr_matrix, outcome: np.ndarray) -> float:
    eta = design @ beta
    return float(np.sum(outcome * eta - np.logaddexp(0.0, eta)))


def gradient(beta: np.ndarray, design: sparse.csr_matrix, outcome: np.ndarray) -> np.ndarray:
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -700, 700)))
    return np.asarray(design.T @ (outcome - probability)).ravel()


def hessian(beta: np.ndarray, design: sparse.csr_matrix) -> np.ndarray:
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -700, 700)))
    weight = probability * (1.0 - probability)
    return -((design.T @ design.multiply(weight[:, None])).toarray())


def fit_mle(
    design: sparse.csr_matrix,
    outcome: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
    initial_coefficients: np.ndarray | None = None,
) -> dict:
    """Ajusta MLE BFGS, opcionalmente continuando desde coeficientes previos."""
    _validate(design, outcome)
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations debe ser un entero positivo.")
    if initial_coefficients is None:
        initial_point = "zeros"
        initial = np.zeros(design.shape[1], dtype=float)
        used_warm_start = False
    else:
        initial = np.asarray(initial_coefficients, dtype=float)
        if initial.ndim != 1 or initial.shape[0] != design.shape[1]:
            raise ValueError("initial_coefficients tiene una dimension incompatible.")
        if not np.isfinite(initial).all():
            raise ValueError("initial_coefficients debe contener valores finitos.")
        initial = initial.copy()
        initial_point = "previous_solution"
        used_warm_start = True
    result = optimize.minimize(
        lambda beta: -log_likelihood(beta, design, outcome),
        initial,
        jac=lambda beta: -gradient(beta, design, outcome),
        method="BFGS",
        options={"gtol": tolerance, "maxiter": max_iterations},
    )
    beta = np.asarray(result.x)
    information = -hessian(beta, design)
    covariance = np.linalg.inv(information)
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -700, 700)))
    gradient_norm = float(np.linalg.norm(gradient(beta, design, outcome)))
    saturated = int(np.count_nonzero((probability <= SATURATION_TOLERANCE) | (probability >= 1 - SATURATION_TOLERANCE)))
    success = bool(getattr(result, "success", False))
    reasons = []
    if not success: reasons.append("optimizer_not_converged")
    if gradient_norm > GRADIENT_TOLERANCE: reasons.append("gradient_above_tolerance")
    if not np.isfinite(beta).all(): reasons.append("non_finite_coefficients")
    if saturated: reasons.append("saturated_probabilities")
    if np.max(np.abs(beta)) > COEFFICIENT_ABSOLUTE_LIMIT: reasons.append("extreme_coefficients")
    objective = getattr(result, "fun", None)
    diagnostics = {"reason_codes": reasons, "reason": reasons[0] if reasons else None, "converged": success, "optimizer_success": success, "optimizer": "BFGS", "tolerance": float(tolerance), "max_iterations": max_iterations, "initial_point": initial_point, "used_warm_start": used_warm_start, "optimizer_status": getattr(result, "status", None), "optimizer_message": getattr(result, "message", None), "iterations": getattr(result, "nit", None), "objective_value": float(objective) if objective is not None and np.isfinite(objective) else None, "log_likelihood": log_likelihood(beta, design, outcome), "gradient_norm": gradient_norm, "max_abs_coefficient": float(np.max(np.abs(beta))), "minimum_probability": float(np.min(probability)), "maximum_probability": float(np.max(probability)), "saturated_probability_count": saturated, "hessian_status": "invertible", "covariance_status": "conventional_valid", "finite_coefficients": bool(np.isfinite(beta).all()), "finite_probabilities": bool(np.isfinite(probability).all()), "publishable": not reasons}
    return {"beta": beta, "covariance": covariance, "probability": probability, "diagnostics": diagnostics, **diagnostics}


def cluster_covariance(design: sparse.csr_matrix, outcome: np.ndarray, beta: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    _validate(design, outcome, clusters)
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -700, 700)))
    bread = np.linalg.inv(-hessian(beta, design))
    _, inverse = np.unique(clusters, return_inverse=True)
    scores = np.zeros((inverse.max() + 1, design.shape[1]))
    for cluster in range(scores.shape[0]):
        rows = np.flatnonzero(inverse == cluster)
        scores[cluster] = np.asarray(design[rows].T @ (outcome[rows] - probability[rows])).ravel()
    meat = scores.T @ scores
    # Misma correccion finita por defecto que statsmodels.cov_cluster:
    # G/(G-1) * (N-1)/(N-k).
    groups, observations, parameters = scores.shape[0], design.shape[0], design.shape[1]
    correction = (groups / (groups - 1)) * ((observations - 1) / (observations - parameters))
    return correction * (bread @ meat @ bread)


def inference(beta: np.ndarray, covariance: np.ndarray) -> dict:
    if not np.isfinite(beta).all() or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T) or np.any(np.diag(covariance) < 0):
        raise ValueError("Covarianza o coeficientes no validos.")
    standard_error = np.sqrt(np.diag(covariance))
    statistic = beta / standard_error
    p_value = 2 * stats.norm.sf(np.abs(statistic))
    critical = stats.norm.ppf(0.975)
    return {"std_error": standard_error, "statistic": statistic, "p_value": p_value, "ci_low": beta - critical * standard_error, "ci_high": beta + critical * standard_error}


def wald_test(beta: np.ndarray, covariance: np.ndarray, indices: tuple[int, ...]) -> dict:
    if not indices or max(indices) >= len(beta):
        raise ValueError("Terminos de contraste ausentes.")
    selected = beta[list(indices)]
    selected_covariance = covariance[np.ix_(indices, indices)]
    if not np.isfinite(selected_covariance).all() or np.linalg.matrix_rank(selected_covariance) != len(indices):
        raise ValueError("Covarianza de contraste no valida.")
    statistic = float(selected @ np.linalg.solve(selected_covariance, selected))
    return {"statistic": statistic, "df": len(indices), "p_value": float(stats.chi2.sf(statistic, len(indices)))}
