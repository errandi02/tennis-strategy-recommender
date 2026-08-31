"""Seleccion metodologica congelada de la politica de evidencia.

Este modulo consume exclusivamente los artefactos agregados publicados por
``evidence_policy_validation``. No lee datos, no reconstruye historia y no
evalua el conjunto de test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

VALIDATION_SUMMARY_PATH = REPORTS_DIR / "evidence_policy_validation_summary.json"
VALIDATION_CANDIDATES_PATH = TABLES_DIR / "evidence_policy_validation_candidates.csv"
VALIDATION_PARETO_PATH = TABLES_DIR / "evidence_policy_validation_pareto.csv"
VALIDATION_BY_FOLD_PATH = TABLES_DIR / "evidence_policy_validation_by_fold.csv"

SUMMARY_PATH = REPORTS_DIR / "evidence_policy_selection_summary.json"
COMPARISON_PATH = TABLES_DIR / "evidence_policy_selection_comparison.csv"
REPRO_DIR = REPORTS_DIR / "repro_evidence_policy_selection"

ANALYSIS_NAME = "evidence_policy_selection"
ANALYSIS_VERSION = "1.0.0"
UPSTREAM_COMMIT = "88fafe7"
PRIMARY_POLICY = "p050_m05_surface_then_global"
GLOBAL_CONTROL = "p050_m05_global_only"
NOT_AVAILABLE_PATH = "no_candidate_satisfied_limits"
FOLDS = (2020, 2021, 2022, 2023)
DIRECTIONS = ("wide", "body", "T")
ROLES = ("server", "opponent")
SURFACES = ("Hard", "Clay", "Grass")

MIN_POINTS = 50
MIN_MATCHES = 5
MIN_WORST_FOLD_COVERAGE = 0.50
MAX_WORST_FOLD_WILSON = 0.25
MAX_WORST_FOLD_STABILITY = 0.08

SHORTLIST = (
    "p025_m03_global_only",
    "p025_m05_global_only",
    "p025_m10_global_only",
    "p050_m03_global_only",
    "p050_m05_global_only",
    "p050_m05_surface_then_global",
    "p050_m10_global_only",
    "p100_m10_global_only",
)

SOURCE_PATHS = (
    VALIDATION_SUMMARY_PATH,
    VALIDATION_CANDIDATES_PATH,
    VALIDATION_PARETO_PATH,
    VALIDATION_BY_FOLD_PATH,
)
UPSTREAM_HASHES = {
    "reports/evidence_policy_validation_summary.json":
        "5AAC50C23DE8F9EECE61A573D445FB1B816D4769BE61D6875A614BE1E0E812BF",
    "reports/tables/evidence_policy_validation_candidates.csv":
        "D819622887DD8670BB4FFCD8A9FFA7A1A6522739870C8890E71689D1CBB8051D",
    "reports/tables/evidence_policy_validation_pareto.csv":
        "8DA16E25CB1571EF901A56A70094616FC73EF462F2616F515CB645C2B1720968",
    "reports/tables/evidence_policy_validation_by_fold.csv":
        "72310FFDB2CB9328274C36C8995B5B0CC8B804F9F9462F9394EE8E592F803E93",
}

PARETO_AXES = (
    "worst_fold_complete_match_coverage",
    "worst_fold_p90_wilson_width",
    "worst_fold_p90_absolute_change",
)

COMPARISON_COLUMNS = [
    "policy_id",
    "min_points",
    "min_matches",
    "scope_policy",
    "pareto_status",
    "passes_structural_minimum",
    "passes_coverage_limit",
    "passes_wilson_limit",
    "passes_stability_limit",
    "passes_all_absolute_limits",
    "worst_fold_complete_match_coverage",
    "worst_fold_wilson_p90",
    "worst_fold_stability_p90",
    "complete_match_coverage",
    "median_wilson_width",
    "selection_role",
    "rejection_reason",
    "selected",
]

BY_FOLD_KEY = [
    "policy_id", "fold", "direction", "role", "stratum_type",
    "surface", "selected_scope",
]

RECONCILIATION_KEYS = {
    "upstream_hashes_exact",
    "upstream_candidate_count_27",
    "pareto_8_non_dominated",
    "pareto_19_dominated",
    "shortlist_exact",
    "candidate_keys_unique",
    "pareto_axes_finite",
    "dominance_reconstructed",
    "required_policies_present",
    "folds_2020_2023_only",
    "test_remains_sealed",
    "comparison_canonical",
    "selection_cardinality_valid",
    "selection_state_coherent",
    "selected_contract_coherent",
    "no_policy_winner_chosen_from_test",
}

FAILURE_REASON_CODES = {
    "primary_pareto_eligible",
    "primary_passes_structural_minimum",
    "primary_passes_coverage_limit",
    "primary_passes_wilson_limit",
    "primary_passes_stability_limit",
    "global_control_pareto_eligible",
    "global_control_passes_structural_minimum",
    "global_control_passes_coverage_limit",
    "global_control_passes_wilson_limit",
    "global_control_passes_stability_limit",
    "primary_fold_contract",
    "primary_scope_contract",
    "global_control_fold_contract",
    "global_control_scope_contract",
    "upstream_artifact_missing",
    "upstream_hash_mismatch",
    "upstream_deserialization_failed",
    "upstream_contract_failure",
}


class UpstreamContractError(ValueError):
    """Error auditable en los artefactos de validacion publicados."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class UpstreamArtifacts:
    summary: dict[str, Any]
    candidates: pd.DataFrame
    pareto: pd.DataFrame
    by_fold: pd.DataFrame
    payloads: Mapping[str, bytes]
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class EvidencePolicySelectionResult:
    summary: dict[str, Any]
    comparison: pd.DataFrame
    publication_fingerprint: str
    upstream: UpstreamArtifacts | None = None


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    return buffer.getvalue().encode("utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("El resultado contiene NaN o infinito.")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    return value


def load_upstream_artifacts(
    paths: tuple[Path, Path, Path, Path] = SOURCE_PATHS,
    expected_hashes: Mapping[str, str] = UPSTREAM_HASHES,
) -> UpstreamArtifacts:
    """Lee una sola vez los cuatro artefactos agregados congelados."""

    if len(paths) != 4:
        raise ValueError("Se requieren exactamente cuatro artefactos upstream.")
    payloads: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise UpstreamContractError("upstream_artifact_missing", f"Falta {path.name}.")
        relative = _relative(path) if path.is_relative_to(ROOT) else path.name
        payload = path.read_bytes()
        digest = _sha256(payload)
        payloads[relative] = payload
        hashes[relative] = digest
        expected = expected_hashes.get(relative)
        if expected is None or digest != expected:
            raise UpstreamContractError(
                "upstream_hash_mismatch",
                f"SHA-256 inesperado para {relative}: {digest}",
            )
    summary_key, candidates_key, pareto_key, by_fold_key = [
        _relative(path) if path.is_relative_to(ROOT) else path.name for path in paths
    ]
    try:
        summary = json.loads(payloads[summary_key].decode("utf-8"))
        candidates = pd.read_csv(io.BytesIO(payloads[candidates_key]))
        pareto = pd.read_csv(io.BytesIO(payloads[pareto_key]))
        by_fold = pd.read_csv(io.BytesIO(payloads[by_fold_key]))
    except Exception as exc:
        raise UpstreamContractError("upstream_deserialization_failed", str(exc)) from exc
    return UpstreamArtifacts(summary, candidates, pareto, by_fold, payloads, hashes)


def _dominates(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
    no_worse = left[0] >= right[0] and left[1] <= right[1] and left[2] <= right[2]
    strictly_better = left[0] > right[0] or left[1] < right[1] or left[2] < right[2]
    return no_worse and strictly_better


def reconstruct_dominators(candidates: pd.DataFrame) -> dict[str, list[str]]:
    """Reconstruye Pareto directamente con floats completos."""

    vectors: dict[str, tuple[float, float, float]] = {}
    for row in candidates.itertuples(index=False):
        values = tuple(float(getattr(row, axis)) for axis in PARETO_AXES)
        if not all(math.isfinite(value) for value in values):
            raise UpstreamContractError("undefined_pareto_axis", "Un eje Pareto no es finito.")
        vectors[str(row.policy_id)] = values
    return {
        policy_id: sorted(
            other for other, other_vector in vectors.items()
            if other != policy_id and _dominates(other_vector, vector)
        )
        for policy_id, vector in vectors.items()
    }


def _assert_numeric_domains(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    if np.isinf(numeric).any():
        raise UpstreamContractError("upstream_non_finite", "Un CSV contiene infinito.")
    for column in frame.columns:
        if "coverage" in column or "proportion" in column:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if ((values < 0) | (values > 1)).any():
                raise UpstreamContractError("upstream_metric_out_of_domain", column)


def validate_upstream_artifacts(upstream: UpstreamArtifacts) -> dict[str, Any]:
    """Valida hashes, sellado, folds, Pareto y coherencia entre artefactos."""

    summary, candidates, pareto, by_fold = (
        upstream.summary, upstream.candidates, upstream.pareto, upstream.by_fold
    )
    if set(upstream.hashes) != set(UPSTREAM_HASHES) or any(
        upstream.hashes.get(path) != digest for path, digest in UPSTREAM_HASHES.items()
    ):
        raise UpstreamContractError("upstream_hash_mismatch", "Los hashes no son los congelados.")
    if summary.get("analysis_status") != "available":
        raise UpstreamContractError("upstream_not_available", "La validacion upstream no esta disponible.")
    if len(candidates) != 27 or not candidates["policy_id"].is_unique:
        raise UpstreamContractError("candidate_contract_mismatch", "Deben existir 27 candidatas unicas.")
    if len(pareto) != 27 or not pareto["policy_id"].is_unique:
        raise UpstreamContractError("pareto_contract_mismatch", "Pareto debe contener 27 claves unicas.")
    if candidates["policy_id"].tolist() != pareto["policy_id"].tolist():
        raise UpstreamContractError("candidate_pareto_key_mismatch", "Candidates y Pareto no reconcilian.")
    if by_fold.duplicated(BY_FOLD_KEY).any():
        raise UpstreamContractError("by_fold_duplicate_key", "La clave by-fold no es unica.")
    if not by_fold["fold"].isin(FOLDS).all() or tuple(sorted(by_fold["fold"].unique())) != FOLDS:
        raise UpstreamContractError("forbidden_fold", "Solo se admiten folds 2020--2023.")
    for frame in (candidates, pareto, by_fold):
        _assert_numeric_domains(frame)

    dominators = reconstruct_dominators(candidates)
    independent_shortlist = tuple(
        policy_id for policy_id in candidates["policy_id"] if not dominators[policy_id]
    )
    if independent_shortlist != SHORTLIST:
        raise UpstreamContractError("shortlist_mismatch", "La shortlist Pareto no es la congelada.")
    published = pareto.set_index("policy_id")
    for policy_id, expected_dominators in dominators.items():
        raw = published.at[policy_id, "dominated_by"]
        observed = [] if pd.isna(raw) or raw == "" else sorted(str(raw).split("|"))
        expected_status = "non_dominated" if not expected_dominators else "dominated"
        if observed != expected_dominators or published.at[policy_id, "pareto_status"] != expected_status:
            raise UpstreamContractError("pareto_dominance_mismatch", policy_id)
        candidate = candidates.loc[candidates["policy_id"].eq(policy_id)].iloc[0]
        for axis in PARETO_AXES:
            if float(candidate[axis]) != float(published.at[policy_id, axis]):
                raise UpstreamContractError("pareto_axis_mismatch", f"{policy_id}:{axis}")

    pareto_summary = summary.get("pareto_summary", {})
    if (
        tuple(pareto_summary.get("non_dominated_policy_ids", ())) != SHORTLIST
        or pareto_summary.get("dominated_policies") != 19
        or pareto_summary.get("not_comparable_policy_ids") != []
        or pareto_summary.get("automatic_selection_performed") is not False
    ):
        raise UpstreamContractError("summary_pareto_mismatch", "El summary no reconcilia con Pareto.")
    sealed = summary.get("sealed_test_contract", {})
    if sealed != {
        "allowed_folds": list(FOLDS),
        "latest_allowed_target_date": "2023-12-31",
        "test_evaluation_runs": 0,
        "test_matches_evaluated": 0,
        "test_rows_evaluated": 0,
        "test_status": "sealed",
        "used_for_method_selection": False,
    }:
        raise UpstreamContractError("sealed_test_contract_mismatch", "El test no permanece sellado.")
    if "test_rows_read" in json.dumps(summary, ensure_ascii=False):
        raise UpstreamContractError("ambiguous_test_read_field", "Campo test_rows_read ambiguo.")
    if not {PRIMARY_POLICY, GLOBAL_CONTROL}.issubset(set(candidates["policy_id"])):
        raise UpstreamContractError("required_policy_missing", "Falta primary o control.")

    # Recalcula el fingerprint upstream usando los bytes publicados.
    core = {key: value for key, value in summary.items() if key != "publication_fingerprint"}
    digest = hashlib.sha256()
    for key in (
        "reports/tables/evidence_policy_validation_candidates.csv",
        "reports/tables/evidence_policy_validation_by_fold.csv",
        "reports/tables/evidence_policy_validation_pareto.csv",
    ):
        digest.update(upstream.payloads[key])
    digest.update(json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    if digest.hexdigest().upper() != summary.get("publication_fingerprint"):
        raise UpstreamContractError("upstream_fingerprint_mismatch", "Fingerprint upstream alterado.")
    return {
        "candidate_count": 27,
        "non_dominated_count": 8,
        "dominated_count": 19,
        "dominators": dominators,
    }


def evaluate_absolute_limits(row: pd.Series | Mapping[str, Any]) -> dict[str, bool]:
    coverage = float(row["worst_fold_complete_match_coverage"])
    wilson = float(row["worst_fold_p90_wilson_width"])
    stability = float(row["worst_fold_p90_absolute_change"])
    if not all(math.isfinite(value) for value in (coverage, wilson, stability)):
        raise ValueError("Los tres limites requieren floats finitos.")
    checks = {
        "passes_coverage_limit": coverage >= MIN_WORST_FOLD_COVERAGE,
        "passes_wilson_limit": wilson <= MAX_WORST_FOLD_WILSON,
        "passes_stability_limit": stability <= MAX_WORST_FOLD_STABILITY,
    }
    checks["passes_all_absolute_limits"] = all(checks.values())
    return checks


def passes_structural_minimum(row: pd.Series | Mapping[str, Any]) -> bool:
    points = row["min_points"]
    matches = row["min_matches"]
    if isinstance(points, (bool, np.bool_)) or isinstance(matches, (bool, np.bool_)):
        return False
    return int(points) >= MIN_POINTS and int(matches) >= MIN_MATCHES


def _overall_fold_rows(by_fold: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    rows = by_fold.loc[
        by_fold["policy_id"].eq(policy_id)
        & by_fold["stratum_type"].eq("overall")
        & by_fold["surface"].eq("<ALL>")
        & by_fold["selected_scope"].eq("<ALL>")
    ].copy()
    expected = {(fold, direction, role) for fold in FOLDS for direction in DIRECTIONS for role in ROLES}
    observed = set(map(tuple, rows[["fold", "direction", "role"]].to_numpy()))
    if observed != expected or len(rows) != len(expected):
        raise UpstreamContractError("fold_contract_mismatch", f"Filas overall incompletas: {policy_id}")
    return rows


def validate_fold_contract(
    row: pd.Series,
    by_fold: pd.DataFrame,
) -> dict[str, Any]:
    policy_id = str(row["policy_id"])
    rows = _overall_fold_rows(by_fold, policy_id)
    if rows["coverage_complete_match"].isna().any():
        raise UpstreamContractError("undefined_fold_coverage", policy_id)
    if (rows["eligible_orientations"].gt(0) & rows["wilson_width_p90"].isna()).any():
        raise UpstreamContractError("undefined_fold_wilson", policy_id)
    comparable = rows["fold"].gt(2020) & rows["comparable_players"].gt(0)
    if (comparable & rows["absolute_change_p90"].isna()).any():
        raise UpstreamContractError("undefined_fold_stability", policy_id)
    if not rows["target_orientations"].eq(rows["target_matches"] * 2).all():
        raise UpstreamContractError("fold_denominator_mismatch", policy_id)
    expected_coverage = rows["complete_matches"] / rows["target_matches"]
    if not np.allclose(rows["coverage_complete_match"], expected_coverage, rtol=0, atol=1e-15):
        raise UpstreamContractError("fold_coverage_mismatch", policy_id)
    coverage_by_fold = rows.groupby("fold", sort=True)["coverage_complete_match"].first()
    for _, group in rows.groupby("fold", sort=False):
        if group["coverage_complete_match"].nunique(dropna=False) != 1:
            raise UpstreamContractError("fold_coverage_not_unique", policy_id)
    derived = {
        "worst_fold_complete_match_coverage": float(coverage_by_fold.min()),
        "worst_fold_p90_wilson_width": float(rows.groupby("fold")["wilson_width_p90"].max().max()),
        "worst_fold_p90_absolute_change": float(rows.loc[comparable, "absolute_change_p90"].max()),
    }
    for key, value in derived.items():
        if not math.isclose(value, float(row[key]), rel_tol=0, abs_tol=1e-15):
            raise UpstreamContractError("candidate_fold_metric_mismatch", f"{policy_id}:{key}")
    return {
        "valid": True,
        "coverage_by_fold": {str(int(fold)): float(value) for fold, value in coverage_by_fold.items()},
        "wilson_defined_when_observed": True,
        "stability_defined_when_comparable": True,
        "denominators_reconciled": True,
    }


def validate_scope_contract(
    row: pd.Series,
    upstream_summary: Mapping[str, Any],
    by_fold: pd.DataFrame,
) -> dict[str, Any]:
    policy_id = str(row["policy_id"])
    scope_policy = str(row["scope_policy"])
    if row["scope_coherence_reconciled"] is not True and row["scope_coherence_reconciled"] != True:
        raise UpstreamContractError("mixed_scope_detected", policy_id)
    contracts = upstream_summary.get("scope_policy_contracts", {})
    if contracts.get("server_and_opponent_scope_must_match") is not True:
        raise UpstreamContractError(
            "mixed_scope_contract_missing", f"Contrato de mixed scope ausente: {policy_id}"
        )
    rows = _overall_fold_rows(by_fold, policy_id)
    if not rows["surface_selections"].add(rows["global_selections"]).add(rows["abstentions"]).eq(
        rows["target_orientations"]
    ).all():
        raise UpstreamContractError("scope_counts_do_not_reconcile", policy_id)
    if scope_policy == "surface_then_global":
        expected_text = "surface only if both roles pass; otherwise global only if both pass"
        if contracts.get(scope_policy) != expected_text:
            raise UpstreamContractError("surface_fallback_contract_mismatch", policy_id)
        if float(row["surface_selection_proportion"]) <= 0 or float(row["global_selection_proportion"]) <= 0:
            raise UpstreamContractError("fallback_not_observed", policy_id)
    elif scope_policy == "global_only":
        if contracts.get(scope_policy) != "global history only":
            raise UpstreamContractError("global_scope_contract_mismatch", policy_id)
        if float(row["surface_selection_proportion"]) != 0:
            raise UpstreamContractError("surface_used_by_global_control", policy_id)
    else:
        raise UpstreamContractError("selection_scope_not_allowed", policy_id)
    surface_rows = by_fold.loc[
        by_fold["policy_id"].eq(policy_id)
        & by_fold["stratum_type"].eq("surface")
    ]
    if set(surface_rows["surface"]) != set(SURFACES):
        raise UpstreamContractError("surface_coverage_missing", policy_id)
    return {
        "valid": True,
        "joint_role_eligibility": True,
        "mixed_scopes": False,
        "fallback_accounted": scope_policy == "surface_then_global",
        "abstention_accounted": True,
        "surfaces_reported": list(SURFACES),
    }


def evaluate_candidate(
    row: pd.Series,
    pareto: pd.DataFrame,
    by_fold: pd.DataFrame,
    upstream_summary: Mapping[str, Any],
) -> dict[str, Any]:
    policy_id = str(row["policy_id"])
    pareto_row = pareto.loc[pareto["policy_id"].eq(policy_id)]
    if len(pareto_row) != 1:
        raise UpstreamContractError("required_policy_missing", policy_id)
    pareto_record = pareto_row.iloc[0]
    published_dominators = pareto_record["dominated_by"]
    pareto_pass = (
        policy_id in SHORTLIST
        and pareto_record["pareto_status"] == "non_dominated"
        and (pd.isna(published_dominators) or published_dominators == "")
        and all(math.isfinite(float(pareto_record[axis])) for axis in PARETO_AXES)
    )
    structural = passes_structural_minimum(row)
    limits = evaluate_absolute_limits(row)
    limits["passes_all_absolute_limits"] = bool(
        structural
        and limits["passes_coverage_limit"]
        and limits["passes_wilson_limit"]
        and limits["passes_stability_limit"]
    )
    fold = validate_fold_contract(row, by_fold)
    scope = validate_scope_contract(row, upstream_summary, by_fold)
    result: dict[str, Any] = {
        "policy_id": policy_id,
        "pareto_eligible": pareto_pass,
        "passes_structural_minimum": structural,
        **limits,
        "fold_contract": fold,
        "scope_contract": scope,
    }
    result["passes_all_selection_criteria"] = bool(
        result["pareto_eligible"]
        and result["passes_structural_minimum"]
        and result["passes_all_absolute_limits"]
        and fold["valid"]
        and scope["valid"]
    )
    result["failed_criteria"] = [
        key for key in (
            "pareto_eligible", "passes_structural_minimum", "passes_coverage_limit",
            "passes_wilson_limit", "passes_stability_limit",
        ) if not result[key]
    ]
    return result


def decide_selection(
    primary_checks: Mapping[str, Any],
    control_descriptive_checks: Mapping[str, Any],
) -> dict[str, Any]:
    """Aplica primary primero y activa el fallback solo si primary falla."""

    def ordered_failures(checks: Mapping[str, Any]) -> list[str]:
        failures: list[str] = []
        if checks.get("pareto_eligible") is not True:
            failures.append("pareto_eligible")
        if checks.get("passes_structural_minimum") is not True:
            failures.append("passes_structural_minimum")
            return failures
        for name in (
            "passes_coverage_limit",
            "passes_wilson_limit",
            "passes_stability_limit",
        ):
            if checks.get(name) is not True:
                failures.append(name)
        fold = checks.get("fold_contract")
        if not isinstance(fold, Mapping) or fold.get("valid") is not True:
            failures.append("fold_contract")
        scope = checks.get("scope_contract")
        if not isinstance(scope, Mapping) or scope.get("valid") is not True:
            failures.append("scope_contract")
        return failures

    primary_failures = ordered_failures(primary_checks)
    if not primary_failures:
        return {
            "selection_status": "selected_primary",
            "selection_path": "primary_candidate_selected",
            "selected_policy": PRIMARY_POLICY,
            "reason_codes": [],
            "primary_evaluated": True,
            "primary_passed": True,
            "control_descriptively_compared": True,
            "fallback_evaluated": False,
            "fallback_activated": False,
            "third_candidate_evaluated": False,
        }
    failed = [f"primary_{item}" for item in primary_failures]
    control_failures = ordered_failures(control_descriptive_checks)
    if not control_failures:
        return {
            "selection_status": "selected_global_fallback",
            "selection_path": "fallback_to_global_control",
            "selected_policy": GLOBAL_CONTROL,
            "reason_codes": failed,
            "primary_evaluated": True,
            "primary_passed": False,
            "control_descriptively_compared": True,
            "fallback_evaluated": True,
            "fallback_activated": True,
            "third_candidate_evaluated": False,
        }
    reasons = failed + [f"global_control_{item}" for item in control_failures]
    return {
        "selection_status": "not_available",
        "selection_path": NOT_AVAILABLE_PATH,
        "selected_policy": None,
        "reason_codes": sorted(set(reasons)),
        "primary_evaluated": True,
        "primary_passed": False,
        "control_descriptively_compared": True,
        "fallback_evaluated": True,
        "fallback_activated": False,
        "third_candidate_evaluated": False,
    }


def build_comparison(
    candidates: pd.DataFrame,
    pareto: pd.DataFrame,
    selected_policy: str | None,
) -> pd.DataFrame:
    candidate_index = candidates.set_index("policy_id")
    pareto_index = pareto.set_index("policy_id")
    rows: list[dict[str, Any]] = []
    for policy_id in SHORTLIST:
        row = candidate_index.loc[policy_id]
        limits = evaluate_absolute_limits(row)
        structural = passes_structural_minimum(row)
        passes_all = bool(
            structural
            and limits["passes_coverage_limit"]
            and limits["passes_wilson_limit"]
            and limits["passes_stability_limit"]
        )
        role = (
            "primary_candidate" if policy_id == PRIMARY_POLICY
            else "global_control" if policy_id == GLOBAL_CONTROL
            else "context_only"
        )
        reasons: list[str] = []
        if int(row["min_points"]) < MIN_POINTS:
            reasons.append("minimum_points_not_met")
        if int(row["min_matches"]) < MIN_MATCHES:
            reasons.append("minimum_matches_not_met")
        for name in ("coverage", "wilson", "stability"):
            if not limits[f"passes_{name}_limit"]:
                reasons.append(f"{name}_limit_not_met")
        if role == "context_only":
            reasons.append("context_only_not_in_primary_fallback_path")
        elif policy_id != selected_policy:
            if role == "global_control" and selected_policy == PRIMARY_POLICY:
                reasons.append("primary_selected_fallback_not_activated")
            else:
                reasons.append("not_selected_by_frozen_decision_path")
        rows.append({
            "policy_id": policy_id,
            "min_points": int(row["min_points"]),
            "min_matches": int(row["min_matches"]),
            "scope_policy": str(row["scope_policy"]),
            "pareto_status": str(pareto_index.at[policy_id, "pareto_status"]),
            "passes_structural_minimum": structural,
            **limits,
            "passes_all_absolute_limits": passes_all,
            "worst_fold_complete_match_coverage": float(row["worst_fold_complete_match_coverage"]),
            "worst_fold_wilson_p90": float(row["worst_fold_p90_wilson_width"]),
            "worst_fold_stability_p90": float(row["worst_fold_p90_absolute_change"]),
            "complete_match_coverage": float(row["coverage_complete_match"]),
            "median_wilson_width": float(row["median_wilson_width"]),
            "selection_role": role,
            "rejection_reason": "|".join(reasons),
            "selected": policy_id == selected_policy,
        })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _policy_metrics(row: pd.Series) -> dict[str, Any]:
    return {
        "policy_id": str(row["policy_id"]),
        "min_points": int(row["min_points"]),
        "min_matches": int(row["min_matches"]),
        "scope_policy": str(row["scope_policy"]),
        "complete_match_coverage": float(row["coverage_complete_match"]),
        "worst_fold_complete_match_coverage": float(row["worst_fold_complete_match_coverage"]),
        "median_wilson_width": float(row["median_wilson_width"]),
        "worst_fold_p90_wilson_width": float(row["worst_fold_p90_wilson_width"]),
        "worst_fold_p90_absolute_change": float(row["worst_fold_p90_absolute_change"]),
        "abstention_proportion": float(row["abstention_proportion"]),
        "surface_selection_proportion": float(row["surface_selection_proportion"]),
        "global_selection_proportion": float(row["global_selection_proportion"]),
        "coverage_by_direction": {
            "wide": float(row["wide_coverage"]),
            "body": float(row["body_coverage"]),
            "T": float(row["T_coverage"]),
        },
        "coverage_by_surface": {
            "Hard": float(row["hard_coverage"]),
            "Clay": float(row["clay_coverage"]),
            "Grass": float(row["grass_coverage"]),
        },
    }


def _comparison_summary(
    primary: pd.Series,
    control: pd.Series,
    primary_checks: Mapping[str, Any],
    control_checks: Mapping[str, Any],
) -> dict[str, Any]:
    columns = {
        "complete_match_coverage": "coverage_complete_match",
        "worst_fold_complete_match_coverage": "worst_fold_complete_match_coverage",
        "median_wilson_width": "median_wilson_width",
        "worst_fold_p90_wilson_width": "worst_fold_p90_wilson_width",
        "worst_fold_p90_absolute_change": "worst_fold_p90_absolute_change",
        "abstention_proportion": "abstention_proportion",
        "surface_selection_proportion": "surface_selection_proportion",
        "global_fallback_proportion": "global_selection_proportion",
    }
    differences = {
        name: float(primary[column]) - float(control[column])
        for name, column in columns.items()
    }
    return {
        "difference_definition": "primary_minus_global_control",
        "differences": differences,
        "coverage_by_direction": {
            direction: {
                "primary": float(primary[f"{direction}_coverage"]),
                "global_control": float(control[f"{direction}_coverage"]),
                "difference": float(primary[f"{direction}_coverage"] - control[f"{direction}_coverage"]),
            }
            for direction in DIRECTIONS
        },
        "coverage_by_surface": {
            surface: {
                "primary": float(primary[f"{surface.lower()}_coverage"]),
                "global_control": float(control[f"{surface.lower()}_coverage"]),
                "difference": float(
                    primary[f"{surface.lower()}_coverage"] - control[f"{surface.lower()}_coverage"]
                ),
            }
            for surface in SURFACES
        },
        "coverage_by_fold": {
            fold: {
                "primary": primary_checks["fold_contract"]["coverage_by_fold"][fold],
                "global_control": control_checks["fold_contract"]["coverage_by_fold"][fold],
                "difference": (
                    primary_checks["fold_contract"]["coverage_by_fold"][fold]
                    - control_checks["fold_contract"]["coverage_by_fold"][fold]
                ),
            }
            for fold in map(str, FOLDS)
        },
        "interpretation": (
            "La preferencia por surface_then_global procede de la regla metodologica congelada; "
            "no demuestra superioridad estadistica."
        ),
    }


def _selected_contract(selected_policy: str | None) -> dict[str, Any] | None:
    if selected_policy is None:
        return None
    surface = selected_policy == PRIMARY_POLICY
    return {
        "policy_id": selected_policy,
        "min_points_per_direction_and_role": MIN_POINTS,
        "min_matches_per_direction_and_role": MIN_MATCHES,
        "scope_policy": "surface_then_global" if surface else "global_only",
        "required_roles": list(ROLES),
        "required_directions": list(DIRECTIONS),
        "joint_role_eligibility": True,
        "surface_priority": surface,
        "joint_global_fallback": surface,
        "abstain_when_evidence_is_insufficient": True,
        "complete_match_requires_orientations": 2,
        "complete_match_requires_directions": 3,
        "outcome": "P(server_won_point)",
        "smoothing": False,
        "imputation": False,
        "mixed_scopes": False,
        "automatic_recommendation": False,
    }


def _publication_fingerprint(comparison: pd.DataFrame, summary_core: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_frame_bytes(comparison))
    digest.update(json.dumps(
        summary_core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    return digest.hexdigest().upper()


def build_selection(upstream: UpstreamArtifacts) -> EvidencePolicySelectionResult:
    audit = validate_upstream_artifacts(upstream)
    primary = upstream.candidates.loc[
        upstream.candidates["policy_id"].eq(PRIMARY_POLICY)
    ].iloc[0]
    control = upstream.candidates.loc[
        upstream.candidates["policy_id"].eq(GLOBAL_CONTROL)
    ].iloc[0]
    primary_checks = evaluate_candidate(primary, upstream.pareto, upstream.by_fold, upstream.summary)
    control_descriptive_checks = evaluate_candidate(
        control, upstream.pareto, upstream.by_fold, upstream.summary
    )
    decision = decide_selection(primary_checks, control_descriptive_checks)
    status = decision["selection_status"]
    path = decision["selection_path"]
    selected_policy = decision["selected_policy"]
    reason_codes = decision["reason_codes"]
    comparison = build_comparison(upstream.candidates, upstream.pareto, selected_policy)

    source_artifacts = [
        {
            "path": relative,
            "bytes": len(upstream.payloads[relative]),
            "sha256": upstream.hashes[relative],
        }
        for relative in UPSTREAM_HASHES
    ]
    primary_metrics = _policy_metrics(primary)
    control_metrics = _policy_metrics(control)
    rejected = [
        {
            "policy_id": str(row.policy_id),
            "selection_role": str(row.selection_role),
            "reason": str(row.rejection_reason),
        }
        for row in comparison.loc[~comparison["selected"]].itertuples(index=False)
    ]
    summary: dict[str, Any] = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "source_artifacts": source_artifacts,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_hashes": dict(UPSTREAM_HASHES),
        "sealed_test_contract": {
            "decision_data": "validation_2020_2023_only",
            "test_status": "sealed",
            "test_read": False,
            "test_rows_evaluated": 0,
            "test_matches_evaluated": 0,
            "test_evaluation_runs": 0,
            "used_for_policy_selection": False,
            "selected_by_test_performance": False,
        },
        "candidate_shortlist": list(SHORTLIST),
        "decision_rule": {
            "ordered_steps": [
                "pareto_eligibility", "structural_minimum", "absolute_limits",
                "fold_coherence", "scope_contract", "primary_then_global_control",
            ],
            "primary_candidate": PRIMARY_POLICY,
            "single_fallback": GLOBAL_CONTROL,
            "third_candidate_attempt_allowed": False,
            "limits_use_unrounded_floats": True,
        },
        "absolute_limits": {
            "minimum_worst_fold_complete_match_coverage": MIN_WORST_FOLD_COVERAGE,
            "maximum_worst_fold_p90_wilson_width": MAX_WORST_FOLD_WILSON,
            "maximum_worst_fold_p90_absolute_change": MAX_WORST_FOLD_STABILITY,
        },
        "primary_candidate": primary_metrics,
        "global_control": control_metrics,
        "candidate_checks": primary_checks,
        "control_descriptive_checks": control_descriptive_checks,
        "fallback_evaluation": {
            "evaluated": decision["fallback_evaluated"],
            "activated": decision["fallback_activated"],
            "control_checks": (
                control_descriptive_checks if decision["fallback_evaluated"] else None
            ),
        },
        "decision_execution": {
            key: decision[key]
            for key in (
                "primary_evaluated", "primary_passed", "control_descriptively_compared",
                "fallback_evaluated", "fallback_activated", "third_candidate_evaluated",
            )
        },
        "comparison": _comparison_summary(
            primary, control, primary_checks, control_descriptive_checks
        ),
        "selection_status": status,
        "selection_path": path,
        "selected_policy": selected_policy,
        "selected_policy_contract": _selected_contract(selected_policy),
        "rejected_alternatives": rejected,
        "reason_codes": reason_codes,
        "reconciliations": {
            "upstream_hashes_exact": True,
            "upstream_candidate_count_27": audit["candidate_count"] == 27,
            "pareto_8_non_dominated": audit["non_dominated_count"] == 8,
            "pareto_19_dominated": audit["dominated_count"] == 19,
            "shortlist_exact": True,
            "candidate_keys_unique": True,
            "pareto_axes_finite": True,
            "dominance_reconstructed": True,
            "required_policies_present": True,
            "folds_2020_2023_only": True,
            "test_remains_sealed": True,
            "comparison_canonical": True,
            "selection_cardinality_valid": int(comparison["selected"].sum()) in (0, 1),
            "selection_state_coherent": (
                (status == "not_available" and selected_policy is None)
                or (status != "not_available" and selected_policy is not None)
            ),
            "selected_contract_coherent": (
                selected_policy is None
                or _selected_contract(selected_policy)["policy_id"] == selected_policy
            ),
            "no_policy_winner_chosen_from_test": True,
        },
        "methodological_rationale": {
            "structural_minimum": (
                "50 puntos y 5 partidos por direccion y rol son una regla operativa "
                "conservadora, no suficiencia estadistica universal."
            ),
            "surface_context": (
                "Surface se prioriza solo con evidencia conjunta; global actua como fallback conjunto."
            ),
            "grass": "Grass es un estrato fragmentado y no modifica la politica a posteriori.",
        },
        "methodological_limits": [
            "Decision basada solo en validacion observacional 2020--2023.",
            "Los limites no son pruebas de significacion ni garantias causales.",
            "La cobertura historica es desigual, especialmente en Grass.",
            "La seleccion no genera recomendaciones tacticas automaticas.",
        ],
        "next_step_constraints": [
            "Mantener el test 2024--2026 sellado hasta cerrar metodologia y metricas.",
            "No cambiar thresholds, smoothing, fallback o features usando el test.",
            "No emitir recomendacion cuando la politica se abstiene.",
            "Interpretar resultados posteriores como asociaciones historicas, no efectos causales.",
        ],
    }
    summary = _json_value(summary)
    fingerprint = _publication_fingerprint(comparison, summary)
    summary["publication_fingerprint"] = fingerprint
    result = EvidencePolicySelectionResult(summary, comparison, fingerprint, upstream)
    validate_result(result)
    return result


def not_available_result(reason_codes: list[str]) -> EvidencePolicySelectionResult:
    comparison = pd.DataFrame(columns=COMPARISON_COLUMNS)
    summary: dict[str, Any] = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "source_artifacts": [],
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_hashes": dict(UPSTREAM_HASHES),
        "sealed_test_contract": {
            "decision_data": "validation_2020_2023_only",
            "test_status": "sealed",
            "test_read": False,
            "test_rows_evaluated": 0,
            "test_matches_evaluated": 0,
            "test_evaluation_runs": 0,
            "used_for_policy_selection": False,
            "selected_by_test_performance": False,
        },
        "candidate_shortlist": list(SHORTLIST),
        "decision_rule": None,
        "absolute_limits": None,
        "primary_candidate": None,
        "global_control": None,
        "candidate_checks": None,
        "control_descriptive_checks": None,
        "fallback_evaluation": {
            "evaluated": False,
            "activated": False,
            "control_checks": None,
        },
        "decision_execution": {
            "primary_evaluated": False,
            "primary_passed": False,
            "control_descriptively_compared": False,
            "fallback_evaluated": False,
            "fallback_activated": False,
            "third_candidate_evaluated": False,
        },
        "comparison": None,
        "selection_status": "not_available",
        "selection_path": NOT_AVAILABLE_PATH,
        "selected_policy": None,
        "selected_policy_contract": None,
        "rejected_alternatives": [],
        "reason_codes": sorted(set(reason_codes)),
        "reconciliations": {"upstream_artifacts_reliable": False},
        "methodological_rationale": None,
        "methodological_limits": ["No se publica una seleccion parcial."],
        "next_step_constraints": ["Corregir los artefactos upstream antes de decidir."],
    }
    fingerprint = _publication_fingerprint(comparison, summary)
    summary["publication_fingerprint"] = fingerprint
    result = EvidencePolicySelectionResult(summary, comparison, fingerprint, None)
    validate_result(result)
    return result


def _require_real_bool(value: Any, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} debe ser booleano real.")
    return bool(value)


def _parse_policy_contract(policy_id: str) -> tuple[int, int, str]:
    match = re.fullmatch(
        r"p(?P<points>\d{3})_m(?P<matches>\d{2})_"
        r"(?P<scope>global_only|surface_only|surface_then_global)",
        policy_id,
    )
    if match is None:
        raise ValueError(f"policy_id mal formado: {policy_id}")
    return int(match["points"]), int(match["matches"]), match["scope"]


def _canonical_rejection_reasons(
    row: pd.Series,
    role: str,
    selected_policy: str | None,
) -> str:
    reasons: list[str] = []
    if int(row["min_points"]) < MIN_POINTS:
        reasons.append("minimum_points_not_met")
    if int(row["min_matches"]) < MIN_MATCHES:
        reasons.append("minimum_matches_not_met")
    if float(row["worst_fold_complete_match_coverage"]) < MIN_WORST_FOLD_COVERAGE:
        reasons.append("coverage_limit_not_met")
    if float(row["worst_fold_p90_wilson_width"]) > MAX_WORST_FOLD_WILSON:
        reasons.append("wilson_limit_not_met")
    if float(row["worst_fold_p90_absolute_change"]) > MAX_WORST_FOLD_STABILITY:
        reasons.append("stability_limit_not_met")
    if role == "context_only":
        reasons.append("context_only_not_in_primary_fallback_path")
    elif row["policy_id"] != selected_policy:
        if role == "global_control" and selected_policy == PRIMARY_POLICY:
            reasons.append("primary_selected_fallback_not_activated")
        else:
            reasons.append("not_selected_by_frozen_decision_path")
    return "|".join(reasons)


def _canonical_comparison_for_validation(
    upstream: UpstreamArtifacts,
    selected_policy: str | None,
) -> pd.DataFrame:
    candidates = upstream.candidates.set_index("policy_id", drop=False)
    pareto = upstream.pareto.set_index("policy_id")
    rows: list[dict[str, Any]] = []
    for policy_id in SHORTLIST:
        row = candidates.loc[policy_id]
        points, matches, scope = _parse_policy_contract(policy_id)
        structural = points >= MIN_POINTS and matches >= MIN_MATCHES
        coverage = float(row["worst_fold_complete_match_coverage"]) >= MIN_WORST_FOLD_COVERAGE
        wilson = float(row["worst_fold_p90_wilson_width"]) <= MAX_WORST_FOLD_WILSON
        stability = float(row["worst_fold_p90_absolute_change"]) <= MAX_WORST_FOLD_STABILITY
        role = (
            "primary_candidate" if policy_id == PRIMARY_POLICY
            else "global_control" if policy_id == GLOBAL_CONTROL
            else "context_only"
        )
        rows.append({
            "policy_id": policy_id,
            "min_points": points,
            "min_matches": matches,
            "scope_policy": scope,
            "pareto_status": str(pareto.at[policy_id, "pareto_status"]),
            "passes_structural_minimum": structural,
            "passes_coverage_limit": coverage,
            "passes_wilson_limit": wilson,
            "passes_stability_limit": stability,
            "passes_all_absolute_limits": structural and coverage and wilson and stability,
            "worst_fold_complete_match_coverage": float(
                row["worst_fold_complete_match_coverage"]
            ),
            "worst_fold_wilson_p90": float(row["worst_fold_p90_wilson_width"]),
            "worst_fold_stability_p90": float(row["worst_fold_p90_absolute_change"]),
            "complete_match_coverage": float(row["coverage_complete_match"]),
            "median_wilson_width": float(row["median_wilson_width"]),
            "selection_role": role,
            "rejection_reason": _canonical_rejection_reasons(row, role, selected_policy),
            "selected": policy_id == selected_policy,
        })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _canonical_candidate_checks(
    upstream: UpstreamArtifacts,
    policy_id: str,
) -> dict[str, Any]:
    row = upstream.candidates.loc[upstream.candidates["policy_id"].eq(policy_id)].iloc[0]
    pareto_row = upstream.pareto.loc[upstream.pareto["policy_id"].eq(policy_id)].iloc[0]
    points, matches, scope = _parse_policy_contract(policy_id)
    if (
        int(row["min_points"]) != points
        or int(row["min_matches"]) != matches
        or str(row["scope_policy"]) != scope
    ):
        raise ValueError(f"Threshold o scope no coincide con policy_id: {policy_id}")
    structural = points >= MIN_POINTS and matches >= MIN_MATCHES
    coverage = float(row["worst_fold_complete_match_coverage"]) >= MIN_WORST_FOLD_COVERAGE
    wilson = float(row["worst_fold_p90_wilson_width"]) <= MAX_WORST_FOLD_WILSON
    stability = float(row["worst_fold_p90_absolute_change"]) <= MAX_WORST_FOLD_STABILITY
    pareto_eligible = (
        policy_id in SHORTLIST
        and pareto_row["pareto_status"] == "non_dominated"
        and (pd.isna(pareto_row["dominated_by"]) or pareto_row["dominated_by"] == "")
        and all(math.isfinite(float(pareto_row[axis])) for axis in PARETO_AXES)
    )
    fold = validate_fold_contract(row, upstream.by_fold)
    scope_contract = validate_scope_contract(row, upstream.summary, upstream.by_fold)
    expected = {
        "policy_id": policy_id,
        "pareto_eligible": pareto_eligible,
        "passes_structural_minimum": structural,
        "passes_coverage_limit": coverage,
        "passes_wilson_limit": wilson,
        "passes_stability_limit": stability,
        "passes_all_absolute_limits": structural and coverage and wilson and stability,
        "fold_contract": fold,
        "scope_contract": scope_contract,
    }
    expected["passes_all_selection_criteria"] = bool(
        pareto_eligible
        and expected["passes_all_absolute_limits"]
        and fold["valid"]
        and scope_contract["valid"]
    )
    expected["failed_criteria"] = [
        key for key in (
            "pareto_eligible", "passes_structural_minimum", "passes_coverage_limit",
            "passes_wilson_limit", "passes_stability_limit",
        )
        if not expected[key]
    ]
    return _json_value(expected)


def _validate_result_semantics(result: EvidencePolicySelectionResult) -> None:
    summary = result.summary
    upstream = result.upstream
    if upstream is None:
        raise ValueError("Un resultado seleccionado requiere upstream validado en memoria.")
    validate_upstream_artifacts(upstream)
    if tuple(summary.get("candidate_shortlist", ())) != SHORTLIST:
        raise ValueError("Shortlist del summary alterada.")
    if summary.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Commit upstream alterado.")
    if summary.get("upstream_hashes") != UPSTREAM_HASHES:
        raise ValueError("Hashes upstream del summary alterados.")
    if summary.get("analysis_name") != ANALYSIS_NAME or summary.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError("Identidad o version del analisis alterada.")
    expected_decision_rule = {
        "ordered_steps": [
            "pareto_eligibility", "structural_minimum", "absolute_limits",
            "fold_coherence", "scope_contract", "primary_then_global_control",
        ],
        "primary_candidate": PRIMARY_POLICY,
        "single_fallback": GLOBAL_CONTROL,
        "third_candidate_attempt_allowed": False,
        "limits_use_unrounded_floats": True,
    }
    if summary.get("decision_rule") != expected_decision_rule:
        raise ValueError("Regla u orden de decision alterados.")
    expected_limits = {
        "minimum_worst_fold_complete_match_coverage": MIN_WORST_FOLD_COVERAGE,
        "maximum_worst_fold_p90_wilson_width": MAX_WORST_FOLD_WILSON,
        "maximum_worst_fold_p90_absolute_change": MAX_WORST_FOLD_STABILITY,
    }
    if summary.get("absolute_limits") != expected_limits:
        raise ValueError("Limites absolutos alterados.")
    expected_sources = [
        {
            "path": relative,
            "bytes": len(upstream.payloads[relative]),
            "sha256": UPSTREAM_HASHES[relative],
        }
        for relative in UPSTREAM_HASHES
    ]
    if summary.get("source_artifacts") != expected_sources:
        raise ValueError("Fuentes upstream publicadas alteradas.")

    primary_checks = _canonical_candidate_checks(upstream, PRIMARY_POLICY)
    control_checks = _canonical_candidate_checks(upstream, GLOBAL_CONTROL)
    if summary.get("candidate_checks") != primary_checks:
        raise ValueError("Checks primary incompletos o incoherentes.")
    if summary.get("control_descriptive_checks") != control_checks:
        raise ValueError("Checks descriptivos del control incoherentes.")

    expected_decision = decide_selection(primary_checks, control_checks)
    status = summary.get("selection_status")
    path = summary.get("selection_path")
    selected_policy = summary.get("selected_policy")
    if (
        status != expected_decision["selection_status"]
        or path != expected_decision["selection_path"]
        or selected_policy != expected_decision["selected_policy"]
        or summary.get("reason_codes") != expected_decision["reason_codes"]
    ):
        raise ValueError("Estado, path, policy o reason codes no reconcilian con la decision.")
    expected_execution = {
        key: expected_decision[key]
        for key in (
            "primary_evaluated", "primary_passed", "control_descriptively_compared",
            "fallback_evaluated", "fallback_activated", "third_candidate_evaluated",
        )
    }
    if summary.get("decision_execution") != expected_execution:
        raise ValueError("Ejecucion primary/fallback incoherente.")
    expected_fallback = {
        "evaluated": expected_decision["fallback_evaluated"],
        "activated": expected_decision["fallback_activated"],
        "control_checks": control_checks if expected_decision["fallback_evaluated"] else None,
    }
    if summary.get("fallback_evaluation") != expected_fallback:
        raise ValueError("Contrato de fallback incoherente.")

    expected_comparison = _canonical_comparison_for_validation(upstream, selected_policy)
    try:
        pd.testing.assert_frame_equal(
            result.comparison,
            expected_comparison,
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(f"Comparison no es canonico: {exc}") from exc

    expected_contract = _selected_contract(selected_policy)
    if summary.get("selected_policy_contract") != expected_contract:
        raise ValueError("Contrato seleccionado parcial o alterado.")
    expected_rejected = [
        {
            "policy_id": str(row.policy_id),
            "selection_role": str(row.selection_role),
            "reason": str(row.rejection_reason),
        }
        for row in expected_comparison.loc[~expected_comparison["selected"]].itertuples(index=False)
    ]
    if summary.get("rejected_alternatives") != expected_rejected:
        raise ValueError("Alternativas rechazadas no reconcilian con comparison.")
    primary = upstream.candidates.loc[upstream.candidates["policy_id"].eq(PRIMARY_POLICY)].iloc[0]
    control = upstream.candidates.loc[upstream.candidates["policy_id"].eq(GLOBAL_CONTROL)].iloc[0]
    if summary.get("primary_candidate") != _json_value(_policy_metrics(primary)):
        raise ValueError("Metricas primary alteradas.")
    if summary.get("global_control") != _json_value(_policy_metrics(control)):
        raise ValueError("Metricas del control alteradas.")
    expected_comparison_summary = _json_value(
        _comparison_summary(primary, control, primary_checks, control_checks)
    )
    if summary.get("comparison") != expected_comparison_summary:
        raise ValueError("Resumen comparativo alterado.")


def validate_result(result: EvidencePolicySelectionResult) -> None:
    if result.comparison.columns.tolist() != COMPARISON_COLUMNS:
        raise ValueError("Esquema de comparison no contractual.")
    summary = result.summary
    status = summary.get("selection_status")
    if status not in {"selected_primary", "selected_global_fallback", "not_available"}:
        raise ValueError("selection_status inesperado.")
    sealed = summary.get("sealed_test_contract", {})
    expected_sealed = {
        "decision_data": "validation_2020_2023_only",
        "test_status": "sealed",
        "test_read": False,
        "test_rows_evaluated": 0,
        "test_matches_evaluated": 0,
        "test_evaluation_runs": 0,
        "used_for_policy_selection": False,
        "selected_by_test_performance": False,
    }
    if sealed != expected_sealed:
        raise ValueError("Contrato de test sellado alterado.")
    path_by_status = {
        "selected_primary": "primary_candidate_selected",
        "selected_global_fallback": "fallback_to_global_control",
        "not_available": NOT_AVAILABLE_PATH,
    }
    if summary.get("selection_path") != path_by_status[status]:
        raise ValueError("selection_path no corresponde al estado.")
    boolean_columns = [
        "passes_structural_minimum", "passes_coverage_limit", "passes_wilson_limit",
        "passes_stability_limit", "passes_all_absolute_limits", "selected",
    ]
    for column in boolean_columns:
        if not all(isinstance(value, (bool, np.bool_)) for value in result.comparison[column]):
            raise ValueError(f"{column} debe contener booleanos reales.")
    selected = (
        result.comparison.loc[result.comparison["selected"]]
        if not result.comparison.empty else result.comparison
    )
    selected_policy = summary.get("selected_policy")
    reason_codes = summary.get("reason_codes")
    if not isinstance(reason_codes, list) or len(reason_codes) != len(set(reason_codes)):
        raise ValueError("Reason codes deben ser una lista unica.")
    if not set(reason_codes).issubset(FAILURE_REASON_CODES):
        raise ValueError("Reason code no contractual.")
    if status == "not_available":
        if selected_policy is not None or len(selected) != 0:
            raise ValueError("not_available no puede publicar seleccion parcial.")
        if not reason_codes:
            raise ValueError("not_available requiere reason codes concretos.")
        if result.upstream is None:
            if not result.comparison.empty:
                raise ValueError("Un fallo upstream no puede publicar comparison parcial.")
        else:
            if len(result.comparison) != len(SHORTLIST):
                raise ValueError("La decision no disponible requiere la shortlist completa.")
            _validate_result_semantics(result)
    else:
        if len(result.comparison) != 8 or len(selected) != 1:
            raise ValueError("Debe existir exactamente una politica seleccionada.")
        if selected.iloc[0]["policy_id"] != selected_policy:
            raise ValueError("Summary y comparison discrepan sobre la seleccion.")
        if status == "selected_primary" and selected_policy != PRIMARY_POLICY:
            raise ValueError("Estado primary incoherente.")
        if status == "selected_global_fallback" and selected_policy != GLOBAL_CONTROL:
            raise ValueError("Estado fallback incoherente.")
        contract = summary.get("selected_policy_contract")
        if not isinstance(contract, dict) or contract.get("policy_id") != selected_policy:
            raise ValueError("Contrato de politica incoherente.")
        if status == "selected_primary" and reason_codes != []:
            raise ValueError("Un estado exitoso no admite reason codes de fallo.")
        if status == "selected_global_fallback" and not reason_codes:
            raise ValueError("El fallback requiere los fallos concretos de primary.")
        _validate_result_semantics(result)
        reconciliations = summary.get("reconciliations", {})
        if set(reconciliations) != RECONCILIATION_KEYS:
            raise ValueError("Claves de reconciliacion ausentes o desconocidas.")
        for key, value in reconciliations.items():
            if not _require_real_bool(value, f"reconciliations.{key}") or value is not True:
                raise ValueError(f"Reconciliacion falsa: {key}")
    if result.upstream is not None and status == "not_available":
        reconciliations = summary.get("reconciliations", {})
        if set(reconciliations) != RECONCILIATION_KEYS:
            raise ValueError("Claves de reconciliacion ausentes o desconocidas.")
        for key, value in reconciliations.items():
            if not _require_real_bool(value, f"reconciliations.{key}") or value is not True:
                raise ValueError(f"Reconciliacion falsa: {key}")
    numeric = result.comparison.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    if np.isinf(numeric).any() or np.isnan(numeric).any():
        raise ValueError("Comparison contiene NaN o infinito.")
    _json_value(summary)
    core = {key: value for key, value in summary.items() if key != "publication_fingerprint"}
    expected = _publication_fingerprint(result.comparison, core)
    if result.publication_fingerprint != expected or summary.get("publication_fingerprint") != expected:
        raise ValueError("Fingerprint de seleccion alterado.")
    if any("C:/" in str(value) or "C:\\" in str(value) for value in summary.get("source_artifacts", [])):
        raise ValueError("El resumen contiene rutas absolutas.")


def serialize_artifacts(result: EvidencePolicySelectionResult) -> tuple[bytes, bytes]:
    validate_result(result)
    summary = (
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    comparison = _frame_bytes(result.comparison)
    loaded = pd.read_csv(io.BytesIO(comparison))
    if loaded.columns.tolist() != COMPARISON_COLUMNS or any(
        column.startswith("Unnamed") for column in loaded.columns
    ):
        raise ValueError("CSV serializado con esquema o indice accidental.")
    return summary, comparison


def _stage_payload(path: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return staged


def write_artifacts(
    result: EvidencePolicySelectionResult,
    summary_path: Path = SUMMARY_PATH,
    comparison_path: Path = COMPARISON_PATH,
) -> None:
    paths = (summary_path, comparison_path)
    if summary_path.resolve() == comparison_path.resolve():
        raise ValueError("Las rutas de salida deben ser distintas.")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    payloads = serialize_artifacts(result)
    originals = {path: path.read_bytes() if path.exists() else None for path in paths}
    staged: list[Path] = []
    replaced: list[Path] = []
    try:
        staged = [_stage_payload(path, payload) for path, payload in zip(paths, payloads)]
        for path, temporary in zip(paths, staged):
            os.replace(temporary, path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            previous = originals[path]
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                rollback = _stage_payload(path, previous)
                os.replace(rollback, path)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def remove_reproduction_directory(path: Path = REPRO_DIR) -> None:
    if path.resolve() != REPRO_DIR.resolve():
        raise ValueError("Solo puede eliminarse el temporal contractual.")
    if path.exists():
        shutil.rmtree(path)


def run_selection_once() -> EvidencePolicySelectionResult:
    try:
        upstream = load_upstream_artifacts()
        return build_selection(upstream)
    except UpstreamContractError as exc:
        return not_available_result([exc.reason_code])


def main() -> None:
    if REPRO_DIR.exists():
        raise FileExistsError(f"El temporal ya existe: {REPRO_DIR}")
    result = run_selection_once()
    write_artifacts(result)
    permanent_payloads = tuple(path.read_bytes() for path in (SUMMARY_PATH, COMPARISON_PATH))
    if permanent_payloads != serialize_artifacts(result):
        raise ValueError("Los permanentes no coinciden con el resultado en memoria.")
    REPRO_DIR.mkdir(parents=False)
    reproduced = (REPRO_DIR / SUMMARY_PATH.name, REPRO_DIR / COMPARISON_PATH.name)
    try:
        write_artifacts(result, *reproduced)
        if tuple(path.read_bytes() for path in reproduced) != permanent_payloads:
            raise ValueError("La reserializacion no es identica byte a byte.")
    finally:
        remove_reproduction_directory(REPRO_DIR)
    if REPRO_DIR.exists():
        raise ValueError("El temporal no fue eliminado.")
    print(json.dumps({
        "selection_status": result.summary["selection_status"],
        "selected_policy": result.summary["selected_policy"],
        "summary": {"bytes": SUMMARY_PATH.stat().st_size, "sha256": _sha256(SUMMARY_PATH.read_bytes())},
        "comparison": {"bytes": COMPARISON_PATH.stat().st_size, "sha256": _sha256(COMPARISON_PATH.read_bytes())},
        "reproduction_directory_exists": REPRO_DIR.exists(),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
