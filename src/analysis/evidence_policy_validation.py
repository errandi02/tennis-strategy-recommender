"""Evaluacion cronologica descriptiva de politicas combinadas de evidencia.

El analisis compara un grid preespecificado sobre los folds de validacion
2020--2023. El test posterior a 2023 permanece sellado: no se calculan ni se
publican metricas sobre el mismo y no se elige automaticamente una politica.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.chronological_validation import (
    EXPECTED_MATCHES,
    EXPECTED_SOURCE_ROWS,
    VALIDATION_END,
    VALIDATION_START,
    VALIDATION_YEARS,
)
from src.analysis.historical_profiles import (
    POINTS_FILE,
    SOURCE_COLUMNS,
    build_historical_snapshots,
    prepare_historical_population,
    validate_snapshots,
)
from src.analysis.player_opponent_direction_features import (
    DIRECTIONS,
    FEATURE_COLUMNS,
    Z_975,
    build_direction_features,
    validate_feature_table,
)
from src.analysis.second_serve_direction_analysis import SURFACE_ORDER
from src.parsing.serve_sequence import parse_sequence


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "evidence_policy_validation_summary.json"
CANDIDATES_PATH = TABLES_DIR / "evidence_policy_validation_candidates.csv"
BY_FOLD_PATH = TABLES_DIR / "evidence_policy_validation_by_fold.csv"
PARETO_PATH = TABLES_DIR / "evidence_policy_validation_pareto.csv"
REPRO_DIR = REPORTS_DIR / "repro_evidence_policy_validation"

ANALYSIS_NAME = "evidence_policy_chronological_validation"
ANALYSIS_VERSION = "1.1.0"
UPSTREAM_COMMITS = {
    "historical_profiles": "68a30c1",
    "player_opponent_direction_features": "a1a41aa",
    "chronological_validation": "3ac28b4",
}
POINT_THRESHOLDS = (25, 50, 100)
MATCH_THRESHOLDS = (3, 5, 10)
SCOPE_POLICIES = ("global_only", "surface_only", "surface_then_global")
ROLES = ("server", "opponent")
FOLDS = (2020, 2021, 2022, 2023)
SELECTED_SCOPES = ("global", "surface")
QUANTILE_METHOD = "linear"
TEST_EVALUATION_RUNS = 0

POLICY_COLUMNS = ["policy_id", "min_points", "min_matches", "scope_policy"]
CANDIDATE_COLUMNS = [
    *POLICY_COLUMNS,
    "validation_matches", "orientation_direction_denominator",
    "eligible_orientation_directions", "coverage_orientation",
    "eligible_matches", "coverage_match", "complete_matches",
    "coverage_complete_match", "worst_fold_complete_match_coverage",
    "complete_match_coverage_range", "wide_coverage", "body_coverage", "T_coverage",
    "hard_coverage", "clay_coverage", "grass_coverage",
    "median_wilson_width", "p90_wilson_width", "worst_fold_p90_wilson_width",
    "median_absolute_change", "p90_absolute_change",
    "worst_fold_p90_absolute_change", "within_005_proportion",
    "within_010_proportion", "valid_stability_comparisons",
    "surface_selection_proportion", "global_selection_proportion",
    "abstention_proportion", "scope_coherence_reconciled",
]
BY_FOLD_COLUMNS = [
    *POLICY_COLUMNS, "fold", "direction", "role", "stratum_type",
    "surface", "selected_scope", "target_matches", "target_orientations",
    "eligible_orientations", "eligible_matches", "complete_matches",
    "abstained_orientations", "abstained_matches", "eligible_players",
    "coverage_orientation", "coverage_match", "coverage_complete_match",
    "surface_selections", "global_selections", "abstentions",
    "surface_selection_proportion", "global_selection_proportion",
    "abstention_proportion", "wilson_width_mean", "wilson_width_median",
    "wilson_width_p75", "wilson_width_p90", "wilson_width_p95",
    "wilson_width_max", "stability_transition", "comparable_rows",
    "comparable_players", "absolute_change_mean", "absolute_change_median",
    "absolute_change_p75", "absolute_change_p90", "spearman_correlation",
    "spearman_reason_code", "within_005_proportion", "within_010_proportion",
]
PARETO_COLUMNS = [
    *POLICY_COLUMNS, "pareto_status", "dominated_by", "reason_code",
    "worst_fold_complete_match_coverage", "worst_fold_p90_wilson_width",
    "worst_fold_p90_absolute_change",
]


@dataclass(frozen=True)
class EvidencePolicyValidationResult:
    candidates: pd.DataFrame
    by_fold: pd.DataFrame
    pareto: pd.DataFrame
    summary: dict[str, Any]
    publication_fingerprint: str
    performance_metrics: tuple[dict[str, Any], ...] = ()


class PerformanceRecorder:
    """Instrumentacion efimera que nunca se serializa."""

    def __init__(self, callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.callback = callback
        self.records: list[dict[str, Any]] = []

    def record(self, phase: str, started: float, rows: int | None = None) -> None:
        record = {
            "phase": phase,
            "duration_seconds": time.perf_counter() - started,
            "rows": None if rows is None else int(rows),
        }
        self.records.append(record)
        if self.callback is not None:
            self.callback(record.copy())

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.copy() for item in self.records)


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _assert_finite_json(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Valor no finito en {path}.")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"Tipo no serializable en {path}: {type(value).__name__}")


def build_candidate_grid() -> pd.DataFrame:
    """Devuelve exactamente el producto cartesiano preespecificado."""

    rows = []
    for min_points in POINT_THRESHOLDS:
        for min_matches in MATCH_THRESHOLDS:
            for scope_policy in SCOPE_POLICIES:
                rows.append(
                    {
                        "policy_id": f"p{min_points:03d}_m{min_matches:02d}_{scope_policy}",
                        "min_points": min_points,
                        "min_matches": min_matches,
                        "scope_policy": scope_policy,
                    }
                )
    grid = pd.DataFrame(rows, columns=POLICY_COLUMNS)
    if len(grid) != 27 or grid["policy_id"].duplicated().any():
        raise ValueError("El grid no contiene 27 politicas unicas.")
    return grid


def _normalized_feature_dates(features: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(features["target_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Las features contienen fechas no validas.")
    normalized = dates.dt.normalize()
    if not dates.eq(normalized).all():
        raise ValueError("Las fechas objetivo deben estar normalizadas al dia civil.")
    return normalized


def select_validation_features(
    features: pd.DataFrame,
    test_match_ids: Iterable[str] = (),
    *,
    evaluation_runs: int = TEST_EVALUATION_RUNS,
) -> pd.DataFrame:
    """Aplica la barrera sellada y devuelve solo objetivos 2020--2023."""

    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError("El esquema de features no coincide con el contrato upstream.")
    if evaluation_runs != 0:
        raise ValueError("test_evaluation_runs debe permanecer en cero.")
    dates = _normalized_feature_dates(features)
    after_cutoff = dates.gt(VALIDATION_END)
    if after_cutoff.any():
        ids = sorted(features.loc[after_cutoff, "target_match_id"].astype(str).unique().tolist())
        raise ValueError(
            f"Targets posteriores a 2023 llegaron a la frontera de validacion: {ids[:5]}"
        )
    forbidden = set(test_match_ids)
    contamination = sorted(set(features["target_match_id"]) & forbidden)
    if contamination:
        raise ValueError(f"Contaminacion con match_id de test: {contamination[:5]}")
    validation = features.loc[dates.ge(VALIDATION_START)].copy(deep=True)
    validation_dates = _normalized_feature_dates(validation)
    if validation.empty or validation_dates.max() > VALIDATION_END:
        raise ValueError("La ventana de validacion no respeta el limite 2023-12-31.")
    years = tuple(sorted(validation_dates.dt.year.unique().tolist()))
    if years != FOLDS:
        raise ValueError(f"Folds de validacion inesperados: {years}")
    validation["fold"] = validation_dates.dt.year.astype("int64")
    if not validation["fold"].isin(FOLDS).all():
        raise ValueError("Una fila no pertenece a los folds permitidos.")
    key = ["target_match_id", "target_player", "direction", "scope"]
    if validation.duplicated(key).any():
        raise ValueError("La clave de features de validacion no es unica.")
    if not validation.groupby(["target_match_id", "direction", "scope"]).size().eq(2).all():
        raise ValueError("Cada partido, direccion y scope debe tener dos orientaciones.")
    return validation.sort_values(
        ["target_date", "target_match_id", "target_player", "direction", "scope"],
        kind="stable",
    ).reset_index(drop=True)


def seal_target_population(
    targets: pd.DataFrame,
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Excluye fisicamente targets e historia posteriores a 2023 antes de construir."""

    target_dates = pd.to_datetime(targets["date"], errors="coerce").dt.normalize()
    eligible_dates = pd.to_datetime(eligible["date"], errors="coerce").dt.normalize()
    if target_dates.isna().any() or eligible_dates.isna().any():
        raise ValueError("La poblacion preparada contiene fechas no validas.")
    allowed_targets = targets.loc[target_dates.le(VALIDATION_END)].copy(deep=True)
    excluded_targets = targets.loc[target_dates.gt(VALIDATION_END)]
    allowed_history = eligible.loc[eligible_dates.le(VALIDATION_END)].copy(deep=True)
    test_ids = set(excluded_targets["match_id"])
    if set(allowed_targets["match_id"]) & test_ids:
        raise ValueError("Un match_id de test permanece entre los targets construibles.")
    if allowed_targets["date"].max() > VALIDATION_END:
        raise ValueError("El sellado fisico no excluyo todos los targets posteriores a 2023.")
    if allowed_history["date"].max() > VALIDATION_END:
        raise ValueError("La historia contiene eventos posteriores al ultimo target permitido.")
    audit = {
        "source_target_matches": int(len(targets)),
        "constructed_target_matches": int(len(allowed_targets)),
        "test_target_matches_excluded_before_construction": int(len(excluded_targets)),
        "test_target_matches_constructed": 0,
        "test_target_rows_constructed": 0,
        "test_feature_rows_constructed": 0,
    }
    if audit["constructed_target_matches"] + audit[
        "test_target_matches_excluded_before_construction"
    ] != audit["source_target_matches"]:
        raise ValueError("Los targets permitidos y excluidos no reconcilian.")
    return (
        allowed_targets.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True),
        allowed_history.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True),
        audit,
    )


def construct_sealed_target_features(
    targets: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    recorder: PerformanceRecorder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], set[str]]:
    """Construye snapshots/features solo despues de aplicar el sellado fisico."""

    recorder = recorder or PerformanceRecorder()
    started = time.perf_counter()
    constructed_targets, allowed_history, audit = seal_target_population(targets, eligible)
    recorder.record("seal_targets_before_construction", started, len(constructed_targets))
    started = time.perf_counter()
    snapshots = build_historical_snapshots(constructed_targets, allowed_history)
    validate_snapshots(snapshots, constructed_targets)
    recorder.record("snapshots", started, len(snapshots))
    started = time.perf_counter()
    features = build_direction_features(snapshots, allowed_history)
    validate_feature_table(features, snapshots)
    recorder.record("features", started, len(features))
    audit.update(
        {
            "constructed_snapshot_rows": int(len(snapshots)),
            "constructed_feature_rows": int(len(features)),
        }
    )
    test_ids = set(targets.loc[targets["date"].gt(VALIDATION_END), "match_id"])
    return snapshots, features, audit, test_ids


def prepare_policy_population(validation_features: pd.DataFrame) -> pd.DataFrame:
    """Une global y surface sin mezclar roles ni recalcular historia."""

    identifiers = [
        "target_match_id", "target_date", "target_player", "opponent",
        "target_surface", "derived_period", "direction", "fold",
    ]
    metrics = [
        "server_direction_matches", "server_direction_points",
        "server_direction_raw_rate", "server_direction_wilson_low",
        "server_direction_wilson_high", "opponent_direction_matches",
        "opponent_direction_points", "opponent_allowed_server_raw_rate",
        "opponent_allowed_server_wilson_low", "opponent_allowed_server_wilson_high",
        "server_direction_last_history_date", "opponent_direction_last_history_date",
    ]
    global_rows = validation_features.loc[
        validation_features["scope"].eq("global"), identifiers + metrics
    ]
    surface_rows = validation_features.loc[
        validation_features["scope"].eq("surface"), identifiers + metrics
    ]
    merged = global_rows.merge(
        surface_rows,
        on=identifiers,
        suffixes=("_global", "_surface"),
        validate="one_to_one",
        sort=False,
    )
    if len(merged) * 2 != len(validation_features):
        raise ValueError("Global y surface no forman pares exactos.")
    for suffix in ("global", "surface"):
        for role in ROLES:
            last = merged[f"{role}_direction_last_history_date_{suffix}"]
            if (last.notna() & last.ge(merged["target_date"])).any():
                raise ValueError("La historia no es estrictamente anterior al partido objetivo.")
    direction_order = {value: index for index, value in enumerate(DIRECTIONS)}
    merged["_direction_order"] = merged["direction"].map(direction_order)
    merged = merged.sort_values(
        ["target_date", "target_match_id", "target_player", "_direction_order"],
        kind="stable",
    ).drop(columns="_direction_order").reset_index(drop=True)
    return merged


def apply_policy(population: pd.DataFrame, policy: pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Aplica una politica preespecificada de forma vectorizada."""

    min_points = int(policy["min_points"])
    min_matches = int(policy["min_matches"])
    scope_policy = str(policy["scope_policy"])
    if min_points not in POINT_THRESHOLDS or min_matches not in MATCH_THRESHOLDS:
        raise ValueError("Threshold fuera del grid preespecificado.")
    if scope_policy not in SCOPE_POLICIES:
        raise ValueError("Politica de scope inesperada.")

    def scope_passes(scope: str) -> pd.Series:
        return (
            population[f"server_direction_points_{scope}"].ge(min_points)
            & population[f"server_direction_matches_{scope}"].ge(min_matches)
            & population[f"opponent_direction_points_{scope}"].ge(min_points)
            & population[f"opponent_direction_matches_{scope}"].ge(min_matches)
        )
    if scope_policy == "global_only":
        selected = np.where(scope_passes("global"), "global", None)
    elif scope_policy == "surface_only":
        selected = np.where(scope_passes("surface"), "surface", None)
    else:
        surface_passes = scope_passes("surface")
        global_passes = scope_passes("global")
        selected = np.select(
            [surface_passes, ~surface_passes & global_passes],
            ["surface", "global"],
            default=None,
        )
    result = population.copy(deep=False).assign(selected_scope=selected)
    result["eligible"] = result["selected_scope"].notna()
    for role in ROLES:
        rate_name = "server_direction_raw_rate" if role == "server" else "opponent_allowed_server_raw_rate"
        low_name = "server_direction_wilson_low" if role == "server" else "opponent_allowed_server_wilson_low"
        high_name = "server_direction_wilson_high" if role == "server" else "opponent_allowed_server_wilson_high"
        if scope_policy == "global_only":
            result[f"{role}_selected_rate"] = result[f"{rate_name}_global"].where(
                result["eligible"]
            )
            result[f"{role}_selected_wilson_width"] = (
                result[f"{high_name}_global"] - result[f"{low_name}_global"]
            ).where(result["eligible"])
        elif scope_policy == "surface_only":
            result[f"{role}_selected_rate"] = result[f"{rate_name}_surface"].where(
                result["eligible"]
            )
            result[f"{role}_selected_wilson_width"] = (
                result[f"{high_name}_surface"] - result[f"{low_name}_surface"]
            ).where(result["eligible"])
        else:
            result[f"{role}_selected_rate"] = np.select(
                [result["selected_scope"].eq("surface"), result["selected_scope"].eq("global")],
                [result[f"{rate_name}_surface"], result[f"{rate_name}_global"]],
                default=np.nan,
            )
            result[f"{role}_selected_wilson_width"] = np.select(
                [result["selected_scope"].eq("surface"), result["selected_scope"].eq("global")],
                [result[f"{high_name}_surface"] - result[f"{low_name}_surface"],
                 result[f"{high_name}_global"] - result[f"{low_name}_global"]],
                default=np.nan,
            )
    result.insert(0, "policy_id", str(policy["policy_id"]))
    result.insert(1, "min_points", min_points)
    result.insert(2, "min_matches", min_matches)
    result.insert(3, "scope_policy", scope_policy)
    return result


def _quantile(values: pd.Series | np.ndarray, probability: float) -> float | None:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return float(np.quantile(array, probability, method=QUANTILE_METHOD))


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _precision(values: pd.Series) -> dict[str, float | None]:
    finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
    return {
        "wilson_width_mean": None if finite.empty else float(finite.mean()),
        "wilson_width_median": _quantile(finite, .5),
        "wilson_width_p75": _quantile(finite, .75),
        "wilson_width_p90": _quantile(finite, .9),
        "wilson_width_p95": _quantile(finite, .95),
        "wilson_width_max": None if finite.empty else float(finite.max()),
    }


def independent_wilson_width(wins: int, points: int) -> float | None:
    """Comprobacion escalar independiente del helper upstream."""

    if points == 0:
        return None
    proportion = wins / points
    denominator = 1.0 + Z_975**2 / points
    centre = (proportion + Z_975**2 / (2.0 * points)) / denominator
    half = Z_975 * np.sqrt(
        proportion * (1.0 - proportion) / points + Z_975**2 / (4.0 * points**2)
    ) / denominator
    return float(min(1.0, centre + half) - max(0.0, centre - half))


def _match_flags(evaluated: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    directional = evaluated.groupby(
        ["fold", "target_match_id", "direction"], sort=False, observed=True
    )["eligible"].agg(["size", "sum"])
    if not directional["size"].eq(2).all():
        raise ValueError("Cada direccion debe contener exactamente dos orientaciones.")
    directional_complete = directional["sum"].eq(2)
    complete = directional_complete.groupby(level=[0, 1], sort=False).agg(["size", "sum"])
    if not complete["size"].eq(3).all():
        raise ValueError("Cada partido debe contener exactamente tres direcciones.")
    complete_match = complete["sum"].eq(3)
    return directional_complete, complete_match


def _empty_stability(current_fold: int) -> dict[str, Any]:
    return {
        "stability_transition": (
            None if current_fold == 2020 else f"{current_fold - 1}_to_{current_fold}"
        ),
        "comparable_rows": 0,
        "comparable_players": 0,
        "absolute_change_mean": None,
        "absolute_change_median": None,
        "absolute_change_p75": None,
        "absolute_change_p90": None,
        "spearman_correlation": None,
        "spearman_reason_code": (
            "no_previous_fold" if current_fold == 2020 else "no_comparable_players"
        ),
        "within_005_proportion": None,
        "within_010_proportion": None,
    }


def _safe_spearman(
    server_series: pd.Series,
    opponent_series: pd.Series,
) -> tuple[float | None, str | None, np.ndarray]:
    """Calcula Spearman solo cuando hay pares finitos y variacion suficiente."""

    server = server_series.to_numpy(dtype="float64")
    opponent = opponent_series.to_numpy(dtype="float64")
    finite = np.isfinite(server) & np.isfinite(opponent)
    server = server[finite]
    opponent = opponent[finite]
    if len(server) < 2:
        return None, "insufficient_comparable_pairs", finite
    server_constant = np.unique(server).size < 2
    opponent_constant = np.unique(opponent).size < 2
    if server_constant and opponent_constant:
        return None, "both_series_constant", finite
    if server_constant:
        return None, "constant_server_series", finite
    if opponent_constant:
        return None, "constant_opponent_series", finite
    correlation = pd.Series(server).corr(pd.Series(opponent), method="spearman")
    if not np.isfinite(correlation):
        raise ValueError("Spearman no finito pese a superar los diagnosticos previos.")
    return float(correlation), None, finite


def _build_stability_cache(evaluated: pd.DataFrame) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Agrega jugadores una vez y reutiliza las transiciones en todos los estratos."""

    eligible = evaluated.loc[evaluated["eligible"]].copy()
    expanded: list[pd.DataFrame] = []
    for stratum_type in ("overall", "surface", "selected_scope", "surface_selected_scope"):
        part = eligible[
            [
                "fold", "direction", "target_player", "target_surface",
                "selected_scope", "server_selected_rate", "opponent_selected_rate",
            ]
        ].copy()
        part["stratum_type"] = stratum_type
        if stratum_type in ("overall", "selected_scope"):
            part["surface"] = "<ALL>"
        else:
            part["surface"] = part["target_surface"]
        if stratum_type in ("overall", "surface"):
            part["scope_label"] = "<ALL>"
        else:
            part["scope_label"] = part["selected_scope"]
        expanded.append(part)
    long = pd.concat(expanded, ignore_index=True)
    role_frames = []
    for role in ROLES:
        part = long.rename(columns={f"{role}_selected_rate": "rate"}).copy(deep=False)
        part["role"] = role
        role_frames.append(part[
            ["fold", "direction", "role", "stratum_type", "surface", "scope_label", "target_player", "rate"]
        ])
    role_long = pd.concat(role_frames, ignore_index=True)
    player_fold = (
        role_long.groupby(
            ["fold", "direction", "role", "stratum_type", "surface", "scope_label", "target_player"],
            sort=False,
            observed=True,
        )["rate"]
        .agg([("rate", "mean"), ("rows", "size")])
        .reset_index()
    )
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    group_columns = ["direction", "role", "stratum_type", "surface", "scope_label"]
    for group_key, group in player_fold.groupby(group_columns, sort=False, observed=True):
        by_fold = {
            int(fold): frame[["target_player", "rate", "rows"]]
            for fold, frame in group.groupby("fold", sort=False, observed=True)
        }
        for current_fold in FOLDS:
            key = (current_fold, *group_key)
            empty = _empty_stability(current_fold)
            if current_fold == 2020:
                cache[key] = empty
                continue
            previous = by_fold.get(current_fold - 1)
            current = by_fold.get(current_fold)
            if previous is None or current is None:
                cache[key] = empty
                continue
            joined = previous.merge(
                current, on="target_player", suffixes=("_previous", "_current"),
                validate="one_to_one",
            )
            finite = np.isfinite(joined["rate_previous"].to_numpy(dtype="float64")) & np.isfinite(
                joined["rate_current"].to_numpy(dtype="float64")
            )
            joined = joined.loc[finite].reset_index(drop=True)
            if joined.empty:
                cache[key] = empty
                continue
            changes = (joined["rate_current"] - joined["rate_previous"]).abs()
            correlation, reason, _ = _safe_spearman(
                joined["rate_previous"], joined["rate_current"]
            )
            cache[key] = {
                "stability_transition": f"{current_fold - 1}_to_{current_fold}",
                "comparable_rows": int(joined["rows_previous"].sum() + joined["rows_current"].sum()),
                "comparable_players": int(len(joined)),
                "absolute_change_mean": float(changes.mean()),
                "absolute_change_median": _quantile(changes, .5),
                "absolute_change_p75": _quantile(changes, .75),
                "absolute_change_p90": _quantile(changes, .9),
                "spearman_correlation": None if correlation is None else float(correlation),
                "spearman_reason_code": reason,
                "within_005_proportion": float(changes.le(.05).mean()),
                "within_010_proportion": float(changes.le(.10).mean()),
            }
    return cache


def build_by_fold(evaluated_policies: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata = [("overall", None, None)]
    strata.extend(("surface", surface, None) for surface in SURFACE_ORDER)
    strata.extend(("selected_scope", None, scope) for scope in SELECTED_SCOPES)
    strata.extend(
        ("surface_selected_scope", surface, scope)
        for surface in SURFACE_ORDER for scope in SELECTED_SCOPES
    )
    for evaluated in evaluated_policies:
        policy = evaluated.iloc[0]
        directional_complete, complete_match = _match_flags(evaluated)
        stability_cache = _build_stability_cache(evaluated)
        directional_cache = {
            (int(fold), direction): frame
            for (fold, direction), frame in evaluated.groupby(
                ["fold", "direction"], sort=False, observed=True
            )
        }
        for fold in FOLDS:
            fold_data = evaluated.loc[evaluated["fold"].eq(fold)]
            target_matches = fold_data["target_match_id"].nunique()
            for direction in DIRECTIONS:
                directional = directional_cache[(fold, direction)]
                for role in ROLES:
                    for stratum_type, surface, selected_scope in strata:
                        base_mask = pd.Series(True, index=directional.index)
                        if surface is not None:
                            base_mask &= directional["target_surface"].eq(surface)
                        denominator = directional.loc[base_mask]
                        eligible_mask = base_mask & directional["eligible"]
                        if selected_scope is not None:
                            eligible_mask &= directional["selected_scope"].eq(selected_scope)
                        eligible = directional.loc[eligible_mask]
                        eligible_by_match = eligible.groupby("target_match_id").size()
                        eligible_matches = int(eligible_by_match.eq(2).sum())
                        base_matches = int(denominator["target_match_id"].nunique())
                        if surface is None:
                            complete_matches = int(complete_match.loc[fold].sum())
                        else:
                            ids = set(denominator["target_match_id"])
                            complete_matches = int(
                                complete_match.loc[fold].loc[
                                    complete_match.loc[fold].index.isin(ids)
                                ].sum()
                            )
                        surface_count = int(eligible["selected_scope"].eq("surface").sum())
                        global_count = int(eligible["selected_scope"].eq("global").sum())
                        abstentions = int(len(denominator) - len(eligible))
                        record = {
                            "policy_id": policy["policy_id"],
                            "min_points": int(policy["min_points"]),
                            "min_matches": int(policy["min_matches"]),
                            "scope_policy": policy["scope_policy"],
                            "fold": int(fold),
                            "direction": direction,
                            "role": role,
                            "stratum_type": stratum_type,
                            "surface": "<ALL>" if surface is None else surface,
                            "selected_scope": "<ALL>" if selected_scope is None else selected_scope,
                            "target_matches": base_matches,
                            "target_orientations": int(len(denominator)),
                            "eligible_orientations": int(len(eligible)),
                            "eligible_matches": eligible_matches,
                            "complete_matches": complete_matches,
                            "abstained_orientations": abstentions,
                            "abstained_matches": int(base_matches - eligible_matches),
                            "eligible_players": int(eligible["target_player"].nunique()),
                            "coverage_orientation": _ratio(len(eligible), len(denominator)),
                            "coverage_match": _ratio(eligible_matches, base_matches),
                            "coverage_complete_match": _ratio(complete_matches, base_matches),
                            "surface_selections": surface_count,
                            "global_selections": global_count,
                            "abstentions": abstentions,
                            "surface_selection_proportion": _ratio(surface_count, len(denominator)),
                            "global_selection_proportion": _ratio(global_count, len(denominator)),
                            "abstention_proportion": _ratio(abstentions, len(denominator)),
                            **_precision(eligible[f"{role}_selected_wilson_width"]),
                            **stability_cache.get(
                                (
                                    fold, direction, role, stratum_type,
                                    "<ALL>" if surface is None else surface,
                                    "<ALL>" if selected_scope is None else selected_scope,
                                ),
                                _empty_stability(fold),
                            ),
                        }
                        rows.append(record)
    result = pd.DataFrame(rows, columns=BY_FOLD_COLUMNS)
    key = [
        "policy_id", "fold", "direction", "role", "stratum_type",
        "surface", "selected_scope",
    ]
    if result.duplicated(key).any():
        raise ValueError("La clave de la tabla por fold no es unica.")
    return result


def _coverage_for_values(frame: pd.DataFrame, column: str, value: str) -> float | None:
    subset = frame.loc[frame[column].eq(value)]
    return _ratio(int(subset["eligible"].sum()), len(subset))


def build_candidates(
    evaluated_policies: list[pd.DataFrame], by_fold: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evaluated in evaluated_policies:
        policy = evaluated.iloc[0]
        _, complete = _match_flags(evaluated)
        match_any = evaluated.groupby(["fold", "target_match_id"], sort=False)["eligible"].any()
        fold_complete = complete.groupby(level=0).mean().reindex(FOLDS)
        widths = pd.concat(
            [evaluated.loc[evaluated["eligible"], f"{role}_selected_wilson_width"] for role in ROLES],
            ignore_index=True,
        )
        overall_rows = by_fold.loc[
            by_fold["policy_id"].eq(policy["policy_id"])
            & by_fold["stratum_type"].eq("overall")
        ]
        fold_width = overall_rows.groupby("fold", sort=False)["wilson_width_p90"].max()
        stability = overall_rows.loc[
            overall_rows["fold"].gt(2020) & overall_rows["absolute_change_p90"].notna()
        ]
        selected_total = int(evaluated["eligible"].sum())
        denominator = len(evaluated)
        record = {
            "policy_id": policy["policy_id"],
            "min_points": int(policy["min_points"]),
            "min_matches": int(policy["min_matches"]),
            "scope_policy": policy["scope_policy"],
            "validation_matches": int(evaluated["target_match_id"].nunique()),
            "orientation_direction_denominator": denominator,
            "eligible_orientation_directions": selected_total,
            "coverage_orientation": _ratio(selected_total, denominator),
            "eligible_matches": int(match_any.sum()),
            "coverage_match": float(match_any.mean()),
            "complete_matches": int(complete.sum()),
            "coverage_complete_match": float(complete.mean()),
            "worst_fold_complete_match_coverage": float(fold_complete.min()),
            "complete_match_coverage_range": float(fold_complete.max() - fold_complete.min()),
            "wide_coverage": _coverage_for_values(evaluated, "direction", "wide"),
            "body_coverage": _coverage_for_values(evaluated, "direction", "body"),
            "T_coverage": _coverage_for_values(evaluated, "direction", "T"),
            "hard_coverage": _coverage_for_values(evaluated, "target_surface", "Hard"),
            "clay_coverage": _coverage_for_values(evaluated, "target_surface", "Clay"),
            "grass_coverage": _coverage_for_values(evaluated, "target_surface", "Grass"),
            "median_wilson_width": _quantile(widths, .5),
            "p90_wilson_width": _quantile(widths, .9),
            "worst_fold_p90_wilson_width": (
                None if fold_width.dropna().empty else float(fold_width.max())
            ),
            "median_absolute_change": _quantile(stability["absolute_change_median"], .5),
            "p90_absolute_change": _quantile(stability["absolute_change_p90"], .9),
            "worst_fold_p90_absolute_change": (
                None if stability.empty else float(stability["absolute_change_p90"].max())
            ),
            "within_005_proportion": (
                None if stability.empty else float(stability["within_005_proportion"].mean())
            ),
            "within_010_proportion": (
                None if stability.empty else float(stability["within_010_proportion"].mean())
            ),
            "valid_stability_comparisons": int(stability["comparable_players"].sum()),
            "surface_selection_proportion": _ratio(
                int(evaluated["selected_scope"].eq("surface").sum()), denominator
            ),
            "global_selection_proportion": _ratio(
                int(evaluated["selected_scope"].eq("global").sum()), denominator
            ),
            "abstention_proportion": _ratio(denominator - selected_total, denominator),
            "scope_coherence_reconciled": True,
        }
        rows.append(record)
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def build_pareto(candidates: pd.DataFrame) -> pd.DataFrame:
    """Calcula dominancia exacta sin redondear los tres ejes."""

    axes = [
        "worst_fold_complete_match_coverage",
        "worst_fold_p90_wilson_width",
        "worst_fold_p90_absolute_change",
    ]
    rows = []
    for candidate in candidates.itertuples(index=False):
        values = [getattr(candidate, axis) for axis in axes]
        if any(pd.isna(value) or not np.isfinite(value) for value in values):
            status = "not_comparable"
            dominators: list[str] = []
            reason = "undefined_pareto_axis"
        else:
            dominators = []
            for other in candidates.itertuples(index=False):
                if other.policy_id == candidate.policy_id:
                    continue
                other_values = [getattr(other, axis) for axis in axes]
                if any(pd.isna(value) or not np.isfinite(value) for value in other_values):
                    continue
                no_worse = (
                    other_values[0] >= values[0]
                    and other_values[1] <= values[1]
                    and other_values[2] <= values[2]
                )
                strictly_better = (
                    other_values[0] > values[0]
                    or other_values[1] < values[1]
                    or other_values[2] < values[2]
                )
                if no_worse and strictly_better:
                    dominators.append(other.policy_id)
            dominators.sort()
            status = "dominated" if dominators else "non_dominated"
            reason = "dominated_on_prespecified_axes" if dominators else None
        rows.append(
            {
                "policy_id": candidate.policy_id,
                "min_points": int(candidate.min_points),
                "min_matches": int(candidate.min_matches),
                "scope_policy": candidate.scope_policy,
                "pareto_status": status,
                "dominated_by": "|".join(dominators),
                "reason_code": reason,
                **{axis: getattr(candidate, axis) for axis in axes},
            }
        )
    return pd.DataFrame(rows, columns=PARETO_COLUMNS)


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    return buffer.getvalue().encode("utf-8")


def _fingerprint(
    candidates: pd.DataFrame,
    by_fold: pd.DataFrame,
    pareto: pd.DataFrame,
    summary_core: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    for frame in (candidates, by_fold, pareto):
        digest.update(_frame_bytes(frame))
    digest.update(
        json.dumps(
            summary_core, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest().upper()


def build_summary(
    *,
    source_rows: int,
    source_matches: int,
    source_players: int,
    eligible_points: int,
    server_wins: int,
    construction_audit: dict[str, int],
    validation_features: pd.DataFrame,
    candidates: pd.DataFrame,
    by_fold: pd.DataFrame,
    pareto: pd.DataFrame,
) -> dict[str, Any]:
    matches_by_fold = (
        validation_features[["fold", "target_match_id"]]
        .drop_duplicates()
        .groupby("fold", sort=True)["target_match_id"].size()
        .reindex(FOLDS, fill_value=0)
    )
    grass = candidates[["policy_id", "grass_coverage", "hard_coverage", "clay_coverage"]].copy()
    grass["lowest_surface_coverage"] = grass["grass_coverage"].le(
        grass[["hard_coverage", "clay_coverage"]].min(axis=1)
    )
    non_dominated = pareto.loc[pareto["pareto_status"].eq("non_dominated"), "policy_id"].tolist()
    not_comparable = pareto.loc[pareto["pareto_status"].eq("not_comparable"), "policy_id"].tolist()
    core: dict[str, Any] = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "source": {
            "path": "data/processed/points_enriched.parquet",
            "columns_used": list(SOURCE_COLUMNS),
            "published_reports_used_as_analytical_source": False,
            "physical_parquet_reads": 1,
            "source_point_rows_read": int(source_rows),
            "source_matches_observed": int(source_matches),
            "complete_source_read_for_contract_reconciliation": True,
            "snapshot_builds": 1,
            "feature_builds": 1,
        },
        "upstream_commits": UPSTREAM_COMMITS,
        "population_contract": {
            "source_point_rows": int(source_rows),
            "source_matches": int(source_matches),
            "source_players": int(source_players),
            "eligible_second_serve_direction_points": int(eligible_points),
            "eligible_server_wins": int(server_wins),
            "upstream_total_snapshots": int(source_matches * 2),
            "upstream_total_feature_rows": int(source_matches * 2 * 3 * 2),
            "validation_matches": int(matches_by_fold.sum()),
            "validation_orientations": int(matches_by_fold.sum() * 2),
            "validation_orientation_direction_rows": int(matches_by_fold.sum() * 2 * 3),
        },
        "target_construction_contract": {
            "latest_target_date": "2023-12-31",
            "source_target_matches": int(construction_audit["source_target_matches"]),
            "constructed_target_matches": int(construction_audit["constructed_target_matches"]),
            "test_target_matches_excluded_before_construction": int(
                construction_audit["test_target_matches_excluded_before_construction"]
            ),
            "constructed_snapshot_rows": int(construction_audit["constructed_snapshot_rows"]),
            "constructed_feature_rows": int(construction_audit["constructed_feature_rows"]),
            "snapshots_per_constructed_match": 2,
            "features_per_snapshot": 6,
            "test_target_rows_constructed": int(
                construction_audit["test_target_rows_constructed"]
            ),
            "test_target_matches_constructed": int(
                construction_audit["test_target_matches_constructed"]
            ),
            "test_feature_rows_constructed": int(
                construction_audit["test_feature_rows_constructed"]
            ),
        },
        "validation_protocol": {
            "type": "rolling_origin_expanding",
            "folds": [
                {
                    "fold": int(fold),
                    "history_end": f"{fold - 1}-12-31",
                    "evaluation_start": f"{fold}-01-01",
                    "evaluation_end": f"{fold}-12-31",
                    "evaluation_matches": int(matches_by_fold.loc[fold]),
                }
                for fold in FOLDS
            ],
            "history_strictly_before_target_calendar_day": True,
            "same_day_history_blocked": True,
        },
        "sealed_test_contract": {
            "test_status": "sealed",
            "test_rows_evaluated": 0,
            "test_matches_evaluated": 0,
            "used_for_method_selection": False,
            "test_evaluation_runs": 0,
            "latest_allowed_target_date": "2023-12-31",
            "allowed_folds": list(FOLDS),
        },
        "candidate_grid": {
            "minimum_direction_points_per_role": list(POINT_THRESHOLDS),
            "minimum_direction_matches_per_role": list(MATCH_THRESHOLDS),
            "scope_policies": list(SCOPE_POLICIES),
            "cartesian_candidates": 27,
            "changed_after_observing_results": False,
        },
        "eligibility_contract": {
            "unit": "target_match_orientation_direction",
            "both_roles_must_pass": True,
            "roles": list(ROLES),
            "directions": list(DIRECTIONS),
            "thresholds_are_inclusive": True,
            "complete_match_requires_two_orientations_and_three_directions": True,
        },
        "scope_policy_contracts": {
            "global_only": "global history only",
            "surface_only": "target-surface history only; no fallback",
            "surface_then_global": "surface only if both roles pass; otherwise global only if both pass",
            "server_and_opponent_scope_must_match": True,
        },
        "analysis_status": "available",
        "reason_codes": [],
        "fold_reconciliations": {
            "matches_by_fold": {str(key): int(value) for key, value in matches_by_fold.items()},
            "fold_sum": int(matches_by_fold.sum()),
            "expected_fold_sum": 1_805,
        },
        "candidate_summary": {
            "candidate_count": int(len(candidates)),
            "columns": CANDIDATE_COLUMNS,
            "no_automatic_winner": True,
        },
        "pareto_contract": {
            "maximize": "worst_fold_complete_match_coverage",
            "minimize": ["worst_fold_p90_wilson_width", "worst_fold_p90_absolute_change"],
            "strict_improvement_required_on_at_least_one_axis": True,
            "unrounded_values_used": True,
            "undefined_axis_status": "not_comparable",
        },
        "pareto_summary": {
            "non_dominated_policy_ids": non_dominated,
            "dominated_policies": int(pareto["pareto_status"].eq("dominated").sum()),
            "not_comparable_policy_ids": not_comparable,
            "automatic_selection_performed": False,
        },
        "surface_summary": {
            "surfaces": list(SURFACE_ORDER),
            "grass_is_lowest_coverage_for_all_candidates": bool(grass["lowest_surface_coverage"].all()),
            "grass_fragmentation_statement": (
                "Grass is the most fragmented coverage stratum in this validation grid."
                if grass["lowest_surface_coverage"].all()
                else "Grass is not uniformly the lowest-coverage stratum across candidates."
            ),
            "used_to_modify_candidate_grid": False,
        },
        "stability_contract": {
            "transitions": ["2020_to_2021", "2021_to_2022", "2022_to_2023"],
            "unit": "eligible player mean historical rate within fold",
            "noncomparable_observations_imputed": False,
            "undefined_spearman_is_null_with_reason_code": True,
            "absolute_change_cuts": [0.05, 0.10],
        },
        "uncertainty_contract": {
            "interval": "two-sided Wilson 95%",
            "z": Z_975,
            "width_summaries": ["mean", "median", "p75", "p90", "p95", "maximum"],
            "quantile_method": QUANTILE_METHOD,
            "zero_denominator_is_null": True,
            "independent_scalar_check_performed": True,
        },
        "reconciliations": {
            "source_population_reconciled": bool(source_matches == EXPECTED_MATCHES),
            "constructed_targets_plus_excluded_reconcile": bool(
                construction_audit["constructed_target_matches"]
                + construction_audit["test_target_matches_excluded_before_construction"]
                == source_matches
            ),
            "two_snapshots_per_constructed_match": bool(
                construction_audit["constructed_snapshot_rows"]
                == construction_audit["constructed_target_matches"] * 2
            ),
            "six_features_per_constructed_snapshot": bool(
                construction_audit["constructed_feature_rows"]
                == construction_audit["constructed_snapshot_rows"] * 6
            ),
            "zero_test_targets_constructed": bool(
                construction_audit["test_target_rows_constructed"] == 0
                and construction_audit["test_target_matches_constructed"] == 0
                and construction_audit["test_feature_rows_constructed"] == 0
            ),
            "validation_fold_sum_reconciled": bool(matches_by_fold.sum() == 1_805),
            "folds_exact": bool(tuple(matches_by_fold.index) == FOLDS),
            "candidate_grid_reconciled": bool(len(candidates) == 27),
            "candidate_keys_unique": bool(not candidates["policy_id"].duplicated().any()),
            "by_fold_keys_unique": bool(not by_fold.duplicated([
                "policy_id", "fold", "direction", "role", "stratum_type",
                "surface", "selected_scope",
            ]).any()),
            "pareto_keys_unique": bool(not pareto["policy_id"].duplicated().any()),
            "test_remains_sealed": True,
            "scope_coherence_reconciled": bool(candidates["scope_coherence_reconciled"].all()),
            "rates_bounded": True,
            "no_nan_or_infinity_serialized": True,
            "no_policy_winner_selected": True,
        },
        "methodological_limits": [
            "The comparison is descriptive and does not identify causal effects.",
            "Narrower Wilson intervals do not imply tactical superiority.",
            "Greater stability does not by itself imply a better recommendation.",
            "Pareto membership does not validate a threshold or select a final policy.",
            "Coverage is conditional on manually charted historical data and the approved parser contract.",
        ],
    }
    return _json_value(core)


def validate_result(result: EvidencePolicyValidationResult) -> None:
    if result.candidates.columns.tolist() != CANDIDATE_COLUMNS:
        raise ValueError("Esquema de candidates no contractual.")
    if result.by_fold.columns.tolist() != BY_FOLD_COLUMNS:
        raise ValueError("Esquema de by_fold no contractual.")
    if result.pareto.columns.tolist() != PARETO_COLUMNS:
        raise ValueError("Esquema de pareto no contractual.")
    if len(result.candidates) != 27 or len(result.pareto) != 27:
        raise ValueError("Deben publicarse exactamente 27 candidatas.")
    if result.candidates["policy_id"].tolist() != build_candidate_grid()["policy_id"].tolist():
        raise ValueError("IDs u orden de candidatas alterado.")
    if result.pareto["policy_id"].tolist() != result.candidates["policy_id"].tolist():
        raise ValueError("Pareto no conserva el orden completo de candidatas.")
    if result.by_fold.duplicated(
        ["policy_id", "fold", "direction", "role", "stratum_type", "surface", "selected_scope"]
    ).any():
        raise ValueError("Clave duplicada en by_fold.")
    if not result.by_fold["fold"].isin(FOLDS).all():
        raise ValueError("Fold fuera de validacion.")
    rate_columns = [column for column in [*CANDIDATE_COLUMNS, *BY_FOLD_COLUMNS] if "coverage" in column or "proportion" in column]
    for frame in (result.candidates, result.by_fold):
        for column in set(rate_columns) & set(frame.columns):
            values = frame[column].dropna()
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Tasa fuera de [0,1]: {column}")
    for frame in (result.candidates, result.by_fold, result.pareto):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError("Un CSV contiene infinito.")
    if result.summary.get("analysis_status") != "available":
        raise ValueError("El resultado publicable debe estar disponible.")
    sealed = result.summary.get("sealed_test_contract", {})
    if sealed != {
        "test_status": "sealed", "test_rows_evaluated": 0, "test_matches_evaluated": 0,
        "used_for_method_selection": False, "test_evaluation_runs": 0,
        "latest_allowed_target_date": "2023-12-31", "allowed_folds": list(FOLDS),
    }:
        raise ValueError("Contrato de test sellado alterado.")
    if "test_rows_read" in json.dumps(result.summary, ensure_ascii=False):
        raise ValueError("El resumen conserva el campo ambiguo test_rows_read.")
    construction = result.summary.get("target_construction_contract", {})
    required_zero = (
        "test_target_rows_constructed",
        "test_target_matches_constructed",
        "test_feature_rows_constructed",
    )
    if any(construction.get(field) != 0 for field in required_zero):
        raise ValueError("Se construyeron targets o features pertenecientes al test.")
    counts = {
        field: construction.get(field)
        for field in (
            "constructed_target_matches", "constructed_snapshot_rows", "constructed_feature_rows"
        )
    }
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in counts.values()):
        raise ValueError("Faltan cardinalidades enteras de construccion.")
    if counts["constructed_snapshot_rows"] != 2 * counts["constructed_target_matches"]:
        raise ValueError("Los snapshots construidos no reconcilian con los targets.")
    if counts["constructed_feature_rows"] != 6 * counts["constructed_snapshot_rows"]:
        raise ValueError("Las features construidas no reconcilian con los snapshots.")
    reconciliations = result.summary.get("reconciliations", {})
    for field in (
        "constructed_targets_plus_excluded_reconcile",
        "two_snapshots_per_constructed_match",
        "six_features_per_constructed_snapshot",
        "zero_test_targets_constructed",
        "folds_exact",
        "candidate_grid_reconciled",
        "candidate_keys_unique",
        "by_fold_keys_unique",
        "pareto_keys_unique",
        "test_remains_sealed",
        "scope_coherence_reconciled",
    ):
        if reconciliations.get(field) is not True:
            raise ValueError(f"Reconciliacion contractual falsa o ausente: {field}")
    _assert_finite_json(result.summary)
    core = {key: value for key, value in result.summary.items() if key != "publication_fingerprint"}
    expected = _fingerprint(result.candidates, result.by_fold, result.pareto, core)
    if result.publication_fingerprint != expected:
        raise ValueError("Fingerprint interno alterado.")
    if result.summary.get("publication_fingerprint") != expected:
        raise ValueError("Fingerprint del resumen alterado.")


def analyze_from_features(
    validation_features: pd.DataFrame,
    *,
    source_rows: int,
    source_matches: int,
    source_players: int,
    eligible_points: int,
    server_wins: int,
    construction_audit: dict[str, int] | None = None,
    recorder: PerformanceRecorder | None = None,
) -> EvidencePolicyValidationResult:
    recorder = recorder or PerformanceRecorder()
    expected_columns = [*FEATURE_COLUMNS, "fold"]
    if validation_features.columns.tolist() != expected_columns:
        raise ValueError("El esquema de features de validacion no es contractual.")
    validation_dates = _normalized_feature_dates(validation_features)
    if not validation_dates.between(VALIDATION_START, VALIDATION_END).all():
        raise ValueError("La evaluacion solo admite targets de validacion 2020--2023.")
    if not validation_features["fold"].isin(FOLDS).all():
        raise ValueError("La evaluacion contiene un fold fuera de 2020--2023.")
    if not validation_features["fold"].eq(validation_dates.dt.year).all():
        raise ValueError("El fold no coincide con el ano civil del target.")
    if construction_audit is None:
        constructed_matches = int(validation_features["target_match_id"].nunique())
        construction_audit = {
            "source_target_matches": int(source_matches),
            "constructed_target_matches": constructed_matches,
            "test_target_matches_excluded_before_construction": int(
                source_matches - constructed_matches
            ),
            "constructed_snapshot_rows": constructed_matches * 2,
            "constructed_feature_rows": int(len(validation_features)),
            "test_target_rows_constructed": 0,
            "test_target_matches_constructed": 0,
            "test_feature_rows_constructed": 0,
        }
    started = time.perf_counter()
    population = prepare_policy_population(validation_features)
    recorder.record("prepare_policy_population", started, len(population))
    started = time.perf_counter()
    grid = build_candidate_grid()
    evaluated = [apply_policy(population, policy) for _, policy in grid.iterrows()]
    recorder.record("policy_grid", started, sum(map(len, evaluated)))
    started = time.perf_counter()
    by_fold = build_by_fold(evaluated)
    recorder.record("coverage_wilson_stability", started, len(by_fold))
    started = time.perf_counter()
    candidates = build_candidates(evaluated, by_fold)
    pareto = build_pareto(candidates)
    recorder.record("pareto", started, len(pareto))
    summary = build_summary(
        source_rows=source_rows,
        source_matches=source_matches,
        source_players=source_players,
        eligible_points=eligible_points,
        server_wins=server_wins,
        construction_audit=construction_audit,
        validation_features=validation_features,
        candidates=candidates,
        by_fold=by_fold,
        pareto=pareto,
    )
    fingerprint = _fingerprint(candidates, by_fold, pareto, summary)
    summary["publication_fingerprint"] = fingerprint
    result = EvidencePolicyValidationResult(
        candidates, by_fold, pareto, summary, fingerprint, recorder.snapshot()
    )
    started = time.perf_counter()
    validate_result(result)
    recorder.record("validation", started, len(by_fold))
    return EvidencePolicyValidationResult(
        candidates, by_fold, pareto, summary, fingerprint, recorder.snapshot()
    )


def build_real_analysis(
    points: pd.DataFrame,
    *,
    parser: Callable = parse_sequence,
    recorder: PerformanceRecorder | None = None,
) -> EvidencePolicyValidationResult:
    """Construye snapshots y features exactamente una vez desde una lectura externa."""

    recorder = recorder or PerformanceRecorder()
    started = time.perf_counter()
    frame, targets, eligible, audit = prepare_historical_population(points, parser=parser)
    recorder.record("preparation", started, len(eligible))
    snapshots, features, construction_audit, test_ids = construct_sealed_target_features(
        targets,
        eligible,
        recorder=recorder,
    )
    started = time.perf_counter()
    validation = select_validation_features(features, test_ids, evaluation_runs=0)
    recorder.record("sealed_test_barrier", started, len(validation))
    source_players = len(set(targets["player_1"]) | set(targets["player_2"]))
    return analyze_from_features(
        validation,
        source_rows=len(frame),
        source_matches=len(targets),
        source_players=source_players,
        eligible_points=len(eligible),
        server_wins=int(eligible["server_won_point"].sum()),
        construction_audit=construction_audit,
        recorder=recorder,
    )


def serialize_artifacts(
    result: EvidencePolicyValidationResult,
) -> tuple[bytes, bytes, bytes, bytes]:
    validate_result(result)
    summary = (
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    payloads = (summary, _frame_bytes(result.candidates), _frame_bytes(result.by_fold), _frame_bytes(result.pareto))
    validate_serialized_artifacts(result, *payloads)
    return payloads


def validate_serialized_artifacts(
    result: EvidencePolicyValidationResult,
    summary: bytes,
    candidates: bytes,
    by_fold: bytes,
    pareto: bytes,
) -> None:
    if json.loads(summary.decode("utf-8")) != result.summary:
        raise ValueError("El JSON serializado no reconcilia.")
    payloads = (candidates, by_fold, pareto)
    expected = (result.candidates, result.by_fold, result.pareto)
    columns = (CANDIDATE_COLUMNS, BY_FOLD_COLUMNS, PARETO_COLUMNS)
    for payload, canonical, schema in zip(payloads, expected, columns):
        observed = pd.read_csv(io.BytesIO(payload))
        if observed.columns.tolist() != schema or any(c.startswith("Unnamed") for c in observed.columns):
            raise ValueError("Esquema CSV serializado alterado.")
        if payload != _frame_bytes(canonical):
            raise ValueError("El CSV serializado no coincide byte a byte con la tabla canonica.")


def not_available_payloads(reason_codes: list[str]) -> tuple[bytes, bytes, bytes, bytes]:
    """Contrato sin resultados parciales para fallos previos a publicacion."""

    summary = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_status": "not_available",
        "reason_codes": sorted(set(reason_codes)),
        "sealed_test_contract": {
            "test_status": "sealed", "test_rows_evaluated": 0,
            "test_matches_evaluated": 0, "used_for_method_selection": False,
            "test_evaluation_runs": 0,
        },
        "partial_results_published": False,
    }
    payload = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    headers = tuple(_frame_bytes(pd.DataFrame(columns=columns)) for columns in (CANDIDATE_COLUMNS, BY_FOLD_COLUMNS, PARETO_COLUMNS))
    return (payload, *headers)


def _stage_payload(path: Path, payload: bytes) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staged = Path(temporary)
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
    result: EvidencePolicyValidationResult,
    summary_path: Path = SUMMARY_PATH,
    candidates_path: Path = CANDIDATES_PATH,
    by_fold_path: Path = BY_FOLD_PATH,
    pareto_path: Path = PARETO_PATH,
) -> None:
    paths = (summary_path, candidates_path, by_fold_path, pareto_path)
    if len({path.resolve() for path in paths}) != 4:
        raise ValueError("Las cuatro rutas de artefactos deben ser distintas.")
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


def _artifact_metadata(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
    }


def remove_reproduction_directory(path: Path = REPRO_DIR) -> None:
    """Elimina exclusivamente el directorio temporal de esta reproduccion."""

    if path.resolve() != REPRO_DIR.resolve():
        raise ValueError("Solo puede eliminarse el temporal contractual de reproducibilidad.")
    if path.exists():
        shutil.rmtree(path)


def main() -> None:
    if REPRO_DIR.exists():
        raise FileExistsError(f"El temporal de reproducibilidad ya existe: {REPRO_DIR}")
    recorder = PerformanceRecorder()
    started = time.perf_counter()
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    recorder.record("read", started, len(points))
    result = build_real_analysis(points, recorder=recorder)
    write_artifacts(result)
    payloads = serialize_artifacts(result)
    permanent = (SUMMARY_PATH, CANDIDATES_PATH, BY_FOLD_PATH, PARETO_PATH)
    if tuple(path.read_bytes() for path in permanent) != payloads:
        raise ValueError("Los artefactos permanentes no coinciden con la serializacion en memoria.")
    REPRO_DIR.mkdir(parents=False)
    reproduced = tuple(REPRO_DIR / path.name for path in permanent)
    try:
        write_artifacts(result, *reproduced)
        byte_identical = [
            source.read_bytes() == reproduction.read_bytes()
            for source, reproduction in zip(permanent, reproduced)
        ]
        if not all(byte_identical):
            raise ValueError("La reserializacion temporal no es identica byte a byte.")
        reproduced_metadata = [_artifact_metadata(path) for path in reproduced]
    finally:
        remove_reproduction_directory()
    if REPRO_DIR.exists():
        raise ValueError("El temporal de reproducibilidad no fue eliminado.")
    print(
        json.dumps(
            {
                "analysis_status": result.summary["analysis_status"],
                "non_dominated_policy_ids": result.summary["pareto_summary"]["non_dominated_policy_ids"],
                "performance_metrics": recorder.snapshot(),
                "artifacts": [_artifact_metadata(path) for path in permanent],
                "byte_identical_reproduction": byte_identical,
                "reproduced_artifacts_before_cleanup": reproduced_metadata,
                "reproduction_directory_removed": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
