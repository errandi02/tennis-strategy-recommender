"""Baseline cronologico, explicable y no causal para direcciones de segundo saque.

El modulo consume una unica lectura del Parquet y reutiliza los constructores
leakage-safe ya publicados.  Solo valida 2020--2023: el test queda fuera de la
construccion de snapshots, features, scores y metricas.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.analysis.chronological_validation import (
    EXPECTED_MATCHES,
    EXPECTED_SOURCE_ROWS,
    VALIDATION_END,
    VALIDATION_START,
)
from src.analysis.evidence_policy_validation import (
    apply_policy,
    construct_sealed_target_features,
    prepare_policy_population,
    select_validation_features,
)
from src.analysis.historical_profiles import POINTS_FILE, SOURCE_COLUMNS, prepare_historical_population
from src.analysis.player_opponent_direction_features import DIRECTIONS
from src.analysis.second_serve_direction_analysis import classify_presence


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "explainable_direction_scoring_summary.json"
BY_FOLD_PATH = TABLES_DIR / "explainable_direction_scoring_by_fold.csv"
BY_DIRECTION_PATH = TABLES_DIR / "explainable_direction_scoring_by_direction.csv"
CALIBRATION_PATH = TABLES_DIR / "explainable_direction_scoring_calibration.csv"
RANKING_PATH = TABLES_DIR / "explainable_direction_scoring_ranking.csv"
REPRO_DIR = REPORTS_DIR / "repro_explainable_direction_scoring"

ANALYSIS_NAME = "explainable_direction_scoring"
ANALYSIS_VERSION = "1.0.0"
POLICY_ID = "p050_m05_surface_then_global"
MIN_POINTS = 50
MIN_MATCHES = 5
SCOPE_POLICY = "surface_then_global"
SCORERS = ("population_only", "server_only", "opponent_only", "server_opponent_equal")
MAIN_SCORER = "server_opponent_equal"
DIRECTION_ORDER = ("wide", "body", "T")
SCOPE_ORDER = ("surface", "global")
FOLDS = (2020, 2021, 2022, 2023)
EPSILON = 1e-15
BIN_COUNT = 10
MAX_TARGET_DATE = pd.Timestamp(VALIDATION_END)
MAX_EXCEPTION_MESSAGE_LENGTH = 320
STAGES = (
    "load_upstream_contracts", "read_source", "prepare_history", "seal_targets",
    "build_snapshots", "build_features", "select_validation_features",
    "prepare_scoring_population", "join_target_outcomes", "compute_scores",
    "probability_metrics", "calibration", "ranking", "validate_result", "serialize", "publish",
)

# Los hashes fijan las tres fuentes agregadas que se permiten consultar.
UPSTREAM_HASHES = {
    "reports/evidence_policy_selection_summary.json": "317CC9F3F603672965C98DD93AB0C4D32F8CF91D86A0FC76558063720839C330",
    "reports/evidence_policy_validation_summary.json": "5AAC50C23DE8F9EECE61A573D445FB1B816D4769BE61D6875A614BE1E0E812BF",
    "reports/chronological_validation_summary.json": "90890B721CC93C82A5DE5F2EB6E3F3D4E5D4D3E83E6C9E678C5569139301799F",
}
UPSTREAM_COMMITS = {
    "historical_profiles": "68a30c1",
    "player_opponent_direction_features": "a1a41aa",
    "chronological_validation": "3ac28b4",
    "evidence_policy_validation": "88fafe7",
    "evidence_policy_selection": "989d02a",
}

BY_FOLD_COLUMNS = [
    "scorer", "fold", "direction", "selected_scope", "n_points", "n_matches",
    "n_orientations", "mean_score", "observed_rate", "calibration_gap", "brier_score",
    "log_loss", "calibration_in_the_large", "ece", "max_calibration_error",
    "eligible_target_points", "scored_target_points", "abstained_target_points",
    "point_coverage", "delta_brier_vs_population", "delta_log_loss_vs_population",
    "delta_ece_vs_population",
]
BY_DIRECTION_COLUMNS = [
    "scorer", "direction", "fold_or_pooled", "n_points", "n_matches", "n_orientations",
    "mean_score", "observed_rate", "calibration_gap", "brier_score", "log_loss",
    "server_component_mean", "opponent_component_mean", "combined_gap_mean",
]
CALIBRATION_COLUMNS = [
    "scorer", "fold_or_pooled", "bin_index", "bin_low", "bin_high", "n_points",
    "n_matches", "mean_score", "observed_rate", "absolute_gap",
]
RANKING_COLUMNS = [
    "scorer", "fold_or_pooled", "comparable_orientations", "comparable_matches",
    "top1_hit_inclusive", "top1_hit_strict", "strict_winner_orientations", "mrr",
    "spearman", "spearman_reason_code", "mean_regret", "median_regret", "p90_regret",
    "regret_at_most_005", "regret_at_most_010",
]
ARTIFACT_PAYLOAD_NAMES = ("by_fold", "by_direction", "calibration", "ranking")
FINGERPRINT_CONTRACT_VERSION = "2"
RECONCILIATION_KEYS = frozenset({
    "upstream_hashes_exact", "policy_exact", "surface_then_global_atomic",
    "targets_no_later_than_2023_12_31", "folds_exact_2020_2023",
    "one_parquet_read", "one_snapshot_construction", "one_feature_construction",
    "two_orientations_per_match", "directions_exact", "feature_keys_unique",
    "strict_history_prior", "zero_self_reference", "zero_temporal_violations",
    "outcome_binary", "scores_finite_in_unit_interval",
    "formula_identity_reconciled", "calibration_bins_complete",
    "ranking_only_comparable_orientations", "summary_and_csv_canonical",
    "no_test_contamination",
})


@dataclass(frozen=True)
class ScoringResult:
    summary: dict[str, Any]
    by_fold: pd.DataFrame
    by_direction: pd.DataFrame
    calibration: pd.DataFrame
    ranking: pd.DataFrame
    publication_fingerprint: str
    performance_metrics: tuple[dict[str, Any], ...] = ()


class ScoringContractError(ValueError):
    """Fallo cerrado que impide publicar una evaluacion parcial."""


class ScoringExecutionError(ScoringContractError):
    """Fallo de una unica frontera, con diagnostico estable para not_available."""

    def __init__(
        self,
        stage: str,
        exception: BaseException,
        completed_stages: list[str],
        partial_population_diagnostics: Mapping[str, Any],
    ) -> None:
        super().__init__(f"{stage}: {type(exception).__name__}: {exception}")
        self.stage = stage
        self.exception_type = type(exception).__name__
        self.exception_message = sanitize_exception_message(str(exception))
        self.completed_stages = tuple(completed_stages)
        self.partial_population_diagnostics = _json_value(dict(partial_population_diagnostics))
        self.reason_code = reason_code_for_failure(stage, exception)


class PerformanceRecorder:
    """Instrumentacion efimera: no se serializa en los artefactos canonicos."""

    def __init__(self, callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.callback = callback
        self.records: list[dict[str, Any]] = []

    def record(self, phase: str, started: float, rows: int | None = None) -> None:
        item = {"phase": phase, "duration_seconds": time.perf_counter() - started, "rows": rows}
        self.records.append(item)
        if self.callback is not None:
            self.callback(item.copy())

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.copy() for item in self.records)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    return buffer.getvalue().encode("utf-8")


def _csv_payloads(frames: tuple[pd.DataFrame, ...]) -> tuple[bytes, ...]:
    return tuple(_frame_bytes(frame) for frame in frames)


def _payload_metadata(payloads: tuple[bytes, ...]) -> tuple[dict[str, str], dict[str, int]]:
    if len(payloads) != len(ARTIFACT_PAYLOAD_NAMES):
        raise ScoringContractError("El contrato exige cuatro payloads CSV.")
    return (
        {name: _sha256(payload) for name, payload in zip(ARTIFACT_PAYLOAD_NAMES, payloads)},
        {name: len(payload) for name, payload in zip(ARTIFACT_PAYLOAD_NAMES, payloads)},
    )


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
        return value if math.isfinite(value) else None
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
        if not math.isfinite(value):
            raise ScoringContractError(f"Valor no finito en {path}.")
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


def sanitize_exception_message(message: str) -> str:
    """Conserva el diagnostico tecnico sin rutas, saltos ni texto ilimitado."""

    normalized = " ".join(str(message).replace("\r", " ").replace("\n", " ").split())
    normalized = normalized.replace(str(ROOT), "<repo>")
    normalized = re.sub(r"[A-Za-z]:\\[^\s\"']+", "<absolute-path>", normalized)
    return normalized[:MAX_EXCEPTION_MESSAGE_LENGTH]


def reason_code_for_failure(stage: str, exception: BaseException) -> str:
    message = str(exception).lower()
    if stage == "select_validation_features":
        if "esquema" in message or "columns" in message:
            return "validation_feature_schema_mismatch"
        if "fold" in message or "2023" in message:
            return "validation_fold_assignment_invalid"
        return "validation_feature_contract_failure"
    if stage == "prepare_scoring_population":
        if "merge" in message or "one_to_one" in message or "clave" in message:
            return "scoring_population_merge_cardinality"
        if "faltan columnas" in message or "schema" in message or "columna" in message:
            return "scoring_population_key_mismatch"
        return "scoring_population_contract_failure"
    if stage == "join_target_outcomes":
        return "target_outcome_schema_mismatch"
    if stage in {"seal_targets", "build_snapshots", "build_features"}:
        return "sealed_feature_construction_failure"
    if stage == "load_upstream_contracts":
        return "upstream_contract_failure"
    if stage == "read_source":
        return "source_read_failure"
    return "unexpected_value_error" if isinstance(exception, ValueError) else "unexpected_execution_failure"


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_and_validate_upstream_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Lee solo los tres resúmenes contractuales y verifica sus hashes exactos."""

    loaded: dict[str, dict[str, Any]] = {}
    for relative, expected in UPSTREAM_HASHES.items():
        path = ROOT / relative
        if not path.exists():
            raise ScoringContractError(f"Falta el artefacto upstream: {relative}")
        payload = path.read_bytes()
        if _sha256(payload) != expected:
            raise ScoringContractError(f"SHA-256 upstream inesperado: {relative}")
        loaded[relative] = json.loads(payload.decode("utf-8"))
    selection = loaded["reports/evidence_policy_selection_summary.json"]
    validation = loaded["reports/evidence_policy_validation_summary.json"]
    chronological = loaded["reports/chronological_validation_summary.json"]
    contract = selection.get("selected_policy_contract")
    if (
        selection.get("selection_status") != "selected_primary"
        or selection.get("selection_path") != "primary_candidate_selected"
        or selection.get("selected_policy") != POLICY_ID
        or not isinstance(contract, dict)
        or contract.get("min_points_per_direction_and_role") != MIN_POINTS
        or contract.get("min_matches_per_direction_and_role") != MIN_MATCHES
        or contract.get("scope_policy") != SCOPE_POLICY
        or contract.get("joint_role_eligibility") is not True
        or contract.get("joint_global_fallback") is not True
        or contract.get("mixed_scopes") is not False
    ):
        raise ScoringContractError("La politica seleccionada no coincide con el contrato congelado.")
    if validation.get("analysis_status") != "available":
        raise ScoringContractError("La validacion upstream no esta disponible.")
    chronological_reconciliations = chronological.get("reconciliations", {})
    if (
        chronological_reconciliations.get("source_rows") != EXPECTED_SOURCE_ROWS
        or chronological_reconciliations.get("unique_matches") != EXPECTED_MATCHES
        or chronological_reconciliations.get("split_match_sum") != EXPECTED_MATCHES
        or chronological_reconciliations.get("test_remains_sealed") is not True
    ):
        raise ScoringContractError("El protocolo cronologico upstream no reconcilia.")
    return selection, validation, chronological


def score_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica la formula congelada sin pesos adaptativos ni smoothing."""

    required = {"server_rate", "opponent_allowed_rate", "population_rate", "eligible"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas para score: {missing}")
    result = frame.copy(deep=True)
    eligible = result["eligible"].eq(True)
    for column in ("server_rate", "opponent_allowed_rate", "population_rate"):
        values = pd.to_numeric(result.loc[eligible, column], errors="coerce")
        if values.isna().any() or ~values.between(0.0, 1.0).all():
            raise ScoringContractError(f"{column} debe ser finita y pertenecer a [0, 1].")
    result["server_component"] = (result["server_rate"] - result["population_rate"]).where(eligible)
    result["opponent_component"] = (
        result["opponent_allowed_rate"] - result["population_rate"]
    ).where(eligible)
    result["combined_score"] = (
        (result["server_rate"] + result["opponent_allowed_rate"]) / 2.0
    ).where(eligible)
    result["combined_gap"] = (result["combined_score"] - result["population_rate"]).where(eligible)
    result["population_only"] = result["population_rate"].where(eligible)
    result["server_only"] = result["server_rate"].where(eligible)
    result["opponent_only"] = result["opponent_allowed_rate"].where(eligible)
    result["server_opponent_equal"] = result["combined_score"]
    identity = (
        result.loc[eligible, "combined_gap"]
        - (result.loc[eligible, "server_component"] + result.loc[eligible, "opponent_component"]) / 2.0
    ).abs()
    if not identity.le(1e-15).all():
        raise ScoringContractError("La identidad algebraica del score no reconcilia.")
    return result


def prepare_scoring_population(validation_features: pd.DataFrame) -> pd.DataFrame:
    """Une las tasas poblacionales al par global/surface sin recalcular historia."""

    required = {"target_date", "target_match_id", "target_player", "opponent", "direction", "scope", "fold"}
    if missing := sorted(required - set(validation_features.columns)):
        raise ValueError(f"Faltan columnas de poblacion scoring: {missing}")
    if not validation_features["direction"].isin(DIRECTION_ORDER).all():
        raise ValueError("Direccion fuera del contrato scoring.")
    if not validation_features["scope"].isin(("global", "surface")).all():
        raise ValueError("Scope fuera del contrato scoring.")
    dates = pd.to_datetime(validation_features["target_date"], errors="coerce").dt.normalize()
    if dates.isna().any() or not dates.between(VALIDATION_START, VALIDATION_END).all():
        raise ValueError("Fechas fuera de la validacion cronologica.")
    folds = pd.to_numeric(validation_features["fold"], errors="coerce")
    if folds.isna().any() or not folds.isin(FOLDS).all() or not folds.eq(dates.dt.year).all():
        raise ValueError("Asignacion de fold invalida.")
    merged = prepare_policy_population(validation_features)
    policy = {
        "policy_id": POLICY_ID,
        "min_points": MIN_POINTS,
        "min_matches": MIN_MATCHES,
        "scope_policy": SCOPE_POLICY,
    }
    selected = apply_policy(merged, policy)
    identifiers = [
        "target_match_id", "target_date", "target_player", "opponent",
        "target_surface", "derived_period", "direction", "fold",
    ]
    population_columns = [
        "population_direction_points", "population_direction_server_wins", "population_direction_raw_rate",
        "population_direction_last_history_date",
    ]
    for scope in ("global", "surface"):
        scoped = validation_features.loc[
            validation_features["scope"].eq(scope), identifiers + population_columns
        ].copy()
        scoped = scoped.rename(columns={column: f"{column}_{scope}" for column in population_columns})
        selected = selected.merge(scoped, on=identifiers, how="left", validate="one_to_one", sort=False)
    selected["population_rate"] = np.select(
        [selected["selected_scope"].eq("surface"), selected["selected_scope"].eq("global")],
        [
            selected["population_direction_raw_rate_surface"],
            selected["population_direction_raw_rate_global"],
        ],
        default=np.nan,
    )
    for role, rate in (("server", "server_selected_rate"), ("opponent", "opponent_selected_rate")):
        selected[f"{role}_rate"] = selected[rate]
        for measure in ("direction_points", "direction_matches"):
            selected[f"{role}_history_{measure}"] = np.select(
                [selected["selected_scope"].eq("surface"), selected["selected_scope"].eq("global")],
                [selected[f"{role}_{measure}_surface"], selected[f"{role}_{measure}_global"]],
                default=np.nan,
            )
    selected["opponent_allowed_rate"] = selected["opponent_selected_rate"]
    selected["evidence_status"] = np.select(
        [selected["selected_scope"].eq("surface"), selected["selected_scope"].eq("global")],
        ["eligible_surface", "eligible_global_fallback"],
        default="insufficient_surface_and_global",
    )
    selected["abstention_reason"] = np.where(selected["eligible"], None, "insufficient_surface_and_global")
    selected = score_components(selected)
    expected_key = ["target_match_id", "target_player", "direction"]
    if selected.duplicated(expected_key).any():
        raise ScoringContractError("La clave orientacion-direccion no es unica tras aplicar politica.")
    if not selected.loc[selected["eligible"], "selected_scope"].isin(SCOPE_ORDER).all():
        raise ScoringContractError("Scope seleccionado fuera del contrato.")
    return selected.sort_values(
        ["target_date", "target_match_id", "target_player", "direction"], kind="stable"
    ).reset_index(drop=True)


def join_target_outcomes(scored_orientations: pd.DataFrame, eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asocia scores prepartido a puntos elegibles 2020--2023, sin usar su outcome."""

    work = eligible.copy(deep=True)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    if work["date"].isna().any():
        raise ScoringContractError("Fechas invalidas en outcomes target.")
    target = work.loc[work["date"].between(VALIDATION_START, VALIDATION_END)].copy()
    target["direction"] = target["direction"].replace({"t": "T"})
    if not target["direction"].isin(DIRECTION_ORDER).all():
        raise ScoringContractError("Direccion target fuera del contrato.")
    target["fold"] = target["date"].dt.year.astype("int64")
    target = target.rename(columns={
        "match_id": "target_match_id", "server_player": "target_player", "surface": "target_surface",
    })
    keys = ["target_match_id", "target_player", "returner_player", "target_surface", "direction", "fold"]
    scored = scored_orientations.rename(columns={"opponent": "returner_player"})
    keep = keys + [
        "target_date", "selected_scope", "eligible", "evidence_status", "abstention_reason",
        "server_history_direction_points", "server_history_direction_matches",
        "opponent_history_direction_points", "opponent_history_direction_matches",
        "server_rate", "opponent_rate", "population_rate", "server_component", "opponent_component",
        "combined_score", "combined_gap", *SCORERS,
    ]
    joined = target.merge(scored[keep], on=keys, how="left", validate="many_to_one", sort=False)
    if joined["eligible"].isna().any():
        raise ScoringContractError("Un punto target no encontro su orientacion-direccion prepartido.")
    if joined["target_date"].notna().any() and not joined["target_date"].eq(joined["date"]).all():
        raise ScoringContractError("La fecha del score no coincide con la del target.")
    joined = joined.rename(columns={"server_won_point": "outcome"})
    outcome_values = joined["outcome"].tolist()
    if any(type(value) is not bool and not isinstance(value, np.bool_) for value in outcome_values):
        raise ScoringContractError("El outcome target debe contener exclusivamente booleanos reales.")
    joined["orientation_id"] = joined["target_match_id"].astype(str) + "|" + joined["target_player"].astype(str)
    scored_points = joined.loc[joined["eligible"].eq(True)].copy()
    if scored_points.empty:
        raise ScoringContractError("La politica no permite puntuar ningun punto de validacion.")
    values = scored_points.loc[:, list(SCORERS)].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ScoringContractError("Los scores elegibles deben ser finitos y pertenecer a [0, 1].")
    return joined.sort_values(["date", "target_match_id", "point_number"], kind="stable").reset_index(drop=True), scored_points


def _score_long(scored_points: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "target_match_id", "date", "fold", "target_player", "returner_player", "orientation_id",
        "direction", "selected_scope", "outcome", "server_component", "opponent_component", "combined_gap",
    ]
    long = scored_points.melt(
        id_vars=identifiers, value_vars=list(SCORERS), var_name="scorer", value_name="score"
    )
    long["squared_error"] = (long["score"] - long["outcome"].astype(float)) ** 2
    probability = np.clip(long["score"].to_numpy(dtype="float64"), EPSILON, 1.0 - EPSILON)
    outcome = long["outcome"].to_numpy(dtype="float64")
    long["log_loss_value"] = -(outcome * np.log(probability) + (1.0 - outcome) * np.log(1.0 - probability))
    return long


def _safe_quantile(values: pd.Series, probability: float) -> float | None:
    array = values.to_numpy(dtype="float64")
    array = array[np.isfinite(array)]
    return None if not len(array) else float(np.quantile(array, probability, method="linear"))


def _metric_row(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {key: None for key in ("mean_score", "observed_rate", "calibration_gap", "brier_score", "log_loss")}
    mean_score = float(group["score"].mean())
    observed = float(group["outcome"].mean())
    return {
        "n_points": int(len(group)),
        "n_matches": int(group["target_match_id"].nunique()),
        "n_orientations": int(group["orientation_id"].nunique()),
        "mean_score": mean_score,
        "observed_rate": observed,
        "calibration_gap": mean_score - observed,
        "brier_score": float(group["squared_error"].mean()),
        "log_loss": float(group["log_loss_value"].mean()),
    }


def build_calibration(long: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, float | None]]]:
    """Construye 10 bins fijos por scorer y fold, mas una fila pooled."""

    rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str], dict[str, float | None]] = {}
    for scorer in SCORERS:
        scorer_rows = long.loc[long["scorer"].eq(scorer)]
        for label, group in [(str(fold), scorer_rows.loc[scorer_rows["fold"].eq(fold)]) for fold in FOLDS] + [("pooled", scorer_rows)]:
            total = len(group)
            if total:
                indexes = np.minimum((group["score"].to_numpy(dtype=float) * BIN_COUNT).astype(int), BIN_COUNT - 1)
            else:
                indexes = np.array([], dtype=int)
            absolute_gaps: list[tuple[int, float]] = []
            for bin_index in range(BIN_COUNT):
                part = group.loc[indexes == bin_index] if total else group.iloc[0:0]
                low, high = bin_index / BIN_COUNT, (bin_index + 1) / BIN_COUNT
                if part.empty:
                    rows.append({"scorer": scorer, "fold_or_pooled": label, "bin_index": bin_index, "bin_low": low, "bin_high": high, "n_points": 0, "n_matches": 0, "mean_score": None, "observed_rate": None, "absolute_gap": None})
                    continue
                mean_score, observed = float(part["score"].mean()), float(part["outcome"].mean())
                gap = abs(mean_score - observed)
                absolute_gaps.append((len(part), gap))
                rows.append({"scorer": scorer, "fold_or_pooled": label, "bin_index": bin_index, "bin_low": low, "bin_high": high, "n_points": int(len(part)), "n_matches": int(part["target_match_id"].nunique()), "mean_score": mean_score, "observed_rate": observed, "absolute_gap": gap})
            summaries[(scorer, label)] = {
                "ece": None if total == 0 else float(sum(count / total * gap for count, gap in absolute_gaps)),
                "max_calibration_error": None if not absolute_gaps else float(max(gap for _, gap in absolute_gaps)),
            }
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS), summaries


def build_probability_tables(long: pd.DataFrame, all_target_points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration, calibration_summary = build_calibration(long)
    coverage = (
        all_target_points.groupby(["fold", "direction"], sort=False, observed=True)
        .agg(eligible_target_points=("point_number", "size"), scored_target_points=("eligible", "sum"))
        .reset_index()
    )
    coverage["abstained_target_points"] = coverage["eligible_target_points"] - coverage["scored_target_points"]
    coverage["point_coverage"] = coverage["scored_target_points"] / coverage["eligible_target_points"]
    rows: list[dict[str, Any]] = []
    for keys, group in long.groupby(["scorer", "fold", "direction", "selected_scope"], sort=False, observed=True):
        scorer, fold, direction, scope = keys
        row = {"scorer": scorer, "fold": int(fold), "direction": direction, "selected_scope": scope, **_metric_row(group)}
        overall = coverage.loc[coverage["fold"].eq(fold) & coverage["direction"].eq(direction)].iloc[0]
        row.update({key: int(overall[key]) if key.endswith("points") else float(overall[key]) for key in ("eligible_target_points", "scored_target_points", "abstained_target_points", "point_coverage")})
        calibration_values = calibration_summary[(scorer, str(fold))]
        row["calibration_in_the_large"] = row["calibration_gap"]
        row.update(calibration_values)
        rows.append(row)
    by_fold = pd.DataFrame(rows)
    for metric, delta in (("brier_score", "delta_brier_vs_population"), ("log_loss", "delta_log_loss_vs_population"), ("ece", "delta_ece_vs_population")):
        population = by_fold.loc[by_fold["scorer"].eq("population_only"), ["fold", "direction", "selected_scope", metric]].rename(columns={metric: "control"})
        by_fold = by_fold.merge(population, on=["fold", "direction", "selected_scope"], how="left", validate="many_to_one")
        by_fold[delta] = by_fold[metric] - by_fold["control"]
        by_fold = by_fold.drop(columns="control")
    by_fold["_scorer"] = pd.Categorical(by_fold["scorer"], SCORERS, ordered=True)
    by_fold["_direction"] = pd.Categorical(by_fold["direction"], DIRECTION_ORDER, ordered=True)
    by_fold["_scope"] = pd.Categorical(by_fold["selected_scope"], SCOPE_ORDER, ordered=True)
    by_fold = by_fold.sort_values(["_scorer", "fold", "_direction", "_scope"], kind="stable").drop(columns=["_scorer", "_direction", "_scope"]).reset_index(drop=True)

    direction_rows: list[dict[str, Any]] = []
    for scorer in SCORERS:
        scorer_long = long.loc[long["scorer"].eq(scorer)]
        for direction in DIRECTION_ORDER:
            direction_long = scorer_long.loc[scorer_long["direction"].eq(direction)]
            for label, group in [(str(fold), direction_long.loc[direction_long["fold"].eq(fold)]) for fold in FOLDS] + [("pooled", direction_long)]:
                metrics = _metric_row(group)
                direction_rows.append({
                    "scorer": scorer, "direction": direction, "fold_or_pooled": label,
                    **metrics,
                    "server_component_mean": None if group.empty else float(group["server_component"].mean()),
                    "opponent_component_mean": None if group.empty else float(group["opponent_component"].mean()),
                    "combined_gap_mean": None if group.empty else float(group["combined_gap"].mean()),
                })
    by_direction = pd.DataFrame(direction_rows, columns=BY_DIRECTION_COLUMNS)
    by_direction["_scorer"] = pd.Categorical(by_direction["scorer"], SCORERS, ordered=True)
    by_direction["_direction"] = pd.Categorical(by_direction["direction"], DIRECTION_ORDER, ordered=True)
    by_direction["_fold"] = pd.Categorical(by_direction["fold_or_pooled"], [str(fold) for fold in FOLDS] + ["pooled"], ordered=True)
    by_direction = by_direction.sort_values(["_scorer", "_direction", "_fold"], kind="stable").drop(columns=["_scorer", "_direction", "_fold"]).reset_index(drop=True)
    return by_fold[BY_FOLD_COLUMNS], by_direction, calibration


def _safe_spearman(scores: pd.Series, observed: pd.Series) -> tuple[float | None, str | None]:
    if len(scores) < 2:
        return None, "insufficient_directions"
    if scores.nunique(dropna=False) < 2 or observed.nunique(dropna=False) < 2:
        return None, "constant_values"
    value = scores.rank(method="average").corr(observed.rank(method="average"), method="pearson")
    return (None, "undefined_correlation") if pd.isna(value) else (float(value), None)


def build_ranking(long: pd.DataFrame) -> pd.DataFrame:
    """Ranking descriptivo solo donde las tres direcciones tienen score y outcome."""

    orientation = (
        long.groupby(["scorer", "fold", "target_match_id", "target_player", "orientation_id", "direction"], sort=False, observed=True)
        .agg(score=("score", "first"), observed_rate=("outcome", "mean"))
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    direction_rank = {direction: index for index, direction in enumerate(DIRECTION_ORDER)}
    for scorer in SCORERS:
        scorer_rows = orientation.loc[orientation["scorer"].eq(scorer)]
        for label, pool in [(str(fold), scorer_rows.loc[scorer_rows["fold"].eq(fold)]) for fold in FOLDS] + [("pooled", scorer_rows)]:
            details: list[dict[str, Any]] = []
            for _, group in pool.groupby(["target_match_id", "target_player", "orientation_id"], sort=False):
                if set(group["direction"]) != set(DIRECTION_ORDER) or len(group) != 3:
                    continue
                ordered = group.assign(_order=group["direction"].map(direction_rank)).sort_values(["score", "_order"], ascending=[False, True], kind="stable")
                best = float(group["observed_rate"].max())
                optimal = set(group.loc[group["observed_rate"].eq(best), "direction"])
                predicted = str(ordered.iloc[0]["direction"])
                rank_optimal = min(int(index + 1) for index, value in enumerate(ordered["direction"]) if value in optimal)
                spearman, reason = _safe_spearman(group["score"], group["observed_rate"])
                details.append({
                    "match_id": str(group["target_match_id"].iloc[0]), "predicted": predicted,
                    "optimal": optimal, "strict": len(optimal) == 1, "top1": predicted in optimal,
                    "mrr": 1.0 / rank_optimal,
                    "regret": best - float(ordered.iloc[0]["observed_rate"]),
                    "spearman": spearman, "spearman_reason": reason,
                })
            if not details:
                rows.append({"scorer": scorer, "fold_or_pooled": label, "comparable_orientations": 0, "comparable_matches": 0, "top1_hit_inclusive": None, "top1_hit_strict": None, "strict_winner_orientations": 0, "mrr": None, "spearman": None, "spearman_reason_code": "no_comparable_orientations", "mean_regret": None, "median_regret": None, "p90_regret": None, "regret_at_most_005": None, "regret_at_most_010": None})
                continue
            detail = pd.DataFrame(details)
            valid_spearman = detail["spearman"].dropna()
            strict = detail.loc[detail["strict"]]
            rows.append({
                "scorer": scorer, "fold_or_pooled": label,
                "comparable_orientations": int(len(detail)), "comparable_matches": int(detail["match_id"].nunique()),
                "top1_hit_inclusive": float(detail["top1"].mean()),
                "top1_hit_strict": None if strict.empty else float(strict["top1"].mean()),
                "strict_winner_orientations": int(len(strict)), "mrr": float(detail["mrr"].mean()),
                "spearman": None if valid_spearman.empty else float(valid_spearman.mean()),
                "spearman_reason_code": "all_undefined" if valid_spearman.empty else None,
                "mean_regret": float(detail["regret"].mean()), "median_regret": _safe_quantile(detail["regret"], .5),
                "p90_regret": _safe_quantile(detail["regret"], .9),
                "regret_at_most_005": float(detail["regret"].le(.05).mean()),
                "regret_at_most_010": float(detail["regret"].le(.10).mean()),
            })
    ranking = pd.DataFrame(rows, columns=RANKING_COLUMNS)
    ranking["_scorer"] = pd.Categorical(ranking["scorer"], SCORERS, ordered=True)
    ranking["_fold"] = pd.Categorical(ranking["fold_or_pooled"], [str(fold) for fold in FOLDS] + ["pooled"], ordered=True)
    return ranking.sort_values(["_scorer", "_fold"], kind="stable").drop(columns=["_scorer", "_fold"]).reset_index(drop=True)


def build_paired_match_comparison(long: pd.DataFrame) -> dict[str, Any]:
    main = long.loc[long["scorer"].eq(MAIN_SCORER)].groupby("target_match_id", sort=False)["squared_error"].mean().rename("main")
    result: dict[str, Any] = {}
    for control in ("population_only", "server_only", "opponent_only"):
        other = long.loc[long["scorer"].eq(control)].groupby("target_match_id", sort=False)["squared_error"].mean().rename("control")
        difference = main.to_frame().join(other, how="inner")["main"] - main.to_frame().join(other, how="inner")["control"]
        result[control] = {
            "unit": "target_match", "pairs": int(len(difference)), "mean_difference_main_minus_control": float(difference.mean()),
            "median_difference_main_minus_control": _safe_quantile(difference, .5), "p25": _safe_quantile(difference, .25), "p75": _safe_quantile(difference, .75),
            "main_lower_brier_proportion": float(difference.lt(0).mean()),
        }
    return result


def build_explainability(scored_points: pd.DataFrame) -> dict[str, Any]:
    orientations = scored_points.drop_duplicates(["target_match_id", "target_player", "direction"])
    server = orientations["server_component"]
    opponent = orientations["opponent_component"]
    dominant = np.where(server.abs().gt(opponent.abs()), "server", np.where(opponent.abs().gt(server.abs()), "opponent", "tie"))
    def distribution(values: pd.Series) -> dict[str, float]:
        return {"minimum": float(values.min()), "p25": _safe_quantile(values, .25), "median": _safe_quantile(values, .5), "p75": _safe_quantile(values, .75), "maximum": float(values.max())}
    return {
        "unit": "eligible_target_match_player_direction", "orientations": int(len(orientations)),
        "server_component_positive": int(server.gt(0).sum()), "server_component_negative": int(server.lt(0).sum()),
        "opponent_component_positive": int(opponent.gt(0).sum()), "opponent_component_negative": int(opponent.lt(0).sum()),
        "both_positive": int((server.gt(0) & opponent.gt(0)).sum()), "both_negative": int((server.lt(0) & opponent.lt(0)).sum()),
        "opposite_signs": int((server * opponent).lt(0).sum()),
        "dominant_component": {key: int((dominant == key).sum()) for key in ("server", "opponent", "tie")},
        "server_component_distribution": distribution(server), "opponent_component_distribution": distribution(opponent),
        "selected_scope": {str(key): int(value) for key, value in orientations["selected_scope"].value_counts().sort_index().items()},
        "equation": "population baseline + 0.5 * server component + 0.5 * opponent component = combined score",
    }


def determine_baseline_status(long: pd.DataFrame) -> tuple[str, list[str], dict[str, Any]]:
    pooled = {scorer: _metric_row(long.loc[long["scorer"].eq(scorer)]) for scorer in SCORERS}
    by_fold = {
        (scorer, fold): _metric_row(long.loc[long["scorer"].eq(scorer) & long["fold"].eq(fold)])
        for scorer in SCORERS for fold in FOLDS
    }
    main, population, server, opponent = (pooled[name] for name in (MAIN_SCORER, "population_only", "server_only", "opponent_only"))
    folds_not_worse = sum(by_fold[(MAIN_SCORER, fold)]["brier_score"] <= by_fold[("population_only", fold)]["brier_score"] for fold in FOLDS)
    checks = {
        "main_brier_not_worse_than_population": main["brier_score"] <= population["brier_score"],
        "main_brier_not_worse_than_worst_individual": main["brier_score"] <= max(server["brier_score"], opponent["brier_score"]),
        "folds_brier_not_worse_than_population": int(folds_not_worse),
        "at_least_three_folds_not_worse": folds_not_worse >= 3,
        "main_log_loss_not_worse_than_population": main["log_loss"] <= population["log_loss"],
    }
    failures = [key for key, value in checks.items() if value is False]
    return ("validated_descriptive_baseline" if not failures else "not_validated"), failures, {"pooled": pooled, "fold_checks": checks}


def _coverage_summary(all_points: pd.DataFrame, scored_points: pd.DataFrame) -> dict[str, Any]:
    complete = scored_points.drop_duplicates(["target_match_id", "target_player", "direction"])
    sets = complete.groupby(["target_match_id", "target_player"], sort=False)["direction"].agg(lambda values: set(values))
    complete_orientation = sets.eq(set(DIRECTION_ORDER))
    complete_by_match = complete_orientation.groupby(level=0).agg(lambda values: len(values) == 2 and values.all())
    return {
        "eligible_target_points": int(len(all_points)), "scored_target_points": int(len(scored_points)),
        "abstained_target_points": int(len(all_points) - len(scored_points)), "point_coverage": float(len(scored_points) / len(all_points)),
        "target_orientations": int(all_points[["target_match_id", "target_player"]].drop_duplicates().shape[0]),
        "scored_orientations": int(scored_points[["target_match_id", "target_player"]].drop_duplicates().shape[0]),
        "complete_direction_sets": int(complete_orientation.sum()), "complete_matches": int(complete_by_match.sum()),
        "surface_selected_orientations": int(scored_points.drop_duplicates(["target_match_id", "target_player", "direction"]).loc[lambda x: x["selected_scope"].eq("surface")].shape[0]),
        "global_fallback_orientations": int(scored_points.drop_duplicates(["target_match_id", "target_player", "direction"]).loc[lambda x: x["selected_scope"].eq("global")].shape[0]),
    }


def build_summary(
    *, selection: Mapping[str, Any], source_audit: Mapping[str, Any], sealed_audit: Mapping[str, Any],
    all_points: pd.DataFrame, scored_points: pd.DataFrame, long: pd.DataFrame,
    by_fold: pd.DataFrame, calibration: pd.DataFrame, ranking: pd.DataFrame,
    status: str, reasons: list[str], validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = _coverage_summary(all_points, scored_points)
    validation_population = {
        "matches": int(all_points["target_match_id"].nunique()), "orientations": int(all_points[["target_match_id", "target_player"]].drop_duplicates().shape[0]),
        "substantive_second_serve_points": int(all_points["point_number"].size), "recognized_direction_points": int(all_points["point_number"].size),
        "server_wins": int(all_points["outcome"].sum()),
        "directions": {direction: int(all_points["direction"].eq(direction).sum()) for direction in DIRECTION_ORDER},
    }
    identity_error = (scored_points["combined_gap"] - (scored_points["server_component"] + scored_points["opponent_component"]) / 2.0).abs()
    identity_tolerance = 1e-15
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "source": {"path": "data/processed/points_enriched.parquet", "columns_used": list(SOURCE_COLUMNS), "single_read": True, "source_point_rows": int(source_audit["source_point_rows"])},
        "upstream_commits": dict(UPSTREAM_COMMITS), "upstream_hashes": dict(UPSTREAM_HASHES),
        "selected_policy_contract": selection["selected_policy_contract"],
        "scorer_contracts": {
            "population_only": "population_rate", "server_only": "server_rate", "opponent_only": "opponent_allowed_rate", "server_opponent_equal": "(server_rate + opponent_allowed_rate) / 2",
            "primary_scorer": MAIN_SCORER, "weights_fixed": {"server": .5, "opponent": .5},
        },
        "formula_contract": {"server_component": "server_rate - population_rate", "opponent_component": "opponent_allowed_rate - population_rate", "combined_score": "(server_rate + opponent_allowed_rate) / 2", "combined_gap": "combined_score - population_rate", "identity": "combined_gap = (server_component + opponent_component) / 2", "smoothing": False, "imputation": False},
        "formula_identity_max_abs_error": float(identity_error.max()),
        "formula_identity_tolerance": identity_tolerance,
        "formula_identity_rows_checked": int(len(identity_error)),
        "sealed_test_contract": {"test_status": "sealed", "test_target_matches_constructed": 0, "test_feature_rows_constructed": 0, "test_points_scored": 0, "test_matches_scored": 0, "test_evaluation_runs": 0, "used_for_method_selection": False, "test_target_matches_excluded_before_construction": int(sealed_audit["test_target_matches_excluded_before_construction"])},
        "population": {"source_matches": int(source_audit["source_matches"]), "eligible_history_points": int(source_audit["eligible_points"]), "constructed_target_matches": int(sealed_audit["constructed_target_matches"]), "constructed_snapshot_rows": int(sealed_audit["constructed_snapshot_rows"]), "constructed_feature_rows": int(sealed_audit["constructed_feature_rows"])},
        "validation_population": validation_population, "coverage": coverage,
        "probability_metrics": validation_metrics["pooled"],
        "calibration": {"bins": BIN_COUNT, "definition": "fixed_width_0.1", "rows": int(len(calibration)), "empty_bins_retained": True, "log_loss_epsilon": EPSILON},
        "ranking": {"rows": int(len(ranking)), "unit": "target_match_id x target_player", "tie_break": list(DIRECTION_ORDER), "noise_limit": "Las tasas target por direccion pueden basarse en pocos puntos; ranking descriptivo secundario."},
        "paired_match_comparison": build_paired_match_comparison(long), "explainability": build_explainability(scored_points),
        "validation_criteria": {"contract_and_temporality": True, "scores_finite_in_unit_interval": True, **validation_metrics["fold_checks"]},
        "baseline_status": status, "reason_codes": sorted(reasons),
        "reconciliations": {"upstream_hashes_exact": True, "policy_exact": True, "surface_then_global_atomic": True, "targets_no_later_than_2023_12_31": True, "folds_exact_2020_2023": True, "one_parquet_read": True, "one_snapshot_construction": True, "one_feature_construction": True, "two_orientations_per_match": True, "directions_exact": True, "feature_keys_unique": True, "strict_history_prior": True, "zero_self_reference": True, "zero_temporal_violations": True, "outcome_binary": True, "scores_finite_in_unit_interval": True, "formula_identity_reconciled": bool(identity_error.le(identity_tolerance).all()), "calibration_bins_complete": len(calibration) == len(SCORERS) * (len(FOLDS) + 1) * BIN_COUNT, "ranking_only_comparable_orientations": True, "summary_and_csv_canonical": True, "no_test_contamination": True},
        "methodological_limits": ["Baseline descriptivo basado en asociaciones historicas; no es causal.", "Los pesos 0.5/0.5 no fueron ajustados a los resultados.", "El ranking secundario puede ser ruidoso por pocos puntos target por direccion.", "No se generan recomendaciones tacticas automaticas."],
    }
    summary = _json_value(summary)
    _assert_finite_json(summary)
    return summary


def _fingerprint(summary: Mapping[str, Any], payloads_or_frames: tuple[bytes, ...] | tuple[pd.DataFrame, ...]) -> str:
    """Fingerprint v2 sobre metadatos y hashes de bytes CSV, nunca sobre CSV reabiertos."""

    if payloads_or_frames and isinstance(payloads_or_frames[0], pd.DataFrame):
        payloads = _csv_payloads(payloads_or_frames)  # Compatibilidad para fixtures en memoria.
    else:
        payloads = payloads_or_frames  # type: ignore[assignment]
    payload_hashes, payload_sizes = _payload_metadata(payloads)
    core = {key: value for key, value in summary.items() if key != "publication_fingerprint"}
    if core.get("fingerprint_contract_version") != FINGERPRINT_CONTRACT_VERSION:
        raise ScoringContractError("Version de contrato de fingerprint inesperada.")
    if core.get("artifact_payload_sha256") != payload_hashes or core.get("artifact_payload_bytes") != payload_sizes:
        raise ScoringContractError("Metadatos de payload CSV no reconcilian.")
    canonical = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256(canonical)


def _finalize_summary(summary: Mapping[str, Any], frames: tuple[pd.DataFrame, ...]) -> tuple[dict[str, Any], tuple[bytes, ...], str]:
    payloads = _csv_payloads(frames)
    payload_hashes, payload_sizes = _payload_metadata(payloads)
    finalized = dict(summary)
    finalized["fingerprint_contract_version"] = FINGERPRINT_CONTRACT_VERSION
    finalized["artifact_payload_sha256"] = payload_hashes
    finalized["artifact_payload_bytes"] = payload_sizes
    fingerprint = _fingerprint(finalized, payloads)
    finalized["publication_fingerprint"] = fingerprint
    return finalized, payloads, fingerprint


def _require_close(actual: Any, expected: Any, field: str, *, tolerance: float = 1e-12) -> None:
    if actual is None or expected is None or not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ScoringContractError(f"Reconciliacion semantica fallida: {field}.")


def _require_exact_bool(value: Any, field: str) -> None:
    if type(value) is not bool:
        raise ScoringContractError(f"{field} debe ser un booleano JSON estricto.")


def _validate_aggregate_semantics(result: ScoringResult) -> None:
    """Reconciliaciones independientes de los agregados publicados."""

    summary, by_fold, by_direction, calibration, ranking = (
        result.summary, result.by_fold, result.by_direction, result.calibration, result.ranking
    )
    expected_scorers = list(SCORERS)
    policy = summary.get("selected_policy_contract", {})
    required_policy = {
        "policy_id": POLICY_ID,
        "min_points_per_direction_and_role": MIN_POINTS,
        "min_matches_per_direction_and_role": MIN_MATCHES,
        "scope_policy": SCOPE_POLICY,
        "joint_role_eligibility": True,
        "joint_global_fallback": True,
        "mixed_scopes": False,
    }
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise ScoringContractError("Contrato de politica seleccionada alterado.")
    if set(by_fold["scorer"]) != set(SCORERS) or set(by_direction["scorer"]) != set(SCORERS):
        raise ScoringContractError("Los scorers publicados no coinciden con el contrato.")
    if by_fold.duplicated(["scorer", "fold", "direction", "selected_scope"]).any():
        raise ScoringContractError("Clave duplicada en by_fold.")
    if by_direction.duplicated(["scorer", "direction", "fold_or_pooled"]).any():
        raise ScoringContractError("Clave duplicada en by_direction.")
    if calibration.duplicated(["scorer", "fold_or_pooled", "bin_index"]).any():
        raise ScoringContractError("Clave duplicada en calibration.")
    if ranking.duplicated(["scorer", "fold_or_pooled"]).any():
        raise ScoringContractError("Clave duplicada en ranking.")
    if set(by_fold["fold"]) != set(FOLDS) or set(by_fold["direction"]) != set(DIRECTION_ORDER):
        raise ScoringContractError("Folds o direcciones inesperados en by_fold.")
    if set(by_fold["selected_scope"]) - set(SCOPE_ORDER):
        raise ScoringContractError("Scope publicado fuera del contrato.")
    def canonical(frame: pd.DataFrame, columns: list[str], orders: list[tuple[str, tuple[Any, ...]]]) -> pd.DataFrame:
        work = frame.copy(deep=True)
        temporary: list[str] = []
        for column, values in orders:
            name = f"__{column}_order"
            work[name] = pd.Categorical(work[column].astype(str), [str(value) for value in values], ordered=True)
            temporary.append(name)
        return work.sort_values(temporary, kind="stable").drop(columns=temporary).reset_index(drop=True)[columns]
    if not by_fold.reset_index(drop=True).equals(canonical(by_fold, BY_FOLD_COLUMNS, [("scorer", SCORERS), ("fold", FOLDS), ("direction", DIRECTION_ORDER), ("selected_scope", SCOPE_ORDER)])):
        raise ScoringContractError("Orden canonical de by_fold alterado.")
    if not by_direction.reset_index(drop=True).equals(canonical(by_direction, BY_DIRECTION_COLUMNS, [("scorer", SCORERS), ("direction", DIRECTION_ORDER), ("fold_or_pooled", (*map(str, FOLDS), "pooled"))])):
        raise ScoringContractError("Orden canonical de by_direction alterado.")
    if not calibration.reset_index(drop=True).equals(canonical(calibration, CALIBRATION_COLUMNS, [("scorer", SCORERS), ("fold_or_pooled", (*map(str, FOLDS), "pooled")), ("bin_index", tuple(range(BIN_COUNT)))])):
        raise ScoringContractError("Orden canonical de calibration alterado.")
    if not ranking.reset_index(drop=True).equals(canonical(ranking, RANKING_COLUMNS, [("scorer", SCORERS), ("fold_or_pooled", (*map(str, FOLDS), "pooled"))])):
        raise ScoringContractError("Orden canonical de ranking alterado.")
    fold_keys = by_fold.loc[by_fold["scorer"].eq(SCORERS[0]), ["fold", "direction", "selected_scope"]]
    for scorer in SCORERS[1:]:
        candidate = by_fold.loc[by_fold["scorer"].eq(scorer), ["fold", "direction", "selected_scope"]]
        if set(map(tuple, candidate.to_numpy())) != set(map(tuple, fold_keys.to_numpy())):
            raise ScoringContractError("Los scorers no comparten las claves by_fold.")
    expected_direction_keys = {(scorer, direction, label) for scorer in SCORERS for direction in DIRECTION_ORDER for label in (*map(str, FOLDS), "pooled")}
    if set(map(tuple, by_direction[["scorer", "direction", "fold_or_pooled"]].astype(str).to_numpy())) != expected_direction_keys:
        raise ScoringContractError("Dominio o cardinalidad de by_direction inesperado.")
    expected_calibration_keys = {(scorer, label, bin_index) for scorer in SCORERS for label in (*map(str, FOLDS), "pooled") for bin_index in range(BIN_COUNT)}
    actual_calibration_keys = {(str(row.scorer), str(row.fold_or_pooled), int(row.bin_index)) for row in calibration.itertuples(index=False)}
    if actual_calibration_keys != expected_calibration_keys:
        raise ScoringContractError("Dominio o cardinalidad de calibration inesperado.")
    expected_ranking_keys = {(scorer, label) for scorer in SCORERS for label in (*map(str, FOLDS), "pooled")}
    if set(map(tuple, ranking[["scorer", "fold_or_pooled"]].astype(str).to_numpy())) != expected_ranking_keys:
        raise ScoringContractError("Dominio o cardinalidad de ranking inesperado.")

    formula = summary.get("formula_contract")
    expected_formula = {
        "server_component": "server_rate - population_rate",
        "opponent_component": "opponent_allowed_rate - population_rate",
        "combined_score": "(server_rate + opponent_allowed_rate) / 2",
        "combined_gap": "combined_score - population_rate",
        "identity": "combined_gap = (server_component + opponent_component) / 2",
        "smoothing": False, "imputation": False,
    }
    if formula != expected_formula or summary.get("scorer_contracts", {}).get("primary_scorer") != MAIN_SCORER:
        raise ScoringContractError("Contrato de formula o scorer principal alterado.")
    contracts = summary.get("scorer_contracts", {})
    if contracts.get("weights_fixed") != {"server": .5, "opponent": .5} or {key for key in contracts if key in SCORERS} != set(SCORERS):
        raise ScoringContractError("Pesos o scorers del contrato alterados.")
    identity_tolerance = summary.get("formula_identity_tolerance")
    identity_error = summary.get("formula_identity_max_abs_error")
    identity_rows = summary.get("formula_identity_rows_checked")
    if not isinstance(identity_rows, int) or identity_rows <= 0 or not isinstance(identity_error, (int, float)) or not isinstance(identity_tolerance, (int, float)):
        raise ScoringContractError("Diagnostico de identidad algebraica incompleto.")
    if not math.isfinite(float(identity_error)) or float(identity_error) < 0 or float(identity_error) > float(identity_tolerance):
        raise ScoringContractError("Error de identidad algebraica fuera de tolerancia.")

    coverage_rows = by_fold.loc[by_fold["scorer"].eq(SCORERS[0])].copy()
    coverage_columns = ["eligible_target_points", "scored_target_points", "abstained_target_points", "point_coverage"]
    if coverage_rows.duplicated(["fold", "direction", "selected_scope"]).any():
        raise ScoringContractError("Cobertura duplicada en el scorer de referencia.")
    for keys, group in by_fold.groupby(["fold", "direction"], sort=False, observed=True):
        if any(group[column].nunique(dropna=False) != 1 for column in coverage_columns):
            raise ScoringContractError(f"Cobertura inconsistente en fold-direccion {keys}.")
    coverage_unique = coverage_rows.drop_duplicates(["fold", "direction"])
    eligible_points = int(coverage_unique["eligible_target_points"].sum())
    scored_points = int(coverage_unique["scored_target_points"].sum())
    abstained_points = int(coverage_unique["abstained_target_points"].sum())
    if eligible_points != scored_points + abstained_points:
        raise ScoringContractError("Cobertura no reconcilia puntos elegibles/scored/abstenciones.")
    coverage = summary.get("coverage", {})
    for key, value in (("eligible_target_points", eligible_points), ("scored_target_points", scored_points), ("abstained_target_points", abstained_points)):
        if coverage.get(key) != value:
            raise ScoringContractError(f"Summary de cobertura alterado: {key}.")
    _require_close(coverage.get("point_coverage"), scored_points / eligible_points, "coverage.point_coverage")
    validation_population = summary.get("validation_population", {})
    if validation_population.get("recognized_direction_points") != eligible_points or validation_population.get("substantive_second_serve_points") != eligible_points:
        raise ScoringContractError("Poblacion de validacion no reconcilia cobertura.")
    direction_totals = coverage_unique.groupby("direction", observed=True)["eligible_target_points"].sum().to_dict()
    if validation_population.get("directions") != {direction: int(direction_totals[direction]) for direction in DIRECTION_ORDER}:
        raise ScoringContractError("Totales de direccion no reconcilian.")

    metric_columns = ("mean_score", "observed_rate", "calibration_gap", "brier_score", "log_loss")
    for scorer in SCORERS:
        scorer_rows = by_direction.loc[by_direction["scorer"].eq(scorer)]
        for direction in DIRECTION_ORDER:
            rows = scorer_rows.loc[scorer_rows["direction"].eq(direction)]
            annual = rows.loc[rows["fold_or_pooled"].astype(str).isin(map(str, FOLDS))]
            pooled = rows.loc[rows["fold_or_pooled"].astype(str).eq("pooled")].iloc[0]
            if int(annual["n_points"].sum()) != int(pooled["n_points"]):
                raise ScoringContractError("Pooled por direccion no reconcilia puntos por fold.")
            for metric in metric_columns:
                _require_close(pooled[metric], np.average(annual[metric], weights=annual["n_points"]), f"by_direction.{scorer}.{direction}.{metric}")
        pooled = scorer_rows.loc[scorer_rows["fold_or_pooled"].astype(str).eq("pooled")]
        weights = pooled["n_points"].to_numpy(dtype=float)
        published = summary.get("probability_metrics", {}).get(scorer, {})
        if int(weights.sum()) != published.get("n_points"):
            raise ScoringContractError(f"n_points pooled alterado para {scorer}.")
        for metric in metric_columns:
            _require_close(published.get(metric), np.average(pooled[metric], weights=weights), f"summary.{scorer}.{metric}")
        for label in (*map(str, FOLDS), "pooled"):
            bins = calibration.loc[(calibration["scorer"].eq(scorer)) & (calibration["fold_or_pooled"].astype(str).eq(label))].sort_values("bin_index")
            if list(bins["bin_index"]) != list(range(BIN_COUNT)) or not np.allclose(bins["bin_low"], np.arange(BIN_COUNT) / BIN_COUNT) or not np.allclose(bins["bin_high"], np.arange(1, BIN_COUNT + 1) / BIN_COUNT):
                raise ScoringContractError("Bines de calibracion alterados.")
            empty = bins["n_points"].eq(0)
            if not bins.loc[empty, "n_matches"].eq(0).all() or bins.loc[empty, ["mean_score", "observed_rate", "absolute_gap"]].notna().any().any():
                raise ScoringContractError("Bin vacio con metricas publicadas.")
            nonempty = bins.loc[~empty]
            if nonempty[["mean_score", "observed_rate", "absolute_gap"]].isna().any().any():
                raise ScoringContractError("Bin no vacio sin metricas.")
            if not nonempty["mean_score"].between(0, 1).all() or not nonempty["observed_rate"].between(0, 1).all() or not nonempty["absolute_gap"].ge(0).all():
                raise ScoringContractError("Metricas de calibracion fuera de dominio.")
            if not np.allclose(nonempty["absolute_gap"], (nonempty["mean_score"] - nonempty["observed_rate"]).abs(), rtol=1e-12, atol=1e-12):
                raise ScoringContractError("Gap de calibracion no reconcilia.")
            total = int(bins["n_points"].sum())
            if total != int(scorer_rows.loc[scorer_rows["fold_or_pooled"].astype(str).eq(label), "n_points"].sum()):
                raise ScoringContractError("Calibracion no reconcilia puntos publicados.")
            ece = float((nonempty["n_points"] * nonempty["absolute_gap"]).sum() / total)
            maximum = float(nonempty["absolute_gap"].max())
            if label != "pooled":
                fold_rows = by_fold.loc[(by_fold["scorer"].eq(scorer)) & (by_fold["fold"].eq(int(label)))]
                if not fold_rows["ece"].map(lambda value: math.isclose(float(value), ece, rel_tol=1e-12, abs_tol=1e-12)).all() or not fold_rows["max_calibration_error"].map(lambda value: math.isclose(float(value), maximum, rel_tol=1e-12, abs_tol=1e-12)).all():
                    raise ScoringContractError("ECE o maximo de calibracion no reconcilian.")

    pooled_metrics = summary["probability_metrics"]
    folds_not_worse = int(sum(
        np.average(
            by_direction.loc[(by_direction["scorer"].eq(MAIN_SCORER)) & (by_direction["fold_or_pooled"].astype(str).eq(str(fold))), "brier_score"],
            weights=by_direction.loc[(by_direction["scorer"].eq(MAIN_SCORER)) & (by_direction["fold_or_pooled"].astype(str).eq(str(fold))), "n_points"],
        ) <= np.average(
            by_direction.loc[(by_direction["scorer"].eq("population_only")) & (by_direction["fold_or_pooled"].astype(str).eq(str(fold))), "brier_score"],
            weights=by_direction.loc[(by_direction["scorer"].eq("population_only")) & (by_direction["fold_or_pooled"].astype(str).eq(str(fold))), "n_points"],
        )
        for fold in FOLDS
    ))
    checks = {
        "main_brier_not_worse_than_population": pooled_metrics[MAIN_SCORER]["brier_score"] <= pooled_metrics["population_only"]["brier_score"],
        "main_brier_not_worse_than_worst_individual": pooled_metrics[MAIN_SCORER]["brier_score"] <= max(pooled_metrics["server_only"]["brier_score"], pooled_metrics["opponent_only"]["brier_score"]),
        "folds_brier_not_worse_than_population": folds_not_worse,
        "main_log_loss_not_worse_than_population": pooled_metrics[MAIN_SCORER]["log_loss"] <= pooled_metrics["population_only"]["log_loss"],
    }
    checks["at_least_three_folds_not_worse"] = bool(checks["folds_brier_not_worse_than_population"] >= 3)
    criteria = summary.get("validation_criteria", {})
    for key, value in checks.items():
        if criteria.get(key) != value:
            raise ScoringContractError(f"Criterio de validacion alterado: {key}.")
    status = summary["baseline_status"]
    failed_predictive = sorted(key for key, value in checks.items() if value is False)
    if status == "validated_descriptive_baseline":
        if failed_predictive or summary.get("reason_codes"):
            raise ScoringContractError("Estado validated no reconcilia criterios predictivos.")
    elif status == "not_validated":
        if not failed_predictive or summary.get("reason_codes") != failed_predictive:
            raise ScoringContractError("Estado not_validated no reconcilia reason codes.")

    explainability = summary.get("explainability", {})
    orientations = explainability.get("orientations")
    if not isinstance(orientations, int) or orientations <= 0:
        raise ScoringContractError("Explicabilidad sin orientaciones validas.")
    if sum(explainability.get("dominant_component", {}).values()) != orientations or sum(explainability.get("selected_scope", {}).values()) != orientations:
        raise ScoringContractError("Explicabilidad no reconcilia sus totales.")
    for component in ("server", "opponent"):
        if explainability.get(f"{component}_component_positive", 0) + explainability.get(f"{component}_component_negative", 0) > orientations:
            raise ScoringContractError("Signos de explicabilidad fuera de dominio.")
    for control in ("population_only", "server_only", "opponent_only"):
        paired = summary.get("paired_match_comparison", {}).get(control, {})
        if paired.get("unit") != "target_match" or paired.get("pairs") != pooled_metrics[MAIN_SCORER].get("n_matches"):
            raise ScoringContractError("Comparacion paired no reconcilia poblacion.")
        for key in ("main_lower_brier_proportion",):
            if not isinstance(paired.get(key), (int, float)) or not 0 <= paired[key] <= 1:
                raise ScoringContractError("Comparacion paired fuera de dominio.")
    for row in ranking.itertuples(index=False):
        if row.comparable_matches > row.comparable_orientations or row.strict_winner_orientations > row.comparable_orientations:
            raise ScoringContractError("Poblacion de ranking incoherente.")
        if row.median_regret is not None and not pd.isna(row.median_regret) and row.p90_regret < row.median_regret:
            raise ScoringContractError("Cuantiles de ranking incoherentes.")
        for value in (row.mean_regret, row.median_regret, row.p90_regret):
            if value is not None and not pd.isna(value) and value < 0:
                raise ScoringContractError("Regret de ranking fuera de dominio.")
        for value in (row.top1_hit_inclusive, row.top1_hit_strict, row.mrr, row.regret_at_most_005, row.regret_at_most_010):
            if value is not None and not pd.isna(value) and not 0 <= value <= 1:
                raise ScoringContractError("Metrica de ranking fuera de dominio.")


def analyze_points(points: pd.DataFrame | None = None, recorder: PerformanceRecorder | None = None) -> ScoringResult:
    """Orquestador unico: conserva la frontera exacta si cualquier etapa falla."""

    recorder = recorder or PerformanceRecorder()
    completed: list[str] = []
    diagnostics: dict[str, Any] = {}
    stage = "load_upstream_contracts"
    try:
        started = time.perf_counter()
        selection, _, _ = load_and_validate_upstream_artifacts()
        recorder.record(stage, started, 3); completed.append(stage)
        stage = "read_source"
        if points is None:
            started = time.perf_counter()
            points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
            recorder.record(stage, started, len(points))
        completed.append(stage)
        stage = "prepare_history"
        started = time.perf_counter()
        frame, targets, eligible, source_audit = prepare_historical_population(points)
        diagnostics.update({"source_point_rows": int(source_audit["source_point_rows"]), "source_matches": int(source_audit["source_matches"]), "eligible_history_points": int(source_audit["eligible_points"])})
        recorder.record(stage, started, len(eligible)); completed.append(stage)
        if source_audit["source_point_rows"] != EXPECTED_SOURCE_ROWS or source_audit["source_matches"] != EXPECTED_MATCHES:
            raise ScoringContractError("La poblacion fuente no coincide con el contrato 1.280.408/7.524.")
        raw_validation = frame.loc[pd.to_datetime(frame["date"]).dt.normalize().between(VALIDATION_START, VALIDATION_END)]
        substantive_count = int(raw_validation["second_serve"].map(classify_presence).eq("substantive").sum())
        stage = "seal_targets"
        started = time.perf_counter()
        snapshots, features, sealed_audit, test_ids = construct_sealed_target_features(targets, eligible, recorder=recorder)
        diagnostics.update({"constructed_target_matches": int(sealed_audit["constructed_target_matches"]), "test_target_matches_excluded_before_construction": int(sealed_audit["test_target_matches_excluded_before_construction"]), "constructed_snapshot_rows": int(len(snapshots)), "constructed_feature_rows": int(len(features))})
        recorder.record(stage, started, len(features)); completed.extend(["seal_targets", "build_snapshots", "build_features"])
        if set(features["target_match_id"]) & test_ids or features["target_date"].max() > MAX_TARGET_DATE:
            raise ScoringContractError("El sellado fisico permitio contaminar features con test.")
        stage = "select_validation_features"
        started = time.perf_counter()
        validation_features = select_validation_features(features, test_ids)
        recorder.record(stage, started, len(validation_features)); completed.append(stage)
        stage = "prepare_scoring_population"
        started = time.perf_counter()
        scored_orientations = prepare_scoring_population(validation_features)
        recorder.record(stage, started, len(scored_orientations)); completed.append(stage)
        stage = "join_target_outcomes"
        started = time.perf_counter()
        all_points, scored_points = join_target_outcomes(scored_orientations, eligible)
        if substantive_count < len(all_points):
            raise ScoringContractError("Los puntos reconocidos exceden los segundos saques sustantivos target.")
        recorder.record(stage, started, len(scored_points)); completed.append(stage)
        stage = "compute_scores"
        started = time.perf_counter()
        long = _score_long(scored_points)
        recorder.record(stage, started, len(long)); completed.append(stage)
        stage = "probability_metrics"
        started = time.perf_counter()
        by_fold, by_direction, calibration = build_probability_tables(long, all_points)
        recorder.record(stage, started, len(by_fold) + len(by_direction)); completed.append(stage)
        stage = "calibration"
        completed.append(stage)
        stage = "ranking"
        started = time.perf_counter()
        ranking = build_ranking(long)
        recorder.record(stage, started, len(ranking)); completed.append(stage)
        status, reasons, validation_metrics = determine_baseline_status(long)
        summary = build_summary(selection=selection, source_audit=source_audit, sealed_audit=sealed_audit, all_points=all_points, scored_points=scored_points, long=long, by_fold=by_fold, calibration=calibration, ranking=ranking, status=status, reasons=reasons, validation_metrics=validation_metrics)
        frames = (by_fold, by_direction, calibration, ranking)
        summary, _, fingerprint = _finalize_summary(summary, frames)
        result = ScoringResult(summary, by_fold, by_direction, calibration, ranking, fingerprint, recorder.snapshot())
        stage = "validate_result"
        validate_result(result); completed.append(stage)
        return result
    except BaseException as exc:
        if isinstance(exc, ScoringExecutionError):
            raise
        raise ScoringExecutionError(stage, exc, completed, diagnostics) from exc


def not_available_result(failure: ScoringExecutionError | str) -> ScoringResult:
    frames = tuple(pd.DataFrame(columns=columns) for columns in (BY_FOLD_COLUMNS, BY_DIRECTION_COLUMNS, CALIBRATION_COLUMNS, RANKING_COLUMNS))
    if isinstance(failure, ScoringExecutionError):
        failure_stage = failure.stage
        exception_type = failure.exception_type
        exception_message = failure.exception_message
        reason_codes = [failure.reason_code]
        completed_stages = list(failure.completed_stages)
        diagnostics = failure.partial_population_diagnostics
    else:
        failure_stage = "load_upstream_contracts"
        exception_type = "ScoringContractError"
        exception_message = sanitize_exception_message(str(failure))
        reason_codes = ["upstream_contract_failure"]
        completed_stages, diagnostics = [], {}
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "baseline_status": "not_available", "reason_codes": reason_codes,
        "failure_stage": failure_stage, "exception_type": exception_type,
        "exception_message": exception_message, "completed_stages": completed_stages,
        "last_completed_stage": None if not completed_stages else completed_stages[-1],
        "partial_population_diagnostics": diagnostics, "inference_published": False,
        "sealed_test_contract": {"test_status": "sealed", "test_target_matches_constructed": 0, "test_feature_rows_constructed": 0, "test_points_scored": 0, "test_matches_scored": 0, "test_evaluation_runs": 0, "used_for_method_selection": False},
        "reconciliations": {"partial_evaluation_published": False},
    }
    summary, _, fingerprint = _finalize_summary(summary, frames)
    return ScoringResult(summary, *frames, fingerprint)


def validate_result(result: ScoringResult, *, artifact_payloads: tuple[bytes, ...] | None = None) -> None:
    frames = (result.by_fold, result.by_direction, result.calibration, result.ranking)
    expected_columns = (BY_FOLD_COLUMNS, BY_DIRECTION_COLUMNS, CALIBRATION_COLUMNS, RANKING_COLUMNS)
    if any(frame.columns.tolist() != columns for frame, columns in zip(frames, expected_columns)):
        raise ScoringContractError("Esquema de artefacto no contractual.")
    status = result.summary.get("baseline_status")
    if status not in {"validated_descriptive_baseline", "not_validated", "not_available"}:
        raise ScoringContractError("Estado de baseline inesperado.")
    sealed = result.summary.get("sealed_test_contract", {})
    required_seal = {"test_status": "sealed", "test_target_matches_constructed": 0, "test_feature_rows_constructed": 0, "test_points_scored": 0, "test_matches_scored": 0, "test_evaluation_runs": 0, "used_for_method_selection": False}
    if any(sealed.get(key) != value for key, value in required_seal.items()):
        raise ScoringContractError("Contrato de test sellado alterado.")
    if status == "not_available":
        if any(not frame.empty for frame in frames) or not result.summary.get("reason_codes"):
            raise ScoringContractError("not_available no admite artefactos parciales.")
        if result.summary.get("failure_stage") not in STAGES:
            raise ScoringContractError("not_available requiere una etapa cerrada.")
        if not isinstance(result.summary.get("exception_type"), str) or not isinstance(result.summary.get("exception_message"), str):
            raise ScoringContractError("not_available requiere tipo y mensaje de excepcion.")
        if len(result.summary["exception_message"]) > MAX_EXCEPTION_MESSAGE_LENGTH or "\n" in result.summary["exception_message"]:
            raise ScoringContractError("El mensaje de excepcion no esta sanitizado.")
        if result.summary.get("inference_published") is not False:
            raise ScoringContractError("not_available no puede publicar inferencia.")
        completed = result.summary.get("completed_stages")
        if not isinstance(completed, list) or any(stage not in STAGES for stage in completed):
            raise ScoringContractError("Las etapas completadas no son contractuales.")
        if result.summary.get("last_completed_stage") != (None if not completed else completed[-1]):
            raise ScoringContractError("last_completed_stage no reconcilia.")
    else:
        if len(result.calibration) != len(SCORERS) * (len(FOLDS) + 1) * BIN_COUNT:
            raise ScoringContractError("Los bins de calibracion no son exhaustivos.")
        if len(result.ranking) != len(SCORERS) * (len(FOLDS) + 1):
            raise ScoringContractError("El ranking no contiene cuatro scorers y cinco niveles.")
        if result.by_fold.empty or result.by_direction.empty:
            raise ScoringContractError("No se permite una evaluacion parcial.")
        numeric = pd.concat([frame.select_dtypes(include=[np.number]) for frame in frames], axis=1)
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ScoringContractError("Los artefactos contienen infinito.")
        if not result.summary.get("reconciliations", {}).get("no_test_contamination"):
            raise ScoringContractError("La reconciliacion de test no pasa.")
        reconciliations = result.summary.get("reconciliations")
        if not isinstance(reconciliations, dict) or set(reconciliations) != RECONCILIATION_KEYS:
            raise ScoringContractError("Dominio de reconciliaciones no contractual.")
        for key, value in reconciliations.items():
            _require_exact_bool(value, f"reconciliations.{key}")
        if status == "validated_descriptive_baseline" and not all(reconciliations.values()):
            raise ScoringContractError("Un baseline validated requiere reconciliaciones verdaderas.")
        _validate_aggregate_semantics(result)
    _assert_finite_json(result.summary)
    payloads = artifact_payloads or _csv_payloads(frames)
    expected = _fingerprint(result.summary, payloads)
    if result.publication_fingerprint != expected or result.summary.get("publication_fingerprint") != expected:
        raise ScoringContractError("Fingerprint de publicacion alterado.")
    if "C:\\" in json.dumps(result.summary, ensure_ascii=False) or "C:/" in json.dumps(result.summary, ensure_ascii=False):
        raise ScoringContractError("El resumen contiene rutas absolutas.")


def serialize_artifacts(result: ScoringResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    validate_result(result)
    summary = (json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    payloads = (summary, *(_frame_bytes(frame) for frame in (result.by_fold, result.by_direction, result.calibration, result.ranking)))
    for payload, columns in zip(payloads[1:], (BY_FOLD_COLUMNS, BY_DIRECTION_COLUMNS, CALIBRATION_COLUMNS, RANKING_COLUMNS)):
        loaded = pd.read_csv(io.BytesIO(payload))
        if loaded.columns.tolist() != columns or any(str(column).startswith("Unnamed") for column in loaded.columns):
            raise ScoringContractError("CSV serializado con indice o esquema accidental.")
    return payloads


def verify_persisted_artifacts(
    summary_path: Path = SUMMARY_PATH,
    by_fold_path: Path = BY_FOLD_PATH,
    by_direction_path: Path = BY_DIRECTION_PATH,
    calibration_path: Path = CALIBRATION_PATH,
    ranking_path: Path = RANKING_PATH,
) -> ScoringResult:
    """Cierra y reabre los artefactos validando hashes sobre bytes exactos."""

    paths = (summary_path, by_fold_path, by_direction_path, calibration_path, ranking_path)
    if not all(path.exists() for path in paths):
        raise ScoringContractError("Falta un artefacto permanente para verificar publicacion.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payloads = tuple(path.read_bytes() for path in paths[1:])
    hashes, sizes = _payload_metadata(payloads)
    if summary.get("fingerprint_contract_version") != FINGERPRINT_CONTRACT_VERSION:
        raise ScoringContractError("Version de fingerprint persistido inesperada.")
    if summary.get("artifact_payload_sha256") != hashes or summary.get("artifact_payload_bytes") != sizes:
        raise ScoringContractError("Hashes o tamanos CSV persistidos no reconcilian.")
    frames = tuple(pd.read_csv(io.BytesIO(payload)) for payload in payloads)
    result = ScoringResult(summary, *frames, str(summary.get("publication_fingerprint", "")))
    validate_result(result, artifact_payloads=payloads)
    return result


def _stage_payload(path: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def write_artifacts(
    result: ScoringResult,
    summary_path: Path = SUMMARY_PATH,
    by_fold_path: Path = BY_FOLD_PATH,
    by_direction_path: Path = BY_DIRECTION_PATH,
    calibration_path: Path = CALIBRATION_PATH,
    ranking_path: Path = RANKING_PATH,
) -> None:
    paths = (summary_path, by_fold_path, by_direction_path, calibration_path, ranking_path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Los cinco artefactos requieren rutas distintas.")
    payloads = serialize_artifacts(result)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    originals = {path: path.read_bytes() if path.exists() else None for path in paths}
    staged: list[Path] = []
    replaced: list[Path] = []
    try:
        staged = [_stage_payload(path, payload) for path, payload in zip(paths, payloads)]
        for temporary, path in zip(staged, paths):
            os.replace(temporary, path); replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            if originals[path] is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(_stage_payload(path, originals[path]), path)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def remove_reproduction_directory(path: Path = REPRO_DIR) -> None:
    if path.resolve() != REPRO_DIR.resolve():
        raise ValueError("Solo puede eliminarse el temporal contractual.")
    if path.exists():
        shutil.rmtree(path)


def main() -> None:
    if REPRO_DIR.exists():
        raise FileExistsError(f"El temporal ya existe: {REPRO_DIR.name}")
    recorder = PerformanceRecorder()
    try:
        result = analyze_points(recorder=recorder)
    except ScoringExecutionError as exc:
        result = not_available_result(exc)
    except (ScoringContractError, ValueError, KeyError) as exc:
        result = not_available_result(str(exc))
    write_artifacts(result)
    permanent_paths = (SUMMARY_PATH, BY_FOLD_PATH, BY_DIRECTION_PATH, CALIBRATION_PATH, RANKING_PATH)
    permanent = tuple(path.read_bytes() for path in permanent_paths)
    if permanent != serialize_artifacts(result):
        raise ScoringContractError("Los artefactos permanentes no coinciden con el resultado en memoria.")
    persisted = verify_persisted_artifacts()
    if persisted.publication_fingerprint != result.publication_fingerprint:
        raise ScoringContractError("La reapertura persistida no conserva el fingerprint v2.")
    REPRO_DIR.mkdir(parents=False)
    repro_paths = tuple(REPRO_DIR / path.name for path in permanent_paths)
    try:
        write_artifacts(result, *repro_paths)
        if tuple(path.read_bytes() for path in repro_paths) != permanent:
            raise ScoringContractError("La reserializacion desde memoria no es byte-identica.")
    finally:
        remove_reproduction_directory(REPRO_DIR)
    if REPRO_DIR.exists():
        raise ScoringContractError("El temporal de reproducibilidad no fue eliminado.")
    print(json.dumps({"baseline_status": result.summary["baseline_status"], "reason_codes": result.summary["reason_codes"], "artifacts": [{"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path.read_bytes())} for path in permanent_paths], "reproduction_directory_exists": False, "performance_metrics": recorder.snapshot()}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
