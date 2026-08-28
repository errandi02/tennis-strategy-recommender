"""Variables descriptivas jugador-rival por direccion del segundo saque.

La tabla completa se conserva en memoria. Los artefactos permanentes contienen
solo cobertura, estabilidad y un resumen metodologico agregado.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.historical_profiles import (
    POINTS_FILE,
    SOURCE_COLUMNS,
    build_historical_snapshots,
    prepare_historical_population,
    validate_snapshots,
)
from src.analysis.second_serve_direction_analysis import PERIOD_ORDER, SURFACE_ORDER, derive_period
from src.parsing.serve_sequence import parse_sequence


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "player_opponent_direction_features_summary.json"
COVERAGE_PATH = TABLES_DIR / "player_opponent_direction_features_coverage.csv"
STABILITY_PATH = TABLES_DIR / "player_opponent_direction_features_stability.csv"
REPRO_DIR = REPORTS_DIR / "repro_player_opponent_direction_features"

ANALYSIS_NAME = "player_opponent_second_serve_direction_features"
ANALYSIS_VERSION = "1.0.0"
BASE_COMMIT = "68a30c16ec0c165febeb3d34d483acbb5b18703a"
DIRECTIONS = ("wide", "body", "T")
INTERNAL_DIRECTION = {"wide": "wide", "body": "body", "t": "T"}
SCOPES = ("global", "surface")
PAIR_STATES = ("neither_observed", "server_only", "opponent_only", "both_observed")
POINT_THRESHOLDS = (0, 10, 25, 50, 100, 250)
MATCH_THRESHOLDS = (0, 1, 3, 5, 10, 20)
Z_975 = 1.959963984540054
QUANTILE_METHOD = "linear"

ID_COLUMNS = [
    "target_match_id", "target_date", "target_player", "opponent",
    "target_surface", "derived_period", "direction", "scope",
]
SERVER_COLUMNS = [
    "server_direction_matches", "server_direction_points", "server_direction_wins",
    "server_direction_raw_rate", "server_direction_wilson_low",
    "server_direction_wilson_high", "server_direction_share", "server_scope_points",
    "server_scope_wins", "server_scope_raw_rate",
    "server_direction_first_history_date", "server_direction_last_history_date",
]
OPPONENT_COLUMNS = [
    "opponent_direction_matches", "opponent_direction_points",
    "opponent_allowed_server_wins", "opponent_return_wins",
    "opponent_allowed_server_raw_rate", "opponent_allowed_server_wilson_low",
    "opponent_allowed_server_wilson_high", "opponent_direction_share",
    "opponent_scope_points", "opponent_scope_allowed_server_wins",
    "opponent_scope_allowed_server_raw_rate",
    "opponent_direction_first_history_date", "opponent_direction_last_history_date",
]
POPULATION_COLUMNS = [
    "population_direction_points", "population_direction_server_wins",
    "population_direction_raw_rate", "population_direction_wilson_low",
    "population_direction_wilson_high", "population_direction_first_history_date",
    "population_direction_last_history_date",
]
COMPARISON_COLUMNS = [
    "server_excess_vs_population", "opponent_allowed_excess_vs_population",
    "server_opponent_gap",
]
EVIDENCE_COLUMNS = ["server_evidence_state", "opponent_evidence_state", "pair_evidence_state"]
FEATURE_COLUMNS = ID_COLUMNS + SERVER_COLUMNS + OPPONENT_COLUMNS + POPULATION_COLUMNS + COMPARISON_COLUMNS + EVIDENCE_COLUMNS

COVERAGE_COLUMNS = [
    "scope", "direction", "metric", "threshold", "stratum_type", "stratum",
    "eligible_rows", "total_rows", "coverage_rate", "complete_matches",
    "total_matches", "complete_match_rate", "covered_players",
]
STABILITY_COLUMNS = [
    "role", "scope", "direction", "threshold_points", "stratum_type", "stratum",
    "observed_rows", "minimum_points", "median_points", "points_q1", "points_q3",
    "median_raw_rate", "raw_rate_q1", "raw_rate_q3", "zero_rate_rows",
    "one_rate_rows", "zero_rate_share", "one_rate_share", "median_wilson_width",
    "wilson_width_p90",
]


@dataclass(frozen=True)
class DirectionFeatureResult:
    features: pd.DataFrame
    coverage: pd.DataFrame
    stability: pd.DataFrame
    summary: dict[str, Any]
    source_contract: dict[str, Any]
    population_audit: dict[str, Any]
    publication_fingerprint: str = ""
    performance_metrics: tuple[dict[str, Any], ...] = ()


class PerformanceRecorder:
    """Cronometro opcional; sus metricas nunca forman parte de artefactos."""

    def __init__(
        self,
        enabled: bool = False,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.callback = callback
        self.log_path = log_path
        self.records: list[dict[str, Any]] = []

    def record(self, phase: str, started: float, rows: int | None = None) -> None:
        if not self.enabled:
            return
        record = {
            "phase": phase,
            "duration_seconds": time.perf_counter() - started,
            "rows": None if rows is None else int(rows),
        }
        self.records.append(record)
        if self.callback is not None:
            self.callback(record.copy())
        if self.log_path is not None:
            payload = {
                "completed_phases": self.records,
                "last_completed_phase": phase,
            }
            self.log_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.copy() for record in self.records)


class FeatureValidationError(ValueError):
    """Fallo contractual que conserva diagnosticos y ejemplos auditables."""

    def __init__(self, message: str, diagnostics: list[dict[str, Any]], examples: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.examples = examples


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
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


def _normalized_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return parsed.dt.normalize()


def wilson_interval(wins: int | pd.Series, points: int | pd.Series):
    """IC95% Wilson bilateral; devuelve nulos cuando el denominador es cero."""

    scalar = np.isscalar(wins) and np.isscalar(points)
    wins_series = pd.Series([wins]) if scalar else pd.Series(wins, copy=False)
    points_series = pd.Series([points]) if scalar else pd.Series(points, copy=False)
    valid = points_series.gt(0)
    low = pd.Series(np.nan, index=points_series.index, dtype="float64")
    high = pd.Series(np.nan, index=points_series.index, dtype="float64")
    if valid.any():
        n = points_series[valid].astype(float)
        p = wins_series[valid].astype(float) / n
        denominator = 1.0 + Z_975**2 / n
        centre = (p + Z_975**2 / (2.0 * n)) / denominator
        half = Z_975 * np.sqrt(p * (1.0 - p) / n + Z_975**2 / (4.0 * n**2)) / denominator
        low.loc[valid] = np.maximum(0.0, centre - half)
        high.loc[valid] = np.minimum(1.0, centre + half)
    if scalar:
        return (None, None) if not valid.iloc[0] else (float(low.iloc[0]), float(high.iloc[0]))
    return low, high


def _empty_state() -> dict[str, Any]:
    return {"matches": 0, "points": 0, "wins": 0, "first": None, "last": None}


def _read_state(store: dict[Any, dict[str, Any]], key: Any) -> dict[str, Any]:
    state = store.get(key)
    return _empty_state() if state is None else state.copy()


def _update_state(store: dict[Any, dict[str, Any]], key: Any, matches: int, points: int, wins: int, date: pd.Timestamp) -> None:
    state = store.setdefault(key, _empty_state())
    state["matches"] += int(matches)
    state["points"] += int(points)
    state["wins"] += int(wins)
    day = pd.Timestamp(date).normalize()
    if state["first"] is None:
        state["first"] = day
    state["last"] = day


def _direction_events(eligible: pd.DataFrame, role_column: str) -> pd.DataFrame:
    return (
        eligible.groupby(["date", "match_id", "surface", role_column, "direction"], sort=False, observed=True)
        .agg(points=("point_number", "size"), wins=("server_won_point", "sum"))
        .reset_index()
        .rename(columns={role_column: "player"})
    )


def _population_events(eligible: pd.DataFrame) -> pd.DataFrame:
    return (
        eligible.groupby(["date", "surface", "direction"], sort=False, observed=True)
        .agg(points=("point_number", "size"), wins=("server_won_point", "sum"))
        .reset_index()
    )


def _rate(wins: int, points: int) -> float | None:
    return None if points == 0 else wins / points


def _state_scope_totals(store: dict[Any, dict[str, Any]], key_builder: Callable[[str], Any]) -> tuple[int, int]:
    states = [_read_state(store, key_builder(direction)) for direction in DIRECTIONS]
    return sum(state["points"] for state in states), sum(state["wins"] for state in states)


def _calculate_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """Calcula tasas, Wilson, shares, gaps y estados en bloques vectorizados."""

    result = features.copy(deep=False)
    specifications = (
        ("server_direction_wins", "server_direction_points", "server_direction_raw_rate", "server_direction_wilson_low", "server_direction_wilson_high"),
        ("opponent_allowed_server_wins", "opponent_direction_points", "opponent_allowed_server_raw_rate", "opponent_allowed_server_wilson_low", "opponent_allowed_server_wilson_high"),
        ("population_direction_server_wins", "population_direction_points", "population_direction_raw_rate", "population_direction_wilson_low", "population_direction_wilson_high"),
    )
    for wins, points, rate, low, high in specifications:
        result[rate] = result[wins].div(result[points].where(result[points].gt(0)))
        result[low], result[high] = wilson_interval(result[wins], result[points])
    result["server_direction_share"] = result["server_direction_points"].div(
        result["server_scope_points"].where(result["server_scope_points"].gt(0))
    )
    result["server_scope_raw_rate"] = result["server_scope_wins"].div(
        result["server_scope_points"].where(result["server_scope_points"].gt(0))
    )
    result["opponent_direction_share"] = result["opponent_direction_points"].div(
        result["opponent_scope_points"].where(result["opponent_scope_points"].gt(0))
    )
    result["opponent_scope_allowed_server_raw_rate"] = result[
        "opponent_scope_allowed_server_wins"
    ].div(result["opponent_scope_points"].where(result["opponent_scope_points"].gt(0)))
    result["server_excess_vs_population"] = (
        result["server_direction_raw_rate"] - result["population_direction_raw_rate"]
    )
    result["opponent_allowed_excess_vs_population"] = (
        result["opponent_allowed_server_raw_rate"] - result["population_direction_raw_rate"]
    )
    result["server_opponent_gap"] = (
        result["server_direction_raw_rate"] - result["opponent_allowed_server_raw_rate"]
    )
    server_observed = result["server_direction_points"].gt(0)
    opponent_observed = result["opponent_direction_points"].gt(0)
    result["server_evidence_state"] = np.where(server_observed, "observed", "no_history")
    result["opponent_evidence_state"] = np.where(opponent_observed, "observed", "no_history")
    result["pair_evidence_state"] = np.select(
        [server_observed & opponent_observed, server_observed, opponent_observed],
        ["both_observed", "server_only", "opponent_only"],
        default="neither_observed",
    )
    return result


def build_direction_features(
    snapshots: pd.DataFrame,
    eligible: pd.DataFrame,
    recorder: PerformanceRecorder | None = None,
) -> pd.DataFrame:
    """Expande snapshots a direccion y scope usando solo dias anteriores."""

    work = eligible.copy(deep=True)
    work["date"] = _normalized_dates(work["date"])
    if work["date"].isna().any():
        raise ValueError("eligible contiene fechas no validas.")
    work["direction"] = work["direction"].map(INTERNAL_DIRECTION)
    if work["direction"].isna().any():
        raise ValueError("eligible contiene direcciones fuera del contrato.")
    recorder = recorder or PerformanceRecorder()
    phase_started = time.perf_counter()
    server_events = _direction_events(work, "server_player")
    opponent_events = _direction_events(work, "returner_player")
    recorder.record(
        "build_directional_daily_aggregates",
        phase_started,
        len(server_events) + len(opponent_events),
    )
    phase_started = time.perf_counter()
    population_events = _population_events(work)
    recorder.record("build_population_baseline", phase_started, len(population_events))
    server_by_date = {day: group for day, group in server_events.groupby("date", sort=False)}
    opponent_by_date = {day: group for day, group in opponent_events.groupby("date", sort=False)}
    population_by_date = {day: group for day, group in population_events.groupby("date", sort=False)}

    base = snapshots.copy(deep=True)
    base["target_date"] = _normalized_dates(base["target_date"])
    base["derived_period"] = base["target_date"].dt.year.map(derive_period)
    server_global: dict[Any, dict[str, Any]] = {}
    server_surface: dict[Any, dict[str, Any]] = {}
    opponent_global: dict[Any, dict[str, Any]] = {}
    opponent_surface: dict[Any, dict[str, Any]] = {}
    population_global: dict[Any, dict[str, Any]] = {}
    population_surface: dict[Any, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    phase_started = time.perf_counter()
    for day, daily_targets in base.groupby("target_date", sort=True):
        daily_targets = daily_targets.sort_values(["target_match_id", "target_player"], kind="stable")
        for target in daily_targets.itertuples(index=False):
            for direction in DIRECTIONS:
                for scope in SCOPES:
                    if scope == "global":
                        server_store, opponent_store, population_store = server_global, opponent_global, population_global
                        server_key = (target.target_player, direction)
                        opponent_key = (target.opponent, direction)
                        population_key = direction
                        server_total_key = lambda d, player=target.target_player: (player, d)
                        opponent_total_key = lambda d, player=target.opponent: (player, d)
                    else:
                        server_store, opponent_store, population_store = server_surface, opponent_surface, population_surface
                        server_key = (target.target_player, target.target_surface, direction)
                        opponent_key = (target.opponent, target.target_surface, direction)
                        population_key = (target.target_surface, direction)
                        server_total_key = lambda d, player=target.target_player, surface=target.target_surface: (player, surface, d)
                        opponent_total_key = lambda d, player=target.opponent, surface=target.target_surface: (player, surface, d)
                    server = _read_state(server_store, server_key)
                    opponent = _read_state(opponent_store, opponent_key)
                    population = _read_state(population_store, population_key)
                    server_scope_points, server_scope_wins = _state_scope_totals(server_store, server_total_key)
                    opponent_scope_points, opponent_scope_wins = _state_scope_totals(opponent_store, opponent_total_key)
                    rows.append({
                        "target_match_id": target.target_match_id,
                        "target_date": pd.Timestamp(target.target_date),
                        "target_player": target.target_player,
                        "opponent": target.opponent,
                        "target_surface": target.target_surface,
                        "derived_period": target.derived_period,
                        "direction": direction,
                        "scope": scope,
                        "server_direction_matches": server["matches"],
                        "server_direction_points": server["points"],
                        "server_direction_wins": server["wins"],
                        "server_scope_points": server_scope_points,
                        "server_scope_wins": server_scope_wins,
                        "server_direction_first_history_date": server["first"],
                        "server_direction_last_history_date": server["last"],
                        "opponent_direction_matches": opponent["matches"],
                        "opponent_direction_points": opponent["points"],
                        "opponent_allowed_server_wins": opponent["wins"],
                        "opponent_return_wins": opponent["points"] - opponent["wins"],
                        "opponent_scope_points": opponent_scope_points,
                        "opponent_scope_allowed_server_wins": opponent_scope_wins,
                        "opponent_direction_first_history_date": opponent["first"],
                        "opponent_direction_last_history_date": opponent["last"],
                        "population_direction_points": population["points"],
                        "population_direction_server_wins": population["wins"],
                        "population_direction_first_history_date": population["first"],
                        "population_direction_last_history_date": population["last"],
                    })
        for event in server_by_date.get(day, pd.DataFrame()).itertuples(index=False):
            _update_state(server_global, (event.player, event.direction), 1, event.points, event.wins, day)
            _update_state(server_surface, (event.player, event.surface, event.direction), 1, event.points, event.wins, day)
        for event in opponent_by_date.get(day, pd.DataFrame()).itertuples(index=False):
            _update_state(opponent_global, (event.player, event.direction), 1, event.points, event.wins, day)
            _update_state(opponent_surface, (event.player, event.surface, event.direction), 1, event.points, event.wins, day)
        for event in population_by_date.get(day, pd.DataFrame()).itertuples(index=False):
            _update_state(population_global, event.direction, 0, event.points, event.wins, day)
            _update_state(population_surface, (event.surface, event.direction), 0, event.points, event.wins, day)

    features = pd.DataFrame(rows)
    direction_order = {value: index for index, value in enumerate(DIRECTIONS)}
    scope_order = {value: index for index, value in enumerate(SCOPES)}
    features["_direction_order"] = features["direction"].map(direction_order)
    features["_scope_order"] = features["scope"].map(scope_order)
    features = features.sort_values(
        ["target_date", "target_match_id", "target_player", "_direction_order", "_scope_order"],
        kind="stable",
    ).drop(columns=["_direction_order", "_scope_order"]).reset_index(drop=True)
    recorder.record("expand_directions_scopes", phase_started, len(features))
    phase_started = time.perf_counter()
    features = _calculate_derived_features(features)
    features = features[FEATURE_COLUMNS]
    recorder.record("calculate_rates_wilson_gaps", phase_started, len(features))
    return features


def _temporal_diagnostics(features: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = {
        "server": ("server_direction_points", "server_direction_first_history_date", "server_direction_last_history_date"),
        "opponent": ("opponent_direction_points", "opponent_direction_first_history_date", "opponent_direction_last_history_date"),
        "population": ("population_direction_points", "population_direction_first_history_date", "population_direction_last_history_date"),
    }
    diagnostics: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    target = _normalized_dates(features["target_date"])
    for prefix, (points_column, first_column, last_column) in definitions.items():
        points = features[points_column]
        first = _normalized_dates(features[first_column])
        last = _normalized_dates(features[last_column])
        positive, zero = points.gt(0), points.eq(0)
        masks = {
            "zero_first_present": zero & first.notna(),
            "zero_last_present": zero & last.notna(),
            "positive_first_missing": positive & first.isna(),
            "positive_last_missing": positive & last.isna(),
            "first_after_last": positive & first.notna() & last.notna() & first.gt(last),
            "last_on_or_after_target": positive & last.notna() & last.ge(target),
            "first_on_or_after_target": positive & first.notna() & first.ge(target),
        }
        for reason_code, mask in masks.items():
            affected = features.loc[mask].sort_values(
                ["target_date", "target_match_id", "target_player", "direction", "scope"], kind="stable"
            )
            diagnostics.append({"prefix": prefix, "reason_code": reason_code, "evaluated_rows": int(len(features)), "violations": int(len(affected))})
            for row in affected.head(10).itertuples(index=False):
                examples.append({
                    "prefix": prefix, "reason_code": reason_code,
                    "target_match_id": row.target_match_id, "target_player": row.target_player,
                    "opponent": row.opponent, "target_date": row.target_date.date().isoformat(),
                    "direction": row.direction, "scope": row.scope,
                    "points": int(getattr(row, points_column)),
                    "first_history_date": _json_value(getattr(row, first_column)),
                    "last_history_date": _json_value(getattr(row, last_column)),
                })
    return diagnostics, examples


def validate_feature_table(features: pd.DataFrame, snapshots: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("El esquema de features no coincide con el contrato.")
    expected_rows = len(snapshots) * len(DIRECTIONS) * len(SCOPES)
    if len(features) != expected_rows:
        raise ValueError(f"Cardinalidad incorrecta: {len(features)} != {expected_rows}.")
    key = ["target_match_id", "target_player", "direction", "scope"]
    if features.duplicated(key).any():
        raise ValueError("La clave de features no es unica.")
    if set(features["direction"]) != set(DIRECTIONS) or set(features["scope"]) != set(SCOPES):
        raise ValueError("Dominios de direction o scope invalidos.")
    count_columns = [column for column in FEATURE_COLUMNS if column.endswith(("_matches", "_points", "_wins"))]
    if features[count_columns].isna().any().any() or not all(pd.api.types.is_integer_dtype(features[c]) for c in count_columns):
        raise ValueError("Los conteos deben ser enteros no nulos.")
    if features[count_columns].lt(0).any().any():
        raise ValueError("Los conteos no pueden ser negativos.")
    if features["server_direction_wins"].gt(features["server_direction_points"]).any():
        raise ValueError("server wins supera points.")
    if features["opponent_allowed_server_wins"].gt(features["opponent_direction_points"]).any():
        raise ValueError("opponent allowed wins supera points.")
    if features["population_direction_server_wins"].gt(features["population_direction_points"]).any():
        raise ValueError("population wins supera points.")
    if (features["opponent_return_wins"] + features["opponent_allowed_server_wins"]).ne(features["opponent_direction_points"]).any():
        raise ValueError("El complemento servidor-restador no reconcilia.")

    group_key = ["target_match_id", "target_player", "scope"]
    for prefix, points, wins in (
        ("server", "server_direction_points", "server_direction_wins"),
        ("opponent", "opponent_direction_points", "opponent_allowed_server_wins"),
    ):
        grouped = features.groupby(group_key, sort=False)
        if grouped[points].sum().ne(grouped[f"{prefix}_scope_points"].first()).any():
            raise ValueError(f"Los puntos por direccion no reconcilian para {prefix}.")
        scope_wins = "server_scope_wins" if prefix == "server" else "opponent_scope_allowed_server_wins"
        if grouped[wins].sum().ne(grouped[scope_wins].first()).any():
            raise ValueError(f"Las victorias por direccion no reconcilian para {prefix}.")
        shares = features[f"{prefix}_direction_share"]
        totals = features[f"{prefix}_scope_points"]
        sums = shares.fillna(0).groupby([features[c] for c in group_key]).sum()
        positive = grouped[f"{prefix}_scope_points"].first().gt(0)
        if not np.allclose(sums[positive], 1.0, atol=1e-12, rtol=0):
            raise ValueError(f"Los shares no suman uno para {prefix}.")
        if shares[totals.eq(0)].notna().any():
            raise ValueError(f"Share no nulo con scope cero para {prefix}.")

    for prefix, wins_column, points_column, rate_column, low_column, high_column in (
        ("server", "server_direction_wins", "server_direction_points", "server_direction_raw_rate", "server_direction_wilson_low", "server_direction_wilson_high"),
        ("opponent", "opponent_allowed_server_wins", "opponent_direction_points", "opponent_allowed_server_raw_rate", "opponent_allowed_server_wilson_low", "opponent_allowed_server_wilson_high"),
        ("population", "population_direction_server_wins", "population_direction_points", "population_direction_raw_rate", "population_direction_wilson_low", "population_direction_wilson_high"),
    ):
        points = features[points_column]
        expected_rate = features[wins_column].div(points.where(points.gt(0)))
        if not np.allclose(features[rate_column], expected_rate, equal_nan=True, atol=0, rtol=1e-15):
            raise ValueError(f"Tasa incorrecta para {prefix}.")
        present = points.gt(0)
        if features.loc[~present, [rate_column, low_column, high_column]].notna().any().any():
            raise ValueError(f"Tasa o Wilson no nulo sin historia para {prefix}.")
        if (
            (features.loc[present, low_column] > features.loc[present, rate_column] + 1e-15)
            | (features.loc[present, rate_column] > features.loc[present, high_column] + 1e-15)
        ).any():
            raise ValueError(f"Wilson no contiene la tasa para {prefix}.")
    finite_columns = [column for column in FEATURE_COLUMNS if column not in ID_COLUMNS + EVIDENCE_COLUMNS and not column.endswith("history_date")]
    numeric = features[finite_columns].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Las features contienen infinito.")
    for column in COMPARISON_COLUMNS:
        if features[column].dropna().abs().gt(1.0 + 1e-12).any():
            raise ValueError(f"Diferencia fuera de [-1,1]: {column}.")
    comparable = features[COMPARISON_COLUMNS].notna().all(axis=1)
    identity_error = (
        features.loc[comparable, "server_opponent_gap"]
        - features.loc[comparable, "server_excess_vs_population"]
        + features.loc[comparable, "opponent_allowed_excess_vs_population"]
    ).abs()
    if identity_error.gt(1e-12).any():
        raise ValueError("La identidad algebraica de gaps no reconcilia.")
    diagnostics, examples = _temporal_diagnostics(features)
    if any(row["violations"] for row in diagnostics):
        codes = sorted({row["reason_code"] for row in diagnostics if row["violations"]})
        raise FeatureValidationError(f"Invariantes temporales incumplidos: {codes}", diagnostics, examples)

    global_rows = features[features["scope"].eq("global")]
    surface_rows = features[features["scope"].eq("surface")]
    joined = surface_rows.merge(global_rows, on=["target_match_id", "target_player", "direction"], suffixes=("_surface", "_global"), validate="one_to_one")
    for column in ("server_direction_matches", "server_direction_points", "server_direction_wins", "opponent_direction_matches", "opponent_direction_points", "opponent_allowed_server_wins", "population_direction_points", "population_direction_server_wins"):
        if joined[f"{column}_surface"].gt(joined[f"{column}_global"]).any():
            raise ValueError(f"El conteo surface supera global: {column}.")
    return diagnostics, examples


def _strata(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    strata = [("overall", "Overall", pd.Series(True, index=frame.index))]
    strata.extend(("surface", surface, frame["target_surface"].eq(surface)) for surface in SURFACE_ORDER)
    strata.extend(("derived_period", period, frame["derived_period"].eq(period)) for period in PERIOD_ORDER)
    return strata


def build_coverage(features: pd.DataFrame) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    role_scopes = (
        ("server_global", "server", "global"), ("server_surface", "server", "surface"),
        ("opponent_global", "opponent", "global"), ("opponent_surface", "opponent", "surface"),
        ("both_global", "both", "global"), ("both_surface", "both", "surface"),
    )
    for output_scope, role, scope in role_scopes:
        base = features.loc[
            features["scope"].eq(scope),
            [
                "target_match_id", "target_player", "target_surface",
                "derived_period", "direction", "server_direction_points",
                "opponent_direction_points", "server_direction_matches",
                "opponent_direction_matches",
            ],
        ]
        metric_tables: list[pd.DataFrame] = []
        for metric, thresholds in (
            ("direction_points", POINT_THRESHOLDS),
            ("direction_matches", MATCH_THRESHOLDS),
        ):
            server_values = base[f"server_{metric}"].to_numpy()
            opponent_values = base[f"opponent_{metric}"].to_numpy()
            threshold_frames = []
            for threshold in thresholds:
                passed = (
                    server_values >= threshold if role == "server" else
                    opponent_values >= threshold if role == "opponent" else
                    (server_values >= threshold) & (opponent_values >= threshold)
                )
                threshold_frames.append(
                    base[["target_match_id", "target_player", "target_surface", "derived_period", "direction"]]
                    .assign(threshold=int(threshold), passed=passed)
                )
            threshold_long = pd.concat(threshold_frames, ignore_index=True)
            stratified = pd.concat(
                [
                    threshold_long.assign(stratum_type="overall", stratum="Overall"),
                    threshold_long.assign(stratum_type="surface", stratum=threshold_long["target_surface"]),
                    threshold_long.assign(stratum_type="derived_period", stratum=threshold_long["derived_period"]),
                ],
                ignore_index=True,
            )
            keys = ["direction", "threshold", "stratum_type", "stratum"]
            totals = stratified.groupby(keys, sort=False, observed=True).agg(
                total_rows=("passed", "size"), eligible_rows=("passed", "sum")
            )
            covered = (
                stratified.loc[stratified["passed"]]
                .groupby(keys, sort=False, observed=True)["target_player"]
                .nunique()
                .rename("covered_players")
            )
            orientations = stratified.groupby(
                keys + ["target_match_id"], sort=False, observed=True
            )["passed"].agg(["size", "sum"])
            if not orientations.empty and not orientations["size"].eq(2).all():
                raise ValueError("Cobertura sin dos orientaciones por partido.")
            matches = orientations.groupby(level=keys, sort=False, observed=True).agg(
                total_matches=("size", "size"),
                complete_matches=("sum", lambda values: int(values.eq(2).sum())),
            )
            stratum_pairs = [("overall", "Overall")]
            stratum_pairs.extend(("surface", value) for value in SURFACE_ORDER)
            stratum_pairs.extend(("derived_period", value) for value in PERIOD_ORDER)
            expected = pd.MultiIndex.from_tuples(
                [
                    (direction, int(threshold), stratum_type, stratum)
                    for direction in DIRECTIONS
                    for threshold in thresholds
                    for stratum_type, stratum in stratum_pairs
                ],
                names=keys,
            )
            combined = totals.join(covered).join(matches).reindex(expected)
            integer_columns = ["eligible_rows", "total_rows", "complete_matches", "total_matches", "covered_players"]
            combined[integer_columns] = combined[integer_columns].fillna(0).astype("int64")
            combined["coverage_rate"] = combined["eligible_rows"].div(
                combined["total_rows"].where(combined["total_rows"].gt(0))
            ).fillna(0.0)
            combined["complete_match_rate"] = combined["complete_matches"].div(
                combined["total_matches"].where(combined["total_matches"].gt(0))
            ).fillna(0.0)
            combined = combined.reset_index()
            combined.insert(0, "scope", output_scope)
            combined.insert(2, "metric", metric)
            metric_tables.append(combined[COVERAGE_COLUMNS])
        scoped = pd.concat(metric_tables, ignore_index=True)
        scoped["_direction"] = scoped["direction"].map({value: index for index, value in enumerate(DIRECTIONS)})
        scoped["_metric"] = scoped["metric"].map({"direction_points": 0, "direction_matches": 1})
        scoped["_stratum"] = scoped[["stratum_type", "stratum"]].apply(
            lambda row: [("overall", "Overall"), *[("surface", x) for x in SURFACE_ORDER], *[("derived_period", x) for x in PERIOD_ORDER]].index(tuple(row)),
            axis=1,
        )
        scoped = scoped.sort_values(
            ["_direction", "_metric", "threshold", "_stratum"], kind="stable"
        ).drop(columns=["_direction", "_metric", "_stratum"])
        tables.append(scoped)
    coverage = pd.concat(tables, ignore_index=True)[COVERAGE_COLUMNS]
    if coverage.duplicated(COVERAGE_COLUMNS[:6]).any() or coverage.isna().any().any():
        raise ValueError("Cobertura con clave duplicada o nulos.")
    return coverage


def _quantile(series: pd.Series, probability: float) -> float | None:
    return None if series.empty else float(np.quantile(series.to_numpy(dtype=float), probability, method=QUANTILE_METHOD))


def build_stability(features: pd.DataFrame) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for role in ("server", "opponent"):
        for scope in SCOPES:
            points_column = f"{role}_direction_points"
            rate_column = "server_direction_raw_rate" if role == "server" else "opponent_allowed_server_raw_rate"
            low_column = "server_direction_wilson_low" if role == "server" else "opponent_allowed_server_wilson_low"
            high_column = "server_direction_wilson_high" if role == "server" else "opponent_allowed_server_wilson_high"
            base = features.loc[
                features["scope"].eq(scope),
                ["target_surface", "derived_period", "direction", points_column, rate_column, low_column, high_column],
            ].rename(columns={points_column: "points", rate_column: "rate", low_column: "low", high_column: "high"})
            base["width"] = base["high"] - base["low"]
            threshold_frames = [
                base.loc[base["points"].gt(0) & base["points"].ge(threshold)].assign(threshold_points=int(threshold))
                for threshold in POINT_THRESHOLDS
            ]
            threshold_long = pd.concat(threshold_frames, ignore_index=True)
            stratified = pd.concat(
                [
                    threshold_long.assign(stratum_type="overall", stratum="Overall"),
                    threshold_long.assign(stratum_type="surface", stratum=threshold_long["target_surface"]),
                    threshold_long.assign(stratum_type="derived_period", stratum=threshold_long["derived_period"]),
                ],
                ignore_index=True,
            )
            keys = ["direction", "threshold_points", "stratum_type", "stratum"]
            grouped = stratified.groupby(keys, sort=False, observed=True)
            aggregated = grouped.agg(
                observed_rows=("points", "size"), minimum_points=("points", "min"),
                median_points=("points", "median"),
                points_q1=("points", lambda values: values.quantile(.25, interpolation=QUANTILE_METHOD)),
                points_q3=("points", lambda values: values.quantile(.75, interpolation=QUANTILE_METHOD)),
                median_raw_rate=("rate", "median"),
                raw_rate_q1=("rate", lambda values: values.quantile(.25, interpolation=QUANTILE_METHOD)),
                raw_rate_q3=("rate", lambda values: values.quantile(.75, interpolation=QUANTILE_METHOD)),
                zero_rate_rows=("rate", lambda values: int(values.eq(0).sum())),
                one_rate_rows=("rate", lambda values: int(values.eq(1).sum())),
                median_wilson_width=("width", "median"),
                wilson_width_p90=("width", lambda values: values.quantile(.9, interpolation=QUANTILE_METHOD)),
            )
            stratum_pairs = [("overall", "Overall")]
            stratum_pairs.extend(("surface", value) for value in SURFACE_ORDER)
            stratum_pairs.extend(("derived_period", value) for value in PERIOD_ORDER)
            expected = pd.MultiIndex.from_tuples(
                [
                    (direction, int(threshold), stratum_type, stratum)
                    for direction in DIRECTIONS
                    for threshold in POINT_THRESHOLDS
                    for stratum_type, stratum in stratum_pairs
                ],
                names=keys,
            )
            aggregated = aggregated.reindex(expected)
            for column in ("observed_rows", "zero_rate_rows", "one_rate_rows"):
                aggregated[column] = aggregated[column].fillna(0).astype("int64")
            aggregated["minimum_points"] = aggregated["minimum_points"].astype("float64")
            aggregated["zero_rate_share"] = aggregated["zero_rate_rows"].div(
                aggregated["observed_rows"].where(aggregated["observed_rows"].gt(0))
            )
            aggregated["one_rate_share"] = aggregated["one_rate_rows"].div(
                aggregated["observed_rows"].where(aggregated["observed_rows"].gt(0))
            )
            aggregated = aggregated.reset_index()
            aggregated.insert(0, "scope", scope)
            aggregated.insert(0, "role", role)
            tables.append(aggregated[STABILITY_COLUMNS])
    stability = pd.concat(tables, ignore_index=True)[STABILITY_COLUMNS]
    if stability.duplicated(STABILITY_COLUMNS[:6]).any():
        raise ValueError("La clave de estabilidad no es unica.")
    return stability


def build_evidence_summary(features: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for direction in DIRECTIONS:
        for scope in SCOPES:
            subset = features[features["direction"].eq(direction) & features["scope"].eq(scope)]
            counts = subset["pair_evidence_state"].value_counts()
            total = int(len(subset))
            row = {"direction": direction, "scope": scope, "total": total}
            for state in PAIR_STATES:
                count = int(counts.get(state, 0)); row[state] = count; row[f"{state}_share"] = count / total if total else 0.0
            rows.append(row)
    return rows


def build_global_surface_comparison(features: pd.DataFrame) -> list[dict[str, Any]]:
    keys = ["target_match_id", "target_player", "direction"]
    global_rows = features[features["scope"].eq("global")]
    surface_rows = features[features["scope"].eq("surface")]
    paired = surface_rows.merge(global_rows, on=keys, suffixes=("_surface", "_global"), validate="one_to_one")
    strata = [("Overall", "Overall", pd.Series(True, index=paired.index))]
    strata.extend((surface, "Overall", paired["target_surface_surface"].eq(surface)) for surface in SURFACE_ORDER)
    strata.extend(("Overall", period, paired["derived_period_surface"].eq(period)) for period in PERIOD_ORDER)
    rows = []
    for role, rate in (("server", "server_direction_raw_rate"), ("opponent", "opponent_allowed_server_raw_rate")):
        for direction in DIRECTIONS:
            direction_rows = paired[paired["direction"].eq(direction)]
            for surface, period, mask in strata:
                subset = direction_rows.loc[mask.reindex(direction_rows.index, fill_value=False)]
                subset = subset.dropna(subset=[f"{rate}_surface", f"{rate}_global"])
                differences = subset[f"{rate}_surface"] - subset[f"{rate}_global"]
                absolute = differences.abs(); count = int(len(subset))
                correlation = None
                if count >= 2 and subset[f"{rate}_surface"].nunique() > 1 and subset[f"{rate}_global"].nunique() > 1:
                    correlation = float(subset[[f"{rate}_surface", f"{rate}_global"]].corr().iloc[0, 1])
                rows.append({
                    "role": role, "direction": direction, "target_surface": surface,
                    "derived_period": period, "comparable_rows": count,
                    "median_surface_minus_global": _quantile(differences, .5),
                    "q1_surface_minus_global": _quantile(differences, .25),
                    "q3_surface_minus_global": _quantile(differences, .75),
                    "median_absolute_difference": _quantile(absolute, .5),
                    "p90_absolute_difference": _quantile(absolute, .9),
                    "correlation": correlation,
                    "positive_share": None if not count else float(differences.gt(0).mean()),
                    "negative_share": None if not count else float(differences.lt(0).mean()),
                    "zero_share": None if not count else float(differences.eq(0).mean()),
                })
    return _json_value(rows)


def _source_contract(frame: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    return {
        "path": "data/processed/points_enriched.parquet", "columns_used": SOURCE_COLUMNS,
        "point_rows": int(len(frame)), "matches": int(len(targets)),
        "players": int(pd.unique(pd.concat([targets["player_1"], targets["player_2"]])).size),
        "date_range": {"first": targets["date"].min().date().isoformat(), "last": targets["date"].max().date().isoformat()},
    }


def build_summary(
    source: dict[str, Any], audit: dict[str, Any], features: pd.DataFrame,
    coverage: pd.DataFrame, stability: pd.DataFrame,
    diagnostics: list[dict[str, Any]], examples: list[dict[str, Any]],
    recorder: PerformanceRecorder | None = None,
) -> dict[str, Any]:
    recorder = recorder or PerformanceRecorder()
    evidence = build_evidence_summary(features)
    phase_started = time.perf_counter()
    global_surface = build_global_surface_comparison(features)
    recorder.record("global_surface_comparison", phase_started, len(global_surface))
    comparable = features[COMPARISON_COLUMNS].notna().all(axis=1)
    identity_error = (
        features.loc[comparable, "server_opponent_gap"]
        - features.loc[comparable, "server_excess_vs_population"]
        + features.loc[comparable, "opponent_allowed_excess_vs_population"]
    ).abs()
    overall_coverage = coverage[coverage["stratum_type"].eq("overall")].to_dict(orient="records")
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "source": source, "base_commit": BASE_COMMIT,
        "temporal_contract": {
            "rule": "history_date < target_match_date", "calendar_day_normalization": True,
            "same_day_matches_are_simultaneous": True, "emit_before_daily_update": True,
            "intraday_ordering_used": False, "match_id_only_stabilizes_output": True,
        },
        "outcome_contract": {
            "common_outcome": "P(server wins point)",
            "server_rate": "server_direction_wins / server_direction_points",
            "opponent_rate": "opponent_allowed_server_wins / opponent_direction_points",
            "population_rate": "population_direction_server_wins / population_direction_points",
            "returner_win_rate_published_as_direct_comparison": False,
        },
        "target_population": source,
        "eligible_history_population": audit,
        "feature_table_contract": {
            "unit": "target_match_id x target_player x direction x scope",
            "rows": int(len(features)), "columns": FEATURE_COLUMNS,
            "key": ["target_match_id", "target_player", "direction", "scope"],
            "full_feature_artifact_published": False,
        },
        "directions": list(DIRECTIONS), "scopes": list(SCOPES),
        "evidence_states": evidence,
        "wilson_contract": {"confidence_level": 0.95, "z": Z_975, "central_estimate": "raw_rate", "smoothing": False, "null_when_denominator_zero": True},
        "population_baseline_contract": {"scopes": {"global": "direction", "surface": "direction x target_surface"}, "daily_pre_update_state": True, "full_dataset_retrospective_baseline": False},
        "comparisons_contract": {"redundancy_acknowledged": True, "identity": "server_opponent_gap = server_excess_vs_population - opponent_allowed_excess_vs_population", "maximum_absolute_identity_error": 0.0 if identity_error.empty else float(identity_error.max())},
        "coverage_contract": {"rows": int(len(coverage)), "point_thresholds": list(POINT_THRESHOLDS), "match_thresholds": list(MATCH_THRESHOLDS), "thresholds_are_descriptive_only": True, "overall_results": _json_value(overall_coverage)},
        "stability_contract": {"rows": int(len(stability)), "quantile_method": QUANTILE_METHOD, "thresholds_are_descriptive_only": True, "wilson_width": "upper - lower"},
        "temporal_diagnostics": {"rows": diagnostics, "examples": examples, "total_violations": int(sum(row["violations"] for row in diagnostics))},
        "reconciliations": {
            "feature_rows": int(len(features)), "unique_feature_keys": int(features[ID_COLUMNS[:1] + ["target_player", "direction", "scope"]].drop_duplicates().shape[0]),
            "coverage_rows": int(len(coverage)), "stability_rows": int(len(stability)),
            "two_orientations_per_match": bool(features.groupby(["target_match_id", "direction", "scope"]).size().eq(2).all()),
            "strictly_prior_dates_only": not any(row["violations"] for row in diagnostics),
            "gap_identity_reconciled": bool(identity_error.le(1e-12).all()),
            "summary_recomputed_completely": True, "csv_payloads_reconciled_completely": True,
        },
        "key_results": {"coverage_overall": _json_value(overall_coverage), "evidence_states": evidence},
        "global_surface_comparison": global_surface,
        "methodological_limits": [
            "Descriptive historical associations; no causal interpretation or recommendation.",
            "No evidence threshold, filtering, smoothing, fallback or score is applied.",
            "Direction shares describe usage, not performance.",
            "Grass results may be fragmented and are reported as an evidence limitation only.",
        ],
    }
    summary = _json_value(summary)
    _assert_finite_json(summary)
    return summary


def _fingerprint(result: DirectionFeatureResult) -> str:
    digest = hashlib.sha256()
    for value in (result.source_contract, result.population_audit, result.summary):
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))
    for frame in (result.features, result.coverage, result.stability):
        digest.update(json.dumps([(str(c), str(d)) for c, d in frame.dtypes.items()]).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def validate_result(result: DirectionFeatureResult, snapshots: pd.DataFrame) -> None:
    diagnostics, examples = validate_feature_table(result.features, snapshots)
    expected_coverage = build_coverage(result.features)
    expected_stability = build_stability(result.features)
    pdt.assert_frame_equal(result.coverage, expected_coverage, check_exact=True)
    pdt.assert_frame_equal(result.stability, expected_stability, check_exact=True)
    expected_summary = build_summary(result.source_contract, result.population_audit, result.features, expected_coverage, expected_stability, diagnostics, examples)
    if result.summary != expected_summary:
        raise ValueError("El resumen completo no coincide con la derivacion canonica.")


def analyze_player_opponent_direction_features(
    points: pd.DataFrame,
    parser: Callable = parse_sequence,
    recorder: PerformanceRecorder | None = None,
) -> DirectionFeatureResult:
    recorder = recorder or PerformanceRecorder()
    phase_started = time.perf_counter()
    prepared_input = points.copy(deep=True)
    if "date" in prepared_input.columns:
        prepared_input["date"] = pd.to_datetime(
            prepared_input["date"], errors="coerce", format="mixed"
        )
    frame, targets, eligible, audit = prepare_historical_population(prepared_input, parser=parser)
    recorder.record("prepare_eligible_points", phase_started, len(eligible))
    phase_started = time.perf_counter()
    snapshots = build_historical_snapshots(targets, eligible)
    validate_snapshots(snapshots, targets)
    recorder.record("build_base_snapshots", phase_started, len(snapshots))
    features = build_direction_features(snapshots, eligible, recorder=recorder)
    phase_started = time.perf_counter()
    diagnostics, examples = validate_feature_table(features, snapshots)
    recorder.record("temporal_validation", phase_started, len(features))
    phase_started = time.perf_counter()
    coverage = build_coverage(features)
    recorder.record("calculate_coverage", phase_started, len(coverage))
    phase_started = time.perf_counter()
    stability = build_stability(features)
    recorder.record("calculate_stability", phase_started, len(stability))
    source = _source_contract(frame, targets)
    phase_started = time.perf_counter()
    summary = build_summary(
        source, audit, features, coverage, stability, diagnostics, examples,
        recorder=recorder,
    )
    recorder.record("build_summary", phase_started, 1)
    provisional = DirectionFeatureResult(features, coverage, stability, summary, source, audit)
    fingerprint = _fingerprint(provisional)
    result = DirectionFeatureResult(
        features, coverage, stability, summary, source, audit, fingerprint,
        recorder.snapshot(),
    )
    phase_started = time.perf_counter()
    validate_result(result, snapshots)
    recorder.record("validate_result", phase_started, len(features))
    return DirectionFeatureResult(
        features, coverage, stability, summary, source, audit, fingerprint,
        recorder.snapshot(),
    )


def serialize_artifacts(result: DirectionFeatureResult) -> tuple[bytes, bytes, bytes]:
    summary_payload = (json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    payloads = []
    for frame in (result.coverage, result.stability):
        buffer = io.StringIO(newline="")
        frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
        payloads.append(buffer.getvalue().encode("utf-8"))
    validate_serialized_payloads(result, summary_payload, payloads[0], payloads[1])
    return summary_payload, payloads[0], payloads[1]


def validate_serialized_payloads(result: DirectionFeatureResult, summary_payload: bytes, coverage_payload: bytes, stability_payload: bytes) -> None:
    if json.loads(summary_payload.decode("utf-8")) != result.summary:
        raise ValueError("El JSON serializado no coincide con el resumen canonico.")
    for name, payload, expected, columns in (
        ("coverage", coverage_payload, result.coverage, COVERAGE_COLUMNS),
        ("stability", stability_payload, result.stability, STABILITY_COLUMNS),
    ):
        loaded = pd.read_csv(io.BytesIO(payload))
        if loaded.columns.tolist() != columns or any(str(column).startswith("Unnamed") for column in loaded.columns):
            raise ValueError(f"Esquema CSV manipulado: {name}.")
        try:
            pdt.assert_frame_equal(loaded, expected, check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-15)
        except AssertionError as error:
            raise ValueError(f"Valores CSV manipulados: {name}.") from error


def _stage_bytes(target: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def write_artifacts(
    result: DirectionFeatureResult,
    summary_path: Path = SUMMARY_PATH,
    coverage_path: Path = COVERAGE_PATH,
    stability_path: Path = STABILITY_PATH,
    recorder: PerformanceRecorder | None = None,
) -> None:
    recorder = recorder or PerformanceRecorder()
    paths = (summary_path, coverage_path, stability_path)
    if len({path.resolve() for path in paths}) != 3:
        raise ValueError("Los tres artefactos requieren rutas distintas.")
    if not result.publication_fingerprint or result.publication_fingerprint != _fingerprint(result):
        raise ValueError("El resultado fue modificado tras la validacion.")
    phase_started = time.perf_counter()
    payloads = serialize_artifacts(result)
    recorder.record("serialize_payloads", phase_started, sum(len(payload) for payload in payloads))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    originals = {path: path.read_bytes() if path.exists() else None for path in paths}
    staged: list[Path] = []
    replaced: list[Path] = []
    phase_started = time.perf_counter()
    try:
        staged = [_stage_bytes(path, payload) for path, payload in zip(paths, payloads)]
        for temporary, target in zip(staged, paths):
            os.replace(temporary, target); replaced.append(target)
    except BaseException:
        for target in reversed(replaced):
            original = originals[target]
            if original is None:
                target.unlink(missing_ok=True)
            else:
                recovery = _stage_bytes(target, original); os.replace(recovery, target)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    recorder.record("publish_artifacts", phase_started, len(paths))


def _hash_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()}


def validate_performance_log_path(log_path: Path | None) -> None:
    """Impide que el diagnostico efimero se escriba dentro del repositorio."""

    if log_path is None:
        return
    resolved_log = log_path.resolve()
    resolved_root = ROOT.resolve()
    if resolved_log == resolved_root or resolved_root in resolved_log.parents:
        raise ValueError("El log de rendimiento debe ubicarse fuera del repositorio.")
    if not resolved_log.parent.exists():
        raise FileNotFoundError("El directorio externo del log de rendimiento no existe.")


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--performance-log", type=Path)
    arguments = argument_parser.parse_args()
    validate_performance_log_path(arguments.performance_log)
    if REPRO_DIR.exists():
        raise FileExistsError(f"El temporal de reproducibilidad ya existe: {REPRO_DIR.name}")
    recorder = PerformanceRecorder(enabled=True, log_path=arguments.performance_log)
    phase_started = time.perf_counter()
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    recorder.record("read_parquet", phase_started, len(points))
    result = analyze_player_opponent_direction_features(points, recorder=recorder)
    write_artifacts(result, recorder=recorder)
    REPRO_DIR.mkdir(parents=False)
    repro_paths = (REPRO_DIR / SUMMARY_PATH.name, REPRO_DIR / COVERAGE_PATH.name, REPRO_DIR / STABILITY_PATH.name)
    write_artifacts(result, *repro_paths)
    permanent_paths = (SUMMARY_PATH, COVERAGE_PATH, STABILITY_PATH)
    reproducible = [permanent.read_bytes() == temporary.read_bytes() for permanent, temporary in zip(permanent_paths, repro_paths)]
    if not all(reproducible):
        raise ValueError("La reserializacion desde memoria no es identica byte a byte.")
    print(json.dumps({"summary": result.summary, "performance_metrics": recorder.snapshot(), "artifacts": [_hash_file(path) for path in permanent_paths], "byte_identical_reproduction": reproducible, "repro_directory": REPRO_DIR.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
