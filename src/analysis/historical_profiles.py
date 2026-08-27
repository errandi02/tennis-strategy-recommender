"""Perfiles historicos descriptivos por jugador sin fuga temporal.

La unidad de salida interna es ``target_match_id x target_player``. Los
artefactos permanentes contienen solo el contrato y cobertura agregada; los
snapshots completos se conservan en memoria para consumidores posteriores.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.second_serve_direction_analysis import (
    POINTS_FILE,
    SOURCE_COLUMNS,
    SURFACE_ORDER,
    derive_period,
    prepare_points,
)
from src.analysis.second_serve_direction_baseline import DIRECTION_LABELS
from src.parsing.serve_sequence import parse_sequence
from src.parsing.serve_sequence_rules import GRAMMAR_VERSION, PARSER_VERSION


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "historical_profiles_summary.json"
COVERAGE_PATH = TABLES_DIR / "historical_profiles_coverage.csv"

ANALYSIS_VERSION = "1.0.0"
SERVE_NUMBER = 2
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
DIRECTION_ORDER = ("wide", "body", "t")
MATCH_THRESHOLDS = (0, 1, 3, 5, 10, 20, 50)
POINT_THRESHOLDS = (0, 25, 50, 100, 250, 500)
SCOPE_ORDER = (
    "server_global",
    "server_surface",
    "opponent_return_global",
    "opponent_return_surface",
    "both_global",
    "both_surface",
)

SERVER_COUNT_SUFFIXES = (
    "matches",
    "points",
    "wins",
    "wide_points",
    "wide_wins",
    "body_points",
    "body_wins",
    "t_points",
    "t_wins",
)
RETURN_COUNT_SUFFIXES = (
    "matches",
    "points",
    "server_wins",
    "return_wins",
    "wide_points",
    "wide_server_wins",
    "body_points",
    "body_server_wins",
    "t_points",
    "t_server_wins",
)

SNAPSHOT_ID_COLUMNS = [
    "target_match_id",
    "target_date",
    "target_player",
    "opponent",
    "target_surface",
]
SNAPSHOT_COLUMNS = SNAPSHOT_ID_COLUMNS + [
    *[f"server_global_{name}" for name in SERVER_COUNT_SUFFIXES],
    "server_global_first_history_date",
    "server_global_last_history_date",
    *[f"server_surface_{name}" for name in SERVER_COUNT_SUFFIXES],
    "server_surface_first_history_date",
    "server_surface_last_history_date",
    *[f"opponent_return_global_{name}" for name in RETURN_COUNT_SUFFIXES],
    "opponent_return_global_first_history_date",
    "opponent_return_global_last_history_date",
    *[f"opponent_return_surface_{name}" for name in RETURN_COUNT_SUFFIXES],
    "opponent_return_surface_first_history_date",
    "opponent_return_surface_last_history_date",
]

COVERAGE_COLUMNS = [
    "scope",
    "metric",
    "threshold",
    "stratum_type",
    "stratum",
    "eligible_snapshots",
    "total_snapshots",
    "coverage_rate",
    "complete_matches",
    "total_matches",
    "complete_match_rate",
    "covered_players",
]


@dataclass(frozen=True)
class HistoricalProfileResult:
    """Resultado inmutable de una unica construccion cronologica."""

    snapshots: pd.DataFrame
    coverage: pd.DataFrame
    summary: dict[str, Any]
    publication_fingerprint: str = ""


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


def _publication_fingerprint(
    snapshots: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    """Vincula la publicacion al resultado que supero las reconciliaciones."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    )
    for frame in (snapshots, coverage):
        schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
        digest.update(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _validate_match_metadata(frame: pd.DataFrame) -> None:
    metadata = ["date", "surface", "player_1", "player_2"]
    varying = frame.groupby("match_id", sort=False)[metadata].nunique(dropna=False).gt(1)
    if varying.any().any():
        bad = sorted(varying.index[varying.any(axis=1)].astype(str).tolist())
        raise ValueError(f"Metadatos no constantes dentro del partido: {bad[:10]}")


def prepare_historical_population(
    points: pd.DataFrame,
    parser: Callable = parse_sequence,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Valida la fuente y deriva la poblacion historica elegible aprobada."""

    parser_cache: dict[tuple[str, int], Any] = {}

    def cached_parser(sequence: str, serve_number: int):
        key = (sequence, serve_number)
        if key not in parser_cache:
            parser_cache[key] = parser(sequence, serve_number=serve_number)
        return parser_cache[key]

    frame, prepared = prepare_points(points, parser=cached_parser)
    # ``date`` is a calendar-day contract. Any time component supplied by an
    # in-memory fixture is deliberately ignored before grouping or ordering.
    frame["date"] = frame["date"].dt.normalize()
    _validate_match_metadata(frame)

    targets = (
        frame[["match_id", "date", "surface", "player_1", "player_2"]]
        .drop_duplicates("match_id")
        .sort_values(["date", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    substantive = prepared["substantive"].copy()
    substantive["date"] = substantive["date"].dt.normalize()
    eligible = substantive[substantive["direction_code"].isin(DIRECTION_LABELS)].copy()
    eligible["direction"] = eligible["direction_code"].map(DIRECTION_LABELS).str.lower()
    eligible = eligible[
        [
            "match_id",
            "point_number",
            "date",
            "surface",
            "server_player",
            "returner_player",
            "server_won_point",
            "direction",
        ]
    ].sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)

    audit = {
        "source_point_rows": int(len(frame)),
        "source_matches": int(targets["match_id"].nunique()),
        "source_players": int(pd.unique(pd.concat([targets["player_1"], targets["player_2"]])).size),
        "eligible_points": int(len(eligible)),
        "eligible_matches": int(eligible["match_id"].nunique()),
        "eligible_servers": int(eligible["server_player"].nunique()),
        "eligible_returners": int(eligible["returner_player"].nunique()),
        "server_wins": int(eligible["server_won_point"].sum()),
        "excluded_direction_0": int(substantive["direction_code"].eq("0").sum()),
        "excluded_without_recognized_direction": int(substantive["direction_code"].isna().sum()),
        "parser_cache_entries": int(len(parser_cache)),
        "parser_cache_key": "(sequence_text, serve_number)",
    }
    for direction in DIRECTION_ORDER:
        population = eligible[eligible["direction"].eq(direction)]
        audit[f"{direction}_points"] = int(len(population))
        audit[f"{direction}_server_wins"] = int(population["server_won_point"].sum())
    return frame, targets, eligible, audit


def _build_role_events(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = eligible.copy()
    for direction in DIRECTION_ORDER:
        work[f"{direction}_points"] = work["direction"].eq(direction).astype("int64")
        work[f"{direction}_wins"] = (
            work["direction"].eq(direction) & work["server_won_point"]
        ).astype("int64")

    server = (
        work.groupby(["server_player", "match_id", "date", "surface"], sort=False, observed=True)
        .agg(
            points=("point_number", "size"),
            wins=("server_won_point", "sum"),
            **{
                name: (name, "sum")
                for name in (
                    "wide_points", "wide_wins", "body_points", "body_wins", "t_points", "t_wins"
                )
            },
        )
        .reset_index()
        .rename(columns={"server_player": "player"})
    )
    server.insert(4, "matches", 1)

    returner = (
        work.groupby(["returner_player", "match_id", "date", "surface"], sort=False, observed=True)
        .agg(
            points=("point_number", "size"),
            server_wins=("server_won_point", "sum"),
            wide_points=("wide_points", "sum"),
            wide_server_wins=("wide_wins", "sum"),
            body_points=("body_points", "sum"),
            body_server_wins=("body_wins", "sum"),
            t_points=("t_points", "sum"),
            t_server_wins=("t_wins", "sum"),
        )
        .reset_index()
        .rename(columns={"returner_player": "player"})
    )
    returner.insert(4, "matches", 1)
    returner["return_wins"] = returner["points"] - returner["server_wins"]
    return server, returner


def _empty_state(count_suffixes: tuple[str, ...]) -> dict[str, Any]:
    return {**{name: 0 for name in count_suffixes}, "first_history_date": None, "last_history_date": None}


def _update_state(
    store: dict[Any, dict[str, Any]],
    key: Any,
    event: pd.Series,
    suffixes: tuple[str, ...],
) -> None:
    state = store.setdefault(key, _empty_state(suffixes))
    for name in suffixes:
        state[name] += int(event[name])
    date = pd.Timestamp(event["date"])
    if state["first_history_date"] is None:
        state["first_history_date"] = date
    state["last_history_date"] = date


def _snapshot_state(
    row: dict[str, Any],
    prefix: str,
    store: dict[Any, dict[str, Any]],
    key: Any,
    suffixes: tuple[str, ...],
) -> None:
    state = store.get(key, _empty_state(suffixes))
    for name in suffixes:
        row[f"{prefix}_{name}"] = int(state[name])
    row[f"{prefix}_first_history_date"] = state["first_history_date"]
    row[f"{prefix}_last_history_date"] = state["last_history_date"]


def build_historical_snapshots(targets: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Construye snapshots antes de actualizar cada bloque de fecha."""

    server_events, return_events = _build_role_events(eligible)
    server_by_date = {date: group for date, group in server_events.groupby("date", sort=False)}
    return_by_date = {date: group for date, group in return_events.groupby("date", sort=False)}

    server_global: dict[Any, dict[str, Any]] = {}
    server_surface: dict[Any, dict[str, Any]] = {}
    return_global: dict[Any, dict[str, Any]] = {}
    return_surface: dict[Any, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for date, daily_targets in targets.groupby("date", sort=True):
        daily_targets = daily_targets.sort_values("match_id", kind="stable")
        for match in daily_targets.itertuples(index=False):
            orientations = ((match.player_1, match.player_2), (match.player_2, match.player_1))
            for target_player, opponent in orientations:
                row: dict[str, Any] = {
                    "target_match_id": match.match_id,
                    "target_date": pd.Timestamp(match.date),
                    "target_player": target_player,
                    "opponent": opponent,
                    "target_surface": match.surface,
                }
                _snapshot_state(row, "server_global", server_global, target_player, SERVER_COUNT_SUFFIXES)
                _snapshot_state(
                    row,
                    "server_surface",
                    server_surface,
                    (target_player, match.surface),
                    SERVER_COUNT_SUFFIXES,
                )
                _snapshot_state(
                    row,
                    "opponent_return_global",
                    return_global,
                    opponent,
                    RETURN_COUNT_SUFFIXES,
                )
                _snapshot_state(
                    row,
                    "opponent_return_surface",
                    return_surface,
                    (opponent, match.surface),
                    RETURN_COUNT_SUFFIXES,
                )
                rows.append(row)

        for _, event in server_by_date.get(date, pd.DataFrame()).iterrows():
            _update_state(server_global, event["player"], event, SERVER_COUNT_SUFFIXES)
            _update_state(server_surface, (event["player"], event["surface"]), event, SERVER_COUNT_SUFFIXES)
        for _, event in return_by_date.get(date, pd.DataFrame()).iterrows():
            _update_state(return_global, event["player"], event, RETURN_COUNT_SUFFIXES)
            _update_state(return_surface, (event["player"], event["surface"]), event, RETURN_COUNT_SUFFIXES)

    snapshots = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    count_columns = [column for column in snapshots if column.endswith(tuple(SERVER_COUNT_SUFFIXES + RETURN_COUNT_SUFFIXES))]
    # The suffix expression also catches all intended count fields; ID/date fields do not match.
    for column in count_columns:
        snapshots[column] = snapshots[column].astype("int64")
    return snapshots.reset_index(drop=True)


def _prior_match_upper_bounds(targets: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], int]:
    participants = pd.concat(
        [
            targets[["match_id", "date", "player_1"]].rename(columns={"player_1": "player"}),
            targets[["match_id", "date", "player_2"]].rename(columns={"player_2": "player"}),
        ],
        ignore_index=True,
    )
    result: dict[tuple[str, pd.Timestamp], int] = {}
    for player, group in participants.groupby("player", sort=False):
        counts = group.groupby("date")["match_id"].nunique().sort_index()
        prior = counts.cumsum().shift(fill_value=0)
        result.update({(player, pd.Timestamp(date)): int(value) for date, value in prior.items()})
    return result


def validate_snapshots(snapshots: pd.DataFrame, targets: pd.DataFrame) -> None:
    if list(snapshots.columns) != SNAPSHOT_COLUMNS:
        raise ValueError("El esquema u orden de columnas de snapshots no coincide con el contrato.")
    if len(snapshots) != 2 * len(targets):
        raise ValueError("Cada partido objetivo debe producir exactamente dos snapshots.")
    if snapshots.duplicated(["target_match_id", "target_player"]).any():
        raise ValueError("La clave (target_match_id, target_player) no es unica.")
    if not snapshots.groupby("target_match_id").size().eq(2).all():
        raise ValueError("Cada target_match_id debe aparecer exactamente dos veces.")

    expected = set()
    for match in targets.itertuples(index=False):
        expected.add((match.match_id, match.player_1, match.player_2, pd.Timestamp(match.date), match.surface))
        expected.add((match.match_id, match.player_2, match.player_1, pd.Timestamp(match.date), match.surface))
    observed = set(
        snapshots[["target_match_id", "target_player", "opponent", "target_date", "target_surface"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError("Los snapshots no reproducen exactamente los dos participantes de cada partido.")

    count_columns = [
        column
        for column in snapshots.columns
        if column not in SNAPSHOT_ID_COLUMNS and not column.endswith("history_date")
    ]
    if snapshots[count_columns].isna().any().any():
        raise ValueError("Los conteos historicos no pueden contener nulos.")
    if not all(pd.api.types.is_integer_dtype(snapshots[column]) for column in count_columns):
        raise TypeError("Los conteos historicos deben ser enteros.")
    if snapshots[count_columns].lt(0).any().any():
        raise ValueError("Los conteos historicos no pueden ser negativos.")

    for prefix, suffixes in (
        ("server_global", SERVER_COUNT_SUFFIXES),
        ("server_surface", SERVER_COUNT_SUFFIXES),
        ("opponent_return_global", RETURN_COUNT_SUFFIXES),
        ("opponent_return_surface", RETURN_COUNT_SUFFIXES),
    ):
        points = snapshots[f"{prefix}_points"]
        matches = snapshots[f"{prefix}_matches"]
        first = snapshots[f"{prefix}_first_history_date"]
        last = snapshots[f"{prefix}_last_history_date"]
        if first.isna().ne(points.eq(0)).any() or last.isna().ne(points.eq(0)).any():
            raise ValueError(f"Fechas historicas nulas incoherentes en {prefix}.")
        populated = points.gt(0)
        if (first[populated] > last[populated]).any() or (
            last[populated] >= snapshots.loc[populated, "target_date"]
        ).any():
            raise ValueError(f"Fuga temporal detectada en {prefix}.")
        if matches.gt(points).any():
            raise ValueError(f"Mas partidos que puntos en {prefix}.")
        if prefix.startswith("server_"):
            if snapshots[f"{prefix}_wins"].gt(points).any():
                raise ValueError(f"Victorias superiores a puntos en {prefix}.")
            if sum((snapshots[f"{prefix}_{d}_points"] for d in DIRECTION_ORDER)).ne(points).any():
                raise ValueError(f"Los puntos por direccion no reconcilian en {prefix}.")
            if sum((snapshots[f"{prefix}_{d}_wins"] for d in DIRECTION_ORDER)).ne(
                snapshots[f"{prefix}_wins"]
            ).any():
                raise ValueError(f"Las victorias por direccion no reconcilian en {prefix}.")
        else:
            if (
                snapshots[f"{prefix}_server_wins"] + snapshots[f"{prefix}_return_wins"]
            ).ne(points).any():
                raise ValueError(f"Victorias de saque y resto no reconcilian en {prefix}.")
            if sum((snapshots[f"{prefix}_{d}_points"] for d in DIRECTION_ORDER)).ne(points).any():
                raise ValueError(f"Los puntos por direccion no reconcilian en {prefix}.")
            if sum((snapshots[f"{prefix}_{d}_server_wins"] for d in DIRECTION_ORDER)).ne(
                snapshots[f"{prefix}_server_wins"]
            ).any():
                raise ValueError(f"Las victorias por direccion no reconcilian en {prefix}.")

    for role in ("server", "opponent_return"):
        for suffix in (
            SERVER_COUNT_SUFFIXES if role == "server" else RETURN_COUNT_SUFFIXES
        ):
            if snapshots[f"{role}_surface_{suffix}"].gt(snapshots[f"{role}_global_{suffix}"]).any():
                raise ValueError(f"El historial de superficie supera al global: {role}_{suffix}.")

    upper = _prior_match_upper_bounds(targets)
    server_upper = np.array(
        [upper[(row.target_player, row.target_date)] for row in snapshots.itertuples(index=False)]
    )
    return_upper = np.array(
        [upper[(row.opponent, row.target_date)] for row in snapshots.itertuples(index=False)]
    )
    if (snapshots["server_global_matches"].to_numpy() > server_upper).any():
        raise ValueError("El historial de saque supera los partidos estrictamente anteriores.")
    if (snapshots["opponent_return_global_matches"].to_numpy() > return_upper).any():
        raise ValueError("El historial de resto supera los partidos estrictamente anteriores.")


def _scope_mask(frame: pd.DataFrame, scope: str, metric: str, threshold: int) -> pd.Series:
    column = "matches" if metric == "historical_matches" else "points"
    if scope == "server_global":
        return frame[f"server_global_{column}"].ge(threshold)
    if scope == "server_surface":
        return frame[f"server_surface_{column}"].ge(threshold)
    if scope == "opponent_return_global":
        return frame[f"opponent_return_global_{column}"].ge(threshold)
    if scope == "opponent_return_surface":
        return frame[f"opponent_return_surface_{column}"].ge(threshold)
    if scope == "both_global":
        return frame[f"server_global_{column}"].ge(threshold) & frame[
            f"opponent_return_global_{column}"
        ].ge(threshold)
    if scope == "both_surface":
        return frame[f"server_surface_{column}"].ge(threshold) & frame[
            f"opponent_return_surface_{column}"
        ].ge(threshold)
    raise ValueError(f"Scope desconocido: {scope}")


def build_coverage_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    work = snapshots.copy()
    work["derived_period"] = work["target_date"].dt.year.map(derive_period)
    strata: list[tuple[str, str, pd.Series]] = [
        ("overall", "Overall", pd.Series(True, index=work.index))
    ]
    strata.extend(("surface", surface, work["target_surface"].eq(surface)) for surface in SURFACE_ORDER)
    strata.extend(("derived_period", period, work["derived_period"].eq(period)) for period in PERIOD_ORDER)

    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        for metric, thresholds in (
            ("historical_matches", MATCH_THRESHOLDS),
            ("historical_points", POINT_THRESHOLDS),
        ):
            for threshold in thresholds:
                passes = _scope_mask(work, scope, metric, threshold)
                for stratum_type, stratum, in_stratum in strata:
                    population = work.loc[in_stratum]
                    selected = passes.loc[in_stratum]
                    total_snapshots = int(len(population))
                    eligible_snapshots = int(selected.sum())
                    per_match = selected.groupby(population["target_match_id"]).agg(["size", "sum"])
                    if not per_match.empty and not per_match["size"].eq(2).all():
                        raise ValueError("Un estrato no contiene exactamente dos orientaciones por partido.")
                    total_matches = int(len(per_match))
                    complete_matches = int(per_match["sum"].eq(2).sum())
                    rows.append(
                        {
                            "scope": scope,
                            "metric": metric,
                            "threshold": int(threshold),
                            "stratum_type": stratum_type,
                            "stratum": stratum,
                            "eligible_snapshots": eligible_snapshots,
                            "total_snapshots": total_snapshots,
                            "coverage_rate": (
                                eligible_snapshots / total_snapshots if total_snapshots else 0.0
                            ),
                            "complete_matches": complete_matches,
                            "total_matches": total_matches,
                            "complete_match_rate": (
                                complete_matches / total_matches if total_matches else 0.0
                            ),
                            "covered_players": int(population.loc[selected, "target_player"].nunique()),
                        }
                    )
    coverage = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    if coverage.duplicated(["scope", "metric", "threshold", "stratum_type", "stratum"]).any():
        raise ValueError("La clave de la tabla de cobertura no es unica.")
    if coverage.isna().any().any():
        raise ValueError("La cobertura no puede contener nulos.")
    return coverage


def build_summary(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    eligible: pd.DataFrame,
    audit: dict[str, Any],
    snapshots: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    overall = coverage[coverage["stratum_type"].eq("overall")].to_dict(orient="records")
    for row in overall:
        row["coverage_rate"] = float(row["coverage_rate"])
        row["complete_match_rate"] = float(row["complete_match_rate"])
    summary: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "source": {
            "path": "data/processed/points_enriched.parquet",
            "columns_used": SOURCE_COLUMNS,
            "point_rows": int(len(frame)),
            "matches": int(len(targets)),
            "date_range": {
                "first": targets["date"].min().date().isoformat(),
                "last": targets["date"].max().date().isoformat(),
            },
        },
        "parser_contract": {
            "parser_version": PARSER_VERSION,
            "grammar_version": GRAMMAR_VERSION,
            "serve_number": SERVE_NUMBER,
            "cache_key": "(sequence_text, serve_number)",
            "eligible_direction_codes": {"4": "wide", "5": "body", "6": "T"},
            "warnings_or_residual_after_recognized_prefix_exclude_point": False,
        },
        "temporal_contract": {
            "rule": "history_date < target_match_date",
            "same_player_same_date_is_simultaneous": True,
            "intraday_ordering_used": False,
            "target_match_excluded": True,
            "history_windows": "all_strictly_prior_history",
        },
        "eligible_history_population": audit,
        "snapshot_population": {
            "unit": "target_match_id x target_player",
            "snapshots": int(len(snapshots)),
            "target_matches": int(snapshots["target_match_id"].nunique()),
            "orientations_per_match": 2,
            "columns": SNAPSHOT_COLUMNS,
            "full_snapshot_artifact_published": False,
        },
        "feature_contract": {
            "values": "raw_counts_only",
            "rates_or_smoothing": False,
            "thresholds_are_descriptive_only": True,
            "fallbacks": False,
            "head_to_head": False,
            "models_or_recommendations": False,
            "roles": ["target_player_as_server", "opponent_as_returner"],
            "contexts": ["global", "target_surface"],
        },
        "coverage_contract": {
            "unit": "target_match_id x target_player",
            "scopes": list(SCOPE_ORDER),
            "match_thresholds": list(MATCH_THRESHOLDS),
            "point_thresholds": list(POINT_THRESHOLDS),
            "both_definition": "target server and opponent return histories pass simultaneously",
            "complete_match_definition": "both target-player orientations pass",
            "strata": {
                "overall": ["Overall"],
                "surface": list(SURFACE_ORDER),
                "derived_period": list(PERIOD_ORDER),
            },
            "rows": int(len(coverage)),
            "overall_results": overall,
        },
        "reconciliations": {
            "two_snapshots_per_match": True,
            "strictly_prior_dates_only": True,
            "same_date_history_excluded": True,
            "target_match_self_reference_count": 0,
            "direction_points_reconciled": True,
            "direction_wins_reconciled": True,
            "return_outcomes_reconciled": True,
            "surface_not_above_global": True,
            "coverage_recomputed_exactly": True,
        },
        "limitations": [
            "Descriptive raw counts; no rates, smoothing, thresholds for modelling or recommendations.",
            "Matches on the same date are treated as simultaneous because no reliable intraday ordering is used.",
            "Exact player identity is inherited from the processed parquet; no alias resolution is applied here.",
        ],
    }
    _assert_finite_json(summary)
    return summary


def validate_published_contract(
    result: HistoricalProfileResult,
    targets: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    frame: pd.DataFrame | None = None,
    audit: dict[str, Any] | None = None,
) -> None:
    validate_snapshots(result.snapshots, targets)
    expected_coverage = build_coverage_table(result.snapshots)
    pdt.assert_frame_equal(result.coverage, expected_coverage, check_exact=True)
    summary = result.summary
    if summary["snapshot_population"]["snapshots"] != 2 * len(targets):
        raise ValueError("El resumen no reconcilia el numero de snapshots.")
    if summary["eligible_history_population"]["eligible_points"] != len(eligible):
        raise ValueError("El resumen no reconcilia los puntos elegibles.")
    if summary["coverage_contract"]["rows"] != len(expected_coverage):
        raise ValueError("El resumen no reconcilia las filas de cobertura.")
    expected_overall = expected_coverage[
        expected_coverage["stratum_type"].eq("overall")
    ].to_dict(orient="records")
    if summary["coverage_contract"]["overall_results"] != expected_overall:
        raise ValueError("El resumen no reconcilia la cobertura Overall.")
    if (frame is None) != (audit is None):
        raise ValueError("frame y audit deben proporcionarse conjuntamente.")
    if frame is not None and audit is not None:
        expected_summary = build_summary(
            frame,
            targets,
            eligible,
            audit,
            result.snapshots,
            expected_coverage,
        )
        if summary != expected_summary:
            raise ValueError("El resumen completo no coincide con la derivacion canonica.")
    _assert_finite_json(summary)


def analyze_historical_profiles(
    points: pd.DataFrame,
    parser: Callable = parse_sequence,
) -> HistoricalProfileResult:
    frame, targets, eligible, audit = prepare_historical_population(points, parser=parser)
    snapshots = build_historical_snapshots(targets, eligible)
    validate_snapshots(snapshots, targets)
    coverage = build_coverage_table(snapshots)
    summary = build_summary(frame, targets, eligible, audit, snapshots, coverage)
    fingerprint = _publication_fingerprint(snapshots, coverage, summary)
    result = HistoricalProfileResult(
        snapshots=snapshots,
        coverage=coverage,
        summary=summary,
        publication_fingerprint=fingerprint,
    )
    validate_published_contract(result, targets, eligible, frame=frame, audit=audit)
    return result


def _stage_bytes(target: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def write_artifacts(
    result: HistoricalProfileResult,
    summary_path: Path = SUMMARY_PATH,
    coverage_path: Path = COVERAGE_PATH,
) -> None:
    """Publica conjuntamente dos artefactos de un resultado ya validado."""

    if summary_path.resolve() == coverage_path.resolve():
        raise ValueError("JSON y CSV requieren rutas de salida distintas.")
    expected_fingerprint = _publication_fingerprint(
        result.snapshots,
        result.coverage,
        result.summary,
    )
    if not result.publication_fingerprint or result.publication_fingerprint != expected_fingerprint:
        raise ValueError("El resultado fue modificado despues de superar las reconciliaciones.")

    summary_payload = (
        json.dumps(
            result.summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    coverage_buffer = io.StringIO(newline="")
    result.coverage.to_csv(
        coverage_buffer,
        index=False,
        lineterminator="\n",
        na_rep="",
    )
    coverage_payload = coverage_buffer.getvalue().encode("utf-8")

    # Both payloads are fully materialised and checked before either permanent
    # target is replaced.
    if json.loads(summary_payload.decode("utf-8")) != result.summary:
        raise ValueError("La serializacion JSON no conserva el resumen validado.")
    serialized_coverage = pd.read_csv(io.BytesIO(coverage_payload))
    if (
        serialized_coverage.columns.tolist() != COVERAGE_COLUMNS
        or len(serialized_coverage) != len(result.coverage)
        or serialized_coverage.isna().any().any()
    ):
        raise ValueError("La serializacion CSV no conserva el contrato validado.")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    originals = {
        summary_path: summary_path.read_bytes() if summary_path.exists() else None,
        coverage_path: coverage_path.read_bytes() if coverage_path.exists() else None,
    }
    staged_summary: Path | None = None
    staged_coverage: Path | None = None
    replaced: list[Path] = []
    try:
        staged_summary = _stage_bytes(summary_path, summary_payload)
        staged_coverage = _stage_bytes(coverage_path, coverage_payload)
        os.replace(staged_summary, summary_path)
        replaced.append(summary_path)
        os.replace(staged_coverage, coverage_path)
        replaced.append(coverage_path)
    except BaseException:
        for target in reversed(replaced):
            original = originals[target]
            if original is None:
                target.unlink(missing_ok=True)
            else:
                recovery = _stage_bytes(target, original)
                os.replace(recovery, target)
        raise
    finally:
        if staged_summary is not None:
            staged_summary.unlink(missing_ok=True)
        if staged_coverage is not None:
            staged_coverage.unlink(missing_ok=True)


def main() -> None:
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    result = analyze_historical_profiles(points)
    write_artifacts(result)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
