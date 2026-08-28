"""Constructor reproducible de splits cronologicos a nivel de partido.

Este modulo no construye features, outcomes, modelos ni recomendaciones. Su
unica fuente es ``points_enriched.parquet`` y publica solo agregados.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import pandas.testing as pdt


ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "chronological_validation_summary.json"
BY_YEAR_PATH = TABLES_DIR / "chronological_validation_by_year.csv"
FOLDS_PATH = TABLES_DIR / "chronological_validation_folds.csv"

SOURCE_COLUMNS = ["match_id", "date", "player_1", "player_2", "surface"]
MATCH_COLUMNS = SOURCE_COLUMNS + ["split"]
ALLOWED_SURFACES = ("Hard", "Clay", "Grass")
SPLIT_ORDER = ("train", "validation", "test")
VALIDATION_YEARS = (2020, 2021, 2022, 2023)

EXPECTED_SOURCE_ROWS = 1_280_408
EXPECTED_MATCHES = 7_524
EXPECTED_FIRST_DATE = pd.Timestamp("1960-05-29")
EXPECTED_LAST_DATE = pd.Timestamp("2026-05-21")
TRAIN_END = pd.Timestamp("2019-12-31")
VALIDATION_START = pd.Timestamp("2020-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = EXPECTED_LAST_DATE
EXPECTED_SPLIT_MATCHES = {"train": 4_188, "validation": 1_805, "test": 1_531}

BY_YEAR_COLUMNS = [
    "year", "split", "is_partial_year", "first_date", "last_date", "matches",
    "players", "new_players_at_year_start", "previously_seen_players_at_year_start",
    "both_players_prior_matches",
    "one_player_prior_matches", "neither_player_prior_matches",
    "both_players_prior_rate", "hard_matches", "clay_matches", "grass_matches",
]
FOLD_COLUMNS = [
    "protocol", "fold_id", "is_final_test", "test_usage_policy",
    "history_update_policy", "feature_definition_policy", "evidence_policy",
    "scoring_parameter_policy", "train_cutoff", "train_last_observed_date", "evaluation_start",
    "evaluation_end", "evaluation_first_observed_date",
    "evaluation_last_observed_date", "train_matches", "evaluation_matches",
    "evaluation_players", "unseen_players_at_evaluation_start",
    "seen_players_at_evaluation_start", "both_players_prior_matches",
    "one_player_prior_matches", "neither_player_prior_matches",
    "both_players_prior_rate", "hard_matches", "clay_matches", "grass_matches",
]


@dataclass(frozen=True)
class ChronologicalValidationResult:
    source_rows: int
    matches: pd.DataFrame
    by_year: pd.DataFrame
    folds: pd.DataFrame
    summary: dict[str, Any]


def _validate_exact_text(series: pd.Series, field: str) -> None:
    if series.isna().any():
        raise ValueError(f"{field} contiene valores nulos.")
    invalid_type = series.map(lambda value: not isinstance(value, str))
    if invalid_type.any():
        values = sorted({type(value).__name__ for value in series.loc[invalid_type]})
        raise TypeError(f"{field} debe contener solo cadenas; tipos inesperados: {values}")
    empty = series.eq("")
    if empty.any():
        raise ValueError(f"{field} contiene cadenas vacias.")
    whitespace = series.map(lambda value: value != value.strip())
    if whitespace.any():
        raise ValueError(f"{field} contiene espacios iniciales o finales.")


def _strict_dates(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("date contiene valores nulos.")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError("date no puede contener zona horaria.")
    if pd.api.types.is_datetime64_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        kinds = set()
        for value in series:
            if isinstance(value, str):
                kinds.add("string")
            elif isinstance(value, (pd.Timestamp, datetime, date)) and not isinstance(value, bool):
                kinds.add("datetime")
            else:
                kinds.add("invalid")
        if "invalid" in kinds:
            raise TypeError("date contiene tipos no temporales.")
        if len(kinds) != 1:
            raise ValueError("date mezcla representaciones temporales.")
        if kinds == {"string"}:
            valid_shape = series.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
            if not valid_shape.all():
                raise ValueError("date debe usar exclusivamente YYYY-MM-DD.")
            parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
        else:
            parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("date contiene fechas invalidas.")
    return parsed.dt.normalize()


def validate_and_reduce_matches(
    points: pd.DataFrame,
    *,
    expected_source_rows: int | None = None,
    expected_matches: int | None = None,
    expected_first_date: pd.Timestamp | None = None,
    expected_last_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Valida las cinco columnas fuente y reduce a una fila por partido."""

    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    if expected_source_rows is not None and len(points) != expected_source_rows:
        raise ValueError(
            f"Filas fuente inesperadas: {len(points)}; esperadas: {expected_source_rows}."
        )
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    _validate_exact_text(work["match_id"], "match_id")
    _validate_exact_text(work["player_1"], "player_1")
    _validate_exact_text(work["player_2"], "player_2")
    _validate_exact_text(work["surface"], "surface")
    work["date"] = _strict_dates(work["date"])

    invalid_surfaces = ~work["surface"].isin(ALLOWED_SURFACES)
    if invalid_surfaces.any():
        values = sorted(work.loc[invalid_surfaces, "surface"].unique().tolist())
        raise ValueError(f"Superficies inesperadas: {values}")
    if work["player_1"].eq(work["player_2"]).any():
        raise ValueError("player_1 y player_2 deben ser distintos.")

    metadata = ["date", "player_1", "player_2", "surface"]
    counts = work.groupby("match_id", sort=False)[metadata].nunique(dropna=False)
    inconsistent = counts.gt(1).any(axis=1)
    if inconsistent.any():
        ids = counts.index[inconsistent].tolist()
        raise ValueError(f"Metadatos inconsistentes dentro de match_id: {ids[:10]}")

    matches = (
        work.drop_duplicates("match_id", keep="first")
        .sort_values(["date", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    if expected_matches is not None and len(matches) != expected_matches:
        raise ValueError(f"Partidos inesperados: {len(matches)}; esperados: {expected_matches}.")
    if matches["match_id"].duplicated().any():
        raise ValueError("match_id no es unico tras la reduccion.")
    if expected_first_date is not None and matches["date"].min() != expected_first_date:
        raise ValueError("La fecha minima no coincide con el contrato fijado.")
    if expected_last_date is not None and matches["date"].max() != expected_last_date:
        raise ValueError("La fecha maxima no coincide con el contrato fijado.")
    return matches


def assign_split(dates: pd.Series) -> pd.Series:
    """Asigna splits solo mediante fechas normalizadas y completas."""

    parsed = _strict_dates(dates)
    outside = parsed.gt(TEST_END)
    if outside.any():
        values = sorted(parsed.loc[outside].dt.date.astype(str).unique().tolist())
        raise ValueError(f"Fechas posteriores al test congelado: {values}")
    result = pd.Series(index=dates.index, dtype="object")
    result.loc[parsed.le(TRAIN_END)] = "train"
    result.loc[parsed.between(VALIDATION_START, VALIDATION_END)] = "validation"
    result.loc[parsed.between(TEST_START, TEST_END)] = "test"
    if result.isna().any():
        raise ValueError("Hay fechas fuera de los intervalos de split.")
    return result


def add_splits(matches: pd.DataFrame) -> pd.DataFrame:
    result = matches.copy(deep=True)
    result["split"] = assign_split(result["date"])
    if result["match_id"].duplicated().any():
        raise ValueError("Un match_id no puede pertenecer a mas de un split.")
    per_date = result.groupby("date", sort=False)["split"].nunique()
    if not per_date.eq(1).all():
        raise ValueError("Todos los partidos de una fecha deben compartir split.")
    return result[MATCH_COLUMNS]


def add_basic_cold_start(matches: pd.DataFrame) -> pd.DataFrame:
    """Anade historia basica: cualquier partido en una fecha anterior."""

    result = matches.copy(deep=True)
    participants = pd.concat(
        [
            result[["player_1", "date"]].rename(columns={"player_1": "player"}),
            result[["player_2", "date"]].rename(columns={"player_2": "player"}),
        ],
        ignore_index=True,
    )
    first_dates = participants.groupby("player", sort=False)["date"].min()
    p1_first = result["player_1"].map(first_dates)
    p2_first = result["player_2"].map(first_dates)
    result["player_1_seen_prior"] = p1_first.lt(result["date"])
    result["player_2_seen_prior"] = p2_first.lt(result["date"])
    seen_count = result[["player_1_seen_prior", "player_2_seen_prior"]].sum(axis=1)
    result["basic_cold_start_state"] = seen_count.map(
        {0: "neither_prior", 1: "one_prior", 2: "both_prior"}
    )
    return result


def _surface_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["surface"].value_counts().reindex(ALLOWED_SURFACES, fill_value=0)
    return {surface.lower(): int(counts.loc[surface]) for surface in ALLOWED_SURFACES}


def _cold_start_counts(frame: pd.DataFrame) -> dict[str, int | float]:
    counts = frame["basic_cold_start_state"].value_counts()
    both = int(counts.get("both_prior", 0))
    one = int(counts.get("one_prior", 0))
    neither = int(counts.get("neither_prior", 0))
    return {
        "both": both,
        "one": one,
        "neither": neither,
        "both_rate": both / len(frame) if len(frame) else 0.0,
    }


def build_by_year(matches: pd.DataFrame) -> pd.DataFrame:
    work = add_basic_cold_start(matches)
    participants = pd.concat(
        [
            work[["player_1", "date"]].rename(columns={"player_1": "player"}),
            work[["player_2", "date"]].rename(columns={"player_2": "player"}),
        ],
        ignore_index=True,
    )
    first_dates = participants.groupby("player", sort=False)["date"].min()
    rows: list[dict[str, Any]] = []
    for year, group in work.groupby(work["date"].dt.year, sort=True):
        players = set(group["player_1"]) | set(group["player_2"])
        year_start = pd.Timestamp(year=int(year), month=1, day=1)
        new_players = sum(first_dates.loc[player] >= year_start for player in players)
        cold = _cold_start_counts(group)
        surfaces = _surface_counts(group)
        rows.append(
            {
                "year": int(year),
                "split": group["split"].iloc[0],
                "is_partial_year": bool(int(year) == 2026),
                "first_date": group["date"].min().date().isoformat(),
                "last_date": group["date"].max().date().isoformat(),
                "matches": int(len(group)),
                "players": int(len(players)),
                "new_players_at_year_start": int(new_players),
                "previously_seen_players_at_year_start": int(len(players) - new_players),
                "both_players_prior_matches": cold["both"],
                "one_player_prior_matches": cold["one"],
                "neither_player_prior_matches": cold["neither"],
                "both_players_prior_rate": cold["both_rate"],
                "hard_matches": surfaces["hard"],
                "clay_matches": surfaces["clay"],
                "grass_matches": surfaces["grass"],
            }
        )
    return pd.DataFrame(rows, columns=BY_YEAR_COLUMNS)


def build_rolling_folds(matches: pd.DataFrame) -> pd.DataFrame:
    work = add_basic_cold_start(matches)
    participants = pd.concat(
        [
            work[["player_1", "date"]].rename(columns={"player_1": "player"}),
            work[["player_2", "date"]].rename(columns={"player_2": "player"}),
        ],
        ignore_index=True,
    )
    first_dates = participants.groupby("player", sort=False)["date"].min()
    rows: list[dict[str, Any]] = []

    def add_fold(
        *,
        protocol: str,
        fold_id: str,
        evaluation_start: pd.Timestamp,
        evaluation_end: pd.Timestamp,
        is_final_test: bool,
        history_update_policy: str,
    ) -> None:
        train = work.loc[work["date"].lt(evaluation_start)]
        evaluation = work.loc[work["date"].between(evaluation_start, evaluation_end)]
        if train.empty or evaluation.empty:
            raise ValueError(f"El fold {fold_id} no contiene train o evaluacion.")
        players = set(evaluation["player_1"]) | set(evaluation["player_2"])
        unseen = sum(first_dates.loc[player] >= evaluation_start for player in players)
        cold = _cold_start_counts(evaluation)
        surfaces = _surface_counts(evaluation)
        rows.append(
            {
                "protocol": protocol,
                "fold_id": fold_id,
                "is_final_test": bool(is_final_test),
                "test_usage_policy": (
                    "sealed_single_final_use" if is_final_test
                    else "method_selection_validation_only"
                ),
                "history_update_policy": history_update_policy,
                "feature_definition_policy": "fixed_before_evaluation",
                "evidence_policy": "fixed_before_evaluation",
                "scoring_parameter_policy": "fixed_before_evaluation",
                "train_cutoff": (evaluation_start - pd.Timedelta(days=1)).date().isoformat(),
                "train_last_observed_date": train["date"].max().date().isoformat(),
                "evaluation_start": evaluation_start.date().isoformat(),
                "evaluation_end": evaluation_end.date().isoformat(),
                "evaluation_first_observed_date": evaluation["date"].min().date().isoformat(),
                "evaluation_last_observed_date": evaluation["date"].max().date().isoformat(),
                "train_matches": int(len(train)),
                "evaluation_matches": int(len(evaluation)),
                "evaluation_players": int(len(players)),
                "unseen_players_at_evaluation_start": int(unseen),
                "seen_players_at_evaluation_start": int(len(players) - unseen),
                "both_players_prior_matches": cold["both"],
                "one_player_prior_matches": cold["one"],
                "neither_player_prior_matches": cold["neither"],
                "both_players_prior_rate": cold["both_rate"],
                "hard_matches": surfaces["hard"],
                "clay_matches": surfaces["clay"],
                "grass_matches": surfaces["grass"],
            }
        )

    for year in VALIDATION_YEARS:
        add_fold(
            protocol="rolling_origin",
            fold_id=f"validation_{year}",
            evaluation_start=pd.Timestamp(year=year, month=1, day=1),
            evaluation_end=pd.Timestamp(year=year, month=12, day=31),
            is_final_test=False,
            history_update_policy="update_after_complete_date",
        )
    add_fold(
        protocol="rolling_origin",
        fold_id="final_test_rolling",
        evaluation_start=TEST_START,
        evaluation_end=TEST_END,
        is_final_test=True,
        history_update_policy="update_after_complete_date",
    )
    add_fold(
        protocol="frozen",
        fold_id="final_test_frozen",
        evaluation_start=TEST_START,
        evaluation_end=TEST_END,
        is_final_test=True,
        history_update_policy="frozen_at_2023-12-31",
    )
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def _split_statistics(matches: pd.DataFrame) -> list[dict[str, Any]]:
    work = add_basic_cold_start(matches)
    participants = pd.concat(
        [
            work[["player_1", "date"]].rename(columns={"player_1": "player"}),
            work[["player_2", "date"]].rename(columns={"player_2": "player"}),
        ],
        ignore_index=True,
    )
    first_dates = participants.groupby("player", sort=False)["date"].min()
    rows = []
    for split in SPLIT_ORDER:
        group = work.loc[work["split"].eq(split)]
        players = set(group["player_1"]) | set(group["player_2"])
        block_start = group["date"].min()
        unseen = sum(first_dates.loc[player] >= block_start for player in players)
        cold = _cold_start_counts(group)
        rows.append(
            {
                "split": split,
                "first_observed_date": block_start.date().isoformat(),
                "last_observed_date": group["date"].max().date().isoformat(),
                "matches": int(len(group)),
                "players": int(len(players)),
                "unseen_players_at_block_start": int(unseen),
                "seen_players_at_block_start": int(len(players) - unseen),
                "basic_cold_start": cold,
                "surface_matches": _surface_counts(group),
            }
        )
    return rows


def build_summary(
    source_rows: int,
    matches: pd.DataFrame,
    by_year: pd.DataFrame,
    folds: pd.DataFrame,
) -> dict[str, Any]:
    split_stats = _split_statistics(matches)
    players = set(matches["player_1"]) | set(matches["player_2"])
    test_matches = int(matches["split"].eq("test").sum())
    validation_matches = int(matches["split"].eq("validation").sum())
    validation_folds = folds.loc[~folds["is_final_test"]].to_dict(orient="records")
    core = {
        "analysis_name": "chronological_validation_protocol",
        "analysis_version": "1.1.0",
        "source_contract": {
            "path": "data/processed/points_enriched.parquet",
            "columns_used": SOURCE_COLUMNS,
            "published_reports_used_as_analytical_source": False,
            "single_parquet_read_per_execution": True,
            "source_point_rows": int(source_rows),
            "matches_after_immediate_reduction": int(len(matches)),
            "players": int(len(players)),
            "date_range": {
                "first": matches["date"].min().date().isoformat(),
                "last": matches["date"].max().date().isoformat(),
            },
        },
        "match_table_contract": {
            "unit": "one_row_per_match_id",
            "published_match_level_table": False,
            "metadata_validated_before_deduplication": True,
            "only_aggregates_are_published": True,
        },
        "date_contract": {
            "calendar_day_normalized": True,
            "accepted_string_format": "YYYY-MM-DD",
            "ambiguous_string_parsing": False,
            "intraday_ordering_used": False,
            "same_date_is_indivisible": True,
            "first_date": "1960-05-29",
            "last_date": "2026-05-21",
        },
        "identity_contract": {
            "fields": ["match_id", "player_1", "player_2"],
            "exact_nonempty_text": True,
            "external_whitespace_allowed": False,
            "player_1_must_differ_from_player_2": True,
            "player_identities_normalized": False,
        },
        "surface_contract": {"allowed_values": list(ALLOWED_SURFACES), "null_allowed": False},
        "split_contract": {
            "assignment_uses_only_normalized_date": True,
            "boundaries": {
                "train": "date <= 2019-12-31",
                "validation": "2020-01-01 <= date <= 2023-12-31",
                "test": "2024-01-01 <= date <= 2026-05-21",
            },
            "unused_or_unassigned_route": False,
            "complete_calendar_dates": True,
        },
        "split_population": split_stats,
        "rolling_origin_contract": {
            "type": "expanding_window",
            "validation_years": list(VALIDATION_YEARS),
            "validation_folds": 4,
            "calendar_day_rule": "all matches on a date are evaluated before that date updates history",
            "history_updates_after_complete_date": True,
            "feature_definition_fixed_during_evaluation": True,
            "evidence_policy_fixed_during_evaluation": True,
            "scoring_and_parameters_fixed_during_evaluation": True,
            "methodological_decisions_limited_to_2020_2023": True,
            "state_may_be_rebuilt_through_2023_only_after_methodology_is_closed": True,
        },
        "rolling_folds": validation_folds,
        "frozen_sensitivity_contract": {
            "freeze_date": "2023-12-31",
            "evaluation_start": "2024-01-01",
            "evaluation_end": "2026-05-21",
            "test_matches": test_matches,
            "features_baselines_and_parameters_remain_fixed": True,
            "test_matches_do_not_update_later_test_predictions": True,
            "same_test_population_as_principal": True,
            "performance_metrics_executed": False,
        },
        "protected_test_contract": {
            "test_status": "sealed",
            "test_used_for_method_selection": False,
            "test_evaluation_runs": 0,
            "test_start": "2024-01-01",
            "test_end": "2026-05-21",
            "test_matches": test_matches,
            "used_once_after_methodology_and_metrics_are_closed": True,
            "excluded_from_selection": [
                "thresholds", "windows", "smoothing", "fallback", "scoring",
                "hyperparameters", "feature_definitions",
            ],
            "descriptive_metadata_only": ["dates", "matches", "players", "surfaces", "basic_cold_start"],
            "performance_results_published": False,
        },
        "cold_start_contract": {
            "definition": "player has any charted match on a strictly earlier calendar date",
            "same_day_does_not_count_as_prior": True,
            "not_direction_specific": True,
            "not_a_sufficient_evidence_measure": True,
            "states": ["neither_prior", "one_prior", "both_prior"],
            "match_state_is_distinct_from_seen_at_fold_start": True,
            "does_not_exclude_matches": True,
        },
        "yearly_contract": {
            "rows": int(len(by_year)),
            "only_observed_years": True,
            "key": ["year"],
            "partial_year_definition": "source fixed at 2026-05-21 before completion of calendar year",
            "nonpartial_does_not_claim_exhaustive_historical_coverage": True,
            "new_players_at_year_start_definition": "players active in year without any match before January 1",
        },
        "folds_table_contract": {
            "rows": int(len(folds)),
            "key": ["protocol", "fold_id"],
            "validation_rows": 4,
            "rolling_test_rows": 1,
            "frozen_test_rows": 1,
            "rolling_and_frozen_test_share_population": True,
            "rows_must_not_be_summed_as_independent_populations": True,
            "only_aggregates_published": True,
        },
        "reconciliations": {
            "source_rows": int(source_rows),
            "unique_matches": int(matches["match_id"].nunique()),
            "split_match_sum": int(sum(row["matches"] for row in split_stats)),
            "annual_match_sum": int(by_year["matches"].sum()),
            "unique_split_per_match": bool(not matches["match_id"].duplicated().any()),
            "single_split_per_calendar_date": bool(
                matches.groupby("date", sort=False)["split"].nunique().eq(1).all()
            ),
            "surface_totals_match": bool(
                by_year[["hard_matches", "clay_matches", "grass_matches"]]
                .sum(axis=1).eq(by_year["matches"]).all()
            ),
            "cold_start_states_exhaustive": bool(
                by_year[[
                    "both_players_prior_matches", "one_player_prior_matches",
                    "neither_player_prior_matches",
                ]].sum(axis=1).eq(by_year["matches"]).all()
            ),
            "validation_fold_union_matches_validation": bool(
                folds.loc[~folds["is_final_test"], "evaluation_matches"].sum()
                == validation_matches
            ),
            "rolling_and_frozen_test_populations_match": bool(
                folds.loc[folds["is_final_test"], "evaluation_matches"].nunique() == 1
            ),
            "test_remains_sealed": True,
            "no_features_outcomes_models_or_recommendations": True,
        },
        "atomic_publication_contract": {
            "all_payloads_validated_before_replace": True,
            "rollback_on_caught_replace_failure": True,
            "physical_transaction_across_power_loss": False,
        },
        "methodological_limits": [
            "Cold start only records prior match presence; it does not measure direction-specific evidence.",
            "The 2026 test segment is partial through 2026-05-21 and must remain frozen to this data release.",
            "No threshold, smoothing, fallback, score, model or recommendation is selected here.",
        ],
        "next_allowed_step": {
            "allowed": [
                "calculate combined evidence policies in train and validation",
                "select candidates with validation folds 2020-2023",
            ],
            "test_remains_sealed": True,
            "scoring_models_or_test_evaluation_authorized": False,
        },
    }
    core["publication_fingerprint"] = _publication_fingerprint(
        source_rows, matches, by_year, folds, core
    )
    return core


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    work = frame.copy(deep=True)
    for column in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[column]):
            work[column] = work[column].dt.strftime("%Y-%m-%d")
    buffer = io.StringIO(newline="")
    work.to_csv(
        buffer, index=False, lineterminator="\n", encoding="utf-8",
        float_format="%.15g", na_rep="",
    )
    return buffer.getvalue().encode("utf-8")


def _publication_fingerprint(
    source_rows: int,
    matches: pd.DataFrame,
    by_year: pd.DataFrame,
    folds: pd.DataFrame,
    summary_without_fingerprint: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(source_rows)).encode("ascii"))
    digest.update(_frame_bytes(matches))
    digest.update(_frame_bytes(by_year))
    digest.update(_frame_bytes(folds))
    digest.update(json.dumps(
        summary_without_fingerprint, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))
    return digest.hexdigest().upper()


def validate_result(result: ChronologicalValidationResult) -> None:
    matches = result.matches
    if matches.columns.tolist() != MATCH_COLUMNS:
        raise ValueError("El esquema de partidos no coincide con el contrato.")
    if result.by_year.columns.tolist() != BY_YEAR_COLUMNS:
        raise ValueError("El esquema anual no coincide con el contrato.")
    if result.folds.columns.tolist() != FOLD_COLUMNS:
        raise ValueError("El esquema de folds no coincide con el contrato.")
    for column in ("match_id", "player_1", "player_2", "surface"):
        _validate_exact_text(matches[column], column)
    normalized_dates = _strict_dates(matches["date"])
    if not matches["date"].eq(normalized_dates).all():
        raise ValueError("Las fechas de partido no estan normalizadas.")
    if matches["player_1"].eq(matches["player_2"]).any():
        raise ValueError("Los participantes de partido deben ser distintos.")
    if not matches["surface"].isin(ALLOWED_SURFACES).all():
        raise ValueError("La tabla de partidos contiene superficies inesperadas.")
    expected_splits = assign_split(matches["date"])
    if not matches["split"].equals(expected_splits):
        raise ValueError("La asignacion de split no coincide con las fechas.")
    if len(matches) != matches["match_id"].nunique():
        raise ValueError("La poblacion de partidos no reconcilia.")
    sorted_matches = matches.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    pdt.assert_frame_equal(matches.reset_index(drop=True), sorted_matches, check_exact=True)
    if not matches.groupby("date", sort=False)["split"].nunique().eq(1).all():
        raise ValueError("Una fecha civil aparece en mas de un split.")

    expected_year = build_by_year(matches)
    expected_folds = build_rolling_folds(matches)
    pdt.assert_frame_equal(result.by_year, expected_year, check_exact=True)
    pdt.assert_frame_equal(result.folds, expected_folds, check_exact=True)
    if int(result.by_year["matches"].sum()) != len(matches):
        raise ValueError("Los partidos anuales no reconcilian.")
    if result.by_year["year"].duplicated().any():
        raise ValueError("La clave anual no es unica.")
    if not result.by_year.loc[result.by_year["year"].eq(2026), "is_partial_year"].eq(True).all():
        raise ValueError("2026 debe estar marcado como parcial.")
    if result.by_year.loc[result.by_year["year"].ne(2026), "is_partial_year"].any():
        raise ValueError("Solo 2026 puede estar marcado como parcial.")
    for row in result.by_year.itertuples(index=False):
        if not str(row.first_date).startswith(str(row.year)) or not str(row.last_date).startswith(str(row.year)):
            raise ValueError("Las fechas anuales no pertenecen al anio publicado.")
        if row.first_date > row.last_date:
            raise ValueError("first_date anual supera last_date.")

    if len(result.folds) != 6 or result.folds.duplicated(["protocol", "fold_id"]).any():
        raise ValueError("La tabla debe contener seis filas contractuales unicas.")
    validation = result.folds.loc[~result.folds["is_final_test"]]
    if len(validation) != 4 or not validation["train_matches"].is_monotonic_increasing:
        raise ValueError("El train rolling-origin no es expansivo.")
    for previous, following in zip(validation.itertuples(index=False), validation.iloc[1:].itertuples(index=False)):
        if following.train_matches != previous.train_matches + previous.evaluation_matches:
            raise ValueError("El train siguiente no incorpora exactamente la evaluacion anterior.")
    if int(validation["evaluation_matches"].sum()) != int(matches["split"].eq("validation").sum()):
        raise ValueError("La union de folds no coincide con validation.")
    final_rows = result.folds.loc[result.folds["is_final_test"]]
    if len(final_rows) != 2 or final_rows["evaluation_matches"].nunique() != 1:
        raise ValueError("Rolling y frozen no comparten la misma poblacion de test.")
    if not final_rows["evaluation_matches"].eq(int(matches["split"].eq("test").sum())).all():
        raise ValueError("Las filas finales no reconcilian con test.")

    expected_summary = build_summary(
        result.source_rows, matches, expected_year, expected_folds
    )
    if result.summary != expected_summary:
        raise ValueError("El resumen no coincide con la derivacion canonica completa.")
    if not all(
        value is True
        for value in result.summary["reconciliations"].values()
        if isinstance(value, bool)
    ):
        raise ValueError("Alguna reconciliacion del resumen ha fallado.")


def build_analysis(points: pd.DataFrame) -> ChronologicalValidationResult:
    source_rows = len(points)
    matches = validate_and_reduce_matches(
        points,
        expected_source_rows=EXPECTED_SOURCE_ROWS,
        expected_matches=EXPECTED_MATCHES,
        expected_first_date=EXPECTED_FIRST_DATE,
        expected_last_date=EXPECTED_LAST_DATE,
    )
    matches = add_splits(matches)
    by_year = build_by_year(matches)
    folds = build_rolling_folds(matches)
    summary = build_summary(source_rows, matches, by_year, folds)
    result = ChronologicalValidationResult(source_rows, matches, by_year, folds, summary)
    validate_result(result)
    if matches["split"].value_counts().to_dict() != EXPECTED_SPLIT_MATCHES:
        raise ValueError("Las poblaciones reales de split no coinciden con el contrato fijado.")
    if len(set(matches["player_1"]) | set(matches["player_2"])) != 1_002:
        raise ValueError("La poblacion real de jugadores no coincide con 1.002.")
    return result


def read_and_build(path: Path = POINTS_PATH) -> ChronologicalValidationResult:
    points = pd.read_parquet(path, columns=SOURCE_COLUMNS)
    return build_analysis(points)


def serialize_artifacts(result: ChronologicalValidationResult) -> tuple[bytes, bytes, bytes]:
    validate_result(result)
    summary = (
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return summary, _frame_bytes(result.by_year), _frame_bytes(result.folds)


def validate_serialized_artifacts(
    result: ChronologicalValidationResult,
    summary_payload: bytes,
    by_year_payload: bytes,
    folds_payload: bytes,
) -> None:
    loaded_summary = json.loads(summary_payload.decode("utf-8"))
    if loaded_summary != result.summary:
        raise ValueError("El JSON serializado no reconcilia.")
    loaded_year = pd.read_csv(io.BytesIO(by_year_payload))
    loaded_folds = pd.read_csv(io.BytesIO(folds_payload))
    if loaded_year.columns.tolist() != BY_YEAR_COLUMNS or loaded_folds.columns.tolist() != FOLD_COLUMNS:
        raise ValueError("El esquema CSV serializado no coincide.")
    if any(column.startswith("Unnamed") for column in [*loaded_year.columns, *loaded_folds.columns]):
        raise ValueError("Los CSV contienen un indice accidental.")
    expected_year = result.by_year.copy()
    expected_folds = result.folds.copy()
    for column in ("first_date", "last_date"):
        expected_year[column] = expected_year[column].astype(str)
    for column in (
        "train_cutoff", "train_last_observed_date", "evaluation_start", "evaluation_end",
        "evaluation_first_observed_date", "evaluation_last_observed_date",
    ):
        expected_folds[column] = expected_folds[column].astype(str)
    pdt.assert_frame_equal(loaded_year, expected_year, check_exact=False, rtol=1e-14, atol=1e-14)
    pdt.assert_frame_equal(loaded_folds, expected_folds, check_exact=False, rtol=1e-14, atol=1e-14)


def _stage_payload(path: Path, payload: bytes) -> Path:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    result: ChronologicalValidationResult,
    summary_path: Path = SUMMARY_PATH,
    by_year_path: Path = BY_YEAR_PATH,
    folds_path: Path = FOLDS_PATH,
) -> None:
    paths = (summary_path, by_year_path, folds_path)
    if len({path.resolve() for path in paths}) != 3:
        raise ValueError("Las rutas de artefactos deben ser distintas.")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    payloads = serialize_artifacts(result)
    validate_serialized_artifacts(result, *payloads)
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


def main() -> None:
    result = read_and_build()
    write_artifacts(result)
    expected_payloads = serialize_artifacts(result)
    written_payloads = (
        SUMMARY_PATH.read_bytes(), BY_YEAR_PATH.read_bytes(), FOLDS_PATH.read_bytes()
    )
    if written_payloads != expected_payloads:
        raise ValueError("Los artefactos escritos no son identicos a la serializacion canonica.")
    stats = {row["split"]: row["matches"] for row in result.summary["split_population"]}
    print(f"Partidos: {len(result.matches)} | splits: {stats} | folds: {len(result.folds)}")


if __name__ == "__main__":
    main()
