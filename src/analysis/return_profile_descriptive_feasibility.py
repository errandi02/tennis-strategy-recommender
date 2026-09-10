"""Infraestructura descriptiva sellada de P09.

El modulo prepara una unica lectura futura del Parquet. Todas las funciones
analiticas aceptan DataFrames en memoria y no realizan E/S al importarse.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import io
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Final, Iterator, Mapping, Sequence

import pandas as pd
import pandas.testing as pdt

from src.analysis.return_profile_feasibility import (
    CLASSIFICATION_CONTRACT_VERSION,
    DOCUMENTED_PROFILE_IDS,
    DOCUMENTED_SHOT_TYPES,
    LATERAL_DIRECTIONS,
    PATTERN_ID,
    RETURN_DEPTHS,
    SHOT_TYPE_FAMILIES,
    ReturnProfileClassification,
    ReturnProfileReason,
    ReturnProfileState,
    parse_and_classify_initial_return_profile,
)


ANALYSIS_NAME: Final = "return_profile_descriptive_feasibility"
ANALYSIS_VERSION: Final = "1.0.0"
UPSTREAM_COMMIT: Final = "f940827"
ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "return_profile_descriptive_feasibility_summary.json"
BY_STATE_PATH = TABLES_DIR / "return_profile_descriptive_feasibility_by_state.csv"
BY_PROFILE_PATH = TABLES_DIR / "return_profile_descriptive_feasibility_by_profile.csv"
BY_GROUP_PATH = TABLES_DIR / "return_profile_descriptive_feasibility_by_group.csv"
OUTCOMES_PATH = TABLES_DIR / "return_profile_descriptive_feasibility_outcomes.csv"

EXTRACTOR_PATH = ROOT / "src" / "analysis" / "return_profile_feasibility.py"
EXTRACTOR_SHA256: Final = "CE27052437A747D4CADEED770C44D21180672ECF4164CF93CE9671ABB9FE8B31"
CHRONOLOGICAL_SUMMARY_PATH = REPORTS_DIR / "chronological_validation_summary.json"
CHRONOLOGICAL_SUMMARY_SHA256: Final = "90890B721CC93C82A5DE5F2EB6E3F3D4E5D4D3E83E6C9E678C5569139301799F"

CUTOFF = pd.Timestamp("2023-12-31")
SURFACES: Final = ("Hard", "Clay", "Grass")
PERIODS: Final = ("to_2009", "2010s", "2020s")
FOLDS: Final = ("pre_validation", "2020", "2021", "2022", "2023")
SERVE_NUMBERS: Final = (1, 2)
STATE_ORDER: Final = tuple(state.value for state in ReturnProfileState)
TYPE_ORDER: Final = tuple(DOCUMENTED_SHOT_TYPES)
DIRECTION_ORDER: Final = ("1", "2", "3")
DEPTH_ORDER: Final = ("7", "8", "9")
PROFILE_ORDER: Final = tuple(
    f"{shot}|{direction}|{depth}"
    for shot in TYPE_ORDER
    for direction in DIRECTION_ORDER
    for depth in DEPTH_ORDER
)
SECOND_SERVE_CONTEXTS: Final = (
    "documented_first_service_fault",
    "second_serve_without_documented_first_fault",
)
SOURCE_COLUMNS: Final = (
    "match_id",
    "point_number",
    "date",
    "surface",
    "server",
    "point_winner",
    "player_1",
    "player_2",
    "first_serve",
    "second_serve",
)
EXPECTED_REAL: Final = {
    "source_rows_read": 1_280_408,
    "source_matches": 7_524,
    "source_players": 1_002,
    "development_point_rows": 1_035_760,
    "development_matches": 5_993,
    "development_servers": 870,
    "excluded_test_matches": 1_531,
    "first_attempts": 1_035_760,
    "second_attempts": 391_103,
    "attempts_total": 1_426_863,
}
TEST_ZERO_FIELDS: Final = (
    "test_target_rows_parsed",
    "test_attempts_constructed",
    "test_profiles_classified",
    "test_outcomes_computed",
    "test_rows_evaluated",
    "test_matches_evaluated",
    "test_scores_computed",
    "test_evaluation_runs",
    "test_recommendations_generated",
)
FAILURE_STAGES: Final = (
    "upstream_validation",
    "read_source_points",
    "source_validation",
    "metadata_and_test_seal",
    "attempt_construction_and_classification",
    "aggregation",
    "outcome_analysis",
    "reconciliation",
    "result_validation",
    "serialization",
    "publication_verification",
)
NOT_AVAILABLE_REASON_CODES: Final = ("execution_failed",)
WILSON_Z: Final = 1.959963984540054
WILSON_BOUNDARY_TOLERANCE: Final = 1e-15
METHODOLOGICAL_LIMITS: Final = (
    "Descriptive observational coverage and outcomes; no causal interpretation.",
    "A complete documented profile is not an optimal strategy or recommendation.",
    "No thresholds, smoothing, significance tests, or automatic rankings are selected.",
    "Sequential multi-file replacement cannot be atomic across an abrupt process or system shutdown.",
)

BY_STATE_COLUMNS: Final = (
    "aggregation_level",
    "group_type",
    "group_value",
    "serve_number",
    "surface",
    "period",
    "validation_fold",
    "state",
    "attempts",
    "state_attempts",
    "state_share",
)
BY_PROFILE_COLUMNS: Final = (
    "profile_id",
    "return_shot_type_code",
    "return_shot_type_description",
    "return_shot_type_family",
    "return_lateral_direction_code",
    "return_lateral_direction_description",
    "return_depth_code",
    "return_depth_description",
    "complete_profile_attempts",
    "share_of_complete_profiles",
    "share_of_all_attempts",
)
BY_GROUP_COLUMNS: Final = (
    "aggregation_level",
    "group_type",
    "group_value",
    "serve_number",
    "surface",
    "period",
    "validation_fold",
    "attempts",
    "complete_profiles",
    "unknown_profiles",
    "not_documented_profiles",
    "unknown_initial_returns",
    "censored_attempts",
    "complete_profile_coverage",
    "abstention_share",
)
OUTCOME_COLUMNS: Final = (
    "aggregation_level",
    "group_type",
    "group_value",
    "serve_number",
    "surface",
    "period",
    "validation_fold",
    "profile_id",
    "return_shot_type_code",
    "return_lateral_direction_code",
    "return_depth_code",
    "trials",
    "returner_wins",
    "server_wins",
    "returner_win_rate",
    "wilson_95_lower",
    "wilson_95_upper",
)
ATTEMPT_COLUMNS: Final = (
    "match_id",
    "point_number",
    "serve_number",
    "date",
    "surface",
    "period",
    "validation_fold",
    "server_player",
    "returner_player",
    "previous_attempt_was_fault",
    "second_serve_context",
    "state",
    "reason_codes",
    "eligible",
    "profile_id",
    "return_shot_type_code",
    "return_shot_type_description",
    "return_shot_type_family",
    "return_lateral_direction_code",
    "return_lateral_direction_description",
    "return_depth_code",
    "return_depth_description",
    "unknown_components",
    "not_documented_components",
    "returner_won_point",
)


class FeasibilityContractError(ValueError):
    """Violacion cerrada del contrato descriptivo P09."""


class _RunFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception, partial: Mapping[str, Any] | None = None) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause
        self.partial = dict(partial or {})


@dataclass(frozen=True)
class FeasibilityResult:
    summary: dict[str, Any]
    by_state: pd.DataFrame
    by_profile: pd.DataFrame
    by_group: pd.DataFrame
    outcomes: pd.DataFrame
    attempts: pd.DataFrame


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def validate_upstream_contracts() -> dict[str, str]:
    if UPSTREAM_COMMIT != "f940827":
        raise FeasibilityContractError("Commit upstream P09 incompatible.")
    try:
        extractor_payload = EXTRACTOR_PATH.read_bytes()
        chronology_payload = CHRONOLOGICAL_SUMMARY_PATH.read_bytes()
    except OSError as exc:
        raise FeasibilityContractError("Falta un contrato upstream congelado.") from exc
    if _sha256_bytes(extractor_payload) != EXTRACTOR_SHA256:
        raise FeasibilityContractError("El extractor P09 no coincide con el hash congelado.")
    if _sha256_bytes(chronology_payload) != CHRONOLOGICAL_SUMMARY_SHA256:
        raise FeasibilityContractError("El contrato cronologico no coincide con el hash congelado.")
    try:
        chronology = json.loads(chronology_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeasibilityContractError("Contrato cronologico ilegible.") from exc
    seal = chronology.get("protected_test_contract", {})
    if (
        seal.get("test_status") != "sealed"
        or seal.get("test_matches") != 1_531
        or seal.get("test_used_for_method_selection") is not False
    ):
        raise FeasibilityContractError("Sellado cronologico upstream incompatible.")
    return {
        "extractor_commit": UPSTREAM_COMMIT,
        "extractor_sha256": EXTRACTOR_SHA256,
        "chronological_summary_sha256": CHRONOLOGICAL_SUMMARY_SHA256,
    }


def read_source_points() -> pd.DataFrame:
    """Unica lectura productiva futura, limitada a columnas contractuales."""
    return pd.read_parquet(POINTS_PATH, columns=list(SOURCE_COLUMNS))


def _strict_binary_index(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value not in (1, 2):
        raise FeasibilityContractError(f"{field} debe ser exactamente el entero 1 o 2.")
    return int(value)


def _strict_point_number(value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise FeasibilityContractError("point_number debe ser entero positivo real.")
    return int(value)


def _strict_text(series: pd.Series, field: str) -> None:
    invalid = series.map(
        lambda value: type(value) is not str or not value or value != value.strip()
    )
    if bool(invalid.any()):
        raise FeasibilityContractError(
            f"{field} debe ser texto no vacio y sin espacios externos."
        )


def _dates(series: pd.Series) -> pd.Series:
    if bool(series.isna().any()):
        raise FeasibilityContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif bool(series.map(lambda value: type(value) is str).all()):
        if not bool(series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()):
            raise FeasibilityContractError("date debe usar YYYY-MM-DD sin ambiguedad.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif bool(
        series.map(
            lambda value: isinstance(value, (pd.Timestamp, datetime, date))
            and not isinstance(value, bool)
        ).all()
    ):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        raise FeasibilityContractError("date contiene tipos ambiguos.")
    if bool(parsed.isna().any()) or getattr(parsed.dt, "tz", None) is not None:
        raise FeasibilityContractError("date invalida o con zona horaria.")
    if bool(parsed.ne(parsed.dt.normalize()).any()):
        raise FeasibilityContractError("date debe estar normalizada al dia civil.")
    return parsed.dt.normalize()


def _sequence_presence(value: object, field: str) -> str:
    if value is None or value is pd.NA or (
        not isinstance(value, str) and pd.isna(value)
    ):
        return "null"
    if type(value) is not str:
        raise FeasibilityContractError(f"{field} debe ser texto o nulo.")
    if value == "":
        return "empty"
    if value.isspace():
        return "whitespace_only"
    return "substantive"


def derive_period(day: pd.Timestamp) -> str:
    if day.year <= 2009:
        return "to_2009"
    if day.year <= 2019:
        return "2010s"
    return "2020s"


def validation_fold(day: pd.Timestamp) -> str:
    if day.year <= 2019:
        return "pre_validation"
    fold = str(day.year)
    if fold not in FOLDS:
        raise FeasibilityContractError("Fecha de desarrollo fuera de folds autorizados.")
    return fold


def validate_source_points(
    points: pd.DataFrame, *, expected_rows: int | None = None
) -> pd.DataFrame:
    if not isinstance(points, pd.DataFrame):
        raise TypeError("points debe ser DataFrame.")
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing:
        raise FeasibilityContractError(f"Faltan columnas requeridas: {missing}")
    if expected_rows is not None and len(points) != expected_rows:
        raise FeasibilityContractError(
            f"source_rows_read inesperado: {len(points)}"
        )
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"):
        _strict_text(work[field], field)
    work["point_number"] = work.point_number.map(_strict_point_number)
    if bool(work.duplicated(["match_id", "point_number"]).any()):
        raise FeasibilityContractError("(match_id, point_number) debe ser clave unica.")
    work["date"] = _dates(work.date)
    unexpected = sorted(set(work.loc[~work.surface.isin(SURFACES), "surface"]))
    if unexpected:
        raise FeasibilityContractError(f"Superficies inesperadas: {unexpected}")
    if bool(work.player_1.eq(work.player_2).any()):
        raise FeasibilityContractError("player_1 y player_2 deben ser distintos.")
    for field in ("server", "point_winner"):
        work[field] = work[field].map(
            lambda value, name=field: _strict_binary_index(value, name)
        )
    inconsistent = (
        work.groupby("match_id", sort=False)[
            ["date", "surface", "player_1", "player_2"]
        ]
        .nunique(dropna=False)
        .gt(1)
    )
    if bool(inconsistent.any().any()):
        raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    if bool(
        work.first_serve.map(
            lambda value: _sequence_presence(value, "first_serve") != "substantive"
        ).any()
    ):
        raise FeasibilityContractError("first_serve debe ser sustantivo.")
    work.second_serve.map(lambda value: _sequence_presence(value, "second_serve"))
    return work.sort_values(
        ["date", "match_id", "point_number"], kind="stable"
    ).reset_index(drop=True)


def split_development_before_parsing(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    development = source.loc[source.date.le(CUTOFF)].copy()
    sealed = source.loc[source.date.gt(CUTOFF)].copy()
    if bool(development.date.gt(CUTOFF).any()):
        raise FeasibilityContractError("El test no quedo sellado antes del extractor.")
    server_names = pd.Series(
        [
            row.player_1 if row.server == 1 else row.player_2
            for row in development.itertuples(index=False)
        ],
        dtype="object",
    )
    counts = {
        "source_rows_read": int(len(source)),
        "source_matches": int(source.match_id.nunique()),
        "source_players": int(len(set(source.player_1) | set(source.player_2))),
        "development_point_rows": int(len(development)),
        "development_matches": int(development.match_id.nunique()),
        "development_servers": int(server_names.nunique()),
        "excluded_test_matches": int(sealed.match_id.nunique()),
    }
    return development.reset_index(drop=True), counts


def _attempt_row(
    row: Any,
    serve_number: int,
    previous_fault: bool,
    classified: ReturnProfileClassification,
) -> dict[str, Any]:
    return {
        "match_id": row.match_id,
        "point_number": row.point_number,
        "serve_number": serve_number,
        "date": row.date,
        "surface": row.surface,
        "period": derive_period(row.date),
        "validation_fold": validation_fold(row.date),
        "server_player": row.player_1 if row.server == 1 else row.player_2,
        "returner_player": row.player_2 if row.server == 1 else row.player_1,
        "previous_attempt_was_fault": previous_fault,
        "second_serve_context": (
            "not_applicable_first_serve"
            if serve_number == 1
            else "documented_first_service_fault"
            if previous_fault
            else "second_serve_without_documented_first_fault"
        ),
        "state": classified.classification_state.value,
        "reason_codes": "|".join(reason.value for reason in classified.reason_codes),
        "eligible": classified.eligible_for_profile_comparison,
        "profile_id": classified.profile_id,
        "return_shot_type_code": classified.return_shot_type_code,
        "return_shot_type_description": classified.return_shot_type_description,
        "return_shot_type_family": classified.return_shot_type_family,
        "return_lateral_direction_code": classified.lateral_direction_code,
        "return_lateral_direction_description": classified.lateral_direction_description,
        "return_depth_code": classified.return_depth_code,
        "return_depth_description": classified.return_depth_description,
        "unknown_components": classified.unknown_components,
        "not_documented_components": classified.not_documented_components,
        "returner_won_point": bool(row.point_winner != row.server),
    }


def construct_attempts(
    development: pd.DataFrame,
    *,
    extractor: Callable[..., ReturnProfileClassification] = parse_and_classify_initial_return_profile,
) -> pd.DataFrame:
    cache: dict[tuple[str, int, bool], ReturnProfileClassification] = {}

    def classify(text: str, number: int, previous_fault: bool) -> ReturnProfileClassification:
        key = (text, number, previous_fault)
        if key not in cache:
            classified = extractor(
                text,
                number,
                previous_attempt_was_fault=previous_fault,
            )
            if not isinstance(classified, ReturnProfileClassification):
                raise FeasibilityContractError("El extractor P09 devolvio un tipo invalido.")
            cache[key] = classified
        return cache[key]

    rows: list[dict[str, Any]] = []
    for row in development.itertuples(index=False):
        if _sequence_presence(row.first_serve, "first_serve") != "substantive":
            raise FeasibilityContractError("first_serve debe ser sustantivo.")
        first = classify(row.first_serve, 1, False)
        rows.append(_attempt_row(row, 1, False, first))
        if _sequence_presence(row.second_serve, "second_serve") == "substantive":
            previous_fault = first.terminal_serve_outcome == "service_fault"
            second = classify(row.second_serve, 2, previous_fault)
            rows.append(_attempt_row(row, 2, previous_fault, second))
    attempts = pd.DataFrame(rows, columns=ATTEMPT_COLUMNS)
    if attempts.empty:
        raise FeasibilityContractError("No se construyeron intentos de desarrollo.")
    if bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()):
        raise FeasibilityContractError("Clave de intento duplicada.")
    attempts = attempts.sort_values(
        ["date", "match_id", "point_number", "serve_number"], kind="stable"
    ).reset_index(drop=True)
    attempts.attrs["cache_entries"] = len(cache)
    return attempts


def _group_coordinates(
    group_type: str,
    value: object,
) -> tuple[str, int, str, str, str]:
    if group_type == "total":
        return "ALL", 0, "ALL", "ALL", "ALL"
    if group_type == "serve_number":
        return str(value), int(value), "ALL", "ALL", "ALL"
    if group_type == "surface":
        return str(value), 0, str(value), "ALL", "ALL"
    if group_type == "period":
        return str(value), 0, "ALL", str(value), "ALL"
    if group_type == "validation_fold":
        return str(value), 0, "ALL", "ALL", str(value)
    raise FeasibilityContractError("group_type no autorizado.")


def _group_specs(
    attempts: pd.DataFrame,
) -> Iterator[tuple[str, object, pd.DataFrame]]:
    """Itera grupos sin retener simultáneamente copias de todas las particiones."""
    yield "total", "ALL", attempts
    for number in SERVE_NUMBERS:
        yield "serve_number", number, attempts.loc[attempts.serve_number.eq(number)]
    for surface in SURFACES:
        yield "surface", surface, attempts.loc[attempts.surface.eq(surface)]
    for period in PERIODS:
        yield "period", period, attempts.loc[attempts.period.eq(period)]
    for fold in FOLDS:
        yield (
            "validation_fold",
            fold,
            attempts.loc[attempts.validation_fold.eq(fold)],
        )


def _profile_metadata(profile_id: str) -> dict[str, str]:
    shot, direction, depth = profile_id.split("|")
    return {
        "profile_id": profile_id,
        "return_shot_type_code": shot,
        "return_shot_type_description": DOCUMENTED_SHOT_TYPES[shot],
        "return_shot_type_family": SHOT_TYPE_FAMILIES[shot],
        "return_lateral_direction_code": direction,
        "return_lateral_direction_description": LATERAL_DIRECTIONS[direction],
        "return_depth_code": depth,
        "return_depth_description": RETURN_DEPTHS[depth],
    }


def build_by_state(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_type, group_value, part in _group_specs(attempts):
        value, number, surface, period, fold = _group_coordinates(
            group_type, group_value
        )
        denominator = int(len(part))
        for state in STATE_ORDER:
            count = int(part.state.eq(state).sum())
            rows.append(
                {
                    "aggregation_level": "state",
                    "group_type": group_type,
                    "group_value": value,
                    "serve_number": number,
                    "surface": surface,
                    "period": period,
                    "validation_fold": fold,
                    "state": state,
                    "attempts": denominator,
                    "state_attempts": count,
                    "state_share": None if denominator == 0 else count / denominator,
                }
            )
    return pd.DataFrame(rows, columns=BY_STATE_COLUMNS)


def build_by_profile(attempts: pd.DataFrame) -> pd.DataFrame:
    eligible = attempts.loc[attempts.state.eq(ReturnProfileState.OBSERVED.value)]
    complete_total = int(len(eligible))
    all_total = int(len(attempts))
    counts = eligible.groupby("profile_id", sort=False).size().to_dict()
    rows = []
    for profile_id in PROFILE_ORDER:
        count = int(counts.get(profile_id, 0))
        rows.append(
            {
                **_profile_metadata(profile_id),
                "complete_profile_attempts": count,
                "share_of_complete_profiles": (
                    None if complete_total == 0 else count / complete_total
                ),
                "share_of_all_attempts": None if all_total == 0 else count / all_total,
            }
        )
    return pd.DataFrame(rows, columns=BY_PROFILE_COLUMNS)


def build_by_group(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_fields = {
        ReturnProfileState.OBSERVED.value: "complete_profiles",
        ReturnProfileState.UNKNOWN.value: "unknown_profiles",
        ReturnProfileState.NOT_DOCUMENTED.value: "not_documented_profiles",
        ReturnProfileState.UNKNOWN_INITIAL.value: "unknown_initial_returns",
        ReturnProfileState.CENSORED.value: "censored_attempts",
    }
    for group_type, group_value, part in _group_specs(attempts):
        value, number, surface, period, fold = _group_coordinates(
            group_type, group_value
        )
        total = int(len(part))
        counts = {
            field: int(part.state.eq(state).sum())
            for state, field in state_fields.items()
        }
        rows.append(
            {
                "aggregation_level": "coverage",
                "group_type": group_type,
                "group_value": value,
                "serve_number": number,
                "surface": surface,
                "period": period,
                "validation_fold": fold,
                "attempts": total,
                **counts,
                "complete_profile_coverage": (
                    None if total == 0 else counts["complete_profiles"] / total
                ),
                "abstention_share": (
                    None if total == 0 else (total - counts["complete_profiles"]) / total
                ),
            }
        )
    return pd.DataFrame(rows, columns=BY_GROUP_COLUMNS)


def _wilson_failure(
    successes: int,
    trials: int,
    rate: float,
    lower: float,
    upper: float,
    context: Mapping[str, Any] | None,
) -> FeasibilityContractError:
    allowed_context = {
        "aggregation_level",
        "group_type",
        "group_value",
        "serve_number",
        "surface",
        "period",
        "validation_fold",
        "profile_id",
    }
    safe_context = {
        key: value for key, value in dict(context or {}).items() if key in allowed_context
    }
    details = {
        "successes": successes,
        "trials": trials,
        "rate": rate,
        "lower": lower,
        "upper": upper,
        **safe_context,
    }
    return FeasibilityContractError(
        "Wilson fuera de rango: "
        + ", ".join(f"{key}={value!r}" for key, value in details.items())
    )


def wilson(
    successes: int,
    trials: int,
    z: float = WILSON_Z,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    if (
        type(successes) is not int
        or type(trials) is not int
        or successes < 0
        or trials < 0
        or successes > trials
    ):
        raise FeasibilityContractError(
            "Wilson requiere enteros reales con 0 <= successes <= trials."
        )
    if trials == 0:
        return None, None
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(
            rate * (1 - rate) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    lower, upper = centre - spread, centre + spread
    if successes == 0:
        lower = 0.0
    if successes == trials:
        upper = 1.0
    if -WILSON_BOUNDARY_TOLERANCE <= lower <= WILSON_BOUNDARY_TOLERANCE:
        lower = 0.0
    if 1 - WILSON_BOUNDARY_TOLERANCE <= upper <= 1 + WILSON_BOUNDARY_TOLERANCE:
        upper = 1.0
    if lower > rate and lower - rate <= WILSON_BOUNDARY_TOLERANCE:
        lower = rate
    if upper < rate and rate - upper <= WILSON_BOUNDARY_TOLERANCE:
        upper = rate
    valid = all(math.isfinite(value) for value in (rate, lower, upper))
    valid = valid and 0 <= lower <= upper <= 1
    valid = valid and (
        lower - WILSON_BOUNDARY_TOLERANCE
        <= rate
        <= upper + WILSON_BOUNDARY_TOLERANCE
    )
    if not valid:
        raise _wilson_failure(successes, trials, rate, lower, upper, context)
    return lower, upper


def build_outcomes(attempts: pd.DataFrame) -> pd.DataFrame:
    eligible = attempts.loc[attempts.state.eq(ReturnProfileState.OBSERVED.value)]
    rows: list[dict[str, Any]] = []
    for group_type, group_value, part in _group_specs(eligible):
        value, number, surface, period, fold = _group_coordinates(
            group_type, group_value
        )
        aggregates = (
            part.groupby("profile_id", sort=False, observed=True)
            .returner_won_point.agg(["size", "sum"])
        )
        grouped = {
            str(row.profile_id): (int(row.size), int(row.sum))
            for row in aggregates.reset_index().itertuples(index=False)
        }
        for profile_id in PROFILE_ORDER:
            trials, returner_wins = grouped.get(profile_id, (0, 0))
            lower, upper = wilson(
                returner_wins,
                trials,
                context={
                    "aggregation_level": "profile_outcome",
                    "group_type": group_type,
                    "group_value": value,
                    "serve_number": number,
                    "surface": surface,
                    "period": period,
                    "validation_fold": fold,
                    "profile_id": profile_id,
                },
            )
            shot, direction, depth = profile_id.split("|")
            rows.append(
                {
                    "aggregation_level": "profile_outcome",
                    "group_type": group_type,
                    "group_value": value,
                    "serve_number": number,
                    "surface": surface,
                    "period": period,
                    "validation_fold": fold,
                    "profile_id": profile_id,
                    "return_shot_type_code": shot,
                    "return_lateral_direction_code": direction,
                    "return_depth_code": depth,
                    "trials": trials,
                    "returner_wins": returner_wins,
                    "server_wins": trials - returner_wins,
                    "returner_win_rate": (
                        None if trials == 0 else returner_wins / trials
                    ),
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                }
            )
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    raise FeasibilityContractError(f"{field} no es entero contractual.")


def _as_optional_float(value: Any, field: str) -> float | None:
    if value is None or value == "" or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FeasibilityContractError(f"{field} no es numerico.") from exc
    if not math.isfinite(number):
        raise FeasibilityContractError(f"{field} no es finito.")
    return number


def _close(actual: Any, expected: float, field: str) -> None:
    value = _as_optional_float(actual, field)
    if value is None or not math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise FeasibilityContractError(f"{field} no reconcilia.")


def _expected_group_keys() -> list[tuple[str, str, int, str, str, str]]:
    return (
        [("total", "ALL", 0, "ALL", "ALL", "ALL")]
        + [("serve_number", str(value), value, "ALL", "ALL", "ALL") for value in SERVE_NUMBERS]
        + [("surface", value, 0, value, "ALL", "ALL") for value in SURFACES]
        + [("period", value, 0, "ALL", value, "ALL") for value in PERIODS]
        + [("validation_fold", value, 0, "ALL", "ALL", value) for value in FOLDS]
    )


def _frame_group_key(row: Any) -> tuple[str, str, int, str, str, str]:
    return (
        str(row.group_type),
        str(row.group_value),
        _as_int(row.serve_number, "serve_number"),
        str(row.surface),
        str(row.period),
        str(row.validation_fold),
    )


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=BY_STATE_COLUMNS),
        pd.DataFrame(columns=BY_PROFILE_COLUMNS),
        pd.DataFrame(columns=BY_GROUP_COLUMNS),
        pd.DataFrame(columns=OUTCOME_COLUMNS),
    )


def _test_seal(excluded_test_matches: int) -> dict[str, Any]:
    return {
        "test_status": "sealed",
        "used_for_method_selection": False,
        "excluded_test_matches": excluded_test_matches,
        **{field: 0 for field in TEST_ZERO_FIELDS},
    }


def _validate_test_seal(summary: Mapping[str, Any]) -> None:
    seal = summary.get("test_seal")
    expected = {
        "test_status",
        "used_for_method_selection",
        "excluded_test_matches",
        *TEST_ZERO_FIELDS,
    }
    if (
        not isinstance(seal, dict)
        or set(seal) != expected
        or seal.get("test_status") != "sealed"
        or seal.get("used_for_method_selection") is not False
    ):
        raise FeasibilityContractError("El test no esta sellado.")
    if _as_int(seal["excluded_test_matches"], "excluded_test_matches") < 0:
        raise FeasibilityContractError("excluded_test_matches invalido.")
    if any(_as_int(seal[field], field) != 0 for field in TEST_ZERO_FIELDS):
        raise FeasibilityContractError("El test contiene uso no autorizado.")


def _validate_table_schemas(result: FeasibilityResult) -> None:
    tables = (
        (result.by_state, BY_STATE_COLUMNS),
        (result.by_profile, BY_PROFILE_COLUMNS),
        (result.by_group, BY_GROUP_COLUMNS),
        (result.outcomes, OUTCOME_COLUMNS),
    )
    for frame, columns in tables:
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != list(columns):
            raise FeasibilityContractError("Schema u orden de columnas invalido.")
        for column in frame.columns:
            if bool(
                frame[column]
                .astype(str)
                .str.contains(r"(?:^|[^A-Za-z])(?:NaN|Infinity)(?:$|[^A-Za-z])", regex=True)
                .any()
            ):
                raise FeasibilityContractError("Tabla contiene literal no finito.")


def _validate_attempts(attempts: pd.DataFrame) -> None:
    if attempts.empty:
        return
    if list(attempts.columns) != list(ATTEMPT_COLUMNS):
        raise FeasibilityContractError("Schema de intentos en memoria invalido.")
    if bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()):
        raise FeasibilityContractError("Clave de intentos duplicada.")
    if not bool(
        attempts.serve_number.map(
            lambda value: isinstance(value, Integral)
            and not isinstance(value, bool)
            and value in SERVE_NUMBERS
        ).all()
    ):
        raise FeasibilityContractError("serve_number de intentos invalido.")
    if not bool(attempts.surface.isin(SURFACES).all()):
        raise FeasibilityContractError("surface de intentos invalida.")
    if not bool(attempts.period.isin(PERIODS).all()):
        raise FeasibilityContractError("period de intentos invalido.")
    if not bool(attempts.validation_fold.isin(FOLDS).all()):
        raise FeasibilityContractError("validation_fold de intentos invalido.")
    if not bool(attempts.state.isin(STATE_ORDER).all()):
        raise FeasibilityContractError("Estado de intento invalido.")
    reason_sets = {
        ReturnProfileState.OBSERVED.value: ({ReturnProfileReason.OBSERVED.value}, True),
        ReturnProfileState.UNKNOWN.value: (
            {
                ReturnProfileReason.UNKNOWN_TYPE.value,
                ReturnProfileReason.UNKNOWN_DIRECTION.value,
                ReturnProfileReason.UNKNOWN_DEPTH.value,
                ReturnProfileReason.DIRECTION_NOT_DOCUMENTED.value,
                ReturnProfileReason.DEPTH_NOT_DOCUMENTED.value,
            },
            False,
        ),
        ReturnProfileState.NOT_DOCUMENTED.value: (
            {
                ReturnProfileReason.DIRECTION_NOT_DOCUMENTED.value,
                ReturnProfileReason.DEPTH_NOT_DOCUMENTED.value,
            },
            False,
        ),
        ReturnProfileState.UNKNOWN_INITIAL.value: (
            {
                ReturnProfileReason.MISSING_PREFIX.value,
                ReturnProfileReason.TRUNCATED.value,
                ReturnProfileReason.INVALID_TYPE.value,
                ReturnProfileReason.MODIFIER.value,
                ReturnProfileReason.AMBIGUOUS_COMPONENT.value,
            },
            True,
        ),
        ReturnProfileState.CENSORED.value: (
            {
                ReturnProfileReason.ACE.value,
                ReturnProfileReason.UNRETURNED.value,
                ReturnProfileReason.FAULT.value,
                ReturnProfileReason.DOUBLE_FAULT.value,
                ReturnProfileReason.SPECIAL.value,
                ReturnProfileReason.INCOMPLETE_LET.value,
            },
            True,
        ),
    }
    for row in attempts.itertuples(index=False):
        reasons = tuple(str(row.reason_codes).split("|"))
        allowed, single = reason_sets[row.state]
        if (
            not reasons
            or any(not reason or reason not in allowed for reason in reasons)
            or len(set(reasons)) != len(reasons)
            or (single and len(reasons) != 1)
        ):
            raise FeasibilityContractError("reason_codes incompatibles con estado.")
        if row.state == ReturnProfileState.UNKNOWN.value and not set(reasons) & {
            ReturnProfileReason.UNKNOWN_TYPE.value,
            ReturnProfileReason.UNKNOWN_DIRECTION.value,
            ReturnProfileReason.UNKNOWN_DEPTH.value,
        }:
            raise FeasibilityContractError("Perfil unknown sin componente unknown.")
    for field in ("previous_attempt_was_fault", "eligible", "returner_won_point"):
        if not bool(attempts[field].map(lambda value: type(value) is bool).all()):
            raise FeasibilityContractError(f"{field} debe ser bool real.")
    expected_context = pd.Series(
        "not_applicable_first_serve", index=attempts.index, dtype="object"
    )
    second = attempts.serve_number.eq(2)
    expected_context.loc[second] = "second_serve_without_documented_first_fault"
    expected_context.loc[second & attempts.previous_attempt_was_fault] = (
        "documented_first_service_fault"
    )
    if not bool(attempts.second_serve_context.eq(expected_context).all()):
        raise FeasibilityContractError("Contexto de segundo saque incompatible.")
    observed = attempts.state.eq(ReturnProfileState.OBSERVED.value)
    expected_ids = (
        attempts.return_shot_type_code.astype(str)
        + "|"
        + attempts.return_lateral_direction_code.astype(str)
        + "|"
        + attempts.return_depth_code.astype(str)
    )
    if not bool(attempts.eligible.eq(observed).all()):
        raise FeasibilityContractError("Elegibilidad incompatible con estado.")
    if not bool(attempts.loc[observed, "profile_id"].eq(expected_ids[observed]).all()):
        raise FeasibilityContractError("profile_id de intento incompatible.")
    if not bool(attempts.loc[observed, "profile_id"].isin(PROFILE_ORDER).all()):
        raise FeasibilityContractError("Perfil observado fuera del catalogo.")
    if bool(attempts.loc[~observed, "profile_id"].notna().any()):
        raise FeasibilityContractError("Un intento no elegible conserva profile_id.")


def _validate_group_tables(result: FeasibilityResult) -> None:
    if not bool(result.by_group.aggregation_level.eq("coverage").all()):
        raise FeasibilityContractError("aggregation_level de by_group invalido.")
    if not bool(result.by_state.aggregation_level.eq("state").all()):
        raise FeasibilityContractError("aggregation_level de by_state invalido.")
    expected_keys = _expected_group_keys()
    group_rows = list(result.by_group.itertuples(index=False))
    group_keys = [_frame_group_key(row) for row in group_rows]
    if group_keys != expected_keys or len(set(group_keys)) != len(group_keys):
        raise FeasibilityContractError("Claves u orden de by_group invalidos.")
    group_map = dict(zip(group_keys, group_rows))

    expected_state_keys = [(*key, state) for key in expected_keys for state in STATE_ORDER]
    state_rows = list(result.by_state.itertuples(index=False))
    state_keys = [(*_frame_group_key(row), str(row.state)) for row in state_rows]
    if state_keys != expected_state_keys or len(set(state_keys)) != len(state_keys):
        raise FeasibilityContractError("Claves u orden de by_state invalidos.")
    state_map = dict(zip(state_keys, state_rows))

    state_fields = {
        ReturnProfileState.OBSERVED.value: "complete_profiles",
        ReturnProfileState.UNKNOWN.value: "unknown_profiles",
        ReturnProfileState.NOT_DOCUMENTED.value: "not_documented_profiles",
        ReturnProfileState.UNKNOWN_INITIAL.value: "unknown_initial_returns",
        ReturnProfileState.CENSORED.value: "censored_attempts",
    }
    for key, row in group_map.items():
        attempts = _as_int(row.attempts, "attempts")
        counts = {
            state: _as_int(getattr(row, field), field)
            for state, field in state_fields.items()
        }
        if attempts < 0 or any(value < 0 for value in counts.values()) or sum(counts.values()) != attempts:
            raise FeasibilityContractError("Estados de grupo no exhaustivos.")
        if attempts:
            _close(row.complete_profile_coverage, counts[STATE_ORDER[0]] / attempts, "complete_profile_coverage")
            _close(row.abstention_share, 1 - counts[STATE_ORDER[0]] / attempts, "abstention_share")
        elif any(_as_optional_float(getattr(row, field), field) is not None for field in ("complete_profile_coverage", "abstention_share")):
            raise FeasibilityContractError("Grupo vacio conserva proporciones.")
        for state in STATE_ORDER:
            state_row = state_map[(*key, state)]
            if _as_int(state_row.attempts, "state denominator") != attempts:
                raise FeasibilityContractError("Denominador de by_state invalido.")
            count = _as_int(state_row.state_attempts, "state_attempts")
            if count != counts[state]:
                raise FeasibilityContractError("by_state y by_group no reconcilian.")
            if attempts:
                _close(state_row.state_share, count / attempts, "state_share")
            elif _as_optional_float(state_row.state_share, "state_share") is not None:
                raise FeasibilityContractError("Estado vacio conserva proporcion.")

    total = group_map[expected_keys[0]]
    for group_type in ("serve_number", "surface", "period", "validation_fold"):
        parts = [row for row in group_rows if row.group_type == group_type]
        for field in ("attempts", *state_fields.values()):
            if sum(_as_int(getattr(row, field), field) for row in parts) != _as_int(getattr(total, field), field):
                raise FeasibilityContractError(f"Particion {group_type} no reconcilia para {field}.")


def _validate_profiles(result: FeasibilityResult) -> None:
    rows = list(result.by_profile.itertuples(index=False))
    if len(rows) != 153 or list(result.by_profile.profile_id) != list(PROFILE_ORDER):
        raise FeasibilityContractError("Catalogo u orden de perfiles invalido.")
    if bool(result.by_profile.profile_id.duplicated().any()):
        raise FeasibilityContractError("profile_id duplicado.")
    complete_total = 0
    all_total = _as_int(result.by_group.iloc[0].attempts, "all attempts")
    for row, profile_id in zip(rows, PROFILE_ORDER):
        metadata = _profile_metadata(profile_id)
        for field, expected in metadata.items():
            if getattr(row, field) != expected:
                raise FeasibilityContractError(f"Metadata de perfil invalida: {field}.")
        count = _as_int(row.complete_profile_attempts, "complete_profile_attempts")
        if count < 0:
            raise FeasibilityContractError("Conteo de perfil negativo.")
        complete_total += count
    expected_complete = _as_int(result.by_group.iloc[0].complete_profiles, "complete profiles")
    if complete_total != expected_complete:
        raise FeasibilityContractError("by_profile no reconcilia con perfiles completos.")
    for row in rows:
        count = _as_int(row.complete_profile_attempts, "complete_profile_attempts")
        if expected_complete:
            _close(row.share_of_complete_profiles, count / expected_complete, "share_of_complete_profiles")
        elif _as_optional_float(row.share_of_complete_profiles, "share_of_complete_profiles") is not None:
            raise FeasibilityContractError("Perfil sin denominador conserva share.")
        if all_total:
            _close(row.share_of_all_attempts, count / all_total, "share_of_all_attempts")
        elif _as_optional_float(row.share_of_all_attempts, "share_of_all_attempts") is not None:
            raise FeasibilityContractError("Perfil sin intentos conserva share end-to-end.")
    if expected_complete:
        _close(result.by_profile.share_of_complete_profiles.sum(), 1.0, "sum profile shares")


def _validate_outcomes(result: FeasibilityResult) -> None:
    if not bool(result.outcomes.aggregation_level.eq("profile_outcome").all()):
        raise FeasibilityContractError("aggregation_level de outcomes invalido.")
    expected_keys = [(*group, profile) for group in _expected_group_keys() for profile in PROFILE_ORDER]
    rows = list(result.outcomes.itertuples(index=False))
    keys = [(*_frame_group_key(row), str(row.profile_id)) for row in rows]
    if keys != expected_keys or len(set(keys)) != len(keys):
        raise FeasibilityContractError("Claves u orden de outcomes invalidos.")
    group_trials: dict[tuple[str, str, int, str, str, str], int] = {
        key: 0 for key in _expected_group_keys()
    }
    total_profile = {
        row.profile_id: _as_int(row.complete_profile_attempts, "profile attempts")
        for row in result.by_profile.itertuples(index=False)
    }
    for row in rows:
        metadata = _profile_metadata(str(row.profile_id))
        for field in (
            "return_shot_type_code",
            "return_lateral_direction_code",
            "return_depth_code",
        ):
            if getattr(row, field) != metadata[field]:
                raise FeasibilityContractError("Componentes de outcome incompatibles.")
        trials = _as_int(row.trials, "trials")
        returner_wins = _as_int(row.returner_wins, "returner_wins")
        server_wins = _as_int(row.server_wins, "server_wins")
        if trials < 0 or not 0 <= returner_wins <= trials or server_wins != trials - returner_wins:
            raise FeasibilityContractError("Conteos de outcome invalidos.")
        group_trials[_frame_group_key(row)] += trials
        lower, upper = wilson(returner_wins, trials, context={"aggregation_level": "validation", "profile_id": row.profile_id})
        if trials == 0:
            if any(_as_optional_float(getattr(row, field), field) is not None for field in ("returner_win_rate", "wilson_95_lower", "wilson_95_upper")):
                raise FeasibilityContractError("Outcome cero conserva tasa o Wilson.")
        else:
            _close(row.returner_win_rate, returner_wins / trials, "returner_win_rate")
            _close(row.wilson_95_lower, float(lower), "wilson_95_lower")
            _close(row.wilson_95_upper, float(upper), "wilson_95_upper")
    group_map = {_frame_group_key(row): row for row in result.by_group.itertuples(index=False)}
    for key, trials in group_trials.items():
        if trials != _as_int(group_map[key].complete_profiles, "group complete_profiles"):
            raise FeasibilityContractError("Outcomes no reconcilian con cobertura de grupo.")
    total_rows = [row for row in rows if row.group_type == "total"]
    for row in total_rows:
        if _as_int(row.trials, "total trials") != total_profile[row.profile_id]:
            raise FeasibilityContractError("Outcome total no reconcilia con by_profile.")


def _component_counts(by_profile: pd.DataFrame, field: str, order: Sequence[str]) -> dict[str, int]:
    return {
        value: int(by_profile.loc[by_profile[field].eq(value), "complete_profile_attempts"].sum())
        for value in order
    }


def _summary_from_tables(
    population: Mapping[str, int],
    attempts: pd.DataFrame,
    by_profile: pd.DataFrame,
    by_group: pd.DataFrame,
    outcomes: pd.DataFrame,
    upstream_contract: Mapping[str, str],
) -> dict[str, Any]:
    total = by_group.iloc[0]
    state_counts = {
        ReturnProfileState.OBSERVED.value: int(total.complete_profiles),
        ReturnProfileState.UNKNOWN.value: int(total.unknown_profiles),
        ReturnProfileState.NOT_DOCUMENTED.value: int(total.not_documented_profiles),
        ReturnProfileState.UNKNOWN_INITIAL.value: int(total.unknown_initial_returns),
        ReturnProfileState.CENSORED.value: int(total.censored_attempts),
    }
    complete = state_counts[ReturnProfileState.OBSERVED.value]
    second = attempts.loc[attempts.serve_number.eq(2)]
    total_outcomes = outcomes.loc[outcomes.group_type.eq("total")]
    return {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "pattern_id": PATTERN_ID,
        "analysis_status": "available_descriptive",
        "extractor_contract": {
            "classification_contract_version": CLASSIFICATION_CONTRACT_VERSION,
            "commit": UPSTREAM_COMMIT,
            "sha256": EXTRACTOR_SHA256,
        },
        "upstream_contracts": dict(upstream_contract),
        "source_contract": {
            "source": "data/processed/points_enriched.parquet",
            "columns": list(SOURCE_COLUMNS),
            "read_count": 1,
        },
        "temporal_contract": {
            "development_cutoff_inclusive": "2023-12-31",
            "test_status": "sealed",
            "period_order": list(PERIODS),
            "validation_fold_order": list(FOLDS),
        },
        "population": dict(population),
        "attempts": {
            "first_attempts": int(population["first_attempts"]),
            "second_attempts": int(population["second_attempts"]),
            "attempts_total": int(population["attempts_total"]),
            "cache_entries": int(attempts.attrs.get("cache_entries", 0)),
        },
        "second_serve_context": {
            "second_attempts": int(len(second)),
            **{
                value: int(second.second_serve_context.eq(value).sum())
                for value in SECOND_SERVE_CONTEXTS
            },
        },
        "state_counts": state_counts,
        "coverage": {
            "denominator_attempts": int(population["attempts_total"]),
            "complete_profile_attempts": complete,
            "complete_profile_share_of_all_attempts": complete / int(population["attempts_total"]),
            "abstained_attempts": int(population["attempts_total"]) - complete,
        },
        "profile_catalogue": {
            "expected_profiles": 153,
            "observed_profiles": int(by_profile.complete_profile_attempts.gt(0).sum()),
            "zero_observation_profiles": int(by_profile.complete_profile_attempts.eq(0).sum()),
            "rare_observed_profiles_fewer_than_10_attempts": int(
                by_profile.complete_profile_attempts.between(1, 9).sum()
            ),
            "rare_definition": "0 < complete_profile_attempts < 10; descriptive_only",
        },
        "component_distributions": {
            "denominator_complete_profiles": complete,
            "return_shot_type": _component_counts(by_profile, "return_shot_type_code", TYPE_ORDER),
            "return_lateral_direction": _component_counts(by_profile, "return_lateral_direction_code", DIRECTION_ORDER),
            "return_depth": _component_counts(by_profile, "return_depth_code", DEPTH_ORDER),
        },
        "outcomes_summary": {
            "unit": "documented_complete_profile_attempt",
            "denominator_complete_profiles": complete,
            "returner_wins": int(total_outcomes.returner_wins.sum()),
            "server_wins": int(total_outcomes.server_wins.sum()),
            "outcome_strata": int(len(outcomes)),
            "outcome_strata_with_zero_trials": int(outcomes.trials.eq(0).sum()),
            "outcome_strata_with_fewer_than_10_trials": int(outcomes.trials.lt(10).sum()),
            "wilson_confidence_level": 0.95,
            "significance_tests_performed": False,
            "automatic_ranking_performed": False,
        },
        "methodological_limits": list(METHODOLOGICAL_LIMITS),
        "test_seal": _test_seal(int(population["excluded_test_matches"])),
        "reconciliations": {
            "attempt_key_unique": True,
            "states_exhaustive": True,
            "profile_catalogue_complete": True,
            "profile_components_reconciled": True,
            "groups_reconciled": True,
            "outcomes_reconciled": True,
            "test_sealed_before_classification": True,
        },
        "fingerprint_contract": {
            "version": "1",
            "algorithm": "sha256",
            "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes",
        },
    }


def _validate_available(result: FeasibilityResult) -> None:
    summary = result.summary
    required = {
        "analysis_name", "analysis_version", "pattern_id", "analysis_status",
        "extractor_contract", "upstream_contracts", "source_contract",
        "temporal_contract", "population", "attempts", "second_serve_context",
        "state_counts", "coverage", "profile_catalogue",
        "component_distributions", "outcomes_summary", "methodological_limits",
        "test_seal", "reconciliations", "fingerprint_contract",
        "artifact_payload_sha256", "artifact_payload_bytes", "publication_fingerprint",
    }
    if set(summary) != required:
        raise FeasibilityContractError("Schema del summary disponible invalido.")
    if (
        summary["analysis_name"] != ANALYSIS_NAME
        or summary["analysis_version"] != ANALYSIS_VERSION
        or summary["pattern_id"] != PATTERN_ID
        or summary["analysis_status"] != "available_descriptive"
    ):
        raise FeasibilityContractError("Identidad del analisis invalida.")
    expected_upstream = {
        "extractor_commit": UPSTREAM_COMMIT,
        "extractor_sha256": EXTRACTOR_SHA256,
        "chronological_summary_sha256": CHRONOLOGICAL_SUMMARY_SHA256,
    }
    if summary["upstream_contracts"] != expected_upstream:
        raise FeasibilityContractError("Contratos upstream no coinciden.")
    if summary["extractor_contract"] != {
        "classification_contract_version": CLASSIFICATION_CONTRACT_VERSION,
        "commit": UPSTREAM_COMMIT,
        "sha256": EXTRACTOR_SHA256,
    }:
        raise FeasibilityContractError("Contrato del extractor invalido.")
    if summary["source_contract"] != {
        "source": "data/processed/points_enriched.parquet",
        "columns": list(SOURCE_COLUMNS),
        "read_count": 1,
    }:
        raise FeasibilityContractError("Contrato de fuente invalido.")
    if summary["temporal_contract"] != {
        "development_cutoff_inclusive": "2023-12-31",
        "test_status": "sealed",
        "period_order": list(PERIODS),
        "validation_fold_order": list(FOLDS),
    }:
        raise FeasibilityContractError("Contrato temporal invalido.")
    population = summary["population"]
    if not isinstance(population, dict) or set(population) != set(EXPECTED_REAL):
        raise FeasibilityContractError("Schema de poblacion invalido.")
    population = {key: _as_int(population[key], key) for key in EXPECTED_REAL}
    if any(value < 0 for value in population.values()):
        raise FeasibilityContractError("Poblacion negativa.")
    if population["first_attempts"] != population["development_point_rows"]:
        raise FeasibilityContractError("Primeros intentos no reconcilian.")
    if population["attempts_total"] != population["first_attempts"] + population["second_attempts"]:
        raise FeasibilityContractError("Intentos no reconcilian.")
    if population["source_matches"] != population["development_matches"] + population["excluded_test_matches"]:
        raise FeasibilityContractError("Particion temporal de partidos invalida.")
    attempts_summary = summary["attempts"]
    if set(attempts_summary) != {"first_attempts", "second_attempts", "attempts_total", "cache_entries"}:
        raise FeasibilityContractError("Schema del bloque attempts invalido.")
    for key in ("first_attempts", "second_attempts", "attempts_total"):
        if _as_int(attempts_summary[key], key) != population[key]:
            raise FeasibilityContractError("Bloque attempts no reconcilia.")
    cache_entries = _as_int(attempts_summary["cache_entries"], "cache_entries")
    if not 0 <= cache_entries <= population["attempts_total"]:
        raise FeasibilityContractError("Cardinalidad de cache invalida.")
    _validate_group_tables(result)
    _validate_profiles(result)
    _validate_outcomes(result)
    total = result.by_group.iloc[0]
    state_counts = {
        ReturnProfileState.OBSERVED.value: int(total.complete_profiles),
        ReturnProfileState.UNKNOWN.value: int(total.unknown_profiles),
        ReturnProfileState.NOT_DOCUMENTED.value: int(total.not_documented_profiles),
        ReturnProfileState.UNKNOWN_INITIAL.value: int(total.unknown_initial_returns),
        ReturnProfileState.CENSORED.value: int(total.censored_attempts),
    }
    if summary["state_counts"] != state_counts:
        raise FeasibilityContractError("Estados summary-CSV no reconcilian.")
    if sum(state_counts.values()) != population["attempts_total"]:
        raise FeasibilityContractError("Estados no reconcilian con poblacion.")
    complete = state_counts[ReturnProfileState.OBSERVED.value]
    coverage = summary["coverage"]
    expected_coverage = {
        "denominator_attempts": population["attempts_total"],
        "complete_profile_attempts": complete,
        "complete_profile_share_of_all_attempts": complete / population["attempts_total"],
        "abstained_attempts": population["attempts_total"] - complete,
    }
    if set(coverage) != set(expected_coverage):
        raise FeasibilityContractError("Schema de coverage invalido.")
    for key in ("denominator_attempts", "complete_profile_attempts", "abstained_attempts"):
        if _as_int(coverage[key], key) != expected_coverage[key]:
            raise FeasibilityContractError("Coverage no reconcilia.")
    _close(coverage["complete_profile_share_of_all_attempts"], expected_coverage["complete_profile_share_of_all_attempts"], "coverage share")
    catalogue = summary["profile_catalogue"]
    expected_catalogue = {
        "expected_profiles": 153,
        "observed_profiles": int(result.by_profile.complete_profile_attempts.gt(0).sum()),
        "zero_observation_profiles": int(result.by_profile.complete_profile_attempts.eq(0).sum()),
        "rare_observed_profiles_fewer_than_10_attempts": int(
            result.by_profile.complete_profile_attempts.map(
                lambda value: 0 < _as_int(value, "complete_profile_attempts") < 10
            ).sum()
        ),
        "rare_definition": "0 < complete_profile_attempts < 10; descriptive_only",
    }
    if catalogue != expected_catalogue:
        raise FeasibilityContractError("Rareza o catalogo summary-CSV invalido.")
    distributions = summary["component_distributions"]
    expected_distributions = {
        "denominator_complete_profiles": complete,
        "return_shot_type": _component_counts(result.by_profile, "return_shot_type_code", TYPE_ORDER),
        "return_lateral_direction": _component_counts(result.by_profile, "return_lateral_direction_code", DIRECTION_ORDER),
        "return_depth": _component_counts(result.by_profile, "return_depth_code", DEPTH_ORDER),
    }
    if distributions != expected_distributions:
        raise FeasibilityContractError("Distribuciones marginales no reconcilian.")
    if any(sum(distributions[key].values()) != complete for key in ("return_shot_type", "return_lateral_direction", "return_depth")):
        raise FeasibilityContractError("Marginales no suman perfiles completos.")
    second = summary["second_serve_context"]
    if set(second) != {"second_attempts", *SECOND_SERVE_CONTEXTS}:
        raise FeasibilityContractError("Schema de contexto de segundo saque invalido.")
    if _as_int(second["second_attempts"], "second_attempts") != population["second_attempts"]:
        raise FeasibilityContractError("Segundo saque no reconcilia.")
    if sum(_as_int(second[key], key) for key in SECOND_SERVE_CONTEXTS) != population["second_attempts"]:
        raise FeasibilityContractError("Contextos de segundo saque no exhaustivos.")
    outcomes = summary["outcomes_summary"]
    total_outcomes = result.outcomes.loc[result.outcomes.group_type.eq("total")]
    expected_outcomes = {
        "unit": "documented_complete_profile_attempt",
        "denominator_complete_profiles": complete,
        "returner_wins": int(total_outcomes.returner_wins.map(lambda value: _as_int(value, "returner_wins")).sum()),
        "server_wins": int(total_outcomes.server_wins.map(lambda value: _as_int(value, "server_wins")).sum()),
        "outcome_strata": len(result.outcomes),
        "outcome_strata_with_zero_trials": int(result.outcomes.trials.map(lambda value: _as_int(value, "trials") == 0).sum()),
        "outcome_strata_with_fewer_than_10_trials": int(result.outcomes.trials.map(lambda value: _as_int(value, "trials") < 10).sum()),
        "wilson_confidence_level": 0.95,
        "significance_tests_performed": False,
        "automatic_ranking_performed": False,
    }
    if outcomes != expected_outcomes or outcomes["returner_wins"] + outcomes["server_wins"] != complete:
        raise FeasibilityContractError("Resumen de outcomes no reconcilia.")
    if summary["methodological_limits"] != list(METHODOLOGICAL_LIMITS):
        raise FeasibilityContractError("Limitaciones metodologicas invalidas.")
    _validate_test_seal(summary)
    if summary["test_seal"]["excluded_test_matches"] != population["excluded_test_matches"]:
        raise FeasibilityContractError("Sellado no reconcilia con poblacion.")
    expected_reconciliations = {
        "attempt_key_unique": True,
        "states_exhaustive": True,
        "profile_catalogue_complete": True,
        "profile_components_reconciled": True,
        "groups_reconciled": True,
        "outcomes_reconciled": True,
        "test_sealed_before_classification": True,
    }
    if summary["reconciliations"] != expected_reconciliations:
        raise FeasibilityContractError("Reconciliaciones declaradas invalidas.")
    if summary["fingerprint_contract"] != {
        "version": "1", "algorithm": "sha256",
        "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes",
    }:
        raise FeasibilityContractError("Contrato de fingerprint invalido.")
    if not result.attempts.empty:
        _validate_attempts(result.attempts)
        expected_tables = (
            build_by_state(result.attempts), build_by_profile(result.attempts),
            build_by_group(result.attempts), build_outcomes(result.attempts),
        )
        try:
            for actual, expected in zip(
                (result.by_state, result.by_profile, result.by_group, result.outcomes),
                expected_tables,
            ):
                pdt.assert_frame_equal(actual, expected)
        except AssertionError as exc:
            raise FeasibilityContractError("Tablas no reconstruibles desde intentos.") from exc


def _validate_not_available(result: FeasibilityResult) -> None:
    required = {
        "analysis_name", "analysis_version", "pattern_id", "analysis_status",
        "reason_codes", "failure", "partial_diagnostics", "test_seal",
        "fingerprint_contract", "artifact_payload_sha256", "artifact_payload_bytes",
        "publication_fingerprint",
    }
    summary = result.summary
    if set(summary) != required:
        raise FeasibilityContractError("Schema not_available invalido.")
    if (
        summary["analysis_name"] != ANALYSIS_NAME
        or summary["analysis_version"] != ANALYSIS_VERSION
        or summary["pattern_id"] != PATTERN_ID
        or summary["analysis_status"] != "not_available"
    ):
        raise FeasibilityContractError("Identidad not_available invalida.")
    if any(not frame.empty for frame in (result.by_state, result.by_profile, result.by_group, result.outcomes)):
        raise FeasibilityContractError("not_available conserva filas parciales.")
    reasons = summary["reason_codes"]
    if not isinstance(reasons, list) or not reasons or any(reason not in NOT_AVAILABLE_REASON_CODES for reason in reasons):
        raise FeasibilityContractError("reason_codes not_available invalidos.")
    failure = summary["failure"]
    if (
        not isinstance(failure, dict)
        or set(failure) != {"stage", "type", "message"}
        or failure["stage"] not in FAILURE_STAGES
        or any(type(failure[key]) is not str or not failure[key] for key in ("type", "message"))
    ):
        raise FeasibilityContractError("Diagnostico de fallo invalido.")
    if re.search(r"(?i)(?:[a-z]:\\|/users/|/home/)", failure["message"]):
        raise FeasibilityContractError("Diagnostico contiene ruta absoluta.")
    if not isinstance(summary["partial_diagnostics"], dict):
        raise FeasibilityContractError("Diagnostico parcial invalido.")
    _validate_test_seal(summary)
    if summary["test_seal"]["excluded_test_matches"] != 1_531:
        raise FeasibilityContractError("not_available no conserva sellado real.")
    if summary["fingerprint_contract"] != {
        "version": "1", "algorithm": "sha256",
        "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes",
    }:
        raise FeasibilityContractError("Fingerprint contract not_available invalido.")


def _table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    payload = buffer.getvalue().encode("utf-8")
    if b"NaN" in payload or b"Infinity" in payload:
        raise FeasibilityContractError("La serializacion contiene valores no finitos.")
    return payload


def _summary_bytes(summary: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fingerprint(summary: Mapping[str, Any], csv_payloads: Sequence[bytes]) -> str:
    stable = dict(summary)
    stable.pop("publication_fingerprint", None)
    digest = hashlib.sha256(_summary_bytes(stable))
    for payload in csv_payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def validate_result(result: FeasibilityResult) -> None:
    if not isinstance(result, FeasibilityResult):
        raise TypeError("FeasibilityResult requerido.")
    _validate_json_values(result.summary)
    _validate_table_schemas(result)
    status = result.summary.get("analysis_status")
    if status == "available_descriptive":
        _validate_available(result)
    elif status == "not_available":
        _validate_not_available(result)
    else:
        raise FeasibilityContractError("analysis_status invalido.")
    frames = (result.by_state, result.by_profile, result.by_group, result.outcomes)
    csv_payloads = tuple(_table_bytes(frame) for frame in frames)
    names = ("by_state", "by_profile", "by_group", "outcomes")
    expected_hashes = {name: _sha256_bytes(payload) for name, payload in zip(names, csv_payloads)}
    expected_sizes = {name: len(payload) for name, payload in zip(names, csv_payloads)}
    if result.summary.get("artifact_payload_sha256") != expected_hashes:
        raise FeasibilityContractError("Hashes CSV no reconcilian.")
    if result.summary.get("artifact_payload_bytes") != expected_sizes:
        raise FeasibilityContractError("Tamanos CSV no reconcilian.")
    if result.summary.get("publication_fingerprint") != _fingerprint(result.summary, csv_payloads):
        raise FeasibilityContractError("publication_fingerprint no reconcilia.")


def _validate_json_values(value: Any) -> None:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise FeasibilityContractError("JSON contiene clave no textual.")
        for item in value.values():
            _validate_json_values(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_values(item)
        return
    if isinstance(value, (datetime, date, pd.Timestamp, Path)):
        raise FeasibilityContractError("JSON contiene timestamp o ruta no serializable.")
    if isinstance(value, float) and not math.isfinite(value):
        raise FeasibilityContractError("JSON contiene valor no finito.")
    if isinstance(value, str):
        if re.search(r"(?i)(?:[a-z]:\\|/users/|/home/)", value):
            raise FeasibilityContractError("JSON contiene ruta absoluta.")
        if re.search(r"(?:^|[^A-Za-z])(?:NaN|Infinity)(?:$|[^A-Za-z])", value):
            raise FeasibilityContractError("JSON contiene literal no finito.")


def finalize_result(result: FeasibilityResult) -> FeasibilityResult:
    frames = (result.by_state, result.by_profile, result.by_group, result.outcomes)
    csv_payloads = tuple(_table_bytes(frame) for frame in frames)
    names = ("by_state", "by_profile", "by_group", "outcomes")
    summary = dict(result.summary)
    summary["artifact_payload_sha256"] = {
        name: _sha256_bytes(payload) for name, payload in zip(names, csv_payloads)
    }
    summary["artifact_payload_bytes"] = {
        name: len(payload) for name, payload in zip(names, csv_payloads)
    }
    summary["publication_fingerprint"] = _fingerprint(summary, csv_payloads)
    finalized = FeasibilityResult(
        summary, result.by_state, result.by_profile, result.by_group,
        result.outcomes, result.attempts,
    )
    validate_result(finalized)
    return finalized


def analyze_points(
    points: pd.DataFrame,
    *,
    upstream_contract: Mapping[str, str],
    expected_population: Mapping[str, int] | None = None,
    extractor: Callable[..., ReturnProfileClassification] = parse_and_classify_initial_return_profile,
    stage_errors: bool = False,
) -> FeasibilityResult:
    def step(stage: str, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except Exception as exc:
            if stage_errors:
                raise _RunFailure(stage, exc) from exc
            raise

    source = step(
        "source_validation",
        lambda: validate_source_points(
            points,
            expected_rows=None if expected_population is None else expected_population["source_rows_read"],
        ),
    )
    development, population = step(
        "metadata_and_test_seal", lambda: split_development_before_parsing(source)
    )
    attempts = step(
        "attempt_construction_and_classification",
        lambda: construct_attempts(development, extractor=extractor),
    )
    population.update(
        {
            "first_attempts": int(attempts.serve_number.eq(1).sum()),
            "second_attempts": int(attempts.serve_number.eq(2).sum()),
            "attempts_total": int(len(attempts)),
        }
    )
    if expected_population is not None and population != dict(expected_population):
        error = FeasibilityContractError("Cardinalidades no reconcilian con el contrato congelado.")
        if stage_errors:
            raise _RunFailure("reconciliation", error, population) from error
        raise error
    by_state = step("aggregation", lambda: build_by_state(attempts))
    by_profile = step("aggregation", lambda: build_by_profile(attempts))
    by_group = step("aggregation", lambda: build_by_group(attempts))
    outcomes = step("outcome_analysis", lambda: build_outcomes(attempts))
    summary = _summary_from_tables(
        population, attempts, by_profile, by_group, outcomes, upstream_contract
    )
    provisional = FeasibilityResult(summary, by_state, by_profile, by_group, outcomes, attempts)
    try:
        return finalize_result(provisional)
    except Exception as exc:
        if stage_errors:
            raise _RunFailure("result_validation", exc, population) from exc
        raise


def _serialize_once(result: FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    return (
        _summary_bytes(result.summary),
        _table_bytes(result.by_state),
        _table_bytes(result.by_profile),
        _table_bytes(result.by_group),
        _table_bytes(result.outcomes),
    )


def serialize_artifacts(result: FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    validate_result(result)
    first = _serialize_once(result)
    second = _serialize_once(result)
    if first != second:
        raise FeasibilityContractError("Serializacion no determinista del mismo objeto.")
    return first


def _sanitize_message(message: str) -> str:
    value = message.replace(str(ROOT), "<repository>")
    value = re.sub(r"(?i)(?<![\w])(?:[a-z]:\\)[^\s,;]+", "<absolute-path>", value)
    value = re.sub(r"(?<!\w)/(?:users|home)/[^\s,;]+", "<absolute-path>", value, flags=re.IGNORECASE)
    return value


def not_available_result(
    error: Exception,
    stage: str,
    *,
    partial_diagnostics: Mapping[str, Any] | None = None,
) -> FeasibilityResult:
    if stage not in FAILURE_STAGES:
        raise FeasibilityContractError("Etapa de fallo invalida.")
    message = _sanitize_message(str(error)) or type(error).__name__
    summary = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "pattern_id": PATTERN_ID,
        "analysis_status": "not_available",
        "reason_codes": ["execution_failed"],
        "failure": {"stage": stage, "type": type(error).__name__, "message": message},
        "partial_diagnostics": dict(partial_diagnostics or {}),
        "test_seal": _test_seal(1_531),
        "fingerprint_contract": {
            "version": "1", "algorithm": "sha256",
            "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes",
        },
    }
    empty = _empty_frames()
    return finalize_result(FeasibilityResult(summary, *empty, pd.DataFrame()))


def _stage(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if staged.read_bytes() != payload:
            raise OSError("El staging no conserva los bytes esperados.")
        return staged
    except Exception:
        if staged.exists():
            staged.unlink()
        raise


def verify_persisted_artifacts(
    *,
    summary_path: Path = SUMMARY_PATH,
    by_state_path: Path = BY_STATE_PATH,
    by_profile_path: Path = BY_PROFILE_PATH,
    by_group_path: Path = BY_GROUP_PATH,
    outcomes_path: Path = OUTCOMES_PATH,
    expected_population: Mapping[str, int] | None = EXPECTED_REAL,
) -> FeasibilityResult:
    paths = (summary_path, by_state_path, by_profile_path, by_group_path, outcomes_path)
    if any(not path.exists() for path in paths):
        raise FeasibilityContractError("Faltan artefactos P09.")
    raw = tuple(path.read_bytes() for path in paths)
    try:
        summary = json.loads(raw[0].decode("utf-8"))
        string_columns = (
            ("aggregation_level", "group_type", "group_value", "surface", "period", "validation_fold", "state"),
            BY_PROFILE_COLUMNS[:8],
            ("aggregation_level", "group_type", "group_value", "surface", "period", "validation_fold"),
            ("aggregation_level", "group_type", "group_value", "surface", "period", "validation_fold", "profile_id", "return_shot_type_code", "return_lateral_direction_code", "return_depth_code"),
        )
        frames = [
            pd.read_csv(
                io.BytesIO(payload),
                keep_default_na=False,
                float_precision="round_trip",
                dtype={column: "object" for column in columns},
            )
            for payload, columns in zip(raw[1:], string_columns)
        ]
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise FeasibilityContractError("Artefactos P09 ilegibles.") from exc
    names = ("by_state", "by_profile", "by_group", "outcomes")
    if summary.get("artifact_payload_sha256") != {
        name: _sha256_bytes(payload) for name, payload in zip(names, raw[1:])
    }:
        raise FeasibilityContractError("Hashes persistidos invalidos.")
    if summary.get("artifact_payload_bytes") != {
        name: len(payload) for name, payload in zip(names, raw[1:])
    }:
        raise FeasibilityContractError("Tamanos persistidos invalidos.")
    if summary.get("publication_fingerprint") != _fingerprint(summary, raw[1:]):
        raise FeasibilityContractError("Fingerprint persistido invalido.")
    result = FeasibilityResult(summary, *frames, pd.DataFrame())
    validate_result(result)
    if (
        summary.get("analysis_status") != "not_available"
        and expected_population is not None
        and summary.get("population") != dict(expected_population)
    ):
        raise FeasibilityContractError("Poblacion persistida distinta del contrato real.")
    if serialize_artifacts(result) != raw:
        raise FeasibilityContractError("Los bytes persistidos no son canonicos.")
    return result


def write_artifacts(
    result: FeasibilityResult,
    *,
    summary_path: Path = SUMMARY_PATH,
    by_state_path: Path = BY_STATE_PATH,
    by_profile_path: Path = BY_PROFILE_PATH,
    by_group_path: Path = BY_GROUP_PATH,
    outcomes_path: Path = OUTCOMES_PATH,
    expected_population: Mapping[str, int] | None = EXPECTED_REAL,
) -> None:
    """Publica con rollback; no cubre un apagado abrupto entre varios replace."""
    payloads = serialize_artifacts(result)
    paths = (summary_path, by_state_path, by_profile_path, by_group_path, outcomes_path)
    staged: list[Path] = []
    previous: dict[Path, bytes | None] = {}
    try:
        for path, payload in zip(paths, payloads):
            staged.append(_stage(path, payload))
        previous = {path: path.read_bytes() if path.exists() else None for path in paths}
        for temporary, destination in zip(staged, paths):
            os.replace(temporary, destination)
        verify_persisted_artifacts(
            summary_path=summary_path,
            by_state_path=by_state_path,
            by_profile_path=by_profile_path,
            by_group_path=by_group_path,
            outcomes_path=outcomes_path,
            expected_population=expected_population,
        )
    except Exception:
        for path, payload in previous.items():
            if payload is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(payload)
        raise
    finally:
        for temporary in staged:
            if temporary.exists():
                temporary.unlink()


def run_real_analysis() -> FeasibilityResult:
    try:
        upstream = validate_upstream_contracts()
    except Exception as exc:
        raise _RunFailure("upstream_validation", exc) from exc
    try:
        points = read_source_points()
    except Exception as exc:
        raise _RunFailure("read_source_points", exc) from exc
    return analyze_points(
        points,
        upstream_contract=upstream,
        expected_population=EXPECTED_REAL,
        stage_errors=True,
    )


def _performance_path(value: str) -> Path:
    path = Path(value).resolve()
    artifacts = {
        item.resolve()
        for item in (SUMMARY_PATH, BY_STATE_PATH, BY_PROFILE_PATH, BY_GROUP_PATH, OUTCOMES_PATH)
    }
    if path in artifacts:
        raise FeasibilityContractError("performance-log no puede coincidir con un artefacto.")
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return path
    raise FeasibilityContractError("performance-log debe estar fuera del repositorio.")


def _write_performance_log(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_summary_bytes(payload))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-log", required=True)
    performance_path = _performance_path(parser.parse_args(argv).performance_log)
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    analysis_started = time.perf_counter()
    try:
        result = run_real_analysis()
    except Exception as exc:
        stage_seconds["analysis"] = time.perf_counter() - analysis_started
        stage = exc.stage if isinstance(exc, _RunFailure) else "result_validation"
        cause = exc.cause if isinstance(exc, _RunFailure) else exc
        partial = exc.partial if isinstance(exc, _RunFailure) else {}
        unavailable = not_available_result(cause, stage, partial_diagnostics=partial)
        publication_started = time.perf_counter()
        try:
            write_artifacts(unavailable)
            status = "not_available"
            failure = unavailable.summary["failure"]
        except Exception as publication_error:
            status = "publication_failed"
            failure = {
                "stage": "publication_verification",
                "type": type(publication_error).__name__,
                "message": _sanitize_message(str(publication_error)),
            }
        stage_seconds["publication"] = time.perf_counter() - publication_started
        _write_performance_log(
            performance_path,
            {
                "analysis_name": ANALYSIS_NAME,
                "analysis_status": status,
                "duration_seconds": time.perf_counter() - started,
                "stage_seconds": stage_seconds,
                "failure": failure,
            },
        )
        raise SystemExit(1)
    stage_seconds["analysis"] = time.perf_counter() - analysis_started
    publication_started = time.perf_counter()
    try:
        write_artifacts(result)
    except Exception as exc:
        stage_seconds["publication"] = time.perf_counter() - publication_started
        _write_performance_log(
            performance_path,
            {
                "analysis_name": ANALYSIS_NAME,
                "analysis_status": "publication_failed",
                "duration_seconds": time.perf_counter() - started,
                "stage_seconds": stage_seconds,
                "failure": {
                    "stage": "publication_verification",
                    "type": type(exc).__name__,
                    "message": _sanitize_message(str(exc)),
                },
            },
        )
        raise SystemExit(1)
    stage_seconds["publication"] = time.perf_counter() - publication_started
    _write_performance_log(
        performance_path,
        {
            "analysis_name": ANALYSIS_NAME,
            "analysis_status": result.summary["analysis_status"],
            "duration_seconds": time.perf_counter() - started,
            "stage_seconds": stage_seconds,
        },
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
