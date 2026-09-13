"""Infraestructura descriptiva sellada de P08, sin E/S de datos al importar."""
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
from typing import Any, Callable, Final, Mapping

import pandas as pd

from src.analysis.return_approach_feasibility import parse_and_classify_initial_return_approach

ANALYSIS_NAME: Final = "return_approach_descriptive_feasibility"
ANALYSIS_VERSION: Final = "1.0.0"
PATTERN_ID: Final = "P08"
ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "return_approach_descriptive_feasibility_summary.json"
BY_STATE_PATH = TABLES_DIR / "return_approach_descriptive_feasibility_by_state.csv"
BY_GROUP_PATH = TABLES_DIR / "return_approach_descriptive_feasibility_by_group.csv"
MARKER_CONTEXT_PATH = TABLES_DIR / "return_approach_descriptive_feasibility_marker_context.csv"
OUTCOMES_PATH = TABLES_DIR / "return_approach_descriptive_feasibility_outcomes.csv"

UPSTREAM_COMMIT = "b3f2b8e"
EXTRACTOR_PATH = ROOT / "src" / "analysis" / "return_approach_feasibility.py"
EXTRACTOR_SHA256 = "AF26034BA5653CC93D50B4636D7253D4A2EFF026ADF6E582FBCAD6E6C65CE3F2"
CHRONOLOGICAL_SUMMARY_PATH = REPORTS_DIR / "chronological_validation_summary.json"
CHRONOLOGICAL_SUMMARY_SHA256 = "90890B721CC93C82A5DE5F2EB6E3F3D4E5D4D3E83E6C9E678C5569139301799F"

CUTOFF = pd.Timestamp("2023-12-31")
SOURCE_COLUMNS = (
    "match_id", "point_number", "date", "surface", "server", "point_winner",
    "player_1", "player_2", "first_serve", "second_serve",
)
SURFACES = ("Hard", "Clay", "Grass")
PERIODS = ("to_2009", "2010s", "2020s")
FOLDS = ("pre_validation", "2020", "2021", "2022", "2023")
STATES = (
    "documented_initial_return_approach",
    "initial_return_approach_not_documented",
    "unknown_initial_return",
    "ineligible_censored",
)
STATE_REASONS: Mapping[str, tuple[str, ...]] = {
    STATES[0]: ("documented_approach_after_initial_return",),
    STATES[1]: ("no_unambiguous_immediate_approach_marker",),
    STATES[2]: (
        "missing_or_ambiguous_service_prefix",
        "unknown_or_invalid_initial_return_type",
        "ambiguous_initial_return_boundary",
    ),
    STATES[3]: (
        "censored_ace", "censored_unreturned_serve", "censored_service_fault",
        "censored_double_fault", "censored_special_event", "censored_incomplete_let",
    ),
}
MARKER_CONTEXTS = (
    "service_and_return_approach_documented",
    "service_approach_documented_return_not_documented",
    "return_approach_documented_service_not_documented",
    "neither_documented", "unknown_marker_context", "ineligible_censored",
)
SECOND_SERVE_CONTEXTS = (
    "documented_first_service_fault",
    "second_serve_without_documented_first_fault",
)
TEST_ZERO_FIELDS = (
    "test_target_rows_parsed", "test_attempts_constructed", "test_terminals_classified",
    "test_outcomes_computed", "test_rows_evaluated", "test_matches_evaluated",
    "test_scores_computed", "test_evaluation_runs", "test_recommendations_generated",
)
FAILURE_STAGES = (
    "upstream_validation", "read_source_points", "source_validation",
    "metadata_and_test_seal", "attempt_construction_and_classification", "aggregation",
    "outcome_analysis", "reconciliation", "result_validation", "serialization",
    "publication_verification",
)
NOT_AVAILABLE_REASONS = ("execution_failed",)
WILSON_Z = 1.959963984540054
WILSON_BOUNDARY_TOLERANCE = 1e-15
EXPECTED_REAL = {
    "source_rows_read": 1_280_408, "source_matches": 7_524, "source_players": 1_002,
    "development_point_rows": 1_035_760, "development_matches": 5_993,
    "development_servers": 870, "excluded_test_matches": 1_531,
    "first_attempts": 1_035_760, "second_attempts": 391_103,
    "attempts_total": 1_426_863,
}

BY_STATE_COLUMNS = (
    "group_type", "serve_number", "surface", "derived_period", "validation_fold",
    "state", "reason_code", "attempts", "matches", "servers", "returners",
    "denominator_attempts", "proportion",
)
BY_GROUP_COLUMNS = (
    "group_type", "serve_number", "surface", "derived_period", "validation_fold",
    "attempts", "matches", "servers", "returners", "positive_attempts",
    "negative_attempts", "unknown_attempts", "censored_attempts",
    "documented_initial_return_approach_prevalence",
)
MARKER_CONTEXT_COLUMNS = (
    "marker_context", "attempts", "matches", "servers", "returners",
    "denominator_attempts", "proportion",
)
OUTCOME_COLUMNS = (
    "state", "attempts", "matches", "servers", "returners", "returner_wins",
    "returner_win_rate", "wilson_low", "wilson_high",
)
ATTEMPT_COLUMNS = (
    "match_id", "point_number", "serve_number", "surface", "derived_period",
    "validation_fold", "server_player", "returner_player", "state", "reason_code",
    "second_serve_context", "service_marker", "return_marker", "returner_won_point",
)


class FeasibilityContractError(ValueError):
    """Violacion cerrada del contrato descriptivo P08."""


class _RunFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


@dataclass(frozen=True)
class FeasibilityResult:
    summary: dict[str, Any]
    by_state: pd.DataFrame
    by_group: pd.DataFrame
    marker_context: pd.DataFrame
    outcomes: pd.DataFrame
    attempts: pd.DataFrame


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def validate_upstream_contracts() -> dict[str, str]:
    if UPSTREAM_COMMIT != "b3f2b8e":
        raise FeasibilityContractError("Commit upstream P08 incompatible.")
    try:
        extractor_hash = _sha256_bytes(EXTRACTOR_PATH.read_bytes())
        chronological_hash = _sha256_bytes(CHRONOLOGICAL_SUMMARY_PATH.read_bytes())
    except OSError as exc:
        raise FeasibilityContractError("Falta un contrato upstream congelado.") from exc
    if extractor_hash != EXTRACTOR_SHA256:
        raise FeasibilityContractError("El extractor P08 no coincide con el blob congelado.")
    if chronological_hash != CHRONOLOGICAL_SUMMARY_SHA256:
        raise FeasibilityContractError("El contrato cronologico no coincide con el blob congelado.")
    try:
        chronology = json.loads(CHRONOLOGICAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeasibilityContractError("No se puede leer el contrato cronologico.") from exc
    seal = chronology.get("protected_test_contract", {})
    if seal.get("test_status") != "sealed" or seal.get("test_matches") != 1_531:
        raise FeasibilityContractError("Sellado cronologico upstream incompatible.")
    return {
        "extractor_commit": UPSTREAM_COMMIT,
        "extractor_sha256": EXTRACTOR_SHA256,
        "chronological_summary_sha256": CHRONOLOGICAL_SUMMARY_SHA256,
    }


def read_source_points() -> pd.DataFrame:
    """Unica lectura productiva del Parquet, limitada a las columnas contractuales."""
    return pd.read_parquet(POINTS_PATH, columns=list(SOURCE_COLUMNS))


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value not in (1, 2):
        raise FeasibilityContractError(f"{field} debe ser exactamente el entero 1 o 2.")
    return int(value)


def _strict_point_number(value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise FeasibilityContractError("point_number debe ser entero positivo real.")
    return int(value)


def _strict_text(series: pd.Series, field: str) -> None:
    invalid = series.map(lambda value: type(value) is not str or not value or value != value.strip())
    if bool(invalid.any()):
        raise FeasibilityContractError(f"{field} debe ser texto no vacio y sin espacios externos.")


def _dates(series: pd.Series) -> pd.Series:
    if bool(series.isna().any()):
        raise FeasibilityContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif bool(series.map(lambda value: type(value) is str).all()):
        if not bool(series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()):
            raise FeasibilityContractError("date debe usar YYYY-MM-DD sin ambiguedad.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif bool(series.map(lambda value: isinstance(value, (pd.Timestamp, datetime, date)) and not isinstance(value, bool)).all()):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        raise FeasibilityContractError("date contiene tipos ambiguos.")
    if bool(parsed.isna().any()) or getattr(parsed.dt, "tz", None) is not None:
        raise FeasibilityContractError("date invalida o con zona horaria.")
    if bool(parsed.ne(parsed.dt.normalize()).any()):
        raise FeasibilityContractError("date debe estar normalizada al dia civil.")
    return parsed.dt.normalize()


def _sequence_presence(value: object, field: str) -> str:
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return "null"
    if type(value) is not str:
        raise FeasibilityContractError(f"{field} debe ser texto o nulo.")
    if value == "":
        return "empty"
    if value.isspace():
        return "whitespace_only"
    return "substantive"


def derive_period(day: pd.Timestamp) -> str:
    return "to_2009" if day.year <= 2009 else "2010s" if day.year <= 2019 else "2020s"


def validation_fold(day: pd.Timestamp) -> str:
    if day.year <= 2019:
        return "pre_validation"
    value = str(day.year)
    if value not in FOLDS:
        raise FeasibilityContractError("Fecha de desarrollo fuera de folds autorizados.")
    return value


def validate_source_points(points: pd.DataFrame, *, expected_rows: int | None = None) -> pd.DataFrame:
    if not isinstance(points, pd.DataFrame):
        raise TypeError("points debe ser DataFrame.")
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing:
        raise FeasibilityContractError(f"Faltan columnas requeridas: {missing}")
    if expected_rows is not None and len(points) != expected_rows:
        raise FeasibilityContractError("source_rows_read inesperado.")
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
        work[field] = work[field].map(lambda value: _strict_index(value, field))
    if bool(work.groupby("match_id", sort=False)[["date", "surface", "player_1", "player_2"]].nunique(dropna=False).gt(1).any().any()):
        raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    if bool(work.first_serve.map(lambda value: _sequence_presence(value, "first_serve") != "substantive").any()):
        raise FeasibilityContractError("first_serve debe ser sustantivo.")
    work.second_serve.map(lambda value: _sequence_presence(value, "second_serve"))
    return work.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)


def split_development_before_parsing(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    development = source.loc[source.date.le(CUTOFF)].copy()
    sealed = source.loc[source.date.gt(CUTOFF)].copy()
    if bool(development.date.gt(CUTOFF).any()):
        raise FeasibilityContractError("El test no quedo sellado antes del extractor.")
    server_names = pd.Series([row.player_1 if row.server == 1 else row.player_2 for row in development.itertuples(index=False)])
    counts = {
        "source_rows_read": int(len(source)), "source_matches": int(source.match_id.nunique()),
        "source_players": int(len(set(source.player_1) | set(source.player_2))),
        "development_point_rows": int(len(development)),
        "development_matches": int(development.match_id.nunique()),
        "development_servers": int(server_names.nunique()),
        "excluded_test_matches": int(sealed.match_id.nunique()),
    }
    return development.reset_index(drop=True), counts


def construct_attempts(development: pd.DataFrame, *, extractor: Callable[..., Any] = parse_and_classify_initial_return_approach) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, bool], Any] = {}
    for row in development.itertuples(index=False):
        first_fault = False
        server_player = row.player_1 if row.server == 1 else row.player_2
        returner_player = row.player_2 if row.server == 1 else row.player_1
        for serve_number, sequence_text in ((1, row.first_serve), (2, row.second_serve)):
            presence = _sequence_presence(sequence_text, "first_serve" if serve_number == 1 else "second_serve")
            if serve_number == 2 and presence != "substantive":
                continue
            if serve_number == 1 and presence != "substantive":
                raise FeasibilityContractError("Primer saque no sustantivo.")
            key = (sequence_text, serve_number, first_fault)
            if key not in cache:
                cache[key] = extractor(sequence_text, serve_number, first_fault)
            classified = cache[key]
            rows.append({
                "match_id": row.match_id, "point_number": row.point_number,
                "serve_number": serve_number, "surface": row.surface,
                "derived_period": derive_period(row.date), "validation_fold": validation_fold(row.date),
                "server_player": server_player, "returner_player": returner_player,
                "state": classified.state.value, "reason_code": classified.reason_code.value,
                "second_serve_context": (
                    "not_applicable_first_serve" if serve_number == 1
                    else "documented_first_service_fault" if first_fault
                    else "second_serve_without_documented_first_fault"
                ),
                "service_marker": classified.service_approach_marker_span is not None,
                "return_marker": classified.return_approach_marker_span is not None,
                "returner_won_point": row.point_winner != row.server,
            })
            if serve_number == 1:
                first_fault = classified.terminal_serve_outcome == "service_fault"
    attempts = pd.DataFrame(rows)
    attempts.attrs["cache_entries"] = len(cache)
    if bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()):
        raise FeasibilityContractError("Clave de intento duplicada.")
    return attempts


def wilson(successes: int, trials: int, z: float = WILSON_Z, *, context: Mapping[str, Any] | None = None) -> tuple[float | None, float | None]:
    if type(successes) is not int or type(trials) is not int or successes < 0 or trials < 0 or successes > trials:
        raise FeasibilityContractError("Wilson requiere enteros reales 0 <= successes <= trials.")
    if trials == 0:
        return None, None
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    lower, upper = centre - spread, centre + spread
    if successes == 0:
        lower = 0.0
    if successes == trials:
        upper = 1.0
    if -WILSON_BOUNDARY_TOLERANCE <= lower < 0:
        lower = 0.0
    if 1 < upper <= 1 + WILSON_BOUNDARY_TOLERANCE:
        upper = 1.0
    valid = all(math.isfinite(value) for value in (rate, lower, upper))
    valid = valid and 0 <= lower <= upper <= 1
    valid = valid and lower - WILSON_BOUNDARY_TOLERANCE <= rate <= upper + WILSON_BOUNDARY_TOLERANCE
    if not valid:
        raise FeasibilityContractError(
            f"Wilson fuera de rango: successes={successes}, trials={trials}, rate={rate}, "
            f"lower={lower}, upper={upper}, context={dict(context or {})}"
        )
    return lower, upper


def _group_specs(attempts: pd.DataFrame) -> list[tuple[str, int, str, str, str, pd.DataFrame]]:
    specs = [("total", 0, "ALL", "ALL", "ALL", attempts)]
    specs += [("serve_number", value, "ALL", "ALL", "ALL", attempts.loc[attempts.serve_number.eq(value)]) for value in (1, 2)]
    specs += [("surface", 0, value, "ALL", "ALL", attempts.loc[attempts.surface.eq(value)]) for value in SURFACES]
    specs += [("derived_period", 0, "ALL", value, "ALL", attempts.loc[attempts.derived_period.eq(value)]) for value in PERIODS]
    specs += [("validation_fold", 0, "ALL", "ALL", value, attempts.loc[attempts.validation_fold.eq(value)]) for value in FOLDS]
    return specs


def _identity_counts(part: pd.DataFrame) -> dict[str, int]:
    return {"attempts": int(len(part)), "matches": int(part.match_id.nunique()), "servers": int(part.server_player.nunique()), "returners": int(part.returner_player.nunique())}


def build_group_tables(attempts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for group_type, serve_number, surface, period, fold, part in _group_specs(attempts):
        coordinates = {"group_type": group_type, "serve_number": serve_number, "surface": surface, "derived_period": period, "validation_fold": fold}
        identities = _identity_counts(part)
        state_counts = {state: int(part.state.eq(state).sum()) for state in STATES}
        for state in STATES:
            for reason in STATE_REASONS[state]:
                selected = part.loc[part.state.eq(state) & part.reason_code.eq(reason)]
                state_rows.append({
                    **coordinates, "state": state, "reason_code": reason,
                    **_identity_counts(selected), "denominator_attempts": len(part),
                    "proportion": None if part.empty else len(selected) / len(part),
                })
        group_rows.append({
            **coordinates, **identities, "positive_attempts": state_counts[STATES[0]],
            "negative_attempts": state_counts[STATES[1]], "unknown_attempts": state_counts[STATES[2]],
            "censored_attempts": state_counts[STATES[3]],
            "documented_initial_return_approach_prevalence": None if part.empty else state_counts[STATES[0]] / len(part),
        })
    return pd.DataFrame(state_rows, columns=BY_STATE_COLUMNS), pd.DataFrame(group_rows, columns=BY_GROUP_COLUMNS)


def derive_marker_context(attempts: pd.DataFrame) -> pd.Series:
    context = pd.Series("unknown_marker_context", index=attempts.index, dtype=object)
    censored = attempts.state.eq(STATES[3])
    negative = attempts.state.eq(STATES[1])
    documented = attempts.state.eq(STATES[0])
    context.loc[censored] = "ineligible_censored"
    context.loc[documented & attempts.service_marker] = "service_and_return_approach_documented"
    context.loc[documented & ~attempts.service_marker] = "return_approach_documented_service_not_documented"
    context.loc[negative & attempts.service_marker] = "service_approach_documented_return_not_documented"
    context.loc[negative & ~attempts.service_marker] = "neither_documented"
    return context


def build_marker_context(attempts: pd.DataFrame) -> pd.DataFrame:
    context = derive_marker_context(attempts)
    rows = []
    for value in MARKER_CONTEXTS:
        selected = attempts.loc[context.eq(value)]
        rows.append({"marker_context": value, **_identity_counts(selected), "denominator_attempts": len(attempts), "proportion": None if attempts.empty else len(selected) / len(attempts)})
    return pd.DataFrame(rows, columns=MARKER_CONTEXT_COLUMNS)


def build_outcomes(attempts: pd.DataFrame) -> pd.DataFrame:
    if not int(attempts.state.eq(STATES[0]).sum()) or not int(attempts.state.eq(STATES[1]).sum()):
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    rows = []
    for state in STATES[:2]:
        selected = attempts.loc[attempts.state.eq(state)]
        wins = int(selected.returner_won_point.sum())
        lower, upper = wilson(wins, len(selected), context={"aggregation_level": "state", "state": state})
        rows.append({"state": state, **_identity_counts(selected), "returner_wins": wins, "returner_win_rate": wins / len(selected), "wilson_low": lower, "wilson_high": upper})
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def analyze_points(
    points: pd.DataFrame,
    *,
    upstream_contract: Mapping[str, str],
    expected_population: Mapping[str, int] | None = None,
    extractor: Callable[..., Any] = parse_and_classify_initial_return_approach,
    stage_errors: bool = False,
) -> FeasibilityResult:
    def step(stage: str, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except Exception as error:
            if stage_errors:
                raise _RunFailure(stage, error) from error
            raise

    source = step("source_validation", lambda: validate_source_points(points, expected_rows=None if expected_population is None else expected_population["source_rows_read"]))
    development, population = step("metadata_and_test_seal", lambda: split_development_before_parsing(source))
    attempts = step("attempt_construction_and_classification", lambda: construct_attempts(development, extractor=extractor))
    population.update({"first_attempts": int(attempts.serve_number.eq(1).sum()), "second_attempts": int(attempts.serve_number.eq(2).sum()), "attempts_total": int(len(attempts))})
    if expected_population is not None and population != dict(expected_population):
        error = FeasibilityContractError("Cardinalidades reales no reconcilian con el contrato congelado.")
        if stage_errors:
            raise _RunFailure("reconciliation", error) from error
        raise error
    by_state, by_group = step("aggregation", lambda: build_group_tables(attempts))
    marker_context = step("aggregation", lambda: build_marker_context(attempts))
    outcomes = step("outcome_analysis", lambda: build_outcomes(attempts))
    positive, negative, unknown, censored = (int(attempts.state.eq(state).sum()) for state in STATES)
    comparable = bool(positive and negative)
    second = attempts.loc[attempts.serve_number.eq(2)]
    summary: dict[str, Any] = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "pattern_id": PATTERN_ID, "analysis_status": "available_descriptive" if comparable else "available_descriptive_not_comparable",
        "upstream_contract": dict(upstream_contract), "population": population,
        "attempt_cache_entries": int(attempts.attrs.get("cache_entries", 0)),
        "state_counts": dict(zip(STATES, (positive, negative, unknown, censored))),
        "coverage": {"denominator_attempts": len(attempts), "documented_initial_return_approach_attempts": positive, "documented_initial_return_approach_prevalence": positive / len(attempts), "not_documented_attempts": negative, "unknown_attempts": unknown, "censored_attempts": censored},
        "second_serve_context": {"second_attempts": len(second), SECOND_SERVE_CONTEXTS[0]: int(second.second_serve_context.eq(SECOND_SERVE_CONTEXTS[0]).sum()), SECOND_SERVE_CONTEXTS[1]: int(second.second_serve_context.eq(SECOND_SERVE_CONTEXTS[1]).sum())},
        "marker_context_counts": dict(zip(marker_context.marker_context, marker_context.attempts.astype(int))),
        "outcome_comparison_status": "available" if comparable else "not_available",
        "reason": None if comparable else "no_observable_nonapproach_comparator",
        "outcome_counts": {row.state: {"attempts": int(row.attempts), "returner_wins": int(row.returner_wins)} for row in outcomes.itertuples(index=False)},
        "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": population["excluded_test_matches"], **{field: 0 for field in TEST_ZERO_FIELDS}},
        "reconciliations": {"attempt_key_unique": True, "states_exhaustive": True, "groups_reconciled": True, "marker_context_exhaustive": True, "outcomes_reconciled": True, "test_sealed": True},
        "fingerprint_contract": {"version": "1", "algorithm": "sha256", "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes"},
    }
    return step("result_validation", lambda: finalize_result(FeasibilityResult(summary, by_state, by_group, marker_context, outcomes, attempts)))


def _table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    payload = buffer.getvalue().encode("utf-8")
    if b"NaN" in payload or b"Infinity" in payload:
        raise FeasibilityContractError("La serializacion contiene valores no finitos.")
    return payload


def _summary_bytes(summary: Mapping[str, Any]) -> bytes:
    return (json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _fingerprint(summary: Mapping[str, Any], csv_payloads: tuple[bytes, ...]) -> str:
    stable = dict(summary)
    stable.pop("publication_fingerprint", None)
    digest = hashlib.sha256(_summary_bytes(stable))
    for payload in csv_payloads:
        digest.update(payload)
    return digest.hexdigest().upper()


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
    if value is None or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise FeasibilityContractError(f"{field} no reconcilia.")


def _expected_group_keys() -> list[tuple[str, int, str, str, str]]:
    return ([('total', 0, 'ALL', 'ALL', 'ALL')]
            + [('serve_number', value, 'ALL', 'ALL', 'ALL') for value in (1, 2)]
            + [('surface', 0, value, 'ALL', 'ALL') for value in SURFACES]
            + [('derived_period', 0, 'ALL', value, 'ALL') for value in PERIODS]
            + [('validation_fold', 0, 'ALL', 'ALL', value) for value in FOLDS])


def _frame_group_key(row: Any) -> tuple[str, int, str, str, str]:
    return str(row.group_type), _as_int(row.serve_number, "serve_number"), str(row.surface), str(row.derived_period), str(row.validation_fold)


def _validate_test_seal(summary: Mapping[str, Any]) -> None:
    seal = summary.get("test_seal")
    expected_keys = {"test_status", "used_for_method_selection", "excluded_test_matches", *TEST_ZERO_FIELDS}
    if not isinstance(seal, dict) or set(seal) != expected_keys or seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False:
        raise FeasibilityContractError("El test no esta sellado.")
    if _as_int(seal["excluded_test_matches"], "excluded_test_matches") < 0 or any(_as_int(seal[field], field) != 0 for field in TEST_ZERO_FIELDS):
        raise FeasibilityContractError("El test contiene uso no autorizado.")


def _validate_attempts_in_memory(result: FeasibilityResult) -> None:
    attempts = result.attempts
    if attempts.empty:
        return
    if list(attempts.columns) != list(ATTEMPT_COLUMNS):
        raise FeasibilityContractError("Schema de intentos en memoria invalido.")
    if bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()):
        raise FeasibilityContractError("Clave de intentos en memoria duplicada.")
    if not attempts.serve_number.map(lambda value: isinstance(value, Integral) and not isinstance(value, bool) and value in (1, 2)).all():
        raise FeasibilityContractError("serve_number de intentos invalido.")
    if not attempts.surface.isin(SURFACES).all() or not attempts.derived_period.isin(PERIODS).all() or not attempts.validation_fold.isin(FOLDS).all():
        raise FeasibilityContractError("Dimensiones temporales de intentos invalidas.")
    if not attempts.apply(lambda row: row.reason_code in STATE_REASONS.get(row.state, ()), axis=1).all():
        raise FeasibilityContractError("Estado/reason de intentos incompatible.")
    allowed_contexts = {"not_applicable_first_serve", *SECOND_SERVE_CONTEXTS}
    if not attempts.second_serve_context.isin(allowed_contexts).all():
        raise FeasibilityContractError("Contexto de segundo saque de intentos invalido.")
    for field in ("service_marker", "return_marker", "returner_won_point"):
        if not attempts[field].map(lambda value: type(value) is bool).all():
            raise FeasibilityContractError(f"{field} de intentos debe ser bool real.")
    expected_state, expected_group = build_group_tables(attempts)
    expected_marker = build_marker_context(attempts)
    expected_outcomes = build_outcomes(attempts)
    try:
        pd.testing.assert_frame_equal(result.by_state, expected_state)
        pd.testing.assert_frame_equal(result.by_group, expected_group)
        pd.testing.assert_frame_equal(result.marker_context, expected_marker)
        pd.testing.assert_frame_equal(result.outcomes, expected_outcomes)
    except AssertionError as error:
        raise FeasibilityContractError("Las tablas no se reconstruyen desde los intentos en memoria.") from error


def _validate_available(result: FeasibilityResult) -> None:
    summary = result.summary
    required = {"analysis_name", "analysis_version", "pattern_id", "analysis_status", "upstream_contract", "population", "attempt_cache_entries", "state_counts", "coverage", "second_serve_context", "marker_context_counts", "outcome_comparison_status", "reason", "outcome_counts", "test_seal", "reconciliations", "fingerprint_contract", "artifact_payload_sha256", "artifact_payload_bytes", "publication_fingerprint"}
    if (
        set(summary) != required
        or summary["analysis_name"] != ANALYSIS_NAME
        or summary["analysis_version"] != ANALYSIS_VERSION
        or summary["pattern_id"] != PATTERN_ID
    ):
        raise FeasibilityContractError("Schema principal disponible invalido.")
    upstream = summary["upstream_contract"]
    if not isinstance(upstream, dict) or set(upstream) != {"extractor_commit", "extractor_sha256", "chronological_summary_sha256"} or not all(type(value) is str and value for value in upstream.values()):
        raise FeasibilityContractError("Contrato upstream invalido.")
    if result.attempts.empty and upstream != {
        "extractor_commit": UPSTREAM_COMMIT,
        "extractor_sha256": EXTRACTOR_SHA256,
        "chronological_summary_sha256": CHRONOLOGICAL_SUMMARY_SHA256,
    }:
        raise FeasibilityContractError("Contrato upstream persistido no coincide con P08 congelado.")
    population = summary["population"]
    if set(population) != set(EXPECTED_REAL) or any(_as_int(population[key], key) < 0 for key in EXPECTED_REAL):
        raise FeasibilityContractError("Schema de poblacion invalido.")
    if population["attempts_total"] != population["first_attempts"] + population["second_attempts"] or population["first_attempts"] != population["development_point_rows"]:
        raise FeasibilityContractError("Cardinalidades de intentos no reconcilian.")
    if population["source_matches"] != population["development_matches"] + population["excluded_test_matches"]:
        raise FeasibilityContractError("Particion de partidos no reconcilia.")
    cache_entries = _as_int(summary["attempt_cache_entries"], "attempt_cache_entries")
    if not 0 <= cache_entries <= population["attempts_total"]:
        raise FeasibilityContractError("Cardinalidad de cache invalida.")
    state_counts = summary["state_counts"]
    if set(state_counts) != set(STATES) or sum(_as_int(state_counts[state], state) for state in STATES) != population["attempts_total"]:
        raise FeasibilityContractError("Estados del summary no reconcilian.")
    coverage = summary["coverage"]
    expected_coverage = {"denominator_attempts": population["attempts_total"], "documented_initial_return_approach_attempts": state_counts[STATES[0]], "not_documented_attempts": state_counts[STATES[1]], "unknown_attempts": state_counts[STATES[2]], "censored_attempts": state_counts[STATES[3]]}
    if set(coverage) != {*expected_coverage, "documented_initial_return_approach_prevalence"} or any(_as_int(coverage.get(key), key) != value for key, value in expected_coverage.items()):
        raise FeasibilityContractError("Cobertura del summary no reconcilia.")
    _close(coverage["documented_initial_return_approach_prevalence"], state_counts[STATES[0]] / population["attempts_total"], "coverage prevalence")
    expected_keys = _expected_group_keys()
    group_keys = [_frame_group_key(row) for row in result.by_group.itertuples(index=False)]
    if group_keys != expected_keys or len(set(group_keys)) != len(group_keys):
        raise FeasibilityContractError("Claves u orden de by_group invalidos.")
    group_map = {key: row for key, row in zip(group_keys, result.by_group.itertuples(index=False))}
    total = group_map[expected_keys[0]]
    expected_total = {"attempts": population["attempts_total"], "positive_attempts": state_counts[STATES[0]], "negative_attempts": state_counts[STATES[1]], "unknown_attempts": state_counts[STATES[2]], "censored_attempts": state_counts[STATES[3]]}
    if any(_as_int(getattr(total, key), key) != value for key, value in expected_total.items()):
        raise FeasibilityContractError("Fila total de by_group invalida.")
    if _as_int(total.matches, "total matches") != population["development_matches"] or _as_int(total.servers, "total servers") != population["development_servers"]:
        raise FeasibilityContractError("Cobertura total de partidos o servidores invalida.")
    serve_rows = {int(row.serve_number): row for row in result.by_group.itertuples(index=False) if row.group_type == "serve_number"}
    if (
        _as_int(serve_rows[1].attempts, "first attempts") != population["first_attempts"]
        or _as_int(serve_rows[2].attempts, "second attempts") != population["second_attempts"]
    ):
        raise FeasibilityContractError("Particion de primeros y segundos intentos invalida.")
    for row in group_map.values():
        attempts = _as_int(row.attempts, "group attempts")
        parts = [_as_int(getattr(row, field), field) for field in ("positive_attempts", "negative_attempts", "unknown_attempts", "censored_attempts")]
        if attempts != sum(parts):
            raise FeasibilityContractError("Estados de grupo no exhaustivos.")
        for field in ("matches", "servers", "returners"):
            if not 0 <= _as_int(getattr(row, field), field) <= attempts:
                raise FeasibilityContractError("Cobertura de identidades de grupo invalida.")
        if attempts:
            _close(row.documented_initial_return_approach_prevalence, parts[0] / attempts, "group prevalence")
        elif _as_optional_float(row.documented_initial_return_approach_prevalence, "group prevalence") is not None:
            raise FeasibilityContractError("Grupo vacio con prevalencia.")
    for group_type in ("serve_number", "surface", "derived_period", "validation_fold"):
        rows = [row for key, row in group_map.items() if key[0] == group_type]
        for field in ("attempts", "positive_attempts", "negative_attempts", "unknown_attempts", "censored_attempts"):
            if sum(_as_int(getattr(row, field), field) for row in rows) != _as_int(getattr(total, field), field):
                raise FeasibilityContractError(f"Particion {group_type} no reconcilia para {field}.")
    expected_state_keys = [(*group, state, reason) for group in expected_keys for state in STATES for reason in STATE_REASONS[state]]
    state_rows = list(result.by_state.itertuples(index=False))
    actual_state_keys = [(*_frame_group_key(row), str(row.state), str(row.reason_code)) for row in state_rows]
    if actual_state_keys != expected_state_keys or len(set(actual_state_keys)) != len(actual_state_keys):
        raise FeasibilityContractError("Claves u orden de by_state invalidos.")
    for key in expected_keys:
        rows = [row for row in state_rows if _frame_group_key(row) == key]
        group = group_map[key]
        denominator = _as_int(group.attempts, "group denominator")
        for row in rows:
            count = _as_int(row.attempts, "state attempts")
            if _as_int(row.denominator_attempts, "denominator_attempts") != denominator:
                raise FeasibilityContractError("Denominador de by_state invalido.")
            if denominator:
                _close(row.proportion, count / denominator, "state proportion")
            elif _as_optional_float(row.proportion, "state proportion") is not None:
                raise FeasibilityContractError("Estado vacio con proporcion.")
            for field in ("matches", "servers", "returners"):
                if not 0 <= _as_int(getattr(row, field), field) <= count:
                    raise FeasibilityContractError("Identidades de by_state invalidas.")
        fields = dict(zip(STATES, ("positive_attempts", "negative_attempts", "unknown_attempts", "censored_attempts")))
        for state, field in fields.items():
            observed = sum(_as_int(row.attempts, "state attempts") for row in rows if row.state == state)
            if observed != _as_int(getattr(group, field), field):
                raise FeasibilityContractError("Razones no reconcilian con el estado.")
    if list(result.marker_context.marker_context) != list(MARKER_CONTEXTS) or set(summary["marker_context_counts"]) != set(MARKER_CONTEXTS):
        raise FeasibilityContractError("Orden o dominio de marker_context invalido.")
    marker_total = 0
    for row in result.marker_context.itertuples(index=False):
        attempts = _as_int(row.attempts, "marker attempts")
        marker_total += attempts
        if _as_int(row.denominator_attempts, "marker denominator") != population["attempts_total"] or summary["marker_context_counts"][row.marker_context] != attempts:
            raise FeasibilityContractError("Marker context no reconcilia.")
        _close(row.proportion, attempts / population["attempts_total"], "marker proportion")
        for field in ("matches", "servers", "returners"):
            if not 0 <= _as_int(getattr(row, field), field) <= attempts:
                raise FeasibilityContractError("Identidades de marker context invalidas.")
    if marker_total != population["attempts_total"]:
        raise FeasibilityContractError("Marker contexts no exhaustivos.")
    marker_relation = {
        STATES[0]: summary["marker_context_counts"]["service_and_return_approach_documented"] + summary["marker_context_counts"]["return_approach_documented_service_not_documented"],
        STATES[1]: summary["marker_context_counts"]["service_approach_documented_return_not_documented"] + summary["marker_context_counts"]["neither_documented"],
        STATES[2]: summary["marker_context_counts"]["unknown_marker_context"],
        STATES[3]: summary["marker_context_counts"]["ineligible_censored"],
    }
    if marker_relation != {state: state_counts[state] for state in STATES}:
        raise FeasibilityContractError("Relacion documental P03/P08 incompatible con los estados P08.")
    second = summary["second_serve_context"]
    if set(second) != {"second_attempts", *SECOND_SERVE_CONTEXTS} or _as_int(second["second_attempts"], "second attempts") != population["second_attempts"] or sum(_as_int(second[key], key) for key in SECOND_SERVE_CONTEXTS) != population["second_attempts"]:
        raise FeasibilityContractError("Contexto de segundo saque invalido.")
    comparable = state_counts[STATES[0]] > 0 and state_counts[STATES[1]] > 0
    expected_status = "available_descriptive" if comparable else "available_descriptive_not_comparable"
    if summary["analysis_status"] != expected_status:
        raise FeasibilityContractError("Status no coincide con comparabilidad.")
    if comparable:
        if summary["outcome_comparison_status"] != "available" or summary["reason"] is not None or list(result.outcomes.state) != list(STATES[:2]) or set(summary["outcome_counts"]) != set(STATES[:2]):
            raise FeasibilityContractError("Contrato de outcomes disponibles invalido.")
        for row in result.outcomes.itertuples(index=False):
            attempts, wins = _as_int(row.attempts, "outcome attempts"), _as_int(row.returner_wins, "returner_wins")
            if attempts != state_counts[row.state] or not 0 <= wins <= attempts or summary["outcome_counts"][row.state] != {"attempts": attempts, "returner_wins": wins}:
                raise FeasibilityContractError("Outcome JSON-CSV no reconcilia.")
            _close(row.returner_win_rate, wins / attempts, "returner_win_rate")
            lower, upper = wilson(wins, attempts, context={"aggregation_level": "state", "state": row.state})
            _close(row.wilson_low, lower, "wilson_low")
            _close(row.wilson_high, upper, "wilson_high")
    elif not result.outcomes.empty or summary["outcome_comparison_status"] != "not_available" or summary["reason"] != "no_observable_nonapproach_comparator" or summary["outcome_counts"] != {}:
        raise FeasibilityContractError("Contrato no comparable invalido.")
    _validate_test_seal(summary)
    if summary["test_seal"]["excluded_test_matches"] != population["excluded_test_matches"]:
        raise FeasibilityContractError("Test excluido no reconcilia con poblacion.")
    expected_reconciliations = {"attempt_key_unique": True, "states_exhaustive": True, "groups_reconciled": True, "marker_context_exhaustive": True, "outcomes_reconciled": True, "test_sealed": True}
    if summary["reconciliations"] != expected_reconciliations:
        raise FeasibilityContractError("Reconciliaciones declaradas invalidas.")
    if summary["fingerprint_contract"] != {
        "version": "1",
        "algorithm": "sha256",
        "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes",
    }:
        raise FeasibilityContractError("Contrato de fingerprint invalido.")
    _validate_attempts_in_memory(result)


def _validate_not_available(result: FeasibilityResult) -> None:
    summary = result.summary
    required = {"analysis_name", "analysis_version", "pattern_id", "analysis_status", "reason_codes", "failure", "partial_diagnostics", "test_seal", "fingerprint_contract", "artifact_payload_sha256", "artifact_payload_bytes", "publication_fingerprint"}
    if (
        set(summary) != required
        or summary["analysis_name"] != ANALYSIS_NAME
        or summary["analysis_version"] != ANALYSIS_VERSION
        or summary["pattern_id"] != PATTERN_ID
    ):
        raise FeasibilityContractError("Schema not_available invalido.")
    if any(not frame.empty for frame in (result.by_state, result.by_group, result.marker_context, result.outcomes)):
        raise FeasibilityContractError("not_available conserva filas parciales.")
    reasons = summary["reason_codes"]
    if not isinstance(reasons, list) or not reasons or any(reason not in NOT_AVAILABLE_REASONS for reason in reasons):
        raise FeasibilityContractError("reason_codes not_available invalidos.")
    failure = summary["failure"]
    if set(failure) != {"stage", "type", "message"} or failure["stage"] not in FAILURE_STAGES or not all(type(failure[key]) is str and failure[key] for key in ("type", "message")):
        raise FeasibilityContractError("Diagnostico de fallo invalido.")
    if _sanitize_message(failure["message"]) != failure["message"]:
        raise FeasibilityContractError("Diagnostico contiene ruta absoluta.")
    if not isinstance(summary["partial_diagnostics"], dict):
        raise FeasibilityContractError("Diagnostico parcial invalido.")
    _validate_test_seal(summary)
    if summary["test_seal"]["excluded_test_matches"] != 1_531:
        raise FeasibilityContractError("not_available no conserva el sellado real.")
    if summary["fingerprint_contract"] != {
        "version": "1",
        "algorithm": "sha256",
        "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes",
    }:
        raise FeasibilityContractError("Contrato de fingerprint not_available invalido.")


def validate_result(result: FeasibilityResult) -> None:
    if not isinstance(result, FeasibilityResult):
        raise TypeError("FeasibilityResult requerido.")
    frames_and_columns = ((result.by_state, BY_STATE_COLUMNS), (result.by_group, BY_GROUP_COLUMNS), (result.marker_context, MARKER_CONTEXT_COLUMNS), (result.outcomes, OUTCOME_COLUMNS))
    for frame, columns in frames_and_columns:
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != list(columns):
            raise FeasibilityContractError("Schema u orden de columnas invalido.")
        if any(bool(frame[column].astype(str).str.contains(r"(?:NaN|Infinity)", regex=True).any()) for column in frame.columns):
            raise FeasibilityContractError("Tabla contiene literal no finito.")
    status = result.summary.get("analysis_status")
    if status == "not_available":
        _validate_not_available(result)
    elif status in ("available_descriptive", "available_descriptive_not_comparable"):
        _validate_available(result)
    else:
        raise FeasibilityContractError("analysis_status invalido.")
    csv_payloads = tuple(_table_bytes(frame) for frame, _ in frames_and_columns)
    names = ("by_state", "by_group", "marker_context", "outcomes")
    hashes = {name: _sha256_bytes(payload) for name, payload in zip(names, csv_payloads)}
    sizes = {name: len(payload) for name, payload in zip(names, csv_payloads)}
    if result.summary.get("artifact_payload_sha256") != hashes or result.summary.get("artifact_payload_bytes") != sizes:
        raise FeasibilityContractError("Hashes o tamanos CSV no reconcilian.")
    if result.summary.get("publication_fingerprint") != _fingerprint(result.summary, csv_payloads):
        raise FeasibilityContractError("publication_fingerprint no reconcilia.")


def finalize_result(result: FeasibilityResult) -> FeasibilityResult:
    csv_payloads = tuple(_table_bytes(frame) for frame in (result.by_state, result.by_group, result.marker_context, result.outcomes))
    names = ("by_state", "by_group", "marker_context", "outcomes")
    summary = dict(result.summary)
    summary["artifact_payload_sha256"] = {name: _sha256_bytes(payload) for name, payload in zip(names, csv_payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, csv_payloads)}
    summary["publication_fingerprint"] = _fingerprint(summary, csv_payloads)
    finalized = FeasibilityResult(summary, result.by_state, result.by_group, result.marker_context, result.outcomes, result.attempts)
    validate_result(finalized)
    return finalized


def _serialize_once(result: FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    return (_summary_bytes(result.summary), _table_bytes(result.by_state), _table_bytes(result.by_group), _table_bytes(result.marker_context), _table_bytes(result.outcomes))


def serialize_artifacts(result: FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    validate_result(result)
    payloads = _serialize_once(result)
    repeated = _serialize_once(result)
    if payloads != repeated:
        raise FeasibilityContractError("Serializacion no determinista del mismo objeto.")
    return payloads


def not_available_result(error: Exception, stage: str, *, partial_diagnostics: Mapping[str, Any] | None = None) -> FeasibilityResult:
    if stage not in FAILURE_STAGES:
        raise FeasibilityContractError("Etapa de fallo invalida.")
    message = _sanitize_message(str(error))
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "pattern_id": PATTERN_ID, "analysis_status": "not_available",
        "reason_codes": ["execution_failed"],
        "failure": {"stage": stage, "type": type(error).__name__, "message": message or type(error).__name__},
        "partial_diagnostics": dict(partial_diagnostics or {}),
        "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": 1_531, **{field: 0 for field in TEST_ZERO_FIELDS}},
        "fingerprint_contract": {"version": "1", "algorithm": "sha256", "serialization": "utf-8_json_sorted_compact_and_exact_csv_bytes"},
    }
    empty = (pd.DataFrame(columns=BY_STATE_COLUMNS), pd.DataFrame(columns=BY_GROUP_COLUMNS), pd.DataFrame(columns=MARKER_CONTEXT_COLUMNS), pd.DataFrame(columns=OUTCOME_COLUMNS))
    return finalize_result(FeasibilityResult(summary, *empty, pd.DataFrame()))


def _stage(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        if staged.read_bytes() != payload:
            raise OSError("El staging no conserva los bytes esperados.")
        return staged
    except Exception:
        if staged.exists():
            staged.unlink()
        raise


def write_artifacts(result: FeasibilityResult, *, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_group_path: Path = BY_GROUP_PATH, marker_context_path: Path = MARKER_CONTEXT_PATH, outcomes_path: Path = OUTCOMES_PATH, expected_population: Mapping[str, int] | None = EXPECTED_REAL) -> None:
    """Publica con rollback; varios replace no cubren un apagado abrupto intermedio."""
    payloads = serialize_artifacts(result)
    if serialize_artifacts(result) != payloads:
        raise FeasibilityContractError("Serializacion no determinista antes de publicar.")
    paths = (summary_path, by_state_path, by_group_path, marker_context_path, outcomes_path)
    staged: list[Path] = []
    previous: dict[Path, bytes | None] = {}
    try:
        for path, payload in zip(paths, payloads):
            staged.append(_stage(path, payload))
        previous = {path: path.read_bytes() if path.exists() else None for path in paths}
        for temporary, destination in zip(staged, paths):
            os.replace(temporary, destination)
        verify_persisted_artifacts(summary_path=summary_path, by_state_path=by_state_path, by_group_path=by_group_path, marker_context_path=marker_context_path, outcomes_path=outcomes_path, expected_population=expected_population)
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


def verify_persisted_artifacts(*, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_group_path: Path = BY_GROUP_PATH, marker_context_path: Path = MARKER_CONTEXT_PATH, outcomes_path: Path = OUTCOMES_PATH, expected_population: Mapping[str, int] | None = EXPECTED_REAL) -> FeasibilityResult:
    paths = (summary_path, by_state_path, by_group_path, marker_context_path, outcomes_path)
    if any(not path.exists() for path in paths):
        raise FeasibilityContractError("Faltan artefactos P08.")
    raw = tuple(path.read_bytes() for path in paths)
    try:
        summary = json.loads(raw[0].decode("utf-8"))
        frames = [pd.read_csv(io.BytesIO(payload), keep_default_na=False, float_precision="round_trip") for payload in raw[1:]]
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise FeasibilityContractError("Artefactos P08 ilegibles.") from exc
    names = ("by_state", "by_group", "marker_context", "outcomes")
    if summary.get("artifact_payload_sha256") != {name: _sha256_bytes(payload) for name, payload in zip(names, raw[1:])}:
        raise FeasibilityContractError("Hashes persistidos invalidos.")
    if summary.get("artifact_payload_bytes") != {name: len(payload) for name, payload in zip(names, raw[1:])}:
        raise FeasibilityContractError("Tamanos persistidos invalidos.")
    if summary.get("publication_fingerprint") != _fingerprint(summary, raw[1:]):
        raise FeasibilityContractError("Fingerprint persistido invalido.")
    result = FeasibilityResult(summary, *frames, pd.DataFrame())
    validate_result(result)
    if summary.get("analysis_status") != "not_available" and expected_population is not None and summary.get("population") != dict(expected_population):
        raise FeasibilityContractError("La poblacion persistida no coincide con el contrato real congelado.")
    if serialize_artifacts(result) != raw:
        raise FeasibilityContractError("Los bytes persistidos no son canonicos.")
    return result


def run_real_analysis() -> FeasibilityResult:
    try:
        upstream = validate_upstream_contracts()
    except Exception as error:
        raise _RunFailure("upstream_validation", error) from error
    try:
        points = read_source_points()
    except Exception as error:
        raise _RunFailure("read_source_points", error) from error
    return analyze_points(points, upstream_contract=upstream, expected_population=EXPECTED_REAL, stage_errors=True)


def _performance_path(value: str) -> Path:
    path = Path(value).resolve()
    artifact_paths = {item.resolve() for item in (SUMMARY_PATH, BY_STATE_PATH, BY_GROUP_PATH, MARKER_CONTEXT_PATH, OUTCOMES_PATH)}
    if path in artifact_paths:
        raise FeasibilityContractError("performance-log no puede coincidir con un artefacto.")
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return path
    raise FeasibilityContractError("performance-log debe estar fuera del repositorio.")


def _sanitize_message(message: str) -> str:
    path_segment = r"[^\\/\s,;:'\"<>\)\]\}]+"
    root_text = str(ROOT)
    root_parts = [part for part in re.split(r"[\\/]+", root_text) if part]
    if root_text.startswith(("\\\\", "//")):
        root_prefix = r"[\\/]{2}"
    elif root_text.startswith(("/", "\\")):
        root_prefix = r"[\\/]"
    else:
        root_prefix = ""
    root_pattern = re.compile(
        r"(?<!\w)"
        + root_prefix
        + r"[\\/]+".join(re.escape(part) for part in root_parts)
        + rf"(?P<suffix>(?:[\\/]+{path_segment})*)"
        + r"(?=$|[\s,;:'\"<>\)\]\}])",
        flags=re.IGNORECASE,
    )

    def replace_repository_path(match: re.Match[str]) -> str:
        suffix_parts = [
            part for part in re.split(r"[\\/]+", match.group("suffix")) if part
        ]
        if any(part in {".", ".."} for part in suffix_parts):
            return "<absolute-path>"
        return "<repository>" + "".join(f"/{part}" for part in suffix_parts)

    sanitized = root_pattern.sub(replace_repository_path, message)
    sanitized = re.sub(
        r"(?i)\bfile:(?:[\\/]{2,})(?=<repository>)", "", sanitized
    )
    local_path_patterns = (
        r"(?i)(?<![\w])(?:file|vscode-file):[\\/]{2,}[^\s,;'\"<>\)\]\}]+",
        r"(?i)(?<![\w])(?:[a-z]:[\\/])[^\s,;:'\"<>\)\]\}]*",
        r"(?i)(?<![\w:])(?:\\\\|//)[^\s,;:'\"<>\)\]\}]+",
        r"(?<![\w])~[\\/][^\s,;:'\"<>\)\]\}]+",
        r"(?<!\w)(?:\.\.[\\/])+(?:[^\s,;:'\"<>\)\]\}]*)",
        r"(?<![\w:>/\\])/(?!/)[^\\/\s,;:'\"<>\)\]\}]+(?:[\\/][^\s,;:'\"<>\)\]\}]*)*",
    )
    for pattern in local_path_patterns:
        sanitized = re.sub(pattern, "<absolute-path>", sanitized)
    return sanitized


def _write_performance_log(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_summary_bytes(payload))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--performance-log", required=True)
    performance_path = _performance_path(parser.parse_args(argv).performance_log)
    started = time.perf_counter(); stage_seconds: dict[str, float] = {}
    analysis_started = time.perf_counter()
    try:
        result = run_real_analysis()
    except Exception as error:
        stage_seconds["analysis"] = time.perf_counter() - analysis_started
        failure_stage = error.stage if isinstance(error, _RunFailure) else "result_validation"
        failure_cause = error.cause if isinstance(error, _RunFailure) else error
        unavailable = not_available_result(failure_cause, failure_stage)
        publication_started = time.perf_counter()
        try:
            write_artifacts(unavailable); status = "not_available"; failure = unavailable.summary["failure"]
        except Exception as publication_error:
            status = "publication_failed"
            failure = {"stage": "publication_verification", "type": type(publication_error).__name__, "message": _sanitize_message(str(publication_error))}
        stage_seconds["publication"] = time.perf_counter() - publication_started
        _write_performance_log(performance_path, {"analysis_name": ANALYSIS_NAME, "analysis_status": status, "duration_seconds": time.perf_counter() - started, "stage_seconds": stage_seconds, "failure": failure})
        raise SystemExit(1)
    stage_seconds["analysis"] = time.perf_counter() - analysis_started
    publication_started = time.perf_counter()
    try:
        write_artifacts(result)
    except Exception as error:
        stage_seconds["publication"] = time.perf_counter() - publication_started
        _write_performance_log(performance_path, {"analysis_name": ANALYSIS_NAME, "analysis_status": "publication_failed", "duration_seconds": time.perf_counter() - started, "stage_seconds": stage_seconds, "failure": {"stage": "publication_verification", "type": type(error).__name__, "message": _sanitize_message(str(error))}})
        raise SystemExit(1)
    stage_seconds["publication"] = time.perf_counter() - publication_started
    _write_performance_log(performance_path, {"analysis_name": ANALYSIS_NAME, "analysis_status": result.summary["analysis_status"], "duration_seconds": time.perf_counter() - started, "stage_seconds": stage_seconds})
    raise SystemExit(0)


if __name__ == "__main__":
    main()
