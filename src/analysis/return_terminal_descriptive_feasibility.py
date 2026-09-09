"""Infraestructura descriptiva sellada de P07, sin ejecutar datos al importar."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.return_terminal_feasibility import (
    ReturnTerminalClassification,
    ReturnTerminalReason,
    ReturnTerminalState,
    parse_and_classify_initial_return_terminal,
    validate_return_terminal_classification,
)


ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "return_terminal_descriptive_feasibility_summary.json"
BY_STATE_PATH = TABLES_DIR / "return_terminal_descriptive_feasibility_by_state.csv"
BY_TERMINAL_PATH = TABLES_DIR / "return_terminal_descriptive_feasibility_by_terminal.csv"
BY_GROUP_PATH = TABLES_DIR / "return_terminal_descriptive_feasibility_by_group.csv"
CONSISTENCY_PATH = TABLES_DIR / "return_terminal_descriptive_feasibility_consistency.csv"

ANALYSIS_NAME = "return_terminal_descriptive_feasibility"
ANALYSIS_VERSION = "1.1.0"
PATTERN_ID = "P07"
UPSTREAM_COMMIT = "fff6118"
P07_EXTRACTOR_PATH = ROOT / "src" / "analysis" / "return_terminal_feasibility.py"
P07_EXTRACTOR_SHA256 = "4F7917F464CA5E775C944DA26868C3D23C5533F5A66F084288BC17CE0B59EF4C"
CHRONOLOGICAL_SUMMARY_PATH = REPORTS_DIR / "chronological_validation_summary.json"
CHRONOLOGICAL_SUMMARY_SHA256 = "90890B721CC93C82A5DE5F2EB6E3F3D4E5D4D3E83E6C9E678C5569139301799F"
CUTOFF = pd.Timestamp("2023-12-31")
ALLOWED_SURFACES = ("Hard", "Clay", "Grass")
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
FOLD_ORDER = ("pre_validation", "validation_2020", "validation_2021", "validation_2022", "validation_2023")
SURFACE_RANKS = {value: index for index, value in enumerate(ALLOWED_SURFACES)}
PERIOD_RANKS = {value: index for index, value in enumerate(PERIOD_ORDER)}
FOLD_RANKS = {value: index for index, value in enumerate(FOLD_ORDER)}
FIRST_SERVE_CONTEXT = "not_applicable_first_serve"
SECOND_SERVE_CONTEXT_ORDER = ("documented_first_service_fault", "second_serve_without_documented_first_fault")
STATE_ORDER = tuple(item.value for item in ReturnTerminalState)
TERMINAL_CLASS_ORDER = ("winner", "forced_error", "unforced_error")
UNFORCED_ERROR_ORDER = ("n", "w", "d", "x", "e", "!")
WILSON_Z = 1.959963984540054
WILSON_BOUNDARY_TOLERANCE = 1e-15
SOURCE_COLUMNS = ("match_id", "point_number", "date", "surface", "server", "point_winner", "player_1", "player_2", "first_serve", "second_serve")
EXPECTED_REAL = {"source_rows_read": 1_280_408, "source_matches": 7_524, "source_players": 1_002, "development_point_rows": 1_035_760, "development_matches": 5_993, "development_servers": 870, "excluded_test_matches": 1_531, "attempts_total": 1_426_863, "first_attempts": 1_035_760, "second_attempts": 391_103}
LEGACY_REPAIR_INPUT_SHA256 = {
    "summary": "C6AD8266BE6AAD8E76A4F4F7A8BA6523E4DE116D3D587A7349A572B676E5AE8F",
    "by_state": "BD5AE93838F490408DDFFFEC0039C87A79A0877B41AFE51E2E66E9940E56E5D2",
    "by_terminal": "4FA3EEE4B46800AAF24629C8095063416F10C5C9169EE42C8A1AB2B58C1EED13",
    "by_group": "E18601C92ED9FBC7502413F0FEB11A35340BCF2B9A3C569BEBBC15B3524F6E94",
    "consistency": "FC8D80154E2B8D13614CB61602CEBADB60268634841FED7A8FD49C6EFEA4402F",
}
LEGACY_REPAIR_INPUT_FINGERPRINT = "F94D2A995EEEF5882525FEC8628CD30845CCF7E5CD993090684946E683996FCE"
TEST_ZERO_FIELDS = ("test_target_rows_parsed", "test_attempts_constructed", "test_terminals_classified", "test_outcomes_computed", "test_rows_evaluated", "test_matches_evaluated", "test_scores_computed", "test_evaluation_runs", "test_recommendations_generated")

STATE_REASON_CODES = {
    ReturnTerminalState.WINNER.value: {ReturnTerminalReason.WINNER.value},
    ReturnTerminalState.FORCED_ERROR.value: {ReturnTerminalReason.FORCED_ERROR.value},
    ReturnTerminalState.UNFORCED_ERROR.value: {
        ReturnTerminalReason.UNFORCED_ERROR_N.value, ReturnTerminalReason.UNFORCED_ERROR_W.value,
        ReturnTerminalReason.UNFORCED_ERROR_D.value, ReturnTerminalReason.UNFORCED_ERROR_X.value,
        ReturnTerminalReason.UNFORCED_ERROR_E.value, ReturnTerminalReason.UNFORCED_ERROR_BANG.value,
    },
    ReturnTerminalState.NOT_DOCUMENTED.value: {
        ReturnTerminalReason.NO_TERMINAL.value, ReturnTerminalReason.NONTERMINAL_CONTENT.value,
        ReturnTerminalReason.CHALLENGE_AFTER_RETURN.value,
    },
    ReturnTerminalState.UNKNOWN_INITIAL.value: {
        ReturnTerminalReason.MISSING_PREFIX.value, ReturnTerminalReason.TRUNCATED.value,
        ReturnTerminalReason.BOUNDARY.value, ReturnTerminalReason.MODIFIER.value,
        ReturnTerminalReason.COMPONENT_ORDER.value, ReturnTerminalReason.ERROR_FORM.value,
        ReturnTerminalReason.TRAILING_CONTENT.value, ReturnTerminalReason.INCONSISTENT.value,
    },
    ReturnTerminalState.CENSORED.value: {
        ReturnTerminalReason.ACE.value, ReturnTerminalReason.UNRETURNED.value,
        ReturnTerminalReason.FAULT.value, ReturnTerminalReason.DOUBLE_FAULT.value,
        ReturnTerminalReason.SPECIAL.value, ReturnTerminalReason.INCOMPLETE_LET.value,
    },
}

BY_STATE_COLUMNS = ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "classification_state", "reason_code", "attempts", "matches", "servers", "returners", "denominator_attempts", "proportion_of_denominator"]
BY_TERMINAL_COLUMNS = ["aggregation_level", "serve_number", "terminal_class", "unforced_error_code", "attempts", "matches", "servers", "returners", "share_of_all_attempts", "share_of_eligible_terminals", "first_serve_attempts", "second_serve_attempts"]
BY_GROUP_COLUMNS = ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "second_serve_context", "attempts", "matches", "servers", "returners", "eligible_terminal_attempts", "winner_attempts", "forced_error_attempts", "unforced_error_attempts", "not_documented_attempts", "unknown_initial_attempts", "censored_attempts", "terminal_coverage", "no_terminal_share", "abstention_share"]
LEGACY_BY_GROUP_COLUMNS = [column for column in BY_GROUP_COLUMNS if column != "second_serve_context"]
CONSISTENCY_COLUMNS = ["aggregation_level", "group_type", "serve_number", "surface", "derived_period", "validation_fold", "terminal_class", "unforced_error_code", "expected_returner_won_point", "attempts", "concordant", "discordant", "concordance_rate", "concordance_wilson_low", "concordance_wilson_high", "returner_wins", "returner_win_rate", "returner_win_wilson_low", "returner_win_wilson_high"]
GROUP_TYPE_RANKS = {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4, "second_serve_context": 5}
STATE_RANKS = {value: index for index, value in enumerate(STATE_ORDER)}
TERMINAL_RANKS = {value: index for index, value in enumerate(TERMINAL_CLASS_ORDER)}
UNFORCED_CODE_RANKS = {value: index for index, value in enumerate(UNFORCED_ERROR_ORDER)}
AGGREGATION_RANKS = {"terminal_class": 0, "unforced_error_code": 1}
SECOND_SERVE_CONTEXT_RANKS = {"ALL": 0, FIRST_SERVE_CONTEXT: 1, **{value: index + 2 for index, value in enumerate(SECOND_SERVE_CONTEXT_ORDER)}}
CSV_SORT_SPECS: dict[str, tuple[tuple[str, ...], Mapping[str, Mapping[Any, int]]]] = {
    "by_state": (("group_type", "serve_number", "surface", "derived_period", "validation_fold", "classification_state", "reason_code"), {"group_type": GROUP_TYPE_RANKS, "surface": SURFACE_RANKS, "derived_period": PERIOD_RANKS, "validation_fold": FOLD_RANKS, "classification_state": STATE_RANKS}),
    "by_terminal": (("aggregation_level", "serve_number", "terminal_class", "unforced_error_code"), {"aggregation_level": AGGREGATION_RANKS, "terminal_class": TERMINAL_RANKS, "unforced_error_code": UNFORCED_CODE_RANKS}),
    "by_group": (("group_type", "serve_number", "surface", "derived_period", "validation_fold", "second_serve_context"), {"group_type": GROUP_TYPE_RANKS, "surface": SURFACE_RANKS, "derived_period": PERIOD_RANKS, "validation_fold": FOLD_RANKS, "second_serve_context": SECOND_SERVE_CONTEXT_RANKS}),
    "consistency": (("aggregation_level", "group_type", "serve_number", "surface", "derived_period", "validation_fold", "terminal_class", "unforced_error_code"), {"aggregation_level": AGGREGATION_RANKS, "group_type": GROUP_TYPE_RANKS, "surface": SURFACE_RANKS, "derived_period": PERIOD_RANKS, "validation_fold": FOLD_RANKS, "terminal_class": TERMINAL_RANKS, "unforced_error_code": UNFORCED_CODE_RANKS}),
}


class FeasibilityContractError(ValueError):
    """Violacion cerrada del contrato P07."""


class _RunFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(str(cause)); self.stage, self.cause = stage, cause


@dataclass(frozen=True)
class FeasibilityResult:
    summary: dict[str, Any]
    by_state: pd.DataFrame
    by_terminal: pd.DataFrame
    by_group: pd.DataFrame
    consistency: pd.DataFrame
    attempts: pd.DataFrame
    publication_fingerprint: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def validate_upstream_contracts() -> dict[str, str]:
    """Comprueba bytes y semantica temporal antes de leer puntos."""
    if UPSTREAM_COMMIT != "fff6118":
        raise FeasibilityContractError("Commit upstream P07 incompatible.")
    try:
        extractor_hash = _sha256_file(P07_EXTRACTOR_PATH)
        chronological_hash = _sha256_file(CHRONOLOGICAL_SUMMARY_PATH)
    except OSError as exc:
        raise FeasibilityContractError("Falta un artefacto upstream congelado.") from exc
    if extractor_hash != P07_EXTRACTOR_SHA256:
        raise FeasibilityContractError("El extractor P07 no coincide con el blob congelado.")
    if chronological_hash != CHRONOLOGICAL_SUMMARY_SHA256:
        raise FeasibilityContractError("El summary cronologico no coincide con el blob congelado.")
    try:
        summary = json.loads(CHRONOLOGICAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeasibilityContractError("No se puede leer el contrato cronologico congelado.") from exc
    frozen = summary.get("frozen_sensitivity_contract", {})
    rolling = summary.get("rolling_origin_contract", {})
    seal = summary.get("protected_test_contract", {})
    source = summary.get("source_contract", {})
    populations = {item.get("split"): item for item in summary.get("split_population", [])}
    if (
        frozen.get("freeze_date") != "2023-12-31"
        or rolling.get("validation_years") != [2020, 2021, 2022, 2023]
        or rolling.get("validation_folds") != 4
        or seal.get("test_status") != "sealed"
        or seal.get("test_matches") != 1_531
        or source.get("matches_after_immediate_reduction") != 7_524
        or source.get("source_point_rows") != 1_280_408
        or source.get("players") != 1_002
        or populations.get("validation", {}).get("matches") != 1_805
        or populations.get("test", {}).get("matches") != 1_531
        or populations.get("train", {}).get("matches") != 4_188
    ):
        raise FeasibilityContractError("Semantica cronologica congelada incompatible.")
    return {"extractor_commit": UPSTREAM_COMMIT, "extractor_sha256": P07_EXTRACTOR_SHA256, "chronological_summary_sha256": CHRONOLOGICAL_SUMMARY_SHA256}


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value not in (1, 2):
        raise FeasibilityContractError(f"{field} debe ser exactamente el entero 1 o 2; recibido {value!r}.")
    return int(value)


def _strict_point_number(value: object) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise FeasibilityContractError(f"point_number debe ser entero positivo real; recibido {value!r}.")
    return int(value)


def _strict_text(series: pd.Series, field: str) -> None:
    invalid = series.map(lambda value: not isinstance(value, str) or not value or value != value.strip())
    if bool(invalid.any()): raise FeasibilityContractError(f"{field} debe ser texto no vacio y sin espacios externos.")


def _dates(series: pd.Series) -> pd.Series:
    if series.isna().any(): raise FeasibilityContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series): parsed = pd.to_datetime(series, errors="coerce")
    elif bool(series.map(lambda value: type(value) is str).all()):
        if not bool(series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()): raise FeasibilityContractError("date debe usar YYYY-MM-DD sin ambiguedad.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif bool(series.map(lambda value: isinstance(value, (pd.Timestamp, datetime, date)) and not isinstance(value, bool)).all()): parsed = pd.to_datetime(series, errors="coerce")
    else: raise FeasibilityContractError("date contiene tipos ambiguos.")
    if parsed.isna().any() or getattr(parsed.dt, "tz", None) is not None: raise FeasibilityContractError("date invalida o con zona horaria.")
    return parsed.dt.normalize()


def _presence(value: object, field: str) -> str:
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)): return "null"
    if type(value) is not str: raise FeasibilityContractError(f"{field} debe ser texto o nulo.")
    if value == "": return "empty"
    if value.isspace(): return "whitespace_only"
    return "substantive"


def derive_period(day: pd.Timestamp) -> str:
    return "to_2009" if day.year <= 2009 else "2010s" if day.year <= 2019 else "2020s"


def validation_fold(day: pd.Timestamp) -> str:
    if day.year <= 2019: return "pre_validation"
    candidate = f"validation_{day.year}"
    if candidate not in FOLD_ORDER: raise FeasibilityContractError("Fecha de desarrollo fuera de folds autorizados.")
    return candidate


def validate_source_points(points: pd.DataFrame, *, expected_rows: int | None = None) -> pd.DataFrame:
    if not isinstance(points, pd.DataFrame): raise TypeError("points debe ser DataFrame.")
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing: raise FeasibilityContractError(f"Faltan columnas requeridas: {missing}")
    if expected_rows is not None and len(points) != expected_rows: raise FeasibilityContractError("source_rows_read inesperado.")
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"): _strict_text(work[field], field)
    work.point_number = work.point_number.map(_strict_point_number)
    if bool(work.duplicated(["match_id", "point_number"]).any()): raise FeasibilityContractError("(match_id, point_number) debe ser clave unica no nula.")
    work.date = _dates(work.date)
    unexpected = sorted(set(work.loc[~work.surface.isin(ALLOWED_SURFACES), "surface"]))
    if unexpected: raise FeasibilityContractError(f"Superficies inesperadas: {unexpected}")
    if bool(work.player_1.eq(work.player_2).any()): raise FeasibilityContractError("player_1 y player_2 deben ser distintos.")
    for field in ("server", "point_winner"): work[field] = work[field].map(lambda value: _strict_index(value, field))
    if bool(work.groupby("match_id", sort=False)[["date", "surface", "player_1", "player_2"]].nunique(dropna=False).gt(1).any().any()): raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    return work.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)


def split_development_before_parsing(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    development, sealed = source.loc[source.date.le(CUTOFF)].copy(), source.loc[source.date.gt(CUTOFF)]
    if development.empty or sealed.empty or bool(development.date.gt(CUTOFF).any()): raise FeasibilityContractError("El sellado temporal no particiona la fuente.")
    counts = {"source_rows_read": int(len(source)), "source_matches": int(source.match_id.nunique()), "source_players": int(len(set(source.player_1) | set(source.player_2))), "development_point_rows": int(len(development)), "development_matches": int(development.match_id.nunique()), "development_servers": int(pd.Series(np.where(development.server.eq(1), development.player_1, development.player_2)).nunique()), "excluded_test_matches": int(sealed.match_id.nunique())}
    return development.reset_index(drop=True), counts


def _attempt_row(row: Any, number: int, previous_fault: bool, context: str, classified: ReturnTerminalClassification) -> dict[str, Any]:
    expected = {"winner": True, "forced_error": False, "unforced_error": False}.get(classified.terminal_kind)
    return {"match_id": row.match_id, "point_number": row.point_number, "serve_number": number, "date": row.date, "surface": row.surface, "derived_period": derive_period(row.date), "validation_fold": validation_fold(row.date), "server_player": row.player_1 if row.server == 1 else row.player_2, "returner_player": row.player_2 if row.server == 1 else row.player_1, "previous_attempt_was_fault": previous_fault, "second_serve_context": context, "classification_state": classified.classification_state.value, "reason_code": classified.reason_code.value, "eligible_for_terminal_comparison": classified.eligible_for_terminal_comparison, "return_event_observed": classified.return_event_observed, "actor": classified.actor, "terminal_class": classified.terminal_kind, "unforced_error_code": classified.error_code if classified.terminal_kind == "unforced_error" else None, "expected_returner_won_point": expected, "terminal_serve_outcome": classified.terminal_serve_outcome, "returner_won_point": bool(row.point_winner != row.server)}


def construct_attempts(development: pd.DataFrame, *, extractor: Callable[..., ReturnTerminalClassification] = parse_and_classify_initial_return_terminal, validator: Callable[[ReturnTerminalClassification], None] = validate_return_terminal_classification) -> pd.DataFrame:
    cache: dict[tuple[str, int, bool], ReturnTerminalClassification] = {}
    def classify(text: str, number: int, prior: bool) -> ReturnTerminalClassification:
        key = (text, number, prior)
        if key not in cache:
            value = extractor(text, number, previous_attempt_was_fault=prior); validator(value); cache[key] = value
        return cache[key]
    rows: list[dict[str, Any]] = []
    for row in development.itertuples(index=False):
        if _presence(row.first_serve, "first_serve") != "substantive": raise FeasibilityContractError("first_serve debe ser sustantivo.")
        first = classify(row.first_serve, 1, False); rows.append(_attempt_row(row, 1, False, FIRST_SERVE_CONTEXT, first))
        if _presence(row.second_serve, "second_serve") == "substantive":
            prior = first.terminal_serve_outcome == "service_fault"
            context = SECOND_SERVE_CONTEXT_ORDER[0] if prior else SECOND_SERVE_CONTEXT_ORDER[1]
            rows.append(_attempt_row(row, 2, prior, context, classify(row.second_serve, 2, prior)))
    attempts = pd.DataFrame(rows)
    if attempts.empty or bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()): raise FeasibilityContractError("Intentos vacios o clave duplicada.")
    attempts = attempts.sort_values(["date", "match_id", "point_number", "serve_number"], kind="stable").reset_index(drop=True)
    if not attempts.loc[attempts.serve_number.eq(1), "second_serve_context"].eq(FIRST_SERVE_CONTEXT).all() or not attempts.loc[attempts.serve_number.eq(2), "second_serve_context"].isin(SECOND_SERVE_CONTEXT_ORDER).all():
        raise FeasibilityContractError("Contexto de segundo saque invalido.")
    attempts.attrs["cache_entries"] = len(cache)
    return attempts


def _counts(part: pd.DataFrame) -> dict[str, int]:
    return {"attempts": int(len(part)), "matches": int(part.match_id.nunique()), "servers": int(part.server_player.nunique()), "returners": int(part.returner_player.nunique())}


def _canonical_sort_key(values: Mapping[str, Any], keys: Sequence[str], ranks: Mapping[str, Mapping[Any, int]]) -> tuple[Any, ...]:
    rank_values: list[int] = []
    raw_values: list[str] = []
    for key in keys:
        value = values[key]
        null = bool(pd.isna(value))
        if key in ranks:
            rank_values.append(len(ranks[key]) if null else ranks[key].get(value, len(ranks[key])))
        raw_values.append("" if null else str(value))
    return tuple([*rank_values, *raw_values])


def _ordered(frame: pd.DataFrame, keys: Sequence[str], ranks: Mapping[str, Mapping[Any, int]] | None = None) -> pd.DataFrame:
    ranks = ranks or {}
    positions = sorted(range(len(frame)), key=lambda position: _canonical_sort_key(frame.iloc[position], keys, ranks))
    return frame.iloc[positions].reset_index(drop=True)


def _group_specs(attempts: pd.DataFrame):
    yield "total", 0, "ALL", "ALL", "ALL", attempts
    for number in (1, 2): yield "serve_number", number, "ALL", "ALL", "ALL", attempts.loc[attempts.serve_number.eq(number)]
    for surface in ALLOWED_SURFACES: yield "surface", 0, surface, "ALL", "ALL", attempts.loc[attempts.surface.eq(surface)]
    for period in PERIOD_ORDER: yield "derived_period", 0, "ALL", period, "ALL", attempts.loc[attempts.derived_period.eq(period)]
    for fold in FOLD_ORDER: yield "validation_fold", 0, "ALL", "ALL", fold, attempts.loc[attempts.validation_fold.eq(fold)]


def build_by_state(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, number, surface, period, fold, part in _group_specs(attempts):
        for state in STATE_ORDER:
            for reason in sorted(STATE_REASON_CODES[state]):
                group = part.loc[part.classification_state.eq(state) & part.reason_code.eq(reason)]
                rows.append({"group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, "classification_state": state, "reason_code": reason, **_counts(group), "denominator_attempts": int(len(part)), "proportion_of_denominator": None if part.empty else len(group) / len(part)})
    return _ordered(pd.DataFrame(rows, columns=BY_STATE_COLUMNS), *CSV_SORT_SPECS["by_state"])


def _terminal_rows(part: pd.DataFrame, number: int) -> list[dict[str, Any]]:
    eligible = part.loc[part.eligible_for_terminal_comparison]
    rows = []
    for terminal in TERMINAL_CLASS_ORDER:
        group = eligible.loc[eligible.terminal_class.eq(terminal)]
        rows.append({"aggregation_level": "terminal_class", "serve_number": number, "terminal_class": terminal, "unforced_error_code": None, **_counts(group), "share_of_all_attempts": None if part.empty else len(group) / len(part), "share_of_eligible_terminals": None if eligible.empty else len(group) / len(eligible), "first_serve_attempts": int(group.serve_number.eq(1).sum()), "second_serve_attempts": int(group.serve_number.eq(2).sum())})
    for code in UNFORCED_ERROR_ORDER:
        group = eligible.loc[eligible.terminal_class.eq("unforced_error") & eligible.unforced_error_code.eq(code)]
        rows.append({"aggregation_level": "unforced_error_code", "serve_number": number, "terminal_class": "unforced_error", "unforced_error_code": code, **_counts(group), "share_of_all_attempts": None if part.empty else len(group) / len(part), "share_of_eligible_terminals": None if eligible.empty else len(group) / len(eligible), "first_serve_attempts": int(group.serve_number.eq(1).sum()), "second_serve_attempts": int(group.serve_number.eq(2).sum())})
    return rows


def build_by_terminal(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = _terminal_rows(attempts, 0)
    for number in (1, 2): rows.extend(_terminal_rows(attempts.loc[attempts.serve_number.eq(number)], number))
    return _ordered(pd.DataFrame(rows, columns=BY_TERMINAL_COLUMNS), *CSV_SORT_SPECS["by_terminal"])


def build_by_group(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, number, surface, period, fold, part in _group_specs(attempts):
        counts = part.classification_state.value_counts()
        eligible = int(part.eligible_for_terminal_comparison.sum())
        rows.append(_group_row(kind, number, surface, period, fold, "ALL", part, counts, eligible))
    second = attempts.loc[attempts.serve_number.eq(2)]
    for context in SECOND_SERVE_CONTEXT_ORDER:
        part = second.loc[second.second_serve_context.eq(context)]
        counts = part.classification_state.value_counts(); eligible = int(part.eligible_for_terminal_comparison.sum())
        rows.append(_group_row("second_serve_context", 2, "ALL", "ALL", "ALL", context, part, counts, eligible))
    return _ordered(pd.DataFrame(rows, columns=BY_GROUP_COLUMNS), *CSV_SORT_SPECS["by_group"])


def _group_row(kind: str, number: int, surface: str, period: str, fold: str, context: str, part: pd.DataFrame, counts: pd.Series, eligible: int) -> dict[str, Any]:
    return {"group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, "second_serve_context": context, **_counts(part), "eligible_terminal_attempts": eligible, "winner_attempts": int(counts.get(ReturnTerminalState.WINNER.value, 0)), "forced_error_attempts": int(counts.get(ReturnTerminalState.FORCED_ERROR.value, 0)), "unforced_error_attempts": int(counts.get(ReturnTerminalState.UNFORCED_ERROR.value, 0)), "not_documented_attempts": int(counts.get(ReturnTerminalState.NOT_DOCUMENTED.value, 0)), "unknown_initial_attempts": int(counts.get(ReturnTerminalState.UNKNOWN_INITIAL.value, 0)), "censored_attempts": int(counts.get(ReturnTerminalState.CENSORED.value, 0)), "terminal_coverage": None if part.empty else eligible / len(part), "no_terminal_share": None if part.empty else int(counts.get(ReturnTerminalState.NOT_DOCUMENTED.value, 0)) / len(part), "abstention_share": None if part.empty else (int(counts.get(ReturnTerminalState.UNKNOWN_INITIAL.value, 0)) + int(counts.get(ReturnTerminalState.CENSORED.value, 0))) / len(part)}


def _wilson_error(successes: int, trials: int, lower: float, upper: float, context: Mapping[str, Any] | None) -> FeasibilityContractError:
    safe = {"successes": successes, "trials": trials, "rate": successes / trials if trials else None, "lower": lower, "upper": upper, **(dict(context) if context else {})}
    return FeasibilityContractError("Wilson fuera de rango: " + ", ".join(f"{key}={value!r}" for key, value in safe.items()))


def wilson(successes: int, trials: int, z: float = WILSON_Z, *, context: Mapping[str, Any] | None = None) -> tuple[float | None, float | None]:
    if type(successes) is not int or type(trials) is not int or successes < 0 or trials < successes: raise FeasibilityContractError("Wilson requiere cuentas enteras coherentes.")
    if trials == 0: return None, None
    rate = successes / trials; denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    lower, upper = centre - spread, centre + spread
    if successes == 0: lower = 0.0
    if successes == trials: upper = 1.0
    if -WILSON_BOUNDARY_TOLERANCE <= lower <= WILSON_BOUNDARY_TOLERANCE: lower = 0.0
    if 1 - WILSON_BOUNDARY_TOLERANCE <= upper <= 1 + WILSON_BOUNDARY_TOLERANCE: upper = 1.0
    if lower > rate and lower - rate <= WILSON_BOUNDARY_TOLERANCE: lower = rate
    if upper < rate and rate - upper <= WILSON_BOUNDARY_TOLERANCE: upper = rate
    if not (math.isfinite(rate) and math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1 and lower - WILSON_BOUNDARY_TOLERANCE <= rate <= upper + WILSON_BOUNDARY_TOLERANCE): raise _wilson_error(successes, trials, lower, upper, context)
    return lower, upper


def _consistency_rows(part: pd.DataFrame, kind: str, number: int, surface: str, period: str, fold: str) -> list[dict[str, Any]]:
    rows = []
    for terminal in TERMINAL_CLASS_ORDER:
        # Unforced errors publish both the terminal-class total and its six
        # documented subcodes; the latter never substitutes the former.
        codes = (None,) if terminal != "unforced_error" else (None, *UNFORCED_ERROR_ORDER)
        for code in codes:
            group = part.loc[part.terminal_class.eq(terminal)]
            if code is not None: group = group.loc[group.unforced_error_code.eq(code)]
            expected = {"winner": True, "forced_error": False, "unforced_error": False}[terminal]
            attempts, concordant = int(len(group)), int(group.returner_won_point.eq(expected).sum())
            wins = int(group.returner_won_point.sum())
            concordance_low, concordance_high = wilson(concordant, attempts, context={"aggregation_level": "terminal_class" if code is None else "unforced_error_code", "group_type": kind, "terminal_class": terminal, "unforced_error_code": code})
            wins_low, wins_high = wilson(wins, attempts, context={"aggregation_level": "terminal_class" if code is None else "unforced_error_code", "group_type": kind, "terminal_class": terminal, "unforced_error_code": code})
            rows.append({"aggregation_level": "terminal_class" if code is None else "unforced_error_code", "group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, "terminal_class": terminal, "unforced_error_code": code, "expected_returner_won_point": expected, "attempts": attempts, "concordant": concordant, "discordant": attempts - concordant, "concordance_rate": None if attempts == 0 else concordant / attempts, "concordance_wilson_low": concordance_low, "concordance_wilson_high": concordance_high, "returner_wins": wins, "returner_win_rate": None if attempts == 0 else wins / attempts, "returner_win_wilson_low": wins_low, "returner_win_wilson_high": wins_high})
    return rows


def build_consistency(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, number, surface, period, fold, part in _group_specs(attempts): rows.extend(_consistency_rows(part, kind, number, surface, period, fold))
    return _ordered(pd.DataFrame(rows, columns=CONSISTENCY_COLUMNS), *CSV_SORT_SPECS["consistency"])


def _test_seal(counts: Mapping[str, int]) -> dict[str, Any]:
    return {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": counts["excluded_test_matches"], **{field: 0 for field in TEST_ZERO_FIELDS}}


def _validate_test_seal(seal: Mapping[str, Any], excluded_test_matches: int | None) -> None:
    if (
        set(seal) != {"test_status", "used_for_method_selection", "excluded_test_matches", *TEST_ZERO_FIELDS}
        or seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False
        or seal.get("excluded_test_matches") != excluded_test_matches
        or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS)
    ):
        raise FeasibilityContractError("Test seal invalido.")


def validate_reconciliations(attempts: pd.DataFrame, by_state: pd.DataFrame, by_terminal: pd.DataFrame, by_group: pd.DataFrame, consistency: pd.DataFrame) -> dict[str, bool]:
    if bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()): raise FeasibilityContractError("Clave de intento duplicada.")
    if set(attempts.classification_state) - set(STATE_ORDER) or bool((attempts.eligible_for_terminal_comparison != attempts.classification_state.isin({ReturnTerminalState.WINNER.value, ReturnTerminalState.FORCED_ERROR.value, ReturnTerminalState.UNFORCED_ERROR.value})).any()): raise FeasibilityContractError("Estados o elegibilidad P07 invalidos.")
    first, second = attempts.loc[attempts.serve_number.eq(1)], attempts.loc[attempts.serve_number.eq(2)]
    if not first.second_serve_context.eq(FIRST_SERVE_CONTEXT).all() or not second.second_serve_context.isin(SECOND_SERVE_CONTEXT_ORDER).all(): raise FeasibilityContractError("Contextos de segundo saque fuera del catalogo.")
    if not second.previous_attempt_was_fault.eq(second.second_serve_context.eq(SECOND_SERVE_CONTEXT_ORDER[0])).all(): raise FeasibilityContractError("previous_attempt_was_fault no reconcilia con su contexto documental.")
    non_context_groups = by_group.loc[~by_group.group_type.eq("second_serve_context")]
    if not non_context_groups.second_serve_context.eq("ALL").all(): raise FeasibilityContractError("Contexto de segundo saque aplicado fuera de su agregado contractual.")
    total_group = by_group.loc[by_group.group_type.eq("total")]
    if len(total_group) != 1 or int(total_group.iloc[0].attempts) != len(attempts): raise FeasibilityContractError("Grupo total no reconcilia.")
    total_state = by_state.loc[by_state.group_type.eq("total")]
    state_counts = total_state.groupby("classification_state", sort=False).attempts.sum().to_dict()
    expected_states = attempts.classification_state.value_counts().to_dict()
    if {key: int(state_counts.get(key, 0)) for key in STATE_ORDER} != {key: int(expected_states.get(key, 0)) for key in STATE_ORDER}: raise FeasibilityContractError("Estados no reconcilian.")
    eligible = int(attempts.eligible_for_terminal_comparison.sum())
    if eligible != int(total_group.iloc[0].eligible_terminal_attempts) or eligible != int(total_group.iloc[0].winner_attempts + total_group.iloc[0].forced_error_attempts + total_group.iloc[0].unforced_error_attempts): raise FeasibilityContractError("Elegibles no reconcilian.")
    context_rows = by_group.loc[by_group.group_type.eq("second_serve_context")]
    if (
        len(context_rows) != len(SECOND_SERVE_CONTEXT_ORDER)
        or bool(context_rows.duplicated(["second_serve_context"]).any())
        or set(context_rows.second_serve_context) != set(SECOND_SERVE_CONTEXT_ORDER)
        or not context_rows.serve_number.eq(2).all()
        or not context_rows[["surface", "derived_period", "validation_fold"]].eq("ALL").all().all()
        or int(context_rows.attempts.sum()) != len(second)
    ):
        raise FeasibilityContractError("Contextos de segundo saque no reconcilian.")
    expected_context_rows = build_by_group(attempts).loc[lambda frame: frame.group_type.eq("second_serve_context")].reset_index(drop=True)
    try:
        pdt.assert_frame_equal(
            context_rows.sort_values("second_serve_context", kind="stable").reset_index(drop=True),
            expected_context_rows.sort_values("second_serve_context", kind="stable").reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as error:
        raise FeasibilityContractError("Filas de contexto de segundo saque no reconcilian con intentos.") from error
    terminal_total = by_terminal.loc[by_terminal.aggregation_level.eq("terminal_class") & by_terminal.serve_number.eq(0)]
    if int(terminal_total.attempts.sum()) != eligible: raise FeasibilityContractError("Clases terminales no reconcilian.")
    unforced_total = by_terminal.loc[by_terminal.aggregation_level.eq("unforced_error_code") & by_terminal.serve_number.eq(0)]
    if int(unforced_total.attempts.sum()) != int(total_group.iloc[0].unforced_error_attempts): raise FeasibilityContractError("Codigos unforced no reconcilian.")
    if not consistency.concordant.add(consistency.discordant).eq(consistency.attempts).all() or not consistency.returner_wins.le(consistency.attempts).all(): raise FeasibilityContractError("Consistencia terminal-outcome invalida.")
    for frame, columns in ((by_state, BY_STATE_COLUMNS), (by_terminal, BY_TERMINAL_COLUMNS), (by_group, BY_GROUP_COLUMNS), (consistency, CONSISTENCY_COLUMNS)):
        if list(frame.columns) != columns or bool(frame.duplicated().any()): raise FeasibilityContractError("Schema o filas duplicadas.")
    try:
        pdt.assert_frame_equal(by_state, build_by_state(attempts), check_dtype=False, check_exact=True)
        pdt.assert_frame_equal(by_terminal, build_by_terminal(attempts), check_dtype=False, check_exact=True)
        pdt.assert_frame_equal(by_group, build_by_group(attempts), check_dtype=False, check_exact=True)
        pdt.assert_frame_equal(consistency, build_consistency(attempts), check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise FeasibilityContractError("Agregados P07 no reconcilian con los intentos.") from exc
    return {"attempt_key_unique": True, "states_exhaustive_exclusive": True, "eligible_terminal_reconciled": True, "unforced_codes_reconciled": True, "second_serve_context_reconciled": True, "groups_reconciled": True, "consistency_reconciled": True, "test_sealed": True}


def _table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(); frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="", encoding="utf-8")
    payload = buffer.getvalue().encode("utf-8")
    if b"NaN" in payload or b"Infinity" in payload: raise FeasibilityContractError("Serializacion contiene no finitos.")
    return payload


def _summary_bytes(summary: Mapping[str, Any]) -> bytes:
    payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if b"NaN" in payload or b"Infinity" in payload: raise FeasibilityContractError("JSON contiene no finitos.")
    return payload + b"\n"


def _fingerprint(summary: Mapping[str, Any], payloads: Sequence[bytes]) -> str:
    copy = dict(summary); copy.pop("publication_fingerprint", None)
    digest = hashlib.sha256(_summary_bytes(copy))
    for payload in payloads: digest.update(payload)
    return digest.hexdigest().upper()


def _summary(counts: Mapping[str, int], attempts: pd.DataFrame, by_state: pd.DataFrame, by_terminal: pd.DataFrame, by_group: pd.DataFrame, consistency: pd.DataFrame, upstream: Mapping[str, str]) -> dict[str, Any]:
    total = by_group.loc[by_group.group_type.eq("total")].iloc[0]
    context_rows = by_group.loc[by_group.group_type.eq("second_serve_context")]
    context_counts = {context: int(context_rows.loc[context_rows.second_serve_context.eq(context), "attempts"].iloc[0]) for context in SECOND_SERVE_CONTEXT_ORDER}
    second_attempts = int(attempts.serve_number.eq(2).sum())
    terminal_counts = {name: int(total[f"{name}_attempts"]) for name in ("winner", "forced_error", "unforced_error")}
    unforced = by_terminal.loc[by_terminal.aggregation_level.eq("unforced_error_code") & by_terminal.serve_number.eq(0)]
    total_consistency = consistency.loc[consistency.group_type.eq("total") & consistency.aggregation_level.eq("terminal_class")]
    return {"analysis_name": ANALYSIS_NAME, "version": ANALYSIS_VERSION, "pattern_id": PATTERN_ID, "analysis_status": "available_descriptive" if int(total.eligible_terminal_attempts) else "available_descriptive_not_comparable", "definition": {"pattern": "documented_immediate_first_return_terminal", "meaning": "terminal_documented_immediately_after_local_first_return", "causal_interpretation": False, "no_rally_reconstruction": True}, "upstream": dict(upstream), "source": {"path": "data/processed/points_enriched.parquet", "columns": list(SOURCE_COLUMNS), "reads": 1}, "temporal_contract": {"development_max_date": "2023-12-31", "validation_folds": [2020, 2021, 2022, 2023], "test_period": "2024-2026"}, "published_scopes": ["total", "serve_number", "surface", "derived_period", "validation_fold", "second_serve_context"], "population": dict(counts), "attempts": {"total": int(len(attempts)), "first": int(attempts.serve_number.eq(1).sum()), "second": second_attempts, "cache_entries": int(attempts.attrs.get("cache_entries", 0))}, "second_serve_context": {"second_attempts": second_attempts, **context_counts, "without_documented_first_fault_proportion": None if second_attempts == 0 else context_counts[SECOND_SERVE_CONTEXT_ORDER[1]] / second_attempts}, "state_counts": {state: int((attempts.classification_state == state).sum()) for state in STATE_ORDER}, "reason_codes": sorted(set(attempts.reason_code)), "terminal_class_counts": terminal_counts, "unforced_error_code_counts": {code: int(unforced.loc[unforced.unforced_error_code.eq(code), "attempts"].iloc[0]) for code in UNFORCED_ERROR_ORDER}, "terminal_coverage": {"eligible_terminal_attempts": int(total.eligible_terminal_attempts), "all_attempts": int(total.attempts), "rate": total.terminal_coverage}, "consistency_audit": {"terminal_attempts": int(total_consistency.attempts.sum()), "discordances": int(total_consistency.discordant.sum()), "interpretation": "diagnostic_only_not_tactical_effectiveness"}, "test_seal": _test_seal(counts), "reconciliations": validate_reconciliations(attempts, by_state, by_terminal, by_group, consistency), "artifact_contracts": {"by_state": BY_STATE_COLUMNS, "by_terminal": BY_TERMINAL_COLUMNS, "by_group": BY_GROUP_COLUMNS, "consistency": CONSISTENCY_COLUMNS, "csv_index": False, "utf8": True, "fixed_order": True, "no_raw_sequences": True}, "fingerprint_contract": {"algorithm": "sha256", "serialization": "utf-8_deterministic", "version": "2"}, "methodological_limits": ["La existencia de second_serve identifica el segundo intento, no una falta inferida.", "El contexto de falta previa es documental y no corrige secuencias ni outcomes.", "Clasificacion exclusivamente desde la secuencia original y el extractor P07.", "No se reconstruyen rallies ni se infiere continuidad valida.", "La concordancia terminal-outcome es diagnostica y no causal; no corrige ninguna fuente.", "El test posterior a 2023 permanece sellado y no se clasifica."]}


def _finalize(summary: dict[str, Any], attempts: pd.DataFrame, by_state: pd.DataFrame, by_terminal: pd.DataFrame, by_group: pd.DataFrame, consistency: pd.DataFrame) -> FeasibilityResult:
    payloads = tuple(_table_bytes(frame) for frame in (by_state, by_terminal, by_group, consistency))
    summary["artifact_payload_sha256"] = dict(zip(("by_state", "by_terminal", "by_group", "consistency"), map(_sha256_bytes, payloads)))
    summary["artifact_payload_bytes"] = dict(zip(("by_state", "by_terminal", "by_group", "consistency"), map(len, payloads)))
    summary["publication_fingerprint"] = _fingerprint(summary, payloads)
    result = FeasibilityResult(summary, by_state, by_terminal, by_group, consistency, attempts, summary["publication_fingerprint"])
    validate_result(result)
    return result


def analyze_points(points: pd.DataFrame, *, expected_population: Mapping[str, int] | None = None, extractor: Callable[..., ReturnTerminalClassification] = parse_and_classify_initial_return_terminal, upstream_contract: Mapping[str, str] | None = None, stage_seconds: dict[str, float] | None = None, wrap_failure_stages: bool = False) -> FeasibilityResult:
    def guarded(stage: str, callback: Callable[[], Any]) -> Any:
        try:
            return callback()
        except _RunFailure:
            raise
        except Exception as error:
            if wrap_failure_stages:
                raise _RunFailure(stage, error) from error
            raise
    clock = time.perf_counter(); source = guarded("source_validation", lambda: validate_source_points(points, expected_rows=None if expected_population is None else expected_population["source_rows_read"]))
    if stage_seconds is not None: stage_seconds["source_validation"] = time.perf_counter() - clock
    clock = time.perf_counter(); development, counts = guarded("metadata_and_test_seal", lambda: split_development_before_parsing(source))
    if stage_seconds is not None: stage_seconds["metadata_and_test_seal"] = time.perf_counter() - clock
    clock = time.perf_counter(); attempts = guarded("attempt_construction_and_classification", lambda: construct_attempts(development, extractor=extractor))
    if stage_seconds is not None: stage_seconds["attempt_construction_and_classification"] = time.perf_counter() - clock
    clock = time.perf_counter(); by_state, by_terminal, by_group, consistency = guarded("aggregation", lambda: (build_by_state(attempts), build_by_terminal(attempts), build_by_group(attempts), build_consistency(attempts)))
    if stage_seconds is not None: stage_seconds["aggregation"] = time.perf_counter() - clock
    if expected_population is not None:
        actual = {**counts, "attempts_total": len(attempts), "first_attempts": int(attempts.serve_number.eq(1).sum()), "second_attempts": int(attempts.serve_number.eq(2).sum())}
        if actual != dict(expected_population):
            if wrap_failure_stages: raise _RunFailure("population_reconciliation", FeasibilityContractError("Cardinalidades reales P07 no reconcilian con upstream."))
            raise FeasibilityContractError("Cardinalidades reales P07 no reconcilian con upstream.")
    upstream = guarded("upstream_validation", validate_upstream_contracts) if upstream_contract is None else dict(upstream_contract)
    required_upstream = {"extractor_commit", "extractor_sha256", "chronological_summary_sha256"}
    if set(upstream) != required_upstream:
        raise FeasibilityContractError("Contrato upstream sintético incompleto.")
    clock = time.perf_counter(); result = guarded("result_validation", lambda: _finalize(_summary(counts, attempts, by_state, by_terminal, by_group, consistency, upstream), attempts, by_state, by_terminal, by_group, consistency))
    if stage_seconds is not None: stage_seconds["result_validation"] = time.perf_counter() - clock
    return result


def validate_result(result: FeasibilityResult) -> None:
    if not isinstance(result, FeasibilityResult): raise TypeError("result debe ser FeasibilityResult.")
    for frame, columns in ((result.by_state, BY_STATE_COLUMNS), (result.by_terminal, BY_TERMINAL_COLUMNS), (result.by_group, BY_GROUP_COLUMNS), (result.consistency, CONSISTENCY_COLUMNS)):
        if list(frame.columns) != columns or any(str(value).startswith("Unnamed") for value in frame.columns):
            raise FeasibilityContractError("Schema de resultado invalido.")
    if result.summary.get("analysis_status") not in {"available_descriptive", "available_descriptive_not_comparable", "not_available"}: raise FeasibilityContractError("analysis_status invalido.")
    if result.summary.get("analysis_status") == "not_available":
        if any(not frame.empty for frame in (result.by_state, result.by_terminal, result.by_group, result.consistency)): raise FeasibilityContractError("not_available no puede publicar agregados.")
        failure, seal = result.summary.get("failure", {}), result.summary.get("test_seal", {})
        if (
            result.summary.get("reason_codes") != ["execution_failed"]
            or set(failure) != {"stage", "type", "message"}
            or not all(isinstance(failure[key], str) and failure[key] for key in failure)
        ):
            raise FeasibilityContractError("Contrato not_available invalido.")
        _validate_test_seal(seal, None)
        return
    _validate_test_seal(result.summary.get("test_seal", {}), result.summary.get("population", {}).get("excluded_test_matches"))
    validate_reconciliations(result.attempts, result.by_state, result.by_terminal, result.by_group, result.consistency)
    payloads = tuple(_table_bytes(frame) for frame in (result.by_state, result.by_terminal, result.by_group, result.consistency))
    names = ("by_state", "by_terminal", "by_group", "consistency")
    if result.summary.get("artifact_payload_sha256") != dict(zip(names, map(_sha256_bytes, payloads))) or result.summary.get("artifact_payload_bytes") != dict(zip(names, map(len, payloads))): raise FeasibilityContractError("Hashes o tamanos no reconcilian.")
    if result.publication_fingerprint != result.summary.get("publication_fingerprint") or result.publication_fingerprint != _fingerprint(result.summary, payloads): raise FeasibilityContractError("Fingerprint invalido.")
    second = result.attempts.loc[result.attempts.serve_number.eq(2)]
    expected_context = {"second_attempts": int(len(second)), **{context: int(second.second_serve_context.eq(context).sum()) for context in SECOND_SERVE_CONTEXT_ORDER}}
    expected_context["without_documented_first_fault_proportion"] = None if not len(second) else expected_context[SECOND_SERVE_CONTEXT_ORDER[1]] / len(second)
    if result.summary.get("second_serve_context") != expected_context:
        raise FeasibilityContractError("Resumen de contexto de segundo saque invalido.")


def serialize_artifacts(result: FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    validate_result(result)
    return (_summary_bytes(result.summary), _table_bytes(result.by_state), _table_bytes(result.by_terminal), _table_bytes(result.by_group), _table_bytes(result.consistency))


def _stage(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return Path(temporary)


def write_artifacts(result: FeasibilityResult, *, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_terminal_path: Path = BY_TERMINAL_PATH, by_group_path: Path = BY_GROUP_PATH, consistency_path: Path = CONSISTENCY_PATH, payloads: tuple[bytes, bytes, bytes, bytes, bytes] | None = None) -> None:
    expected = serialize_artifacts(result)
    if serialize_artifacts(result) != expected: raise FeasibilityContractError("Serializacion no determinista.")
    if payloads is not None and payloads != expected: raise FeasibilityContractError("Payloads no corresponden al resultado.")
    paths, staged, previous = (summary_path, by_state_path, by_terminal_path, by_group_path, consistency_path), [], {}
    try:
        for path, payload in zip(paths, expected): staged.append(_stage(path, payload))
        for path in paths: previous[path] = None if not path.exists() else path.read_bytes()
        for path, temporary in zip(paths, staged): os.replace(temporary, path)
        verify_persisted_artifacts(summary_path=summary_path, by_state_path=by_state_path, by_terminal_path=by_terminal_path, by_group_path=by_group_path, consistency_path=consistency_path)
    except Exception:
        for path, payload in previous.items():
            if payload is None:
                if path.exists(): path.unlink()
            else: path.write_bytes(payload)
        raise
    finally:
        for temporary in staged:
            if temporary.exists(): temporary.unlink()


def _read_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != list(columns): raise FeasibilityContractError("Schema persistido invalido.")
    return frame


def _persisted_frames_are_canonically_ordered(by_state: pd.DataFrame, by_terminal: pd.DataFrame, by_group: pd.DataFrame, consistency: pd.DataFrame) -> bool:
    def ordered(frame: pd.DataFrame, spec_name: str) -> bool:
        keys, ranks = CSV_SORT_SPECS[spec_name]
        observed = [_canonical_sort_key(row, keys, ranks) for _, row in frame.iterrows()]
        return observed == sorted(observed)
    return (
        ordered(by_state, "by_state")
        and ordered(by_terminal, "by_terminal")
        and ordered(by_group, "by_group")
        and ordered(consistency, "consistency")
    )


def _validate_persisted_key_domains(by_state: pd.DataFrame, by_terminal: pd.DataFrame, by_group: pd.DataFrame, consistency: pd.DataFrame) -> None:
    """Cierra las claves contractuales persistidas sin reabrir los puntos."""
    expected_state_groups = {("total", 0, "ALL", "ALL", "ALL")}
    expected_state_groups |= {("serve_number", number, "ALL", "ALL", "ALL") for number in (1, 2)}
    expected_state_groups |= {("surface", 0, surface, "ALL", "ALL") for surface in ALLOWED_SURFACES}
    expected_state_groups |= {("derived_period", 0, "ALL", period, "ALL") for period in PERIOD_ORDER}
    expected_state_groups |= {("validation_fold", 0, "ALL", "ALL", fold) for fold in FOLD_ORDER}
    expected_groups = {(*group, "ALL") for group in expected_state_groups}
    expected_groups |= {("second_serve_context", 2, "ALL", "ALL", "ALL", context) for context in SECOND_SERVE_CONTEXT_ORDER}
    group_columns = ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "second_serve_context"]
    actual_groups = {tuple(row) for row in by_group.loc[:, group_columns].itertuples(index=False, name=None)}
    if actual_groups != expected_groups or len(by_group) != len(expected_groups):
        raise FeasibilityContractError("Claves de grupo persistidas incompletas o adicionales.")
    state_columns = ["group_type", "serve_number", "surface", "derived_period", "validation_fold"]
    expected_state = {(*group, state, reason) for group in expected_state_groups for state, reasons in STATE_REASON_CODES.items() for reason in reasons}
    actual_state = {tuple(row) for row in by_state.loc[:, [*state_columns, "classification_state", "reason_code"]].itertuples(index=False, name=None)}
    if actual_state != expected_state or len(by_state) != len(expected_state):
        raise FeasibilityContractError("Claves by_state persistidas incompletas o adicionales.")
    expected_terminal = {("terminal_class", number, terminal, "") for number in (0, 1, 2) for terminal in TERMINAL_CLASS_ORDER}
    expected_terminal |= {("unforced_error_code", number, "unforced_error", code) for number in (0, 1, 2) for code in UNFORCED_ERROR_ORDER}
    actual_terminal = {(str(row.aggregation_level), int(row.serve_number), str(row.terminal_class), "" if pd.isna(row.unforced_error_code) else str(row.unforced_error_code)) for row in by_terminal.itertuples(index=False)}
    if actual_terminal != expected_terminal or len(by_terminal) != len(expected_terminal):
        raise FeasibilityContractError("Claves by_terminal persistidas incompletas o adicionales.")
    expected_consistency = set()
    for group in expected_state_groups:
        expected_consistency |= {("terminal_class", *group, terminal, "") for terminal in TERMINAL_CLASS_ORDER}
        expected_consistency |= {("unforced_error_code", *group, "unforced_error", code) for code in UNFORCED_ERROR_ORDER}
    actual_consistency = {(str(row.aggregation_level), str(row.group_type), int(row.serve_number), str(row.surface), str(row.derived_period), str(row.validation_fold), str(row.terminal_class), "" if pd.isna(row.unforced_error_code) else str(row.unforced_error_code)) for row in consistency.itertuples(index=False)}
    if actual_consistency != expected_consistency or len(consistency) != len(expected_consistency):
        raise FeasibilityContractError("Claves consistency persistidas incompletas o adicionales.")


def _validate_available_persisted_content(summary: Mapping[str, Any], frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], *, require_canonical_order: bool) -> None:
    by_state, by_terminal, by_group, consistency = frames
    if summary.get("analysis_status") not in {"available_descriptive", "available_descriptive_not_comparable"}: raise FeasibilityContractError("Estado persistido invalido.")
    if any(bool(frame.duplicated().any()) for frame in frames): raise FeasibilityContractError("Filas persistidas duplicadas.")
    if require_canonical_order and not _persisted_frames_are_canonically_ordered(*frames): raise FeasibilityContractError("Orden persistido no canónico.")
    _validate_persisted_key_domains(*frames)
    total_group = by_group.loc[by_group.group_type.eq("total")]
    if len(total_group) != 1 or int(total_group.iloc[0].attempts) != int(summary["attempts"]["total"]): raise FeasibilityContractError("Total persistido invalido.")
    total = total_group.iloc[0]
    population, attempt_summary = summary.get("population", {}), summary.get("attempts", {})
    if (
        int(population.get("development_matches", -1)) != int(total.matches)
        or int(population.get("development_servers", -1)) != int(total.servers)
        or int(attempt_summary.get("first", -1)) != int(by_group.loc[by_group.group_type.eq("serve_number") & by_group.serve_number.eq(1), "attempts"].sum())
        or int(attempt_summary.get("second", -1)) != int(by_group.loc[by_group.group_type.eq("serve_number") & by_group.serve_number.eq(2), "attempts"].sum())
    ):
        raise FeasibilityContractError("Resumen persistido no reconcilia con grupos.")
    context_rows = by_group.loc[by_group.group_type.eq("second_serve_context")]
    if len(context_rows) != len(SECOND_SERVE_CONTEXT_ORDER) or set(context_rows.second_serve_context) != set(SECOND_SERVE_CONTEXT_ORDER): raise FeasibilityContractError("Contextos de segundo saque persistidos invalidos.")
    expected_context = {"second_attempts": int(attempt_summary["second"])}
    expected_context.update({context: int(context_rows.loc[context_rows.second_serve_context.eq(context), "attempts"].iloc[0]) for context in SECOND_SERVE_CONTEXT_ORDER})
    expected_context["without_documented_first_fault_proportion"] = None if expected_context["second_attempts"] == 0 else expected_context[SECOND_SERVE_CONTEXT_ORDER[1]] / expected_context["second_attempts"]
    if summary.get("second_serve_context") != expected_context: raise FeasibilityContractError("Resumen persistido de contexto de segundo saque invalido.")
    state_total = by_state.loc[by_state.group_type.eq("total")].groupby("classification_state", sort=False).attempts.sum().to_dict()
    expected_state_counts = {state: int(state_total.get(state, 0)) for state in STATE_ORDER}
    if summary.get("state_counts") != expected_state_counts: raise FeasibilityContractError("Resumen persistido no reconcilia con estados.")
    terminal_total = by_terminal.loc[(by_terminal.aggregation_level.eq("terminal_class")) & (by_terminal.serve_number.eq(0))]
    expected_terminal_counts = {terminal: int(terminal_total.loc[terminal_total.terminal_class.eq(terminal), "attempts"].sum()) for terminal in TERMINAL_CLASS_ORDER}
    if summary.get("terminal_class_counts") != expected_terminal_counts or int(total.eligible_terminal_attempts) != sum(expected_terminal_counts.values()): raise FeasibilityContractError("Resumen persistido no reconcilia con terminales.")
    coverage = summary.get("terminal_coverage", {})
    expected_rate = None if int(total.attempts) == 0 else int(total.eligible_terminal_attempts) / int(total.attempts)
    if coverage != {"eligible_terminal_attempts": int(total.eligible_terminal_attempts), "all_attempts": int(total.attempts), "rate": expected_rate}: raise FeasibilityContractError("Cobertura terminal persistida invalida.")
    if not consistency.concordant.add(consistency.discordant).eq(consistency.attempts).all() or not consistency.returner_wins.le(consistency.attempts).all(): raise FeasibilityContractError("Consistencia terminal-outcome persistida invalida.")
    _validate_test_seal(summary.get("test_seal", {}), population.get("excluded_test_matches"))


def verify_persisted_artifacts(*, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_terminal_path: Path = BY_TERMINAL_PATH, by_group_path: Path = BY_GROUP_PATH, consistency_path: Path = CONSISTENCY_PATH) -> None:
    paths = (summary_path, by_state_path, by_terminal_path, by_group_path, consistency_path)
    if any(not path.exists() for path in paths): raise FeasibilityContractError("Faltan artefactos persistidos.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    legacy_not_available = summary.get("analysis_status") == "not_available" and summary.get("version") == "1.0.0"
    by_group_columns = LEGACY_BY_GROUP_COLUMNS if legacy_not_available else BY_GROUP_COLUMNS
    frames = (_read_csv(by_state_path, BY_STATE_COLUMNS), _read_csv(by_terminal_path, BY_TERMINAL_COLUMNS), _read_csv(by_group_path, by_group_columns), _read_csv(consistency_path, CONSISTENCY_COLUMNS))
    payloads = tuple(path.read_bytes() for path in paths[1:])
    names = ("by_state", "by_terminal", "by_group", "consistency")
    if summary.get("artifact_payload_sha256") != dict(zip(names, map(_sha256_bytes, payloads))) or summary.get("artifact_payload_bytes") != dict(zip(names, map(len, payloads))) or summary.get("publication_fingerprint") != _fingerprint(summary, payloads): raise FeasibilityContractError("Integridad persistida invalida.")
    if summary.get("analysis_status") == "not_available":
        if any(not frame.empty for frame in frames): raise FeasibilityContractError("not_available persistido publica filas.")
        seal, failure = summary.get("test_seal", {}), summary.get("failure", {})
        if (
            summary.get("reason_codes") != ["execution_failed"]
            or set(failure) != {"stage", "type", "message"}
        ):
            raise FeasibilityContractError("Contrato not_available persistido invalido.")
        _validate_test_seal(seal, None)
        return
    if summary.get("analysis_status") not in {"available_descriptive", "available_descriptive_not_comparable"}: raise FeasibilityContractError("Estado persistido invalido.")
    if any(bool(frame.duplicated().any()) for frame in frames): raise FeasibilityContractError("Filas persistidas duplicadas.")
    if not _persisted_frames_are_canonically_ordered(*frames): raise FeasibilityContractError("Orden persistido no canónico.")
    _validate_persisted_key_domains(*frames)
    total_group = frames[2].loc[frames[2].group_type.eq("total")]
    if len(total_group) != 1 or int(total_group.iloc[0].attempts) != int(summary["attempts"]["total"]): raise FeasibilityContractError("Total persistido invalido.")
    total = total_group.iloc[0]
    population, attempt_summary = summary.get("population", {}), summary.get("attempts", {})
    if (
        int(population.get("development_matches", -1)) != int(total.matches)
        or int(population.get("development_servers", -1)) != int(total.servers)
        or int(attempt_summary.get("first", -1)) != int(frames[2].loc[frames[2].group_type.eq("serve_number") & frames[2].serve_number.eq(1), "attempts"].sum())
        or int(attempt_summary.get("second", -1)) != int(frames[2].loc[frames[2].group_type.eq("serve_number") & frames[2].serve_number.eq(2), "attempts"].sum())
    ):
        raise FeasibilityContractError("Resumen persistido no reconcilia con grupos.")
    context_rows = frames[2].loc[frames[2].group_type.eq("second_serve_context")]
    if len(context_rows) != len(SECOND_SERVE_CONTEXT_ORDER) or set(context_rows.second_serve_context) != set(SECOND_SERVE_CONTEXT_ORDER):
        raise FeasibilityContractError("Contextos de segundo saque persistidos invalidos.")
    expected_context = {"second_attempts": int(attempt_summary["second"])}
    expected_context.update({context: int(context_rows.loc[context_rows.second_serve_context.eq(context), "attempts"].iloc[0]) for context in SECOND_SERVE_CONTEXT_ORDER})
    expected_context["without_documented_first_fault_proportion"] = None if expected_context["second_attempts"] == 0 else expected_context[SECOND_SERVE_CONTEXT_ORDER[1]] / expected_context["second_attempts"]
    if summary.get("second_serve_context") != expected_context:
        raise FeasibilityContractError("Resumen persistido de contexto de segundo saque invalido.")
    state_total = frames[0].loc[frames[0].group_type.eq("total")].groupby("classification_state", sort=False).attempts.sum().to_dict()
    expected_state_counts = {state: int(state_total.get(state, 0)) for state in STATE_ORDER}
    if summary.get("state_counts") != expected_state_counts:
        raise FeasibilityContractError("Resumen persistido no reconcilia con estados.")
    terminal_total = frames[1].loc[(frames[1].aggregation_level.eq("terminal_class")) & (frames[1].serve_number.eq(0))]
    expected_terminal_counts = {terminal: int(terminal_total.loc[terminal_total.terminal_class.eq(terminal), "attempts"].sum()) for terminal in TERMINAL_CLASS_ORDER}
    if summary.get("terminal_class_counts") != expected_terminal_counts or int(total.eligible_terminal_attempts) != sum(expected_terminal_counts.values()):
        raise FeasibilityContractError("Resumen persistido no reconcilia con terminales.")
    coverage = summary.get("terminal_coverage", {})
    expected_rate = None if int(total.attempts) == 0 else int(total.eligible_terminal_attempts) / int(total.attempts)
    if coverage != {"eligible_terminal_attempts": int(total.eligible_terminal_attempts), "all_attempts": int(total.attempts), "rate": expected_rate}:
        raise FeasibilityContractError("Cobertura terminal persistida invalida.")
    if not frames[3].concordant.add(frames[3].discordant).eq(frames[3].attempts).all() or not frames[3].returner_wins.le(frames[3].attempts).all():
        raise FeasibilityContractError("Consistencia terminal-outcome persistida invalida.")
    _validate_test_seal(summary.get("test_seal", {}), population.get("excluded_test_matches"))
    repair = summary.get("artifact_repair")
    if repair is not None and repair != {
        "mode": "artifact_only_canonical_reserialization",
        "real_analysis_rerun": False,
        "source_publication_fingerprint": LEGACY_REPAIR_INPUT_FINGERPRINT,
        "historical_real_analysis_attempts": 2,
        "historical_run_2_status": "publication_verification_failed",
        "reordered_artifacts": ["by_state", "by_group", "consistency"],
        "analytical_values_changed": False,
        "reason": "canonical_surface_order",
    }:
        raise FeasibilityContractError("Provenance de reparacion artifact-only invalida.")


def _repair_paths(*, summary_path: Path | None, by_state_path: Path | None, by_terminal_path: Path | None, by_group_path: Path | None, consistency_path: Path | None) -> dict[str, Path]:
    paths = {
        "summary": SUMMARY_PATH if summary_path is None else Path(summary_path),
        "by_state": BY_STATE_PATH if by_state_path is None else Path(by_state_path),
        "by_terminal": BY_TERMINAL_PATH if by_terminal_path is None else Path(by_terminal_path),
        "by_group": BY_GROUP_PATH if by_group_path is None else Path(by_group_path),
        "consistency": CONSISTENCY_PATH if consistency_path is None else Path(consistency_path),
    }
    expected = {"summary": SUMMARY_PATH, "by_state": BY_STATE_PATH, "by_terminal": BY_TERMINAL_PATH, "by_group": BY_GROUP_PATH, "consistency": CONSISTENCY_PATH}
    if len({path.resolve() for path in paths.values()}) != len(paths): raise FeasibilityContractError("Rutas de reparacion duplicadas.")
    for name, path in paths.items():
        try: path.resolve().relative_to(REPORTS_DIR.resolve())
        except ValueError as error: raise FeasibilityContractError("Ruta de reparacion fuera de reports/.") from error
        if path.resolve() != expected[name].resolve(): raise FeasibilityContractError("La reparacion solo acepta las cinco rutas P07 publicadas.")
    return paths


def _raw_csv(payload: bytes) -> tuple[list[str], list[list[str]]]:
    try: rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as error: raise FeasibilityContractError("CSV persistido no legible como UTF-8 contractual.") from error
    if not rows: raise FeasibilityContractError("CSV persistido vacio.")
    return rows[0], rows[1:]


def _raw_csv_payload(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(header); writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _raw_row_multiset_hash(rows: Sequence[Sequence[str]]) -> str:
    encoded = sorted(b"\x1f".join(field.encode("utf-8") for field in row) for row in rows)
    return hashlib.sha256(b"\x1e".join(encoded)).hexdigest().upper()


def _canonical_csv_payload(name: str, payload: bytes) -> bytes:
    header, rows = _raw_csv(payload)
    expected_columns = {"by_state": BY_STATE_COLUMNS, "by_terminal": BY_TERMINAL_COLUMNS, "by_group": BY_GROUP_COLUMNS, "consistency": CONSISTENCY_COLUMNS}[name]
    if header != expected_columns or any(len(row) != len(header) for row in rows): raise FeasibilityContractError("Schema CSV legacy invalido para reparacion.")
    if name == "by_terminal": return payload
    keys, ranks = CSV_SORT_SPECS[name]
    ordered = sorted(rows, key=lambda row: _canonical_sort_key(dict(zip(header, row)), keys, ranks))
    candidate = _raw_csv_payload(header, ordered)
    candidate_header, candidate_rows = _raw_csv(candidate)
    if candidate_header != header or len(candidate_rows) != len(rows) or _raw_row_multiset_hash(candidate_rows) != _raw_row_multiset_hash(rows): raise FeasibilityContractError("Reserializacion CSV altera contenido literal.")
    return candidate


def _analytical_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"artifact_payload_sha256", "artifact_payload_bytes", "publication_fingerprint", "artifact_repair", "fingerprint_contract"}
    return {key: value for key, value in summary.items() if key not in excluded}


def _validate_legacy_repair_input(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, bytes], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
    if hashes != LEGACY_REPAIR_INPUT_SHA256: raise FeasibilityContractError("Los hashes legacy no coinciden; reparacion bloqueada.")
    summary = json.loads(payloads["summary"].decode("utf-8"))
    if summary.get("publication_fingerprint") != LEGACY_REPAIR_INPUT_FINGERPRINT or "artifact_repair" in summary: raise FeasibilityContractError("Fingerprint legacy o estado de migracion incompatibles.")
    table_payloads = tuple(payloads[name] for name in ("by_state", "by_terminal", "by_group", "consistency"))
    names = ("by_state", "by_terminal", "by_group", "consistency")
    if summary.get("artifact_payload_sha256") != dict(zip(names, map(_sha256_bytes, table_payloads))) or summary.get("artifact_payload_bytes") != dict(zip(names, map(len, table_payloads))) or summary.get("publication_fingerprint") != _fingerprint(summary, table_payloads): raise FeasibilityContractError("Integridad legacy incompatible.")
    frames = (_read_csv(paths["by_state"], BY_STATE_COLUMNS), _read_csv(paths["by_terminal"], BY_TERMINAL_COLUMNS), _read_csv(paths["by_group"], BY_GROUP_COLUMNS), _read_csv(paths["consistency"], CONSISTENCY_COLUMNS))
    _validate_available_persisted_content(summary, frames, require_canonical_order=False)
    for frame in (frames[0], frames[2], frames[3]):
        observed = list(frame.loc[frame.group_type.eq("surface"), "surface"].drop_duplicates())
        if observed != ["Clay", "Grass", "Hard"]: raise FeasibilityContractError("El input no presenta exactamente el orden legacy de superficies.")
    return summary, payloads, frames


def reserialize_persisted_artifacts_canonically(*, summary_path: Path | None = None, by_state_path: Path | None = None, by_terminal_path: Path | None = None, by_group_path: Path | None = None, consistency_path: Path | None = None) -> dict[str, Any]:
    """Migra una sola vez el snapshot P07 legacy sin consultar datos fuente."""
    paths = _repair_paths(summary_path=summary_path, by_state_path=by_state_path, by_terminal_path=by_terminal_path, by_group_path=by_group_path, consistency_path=consistency_path)
    legacy_summary, legacy_payloads, _ = _validate_legacy_repair_input(paths)
    candidate_payloads = {
        "by_state": _canonical_csv_payload("by_state", legacy_payloads["by_state"]),
        "by_terminal": _canonical_csv_payload("by_terminal", legacy_payloads["by_terminal"]),
        "by_group": _canonical_csv_payload("by_group", legacy_payloads["by_group"]),
        "consistency": _canonical_csv_payload("consistency", legacy_payloads["consistency"]),
    }
    if candidate_payloads["by_terminal"] != legacy_payloads["by_terminal"]: raise FeasibilityContractError("by_terminal no puede cambiar en reparacion artifact-only.")
    candidate_summary = dict(legacy_summary)
    candidate_summary["artifact_repair"] = {"mode": "artifact_only_canonical_reserialization", "real_analysis_rerun": False, "source_publication_fingerprint": LEGACY_REPAIR_INPUT_FINGERPRINT, "historical_real_analysis_attempts": 2, "historical_run_2_status": "publication_verification_failed", "reordered_artifacts": ["by_state", "by_group", "consistency"], "analytical_values_changed": False, "reason": "canonical_surface_order"}
    candidate_summary["fingerprint_contract"] = {**candidate_summary["fingerprint_contract"], "version": "3"}
    ordered_payloads = tuple(candidate_payloads[name] for name in ("by_state", "by_terminal", "by_group", "consistency"))
    candidate_summary["artifact_payload_sha256"] = dict(zip(("by_state", "by_terminal", "by_group", "consistency"), map(_sha256_bytes, ordered_payloads)))
    candidate_summary["artifact_payload_bytes"] = dict(zip(("by_state", "by_terminal", "by_group", "consistency"), map(len, ordered_payloads)))
    candidate_summary["publication_fingerprint"] = _fingerprint(candidate_summary, ordered_payloads)
    if _analytical_summary(candidate_summary) != _analytical_summary(legacy_summary): raise FeasibilityContractError("La reparacion altera contenido analitico del summary.")
    candidate_frames = tuple(pd.read_csv(io.StringIO(payload.decode("utf-8")), keep_default_na=False) for payload in ordered_payloads)
    _validate_available_persisted_content(candidate_summary, candidate_frames, require_canonical_order=True)
    final_payloads = (_summary_bytes(candidate_summary), *ordered_payloads)
    ordered_paths = tuple(paths[name] for name in ("summary", "by_state", "by_terminal", "by_group", "consistency"))
    staged: list[Path] = []; previous = dict(legacy_payloads)
    try:
        for path, payload in zip(ordered_paths, final_payloads): staged.append(_stage(path, payload))
        for path, temporary in zip(ordered_paths, staged): os.replace(temporary, path)
        verify_persisted_artifacts(summary_path=paths["summary"], by_state_path=paths["by_state"], by_terminal_path=paths["by_terminal"], by_group_path=paths["by_group"], consistency_path=paths["consistency"])
    except Exception:
        for name, path in paths.items(): path.write_bytes(previous[name])
        raise
    finally:
        for temporary in staged:
            if temporary.exists(): temporary.unlink()
    return {"publication_fingerprint": candidate_summary["publication_fingerprint"], "reordered_artifacts": candidate_summary["artifact_repair"]["reordered_artifacts"], "by_terminal_unchanged": True}


def read_source_points() -> pd.DataFrame:
    return pd.read_parquet(POINTS_PATH, columns=list(SOURCE_COLUMNS))


def run_real_analysis(*, stage_seconds: dict[str, float] | None = None) -> FeasibilityResult:
    clock = time.perf_counter()
    try: upstream = validate_upstream_contracts()
    except Exception as error: raise _RunFailure("upstream_validation", error) from error
    if stage_seconds is not None: stage_seconds["upstream_validation"] = time.perf_counter() - clock
    clock = time.perf_counter()
    try: source = read_source_points()
    except Exception as error: raise _RunFailure("read_source_points", error) from error
    if stage_seconds is not None: stage_seconds["read_source_points"] = time.perf_counter() - clock
    try: return analyze_points(source, expected_population=EXPECTED_REAL, upstream_contract=upstream, stage_seconds=stage_seconds, wrap_failure_stages=True)
    except _RunFailure: raise
    except Exception as error: raise _RunFailure("analyze_points", error) from error


def _performance_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    try: path.relative_to(ROOT)
    except ValueError: return path
    raise FeasibilityContractError("--performance-log debe estar fuera del repositorio.")


def _sanitized_message(error: Exception) -> str:
    return str(error).replace(str(ROOT), "<repo>").replace("\n", " ").strip()[:240] or type(error).__name__


def _not_available_result(stage: str, error: Exception) -> FeasibilityResult:
    empty = tuple(pd.DataFrame(columns=columns) for columns in (BY_STATE_COLUMNS, BY_TERMINAL_COLUMNS, BY_GROUP_COLUMNS, CONSISTENCY_COLUMNS))
    summary = {"analysis_name": ANALYSIS_NAME, "version": ANALYSIS_VERSION, "pattern_id": PATTERN_ID, "analysis_status": "not_available", "reason_codes": ["execution_failed"], "failure": {"stage": stage, "type": type(error).__name__, "message": _sanitized_message(error)}, "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": None, **{field: 0 for field in TEST_ZERO_FIELDS}}, "artifact_contracts": {"by_state": BY_STATE_COLUMNS, "by_terminal": BY_TERMINAL_COLUMNS, "by_group": BY_GROUP_COLUMNS, "consistency": CONSISTENCY_COLUMNS}}
    payloads = tuple(_table_bytes(frame) for frame in empty)
    summary["artifact_payload_sha256"] = dict(zip(("by_state", "by_terminal", "by_group", "consistency"), map(_sha256_bytes, payloads)))
    summary["artifact_payload_bytes"] = dict(zip(("by_state", "by_terminal", "by_group", "consistency"), map(len, payloads)))
    summary["publication_fingerprint"] = _fingerprint(summary, payloads)
    return FeasibilityResult(summary, *empty, pd.DataFrame(), summary["publication_fingerprint"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P07: terminador documentado inmediatamente del primer resto.")
    parser.add_argument("--performance-log", required=True); args = parser.parse_args(argv)
    performance = _performance_path(args.performance_log); started = time.perf_counter(); error: Exception | None = None; failure_stage: str | None = None; stages: dict[str, float] = {}; result: FeasibilityResult | None = None
    try:
        try:
            result = run_real_analysis(stage_seconds=stages)
        except _RunFailure as failure:
            error, failure_stage, result = failure.cause, failure.stage, _not_available_result(failure.stage, failure.cause)
        except Exception as failure:
            error, failure_stage, result = failure, "run_real_analysis", _not_available_result("run_real_analysis", failure)
        try:
            write_artifacts(result, summary_path=SUMMARY_PATH, by_state_path=BY_STATE_PATH, by_terminal_path=BY_TERMINAL_PATH, by_group_path=BY_GROUP_PATH, consistency_path=CONSISTENCY_PATH)
        except Exception as failure:
            error, failure_stage = failure, "artifact_publication"
    finally:
        performance.parent.mkdir(parents=True, exist_ok=True)
        status = "publication_failed" if failure_stage == "artifact_publication" else ("not_available" if result is None else result.summary["analysis_status"])
        payload: dict[str, Any] = {"analysis_name": ANALYSIS_NAME, "analysis_status": status, "duration_seconds": time.perf_counter() - started, "stage_seconds": stages}
        if error is not None: payload["failure"] = {"stage": failure_stage, "type": type(error).__name__, "message": _sanitized_message(error)}
        performance.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    if error is not None: raise SystemExit(1)


if __name__ == "__main__": main()
