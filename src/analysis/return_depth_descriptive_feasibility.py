"""P05: infraestructura descriptiva sellada para profundidad del primer resto.

Este modulo no se ejecuta al importarlo. Su CLI futuro lee una sola vez la
fuente de puntos, excluye el test antes de clasificar y publica cinco
artefactos de forma transaccional. Las funciones puras aceptan DataFrames
sinteticos para poder probar el contrato sin datos locales.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.return_depth_feasibility import (
    RETURN_DEPTHS,
    ReturnDepthClassification,
    ReturnDepthReason,
    ReturnDepthState,
    parse_and_classify_initial_return_depth,
    validate_return_depth_classification,
)

ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "return_depth_descriptive_feasibility_summary.json"
BY_STATE_PATH = TABLES_DIR / "return_depth_descriptive_feasibility_by_state.csv"
BY_DEPTH_PATH = TABLES_DIR / "return_depth_descriptive_feasibility_by_depth.csv"
BY_GROUP_PATH = TABLES_DIR / "return_depth_descriptive_feasibility_by_group.csv"
OUTCOMES_PATH = TABLES_DIR / "return_depth_descriptive_feasibility_outcomes.csv"

ANALYSIS_NAME = "return_depth_descriptive_feasibility"
ANALYSIS_VERSION = "1.0.0"
UPSTREAM_COMMIT = "33ee946"
CUTOFF = pd.Timestamp("2023-12-31")
ALLOWED_SURFACES = ("Hard", "Clay", "Grass")
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
FOLD_ORDER = ("pre_validation", "validation_2020", "validation_2021", "validation_2022", "validation_2023")
STATE_ORDER = tuple(state.value for state in ReturnDepthState)
DEPTH_ORDER = ("7", "8", "9")
STATE_REASON_CODES = {
    ReturnDepthState.OBSERVED.value: {ReturnDepthReason.DEPTH_7.value, ReturnDepthReason.DEPTH_8.value, ReturnDepthReason.DEPTH_9.value},
    ReturnDepthState.UNKNOWN.value: {ReturnDepthReason.DEPTH_0.value},
    ReturnDepthState.NOT_DOCUMENTED.value: {ReturnDepthReason.NO_DEPTH.value},
    ReturnDepthState.UNKNOWN_INITIAL.value: {ReturnDepthReason.MISSING_PREFIX.value, ReturnDepthReason.BOUNDARY.value, ReturnDepthReason.MODIFIER.value, ReturnDepthReason.UNKNOWN_SHOT.value, ReturnDepthReason.TRUNCATED.value, ReturnDepthReason.INCONSISTENT.value},
    ReturnDepthState.CENSORED.value: {ReturnDepthReason.ACE.value, ReturnDepthReason.UNRETURNED.value, ReturnDepthReason.FAULT.value, ReturnDepthReason.DOUBLE_FAULT.value, ReturnDepthReason.SPECIAL.value, ReturnDepthReason.INCOMPLETE_LET.value},
}
SOURCE_COLUMNS = ("match_id", "point_number", "date", "surface", "server", "point_winner", "player_1", "player_2", "first_serve", "second_serve")
EXPECTED_REAL = {"source_rows_read": 1_280_408, "source_matches": 7_524, "source_players": 1_002, "development_point_rows": 1_035_760, "development_matches": 5_993, "development_servers": 870, "excluded_test_target_matches": 1_531, "attempts_total": 1_426_863, "first_attempts": 1_035_760, "second_attempts": 391_103}
PUBLISHED_ARTIFACT_CONTRACT = {
    "summary": (5_041, "8DA410F74B226DCEDB3649A3C4FB90B8497244B60BF7C4EE2DBB61A2C0D6B567"),
    "by_state": (4_356, "E1850F03424CB1EBF26F25E6B297A94199D53C5AD68A0D6D1AA9D2A205F6ABA5"),
    "by_depth": (1_210, "DE07B44C99873FFEEF963CC5D46C4C3E880109748BEBBC972510BAE2507966AD"),
    "by_group": (1_854, "8FF83030223E26AA794BB790FF8301F4CFE76CB2CBB68B719E354014A7560A30"),
    "outcomes": (6_397, "62AA53002E99FC7AF11F1C0A7A4BA9DF6D4E81706F57CAC0F3A528A2CF3E6384"),
}
PUBLISHED_PUBLICATION_FINGERPRINT = "96333493F586D2F65183A12F6443ACF7F1DC2D2999B96C58AC6C07B27DD1C06B"
TEST_ZERO_FIELDS = ("test_target_rows_parsed", "test_attempts_constructed", "test_returns_classified", "test_outcomes_computed", "test_rows_evaluated", "test_matches_evaluated", "test_scores_computed", "test_evaluation_runs", "test_recommendations_generated")

BY_STATE_COLUMNS = ["group_type", "serve_number", "classification_state", "reason_code", "attempts", "matches", "servers", "returners", "denominator_attempts", "proportion_of_denominator"]
BY_DEPTH_COLUMNS = ["group_type", "serve_number", "depth_code", "depth_description", "attempts", "matches", "servers", "returners", "observed_depth_denominator", "proportion_of_observed_depths", "end_to_end_denominator", "end_to_end_coverage"]
BY_GROUP_COLUMNS = ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "attempts", "matches", "servers", "returners", "observed_depth_attempts", "unknown_depth_attempts", "not_documented_attempts", "unknown_initial_attempts", "censored_attempts", "observed_depth_numerator", "observed_depth_denominator", "observed_depth_coverage"]
OUTCOME_COLUMNS = ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "depth_code", "depth_description", "attempts", "matches", "servers", "returners", "returner_point_wins", "returner_point_losses", "returner_point_win_rate", "wilson_95_lower", "wilson_95_upper"]


class FeasibilityContractError(ValueError):
    """Violacion cerrada del contrato P05."""


@dataclass(frozen=True)
class DescriptiveResult:
    summary: dict[str, Any]
    by_state: pd.DataFrame
    by_depth: pd.DataFrame
    by_group: pd.DataFrame
    outcomes: pd.DataFrame
    attempts: pd.DataFrame
    publication_fingerprint: str


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value not in (1, 2):
        raise FeasibilityContractError(f"{field} debe ser exactamente el entero 1 o 2; recibido {value!r}.")
    return int(value)


def _text(series: pd.Series, field: str) -> None:
    bad = series.map(lambda value: not isinstance(value, str) or not value or value != value.strip())
    if bool(bad.any()):
        raise FeasibilityContractError(f"{field} debe ser texto no vacio y sin espacios externos.")


def _dates(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise FeasibilityContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif bool(series.map(lambda value: isinstance(value, str)).all()):
        if not bool(series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()):
            raise FeasibilityContractError("date debe usar YYYY-MM-DD sin ambiguedad.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif bool(series.map(lambda value: isinstance(value, (pd.Timestamp, datetime, date)) and not isinstance(value, bool)).all()):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        raise FeasibilityContractError("date contiene tipos ambiguos.")
    if parsed.isna().any() or getattr(parsed.dt, "tz", None) is not None:
        raise FeasibilityContractError("date invalida o con zona horaria.")
    return parsed.dt.normalize()


def _presence(value: object, field: str) -> str:
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return "null"
    if not isinstance(value, str):
        raise FeasibilityContractError(f"{field} debe ser texto o nulo.")
    if value == "": return "empty"
    if value.isspace(): return "whitespace_only"
    return "substantive"


def derive_period(day: pd.Timestamp) -> str:
    if day.year <= 2009: return "to_2009"
    if day.year <= 2019: return "2010s"
    return "2020s"


def validation_fold(day: pd.Timestamp) -> str:
    if day.year <= 2019: return "pre_validation"
    value = f"validation_{day.year}"
    if value not in FOLD_ORDER:
        raise FeasibilityContractError("Fecha de desarrollo fuera de los folds aprobados.")
    return value


def validate_source_points(points: pd.DataFrame, *, expected_rows: int | None = None) -> pd.DataFrame:
    """Valida la fuente en memoria antes del sellado temporal."""
    if not isinstance(points, pd.DataFrame): raise TypeError("points debe ser DataFrame.")
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing: raise FeasibilityContractError(f"Faltan columnas requeridas: {missing}")
    if expected_rows is not None and len(points) != expected_rows: raise FeasibilityContractError(f"source_rows_read inesperado: {len(points)}")
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"): _text(work[field], field)
    if work["point_number"].isna().any() or bool(work.duplicated(["match_id", "point_number"]).any()):
        raise FeasibilityContractError("(match_id, point_number) debe ser clave unica y no nula.")
    if bool(work["point_number"].map(lambda value: isinstance(value, bool) or (isinstance(value, str) and (not value or value != value.strip()))).any()):
        raise FeasibilityContractError("point_number invalido.")
    work["date"] = _dates(work["date"])
    unexpected = sorted(set(work.loc[~work.surface.isin(ALLOWED_SURFACES), "surface"]))
    if unexpected: raise FeasibilityContractError(f"Superficies inesperadas: {unexpected}")
    if bool(work.player_1.eq(work.player_2).any()): raise FeasibilityContractError("player_1 y player_2 deben ser distintos.")
    for field in ("server", "point_winner"): work[field] = work[field].map(lambda value: _strict_index(value, field))
    metadata = ["date", "surface", "player_1", "player_2"]
    if bool(work.groupby("match_id", sort=False)[metadata].nunique(dropna=False).gt(1).any().any()):
        raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    return work.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)


def split_development_before_parsing(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    development = source.loc[source.date.le(CUTOFF)].copy()
    sealed = source.loc[source.date.gt(CUTOFF)]
    if development.empty or sealed.empty or bool(development.date.gt(CUTOFF).any()):
        raise FeasibilityContractError("El sellado temporal no particiona la fuente.")
    counts = {"source_rows_read": int(len(source)), "source_matches": int(source.match_id.nunique()), "source_players": int(len(set(source.player_1) | set(source.player_2))), "development_point_rows": int(len(development)), "development_matches": int(development.match_id.nunique()), "development_servers": int(pd.Series(np.where(development.server.eq(1), development.player_1, development.player_2)).nunique()), "excluded_test_target_matches": int(sealed.match_id.nunique())}
    return development.reset_index(drop=True), counts


def _attempt_row(row: Any, text: str, number: int, prior_fault: bool, classified: ReturnDepthClassification) -> dict[str, Any]:
    validate_return_depth_classification(classified)
    server_player = row.player_1 if row.server == 1 else row.player_2
    returner_player = row.player_2 if row.server == 1 else row.player_1
    return {"match_id": row.match_id, "point_number": row.point_number, "serve_number": number, "date": row.date, "surface": row.surface, "derived_period": derive_period(row.date), "validation_fold": validation_fold(row.date), "server_player": server_player, "returner_player": returner_player, "sequence_text": text, "previous_attempt_was_fault": prior_fault, "returner_won_point": bool(row.point_winner != row.server), "classification_state": classified.classification_state.value, "reason_code": classified.reason_codes[0].value, "eligible_for_depth_comparison": classified.eligible_for_depth_comparison, "return_event_observed": classified.return_event_observed, "actor": classified.actor, "return_shot_type": classified.return_shot_type, "lateral_direction_code": classified.lateral_direction_code, "return_depth_code": classified.return_depth_code, "return_depth_label": classified.return_depth_label, "terminal_serve_outcome": classified.terminal_serve_outcome, "parser_warning_codes": classified.parser_warning_codes, "parser_warning_spans": classified.parser_warning_spans, "residual_spans": classified.residual_spans}


def construct_attempts(development: pd.DataFrame, *, extractor: Callable[..., ReturnDepthClassification] = parse_and_classify_initial_return_depth) -> pd.DataFrame:
    """Construye intentos y reutiliza la clasificacion por texto, saque y contexto."""
    cache: dict[tuple[str, int, bool], ReturnDepthClassification] = {}
    def classify(text: str, number: int, prior: bool) -> ReturnDepthClassification:
        key = (text, number, prior)
        if key not in cache: cache[key] = extractor(text, number, previous_attempt_was_fault=prior)
        return cache[key]
    rows: list[dict[str, Any]] = []
    for row in development.itertuples(index=False):
        if _presence(row.first_serve, "first_serve") != "substantive":
            raise FeasibilityContractError("first_serve debe ser sustantivo para cada punto de desarrollo.")
        first = classify(row.first_serve, 1, False)
        rows.append(_attempt_row(row, row.first_serve, 1, False, first))
        if _presence(row.second_serve, "second_serve") == "substantive":
            prior = first.terminal_serve_outcome == "service_fault"
            second = classify(row.second_serve, 2, prior)
            rows.append(_attempt_row(row, row.second_serve, 2, prior, second))
    attempts = pd.DataFrame(rows)
    if attempts.empty or bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()):
        raise FeasibilityContractError("Intentos vacios o clave de intento duplicada.")
    attempts = attempts.sort_values(["date", "match_id", "point_number", "serve_number"], kind="stable").reset_index(drop=True)
    attempts.attrs["cache_entries"] = len(cache)
    return attempts


def _counts(part: pd.DataFrame) -> dict[str, int]:
    return {"attempts": int(len(part)), "matches": int(part.match_id.nunique()), "servers": int(part.server_player.nunique()), "returners": int(part.returner_player.nunique())}


def _ordered(frame: pd.DataFrame, keys: Sequence[str], ranks: Mapping[str, Mapping[Any, int]] | None = None) -> pd.DataFrame:
    work = frame.copy()
    ranks = ranks or {}
    auxiliary = []
    for key in keys:
        if key in ranks:
            name = f"__{key}"; work[name] = work[key].map(ranks[key]); auxiliary.append(name)
    return work.sort_values([*auxiliary, *keys], kind="stable").drop(columns=auxiliary).reset_index(drop=True)


def _state_rows(attempts: pd.DataFrame, group_type: str, number: int, part: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (state, reason), group in part.groupby(["classification_state", "reason_code"], sort=False, observed=True):
        rows.append({"group_type": group_type, "serve_number": number, "classification_state": state, "reason_code": reason, **_counts(group), "denominator_attempts": int(len(part)), "proportion_of_denominator": len(group) / len(part)})
    return rows


def build_by_state(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = _state_rows(attempts, "total", 0, attempts)
    for number in (1, 2):
        part = attempts.loc[attempts.serve_number.eq(number)]
        if not part.empty: rows.extend(_state_rows(attempts, "serve_number", number, part))
    ranks = {"group_type": {"total": 0, "serve_number": 1}, "classification_state": {value: index for index, value in enumerate(STATE_ORDER)}}
    return _ordered(pd.DataFrame(rows, columns=BY_STATE_COLUMNS), ["group_type", "serve_number", "classification_state", "reason_code"], ranks)


def _depth_rows(part: pd.DataFrame, group_type: str, number: int) -> list[dict[str, Any]]:
    observed = part.loc[part.classification_state.eq(ReturnDepthState.OBSERVED.value)]
    rows = []
    for code in DEPTH_ORDER:
        group = observed.loc[observed.return_depth_code.eq(code)]
        rows.append({"group_type": group_type, "serve_number": number, "depth_code": code, "depth_description": RETURN_DEPTHS[code], **_counts(group), "observed_depth_denominator": int(len(observed)), "proportion_of_observed_depths": None if observed.empty else len(group) / len(observed), "end_to_end_denominator": int(len(part)), "end_to_end_coverage": None if part.empty else len(group) / len(part)})
    return rows


def build_by_depth(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = _depth_rows(attempts, "total", 0)
    for number in (1, 2):
        part = attempts.loc[attempts.serve_number.eq(number)]
        if not part.empty: rows.extend(_depth_rows(part, "serve_number", number))
    ranks = {"group_type": {"total": 0, "serve_number": 1}, "depth_code": {value: index for index, value in enumerate(DEPTH_ORDER)}}
    return _ordered(pd.DataFrame(rows, columns=BY_DEPTH_COLUMNS), ["group_type", "serve_number", "depth_code"], ranks)


def _group_specs(attempts: pd.DataFrame):
    yield "total", 0, "ALL", "ALL", "ALL", attempts
    for number in (1, 2):
        part = attempts.loc[attempts.serve_number.eq(number)]
        if not part.empty: yield "serve_number", number, "ALL", "ALL", "ALL", part
    for surface, part in attempts.groupby("surface", sort=False, observed=True): yield "surface", 0, surface, "ALL", "ALL", part
    for period, part in attempts.groupby("derived_period", sort=False, observed=True): yield "derived_period", 0, "ALL", period, "ALL", part
    for fold, part in attempts.groupby("validation_fold", sort=False, observed=True): yield "validation_fold", 0, "ALL", "ALL", fold, part


def build_by_group(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, number, surface, period, fold, part in _group_specs(attempts):
        counts = part.classification_state.value_counts()
        observed = int(counts.get(ReturnDepthState.OBSERVED.value, 0))
        rows.append({"group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, **_counts(part), "observed_depth_attempts": observed, "unknown_depth_attempts": int(counts.get(ReturnDepthState.UNKNOWN.value, 0)), "not_documented_attempts": int(counts.get(ReturnDepthState.NOT_DOCUMENTED.value, 0)), "unknown_initial_attempts": int(counts.get(ReturnDepthState.UNKNOWN_INITIAL.value, 0)), "censored_attempts": int(counts.get(ReturnDepthState.CENSORED.value, 0)), "observed_depth_numerator": observed, "observed_depth_denominator": int(len(part)), "observed_depth_coverage": observed / len(part)})
    ranks = {"group_type": {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}}
    return _ordered(pd.DataFrame(rows, columns=BY_GROUP_COLUMNS), ["group_type", "serve_number", "surface", "derived_period", "validation_fold"], ranks)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if type(successes) is not int or type(trials) is not int or successes < 0 or trials < successes: raise FeasibilityContractError("Wilson requiere cuentas enteras coherentes.")
    if trials == 0: return None, None
    rate = successes / trials; denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    lower, upper = centre - spread, centre + spread
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= rate <= upper <= 1): raise FeasibilityContractError("Wilson fuera de rango.")
    return lower, upper


def build_outcomes(attempts: pd.DataFrame) -> pd.DataFrame:
    observed = attempts.loc[attempts.classification_state.eq(ReturnDepthState.OBSERVED.value) & attempts.return_depth_code.isin(DEPTH_ORDER)]
    rows = []
    for kind, number, surface, period, fold, part in _group_specs(observed):
        for code in DEPTH_ORDER:
            group = part.loc[part.return_depth_code.eq(code)]
            wins = int(group.returner_won_point.sum()); total = int(len(group)); lower, upper = wilson(wins, total)
            rows.append({"group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, "depth_code": code, "depth_description": RETURN_DEPTHS[code], **_counts(group), "returner_point_wins": wins, "returner_point_losses": total - wins, "returner_point_win_rate": None if total == 0 else wins / total, "wilson_95_lower": lower, "wilson_95_upper": upper})
    ranks = {"group_type": {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}, "depth_code": {value: index for index, value in enumerate(DEPTH_ORDER)}}
    return _ordered(pd.DataFrame(rows, columns=OUTCOME_COLUMNS), ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "depth_code"], ranks)


def _test_seal(counts: Mapping[str, int]) -> dict[str, Any]:
    return {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_target_matches": counts["excluded_test_target_matches"], **{field: 0 for field in TEST_ZERO_FIELDS}}


def _summary(counts: Mapping[str, int], attempts: pd.DataFrame, by_state: pd.DataFrame, by_depth: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    state_counts = {state: int((attempts.classification_state == state).sum()) for state in STATE_ORDER}
    depth_counts = {code: int((attempts.return_depth_code == code).sum()) for code in DEPTH_ORDER}
    observed = state_counts[ReturnDepthState.OBSERVED.value]
    return {"analysis_name": ANALYSIS_NAME, "version": ANALYSIS_VERSION, "source": {"path": "data/processed/points_enriched.parquet", "columns": list(SOURCE_COLUMNS), "reads": 1}, "upstream_commit": UPSTREAM_COMMIT, "analysis_status": "available_descriptive", "reason_codes": sorted(set(attempts.reason_code)), "population": dict(counts), "attempts": {"total": int(len(attempts)), "first": int((attempts.serve_number == 1).sum()), "second": int((attempts.serve_number == 2).sum()), "cache_entries": int(attempts.attrs.get("cache_entries", 0))}, "state_counts": state_counts, "depth_counts": depth_counts, "coverage": {"observed_depth_attempts": observed, "all_attempts": int(len(attempts)), "end_to_end_coverage": observed / len(attempts), "observed_depth_share": {code: None if observed == 0 else depth_counts[code] / observed for code in DEPTH_ORDER}}, "outcomes_status": "available_descriptive" if observed else "not_available_no_observed_depth", "test_seal": _test_seal(counts), "reconciliations": validate_reconciliations(attempts, by_state, by_depth, by_group, outcomes), "artifact_contracts": {"by_state": BY_STATE_COLUMNS, "by_depth": BY_DEPTH_COLUMNS, "by_group": BY_GROUP_COLUMNS, "outcomes": OUTCOME_COLUMNS}, "fingerprint_contract": {"algorithm": "sha256", "serialization": "utf-8_deterministic"}, "methodological_limits": ["Analisis descriptivo y no causal.", "Los outcomes se condicionan a restos con profundidad documentada 7/8/9.", "La observabilidad y la censura pueden sesgar las tasas descriptivas.", "El test posterior a 2023 permanece sellado y no se clasifica." ]}


def _assert_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if frame.columns.tolist() != list(columns) or any(str(item).startswith("Unnamed") for item in frame.columns):
        raise FeasibilityContractError(f"Schema u orden de {name} invalido.")


def validate_reconciliations(attempts: pd.DataFrame, by_state: pd.DataFrame, by_depth: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, bool]:
    for frame, columns, name in ((by_state, BY_STATE_COLUMNS, "by_state"), (by_depth, BY_DEPTH_COLUMNS, "by_depth"), (by_group, BY_GROUP_COLUMNS, "by_group"), (outcomes, OUTCOME_COLUMNS, "outcomes")):_assert_columns(frame, columns, name)
    if bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()): raise FeasibilityContractError("Clave de intentos duplicada.")
    if set(attempts.classification_state) - set(STATE_ORDER): raise FeasibilityContractError("Estado fuera del catalogo P05.")
    if not attempts.eligible_for_depth_comparison.eq(attempts.classification_state.eq(ReturnDepthState.OBSERVED.value)).all(): raise FeasibilityContractError("Elegibilidad incompatible con estado.")
    if bool(by_state.duplicated(["group_type", "serve_number", "classification_state", "reason_code"]).any()): raise FeasibilityContractError("Clave duplicada en by_state.")
    total_states = by_state.loc[by_state.group_type.eq("total")]
    if int(total_states.attempts.sum()) != len(attempts) or not np.allclose(total_states.proportion_of_denominator, total_states.attempts / total_states.denominator_attempts, rtol=0, atol=1e-15): raise FeasibilityContractError("Estados no reconcilian.")
    if set(by_depth.depth_code) != set(DEPTH_ORDER) or bool(by_depth.duplicated(["group_type", "serve_number", "depth_code"]).any()): raise FeasibilityContractError("Claves o dominio de profundidad invalidos.")
    total_depth = by_depth.loc[by_depth.group_type.eq("total")]
    observed = int((attempts.classification_state == ReturnDepthState.OBSERVED.value).sum())
    if int(total_depth.attempts.sum()) != observed or not total_depth.observed_depth_denominator.eq(observed).all() or not np.allclose(total_depth.end_to_end_coverage, total_depth.attempts / len(attempts), rtol=0, atol=1e-15): raise FeasibilityContractError("Profundidades no reconcilian.")
    if bool(by_group.duplicated(["group_type", "serve_number", "surface", "derived_period", "validation_fold"]).any()): raise FeasibilityContractError("Clave duplicada en by_group.")
    group_total = by_group.loc[by_group.group_type.eq("total")]
    if len(group_total) != 1 or int(group_total.iloc[0].attempts) != len(attempts) or int(group_total.iloc[0].observed_depth_attempts) != observed: raise FeasibilityContractError("Grupo total no reconcilia.")
    if not np.allclose(by_group.observed_depth_coverage, by_group.observed_depth_numerator / by_group.observed_depth_denominator, rtol=0, atol=1e-15): raise FeasibilityContractError("Cobertura de grupos invalida.")
    if bool(outcomes.duplicated(["group_type", "serve_number", "surface", "derived_period", "validation_fold", "depth_code"]).any()): raise FeasibilityContractError("Clave duplicada en outcomes.")
    if set(outcomes.depth_code) != set(DEPTH_ORDER): raise FeasibilityContractError("Dominio de outcomes invalido.")
    if not outcomes.returner_point_losses.eq(outcomes.attempts - outcomes.returner_point_wins).all(): raise FeasibilityContractError("Complemento de outcomes invalido.")
    populated = outcomes.loc[outcomes.attempts.gt(0)]
    if not np.allclose(populated.returner_point_win_rate, populated.returner_point_wins / populated.attempts, rtol=0, atol=1e-15): raise FeasibilityContractError("Tasas de outcomes invalidas.")
    for row in populated.itertuples(index=False):
        if (row.wilson_95_lower, row.wilson_95_upper) != wilson(int(row.returner_point_wins), int(row.attempts)): raise FeasibilityContractError("Wilson no reconcilia.")
    return {"attempt_key_unique": True, "states_exhaustive_exclusive": True, "observed_equals_depth_sum": True, "coverage_reconciled": True, "groups_reconciled": True, "outcomes_reconciled": True, "test_sealed": True}


def _table_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", na_rep="<NULL>").encode("utf-8")


def _fingerprint(summary: Mapping[str, Any], table_payloads: Sequence[bytes]) -> str:
    copy = dict(summary); copy.pop("publication_fingerprint", None)
    canonical = json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(); digest.update(canonical)
    for payload in table_payloads: digest.update(payload)
    return digest.hexdigest().upper()


def _finalize(summary: dict[str, Any], attempts: pd.DataFrame, by_state: pd.DataFrame, by_depth: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> DescriptiveResult:
    payloads = tuple(_table_bytes(frame) for frame in (by_state, by_depth, by_group, outcomes))
    summary = dict(summary)
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_depth", "by_group", "outcomes"), payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(("by_state", "by_depth", "by_group", "outcomes"), payloads)}
    fingerprint = _fingerprint(summary, payloads); summary["publication_fingerprint"] = fingerprint
    return DescriptiveResult(summary, by_state, by_depth, by_group, outcomes, attempts, fingerprint)


def analyze_points(points: pd.DataFrame, *, expected_population: Mapping[str, int] | None = None, extractor: Callable[..., ReturnDepthClassification] = parse_and_classify_initial_return_depth) -> DescriptiveResult:
    source = validate_source_points(points, expected_rows=None if expected_population is None else expected_population["source_rows_read"])
    development, counts = split_development_before_parsing(source)
    if expected_population is not None:
        for key, expected in expected_population.items():
            if key in counts and counts[key] != expected: raise FeasibilityContractError(f"{key} inesperado: {counts[key]}")
    attempts = construct_attempts(development, extractor=extractor)
    if expected_population is not None:
        expected_attempts = {"attempts_total": len(attempts), "first_attempts": int((attempts.serve_number == 1).sum()), "second_attempts": int((attempts.serve_number == 2).sum())}
        for key, actual in expected_attempts.items():
            if actual != expected_population[key]: raise FeasibilityContractError(f"{key} inesperado: {actual}")
    by_state, by_depth, by_group, outcomes = build_by_state(attempts), build_by_depth(attempts), build_by_group(attempts), build_outcomes(attempts)
    result = _finalize(_summary(counts, attempts, by_state, by_depth, by_group, outcomes), attempts, by_state, by_depth, by_group, outcomes)
    validate_result(result)
    return result


def not_available_result(error: BaseException) -> DescriptiveResult:
    message = str(error).replace("\n", " ").strip()[:240] or type(error).__name__
    summary = {"analysis_name": ANALYSIS_NAME, "version": ANALYSIS_VERSION, "source": {"path": "data/processed/points_enriched.parquet", "columns": list(SOURCE_COLUMNS), "reads": 1}, "upstream_commit": UPSTREAM_COMMIT, "analysis_status": "not_available", "reason_codes": ["execution_failed"], "failure": {"stage": "run_real_analysis", "message": message}, "population": {}, "attempts": {}, "state_counts": {}, "depth_counts": {}, "coverage": {}, "outcomes_status": "not_available", "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_target_matches": None, **{field: 0 for field in TEST_ZERO_FIELDS}}, "reconciliations": {}, "artifact_contracts": {"by_state": BY_STATE_COLUMNS, "by_depth": BY_DEPTH_COLUMNS, "by_group": BY_GROUP_COLUMNS, "outcomes": OUTCOME_COLUMNS}, "fingerprint_contract": {"algorithm": "sha256", "serialization": "utf-8_deterministic"}, "methodological_limits": ["Analisis no disponible; no se publican agregados parciales."]}
    empty = (pd.DataFrame(columns=BY_STATE_COLUMNS), pd.DataFrame(columns=BY_DEPTH_COLUMNS), pd.DataFrame(columns=BY_GROUP_COLUMNS), pd.DataFrame(columns=OUTCOME_COLUMNS))
    return _finalize(summary, pd.DataFrame(), *empty)


def validate_result(result: DescriptiveResult) -> None:
    if not isinstance(result, DescriptiveResult): raise TypeError("result debe ser DescriptiveResult.")
    for frame, columns, name in ((result.by_state, BY_STATE_COLUMNS, "by_state"), (result.by_depth, BY_DEPTH_COLUMNS, "by_depth"), (result.by_group, BY_GROUP_COLUMNS, "by_group"), (result.outcomes, OUTCOME_COLUMNS, "outcomes")):_assert_columns(frame, columns, name)
    payloads = tuple(_table_bytes(frame) for frame in (result.by_state, result.by_depth, result.by_group, result.outcomes))
    expected_hashes = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_depth", "by_group", "outcomes"), payloads)}
    expected_sizes = {name: len(payload) for name, payload in zip(("by_state", "by_depth", "by_group", "outcomes"), payloads)}
    if result.summary.get("artifact_payload_sha256") != expected_hashes or result.summary.get("artifact_payload_bytes") != expected_sizes:
        raise FeasibilityContractError("Hashes de artefactos invalidos.")
    if result.summary.get("publication_fingerprint") != _fingerprint(result.summary, payloads) or result.publication_fingerprint != result.summary["publication_fingerprint"]: raise FeasibilityContractError("Fingerprint invalido.")
    if result.summary.get("analysis_status") not in {"available_descriptive", "not_available"}:
        raise FeasibilityContractError("analysis_status invalido.")
    if result.summary.get("analysis_status") == "not_available":
        if any(not frame.empty for frame in (result.by_state, result.by_depth, result.by_group, result.outcomes)): raise FeasibilityContractError("not_available no puede publicar filas.")
        return
    validate_reconciliations(result.attempts, result.by_state, result.by_depth, result.by_group, result.outcomes)
    summary = result.summary
    if summary.get("state_counts") != {state: int((result.attempts.classification_state == state).sum()) for state in STATE_ORDER}: raise FeasibilityContractError("state_counts no reconcilia.")
    if summary.get("depth_counts") != {code: int((result.attempts.return_depth_code == code).sum()) for code in DEPTH_ORDER}: raise FeasibilityContractError("depth_counts no reconcilia.")
    attempts = summary.get("attempts", {}); population = summary.get("population", {}); observed = int((result.attempts.classification_state == ReturnDepthState.OBSERVED.value).sum())
    if attempts.get("total") != len(result.attempts) or attempts.get("first") != int((result.attempts.serve_number == 1).sum()) or attempts.get("second") != int((result.attempts.serve_number == 2).sum()) or population.get("development_point_rows") != attempts.get("first") or population.get("development_matches") != int(result.attempts.match_id.nunique()) or population.get("development_servers") != int(result.attempts.server_player.nunique()):
        raise FeasibilityContractError("Poblacion o intentos del resumen no reconcilian.")
    coverage = summary.get("coverage", {})
    if coverage.get("observed_depth_attempts") != observed or coverage.get("all_attempts") != len(result.attempts) or coverage.get("end_to_end_coverage") != observed / len(result.attempts):
        raise FeasibilityContractError("Cobertura del resumen no reconcilia.")
    seal = summary.get("test_seal", {})
    if seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS): raise FeasibilityContractError("Test seal invalido.")


def serialize_artifacts(result: DescriptiveResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    validate_result(result)
    summary = (json.dumps(result.summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return (summary, *(_table_bytes(frame) for frame in (result.by_state, result.by_depth, result.by_group, result.outcomes)))


def _stage(path: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return Path(name)


def write_artifacts(result: DescriptiveResult, *, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_depth_path: Path = BY_DEPTH_PATH, by_group_path: Path = BY_GROUP_PATH, outcomes_path: Path = OUTCOMES_PATH, payloads: tuple[bytes, bytes, bytes, bytes, bytes] | None = None) -> None:
    """Prepara los cinco ficheros, reemplaza, y revierte ante un fallo normal."""
    targets = (summary_path, by_state_path, by_depth_path, by_group_path, outcomes_path)
    canonical_payloads = serialize_artifacts(result)
    payloads = canonical_payloads if payloads is None else payloads
    if len(payloads) != 5 or any(not isinstance(value, bytes) for value in payloads): raise FeasibilityContractError("Payloads invalidos.")
    if payloads != canonical_payloads: raise FeasibilityContractError("Payloads no corresponden al resultado validado.")
    for target in targets: target.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []; backups: dict[Path, bytes | None] = {}
    try:
        for target, payload in zip(targets, payloads): staged.append(_stage(target, payload))
        backups = {target: target.read_bytes() if target.exists() else None for target in targets}
        for temporary, target in zip(staged, targets): os.replace(temporary, target)
    except BaseException:
        for target, previous in backups.items():
            if previous is None:
                if target.exists(): target.unlink()
            else: target.write_bytes(previous)
        raise
    finally:
        for temporary in staged: temporary.unlink(missing_ok=True)


def _read_csv_payload(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, na_values=["<NULL>"])
    _assert_columns(frame, columns, path.name)
    text_columns = {"group_type", "classification_state", "reason_code", "depth_code", "depth_description", "surface", "derived_period", "validation_fold"}
    for column in columns:
        if column in text_columns:
            if frame[column].isna().any():
                raise FeasibilityContractError(f"{column} no puede ser nulo en artefacto persistido.")
            frame[column] = frame[column].astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def verify_persisted_artifacts(*, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_depth_path: Path = BY_DEPTH_PATH, by_group_path: Path = BY_GROUP_PATH, outcomes_path: Path = OUTCOMES_PATH, expected_population: Mapping[str, int] | None = None, frozen_publication: bool = False) -> None:
    """Valida los cinco artefactos persistidos sin abrir Parquet."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = (_read_csv_payload(by_state_path, BY_STATE_COLUMNS), _read_csv_payload(by_depth_path, BY_DEPTH_COLUMNS), _read_csv_payload(by_group_path, BY_GROUP_COLUMNS), _read_csv_payload(outcomes_path, OUTCOME_COLUMNS))
    payloads = tuple(path.read_bytes() for path in (by_state_path, by_depth_path, by_group_path, outcomes_path))
    expected_hashes = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_depth", "by_group", "outcomes"), payloads)}
    if summary.get("analysis_name") != ANALYSIS_NAME or summary.get("version") != ANALYSIS_VERSION or summary.get("upstream_commit") != UPSTREAM_COMMIT: raise FeasibilityContractError("Identidad persistida invalida.")
    if summary.get("artifact_payload_sha256") != expected_hashes or summary.get("publication_fingerprint") != _fingerprint(summary, payloads): raise FeasibilityContractError("Fingerprint persistido invalido.")
    if summary.get("analysis_status") not in {"available_descriptive", "not_available"}:
        raise FeasibilityContractError("analysis_status persistido invalido.")
    if summary.get("analysis_status") == "not_available":
        if any(not frame.empty for frame in frames): raise FeasibilityContractError("not_available persistido con filas.")
        return
    state, depth, groups, outcomes = frames
    if state.duplicated(["group_type", "serve_number", "classification_state", "reason_code"]).any() or depth.duplicated(["group_type", "serve_number", "depth_code"]).any() or groups.duplicated(["group_type", "serve_number", "surface", "derived_period", "validation_fold"]).any() or outcomes.duplicated(["group_type", "serve_number", "surface", "derived_period", "validation_fold", "depth_code"]).any(): raise FeasibilityContractError("Claves persistidas duplicadas.")
    if set(depth.depth_code) != set(DEPTH_ORDER) or set(outcomes.depth_code) != set(DEPTH_ORDER): raise FeasibilityContractError("Dominio de profundidad persistido invalido.")
    if set(state.group_type) - {"total", "serve_number"} or set(depth.group_type) - {"total", "serve_number"} or not state.loc[state.group_type.eq("total"), "serve_number"].eq(0).all() or not depth.loc[depth.group_type.eq("total"), "serve_number"].eq(0).all() or not state.loc[state.group_type.eq("serve_number"), "serve_number"].isin((1, 2)).all() or not depth.loc[depth.group_type.eq("serve_number"), "serve_number"].isin((1, 2)).all():
        raise FeasibilityContractError("Dominio de serve_number persistido invalido.")
    allowed_group_types = {"total", "serve_number", "surface", "derived_period", "validation_fold"}
    if set(groups.group_type) - allowed_group_types or set(outcomes.group_type) - allowed_group_types:
        raise FeasibilityContractError("group_type persistido invalido.")
    for frame in (groups, outcomes):
        if not frame.loc[frame.group_type.eq("total"), ["serve_number", "surface", "derived_period", "validation_fold"]].eq([0, "ALL", "ALL", "ALL"]).all(axis=None): raise FeasibilityContractError("Grupo total persistido invalido.")
        if not frame.loc[frame.group_type.eq("serve_number"), "serve_number"].isin((1, 2)).all() or not frame.loc[frame.group_type.eq("surface"), "surface"].isin(ALLOWED_SURFACES).all() or not frame.loc[frame.group_type.eq("derived_period"), "derived_period"].isin(PERIOD_ORDER).all() or not frame.loc[frame.group_type.eq("validation_fold"), "validation_fold"].isin(FOLD_ORDER).all(): raise FeasibilityContractError("Dominio de grupo persistido invalido.")
    if set(state.classification_state) - set(STATE_ORDER) or any(row.reason_code not in STATE_REASON_CODES.get(row.classification_state, set()) for row in state.itertuples(index=False)):
        raise FeasibilityContractError("Estado o reason_code persistido invalido.")
    ranks = {"group_type": {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}, "classification_state": {value: index for index, value in enumerate(STATE_ORDER)}, "depth_code": {value: index for index, value in enumerate(DEPTH_ORDER)}}
    if not state.equals(_ordered(state, ["group_type", "serve_number", "classification_state", "reason_code"], ranks)) or not depth.equals(_ordered(depth, ["group_type", "serve_number", "depth_code"], ranks)) or not groups.equals(_ordered(groups, ["group_type", "serve_number", "surface", "derived_period", "validation_fold"], ranks)) or not outcomes.equals(_ordered(outcomes, ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "depth_code"], ranks)):
        raise FeasibilityContractError("Orden persistido invalido.")
    total_states = state.loc[state.group_type.eq("total")]
    total_depth = depth.loc[depth.group_type.eq("total")]
    total_group = groups.loc[groups.group_type.eq("total")]
    if len(total_group) != 1 or int(total_states.attempts.sum()) != int(total_group.iloc[0].attempts) or int(total_depth.attempts.sum()) != int(total_group.iloc[0].observed_depth_attempts): raise FeasibilityContractError("Reconciliacion persistida de totales invalida.")
    persisted_counts = {state_name: int(total_states.loc[total_states.classification_state.eq(state_name), "attempts"].sum()) for state_name in STATE_ORDER}
    if summary.get("state_counts") != persisted_counts or summary.get("depth_counts") != {code: int(total_depth.loc[total_depth.depth_code.eq(code), "attempts"].sum()) for code in DEPTH_ORDER}:
        raise FeasibilityContractError("Resumen persistido no reconcilia estados o profundidades.")
    first_attempts = int(state.loc[(state.group_type.eq("serve_number")) & (state.serve_number.eq(1)), "attempts"].sum())
    second_attempts = int(state.loc[(state.group_type.eq("serve_number")) & (state.serve_number.eq(2)), "attempts"].sum())
    summary_attempts = summary.get("attempts", {})
    if summary_attempts.get("total") != int(total_group.iloc[0].attempts) or summary_attempts.get("first") != first_attempts or summary_attempts.get("second") != second_attempts or first_attempts + second_attempts != int(total_group.iloc[0].attempts):
        raise FeasibilityContractError("Intentos persistidos no reconcilian.")
    observed_by_serve = state.loc[(state.group_type.eq("serve_number")) & state.classification_state.eq(ReturnDepthState.OBSERVED.value)].groupby("serve_number", sort=False).attempts.sum().to_dict()
    expected_depth_keys = {("total", 0, code) for code in DEPTH_ORDER} | {("serve_number", int(number), code) for number in observed_by_serve for code in DEPTH_ORDER}
    actual_depth_keys = set(map(tuple, depth[["group_type", "serve_number", "depth_code"]].itertuples(index=False, name=None)))
    if actual_depth_keys != expected_depth_keys:
        raise FeasibilityContractError("Conjunto de filas by_depth persistido invalido.")
    for number, observed_count in observed_by_serve.items():
        part = depth.loc[(depth.group_type.eq("serve_number")) & depth.serve_number.eq(number)]
        if int(part.attempts.sum()) != int(observed_count) or not part.observed_depth_denominator.eq(observed_count).all():
            raise FeasibilityContractError("Profundidades por saque no reconcilian.")
    population = summary.get("population", {})
    if population.get("development_point_rows") != first_attempts or population.get("development_matches") != int(total_group.iloc[0].matches) or population.get("development_servers") != int(total_group.iloc[0].servers):
        raise FeasibilityContractError("Poblacion de desarrollo persistida no reconcilia.")
    coverage = summary.get("coverage", {})
    expected_coverage = int(total_group.iloc[0].observed_depth_attempts) / int(total_group.iloc[0].attempts)
    if coverage.get("observed_depth_attempts") != int(total_group.iloc[0].observed_depth_attempts) or coverage.get("all_attempts") != int(total_group.iloc[0].attempts) or not math.isclose(coverage.get("end_to_end_coverage"), expected_coverage, rel_tol=0, abs_tol=1e-15):
        raise FeasibilityContractError("Cobertura persistida no reconcilia.")
    if expected_population is not None and any(population.get(key) != value for key, value in expected_population.items() if key in population):
        raise FeasibilityContractError("Poblacion persistida no corresponde al contrato esperado.")
    if not np.allclose(total_depth.proportion_of_observed_depths, total_depth.attempts / total_depth.observed_depth_denominator, rtol=0, atol=1e-15): raise FeasibilityContractError("Proporcion condicionada persistida invalida.")
    populated = outcomes.loc[outcomes.attempts.gt(0)]
    if not populated.returner_point_losses.eq(populated.attempts - populated.returner_point_wins).all() or not np.allclose(populated.returner_point_win_rate, populated.returner_point_wins / populated.attempts, rtol=0, atol=1e-15): raise FeasibilityContractError("Outcomes persistidos invalidos.")
    outcome_total = outcomes.loc[outcomes.group_type.eq("total")]
    if not outcome_total.groupby("depth_code", sort=False).attempts.sum().reindex(DEPTH_ORDER, fill_value=0).eq(total_depth.set_index("depth_code").attempts.reindex(DEPTH_ORDER, fill_value=0)).all():
        raise FeasibilityContractError("Outcomes persistidos no reconcilian profundidades.")
    group_keys = ["group_type", "serve_number", "surface", "derived_period", "validation_fold"]
    outcome_groups = groups.loc[groups.observed_depth_attempts.gt(0), group_keys + ["observed_depth_attempts"]]
    expected_outcome_keys = {(*key, code) for key in outcome_groups[group_keys].itertuples(index=False, name=None) for code in DEPTH_ORDER}
    actual_outcome_keys = set(map(tuple, outcomes[[*group_keys, "depth_code"]].itertuples(index=False, name=None)))
    if actual_outcome_keys != expected_outcome_keys:
        raise FeasibilityContractError("Conjunto de filas outcomes persistido invalido.")
    outcome_sums = outcomes.groupby(group_keys, sort=False).attempts.sum().reset_index(name="outcome_attempts")
    expected_outcome_sums = outcome_groups.rename(columns={"observed_depth_attempts": "outcome_attempts"})
    if not outcome_sums.sort_values(group_keys).reset_index(drop=True).equals(expected_outcome_sums.sort_values(group_keys).reset_index(drop=True)):
        raise FeasibilityContractError("Outcomes por grupo no reconcilian.")
    for row in populated.itertuples(index=False):
        lower, upper = wilson(int(row.returner_point_wins), int(row.attempts))
        if not math.isclose(float(row.wilson_95_lower), lower, rel_tol=0, abs_tol=1e-15) or not math.isclose(float(row.wilson_95_upper), upper, rel_tol=0, abs_tol=1e-15):
            raise FeasibilityContractError("Wilson persistido invalido.")
    seal = summary.get("test_seal", {})
    if seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS): raise FeasibilityContractError("Sellado persistido invalido.")
    if frozen_publication:
        actual = {"summary": summary_path.read_bytes(), "by_state": payloads[0], "by_depth": payloads[1], "by_group": payloads[2], "outcomes": payloads[3]}
        for name, payload in actual.items():
            expected_size, expected_hash = PUBLISHED_ARTIFACT_CONTRACT[name]
            if len(payload) != expected_size or hashlib.sha256(payload).hexdigest().upper() != expected_hash:
                raise FeasibilityContractError(f"Artefacto publicado congelado invalido: {name}.")
        if summary.get("publication_fingerprint") != PUBLISHED_PUBLICATION_FINGERPRINT:
            raise FeasibilityContractError("Fingerprint publicado congelado invalido.")


def publish_artifacts(result: DescriptiveResult) -> None:
    payloads = serialize_artifacts(result)
    if serialize_artifacts(result) != payloads: raise FeasibilityContractError("Reserializacion no determinista.")
    write_artifacts(result, payloads=payloads)
    verify_persisted_artifacts()


def read_source_points(path: Path = POINTS_PATH) -> pd.DataFrame:
    """Unica lectura futura del Parquet; no se llama durante pruebas sinteticas."""
    return pd.read_parquet(path, columns=list(SOURCE_COLUMNS))


def run_real_analysis() -> DescriptiveResult:
    return analyze_points(read_source_points(), expected_population=EXPECTED_REAL)


def _performance_path(value: str | None) -> Path | None:
    if value is None: return None
    path = Path(value).resolve()
    if path == ROOT or ROOT in path.parents: raise FeasibilityContractError("--performance-log debe estar fuera del repositorio.")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P05: profundidad documentada del primer resto.")
    parser.add_argument("--performance-log")
    arguments = parser.parse_args(argv)
    # Se valida antes de abrir la fuente: una ruta prohibida nunca inicia la
    # ejecucion real ni deja artefactos parciales.
    destination = _performance_path(arguments.performance_log)
    started = time.perf_counter(); error: BaseException | None = None
    try: result = run_real_analysis()
    except BaseException as caught: error = caught; result = not_available_result(caught)
    publish_artifacts(result)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"analysis_name": ANALYSIS_NAME, "elapsed_seconds": time.perf_counter() - started, "analysis_status": result.summary["analysis_status"]}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if error is not None: raise SystemExit(1)


if __name__ == "__main__":
    main()
