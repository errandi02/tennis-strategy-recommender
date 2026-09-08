"""P06: infraestructura descriptiva sellada para el tipo del primer resto.

El módulo no lee datos al importarse. Sus transformaciones aceptan DataFrames
sintéticos; el CLI futuro lee la fuente una sola vez, sella el test antes de
construir intentos y publica únicamente agregados transaccionales.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import argparse
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

from src.analysis.return_shot_type_feasibility import (
    DOCUMENTED_SHOT_TYPES,
    SHOT_TYPE_FAMILIES,
    SHOT_TYPE_FAMILY_MEMBERS,
    ReturnShotTypeClassification,
    ReturnShotTypeReason,
    ReturnShotTypeState,
    parse_and_classify_initial_return_shot_type,
    validate_return_shot_type_classification,
)


ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "return_shot_type_descriptive_feasibility_summary.json"
BY_STATE_PATH = TABLES_DIR / "return_shot_type_descriptive_feasibility_by_state.csv"
BY_TYPE_PATH = TABLES_DIR / "return_shot_type_descriptive_feasibility_by_type.csv"
BY_GROUP_PATH = TABLES_DIR / "return_shot_type_descriptive_feasibility_by_group.csv"
OUTCOMES_PATH = TABLES_DIR / "return_shot_type_descriptive_feasibility_outcomes.csv"

ANALYSIS_NAME = "return_shot_type_descriptive_feasibility"
ANALYSIS_VERSION = "1.0.0"
UPSTREAM_COMMIT = "41b0e5c"
CUTOFF = pd.Timestamp("2023-12-31")
WILSON_BOUNDARY_TOLERANCE = 1e-15
ALLOWED_SURFACES = ("Hard", "Clay", "Grass")
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
FOLD_ORDER = ("pre_validation", "validation_2020", "validation_2021", "validation_2022", "validation_2023")
STATE_ORDER = tuple(state.value for state in ReturnShotTypeState)
TYPE_ORDER = tuple(DOCUMENTED_SHOT_TYPES)
FAMILY_ORDER = tuple(SHOT_TYPE_FAMILY_MEMBERS)
SOURCE_COLUMNS = ("match_id", "point_number", "date", "surface", "server", "point_winner", "player_1", "player_2", "first_serve", "second_serve")
EXPECTED_REAL = {"source_rows_read": 1_280_408, "source_matches": 7_524, "source_players": 1_002, "development_point_rows": 1_035_760, "development_matches": 5_993, "development_servers": 870, "excluded_test_matches": 1_531, "attempts_total": 1_426_863, "first_attempts": 1_035_760, "second_attempts": 391_103}
TEST_ZERO_FIELDS = ("test_target_rows_parsed", "test_attempts_constructed", "test_types_classified", "test_outcomes_computed", "test_rows_evaluated", "test_matches_evaluated", "test_scores_computed", "test_evaluation_runs", "test_recommendations_generated")
STATE_REASON_CODES = {
    ReturnShotTypeState.OBSERVED.value: {f"documented_return_shot_type_{code}" for code in TYPE_ORDER},
    ReturnShotTypeState.UNKNOWN.value: {ReturnShotTypeReason.Q.value},
    ReturnShotTypeState.UNKNOWN_INITIAL.value: {
        ReturnShotTypeReason.MISSING_PREFIX.value,
        ReturnShotTypeReason.BOUNDARY.value,
        ReturnShotTypeReason.MODIFIER.value,
        ReturnShotTypeReason.TRUNCATED.value,
        ReturnShotTypeReason.INCONSISTENT.value,
    },
    ReturnShotTypeState.CENSORED.value: {
        ReturnShotTypeReason.ACE.value,
        ReturnShotTypeReason.UNRETURNED.value,
        ReturnShotTypeReason.FAULT.value,
        ReturnShotTypeReason.DOUBLE_FAULT.value,
        ReturnShotTypeReason.SPECIAL.value,
        ReturnShotTypeReason.INCOMPLETE_LET.value,
    },
}

BY_STATE_COLUMNS = ["group_type", "serve_number", "classification_state", "reason_code", "attempts", "matches", "servers", "returners", "denominator_attempts", "proportion_of_denominator"]
BY_TYPE_COLUMNS = ["aggregation_level", "serve_number", "shot_type_code", "shot_type_description", "shot_type_family", "attempts", "matches", "servers", "returners", "observed_type_denominator", "share_among_observed_types", "end_to_end_denominator", "end_to_end_coverage"]
BY_GROUP_COLUMNS = ["group_type", "serve_number", "surface", "derived_period", "validation_fold", "attempts", "matches", "servers", "returners", "observed_type_attempts", "unknown_type_attempts", "unknown_initial_attempts", "censored_attempts", "observed_type_numerator", "observed_type_denominator", "observed_type_coverage"]
OUTCOME_COLUMNS = ["aggregation_level", "group_type", "serve_number", "surface", "derived_period", "validation_fold", "shot_type_code", "shot_type_description", "shot_type_family", "attempts", "matches", "servers", "returners", "returner_wins", "returner_losses", "returner_win_rate", "wilson_95_lower", "wilson_95_upper"]


class FeasibilityContractError(ValueError):
    """Violación cerrada de fuente, sellado, resultado o publicación P06."""


class _RunFailure(RuntimeError):
    """Fallo único etiquetado antes de publicar un resultado no disponible."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


@dataclass(frozen=True)
class DescriptiveResult:
    summary: dict[str, Any]
    by_state: pd.DataFrame
    by_type: pd.DataFrame
    by_group: pd.DataFrame
    outcomes: pd.DataFrame
    attempts: pd.DataFrame
    publication_fingerprint: str


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value not in (1, 2):
        raise FeasibilityContractError(f"{field} debe ser exactamente el entero 1 o 2; recibido {value!r}.")
    return int(value)


def _text(series: pd.Series, field: str) -> None:
    invalid = series.map(lambda value: not isinstance(value, str) or not value or value != value.strip())
    if bool(invalid.any()):
        raise FeasibilityContractError(f"{field} debe ser texto no vacío y sin espacios externos.")


def _dates(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise FeasibilityContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif bool(series.map(lambda value: isinstance(value, str)).all()):
        if not bool(series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()):
            raise FeasibilityContractError("date debe usar YYYY-MM-DD sin ambigüedad.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif bool(series.map(lambda value: isinstance(value, (pd.Timestamp, datetime, date)) and not isinstance(value, bool)).all()):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        raise FeasibilityContractError("date contiene tipos ambiguos.")
    if parsed.isna().any() or getattr(parsed.dt, "tz", None) is not None:
        raise FeasibilityContractError("date inválida o con zona horaria.")
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
    fold = f"validation_{day.year}"
    if fold not in FOLD_ORDER:
        raise FeasibilityContractError("Fecha de desarrollo fuera de folds aprobados.")
    return fold


def validate_source_points(points: pd.DataFrame, *, expected_rows: int | None = None) -> pd.DataFrame:
    if not isinstance(points, pd.DataFrame): raise TypeError("points debe ser DataFrame.")
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing: raise FeasibilityContractError(f"Faltan columnas requeridas: {missing}")
    if expected_rows is not None and len(points) != expected_rows: raise FeasibilityContractError(f"source_rows_read inesperado: {len(points)}")
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"): _text(work[field], field)
    if work.point_number.isna().any() or bool(work.duplicated(["match_id", "point_number"]).any()):
        raise FeasibilityContractError("(match_id, point_number) debe ser clave única no nula.")
    work.date = _dates(work.date)
    unexpected = sorted(set(work.loc[~work.surface.isin(ALLOWED_SURFACES), "surface"]))
    if unexpected: raise FeasibilityContractError(f"Superficies inesperadas: {unexpected}")
    if bool(work.player_1.eq(work.player_2).any()): raise FeasibilityContractError("player_1 y player_2 deben ser distintos.")
    for field in ("server", "point_winner"): work[field] = work[field].map(lambda value: _strict_index(value, field))
    if bool(work.groupby("match_id", sort=False)[["date", "surface", "player_1", "player_2"]].nunique(dropna=False).gt(1).any().any()):
        raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    return work.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)


def split_development_before_parsing(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    development = source.loc[source.date.le(CUTOFF)].copy()
    sealed = source.loc[source.date.gt(CUTOFF)]
    if development.empty or sealed.empty or bool(development.date.gt(CUTOFF).any()): raise FeasibilityContractError("El sellado temporal no particiona la fuente.")
    counts = {"source_rows_read": int(len(source)), "source_matches": int(source.match_id.nunique()), "source_players": int(len(set(source.player_1) | set(source.player_2))), "development_point_rows": int(len(development)), "development_matches": int(development.match_id.nunique()), "development_servers": int(pd.Series(np.where(development.server.eq(1), development.player_1, development.player_2)).nunique()), "excluded_test_matches": int(sealed.match_id.nunique())}
    return development.reset_index(drop=True), counts


def _attempt_row(row: Any, text: str, number: int, previous_fault: bool, classified: ReturnShotTypeClassification) -> dict[str, Any]:
    return {"match_id": row.match_id, "point_number": row.point_number, "serve_number": number, "date": row.date, "surface": row.surface, "derived_period": derive_period(row.date), "validation_fold": validation_fold(row.date), "server_player": row.player_1 if row.server == 1 else row.player_2, "returner_player": row.player_2 if row.server == 1 else row.player_1, "previous_attempt_was_fault": previous_fault, "returner_won_point": bool(row.point_winner != row.server), "classification_state": classified.classification_state.value, "reason_code": classified.reason_codes[0].value, "eligible_for_type_comparison": classified.eligible_for_type_comparison, "return_event_observed": classified.return_event_observed, "actor": classified.actor, "shot_type_code": classified.return_shot_type_code, "shot_type_description": classified.documented_description, "shot_type_family": classified.shot_type_family, "terminal_serve_outcome": classified.terminal_serve_outcome, "parser_warning_codes": classified.parser_warning_codes, "parser_warning_spans": classified.parser_warning_spans, "residual_spans": classified.residual_spans}


def construct_attempts(development: pd.DataFrame, *, extractor: Callable[..., ReturnShotTypeClassification] = parse_and_classify_initial_return_shot_type) -> pd.DataFrame:
    cache: dict[tuple[str, int, bool], ReturnShotTypeClassification] = {}
    def classify(text: str, number: int, prior: bool) -> ReturnShotTypeClassification:
        key = (text, number, prior)
        if key not in cache:
            classified = extractor(text, number, previous_attempt_was_fault=prior)
            validate_return_shot_type_classification(classified)
            cache[key] = classified
        return cache[key]
    rows: list[dict[str, Any]] = []
    for row in development.itertuples(index=False):
        if _presence(row.first_serve, "first_serve") != "substantive": raise FeasibilityContractError("first_serve debe ser sustantivo.")
        first = classify(row.first_serve, 1, False)
        rows.append(_attempt_row(row, row.first_serve, 1, False, first))
        if _presence(row.second_serve, "second_serve") == "substantive":
            prior = first.terminal_serve_outcome == "service_fault"
            rows.append(_attempt_row(row, row.second_serve, 2, prior, classify(row.second_serve, 2, prior)))
    attempts = pd.DataFrame(rows)
    if attempts.empty or bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()): raise FeasibilityContractError("Intentos vacíos o clave duplicada.")
    attempts = attempts.sort_values(["date", "match_id", "point_number", "serve_number"], kind="stable").reset_index(drop=True)
    attempts.attrs["cache_entries"] = len(cache)
    return attempts


def _counts(part: pd.DataFrame) -> dict[str, int]:
    return {"attempts": int(len(part)), "matches": int(part.match_id.nunique()), "servers": int(part.server_player.nunique()), "returners": int(part.returner_player.nunique())}


def _ordered(frame: pd.DataFrame, keys: Sequence[str], ranks: Mapping[str, Mapping[Any, int]] | None = None) -> pd.DataFrame:
    work, ranks, auxiliary = frame.copy(), ranks or {}, []
    for key in keys:
        if key in ranks:
            name = f"__{key}"; work[name] = work[key].map(ranks[key]); auxiliary.append(name)
    return work.sort_values([*auxiliary, *keys], kind="stable").drop(columns=auxiliary).reset_index(drop=True)


def build_by_state(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, number, part in [("total", 0, attempts), *(("serve_number", value, attempts.loc[attempts.serve_number.eq(value)]) for value in (1, 2))]:
        for (state, reason), group in part.groupby(["classification_state", "reason_code"], sort=False, observed=True):
            rows.append({"group_type": kind, "serve_number": number, "classification_state": state, "reason_code": reason, **_counts(group), "denominator_attempts": int(len(part)), "proportion_of_denominator": len(group) / len(part)})
    return _ordered(pd.DataFrame(rows, columns=BY_STATE_COLUMNS), ["group_type", "serve_number", "classification_state", "reason_code"], {"group_type": {"total": 0, "serve_number": 1}, "classification_state": {value: index for index, value in enumerate(STATE_ORDER)}})


def _type_rows(part: pd.DataFrame, number: int) -> list[dict[str, Any]]:
    observed = part.loc[part.classification_state.eq(ReturnShotTypeState.OBSERVED.value)]
    rows = []
    for code in TYPE_ORDER:
        group = observed.loc[observed.shot_type_code.eq(code)]
        rows.append({"aggregation_level": "code", "serve_number": number, "shot_type_code": code, "shot_type_description": DOCUMENTED_SHOT_TYPES[code], "shot_type_family": SHOT_TYPE_FAMILIES[code], **_counts(group), "observed_type_denominator": int(len(observed)), "share_among_observed_types": None if observed.empty else len(group) / len(observed), "end_to_end_denominator": int(len(part)), "end_to_end_coverage": None if part.empty else len(group) / len(part)})
    for family in FAMILY_ORDER:
        codes = SHOT_TYPE_FAMILY_MEMBERS[family]; group = observed.loc[observed.shot_type_code.isin(codes)]
        rows.append({"aggregation_level": "family", "serve_number": number, "shot_type_code": None, "shot_type_description": None, "shot_type_family": family, **_counts(group), "observed_type_denominator": int(len(observed)), "share_among_observed_types": None if observed.empty else len(group) / len(observed), "end_to_end_denominator": int(len(part)), "end_to_end_coverage": None if part.empty else len(group) / len(part)})
    return rows


def build_by_type(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = _type_rows(attempts, 0)
    for number in (1, 2): rows.extend(_type_rows(attempts.loc[attempts.serve_number.eq(number)], number))
    rank = {"code": 0, "family": 1}
    return _ordered(pd.DataFrame(rows, columns=BY_TYPE_COLUMNS), ["aggregation_level", "serve_number", "shot_type_family", "shot_type_code"], {"aggregation_level": rank, "shot_type_family": {value: index for index, value in enumerate(FAMILY_ORDER)}, "shot_type_code": {value: index for index, value in enumerate(TYPE_ORDER)}})


def _group_specs(attempts: pd.DataFrame):
    yield "total", 0, "ALL", "ALL", "ALL", attempts
    for number in (1, 2): yield "serve_number", number, "ALL", "ALL", "ALL", attempts.loc[attempts.serve_number.eq(number)]
    for surface, part in attempts.groupby("surface", sort=False, observed=True): yield "surface", 0, surface, "ALL", "ALL", part
    for period, part in attempts.groupby("derived_period", sort=False, observed=True): yield "derived_period", 0, "ALL", period, "ALL", part
    for fold, part in attempts.groupby("validation_fold", sort=False, observed=True): yield "validation_fold", 0, "ALL", "ALL", fold, part


def build_by_group(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, number, surface, period, fold, part in _group_specs(attempts):
        counts = part.classification_state.value_counts(); observed = int(counts.get(ReturnShotTypeState.OBSERVED.value, 0))
        rows.append({"group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, **_counts(part), "observed_type_attempts": observed, "unknown_type_attempts": int(counts.get(ReturnShotTypeState.UNKNOWN.value, 0)), "unknown_initial_attempts": int(counts.get(ReturnShotTypeState.UNKNOWN_INITIAL.value, 0)), "censored_attempts": int(counts.get(ReturnShotTypeState.CENSORED.value, 0)), "observed_type_numerator": observed, "observed_type_denominator": int(len(part)), "observed_type_coverage": observed / len(part)})
    return _ordered(pd.DataFrame(rows, columns=BY_GROUP_COLUMNS), ["group_type", "serve_number", "surface", "derived_period", "validation_fold"], {"group_type": {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}})


def _wilson_failure(successes: int, trials: int, rate: float, lower: float, upper: float, context: Mapping[str, Any] | None) -> FeasibilityContractError:
    details = {
        "successes": successes,
        "trials": trials,
        "rate": rate,
        "lower": lower,
        "upper": upper,
        **(dict(context) if context is not None else {}),
    }
    safe = ", ".join(f"{key}={value!r}" for key, value in details.items())
    return FeasibilityContractError(f"Wilson fuera de rango: {safe}")


def wilson(successes: int, trials: int, z: float = 1.959963984540054, *, context: Mapping[str, Any] | None = None) -> tuple[float | None, float | None]:
    """Intervalo Wilson sin redondear; solo ajusta ruido de borde <= 1e-15."""
    if type(successes) is not int or type(trials) is not int or successes < 0 or trials < successes: raise FeasibilityContractError("Wilson requiere cuentas enteras coherentes.")
    if trials == 0: return None, None
    rate = successes / trials; denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    lower, upper = centre - spread, centre + spread
    # Los extremos p=0 y p=1 son exactos matemáticamente; el cálculo binario
    # puede desplazarlos unas ulps (por ejemplo 0/3).
    if successes == 0: lower = 0.0
    if successes == trials: upper = 1.0
    if -WILSON_BOUNDARY_TOLERANCE <= lower <= WILSON_BOUNDARY_TOLERANCE: lower = 0.0
    if 1 - WILSON_BOUNDARY_TOLERANCE <= upper <= 1 + WILSON_BOUNDARY_TOLERANCE: upper = 1.0
    if lower > rate and lower - rate <= WILSON_BOUNDARY_TOLERANCE: lower = rate
    if upper < rate and rate - upper <= WILSON_BOUNDARY_TOLERANCE: upper = rate
    if not (math.isfinite(rate) and math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1 and lower - WILSON_BOUNDARY_TOLERANCE <= rate <= upper + WILSON_BOUNDARY_TOLERANCE):
        raise _wilson_failure(successes, trials, rate, lower, upper, context)
    return lower, upper


def _outcome_rows(part: pd.DataFrame, level: str, kind: str, number: int, surface: str, period: str, fold: str) -> list[dict[str, Any]]:
    rows = []
    specs = [("code", code, DOCUMENTED_SHOT_TYPES[code], SHOT_TYPE_FAMILIES[code], {code}) for code in TYPE_ORDER]
    specs += [("family", None, None, family, SHOT_TYPE_FAMILY_MEMBERS[family]) for family in FAMILY_ORDER]
    for aggregation, code, description, family, codes in specs:
        group = part.loc[part.shot_type_code.isin(codes)]; wins, total = int(group.returner_won_point.sum()), int(len(group)); lower, upper = wilson(wins, total, context={"aggregation_level": aggregation, "group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, "shot_type_code": code, "shot_type_family": family})
        rows.append({"aggregation_level": aggregation, "group_type": kind, "serve_number": number, "surface": surface, "derived_period": period, "validation_fold": fold, "shot_type_code": code, "shot_type_description": description, "shot_type_family": family, **_counts(group), "returner_wins": wins, "returner_losses": total - wins, "returner_win_rate": None if total == 0 else wins / total, "wilson_95_lower": lower, "wilson_95_upper": upper})
    return rows


def build_outcomes(attempts: pd.DataFrame) -> pd.DataFrame:
    observed = attempts.loc[attempts.classification_state.eq(ReturnShotTypeState.OBSERVED.value)]
    rows = []
    for kind, number, surface, period, fold, part in _group_specs(observed): rows.extend(_outcome_rows(part, "", kind, number, surface, period, fold))
    return _ordered(pd.DataFrame(rows, columns=OUTCOME_COLUMNS), ["aggregation_level", "group_type", "serve_number", "surface", "derived_period", "validation_fold", "shot_type_family", "shot_type_code"], {"aggregation_level": {"code": 0, "family": 1}, "group_type": {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}, "shot_type_family": {value: index for index, value in enumerate(FAMILY_ORDER)}, "shot_type_code": {value: index for index, value in enumerate(TYPE_ORDER)}})


def _test_seal(counts: Mapping[str, int]) -> dict[str, Any]:
    return {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": counts["excluded_test_matches"], **{field: 0 for field in TEST_ZERO_FIELDS}}


def _summary(counts: Mapping[str, int], attempts: pd.DataFrame, by_state: pd.DataFrame, by_type: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    state_counts = {state: int(attempts.classification_state.eq(state).sum()) for state in STATE_ORDER}
    observed = state_counts[ReturnShotTypeState.OBSERVED.value]
    type_counts = {code: int(attempts.shot_type_code.eq(code).sum()) for code in TYPE_ORDER}
    family_counts = {family: int(attempts.shot_type_family.eq(family).sum()) for family in FAMILY_ORDER}
    return {"analysis_name": ANALYSIS_NAME, "version": ANALYSIS_VERSION, "upstream_commit": UPSTREAM_COMMIT, "source": {"path": "data/processed/points_enriched.parquet", "columns": list(SOURCE_COLUMNS), "reads": 1}, "analysis_status": "available_descriptive", "reason_codes": sorted(set(attempts.reason_code)), "population": dict(counts), "attempts": {"total": int(len(attempts)), "first": int(attempts.serve_number.eq(1).sum()), "second": int(attempts.serve_number.eq(2).sum()), "cache_entries": int(attempts.attrs.get("cache_entries", 0))}, "state_counts": state_counts, "observed_type_counts": type_counts, "family_counts": family_counts, "coverage": {"observed_type_attempts": observed, "all_attempts": int(len(attempts)), "end_to_end_coverage": observed / len(attempts), "share_among_observed_types": {code: None if observed == 0 else type_counts[code] / observed for code in TYPE_ORDER}}, "outcomes_status": "available_descriptive" if observed else "not_available_no_observed_types", "test_seal": _test_seal(counts), "reconciliations": validate_reconciliations(attempts, by_state, by_type, by_group, outcomes), "artifact_contracts": {"by_state": BY_STATE_COLUMNS, "by_type": BY_TYPE_COLUMNS, "by_group": BY_GROUP_COLUMNS, "outcomes": OUTCOME_COLUMNS, "csv_index": False, "utf8": True, "fixed_order": True, "no_raw_sequences": True}, "fingerprint_contract": {"algorithm": "sha256", "serialization": "utf-8_deterministic"}, "methodological_limits": ["Análisis descriptivo y no causal.", "Los outcomes se condicionan a tipos de retorno documentados.", "q, unknown inicial y censura no son categorías tácticas comparables.", "Las familias son agregados secundarios fijados antes de observar frecuencias; t no es técnica homogénea.", "El test posterior a 2023 permanece sellado y no se clasifica."]}


def _assert_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if frame.columns.tolist() != list(columns) or any(str(value).startswith("Unnamed") for value in frame.columns): raise FeasibilityContractError(f"Schema u orden de {name} inválido.")


def _rate(actual: object, numerator: int, denominator: int, field: str) -> None:
    expected = None if denominator == 0 else numerator / denominator
    if expected is None:
        if actual is not None and not pd.isna(actual): raise FeasibilityContractError(f"{field} debe ser nulo.")
    elif not isinstance(actual, (int, float, np.floating)) or not math.isfinite(float(actual)) or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-15): raise FeasibilityContractError(f"{field} no reconcilia.")


def _validate_published_group_dimensions(groups: pd.DataFrame) -> None:
    """Cierra dominios y combinaciones permitidas sin reconstruir puntos."""
    allowed = {"total", "serve_number", "surface", "derived_period", "validation_fold"}
    if set(groups.group_type) - allowed:
        raise FeasibilityContractError("group_type persistido fuera del catálogo.")
    for row in groups.itertuples(index=False):
        if row.group_type == "total":
            valid = row.serve_number == 0 and row.surface == "ALL" and row.derived_period == "ALL" and row.validation_fold == "ALL"
        elif row.group_type == "serve_number":
            valid = row.serve_number in (1, 2) and row.surface == "ALL" and row.derived_period == "ALL" and row.validation_fold == "ALL"
        elif row.group_type == "surface":
            valid = row.serve_number == 0 and row.surface in ALLOWED_SURFACES and row.derived_period == "ALL" and row.validation_fold == "ALL"
        elif row.group_type == "derived_period":
            valid = row.serve_number == 0 and row.surface == "ALL" and row.derived_period in PERIOD_ORDER and row.validation_fold == "ALL"
        else:
            valid = row.serve_number == 0 and row.surface == "ALL" and row.derived_period == "ALL" and row.validation_fold in FOLD_ORDER
        if not valid:
            raise FeasibilityContractError("Dimensiones persistidas de grupo incompatibles.")
    expected = _ordered(groups, ["group_type", "serve_number", "surface", "derived_period", "validation_fold"], {"group_type": {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}})
    try:
        pdt.assert_frame_equal(groups.reset_index(drop=True), expected, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise FeasibilityContractError("Orden persistido by_group inválido.") from exc


def validate_reconciliations(attempts: pd.DataFrame, by_state: pd.DataFrame, by_type: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, bool]:
    for frame, columns, name in ((by_state, BY_STATE_COLUMNS, "by_state"), (by_type, BY_TYPE_COLUMNS, "by_type"), (by_group, BY_GROUP_COLUMNS, "by_group"), (outcomes, OUTCOME_COLUMNS, "outcomes")): _assert_columns(frame, columns, name)
    if bool(attempts.duplicated(["match_id", "point_number", "serve_number"]).any()): raise FeasibilityContractError("Clave de intento duplicada.")
    if set(attempts.classification_state) != set(STATE_ORDER): raise FeasibilityContractError("Estados P06 no exhaustivos.")
    if not attempts.eligible_for_type_comparison.eq(attempts.classification_state.eq(ReturnShotTypeState.OBSERVED.value)).all(): raise FeasibilityContractError("Elegibilidad incompatible con estado.")
    if bool(by_state.duplicated(["group_type", "serve_number", "classification_state", "reason_code"]).any()): raise FeasibilityContractError("Clave by_state duplicada.")
    total_state = by_state.loc[by_state.group_type.eq("total")]
    if int(total_state.attempts.sum()) != len(attempts) or not total_state.denominator_attempts.eq(len(attempts)).all(): raise FeasibilityContractError("Estados totales no reconcilian.")
    for row in total_state.itertuples(index=False): _rate(row.proportion_of_denominator, int(row.attempts), int(row.denominator_attempts), "by_state proportion")
    try:
        pdt.assert_frame_equal(by_state, build_by_state(attempts), check_dtype=False, check_exact=True)
        pdt.assert_frame_equal(by_type, build_by_type(attempts), check_dtype=False, check_exact=True)
        pdt.assert_frame_equal(by_group, build_by_group(attempts), check_dtype=False, check_exact=True)
        pdt.assert_frame_equal(outcomes, build_outcomes(attempts), check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise FeasibilityContractError("Agregados no reconcilian con los intentos.") from exc
    return {"attempt_key_unique": True, "states_exhaustive_exclusive": True, "codes_reconciled": True, "families_reconciled": True, "groups_reconciled": True, "outcomes_reconciled": True, "test_sealed": True}


def _table_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", encoding="utf-8", na_rep="", float_format="%.15g")
    return buffer.getvalue().encode("utf-8")


def _summary_bytes(summary: Mapping[str, Any]) -> bytes:
    return (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _fingerprint(summary: Mapping[str, Any], table_payloads: Sequence[bytes]) -> str:
    copy = dict(summary); copy.pop("publication_fingerprint", None)
    digest = hashlib.sha256(json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for payload in table_payloads: digest.update(payload)
    return digest.hexdigest().upper()


def _finalize(summary: dict[str, Any], attempts: pd.DataFrame, by_state: pd.DataFrame, by_type: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> DescriptiveResult:
    tables = (by_state, by_type, by_group, outcomes); payloads = tuple(_table_bytes(frame) for frame in tables); names = ("by_state", "by_type", "by_group", "outcomes")
    summary = dict(summary); summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, payloads)}; summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, payloads)}
    fingerprint = _fingerprint(summary, payloads); summary["publication_fingerprint"] = fingerprint
    return DescriptiveResult(summary, by_state, by_type, by_group, outcomes, attempts, fingerprint)


def analyze_points(points: pd.DataFrame, *, expected_population: Mapping[str, int] | None = None, extractor: Callable[..., ReturnShotTypeClassification] = parse_and_classify_initial_return_shot_type) -> DescriptiveResult:
    source = validate_source_points(points, expected_rows=None if expected_population is None else expected_population["source_rows_read"])
    development, counts = split_development_before_parsing(source)
    if expected_population is not None:
        for key, expected in expected_population.items():
            if key in counts and counts[key] != expected: raise FeasibilityContractError(f"{key} inesperado: {counts[key]}")
    attempts = construct_attempts(development, extractor=extractor)
    if expected_population is not None:
        actual_attempts = {"attempts_total": len(attempts), "first_attempts": int(attempts.serve_number.eq(1).sum()), "second_attempts": int(attempts.serve_number.eq(2).sum())}
        for key, actual in actual_attempts.items():
            if actual != expected_population[key]: raise FeasibilityContractError(f"{key} inesperado: {actual}")
    by_state, by_type, by_group, outcomes = build_by_state(attempts), build_by_type(attempts), build_by_group(attempts), build_outcomes(attempts)
    result = _finalize(_summary(counts, attempts, by_state, by_type, by_group, outcomes), attempts, by_state, by_type, by_group, outcomes)
    validate_result(result)
    return result


def not_available_result(error: BaseException, *, stage: str = "analyze_points") -> DescriptiveResult:
    message = str(error).replace("\n", " ").strip()[:240] or type(error).__name__
    empty = (pd.DataFrame(columns=BY_STATE_COLUMNS), pd.DataFrame(columns=BY_TYPE_COLUMNS), pd.DataFrame(columns=BY_GROUP_COLUMNS), pd.DataFrame(columns=OUTCOME_COLUMNS))
    summary = {"analysis_name": ANALYSIS_NAME, "version": ANALYSIS_VERSION, "upstream_commit": UPSTREAM_COMMIT, "source": {"path": "data/processed/points_enriched.parquet", "columns": list(SOURCE_COLUMNS), "reads": 1}, "analysis_status": "not_available", "reason_codes": ["execution_failed"], "failure": {"stage": stage, "type": type(error).__name__, "message": message}, "population": {}, "attempts": {}, "state_counts": {}, "observed_type_counts": {}, "family_counts": {}, "coverage": {}, "outcomes_status": "not_available", "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": None, **{field: 0 for field in TEST_ZERO_FIELDS}}, "reconciliations": {}, "artifact_contracts": {"by_state": BY_STATE_COLUMNS, "by_type": BY_TYPE_COLUMNS, "by_group": BY_GROUP_COLUMNS, "outcomes": OUTCOME_COLUMNS, "csv_index": False, "utf8": True, "fixed_order": True, "no_raw_sequences": True}, "fingerprint_contract": {"algorithm": "sha256", "serialization": "utf-8_deterministic"}, "methodological_limits": ["Análisis no disponible; no se publican agregados parciales."]}
    return _finalize(summary, pd.DataFrame(), *empty)


def validate_result(result: DescriptiveResult) -> None:
    if not isinstance(result, DescriptiveResult): raise TypeError("result debe ser DescriptiveResult.")
    for frame, columns, name in ((result.by_state, BY_STATE_COLUMNS, "by_state"), (result.by_type, BY_TYPE_COLUMNS, "by_type"), (result.by_group, BY_GROUP_COLUMNS, "by_group"), (result.outcomes, OUTCOME_COLUMNS, "outcomes")): _assert_columns(frame, columns, name)
    payloads = tuple(_table_bytes(frame) for frame in (result.by_state, result.by_type, result.by_group, result.outcomes)); names = ("by_state", "by_type", "by_group", "outcomes")
    if result.summary.get("artifact_payload_sha256") != {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, payloads)} or result.summary.get("artifact_payload_bytes") != {name: len(payload) for name, payload in zip(names, payloads)}: raise FeasibilityContractError("Hashes de artefactos inválidos.")
    if result.publication_fingerprint != result.summary.get("publication_fingerprint") or result.publication_fingerprint != _fingerprint(result.summary, payloads): raise FeasibilityContractError("Fingerprint inválido.")
    if result.summary.get("analysis_status") not in {"available_descriptive", "not_available"}: raise FeasibilityContractError("analysis_status inválido.")
    if result.summary["analysis_status"] == "not_available":
        if any(not frame.empty for frame in (result.by_state, result.by_type, result.by_group, result.outcomes)): raise FeasibilityContractError("not_available no publica filas.")
        failure = result.summary.get("failure", {})
        if failure.get("stage") not in {"read_source_points", "analyze_points"} or not isinstance(failure.get("type"), str) or not failure["type"] or not isinstance(failure.get("message"), str) or not failure["message"]:
            raise FeasibilityContractError("not_available debe conservar un fallo sanitizado y etiquetado.")
        return
    validate_reconciliations(result.attempts, result.by_state, result.by_type, result.by_group, result.outcomes)
    summary = result.summary; expected_states = {state: int(result.attempts.classification_state.eq(state).sum()) for state in STATE_ORDER}; expected_types = {code: int(result.attempts.shot_type_code.eq(code).sum()) for code in TYPE_ORDER}; expected_families = {family: int(result.attempts.shot_type_family.eq(family).sum()) for family in FAMILY_ORDER}
    if summary.get("state_counts") != expected_states or summary.get("observed_type_counts") != expected_types or summary.get("family_counts") != expected_families: raise FeasibilityContractError("Resumen de estados, tipos o familias no reconcilia.")
    observed = expected_states[ReturnShotTypeState.OBSERVED.value]
    if summary.get("coverage", {}).get("observed_type_attempts") != observed or summary["coverage"].get("all_attempts") != len(result.attempts): raise FeasibilityContractError("Cobertura no reconcilia.")
    _rate(summary["coverage"].get("end_to_end_coverage"), observed, len(result.attempts), "summary coverage")
    seal = summary.get("test_seal", {})
    if set(seal) != {"test_status", "used_for_method_selection", "excluded_test_matches", *TEST_ZERO_FIELDS} or seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or seal.get("excluded_test_matches") != summary.get("population", {}).get("excluded_test_matches") or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS): raise FeasibilityContractError("Test seal inválido.")


def serialize_artifacts(result: DescriptiveResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    validate_result(result)
    return (_summary_bytes(result.summary), _table_bytes(result.by_state), _table_bytes(result.by_type), _table_bytes(result.by_group), _table_bytes(result.outcomes))


def _stage(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    finally:
        handle.close()
    return Path(handle.name)


def write_artifacts(result: DescriptiveResult, *, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_type_path: Path = BY_TYPE_PATH, by_group_path: Path = BY_GROUP_PATH, outcomes_path: Path = OUTCOMES_PATH, payloads: tuple[bytes, bytes, bytes, bytes, bytes] | None = None) -> None:
    expected = serialize_artifacts(result)
    if serialize_artifacts(result) != expected: raise FeasibilityContractError("Serialización no determinista.")
    if payloads is not None and payloads != expected: raise FeasibilityContractError("Payloads no derivados del resultado validado.")
    destinations, staged, backups, replaced = (summary_path, by_state_path, by_type_path, by_group_path, outcomes_path), [], {}, []
    try:
        for path, payload in zip(destinations, expected): staged.append(_stage(path, payload))
        backups = {path: path.read_bytes() for path in destinations if path.exists()}
        for source, target in zip(staged, destinations): os.replace(source, target); replaced.append(target)
        verify_persisted_artifacts(summary_path=summary_path, by_state_path=by_state_path, by_type_path=by_type_path, by_group_path=by_group_path, outcomes_path=outcomes_path)
    except BaseException:
        for target in reversed(replaced):
            if target in backups:
                rollback = _stage(target, backups[target]); os.replace(rollback, target)
            elif target.exists(): target.unlink()
        raise
    finally:
        for path in staged:
            if path.exists(): path.unlink()


def _read_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    _assert_columns(frame, columns, path.name)
    return frame


def verify_persisted_artifacts(*, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_type_path: Path = BY_TYPE_PATH, by_group_path: Path = BY_GROUP_PATH, outcomes_path: Path = OUTCOMES_PATH) -> None:
    payloads = (summary_path.read_bytes(), by_state_path.read_bytes(), by_type_path.read_bytes(), by_group_path.read_bytes(), outcomes_path.read_bytes())
    if any(token in payload for payload in payloads for token in (b"NaN", b"Infinity", b"-Infinity")):
        raise FeasibilityContractError("Los artefactos persistidos contienen valores no finitos literales.")
    summary = json.loads(payloads[0].decode("utf-8")); frames = (_read_csv(by_state_path, BY_STATE_COLUMNS), _read_csv(by_type_path, BY_TYPE_COLUMNS), _read_csv(by_group_path, BY_GROUP_COLUMNS), _read_csv(outcomes_path, OUTCOME_COLUMNS)); names = ("by_state", "by_type", "by_group", "outcomes")
    expected_hashes = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, payloads[1:])}; expected_sizes = {name: len(payload) for name, payload in zip(names, payloads[1:])}
    if summary.get("artifact_payload_sha256") != expected_hashes or summary.get("artifact_payload_bytes") != expected_sizes or summary.get("publication_fingerprint") != _fingerprint(summary, payloads[1:]): raise FeasibilityContractError("Fingerprint o hashes persistidos inválidos.")
    if summary.get("analysis_status") == "not_available":
        if any(not frame.empty for frame in frames): raise FeasibilityContractError("not_available persistido publica filas.")
        failure = summary.get("failure", {})
        if failure.get("stage") not in {"read_source_points", "analyze_points"} or not isinstance(failure.get("type"), str) or not failure["type"] or not isinstance(failure.get("message"), str) or not failure["message"]:
            raise FeasibilityContractError("not_available persistido no conserva el fallo.")
        return
    if summary.get("analysis_status") != "available_descriptive": raise FeasibilityContractError("analysis_status persistido inválido.")
    state, types, groups, outcomes = frames
    if bool(state.duplicated(["group_type", "serve_number", "classification_state", "reason_code"]).any()) or bool(types.duplicated(["aggregation_level", "serve_number", "shot_type_code", "shot_type_family"]).any()) or bool(groups.duplicated(["group_type", "serve_number", "surface", "derived_period", "validation_fold"]).any()) or bool(outcomes.duplicated(["aggregation_level", "group_type", "serve_number", "surface", "derived_period", "validation_fold", "shot_type_code", "shot_type_family"]).any()): raise FeasibilityContractError("Clave persistida duplicada.")
    if set(state.group_type) - {"total", "serve_number"} or set(state.classification_state) != set(STATE_ORDER):
        raise FeasibilityContractError("Dominios persistidos by_state inválidos.")
    for row in state.itertuples(index=False):
        if row.reason_code not in STATE_REASON_CODES[str(row.classification_state)]:
            raise FeasibilityContractError("reason_code persistido incompatible con estado.")
        if (row.group_type == "total" and row.serve_number != 0) or (row.group_type == "serve_number" and row.serve_number not in (1, 2)):
            raise FeasibilityContractError("Dimensiones persistidas by_state incompatibles.")
    state_order = _ordered(state, ["group_type", "serve_number", "classification_state", "reason_code"], {"group_type": {"total": 0, "serve_number": 1}, "classification_state": {value: index for index, value in enumerate(STATE_ORDER)}})
    try:
        pdt.assert_frame_equal(state.reset_index(drop=True), state_order, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise FeasibilityContractError("Orden persistido by_state inválido.") from exc
    total_state = state.loc[state.group_type.eq("total")]
    total_attempts = int(total_state.attempts.sum())
    if total_attempts != int(summary.get("attempts", {}).get("total", -1)) or set(state.classification_state) - set(STATE_ORDER): raise FeasibilityContractError("Estados persistidos inválidos.")
    for row in total_state.itertuples(index=False): _rate(float(row.proportion_of_denominator), int(row.attempts), int(row.denominator_attempts), "state persistido")
    persisted_states = {key: int(total_state.loc[total_state.classification_state.eq(key), "attempts"].sum()) for key in STATE_ORDER}
    if summary.get("state_counts") != persisted_states or sum(persisted_states.values()) != total_attempts:
        raise FeasibilityContractError("state_counts persistido no reconcilia.")
    total_codes = types.loc[types.aggregation_level.eq("code") & types.serve_number.eq(0)]
    total_families = types.loc[types.aggregation_level.eq("family") & types.serve_number.eq(0)]
    if set(total_codes.shot_type_code) != set(TYPE_ORDER) or set(total_families.shot_type_family) != set(FAMILY_ORDER): raise FeasibilityContractError("Catálogo persistido de tipos o familias inválido.")
    observed = int(summary["state_counts"][ReturnShotTypeState.OBSERVED.value])
    if int(total_codes.attempts.sum()) != observed or int(total_families.attempts.sum()) != observed: raise FeasibilityContractError("Tipos/familias persistidos no reconcilian.")
    observed_reasons = {
        f"documented_return_shot_type_{code}": int(total_codes.loc[total_codes.shot_type_code.eq(code), "attempts"].iloc[0])
        for code in TYPE_ORDER
        if int(total_codes.loc[total_codes.shot_type_code.eq(code), "attempts"].iloc[0]) > 0
    }
    actual_observed_reasons = {
        str(row.reason_code): int(row.attempts)
        for row in total_state.loc[total_state.classification_state.eq(ReturnShotTypeState.OBSERVED.value)].itertuples(index=False)
    }
    if actual_observed_reasons != observed_reasons:
        raise FeasibilityContractError("Estados observados persistidos no reconcilian con codigos.")
    unknown_state = total_state.loc[total_state.classification_state.eq(ReturnShotTypeState.UNKNOWN.value)]
    if len(unknown_state) != 1 or unknown_state.iloc[0].reason_code != ReturnShotTypeReason.Q.value:
        raise FeasibilityContractError("q debe permanecer separado como tipo desconocido.")
    expected_type_keys = {
        ("code", number, code, SHOT_TYPE_FAMILIES[code])
        for number in (0, 1, 2)
        for code in TYPE_ORDER
    } | {
        ("family", number, "", family)
        for number in (0, 1, 2)
        for family in FAMILY_ORDER
    }
    actual_type_keys = {(str(row.aggregation_level), int(row.serve_number), str(row.shot_type_code), str(row.shot_type_family)) for row in types.itertuples(index=False)}
    if actual_type_keys != expected_type_keys or len(types) != len(expected_type_keys): raise FeasibilityContractError("Claves persistidas by_type inválidas.")
    for row in types.itertuples(index=False):
        if row.aggregation_level == "code":
            if row.shot_type_code not in DOCUMENTED_SHOT_TYPES or row.shot_type_description != DOCUMENTED_SHOT_TYPES[row.shot_type_code] or row.shot_type_family != SHOT_TYPE_FAMILIES[row.shot_type_code]: raise FeasibilityContractError("Código, descripción o familia persistida inválida.")
        elif row.shot_type_code != "" or row.shot_type_description != "" or row.shot_type_family not in FAMILY_ORDER: raise FeasibilityContractError("Fila familia persistida inválida.")
        _rate(row.share_among_observed_types, int(row.attempts), int(row.observed_type_denominator), "by_type share")
        _rate(row.end_to_end_coverage, int(row.attempts), int(row.end_to_end_denominator), "by_type coverage")
    for number in (0, 1, 2):
        part_codes = types.loc[types.aggregation_level.eq("code") & types.serve_number.eq(number)]
        part_families = types.loc[types.aggregation_level.eq("family") & types.serve_number.eq(number)]
        if int(part_codes.attempts.sum()) != int(part_families.attempts.sum()) or not part_codes.observed_type_denominator.nunique() == 1 or not part_families.observed_type_denominator.nunique() == 1 or int(part_codes.observed_type_denominator.iloc[0]) != int(part_families.observed_type_denominator.iloc[0]): raise FeasibilityContractError("Denominadores by_type persistidos inválidos.")
    for family, codes in SHOT_TYPE_FAMILY_MEMBERS.items():
        family_total = int(total_families.loc[total_families.shot_type_family.eq(family), "attempts"].iloc[0]); code_total = int(total_codes.loc[total_codes.shot_type_code.isin(codes), "attempts"].sum())
        if family_total != code_total: raise FeasibilityContractError("Familia persistida no deriva de códigos.")
    expected_type_order = _ordered(
        types,
        ["aggregation_level", "serve_number", "shot_type_family", "shot_type_code"],
        {
            "aggregation_level": {"code": 0, "family": 1},
            "shot_type_family": {value: index for index, value in enumerate(FAMILY_ORDER)},
            "shot_type_code": {value: index for index, value in enumerate(TYPE_ORDER)},
        },
    )
    try:
        pdt.assert_frame_equal(types.reset_index(drop=True), expected_type_order, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise FeasibilityContractError("Orden persistido by_type inválido.") from exc
    total_group = groups.loc[groups.group_type.eq("total")]
    _validate_published_group_dimensions(groups)
    if len(total_group) != 1 or int(total_group.iloc[0].attempts) != total_attempts or int(total_group.iloc[0].observed_type_attempts) != observed: raise FeasibilityContractError("Grupo total persistido no reconcilia.")
    if not groups.observed_type_attempts.add(groups.unknown_type_attempts).add(groups.unknown_initial_attempts).add(groups.censored_attempts).eq(groups.attempts).all(): raise FeasibilityContractError("Estados de grupo persistidos no reconcilian.")
    for row in groups.itertuples(index=False): _rate(row.observed_type_coverage, int(row.observed_type_numerator), int(row.observed_type_denominator), "group coverage")
    expected_group_sets = {
        "total": {(0, "ALL", "ALL", "ALL")},
        "serve_number": {(1, "ALL", "ALL", "ALL"), (2, "ALL", "ALL", "ALL")},
        "surface": {(0, surface, "ALL", "ALL") for surface in ALLOWED_SURFACES},
        "derived_period": {(0, "ALL", period, "ALL") for period in PERIOD_ORDER},
        "validation_fold": {(0, "ALL", "ALL", fold) for fold in FOLD_ORDER},
    }
    enforce_frozen_groups = summary.get("population", {}).get("source_rows_read") == EXPECTED_REAL["source_rows_read"]
    for group_type, expected_keys in expected_group_sets.items():
        actual_keys = {(int(row.serve_number), str(row.surface), str(row.derived_period), str(row.validation_fold)) for row in groups.loc[groups.group_type.eq(group_type)].itertuples(index=False)}
        if enforce_frozen_groups and actual_keys != expected_keys:
            raise FeasibilityContractError("Conjunto persistido de grupos incompleto o adicional.")
        partition = groups.loc[groups.group_type.eq(group_type)]
        if any(int(partition[column].sum()) != int(total_group.iloc[0][column]) for column in ("attempts", "observed_type_attempts", "unknown_type_attempts", "unknown_initial_attempts", "censored_attempts")):
            raise FeasibilityContractError("Particion persistida de grupos no reconcilia con el total.")
    for number in (0, 1, 2):
        group_row = total_group.iloc[0] if number == 0 else groups.loc[groups.group_type.eq("serve_number") & groups.serve_number.eq(number)].iloc[0]
        code_rows = types.loc[types.aggregation_level.eq("code") & types.serve_number.eq(number)]
        if not code_rows.observed_type_denominator.eq(int(group_row.observed_type_attempts)).all() or not code_rows.end_to_end_denominator.eq(int(group_row.attempts)).all():
            raise FeasibilityContractError("Denominadores by_type no reconcilian con grupos.")
        state_rows = total_state if number == 0 else state.loc[state.group_type.eq("serve_number") & state.serve_number.eq(number)]
        state_counts = state_rows.groupby("classification_state", sort=False).attempts.sum().to_dict()
        expected_counts = {
            ReturnShotTypeState.OBSERVED.value: int(group_row.observed_type_attempts),
            ReturnShotTypeState.UNKNOWN.value: int(group_row.unknown_type_attempts),
            ReturnShotTypeState.UNKNOWN_INITIAL.value: int(group_row.unknown_initial_attempts),
            ReturnShotTypeState.CENSORED.value: int(group_row.censored_attempts),
        }
        if {key: int(state_counts.get(key, 0)) for key in STATE_ORDER} != expected_counts:
            raise FeasibilityContractError("Estados por saque no reconcilian con grupos.")
    population = summary.get("population", {})
    if population.get("development_matches") != int(total_group.iloc[0].matches) or population.get("development_servers") != int(total_group.iloc[0].servers) or population.get("development_point_rows") != int(summary["attempts"].get("first", -1)):
        raise FeasibilityContractError("Población persistida no reconcilia con agregados.")
    if int(summary["attempts"].get("first", -1)) + int(summary["attempts"].get("second", -1)) != total_attempts:
        raise FeasibilityContractError("Primeros y segundos persistidos no reconcilian.")
    observed_type_counts = {code: int(total_codes.loc[total_codes.shot_type_code.eq(code), "attempts"].iloc[0]) for code in TYPE_ORDER}
    family_counts = {family: int(total_families.loc[total_families.shot_type_family.eq(family), "attempts"].iloc[0]) for family in FAMILY_ORDER}
    coverage = summary.get("coverage", {})
    if summary.get("observed_type_counts") != observed_type_counts or summary.get("family_counts") != family_counts:
        raise FeasibilityContractError("Resumen persistido de tipos o familias no reconcilia.")
    if coverage.get("observed_type_attempts") != observed or coverage.get("all_attempts") != total_attempts:
        raise FeasibilityContractError("Cobertura persistida no reconcilia.")
    _rate(coverage.get("end_to_end_coverage"), observed, total_attempts, "summary coverage")
    expected_shares = {code: None if observed == 0 else observed_type_counts[code] / observed for code in TYPE_ORDER}
    if coverage.get("share_among_observed_types") != expected_shares:
        raise FeasibilityContractError("Shares persistidos no reconcilian.")
    if set(outcomes.aggregation_level) != {"code", "family"} or not outcomes.returner_wins.add(outcomes.returner_losses).eq(outcomes.attempts).all(): raise FeasibilityContractError("Outcomes persistidos inválidos.")
    populated = outcomes.loc[outcomes.attempts.astype(int).gt(0)]
    for row in populated.itertuples(index=False):
        _rate(float(row.returner_win_rate), int(row.returner_wins), int(row.attempts), "outcome rate")
        lower, upper = wilson(int(row.returner_wins), int(row.attempts))
        if not (math.isclose(float(row.wilson_95_lower), float(lower), rel_tol=0, abs_tol=1e-15) and math.isclose(float(row.wilson_95_upper), float(upper), rel_tol=0, abs_tol=1e-15)):
            raise FeasibilityContractError("Wilson persistido no reconcilia.")
    empty_outcomes = outcomes.loc[outcomes.attempts.astype(int).eq(0)]
    for row in empty_outcomes.itertuples(index=False):
        if int(row.returner_wins) != 0 or int(row.returner_losses) != 0 or any(value != "" and not pd.isna(value) for value in (row.returner_win_rate, row.wilson_95_lower, row.wilson_95_upper)):
            raise FeasibilityContractError("Outcome vacío persistido contiene inferencia.")
    total_outcome_codes = outcomes.loc[outcomes.aggregation_level.eq("code") & outcomes.group_type.eq("total")]
    if set(total_outcome_codes.shot_type_code) != set(TYPE_ORDER) or {code: int(total_outcome_codes.loc[total_outcome_codes.shot_type_code.eq(code), "attempts"].iloc[0]) for code in TYPE_ORDER} != {code: int(total_codes.loc[total_codes.shot_type_code.eq(code), "attempts"].iloc[0]) for code in TYPE_ORDER}:
        raise FeasibilityContractError("Outcomes por código no reconcilian con by_type.")
    expected_outcome_keys = {
        (level, str(row.group_type), int(row.serve_number), str(row.surface), str(row.derived_period), str(row.validation_fold), code, family)
        for row in groups.loc[groups.observed_type_attempts.astype(int).gt(0)].itertuples(index=False)
        for level, code, family in [
            *[("code", code, SHOT_TYPE_FAMILIES[code]) for code in TYPE_ORDER],
            *[("family", "", family) for family in FAMILY_ORDER],
        ]
    }
    actual_outcome_keys = {
        (str(row.aggregation_level), str(row.group_type), int(row.serve_number), str(row.surface), str(row.derived_period), str(row.validation_fold), str(row.shot_type_code), str(row.shot_type_family))
        for row in outcomes.itertuples(index=False)
    }
    if actual_outcome_keys != expected_outcome_keys or len(outcomes) != len(expected_outcome_keys):
        raise FeasibilityContractError("Claves persistidas outcomes inválidas.")
    for group in groups.loc[groups.observed_type_attempts.astype(int).gt(0)].itertuples(index=False):
        mask = (outcomes.group_type.eq(group.group_type) & outcomes.serve_number.eq(group.serve_number) & outcomes.surface.eq(group.surface) & outcomes.derived_period.eq(group.derived_period) & outcomes.validation_fold.eq(group.validation_fold))
        group_outcomes = outcomes.loc[mask]
        code_outcomes = group_outcomes.loc[group_outcomes.aggregation_level.eq("code")]
        family_outcomes = group_outcomes.loc[group_outcomes.aggregation_level.eq("family")]
        if int(code_outcomes.attempts.sum()) != int(group.observed_type_attempts) or int(family_outcomes.attempts.sum()) != int(group.observed_type_attempts) or int(code_outcomes.returner_wins.sum()) != int(family_outcomes.returner_wins.sum()) or int(code_outcomes.returner_losses.sum()) != int(family_outcomes.returner_losses.sum()):
            raise FeasibilityContractError("Outcomes persistidos no reconcilian con el grupo.")
        for family, codes in SHOT_TYPE_FAMILY_MEMBERS.items():
            family_row = family_outcomes.loc[family_outcomes.shot_type_family.eq(family)].iloc[0]
            code_rows = code_outcomes.loc[code_outcomes.shot_type_code.isin(codes)]
            if any(int(family_row[column]) != int(code_rows[column].sum()) for column in ("attempts", "returner_wins", "returner_losses")):
                raise FeasibilityContractError("Outcome de familia persistido no deriva de sus codigos.")
    if summary.get("reason_codes") != sorted(set(state.reason_code)):
        raise FeasibilityContractError("reason_codes persistidos no reconcilian.")
    if summary.get("source") != {"path": "data/processed/points_enriched.parquet", "columns": list(SOURCE_COLUMNS), "reads": 1}:
        raise FeasibilityContractError("Contrato persistido de fuente invalido.")
    seal = summary.get("test_seal", {})
    if set(seal) != {"test_status", "used_for_method_selection", "excluded_test_matches", *TEST_ZERO_FIELDS} or seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or seal.get("excluded_test_matches") != population.get("excluded_test_matches") or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS): raise FeasibilityContractError("Test seal persistido inválido.")


def read_source_points() -> pd.DataFrame:
    return pd.read_parquet(POINTS_PATH, columns=list(SOURCE_COLUMNS))


def run_real_analysis() -> DescriptiveResult:
    try:
        source = read_source_points()
    except Exception as error:
        raise _RunFailure("read_source_points", error) from error
    try:
        return analyze_points(source, expected_population=EXPECTED_REAL)
    except Exception as error:
        raise _RunFailure("analyze_points", error) from error


def _performance_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    try: path.relative_to(ROOT)
    except ValueError: return path
    raise FeasibilityContractError("--performance-log debe estar fuera del repositorio.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P06: tipo documentado del primer resto.")
    parser.add_argument("--performance-log", required=True)
    args = parser.parse_args(argv)
    performance_path = _performance_path(args.performance_log)
    started = time.perf_counter(); error: Exception | None = None
    try:
        result = run_real_analysis()
    except _RunFailure as failure:
        error = failure.cause
        result = not_available_result(failure.cause, stage=failure.stage)
    write_artifacts(result)
    elapsed = time.perf_counter() - started
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    performance_path.write_text(json.dumps({"analysis_name": ANALYSIS_NAME, "duration_seconds": elapsed, "analysis_status": result.summary["analysis_status"]}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
