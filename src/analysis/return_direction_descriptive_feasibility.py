"""P04: estudio descriptivo sellado de la direccion lateral del primer resto.

La unidad es un intento de saque sustantivo. El test posterior a 2023 se lee
solo para sellarlo y nunca se transforma en intentos ni se entrega al extractor.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.first_serve_direction_feasibility import validate_chronological_contract
from src.analysis.return_direction_feasibility import (
    LATERAL_DIRECTIONS,
    RETURN_DEPTHS,
    SHOT_TYPES,
    ReturnDirectionClassification,
    ReturnDirectionReason,
    ReturnDirectionState,
    parse_and_classify_initial_return_direction,
    validate_return_direction_classification,
)

ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "return_direction_descriptive_feasibility_summary.json"
BY_STATE_PATH = TABLES_DIR / "return_direction_descriptive_feasibility_by_state.csv"
BY_DIRECTION_PATH = TABLES_DIR / "return_direction_descriptive_feasibility_by_direction.csv"
BY_GROUP_PATH = TABLES_DIR / "return_direction_descriptive_feasibility_by_group.csv"
OUTCOMES_PATH = TABLES_DIR / "return_direction_descriptive_feasibility_outcomes.csv"

ANALYSIS_NAME = "return_direction_descriptive_feasibility"
ANALYSIS_VERSION = "1.0.0"
FINGERPRINT_CONTRACT_VERSION = "1"
CUTOFF = pd.Timestamp("2023-12-31")
ALLOWED_SURFACES = ("Hard", "Clay", "Grass")
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
FOLD_ORDER = ("pre_validation", "validation_2020", "validation_2021", "validation_2022", "validation_2023")
STATE_ORDER = tuple(state.value for state in ReturnDirectionState)
DIRECTION_ORDER = ("1", "2", "3", "0")
SOURCE_COLUMNS = ["match_id", "point_number", "date", "surface", "server", "point_winner", "player_1", "player_2", "first_serve", "second_serve"]
BY_STATE_COLUMNS = ["serve_number", "classification_state", "reason_code", "attempts", "matches", "servers", "returners", "return_opportunities", "return_events", "eligible_directions", "share_of_attempts"]
BY_DIRECTION_COLUMNS = ["dimension", "serve_number", "code", "label", "attempts", "matches", "servers", "returners", "share_of_dimension"]
BY_GROUP_COLUMNS = ["dimension", "serve_number", "surface", "derived_period", "validation_fold", "attempts", "return_opportunities", "return_events", "eligible_directions", "opportunity_numerator", "opportunity_denominator", "opportunity_rate", "event_numerator", "event_denominator", "event_rate", "direction_numerator", "direction_denominator", "direction_rate", "end_to_end_numerator", "end_to_end_denominator", "end_to_end_rate"]
OUTCOME_COLUMNS = ["stratum", "serve_number", "surface", "derived_period", "lateral_direction_code", "lateral_direction_label", "attempts", "matches", "returners", "servers", "returner_point_wins", "returner_point_win_rate", "wilson_95_lower", "wilson_95_upper", "server_point_wins", "server_point_win_rate"]
TEST_ZERO_FIELDS = (
    "test_target_rows_parsed", "test_attempts_constructed", "test_returns_classified",
    "test_scores_computed", "test_outcomes_computed", "test_rows_evaluated",
    "test_matches_evaluated", "test_evaluation_runs", "test_recommendations_generated",
)
EXPECTED = {"source_rows_read": 1_280_408, "source_matches": 7_524, "source_players": 1_002, "development_point_rows": 1_035_760, "development_matches": 5_993, "development_servers": 870, "excluded_test_target_matches": 1_531}
STATE_REASONS = {
    state.value: {reason.value for reason in ReturnDirectionReason if reason in reasons}
    for state, reasons in {
        ReturnDirectionState.OBSERVED: {ReturnDirectionReason.D1, ReturnDirectionReason.D2, ReturnDirectionReason.D3},
        ReturnDirectionState.UNKNOWN: {ReturnDirectionReason.D0, ReturnDirectionReason.NO_DIRECTION},
        ReturnDirectionState.UNKNOWN_INITIAL: {ReturnDirectionReason.MISSING_PREFIX, ReturnDirectionReason.BOUNDARY, ReturnDirectionReason.MODIFIER, ReturnDirectionReason.UNKNOWN_SHOT, ReturnDirectionReason.TRUNCATED, ReturnDirectionReason.INCONSISTENT},
        ReturnDirectionState.ACE: {ReturnDirectionReason.ACE},
        ReturnDirectionState.UNRETURNED: {ReturnDirectionReason.UNRETURNED},
        ReturnDirectionState.FAULT: {ReturnDirectionReason.FAULT, ReturnDirectionReason.DOUBLE_FAULT},
        ReturnDirectionState.SPECIAL: {ReturnDirectionReason.SPECIAL, ReturnDirectionReason.INCOMPLETE_LET, ReturnDirectionReason.CHALLENGE_PENALTY},
    }.items()
}
LATERAL_LABELS = dict(LATERAL_DIRECTIONS)
DEPTH_LABELS = dict(RETURN_DEPTHS)
SHOT_LABELS = dict(SHOT_TYPES)
RATE_TOLERANCE = 1e-12


class FeasibilityContractError(ValueError):
    """Violacion de fuente, sellado, clasificacion o publicacion P04."""


@dataclass(frozen=True)
class DescriptiveResult:
    summary: dict[str, Any]
    by_state: pd.DataFrame
    by_direction: pd.DataFrame
    by_group: pd.DataFrame
    outcomes: pd.DataFrame
    attempts: pd.DataFrame
    publication_fingerprint: str


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value not in (1, 2):
        raise FeasibilityContractError(f"{field} debe ser exactamente 1 o 2; recibido {value!r}.")
    return int(value)


def _text(series: pd.Series, field: str) -> None:
    invalid = series.map(lambda x: not isinstance(x, str) or not x or x != x.strip())
    if invalid.any():
        raise FeasibilityContractError(f"{field} debe ser texto no vacio y sin espacios externos.")


def _dates(series: pd.Series) -> pd.Series:
    if series.isna().any(): raise FeasibilityContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif series.map(lambda x: isinstance(x, str)).all():
        if not series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(): raise FeasibilityContractError("date debe ser YYYY-MM-DD.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif series.map(lambda x: isinstance(x, (pd.Timestamp, datetime, date)) and not isinstance(x, bool)).all():
        parsed = pd.to_datetime(series, errors="coerce")
    else: raise FeasibilityContractError("date contiene tipos ambiguos.")
    if parsed.isna().any() or getattr(parsed.dt, "tz", None) is not None: raise FeasibilityContractError("date invalida o con zona horaria.")
    return parsed.dt.normalize()


def _presence(value: object, field: str) -> str:
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)): return "null"
    if not isinstance(value, str): raise FeasibilityContractError(f"{field} debe ser texto o nulo.")
    if value == "": return "empty"
    if value.isspace(): return "whitespace_only"
    return "substantive"


def derive_period(year: int) -> str:
    if year <= 2009: return "to_2009"
    if year <= 2019: return "2010s"
    return "2020s"


def validation_fold(day: pd.Timestamp) -> str:
    if day.year <= 2019: return "pre_validation"
    value = f"validation_{day.year}"
    if value not in FOLD_ORDER: raise FeasibilityContractError("Fecha de desarrollo fuera del protocolo de folds.")
    return value


def validate_source_points(points: pd.DataFrame, *, expected_rows: int | None = None) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing: raise FeasibilityContractError(f"Faltan columnas: {missing}")
    if expected_rows is not None and len(points) != expected_rows: raise FeasibilityContractError(f"source_rows_read inesperado: {len(points)}")
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"): _text(work[field], field)
    if work["point_number"].isna().any() or work.duplicated(["match_id", "point_number"]).any(): raise FeasibilityContractError("(match_id, point_number) debe ser clave unica no nula.")
    if work["point_number"].map(lambda x: isinstance(x, bool) or (isinstance(x, str) and (not x or x != x.strip()))).any(): raise FeasibilityContractError("point_number invalido.")
    work["date"] = _dates(work["date"])
    if not bool(work["surface"].isin(ALLOWED_SURFACES).all()): raise FeasibilityContractError(f"Superficies inesperadas: {sorted(work.loc[~work.surface.isin(ALLOWED_SURFACES), 'surface'].unique())}")
    if work.player_1.eq(work.player_2).any(): raise FeasibilityContractError("player_1 y player_2 deben ser distintos.")
    for field in ("server", "point_winner"): work[field] = work[field].map(lambda x: _strict_index(x, field))
    metadata = ["date", "surface", "player_1", "player_2"]
    if work.groupby("match_id", sort=False)[metadata].nunique(dropna=False).gt(1).any().any(): raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    return work.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)


def split_development_before_parsing(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    development = source.loc[source.date.le(CUTOFF)].copy()
    sealed = source.loc[source.date.gt(CUTOFF)]
    if development.empty or sealed.empty or development.date.gt(CUTOFF).any(): raise FeasibilityContractError("El sellado temporal no particiona la fuente.")
    counts = {
        "source_rows_read": int(len(source)), "source_matches": int(source.match_id.nunique()),
        "source_players": int(len(set(source.player_1) | set(source.player_2))),
        "development_point_rows": int(len(development)), "development_matches": int(development.match_id.nunique()),
        "development_servers": int(pd.Series(np.where(development.server.eq(1), development.player_1, development.player_2)).nunique()),
        "excluded_test_target_matches": int(sealed.match_id.nunique()),
    }
    return development.reset_index(drop=True), counts


def _attempt_row(row: Any, sequence: str, number: int, prior_fault: bool, classification: ReturnDirectionClassification) -> dict[str, Any]:
    validate_return_direction_classification(classification)
    server_player = row.player_1 if row.server == 1 else row.player_2
    returner_player = row.player_2 if row.server == 1 else row.player_1
    return {
        "match_id": row.match_id, "point_number": row.point_number, "serve_number": number,
        "date": row.date, "surface": row.surface, "derived_period": derive_period(int(row.date.year)),
        "validation_fold": validation_fold(row.date), "server_player": server_player,
        "returner_player": returner_player, "sequence_text": sequence,
        "previous_attempt_was_fault": prior_fault, "server_won_point": bool(row.point_winner == row.server),
        "returner_won_point": bool(row.point_winner != row.server),
        "classification_state": classification.classification_state.value, "reason_code": classification.reason_code.value,
        "return_opportunity_observed": classification.return_opportunity_observed,
        "return_event_observed": classification.return_event_observed,
        "eligible_for_direction_analysis": classification.eligible_for_direction_analysis,
        "lateral_direction_code": classification.lateral_direction_code,
        "lateral_direction_label": classification.lateral_direction_label,
        "return_shot_type_code": classification.return_shot_type_code,
        "return_shot_type_label": classification.return_shot_type_label,
        "return_depth_code": classification.return_depth_code, "return_depth_label": classification.return_depth_label,
        "actor": classification.return_actor, "terminal_serve_outcome": classification.terminal_serve_outcome,
        "initial_annotation_code": classification.initial_annotation_code,
        "warning_codes": classification.parser_warning_codes, "residual_spans": classification.residual_spans,
    }


def construct_attempts(development: pd.DataFrame, *, extractor: Callable[..., ReturnDirectionClassification] = parse_and_classify_initial_return_direction) -> pd.DataFrame:
    """Construye maximo un intento por numero de saque, solo en desarrollo."""
    cache: dict[tuple[str, int, bool], ReturnDirectionClassification] = {}
    def classify(text: str, number: int, prior: bool) -> ReturnDirectionClassification:
        key = (text, number, prior)
        if key not in cache: cache[key] = extractor(text, number, previous_attempt_was_fault=prior)
        return cache[key]
    rows: list[dict[str, Any]] = []
    for row in development.itertuples(index=False):
        first = None
        if _presence(row.first_serve, "first_serve") == "substantive":
            first = classify(row.first_serve, 1, False)
            rows.append(_attempt_row(row, row.first_serve, 1, False, first))
        if _presence(row.second_serve, "second_serve") == "substantive":
            prior = bool(first is not None and first.terminal_serve_outcome == "service_fault")
            second = classify(row.second_serve, 2, prior)
            rows.append(_attempt_row(row, row.second_serve, 2, prior, second))
    attempts = pd.DataFrame(rows)
    if attempts.empty: raise FeasibilityContractError("No hay intentos sustantivos.")
    if attempts.duplicated(["match_id", "point_number", "serve_number"]).any(): raise FeasibilityContractError("Clave de intento duplicada.")
    if not attempts.server_won_point.eq(~attempts.returner_won_point).all(): raise FeasibilityContractError("Outcomes server/returner no reconcilian.")
    attempts = attempts.sort_values(["date", "match_id", "point_number", "serve_number"], kind="stable").reset_index(drop=True)
    attempts.attrs["cache_entries"] = len(cache)
    return attempts


def _count(frame: pd.DataFrame) -> dict[str, int]:
    return {"attempts": int(len(frame)), "matches": int(frame.match_id.nunique()), "servers": int(frame.server_player.nunique()), "returners": int(frame.returner_player.nunique())}


def build_by_state(attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (number, state, reason), part in attempts.groupby(["serve_number", "classification_state", "reason_code"], sort=False, observed=True):
        rows.append({"serve_number": number, "classification_state": state, "reason_code": reason, **_count(part), "return_opportunities": int(part.return_opportunity_observed.sum()), "return_events": int(part.return_event_observed.sum()), "eligible_directions": int(part.eligible_for_direction_analysis.sum()), "share_of_attempts": len(part) / len(attempts)})
    return pd.DataFrame(rows, columns=BY_STATE_COLUMNS).sort_values(["serve_number", "classification_state", "reason_code"], kind="stable").reset_index(drop=True)


def build_by_direction(attempts: pd.DataFrame) -> pd.DataFrame:
    specs = [("lateral_direction", "lateral_direction_code", "lateral_direction_label"), ("shot_type", "return_shot_type_code", "return_shot_type_label"), ("return_depth", "return_depth_code", "return_depth_label")]
    rows = []
    for dimension, code, label in specs:
        subset = attempts.loc[attempts[code].notna()]
        for (number, value, text), part in subset.groupby(["serve_number", code, label], sort=False, observed=True):
            rows.append({"dimension": dimension, "serve_number": number, "code": value, "label": text, **_count(part), "share_of_dimension": len(part) / len(subset.loc[subset.serve_number.eq(number)])})
    ordering = {"lateral_direction": 0, "shot_type": 1, "return_depth": 2}
    return pd.DataFrame(rows, columns=BY_DIRECTION_COLUMNS).sort_values(["dimension", "serve_number", "code"], key=lambda col: col.map(ordering) if col.name == "dimension" else col, kind="stable").reset_index(drop=True)


def _coverage(part: pd.DataFrame) -> dict[str, Any]:
    attempts = len(part); opportunities = int(part.return_opportunity_observed.sum()); events = int(part.return_event_observed.sum()); eligible = int(part.eligible_for_direction_analysis.sum())
    return {"attempts": attempts, "return_opportunities": opportunities, "return_events": events, "eligible_directions": eligible,
            "opportunity_numerator": opportunities, "opportunity_denominator": attempts, "opportunity_rate": None if not attempts else opportunities / attempts,
            "event_numerator": events, "event_denominator": opportunities, "event_rate": None if not opportunities else events / opportunities,
            "direction_numerator": eligible, "direction_denominator": events, "direction_rate": None if not events else eligible / events,
            "end_to_end_numerator": eligible, "end_to_end_denominator": attempts, "end_to_end_rate": None if not attempts else eligible / attempts}


def build_by_group(attempts: pd.DataFrame) -> pd.DataFrame:
    specs = [("total", []), ("serve_number", ["serve_number"]), ("surface", ["surface"]), ("derived_period", ["derived_period"]), ("validation_fold", ["validation_fold"])]
    rows = []
    for dimension, keys in specs:
        groups = [((), attempts)] if not keys else attempts.groupby(keys, sort=False, observed=True)
        for key, part in groups:
            values = key if isinstance(key, tuple) else (key,)
            attrs = dict(zip(keys, values))
            rows.append({"dimension": dimension, "serve_number": attrs.get("serve_number", 0), "surface": attrs.get("surface", "ALL"), "derived_period": attrs.get("derived_period", "ALL"), "validation_fold": attrs.get("validation_fold", "ALL"), **_coverage(part)})
    order = {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}
    return pd.DataFrame(rows, columns=BY_GROUP_COLUMNS).sort_values(["dimension", "serve_number", "surface", "derived_period", "validation_fold"], key=lambda col: col.map(order) if col.name == "dimension" else col, kind="stable").reset_index(drop=True)


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
    eligible = attempts.loc[attempts.eligible_for_direction_analysis & attempts.lateral_direction_code.isin(("1", "2", "3"))].copy()
    rows = []
    specs = [("total", []), ("serve_number", ["serve_number"]), ("surface", ["surface"]), ("derived_period", ["derived_period"])]
    for stratum, keys in specs:
        groups = [((), eligible)] if not keys else eligible.groupby(keys, sort=False, observed=True)
        for key, part in groups:
            values = key if isinstance(key, tuple) else (key,); attrs = dict(zip(keys, values))
            for (code, label), direction in part.groupby(["lateral_direction_code", "lateral_direction_label"], sort=False, observed=True):
                wins = int(direction.returner_won_point.sum()); lower, upper = wilson(wins, len(direction))
                rows.append({"stratum": stratum, "serve_number": attrs.get("serve_number", 0), "surface": attrs.get("surface", "ALL"), "derived_period": attrs.get("derived_period", "ALL"), "lateral_direction_code": code, "lateral_direction_label": label, **_count(direction), "returner_point_wins": wins, "returner_point_win_rate": wins / len(direction), "wilson_95_lower": lower, "wilson_95_upper": upper, "server_point_wins": int(direction.server_won_point.sum()), "server_point_win_rate": float(direction.server_won_point.mean())})
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS).sort_values(["stratum", "serve_number", "surface", "derived_period", "lateral_direction_code"], kind="stable").reset_index(drop=True)


def _coverage_summary(attempts: pd.DataFrame) -> dict[str, dict[str, Any]]:
    c = _coverage(attempts); state = attempts.classification_state
    unknown = int(state.eq(ReturnDirectionState.UNKNOWN.value).sum())
    ambiguous = int(state.eq(ReturnDirectionState.UNKNOWN_INITIAL.value).sum())
    return {"opportunity_coverage": {"numerator": c["opportunity_numerator"], "denominator": c["opportunity_denominator"], "proportion": c["opportunity_rate"]}, "event_localization_coverage": {"numerator": c["event_numerator"], "denominator": c["event_denominator"], "proportion": c["event_rate"]}, "lateral_direction_coverage": {"numerator": c["direction_numerator"], "denominator": c["direction_denominator"], "proportion": c["direction_rate"]}, "end_to_end_directional_coverage": {"numerator": c["end_to_end_numerator"], "denominator": c["end_to_end_denominator"], "proportion": c["end_to_end_rate"]}, "direction_unknown_summary": {"numerator": unknown, "denominator": c["return_events"], "proportion": None if not c["return_events"] else unknown / c["return_events"]}, "ambiguous_initial_return_share": {"numerator": ambiguous, "denominator": c["attempts"], "proportion": None if not c["attempts"] else ambiguous / c["attempts"]}}


def _summary(counts: Mapping[str, int], attempts: pd.DataFrame, by_state: pd.DataFrame, by_direction: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    state_counts = {state: int((attempts.classification_state == state).sum()) for state in STATE_ORDER}
    reason_counts = {str(key): int(value) for key, value in attempts.reason_code.value_counts(sort=False).sort_index().items()}
    direction_counts = {code: int((attempts.lateral_direction_code == code).sum()) for code in DIRECTION_ORDER}
    coverage = _coverage_summary(attempts)
    all_three = all(direction_counts[code] > 0 for code in ("1", "2", "3"))
    status = "available_descriptive" if all_three and not outcomes.empty else "available_descriptive_limited"
    summary: dict[str, Any] = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION, "analysis_status": status,
        "source": {"path": "data/processed/points_enriched.parquet", "columns_used": SOURCE_COLUMNS, "single_parquet_read_per_execution": True, "published_reports_used_as_analytical_source": False},
        "upstream_contracts": {"chronological_validation": "validated", "extractor": "return_direction_feasibility", "classification_contract_version": "1.0.0"},
        "definition": {"pattern_id": "P04", "name": "documented_lateral_direction_of_initial_return", "lateral_codes": {"1": "right_side_of_right_handed_opponent_or_left_of_left_handed", "2": "centre", "3": "left_side_of_right_handed_opponent_or_right_of_left_handed", "0": "unknown"}, "not_crosscourt_or_parallel": True, "depth_is_separate_dimension": True},
        "unit": {"name": "return_opportunity_per_substantive_serve_attempt", "key": ["match_id", "point_number", "serve_number"], "second_serve_only_when_substantive": True},
        "population": dict(counts), "exclusions": {"non_substantive_first_or_second_serves_do_not_create_attempts": True},
        "test_seal": {"test_status": "sealed", "used_for_method_selection": False, **{field: 0 for field in TEST_ZERO_FIELDS}},
        "classification_contract": {"states": list(STATE_ORDER), "eligible_state": ReturnDirectionState.OBSERVED.value, "unknown_and_censored_excluded_from_outcomes": True, "cache_key": ["sequence_text", "serve_number", "previous_attempt_was_fault"]},
        "state_counts": state_counts, "reason_code_counts": reason_counts, "direction_counts": direction_counts,
        **coverage,
        "censoring_summary": {state: {"numerator": state_counts[state], "denominator": len(attempts), "proportion": state_counts[state] / len(attempts)} for state in (ReturnDirectionState.ACE.value, ReturnDirectionState.UNRETURNED.value, ReturnDirectionState.FAULT.value, ReturnDirectionState.SPECIAL.value)},
        "shot_type_summary": {str(k): int(v) for k, v in attempts.return_shot_type_code.value_counts(dropna=True, sort=False).sort_index().items()},
        "return_depth_summary": {"7": int((attempts.return_depth_code == "7").sum()), "8": int((attempts.return_depth_code == "8").sum()), "9": int((attempts.return_depth_code == "9").sum()), "0": int((attempts.return_depth_code == "0").sum()), "absent": int(attempts.return_depth_code.isna().sum())},
        "outcomes_summary": {"eligible_attempts": int(len(outcomes) and attempts.eligible_for_direction_analysis.sum()), "outcome_rows": int(len(outcomes)), "wilson_z": 1.959963984540054, "observational_only": True},
        "reconciliations": {},
        "methodological_limits": ["Analisis descriptivo y observacional, condicionado a devoluciones observables.", "Ace, saque no devuelto y faults censuran oportunidades; no son categorias de retorno.", "Puede existir confusion por calidad y direccion del saque, servidor, restador, superficie y periodo.", "1/2/3 no significa cruzado, paralelo, izquierda o derecha universal; profundidad es separada.", "No se seleccionan thresholds, perfiles, modelos, scoring ni recomendaciones; test 2024-2026 sellado."],
        "artifact_contracts": {"csv_index": False, "utf8": True, "fixed_order": True, "no_raw_sequences": True}, "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
    }
    summary["reconciliations"] = validate_reconciliations(attempts, by_state, by_direction, by_group, outcomes)
    summary["publication_fingerprint"] = None
    return summary


def validate_reconciliations(attempts: pd.DataFrame, by_state: pd.DataFrame, by_direction: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, bool]:
    total = len(attempts); eligible = attempts.loc[attempts.eligible_for_direction_analysis]
    checks = {
        "attempt_key_unique": not attempts.duplicated(["match_id", "point_number", "serve_number"]).any(),
        "first_second_exhaustive": int((attempts.serve_number == 1).sum() + (attempts.serve_number == 2).sum()) == total,
        "states_exhaustive": int(by_state.attempts.sum()) == total,
        "reasons_exhaustive": int(by_state.groupby("classification_state", observed=True).attempts.sum().sum()) == total,
        "opportunities_reconciled": int(attempts.return_opportunity_observed.sum()) == int(by_state.return_opportunities.sum()),
        "events_reconciled": int(attempts.return_event_observed.sum()) == int(by_state.return_events.sum()),
        "eligible_reconciled": int(len(eligible)) == int(by_state.eligible_directions.sum()),
        "lateral_eligible_exact": int((attempts.lateral_direction_code.isin(("1", "2", "3"))).sum()) == len(eligible),
        "outcomes_only_eligible": outcomes.empty or outcomes.lateral_direction_code.isin(("1", "2", "3")).all(),
        "outcomes_wins_reconcile": outcomes.empty or outcomes.server_point_wins.add(outcomes.returner_point_wins).eq(outcomes.attempts).all(),
        "surface_groups_reconcile": int(by_group.loc[by_group.dimension == "surface", "attempts"].sum()) == total,
        "period_groups_reconcile": int(by_group.loc[by_group.dimension == "derived_period", "attempts"].sum()) == total,
        "fold_groups_reconcile": int(by_group.loc[by_group.dimension == "validation_fold", "attempts"].sum()) == total,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()): raise FeasibilityContractError(f"Reconciliaciones fallidas: {[key for key, value in checks.items() if not value]}")
    return checks


def analyze_points(points: pd.DataFrame, *, expected_population: Mapping[str, int] | None = None, extractor: Callable[..., ReturnDirectionClassification] = parse_and_classify_initial_return_direction) -> DescriptiveResult:
    source = validate_source_points(points, expected_rows=None if expected_population is None else expected_population["source_rows_read"])
    development, counts = split_development_before_parsing(source)
    if expected_population is not None and counts != dict(expected_population): raise FeasibilityContractError(f"Poblacion congelada no reconcilia: {counts}")
    attempts = construct_attempts(development, extractor=extractor)
    by_state, by_direction, by_group, outcomes = build_by_state(attempts), build_by_direction(attempts), build_by_group(attempts), build_outcomes(attempts)
    summary = _summary(counts, attempts, by_state, by_direction, by_group, outcomes)
    result = DescriptiveResult(summary, by_state, by_direction, by_group, outcomes, attempts, "")
    payloads = serialize_artifacts(result)
    fingerprint = _fingerprint(summary, payloads[1:])
    result = DescriptiveResult({**summary, "publication_fingerprint": fingerprint}, by_state, by_direction, by_group, outcomes, attempts, fingerprint)
    validate_result(result)
    return result


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    work = frame.copy(deep=True)
    for column in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[column]): work[column] = work[column].dt.strftime("%Y-%m-%d")
    buffer = io.StringIO(newline="")
    work.to_csv(buffer, index=False, lineterminator="\n", encoding="utf-8", float_format="%.15g", na_rep="")
    return buffer.getvalue().encode("utf-8")


def _fingerprint(summary: Mapping[str, Any], table_payloads: Sequence[bytes]) -> str:
    core = dict(summary); core.pop("publication_fingerprint", None)
    digest = hashlib.sha256(); digest.update(FINGERPRINT_CONTRACT_VERSION.encode("ascii"))
    for payload in table_payloads: digest.update(payload)
    digest.update(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return digest.hexdigest().upper()


def validate_result(result: DescriptiveResult) -> None:
    for frame, columns in ((result.by_state, BY_STATE_COLUMNS), (result.by_direction, BY_DIRECTION_COLUMNS), (result.by_group, BY_GROUP_COLUMNS), (result.outcomes, OUTCOME_COLUMNS)):
        if frame.columns.tolist() != columns or any(column.startswith("Unnamed") for column in frame.columns): raise FeasibilityContractError("Esquema CSV invalido.")
        if not frame.empty and not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(): raise FeasibilityContractError("CSV contiene NaN o infinito.")
    expected_state, expected_direction, expected_group, expected_outcomes = build_by_state(result.attempts), build_by_direction(result.attempts), build_by_group(result.attempts), build_outcomes(result.attempts)
    for actual, expected in ((result.by_state, expected_state), (result.by_direction, expected_direction), (result.by_group, expected_group), (result.outcomes, expected_outcomes)):
        pdt.assert_frame_equal(actual, expected, check_exact=True)
    expected_summary = _summary(result.summary["population"], result.attempts, expected_state, expected_direction, expected_group, expected_outcomes)
    expected_payloads = ( _frame_bytes(expected_state), _frame_bytes(expected_direction), _frame_bytes(expected_group), _frame_bytes(expected_outcomes) )
    expected_fingerprint = _fingerprint(expected_summary, expected_payloads)
    expected_summary["publication_fingerprint"] = expected_fingerprint
    if result.summary != expected_summary or result.publication_fingerprint != expected_fingerprint: raise FeasibilityContractError("Summary o fingerprint no reconcilia.")
    if not all(value is True for value in result.summary["reconciliations"].values()): raise FeasibilityContractError("Reconciliacion publicada falsa.")


def serialize_artifacts(result: DescriptiveResult) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    if result.publication_fingerprint: validate_result(result)
    summary = (json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return summary, _frame_bytes(result.by_state), _frame_bytes(result.by_direction), _frame_bytes(result.by_group), _frame_bytes(result.outcomes)


def _stage(path: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True); raise
    return temporary


def write_artifacts(result: DescriptiveResult, *, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_direction_path: Path = BY_DIRECTION_PATH, by_group_path: Path = BY_GROUP_PATH, outcomes_path: Path = OUTCOMES_PATH) -> None:
    paths = (summary_path, by_state_path, by_direction_path, by_group_path, outcomes_path)
    if len({path.resolve() for path in paths}) != 5: raise FeasibilityContractError("Rutas de artefacto no son distintas.")
    for path in paths: path.parent.mkdir(parents=True, exist_ok=True)
    first = serialize_artifacts(result); second = serialize_artifacts(result)
    if first != second: raise FeasibilityContractError("Serializacion no determinista.")
    originals = {path: path.read_bytes() if path.exists() else None for path in paths}; staged: list[Path] = []; replaced: list[Path] = []
    try:
        for path, payload in zip(paths, first): staged.append(_stage(path, payload))
        for path, temporary in zip(paths, staged): os.replace(temporary, path); replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            if originals[path] is None: path.unlink(missing_ok=True)
            else: os.replace(_stage(path, originals[path]), path)
        raise
    finally:
        for temporary in staged: temporary.unlink(missing_ok=True)
    if tuple(path.read_bytes() for path in paths) != first: raise FeasibilityContractError("Persistencia no coincide con serializacion.")


def _fail(message: str) -> None:
    raise FeasibilityContractError(message)


def _integer(value: object, field: str, *, minimum: int = 0, allowed: set[int] | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _fail(f"{field} debe ser entero real.")
    number = int(value)
    if number < minimum or (allowed is not None and number not in allowed): _fail(f"{field} fuera de dominio: {value!r}.")
    return number


def _code(value: object, field: str) -> str:
    if isinstance(value, str):
        if not value or value != value.strip(): _fail(f"{field} debe ser codigo no vacio.")
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _fail(f"{field} debe ser codigo textual o entero sin coercion.")
    return str(int(value))


def _rate(value: object, numerator: int, denominator: int, field: str) -> None:
    if denominator == 0:
        if value not in (None, "") and not pd.isna(value): _fail(f"{field} debe ser nulo con denominador cero.")
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        _fail(f"{field} debe ser numerico finito.")
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 1 or not math.isclose(float(value), numerator / denominator, rel_tol=0, abs_tol=RATE_TOLERANCE):
        _fail(f"{field} no reconcilia con su numerador y denominador.")


def _canonical(frame: pd.DataFrame, keys: Sequence[str], name: str, *, rank: Mapping[str, int] | None = None) -> None:
    positions = list(range(len(frame)))
    def key(position: int) -> tuple[Any, ...]:
        values: list[Any] = []
        for column in keys:
            value = frame.iloc[position][column]
            values.append(rank.get(value, 99) if rank is not None and column == "dimension" else str(value))
        return tuple(values)
    if positions != sorted(positions, key=key): _fail(f"Orden canonico invalido en {name}.")


def _read_csv_payload(payload: bytes, columns: Sequence[str], name: str) -> pd.DataFrame:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FeasibilityContractError(f"{name} no es UTF-8.") from error
    if not text.endswith("\n") or "\r" in text or "\x00" in text: _fail(f"Formato de bytes invalido en {name}.")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != list(columns) or any(len(row) != len(columns) for row in rows[1:]): _fail(f"Schema o filas CSV invalidas en {name}.")
    if any(cell.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"} for row in rows[1:] for cell in row): _fail(f"NaN o infinito literal en {name}.")
    return pd.read_csv(io.BytesIO(payload), keep_default_na=False)


def _validate_state_table(frame: pd.DataFrame, summary: Mapping[str, Any]) -> None:
    if frame.duplicated(["serve_number", "classification_state", "reason_code"]).any(): _fail("Clave duplicada en by_state.")
    _canonical(frame, ["serve_number", "classification_state", "reason_code"], "by_state")
    total = _integer(summary["end_to_end_directional_coverage"]["denominator"], "total_attempts", minimum=1)
    aggregate_states: dict[str, int] = {state: 0 for state in STATE_ORDER}; aggregate_reasons: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        serve = _integer(row.serve_number, "by_state.serve_number", allowed={1, 2})
        state, reason = row.classification_state, row.reason_code
        if state not in STATE_REASONS or reason not in STATE_REASONS[state]: _fail("Estado o reason_code no permitido en by_state.")
        attempts = _integer(row.attempts, "by_state.attempts", minimum=1)
        for field in ("matches", "servers", "returners"):
            value = _integer(getattr(row, field), f"by_state.{field}", minimum=1)
            if value > attempts: _fail(f"by_state.{field} supera intentos.")
        observed = state == ReturnDirectionState.OBSERVED.value
        localized = state in {ReturnDirectionState.OBSERVED.value, ReturnDirectionState.UNKNOWN.value}
        expected = (attempts if localized else 0, attempts if localized else 0, attempts if observed else 0)
        actual = tuple(_integer(getattr(row, field), f"by_state.{field}") for field in ("return_opportunities", "return_events", "eligible_directions"))
        if actual != expected: _fail("Flags agregados incompatibles con estado P04.")
        _rate(row.share_of_attempts, attempts, total, "by_state.share_of_attempts")
        aggregate_states[state] = aggregate_states.get(state, 0) + attempts
        aggregate_reasons[reason] = aggregate_reasons.get(reason, 0) + attempts
        if serve not in (1, 2): _fail("serve_number invalido.")
    if sum(aggregate_states.values()) != total or aggregate_states != summary["state_counts"] or aggregate_reasons != summary["reason_code_counts"]: _fail("by_state no reconcilia con summary.")


def _validate_direction_table(frame: pd.DataFrame, summary: Mapping[str, Any], by_group: pd.DataFrame) -> None:
    if frame.duplicated(["dimension", "serve_number", "code"]).any(): _fail("Clave duplicada en by_direction.")
    _canonical(frame, ["dimension", "serve_number", "code"], "by_direction", rank={"lateral_direction": 0, "shot_type": 1, "return_depth": 2})
    dimensions = {"lateral_direction", "shot_type", "return_depth"}
    if set(frame.dimension) != dimensions: _fail("Dimensiones de by_direction incompletas o adicionales.")
    expected_labels = {"lateral_direction": LATERAL_LABELS, "shot_type": SHOT_LABELS, "return_depth": DEPTH_LABELS}
    totals: dict[str, int] = {key: 0 for key in dimensions}
    lateral: dict[str, int] = {code: 0 for code in DIRECTION_ORDER}
    depths: dict[str, int] = {code: 0 for code in ("7", "8", "9", "0")}
    shots: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        dimension = row.dimension; code = _code(row.code, "by_direction.code")
        _integer(row.serve_number, "by_direction.serve_number", allowed={1, 2})
        if code not in expected_labels[dimension] or row.label != expected_labels[dimension][code]: _fail("Codigo o etiqueta documental invalida en by_direction.")
        attempts = _integer(row.attempts, "by_direction.attempts", minimum=1)
        for field in ("matches", "servers", "returners"):
            value = _integer(getattr(row, field), f"by_direction.{field}", minimum=1)
            if value > attempts: _fail(f"by_direction.{field} supera intentos.")
        totals[dimension] += attempts
        if dimension == "lateral_direction": lateral[code] = lateral.get(code, 0) + attempts
        elif dimension == "return_depth": depths[code] = depths.get(code, 0) + attempts
        else: shots[code] = shots.get(code, 0) + attempts
    for (dimension, serve), part in frame.groupby(["dimension", "serve_number"], sort=False, observed=True):
        if not math.isclose(float(part.share_of_dimension.sum()), 1.0, abs_tol=RATE_TOLERANCE): _fail("share_of_dimension no suma uno.")
    total_row = by_group.loc[by_group.dimension.eq("total")].iloc[0]
    if lateral != {str(key): int(value) for key, value in summary["direction_counts"].items()}: _fail("Direcciones laterales no reconcilian con summary.")
    if sum(lateral[code] for code in ("1", "2", "3")) != _integer(total_row.eligible_directions, "eligible_directions"): _fail("Direcciones observadas no reconcilian.")
    if totals["shot_type"] != _integer(total_row.return_events, "return_events"): _fail("Shot types no reconcilian con eventos localizados.")
    expected_depths = {code: int(summary["return_depth_summary"][code]) for code in depths}
    if depths != expected_depths or int(summary["return_depth_summary"]["absent"]) != _integer(total_row.attempts, "attempts") - totals["return_depth"]: _fail("Profundidad no reconcilia con summary.")
    if shots != {str(key): int(value) for key, value in summary["shot_type_summary"].items()}: _fail("Shot types no reconcilian con summary.")


def _validate_group_table(frame: pd.DataFrame, summary: Mapping[str, Any], *, frozen: bool) -> None:
    keys = ["dimension", "serve_number", "surface", "derived_period", "validation_fold"]
    if frame.duplicated(keys).any(): _fail("Clave duplicada en by_group.")
    ranks = {"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4}
    _canonical(frame, keys, "by_group", rank=ranks)
    if set(frame.dimension) != set(ranks): _fail("Dimensiones de by_group incompletas o adicionales.")
    total_rows = frame.loc[frame.dimension.eq("total")]
    if len(total_rows) != 1: _fail("by_group requiere exactamente un total.")
    total = total_rows.iloc[0]
    expected_fields = {"total": (0, "ALL", "ALL", "ALL"), "serve_number": None, "surface": None, "derived_period": None, "validation_fold": None}
    if (total.serve_number, total.surface, total.derived_period, total.validation_fold) != expected_fields["total"]: _fail("Fila total invalida.")
    count_columns = ("attempts", "return_opportunities", "return_events", "eligible_directions")
    for row in frame.itertuples(index=False):
        dimension = row.dimension
        _integer(row.serve_number, "by_group.serve_number", allowed={0, 1, 2})
        if dimension != "serve_number" and row.serve_number != 0: _fail("serve_number no aplicable debe ser cero.")
        if dimension == "serve_number" and (row.surface, row.derived_period, row.validation_fold) != ("ALL", "ALL", "ALL"): _fail("Fila serve_number invalida.")
        if dimension == "surface" and (row.surface not in ALLOWED_SURFACES or row.derived_period != "ALL" or row.validation_fold != "ALL"): _fail("Fila surface invalida.")
        if dimension == "derived_period" and (row.surface != "ALL" or row.derived_period not in PERIOD_ORDER or row.validation_fold != "ALL"): _fail("Fila derived_period invalida.")
        if dimension == "validation_fold" and (row.surface != "ALL" or row.derived_period != "ALL" or row.validation_fold not in FOLD_ORDER): _fail("Fila validation_fold invalida.")
        values = {field: _integer(getattr(row, field), f"by_group.{field}") for field in count_columns}
        if not values["eligible_directions"] <= values["return_events"] <= values["return_opportunities"] <= values["attempts"]: _fail("Cobertura by_group fuera de orden.")
        if values["return_events"] != values["return_opportunities"]: _fail("Eventos y oportunidades P04 deben coincidir.")
        for prefix, numerator, denominator, rate in (("opportunity", values["return_opportunities"], values["attempts"], row.opportunity_rate), ("event", values["return_events"], values["return_opportunities"], row.event_rate), ("direction", values["eligible_directions"], values["return_events"], row.direction_rate), ("end_to_end", values["eligible_directions"], values["attempts"], row.end_to_end_rate)):
            if _integer(getattr(row, f"{prefix}_numerator"), f"by_group.{prefix}_numerator") != numerator or _integer(getattr(row, f"{prefix}_denominator"), f"by_group.{prefix}_denominator") != denominator: _fail("Numerador o denominador by_group invalido.")
            _rate(rate, numerator, denominator, f"by_group.{prefix}_rate")
    total_counts = {field: _integer(total[field], f"total.{field}") for field in count_columns}
    for dimension in ("serve_number", "surface", "derived_period", "validation_fold"):
        part = frame.loc[frame.dimension.eq(dimension)]
        if part.empty or any(int(part[field].sum()) != total_counts[field] for field in count_columns): _fail(f"{dimension} no reconcilia con total.")
    if frozen and (set(frame.loc[frame.dimension.eq("serve_number"), "serve_number"]) != {1, 2} or set(frame.loc[frame.dimension.eq("surface"), "surface"]) != set(ALLOWED_SURFACES) or set(frame.loc[frame.dimension.eq("derived_period"), "derived_period"]) != set(PERIOD_ORDER) or set(frame.loc[frame.dimension.eq("validation_fold"), "validation_fold"]) != set(FOLD_ORDER)):
        _fail("Cobertura congelada incompleta.")
    summary_total = summary["end_to_end_directional_coverage"]
    if int(summary_total["numerator"]) != total_counts["eligible_directions"] or int(summary_total["denominator"]) != total_counts["attempts"]: _fail("Cobertura total no reconcilia con by_group.")
    _rate(summary_total["proportion"], total_counts["eligible_directions"], total_counts["attempts"], "summary.end_to_end_directional_coverage")
    for name, numerator, denominator in (("opportunity_coverage", total_counts["return_opportunities"], total_counts["attempts"]), ("event_localization_coverage", total_counts["return_events"], total_counts["return_opportunities"]), ("lateral_direction_coverage", total_counts["eligible_directions"], total_counts["return_events"])):
        item = summary[name]
        if int(item["numerator"]) != numerator or int(item["denominator"]) != denominator: _fail(f"{name} no reconcilia.")
        _rate(item["proportion"], numerator, denominator, f"summary.{name}")


def _validate_outcomes_table(frame: pd.DataFrame, by_direction: pd.DataFrame, summary: Mapping[str, Any], *, frozen: bool) -> None:
    keys = ["stratum", "serve_number", "surface", "derived_period", "lateral_direction_code"]
    if frame.duplicated(keys).any(): _fail("Clave duplicada en outcomes.")
    _canonical(frame, keys, "outcomes")
    expected_labels = LATERAL_LABELS
    totals: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        code = _code(row.lateral_direction_code, "outcomes.lateral_direction_code")
        if code not in {"1", "2", "3"} or row.lateral_direction_label != expected_labels[code]: _fail("Outcome para direccion unknown o censurada.")
        serve = _integer(row.serve_number, "outcomes.serve_number", allowed={0, 1, 2})
        if row.stratum == "total" and (serve, row.surface, row.derived_period) != (0, "ALL", "ALL"): _fail("Fila outcome total invalida.")
        if row.stratum == "serve_number" and (serve not in {1, 2} or row.surface != "ALL" or row.derived_period != "ALL"): _fail("Fila outcome serve invalida.")
        if row.stratum == "surface" and (serve != 0 or row.surface not in ALLOWED_SURFACES or row.derived_period != "ALL"): _fail("Fila outcome surface invalida.")
        if row.stratum == "derived_period" and (serve != 0 or row.surface != "ALL" or row.derived_period not in PERIOD_ORDER): _fail("Fila outcome periodo invalida.")
        if row.stratum not in {"total", "serve_number", "surface", "derived_period"}: _fail("Stratum outcome invalido.")
        attempts = _integer(row.attempts, "outcomes.attempts", minimum=1)
        wins = _integer(row.returner_point_wins, "outcomes.returner_point_wins")
        server_wins = _integer(row.server_point_wins, "outcomes.server_point_wins")
        if wins + server_wins != attempts: _fail("Outcomes complementarios no reconcilian.")
        for field in ("matches", "returners", "servers"):
            value = _integer(getattr(row, field), f"outcomes.{field}", minimum=1)
            if value > attempts: _fail(f"outcomes.{field} supera intentos.")
        _rate(row.returner_point_win_rate, wins, attempts, "outcomes.returner_point_win_rate")
        _rate(row.server_point_win_rate, server_wins, attempts, "outcomes.server_point_win_rate")
        lower, upper = wilson(wins, attempts)
        if not math.isclose(float(row.wilson_95_lower), lower, rel_tol=0, abs_tol=RATE_TOLERANCE) or not math.isclose(float(row.wilson_95_upper), upper, rel_tol=0, abs_tol=RATE_TOLERANCE): _fail("Wilson no reconcilia con outcomes.")
        if not 0 <= float(row.wilson_95_lower) <= float(row.returner_point_win_rate) <= float(row.wilson_95_upper) <= 1: _fail("Wilson fuera de [0,1].")
        if row.stratum == "total": totals[code] = attempts
    observed = {code: int(value) for code, value in summary["direction_counts"].items() if code in {"1", "2", "3"}}
    if totals != observed: _fail("Outcomes totales no reconcilian con direcciones observadas.")
    for stratum in ("serve_number", "surface", "derived_period"):
        part = frame.loc[frame.stratum.eq(stratum)]
        if set(part.lateral_direction_code.astype(str)) != set(observed) or {str(code): int(value) for code, value in part.groupby("lateral_direction_code", observed=True).attempts.sum().items()} != observed: _fail(f"Outcomes {stratum} no reconcilian con total.")
    if int(summary["outcomes_summary"]["eligible_attempts"]) != sum(totals.values()) or int(summary["outcomes_summary"]["outcome_rows"]) != len(frame) or not math.isclose(float(summary["outcomes_summary"]["wilson_z"]), 1.959963984540054, abs_tol=RATE_TOLERANCE): _fail("Summary de outcomes invalido.")
    if frozen and len(frame) != 27: _fail("Numero congelado de filas outcomes invalido.")


def _validate_persisted_semantics(summary: Mapping[str, Any], by_state: pd.DataFrame, by_direction: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame, *, expected_population: Mapping[str, int] | None) -> None:
    if summary.get("analysis_name") != ANALYSIS_NAME or summary.get("analysis_version") != ANALYSIS_VERSION: _fail("Identidad de analisis invalida.")
    if expected_population is not None and summary.get("population") != dict(expected_population): _fail("Poblacion publicada no coincide con el contrato.")
    if not isinstance(summary.get("population"), Mapping): _fail("Poblacion publicada invalida.")
    total_attempts = _integer(summary["end_to_end_directional_coverage"]["denominator"], "summary.total_attempts", minimum=1)
    if _integer(summary["population"]["development_point_rows"], "population.development_point_rows", minimum=1) > total_attempts: _fail("Puntos de desarrollo superan intentos.")
    seal = summary.get("test_seal")
    if not isinstance(seal, Mapping) or seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or set(seal) != {"test_status", "used_for_method_selection", *TEST_ZERO_FIELDS}: _fail("Contrato de sellado test invalido.")
    if any(_integer(seal[field], f"test_seal.{field}") != 0 for field in TEST_ZERO_FIELDS): _fail("El test sellado contiene uso publicado.")
    if summary.get("definition", {}).get("lateral_codes") != {"1": LATERAL_LABELS["1"], "2": LATERAL_LABELS["2"], "3": LATERAL_LABELS["3"], "0": LATERAL_LABELS["0"]} or summary.get("definition", {}).get("depth_is_separate_dimension") is not True or summary.get("definition", {}).get("not_crosscourt_or_parallel") is not True: _fail("Definicion P04 invalida.")
    if summary.get("unit", {}).get("key") != ["match_id", "point_number", "serve_number"] or summary.get("classification_contract", {}).get("states") != list(STATE_ORDER) or summary.get("classification_contract", {}).get("unknown_and_censored_excluded_from_outcomes") is not True: _fail("Unidad o estados P04 invalidos.")
    frozen = expected_population is not None and dict(expected_population) == EXPECTED
    _validate_group_table(by_group, summary, frozen=frozen)
    _validate_state_table(by_state, summary)
    _validate_direction_table(by_direction, summary, by_group)
    _validate_outcomes_table(outcomes, by_direction, summary, frozen=frozen)
    directions = summary["direction_counts"]
    all_three = all(int(directions.get(code, -1)) > 0 for code in ("1", "2", "3"))
    expected_status = "available_descriptive" if all_three and not outcomes.empty else "available_descriptive_limited"
    if summary.get("analysis_status") != expected_status: _fail("analysis_status no corresponde a los artefactos publicados.")
    if summary.get("outcomes_summary", {}).get("observational_only") is not True: _fail("Outcomes deben permanecer observacionales.")
    expected_reconciliations = {"attempt_key_unique", "first_second_exhaustive", "states_exhaustive", "reasons_exhaustive", "opportunities_reconciled", "events_reconciled", "eligible_reconciled", "lateral_eligible_exact", "outcomes_only_eligible", "outcomes_wins_reconcile", "surface_groups_reconcile", "period_groups_reconcile", "fold_groups_reconcile"}
    if set(summary.get("reconciliations", {})) != expected_reconciliations or not all(value is True for value in summary["reconciliations"].values()): _fail("Reconciliaciones publicadas invalidas.")


def verify_persisted_artifacts(*, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_direction_path: Path = BY_DIRECTION_PATH, by_group_path: Path = BY_GROUP_PATH, outcomes_path: Path = OUTCOMES_PATH, expected_population: Mapping[str, int] | None = EXPECTED) -> None:
    paths = (summary_path, by_state_path, by_direction_path, by_group_path, outcomes_path)
    payloads = tuple(path.read_bytes() for path in paths)
    try:
        summary = json.loads(payloads[0].decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FeasibilityContractError("Summary JSON invalido o no finito.") from error
    if (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") != payloads[0]: _fail("Summary JSON no tiene serializacion canonica.")
    tables = tuple(_read_csv_payload(payload, columns, name) for payload, columns, name in zip(payloads[1:], (BY_STATE_COLUMNS, BY_DIRECTION_COLUMNS, BY_GROUP_COLUMNS, OUTCOME_COLUMNS), ("by_state", "by_direction", "by_group", "outcomes")))
    if _fingerprint(summary, payloads[1:]) != summary.get("publication_fingerprint"): _fail("Fingerprint persistido invalido.")
    _validate_persisted_semantics(summary, *tables, expected_population=expected_population)


def refresh_persisted_summary_test_seal(*, summary_path: Path = SUMMARY_PATH, by_state_path: Path = BY_STATE_PATH, by_direction_path: Path = BY_DIRECTION_PATH, by_group_path: Path = BY_GROUP_PATH, outcomes_path: Path = OUTCOMES_PATH, expected_population: Mapping[str, int] | None = EXPECTED) -> None:
    """Actualiza solo el contrato de sellado desde bytes P04 ya persistidos.

    No reconstruye intentos ni abre el Parquet. Esta migracion puntual no forma
    parte del CLI y conserva intactos los cuatro CSV publicados.
    """
    paths = (summary_path, by_state_path, by_direction_path, by_group_path, outcomes_path)
    original = summary_path.read_bytes()
    payloads = tuple(path.read_bytes() for path in paths)
    summary = json.loads(original.decode("utf-8"))
    seal = summary.get("test_seal")
    legacy = {"test_status", "used_for_method_selection", "test_target_rows_parsed", "test_attempts_constructed", "test_returns_classified", "test_rows_evaluated", "test_matches_evaluated", "test_evaluation_runs"}
    if not isinstance(seal, dict) or not set(seal).issubset({"test_status", "used_for_method_selection", *TEST_ZERO_FIELDS}) or not set(seal).issuperset(legacy): _fail("No es seguro migrar un contrato de sellado desconocido.")
    if seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or any(seal.get(field, 0) != 0 for field in TEST_ZERO_FIELDS): _fail("No es seguro migrar un test no sellado.")
    summary["test_seal"] = {"test_status": "sealed", "used_for_method_selection": False, **{field: 0 for field in TEST_ZERO_FIELDS}}
    summary["publication_fingerprint"] = _fingerprint(summary, payloads[1:])
    refreshed = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = _stage(summary_path, refreshed)
    try:
        os.replace(temporary, summary_path)
        verify_persisted_artifacts(summary_path=summary_path, by_state_path=by_state_path, by_direction_path=by_direction_path, by_group_path=by_group_path, outcomes_path=outcomes_path, expected_population=expected_population)
    except Exception:
        if summary_path.read_bytes() != original: os.replace(_stage(summary_path, original), summary_path)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    if tuple(path.read_bytes() for path in paths[1:]) != payloads[1:]: _fail("La migracion de summary modifico un CSV.")


def read_source_points(path: Path = POINTS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path, columns=SOURCE_COLUMNS)


def run_real_analysis() -> DescriptiveResult:
    validate_chronological_contract()
    for path in (SUMMARY_PATH, BY_STATE_PATH, BY_DIRECTION_PATH, BY_GROUP_PATH, OUTCOMES_PATH):
        if path.exists(): raise FeasibilityContractError("Ya existe un artefacto P04; no se sobrescribe sin auditoria.")
    points = read_source_points()  # unica lectura fisica autorizada
    return analyze_points(points, expected_population=EXPECTED)


def main() -> None:
    result = run_real_analysis()
    write_artifacts(result)
    verify_persisted_artifacts()
    print(f"P04 {result.summary['analysis_status']} | intentos={len(result.attempts)} | direcciones={result.summary['direction_counts']}")


if __name__ == "__main__":
    main()
