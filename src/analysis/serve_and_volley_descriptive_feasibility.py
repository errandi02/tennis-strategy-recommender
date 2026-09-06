"""Estudio descriptivo sellado de P03: intencion anotada de saque y volea.

P03 solo observa el marcador literal ``+`` inmediato al prefijo de servicio.
No infiere una volea, un actor, causalidad ni una recomendacion. La fuente es
unicamente ``points_enriched.parquet`` y la frontera temporal se aplica antes
de invocar el extractor P03.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.first_serve_direction_feasibility import (
    EXPECTED_CHRONOLOGY_BINARY_SHA256_LF,
    EXPECTED_CHRONOLOGY_CANONICAL_SHA256,
    EXPECTED_CHRONOLOGY_COMMIT,
    EXPECTED_CHRONOLOGY_PUBLICATION_FINGERPRINT,
    validate_chronological_contract,
)
from src.analysis.second_serve_direction_analysis import derive_period
from src.analysis.serve_and_volley_feasibility import (
    AnalysisState,
    ReasonCode,
    ServeAndVolleyClassification,
    parse_and_classify_serve_and_volley_attempt,
    validate_serve_and_volley_classification,
)


ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "serve_and_volley_descriptive_feasibility_summary.json"
BY_STATE_PATH = TABLES_DIR / "serve_and_volley_descriptive_feasibility_by_state.csv"
BY_GROUP_PATH = TABLES_DIR / "serve_and_volley_descriptive_feasibility_by_group.csv"
OUTCOMES_PATH = TABLES_DIR / "serve_and_volley_descriptive_feasibility_outcomes.csv"

ANALYSIS_NAME = "serve_and_volley_descriptive_feasibility"
ANALYSIS_VERSION = "1.0.0"
PATTERN_ID = "P03"
FINGERPRINT_CONTRACT_VERSION = "1"
CUTOFF = pd.Timestamp("2023-12-31")
EXPECTED_SOURCE_ROWS = 1_280_408
EXPECTED_SOURCE_MATCHES = 7_524
EXPECTED_SOURCE_PLAYERS = 1_002
EXPECTED_DEVELOPMENT_ROWS = 1_035_760
EXPECTED_DEVELOPMENT_MATCHES = 5_993
EXPECTED_DEVELOPMENT_SERVERS = 870
EXPECTED_TEST_MATCHES = 1_531
EXPECTED_PUBLISHED_ARTIFACT_SHA256 = {
    "summary": "D2D5965FC5C38CC5B6341D3ADC09A4D7B938ED558FD54747BCE83A312C2917AD",
    "by_state": "616073DB6C3B574B12C515E16EE85C095E685DAAD4A10FE8F1312E249496E3CA",
    "by_group": "56FD4885DDCFAB063F8B7079CC5582AE5A39C65F582B4DA4780B220ACC652751",
    "outcomes": "6CC7BE864E4D4055CC2755A17C1906C46D309D3DACE8783EE02E9713931C67BC",
}
ALLOWED_SURFACES = ("Hard", "Clay", "Grass")
DIRECTION_ORDER = ("wide", "body", "T", "unknown")
STATE_ORDER = tuple(state.value for state in AnalysisState)
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
FOLD_ORDER = ("pre_validation", "validation_2020", "validation_2021", "validation_2022", "validation_2023")

SOURCE_COLUMNS = [
    "match_id", "point_number", "date", "surface", "server", "point_winner",
    "player_1", "player_2", "first_serve", "second_serve",
]
BY_STATE_COLUMNS = [
    "analysis_state", "reason_code", "attempts", "matches", "servers",
    "explicit_tagged_attempts", "eligible_attempts", "unknown_attempts",
    "censored_attempts", "not_explicitly_tagged_attempts", "share_of_attempts",
    "coverage_denominator", "coverage_rate",
]
BY_GROUP_COLUMNS = [
    "dimension", "serve_number", "direction", "surface", "derived_period",
    "validation_fold", "analysis_state", "attempts", "matches", "servers",
    "explicit_tagged_attempts", "eligible_attempts", "unknown_attempts",
    "censored_attempts", "not_explicitly_tagged_attempts", "share_of_attempts",
    "coverage_denominator", "coverage_rate",
]
OUTCOME_COLUMNS = [
    "analysis_state", "attempts", "matches", "servers", "server_point_wins",
    "server_point_win_rate", "wilson_95_lower", "wilson_95_upper",
    "difference_vs_not_explicitly_tagged",
]
TEST_ZERO_FIELDS = (
    "test_target_rows_parsed", "test_attempts_constructed", "test_rows_evaluated",
    "test_matches_evaluated", "test_evaluation_runs",
)
METHODOLOGICAL_LIMITS = [
    "La etiqueta + describe intencion anotada, no realizacion fisica de saque y volea.",
    "El parser actual deja rallies no etiquetados como residuo; no se fabrican negativos desde unknown.",
    "Los outcomes de faults no se usan como efectividad del intento fallado.",
    "Analisis descriptivo observacional: sin thresholds, scoring, modelos, recomendaciones ni causalidad.",
    "El test posterior a 2023 permanece sellado y no se parsea ni evalua.",
]


class FeasibilityContractError(ValueError):
    """Incumplimiento de fuente, sellado, resultado o artefacto P03."""


@dataclass(frozen=True)
class DescriptiveResult:
    summary: dict[str, Any]
    by_state: pd.DataFrame
    by_group: pd.DataFrame
    outcomes: pd.DataFrame
    development: pd.DataFrame
    attempts: pd.DataFrame
    publication_fingerprint: str


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value not in (1, 2):
        raise FeasibilityContractError(f"Dominio invalido en {field}: {value!r}")
    return int(value)


def _exact_text(series: pd.Series, field: str) -> None:
    invalid = series.map(lambda value: not isinstance(value, str) or value == "" or value != value.strip())
    if invalid.any():
        raise FeasibilityContractError(f"{field} debe ser texto no vacio, sin espacios externos.")


def _strict_dates(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise FeasibilityContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif series.map(lambda value: isinstance(value, str)).all():
        if not series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise FeasibilityContractError("date debe usar exclusivamente YYYY-MM-DD.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif series.map(lambda value: isinstance(value, (pd.Timestamp, datetime, date)) and not isinstance(value, bool)).all():
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        raise FeasibilityContractError("date contiene tipos mixtos o ambiguos.")
    if parsed.isna().any() or getattr(parsed.dt, "tz", None) is not None:
        raise FeasibilityContractError("date contiene fechas invalidas o con zona horaria.")
    return parsed.dt.normalize()


def _presence(value: object, field: str) -> str:
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return "null"
    if not isinstance(value, str):
        raise FeasibilityContractError(f"{field} debe ser cadena o nulo.")
    if value == "":
        return "empty"
    if value.isspace():
        return "whitespace_only"
    return "substantive"


def _validation_fold(dates: pd.Series) -> pd.Series:
    years = dates.dt.year
    values = np.where(years.le(2019), "pre_validation", "validation_" + years.astype(str))
    result = pd.Series(values, index=dates.index, dtype="object")
    if not result.isin(FOLD_ORDER).all():
        raise FeasibilityContractError("Fold derivado fuera del protocolo 2020-2023.")
    return result


def validate_source_points(points: pd.DataFrame, *, expected_source_rows: int | None = None) -> pd.DataFrame:
    """Valida la fuente completa, sin invocar parser ni construir intentos."""
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing:
        raise FeasibilityContractError(f"Faltan columnas fuente: {missing}")
    if expected_source_rows is not None and len(points) != expected_source_rows:
        raise FeasibilityContractError(f"source_rows_read inesperado: {len(points)}")
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"):
        _exact_text(work[field], field)
    if work[["match_id", "point_number"]].isna().any().any() or work.duplicated(["match_id", "point_number"]).any():
        raise FeasibilityContractError("La clave (match_id, point_number) debe ser unica y no nula.")
    work["date"] = _strict_dates(work["date"])
    if work.date.gt(pd.Timestamp("2026-05-21")).any():
        raise FeasibilityContractError("Existen fechas posteriores al contrato cronologico.")
    invalid_surface = ~work.surface.isin(ALLOWED_SURFACES)
    if invalid_surface.any():
        raise FeasibilityContractError(f"Superficies inesperadas: {sorted(work.loc[invalid_surface, 'surface'].unique())}")
    if work.player_1.eq(work.player_2).any():
        raise FeasibilityContractError("player_1 y player_2 deben ser distintos.")
    for field in ("server", "point_winner"):
        work[field] = work[field].map(lambda value: _strict_index(value, field))
    metadata = ["date", "surface", "player_1", "player_2"]
    if work.groupby("match_id", sort=False)[metadata].nunique(dropna=False).gt(1).any().any():
        raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    return work.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)


def split_development_before_parsing(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica el sellado a las filas, antes de cualquier construccion o parser."""
    development = source.loc[source.date.le(CUTOFF)].copy()
    test = source.loc[source.date.gt(CUTOFF)]
    if development.empty or test.empty:
        raise FeasibilityContractError("La frontera temporal no particiona la fuente.")
    if development.date.gt(CUTOFF).any():
        raise FeasibilityContractError("El desarrollo contiene filas selladas de test.")
    counts = {
        "source_rows_read": int(len(source)),
        "source_matches": int(source.match_id.nunique()),
        "source_players": int(len(set(source.player_1) | set(source.player_2))),
        "development_target_rows": int(len(development)),
        "development_target_matches": int(development.match_id.nunique()),
        "development_servers": int(pd.Series(np.where(development.server.eq(1), development.player_1, development.player_2)).nunique()),
        "excluded_test_target_matches": int(test.match_id.nunique()),
    }
    return development.reset_index(drop=True), counts


def _classified_row(
    row: pd.Series,
    sequence_text: str,
    serve_number: int,
    prior_fault: bool,
    classifier: Callable[[str, int], ServeAndVolleyClassification],
) -> dict[str, Any]:
    classification = classifier(sequence_text, serve_number, prior_fault)
    validate_serve_and_volley_classification(classification)
    server_player = row.player_1 if row.server == 1 else row.player_2
    return {
        "match_id": row.match_id,
        "point_number": row.point_number,
        "serve_number": serve_number,
        "date": row.date,
        "surface": row.surface,
        "derived_period": derive_period(int(row.date.year)),
        "validation_fold": _validation_fold(pd.Series([row.date])).iloc[0],
        "server_player": server_player,
        "server_won_point": bool(row.point_winner == row.server),
        "direction": classification.service_direction_label or "unknown",
        "explicit_intent_tagged": classification.explicit_intent_tagged,
        "analysis_state": classification.analysis_state.value,
        "reason_code": classification.reason_code.value,
        "eligible_for_outcome_comparison": classification.eligible_for_outcome_comparison,
        "terminal_serve_outcome": classification.terminal_serve_outcome,
        "has_warnings": bool(classification.parser_warning_codes),
        "warning_codes": classification.parser_warning_codes,
        "has_residual": classification.has_residual,
        "marker_start": classification.marker_start,
        "marker_end": classification.marker_end,
        "sequence_length": classification.sequence_length,
    }


def construct_attempts(
    development: pd.DataFrame,
    *,
    extractor: Callable[..., ServeAndVolleyClassification] = parse_and_classify_serve_and_volley_attempt,
) -> pd.DataFrame:
    """Construye como maximo dos intentos por punto sin inventar segundos saques."""
    cache: dict[tuple[str, int, bool], ServeAndVolleyClassification] = {}

    def classify(sequence_text: str, serve_number: int, prior_fault: bool) -> ServeAndVolleyClassification:
        key = (sequence_text, serve_number, prior_fault)
        if key not in cache:
            cache[key] = extractor(sequence_text, serve_number, previous_attempt_was_fault=prior_fault)
        return cache[key]

    rows: list[dict[str, Any]] = []
    for row in development.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        first_presence = _presence(row.first_serve, "first_serve")
        first: ServeAndVolleyClassification | None = None
        if first_presence == "substantive":
            first = classify(row.first_serve, 1, False)
            rows.append(_classified_row(row_series, row.first_serve, 1, False, classify))
        second_presence = _presence(row.second_serve, "second_serve")
        if second_presence == "substantive":
            prior_fault = bool(first is not None and first.terminal_serve_outcome == "service_fault")
            rows.append(_classified_row(row_series, row.second_serve, 2, prior_fault, classify))
    attempts = pd.DataFrame(rows)
    required = {"match_id", "point_number", "serve_number"}
    if attempts.empty:
        raise FeasibilityContractError("No se construyeron intentos sustantivos.")
    if attempts.duplicated(["match_id", "point_number", "serve_number"]).any() or not required.issubset(attempts.columns):
        raise FeasibilityContractError("La clave de intento debe ser unica.")
    attempts = attempts.sort_values(["date", "match_id", "point_number", "serve_number"], kind="stable").reset_index(drop=True)
    attempts.attrs["cache_entries"] = len(cache)
    return attempts


def _stats(part: pd.DataFrame, denominator: int) -> dict[str, Any]:
    attempts = int(len(part))
    state = part.analysis_state if attempts else pd.Series(dtype="object")
    observable = int(state.isin((AnalysisState.POSITIVE_TAGGED.value, AnalysisState.NOT_EXPLICITLY_TAGGED.value)).sum())
    return {
        "attempts": attempts,
        "matches": int(part.match_id.nunique()),
        "servers": int(part.server_player.nunique()),
        "explicit_tagged_attempts": int(part.explicit_intent_tagged.sum()),
        "eligible_attempts": int(part.eligible_for_outcome_comparison.sum()),
        "unknown_attempts": int(state.eq(AnalysisState.UNKNOWN.value).sum()),
        "censored_attempts": int(state.eq(AnalysisState.INELIGIBLE_CENSORED.value).sum()),
        "not_explicitly_tagged_attempts": int(state.eq(AnalysisState.NOT_EXPLICITLY_TAGGED.value).sum()),
        "share_of_attempts": None if denominator == 0 else attempts / denominator,
        "coverage_denominator": attempts,
        "coverage_rate": None if attempts == 0 else observable / attempts,
    }


def _ordered(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    orders = {
        "dimension": ("total", "serve_number", "direction", "surface", "derived_period", "validation_fold", "serve_number_state", "serve_number_direction_state", "surface_state", "derived_period_state"),
        "serve_number": (0, 1, 2),
        "direction": ("ALL",) + DIRECTION_ORDER,
        "surface": ("ALL",) + ALLOWED_SURFACES,
        "derived_period": ("ALL",) + PERIOD_ORDER,
        "validation_fold": ("ALL",) + FOLD_ORDER,
        "analysis_state": ("ALL",) + STATE_ORDER,
    }
    work = frame.copy()
    temporary: list[str] = []
    for column in columns:
        if column in orders:
            if not set(work[column]).issubset(set(orders[column])):
                raise FeasibilityContractError(f"Dominio inesperado en {column}.")
            temporary_name = f"__{column}"
            work[temporary_name] = pd.Categorical(work[column], categories=orders[column], ordered=True)
            temporary.append(temporary_name)
    return work.sort_values([f"__{column}" if column in orders else column for column in columns], kind="stable").drop(columns=temporary).reset_index(drop=True)


def build_by_state(attempts: pd.DataFrame) -> pd.DataFrame:
    total = len(attempts)
    rows = []
    for (state, reason), part in attempts.groupby(["analysis_state", "reason_code"], sort=False, observed=True):
        rows.append({"analysis_state": state, "reason_code": reason, **_stats(part, total)})
    return _ordered(pd.DataFrame(rows, columns=BY_STATE_COLUMNS), ["analysis_state", "reason_code"])


def _group_rows(attempts: pd.DataFrame, dimension: str, keys: list[str]) -> list[dict[str, Any]]:
    total = len(attempts)
    rows = []
    groups = [((), attempts)] if not keys else attempts.groupby(keys, sort=False, observed=True)
    for key, part in groups:
        values = key if isinstance(key, tuple) else (key,)
        attributes = dict(zip(keys, values))
        rows.append({
            "dimension": dimension,
            "serve_number": attributes.get("serve_number", 0),
            "direction": attributes.get("direction", "ALL"),
            "surface": attributes.get("surface", "ALL"),
            "derived_period": attributes.get("derived_period", "ALL"),
            "validation_fold": attributes.get("validation_fold", "ALL"),
            "analysis_state": attributes.get("analysis_state", "ALL"),
            **_stats(part, total if dimension == "total" else len(attempts) if len(keys) == 1 else int(len(attempts))),
        })
    return rows


def build_by_group(attempts: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("total", []), ("serve_number", ["serve_number"]), ("direction", ["direction"]),
        ("surface", ["surface"]), ("derived_period", ["derived_period"]),
        ("validation_fold", ["validation_fold"]), ("serve_number_state", ["serve_number", "analysis_state"]),
        ("serve_number_direction_state", ["serve_number", "direction", "analysis_state"]),
        ("surface_state", ["surface", "analysis_state"]), ("derived_period_state", ["derived_period", "analysis_state"]),
    ]
    rows = [row for dimension, keys in specs for row in _group_rows(attempts, dimension, keys)]
    return _ordered(pd.DataFrame(rows, columns=BY_GROUP_COLUMNS), ["dimension", "serve_number", "direction", "surface", "derived_period", "validation_fold", "analysis_state"])


def _wilson(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return centre - spread, centre + spread


def build_outcomes(attempts: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    comparable = attempts.loc[
        attempts.analysis_state.isin((AnalysisState.POSITIVE_TAGGED.value, AnalysisState.NOT_EXPLICITLY_TAGGED.value))
        & attempts.direction.ne("unknown")
        & attempts.eligible_for_outcome_comparison
    ]
    required_states = {AnalysisState.POSITIVE_TAGGED.value, AnalysisState.NOT_EXPLICITLY_TAGGED.value}
    if set(comparable.analysis_state) != required_states:
        return pd.DataFrame(columns=OUTCOME_COLUMNS), "not_available:no_observable_untagged_comparator"
    rows = []
    rates: dict[str, float] = {}
    for state in (AnalysisState.POSITIVE_TAGGED.value, AnalysisState.NOT_EXPLICITLY_TAGGED.value):
        part = comparable.loc[comparable.analysis_state.eq(state)]
        wins = int(part.server_won_point.sum())
        rate = wins / len(part)
        rates[state] = rate
        lower, upper = _wilson(wins, len(part))
        rows.append({
            "analysis_state": state, "attempts": int(len(part)), "matches": int(part.match_id.nunique()),
            "servers": int(part.server_player.nunique()), "server_point_wins": wins,
            "server_point_win_rate": rate, "wilson_95_lower": lower, "wilson_95_upper": upper,
            "difference_vs_not_explicitly_tagged": None,
        })
    for row in rows:
        if row["analysis_state"] == AnalysisState.POSITIVE_TAGGED.value:
            row["difference_vs_not_explicitly_tagged"] = rates[AnalysisState.POSITIVE_TAGGED.value] - rates[AnalysisState.NOT_EXPLICITLY_TAGGED.value]
        else:
            row["difference_vs_not_explicitly_tagged"] = 0.0
    return _ordered(pd.DataFrame(rows, columns=OUTCOME_COLUMNS), ["analysis_state"]), "available_descriptive_comparison"


def _reason_counts(attempts: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in attempts.reason_code.value_counts(sort=False).sort_index().items()}


def _state_counts(attempts: pd.DataFrame) -> dict[str, int]:
    return {state: int(attempts.analysis_state.eq(state).sum()) for state in STATE_ORDER}


def _expected_coverage_summary(
    *,
    total: int,
    positive: int,
    not_explicitly_tagged: int,
    explicit_markers: int,
    explicit_markers_unknown: int,
    explicit_markers_censored: int,
    unknown: int,
    censored: int,
) -> dict[str, Any]:
    """Deriva denominadores y numeradores sin confundir etiqueta y comparador."""
    comparable = positive + not_explicitly_tagged
    return {
        "marker_coverage_definition": "attempts_with_explicit_immediate_marker_observed",
        "marker_coverage_denominator": total,
        "marker_determinable_attempts": explicit_markers,
        "marker_coverage_rate": None if total == 0 else explicit_markers / total,
        "comparative_coverage_definition": "attempts_in_positive_tagged_or_not_explicitly_tagged",
        "comparative_coverage_denominator": total,
        "comparative_attempts": comparable,
        "comparative_coverage_rate": None if total == 0 else comparable / total,
        "explicit_marker_attempts": explicit_markers,
        "explicit_marker_rate": None if total == 0 else explicit_markers / total,
        "explicit_marker_positive_tagged_attempts": positive,
        "explicit_marker_unknown_attempts": explicit_markers_unknown,
        "explicit_marker_censored_attempts": explicit_markers_censored,
        "positive_tagged_attempts": positive,
        "positive_tagged_rate": None if total == 0 else positive / total,
        "censored_attempts": censored,
        "unknown_attempts": unknown,
    }


def _validate_coverage_summary(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Comprueba enteros, definiciones y tasas derivadas con precisión estable."""
    for key, value in expected.items():
        actual = observed.get(key)
        if isinstance(value, float):
            if actual is None or not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-15):
                raise FeasibilityContractError(f"Cobertura invalida en {key}.")
        elif actual != value:
            raise FeasibilityContractError(f"Cobertura invalida en {key}.")


def _summary(
    source_counts: Mapping[str, int], attempts: pd.DataFrame, by_state: pd.DataFrame,
    by_group: pd.DataFrame, outcomes: pd.DataFrame, outcome_status: str,
) -> dict[str, Any]:
    states = _state_counts(attempts)
    total = len(attempts)
    explicit_markers = int(attempts.explicit_intent_tagged.sum())
    explicit_markers_unknown = int((attempts.explicit_intent_tagged & attempts.analysis_state.eq(AnalysisState.UNKNOWN.value)).sum())
    explicit_markers_censored = int((attempts.explicit_intent_tagged & attempts.analysis_state.eq(AnalysisState.INELIGIBLE_CENSORED.value)).sum())
    status = "available_descriptive" if outcome_status == "available_descriptive_comparison" else "available_descriptive_not_comparable"
    terminal = attempts.terminal_serve_outcome.fillna("none").value_counts(sort=False).sort_index()
    warning_codes: dict[str, int] = {}
    # Solo se publican agregados; los codigos no exponen secuencias individuales.
    for codes in attempts.warning_codes:
        for code in codes:
            warning_codes[code] = warning_codes.get(code, 0) + 1
    return {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "pattern_id": PATTERN_ID,
        "analysis_status": status,
        "source": {"path": "data/processed/points_enriched.parquet", "columns_used": SOURCE_COLUMNS, "physical_reads": 1},
        "upstream_contracts": {
            "extractor_commit": "16d25c0",
            "chronological_commit": EXPECTED_CHRONOLOGY_COMMIT,
            "chronological_binary_sha256_lf": EXPECTED_CHRONOLOGY_BINARY_SHA256_LF,
            "chronological_canonical_json_sha256": EXPECTED_CHRONOLOGY_CANONICAL_SHA256,
            "chronological_publication_fingerprint": EXPECTED_CHRONOLOGY_PUBLICATION_FINGERPRINT,
        },
        "definition": {"pattern": "literal_plus_immediately_after_service_prefix", "criterion": "marker.start == service_prefix.end", "meaning": "explicitly_annotated_intent", "physical_serve_and_volley_proven": False, "causal_interpretation": False},
        "unit": {"name": "serve_attempt_within_point", "key": ["match_id", "point_number", "serve_number"], "second_attempt_rule": "only_substantive_second_serve"},
        "population": {**dict(source_counts), "attempts": int(total), "first_serve_attempts": int(attempts.serve_number.eq(1).sum()), "second_serve_attempts": int(attempts.serve_number.eq(2).sum())},
        "exclusions": {"second_serve_non_substantive_not_constructed": True, "unknown_not_recoded_as_negative": True, "censored_not_recoded_as_negative": True, "unknown_direction_excluded_from_outcome_comparison": True},
        "test_seal": {"test_status": "sealed", "used_for_method_selection": False, **{field: 0 for field in TEST_ZERO_FIELDS}},
        "classification_contract": {"states": list(STATE_ORDER), "directions": list(DIRECTION_ORDER), "extractor_reused": True, "positive_is_explicit_tag_only": True},
        "state_counts": states,
        "reason_code_counts": _reason_counts(attempts),
        "coverage_summary": _expected_coverage_summary(
            total=int(total),
            positive=states[AnalysisState.POSITIVE_TAGGED.value],
            not_explicitly_tagged=states[AnalysisState.NOT_EXPLICITLY_TAGGED.value],
            explicit_markers=explicit_markers,
            explicit_markers_unknown=explicit_markers_unknown,
            explicit_markers_censored=explicit_markers_censored,
            unknown=states[AnalysisState.UNKNOWN.value],
            censored=states[AnalysisState.INELIGIBLE_CENSORED.value],
        ),
        "outcome_comparison_status": outcome_status,
        "outcomes_summary": [] if outcomes.empty else outcomes.to_dict("records"),
        "warnings_and_residuals": {
            "attempts_with_warnings": int(attempts.has_warnings.sum()),
            "attempts_with_residual": int(attempts.has_residual.sum()),
            "explicit_marker_with_residual": int((attempts.explicit_intent_tagged & attempts.has_residual).sum()),
            "unknown_by_boundary_residual": int(((attempts.analysis_state == AnalysisState.UNKNOWN.value) & (attempts.reason_code == ReasonCode.AMBIGUOUS_POST_PREFIX.value)).sum()),
            "warning_state_counts": warning_codes,
            "terminal_serve_outcome_counts": {str(key): int(value) for key, value in terminal.items()},
            "explicit_marker_in_unknown_attempts": explicit_markers_unknown,
            "explicit_marker_in_censored_attempts": explicit_markers_censored,
        },
        "reconciliations": validate_reconciliations(attempts, by_state, by_group, outcomes, outcome_status),
        "methodological_limits": METHODOLOGICAL_LIMITS,
        "artifact_contracts": {"csv_null_representation": "<NULL>", "utf8": True, "index": False, "timestamps": False, "tables": {"by_state": BY_STATE_COLUMNS, "by_group": BY_GROUP_COLUMNS, "outcomes": OUTCOME_COLUMNS}},
        "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
    }


def analyze_points(points: pd.DataFrame) -> DescriptiveResult:
    source = validate_source_points(points)
    development, source_counts = split_development_before_parsing(source)
    attempts = construct_attempts(development)
    by_state = build_by_state(attempts)
    by_group = build_by_group(attempts)
    outcomes, outcome_status = build_outcomes(attempts)
    summary = _summary(source_counts, attempts, by_state, by_group, outcomes, outcome_status)
    payloads = tuple(_frame_bytes(frame) for frame in (by_state, by_group, outcomes))
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    fingerprint = _fingerprint(summary, payloads)
    summary["publication_fingerprint"] = fingerprint
    result = DescriptiveResult(summary, by_state, by_group, outcomes, development, attempts, fingerprint)
    validate_result(result)
    return result


def validate_reconciliations(attempts: pd.DataFrame, by_state: pd.DataFrame, by_group: pd.DataFrame, outcomes: pd.DataFrame, outcome_status: str) -> dict[str, bool]:
    if attempts.duplicated(["match_id", "point_number", "serve_number"]).any():
        raise FeasibilityContractError("Duplicado de clave de intento.")
    if not attempts.direction.isin(DIRECTION_ORDER).all() or not attempts.surface.isin(ALLOWED_SURFACES).all() or not attempts.derived_period.isin(PERIOD_ORDER).all() or not attempts.validation_fold.isin(FOLD_ORDER).all():
        raise FeasibilityContractError("Dominio descriptivo invalido.")
    if attempts.loc[attempts.analysis_state.isin((AnalysisState.UNKNOWN.value, AnalysisState.INELIGIBLE_CENSORED.value)), "eligible_for_outcome_comparison"].any():
        raise FeasibilityContractError("Unknown/censored no pueden ser elegibles.")
    expected_state, expected_group = build_by_state(attempts), build_by_group(attempts)
    try:
        pdt.assert_frame_equal(by_state.reset_index(drop=True), expected_state, check_dtype=False)
        pdt.assert_frame_equal(by_group.reset_index(drop=True), expected_group, check_dtype=False)
    except AssertionError as error:
        raise FeasibilityContractError("Agregados no coinciden con derivacion canonica.") from error
    if int(by_state.attempts.sum()) != len(attempts):
        raise FeasibilityContractError("Estados no reconcilian intentos.")
    if not by_state.groupby("analysis_state", sort=False).attempts.sum().reindex(STATE_ORDER, fill_value=0).eq(pd.Series(_state_counts(attempts)).reindex(STATE_ORDER)).all():
        raise FeasibilityContractError("Estados no reconcilian por categoria.")
    if outcome_status.startswith("not_available") and not outcomes.empty:
        raise FeasibilityContractError("No se permite outcomes sin comparador observable.")
    if outcome_status == "available_descriptive_comparison":
        expected_outcomes, _ = build_outcomes(attempts)
        try:
            pdt.assert_frame_equal(outcomes.reset_index(drop=True), expected_outcomes, check_dtype=False)
        except AssertionError as error:
            raise FeasibilityContractError("Outcomes no reconcilian.") from error
    return {
        "attempt_key_unique": True, "states_exhaustive": True, "state_reason_reconciled": True,
        "groups_rebuilt_from_attempts": True, "outcomes_not_compared_with_unknown_or_censored": True,
        "test_attempts_zero": True,
    }


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8", na_rep="<NULL>", lineterminator="\n").encode("utf-8")


def _fingerprint(summary: Mapping[str, Any], payloads: Sequence[bytes]) -> str:
    core = copy.deepcopy(dict(summary))
    core.pop("publication_fingerprint", None)
    digest = hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for payload in payloads:
        digest.update(payload)
    return digest.hexdigest().upper()


def validate_result(result: DescriptiveResult) -> None:
    status = result.summary.get("analysis_status")
    if status not in {"available_descriptive", "available_descriptive_not_comparable", "not_available"}:
        raise FeasibilityContractError("analysis_status invalido.")
    for frame, columns in zip((result.by_state, result.by_group, result.outcomes), (BY_STATE_COLUMNS, BY_GROUP_COLUMNS, OUTCOME_COLUMNS)):
        if frame.columns.tolist() != columns or any(str(value).startswith("Unnamed") for value in frame.columns):
            raise FeasibilityContractError("Schema de artefacto invalido.")
    seal = result.summary.get("test_seal", {})
    if seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS):
        raise FeasibilityContractError("Sellado del test invalido.")
    if status == "not_available":
        if any(not frame.empty for frame in (result.by_state, result.by_group, result.outcomes)):
            raise FeasibilityContractError("not_available no puede publicar filas.")
    else:
        validate_reconciliations(result.attempts, result.by_state, result.by_group, result.outcomes, result.summary["outcome_comparison_status"])
        counts = result.summary.get("state_counts", {})
        if counts != _state_counts(result.attempts):
            raise FeasibilityContractError("Summary de estados no reconcilia.")
        if result.summary.get("reason_code_counts") != _reason_counts(result.attempts):
            raise FeasibilityContractError("Summary de reason codes no reconcilia.")
        population = result.summary.get("population", {})
        if population.get("attempts") != len(result.attempts) or population.get("first_serve_attempts") != int(result.attempts.serve_number.eq(1).sum()) or population.get("second_serve_attempts") != int(result.attempts.serve_number.eq(2).sum()):
            raise FeasibilityContractError("Summary de intentos primero/segundo no reconcilia.")
        explicit = int(result.attempts.explicit_intent_tagged.sum())
        explicit_unknown = int((result.attempts.explicit_intent_tagged & result.attempts.analysis_state.eq(AnalysisState.UNKNOWN.value)).sum())
        explicit_censored = int((result.attempts.explicit_intent_tagged & result.attempts.analysis_state.eq(AnalysisState.INELIGIBLE_CENSORED.value)).sum())
        _validate_coverage_summary(
            result.summary.get("coverage_summary", {}),
            _expected_coverage_summary(
                total=len(result.attempts),
                positive=counts[AnalysisState.POSITIVE_TAGGED.value],
                not_explicitly_tagged=counts[AnalysisState.NOT_EXPLICITLY_TAGGED.value],
                explicit_markers=explicit,
                explicit_markers_unknown=explicit_unknown,
                explicit_markers_censored=explicit_censored,
                unknown=counts[AnalysisState.UNKNOWN.value],
                censored=counts[AnalysisState.INELIGIBLE_CENSORED.value],
            ),
        )
        if status == "available_descriptive" and result.summary["outcome_comparison_status"] != "available_descriptive_comparison":
            raise FeasibilityContractError("Estado disponible requiere comparador publicable.")
        if status == "available_descriptive_not_comparable" and not result.outcomes.empty:
            raise FeasibilityContractError("Estado no comparable no publica outcomes.")
    payloads = tuple(_frame_bytes(frame) for frame in (result.by_state, result.by_group, result.outcomes))
    hashes = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    sizes = {name: len(payload) for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    if result.summary.get("artifact_payload_sha256") != hashes or result.summary.get("artifact_payload_bytes") != sizes:
        raise FeasibilityContractError("Hashes o tamanos no reconcilian.")
    fingerprint = _fingerprint(result.summary, payloads)
    if result.publication_fingerprint != fingerprint or result.summary.get("publication_fingerprint") != fingerprint:
        raise FeasibilityContractError("Fingerprint invalido.")


def serialize_artifacts(result: DescriptiveResult) -> tuple[bytes, bytes, bytes, bytes]:
    validate_result(result)
    summary = (json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return (summary, _frame_bytes(result.by_state), _frame_bytes(result.by_group), _frame_bytes(result.outcomes))


def _stage(path: Path, payload: bytes) -> Path:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return Path(temporary)


def write_artifacts(
    result: DescriptiveResult,
    *,
    summary_path: Path = SUMMARY_PATH,
    by_state_path: Path = BY_STATE_PATH,
    by_group_path: Path = BY_GROUP_PATH,
    outcomes_path: Path = OUTCOMES_PATH,
    payloads: tuple[bytes, bytes, bytes, bytes] | None = None,
) -> None:
    """Publica cuatro archivos; excepciones normales restauran bytes previos.

    Un apagado abrupto entre ``os.replace`` sigue siendo una limitacion del
    filesystem: no existe transaccion de directorio portable para esos cuatro
    destinos.
    """
    targets = (summary_path, by_state_path, by_group_path, outcomes_path)
    payloads = serialize_artifacts(result) if payloads is None else payloads
    if len(payloads) != 4 or any(not isinstance(payload, bytes) for payload in payloads):
        raise FeasibilityContractError("Payloads de publicacion invalidos.")
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    backups: dict[Path, bytes | None] = {}
    try:
        for target, payload in zip(targets, payloads):
            staged.append(_stage(target, payload))
        backups = {target: target.read_bytes() if target.exists() else None for target in targets}
        for temporary, target in zip(staged, targets):
            os.replace(temporary, target)
    except BaseException:
        for target, previous in backups.items():
            if previous is None:
                if target.exists():
                    target.unlink()
            else:
                target.write_bytes(previous)
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def verify_persisted_artifacts(
    summary_path: Path = SUMMARY_PATH,
    by_state_path: Path = BY_STATE_PATH,
    by_group_path: Path = BY_GROUP_PATH,
    outcomes_path: Path = OUTCOMES_PATH,
    *,
    result: DescriptiveResult | None = None,
    expected_payloads: tuple[bytes, bytes, bytes, bytes] | None = None,
) -> None:
    summary_payload = summary_path.read_bytes()
    summary = json.loads(summary_payload.decode("utf-8"))
    payloads = tuple(path.read_bytes() for path in (by_state_path, by_group_path, outcomes_path))
    canonical_paths = tuple(path.resolve() for path in (summary_path, by_state_path, by_group_path, outcomes_path)) == tuple(path.resolve() for path in (SUMMARY_PATH, BY_STATE_PATH, BY_GROUP_PATH, OUTCOMES_PATH))
    if canonical_paths:
        published_hashes = {
            "summary": hashlib.sha256(summary_payload).hexdigest().upper(),
            **{name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)},
        }
        if published_hashes != EXPECTED_PUBLISHED_ARTIFACT_SHA256:
            raise FeasibilityContractError("Los artefactos P03 publicados no coinciden con el contrato congelado.")
    frames = tuple(pd.read_csv(path, keep_default_na=False, na_values=["<NULL>"]) for path in (by_state_path, by_group_path, outcomes_path))
    for frame, columns in zip(frames, (BY_STATE_COLUMNS, BY_GROUP_COLUMNS, OUTCOME_COLUMNS)):
        if frame.columns.tolist() != columns:
            raise FeasibilityContractError("Schema persistido invalido.")
    hashes = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    sizes = {name: len(payload) for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    if summary.get("artifact_payload_sha256") != hashes or summary.get("artifact_payload_bytes") != sizes or summary.get("publication_fingerprint") != _fingerprint(summary, payloads):
        raise FeasibilityContractError("Contrato persistido de fingerprint invalido.")
    if result is not None:
        replay = serialize_artifacts(result)
        if expected_payloads is not None and replay != expected_payloads:
            raise FeasibilityContractError("Reserializacion no determinista.")
        if replay != (summary_payload, *payloads):
            raise FeasibilityContractError("Los permanentes no coinciden con el mismo objeto.")
    # Validacion semantica de artefactos sin reabrir la fuente.
    if summary.get("analysis_name") != ANALYSIS_NAME or summary.get("analysis_version") != ANALYSIS_VERSION or summary.get("pattern_id") != PATTERN_ID:
        raise FeasibilityContractError("Identidad persistida invalida.")
    if summary.get("source") != {"path": "data/processed/points_enriched.parquet", "columns_used": SOURCE_COLUMNS, "physical_reads": 1}:
        raise FeasibilityContractError("Contrato de fuente persistido invalido.")
    expected_upstream = {
        "extractor_commit": "16d25c0", "chronological_commit": EXPECTED_CHRONOLOGY_COMMIT,
        "chronological_binary_sha256_lf": EXPECTED_CHRONOLOGY_BINARY_SHA256_LF,
        "chronological_canonical_json_sha256": EXPECTED_CHRONOLOGY_CANONICAL_SHA256,
        "chronological_publication_fingerprint": EXPECTED_CHRONOLOGY_PUBLICATION_FINGERPRINT,
    }
    if summary.get("upstream_contracts") != expected_upstream:
        raise FeasibilityContractError("Contrato upstream persistido invalido.")
    if summary.get("definition") != {"pattern": "literal_plus_immediately_after_service_prefix", "criterion": "marker.start == service_prefix.end", "meaning": "explicitly_annotated_intent", "physical_serve_and_volley_proven": False, "causal_interpretation": False}:
        raise FeasibilityContractError("Definicion persistida invalida.")
    if summary.get("methodological_limits") != METHODOLOGICAL_LIMITS:
        raise FeasibilityContractError("Limitaciones persistidas invalidas.")
    seal = summary.get("test_seal", {})
    if seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS):
        raise FeasibilityContractError("Sellado persistido invalido.")
    status = summary.get("analysis_status")
    if status not in {"available_descriptive", "available_descriptive_not_comparable", "not_available"}:
        raise FeasibilityContractError("Estado persistido invalido.")
    if status == "available_descriptive_not_comparable":
        if summary.get("outcome_comparison_status") != "not_available:no_observable_untagged_comparator" or not frames[2].empty:
            raise FeasibilityContractError("Outcomes persistidos incompatibles con no comparabilidad.")
    elif status == "available_descriptive":
        if summary.get("outcome_comparison_status") != "available_descriptive_comparison" or frames[2].empty:
            raise FeasibilityContractError("Estado comparable persistido invalido.")
    elif any(not frame.empty for frame in frames):
        raise FeasibilityContractError("not_available persistido no puede publicar filas.")
    if frames[0].duplicated(["analysis_state", "reason_code"]).any() or frames[1].duplicated(["dimension", "serve_number", "direction", "surface", "derived_period", "validation_fold", "analysis_state"]).any():
        raise FeasibilityContractError("Claves persistidas duplicadas.")
    if not frames[0].equals(_ordered(frames[0], ["analysis_state", "reason_code"])) or not frames[1].equals(_ordered(frames[1], ["dimension", "serve_number", "direction", "surface", "derived_period", "validation_fold", "analysis_state"])):
        raise FeasibilityContractError("Orden persistido invalido.")
    valid_reasons = {
        state.value: {reason.value for reason in ReasonCode if reason in {
            AnalysisState.POSITIVE_TAGGED: {ReasonCode.EXPLICIT_MARKER},
            AnalysisState.NOT_EXPLICITLY_TAGGED: {ReasonCode.NO_IMMEDIATE_MARKER},
            AnalysisState.UNKNOWN: {ReasonCode.UNKNOWN_DIRECTION, ReasonCode.MISSING_PREFIX, ReasonCode.AMBIGUOUS_POST_PREFIX},
            AnalysisState.INELIGIBLE_CENSORED: {ReasonCode.CENSORED_ACE, ReasonCode.CENSORED_UNRETURNED, ReasonCode.CENSORED_FAULT, ReasonCode.CENSORED_DOUBLE_FAULT, ReasonCode.CENSORED_SPECIAL, ReasonCode.CENSORED_INCOMPLETE_LET},
        }[state]}
        for state in AnalysisState
    }
    if any(row.reason_code not in valid_reasons.get(row.analysis_state, set()) for row in frames[0].itertuples(index=False)):
        raise FeasibilityContractError("Reason codes persistidos incompatibles con su estado.")
    metric_columns = ("attempts", "matches", "servers", "explicit_tagged_attempts", "eligible_attempts", "unknown_attempts", "censored_attempts", "not_explicitly_tagged_attempts", "coverage_denominator")
    for frame in frames[:2]:
        for column in metric_columns:
            if frame[column].isna().any() or not np.isfinite(frame[column].astype(float)).all() or not frame[column].astype(float).eq(frame[column].astype(float).round()).all() or frame[column].astype(float).lt(0).any():
                raise FeasibilityContractError(f"Metrica persistida invalida en {column}.")
        if frame.share_of_attempts.isna().any() or frame.coverage_rate.isna().any() or not frame.share_of_attempts.between(0, 1).all() or not frame.coverage_rate.between(0, 1).all():
            raise FeasibilityContractError("Tasas persistidas invalidas.")
    states = {state: int(frames[0].loc[frames[0].analysis_state.eq(state), "attempts"].sum()) for state in STATE_ORDER}
    reason_counts = {str(row.reason_code): int(row.attempts) for row in frames[0].itertuples(index=False)}
    if summary.get("state_counts") != states or summary.get("reason_code_counts") != reason_counts or int(frames[0].attempts.sum()) != summary.get("population", {}).get("attempts"):
        raise FeasibilityContractError("Estados persistidos no reconcilian.")
    if not set(frames[0].analysis_state).issubset(STATE_ORDER) or not set(frames[1].analysis_state).issubset(("ALL",) + STATE_ORDER):
        raise FeasibilityContractError("Dominios de estado persistidos invalidos.")
    total_from_states = int(frames[0].attempts.sum())
    for row in frames[0].itertuples(index=False):
        if not math.isclose(float(row.share_of_attempts), int(row.attempts) / total_from_states, rel_tol=0, abs_tol=1e-15) or int(row.coverage_denominator) != int(row.attempts) or not math.isclose(float(row.coverage_rate), int(row.eligible_attempts) / int(row.attempts), rel_tol=0, abs_tol=1e-15):
            raise FeasibilityContractError("Denominadores persistidos de estados invalidos.")
        expected_components = {
            AnalysisState.POSITIVE_TAGGED.value: (int(row.attempts), int(row.attempts), 0, 0, 0),
            AnalysisState.NOT_EXPLICITLY_TAGGED.value: (0, int(row.attempts), 0, 0, int(row.attempts)),
            AnalysisState.UNKNOWN.value: (None, 0, int(row.attempts), 0, 0),
            AnalysisState.INELIGIBLE_CENSORED.value: (None, 0, 0, int(row.attempts), 0),
        }[row.analysis_state]
        explicit_expected, eligible_expected, unknown_expected, censored_expected, untagged_expected = expected_components
        if (explicit_expected is not None and int(row.explicit_tagged_attempts) != explicit_expected) or int(row.eligible_attempts) != eligible_expected or int(row.unknown_attempts) != unknown_expected or int(row.censored_attempts) != censored_expected or int(row.not_explicitly_tagged_attempts) != untagged_expected:
            raise FeasibilityContractError("Componentes persistidos incompatibles con analysis_state.")
    if not set(frames[1].direction).issubset(("ALL",) + DIRECTION_ORDER) or not set(frames[1].surface).issubset(("ALL",) + ALLOWED_SURFACES) or not set(frames[1].derived_period).issubset(("ALL",) + PERIOD_ORDER) or not set(frames[1].validation_fold).issubset(("ALL",) + FOLD_ORDER):
        raise FeasibilityContractError("Dominios de grupo persistidos invalidos.")
    total_rows = frames[1].loc[frames[1].dimension.eq("total")]
    if len(total_rows) != 1 or int(total_rows.iloc[0].attempts) != int(summary["population"]["attempts"]):
        raise FeasibilityContractError("Total persistido de grupos invalido.")
    total_attempts = int(summary["population"]["attempts"])
    for dimension in ("serve_number", "direction", "surface", "derived_period", "validation_fold", "serve_number_state", "serve_number_direction_state", "surface_state", "derived_period_state"):
        part = frames[1].loc[frames[1].dimension.eq(dimension)]
        if part.empty or int(part.attempts.sum()) != total_attempts:
            raise FeasibilityContractError(f"Particion persistida invalida en {dimension}.")
    serve_rows = frames[1].loc[frames[1].dimension.eq("serve_number")]
    if set(serve_rows.serve_number) != {1, 2} or int(serve_rows.loc[serve_rows.serve_number.eq(1), "attempts"].sum()) != int(summary["population"].get("first_serve_attempts")) or int(serve_rows.loc[serve_rows.serve_number.eq(2), "attempts"].sum()) != int(summary["population"].get("second_serve_attempts")):
        raise FeasibilityContractError("Intentos primero/segundo persistidos no reconcilian.")
    if int(total_rows.iloc[0].matches) != int(summary["population"].get("development_target_matches")) or int(total_rows.iloc[0].servers) != int(summary["population"].get("development_servers")):
        raise FeasibilityContractError("Poblacion de desarrollo persistida no reconcilia grupos.")
    if not all(math.isclose(float(row.share_of_attempts), int(row.attempts) / total_attempts, rel_tol=0, abs_tol=1e-15) and int(row.coverage_denominator) == int(row.attempts) and math.isclose(float(row.coverage_rate), int(row.eligible_attempts) / int(row.attempts), rel_tol=0, abs_tol=1e-15) for row in frames[1].itertuples(index=False)):
        raise FeasibilityContractError("Denominadores persistidos de grupos invalidos.")
    explicit = int(frames[0].explicit_tagged_attempts.sum())
    explicit_unknown = int(frames[0].loc[frames[0].analysis_state.eq(AnalysisState.UNKNOWN.value), "explicit_tagged_attempts"].sum())
    explicit_censored = int(frames[0].loc[frames[0].analysis_state.eq(AnalysisState.INELIGIBLE_CENSORED.value), "explicit_tagged_attempts"].sum())
    coverage = summary.get("coverage_summary", {})
    denominator = int(summary["population"]["attempts"])
    _validate_coverage_summary(
        coverage,
        _expected_coverage_summary(
            total=denominator,
            positive=states[AnalysisState.POSITIVE_TAGGED.value],
            not_explicitly_tagged=states[AnalysisState.NOT_EXPLICITLY_TAGGED.value],
            explicit_markers=explicit,
            explicit_markers_unknown=explicit_unknown,
            explicit_markers_censored=explicit_censored,
            unknown=states[AnalysisState.UNKNOWN.value],
            censored=states[AnalysisState.INELIGIBLE_CENSORED.value],
        ),
    )


def publish_artifacts(result: DescriptiveResult, **paths: Path) -> None:
    """Reserializa tres veces el mismo objeto sin reconstruir datos."""
    first = serialize_artifacts(result)
    if serialize_artifacts(result) != first:
        raise FeasibilityContractError("Reserializacion del mismo resultado no determinista antes de publicar.")
    write_artifacts(result, payloads=first, **paths)
    verify_persisted_artifacts(result=result, expected_payloads=first, **paths)


def not_available_result(error: BaseException, stage: str = "source_or_contract") -> DescriptiveResult:
    frames = tuple(pd.DataFrame(columns=columns) for columns in (BY_STATE_COLUMNS, BY_GROUP_COLUMNS, OUTCOME_COLUMNS))
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION, "pattern_id": PATTERN_ID,
        "analysis_status": "not_available", "reason_codes": ["source_or_contract_validation_failed"],
        "failure": {"stage": stage, "exception_type": type(error).__name__, "message": str(error).replace(str(ROOT), "<ROOT>")[:500]},
        "test_seal": {"test_status": "sealed", "used_for_method_selection": False, **{field: 0 for field in TEST_ZERO_FIELDS}},
        "artifact_contracts": {"csv_null_representation": "<NULL>", "utf8": True, "index": False, "tables": {"by_state": BY_STATE_COLUMNS, "by_group": BY_GROUP_COLUMNS, "outcomes": OUTCOME_COLUMNS}},
        "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
    }
    payloads = tuple(_frame_bytes(frame) for frame in frames)
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    fingerprint = _fingerprint(summary, payloads)
    summary["publication_fingerprint"] = fingerprint
    return DescriptiveResult(summary, *frames, pd.DataFrame(columns=SOURCE_COLUMNS), pd.DataFrame(), fingerprint)


def read_source_points(path: Path = POINTS_PATH) -> pd.DataFrame:
    """La unica lectura fisica del Parquet en la ejecucion real."""
    try:
        return pd.read_parquet(path, columns=SOURCE_COLUMNS)
    except (OSError, ValueError, ImportError) as error:
        raise FeasibilityContractError("No se pudo leer la fuente Parquet requerida.") from error


def _validate_frozen_real_population(result: DescriptiveResult) -> None:
    population = result.summary.get("population", {})
    expected = {
        "source_rows_read": EXPECTED_SOURCE_ROWS,
        "source_matches": EXPECTED_SOURCE_MATCHES,
        "source_players": EXPECTED_SOURCE_PLAYERS,
        "development_target_rows": EXPECTED_DEVELOPMENT_ROWS,
        "development_target_matches": EXPECTED_DEVELOPMENT_MATCHES,
        "development_servers": EXPECTED_DEVELOPMENT_SERVERS,
        "excluded_test_target_matches": EXPECTED_TEST_MATCHES,
    }
    observed = {key: population.get(key) for key in expected}
    if observed != expected:
        raise FeasibilityContractError(f"Poblacion real no coincide con el contrato congelado: {observed!r}")


def run_real_analysis() -> DescriptiveResult:
    validate_chronological_contract()
    source = read_source_points()
    result = analyze_points(source)
    _validate_frozen_real_population(result)
    return result


def main() -> None:
    try:
        result = run_real_analysis()
    except BaseException as error:
        result = not_available_result(error)
    publish_artifacts(result)
    print(json.dumps({"analysis_status": result.summary["analysis_status"], "publication_fingerprint": result.publication_fingerprint}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
