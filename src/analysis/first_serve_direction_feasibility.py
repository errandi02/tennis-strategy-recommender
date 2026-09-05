"""Estudio descriptivo y sellado de viabilidad para P02, direccion del primer saque."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pyarrow.dataset as ds

from src.analysis.second_serve_direction_analysis import derive_period
from src.parsing.serve_sequence import (
    ServicePrefix, ServeAce, ServeFault, ServeUnreturned,
    SpecialCodeStructure, parse_sequence,
)
from src.parsing.serve_sequence_rules import GRAMMAR_VERSION, PARSER_VERSION


ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = ROOT / "data" / "processed" / "points_enriched.parquet"
CHRONOLOGY_PATH = ROOT / "reports" / "chronological_validation_summary.json"
CHRONOLOGY_BY_YEAR_PATH = ROOT / "reports" / "tables" / "chronological_validation_by_year.csv"
CHRONOLOGY_FOLDS_PATH = ROOT / "reports" / "tables" / "chronological_validation_folds.csv"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "first_serve_direction_feasibility_summary.json"
BY_DIRECTION_PATH = TABLES_DIR / "first_serve_direction_feasibility_by_direction.csv"
OUTCOMES_PATH = TABLES_DIR / "first_serve_direction_feasibility_outcomes.csv"
COVERAGE_PATH = TABLES_DIR / "first_serve_direction_feasibility_coverage.csv"

ANALYSIS_NAME = "first_serve_direction_feasibility"
ANALYSIS_VERSION = "1.0.0"
PATTERN_ID = "P02"
CUTOFF = pd.Timestamp("2023-12-31")
EXPECTED_CHRONOLOGY_COMMIT = "3ac28b4d960e5b2e5e54dd66cb6a8a866f953712"
EXPECTED_CHRONOLOGY_BLOB = "51dc59ed4f726ad7de6aabb0bd54639abe0afbff"
EXPECTED_CHRONOLOGY_BINARY_SHA256_LF = "90890B721CC93C82A5DE5F2EB6E3F3D4E5D4D3E83E6C9E678C5569139301799F"
EXPECTED_CHRONOLOGY_CANONICAL_SHA256 = "801627A74D253B0F65C5EC458C7814CCF99491FB171DE2E2A5DA47FFABD13A51"
EXPECTED_CHRONOLOGY_PUBLICATION_FINGERPRINT = "E889A679E08BB16FAF5AB01AA0D63A7623BE0825A0F945626A1D19301C8DFEAD"
EXPECTED_DEVELOPMENT_MATCHES = 5_993
EXPECTED_TEST_EXCLUDED = 1_531
PUBLISHED_ARTIFACT_CONTRACT = {
    "summary": (24_698, "6BC1498AE7C65E079428F90ACFD53665ECD815459600ABFF658576BC0CEC5F0B"),
    "by_direction": (6_496, "F6E364002DB9C965CE50819CDE221243B53906FBF9D1CDC4FE293AA75E68A9B8"),
    "outcomes": (14_115, "5A7F7CDDBA9ADAB5CEA6ED892EAB4368844FF16F9BF855EF11DB6BD8BCE74C05"),
    "coverage": (32_071, "5D64E331BFDFD5E09B94217175407A7F7B15119680794BDD12F37E9A61692911"),
}
PUBLISHED_PUBLICATION_FINGERPRINT = "A5820C693AE572654185257350D207019E2630BC3275F5A938F370E11F4A111A"
PUBLISHED_POPULATION = {
    "development_point_rows": 1_035_760,
    "development_matches": 5_993,
    "development_servers": 870,
    "directional_points": 1_019_891,
    "directional_matches": 5_993,
    "directional_servers": 870,
    "server_wins": 652_794,
}
ALLOWED_SURFACES = ("Hard", "Clay", "Grass")
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
DIRECTION_ORDER = ("wide", "body", "T")
DIRECTION_CODES = {"4": "wide", "5": "body", "6": "T"}
SPECIAL_CODES = ("S", "R", "P", "Q", "V")
IMMEDIATE_OUTCOMES = ("ace", "unreturned", "fault", "continuation_or_other")
POINT_CUTS = (25, 50, 100)
MATCH_CUTS = (3, 5, 10)
TEST_ZERO_FIELDS = (
    "test_target_rows_constructed", "test_target_matches_constructed",
    "test_feature_rows_constructed", "test_recommendation_rows",
    "test_points_scored", "test_matches_scored", "test_rows_evaluated",
    "test_matches_evaluated", "test_evaluation_runs",
)
SOURCE_COLUMNS = [
    "match_id", "point_number", "date", "surface", "server", "point_winner",
    "player_1", "player_2", "first_serve",
]
BY_DIRECTION_COLUMNS = [
    "dimension", "direction", "surface", "derived_period", "analytical_points",
    "matches", "servers", "server_wins", "server_win_rate",
    "proportion_of_directional_population", "ace_points", "unreturned_points",
    "fault_points", "continuation_or_other_points", "parser_warning_points",
    "residual_points", "fully_consumed_points",
]
OUTCOME_COLUMNS = [
    "dimension", "direction", "surface", "derived_period", "immediate_outcome",
    "points", "matches", "servers", "server_wins", "server_win_rate",
]
COVERAGE_COLUMNS = [
    "scope", "surface", "direction", "cut_kind", "point_cut", "match_cut",
    "directional_points", "directional_matches", "directional_server_wins",
    "first_date", "last_date",
    "possible_servers", "observed_servers", "zero_evidence_servers",
    "eligible_servers", "eligible_server_rate", "eligible_matches",
    "min_points", "p25_points", "median_points", "p75_points", "p90_points",
    "max_points", "min_matches", "p25_matches", "median_matches", "p75_matches",
    "p90_matches", "max_matches",
]


class FeasibilityContractError(ValueError):
    """Contrato de datos, resultados o artefactos incumplido."""


@dataclass(frozen=True)
class FeasibilityResult:
    summary: dict[str, Any]
    by_direction: pd.DataFrame
    outcomes: pd.DataFrame
    coverage: pd.DataFrame
    development: pd.DataFrame
    directional: pd.DataFrame
    publication_fingerprint: str


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value not in (1, 2):
        raise FeasibilityContractError(f"Dominio invalido en {field}: {value!r}")
    return int(value)


def _presence(value: object) -> str:
    if value is None or value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return "null"
    if not isinstance(value, str):
        raise FeasibilityContractError("first_serve debe ser cadena o nulo.")
    if value == "":
        return "empty"
    if value.isspace():
        return "whitespace_only"
    return "substantive"


def _exact_text(series: pd.Series, name: str) -> None:
    if series.isna().any() or not series.map(lambda x: isinstance(x, str) and x != "" and x == x.strip()).all():
        raise FeasibilityContractError(f"{name} debe contener texto exacto no vacio.")


def _reject_json_constant(value: str) -> None:
    raise FeasibilityContractError(f"JSON cronologico contiene constante no finita: {value}.")


def canonical_json_bytes(payload: bytes) -> bytes:
    """JSON UTF-8 compacto, sin BOM, con claves ordenadas y sin no-finitos.

    Es deliberadamente independiente de LF/CRLF y del orden textual de las
    claves, pero no de ningun valor semantico del contrato.
    """
    if payload.startswith(b"\xef\xbb\xbf"):
        raise FeasibilityContractError("El JSON cronologico no puede contener BOM.")
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeasibilityContractError("JSON cronologico invalido.") from error
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FeasibilityContractError("JSON cronologico no es canonico finito.") from error


def canonical_json_sha256(payload: bytes) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def _assert_no_unsafe_chronology_values(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FeasibilityContractError(f"Valor no finito en {path}.")
    if isinstance(value, str):
        if re.search(r"(?:[A-Za-z]:[\\/]|^/|\\\\)", value):
            raise FeasibilityContractError(f"Ruta absoluta en {path}.")
        if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", value):
            raise FeasibilityContractError(f"Timestamp variable en {path}.")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_unsafe_chronology_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_unsafe_chronology_values(item, f"{path}[{index}]")


def _validate_chronology_commit() -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", EXPECTED_CHRONOLOGY_COMMIT):
        raise FeasibilityContractError("Commit cronologico congelado invalido.")
    result = subprocess.run(
        ["git", "rev-parse", f"{EXPECTED_CHRONOLOGY_COMMIT}:reports/chronological_validation_summary.json"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    blob = result.stdout.decode("ascii", errors="replace").strip()
    if result.returncode != 0 or blob != EXPECTED_CHRONOLOGY_BLOB:
        raise FeasibilityContractError("Commit/blob cronologico upstream inesperado.")


def _validate_chronology_tables(by_year_path: Path, folds_path: Path) -> None:
    try:
        by_year = pd.read_csv(by_year_path)
        folds = pd.read_csv(folds_path)
    except (OSError, pd.errors.ParserError) as error:
        raise FeasibilityContractError("Tablas cronologicas upstream no disponibles.") from error
    required_year = {"year", "split", "is_partial_year", "last_date", "matches"}
    required_fold = {"fold_id", "protocol", "is_final_test", "evaluation_matches"}
    if not required_year.issubset(by_year.columns) or not required_fold.issubset(folds.columns):
        raise FeasibilityContractError("Schema de tablas cronologicas inesperado.")
    if int(by_year["matches"].sum()) != 7_524:
        raise FeasibilityContractError("Tabla anual cronologica no reconcilia partidos.")
    partial_2026 = by_year.loc[by_year["year"].eq(2026)]
    if len(partial_2026) != 1 or not bool(partial_2026.iloc[0]["is_partial_year"]) or str(partial_2026.iloc[0]["last_date"]) != "2026-05-21":
        raise FeasibilityContractError("Tabla anual no identifica 2026 parcial.")
    validation = folds.loc[folds["fold_id"].astype(str).str.startswith("validation_")]
    expected_ids = ["validation_2020", "validation_2021", "validation_2022", "validation_2023"]
    if validation["fold_id"].tolist() != expected_ids or validation["evaluation_matches"].tolist() != [168, 371, 646, 620]:
        raise FeasibilityContractError("Tabla de folds cronologicos inesperada u ordenada incorrectamente.")
    if int(validation["evaluation_matches"].sum()) != 1_805:
        raise FeasibilityContractError("Folds cronologicos no reconcilian validation.")


def _validate_chronology_semantics(contract: Mapping[str, Any]) -> None:
    _assert_no_unsafe_chronology_values(contract)
    source = contract.get("source_contract", {})
    reconciliations = contract.get("reconciliations", {})
    population = {row.get("split"): row for row in contract.get("split_population", []) if isinstance(row, Mapping)}
    expected_splits = {"train": 4_188, "validation": 1_805, "test": 1_531}
    if source.get("matches_after_immediate_reduction") != 7_524 or reconciliations.get("unique_matches") != 7_524 or reconciliations.get("split_match_sum") != 7_524:
        raise FeasibilityContractError("Cardinalidad total cronologica inesperada.")
    if {name: row.get("matches") for name, row in population.items()} != expected_splits:
        raise FeasibilityContractError("Cardinalidades train/validation/test inesperadas.")
    folds = contract.get("rolling_folds", [])
    expected_folds = [("validation_2020", 168), ("validation_2021", 371), ("validation_2022", 646), ("validation_2023", 620)]
    observed_folds = [(row.get("fold_id"), row.get("evaluation_matches")) for row in folds if isinstance(row, Mapping) and str(row.get("fold_id", "")).startswith("validation_")]
    if observed_folds != expected_folds or sum(value for _, value in observed_folds) != 1_805:
        raise FeasibilityContractError("Folds cronologicos inesperados u ordenados incorrectamente.")
    protected = contract.get("protected_test_contract", {})
    if protected.get("test_status") != "sealed" or protected.get("test_evaluation_runs") != 0 or protected.get("test_used_for_method_selection") is not False:
        raise FeasibilityContractError("Sellado upstream del test invalido.")
    if contract.get("date_contract", {}).get("last_date") != "2026-05-21":
        raise FeasibilityContractError("Fecha maxima cronologica inesperada.")
    partial = contract.get("yearly_contract", {}).get("partial_year_definition")
    if partial != "source fixed at 2026-05-21 before completion of calendar year":
        raise FeasibilityContractError("Declaracion de 2026 parcial inesperada.")
    if contract.get("publication_fingerprint") != EXPECTED_CHRONOLOGY_PUBLICATION_FINGERPRINT:
        raise FeasibilityContractError("Fingerprint upstream cronologico inesperado.")


def validate_chronological_contract(
    path: Path = CHRONOLOGY_PATH,
    by_year_path: Path = CHRONOLOGY_BY_YEAR_PATH,
    folds_path: Path = CHRONOLOGY_FOLDS_PATH,
) -> dict[str, Any]:
    """Congela procedencia Git, semantica y JSON portable antes del Parquet."""
    _validate_chronology_commit()
    payload = path.read_bytes()
    if canonical_json_sha256(payload) != EXPECTED_CHRONOLOGY_CANONICAL_SHA256:
        raise FeasibilityContractError("Hash JSON canonico del protocolo cronologico inesperado.")
    contract = json.loads(canonical_json_bytes(payload).decode("utf-8"), parse_constant=_reject_json_constant)
    _validate_chronology_semantics(contract)
    _validate_chronology_tables(by_year_path, folds_path)
    return contract


def _arrow_cutoff_scalar(field_type: pa.DataType) -> pa.Scalar:
    if pa.types.is_timestamp(field_type):
        return pa.scalar(CUTOFF.to_pydatetime(), type=field_type)
    if pa.types.is_date32(field_type) or pa.types.is_date64(field_type):
        return pa.scalar(CUTOFF.date(), type=field_type)
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return pa.scalar("2023-12-31", type=field_type)
    raise FeasibilityContractError(f"Tipo Arrow no compatible para date: {field_type}")


def read_development_points(path: Path = POINTS_PATH) -> pd.DataFrame:
    """Una unica lectura fisicamente filtrada; nunca materializa filas del test."""
    dataset = ds.dataset(path, format="parquet")
    missing = sorted(set(SOURCE_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise FeasibilityContractError(f"Faltan columnas fuente: {missing}")
    date_type = dataset.schema.field("date").type
    table = dataset.to_table(columns=SOURCE_COLUMNS, filter=ds.field("date") <= _arrow_cutoff_scalar(date_type))
    frame = table.to_pandas()
    return validate_development_points(frame, expected_matches=EXPECTED_DEVELOPMENT_MATCHES)


def validate_development_points(points: pd.DataFrame, *, expected_matches: int | None = None) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(points.columns))
    if missing:
        raise FeasibilityContractError(f"Faltan columnas obligatorias: {missing}")
    work = points.loc[:, SOURCE_COLUMNS].copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"):
        _exact_text(work[field], field)
    if work[["match_id", "point_number"]].isna().any().any() or work.duplicated(["match_id", "point_number"]).any():
        raise FeasibilityContractError("La clave (match_id, point_number) debe ser unica y no nula.")
    raw_dates = work["date"]
    if raw_dates.map(lambda value: isinstance(value, str)).all():
        if not raw_dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise FeasibilityContractError("date debe usar exactamente YYYY-MM-DD.")
        dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    else:
        if not raw_dates.map(lambda value: isinstance(value, (pd.Timestamp, datetime, date, np.datetime64))).all():
            raise FeasibilityContractError("date contiene tipos mixtos o ambiguos.")
        dates = pd.to_datetime(raw_dates, errors="coerce")
    if dates.isna().any() or dates.dt.tz is not None:
        raise FeasibilityContractError("date contiene fechas invalidas o con zona horaria.")
    work["date"] = dates.dt.normalize()
    if work["date"].gt(CUTOFF).any():
        raise FeasibilityContractError("Se recibieron filas posteriores a 2023-12-31.")
    if not work["surface"].isin(ALLOWED_SURFACES).all():
        raise FeasibilityContractError(f"Superficies invalidas: {sorted(work.loc[~work.surface.isin(ALLOWED_SURFACES), 'surface'].unique())}")
    if work["player_1"].eq(work["player_2"]).any():
        raise FeasibilityContractError("player_1 y player_2 deben ser distintos.")
    for field in ("server", "point_winner"):
        work[field] = work[field].map(lambda value: _strict_index(value, field))
    metadata = ["date", "surface", "player_1", "player_2"]
    if work.groupby("match_id", sort=False)[metadata].nunique(dropna=False).gt(1).any().any():
        raise FeasibilityContractError("Metadatos inconsistentes dentro de match_id.")
    if expected_matches is not None and work["match_id"].nunique() != expected_matches:
        raise FeasibilityContractError(f"Partidos de desarrollo inesperados: {work.match_id.nunique()}.")
    work["derived_period"] = work["date"].dt.year.map(derive_period)
    if not work["derived_period"].isin(PERIOD_ORDER).all():
        raise FeasibilityContractError("Periodo derivado inesperado.")
    work["server_player"] = np.where(work.server.eq(1), work.player_1, work.player_2)
    work["server_won_point"] = work.point_winner.eq(work.server)
    work["presence_state"] = work.first_serve.map(_presence)
    return work.sort_values(["date", "match_id", "point_number"], kind="stable").reset_index(drop=True)


def _facts(parsed: Any) -> dict[str, Any]:
    structure = parsed.structure
    prefix = structure.prefix if isinstance(structure, (ServeAce, ServeUnreturned, ServeFault)) else structure if isinstance(structure, ServicePrefix) else None
    direction_code = prefix.direction if prefix is not None else None
    if isinstance(structure, ServeAce):
        outcome = "ace"
    elif isinstance(structure, ServeUnreturned):
        outcome = "unreturned"
    elif isinstance(structure, ServeFault):
        outcome = "fault"
    else:
        outcome = "continuation_or_other"
    special = structure.code if isinstance(structure, SpecialCodeStructure) else None
    return {
        "direction_code": direction_code,
        "immediate_outcome": outcome,
        "parser_warning": bool(parsed.warnings),
        "has_residual": bool(parsed.residual_spans or parsed.residual_text),
        "fully_consumed": bool(parsed.structural_status.value == "consistent" and not parsed.residual_spans and not parsed.residual_text),
        "special_code": special,
        "structure_type": "none" if structure is None else structure.structure_type,
    }


def prepare_population(points: pd.DataFrame, parser: Callable = parse_sequence) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = validate_development_points(points)
    substantive = frame.loc[frame.presence_state.eq("substantive")].copy()
    cache: dict[tuple[str, int], Any] = {}
    def cached(value: str) -> Any:
        key = (value, 1)
        if key not in cache:
            cache[key] = parser(value, serve_number=1)
        return cache[key]
    facts = pd.DataFrame([_facts(cached(value)) for value in substantive.first_serve], index=substantive.index)
    substantive = pd.concat([substantive, facts], axis=1)
    directional = substantive.loc[substantive.direction_code.isin(DIRECTION_CODES)].copy()
    directional["direction"] = directional.direction_code.map(DIRECTION_CODES)
    counts = {
        "actionable_direction": int(len(directional)),
        "direction_unknown_0": int(substantive.direction_code.eq("0").sum()),
        "special_unit_sequence": int(substantive.special_code.isin(SPECIAL_CODES).sum()),
        "unrecognized_or_no_initial_direction": int((substantive.direction_code.isna() & ~substantive.special_code.isin(SPECIAL_CODES)).sum()),
        "null_sequence": int(frame.presence_state.eq("null").sum()),
        "empty_sequence": int(frame.presence_state.eq("empty").sum()),
        "whitespace_only_sequence": int(frame.presence_state.eq("whitespace_only").sum()),
    }
    if sum(counts.values()) != len(frame):
        raise FeasibilityContractError("Las exclusiones no particionan la fuente de desarrollo.")
    special_rows = []
    for code in SPECIAL_CODES:
        part = substantive.loc[substantive.special_code.eq(code)]
        special_rows.append({"code": code, "points": int(len(part)), "matches": int(part.match_id.nunique()), "servers": int(part.server_player.nunique()), "treatment": "excluded_special_no_actionable_direction", "reason_code": "special_unit_sequence"})
    return frame, directional, {"cache_entries": len(cache), "exclusion_counts": counts, "special_sequences": special_rows}


def _ordered(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    work = frame.copy()
    orders = {"dimension": ("pooled", "surface", "derived_period", "surface_derived_period"), "direction": DIRECTION_ORDER, "surface": ("ALL",) + ALLOWED_SURFACES, "derived_period": ("ALL",) + PERIOD_ORDER, "immediate_outcome": IMMEDIATE_OUTCOMES, "scope": ("global", "surface"), "cut_kind": ("point_only", "match_only", "joint")}
    temporary = []
    for column in columns:
        if column in orders:
            unexpected = set(work[column].dropna()) - set(orders[column])
            if unexpected:
                raise FeasibilityContractError(f"Dominio invalido en {column}: {sorted(unexpected)!r}")
            name = f"__{column}"
            work[name] = pd.Categorical(work[column], categories=orders[column], ordered=True)
            temporary.append(name)
    result = work.sort_values([f"__{column}" if column in orders else column for column in columns], kind="stable").drop(columns=temporary).reset_index(drop=True)
    return result


def _aggregate_directional(directional: pd.DataFrame, dimension: str, group_columns: list[str], surface: str, period: str) -> list[dict[str, Any]]:
    groups = [((), directional)] if not group_columns else directional.groupby(group_columns, sort=False, observed=True)
    rows = []
    for key, part in groups:
        key = key if isinstance(key, tuple) else (key,)
        attrs = dict(zip(group_columns, key))
        for direction in DIRECTION_ORDER:
            selected = part.loc[part.direction.eq(direction)]
            denominator = len(directional) if dimension == "pooled" else len(part)
            outcome_counts = selected.immediate_outcome.value_counts()
            rows.append({
                "dimension": dimension, "direction": direction,
                "surface": attrs.get("surface", surface), "derived_period": attrs.get("derived_period", period),
                "analytical_points": int(len(selected)), "matches": int(selected.match_id.nunique()), "servers": int(selected.server_player.nunique()),
                "server_wins": int(selected.server_won_point.sum()), "server_win_rate": None if selected.empty else float(selected.server_won_point.mean()),
                "proportion_of_directional_population": None if not denominator else float(len(selected) / denominator),
                **{f"{outcome}_points": int(outcome_counts.get(outcome, 0)) for outcome in IMMEDIATE_OUTCOMES},
                "parser_warning_points": int(selected.parser_warning.sum()), "residual_points": int(selected.has_residual.sum()), "fully_consumed_points": int(selected.fully_consumed.sum()),
            })
    return rows


def build_by_direction(directional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows += _aggregate_directional(directional, "pooled", [], "ALL", "ALL")
    rows += _aggregate_directional(directional, "surface", ["surface"], "ALL", "ALL")
    rows += _aggregate_directional(directional, "derived_period", ["derived_period"], "ALL", "ALL")
    rows += _aggregate_directional(directional, "surface_derived_period", ["surface", "derived_period"], "ALL", "ALL")
    return _ordered(pd.DataFrame(rows, columns=BY_DIRECTION_COLUMNS), ["dimension", "direction", "surface", "derived_period"])


def build_outcomes(directional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("pooled", [], "ALL", "ALL"), ("surface", ["surface"], "ALL", "ALL"), ("derived_period", ["derived_period"], "ALL", "ALL"), ("surface_derived_period", ["surface", "derived_period"], "ALL", "ALL")]
    for dimension, columns, fixed_surface, fixed_period in specs:
        groups = [((), directional)] if not columns else directional.groupby(columns, sort=False, observed=True)
        for key, part in groups:
            key = key if isinstance(key, tuple) else (key,)
            attrs = dict(zip(columns, key))
            for direction in DIRECTION_ORDER:
                by_direction = part.loc[part.direction.eq(direction)]
                for outcome in IMMEDIATE_OUTCOMES:
                    selected = by_direction.loc[by_direction.immediate_outcome.eq(outcome)]
                    rows.append({"dimension": dimension, "direction": direction, "surface": attrs.get("surface", fixed_surface), "derived_period": attrs.get("derived_period", fixed_period), "immediate_outcome": outcome, "points": int(len(selected)), "matches": int(selected.match_id.nunique()), "servers": int(selected.server_player.nunique()), "server_wins": int(selected.server_won_point.sum()), "server_win_rate": None if selected.empty else float(selected.server_won_point.mean())})
    return _ordered(pd.DataFrame(rows, columns=OUTCOME_COLUMNS), ["dimension", "direction", "surface", "derived_period", "immediate_outcome"])


def _percentiles(values: pd.Series) -> dict[str, float]:
    result = np.percentile(values.astype(float), [0, 25, 50, 75, 90, 100], method="linear")
    return dict(zip(("min", "p25", "median", "p75", "p90", "max"), map(float, result)))


def build_coverage(frame: pd.DataFrame, directional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("global", "ALL", frame), *[("surface", surface, frame.loc[frame.surface.eq(surface)]) for surface in ALLOWED_SURFACES]]
    cuts = [("point_only", point, 0) for point in POINT_CUTS] + [("match_only", 0, match) for match in MATCH_CUTS] + [("joint", point, match) for point in POINT_CUTS for match in MATCH_CUTS]
    for scope, surface, base in specs:
        possible = sorted(base.server_player.unique())
        for direction in DIRECTION_ORDER:
            events = directional if scope == "global" else directional.loc[directional.surface.eq(surface)]
            events = events.loc[events.direction.eq(direction)]
            observed = events.groupby("server_player", sort=False).agg(points=("point_number", "size"), matches=("match_id", "nunique"))
            support = observed.reindex(possible, fill_value=0)
            stats_points, stats_matches = _percentiles(support.points), _percentiles(support.matches)
            for kind, point_cut, match_cut in cuts:
                eligible = support.loc[support.points.ge(point_cut) & support.matches.ge(match_cut)]
                eligible_players = set(eligible.index)
                eligible_matches = int(events.loc[events.server_player.isin(eligible_players), "match_id"].nunique())
                rows.append({
                    "scope": scope, "surface": surface, "direction": direction,
                    "cut_kind": kind, "point_cut": point_cut, "match_cut": match_cut,
                    "directional_points": int(len(events)),
                    "directional_matches": int(events.match_id.nunique()),
                    "directional_server_wins": int(events.server_won_point.sum()),
                    "first_date": None if events.empty else events.date.min().strftime("%Y-%m-%d"),
                    "last_date": None if events.empty else events.date.max().strftime("%Y-%m-%d"),
                    "possible_servers": len(possible), "observed_servers": int(len(observed)),
                    "zero_evidence_servers": int(len(possible) - len(observed)),
                    "eligible_servers": int(len(eligible)),
                    "eligible_server_rate": None if not possible else float(len(eligible) / len(possible)),
                    "eligible_matches": eligible_matches,
                    **{f"{name}_points": value for name, value in stats_points.items()},
                    **{f"{name}_matches": value for name, value in stats_matches.items()},
                })
    return _ordered(pd.DataFrame(rows, columns=COVERAGE_COLUMNS), ["scope", "surface", "direction", "cut_kind", "point_cut", "match_cut"])


def validate_reconciliations(frame: pd.DataFrame, directional: pd.DataFrame, by_direction: pd.DataFrame, outcomes: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, bool]:
    expected_by_direction = build_by_direction(directional)
    expected_outcomes = build_outcomes(directional)
    expected_coverage = build_coverage(frame, directional)
    try:
        pdt.assert_frame_equal(by_direction.reset_index(drop=True), expected_by_direction, check_dtype=False, check_like=False)
        pdt.assert_frame_equal(outcomes.reset_index(drop=True), expected_outcomes, check_dtype=False, check_like=False)
        pdt.assert_frame_equal(coverage.reset_index(drop=True), expected_coverage, check_dtype=False, check_like=False)
    except AssertionError as error:
        raise FeasibilityContractError("Agregado publicado no coincide con la derivacion canonica.") from error
    if len(directional) != int(by_direction.loc[by_direction.dimension.eq("pooled"), "analytical_points"].sum()):
        raise FeasibilityContractError("Poblacion direccional no reconcilia.")
    for dimension in by_direction.dimension.unique():
        grouped = by_direction.loc[by_direction.dimension.eq(dimension)]
        for _, row in grouped.iterrows():
            if sum(int(row[f"{name}_points"]) for name in IMMEDIATE_OUTCOMES) != int(row.analytical_points):
                raise FeasibilityContractError("Outcomes inmediatos no exhaustivos.")
    keys = ["dimension", "direction", "surface", "derived_period"]
    outcome_totals = outcomes.groupby(keys, sort=False).points.sum().reset_index()
    expected = by_direction[keys + ["analytical_points"]].rename(columns={"analytical_points": "points"})
    pdt.assert_frame_equal(_ordered(outcome_totals, keys), _ordered(expected, keys), check_dtype=False)
    if coverage.duplicated(["scope", "surface", "direction", "cut_kind", "point_cut", "match_cut"]).any():
        raise FeasibilityContractError("Cobertura con claves duplicadas.")
    return {"outcomes_exhaustive": True, "outcomes_reconciled_by_direction_surface_period": True, "coverage_reconciled_from_directional_population": True, "coverage_key_unique": True, "directional_population_reconciled": True}


def _seal() -> dict[str, Any]:
    return {"test_status": "sealed", **{field: 0 for field in TEST_ZERO_FIELDS}, "used_for_method_selection": False, "test_target_matches_excluded_before_construction": EXPECTED_TEST_EXCLUDED}


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8", na_rep="<NULL>", lineterminator="\n", date_format="%Y-%m-%d").encode("utf-8")


def _fingerprint(summary: Mapping[str, Any], payloads: Sequence[bytes]) -> str:
    work = copy.deepcopy(dict(summary)); work.pop("publication_fingerprint", None)
    digest = hashlib.sha256(json.dumps(work, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    for payload in payloads:
        digest.update(payload)
    return digest.hexdigest().upper()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convierte los nulos tabulares en null JSON, nunca en NaN."""
    records: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        records.append({
            key: None if value is pd.NA or (isinstance(value, float) and math.isnan(value)) else value
            for key, value in record.items()
        })
    return records


def build_summary(frame: pd.DataFrame, directional: pd.DataFrame, prepared: Mapping[str, Any], by_direction: pd.DataFrame, outcomes: pd.DataFrame, coverage: pd.DataFrame, chronology: Mapping[str, Any]) -> dict[str, Any]:
    exclusions = prepared["exclusion_counts"]
    parser_diagnostics = {"warning_points": int(directional.parser_warning.sum()), "residual_points": int(directional.has_residual.sum()), "fully_consumed_points": int(directional.fully_consumed.sum()), "cache_entries": int(prepared["cache_entries"]), "special_sequences": prepared["special_sequences"]}
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION, "pattern_id": PATTERN_ID, "analysis_status": "available_descriptive",
        "source": {"path": "data/processed/points_enriched.parquet", "columns_used": SOURCE_COLUMNS},
        "source_filter": {"predicate": "date <= 2023-12-31", "applied_during_arrow_scan": True, "post_read_later_rows": 0, "development_matches": int(frame.match_id.nunique())},
        "upstream_chronological_contract": {
            "path": "reports/chronological_validation_summary.json",
            "commit": EXPECTED_CHRONOLOGY_COMMIT,
            "blob": EXPECTED_CHRONOLOGY_BLOB,
            "canonical_json_sha256": EXPECTED_CHRONOLOGY_CANONICAL_SHA256,
            "binary_sha256_lf_provenance": EXPECTED_CHRONOLOGY_BINARY_SHA256_LF,
            "publication_fingerprint": EXPECTED_CHRONOLOGY_PUBLICATION_FINGERPRINT,
            "total_matches": 7_524, "development_matches": 5_993,
            "excluded_test_matches": 1_531, "train_matches": 4_188,
            "validation_matches": 1_805,
            "fold_matches": {"2020": 168, "2021": 371, "2022": 646, "2023": 620},
        },
        "population": {"development_point_rows": int(len(frame)), "development_matches": int(frame.match_id.nunique()), "development_servers": int(frame.server_player.nunique()), "directional_points": int(len(directional)), "directional_matches": int(directional.match_id.nunique()), "directional_servers": int(directional.server_player.nunique()), "server_wins": int(directional.server_won_point.sum())},
        "exclusions": {"counts": exclusions, "partition_denominator": int(len(frame)), "mutually_exclusive": True, "exclusions_do_not_filter_first_serve_faults_or_parser_residuals": True},
        "directions": _json_records(by_direction.loc[by_direction.dimension.eq("pooled")]),
        "immediate_outcome_contract": {"categories": list(IMMEDIATE_OUTCOMES), "definition": "Clasificacion del parser actual sobre el prefijo inicial; no filtra la poblacion direccional.", "outcome_principal": "server_won_point es el resultado final del punto, incluso tras falta del primer saque.", "causal_interpretation": False},
        "immediate_outcomes": _json_records(outcomes.loc[outcomes.dimension.eq("pooled")]),
        "surfaces": _json_records(by_direction.loc[by_direction.dimension.eq("surface")]), "periods": _json_records(by_direction.loc[by_direction.dimension.eq("derived_period")]),
        "parser_contract": {"parser_version": PARSER_VERSION, "grammar_version": GRAMMAR_VERSION, "serve_number": 1, "cache_key": "(sequence_text, serve_number)", "actionable_direction_codes": DIRECTION_CODES, "direction_0_actionable": False},
        "parser_diagnostics": parser_diagnostics,
        "structural_coverage_contract": {"status": "descriptive_upper_bound_only", "history_constructed": False, "joint_role_coverage_status": "not_evaluated", "chronological_evidence_status": "not_evaluated", "description": "Cobertura retrospectiva del desarrollo hasta 2023; no representa evidencia disponible antes de un partido."},
        "candidate_cuts": {"point_cuts": list(POINT_CUTS), "match_cuts": list(MATCH_CUTS), "selection_status": "not_selected", "threshold_selection_status": "not_selected", "comparability_note": "Cortes descriptivos, no politica de evidencia."},
        "reconciliations": validate_reconciliations(frame, directional, by_direction, outcomes, coverage),
        "feasibility_description": {"direction_population_available": bool(len(directional)), "all_three_directions_present": set(directional.direction) == set(DIRECTION_ORDER), "all_surfaces_present": set(directional.surface) == set(ALLOWED_SURFACES), "all_periods_present": set(directional.derived_period) == set(PERIOD_ORDER), "player_support_described": True, "parser_extension_required": False, "unresolved_ambiguities": ["No se interpreta el rally posterior ni el lado del saque.", "La cobertura publicada no es elegibilidad historica conjunta."], "next_required_analysis": "Construir perfiles historicos leakage-safe servidor/restador y evaluar evidencia conjunta sin reutilizar thresholds automaticamente."},
        "next_required_analysis": "Viabilidad leakage-safe de perfiles historicos por direccion de primer saque y cobertura conjunta server/opponent.",
        "chronological_seal": _seal(),
        "methodological_limits": ["Analisis descriptivo y no causal.", "Incluye faltas de primer saque: la direccion es el intento y server_won_point el resultado final del punto.", "No hay perfiles historicos, thresholds seleccionados, scoring, ranking ni recomendacion.", "No se procesan filas posteriores a 2023-12-31."],
        "artifact_contract": {"csv_null_representation": "<NULL>", "utf8": True, "index": False, "timestamps": False, "tables": {"by_direction": BY_DIRECTION_COLUMNS, "outcomes": OUTCOME_COLUMNS, "coverage": COVERAGE_COLUMNS}},
        "fingerprint_contract_version": "1",
    }
    return summary


def analyze_development_points(points: pd.DataFrame, chronology: Mapping[str, Any] | None = None, parser: Callable = parse_sequence) -> FeasibilityResult:
    frame, directional, prepared = prepare_population(points, parser=parser)
    by_direction = build_by_direction(directional); outcomes = build_outcomes(directional); coverage = build_coverage(frame, directional)
    summary = build_summary(frame, directional, prepared, by_direction, outcomes, coverage, chronology or {})
    payloads = tuple(_frame_bytes(frame_) for frame_ in (by_direction, outcomes, coverage))
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(value).hexdigest().upper() for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)}
    summary["artifact_payload_bytes"] = {name: len(value) for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)}
    fingerprint = _fingerprint(summary, payloads); summary["publication_fingerprint"] = fingerprint
    result = FeasibilityResult(summary, by_direction, outcomes, coverage, frame, directional, fingerprint)
    validate_result(result)
    return result


def not_available_result(error: BaseException) -> FeasibilityResult:
    reason = "chronological_contract_or_source_validation_failed"
    summary = {"analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION, "pattern_id": PATTERN_ID, "analysis_status": "not_available", "reason_codes": [reason], "failure": {"stage": "source_validation", "exception_type": type(error).__name__, "message": str(error).replace(str(ROOT), "<ROOT>")[:500]}, "chronological_seal": _seal(), "artifact_contract": {"csv_null_representation": "<NULL>", "utf8": True, "index": False, "tables": {"by_direction": BY_DIRECTION_COLUMNS, "outcomes": OUTCOME_COLUMNS, "coverage": COVERAGE_COLUMNS}}, "fingerprint_contract_version": "1"}
    frames = tuple(pd.DataFrame(columns=columns) for columns in (BY_DIRECTION_COLUMNS, OUTCOME_COLUMNS, COVERAGE_COLUMNS))
    payloads = tuple(_frame_bytes(frame) for frame in frames)
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(value).hexdigest().upper() for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)}
    summary["artifact_payload_bytes"] = {name: len(value) for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)}
    fingerprint = _fingerprint(summary, payloads); summary["publication_fingerprint"] = fingerprint
    return FeasibilityResult(summary, *frames, pd.DataFrame(columns=SOURCE_COLUMNS), pd.DataFrame(), fingerprint)


def validate_result(result: FeasibilityResult) -> None:
    if result.summary.get("analysis_status") not in {"available_descriptive", "not_available"}:
        raise FeasibilityContractError("analysis_status invalido.")
    seal = result.summary.get("chronological_seal", {})
    if seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or seal.get("test_target_matches_excluded_before_construction") != EXPECTED_TEST_EXCLUDED or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS):
        raise FeasibilityContractError("Sellado del test invalido.")
    for frame, columns in zip((result.by_direction, result.outcomes, result.coverage), (BY_DIRECTION_COLUMNS, OUTCOME_COLUMNS, COVERAGE_COLUMNS)):
        if frame.columns.tolist() != columns or any(str(value).startswith("Unnamed") for value in frame.columns):
            raise FeasibilityContractError("Schema de artefacto invalido.")
    if result.summary["analysis_status"] == "not_available":
        if not result.by_direction.empty or not result.outcomes.empty or not result.coverage.empty:
            raise FeasibilityContractError("not_available no puede publicar datos parciales.")
    else:
        if result.development.date.gt(CUTOFF).any() or result.development.match_id.nunique() != EXPECTED_DEVELOPMENT_MATCHES and len(result.development) > 100:
            raise FeasibilityContractError("Poblacion de desarrollo invalida.")
        validate_reconciliations(result.development, result.directional, result.by_direction, result.outcomes, result.coverage)
    payloads = tuple(_frame_bytes(frame) for frame in (result.by_direction, result.outcomes, result.coverage))
    if result.summary.get("artifact_payload_sha256") != {name: hashlib.sha256(value).hexdigest().upper() for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)}:
        raise FeasibilityContractError("Hashes de artefactos no reconcilian.")
    expected = _fingerprint(result.summary, payloads)
    if result.publication_fingerprint != expected or result.summary.get("publication_fingerprint") != expected:
        raise FeasibilityContractError("Fingerprint invalido.")


def serialize_artifacts(result: FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes]:
    validate_result(result)
    payloads = tuple(_frame_bytes(frame) for frame in (result.by_direction, result.outcomes, result.coverage))
    summary = (json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return (summary, *payloads)


def _stage(path: Path, payload: bytes) -> Path:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return Path(temporary)


def write_artifacts(result: FeasibilityResult, *, summary_path: Path = SUMMARY_PATH, by_direction_path: Path = BY_DIRECTION_PATH, outcomes_path: Path = OUTCOMES_PATH, coverage_path: Path = COVERAGE_PATH, payloads: tuple[bytes, bytes, bytes, bytes] | None = None) -> None:
    """Publica cuatro ficheros ya validados mediante sustituciones por fichero.

    No existe una transacción de directorio que pueda sobrevivir un apagado
    abrupto entre dos ``os.replace``; las excepciones normales sí restauran el
    estado anterior y eliminan todos los temporales creados por esta función.
    """
    targets = (summary_path, by_direction_path, outcomes_path, coverage_path)
    if payloads is None:
        payloads = serialize_artifacts(result)
    if len(payloads) != len(targets) or any(not isinstance(payload, bytes) for payload in payloads):
        raise FeasibilityContractError("Payloads de publicacion invalidos.")
    for target in targets: target.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []; backups: dict[Path, bytes | None] = {}
    try:
        # Se conserva cada temporal inmediatamente para poder limpiarlo si el
        # staging falla antes de completar los cuatro payloads.
        for path, payload in zip(targets, payloads):
            staged.append(_stage(path, payload))
        backups = {path: path.read_bytes() if path.exists() else None for path in targets}
        for temporary, target in zip(staged, targets): os.replace(temporary, target)
    except BaseException:
        for path, prior in backups.items():
            if prior is not None: path.write_bytes(prior)
            elif path.exists(): path.unlink()
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def _assert_persisted_semantics(summary: Mapping[str, Any], by_direction: pd.DataFrame, outcomes: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """Reconciliaciones deterministas posibles solo con los artefactos agregados."""
    if summary.get("analysis_name") != ANALYSIS_NAME or summary.get("analysis_version") != ANALYSIS_VERSION or summary.get("pattern_id") != PATTERN_ID:
        raise FeasibilityContractError("Identidad persistida del analisis invalida.")
    if summary.get("analysis_status") != "available_descriptive":
        raise FeasibilityContractError("Estado persistido disponible invalido.")
    if summary.get("population") != PUBLISHED_POPULATION:
        raise FeasibilityContractError("Poblacion persistida inesperada.")
    seal = summary.get("chronological_seal", {})
    if seal.get("test_status") != "sealed" or seal.get("used_for_method_selection") is not False or seal.get("test_target_matches_excluded_before_construction") != EXPECTED_TEST_EXCLUDED or any(seal.get(field) != 0 for field in TEST_ZERO_FIELDS):
        raise FeasibilityContractError("Sellado persistido del test invalido.")
    if summary.get("methodological_limits") != [
        "Analisis descriptivo y no causal.",
        "Incluye faltas de primer saque: la direccion es el intento y server_won_point el resultado final del punto.",
        "No hay perfiles historicos, thresholds seleccionados, scoring, ranking ni recomendacion.",
        "No se procesan filas posteriores a 2023-12-31.",
    ]:
        raise FeasibilityContractError("Limitaciones metodologicas persistidas invalidas.")
    if summary.get("source", {}).get("path") != "data/processed/points_enriched.parquet" or summary.get("source_filter") != {"applied_during_arrow_scan": True, "development_matches": EXPECTED_DEVELOPMENT_MATCHES, "post_read_later_rows": 0, "predicate": "date <= 2023-12-31"}:
        raise FeasibilityContractError("Contrato de fuente persistido invalido.")
    if summary.get("parser_contract", {}).get("serve_number") != 1 or summary["parser_contract"].get("cache_key") != "(sequence_text, serve_number)" or summary["parser_contract"].get("actionable_direction_codes") != DIRECTION_CODES or summary["parser_contract"].get("direction_0_actionable") is not False:
        raise FeasibilityContractError("Contrato de parser persistido invalido.")
    expected_upstream = {
        "path": "reports/chronological_validation_summary.json", "commit": EXPECTED_CHRONOLOGY_COMMIT,
        "blob": EXPECTED_CHRONOLOGY_BLOB, "canonical_json_sha256": EXPECTED_CHRONOLOGY_CANONICAL_SHA256,
        "binary_sha256_lf_provenance": EXPECTED_CHRONOLOGY_BINARY_SHA256_LF,
        "publication_fingerprint": EXPECTED_CHRONOLOGY_PUBLICATION_FINGERPRINT,
        "total_matches": 7_524, "development_matches": EXPECTED_DEVELOPMENT_MATCHES,
        "excluded_test_matches": EXPECTED_TEST_EXCLUDED, "train_matches": 4_188,
        "validation_matches": 1_805, "fold_matches": {"2020": 168, "2021": 371, "2022": 646, "2023": 620},
    }
    if summary.get("upstream_chronological_contract") != expected_upstream:
        raise FeasibilityContractError("Contrato cronologico persistido invalido.")
    dimensions = {
        *(('pooled', direction, 'ALL', 'ALL') for direction in DIRECTION_ORDER),
        *(('surface', direction, surface, 'ALL') for surface in ALLOWED_SURFACES for direction in DIRECTION_ORDER),
        *(('derived_period', direction, 'ALL', period) for period in PERIOD_ORDER for direction in DIRECTION_ORDER),
        *(('surface_derived_period', direction, surface, period) for surface in ALLOWED_SURFACES for period in PERIOD_ORDER for direction in DIRECTION_ORDER),
    }
    if set(map(tuple, by_direction[["dimension", "direction", "surface", "derived_period"]].itertuples(index=False, name=None))) != dimensions:
        raise FeasibilityContractError("Dominio de grupos persistidos invalido.")
    if set(outcomes.immediate_outcome) != set(IMMEDIATE_OUTCOMES) or set(outcomes.direction) != set(DIRECTION_ORDER):
        raise FeasibilityContractError("Dominio de outcomes persistidos invalido.")
    def assert_summary_records(name: str, frame: pd.DataFrame, dimension: str) -> None:
        published = frame.loc[frame.dimension.eq(dimension)].reset_index(drop=True)
        summary_frame = pd.DataFrame(summary.get(name), columns=published.columns).reset_index(drop=True)
        try:
            # El CSV puede redondear la ultima cifra binaria de un decimal; la
            # tolerancia es menor que una unidad del ultimo decimal publicado.
            pdt.assert_frame_equal(summary_frame, published, check_dtype=False, check_exact=False, rtol=0, atol=1e-15)
        except AssertionError as error:
            raise FeasibilityContractError("Resumen persistido no coincide con sus tablas.") from error
    assert_summary_records("directions", by_direction, "pooled")
    assert_summary_records("surfaces", by_direction, "surface")
    assert_summary_records("periods", by_direction, "derived_period")
    assert_summary_records("immediate_outcomes", outcomes, "pooled")
    numeric = ["analytical_points", "matches", "server_wins", "ace_points", "unreturned_points", "fault_points", "continuation_or_other_points", "parser_warning_points", "residual_points", "fully_consumed_points"]
    leaf = by_direction.loc[by_direction.dimension.eq("surface_derived_period")]
    for dimension, group_columns in (("surface", ["surface", "direction"]), ("derived_period", ["derived_period", "direction"]), ("pooled", ["direction"])):
        derived = leaf.groupby(group_columns, sort=False)[numeric].sum().reset_index()
        published = by_direction.loc[by_direction.dimension.eq(dimension), group_columns + numeric]
        derived = derived.sort_values(group_columns).reset_index(drop=True)
        published = published.sort_values(group_columns).reset_index(drop=True)
        try:
            pdt.assert_frame_equal(derived, published, check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise FeasibilityContractError("Agregacion persistida por dimension invalida.") from error
    outcome_sums = outcomes.groupby(["dimension", "direction", "surface", "derived_period"], sort=False)[["points", "server_wins"]].sum().reset_index()
    expected_outcomes = by_direction[["dimension", "direction", "surface", "derived_period", "analytical_points", "server_wins"]].rename(columns={"analytical_points": "points"})
    outcome_sums = outcome_sums.sort_values(list(expected_outcomes.columns[:4])).reset_index(drop=True)
    expected_outcomes = expected_outcomes.sort_values(list(expected_outcomes.columns[:4])).reset_index(drop=True)
    try:
        pdt.assert_frame_equal(outcome_sums, expected_outcomes, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise FeasibilityContractError("Outcomes persistidos no reconcilian grupos.") from error
    if not np.allclose(by_direction.server_win_rate, by_direction.server_wins / by_direction.analytical_points, rtol=0, atol=1e-15):
        raise FeasibilityContractError("Proporciones persistidas by_direction invalidas.")
    denominator_columns = {"pooled": [], "surface": ["surface"], "derived_period": ["derived_period"], "surface_derived_period": ["surface", "derived_period"]}
    for dimension, group_columns in denominator_columns.items():
        subset = by_direction.loc[by_direction.dimension.eq(dimension)]
        denominator = pd.Series(len(subset) and subset.analytical_points.sum(), index=subset.index) if not group_columns else subset.groupby(group_columns, sort=False).analytical_points.transform("sum")
        if not np.allclose(subset.proportion_of_directional_population, subset.analytical_points / denominator, rtol=0, atol=1e-15):
            raise FeasibilityContractError("Proporciones persistidas by_direction invalidas.")
    if not np.allclose(outcomes.server_win_rate, outcomes.server_wins / outcomes.points, rtol=0, atol=1e-15):
        raise FeasibilityContractError("Proporciones persistidas outcomes invalidas.")
    expected_coverage = {
        (scope, surface, direction, cut_kind, point_cut, match_cut)
        for scope, surface in (("global", "ALL"), *(("surface", surface) for surface in ALLOWED_SURFACES))
        for direction in DIRECTION_ORDER
        for cut_kind, point_cut, match_cut in ([('point_only', value, 0) for value in POINT_CUTS] + [('match_only', 0, value) for value in MATCH_CUTS] + [('joint', point, match) for point in POINT_CUTS for match in MATCH_CUTS])
    }
    coverage_keys = set(map(tuple, coverage[["scope", "surface", "direction", "cut_kind", "point_cut", "match_cut"]].itertuples(index=False, name=None)))
    if coverage_keys != expected_coverage:
        raise FeasibilityContractError("Dominio de coverage persistido invalido.")
    lookup = by_direction.set_index(["dimension", "direction", "surface", "derived_period"])
    for row in coverage.itertuples(index=False):
        key = ("pooled", row.direction, "ALL", "ALL") if row.scope == "global" else ("surface", row.direction, row.surface, "ALL")
        source = lookup.loc[key]
        if (int(row.directional_points), int(row.directional_matches), int(row.directional_server_wins), int(row.observed_servers)) != (int(source.analytical_points), int(source.matches), int(source.server_wins), int(source.servers)):
            raise FeasibilityContractError("Coverage persistido no reconcilia grupos.")
        expected_possible = PUBLISHED_POPULATION["development_servers"] if row.scope == "global" else int(by_direction.loc[(by_direction.dimension.eq("surface")) & (by_direction.surface.eq(row.surface)), "servers"].max())
        if int(row.possible_servers) != expected_possible:
            raise FeasibilityContractError("Universo de servidores de coverage invalido.")
        if int(row.zero_evidence_servers) != int(row.possible_servers) - int(row.observed_servers) or int(row.eligible_servers) < 0 or int(row.eligible_servers) > int(row.observed_servers) or int(row.eligible_matches) < 0 or int(row.eligible_matches) > int(row.directional_matches):
            raise FeasibilityContractError("Cobertura persistida invalida.")
        if not math.isclose(float(row.eligible_server_rate), int(row.eligible_servers) / int(row.possible_servers), rel_tol=0, abs_tol=1e-15):
            raise FeasibilityContractError("Tasa de cobertura persistida invalida.")
        if not (pd.Timestamp(row.first_date) <= pd.Timestamp(row.last_date) <= CUTOFF):
            raise FeasibilityContractError("Fechas de coverage persistidas invalidas.")


def verify_persisted_artifacts(summary_path: Path = SUMMARY_PATH, by_direction_path: Path = BY_DIRECTION_PATH, outcomes_path: Path = OUTCOMES_PATH, coverage_path: Path = COVERAGE_PATH, *, result: FeasibilityResult | None = None, expected_payloads: tuple[bytes, bytes, bytes, bytes] | None = None, frozen_publication: bool = False, semantic_contract: bool = False) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = tuple(pd.read_csv(path, keep_default_na=False, na_values=["<NULL>"]) for path in (by_direction_path, outcomes_path, coverage_path))
    payloads = tuple(path.read_bytes() for path in (by_direction_path, outcomes_path, coverage_path))
    hashes = {name: hashlib.sha256(value).hexdigest().upper() for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)}
    summary_payload = summary_path.read_bytes()
    if summary.get("artifact_payload_sha256") != hashes or summary.get("artifact_payload_bytes") != {name: len(value) for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)} or summary.get("publication_fingerprint") != _fingerprint(summary, payloads):
        raise FeasibilityContractError("Contrato persistido de fingerprint invalido.")
    for frame, columns in zip(frames, (BY_DIRECTION_COLUMNS, OUTCOME_COLUMNS, COVERAGE_COLUMNS)):
        if frame.columns.tolist() != columns: raise FeasibilityContractError("Schema persistido invalido.")
    if result is not None:
        replay = serialize_artifacts(result)
        if expected_payloads is not None and replay != expected_payloads:
            raise FeasibilityContractError("Reserializacion del mismo resultado no determinista.")
        if replay != (summary_payload, *payloads):
            raise FeasibilityContractError("Payloads persistidos no coinciden con el mismo resultado.")
    if summary.get("analysis_status") == "not_available":
        if any(not frame.empty for frame in frames): raise FeasibilityContractError("not_available persistido con filas.")
        return
    if frames[0].duplicated(["dimension", "direction", "surface", "derived_period"]).any() or frames[1].duplicated(["dimension", "direction", "surface", "derived_period", "immediate_outcome"]).any() or frames[2].duplicated(["scope", "surface", "direction", "cut_kind", "point_cut", "match_cut"]).any():
        raise FeasibilityContractError("Claves persistidas duplicadas.")
    if not frames[0].equals(_ordered(frames[0], ["dimension", "direction", "surface", "derived_period"])): raise FeasibilityContractError("Orden persistido by_direction invalido.")
    if not frames[1].equals(_ordered(frames[1], ["dimension", "direction", "surface", "derived_period", "immediate_outcome"])): raise FeasibilityContractError("Orden persistido outcomes invalido.")
    if not frames[2].equals(_ordered(frames[2], ["scope", "surface", "direction", "cut_kind", "point_cut", "match_cut"])): raise FeasibilityContractError("Orden persistido coverage invalido.")
    pooled = frames[0].loc[frames[0].dimension.eq("pooled")]
    if int(pooled.analytical_points.sum()) != summary["population"]["directional_points"] or not (pooled[[f"{value}_points" for value in IMMEDIATE_OUTCOMES]].sum(axis=1).eq(pooled.analytical_points)).all():
        raise FeasibilityContractError("Reconciliacion persistida invalida.")
    if frozen_publication or semantic_contract:
        _assert_persisted_semantics(summary, *frames)
    if frozen_publication:
        actual = (summary_payload, *payloads)
        for name, payload in zip(("summary", "by_direction", "outcomes", "coverage"), actual):
            expected_size, expected_hash = PUBLISHED_ARTIFACT_CONTRACT[name]
            if len(payload) != expected_size or hashlib.sha256(payload).hexdigest().upper() != expected_hash:
                raise FeasibilityContractError(f"Artefacto publicado congelado invalido: {name}.")
        if summary.get("publication_fingerprint") != PUBLISHED_PUBLICATION_FINGERPRINT:
            raise FeasibilityContractError("Fingerprint publicado congelado invalido.")


def publish_artifacts(result: FeasibilityResult, *, summary_path: Path = SUMMARY_PATH, by_direction_path: Path = BY_DIRECTION_PATH, outcomes_path: Path = OUTCOMES_PATH, coverage_path: Path = COVERAGE_PATH) -> None:
    """Publica y reserializa el mismo objeto, sin releer ni reconstruir datos."""
    payloads = serialize_artifacts(result)
    if serialize_artifacts(result) != payloads:
        raise FeasibilityContractError("Reserializacion del mismo resultado no determinista antes de publicar.")
    write_artifacts(result, summary_path=summary_path, by_direction_path=by_direction_path, outcomes_path=outcomes_path, coverage_path=coverage_path, payloads=payloads)
    verify_persisted_artifacts(summary_path, by_direction_path, outcomes_path, coverage_path, result=result, expected_payloads=payloads)


def run_real_analysis() -> FeasibilityResult:
    chronology = validate_chronological_contract()
    points = read_development_points()
    return analyze_development_points(points, chronology=chronology)


def main() -> None:
    try:
        result = run_real_analysis()
    except BaseException as error:
        result = not_available_result(error)
    publish_artifacts(result)
    print(json.dumps({"analysis_status": result.summary["analysis_status"], "publication_fingerprint": result.publication_fingerprint}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
