"""Analisis descriptivo del segundo servicio por direccion y contexto."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.parsing.serve_sequence import (
    ServeAce,
    ServeFault,
    ServeUnreturned,
    ServicePrefix,
    parse_sequence,
)
from src.parsing.serve_sequence_rules import GRAMMAR_VERSION, PARSER_VERSION


ROOT = Path(__file__).resolve().parents[2]
POINTS_FILE = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_FILE = REPORTS_DIR / "second_serve_direction_summary.json"

SOURCE_COLUMNS = [
    "match_id", "point_number", "date", "surface", "server",
    "point_winner", "player_1", "player_2", "second_serve",
]
REQUIRED_COLUMNS = frozenset(SOURCE_COLUMNS)
SURFACE_ORDER = ("Hard", "Clay", "Grass")
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
DIRECTION_ORDER = ("4", "5", "6", "0")
DIRECTION_LABELS = {"4": "wide", "5": "body", "6": "t", "0": "unknown"}
DIMENSION_ORDER = ("overall", "surface", "derived_period", "surface_derived_period")
PRESENCE_STATES = ("null", "empty", "whitespace_only", "substantive")
PERCENTILE_METHOD = "linear"
CUTS = (10, 25, 50, 100)

GROUP_COLUMNS = [
    "dimension", "surface", "derived_period", "direction_code",
    "direction_label", "tactically_actionable", "group_second_serve_points",
    "direction_points", "direction_proportion", "matches", "servers",
    "server_wins", "server_win_rate", "conservative_aces",
    "conservative_unreturned", "conservative_faults",
]
COVERAGE_COLUMNS = [
    "server_player", "surface", "derived_period", "direction_code",
    "direction_label", "tactically_actionable", "points", "matches",
    "server_wins", "server_win_rate", "first_date", "last_date",
]


def classify_presence(value: object) -> str:
    if value is None or value is pd.NA or (
        not isinstance(value, str) and pd.isna(value)
    ):
        return "null"
    if not isinstance(value, str):
        raise TypeError("second_serve debe contener cadenas o nulos.")
    if value == "":
        return "empty"
    if value.isspace():
        return "whitespace_only"
    return "substantive"


def derive_period(year: int) -> str:
    if year <= 2009:
        return "to_2009"
    if year <= 2019:
        return "2010s"
    return "2020s"


def validate_points(points: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(points.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    frame = points[SOURCE_COLUMNS].copy(deep=True)
    if frame[["match_id", "point_number"]].isna().any().any():
        raise ValueError("Las claves de punto no pueden contener nulos.")
    if frame.duplicated(["match_id", "point_number"]).any():
        raise ValueError("La clave global (match_id, point_number) no es unica.")
    for column in ("server", "point_winner"):
        invalid = ~frame[column].map(
            lambda value: (
                isinstance(value, Integral)
                and not isinstance(value, bool)
                and value in (1, 2)
            )
        )
        if invalid.any():
            values = sorted(frame.loc[invalid, column].astype(str).unique())
            raise ValueError(f"Dominio invalido en {column}: {values}")
    if frame[["player_1", "player_2"]].isna().any().any():
        raise ValueError("Los jugadores no pueden contener nulos.")
    if frame["player_1"].eq(frame["player_2"]).any():
        raise ValueError("player_1 y player_2 deben ser distintos.")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date contiene fechas nulas o no validas.")
    frame["date"] = dates
    invalid_surface = ~frame["surface"].isin(SURFACE_ORDER)
    if invalid_surface.any():
        values = sorted(frame.loc[invalid_surface, "surface"].astype(str).unique())
        raise ValueError(f"Superficies no permitidas: {values}")
    if PARSER_VERSION != "0.2.0":
        raise ValueError(f"Version de parser inesperada: {PARSER_VERSION}")
    if GRAMMAR_VERSION != "mcp-0.3.2-project-0.2":
        raise ValueError(f"Version gramatical inesperada: {GRAMMAR_VERSION}")
    return frame


def _parse_facts(result) -> dict:
    structure = result.structure
    if isinstance(structure, (ServeAce, ServeUnreturned, ServeFault)):
        prefix = structure.prefix
    elif isinstance(structure, ServicePrefix):
        prefix = structure
    else:
        prefix = None
    direction = None if prefix is None else prefix.direction
    complete = (
        structure is not None
        and result.structural_status.value == "consistent"
        and not result.residual_spans
        and not result.residual_text
        and not result.warnings
        and len(result.consumed_spans) == 1
        and result.consumed_spans[0].start == 0
        and result.consumed_spans[0].end == len(result.raw_sequence)
    )
    outcomes = {ServeAce: "ace", ServeUnreturned: "unreturned", ServeFault: "fault"}
    conservative = outcomes.get(type(structure), "not_assigned") if complete else "not_assigned"
    return {
        "structure_type": "none" if structure is None else type(structure).__name__,
        "structural_status": result.structural_status.value,
        "has_warnings": bool(result.warnings),
        "has_residual": bool(result.residual_spans or result.residual_text),
        "lexical_coverage": result.lexical_coverage.proportion,
        "syntactic_coverage": result.syntactic_coverage.proportion,
        "semantic_coverage": result.semantic_coverage.proportion,
        "direction_code": direction,
        "conservative_serve_outcome": conservative,
    }


def prepare_points(
    points: pd.DataFrame,
    parser: Callable = parse_sequence,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = validate_points(points)
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["derived_period"] = frame["year"].map(derive_period)
    frame["server_player"] = np.where(frame["server"].eq(1), frame["player_1"], frame["player_2"])
    frame["returner_player"] = np.where(frame["server"].eq(1), frame["player_2"], frame["player_1"])
    frame["server_won_point"] = frame["server"].eq(frame["point_winner"])
    frame["second_serve_presence_state"] = frame["second_serve"].map(classify_presence)
    substantive = frame[frame["second_serve_presence_state"].eq("substantive")].copy()
    cache = {value: _parse_facts(parser(value, serve_number=2)) for value in substantive["second_serve"].unique()}
    facts = pd.DataFrame(substantive["second_serve"].map(cache).tolist(), index=substantive.index)
    substantive = pd.concat([substantive, facts], axis=1)
    substantive["direction_label"] = substantive["direction_code"].map(DIRECTION_LABELS)
    substantive["tactically_actionable"] = substantive["direction_code"].isin(["4", "5", "6"])
    return frame, {"substantive": substantive, "cache": cache}


def _ordered(frame: pd.DataFrame, column: str, order: tuple[str, ...]) -> pd.Series:
    return pd.Categorical(frame[column], categories=order, ordered=True)


def build_group_table(substantive: pd.DataFrame) -> pd.DataFrame:
    directional = substantive[substantive["direction_code"].notna()].copy()
    rows = []
    specs = [
        ("overall", [], "<ALL>", "<ALL>"),
        ("surface", ["surface"], None, "<ALL>"),
        ("derived_period", ["derived_period"], "<ALL>", None),
        ("surface_derived_period", ["surface", "derived_period"], None, None),
    ]
    for dimension, keys, fixed_surface, fixed_period in specs:
        groups = [((), substantive)] if not keys else substantive.groupby(keys, sort=False, observed=True)
        for key, population in groups:
            key = key if isinstance(key, tuple) else (key,)
            values = dict(zip(keys, key))
            surface = fixed_surface if fixed_surface is not None else values["surface"]
            period = fixed_period if fixed_period is not None else values["derived_period"]
            selected = directional.loc[population.index.intersection(directional.index)]
            for direction in DIRECTION_ORDER:
                part = selected[selected["direction_code"].eq(direction)]
                denominator = len(population)
                numerator = len(part)
                rows.append({
                    "dimension": dimension, "surface": surface,
                    "derived_period": period, "direction_code": direction,
                    "direction_label": DIRECTION_LABELS[direction],
                    "tactically_actionable": direction != "0",
                    "group_second_serve_points": denominator,
                    "direction_points": numerator,
                    "direction_proportion": numerator / denominator if denominator else None,
                    "matches": part["match_id"].nunique(),
                    "servers": part["server_player"].nunique(),
                    "server_wins": int(part["server_won_point"].sum()),
                    "server_win_rate": part["server_won_point"].mean() if numerator else None,
                    "conservative_aces": int(part["conservative_serve_outcome"].eq("ace").sum()),
                    "conservative_unreturned": int(part["conservative_serve_outcome"].eq("unreturned").sum()),
                    "conservative_faults": int(part["conservative_serve_outcome"].eq("fault").sum()),
                })
    result = pd.DataFrame(rows, columns=GROUP_COLUMNS)
    result["_dimension"] = _ordered(result, "dimension", DIMENSION_ORDER)
    result["_surface"] = _ordered(result, "surface", ("<ALL>",) + SURFACE_ORDER)
    result["_period"] = _ordered(result, "derived_period", ("<ALL>",) + PERIOD_ORDER)
    result["_direction"] = _ordered(result, "direction_code", DIRECTION_ORDER)
    return result.sort_values(["_dimension", "_surface", "_period", "_direction"]).drop(columns=["_dimension", "_surface", "_period", "_direction"]).reset_index(drop=True)


def build_coverage_table(substantive: pd.DataFrame) -> pd.DataFrame:
    directional = substantive[substantive["direction_code"].notna()]
    result = directional.groupby(
        ["server_player", "surface", "derived_period", "direction_code"],
        as_index=False, sort=False, observed=True,
    ).agg(points=("point_number", "size"), matches=("match_id", "nunique"),
          server_wins=("server_won_point", "sum"), first_date=("date", "min"), last_date=("date", "max"))
    result["direction_label"] = result["direction_code"].map(DIRECTION_LABELS)
    result["tactically_actionable"] = result["direction_code"].ne("0")
    result["server_win_rate"] = result["server_wins"] / result["points"]
    result = result[COVERAGE_COLUMNS]
    result["_surface"] = _ordered(result, "surface", SURFACE_ORDER)
    result["_period"] = _ordered(result, "derived_period", PERIOD_ORDER)
    result["_direction"] = _ordered(result, "direction_code", DIRECTION_ORDER)
    return result.sort_values(["server_player", "_surface", "_period", "_direction"]).drop(columns=["_surface", "_period", "_direction"]).reset_index(drop=True)


def _distribution(values: pd.Series) -> dict:
    names = ("minimum", "p25", "median", "p75", "p90", "p95", "p99", "maximum")
    numbers = np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100], method=PERCENTILE_METHOD)
    return {name: float(value) for name, value in zip(names, numbers)}


def _coverage_distribution(coverage: pd.DataFrame) -> dict:
    denominator = len(coverage)
    return {
        "unit": "server_player_surface_derived_period_direction",
        "combinations": denominator,
        "percentile_method": PERCENTILE_METHOD,
        "points_per_combination": _distribution(coverage["points"]),
        "matches_per_combination": _distribution(coverage["matches"]),
        "descriptive_point_cuts": {
            f"at_least_{cut}_points": {
                "numerator": int(coverage["points"].ge(cut).sum()),
                "denominator": denominator,
                "proportion": float(coverage["points"].ge(cut).mean()),
                "represented_servers": int(coverage.loc[coverage["points"].ge(cut), "server_player"].nunique()),
            } for cut in CUTS
        },
    }


def _annual_fragmentation(substantive: pd.DataFrame) -> dict:
    directional = substantive[substantive["direction_code"].notna()]
    annual = directional.groupby(["server_player", "surface", "year", "direction_code"], as_index=False).agg(
        points=("point_number", "size"), matches=("match_id", "nunique"))
    result = _coverage_distribution(annual.rename(columns={"year": "derived_period"}))
    result["unit"] = "server_player_surface_year_direction"
    result["conclusion"] = "El ano fragmenta la muestra; derived_period es la dimension temporal principal."
    return result


def _assert_frame_matches(
    name: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if actual.duplicated(key_columns).any():
        raise ValueError(f"{name}: clave no unica.")
    if expected.duplicated(key_columns).any():
        raise ValueError(f"{name}: clave canonica no unica.")
    actual_keys = set(map(tuple, actual[key_columns].to_numpy()))
    expected_keys = set(map(tuple, expected[key_columns].to_numpy()))
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        extra = sorted(actual_keys - expected_keys)[:5]
        raise ValueError(f"{name}: claves no reconcilian; faltan={missing}, sobran={extra}.")
    actual_sorted = actual.sort_values(key_columns).reset_index(drop=True)
    expected_sorted = expected.sort_values(key_columns).reset_index(drop=True)
    actual_sorted = actual_sorted[expected_sorted.columns]
    try:
        pdt.assert_frame_equal(
            actual_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"{name}: valores publicados no reconcilian.") from exc


def _canonical_group_table(substantive: pd.DataFrame) -> pd.DataFrame:
    return build_group_table(substantive)


def _canonical_coverage_table(substantive: pd.DataFrame) -> pd.DataFrame:
    return build_coverage_table(substantive)


def _canonical_outcome_counts(substantive: pd.DataFrame) -> dict[str, int]:
    counts = substantive["conservative_serve_outcome"].value_counts()
    return {
        outcome: int(counts.get(outcome, 0))
        for outcome in ("not_assigned", "fault", "ace", "unreturned")
    }


def _canonical_contradictions(substantive: pd.DataFrame) -> list[dict]:
    expected = {"ace": True, "unreturned": True, "fault": False}
    rows = []
    for outcome, expected_win in expected.items():
        selected = substantive[substantive["conservative_serve_outcome"].eq(outcome)]
        bad = selected[selected["server_won_point"].ne(expected_win)].sort_values(
            ["match_id", "point_number"]
        )
        rows.append({
            "outcome": outcome,
            "rows": len(selected),
            "contradictions": len(bad),
            "denominator": len(selected),
            "proportion": None if not len(selected) else len(bad) / len(selected),
            "examples": bad[
                ["match_id", "point_number", "second_serve", "server", "point_winner"]
            ].head(5).to_dict("records"),
        })
    return rows


def _validate_summary_contract(
    frame: pd.DataFrame,
    substantive: pd.DataFrame,
    groups: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: dict,
) -> None:
    presence = summary["population"]["presence_states"]
    if set(presence) != set(PRESENCE_STATES):
        raise ValueError("Los estados de presencia publicados no son exactos.")
    for state in PRESENCE_STATES:
        actual = presence[state]
        expected_numerator = int(frame["second_serve_presence_state"].eq(state).sum())
        expected = {
            "numerator": expected_numerator,
            "denominator": len(frame),
            "proportion": expected_numerator / len(frame) if len(frame) else None,
        }
        if actual != expected:
            raise ValueError(f"El estado de presencia {state} no reconcilia.")
    expected_population = {
        "all_points": len(frame),
        "presence_states": presence,
        "substantive_second_serve_points": len(substantive),
        "matches": int(substantive["match_id"].nunique()),
        "servers": int(substantive["server_player"].nunique()),
    }
    if summary["population"] != expected_population:
        raise ValueError("La poblacion publicada no reconcilia.")
    directional = substantive[substantive["direction_code"].notna()]
    expected_parser = {
        "denominator_substantive_sequences": len(substantive),
        "recognized_direction": int(substantive["direction_code"].notna().sum()),
        "without_direction": int(substantive["direction_code"].isna().sum()),
        "structural_status": {
            str(k): int(v)
            for k, v in substantive["structural_status"].value_counts().sort_index().items()
        },
        "with_warnings": int(substantive["has_warnings"].sum()),
        "with_residual": int(substantive["has_residual"].sum()),
        "unique_sequences_parsed": int(substantive["second_serve"].nunique()),
    }
    if summary["parser_coverage"] != expected_parser:
        raise ValueError("La cobertura del parser publicada no reconcilia.")
    if summary.get("conservative_outcome_counts") != _canonical_outcome_counts(substantive):
        raise ValueError("Los conteos de outcomes conservadores no reconcilian.")
    if sum(summary["conservative_outcome_counts"].values()) != len(substantive):
        raise ValueError("Los outcomes conservadores no suman la poblacion sustantiva.")
    if summary["conservative_outcome_reconciliation"] != _canonical_contradictions(substantive):
        raise ValueError("Las contradicciones publicadas no reconcilian.")
    _assert_frame_matches(
        "direcciones publicadas",
        pd.DataFrame(summary["directions"]),
        groups[groups["dimension"].eq("overall")],
        ["dimension", "surface", "derived_period", "direction_code"],
    )
    _assert_frame_matches(
        "resultados por superficie publicados",
        pd.DataFrame(summary["results_by_surface"]),
        groups[groups["dimension"].eq("surface")],
        ["dimension", "surface", "derived_period", "direction_code"],
    )
    _assert_frame_matches(
        "resultados por periodo publicados",
        pd.DataFrame(summary["results_by_period"]),
        groups[groups["dimension"].eq("derived_period")],
        ["dimension", "surface", "derived_period", "direction_code"],
    )
    if int(coverage["points"].sum()) != len(directional):
        raise ValueError("La cobertura publicada no reconcilia la direccion reconocida.")


def validate_reconciliations(
    frame: pd.DataFrame,
    substantive: pd.DataFrame,
    groups: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: dict | None = None,
) -> dict:
    expected_states = frame["second_serve"].map(classify_presence)
    if not frame["second_serve_presence_state"].eq(expected_states).all():
        raise ValueError("Los estados de presencia no corresponden a second_serve.")
    states = frame["second_serve_presence_state"].value_counts()
    if int(states.sum()) != len(frame) or set(states.index) - set(PRESENCE_STATES):
        raise ValueError("Los estados de presencia no reconcilian.")
    directional = substantive[substantive["direction_code"].notna()]
    if len(directional) + int(substantive["direction_code"].isna().sum()) != len(substantive):
        raise ValueError("La cobertura de direccion no reconcilia.")
    _assert_frame_matches(
        "grupos",
        groups,
        _canonical_group_table(substantive),
        ["dimension", "surface", "derived_period", "direction_code"],
    )
    if coverage["server_wins"].gt(coverage["points"]).any():
        raise ValueError("server_wins no puede superar points.")
    if coverage["first_date"].gt(coverage["last_date"]).any():
        raise ValueError("Las fechas de cobertura no estan ordenadas.")
    _assert_frame_matches(
        "cobertura",
        coverage,
        _canonical_coverage_table(substantive),
        ["server_player", "surface", "derived_period", "direction_code"],
    )
    outcomes = substantive["conservative_serve_outcome"]
    if not outcomes.isin(["ace", "unreturned", "fault", "not_assigned"]).all():
        raise ValueError("Los outcomes conservadores no son excluyentes.")
    if sum(_canonical_outcome_counts(substantive).values()) != len(substantive):
        raise ValueError("Los outcomes conservadores no reconcilian.")
    if summary is not None:
        _validate_summary_contract(frame, substantive, groups, coverage, summary)
    return {"presence_states_equal_source_rows": True, "direction_partition_complete": True,
            "group_dimensions_reconciled": True, "coverage_points_reconciled": True,
            "coverage_key_unique": True, "proportions_reconciled": True,
            "conservative_outcomes_mutually_exclusive": True,
            "summary_contract_reconciled": summary is not None}


def build_summary(frame: pd.DataFrame, substantive: pd.DataFrame, groups: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    directional = substantive[substantive["direction_code"].notna()]
    presence = {state: {"numerator": int(frame["second_serve_presence_state"].eq(state).sum()), "denominator": len(frame), "proportion": float(frame["second_serve_presence_state"].eq(state).mean())} for state in PRESENCE_STATES}
    contradictions = _canonical_contradictions(substantive)
    parser_coverage = {
        "denominator_substantive_sequences": len(substantive),
        "recognized_direction": int(substantive["direction_code"].notna().sum()),
        "without_direction": int(substantive["direction_code"].isna().sum()),
        "structural_status": {str(k): int(v) for k, v in substantive["structural_status"].value_counts().sort_index().items()},
        "with_warnings": int(substantive["has_warnings"].sum()),
        "with_residual": int(substantive["has_residual"].sum()),
        "unique_sequences_parsed": int(substantive["second_serve"].nunique()),
    }
    direction_rows = groups[groups["dimension"].eq("overall")]
    summary = {
        "schema_version": 1,
        "source": {"path": "data/processed/points_enriched.parquet", "columns_used": SOURCE_COLUMNS, "point_rows": len(frame)},
        "parser": {"version": PARSER_VERSION, "grammar_version": GRAMMAR_VERSION, "serve_number": 2},
        "definitions": {"analytical_population": "Puntos con secuencia sustantiva de segundo servicio; no equivale necesariamente a todos los segundos servicios jugados.", "derived_period": {"to_2009": "year <= 2009", "2010s": "2010 <= year <= 2019", "2020s": "year >= 2020"}, "direction_codes": DIRECTION_LABELS, "outcome_policy": "Solo estructuras completas, consistentes, sin residuo ni warnings.", "percentile_method": PERCENTILE_METHOD},
        "methodological_flags": {"descriptive_analysis": True, "causal_effect_estimated": False, "model_feature_approved": False, "model_thresholds_approved": False},
        "population": {"all_points": len(frame), "presence_states": presence, "substantive_second_serve_points": len(substantive), "matches": int(substantive["match_id"].nunique()), "servers": int(substantive["server_player"].nunique())},
        "parser_coverage": parser_coverage,
        "conservative_outcome_counts": _canonical_outcome_counts(substantive),
        "directions": direction_rows.to_dict("records"),
        "results_by_surface": groups[groups["dimension"].eq("surface")].to_dict("records"),
        "results_by_period": groups[groups["dimension"].eq("derived_period")].to_dict("records"),
        "conservative_outcome_reconciliation": contradictions,
        "coverage_distribution": {"all_directions": _coverage_distribution(coverage), "tactically_actionable_directions": _coverage_distribution(coverage[coverage["tactically_actionable"]])},
        "annual_fragmentation": _annual_fragmentation(substantive),
        "decisions": {"rally_length_included": False, "service_side_included": False, "direction_zero_tactically_actionable": False, "descriptive_cuts_are_model_thresholds": False},
        "limitations": ["Analisis observacional y no causal.", "La poblacion esta condicionada a una secuencia sustantiva de segundo servicio.", "La cobertura varia entre jugadores, superficies y periodos.", "No se interpreta el rally residual.", "No se calcula rally length ni lado de servicio."],
        "example_selection_policy": "Hasta cinco contradicciones por outcome, ordenadas por match_id y point_number.",
    }
    summary["reconciliations"] = validate_reconciliations(
        frame, substantive, groups, coverage, summary
    )
    return summary


def analyze_second_serve_directions(points: pd.DataFrame, parser: Callable = parse_sequence):
    frame, prepared = prepare_points(points, parser=parser)
    substantive = prepared["substantive"]
    groups = build_group_table(substantive)
    coverage = build_coverage_table(substantive)
    summary = build_summary(frame, substantive, groups, coverage)
    return summary, groups, coverage, substantive


def write_artifacts(summary: dict, groups: pd.DataFrame, coverage: pd.DataFrame, reports_dir: Path = REPORTS_DIR, tables_dir: Path = TABLES_DIR) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "second_serve_direction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    groups.to_csv(tables_dir / "second_serve_direction_by_group.csv", index=False, encoding="utf-8", na_rep="<NULL>", lineterminator="\n")
    coverage.to_csv(tables_dir / "second_serve_direction_coverage.csv", index=False, encoding="utf-8", na_rep="<NULL>", date_format="%Y-%m-%d", lineterminator="\n")


def main() -> None:
    if not POINTS_FILE.exists():
        raise FileNotFoundError(f"No se encontro: {POINTS_FILE}")
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    summary, groups, coverage, _ = analyze_second_serve_directions(points)
    write_artifacts(summary, groups, coverage)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
