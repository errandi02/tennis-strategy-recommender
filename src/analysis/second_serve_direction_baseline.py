"""Baseline descriptivo de direccion del segundo servicio."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.second_serve_direction_analysis import (
    CUTS,
    PERIOD_ORDER,
    POINTS_FILE,
    REPORTS_DIR,
    SOURCE_COLUMNS,
    SURFACE_ORDER,
    TABLES_DIR,
    prepare_points,
)


DIRECTION_ORDER = ("wide", "body", "T")
DIRECTION_LABELS = {"4": "wide", "5": "body", "6": "T"}
PERCENTILE_METHOD = "linear"
SUMMARY_PATH = REPORTS_DIR / "second_serve_direction_baseline_summary.json"
GROUP_TABLE_PATH = TABLES_DIR / "second_serve_direction_baseline_by_group.csv"
STATISTICAL_PACKAGES = ("statsmodels", "sklearn", "scipy")

ANALYTIC_COLUMNS = [
    "match_id",
    "point_number",
    "server_player",
    "surface",
    "derived_period",
    "direction",
    "server_won_point",
]

GROUP_COLUMNS = [
    "dimension",
    "direction",
    "surface",
    "derived_period",
    "points",
    "matches",
    "servers",
    "server_wins",
    "server_win_rate",
]


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, float) or np.isfinite(value)


def _assert_no_nan_or_inf(value: Any, path: str = "root") -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_nan_or_inf(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_nan_or_inf(item, f"{path}[{index}]")
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"Valor no finito en {path}.")


def _available_statistical_packages() -> dict[str, bool]:
    return {
        package: importlib.util.find_spec(package) is not None
        for package in STATISTICAL_PACKAGES
    }


def build_analytic_population(points: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filtra la poblacion aprobada sin introducir nuevas reglas de parser."""
    _, prepared = prepare_points(points)
    substantive = prepared["substantive"]
    # The analytical contract only concerns the recognised initial serve
    # prefix. Rally residue and parser warnings after that prefix are not a
    # reason to discard the observed point outcome.
    eligible = substantive[substantive["direction_code"].isin(DIRECTION_LABELS)].copy()
    eligible["direction"] = eligible["direction_code"].map(DIRECTION_LABELS)
    analytic = eligible[ANALYTIC_COLUMNS].copy()
    analytic["direction"] = pd.Categorical(
        analytic["direction"],
        categories=DIRECTION_ORDER,
        ordered=True,
    )
    analytic = analytic.sort_values(
        ["match_id", "point_number", "direction"],
    ).reset_index(drop=True)
    audit = {
        "source_rows": int(len(points)),
        "substantive_second_serve_rows": int(len(substantive)),
        "unique_sequences_parsed": int(len(prepared["cache"])),
        "eligible_rows": int(len(analytic)),
        "recognized_direction_rows": int(substantive["direction_code"].notna().sum()),
        "excluded_direction_0": int(substantive["direction_code"].eq("0").sum()),
        "excluded_without_recognized_direction": int(substantive["direction_code"].isna().sum()),
        "conservative_outcome_diagnostic": {
            outcome: int(substantive["conservative_serve_outcome"].eq(outcome).sum())
            for outcome in ("ace", "unreturned", "fault", "not_assigned")
        },
    }
    return analytic, audit


def _aggregate(frame: pd.DataFrame, keys: list[str], dimension: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(keys, as_index=False, observed=True, sort=False)
        .agg(
            points=("point_number", "size"),
            matches=("match_id", "nunique"),
            servers=("server_player", "nunique"),
            server_wins=("server_won_point", "sum"),
        )
    )
    grouped["dimension"] = dimension
    for column, value in (("surface", "<ALL>"), ("derived_period", "<ALL>")):
        if column not in grouped:
            grouped[column] = value
    grouped["server_wins"] = grouped["server_wins"].astype(int)
    grouped["server_win_rate"] = grouped["server_wins"] / grouped["points"]
    return grouped[GROUP_COLUMNS]


def build_group_table(analytic: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _aggregate(analytic, ["direction"], "direction"),
        _aggregate(analytic, ["direction", "surface"], "direction_surface"),
        _aggregate(analytic, ["direction", "derived_period"], "direction_period"),
        _aggregate(
            analytic,
            ["direction", "surface", "derived_period"],
            "direction_surface_period",
        ),
    ]
    table = pd.concat(rows, ignore_index=True)
    table = table[table["points"].gt(0)].copy()
    table["_dimension"] = pd.Categorical(
        table["dimension"],
        categories=(
            "direction",
            "direction_surface",
            "direction_period",
            "direction_surface_period",
        ),
        ordered=True,
    )
    table["_direction"] = pd.Categorical(
        table["direction"],
        categories=DIRECTION_ORDER,
        ordered=True,
    )
    table["_surface"] = pd.Categorical(
        table["surface"],
        categories=("<ALL>",) + SURFACE_ORDER,
        ordered=True,
    )
    table["_period"] = pd.Categorical(
        table["derived_period"],
        categories=("<ALL>",) + PERIOD_ORDER,
        ordered=True,
    )
    return table.sort_values(
        ["_dimension", "_direction", "_surface", "_period"],
    ).drop(
        columns=["_dimension", "_direction", "_surface", "_period"],
    ).reset_index(drop=True)


def build_coverage_table(analytic: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        analytic.groupby(
            ["server_player", "surface", "derived_period", "direction"],
            as_index=False,
            observed=True,
            sort=False,
        )
        .agg(points=("point_number", "size"), matches=("match_id", "nunique"))
    )
    coverage["_direction"] = pd.Categorical(
        coverage["direction"],
        categories=DIRECTION_ORDER,
        ordered=True,
    )
    coverage["_surface"] = pd.Categorical(
        coverage["surface"],
        categories=SURFACE_ORDER,
        ordered=True,
    )
    coverage["_period"] = pd.Categorical(
        coverage["derived_period"],
        categories=PERIOD_ORDER,
        ordered=True,
    )
    return coverage.sort_values(
        ["server_player", "_surface", "_period", "_direction"],
    ).drop(
        columns=["_direction", "_surface", "_period"],
    ).reset_index(drop=True)


def _distribution(values: pd.Series) -> dict[str, float]:
    percentiles = np.percentile(
        values,
        [25, 50, 75, 90, 95, 99],
        method=PERCENTILE_METHOD,
    )
    return {
        "p25": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
    }


def build_coverage_summary(coverage: pd.DataFrame) -> dict[str, Any]:
    denominator = int(len(coverage))
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
            }
            for cut in CUTS
        },
    }


def build_identifiability_summary(analytic: pd.DataFrame) -> dict[str, Any]:
    directions_per_server = analytic.groupby(
        "server_player",
        observed=True,
    )["direction"].nunique()
    observations_per_server = analytic.groupby(
        "server_player",
        observed=True,
    ).size()
    return {
        "servers": int(analytic["server_player"].nunique()),
        "matches": int(analytic["match_id"].nunique()),
        "observations_by_direction": {
            direction: int(analytic["direction"].eq(direction).sum())
            for direction in DIRECTION_ORDER
        },
        "servers_with_multiple_directions": int(directions_per_server.ge(2).sum()),
        "servers_with_three_directions": int(directions_per_server.eq(3).sum()),
        "servers_with_at_least_10_points": int(observations_per_server.ge(10).sum()),
        "surface_period_groups": int(
            analytic[["surface", "derived_period"]].drop_duplicates().shape[0]
        ),
    }


def build_adjusted_analysis(analytic: pd.DataFrame) -> dict[str, Any]:
    packages = _available_statistical_packages()
    if not packages["statsmodels"]:
        return {
            "status": "not_available",
            "reason": "statsmodels no esta disponible en el entorno y no se anaden dependencias en esta fase.",
            "formula": "server_won_point ~ direction + surface + derived_period + server_player",
            "variables": ["direction", "surface", "derived_period", "server_player"],
            "available_packages": packages,
        }
    return {
        "status": "not_available",
        "reason": "El ajuste queda pendiente de aprobacion metodologica aunque exista soporte estadistico.",
        "formula": "server_won_point ~ direction + surface + derived_period + server_player",
        "variables": ["direction", "surface", "derived_period", "server_player"],
        "available_packages": packages,
    }


def _canonical_group_table(analytic: pd.DataFrame) -> pd.DataFrame:
    return build_group_table(analytic)


def _canonical_coverage_table(analytic: pd.DataFrame) -> pd.DataFrame:
    return build_coverage_table(analytic)


def _assert_frame_matches(
    name: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if actual.duplicated(key_columns).any():
        raise ValueError(f"{name}: clave no unica.")
    actual_keys = set(map(tuple, actual[key_columns].to_numpy()))
    expected_keys = set(map(tuple, expected[key_columns].to_numpy()))
    if actual_keys != expected_keys:
        raise ValueError(f"{name}: claves no reconcilian.")
    actual_sorted = actual.sort_values(key_columns).reset_index(drop=True)
    expected_sorted = expected.sort_values(key_columns).reset_index(drop=True)
    try:
        pdt.assert_frame_equal(
            actual_sorted[expected_sorted.columns],
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"{name}: valores no reconcilian.") from exc


def validate_reconciliations(
    analytic: pd.DataFrame,
    groups: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: dict[str, Any] | None = None,
) -> dict[str, bool]:
    if analytic.empty:
        raise ValueError("La poblacion analitica no puede estar vacia.")
    if analytic.duplicated(["match_id", "point_number"]).any():
        raise ValueError("La clave de punto no es unica.")
    if set(analytic["direction"].astype(str)) - set(DIRECTION_ORDER):
        raise ValueError("Direccion fuera del catalogo del baseline.")
    if not analytic["server_won_point"].isin([True, False]).all():
        raise ValueError("server_won_point debe ser booleano.")
    _assert_frame_matches(
        "grupos",
        groups,
        _canonical_group_table(analytic),
        ["dimension", "direction", "surface", "derived_period"],
    )
    _assert_frame_matches(
        "cobertura",
        coverage,
        _canonical_coverage_table(analytic),
        ["server_player", "surface", "derived_period", "direction"],
    )
    for name, table in (("grupos", groups), ("cobertura", coverage)):
        if table["points"].le(0).any():
            raise ValueError(f"{name}: points debe ser positivo.")
        if table["matches"].gt(table["points"]).any():
            raise ValueError(f"{name}: matches no puede superar points.")
    if groups["server_wins"].gt(groups["points"]).any():
        raise ValueError("grupos: wins no puede superar points.")
    if not np.allclose(groups["server_win_rate"], groups["server_wins"] / groups["points"]):
        raise ValueError("grupos: rates no reconcilian.")
    if int(groups[groups["dimension"].eq("direction")]["points"].sum()) != len(analytic):
        raise ValueError("La suma por direccion no reconcilia la poblacion.")
    if int(groups[groups["dimension"].eq("direction")]["server_wins"].sum()) != int(
        analytic["server_won_point"].sum()
    ):
        raise ValueError("La suma de wins no reconcilia.")
    if int(coverage["points"].sum()) != len(analytic):
        raise ValueError("La cobertura no reconcilia la poblacion.")
    for table_name, table in (("grupos", groups), ("cobertura", coverage)):
        numeric = table.select_dtypes(include=[np.number])
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{table_name}: contiene NaN o infinito.")
    if summary is not None:
        _assert_no_nan_or_inf(summary)
        if summary["population"]["analytic_points"] != len(analytic):
            raise ValueError("summary: poblacion no reconcilia.")
        if summary["descriptive"]["overall"]["points"] != len(analytic):
            raise ValueError("summary: overall no reconcilia.")
        if summary["descriptive"]["overall"]["server_wins"] != int(
            analytic["server_won_point"].sum()
        ):
            raise ValueError("summary: wins no reconcilian.")
        if summary["coverage"] != build_coverage_summary(coverage):
            raise ValueError("summary: cobertura no reconcilia.")
    return {
        "direction_points_equal_population": True,
        "wins_equal_global_wins": True,
        "rates_equal_wins_over_points": True,
        "group_keys_unique": True,
        "coverage_equals_population": True,
        "no_nan_or_inf": True,
        "summary_contract_reconciled": summary is not None,
    }


def build_summary(
    analytic: pd.DataFrame,
    audit: dict[str, Any],
    groups: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    overall = {
        "points": int(len(analytic)),
        "matches": int(analytic["match_id"].nunique()),
        "servers": int(analytic["server_player"].nunique()),
        "server_wins": int(analytic["server_won_point"].sum()),
        "server_win_rate": float(analytic["server_won_point"].mean()),
    }
    summary = {
        "analysis": "second_serve_direction_baseline",
        "version": 1,
        "source": {
            "path": "data/processed/points_enriched.parquet",
            "columns_used": SOURCE_COLUMNS,
        },
        "population": {
            "definition": "Segundo saque sustantivo con prefijo inicial reconocido y direccion 4, 5 o 6; el rally posterior no es filtro.",
            "analytic_points": int(len(analytic)),
            "matches": int(analytic["match_id"].nunique()),
            "servers": int(analytic["server_player"].nunique()),
            "audit": audit,
        },
        "descriptive": {
            "overall": overall,
            "by_direction": groups[groups["dimension"].eq("direction")].to_dict("records"),
            "association_language": "Las diferencias son descriptivas y no estiman efectos causales.",
        },
        "coverage": build_coverage_summary(coverage),
        "identifiability": build_identifiability_summary(analytic),
        "adjusted_analysis": build_adjusted_analysis(analytic),
        "methodological_limits": [
            "No se producen recomendaciones tacticas.",
            "No se estima un efecto causal.",
            "No se usa rally length, resto, golpes ni informacion posterior al punto.",
            "No hay train/test split ni prediccion online.",
            "Los cortes de cobertura son descriptivos, no umbrales aprobados.",
        ],
    }
    summary["reconciliations"] = validate_reconciliations(
        analytic,
        groups,
        coverage,
        summary,
    )
    return summary


def analyze_second_serve_direction_baseline(points: pd.DataFrame):
    analytic, audit = build_analytic_population(points)
    groups = build_group_table(analytic)
    coverage = build_coverage_table(analytic)
    summary = build_summary(analytic, audit, groups, coverage)
    return summary, groups, coverage, analytic


def write_artifacts(
    summary: dict[str, Any],
    groups: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "second_serve_direction_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    groups.to_csv(
        tables_dir / "second_serve_direction_baseline_by_group.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def main() -> None:
    points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
    summary, groups, _, _ = analyze_second_serve_direction_baseline(points)
    write_artifacts(summary, groups)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
