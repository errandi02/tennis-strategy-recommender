import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POINTS_FILE = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

REQUIRED_COLUMNS = {
    "match_id",
    "point_number",
    "player_1",
    "player_2",
    "date",
    "surface",
}
MATCH_METADATA_COLUMNS = ["player_1", "player_2", "date", "surface"]
ALLOWED_SURFACES = {"Hard", "Clay", "Grass"}
DESCRIPTIVE_MATCH_CUTS = (1, 3, 5, 10)


def validate_coverage_points(points: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_COLUMNS - set(points.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")

    validated = points.copy(deep=True)

    if validated[["match_id", "point_number"]].isna().any().any():
        raise ValueError("Las claves de punto no pueden contener nulos.")

    if validated.duplicated(["match_id", "point_number"]).any():
        raise ValueError("La clave (match_id, point_number) no es unica.")

    parsed_dates = pd.to_datetime(
        validated["date"],
        format="mixed",
        errors="coerce",
    )
    if parsed_dates.isna().any():
        raise ValueError("Hay fechas ausentes o no validas.")
    validated["date"] = parsed_dates

    if validated[["player_1", "player_2"]].isna().any().any():
        raise ValueError("Los jugadores no pueden contener nulos.")

    invalid_surfaces = ~validated["surface"].isin(ALLOWED_SURFACES)
    if invalid_surfaces.any():
        values = sorted(
            validated.loc[invalid_surfaces, "surface"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(f"Superficies no permitidas: {values}")

    metadata_counts = validated.groupby("match_id", sort=False)[
        MATCH_METADATA_COLUMNS
    ].nunique(dropna=False)
    inconsistent_matches = metadata_counts.gt(1).any(axis=1)
    if inconsistent_matches.any():
        match_ids = metadata_counts.index[inconsistent_matches].tolist()
        raise ValueError(
            f"Metadatos no constantes dentro del partido: {match_ids}"
        )

    same_player = validated["player_1"].eq(validated["player_2"])
    if same_player.any():
        raise ValueError("Cada partido debe contener dos jugadores distintos.")

    return validated


def reduce_to_matches(points: pd.DataFrame) -> pd.DataFrame:
    validated = validate_coverage_points(points)
    matches = (
        validated.groupby("match_id", sort=False, as_index=False)
        .agg(
            player_1=("player_1", "first"),
            player_2=("player_2", "first"),
            date=("date", "first"),
            surface=("surface", "first"),
            point_count=("point_number", "size"),
        )
    )
    matches["year"] = matches["date"].dt.year.astype(int)
    return matches


def expand_match_participants(matches: pd.DataFrame) -> pd.DataFrame:
    columns = ["match_id", "date", "surface", "year", "point_count"]
    player_1 = matches[columns + ["player_1"]].rename(
        columns={"player_1": "player"}
    )
    player_2 = matches[columns + ["player_2"]].rename(
        columns={"player_2": "player"}
    )
    participants = pd.concat([player_1, player_2], ignore_index=True)

    participant_counts = participants.groupby("match_id").size()
    if not participant_counts.eq(2).all():
        raise ValueError("Cada partido debe aportar exactamente dos registros.")

    return participants


def aggregate_player_coverage(participants: pd.DataFrame) -> pd.DataFrame:
    return (
        participants.groupby(
            ["player", "surface", "year"],
            sort=True,
            as_index=False,
        )
        .agg(
            matches=("match_id", "nunique"),
            points=("point_count", "sum"),
        )
        .sort_values(["player", "surface", "year"])
        .reset_index(drop=True)
    )


def aggregate_surface_year(
    matches: pd.DataFrame,
    participants: pd.DataFrame,
) -> pd.DataFrame:
    totals = (
        matches.groupby(["surface", "year"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            points=("point_count", "sum"),
        )
    )
    players = (
        participants.groupby(["surface", "year"], as_index=False)
        .agg(distinct_players=("player", "nunique"))
    )
    return (
        totals.merge(
            players,
            on=["surface", "year"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["year", "surface"])
        .reset_index(drop=True)
    )


def _distribution(values: pd.Series) -> dict:
    quantiles = values.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.9]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "maximum": float(values.max()),
    }


def validate_coverage_reconciliation(
    matches: pd.DataFrame,
    participants: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict:
    participant_counts = participants.groupby("match_id").size()
    expected_match_ids = set(matches["match_id"])
    participant_match_ids = set(participant_counts.index)
    if (
        participant_match_ids != expected_match_ids
        or not participant_counts.eq(2).all()
    ):
        raise ValueError("Cada partido debe aportar exactamente dos participantes.")

    total_points = int(matches["point_count"].sum())
    coverage_points = int(coverage["points"].sum())
    expected_coverage_points = 2 * total_points
    if coverage_points != expected_coverage_points:
        raise ValueError("La suma de puntos de cobertura no reconcilia.")

    return {
        "expected_participant_rows": int(2 * len(matches)),
        "actual_participant_rows": int(len(participants)),
        "expected_coverage_points": int(expected_coverage_points),
        "actual_coverage_points": coverage_points,
    }


def build_coverage_summary(
    matches: pd.DataFrame,
    participants: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict:
    total_points = int(matches["point_count"].sum())
    reconciliation = validate_coverage_reconciliation(
        matches,
        participants,
        coverage,
    )

    surface_distribution = {}
    for surface in sorted(ALLOWED_SURFACES):
        surface_matches = matches.loc[matches["surface"].eq(surface)]
        surface_participants = participants.loc[
            participants["surface"].eq(surface)
        ]
        surface_distribution[surface] = {
            "matches": int(surface_matches["match_id"].nunique()),
            "points": int(surface_matches["point_count"].sum()),
            "distinct_players": int(surface_participants["player"].nunique()),
        }

    denominator = int(len(coverage))
    cuts = {
        f"at_least_{cut}_matches": {
            "numerator": int(coverage["matches"].ge(cut).sum()),
            "proportion": float(coverage["matches"].ge(cut).mean()),
        }
        for cut in DESCRIPTIVE_MATCH_CUTS
    }

    return {
        "source": "data/processed/points_enriched.parquet",
        "point_rows": total_points,
        "unique_matches": int(matches["match_id"].nunique()),
        "participant_rows": int(len(participants)),
        "player_surface_year_combinations": int(len(coverage)),
        "distinct_players": int(participants["player"].nunique()),
        "first_year": int(matches["year"].min()),
        "last_year": int(matches["year"].max()),
        "surface_distribution": surface_distribution,
        "matches_per_combination": _distribution(coverage["matches"]),
        "points_per_combination": _distribution(coverage["points"]),
        "descriptive_match_cuts": {
            "unit": "player_surface_year_combination",
            "proportion_scale": "0_to_1",
            "denominator": denominator,
            "cuts": cuts,
        },
        "descriptive_cuts_are_model_thresholds": False,
        "reconciliation": reconciliation,
    }


def analyze_coverage(points: pd.DataFrame) -> tuple:
    matches = reduce_to_matches(points)
    participants = expand_match_participants(matches)
    coverage = aggregate_player_coverage(participants)
    surface_year = aggregate_surface_year(matches, participants)
    summary = build_coverage_summary(matches, participants, coverage)
    return matches, participants, coverage, surface_year, summary


def create_figures(
    coverage: pd.DataFrame,
    surface_year: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6))
    for surface in sorted(ALLOWED_SURFACES):
        frame = surface_year.loc[surface_year["surface"].eq(surface)]
        axis.plot(frame["year"], frame["matches"], label=surface)
    axis.set_title("Partidos disponibles por superficie y ano")
    axis.set_xlabel("Ano")
    axis.set_ylabel("Partidos")
    axis.legend(title="Superficie")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "coverage_by_surface_year.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.hist(coverage["matches"], bins=30, edgecolor="black")
    axis.set_title("Distribucion de partidos por jugador-superficie-ano")
    axis.set_xlabel("Partidos por combinacion")
    axis.set_ylabel("Numero de combinaciones")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "coverage_distribution.png", dpi=150)
    plt.close(figure)


def write_coverage_tables(
    coverage: pd.DataFrame,
    surface_year: pd.DataFrame,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(
        tables_dir / "coverage_by_player_surface_year.csv",
        index=False,
    )
    surface_year.to_csv(
        tables_dir / "coverage_by_surface_year.csv",
        index=False,
    )


def main() -> None:
    if not POINTS_FILE.exists():
        raise FileNotFoundError(f"No se encontro: {POINTS_FILE}")

    points = pd.read_parquet(POINTS_FILE)
    _, _, coverage, surface_year, summary = analyze_coverage(points)

    write_coverage_tables(coverage, surface_year, TABLES_DIR)
    (REPORTS_DIR / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    create_figures(coverage, surface_year)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
