import csv
import json
from pathlib import Path

import pandas as pd

from src.data.quality_rules import (
    CONFLICTING_MATCH_IDS,
    MATCH_METADATA_CORRECTIONS,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "match_charting_project"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

MATCH_COLUMNS = {
    "Player 1": "player_1",
    "Player 2": "player_2",
    "Pl 1 hand": "player_1_hand",
    "Pl 2 hand": "player_2_hand",
    "Date": "date",
    "Tournament": "tournament",
    "Round": "round",
    "Time": "time",
    "Court": "court",
    "Surface": "surface",
    "Umpire": "umpire",
    "Best of": "best_of",
    "Final TB?": "final_tiebreak",
    "Charted by": "charted_by",
}

POINT_COLUMNS = {
    "Pt": "point_number",
    "Set1": "sets_player_1",
    "Set2": "sets_player_2",
    "Gm1": "games_player_1",
    "Gm2": "games_player_2",
    "Pts": "game_score",
    "Gm#": "game_number",
    "TbSet": "tiebreak_set",
    "Svr": "server",
    "1st": "first_serve",
    "2nd": "second_serve",
    "Notes": "notes",
    "PtWinner": "point_winner",
}

ALLOWED_SURFACES = {"Hard", "Clay", "Grass"}
ALLOWED_BEST_OF = {"3", "5"}


def count_malformed_csv_rows(csv_path: Path) -> int:
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        expected_columns = len(header)
        return sum(len(row) != expected_columns for row in reader)


def apply_match_metadata_corrections(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    corrected = matches.copy(deep=True)
    corrected_match_ids = set()

    for match_id, values in MATCH_METADATA_CORRECTIONS.items():
        match_mask = corrected["match_id"].eq(match_id)
        match_changed = False

        for column, value in values.items():
            if pd.isna(value):
                values_are_equal = corrected.loc[match_mask, column].isna()
            else:
                values_are_equal = (
                    corrected.loc[match_mask, column]
                    .eq(value)
                    .fillna(False)
                )

            if not values_are_equal.all():
                match_changed = True

            corrected.loc[match_mask, column] = value

        if match_changed:
            corrected_match_ids.add(match_id)

    return corrected, len(corrected_match_ids)


def validate_clean_matches(matches: pd.DataFrame) -> None:
    invalid_surfaces = ~matches["surface"].isin(ALLOWED_SURFACES)
    if invalid_surfaces.any():
        values = sorted(
            matches.loc[invalid_surfaces, "surface"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(f"Superficies no permitidas: {values}")

    non_null_best_of = matches["best_of"].dropna()
    invalid_best_of = ~non_null_best_of.isin(ALLOWED_BEST_OF)
    if invalid_best_of.any():
        values = sorted(non_null_best_of.loc[invalid_best_of].unique())
        raise ValueError(f"Valores de best_of no permitidos: {values}")

    if matches["match_id"].duplicated().any():
        raise ValueError("match_id no es único en los metadatos.")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(
        RAW_DIR / "charting-m-matches.csv",
        dtype=str,
    )

    point_frames = [
        pd.read_csv(path, dtype=str)
        for path in sorted(RAW_DIR.glob("charting-m-points-*.csv"))
    ]

    if len(point_frames) != 3:
        raise RuntimeError("Se esperaban exactamente tres archivos de puntos.")

    points = pd.concat(point_frames, ignore_index=True)
    return matches, points


def clean_matches(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    parsed_dates = pd.to_datetime(
        matches["Date"],
        format="%Y%m%d",
        errors="coerce",
    )

    invalid_date_rows = int(parsed_dates.isna().sum())

    clean = matches.loc[parsed_dates.notna()].copy()
    clean["Date"] = parsed_dates.loc[parsed_dates.notna()]

    clean = clean.drop_duplicates(subset=["match_id"], keep="first")
    clean = clean.rename(columns=MATCH_COLUMNS)
    clean, metadata_rows_corrected = apply_match_metadata_corrections(
        clean
    )
    validate_clean_matches(clean)

    return clean, invalid_date_rows, metadata_rows_corrected


def clean_points(points: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_rows = len(points)

    excluded_mask = points["match_id"].isin(CONFLICTING_MATCH_IDS)
    excluded_rows = int(excluded_mask.sum())

    clean = points.loc[~excluded_mask].copy()

    rows_before_deduplication = len(clean)
    clean = clean.drop_duplicates()
    exact_duplicate_rows_removed = rows_before_deduplication - len(clean)

    remaining_duplicate_keys = int(
        clean.duplicated(["match_id", "Pt"]).sum()
    )

    if remaining_duplicate_keys:
        raise ValueError(
            "Siguen existiendo claves (match_id, Pt) duplicadas."
        )

    clean = clean.rename(columns=POINT_COLUMNS)

    integer_columns = [
        "point_number",
        "sets_player_1",
        "sets_player_2",
        "games_player_1",
        "games_player_2",
        "tiebreak_set",
        "server",
        "point_winner",
    ]

    for column in integer_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        ).astype("Int64")

    summary = {
        "original_point_rows": int(original_rows),
        "conflicting_matches_excluded": len(CONFLICTING_MATCH_IDS),
        "rows_from_conflicting_matches_excluded": excluded_rows,
        "exact_duplicate_rows_removed": int(exact_duplicate_rows_removed),
        "clean_point_rows": int(len(clean)),
    }

    return clean, summary


def main() -> None:
    matches_raw, points_raw = load_raw_data()

    malformed_csv_rows = count_malformed_csv_rows(
        RAW_DIR / "charting-m-matches.csv"
    )
    matches, invalid_date_rows, metadata_rows_corrected = clean_matches(
        matches_raw
    )
    points, point_summary = clean_points(points_raw)

    enriched = points.merge(
        matches,
        on="match_id",
        how="inner",
        validate="many_to_one",
    )

    matches_without_points = int(
        (~matches["match_id"].isin(points["match_id"])).sum()
    )

    points_without_metadata = len(points) - len(enriched)

    if points_without_metadata:
        raise ValueError(
            f"Hay {points_without_metadata} puntos sin metadatos."
        )

    if enriched.duplicated(["match_id", "point_number"]).any():
        raise ValueError("La clave de punto no es única tras la unión.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    matches.to_parquet(
        PROCESSED_DIR / "matches_clean.parquet",
        index=False,
    )
    enriched.to_parquet(
        PROCESSED_DIR / "points_enriched.parquet",
        index=False,
    )

    report = {
        "raw_match_rows": int(len(matches_raw)),
        "malformed_csv_rows_detected": malformed_csv_rows,
        "invalid_match_rows_removed": invalid_date_rows,
        "metadata_rows_corrected": metadata_rows_corrected,
        "unknown_surface_matches": int(matches["surface"].isna().sum()),
        "clean_match_rows": int(len(matches)),
        **point_summary,
        "matches_without_points": matches_without_points,
        "enriched_point_rows": int(len(enriched)),
        "unique_enriched_matches": int(enriched["match_id"].nunique()),
    }

    (REPORT_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
