import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "match_charting_project"
REPORT_DIR = ROOT / "reports"
OUTPUT_FILE = REPORT_DIR / "data_inventory.json"

MATCHES_FILE = RAW_DIR / "charting-m-matches.csv"
POINT_FILES = sorted(RAW_DIR.glob("charting-m-points-*.csv"))


def main() -> None:
    if not MATCHES_FILE.exists():
        raise FileNotFoundError(f"No se encontró: {MATCHES_FILE}")

    if len(POINT_FILES) != 3:
        raise RuntimeError(
            f"Se esperaban 3 archivos de puntos y se encontraron {len(POINT_FILES)}"
        )

    matches = pd.read_csv(MATCHES_FILE, dtype=str)
    point_frames = []
    point_file_summary = {}

    expected_columns = None
    columns_consistent = True

    for path in POINT_FILES:
        frame = pd.read_csv(path, dtype=str)

        columns = frame.columns.tolist()
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            columns_consistent = False

        point_file_summary[path.name] = {
            "rows": int(len(frame)),
            "columns": columns,
            "unique_matches": int(frame["match_id"].nunique(dropna=True)),
            "missing_match_id": int(frame["match_id"].isna().sum()),
            "missing_point_number": int(frame["Pt"].isna().sum()),
            "missing_first_serve_sequence": int(frame["1st"].isna().sum()),
            "missing_second_serve_sequence": int(frame["2nd"].isna().sum()),
        }

        point_frames.append(frame)

    points = pd.concat(point_frames, ignore_index=True)

    point_keys = points[["match_id", "Pt"]]
    point_match_ids = set(points["match_id"].dropna())
    metadata_match_ids = set(matches["match_id"].dropna())

    parsed_dates = pd.to_datetime(
        matches["Date"],
        format="%Y%m%d",
        errors="coerce",
    )

    report = {
        "source": {
            "repository": (
                "https://github.com/JeffSackmann/"
                "tennis_MatchChartingProject"
            ),
            "commit": "2c59eef194967e688b69e73df344184a06322cd8",
        },
        "matches": {
            "rows": int(len(matches)),
            "columns": matches.columns.tolist(),
            "unique_match_ids": int(matches["match_id"].nunique(dropna=True)),
            "duplicate_match_ids": int(matches["match_id"].duplicated().sum()),
            "missing_match_ids": int(matches["match_id"].isna().sum()),
            "invalid_dates": int(parsed_dates.isna().sum()),
            "minimum_date": (
                parsed_dates.min().date().isoformat()
                if parsed_dates.notna().any()
                else None
            ),
            "maximum_date": (
                parsed_dates.max().date().isoformat()
                if parsed_dates.notna().any()
                else None
            ),
        },
        "points": {
            "total_rows": int(len(points)),
            "unique_matches": int(points["match_id"].nunique(dropna=True)),
            "duplicate_match_point_keys": int(
                point_keys.duplicated().sum()
            ),
            "columns_consistent_between_files": columns_consistent,
            "files": point_file_summary,
        },
        "relationship": {
            "point_matches_missing_from_metadata": int(
                len(point_match_ids - metadata_match_ids)
            ),
            "metadata_matches_without_points": int(
                len(metadata_match_ids - point_match_ids)
            ),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nInforme guardado en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()