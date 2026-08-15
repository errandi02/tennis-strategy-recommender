from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "match_charting_project"

matches = pd.read_csv(
    RAW_DIR / "charting-m-matches.csv",
    dtype=str,
)

point_frames = []

for path in sorted(RAW_DIR.glob("charting-m-points-*.csv")):
    frame = pd.read_csv(path, dtype=str)
    frame["source_file"] = path.name
    frame["source_row"] = range(2, len(frame) + 2)
    point_frames.append(frame)

points = pd.concat(point_frames, ignore_index=True)


print("\n=== FILAS DE PARTIDOS CON match_id DUPLICADO ===")
duplicate_matches = matches[
    matches.duplicated("match_id", keep=False)
].sort_values("match_id")

print(duplicate_matches.to_string(index=False))


print("\n=== FECHAS INVÁLIDAS ===")
parsed_dates = pd.to_datetime(
    matches["Date"],
    format="%Y%m%d",
    errors="coerce",
)

invalid_dates = matches.loc[
    parsed_dates.isna(),
    ["match_id", "Date", "Player 1", "Player 2", "Tournament"],
]

print(invalid_dates.to_string(index=False))


print("\n=== PARTIDOS CON METADATOS PERO SIN PUNTOS ===")
point_match_ids = set(points["match_id"])
matches_without_points = matches[
    ~matches["match_id"].isin(point_match_ids)
][
    ["match_id", "Date", "Player 1", "Player 2", "Tournament"]
].sort_values(["Date", "match_id"])

print(matches_without_points.to_string(index=False))
print(f"\nTotal: {len(matches_without_points)}")


print("\n=== RESUMEN DE CLAVES (match_id, Pt) DUPLICADAS ===")
duplicate_points = points[
    points.duplicated(["match_id", "Pt"], keep=False)
].sort_values(["match_id", "Pt", "source_file", "source_row"])

duplicate_key_summary = (
    duplicate_points
    .groupby(["match_id", "Pt"], dropna=False)
    .agg(
        rows=("Pt", "size"),
        source_files=("source_file", lambda values: ", ".join(sorted(set(values)))),
    )
    .reset_index()
)

print(duplicate_key_summary.head(30).to_string(index=False))
print(f"\nClaves duplicadas distintas: {len(duplicate_key_summary)}")
print(f"Filas implicadas: {len(duplicate_points)}")


print("\n=== PRIMERAS FILAS DUPLICADAS COMPLETAS ===")
columns_to_show = [
    "match_id",
    "Pt",
    "Svr",
    "1st",
    "2nd",
    "PtWinner",
    "source_file",
    "source_row",
]

print(duplicate_points[columns_to_show].head(40).to_string(index=False))


print("\n=== DUPLICADOS POR PARTIDO: EXACTOS Y CONFLICTIVOS ===")

data_columns = [
    column
    for column in points.columns
    if column not in {"source_file", "source_row"}
]

duplicate_analysis = []

for (match_id, point_number), group in duplicate_points.groupby(
    ["match_id", "Pt"],
    dropna=False,
):
    distinct_versions = len(group[data_columns].drop_duplicates())

    duplicate_analysis.append(
        {
            "match_id": match_id,
            "Pt": point_number,
            "rows": len(group),
            "distinct_versions": distinct_versions,
            "conflict": distinct_versions > 1,
        }
    )

duplicate_analysis = pd.DataFrame(duplicate_analysis)

summary_by_match = (
    duplicate_analysis
    .groupby("match_id")
    .agg(
        duplicated_point_keys=("Pt", "size"),
        exact_duplicate_keys=("conflict", lambda values: int((~values).sum())),
        conflicting_keys=("conflict", "sum"),
    )
    .reset_index()
    .sort_values(
        ["conflicting_keys", "duplicated_point_keys"],
        ascending=False,
    )
)

print(summary_by_match.to_string(index=False))

print("\n=== TOTALES ===")
print(f"Partidos afectados: {duplicate_analysis['match_id'].nunique()}")
print(
    "Claves duplicadas exactas: "
    f"{int((~duplicate_analysis['conflict']).sum())}"
)
print(
    "Claves con versiones conflictivas: "
    f"{int(duplicate_analysis['conflict'].sum())}"
)


print("\n=== DETALLE DE LOS PRIMEROS CONFLICTOS ===")

conflicting_keys = duplicate_analysis.loc[
    duplicate_analysis["conflict"],
    ["match_id", "Pt"],
]

conflicting_rows = duplicate_points.merge(
    conflicting_keys,
    on=["match_id", "Pt"],
    how="inner",
)

print(
    conflicting_rows[columns_to_show]
    .sort_values(["match_id", "Pt", "source_row"])
    .head(100)
    .to_string(index=False)
)

conflicting_match_ids = (
    summary_by_match.loc[
        summary_by_match["conflicting_keys"] > 0,
        "match_id",
    ]
    .sort_values()
    .tolist()
)

print("\n=== PARTIDOS QUE SE EXCLUIRÁN ===")
for match_id in conflicting_match_ids:
    print(match_id)