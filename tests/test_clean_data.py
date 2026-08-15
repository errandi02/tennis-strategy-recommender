import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.clean_data import (
    apply_match_metadata_corrections,
    count_malformed_csv_rows,
    validate_clean_matches,
)
from src.data.quality_rules import (
    CONFLICTING_MATCH_IDS,
    MATCH_METADATA_CORRECTIONS,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_FILE = ROOT / "reports" / "cleaning_report.json"
CORRECTED_MATCH_ID = (
    "20240915-M-Davis_Cup_World_Group-RR-"
    "Tallon_Griekspoor-Flavio_Cobolli"
)


def load_processed_data():
    required_files = [
        PROCESSED_DIR / "matches_clean.parquet",
        PROCESSED_DIR / "points_enriched.parquet",
    ]
    if not all(path.exists() for path in required_files):
        pytest.skip("No están disponibles los Parquet procesados locales.")

    matches = pd.read_parquet(
        PROCESSED_DIR / "matches_clean.parquet"
    )
    points = pd.read_parquet(
        PROCESSED_DIR / "points_enriched.parquet"
    )
    return matches, points


def test_expected_number_of_conflicting_matches():
    assert len(CONFLICTING_MATCH_IDS) == 8


def test_metadata_corrections_only_include_approved_match():
    assert set(MATCH_METADATA_CORRECTIONS) == {CORRECTED_MATCH_ID}


def make_synthetic_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [CORRECTED_MATCH_ID, "unmodified-match"],
            "surface": ["Eva Asderaki-Moore", "Clay"],
            "umpire": ["3", "Known Umpire"],
            "best_of": ["1", "5"],
            "final_tiebreak": [None, "A"],
            "charted_by": [None, "Known Charter"],
        }
    )


def test_known_metadata_correction_is_applied():
    original = make_synthetic_matches()
    original_snapshot = original.copy(deep=True)

    corrected, corrected_match_count = apply_match_metadata_corrections(
        original
    )

    row = corrected.loc[corrected["match_id"].eq(CORRECTED_MATCH_ID)].iloc[0]
    assert row["surface"] == "Hard"
    assert row["umpire"] == "Eva Asderaki-Moore"
    assert row["best_of"] == "3"
    assert row["final_tiebreak"] == "1"
    assert pd.isna(row["charted_by"])
    assert corrected_match_count == 1
    pd.testing.assert_frame_equal(original, original_snapshot)
    assert corrected is not original


def test_unlisted_match_is_not_modified():
    original = make_synthetic_matches()
    original_snapshot = original.copy(deep=True)

    corrected, corrected_match_count = apply_match_metadata_corrections(
        original
    )

    pd.testing.assert_series_equal(
        corrected.loc[1],
        original.loc[1],
    )
    assert corrected_match_count == 1
    pd.testing.assert_frame_equal(original, original_snapshot)
    assert corrected is not original


def test_already_correct_metadata_is_not_counted_as_changed():
    original = pd.DataFrame(
        [{"match_id": CORRECTED_MATCH_ID, **MATCH_METADATA_CORRECTIONS[CORRECTED_MATCH_ID]}]
    )
    original_snapshot = original.copy(deep=True)

    corrected, corrected_match_count = apply_match_metadata_corrections(
        original
    )

    assert corrected_match_count == 0
    pd.testing.assert_frame_equal(original, original_snapshot)
    pd.testing.assert_frame_equal(corrected, original_snapshot)
    assert corrected is not original


def test_invalid_surface_raises_error():
    matches = make_synthetic_matches().iloc[[1]].copy()
    matches.loc[:, "surface"] = "Carpet"

    with pytest.raises(ValueError, match="Superficies no permitidas"):
        validate_clean_matches(matches)


def test_invalid_best_of_raises_error():
    matches = make_synthetic_matches().iloc[[1]].copy()
    matches.loc[:, "best_of"] = "1"

    with pytest.raises(ValueError, match="best_of no permitidos"):
        validate_clean_matches(matches)


def test_null_best_of_is_allowed():
    matches = make_synthetic_matches().iloc[[1]].copy()
    matches.loc[:, "best_of"] = None

    validate_clean_matches(matches)


def test_count_malformed_csv_rows_uses_physical_row_lengths(tmp_path):
    csv_path = tmp_path / "matches.csv"
    csv_path.write_text(
        "one,two,three\n1,2,3\n4,5\n6,7,8,9\n",
        encoding="utf-8",
    )

    assert count_malformed_csv_rows(csv_path) == 2


@pytest.mark.integration
def test_match_ids_are_unique():
    matches, _ = load_processed_data()

    assert matches["match_id"].notna().all()
    assert not matches["match_id"].duplicated().any()


@pytest.mark.integration
def test_match_dates_are_valid():
    matches, _ = load_processed_data()

    assert matches["date"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(matches["date"])


@pytest.mark.integration
def test_point_keys_are_unique():
    _, points = load_processed_data()

    assert points["match_id"].notna().all()
    assert points["point_number"].notna().all()
    assert not points.duplicated(
        ["match_id", "point_number"]
    ).any()


@pytest.mark.integration
def test_all_points_have_match_metadata():
    matches, points = load_processed_data()

    assert set(points["match_id"]).issubset(set(matches["match_id"]))
    assert points["player_1"].notna().all()
    assert points["player_2"].notna().all()


@pytest.mark.integration
def test_conflicting_matches_are_excluded():
    _, points = load_processed_data()

    remaining_ids = set(points["match_id"])
    assert remaining_ids.isdisjoint(CONFLICTING_MATCH_IDS)


@pytest.mark.integration
def test_corrected_match_metadata():
    matches, points = load_processed_data()

    match_row = matches.loc[matches["match_id"].eq(CORRECTED_MATCH_ID)].iloc[0]
    point_metadata = points.loc[
        points["match_id"].eq(CORRECTED_MATCH_ID),
        ["surface", "umpire", "best_of", "final_tiebreak", "charted_by"],
    ].drop_duplicates()

    assert match_row["surface"] == "Hard"
    assert match_row["umpire"] == "Eva Asderaki-Moore"
    assert match_row["best_of"] == "3"
    assert match_row["final_tiebreak"] == "1"
    assert pd.isna(match_row["charted_by"])
    assert len(point_metadata) == 1
    point_row = point_metadata.iloc[0]
    assert point_row["surface"] == match_row["surface"]
    assert point_row["umpire"] == match_row["umpire"]
    assert point_row["best_of"] == match_row["best_of"]
    assert point_row["final_tiebreak"] == match_row["final_tiebreak"]
    assert pd.isna(point_row["charted_by"])


@pytest.mark.integration
def test_expected_cleaning_totals():
    if not REPORT_FILE.exists():
        pytest.skip("No está disponible el informe local de limpieza.")

    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    assert report["raw_match_rows"] == 7566
    assert report["malformed_csv_rows_detected"] == 2
    assert report["invalid_match_rows_removed"] == 1
    assert report["metadata_rows_corrected"] == 1
    assert report["unknown_surface_matches"] == 0
    assert report["clean_match_rows"] == 7565
    assert report["original_point_rows"] == 1_284_276
    assert report["conflicting_matches_excluded"] == 8
    assert report["rows_from_conflicting_matches_excluded"] == 3493
    assert report["exact_duplicate_rows_removed"] == 375
    assert report["clean_point_rows"] == 1_280_408
    assert report["matches_without_points"] == 41
    assert report["enriched_point_rows"] == 1_280_408
    assert report["unique_enriched_matches"] == 7524
