import json
from pathlib import Path

import pandas as pd

from src.data.quality_rules import CONFLICTING_MATCH_IDS


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_FILE = ROOT / "reports" / "cleaning_report.json"


def load_processed_data():
    matches = pd.read_parquet(
        PROCESSED_DIR / "matches_clean.parquet"
    )
    points = pd.read_parquet(
        PROCESSED_DIR / "points_enriched.parquet"
    )
    return matches, points


def test_expected_number_of_conflicting_matches():
    assert len(CONFLICTING_MATCH_IDS) == 8


def test_match_ids_are_unique():
    matches, _ = load_processed_data()

    assert matches["match_id"].notna().all()
    assert not matches["match_id"].duplicated().any()


def test_match_dates_are_valid():
    matches, _ = load_processed_data()

    assert matches["date"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(matches["date"])


def test_point_keys_are_unique():
    _, points = load_processed_data()

    assert points["match_id"].notna().all()
    assert points["point_number"].notna().all()
    assert not points.duplicated(
        ["match_id", "point_number"]
    ).any()


def test_all_points_have_match_metadata():
    matches, points = load_processed_data()

    assert set(points["match_id"]).issubset(set(matches["match_id"]))
    assert points["player_1"].notna().all()
    assert points["player_2"].notna().all()


def test_conflicting_matches_are_excluded():
    _, points = load_processed_data()

    remaining_ids = set(points["match_id"])
    assert remaining_ids.isdisjoint(CONFLICTING_MATCH_IDS)


def test_expected_cleaning_totals():
    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    assert report["raw_match_rows"] == 7566
    assert report["invalid_match_rows_removed"] == 1
    assert report["clean_match_rows"] == 7565
    assert report["original_point_rows"] == 1_284_276
    assert report["conflicting_matches_excluded"] == 8
    assert report["rows_from_conflicting_matches_excluded"] == 3493
    assert report["exact_duplicate_rows_removed"] == 375
    assert report["clean_point_rows"] == 1_280_408
    assert report["matches_without_points"] == 41
    assert report["enriched_point_rows"] == 1_280_408
    assert report["unique_enriched_matches"] == 7524