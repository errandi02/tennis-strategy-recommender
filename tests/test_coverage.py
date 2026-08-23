from pathlib import Path

import pandas as pd
import pytest

from src.analysis.coverage import (
    aggregate_player_coverage,
    analyze_coverage,
    expand_match_participants,
    reduce_to_matches,
    validate_coverage_points,
    validate_coverage_reconciliation,
    write_coverage_tables,
)


ROOT = Path(__file__).resolve().parents[1]
POINTS_FILE = ROOT / "data" / "processed" / "points_enriched.parquet"


def make_points() -> pd.DataFrame:
    rows = []
    groups = [
        ("A", "B", "Hard", 2020, 10, 2),
        ("C", "D", "Clay", 2021, 5, 3),
        ("E", "F", "Grass", 2022, 3, 4),
        ("G", "H", "Hard", 2023, 1, 5),
    ]
    match_sequence = 0
    for player_1, player_2, surface, year, match_count, point_count in groups:
        for match_in_group in range(match_count):
            match_sequence += 1
            match_id = f"m{match_sequence}"
            date = f"{year}-01-{match_in_group + 1:02d}"
            for point_number in range(1, point_count + 1):
                rows.append(
                    {
                        "match_id": match_id,
                        "point_number": point_number,
                        "player_1": player_1,
                        "player_2": player_2,
                        "date": date,
                        "surface": surface,
                    }
                )
    return pd.DataFrame(rows)


def expected_player_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", "Hard", 2020, 10, 20),
            ("B", "Hard", 2020, 10, 20),
            ("C", "Clay", 2021, 5, 15),
            ("D", "Clay", 2021, 5, 15),
            ("E", "Grass", 2022, 3, 12),
            ("F", "Grass", 2022, 3, 12),
            ("G", "Hard", 2023, 1, 5),
            ("H", "Hard", 2023, 1, 5),
        ],
        columns=["player", "surface", "year", "matches", "points"],
    )


def expected_surface_year() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Hard", 2020, 10, 20, 2),
            ("Clay", 2021, 5, 15, 2),
            ("Grass", 2022, 3, 12, 2),
            ("Hard", 2023, 1, 5, 2),
        ],
        columns=["surface", "year", "matches", "points", "distinct_players"],
    )


def test_missing_required_columns_raise_error():
    with pytest.raises(ValueError, match="columnas obligatorias"):
        validate_coverage_points(make_points().drop(columns="surface"))


def test_duplicate_point_key_raises_error():
    duplicate = pd.concat([make_points(), make_points().iloc[[0]]])
    with pytest.raises(ValueError, match="no es unica"):
        validate_coverage_points(duplicate)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("date", "not-a-date", "fechas"),
        ("player_1", None, "jugadores"),
        ("surface", "Carpet", "Superficies"),
    ],
)
def test_invalid_required_values_raise_error(column, value, message):
    points = make_points()
    points.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_coverage_points(points)


def test_inconsistent_metadata_and_equal_players_raise_errors():
    inconsistent = make_points()
    inconsistent.loc[0, "surface"] = "Grass"
    with pytest.raises(ValueError, match="Metadatos no constantes"):
        validate_coverage_points(inconsistent)

    equal_players = make_points()
    equal_players.loc[equal_players["match_id"].eq("m1"), "player_2"] = "A"
    with pytest.raises(ValueError, match="dos jugadores distintos"):
        validate_coverage_points(equal_players)


def test_match_reduction_and_exact_aggregations():
    matches, participants, coverage, surface_year, _ = analyze_coverage(
        make_points()
    )

    assert len(matches) == 19
    assert matches["point_count"].sum() == 52
    assert len(participants) == 38
    assert participants.groupby("match_id").size().eq(2).all()
    pd.testing.assert_frame_equal(
        coverage,
        expected_player_coverage(),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        surface_year,
        expected_surface_year(),
        check_dtype=False,
    )
    assert surface_year["points"].sum() == 52
    assert coverage["points"].sum() == 104


def test_summary_cuts_percentiles_and_reconciliation():
    _, _, _, _, summary = analyze_coverage(make_points())

    assert summary["point_rows"] == 52
    assert summary["unique_matches"] == 19
    assert summary["distinct_players"] == 8
    assert summary["first_year"] == 2020
    assert summary["last_year"] == 2023
    assert summary["descriptive_cuts_are_model_thresholds"] is False

    cuts = summary["descriptive_match_cuts"]
    assert cuts["unit"] == "player_surface_year_combination"
    assert cuts["proportion_scale"] == "0_to_1"
    assert cuts["denominator"] == 8
    assert cuts["cuts"] == {
        "at_least_1_matches": {"numerator": 8, "proportion": 1.0},
        "at_least_3_matches": {"numerator": 6, "proportion": 0.75},
        "at_least_5_matches": {"numerator": 4, "proportion": 0.5},
        "at_least_10_matches": {"numerator": 2, "proportion": 0.25},
    }
    assert summary["matches_per_combination"] == {
        "minimum": 1.0,
        "mean": 4.75,
        "median": 4.0,
        "p25": 2.5,
        "p50": 4.0,
        "p75": 6.25,
        "p90": 10.0,
        "p95": 10.0,
        "p99": 10.0,
        "maximum": 10.0,
    }
    assert summary["reconciliation"] == {
        "expected_participant_rows": 38,
        "actual_participant_rows": 38,
        "expected_coverage_points": 104,
        "actual_coverage_points": 104,
    }


def test_reconciliation_rejects_incorrect_participants():
    matches = reduce_to_matches(make_points())
    valid_participants = expand_match_participants(matches)
    missing_participant = valid_participants.iloc[1:].copy()
    missing_coverage = aggregate_player_coverage(missing_participant)

    with pytest.raises(ValueError, match="exactamente dos participantes"):
        validate_coverage_reconciliation(
            matches,
            missing_participant,
            missing_coverage,
        )

    extra_match = valid_participants.iloc[:2].copy()
    extra_match.loc[:, "match_id"] = "unexpected-match"
    extra_participants = pd.concat(
        [valid_participants, extra_match],
        ignore_index=True,
    )
    extra_coverage = aggregate_player_coverage(extra_participants)

    with pytest.raises(ValueError, match="exactamente dos participantes"):
        validate_coverage_reconciliation(
            matches,
            extra_participants,
            extra_coverage,
        )


def test_reconciliation_rejects_incorrect_points():
    matches = reduce_to_matches(make_points())
    participants = expand_match_participants(matches)
    coverage = aggregate_player_coverage(participants)
    coverage.loc[0, "points"] += 1

    with pytest.raises(ValueError, match="puntos de cobertura no reconcilia"):
        validate_coverage_reconciliation(matches, participants, coverage)


def test_csv_round_trip_has_no_index(tmp_path):
    _, _, coverage, surface_year, _ = analyze_coverage(make_points())

    write_coverage_tables(coverage, surface_year, tmp_path)
    loaded_coverage = pd.read_csv(
        tmp_path / "coverage_by_player_surface_year.csv"
    )
    loaded_surface_year = pd.read_csv(
        tmp_path / "coverage_by_surface_year.csv"
    )

    assert not any(column.startswith("Unnamed") for column in loaded_coverage)
    assert not any(column.startswith("Unnamed") for column in loaded_surface_year)
    assert list(loaded_coverage.columns) == list(expected_player_coverage().columns)
    assert list(loaded_surface_year.columns) == list(expected_surface_year().columns)
    pd.testing.assert_frame_equal(
        loaded_coverage,
        expected_player_coverage(),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        loaded_surface_year,
        expected_surface_year(),
        check_dtype=False,
    )


@pytest.mark.integration
def test_real_points_coverage_reconciles():
    if not POINTS_FILE.exists():
        pytest.skip("No esta disponible points_enriched.parquet.")

    points = pd.read_parquet(POINTS_FILE)
    matches, participants, coverage, _, summary = analyze_coverage(points)

    assert len(points) == 1_280_408
    assert len(matches) == 7_524
    assert len(participants) == 15_048
    assert len(coverage) == 5_114
    assert coverage["points"].sum() == 2_560_816
    assert summary["point_rows"] == 1_280_408
    assert summary["unique_matches"] == 7_524
    assert summary["surface_distribution"] == {
        "Clay": {
            "matches": 1_872,
            "points": 311_240,
            "distinct_players": 587,
        },
        "Grass": {
            "matches": 829,
            "points": 168_079,
            "distinct_players": 359,
        },
        "Hard": {
            "matches": 4_823,
            "points": 801_089,
            "distinct_players": 753,
        },
    }
    cuts = summary["descriptive_match_cuts"]
    assert cuts["denominator"] == 5_114
    assert {
        key: value["numerator"] for key, value in cuts["cuts"].items()
    } == {
        "at_least_1_matches": 5_114,
        "at_least_3_matches": 1_493,
        "at_least_5_matches": 822,
        "at_least_10_matches": 297,
    }
