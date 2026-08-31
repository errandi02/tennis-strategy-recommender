from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.analysis.evidence_policy_validation as module
from src.analysis.evidence_policy_validation import (
    BY_FOLD_COLUMNS,
    CANDIDATE_COLUMNS,
    PARETO_COLUMNS,
    SCOPE_POLICIES,
    EvidencePolicyValidationResult,
    analyze_from_features,
    apply_policy,
    build_by_fold,
    build_candidate_grid,
    build_candidates,
    build_pareto,
    construct_sealed_target_features,
    independent_wilson_width,
    not_available_payloads,
    prepare_policy_population,
    seal_target_population,
    select_validation_features,
    serialize_artifacts,
    validate_result,
    validate_serialized_artifacts,
    write_artifacts,
)
from src.analysis.player_opponent_direction_features import FEATURE_COLUMNS, Z_975


def _manual_wilson(wins: int, points: int) -> tuple[float | None, float | None]:
    if points == 0:
        return None, None
    proportion = wins / points
    denominator = 1 + Z_975**2 / points
    centre = (proportion + Z_975**2 / (2 * points)) / denominator
    half = Z_975 * np.sqrt(
        proportion * (1 - proportion) / points + Z_975**2 / (4 * points**2)
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _synthetic_features() -> pd.DataFrame:
    rows = []
    dates = [pd.Timestamp(f"{year}-06-15") for year in range(2020, 2024)]
    surfaces = ["Hard", "Clay", "Grass", "Hard"]
    global_points = [25, 50, 100, 150]
    global_matches = [3, 5, 10, 15]
    surface_points = [25, 40, 60, 110]
    surface_matches = [3, 4, 6, 11]
    for fold_index, (date, surface) in enumerate(zip(dates, surfaces)):
        match_id = f"v{date.year}"
        for target_player, opponent, player_offset in (("A", "B", 0), ("B", "A", 1)):
            for direction_index, direction in enumerate(("wide", "body", "T")):
                for scope in ("global", "surface"):
                    points = (global_points if scope == "global" else surface_points)[fold_index]
                    matches = (global_matches if scope == "global" else surface_matches)[fold_index]
                    server_wins = int(round(points * (0.42 + 0.08 * player_offset + 0.02 * direction_index)))
                    opponent_wins = int(round(points * (0.48 + 0.04 * player_offset - 0.01 * direction_index)))
                    population_wins = int(round(points * 0.5))
                    server_low, server_high = _manual_wilson(server_wins, points)
                    opponent_low, opponent_high = _manual_wilson(opponent_wins, points)
                    population_low, population_high = _manual_wilson(population_wins, points)
                    row = {column: None for column in FEATURE_COLUMNS}
                    row.update(
                        {
                            "target_match_id": match_id,
                            "target_date": date,
                            "target_player": target_player,
                            "opponent": opponent,
                            "target_surface": surface,
                            "derived_period": "2020s",
                            "direction": direction,
                            "scope": scope,
                            "server_direction_matches": matches,
                            "server_direction_points": points,
                            "server_direction_wins": server_wins,
                            "server_direction_raw_rate": server_wins / points,
                            "server_direction_wilson_low": server_low,
                            "server_direction_wilson_high": server_high,
                            "server_direction_share": 1 / 3,
                            "server_scope_points": points * 3,
                            "server_scope_wins": server_wins * 3,
                            "server_scope_raw_rate": server_wins / points,
                            "server_direction_first_history_date": pd.Timestamp("2010-01-01"),
                            "server_direction_last_history_date": date - pd.Timedelta(days=1),
                            "opponent_direction_matches": matches,
                            "opponent_direction_points": points,
                            "opponent_allowed_server_wins": opponent_wins,
                            "opponent_return_wins": points - opponent_wins,
                            "opponent_allowed_server_raw_rate": opponent_wins / points,
                            "opponent_allowed_server_wilson_low": opponent_low,
                            "opponent_allowed_server_wilson_high": opponent_high,
                            "opponent_direction_share": 1 / 3,
                            "opponent_scope_points": points * 3,
                            "opponent_scope_allowed_server_wins": opponent_wins * 3,
                            "opponent_scope_allowed_server_raw_rate": opponent_wins / points,
                            "opponent_direction_first_history_date": pd.Timestamp("2010-01-01"),
                            "opponent_direction_last_history_date": date - pd.Timedelta(days=1),
                            "population_direction_points": points,
                            "population_direction_server_wins": population_wins,
                            "population_direction_raw_rate": population_wins / points,
                            "population_direction_wilson_low": population_low,
                            "population_direction_wilson_high": population_high,
                            "population_direction_first_history_date": pd.Timestamp("2010-01-01"),
                            "population_direction_last_history_date": date - pd.Timedelta(days=1),
                            "server_excess_vs_population": server_wins / points - population_wins / points,
                            "opponent_allowed_excess_vs_population": opponent_wins / points - population_wins / points,
                            "server_opponent_gap": server_wins / points - opponent_wins / points,
                            "server_evidence_state": "observed",
                            "opponent_evidence_state": "observed",
                            "pair_evidence_state": "both_observed",
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def validation_features() -> pd.DataFrame:
    return select_validation_features(_synthetic_features(), set())


@pytest.fixture(scope="module")
def population(validation_features) -> pd.DataFrame:
    return prepare_policy_population(validation_features)


@pytest.fixture(scope="module")
def synthetic_result(validation_features) -> EvidencePolicyValidationResult:
    return analyze_from_features(
        validation_features,
        source_rows=100,
        source_matches=4,
        source_players=2,
        eligible_points=90,
        server_wins=45,
    )


def _policy(min_points=25, min_matches=3, scope_policy="global_only") -> dict:
    return {
        "policy_id": f"p{min_points:03d}_m{min_matches:02d}_{scope_policy}",
        "min_points": min_points,
        "min_matches": min_matches,
        "scope_policy": scope_policy,
    }


def test_candidate_grid_is_exact_cartesian_product_and_ordered():
    grid = build_candidate_grid()
    assert grid.columns.tolist() == ["policy_id", "min_points", "min_matches", "scope_policy"]
    assert len(grid) == 27
    assert grid.iloc[0].to_dict() == _policy()
    assert grid.iloc[-1].to_dict() == _policy(100, 10, "surface_then_global")
    assert set(map(tuple, grid[["min_points", "min_matches", "scope_policy"]].to_numpy())) == {
        (points, matches, scope)
        for points in (25, 50, 100)
        for matches in (3, 5, 10)
        for scope in SCOPE_POLICIES
    }
    assert grid["policy_id"].is_unique


@pytest.mark.parametrize(
    "bad_date,error",
    [("2019-12-31", "Folds de validacion"), ("2024-01-01", "posteriores a 2023")],
)
def test_validation_requires_exact_four_folds(bad_date, error):
    features = _synthetic_features()
    features.loc[features["target_match_id"].eq("v2020"), "target_date"] = pd.Timestamp(bad_date)
    with pytest.raises(ValueError, match=error):
        select_validation_features(features, set())


def test_sealed_test_barrier_rejects_forbidden_match_id():
    with pytest.raises(ValueError, match="Contaminacion"):
        select_validation_features(_synthetic_features(), {"v2022"})


def test_sealed_test_barrier_rejects_nonzero_evaluation_runs():
    with pytest.raises(ValueError, match="cero"):
        select_validation_features(_synthetic_features(), set(), evaluation_runs=1)


def test_intraday_time_components_are_rejected_before_policy_aggregation():
    features = _synthetic_features()
    features.loc[0, "target_date"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="normalizadas"):
        select_validation_features(features, set())


def test_history_must_be_strictly_before_target_day(validation_features):
    features = validation_features.copy(deep=True)
    features.loc[0, "server_direction_last_history_date"] = features.loc[0, "target_date"]
    with pytest.raises(ValueError, match="estrictamente anterior"):
        prepare_policy_population(features)


def _target_and_history_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = []
    eligible = []
    for number, year in enumerate(range(2019, 2025), start=1):
        match_id = f"m{year}"
        day = pd.Timestamp(f"{year}-06-15")
        targets.append([match_id, day, "Hard", "A", "B"])
        eligible.append([match_id, number, day, "Hard", "A", "B", True, "wide"])
    return (
        pd.DataFrame(targets, columns=["match_id", "date", "surface", "player_1", "player_2"]),
        pd.DataFrame(
            eligible,
            columns=[
                "match_id", "point_number", "date", "surface", "server_player",
                "returner_player", "server_won_point", "direction",
            ],
        ),
    )


def test_physical_seal_excludes_test_before_snapshot_and_feature_builders(monkeypatch):
    targets, eligible = _target_and_history_fixture()
    original_snapshots = module.build_historical_snapshots
    original_features = module.build_direction_features
    observed = {}

    def snapshot_spy(received_targets, received_history):
        observed["snapshot_ids"] = set(received_targets["match_id"])
        observed["snapshot_max_date"] = received_targets["date"].max()
        assert received_history["date"].max() <= pd.Timestamp("2023-12-31")
        return original_snapshots(received_targets, received_history)

    def feature_spy(received_snapshots, received_history):
        observed["feature_ids"] = set(received_snapshots["target_match_id"])
        observed["feature_max_date"] = received_snapshots["target_date"].max()
        return original_features(received_snapshots, received_history)

    monkeypatch.setattr(module, "build_historical_snapshots", snapshot_spy)
    monkeypatch.setattr(module, "build_direction_features", feature_spy)
    snapshots, features, audit, test_ids = construct_sealed_target_features(targets, eligible)
    assert test_ids == {"m2024"}
    assert "m2024" not in observed["snapshot_ids"] | observed["feature_ids"]
    assert observed["snapshot_max_date"] == observed["feature_max_date"] == pd.Timestamp("2023-06-15")
    assert len(snapshots) == 5 * 2
    assert len(features) == 5 * 2 * 3 * 2
    assert audit == {
        "source_target_matches": 6,
        "constructed_target_matches": 5,
        "test_target_matches_excluded_before_construction": 1,
        "test_target_matches_constructed": 0,
        "test_target_rows_constructed": 0,
        "test_feature_rows_constructed": 0,
        "constructed_snapshot_rows": 10,
        "constructed_feature_rows": 60,
    }


def test_added_2024_feature_row_is_rejected_not_silently_discarded():
    features = _synthetic_features()
    future = features.iloc[[0]].copy(deep=True)
    future["target_match_id"] = "test_2024"
    future["target_date"] = pd.Timestamp("2024-01-01")
    contaminated = pd.concat([features, future], ignore_index=True)
    with pytest.raises(ValueError, match="posteriores a 2023"):
        select_validation_features(contaminated)


def test_direct_analysis_route_rejects_post_2023_target(validation_features):
    contaminated = validation_features.copy(deep=True)
    contaminated.loc[0, "target_date"] = pd.Timestamp("2024-01-01")
    contaminated.loc[0, "fold"] = 2024
    with pytest.raises(ValueError, match="solo admite targets de validacion"):
        analyze_from_features(
            contaminated,
            source_rows=4,
            source_matches=4,
            source_players=2,
            eligible_points=24,
            server_wins=12,
        )


def test_global_only_uses_only_global_and_inclusive_thresholds(population):
    evaluated = apply_policy(population, _policy(50, 5, "global_only"))
    assert not evaluated.loc[evaluated["fold"].eq(2020), "eligible"].any()
    assert evaluated.loc[evaluated["fold"].ge(2021), "eligible"].all()
    assert set(evaluated.loc[evaluated["eligible"], "selected_scope"]) == {"global"}


def test_global_only_never_accesses_surface_columns(population):
    forbidden = [
        column for column in population.columns
        if column.endswith("_surface") and column != "target_surface"
    ]
    evaluated = apply_policy(population.drop(columns=forbidden), _policy(scope_policy="global_only"))
    assert set(evaluated.loc[evaluated["eligible"], "selected_scope"]) == {"global"}


def test_surface_only_has_no_fallback(population):
    evaluated = apply_policy(population, _policy(50, 5, "surface_only"))
    assert not evaluated.loc[evaluated["fold"].isin([2020, 2021]), "eligible"].any()
    assert evaluated.loc[evaluated["fold"].isin([2022, 2023]), "eligible"].all()
    assert set(evaluated.loc[evaluated["eligible"], "selected_scope"]) == {"surface"}


def test_surface_only_never_accesses_global_columns(population):
    forbidden = [column for column in population.columns if column.endswith("_global")]
    evaluated = apply_policy(population.drop(columns=forbidden), _policy(scope_policy="surface_only"))
    assert set(evaluated.loc[evaluated["eligible"], "selected_scope"]) == {"surface"}


def test_surface_then_global_fallback_is_explicit_and_atomic(population):
    evaluated = apply_policy(population, _policy(50, 5, "surface_then_global"))
    assert not evaluated.loc[evaluated["fold"].eq(2020), "eligible"].any()
    assert set(evaluated.loc[evaluated["fold"].eq(2021), "selected_scope"]) == {"global"}
    assert set(evaluated.loc[evaluated["fold"].isin([2022, 2023]), "selected_scope"]) == {"surface"}
    assert not evaluated.loc[evaluated["eligible"], "selected_scope"].isna().any()


@pytest.mark.parametrize(
    "server_points,server_matches,opponent_points,opponent_matches",
    [(49, 5, 50, 5), (50, 4, 50, 5), (50, 5, 49, 5), (50, 5, 50, 4)],
)
def test_both_roles_must_pass_points_and_matches(
    population, server_points, server_matches, opponent_points, opponent_matches
):
    one = population.iloc[[6]].copy(deep=True)
    one["server_direction_points_global"] = server_points
    one["server_direction_matches_global"] = server_matches
    one["opponent_direction_points_global"] = opponent_points
    one["opponent_direction_matches_global"] = opponent_matches
    evaluated = apply_policy(one, _policy(50, 5, "global_only"))
    assert not evaluated.iloc[0]["eligible"]
    assert evaluated.iloc[0]["selected_scope"] is None


def test_mixed_role_scopes_are_never_combined(population):
    one = population.iloc[[6]].copy(deep=True)
    one["server_direction_points_surface"] = 100
    one["server_direction_matches_surface"] = 10
    one["opponent_direction_points_surface"] = 0
    one["opponent_direction_matches_surface"] = 0
    one["server_direction_points_global"] = 0
    one["server_direction_matches_global"] = 0
    one["opponent_direction_points_global"] = 100
    one["opponent_direction_matches_global"] = 10
    evaluated = apply_policy(one, _policy(50, 5, "surface_then_global"))
    assert not evaluated.iloc[0]["eligible"]
    assert evaluated.iloc[0]["selected_scope"] is None


@pytest.mark.parametrize("threshold", [25, 50, 100])
def test_each_point_threshold_is_inclusive_and_one_less_fails(population, threshold):
    one = population.iloc[[0]].copy(deep=True)
    for role in ("server", "opponent"):
        one[f"{role}_direction_points_global"] = threshold
        one[f"{role}_direction_matches_global"] = 10
    assert apply_policy(one, _policy(threshold, 10)).iloc[0].eligible
    one["server_direction_points_global"] = threshold - 1
    assert not apply_policy(one, _policy(threshold, 10)).iloc[0].eligible


@pytest.mark.parametrize("threshold", [3, 5, 10])
def test_each_match_threshold_is_inclusive_and_one_less_fails(population, threshold):
    one = population.iloc[[0]].copy(deep=True)
    for role in ("server", "opponent"):
        one[f"{role}_direction_points_global"] = 100
        one[f"{role}_direction_matches_global"] = threshold
    assert apply_policy(one, _policy(100, threshold)).iloc[0].eligible
    one["opponent_direction_matches_global"] = threshold - 1
    assert not apply_policy(one, _policy(100, threshold)).iloc[0].eligible


def test_complete_match_requires_all_six_orientation_direction_rows(population):
    evaluated = apply_policy(population, _policy())
    by_fold = build_by_fold([evaluated])
    overall = by_fold.query("fold == 2020 and direction == 'wide' and role == 'server' and stratum_type == 'overall'").iloc[0]
    assert overall.target_matches == 1
    assert overall.target_orientations == 2
    assert overall.eligible_matches == 1
    assert overall.complete_matches == 1
    broken = evaluated.copy(deep=True)
    broken.loc[
        broken["fold"].eq(2020) & broken["target_player"].eq("A") & broken["direction"].eq("T"),
        ["eligible", "selected_scope"],
    ] = [False, None]
    broken_table = build_by_fold([broken])
    broken_row = broken_table.query("fold == 2020 and direction == 'wide' and role == 'server' and stratum_type == 'overall'").iloc[0]
    assert broken_row.complete_matches == 0


def test_denominators_and_coverages_are_manual(population):
    evaluated = apply_policy(population, _policy(50, 5, "global_only"))
    table = build_by_fold([evaluated])
    row = table.query("fold == 2021 and direction == 'body' and role == 'opponent' and stratum_type == 'overall'").iloc[0]
    assert (row.target_matches, row.target_orientations) == (1, 2)
    assert (row.eligible_orientations, row.eligible_matches, row.complete_matches) == (2, 1, 1)
    assert (row.coverage_orientation, row.coverage_match, row.coverage_complete_match) == (1.0, 1.0, 1.0)
    row_2020 = table.query("fold == 2020 and direction == 'body' and role == 'opponent' and stratum_type == 'overall'").iloc[0]
    assert row_2020.coverage_orientation == 0.0
    assert row_2020.abstained_orientations == 2


@pytest.mark.parametrize("wins,points", [(0, 25), (10, 25), (25, 25), (42, 100)])
def test_wilson_width_matches_independent_numeric_formula(wins, points):
    low, high = _manual_wilson(wins, points)
    assert independent_wilson_width(wins, points) == pytest.approx(high - low, abs=1e-15)


def test_wilson_zero_denominator_is_null():
    assert independent_wilson_width(0, 0) is None


def test_stability_uses_only_comparable_players_and_known_transitions(population):
    evaluated = apply_policy(population, _policy())
    table = build_by_fold([evaluated])
    first = table.query("fold == 2020 and direction == 'wide' and role == 'server' and stratum_type == 'overall'").iloc[0]
    second = table.query("fold == 2021 and direction == 'wide' and role == 'server' and stratum_type == 'overall'").iloc[0]
    assert first.spearman_reason_code == "no_previous_fold"
    assert second.stability_transition == "2020_to_2021"
    assert second.comparable_players == 2
    assert second.comparable_rows == 4
    assert 0 <= second.within_005_proportion <= 1


def test_noncomparable_players_are_not_imputed(population):
    evaluated = apply_policy(population, _policy())
    evaluated.loc[evaluated["fold"].eq(2020), "target_player"] = [f"old_{i}" for i in range(sum(evaluated["fold"].eq(2020)))]
    table = build_by_fold([evaluated])
    row = table.query("fold == 2021 and direction == 'wide' and role == 'server' and stratum_type == 'overall'").iloc[0]
    assert row.comparable_players == 0
    assert pd.isna(row.absolute_change_p90)
    assert row.spearman_reason_code == "no_comparable_players"


def test_undefined_spearman_is_null_with_reason(population):
    evaluated = apply_policy(population, _policy())
    evaluated["server_selected_rate"] = 0.5
    table = build_by_fold([evaluated])
    row = table.query("fold == 2021 and direction == 'wide' and role == 'server' and stratum_type == 'overall'").iloc[0]
    assert pd.isna(row.spearman_correlation)
    assert row.spearman_reason_code == "both_series_constant"


@pytest.mark.parametrize(
    "server,opponent,reason",
    [
        ([0.4], [0.5], "insufficient_comparable_pairs"),
        ([0.4, 0.4], [0.5, 0.6], "constant_server_series"),
        ([0.4, 0.5], [0.6, 0.6], "constant_opponent_series"),
        ([0.4, 0.4], [0.6, 0.6], "both_series_constant"),
    ],
)
def test_spearman_degenerate_inputs_are_null_without_warning(server, opponent, reason):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlation, observed_reason, finite = module._safe_spearman(
            pd.Series(server), pd.Series(opponent)
        )
    assert correlation is None
    assert observed_reason == reason
    assert finite.all()
    assert caught == []


def test_spearman_valid_nonconstant_case_is_computed_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlation, reason, finite = module._safe_spearman(
            pd.Series([0.1, 0.2, 0.3]), pd.Series([0.6, 0.5, 0.4])
        )
    assert correlation == pytest.approx(-1.0)
    assert reason is None
    assert finite.all()
    assert caught == []


def test_surface_metrics_and_grass_stratum_are_explicit(synthetic_result):
    table = synthetic_result.by_fold
    grass = table.query("surface == 'Grass' and stratum_type == 'surface'")
    assert not grass.empty
    assert set(table.loc[table.stratum_type.eq("surface"), "surface"]) == {"Hard", "Clay", "Grass"}
    assert "grass_is_lowest_coverage_for_all_candidates" in synthetic_result.summary["surface_summary"]


def _candidate_frame(rows):
    frame = pd.DataFrame(rows)
    for column in CANDIDATE_COLUMNS:
        if column not in frame:
            frame[column] = True if column == "scope_coherence_reconciled" else 0.0
    return frame[CANDIDATE_COLUMNS]


def test_pareto_dominance_and_equality_without_dominance():
    candidates = _candidate_frame(
        [
            {**_policy(), "worst_fold_complete_match_coverage": .7, "worst_fold_p90_wilson_width": .2, "worst_fold_p90_absolute_change": .1},
            {**_policy(50), "worst_fold_complete_match_coverage": .6, "worst_fold_p90_wilson_width": .3, "worst_fold_p90_absolute_change": .2},
            {**_policy(100), "worst_fold_complete_match_coverage": .7, "worst_fold_p90_wilson_width": .2, "worst_fold_p90_absolute_change": .1},
        ]
    )
    pareto = build_pareto(candidates)
    assert pareto.loc[pareto.policy_id.eq(_policy(50)["policy_id"]), "pareto_status"].item() == "dominated"
    assert pareto.loc[pareto.policy_id.eq(_policy()["policy_id"]), "pareto_status"].item() == "non_dominated"
    assert pareto.loc[pareto.policy_id.eq(_policy(100)["policy_id"]), "pareto_status"].item() == "non_dominated"


def test_pareto_not_comparable_and_unrounded_values():
    candidates = _candidate_frame(
        [
            {**_policy(), "worst_fold_complete_match_coverage": .7000000000001, "worst_fold_p90_wilson_width": .2, "worst_fold_p90_absolute_change": .1},
            {**_policy(50), "worst_fold_complete_match_coverage": .7, "worst_fold_p90_wilson_width": .2, "worst_fold_p90_absolute_change": .1},
            {**_policy(100), "worst_fold_complete_match_coverage": np.nan, "worst_fold_p90_wilson_width": .2, "worst_fold_p90_absolute_change": .1},
        ]
    )
    pareto = build_pareto(candidates)
    second = pareto.loc[pareto.policy_id.eq(_policy(50)["policy_id"])].iloc[0]
    assert second.pareto_status == "dominated"
    assert second.dominated_by == _policy()["policy_id"]
    third = pareto.loc[pareto.policy_id.eq(_policy(100)["policy_id"])].iloc[0]
    assert third.pareto_status == "not_comparable"
    assert third.reason_code == "undefined_pareto_axis"


def test_full_synthetic_candidate_metrics_are_known(synthetic_result):
    candidate = synthetic_result.candidates.loc[
        synthetic_result.candidates.policy_id.eq(_policy(50, 5, "global_only")["policy_id"])
    ].iloc[0]
    assert candidate.validation_matches == 4
    assert candidate.orientation_direction_denominator == 24
    assert candidate.eligible_orientation_directions == 18
    assert candidate.coverage_orientation == .75
    assert candidate.complete_matches == 3
    assert candidate.coverage_complete_match == .75
    assert candidate.wide_coverage == .75
    assert candidate.body_coverage == .75
    assert candidate.T_coverage == .75


def test_summary_seals_test_and_does_not_select_winner(synthetic_result):
    sealed = synthetic_result.summary["sealed_test_contract"]
    assert sealed["test_status"] == "sealed"
    assert sealed["test_rows_evaluated"] == 0
    assert sealed["test_matches_evaluated"] == 0
    assert sealed["used_for_method_selection"] is False
    assert sealed["test_evaluation_runs"] == 0
    assert synthetic_result.summary["pareto_summary"]["automatic_selection_performed"] is False
    assert "test_rows_read" not in json.dumps(synthetic_result.summary)
    construction = synthetic_result.summary["target_construction_contract"]
    assert construction["constructed_target_matches"] == 4
    assert construction["constructed_snapshot_rows"] == 8
    assert construction["constructed_feature_rows"] == 48
    assert construction["test_target_rows_constructed"] == 0
    assert construction["test_target_matches_constructed"] == 0
    assert construction["test_feature_rows_constructed"] == 0


@pytest.mark.parametrize("table_name", ["candidates", "by_fold", "pareto"])
def test_post_validation_table_mutation_is_detected(synthetic_result, table_name):
    altered = getattr(synthetic_result, table_name).copy(deep=True)
    numeric = altered.select_dtypes(include=[np.number]).columns[0]
    altered.loc[0, numeric] = altered.loc[0, numeric] + 1
    mutated = replace(synthetic_result, **{table_name: altered})
    with pytest.raises(ValueError):
        validate_result(mutated)


def test_summary_pareto_wilson_coverage_and_selected_scope_mutations_are_detected(synthetic_result):
    for path in ("sealed", "pareto", "coverage", "wilson", "constructed_snapshots", "constructed_features"):
        summary = json.loads(json.dumps(synthetic_result.summary))
        if path == "sealed":
            summary["sealed_test_contract"]["test_evaluation_runs"] = 1
        elif path == "pareto":
            summary["pareto_summary"]["automatic_selection_performed"] = True
        elif path == "coverage":
            summary["fold_reconciliations"]["fold_sum"] += 1
        elif path == "wilson":
            summary["uncertainty_contract"]["z"] = 1.0
        elif path == "constructed_snapshots":
            summary["target_construction_contract"]["constructed_snapshot_rows"] += 1
        else:
            summary["target_construction_contract"]["constructed_feature_rows"] += 1
        with pytest.raises(ValueError):
            validate_result(replace(synthetic_result, summary=summary))


@pytest.mark.parametrize(
    "field",
    ["test_target_rows_constructed", "test_target_matches_constructed", "test_feature_rows_constructed"],
)
def test_each_constructed_test_field_must_remain_zero(synthetic_result, field):
    summary = json.loads(json.dumps(synthetic_result.summary))
    summary["target_construction_contract"][field] = 1
    with pytest.raises(ValueError, match="test"):
        validate_result(replace(synthetic_result, summary=summary))


def test_ambiguous_legacy_test_rows_read_field_is_rejected(synthetic_result):
    summary = json.loads(json.dumps(synthetic_result.summary))
    summary["sealed_test_contract"]["test_rows_read"] = 0
    with pytest.raises(ValueError, match="sellado|ambiguo"):
        validate_result(replace(synthetic_result, summary=summary))


def test_nan_and_infinity_are_rejected(synthetic_result):
    candidates = synthetic_result.candidates.copy(deep=True)
    candidates.loc[0, "coverage_orientation"] = np.inf
    with pytest.raises(ValueError, match=r"infinito|\[0,1\]"):
        validate_result(replace(synthetic_result, candidates=candidates))


def test_serialization_is_deterministic_utf8_index_free_and_exact(synthetic_result):
    first = serialize_artifacts(synthetic_result)
    second = serialize_artifacts(synthetic_result)
    assert first == second
    assert json.loads(first[0].decode("utf-8"))["analysis_status"] == "available"
    for payload, columns in zip(first[1:], (CANDIDATE_COLUMNS, BY_FOLD_COLUMNS, PARETO_COLUMNS)):
        loaded = pd.read_csv(pd.io.common.BytesIO(payload))
        assert loaded.columns.tolist() == columns
        assert not any(column.startswith("Unnamed") for column in loaded.columns)


def test_serialized_payload_manipulation_is_rejected(synthetic_result):
    payloads = list(serialize_artifacts(synthetic_result))
    payloads[2] = payloads[2].replace(b"global_only", b"global_BAD", 1)
    with pytest.raises(ValueError):
        validate_serialized_artifacts(synthetic_result, *payloads)


@pytest.mark.parametrize("failure_position", [1, 2, 3, 4])
def test_atomic_publication_rolls_back_every_replace_position(
    tmp_path, monkeypatch, synthetic_result, failure_position
):
    paths = [tmp_path / name for name in ("summary.json", "candidates.csv", "fold.csv", "pareto.csv")]
    originals = []
    for index, path in enumerate(paths):
        payload = f"original-{index}".encode()
        path.write_bytes(payload)
        originals.append(payload)
    real_replace = module.os.replace
    calls = 0

    def failing_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == failure_position:
            raise OSError("controlled replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="controlled"):
        write_artifacts(synthetic_result, *paths)
    assert [path.read_bytes() for path in paths] == originals
    assert not list(tmp_path.glob("*.tmp"))


def test_not_available_contract_contains_only_headers_and_reason_codes():
    payloads = not_available_payloads(["reconciliation_failed"])
    summary = json.loads(payloads[0])
    assert summary["analysis_status"] == "not_available"
    assert summary["partial_results_published"] is False
    for payload, columns in zip(payloads[1:], (CANDIDATE_COLUMNS, BY_FOLD_COLUMNS, PARETO_COLUMNS)):
        table = pd.read_csv(pd.io.common.BytesIO(payload))
        assert table.columns.tolist() == columns
        assert table.empty


def test_static_real_flow_has_one_read_snapshot_build_feature_build_and_barrier():
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = []
    for call in calls:
        if isinstance(call.func, ast.Name):
            names.append(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.append(call.func.attr)
    assert names.count("read_parquet") == 1
    assert names.count("build_historical_snapshots") == 1
    assert names.count("build_direction_features") == 1
    assert names.count("select_validation_features") == 1
    construct = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "construct_sealed_target_features"
    )
    construct_calls = []
    for node in ast.walk(construct):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                construct_calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                construct_calls.append(node.func.attr)
    assert construct_calls.index("seal_target_population") < construct_calls.index(
        "build_historical_snapshots"
    ) < construct_calls.index("build_direction_features")
    assert "2024" not in source.split("def build_summary", 1)[0]


@pytest.mark.integration
def test_published_real_contract_if_artifacts_exist():
    paths = (module.SUMMARY_PATH, module.CANDIDATES_PATH, module.BY_FOLD_PATH, module.PARETO_PATH)
    if not all(path.exists() for path in paths):
        pytest.skip("Faltan los artefactos publicados.")
    summary = json.loads(module.SUMMARY_PATH.read_text(encoding="utf-8"))
    candidates = pd.read_csv(module.CANDIDATES_PATH)
    by_fold = pd.read_csv(module.BY_FOLD_PATH)
    pareto = pd.read_csv(module.PARETO_PATH)
    assert summary["population_contract"]["source_point_rows"] == 1_280_408
    assert summary["population_contract"]["source_matches"] == 7_524
    assert summary["population_contract"]["source_players"] == 1_002
    assert summary["population_contract"]["eligible_second_serve_direction_points"] == 481_190
    assert summary["population_contract"]["eligible_server_wins"] == 245_683
    assert summary["population_contract"]["upstream_total_snapshots"] == 15_048
    assert summary["population_contract"]["upstream_total_feature_rows"] == 90_288
    construction = summary["target_construction_contract"]
    assert construction["constructed_target_matches"] == 5_993
    assert construction["constructed_snapshot_rows"] == 11_986
    assert construction["constructed_feature_rows"] == 71_916
    assert construction["test_target_matches_excluded_before_construction"] == 1_531
    assert construction["test_target_rows_constructed"] == 0
    assert construction["test_target_matches_constructed"] == 0
    assert construction["test_feature_rows_constructed"] == 0
    assert summary["fold_reconciliations"]["matches_by_fold"] == {
        "2020": 168, "2021": 371, "2022": 646, "2023": 620
    }
    assert summary["sealed_test_contract"]["test_rows_evaluated"] == 0
    assert summary["sealed_test_contract"]["test_evaluation_runs"] == 0
    assert len(candidates) == len(pareto) == 27
    assert by_fold["fold"].isin([2020, 2021, 2022, 2023]).all()
    assert summary["analysis_status"] == "available"
