from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.explainable_direction_scoring as module
from src.analysis.evidence_policy_validation import select_validation_features
from src.analysis.player_opponent_direction_features import FEATURE_COLUMNS
from src.analysis.explainable_direction_scoring import (
    BY_DIRECTION_COLUMNS,
    BY_FOLD_COLUMNS,
    CALIBRATION_COLUMNS,
    MAIN_SCORER,
    RANKING_COLUMNS,
    SCORERS,
    ScoringContractError,
    ScoringResult,
    _safe_spearman,
    build_calibration,
    build_paired_match_comparison,
    build_probability_tables,
    build_ranking,
    determine_baseline_status,
    join_target_outcomes,
    not_available_result,
    score_components,
    serialize_artifacts,
    validate_result,
    write_artifacts,
)


def _orientation_rows() -> pd.DataFrame:
    rows = []
    for match, fold in (("m1", 2020), ("m2", 2021), ("m3", 2022), ("m4", 2023)):
        for player, opponent in (("server", "returner"), ("returner", "server")):
            for index, direction in enumerate(("wide", "body", "T")):
                rows.append({
                    "target_match_id": match,
                    "target_date": pd.Timestamp(f"{fold}-01-02"),
                    "target_player": player,
                    "opponent": opponent,
                    "target_surface": "Hard",
                    "direction": direction,
                    "selected_scope": "surface" if index != 2 else "global",
                    "eligible": True,
                    "server_rate": (.70, .55, .45)[index],
                    "opponent_allowed_rate": (.60, .50, .40)[index],
                    "population_rate": (.50, .50, .50)[index],
                    "server_history_direction_points": 50,
                    "server_history_direction_matches": 5,
                    "opponent_history_direction_points": 50,
                    "opponent_history_direction_matches": 5,
                })
    return score_components(pd.DataFrame(rows).assign(
        fold=lambda x: x["target_date"].dt.year,
        evidence_status="eligible_surface",
        abstention_reason=None,
        opponent_rate=lambda x: x["opponent_allowed_rate"],
    ))


def _upstream_schema_features() -> pd.DataFrame:
    """Schema real: 4 folds x 2 orientaciones x 3 direcciones x 2 scopes."""

    rows = []
    for year in module.FOLDS:
        for player, opponent in ((f"a{year}", f"b{year}"), (f"b{year}", f"a{year}")):
            for direction in ("wide", "body", "T"):
                for scope in ("global", "surface"):
                    date = pd.Timestamp(year=year, month=1, day=2)
                    row = {column: None for column in FEATURE_COLUMNS}
                    row.update({
                        "target_match_id": f"m{year}", "target_date": date,
                        "target_player": player, "opponent": opponent, "target_surface": "Hard",
                        "derived_period": "2020s", "direction": direction, "scope": scope,
                        "server_direction_matches": 5, "server_direction_points": 50,
                        "server_direction_wins": 30, "server_direction_raw_rate": .6,
                        "server_direction_wilson_low": .45, "server_direction_wilson_high": .73,
                        "server_direction_share": 1 / 3, "server_scope_points": 150,
                        "server_scope_wins": 90, "server_scope_raw_rate": .6,
                        "server_direction_first_history_date": pd.Timestamp("2019-01-01"),
                        "server_direction_last_history_date": pd.Timestamp("2019-12-31"),
                        "opponent_direction_matches": 5, "opponent_direction_points": 50,
                        "opponent_allowed_server_wins": 25, "opponent_return_wins": 25,
                        "opponent_allowed_server_raw_rate": .5,
                        "opponent_allowed_server_wilson_low": .36,
                        "opponent_allowed_server_wilson_high": .64,
                        "opponent_direction_share": 1 / 3, "opponent_scope_points": 150,
                        "opponent_scope_allowed_server_wins": 75,
                        "opponent_scope_allowed_server_raw_rate": .5,
                        "opponent_direction_first_history_date": pd.Timestamp("2019-01-01"),
                        "opponent_direction_last_history_date": pd.Timestamp("2019-12-31"),
                        "population_direction_points": 1000,
                        "population_direction_server_wins": 550,
                        "population_direction_raw_rate": .55,
                        "population_direction_wilson_low": .52,
                        "population_direction_wilson_high": .58,
                        "population_direction_first_history_date": pd.Timestamp("1960-01-01"),
                        "population_direction_last_history_date": pd.Timestamp("2019-12-31"),
                        "server_excess_vs_population": .05,
                        "opponent_allowed_excess_vs_population": -.05,
                        "server_opponent_gap": .1,
                        "server_evidence_state": "observed", "opponent_evidence_state": "observed",
                        "pair_evidence_state": "both_observed",
                    })
                    rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _eligible_points(orientations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in orientations.itertuples(index=False):
        for point_number, outcome in ((1, True), (2, False)):
            rows.append({
                "match_id": row.target_match_id,
                "point_number": f"{row.target_player}-{row.direction}-{point_number}",
                "date": row.target_date,
                "surface": row.target_surface,
                "server_player": row.target_player,
                "returner_player": row.opponent,
                "server_won_point": outcome,
                "direction": row.direction.lower() if row.direction != "T" else "t",
            })
    return pd.DataFrame(rows)


def _scored_long() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orientations = _orientation_rows()
    all_points, scored = join_target_outcomes(orientations, _eligible_points(orientations))
    long = module._score_long(scored)
    return all_points, scored, long


def _synthetic_result() -> ScoringResult:
    all_points, scored, long = _scored_long()
    by_fold, by_direction, calibration = build_probability_tables(long, all_points)
    ranking = build_ranking(long)
    status, reasons, checks = determine_baseline_status(long)
    identity = (scored["combined_gap"] - (scored["server_component"] + scored["opponent_component"]) / 2).abs()
    summary = {
        "analysis_name": module.ANALYSIS_NAME,
        "analysis_version": module.ANALYSIS_VERSION,
        "baseline_status": status,
        "reason_codes": sorted(reasons),
        "selected_policy_contract": {
            "policy_id": module.POLICY_ID,
            "min_points_per_direction_and_role": module.MIN_POINTS,
            "min_matches_per_direction_and_role": module.MIN_MATCHES,
            "scope_policy": module.SCOPE_POLICY,
            "joint_role_eligibility": True,
            "joint_global_fallback": True,
            "mixed_scopes": False,
        },
        "scorer_contracts": {
            "population_only": "population_rate", "server_only": "server_rate",
            "opponent_only": "opponent_allowed_rate",
            "server_opponent_equal": "(server_rate + opponent_allowed_rate) / 2",
            "primary_scorer": module.MAIN_SCORER,
            "weights_fixed": {"server": .5, "opponent": .5},
        },
        "formula_contract": {
            "server_component": "server_rate - population_rate",
            "opponent_component": "opponent_allowed_rate - population_rate",
            "combined_score": "(server_rate + opponent_allowed_rate) / 2",
            "combined_gap": "combined_score - population_rate",
            "identity": "combined_gap = (server_component + opponent_component) / 2",
            "smoothing": False, "imputation": False,
        },
        "formula_identity_max_abs_error": float(identity.max()),
        "formula_identity_tolerance": 1e-15,
        "formula_identity_rows_checked": int(len(identity)),
        "sealed_test_contract": {
            "test_status": "sealed", "test_target_matches_constructed": 0,
            "test_feature_rows_constructed": 0, "test_points_scored": 0,
            "test_matches_scored": 0, "test_evaluation_runs": 0,
            "used_for_method_selection": False,
        },
        "coverage": module._coverage_summary(all_points, scored),
        "validation_population": {
            "recognized_direction_points": int(len(all_points)),
            "substantive_second_serve_points": int(len(all_points)),
            "directions": {direction: int(all_points.direction.eq(direction).sum()) for direction in module.DIRECTION_ORDER},
        },
        "probability_metrics": checks["pooled"],
        "calibration": {"bins": module.BIN_COUNT, "definition": "fixed_width_0.1", "rows": len(calibration), "empty_bins_retained": True, "log_loss_epsilon": module.EPSILON},
        "ranking": {"rows": len(ranking)},
        "paired_match_comparison": build_paired_match_comparison(long),
        "explainability": module.build_explainability(scored),
        "reconciliations": {key: True for key in module.RECONCILIATION_KEYS},
        "validation_criteria": {"contract_and_temporality": True, "scores_finite_in_unit_interval": True, **checks["fold_checks"]},
    }
    summary, _, fingerprint = module._finalize_summary(summary, (by_fold, by_direction, calibration, ranking))
    return ScoringResult(summary, by_fold, by_direction, calibration, ranking, fingerprint)


def test_formula_is_exact_symmetric_and_preserves_identity():
    result = score_components(pd.DataFrame([{
        "server_rate": .8, "opponent_allowed_rate": .6, "population_rate": .5, "eligible": True,
    }]))
    row = result.iloc[0]
    assert row.server_component == pytest.approx(.3)
    assert row.opponent_component == pytest.approx(.1)
    assert row.combined_score == pytest.approx(.7)
    assert row.combined_gap == pytest.approx(.2)
    assert row.combined_gap == pytest.approx((row.server_component + row.opponent_component) / 2)
    assert row.population_only == pytest.approx(.5)
    assert row.server_only == pytest.approx(.8)
    assert row.opponent_only == pytest.approx(.6)
    assert row.server_opponent_equal == pytest.approx(.7)


@pytest.mark.parametrize("column,value", [("server_rate", np.nan), ("opponent_allowed_rate", np.inf), ("population_rate", 1.1)])
def test_formula_rejects_non_finite_or_out_of_range_probabilities(column, value):
    frame = pd.DataFrame([{"server_rate": .5, "opponent_allowed_rate": .5, "population_rate": .5, "eligible": True}])
    frame.loc[0, column] = value
    with pytest.raises(ScoringContractError):
        score_components(frame)


def test_ineligible_orientation_is_abstained_without_imputation():
    result = score_components(pd.DataFrame([{
        "server_rate": np.nan, "opponent_allowed_rate": np.nan, "population_rate": np.nan, "eligible": False,
    }]))
    assert result.loc[0, list(SCORERS)].isna().all()
    assert pd.isna(result.loc[0, "combined_gap"])


def test_real_upstream_feature_schema_reproduces_and_fixes_scoring_population_contract():
    features = _upstream_schema_features()
    validation = select_validation_features(features)
    prepared = module.prepare_scoring_population(validation)
    assert len(validation) == 48
    assert len(prepared) == 24
    assert prepared[["target_match_id", "target_player", "direction"]].duplicated().sum() == 0
    assert prepared["selected_scope"].eq("surface").all()
    assert prepared["eligible"].all()
    assert prepared["opponent_allowed_rate"].eq(.5).all()
    assert prepared["combined_score"].eq(.55).all()


def test_validation_feature_schema_folds_keys_and_test_barrier_are_strict():
    features = _upstream_schema_features()
    with pytest.raises(ValueError, match="esquema"):
        select_validation_features(features.drop(columns=FEATURE_COLUMNS[-1]))
    after = features.copy(deep=True)
    after.loc[0, "target_date"] = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="2023"):
        select_validation_features(after)
    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unica"):
        select_validation_features(duplicated)
    with pytest.raises(ValueError, match="Contaminacion"):
        select_validation_features(features, test_match_ids={"m2020"})


def test_scoring_population_rejects_missing_scope_pair_and_invalid_direction_contract():
    features = _upstream_schema_features()
    missing_scope = features.loc[~((features["target_match_id"].eq("m2020")) & features["scope"].eq("surface"))]
    with pytest.raises(ValueError, match="pares exactos|one_to_one"):
        module.prepare_scoring_population(select_validation_features(missing_scope))
    invalid_direction = features.copy(deep=True)
    invalid_direction.loc[0, "direction"] = "unknown"
    with pytest.raises(ValueError):
        module.prepare_scoring_population(select_validation_features(invalid_direction))


def test_join_excludes_test_and_uses_only_prebuilt_orientation_score():
    orientations = _orientation_rows().iloc[:3].copy()
    points = _eligible_points(orientations)
    points.loc[0, "date"] = pd.Timestamp("2024-01-01")
    all_points, scored = join_target_outcomes(orientations, points)
    assert all_points["date"].max() <= pd.Timestamp("2023-12-31")
    assert len(scored) == len(all_points)
    assert scored["outcome"].isin([True, False]).all()


def test_join_rejects_missing_direction_score():
    orientations = _orientation_rows().iloc[:2].copy()
    with pytest.raises(ScoringContractError, match="no encontro"):
        join_target_outcomes(orientations, _eligible_points(_orientation_rows().iloc[:3]))


@pytest.mark.parametrize(
    "values,accepted",
    [
        ([True, False], True),
        ([np.bool_(True), np.bool_(False)], True),
        ([1, 0], False), ([1.0, 0.0], False), (["1", "0"], False),
        ([True, 1], False), ([True, None], False),
    ],
)
def test_join_requires_strict_boolean_outcome(values, accepted):
    orientations = _orientation_rows().iloc[:1].copy()
    points = pd.concat([_eligible_points(orientations), _eligible_points(orientations)], ignore_index=True)
    points["point_number"] = ["first", "second", "third", "fourth"]
    points["server_won_point"] = pd.Series((values * 2)[:4], dtype="object")
    if accepted:
        _, scored = join_target_outcomes(orientations, points)
        assert len(scored) == 4
    else:
        with pytest.raises(ScoringContractError, match="booleanos reales"):
            join_target_outcomes(orientations, points)


def test_probability_metrics_brier_logloss_and_clip_are_manual():
    all_points, scored, long = _scored_long()
    row = long.loc[long.scorer.eq("population_only")].iloc[0]
    assert row.squared_error == pytest.approx((.5 - 1.0) ** 2)
    assert row.log_loss_value == pytest.approx(-np.log(.5))
    zero = scored.iloc[[0]].copy()
    zero.loc[:, list(SCORERS)] = 0.0
    zero["outcome"] = True
    clipped = module._score_long(zero)
    assert np.isfinite(clipped.loc[clipped.index[0], "log_loss_value"])
    by_fold, _, calibration = build_probability_tables(long, all_points)
    assert by_fold["brier_score"].between(0, 1).all()
    assert len(calibration) == 200


def test_fixed_bins_include_boundaries_and_empty_rows():
    long = pd.DataFrame({
        "scorer": ["population_only", "population_only"], "fold": [2020, 2020],
        "score": [0.0, 1.0], "outcome": [False, True], "target_match_id": ["a", "b"],
    })
    calibration, summary = build_calibration(long)
    first = calibration.query("scorer == 'population_only' and fold_or_pooled == '2020' and bin_index == 0").iloc[0]
    last = calibration.query("scorer == 'population_only' and fold_or_pooled == '2020' and bin_index == 9").iloc[0]
    empty = calibration.query("scorer == 'population_only' and fold_or_pooled == '2020' and bin_index == 1").iloc[0]
    assert (first.n_points, last.n_points, empty.n_points) == (1, 1, 0)
    assert pd.isna(empty.mean_score) and pd.isna(empty.observed_rate)
    assert summary[("population_only", "2020")]["ece"] == pytest.approx(0.0)


def test_ece_is_computed_from_fixed_bin_weights():
    long = pd.DataFrame({
        "scorer": ["population_only", "population_only"], "fold": [2020, 2020],
        "score": [.1, .9], "outcome": [True, True], "target_match_id": ["a", "b"],
    })
    _, values = build_calibration(long)
    assert values[("population_only", "2020")]["ece"] == pytest.approx((.9 + .1) / 2)


def test_ranking_uses_canonical_tie_break_observed_ties_mrr_and_regret():
    rows = []
    for direction, score, outcome in (("wide", .6, .5), ("body", .6, .7), ("T", .2, .7)):
        rows.append({"scorer": "population_only", "fold": 2020, "target_match_id": "m", "target_player": "p", "orientation_id": "m|p", "direction": direction, "score": score, "outcome": outcome})
    long = pd.DataFrame(rows * 4)
    long["scorer"] = np.repeat(SCORERS, 3)
    ranking = build_ranking(long)
    row = ranking.query("scorer == 'population_only' and fold_or_pooled == '2020'").iloc[0]
    assert row.top1_hit_inclusive == pytest.approx(0.0)
    assert row.top1_hit_strict is None
    assert row.mrr == pytest.approx(.5)
    assert row.mean_regret == pytest.approx(.2)


@pytest.mark.parametrize(
    "scores,observed,reason",
    [([.5], [.5], "insufficient_directions"), ([.5, .5, .5], [.1, .2, .3], "constant_values"), ([.1, .2, .3], [.5, .5, .5], "constant_values")],
)
def test_spearman_undefined_is_null_without_warning(scores, observed, reason):
    value, observed_reason = _safe_spearman(pd.Series(scores), pd.Series(observed))
    assert value is None and observed_reason == reason


def test_spearman_valid_and_paired_brier_comparison_are_manual():
    _, _, long = _scored_long()
    value, reason = _safe_spearman(pd.Series([.1, .2, .3]), pd.Series([.1, .2, .3]))
    assert value == pytest.approx(1.0) and reason is None
    paired = build_paired_match_comparison(long)
    assert set(paired) == {"population_only", "server_only", "opponent_only"}
    assert paired["population_only"]["pairs"] == 4


def test_validation_criteria_can_validate_and_not_validate():
    rows = []
    for fold in module.FOLDS:
        for scorer, score in (("population_only", .4), ("server_only", .8), ("opponent_only", .8), (MAIN_SCORER, .4)):
            rows.append({"scorer": scorer, "fold": fold, "score": score, "outcome": False, "target_match_id": str(fold), "orientation_id": str(fold), "squared_error": score**2, "log_loss_value": -np.log(1 - score)})
    status, reasons, _ = determine_baseline_status(pd.DataFrame(rows))
    assert status == "validated_descriptive_baseline" and reasons == []
    invalid = pd.DataFrame(rows)
    invalid.loc[invalid.scorer.eq(MAIN_SCORER), ["score", "squared_error", "log_loss_value"]] = [.9, .81, -np.log(.1)]
    status, reasons, _ = determine_baseline_status(invalid)
    assert status == "not_validated" and "main_brier_not_worse_than_population" in reasons


def test_tables_have_exact_order_and_counts():
    all_points, _, long = _scored_long()
    by_fold, by_direction, calibration = build_probability_tables(long, all_points)
    ranking = build_ranking(long)
    assert by_fold.columns.tolist() == BY_FOLD_COLUMNS
    assert by_direction.columns.tolist() == BY_DIRECTION_COLUMNS
    assert calibration.columns.tolist() == CALIBRATION_COLUMNS and len(calibration) == 200
    assert ranking.columns.tolist() == RANKING_COLUMNS and len(ranking) == 20
    assert ranking[["scorer", "fold_or_pooled"]].values.tolist()[:5] == [["population_only", "2020"], ["population_only", "2021"], ["population_only", "2022"], ["population_only", "2023"], ["population_only", "pooled"]]


def test_not_available_has_no_partial_evaluation():
    result = not_available_result("upstream_hash_mismatch")
    validate_result(result)
    assert result.summary["baseline_status"] == "not_available"
    assert all(frame.empty for frame in (result.by_fold, result.by_direction, result.calibration, result.ranking))


def test_not_available_preserves_exact_stage_type_sanitized_message_and_reason_code():
    failure = module.ScoringExecutionError(
        "prepare_scoring_population",
        ValueError("Faltan columnas para score: ['opponent_allowed_rate']\nC:\\Users\\Errandi\\secret"),
        ["load_upstream_contracts", "read_source", "prepare_history", "seal_targets", "build_snapshots", "build_features", "select_validation_features"],
        {"constructed_feature_rows": 72},
    )
    result = not_available_result(failure)
    validate_result(result)
    summary = result.summary
    assert summary["failure_stage"] == "prepare_scoring_population"
    assert summary["exception_type"] == "ValueError"
    assert summary["reason_codes"] == ["scoring_population_key_mismatch"]
    assert "opponent_allowed_rate" in summary["exception_message"]
    assert "C:\\Users" not in summary["exception_message"] and "\n" not in summary["exception_message"]
    assert summary["last_completed_stage"] == "select_validation_features"
    assert summary["partial_population_diagnostics"] == {"constructed_feature_rows": 72}
    assert summary["inference_published"] is False


def test_validation_rejects_fingerprint_test_seal_and_nonfinite_manipulation():
    result = _synthetic_result()
    validate_result(result)
    summary = json.loads(json.dumps(result.summary))
    summary["sealed_test_contract"]["test_evaluation_runs"] = 1
    with pytest.raises(ScoringContractError):
        validate_result(replace(result, summary=summary))
    damaged = result.by_fold.copy(deep=True)
    damaged.loc[0, "brier_score"] = np.inf
    with pytest.raises(ScoringContractError):
        validate_result(replace(result, by_fold=damaged))
    with pytest.raises(ScoringContractError, match="Fingerprint"):
        validate_result(replace(result, publication_fingerprint="0" * 64))


@pytest.mark.parametrize(
    "mutation",
    [
        "baseline_status", "criterion", "coverage", "formula", "weight", "policy",
        "scope", "brier", "log_loss", "calibration", "ranking", "paired",
        "explainability", "reconciliation_missing", "reconciliation_extra",
        "reconciliation_coercible", "csv_duplicate", "csv_order",
    ],
)
def test_semantic_mutations_are_rejected_after_recalculating_fingerprint(mutation):
    result = _synthetic_result()
    summary = json.loads(json.dumps(result.summary))
    by_fold = result.by_fold.copy(deep=True)
    by_direction = result.by_direction.copy(deep=True)
    calibration = result.calibration.copy(deep=True)
    ranking = result.ranking.copy(deep=True)
    if mutation == "baseline_status":
        summary["baseline_status"] = "validated_descriptive_baseline"
    elif mutation == "criterion":
        current = summary["validation_criteria"]["main_brier_not_worse_than_population"]
        summary["validation_criteria"]["main_brier_not_worse_than_population"] = not current
    elif mutation == "coverage":
        summary["coverage"]["scored_target_points"] += 1
    elif mutation == "formula":
        summary["formula_contract"]["combined_score"] = "server_rate"
    elif mutation == "weight":
        summary["scorer_contracts"]["weights_fixed"]["server"] = .6
    elif mutation == "policy":
        summary["selected_policy_contract"]["policy_id"] = "other"
    elif mutation == "scope":
        by_fold.loc[0, "selected_scope"] = "other"
    elif mutation == "brier":
        by_direction.loc[0, "brier_score"] += .1
    elif mutation == "log_loss":
        by_direction.loc[0, "log_loss"] += .1
    elif mutation == "calibration":
        row = calibration.index[calibration["n_points"].gt(0)][0]
        calibration.loc[row, "absolute_gap"] += .1
    elif mutation == "ranking":
        ranking.loc[0, "mean_regret"] = -.1
    elif mutation == "paired":
        summary["paired_match_comparison"]["population_only"]["pairs"] += 1
    elif mutation == "explainability":
        summary["explainability"]["dominant_component"]["server"] += 1
    elif mutation == "reconciliation_missing":
        summary["reconciliations"].pop("policy_exact")
    elif mutation == "reconciliation_extra":
        summary["reconciliations"]["extra"] = True
    elif mutation == "reconciliation_coercible":
        summary["reconciliations"]["policy_exact"] = 1
    elif mutation == "csv_duplicate":
        by_fold = pd.concat([by_fold, by_fold.iloc[[0]]], ignore_index=True)
    elif mutation == "csv_order":
        ranking = ranking.iloc[::-1].reset_index(drop=True)
    summary, _, fingerprint = module._finalize_summary(summary, (by_fold, by_direction, calibration, ranking))
    with pytest.raises(ScoringContractError):
        validate_result(ScoringResult(summary, by_fold, by_direction, calibration, ranking, fingerprint))


def test_serialization_is_deterministic_utf8_and_without_indices():
    result = _synthetic_result()
    first, second = serialize_artifacts(result), serialize_artifacts(result)
    assert first == second
    for payload, columns in zip(first[1:], (BY_FOLD_COLUMNS, BY_DIRECTION_COLUMNS, CALIBRATION_COLUMNS, RANKING_COLUMNS)):
        frame = pd.read_csv(io.BytesIO(payload))
        assert frame.columns.tolist() == columns
        assert not any(column.startswith("Unnamed") for column in frame.columns)


def test_fingerprint_v2_verifies_exact_persisted_bytes_and_rejects_mutations(tmp_path):
    result = _synthetic_result()
    paths = tuple(tmp_path / name for name in ("summary.json", "fold.csv", "direction.csv", "calibration.csv", "ranking.csv"))
    write_artifacts(result, *paths)
    reopened = module.verify_persisted_artifacts(*paths)
    assert reopened.publication_fingerprint == result.publication_fingerprint
    assert reopened.summary["fingerprint_contract_version"] == "2"
    assert reopened.summary["artifact_payload_sha256"]["by_fold"] == hashlib.sha256(paths[1].read_bytes()).hexdigest().upper()
    original = paths[1].read_bytes()
    paths[1].write_bytes(original.replace(b"\n", b"\r\n", 1))
    with pytest.raises(ScoringContractError, match="Hashes o tamanos"):
        module.verify_persisted_artifacts(*paths)
    paths[1].write_bytes(original)
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    summary["publication_fingerprint"] = "0" * 64
    paths[0].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ScoringContractError, match="Fingerprint"):
        module.verify_persisted_artifacts(*paths)


@pytest.mark.parametrize("replacement_position", range(1, 6))
@pytest.mark.parametrize("preexisting", [True, False])
def test_atomic_publication_and_rollback_across_five_artifacts(tmp_path, monkeypatch, replacement_position, preexisting):
    result = _synthetic_result()
    paths = tuple(tmp_path / name for name in ("summary.json", "fold.csv", "direction.csv", "calibration.csv", "ranking.csv"))
    if preexisting:
        write_artifacts(result, *paths)
        originals = tuple(path.read_bytes() for path in paths)
    else:
        originals = (None,) * len(paths)
    real_replace, calls = module.os.replace, 0
    def fail_replacement(source, destination):
        nonlocal calls
        calls += 1
        if calls == replacement_position:
            raise OSError("controlled replacement failure")
        return real_replace(source, destination)
    monkeypatch.setattr(module.os, "replace", fail_replacement)
    with pytest.raises(OSError, match="controlled"):
        write_artifacts(result, *paths)
    assert tuple(path.read_bytes() if path.exists() else None for path in paths) == originals
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(module.os, "replace", real_replace)
    write_artifacts(result, *paths)
    assert tuple(path.read_bytes() for path in paths) == serialize_artifacts(result)


def test_upstream_hash_policy_and_static_single_read_contract(monkeypatch):
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert source.count("pd.read_parquet(") == 1
    assert source.count("construct_sealed_target_features(") == 1
    assert "pd.read_parquet(" in source
    monkeypatch.setitem(module.UPSTREAM_HASHES, "reports/evidence_policy_selection_summary.json", "0" * 64)
    with pytest.raises(ScoringContractError, match="SHA-256"):
        module.load_and_validate_upstream_artifacts()


@pytest.mark.integration
def test_real_artifacts_contract_without_reexecuting_parquet_analysis():
    paths = (module.SUMMARY_PATH, module.BY_FOLD_PATH, module.BY_DIRECTION_PATH, module.CALIBRATION_PATH, module.RANKING_PATH)
    if not all(path.exists() for path in paths):
        pytest.skip("Aun no existen artefactos reales; no se ejecuta el Parquet desde pytest.")
    reopened = module.verify_persisted_artifacts(*paths)
    summary = reopened.summary
    assert summary["sealed_test_contract"]["test_status"] == "sealed"
    assert summary["sealed_test_contract"]["test_evaluation_runs"] == 0
    assert summary["selected_policy_contract"]["policy_id"] == module.POLICY_ID
    assert summary["fingerprint_contract_version"] == "2"
    assert summary["formula_identity_max_abs_error"] <= summary["formula_identity_tolerance"]
    assert len(pd.read_csv(module.CALIBRATION_PATH)) == 200
    assert len(pd.read_csv(module.RANKING_PATH)) == 20
