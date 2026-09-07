"""Pruebas sintéticas de P04; no leen Parquet ni artefactos permanentes."""
from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

from src.analysis import return_direction_descriptive_feasibility as p04


def _row(match_id, point, day, surface, server, winner, first, second=None):
    return {"match_id": match_id, "point_number": point, "date": day, "surface": surface,
            "server": server, "point_winner": winner, "player_1": "Player One",
            "player_2": "Player Two", "first_serve": first, "second_serve": second}


@pytest.fixture
def points():
    return pd.DataFrame([
        _row("m1", 1, "2023-01-01", "Hard", 1, 1, "6f1"),
        _row("m2", 1, "2023-02-01", "Clay", 2, 1, "5n", "5b2"),
        _row("m3", 1, "2023-03-01", "Grass", 1, 2, "4r37"),
        _row("m4", 1, "2023-04-01", "Hard", 1, 2, "6f0"),
        _row("m5", 1, "2024-01-01", "Hard", 1, 1, "6f3"),
    ])


def test_source_domains_and_metadata_are_strict(points):
    source = p04.validate_source_points(points)
    assert len(source) == 5
    for field, bad in (("server", True), ("point_winner", "1"), ("surface", "Carpet"), ("player_1", " Player One")):
        changed = points.copy(); changed[field] = changed[field].astype("object"); changed.loc[0, field] = bad
        with pytest.raises(p04.FeasibilityContractError):
            p04.validate_source_points(changed)
    duplicate = pd.concat([points, points.iloc[[0]]], ignore_index=True)
    with pytest.raises(p04.FeasibilityContractError): p04.validate_source_points(duplicate)


def test_seal_applies_before_extractor_and_second_serve_is_not_fabricated(points):
    source = p04.validate_source_points(points)
    development, counts = p04.split_development_before_parsing(source)
    calls = []
    def extractor(text, number, *, previous_attempt_was_fault=False):
        calls.append((text, number, previous_attempt_was_fault))
        return p04.parse_and_classify_initial_return_direction(text, number, previous_attempt_was_fault=previous_attempt_was_fault)
    attempts = p04.construct_attempts(development, extractor=extractor)
    assert counts["excluded_test_target_matches"] == 1
    assert all(text != "6f3" for text, _, _ in calls)
    assert attempts.groupby(["match_id", "point_number", "serve_number"]).size().eq(1).all()
    assert attempts.loc[attempts.match_id.eq("m2") & attempts.serve_number.eq(2), "previous_attempt_was_fault"].item() is True
    assert not attempts.loc[attempts.match_id.eq("m1"), "serve_number"].eq(2).any()


def test_analysis_coverage_outcomes_and_depth_are_separate(points):
    result = p04.analyze_points(points)
    assert result.summary["analysis_status"] == "available_descriptive"
    assert result.summary["population"]["development_point_rows"] == 4
    assert result.summary["test_seal"]["test_attempts_constructed"] == 0
    assert result.summary["direction_counts"] == {"1": 1, "2": 1, "3": 1, "0": 1}
    assert set(result.outcomes.lateral_direction_code) == {"1", "2", "3"}
    assert "0" not in set(result.outcomes.lateral_direction_code)
    depth = result.by_direction.loc[result.by_direction.dimension.eq("return_depth")]
    assert depth.loc[depth.code.eq("7"), "attempts"].item() == 1
    assert all(result.outcomes.returner_point_wins + result.outcomes.server_point_wins == result.outcomes.attempts)


def test_wilson_matches_independent_known_reference():
    lower, upper = p04.wilson(5, 10)
    assert lower == pytest.approx(0.2365930905, abs=1e-9)
    assert upper == pytest.approx(0.7634069095, abs=1e-9)
    assert p04.wilson(0, 0) == (None, None)
    with pytest.raises(p04.FeasibilityContractError): p04.wilson(3, 2)


def test_result_validation_rejects_semantic_mutations(points):
    result = p04.analyze_points(points)
    for changed in (
        replace(result, by_state=result.by_state.iloc[:-1].copy()),
        replace(result, outcomes=result.outcomes.assign(returner_point_wins=0)),
        replace(result, summary={**result.summary, "analysis_status": "not_available"}),
    ):
        with pytest.raises((p04.FeasibilityContractError, AssertionError)):
            p04.validate_result(changed)


def test_limited_status_when_a_lateral_category_is_absent(points):
    limited = points.loc[~points.match_id.eq("m3")].copy()
    result = p04.analyze_points(limited)
    assert result.summary["analysis_status"] == "available_descriptive_limited"


def test_serialization_and_atomic_rollback(points, tmp_path, monkeypatch):
    result = p04.analyze_points(points)
    paths = tuple(tmp_path / name for name in ("summary.json", "state.csv", "direction.csv", "group.csv", "outcomes.csv"))
    p04.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])
    first = tuple(path.read_bytes() for path in paths)
    assert first == p04.serialize_artifacts(result)
    original_replace = p04.os.replace; calls = {"count": 0}
    def fail_third(source, target):
        calls["count"] += 1
        if calls["count"] == 3: raise OSError("synthetic replace failure")
        return original_replace(source, target)
    monkeypatch.setattr(p04.os, "replace", fail_third)
    with pytest.raises(OSError):
        p04.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])
    assert tuple(path.read_bytes() for path in paths) == first
    assert not list(tmp_path.glob("*.tmp"))


def test_persisted_fingerprint_rejects_byte_mutation(points, tmp_path):
    result = p04.analyze_points(points)
    paths = tuple(tmp_path / name for name in ("summary.json", "state.csv", "direction.csv", "group.csv", "outcomes.csv"))
    p04.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])
    p04.verify_persisted_artifacts(summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4], expected_population=result.summary["population"])
    paths[1].write_bytes(paths[1].read_bytes() + b"\n")
    with pytest.raises(p04.FeasibilityContractError):
        p04.verify_persisted_artifacts(summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4], expected_population=result.summary["population"])


def test_no_real_source_or_p04_artifacts_are_read_by_synthetic_build(points, monkeypatch):
    monkeypatch.setattr(p04.pd, "read_parquet", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no parquet")))
    result = p04.analyze_points(points)
    assert len(result.attempts) == 5


def _persisted_paths(result, tmp_path):
    paths = tuple(tmp_path / name for name in ("summary.json", "state.csv", "direction.csv", "group.csv", "outcomes.csv"))
    p04.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])
    return paths


def _resign(paths):
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    payloads = tuple(path.read_bytes() for path in paths[1:])
    summary["publication_fingerprint"] = p04._fingerprint(summary, payloads)
    paths[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")


def _write_frame(paths, position, frame):
    paths[position].write_bytes(p04._frame_bytes(frame))


@pytest.mark.parametrize("mutator", [
    lambda summary, tables: summary.__setitem__("analysis_status", "not_available"),
    lambda summary, tables: summary["population"].__setitem__("development_point_rows", summary["population"]["development_point_rows"] + 1),
    lambda summary, tables: summary["end_to_end_directional_coverage"].__setitem__("denominator", summary["end_to_end_directional_coverage"]["denominator"] + 1),
    lambda summary, tables: summary["direction_counts"].__setitem__("1", summary["direction_counts"]["1"] + 1),
    lambda summary, tables: summary["direction_counts"].__setitem__("0", 7),
    lambda summary, tables: summary["opportunity_coverage"].__setitem__("proportion", 0.1),
    lambda summary, tables: summary["state_counts"].__setitem__("ace", 1),
    lambda summary, tables: summary["reason_code_counts"].__setitem__("censored_ace", 1),
    lambda summary, tables: summary["test_seal"].__setitem__("test_scores_computed", 1),
    lambda summary, tables: summary["test_seal"].__setitem__("used_for_method_selection", True),
    lambda summary, tables: summary["test_seal"].__setitem__("test_status", "open"),
])
def test_persisted_semantics_rejects_resigned_summary_mutations(points, tmp_path, mutator):
    result = p04.analyze_points(points); paths = _persisted_paths(result, tmp_path)
    summary = json.loads(paths[0].read_text(encoding="utf-8")); mutator(summary, [])
    paths[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    _resign(paths)
    with pytest.raises(p04.FeasibilityContractError):
        p04.verify_persisted_artifacts(summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4], expected_population=result.summary["population"])


@pytest.mark.parametrize("mutation", [
    "state_reason", "group_denominator", "group_rate", "serve_number", "surface", "period", "duplicate", "missing", "reversed", "columns", "outcome_unknown", "outcome_numerator", "outcome_rate", "outcome_wilson",
])
def test_persisted_semantics_rejects_resigned_table_mutations(points, tmp_path, mutation):
    result = p04.analyze_points(points); paths = _persisted_paths(result, tmp_path)
    state, direction, group, outcomes = (pd.read_csv(path) for path in paths[1:])
    if mutation == "state_reason": state.loc[0, "reason_code"] = "censored_ace"; _write_frame(paths, 1, state)
    elif mutation == "group_denominator": group.loc[0, "opportunity_denominator"] += 1; _write_frame(paths, 3, group)
    elif mutation == "group_rate": group.loc[0, "end_to_end_rate"] = 0.1; _write_frame(paths, 3, group)
    elif mutation == "serve_number": state.loc[0, "serve_number"] = 3; _write_frame(paths, 1, state)
    elif mutation == "surface": group.loc[group.dimension.eq("surface")].index[0]; group.loc[group.loc[group.dimension.eq("surface")].index[0], "surface"] = "Carpet"; _write_frame(paths, 3, group)
    elif mutation == "period": group.loc[group.loc[group.dimension.eq("derived_period")].index[0], "derived_period"] = "future"; _write_frame(paths, 3, group)
    elif mutation == "duplicate": _write_frame(paths, 1, pd.concat([state, state.iloc[[0]]], ignore_index=True))
    elif mutation == "missing": _write_frame(paths, 3, group.iloc[:-1].copy())
    elif mutation == "reversed": _write_frame(paths, 2, direction.iloc[::-1].reset_index(drop=True))
    elif mutation == "columns": _write_frame(paths, 2, direction.loc[:, list(reversed(direction.columns))])
    elif mutation == "outcome_unknown": outcomes.loc[0, "lateral_direction_code"] = 0; _write_frame(paths, 4, outcomes)
    elif mutation == "outcome_numerator": outcomes.loc[0, "returner_point_wins"] += 1; _write_frame(paths, 4, outcomes)
    elif mutation == "outcome_rate": outcomes["returner_point_win_rate"] = outcomes["returner_point_win_rate"].astype(float); outcomes.loc[0, "returner_point_win_rate"] = 0.1; _write_frame(paths, 4, outcomes)
    elif mutation == "outcome_wilson": outcomes["wilson_95_lower"] = outcomes["wilson_95_lower"].astype(float); outcomes.loc[0, "wilson_95_lower"] = 0.123; _write_frame(paths, 4, outcomes)
    _resign(paths)
    with pytest.raises(p04.FeasibilityContractError):
        p04.verify_persisted_artifacts(summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4], expected_population=result.summary["population"])


@pytest.mark.parametrize("failure_at", range(1, 6))
@pytest.mark.parametrize("with_originals", [False, True])
def test_atomic_publication_rolls_back_all_replace_positions(points, tmp_path, monkeypatch, failure_at, with_originals):
    result = p04.analyze_points(points)
    paths = tuple(tmp_path / name for name in ("summary.json", "state.csv", "direction.csv", "group.csv", "outcomes.csv"))
    originals = tuple(f"before-{index}".encode() for index in range(5))
    if with_originals:
        for path, payload in zip(paths, originals): path.write_bytes(payload)
    replace = p04.os.replace; calls = {"count": 0}
    def fail_at(source, target):
        calls["count"] += 1
        if calls["count"] == failure_at: raise OSError("synthetic replace failure")
        return replace(source, target)
    monkeypatch.setattr(p04.os, "replace", fail_at)
    with pytest.raises(OSError):
        p04.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])
    if with_originals: assert tuple(path.read_bytes() for path in paths) == originals
    else: assert not any(path.exists() for path in paths)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("failure_at", range(1, 6))
def test_atomic_publication_cleans_all_staging_failures(points, tmp_path, monkeypatch, failure_at):
    result = p04.analyze_points(points)
    paths = tuple(tmp_path / name for name in ("summary.json", "state.csv", "direction.csv", "group.csv", "outcomes.csv"))
    stage = p04._stage; calls = {"count": 0}
    def fail_at(path, payload):
        calls["count"] += 1
        if calls["count"] == failure_at: raise OSError("synthetic stage failure")
        return stage(path, payload)
    monkeypatch.setattr(p04, "_stage", fail_at)
    with pytest.raises(OSError):
        p04.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])
    assert not any(path.exists() for path in paths)
    assert not list(tmp_path.glob("*.tmp"))


def test_all_payloads_are_staged_before_any_replace(points, tmp_path, monkeypatch):
    result = p04.analyze_points(points)
    paths = tuple(tmp_path / name for name in ("summary.json", "state.csv", "direction.csv", "group.csv", "outcomes.csv"))
    stage = p04._stage; replace = p04.os.replace; staged = {"count": 0}
    def count_stage(path, payload): staged["count"] += 1; return stage(path, payload)
    def require_all(source, target):
        assert staged["count"] == 5
        return replace(source, target)
    monkeypatch.setattr(p04, "_stage", count_stage); monkeypatch.setattr(p04.os, "replace", require_all)
    p04.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])


def test_test_seal_refresh_only_changes_summary_and_revalidates(points, tmp_path):
    result = p04.analyze_points(points); paths = _persisted_paths(result, tmp_path)
    original_tables = tuple(path.read_bytes() for path in paths[1:])
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    for field in ("test_scores_computed", "test_outcomes_computed", "test_recommendations_generated"):
        del summary["test_seal"][field]
    paths[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    _resign(paths)
    p04.refresh_persisted_summary_test_seal(summary_path=paths[0], by_state_path=paths[1], by_direction_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4], expected_population=result.summary["population"])
    repaired = json.loads(paths[0].read_text(encoding="utf-8"))
    assert all(repaired["test_seal"][field] == 0 for field in p04.TEST_ZERO_FIELDS)
    assert tuple(path.read_bytes() for path in paths[1:]) == original_tables
