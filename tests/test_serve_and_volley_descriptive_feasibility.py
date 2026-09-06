"""Pruebas sintéticas de P03 descriptivo; no abren Parquet ni artefactos reales."""

from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import serve_and_volley_descriptive_feasibility as p03
from src.analysis.serve_and_volley_feasibility import (
    AnalysisState,
    ReasonCode,
    parse_and_classify_serve_and_volley_attempt,
)


def synthetic_points() -> pd.DataFrame:
    """Fixture literal: dos partidos desarrollo y uno de test sellado."""
    return pd.DataFrame([
        {"match_id": "m1", "point_number": 1, "date": "2019-06-01", "surface": "Clay", "server": 1, "point_winner": 1, "player_1": "Alice", "player_2": "Bea", "first_serve": "4+b27v1*", "second_serve": None},
        {"match_id": "m1", "point_number": 2, "date": "2019-06-01", "surface": "Clay", "server": 2, "point_winner": 1, "player_1": "Alice", "player_2": "Bea", "first_serve": "5n", "second_serve": "6+b27v1*"},
        {"match_id": "m2", "point_number": 1, "date": "2020-02-02", "surface": "Hard", "server": 1, "point_winner": 1, "player_1": "Cara", "player_2": "Dana", "first_serve": "4b27v1*", "second_serve": " "},
        {"match_id": "m3", "point_number": 1, "date": "2024-01-01", "surface": "Grass", "server": 1, "point_winner": 2, "player_1": "Eve", "player_2": "Fran", "first_serve": "6+test", "second_serve": "5+test"},
    ])


def _analysis_result() -> p03.DescriptiveResult:
    return p03.analyze_points(synthetic_points())


def test_construction_uses_only_substantive_second_serve_and_unique_attempt_key():
    result = _analysis_result()
    attempts = result.attempts
    assert len(attempts) == 4
    assert attempts.groupby(["match_id", "point_number", "serve_number"]).size().max() == 1
    assert attempts.serve_number.value_counts().to_dict() == {1: 3, 2: 1}
    assert set(attempts.match_id) == {"m1", "m2"}
    assert result.summary["population"]["excluded_test_target_matches"] == 1


def test_seal_precedes_parser_calls_and_test_is_never_constructed():
    source = p03.validate_source_points(synthetic_points())
    development, _ = p03.split_development_before_parsing(source)
    calls: list[tuple[str, int]] = []

    def counted(sequence: str, serve_number: int, *, previous_attempt_was_fault: bool = False):
        calls.append((sequence, serve_number))
        return parse_and_classify_serve_and_volley_attempt(sequence, serve_number, previous_attempt_was_fault=previous_attempt_was_fault)

    attempts = p03.construct_attempts(development, extractor=counted)
    assert len(attempts) == 4
    assert ("6+b27v1*", 2) in calls  # desarrollo m1, punto 2
    assert ("6+test", 1) not in calls
    assert ("5+test", 2) not in calls
    assert development.date.max() <= p03.CUTOFF
    assert not any(attempts.match_id.eq("m3"))


@pytest.mark.parametrize("column,value", [
    ("server", True), ("server", "1"), ("server", 1.0),
    ("point_winner", False), ("point_winner", "2"), ("point_winner", 2.0),
])
def test_server_and_winner_domains_are_strict(column, value):
    frame = synthetic_points()
    frame[column] = frame[column].astype("object")
    frame.loc[0, column] = value
    with pytest.raises(p03.FeasibilityContractError):
        p03.validate_source_points(frame)


@pytest.mark.parametrize("field,value", [
    ("surface", "Carpet"), ("player_1", " "), ("date", "2020/01/01"),
])
def test_source_metadata_domains_are_strict(field, value):
    frame = synthetic_points()
    frame.loc[0, field] = value
    with pytest.raises(p03.FeasibilityContractError):
        p03.validate_source_points(frame)


def test_states_reason_codes_and_unknown_direction_are_preserved_without_recoding():
    result = _analysis_result()
    counts = result.summary["state_counts"]
    assert counts == {
        "positive_tagged": 2,
        "not_explicitly_tagged": 0,
        "unknown": 1,
        "ineligible_censored": 1,
    }
    assert result.summary["reason_code_counts"][ReasonCode.EXPLICIT_MARKER.value] == 2
    assert result.summary["reason_code_counts"][ReasonCode.CENSORED_FAULT.value] == 1
    assert result.summary["reason_code_counts"][ReasonCode.AMBIGUOUS_POST_PREFIX.value] == 1
    assert result.summary["outcome_comparison_status"] == "not_available:no_observable_untagged_comparator"
    assert result.outcomes.empty
    assert result.summary["analysis_status"] == "available_descriptive_not_comparable"


def test_explicit_marker_prevalence_is_distinct_from_comparable_positive_tagged():
    frame = synthetic_points()
    frame.loc[2, "first_serve"] = "0+b27v1*"
    result = p03.analyze_points(frame)
    coverage = result.summary["coverage_summary"]
    assert result.summary["state_counts"]["positive_tagged"] == 2
    assert result.summary["state_counts"]["unknown"] == 1
    assert coverage["explicit_marker_attempts"] == 3
    assert coverage["explicit_marker_positive_tagged_attempts"] == 2
    assert coverage["explicit_marker_unknown_attempts"] == 1
    assert coverage["explicit_marker_censored_attempts"] == 0
    assert coverage["marker_determinable_attempts"] == 3
    assert coverage["comparative_attempts"] == 2
    assert coverage["marker_coverage_rate"] == pytest.approx(3 / 4)
    assert coverage["comparative_coverage_rate"] == pytest.approx(2 / 4)


def test_second_fault_context_is_forwarded_only_when_first_attempt_is_fault():
    source = p03.validate_source_points(synthetic_points())
    development, _ = p03.split_development_before_parsing(source)
    attempts = p03.construct_attempts(development)
    second = attempts.loc[attempts.serve_number.eq(2)].iloc[0]
    assert second.reason_code == ReasonCode.EXPLICIT_MARKER.value
    assert second.analysis_state == AnalysisState.POSITIVE_TAGGED.value


def test_wilson_interval_has_literal_known_value():
    lower, upper = p03._wilson(5, 10)
    assert lower == pytest.approx(0.236593090512564, abs=1e-15)
    assert upper == pytest.approx(0.763406909487436, abs=1e-15)


def test_unknown_and_censored_are_never_outcome_comparators():
    result = _analysis_result()
    assert not result.attempts.loc[result.attempts.analysis_state.isin(["unknown", "ineligible_censored"]), "eligible_for_outcome_comparison"].any()
    assert result.outcomes.empty


def _comparable_attempts() -> pd.DataFrame:
    result = _analysis_result()
    attempts = result.attempts.copy(deep=True)
    unknown = attempts.index[attempts.analysis_state.eq("unknown")][0]
    attempts.loc[unknown, "analysis_state"] = "not_explicitly_tagged"
    attempts.loc[unknown, "reason_code"] = "no_immediate_explicit_marker"
    attempts.loc[unknown, "eligible_for_outcome_comparison"] = True
    return attempts


def test_comparable_fixture_publishes_only_positive_and_observable_untagged():
    attempts = _comparable_attempts()
    outcomes, status = p03.build_outcomes(attempts)
    assert status == "available_descriptive_comparison"
    assert outcomes.analysis_state.tolist() == ["positive_tagged", "not_explicitly_tagged"]
    assert outcomes.attempts.tolist() == [2, 1]
    assert outcomes.loc[outcomes.analysis_state.eq("positive_tagged"), "difference_vs_not_explicitly_tagged"].iloc[0] == pytest.approx(-0.5)


def test_group_aggregates_are_deterministic_and_reconcile():
    result = _analysis_result()
    rebuilt = p03.build_by_group(result.attempts)
    pd.testing.assert_frame_equal(result.by_group, rebuilt, check_dtype=False)
    total = result.by_group.loc[result.by_group.dimension.eq("total")].iloc[0]
    assert total.attempts == 4
    assert total.coverage_denominator == 4
    assert total.coverage_rate == pytest.approx(0.5)
    assert set(result.by_group.loc[result.by_group.dimension.eq("surface"), "surface"]) == {"Hard", "Clay"}
    assert set(result.by_group.loc[result.by_group.dimension.eq("derived_period"), "derived_period"]) == {"2010s", "2020s"}


def test_summary_seal_and_test_counters_are_zero():
    summary = _analysis_result().summary
    assert summary["test_seal"]["test_status"] == "sealed"
    assert summary["test_seal"]["used_for_method_selection"] is False
    assert all(summary["test_seal"][field] == 0 for field in p03.TEST_ZERO_FIELDS)


def test_validator_rejects_semantic_mutations_even_when_fingerprint_is_recomputed():
    result = _analysis_result()
    by_state = result.by_state.copy(deep=True)
    by_state.loc[0, "attempts"] += 1
    summary = result.summary.copy()
    payloads = tuple(p03._frame_bytes(frame) for frame in (by_state, result.by_group, result.outcomes))
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(("by_state", "by_group", "outcomes"), payloads)}
    summary["publication_fingerprint"] = p03._fingerprint(summary, payloads)
    with pytest.raises(p03.FeasibilityContractError):
        p03.validate_result(replace(result, summary=summary, by_state=by_state, publication_fingerprint=summary["publication_fingerprint"]))


@pytest.mark.parametrize("path", [
    ("population", "first_serve_attempts"),
    ("population", "second_serve_attempts"),
    ("coverage_summary", "explicit_marker_attempts"),
    ("coverage_summary", "comparative_attempts"),
])
def test_result_validator_rejects_derived_summary_mutations(path):
    result = _analysis_result()
    summary = copy.deepcopy(result.summary)
    summary[path[0]][path[1]] += 1
    payloads = tuple(p03._frame_bytes(frame) for frame in (result.by_state, result.by_group, result.outcomes))
    summary["publication_fingerprint"] = p03._fingerprint(summary, payloads)
    with pytest.raises(p03.FeasibilityContractError):
        p03.validate_result(replace(result, summary=summary, publication_fingerprint=summary["publication_fingerprint"]))


def test_persisted_validator_rejects_contract_mutations_with_rekeyed_fingerprint(tmp_path: Path):
    result = _analysis_result()
    paths = tuple(tmp_path / value for value in ("summary.json", "state.csv", "group.csv", "outcomes.csv"))
    p03.publish_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_group_path=paths[2], outcomes_path=paths[3])
    summary = __import__("json").loads(paths[0].read_text(encoding="utf-8"))
    summary["definition"]["physical_serve_and_volley_proven"] = True
    payloads = tuple(path.read_bytes() for path in paths[1:])
    summary["publication_fingerprint"] = p03._fingerprint(summary, payloads)
    paths[0].write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(p03.FeasibilityContractError):
        p03.verify_persisted_artifacts(*paths)


def test_persisted_validator_rejects_coverage_mutation_with_rekeyed_fingerprint(tmp_path: Path):
    result = _analysis_result()
    paths = tuple(tmp_path / value for value in ("summary.json", "state.csv", "group.csv", "outcomes.csv"))
    p03.publish_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_group_path=paths[2], outcomes_path=paths[3])
    summary = __import__("json").loads(paths[0].read_text(encoding="utf-8"))
    summary["coverage_summary"]["explicit_marker_attempts"] += 1
    payloads = tuple(path.read_bytes() for path in paths[1:])
    summary["publication_fingerprint"] = p03._fingerprint(summary, payloads)
    paths[0].write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(p03.FeasibilityContractError):
        p03.verify_persisted_artifacts(*paths)


def test_persisted_validator_rejects_not_available_with_published_rows(tmp_path: Path):
    result = _analysis_result()
    paths = tuple(tmp_path / value for value in ("summary.json", "state.csv", "group.csv", "outcomes.csv"))
    p03.publish_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_group_path=paths[2], outcomes_path=paths[3])
    summary = __import__("json").loads(paths[0].read_text(encoding="utf-8"))
    summary["analysis_status"] = "not_available"
    payloads = tuple(path.read_bytes() for path in paths[1:])
    summary["publication_fingerprint"] = p03._fingerprint(summary, payloads)
    paths[0].write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(p03.FeasibilityContractError):
        p03.verify_persisted_artifacts(*paths)


@pytest.mark.parametrize("failure_index", range(4))
@pytest.mark.parametrize("preexisting", [False, True])
def test_atomic_rollback_restores_every_artifact_position(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int, preexisting: bool):
    result = _analysis_result()
    paths = tuple(tmp_path / value for value in ("summary.json", "state.csv", "group.csv", "outcomes.csv"))
    if preexisting:
        for path in paths:
            path.write_bytes(b"old:" + path.name.encode("ascii"))
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    original = p03.os.replace
    calls = {"n": 0}

    def fail(source, target):
        if calls["n"] == failure_index:
            calls["n"] += 1
            raise OSError("synthetic replace failure")
        calls["n"] += 1
        original(source, target)

    monkeypatch.setattr(p03.os, "replace", fail)
    with pytest.raises(OSError, match="synthetic"):
        p03.write_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_group_path=paths[2], outcomes_path=paths[3])
    assert {path: path.read_bytes() if path.exists() else None for path in paths} == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_serialization_is_deterministic_and_persisted_contract_reopens(tmp_path: Path):
    result = _analysis_result()
    paths = tuple(tmp_path / value for value in ("summary.json", "state.csv", "group.csv", "outcomes.csv"))
    first = p03.serialize_artifacts(result)
    assert p03.serialize_artifacts(result) == first
    p03.publish_artifacts(result, summary_path=paths[0], by_state_path=paths[1], by_group_path=paths[2], outcomes_path=paths[3])
    assert tuple(path.read_bytes() for path in paths) == first
    p03.verify_persisted_artifacts(*paths, result=result, expected_payloads=first)


def test_not_available_has_headers_only_and_no_nonfinite_serialization():
    result = p03.not_available_result(ValueError("synthetic"))
    p03.validate_result(result)
    payloads = p03.serialize_artifacts(result)
    assert all(frame.empty for frame in (result.by_state, result.by_group, result.outcomes))
    assert b"NaN" not in b"".join(payloads)
    assert b"Infinity" not in b"".join(payloads)


def test_module_has_no_real_data_read_at_import_or_unapproved_analysis_scope():
    source = Path(p03.__file__).read_text(encoding="utf-8")
    assert source.count("read_parquet(") == 1  # La unica lectura esta aislada en read_source_points.
    assert "def main" in source
    assert "def recommend" not in source
    assert "statsmodels" not in source


@pytest.mark.integration
def test_real_published_p03_artifacts_if_available():
    if not p03.SUMMARY_PATH.exists():
        pytest.skip("Los artefactos P03 aún no se han publicado.")
    p03.verify_persisted_artifacts()
    summary = __import__("json").loads(p03.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["analysis_status"] == "available_descriptive_not_comparable"
    assert summary["population"]["attempts"] == 1_426_863
    assert summary["state_counts"] == {
        "positive_tagged": 130_963, "not_explicitly_tagged": 0,
        "unknown": 767_363, "ineligible_censored": 528_537,
    }
