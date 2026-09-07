"""Contrato sintetico P05; no abre puntos ni Parquet."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import return_depth_descriptive_feasibility as p05


def points() -> pd.DataFrame:
    return pd.DataFrame([
        {"match_id": "m1", "point_number": 1, "date": "2019-01-01", "surface": "Hard", "server": 1, "point_winner": 1, "player_1": "A", "player_2": "B", "first_serve": "6f27", "second_serve": None},
        {"match_id": "m1", "point_number": 2, "date": "2019-01-01", "surface": "Hard", "server": 2, "point_winner": 1, "player_1": "A", "player_2": "B", "first_serve": "6f17", "second_serve": None},
        {"match_id": "m2", "point_number": 1, "date": "2020-02-02", "surface": "Clay", "server": 2, "point_winner": 2, "player_1": "C", "player_2": "D", "first_serve": "5n", "second_serve": "6f08"},
        {"match_id": "m3", "point_number": 1, "date": "2023-03-03", "surface": "Grass", "server": 1, "point_winner": 2, "player_1": "E", "player_2": "F", "first_serve": "6f20", "second_serve": ""},
        {"match_id": "m4", "point_number": 1, "date": "2024-01-01", "surface": "Hard", "server": 1, "point_winner": 1, "player_1": "G", "player_2": "H", "first_serve": "6f39", "second_serve": None},
    ])


def result(): return p05.analyze_points(points())


def paths(root: Path): return tuple(root / value for value in ("summary.json", "state.csv", "depth.csv", "group.csv", "outcomes.csv"))


def write(result_value, destinations):
    p05.write_artifacts(result_value, summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4])


def resign(destinations):
    summary = json.loads(destinations[0].read_text(encoding="utf-8"))
    payloads = tuple(path.read_bytes() for path in destinations[1:])
    names = ("by_state", "by_depth", "by_group", "outcomes")
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, payloads)}
    summary["publication_fingerprint"] = p05._fingerprint(summary, payloads)
    destinations[0].write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_attempts_split_before_extraction_context_and_cache():
    seen = []
    def extractor(text, number, previous_attempt_was_fault=False):
        seen.append((text, number, previous_attempt_was_fault))
        return p05.parse_and_classify_initial_return_depth(text, number, previous_attempt_was_fault)
    source = p05.validate_source_points(points())
    development, counts = p05.split_development_before_parsing(source)
    attempts = p05.construct_attempts(development, extractor=extractor)
    assert counts["excluded_test_target_matches"] == 1
    assert len(attempts) == 5 and attempts.serve_number.value_counts().to_dict() == {1: 4, 2: 1}
    assert attempts.loc[attempts.serve_number.eq(2), "previous_attempt_was_fault"].tolist() == [True]
    assert all(text != "6f39" for text, _, _ in seen)
    assert attempts.attrs["cache_entries"] == len(set(seen))
    assert not attempts.duplicated(["match_id", "point_number", "serve_number"]).any()


@pytest.mark.parametrize("field,value", [("server", True), ("server", "1"), ("server", 1.0), ("point_winner", None), ("surface", "Carpet"), ("date", "01/02/2020")])
def test_source_domains_are_strict(field, value):
    frame = points(); frame[field] = frame[field].astype(object); frame.loc[0, field] = value
    with pytest.raises((TypeError, ValueError, p05.FeasibilityContractError)): p05.validate_source_points(frame)


def test_first_serve_must_be_substantive_and_attempt_key_is_checked():
    source = p05.validate_source_points(points()); development, _ = p05.split_development_before_parsing(source)
    broken = development.copy(); broken.loc[0, "first_serve"] = None
    with pytest.raises(p05.FeasibilityContractError, match="first_serve"): p05.construct_attempts(broken)
    duplicate = development.copy(); duplicate.loc[1, "point_number"] = 1
    with pytest.raises(p05.FeasibilityContractError): p05.construct_attempts(duplicate)


def test_states_depths_denominators_groups_and_outcomes_are_exact():
    actual = result()
    assert actual.summary["attempts"] == {"total": 5, "first": 4, "second": 1, "cache_entries": 5}
    assert actual.summary["state_counts"] == {"return_depth_observed": 3, "return_depth_unknown": 1, "return_depth_not_documented": 0, "unknown_initial_return": 0, "ineligible_censored": 1}
    assert actual.summary["depth_counts"] == {"7": 2, "8": 1, "9": 0}
    total = actual.by_depth.loc[actual.by_depth.group_type.eq("total")]
    assert total.attempts.tolist() == [2, 1, 0]
    assert total.observed_depth_denominator.tolist() == [3, 3, 3]
    assert total.end_to_end_coverage.tolist() == [0.4, 0.2, 0.0]
    outcome = actual.outcomes.loc[(actual.outcomes.group_type.eq("total")) & actual.outcomes.depth_code.eq("7")].iloc[0]
    assert (outcome.attempts, outcome.returner_point_wins, outcome.returner_point_losses, outcome.returner_point_win_rate) == (2, 1, 1, 0.5)
    assert actual.summary["test_seal"]["test_status"] == "sealed"
    assert actual.summary["test_seal"]["used_for_method_selection"] is False
    assert all(actual.summary["test_seal"][field] == 0 for field in p05.TEST_ZERO_FIELDS)


def test_wilson_matches_independent_reference():
    assert p05.wilson(5, 10) == pytest.approx((0.2365930905, 0.7634069095), abs=1e-10)
    assert p05.wilson(0, 0) == (None, None)


def test_result_validator_rejects_semantic_mutations_even_if_rehashed():
    actual = result(); depth = actual.by_depth.copy(); depth.loc[0, "attempts"] += 1
    mutated = p05._finalize(dict(actual.summary), actual.attempts, actual.by_state, depth, actual.by_group, actual.outcomes)
    with pytest.raises(p05.FeasibilityContractError): p05.validate_result(mutated)
    with pytest.raises(p05.FeasibilityContractError): p05.validate_result(replace(actual, summary={**actual.summary, "analysis_status": "not_available"}))


def test_serialization_is_deterministic_and_persisted_contract_is_valid(tmp_path):
    actual = result(); first = p05.serialize_artifacts(actual)
    assert p05.serialize_artifacts(actual) == first
    destinations = paths(tmp_path); write(actual, destinations)
    p05.verify_persisted_artifacts(summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4])


@pytest.mark.parametrize("target", ["status", "state", "state_depth_incompatible", "population", "attempts", "coverage", "depth", "code", "denominator", "group", "outcome", "wilson", "serve_number", "duplicate", "deleted", "deleted_non_total", "order", "column_order", "future_fold", "seal", "test_status", "method_selection", "not_available"])
def test_persisted_semantics_reject_rehashed_manipulations(target, tmp_path):
    destinations = paths(tmp_path); write(result(), destinations)
    if target == "status":
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["analysis_status"] = "unexpected"; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    elif target == "state":
        frame = pd.read_csv(destinations[1]); frame.loc[0, "classification_state"] = "unknown_initial_return"; frame.to_csv(destinations[1], index=False, lineterminator="\n")
    elif target == "state_depth_incompatible":
        frame = pd.read_csv(destinations[1]); frame.loc[0, "classification_state"] = "return_depth_not_documented"; frame.loc[0, "reason_code"] = "no_documented_depth_after_lateral_direction"; frame.to_csv(destinations[1], index=False, lineterminator="\n")
    elif target == "population":
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["population"]["development_point_rows"] += 1; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    elif target == "attempts":
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["attempts"]["total"] += 1; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    elif target == "coverage":
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["coverage"]["end_to_end_coverage"] = 0.9; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    elif target == "depth":
        frame = pd.read_csv(destinations[2]); frame.loc[0, "attempts"] += 1; frame.to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "code":
        frame = pd.read_csv(destinations[2]); frame.loc[0, "depth_code"] = 0; frame.to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "denominator":
        frame = pd.read_csv(destinations[2]); frame.loc[0, "observed_depth_denominator"] += 1; frame.to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "group":
        frame = pd.read_csv(destinations[3]); frame.loc[0, "attempts"] += 1; frame.to_csv(destinations[3], index=False, lineterminator="\n")
    elif target == "outcome":
        frame = pd.read_csv(destinations[4]); frame.loc[0, "returner_point_wins"] += 1; frame.to_csv(destinations[4], index=False, lineterminator="\n")
    elif target == "wilson":
        frame = pd.read_csv(destinations[4]); frame.loc[0, "wilson_95_lower"] = "0.0"; frame.to_csv(destinations[4], index=False, lineterminator="\n")
    elif target == "serve_number":
        frame = pd.read_csv(destinations[2]); frame.loc[0, "serve_number"] = 9; frame.to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "duplicate":
        frame = pd.read_csv(destinations[2]); frame.loc[len(frame)] = frame.iloc[0]; frame.to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "deleted":
        frame = pd.read_csv(destinations[2]); frame.iloc[1:].to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "deleted_non_total":
        frame = pd.read_csv(destinations[2]); frame.loc[~((frame.group_type.eq("serve_number")) & (frame.serve_number.eq(1)) & (frame.depth_code.eq(7)))].to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "order":
        frame = pd.read_csv(destinations[2]); frame.iloc[::-1].to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "column_order":
        frame = pd.read_csv(destinations[2]); frame.loc[:, list(reversed(frame.columns))].to_csv(destinations[2], index=False, lineterminator="\n")
    elif target == "future_fold":
        frame = pd.read_csv(destinations[3]); frame.loc[frame.group_type.eq("validation_fold"), "validation_fold"] = "validation_2024"; frame.to_csv(destinations[3], index=False, lineterminator="\n")
    elif target == "not_available":
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["analysis_status"] = "not_available"; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    elif target == "test_status":
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["test_seal"]["test_status"] = "open"; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    elif target == "method_selection":
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["test_seal"]["used_for_method_selection"] = True; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    else:
        summary = json.loads(destinations[0].read_text(encoding="utf-8")); summary["test_seal"][p05.TEST_ZERO_FIELDS[0]] = 1; destinations[0].write_text(json.dumps(summary), encoding="utf-8")
    resign(destinations)
    with pytest.raises(p05.FeasibilityContractError): p05.verify_persisted_artifacts(summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4])


@pytest.mark.parametrize("failure_at", range(5))
@pytest.mark.parametrize("with_originals", [False, True])
def test_atomic_publication_rolls_back_all_five_replacements(tmp_path, monkeypatch, failure_at, with_originals):
    destinations = paths(tmp_path); actual = result(); old = tuple(f"old-{index}".encode() for index in range(5))
    if with_originals:
        for path, payload in zip(destinations, old): path.write_bytes(payload)
    original, calls = p05.os.replace, []
    def fail_at(source, target):
        calls.append(target)
        if len(calls) == failure_at + 1: raise OSError("forced replace failure")
        return original(source, target)
    monkeypatch.setattr(p05.os, "replace", fail_at)
    with pytest.raises(OSError): p05.write_artifacts(actual, summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4])
    assert not list(tmp_path.glob(".*.tmp"))
    assert tuple(path.read_bytes() for path in destinations) == old if with_originals else not any(path.exists() for path in destinations)


@pytest.mark.parametrize("failure_at", range(5))
def test_atomic_publication_cleans_staging_failures(tmp_path, monkeypatch, failure_at):
    destinations = paths(tmp_path); original, calls = p05._stage, []
    def fail_at(path, payload):
        calls.append(path)
        if len(calls) == failure_at + 1: raise OSError("forced stage failure")
        return original(path, payload)
    monkeypatch.setattr(p05, "_stage", fail_at)
    with pytest.raises(OSError): p05.write_artifacts(result(), summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4])
    assert not list(tmp_path.glob(".*.tmp")) and not any(path.exists() for path in destinations)


def test_publication_rejects_payloads_not_derived_from_the_validated_result(tmp_path):
    destinations = paths(tmp_path)
    payloads = list(p05.serialize_artifacts(result())); payloads[2] = b"tampered"
    with pytest.raises(p05.FeasibilityContractError):
        p05.write_artifacts(result(), summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4], payloads=tuple(payloads))
    assert not any(path.exists() for path in destinations)


def test_not_available_has_only_headers_and_is_serializable():
    actual = p05.not_available_result(RuntimeError("unsafe\nmessage")); p05.validate_result(actual)
    payloads = p05.serialize_artifacts(actual)
    assert all(payload.count(b"\n") == 1 for payload in payloads[1:])
    assert b"NaN" not in payloads[0] and b"Infinity" not in payloads[0]


def test_frozen_contract_rejects_resigned_payload_changes(tmp_path, monkeypatch):
    destinations = paths(tmp_path); write(result(), destinations)
    payloads = {"summary": destinations[0].read_bytes(), "by_state": destinations[1].read_bytes(), "by_depth": destinations[2].read_bytes(), "by_group": destinations[3].read_bytes(), "outcomes": destinations[4].read_bytes()}
    monkeypatch.setattr(p05, "PUBLISHED_ARTIFACT_CONTRACT", {name: (len(payload), hashlib.sha256(payload).hexdigest().upper()) for name, payload in payloads.items()})
    summary = json.loads(payloads["summary"].decode("utf-8")); monkeypatch.setattr(p05, "PUBLISHED_PUBLICATION_FINGERPRINT", summary["publication_fingerprint"])
    p05.verify_persisted_artifacts(summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4], frozen_publication=True)
    frame = pd.read_csv(destinations[2]); frame.loc[0, "attempts"] += 1; frame.to_csv(destinations[2], index=False, lineterminator="\n"); resign(destinations)
    with pytest.raises(p05.FeasibilityContractError): p05.verify_persisted_artifacts(summary_path=destinations[0], by_state_path=destinations[1], by_depth_path=destinations[2], by_group_path=destinations[3], outcomes_path=destinations[4], frozen_publication=True)


@pytest.mark.integration
def test_persisted_integration_validates_only_artifacts_if_present():
    destinations = (p05.SUMMARY_PATH, p05.BY_STATE_PATH, p05.BY_DEPTH_PATH, p05.BY_GROUP_PATH, p05.OUTCOMES_PATH)
    if not all(path.exists() for path in destinations): pytest.skip("Artefactos P05 aun no generados.")
    p05.verify_persisted_artifacts(expected_population=p05.EXPECTED_REAL, frozen_publication=True)
    summary = json.loads(p05.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["analysis_status"] == "available_descriptive"
    assert summary["population"] == {key: value for key, value in p05.EXPECTED_REAL.items() if key in summary["population"]}
    assert summary["attempts"] | {"cache_entries": summary["attempts"]["cache_entries"]} == {"total": 1_426_863, "first": 1_035_760, "second": 391_103, "cache_entries": summary["attempts"]["cache_entries"]}
    assert summary["state_counts"] == {"return_depth_observed": 601_246, "return_depth_unknown": 0, "return_depth_not_documented": 229_225, "unknown_initial_return": 67_855, "ineligible_censored": 528_537}
    assert summary["depth_counts"] == {"7": 159_482, "8": 293_400, "9": 148_364}
    assert summary["test_seal"]["excluded_test_target_matches"] == 1_531
    assert all(summary["test_seal"][field] == 0 for field in p05.TEST_ZERO_FIELDS)


def test_cli_contract_is_prepared_without_invoking_data_read(tmp_path, monkeypatch):
    assert p05._performance_path(str(tmp_path / "outside-log.json")).is_absolute()
    with pytest.raises(p05.FeasibilityContractError): p05._performance_path(str(p05.ROOT / "inside.json"))
    monkeypatch.setattr(p05, "run_real_analysis", lambda: (_ for _ in ()).throw(AssertionError("No debe ejecutarse.")))
    with pytest.raises(p05.FeasibilityContractError): p05.main(["--performance-log", str(p05.ROOT / "inside.json")])
    source = open(p05.__file__, encoding="utf-8").read()
    assert "--performance-log" in source and "columns=list(SOURCE_COLUMNS)" in source
