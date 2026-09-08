"""Contrato sintético P06 descriptivo; no abre Parquet ni artefactos reales."""
from dataclasses import replace
import hashlib
import json
import math

import pandas as pd
import pytest

from src.analysis import return_shot_type_descriptive_feasibility as p06


def points() -> pd.DataFrame:
    """Fixture manual con los cuatro estados y test sellado."""
    return pd.DataFrame([
        {"match_id": "m1", "point_number": 1, "date": "2019-12-31", "surface": "Hard", "server": 1, "point_winner": 1, "player_1": "A", "player_2": "B", "first_serve": "6f", "second_serve": None},
        {"match_id": "m2", "point_number": 1, "date": "2020-01-02", "surface": "Clay", "server": 2, "point_winner": 2, "player_1": "C", "player_2": "D", "first_serve": "5n", "second_serve": "6b2"},
        {"match_id": "m3", "point_number": 1, "date": "2021-02-03", "surface": "Grass", "server": 1, "point_winner": 2, "player_1": "E", "player_2": "F", "first_serve": "6q", "second_serve": None},
        {"match_id": "m4", "point_number": 1, "date": "2022-03-04", "surface": "Hard", "server": 2, "point_winner": 1, "player_1": "G", "player_2": "H", "first_serve": "6?f", "second_serve": None},
        {"match_id": "m5", "point_number": 1, "date": "2023-12-31", "surface": "Clay", "server": 1, "point_winner": 2, "player_1": "I", "player_2": "J", "first_serve": "5n", "second_serve": "6r"},
        {"match_id": "m6", "point_number": 1, "date": "2024-01-01", "surface": "Hard", "server": 1, "point_winner": 1, "player_1": "K", "player_2": "L", "first_serve": "6s", "second_serve": None},
    ])


def result():
    return p06.analyze_points(points())


def independent_wilson(successes: int, trials: int, z: float = 1.959963984540054):
    """Referencia numérica independiente de la política de borde P06."""
    if trials == 0:
        return None, None
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    lower, upper = centre - spread, centre + spread
    if successes == 0:
        lower = 0.0
    if successes == trials:
        upper = 1.0
    return lower, upper


@pytest.fixture(scope="module")
def base_result():
    """Resultado sintético reutilizable; cada mutación opera sobre una copia."""
    return result()


@pytest.fixture(scope="module")
def base_payloads(base_result):
    return p06.serialize_artifacts(base_result)


def destinations(root):
    return tuple(root / name for name in ("summary.json", "state.csv", "type.csv", "group.csv", "outcomes.csv"))


def write(value, paths):
    p06.write_artifacts(value, summary_path=paths[0], by_state_path=paths[1], by_type_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])


def write_payloads(payloads, paths):
    """Prepara mutaciones semánticas sin repetir la ruta atómica ya probada."""
    for path, payload in zip(paths, payloads):
        path.write_bytes(payload)


def resign(paths):
    summary = json.loads(paths[0].read_text(encoding="utf-8")); payloads = tuple(path.read_bytes() for path in paths[1:]); names = ("by_state", "by_type", "by_group", "outcomes")
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, payloads)}
    summary["publication_fingerprint"] = p06._fingerprint(summary, payloads)
    paths[0].write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_source_is_sealed_before_extraction_and_uses_required_cache_key(monkeypatch):
    seen, validations = [], []
    original_validate = p06.validate_return_shot_type_classification
    def validating(classification):
        validations.append(classification)
        return original_validate(classification)
    monkeypatch.setattr(p06, "validate_return_shot_type_classification", validating)
    def extractor(text, number, previous_attempt_was_fault=False):
        seen.append((text, number, previous_attempt_was_fault))
        return p06.parse_and_classify_initial_return_shot_type(text, number, previous_attempt_was_fault)
    source = p06.validate_source_points(points()); development, counts = p06.split_development_before_parsing(source)
    attempts = p06.construct_attempts(development, extractor=extractor)
    assert counts["excluded_test_matches"] == 1
    assert len(attempts) == 7 and attempts.serve_number.value_counts().to_dict() == {1: 5, 2: 2}
    assert attempts.loc[attempts.serve_number.eq(2), "previous_attempt_was_fault"].tolist() == [True, True]
    assert all(text != "6s" for text, _, _ in seen)
    assert attempts.attrs["cache_entries"] == len(set(seen))
    assert len(validations) == len(set(seen))
    assert not attempts.duplicated(["match_id", "point_number", "serve_number"]).any()


@pytest.mark.parametrize("field,value", [("server", True), ("server", "1"), ("server", 1.0), ("point_winner", None), ("surface", "Carpet"), ("date", "01/02/2020")])
def test_source_domains_are_strict(field, value):
    frame = points(); frame[field] = frame[field].astype(object); frame.loc[0, field] = value
    with pytest.raises((TypeError, ValueError, p06.FeasibilityContractError)): p06.validate_source_points(frame)


def test_first_serve_is_required_substantive_and_attempt_key_is_unique():
    development, _ = p06.split_development_before_parsing(p06.validate_source_points(points()))
    broken = development.copy(); broken.loc[0, "first_serve"] = None
    with pytest.raises(p06.FeasibilityContractError): p06.construct_attempts(broken)
    duplicated = development.copy(); duplicated.loc[1, "point_number"] = 1; duplicated.loc[1, "match_id"] = "m1"
    with pytest.raises(p06.FeasibilityContractError): p06.construct_attempts(duplicated)


def test_summary_states_codes_families_denominators_and_outcomes_are_exact():
    actual = result()
    assert actual.summary["attempts"] == {"total": 7, "first": 5, "second": 2, "cache_entries": 6}
    assert actual.summary["state_counts"] == {"return_shot_type_observed": 3, "return_shot_type_unknown": 1, "unknown_initial_return": 1, "ineligible_censored": 2}
    assert actual.summary["observed_type_counts"] == {code: (1 if code in {"f", "b", "r"} else 0) for code in p06.TYPE_ORDER}
    assert actual.summary["family_counts"] == {family: (2 if family == "groundstroke" else 1 if family == "slice" else 0) for family in p06.FAMILY_ORDER}
    assert actual.summary["coverage"]["end_to_end_coverage"] == pytest.approx(3 / 7)
    total_codes = actual.by_type.loc[(actual.by_type.aggregation_level.eq("code")) & actual.by_type.serve_number.eq(0)]
    total_families = actual.by_type.loc[(actual.by_type.aggregation_level.eq("family")) & actual.by_type.serve_number.eq(0)]
    assert total_codes.shot_type_code.tolist() == list(p06.TYPE_ORDER)
    assert int(total_codes.attempts.sum()) == int(total_families.attempts.sum()) == 3
    outcome = actual.outcomes.loc[(actual.outcomes.aggregation_level.eq("code")) & actual.outcomes.group_type.eq("total") & actual.outcomes.shot_type_code.eq("f")].iloc[0]
    assert (outcome.attempts, outcome.returner_wins, outcome.returner_losses, outcome.returner_win_rate) == (1, 0, 1, 0.0)
    assert actual.summary["test_seal"]["test_status"] == "sealed"
    assert actual.summary["test_seal"]["used_for_method_selection"] is False
    assert all(actual.summary["test_seal"][field] == 0 for field in p06.TEST_ZERO_FIELDS)


@pytest.mark.parametrize("successes,trials", [(1, 3), (2, 3), (5, 10)])
def test_wilson_interior_values_match_independent_formula(successes, trials):
    assert p06.wilson(successes, trials) == pytest.approx(independent_wilson(successes, trials), abs=1e-15)


@pytest.mark.parametrize("trials", [1, 3, 10, 1_426_863])
def test_wilson_zero_successes_has_exact_zero_lower_bound(trials):
    lower, upper = p06.wilson(0, trials)
    assert lower == 0.0 and 0 <= upper <= 1


@pytest.mark.parametrize("trials", [1, 3, 10, 1_426_863])
def test_wilson_all_successes_has_exact_one_upper_bound(trials):
    lower, upper = p06.wilson(trials, trials)
    assert upper == 1.0 and 0 <= lower <= 1


def test_wilson_zero_trials_and_symmetry_are_explicit():
    assert p06.wilson(0, 0) == (None, None)
    lower, upper = p06.wilson(1, 3)
    complement_lower, complement_upper = p06.wilson(2, 3)
    assert lower == pytest.approx(1 - complement_upper, abs=1e-15)
    assert upper == pytest.approx(1 - complement_lower, abs=1e-15)


@pytest.mark.parametrize("successes,trials", [(-1, 3), (4, 3), (True, 3), (0.0, 3), ("0", 3), (None, 3), (0, True), (0, 3.0), (0, "3"), (0, None)])
def test_wilson_rejects_invalid_or_coercible_counts(successes, trials):
    with pytest.raises(p06.FeasibilityContractError):
        p06.wilson(successes, trials)


def test_wilson_rejects_material_deviation_and_preserves_safe_context(monkeypatch):
    monkeypatch.setattr(p06.math, "sqrt", lambda _value: 1.0)
    context = {"aggregation_level": "code", "group_type": "surface", "surface": "Hard", "shot_type_code": "f"}
    with pytest.raises(p06.FeasibilityContractError) as caught:
        p06.wilson(1, 3, context=context)
    message = str(caught.value)
    for fragment in ("successes=1", "trials=3", "rate=", "lower=", "upper=", "aggregation_level='code'", "group_type='surface'", "surface='Hard'", "shot_type_code='f'"):
        assert fragment in message


def test_result_validation_rebuilds_all_tables_and_rejects_rehashed_mutations():
    actual = result(); changed = actual.by_type.copy(); changed.loc[0, "attempts"] += 1
    mutated = p06._finalize(dict(actual.summary), actual.attempts, actual.by_state, changed, actual.by_group, actual.outcomes)
    with pytest.raises(p06.FeasibilityContractError): p06.validate_result(mutated)
    with pytest.raises(p06.FeasibilityContractError): p06.validate_result(replace(actual, summary={**actual.summary, "analysis_status": "not_available"}))


def test_serialization_is_deterministic_and_persisted_contract_is_valid(tmp_path, base_result):
    actual = base_result; assert p06.serialize_artifacts(actual) == p06.serialize_artifacts(actual)
    paths = destinations(tmp_path); write(actual, paths)
    p06.verify_persisted_artifacts(summary_path=paths[0], by_state_path=paths[1], by_type_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])


@pytest.mark.parametrize("target", ["status", "population", "attempts", "state", "q_observed", "reason", "code", "description", "family", "family_deleted", "aggregation_level", "count", "share", "coverage", "outcome", "outcome_denominator", "family_outcome", "wilson", "group", "group_type", "future_fold", "serve", "duplicate", "deleted", "order", "columns", "seal", "not_available"])
def test_persisted_contract_rejects_resigned_semantic_mutations(target, tmp_path, base_payloads):
    paths = destinations(tmp_path); write_payloads(base_payloads, paths)
    if target in {"status", "population", "attempts", "seal", "not_available"}:
        summary = json.loads(paths[0].read_text(encoding="utf-8"))
        if target == "status": summary["analysis_status"] = "other"
        elif target == "population": summary["population"]["development_matches"] += 1
        elif target == "attempts": summary["attempts"]["total"] += 1
        elif target == "seal": summary["test_seal"][p06.TEST_ZERO_FIELDS[0]] = 1
        else: summary["analysis_status"] = "not_available"
        paths[0].write_text(json.dumps(summary), encoding="utf-8")
    elif target in {"code", "description", "family", "family_deleted", "aggregation_level", "count", "share", "coverage", "duplicate", "deleted", "order", "columns", "serve"}:
        frame = pd.read_csv(paths[2])
        code_row = frame.index[frame.aggregation_level.eq("code")][0]
        if target == "code": frame.loc[code_row, "shot_type_code"] = "q"
        elif target == "description": frame.loc[code_row, "shot_type_description"] = "changed"
        elif target == "family": frame.loc[code_row, "shot_type_family"] = "slice"
        elif target == "aggregation_level": frame.loc[code_row, "aggregation_level"] = "other"
        elif target == "count": frame.loc[code_row, "attempts"] += 1
        elif target == "share": frame.loc[code_row, "share_among_observed_types"] = 0.9
        elif target == "coverage": frame.loc[code_row, "end_to_end_coverage"] = 0.9
        elif target == "duplicate": frame.loc[len(frame)] = frame.iloc[code_row]
        elif target == "deleted": frame = frame.drop(index=code_row)
        elif target == "family_deleted": frame = frame.drop(index=frame.index[frame.aggregation_level.eq("family")][0])
        elif target == "order": frame = frame.iloc[::-1]
        elif target == "columns": frame = frame.loc[:, list(reversed(frame.columns))]
        else: frame.loc[code_row, "serve_number"] = 9
        frame.to_csv(paths[2], index=False, lineterminator="\n")
    elif target in {"state", "q_observed", "reason"}:
        frame = pd.read_csv(paths[1])
        if target == "state": frame.loc[0, "classification_state"] = "return_shot_type_unknown"
        elif target == "q_observed": frame.loc[0, "reason_code"] = "documented_unknown_return_shot_type_q"
        else: frame.loc[0, "reason_code"] = "censored_ace"
        frame.to_csv(paths[1], index=False, lineterminator="\n")
    elif target in {"group", "group_type", "future_fold"}:
        frame = pd.read_csv(paths[3])
        if target == "group": frame.loc[0, "observed_type_attempts"] += 1
        elif target == "group_type": frame.loc[0, "group_type"] = "other"
        else:
            fold_row = frame.index[frame.group_type.eq("validation_fold")][0]
            frame.loc[fold_row, "validation_fold"] = "validation_2024"
        frame.to_csv(paths[3], index=False, lineterminator="\n")
    else:
        frame = pd.read_csv(paths[4])
        if target == "outcome": frame.loc[0, "returner_wins"] += 1
        elif target == "outcome_denominator": frame.loc[0, "attempts"] += 1
        elif target == "family_outcome": frame.loc[frame.index[frame.aggregation_level.eq("family")][0], "attempts"] += 1
        else: frame.loc[0, "wilson_95_lower"] = 0.123
        frame.to_csv(paths[4], index=False, lineterminator="\n")
    resign(paths)
    with pytest.raises((ValueError, TypeError, p06.FeasibilityContractError)): p06.verify_persisted_artifacts(summary_path=paths[0], by_state_path=paths[1], by_type_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])


@pytest.mark.parametrize("field", p06.TEST_ZERO_FIELDS)
def test_persisted_contract_rejects_each_nonzero_test_counter(field, tmp_path, base_payloads):
    paths = destinations(tmp_path); write_payloads(base_payloads, paths)
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    summary["test_seal"][field] = 1
    paths[0].write_text(json.dumps(summary), encoding="utf-8")
    resign(paths)
    with pytest.raises(p06.FeasibilityContractError):
        p06.verify_persisted_artifacts(summary_path=paths[0], by_state_path=paths[1], by_type_path=paths[2], by_group_path=paths[3], outcomes_path=paths[4])


@pytest.mark.parametrize("failure_at", range(5))
@pytest.mark.parametrize("with_originals", [False, True])
def test_atomic_publication_rolls_back_all_five_replacements(tmp_path, monkeypatch, failure_at, with_originals, base_result):
    paths, actual = destinations(tmp_path), base_result; old = tuple(f"old-{index}".encode() for index in range(5))
    if with_originals:
        for path, payload in zip(paths, old): path.write_bytes(payload)
    original, calls = p06.os.replace, []
    def fail_once(source, target):
        calls.append(target)
        if len(calls) == failure_at + 1: raise OSError("forced replacement failure")
        return original(source, target)
    monkeypatch.setattr(p06.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(p06.os, "replace", fail_once)
    with pytest.raises(OSError): write(actual, paths)
    assert not list(tmp_path.glob(".*.tmp"))
    if with_originals: assert tuple(path.read_bytes() for path in paths) == old
    else: assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("failure_at", range(5))
def test_atomic_publication_cleans_each_staging_failure(tmp_path, monkeypatch, failure_at, base_result):
    paths, original, calls = destinations(tmp_path), p06._stage, []
    def fail_once(path, payload):
        calls.append(path)
        if len(calls) == failure_at + 1: raise OSError("forced staging failure")
        return original(path, payload)
    monkeypatch.setattr(p06.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(p06, "_stage", fail_once)
    with pytest.raises(OSError): write(base_result, paths)
    assert not list(tmp_path.glob(".*.tmp")) and not any(path.exists() for path in paths)


def test_not_available_is_header_only_and_contains_no_nonfinite_values():
    actual = p06.not_available_result(RuntimeError("unsafe\nmessage")); p06.validate_result(actual)
    assert all(payload.count(b"\n") == 1 for payload in p06.serialize_artifacts(actual)[1:])
    assert b"NaN" not in p06.serialize_artifacts(actual)[0] and b"Infinity" not in p06.serialize_artifacts(actual)[0]


@pytest.mark.parametrize("failure_stage", ["read_source_points", "analyze_points"])
def test_main_publishes_only_header_artifacts_and_nonzero_exit_after_single_failure(tmp_path, monkeypatch, failure_stage):
    captured = []
    monkeypatch.setattr(p06, "write_artifacts", lambda value: captured.append(value))
    if failure_stage == "read_source_points":
        monkeypatch.setattr(p06, "read_source_points", lambda: (_ for _ in ()).throw(RuntimeError("read\nfailed")))
    else:
        original_analyze = p06.analyze_points
        monkeypatch.setattr(p06, "read_source_points", points)
        def after_aggregates(frame, **kwargs):
            original_analyze(frame)
            raise RuntimeError("aggregate\nfailed")
        monkeypatch.setattr(p06, "analyze_points", after_aggregates)
    with pytest.raises(SystemExit) as caught:
        p06.main(["--performance-log", str(tmp_path / f"{failure_stage}.json")])
    assert caught.value.code == 1 and len(captured) == 1
    actual = captured[0]
    assert actual.summary["analysis_status"] == "not_available"
    assert actual.summary["failure"] == {"stage": failure_stage, "type": "RuntimeError", "message": "read failed" if failure_stage == "read_source_points" else "aggregate failed"}
    assert all(frame.empty for frame in (actual.by_state, actual.by_type, actual.by_group, actual.outcomes))
    assert (tmp_path / f"{failure_stage}.json").exists()


def test_cli_contract_is_prepared_without_calling_parquet(tmp_path, monkeypatch):
    assert p06._performance_path(str(tmp_path / "p06.json")).is_absolute()
    with pytest.raises(p06.FeasibilityContractError): p06._performance_path(str(p06.ROOT / "inside.json"))
    monkeypatch.setattr(p06, "run_real_analysis", lambda: (_ for _ in ()).throw(AssertionError("No debe ejecutarse.")))
    with pytest.raises(p06.FeasibilityContractError): p06.main(["--performance-log", str(p06.ROOT / "inside.json")])
    source = open(p06.__file__, encoding="utf-8").read()
    assert source.count("read_parquet(") == 1 and "columns=list(SOURCE_COLUMNS)" in source


@pytest.mark.integration
def test_persisted_integration_is_deselected_or_skips_without_parquet():
    paths = (p06.SUMMARY_PATH, p06.BY_STATE_PATH, p06.BY_TYPE_PATH, p06.BY_GROUP_PATH, p06.OUTCOMES_PATH)
    if not all(path.exists() for path in paths): pytest.skip("Artefactos P06 aún no generados.")
    p06.verify_persisted_artifacts()
    summary = json.loads(p06.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["analysis_status"] == "available_descriptive"
    assert summary["population"] == {"source_rows_read": 1_280_408, "source_matches": 7_524, "source_players": 1_002, "development_point_rows": 1_035_760, "development_matches": 5_993, "development_servers": 870, "excluded_test_matches": 1_531}
    assert summary["attempts"] == {"total": 1_426_863, "first": 1_035_760, "second": 391_103, "cache_entries": 490_548}
    assert summary["state_counts"] == {"return_shot_type_observed": 881_717, "return_shot_type_unknown": 1_691, "unknown_initial_return": 14_918, "ineligible_censored": 528_537}
    assert summary["coverage"]["end_to_end_coverage"] == pytest.approx(0.6179408955169488)
    assert summary["test_seal"]["excluded_test_matches"] == 1_531
    assert all(summary["test_seal"][field] == 0 for field in p06.TEST_ZERO_FIELDS)
