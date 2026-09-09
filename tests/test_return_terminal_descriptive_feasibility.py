"""Contrato sintético de P07; no abre datos ni artefactos permanentes."""
from dataclasses import replace
import csv
import hashlib
import inspect
import io
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import return_terminal_descriptive_feasibility as p07


SYNTHETIC_UPSTREAM = {
    "extractor_commit": "synthetic",
    "extractor_sha256": "A" * 64,
    "chronological_summary_sha256": "B" * 64,
}


FROZEN_P07_ARTIFACTS = {
    "summary": (6316, "ECD31944D7253A48C468D4C2AC6A8C1004413431F332DA7E66317CB05B34B096"),
    "by_state": (44616, "1AA13A0A1F6DE890670514E11799A75FC2DDF357E366E5D8397A34428105CE77"),
    "by_terminal": (2925, "4FA3EEE4B46800AAF24629C8095063416F10C5C9169EE42C8A1AB2B58C1EED13"),
    "by_group": (2853, "AA42694A859E38C7D5A790B10C3F983034F76173ABF3C65399BB8B0B2EC012E2"),
    "consistency": (18977, "D099EEF9E110DD0D0FC8A568D28AD9B66220D473837B3BE13E9837C01CE18FD5"),
}
FROZEN_P07_FINGERPRINT = "C96BE1AFA442B22DB0FEE51C4264FA5347A3B6DDA2295EF9B1720A58994D27C1"


def points() -> pd.DataFrame:
    """Fixture manual: tres terminales, abstenciones y una fila sellada."""
    return pd.DataFrame([
        {"match_id": "m1", "point_number": 1, "date": "2019-12-31", "surface": "Hard", "server": 1, "point_winner": 2, "player_1": "A", "player_2": "B", "first_serve": "6f*", "second_serve": None},
        {"match_id": "m2", "point_number": 1, "date": "2020-01-02", "surface": "Clay", "server": 2, "point_winner": 2, "player_1": "C", "player_2": "D", "first_serve": "5n", "second_serve": "6f#"},
        {"match_id": "m3", "point_number": 1, "date": "2021-02-03", "surface": "Grass", "server": 1, "point_winner": 2, "player_1": "E", "player_2": "F", "first_serve": "6f2d@", "second_serve": None},
        {"match_id": "m4", "point_number": 1, "date": "2022-03-04", "surface": "Hard", "server": 2, "point_winner": 1, "player_1": "G", "player_2": "H", "first_serve": "6f", "second_serve": None},
        {"match_id": "m5", "point_number": 1, "date": "2023-12-31", "surface": "Clay", "server": 1, "point_winner": 1, "player_1": "I", "player_2": "J", "first_serve": "6*", "second_serve": None},
        {"match_id": "m6", "point_number": 1, "date": "2024-01-01", "surface": "Hard", "server": 1, "point_winner": 1, "player_1": "K", "player_2": "L", "first_serve": "4b*", "second_serve": None},
    ])


def result():
    return p07.analyze_points(points(), upstream_contract=SYNTHETIC_UPSTREAM)


@pytest.fixture(scope="module")
def base_result():
    return result()


@pytest.fixture(scope="module")
def base_payloads(base_result):
    return p07.serialize_artifacts(base_result)


def independent_wilson(successes: int, trials: int, z: float = 1.959963984540054):
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


def paths(root):
    return tuple(root / name for name in ("summary.json", "state.csv", "terminal.csv", "group.csv", "consistency.csv"))


def write(value, destination):
    p07.write_artifacts(value, summary_path=destination[0], by_state_path=destination[1], by_terminal_path=destination[2], by_group_path=destination[3], consistency_path=destination[4])


def raw_csv(payload):
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    return rows[0], rows[1:]


def raw_csv_bytes(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header); writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def legacy_surface_payload(name, payload):
    if name == "by_terminal":
        return payload
    header, rows = raw_csv(payload)
    keys, ranks = p07.CSV_SORT_SPECS[name]
    legacy_ranks = dict(ranks)
    legacy_ranks["surface"] = {"Clay": 0, "Grass": 1, "Hard": 2}
    rows = sorted(rows, key=lambda row: p07._canonical_sort_key(dict(zip(header, row)), keys, legacy_ranks))
    return raw_csv_bytes(header, rows)


def repair_fixture(monkeypatch, tmp_path, base_payloads):
    reports, tables = tmp_path / "reports", tmp_path / "reports" / "tables"
    summary_path = reports / "return_terminal_descriptive_feasibility_summary.json"
    table_paths = {
        "by_state": tables / "return_terminal_descriptive_feasibility_by_state.csv",
        "by_terminal": tables / "return_terminal_descriptive_feasibility_by_terminal.csv",
        "by_group": tables / "return_terminal_descriptive_feasibility_by_group.csv",
        "consistency": tables / "return_terminal_descriptive_feasibility_consistency.csv",
    }
    tables.mkdir(parents=True)
    current_summary = json.loads(base_payloads[0].decode("utf-8"))
    legacy_tables = {name: legacy_surface_payload(name, payload) for name, payload in zip(("by_state", "by_terminal", "by_group", "consistency"), base_payloads[1:])}
    current_summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in legacy_tables.items()}
    current_summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in legacy_tables.items()}
    current_summary["publication_fingerprint"] = p07._fingerprint(current_summary, tuple(legacy_tables[name] for name in ("by_state", "by_terminal", "by_group", "consistency")))
    summary_payload = p07._summary_bytes(current_summary)
    summary_path.write_bytes(summary_payload)
    for name, path in table_paths.items(): path.write_bytes(legacy_tables[name])
    monkeypatch.setattr(p07, "REPORTS_DIR", reports)
    monkeypatch.setattr(p07, "SUMMARY_PATH", summary_path)
    monkeypatch.setattr(p07, "BY_STATE_PATH", table_paths["by_state"])
    monkeypatch.setattr(p07, "BY_TERMINAL_PATH", table_paths["by_terminal"])
    monkeypatch.setattr(p07, "BY_GROUP_PATH", table_paths["by_group"])
    monkeypatch.setattr(p07, "CONSISTENCY_PATH", table_paths["consistency"])
    legacy_payloads = {"summary": summary_payload, **legacy_tables}
    monkeypatch.setattr(p07, "LEGACY_REPAIR_INPUT_SHA256", {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in legacy_payloads.items()})
    monkeypatch.setattr(p07, "LEGACY_REPAIR_INPUT_FINGERPRINT", current_summary["publication_fingerprint"])
    return {"summary": summary_path, **table_paths}, legacy_payloads, current_summary


def test_source_is_sealed_before_extraction_and_second_attempt_uses_documented_fault_context_only():
    seen = []
    source = p07.validate_source_points(points())
    development, counts = p07.split_development_before_parsing(source)
    def extractor(text, number, previous_attempt_was_fault=False):
        seen.append((text, number, previous_attempt_was_fault))
        return p07.parse_and_classify_initial_return_terminal(text, number, previous_attempt_was_fault)
    attempts = p07.construct_attempts(development, extractor=extractor)
    assert counts["excluded_test_matches"] == 1
    assert len(attempts) == 6
    assert attempts.serve_number.value_counts().to_dict() == {1: 5, 2: 1}
    assert attempts.loc[attempts.serve_number.eq(2), "previous_attempt_was_fault"].tolist() == [True]
    assert attempts.loc[attempts.serve_number.eq(1), "second_serve_context"].eq(p07.FIRST_SERVE_CONTEXT).all()
    assert attempts.loc[attempts.serve_number.eq(2), "second_serve_context"].tolist() == ["documented_first_service_fault"]
    assert all(text != points().iloc[-1].first_serve for text, _, _ in seen)
    broken = development.copy(); broken.loc[broken.match_id.eq("m1"), "second_serve"] = "6f*"
    corrected = p07.construct_attempts(broken)
    contextual_second = corrected.loc[(corrected.match_id.eq("m1")) & (corrected.serve_number.eq(2))].iloc[0]
    assert len(corrected) == 7
    assert not bool(contextual_second.previous_attempt_was_fault)
    assert contextual_second.second_serve_context == "second_serve_without_documented_first_fault"
    assert contextual_second.classification_state == "documented_return_winner"


def test_second_fault_is_double_fault_only_when_first_fault_is_documented():
    source = points().iloc[[1, 4, 5]].copy()
    source.loc[source.match_id.eq("m2"), "second_serve"] = "6n"
    source.loc[source.match_id.eq("m5"), "second_serve"] = "6n"
    development, _ = p07.split_development_before_parsing(p07.validate_source_points(source))
    attempts = p07.construct_attempts(development)
    documented = attempts.loc[(attempts.match_id.eq("m2")) & (attempts.serve_number.eq(2))].iloc[0]
    undocumented = attempts.loc[(attempts.match_id.eq("m5")) & (attempts.serve_number.eq(2))].iloc[0]
    assert bool(documented.previous_attempt_was_fault)
    assert documented.second_serve_context == "documented_first_service_fault"
    assert documented.reason_code == "censored_double_fault"
    assert not bool(undocumented.previous_attempt_was_fault)
    assert undocumented.second_serve_context == "second_serve_without_documented_first_fault"
    assert undocumented.reason_code == "censored_service_fault"
    assert documented.terminal_serve_outcome == "double_fault"
    assert undocumented.terminal_serve_outcome == "service_fault"
    assert attempts.attrs["cache_entries"] == 4


def test_second_context_categories_are_exhaustive_without_changing_p07_outputs():
    source = points().iloc[[0, 1, 4, 5]].copy()
    source.loc[source.match_id.eq("m1"), "second_serve"] = "6f*"
    source.loc[source.match_id.eq("m5"), "second_serve"] = "6f*"
    development, _ = p07.split_development_before_parsing(p07.validate_source_points(source))
    attempts = p07.construct_attempts(development)
    by_state, by_terminal, by_group, consistency = (p07.build_by_state(attempts), p07.build_by_terminal(attempts), p07.build_by_group(attempts), p07.build_consistency(attempts))
    contexts = by_group.loc[by_group.group_type.eq("second_serve_context")]
    assert set(contexts.second_serve_context) == set(p07.SECOND_SERVE_CONTEXT_ORDER)
    assert int(contexts.attempts.sum()) == int(attempts.serve_number.eq(2).sum()) == 3
    assert int(contexts.eligible_terminal_attempts.sum()) == int(attempts.loc[attempts.serve_number.eq(2), "eligible_for_terminal_comparison"].sum())
    p07.validate_reconciliations(attempts, by_state, by_terminal, by_group, consistency)


@pytest.mark.parametrize("field,value", [
    ("server", True), ("server", "1"), ("server", 1.0), ("point_winner", None),
    ("surface", "Carpet"), ("date", "01/02/2020"), ("point_number", 1.0),
])
def test_source_domains_are_strict(field, value):
    frame = points(); frame[field] = frame[field].astype(object); frame.loc[0, field] = value
    with pytest.raises((TypeError, ValueError, p07.FeasibilityContractError)):
        p07.validate_source_points(frame)


def test_summary_states_terminal_outcomes_denominators_and_test_seal_are_exact(base_result):
    actual = base_result
    assert actual.summary["population"] == {
        "source_rows_read": 6, "source_matches": 6, "source_players": 12,
        "development_point_rows": 5, "development_matches": 5,
        "development_servers": 5, "excluded_test_matches": 1,
    }
    assert actual.summary["attempts"] == {"total": 6, "first": 5, "second": 1, "cache_entries": 6}
    assert actual.summary["second_serve_context"] == {
        "second_attempts": 1,
        "documented_first_service_fault": 1,
        "second_serve_without_documented_first_fault": 0,
        "without_documented_first_fault_proportion": 0.0,
    }
    assert actual.summary["state_counts"] == {
        "documented_return_winner": 1, "documented_return_forced_error": 1,
        "documented_return_unforced_error": 1, "return_terminal_not_documented": 1,
        "unknown_initial_return": 0, "ineligible_censored": 2,
    }
    assert actual.summary["terminal_class_counts"] == {"winner": 1, "forced_error": 1, "unforced_error": 1}
    assert actual.summary["terminal_coverage"] == {"eligible_terminal_attempts": 3, "all_attempts": 6, "rate": 0.5}
    assert actual.summary["consistency_audit"] == {"terminal_attempts": 3, "discordances": 1, "interpretation": "diagnostic_only_not_tactical_effectiveness"}
    total = actual.consistency.loc[(actual.consistency.group_type == "total") & (actual.consistency.aggregation_level == "terminal_class")]
    assert total.set_index("terminal_class").loc["winner", ["attempts", "concordant", "discordant"]].tolist() == [1, 1, 0]
    assert total.set_index("terminal_class").loc["forced_error", ["attempts", "concordant", "discordant"]].tolist() == [1, 1, 0]
    assert total.set_index("terminal_class").loc["unforced_error", ["attempts", "concordant", "discordant"]].tolist() == [1, 0, 1]
    seal = actual.summary["test_seal"]
    assert seal["test_status"] == "sealed" and seal["used_for_method_selection"] is False
    assert seal["excluded_test_matches"] == 1 and all(seal[field] == 0 for field in p07.TEST_ZERO_FIELDS)


@pytest.mark.parametrize("successes,trials", [(0, 1), (0, 3), (1, 3), (2, 3), (3, 3), (5, 10)])
def test_wilson_matches_independent_formula_and_exact_boundaries(successes, trials):
    lower, upper = p07.wilson(successes, trials)
    expected = independent_wilson(successes, trials)
    assert (lower, upper) == pytest.approx(expected, abs=1e-15)
    assert 0 <= lower <= successes / trials <= upper <= 1


@pytest.mark.parametrize("successes,trials", [(-1, 3), (4, 3), (True, 3), (0.0, 3), ("0", 3), (0, True), (0, 3.0)])
def test_wilson_rejects_invalid_counts(successes, trials):
    with pytest.raises(p07.FeasibilityContractError):
        p07.wilson(successes, trials)


def test_wilson_preserves_sanitized_group_context_on_material_failure(monkeypatch):
    monkeypatch.setattr(p07.math, "sqrt", lambda _value: 1.0)
    with pytest.raises(p07.FeasibilityContractError) as caught:
        p07.wilson(1, 3, context={"aggregation_level": "terminal_class", "group_type": "surface", "surface": "Hard", "terminal_class": "winner"})
    for fragment in ("successes=1", "trials=3", "aggregation_level='terminal_class'", "group_type='surface'", "surface='Hard'"):
        assert fragment in str(caught.value)


def test_result_validation_rejects_semantic_mutations_even_when_rehashed(base_result):
    actual = base_result
    changed = actual.by_group.copy(); changed.loc[changed.group_type.eq("total"), "winner_attempts"] += 1
    with pytest.raises(p07.FeasibilityContractError):
        p07.validate_result(replace(actual, by_group=changed))
    with pytest.raises(p07.FeasibilityContractError):
        p07.validate_result(replace(actual, summary={**actual.summary, "analysis_status": "not_available"}))


@pytest.mark.parametrize("target", ["missing", "unknown", "count", "first", "proportion", "invented_fault"])
def test_context_contract_rejects_semantic_mutations(target, base_result):
    actual = base_result
    changed = actual.by_group.copy()
    summary = dict(actual.summary)
    if target == "missing":
        changed = changed.loc[~((changed.group_type == "second_serve_context") & (changed.second_serve_context == p07.SECOND_SERVE_CONTEXT_ORDER[1]))].reset_index(drop=True)
    elif target == "unknown":
        changed.loc[changed.group_type.eq("second_serve_context"), "second_serve_context"] = "unknown_context"
    elif target == "count":
        changed.loc[changed.group_type.eq("second_serve_context"), "attempts"] += 1
    elif target == "first":
        changed.loc[changed.group_type.eq("serve_number") & changed.serve_number.eq(1), "second_serve_context"] = "documented_first_service_fault"
    elif target == "proportion":
        summary["second_serve_context"] = {**summary["second_serve_context"], "without_documented_first_fault_proportion": 1.0}
    else:
        attempts = actual.attempts.copy()
        attempts.loc[attempts.serve_number.eq(2), "previous_attempt_was_fault"] = False
        with pytest.raises(p07.FeasibilityContractError):
            p07.validate_result(replace(actual, attempts=attempts))
        return
    with pytest.raises(p07.FeasibilityContractError):
        p07.validate_result(replace(actual, by_group=changed, summary=summary))


def test_serialization_is_deterministic_atomic_and_persisted_contract_is_valid(tmp_path, base_result, base_payloads):
    assert p07.serialize_artifacts(base_result) == base_payloads
    empty = tuple(pd.DataFrame(columns=columns) for columns in (p07.BY_STATE_COLUMNS, p07.BY_TERMINAL_COLUMNS, p07.BY_GROUP_COLUMNS, p07.CONSISTENCY_COLUMNS))
    unavailable = p07._finalize({"analysis_status": "not_available", "reason_codes": ["execution_failed"], "failure": {"stage": "synthetic", "type": "RuntimeError", "message": "synthetic failure"}, "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": None, **{field: 0 for field in p07.TEST_ZERO_FIELDS}}}, pd.DataFrame(), *empty)
    destination = paths(tmp_path); write(unavailable, destination)
    p07.verify_persisted_artifacts(summary_path=destination[0], by_state_path=destination[1], by_terminal_path=destination[2], by_group_path=destination[3], consistency_path=destination[4])
    assert not list(tmp_path.glob("*.tmp"))


def test_persisted_order_uses_serialization_rank_precedence_after_csv_roundtrip(tmp_path, base_payloads):
    """La versión previa intercalaba las claves y rechazaba este CSV válido."""
    destination = paths(tmp_path)
    for path, payload in zip(destination, base_payloads): path.write_bytes(payload)
    by_state = p07._read_csv(destination[1], p07.BY_STATE_COLUMNS)
    by_terminal = p07._read_csv(destination[2], p07.BY_TERMINAL_COLUMNS)
    by_group = p07._read_csv(destination[3], p07.BY_GROUP_COLUMNS)
    consistency = p07._read_csv(destination[4], p07.CONSISTENCY_COLUMNS)
    state_ranks = {value: index for index, value in enumerate(p07.STATE_ORDER)}
    def previous_interleaved_key(row):
        values = []
        for column in ("group_type", "serve_number", "surface", "derived_period", "validation_fold", "classification_state", "reason_code"):
            value = getattr(row, column)
            if column == "group_type": values.append({"total": 0, "serve_number": 1, "surface": 2, "derived_period": 3, "validation_fold": 4, "second_serve_context": 5}.get(value, 6))
            if column == "classification_state": values.append(state_ranks.get(value, len(state_ranks)))
            values.append("" if pd.isna(value) else str(value))
        return tuple(values)
    previous_keys = [previous_interleaved_key(row) for row in by_state.itertuples(index=False)]
    assert previous_keys != sorted(previous_keys)
    assert p07._persisted_frames_are_canonically_ordered(by_state, by_terminal, by_group, consistency)
    p07.verify_persisted_artifacts(summary_path=destination[0], by_state_path=destination[1], by_terminal_path=destination[2], by_group_path=destination[3], consistency_path=destination[4])


def test_surface_period_and_fold_orders_are_contractual_not_alphabetical_after_roundtrip(tmp_path, base_payloads):
    destination = paths(tmp_path)
    for path, payload in zip(destination, base_payloads): path.write_bytes(payload)
    for path, columns in ((destination[1], p07.BY_STATE_COLUMNS), (destination[3], p07.BY_GROUP_COLUMNS), (destination[4], p07.CONSISTENCY_COLUMNS)):
        frame = p07._read_csv(path, columns)
        assert list(frame.loc[frame.group_type.eq("surface"), "surface"].drop_duplicates()) == list(p07.ALLOWED_SURFACES)
        assert list(frame.loc[frame.group_type.eq("derived_period"), "derived_period"].drop_duplicates()) == list(p07.PERIOD_ORDER)
        assert list(frame.loc[frame.group_type.eq("validation_fold"), "validation_fold"].drop_duplicates()) == list(p07.FOLD_ORDER)


def test_artifact_only_repair_preserves_literal_rows_and_updates_only_publication_contract(monkeypatch, tmp_path, base_payloads):
    destination, before, legacy_summary = repair_fixture(monkeypatch, tmp_path, base_payloads)
    result = p07.reserialize_persisted_artifacts_canonically()
    after = {name: path.read_bytes() for name, path in destination.items()}
    assert result["reordered_artifacts"] == ["by_state", "by_group", "consistency"]
    assert after["by_terminal"] == before["by_terminal"]
    for name in ("by_state", "by_group", "consistency"):
        old_header, old_rows = raw_csv(before[name]); new_header, new_rows = raw_csv(after[name])
        assert new_header == old_header and len(new_rows) == len(old_rows)
        assert p07._raw_row_multiset_hash(new_rows) == p07._raw_row_multiset_hash(old_rows)
        assert any(old != new for old, new in zip(old_rows, new_rows))
    summary = json.loads(after["summary"].decode("utf-8"))
    assert p07._analytical_summary(summary) == p07._analytical_summary(legacy_summary)
    assert summary["artifact_repair"] == {
        "mode": "artifact_only_canonical_reserialization", "real_analysis_rerun": False,
        "source_publication_fingerprint": legacy_summary["publication_fingerprint"],
        "historical_real_analysis_attempts": 2, "historical_run_2_status": "publication_verification_failed",
        "reordered_artifacts": ["by_state", "by_group", "consistency"],
        "analytical_values_changed": False, "reason": "canonical_surface_order",
    }
    p07.verify_persisted_artifacts(summary_path=destination["summary"], by_state_path=destination["by_state"], by_terminal_path=destination["by_terminal"], by_group_path=destination["by_group"], consistency_path=destination["consistency"])
    with pytest.raises(p07.FeasibilityContractError, match="hashes legacy"):
        p07.reserialize_persisted_artifacts_canonically()
    assert not list((tmp_path / "reports").rglob("*.tmp"))


def test_artifact_only_repair_rejects_nonlegacy_input_before_writing(monkeypatch, tmp_path, base_payloads):
    destination, before, _ = repair_fixture(monkeypatch, tmp_path, base_payloads)
    destination["by_group"].write_bytes(before["by_group"] + b" ")
    with pytest.raises(p07.FeasibilityContractError, match="hashes legacy"):
        p07.reserialize_persisted_artifacts_canonically()
    assert destination["by_group"].read_bytes() == before["by_group"] + b" "


@pytest.mark.parametrize("failure_mode,index", [("stage", value) for value in range(1, 6)] + [("replace", value) for value in range(1, 6)] + [("verify", 0)])
def test_artifact_only_repair_rolls_back_every_transaction_failure(monkeypatch, tmp_path, base_payloads, failure_mode, index):
    destination, before, _ = repair_fixture(monkeypatch, tmp_path, base_payloads)
    if failure_mode == "stage":
        original, calls = p07._stage, []
        def fail_stage(path, payload):
            calls.append(path)
            if len(calls) == index: raise OSError("stage repair failure")
            return original(path, payload)
        monkeypatch.setattr(p07, "_stage", fail_stage)
    elif failure_mode == "replace":
        original, calls = p07.os.replace, []
        def fail_replace(source, target):
            calls.append(target)
            if len(calls) == index: raise OSError("replace repair failure")
            return original(source, target)
        monkeypatch.setattr(p07.os, "replace", fail_replace)
    else:
        monkeypatch.setattr(p07, "verify_persisted_artifacts", lambda **_kwargs: (_ for _ in ()).throw(p07.FeasibilityContractError("verify repair failure")))
    with pytest.raises((OSError, p07.FeasibilityContractError)):
        p07.reserialize_persisted_artifacts_canonically()
    assert {name: path.read_bytes() for name, path in destination.items()} == before
    assert not list((tmp_path / "reports").rglob("*.tmp"))


def test_artifact_only_repair_isolated_from_source_reading_and_rejects_foreign_paths(monkeypatch, tmp_path, base_payloads):
    destination, _, _ = repair_fixture(monkeypatch, tmp_path, base_payloads)
    source = inspect.getsource(p07.reserialize_persisted_artifacts_canonically)
    assert "read_parquet" not in source and "run_real_analysis" not in source and "parse_and_classify" not in source
    with pytest.raises(p07.FeasibilityContractError, match="fuera de reports"):
        p07.reserialize_persisted_artifacts_canonically(summary_path=tmp_path / "foreign.json")
    assert destination["summary"].exists()


@pytest.mark.parametrize("target", ["missing", "duplicate", "unknown", "columns"])
def test_persisted_key_contract_rejects_missing_duplicate_unknown_or_reordered_columns(target, tmp_path, base_payloads):
    destination = paths(tmp_path)
    for path, payload in zip(destination, base_payloads): path.write_bytes(payload)
    frame = pd.read_csv(destination[3])
    if target == "missing":
        frame = frame.iloc[1:].reset_index(drop=True)
    elif target == "duplicate":
        frame = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
    elif target == "unknown":
        frame.loc[0, "group_type"] = "unknown_group"
    else:
        frame = frame.loc[:, list(reversed(frame.columns))]
    frame.to_csv(destination[3], index=False, lineterminator="\n")
    summary = json.loads(destination[0].read_text(encoding="utf-8")); table_payloads = tuple(path.read_bytes() for path in destination[1:])
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_state", "by_terminal", "by_group", "consistency"), table_payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(("by_state", "by_terminal", "by_group", "consistency"), table_payloads)}
    summary["publication_fingerprint"] = p07._fingerprint(summary, table_payloads)
    destination[0].write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    with pytest.raises(p07.FeasibilityContractError):
        p07.verify_persisted_artifacts(summary_path=destination[0], by_state_path=destination[1], by_terminal_path=destination[2], by_group_path=destination[3], consistency_path=destination[4])


def test_post_write_verification_failure_restores_preexisting_bytes_and_cleans_temporaries(monkeypatch, tmp_path, base_result):
    destination = paths(tmp_path); originals = tuple(f"previous-{index}".encode() for index in range(5))
    for path, payload in zip(destination, originals): path.write_bytes(payload)
    monkeypatch.setattr(p07, "verify_persisted_artifacts", lambda **_kwargs: (_ for _ in ()).throw(p07.FeasibilityContractError("synthetic post-write failure")))
    with pytest.raises(p07.FeasibilityContractError, match="post-write"):
        write(base_result, destination)
    assert tuple(path.read_bytes() for path in destination) == originals
    assert not list(tmp_path.glob("*.tmp"))


def test_main_records_external_performance_log_and_exits_one_after_post_write_verification_failure(monkeypatch, tmp_path, base_result):
    destination = paths(tmp_path / "published")
    monkeypatch.setattr(p07, "SUMMARY_PATH", destination[0]); monkeypatch.setattr(p07, "BY_STATE_PATH", destination[1])
    monkeypatch.setattr(p07, "BY_TERMINAL_PATH", destination[2]); monkeypatch.setattr(p07, "BY_GROUP_PATH", destination[3]); monkeypatch.setattr(p07, "CONSISTENCY_PATH", destination[4])
    monkeypatch.setattr(p07, "run_real_analysis", lambda **_kwargs: base_result)
    monkeypatch.setattr(p07, "verify_persisted_artifacts", lambda **_kwargs: (_ for _ in ()).throw(p07.FeasibilityContractError("synthetic post-write failure")))
    performance = tmp_path / "external-performance.json"
    with pytest.raises(SystemExit) as caught:
        p07.main(["--performance-log", str(performance)])
    assert caught.value.code == 1
    assert all(not path.exists() for path in destination)
    record = json.loads(performance.read_text(encoding="utf-8"))
    assert record["analysis_status"] == "publication_failed"
    assert record["failure"] == {"stage": "artifact_publication", "type": "FeasibilityContractError", "message": "synthetic post-write failure"}
    assert not list(destination[0].parent.glob("*.tmp"))


@pytest.mark.parametrize("target", ["summary_population", "state_count", "terminal_count", "group_count", "consistency_count", "seal", "order"])
def test_persisted_contract_rejects_rehashed_semantic_mutations(target, tmp_path, base_payloads):
    destination = paths(tmp_path); payloads = base_payloads
    for path, payload in zip(destination, payloads): path.write_bytes(payload)
    if target in {"summary_population", "seal"}:
        summary = json.loads(destination[0].read_text(encoding="utf-8"))
        if target == "summary_population": summary["population"]["development_matches"] += 1
        else: summary["test_seal"][p07.TEST_ZERO_FIELDS[0]] = 1
        destination[0].write_text(json.dumps(summary), encoding="utf-8")
    else:
        index = {"state_count": 1, "terminal_count": 2, "group_count": 3, "consistency_count": 4, "order": 2}[target]
        frame = pd.read_csv(destination[index])
        if target in {"state_count", "terminal_count", "group_count", "consistency_count"}: frame.loc[0, "attempts"] += 1
        elif target == "order": frame = frame.iloc[::-1]
        frame.to_csv(destination[index], index=False, lineterminator="\n")
    summary = json.loads(destination[0].read_text(encoding="utf-8")); table_payloads = tuple(path.read_bytes() for path in destination[1:])
    names = ("by_state", "by_terminal", "by_group", "consistency")
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, table_payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, table_payloads)}
    summary["publication_fingerprint"] = p07._fingerprint(summary, table_payloads)
    destination[0].write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    with pytest.raises(p07.FeasibilityContractError):
        p07.verify_persisted_artifacts(summary_path=destination[0], by_state_path=destination[1], by_terminal_path=destination[2], by_group_path=destination[3], consistency_path=destination[4])


def test_upstream_contract_fails_closed_on_hash_or_semantic_change(monkeypatch, tmp_path):
    extractor, chronological = tmp_path / "extractor.py", tmp_path / "chronological.json"
    extractor.write_text("x", encoding="utf-8")
    valid_contract = {"frozen_sensitivity_contract": {"freeze_date": "2023-12-31"}, "rolling_origin_contract": {"validation_years": [2020, 2021, 2022, 2023], "validation_folds": 4}, "protected_test_contract": {"test_status": "sealed", "test_matches": 1531}, "source_contract": {"matches_after_immediate_reduction": 7524, "source_point_rows": 1280408, "players": 1002}, "split_population": [{"split": "train", "matches": 4188}, {"split": "validation", "matches": 1805}, {"split": "test", "matches": 1531}]}
    chronological.write_text(json.dumps(valid_contract), encoding="utf-8")
    monkeypatch.setattr(p07, "P07_EXTRACTOR_PATH", extractor); monkeypatch.setattr(p07, "CHRONOLOGICAL_SUMMARY_PATH", chronological)
    monkeypatch.setattr(p07, "P07_EXTRACTOR_SHA256", p07._sha256_file(extractor)); monkeypatch.setattr(p07, "CHRONOLOGICAL_SUMMARY_SHA256", p07._sha256_file(chronological))
    assert p07.validate_upstream_contracts()["extractor_commit"] == p07.UPSTREAM_COMMIT
    valid_contract["source_contract"]["matches_after_immediate_reduction"] = 1
    chronological.write_text(json.dumps(valid_contract), encoding="utf-8")
    monkeypatch.setattr(p07, "CHRONOLOGICAL_SUMMARY_SHA256", p07._sha256_file(chronological))
    with pytest.raises(p07.FeasibilityContractError): p07.validate_upstream_contracts()
    monkeypatch.setattr(p07, "UPSTREAM_COMMIT", "wrong")
    with pytest.raises(p07.FeasibilityContractError): p07.validate_upstream_contracts()


def test_upstream_absence_and_bad_hash_fail_before_any_analysis(monkeypatch, tmp_path):
    missing = tmp_path / "missing.py"
    monkeypatch.setattr(p07, "P07_EXTRACTOR_PATH", missing)
    with pytest.raises(p07.FeasibilityContractError, match="Falta"):
        p07.validate_upstream_contracts()


def test_forbidden_performance_path_stops_before_run_or_source_access(monkeypatch):
    calls = []
    monkeypatch.setattr(p07, "run_real_analysis", lambda **_kwargs: calls.append("run"))
    with pytest.raises(p07.FeasibilityContractError, match="fuera"):
        p07.main(["--performance-log", str(p07.ROOT / "forbidden.json")])
    assert calls == []


def test_real_wrapper_preserves_the_precise_failed_stage_without_reading_parquet(monkeypatch):
    monkeypatch.setattr(p07, "validate_upstream_contracts", lambda: SYNTHETIC_UPSTREAM)
    monkeypatch.setattr(p07, "read_source_points", lambda: pd.DataFrame())
    with pytest.raises(p07._RunFailure) as caught:
        p07.run_real_analysis()
    assert caught.value.stage == "source_validation"


def test_productive_reader_is_single_and_attempt_construction_is_not_rowwise_iterrows():
    module_source = inspect.getsource(p07)
    assert module_source.count("pd.read_parquet(") == 1
    assert "columns=list(SOURCE_COLUMNS)" in module_source
    assert ".iterrows(" not in inspect.getsource(p07.construct_attempts)


def test_not_available_contract_allows_headers_only_and_no_partial_rows():
    empty = tuple(pd.DataFrame(columns=columns) for columns in (p07.BY_STATE_COLUMNS, p07.BY_TERMINAL_COLUMNS, p07.BY_GROUP_COLUMNS, p07.CONSISTENCY_COLUMNS))
    summary = {"analysis_status": "not_available", "reason_codes": ["execution_failed"], "failure": {"stage": "synthetic", "type": "RuntimeError", "message": "synthetic failure"}, "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": None, **{field: 0 for field in p07.TEST_ZERO_FIELDS}}}
    result = p07.FeasibilityResult(summary, *empty, pd.DataFrame(), "unused")
    p07.validate_result(result)


@pytest.mark.parametrize("field", p07.TEST_ZERO_FIELDS)
def test_each_sealed_test_counter_is_individually_rejected(field, base_result):
    summary = {**base_result.summary, "test_seal": {**base_result.summary["test_seal"], field: 1}}
    with pytest.raises(p07.FeasibilityContractError, match="Test seal"):
        p07.validate_result(replace(base_result, summary=summary))


def test_cache_is_exactly_once_per_contractual_key_and_outcomes_follow_classification():
    source = points().iloc[[0, 1, 5]].copy()
    duplicate = source.iloc[[0]].copy(); duplicate.loc[:, "match_id"] = "m7"; duplicate.loc[:, "date"] = "2020-01-03"
    source = pd.concat((source, duplicate), ignore_index=True)
    development, _ = p07.split_development_before_parsing(p07.validate_source_points(source))
    parser_calls, validation_calls = [], []
    def extractor(text, number, previous_attempt_was_fault=False):
        parser_calls.append((text, number, previous_attempt_was_fault))
        return p07.parse_and_classify_initial_return_terminal(text, number, previous_attempt_was_fault)
    def validator(value):
        validation_calls.append(value)
        p07.validate_return_terminal_classification(value)
    attempts = p07.construct_attempts(development, extractor=extractor, validator=validator)
    assert parser_calls == [("6f*", 1, False), ("5n", 1, False), ("6f#", 2, True)]
    assert len(validation_calls) == len(parser_calls) == 3
    winner = attempts.loc[attempts.terminal_class.eq("winner")].iloc[0]
    assert pd.api.types.is_bool_dtype(attempts.returner_won_point) and bool(winner.returner_won_point) is True


@pytest.mark.parametrize("field,value", [("first_serve", 1), ("first_serve", True), ("second_serve", 2.0)])
def test_sequence_domains_are_rejected_without_coercion(field, value):
    source = points(); source[field] = source[field].astype(object); source.loc[1, "first_serve"] = "5n"; source.loc[1, field] = value
    development, _ = p07.split_development_before_parsing(p07.validate_source_points(source))
    with pytest.raises(p07.FeasibilityContractError):
        p07.construct_attempts(development)


@pytest.mark.parametrize("sequence,state,eligible", [
    ("6q*", "documented_return_winner", True),
    ("6f27b1*", "return_terminal_not_documented", False),
    ("6?f#", "unknown_initial_return", False),
    ("6*", "ineligible_censored", False),
])
def test_p07_states_are_exclusively_extractor_states(sequence, state, eligible):
    classified = p07.parse_and_classify_initial_return_terminal(sequence, 1)
    assert classified.classification_state.value == state
    assert classified.eligible_for_terminal_comparison is eligible


def test_staging_failure_and_each_replace_failure_rollback_all_bytes(monkeypatch, tmp_path):
    empty = tuple(pd.DataFrame(columns=columns) for columns in (p07.BY_STATE_COLUMNS, p07.BY_TERMINAL_COLUMNS, p07.BY_GROUP_COLUMNS, p07.CONSISTENCY_COLUMNS))
    unavailable = p07._finalize({"analysis_status": "not_available", "reason_codes": ["execution_failed"], "failure": {"stage": "synthetic", "type": "RuntimeError", "message": "synthetic failure"}, "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": None, **{field: 0 for field in p07.TEST_ZERO_FIELDS}}}, pd.DataFrame(), *empty)
    destination = paths(tmp_path); originals = tuple(f"old-{index}".encode() for index in range(5))
    for path, payload in zip(destination, originals): path.write_bytes(payload)
    original_stage = p07._stage; staged = []
    def failing_stage(path, payload):
        staged.append(path)
        if len(staged) == 3: raise OSError("staging failure")
        return original_stage(path, payload)
    monkeypatch.setattr(p07, "_stage", failing_stage)
    with pytest.raises(OSError): write(unavailable, destination)
    assert tuple(path.read_bytes() for path in destination) == originals
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(p07, "_stage", original_stage)
    original_replace = p07.os.replace
    for failing_index in range(5):
        calls = []
        def failing_replace(source, target, index=failing_index):
            calls.append(target)
            if len(calls) == index + 1: raise OSError("replace failure")
            return original_replace(source, target)
        monkeypatch.setattr(p07.os, "replace", failing_replace)
        with pytest.raises(OSError): write(unavailable, destination)
        assert tuple(path.read_bytes() for path in destination) == originals
        assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(p07.os, "replace", original_replace)


def test_atomic_failures_leave_no_partial_publication_when_no_previous_files(monkeypatch, tmp_path):
    empty = tuple(pd.DataFrame(columns=columns) for columns in (p07.BY_STATE_COLUMNS, p07.BY_TERMINAL_COLUMNS, p07.BY_GROUP_COLUMNS, p07.CONSISTENCY_COLUMNS))
    unavailable = p07._finalize({"analysis_status": "not_available", "reason_codes": ["execution_failed"], "failure": {"stage": "synthetic", "type": "RuntimeError", "message": "synthetic failure"}, "test_seal": {"test_status": "sealed", "used_for_method_selection": False, "excluded_test_matches": None, **{field: 0 for field in p07.TEST_ZERO_FIELDS}}}, pd.DataFrame(), *empty)
    original_stage, original_replace = p07._stage, p07.os.replace
    for stage_index in range(5):
        destination = tuple(tmp_path / f"stage_{stage_index}_{name}" for name in ("summary.json", "state.csv", "terminal.csv", "group.csv", "consistency.csv")); calls = []
        def failing_stage(path, payload, index=stage_index):
            calls.append(path)
            if len(calls) == index + 1: raise OSError("staging failure")
            return original_stage(path, payload)
        monkeypatch.setattr(p07, "_stage", failing_stage)
        with pytest.raises(OSError): write(unavailable, destination)
        assert all(not path.exists() for path in destination) and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(p07, "_stage", original_stage)
    for replace_index in range(5):
        destination = tuple(tmp_path / f"replace_{replace_index}_{name}" for name in ("summary.json", "state.csv", "terminal.csv", "group.csv", "consistency.csv")); calls = []
        def failing_replace(source, target, index=replace_index):
            calls.append(target)
            if len(calls) == index + 1: raise OSError("replace failure")
            return original_replace(source, target)
        monkeypatch.setattr(p07.os, "replace", failing_replace)
        with pytest.raises(OSError): write(unavailable, destination)
        assert all(not path.exists() for path in destination) and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(p07.os, "replace", original_replace)


@pytest.mark.integration
def test_frozen_p07_publication_is_artifact_only_and_semantically_sealed():
    """Comprueba la publicaciÃ³n P07 sin fuente de puntos ni cÃ³digo productivo."""
    root = Path(__file__).resolve().parents[1]
    paths = {
        "summary": root / "reports" / "return_terminal_descriptive_feasibility_summary.json",
        "by_state": root / "reports" / "tables" / "return_terminal_descriptive_feasibility_by_state.csv",
        "by_terminal": root / "reports" / "tables" / "return_terminal_descriptive_feasibility_by_terminal.csv",
        "by_group": root / "reports" / "tables" / "return_terminal_descriptive_feasibility_by_group.csv",
        "consistency": root / "reports" / "tables" / "return_terminal_descriptive_feasibility_consistency.csv",
    }
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    for name, payload in payloads.items():
        size, digest = FROZEN_P07_ARTIFACTS[name]
        assert len(payload) == size
        assert hashlib.sha256(payload).hexdigest().upper() == digest
        assert b"NaN" not in payload and b"Infinity" not in payload and b"C:\\Users\\" not in payload

    summary = json.loads(payloads["summary"].decode("utf-8"))
    tables = {
        name: list(csv.DictReader(io.StringIO(payload.decode("utf-8"), newline="")))
        for name, payload in payloads.items()
        if name != "summary"
    }
    assert summary["publication_fingerprint"] == FROZEN_P07_FINGERPRINT
    assert summary["analysis_status"] == "available_descriptive"
    assert summary["upstream"]["extractor_commit"] == "fff6118"
    assert summary["artifact_repair"] == {
        "mode": "artifact_only_canonical_reserialization",
        "real_analysis_rerun": False,
        "source_publication_fingerprint": "F94D2A995EEEF5882525FEC8628CD30845CCF7E5CD993090684946E683996FCE",
        "historical_real_analysis_attempts": 2,
        "historical_run_2_status": "publication_verification_failed",
        "reordered_artifacts": ["by_state", "by_group", "consistency"],
        "analytical_values_changed": False,
        "reason": "canonical_surface_order",
    }
    assert summary["population"] == {
        "development_matches": 5993, "development_point_rows": 1035760,
        "development_servers": 870, "excluded_test_matches": 1531,
        "source_matches": 7524, "source_players": 1002, "source_rows_read": 1280408,
    }
    assert summary["attempts"] == {"cache_entries": 490548, "first": 1035760, "second": 391103, "total": 1426863}
    assert summary["second_serve_context"] == {
        "second_attempts": 391103, "documented_first_service_fault": 382143,
        "second_serve_without_documented_first_fault": 8960,
        "without_documented_first_fault_proportion": 8960 / 391103,
    }
    assert summary["state_counts"] == {
        "documented_return_winner": 17955, "documented_return_forced_error": 16550,
        "documented_return_unforced_error": 39673, "return_terminal_not_documented": 650380,
        "unknown_initial_return": 173768, "ineligible_censored": 528537,
    }
    assert summary["terminal_class_counts"] == {"winner": 17955, "forced_error": 16550, "unforced_error": 39673}
    assert summary["unforced_error_code_counts"] == {"n": 14541, "w": 8063, "d": 15158, "x": 1480, "e": 10, "!": 421}
    seal = summary["test_seal"]
    assert seal["test_status"] == "sealed" and seal["used_for_method_selection"] is False
    assert all(seal[field] == 0 for field in p07.TEST_ZERO_FIELDS)

    as_int = lambda row, field: int(row[field])
    state_totals = {}
    for row in tables["by_state"]:
        if row["group_type"] == "total":
            state_totals[row["classification_state"]] = state_totals.get(row["classification_state"], 0) + as_int(row, "attempts")
    assert state_totals == summary["state_counts"]
    assert sum(state_totals.values()) == summary["attempts"]["total"]
    total_group = [row for row in tables["by_group"] if row["group_type"] == "total"]
    assert len(total_group) == 1 and as_int(total_group[0], "attempts") == 1426863
    for group_type, column, expected in (
        ("surface", "surface", ["Hard", "Clay", "Grass"]),
        ("derived_period", "derived_period", ["to_2009", "2010s", "2020s"]),
        ("validation_fold", "validation_fold", ["pre_validation", "validation_2020", "validation_2021", "validation_2022", "validation_2023"]),
    ):
        rows = [row for row in tables["by_group"] if row["group_type"] == group_type]
        assert [row[column] for row in rows] == expected
        assert sum(as_int(row, "attempts") for row in rows) == 1426863
    contexts = {row["second_serve_context"]: as_int(row, "attempts") for row in tables["by_group"] if row["group_type"] == "second_serve_context"}
    assert contexts == {"documented_first_service_fault": 382143, "second_serve_without_documented_first_fault": 8960}

    terminal_totals = {row["terminal_class"]: as_int(row, "attempts") for row in tables["by_terminal"] if row["aggregation_level"] == "terminal_class" and row["serve_number"] == "0"}
    code_totals = {row["unforced_error_code"]: as_int(row, "attempts") for row in tables["by_terminal"] if row["aggregation_level"] == "unforced_error_code" and row["serve_number"] == "0"}
    assert terminal_totals == summary["terminal_class_counts"] and code_totals == summary["unforced_error_code_counts"]
    assert terminal_totals["unforced_error"] == sum(code_totals.values())
    assert sum(terminal_totals.values()) == summary["terminal_coverage"]["eligible_terminal_attempts"] == 74178
    for row in tables["consistency"]:
        attempts = as_int(row, "attempts")
        assert attempts == as_int(row, "concordant") + as_int(row, "discordant")
        assert as_int(row, "returner_wins") <= attempts
        if attempts == 0:
            assert row["concordance_rate"] == row["concordance_wilson_low"] == row["concordance_wilson_high"] == ""
        else:
            low, rate, high = map(float, (row["concordance_wilson_low"], row["concordance_rate"], row["concordance_wilson_high"]))
            assert 0 <= low <= rate <= high <= 1
