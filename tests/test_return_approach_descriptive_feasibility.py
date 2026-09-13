"""Contrato sintético P08: no abre datos ni artefactos permanentes."""
from dataclasses import replace
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis import return_approach_descriptive_feasibility as p08


SYNTHETIC_UPSTREAM = {
    "extractor_commit": p08.UPSTREAM_COMMIT,
    "extractor_sha256": p08.EXTRACTOR_SHA256,
    "chronological_summary_sha256": p08.CHRONOLOGICAL_SUMMARY_SHA256,
}

FROZEN_P08_ARTIFACTS = {
    "summary": (2_819, "004768BCA1B9FF93413F41F641546F0A9F2A54689F72C6951C87EE43F6514DC4"),
    "by_state": (18_922, "588B70494A9AC78B0A29A6F3392B10D722B485F2F4F39AC4119C160AA52FB41F"),
    "by_group": (1_477, "A030BE4844550589A512807D82049788BBF10A856A62C3A5EF086EEB9CD71C48"),
    "marker_context": (512, "CBEA3504763A6EBD1FA1C45668BF65D9C8C23BE763B2D9447DA88075C9CA9900"),
    "outcomes": (96, "2F0AC9AFABCA2D18AC94C1F0B79B2F43B4A2E9A317CB3B068FA552860D383F38"),
}
FROZEN_P08_FINGERPRINT = "54CA82A1985738260FE529A54CD67C00C02A353CE2931671ABD1738A35107A1E"


def points() -> pd.DataFrame:
    """Diez intentos de desarrollo, seis contextos y una fila de test sellada."""
    rows = [
        ("m0", "2009-01-01", "Hard", 1, 2, "A0", "B0", "6f+", None),
        ("m1", "2019-01-01", "Clay", 2, 2, "A1", "B1", "6f", None),
        ("m2", "2020-01-01", "Grass", 1, 1, "A2", "B2", "6q", None),
        ("m3", "2021-01-01", "Hard", 2, 1, "A3", "B3", "6*", None),
        ("m4", "2022-01-01", "Clay", 1, 2, "A4", "B4", "6+f+", None),
        ("m5", "2023-01-01", "Grass", 2, 1, "A5", "B5", "6+f", None),
        ("m6", "2023-02-01", "Hard", 1, 2, "A6", "B6", "6n", "6f+"),
        ("m7", "2023-03-01", "Clay", 2, 1, "A7", "B7", "6f", "6f+"),
        ("m8", "2024-01-01", "Grass", 1, 1, "A8", "B8", "6b+", None),
    ]
    return pd.DataFrame([
        {"match_id": match, "point_number": 1, "date": day, "surface": surface,
         "server": server, "point_winner": winner, "player_1": first, "player_2": second,
         "first_serve": first_serve, "second_serve": second_serve}
        for match, day, surface, server, winner, first, second, first_serve, second_serve in rows
    ])


def synthetic_extractor(text, number, previous_fault=False):
    if text in {"6f", "6+f"}:
        return SimpleNamespace(
            state=SimpleNamespace(value=p08.STATES[1]),
            reason_code=SimpleNamespace(value="no_unambiguous_immediate_approach_marker"),
            service_approach_marker_span=(1, 2) if text == "6+f" else None,
            return_approach_marker_span=None,
            terminal_serve_outcome=None,
        )
    return p08.parse_and_classify_initial_return_approach(text, number, previous_fault)


def result() -> p08.FeasibilityResult:
    return p08.analyze_points(points(), upstream_contract=SYNTHETIC_UPSTREAM, extractor=synthetic_extractor)


@pytest.fixture(scope="module")
def base_result():
    return result()


def paths(root: Path) -> tuple[Path, ...]:
    return tuple(root / name for name in ("summary.json", "state.csv", "group.csv", "marker.csv", "outcomes.csv"))


def write(value: p08.FeasibilityResult, destination: tuple[Path, ...]) -> None:
    p08.write_artifacts(
        value, summary_path=destination[0], by_state_path=destination[1],
        by_group_path=destination[2], marker_context_path=destination[3],
        outcomes_path=destination[4], expected_population=None,
    )


def resign(value: p08.FeasibilityResult) -> p08.FeasibilityResult:
    frames = (value.by_state, value.by_group, value.marker_context, value.outcomes)
    payloads = tuple(p08._table_bytes(frame) for frame in frames)
    names = ("by_state", "by_group", "marker_context", "outcomes")
    summary = json.loads(json.dumps(value.summary))
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, payloads)}
    summary["publication_fingerprint"] = p08._fingerprint(summary, payloads)
    return replace(value, summary=summary)


def test_analysis_population_states_contexts_and_outcomes_are_exact(base_result):
    actual = base_result
    assert actual.summary["population"] == {
        "source_rows_read": 9, "source_matches": 9, "source_players": 18,
        "development_point_rows": 8, "development_matches": 8,
        "development_servers": 8, "excluded_test_matches": 1,
        "first_attempts": 8, "second_attempts": 2, "attempts_total": 10,
    }
    assert actual.summary["state_counts"] == dict(zip(p08.STATES, (4, 3, 1, 2)))
    assert actual.summary["analysis_status"] == "available_descriptive"
    assert actual.summary["outcome_comparison_status"] == "available"
    assert actual.summary["reason"] is None
    assert list(actual.outcomes.state) == list(p08.STATES[:2])
    assert actual.summary["second_serve_context"] == {
        "second_attempts": 2,
        "documented_first_service_fault": 1,
        "second_serve_without_documented_first_fault": 1,
    }


def test_sealing_occurs_before_extraction_and_test_is_never_parsed(monkeypatch):
    seen = []
    original = p08.parse_and_classify_initial_return_approach
    def recording_extractor(*args):
        seen.append(args)
        return synthetic_extractor(*args)
    monkeypatch.setattr(p08, "parse_and_classify_initial_return_approach", lambda *args: (seen.append(args) or original(*args)))
    actual = p08.analyze_points(points(), upstream_contract=SYNTHETIC_UPSTREAM, extractor=recording_extractor)
    assert len(actual.attempts) == 10
    assert not any(args[0] == "6b+" for args in seen)
    assert all(actual.summary["test_seal"][field] == 0 for field in p08.TEST_ZERO_FIELDS)


def test_cache_key_includes_text_serve_and_previous_fault():
    source = p08.validate_source_points(points())
    development, _ = p08.split_development_before_parsing(source)
    calls = []
    attempts = p08.construct_attempts(development, extractor=lambda *args: (calls.append(args) or synthetic_extractor(*args)))
    assert len(calls) == len(set(calls)) == attempts.attrs["cache_entries"]
    assert ("6f+", 2, True) in calls and ("6f+", 2, False) in calls


def test_source_reader_and_control_flow_are_static_and_single():
    source = inspect.getsource(p08)
    assert source.count("pd.read_parquet(") == 1
    assert source.count("construct_attempts(development, extractor=extractor)") == 1
    assert source.index("split_development_before_parsing(source)") < source.index("construct_attempts(development, extractor=extractor)")
    assert ".iterrows(" not in source and ".itertuples(" in source
    body = inspect.getsource(p08.construct_attempts)
    assert body.count("extractor(sequence_text, serve_number, first_fault)") == 1


@pytest.mark.parametrize("column", p08.SOURCE_COLUMNS)
def test_missing_source_column_is_rejected(column):
    with pytest.raises(p08.FeasibilityContractError):
        p08.validate_source_points(points().drop(columns=column))


@pytest.mark.parametrize("field,value", [
    ("server", True), ("server", "1"), ("server", 1.0),
    ("point_winner", False), ("point_winner", "2"), ("point_winner", 2.0),
    ("surface", "Carpet"), ("surface", None), ("date", "01/02/2020"),
    ("date", "bad"), ("date", pd.Timestamp("2020-01-01 12:00:00")),
    ("point_number", True), ("point_number", 1.0),
    ("match_id", " m0"), ("player_1", 1), ("player_2", ""),
    ("first_serve", None), ("second_serve", 6),
])
def test_source_domains_are_strict(field, value):
    value_frame = points()
    value_frame[field] = value_frame[field].astype(object)
    value_frame.loc[0, field] = value
    with pytest.raises((TypeError, ValueError, p08.FeasibilityContractError)):
        p08.validate_source_points(value_frame)


def test_duplicate_point_key_and_equal_players_are_rejected():
    duplicate = pd.concat([points(), points().iloc[[0]]], ignore_index=True)
    with pytest.raises(p08.FeasibilityContractError):
        p08.validate_source_points(duplicate)
    equal = points(); equal.loc[0, "player_2"] = equal.loc[0, "player_1"]
    with pytest.raises(p08.FeasibilityContractError):
        p08.validate_source_points(equal)


def test_match_metadata_cannot_cross_the_temporal_seal():
    value = pd.concat([points(), points().iloc[[0]]], ignore_index=True)
    value.loc[value.index[-1], ["point_number", "date"]] = [2, "2024-01-01"]
    with pytest.raises(p08.FeasibilityContractError, match="Metadatos inconsistentes"):
        p08.validate_source_points(value)


def test_group_partitions_and_canonical_orders(base_result):
    group = base_result.by_group
    assert [tuple(row) for row in group[["group_type", "serve_number", "surface", "derived_period", "validation_fold"]].itertuples(index=False, name=None)] == p08._expected_group_keys()
    assert list(group.loc[group.group_type.eq("surface"), "surface"]) == list(p08.SURFACES)
    assert list(group.loc[group.group_type.eq("derived_period"), "derived_period"]) == list(p08.PERIODS)
    assert list(group.loc[group.group_type.eq("validation_fold"), "validation_fold"]) == list(p08.FOLDS)
    total = int(group.iloc[0].attempts)
    for dimension in ("serve_number", "surface", "derived_period", "validation_fold"):
        assert int(group.loc[group.group_type.eq(dimension), "attempts"].sum()) == total


def test_state_reason_matrix_is_closed_and_exhaustive(base_result):
    expected_per_group = sum(len(p08.STATE_REASONS[state]) for state in p08.STATES)
    assert len(base_result.by_state) == len(p08._expected_group_keys()) * expected_per_group
    assert set(zip(base_result.by_state.state, base_result.by_state.reason_code)) == {
        (state, reason) for state in p08.STATES for reason in p08.STATE_REASONS[state]
    }
    assert int(base_result.by_state.loc[base_result.by_state.group_type.eq("total"), "attempts"].sum()) == 10


def test_marker_context_is_closed_exhaustive_and_not_causal(base_result):
    table = base_result.marker_context
    assert list(table.marker_context) == list(p08.MARKER_CONTEXTS)
    assert int(table.attempts.sum()) == 10
    assert table.set_index("marker_context").attempts.astype(int).to_dict() == {
        "service_and_return_approach_documented": 1,
        "service_approach_documented_return_not_documented": 1,
        "return_approach_documented_service_not_documented": 3,
        "neither_documented": 2,
        "unknown_marker_context": 1,
        "ineligible_censored": 2,
    }


def test_outcomes_exclude_unknown_and_censored(base_result):
    assert int(base_result.outcomes.attempts.sum()) == 7
    assert int(base_result.outcomes.returner_wins.sum()) == 6
    assert not set(p08.STATES[2:]) & set(base_result.outcomes.state)


def test_no_negative_means_no_comparison():
    value = points()
    value.loc[value.first_serve.isin(["6f", "6+f"]), "first_serve"] = "6q"
    actual = p08.analyze_points(value, upstream_contract=SYNTHETIC_UPSTREAM, extractor=synthetic_extractor)
    assert actual.summary["analysis_status"] == "available_descriptive_not_comparable"
    assert actual.summary["outcome_comparison_status"] == "not_available"
    assert actual.summary["reason"] == "no_observable_nonapproach_comparator"
    assert actual.outcomes.empty


def independent_wilson(successes, trials, z=p08.WILSON_Z):
    if trials == 0:
        return None, None
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return (0.0 if successes == 0 else centre - spread, 1.0 if successes == trials else centre + spread)


@pytest.mark.parametrize("successes,trials", [(0, 1), (0, 3), (0, 10), (1, 1), (3, 3), (10, 10), (1, 3), (2, 3), (0, 0)])
def test_wilson_boundaries_and_independent_values(successes, trials):
    actual = p08.wilson(successes, trials)
    expected = independent_wilson(successes, trials)
    assert actual == pytest.approx(expected) if trials else actual == expected
    if trials:
        assert 0 <= actual[0] <= successes / trials <= actual[1] <= 1


@pytest.mark.parametrize("successes,trials", [(True, 1), (False, 1), (1.0, 2), ("1", 2), (None, 2), (-1, 2), (3, 2), (1, -1)])
def test_wilson_rejects_invalid_domains(successes, trials):
    with pytest.raises(p08.FeasibilityContractError):
        p08.wilson(successes, trials)


def test_wilson_material_deviation_and_context_are_reported(monkeypatch):
    original_sqrt = p08.math.sqrt
    monkeypatch.setattr(p08.math, "sqrt", lambda value: original_sqrt(value) + 10)
    with pytest.raises(p08.FeasibilityContractError, match="aggregation_level.*surface"):
        p08.wilson(1, 3, context={"aggregation_level": "surface"})


@pytest.mark.parametrize("field", p08.TEST_ZERO_FIELDS)
def test_each_test_counter_is_sealed(base_result, field):
    assert base_result.summary["test_seal"][field] == 0


def test_serialization_is_deterministic_and_fingerprint_uses_exact_csv(base_result):
    first = p08.serialize_artifacts(base_result)
    second = p08.serialize_artifacts(base_result)
    assert first == second and len(first) == 5
    assert base_result.summary["artifact_payload_bytes"] == dict(zip(("by_state", "by_group", "marker_context", "outcomes"), map(len, first[1:])))
    assert base_result.summary["publication_fingerprint"] == p08._fingerprint(base_result.summary, first[1:])
    assert not any(b"Unnamed" in payload for payload in first[1:])


def test_serialization_detects_simulated_nondeterminism(monkeypatch, base_result):
    original = p08._serialize_once; calls = []
    def unstable(value):
        calls.append(1)
        payloads = original(value)
        return payloads if len(calls) == 1 else (payloads[0] + b"x", *payloads[1:])
    monkeypatch.setattr(p08, "_serialize_once", unstable)
    with pytest.raises(p08.FeasibilityContractError, match="no determinista"):
        p08.serialize_artifacts(base_result)


def test_persisted_round_trip_repeats_semantic_validation(tmp_path, base_result):
    destination = paths(tmp_path)
    write(base_result, destination)
    restored = p08.verify_persisted_artifacts(
        summary_path=destination[0], by_state_path=destination[1], by_group_path=destination[2],
        marker_context_path=destination[3], outcomes_path=destination[4], expected_population=None,
    )
    assert p08.serialize_artifacts(restored) == tuple(path.read_bytes() for path in destination)


def test_persisted_resigned_semantic_mutation_is_rejected(tmp_path, base_result):
    destination = paths(tmp_path)
    payloads = list(p08.serialize_artifacts(base_result))
    group = base_result.by_group.copy(deep=True)
    group.loc[0, "positive_attempts"] += 1
    payloads[2] = p08._table_bytes(group)
    summary = json.loads(payloads[0].decode("utf-8"))
    csv_payloads = tuple(payloads[1:])
    names = ("by_state", "by_group", "marker_context", "outcomes")
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, csv_payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, csv_payloads)}
    summary["publication_fingerprint"] = p08._fingerprint(summary, csv_payloads)
    payloads[0] = p08._summary_bytes(summary)
    for path, payload in zip(destination, payloads):
        path.write_bytes(payload)
    with pytest.raises(p08.FeasibilityContractError, match="by_group"):
        p08.verify_persisted_artifacts(
            summary_path=destination[0], by_state_path=destination[1], by_group_path=destination[2],
            marker_context_path=destination[3], outcomes_path=destination[4], expected_population=None,
        )


@pytest.mark.parametrize("mutation", [
    "status", "population", "cardinality", "states", "coverage", "comparability",
    "reason", "marker_count", "marker_proportion", "outcome_count", "outcome_wilson",
    "serve_number", "surface", "period", "future_fold", "test_seal", "duplicate",
    "removed", "additional", "row_order", "column_order", "missing_column",
    "nonfinite_literal", "reconciliation",
])
def test_resigned_semantic_mutations_are_rejected(base_result, mutation):
    summary = json.loads(json.dumps(base_result.summary))
    by_state = base_result.by_state.copy(deep=True)
    by_group = base_result.by_group.copy(deep=True)
    marker = base_result.marker_context.copy(deep=True)
    outcomes = base_result.outcomes.copy(deep=True)
    if mutation == "status": summary["analysis_status"] = "not_available"
    elif mutation == "population": summary["population"]["source_matches"] += 1
    elif mutation == "cardinality": summary["population"]["attempts_total"] += 1
    elif mutation == "states": summary["state_counts"][p08.STATES[0]] += 1
    elif mutation == "coverage": summary["coverage"]["unknown_attempts"] += 1
    elif mutation == "comparability": summary["outcome_comparison_status"] = "not_available"
    elif mutation == "reason": summary["reason"] = "invented"
    elif mutation == "marker_count": marker.loc[0, "attempts"] += 1
    elif mutation == "marker_proportion": marker.loc[0, "proportion"] += 0.01
    elif mutation == "outcome_count": outcomes.loc[0, "returner_wins"] -= 1
    elif mutation == "outcome_wilson": outcomes.loc[0, "wilson_low"] += 0.01
    elif mutation == "serve_number": by_group.loc[1, "serve_number"] = 3
    elif mutation == "surface": by_group.loc[3, "surface"] = "Carpet"
    elif mutation == "period": by_group.loc[6, "derived_period"] = "bad"
    elif mutation == "future_fold": by_group.loc[9, "validation_fold"] = "2024"
    elif mutation == "test_seal": summary["test_seal"][p08.TEST_ZERO_FIELDS[0]] = 1
    elif mutation == "duplicate": by_state = pd.concat([by_state, by_state.iloc[[0]]], ignore_index=True)
    elif mutation == "removed": by_group = by_group.iloc[:-1].copy()
    elif mutation == "additional": by_group = pd.concat([by_group, by_group.iloc[[-1]]], ignore_index=True)
    elif mutation == "row_order": by_group = by_group.iloc[::-1].reset_index(drop=True)
    elif mutation == "column_order": by_group = by_group[list(reversed(by_group.columns))]
    elif mutation == "missing_column": by_state = by_state.drop(columns="matches")
    elif mutation == "nonfinite_literal": by_state.loc[0, "reason_code"] = "not-finite"
    else: summary["reconciliations"]["groups_reconciled"] = False
    tampered = resign(replace(base_result, summary=summary, by_state=by_state, by_group=by_group, marker_context=marker, outcomes=outcomes))
    with pytest.raises(p08.FeasibilityContractError):
        p08.validate_result(tampered)


def test_not_available_is_closed_header_only_and_sanitized():
    unavailable = p08.not_available_result(RuntimeError(str(p08.ROOT / "secret")), "source_validation", partial_diagnostics={"source_rows": 7})
    assert unavailable.summary["analysis_status"] == "not_available"
    assert unavailable.summary["reason_codes"] == ["execution_failed"]
    assert unavailable.summary["failure"]["message"] == "<repository>/secret"
    assert unavailable.summary["partial_diagnostics"] == {"source_rows": 7}
    assert all(frame.empty for frame in (unavailable.by_state, unavailable.by_group, unavailable.marker_context, unavailable.outcomes))
    assert all(len(payload.decode("utf-8").splitlines()) == 1 for payload in p08.serialize_artifacts(unavailable)[1:])


def test_repository_paths_are_sanitized_portably_and_deterministically():
    parts = [part for part in re.split(r"[\\/]+", str(p08.ROOT)) if part]
    prefix = "\\" if str(p08.ROOT).startswith(("/", "\\")) else ""
    windows_root = prefix + "\\".join(parts)
    posix_root = ("/" if prefix else "") + "/".join(parts)
    mixed_root = prefix + "".join(
        part if index == 0 else ("/" if index % 2 else "\\") + part
        for index, part in enumerate(parts)
    )

    assert p08._sanitize_message(windows_root) == "<repository>"
    assert p08._sanitize_message(f"{posix_root}/secret") == "<repository>/secret"
    assert p08._sanitize_message(f"{mixed_root}\\one/two") == "<repository>/one/two"
    assert p08._sanitize_message(
        f"first={windows_root}\\one; second={posix_root}/two/three"
    ) == "first=<repository>/one; second=<repository>/two/three"


@pytest.mark.parametrize(
    "message",
    (
        r"C:\Users\alice\secret",
        r"\\server\share\secret",
        "/Users/alice/secret",
        "/home/alice/secret",
        "~/secret",
        "../secret",
        "folder/../../secret",
        "file:///Users/alice/secret",
        r"file:///C:\Users\alice\secret",
        "vscode-file:///Users/alice/secret",
    ),
)
def test_non_repository_local_paths_are_redacted(message):
    sanitized = p08._sanitize_message(message)
    assert sanitized == "<absolute-path>" or sanitized.endswith("/<absolute-path>")
    assert "secret" not in sanitized
    assert ".." not in sanitized


def test_repository_file_uri_and_traversal_are_sanitized_without_leaking():
    posix_root = str(p08.ROOT).replace("\\", "/")
    assert p08._sanitize_message(f"file:///{posix_root}/secret") == "<repository>/secret"
    assert p08._sanitize_message(f"{posix_root}/../secret") == "<absolute-path>"


@pytest.mark.parametrize(
    ("root", "message"),
    (
        (r"C:\work\project", r"C:/work\project/secret"),
        ("/Users/alice/project", "/Users/alice/project/secret"),
        ("/home/alice/project", r"\home/alice\project/secret"),
        ("/Users/alice/project", "file:////Users/alice/project/secret"),
    ),
)
def test_repository_sanitization_is_independent_of_host_path_flavour(monkeypatch, root, message):
    monkeypatch.setattr(p08, "ROOT", root)
    assert p08._sanitize_message(message) == "<repository>/secret"


def test_messages_without_local_paths_keep_their_allowed_content():
    message = "synthetic ratio 1/2; public URL https://example.test/a/b"
    assert p08._sanitize_message(message) == message


@pytest.mark.parametrize(
    "unsafe_message",
    (r"C:\Users\alice\secret", "/Users/alice/secret", "~/secret", "../secret"),
)
def test_not_available_contract_rejects_unsanitized_local_paths(unsafe_message):
    unavailable = p08.not_available_result(RuntimeError("synthetic"), "source_validation")
    summary = json.loads(json.dumps(unavailable.summary))
    summary["failure"]["message"] = unsafe_message
    with pytest.raises(p08.FeasibilityContractError, match="ruta absoluta"):
        p08.validate_result(resign(replace(unavailable, summary=summary)))


def test_not_available_with_any_partial_row_is_rejected():
    unavailable = p08.not_available_result(RuntimeError("synthetic"), "aggregation")
    bad = replace(unavailable, by_group=pd.DataFrame([{column: "" for column in p08.BY_GROUP_COLUMNS}], columns=p08.BY_GROUP_COLUMNS))
    bad = resign(bad)
    with pytest.raises(p08.FeasibilityContractError, match="filas parciales"):
        p08.validate_result(bad)


@pytest.mark.parametrize("mode,index,preexisting", [
    *(("stage", index, previous) for previous in (False, True) for index in range(5)),
    *(("replace", index, previous) for previous in (False, True) for index in range(5)),
])
def test_every_publication_failure_rolls_back_and_cleans(monkeypatch, tmp_path, base_result, mode, index, preexisting):
    destination = paths(tmp_path)
    before = [f"old-{position}".encode() for position in range(5)]
    if preexisting:
        for path, payload in zip(destination, before):
            path.write_bytes(payload)
    if mode == "stage":
        original = p08._stage; calls = []
        def broken(path, payload):
            calls.append(path)
            if len(calls) == index + 1:
                raise OSError("stage")
            return original(path, payload)
        monkeypatch.setattr(p08, "_stage", broken)
    else:
        original = p08.os.replace; calls = []
        def broken(source, target):
            calls.append(target)
            if len(calls) == index + 1:
                raise OSError("replace")
            return original(source, target)
        monkeypatch.setattr(p08.os, "replace", broken)
    with pytest.raises(OSError):
        write(base_result, destination)
    assert [path.read_bytes() for path in destination] == before if preexisting else not any(path.exists() for path in destination)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("preexisting", [False, True])
def test_post_write_verification_failure_rolls_back(monkeypatch, tmp_path, base_result, preexisting):
    destination = paths(tmp_path); before = [b"old"] * 5
    if preexisting:
        for path in destination:
            path.write_bytes(b"old")
    monkeypatch.setattr(p08, "verify_persisted_artifacts", lambda **kwargs: (_ for _ in ()).throw(p08.FeasibilityContractError("verify")))
    with pytest.raises(p08.FeasibilityContractError):
        write(base_result, destination)
    assert [path.read_bytes() for path in destination] == before if preexisting else not any(path.exists() for path in destination)
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_performance_path_stops_before_analysis(monkeypatch):
    calls = []
    monkeypatch.setattr(p08, "run_real_analysis", lambda: calls.append(True))
    with pytest.raises(p08.FeasibilityContractError):
        p08.main(["--performance-log", str(p08.ROOT / "bad.json")])
    assert calls == []


def test_real_flow_validates_upstream_before_reading(monkeypatch):
    calls = []
    monkeypatch.setattr(p08, "validate_upstream_contracts", lambda: (_ for _ in ()).throw(RuntimeError("upstream")))
    monkeypatch.setattr(p08, "read_source_points", lambda: calls.append("read"))
    with pytest.raises(p08._RunFailure) as error:
        p08.run_real_analysis()
    assert error.value.stage == "upstream_validation" and calls == []


def test_real_flow_preserves_read_failure_stage(monkeypatch):
    monkeypatch.setattr(p08, "validate_upstream_contracts", lambda: SYNTHETIC_UPSTREAM)
    monkeypatch.setattr(p08, "read_source_points", lambda: (_ for _ in ()).throw(OSError("read")))
    with pytest.raises(p08._RunFailure) as error:
        p08.run_real_analysis()
    assert error.value.stage == "read_source_points" and isinstance(error.value.cause, OSError)


def test_cli_available_exit_zero_and_external_log(monkeypatch, tmp_path, base_result):
    log = tmp_path / "perf.json"; written = []
    monkeypatch.setattr(p08, "run_real_analysis", lambda: base_result)
    monkeypatch.setattr(p08, "write_artifacts", lambda value: written.append(value))
    with pytest.raises(SystemExit) as error:
        p08.main(["--performance-log", str(log)])
    payload = json.loads(log.read_text(encoding="utf-8"))
    assert error.value.code == 0 and written == [base_result]
    assert payload["analysis_status"] == "available_descriptive" and payload["analysis_name"] == p08.ANALYSIS_NAME


def test_cli_analysis_failure_publishes_not_available(monkeypatch, tmp_path):
    log = tmp_path / "perf.json"; written = []
    monkeypatch.setattr(p08, "run_real_analysis", lambda: (_ for _ in ()).throw(RuntimeError("synthetic")))
    monkeypatch.setattr(p08, "write_artifacts", lambda value: written.append(value))
    with pytest.raises(SystemExit) as error:
        p08.main(["--performance-log", str(log)])
    payload = json.loads(log.read_text(encoding="utf-8"))
    assert error.value.code == 1 and written[0].summary["analysis_status"] == "not_available"
    assert payload["analysis_status"] == "not_available" and payload["failure"]["message"] == "synthetic"


def test_cli_publication_failure_does_not_mask_available_result(monkeypatch, tmp_path, base_result):
    log = tmp_path / "perf.json"; calls = []
    monkeypatch.setattr(p08, "run_real_analysis", lambda: base_result)
    monkeypatch.setattr(p08, "write_artifacts", lambda value: (calls.append(value) or (_ for _ in ()).throw(OSError("publish"))))
    with pytest.raises(SystemExit) as error:
        p08.main(["--performance-log", str(log)])
    payload = json.loads(log.read_text(encoding="utf-8"))
    assert error.value.code == 1 and calls == [base_result]
    assert payload["analysis_status"] == "publication_failed"
    assert payload["failure"] == {"stage": "publication_verification", "type": "OSError", "message": "publish"}


def frozen_paths() -> dict[str, Path]:
    return {
        "summary": p08.SUMMARY_PATH,
        "by_state": p08.BY_STATE_PATH,
        "by_group": p08.BY_GROUP_PATH,
        "marker_context": p08.MARKER_CONTEXT_PATH,
        "outcomes": p08.OUTCOMES_PATH,
    }


def load_frozen_payload() -> tuple[dict, dict[str, pd.DataFrame]]:
    artifact_paths = frozen_paths()
    summary = json.loads(artifact_paths["summary"].read_text(encoding="utf-8"))
    frames = {
        name: pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
        for name, path in artifact_paths.items() if name != "summary"
    }
    return summary, frames


def write_resigned_payload(summary, frames, destination):
    names = ("by_state", "by_group", "marker_context", "outcomes")
    csv_payloads = tuple(
        frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")
        for frame in (frames[name] for name in names)
    )
    summary["artifact_payload_sha256"] = {
        name: hashlib.sha256(payload).hexdigest().upper()
        for name, payload in zip(names, csv_payloads)
    }
    summary["artifact_payload_bytes"] = {
        name: len(payload) for name, payload in zip(names, csv_payloads)
    }
    summary["publication_fingerprint"] = p08._fingerprint(summary, csv_payloads)
    payloads = (p08._summary_bytes(summary), *csv_payloads)
    for path, payload in zip(destination, payloads):
        path.write_bytes(payload)
    assert summary["artifact_payload_sha256"] == {
        name: hashlib.sha256(payload).hexdigest().upper()
        for name, payload in zip(names, csv_payloads)
    }
    assert summary["publication_fingerprint"] == p08._fingerprint(summary, csv_payloads)


@pytest.mark.integration
def test_frozen_p08_publication_is_exact_artifact_only(monkeypatch):
    artifact_paths = frozen_paths()
    assert all(path.is_file() for path in artifact_paths.values())
    for name, path in artifact_paths.items():
        payload = path.read_bytes()
        assert (len(payload), hashlib.sha256(payload).hexdigest().upper()) == FROZEN_P08_ARTIFACTS[name]
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Parquet no autorizado")))
    actual = p08.verify_persisted_artifacts()
    assert actual.summary["publication_fingerprint"] == FROZEN_P08_FINGERPRINT
    assert actual.summary["analysis_status"] == "available_descriptive_not_comparable"
    assert actual.summary["population"] == p08.EXPECTED_REAL
    assert actual.summary["state_counts"] == {
        "documented_initial_return_approach": 12_434,
        "initial_return_approach_not_documented": 0,
        "unknown_initial_return": 885_892,
        "ineligible_censored": 528_537,
    }
    assert actual.summary["coverage"] == {
        "denominator_attempts": 1_426_863,
        "documented_initial_return_approach_attempts": 12_434,
        "documented_initial_return_approach_prevalence": 0.008714221337297275,
        "not_documented_attempts": 0,
        "unknown_attempts": 885_892,
        "censored_attempts": 528_537,
    }
    assert actual.summary["marker_context_counts"] == {
        "service_and_return_approach_documented": 682,
        "service_approach_documented_return_not_documented": 0,
        "return_approach_documented_service_not_documented": 11_752,
        "neither_documented": 0,
        "unknown_marker_context": 885_892,
        "ineligible_censored": 528_537,
    }
    assert actual.summary["outcome_comparison_status"] == "not_available"
    assert actual.summary["reason"] == "no_observable_nonapproach_comparator"
    assert actual.outcomes.empty
    assert actual.summary["test_seal"] == {
        "test_status": "sealed", "used_for_method_selection": False,
        "excluded_test_matches": 1_531, **{field: 0 for field in p08.TEST_ZERO_FIELDS},
    }
    combined = b"".join(path.read_bytes() for path in artifact_paths.values())
    assert b"NaN" not in combined and b"Infinity" not in combined and b"C:\\Users\\" not in combined
    assert b"generated_at" not in combined and b"timestamp" not in combined
    related_temporaries = [
        path for path in p08.REPORTS_DIR.rglob("*") if path.is_file()
        and (path.name.startswith(".return_approach_descriptive_feasibility")
             or path.name.startswith("repro_return_approach_descriptive_feasibility")
             or (path.suffix == ".tmp" and "return_approach" in path.name))
    ]
    assert related_temporaries == []
    assert "read_parquet" not in inspect.getsource(p08.verify_persisted_artifacts)


ARTIFACT_MUTATIONS = (
    "analysis_status", "outcome_status", "reason", "population", "first_attempts",
    "second_attempts", "attempts_total", "state", "prevalence", "denominator",
    "proportion", "marker_name", "marker_relation", "serve_number", "surface",
    "period", "future_fold", "row_removed", "row_duplicated", "row_added",
    "row_order", "column_order", "column_removed", "outcomes_row",
    *(f"test_counter:{field}" for field in p08.TEST_ZERO_FIELDS),
    "excluded_test_matches", "test_status", "used_for_method_selection",
    "infinity_literal", "required_key", "fingerprint_contract", "upstream_contract",
)


@pytest.mark.integration
@pytest.mark.parametrize("mutation", ARTIFACT_MUTATIONS)
def test_resigned_frozen_artifact_mutations_fail_semantically(tmp_path, mutation):
    summary, frames = load_frozen_payload()
    if mutation == "analysis_status": summary["analysis_status"] = "available_descriptive"
    elif mutation == "outcome_status": summary["outcome_comparison_status"] = "available"
    elif mutation == "reason": summary["reason"] = "invented"
    elif mutation == "population": summary["population"]["source_rows_read"] += 1
    elif mutation == "first_attempts": summary["population"]["first_attempts"] += 1
    elif mutation == "second_attempts": summary["population"]["second_attempts"] += 1
    elif mutation == "attempts_total": summary["population"]["attempts_total"] += 1
    elif mutation == "state":
        summary["state_counts"][p08.STATES[0]] += 1
        summary["state_counts"][p08.STATES[2]] -= 1
    elif mutation == "prevalence": summary["coverage"]["documented_initial_return_approach_prevalence"] += 0.01
    elif mutation == "denominator": frames["by_state"].loc[0, "denominator_attempts"] += 1
    elif mutation == "proportion": frames["by_state"].loc[0, "proportion"] += 0.01
    elif mutation == "marker_name": frames["marker_context"].loc[0, "marker_context"] = "invented"
    elif mutation == "marker_relation":
        frames["marker_context"].loc[0, "attempts"] -= 1
        frames["marker_context"].loc[0, "proportion"] = frames["marker_context"].loc[0, "attempts"] / 1_426_863
        frames["marker_context"].loc[4, "attempts"] += 1
        frames["marker_context"].loc[4, "proportion"] = frames["marker_context"].loc[4, "attempts"] / 1_426_863
        summary["marker_context_counts"][p08.MARKER_CONTEXTS[0]] -= 1
        summary["marker_context_counts"][p08.MARKER_CONTEXTS[4]] += 1
    elif mutation == "serve_number": frames["by_group"].loc[1, "serve_number"] = 2
    elif mutation == "surface": frames["by_group"].loc[3, "surface"] = "Carpet"
    elif mutation == "period": frames["by_group"].loc[6, "derived_period"] = "bad"
    elif mutation == "future_fold": frames["by_group"].loc[10, "validation_fold"] = "2024"
    elif mutation == "row_removed": frames["by_group"] = frames["by_group"].iloc[:-1].copy()
    elif mutation == "row_duplicated": frames["by_state"] = pd.concat([frames["by_state"], frames["by_state"].iloc[[0]]], ignore_index=True)
    elif mutation == "row_added": frames["marker_context"] = pd.concat([frames["marker_context"], frames["marker_context"].iloc[[0]]], ignore_index=True)
    elif mutation == "row_order": frames["by_group"] = frames["by_group"].iloc[::-1].reset_index(drop=True)
    elif mutation == "column_order": frames["by_group"] = frames["by_group"][list(reversed(frames["by_group"].columns))]
    elif mutation == "column_removed": frames["by_state"] = frames["by_state"].drop(columns="matches")
    elif mutation == "outcomes_row":
        frames["outcomes"] = pd.DataFrame([{
            "state": p08.STATES[0], "attempts": 12_434, "matches": 1,
            "servers": 1, "returners": 1, "returner_wins": 1,
            "returner_win_rate": 1 / 12_434, "wilson_low": 0.0, "wilson_high": 1.0,
        }], columns=p08.OUTCOME_COLUMNS)
    elif mutation.startswith("test_counter:"): summary["test_seal"][mutation.split(":", 1)[1]] = 1
    elif mutation == "excluded_test_matches": summary["test_seal"]["excluded_test_matches"] += 1
    elif mutation == "test_status": summary["test_seal"]["test_status"] = "opened"
    elif mutation == "used_for_method_selection": summary["test_seal"]["used_for_method_selection"] = True
    elif mutation == "infinity_literal": frames["by_state"].loc[0, "reason_code"] = "Infinity"
    elif mutation == "required_key": del summary["coverage"]
    elif mutation == "fingerprint_contract": summary["fingerprint_contract"]["version"] = "2"
    else: summary["upstream_contract"]["extractor_commit"] = "foreign"
    destination = paths(tmp_path)
    write_resigned_payload(summary, frames, destination)
    with pytest.raises((p08.FeasibilityContractError, ValueError)):
        p08.verify_persisted_artifacts(
            summary_path=destination[0], by_state_path=destination[1],
            by_group_path=destination[2], marker_context_path=destination[3],
            outcomes_path=destination[4],
        )
