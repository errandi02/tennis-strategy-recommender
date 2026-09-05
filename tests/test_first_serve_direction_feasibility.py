"""Pruebas sintéticas del estudio descriptivo P02.

La integración sólo valida artefactos ya publicados: no reabre el Parquet ni
repite el análisis durante la suite completa.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import first_serve_direction_feasibility as feasibility
from src.parsing.serve_sequence import (
    ServeAce,
    ServeFault,
    ServeUnreturned,
    SpecialCodeStructure,
    parse_sequence,
)


def synthetic_points() -> pd.DataFrame:
    """Fixture escrito literalmente; no deriva expectativas del código P02."""
    matches = [
        ("m01", "2009-12-31", "Hard", "Alice", "Bob", [
            (1, 1, 1, "4*"), (2, 1, 1, "5#"), (3, 1, 2, "6n"),
            (4, 1, 2, "4f3*"), (5, 1, 1, "0"), (6, 1, 1, "S"),
        ]),
        ("m02", "2015-06-01", "Clay", "Bob", "Cara", [
            (1, 1, 1, "4"), (2, 1, 2, "5w"), (3, 1, 1, "6?"),
            (4, 1, 2, "R"), (5, 1, 1, "?"),
        ]),
        ("m03", "2020-07-01", "Grass", "Cara", "Dan", [
            (1, 1, 1, "4d"), (2, 1, 2, "5x"), (3, 1, 1, "6g"),
            (4, 1, 2, "P"), (5, 1, 1, "Q"), (6, 1, 2, "V"),
        ]),
        ("m04", "2021-01-03", "Hard", "Eve", "Alice", [
            (1, 2, 2, "4e"), (2, 2, 1, "5!"), (3, 2, 2, "6"),
            (4, 1, 1, " "), (5, 1, 1, ""), (6, 1, 1, None),
        ]),
    ]
    rows = []
    for match_id, date, surface, player_1, player_2, points in matches:
        for point_number, server, winner, first_serve in points:
            rows.append({
                "match_id": match_id, "point_number": point_number, "date": date,
                "surface": surface, "server": server, "point_winner": winner,
                "player_1": player_1, "player_2": player_2,
                "first_serve": first_serve,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_result() -> feasibility.FeasibilityResult:
    return feasibility.analyze_development_points(synthetic_points())


def test_first_serve_parser_contract_and_immediate_outcomes() -> None:
    assert parse_sequence("4*", serve_number=1).structure.__class__ is ServeAce
    assert parse_sequence("5#", serve_number=1).structure.__class__ is ServeUnreturned
    for code in "nwdxge!":
        assert parse_sequence(f"6{code}", serve_number=1).structure.__class__ is ServeFault
    # El rally posterior no transforma el prefijo de saque en un fallo.
    assert parse_sequence("4f3*", serve_number=1).structure.direction == "4"
    assert parse_sequence("6?", serve_number=1).structure.direction == "6"
    assert parse_sequence("0", serve_number=1).structure.direction == "0"
    for code in "SRPQV":
        parsed = parse_sequence(code, serve_number=1)
        assert isinstance(parsed.structure, SpecialCodeStructure)
        assert parsed.structure.code == code
    assert parse_sequence(" 4*", serve_number=1).residual_text == " 4*"
    assert parse_sequence("4*", serve_number=2).serve_number == 2


def test_canonical_chronology_contract_accepts_current_version_and_is_eol_portable() -> None:
    payload = feasibility.CHRONOLOGY_PATH.read_bytes()
    crlf = payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    parsed = json.loads(payload)
    reordered = json.dumps(dict(reversed(list(parsed.items()))), ensure_ascii=False, indent=3).encode("utf-8")
    assert feasibility.canonical_json_sha256(payload) == feasibility.EXPECTED_CHRONOLOGY_CANONICAL_SHA256
    assert feasibility.canonical_json_sha256(crlf) == feasibility.EXPECTED_CHRONOLOGY_CANONICAL_SHA256
    assert feasibility.canonical_json_sha256(reordered) == feasibility.EXPECTED_CHRONOLOGY_CANONICAL_SHA256
    contract = feasibility.validate_chronological_contract()
    assert contract["reconciliations"]["unique_matches"] == 7_524


def test_upstream_contract_rejects_wrong_hash_commit_and_nonfinite_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = feasibility.CHRONOLOGY_PATH.read_bytes()
    monkeypatch.setattr(feasibility, "EXPECTED_CHRONOLOGY_CANONICAL_SHA256", "0" * 64)
    with pytest.raises(feasibility.FeasibilityContractError, match="canonico"):
        feasibility.validate_chronological_contract()
    monkeypatch.undo()
    monkeypatch.setattr(feasibility, "EXPECTED_CHRONOLOGY_COMMIT", "0" * 40)
    with pytest.raises(feasibility.FeasibilityContractError, match="Commit/blob"):
        feasibility.validate_chronological_contract()
    for bad in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'\xef\xbb\xbf{}'):
        with pytest.raises(feasibility.FeasibilityContractError):
            feasibility.canonical_json_bytes(bad)
    assert payload


@pytest.mark.parametrize("mutation", [
    lambda contract: contract["split_population"][0].update(matches=4_187),
    lambda contract: contract["rolling_folds"].reverse(),
    lambda contract: contract["protected_test_contract"].update(test_status="open"),
    lambda contract: contract["protected_test_contract"].update(test_evaluation_runs=1),
    lambda contract: contract["date_contract"].update(last_date="2026-05-22"),
    lambda contract: contract["yearly_contract"].update(partial_year_definition="complete"),
    lambda contract: contract["source_contract"].update(path="C:/absolute/path.parquet"),
    lambda contract: contract["date_contract"].update(generated_at="2026-01-01T00:00:00"),
])
def test_upstream_contract_rejects_semantic_mutations_even_when_hash_is_rebound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation,
) -> None:
    contract = json.loads(feasibility.CHRONOLOGY_PATH.read_text(encoding="utf-8"))
    mutation(contract)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True), encoding="utf-8", newline="\n")
    monkeypatch.setattr(feasibility, "EXPECTED_CHRONOLOGY_CANONICAL_SHA256", feasibility.canonical_json_sha256(path.read_bytes()))
    with pytest.raises(feasibility.FeasibilityContractError):
        feasibility.validate_chronological_contract(path=path)


@pytest.mark.parametrize("mutation", [
    lambda frame: frame.__setitem__("evaluation_matches", frame["evaluation_matches"].where(~frame["fold_id"].eq("validation_2020"), 169)),
    lambda frame: frame.sort_values("fold_id", ascending=False, inplace=True),
])
def test_upstream_contract_rejects_fold_table_mutations(
    tmp_path: Path, mutation,
) -> None:
    folds = pd.read_csv(feasibility.CHRONOLOGY_FOLDS_PATH)
    mutation(folds)
    path = tmp_path / "folds.csv"; folds.to_csv(path, index=False)
    with pytest.raises(feasibility.FeasibilityContractError):
        feasibility.validate_chronological_contract(folds_path=path)


def test_invalid_upstream_contract_never_reads_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feasibility, "EXPECTED_CHRONOLOGY_CANONICAL_SHA256", "0" * 64)
    monkeypatch.setattr(feasibility, "read_development_points", lambda: (_ for _ in ()).throw(AssertionError("Parquet read")))
    with pytest.raises(feasibility.FeasibilityContractError, match="canonico"):
        feasibility.run_real_analysis()


def test_population_outcomes_exclusions_and_cache_are_exact() -> None:
    calls: list[tuple[str, int]] = []
    def counting_parser(value: str, serve_number: int):
        calls.append((value, serve_number))
        return parse_sequence(value, serve_number=serve_number)

    frame, directional, prepared = feasibility.prepare_population(synthetic_points(), parser=counting_parser)
    # 13 actionable rows: wide=5, body=4, T=4.  Four outcomes are exhaustive.
    assert len(frame) == 23
    assert len(directional) == 13
    assert directional.direction.value_counts().to_dict() == {"wide": 5, "body": 4, "T": 4}
    assert directional.immediate_outcome.value_counts().to_dict() == {
        "fault": 7, "continuation_or_other": 4, "ace": 1, "unreturned": 1,
    }
    assert prepared["exclusion_counts"] == {
        "actionable_direction": 13, "direction_unknown_0": 1,
        "special_unit_sequence": 5, "unrecognized_or_no_initial_direction": 1,
        "null_sequence": 1, "empty_sequence": 1, "whitespace_only_sequence": 1,
    }
    assert all(number == 1 for _, number in calls)
    assert len(calls) == len({value for value in frame.first_serve if isinstance(value, str) and value and not value.isspace()})


def test_aggregates_reconcile_by_direction_surface_and_period(synthetic_result: feasibility.FeasibilityResult) -> None:
    result = synthetic_result
    pooled = result.by_direction.query("dimension == 'pooled'")
    assert pooled[["direction", "analytical_points", "matches", "server_wins"]].to_dict("records") == [
        {"direction": "wide", "analytical_points": 5, "matches": 4, "server_wins": 4},
        {"direction": "body", "analytical_points": 4, "matches": 4, "server_wins": 1},
        {"direction": "T", "analytical_points": 4, "matches": 4, "server_wins": 3},
    ]
    assert int(result.outcomes.query("dimension == 'pooled'").points.sum()) == 13
    assert int(result.by_direction.query("dimension == 'surface'").analytical_points.sum()) == 13
    assert int(result.by_direction.query("dimension == 'derived_period'").analytical_points.sum()) == 13
    assert result.summary["reconciliations"]["outcomes_reconciled_by_direction_surface_period"] is True
    assert result.summary["immediate_outcome_contract"]["causal_interpretation"] is False


def test_coverage_has_global_surface_quantiles_zero_evidence_and_inclusive_cuts(synthetic_result: feasibility.FeasibilityResult) -> None:
    result = synthetic_result
    coverage = result.coverage
    assert coverage.shape[0] == 4 * 3 * (3 + 3 + 9)
    global_wide_25 = coverage.query("scope == 'global' and direction == 'wide' and cut_kind == 'point_only' and point_cut == 25")
    assert global_wide_25.iloc[0][["possible_servers", "observed_servers", "zero_evidence_servers", "eligible_servers"]].tolist() == [4, 3, 1, 0]
    assert global_wide_25.iloc[0]["directional_points"] == 5
    assert global_wide_25.iloc[0]["first_date"] == "2009-12-31"
    assert global_wide_25.iloc[0]["last_date"] == "2021-01-03"
    inclusive = coverage.query("scope == 'global' and direction == 'wide' and cut_kind == 'joint' and point_cut == 0 and match_cut == 0")
    assert inclusive.empty  # Contract never represents a no-cut row.
    assert set(coverage.cut_kind) == {"point_only", "match_only", "joint"}
    assert set(coverage.loc[coverage.cut_kind.eq("joint"), ["point_cut", "match_cut"]].itertuples(index=False, name=None)) == {
        (25, 3), (25, 5), (25, 10), (50, 3), (50, 5), (50, 10), (100, 3), (100, 5), (100, 10),
    }


@pytest.mark.parametrize("field,value", [
    ("surface", "Carpet"), ("server", True), ("server", 1.0),
    ("server", "1"), ("point_winner", False), ("point_winner", 2.0),
])
def test_invalid_domains_are_rejected(field: str, value: object) -> None:
    points = synthetic_points(); points[field] = points[field].astype(object); points.loc[0, field] = value
    with pytest.raises(feasibility.FeasibilityContractError):
        feasibility.validate_development_points(points)


def test_null_surface_invalid_date_later_date_and_inconsistent_metadata_are_rejected() -> None:
    for field, value in (("surface", None), ("date", "not-a-date"), ("date", "2024-01-01")):
        points = synthetic_points(); points.loc[0, field] = value
        with pytest.raises(feasibility.FeasibilityContractError):
            feasibility.validate_development_points(points)
    points = synthetic_points(); points.loc[1, "surface"] = "Clay"
    with pytest.raises(feasibility.FeasibilityContractError):
        feasibility.validate_development_points(points)


def test_period_invalid_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feasibility, "derive_period", lambda _: "not-a-period")
    with pytest.raises(feasibility.FeasibilityContractError):
        feasibility.validate_development_points(synthetic_points())


def test_deterministic_serialization_order_keys_and_fingerprint(synthetic_result: feasibility.FeasibilityResult) -> None:
    first = synthetic_result
    second = feasibility.analyze_development_points(synthetic_points().sample(frac=1, random_state=7))
    assert feasibility.serialize_artifacts(first) == feasibility.serialize_artifacts(second)
    for frame, columns in ((first.by_direction, feasibility.BY_DIRECTION_COLUMNS), (first.outcomes, feasibility.OUTCOME_COLUMNS), (first.coverage, feasibility.COVERAGE_COLUMNS)):
        assert frame.columns.tolist() == columns
        assert not any(column.startswith("Unnamed") for column in frame.columns)
    assert first.summary["chronological_seal"]["test_status"] == "sealed"
    assert all(first.summary["chronological_seal"][key] == 0 for key in feasibility.TEST_ZERO_FIELDS)


def test_mutation_with_recalculated_fingerprint_fails_semantic_contract(synthetic_result: feasibility.FeasibilityResult) -> None:
    result = synthetic_result
    altered = result.coverage.copy(deep=True); altered.loc[0, "eligible_servers"] += 1
    summary = copy.deepcopy(result.summary)
    payloads = tuple(feasibility._frame_bytes(frame) for frame in (result.by_direction, result.outcomes, altered))
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(value).hexdigest().upper() for name, value in zip(("by_direction", "outcomes", "coverage"), payloads)}
    summary["publication_fingerprint"] = feasibility._fingerprint(summary, payloads)
    tampered = feasibility.FeasibilityResult(summary, result.by_direction, result.outcomes, altered, result.development, result.directional, summary["publication_fingerprint"])
    with pytest.raises(feasibility.FeasibilityContractError, match="derivacion canonica"):
        feasibility.validate_result(tampered)


def test_not_available_has_only_headers_and_serializes_without_nonfinite_values() -> None:
    result = feasibility.not_available_result(ValueError(f"{feasibility.ROOT}\\bad"))
    feasibility.validate_result(result)
    payloads = feasibility.serialize_artifacts(result)
    assert all(frame.empty for frame in (result.by_direction, result.outcomes, result.coverage))
    assert b"NaN" not in payloads[0] and b"Infinity" not in payloads[0]
    assert "<ROOT>" in result.summary["failure"]["message"]


@pytest.mark.parametrize("failure_index", range(4))
@pytest.mark.parametrize("with_previous", [False, True])
def test_atomic_publication_rolls_back_each_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int, with_previous: bool,
    synthetic_result: feasibility.FeasibilityResult,
) -> None:
    paths = tuple(tmp_path / name for name in ("summary.json", "by.csv", "outcomes.csv", "coverage.csv"))
    expected_prior = {}
    if with_previous:
        for path in paths:
            path.write_bytes(b"previous:" + path.name.encode("ascii"))
    expected_prior = {path: path.read_bytes() if path.exists() else None for path in paths}
    original_replace = feasibility.os.replace
    calls = {"count": 0}
    def fail_on_one(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        if calls["count"] == failure_index:
            calls["count"] += 1
            raise OSError("forced replacement failure")
        calls["count"] += 1
        original_replace(source, target)
    monkeypatch.setattr(feasibility.os, "replace", fail_on_one)
    with pytest.raises(OSError, match="forced"):
        feasibility.write_artifacts(synthetic_result, summary_path=paths[0], by_direction_path=paths[1], outcomes_path=paths[2], coverage_path=paths[3])
    assert {path: path.read_bytes() if path.exists() else None for path in paths} == expected_prior
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_publication_and_persisted_validation(tmp_path: Path, synthetic_result: feasibility.FeasibilityResult) -> None:
    result = synthetic_result
    paths = tuple(tmp_path / name for name in ("summary.json", "by.csv", "outcomes.csv", "coverage.csv"))
    feasibility.write_artifacts(result, summary_path=paths[0], by_direction_path=paths[1], outcomes_path=paths[2], coverage_path=paths[3])
    feasibility.verify_persisted_artifacts(summary_path=paths[0], by_direction_path=paths[1], outcomes_path=paths[2], coverage_path=paths[3])
    assert all(path.exists() for path in paths)


def test_publication_reserializes_the_same_result_without_reanalysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_result: feasibility.FeasibilityResult) -> None:
    paths = tuple(tmp_path / name for name in ("summary.json", "by.csv", "outcomes.csv", "coverage.csv"))
    monkeypatch.setattr(feasibility, "analyze_development_points", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("second analysis")))
    calls = {"count": 0}
    original = feasibility.serialize_artifacts
    def counted(result: feasibility.FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes]:
        calls["count"] += 1
        return original(result)
    monkeypatch.setattr(feasibility, "serialize_artifacts", counted)
    feasibility.publish_artifacts(synthetic_result, summary_path=paths[0], by_direction_path=paths[1], outcomes_path=paths[2], coverage_path=paths[3])
    assert calls["count"] == 3
    assert tuple(path.read_bytes() for path in paths) == original(synthetic_result)
    assert not list(tmp_path.glob(".*.tmp"))


def test_nondeterministic_reserialization_fails_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_result: feasibility.FeasibilityResult) -> None:
    paths = tuple(tmp_path / name for name in ("summary.json", "by.csv", "outcomes.csv", "coverage.csv"))
    original = feasibility.serialize_artifacts
    calls = {"count": 0}
    def nondeterministic(result: feasibility.FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes]:
        calls["count"] += 1
        payloads = original(result)
        return payloads if calls["count"] == 1 else (payloads[0] + b" ", *payloads[1:])
    monkeypatch.setattr(feasibility, "serialize_artifacts", nondeterministic)
    with pytest.raises(feasibility.FeasibilityContractError, match="no determinista"):
        feasibility.publish_artifacts(synthetic_result, summary_path=paths[0], by_direction_path=paths[1], outcomes_path=paths[2], coverage_path=paths[3])
    assert not any(path.exists() for path in paths)
    assert not list(tmp_path.glob(".*.tmp"))


def test_not_available_publication_also_reserializes_the_same_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = tuple(tmp_path / name for name in ("summary.json", "by.csv", "outcomes.csv", "coverage.csv"))
    result = feasibility.not_available_result(ValueError("synthetic"))
    calls = {"count": 0}
    original = feasibility.serialize_artifacts
    def counted(value: feasibility.FeasibilityResult) -> tuple[bytes, bytes, bytes, bytes]:
        calls["count"] += 1
        return original(value)
    monkeypatch.setattr(feasibility, "serialize_artifacts", counted)
    feasibility.publish_artifacts(result, summary_path=paths[0], by_direction_path=paths[1], outcomes_path=paths[2], coverage_path=paths[3])
    assert calls["count"] == 3
    assert tuple(path.read_bytes() for path in paths) == original(result)


def test_staging_failure_removes_already_staged_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_result: feasibility.FeasibilityResult) -> None:
    paths = tuple(tmp_path / name for name in ("summary.json", "by.csv", "outcomes.csv", "coverage.csv"))
    original = feasibility._stage
    calls = {"count": 0}
    def fail_second_stage(path: Path, payload: bytes) -> Path:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("forced staging failure")
        return original(path, payload)
    monkeypatch.setattr(feasibility, "_stage", fail_second_stage)
    with pytest.raises(OSError, match="forced staging"):
        feasibility.write_artifacts(synthetic_result, summary_path=paths[0], by_direction_path=paths[1], outcomes_path=paths[2], coverage_path=paths[3])
    assert not any(path.exists() for path in paths)
    assert not list(tmp_path.glob(".*.tmp"))


def _copied_published_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sources = (feasibility.SUMMARY_PATH, feasibility.BY_DIRECTION_PATH, feasibility.OUTCOMES_PATH, feasibility.COVERAGE_PATH)
    paths = tuple(tmp_path / path.name for path in sources)
    for source, target in zip(sources, paths):
        target.write_bytes(source.read_bytes())
    return paths


def _rewrite_mutated_published_artifacts(paths: tuple[Path, Path, Path, Path], summary: dict, frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    for path, frame in zip(paths[1:], frames):
        path.write_bytes(feasibility._frame_bytes(frame))
    payloads = tuple(path.read_bytes() for path in paths[1:])
    summary["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(("by_direction", "outcomes", "coverage"), payloads)}
    summary["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(("by_direction", "outcomes", "coverage"), payloads)}
    summary["publication_fingerprint"] = feasibility._fingerprint(summary, payloads)
    paths[0].write_bytes((json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def test_published_artifacts_match_frozen_contract_without_parquet_access() -> None:
    feasibility.verify_persisted_artifacts(frozen_publication=True)


@pytest.mark.parametrize("mutation", [
    lambda summary, frames: summary.__setitem__("analysis_status", "not_available"),
    lambda summary, frames: summary["population"].__setitem__("directional_points", 1),
    lambda summary, frames: frames[0].loc.__setitem__((0, "direction"), "unknown"),
    lambda summary, frames: frames[1].loc.__setitem__((0, "immediate_outcome"), "unknown"),
    lambda summary, frames: frames[2].loc.__setitem__((0, "directional_points"), 1),
    lambda summary, frames: frames[2].loc.__setitem__((frames[2].index[frames[2].scope.eq("surface")][0], "surface"), "Carpet"),
    lambda summary, frames: frames[0].loc.__setitem__((frames[0].index[frames[0].dimension.eq("derived_period")][0], "derived_period"), "future"),
    lambda summary, frames: frames[0].sort_values("direction", ascending=False, inplace=True),
    lambda summary, frames: frames.__setitem__(0, pd.concat([frames[0], frames[0].iloc[[0]]], ignore_index=True)),
    lambda summary, frames: summary["chronological_seal"].__setitem__("test_status", "open"),
    lambda summary, frames: summary["chronological_seal"].__setitem__("test_evaluation_runs", 1),
    lambda summary, frames: summary["methodological_limits"].__setitem__(0, "causal"),
    lambda summary, frames: summary["upstream_chronological_contract"]["fold_matches"].__setitem__("2020", 1),
])
def test_semantic_validator_rejects_mutations_after_fingerprint_recalculation(tmp_path: Path, mutation) -> None:
    paths = _copied_published_artifacts(tmp_path)
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    frames = list(pd.read_csv(path, keep_default_na=False, na_values=["<NULL>"]) for path in paths[1:])
    mutation(summary, frames)
    _rewrite_mutated_published_artifacts(paths, summary, tuple(frames))
    with pytest.raises(feasibility.FeasibilityContractError):
        feasibility.verify_persisted_artifacts(*paths, semantic_contract=True)


def test_internal_payload_metadata_and_fingerprint_tampering_are_rejected(tmp_path: Path) -> None:
    paths = _copied_published_artifacts(tmp_path)
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    frames = tuple(pd.read_csv(path, keep_default_na=False, na_values=["<NULL>"]) for path in paths[1:])
    _rewrite_mutated_published_artifacts(paths, summary, frames)
    summary["artifact_payload_bytes"]["by_direction"] += 1
    payloads = tuple(path.read_bytes() for path in paths[1:])
    summary["publication_fingerprint"] = feasibility._fingerprint(summary, payloads)
    paths[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(feasibility.FeasibilityContractError):
        feasibility.verify_persisted_artifacts(*paths, semantic_contract=True)


@pytest.mark.integration
def test_real_published_artifacts_if_available() -> None:
    if not feasibility.SUMMARY_PATH.exists():
        pytest.skip("Los artefactos P02 aún no se han publicado.")
    # Esta integración verifica solo los artefactos persistidos; no abre el Parquet.
    feasibility.verify_persisted_artifacts(frozen_publication=True)
    summary = json.loads(feasibility.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["analysis_status"] in {"available_descriptive", "not_available"}
