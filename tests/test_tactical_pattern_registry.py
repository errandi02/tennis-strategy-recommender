from __future__ import annotations

import ast
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re

import pytest

from src.analysis import tactical_pattern_registry as registry


def _set_nested(target: dict, path: tuple[str, ...], value: object) -> None:
    cursor = target
    for component in path[:-1]:
        cursor = cursor.setdefault(component, {})
    cursor[path[-1]] = value


def _synthetic_summary(spec: registry.SourceSpec, eligible: int, denominator: int) -> dict:
    seal_key = "chronological_seal" if spec.pattern_id == "P02" else "test_seal"
    excluded_key = (
        "test_target_matches_excluded_before_construction"
        if spec.pattern_id == "P02"
        else "excluded_test_matches"
    )
    summary = {
        "analysis_status": spec.analysis_status,
        "population": {
            "development_matches": 5993,
            "development_point_rows": 1035760,
            "excluded_test_matches": 1531,
        },
        seal_key: {
            "test_status": "sealed",
            "used_for_method_selection": False,
            excluded_key: 1531,
            "test_evaluation_runs": 0,
        },
        "reconciliations": {"synthetic": True},
        "publication_fingerprint": spec.publication_fingerprint,
    }
    _set_nested(summary, spec.eligible_path, eligible)
    _set_nested(summary, spec.denominator_path, denominator)
    if spec.coverage_path:
        _set_nested(summary, spec.coverage_path, eligible / denominator)
    if spec.state_path:
        _set_nested(summary, spec.state_path, eligible)
    return summary


def _synthetic_sources() -> tuple[registry.VerifiedSource, ...]:
    sources = []
    for offset, spec in enumerate(registry.SOURCE_SPECS, start=1):
        denominator = 1000 + offset
        eligible = 100 + offset
        sources.append(
            registry.VerifiedSource(spec, registry._freeze(_synthetic_summary(spec, eligible, denominator)))
        )
    return tuple(sources)


def _valid_result() -> registry.RegistryResult:
    sources = _synthetic_sources()
    rows = tuple(registry._row_from_source(source) for source in sources)
    return registry._finalize(rows, sources)


def _resigned_rows(result: registry.RegistryResult, rows: list[dict]) -> registry.RegistryResult:
    frozen_rows = tuple(registry._freeze(row) for row in rows)
    csv_payload = registry._registry_csv_bytes(frozen_rows)
    summary = registry._summary_core(frozen_rows, result.sources, csv_payload)
    summary["publication_fingerprint"] = registry._fingerprint(summary, csv_payload)
    return registry.RegistryResult(registry._freeze(summary), frozen_rows, result.sources)


def _resigned_summary(result: registry.RegistryResult, mutate) -> registry.RegistryResult:
    summary = registry._thaw(result.summary)
    mutate(summary)
    csv_payload = registry._registry_csv_bytes(result.rows)
    summary["publication_fingerprint"] = registry._fingerprint(summary, csv_payload)
    return registry.RegistryResult(registry._freeze(summary), result.rows, result.sources)


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        ({"upstream_available": False, "explicitly_comparable": True, "diagnostic_only": False, "sufficient_identifiability": True, "usable_as_explainable_input": True}, "not_available"),
        ({"upstream_available": True, "explicitly_comparable": False, "diagnostic_only": False, "sufficient_identifiability": True, "usable_as_explainable_input": True}, "blocked_no_comparator"),
        ({"upstream_available": True, "explicitly_comparable": True, "diagnostic_only": True, "sufficient_identifiability": True, "usable_as_explainable_input": True}, "diagnostic_only"),
        ({"upstream_available": True, "explicitly_comparable": True, "diagnostic_only": False, "sufficient_identifiability": False, "usable_as_explainable_input": True}, "blocked_low_identifiability"),
        ({"upstream_available": True, "explicitly_comparable": True, "diagnostic_only": False, "sufficient_identifiability": True, "usable_as_explainable_input": True}, "eligible_component"),
        ({"upstream_available": True, "explicitly_comparable": True, "diagnostic_only": False, "sufficient_identifiability": True, "usable_as_explainable_input": False}, "diagnostic_only"),
    ],
)
def test_readiness_is_pure_and_property_driven(properties: dict, expected: str) -> None:
    assert registry.recommendation_readiness(**properties) == expected
    # pattern_id no forma parte de la API ni de la decision.
    assert "pattern_id" not in registry.recommendation_readiness.__annotations__


@pytest.mark.parametrize("bad", [1, 0, "true", None, 1.0])
def test_readiness_rejects_coercible_booleans(bad: object) -> None:
    with pytest.raises(TypeError, match="bool real"):
        registry.recommendation_readiness(
            upstream_available=bad,  # type: ignore[arg-type]
            explicitly_comparable=True,
            diagnostic_only=False,
            sufficient_identifiability=True,
            usable_as_explainable_input=True,
        )


def test_registry_has_exact_eight_ordered_rows_and_contractual_readiness() -> None:
    result = _valid_result()
    assert tuple(row["pattern_id"] for row in result.rows) == registry.PATTERN_IDS
    assert tuple(tuple(row.keys()) for row in result.rows) == (registry.CSV_COLUMNS,) * 8
    assert [row["recommendation_readiness"] for row in result.rows] == [
        "eligible_component",
        "blocked_no_comparator",
        "eligible_component",
        "eligible_component",
        "eligible_component",
        "diagnostic_only",
        "blocked_no_comparator",
        "eligible_component",
    ]
    assert result.summary["eligible_components"] == ("P02", "P04", "P05", "P06", "P09")
    assert result.summary["diagnostic_only_patterns"] == ("P07",)
    assert result.summary["blocked_patterns"] == ("P03", "P08")


def test_serialization_is_deterministic_utf8_and_without_index() -> None:
    result = _valid_result()
    first = registry.serialize_registry_result(result)
    second = registry.serialize_registry_result(result)
    assert first == second
    assert first[0].endswith(b"\n") and first[1].endswith(b"\n")
    assert first[1].decode("utf-8").splitlines()[0].split(",") == list(registry.CSV_COLUMNS)
    assert "Unnamed" not in first[1].decode("utf-8")
    assert b"NaN" not in b"".join(first) and b"Infinity" not in b"".join(first)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(2),
        lambda rows: rows.append(dict(rows[0])),
        lambda rows: rows.__setitem__(1, dict(rows[0])),
        lambda rows: rows.reverse(),
    ],
    ids=["missing", "additional", "duplicate", "reversed"],
)
def test_registry_rejects_bad_row_set_even_when_resigned(mutation) -> None:
    result = _valid_result()
    rows = [registry._thaw(row) for row in result.rows]
    mutation(rows)
    with pytest.raises(registry.RegistryContractError):
        candidate = _resigned_rows(result, rows)
        registry.validate_registry_result(candidate)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("source_summary_path", "C:/absolute/summary.json"),
        ("source_summary_path", "reports/not_allowlisted.json"),
        ("source_summary_sha256", "a" * 64),
        ("source_summary_sha256", "A" * 63),
        ("source_commit", "deadbeef"),
        ("analysis_status", "available"),
        ("evidence_level", "causal"),
        ("recommendation_readiness", "recommended"),
        ("unit", "row"),
        ("actor", "player"),
        ("has_observable_comparator", 1),
        ("outcomes_available", "true"),
        ("test_used_for_method_selection", 0),
        ("coverage", 0.999),
        ("eligible_count", 999999),
        ("denominator_count", 1),
        ("test_status", "open"),
        ("excluded_test_matches", 1530),
        ("principal_limitation", "alterada"),
    ],
)
def test_resigned_row_mutations_fail_semantically(column: str, value: object) -> None:
    result = _valid_result()
    rows = [registry._thaw(row) for row in result.rows]
    rows[0][column] = value
    with pytest.raises((registry.RegistryContractError, TypeError, ValueError)):
        candidate = _resigned_rows(result, rows)
        registry.validate_registry_result(candidate)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda summary: summary.__setitem__("registry_status", "not_available"),
        lambda summary: summary["counts_by_analysis_status"].__setitem__("available_descriptive", 0),
        lambda summary: summary["counts_by_evidence_level"].__setitem__("descriptive_conditioned", 0),
        lambda summary: summary["counts_by_recommendation_readiness"].__setitem__("eligible_component", 0),
        lambda summary: summary["eligible_components"].append("P03"),
        lambda summary: summary["blocked_patterns"].clear(),
        lambda summary: summary["reconciliations"].__setitem__("coverage_reconciled", False),
        lambda summary: summary["upstream_summaries"][0].__setitem__("summary_sha256", "B" * 64),
        lambda summary: summary["upstream_summaries"][0].__setitem__("publication_fingerprint", "B" * 64),
        lambda summary: summary["upstream_summaries"][0].__setitem__("source_commit", "0" * 40),
        lambda summary: summary["upstream_summaries"].reverse(),
    ],
    ids=[
        "status", "analysis_counts", "evidence_counts", "readiness_counts", "eligible_list",
        "blocked_list", "reconciliation", "upstream_hash", "upstream_fingerprint", "commit", "source_order",
    ],
)
def test_resigned_summary_mutations_fail_semantically(mutate) -> None:
    result = _valid_result()
    candidate = _resigned_summary(result, mutate)
    with pytest.raises(registry.RegistryContractError):
        registry.validate_registry_result(candidate)


def _fake_git(expected_commit: str, *, tracked: bool = True, dirty: bool = False):
    def run(_root: Path, *args: str) -> str:
        if args[:2] == ("ls-files", "--error-unmatch"):
            if not tracked:
                raise registry.RegistryContractError("untracked")
            return args[-1]
        if args[:2] == ("status", "--porcelain"):
            return " M " + args[-1] if dirty else ""
        if args[:3] == ("log", "-1", "--format=%H"):
            return expected_commit
        if args[:2] == ("merge-base", registry.EXPECTED_REPOSITORY_COMMIT):
            return registry.EXPECTED_REPOSITORY_COMMIT
        raise AssertionError(args)
    return run


def test_git_preflight_requires_frozen_upstream_ancestry(tmp_path: Path) -> None:
    def diverged(_root: Path, *args: str) -> str:
        assert args[:2] == ("merge-base", registry.EXPECTED_REPOSITORY_COMMIT)
        return "0" * 40
    with pytest.raises(registry.RegistryContractError, match="no desciende"):
        registry._git_preflight(tmp_path, diverged)


def _single_source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original = registry.SOURCE_BY_ID["P05"]
    summary_path = "aggregates/p05.json"
    csv_path = "aggregates/p05.csv"
    summary = _synthetic_summary(original, 3, 10)
    summary["publication_fingerprint"] = "A" * 64
    summary_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode()
    csv_bytes = b"group_type,observed_depth_numerator,observed_depth_denominator,observed_depth_coverage\ntotal,3,10,0.3\n"
    (tmp_path / "aggregates").mkdir()
    (tmp_path / summary_path).write_bytes(summary_bytes)
    (tmp_path / csv_path).write_bytes(csv_bytes)
    spec = replace(
        original,
        summary_path=summary_path,
        summary_size=len(summary_bytes),
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest().upper(),
        publication_fingerprint="A" * 64,
        source_commit="1" * 40,
        artifacts=(registry.ArtifactSpec(csv_path, len(csv_bytes), hashlib.sha256(csv_bytes).hexdigest().upper(), "by_group"),),
        outcome_artifact=None,
        outcomes_available=False,
    )
    monkeypatch.setattr(registry, "SUMMARY_ALLOWLIST", frozenset({summary_path}))
    monkeypatch.setattr(registry, "ARTIFACT_ALLOWLIST", frozenset({csv_path}))
    return spec, summary_path, csv_path


def test_synthetic_upstream_source_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _, _ = _single_source_fixture(tmp_path, monkeypatch)
    source = registry._load_source(spec, tmp_path, _fake_git(spec.source_commit))
    assert source.spec.pattern_id == "P05"
    assert registry._nested(source.summary, spec.eligible_path) == 3


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("missing", "Falta"),
        ("hash", "SHA-256"),
        ("size", "SHA-256"),
        ("fingerprint", "Fingerprint"),
        ("status", "Status"),
        ("test_open", "sellado"),
        ("test_used", "seleccion"),
        ("test_count", "contador"),
        ("test_count_coercible", "contador"),
        ("untracked", "untracked"),
        ("dirty", "difiere"),
        ("commit", "commit"),
    ],
)
def test_upstream_failures_are_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, expected: str
) -> None:
    spec, summary_path, _ = _single_source_fixture(tmp_path, monkeypatch)
    git_runner = _fake_git(spec.source_commit)
    if failure == "missing":
        (tmp_path / summary_path).unlink()
    elif failure == "hash":
        (tmp_path / summary_path).write_bytes(b"changed")
    elif failure == "size":
        spec = replace(spec, summary_size=spec.summary_size + 1)
    elif failure in {"fingerprint", "status", "test_open", "test_used", "test_count", "test_count_coercible"}:
        summary = json.loads((tmp_path / summary_path).read_text())
        if failure == "fingerprint":
            summary["publication_fingerprint"] = "B" * 64
        elif failure == "status":
            summary["analysis_status"] = "unknown"
        elif failure == "test_open":
            summary["test_seal"]["test_status"] = "open"
        elif failure == "test_used":
            summary["test_seal"]["used_for_method_selection"] = True
        elif failure == "test_count":
            summary["test_seal"]["test_evaluation_runs"] = 1
        else:
            summary["test_seal"]["test_evaluation_runs"] = "0"
        payload = (json.dumps(summary, sort_keys=True) + "\n").encode()
        (tmp_path / summary_path).write_bytes(payload)
        spec = replace(spec, summary_size=len(payload), summary_sha256=hashlib.sha256(payload).hexdigest().upper())
    elif failure == "untracked":
        git_runner = _fake_git(spec.source_commit, tracked=False)
    elif failure == "dirty":
        git_runner = _fake_git(spec.source_commit, dirty=True)
    elif failure == "commit":
        git_runner = _fake_git("2" * 40)
    with pytest.raises(registry.RegistryContractError, match=expected):
        registry._load_source(spec, tmp_path, git_runner)


def test_not_available_has_header_only_and_sanitized_failure() -> None:
    error = registry.RegistryContractError(
        "fallo en C:/Users/name/private.csv", stage="upstream_hash_validation",
        reason_code="upstream_hash_mismatch", pattern_id="P05", artifact_path="reports/p05.json",
    )
    result = registry.not_available_result(error)
    summary_bytes, csv_bytes = registry.serialize_registry_result(result)
    assert result.summary["registry_status"] == "not_available"
    assert result.rows == ()
    assert len(csv_bytes.decode().splitlines()) == 1
    assert "C:/Users" not in summary_bytes.decode()
    assert result.summary["failure"]["pattern_id"] == "P05"
    assert result.summary["reason_codes"] == ("upstream_hash_mismatch",)


def test_round_trip_parser_preserves_schema_and_values() -> None:
    result = _valid_result()
    _, payload = registry.serialize_registry_result(result)
    parsed = registry._parse_registry_csv(payload)
    assert tuple(registry._thaw(row) for row in parsed) == tuple(registry._thaw(row) for row in result.rows)


def test_persisted_resigned_semantic_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _valid_result()
    rows = [registry._thaw(row) for row in result.rows]
    rows[0]["recommendation_readiness"] = "diagnostic_only"
    candidate = _resigned_rows(result, rows)
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "registry.csv"
    summary_path.write_bytes(registry._json_bytes(registry._thaw(candidate.summary)))
    csv_path.write_bytes(registry._registry_csv_bytes(candidate.rows))
    by_id = {source.spec.pattern_id: source for source in result.sources}
    monkeypatch.setattr(registry, "_git_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        registry,
        "_load_source",
        lambda spec, _root, _runner: by_id[spec.pattern_id],
    )
    with pytest.raises(registry.RegistryContractError, match="contratos upstream"):
        registry.verify_persisted_registry_artifacts(
            summary_path=summary_path,
            registry_csv_path=csv_path,
            repository_root=tmp_path,
            git_runner=lambda *_args: "",
        )


def _write_with_mock_verify(
    monkeypatch: pytest.MonkeyPatch, result: registry.RegistryResult, summary_path: Path, csv_path: Path
) -> None:
    def verify(**kwargs):
        assert kwargs["summary_path"].read_bytes() == registry.serialize_registry_result(result)[0]
        assert kwargs["registry_csv_path"].read_bytes() == registry.serialize_registry_result(result)[1]
        return result
    monkeypatch.setattr(registry, "verify_persisted_registry_artifacts", verify)
    registry.write_registry_artifacts(result, summary_path=summary_path, registry_csv_path=csv_path)


def test_atomic_write_succeeds_and_leaves_no_temporaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _valid_result()
    summary_path, csv_path = tmp_path / "summary.json", tmp_path / "registry.csv"
    _write_with_mock_verify(monkeypatch, result, summary_path, csv_path)
    assert (summary_path.read_bytes(), csv_path.read_bytes()) == registry.serialize_registry_result(result)
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure_call", [1, 2])
@pytest.mark.parametrize("with_previous", [False, True])
def test_staging_failures_preserve_previous_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_call: int, with_previous: bool
) -> None:
    result = _valid_result()
    paths = (tmp_path / "summary.json", tmp_path / "registry.csv")
    if with_previous:
        for path, content in zip(paths, (b"old-summary", b"old-csv")):
            path.write_bytes(content)
    original = registry._stage_payload
    calls = 0
    def stage(path: Path, payload: bytes):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("staging failed")
        return original(path, payload)
    monkeypatch.setattr(registry, "_stage_payload", stage)
    with pytest.raises(OSError, match="staging failed"):
        registry.write_registry_artifacts(result, summary_path=paths[0], registry_csv_path=paths[1])
    if with_previous:
        assert paths[0].read_bytes() == b"old-summary" and paths[1].read_bytes() == b"old-csv"
    else:
        assert not paths[0].exists() and not paths[1].exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure_call", [1, 2])
@pytest.mark.parametrize("with_previous", [False, True])
def test_replace_failures_rollback_with_and_without_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_call: int, with_previous: bool
) -> None:
    result = _valid_result()
    paths = (tmp_path / "summary.json", tmp_path / "registry.csv")
    if with_previous:
        paths[0].write_bytes(b"old-summary")
        paths[1].write_bytes(b"old-csv")
    original = registry.os.replace
    calls = 0
    def fail_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("replace failed")
        return original(source, destination)
    monkeypatch.setattr(registry.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        registry.write_registry_artifacts(result, summary_path=paths[0], registry_csv_path=paths[1])
    if with_previous:
        assert paths[0].read_bytes() == b"old-summary" and paths[1].read_bytes() == b"old-csv"
    else:
        assert not paths[0].exists() and not paths[1].exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("with_previous", [False, True])
def test_post_write_verification_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_previous: bool
) -> None:
    result = _valid_result()
    paths = (tmp_path / "summary.json", tmp_path / "registry.csv")
    if with_previous:
        paths[0].write_bytes(b"old-summary")
        paths[1].write_bytes(b"old-csv")
    def fail(**_kwargs):
        raise registry.RegistryContractError("verify failed", stage="publication_verification")
    monkeypatch.setattr(registry, "verify_persisted_registry_artifacts", fail)
    with pytest.raises(registry.RegistryContractError, match="verify failed"):
        registry.write_registry_artifacts(result, summary_path=paths[0], registry_csv_path=paths[1])
    if with_previous:
        assert paths[0].read_bytes() == b"old-summary" and paths[1].read_bytes() == b"old-csv"
    else:
        assert not paths[0].exists() and not paths[1].exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_simulated_nondeterminism_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _valid_result()
    original = registry._serialize_once
    calls = 0
    def unstable(candidate):
        nonlocal calls
        calls += 1
        summary, table = original(candidate)
        return (summary + (b" " if calls == 2 else b""), table)
    monkeypatch.setattr(registry, "_serialize_once", unstable)
    with pytest.raises(registry.RegistryContractError, match="no determinista"):
        registry.serialize_registry_result(result)


def test_static_security_contract_contains_no_analytical_io() -> None:
    path = Path(registry.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "read_parquet" not in calls
    assert not {"pandas", "pyarrow"} & imports
    assert not re.search(r"(?:^|[\"'])data/(?:raw|processed)/", source)
    assert "snapshot" not in source.lower()
    assert "feature" not in source.lower()
    assert "datetime.now" not in source and "utcnow" not in source


@pytest.mark.integration
def test_real_published_upstreams_build_and_persisted_contract(tmp_path: Path) -> None:
    result = registry.build_registry()
    assert result.summary["registry_status"] == "available_registry"
    assert len(result.rows) == 8
    assert [row["eligible_count"] for row in result.rows] == [
        1019891, 130963, 830471, 601246, 881717, 74178, 12434, 601246
    ]
    assert [row["denominator_count"] for row in result.rows] == [
        1035760, 1426863, 1426863, 1426863, 1426863, 1426863, 1426863, 1426863
    ]
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "registry.csv"
    registry.write_registry_artifacts(result, summary_path=summary_path, registry_csv_path=csv_path)
    reopened = registry.verify_persisted_registry_artifacts(summary_path=summary_path, registry_csv_path=csv_path)
    assert registry.serialize_registry_result(reopened) == (summary_path.read_bytes(), csv_path.read_bytes())
