"""Pruebas adversariales P13: proveedor persistido privado y verificable.

Bateria sintetica sobre el snapshot P13 (construccion, serializacion,
persistencia atomica, reconstruccion, provider inmutable, wire P12 y
arquitectura). No ejecuta P10, no lee datos reales, no usa red ni
sealed tests. Las identidades son sentinelas privadas de sintesis.
"""

from __future__ import annotations

import ast
import concurrent.futures
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.recommender.persisted_tactical_recommendation_provider as p13
from src.api.app import create_app
from src.recommender.tactical_prioritization import (
    PRIORITIZATION_CONTRACT_VERSION,
    default_tactical_scoring_policy,
    tactical_prioritization_result_fingerprint,
    validate_tactical_prioritization_result,
)
from src.recommender.tactical_recommendation_contract import (
    PUBLIC_FINGERPRINT_DOMAIN,
    build_public_tactical_recommendation,
    canonical_tactical_recommendation_json,
)
from src.recommender.tactical_recommendation_service import (
    InternalServiceError,
    InvalidRequestError,
    ProviderUnavailableError,
    RecommendationNotFoundError,
    TacticalRecommendationService,
    UpstreamContractViolationError,
)
from test_tactical_recommendation_service import (
    AS_OF_DATE,
    _RecordingProvider,
    _prioritization,
    _sentinel_scan,
    _valid_query,
)


_PLAYER = "PLAYER_SECRET_123"
_OPPONENT = "OPPONENT_SECRET_456"
_PLAYER_B = "Irene"
_OPPONENT_B = "Jules"
_PLAYER_C = "Karl"
_OPPONENT_C = "Mona"
_HEALTH_PATH = "/healthz"
_POST_PATH = "/api/v1/recommendations"
_MODULE_PATH = Path(p13.__file__).resolve()


# --------------------------------------------------------------------------
# Fixtures y helpers
# --------------------------------------------------------------------------


def _base_results() -> tuple:
    return (
        _prioritization(
            "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
        ),
        _prioritization(
            "absent",
            0,
            AS_OF_DATE,
            False,
            player=_PLAYER_B,
            opponent=_OPPONENT_B,
        ),
        _prioritization(
            "partial",
            0,
            AS_OF_DATE,
            True,
            player=_PLAYER_C,
            opponent=_OPPONENT_C,
        ),
        _prioritization(
            "available",
            1,
            AS_OF_DATE,
            False,
            player=_PLAYER_B,
            opponent=_PLAYER_C,
        ),
    )


def _identity_sentinels() -> tuple:
    return (
        _PLAYER,
        _OPPONENT,
        _PLAYER_B,
        _OPPONENT_B,
        _PLAYER_C,
        _OPPONENT_C,
    )


def _write_snapshot(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


@pytest.fixture(scope="module")
def base_snapshot() -> "p13.PersistedTacticalRecommendationSnapshot":
    results = _base_results()
    for result in results:
        validate_tactical_prioritization_result(result)
    return p13.build_persisted_tactical_recommendation_snapshot(results)


@pytest.fixture()
def snapshot_file(base_snapshot, tmp_path) -> Path:
    destination = tmp_path / "snapshot.json"
    p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, destination)
    return destination


@pytest.fixture()
def provider_path(snapshot_file, tmp_path) -> Path:
    target = tmp_path / "provider-snapshot.json"
    shutil.copyfile(snapshot_file, target)
    return target


# --------------------------------------------------------------------------
# A. Construccion
# --------------------------------------------------------------------------


def test_build_roundtrips_all_states_and_variants(tmp_path):
    cases = (
        ("available", 0, False),
        ("available", 0, True),
        ("available", 1, False),
        ("partial", 0, False),
        ("absent", 0, False),
    )
    for index, (state, variant, reversed_input) in enumerate(cases):
        result = _prioritization(
            state, variant, AS_OF_DATE, reversed_input,
            player=_PLAYER, opponent=_OPPONENT,
        )
        snapshot = p13.build_persisted_tactical_recommendation_snapshot([result])
        raw = p13.serialize_persisted_tactical_recommendation_snapshot(snapshot)
        loaded = p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(tmp_path, f"case-{index}.json", raw)
        )
        entry = loaded.entries[0]
        assert entry.key.player == _PLAYER
        assert entry.key.opponent == _OPPONENT
        assert entry.key.as_of_date == AS_OF_DATE
        assert entry.result == result
        assert entry.result is not result


def test_build_is_order_independent_and_canonically_sorted(base_snapshot):
    results = _base_results()
    shuffled = (results[2], results[0], results[3], results[1])
    first = p13.build_persisted_tactical_recommendation_snapshot(results)
    second = p13.build_persisted_tactical_recommendation_snapshot(shuffled)
    assert first.fingerprint == second.fingerprint
    assert (
        p13.serialize_persisted_tactical_recommendation_snapshot(first)
        == p13.serialize_persisted_tactical_recommendation_snapshot(second)
        == p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    )
    keys = [
        (entry.key.player, entry.key.opponent, entry.key.as_of_date.isoformat())
        for entry in second.entries
    ]
    assert keys == sorted(keys)


def test_build_rejects_exact_duplicate_keys():
    result = _prioritization(
        "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.build_persisted_tactical_recommendation_snapshot([result, result])


def test_build_rejects_conflicting_results_for_same_key():
    first = _prioritization(
        "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    second = _prioritization(
        "absent", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.build_persisted_tactical_recommendation_snapshot([first, second])


@pytest.mark.parametrize("bad", [None, 1, "x", b"x", {"a": 1}, 0.5])
def test_build_rejects_non_result_objects(bad):
    with pytest.raises(p13.SnapshotUpstreamInvalidError):
        p13.build_persisted_tactical_recommendation_snapshot([bad])


@pytest.mark.parametrize("bad", [None, 123, "x", b"x"])
def test_build_rejects_non_sequence(bad):
    with pytest.raises(TypeError):
        p13.build_persisted_tactical_recommendation_snapshot(bad)


def test_build_enforces_entry_cap(monkeypatch):
    monkeypatch.setattr(p13, "MAX_ENTRIES", 1)
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.build_persisted_tactical_recommendation_snapshot(list(_base_results()))


def test_build_consumes_at_most_cap_plus_one(monkeypatch):
    monkeypatch.setattr(p13, "MAX_ENTRIES", 2)
    consumed = 0

    def unbounded():
        nonlocal consumed
        while True:
            consumed += 1
            yield _base_results()[0]

    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.build_persisted_tactical_recommendation_snapshot(unbounded())
    assert consumed == 3


def test_build_discards_hostile_iterator_exception():
    def hostile():
        yield _base_results()[0]
        raise RuntimeError("PRIVATE_ITERATOR_SECRET")

    with pytest.raises(p13.SnapshotUpstreamInvalidError) as captured:
        p13.build_persisted_tactical_recommendation_snapshot(hostile())
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_build_empty_snapshot_is_valid(tmp_path):
    empty = p13.build_persisted_tactical_recommendation_snapshot([])
    assert empty.entry_count == 0
    assert empty.entries == ()
    assert empty.policies == (default_tactical_scoring_policy(),)
    raw = p13.serialize_persisted_tactical_recommendation_snapshot(empty)
    loaded = p13.load_persisted_tactical_recommendation_snapshot(
        _write_snapshot(tmp_path, "empty.json", raw)
    )
    assert loaded.entry_count == 0
    assert loaded.fingerprint == empty.fingerprint


def test_build_key_is_derived_from_result_identity(base_snapshot):
    for entry in base_snapshot.entries:
        summary = entry.result.matchup_query
        assert entry.key.player == summary.player
        assert entry.key.opponent == summary.opponent
        assert entry.key.as_of_date == date.fromisoformat(summary.as_of_date)


@pytest.mark.parametrize(
    ("player", "opponent", "as_of_date"),
    [
        (True, "B", AS_OF_DATE),
        (1, "B", AS_OF_DATE),
        ("A", False, AS_OF_DATE),
        ("A", "A", AS_OF_DATE),
        (" A", "B", AS_OF_DATE),
        ("A", "B", "2021-01-01"),
        ("A", "B", datetime(2021, 1, 1)),
    ],
)
def test_snapshot_key_rejects_coercions_and_impossible_identity(
    player, opponent, as_of_date
):
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.TacticalRecommendationSnapshotKey(player, opponent, as_of_date)


def test_snapshot_key_identifier_length_boundary():
    p13.TacticalRecommendationSnapshotKey(
        "A" * 64, "B" * 64, AS_OF_DATE
    )
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.TacticalRecommendationSnapshotKey("A" * 65, "B", AS_OF_DATE)


# --------------------------------------------------------------------------
# B. Serializacion
# --------------------------------------------------------------------------


def test_serialize_bytes_are_canonical_and_closed(base_snapshot):
    raw = p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    assert raw.endswith(b"}")
    assert not raw.endswith(b"\n")
    assert b": " not in raw and b", " not in raw
    assert b"NaN" not in raw and b"Infinity" not in raw
    raw.decode("utf-8")
    structure = json.loads(raw)
    assert list(structure) == sorted(structure)
    for entry in structure["entries"]:
        assert list(entry) == sorted(entry)
        assert list(entry["result"]) == sorted(entry["result"])
        for ranking in entry["result"]["rankings"]:
            assert list(ranking) == sorted(ranking)


def test_serialize_is_deterministic(base_snapshot):
    first = p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    second = p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    assert first == second


def test_serialize_rejects_wrong_types(base_snapshot):
    for bad in (42, "x", None, {"contract": "persisted_tactical_recommendation_snapshot"}):
        with pytest.raises(p13.SnapshotIntegrityError):
            p13.serialize_persisted_tactical_recommendation_snapshot(bad)


def test_snapshot_subclasses_cannot_be_constructed_or_serialized(base_snapshot):
    subclass_type = type(
        "_Subclass",
        (p13.PersistedTacticalRecommendationSnapshot,),
        {"__slots__": ()},
    )
    with pytest.raises((TypeError, p13.PersistedSnapshotError)):
        subclass_type(
            base_snapshot.contract_name,
            base_snapshot.schema_version,
            base_snapshot.format_version,
            0,
            base_snapshot.policies,
            base_snapshot.upstream_result_contract,
            base_snapshot.upstream_result_contract_version,
            (),
            base_snapshot.fingerprint,
        )


def test_snapshot_fingerprint_changes_with_content(base_snapshot):
    different = p13.build_persisted_tactical_recommendation_snapshot(
        [
            _prioritization(
                "partial", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
            )
        ]
    )
    assert different.fingerprint != base_snapshot.fingerprint


def test_entry_fingerprints_are_distinct_and_well_formed(base_snapshot):
    fingerprints = {entry.entry_fingerprint for entry in base_snapshot.entries}
    assert len(fingerprints) == len(base_snapshot.entries)
    for entry in base_snapshot.entries:
        assert re.fullmatch(r"[0-9A-F]{64}", entry.entry_fingerprint)
        assert re.fullmatch(r"[0-9A-F]{64}", entry.result_fingerprint)


# --------------------------------------------------------------------------
# C. Persistencia / carga
# --------------------------------------------------------------------------


def test_persist_then_load_is_byte_exact(base_snapshot, tmp_path):
    destination = tmp_path / "snap.json"
    p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, destination)
    raw = destination.read_bytes()
    assert raw == p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    loaded = p13.load_persisted_tactical_recommendation_snapshot(destination)
    assert p13.serialize_persisted_tactical_recommendation_snapshot(loaded) == raw
    assert (
        p13.verify_persisted_tactical_recommendation_snapshot(destination)
        == base_snapshot.fingerprint
    )
    assert [p.name for p in tmp_path.iterdir()] == ["snap.json"]


def test_persist_rejects_missing_parent_and_keeps_tree_clean(tmp_path):
    child = tmp_path / "seed.txt"
    child.write_bytes(b"a")
    destination = child / "nested" / "snap.json"
    empty = p13.build_persisted_tactical_recommendation_snapshot([])
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.persist_persisted_tactical_recommendation_snapshot(empty, destination)
    assert [p.name for p in tmp_path.iterdir()] == ["seed.txt"]


def test_persist_rejects_symlink_destination(base_snapshot, tmp_path):
    target = tmp_path / "real.json"
    target.write_bytes(b"pre-existing")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, link)
    assert target.read_bytes() == b"pre-existing"
    assert link.is_symlink()


@pytest.mark.parametrize(
    ("bad_snapshot", "bad_destination"),
    [(42, "x.json"), ("x", "y.json"), (None, None)],
)
def test_persist_rejects_wrong_types(bad_snapshot, bad_destination, tmp_path):
    with pytest.raises((p13.SnapshotIncompatibleError, TypeError)):
        p13.persist_persisted_tactical_recommendation_snapshot(
            bad_snapshot, tmp_path / (bad_destination or "x.json")
        )
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, 123)
    assert list(tmp_path.iterdir()) == []


def test_persist_overwrites_existing_destination_atomically(base_snapshot, tmp_path):
    destination = tmp_path / "snap.json"
    destination.write_bytes(b"stale-bytes")
    p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, destination)
    assert (
        destination.read_bytes()
        == p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    )
    leftovers = [
        p.name
        for p in tmp_path.iterdir()
        if p.name.startswith(".persisted-tactical-snapshot-")
    ]
    assert leftovers == []


def test_persist_failure_before_replace_preserves_destination(
    base_snapshot, tmp_path, monkeypatch
):
    destination = tmp_path / "snap.json"
    p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, destination)
    before = destination.read_bytes()

    def fail_replace(_source, _destination):
        raise PermissionError("PRIVATE_PATH_SENTINEL")

    monkeypatch.setattr(p13.os, "replace", fail_replace)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, destination)
    assert destination.read_bytes() == before
    leftovers = [
        p.name
        for p in tmp_path.iterdir()
        if p.name.startswith(".persisted-tactical-snapshot-")
    ]
    assert leftovers == []


def test_load_rejects_structurally_incompatible_documents(provider_path):
    raw = provider_path.read_bytes()

    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(provider_path.parent, "array.json", b"[]")
        )
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent,
                "missing-key.json",
                raw.replace(b'"contract"', b'"contractx"'),
            )
        )
    doc = json.loads(raw)
    doc["schema_version"] = "9.9.9"
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "bad-schema.json", json.dumps(doc).encode()
            )
        )
    doc = json.loads(raw)
    doc["contract"] = "something_else"
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "bad-contract.json", json.dumps(doc).encode()
            )
        )


def test_load_rejects_io_level_defects(provider_path, tmp_path, monkeypatch):
    with pytest.raises(p13.SnapshotNotFoundError):
        p13.load_persisted_tactical_recommendation_snapshot(
            tmp_path / "missing.json"
        )
    directory = tmp_path / "adirectory.json"
    directory.mkdir()
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.load_persisted_tactical_recommendation_snapshot(directory)
    link = tmp_path / "link.json"
    link.symlink_to(provider_path)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.load_persisted_tactical_recommendation_snapshot(link)
    empty = _write_snapshot(tmp_path, "empty.json", b"")
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.load_persisted_tactical_recommendation_snapshot(empty)
    monkeypatch.setattr(p13, "MAX_SNAPSHOT_BYTES", 10)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.load_persisted_tactical_recommendation_snapshot(provider_path)
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.load_persisted_tactical_recommendation_snapshot(123)


def test_load_rejects_json_level_corruption(provider_path):
    raw = provider_path.read_bytes()

    with pytest.raises(p13.SnapshotMalformedError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(provider_path.parent, "non-utf8.json", raw[:-1] + b"\xff\xfe")
        )
    duplicate = raw.replace(
        b'"schema_version":"1.0.0"',
        b'"schema_version":"1.0.0","schema_version":"1.0.0"',
    )
    with pytest.raises(p13.SnapshotMalformedError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(provider_path.parent, "duplicate-keys.json", duplicate)
        )
    nan = raw.replace(
        b'"abstained_candidate_count":0', b'"abstained_candidate_count":NaN'
    )
    with pytest.raises(p13.SnapshotMalformedError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(provider_path.parent, "nan.json", nan)
        )


def test_load_rejects_integrity_violations(provider_path):
    raw = provider_path.read_bytes()
    fingerprint = json.loads(raw)["fingerprint"].encode()

    flipped = raw.replace(
        b'"fingerprint":"' + fingerprint, b'"fingerprint":"0' + fingerprint[1:]
    )
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(provider_path.parent, "flipped-fp.json", flipped)
        )

    doc = json.loads(raw)
    doc["entry_count"] += 1
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "bad-count.json", json.dumps(doc).encode()
            )
        )

    doc = json.loads(raw)
    doc["entries"] = [doc["entries"][1], doc["entries"][0]] + doc["entries"][2:]
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "reordered.json", json.dumps(doc).encode()
            )
        )

    doc = json.loads(raw)
    doc["entries"] = doc["entries"] * 2
    doc["entry_count"] *= 2
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent,
                "duplicated-entries.json",
                json.dumps(doc).encode(),
            )
        )

    doc = json.loads(raw)
    doc["entries"][0]["player"] = "Mallory"
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent,
                "identity-mismatch.json",
                json.dumps(doc).encode(),
            )
        )

    doc = json.loads(raw)
    doc["entries"][0]["as_of_date"] = "2021-02-30"
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "bad-date.json", json.dumps(doc).encode()
            )
        )

    doc = json.loads(raw)
    doc["entries"][0]["player"] = "bad\nkey"
    with pytest.raises(p13.SnapshotMalformedError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "control-char-key.json", json.dumps(doc).encode()
            )
        )

    doc = json.loads(raw)
    doc["entries"][0]["player"] = _PLAYER.upper()
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent,
                "case-changed-key.json",
                json.dumps(doc).encode(),
            )
        )

    doc = json.loads(raw)
    doc["entries"][0]["result_fingerprint"] = "F" * 64
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent,
                "bad-result-fp.json",
                json.dumps(doc).encode(),
            )
        )


def test_load_rejects_upstream_invalid_results(provider_path):
    raw = provider_path.read_bytes()

    doc = json.loads(raw)
    doc["entries"][0]["result"]["rankings"][0]["pattern_id"] = "not_a_real_pattern"
    with pytest.raises(p13.SnapshotUpstreamInvalidError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "bad-pattern.json", json.dumps(doc).encode()
            )
        )

    doc = json.loads(raw)
    doc["entries"][0]["result"]["reason_codes"] = ["not_a_real_code"]
    with pytest.raises(p13.SnapshotUpstreamInvalidError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "bad-reason.json", json.dumps(doc).encode()
            )
        )

    doc = json.loads(raw)
    doc["entries"][0]["result"]["matchup_query"]["opponent"] = _OPPONENT.upper()
    with pytest.raises(p13.SnapshotUpstreamInvalidError):
        p13.load_persisted_tactical_recommendation_snapshot(
            _write_snapshot(
                provider_path.parent, "query-mismatch.json", json.dumps(doc).encode()
            )
        )


def test_roundtrip_preserves_p10_contract_sensitive_fields(base_snapshot, tmp_path):
    raw = p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    loaded = p13.load_persisted_tactical_recommendation_snapshot(
        _write_snapshot(tmp_path, "snap.json", raw)
    )
    by_key = {entry.key: entry for entry in loaded.entries}
    for entry in base_snapshot.entries:
        rebuild = by_key[entry.key].result
        validate_tactical_prioritization_result(rebuild)
        assert entry.result == rebuild
        assert tactical_prioritization_result_fingerprint(rebuild) == entry.result_fingerprint


# --------------------------------------------------------------------------
# D. Provider
# --------------------------------------------------------------------------


def test_provider_lookup_returns_stable_identity(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    query = _valid_query(_PLAYER, _OPPONENT, AS_OF_DATE)
    first = provider.fetch_tactical_prioritization(query)
    for _ in range(3):
        assert provider.fetch_tactical_prioritization(query) is first
    assert first == _prioritization(
        "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    service = TacticalRecommendationService(provider)
    assert service.recommend(query) == build_public_tactical_recommendation(first)


def test_provider_missing_key_is_404(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    with pytest.raises(RecommendationNotFoundError):
        provider.fetch_tactical_prioritization(
            _valid_query(_PLAYER_C, _OPPONENT, AS_OF_DATE)
        )
    service = TacticalRecommendationService(provider)
    with pytest.raises(RecommendationNotFoundError):
        service.recommend(_valid_query(_PLAYER_C, _OPPONENT, AS_OF_DATE))


def test_provider_does_not_fall_back_across_orientation_or_date(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    with pytest.raises(RecommendationNotFoundError):
        provider.fetch_tactical_prioritization(
            _valid_query(_OPPONENT, _PLAYER, AS_OF_DATE)
        )
    with pytest.raises(RecommendationNotFoundError):
        provider.fetch_tactical_prioritization(
            _valid_query(_PLAYER, _OPPONENT, date(2021, 1, 2))
        )


def test_provider_rejects_non_query_objects(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    with pytest.raises(InvalidRequestError):
        provider.fetch_tactical_prioritization(SimpleNamespace())

    class _BogusQuery:
        player_id = 1
        opponent_id = _OPPONENT
        as_of_date = AS_OF_DATE

    with pytest.raises(InvalidRequestError):
        provider.fetch_tactical_prioritization(_BogusQuery())


def test_provider_creation_maps_snapshot_failures(tmp_path, provider_path):
    with pytest.raises(ProviderUnavailableError):
        p13.create_persisted_tactical_recommendation_provider(
            tmp_path / "missing.json"
        )
    with pytest.raises(ProviderUnavailableError):
        p13.create_persisted_tactical_recommendation_provider(123)
    corrupted = _write_snapshot(
        tmp_path, "corrupted.json", provider_path.read_bytes()[:-1] + b"\xff"
    )
    with pytest.raises(UpstreamContractViolationError):
        p13.create_persisted_tactical_recommendation_provider(corrupted)
    ok = p13.create_persisted_tactical_recommendation_provider(provider_path)
    assert not hasattr(ok, "snapshot_path")
    assert str(provider_path) not in repr(ok)


def test_provider_is_frozen_and_slotted(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    assert not hasattr(provider, "__dict__")
    assert not hasattr(provider.snapshot, "__dict__")
    with pytest.raises((AttributeError, FrozenInstanceError)):
        provider.snapshot = None
    entry = provider.snapshot.entries[0]
    assert not hasattr(entry, "__dict__")
    assert not hasattr(entry.key, "__dict__")
    with pytest.raises((AttributeError, FrozenInstanceError)):
        entry.key = entry.key
    with pytest.raises((AttributeError, FrozenInstanceError)):
        entry.key.player = "x"
    with pytest.raises((AttributeError, FrozenInstanceError)):
        entry.result.state = "unavailable"


def test_provider_snapshot_index_is_frozen_view(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    index = provider.snapshot.index
    assert len(index) == provider.snapshot.entry_count
    with pytest.raises(TypeError):
        index[provider.snapshot.entries[0].key] = None


def test_provider_ignores_disk_changes_after_creation(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    query = _valid_query(_PLAYER, _OPPONENT, AS_OF_DATE)
    before = provider.fetch_tactical_prioritization(query)
    provider_path.write_bytes(b"{}")
    assert provider.fetch_tactical_prioritization(query) is before
    provider_path.unlink()
    assert provider.fetch_tactical_prioritization(query) is before


def test_provider_is_concurrency_safe(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    queries = (
        (_PLAYER, _OPPONENT, "available"),
        (_PLAYER_B, _OPPONENT_B, "not_available"),
        (_PLAYER_B, _PLAYER_C, "available"),
        (_PLAYER_C, _OPPONENT_C, "partially_available"),
    )

    def one(index: int) -> bool:
        player, opponent, state = queries[index % len(queries)]
        result = provider.fetch_tactical_prioritization(
            _valid_query(player, opponent, AS_OF_DATE)
        )
        assert result.matchup_query.player == player
        assert result.matchup_query.opponent == opponent
        assert result.state.value == state
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(1 for ok in pool.map(one, range(300)) if ok) == 300


# --------------------------------------------------------------------------
# E. Integracion P12
# --------------------------------------------------------------------------


def _api_client_from_snapshot(snapshot, tmp_path) -> TestClient:
    destination = tmp_path / "api-snapshot.json"
    p13.persist_persisted_tactical_recommendation_snapshot(snapshot, destination)
    provider = p13.create_persisted_tactical_recommendation_provider(destination)
    return TestClient(create_app(provider))


def test_api_returns_exact_p11_bytes_and_strong_etag(base_snapshot, tmp_path):
    client = _api_client_from_snapshot(base_snapshot, tmp_path)
    response = client.post(
        _POST_PATH,
        json={
            "player_id": _PLAYER,
            "opponent_id": _OPPONENT,
            "as_of_date": "2021-01-01",
        },
    )
    assert response.status_code == 200
    body = response.content
    payload = json.loads(body)
    published = payload.pop("fingerprint")
    core = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    recomputed = sha256(PUBLIC_FINGERPRINT_DOMAIN + core).hexdigest().upper()
    assert published == recomputed
    assert response.headers["etag"] == f'"{recomputed}"'
    source = _prioritization(
        "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    assert body == canonical_tactical_recommendation_json(
        build_public_tactical_recommendation(source)
    )
    _sentinel_scan(body, *_identity_sentinels())
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_api_public_bytes_do_not_depend_on_private_identity():
    one = _prioritization(
        "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    two = _prioritization(
        "available", 0, AS_OF_DATE, False, player=_PLAYER_B, opponent=_OPPONENT_B
    )
    assert (
        canonical_tactical_recommendation_json(
            build_public_tactical_recommendation(one)
        )
        == canonical_tactical_recommendation_json(
            build_public_tactical_recommendation(two)
        )
    )


def test_api_404_envelope_leaks_no_identity(base_snapshot, tmp_path):
    client = _api_client_from_snapshot(base_snapshot, tmp_path)
    response = client.post(
        _POST_PATH,
        json={
            "player_id": "Ghost",
            "opponent_id": "Phantom",
            "as_of_date": "2021-01-01",
        },
    )
    assert response.status_code == 404
    assert "error" in json.loads(response.text)
    for sentinel in _identity_sentinels() + ("Ghost", "Phantom"):
        assert sentinel not in response.text


def test_api_invalid_body_never_reaches_provider(tmp_path):
    snapshot = p13.build_persisted_tactical_recommendation_snapshot(
        [
            _prioritization(
                "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
            )
        ]
    )
    destination = tmp_path / "counting.json"
    p13.persist_persisted_tactical_recommendation_snapshot(snapshot, destination)
    inner = p13.create_persisted_tactical_recommendation_provider(destination)
    provider = _CountingProvider(inner)
    client = TestClient(create_app(provider))
    for bad_body in (
        {"player_id": "", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
        {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": "2021-02-30"},
        {"player_id": _PLAYER, "opponent_id": _OPPONENT},
    ):
        assert client.post(_POST_PATH, json=bad_body).status_code == 422
    assert provider.calls == 0
    assert client.get(_HEALTH_PATH).status_code == 200
    assert provider.calls == 0
    found = client.post(
        _POST_PATH,
        json={
            "player_id": _PLAYER,
            "opponent_id": _OPPONENT,
            "as_of_date": "2021-01-01",
        },
    )
    assert found.status_code == 200
    assert provider.calls == 1


class _CountingProvider:
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def fetch_tactical_prioritization(self, query):
        self.calls += 1
        return self._inner.fetch_tactical_prioritization(query)


def test_provider_failure_maps_to_p12_envelope():
    client = TestClient(
        create_app(_RecordingProvider(None, raise_error=ProviderUnavailableError()))
    )
    response = client.post(
        _POST_PATH,
        json={
            "player_id": _PLAYER,
            "opponent_id": _OPPONENT,
            "as_of_date": "2021-01-01",
        },
    )
    assert response.status_code == 503
    assert "error" in json.loads(response.text)
    _sentinel_scan(response.text, _PLAYER, _OPPONENT)


def test_public_surfaces_do_not_expose_private_sentinels(base_snapshot, tmp_path):
    client = _api_client_from_snapshot(base_snapshot, tmp_path)
    sentinels = (
        "DATE_SECRET_2099_12_31",
        r"C:\PRIVATE\snapshot.json",
        "/home/private/snapshot.json",
        r"\\server\private\snapshot.json",
        "file:///private/snapshot.json",
        "SEQUENCE_SECRET_6f27",
        "MATCH_SECRET_123",
        "POINT_SECRET_456",
        "OUTCOME_SECRET_WIN",
        "EXCEPTION_SECRET_TRACE",
    )
    error = client.post(
        _POST_PATH,
        json={"player_id": "Ghost", "opponent_id": "Phantom", "as_of_date": "2021-01-01"},
    )
    health = client.get(_HEALTH_PATH)
    openapi = client.get("/openapi.json")
    for surface in (
        error.content,
        health.content,
        openapi.content,
        repr(RecommendationNotFoundError()).encode(),
        error.headers.__repr__().encode(),
        health.headers.__repr__().encode(),
    ):
        for sentinel in sentinels + (_PLAYER, _OPPONENT):
            assert sentinel.encode() not in surface
    assert set(client.app.openapi()["paths"]) == {_HEALTH_PATH, _POST_PATH}


# --------------------------------------------------------------------------
# F. Arquitectura
# --------------------------------------------------------------------------


_FORBIDDEN_IMPORTS = frozenset(
    {
        "pandas",
        "numpy",
        "polars",
        "sklearn",
        "scipy",
        "pyarrow",
        "torch",
        "tensorflow",
        "fastapi",
        "starlette",
        "httpx",
        "requests",
        "urllib",
        "http",
        "socket",
        "ssl",
        "smtplib",
        "sqlite3",
        "psycopg2",
        "sqlalchemy",
        "subprocess",
        "shutil",
        "pickle",
        "pickletools",
        "shelve",
        "marshal",
        "copyreg",
        "asyncio",
    }
)


def _module_tree() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def test_module_has_no_forbidden_imports():
    imported = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported & _FORBIDDEN_IMPORTS == set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "src.analysis.tactical_recommender_pipeline"


def test_module_has_no_import_time_side_effects():
    tree = _module_tree()
    allowed = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.ClassDef,
        ast.Assign,
        ast.AnnAssign,
    )
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ) and isinstance(statement.value.value, str):
            continue
        assert isinstance(statement, allowed), ast.dump(statement)
    for node in _top_level_calls(tree):
        if isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "eval", "exec", "__import__"}
        elif isinstance(node.func, ast.Attribute):
            assert node.func.attr not in _IO_BANNED_NAMES


def _top_level_calls(tree: ast.Module):
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                yield node


def test_module_forbids_asserts_eval_and_dynamic_imports():
    for node in ast.walk(_module_tree()):
        assert not isinstance(node, ast.Assert)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "__import__"}


_IO_BANNED_NAMES = {
    "open",
    "read_bytes",
    "read_text",
    "write_bytes",
    "write_text",
    "exists",
    "stat",
    "iterdir",
    "unlink",
    "rename",
    "replace",
    "listdir",
}


def test_provider_class_performs_no_io():
    tree = _module_tree()
    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "PersistedTacticalRecommendationProvider"
    )
    for node in ast.walk(class_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in _IO_BANNED_NAMES
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in _IO_BANNED_NAMES
    names = {node.id for node in ast.walk(class_node) if isinstance(node, ast.Name)}
    assert "os" not in names
    assert "Path" not in names
    assert "tempfile" not in names


def test_module_has_no_hardcoded_paths():
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not re.search(r"/Users/|/home/|/etc/|/var/|/tmp/", node.value)


def test_error_mapping_is_closed():
    mapping = p13._SNAPSHOT_ERROR_TO_SERVICE_ERROR
    assert frozenset(mapping) == frozenset(
        {
            p13.SnapshotNotFoundError,
            p13.SnapshotUnavailableError,
            p13.SnapshotIncompatibleError,
            p13.SnapshotMalformedError,
            p13.SnapshotIntegrityError,
            p13.SnapshotUpstreamInvalidError,
        }
    )
    import src.recommender.tactical_recommendation_service as service_module

    for service_error in mapping.values():
        assert issubclass(service_error, service_module.TacticalRecommendationServiceError)


def test_public_exports_resolve():
    for name in p13.__all__:
        assert hasattr(p13, name)


def test_snapshot_constants_are_p13_owned():
    assert p13.SNAPSHOT_CONTRACT_NAME == "persisted_tactical_recommendation_snapshot"
    assert p13.SNAPSHOT_SCHEMA_VERSION == "1.0.0"
    assert p13.SNAPSHOT_FORMAT_VERSION == 1
    assert p13.SNAPSHOT_UPSTREAM_RESULT_CONTRACT == "tactical_prioritization_result"
    assert PRIORITIZATION_CONTRACT_VERSION
    assert p13.SNAPSHOT_ENTRY_DOMAIN != p13.SNAPSHOT_SNAPSHOT_DOMAIN


# --------------------------------------------------------------------------
# G. Endurecimiento adversarial independiente
# --------------------------------------------------------------------------


def _independent_json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _independently_resign_document(document: dict) -> bytes:
    for entry in document["entries"]:
        payload = {
            "as_of_date": entry["as_of_date"],
            "opponent": entry["opponent"],
            "player": entry["player"],
            "result_fingerprint": entry["result_fingerprint"],
        }
        entry["entry_fingerprint"] = sha256(
            b"tennis-persisted-tactical-recommendation-entry\x00"
            + _independent_json_bytes(payload)
        ).hexdigest().upper()
    unsigned = {key: value for key, value in document.items() if key != "fingerprint"}
    document["fingerprint"] = sha256(
        b"tennis-persisted-tactical-recommendation-snapshot\x00"
        + _independent_json_bytes(unsigned)
    ).hexdigest().upper()
    return _independent_json_bytes(document)


def test_fingerprints_are_independently_recomputed_and_domain_separated(base_snapshot):
    document = json.loads(
        p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    )
    expected_domains = {
        b"tennis-persisted-tactical-recommendation-entry\x00",
        b"tennis-persisted-tactical-recommendation-snapshot\x00",
    }
    assert {p13.SNAPSHOT_ENTRY_DOMAIN, p13.SNAPSHOT_SNAPSHOT_DOMAIN} == expected_domains
    upstream_domains = {
        b"tennis-tactical-prioritization-result\x00",
        b"tennis-public-tactical-recommendation\x00",
        b"tennis-tactical-recommender-pipeline\x00",
        b"tennis-tactical-feature-vector\x00",
        b"tennis-tactical-player-profile\x00",
        b"tennis-tactical-matchup-evidence\x00",
    }
    assert expected_domains.isdisjoint(upstream_domains)
    for entry in document["entries"]:
        payload = {
            "as_of_date": entry["as_of_date"],
            "opponent": entry["opponent"],
            "player": entry["player"],
            "result_fingerprint": entry["result_fingerprint"],
        }
        expected = sha256(
            b"tennis-persisted-tactical-recommendation-entry\x00"
            + _independent_json_bytes(payload)
        ).hexdigest().upper()
        assert entry["entry_fingerprint"] == expected
    published = document.pop("fingerprint")
    expected = sha256(
        b"tennis-persisted-tactical-recommendation-snapshot\x00"
        + _independent_json_bytes(document)
    ).hexdigest().upper()
    assert published == expected


def test_resigned_semantically_invalid_payload_is_rejected(provider_path):
    document = json.loads(provider_path.read_bytes())
    document["entries"][0]["result"]["reason_codes"] = ["forged_reason"]
    forged = provider_path.parent / "forged.json"
    forged.write_bytes(_independently_resign_document(document))
    with pytest.raises(p13.SnapshotUpstreamInvalidError):
        p13.load_persisted_tactical_recommendation_snapshot(forged)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xef\xbb\xbf{}",
        b"{} trailing",
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
    ],
)
def test_load_rejects_hostile_json_envelopes(tmp_path, payload):
    path = _write_snapshot(tmp_path, "hostile.json", payload)
    with pytest.raises(p13.SnapshotMalformedError):
        p13.load_persisted_tactical_recommendation_snapshot(path)


def test_json_resource_limits_have_exact_boundaries(monkeypatch):
    monkeypatch.setattr(p13, "MAX_JSON_STRING_LENGTH", 3)
    monkeypatch.setattr(p13, "MAX_JSON_ARRAY_ITEMS", 2)
    monkeypatch.setattr(p13, "MAX_JSON_OBJECT_KEYS", 2)
    monkeypatch.setattr(p13, "MAX_JSON_INTEGER_ABS", 7)
    monkeypatch.setattr(p13, "MAX_JSON_DEPTH", 2)
    for accepted in ("abc", [1, 2], {"a": 1, "b": 2}, 7, -7, [[0]]):
        p13._check_json_tree(accepted)
    for rejected in ("abcd", [1, 2, 3], {"a": 1, "b": 2, "c": 3}, 8, -8, [[[0]]]):
        with pytest.raises(p13.SnapshotMalformedError):
            p13._check_json_tree(rejected)


def test_serialization_enforces_byte_limit(base_snapshot, monkeypatch):
    raw = p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    monkeypatch.setattr(p13, "MAX_SNAPSHOT_BYTES", len(raw) - 1)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)


@pytest.mark.parametrize(
    "unsafe",
    [
        "file:///private/snapshot.json",
        "vscode://file/private/snapshot.json",
        "../snapshot.json",
        "safe/../snapshot.json",
        "~/snapshot.json",
        r"\\server\share\snapshot.json",
        "//server/share/snapshot.json",
        "snapshot.json\x00tail",
    ],
)
def test_path_contract_rejects_uri_unc_home_traversal_and_nul(unsafe):
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.load_persisted_tactical_recommendation_snapshot(unsafe)


def test_custom_pathlike_is_not_executed():
    class HostilePath:
        called = False

        def __fspath__(self):
            self.called = True
            raise AssertionError("must not execute custom PathLike")

    value = HostilePath()
    with pytest.raises(p13.SnapshotIncompatibleError):
        p13.load_persisted_tactical_recommendation_snapshot(value)
    assert value.called is False


def test_symlinked_parent_is_rejected_for_load_and_persist(base_snapshot, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this platform")
    destination = real / "snapshot.json"
    p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, destination)
    alias = linked / "snapshot.json"
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.load_persisted_tactical_recommendation_snapshot(alias)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, linked / "new.json")


def test_post_replace_failure_does_not_claim_rollback(base_snapshot, tmp_path, monkeypatch):
    destination = tmp_path / "snapshot.json"
    destination.write_bytes(b"old")
    real_replace = p13.os.replace

    def replace_then_fail(source, target):
        real_replace(source, target)
        raise OSError("failure after commit point")

    monkeypatch.setattr(p13.os, "replace", replace_then_fail)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.persist_persisted_tactical_recommendation_snapshot(base_snapshot, destination)
    assert destination.read_bytes() == p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)


def test_provider_direct_construction_is_validated(base_snapshot):
    with pytest.raises(p13.SnapshotIntegrityError):
        p13.PersistedTacticalRecommendationProvider(snapshot=object())
    provider = p13.PersistedTacticalRecommendationProvider(snapshot=base_snapshot)
    assert provider.snapshot is base_snapshot


def test_provider_revalidates_exact_query_object(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    query = _valid_query(_PLAYER, _OPPONENT, AS_OF_DATE)
    object.__setattr__(query, "player_id", 1)
    with pytest.raises(InvalidRequestError) as captured:
        provider.fetch_tactical_prioritization(query)
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_public_error_boundary_discards_sensitive_exception_chain(monkeypatch):
    class ManipulatedSnapshotError(p13.SnapshotUnavailableError):
        pass

    def fail(_path):
        try:
            raise OSError("C:\\PRIVATE\\secret /home/private file:///secret")
        except OSError as original:
            raise ManipulatedSnapshotError() from original

    monkeypatch.setattr(p13, "load_persisted_tactical_recommendation_snapshot", fail)
    with pytest.raises(InternalServiceError) as captured:
        p13.create_persisted_tactical_recommendation_provider("snapshot.json")
    error = captured.value
    assert error.args == ()
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "secret" not in repr(error).casefold()


def test_concurrent_lookups_return_exact_cached_objects(provider_path):
    provider = p13.create_persisted_tactical_recommendation_provider(provider_path)
    queries = tuple(
        _valid_query(entry.key.player, entry.key.opponent, entry.key.as_of_date)
        for entry in provider.snapshot.entries
    )
    expected = {
        (query.player_id, query.opponent_id, query.as_of_date):
        provider.fetch_tactical_prioritization(query)
        for query in queries
    }

    def fetch(index):
        query = queries[index % len(queries)]
        result = provider.fetch_tactical_prioritization(query)
        return result is expected[(query.player_id, query.opponent_id, query.as_of_date)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        assert all(pool.map(fetch, range(600)))


def test_two_persisted_identities_produce_identical_public_wire_bytes(tmp_path):
    results = (
        _prioritization("available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT),
        _prioritization("available", 0, AS_OF_DATE, False, player=_PLAYER_B, opponent=_OPPONENT_B),
    )
    client = _api_client_from_snapshot(
        p13.build_persisted_tactical_recommendation_snapshot(results), tmp_path
    )
    payloads = []
    for player, opponent in ((_PLAYER, _OPPONENT), (_PLAYER_B, _OPPONENT_B)):
        response = client.post(
            _POST_PATH,
            json={"player_id": player, "opponent_id": opponent, "as_of_date": AS_OF_DATE.isoformat()},
        )
        assert response.status_code == 200
        payloads.append(response.content)
    assert payloads[0] == payloads[1]


def test_provider_lookup_ast_contains_no_serialization_or_fingerprint_calls():
    tree = _module_tree()
    provider = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "PersistedTacticalRecommendationProvider"
    )
    forbidden = {"json", "dumps", "loads", "sha256", "serialize", "fingerprint", "load", "persist"}
    called = set()
    for node in ast.walk(provider):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert called.isdisjoint(forbidden)


# --------------------------------------------------------------------------
# H. Capacidad (P16): 256 MiB para el universo de 3.610 entradas
# --------------------------------------------------------------------------


def test_limite_es_256_mib_exacto_con_metadata_de_capacidad():
    assert p13.MAX_SNAPSHOT_BYTES == 256 * 1024 * 1024 == 268_435_456
    assert p13.CAPACITY_DESIGN_UNIVERSE_ENTRIES == 3_610
    assert p13.CAPACITY_DESIGN_SYNTHETIC_TARGETS_MEASURED == 352
    assert type(p13.CAPACITY_DESIGN_SYNTHETIC_TARGETS_MEASURED) is int
    representative = p13.CAPACITY_DESIGN_REPRESENTATIVE_SNAPSHOT_BYTES
    free = p13.MAX_SNAPSHOT_BYTES - representative
    assert representative == 196_817_992
    assert representative / (1024 * 1024) == pytest.approx(187.7, abs=0.01)
    assert representative / p13.MAX_SNAPSHOT_BYTES == pytest.approx(
        0.733, abs=0.001
    )
    assert free == 71_617_464
    assert free / p13.MAX_SNAPSHOT_BYTES == pytest.approx(0.267, abs=0.001)
    assert p13.MAX_SNAPSHOT_BYTES / representative - 1 == pytest.approx(
        0.364, abs=0.001
    )
    # El snapshot representativo cabe con margen minimo del 20 %.
    assert (
        representative * 5 <= p13.MAX_SNAPSHOT_BYTES * 4
    )
    # Entrada tipica x universo ~= snapshot representativo (envoltorio
    # global < 0,1 %): la estimacion independiente se autoconsiste.
    estimated = p13.CAPACITY_DESIGN_TYPICAL_ENTRY_BYTES * 3_610
    delta = abs(estimated - p13.CAPACITY_DESIGN_REPRESENTATIVE_SNAPSHOT_BYTES)
    assert delta * 100 <= p13.CAPACITY_DESIGN_REPRESENTATIVE_SNAPSHOT_BYTES
    # Techo adversarial de entrada x universo = techo adversarial exacto.
    assert (
        p13.CAPACITY_DESIGN_MAX_ENTRY_BYTES
        * p13.CAPACITY_DESIGN_UNIVERSE_ENTRIES
        == p13.CAPACITY_DESIGN_ADVERSARIAL_SNAPSHOT_BYTES
    )
    # Intencion de diseno: el limite cubre lo representativo con margen
    # y es deliberadamente inferior al techo adversarial (defensa final
    # contra ficheros hostiles).
    assert (
        p13.CAPACITY_DESIGN_REPRESENTATIVE_SNAPSHOT_BYTES * 5
        <= p13.MAX_SNAPSHOT_BYTES * 4
    )
    assert p13.CAPACITY_DESIGN_ADVERSARIAL_SNAPSHOT_BYTES > p13.MAX_SNAPSHOT_BYTES
    # Pico operativo estimado acotado (serializacion/carga/indice).
    assert 1_000_000_000 < p13.CAPACITY_DESIGN_MEMORY_PEAK_BYTES < 4_000_000_000
    assert "privados offline" in p13.SNAPSHOT_PURPOSE


def test_limite_exacto_aceptado_y_limite_menos_uno_rechazado(
    base_snapshot, provider_path, tmp_path, monkeypatch
):
    raw = p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    exact = len(raw)
    # Limite exacto: aceptado en serializacion y en carga.
    monkeypatch.setattr(p13, "MAX_SNAPSHOT_BYTES", exact)
    assert (
        len(p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot))
        == exact
    )
    loaded = p13.load_persisted_tactical_recommendation_snapshot(provider_path)
    assert loaded.fingerprint == base_snapshot.fingerprint
    # Limite - 1: rechazado en lstat (antes de abrir) y en serializacion;
    # el error de cierre es de disponibilidad, no de parseo.
    monkeypatch.setattr(p13, "MAX_SNAPSHOT_BYTES", exact - 1)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.serialize_persisted_tactical_recommendation_snapshot(base_snapshot)
    shrunken = _write_snapshot(tmp_path, "limite-menos-uno.json", raw)
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.load_persisted_tactical_recommendation_snapshot(shrunken)


def test_archivo_que_crece_entre_lstat_y_lectura_se_rechaza(
    provider_path, monkeypatch
):
    raw = provider_path.read_bytes()
    limit = len(raw) - 64
    monkeypatch.setattr(p13, "MAX_SNAPSHOT_BYTES", limit)

    class _ForgedLstatPath(p13.Path):
        """Simula crecimiento: lstat reporta tamano menor al real."""

        def lstat(self):
            real = tuple(super().lstat())
            # st_size esta en el indice 6 de os.stat_result.
            return os.stat_result(real[:6] + (limit,) + real[7:])

    monkeypatch.setattr(p13, "Path", _ForgedLstatPath)
    # _snapshot_path solo acepta str o Path; se pasa str para que la
    # reconstruccion final use la clase parcheada (lstat falsificado).
    # lstat pasa (tamano reportado == limite), pero la lectura acotada
    # (limite + 1) y la comprobacion final rechazan el tamano real.
    with pytest.raises(p13.SnapshotUnavailableError):
        p13.load_persisted_tactical_recommendation_snapshot(str(provider_path))


def _entry_canonical_bytes(entry) -> int:
    return len(
        json.dumps(
            p13._entry_payload(entry),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def test_capacidad_material_3610_entradas_con_margen_y_lookup_exacto(
    tmp_path
):
    """Prueba material acotada con 24 entradas exclusivamente sinteticas.

    La estimacion contractual se obtuvo con 352 objetivos sinteticos;
    esta regresion material usa una muestra de 24 para mantener un coste
    razonable. Mide bytes/entrada sobre resultados sinteticos validos
    (available, partial, absent). Comprueba: (a) el snapshot material completo
    serializa dentro del limite; (b) la base de diseno representativa
    (auditoria P16 del pipeline P10: 196,8 MB) cubre el limite con
    margen >= 20 % y queda bajo umbral de 204,8 MiB; (c) canaria de
    esquema: ninguna entrada supera el techo medido de entrada
    saturada (26 candidatos, catalogo cerrado). El techo adversarial
    del snapshot completo (3610 entradas saturadas, ~379,1 MiB)
    supera deliberadamente el limite: 256 MiB es la defensa final
    contra ficheros hostiles, y si un snapshot legitimo llegara a
    superar el limite, la construccion P14 falla con razon cerrada.
    Verifica ademas que el provider conserva el lookup exacto.
    """
    states = ("available", "partial", "absent")
    results = []
    for index in range(24):
        player = f"CAPPlayer{index:03d}"
        opponent = f"CAPRival{index:03d}"
        results.append(
            _prioritization(
                states[index % 3],
                index % 8,
                AS_OF_DATE,
                False,
                player=player,
                opponent=opponent,
            )
        )
    snapshot = p13.build_persisted_tactical_recommendation_snapshot(results)
    raw = p13.serialize_persisted_tactical_recommendation_snapshot(snapshot)
    per_entry = [_entry_canonical_bytes(entry) for entry in snapshot.entries]
    universe = p13.CAPACITY_DESIGN_UNIVERSE_ENTRIES
    assert snapshot.entry_count == 24
    # (a) Snapshot material completo dentro del limite.
    assert len(raw) < p13.MAX_SNAPSHOT_BYTES
    # (b) Base de diseno representativa: margen >= 20 % y umbral 204,8 MiB.
    representative = p13.CAPACITY_DESIGN_REPRESENTATIVE_SNAPSHOT_BYTES
    assert representative * 5 <= p13.MAX_SNAPSHOT_BYTES * 4
    assert representative <= 204_800_000
    # Coherencia de la proyeccion: tipica x universo ~= representativa.
    estimated = p13.CAPACITY_DESIGN_TYPICAL_ENTRY_BYTES * universe
    delta = abs(estimated - representative)
    assert delta * 100 <= representative
    # (c) Canaria de esquema: la peor entrada medida no supera el techo
    #     saturado documentado (falla si el esquema crece silenciosamente).
    assert max(per_entry) <= p13.CAPACITY_DESIGN_MAX_ENTRY_BYTES
    # Techo adversarial documentado (defensa, no objetivo de capacidad).
    assert p13.CAPACITY_DESIGN_MAX_ENTRY_BYTES * universe > p13.MAX_SNAPSHOT_BYTES
    # Provider: lookup exacto conservado sobre el snapshot medido.
    provider = p13.PersistedTacticalRecommendationProvider(snapshot=snapshot)
    hit = provider.fetch_tactical_prioritization(
        _valid_query("CAPPlayer001", "CAPRival001", AS_OF_DATE)
    )
    assert (
        hit
        is snapshot.index[
            p13.TacticalRecommendationSnapshotKey(
                "CAPPlayer001", "CAPRival001", AS_OF_DATE
            )
        ].result
    )
    with pytest.raises(RecommendationNotFoundError):
        provider.fetch_tactical_prioritization(
            _valid_query("CAPAusente001", "CAPAusente002", AS_OF_DATE)
        )
