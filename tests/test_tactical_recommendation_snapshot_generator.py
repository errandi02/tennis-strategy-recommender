"""Pruebas adversariales P14: generador offline puro results->snapshot P13.

Bateria 100% sintetica sobre el generador P14 (auditoria contractual
contra el codigo P10, construccion, politicas, temporal, resultados,
serializacion canonica, persistencia delegada a P13, rendimiento,
arquitectura por AST y privacidad recursiva). No ejecuta P10, no lee
Parquet/CSV, no accede a ``data/`` ni a ``reports/``, no usa red, no
reconstruye el snapshot real y no toca el test sellado 2024-2026 (la
frontera se verifica con fechas sinteticas y constante monkeypatcheada).
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import FrozenInstanceError
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

import src.analysis.tactical_recommendation_snapshot_generator as p14
import src.analysis.tactical_recommender_pipeline as p10_pipeline
import src.recommender.persisted_tactical_recommendation_provider as p13
import test_tactical_recommendation_service as svc
from src.recommender.tactical_feature_encoder import TacticalEncodingPolicy
from src.recommender.tactical_prioritization import (
    default_tactical_scoring_policy,
    tactical_prioritization_result_fingerprint,
)

AS_OF = svc.AS_OF_DATE
SENTINEL_PLAYER = "ZxyPlayer"
SENTINEL_OPPONENT = "ZxyOpponent"
SENTINELS = (SENTINEL_PLAYER, SENTINEL_OPPONENT)


def _r(
    state: str,
    variant: int,
    player: str,
    opponent: str,
    as_of: date = AS_OF,
    reversed_input: bool = False,
):
    return svc._prioritization(
        state, variant, as_of, reversed_input, player=player, opponent=opponent
    )


def _four():
    return (
        _r("available", 0, SENTINEL_PLAYER, "Bob"),
        _r("absent", 0, "Bob", SENTINEL_PLAYER),
        _r("partial", 0, "Irene", "Jules", reversed_input=True),
        _r("available", 1, "Jules", "Irene"),
    )


def _sentinel_pair(as_of: date = AS_OF):
    return (
        _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT, as_of),
        _r("absent", 0, SENTINEL_OPPONENT, SENTINEL_PLAYER, as_of),
    )


def _clone_result(src, **overrides):
    clone = object.__new__(type(src))
    for name, value in vars(src).items():
        object.__setattr__(clone, name, overrides.get(name, value))
    return clone


def _clone_summary(src, **overrides):
    clone = object.__new__(type(src))
    for name, value in vars(src).items():
        object.__setattr__(clone, name, overrides.get(name, value))
    return clone


def _mini_p10_policy(*, required_entries: int, required_pairs: int, folds):
    policy = object.__new__(p14.SnapshotGenerationPolicy)
    values = dict(
        mode=p14.SNAPSHOT_GENERATION_MODE_P10_OFFLINE,
        scoring_policy=default_tactical_scoring_policy(),
        expected_encoder_policy=TacticalEncodingPolicy.COMPONENT_ONLY,
        expected_scope_strategy="global_only",
        expected_requested_patterns=("P02", "P04", "P05", "P06"),
        expected_serve_numbers=(1, 2),
        expected_minimum_labeled_attempts=50,
        expected_minimum_matches=5,
        expected_window_days=None,
        expected_top_k=3,
        required_entries=required_entries,
        required_reciprocal_pairs=required_pairs,
        expected_fold_orientations=folds,
    )
    for name, value in values.items():
        object.__setattr__(policy, name, value)
    return policy


def _p10_records(results, match_ids=None):
    values = tuple(results)
    if match_ids is None:
        match_ids = tuple(f"MATCH_{index // 2}" for index in range(len(values)))
    return tuple(
        p14.P10OfflineSnapshotRecord(
            target_match_id=match_ids[index],
            fold=f"validation_{result.matchup_query.as_of_date[:4]}",
            orientation=(
                "player_1_vs_player_2" if index % 2 == 0
                else "player_2_vs_player_1"
            ),
            prioritization=result,
        )
        for index, result in enumerate(values)
    )


class _CountingIterator:
    def __init__(self, items):
        self._inner = iter(items)
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        return next(self._inner)


def _private_scan(obj, sentinels=SENTINELS, _depth: int = 0):
    hits = []

    def walk(value, depth):
        if depth > 8:
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (str, bytes)):
            text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
            for sentinel in sentinels:
                if sentinel in text:
                    hits.append(value if isinstance(value, str) else value[:80])
        elif isinstance(value, BaseException):
            walk(str(value), depth + 1)
            for attr in ("stage", "reason_code", "__cause__", "__context__"):
                walk(getattr(value, attr, None), depth + 1)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(key, depth + 1)
                walk(item, depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                walk(item, depth + 1)
        elif hasattr(value, "__dict__"):
            for key, item in vars(value).items():
                walk(key, depth + 1)
                walk(item, depth + 1)

    walk(obj, _depth)
    return hits


# --------------------------------------------------------------------------
# A. Auditoria contractual contra el codigo P10 (solo lectura de API/fuente)
# --------------------------------------------------------------------------


def test_a1_pipeline_exposes_target_results_surface():
    pipeline_fields = {field.name: field for field in dataclasses.fields(p10_pipeline.TacticalPipelineResult)}
    assert "target_results" in pipeline_fields
    target_fields = {field.name for field in dataclasses.fields(p10_pipeline.TacticalTargetResult)}
    assert {"target", "evidence", "prioritization", "state"} <= target_fields


def test_a2_pipeline_persistence_boundary_is_aggregate_safe_summary():
    assert callable(getattr(p10_pipeline, "_safe_summary", None))


def test_a3_closed_generation_stages_and_reason_codes():
    assert p14.GENERATION_STAGES == frozenset(
        {"input", "upstream", "policy", "temporal", "key", "limit", "universe", "build"}
    )
    assert p14.GENERATION_REASON_CODES == frozenset(
        {
            "empty_input",
            "element_not_tactical_prioritization_result",
            "element_not_p10_offline_snapshot_record",
            "upstream_result_invalid",
            "scoring_policy_mismatch",
            "matchup_query_policy_mismatch",
            "ranking_policy_mismatch",
            "as_of_date_not_civil",
            "as_of_date_in_sealed_test_range",
            "player_equals_opponent",
            "duplicate_key_exact",
            "duplicate_key_conflicting",
            "entry_limit_exceeded",
            "universe_entry_count_mismatch",
            "universe_reciprocal_orientation_missing",
            "universe_reciprocal_pair_count_mismatch",
            "universe_fold_distribution_mismatch",
            "p13_snapshot_build_failed",
            "canonical_serialization_mismatch",
            "p13_verification_mismatch",
        }
    )
    assert set(p14._ERROR_MESSAGES) == p14.GENERATION_REASON_CODES - {"empty_input"}
    for code, message in p14._ERROR_MESSAGES.items():
        assert isinstance(message, str) and 0 < len(message) < 300
        assert "Zxy" not in message


def test_a4_p10_policy_universe_is_closed():
    policy = p14.P10_OFFLINE_SNAPSHOT_GENERATION_POLICY
    assert policy.required_entries == 3610
    assert policy.required_reciprocal_pairs == 1805
    assert policy.expected_fold_orientations == (
        (2020, 336),
        (2021, 742),
        (2022, 1292),
        (2023, 1240),
    )
    assert sum(count for _, count in policy.expected_fold_orientations) == 3610
    assert policy.required_reciprocal_pairs * 2 == 3610


# --------------------------------------------------------------------------
# B. Construccion, contenedores, limites, duplicados, elementos invalidos
# --------------------------------------------------------------------------


def test_b1_generate_tuple_counts_and_diagnostics():
    result = p14.generate_tactical_recommendation_snapshot(_four())
    assert result.contract == p14.GENERATION_CONTRACT_NAME
    assert result.schema_version == p14.GENERATION_SCHEMA_VERSION
    assert result.state == p14.SNAPSHOT_GENERATION_STATE_AVAILABLE
    assert result.reason_codes == ()
    assert result.inputs_received == 4
    assert result.entries_generated == 4
    assert result.available_entries == 2
    assert result.partially_available_entries == 1
    assert result.not_available_entries == 1
    assert result.fold_orientations == ((2021, 4),)
    assert result.serialized_bytes == len(
        p13.serialize_persisted_tactical_recommendation_snapshot(result.snapshot)
    )
    assert result.snapshot_fingerprint == result.snapshot.fingerprint
    diagnostics = result.diagnostics
    assert (
        diagnostics.elements_inspected
        == diagnostics.results_validated
        == diagnostics.entries_built
        == 4
    )
    assert diagnostics.builder_calls == 1
    assert diagnostics.serializations == 1
    assert diagnostics.persistences == 0
    assert diagnostics.verifications == 0
    assert diagnostics.canonical_bytes == result.serialized_bytes
    assert diagnostics.estimated_universe_bytes == 0
    assert len(result.snapshot_fingerprint) == 64


def test_b2_order_and_container_independence():
    items = _four()
    reference = p14.generate_tactical_recommendation_snapshot(items)
    variants = (
        list(items),
        reversed(items),
        (x for x in items),
        tuple(reversed(items)),
    )
    for variant in variants:
        result = p14.generate_tactical_recommendation_snapshot(variant)
        assert result.snapshot_fingerprint == reference.snapshot_fingerprint
        assert (
            p14.canonical_snapshot_generation_json(result)
            == p14.canonical_snapshot_generation_json(reference)
        )
        for field in dataclasses.fields(result):
            assert getattr(result, field.name) == getattr(reference, field.name)


def test_b3_single_pass_iteration(monkeypatch):
    iterator = _CountingIterator(_four())
    result = p14.generate_tactical_recommendation_snapshot(iterator)
    assert result.entries_generated == 4
    # islice realiza exactamente una sonda de agotamiento extra en el
    # extremo; nunca un segundo paso sobre los datos.
    assert iterator.calls <= 5
    items = _four()
    probe = _CountingIterator(list(items))
    monkeypatch.setattr(p13, "MAX_ENTRIES", 2)
    with pytest.raises(p14.SnapshotGenerationLimitError):
        p14.generate_tactical_recommendation_snapshot(probe)
    # Con limite activo, islice se detiene en limite+1 sin sonda extra:
    # cada elemento se visita exactamente una vez (O(N), un solo paso).
    assert probe.calls == 3


def test_b4_empty_input_generic_state():
    result = p14.generate_tactical_recommendation_snapshot([])
    assert result.state == p14.SNAPSHOT_GENERATION_STATE_EMPTY_INPUT
    assert result.reason_codes == ("empty_input",)
    assert result.inputs_received == 0
    assert result.entries_generated == 0
    assert result.available_entries == 0
    assert result.partially_available_entries == 0
    assert result.not_available_entries == 0
    assert result.fold_orientations == ()
    assert result.snapshot.entry_count == 0
    p13.validate_persisted_tactical_recommendation_snapshot(result.snapshot)
    assert result.serialized_bytes == len(
        p13.serialize_persisted_tactical_recommendation_snapshot(result.snapshot)
    )
    assert result.diagnostics.estimated_universe_bytes == 0


def test_b5_empty_input_p10_policy_universe_mismatch():
    with pytest.raises(p14.SnapshotGenerationUniverseError) as excinfo:
        p14.generate_tactical_recommendation_snapshot(
            [], policy=p14.P10_OFFLINE_SNAPSHOT_GENERATION_POLICY
        )
    assert excinfo.value.stage == "universe"
    assert excinfo.value.reason_code == "universe_entry_count_mismatch"


@pytest.mark.parametrize("bad_input", [42, None, True, "abc", b"abc", {"a": 1}, object()])
def test_b6_non_iterable_inputs(bad_input):
    with pytest.raises(TypeError):
        p14.generate_tactical_recommendation_snapshot(bad_input)


@pytest.mark.parametrize(
    "bad_element",
    [{"x": 1}, SimpleNamespace(state="available"), ("tuple",), 7, None],
)
def test_b7_wrong_element_types(bad_element):
    with pytest.raises(p14.SnapshotGenerationInputError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([_r("available", 0, "A", "B"), bad_element])
    assert excinfo.value.stage == "input"
    assert (
        excinfo.value.reason_code
        == "element_not_tactical_prioritization_result"
    )


def test_b8_exact_duplicate_rejected():
    first = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    with pytest.raises(p14.SnapshotGenerationKeyError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([first, first])
    assert excinfo.value.stage == "key"
    assert excinfo.value.reason_code == "duplicate_key_exact"


def test_b9_conflicting_duplicate_rejected():
    first = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    second = _r("partial", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    assert (
        tactical_prioritization_result_fingerprint(first)
        != tactical_prioritization_result_fingerprint(second)
    )
    with pytest.raises(p14.SnapshotGenerationKeyError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([first, second])
    assert excinfo.value.reason_code == "duplicate_key_conflicting"


@pytest.mark.parametrize(
    ("entry_count", "max_entries"),
    [(3, 2), (2, 2)],
)
def test_b10_entry_limit_enforcement(entry_count, max_entries, monkeypatch):
    results = [
        _r("available", 0, "KaiOne", "LenBee"),
        _r("absent", 0, "LenBee", "KaiOne"),
        _r("partial", 0, "MoSea", "NiaDor"),
    ][:entry_count]
    monkeypatch.setattr(p13, "MAX_ENTRIES", max_entries)
    if entry_count > max_entries:
        with pytest.raises(p14.SnapshotGenerationLimitError) as excinfo:
            p14.generate_tactical_recommendation_snapshot(results)
        assert excinfo.value.stage == "limit"
        assert excinfo.value.reason_code == "entry_limit_exceeded"
    else:
        result = p14.generate_tactical_recommendation_snapshot(results)
        assert result.entries_generated == entry_count


def test_b11_upstream_invalid_result(monkeypatch):
    source = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    broken = _clone_result(source, matchup_query=None)
    with pytest.raises(p14.SnapshotGenerationUpstreamError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([broken])
    assert excinfo.value.stage == "upstream"
    assert excinfo.value.reason_code == "upstream_result_invalid"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_b11_hostile_iterator_error_is_sanitized():
    def hostile():
        yield _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
        raise RuntimeError("PRIVATE_ITERATOR_SECRET")

    with pytest.raises(p14.SnapshotGenerationInputError) as excinfo:
        p14.generate_tactical_recommendation_snapshot(hostile())
    assert excinfo.value.reason_code == "element_not_tactical_prioritization_result"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert "PRIVATE_ITERATOR_SECRET" not in repr(excinfo.value)


def test_b12_self_match_branch(monkeypatch):
    source = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    summary = _clone_summary(source.matchup_query, player=source.matchup_query.opponent)
    tampered = _clone_result(source, matchup_query=summary)
    monkeypatch.setattr(p14, "validate_tactical_prioritization_result", lambda _x: None)
    with pytest.raises(p14.SnapshotGenerationKeyError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([tampered])
    assert excinfo.value.stage == "key"
    assert excinfo.value.reason_code == "player_equals_opponent"


@pytest.mark.parametrize("as_of", ["2021-1-01", "2750-13-01", 20210101])
def test_b13_non_civil_as_of(monkeypatch, as_of):
    source = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    summary = _clone_summary(source.matchup_query, as_of_date=as_of)
    tampered = _clone_result(source, matchup_query=summary)
    monkeypatch.setattr(p14, "validate_tactical_prioritization_result", lambda _x: None)
    with pytest.raises(p14.SnapshotGenerationTemporalError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([tampered])
    assert excinfo.value.stage == "temporal"
    assert excinfo.value.reason_code == "as_of_date_not_civil"


# --------------------------------------------------------------------------
# C. Politica congelada
# --------------------------------------------------------------------------


def _mutated_scoring_policy():
    base = default_tactical_scoring_policy()
    policy = object.__new__(type(base))
    for name, value in vars(base).items():
        object.__setattr__(policy, name, value)
    object.__setattr__(policy, "executor_weight", 0.6)
    return policy


def _generic_policy(**overrides):
    base = dict(
        mode=p14.SNAPSHOT_GENERATION_MODE_GENERIC,
        scoring_policy=default_tactical_scoring_policy(),
        expected_encoder_policy=TacticalEncodingPolicy.COMPONENT_ONLY,
        expected_scope_strategy="global_only",
        expected_requested_patterns=("P02", "P04", "P05", "P06"),
        expected_serve_numbers=(1, 2),
        expected_minimum_labeled_attempts=50,
        expected_minimum_matches=5,
        expected_window_days=None,
        expected_top_k=3,
        required_entries=None,
        required_reciprocal_pairs=None,
        expected_fold_orientations=(),
    )
    base.update(overrides)
    return p14.SnapshotGenerationPolicy(**base)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"scoring_policy": _mutated_scoring_policy()},
            "scoring_policy_mismatch",
        ),
        (
            {"expected_encoder_policy": TacticalEncodingPolicy.PROFILE_ONLY},
            "matchup_query_policy_mismatch",
        ),
        (
            {"expected_requested_patterns": ("P02", "P04", "P05")},
            "matchup_query_policy_mismatch",
        ),
        ({"expected_serve_numbers": (1,)}, "matchup_query_policy_mismatch"),
        (
            {"expected_minimum_labeled_attempts": 10},
            "matchup_query_policy_mismatch",
        ),
        ({"expected_minimum_matches": 1}, "matchup_query_policy_mismatch"),
        ({"expected_window_days": 365}, "matchup_query_policy_mismatch"),
        ({"expected_top_k": 4}, "ranking_policy_mismatch"),
        (
            # El orden del patron se verifica primero en el summary:
            # el chequeo de orden de rankings es defensa en profundidad.
            {"expected_requested_patterns": ("P02", "P05", "P04", "P06")},
            "matchup_query_policy_mismatch",
        ),
    ],
)
def test_c1_policy_mutation_battery(overrides, reason):
    result = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    policy = _generic_policy(**overrides)
    with pytest.raises(p14.SnapshotGenerationPolicyError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([result], policy=policy)
    assert excinfo.value.stage == "policy"
    assert excinfo.value.reason_code == reason


@pytest.mark.parametrize("bad_policy", [None, {}, "generic", SimpleNamespace(mode="generic")])
def test_c2_policy_type_strictness(bad_policy):
    with pytest.raises((TypeError, ValueError)):
        p14.validate_snapshot_generation_policy(bad_policy)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "bogus"},
        {"expected_serve_numbers": ()},
        {"expected_serve_numbers": (1, True)},
        {"expected_top_k": 0},
        {"expected_top_k": True},
        {"expected_minimum_labeled_attempts": -1},
        {"expected_minimum_matches": -5},
        {"expected_window_days": 365.0},
        {"expected_fold_orientations": ((2021, -1),)},
        {"expected_fold_orientations": ((2021, "4"),)},
        {"scoring_policy": SimpleNamespace(allowed_scope="global")},
    ],
)
def test_c3_policy_field_validation_battery(overrides):
    with pytest.raises(ValueError):
        _generic_policy(**overrides)


def test_c4_generic_mode_forbids_required_universe():
    with pytest.raises(ValueError):
        _generic_policy(required_entries=3610)
    with pytest.raises(ValueError):
        _generic_policy(required_reciprocal_pairs=1805)
    with pytest.raises(ValueError):
        _generic_policy(expected_fold_orientations=((2021, 1),))


@pytest.mark.parametrize(
    "overrides",
    [
        {"required_entries": 3609},
        {"required_entries": 3611},
        {"required_entries": True},
        {"required_reciprocal_pairs": 1804},
        {
            "expected_fold_orientations": (
                (2020, 336),
                (2021, 742),
                (2022, 1292),
                (2023, 1239),
            )
        },
        {
            "expected_fold_orientations": (
                (2019, 336),
                (2021, 742),
                (2022, 1292),
                (2023, 1240),
            )
        },
        {"expected_top_k": 4},
        {"expected_minimum_labeled_attempts": 51},
        {"expected_requested_patterns": ("P02", "P04", "P05", "P03")},
        {"expected_serve_numbers": (1,)},
        {"expected_window_days": 0},
        {"expected_encoder_policy": TacticalEncodingPolicy.PROFILE_ONLY},
    ],
)
def test_c5_p10_policy_exactness_battery(overrides):
    base = {
        "mode": p14.SNAPSHOT_GENERATION_MODE_P10_OFFLINE,
        "scoring_policy": default_tactical_scoring_policy(),
        "expected_encoder_policy": TacticalEncodingPolicy.COMPONENT_ONLY,
        "expected_scope_strategy": "global_only",
        "expected_requested_patterns": ("P02", "P04", "P05", "P06"),
        "expected_serve_numbers": (1, 2),
        "expected_minimum_labeled_attempts": 50,
        "expected_minimum_matches": 5,
        "expected_window_days": None,
        "expected_top_k": 3,
        "required_entries": 3610,
        "required_reciprocal_pairs": 1805,
        "expected_fold_orientations": (
            (2020, 336),
            (2021, 742),
            (2022, 1292),
            (2023, 1240),
        ),
    }
    base.update(overrides)
    with pytest.raises(ValueError):
        p14.SnapshotGenerationPolicy(**base)


def test_c6_policy_is_frozen():
    policy = p14.GENERIC_SNAPSHOT_GENERATION_POLICY
    with pytest.raises(FrozenInstanceError):
        policy.mode = "bogus"
    with pytest.raises(FrozenInstanceError):
        policy.expected_top_k = 4


def test_c7_policy_identity_preserved_in_result():
    custom = _generic_policy()
    result = p14.generate_tactical_recommendation_snapshot(
        [
            _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT),
            _r("absent", 0, SENTINEL_OPPONENT, SENTINEL_PLAYER),
        ],
        policy=custom,
    )
    assert result.policy is custom


# --------------------------------------------------------------------------
# D. Frontera temporal sellada
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("as_of", "expected_state"),
    [
        (date(2023, 12, 31), "available"),
        (date(2024, 1, 1), "sealed"),
        (date(2025, 6, 30), "sealed"),
        (date(2026, 12, 31), "sealed"),
    ],
)
def test_d1_development_cutoff_boundary(as_of, expected_state):
    result = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT, as_of)
    if expected_state == "available":
        generated = p14.generate_tactical_recommendation_snapshot([result])
        assert generated.state == p14.SNAPSHOT_GENERATION_STATE_AVAILABLE
    else:
        with pytest.raises(p14.SnapshotGenerationTemporalError) as excinfo:
            p14.generate_tactical_recommendation_snapshot([result])
        assert excinfo.value.stage == "temporal"
        assert excinfo.value.reason_code == "as_of_date_in_sealed_test_range"


def test_d2_error_carries_closed_stage_reason_and_constant_message():
    result = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT, date(2024, 1, 1))
    with pytest.raises(p14.SnapshotGenerationTemporalError) as excinfo:
        p14.generate_tactical_recommendation_snapshot([result])
    error = excinfo.value
    assert isinstance(error, p14.TacticalRecommendationSnapshotGenerationError)
    assert error.stage in p14.GENERATION_STAGES
    assert error.reason_code in p14.GENERATION_REASON_CODES
    assert str(error) == p14._ERROR_MESSAGES[error.reason_code]


def test_d3_sealed_constant_enforced_by_generator_and_validator(monkeypatch):
    pair = _sentinel_pair()
    monkeypatch.setattr(p14, "SEALED_TEST_FIRST_DAY", date(2020, 1, 1))
    with pytest.raises(p14.SnapshotGenerationTemporalError):
        p14.generate_tactical_recommendation_snapshot(pair)
    monkeypatch.undo()
    generated = p14.generate_tactical_recommendation_snapshot(pair)
    monkeypatch.setattr(p14, "SEALED_TEST_FIRST_DAY", date(2020, 1, 1))
    with pytest.raises(ValueError, match="test sellado"):
        p14.validate_tactical_recommendation_snapshot_generation_result(generated)


def test_d4_sealed_result_rejected_before_key_and_policy_stages():
    first, second = _sentinel_pair(date(2024, 1, 1))
    with pytest.raises(p14.SnapshotGenerationTemporalError):
        p14.generate_tactical_recommendation_snapshot([first, first, second])


# --------------------------------------------------------------------------
# E. Resultado, particiones, universo P10, validador reconstructivo
# --------------------------------------------------------------------------


def test_e1_mixed_state_partition():
    result = p14.generate_tactical_recommendation_snapshot(_four())
    assert (
        result.available_entries
        + result.partially_available_entries
        + result.not_available_entries
        == result.entries_generated
    )


def test_e2_multi_fold_generic():
    results = (
        _r("available", 0, "KaiOne", "LenBee", date(2022, 6, 15)),
        _r("absent", 0, "LenBee", "KaiOne", date(2022, 6, 15)),
        _r("available", 0, "MoSea", "NiaDor", date(2023, 6, 15)),
        _r("absent", 0, "NiaDor", "MoSea", date(2023, 6, 15)),
    )
    result = p14.generate_tactical_recommendation_snapshot(results)
    assert result.fold_orientations == ((2022, 2), (2023, 2))


def test_e3_result_metadata_and_freeze():
    result = p14.generate_tactical_recommendation_snapshot(_sentinel_pair())
    assert result.policy is p14.GENERIC_SNAPSHOT_GENERATION_POLICY
    with pytest.raises(FrozenInstanceError):
        result.entries_generated = 1
    with pytest.raises(FrozenInstanceError):
        result.diagnostics.builder_calls = 2
    assert type(result.snapshot) is p13.PersistedTacticalRecommendationSnapshot
    p13.validate_persisted_tactical_recommendation_snapshot(result.snapshot)


def test_e4_p10_full_path_mini_policy_exact_universe(monkeypatch):
    monkeypatch.setattr(p14, "validate_snapshot_generation_policy", lambda _p: None)
    results = _four()
    policy = _mini_p10_policy(required_entries=4, required_pairs=2, folds=((2021, 4),))
    result = p14.generate_tactical_recommendation_snapshot(
        _p10_records(results), policy=policy
    )
    assert result.state == p14.SNAPSHOT_GENERATION_STATE_AVAILABLE
    assert result.fold_orientations == ((2021, 4),)
    assert result.diagnostics.estimated_universe_bytes == result.serialized_bytes
    assert result.policy is policy


def test_e4b_p10_reciprocal_orientation_missing(monkeypatch):
    monkeypatch.setattr(p14, "validate_snapshot_generation_policy", lambda _p: None)
    results = [
        _r("available", 0, "A1", "B1"),
        _r("absent", 0, "B1", "A1"),
        _r("available", 0, "C1", "D1"),
    ]
    policy = _mini_p10_policy(required_entries=3, required_pairs=2, folds=((2021, 3),))
    records = _p10_records(results, ("MATCH_0", "MATCH_0", "MATCH_1"))
    with pytest.raises(p14.SnapshotGenerationUniverseError) as excinfo:
        p14.generate_tactical_recommendation_snapshot(records, policy=policy)
    assert excinfo.value.stage == "universe"
    assert (
        excinfo.value.reason_code == "universe_reciprocal_orientation_missing"
    )


def test_e4c_p10_fold_distribution_mismatch(monkeypatch):
    monkeypatch.setattr(p14, "validate_snapshot_generation_policy", lambda _p: None)
    results = (
        _r("available", 0, "A1", "B1"),
        _r("absent", 0, "B1", "A1"),
        _r("available", 0, "A2", "B2", date(2022, 6, 15)),
        _r("absent", 0, "B2", "A2", date(2022, 6, 15)),
    )
    policy = _mini_p10_policy(required_entries=4, required_pairs=2, folds=((2021, 4),))
    with pytest.raises(p14.SnapshotGenerationUniverseError) as excinfo:
        p14.generate_tactical_recommendation_snapshot(
            _p10_records(results), policy=policy
        )
    assert excinfo.value.reason_code == "universe_fold_distribution_mismatch"


def test_e4d_p10_pair_count_mismatch(monkeypatch):
    monkeypatch.setattr(p14, "validate_snapshot_generation_policy", lambda _p: None)
    policy = _mini_p10_policy(required_entries=4, required_pairs=3, folds=((2021, 4),))
    with pytest.raises(p14.SnapshotGenerationUniverseError) as excinfo:
        p14.generate_tactical_recommendation_snapshot(
            _p10_records(_four()), policy=policy
        )
    assert excinfo.value.reason_code == "universe_reciprocal_pair_count_mismatch"


def test_e4d_modes_require_distinct_element_contracts(monkeypatch):
    monkeypatch.setattr(p14, "validate_snapshot_generation_policy", lambda _p: None)
    policy = _mini_p10_policy(required_entries=2, required_pairs=1, folds=((2021, 2),))
    with pytest.raises(p14.SnapshotGenerationInputError) as p10_error:
        p14.generate_tactical_recommendation_snapshot(_sentinel_pair(), policy=policy)
    assert p10_error.value.reason_code == "element_not_p10_offline_snapshot_record"
    with pytest.raises(p14.SnapshotGenerationInputError):
        p14.generate_tactical_recommendation_snapshot(_p10_records(_sentinel_pair()))


def test_e4d_same_players_same_day_different_matches_are_not_false_pairs(monkeypatch):
    monkeypatch.setattr(p14, "validate_snapshot_generation_policy", lambda _p: None)
    first, second = _sentinel_pair()
    records = (
        p14.P10OfflineSnapshotRecord("MATCH_A", "validation_2021", "player_1_vs_player_2", first),
        p14.P10OfflineSnapshotRecord("MATCH_A", "validation_2021", "player_2_vs_player_1", second),
        p14.P10OfflineSnapshotRecord("MATCH_B", "validation_2021", "player_1_vs_player_2", first),
        p14.P10OfflineSnapshotRecord("MATCH_B", "validation_2021", "player_2_vs_player_1", second),
    )
    policy = _mini_p10_policy(required_entries=4, required_pairs=2, folds=((2021, 4),))
    with pytest.raises(p14.SnapshotGenerationKeyError) as captured:
        p14.generate_tactical_recommendation_snapshot(records, policy=policy)
    assert captured.value.reason_code == "duplicate_key_exact"


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_match_id": ""},
        {"target_match_id": True},
        {"fold": "validation_2022"},
        {"orientation": "reversed"},
        {"prioritization": object()},
    ],
)
def test_e4d_p10_record_is_strict_and_temporally_coherent(overrides):
    values = {
        "target_match_id": "MATCH_A",
        "fold": "validation_2021",
        "orientation": "player_1_vs_player_2",
        "prioritization": _sentinel_pair()[0],
    }
    values.update(overrides)
    with pytest.raises((TypeError, p14.TacticalRecommendationSnapshotGenerationError)):
        p14.P10OfflineSnapshotRecord(**values)


@pytest.mark.parametrize(
    ("records", "policy", "expected"),
    [
        (
            _p10_records((
                _r("available", 0, "A", "B"),
                _r("absent", 0, "B", "A"),
                _r("available", 0, "C", "D"),
                _r("absent", 0, "D", "C"),
            )),
            SimpleNamespace(required_reciprocal_pairs=2, expected_fold_orientations=((2021, 4),)),
            ((2021, 4),),
        ),
        (
            _p10_records((
                _r("available", 0, "A", "B"),
                _r("absent", 0, "B", "A"),
                _r("available", 0, "C", "D"),
            ), ("MATCH_0", "MATCH_0", "MATCH_1")),
            SimpleNamespace(required_reciprocal_pairs=2, expected_fold_orientations=((2021, 3),)),
            "universe_reciprocal_orientation_missing",
        ),
        (
            _p10_records((
                _r("available", 0, "A", "B"),
                _r("absent", 0, "B", "A"),
                _r("available", 0, "C", "D"),
                _r("absent", 0, "D", "C"),
            )),
            SimpleNamespace(required_reciprocal_pairs=3, expected_fold_orientations=((2021, 4),)),
            "universe_reciprocal_pair_count_mismatch",
        ),
        (
            _p10_records((
                _r("available", 0, "A", "B"),
                _r("absent", 0, "B", "A"),
                _r("available", 0, "C", "D", date(2022, 1, 1)),
                _r("absent", 0, "D", "C", date(2022, 1, 1)),
            )),
            SimpleNamespace(required_reciprocal_pairs=2, expected_fold_orientations=((2021, 4),)),
            "universe_fold_distribution_mismatch",
        ),
    ],
)
def test_e4e_reconcile_direct(records, policy, expected):
    if isinstance(expected, str):
        with pytest.raises(p14.SnapshotGenerationUniverseError) as excinfo:
            p14._reconcile_p10_universe(records, policy)
        assert excinfo.value.reason_code == expected
    else:
        assert p14._reconcile_p10_universe(records, policy) == expected


def test_e5_validator_rejects_wrong_types():
    with pytest.raises(TypeError):
        p14.validate_tactical_recommendation_snapshot_generation_result(None)
    with pytest.raises(TypeError):
        p14.validate_tactical_recommendation_snapshot_generation_result({"x": 1})
    with pytest.raises(TypeError):
        p14.validate_tactical_recommendation_snapshot_generation_result(object())


@pytest.mark.parametrize(
    "overrides",
    [
        {"entries_generated": 3},
        {"inputs_received": 0},
        {"available_entries": 0},
        {"partially_available_entries": 2},
        {"not_available_entries": 3},
        {"fold_orientations": ((2020, 1),)},
        {"reason_codes": ("empty_input",)},
        {"state": p14.SNAPSHOT_GENERATION_STATE_EMPTY_INPUT},
        {"contract": "other_contract"},
        {"schema_version": "2.0.0"},
        {"serialized_bytes": 123},
        {"snapshot_fingerprint": "0" * 64},
        {"fold_orientations": ((True, 1),)},
        {"reason_codes": ("bogus_code",)},
        {"diagnostics": {"builder_calls": 0}},
        {"diagnostics": {"serializations": 0}},
        {"diagnostics": {"canonical_bytes": 42}},
        {"diagnostics": {"elements_inspected": 3}},
    ],
)
def test_e5a_validator_tamper_battery(overrides):
    baseline = p14.generate_tactical_recommendation_snapshot(
        list(_sentinel_pair())
    )
    if "diagnostics" in overrides and isinstance(overrides["diagnostics"], dict):
        clone = _clone_result(
            baseline,
            diagnostics=dataclasses.replace(
                baseline.diagnostics, **overrides["diagnostics"]
            ),
        )
    else:
        clone = _clone_result(baseline, **overrides)
    with pytest.raises((TypeError, ValueError)):
        p14.validate_tactical_recommendation_snapshot_generation_result(clone)


def test_e5b_validator_detects_sealed_dates_inside_snapshot():
    sealed = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT, date(2024, 1, 1))
    sealed_snapshot = p13.build_persisted_tactical_recommendation_snapshot((sealed,))
    baseline = p14.generate_tactical_recommendation_snapshot(
        list(_sentinel_pair())
    )
    serialized = len(
        p13.serialize_persisted_tactical_recommendation_snapshot(sealed_snapshot)
    )
    clone = _clone_result(
        baseline,
        snapshot=sealed_snapshot,
        snapshot_fingerprint=sealed_snapshot.fingerprint,
        serialized_bytes=serialized,
        fold_orientations=((2024, 1),),
        inputs_received=1,
        entries_generated=1,
        available_entries=1,
        partially_available_entries=0,
        not_available_entries=0,
        diagnostics=dataclasses.replace(
            baseline.diagnostics,
            elements_inspected=1,
            results_validated=1,
            entries_built=1,
            canonical_bytes=serialized,
        ),
    )
    with pytest.raises(ValueError, match="test sellado"):
        p14.validate_tactical_recommendation_snapshot_generation_result(clone)


def test_e5c_validator_rejects_p10_estimation_tamper(monkeypatch):
    monkeypatch.setattr(p14, "validate_snapshot_generation_policy", lambda _p: None)
    policy = _mini_p10_policy(required_entries=4, required_pairs=2, folds=((2021, 4),))
    baseline = p14.generate_tactical_recommendation_snapshot(
        _p10_records(_four()), policy=policy
    )
    clone = _clone_result(
        baseline,
        diagnostics=dataclasses.replace(
            baseline.diagnostics,
            estimated_universe_bytes=baseline.diagnostics.estimated_universe_bytes + 1,
        ),
    )
    with pytest.raises(ValueError, match="Estimacion de universo"):
        p14.validate_tactical_recommendation_snapshot_generation_result(clone)


# --------------------------------------------------------------------------
# F. Serializacion canonica y fingerprint
# --------------------------------------------------------------------------


def test_f1_canonical_json_closed_shape():
    result = p14.generate_tactical_recommendation_snapshot(_sentinel_pair())
    payload = json.loads(p14.canonical_snapshot_generation_json(result).decode("utf-8"))
    assert set(payload) == {
        "available_entries",
        "contract",
        "diagnostics",
        "entries_generated",
        "fold_orientations",
        "inputs_received",
        "partially_available_entries",
        "not_available_entries",
        "policy",
        "reason_codes",
        "schema_version",
        "serialized_bytes",
        "snapshot_fingerprint",
        "state",
    }
    assert set(payload["diagnostics"]) == {
        "canonical_bytes",
        "elements_inspected",
        "entries_built",
        "builder_calls",
        "persistences",
        "serializations",
        "verifications",
        "results_validated",
        "estimated_universe_bytes",
    }
    assert set(payload["policy"]) == {
        "expected_encoder_policy",
        "expected_fold_orientations",
        "expected_minimum_labeled_attempts",
        "expected_minimum_matches",
        "expected_requested_patterns",
        "expected_scope_strategy",
        "expected_serve_numbers",
        "expected_top_k",
        "expected_window_days",
        "mode",
        "required_entries",
        "required_reciprocal_pairs",
        "scoring_policy",
    }
    assert set(payload["policy"]["scoring_policy"]) == {
        "abstention_rules",
        "allowed_scope",
        "comparability_rule",
        "contract_version",
        "executor_weight",
        "formula",
        "interpretation",
        "name",
        "opponent_allowed_weight",
        "reconciliations",
        "uncertainty_method",
        "version",
    }
    assert payload["contract"] == p14.GENERATION_CONTRACT_NAME
    assert payload["schema_version"] == p14.GENERATION_SCHEMA_VERSION


def test_f2_canonical_has_no_privacy_identifiers():
    result = p14.generate_tactical_recommendation_snapshot(_sentinel_pair())
    text = p14.canonical_snapshot_generation_json(result).decode("utf-8")
    for sentinel in SENTINELS:
        assert sentinel not in text
    snapshot_text = p13.serialize_persisted_tactical_recommendation_snapshot(
        result.snapshot
    ).decode("utf-8")
    assert SENTINEL_PLAYER in snapshot_text
    assert SENTINEL_OPPONENT in snapshot_text


def test_f3_generation_fingerprint_derivation():
    result = p14.generate_tactical_recommendation_snapshot(_sentinel_pair())
    canonical = p14.canonical_snapshot_generation_json(result)
    independent_payload = json.loads(canonical.decode("utf-8"))
    assert "fingerprint" not in independent_payload
    independently_serialized = json.dumps(
        independent_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected = (
        sha256(
            b"tennis-tactical-recommendation-snapshot-generation\x00"
            + independently_serialized
        ).hexdigest().upper()
    )
    assert p14.GENERATION_FINGERPRINT_DOMAIN == b"tennis-tactical-recommendation-snapshot-generation\x00"
    assert p14.snapshot_generation_fingerprint(result) == expected
    assert len(expected) == 64
    assert expected == expected.upper()
    canonical_payload = json.loads(canonical.decode("utf-8"))
    assert canonical_payload["snapshot_fingerprint"] == result.snapshot_fingerprint
    other = p14.generate_tactical_recommendation_snapshot(
        (
            _r("available", 0, "Q1", "R1"),
            _r("absent", 0, "R1", "Q1"),
        )
    )
    assert p14.snapshot_generation_fingerprint(other) != expected


def test_f4_no_timestamp_fields_in_canonical_payload():
    result = p14.generate_tactical_recommendation_snapshot(_sentinel_pair())
    payload = json.loads(p14.canonical_snapshot_generation_json(result).decode("utf-8"))
    flagged = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                text = key.lower()
                if (
                    "timestamp" in text
                    or "created" in text
                    or "updated" in text
                    or key in ("time", "ts")
                    or key.endswith(("_at", "_ts"))
                ):
                    flagged.append(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    assert flagged == []


def test_f5_persist_wrapper_diagnostics_delta(tmp_path):
    destination = tmp_path / "snapshot.json"
    generated = p14.generate_tactical_recommendation_snapshot(_sentinel_pair())
    persisted = p14.generate_and_persist_tactical_recommendation_snapshot(
        _sentinel_pair(), destination
    )
    for field in dataclasses.fields(persisted):
        if field.name == "diagnostics":
            continue
        assert (
            getattr(persisted, field.name) == getattr(generated, field.name)
        )
    assert (
        persisted.diagnostics.serializations
        == generated.diagnostics.serializations + 2
    )
    assert persisted.diagnostics.persistences == 1
    assert persisted.diagnostics.verifications == 1
    for name in ("elements_inspected", "results_validated", "entries_built", "builder_calls", "canonical_bytes", "estimated_universe_bytes"):
        assert getattr(persisted.diagnostics, name) == getattr(generated.diagnostics, name)


# --------------------------------------------------------------------------
# G. Persistencia delegada a P13
# --------------------------------------------------------------------------


def test_g1_generate_and_persist_roundtrip(tmp_path):
    destination = tmp_path / "snapshot.json"
    results = _sentinel_pair()
    result = p14.generate_and_persist_tactical_recommendation_snapshot(
        results, destination
    )
    expected_bytes = p13.serialize_persisted_tactical_recommendation_snapshot(
        result.snapshot
    )
    assert destination.read_bytes() == expected_bytes
    assert (
        p13.verify_persisted_tactical_recommendation_snapshot(destination)
        == result.snapshot_fingerprint
    )
    loaded = p13.load_persisted_tactical_recommendation_snapshot(destination)
    assert loaded.fingerprint == result.snapshot_fingerprint


def test_g2_delegated_call_counts(monkeypatch, tmp_path):
    calls = {"build": 0, "persist": 0, "verify": 0}
    real_build = p13.build_persisted_tactical_recommendation_snapshot
    real_persist = p13.persist_persisted_tactical_recommendation_snapshot
    real_verify = p13.verify_persisted_tactical_recommendation_snapshot

    def counting_build(*args, **kwargs):
        calls["build"] += 1
        return real_build(*args, **kwargs)

    def counting_persist(*args, **kwargs):
        calls["persist"] += 1
        return real_persist(*args, **kwargs)

    def counting_verify(*args, **kwargs):
        calls["verify"] += 1
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(p13, "build_persisted_tactical_recommendation_snapshot", counting_build)
    monkeypatch.setattr(p13, "persist_persisted_tactical_recommendation_snapshot", counting_persist)
    monkeypatch.setattr(p13, "verify_persisted_tactical_recommendation_snapshot", counting_verify)
    destination = tmp_path / "snapshot.json"
    p14.generate_tactical_recommendation_snapshot(_sentinel_pair())
    assert calls["build"] == 1
    assert calls["persist"] == 0
    assert calls["verify"] == 0
    p14.generate_and_persist_tactical_recommendation_snapshot(
        _sentinel_pair(), destination
    )
    assert calls["build"] == 2
    assert calls["persist"] == 1
    assert calls["verify"] == 1


def test_g3_p13_persist_failure_propagates(tmp_path):
    destination = tmp_path / "missing" / "parent" / "snapshot.json"
    with pytest.raises(p13.SnapshotUnavailableError):
        p14.generate_and_persist_tactical_recommendation_snapshot(
            _sentinel_pair(), destination
        )


def test_g4_p13_verify_mismatch_wrapped(monkeypatch, tmp_path):
    destination = tmp_path / "snapshot.json"
    monkeypatch.setattr(
        p13,
        "verify_persisted_tactical_recommendation_snapshot",
        lambda _path: "0" * 64,
    )
    with pytest.raises(p14.SnapshotGenerationBuildError) as excinfo:
        p14.generate_and_persist_tactical_recommendation_snapshot(
            _sentinel_pair(), destination
        )
    assert excinfo.value.stage == "build"
    assert excinfo.value.reason_code == "p13_verification_mismatch"


def test_g5_no_temporary_artifacts(tmp_path):
    destination = tmp_path / "snapshot.json"
    p14.generate_and_persist_tactical_recommendation_snapshot(
        _sentinel_pair(), destination
    )
    assert [path.name for path in sorted(tmp_path.iterdir())] == ["snapshot.json"]


# --------------------------------------------------------------------------
# H. Rendimiento / arquitectura
# --------------------------------------------------------------------------


def test_h1_single_pass_and_scale_guard():
    pairs = [
        (f"S{i:02d}", f"T{i:02d}") for i in range(16)
    ]
    results = []
    for player, opponent in pairs:
        results.extend((_r("available", 0, player, opponent), _r("absent", 0, opponent, player)))
    iterator = _CountingIterator(results)
    started = time.perf_counter()
    result = p14.generate_tactical_recommendation_snapshot(iterator)
    elapsed = time.perf_counter() - started
    # islice hace una sonda de agotamiento extra sobre un iterator finito;
    # nunca un segundo paso completo sobre los datos.
    assert iterator.calls <= 33
    assert result.entries_generated == 32
    assert elapsed < 120.0


def test_h2_ast_architecture_guard():
    source = Path(p14.__file__).read_bytes().decode("utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    allowed = {
        "__future__",
        "dataclasses",
        "datetime",
        "hashlib",
        "itertools",
        "json",
        "re",
        "typing",
        "types",
        "src.recommender",
        "src.recommender.persisted_tactical_recommendation_provider",
        "src.recommender.tactical_feature_encoder",
        "src.recommender.tactical_prioritization",
    }
    assert imported <= allowed
    banned_names = {
        "open",
        "eval",
        "exec",
        "input",
        "breakpoint",
        "compile",
        "globals",
        "locals",
        "__import__",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned_names:
            pytest.fail(f"Nombre banido usado en el modulo P14: {node.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_names


def test_h3_p14_resuelve_el_limite_p13_de_forma_dinamica(monkeypatch):
    result = p14.generate_tactical_recommendation_snapshot(_four())
    # Limite productivo (P16): 256 MiB; el snapshot sintetico pasa.
    assert p13.MAX_SNAPSHOT_BYTES == 256 * 1024 * 1024
    assert result.serialized_bytes < p13.MAX_SNAPSHOT_BYTES
    # El limite se resuelve en tiempo de llamada: reducido a un byte
    # menos que el serializado, la construccion falla con rationale
    # cerrado y sin snapshot parcial.
    monkeypatch.setattr(p13, "MAX_SNAPSHOT_BYTES", result.serialized_bytes - 1)
    with pytest.raises(p14.TacticalRecommendationSnapshotGenerationError) as exc:
        p14.generate_tactical_recommendation_snapshot(_four())
    assert exc.value.reason_code == "p13_snapshot_build_failed"


# --------------------------------------------------------------------------
# I. Privacidad recursiva (sentinelas)
# --------------------------------------------------------------------------


def test_i1_result_surface_has_no_privacy_identifiers():
    result = p14.generate_tactical_recommendation_snapshot(
        [_r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)]
    )
    surface = {
        field.name: getattr(result, field.name)
        for field in dataclasses.fields(result)
        if field.name != "snapshot"
    }
    assert _private_scan(surface) == []
    assert _private_scan(p14.canonical_snapshot_generation_json(result)) == []
    assert _private_scan(p14.snapshot_generation_fingerprint(result)) == []


@pytest.mark.parametrize(
    ("builder", "error_cls", "stage", "reason"),
    [
        (
            lambda ctx: [ctx["source"], {"bad": "element"}],
            p14.SnapshotGenerationInputError,
            "input",
            "element_not_tactical_prioritization_result",
        ),
        (
            lambda ctx: [_clone_result(ctx["source"], matchup_query=None)],
            p14.SnapshotGenerationUpstreamError,
            "upstream",
            "upstream_result_invalid",
        ),
        (
            lambda ctx: [ctx["source"]],
            p14.SnapshotGenerationPolicyError,
            "policy",
            "scoring_policy_mismatch",
        ),
        (
            lambda ctx: [
                _r(
                    "available",
                    0,
                    SENTINEL_PLAYER,
                    SENTINEL_OPPONENT,
                    date(2024, 1, 1),
                )
            ],
            p14.SnapshotGenerationTemporalError,
            "temporal",
            "as_of_date_in_sealed_test_range",
        ),
        (
            lambda ctx: [ctx["source"], ctx["source"]],
            p14.SnapshotGenerationKeyError,
            "key",
            "duplicate_key_exact",
        ),
    ],
)
def test_i2_error_surface_has_no_privacy_identifiers(monkeypatch, builder, error_cls, stage, reason):
    source = _r("available", 0, SENTINEL_PLAYER, SENTINEL_OPPONENT)
    ctx = {"source": source}
    if stage == "policy":
        policy = _generic_policy(scoring_policy=_mutated_scoring_policy())
        with pytest.raises(error_cls) as excinfo:
            p14.generate_tactical_recommendation_snapshot(builder(ctx), policy=policy)
    else:
        with pytest.raises(error_cls) as excinfo:
            p14.generate_tactical_recommendation_snapshot(builder(ctx))
    error = excinfo.value
    assert error.stage == stage
    assert error.reason_code == reason
    assert _private_scan(error) == []
    assert _private_scan(str(error)) == []
    assert _private_scan(type(error).__name__) == []


# --------------------------------------------------------------------------
# J. Superficie publica
# --------------------------------------------------------------------------


def test_j1_public_surface_exact():
    expected = (
        "GENERATION_CONTRACT_NAME",
        "GENERATION_FINGERPRINT_DOMAIN",
        "GENERATION_REASON_CODES",
        "GENERATION_SCHEMA_VERSION",
        "GENERATION_STAGES",
        "GENERIC_SNAPSHOT_GENERATION_POLICY",
        "P10_OFFLINE_SNAPSHOT_GENERATION_POLICY",
        "P10OfflineSnapshotRecord",
        "SEALED_TEST_FIRST_DAY",
        "SNAPSHOT_GENERATION_MODE_GENERIC",
        "SNAPSHOT_GENERATION_MODE_P10_OFFLINE",
        "SNAPSHOT_GENERATION_STATE_AVAILABLE",
        "SNAPSHOT_GENERATION_STATE_EMPTY_INPUT",
        "SnapshotGenerationBuildError",
        "SnapshotGenerationDiagnostics",
        "SnapshotGenerationInputError",
        "SnapshotGenerationKeyError",
        "SnapshotGenerationLimitError",
        "SnapshotGenerationPolicy",
        "SnapshotGenerationPolicyError",
        "SnapshotGenerationTemporalError",
        "SnapshotGenerationUniverseError",
        "SnapshotGenerationUpstreamError",
        "TacticalRecommendationSnapshotGenerationError",
        "TacticalRecommendationSnapshotGenerationResult",
        "canonical_snapshot_generation_json",
        "generate_and_persist_tactical_recommendation_snapshot",
        "generate_tactical_recommendation_snapshot",
        "snapshot_generation_fingerprint",
        "validate_snapshot_generation_policy",
        "validate_tactical_recommendation_snapshot_generation_result",
    )
    assert tuple(p14.__all__) == expected
    for name in expected:
        assert hasattr(p14, name)
    assert issubclass(p14.SnapshotGenerationInputError, p14.TacticalRecommendationSnapshotGenerationError)
    assert p14.GENERATION_FINGERPRINT_DOMAIN.endswith(b"\x00")
