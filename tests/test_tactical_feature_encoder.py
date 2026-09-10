"""Pruebas sinteticas del codificador estable de senales tacticas."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import src.recommender.tactical_feature_encoder as encoder
from src.recommender.tactical_feature_encoder import (
    BATCH_FINGERPRINT_DOMAIN,
    CONTEXT_FEATURE_NAMES,
    FEATURE_CONTRACT_VERSION,
    MASK_FEATURE_NAMES,
    P02_FEATURE_NAMES,
    P04_FEATURE_NAMES,
    P05_FEATURE_NAMES,
    P06_CODE_ORDER,
    P06_FEATURE_NAMES,
    P09_FEATURE_NAMES,
    P09_PROFILE_ORDER,
    PATTERN_ORDER,
    SCHEMA_FINGERPRINT_DOMAIN,
    VECTOR_FINGERPRINT_DOMAIN,
    TacticalEncodingPolicy,
    TacticalFeatureAvailability,
    TacticalFeatureBatch,
    TacticalFeatureContractError,
    TacticalFeatureSchema,
    TacticalFeatureVector,
    build_tactical_feature_schema,
    canonical_feature_batch_json,
    canonical_feature_schema_json,
    canonical_feature_vector_json,
    encode_tactical_attempt,
    encode_tactical_batch,
    feature_batch_fingerprint,
    feature_schema_fingerprint,
    feature_vector_fingerprint,
    validate_tactical_feature_batch,
    validate_tactical_feature_schema,
    validate_tactical_feature_vector,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
)


MODULE_PATH = Path(encoder.__file__)
MANUAL_SHOT_ORDER = tuple("fbrsvzopuylmhijkt")
MANUAL_PROFILE_ORDER = tuple(
    sorted(
        f"{shot}|{direction}|{depth}"
        for shot in MANUAL_SHOT_ORDER
        for direction in ("1", "2", "3")
        for depth in ("7", "8", "9")
    )
)
POLICIES = tuple(TacticalEncodingPolicy)


def _extraction(sequence="4f17", serve_number=1, previous_fault=False):
    request = AttemptSignalRequest(
        ORCHESTRATOR_CONTRACT_VERSION,
        sequence,
        serve_number,
        previous_fault,
    )
    return extract_tactical_signals_for_attempt(request)


def _schema(policy=TacticalEncodingPolicy.COMPONENT_ONLY):
    return build_tactical_feature_schema(policy)


def _vector(
    sequence="4f17",
    serve_number=1,
    previous_fault=False,
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
):
    return encode_tactical_attempt(
        _extraction(sequence, serve_number, previous_fault), _schema(policy)
    )


def _value(vector, name):
    schema = _schema(vector.policy)
    index = dict(schema.name_to_index)
    return vector.values[index[name]]


def _changed_values(vector, updates):
    values = list(vector.values)
    index = dict(_schema(vector.policy).name_to_index)
    for name, value in updates.items():
        values[index[name]] = value
    return tuple(values)


def _batch(extractions, schema):
    vectors = tuple(encode_tactical_attempt(item, schema) for item in extractions)
    return encode_tactical_batch(vectors, schema)


@pytest.mark.parametrize(
    ("policy", "tactical_count", "total_count", "encoded", "redundant", "purpose"),
    [
        (
            TacticalEncodingPolicy.COMPONENT_ONLY,
            26,
            39,
            ("P02", "P04", "P05", "P06"),
            False,
            "component_features_for_future_models",
        ),
        (
            TacticalEncodingPolicy.PROFILE_ONLY,
            156,
            169,
            ("P02", "P09"),
            False,
            "profile_features_for_future_models",
        ),
        (
            TacticalEncodingPolicy.COMPONENTS_AND_PROFILE,
            179,
            192,
            ("P02", "P04", "P05", "P06", "P09"),
            True,
            "audit_or_interaction_research_only",
        ),
    ],
)
def test_three_schemas_have_exact_contractual_sizes_and_purposes(
    policy, tactical_count, total_count, encoded, redundant, purpose
):
    schema = _schema(policy)
    assert len(schema.tactical_feature_names) == tactical_count
    assert schema.feature_count == total_count == len(schema.feature_names)
    assert schema.encoded_pattern_order == encoded
    assert schema.redundant_representation is redundant
    assert schema.contractual_purpose == purpose
    assert schema.pattern_catalog == ("P02", "P04", "P05", "P06", "P09")
    assert schema.feature_names == (
        schema.tactical_feature_names
        + schema.mask_feature_names
        + schema.context_feature_names
    )


def test_default_policy_is_component_only():
    schema = build_tactical_feature_schema()
    assert schema.policy is TacticalEncodingPolicy.COMPONENT_ONLY
    assert schema == _schema(TacticalEncodingPolicy.COMPONENT_ONLY)


def test_policy_and_availability_domains_are_exact_and_closed():
    assert tuple(item.value for item in TacticalEncodingPolicy) == (
        "component_only",
        "profile_only",
        "components_and_profile",
    )
    assert tuple(item.value for item in TacticalFeatureAvailability) == (
        "tactical_features_available",
        "no_tactical_feature_observed",
    )


def test_feature_catalogues_names_and_order_are_manual_and_exact():
    assert P02_FEATURE_NAMES == tuple(
        f"p02.first_serve_direction.{code}" for code in ("4", "5", "6")
    )
    assert P04_FEATURE_NAMES == tuple(
        f"p04.return_direction.{code}" for code in ("1", "2", "3")
    )
    assert P05_FEATURE_NAMES == tuple(
        f"p05.return_depth.{code}" for code in ("7", "8", "9")
    )
    assert P06_CODE_ORDER == MANUAL_SHOT_ORDER
    assert P06_FEATURE_NAMES == tuple(
        f"p06.return_shot_type.{code}" for code in MANUAL_SHOT_ORDER
    )
    assert P09_PROFILE_ORDER == MANUAL_PROFILE_ORDER
    assert P09_FEATURE_NAMES == tuple(
        f"p09.return_profile.{profile}" for profile in MANUAL_PROFILE_ORDER
    )
    assert len(P06_FEATURE_NAMES) == 17
    assert len(P09_FEATURE_NAMES) == 153
    assert MASK_FEATURE_NAMES == (
        "mask.p02.applicable",
        "mask.p02.observed",
        "mask.p04.applicable",
        "mask.p04.observed",
        "mask.p05.applicable",
        "mask.p05.observed",
        "mask.p06.applicable",
        "mask.p06.observed",
        "mask.p09.applicable",
        "mask.p09.observed",
    )
    assert CONTEXT_FEATURE_NAMES == (
        "context.serve_number.1",
        "context.serve_number.2",
        "context.previous_attempt_was_fault",
    )


def test_each_policy_uses_the_exact_tactical_catalogue():
    assert _schema(TacticalEncodingPolicy.COMPONENT_ONLY).tactical_feature_names == (
        P02_FEATURE_NAMES + P04_FEATURE_NAMES + P05_FEATURE_NAMES + P06_FEATURE_NAMES
    )
    assert _schema(TacticalEncodingPolicy.PROFILE_ONLY).tactical_feature_names == (
        P02_FEATURE_NAMES + P09_FEATURE_NAMES
    )
    assert _schema(TacticalEncodingPolicy.COMPONENTS_AND_PROFILE).tactical_feature_names == (
        P02_FEATURE_NAMES
        + P04_FEATURE_NAMES
        + P05_FEATURE_NAMES
        + P06_FEATURE_NAMES
        + P09_FEATURE_NAMES
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_schema_indices_are_unique_contiguous_and_deterministic(policy):
    first = _schema(policy)
    second = _schema(policy)
    assert first == second
    assert first.name_to_index == tuple(
        (name, index) for index, name in enumerate(first.feature_names)
    )
    assert len(set(first.feature_names)) == first.feature_count
    assert tuple(index for _, index in first.name_to_index) == tuple(
        range(first.feature_count)
    )
    validate_tactical_feature_schema(first)


@pytest.mark.parametrize("policy", POLICIES)
def test_unknown_zero_and_q_are_never_tactical_categories(policy):
    names = _schema(policy).tactical_feature_names
    assert not any(name.endswith(".0") or name.endswith(".q") for name in names)
    assert not any("unknown" in name for name in names)


def test_schema_fingerprints_are_deterministic_and_distinct_by_policy():
    fingerprints = [feature_schema_fingerprint(_schema(policy)) for policy in POLICIES]
    assert len(set(fingerprints)) == 3
    assert all(len(value) == 64 and value == value.upper() for value in fingerprints)
    assert fingerprints == [feature_schema_fingerprint(_schema(policy)) for policy in POLICIES]


@pytest.mark.parametrize("bad", ["component_only", None, 0, True, object()])
def test_schema_rejects_free_or_coercible_policy_values(bad):
    with pytest.raises(TypeError, match="TacticalEncodingPolicy"):
        build_tactical_feature_schema(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        {"contract_version": "9.9.9"},
        {"policy": "component_only"},
        {"feature_names": ()},
        {"tactical_feature_names": ()},
        {"mask_feature_names": ()},
        {"context_feature_names": ()},
        {"name_to_index": ()},
        {"feature_count": 38},
        {"feature_count": True},
        {"pattern_catalog": ("P02",)},
        {"encoded_pattern_order": ("P02",)},
        {"redundant_representation": True},
        {"contractual_purpose": "invented"},
    ],
)
def test_schema_validator_rejects_each_public_field_mutation(mutation):
    with pytest.raises((TypeError, TacticalFeatureContractError)):
        replace(_schema(), **mutation)


def test_schema_rejects_duplicate_reordered_and_mutable_names_or_mapping():
    schema = _schema()
    mutations = (
        {"feature_names": (schema.feature_names[0], *schema.feature_names[:-1])},
        {"feature_names": tuple(reversed(schema.feature_names))},
        {"feature_names": list(schema.feature_names)},
        {"name_to_index": dict(schema.name_to_index)},
        {"name_to_index": ((schema.feature_names[0], 0), (schema.feature_names[1], 0))},
    )
    for mutation in mutations:
        with pytest.raises(TacticalFeatureContractError):
            replace(schema, **mutation)


@pytest.mark.parametrize(
    ("code", "feature"),
    [("4", "p02.first_serve_direction.4"), ("5", "p02.first_serve_direction.5"), ("6", "p02.first_serve_direction.6")],
)
@pytest.mark.parametrize("policy", POLICIES)
def test_p02_observed_codes_activate_exactly_one_feature_in_every_policy(
    code, feature, policy
):
    vector = _vector(f"{code}f17", policy=policy)
    assert _value(vector, feature) == 1
    assert sum(_value(vector, name) for name in P02_FEATURE_NAMES) == 1
    assert _value(vector, "mask.p02.applicable") == 1
    assert _value(vector, "mask.p02.observed") == 1


@pytest.mark.parametrize("sequence", ["0f17", "S"])
def test_p02_unknown_or_censored_uses_masks_without_false_category(sequence):
    vector = _vector(sequence)
    assert sum(_value(vector, name) for name in P02_FEATURE_NAMES) == 0
    assert _value(vector, "mask.p02.applicable") == 1
    assert _value(vector, "mask.p02.observed") == 0


@pytest.mark.parametrize("previous_fault", [False, True])
def test_p02_is_not_applicable_and_never_active_on_second_serve(previous_fault):
    vector = _vector("4f17", 2, previous_fault)
    assert sum(_value(vector, name) for name in P02_FEATURE_NAMES) == 0
    assert _value(vector, "mask.p02.applicable") == 0
    assert _value(vector, "mask.p02.observed") == 0


@pytest.mark.parametrize("direction", ["1", "2", "3"])
def test_p04_observed_directions_activate_exact_component(direction):
    vector = _vector(f"4f{direction}7")
    assert _value(vector, f"p04.return_direction.{direction}") == 1
    assert sum(_value(vector, name) for name in P04_FEATURE_NAMES) == 1
    assert _value(vector, "mask.p04.observed") == 1


@pytest.mark.parametrize("sequence", ["4f07", "4f#", "4*"])
def test_p04_unknown_no_direction_and_censorship_never_become_categories(sequence):
    vector = _vector(sequence)
    assert sum(_value(vector, name) for name in P04_FEATURE_NAMES) == 0
    assert _value(vector, "mask.p04.observed") == 0


@pytest.mark.parametrize("depth", ["7", "8", "9"])
def test_p05_observed_depths_activate_exact_component(depth):
    vector = _vector(f"4f1{depth}")
    assert _value(vector, f"p05.return_depth.{depth}") == 1
    assert sum(_value(vector, name) for name in P05_FEATURE_NAMES) == 1
    assert _value(vector, "mask.p05.observed") == 1


@pytest.mark.parametrize("sequence", ["4f10", "4f1", "4q17", "4*"])
def test_p05_unknown_not_documented_unknown_initial_and_censored_use_no_category(sequence):
    vector = _vector(sequence)
    assert sum(_value(vector, name) for name in P05_FEATURE_NAMES) == 0
    assert _value(vector, "mask.p05.observed") == 0


@pytest.mark.parametrize("shot", MANUAL_SHOT_ORDER)
def test_all_17_p06_codes_activate_their_exact_feature(shot):
    vector = _vector(f"4{shot}17")
    assert _value(vector, f"p06.return_shot_type.{shot}") == 1
    assert sum(_value(vector, name) for name in P06_FEATURE_NAMES) == 1
    assert _value(vector, "mask.p06.observed") == 1


@pytest.mark.parametrize("sequence", ["4q17", "4?f17", "4*"])
def test_p06_q_unknown_and_censored_never_become_categories(sequence):
    vector = _vector(sequence)
    assert sum(_value(vector, name) for name in P06_FEATURE_NAMES) == 0
    assert _value(vector, "mask.p06.observed") == 0


@pytest.mark.parametrize("profile", MANUAL_PROFILE_ORDER)
def test_all_153_p09_profiles_activate_exact_profile_feature(profile):
    shot, direction, depth = profile.split("|")
    vector = _vector(
        f"4{shot}{direction}{depth}",
        policy=TacticalEncodingPolicy.PROFILE_ONLY,
    )
    assert _value(vector, f"p09.return_profile.{profile}") == 1
    assert sum(_value(vector, name) for name in P09_FEATURE_NAMES) == 1
    assert _value(vector, "mask.p09.observed") == 1


@pytest.mark.parametrize("sequence", ["4q17", "4f07", "4f10", "4f1", "4*"])
def test_p09_unknown_not_documented_and_censored_use_no_profile_category(sequence):
    vector = _vector(sequence, policy=TacticalEncodingPolicy.PROFILE_ONLY)
    assert sum(_value(vector, name) for name in P09_FEATURE_NAMES) == 0
    assert _value(vector, "mask.p09.observed") == 0


def test_policy_anti_double_counting_is_exact_for_complete_profile():
    component = _vector("4f17", policy=TacticalEncodingPolicy.COMPONENT_ONLY)
    profile = _vector("4f17", policy=TacticalEncodingPolicy.PROFILE_ONLY)
    both = _vector("4f17", policy=TacticalEncodingPolicy.COMPONENTS_AND_PROFILE)
    assert dict(component.diagnostics)["active_tactical_feature_count"] == 4
    assert dict(profile.diagnostics)["active_tactical_feature_count"] == 2
    assert dict(both.diagnostics)["active_tactical_feature_count"] == 5
    assert not any(name.startswith("p09.") for name in component.active_feature_names)
    assert not any(
        name.startswith(("p04.", "p05.", "p06."))
        for name in profile.active_feature_names
    )
    assert "p09.return_profile.f|1|7" in both.active_feature_names
    assert dict(component.diagnostics)["policy_excluded_observed_signal_count"] == 1
    assert dict(profile.diagnostics)["policy_excluded_observed_signal_count"] == 3
    assert dict(both.diagnostics)["redundant_representation"] == 1


def test_masks_are_identical_across_policies_for_same_extraction():
    vectors = [_vector("4f17", policy=policy) for policy in POLICIES]
    for mask in MASK_FEATURE_NAMES:
        assert {_value(vector, mask) for vector in vectors} == {_value(vectors[0], mask)}


def test_observed_pattern_excluded_by_policy_keeps_observed_mask_only():
    component = _vector("4f17", policy=TacticalEncodingPolicy.COMPONENT_ONLY)
    profile = _vector("4f17", policy=TacticalEncodingPolicy.PROFILE_ONLY)
    assert _value(component, "mask.p09.observed") == 1
    assert not any(name.startswith("p09.") for name in component.active_feature_names)
    for pattern in ("p04", "p05", "p06"):
        assert _value(profile, f"mask.{pattern}.observed") == 1
        assert not any(name.startswith(f"{pattern}.") for name in profile.active_feature_names)


def test_only_policy_excluded_observations_can_yield_no_tactical_feature():
    vector = _vector(
        "4f1",
        serve_number=2,
        policy=TacticalEncodingPolicy.PROFILE_ONLY,
    )
    assert vector.observed_signal_count == 2
    assert dict(vector.diagnostics)["policy_excluded_observed_signal_count"] == 2
    assert dict(vector.diagnostics)["active_tactical_feature_count"] == 0
    assert vector.availability_state is TacticalFeatureAvailability.NOT_OBSERVED


@pytest.mark.parametrize(
    ("serve_number", "previous_fault", "expected"),
    [(1, False, (1, 0, 0)), (2, False, (0, 1, 0)), (2, True, (0, 1, 1))],
)
def test_context_features_are_exact(serve_number, previous_fault, expected):
    vector = _vector("4f17", serve_number, previous_fault)
    actual = tuple(_value(vector, name) for name in CONTEXT_FEATURE_NAMES)
    assert actual == expected
    assert actual[0] + actual[1] == 1


def test_all_patterns_abstained_has_no_tactical_feature_observed():
    vector = _vector("S")
    schema = _schema()
    assert sum(_value(vector, name) for name in schema.tactical_feature_names) == 0
    assert vector.observed_signal_count == 0
    assert vector.applicable_pattern_count == 5
    assert vector.abstention_count == 5
    assert vector.availability_state is TacticalFeatureAvailability.NOT_OBSERVED


def test_mixed_signals_and_abstentions_reconcile_counts():
    vector = _vector("4f1")
    assert vector.observed_signal_count == 3
    assert vector.applicable_pattern_count == 5
    assert vector.abstention_count == 2
    assert dict(vector.diagnostics)["active_tactical_feature_count"] == 3
    assert vector.availability_state is TacticalFeatureAvailability.AVAILABLE


def test_encoder_accepts_only_validated_extraction_and_exact_schema(monkeypatch):
    extraction = _extraction()
    schema = _schema()
    calls = []
    original = encoder.validate_attempt_signal_extraction

    def spy(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(encoder, "validate_attempt_signal_extraction", spy)
    encode_tactical_attempt(extraction, schema)
    assert calls == [extraction]
    with pytest.raises(TypeError, match="AttemptSignalExtraction"):
        encode_tactical_attempt({}, schema)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TacticalFeatureSchema"):
        encode_tactical_attempt(extraction, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AttemptSignalExtraction"):
        encode_tactical_attempt(extraction.bundle, schema)  # type: ignore[arg-type]


def test_complete_redundant_vector_reconciles_p09_with_components():
    vector = _vector("6b39", policy=TacticalEncodingPolicy.COMPONENTS_AND_PROFILE)
    for name in (
        "p04.return_direction.3",
        "p05.return_depth.9",
        "p06.return_shot_type.b",
        "p09.return_profile.b|3|9",
    ):
        assert _value(vector, name) == 1
    validate_tactical_feature_vector(vector)


def test_redundant_vector_rejects_profile_component_contradiction():
    vector = _vector("4f17", policy=TacticalEncodingPolicy.COMPONENTS_AND_PROFILE)
    values = _changed_values(
        vector,
        {"p04.return_direction.1": 0, "p04.return_direction.2": 1},
    )
    active_names = tuple(
        name
        for name, value in zip(_schema(vector.policy).feature_names, values)
        if value
    )
    with pytest.raises(TacticalFeatureContractError, match="P09"):
        replace(vector, values=values, active_feature_names=active_names)


@pytest.mark.parametrize(
    "mutation",
    [
        {"contract_version": "9.9.9"},
        {"schema_fingerprint": "A" * 64},
        {"schema_fingerprint": None},
        {"policy": "component_only"},
        {"values": ()},
        {"serve_number": 2},
        {"active_feature_count": 0},
        {"active_feature_names": ()},
        {"observed_signal_count": 0},
        {"applicable_pattern_count": 4},
        {"abstention_count": 1},
        {"availability_state": TacticalFeatureAvailability.NOT_OBSERVED},
        {"diagnostics": ()},
        {"diagnostics": []},
    ],
)
def test_vector_validator_rejects_each_public_field_mutation(mutation):
    with pytest.raises((TypeError, TacticalFeatureContractError)):
        replace(_vector(), **mutation)


@pytest.mark.parametrize("bad_value", [True, False, 2, -1, 1.0, "1", None, float("nan"), float("inf")])
def test_vector_rejects_bool_coercions_nonbinary_and_nonfinite_values(bad_value):
    vector = _vector()
    values = (bad_value, *vector.values[1:])
    with pytest.raises(TacticalFeatureContractError, match="0/1"):
        replace(vector, values=values)


def test_vector_rejects_invalid_one_hot_masks_and_multiple_categories():
    vector = _vector()
    mutations = (
        {
            "context.serve_number.1": 0,
            "context.serve_number.2": 0,
        },
        {"mask.p02.applicable": 0},
        {"mask.p02.observed": 0},
        {"p02.first_serve_direction.5": 1},
    )
    for updates in mutations:
        with pytest.raises(TacticalFeatureContractError):
            replace(vector, values=_changed_values(vector, updates))


def test_vector_rejects_previous_fault_on_first_serve():
    vector = _vector()
    with pytest.raises(TacticalFeatureContractError, match="previous_attempt"):
        replace(
            vector,
            values=_changed_values(vector, {"context.previous_attempt_was_fault": 1}),
        )


def test_single_and_multiple_row_batches_preserve_exact_order():
    schema = _schema()
    first = _extraction("4f17")
    second = _extraction("5b39")
    single = _batch((first,), schema)
    multiple = _batch((first, second), schema)
    assert single.row_count == 1 and single.column_count == 39
    assert multiple.row_count == 2 and multiple.column_count == 39
    assert multiple.vectors == (
        encode_tactical_attempt(first, schema),
        encode_tactical_attempt(second, schema),
    )
    validate_tactical_feature_batch(single)
    validate_tactical_feature_batch(multiple)


def test_batch_allows_repeated_rows_without_deduplication():
    extraction = _extraction()
    batch = _batch((extraction, extraction), _schema())
    assert batch.row_count == 2
    assert batch.vectors[0] == batch.vectors[1]


def test_batch_rejects_empty_list_and_non_tuple_inputs():
    with pytest.raises(TacticalFeatureContractError, match="vacio"):
        encode_tactical_batch((), _schema())
    with pytest.raises(TypeError, match="tuple"):
        encode_tactical_batch([_vector()], _schema())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TacticalFeatureVector"):
        encode_tactical_batch((_extraction(),), _schema())  # type: ignore[arg-type]


def test_batch_rejects_mixed_schema_and_mixed_policy_rows():
    component = _vector(policy=TacticalEncodingPolicy.COMPONENT_ONLY)
    profile = _vector(policy=TacticalEncodingPolicy.PROFILE_ONLY)
    schema = _schema()
    diagnostics = (
        ("row_count", 2),
        ("column_count", 39),
        ("total_active_feature_count", component.active_feature_count + profile.active_feature_count),
        ("total_observed_signal_count", component.observed_signal_count + profile.observed_signal_count),
        ("redundant_representation", 0),
    )
    with pytest.raises(TacticalFeatureContractError, match="schema y policy"):
        TacticalFeatureBatch(
            FEATURE_CONTRACT_VERSION,
            feature_schema_fingerprint(schema),
            TacticalEncodingPolicy.COMPONENT_ONLY,
            (component, profile),
            2,
            39,
            diagnostics,
        )


def test_batch_rejects_a_vector_manipulated_after_construction():
    vector = _vector()
    object.__setattr__(vector, "observed_signal_count", 0)
    with pytest.raises(TacticalFeatureContractError, match="observed_signal_count"):
        TacticalFeatureBatch(
            FEATURE_CONTRACT_VERSION,
            feature_schema_fingerprint(_schema()),
            TacticalEncodingPolicy.COMPONENT_ONLY,
            (vector,),
            1,
            39,
            (
                ("row_count", 1),
                ("column_count", 39),
                ("total_active_feature_count", vector.active_feature_count),
                ("total_observed_signal_count", 0),
                ("redundant_representation", 0),
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"contract_version": "9.9.9"},
        {"schema_fingerprint": "A" * 64},
        {"policy": "component_only"},
        {"vectors": ()},
        {"vectors": []},
        {"row_count": 2},
        {"row_count": True},
        {"column_count": 40},
        {"diagnostics": ()},
        {"diagnostics": []},
    ],
)
def test_batch_validator_rejects_each_public_field_mutation(mutation):
    batch = encode_tactical_batch((_vector(),), _schema())
    with pytest.raises((TypeError, TacticalFeatureContractError)):
        replace(batch, **mutation)


def test_batch_order_is_semantic_and_changes_fingerprint():
    schema = _schema()
    first = _extraction("4f17")
    second = _extraction("5b39")
    first_vector = encode_tactical_attempt(first, schema)
    second_vector = encode_tactical_attempt(second, schema)
    forward = encode_tactical_batch((first_vector, second_vector), schema)
    reverse = encode_tactical_batch((second_vector, first_vector), schema)
    assert forward.vectors == tuple(reversed(reverse.vectors))
    assert feature_batch_fingerprint(forward) != feature_batch_fingerprint(reverse)


def test_all_public_dataclasses_are_frozen_and_deep_collections_are_tuples():
    schema = _schema()
    vector = _vector()
    batch = encode_tactical_batch((vector,), schema)
    for instance, name, value in (
        (schema, "feature_count", 0),
        (vector, "serve_number", 2),
        (batch, "row_count", 0),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(instance, name, value)
    assert all(
        type(value) is tuple
        for value in (
            schema.feature_names,
            schema.name_to_index,
            vector.values,
            vector.active_feature_names,
            vector.diagnostics,
            batch.vectors,
            batch.diagnostics,
        )
    )


def test_public_dataclass_schemas_are_closed():
    assert tuple(item.name for item in fields(TacticalFeatureSchema)) == (
        "contract_version",
        "policy",
        "feature_names",
        "tactical_feature_names",
        "mask_feature_names",
        "context_feature_names",
        "name_to_index",
        "feature_count",
        "pattern_catalog",
        "encoded_pattern_order",
        "redundant_representation",
        "contractual_purpose",
    )
    assert tuple(item.name for item in fields(TacticalFeatureVector)) == (
        "contract_version",
        "schema_fingerprint",
        "policy",
        "values",
        "serve_number",
        "active_feature_count",
        "active_feature_names",
        "observed_signal_count",
        "applicable_pattern_count",
        "abstention_count",
        "availability_state",
        "diagnostics",
    )
    assert tuple(item.name for item in fields(TacticalFeatureBatch)) == (
        "contract_version",
        "schema_fingerprint",
        "policy",
        "vectors",
        "row_count",
        "column_count",
        "diagnostics",
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_canonical_serializers_are_compact_sorted_utf8_and_deterministic(policy):
    schema = _schema(policy)
    vector = _vector(policy=policy)
    batch = encode_tactical_batch((vector,), schema)
    for payload in (
        canonical_feature_schema_json(schema),
        canonical_feature_vector_json(vector),
        canonical_feature_batch_json(batch),
    ):
        structure = json.loads(payload)
        assert payload == json.dumps(
            structure,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert b": " not in payload and b", " not in payload


def test_fingerprint_domains_and_hashes_are_exact_and_separate():
    schema = _schema()
    vector = _vector()
    batch = encode_tactical_batch((vector,), schema)
    payloads = (
        (SCHEMA_FINGERPRINT_DOMAIN, canonical_feature_schema_json(schema), feature_schema_fingerprint(schema)),
        (VECTOR_FINGERPRINT_DOMAIN, canonical_feature_vector_json(vector), feature_vector_fingerprint(vector)),
        (BATCH_FINGERPRINT_DOMAIN, canonical_feature_batch_json(batch), feature_batch_fingerprint(batch)),
    )
    assert len({domain for domain, _, _ in payloads}) == 3
    assert len({fingerprint for _, _, fingerprint in payloads}) == 3
    for domain, payload, fingerprint in payloads:
        assert fingerprint == sha256(domain + payload).hexdigest().upper()


def test_semantic_vector_and_batch_changes_change_fingerprints():
    wide = _vector("4f17")
    body = _vector("5f17")
    assert feature_vector_fingerprint(wide) != feature_vector_fingerprint(body)
    schema = _schema()
    assert feature_batch_fingerprint(
        encode_tactical_batch((_vector("4f17"),), schema)
    ) != feature_batch_fingerprint(
        encode_tactical_batch((_vector("5f17"),), schema)
    )


def test_serializations_contain_no_raw_or_individual_information():
    raw = "cc6+b39?"
    schema = _schema(TacticalEncodingPolicy.COMPONENTS_AND_PROFILE)
    vector = encode_tactical_attempt(_extraction(raw), schema)
    batch = encode_tactical_batch((vector,), schema)
    payload = b"".join(
        (
            canonical_feature_schema_json(schema),
            canonical_feature_vector_json(vector),
            canonical_feature_batch_json(batch),
        )
    ).decode("utf-8").lower()
    assert raw.lower() not in payload
    for forbidden in (
        "sequence_text",
        "raw_sequence",
        "parse_result",
        '"tokens"',
        '"spans"',
        '"warnings"',
        "residual_text",
        "match_id",
        "point_number",
        '"server":',
        "point_winner",
        "returner_won_point",
        '"date"',
        "timestamp",
        '"path"',
        "test_",
        "evaluation",
        "scoring",
        "recommendation",
        "c:\\",
        "file://",
        "../",
    ):
        assert forbidden not in payload


@pytest.mark.parametrize(
    "unsafe",
    [
        {"match_id": "m1"},
        {"test_rows": 1},
        {"nested": {"point_number": 1}},
        {"nested": [{"safe": "C:\\private\\x"}]},
        {"nested": [{"safe": "/private/x"}]},
        {"nested": [{"safe": "\\\\host\\share"}]},
        {"nested": [{"safe": "file://private"}]},
        {"nested": [{"safe": "../private"}]},
        {"nested": float("nan")},
        {"nested": float("inf")},
        {"nested": object()},
        {1: "non-string-key"},
    ],
)
def test_recursive_security_rejects_sensitive_mutable_payload_content(unsafe):
    with pytest.raises(TacticalFeatureContractError):
        encoder._validate_public_tree(unsafe)


def test_schema_vector_and_batch_validators_reject_wrong_object_types():
    for validator, value in (
        (validate_tactical_feature_schema, {}),
        (validate_tactical_feature_vector, ()),
        (validate_tactical_feature_batch, object()),
    ):
        with pytest.raises(TypeError):
            validator(value)  # type: ignore[arg-type]


def test_product_module_has_no_parser_extractors_adapters_io_or_numeric_frameworks():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_import_roots = {
        "pandas",
        "numpy",
        "sklearn",
        "pyarrow",
        "polars",
        "argparse",
        "click",
        "typer",
    }
    forbidden_calls = {
        "open",
        "read_csv",
        "read_parquet",
        "to_csv",
        "to_json",
        "write_text",
        "write_bytes",
        "parse_sequence",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_import_roots for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_import_roots
            assert not node.module.startswith("src.analysis")
            assert "explainable_direction_recommender" not in node.module
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            assert name not in forbidden_calls
            assert not name.startswith(("classify_", "adapt_", "parse_and_classify_"))
    lowered = source.lower()
    assert "data/" not in lowered
    assert "reports/" not in lowered
    assert "__main__" not in lowered


def test_product_module_contains_no_assert_and_no_new_dependency_boundary():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports <= {"__future__", "dataclasses", "enum", "hashlib", "json", "re", "typing", "src"}
