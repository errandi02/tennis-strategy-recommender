"""Pruebas sinteticas y adversariales del orquestador tactico por intento."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pytest

import src.recommender.tactical_signal_orchestrator as orchestrator
from src.recommender.tactical_signal_contract import (
    ELIGIBLE_PATTERN_ORDER,
    tactical_signal_bundle_fingerprint,
)
from src.recommender.tactical_signal_orchestrator import (
    ATTEMPT_FINGERPRINT_DOMAIN,
    ORCHESTRATOR_CONTRACT_VERSION,
    SECOND_SERVE_NON_APPLICABLE_REASON,
    AttemptExtractionState,
    AttemptSignalExtraction,
    AttemptSignalRequest,
    NonApplicablePattern,
    TacticalSignalOrchestrationError,
    attempt_extraction_fingerprint,
    canonical_attempt_extraction_json,
    extract_tactical_signals_for_attempt,
    validate_attempt_signal_extraction,
)


MODULE_PATH = Path(orchestrator.__file__)
ALL_PATTERNS = ("P02", "P04", "P05", "P06", "P09")
RETURN_PATTERNS = ("P04", "P05", "P06", "P09")


def _request(
    sequence: str = "4f17",
    serve_number: int = 1,
    previous_fault: bool = False,
) -> AttemptSignalRequest:
    return AttemptSignalRequest(
        contract_version=ORCHESTRATOR_CONTRACT_VERSION,
        sequence_text=sequence,
        serve_number=serve_number,
        previous_attempt_was_fault=previous_fault,
    )


def _extract(
    sequence: str = "4f17",
    serve_number: int = 1,
    previous_fault: bool = False,
) -> AttemptSignalExtraction:
    return extract_tactical_signals_for_attempt(
        _request(sequence, serve_number, previous_fault)
    )


def _by_pattern(extraction: AttemptSignalExtraction):
    return {item.requested_pattern_id: item for item in extraction.adaptations}


def test_first_serve_complete_profile_produces_all_five_signals():
    result = _extract("4f17")
    by_pattern = _by_pattern(result)
    assert result.requested_patterns == ALL_PATTERNS == ELIGIBLE_PATTERN_ORDER
    assert result.applicable_patterns == ALL_PATTERNS
    assert result.non_applicable_patterns == ()
    assert tuple(by_pattern) == ALL_PATTERNS
    assert tuple(signal.pattern_id for signal in result.bundle.signals) == ALL_PATTERNS
    assert result.bundle.abstentions == ()
    assert result.active_signal_count == 5
    assert result.abstention_count == 0
    assert result.extraction_state is AttemptExtractionState.SIGNALS_AVAILABLE
    assert result.reason_codes == ("signals_available",)
    assert by_pattern["P02"].signal.tactical_value == "wide"
    assert by_pattern["P04"].signal.tactical_value == "1"
    assert by_pattern["P05"].signal.tactical_value == "7"
    assert by_pattern["P06"].signal.tactical_value == "f"
    assert by_pattern["P09"].signal.tactical_value == "f|1|7"


@pytest.mark.parametrize("previous_fault", [False, True])
def test_second_serve_complete_profile_has_four_signals_and_explicit_p02_non_applicability(
    previous_fault,
):
    result = _extract("4f17", 2, previous_fault)
    assert result.requested_patterns == ALL_PATTERNS
    assert result.applicable_patterns == RETURN_PATTERNS
    assert result.non_applicable_patterns == (
        NonApplicablePattern("P02", SECOND_SERVE_NON_APPLICABLE_REASON),
    )
    assert tuple(item.requested_pattern_id for item in result.adaptations) == RETURN_PATTERNS
    assert tuple(signal.pattern_id for signal in result.bundle.signals) == RETURN_PATTERNS
    assert result.active_signal_count == 4
    assert result.abstention_count == 0
    assert result.reason_codes == (
        "signals_available",
        SECOND_SERVE_NON_APPLICABLE_REASON,
    )
    assert result.bundle.context.serve_number == 2


@pytest.mark.parametrize(
    ("code", "expected_value", "active"),
    [("4", "wide", True), ("5", "body", True), ("6", "T", True), ("0", None, False)],
)
def test_p02_direction_domain_is_preserved(code, expected_value, active):
    result = _by_pattern(_extract(f"{code}f17"))["P02"]
    assert (result.signal is not None) is active
    assert (None if result.signal is None else result.signal.tactical_value) == expected_value
    if code == "0":
        assert result.abstained is True
        assert result.upstream_state == "first_serve_direction_unknown"
        assert result.context.serve_direction == "unknown"


@pytest.mark.parametrize(
    ("direction", "expected_value", "active"),
    [("1", "1", True), ("2", "2", True), ("3", "3", True), ("0", None, False)],
)
def test_p04_lateral_direction_domain_is_preserved(direction, expected_value, active):
    result = _by_pattern(_extract(f"4f{direction}7"))["P04"]
    assert (result.signal is not None) is active
    assert (None if result.signal is None else result.signal.tactical_value) == expected_value


@pytest.mark.parametrize(
    ("suffix", "expected_value", "condition"),
    [
        ("7", "7", None),
        ("8", "8", None),
        ("9", "9", None),
        ("0", None, "unknown"),
        ("", None, "not_documented"),
    ],
)
def test_p05_depth_domain_unknown_and_not_documented(suffix, expected_value, condition):
    result = _by_pattern(_extract(f"4f1{suffix}"))["P05"]
    assert (None if result.signal is None else result.signal.tactical_value) == expected_value
    assert result.abstention_condition == condition


@pytest.mark.parametrize("shot", tuple("fbrsvzopuylmhijkt"))
def test_all_17_p06_documented_codes_are_active(shot):
    result = _by_pattern(_extract(f"4{shot}17"))["P06"]
    assert result.signal is not None
    assert result.signal.tactical_value == shot


def test_p06_q_is_unknown_and_never_active():
    result = _by_pattern(_extract("4q17"))["P06"]
    assert result.signal is None
    assert result.abstention_condition == "unknown"
    assert result.upstream_state == "return_shot_type_unknown"


@pytest.mark.parametrize(
    ("sequence", "state", "condition"),
    [
        ("4f17", "documented_initial_return_profile", None),
        ("4q17", "initial_return_profile_unknown", "unknown"),
        ("4f07", "initial_return_profile_unknown", "unknown"),
        ("4f10", "initial_return_profile_unknown", "unknown"),
        ("4f1", "initial_return_profile_not_documented", "not_documented"),
    ],
)
def test_p09_complete_unknown_and_not_documented_profiles(sequence, state, condition):
    result = _by_pattern(_extract(sequence))["P09"]
    assert result.upstream_state == state
    assert result.abstention_condition == condition
    assert (result.signal is not None) is (condition is None)


@pytest.mark.parametrize(
    ("sequence", "serve_number", "previous_fault"),
    [
        ("4*", 1, False),
        ("4#", 1, False),
        ("4n", 1, False),
        ("4n", 2, True),
        ("S", 1, False),
        ("R", 1, False),
        ("P", 1, False),
        ("Q", 1, False),
        ("c", 1, False),
    ],
)
def test_terminals_specials_double_fault_and_incomplete_let_never_create_return_signals(
    sequence, serve_number, previous_fault
):
    result = _extract(sequence, serve_number, previous_fault)
    for pattern_id in RETURN_PATTERNS:
        assert _by_pattern(result)[pattern_id].signal is None


def test_fault_keeps_first_serve_direction_but_abstains_all_return_patterns():
    result = _extract("5n")
    assert tuple(signal.pattern_id for signal in result.bundle.signals) == ("P02",)
    assert tuple(item.requested_pattern_id for item in result.bundle.abstentions) == RETURN_PATTERNS
    assert result.active_signal_count == 1
    assert result.abstention_count == 4


def test_partial_return_bundle_keeps_signals_and_explicit_abstentions():
    result = _extract("4f1")
    assert tuple(signal.pattern_id for signal in result.bundle.signals) == (
        "P02",
        "P04",
        "P06",
    )
    assert tuple(item.requested_pattern_id for item in result.bundle.abstentions) == (
        "P05",
        "P09",
    )
    assert result.active_signal_count == 3
    assert result.abstention_count == 2


def test_all_applicable_patterns_can_abstain_with_reachable_global_state():
    result = _extract("S")
    assert result.bundle.signals == ()
    assert len(result.bundle.abstentions) == 5
    assert result.active_signal_count == 0
    assert result.abstention_count == 5
    assert result.extraction_state is AttemptExtractionState.ALL_APPLICABLE_PATTERNS_ABSTAINED
    assert result.reason_codes == ("all_applicable_patterns_abstained",)


def test_p09_reconciles_exactly_with_p04_p05_and_p06():
    by_pattern = _by_pattern(_extract("6b39"))
    assert by_pattern["P04"].signal.tactical_value == "3"
    assert by_pattern["P05"].signal.tactical_value == "9"
    assert by_pattern["P06"].signal.tactical_value == "b"
    assert by_pattern["P09"].signal.tactical_value == "b|3|9"
    assert by_pattern["P09"].signal.components == (
        ("return_shot_type", "b"),
        ("return_lateral_direction", "3"),
        ("return_depth", "9"),
    )


@pytest.mark.parametrize("bad_text", [None, 4, True, b"4f17", object()])
def test_request_rejects_non_string_sequence_before_parser(monkeypatch, bad_text):
    calls = []
    monkeypatch.setattr(orchestrator, "parse_sequence", lambda *args: calls.append(args))
    with pytest.raises(TypeError, match="sequence_text"):
        AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION, bad_text, 1, False  # type: ignore[arg-type]
        )
    assert calls == []


@pytest.mark.parametrize("bad_serve", [True, False, 1.0, 2.0, "1", "2", None, 0, 3])
def test_request_rejects_non_exact_serve_number_before_parser(monkeypatch, bad_serve):
    calls = []
    monkeypatch.setattr(orchestrator, "parse_sequence", lambda *args: calls.append(args))
    with pytest.raises(TypeError, match="serve_number"):
        AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION, "4f17", bad_serve, False  # type: ignore[arg-type]
        )
    assert calls == []


@pytest.mark.parametrize("bad_context", [0, 1, "false", None, 0.0, object()])
def test_request_rejects_non_exact_fault_context_before_parser(monkeypatch, bad_context):
    calls = []
    monkeypatch.setattr(orchestrator, "parse_sequence", lambda *args: calls.append(args))
    with pytest.raises(TypeError, match="previous_attempt_was_fault"):
        AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION, "4f17", 1, bad_context  # type: ignore[arg-type]
        )
    assert calls == []


def test_first_serve_previous_fault_is_rejected_before_parser(monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator, "parse_sequence", lambda *args: calls.append(args))
    with pytest.raises(TacticalSignalOrchestrationError, match="primer saque"):
        _request("4f17", 1, True)
    assert calls == []


def test_wrong_request_type_is_rejected_before_parser(monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator, "parse_sequence", lambda *args: calls.append(args))
    with pytest.raises(TypeError, match="AttemptSignalRequest"):
        extract_tactical_signals_for_attempt({"sequence_text": "4f17"})  # type: ignore[arg-type]
    assert calls == []


def test_parser_is_called_once_and_same_parse_object_reaches_every_low_level_stage(
    monkeypatch,
):
    parser_calls = []
    classifier_parse_ids = []
    adapter_parse_ids = []
    real_parser = orchestrator.parse_sequence

    def counted_parser(*args):
        parsed = real_parser(*args)
        parser_calls.append(parsed)
        return parsed

    monkeypatch.setattr(orchestrator, "parse_sequence", counted_parser)
    for name in (
        "classify_first_serve_direction",
        "classify_initial_return_direction",
        "classify_initial_return_depth",
        "classify_initial_return_shot_type",
        "classify_initial_return_profile",
    ):
        original = getattr(orchestrator, name)

        def wrapper(*args, __original=original, **kwargs):
            classifier_parse_ids.append(id(args[2]))
            return __original(*args, **kwargs)

        monkeypatch.setattr(orchestrator, name, wrapper)
    for name in (
        "adapt_return_direction_signal",
        "adapt_return_depth_signal",
        "adapt_return_shot_type_signal",
        "adapt_return_profile_signal",
    ):
        original = getattr(orchestrator, name)

        def wrapper(*args, __original=original, **kwargs):
            adapter_parse_ids.append(id(kwargs["parsed"]))
            return __original(*args, **kwargs)

        monkeypatch.setattr(orchestrator, name, wrapper)

    result = _extract("4f17")
    assert result.active_signal_count == 5
    assert len(parser_calls) == 1
    expected_id = id(parser_calls[0])
    assert classifier_parse_ids == [expected_id] * 5
    assert adapter_parse_ids == [expected_id] * 4


def test_second_serve_uses_one_parse_for_exactly_four_low_level_classifiers(monkeypatch):
    real_parser = orchestrator.parse_sequence
    calls = []

    def counted(*args):
        calls.append(args)
        return real_parser(*args)

    monkeypatch.setattr(orchestrator, "parse_sequence", counted)
    monkeypatch.setattr(
        orchestrator,
        "classify_first_serve_direction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("P02 no debe clasificarse en segundo saque")
        ),
    )
    result = _extract("4f17", 2, True)
    assert len(calls) == 1
    assert tuple(item.requested_pattern_id for item in result.adaptations) == RETURN_PATTERNS


def test_contract_failure_is_sanitized_and_never_builds_a_partial_bundle(monkeypatch):
    bundle_calls = []

    def broken(*args, **kwargs):
        raise ValueError("secuencia-secreta-4f17")

    monkeypatch.setattr(orchestrator, "classify_initial_return_depth", broken)
    monkeypatch.setattr(
        orchestrator,
        "build_tactical_signal_bundle",
        lambda *args: bundle_calls.append(args),
    )
    with pytest.raises(TacticalSignalOrchestrationError) as captured:
        _extract("4f17")
    assert "P05.classifier" in str(captured.value)
    assert "4f17" not in str(captured.value)
    assert "secuencia-secreta" not in str(captured.value)
    assert bundle_calls == []


def test_unexpected_exception_is_not_captured_as_abstention(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "classify_initial_return_depth",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("programmer bug")),
    )
    with pytest.raises(RuntimeError, match="programmer bug"):
        _extract("4f17")


@pytest.mark.parametrize(
    "mutation",
    [
        {"contract_version": "9.9.9"},
        {"serve_number": 2},
        {"previous_attempt_was_fault": True},
        {"requested_patterns": ("P04", "P05", "P06", "P09")},
        {"requested_patterns": ("P02", "P03", "P04", "P05", "P06")},
        {"applicable_patterns": ("P02", "P04", "P05", "P06")},
        {
            "non_applicable_patterns": (
                NonApplicablePattern("P02", SECOND_SERVE_NON_APPLICABLE_REASON),
            )
        },
        {"active_signal_count": 4},
        {"abstention_count": 1},
        {"extraction_state": AttemptExtractionState.ALL_APPLICABLE_PATTERNS_ABSTAINED},
        {"reason_codes": ()},
        {"reason_codes": ("signals_available", "signals_available")},
        {"reason_codes": ("invented",)},
        {"diagnostics": (("active_signal_count", 5),)},
        {"diagnostics": [["requested_pattern_count", 5]]},
    ],
)
def test_validator_rejects_each_manipulated_public_scalar_or_collection(mutation):
    with pytest.raises((TypeError, TacticalSignalOrchestrationError)):
        replace(_extract("4f17"), **mutation)


def test_validator_rejects_reordered_adaptations():
    canonical = _extract("4f17")
    reordered = (canonical.adaptations[1], canonical.adaptations[0], *canonical.adaptations[2:])
    with pytest.raises(TacticalSignalOrchestrationError, match="fuera de orden"):
        replace(canonical, adaptations=reordered)


def test_validator_rejects_custom_objects_in_adaptations_explicitly():
    canonical = _extract("4f17")
    with pytest.raises(TypeError, match="SignalAdaptationResult exacto"):
        replace(canonical, adaptations=(object(), *canonical.adaptations[1:]))


def test_validator_rejects_result_and_bundle_from_different_attempts():
    first = _extract("4f17")
    other = _extract("5b39")
    with pytest.raises(TacticalSignalOrchestrationError, match="bundle"):
        replace(first, bundle=other.bundle)


def test_validator_rejects_missing_p02_on_first_serve():
    canonical = _extract("4f17")
    with pytest.raises(TacticalSignalOrchestrationError, match="Adaptaciones"):
        replace(canonical, adaptations=canonical.adaptations[1:])


def test_validator_rejects_illegal_p02_on_second_serve():
    second = _extract("4f17", 2, True)
    first_p02 = _extract("4f17").adaptations[0]
    with pytest.raises(TacticalSignalOrchestrationError):
        replace(second, adaptations=(first_p02, *second.adaptations))


@pytest.mark.parametrize("previous_fault", [False, True])
def test_second_serve_fault_context_cannot_be_changed_after_classification(previous_fault):
    canonical = _extract("4n", 2, previous_fault)
    with pytest.raises(TacticalSignalOrchestrationError, match="falta previa"):
        replace(canonical, previous_attempt_was_fault=not previous_fault)


def test_global_reason_order_is_canonical_for_second_serve():
    result = _extract("4f17", 2, True)
    with pytest.raises(TacticalSignalOrchestrationError, match="Reason codes"):
        replace(result, reason_codes=tuple(reversed(result.reason_codes)))


@pytest.mark.parametrize("pattern_id", ["P03", "P07", "P08"])
def test_excluded_patterns_cannot_be_injected(pattern_id):
    canonical = _extract("4f17")
    requested = (*canonical.requested_patterns[:-1], pattern_id)
    with pytest.raises(TacticalSignalOrchestrationError, match="solicitados"):
        replace(canonical, requested_patterns=requested)


def test_request_extraction_and_non_applicability_are_deeply_immutable():
    request = _request("4f17")
    result = _extract("4f17", 2, True)
    with pytest.raises(FrozenInstanceError):
        request.serve_number = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.active_signal_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.non_applicable_patterns[0].reason_code = "changed"  # type: ignore[misc]
    assert type(result.requested_patterns) is tuple
    assert type(result.applicable_patterns) is tuple
    assert type(result.adaptations) is tuple
    assert type(result.diagnostics) is tuple
    assert all(type(item) is tuple for item in result.diagnostics)


def test_request_and_output_schemas_are_closed_and_minimal():
    assert tuple(field.name for field in fields(AttemptSignalRequest)) == (
        "contract_version",
        "sequence_text",
        "serve_number",
        "previous_attempt_was_fault",
    )
    assert tuple(field.name for field in fields(AttemptSignalExtraction)) == (
        "contract_version",
        "serve_number",
        "previous_attempt_was_fault",
        "requested_patterns",
        "applicable_patterns",
        "non_applicable_patterns",
        "adaptations",
        "bundle",
        "active_signal_count",
        "abstention_count",
        "extraction_state",
        "reason_codes",
        "diagnostics",
    )


def test_extraction_and_json_are_deterministic():
    first = _extract("cc6+b39")
    second = _extract("cc6+b39")
    assert first == second
    assert canonical_attempt_extraction_json(first) == canonical_attempt_extraction_json(second)
    assert attempt_extraction_fingerprint(first) == attempt_extraction_fingerprint(second)


def test_canonical_json_is_compact_sorted_utf8_and_covers_complete_public_output():
    result = _extract("4f17")
    payload = canonical_attempt_extraction_json(result)
    structure = json.loads(payload)
    assert isinstance(payload, bytes)
    assert payload == json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert b": " not in payload and b", " not in payload
    assert set(structure) == {
        "abstention_count",
        "active_signal_count",
        "adaptations",
        "applicable_patterns",
        "bundle",
        "contract_version",
        "diagnostics",
        "extraction_state",
        "non_applicable_patterns",
        "previous_attempt_was_fault",
        "reason_codes",
        "requested_patterns",
        "serve_number",
    }
    assert len(structure["adaptations"]) == 5
    assert structure["bundle"]["signals"]


def test_fingerprint_uses_uppercase_sha256_and_distinct_domain():
    result = _extract("4f17")
    payload = canonical_attempt_extraction_json(result)
    expected = sha256(ATTEMPT_FINGERPRINT_DOMAIN + payload).hexdigest().upper()
    assert attempt_extraction_fingerprint(result) == expected
    assert len(expected) == 64 and expected == expected.upper()
    assert ATTEMPT_FINGERPRINT_DOMAIN not in {
        b"tennis-tactical-signal\x00",
        b"tennis-tactical-signal-bundle\x00",
    }
    assert expected != tactical_signal_bundle_fingerprint(result.bundle)


def test_any_semantic_change_changes_attempt_fingerprint():
    first = _extract("4f17")
    direction_changed = _extract("5f17")
    context_changed = _extract("4f17", 2, False)
    assert attempt_extraction_fingerprint(first) != attempt_extraction_fingerprint(direction_changed)
    assert attempt_extraction_fingerprint(first) != attempt_extraction_fingerprint(context_changed)


def test_serialized_output_never_contains_raw_parse_or_individual_fields():
    raw = "cc6+b39?"
    payload = canonical_attempt_extraction_json(_extract(raw)).decode("utf-8")
    lowered = payload.lower()
    assert raw not in payload
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
        assert forbidden not in lowered


@pytest.mark.parametrize(
    "unsafe",
    [
        {"match_id": "m1"},
        {"test_rows": 1},
        {"nested": {"point_number": 7}},
        {"nested": [{"safe": "C:\\secret\\file"}]},
        {"nested": [{"safe": "file://private"}]},
        {"nested": [{"safe": "../private"}]},
        {"nested": object()},
        {"nested": float("nan")},
        {"nested": float("inf")},
    ],
)
def test_recursive_public_security_guard_rejects_sensitive_or_non_contract_values(unsafe):
    with pytest.raises(TacticalSignalOrchestrationError):
        orchestrator._validate_public_tree(unsafe)


def test_public_validator_accepts_canonical_first_and_second_attempts():
    validate_attempt_signal_extraction(_extract("4f17"))
    validate_attempt_signal_extraction(_extract("4f17", 2, False))
    validate_attempt_signal_extraction(_extract("4f17", 2, True))


def test_no_high_level_parse_and_classify_api_is_imported_or_called():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(not alias.name.startswith("parse_and_classify_") for alias in node.names)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert not node.func.id.startswith("parse_and_classify_")
            elif isinstance(node.func, ast.Attribute):
                assert not node.func.attr.startswith("parse_and_classify_")


def test_static_scope_has_one_parser_call_and_no_io_data_product_or_cli_dependencies():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parser_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_sequence"
    ]
    assert len(parser_calls) == 1
    forbidden_import_roots = {
        "pandas",
        "numpy",
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
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_import_roots for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_import_roots
            assert "explainable_direction_recommender" not in node.module
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            assert name not in forbidden_calls
    lowered = source.lower()
    assert "reports/" not in lowered
    assert "data/" not in lowered
    assert "__main__" not in lowered
    assert "argparse" not in lowered
    assert "p03" not in lowered
    assert "p07" not in lowered
    assert "p08" not in lowered


def test_product_validation_contains_no_assert_statements():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_only_authorized_low_level_classifiers_and_common_adapters_are_wired():
    source = inspect.getsource(orchestrator.extract_tactical_signals_for_attempt)
    for required in (
        "classify_first_serve_direction",
        "classify_initial_return_direction",
        "classify_initial_return_depth",
        "classify_initial_return_shot_type",
        "classify_initial_return_profile",
        "adapt_first_serve_direction_signal",
        "adapt_return_direction_signal",
        "adapt_return_depth_signal",
        "adapt_return_shot_type_signal",
        "adapt_return_profile_signal",
        "build_tactical_signal_bundle",
    ):
        assert source.count(required) == 1
