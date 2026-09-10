"""Pruebas sinteticas del contrato comun de senales tacticas."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

import src.recommender.tactical_signal_contract as contract_module

from src.analysis.first_serve_direction_classification import (
    FirstServeDirectionClassification,
    FirstServeDirectionReason,
    FirstServeDirectionState,
    classify_first_serve_direction,
)
from src.analysis.return_depth_feasibility import (
    ReturnDepthClassification,
    ReturnDepthReason,
    classify_initial_return_depth,
)
from src.analysis.return_direction_feasibility import (
    ReturnDirectionClassification,
    ReturnDirectionReason,
    classify_initial_return_direction,
)
from src.analysis.return_profile_feasibility import (
    DOCUMENTED_PROFILE_IDS,
    ReturnProfileClassification,
    classify_initial_return_profile,
)
from src.analysis.return_shot_type_feasibility import (
    DOCUMENTED_SHOT_TYPES,
    ReturnShotTypeClassification,
    ReturnShotTypeReason,
    classify_initial_return_shot_type,
)
from src.parsing.serve_sequence import ParseResult, parse_sequence
from src.recommender.tactical_signal_contract import (
    ADAPTER_VERSION,
    ADAPTABLE_PATTERN_IDS,
    ADAPTABLE_PATTERN_ORDER,
    ELIGIBLE_PATTERN_ORDER,
    P02_AGGREGATE_PUBLICATION_FINGERPRINT,
    REGISTRY_READINESS,
    SIGNAL_CONTRACT_VERSION,
    SignalAdaptationResult,
    SignalContext,
    SignalProvenance,
    TacticalSignal,
    TacticalSignalBundle,
    TacticalSignalContractError,
    adapt_first_serve_direction_signal,
    adapt_return_depth_signal,
    adapt_return_direction_signal,
    adapt_return_profile_signal,
    adapt_return_shot_type_signal,
    adapt_tactical_signal,
    build_tactical_signal_bundle,
    bundle_to_json_structure,
    canonical_json_bytes,
    signal_to_json_structure,
    tactical_signal_bundle_fingerprint,
    tactical_signal_fingerprint,
    validate_signal_adaptation_result,
    validate_tactical_signal,
    validate_tactical_signal_bundle,
)


MODULE_PATH = Path("src/recommender/tactical_signal_contract.py")


def _parsed(sequence: str, serve_number: int = 1) -> ParseResult:
    return parse_sequence(sequence, serve_number)


def _p02(sequence: str):
    parsed = _parsed(sequence, 1)
    return classify_first_serve_direction(sequence, 1, parsed)


def _p04(sequence: str, serve_number: int = 1, previous_fault: bool = False):
    parsed = _parsed(sequence, serve_number)
    result = classify_initial_return_direction(
        sequence,
        serve_number,
        parsed,
        previous_attempt_was_fault=previous_fault,
    )
    return result, parsed


def _p05(sequence: str, serve_number: int = 1, previous_fault: bool = False):
    parsed = _parsed(sequence, serve_number)
    result = classify_initial_return_depth(
        sequence, serve_number, parsed, previous_fault
    )
    return result, parsed


def _p06(sequence: str, serve_number: int = 1, previous_fault: bool = False):
    parsed = _parsed(sequence, serve_number)
    result = classify_initial_return_shot_type(
        sequence, serve_number, parsed, previous_fault
    )
    return result, parsed


def _p09(sequence: str, serve_number: int = 1, previous_fault: bool = False):
    parsed = _parsed(sequence, serve_number)
    result = classify_initial_return_profile(
        sequence, serve_number, parsed, previous_fault
    )
    return result, parsed


def _signal(pattern_id: str = "P04") -> TacticalSignal:
    if pattern_id == "P02":
        result = adapt_first_serve_direction_signal(_p02("4f17"))
    elif pattern_id == "P04":
        classification, parsed = _p04("4f17")
        result = adapt_return_direction_signal(classification, parsed=parsed)
    elif pattern_id == "P05":
        classification, parsed = _p05("4f17")
        result = adapt_return_depth_signal(classification, parsed=parsed)
    elif pattern_id == "P06":
        classification, parsed = _p06("4f17")
        result = adapt_return_shot_type_signal(classification, parsed=parsed)
    else:
        classification, parsed = _p09("4f17")
        result = adapt_return_profile_signal(classification, parsed=parsed)
    assert result.signal is not None
    return result.signal


@pytest.mark.parametrize(
    ("code", "value"),
    [("4", "wide"), ("5", "body"), ("6", "T")],
)
def test_p02_adapts_each_documented_first_serve_direction(code, value):
    result = adapt_first_serve_direction_signal(_p02(f"{code}f17"))
    assert result.abstained is False
    assert result.signal is not None
    assert result.signal.pattern_id == "P02"
    assert result.signal.canonical_name == "first_serve_direction"
    assert result.signal.unit == "point"
    assert result.signal.actor == "server"
    assert result.signal.tactical_value == value
    assert result.signal.components == (("serve_direction", value),)
    assert result.signal.context == SignalContext(serve_number=1, serve_direction=value)
    assert result.reason_codes == ("actionable_direction",)
    assert result.provenance.upstream_pattern == "first_serve_direction_classification"
    assert (
        result.provenance.upstream_publication_fingerprint
        == P02_AGGREGATE_PUBLICATION_FINGERPRINT
    )


@pytest.mark.parametrize("serve_number", [1, 2])
@pytest.mark.parametrize("code", ["1", "2", "3"])
def test_p04_adapts_all_documented_directions_on_both_serves(serve_number, code):
    sequence = f"4f{code}7"
    classification, parsed = _p04(sequence, serve_number, serve_number == 2)
    result = adapt_return_direction_signal(classification, parsed=parsed)
    assert result.signal is not None
    assert result.signal.tactical_value == code
    assert result.signal.context.return_lateral_direction == code
    assert result.reason_codes == (f"observed_lateral_direction_{code}",)


@pytest.mark.parametrize("serve_number", [1, 2])
@pytest.mark.parametrize("code", ["7", "8", "9"])
def test_p05_adapts_all_documented_depths_on_both_serves(serve_number, code):
    sequence = f"4f1{code}"
    classification, parsed = _p05(sequence, serve_number, serve_number == 2)
    result = adapt_return_depth_signal(classification, parsed=parsed)
    assert result.signal is not None
    assert result.signal.tactical_value == code
    assert result.signal.context.return_depth == code
    assert result.reason_codes == (f"documented_depth_{code}",)


@pytest.mark.parametrize("serve_number", [1, 2])
@pytest.mark.parametrize("code", sorted(DOCUMENTED_SHOT_TYPES))
def test_p06_adapts_all_17_codes_on_both_serves(serve_number, code):
    classification, parsed = _p06(f"4{code}", serve_number, serve_number == 2)
    result = adapt_return_shot_type_signal(classification, parsed=parsed)
    assert result.signal is not None
    assert result.signal.tactical_value == code
    assert result.signal.context.return_shot_type == code
    assert result.reason_codes == (f"documented_return_shot_type_{code}",)


@pytest.mark.parametrize("profile_id", sorted(DOCUMENTED_PROFILE_IDS))
def test_p09_adapts_every_one_of_the_153_profile_ids(profile_id):
    shot, direction, depth = profile_id.split("|")
    classification, parsed = _p09(f"4{shot}{direction}{depth}")
    result = adapt_return_profile_signal(classification, parsed=parsed)
    assert result.signal is not None
    assert result.signal.tactical_value == profile_id
    assert result.signal.components == (
        ("return_shot_type", shot),
        ("return_lateral_direction", direction),
        ("return_depth", depth),
    )
    assert result.signal.context.return_profile == profile_id
    assert result.reason_codes == ("documented_type_direction_depth_profile",)


@pytest.mark.parametrize("serve_number", [1, 2])
def test_p09_supports_first_and_second_serve(serve_number):
    classification, parsed = _p09("5b39", serve_number, serve_number == 2)
    result = adapt_return_profile_signal(classification, parsed=parsed)
    assert result.signal is not None
    assert result.signal.context.serve_number == serve_number
    assert result.signal.context.serve_direction == "body"
    assert result.signal.tactical_value == "b|3|9"


@pytest.mark.parametrize(
    ("adapter", "factory", "sequence", "condition"),
    [
        (adapt_return_direction_signal, _p04, "4f0", "unknown"),
        (adapt_return_depth_signal, _p05, "4f10", "unknown"),
        (adapt_return_depth_signal, _p05, "4f1", "not_documented"),
        (adapt_return_shot_type_signal, _p06, "4q", "unknown"),
        (adapt_return_profile_signal, _p09, "4q17", "unknown"),
        (adapt_return_profile_signal, _p09, "4f1", "not_documented"),
    ],
)
def test_unknown_and_not_documented_states_abstain(adapter, factory, sequence, condition):
    classification, parsed = factory(sequence)
    result = adapter(classification, parsed=parsed)
    assert result.signal is None
    assert result.abstained is True
    assert result.abstention_condition == condition
    assert result.diagnostics[0] == ("component_count", 0)


@pytest.mark.parametrize(
    ("adapter", "factory"),
    [
        (adapt_return_direction_signal, _p04),
        (adapt_return_depth_signal, _p05),
        (adapt_return_shot_type_signal, _p06),
        (adapt_return_profile_signal, _p09),
    ],
)
def test_censored_serve_terminal_abstains(adapter, factory):
    classification, parsed = factory("4*")
    result = adapter(classification, parsed=parsed)
    assert result.signal is None
    assert result.abstention_condition == "censored"
    assert result.reason_codes == ("censored_ace",)


@pytest.mark.parametrize(
    ("sequence", "serve_number", "previous_fault", "state", "reason", "condition"),
    [
        ("", 1, False, "unknown_initial_return", "missing_service_prefix", "unknown"),
        ("6f0", 1, False, "return_direction_unknown", "documented_unknown_direction_0", "unknown"),
        ("6f#", 1, False, "return_direction_unknown", "localized_return_without_direction", "unknown"),
        ("6f7", 1, False, "unknown_initial_return", "ambiguous_post_service_boundary", "unknown"),
        ("6++f2", 1, False, "unknown_initial_return", "unsupported_initial_modifier", "unknown"),
        ("6q2", 1, False, "unknown_initial_return", "unknown_return_shot_type", "unknown"),
        ("6", 1, False, "unknown_initial_return", "truncated_return_event", "unknown"),
        ("5*", 1, False, "ace", "censored_ace", "censored"),
        ("5#", 1, False, "unreturned_serve", "censored_unreturned_serve", "censored"),
        ("5n", 1, False, "service_fault", "censored_service_fault", "censored"),
        ("5n", 2, True, "service_fault", "censored_double_fault", "censored"),
        ("S", 1, False, "special_or_incomplete", "challenge_or_penalty_before_return", "censored"),
        ("V", 1, False, "special_or_incomplete", "special_event_before_return", "censored"),
        ("cc", 1, False, "special_or_incomplete", "incomplete_let", "censored"),
    ],
)
def test_p04_all_reachable_abstention_reasons_are_preserved(
    sequence, serve_number, previous_fault, state, reason, condition
):
    classification, parsed = _p04(sequence, serve_number, previous_fault)
    result = adapt_return_direction_signal(classification, parsed=parsed)
    assert result.signal is None
    assert result.upstream_state == state
    assert result.reason_codes == (reason,)
    assert result.abstention_condition == condition


@pytest.mark.parametrize(
    ("sequence", "serve_number", "previous_fault", "state", "reason", "condition"),
    [
        ("", 1, False, "unknown_initial_return", "missing_service_prefix", "unknown"),
        ("6f10", 1, False, "return_depth_unknown", "documented_unknown_depth_0", "unknown"),
        ("6f1", 1, False, "return_depth_not_documented", "no_documented_depth_after_lateral_direction", "not_documented"),
        ("6f7", 1, False, "unknown_initial_return", "ambiguous_post_service_boundary", "unknown"),
        ("6++f17", 1, False, "unknown_initial_return", "unsupported_initial_modifier", "unknown"),
        ("6q17", 1, False, "unknown_initial_return", "unknown_return_shot_type", "unknown"),
        ("6", 1, False, "unknown_initial_return", "truncated_return_event", "unknown"),
        ("5*", 1, False, "ineligible_censored", "censored_ace", "censored"),
        ("5#", 1, False, "ineligible_censored", "censored_unreturned_serve", "censored"),
        ("5n", 1, False, "ineligible_censored", "censored_service_fault", "censored"),
        ("5n", 2, True, "ineligible_censored", "censored_double_fault", "censored"),
        ("V", 1, False, "ineligible_censored", "special_event_before_return", "censored"),
        ("cc", 1, False, "ineligible_censored", "incomplete_let", "censored"),
    ],
)
def test_p05_all_reachable_abstention_reasons_are_preserved(
    sequence, serve_number, previous_fault, state, reason, condition
):
    classification, parsed = _p05(sequence, serve_number, previous_fault)
    result = adapt_return_depth_signal(classification, parsed=parsed)
    assert result.signal is None
    assert result.upstream_state == state
    assert result.reason_codes == (reason,)
    assert result.abstention_condition == condition


@pytest.mark.parametrize(
    ("sequence", "serve_number", "previous_fault", "state", "reason", "condition"),
    [
        ("", 1, False, "unknown_initial_return", "missing_service_prefix", "unknown"),
        ("6q", 1, False, "return_shot_type_unknown", "documented_unknown_return_shot_type_q", "unknown"),
        ("6?f", 1, False, "unknown_initial_return", "ambiguous_post_service_boundary", "unknown"),
        ("6++f", 1, False, "unknown_initial_return", "unsupported_initial_modifier", "unknown"),
        ("6", 1, False, "unknown_initial_return", "truncated_return_event", "unknown"),
        ("5*", 1, False, "ineligible_censored", "censored_ace", "censored"),
        ("5#", 1, False, "ineligible_censored", "censored_unreturned_serve", "censored"),
        ("5n", 1, False, "ineligible_censored", "censored_service_fault", "censored"),
        ("5n", 2, True, "ineligible_censored", "censored_double_fault", "censored"),
        ("V", 1, False, "ineligible_censored", "special_event_before_return", "censored"),
        ("cc", 1, False, "ineligible_censored", "incomplete_let", "censored"),
    ],
)
def test_p06_all_reachable_abstention_reasons_are_preserved(
    sequence, serve_number, previous_fault, state, reason, condition
):
    classification, parsed = _p06(sequence, serve_number, previous_fault)
    result = adapt_return_shot_type_signal(classification, parsed=parsed)
    assert result.signal is None
    assert result.upstream_state == state
    assert result.reason_codes == (reason,)
    assert result.abstention_condition == condition


@pytest.mark.parametrize(
    ("sequence", "state", "reason", "condition"),
    [
        ("0f17", "first_serve_direction_unknown", "direction_unknown_0", "unknown"),
        ("", "unknown_first_serve", "empty_sequence", "unknown"),
        (" ", "unknown_first_serve", "whitespace_only_sequence", "unknown"),
        ("c", "unknown_first_serve", "unrecognized_or_no_initial_direction", "unknown"),
        ("S", "ineligible_censored", "special_unit_sequence", "censored"),
    ],
)
def test_p02_all_abstention_states_preserve_upstream_contract(
    sequence, state, reason, condition
):
    result = adapt_first_serve_direction_signal(_p02(sequence))
    assert result.signal is None
    assert result.abstained is True
    assert result.upstream_state == state
    assert result.reason_codes == (reason,)
    assert result.abstention_condition == condition
    assert result.diagnostics == (("component_count", 0),)


@pytest.mark.parametrize("sequence", ["4*", "5#", "6n"])
def test_p02_service_terminal_does_not_erase_documented_direction(sequence):
    result = adapt_first_serve_direction_signal(_p02(sequence))
    assert result.signal is not None
    assert result.signal.tactical_value in {"wide", "body", "T"}


@pytest.mark.parametrize("wrong", [object(), _parsed("4", 1), _p04("4f17")[0]])
def test_p02_adapter_accepts_only_exact_individual_classification(wrong):
    with pytest.raises(TypeError, match="FirstServeDirectionClassification exacta"):
        adapt_first_serve_direction_signal(wrong)  # type: ignore[arg-type]


def test_p02_adapter_rejects_wrong_provenance_type():
    with pytest.raises(TypeError, match="provenance"):
        adapt_first_serve_direction_signal(_p02("4"), provenance=object())  # type: ignore[arg-type]


def test_p02_adapter_calls_public_validator_once_without_parser_or_extractor(
    monkeypatch,
):
    classification = _p02("4f17")
    calls = []
    real_validator = contract_module.validate_first_serve_direction_classification

    def counted_validator(value):
        calls.append(value)
        return real_validator(value)

    monkeypatch.setattr(
        contract_module,
        "validate_first_serve_direction_classification",
        counted_validator,
    )
    result = adapt_first_serve_direction_signal(classification)
    assert result.signal is not None
    assert calls == [classification]


def test_p02_rejects_incorrect_aggregate_publication_fingerprint():
    canonical = adapt_first_serve_direction_signal(_p02("4")).provenance
    manipulated = replace(
        canonical,
        upstream_publication_fingerprint="B" * 64,
    )
    with pytest.raises(TacticalSignalContractError, match="procedencia"):
        adapt_first_serve_direction_signal(_p02("4"), provenance=manipulated)


@pytest.mark.parametrize(
    ("adapter", "wrong"),
    [
        (adapt_return_direction_signal, ReturnDepthClassification),
        (adapt_return_depth_signal, ReturnShotTypeClassification),
        (adapt_return_shot_type_signal, ReturnProfileClassification),
        (adapt_return_profile_signal, ReturnDirectionClassification),
    ],
)
def test_adapters_reject_incompatible_upstream_types(adapter, wrong):
    factories = {
        ReturnDepthClassification: _p05,
        ReturnShotTypeClassification: _p06,
        ReturnProfileClassification: _p09,
        ReturnDirectionClassification: _p04,
    }
    classification, parsed = factories[wrong]("4f17")
    with pytest.raises(TypeError):
        adapter(classification, parsed=parsed)


@pytest.mark.parametrize("pattern_id", ["P03", "P07", "P08", "P01", "P10", "p04", ""])
def test_generic_adapter_rejects_non_eligible_patterns(pattern_id):
    with pytest.raises(TacticalSignalContractError, match="Solo P02/P04/P05/P06/P09"):
        adapt_tactical_signal(pattern_id, object())


def test_generic_adapter_dispatches_all_safely_adaptable_patterns():
    cases = [adapt_tactical_signal("P02", _p02("4f17"))]
    for pattern_id, factory in (("P04", _p04), ("P05", _p05), ("P06", _p06), ("P09", _p09)):
        classification, parsed = factory("4f17")
        cases.append(adapt_tactical_signal(pattern_id, classification, parsed=parsed))
    assert [item.requested_pattern_id for item in cases] == list(ADAPTABLE_PATTERN_ORDER)


def test_generic_p02_dispatch_rejects_separate_parse_result():
    with pytest.raises(TacticalSignalContractError, match="no acepta ParseResult separado"):
        adapt_tactical_signal("P02", _p02("4f17"), parsed=_parsed("4f17", 1))


def test_p02_upstream_semantic_mutation_is_rejected():
    classification = _p02("4f17")
    with pytest.raises(ValueError):
        replace(classification, eligible_for_direction_signal=False)


@pytest.mark.parametrize(
    ("adapter", "factory", "field", "value"),
    [
        (adapt_return_direction_signal, _p04, "eligible_for_direction_analysis", False),
        (adapt_return_depth_signal, _p05, "eligible_for_depth_comparison", False),
        (adapt_return_shot_type_signal, _p06, "eligible_for_type_comparison", False),
        (adapt_return_profile_signal, _p09, "eligible_for_profile_comparison", False),
    ],
)
def test_upstream_semantic_mutations_are_rejected(adapter, factory, field, value):
    classification, parsed = factory("4f17")
    manipulated = replace(classification, **{field: value})
    with pytest.raises(ValueError):
        adapter(manipulated, parsed=parsed)


@pytest.mark.parametrize(
    ("adapter", "factory", "reason_field", "inconsistent_reason"),
    [
        (
            adapt_return_direction_signal,
            _p04,
            "reason_code",
            ReturnDirectionReason.INCONSISTENT,
        ),
        (
            adapt_return_depth_signal,
            _p05,
            "reason_codes",
            (ReturnDepthReason.INCONSISTENT,),
        ),
        (
            adapt_return_shot_type_signal,
            _p06,
            "reason_codes",
            (ReturnShotTypeReason.INCONSISTENT,),
        ),
    ],
)
def test_defensive_unreachable_upstream_reasons_cannot_cross_adapter_boundary(
    adapter, factory, reason_field, inconsistent_reason
):
    classification, parsed = factory("")
    manipulated = replace(classification, **{reason_field: inconsistent_reason})
    with pytest.raises(ValueError):
        adapter(manipulated, parsed=parsed)


@pytest.mark.parametrize(
    ("adapter", "factory"),
    [
        (adapt_return_direction_signal, _p04),
        (adapt_return_depth_signal, _p05),
        (adapt_return_shot_type_signal, _p06),
        (adapt_return_profile_signal, _p09),
    ],
)
def test_parse_result_from_other_text_is_rejected(adapter, factory):
    classification, _ = factory("4f17")
    with pytest.raises(ValueError, match="no corresponde"):
        adapter(classification, parsed=_parsed("5f17"))


def test_provenance_is_propagated_exactly():
    provenance = SignalProvenance(
        upstream_pattern="documented_initial_return_depth",
        adapter_version=ADAPTER_VERSION,
        upstream_contract_version="1.0.0",
        registry_readiness=REGISTRY_READINESS,
        upstream_publication_fingerprint="A" * 64,
        synthetic_origin=True,
    )
    classification, parsed = _p05("4f17")
    result = adapt_return_depth_signal(classification, parsed=parsed, provenance=provenance)
    assert result.provenance is provenance
    assert result.signal is not None and result.signal.provenance is provenance


@pytest.mark.parametrize(
    "bad_fingerprint",
    ["a" * 64, "A" * 63, "G" * 64, "", 1, True],
)
def test_provenance_rejects_invalid_fingerprint(bad_fingerprint):
    with pytest.raises(TacticalSignalContractError, match="Fingerprint"):
        SignalProvenance(
            upstream_pattern="documented_initial_return_depth",
            adapter_version=ADAPTER_VERSION,
            upstream_contract_version="1.0.0",
            registry_readiness=REGISTRY_READINESS,
            upstream_publication_fingerprint=bad_fingerprint,
            synthetic_origin=True,
        )


@pytest.mark.parametrize(
    "bad_value",
    [r"C:\private\file", "/private/file", "2026-09-10", "raw sequence", "player name"],
)
def test_provenance_rejects_paths_dates_and_free_text(bad_value):
    with pytest.raises(TacticalSignalContractError):
        SignalProvenance(
            upstream_pattern=bad_value,
            adapter_version=ADAPTER_VERSION,
            upstream_contract_version="1.0.0",
            registry_readiness=REGISTRY_READINESS,
            upstream_publication_fingerprint=None,
            synthetic_origin=True,
        )


@pytest.mark.parametrize("bad", [False, 1, "true", None])
def test_provenance_requires_explicit_true_synthetic_origin(bad):
    with pytest.raises(TacticalSignalContractError):
        SignalProvenance(
            upstream_pattern="documented_initial_return_depth",
            adapter_version=ADAPTER_VERSION,
            upstream_contract_version="1.0.0",
            registry_readiness=REGISTRY_READINESS,
            upstream_publication_fingerprint=None,
            synthetic_origin=bad,
        )


def test_bundle_complete_second_attempt_return_context_is_coherent():
    sequence = "4f17"
    parsed = _parsed(sequence, 2)
    results = []
    for adapter, classifier in (
        (adapt_return_direction_signal, classify_initial_return_direction),
        (adapt_return_depth_signal, classify_initial_return_depth),
        (adapt_return_shot_type_signal, classify_initial_return_shot_type),
        (adapt_return_profile_signal, classify_initial_return_profile),
    ):
        if classifier is classify_initial_return_direction:
            classification = classifier(sequence, 2, parsed, previous_attempt_was_fault=True)
        else:
            classification = classifier(sequence, 2, parsed, True)
        results.append(adapter(classification, parsed=parsed))
    bundle = build_tactical_signal_bundle(tuple(reversed(results)))
    assert tuple(item.pattern_id for item in bundle.signals) == ("P04", "P05", "P06", "P09")
    assert bundle.abstentions == ()
    assert bundle.context == SignalContext(
        serve_number=2,
        serve_direction="wide",
        return_shot_type="f",
        return_lateral_direction="1",
        return_depth="7",
        return_profile="f|1|7",
    )


def test_bundle_allows_partial_signals_and_separate_abstentions():
    direction, parsed_direction = _p04("4f17")
    depth, parsed_depth = _p05("4f10")
    shot, parsed_shot = _p06("4f17")
    bundle = build_tactical_signal_bundle(
        (
            adapt_return_shot_type_signal(shot, parsed=parsed_shot),
            adapt_return_depth_signal(depth, parsed=parsed_depth),
            adapt_return_direction_signal(direction, parsed=parsed_direction),
        )
    )
    assert tuple(item.pattern_id for item in bundle.signals) == ("P04", "P06")
    assert tuple(item.requested_pattern_id for item in bundle.abstentions) == ("P05",)
    assert bundle.abstentions[0].abstention_condition == "unknown"


def test_p02_can_form_an_isolated_first_attempt_bundle():
    result = adapt_first_serve_direction_signal(_p02("6"))
    bundle = build_tactical_signal_bundle((result,))
    assert tuple(item.pattern_id for item in bundle.signals) == ("P02",)
    assert bundle.context == SignalContext(serve_number=1, serve_direction="T")


def test_p02_combines_with_first_attempt_return_signals_in_canonical_order():
    results = [adapt_first_serve_direction_signal(_p02("4f17"))]
    for pattern_id in ("P04", "P05", "P06", "P09"):
        results.append(_signal_result_for_pattern(pattern_id))
    bundle = build_tactical_signal_bundle(tuple(reversed(results)))
    assert tuple(item.pattern_id for item in bundle.signals) == ELIGIBLE_PATTERN_ORDER
    assert bundle.context == SignalContext(
        serve_number=1,
        serve_direction="wide",
        return_shot_type="f",
        return_lateral_direction="1",
        return_depth="7",
        return_profile="f|1|7",
    )


def test_single_attempt_bundle_rejects_p02_with_second_attempt_return_signal():
    depth, parsed = _p05("4f17", 2, True)
    with pytest.raises(TacticalSignalContractError, match="serve_number"):
        build_tactical_signal_bundle(
            (
                adapt_first_serve_direction_signal(_p02("4f17")),
                adapt_return_depth_signal(depth, parsed=parsed),
            )
        )


def test_p02_abstention_can_coexist_with_first_attempt_return_signal():
    direction, parsed = _p04("0f17")
    bundle = build_tactical_signal_bundle(
        (
            adapt_return_direction_signal(direction, parsed=parsed),
            adapt_first_serve_direction_signal(_p02("0f17")),
        )
    )
    assert tuple(item.pattern_id for item in bundle.signals) == ("P04",)
    assert tuple(item.requested_pattern_id for item in bundle.abstentions) == ("P02",)
    assert bundle.context.serve_number == 1
    assert bundle.context.serve_direction == "unknown"


@pytest.mark.parametrize(
    "pattern_ids",
    [
        ("P04",),
        ("P05",),
        ("P06",),
        ("P09",),
        ("P04", "P05"),
        ("P04", "P06"),
        ("P05", "P06"),
        ("P04", "P05", "P06"),
        ("P04", "P05", "P06", "P09"),
    ],
)
def test_all_relevant_return_bundle_subsets_are_supported(pattern_ids):
    results = tuple(
        _signal_result_for_pattern(pattern_id) for pattern_id in reversed(pattern_ids)
    )
    bundle = build_tactical_signal_bundle(results)
    assert tuple(signal.pattern_id for signal in bundle.signals) == pattern_ids
    assert bundle.context.serve_number == 1


def _signal_result_for_pattern(pattern_id):
    factories = {"P04": _p04, "P05": _p05, "P06": _p06, "P09": _p09}
    adapters = {
        "P04": adapt_return_direction_signal,
        "P05": adapt_return_depth_signal,
        "P06": adapt_return_shot_type_signal,
        "P09": adapt_return_profile_signal,
    }
    classification, parsed = factories[pattern_id]("4f17")
    return adapters[pattern_id](classification, parsed=parsed)


def test_p09_abstention_can_coexist_with_observed_available_components():
    direction, parsed = _p04("4f1")
    shot, shot_parsed = _p06("4f1")
    profile, profile_parsed = _p09("4f1")
    bundle = build_tactical_signal_bundle(
        (
            adapt_return_profile_signal(profile, parsed=profile_parsed),
            adapt_return_shot_type_signal(shot, parsed=shot_parsed),
            adapt_return_direction_signal(direction, parsed=parsed),
        )
    )
    assert tuple(signal.pattern_id for signal in bundle.signals) == ("P04", "P06")
    assert tuple(item.requested_pattern_id for item in bundle.abstentions) == ("P09",)
    assert bundle.abstentions[0].abstention_condition == "not_documented"


def test_same_pattern_cannot_be_signal_and_abstention_in_one_bundle():
    active = _signal_result_for_pattern("P05")
    unknown, parsed = _p05("4f10")
    abstention = adapt_return_depth_signal(unknown, parsed=parsed)
    with pytest.raises(TacticalSignalContractError, match="duplicados"):
        build_tactical_signal_bundle((active, abstention))


def test_empty_bundle_is_explicitly_invalid():
    with pytest.raises(TacticalSignalContractError, match="no puede estar vacio"):
        build_tactical_signal_bundle(())


def test_bundle_rejects_duplicate_pattern():
    classification, parsed = _p04("4f17")
    result = adapt_return_direction_signal(classification, parsed=parsed)
    with pytest.raises(TacticalSignalContractError, match="duplicados"):
        build_tactical_signal_bundle((result, result))


def test_bundle_rejects_p09_component_contradiction():
    profile, parsed_profile = _p09("4f17")
    direction, parsed_direction = _p04("4f27")
    with pytest.raises(TacticalSignalContractError, match="Contexto contradictorio|contradice"):
        build_tactical_signal_bundle(
            (
                adapt_return_profile_signal(profile, parsed=parsed_profile),
                adapt_return_direction_signal(direction, parsed=parsed_direction),
            )
        )


def test_bundle_rejects_context_from_different_serve_attempt():
    direction, parsed_direction = _p04("4f17", 1)
    depth, parsed_depth = _p05("4f17", 2, True)
    with pytest.raises(TacticalSignalContractError, match="serve_number"):
        build_tactical_signal_bundle(
            (
                adapt_return_direction_signal(direction, parsed=parsed_direction),
                adapt_return_depth_signal(depth, parsed=parsed_depth),
            )
        )


def test_bundle_validator_rejects_noncanonical_signal_order():
    p04 = _signal("P04")
    p05 = _signal("P05")
    context = SignalContext(
        serve_number=1,
        serve_direction="wide",
        return_shot_type="f",
        return_lateral_direction="1",
        return_depth="7",
    )
    with pytest.raises(TacticalSignalContractError, match="orden canonico"):
        TacticalSignalBundle(SIGNAL_CONTRACT_VERSION, (p05, p04), (), context)


@pytest.mark.parametrize("pattern_id", ADAPTABLE_PATTERN_ORDER)
def test_signals_and_nested_collections_are_immutable(pattern_id):
    signal = _signal(pattern_id)
    with pytest.raises(FrozenInstanceError):
        signal.tactical_value = "changed"  # type: ignore[misc]
    assert type(signal.components) is tuple
    assert all(type(item) is tuple for item in signal.components)
    assert type(signal.reason_codes) is tuple
    assert type(signal.diagnostics) is tuple


def test_context_rejects_bool_float_string_and_unknown_domains():
    for bad in (True, 1.0, "1", 0, 3):
        with pytest.raises(TacticalSignalContractError):
            SignalContext(serve_number=bad)
    with pytest.raises(TacticalSignalContractError):
        SignalContext(return_shot_type="q")
    with pytest.raises(TacticalSignalContractError):
        SignalContext(return_lateral_direction="0")
    with pytest.raises(TacticalSignalContractError):
        SignalContext(return_depth="0")


def test_context_rejects_incoherent_profile_components():
    with pytest.raises(TacticalSignalContractError, match="perfil contradice"):
        SignalContext(
            return_shot_type="f",
            return_lateral_direction="2",
            return_depth="7",
            return_profile="f|1|7",
        )


def test_adaptation_result_exclusive_presence_contract():
    signal = _signal("P04")
    classification, parsed = _p04("4f17")
    active = adapt_return_direction_signal(classification, parsed=parsed)
    with pytest.raises(TacticalSignalContractError):
        replace(
            active,
            abstained=True,
            abstention_condition="unknown",
        )
    with pytest.raises(TacticalSignalContractError):
        replace(active, signal=None, abstained=False)
    assert signal.pattern_id == "P04"


def test_signal_validator_rejects_derived_mutations():
    signal = _signal("P09")
    mutations = (
        {"contract_version": "2.0.0"},
        {"pattern_id": "P05"},
        {"canonical_name": "wrong"},
        {"unit": "point"},
        {"actor": "server"},
        {"eligible": False},
        {"comparable": False},
        {"outcome_available": False},
        {"registry_readiness": "diagnostic_only"},
        {"tactical_value": "f|1|8"},
        {"components": list(signal.components)},
        {"reason_codes": list(signal.reason_codes)},
        {"reason_codes": ("documented_depth_7",)},
        {"upstream_state": "return_depth_observed"},
        {"abstention_condition": "unknown"},
        {"diagnostics": (("component_count", 2),)},
    )
    for mutation in mutations:
        with pytest.raises((TacticalSignalContractError, TypeError)):
            replace(signal, **mutation)


@pytest.mark.parametrize(
    "bad_count",
    [True, 1.0, float("nan"), float("inf"), -1, "1", None, [], {}, (), object()],
)
def test_diagnostics_reject_coercions_nan_infinity_and_negative(bad_count):
    signal = _signal("P04")
    diagnostics = (("component_count", bad_count),)
    with pytest.raises(TacticalSignalContractError):
        replace(signal, diagnostics=diagnostics)


def test_diagnostics_reject_individual_or_unsafe_fields_recursively():
    signal = _signal("P04")
    unsafe = (
        ("component_count", 1),
        ("match_id", 1),
    )
    with pytest.raises(TacticalSignalContractError):
        replace(signal, diagnostics=unsafe)


def test_canonical_signal_serialization_and_fingerprint_are_deterministic():
    first = _signal("P09")
    second = _signal("P09")
    first_bytes = canonical_json_bytes(first)
    assert first_bytes == canonical_json_bytes(second)
    assert tactical_signal_fingerprint(first) == tactical_signal_fingerprint(second)
    assert tactical_signal_fingerprint(first) == tactical_signal_fingerprint(first)
    decoded = json.loads(first_bytes.decode("utf-8"))
    assert decoded == signal_to_json_structure(first)
    assert b"NaN" not in first_bytes and b"Infinity" not in first_bytes


def test_semantic_signal_change_changes_fingerprint():
    first = _signal("P04")
    classification, parsed = _p04("4f27")
    second_result = adapt_return_direction_signal(classification, parsed=parsed)
    assert second_result.signal is not None
    assert tactical_signal_fingerprint(first) != tactical_signal_fingerprint(second_result.signal)


def test_provenance_fingerprint_change_changes_signal_fingerprint():
    classification, parsed = _p05("4f17")
    first_provenance = SignalProvenance(
        upstream_pattern="documented_initial_return_depth",
        adapter_version=ADAPTER_VERSION,
        upstream_contract_version="1.0.0",
        registry_readiness=REGISTRY_READINESS,
        upstream_publication_fingerprint="A" * 64,
        synthetic_origin=True,
    )
    second_provenance = replace(
        first_provenance, upstream_publication_fingerprint="B" * 64
    )
    first = adapt_return_depth_signal(
        classification, parsed=parsed, provenance=first_provenance
    )
    second = adapt_return_depth_signal(
        classification, parsed=parsed, provenance=second_provenance
    )
    assert first.signal is not None and second.signal is not None
    assert tactical_signal_fingerprint(first.signal) != tactical_signal_fingerprint(second.signal)


def test_canonical_bundle_serialization_and_fingerprint_are_deterministic():
    classification, parsed = _p09("4f17")
    profile = adapt_return_profile_signal(classification, parsed=parsed)
    shot, shot_parsed = _p06("4f17")
    shot_result = adapt_return_shot_type_signal(shot, parsed=shot_parsed)
    first = build_tactical_signal_bundle((profile, shot_result))
    second = build_tactical_signal_bundle((shot_result, profile))
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert tactical_signal_bundle_fingerprint(first) == tactical_signal_bundle_fingerprint(second)
    assert json.loads(canonical_json_bytes(first)) == bundle_to_json_structure(first)


def test_signal_and_bundle_fingerprints_use_distinct_domains():
    signal = _signal("P04")
    bundle = build_tactical_signal_bundle((_signal_result("P04"),))
    assert tactical_signal_fingerprint(signal) != tactical_signal_bundle_fingerprint(bundle)
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'b"tennis-tactical-signal\\x00"' in source
    assert 'b"tennis-tactical-signal-bundle\\x00"' in source


def test_semantic_bundle_change_changes_fingerprint():
    p04_one, parsed_one = _p04("4f17")
    p04_two, parsed_two = _p04("4f27")
    first = build_tactical_signal_bundle((adapt_return_direction_signal(p04_one, parsed=parsed_one),))
    second = build_tactical_signal_bundle((adapt_return_direction_signal(p04_two, parsed=parsed_two),))
    assert tactical_signal_bundle_fingerprint(first) != tactical_signal_bundle_fingerprint(second)


@pytest.mark.parametrize("value", [None, {}, [], (), object(), "P04", 1, True])
def test_serializer_rejects_non_contract_objects(value):
    with pytest.raises(TypeError):
        canonical_json_bytes(value)  # type: ignore[arg-type]


def test_serialized_contract_never_contains_individual_data_or_raw_sequences():
    bundle = build_tactical_signal_bundle(
        (
            adapt_first_serve_direction_signal(_p02("4f17")),
            _signal_result("P04"),
            _signal_result("P09"),
        )
    )
    payload = canonical_json_bytes(bundle).decode("utf-8").lower()
    forbidden = (
        "sequence_text",
        "raw_sequence",
        "match_id",
        "point_number",
        "player_1",
        "player_2",
        "point_winner",
        "timestamp",
    )
    assert all(term not in payload for term in forbidden)
    assert "4f17" not in payload
    assert "residual" not in payload
    assert "span" not in payload
    assert "warning" not in payload


def _signal_result(pattern_id: str) -> SignalAdaptationResult:
    if pattern_id == "P04":
        classification, parsed = _p04("4f17")
        return adapt_return_direction_signal(classification, parsed=parsed)
    classification, parsed = _p09("4f17")
    return adapt_return_profile_signal(classification, parsed=parsed)


def test_public_validators_accept_canonical_objects():
    result = _signal_result("P09")
    assert result.signal is not None
    validate_tactical_signal(result.signal)
    validate_signal_adaptation_result(result)
    bundle = build_tactical_signal_bundle((result,))
    validate_tactical_signal_bundle(bundle)


def test_module_has_no_io_pandas_or_product_recommender_dependency():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "pandas" not in imports
    assert not any("explainable_direction_recommender" in name for name in imported_from)
    assert not any(isinstance(node, (ast.With, ast.AsyncWith)) for node in ast.walk(tree))
    forbidden_calls = {"open", "read_csv", "read_parquet", "to_csv", "to_json", "parse_sequence"}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert forbidden_calls.isdisjoint(calls)


def test_adapter_source_does_not_import_or_call_parser_entry_point():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.parsing.serve_sequence":
            assert all(alias.name != "parse_sequence" for alias in node.names)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id != "parse_sequence"
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr != "parse_sequence"


def test_registry_eligibility_and_safe_adapter_catalogues_are_aligned_and_closed():
    assert ELIGIBLE_PATTERN_ORDER == ("P02", "P04", "P05", "P06", "P09")
    assert ADAPTABLE_PATTERN_ORDER == ELIGIBLE_PATTERN_ORDER
    assert ADAPTABLE_PATTERN_IDS == frozenset(ADAPTABLE_PATTERN_ORDER)
    assert frozenset(contract_module._PATTERN_METADATA) == ADAPTABLE_PATTERN_IDS
    assert "P02" in ADAPTABLE_PATTERN_IDS
    assert (
        contract_module._PATTERN_METADATA["P02"]["publication_fingerprint"]
        == P02_AGGREGATE_PUBLICATION_FINGERPRINT
    )


def test_signal_context_and_provenance_are_frozen():
    context = SignalContext(serve_number=1)
    provenance = _signal("P04").provenance
    with pytest.raises(FrozenInstanceError):
        context.serve_number = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        provenance.synthetic_origin = False  # type: ignore[misc]


def test_private_contract_catalogues_are_deeply_immutable():
    with pytest.raises(TypeError):
        contract_module._PATTERN_METADATA["P04"]["actor"] = "server"
    with pytest.raises(TypeError):
        contract_module._ACTIVE_REASON_BY_PATTERN["P04"]["1"] = "changed"
    with pytest.raises(TypeError):
        contract_module._ABSTENTION_CONTRACTS["P05"]["return_depth_unknown"] = ()
    allowed = contract_module._ABSTENTION_CONTRACTS["P05"]["return_depth_unknown"][1]
    assert type(allowed) is frozenset
    with pytest.raises(TypeError):
        contract_module._P09_ABSTENTION_REASON_TUPLES[
            "initial_return_profile_unknown"
        ] = frozenset()
    assert type(
        contract_module._P09_ABSTENTION_REASON_TUPLES[
            "initial_return_profile_unknown"
        ]
    ) is frozenset


def test_bundle_is_frozen_and_uses_tuples():
    bundle = build_tactical_signal_bundle((_signal_result("P04"),))
    with pytest.raises(FrozenInstanceError):
        bundle.signals = ()  # type: ignore[misc]
    assert type(bundle.signals) is tuple
    assert type(bundle.abstentions) is tuple


def test_bundle_validator_rejects_each_inconsistent_public_field():
    bundle = build_tactical_signal_bundle((_signal_result("P04"),))
    mutations = (
        {"contract_version": "2.0.0"},
        {"signals": list(bundle.signals)},
        {"abstentions": list(bundle.abstentions)},
        {"context": SignalContext(serve_number=2, return_lateral_direction="1")},
        {"signals": (bundle.signals[0], bundle.signals[0])},
    )
    for mutation in mutations:
        with pytest.raises((TacticalSignalContractError, TypeError)):
            replace(bundle, **mutation)


def test_result_rejects_list_reason_codes_and_diagnostics():
    result = _signal_result("P04")
    with pytest.raises(TacticalSignalContractError):
        replace(result, reason_codes=list(result.reason_codes))
    with pytest.raises(TacticalSignalContractError):
        replace(result, diagnostics=list(result.diagnostics))


def test_abstention_rejects_mutated_state_reason_and_condition():
    classification, parsed = _p05("4f10")
    result = adapt_return_depth_signal(classification, parsed=parsed)
    assert result.signal is None
    for mutation in (
        {"upstream_state": "return_depth_not_documented"},
        {"reason_codes": ("no_documented_depth_after_lateral_direction",)},
        {"abstention_condition": "censored"},
        {"upstream_state": "arbitrary_state"},
    ):
        with pytest.raises(TacticalSignalContractError):
            replace(result, **mutation)


def test_adaptation_result_rejects_all_inconsistent_public_field_mutations():
    result = _signal_result("P04")
    assert result.signal is not None
    mutations = (
        {"contract_version": "2.0.0"},
        {"requested_pattern_id": "P05"},
        {"signal": _signal("P05")},
        {"abstained": True},
        {"upstream_state": "return_direction_unknown"},
        {"reason_codes": ("localized_return_without_direction",)},
        {"abstention_condition": "unknown"},
        {"context": SignalContext(serve_number=2, return_lateral_direction="1")},
        {"diagnostics": (("component_count", 0),)},
        {"provenance": _signal("P05").provenance},
    )
    for mutation in mutations:
        with pytest.raises(TacticalSignalContractError):
            replace(result, **mutation)


def test_wrong_provenance_pattern_is_rejected_by_adapter():
    provenance = SignalProvenance(
        upstream_pattern="documented_initial_return_shot_type",
        adapter_version=ADAPTER_VERSION,
        upstream_contract_version="1.0.0",
        registry_readiness=REGISTRY_READINESS,
        upstream_publication_fingerprint=None,
        synthetic_origin=True,
    )
    classification, parsed = _p05("4f17")
    with pytest.raises(TacticalSignalContractError, match="procedencia"):
        adapt_return_depth_signal(classification, parsed=parsed, provenance=provenance)


def test_provenance_rejects_unknown_pattern_and_non_semantic_version():
    with pytest.raises(TacticalSignalContractError, match="catalogo cerrado"):
        SignalProvenance(
            upstream_pattern="RogerFederer",
            adapter_version=ADAPTER_VERSION,
            upstream_contract_version="1.0.0",
            registry_readiness=REGISTRY_READINESS,
            upstream_publication_fingerprint=None,
            synthetic_origin=True,
        )
    with pytest.raises(TacticalSignalContractError, match="semver"):
        SignalProvenance(
            upstream_pattern="documented_initial_return_depth",
            adapter_version=ADAPTER_VERSION,
            upstream_contract_version="latest",
            registry_readiness=REGISTRY_READINESS,
            upstream_publication_fingerprint=None,
            synthetic_origin=True,
        )


@pytest.mark.parametrize(
    "unsafe_value",
    [
        r"C:\Users\Name\secret",
        r"\\server\share\secret",
        "/home/name/secret",
        "file://secret",
        "../secret",
        "sequence_text",
        "4f17",
        "match_id",
        "point_number",
        "RogerFederer",
        "server",
        "point_winner",
        "returner_won_point",
        "2026-09-10",
        "test_metric",
        "evaluation",
        "scoring",
        "recommendation",
        "timestamp",
    ],
)
def test_closed_provenance_rejects_sensitive_and_operational_content(unsafe_value):
    with pytest.raises(TacticalSignalContractError):
        SignalProvenance(
            upstream_pattern=unsafe_value,
            adapter_version=ADAPTER_VERSION,
            upstream_contract_version="1.0.0",
            registry_readiness=REGISTRY_READINESS,
            upstream_publication_fingerprint=None,
            synthetic_origin=True,
        )


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "sequence_text",
        "raw_sequence",
        "match_id",
        "point_number",
        "player",
        "player_id",
        "server",
        "point_winner",
        "returner_won_point",
        "date",
        "test_score",
        "timestamp",
    ],
)
def test_closed_context_rejects_unknown_sensitive_fields(forbidden_key):
    with pytest.raises(TypeError):
        SignalContext(**{forbidden_key: "unsafe"})


def test_p09_preserves_every_reachable_component_reason_combination():
    expected = {
        "4q": (
            "documented_unknown_shot_type_q",
            "lateral_direction_not_documented",
            "return_depth_not_documented",
        ),
        "4q0": (
            "documented_unknown_shot_type_q",
            "documented_unknown_lateral_direction_0",
            "return_depth_not_documented",
        ),
        "4q00": (
            "documented_unknown_shot_type_q",
            "documented_unknown_lateral_direction_0",
            "documented_unknown_return_depth_0",
        ),
        "4q07": (
            "documented_unknown_shot_type_q",
            "documented_unknown_lateral_direction_0",
        ),
        "4q1": (
            "documented_unknown_shot_type_q",
            "return_depth_not_documented",
        ),
        "4q10": (
            "documented_unknown_shot_type_q",
            "documented_unknown_return_depth_0",
        ),
        "4q17": ("documented_unknown_shot_type_q",),
        "4f": (
            "lateral_direction_not_documented",
            "return_depth_not_documented",
        ),
        "4f0": (
            "documented_unknown_lateral_direction_0",
            "return_depth_not_documented",
        ),
        "4f00": (
            "documented_unknown_lateral_direction_0",
            "documented_unknown_return_depth_0",
        ),
        "4f07": ("documented_unknown_lateral_direction_0",),
        "4f1": ("return_depth_not_documented",),
        "4f10": ("documented_unknown_return_depth_0",),
    }
    for sequence, reasons in expected.items():
        classification, parsed = _p09(sequence)
        result = adapt_return_profile_signal(classification, parsed=parsed)
        assert result.signal is None
        assert result.reason_codes == reasons
        validate_signal_adaptation_result(result)


def test_p09_rejects_reordered_or_cross_state_reason_codes():
    classification, parsed = _p09("4q")
    result = adapt_return_profile_signal(classification, parsed=parsed)
    with pytest.raises(TacticalSignalContractError, match="combinacion upstream alcanzable"):
        replace(result, reason_codes=tuple(reversed(result.reason_codes)))
    with pytest.raises(TacticalSignalContractError):
        replace(
            result,
            upstream_state="initial_return_profile_not_documented",
            reason_codes=("lateral_direction_not_documented",),
            abstention_condition="unknown",
        )


def test_p09_rejects_ordered_but_unreachable_reason_combination():
    classification, parsed = _p09("4q00")
    result = adapt_return_profile_signal(classification, parsed=parsed)
    with pytest.raises(TacticalSignalContractError, match="combinacion upstream alcanzable"):
        replace(
            result,
            reason_codes=(
                "documented_unknown_shot_type_q",
                "lateral_direction_not_documented",
                "documented_unknown_return_depth_0",
            ),
        )


@pytest.mark.parametrize(
    ("sequence", "serve_number", "previous_fault", "state", "reason", "condition"),
    [
        ("+6f17", 1, False, "unknown_initial_return", "missing_or_ambiguous_service_prefix", "unknown"),
        ("6", 1, False, "unknown_initial_return", "truncated_before_initial_return_type", "unknown"),
        ("6F17", 1, False, "unknown_initial_return", "invalid_initial_return_type", "unknown"),
        ("6++f17", 1, False, "unknown_initial_return", "unsupported_modifier_before_initial_return_type", "unknown"),
        ("6f7", 1, False, "unknown_initial_return", "ambiguous_initial_return_component_order", "unknown"),
        ("5*", 1, False, "ineligible_censored", "censored_ace", "censored"),
        ("5#", 1, False, "ineligible_censored", "censored_unreturned_serve", "censored"),
        ("5n", 1, False, "ineligible_censored", "censored_service_fault", "censored"),
        ("5n", 2, True, "ineligible_censored", "censored_double_fault", "censored"),
        ("V", 1, False, "ineligible_censored", "censored_special_event", "censored"),
        ("cc", 1, False, "ineligible_censored", "censored_incomplete_let", "censored"),
    ],
)
def test_p09_all_single_reason_abstention_states_are_preserved(
    sequence, serve_number, previous_fault, state, reason, condition
):
    classification, parsed = _p09(sequence, serve_number, previous_fault)
    result = adapt_return_profile_signal(classification, parsed=parsed)
    assert result.signal is None
    assert result.upstream_state == state
    assert result.reason_codes == (reason,)
    assert result.abstention_condition == condition


@pytest.mark.parametrize(
    ("module_name", "adapter", "factory"),
    [
        ("src.analysis.return_direction_feasibility", adapt_return_direction_signal, _p04),
        ("src.analysis.return_depth_feasibility", adapt_return_depth_signal, _p05),
        ("src.analysis.return_shot_type_feasibility", adapt_return_shot_type_signal, _p06),
        ("src.analysis.return_profile_feasibility", adapt_return_profile_signal, _p09),
    ],
)
def test_adapters_never_rerun_upstream_parser(monkeypatch, module_name, adapter, factory):
    classification, parsed = factory("4f17")

    def forbidden_parser(*args, **kwargs):
        raise AssertionError("El adaptador no puede ejecutar parse_sequence.")

    monkeypatch.setattr(f"{module_name}.parse_sequence", forbidden_parser)
    result = adapter(classification, parsed=parsed)
    assert result.signal is not None


def test_profile_zero_components_abstain_and_never_become_categories():
    for sequence in ("4q17", "4f07", "4f10"):
        classification, parsed = _p09(sequence)
        result = adapt_return_profile_signal(classification, parsed=parsed)
        assert result.signal is None
        assert result.abstained
        payload = json.dumps(
            {
                "context": {
                    "shot": result.context.return_shot_type,
                    "direction": result.context.return_lateral_direction,
                    "depth": result.context.return_depth,
                }
            }
        )
        assert '"q"' not in payload


def test_p04_p05_p06_p09_preserve_upstream_reason_order_without_coercion():
    cases = []
    for adapter, factory in (
        (adapt_return_direction_signal, _p04),
        (adapt_return_depth_signal, _p05),
        (adapt_return_shot_type_signal, _p06),
        (adapt_return_profile_signal, _p09),
    ):
        classification, parsed = factory("4f17")
        result = adapter(classification, parsed=parsed)
        upstream = (
            (classification.reason_code.value,)
            if hasattr(classification, "reason_code")
            else tuple(reason.value for reason in classification.reason_codes)
        )
        assert result.reason_codes == upstream
        assert all(type(code) is str for code in result.reason_codes)


def test_no_signal_can_be_constructed_with_unknown_or_zero_value():
    signal = _signal("P04")
    with pytest.raises(TacticalSignalContractError):
        replace(
            signal,
            tactical_value="0",
            components=(("return_lateral_direction", "0"),),
            context=SignalContext(serve_number=1),
        )


def test_json_uses_compact_sorted_utf8_contract():
    signal = _signal("P06")
    payload = canonical_json_bytes(signal)
    assert b": " not in payload
    assert b", " not in payload
    assert payload == canonical_json_bytes(signal)
    assert len(tactical_signal_fingerprint(signal)) == 64
