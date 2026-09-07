"""Contrato sintético P06; no abre datos ni artefactos."""
from dataclasses import FrozenInstanceError, replace

import pytest

from src.analysis import return_depth_feasibility as p05
from src.analysis import return_direction_feasibility as p04
from src.analysis import return_shot_type_feasibility as p06
from src.parsing.serve_sequence import parse_sequence


EXPECTED_TYPES = {
    "f": ("derecha, excluidos slices y golpes especiales", "groundstroke"),
    "b": ("revés, excluidos slices y golpes especiales", "groundstroke"),
    "r": ("slice de derecha", "slice"),
    "s": ("slice de revés", "slice"),
    "v": ("volea de derecha", "volley"),
    "z": ("volea de revés", "volley"),
    "o": ("remate estándar", "overhead"),
    "p": ("remate de revés", "overhead"),
    "u": ("dejada de derecha", "drop_shot"),
    "y": ("dejada de revés", "drop_shot"),
    "l": ("globo de derecha", "lob"),
    "m": ("globo de revés", "lob"),
    "h": ("media volea de derecha", "half_volley"),
    "i": ("media volea de revés", "half_volley"),
    "j": ("volea liftada de derecha", "swinging_volley"),
    "k": ("volea liftada de revés", "swinging_volley"),
    "t": ("golpe especial, incluido trick shot o tweener", "special"),
}


@pytest.mark.parametrize("code", list(EXPECTED_TYPES))
def test_all_documented_types_are_observed_at_the_formal_cursor(code):
    description, family = EXPECTED_TYPES[code]
    result = p06.parse_and_classify_initial_return_shot_type(f"6{code}", 1)
    assert result.classification_state is p06.ReturnShotTypeState.OBSERVED
    assert result.return_shot_type_code == code
    assert result.documented_description == description
    assert result.shot_type_family == family
    assert result.reason_codes == (getattr(p06.ReturnShotTypeReason, code.upper()),)
    assert result.eligible_for_type_comparison is True
    assert result.return_event_observed is True
    assert result.actor == "returner"
    assert result.shot_type_span == result.return_event_span == (1, 2)


def test_documented_families_are_fixed_exhaustive_and_disjoint():
    expected = {
        "groundstroke": frozenset({"f", "b"}), "slice": frozenset({"r", "s"}),
        "volley": frozenset({"v", "z"}), "overhead": frozenset({"o", "p"}),
        "drop_shot": frozenset({"u", "y"}), "lob": frozenset({"l", "m"}),
        "half_volley": frozenset({"h", "i"}), "swinging_volley": frozenset({"j", "k"}),
        "special": frozenset({"t"}),
    }
    assert dict(p06.SHOT_TYPE_FAMILY_MEMBERS) == expected
    flattened = [code for codes in expected.values() for code in codes]
    assert set(flattened) == set(EXPECTED_TYPES)
    assert len(flattened) == len(set(flattened))
    assert p06.UNKNOWN_SHOT_TYPE_CODE not in p06.SHOT_TYPE_FAMILIES


def test_public_state_values_are_exact_and_have_no_short_aliases():
    assert {state.value for state in p06.ReturnShotTypeState} == {
        "return_shot_type_observed",
        "return_shot_type_unknown",
        "unknown_initial_return",
        "ineligible_censored",
    }


@pytest.mark.parametrize("sequence", ["6q", "6q27"])
def test_literal_q_is_localized_unknown_but_never_eligible(sequence):
    result = p06.parse_and_classify_initial_return_shot_type(sequence, 1)
    assert result.classification_state is p06.ReturnShotTypeState.UNKNOWN
    assert result.reason_codes == (p06.ReturnShotTypeReason.Q,)
    assert result.return_shot_type_code == "q"
    assert result.documented_description == "tipo de golpe desconocido"
    assert result.shot_type_family is None
    assert result.return_event_observed is True
    assert result.actor == "returner"
    assert result.eligible_for_type_comparison is False


@pytest.mark.parametrize("sequence,number,prior,reason,terminal", [
    ("5*", 1, False, p06.ReturnShotTypeReason.ACE, "ace"),
    ("5#", 1, False, p06.ReturnShotTypeReason.UNRETURNED, "unreturned_serve"),
    ("5n", 1, False, p06.ReturnShotTypeReason.FAULT, "service_fault"),
    ("5n", 2, True, p06.ReturnShotTypeReason.DOUBLE_FAULT, "double_fault"),
    ("cc", 1, False, p06.ReturnShotTypeReason.INCOMPLETE_LET, "incomplete_let"),
])
def test_pre_return_terminals_are_censored(sequence, number, prior, reason, terminal):
    result = p06.parse_and_classify_initial_return_shot_type(sequence, number, prior)
    assert result.classification_state is p06.ReturnShotTypeState.CENSORED
    assert result.reason_codes == (reason,)
    assert result.terminal_serve_outcome == terminal
    assert result.return_event_observed is False
    assert result.actor is None
    assert result.return_shot_type_code is None
    assert result.eligible_for_type_comparison is False


@pytest.mark.parametrize("sequence", ["P", "Q", "R", "S", "V"])
def test_all_supported_specials_are_censored_before_a_return(sequence):
    result = p06.parse_and_classify_initial_return_shot_type(sequence, 1)
    assert result.classification_state is p06.ReturnShotTypeState.CENSORED
    assert result.reason_codes == (p06.ReturnShotTypeReason.SPECIAL,)
    assert result.actor is None


def test_second_fault_requires_explicit_prior_fault_context():
    result = p06.parse_and_classify_initial_return_shot_type("5n", 2)
    assert result.reason_codes == (p06.ReturnShotTypeReason.FAULT,)
    with pytest.raises(ValueError, match="segundo saque"):
        p06.parse_and_classify_initial_return_shot_type("5n", 1, True)


def test_previous_fault_is_part_of_the_canonical_double_fault_result():
    result = p06.parse_and_classify_initial_return_shot_type("5n", 2, True)
    with pytest.raises((TypeError, ValueError)):
        p06.validate_return_shot_type_classification(
            replace(result, previous_attempt_was_fault=False)
        )


@pytest.mark.parametrize("sequence", [
    "6", "6+", "6++f", "6-f", "6=f", "6;f", "6^f", "6 f", "6?f", "6F",
])
def test_no_character_is_skipped_to_find_a_later_type(sequence):
    result = p06.parse_and_classify_initial_return_shot_type(sequence, 1)
    assert result.classification_state is p06.ReturnShotTypeState.UNKNOWN_INITIAL
    assert result.return_event_observed is False
    assert result.actor is None
    assert result.return_shot_type_code is None
    assert result.eligible_for_type_comparison is False


def test_uppercase_type_is_not_normalized():
    result = p06.parse_and_classify_initial_return_shot_type("6F", 1)
    assert result.classification_state is p06.ReturnShotTypeState.UNKNOWN_INITIAL
    assert result.return_shot_type_code is None
    assert result.documented_description is None


def test_initial_marker_lets_unknown_service_direction_and_spans_are_literal():
    result = p06.parse_and_classify_initial_return_shot_type("cc0+f270", 2)
    assert result.service_prefix_span == (0, 3)
    assert result.marker_literal == "+"
    assert result.marker_span == (3, 4)
    assert result.shot_type_span == result.return_event_span == (4, 5)
    assert result.return_shot_type_code == "f"
    assert result.classification_state is p06.ReturnShotTypeState.OBSERVED
    assert result.sequence_text[slice(*result.shot_type_span)] == "f"


@pytest.mark.parametrize("sequence", ["6f", "6f2", "6f27", "6f200", "6f2d@", "6f#", "6f27C"])
def test_lateral_depth_zero_errors_and_later_residual_do_not_erase_type(sequence):
    result = p06.parse_and_classify_initial_return_shot_type(sequence, 1)
    assert result.classification_state is p06.ReturnShotTypeState.OBSERVED
    assert result.return_shot_type_code == "f"
    assert result.shot_type_span == (1, 2)
    assert result.marker_span is None


def test_late_marker_is_not_reinterpreted_as_p03_marker():
    result = p06.parse_and_classify_initial_return_shot_type("6f+2", 1)
    assert result.return_shot_type_code == "f"
    assert result.marker_literal is None
    assert result.marker_span is None


def test_warnings_with_equal_codes_at_distinct_spans_are_preserved():
    result = p06.parse_and_classify_initial_return_shot_type("6f??", 1)
    assert result.return_shot_type_code == "f"
    assert result.parser_warning_codes.count("W_UNKNOWN_CHARACTER") == 2
    assert result.parser_warning_spans[-2:] == ((2, 3), (3, 4))
    assert result.parser_warning_spans == tuple(sorted(result.parser_warning_spans))


@pytest.mark.parametrize("value", ["1", 1.0, True, None, 3])
def test_serve_number_domain_is_strict(value):
    with pytest.raises((TypeError, ValueError)):
        p06.parse_and_classify_initial_return_shot_type("6f", value)


def test_low_level_rejects_crossed_parse_results_and_canonical_parse_mutations():
    with pytest.raises(ValueError, match="coincidir"):
        p06.classify_initial_return_shot_type("6f", 1, parse_sequence("5f", 1))
    parsed = parse_sequence("6f", 1)
    for altered in (replace(parsed, tokens=()), replace(parsed, tokens=tuple(reversed(parsed.tokens)))):
        with pytest.raises((TypeError, ValueError)):
            p06.classify_initial_return_shot_type("6f", 1, altered)


def test_high_level_parses_once_preserves_parse_result_and_is_deterministic(monkeypatch):
    original, calls = p06.parse_sequence, []

    def spy(text, number):
        parsed = original(text, number)
        calls.append(parsed)
        return parsed

    monkeypatch.setattr(p06, "parse_sequence", spy)
    first = p06.parse_and_classify_initial_return_shot_type("6+f27", 1)
    second = p06.parse_and_classify_initial_return_shot_type("6+f27", 1)
    assert first == second
    assert len(calls) == 2
    assert calls[0] == original("6+f27", 1)
    assert calls[1] == original("6+f27", 1)


def test_result_is_frozen_and_validator_rejects_adversarial_mutations():
    result = p06.parse_and_classify_initial_return_shot_type("cc6+f27??", 1)
    parsed = parse_sequence(result.sequence_text, result.serve_number)
    with pytest.raises(FrozenInstanceError):
        result.return_shot_type_code = "b"
    mutations = (
        replace(result, sequence_text="cc6+b27??"),
        replace(result, serve_number=2),
        replace(result, previous_attempt_was_fault=True),
        replace(result, classification_state=p06.ReturnShotTypeState.UNKNOWN),
        replace(result, reason_codes=(p06.ReturnShotTypeReason.Q,)),
        replace(result, eligible_for_type_comparison=False),
        replace(result, return_event_observed=False),
        replace(result, actor="server"),
        replace(result, return_shot_type_code="q"),
        replace(result, documented_description="changed"),
        replace(result, shot_type_family="slice"),
        replace(result, service_prefix_span=(0, 1)),
        replace(result, marker_literal=None),
        replace(result, marker_span=(0, 1)),
        replace(result, shot_type_span=(1, 2)),
        replace(result, return_event_span=(1, 2)),
        replace(result, parser_warning_codes=()),
        replace(result, parser_warning_spans=()),
        replace(result, residual_spans=()),
        replace(result, terminal_serve_outcome="ace"),
        replace(result, classification_contract_version="other"),
    )
    for mutated in mutations:
        with pytest.raises((TypeError, ValueError)):
            p06.validate_return_shot_type_classification(mutated, parsed=parsed)


def test_validator_rejects_semantically_impossible_unknown_and_censored_fields():
    unknown = p06.parse_and_classify_initial_return_shot_type("6q", 1)
    censored = p06.parse_and_classify_initial_return_shot_type("5*", 1)
    for mutated in (
        replace(unknown, return_shot_type_code="f", documented_description=EXPECTED_TYPES["f"][0]),
        replace(unknown, eligible_for_type_comparison=True),
        replace(censored, actor="returner", return_event_observed=True),
        replace(censored, shot_type_span=(0, 1), return_event_span=(0, 1)),
    ):
        with pytest.raises((TypeError, ValueError)):
            p06.validate_return_shot_type_classification(mutated)


def test_p06_isolated_extension_preserves_p04_p05_contracts_and_shared_spans():
    p06_only = p06.parse_and_classify_initial_return_shot_type("6f", 1)
    p04_only = p04.parse_and_classify_initial_return_direction("6f", 1)
    p05_only = p05.parse_and_classify_initial_return_depth("6f", 1)
    assert p06_only.classification_state is p06.ReturnShotTypeState.OBSERVED
    assert p04_only.classification_state is p04.ReturnDirectionState.UNKNOWN_INITIAL
    assert p05_only.classification_state is p05.ReturnDepthState.UNKNOWN_INITIAL

    p06_shared = p06.parse_and_classify_initial_return_shot_type("6f27", 1)
    p04_shared = p04.parse_and_classify_initial_return_direction("6f27", 1)
    p05_shared = p05.parse_and_classify_initial_return_depth("6f27", 1)
    assert p06_shared.shot_type_span == (
        p04_shared.return_event_start,
        p04_shared.return_event_start + 1,
    )
    assert p06_shared.shot_type_span == p05_shared.shot_type_span
    assert p06_shared.return_shot_type_code == p04_shared.return_shot_type_code == p05_shared.return_shot_type == "f"
    assert p06.parse_and_classify_initial_return_shot_type("6q27", 1).eligible_for_type_comparison is False


def test_module_has_no_io_or_productive_dependency_on_p04_or_p05():
    source = open(p06.__file__, encoding="utf-8").read()
    for forbidden in (
        "read_parquet", "read_csv", "pandas", "pyarrow", "DataFrame", "data/",
        "return_direction_feasibility", "return_depth_feasibility",
    ):
        assert forbidden not in source
