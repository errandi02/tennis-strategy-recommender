"""Contrato sintetico P05; no abre puntos, Parquet ni artefactos."""
from dataclasses import FrozenInstanceError, replace

import pytest

from src.analysis import return_depth_feasibility as p05
from src.analysis import return_direction_feasibility as p04
from src.parsing.serve_sequence import parse_sequence


@pytest.mark.parametrize("sequence,depth,label,span", [
    ("6f27", "7", "service_boxes", (3, 4)),
    ("6f28", "8", "behind_service_line_closer_to_service_line", (3, 4)),
    ("6f29", "9", "closer_to_baseline", (3, 4)),
    ("6f07", "7", "service_boxes", (3, 4)),
])
def test_documented_depths_are_localized_with_manual_spans(sequence, depth, label, span):
    result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert result.classification_state is p05.ReturnDepthState.OBSERVED
    assert result.return_depth_code == depth
    assert result.return_depth_label == label
    assert result.return_depth_span == span
    assert result.actor == "returner"
    assert result.eligible_for_depth_comparison is True


@pytest.mark.parametrize("sequence,depth", [
    ("6f17", "7"), ("6f28", "8"), ("6f39", "9"),
    ("6f07", "7"), ("6f08", "8"), ("6f09", "9"),
])
def test_depth_position_is_explicit_after_each_documented_lateral_code(sequence, depth):
    result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert result.lateral_direction_span == (2, 3)
    assert result.return_depth_span == (3, 4)
    assert result.return_depth_code == depth
    assert result.classification_state is p05.ReturnDepthState.OBSERVED


@pytest.mark.parametrize("lateral", ["1", "2", "3", "0"])
def test_all_lateral_codes_can_precede_documented_depth(lateral):
    result = p05.parse_and_classify_initial_return_depth(f"6f{lateral}8", 1)
    assert result.lateral_direction_code == lateral
    assert result.return_depth_code == "8"
    assert result.classification_state is p05.ReturnDepthState.OBSERVED


def test_contextual_zero_is_depth_only_after_lateral_position():
    unknown = p05.parse_and_classify_initial_return_depth("6f20", 1)
    later = p05.parse_and_classify_initial_return_depth("6f208", 1)
    residual = p05.parse_and_classify_initial_return_depth("6f270", 1)
    assert unknown.classification_state is p05.ReturnDepthState.UNKNOWN
    assert unknown.return_depth_code == "0"
    assert unknown.eligible_for_depth_comparison is False
    assert later.return_depth_code == "0" and later.return_depth_span == (3, 4)
    assert later.sequence_text[later.residual_spans[0][0]:] == "f208"
    assert residual.return_depth_code == "7" and residual.return_depth_span == (3, 4)


@pytest.mark.parametrize("sequence", ["6f10", "6f20", "6f30", "6f00"])
def test_zero_is_unknown_depth_only_in_the_formal_depth_position(sequence):
    result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert result.classification_state is p05.ReturnDepthState.UNKNOWN
    assert result.return_depth_code == "0"
    assert result.return_depth_span == (3, 4)
    assert result.eligible_for_depth_comparison is False


def test_multiple_zeros_are_disambiguated_only_by_the_local_cursor():
    result = p05.parse_and_classify_initial_return_depth("0f00", 1)
    assert result.service_prefix_span == (0, 1)
    assert result.lateral_direction_code == "0"
    assert result.return_depth_code == "0"
    assert result.return_depth_span == (3, 4)


def test_depth_without_lateral_is_never_reinterpreted():
    result = p05.parse_and_classify_initial_return_depth("6f7", 1)
    assert result.classification_state is p05.ReturnDepthState.UNKNOWN_INITIAL
    assert result.return_event_observed is False
    assert result.return_depth_code is None
    assert result.actor is None


@pytest.mark.parametrize("sequence", ["6f7", "6f8", "6f9", "67", "617"])
def test_depth_without_a_complete_documented_return_event_is_unknown(sequence):
    result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert result.classification_state is p05.ReturnDepthState.UNKNOWN_INITIAL
    assert result.return_depth_code is None
    assert result.return_event_observed is False


@pytest.mark.parametrize("sequence", ["6f2", "6f2d@", "0b3*"])
def test_optional_absent_depth_is_not_a_tactical_category(sequence):
    result = p05.parse_and_classify_initial_return_depth(sequence, 2)
    assert result.classification_state is p05.ReturnDepthState.NOT_DOCUMENTED
    assert result.return_event_observed is True
    assert result.return_depth_code is None
    assert result.eligible_for_depth_comparison is False


def test_first_and_second_serve_and_unknown_service_direction_do_not_change_actor():
    first = p05.parse_and_classify_initial_return_depth("0f17", 1)
    second = p05.parse_and_classify_initial_return_depth("0f17", 2)
    assert first.actor == second.actor == "returner"
    assert first.return_depth_code == second.return_depth_code == "7"
    assert first.service_prefix_span == second.service_prefix_span == (0, 1)


@pytest.mark.parametrize("sequence", ["6+f29", "c6f18", "cc6f18"])
def test_marker_and_lets_advance_the_formal_cursor(sequence):
    result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert result.return_depth_code in {"8", "9"}
    assert result.classification_state is p05.ReturnDepthState.OBSERVED
    if "+" in sequence:
        assert result.marker_literal == "+"
        assert result.marker_span == (1, 2)
    else:
        assert result.service_prefix_span[1] == sequence.index("f")


@pytest.mark.parametrize("sequence,number,prior,terminal", [
    ("5*", 1, False, "ace"), ("5#", 1, False, "unreturned_serve"),
    ("5n", 1, False, "service_fault"), ("5n", 2, True, "double_fault"),
    ("S", 1, False, "special_event"), ("cc", 1, False, "incomplete_let"),
])
def test_terminals_are_censored_before_any_return(sequence, number, prior, terminal):
    result = p05.parse_and_classify_initial_return_depth(sequence, number, prior)
    assert result.classification_state is p05.ReturnDepthState.CENSORED
    assert result.terminal_serve_outcome == terminal
    assert result.return_event_observed is False
    assert result.eligible_for_depth_comparison is False


def test_second_fault_without_prior_context_is_not_relabelled():
    result = p05.parse_and_classify_initial_return_depth("5n", 2)
    assert result.reason_codes == (p05.ReturnDepthReason.FAULT,)
    assert result.terminal_serve_outcome == "service_fault"


@pytest.mark.parametrize("sequence", ["P", "Q", "R", "S", "V"])
def test_all_parser_specials_are_censored(sequence):
    result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert result.classification_state is p05.ReturnDepthState.CENSORED
    assert result.return_event_observed is False


@pytest.mark.parametrize("sequence", ["6+", "6++f27", "6-f27", "6q27", "6 f27", "6f", "6f+27"])
def test_ambiguous_initial_event_is_unknown(sequence):
    result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert result.classification_state is p05.ReturnDepthState.UNKNOWN_INITIAL
    assert result.return_event_observed is False


def test_post_depth_errors_and_residual_do_not_erase_local_depth():
    result = p05.parse_and_classify_initial_return_depth("6f27d@", 1)
    assert result.return_depth_code == "7"
    assert result.classification_state is p05.ReturnDepthState.OBSERVED
    assert result.parser_warning_codes
    assert result.residual_spans


def test_late_marker_is_not_consumed_as_p03_marker():
    result = p05.parse_and_classify_initial_return_depth("6f2+7", 1)
    assert result.marker_literal is None
    assert result.classification_state is p05.ReturnDepthState.NOT_DOCUMENTED
    assert result.return_depth_code is None


def test_only_one_depth_position_is_consumed_and_later_depth_is_residual():
    result = p05.parse_and_classify_initial_return_depth("6f277", 1)
    assert result.return_depth_code == "7"
    assert result.return_depth_span == (3, 4)
    assert result.sequence_text[result.return_depth_span[1]:] == "7"
    assert result.residual_spans == ((1, 5),)


def test_low_level_rejects_crossed_parse_result_and_strict_domains():
    with pytest.raises(ValueError):
        p05.classify_initial_return_depth("6f27", 1, parse_sequence("5f27", 1))
    for value in ("1", 1.0, True, None, 3):
        with pytest.raises((TypeError, ValueError)):
            p05.parse_and_classify_initial_return_depth("6f27", value)
    with pytest.raises(ValueError):
        p05.parse_and_classify_initial_return_depth("6f27", 1, True)
    with pytest.raises(ValueError):
        p05.classify_initial_return_depth("6f27", 2, parse_sequence("6f27", 1))


def test_high_level_parses_once_and_parse_result_is_unchanged(monkeypatch):
    original, calls = p05.parse_sequence, []
    def spy(text, number):
        parsed = original(text, number)
        calls.append(parsed)
        return parsed
    monkeypatch.setattr(p05, "parse_sequence", spy)
    result = p05.parse_and_classify_initial_return_depth("6f27", 1)
    assert result.return_depth_code == "7"
    assert len(calls) == 1
    assert calls[0] == original("6f27", 1)


def test_validator_rejects_mutations_and_result_is_immutable():
    result = p05.parse_and_classify_initial_return_depth("6+f27d", 1)
    with pytest.raises(FrozenInstanceError):
        result.return_depth_code = "8"
    for mutation in (
        replace(result, return_depth_code="8"),
        replace(result, return_depth_span=(2, 3)),
        replace(result, actor=None),
        replace(result, eligible_for_depth_comparison=False),
        replace(result, reason_codes=(p05.ReturnDepthReason.DEPTH_0,)),
        replace(result, parser_warning_codes=()),
        replace(result, residual_spans=()),
    ):
        with pytest.raises((TypeError, ValueError)):
            p05.validate_return_depth_classification(mutation)


@pytest.mark.parametrize("mutation", [
    lambda result: replace(result, sequence_text="5f27"),
    lambda result: replace(result, serve_number="1"),
    lambda result: replace(result, classification_state=p05.ReturnDepthState.UNKNOWN),
    lambda result: replace(result, reason_codes=(p05.ReturnDepthReason.NO_DEPTH,)),
    lambda result: replace(result, actor="server"),
    lambda result: replace(result, return_shot_type="b"),
    lambda result: replace(result, lateral_direction_code="1"),
    lambda result: replace(result, return_depth_code="0"),
    lambda result: replace(result, return_depth_label="unknown"),
    lambda result: replace(result, return_depth_span=(2, 3)),
    lambda result: replace(result, marker_literal=None),
    lambda result: replace(result, marker_span=(0, 1)),
    lambda result: replace(result, terminal_serve_outcome="ace"),
    lambda result: replace(result, parser_warning_spans=()),
    lambda result: replace(result, residual_spans=()),
    lambda result: replace(result, return_event_observed=False),
])
def test_validator_rejects_each_semantic_field_mutation(mutation):
    result = p05.parse_and_classify_initial_return_depth("6+f27d", 1)
    with pytest.raises((TypeError, ValueError)):
        p05.validate_return_depth_classification(mutation(result))


def test_validator_rejects_fault_context_and_impossible_state_combinations():
    double_fault = p05.parse_and_classify_initial_return_depth("5n", 2, True)
    observed = p05.parse_and_classify_initial_return_depth("6f27", 1)
    unknown = p05.parse_and_classify_initial_return_depth("6f20", 1)
    censored = p05.parse_and_classify_initial_return_depth("5*", 1)
    mutations = (
        replace(double_fault, previous_attempt_was_fault=False),
        replace(observed, return_depth_code="0", return_depth_label="unknown"),
        replace(unknown, return_depth_code="7", return_depth_label=p05.RETURN_DEPTHS["7"]),
        replace(observed, lateral_direction_code=None, lateral_direction_label=None, lateral_direction_span=None),
        replace(censored, actor="returner", return_event_observed=True),
        replace(unknown, eligible_for_depth_comparison=True),
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)):
            p05.validate_return_depth_classification(mutation)


def test_warning_duplicates_at_distinct_spans_are_preserved_in_canonical_order():
    result = p05.parse_and_classify_initial_return_depth("6f27??", 1)
    assert result.parser_warning_codes.count("W_UNKNOWN_CHARACTER") == 2
    assert result.parser_warning_spans[-2:] == ((4, 5), (5, 6))
    assert result.parser_warning_spans == tuple(sorted(result.parser_warning_spans))


def test_spans_and_literals_are_exact_half_open_intervals():
    result = p05.parse_and_classify_initial_return_depth("cc6+f07", 1)
    assert result.service_prefix_span == (0, 3)
    assert result.marker_span == (3, 4)
    assert result.shot_type_span == (4, 5)
    assert result.lateral_direction_span == (5, 6)
    assert result.return_depth_span == (6, 7)
    assert result.return_event_span == (4, 7)
    assert result.sequence_text[slice(*result.return_depth_span)] == "7"


@pytest.mark.parametrize("sequence", ["6f17", "6f28", "6f39", "6f20"])
def test_p04_and_p05_share_spans_only_for_lateral_123(sequence):
    p04_result = p04.parse_and_classify_initial_return_direction(sequence, 1)
    p05_result = p05.parse_and_classify_initial_return_depth(sequence, 1)
    assert p04_result.return_depth_code == p05_result.return_depth_code
    assert (p04_result.return_depth_start, p04_result.return_depth_end) == p05_result.return_depth_span


def test_p05_extends_contextual_lateral_zero_without_modifying_p04():
    before = p04.parse_and_classify_initial_return_direction("6f07", 1)
    p05_result = p05.parse_and_classify_initial_return_depth("6f07", 1)
    after = p04.parse_and_classify_initial_return_direction("6f07", 1)
    assert before == after
    assert before.return_depth_code is None
    assert p05_result.return_depth_code == "7"


def test_static_scope_has_no_io_data_or_p04_dependency():
    source = open(p05.__file__, encoding="utf-8").read()
    for forbidden in ("read_parquet", "read_csv", "pandas", "pyarrow", "DataFrame", "data/", "write_", "return_direction_feasibility"):
        assert forbidden not in source
