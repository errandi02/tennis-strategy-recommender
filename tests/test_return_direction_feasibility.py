"""Contrato sintético de P04; no abre datos ni artefactos."""
from dataclasses import FrozenInstanceError, replace

import pytest

from src.analysis import return_direction_feasibility as p04
from src.parsing.serve_sequence import parse_sequence


@pytest.mark.parametrize("sequence,direction,span", [
    ("6f1", "1", (2, 3)), ("6b2d@", "2", (2, 3)), ("4r3", "3", (2, 3)),
])
def test_documented_lateral_directions_have_manual_spans(sequence, direction, span):
    result = p04.parse_and_classify_initial_return_direction(sequence, 1)
    assert result.classification_state is p04.ReturnDirectionState.OBSERVED
    assert result.lateral_direction_code == direction
    assert (result.lateral_direction_start, result.lateral_direction_end) == span
    assert result.return_actor == "returner"
    assert result.eligible_for_direction_analysis is True


@pytest.mark.parametrize("shot", list("fbrsvzopuy lmhijkt".replace(" ", "")))
def test_all_documented_shot_types_can_anchor_a_local_return(shot):
    result = p04.parse_and_classify_initial_return_direction(f"5{shot}2", 1)
    assert result.return_shot_type_code == shot
    assert result.lateral_direction_code == "2"


def test_official_examples_and_depth_are_not_direction():
    unknown = p04.parse_and_classify_initial_return_direction("6f#", 1)
    observed = p04.parse_and_classify_initial_return_direction("6f2d@", 1)
    depth = p04.parse_and_classify_initial_return_direction("6f37", 1)
    assert unknown.classification_state is p04.ReturnDirectionState.UNKNOWN
    assert observed.lateral_direction_code == "2"
    assert depth.lateral_direction_code == "3"
    assert depth.return_depth_code == "7"
    assert depth.return_depth_label == "service_boxes"


@pytest.mark.parametrize("sequence,reason", [("6f0", p04.ReturnDirectionReason.D0), ("6f#", p04.ReturnDirectionReason.NO_DIRECTION), ("6f7", p04.ReturnDirectionReason.BOUNDARY), ("6q2", p04.ReturnDirectionReason.UNKNOWN_SHOT), ("6f", p04.ReturnDirectionReason.TRUNCATED)])
def test_unknown_or_invalid_local_events_are_not_eligible(sequence, reason):
    result = p04.parse_and_classify_initial_return_direction(sequence, 1)
    assert result.reason_code is reason
    assert result.eligible_for_direction_analysis is False


def test_initial_p03_annotation_is_consumed_but_late_annotation_is_not():
    initial = p04.parse_and_classify_initial_return_direction("4+f2", 1)
    late = p04.parse_and_classify_initial_return_direction("4f+2", 1)
    assert initial.initial_annotation_code == "+"
    assert initial.initial_annotation_start == 1
    assert initial.return_event_start == 2
    assert late.classification_state is p04.ReturnDirectionState.UNKNOWN_INITIAL


@pytest.mark.parametrize("sequence,number,prior,state,terminal", [
    ("5*", 1, False, p04.ReturnDirectionState.ACE, "ace"),
    ("5#", 1, False, p04.ReturnDirectionState.UNRETURNED, "unreturned_serve"),
    ("5n", 1, False, p04.ReturnDirectionState.FAULT, "service_fault"),
    ("5n", 2, True, p04.ReturnDirectionState.FAULT, "double_fault"),
    ("S", 1, False, p04.ReturnDirectionState.SPECIAL, "special_event"),
    ("cc", 1, False, p04.ReturnDirectionState.SPECIAL, "incomplete_let"),
])
def test_terminals_never_create_a_return(sequence, number, prior, state, terminal):
    result = p04.parse_and_classify_initial_return_direction(sequence, number, previous_attempt_was_fault=prior)
    assert result.classification_state is state
    assert result.terminal_serve_outcome == terminal
    assert result.return_event_observed is False
    assert result.return_actor is None


@pytest.mark.parametrize("value", ["1", 1.0, True, None, 3])
def test_serve_number_domain_is_strict(value):
    with pytest.raises((TypeError, ValueError)):
        p04.parse_and_classify_initial_return_direction("6f2", value)


def test_parse_result_mismatch_and_semantic_mutations_fail():
    parsed = parse_sequence("6f2", 1)
    with pytest.raises(ValueError): p04.classify_initial_return_direction("5f2", 1, parsed)
    result = p04.parse_and_classify_initial_return_direction("6f2", 1)
    for mutation in (
        replace(result, lateral_direction_code="7"),
        replace(result, classification_state=p04.ReturnDirectionState.OBSERVED, lateral_direction_code="0"),
        replace(result, eligible_for_direction_analysis=False),
        replace(result, return_actor=None),
    ):
        with pytest.raises((TypeError, ValueError)): p04.validate_return_direction_classification(mutation)


def test_no_data_or_regex_scope_in_module():
    source = open(p04.__file__, encoding="utf-8").read()
    for forbidden in ("read_parquet", "pandas", "pyarrow", "duckdb", "polars", "argparse", "DataFrame", "data/"):
        assert forbidden not in source


@pytest.mark.parametrize("depth,label", [
    ("7", "service_boxes"),
    ("8", "behind_service_line_closer_to_service_line"),
    ("9", "closer_to_baseline"),
    ("0", "unknown"),
])
def test_depth_is_contextual_and_never_lateral(depth, label):
    result = p04.parse_and_classify_initial_return_direction(f"6f2{depth}", 1)
    assert result.lateral_direction_code == "2"
    assert result.return_depth_code == depth
    assert result.return_depth_label == label
    assert result.eligible_for_direction_analysis is True


@pytest.mark.parametrize("sequence", ["6f7", "6f72", "6f82", "6+", "6++f2", "6 f2"])
def test_depth_before_direction_and_boundary_noise_are_not_accepted(sequence):
    result = p04.parse_and_classify_initial_return_direction(sequence, 1)
    assert result.classification_state is p04.ReturnDirectionState.UNKNOWN_INITIAL
    assert result.return_actor is None
    assert result.eligible_for_direction_analysis is False


@pytest.mark.parametrize("prefix", ["4", "5", "6", "0", "cc6"])
def test_prefixes_lets_and_second_serve_preserve_the_local_cursor(prefix):
    result = p04.parse_and_classify_initial_return_direction(f"{prefix}+b3", 2)
    assert result.serve_number == 2
    assert result.initial_annotation_code == "+"
    assert result.return_event_start == len(prefix) + 1
    assert result.return_event_end == len(prefix) + 3
    assert result.return_shot_type_code == "b"
    assert result.lateral_direction_code == "3"


def test_second_fault_without_context_is_not_relabelled_and_first_fault_context_rejected():
    single = p04.parse_and_classify_initial_return_direction("5n", 2)
    assert single.reason_code is p04.ReturnDirectionReason.FAULT
    assert single.terminal_serve_outcome == "service_fault"
    with pytest.raises(ValueError, match="segundo saque"):
        p04.parse_and_classify_initial_return_direction("5n", 1, previous_attempt_was_fault=True)


@pytest.mark.parametrize("sequence", ["P", "Q", "R", "S", "V"])
def test_specials_penalties_and_challenge_are_censored(sequence):
    result = p04.parse_and_classify_initial_return_direction(sequence, 1)
    assert result.classification_state is p04.ReturnDirectionState.SPECIAL
    assert result.return_event_observed is False
    assert result.terminal_serve_outcome == "special_event"


def test_only_the_first_local_event_is_extracted_and_not_a_later_volley():
    result = p04.parse_and_classify_initial_return_direction("6f2v3", 1)
    assert (result.return_event_start, result.return_event_end) == (1, 3)
    assert result.return_shot_type_code == "f"
    assert result.lateral_direction_code == "2"
    assert result.return_actor == "returner"


def test_parser_contract_rejects_missing_or_disordered_tokens_before_p04():
    parsed = parse_sequence("6f2", 1)
    for broken in (replace(parsed, tokens=()), replace(parsed, tokens=tuple(reversed(parsed.tokens)))):
        with pytest.raises((TypeError, ValueError)):
            p04.classify_initial_return_direction("6f2", 1, broken)


def test_deep_immutability_and_semantic_validator_rejects_individual_contract_breaks():
    result = p04.parse_and_classify_initial_return_direction("6+f27", 1)
    with pytest.raises(FrozenInstanceError):
        result.lateral_direction_code = "1"
    mutations = (
        replace(result, lateral_direction_label="crosscourt"),
        replace(result, return_depth_code="2"),
        replace(result, return_depth_start=result.return_depth_start + 1),
        replace(result, return_event_start=result.return_event_start + 1),
        replace(result, initial_annotation_start=0),
        replace(result, return_opportunity_observed=False),
        replace(result, parser_warning_codes=("bad",), parser_warning_spans=((0, 1),)),
        replace(result, classification_contract_version="other"),
    )
    for mutated in mutations:
        with pytest.raises((TypeError, ValueError)):
            p04.validate_return_direction_classification(mutated)


def test_unknown_local_return_has_actor_but_unknown_initial_and_censorship_do_not():
    localized = p04.parse_and_classify_initial_return_direction("6f#", 1)
    unknown_initial = p04.parse_and_classify_initial_return_direction("6?f2", 1)
    censored = p04.parse_and_classify_initial_return_direction("6*", 1)
    assert localized.return_actor == "returner"
    assert localized.classification_state is p04.ReturnDirectionState.UNKNOWN
    assert unknown_initial.return_actor is None
    assert censored.return_actor is None


def test_outcome_fields_are_not_part_of_the_p04_api_or_source():
    assert "point_winner" not in p04.ReturnDirectionClassification.__dataclass_fields__
    source = open(p04.__file__, encoding="utf-8").read()
    for forbidden in ("point_winner", "server_won_point", "returner_won_point"):
        assert forbidden not in source


@pytest.mark.parametrize("text,foreign", [
    ("6f2d@", "6f3d@"),
    ("4+f37", "4f37"),
    ("cc6f2", "c6f2"),
])
def test_low_level_rejects_parse_results_from_another_text(text, foreign):
    with pytest.raises(ValueError, match="coincidir"):
        p04.classify_initial_return_direction(text, 1, parse_sequence(foreign, 1))


def test_low_level_rejects_literal_span_and_residual_manipulations():
    parsed = parse_sequence("6f2d@", 1)
    altered_literal = replace(parsed, tokens=(replace(parsed.tokens[0], raw_text="5"), *parsed.tokens[1:]))
    altered_residual = replace(parsed, residual_spans=())
    for broken in (altered_literal, altered_residual):
        with pytest.raises((TypeError, ValueError)):
            p04.classify_initial_return_direction("6f2d@", 1, broken)


def test_result_validator_reconstructs_text_parse_warnings_and_residuals():
    result = p04.parse_and_classify_initial_return_direction("6f2d@", 1)
    for mutated in (
        replace(result, sequence_text="6f3d@"),
        replace(result, has_residual=False),
        replace(result, residual_spans=()),
        replace(result, parser_warning_codes=()),
        replace(result, return_shot_type_code="b", return_shot_type_label="backhand"),
        replace(result, lateral_direction_code="3", lateral_direction_label=p04.LATERAL_DIRECTIONS["3"], reason_code=p04.ReturnDirectionReason.D3),
    ):
        with pytest.raises((TypeError, ValueError)):
            p04.validate_return_direction_classification(mutated)


def test_high_level_parses_once_and_does_not_mutate_the_parse_result(monkeypatch):
    original = p04.parse_sequence
    calls = []
    def spy(text, number):
        parsed = original(text, number)
        calls.append(parsed)
        return parsed
    monkeypatch.setattr(p04, "parse_sequence", spy)
    result = p04.parse_and_classify_initial_return_direction("6f2", 1)
    assert result.lateral_direction_code == "2"
    assert len(calls) == 1
    assert calls[0] == original("6f2", 1)


@pytest.mark.parametrize("sequence,expected_depth", [
    ("6f07", None), ("6f08", None), ("6f09", None), ("6f00", None),
])
def test_contextual_zero_is_lateral_only_at_the_first_post_shot_position(sequence, expected_depth):
    result = p04.parse_and_classify_initial_return_direction(sequence, 1)
    assert result.classification_state is p04.ReturnDirectionState.UNKNOWN
    assert result.reason_code is p04.ReturnDirectionReason.D0
    assert result.lateral_direction_code == "0"
    assert result.return_depth_code is expected_depth
    assert result.eligible_for_direction_analysis is False


def test_unknown_service_direction_does_not_block_an_unequivocal_return_direction():
    result = p04.parse_and_classify_initial_return_direction("0f1", 1)
    assert result.service_direction_code == "0"
    assert result.service_direction_label == "unknown"
    assert result.lateral_direction_code == "1"
    assert result.classification_state is p04.ReturnDirectionState.OBSERVED


def test_contextual_terminators_after_the_return_are_not_service_terminals():
    for sequence in ("6f1*", "6f2d@", "6f37#"):
        result = p04.parse_and_classify_initial_return_direction(sequence, 1)
        assert result.classification_state is p04.ReturnDirectionState.OBSERVED
        assert result.terminal_serve_outcome is None


def test_double_fault_contract_cannot_be_forged_without_second_serve_and_context():
    result = p04.parse_and_classify_initial_return_direction("5n", 2, previous_attempt_was_fault=True)
    for mutated in (
        replace(result, previous_attempt_was_fault=False),
        replace(result, serve_number=1),
    ):
        with pytest.raises((TypeError, ValueError)):
            p04.validate_return_direction_classification(mutated)


def test_static_scope_excludes_all_forbidden_input_output_and_analysis_interfaces():
    source = open(p04.__file__, encoding="utf-8").read()
    for forbidden in ("read_csv", "open(", "pathlib", "write_", "argparse", "DataFrame", "score", "recommend"):
        assert forbidden not in source
