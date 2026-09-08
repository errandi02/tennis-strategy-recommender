from dataclasses import replace

import pytest

from src.analysis import return_terminal_feasibility as p07
from src.analysis import return_depth_feasibility as p05
from src.analysis import return_direction_feasibility as p04
from src.analysis import return_shot_type_feasibility as p06
from src.analysis import serve_and_volley_feasibility as p03
from src.parsing.serve_sequence import parse_sequence, serialize_parse_result


def classify(sequence: str, serve_number: int = 1, previous_fault: bool = False):
    return p07.parse_and_classify_initial_return_terminal(
        sequence, serve_number, previous_fault
    )


@pytest.mark.parametrize(
    ("sequence", "state", "reason", "kind", "event", "terminal"),
    [
        ("6f*", p07.ReturnTerminalState.WINNER, p07.ReturnTerminalReason.WINNER, "winner", (1, 2), (2, 3)),
        ("6f#", p07.ReturnTerminalState.FORCED_ERROR, p07.ReturnTerminalReason.FORCED_ERROR, "forced_error", (1, 2), (2, 3)),
        ("6f2d@", p07.ReturnTerminalState.UNFORCED_ERROR, p07.ReturnTerminalReason.UNFORCED_ERROR_D, "unforced_error", (1, 3), (4, 5)),
        ("c4+b27#", p07.ReturnTerminalState.FORCED_ERROR, p07.ReturnTerminalReason.FORCED_ERROR, "forced_error", (3, 6), (6, 7)),
        ("0q0!@", p07.ReturnTerminalState.UNFORCED_ERROR, p07.ReturnTerminalReason.UNFORCED_ERROR_BANG, "unforced_error", (1, 3), (4, 5)),
    ],
)
def test_documented_immediate_terminals_are_local_and_literal(sequence, state, reason, kind, event, terminal):
    result = classify(sequence)
    assert result.classification_state is state
    assert result.reason_code is reason
    assert result.terminal_kind == kind
    assert result.terminal_literal == sequence[terminal[0]:terminal[1]]
    assert result.return_event_span == event
    assert result.terminal_span == terminal
    assert result.actor == "returner"
    assert result.eligible_for_terminal_comparison is True
    assert result.sequence_text[result.terminal_span[0]:result.terminal_span[1]] in {"*", "#", "@"}


def test_unforced_error_preserves_error_span_and_q_is_not_recast_as_type_category():
    result = classify("6q2d@")
    assert result.return_shot_type_code == "q"
    assert result.return_event_observed is True
    assert result.error_code == "d"
    assert result.error_span == (3, 4)
    assert result.terminal_span == (4, 5)


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("n", p07.ReturnTerminalReason.UNFORCED_ERROR_N),
        ("w", p07.ReturnTerminalReason.UNFORCED_ERROR_W),
        ("d", p07.ReturnTerminalReason.UNFORCED_ERROR_D),
        ("x", p07.ReturnTerminalReason.UNFORCED_ERROR_X),
        ("e", p07.ReturnTerminalReason.UNFORCED_ERROR_E),
        ("!", p07.ReturnTerminalReason.UNFORCED_ERROR_BANG),
    ],
)
def test_all_and_only_documented_unforced_error_codes_are_observed(code, reason):
    result = classify(f"6f2{code}@")
    assert result.classification_state is p07.ReturnTerminalState.UNFORCED_ERROR
    assert result.reason_code is reason
    assert result.error_code == code
    assert result.terminal_literal == "@"


def test_second_serve_uses_explicit_prior_fault_context_only():
    result = classify("6f*", 2, True)
    assert result.classification_state is p07.ReturnTerminalState.WINNER
    assert result.previous_attempt_was_fault is True
    with pytest.raises(ValueError, match="Solo un segundo saque"):
        classify("6f*", 1, True)


@pytest.mark.parametrize(
    ("sequence", "serve_number", "previous_fault", "reason", "terminal"),
    [
        ("6*", 1, False, p07.ReturnTerminalReason.ACE, "ace"),
        ("6#", 1, False, p07.ReturnTerminalReason.UNRETURNED, "unreturned_serve"),
        ("6n", 1, False, p07.ReturnTerminalReason.FAULT, "service_fault"),
        ("6n", 2, True, p07.ReturnTerminalReason.DOUBLE_FAULT, "double_fault"),
        ("S", 1, False, p07.ReturnTerminalReason.SPECIAL, "special_event"),
        ("R", 1, False, p07.ReturnTerminalReason.SPECIAL, "special_event"),
        ("P", 1, False, p07.ReturnTerminalReason.SPECIAL, "special_event"),
        ("Q", 1, False, p07.ReturnTerminalReason.SPECIAL, "special_event"),
        ("V", 1, False, p07.ReturnTerminalReason.SPECIAL, "special_event"),
        ("cc", 1, False, p07.ReturnTerminalReason.INCOMPLETE_LET, "incomplete_let"),
    ],
)
def test_pre_return_terminals_are_censored(sequence, serve_number, previous_fault, reason, terminal):
    result = classify(sequence, serve_number, previous_fault)
    assert result.classification_state is p07.ReturnTerminalState.CENSORED
    assert result.reason_code is reason
    assert result.terminal_serve_outcome == terminal
    assert result.return_event_observed is False
    assert result.actor is None
    assert result.eligible_for_terminal_comparison is False


@pytest.mark.parametrize(
    ("sequence", "state", "reason"),
    [
        ("6f27b1*", p07.ReturnTerminalState.NOT_DOCUMENTED, p07.ReturnTerminalReason.NONTERMINAL_CONTENT),
        ("6b29C", p07.ReturnTerminalState.NOT_DOCUMENTED, p07.ReturnTerminalReason.CHALLENGE_AFTER_RETURN),
        ("6f", p07.ReturnTerminalState.NOT_DOCUMENTED, p07.ReturnTerminalReason.NO_TERMINAL),
        ("6f2", p07.ReturnTerminalState.NOT_DOCUMENTED, p07.ReturnTerminalReason.NO_TERMINAL),
        ("6f27", p07.ReturnTerminalState.NOT_DOCUMENTED, p07.ReturnTerminalReason.NO_TERMINAL),
        ("6f@", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.COMPONENT_ORDER),
        ("6f2g@", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.ERROR_FORM),
        ("6f7*", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.COMPONENT_ORDER),
        ("6f+2*", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.COMPONENT_ORDER),
        ("6f2d@x", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.TRAILING_CONTENT),
        ("6f* ", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.TRAILING_CONTENT),
        ("6f*x", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.TRAILING_CONTENT),
        ("6f##", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.TRAILING_CONTENT),
        ("6f22*", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.BOUNDARY),
        ("6f277*", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.BOUNDARY),
        ("6f200#", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.BOUNDARY),
        ("6+", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.TRUNCATED),
        ("6++f#", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.MODIFIER),
        ("6?f#", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.BOUNDARY),
        ("6F#", p07.ReturnTerminalState.UNKNOWN_INITIAL, p07.ReturnTerminalReason.BOUNDARY),
    ],
)
def test_no_terminal_is_invented_from_later_or_ambiguous_content(sequence, state, reason):
    result = classify(sequence)
    assert result.classification_state is state
    assert result.reason_code is reason
    assert result.eligible_for_terminal_comparison is False
    if state is p07.ReturnTerminalState.NOT_DOCUMENTED:
        assert result.return_event_observed is True
        assert result.actor == "returner"


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_serve_number_domain_is_strict(value):
    with pytest.raises(ValueError):
        classify("6f*", value)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_fault_context_domain_is_strict(value):
    with pytest.raises(TypeError):
        p07.parse_and_classify_initial_return_terminal("6f*", 1, value)


@pytest.mark.parametrize("value", [None, b"6f*", 6, True])
def test_sequence_text_domain_is_strict(value):
    with pytest.raises(TypeError):
        p07.parse_and_classify_initial_return_terminal(value, 1)


def test_low_level_rejects_foreign_parse_result_and_preserves_parse_once(monkeypatch):
    with pytest.raises(ValueError, match="deben coincidir"):
        p07.classify_initial_return_terminal("6f*", 1, parse_sequence("5f*", 1))
    with pytest.raises(ValueError, match="deben coincidir"):
        p07.classify_initial_return_terminal("6f*", 1, parse_sequence("6f*", 2))

    calls = []
    original = p07.parse_sequence

    def counted(raw, serve_number):
        calls.append((raw, serve_number))
        return original(raw, serve_number)

    monkeypatch.setattr(p07, "parse_sequence", counted)
    result = p07.parse_and_classify_initial_return_terminal("6f2d@", 1)
    assert result.classification_state is p07.ReturnTerminalState.UNFORCED_ERROR
    assert calls == [("6f2d@", 1)]


def test_parse_result_is_not_mutated_by_classification():
    parsed = parse_sequence("6f27#", 1)
    before = serialize_parse_result(parsed)
    result = p07.classify_initial_return_terminal("6f27#", 1, parsed)
    assert result.classification_state is p07.ReturnTerminalState.FORCED_ERROR
    assert serialize_parse_result(parsed) == before


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, sequence_text="5f2d@"),
        lambda value: replace(value, serve_number=2),
        lambda value: replace(value, classification_state=p07.ReturnTerminalState.WINNER),
        lambda value: replace(value, reason_code=p07.ReturnTerminalReason.WINNER),
        lambda value: replace(value, eligible_for_terminal_comparison=False),
        lambda value: replace(value, actor=None),
        lambda value: replace(value, service_prefix_span=(0, 2)),
        lambda value: replace(value, shot_type_span=(2, 3)),
        lambda value: replace(value, return_shot_type_code="b"),
        lambda value: replace(value, lateral_direction_span=(3, 4)),
        lambda value: replace(value, lateral_direction_code="3"),
        lambda value: replace(value, return_depth_span=(2, 3)),
        lambda value: replace(value, return_depth_code="7"),
        lambda value: replace(value, terminal_kind="winner"),
        lambda value: replace(value, terminal_literal="#"),
        lambda value: replace(value, terminal_span=(3, 4)),
        lambda value: replace(value, error_span=(2, 3)),
        lambda value: replace(value, error_code="n"),
        lambda value: replace(value, return_event_span=(1, 2)),
        lambda value: replace(value, marker_span=(0, 1)),
        lambda value: replace(value, parser_warning_codes=()),
        lambda value: replace(value, parser_warning_codes=tuple(reversed(value.parser_warning_codes)), parser_warning_spans=tuple(reversed(value.parser_warning_spans))),
        lambda value: replace(value, parser_warning_spans=value.parser_warning_spans + (value.parser_warning_spans[0],)),
        lambda value: replace(value, residual_spans=()),
        lambda value: replace(value, residual_spans=value.residual_spans + (value.residual_spans[0],)),
        lambda value: replace(value, terminal_serve_outcome="ace"),
        lambda value: replace(value, previous_attempt_was_fault=True),
        lambda value: replace(value, classification_contract_version="broken"),
    ],
)
def test_validator_reconstructs_semantics_and_rejects_mutations(change):
    result = classify("6f2d@")
    with pytest.raises((TypeError, ValueError)):
        p07.validate_return_terminal_classification(change(result), parse_sequence("6f2d@", 1))


def test_result_is_immutable_and_no_public_outcome_field_exists():
    result = classify("6f#")
    with pytest.raises(Exception):
        result.actor = "server"
    for forbidden in ("point_winner", "server", "returner_won_point"):
        assert forbidden not in p07.ReturnTerminalClassification.__dataclass_fields__


def test_contract_does_not_depend_on_p04_p05_or_p06_results():
    result = classify("6f27#")
    assert result.shot_type_span == (1, 2)
    assert result.lateral_direction_span == (2, 3)
    assert result.return_depth_span == (3, 4)
    assert result.return_event_span == (1, 4)
    assert result.classification_state is p07.ReturnTerminalState.FORCED_ERROR


def test_p03_to_p06_spans_are_preserved_without_productive_dependencies():
    p07_winner = classify("6f*")
    p06_type = p06.parse_and_classify_initial_return_shot_type("6f*", 1)
    assert (p07_winner.return_shot_type_code, p07_winner.shot_type_span) == (p06_type.return_shot_type_code, p06_type.shot_type_span)

    p07_error = classify("6f2d@")
    p04_direction = p04.parse_and_classify_initial_return_direction("6f2d@", 1)
    assert p07_error.return_event_span[0] == p04_direction.return_event_start
    assert p07_error.lateral_direction_span == (p04_direction.lateral_direction_start, p04_direction.lateral_direction_end)

    p07_depth = classify("6f27*")
    p05_depth = p05.parse_and_classify_initial_return_depth("6f27*", 1)
    assert (p07_depth.shot_type_span, p07_depth.lateral_direction_span, p07_depth.return_depth_span) == (p05_depth.shot_type_span, p05_depth.lateral_direction_span, p05_depth.return_depth_span)

    p07_zero = classify("6f07#")
    p05_zero = p05.parse_and_classify_initial_return_depth("6f07#", 1)
    assert p07_zero.return_depth_span == p05_zero.return_depth_span

    p07_marker = classify("4+b27#")
    p03_marker = p03.parse_and_classify_serve_and_volley_attempt("4+b27#", 1)
    assert p07_marker.marker_span == (p03_marker.marker_start, p03_marker.marker_end)
