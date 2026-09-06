"""Pruebas sintéticas del extractor P03; no acceden a datos ni artefactos."""

from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from src.analysis import serve_and_volley_feasibility as p03
from src.parsing.serve_sequence import ParseResult, Token, parse_sequence, serialize_parse_result


def classify(raw: str, serve_number: int, *, prior_fault: bool = False):
    parsed = parse_sequence(raw, serve_number)
    return p03.classify_serve_and_volley_attempt(
        raw, serve_number, parsed, previous_attempt_was_fault=prior_fault
    )


@pytest.mark.parametrize(
    ("raw", "serve_number", "direction", "label", "prefix_end"),
    [
        ("4+b27v1*", 1, "4", "wide", 1),
        ("5+b27v1*", 1, "5", "body", 1),
        ("6+b27v1*", 2, "6", "T", 1),
        ("c5+b27v1*", 1, "5", "body", 2),
        ("cc4+b27v1*", 1, "4", "wide", 3),
    ],
)
def test_immediate_marker_is_literal_and_has_exact_span(raw, serve_number, direction, label, prefix_end):
    result = classify(raw, serve_number)
    assert result.analysis_state is p03.AnalysisState.POSITIVE_TAGGED
    assert result.reason_code is p03.ReasonCode.EXPLICIT_MARKER
    assert result.explicit_intent_tagged is True
    assert (result.marker_start, result.marker_end) == (prefix_end, prefix_end + 1)
    assert result.marker_literal == "+"
    assert (result.service_prefix_start, result.service_prefix_end) == (0, prefix_end)
    assert (result.service_direction_code, result.service_direction_label) == (direction, label)
    assert result.eligible_for_outcome_comparison is True


@pytest.mark.parametrize("raw", ["4b+27v1*", "4f2v1+", "4b++", "4f2+v1*"])
def test_late_or_noninitial_marker_is_unknown_not_a_negative_p03_label(raw):
    result = classify(raw, 1)
    assert result.analysis_state is p03.AnalysisState.UNKNOWN
    assert result.reason_code is p03.ReasonCode.AMBIGUOUS_POST_PREFIX
    assert result.explicit_intent_tagged is False
    assert result.marker_start is result.marker_end is None
    assert result.eligible_for_outcome_comparison is False


def test_only_the_immediate_marker_counts_when_multiple_exist():
    result = classify("4+b+27v1*", 1)
    assert result.explicit_intent_tagged is True
    assert (result.marker_start, result.marker_end) == (1, 2)


@pytest.mark.parametrize("raw", ["4v1*", "4z1*", "4h1*", "4i1*", "4j1*", "4k1*", "4o1*", "4p1*"])
def test_lexical_net_shot_codes_are_unknown_without_structured_rally_content(raw):
    result = classify(raw, 1)
    assert result.analysis_state is p03.AnalysisState.UNKNOWN
    assert result.reason_code is p03.ReasonCode.AMBIGUOUS_POST_PREFIX
    assert result.explicit_intent_tagged is False
    assert result.eligible_for_outcome_comparison is False


@pytest.mark.parametrize(
    ("raw", "terminal", "reason"),
    [
        ("4*", "ace", p03.ReasonCode.CENSORED_ACE),
        ("5#", "unreturned_serve", p03.ReasonCode.CENSORED_UNRETURNED),
        ("S", "special_event", p03.ReasonCode.CENSORED_SPECIAL),
        ("R", "special_event", p03.ReasonCode.CENSORED_SPECIAL),
        ("P", "special_event", p03.ReasonCode.CENSORED_SPECIAL),
        ("Q", "special_event", p03.ReasonCode.CENSORED_SPECIAL),
        ("V", "special_event", p03.ReasonCode.CENSORED_SPECIAL),
        ("c", "incomplete_let", p03.ReasonCode.CENSORED_INCOMPLETE_LET),
    ],
)
def test_structured_terminal_events_are_censored(raw, terminal, reason):
    result = classify(raw, 1)
    assert result.analysis_state is p03.AnalysisState.INELIGIBLE_CENSORED
    assert result.reason_code is reason
    assert result.terminal_serve_outcome == terminal
    assert result.eligible_for_outcome_comparison is False


@pytest.mark.parametrize("code", ["n", "w", "d", "x", "g", "e", "!"])
@pytest.mark.parametrize("serve_number", [1, 2])
def test_all_structured_faults_are_censored(code, serve_number):
    result = classify(f"4{code}", serve_number)
    assert result.analysis_state is p03.AnalysisState.INELIGIBLE_CENSORED
    assert result.reason_code is p03.ReasonCode.CENSORED_FAULT
    assert result.terminal_serve_outcome == "service_fault"


def test_second_fault_is_only_called_double_fault_with_explicit_prior_context():
    first = classify("4n", 1)
    second = classify("6d", 2, prior_fault=True)
    assert first.reason_code is p03.ReasonCode.CENSORED_FAULT
    assert second.reason_code is p03.ReasonCode.CENSORED_DOUBLE_FAULT
    assert second.terminal_serve_outcome == "double_fault"


@pytest.mark.parametrize("raw", ["4+*", "4+#"])
def test_marker_followed_by_unstructured_terminal_character_keeps_tag_but_not_terminal_claim(raw):
    """El parser actual no estructura outcomes posteriores al marcador ``+``."""
    result = classify(raw, 1)
    assert result.analysis_state is p03.AnalysisState.POSITIVE_TAGGED
    assert result.explicit_intent_tagged is True
    assert result.terminal_serve_outcome is None
    assert result.has_residual is True


@pytest.mark.parametrize("raw", ["", "4", "4?+", "4 +", "?4+", "C"])
def test_missing_or_ambiguous_prefix_content_is_unknown(raw):
    result = classify(raw, 1)
    assert result.analysis_state is p03.AnalysisState.UNKNOWN
    assert result.eligible_for_outcome_comparison is False
    assert result.reason_code in {
        p03.ReasonCode.MISSING_PREFIX,
        p03.ReasonCode.AMBIGUOUS_POST_PREFIX,
    }


def test_null_sequence_is_rejected_by_p03_contract():
    with pytest.raises(TypeError):
        p03.parse_and_classify_serve_and_volley_attempt(None, 1)
    with pytest.raises(TypeError):
        p03.classify_serve_and_volley_attempt(None, 1, parse_sequence(None, 1))


def test_unknown_direction_is_explicitly_unknown_not_an_invalid_sequence():
    result = classify("0+f1", 1)
    assert result.service_direction_code == "0"
    assert result.service_direction_label == "unknown"
    assert result.explicit_intent_tagged is True
    assert result.analysis_state is p03.AnalysisState.UNKNOWN
    assert result.reason_code is p03.ReasonCode.UNKNOWN_DIRECTION
    assert result.eligible_for_outcome_comparison is False


def test_structured_terminal_has_precedence_over_unknown_direction():
    result = classify("0*", 1)
    assert result.analysis_state is p03.AnalysisState.INELIGIBLE_CENSORED
    assert result.reason_code is p03.ReasonCode.CENSORED_ACE
    assert result.eligible_for_outcome_comparison is False


def test_warning_or_residue_after_an_unambiguous_marker_does_not_erase_the_tag():
    result = classify("4+?", 1)
    assert result.analysis_state is p03.AnalysisState.POSITIVE_TAGGED
    assert result.has_residual is True
    assert "W_DOCUMENTED_TOKEN_NOT_CONSUMED" in result.parser_warning_codes
    assert "W_UNKNOWN_CHARACTER" in result.parser_warning_codes


def test_repeated_warning_codes_at_distinct_spans_are_preserved_and_auditable():
    result = classify("4+bb", 1)
    warning_items = tuple(zip(result.parser_warning_codes, result.parser_warning_spans))
    assert len(warning_items) == len(set(warning_items))
    assert warning_items == tuple(sorted(warning_items))
    repeated = [item for item in warning_items if item[0] == "W_DOCUMENTED_TOKEN_NOT_CONSUMED"]
    assert len(repeated) >= 2
    assert len({span for _, span in repeated}) == len(repeated)
    p03.validate_serve_and_volley_classification(result)


def test_core_uses_the_supplied_parse_result_and_never_mutates_it():
    parsed = parse_sequence("4+b27v1*", 1)
    before = serialize_parse_result(parsed)
    result = p03.classify_serve_and_volley_attempt("4+b27v1*", 1, parsed)
    assert result == p03.classify_serve_and_volley_attempt("4+b27v1*", 1, parsed)
    assert serialize_parse_result(parsed) == before


@pytest.mark.parametrize(
    ("raw", "foreign_raw"),
    [
        ("4+b27v1*", "4b27v1*"),
        ("4b+27v1*", "4+b27v1*"),
        ("c4+b27v1*", "4+b27v1*"),
    ],
)
def test_core_rejects_parse_result_from_another_sequence(raw, foreign_raw):
    with pytest.raises(ValueError, match="deben coincidir"):
        p03.classify_serve_and_volley_attempt(raw, 1, parse_sequence(foreign_raw, 1))


def test_core_rejects_parse_result_with_prefix_or_marker_from_another_sequence():
    parsed = parse_sequence("4+b27v1*", 1)
    foreign_prefix = parse_sequence("5+b27v1*", 1).structure
    assert foreign_prefix is not None
    with pytest.raises(ValueError):
        p03.classify_serve_and_volley_attempt(
            "4+b27v1*", 1, replace(parsed, structure=foreign_prefix)
        )


def test_helper_invokes_the_existing_parser_exactly_once(monkeypatch):
    calls = []
    original = p03.parse_sequence

    def counted(raw, serve_number):
        calls.append((raw, serve_number))
        return original(raw, serve_number)

    monkeypatch.setattr(p03, "parse_sequence", counted)
    result = p03.parse_and_classify_serve_and_volley_attempt("6+f1", 2)
    assert calls == [("6+f1", 2)]
    assert result.analysis_state is p03.AnalysisState.POSITIVE_TAGGED


def test_marker_detection_rejects_naive_substring_and_net_shot_implementations():
    late = classify("4b+27v1*", 1)
    volley = classify("4v1*", 1)
    immediate = classify("4+b27v1*", 1)
    assert (late.explicit_intent_tagged, volley.explicit_intent_tagged, immediate.explicit_intent_tagged) == (False, False, True)


def test_not_explicitly_tagged_contract_requires_a_nonmarker_comparable_sequence():
    """Estado reservado para cuando un parser futuro estructure el rally inicial."""
    positive = classify("4+b27v1*", 1)
    hypothetical = replace(
        positive,
        explicit_intent_tagged=False,
        marker_start=None,
        marker_end=None,
        marker_literal=None,
        analysis_state=p03.AnalysisState.NOT_EXPLICITLY_TAGGED,
        reason_code=p03.ReasonCode.NO_IMMEDIATE_MARKER,
    )
    p03.validate_serve_and_volley_classification(hypothetical)
    assert hypothetical.eligible_for_outcome_comparison is True


def test_prefix_boundary_residual_is_never_published_as_not_explicitly_tagged():
    result = classify("4b27v1*", 1)
    assert result.has_residual is True
    assert result.analysis_state is p03.AnalysisState.UNKNOWN
    assert result.reason_code is p03.ReasonCode.AMBIGUOUS_POST_PREFIX


@pytest.mark.parametrize("serve_number", [-1, 0, 3, True, False, "1", "2", 1.0, 2.0, None])
def test_serve_number_domain_is_strict(serve_number):
    with pytest.raises(ValueError):
        p03.parse_and_classify_serve_and_volley_attempt("4+f1", serve_number)


@pytest.mark.parametrize("sequence", [1, 1.0, [], object()])
def test_nontext_sequence_is_rejected(sequence):
    with pytest.raises(TypeError):
        p03.parse_and_classify_serve_and_volley_attempt(sequence, 1)


@pytest.mark.parametrize("prior_fault", [0, 1, "false", None])
def test_previous_attempt_was_fault_is_strict_bool(prior_fault):
    with pytest.raises(TypeError):
        p03.parse_and_classify_serve_and_volley_attempt(
            "4n", 2, previous_attempt_was_fault=prior_fault
        )


def test_rejects_incompatible_or_mutated_parse_result():
    parsed = parse_sequence("4+f1", 1)
    with pytest.raises(TypeError):
        p03.classify_serve_and_volley_attempt("4+f1", 1, object())
    with pytest.raises(ValueError):
        p03.classify_serve_and_volley_attempt("5+f1", 1, parsed)
    broken_tokens = replace(parsed, tokens=tuple(reversed(parsed.tokens)))
    with pytest.raises(ValueError):
        p03.classify_serve_and_volley_attempt("4+f1", 1, broken_tokens)
    broken_spans = replace(parsed, consumed_spans=())
    with pytest.raises(ValueError):
        p03.classify_serve_and_volley_attempt("4+f1", 1, broken_spans)
    truncated = parse_sequence("4+", 1)
    with pytest.raises(ValueError, match="deben coincidir"):
        p03.classify_serve_and_volley_attempt("4+f1", 1, truncated)
    changed_character = replace(
        parsed,
        tokens=(
            parsed.tokens[0],
            Token(
                token_type=parsed.tokens[1].token_type,
                raw_text="x",
                start=1,
                end=2,
                candidate_families=parsed.tokens[1].candidate_families,
                rule_id=parsed.tokens[1].rule_id,
                evidence_level=parsed.tokens[1].evidence_level,
            ),
            *parsed.tokens[2:],
        ),
    )
    with pytest.raises(ValueError):
        p03.classify_serve_and_volley_attempt("4+f1", 1, changed_character)
    overlapping_token = replace(parsed.tokens[1], start=0, end=1)
    with pytest.raises(ValueError):
        p03.classify_serve_and_volley_attempt(
            "4+f1", 1, replace(parsed, tokens=(parsed.tokens[0], overlapping_token, *parsed.tokens[2:]))
        )
    with pytest.raises(ValueError):
        p03.classify_serve_and_volley_attempt("4+f1", 2, parsed)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: replace(value, analysis_state=p03.AnalysisState.UNKNOWN),
        lambda value: replace(value, analysis_state="positive_tagged"),
        lambda value: replace(value, reason_code=p03.ReasonCode.NO_IMMEDIATE_MARKER),
        lambda value: replace(value, explicit_intent_tagged=False),
        lambda value: replace(value, eligible_for_outcome_comparison=False),
        lambda value: replace(value, service_direction_code="9"),
        lambda value: replace(value, marker_end=3),
        lambda value: replace(value, marker_literal="x"),
        lambda value: replace(value, marker_start=2, marker_end=3),
        lambda value: replace(value, marker_start=value.sequence_length, marker_end=value.sequence_length + 1),
        lambda value: replace(value, terminal_serve_outcome="ace"),
        lambda value: replace(value, parser_warning_codes=("Z", "A")),
        lambda value: replace(
            value,
            parser_warning_codes=("A", "A"),
            parser_warning_spans=((0, 1), (0, 1)),
        ),
        lambda value: replace(value, parser_warning_spans=((0, value.sequence_length + 1),)),
        lambda value: replace(value, classification_contract_version="0"),
        lambda value: replace(value, explicit_intent_tagged=1),
        lambda value: replace(value, marker_start=1.0, marker_end=2.0),
    ],
)
def test_validator_rejects_individual_contract_mutations(change):
    valid = classify("4+b27v1*", 1)
    with pytest.raises((TypeError, ValueError)):
        p03.validate_serve_and_volley_classification(change(valid))


def test_validator_rejects_empty_warning_text_without_relying_on_span_mismatch():
    valid = classify("4+b27v1*", 1)
    assert valid.parser_warning_codes
    blank_first_code = ("", *valid.parser_warning_codes[1:])
    with pytest.raises(ValueError):
        p03.validate_serve_and_volley_classification(
            replace(valid, parser_warning_codes=blank_first_code)
        )


def test_validator_rejects_terminal_precedence_and_ineligible_state_mutations():
    censored = classify("4*", 1)
    with pytest.raises(ValueError):
        p03.validate_serve_and_volley_classification(
            replace(censored, terminal_serve_outcome="unreturned_serve")
        )
    with pytest.raises(ValueError):
        p03.validate_serve_and_volley_classification(
            replace(censored, eligible_for_outcome_comparison=True)
        )


def test_contract_can_preserve_an_explicit_marker_if_a_future_parser_censors_it():
    tagged = classify("4+b27v1*", 1)
    future_censored = replace(
        tagged,
        analysis_state=p03.AnalysisState.INELIGIBLE_CENSORED,
        reason_code=p03.ReasonCode.CENSORED_ACE,
        eligible_for_outcome_comparison=False,
        terminal_serve_outcome="ace",
    )
    p03.validate_serve_and_volley_classification(future_censored)
    assert future_censored.explicit_intent_tagged is True


def test_second_fault_requires_explicit_prior_fault_context_and_never_uses_point_outcome():
    without_context = classify("4n", 2)
    with_context = classify("4n", 2, prior_fault=True)
    assert without_context.reason_code is p03.ReasonCode.CENSORED_FAULT
    assert with_context.reason_code is p03.ReasonCode.CENSORED_DOUBLE_FAULT
    assert "point_winner" not in inspect.signature(p03.classify_serve_and_volley_attempt).parameters


def test_result_dataclass_is_immutable_and_public_parser_contract_is_unchanged():
    result = classify("4+f1", 1)
    with pytest.raises(Exception):
        result.marker_start = 0
    assert set(ParseResult.__dataclass_fields__) == {
        "raw_sequence", "serve_number", "input_state", "tokens", "structure",
        "consumed_spans", "residual_spans", "residual_text", "warnings", "errors",
        "lexical_coverage", "syntactic_coverage", "semantic_coverage", "lexical_status",
        "structural_status", "has_no_warnings", "parser_version", "grammar_version",
    }


def test_module_is_synthetic_and_has_no_data_or_execution_dependencies():
    source = inspect.getsource(p03)
    forbidden = ("read_parquet", "data/", "pandas", "pyarrow", "duckdb", "polars", "argparse", "open(", "write_", "to_csv", "causal", "recommend")
    assert all(value not in source for value in forbidden)
