import json
from dataclasses import FrozenInstanceError, replace

import pytest

from src.parsing.serve_sequence import (
    Coverage,
    Diagnostic,
    InputState,
    LexicalStatus,
    Span,
    StructuralStatus,
    _DIRECTION_VALUES,
    _SPECIAL_MEANINGS,
    parse_sequence,
    serialize_parse_result,
    serialize_parse_result_json,
    tokenize_sequence,
    validate_parsed_sequence,
)
from src.parsing.serve_sequence_rules import (
    CHARACTER_RULES,
    GRAMMAR_VERSION,
    PARSER_VERSION,
    RULES,
    RULES_BY_ID,
    _build_rule_index,
    get_rule,
)


def warning_codes(result):
    return [warning.code for warning in result.warnings]


def assert_exact_partition(result):
    raw = result.raw_sequence
    if raw is None:
        assert result.consumed_spans == result.residual_spans == ()
        return
    spans = sorted(result.consumed_spans + result.residual_spans, key=lambda item: item.start)
    assert all(left.end <= right.start for left, right in zip(spans, spans[1:]))
    assert "".join(raw[span.start:span.end] for span in spans) == raw


def test_null_input():
    result = parse_sequence(None, 1)
    assert result.input_state is InputState.NULL
    assert result.lexical_status is LexicalStatus.NOT_APPLICABLE
    assert result.structural_status is StructuralStatus.NOT_ASSESSED
    assert result.raw_sequence is None
    assert result.lexical_coverage.proportion is None
    assert result.syntactic_coverage.proportion is None
    assert result.semantic_coverage.proportion is None


def test_empty_input():
    result = parse_sequence("", 2)
    assert result.input_state is InputState.EMPTY
    assert result.lexical_status is LexicalStatus.NOT_APPLICABLE
    assert result.structural_status is StructuralStatus.NOT_ASSESSED
    assert result.lexical_coverage == result.syntactic_coverage == result.semantic_coverage
    assert result.lexical_coverage.proportion is None


@pytest.mark.parametrize(
    ("raw", "lets", "direction", "value"),
    [("4", (), "4", "wide"), ("c4", ("c",), "4", "wide"), ("cc4", ("c", "c"), "4", "wide"), ("0", (), "0", "unknown")],
)
def test_supported_serve_prefixes(raw, lets, direction, value):
    result = parse_sequence(raw, 1)
    assert result.structure.lets == lets
    assert result.structure.direction == direction
    assert result.structure.direction_value == value
    assert result.consumed_spans == (Span(0, len(raw)),)
    assert result.residual_spans == ()
    assert result.structural_status is StructuralStatus.CONSISTENT
    assert result.has_no_warnings


@pytest.mark.parametrize("code", ["S", "R", "P", "Q"])
def test_unit_special_codes_in_first_serve(code):
    result = parse_sequence(code, 1)
    assert result.structure.structure_type == "special_code"
    assert result.structure.code == code
    assert result.structure.resolves_point is True
    assert result.semantic_coverage.numerator == 1
    assert result.has_no_warnings


def test_time_violation_in_first_serve_does_not_resolve_point():
    result = parse_sequence("V", 1)
    assert result.structure.structure_type == "time_violation"
    assert result.structure.resolves_point is False
    assert result.semantic_coverage.proportion == 1.0


def test_time_violation_in_second_serve_is_out_of_context():
    result = parse_sequence("V", 2)
    assert result.structure is None
    assert result.residual_text == "V"
    assert warning_codes(result) == ["W_DOCUMENTED_TOKEN_NOT_CONSUMED", "W_TOKEN_OUT_OF_CONTEXT"]


def test_special_code_with_extra_content_is_not_interpreted():
    result = parse_sequence("R4", 1)
    assert result.structure is None
    assert result.consumed_spans == ()
    assert result.residual_text == "R4"
    assert "W_SPECIAL_CODE_WITH_EXTRA_CONTENT" in warning_codes(result)
    assert warning_codes(result).count("W_DOCUMENTED_TOKEN_NOT_CONSUMED") == 2


def test_documented_character_can_be_lexical_but_not_syntactic():
    result = parse_sequence("*", 1)
    assert result.tokens[0].candidate_families == ("terminator",)
    assert result.lexical_coverage.proportion == 1.0
    assert result.syntactic_coverage.proportion == 0.0
    assert warning_codes(result) == ["W_DOCUMENTED_TOKEN_NOT_CONSUMED"]


def test_undocumented_character_is_not_lexically_recognized():
    result = parse_sequence("?", 1)
    assert result.tokens[0].token_type == "undocumented_character"
    assert result.lexical_status is LexicalStatus.UNCONSUMED
    assert result.lexical_coverage.numerator == 0
    assert warning_codes(result) == ["W_UNKNOWN_CHARACTER"]


@pytest.mark.parametrize("raw", [" ", "\t", "\n", "\u2003"])
def test_all_unicode_whitespace_is_preserved_as_residue(raw):
    result = parse_sequence(raw, 2)
    assert result.input_state is InputState.PRESENT
    assert result.raw_sequence == raw
    assert result.tokens[0].raw_text == raw
    assert result.tokens[0].token_type == "undocumented_whitespace"
    assert result.residual_text == raw
    assert result.lexical_coverage.proportion == 0.0
    assert warning_codes(result) == ["W_UNDOCUMENTED_WHITESPACE"]


def test_lexical_recovery_does_not_create_syntax_across_residue():
    result = parse_sequence("4?6", 1)
    assert [token.raw_text for token in result.tokens] == ["4", "?", "6"]
    assert [token.token_type for token in result.tokens] == ["documented_character", "undocumented_character", "documented_character"]
    assert result.consumed_spans == (Span(0, 1),)
    assert result.residual_spans == (Span(1, 3),)
    assert result.residual_text == "?6"
    assert result.structure.direction == "4"


def test_a71_example_is_not_corrected_or_extended():
    result = parse_sequence("4+w", 1)
    assert result.raw_sequence == "4+w"
    assert result.structure.direction == "4"
    assert result.residual_text == "+w"
    assert warning_codes(result) == ["W_DOCUMENTED_TOKEN_NOT_CONSUMED", "W_DOCUMENTED_TOKEN_NOT_CONSUMED"]
    assert all(warning.code != "W_OFFICIAL_INCONSISTENCY_A71" for warning in result.warnings)


def test_spans_cover_input_without_overlap_and_reconstruct_exactly():
    result = parse_sequence("cc4?\t6", 1)
    assert_exact_partition(result)
    assert result.tokens == tuple(sorted(result.tokens, key=lambda token: token.start))
    assert all(token.raw_text == result.raw_sequence[token.start:token.end] for token in result.tokens)


def test_coverage_uses_original_unicode_length():
    result = parse_sequence("4? ", 1)
    assert (result.lexical_coverage.numerator, result.lexical_coverage.denominator, result.lexical_coverage.proportion) == (1, 3, 1 / 3)
    assert (result.syntactic_coverage.numerator, result.syntactic_coverage.denominator, result.syntactic_coverage.proportion) == (1, 3, 1 / 3)
    assert (result.semantic_coverage.numerator, result.semantic_coverage.denominator, result.semantic_coverage.proportion) == (1, 3, 1 / 3)
    assert result.lexical_status is LexicalStatus.PARTIALLY_CONSUMED
    assert result.structural_status is StructuralStatus.INCOMPLETE


def test_multiple_warnings_have_total_deterministic_order():
    result = parse_sequence("4 ?+", 1)
    observed = [(item.start, item.end, item.code) for item in result.warnings]
    assert observed == [
        (1, 2, "W_UNDOCUMENTED_WHITESPACE"),
        (2, 3, "W_UNKNOWN_CHARACTER"),
        (3, 4, "W_DOCUMENTED_TOKEN_NOT_CONSUMED"),
    ]
    assert result.has_no_warnings is False


@pytest.mark.parametrize("raw", [7, 3.5, [], object()])
def test_invalid_input_type_raises(raw):
    with pytest.raises(TypeError, match="str o None"):
        parse_sequence(raw, 1)


@pytest.mark.parametrize("serve_number", [0, 3, True, "1", None])
def test_invalid_serve_number_raises(serve_number):
    with pytest.raises(ValueError, match="1 o 2"):
        parse_sequence("4", serve_number)


def test_nonexistent_rule_raises():
    with pytest.raises(ValueError, match="rule_id inexistente"):
        get_rule("LEX-DOES-NOT-EXIST")


def test_validation_rejects_broken_reconstruction():
    result = parse_sequence("4+w", 1)
    broken = replace(result, residual_spans=(Span(2, 3),))
    with pytest.raises(ValueError, match="residual_spans"):
        validate_parsed_sequence(broken)


def test_serialization_contains_only_json_primitives():
    serialized = serialize_parse_result(parse_sequence("c4+", 2))
    assert json.loads(json.dumps(serialized, ensure_ascii=False)) == serialized
    assert serialized["parser_version"] == PARSER_VERSION
    assert serialized["grammar_version"] == GRAMMAR_VERSION


def test_json_serialization_is_deterministic():
    result = parse_sequence("4?\t", 1)
    first = serialize_parse_result_json(result)
    second = serialize_parse_result_json(result)
    assert first == second
    assert json.loads(first)["raw_sequence"] == "4?\t"


def test_results_and_rule_catalog_are_immutable():
    result = parse_sequence("4", 1)
    with pytest.raises(FrozenInstanceError):
        result.raw_sequence = "5"
    with pytest.raises(TypeError):
        RULES_BY_ID["new"] = RULES[0]
    with pytest.raises(TypeError):
        CHARACTER_RULES["$"] = RULES[0]
    with pytest.raises(TypeError):
        _DIRECTION_VALUES["4"] = "changed"
    with pytest.raises(TypeError):
        _SPECIAL_MEANINGS["R"] = "changed"
    with pytest.raises(FrozenInstanceError):
        RULES[0].candidate_families += ("changed",)
    with pytest.raises(FrozenInstanceError):
        result.structure.lets += ("c",)


def test_serialized_mutation_does_not_modify_nested_result():
    result = parse_sequence("c4+", 1)
    serialized = serialize_parse_result(result)
    serialized["tokens"][0]["candidate_families"].append("changed")
    serialized["structure"]["lets"].append("changed")
    serialized["warnings"].clear()
    assert result.tokens[0].candidate_families == ("let",)
    assert result.structure.lets == ("c",)
    assert len(result.warnings) == 1


def test_duplicate_rule_ids_are_rejected_before_indexing():
    duplicate = replace(RULES[1], rule_id=RULES[0].rule_id)
    with pytest.raises(ValueError, match="rule_id duplicados"):
        _build_rule_index((RULES[0], duplicate))


def test_synthetic_non_bmp_character_uses_project_interpretation_provenance():
    result = parse_sequence(chr(0x1F600), 1)
    token = result.tokens[0]
    rule = get_rule(token.rule_id)
    assert token.start == 0 and token.end == 1
    assert token.evidence_level == "project_interpretation"
    assert rule.evidence_level == "project_interpretation"
    assert "complemento del vocabulario oficial inspeccionado" in rule.source
    assert warning_codes(result) == ["W_UNKNOWN_CHARACTER"]


def _contract_corruptions():
    base = parse_sequence("4+w", 1)
    star = parse_sequence("*", 1)
    special = parse_sequence("R", 1)
    null = parse_sequence(None, 1)
    empty = parse_sequence("", 1)
    tokens = base.tokens
    warnings = base.warnings
    return (
        ("token_omitted", replace(base, tokens=tokens[:-1])),
        ("token_duplicated", replace(base, tokens=tokens + (tokens[-1],))),
        ("tokens_disordered", replace(base, tokens=tuple(reversed(tokens)))),
        ("tokens_overlapping", replace(base, tokens=(tokens[0], replace(tokens[1], start=0, end=1, raw_text="4"), tokens[2]))),
        ("token_raw_text_false", replace(base, tokens=(replace(tokens[0], raw_text="5"),) + tokens[1:])),
        ("token_rule_id_false", replace(base, tokens=(replace(tokens[0], rule_id="LEX-MISSING"),) + tokens[1:])),
        ("token_families_false", replace(base, tokens=(replace(tokens[0], candidate_families=("shot_type",)),) + tokens[1:])),
        ("tokenization_not_exhaustive", replace(base, tokens=tokens[1:])),
        ("residual_text_false", replace(base, residual_text="WRONG")),
        ("residual_noncanonical", replace(base, residual_spans=(Span(2, 3),))),
        ("adjacent_residual_not_merged", replace(base, residual_spans=(Span(1, 2), Span(2, 3)))),
        ("lexical_coverage_false", replace(base, lexical_coverage=Coverage(2, 3, 2 / 3))),
        ("syntactic_coverage_false", replace(base, syntactic_coverage=Coverage(2, 3, 2 / 3))),
        ("semantic_coverage_false", replace(base, semantic_coverage=Coverage(2, 3, 2 / 3))),
        ("proportion_false", replace(base, lexical_coverage=Coverage(3, 3, 0.5))),
        ("negative_numerator", replace(base, lexical_coverage=Coverage(-1, 3, -1 / 3))),
        ("numerator_above_denominator", replace(base, lexical_coverage=Coverage(4, 3, 4 / 3))),
        ("structure_end_false", replace(base, structure=replace(base.structure, end=3))),
        ("structure_direction_false", replace(base, structure=replace(base.structure, direction="5"))),
        ("structure_meaning_false", replace(special, structure=replace(special.structure, meaning="changed"))),
        ("consumed_without_structure", replace(star, consumed_spans=(Span(0, 1),), residual_spans=())),
        ("structure_without_consumed", replace(base, consumed_spans=(), residual_spans=(Span(0, 3),))),
        ("warning_flag_false", replace(base, has_no_warnings=True)),
        ("warning_removed", replace(base, warnings=warnings[:-1])),
        ("warning_added", replace(base, warnings=warnings + (warnings[-1],))),
        ("warnings_disordered", replace(base, warnings=tuple(reversed(warnings)))),
        ("warning_span_false", replace(base, warnings=(replace(warnings[0], start=0),) + warnings[1:])),
        ("input_state_false", replace(base, input_state=InputState.EMPTY)),
        ("lexical_status_false", replace(base, lexical_status=LexicalStatus.UNCONSUMED)),
        ("structural_status_false", replace(base, structural_status=StructuralStatus.CONSISTENT)),
        ("parser_version_false", replace(base, parser_version="9.9.9")),
        ("grammar_version_false", replace(base, grammar_version="changed")),
        ("errors_nonempty", replace(base, errors=(Diagnostic("E", 0, 1, "technical"),))),
        ("null_result_corrupt", replace(null, residual_text="WRONG")),
        ("empty_result_corrupt", replace(empty, lexical_coverage=Coverage(1, 0, None))),
    )


@pytest.mark.parametrize(
    ("corruption_name", "corrupted"),
    _contract_corruptions(),
    ids=[item[0] for item in _contract_corruptions()],
)
def test_contract_corruptions_fail_validation_and_serialization(
    corruption_name, corrupted
):
    with pytest.raises((TypeError, ValueError)):
        validate_parsed_sequence(corrupted)
    with pytest.raises((TypeError, ValueError)):
        serialize_parse_result(corrupted)


@pytest.mark.parametrize("raw", [None, "", "4", "4+w", "?"])
def test_output_does_not_claim_validity_or_parseability(raw):
    payload = serialize_parse_result(parse_sequence(raw, 1))
    forbidden = {"valid", "invalid", "parseable"}

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    assert keys(payload).isdisjoint(forbidden)
