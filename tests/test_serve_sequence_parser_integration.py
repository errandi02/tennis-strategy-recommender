from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.parsing.serve_sequence import Span, parse_sequence


pytestmark = pytest.mark.integration
POINTS_FILE = Path("data/processed/points_enriched.parquet")

CASES = (
    ("19600529-M-Roland_Garros-F-Nicola_Pietrangeli-Luis_Ayala", 44, "first_serve", "6*", "serve_ace", 2),
    ("19710704-M-Wimbledon-F-John_Newcombe-Stan_Smith", 179, "second_serve", "4*", "serve_ace", 2),
    ("19750101-M-Australian_Open-F-Jimmy_Connors-John_Newcombe", 162, "first_serve", "4#", "serve_unreturned", 2),
    ("19850701-M-Wimbledon-R16-Boris_Becker-Tim_Mayotte", 78, "second_serve", "c4#", "serve_unreturned", 3),
    ("19600529-M-Roland_Garros-F-Nicola_Pietrangeli-Luis_Ayala", 11, "first_serve", "6d", "serve_fault", 2),
    ("19600704-M-Wimbledon-F-Rod_Laver-Neale_Fraser", 49, "second_serve", "4d", "serve_fault", 2),
    ("19971002-M-Basel-QF-Mark_Philippoussis-Yevgeny_Kafelnikov", 34, "first_serve", "6wd", "serve_fault", 2),
    ("19850629-M-Wimbledon-R32-Joakim_Nystrom-Boris_Becker", 204, "first_serve", "4w ", "serve_fault", 2),
    ("19740707-M-Wimbledon-F-Ken_Rosewall-Jimmy_Connors", 146, "first_serve", "6f#", "serve_prefix", 1),
    ("20041002-M-Bangkok-SF-Andy_Roddick-Marat_Safin", 1, "first_serve", "4f3*", "serve_prefix", 1),
    ("19690703-M-Wimbledon-SF-Rod_Laver-Arthur_Ashe", 5, "first_serve", "4+w", "serve_prefix", 1),
    ("20131005-M-Tokyo-SF-Nicolas_Almagro-Juan_Martin_Del_Potro", 121, "first_serve", "6*f28f1f1b3-", "serve_ace", 2),
)

INDEPENDENT_LEXICON = {}


def _register(characters, rule_id, families):
    for character in characters:
        assert character not in INDEPENDENT_LEXICON
        INDEPENDENT_LEXICON[character] = (rule_id, families)


_register("c", "LEX-LET", ("let",))
_register("456", "LEX-SERVE-DIRECTION", ("serve_direction",))
_register("0", "LEX-CONTEXTUAL-ZERO", ("serve_direction", "shot_direction", "return_depth"))
_register("fbrsvzopuylmhijktq", "LEX-SHOT-TYPE", ("shot_type",))
_register("123", "LEX-SHOT-DIRECTION", ("shot_direction",))
_register("789", "LEX-RETURN-DEPTH", ("return_depth",))
_register("nwdxe!", "LEX-FAULT-ERROR", ("serve_fault", "shot_error"))
_register("g", "LEX-FOOT-FAULT", ("serve_fault",))
_register("+-=;^", "LEX-MODIFIER", ("modifier",))
_register("*#@", "LEX-TERMINATOR", ("terminator",))
_register("C", "LEX-CHALLENGE", ("terminator", "special_code"))
_register("SRPQ", "LEX-SPECIAL-CODE", ("special_code",))
_register("V", "LEX-TIME-VIOLATION", ("special_code",))

WARNING_MESSAGES = {
    "W_DOCUMENTED_TOKEN_NOT_CONSUMED": "Caracter documentado no consumido por el parser 0.2.0.",
    "W_UNKNOWN_CHARACTER": "Caracter sin regla en el vocabulario documental.",
    "W_UNDOCUMENTED_WHITESPACE": "Whitespace Unicode no documentado conservado como residuo.",
    "W_CONTENT_AFTER_SERVICE_OUTCOME": "Existe contenido residual posterior al resultado del servicio.",
    "W_UNDOCUMENTED_SERVE_FAULT_COMBINATION": "La combinacion de codigos de fallo no esta documentada.",
}
FAULT_TYPES = {"n": "net", "w": "wide", "d": "long", "x": "wide_and_long", "g": "foot_fault", "e": "unknown", "!": "framed"}
DIRECTION_VALUES = {"4": "wide", "5": "body", "6": "down_the_t", "0": "unknown"}


def load_fixed_sample():
    if not POINTS_FILE.exists():
        pytest.skip("No esta disponible points_enriched.parquet.")
    return pq.read_table(
        POINTS_FILE,
        columns=["match_id", "point_number", "first_serve", "second_serve"],
        filters=[("match_id", "in", sorted({case[0] for case in CASES}))],
    ).to_pandas()


def expected_token(character, index):
    if character.isspace():
        return ("undocumented_whitespace", character, index, index + 1, (), "LEX-WHITESPACE", "project_interpretation")
    if character in INDEPENDENT_LEXICON:
        rule_id, families = INDEPENDENT_LEXICON[character]
        return ("documented_character", character, index, index + 1, families, rule_id, "official_explicit")
    return ("undocumented_character", character, index, index + 1, (), "LEX-UNDOCUMENTED", "project_interpretation")


def expected_warnings(raw, structure_type, consumed_length):
    warnings = []
    if structure_type in {"serve_ace", "serve_unreturned", "serve_fault"} and consumed_length < len(raw):
        if structure_type == "serve_fault" and raw[consumed_length] in FAULT_TYPES:
            code = "W_UNDOCUMENTED_SERVE_FAULT_COMBINATION"
        else:
            code = "W_CONTENT_AFTER_SERVICE_OUTCOME"
        warnings.append((code, consumed_length, len(raw), WARNING_MESSAGES[code]))
    for index, character in enumerate(raw):
        if index < consumed_length:
            continue
        if character.isspace():
            code = "W_UNDOCUMENTED_WHITESPACE"
        elif character in INDEPENDENT_LEXICON:
            code = "W_DOCUMENTED_TOKEN_NOT_CONSUMED"
        else:
            code = "W_UNKNOWN_CHARACTER"
        warnings.append((code, index, index + 1, WARNING_MESSAGES[code]))
    return tuple(sorted(warnings, key=lambda item: (item[1], item[2], item[0], item[3])))


def prefix_tuple(prefix):
    return (
        prefix.structure_type,
        prefix.lets,
        prefix.direction,
        prefix.direction_value,
        prefix.start,
        prefix.end,
        prefix.rule_ids,
    )


def test_fixed_real_sequences_match_parser_020_contract():
    sample = load_fixed_sample()
    for match_id, point_number, column, expected_raw, structure_type, consumed_length in CASES:
        selected = sample[(sample["match_id"] == match_id) & (sample["point_number"] == point_number)]
        assert len(selected) == 1
        raw = selected.iloc[0][column]
        assert raw == expected_raw
        serve_number = 1 if column == "first_serve" else 2
        result = parse_sequence(raw, serve_number)

        expected_tokens = tuple(expected_token(character, index) for index, character in enumerate(raw))
        actual_tokens = tuple((token.token_type, token.raw_text, token.start, token.end, token.candidate_families, token.rule_id, token.evidence_level) for token in result.tokens)
        assert actual_tokens == expected_tokens
        assert result.consumed_spans == (Span(0, consumed_length),)
        expected_residual = (Span(consumed_length, len(raw)),) if consumed_length < len(raw) else ()
        assert result.residual_spans == expected_residual
        assert result.residual_text == raw[consumed_length:]

        structure = result.structure
        prefix = structure if structure_type == "serve_prefix" else structure.prefix
        prefix_length = consumed_length if structure_type == "serve_prefix" else consumed_length - 1
        lets = tuple(raw[: prefix_length - 1])
        direction = raw[prefix_length - 1]
        expected_rule_ids = tuple("LEX-LET" for _ in lets) + (("LEX-CONTEXTUAL-ZERO" if direction == "0" else "LEX-SERVE-DIRECTION"),)
        assert prefix_tuple(prefix) == ("serve_prefix", lets, direction, DIRECTION_VALUES[direction], 0, prefix_length, expected_rule_ids)

        if structure_type == "serve_ace":
            assert (structure.structure_type, structure.outcome_code, structure.start, structure.end, structure.rule_id, structure.resolves_point) == ("serve_ace", "*", 0, consumed_length, "SYN-SERVE-ACE", True)
        elif structure_type == "serve_unreturned":
            assert (structure.structure_type, structure.outcome_code, structure.start, structure.end, structure.rule_id, structure.resolves_point) == ("serve_unreturned", "#", 0, consumed_length, "SYN-SERVE-UNRETURNED", True)
        elif structure_type == "serve_fault":
            code = raw[consumed_length - 1]
            assert (structure.structure_type, structure.fault_code, structure.fault_type, structure.start, structure.end, structure.rule_id, structure.resolves_point) == ("serve_fault", code, FAULT_TYPES[code], 0, consumed_length, "SYN-SERVE-FAULT", False if serve_number == 1 else None)
        else:
            assert structure.structure_type == "serve_prefix"

        actual_warnings = tuple((warning.code, warning.start, warning.end, warning.message) for warning in result.warnings)
        assert actual_warnings == expected_warnings(raw, structure_type, consumed_length)
        lexical_numerator = sum(character in INDEPENDENT_LEXICON for character in raw)
        assert (result.lexical_coverage.numerator, result.lexical_coverage.denominator, result.lexical_coverage.proportion) == (lexical_numerator, len(raw), lexical_numerator / len(raw))
        assert (result.syntactic_coverage.numerator, result.syntactic_coverage.denominator, result.syntactic_coverage.proportion) == (consumed_length, len(raw), consumed_length / len(raw))
        assert (result.semantic_coverage.numerator, result.semantic_coverage.denominator, result.semantic_coverage.proportion) == (consumed_length, len(raw), consumed_length / len(raw))
        expected_lexical_status = "fully_consumed" if lexical_numerator == len(raw) else "unconsumed" if lexical_numerator == 0 else "partially_consumed"
        assert result.lexical_status.value == expected_lexical_status
        assert result.structural_status.value == ("consistent" if consumed_length == len(raw) else "incomplete")
        assert result.has_no_warnings == (len(actual_warnings) == 0)
        assert result.parser_version == "0.2.0"
        assert result.grammar_version == "mcp-0.3.2-project-0.2"
