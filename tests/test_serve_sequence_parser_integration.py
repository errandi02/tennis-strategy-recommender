from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.parsing.serve_sequence import Span, parse_sequence


pytestmark = pytest.mark.integration
POINTS_FILE = Path("data/processed/points_enriched.parquet")

CASES = (
    ("19800705-M-Wimbledon-SF-John_Mcenroe-Jimmy_Connors", 240, "first_serve", "5*", "serve_prefix"),
    ("19761011-M-WITC_Hilton_Head-SF-Rod_Laver-Bjorn_Borg", 105, "first_serve", "6#", "serve_prefix"),
    ("19780125-M-Pepsi_Grand_Slam-SF-Brian_Gottfried-Bjorn_Borg", 66, "first_serve", "4+b27v1*", "serve_prefix"),
    ("19690703-M-Wimbledon-SF-Rod_Laver-Arthur_Ashe", 5, "first_serve", "4+w", "serve_prefix"),
    ("19850629-M-Wimbledon-R32-Joakim_Nystrom-Boris_Becker", 204, "first_serve", "4w ", "serve_prefix"),
    ("19810607-M-Roland_Garros-F-Bjorn_Borg-Ivan_Lendl", 169, "second_serve", "  ", None),
    ("19960703-M-Wimbledon-QF-Richard_Krajicek-Pete_Sampras", 171, "second_serve", "Play suspended- resumed next day at 550 PM          ", None),
    ("19600529-M-Roland_Garros-F-Nicola_Pietrangeli-Luis_Ayala", 1, "first_serve", "R", "special_code"),
    ("20140104-M-Chennai-SF-Edouard_Roger_Vasselin-Marcel_Granollers", 28, "first_serve", "V", "time_violation"),
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
    "W_DOCUMENTED_TOKEN_NOT_CONSUMED": "Caracter documentado no consumido por el parser 0.1.0.",
    "W_UNKNOWN_CHARACTER": "Caracter sin regla en el vocabulario documental.",
    "W_UNDOCUMENTED_WHITESPACE": "Whitespace Unicode no documentado conservado como residuo.",
    "W_SPECIAL_CODE_WITH_EXTRA_CONTENT": "El codigo especial no es una secuencia unitaria completa.",
}


def load_fixed_sample():
    if not POINTS_FILE.exists():
        pytest.skip("No esta disponible points_enriched.parquet.")
    match_ids = sorted({case[0] for case in CASES})
    return pq.read_table(
        POINTS_FILE,
        columns=["match_id", "point_number", "first_serve", "second_serve"],
        filters=[("match_id", "in", match_ids)],
    ).to_pandas()


def expected_token(character, index):
    if character.isspace():
        return ("undocumented_whitespace", character, index, index + 1, (), "LEX-WHITESPACE", "project_interpretation")
    if character in INDEPENDENT_LEXICON:
        rule_id, families = INDEPENDENT_LEXICON[character]
        return ("documented_character", character, index, index + 1, families, rule_id, "official_explicit")
    return ("undocumented_character", character, index, index + 1, (), "LEX-UNDOCUMENTED", "project_interpretation")


def expected_warning_tuples(raw, consumed_length):
    warnings = []
    special_positions = [index for index, character in enumerate(raw) if character in "SRPQV"]
    if len(raw) > 1 and special_positions:
        index = special_positions[0]
        code = "W_SPECIAL_CODE_WITH_EXTRA_CONTENT"
        warnings.append((code, index, index + 1, WARNING_MESSAGES[code]))
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


def test_fixed_real_sequences_match_the_complete_minimal_contract():
    sample = load_fixed_sample()
    direction_values = {"4": "wide", "5": "body", "6": "down_the_t"}
    special_meanings = {"R": "unobserved_point_awarded_to_returner", "V": "first_serve_lost_by_time_violation"}
    for match_id, point_number, column, expected_raw, structure_type in CASES:
        selected = sample[(sample["match_id"] == match_id) & (sample["point_number"] == point_number)]
        assert len(selected) == 1
        raw = selected.iloc[0][column]
        assert raw == expected_raw
        result = parse_sequence(raw, 1 if column == "first_serve" else 2)

        expected_tokens = tuple(expected_token(character, index) for index, character in enumerate(raw))
        actual_tokens = tuple((token.token_type, token.raw_text, token.start, token.end, token.candidate_families, token.rule_id, token.evidence_level) for token in result.tokens)
        assert actual_tokens == expected_tokens

        consumed_length = 1 if structure_type is not None else 0
        expected_consumed = (Span(0, consumed_length),) if consumed_length else ()
        expected_residual = (Span(consumed_length, len(raw)),) if consumed_length < len(raw) else ()
        assert result.consumed_spans == expected_consumed
        assert result.residual_spans == expected_residual
        assert result.residual_text == raw[consumed_length:]

        if structure_type is None:
            assert result.structure is None
        elif structure_type == "serve_prefix":
            assert (result.structure.structure_type, result.structure.lets, result.structure.direction, result.structure.direction_value, result.structure.start, result.structure.end) == ("serve_prefix", (), raw[0], direction_values[raw[0]], 0, 1)
        else:
            assert (result.structure.structure_type, result.structure.code, result.structure.meaning, result.structure.start, result.structure.end, result.structure.resolves_point) == (structure_type, raw, special_meanings[raw], 0, 1, structure_type == "special_code")

        actual_warnings = tuple((warning.code, warning.start, warning.end, warning.message) for warning in result.warnings)
        assert actual_warnings == expected_warning_tuples(raw, consumed_length)
        lexical_numerator = sum(character in INDEPENDENT_LEXICON for character in raw)
        assert (result.lexical_coverage.numerator, result.lexical_coverage.denominator, result.lexical_coverage.proportion) == (lexical_numerator, len(raw), lexical_numerator / len(raw))
        assert (result.syntactic_coverage.numerator, result.syntactic_coverage.denominator, result.syntactic_coverage.proportion) == (consumed_length, len(raw), consumed_length / len(raw))
        assert (result.semantic_coverage.numerator, result.semantic_coverage.denominator, result.semantic_coverage.proportion) == (consumed_length, len(raw), consumed_length / len(raw))
        expected_lexical_status = "fully_consumed" if lexical_numerator == len(raw) else "unconsumed" if lexical_numerator == 0 else "partially_consumed"
        expected_structural_status = "not_assessed" if structure_type is None else "consistent" if consumed_length == len(raw) else "incomplete"
        assert result.lexical_status.value == expected_lexical_status
        assert result.structural_status.value == expected_structural_status
        assert result.has_no_warnings == (len(actual_warnings) == 0)
