"""Parser tolerante 0.1.0 de prefijos minimos de servicio."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from math import isfinite
from types import MappingProxyType
from typing import Any

from .serve_sequence_rules import (
    CHARACTER_RULES,
    GRAMMAR_VERSION,
    PARSER_VERSION,
    get_rule,
)


class InputState(str, Enum):
    NULL = "null"
    EMPTY = "empty"
    PRESENT = "present"


class LexicalStatus(str, Enum):
    FULLY_CONSUMED = "fully_consumed"
    PARTIALLY_CONSUMED = "partially_consumed"
    UNCONSUMED = "unconsumed"
    NOT_APPLICABLE = "not_applicable"


class StructuralStatus(str, Enum):
    CONSISTENT = "consistent"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"
    INCONSISTENT = "inconsistent"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Token:
    token_type: str
    raw_text: str
    start: int
    end: int
    candidate_families: tuple[str, ...]
    rule_id: str
    evidence_level: str


@dataclass(frozen=True)
class Diagnostic:
    code: str
    start: int
    end: int
    message: str


@dataclass(frozen=True)
class Coverage:
    numerator: int
    denominator: int
    proportion: float | None


@dataclass(frozen=True)
class ServicePrefix:
    structure_type: str
    lets: tuple[str, ...]
    direction: str
    direction_value: str
    start: int
    end: int
    rule_ids: tuple[str, ...]


@dataclass(frozen=True)
class ServeAce:
    structure_type: str
    prefix: ServicePrefix
    outcome_code: str
    start: int
    end: int
    rule_id: str
    resolves_point: bool


@dataclass(frozen=True)
class ServeUnreturned:
    structure_type: str
    prefix: ServicePrefix
    outcome_code: str
    start: int
    end: int
    rule_id: str
    resolves_point: bool


@dataclass(frozen=True)
class ServeFault:
    structure_type: str
    prefix: ServicePrefix
    fault_code: str
    fault_type: str
    start: int
    end: int
    rule_id: str
    resolves_point: bool | None


@dataclass(frozen=True)
class SpecialCodeStructure:
    structure_type: str
    code: str
    meaning: str
    start: int
    end: int
    rule_id: str
    resolves_point: bool


@dataclass(frozen=True)
class ParseResult:
    raw_sequence: str | None
    serve_number: int
    input_state: InputState
    tokens: tuple[Token, ...]
    structure: ServicePrefix | ServeAce | ServeUnreturned | ServeFault | SpecialCodeStructure | None
    consumed_spans: tuple[Span, ...]
    residual_spans: tuple[Span, ...]
    residual_text: str
    warnings: tuple[Diagnostic, ...]
    errors: tuple[Diagnostic, ...]
    lexical_coverage: Coverage
    syntactic_coverage: Coverage
    semantic_coverage: Coverage
    lexical_status: LexicalStatus
    structural_status: StructuralStatus
    has_no_warnings: bool
    parser_version: str = PARSER_VERSION
    grammar_version: str = GRAMMAR_VERSION


def _immutable_mapping(items: tuple[tuple[str, str], ...]):
    return MappingProxyType(dict(items))


_DIRECTION_VALUES = _immutable_mapping(
    (("4", "wide"), ("5", "body"), ("6", "down_the_t"), ("0", "unknown"))
)
_SPECIAL_MEANINGS = _immutable_mapping(
    (
        ("S", "unobserved_point_awarded_to_server"),
        ("R", "unobserved_point_awarded_to_returner"),
        ("P", "point_penalty_against_server"),
        ("Q", "point_penalty_against_returner"),
    )
)
_FAULT_TYPES = _immutable_mapping(
    (
        ("n", "net"),
        ("w", "wide"),
        ("d", "long"),
        ("x", "wide_and_long"),
        ("g", "foot_fault"),
        ("e", "unknown"),
        ("!", "framed"),
    )
)


def _validate_input(raw: str | None, serve_number: int) -> None:
    if raw is not None and not isinstance(raw, str):
        raise TypeError("raw_sequence debe ser str o None.")
    if type(serve_number) is not int or serve_number not in (1, 2):
        raise ValueError("serve_number debe ser 1 o 2.")


def tokenize_sequence(raw: str | None) -> tuple[Token, ...]:
    """Tokeniza atomica y literalmente sin interpretar el contexto."""
    if raw is not None and not isinstance(raw, str):
        raise TypeError("raw_sequence debe ser str o None.")
    if raw is None:
        return ()
    tokens = []
    for index, character in enumerate(raw):
        if character.isspace():
            rule = get_rule("LEX-WHITESPACE")
            token_type = "undocumented_whitespace"
        elif character in CHARACTER_RULES:
            rule = CHARACTER_RULES[character]
            token_type = "documented_character"
        else:
            rule = get_rule("LEX-UNDOCUMENTED")
            token_type = "undocumented_character"
        tokens.append(
            Token(
                token_type=token_type,
                raw_text=character,
                start=index,
                end=index + 1,
                candidate_families=rule.candidate_families,
                rule_id=rule.rule_id,
                evidence_level=rule.evidence_level,
            )
        )
    return tuple(tokens)


def _complement_spans(length: int, consumed: tuple[Span, ...]) -> tuple[Span, ...]:
    residual = []
    cursor = 0
    for span in consumed:
        if cursor < span.start:
            residual.append(Span(cursor, span.start))
        cursor = span.end
    if cursor < length:
        residual.append(Span(cursor, length))
    return tuple(residual)


def _coverage(numerator: int, denominator: int) -> Coverage:
    return Coverage(numerator, denominator, None if denominator == 0 else numerator / denominator)


def _canonical_structure(raw: str, tokens: tuple[Token, ...], serve_number: int):
    if serve_number == 1 and len(raw) == 1 and raw in _SPECIAL_MEANINGS:
        return SpecialCodeStructure("special_code", raw, _SPECIAL_MEANINGS[raw], 0, 1, tokens[0].rule_id, True)
    if serve_number == 1 and raw == "V":
        return SpecialCodeStructure("time_violation", "V", "first_serve_lost_by_time_violation", 0, 1, tokens[0].rule_id, False)
    cursor = 0
    while cursor < len(raw) and raw[cursor] == "c":
        cursor += 1
    if cursor < len(raw) and raw[cursor] in _DIRECTION_VALUES:
        prefix_end = cursor + 1
        prefix = ServicePrefix(
            "serve_prefix",
            tuple(raw[:cursor]),
            raw[cursor],
            _DIRECTION_VALUES[raw[cursor]],
            0,
            prefix_end,
            tuple(token.rule_id for token in tokens[:prefix_end]),
        )
        if prefix_end >= len(raw):
            return prefix
        outcome = raw[prefix_end]
        outcome_end = prefix_end + 1
        if outcome == "*":
            return ServeAce(
                "serve_ace", prefix, "*", 0, outcome_end, "SYN-SERVE-ACE", True
            )
        if outcome == "#":
            return ServeUnreturned(
                "serve_unreturned",
                prefix,
                "#",
                0,
                outcome_end,
                "SYN-SERVE-UNRETURNED",
                True,
            )
        if outcome in _FAULT_TYPES:
            return ServeFault(
                "serve_fault",
                prefix,
                outcome,
                _FAULT_TYPES[outcome],
                0,
                outcome_end,
                "SYN-SERVE-FAULT",
                False if serve_number == 1 else None,
            )
        return prefix
    return None


def _warning_sort_key(warning: Diagnostic) -> tuple[int, int, str, str]:
    return (warning.start, warning.end, warning.code, warning.message)


def _canonical_consumed_spans(
    structure: ServicePrefix
    | ServeAce
    | ServeUnreturned
    | ServeFault
    | SpecialCodeStructure
    | None,
) -> tuple[Span, ...]:
    return () if structure is None else (Span(structure.start, structure.end),)


def _canonical_warnings(
    raw: str,
    tokens: tuple[Token, ...],
    consumed: tuple[Span, ...],
    structure: ServicePrefix
    | ServeAce
    | ServeUnreturned
    | ServeFault
    | SpecialCodeStructure
    | None,
) -> tuple[Diagnostic, ...]:
    consumed_positions = {
        position for span in consumed for position in range(span.start, span.end)
    }
    warnings = []
    if isinstance(structure, (ServeAce, ServeUnreturned, ServeFault)) and (
        structure.end < len(raw)
    ):
        residual_start = structure.end
        if isinstance(structure, ServeFault) and raw[residual_start] in _FAULT_TYPES:
            code = "W_UNDOCUMENTED_SERVE_FAULT_COMBINATION"
            message = "La combinacion de codigos de fallo no esta documentada."
        else:
            code = "W_CONTENT_AFTER_SERVICE_OUTCOME"
            message = "Existe contenido residual posterior al resultado del servicio."
        warnings.append(
            Diagnostic(code, residual_start, len(raw), message)
        )
    special_with_extra = len(raw) > 1 and any(
        character in "SRPQV" for character in raw
    )
    if special_with_extra:
        first = min(
            index for index, character in enumerate(raw) if character in "SRPQV"
        )
        warnings.append(
            Diagnostic(
                "W_SPECIAL_CODE_WITH_EXTRA_CONTENT",
                first,
                first + 1,
                "El codigo especial no es una secuencia unitaria completa.",
            )
        )
    for token in tokens:
        if token.start in consumed_positions:
            continue
        if token.token_type == "undocumented_whitespace":
            code = "W_UNDOCUMENTED_WHITESPACE"
            message = "Whitespace Unicode no documentado conservado como residuo."
        elif token.token_type == "undocumented_character":
            code = "W_UNKNOWN_CHARACTER"
            message = "Caracter sin regla en el vocabulario documental."
        else:
            code = "W_DOCUMENTED_TOKEN_NOT_CONSUMED"
            message = f"Caracter documentado no consumido por el parser {PARSER_VERSION}."
        warnings.append(Diagnostic(code, token.start, token.end, message))
        if token.raw_text in "SRPQV" and not special_with_extra:
            warnings.append(
                Diagnostic(
                    "W_TOKEN_OUT_OF_CONTEXT",
                    token.start,
                    token.end,
                    "Codigo con regla implementada fuera del contexto autorizado.",
                )
            )
    return tuple(sorted(warnings, key=_warning_sort_key))


def _canonical_coverages(
    raw: str,
    tokens: tuple[Token, ...],
    consumed: tuple[Span, ...],
    structure: ServicePrefix
    | ServeAce
    | ServeUnreturned
    | ServeFault
    | SpecialCodeStructure
    | None,
) -> tuple[Coverage, Coverage, Coverage]:
    lexical_numerator = sum(
        token.token_type == "documented_character" for token in tokens
    )
    syntactic_numerator = sum(span.end - span.start for span in consumed)
    semantic_numerator = 0
    if isinstance(
        structure,
        (ServicePrefix, ServeAce, ServeUnreturned, ServeFault, SpecialCodeStructure),
    ):
        semantic_numerator = structure.end - structure.start
    return (
        _coverage(lexical_numerator, len(raw)),
        _coverage(syntactic_numerator, len(raw)),
        _coverage(semantic_numerator, len(raw)),
    )


def _canonical_statuses(
    raw: str | None,
    lexical_coverage: Coverage,
    structure: ServicePrefix
    | ServeAce
    | ServeUnreturned
    | ServeFault
    | SpecialCodeStructure
    | None,
    residual: tuple[Span, ...],
) -> tuple[LexicalStatus, StructuralStatus]:
    if raw is None or raw == "":
        return LexicalStatus.NOT_APPLICABLE, StructuralStatus.NOT_ASSESSED
    if lexical_coverage.numerator == lexical_coverage.denominator:
        lexical_status = LexicalStatus.FULLY_CONSUMED
    elif lexical_coverage.numerator == 0:
        lexical_status = LexicalStatus.UNCONSUMED
    else:
        lexical_status = LexicalStatus.PARTIALLY_CONSUMED
    if structure is None:
        structural_status = StructuralStatus.NOT_ASSESSED
    elif residual:
        structural_status = StructuralStatus.INCOMPLETE
    else:
        structural_status = StructuralStatus.CONSISTENT
    return lexical_status, structural_status


def parse_sequence(raw: str | None, serve_number: int) -> ParseResult:
    """Construye solo las estructuras aprobadas y conserva todo residuo."""
    _validate_input(raw, serve_number)
    tokens = tokenize_sequence(raw)
    if raw is None:
        result = ParseResult(None, serve_number, InputState.NULL, (), None, (), (), "", (), (), _coverage(0, 0), _coverage(0, 0), _coverage(0, 0), LexicalStatus.NOT_APPLICABLE, StructuralStatus.NOT_ASSESSED, True)
        validate_parsed_sequence(result)
        return result
    if raw == "":
        result = ParseResult("", serve_number, InputState.EMPTY, (), None, (), (), "", (), (), _coverage(0, 0), _coverage(0, 0), _coverage(0, 0), LexicalStatus.NOT_APPLICABLE, StructuralStatus.NOT_ASSESSED, True)
        validate_parsed_sequence(result)
        return result

    structure = _canonical_structure(raw, tokens, serve_number)
    consumed = _canonical_consumed_spans(structure)
    residual = _complement_spans(len(raw), consumed)
    warnings_tuple = _canonical_warnings(raw, tokens, consumed, structure)
    lexical_coverage, syntactic_coverage, semantic_coverage = (
        _canonical_coverages(raw, tokens, consumed, structure)
    )
    lexical_status, structural_status = _canonical_statuses(
        raw, lexical_coverage, structure, residual
    )
    residual_text = "".join(raw[span.start:span.end] for span in residual)
    result = ParseResult(
        raw, serve_number, InputState.PRESENT, tokens, structure, consumed, residual,
        residual_text, warnings_tuple, (), lexical_coverage,
        syntactic_coverage, semantic_coverage,
        lexical_status, structural_status, not warnings_tuple,
    )
    validate_parsed_sequence(result)
    return result


def _validate_tokens(
    actual: tuple[Token, ...], expected: tuple[Token, ...], raw: str | None
) -> None:
    if not isinstance(actual, tuple):
        raise TypeError("tokens debe ser una tupla inmutable.")
    expected_length = 0 if raw is None else len(raw)
    if len(actual) != expected_length:
        raise ValueError("Debe existir exactamente un token por posicion.")
    for index, token in enumerate(actual):
        if not isinstance(token, Token):
            raise TypeError("Todos los tokens deben ser objetos Token.")
        get_rule(token.rule_id)
        if not isinstance(token.candidate_families, tuple):
            raise TypeError("candidate_families debe ser una tupla inmutable.")
        if len(token.candidate_families) != len(set(token.candidate_families)):
            raise ValueError("candidate_families contiene duplicados.")
        if token.start != index or token.end != index + 1:
            raise ValueError("Los tokens deben estar ordenados y cubrir [i, i + 1).")
        if raw is None or token.raw_text != raw[index : index + 1]:
            raise ValueError("raw_text no coincide con el span del token.")
        if token != expected[index]:
            raise ValueError("El token no coincide con la clasificacion canonica.")


def _validate_span_tuple(
    name: str, actual: tuple[Span, ...], expected: tuple[Span, ...]
) -> None:
    if not isinstance(actual, tuple) or not all(
        isinstance(span, Span) for span in actual
    ):
        raise TypeError(f"{name} debe ser una tupla inmutable de Span.")
    if actual != expected:
        raise ValueError(f"{name} no coincide con los spans canonicos.")


def _validate_coverage(name: str, actual: Coverage, expected: Coverage) -> None:
    if not isinstance(actual, Coverage):
        raise TypeError(f"{name} debe ser Coverage.")
    if type(actual.numerator) is not int or type(actual.denominator) is not int:
        raise TypeError(f"{name} debe contener cuentas enteras.")
    if actual.numerator < 0 or actual.denominator < 0:
        raise ValueError(f"{name} contiene cuentas negativas.")
    if actual.numerator > actual.denominator:
        raise ValueError(f"{name} tiene numerador mayor que denominador.")
    if actual.denominator == 0:
        if actual.proportion is not None:
            raise ValueError(f"{name} debe tener proporcion None.")
    else:
        if not isinstance(actual.proportion, (int, float)) or not isfinite(
            actual.proportion
        ):
            raise ValueError(f"{name} contiene una proporcion no finita.")
        if not 0.0 <= actual.proportion <= 1.0:
            raise ValueError(f"{name} contiene una proporcion fuera de [0, 1].")
        if actual.proportion != actual.numerator / actual.denominator:
            raise ValueError(f"{name} contiene una proporcion incoherente.")
    if actual != expected:
        raise ValueError(f"{name} no coincide con la cobertura canonica.")


def validate_parsed_sequence(result: ParseResult) -> None:
    """Reconcilia todo el contrato con derivaciones canonicas puras."""
    if not isinstance(result, ParseResult):
        raise TypeError("result debe ser ParseResult.")
    raw = result.raw_sequence
    _validate_input(raw, result.serve_number)
    if result.parser_version != PARSER_VERSION:
        raise ValueError("parser_version no coincide con PARSER_VERSION.")
    if result.grammar_version != GRAMMAR_VERSION:
        raise ValueError("grammar_version no coincide con GRAMMAR_VERSION.")
    if not isinstance(result.errors, tuple) or result.errors:
        raise ValueError("errors debe ser una tupla vacia en el parser 0.1.0.")

    if raw is None:
        expected_input_state = InputState.NULL
    elif raw == "":
        expected_input_state = InputState.EMPTY
    else:
        expected_input_state = InputState.PRESENT
    if result.input_state is not expected_input_state:
        raise ValueError("input_state no coincide con raw_sequence.")

    expected_tokens = tokenize_sequence(raw)
    _validate_tokens(result.tokens, expected_tokens, raw)
    expected_structure = (
        None
        if raw is None or raw == ""
        else _canonical_structure(raw, expected_tokens, result.serve_number)
    )
    if result.structure != expected_structure:
        raise ValueError("structure no coincide con la estructura canonica soportada.")

    expected_consumed = _canonical_consumed_spans(expected_structure)
    _validate_span_tuple("consumed_spans", result.consumed_spans, expected_consumed)
    expected_residual = (
        () if raw is None else _complement_spans(len(raw), expected_consumed)
    )
    _validate_span_tuple("residual_spans", result.residual_spans, expected_residual)
    expected_residual_text = (
        ""
        if raw is None
        else "".join(raw[span.start : span.end] for span in expected_residual)
    )
    if result.residual_text != expected_residual_text:
        raise ValueError("residual_text no coincide con residual_spans.")

    if raw is None:
        expected_coverages = (_coverage(0, 0),) * 3
    else:
        expected_coverages = _canonical_coverages(
            raw, expected_tokens, expected_consumed, expected_structure
        )
    for name, actual, expected in zip(
        ("lexical_coverage", "syntactic_coverage", "semantic_coverage"),
        (
            result.lexical_coverage,
            result.syntactic_coverage,
            result.semantic_coverage,
        ),
        expected_coverages,
    ):
        _validate_coverage(name, actual, expected)

    expected_lexical_status, expected_structural_status = _canonical_statuses(
        raw, expected_coverages[0], expected_structure, expected_residual
    )
    if result.lexical_status is not expected_lexical_status:
        raise ValueError("lexical_status no coincide con la cobertura lexica.")
    if result.structural_status is not expected_structural_status:
        raise ValueError("structural_status no coincide con estructura y residuo.")

    expected_warnings = (
        ()
        if raw is None or raw == ""
        else _canonical_warnings(
            raw, expected_tokens, expected_consumed, expected_structure
        )
    )
    if not isinstance(result.warnings, tuple) or not all(
        isinstance(warning, Diagnostic) for warning in result.warnings
    ):
        raise TypeError("warnings debe ser una tupla inmutable de Diagnostic.")
    if result.warnings != expected_warnings:
        raise ValueError("warnings no coincide con los diagnosticos canonicos.")
    if result.has_no_warnings != (len(result.warnings) == 0):
        raise ValueError("has_no_warnings no coincide con warnings.")


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, dict):
        return {key: _primitive(value[key]) for key in sorted(value)}
    return value


def serialize_parse_result(result: ParseResult) -> dict[str, Any]:
    """Serializa de forma determinista a primitivas compatibles con JSON."""
    validate_parsed_sequence(result)
    primitive = _primitive(result)
    json.dumps(primitive, ensure_ascii=False, sort_keys=True)
    return primitive


def serialize_parse_result_json(result: ParseResult) -> str:
    return json.dumps(serialize_parse_result(result), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
