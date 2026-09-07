"""P05: extractor sintetico, local y conservador de profundidad del primer resto.

No abre datos ni interpreta el rally.  Solo reconoce la profundidad cuando la
notacion local documentada ``tipo + lateral + profundidad`` esta completa.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.parsing.serve_sequence import (
    Diagnostic,
    ParseResult,
    ServicePrefix,
    ServeAce,
    ServeFault,
    ServeUnreturned,
    SpecialCodeStructure,
    parse_sequence,
    validate_parsed_sequence,
)


CLASSIFICATION_CONTRACT_VERSION: Final = "1.0.0"
SHOT_TYPES: Final = {
    "f": "forehand", "b": "backhand", "r": "forehand_slice",
    "s": "backhand_slice", "v": "forehand_volley", "z": "backhand_volley",
    "o": "overhead", "p": "backhand_overhead", "u": "forehand_drop_shot",
    "y": "backhand_drop_shot", "l": "forehand_lob", "m": "backhand_lob",
    "h": "forehand_half_volley", "i": "backhand_half_volley",
    "j": "forehand_topspin_volley", "k": "backhand_topspin_volley",
    "t": "special_shot",
}
LATERAL_DIRECTIONS: Final = {
    "1": "right_side_of_right_handed_opponent_or_left_of_left_handed",
    "2": "centre",
    "3": "left_side_of_right_handed_opponent_or_right_of_left_handed",
    "0": "unknown",
}
RETURN_DEPTHS: Final = {
    "7": "service_boxes",
    "8": "behind_service_line_closer_to_service_line",
    "9": "closer_to_baseline",
    "0": "unknown",
}
SERVICE_DIRECTIONS: Final = {"4": "wide", "5": "body", "6": "down_the_t", "0": "unknown"}


class ReturnDepthState(str, Enum):
    OBSERVED = "return_depth_observed"
    UNKNOWN = "return_depth_unknown"
    NOT_DOCUMENTED = "return_depth_not_documented"
    UNKNOWN_INITIAL = "unknown_initial_return"
    CENSORED = "ineligible_censored"


class ReturnDepthReason(str, Enum):
    DEPTH_7 = "documented_depth_7"
    DEPTH_8 = "documented_depth_8"
    DEPTH_9 = "documented_depth_9"
    DEPTH_0 = "documented_unknown_depth_0"
    NO_DEPTH = "no_documented_depth_after_lateral_direction"
    MISSING_PREFIX = "missing_service_prefix"
    BOUNDARY = "ambiguous_post_service_boundary"
    MODIFIER = "unsupported_initial_modifier"
    UNKNOWN_SHOT = "unknown_return_shot_type"
    TRUNCATED = "truncated_return_event"
    INCONSISTENT = "inconsistent_token_spans"
    ACE = "censored_ace"
    UNRETURNED = "censored_unreturned_serve"
    FAULT = "censored_service_fault"
    DOUBLE_FAULT = "censored_double_fault"
    SPECIAL = "special_event_before_return"
    INCOMPLETE_LET = "incomplete_let"


_REASONS: Final = {
    ReturnDepthState.OBSERVED: frozenset({ReturnDepthReason.DEPTH_7, ReturnDepthReason.DEPTH_8, ReturnDepthReason.DEPTH_9}),
    ReturnDepthState.UNKNOWN: frozenset({ReturnDepthReason.DEPTH_0}),
    ReturnDepthState.NOT_DOCUMENTED: frozenset({ReturnDepthReason.NO_DEPTH}),
    ReturnDepthState.UNKNOWN_INITIAL: frozenset({ReturnDepthReason.MISSING_PREFIX, ReturnDepthReason.BOUNDARY, ReturnDepthReason.MODIFIER, ReturnDepthReason.UNKNOWN_SHOT, ReturnDepthReason.TRUNCATED, ReturnDepthReason.INCONSISTENT}),
    ReturnDepthState.CENSORED: frozenset({ReturnDepthReason.ACE, ReturnDepthReason.UNRETURNED, ReturnDepthReason.FAULT, ReturnDepthReason.DOUBLE_FAULT, ReturnDepthReason.SPECIAL, ReturnDepthReason.INCOMPLETE_LET}),
}
_TERMINALS: Final = {
    ReturnDepthReason.ACE: "ace", ReturnDepthReason.UNRETURNED: "unreturned_serve",
    ReturnDepthReason.FAULT: "service_fault", ReturnDepthReason.DOUBLE_FAULT: "double_fault",
    ReturnDepthReason.SPECIAL: "special_event", ReturnDepthReason.INCOMPLETE_LET: "incomplete_let",
}


@dataclass(frozen=True)
class ReturnDepthClassification:
    sequence_text: str
    serve_number: int
    previous_attempt_was_fault: bool
    classification_state: ReturnDepthState
    reason_codes: tuple[ReturnDepthReason, ...]
    eligible_for_depth_comparison: bool
    return_event_observed: bool
    actor: str | None
    return_shot_type: str | None
    return_shot_type_label: str | None
    lateral_direction_code: str | None
    lateral_direction_label: str | None
    return_depth_code: str | None
    return_depth_label: str | None
    service_prefix_span: tuple[int, int] | None
    marker_literal: str | None
    marker_span: tuple[int, int] | None
    shot_type_span: tuple[int, int] | None
    lateral_direction_span: tuple[int, int] | None
    return_depth_span: tuple[int, int] | None
    return_event_span: tuple[int, int] | None
    parser_warning_codes: tuple[str, ...]
    parser_warning_spans: tuple[tuple[int, int], ...]
    residual_spans: tuple[tuple[int, int], ...]
    terminal_serve_outcome: str | None
    classification_contract_version: str = CLASSIFICATION_CONTRACT_VERSION


def _serve_number(value: object) -> int:
    if type(value) is not int or value not in (1, 2):
        raise ValueError("serve_number debe ser exactamente el entero 1 o 2.")
    return value


def _input(sequence_text: object, serve_number: object, previous_attempt_was_fault: object) -> None:
    if not isinstance(sequence_text, str):
        raise TypeError("sequence_text debe ser str.")
    _serve_number(serve_number)
    if type(previous_attempt_was_fault) is not bool:
        raise TypeError("previous_attempt_was_fault debe ser bool real.")
    if previous_attempt_was_fault and serve_number != 2:
        raise ValueError("Solo un segundo saque puede seguir a un fault previo.")


def _prefix(parsed: ParseResult) -> ServicePrefix | None:
    if isinstance(parsed.structure, ServicePrefix):
        return parsed.structure
    if isinstance(parsed.structure, (ServeAce, ServeUnreturned, ServeFault)):
        return parsed.structure.prefix
    return None


def _terminal(parsed: ParseResult, previous_fault: bool) -> tuple[ReturnDepthReason, str] | None:
    if isinstance(parsed.structure, ServeAce):
        return ReturnDepthReason.ACE, "ace"
    if isinstance(parsed.structure, ServeUnreturned):
        return ReturnDepthReason.UNRETURNED, "unreturned_serve"
    if isinstance(parsed.structure, ServeFault):
        if parsed.serve_number == 2 and previous_fault:
            return ReturnDepthReason.DOUBLE_FAULT, "double_fault"
        return ReturnDepthReason.FAULT, "service_fault"
    if isinstance(parsed.structure, SpecialCodeStructure):
        return ReturnDepthReason.SPECIAL, "special_event"
    if parsed.structure is None and parsed.raw_sequence and all(token.raw_text == "c" for token in parsed.tokens):
        return ReturnDepthReason.INCOMPLETE_LET, "incomplete_let"
    return None


def _warning_fields(parsed: ParseResult) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    warnings = tuple(sorted(parsed.warnings, key=lambda item: (item.start, item.end, item.code, item.message)))
    return tuple(item.code for item in warnings), tuple((item.start, item.end) for item in warnings)


def _make_result(
    parsed: ParseResult, state: ReturnDepthState, reason: ReturnDepthReason, *,
    previous_fault: bool, prefix: ServicePrefix | None = None, marker: tuple[int, int] | None = None,
    shot_span: tuple[int, int] | None = None, lateral_span: tuple[int, int] | None = None,
    depth_span: tuple[int, int] | None = None, terminal: str | None = None,
) -> ReturnDepthClassification:
    event_span = None if shot_span is None else (shot_span[0], (depth_span or lateral_span or shot_span)[1])
    shot = None if shot_span is None else parsed.raw_sequence[shot_span[0]]
    lateral = None if lateral_span is None else parsed.raw_sequence[lateral_span[0]]
    depth = None if depth_span is None else parsed.raw_sequence[depth_span[0]]
    warning_codes, warning_spans = _warning_fields(parsed)
    return ReturnDepthClassification(
        sequence_text=parsed.raw_sequence or "", serve_number=parsed.serve_number,
        previous_attempt_was_fault=previous_fault, classification_state=state,
        reason_codes=(reason,), eligible_for_depth_comparison=state is ReturnDepthState.OBSERVED,
        return_event_observed=event_span is not None, actor="returner" if event_span else None,
        return_shot_type=shot, return_shot_type_label=None if shot is None else SHOT_TYPES[shot],
        lateral_direction_code=lateral, lateral_direction_label=None if lateral is None else LATERAL_DIRECTIONS[lateral],
        return_depth_code=depth, return_depth_label=None if depth is None else RETURN_DEPTHS[depth],
        service_prefix_span=None if prefix is None else (prefix.start, prefix.end),
        marker_literal=None if marker is None else "+", marker_span=marker,
        shot_type_span=shot_span, lateral_direction_span=lateral_span, return_depth_span=depth_span,
        return_event_span=event_span, parser_warning_codes=warning_codes,
        parser_warning_spans=warning_spans,
        residual_spans=tuple((span.start, span.end) for span in parsed.residual_spans),
        terminal_serve_outcome=terminal,
    )


def _classify(parsed: ParseResult, previous_fault: bool) -> ReturnDepthClassification:
    terminal = _terminal(parsed, previous_fault)
    prefix = _prefix(parsed)
    if terminal is not None:
        return _make_result(parsed, ReturnDepthState.CENSORED, terminal[0], previous_fault=previous_fault, prefix=prefix, terminal=terminal[1])
    if prefix is None:
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.MISSING_PREFIX, previous_fault=previous_fault)
    raw = parsed.raw_sequence or ""
    if prefix.start != 0 or prefix.end <= 0 or prefix.end > len(raw):
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.INCONSISTENT, previous_fault=previous_fault, prefix=prefix)
    cursor, marker = prefix.end, None
    if cursor < len(raw) and raw[cursor] == "+":
        marker = (cursor, cursor + 1)
        cursor += 1
    if cursor >= len(raw):
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.TRUNCATED, previous_fault=previous_fault, prefix=prefix, marker=marker)
    char = raw[cursor]
    if char in {"+", "-", "=", ";", "^"}:
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.MODIFIER, previous_fault=previous_fault, prefix=prefix, marker=marker)
    if char == "q":
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.UNKNOWN_SHOT, previous_fault=previous_fault, prefix=prefix, marker=marker)
    if char not in SHOT_TYPES:
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.BOUNDARY, previous_fault=previous_fault, prefix=prefix, marker=marker)
    shot_span = (cursor, cursor + 1)
    cursor += 1
    if cursor >= len(raw):
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.TRUNCATED, previous_fault=previous_fault, prefix=prefix, marker=marker)
    if raw[cursor] not in LATERAL_DIRECTIONS:
        return _make_result(parsed, ReturnDepthState.UNKNOWN_INITIAL, ReturnDepthReason.BOUNDARY, previous_fault=previous_fault, prefix=prefix, marker=marker)
    lateral_span = (cursor, cursor + 1)
    cursor += 1
    if cursor >= len(raw) or raw[cursor] not in RETURN_DEPTHS:
        return _make_result(parsed, ReturnDepthState.NOT_DOCUMENTED, ReturnDepthReason.NO_DEPTH, previous_fault=previous_fault, prefix=prefix, marker=marker, shot_span=shot_span, lateral_span=lateral_span)
    depth_span = (cursor, cursor + 1)
    code = raw[cursor]
    if code == "0":
        return _make_result(parsed, ReturnDepthState.UNKNOWN, ReturnDepthReason.DEPTH_0, previous_fault=previous_fault, prefix=prefix, marker=marker, shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span)
    reason = {"7": ReturnDepthReason.DEPTH_7, "8": ReturnDepthReason.DEPTH_8, "9": ReturnDepthReason.DEPTH_9}[code]
    return _make_result(parsed, ReturnDepthState.OBSERVED, reason, previous_fault=previous_fault, prefix=prefix, marker=marker, shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span)


def classify_initial_return_depth(sequence_text: str, serve_number: int, parsed: ParseResult, previous_attempt_was_fault: bool = False) -> ReturnDepthClassification:
    """Clasifica P05 usando exclusivamente el cursor local posterior al prefijo."""
    _input(sequence_text, serve_number, previous_attempt_was_fault)
    if not isinstance(parsed, ParseResult):
        raise TypeError("parsed debe ser ParseResult.")
    if parsed.raw_sequence != sequence_text or parsed.serve_number != serve_number:
        raise ValueError("sequence_text y serve_number deben coincidir con ParseResult.")
    validate_parsed_sequence(parsed)
    result = _classify(parsed, previous_attempt_was_fault)
    validate_return_depth_classification(result, parsed=parsed)
    return result


def parse_and_classify_initial_return_depth(sequence_text: str, serve_number: int, previous_attempt_was_fault: bool = False) -> ReturnDepthClassification:
    """Ruta de alto nivel: invoca el parser exactamente una vez."""
    _input(sequence_text, serve_number, previous_attempt_was_fault)
    parsed = parse_sequence(sequence_text, serve_number)
    return classify_initial_return_depth(sequence_text, serve_number, parsed, previous_attempt_was_fault)


def _span(name: str, value: object, length: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not int or type(value[1]) is not int:
        raise TypeError(f"{name} debe ser un span (inicio, fin) de enteros reales.")
    if value[0] < 0 or value[1] <= value[0] or value[1] > length:
        raise ValueError(f"{name} fuera de rango.")
    return value


def validate_return_depth_classification(result: ReturnDepthClassification, *, parsed: ParseResult | None = None) -> None:
    """Reconstituye el contrato P05 canonico y rechaza mutaciones derivadas."""
    if not isinstance(result, ReturnDepthClassification):
        raise TypeError("result debe ser ReturnDepthClassification.")
    _input(result.sequence_text, result.serve_number, result.previous_attempt_was_fault)
    if result.classification_contract_version != CLASSIFICATION_CONTRACT_VERSION:
        raise ValueError("classification_contract_version invalida.")
    if not isinstance(result.classification_state, ReturnDepthState) or not isinstance(result.reason_codes, tuple) or len(result.reason_codes) != 1 or not isinstance(result.reason_codes[0], ReturnDepthReason):
        raise TypeError("Estado o reason_codes fuera del catalogo cerrado.")
    if result.reason_codes[0] not in _REASONS[result.classification_state]:
        raise ValueError("reason_code incompatible con estado.")
    for name in ("service_prefix_span", "marker_span", "shot_type_span", "lateral_direction_span", "return_depth_span", "return_event_span"):
        _span(name, getattr(result, name), len(result.sequence_text))
    if parsed is None:
        parsed = parse_sequence(result.sequence_text, result.serve_number)
    if not isinstance(parsed, ParseResult) or parsed.raw_sequence != result.sequence_text or parsed.serve_number != result.serve_number:
        raise ValueError("ParseResult no corresponde a la clasificacion.")
    validate_parsed_sequence(parsed)
    expected = _classify(parsed, result.previous_attempt_was_fault)
    if result != expected:
        raise ValueError("La clasificacion no coincide con la derivacion canonica P05.")

