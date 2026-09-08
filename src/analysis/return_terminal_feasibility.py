"""P07: terminador documentado inmediatamente del primer resto.

Este extractor es deliberadamente local. Reconoce un resultado solo si el
terminador cierra el mismo evento de primer resto; no reconstruye el rally ni
atribuye golpes posteriores a ningun jugador.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.parsing.serve_sequence import (
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
DOCUMENTED_SHOT_TYPES: Final = frozenset("fbrsvzopuy lmhijkt".replace(" ", ""))
UNKNOWN_SHOT_TYPE: Final = "q"
LATERAL_DIRECTIONS: Final = frozenset({"0", "1", "2", "3"})
RETURN_DEPTHS: Final = frozenset({"0", "7", "8", "9"})
UNFORCED_ERROR_CODES: Final = frozenset({"n", "w", "d", "x", "e", "!"})
UNSUPPORTED_INITIAL_MODIFIERS: Final = frozenset({"+", "-", "=", ";", "^"})


class ReturnTerminalState(str, Enum):
    WINNER = "documented_return_winner"
    FORCED_ERROR = "documented_return_forced_error"
    UNFORCED_ERROR = "documented_return_unforced_error"
    NOT_DOCUMENTED = "return_terminal_not_documented"
    UNKNOWN_INITIAL = "unknown_initial_return"
    CENSORED = "ineligible_censored"


class ReturnTerminalReason(str, Enum):
    WINNER = "immediate_return_winner"
    FORCED_ERROR = "immediate_return_forced_error"
    UNFORCED_ERROR_N = "immediate_return_unforced_error_n"
    UNFORCED_ERROR_W = "immediate_return_unforced_error_w"
    UNFORCED_ERROR_D = "immediate_return_unforced_error_d"
    UNFORCED_ERROR_X = "immediate_return_unforced_error_x"
    UNFORCED_ERROR_E = "immediate_return_unforced_error_e"
    UNFORCED_ERROR_BANG = "immediate_return_unforced_error_!"
    NO_TERMINAL = "local_return_without_terminal"
    NONTERMINAL_CONTENT = "nonterminal_content_after_local_return"
    CHALLENGE_AFTER_RETURN = "challenge_after_local_return"
    MISSING_PREFIX = "missing_service_prefix"
    TRUNCATED = "truncated_return_event"
    BOUNDARY = "ambiguous_initial_boundary"
    MODIFIER = "unsupported_initial_modifier"
    COMPONENT_ORDER = "unsupported_local_component_order"
    ERROR_FORM = "unsupported_error_terminal_form"
    TRAILING_CONTENT = "terminal_with_trailing_content"
    INCONSISTENT = "inconsistent_parse_result"
    ACE = "censored_ace"
    UNRETURNED = "censored_unreturned_serve"
    FAULT = "censored_service_fault"
    DOUBLE_FAULT = "censored_double_fault"
    SPECIAL = "censored_special_event"
    INCOMPLETE_LET = "censored_incomplete_let"


_UNFORCED_REASONS: Final = {
    "n": ReturnTerminalReason.UNFORCED_ERROR_N,
    "w": ReturnTerminalReason.UNFORCED_ERROR_W,
    "d": ReturnTerminalReason.UNFORCED_ERROR_D,
    "x": ReturnTerminalReason.UNFORCED_ERROR_X,
    "e": ReturnTerminalReason.UNFORCED_ERROR_E,
    "!": ReturnTerminalReason.UNFORCED_ERROR_BANG,
}
_REASONS: Final = {
    ReturnTerminalState.WINNER: frozenset({ReturnTerminalReason.WINNER}),
    ReturnTerminalState.FORCED_ERROR: frozenset({ReturnTerminalReason.FORCED_ERROR}),
    ReturnTerminalState.UNFORCED_ERROR: frozenset(_UNFORCED_REASONS.values()),
    ReturnTerminalState.NOT_DOCUMENTED: frozenset({
        ReturnTerminalReason.NO_TERMINAL,
        ReturnTerminalReason.NONTERMINAL_CONTENT,
        ReturnTerminalReason.CHALLENGE_AFTER_RETURN,
    }),
    ReturnTerminalState.UNKNOWN_INITIAL: frozenset({
        ReturnTerminalReason.MISSING_PREFIX,
        ReturnTerminalReason.TRUNCATED,
        ReturnTerminalReason.BOUNDARY,
        ReturnTerminalReason.MODIFIER,
        ReturnTerminalReason.COMPONENT_ORDER,
        ReturnTerminalReason.ERROR_FORM,
        ReturnTerminalReason.TRAILING_CONTENT,
        ReturnTerminalReason.INCONSISTENT,
    }),
    ReturnTerminalState.CENSORED: frozenset({
        ReturnTerminalReason.ACE,
        ReturnTerminalReason.UNRETURNED,
        ReturnTerminalReason.FAULT,
        ReturnTerminalReason.DOUBLE_FAULT,
        ReturnTerminalReason.SPECIAL,
        ReturnTerminalReason.INCOMPLETE_LET,
    }),
}
_SERVE_TERMINALS: Final = {
    ReturnTerminalReason.ACE: "ace",
    ReturnTerminalReason.UNRETURNED: "unreturned_serve",
    ReturnTerminalReason.FAULT: "service_fault",
    ReturnTerminalReason.DOUBLE_FAULT: "double_fault",
    ReturnTerminalReason.SPECIAL: "special_event",
    ReturnTerminalReason.INCOMPLETE_LET: "incomplete_let",
}
_ELIGIBLE_STATES: Final = frozenset({
    ReturnTerminalState.WINNER,
    ReturnTerminalState.FORCED_ERROR,
    ReturnTerminalState.UNFORCED_ERROR,
})


@dataclass(frozen=True)
class ReturnTerminalClassification:
    sequence_text: str
    serve_number: int
    previous_attempt_was_fault: bool
    classification_state: ReturnTerminalState
    reason_code: ReturnTerminalReason
    eligible_for_terminal_comparison: bool
    return_event_observed: bool
    actor: str | None
    service_prefix_span: tuple[int, int] | None
    marker_literal: str | None
    marker_span: tuple[int, int] | None
    return_shot_type_code: str | None
    shot_type_span: tuple[int, int] | None
    lateral_direction_code: str | None
    lateral_direction_span: tuple[int, int] | None
    return_depth_code: str | None
    return_depth_span: tuple[int, int] | None
    return_event_span: tuple[int, int] | None
    terminal_kind: str | None
    terminal_literal: str | None
    terminal_span: tuple[int, int] | None
    error_code: str | None
    error_span: tuple[int, int] | None
    parser_warning_codes: tuple[str, ...]
    parser_warning_spans: tuple[tuple[int, int], ...]
    residual_spans: tuple[tuple[int, int], ...]
    terminal_serve_outcome: str | None
    classification_contract_version: str = CLASSIFICATION_CONTRACT_VERSION


def _validate_input(
    sequence_text: object, serve_number: object, previous_attempt_was_fault: object
) -> None:
    if type(sequence_text) is not str:
        raise TypeError("sequence_text debe ser str real.")
    if type(serve_number) is not int or serve_number not in (1, 2):
        raise ValueError("serve_number debe ser exactamente el entero 1 o 2.")
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


def _serve_terminal(
    parsed: ParseResult, previous_fault: bool
) -> tuple[ReturnTerminalReason, str] | None:
    if isinstance(parsed.structure, ServeAce):
        return ReturnTerminalReason.ACE, "ace"
    if isinstance(parsed.structure, ServeUnreturned):
        return ReturnTerminalReason.UNRETURNED, "unreturned_serve"
    if isinstance(parsed.structure, ServeFault):
        if parsed.serve_number == 2 and previous_fault:
            return ReturnTerminalReason.DOUBLE_FAULT, "double_fault"
        return ReturnTerminalReason.FAULT, "service_fault"
    if isinstance(parsed.structure, SpecialCodeStructure):
        return ReturnTerminalReason.SPECIAL, "special_event"
    if (
        parsed.structure is None
        and parsed.raw_sequence
        and all(token.raw_text == "c" for token in parsed.tokens)
    ):
        return ReturnTerminalReason.INCOMPLETE_LET, "incomplete_let"
    return None


def _warning_fields(
    parsed: ParseResult,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    warnings = tuple(
        sorted(parsed.warnings, key=lambda item: (item.start, item.end, item.code, item.message))
    )
    return (
        tuple(item.code for item in warnings),
        tuple((item.start, item.end) for item in warnings),
    )


def _make_result(
    parsed: ParseResult,
    state: ReturnTerminalState,
    reason: ReturnTerminalReason,
    *,
    previous_fault: bool,
    prefix: ServicePrefix | None = None,
    marker_span: tuple[int, int] | None = None,
    shot_span: tuple[int, int] | None = None,
    lateral_span: tuple[int, int] | None = None,
    depth_span: tuple[int, int] | None = None,
    event_span: tuple[int, int] | None = None,
    terminal_kind: str | None = None,
    terminal_span: tuple[int, int] | None = None,
    error_span: tuple[int, int] | None = None,
    serve_terminal: str | None = None,
) -> ReturnTerminalClassification:
    raw = parsed.raw_sequence or ""
    warning_codes, warning_spans = _warning_fields(parsed)
    return ReturnTerminalClassification(
        sequence_text=raw,
        serve_number=parsed.serve_number,
        previous_attempt_was_fault=previous_fault,
        classification_state=state,
        reason_code=reason,
        eligible_for_terminal_comparison=state in _ELIGIBLE_STATES,
        return_event_observed=event_span is not None,
        actor="returner" if event_span is not None else None,
        service_prefix_span=None if prefix is None else (prefix.start, prefix.end),
        marker_literal=None if marker_span is None else "+",
        marker_span=marker_span,
        return_shot_type_code=None if shot_span is None else raw[shot_span[0]],
        shot_type_span=shot_span,
        lateral_direction_code=None if lateral_span is None else raw[lateral_span[0]],
        lateral_direction_span=lateral_span,
        return_depth_code=None if depth_span is None else raw[depth_span[0]],
        return_depth_span=depth_span,
        return_event_span=event_span,
        terminal_kind=terminal_kind,
        terminal_literal=None if terminal_span is None else raw[terminal_span[0]:terminal_span[1]],
        terminal_span=terminal_span,
        error_code=None if error_span is None else raw[error_span[0]],
        error_span=error_span,
        parser_warning_codes=warning_codes,
        parser_warning_spans=warning_spans,
        residual_spans=tuple((span.start, span.end) for span in parsed.residual_spans),
        terminal_serve_outcome=serve_terminal,
    )


def _local_event_end(raw: str, cursor: int) -> tuple[int, tuple[int, int] | None, tuple[int, int] | None, ReturnTerminalReason | None]:
    """Consume solo los componentes locales autorizados del primer resto."""
    end = cursor + 1
    lateral_span: tuple[int, int] | None = None
    depth_span: tuple[int, int] | None = None
    if end == len(raw):
        return end, lateral_span, depth_span, None
    if raw[end] in LATERAL_DIRECTIONS:
        lateral_span = (end, end + 1)
        end += 1
        if end < len(raw) and raw[end] in RETURN_DEPTHS:
            depth_span = (end, end + 1)
            end += 1
    elif raw[end] in RETURN_DEPTHS:
        return end, lateral_span, depth_span, ReturnTerminalReason.COMPONENT_ORDER
    return end, lateral_span, depth_span, None


def _classify(parsed: ParseResult, previous_fault: bool) -> ReturnTerminalClassification:
    serve_terminal = _serve_terminal(parsed, previous_fault)
    prefix = _prefix(parsed)
    if serve_terminal is not None:
        return _make_result(
            parsed,
            ReturnTerminalState.CENSORED,
            serve_terminal[0],
            previous_fault=previous_fault,
            prefix=prefix,
            serve_terminal=serve_terminal[1],
        )
    if prefix is None:
        return _make_result(
            parsed,
            ReturnTerminalState.UNKNOWN_INITIAL,
            ReturnTerminalReason.MISSING_PREFIX,
            previous_fault=previous_fault,
        )
    raw = parsed.raw_sequence or ""
    if prefix.start != 0 or prefix.end <= 0 or prefix.end > len(raw):
        return _make_result(
            parsed,
            ReturnTerminalState.UNKNOWN_INITIAL,
            ReturnTerminalReason.INCONSISTENT,
            previous_fault=previous_fault,
            prefix=prefix,
        )

    cursor = prefix.end
    marker_span = None
    if cursor < len(raw) and raw[cursor] == "+":
        marker_span = (cursor, cursor + 1)
        cursor += 1
    if cursor >= len(raw):
        return _make_result(
            parsed,
            ReturnTerminalState.UNKNOWN_INITIAL,
            ReturnTerminalReason.TRUNCATED,
            previous_fault=previous_fault,
            prefix=prefix,
            marker_span=marker_span,
        )
    shot = raw[cursor]
    if shot in UNSUPPORTED_INITIAL_MODIFIERS:
        return _make_result(
            parsed,
            ReturnTerminalState.UNKNOWN_INITIAL,
            ReturnTerminalReason.MODIFIER,
            previous_fault=previous_fault,
            prefix=prefix,
            marker_span=marker_span,
        )
    if shot not in DOCUMENTED_SHOT_TYPES | {UNKNOWN_SHOT_TYPE}:
        return _make_result(
            parsed,
            ReturnTerminalState.UNKNOWN_INITIAL,
            ReturnTerminalReason.BOUNDARY,
            previous_fault=previous_fault,
            prefix=prefix,
            marker_span=marker_span,
        )

    shot_span = (cursor, cursor + 1)
    event_end, lateral_span, depth_span, event_error = _local_event_end(raw, cursor)
    if event_error is not None:
        return _make_result(
            parsed,
            ReturnTerminalState.UNKNOWN_INITIAL,
            event_error,
            previous_fault=previous_fault,
            prefix=prefix,
            marker_span=marker_span,
        )
    event_span = (cursor, event_end)
    if event_end == len(raw):
        return _make_result(
            parsed,
            ReturnTerminalState.NOT_DOCUMENTED,
            ReturnTerminalReason.NO_TERMINAL,
            previous_fault=previous_fault,
            prefix=prefix,
            marker_span=marker_span,
            shot_span=shot_span,
            lateral_span=lateral_span,
            depth_span=depth_span,
            event_span=event_span,
        )

    next_character = raw[event_end]
    if next_character == "*":
        if event_end + 1 != len(raw):
            return _make_result(
                parsed, ReturnTerminalState.UNKNOWN_INITIAL, ReturnTerminalReason.TRAILING_CONTENT,
                previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
                shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
            )
        return _make_result(
            parsed, ReturnTerminalState.WINNER, ReturnTerminalReason.WINNER,
            previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
            shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
            terminal_kind="winner", terminal_span=(event_end, event_end + 1),
        )
    if next_character == "#":
        if event_end + 1 != len(raw):
            return _make_result(
                parsed, ReturnTerminalState.UNKNOWN_INITIAL, ReturnTerminalReason.TRAILING_CONTENT,
                previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
                shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
            )
        return _make_result(
            parsed, ReturnTerminalState.FORCED_ERROR, ReturnTerminalReason.FORCED_ERROR,
            previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
            shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
            terminal_kind="forced_error", terminal_span=(event_end, event_end + 1),
        )
    if next_character in UNFORCED_ERROR_CODES:
        error_span = (event_end, event_end + 1)
        at = event_end + 1
        if at >= len(raw) or raw[at] != "@":
            return _make_result(
                parsed, ReturnTerminalState.UNKNOWN_INITIAL, ReturnTerminalReason.ERROR_FORM,
                previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
                shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
            )
        if at + 1 != len(raw):
            return _make_result(
                parsed, ReturnTerminalState.UNKNOWN_INITIAL, ReturnTerminalReason.TRAILING_CONTENT,
                previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
                shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
            )
        return _make_result(
            parsed, ReturnTerminalState.UNFORCED_ERROR, _UNFORCED_REASONS[next_character],
            previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
            shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
            terminal_kind="unforced_error", terminal_span=(at, at + 1), error_span=error_span,
        )
    if next_character == "g":
        # ``g`` se documenta exclusivamente como falta de pie del saque, no
        # como componente de un error de rally.
        return _make_result(
            parsed, ReturnTerminalState.UNKNOWN_INITIAL, ReturnTerminalReason.ERROR_FORM,
            previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
            shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
        )
    if next_character == "C":
        return _make_result(
            parsed, ReturnTerminalState.NOT_DOCUMENTED, ReturnTerminalReason.CHALLENGE_AFTER_RETURN,
            previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
            shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
        )
    if next_character in UNSUPPORTED_INITIAL_MODIFIERS or next_character == "@":
        return _make_result(
            parsed, ReturnTerminalState.UNKNOWN_INITIAL, ReturnTerminalReason.COMPONENT_ORDER,
            previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
            shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
        )
    if next_character in DOCUMENTED_SHOT_TYPES | {UNKNOWN_SHOT_TYPE}:
        return _make_result(
            parsed, ReturnTerminalState.NOT_DOCUMENTED, ReturnTerminalReason.NONTERMINAL_CONTENT,
            previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
            shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
        )
    return _make_result(
        parsed, ReturnTerminalState.UNKNOWN_INITIAL, ReturnTerminalReason.BOUNDARY,
        previous_fault=previous_fault, prefix=prefix, marker_span=marker_span,
        shot_span=shot_span, lateral_span=lateral_span, depth_span=depth_span, event_span=event_span,
    )


def classify_initial_return_terminal(
    sequence_text: str,
    serve_number: int,
    parsed: ParseResult,
    previous_attempt_was_fault: bool = False,
) -> ReturnTerminalClassification:
    """Clasifica el unico terminador unido al primer resto local, si existe."""
    _validate_input(sequence_text, serve_number, previous_attempt_was_fault)
    if not isinstance(parsed, ParseResult):
        raise TypeError("parsed debe ser ParseResult.")
    if parsed.raw_sequence != sequence_text or parsed.serve_number != serve_number:
        raise ValueError("sequence_text y serve_number deben coincidir con ParseResult.")
    validate_parsed_sequence(parsed)
    result = _classify(parsed, previous_attempt_was_fault)
    validate_return_terminal_classification(result, parsed)
    return result


def parse_and_classify_initial_return_terminal(
    sequence_text: str,
    serve_number: int,
    previous_attempt_was_fault: bool = False,
) -> ReturnTerminalClassification:
    """Ruta de alto nivel; llama al parser exactamente una vez."""
    _validate_input(sequence_text, serve_number, previous_attempt_was_fault)
    parsed = parse_sequence(sequence_text, serve_number)
    return classify_initial_return_terminal(
        sequence_text, serve_number, parsed, previous_attempt_was_fault
    )


def validate_return_terminal_classification(
    classification: ReturnTerminalClassification,
    parsed: ParseResult | None = None,
) -> None:
    """Reconstruye el contrato P07 y rechaza mutaciones semanticas."""
    if not isinstance(classification, ReturnTerminalClassification):
        raise TypeError("classification debe ser ReturnTerminalClassification.")
    _validate_input(
        classification.sequence_text,
        classification.serve_number,
        classification.previous_attempt_was_fault,
    )
    if classification.classification_contract_version != CLASSIFICATION_CONTRACT_VERSION:
        raise ValueError("classification_contract_version invalida.")
    if not isinstance(classification.classification_state, ReturnTerminalState) or not isinstance(classification.reason_code, ReturnTerminalReason):
        raise TypeError("Estado o reason_code fuera del catalogo cerrado.")
    if classification.reason_code not in _REASONS[classification.classification_state]:
        raise ValueError("reason_code incompatible con estado.")
    if type(classification.eligible_for_terminal_comparison) is not bool or type(classification.return_event_observed) is not bool:
        raise TypeError("Los flags deben ser bool reales.")
    if parsed is None:
        parsed = parse_sequence(classification.sequence_text, classification.serve_number)
    if not isinstance(parsed, ParseResult) or parsed.raw_sequence != classification.sequence_text or parsed.serve_number != classification.serve_number:
        raise ValueError("ParseResult no corresponde a la clasificacion.")
    validate_parsed_sequence(parsed)
    expected = _classify(parsed, classification.previous_attempt_was_fault)
    if classification != expected:
        raise ValueError("La clasificacion no coincide con la derivacion canonica P07.")
