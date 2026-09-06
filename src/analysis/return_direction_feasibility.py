"""P04: extractor local, sintetico y conservador del primer resto.

No hace E/S, no interpreta el rally completo ni atribuye golpes posteriores.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.parsing.serve_sequence import (
    ParseResult, ServicePrefix, ServeAce, ServeFault, ServeUnreturned,
    SpecialCodeStructure, parse_sequence, validate_parsed_sequence,
)

CLASSIFICATION_CONTRACT_VERSION: Final = "1.0.0"
SHOT_TYPES: Final = {
    "f": "forehand", "b": "backhand", "r": "forehand_slice", "s": "backhand_slice",
    "v": "forehand_volley", "z": "backhand_volley", "o": "overhead",
    "p": "backhand_overhead", "u": "forehand_drop_shot", "y": "backhand_drop_shot",
    "l": "forehand_lob", "m": "backhand_lob", "h": "forehand_half_volley",
    "i": "backhand_half_volley", "j": "forehand_topspin_volley",
    "k": "backhand_topspin_volley", "t": "special_shot", "q": "unknown_shot_type",
}
LATERAL_DIRECTIONS: Final = {
    "1": "right_side_of_right_handed_opponent_or_left_of_left_handed",
    "2": "centre",
    "3": "left_side_of_right_handed_opponent_or_right_of_left_handed",
    "0": "unknown",
}
RETURN_DEPTHS: Final = {
    "7": "service_boxes", "8": "behind_service_line_closer_to_service_line",
    "9": "closer_to_baseline", "0": "unknown",
}
SERVICE_DIRECTIONS: Final = {"4": "wide", "5": "body", "6": "down_the_t", "0": "unknown"}


class ReturnDirectionState(str, Enum):
    OBSERVED = "return_direction_observed"
    UNKNOWN = "return_direction_unknown"
    UNKNOWN_INITIAL = "unknown_initial_return"
    ACE = "ace"
    UNRETURNED = "unreturned_serve"
    FAULT = "service_fault"
    SPECIAL = "special_or_incomplete"


class ReturnDirectionReason(str, Enum):
    D1 = "observed_lateral_direction_1"
    D2 = "observed_lateral_direction_2"
    D3 = "observed_lateral_direction_3"
    D0 = "documented_unknown_direction_0"
    NO_DIRECTION = "localized_return_without_direction"
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
    CHALLENGE_PENALTY = "challenge_or_penalty_before_return"


_REASONS: Final = {
    ReturnDirectionState.OBSERVED: frozenset({ReturnDirectionReason.D1, ReturnDirectionReason.D2, ReturnDirectionReason.D3}),
    ReturnDirectionState.UNKNOWN: frozenset({ReturnDirectionReason.D0, ReturnDirectionReason.NO_DIRECTION}),
    ReturnDirectionState.UNKNOWN_INITIAL: frozenset({ReturnDirectionReason.MISSING_PREFIX, ReturnDirectionReason.BOUNDARY, ReturnDirectionReason.MODIFIER, ReturnDirectionReason.UNKNOWN_SHOT, ReturnDirectionReason.TRUNCATED, ReturnDirectionReason.INCONSISTENT}),
    ReturnDirectionState.ACE: frozenset({ReturnDirectionReason.ACE}),
    ReturnDirectionState.UNRETURNED: frozenset({ReturnDirectionReason.UNRETURNED}),
    ReturnDirectionState.FAULT: frozenset({ReturnDirectionReason.FAULT, ReturnDirectionReason.DOUBLE_FAULT}),
    ReturnDirectionState.SPECIAL: frozenset({ReturnDirectionReason.SPECIAL, ReturnDirectionReason.INCOMPLETE_LET, ReturnDirectionReason.CHALLENGE_PENALTY}),
}
_TERMINALS: Final = {
    ReturnDirectionReason.ACE: "ace", ReturnDirectionReason.UNRETURNED: "unreturned_serve",
    ReturnDirectionReason.FAULT: "service_fault", ReturnDirectionReason.DOUBLE_FAULT: "double_fault",
    ReturnDirectionReason.SPECIAL: "special_event", ReturnDirectionReason.CHALLENGE_PENALTY: "special_event",
    ReturnDirectionReason.INCOMPLETE_LET: "incomplete_let",
}
_CENSORED: Final = frozenset({ReturnDirectionState.ACE, ReturnDirectionState.UNRETURNED, ReturnDirectionState.FAULT, ReturnDirectionState.SPECIAL})


@dataclass(frozen=True)
class ReturnDirectionClassification:
    serve_number: int
    sequence_length: int
    service_direction_code: str | None
    service_direction_label: str | None
    service_prefix_start: int | None
    service_prefix_end: int | None
    initial_annotation_code: str | None
    initial_annotation_start: int | None
    initial_annotation_end: int | None
    return_event_start: int | None
    return_event_end: int | None
    return_actor: str | None
    return_shot_type_code: str | None
    return_shot_type_label: str | None
    lateral_direction_code: str | None
    lateral_direction_label: str | None
    lateral_direction_start: int | None
    lateral_direction_end: int | None
    return_depth_code: str | None
    return_depth_label: str | None
    return_depth_start: int | None
    return_depth_end: int | None
    classification_state: ReturnDirectionState
    reason_code: ReturnDirectionReason
    return_opportunity_observed: bool
    return_event_observed: bool
    eligible_for_direction_analysis: bool
    terminal_serve_outcome: str | None
    parser_warning_codes: tuple[str, ...]
    parser_warning_spans: tuple[tuple[int, int], ...]
    has_residual: bool
    residual_spans: tuple[tuple[int, int], ...]
    previous_attempt_was_fault: bool
    sequence_text: str
    classification_contract_version: str = CLASSIFICATION_CONTRACT_VERSION


def _serve_number(value: object) -> int:
    if type(value) is not int or value not in (1, 2):
        raise ValueError("serve_number debe ser exactamente el entero 1 o 2.")
    return value


def _prefix(parsed: ParseResult) -> ServicePrefix | None:
    if isinstance(parsed.structure, ServicePrefix):
        return parsed.structure
    if isinstance(parsed.structure, (ServeAce, ServeUnreturned, ServeFault)):
        return parsed.structure.prefix
    return None


def _terminal(parsed: ParseResult, prior_fault: bool) -> tuple[ReturnDirectionState, ReturnDirectionReason, str] | None:
    structure = parsed.structure
    if isinstance(structure, ServeAce): return ReturnDirectionState.ACE, ReturnDirectionReason.ACE, "ace"
    if isinstance(structure, ServeUnreturned): return ReturnDirectionState.UNRETURNED, ReturnDirectionReason.UNRETURNED, "unreturned_serve"
    if isinstance(structure, ServeFault):
        if parsed.serve_number == 2 and prior_fault:
            return ReturnDirectionState.FAULT, ReturnDirectionReason.DOUBLE_FAULT, "double_fault"
        return ReturnDirectionState.FAULT, ReturnDirectionReason.FAULT, "service_fault"
    if isinstance(structure, SpecialCodeStructure):
        reason = ReturnDirectionReason.CHALLENGE_PENALTY if structure.code in {"C", "P", "Q", "R", "S"} else ReturnDirectionReason.SPECIAL
        return ReturnDirectionState.SPECIAL, reason, "special_event"
    if structure is None and parsed.raw_sequence and all(token.raw_text == "c" for token in parsed.tokens):
        return ReturnDirectionState.SPECIAL, ReturnDirectionReason.INCOMPLETE_LET, "incomplete_let"
    return None


def _warning_fields(parsed: ParseResult) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    ordered = tuple(sorted(parsed.warnings, key=lambda item: (item.start, item.end, item.code)))
    return tuple(item.code for item in ordered), tuple((item.start, item.end) for item in ordered)


def _result(parsed: ParseResult, state: ReturnDirectionState, reason: ReturnDirectionReason, *, prefix: ServicePrefix | None = None, annotation: int | None = None, event: tuple[int, int] | None = None, shot: str | None = None, direction: str | None = None, depth: str | None = None, terminal: str | None = None, previous_attempt_was_fault: bool = False) -> ReturnDirectionClassification:
    start, end = (None, None) if event is None else event
    warning_codes, warning_spans = _warning_fields(parsed)
    result = ReturnDirectionClassification(
        parsed.serve_number, len(parsed.raw_sequence or ""),
        None if prefix is None else prefix.direction, None if prefix is None else prefix.direction_value,
        None if prefix is None else prefix.start, None if prefix is None else prefix.end,
        None if annotation is None else "+", annotation, None if annotation is None else annotation + 1,
        start, end, None if event is None else "returner", shot, None if shot is None else SHOT_TYPES[shot],
        direction, None if direction is None else LATERAL_DIRECTIONS[direction], None if direction is None else start + 1, None if direction is None else start + 2,
        depth, None if depth is None else RETURN_DEPTHS[depth], None if depth is None else end - 1, None if depth is None else end,
        state, reason, event is not None, event is not None, state is ReturnDirectionState.OBSERVED,
        terminal, warning_codes, warning_spans, bool(parsed.residual_spans),
        tuple((span.start, span.end) for span in parsed.residual_spans),
        previous_attempt_was_fault, parsed.raw_sequence,
    )
    validate_return_direction_classification(result, parsed=parsed)
    return result


def classify_initial_return_direction(sequence_text: str, serve_number: int, parsed: ParseResult, *, previous_attempt_was_fault: bool = False) -> ReturnDirectionClassification:
    """Usa el cursor posterior al prefijo para localizar solo el primer retorno."""
    if not isinstance(sequence_text, str): raise TypeError("sequence_text debe ser str.")
    _serve_number(serve_number)
    if type(previous_attempt_was_fault) is not bool: raise TypeError("previous_attempt_was_fault debe ser bool.")
    if previous_attempt_was_fault and serve_number != 2: raise ValueError("Solo un segundo saque puede seguir a un fault previo.")
    if not isinstance(parsed, ParseResult): raise TypeError("parsed debe ser ParseResult.")
    if parsed.raw_sequence != sequence_text or parsed.serve_number != serve_number: raise ValueError("sequence_text y serve_number deben coincidir con ParseResult.")
    validate_parsed_sequence(parsed)
    terminal = _terminal(parsed, previous_attempt_was_fault)
    prefix = _prefix(parsed)
    def emit(state: ReturnDirectionState, reason: ReturnDirectionReason, **kwargs: object) -> ReturnDirectionClassification:
        return _result(parsed, state, reason, previous_attempt_was_fault=previous_attempt_was_fault, **kwargs)
    if terminal is not None:
        state, reason, terminal_outcome = terminal
        return emit(state, reason, prefix=prefix, terminal=terminal_outcome)
    if prefix is None: return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.MISSING_PREFIX)
    if prefix.start != 0 or prefix.end <= 0 or prefix.end > len(sequence_text):
        return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.INCONSISTENT, prefix=prefix)
    cursor, annotation = prefix.end, None
    if cursor < len(sequence_text) and sequence_text[cursor] == "+":
        annotation, cursor = cursor, cursor + 1
    if cursor >= len(sequence_text): return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.TRUNCATED, prefix=prefix, annotation=annotation)
    character = sequence_text[cursor]
    if character in {"+", "-", "=", ";", "^"}: return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.MODIFIER, prefix=prefix, annotation=annotation)
    if character == "q": return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.UNKNOWN_SHOT, prefix=prefix, annotation=annotation)
    if character not in SHOT_TYPES: return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.BOUNDARY, prefix=prefix, annotation=annotation)
    shot, next_cursor = character, cursor + 1
    if next_cursor >= len(sequence_text): return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.TRUNCATED, prefix=prefix, annotation=annotation)
    component = sequence_text[next_cursor]
    if component in {"1", "2", "3"}:
        end, depth = next_cursor + 1, None
        if end < len(sequence_text) and sequence_text[end] in RETURN_DEPTHS:
            depth, end = sequence_text[end], end + 1
        reason = {"1": ReturnDirectionReason.D1, "2": ReturnDirectionReason.D2, "3": ReturnDirectionReason.D3}[component]
        return emit(ReturnDirectionState.OBSERVED, reason, prefix=prefix, annotation=annotation, event=(cursor, end), shot=shot, direction=component, depth=depth)
    if component == "0":
        return emit(ReturnDirectionState.UNKNOWN, ReturnDirectionReason.D0, prefix=prefix, annotation=annotation, event=(cursor, next_cursor + 1), shot=shot, direction="0")
    if component in {"*", "#", "@", "n", "w", "d", "x", "e", "!"}:
        return emit(ReturnDirectionState.UNKNOWN, ReturnDirectionReason.NO_DIRECTION, prefix=prefix, annotation=annotation, event=(cursor, next_cursor), shot=shot)
    return emit(ReturnDirectionState.UNKNOWN_INITIAL, ReturnDirectionReason.BOUNDARY, prefix=prefix, annotation=annotation)


def parse_and_classify_initial_return_direction(sequence_text: str, serve_number: int, *, previous_attempt_was_fault: bool = False) -> ReturnDirectionClassification:
    if not isinstance(sequence_text, str): raise TypeError("sequence_text debe ser str.")
    _serve_number(serve_number)
    return classify_initial_return_direction(sequence_text, serve_number, parse_sequence(sequence_text, serve_number), previous_attempt_was_fault=previous_attempt_was_fault)


def _span(name: str, start: object, end: object, length: int) -> tuple[int, int] | None:
    if (start is None) != (end is None): raise ValueError(f"{name} requiere dos extremos o ninguno.")
    if start is None: return None
    if type(start) is not int or type(end) is not int: raise TypeError(f"{name} debe usar enteros reales.")
    if start < 0 or end <= start or end > length: raise ValueError(f"{name} fuera de rango.")
    return start, end


def validate_return_direction_classification(result: ReturnDirectionClassification, *, parsed: ParseResult | None = None) -> None:
    """Valida dominios, spans, cursor, actor y catalogo cerrado de P04."""
    if not isinstance(result, ReturnDirectionClassification): raise TypeError("result debe ser ReturnDirectionClassification.")
    _serve_number(result.serve_number)
    if type(result.sequence_length) is not int or result.sequence_length < 0: raise ValueError("sequence_length invalida.")
    if not isinstance(result.sequence_text, str) or result.sequence_length != len(result.sequence_text): raise ValueError("sequence_text y sequence_length no reconcilian.")
    if type(result.previous_attempt_was_fault) is not bool: raise TypeError("previous_attempt_was_fault debe ser bool real.")
    if result.previous_attempt_was_fault and result.serve_number != 2: raise ValueError("Solo el segundo saque puede tener fault previo.")
    if result.classification_contract_version != CLASSIFICATION_CONTRACT_VERSION: raise ValueError("classification_contract_version invalida.")
    if not isinstance(result.classification_state, ReturnDirectionState) or not isinstance(result.reason_code, ReturnDirectionReason): raise TypeError("Estado o reason_code fuera de catalogo.")
    if result.reason_code not in _REASONS[result.classification_state]: raise ValueError("reason_code incompatible con estado.")
    for flag in (result.return_opportunity_observed, result.return_event_observed, result.eligible_for_direction_analysis, result.has_residual):
        if type(flag) is not bool: raise TypeError("Los flags deben ser bool reales.")
    prefix = _span("service_prefix", result.service_prefix_start, result.service_prefix_end, result.sequence_length)
    annotation = _span("initial_annotation", result.initial_annotation_start, result.initial_annotation_end, result.sequence_length)
    event = _span("return_event", result.return_event_start, result.return_event_end, result.sequence_length)
    lateral = _span("lateral_direction", result.lateral_direction_start, result.lateral_direction_end, result.sequence_length)
    depth = _span("return_depth", result.return_depth_start, result.return_depth_end, result.sequence_length)
    if parsed is None:
        parsed = parse_sequence(result.sequence_text, result.serve_number)
    elif not isinstance(parsed, ParseResult) or parsed.raw_sequence != result.sequence_text or parsed.serve_number != result.serve_number:
        raise ValueError("ParseResult no corresponde a la clasificacion.")
    validate_parsed_sequence(parsed)
    expected_prefix = _prefix(parsed)
    expected_prefix_fields = (None, None, None, None) if expected_prefix is None else (expected_prefix.direction, expected_prefix.direction_value, expected_prefix.start, expected_prefix.end)
    if (result.service_direction_code, result.service_direction_label, result.service_prefix_start, result.service_prefix_end) != expected_prefix_fields:
        raise ValueError("El prefijo no corresponde a sequence_text.")
    expected_warning_codes, expected_warning_spans = _warning_fields(parsed)
    if (result.parser_warning_codes, result.parser_warning_spans) != (expected_warning_codes, expected_warning_spans):
        raise ValueError("Warnings no corresponden al parseo canonico.")
    expected_residual_spans = tuple((span.start, span.end) for span in parsed.residual_spans)
    if not isinstance(result.residual_spans, tuple) or result.residual_spans != expected_residual_spans or result.has_residual != bool(expected_residual_spans):
        raise ValueError("Residual no corresponde al parseo canonico.")
    if result.service_direction_code is None:
        if result.service_direction_label is not None or prefix is not None: raise ValueError("Prefijo ausente con direccion.")
    elif result.service_direction_code not in SERVICE_DIRECTIONS or result.service_direction_label != SERVICE_DIRECTIONS[result.service_direction_code] or prefix is None:
        raise ValueError("Direccion de servicio invalida.")
    if annotation is None:
        if result.initial_annotation_code is not None: raise ValueError("Anotacion sin span.")
    elif result.initial_annotation_code != "+" or prefix is None or annotation != (prefix[1], prefix[1] + 1):
        raise ValueError("La anotacion debe ser + contiguo al prefijo.")
    elif result.sequence_text[annotation[0]] != "+": raise ValueError("Anotacion no coincide con sequence_text.")
    if result.return_event_observed != (event is not None) or result.return_opportunity_observed != (event is not None): raise ValueError("Flags de evento incoherentes.")
    if event is None:
        if any(value is not None for value in (result.return_actor, result.return_shot_type_code, result.return_shot_type_label, result.lateral_direction_code, result.lateral_direction_label, lateral, result.return_depth_code, result.return_depth_label, depth)): raise ValueError("Sin evento no puede haber componentes de retorno.")
    else:
        expected_start = annotation[1] if annotation is not None else prefix[1] if prefix is not None else None
        if event[0] != expected_start or result.return_actor != "returner": raise ValueError("El actor solo existe en el evento inicial formal.")
        if result.return_shot_type_code not in SHOT_TYPES or result.return_shot_type_code == "q" or result.return_shot_type_label != SHOT_TYPES[result.return_shot_type_code]: raise ValueError("Tipo de retorno invalido.")
        if result.sequence_text[event[0]] != result.return_shot_type_code: raise ValueError("Tipo de retorno no coincide con sequence_text.")
    if result.lateral_direction_code not in {None, "0", "1", "2", "3"} or result.return_depth_code not in {None, "0", "7", "8", "9"}: raise ValueError("Dominios lateral/profundidad invalidos.")
    if (result.lateral_direction_code is None) != (lateral is None) or (result.return_depth_code is None) != (depth is None): raise ValueError("Codigo y span deben coexistir.")
    if result.lateral_direction_code is not None and (event is None or lateral != (event[0] + 1, event[0] + 2) or event[1] < lateral[1] or result.lateral_direction_label != LATERAL_DIRECTIONS[result.lateral_direction_code]): raise ValueError("Direccion lateral invalida.")
    if lateral is not None and result.sequence_text[lateral[0]] != result.lateral_direction_code: raise ValueError("Direccion lateral no coincide con sequence_text.")
    if result.lateral_direction_code is None and result.lateral_direction_label is not None: raise ValueError("Etiqueta lateral sin codigo.")
    if result.return_depth_code is not None and (event is None or result.lateral_direction_code not in {"1", "2", "3"} or depth != (event[1] - 1, event[1]) or depth[0] != lateral[1] or result.return_depth_label != RETURN_DEPTHS[result.return_depth_code]): raise ValueError("Profundidad invalida.")
    if depth is not None and result.sequence_text[depth[0]] != result.return_depth_code: raise ValueError("Profundidad no coincide con sequence_text.")
    if result.return_depth_code is None and result.return_depth_label is not None: raise ValueError("Etiqueta de profundidad sin codigo.")
    observed = result.classification_state is ReturnDirectionState.OBSERVED
    if result.eligible_for_direction_analysis != observed: raise ValueError("Elegibilidad incompatible.")
    if observed and result.lateral_direction_code not in {"1", "2", "3"}: raise ValueError("Observed exige 1, 2 o 3.")
    if result.classification_state is ReturnDirectionState.UNKNOWN:
        if event is None: raise ValueError("Unknown requiere evento localizado.")
        if result.reason_code is ReturnDirectionReason.D0 and result.lateral_direction_code != "0": raise ValueError("D0 exige codigo 0.")
        if result.reason_code is ReturnDirectionReason.NO_DIRECTION and result.lateral_direction_code is not None: raise ValueError("NO_DIRECTION no declara lateral.")
    if result.classification_state is ReturnDirectionState.UNKNOWN_INITIAL and event is not None: raise ValueError("Unknown initial no contiene evento.")
    if result.classification_state is ReturnDirectionState.OBSERVED:
        expected_reason = {"1": ReturnDirectionReason.D1, "2": ReturnDirectionReason.D2, "3": ReturnDirectionReason.D3}[result.lateral_direction_code]
        if result.reason_code is not expected_reason: raise ValueError("Reason observed no corresponde a direccion lateral.")
    if result.reason_code is ReturnDirectionReason.NO_DIRECTION and event != (event[0], event[0] + 1): raise ValueError("Retorno localizado sin direccion termina en el tipo de golpe.")
    if result.reason_code is ReturnDirectionReason.D0 and event != (event[0], event[0] + 2): raise ValueError("Direccion 0 debe ocupar el campo lateral inmediato.")
    if result.classification_state is ReturnDirectionState.OBSERVED:
        expected_event_end = depth[1] if depth is not None else lateral[1]
        if event[1] != expected_event_end: raise ValueError("El evento observado termina en su ultimo componente local.")
    if result.classification_state in _CENSORED:
        terminal = _terminal(parsed, result.previous_attempt_was_fault)
        if event is not None or result.terminal_serve_outcome != _TERMINALS[result.reason_code] or terminal != (result.classification_state, result.reason_code, result.terminal_serve_outcome): raise ValueError("Censura incompatible.")
    elif result.terminal_serve_outcome is not None: raise ValueError("Solo censuras declaran terminal.")
    if not isinstance(result.parser_warning_codes, tuple) or not isinstance(result.parser_warning_spans, tuple) or len(result.parser_warning_codes) != len(result.parser_warning_spans): raise ValueError("Warnings invalidos.")
    items = tuple(zip(result.parser_warning_spans, result.parser_warning_codes))
    for span, code in items:
        if not isinstance(code, str) or not code.startswith("W_") or not isinstance(span, tuple) or len(span) != 2: raise TypeError("Warning invalido.")
        _span("warning", span[0], span[1], result.sequence_length)
    if items != tuple(sorted(items, key=lambda item: (item[0][0], item[0][1], item[1]))) or len(set(items)) != len(items): raise ValueError("Warnings deben ser ordenados; iguales posiciones no se duplican.")
