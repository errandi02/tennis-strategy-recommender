"""P06: extractor sintético, local y conservador del tipo del primer resto.

No abre datos ni interpreta el rally. Reconoce únicamente el código literal
situado en el cursor formal posterior al prefijo de servicio y, de existir, a
un marcador ``+`` inmediato de P03.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping

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
DOCUMENTED_SHOT_TYPES: Final[Mapping[str, str]] = MappingProxyType({
    "f": "derecha, excluidos slices y golpes especiales",
    "b": "revés, excluidos slices y golpes especiales",
    "r": "slice de derecha",
    "s": "slice de revés",
    "v": "volea de derecha",
    "z": "volea de revés",
    "o": "remate estándar",
    "p": "remate de revés",
    "u": "dejada de derecha",
    "y": "dejada de revés",
    "l": "globo de derecha",
    "m": "globo de revés",
    "h": "media volea de derecha",
    "i": "media volea de revés",
    "j": "volea liftada de derecha",
    "k": "volea liftada de revés",
    "t": "golpe especial, incluido trick shot o tweener",
})
UNKNOWN_SHOT_TYPE_CODE: Final = "q"
UNKNOWN_SHOT_TYPE_DESCRIPTION: Final = "tipo de golpe desconocido"
SHOT_TYPE_FAMILY_MEMBERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "groundstroke": frozenset({"f", "b"}),
    "slice": frozenset({"r", "s"}),
    "volley": frozenset({"v", "z"}),
    "overhead": frozenset({"o", "p"}),
    "drop_shot": frozenset({"u", "y"}),
    "lob": frozenset({"l", "m"}),
    "half_volley": frozenset({"h", "i"}),
    "swinging_volley": frozenset({"j", "k"}),
    "special": frozenset({"t"}),
})
SHOT_TYPE_FAMILIES: Final[Mapping[str, str]] = MappingProxyType({
    code: family
    for family, codes in SHOT_TYPE_FAMILY_MEMBERS.items()
    for code in codes
})


def _validate_family_catalogue() -> None:
    observed = set(DOCUMENTED_SHOT_TYPES)
    members = [code for codes in SHOT_TYPE_FAMILY_MEMBERS.values() for code in codes]
    if set(members) != observed or len(members) != len(set(members)):
        raise ValueError("Las familias P06 deben ser exhaustivas y excluyentes.")
    if set(SHOT_TYPE_FAMILIES) != observed:
        raise ValueError("El indice codigo-familia P06 no reconcilia.")
    if UNKNOWN_SHOT_TYPE_CODE in SHOT_TYPE_FAMILIES:
        raise ValueError("q no puede pertenecer a una familia comparable.")


_validate_family_catalogue()


class ReturnShotTypeState(str, Enum):
    OBSERVED = "return_shot_type_observed"
    UNKNOWN = "return_shot_type_unknown"
    UNKNOWN_INITIAL = "unknown_initial_return"
    CENSORED = "ineligible_censored"


class ReturnShotTypeReason(str, Enum):
    F = "documented_return_shot_type_f"
    B = "documented_return_shot_type_b"
    R = "documented_return_shot_type_r"
    S = "documented_return_shot_type_s"
    V = "documented_return_shot_type_v"
    Z = "documented_return_shot_type_z"
    O = "documented_return_shot_type_o"
    P = "documented_return_shot_type_p"
    U = "documented_return_shot_type_u"
    Y = "documented_return_shot_type_y"
    L = "documented_return_shot_type_l"
    M = "documented_return_shot_type_m"
    H = "documented_return_shot_type_h"
    I = "documented_return_shot_type_i"
    J = "documented_return_shot_type_j"
    K = "documented_return_shot_type_k"
    T = "documented_return_shot_type_t"
    Q = "documented_unknown_return_shot_type_q"
    MISSING_PREFIX = "missing_service_prefix"
    BOUNDARY = "ambiguous_post_service_boundary"
    MODIFIER = "unsupported_initial_modifier"
    TRUNCATED = "truncated_return_event"
    INCONSISTENT = "inconsistent_token_spans"
    ACE = "censored_ace"
    UNRETURNED = "censored_unreturned_serve"
    FAULT = "censored_service_fault"
    DOUBLE_FAULT = "censored_double_fault"
    SPECIAL = "special_event_before_return"
    INCOMPLETE_LET = "incomplete_let"


_CODE_REASONS: Final[Mapping[str, ReturnShotTypeReason]] = MappingProxyType({
    code: getattr(ReturnShotTypeReason, code.upper()) for code in DOCUMENTED_SHOT_TYPES
})
_REASONS: Final[Mapping[ReturnShotTypeState, frozenset[ReturnShotTypeReason]]] = MappingProxyType({
    ReturnShotTypeState.OBSERVED: frozenset(_CODE_REASONS.values()),
    ReturnShotTypeState.UNKNOWN: frozenset({ReturnShotTypeReason.Q}),
    ReturnShotTypeState.UNKNOWN_INITIAL: frozenset({
        ReturnShotTypeReason.MISSING_PREFIX,
        ReturnShotTypeReason.BOUNDARY,
        ReturnShotTypeReason.MODIFIER,
        ReturnShotTypeReason.TRUNCATED,
        ReturnShotTypeReason.INCONSISTENT,
    }),
    ReturnShotTypeState.CENSORED: frozenset({
        ReturnShotTypeReason.ACE,
        ReturnShotTypeReason.UNRETURNED,
        ReturnShotTypeReason.FAULT,
        ReturnShotTypeReason.DOUBLE_FAULT,
        ReturnShotTypeReason.SPECIAL,
        ReturnShotTypeReason.INCOMPLETE_LET,
    }),
})
_TERMINALS: Final[Mapping[ReturnShotTypeReason, str]] = MappingProxyType({
    ReturnShotTypeReason.ACE: "ace",
    ReturnShotTypeReason.UNRETURNED: "unreturned_serve",
    ReturnShotTypeReason.FAULT: "service_fault",
    ReturnShotTypeReason.DOUBLE_FAULT: "double_fault",
    ReturnShotTypeReason.SPECIAL: "special_event",
    ReturnShotTypeReason.INCOMPLETE_LET: "incomplete_let",
})


@dataclass(frozen=True)
class ReturnShotTypeClassification:
    sequence_text: str
    serve_number: int
    previous_attempt_was_fault: bool
    classification_state: ReturnShotTypeState
    reason_codes: tuple[ReturnShotTypeReason, ...]
    eligible_for_type_comparison: bool
    return_event_observed: bool
    actor: str | None
    return_shot_type_code: str | None
    documented_description: str | None
    shot_type_family: str | None
    service_prefix_span: tuple[int, int] | None
    marker_literal: str | None
    marker_span: tuple[int, int] | None
    shot_type_span: tuple[int, int] | None
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


def _input(
    sequence_text: object, serve_number: object, previous_attempt_was_fault: object
) -> None:
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


def _terminal(
    parsed: ParseResult, previous_fault: bool
) -> tuple[ReturnShotTypeReason, str] | None:
    if isinstance(parsed.structure, ServeAce):
        return ReturnShotTypeReason.ACE, "ace"
    if isinstance(parsed.structure, ServeUnreturned):
        return ReturnShotTypeReason.UNRETURNED, "unreturned_serve"
    if isinstance(parsed.structure, ServeFault):
        if parsed.serve_number == 2 and previous_fault:
            return ReturnShotTypeReason.DOUBLE_FAULT, "double_fault"
        return ReturnShotTypeReason.FAULT, "service_fault"
    if isinstance(parsed.structure, SpecialCodeStructure):
        return ReturnShotTypeReason.SPECIAL, "special_event"
    if (
        parsed.structure is None
        and parsed.raw_sequence
        and all(token.raw_text == "c" for token in parsed.tokens)
    ):
        return ReturnShotTypeReason.INCOMPLETE_LET, "incomplete_let"
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
    state: ReturnShotTypeState,
    reason: ReturnShotTypeReason,
    *,
    previous_fault: bool,
    prefix: ServicePrefix | None = None,
    marker: tuple[int, int] | None = None,
    shot_span: tuple[int, int] | None = None,
    terminal: str | None = None,
) -> ReturnShotTypeClassification:
    raw = parsed.raw_sequence or ""
    code = None if shot_span is None else raw[shot_span[0]]
    description = (
        DOCUMENTED_SHOT_TYPES.get(code)
        if code in DOCUMENTED_SHOT_TYPES
        else UNKNOWN_SHOT_TYPE_DESCRIPTION if code == UNKNOWN_SHOT_TYPE_CODE else None
    )
    warning_codes, warning_spans = _warning_fields(parsed)
    return ReturnShotTypeClassification(
        sequence_text=raw,
        serve_number=parsed.serve_number,
        previous_attempt_was_fault=previous_fault,
        classification_state=state,
        reason_codes=(reason,),
        eligible_for_type_comparison=state is ReturnShotTypeState.OBSERVED,
        return_event_observed=shot_span is not None,
        actor="returner" if shot_span is not None else None,
        return_shot_type_code=code,
        documented_description=description,
        shot_type_family=None if code is None else SHOT_TYPE_FAMILIES.get(code),
        service_prefix_span=None if prefix is None else (prefix.start, prefix.end),
        marker_literal=None if marker is None else "+",
        marker_span=marker,
        shot_type_span=shot_span,
        return_event_span=shot_span,
        parser_warning_codes=warning_codes,
        parser_warning_spans=warning_spans,
        residual_spans=tuple((span.start, span.end) for span in parsed.residual_spans),
        terminal_serve_outcome=terminal,
    )


def _classify(
    parsed: ParseResult, previous_fault: bool
) -> ReturnShotTypeClassification:
    terminal = _terminal(parsed, previous_fault)
    prefix = _prefix(parsed)
    if terminal is not None:
        return _make_result(
            parsed,
            ReturnShotTypeState.CENSORED,
            terminal[0],
            previous_fault=previous_fault,
            prefix=prefix,
            terminal=terminal[1],
        )
    if prefix is None:
        return _make_result(
            parsed,
            ReturnShotTypeState.UNKNOWN_INITIAL,
            ReturnShotTypeReason.MISSING_PREFIX,
            previous_fault=previous_fault,
        )
    raw = parsed.raw_sequence or ""
    if prefix.start != 0 or prefix.end <= 0 or prefix.end > len(raw):
        return _make_result(
            parsed,
            ReturnShotTypeState.UNKNOWN_INITIAL,
            ReturnShotTypeReason.INCONSISTENT,
            previous_fault=previous_fault,
            prefix=prefix,
        )
    cursor, marker = prefix.end, None
    if cursor < len(raw) and raw[cursor] == "+":
        marker = (cursor, cursor + 1)
        cursor += 1
    if cursor >= len(raw):
        return _make_result(
            parsed,
            ReturnShotTypeState.UNKNOWN_INITIAL,
            ReturnShotTypeReason.TRUNCATED,
            previous_fault=previous_fault,
            prefix=prefix,
            marker=marker,
        )
    code = raw[cursor]
    if code in {"+", "-", "=", ";", "^"}:
        return _make_result(
            parsed,
            ReturnShotTypeState.UNKNOWN_INITIAL,
            ReturnShotTypeReason.MODIFIER,
            previous_fault=previous_fault,
            prefix=prefix,
            marker=marker,
        )
    if code == UNKNOWN_SHOT_TYPE_CODE:
        return _make_result(
            parsed,
            ReturnShotTypeState.UNKNOWN,
            ReturnShotTypeReason.Q,
            previous_fault=previous_fault,
            prefix=prefix,
            marker=marker,
            shot_span=(cursor, cursor + 1),
        )
    if code not in DOCUMENTED_SHOT_TYPES:
        return _make_result(
            parsed,
            ReturnShotTypeState.UNKNOWN_INITIAL,
            ReturnShotTypeReason.BOUNDARY,
            previous_fault=previous_fault,
            prefix=prefix,
            marker=marker,
        )
    return _make_result(
        parsed,
        ReturnShotTypeState.OBSERVED,
        _CODE_REASONS[code],
        previous_fault=previous_fault,
        prefix=prefix,
        marker=marker,
        shot_span=(cursor, cursor + 1),
    )


def classify_initial_return_shot_type(
    sequence_text: str,
    serve_number: int,
    parsed: ParseResult,
    previous_attempt_was_fault: bool = False,
) -> ReturnShotTypeClassification:
    """Clasifica exclusivamente el tipo literal del primer resto local."""
    _input(sequence_text, serve_number, previous_attempt_was_fault)
    if not isinstance(parsed, ParseResult):
        raise TypeError("parsed debe ser ParseResult.")
    if parsed.raw_sequence != sequence_text or parsed.serve_number != serve_number:
        raise ValueError("sequence_text y serve_number deben coincidir con ParseResult.")
    validate_parsed_sequence(parsed)
    result = _classify(parsed, previous_attempt_was_fault)
    validate_return_shot_type_classification(result, parsed=parsed)
    return result


def parse_and_classify_initial_return_shot_type(
    sequence_text: str,
    serve_number: int,
    previous_attempt_was_fault: bool = False,
) -> ReturnShotTypeClassification:
    """Ruta de alto nivel: invoca el parser exactamente una vez."""
    _input(sequence_text, serve_number, previous_attempt_was_fault)
    parsed = parse_sequence(sequence_text, serve_number)
    return classify_initial_return_shot_type(
        sequence_text, serve_number, parsed, previous_attempt_was_fault
    )


def _span(name: str, value: object, length: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise TypeError(f"{name} debe ser un span (inicio, fin) de enteros reales.")
    if value[0] < 0 or value[1] <= value[0] or value[1] > length:
        raise ValueError(f"{name} fuera de rango.")
    return value


def validate_return_shot_type_classification(
    result: ReturnShotTypeClassification, *, parsed: ParseResult | None = None
) -> None:
    """Reconstruye el contrato P06 y rechaza toda mutación semántica."""
    if not isinstance(result, ReturnShotTypeClassification):
        raise TypeError("result debe ser ReturnShotTypeClassification.")
    _input(result.sequence_text, result.serve_number, result.previous_attempt_was_fault)
    if result.classification_contract_version != CLASSIFICATION_CONTRACT_VERSION:
        raise ValueError("classification_contract_version invalida.")
    if (
        not isinstance(result.classification_state, ReturnShotTypeState)
        or not isinstance(result.reason_codes, tuple)
        or len(result.reason_codes) != 1
        or not isinstance(result.reason_codes[0], ReturnShotTypeReason)
    ):
        raise TypeError("Estado o reason_codes fuera del catalogo cerrado.")
    if result.reason_codes[0] not in _REASONS[result.classification_state]:
        raise ValueError("reason_code incompatible con estado.")
    for name in (
        "service_prefix_span",
        "marker_span",
        "shot_type_span",
        "return_event_span",
    ):
        _span(name, getattr(result, name), len(result.sequence_text))
    if parsed is None:
        parsed = parse_sequence(result.sequence_text, result.serve_number)
    if (
        not isinstance(parsed, ParseResult)
        or parsed.raw_sequence != result.sequence_text
        or parsed.serve_number != result.serve_number
    ):
        raise ValueError("ParseResult no corresponde a la clasificacion.")
    validate_parsed_sequence(parsed)
    expected = _classify(parsed, result.previous_attempt_was_fault)
    if result != expected:
        raise ValueError("La clasificacion no coincide con la derivacion canonica P06.")
