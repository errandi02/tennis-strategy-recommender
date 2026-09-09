"""P09: perfil documental compuesto del primer resto.

Extension sintetica y aislada. Solo reconoce el evento local que comienza
inmediatamente despues de un ``ServicePrefix`` y de un marcador de saque-red
opcional. No hace E/S, no reconstruye el rally y no interpreta outcomes.
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
PATTERN_ID: Final = "documented_initial_return_profile"
PROFILE_COMPONENTS: Final[tuple[str, ...]] = (
    "return_shot_type",
    "lateral_direction",
    "return_depth",
)

DOCUMENTED_SHOT_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
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
    }
)
UNKNOWN_SHOT_TYPE_CODE: Final = "q"
UNKNOWN_SHOT_TYPE_DESCRIPTION: Final = "tipo de golpe desconocido"
SHOT_TYPE_FAMILY_MEMBERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "groundstroke": frozenset({"f", "b"}),
        "slice": frozenset({"r", "s"}),
        "volley": frozenset({"v", "z"}),
        "overhead": frozenset({"o", "p"}),
        "drop_shot": frozenset({"u", "y"}),
        "lob": frozenset({"l", "m"}),
        "half_volley": frozenset({"h", "i"}),
        "swinging_volley": frozenset({"j", "k"}),
        "special": frozenset({"t"}),
    }
)
SHOT_TYPE_FAMILIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        code: family
        for family, codes in SHOT_TYPE_FAMILY_MEMBERS.items()
        for code in codes
    }
)
LATERAL_DIRECTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "1": "right_side_of_right_handed_opponent_or_left_of_left_handed",
        "2": "centre",
        "3": "left_side_of_right_handed_opponent_or_right_of_left_handed",
        "0": "unknown",
    }
)
RETURN_DEPTHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "7": "service_boxes",
        "8": "behind_service_line_closer_to_service_line",
        "9": "closer_to_baseline",
        "0": "unknown",
    }
)
SERVICE_DIRECTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {"4": "wide", "5": "body", "6": "down_the_t", "0": "unknown"}
)
DOCUMENTED_PROFILE_IDS: Final[frozenset[str]] = frozenset(
    f"{shot}|{direction}|{depth}"
    for shot in DOCUMENTED_SHOT_TYPES
    for direction in ("1", "2", "3")
    for depth in ("7", "8", "9")
)


class ReturnProfileState(str, Enum):
    OBSERVED = "documented_initial_return_profile"
    UNKNOWN = "initial_return_profile_unknown"
    NOT_DOCUMENTED = "initial_return_profile_not_documented"
    UNKNOWN_INITIAL = "unknown_initial_return"
    CENSORED = "ineligible_censored"


class ReturnProfileReason(str, Enum):
    OBSERVED = "documented_type_direction_depth_profile"
    UNKNOWN_TYPE = "documented_unknown_shot_type_q"
    UNKNOWN_DIRECTION = "documented_unknown_lateral_direction_0"
    UNKNOWN_DEPTH = "documented_unknown_return_depth_0"
    DIRECTION_NOT_DOCUMENTED = "lateral_direction_not_documented"
    DEPTH_NOT_DOCUMENTED = "return_depth_not_documented"
    MISSING_PREFIX = "missing_or_ambiguous_service_prefix"
    TRUNCATED = "truncated_before_initial_return_type"
    INVALID_TYPE = "invalid_initial_return_type"
    MODIFIER = "unsupported_modifier_before_initial_return_type"
    AMBIGUOUS_COMPONENT = "ambiguous_initial_return_component_order"
    ACE = "censored_ace"
    UNRETURNED = "censored_unreturned_serve"
    FAULT = "censored_service_fault"
    DOUBLE_FAULT = "censored_double_fault"
    SPECIAL = "censored_special_event"
    INCOMPLETE_LET = "censored_incomplete_let"


_UNKNOWN_REASONS: Final = frozenset(
    {
        ReturnProfileReason.UNKNOWN_TYPE,
        ReturnProfileReason.UNKNOWN_DIRECTION,
        ReturnProfileReason.UNKNOWN_DEPTH,
    }
)
_MISSING_REASONS: Final = frozenset(
    {
        ReturnProfileReason.DIRECTION_NOT_DOCUMENTED,
        ReturnProfileReason.DEPTH_NOT_DOCUMENTED,
    }
)
_UNKNOWN_INITIAL_REASONS: Final = frozenset(
    {
        ReturnProfileReason.MISSING_PREFIX,
        ReturnProfileReason.TRUNCATED,
        ReturnProfileReason.INVALID_TYPE,
        ReturnProfileReason.MODIFIER,
        ReturnProfileReason.AMBIGUOUS_COMPONENT,
    }
)
_CENSORED_REASONS: Final = frozenset(
    {
        ReturnProfileReason.ACE,
        ReturnProfileReason.UNRETURNED,
        ReturnProfileReason.FAULT,
        ReturnProfileReason.DOUBLE_FAULT,
        ReturnProfileReason.SPECIAL,
        ReturnProfileReason.INCOMPLETE_LET,
    }
)
_TERMINALS: Final[Mapping[str, tuple[ReturnProfileReason, str]]] = MappingProxyType(
    {
        "ace": (ReturnProfileReason.ACE, "ace"),
        "unreturned": (ReturnProfileReason.UNRETURNED, "unreturned_serve"),
        "fault": (ReturnProfileReason.FAULT, "service_fault"),
        "double_fault": (ReturnProfileReason.DOUBLE_FAULT, "double_fault"),
        "special": (ReturnProfileReason.SPECIAL, "special_event"),
        "incomplete_let": (ReturnProfileReason.INCOMPLETE_LET, "incomplete_let"),
    }
)
_POST_RETURN_BOUNDARY_CODES: Final = frozenset("*#@nwdxe!+-=;^C")


@dataclass(frozen=True)
class RecordedWarning:
    code: str
    message: str
    literal: str
    span: tuple[int, int]


@dataclass(frozen=True)
class RecordedResidual:
    literal: str
    span: tuple[int, int]


@dataclass(frozen=True)
class ReturnProfileClassification:
    contract_version: str
    pattern_id: str
    sequence_text: str
    serve_number: int
    previous_attempt_was_fault: bool
    classification_state: ReturnProfileState
    reason_codes: tuple[ReturnProfileReason, ...]
    eligible_for_profile_comparison: bool
    return_event_localized: bool
    profile_complete: bool
    shot_type_observed: bool
    lateral_direction_observed: bool
    return_depth_observed: bool
    unknown_components: tuple[str, ...]
    not_documented_components: tuple[str, ...]
    actor: str | None
    profile_id: str | None
    service_prefix_literal: str | None
    service_prefix_span: tuple[int, int] | None
    service_let_count: int | None
    service_direction_code: str | None
    service_direction_description: str | None
    service_approach_marker_literal: str | None
    service_approach_marker_span: tuple[int, int] | None
    return_shot_type_code: str | None
    return_shot_type_description: str | None
    return_shot_type_family: str | None
    return_shot_type_span: tuple[int, int] | None
    lateral_direction_code: str | None
    lateral_direction_description: str | None
    lateral_direction_span: tuple[int, int] | None
    return_depth_code: str | None
    return_depth_description: str | None
    return_depth_span: tuple[int, int] | None
    return_event_span: tuple[int, int] | None
    terminal_serve_outcome: str | None
    parser_warnings: tuple[RecordedWarning, ...]
    parser_residuals: tuple[RecordedResidual, ...]
    trailing_residuals: tuple[RecordedResidual, ...]


def _validate_inputs(
    sequence_text: object,
    serve_number: object,
    previous_attempt_was_fault: object,
) -> None:
    if type(sequence_text) is not str:
        raise TypeError("sequence_text debe ser str real.")
    if type(serve_number) is not int or serve_number not in (1, 2):
        raise TypeError("serve_number debe ser int real 1 o 2.")
    if type(previous_attempt_was_fault) is not bool:
        raise TypeError("previous_attempt_was_fault debe ser bool real.")
    if previous_attempt_was_fault and serve_number != 2:
        raise ValueError("Solo un segundo saque puede seguir a un fault previo.")


def _prefix(parsed: ParseResult) -> ServicePrefix | None:
    structure = parsed.structure
    if isinstance(structure, ServicePrefix):
        return structure
    if isinstance(structure, (ServeAce, ServeUnreturned, ServeFault)):
        return structure.prefix
    return None


def _terminal(
    parsed: ParseResult, previous_attempt_was_fault: bool
) -> tuple[ReturnProfileReason, str] | None:
    structure = parsed.structure
    if isinstance(structure, ServeAce):
        return _TERMINALS["ace"]
    if isinstance(structure, ServeUnreturned):
        return _TERMINALS["unreturned"]
    if isinstance(structure, ServeFault):
        if parsed.serve_number == 2 and previous_attempt_was_fault:
            return _TERMINALS["double_fault"]
        return _TERMINALS["fault"]
    if isinstance(structure, SpecialCodeStructure):
        return _TERMINALS["special"]
    if (
        structure is None
        and parsed.raw_sequence
        and all(token.raw_text == "c" for token in parsed.tokens)
    ):
        return _TERMINALS["incomplete_let"]
    return None


def _diagnostics(
    parsed: ParseResult,
) -> tuple[tuple[RecordedWarning, ...], tuple[RecordedResidual, ...]]:
    raw = parsed.raw_sequence or ""
    warnings = tuple(
        RecordedWarning(
            item.code,
            item.message,
            raw[item.start:item.end],
            (item.start, item.end),
        )
        for item in sorted(
            parsed.warnings,
            key=lambda item: (item.start, item.end, item.code, item.message),
        )
    )
    residuals = tuple(
        RecordedResidual(raw[span.start:span.end], (span.start, span.end))
        for span in parsed.residual_spans
    )
    return warnings, residuals


def _reason_codes(
    unknown_components: tuple[str, ...],
    missing_components: tuple[str, ...],
) -> tuple[ReturnProfileReason, ...]:
    unknown_lookup = {
        "return_shot_type": ReturnProfileReason.UNKNOWN_TYPE,
        "lateral_direction": ReturnProfileReason.UNKNOWN_DIRECTION,
        "return_depth": ReturnProfileReason.UNKNOWN_DEPTH,
    }
    missing_lookup = {
        "lateral_direction": ReturnProfileReason.DIRECTION_NOT_DOCUMENTED,
        "return_depth": ReturnProfileReason.DEPTH_NOT_DOCUMENTED,
    }
    return tuple(
        [unknown_lookup[item] for item in PROFILE_COMPONENTS if item in unknown_components]
        + [missing_lookup[item] for item in PROFILE_COMPONENTS if item in missing_components]
    )


def _make_result(
    parsed: ParseResult,
    previous_attempt_was_fault: bool,
    state: ReturnProfileState,
    reasons: tuple[ReturnProfileReason, ...],
    *,
    prefix: ServicePrefix | None = None,
    marker_span: tuple[int, int] | None = None,
    shot_span: tuple[int, int] | None = None,
    lateral_span: tuple[int, int] | None = None,
    depth_span: tuple[int, int] | None = None,
    unknown_components: tuple[str, ...] = (),
    missing_components: tuple[str, ...] = (),
    terminal: str | None = None,
) -> ReturnProfileClassification:
    raw = parsed.raw_sequence or ""
    localized = shot_span is not None
    shot_code = raw[shot_span[0]] if shot_span is not None else None
    lateral_code = raw[lateral_span[0]] if lateral_span is not None else None
    depth_code = raw[depth_span[0]] if depth_span is not None else None
    shot_observed = shot_code in DOCUMENTED_SHOT_TYPES
    lateral_observed = lateral_code in {"1", "2", "3"}
    depth_observed = depth_code in {"7", "8", "9"}
    complete = shot_observed and lateral_observed and depth_observed
    eligible = state is ReturnProfileState.OBSERVED
    profile_id = f"{shot_code}|{lateral_code}|{depth_code}" if eligible else None
    event_span = None
    if localized:
        event_span = (shot_span[0], (depth_span or lateral_span or shot_span)[1])
    warnings, parser_residuals = _diagnostics(parsed)
    trailing_residuals = ()
    if event_span is not None and event_span[1] < len(raw):
        trailing_residuals = (
            RecordedResidual(raw[event_span[1]:], (event_span[1], len(raw))),
        )
    return ReturnProfileClassification(
        contract_version=CLASSIFICATION_CONTRACT_VERSION,
        pattern_id=PATTERN_ID,
        sequence_text=raw,
        serve_number=parsed.serve_number,
        previous_attempt_was_fault=previous_attempt_was_fault,
        classification_state=state,
        reason_codes=reasons,
        eligible_for_profile_comparison=eligible,
        return_event_localized=localized,
        profile_complete=complete,
        shot_type_observed=shot_observed,
        lateral_direction_observed=lateral_observed,
        return_depth_observed=depth_observed,
        unknown_components=unknown_components,
        not_documented_components=missing_components,
        actor="returner" if localized else None,
        profile_id=profile_id,
        service_prefix_literal=None if prefix is None else raw[prefix.start:prefix.end],
        service_prefix_span=None if prefix is None else (prefix.start, prefix.end),
        service_let_count=None if prefix is None else len(prefix.lets),
        service_direction_code=None if prefix is None else prefix.direction,
        service_direction_description=(
            None if prefix is None else SERVICE_DIRECTIONS[prefix.direction]
        ),
        service_approach_marker_literal=None if marker_span is None else "+",
        service_approach_marker_span=marker_span,
        return_shot_type_code=shot_code,
        return_shot_type_description=(
            DOCUMENTED_SHOT_TYPES.get(shot_code)
            if shot_code != UNKNOWN_SHOT_TYPE_CODE
            else UNKNOWN_SHOT_TYPE_DESCRIPTION
        ),
        return_shot_type_family=SHOT_TYPE_FAMILIES.get(shot_code),
        return_shot_type_span=shot_span,
        lateral_direction_code=lateral_code,
        lateral_direction_description=LATERAL_DIRECTIONS.get(lateral_code),
        lateral_direction_span=lateral_span,
        return_depth_code=depth_code,
        return_depth_description=RETURN_DEPTHS.get(depth_code),
        return_depth_span=depth_span,
        return_event_span=event_span,
        terminal_serve_outcome=terminal,
        parser_warnings=warnings,
        parser_residuals=parser_residuals,
        trailing_residuals=trailing_residuals,
    )


def _unknown_initial(
    parsed: ParseResult,
    previous_attempt_was_fault: bool,
    reason: ReturnProfileReason,
    *,
    prefix: ServicePrefix | None = None,
    marker_span: tuple[int, int] | None = None,
) -> ReturnProfileClassification:
    return _make_result(
        parsed,
        previous_attempt_was_fault,
        ReturnProfileState.UNKNOWN_INITIAL,
        (reason,),
        prefix=prefix,
        marker_span=marker_span,
    )


def _canonical(
    parsed: ParseResult, previous_attempt_was_fault: bool
) -> ReturnProfileClassification:
    raw = parsed.raw_sequence or ""
    prefix = _prefix(parsed)
    terminal = _terminal(parsed, previous_attempt_was_fault)
    if terminal is not None:
        return _make_result(
            parsed,
            previous_attempt_was_fault,
            ReturnProfileState.CENSORED,
            (terminal[0],),
            prefix=prefix,
            terminal=terminal[1],
        )
    if prefix is None or prefix.start != 0 or prefix.end > len(raw):
        return _unknown_initial(
            parsed, previous_attempt_was_fault, ReturnProfileReason.MISSING_PREFIX
        )

    cursor = prefix.end
    marker_span = None
    if cursor < len(raw) and raw[cursor] == "+":
        marker_span = (cursor, cursor + 1)
        cursor += 1
    if cursor >= len(raw):
        return _unknown_initial(
            parsed,
            previous_attempt_was_fault,
            ReturnProfileReason.TRUNCATED,
            prefix=prefix,
            marker_span=marker_span,
        )
    if raw[cursor] in {"+", "-", "=", ";", "^"}:
        return _unknown_initial(
            parsed,
            previous_attempt_was_fault,
            ReturnProfileReason.MODIFIER,
            prefix=prefix,
            marker_span=marker_span,
        )
    if raw[cursor] not in DOCUMENTED_SHOT_TYPES and raw[cursor] != UNKNOWN_SHOT_TYPE_CODE:
        return _unknown_initial(
            parsed,
            previous_attempt_was_fault,
            ReturnProfileReason.INVALID_TYPE,
            prefix=prefix,
            marker_span=marker_span,
        )

    shot_span = (cursor, cursor + 1)
    unknown = ("return_shot_type",) if raw[cursor] == UNKNOWN_SHOT_TYPE_CODE else ()
    cursor += 1
    lateral_span = None
    depth_span = None
    missing: tuple[str, ...]

    if cursor >= len(raw):
        missing = ("lateral_direction", "return_depth")
    elif raw[cursor] in LATERAL_DIRECTIONS:
        lateral_span = (cursor, cursor + 1)
        if raw[cursor] == "0":
            unknown += ("lateral_direction",)
        cursor += 1
        if cursor < len(raw) and raw[cursor] in RETURN_DEPTHS:
            depth_span = (cursor, cursor + 1)
            if raw[cursor] == "0":
                unknown += ("return_depth",)
            missing = ()
        elif cursor >= len(raw) or raw[cursor] in _POST_RETURN_BOUNDARY_CODES:
            missing = ("return_depth",)
        else:
            return _unknown_initial(
                parsed,
                previous_attempt_was_fault,
                ReturnProfileReason.AMBIGUOUS_COMPONENT,
                prefix=prefix,
                marker_span=marker_span,
            )
    elif raw[cursor] in _POST_RETURN_BOUNDARY_CODES:
        missing = ("lateral_direction", "return_depth")
    else:
        return _unknown_initial(
            parsed,
            previous_attempt_was_fault,
            ReturnProfileReason.AMBIGUOUS_COMPONENT,
            prefix=prefix,
            marker_span=marker_span,
        )

    reasons = _reason_codes(unknown, missing)
    if unknown:
        state = ReturnProfileState.UNKNOWN
    elif missing:
        state = ReturnProfileState.NOT_DOCUMENTED
    else:
        state = ReturnProfileState.OBSERVED
        reasons = (ReturnProfileReason.OBSERVED,)
    return _make_result(
        parsed,
        previous_attempt_was_fault,
        state,
        reasons,
        prefix=prefix,
        marker_span=marker_span,
        shot_span=shot_span,
        lateral_span=lateral_span,
        depth_span=depth_span,
        unknown_components=unknown,
        missing_components=missing,
    )


def _validate_recorded_span(
    name: str,
    value: object,
    raw: str,
    expected_literal: str | None = None,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise TypeError(f"{name} debe ser un span de enteros reales.")
    start, end = value
    if start < 0 or end <= start or end > len(raw):
        raise ValueError(f"{name} fuera de rango.")
    if expected_literal is not None and raw[start:end] != expected_literal:
        raise ValueError(f"{name} no coincide con su literal.")
    return value


def _validate_result_structure(result: ReturnProfileClassification) -> None:
    raw = result.sequence_text
    prefix = _validate_recorded_span(
        "service_prefix_span", result.service_prefix_span, raw, result.service_prefix_literal
    )
    marker = _validate_recorded_span(
        "service_approach_marker_span",
        result.service_approach_marker_span,
        raw,
        result.service_approach_marker_literal,
    )
    shot = _validate_recorded_span(
        "return_shot_type_span", result.return_shot_type_span, raw, result.return_shot_type_code
    )
    lateral = _validate_recorded_span(
        "lateral_direction_span", result.lateral_direction_span, raw, result.lateral_direction_code
    )
    depth = _validate_recorded_span(
        "return_depth_span", result.return_depth_span, raw, result.return_depth_code
    )
    event = _validate_recorded_span("return_event_span", result.return_event_span, raw)
    ordered = [span for span in (prefix, marker, shot, lateral, depth) if span is not None]
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Los spans de componentes se solapan o estan desordenados.")
    for span in (marker, shot, lateral, depth):
        if span is not None and span[1] - span[0] != 1:
            raise ValueError("Los codigos y el marcador deben medir un caracter.")
    components = [span for span in (shot, lateral, depth) if span is not None]
    expected_event = None if not components else (components[0][0], components[-1][1])
    if event != expected_event:
        raise ValueError("return_event_span no cubre exactamente sus componentes.")
    for collection_name in ("parser_warnings", "parser_residuals", "trailing_residuals"):
        collection = getattr(result, collection_name)
        if not isinstance(collection, tuple):
            raise TypeError(f"{collection_name} debe ser tuple inmutable.")
        for item in collection:
            span = _validate_recorded_span(
                f"{collection_name}.span", item.span, raw, item.literal
            )
            if span is None:
                raise ValueError(f"{collection_name} no admite span nulo.")


def _validate_state_contract(result: ReturnProfileClassification) -> None:
    if not isinstance(result.classification_state, ReturnProfileState):
        raise TypeError("classification_state fuera del catalogo cerrado.")
    if (
        not isinstance(result.reason_codes, tuple)
        or not result.reason_codes
        or any(not isinstance(reason, ReturnProfileReason) for reason in result.reason_codes)
    ):
        raise TypeError("reason_codes fuera del catalogo cerrado.")
    for value in (
        result.eligible_for_profile_comparison,
        result.return_event_localized,
        result.profile_complete,
        result.shot_type_observed,
        result.lateral_direction_observed,
        result.return_depth_observed,
    ):
        if type(value) is not bool:
            raise TypeError("Los flags contractuales deben ser bool reales.")
    if not isinstance(result.unknown_components, tuple) or not isinstance(
        result.not_documented_components, tuple
    ):
        raise TypeError("Las listas de componentes deben ser tuples inmutables.")
    if tuple(item for item in PROFILE_COMPONENTS if item in result.unknown_components) != result.unknown_components:
        raise ValueError("unknown_components debe respetar orden y unicidad.")
    if tuple(item for item in PROFILE_COMPONENTS if item in result.not_documented_components) != result.not_documented_components:
        raise ValueError("not_documented_components debe respetar orden y unicidad.")
    state = result.classification_state
    reasons = set(result.reason_codes)
    if state is ReturnProfileState.OBSERVED:
        if result.reason_codes != (ReturnProfileReason.OBSERVED,):
            raise ValueError("Perfil completo con reason_codes incompatibles.")
    elif state is ReturnProfileState.UNKNOWN:
        if not reasons & _UNKNOWN_REASONS or not reasons <= _UNKNOWN_REASONS | _MISSING_REASONS:
            raise ValueError("Perfil unknown sin componente documental unknown.")
    elif state is ReturnProfileState.NOT_DOCUMENTED:
        if not reasons or not reasons <= _MISSING_REASONS:
            raise ValueError("Perfil no documentado con reason_codes incompatibles.")
    elif state is ReturnProfileState.UNKNOWN_INITIAL:
        if len(reasons) != 1 or not reasons <= _UNKNOWN_INITIAL_REASONS:
            raise ValueError("Unknown initial return con reason_code incompatible.")
    elif len(reasons) != 1 or not reasons <= _CENSORED_REASONS:
        raise ValueError("Censura con reason_code incompatible.")
    if result.eligible_for_profile_comparison != (state is ReturnProfileState.OBSERVED):
        raise ValueError("Elegibilidad incompatible con el estado.")
    if result.profile_complete != (
        result.shot_type_observed
        and result.lateral_direction_observed
        and result.return_depth_observed
    ):
        raise ValueError("Flags de completitud no reconcilian.")
    if result.profile_complete != (state is ReturnProfileState.OBSERVED):
        raise ValueError("Solo el perfil completo puede estar observado.")
    if result.profile_id is not None:
        expected = (
            f"{result.return_shot_type_code}|{result.lateral_direction_code}|"
            f"{result.return_depth_code}"
        )
        if result.profile_id != expected or result.profile_id not in DOCUMENTED_PROFILE_IDS:
            raise ValueError("profile_id no corresponde a sus tres componentes.")
    elif state is ReturnProfileState.OBSERVED:
        raise ValueError("El perfil completo requiere profile_id.")
    if state in {ReturnProfileState.UNKNOWN_INITIAL, ReturnProfileState.CENSORED}:
        if result.actor is not None or result.return_event_localized:
            raise ValueError("Un evento no localizado o censurado no publica actor.")
    elif result.actor != "returner" or not result.return_event_localized:
        raise ValueError("El evento local debe atribuirse al returner.")


def classify_initial_return_profile(
    sequence_text: str,
    serve_number: int,
    parsed: ParseResult,
    previous_attempt_was_fault: bool = False,
) -> ReturnProfileClassification:
    """Clasifica P09 reutilizando un ``ParseResult`` ya validado."""
    _validate_inputs(sequence_text, serve_number, previous_attempt_was_fault)
    if not isinstance(parsed, ParseResult):
        raise TypeError("parsed debe ser ParseResult.")
    if parsed.raw_sequence != sequence_text or parsed.serve_number != serve_number:
        raise ValueError("ParseResult no corresponde al texto o numero de saque.")
    validate_parsed_sequence(parsed)
    result = _canonical(parsed, previous_attempt_was_fault)
    validate_return_profile_classification(result, parsed=parsed)
    return result


def parse_and_classify_initial_return_profile(
    sequence_text: str,
    serve_number: int,
    previous_attempt_was_fault: bool = False,
) -> ReturnProfileClassification:
    """Valida entradas y realiza exactamente una llamada al parser."""
    _validate_inputs(sequence_text, serve_number, previous_attempt_was_fault)
    parsed = parse_sequence(sequence_text, serve_number)
    return classify_initial_return_profile(
        sequence_text, serve_number, parsed, previous_attempt_was_fault
    )


def validate_return_profile_classification(
    result: ReturnProfileClassification,
    *,
    parsed: ParseResult | None = None,
) -> None:
    """Reconstruye el contrato completo P09 y rechaza mutaciones."""
    if not isinstance(result, ReturnProfileClassification):
        raise TypeError("result debe ser ReturnProfileClassification.")
    _validate_inputs(
        result.sequence_text,
        result.serve_number,
        result.previous_attempt_was_fault,
    )
    if result.contract_version != CLASSIFICATION_CONTRACT_VERSION:
        raise ValueError("contract_version invalida.")
    if result.pattern_id != PATTERN_ID:
        raise ValueError("pattern_id invalido.")
    _validate_state_contract(result)
    _validate_result_structure(result)
    if parsed is None:
        parsed = parse_sequence(result.sequence_text, result.serve_number)
    if (
        not isinstance(parsed, ParseResult)
        or parsed.raw_sequence != result.sequence_text
        or parsed.serve_number != result.serve_number
    ):
        raise ValueError("ParseResult no corresponde a la clasificacion P09.")
    validate_parsed_sequence(parsed)
    expected = _canonical(parsed, result.previous_attempt_was_fault)
    if result != expected:
        raise ValueError("La clasificacion no coincide con la derivacion canonica P09.")
