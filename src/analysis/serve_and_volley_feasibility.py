"""Clasificacion sintética de P03: intento explícitamente anotado de saque y volea.

La señal P03 es el modificador literal ``+`` contiguo al prefijo de servicio.
Representa una etiqueta de intención anotada, no una volea realizada, un actor
del rally ni una explicación del resultado. Este módulo no lee datos ni crea
artefactos.
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
_DIRECTIONS: Final = {"4": "wide", "5": "body", "6": "T", "0": "unknown"}
_SUBSTANTIVE_DIRECTIONS: Final = frozenset({"4", "5", "6"})


class AnalysisState(str, Enum):
    POSITIVE_TAGGED = "positive_tagged"
    NOT_EXPLICITLY_TAGGED = "not_explicitly_tagged"
    UNKNOWN = "unknown"
    INELIGIBLE_CENSORED = "ineligible_censored"


class ReasonCode(str, Enum):
    EXPLICIT_MARKER = "explicit_marker_immediately_after_service"
    NO_IMMEDIATE_MARKER = "no_immediate_explicit_marker"
    UNKNOWN_DIRECTION = "unknown_service_direction"
    MISSING_PREFIX = "missing_service_prefix"
    AMBIGUOUS_POST_PREFIX = "ambiguous_post_prefix_content"
    CENSORED_ACE = "censored_ace"
    CENSORED_UNRETURNED = "censored_unreturned_serve"
    CENSORED_FAULT = "censored_service_fault"
    CENSORED_DOUBLE_FAULT = "censored_double_fault"
    CENSORED_SPECIAL = "censored_special_event"
    CENSORED_INCOMPLETE_LET = "censored_incomplete_let"


_REASONS_BY_STATE: Final = {
    AnalysisState.POSITIVE_TAGGED: frozenset({ReasonCode.EXPLICIT_MARKER}),
    AnalysisState.NOT_EXPLICITLY_TAGGED: frozenset({ReasonCode.NO_IMMEDIATE_MARKER}),
    AnalysisState.UNKNOWN: frozenset({
        ReasonCode.UNKNOWN_DIRECTION,
        ReasonCode.MISSING_PREFIX,
        ReasonCode.AMBIGUOUS_POST_PREFIX,
    }),
    AnalysisState.INELIGIBLE_CENSORED: frozenset({
        ReasonCode.CENSORED_ACE,
        ReasonCode.CENSORED_UNRETURNED,
        ReasonCode.CENSORED_FAULT,
        ReasonCode.CENSORED_DOUBLE_FAULT,
        ReasonCode.CENSORED_SPECIAL,
        ReasonCode.CENSORED_INCOMPLETE_LET,
    }),
}
_TERMINALS_BY_REASON: Final = {
    ReasonCode.CENSORED_ACE: "ace",
    ReasonCode.CENSORED_UNRETURNED: "unreturned_serve",
    ReasonCode.CENSORED_FAULT: "service_fault",
    ReasonCode.CENSORED_DOUBLE_FAULT: "double_fault",
    ReasonCode.CENSORED_SPECIAL: "special_event",
    ReasonCode.CENSORED_INCOMPLETE_LET: "incomplete_let",
}


@dataclass(frozen=True)
class ServeAndVolleyClassification:
    """Resultado auditable de un único intento de saque.

    ``explicit_intent_tagged`` y ``analysis_state`` son deliberadamente
    independientes: una futura estructura terminal podría conservar un marcador
    observado y, a la vez, resultar censurada. El parser actual no produce esa
    combinación para secuencias válidas, pero el contrato no pierde esa dimensión.
    """

    serve_number: int
    sequence_length: int
    service_direction_code: str | None
    service_direction_label: str | None
    service_prefix_start: int | None
    service_prefix_end: int | None
    explicit_intent_tagged: bool
    marker_start: int | None
    marker_end: int | None
    marker_literal: str | None
    analysis_state: AnalysisState
    reason_code: ReasonCode
    eligible_for_outcome_comparison: bool
    terminal_serve_outcome: str | None
    parser_warning_codes: tuple[str, ...]
    parser_warning_spans: tuple[tuple[int, int], ...]
    has_residual: bool
    classification_contract_version: str = CLASSIFICATION_CONTRACT_VERSION


def _validate_serve_number(serve_number: object) -> int:
    if type(serve_number) is not int or serve_number not in (1, 2):
        raise ValueError("serve_number debe ser exactamente el entero 1 o 2.")
    return serve_number


def _validate_sequence_text(sequence_text: object) -> str:
    if not isinstance(sequence_text, str):
        raise TypeError("sequence_text debe ser str.")
    return sequence_text


def _service_prefix(parsed: ParseResult) -> ServicePrefix | None:
    structure = parsed.structure
    if isinstance(structure, ServicePrefix):
        return structure
    if isinstance(structure, (ServeAce, ServeUnreturned, ServeFault)):
        return structure.prefix
    return None


def _terminal_outcome(parsed: ParseResult, *, previous_attempt_was_fault: bool) -> tuple[str | None, ReasonCode | None]:
    structure = parsed.structure
    if structure is None and parsed.raw_sequence and all(token.raw_text == "c" for token in parsed.tokens):
        return "incomplete_let", ReasonCode.CENSORED_INCOMPLETE_LET
    if isinstance(structure, ServeAce):
        return "ace", ReasonCode.CENSORED_ACE
    if isinstance(structure, ServeUnreturned):
        return "unreturned_serve", ReasonCode.CENSORED_UNRETURNED
    if isinstance(structure, ServeFault):
        if parsed.serve_number == 2 and previous_attempt_was_fault:
            return "double_fault", ReasonCode.CENSORED_DOUBLE_FAULT
        return "service_fault", ReasonCode.CENSORED_FAULT
    if isinstance(structure, SpecialCodeStructure):
        return "special_event", ReasonCode.CENSORED_SPECIAL
    return None, None


def _classification(
    *,
    parsed: ParseResult,
    prefix: ServicePrefix | None,
    marker_start: int | None,
    terminal_outcome: str | None,
    terminal_reason: ReasonCode | None,
) -> tuple[AnalysisState, ReasonCode, bool]:
    if prefix is None:
        if terminal_outcome is not None:
            assert terminal_reason is not None
            return AnalysisState.INELIGIBLE_CENSORED, terminal_reason, False
        return AnalysisState.UNKNOWN, ReasonCode.MISSING_PREFIX, False
    if terminal_outcome is not None:
        assert terminal_reason is not None
        return AnalysisState.INELIGIBLE_CENSORED, terminal_reason, False
    if prefix.direction not in _SUBSTANTIVE_DIRECTIONS:
        return AnalysisState.UNKNOWN, ReasonCode.UNKNOWN_DIRECTION, False
    if marker_start is not None:
        return AnalysisState.POSITIVE_TAGGED, ReasonCode.EXPLICIT_MARKER, True
    if prefix.end >= len(parsed.raw_sequence or ""):
        return AnalysisState.UNKNOWN, ReasonCode.AMBIGUOUS_POST_PREFIX, False
    if any(span.start == prefix.end for span in parsed.residual_spans):
        return AnalysisState.UNKNOWN, ReasonCode.AMBIGUOUS_POST_PREFIX, False
    first_post_prefix = parsed.tokens[prefix.end]
    if first_post_prefix.token_type != "documented_character":
        return AnalysisState.UNKNOWN, ReasonCode.AMBIGUOUS_POST_PREFIX, False
    return AnalysisState.NOT_EXPLICITLY_TAGGED, ReasonCode.NO_IMMEDIATE_MARKER, True


def classify_serve_and_volley_attempt(
    sequence_text: str,
    serve_number: int,
    parsed: ParseResult,
    *,
    previous_attempt_was_fault: bool = False,
) -> ServeAndVolleyClassification:
    """Clasifica un intento usando exclusivamente el resultado del parser actual.

    ``parsed`` debe proceder exactamente de ``sequence_text`` y ``serve_number``;
    el contrato se reconcilia con ``validate_parsed_sequence`` antes de observar
    su prefijo. Este es el punto de entrada de bajo nivel para una secuencia ya
    emparejada con su ``ParseResult``.

    ``previous_attempt_was_fault`` solo aporta contexto explícito para distinguir
    un fault de segundo saque como doble falta; no se infiere de resultados del
    punto ni de una búsqueda de caracteres.
    """
    serve_number = _validate_serve_number(serve_number)
    if type(previous_attempt_was_fault) is not bool:
        raise TypeError("previous_attempt_was_fault debe ser bool.")
    if previous_attempt_was_fault and serve_number != 2:
        raise ValueError("Solo un segundo saque puede seguir a un fault previo.")
    if not isinstance(parsed, ParseResult):
        raise TypeError("parsed debe ser ParseResult.")
    sequence_text = _validate_sequence_text(sequence_text)
    if parsed.serve_number != serve_number or parsed.raw_sequence != sequence_text:
        raise ValueError("sequence_text y serve_number deben coincidir con ParseResult.")
    validate_parsed_sequence(parsed)

    prefix = _service_prefix(parsed)
    immediate_marker = None
    if prefix is not None and prefix.end < len(sequence_text):
        token = parsed.tokens[prefix.end]
        if token.raw_text == "+" and token.start == prefix.end and token.end == prefix.end + 1:
            immediate_marker = token.start
    terminal_outcome, terminal_reason = _terminal_outcome(
        parsed, previous_attempt_was_fault=previous_attempt_was_fault
    )
    state, reason, eligible = _classification(
        parsed=parsed,
        prefix=prefix,
        marker_start=immediate_marker,
        terminal_outcome=terminal_outcome,
        terminal_reason=terminal_reason,
    )
    result = ServeAndVolleyClassification(
        serve_number=serve_number,
        sequence_length=len(sequence_text),
        service_direction_code=None if prefix is None else prefix.direction,
        service_direction_label=None if prefix is None else _DIRECTIONS[prefix.direction],
        service_prefix_start=None if prefix is None else prefix.start,
        service_prefix_end=None if prefix is None else prefix.end,
        explicit_intent_tagged=immediate_marker is not None,
        marker_start=immediate_marker,
        marker_end=None if immediate_marker is None else immediate_marker + 1,
        marker_literal=None if immediate_marker is None else "+",
        analysis_state=state,
        reason_code=reason,
        eligible_for_outcome_comparison=eligible,
        terminal_serve_outcome=terminal_outcome,
        parser_warning_codes=tuple(item[0] for item in sorted((warning.code, warning.start, warning.end) for warning in parsed.warnings)),
        parser_warning_spans=tuple((item[1], item[2]) for item in sorted((warning.code, warning.start, warning.end) for warning in parsed.warnings)),
        has_residual=bool(parsed.residual_spans),
    )
    validate_serve_and_volley_classification(result)
    return result


def parse_and_classify_serve_and_volley_attempt(
    sequence_text: str,
    serve_number: int,
    *,
    previous_attempt_was_fault: bool = False,
) -> ServeAndVolleyClassification:
    """Auxiliar sin E/S que reutiliza el parser productivo existente."""
    sequence_text = _validate_sequence_text(sequence_text)
    serve_number = _validate_serve_number(serve_number)
    return classify_serve_and_volley_attempt(
        sequence_text,
        serve_number,
        parse_sequence(sequence_text, serve_number),
        previous_attempt_was_fault=previous_attempt_was_fault,
    )


def validate_serve_and_volley_classification(result: ServeAndVolleyClassification) -> None:
    """Comprueba invariantes independientes del parser y del origen de datos."""
    if not isinstance(result, ServeAndVolleyClassification):
        raise TypeError("result debe ser ServeAndVolleyClassification.")
    _validate_serve_number(result.serve_number)
    if type(result.sequence_length) is not int or result.sequence_length < 0:
        raise ValueError("sequence_length invalida.")
    if type(result.explicit_intent_tagged) is not bool or type(result.eligible_for_outcome_comparison) is not bool or type(result.has_residual) is not bool:
        raise TypeError("Los campos booleanos deben ser bool reales.")
    if result.classification_contract_version != CLASSIFICATION_CONTRACT_VERSION:
        raise ValueError("classification_contract_version invalida.")
    if not isinstance(result.analysis_state, AnalysisState) or not isinstance(result.reason_code, ReasonCode):
        raise TypeError("analysis_state y reason_code deben pertenecer a sus catalogos cerrados.")
    if result.reason_code not in _REASONS_BY_STATE[result.analysis_state]:
        raise ValueError("reason_code incompatible con analysis_state.")
    if not isinstance(result.parser_warning_codes, tuple) or not isinstance(result.parser_warning_spans, tuple) or len(result.parser_warning_codes) != len(result.parser_warning_spans):
        raise ValueError("Warnings invalidos.")
    warning_items = tuple(zip(result.parser_warning_codes, result.parser_warning_spans))
    if any(not isinstance(code, str) or not code or not isinstance(span, tuple) or len(span) != 2 or type(span[0]) is not int or type(span[1]) is not int or span[0] < 0 or span[1] <= span[0] or span[1] > result.sequence_length for code, span in warning_items):
        raise ValueError("Warnings invalidos.")
    if warning_items != tuple(sorted(warning_items)) or len(set(warning_items)) != len(warning_items):
        raise ValueError("Warnings deben ser ordenados y unicos por codigo y span.")
    prefix_bounds = (result.service_prefix_start, result.service_prefix_end)
    marker_bounds = (result.marker_start, result.marker_end)
    if (prefix_bounds[0] is None) != (prefix_bounds[1] is None) or (marker_bounds[0] is None) != (marker_bounds[1] is None):
        raise ValueError("Los spans deben ser ambos nulos o ambos enteros.")
    for start, end in (prefix_bounds, marker_bounds):
        if start is not None and (type(start) is not int or type(end) is not int or start < 0 or end <= start or end > result.sequence_length):
            raise ValueError("Span invalido.")
    if result.service_direction_code is None:
        if result.service_direction_label is not None or result.service_prefix_start is not None:
            raise ValueError("Un prefijo ausente no puede tener direccion o span.")
    elif result.service_direction_code not in _DIRECTIONS or result.service_direction_label != _DIRECTIONS[result.service_direction_code]:
        raise ValueError("Direccion de servicio invalida.")
    if result.marker_start is not None and result.marker_end != result.marker_start + 1:
        raise ValueError("El marcador debe tener longitud uno.")
    if (result.marker_literal is None) != (result.marker_start is None) or result.marker_literal not in {None, "+"}:
        raise ValueError("marker_literal invalido.")
    if result.explicit_intent_tagged != (result.marker_start is not None):
        raise ValueError("Marcador y explicit_intent_tagged no reconcilian.")
    if result.explicit_intent_tagged and result.marker_start != result.service_prefix_end:
        raise ValueError("El marcador debe comenzar al final del prefijo.")
    comparable = result.analysis_state in {AnalysisState.POSITIVE_TAGGED, AnalysisState.NOT_EXPLICITLY_TAGGED}
    if result.eligible_for_outcome_comparison is not comparable:
        raise ValueError("Elegibilidad incompatible con analysis_state.")
    if comparable and result.service_direction_code not in _SUBSTANTIVE_DIRECTIONS:
        raise ValueError("Los estados comparables exigen direccion sustantiva.")
    if result.analysis_state is AnalysisState.POSITIVE_TAGGED and not result.explicit_intent_tagged:
        raise ValueError("positive_tagged exige marcador.")
    if result.analysis_state is AnalysisState.NOT_EXPLICITLY_TAGGED and result.explicit_intent_tagged:
        raise ValueError("not_explicitly_tagged exige ausencia de marcador.")
    if result.analysis_state is AnalysisState.INELIGIBLE_CENSORED and result.terminal_serve_outcome is None:
        raise ValueError("La censura requiere terminal_serve_outcome.")
    if result.analysis_state is not AnalysisState.INELIGIBLE_CENSORED and result.terminal_serve_outcome is not None:
        raise ValueError("Solo la censura puede publicar terminal_serve_outcome.")
    if result.analysis_state is AnalysisState.INELIGIBLE_CENSORED and result.terminal_serve_outcome != _TERMINALS_BY_REASON[result.reason_code]:
        raise ValueError("terminal_serve_outcome incompatible con reason_code.")
