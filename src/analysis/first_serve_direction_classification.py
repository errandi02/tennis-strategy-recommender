"""Contrato individual sintetico para la direccion documentada del primer saque.

El modulo consume exclusivamente texto y un ``ParseResult`` ya calculado. No
hace E/S ni reproduce el analisis agregado P02. Las direcciones 4/5/6 siguen
siendo observables cuando el parser estructura un ace, un saque no devuelto o
una falta, igual que en el contrato descriptivo publicado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final

from src.parsing.serve_sequence import (
    Diagnostic,
    ParseResult,
    ServeAce,
    ServeFault,
    ServeUnreturned,
    ServicePrefix,
    Span,
    SpecialCodeStructure,
    parse_sequence,
    validate_parsed_sequence,
)


CLASSIFICATION_CONTRACT_VERSION: Final = "1.0.0"
PATTERN_ID: Final = "P02"
PATTERN_NAME: Final = "first_serve_direction"
FIRST_SERVE_NUMBER: Final = 1

DIRECTION_DESCRIPTIONS: Final = MappingProxyType(
    {
        "4": "wide",
        "5": "body",
        "6": "down_the_t",
        "0": "unknown",
    }
)
OBSERVABLE_DIRECTION_CODES: Final = frozenset({"4", "5", "6"})
SPECIAL_UNIT_CODES: Final = frozenset({"S", "R", "P", "Q", "V"})


class FirstServeDirectionContractError(ValueError):
    """Incumplimiento del contrato individual sintetico P02."""


class FirstServeDirectionState(str, Enum):
    OBSERVED = "first_serve_direction_observed"
    UNKNOWN = "first_serve_direction_unknown"
    UNKNOWN_FIRST_SERVE = "unknown_first_serve"
    CENSORED = "ineligible_censored"


class FirstServeDirectionReason(str, Enum):
    OBSERVED = "actionable_direction"
    UNKNOWN_DIRECTION = "direction_unknown_0"
    EMPTY = "empty_sequence"
    WHITESPACE = "whitespace_only_sequence"
    UNRECOGNIZED = "unrecognized_or_no_initial_direction"
    SPECIAL = "special_unit_sequence"


@dataclass(frozen=True)
class FirstServeDirectionClassification:
    contract_version: str
    sequence_text: str
    serve_number: int
    classification_state: FirstServeDirectionState
    reason_codes: tuple[FirstServeDirectionReason, ...]
    eligible_for_direction_signal: bool
    actor: str | None
    direction_code: str | None
    direction_description: str | None
    direction_literal: str | None
    direction_span: tuple[int, int] | None
    service_prefix_literal: str | None
    service_prefix_span: tuple[int, int] | None
    service_let_count: int | None
    service_terminal_kind: str | None
    service_terminal_code: str | None
    service_terminal_span: tuple[int, int] | None
    warnings: tuple[Diagnostic, ...]
    residual_spans: tuple[Span, ...]
    residual_text: str
    source_parse: ParseResult = field(repr=False)

    def __post_init__(self) -> None:
        validate_first_serve_direction_classification(self)


def _validate_inputs(sequence_text: object, serve_number: object) -> tuple[str, int]:
    if type(sequence_text) is not str:
        raise TypeError("sequence_text debe ser str real.")
    if type(serve_number) is not int or serve_number != FIRST_SERVE_NUMBER:
        raise FirstServeDirectionContractError(
            "P02 exige serve_number como int real igual a 1."
        )
    return sequence_text, serve_number


def _validate_source_parse(
    sequence_text: str,
    serve_number: int,
    parsed: object,
) -> ParseResult:
    if type(parsed) is not ParseResult:
        raise TypeError("parsed debe ser ParseResult exacto.")
    validate_parsed_sequence(parsed)
    if parsed.raw_sequence != sequence_text or parsed.serve_number != serve_number:
        raise FirstServeDirectionContractError(
            "sequence_text, serve_number y ParseResult no corresponden al mismo parseo."
        )
    return parsed


def _prefix_from_structure(structure: object) -> ServicePrefix | None:
    if type(structure) is ServicePrefix:
        return structure
    if type(structure) in (ServeAce, ServeUnreturned, ServeFault):
        return structure.prefix
    return None


def _terminal_fields(
    structure: object,
    prefix: ServicePrefix | None,
) -> tuple[str | None, str | None, tuple[int, int] | None]:
    if prefix is None:
        return None, None, None
    if type(structure) is ServeAce:
        return "ace", structure.outcome_code, (prefix.end, structure.end)
    if type(structure) is ServeUnreturned:
        return "unreturned", structure.outcome_code, (prefix.end, structure.end)
    if type(structure) is ServeFault:
        return "fault", structure.fault_code, (prefix.end, structure.end)
    return None, None, None


def _reason_without_prefix(
    sequence_text: str,
    structure: object,
) -> tuple[FirstServeDirectionState, FirstServeDirectionReason]:
    if type(structure) is SpecialCodeStructure:
        return FirstServeDirectionState.CENSORED, FirstServeDirectionReason.SPECIAL
    if sequence_text == "":
        return FirstServeDirectionState.UNKNOWN_FIRST_SERVE, FirstServeDirectionReason.EMPTY
    if sequence_text.isspace():
        return (
            FirstServeDirectionState.UNKNOWN_FIRST_SERVE,
            FirstServeDirectionReason.WHITESPACE,
        )
    return (
        FirstServeDirectionState.UNKNOWN_FIRST_SERVE,
        FirstServeDirectionReason.UNRECOGNIZED,
    )


def _canonical_fields(sequence_text: str, parsed: ParseResult) -> dict[str, object]:
    structure = parsed.structure
    prefix = _prefix_from_structure(structure)
    terminal_kind, terminal_code, terminal_span = _terminal_fields(structure, prefix)

    if prefix is None:
        state, reason = _reason_without_prefix(sequence_text, structure)
        actor = None
        direction_code = None
        direction_description = None
        direction_literal = None
        direction_span = None
        prefix_literal = None
        prefix_span = None
        let_count = None
    else:
        direction_code = prefix.direction
        if direction_code in OBSERVABLE_DIRECTION_CODES:
            state = FirstServeDirectionState.OBSERVED
            reason = FirstServeDirectionReason.OBSERVED
            eligible = True
        elif direction_code == "0":
            state = FirstServeDirectionState.UNKNOWN
            reason = FirstServeDirectionReason.UNKNOWN_DIRECTION
            eligible = False
        else:
            raise FirstServeDirectionContractError(
                "El prefijo contiene una direccion fuera del vocabulario cerrado."
            )
        if prefix.direction_value != DIRECTION_DESCRIPTIONS[direction_code]:
            raise FirstServeDirectionContractError(
                "La descripcion del prefijo no coincide con la documentacion cerrada."
            )
        actor = "server"
        direction_description = DIRECTION_DESCRIPTIONS[direction_code]
        direction_span = (prefix.end - 1, prefix.end)
        direction_literal = sequence_text[direction_span[0] : direction_span[1]]
        prefix_span = (prefix.start, prefix.end)
        prefix_literal = sequence_text[prefix.start : prefix.end]
        let_count = len(prefix.lets)

    if prefix is None:
        eligible = False

    return {
        "classification_state": state,
        "reason_codes": (reason,),
        "eligible_for_direction_signal": eligible,
        "actor": actor,
        "direction_code": direction_code,
        "direction_description": direction_description,
        "direction_literal": direction_literal,
        "direction_span": direction_span,
        "service_prefix_literal": prefix_literal,
        "service_prefix_span": prefix_span,
        "service_let_count": let_count,
        "service_terminal_kind": terminal_kind,
        "service_terminal_code": terminal_code,
        "service_terminal_span": terminal_span,
        "warnings": parsed.warnings,
        "residual_spans": parsed.residual_spans,
        "residual_text": parsed.residual_text,
    }


def classify_first_serve_direction(
    sequence_text: str,
    serve_number: int,
    parsed: ParseResult,
) -> FirstServeDirectionClassification:
    """Clasifica un parseo ya emparejado sin volver a ejecutar el parser."""
    text, number = _validate_inputs(sequence_text, serve_number)
    source_parse = _validate_source_parse(text, number, parsed)
    return FirstServeDirectionClassification(
        contract_version=CLASSIFICATION_CONTRACT_VERSION,
        sequence_text=text,
        serve_number=number,
        source_parse=source_parse,
        **_canonical_fields(text, source_parse),
    )


def parse_and_classify_first_serve_direction(
    sequence_text: str,
    serve_number: int,
) -> FirstServeDirectionClassification:
    """Valida tipos, ejecuta una vez el parser y clasifica P02."""
    text, number = _validate_inputs(sequence_text, serve_number)
    parsed = parse_sequence(text, number)
    return classify_first_serve_direction(text, number, parsed)


def _validate_span(
    name: str,
    span: object,
    literal: object,
    sequence_text: str,
) -> None:
    if span is None:
        if literal is not None:
            raise FirstServeDirectionContractError(f"{name} ausente exige literal ausente.")
        return
    if (
        type(span) is not tuple
        or len(span) != 2
        or any(type(value) is not int for value in span)
        or not 0 <= span[0] < span[1] <= len(sequence_text)
    ):
        raise FirstServeDirectionContractError(f"{name} debe ser un span half-open valido.")
    if type(literal) is not str or sequence_text[span[0] : span[1]] != literal:
        raise FirstServeDirectionContractError(f"{name} no reconstruye su literal.")


def validate_first_serve_direction_classification(
    classification: FirstServeDirectionClassification,
    parsed: ParseResult | None = None,
) -> None:
    """Reconstruye todos los campos desde el ``ParseResult`` canonico conservado."""
    if type(classification) is not FirstServeDirectionClassification:
        raise TypeError("classification debe ser FirstServeDirectionClassification exacto.")
    text, number = _validate_inputs(
        classification.sequence_text, classification.serve_number
    )
    if classification.contract_version != CLASSIFICATION_CONTRACT_VERSION:
        raise FirstServeDirectionContractError("contract_version incorrecta.")
    source_parse = _validate_source_parse(text, number, classification.source_parse)
    if parsed is not None:
        explicit_parse = _validate_source_parse(text, number, parsed)
        if explicit_parse != source_parse:
            raise FirstServeDirectionContractError(
                "El ParseResult explicito no coincide con el conservado."
            )
    if type(classification.classification_state) is not FirstServeDirectionState:
        raise TypeError("classification_state fuera del enum cerrado.")
    if (
        type(classification.reason_codes) is not tuple
        or len(classification.reason_codes) != 1
        or type(classification.reason_codes[0]) is not FirstServeDirectionReason
    ):
        raise TypeError("reason_codes debe ser tuple canonica de un unico enum.")
    if type(classification.eligible_for_direction_signal) is not bool:
        raise TypeError("eligible_for_direction_signal debe ser bool real.")
    if classification.actor is not None and type(classification.actor) is not str:
        raise TypeError("actor debe ser str o None.")
    if classification.service_let_count is not None and (
        type(classification.service_let_count) is not int
        or classification.service_let_count < 0
    ):
        raise TypeError("service_let_count debe ser int no negativo o None.")
    if type(classification.warnings) is not tuple or any(
        type(item) is not Diagnostic for item in classification.warnings
    ):
        raise TypeError("warnings debe ser tuple inmutable de Diagnostic exactos.")
    if type(classification.residual_spans) is not tuple or any(
        type(item) is not Span for item in classification.residual_spans
    ):
        raise TypeError("residual_spans debe ser tuple inmutable de Span exactos.")
    if type(classification.residual_text) is not str:
        raise TypeError("residual_text debe ser str real.")

    _validate_span(
        "direction_span",
        classification.direction_span,
        classification.direction_literal,
        text,
    )
    _validate_span(
        "service_prefix_span",
        classification.service_prefix_span,
        classification.service_prefix_literal,
        text,
    )
    _validate_span(
        "service_terminal_span",
        classification.service_terminal_span,
        classification.service_terminal_code,
        text,
    )

    expected = _canonical_fields(text, source_parse)
    for field_name, expected_value in expected.items():
        if getattr(classification, field_name) != expected_value:
            raise FirstServeDirectionContractError(
                f"{field_name} no coincide con la clasificacion canonica."
            )


__all__ = [
    "CLASSIFICATION_CONTRACT_VERSION",
    "DIRECTION_DESCRIPTIONS",
    "FIRST_SERVE_NUMBER",
    "FirstServeDirectionClassification",
    "FirstServeDirectionContractError",
    "FirstServeDirectionReason",
    "FirstServeDirectionState",
    "OBSERVABLE_DIRECTION_CODES",
    "PATTERN_ID",
    "PATTERN_NAME",
    "SPECIAL_UNIT_CODES",
    "classify_first_serve_direction",
    "parse_and_classify_first_serve_direction",
    "validate_first_serve_direction_classification",
]
