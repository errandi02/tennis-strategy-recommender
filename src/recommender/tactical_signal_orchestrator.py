"""Orquestacion sintetica de un intento hacia senales P02/P04/P05/P06/P09.

El modulo valida una request minima, ejecuta el parser exactamente una vez y
reutiliza el mismo ``ParseResult`` en todos los clasificadores aplicables. La
salida publica omite deliberadamente texto, parseo, spans y datos individuales.
No realiza E/S, scoring, recomendaciones ni consultas a artefactos.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Callable, Final, TypeVar

from src.analysis.first_serve_direction_classification import (
    classify_first_serve_direction,
)
from src.analysis.return_depth_feasibility import classify_initial_return_depth
from src.analysis.return_direction_feasibility import (
    classify_initial_return_direction,
)
from src.analysis.return_profile_feasibility import classify_initial_return_profile
from src.analysis.return_shot_type_feasibility import (
    classify_initial_return_shot_type,
)
from src.parsing.serve_sequence import parse_sequence
from src.recommender.tactical_signal_contract import (
    ELIGIBLE_PATTERN_ORDER,
    SignalAdaptationResult,
    SignalContext,
    SignalProvenance,
    TacticalSignalBundle,
    TacticalSignalContractError,
    adapt_first_serve_direction_signal,
    adapt_return_depth_signal,
    adapt_return_direction_signal,
    adapt_return_profile_signal,
    adapt_return_shot_type_signal,
    build_tactical_signal_bundle,
    bundle_to_json_structure,
    signal_to_json_structure,
    validate_signal_adaptation_result,
    validate_tactical_signal_bundle,
)


ORCHESTRATOR_CONTRACT_VERSION: Final = "1.0.0"
REQUESTED_PATTERN_ORDER: Final = ELIGIBLE_PATTERN_ORDER
RETURN_PATTERN_ORDER: Final = ("P04", "P05", "P06", "P09")
SECOND_SERVE_NON_APPLICABLE_REASON: Final = (
    "pattern_not_applicable_to_second_serve"
)
ATTEMPT_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-attempt-extraction\x00"

_DIAGNOSTIC_ORDER: Final = (
    "requested_pattern_count",
    "applicable_pattern_count",
    "non_applicable_pattern_count",
    "active_signal_count",
    "abstention_count",
)
_FORBIDDEN_OUTPUT_KEYS: Final = frozenset(
    {
        "sequence_text",
        "raw_sequence",
        "parse_result",
        "tokens",
        "spans",
        "warnings",
        "residuals",
        "residual_text",
        "terminal_literal",
        "match_id",
        "point_number",
        "player",
        "players",
        "server",
        "point_winner",
        "returner_won_point",
        "date",
        "timestamp",
        "path",
    }
)
_FORBIDDEN_OUTPUT_TERMS: Final = (
    "evaluation",
    "scoring",
    "recommendation",
)
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_T = TypeVar("_T")


class AttemptExtractionState(str, Enum):
    SIGNALS_AVAILABLE = "signals_available"
    ALL_APPLICABLE_PATTERNS_ABSTAINED = "all_applicable_patterns_abstained"


class TacticalSignalOrchestrationError(ValueError):
    """Error cerrado y sin contenido de la secuencia original."""


@dataclass(frozen=True)
class NonApplicablePattern:
    pattern_id: str
    reason_code: str

    def __post_init__(self) -> None:
        _validate_non_applicable_pattern(self)


@dataclass(frozen=True)
class AttemptSignalRequest:
    contract_version: str
    sequence_text: str
    serve_number: int
    previous_attempt_was_fault: bool

    def __post_init__(self) -> None:
        _validate_request(self)


@dataclass(frozen=True)
class AttemptSignalExtraction:
    contract_version: str
    serve_number: int
    previous_attempt_was_fault: bool
    requested_patterns: tuple[str, ...]
    applicable_patterns: tuple[str, ...]
    non_applicable_patterns: tuple[NonApplicablePattern, ...]
    adaptations: tuple[SignalAdaptationResult, ...]
    bundle: TacticalSignalBundle
    active_signal_count: int
    abstention_count: int
    extraction_state: AttemptExtractionState
    reason_codes: tuple[str, ...]
    diagnostics: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        validate_attempt_signal_extraction(self)


def _validate_request(request: AttemptSignalRequest) -> None:
    if type(request) is not AttemptSignalRequest:
        raise TypeError("request debe ser AttemptSignalRequest exacta.")
    if request.contract_version != ORCHESTRATOR_CONTRACT_VERSION:
        raise TacticalSignalOrchestrationError("contract_version de request invalida.")
    if type(request.sequence_text) is not str:
        raise TypeError("sequence_text debe ser str real.")
    if type(request.serve_number) is not int or request.serve_number not in (1, 2):
        raise TypeError("serve_number debe ser int real 1 o 2.")
    if type(request.previous_attempt_was_fault) is not bool:
        raise TypeError("previous_attempt_was_fault debe ser bool real.")
    if request.serve_number == 1 and request.previous_attempt_was_fault:
        raise TacticalSignalOrchestrationError(
            "Un primer saque no puede declarar una falta previa."
        )


def _validate_non_applicable_pattern(item: NonApplicablePattern) -> None:
    if type(item) is not NonApplicablePattern:
        raise TypeError("La no aplicabilidad exige NonApplicablePattern exacto.")
    if item.pattern_id != "P02":
        raise TacticalSignalOrchestrationError(
            "Solo P02 puede ser no aplicable en este contrato."
        )
    if item.reason_code != SECOND_SERVE_NON_APPLICABLE_REASON:
        raise TacticalSignalOrchestrationError(
            "Reason code de no aplicabilidad incorrecto."
        )


def _run_contract_step(stage: str, operation: Callable[[], _T]) -> _T:
    """Sanitiza solo errores contractuales esperados, nunca excepciones genericas."""
    try:
        return operation()
    except (TypeError, ValueError, TacticalSignalContractError) as error:
        error_type = type(error).__name__
        raise TacticalSignalOrchestrationError(
            f"Fallo contractual sanitizado en {stage} ({error_type})."
        ) from None


def _expected_applicability(
    serve_number: int,
) -> tuple[tuple[str, ...], tuple[NonApplicablePattern, ...]]:
    if serve_number == 1:
        return REQUESTED_PATTERN_ORDER, ()
    return RETURN_PATTERN_ORDER, (
        NonApplicablePattern("P02", SECOND_SERVE_NON_APPLICABLE_REASON),
    )


def _expected_state_and_reasons(
    *, serve_number: int, active_signal_count: int
) -> tuple[AttemptExtractionState, tuple[str, ...]]:
    if active_signal_count:
        state = AttemptExtractionState.SIGNALS_AVAILABLE
    else:
        state = AttemptExtractionState.ALL_APPLICABLE_PATTERNS_ABSTAINED
    reasons = [state.value]
    if serve_number == 2:
        reasons.append(SECOND_SERVE_NON_APPLICABLE_REASON)
    return state, tuple(reasons)


def _expected_diagnostics(
    *, applicable_count: int, active_count: int, abstention_count: int
) -> tuple[tuple[str, int], ...]:
    return (
        ("requested_pattern_count", len(REQUESTED_PATTERN_ORDER)),
        ("applicable_pattern_count", applicable_count),
        (
            "non_applicable_pattern_count",
            len(REQUESTED_PATTERN_ORDER) - applicable_count,
        ),
        ("active_signal_count", active_count),
        ("abstention_count", abstention_count),
    )


def _validate_diagnostics(
    diagnostics: object, *, expected: tuple[tuple[str, int], ...]
) -> None:
    if type(diagnostics) is not tuple:
        raise TacticalSignalOrchestrationError("diagnostics debe ser tuple inmutable.")
    if diagnostics != expected:
        raise TacticalSignalOrchestrationError(
            "diagnostics no reconcilia con los conteos canonicos."
        )
    for item in diagnostics:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] < 0
        ):
            raise TacticalSignalOrchestrationError(
                "diagnostics solo admite pares inmutables de conteos."
            )
    if tuple(key for key, _ in diagnostics) != _DIAGNOSTIC_ORDER:
        raise TacticalSignalOrchestrationError(
            "diagnostics no usa el orden contractual."
        )


def extract_tactical_signals_for_attempt(
    request: AttemptSignalRequest,
) -> AttemptSignalExtraction:
    """Extrae todas las senales aplicables reutilizando un unico parseo."""
    _validate_request(request)
    parsed = _run_contract_step(
        "parser", lambda: parse_sequence(request.sequence_text, request.serve_number)
    )
    adaptations: list[SignalAdaptationResult] = []

    if request.serve_number == 1:
        p02 = _run_contract_step(
            "P02.classifier",
            lambda: classify_first_serve_direction(
                request.sequence_text, request.serve_number, parsed
            ),
        )
        adaptations.append(
            _run_contract_step(
                "P02.adapter", lambda: adapt_first_serve_direction_signal(p02)
            )
        )

    p04 = _run_contract_step(
        "P04.classifier",
        lambda: classify_initial_return_direction(
            request.sequence_text,
            request.serve_number,
            parsed,
            previous_attempt_was_fault=request.previous_attempt_was_fault,
        ),
    )
    adaptations.append(
        _run_contract_step(
            "P04.adapter",
            lambda: adapt_return_direction_signal(p04, parsed=parsed),
        )
    )

    p05 = _run_contract_step(
        "P05.classifier",
        lambda: classify_initial_return_depth(
            request.sequence_text,
            request.serve_number,
            parsed,
            previous_attempt_was_fault=request.previous_attempt_was_fault,
        ),
    )
    adaptations.append(
        _run_contract_step(
            "P05.adapter", lambda: adapt_return_depth_signal(p05, parsed=parsed)
        )
    )

    p06 = _run_contract_step(
        "P06.classifier",
        lambda: classify_initial_return_shot_type(
            request.sequence_text,
            request.serve_number,
            parsed,
            previous_attempt_was_fault=request.previous_attempt_was_fault,
        ),
    )
    adaptations.append(
        _run_contract_step(
            "P06.adapter",
            lambda: adapt_return_shot_type_signal(p06, parsed=parsed),
        )
    )

    p09 = _run_contract_step(
        "P09.classifier",
        lambda: classify_initial_return_profile(
            request.sequence_text,
            request.serve_number,
            parsed,
            previous_attempt_was_fault=request.previous_attempt_was_fault,
        ),
    )
    adaptations.append(
        _run_contract_step(
            "P09.adapter", lambda: adapt_return_profile_signal(p09, parsed=parsed)
        )
    )

    adaptation_tuple = tuple(adaptations)
    bundle = _run_contract_step(
        "bundle", lambda: build_tactical_signal_bundle(adaptation_tuple)
    )
    applicable, non_applicable = _expected_applicability(request.serve_number)
    active_count = sum(result.signal is not None for result in adaptation_tuple)
    abstention_count = len(adaptation_tuple) - active_count
    state, reasons = _expected_state_and_reasons(
        serve_number=request.serve_number, active_signal_count=active_count
    )
    return AttemptSignalExtraction(
        contract_version=ORCHESTRATOR_CONTRACT_VERSION,
        serve_number=request.serve_number,
        previous_attempt_was_fault=request.previous_attempt_was_fault,
        requested_patterns=REQUESTED_PATTERN_ORDER,
        applicable_patterns=applicable,
        non_applicable_patterns=non_applicable,
        adaptations=adaptation_tuple,
        bundle=bundle,
        active_signal_count=active_count,
        abstention_count=abstention_count,
        extraction_state=state,
        reason_codes=reasons,
        diagnostics=_expected_diagnostics(
            applicable_count=len(applicable),
            active_count=active_count,
            abstention_count=abstention_count,
        ),
    )


def validate_attempt_signal_extraction(extraction: AttemptSignalExtraction) -> None:
    """Valida exhaustivamente la salida segura sin fingir reconstruir el parseo."""
    if type(extraction) is not AttemptSignalExtraction:
        raise TypeError("extraction debe ser AttemptSignalExtraction exacta.")
    if extraction.contract_version != ORCHESTRATOR_CONTRACT_VERSION:
        raise TacticalSignalOrchestrationError("contract_version de salida invalida.")
    if type(extraction.serve_number) is not int or extraction.serve_number not in (1, 2):
        raise TacticalSignalOrchestrationError("serve_number de salida invalido.")
    if type(extraction.previous_attempt_was_fault) is not bool:
        raise TacticalSignalOrchestrationError(
            "previous_attempt_was_fault de salida debe ser bool real."
        )
    if extraction.serve_number == 1 and extraction.previous_attempt_was_fault:
        raise TacticalSignalOrchestrationError(
            "El contexto de falta previa contradice el primer saque."
        )
    if extraction.requested_patterns != REQUESTED_PATTERN_ORDER:
        raise TacticalSignalOrchestrationError(
            "Los patrones solicitados no usan el catalogo y orden cerrados."
        )
    applicable, non_applicable = _expected_applicability(extraction.serve_number)
    if extraction.applicable_patterns != applicable:
        raise TacticalSignalOrchestrationError("Aplicabilidad incompatible con el saque.")
    if extraction.non_applicable_patterns != non_applicable:
        raise TacticalSignalOrchestrationError(
            "No aplicabilidad incompatible con el saque."
        )
    if type(extraction.adaptations) is not tuple:
        raise TacticalSignalOrchestrationError("adaptations debe ser tuple inmutable.")
    for result in extraction.adaptations:
        if type(result) is not SignalAdaptationResult:
            raise TypeError(
                "Cada adaptacion debe ser SignalAdaptationResult exacto."
            )
        validate_signal_adaptation_result(result)
    if tuple(item.requested_pattern_id for item in extraction.adaptations) != applicable:
        raise TacticalSignalOrchestrationError(
            "Adaptaciones ausentes, adicionales o fuera de orden."
        )
    if len({item.requested_pattern_id for item in extraction.adaptations}) != len(
        extraction.adaptations
    ):
        raise TacticalSignalOrchestrationError("No se admiten adaptaciones duplicadas.")
    for result in extraction.adaptations:
        if result.context.serve_number != extraction.serve_number:
            raise TacticalSignalOrchestrationError(
                "Una adaptacion pertenece a otro intento de saque."
            )
    fault_reasons = {
        code
        for result in extraction.adaptations
        for code in result.reason_codes
        if code in {"censored_service_fault", "censored_double_fault"}
    }
    if fault_reasons:
        expected_fault_reason = (
            "censored_double_fault"
            if extraction.serve_number == 2
            and extraction.previous_attempt_was_fault
            else "censored_service_fault"
        )
        if fault_reasons != {expected_fault_reason}:
            raise TacticalSignalOrchestrationError(
                "El contexto de falta previa no reconcilia con las abstenciones."
            )
    validate_tactical_signal_bundle(extraction.bundle)
    expected_signals = tuple(
        result.signal for result in extraction.adaptations if result.signal is not None
    )
    expected_abstentions = tuple(
        result for result in extraction.adaptations if result.signal is None
    )
    if extraction.bundle.signals != expected_signals:
        raise TacticalSignalOrchestrationError(
            "Las senales del bundle no coinciden con las adaptaciones."
        )
    if extraction.bundle.abstentions != expected_abstentions:
        raise TacticalSignalOrchestrationError(
            "Las abstenciones del bundle no coinciden con las adaptaciones."
        )
    if extraction.bundle.context.serve_number != extraction.serve_number:
        raise TacticalSignalOrchestrationError(
            "El contexto del bundle pertenece a otro intento."
        )
    if extraction.serve_number == 2 and any(
        pattern_id == "P02"
        for pattern_id in (
            *(signal.pattern_id for signal in extraction.bundle.signals),
            *(item.requested_pattern_id for item in extraction.bundle.abstentions),
        )
    ):
        raise TacticalSignalOrchestrationError(
            "P02 no puede aparecer en un bundle de segundo saque."
        )
    active_count = len(expected_signals)
    abstention_count = len(expected_abstentions)
    if type(extraction.active_signal_count) is not int or (
        extraction.active_signal_count != active_count
    ):
        raise TacticalSignalOrchestrationError("active_signal_count no reconcilia.")
    if type(extraction.abstention_count) is not int or (
        extraction.abstention_count != abstention_count
    ):
        raise TacticalSignalOrchestrationError("abstention_count no reconcilia.")
    if active_count + abstention_count != len(applicable):
        raise TacticalSignalOrchestrationError(
            "Senales y abstenciones no agotan los patrones aplicables."
        )
    expected_state, expected_reasons = _expected_state_and_reasons(
        serve_number=extraction.serve_number, active_signal_count=active_count
    )
    if type(extraction.extraction_state) is not AttemptExtractionState or (
        extraction.extraction_state is not expected_state
    ):
        raise TacticalSignalOrchestrationError("Estado global no reconciliado.")
    if type(extraction.reason_codes) is not tuple or (
        extraction.reason_codes != expected_reasons
    ):
        raise TacticalSignalOrchestrationError(
            "Reason codes globales ausentes, adicionales o fuera de orden."
        )
    if len(set(extraction.reason_codes)) != len(extraction.reason_codes):
        raise TacticalSignalOrchestrationError(
            "Reason codes globales no admiten duplicados."
        )
    _validate_diagnostics(
        extraction.diagnostics,
        expected=_expected_diagnostics(
            applicable_count=len(applicable),
            active_count=active_count,
            abstention_count=abstention_count,
        ),
    )


def _context_structure(context: SignalContext) -> dict[str, object]:
    return {
        "return_depth": context.return_depth,
        "return_lateral_direction": context.return_lateral_direction,
        "return_profile": context.return_profile,
        "return_shot_type": context.return_shot_type,
        "serve_direction": context.serve_direction,
        "serve_number": context.serve_number,
    }


def _provenance_structure(provenance: SignalProvenance) -> dict[str, object]:
    return {
        "adapter_version": provenance.adapter_version,
        "registry_readiness": provenance.registry_readiness,
        "synthetic_origin": provenance.synthetic_origin,
        "upstream_contract_version": provenance.upstream_contract_version,
        "upstream_pattern": provenance.upstream_pattern,
        "upstream_publication_fingerprint": provenance.upstream_publication_fingerprint,
    }


def _adaptation_structure(result: SignalAdaptationResult) -> dict[str, object]:
    validate_signal_adaptation_result(result)
    return {
        "abstained": result.abstained,
        "abstention_condition": result.abstention_condition,
        "context": _context_structure(result.context),
        "contract_version": result.contract_version,
        "diagnostics": {key: value for key, value in result.diagnostics},
        "provenance": _provenance_structure(result.provenance),
        "reason_codes": list(result.reason_codes),
        "requested_pattern_id": result.requested_pattern_id,
        "signal": None
        if result.signal is None
        else signal_to_json_structure(result.signal),
        "upstream_state": result.upstream_state,
    }


def _public_structure(extraction: AttemptSignalExtraction) -> dict[str, object]:
    validate_attempt_signal_extraction(extraction)
    structure: dict[str, object] = {
        "abstention_count": extraction.abstention_count,
        "active_signal_count": extraction.active_signal_count,
        "adaptations": [_adaptation_structure(item) for item in extraction.adaptations],
        "applicable_patterns": list(extraction.applicable_patterns),
        "bundle": bundle_to_json_structure(extraction.bundle),
        "contract_version": extraction.contract_version,
        "diagnostics": {key: value for key, value in extraction.diagnostics},
        "extraction_state": extraction.extraction_state.value,
        "non_applicable_patterns": [
            {"pattern_id": item.pattern_id, "reason_code": item.reason_code}
            for item in extraction.non_applicable_patterns
        ],
        "previous_attempt_was_fault": extraction.previous_attempt_was_fault,
        "reason_codes": list(extraction.reason_codes),
        "requested_patterns": list(extraction.requested_patterns),
        "serve_number": extraction.serve_number,
    }
    _validate_public_tree(structure)
    return structure


def _validate_public_tree(value: object, *, key: str | None = None) -> None:
    """Revisa recursivamente la estructura generada antes de serializarla."""
    if key is not None:
        lowered = key.lower()
        if (
            lowered in _FORBIDDEN_OUTPUT_KEYS
            or lowered.startswith("test_")
            or any(term in lowered for term in _FORBIDDEN_OUTPUT_TERMS)
        ):
            raise TacticalSignalOrchestrationError(
                "La salida contiene una clave no autorizada."
            )
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        lowered_value = value.lower()
        if (
            "file://" in lowered_value
            or _WINDOWS_PATH.match(value)
            or value.startswith(("/", "\\\\"))
            or "../" in value
            or "..\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise TacticalSignalOrchestrationError(
                "La salida contiene texto operativo no autorizado."
            )
        return
    if type(value) is list:
        for item in value:
            _validate_public_tree(item)
        return
    if type(value) is dict:
        for child_key, child_value in value.items():
            if type(child_key) is not str:
                raise TacticalSignalOrchestrationError(
                    "La estructura JSON exige claves textuales."
                )
            _validate_public_tree(child_value, key=child_key)
        return
    raise TacticalSignalOrchestrationError(
        "La salida contiene un objeto no contractual."
    )


def canonical_attempt_extraction_json(extraction: AttemptSignalExtraction) -> bytes:
    """Serializa toda la salida publica a JSON canonico UTF-8 en memoria."""
    return json.dumps(
        _public_structure(extraction),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def attempt_extraction_fingerprint(extraction: AttemptSignalExtraction) -> str:
    """Calcula SHA-256 con dominio distinto de senales y bundles comunes."""
    return sha256(
        ATTEMPT_FINGERPRINT_DOMAIN + canonical_attempt_extraction_json(extraction)
    ).hexdigest().upper()


__all__ = [
    "ATTEMPT_FINGERPRINT_DOMAIN",
    "ORCHESTRATOR_CONTRACT_VERSION",
    "REQUESTED_PATTERN_ORDER",
    "RETURN_PATTERN_ORDER",
    "SECOND_SERVE_NON_APPLICABLE_REASON",
    "AttemptExtractionState",
    "AttemptSignalExtraction",
    "AttemptSignalRequest",
    "NonApplicablePattern",
    "TacticalSignalOrchestrationError",
    "attempt_extraction_fingerprint",
    "canonical_attempt_extraction_json",
    "extract_tactical_signals_for_attempt",
    "validate_attempt_signal_extraction",
]
