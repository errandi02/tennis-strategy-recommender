"""Contrato sintetico comun para senales tacticas P02/P04/P05/P06/P09.

Este modulo no hace E/S, no ejecuta parsers ni extractores y no calcula
outcomes, scoring o recomendaciones. Los adaptadores consumen resultados
upstream ya calculados y, cuando el validador upstream necesita el parseo
original, exigen el mismo ``ParseResult`` para evitar una segunda ejecucion.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Final

from src.analysis.first_serve_direction_classification import (
    CLASSIFICATION_CONTRACT_VERSION as P02_CONTRACT_VERSION,
    FirstServeDirectionClassification,
    FirstServeDirectionState,
    validate_first_serve_direction_classification,
)
from src.analysis.return_depth_feasibility import (
    CLASSIFICATION_CONTRACT_VERSION as P05_CONTRACT_VERSION,
    ReturnDepthClassification,
    ReturnDepthState,
    validate_return_depth_classification,
)
from src.analysis.return_direction_feasibility import (
    CLASSIFICATION_CONTRACT_VERSION as P04_CONTRACT_VERSION,
    ReturnDirectionClassification,
    ReturnDirectionState,
    validate_return_direction_classification,
)
from src.analysis.return_profile_feasibility import (
    CLASSIFICATION_CONTRACT_VERSION as P09_CONTRACT_VERSION,
    DOCUMENTED_PROFILE_IDS,
    ReturnProfileClassification,
    ReturnProfileState,
    validate_return_profile_classification,
)
from src.analysis.return_shot_type_feasibility import (
    CLASSIFICATION_CONTRACT_VERSION as P06_CONTRACT_VERSION,
    DOCUMENTED_SHOT_TYPES,
    ReturnShotTypeClassification,
    ReturnShotTypeState,
    validate_return_shot_type_classification,
)
from src.parsing.serve_sequence import (
    ParseResult,
)


SIGNAL_CONTRACT_VERSION: Final = "1.0.0"
ADAPTER_VERSION: Final = "1.0.0"
ELIGIBLE_PATTERN_ORDER: Final = ("P02", "P04", "P05", "P06", "P09")
ELIGIBLE_PATTERN_IDS: Final = frozenset(ELIGIBLE_PATTERN_ORDER)
ADAPTABLE_PATTERN_ORDER: Final = ELIGIBLE_PATTERN_ORDER
ADAPTABLE_PATTERN_IDS: Final = frozenset(ADAPTABLE_PATTERN_ORDER)
REGISTRY_READINESS: Final = "eligible_component"
P02_AGGREGATE_PUBLICATION_FINGERPRINT: Final = (
    "A5820C693AE572654185257350D207019E2630BC3275F5A938F370E11F4A111A"
)

_PATTERN_METADATA: Final = MappingProxyType(
    {
        key: MappingProxyType(value)
        for key, value in {
        "P02": {
            "canonical_name": "first_serve_direction",
            "unit": "point",
            "actor": "server",
            "upstream_pattern": "first_serve_direction_classification",
            "observed_state": FirstServeDirectionState.OBSERVED.value,
            "upstream_version": P02_CONTRACT_VERSION,
            "publication_fingerprint": P02_AGGREGATE_PUBLICATION_FINGERPRINT,
        },
        "P04": {
            "canonical_name": "documented_initial_return_lateral_direction",
            "unit": "initial_return_event",
            "actor": "returner",
            "upstream_pattern": "documented_lateral_direction_of_initial_return",
            "observed_state": ReturnDirectionState.OBSERVED.value,
            "upstream_version": P04_CONTRACT_VERSION,
        },
        "P05": {
            "canonical_name": "documented_initial_return_depth",
            "unit": "initial_return_event",
            "actor": "returner",
            "upstream_pattern": "documented_initial_return_depth",
            "observed_state": ReturnDepthState.OBSERVED.value,
            "upstream_version": P05_CONTRACT_VERSION,
        },
        "P06": {
            "canonical_name": "documented_initial_return_shot_type",
            "unit": "initial_return_event",
            "actor": "returner",
            "upstream_pattern": "documented_initial_return_shot_type",
            "observed_state": ReturnShotTypeState.OBSERVED.value,
            "upstream_version": P06_CONTRACT_VERSION,
        },
        "P09": {
            "canonical_name": "documented_initial_return_profile",
            "unit": "initial_return_event",
            "actor": "returner_profile",
            "upstream_pattern": "documented_initial_return_profile",
            "observed_state": ReturnProfileState.OBSERVED.value,
            "upstream_version": P09_CONTRACT_VERSION,
        },
        }.items()
    }
)
_SAFE_DIAGNOSTIC_KEYS: Final = frozenset({"component_count"})
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:|-]+$")
_SEMANTIC_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_DATE_TEXT = re.compile(r"(?:^|\D)\d{4}-\d{2}-\d{2}(?:\D|$)")
_ACTIVE_REASON_BY_PATTERN: Final = MappingProxyType(
    {
        key: MappingProxyType(value)
        for key, value in {
        "P02": {
            "wide": "actionable_direction",
            "body": "actionable_direction",
            "T": "actionable_direction",
        },
        "P04": {"1": "observed_lateral_direction_1", "2": "observed_lateral_direction_2", "3": "observed_lateral_direction_3"},
        "P05": {"7": "documented_depth_7", "8": "documented_depth_8", "9": "documented_depth_9"},
        "P06": {code: f"documented_return_shot_type_{code}" for code in DOCUMENTED_SHOT_TYPES},
        "P09": {profile: "documented_type_direction_depth_profile" for profile in DOCUMENTED_PROFILE_IDS},
        }.items()
    }
)
_ABSTENTION_CONTRACTS: Final = MappingProxyType(
    {
        key: MappingProxyType(value)
        for key, value in {
        "P02": {
            "first_serve_direction_unknown": (
                "unknown",
                frozenset({"direction_unknown_0"}),
            ),
            "unknown_first_serve": (
                "unknown",
                frozenset(
                    {
                        "empty_sequence",
                        "whitespace_only_sequence",
                        "unrecognized_or_no_initial_direction",
                    }
                ),
            ),
            "ineligible_censored": (
                "censored",
                frozenset({"special_unit_sequence"}),
            ),
        },
        "P04": {
            "return_direction_unknown": (
                "unknown",
                frozenset({"documented_unknown_direction_0", "localized_return_without_direction"}),
            ),
            "unknown_initial_return": (
                "unknown",
                frozenset({"missing_service_prefix", "ambiguous_post_service_boundary", "unsupported_initial_modifier", "unknown_return_shot_type", "truncated_return_event"}),
            ),
            "ace": ("censored", frozenset({"censored_ace"})),
            "unreturned_serve": ("censored", frozenset({"censored_unreturned_serve"})),
            "service_fault": ("censored", frozenset({"censored_service_fault", "censored_double_fault"})),
            "special_or_incomplete": ("censored", frozenset({"special_event_before_return", "incomplete_let", "challenge_or_penalty_before_return"})),
        },
        "P05": {
            "return_depth_unknown": ("unknown", frozenset({"documented_unknown_depth_0"})),
            "return_depth_not_documented": ("not_documented", frozenset({"no_documented_depth_after_lateral_direction"})),
            "unknown_initial_return": ("unknown", frozenset({"missing_service_prefix", "ambiguous_post_service_boundary", "unsupported_initial_modifier", "unknown_return_shot_type", "truncated_return_event"})),
            "ineligible_censored": ("censored", frozenset({"censored_ace", "censored_unreturned_serve", "censored_service_fault", "censored_double_fault", "special_event_before_return", "incomplete_let"})),
        },
        "P06": {
            "return_shot_type_unknown": ("unknown", frozenset({"documented_unknown_return_shot_type_q"})),
            "unknown_initial_return": ("unknown", frozenset({"missing_service_prefix", "ambiguous_post_service_boundary", "unsupported_initial_modifier", "truncated_return_event"})),
            "ineligible_censored": ("censored", frozenset({"censored_ace", "censored_unreturned_serve", "censored_service_fault", "censored_double_fault", "special_event_before_return", "incomplete_let"})),
        },
        "P09": {
            "initial_return_profile_unknown": ("unknown", frozenset({"documented_unknown_shot_type_q", "documented_unknown_lateral_direction_0", "documented_unknown_return_depth_0", "lateral_direction_not_documented", "return_depth_not_documented"})),
            "initial_return_profile_not_documented": ("not_documented", frozenset({"lateral_direction_not_documented", "return_depth_not_documented"})),
            "unknown_initial_return": ("unknown", frozenset({"missing_or_ambiguous_service_prefix", "truncated_before_initial_return_type", "invalid_initial_return_type", "unsupported_modifier_before_initial_return_type", "ambiguous_initial_return_component_order"})),
            "ineligible_censored": ("censored", frozenset({"censored_ace", "censored_unreturned_serve", "censored_service_fault", "censored_double_fault", "censored_special_event", "censored_incomplete_let"})),
        },
        }.items()
    }
)
_P09_ABSTENTION_REASON_TUPLES: Final = MappingProxyType(
    {
        ReturnProfileState.UNKNOWN.value: frozenset(
            {
                ("documented_unknown_shot_type_q",),
                (
                    "documented_unknown_shot_type_q",
                    "documented_unknown_lateral_direction_0",
                ),
                (
                    "documented_unknown_shot_type_q",
                    "return_depth_not_documented",
                ),
                (
                    "documented_unknown_shot_type_q",
                    "documented_unknown_lateral_direction_0",
                    "documented_unknown_return_depth_0",
                ),
                (
                    "documented_unknown_shot_type_q",
                    "documented_unknown_lateral_direction_0",
                    "return_depth_not_documented",
                ),
                (
                    "documented_unknown_shot_type_q",
                    "lateral_direction_not_documented",
                    "return_depth_not_documented",
                ),
                (
                    "documented_unknown_shot_type_q",
                    "documented_unknown_return_depth_0",
                ),
                ("documented_unknown_lateral_direction_0",),
                (
                    "documented_unknown_lateral_direction_0",
                    "documented_unknown_return_depth_0",
                ),
                (
                    "documented_unknown_lateral_direction_0",
                    "return_depth_not_documented",
                ),
                ("documented_unknown_return_depth_0",),
            }
        ),
        ReturnProfileState.NOT_DOCUMENTED.value: frozenset(
            {
                ("return_depth_not_documented",),
                (
                    "lateral_direction_not_documented",
                    "return_depth_not_documented",
                ),
            }
        ),
    }
)


class TacticalSignalContractError(ValueError):
    """Error cerrado del contrato de senales tacticas."""


@dataclass(frozen=True)
class SignalContext:
    serve_number: int | None = None
    serve_direction: str | None = None
    return_shot_type: str | None = None
    return_lateral_direction: str | None = None
    return_depth: str | None = None
    return_profile: str | None = None

    def __post_init__(self) -> None:
        _validate_signal_context(self)


@dataclass(frozen=True)
class SignalProvenance:
    upstream_pattern: str
    adapter_version: str
    upstream_contract_version: str
    registry_readiness: str
    upstream_publication_fingerprint: str | None
    synthetic_origin: bool

    def __post_init__(self) -> None:
        _validate_provenance(self)


@dataclass(frozen=True)
class TacticalSignal:
    contract_version: str
    pattern_id: str
    canonical_name: str
    unit: str
    actor: str
    tactical_value: str
    components: tuple[tuple[str, str], ...]
    upstream_state: str
    eligible: bool
    comparable: bool
    outcome_available: bool
    registry_readiness: str
    reason_codes: tuple[str, ...]
    context: SignalContext
    provenance: SignalProvenance
    abstention_condition: None
    diagnostics: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        validate_tactical_signal(self)


@dataclass(frozen=True)
class SignalAdaptationResult:
    contract_version: str
    requested_pattern_id: str
    signal: TacticalSignal | None
    abstained: bool
    upstream_state: str
    reason_codes: tuple[str, ...]
    abstention_condition: str | None
    context: SignalContext
    diagnostics: tuple[tuple[str, int], ...]
    provenance: SignalProvenance

    def __post_init__(self) -> None:
        validate_signal_adaptation_result(self)


@dataclass(frozen=True)
class TacticalSignalBundle:
    contract_version: str
    signals: tuple[TacticalSignal, ...]
    abstentions: tuple[SignalAdaptationResult, ...]
    context: SignalContext

    def __post_init__(self) -> None:
        validate_tactical_signal_bundle(self)


def _strict_text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise TacticalSignalContractError(f"{name} debe ser str no vacio y sin espacios externos.")
    if not _SAFE_TOKEN.fullmatch(value):
        raise TacticalSignalContractError(f"{name} contiene texto no contractual.")
    if _WINDOWS_ABSOLUTE.match(value) or value.startswith("/") or _DATE_TEXT.search(value):
        raise TacticalSignalContractError(f"{name} contiene contenido individual o una ruta.")
    return value


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TacticalSignalContractError(f"{name} debe ser bool real.")
    return value


def _strict_tuple(name: str, value: object) -> tuple:
    if type(value) is not tuple:
        raise TacticalSignalContractError(f"{name} debe ser tuple inmutable.")
    return value


def _validate_signal_context(context: SignalContext) -> None:
    if type(context) is not SignalContext:
        raise TacticalSignalContractError("context debe ser SignalContext exacto.")
    if context.serve_number is not None and (
        type(context.serve_number) is not int or context.serve_number not in (1, 2)
    ):
        raise TacticalSignalContractError("serve_number debe ser int real 1/2 o None.")
    domains = {
        "serve_direction": frozenset({"wide", "body", "T", "unknown"}),
        "return_shot_type": frozenset(DOCUMENTED_SHOT_TYPES),
        "return_lateral_direction": frozenset({"1", "2", "3"}),
        "return_depth": frozenset({"7", "8", "9"}),
        "return_profile": DOCUMENTED_PROFILE_IDS,
    }
    for name, domain in domains.items():
        value = getattr(context, name)
        if value is not None and (type(value) is not str or value not in domain):
            raise TacticalSignalContractError(f"{name} fuera del dominio cerrado.")
    if context.return_profile is not None:
        shot, direction, depth = context.return_profile.split("|")
        expected = (shot, direction, depth)
        actual = (
            context.return_shot_type,
            context.return_lateral_direction,
            context.return_depth,
        )
        if any(value is not None for value in actual) and any(
            value is not None and value != expected[index]
            for index, value in enumerate(actual)
        ):
            raise TacticalSignalContractError("El perfil contradice sus componentes de contexto.")


def _validate_provenance(provenance: SignalProvenance) -> None:
    if type(provenance) is not SignalProvenance:
        raise TacticalSignalContractError("provenance debe ser SignalProvenance exacto.")
    for name in (
        "upstream_pattern",
        "adapter_version",
        "upstream_contract_version",
        "registry_readiness",
    ):
        _strict_text(name, getattr(provenance, name))
    if provenance.adapter_version != ADAPTER_VERSION:
        raise TacticalSignalContractError("adapter_version no coincide con el contrato.")
    allowed_patterns = frozenset(
        metadata["upstream_pattern"] for metadata in _PATTERN_METADATA.values()
    )
    if provenance.upstream_pattern not in allowed_patterns:
        raise TacticalSignalContractError("upstream_pattern no pertenece al catalogo cerrado.")
    if _SEMANTIC_VERSION.fullmatch(provenance.upstream_contract_version) is None:
        raise TacticalSignalContractError("upstream_contract_version debe ser semver estable.")
    if provenance.registry_readiness != REGISTRY_READINESS:
        raise TacticalSignalContractError("Solo se admite readiness eligible_component.")
    _strict_bool("synthetic_origin", provenance.synthetic_origin)
    if provenance.synthetic_origin is not True:
        raise TacticalSignalContractError("La procedencia debe declarar origen sintetico.")
    fingerprint = provenance.upstream_publication_fingerprint
    if fingerprint is not None and (
        type(fingerprint) is not str
        or re.fullmatch(r"[0-9A-F]{64}", fingerprint) is None
    ):
        raise TacticalSignalContractError("Fingerprint upstream invalido.")


def _validate_reason_codes(reason_codes: object) -> tuple[str, ...]:
    codes = _strict_tuple("reason_codes", reason_codes)
    if not codes:
        raise TacticalSignalContractError("reason_codes no puede estar vacio.")
    for code in codes:
        _strict_text("reason_code", code)
    if len(set(codes)) != len(codes):
        raise TacticalSignalContractError("reason_codes no admite duplicados.")
    return codes


def _validate_diagnostics(diagnostics: object, *, component_count: int) -> None:
    items = _strict_tuple("diagnostics", diagnostics)
    expected_keys = ("component_count",)
    if tuple(item[0] for item in items if type(item) is tuple and len(item) == 2) != expected_keys:
        raise TacticalSignalContractError("diagnostics debe usar esquema y orden canonicos.")
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise TacticalSignalContractError("Cada diagnostico debe ser un par inmutable.")
        key, value = item
        if type(key) is not str or key not in _SAFE_DIAGNOSTIC_KEYS:
            raise TacticalSignalContractError("Clave diagnostica no autorizada.")
        if type(value) is not int or value < 0:
            raise TacticalSignalContractError("Los diagnosticos solo admiten conteos enteros.")
    if items[0][1] != component_count:
        raise TacticalSignalContractError("component_count no reconcilia.")


def _expected_components(pattern_id: str, context: SignalContext) -> tuple[tuple[str, str], ...]:
    if pattern_id == "P02":
        return (("serve_direction", context.serve_direction),)  # type: ignore[arg-type]
    if pattern_id == "P04":
        return (("return_lateral_direction", context.return_lateral_direction),)  # type: ignore[arg-type]
    if pattern_id == "P05":
        return (("return_depth", context.return_depth),)  # type: ignore[arg-type]
    if pattern_id == "P06":
        return (("return_shot_type", context.return_shot_type),)  # type: ignore[arg-type]
    return (
        ("return_shot_type", context.return_shot_type),  # type: ignore[arg-type]
        ("return_lateral_direction", context.return_lateral_direction),  # type: ignore[arg-type]
        ("return_depth", context.return_depth),  # type: ignore[arg-type]
    )


def validate_tactical_signal(signal: TacticalSignal) -> None:
    """Valida reconstructivamente una senal observable y todos sus derivados."""
    if type(signal) is not TacticalSignal:
        raise TypeError("signal debe ser TacticalSignal exacto.")
    if signal.contract_version != SIGNAL_CONTRACT_VERSION:
        raise TacticalSignalContractError("contract_version invalida.")
    if type(signal.pattern_id) is not str or signal.pattern_id not in ADAPTABLE_PATTERN_IDS:
        raise TacticalSignalContractError("pattern_id no dispone de adaptador individual seguro.")
    metadata = _PATTERN_METADATA[signal.pattern_id]
    for field_name in ("canonical_name", "unit", "actor"):
        if getattr(signal, field_name) != metadata[field_name]:
            raise TacticalSignalContractError(f"{field_name} no coincide con el registro cerrado.")
    _validate_signal_context(signal.context)
    _validate_provenance(signal.provenance)
    if signal.provenance.upstream_pattern != metadata["upstream_pattern"]:
        raise TacticalSignalContractError("Patron de procedencia incorrecto.")
    if signal.provenance.upstream_contract_version != metadata["upstream_version"]:
        raise TacticalSignalContractError("Version upstream incorrecta.")
    expected_publication = metadata.get("publication_fingerprint")
    if (
        expected_publication is not None
        and signal.provenance.upstream_publication_fingerprint != expected_publication
    ):
        raise TacticalSignalContractError("Fingerprint agregado upstream incorrecto.")
    if signal.upstream_state != metadata["observed_state"]:
        raise TacticalSignalContractError("Una senal activa exige estado observado.")
    for name in ("eligible", "comparable", "outcome_available"):
        if getattr(signal, name) is not True:
            raise TacticalSignalContractError(f"{name} debe ser True en una senal activa.")
    if signal.registry_readiness != REGISTRY_READINESS:
        raise TacticalSignalContractError("Readiness no elegible.")
    if signal.abstention_condition is not None:
        raise TacticalSignalContractError("Una senal activa no puede tener abstencion.")
    _validate_reason_codes(signal.reason_codes)
    expected_components = _expected_components(signal.pattern_id, signal.context)
    if type(signal.components) is not tuple or signal.components != expected_components:
        raise TacticalSignalContractError("Los componentes no coinciden con el contexto.")
    if any(type(pair) is not tuple or len(pair) != 2 for pair in signal.components):
        raise TacticalSignalContractError("components debe contener pares inmutables.")
    if any(value is None for _, value in signal.components):
        raise TacticalSignalContractError("Una senal no puede contener componentes ausentes.")
    expected_value = (
        signal.context.return_profile
        if signal.pattern_id == "P09"
        else signal.components[0][1]
    )
    if type(signal.tactical_value) is not str or signal.tactical_value != expected_value:
        raise TacticalSignalContractError("tactical_value no reconcilia con componentes.")
    expected_reason = _ACTIVE_REASON_BY_PATTERN[signal.pattern_id].get(signal.tactical_value)
    if signal.reason_codes != (expected_reason,):
        raise TacticalSignalContractError("reason_codes no reconcilia con la categoria activa.")
    if signal.pattern_id == "P02" and signal.context.serve_number != 1:
        raise TacticalSignalContractError("P02 exige el primer intento de saque.")
    if signal.pattern_id != "P02" and signal.context.serve_number not in (1, 2):
        raise TacticalSignalContractError("El evento de resto exige numero de saque.")
    _validate_diagnostics(signal.diagnostics, component_count=len(signal.components))


def validate_signal_adaptation_result(result: SignalAdaptationResult) -> None:
    """Reconcilia presencia/abstencion sin aceptar estados intermedios."""
    if type(result) is not SignalAdaptationResult:
        raise TypeError("result debe ser SignalAdaptationResult exacto.")
    if result.contract_version != SIGNAL_CONTRACT_VERSION:
        raise TacticalSignalContractError("contract_version invalida.")
    if type(result.requested_pattern_id) is not str or result.requested_pattern_id not in ADAPTABLE_PATTERN_IDS:
        raise TacticalSignalContractError("Patron solicitado sin adaptador individual seguro.")
    _strict_bool("abstained", result.abstained)
    _strict_text("upstream_state", result.upstream_state)
    _validate_reason_codes(result.reason_codes)
    _validate_signal_context(result.context)
    _validate_provenance(result.provenance)
    metadata = _PATTERN_METADATA[result.requested_pattern_id]
    if (
        result.provenance.upstream_pattern != metadata["upstream_pattern"]
        or result.provenance.upstream_contract_version != metadata["upstream_version"]
    ):
        raise TacticalSignalContractError("Procedencia incompatible con el patron solicitado.")
    expected_publication = metadata.get("publication_fingerprint")
    if (
        expected_publication is not None
        and result.provenance.upstream_publication_fingerprint != expected_publication
    ):
        raise TacticalSignalContractError("Fingerprint agregado upstream incompatible.")
    if result.signal is not None:
        if result.abstained or result.abstention_condition is not None:
            raise TacticalSignalContractError("Una senal presente no puede abstener.")
        validate_tactical_signal(result.signal)
        if result.signal.pattern_id != result.requested_pattern_id:
            raise TacticalSignalContractError("La senal no corresponde al patron solicitado.")
        if (
            result.upstream_state != result.signal.upstream_state
            or result.reason_codes != result.signal.reason_codes
            or result.context != result.signal.context
            or result.diagnostics != result.signal.diagnostics
            or result.provenance != result.signal.provenance
        ):
            raise TacticalSignalContractError("El resultado no reconcilia con la senal.")
        _validate_diagnostics(result.diagnostics, component_count=len(result.signal.components))
    else:
        if not result.abstained or result.abstention_condition not in {
            "unknown",
            "not_documented",
            "censored",
            "ineligible",
        }:
            raise TacticalSignalContractError("La ausencia de senal exige abstencion explicita.")
        contract = _ABSTENTION_CONTRACTS[result.requested_pattern_id].get(result.upstream_state)
        if contract is None:
            raise TacticalSignalContractError("Estado upstream no permitido para la abstencion.")
        expected_condition, allowed_reasons = contract
        if result.abstention_condition != expected_condition:
            raise TacticalSignalContractError("Condicion de abstencion incompatible con el estado.")
        if result.requested_pattern_id == "P09" and result.upstream_state in (
            ReturnProfileState.UNKNOWN.value,
            ReturnProfileState.NOT_DOCUMENTED.value,
        ):
            if result.reason_codes not in _P09_ABSTENTION_REASON_TUPLES[result.upstream_state]:
                raise TacticalSignalContractError(
                    "reason_codes P09 no corresponde a una combinacion upstream alcanzable."
                )
        elif len(result.reason_codes) != 1 or result.reason_codes[0] not in allowed_reasons:
            raise TacticalSignalContractError("reason_codes incompatible con la abstencion.")
        _validate_diagnostics(result.diagnostics, component_count=0)


def _merge_contexts(contexts: tuple[SignalContext, ...]) -> SignalContext:
    names = (
        "serve_number",
        "serve_direction",
        "return_shot_type",
        "return_lateral_direction",
        "return_depth",
        "return_profile",
    )
    merged: dict[str, object] = {}
    for name in names:
        values = {getattr(context, name) for context in contexts if getattr(context, name) is not None}
        if len(values) > 1:
            raise TacticalSignalContractError(f"Contexto contradictorio en {name}.")
        merged[name] = next(iter(values)) if values else None
    return SignalContext(**merged)  # type: ignore[arg-type]


def validate_tactical_signal_bundle(bundle: TacticalSignalBundle) -> None:
    """Valida orden, unicidad y coherencia cruzada de un bundle sintetico."""
    if type(bundle) is not TacticalSignalBundle:
        raise TypeError("bundle debe ser TacticalSignalBundle exacto.")
    if bundle.contract_version != SIGNAL_CONTRACT_VERSION:
        raise TacticalSignalContractError("contract_version invalida.")
    _strict_tuple("signals", bundle.signals)
    _strict_tuple("abstentions", bundle.abstentions)
    if not bundle.signals and not bundle.abstentions:
        raise TacticalSignalContractError("Un bundle debe contener al menos un resultado.")
    for signal in bundle.signals:
        validate_tactical_signal(signal)
    for result in bundle.abstentions:
        validate_signal_adaptation_result(result)
        if not result.abstained:
            raise TacticalSignalContractError("abstentions solo admite abstenciones.")
    signal_ids = tuple(signal.pattern_id for signal in bundle.signals)
    abstention_ids = tuple(result.requested_pattern_id for result in bundle.abstentions)
    pattern_ids = signal_ids + abstention_ids
    if len(set(pattern_ids)) != len(pattern_ids):
        raise TacticalSignalContractError("El bundle no admite patrones duplicados.")
    if signal_ids != tuple(sorted(signal_ids, key=ELIGIBLE_PATTERN_ORDER.index)):
        raise TacticalSignalContractError("Las senales no usan el orden canonico.")
    if abstention_ids != tuple(sorted(abstention_ids, key=ELIGIBLE_PATTERN_ORDER.index)):
        raise TacticalSignalContractError("Las abstenciones no usan el orden canonico.")
    contexts = tuple(signal.context for signal in bundle.signals) + tuple(
        result.context for result in bundle.abstentions
    )
    expected_context = _merge_contexts(contexts)
    if bundle.context != expected_context:
        raise TacticalSignalContractError("El contexto agregado no reconcilia.")
    active = {signal.pattern_id: signal for signal in bundle.signals}
    if "P09" in active:
        profile = active["P09"].context
        crosswalk = {
            "P04": ("return_lateral_direction", profile.return_lateral_direction),
            "P05": ("return_depth", profile.return_depth),
            "P06": ("return_shot_type", profile.return_shot_type),
        }
        for pattern_id, (field_name, expected) in crosswalk.items():
            if pattern_id in active and getattr(active[pattern_id].context, field_name) != expected:
                raise TacticalSignalContractError("P09 contradice un componente activo.")


def _diagnostics(*, component_count: int) -> tuple[tuple[str, int], ...]:
    return (("component_count", component_count),)


def _provenance(pattern_id: str, supplied: SignalProvenance | None) -> SignalProvenance:
    metadata = _PATTERN_METADATA[pattern_id]
    if supplied is None:
        return SignalProvenance(
            upstream_pattern=metadata["upstream_pattern"],
            adapter_version=ADAPTER_VERSION,
            upstream_contract_version=metadata["upstream_version"],
            registry_readiness=REGISTRY_READINESS,
            upstream_publication_fingerprint=metadata.get("publication_fingerprint"),
            synthetic_origin=True,
        )
    if type(supplied) is not SignalProvenance:
        raise TypeError("provenance debe ser SignalProvenance exacto.")
    _validate_provenance(supplied)
    if (
        supplied.upstream_pattern != metadata["upstream_pattern"]
        or supplied.upstream_contract_version != metadata["upstream_version"]
        or (
            metadata.get("publication_fingerprint") is not None
            and supplied.upstream_publication_fingerprint
            != metadata["publication_fingerprint"]
        )
    ):
        raise TacticalSignalContractError("La procedencia no corresponde al adaptador.")
    return supplied


def _active_result(
    pattern_id: str,
    *,
    tactical_value: str,
    components: tuple[tuple[str, str], ...],
    upstream_state: str,
    reason_codes: tuple[str, ...],
    context: SignalContext,
    diagnostics: tuple[tuple[str, int], ...],
    provenance: SignalProvenance,
) -> SignalAdaptationResult:
    metadata = _PATTERN_METADATA[pattern_id]
    signal = TacticalSignal(
        contract_version=SIGNAL_CONTRACT_VERSION,
        pattern_id=pattern_id,
        canonical_name=metadata["canonical_name"],
        unit=metadata["unit"],
        actor=metadata["actor"],
        tactical_value=tactical_value,
        components=components,
        upstream_state=upstream_state,
        eligible=True,
        comparable=True,
        outcome_available=True,
        registry_readiness=REGISTRY_READINESS,
        reason_codes=reason_codes,
        context=context,
        provenance=provenance,
        abstention_condition=None,
        diagnostics=diagnostics,
    )
    return SignalAdaptationResult(
        contract_version=SIGNAL_CONTRACT_VERSION,
        requested_pattern_id=pattern_id,
        signal=signal,
        abstained=False,
        upstream_state=upstream_state,
        reason_codes=reason_codes,
        abstention_condition=None,
        context=context,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _abstained_result(
    pattern_id: str,
    *,
    upstream_state: str,
    reason_codes: tuple[str, ...],
    condition: str,
    context: SignalContext,
    diagnostics: tuple[tuple[str, int], ...],
    provenance: SignalProvenance,
) -> SignalAdaptationResult:
    return SignalAdaptationResult(
        contract_version=SIGNAL_CONTRACT_VERSION,
        requested_pattern_id=pattern_id,
        signal=None,
        abstained=True,
        upstream_state=upstream_state,
        reason_codes=reason_codes,
        abstention_condition=condition,
        context=context,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def adapt_first_serve_direction_signal(
    classification: FirstServeDirectionClassification,
    *,
    provenance: SignalProvenance | None = None,
) -> SignalAdaptationResult:
    """Adapta la clasificacion individual P02 sin parser ni texto publicado."""
    if type(classification) is not FirstServeDirectionClassification:
        raise TypeError("P02 exige FirstServeDirectionClassification exacta.")
    validate_first_serve_direction_classification(classification)
    source = _provenance("P02", provenance)
    observed = classification.classification_state is FirstServeDirectionState.OBSERVED
    context = SignalContext(
        serve_number=1,
        serve_direction=_serve_context(classification.direction_code),
    )
    reasons = tuple(reason.value for reason in classification.reason_codes)
    diagnostics = _diagnostics(component_count=1 if observed else 0)
    if observed:
        value = context.serve_direction
        if value not in {"wide", "body", "T"}:
            raise TacticalSignalContractError("P02 observado sin direccion 4/5/6.")
        return _active_result(
            "P02",
            tactical_value=value,
            components=(("serve_direction", value),),
            upstream_state=classification.classification_state.value,
            reason_codes=reasons,
            context=context,
            diagnostics=diagnostics,
            provenance=source,
        )
    condition = (
        "censored"
        if classification.classification_state is FirstServeDirectionState.CENSORED
        else "unknown"
    )
    return _abstained_result(
        "P02",
        upstream_state=classification.classification_state.value,
        reason_codes=reasons,
        condition=condition,
        context=context,
        diagnostics=diagnostics,
        provenance=source,
    )


def _require_parsed(parsed: object) -> ParseResult:
    if type(parsed) is not ParseResult:
        raise TypeError("El adaptador exige el ParseResult original exacto.")
    return parsed


def _serve_context(code: str | None) -> str | None:
    if code is None:
        return None
    return {"4": "wide", "5": "body", "6": "T", "0": "unknown"}[code]


def adapt_return_direction_signal(
    classification: ReturnDirectionClassification,
    *,
    parsed: ParseResult,
    provenance: SignalProvenance | None = None,
) -> SignalAdaptationResult:
    """Adapta P04 sin volver a ejecutar parser ni extractor."""
    if type(classification) is not ReturnDirectionClassification:
        raise TypeError("P04 exige ReturnDirectionClassification exacto.")
    validate_return_direction_classification(classification, parsed=_require_parsed(parsed))
    source = _provenance("P04", provenance)
    observed = classification.classification_state is ReturnDirectionState.OBSERVED
    context = SignalContext(
        serve_number=classification.serve_number,
        serve_direction=_serve_context(classification.service_direction_code),
        return_shot_type=(classification.return_shot_type_code if classification.return_shot_type_code in DOCUMENTED_SHOT_TYPES else None),
        return_lateral_direction=(classification.lateral_direction_code if observed else None),
        return_depth=(classification.return_depth_code if classification.return_depth_code in {"7", "8", "9"} else None),
    )
    reasons = (classification.reason_code.value,)
    diagnostics = _diagnostics(component_count=1 if observed else 0)
    if observed:
        value = classification.lateral_direction_code
        if value not in {"1", "2", "3"}:
            raise TacticalSignalContractError("P04 observado sin lateralidad 1/2/3.")
        return _active_result(
            "P04",
            tactical_value=value,
            components=(("return_lateral_direction", value),),
            upstream_state=classification.classification_state.value,
            reason_codes=reasons,
            context=context,
            diagnostics=diagnostics,
            provenance=source,
        )
    condition = "unknown" if classification.classification_state in {
        ReturnDirectionState.UNKNOWN,
        ReturnDirectionState.UNKNOWN_INITIAL,
    } else "censored"
    return _abstained_result(
        "P04",
        upstream_state=classification.classification_state.value,
        reason_codes=reasons,
        condition=condition,
        context=context,
        diagnostics=diagnostics,
        provenance=source,
    )


def adapt_return_depth_signal(
    classification: ReturnDepthClassification,
    *,
    parsed: ParseResult,
    provenance: SignalProvenance | None = None,
) -> SignalAdaptationResult:
    """Adapta P05 sin volver a ejecutar parser ni extractor."""
    if type(classification) is not ReturnDepthClassification:
        raise TypeError("P05 exige ReturnDepthClassification exacto.")
    validate_return_depth_classification(classification, parsed=_require_parsed(parsed))
    source = _provenance("P05", provenance)
    observed = classification.classification_state is ReturnDepthState.OBSERVED
    context = SignalContext(
        serve_number=classification.serve_number,
        return_shot_type=(classification.return_shot_type if classification.return_shot_type in DOCUMENTED_SHOT_TYPES else None),
        return_lateral_direction=(classification.lateral_direction_code if classification.lateral_direction_code in {"1", "2", "3"} else None),
        return_depth=(classification.return_depth_code if observed else None),
    )
    reasons = tuple(reason.value for reason in classification.reason_codes)
    diagnostics = _diagnostics(component_count=1 if observed else 0)
    if observed:
        value = classification.return_depth_code
        if value not in {"7", "8", "9"}:
            raise TacticalSignalContractError("P05 observado sin profundidad 7/8/9.")
        return _active_result(
            "P05",
            tactical_value=value,
            components=(("return_depth", value),),
            upstream_state=classification.classification_state.value,
            reason_codes=reasons,
            context=context,
            diagnostics=diagnostics,
            provenance=source,
        )
    condition = {
        ReturnDepthState.UNKNOWN: "unknown",
        ReturnDepthState.NOT_DOCUMENTED: "not_documented",
        ReturnDepthState.UNKNOWN_INITIAL: "unknown",
        ReturnDepthState.CENSORED: "censored",
    }[classification.classification_state]
    return _abstained_result(
        "P05",
        upstream_state=classification.classification_state.value,
        reason_codes=reasons,
        condition=condition,
        context=context,
        diagnostics=diagnostics,
        provenance=source,
    )


def adapt_return_shot_type_signal(
    classification: ReturnShotTypeClassification,
    *,
    parsed: ParseResult,
    provenance: SignalProvenance | None = None,
) -> SignalAdaptationResult:
    """Adapta P06 sin volver a ejecutar parser ni extractor."""
    if type(classification) is not ReturnShotTypeClassification:
        raise TypeError("P06 exige ReturnShotTypeClassification exacto.")
    validate_return_shot_type_classification(classification, parsed=_require_parsed(parsed))
    source = _provenance("P06", provenance)
    observed = classification.classification_state is ReturnShotTypeState.OBSERVED
    context = SignalContext(
        serve_number=classification.serve_number,
        return_shot_type=(classification.return_shot_type_code if observed else None),
    )
    reasons = tuple(reason.value for reason in classification.reason_codes)
    diagnostics = _diagnostics(component_count=1 if observed else 0)
    if observed:
        value = classification.return_shot_type_code
        if value not in DOCUMENTED_SHOT_TYPES:
            raise TacticalSignalContractError("P06 observado sin tipo documentado.")
        return _active_result(
            "P06",
            tactical_value=value,
            components=(("return_shot_type", value),),
            upstream_state=classification.classification_state.value,
            reason_codes=reasons,
            context=context,
            diagnostics=diagnostics,
            provenance=source,
        )
    condition = "unknown" if classification.classification_state in {
        ReturnShotTypeState.UNKNOWN,
        ReturnShotTypeState.UNKNOWN_INITIAL,
    } else "censored"
    return _abstained_result(
        "P06",
        upstream_state=classification.classification_state.value,
        reason_codes=reasons,
        condition=condition,
        context=context,
        diagnostics=diagnostics,
        provenance=source,
    )


def adapt_return_profile_signal(
    classification: ReturnProfileClassification,
    *,
    parsed: ParseResult,
    provenance: SignalProvenance | None = None,
) -> SignalAdaptationResult:
    """Adapta P09 sin volver a ejecutar parser ni extractor."""
    if type(classification) is not ReturnProfileClassification:
        raise TypeError("P09 exige ReturnProfileClassification exacto.")
    validate_return_profile_classification(classification, parsed=_require_parsed(parsed))
    source = _provenance("P09", provenance)
    observed = classification.classification_state is ReturnProfileState.OBSERVED
    context = SignalContext(
        serve_number=classification.serve_number,
        serve_direction=_serve_context(classification.service_direction_code),
        return_shot_type=(classification.return_shot_type_code if classification.return_shot_type_code in DOCUMENTED_SHOT_TYPES else None),
        return_lateral_direction=(classification.lateral_direction_code if classification.lateral_direction_code in {"1", "2", "3"} else None),
        return_depth=(classification.return_depth_code if classification.return_depth_code in {"7", "8", "9"} else None),
        return_profile=(classification.profile_id if observed else None),
    )
    reasons = tuple(reason.value for reason in classification.reason_codes)
    diagnostics = _diagnostics(component_count=3 if observed else 0)
    if observed:
        profile_id = classification.profile_id
        if profile_id not in DOCUMENTED_PROFILE_IDS:
            raise TacticalSignalContractError("P09 observado sin profile_id valido.")
        components = (
            ("return_shot_type", classification.return_shot_type_code),
            ("return_lateral_direction", classification.lateral_direction_code),
            ("return_depth", classification.return_depth_code),
        )
        return _active_result(
            "P09",
            tactical_value=profile_id,
            components=components,  # type: ignore[arg-type]
            upstream_state=classification.classification_state.value,
            reason_codes=reasons,
            context=context,
            diagnostics=diagnostics,
            provenance=source,
        )
    condition = {
        ReturnProfileState.UNKNOWN: "unknown",
        ReturnProfileState.NOT_DOCUMENTED: "not_documented",
        ReturnProfileState.UNKNOWN_INITIAL: "unknown",
        ReturnProfileState.CENSORED: "censored",
    }[classification.classification_state]
    return _abstained_result(
        "P09",
        upstream_state=classification.classification_state.value,
        reason_codes=reasons,
        condition=condition,
        context=context,
        diagnostics=diagnostics,
        provenance=source,
    )


def adapt_tactical_signal(
    pattern_id: str,
    upstream_result: object,
    *,
    parsed: ParseResult | None = None,
    provenance: SignalProvenance | None = None,
) -> SignalAdaptationResult:
    """Despacho cerrado; P03/P07/P08 y patrones no registrados se rechazan."""
    if type(pattern_id) is not str or pattern_id not in ELIGIBLE_PATTERN_IDS:
        raise TacticalSignalContractError("Solo P02/P04/P05/P06/P09 son adaptables.")
    if pattern_id == "P02":
        if parsed is not None:
            raise TacticalSignalContractError("P02 no acepta ParseResult separado.")
        return adapt_first_serve_direction_signal(upstream_result, provenance=provenance)  # type: ignore[arg-type]
    if parsed is None:
        raise TacticalSignalContractError("El adaptador exige el ParseResult original.")
    adapters = {
        "P04": adapt_return_direction_signal,
        "P05": adapt_return_depth_signal,
        "P06": adapt_return_shot_type_signal,
        "P09": adapt_return_profile_signal,
    }
    return adapters[pattern_id](upstream_result, parsed=parsed, provenance=provenance)  # type: ignore[arg-type]


def build_tactical_signal_bundle(
    results: tuple[SignalAdaptationResult, ...],
) -> TacticalSignalBundle:
    """Agrupa resultados de un unico intento de saque sin completar ausencias.

    Para representar primer y segundo intento deben construirse bundles
    distintos. La reconciliacion de contexto rechaza cualquier mezcla.
    """
    _strict_tuple("results", results)
    if not results:
        raise TacticalSignalContractError("results no puede estar vacio.")
    for result in results:
        validate_signal_adaptation_result(result)
    ordered = tuple(sorted(results, key=lambda item: ELIGIBLE_PATTERN_ORDER.index(item.requested_pattern_id)))
    if len({item.requested_pattern_id for item in ordered}) != len(ordered):
        raise TacticalSignalContractError("No se admiten patrones duplicados.")
    signals = tuple(item.signal for item in ordered if item.signal is not None)
    abstentions = tuple(item for item in ordered if item.signal is None)
    contexts = tuple(item.context for item in ordered)
    return TacticalSignalBundle(
        contract_version=SIGNAL_CONTRACT_VERSION,
        signals=signals,
        abstentions=abstentions,
        context=_merge_contexts(contexts),
    )


def _diagnostics_structure(diagnostics: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return {key: value for key, value in diagnostics}


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


def signal_to_json_structure(signal: TacticalSignal) -> dict[str, object]:
    """Devuelve una estructura JSON-safe cerrada, sin datos individuales."""
    validate_tactical_signal(signal)
    return {
        "abstention_condition": signal.abstention_condition,
        "actor": signal.actor,
        "canonical_name": signal.canonical_name,
        "comparable": signal.comparable,
        "components": [
            {"name": name, "value": value} for name, value in signal.components
        ],
        "context": _context_structure(signal.context),
        "contract_version": signal.contract_version,
        "diagnostics": _diagnostics_structure(signal.diagnostics),
        "eligible": signal.eligible,
        "outcome_available": signal.outcome_available,
        "pattern_id": signal.pattern_id,
        "provenance": _provenance_structure(signal.provenance),
        "reason_codes": list(signal.reason_codes),
        "registry_readiness": signal.registry_readiness,
        "tactical_value": signal.tactical_value,
        "unit": signal.unit,
        "upstream_state": signal.upstream_state,
    }


def _adaptation_to_json_structure(result: SignalAdaptationResult) -> dict[str, object]:
    validate_signal_adaptation_result(result)
    return {
        "abstained": result.abstained,
        "abstention_condition": result.abstention_condition,
        "context": _context_structure(result.context),
        "contract_version": result.contract_version,
        "diagnostics": _diagnostics_structure(result.diagnostics),
        "provenance": _provenance_structure(result.provenance),
        "reason_codes": list(result.reason_codes),
        "requested_pattern_id": result.requested_pattern_id,
        "signal": None if result.signal is None else signal_to_json_structure(result.signal),
        "upstream_state": result.upstream_state,
    }


def bundle_to_json_structure(bundle: TacticalSignalBundle) -> dict[str, object]:
    """Devuelve el bundle en orden contractual y con abstenciones separadas."""
    validate_tactical_signal_bundle(bundle)
    return {
        "abstentions": [_adaptation_to_json_structure(item) for item in bundle.abstentions],
        "context": _context_structure(bundle.context),
        "contract_version": bundle.contract_version,
        "signals": [signal_to_json_structure(item) for item in bundle.signals],
    }


def canonical_json_bytes(value: TacticalSignal | TacticalSignalBundle) -> bytes:
    """Serializa solo tipos contractuales con JSON canonico UTF-8."""
    if type(value) is TacticalSignal:
        structure = signal_to_json_structure(value)
    elif type(value) is TacticalSignalBundle:
        structure = bundle_to_json_structure(value)
    else:
        raise TypeError("Solo se serializan TacticalSignal o TacticalSignalBundle.")
    return json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def tactical_signal_fingerprint(signal: TacticalSignal) -> str:
    payload = b"tennis-tactical-signal\x00" + canonical_json_bytes(signal)
    return sha256(payload).hexdigest().upper()


def tactical_signal_bundle_fingerprint(bundle: TacticalSignalBundle) -> str:
    payload = b"tennis-tactical-signal-bundle\x00" + canonical_json_bytes(bundle)
    return sha256(payload).hexdigest().upper()


__all__ = [
    "ADAPTER_VERSION",
    "ADAPTABLE_PATTERN_IDS",
    "ADAPTABLE_PATTERN_ORDER",
    "ELIGIBLE_PATTERN_IDS",
    "ELIGIBLE_PATTERN_ORDER",
    "P02_AGGREGATE_PUBLICATION_FINGERPRINT",
    "REGISTRY_READINESS",
    "SIGNAL_CONTRACT_VERSION",
    "SignalAdaptationResult",
    "SignalContext",
    "SignalProvenance",
    "TacticalSignal",
    "TacticalSignalBundle",
    "TacticalSignalContractError",
    "adapt_return_depth_signal",
    "adapt_return_direction_signal",
    "adapt_return_profile_signal",
    "adapt_return_shot_type_signal",
    "adapt_first_serve_direction_signal",
    "adapt_tactical_signal",
    "build_tactical_signal_bundle",
    "bundle_to_json_structure",
    "canonical_json_bytes",
    "signal_to_json_structure",
    "tactical_signal_bundle_fingerprint",
    "tactical_signal_fingerprint",
    "validate_signal_adaptation_result",
    "validate_tactical_signal",
    "validate_tactical_signal_bundle",
]
