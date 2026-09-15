"""UI Streamlit local P18: interfaz sobre la API HTTP P17 (solo loopback).

Interfaz Streamlit sencilla, en español y sin estado, que consume
EXCLUSIVAMENTE la API HTTP local P17 mediante HTTP (cliente httpx ya
disponibles en el entorno). No importa ningun modulo P10-P16, no
carga el snapshot P13, no lee Parquet/CSV/reports, no conoce la
variable de entorno de ruta del snapshot, no inicia Uvicorn, no hace
analytics/telemetria y no ejecuta nada al importar.

Contrato cerrado del modulo:

- Endpoint fijo por defecto ``http://127.0.0.1:8000``; solo se acepta
  configurar otra URL loopback segura: esquema ``http`` (unicamente),
  host exactamente ``127.0.0.1``, puerto opcional ``1..65535``
  estricto, sin credenciales, sin path, sin query ni fragmento.
- Validacion local estricta de jugador/rival (mismo contrato de
  identificador que la API P12: 1..64 chars, sin CONTROL, sin
  fragmentos prohibidos) y fecha civil ISO estricta.
- Un boton explícito, una unica peticion por pulsacion y CERO reintentos
  automaticos; timeout cliente explicito y cerrado (30 s).
- Respuestas: 200 se parsea con estrictitud total contra el contrato
  publico P11 y se renderiza; los status 404/422/500/502/503/504 y los
  fallos de red local (conexion rechazada, timeout, respuesta
  incompatible) producen mensajes cerrados en espanol, sin rutas, sin
  request IDs, sin tracebacks ni errores internos.
- Nada de identidad, payload, URL ni respuesta se escribe en logs,
  memoria de sesion, caché ni analytics: sin estado global persistible.
- La interfaz no afirma causalidad ni garantía de éxito: las fichas
  son evidencia historica observacional; si el sistema abstiene o la
  orientación no está disponible, la UI lo muestra sin inventar
  recomendaciones.
- Ejecucion: ``streamlit run src/ui/streamlit_app.py
  --browser.gatherUsageStats false`` (el guard
  ``__main__`` lanza la UI); importar el modulo no inicia Streamlit ni
  red.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx

import streamlit as st

from src.recommender import tactical_recommendation_contract as p11


UI_API_BASE_DEFAULT: Final = "http://127.0.0.1:8000"
UI_API_HOST_ALLOWED: Final = "127.0.0.1"
UI_CONTAINER_API_BASE_URL: Final = "http://api:8000"
UI_MIN_PORT: Final = 1
UI_MAX_PORT: Final = 65535
UI_REQUEST_TIMEOUT_SECONDS: Final = 30.0
UI_MAX_RESPONSE_BYTES: Final = 1024 * 1024
UI_RECOMMENDATIONS_PATH: Final = "/api/v1/recommendations"
UI_MAX_IDENTIFIER_LENGTH: Final = 64
UI_IDENTIFIER_CONTROL: Final = frozenset(
    tuple(range(0x00, 0x20)) + (0x7F,) + tuple(range(0x80, 0xA0))
)
UI_IDENTIFIER_FORBIDDEN_FRAGMENTS: Final = ("/", "\\", "://", "..", "~")
_CIVIL_ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_STRICT_PORT: Final = re.compile(r"[1-9][0-9]{0,4}")


_UI_URL_MESSAGES: Final = {
    "scheme": "Solo se acepta http (loopback local).",
    "host": "Solo se acepta el host 127.0.0.1 (loopback local).",
    "credentials": "No se aceptan credenciales en la URL local.",
    "path": "La URL del servicio debe ser la base (sin path).",
    "query": "La URL del servicio no admite query strings.",
    "fragment": "La URL del servicio no admite fragmentos.",
    "port": "El puerto debe ser un entero estricto entre 1 y 65535.",
    "format": "La URL debe ser http://127.0.0.1[:puerto] sin espacios.",
}

_UI_LOCAL_OUTCOMES: Final = {
    "ok": "ok",
    "status_error": "status_error",
    "incompatible_response": "incompatible_response",
    "connection": "connection",
    "timeout": "timeout",
}

_UI_STATUS_MESSAGES: Final = {
    404: (
        "No existe una ficha pública para esa orientación "
        "(jugador–rival–fecha)."
    ),
    422: (
        "La solicitud no cumple el contrato de la API: revisa jugador, "
        "rival y formato de fecha."
    ),
    500: "Fallo interno del servicio local. Inténtalo más tarde.",
    502: "El servicio local devolvió un resultado fuera de contrato.",
    503: (
        "La ficha existe pero no está disponible ahora, o el servicio "
        "local no está listo. Inténtalo más tarde."
    ),
    504: "El servicio local tardó demasiado en responder.",
}

_UI_LOCAL_MESSAGES: Final = {
    "incompatible_response": (
        "El servicio local devolvió una respuesta incompatible con el "
        "contrato público."
    ),
    "connection": (
        "No se pudo conectar al servicio local. Comprueba que el "
        "proceso P17 ya está corriendo."
    ),
    "timeout": "El servicio local no respondió dentro del tiempo límite.",
    "unexpected": (
        "No se pudo completar la solicitud (fallo local sanitizado)."
    ),
}


class PublicContractError(ValueError):
    """La respuesta no cumple el contrato público P11 cerrado."""

    __slots__ = ()


class _IdentifierContractError(ValueError):
    """Identificador local fuera del contrato cerrado (mensaje propio)."""

    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        object.__setattr__(self, "message", message)
        super().__init__(message)


def validate_api_base_url(raw: object) -> str:
    """Valida y normaliza la URL del servicio local (solo loopback http)."""
    if type(raw) is not str or not raw or raw != raw.strip():
        raise ValueError(_UI_URL_MESSAGES["format"])
    parts = urlsplit(raw)
    if parts.scheme != "http":
        raise ValueError(_UI_URL_MESSAGES["scheme"])
    netloc = parts.netloc
    if "@" in netloc:
        raise ValueError(_UI_URL_MESSAGES["credentials"])
    if ":" in netloc:
        host_part, port_str = netloc.rsplit(":", 1)
        if _STRICT_PORT.fullmatch(port_str) is None or not (
            UI_MIN_PORT <= int(port_str) <= UI_MAX_PORT
        ):
            raise ValueError(_UI_URL_MESSAGES["port"])
        port = int(port_str)
    else:
        host_part = netloc
        port = 8000
    if host_part != UI_API_HOST_ALLOWED:
        raise ValueError(_UI_URL_MESSAGES["host"])
    if parts.path not in ("", None):
        raise ValueError(_UI_URL_MESSAGES["path"])
    if parts.query:
        raise ValueError(_UI_URL_MESSAGES["query"])
    if parts.fragment:
        raise ValueError(_UI_URL_MESSAGES["fragment"])
    return f"http://{UI_API_HOST_ALLOWED}:{port}"


def validate_container_api_base_url(raw: object) -> str:
    """Valida el endpoint interno fijo del modo contenedor P19.

    No parsea ni acepta entrada del usuario: el unico valor aceptado es
    la constante ``UI_CONTAINER_API_BASE_URL`` (servicio Compose interno
    ``api``). Cualquier otro valor se rechaza; esta funcion nunca
    produce un host distinto de la constante cerrada.
    """
    if raw != UI_CONTAINER_API_BASE_URL:
        raise ValueError(_UI_URL_MESSAGES["host"])
    return UI_CONTAINER_API_BASE_URL


def validate_local_identifier(raw: object, kind: str) -> str:
    """Valida jugador/rival con el mismo criterio cerrado que la API P12."""
    label = "Jugador" if kind == "player" else "Rival"
    closed = f"{label}: identificador fuera del contrato."
    if type(raw) is not str:
        raise _IdentifierContractError(closed)
    if (
        not raw
        or raw != raw.strip()
        or len(raw) > UI_MAX_IDENTIFIER_LENGTH
        or any(ord(char) in UI_IDENTIFIER_CONTROL for char in raw)
        or any(fragment in raw for fragment in UI_IDENTIFIER_FORBIDDEN_FRAGMENTS)
    ):
        raise _IdentifierContractError(closed)
    return raw


def _is_civil_iso_date(raw: object) -> bool:
    if type(raw) is not str or raw != raw.strip():
        return False
    if _CIVIL_ISO_DATE.fullmatch(raw) is None:
        return False
    try:
        date.fromisoformat(raw)
    except ValueError:
        return False
    return True


def validate_local_date(raw: object) -> str:
    closed = "Fecha: usa el formato YYYY-MM-DD con una fecha civil real."
    if not _is_civil_iso_date(raw):
        raise _IdentifierContractError(closed)
    return raw  # type: ignore[return-value]


@dataclass(frozen=True)
class PublicEvidenceComponent:
    perspective: str
    scope: str
    evidence_state: str
    labeled_activations: int
    successes: int
    failures: int
    distinct_matches: int
    success_rate: float | None
    wilson_lower: float | None
    wilson_upper: float | None


@dataclass(frozen=True)
class PublicOption:
    pattern_id: str
    category: str
    tactical_opportunity: str
    actor: str
    status: str
    reason_codes: tuple[str, ...]
    executor_evidence: PublicEvidenceComponent
    opponent_allowed_evidence: PublicEvidenceComponent
    score: float | None
    descriptive_uncertainty_envelope: tuple[float, float] | None
    rank_position: int | None
    tie_group: int | None
    canonical_explanation: str


@dataclass(frozen=True)
class PublicPatternCard:
    pattern_id: str
    actor: str
    tactical_opportunity: str
    status: str
    status_reason_codes: tuple[str, ...]
    categories: tuple[str, ...]
    options: tuple[PublicOption, ...]
    ranked_options: tuple[str, ...]
    top_options: tuple[str, ...]
    abstained_options: tuple[str, ...]
    total_options: int
    scored_options: int
    abstained_options_count: int


@dataclass(frozen=True)
class PublicRecommendation:
    status: str
    status_reason_codes: tuple[str, ...]
    cards: tuple[PublicPatternCard, ...]
    limitations: tuple[str, ...]


_RESPONSE_KEYS: Final = frozenset(
    {
        "contract_version",
        "status",
        "status_reason_codes",
        "methodology",
        "combination",
        "score_formula",
        "uncertainty_method",
        "executor_weight",
        "opponent_weight",
        "encoder_policy",
        "evidence_scope",
        "minimum_labeled_activations",
        "minimum_distinct_matches",
        "requested_top_k",
        "ranking_scope",
        "global_cross_pattern_ranking",
        "cards",
        "limitations",
        "reconciliations",
        "fingerprint",
    }
)
_CARD_KEYS: Final = frozenset(
    {
        "pattern_id",
        "actor",
        "tactical_opportunity",
        "status",
        "status_reason_codes",
        "categories",
        "options",
        "ranked_options",
        "top_options",
        "abstained_options",
        "requested_top_k",
        "effective_top_k",
        "tie_expanded",
        "tie_group_count",
        "total_options",
        "scored_options",
        "abstained_options_count",
        "reconciliations",
    }
)
_OPTION_KEYS: Final = frozenset(
    {
        "pattern_id",
        "category",
        "tactical_opportunity",
        "actor",
        "status",
        "reason_codes",
        "executor_evidence",
        "opponent_allowed_evidence",
        "score_formula",
        "score",
        "descriptive_uncertainty_envelope",
        "rank_position",
        "tie_group",
        "canonical_explanation",
    }
)
_EVIDENCE_KEYS: Final = frozenset(
    {
        "perspective",
        "scope",
        "evidence_state",
        "labeled_activations",
        "successes",
        "failures",
        "distinct_matches",
        "success_rate",
        "wilson_lower",
        "wilson_upper",
    }
)
_CARD_RECONCILIATION_KEYS: Final = frozenset(
    {
        "catalog_complete",
        "options_in_catalog_order",
        "ranked_and_abstained_partition",
        "rank_positions_reconciled",
        "tie_expansions_preserved",
        "counts_reconciled",
    }
)
_RESPONSE_RECONCILIATION_KEYS: Final = frozenset(
    {
        "policy_frozen",
        "four_scoreable_patterns_ordered",
        "no_cross_pattern_ranking",
        "status_reconciled",
        "identity_and_upstream_fingerprints_redacted",
        "descriptive_not_causal",
    }
)
_RESPONSE_STATUSES: Final = frozenset(
    {"available", "partially_available", "not_available"}
)
_OPTION_STATUSES: Final = frozenset(
    {
        "ranked",
        "abstained_insufficient_evidence",
        "not_applicable",
        "not_available",
    }
)
_PATTERNS: Final = frozenset({"P02", "P04", "P05", "P06"})
_ACTORS: Final = frozenset({"server", "returner"})
_PUBLIC_PATTERN_CONTRACT: Final = {
    "P02": ("first_serve_direction", "server"),
    "P04": ("initial_return_direction", "returner"),
    "P05": ("initial_return_depth", "returner"),
    "P06": ("initial_return_shot_type", "returner"),
}
_PERSPECTIVES: Final = frozenset({"executor", "opponent_allowed"})
_EVIDENCE_SCOPES: Final = frozenset({"global"})
_EVIDENCE_STATES: Final = frozenset(
    {
        "available",
        "insufficient_labeled_attempts",
        "insufficient_matches",
        "no_observed_category",
        "not_applicable",
        "not_available",
    }
)
_UNIT_RATES: Final = "success_rate|wilson_lower|wilson_upper"


def _require_keys(value: object, keys: frozenset[str], label: str) -> dict:
    if type(value) is not dict:
        raise PublicContractError(f"{label} debe ser un objeto JSON.")
    if set(value.keys()) != set(keys):
        raise PublicContractError(f"{label} fuera del contrato público.")
    return value  # type: ignore[return-value]


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("objeto JSON con clave duplicada")
        result[key] = value
    return result


def _require_str(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise PublicContractError(f"{label} fuera de contrato.")
    return value


def _require_str_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item for item in value)
    ):
        raise PublicContractError(f"{label} fuera de contrato.")
    return tuple(value)


def _require_int(value: object, label: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise PublicContractError(f"{label} fuera de contrato.")
    return value


def _require_rate(value: object, label: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise PublicContractError(f"{label} fuera de contrato.")
    return value


def _require_all_true(value: object, keys: frozenset[str], label: str) -> None:
    record = _require_keys(value, keys, label)
    if any(item is not True for item in record.values()):
        raise PublicContractError(f"{label} fuera de contrato.")


def _require_pattern_binding(record: dict, label: str) -> None:
    pattern = record["pattern_id"]
    if pattern not in _PATTERNS:
        raise PublicContractError(f"{label}.pattern_id fuera de contrato.")
    opportunity, actor = _PUBLIC_PATTERN_CONTRACT[pattern]
    if (
        record["tactical_opportunity"] != opportunity
        or record["actor"] != actor
    ):
        raise PublicContractError(
            f"{label}: oportunidad/actor fuera del contrato del patron."
        )


def _parse_evidence(value: object, label: str) -> PublicEvidenceComponent:
    record = _require_keys(value, _EVIDENCE_KEYS, label)
    perspective = _require_str(record["perspective"], f"{label}.perspective")
    if perspective not in _PERSPECTIVES:
        raise PublicContractError(f"{label}.perspective fuera de contrato.")
    scope = _require_str(record["scope"], f"{label}.scope")
    if scope not in _EVIDENCE_SCOPES:
        raise PublicContractError(f"{label}.scope fuera de contrato.")
    evidence_state = _require_str(
        record["evidence_state"], f"{label}.evidence_state"
    )
    if evidence_state not in _EVIDENCE_STATES:
        raise PublicContractError(f"{label}.evidence_state fuera de contrato.")
    labeled = _require_int(record["labeled_activations"], "labeled_activations")
    successes = _require_int(record["successes"], "successes")
    failures = _require_int(record["failures"], "failures")
    distinct = _require_int(record["distinct_matches"], "distinct_matches")
    if successes + failures != labeled or distinct > labeled:
        raise PublicContractError(f"{label} no reconcilia sus activaciones.")
    rate = _require_rate(record["success_rate"], f"{label}.success_rate")
    lower = _require_rate(record["wilson_lower"], f"{label}.wilson_lower")
    upper = _require_rate(record["wilson_upper"], f"{label}.wilson_upper")
    values = (rate, lower, upper)
    if labeled == 0:
        if any(item is not None for item in values):
            raise PublicContractError(
                f"{label}: sin activations no hay tasa ni Wilson."
            )
    elif any(item is None for item in values):
        raise PublicContractError(
            f"{label}: con activations deben existir tasa y Wilson."
        )
    return PublicEvidenceComponent(
        perspective=perspective,
        scope=scope,
        evidence_state=evidence_state,
        labeled_activations=labeled,
        successes=successes,
        failures=failures,
        distinct_matches=distinct,
        success_rate=rate,
        wilson_lower=lower,
        wilson_upper=upper,
    )


def _parse_option(value: object, index: int) -> PublicOption:
    record = _require_keys(value, _OPTION_KEYS, f"opcion[{index}]")
    _require_pattern_binding(record, f"opcion[{index}]")
    if record["status"] not in _OPTION_STATUSES:
        raise PublicContractError(f"opcion[{index}].status fuera de contrato.")
    envelope = record["descriptive_uncertainty_envelope"]
    if envelope is not None:
        if (
            type(envelope) is not list
            or len(envelope) != 2
            or any(
                type(item) is not float or isinstance(item, bool)
                for item in envelope
            )
            or not 0.0 <= envelope[0] <= envelope[1] <= 1.0
        ):
            raise PublicContractError(
                f"opcion[{index}].descriptive_uncertainty_envelope fuera de contrato."
            )
        envelope_value: tuple[float, float] | None = (
            float(envelope[0]),
            float(envelope[1]),
        )
    else:
        envelope_value = None
    rank = record["rank_position"]
    tie = record["tie_group"]
    if rank is not None and (
        type(rank) is not int or isinstance(rank, bool) or rank < 1
    ):
        raise PublicContractError(f"opcion[{index}].rank_position fuera de contrato.")
    if tie is not None and (
        type(tie) is not int or isinstance(tie, bool) or tie < 1
    ):
        raise PublicContractError(f"opcion[{index}].tie_group fuera de contrato.")
    score = _require_rate(record["score"], "score")
    if record["status"] == "ranked":
        if (
            score is None
            or envelope_value is None
            or rank is None
            or tie is None
            or not envelope_value[0] <= score <= envelope_value[1]
        ):
            raise PublicContractError(
                f"opcion[{index}].ranked incompleta o sin score en envolvente."
            )
    elif (
        score is not None
        or envelope_value is not None
        or rank is not None
        or tie is not None
    ):
        raise PublicContractError(
            f"opcion[{index}].abstencion conserva score o ranking."
        )
    return PublicOption(
        pattern_id=record["pattern_id"],
        category=_require_str(record["category"], "category"),
        tactical_opportunity=_require_str(
            record["tactical_opportunity"], "tactical_opportunity"
        ),
        actor=record["actor"],
        status=record["status"],
        reason_codes=_require_str_tuple(record["reason_codes"], "reason_codes"),
        executor_evidence=_parse_evidence(
            record["executor_evidence"], f"opcion[{index}].executor_evidence"
        ),
        opponent_allowed_evidence=_parse_evidence(
            record["opponent_allowed_evidence"],
            f"opcion[{index}].opponent_allowed_evidence",
        ),
        score=score,
        descriptive_uncertainty_envelope=envelope_value,
        rank_position=rank,
        tie_group=tie,
        canonical_explanation=_require_str(
            record["canonical_explanation"], "canonical_explanation"
        ),
    )


def _parse_card(value: object, index: int) -> PublicPatternCard:
    record = _require_keys(value, _CARD_KEYS, f"tarjeta[{index}]")
    _require_pattern_binding(record, f"tarjeta[{index}]")
    if record["status"] not in _RESPONSE_STATUSES:
        raise PublicContractError(f"tarjeta[{index}].status fuera de contrato.")
    options_raw = record["options"]
    if type(options_raw) is not list:
        raise PublicContractError(f"tarjeta[{index}].options fuera de contrato.")
    options = tuple(_parse_option(item, number) for number, item in enumerate(options_raw))
    _require_all_true(
        record["reconciliations"],
        _CARD_RECONCILIATION_KEYS,
        f"tarjeta[{index}].reconciliations",
    )
    return PublicPatternCard(
        pattern_id=record["pattern_id"],
        actor=record["actor"],
        tactical_opportunity=_require_str(
            record["tactical_opportunity"], "tactical_opportunity"
        ),
        status=record["status"],
        status_reason_codes=_require_str_tuple(
            record["status_reason_codes"], "status_reason_codes"
        ),
        categories=_require_str_tuple(record["categories"], "categories"),
        options=options,
        ranked_options=_require_str_tuple(
            record["ranked_options"], "ranked_options"
        ),
        top_options=_require_str_tuple(record["top_options"], "top_options"),
        abstained_options=_require_str_tuple(
            record["abstained_options"], "abstained_options"
        ),
        total_options=_require_int(record["total_options"], "total_options"),
        scored_options=_require_int(record["scored_options"], "scored_options"),
        abstained_options_count=_require_int(
            record["abstained_options_count"], "abstained_options_count"
        ),
    )


def _p11_evidence(value: object, label: str) -> p11.PublicEvidenceComponent:
    record = _require_keys(value, _EVIDENCE_KEYS, label)
    return p11.PublicEvidenceComponent(**record)


def _p11_option(value: object, index: int) -> p11.TacticalRecommendationOption:
    record = _require_keys(value, _OPTION_KEYS, f"opcion[{index}]")
    envelope = record["descriptive_uncertainty_envelope"]
    return p11.TacticalRecommendationOption(
        pattern_id=record["pattern_id"],
        category=record["category"],
        tactical_opportunity=record["tactical_opportunity"],
        actor=record["actor"],
        status=record["status"],
        reason_codes=tuple(record["reason_codes"]),
        executor_evidence=_p11_evidence(
            record["executor_evidence"], f"opcion[{index}].executor_evidence"
        ),
        opponent_allowed_evidence=_p11_evidence(
            record["opponent_allowed_evidence"],
            f"opcion[{index}].opponent_allowed_evidence",
        ),
        score_formula=record["score_formula"],
        score=record["score"],
        descriptive_uncertainty_envelope=(
            None if envelope is None else tuple(envelope)
        ),
        rank_position=record["rank_position"],
        tie_group=record["tie_group"],
        canonical_explanation=record["canonical_explanation"],
    )


def _p11_card(value: object, index: int) -> p11.TacticalPatternCard:
    record = _require_keys(value, _CARD_KEYS, f"tarjeta[{index}]")
    options_raw = record["options"]
    if type(options_raw) is not list:
        raise PublicContractError(f"tarjeta[{index}].options fuera de contrato.")
    options = tuple(
        _p11_option(item, number) for number, item in enumerate(options_raw)
    )
    by_category = {item.category: item for item in options}

    def selected(field: str) -> tuple[p11.TacticalRecommendationOption, ...]:
        values = record[field]
        if type(values) is not list or any(type(item) is not str for item in values):
            raise PublicContractError(f"tarjeta[{index}].{field} fuera de contrato.")
        try:
            return tuple(by_category[item] for item in values)
        except KeyError:
            raise PublicContractError(
                f"tarjeta[{index}].{field} fuera de contrato."
            ) from None

    reconciliations = record["reconciliations"]
    if type(reconciliations) is not dict:
        raise PublicContractError(
            f"tarjeta[{index}].reconciliations fuera de contrato."
        )
    return p11.TacticalPatternCard(
        pattern_id=record["pattern_id"],
        actor=record["actor"],
        tactical_opportunity=record["tactical_opportunity"],
        status=record["status"],
        status_reason_codes=tuple(record["status_reason_codes"]),
        categories=tuple(record["categories"]),
        options=options,
        ranked_options=selected("ranked_options"),
        top_options=selected("top_options"),
        abstained_options=selected("abstained_options"),
        requested_top_k=record["requested_top_k"],
        effective_top_k=record["effective_top_k"],
        tie_expanded=record["tie_expanded"],
        tie_group_count=record["tie_group_count"],
        total_options=record["total_options"],
        scored_options=record["scored_options"],
        abstained_options_count=record["abstained_options_count"],
        reconciliations=MappingProxyType(dict(reconciliations)),
    )


def _validate_exact_p11_payload(record: dict) -> None:
    cards_raw = record["cards"]
    if type(cards_raw) is not list:
        raise PublicContractError("cards debe ser una lista.")
    cards = tuple(_p11_card(item, index) for index, item in enumerate(cards_raw))
    reconciliations = record["reconciliations"]
    if type(reconciliations) is not dict:
        raise PublicContractError("respuesta.reconciliations fuera de contrato.")
    response = p11.TacticalRecommendationResponse(
        contract_version=record["contract_version"],
        status=record["status"],
        status_reason_codes=tuple(record["status_reason_codes"]),
        methodology=record["methodology"],
        combination=record["combination"],
        score_formula=record["score_formula"],
        uncertainty_method=record["uncertainty_method"],
        executor_weight=record["executor_weight"],
        opponent_weight=record["opponent_weight"],
        encoder_policy=record["encoder_policy"],
        evidence_scope=record["evidence_scope"],
        minimum_labeled_activations=record["minimum_labeled_activations"],
        minimum_distinct_matches=record["minimum_distinct_matches"],
        requested_top_k=record["requested_top_k"],
        ranking_scope=record["ranking_scope"],
        global_cross_pattern_ranking=record["global_cross_pattern_ranking"],
        cards=cards,
        limitations=tuple(record["limitations"]),
        reconciliations=MappingProxyType(dict(reconciliations)),
        fingerprint=record["fingerprint"],
    )
    p11.validate_public_tactical_recommendation(response)


def parse_public_recommendation(payload: object) -> PublicRecommendation:
    """Parse estricto del cuerpo 200 contra el contrato público P11."""
    record = _require_keys(payload, _RESPONSE_KEYS, "respuesta")
    try:
        _validate_exact_p11_payload(record)
    except PublicContractError:
        raise
    except Exception:
        raise PublicContractError(
            "respuesta fuera del contrato publico P11."
        ) from None
    status = record["status"]
    if status not in _RESPONSE_STATUSES:
        raise PublicContractError("respuesta.status fuera de contrato.")
    _require_all_true(
        record["reconciliations"],
        _RESPONSE_RECONCILIATION_KEYS,
        "respuesta.reconciliations",
    )
    cards_raw = record["cards"]
    if type(cards_raw) is not list:
        raise PublicContractError("cards debe ser una lista.")
    cards = tuple(_parse_card(item, number) for number, item in enumerate(cards_raw))
    return PublicRecommendation(
        status=status,
        status_reason_codes=_require_str_tuple(
            record["status_reason_codes"], "status_reason_codes"
        ),
        cards=cards,
        limitations=_require_str_tuple(record["limitations"], "limitations"),
    )


@dataclass(frozen=True)
class UIOutcome:
    """Resultado cerrado de una peticion UI (kind + mensaje publico)."""

    kind: str
    message: str
    model: PublicRecommendation | None = None
    status_code: int | None = None


def fetch_recommendation(
    client: object,
    base_url: str,
    player: str,
    opponent: str,
    as_of: str,
    *,
    container_mode: bool = False,
) -> UIOutcome:
    """UNICA peticion POST por llamada; cero reintentos, contrato cerrado.

    ``client`` ofrece el contexto ``stream`` de httpx. El limite se
    aplica incrementalmente a los bytes decodificados antes de parsear
    JSON. La URL, el cuerpo y el timeout exactos son parte del contrato;
    ningun error interno se propaga al mensaje publico. ``container_mode``
    (por defecto ``False``, comportamiento local identico al previo a
    P19) selecciona el validador de URL cerrado: en modo contenedor solo
    se acepta la constante fija del servicio interno, nunca una URL
    proveniente del usuario.
    """
    try:
        validated_base_url = (
            validate_container_api_base_url(base_url)
            if container_mode
            else validate_api_base_url(base_url)
        )
        validated_player = validate_local_identifier(player, "player")
        validated_opponent = validate_local_identifier(opponent, "opponent")
        validated_date = validate_local_date(as_of)
        if validated_player == validated_opponent:
            raise _IdentifierContractError("orientacion fuera de contrato")
    except (ValueError, _IdentifierContractError):
        return UIOutcome(
            _UI_LOCAL_OUTCOMES["incompatible_response"],
            _UI_LOCAL_MESSAGES["unexpected"],
        )
    url = f"{validated_base_url}{UI_RECOMMENDATIONS_PATH}"
    body = {
        "player_id": validated_player,
        "opponent_id": validated_opponent,
        "as_of_date": validated_date,
    }
    try:
        with client.stream(
            "POST",
            url,
            json=body,
            timeout=UI_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
        ) as response:
            status_code = response.status_code
            if type(status_code) is not int or isinstance(status_code, bool):
                return UIOutcome(
                    _UI_LOCAL_OUTCOMES["incompatible_response"],
                    _UI_LOCAL_MESSAGES["incompatible_response"],
                )
            if status_code != 200:
                message = _UI_STATUS_MESSAGES.get(
                    status_code, _UI_LOCAL_MESSAGES["unexpected"]
                )
                return UIOutcome(
                    _UI_LOCAL_OUTCOMES["status_error"],
                    message,
                    status_code=status_code,
                )
            content_length = response.headers.get("content-length")
            if content_length is not None:
                if not content_length.isascii() or not content_length.isdecimal():
                    return UIOutcome(
                        _UI_LOCAL_OUTCOMES["incompatible_response"],
                        _UI_LOCAL_MESSAGES["incompatible_response"],
                        status_code=status_code,
                    )
                if int(content_length) > UI_MAX_RESPONSE_BYTES:
                    return UIOutcome(
                        _UI_LOCAL_OUTCOMES["incompatible_response"],
                        _UI_LOCAL_MESSAGES["incompatible_response"],
                        status_code=status_code,
                    )
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_bytes():
                if type(chunk) is not bytes:
                    return UIOutcome(
                        _UI_LOCAL_OUTCOMES["incompatible_response"],
                        _UI_LOCAL_MESSAGES["incompatible_response"],
                        status_code=status_code,
                    )
                received += len(chunk)
                if received > UI_MAX_RESPONSE_BYTES:
                    return UIOutcome(
                        _UI_LOCAL_OUTCOMES["incompatible_response"],
                        _UI_LOCAL_MESSAGES["incompatible_response"],
                        status_code=status_code,
                    )
                chunks.append(chunk)
            content = b"".join(chunks)
    except httpx.TimeoutException:
        return UIOutcome(
            _UI_LOCAL_OUTCOMES["timeout"], _UI_LOCAL_MESSAGES["timeout"]
        )
    except httpx.HTTPError:
        return UIOutcome(
            _UI_LOCAL_OUTCOMES["connection"], _UI_LOCAL_MESSAGES["connection"]
        )
    except Exception:
        return UIOutcome(
            _UI_LOCAL_OUTCOMES["incompatible_response"],
            _UI_LOCAL_MESSAGES["unexpected"],
        )
    try:
        payload = json.loads(
            content, object_pairs_hook=_json_object_without_duplicate_keys
        )
    except (ValueError, TypeError, UnicodeDecodeError, AttributeError):
        return UIOutcome(
            _UI_LOCAL_OUTCOMES["incompatible_response"],
            _UI_LOCAL_MESSAGES["incompatible_response"],
            status_code=status_code,
        )
    try:
        model = parse_public_recommendation(payload)
    except PublicContractError:
        return UIOutcome(
            _UI_LOCAL_OUTCOMES["incompatible_response"],
            _UI_LOCAL_MESSAGES["incompatible_response"],
            status_code=status_code,
        )
    return UIOutcome(
        _UI_LOCAL_OUTCOMES["ok"], "", model=model, status_code=status_code
    )


def outcome_kind_label(outcome: UIOutcome) -> str:
    """Etiqueta cerrada en español del resultado de la peticion."""
    if outcome.kind == _UI_LOCAL_OUTCOMES["ok"]:
        return ""
    if outcome.kind == _UI_LOCAL_OUTCOMES["status_error"]:
        return f"(status {outcome.status_code})"
    return "(fallo local del cliente)"


_OPPORTUNITY_LABELS: Final = {
    "first_serve_direction": "Dirección del primer saque",
    "initial_return_direction": "Dirección del primer resto",
    "initial_return_depth": "Profundidad del primer resto",
    "initial_return_shot_type": "Tipo de golpe del primer resto",
}
_ACTOR_LABELS: Final = {"server": "Sacador", "returner": "Resto"}
_CARD_STATUS_LABELS: Final = {
    "available": "Disponible",
    "partially_available": "Parcialmente disponible",
    "not_available": "No disponible",
}
_OPTION_STATUS_LABELS: Final = {
    "ranked": "Clasificada",
    "abstained_insufficient_evidence": "Abstención (evidencia insuficiente)",
    "not_applicable": "No aplicable",
    "not_available": "No disponible",
}
_RESPONSE_STATUS_LABELS: Final = {
    "available": "Disponible",
    "partially_available": "Parcialmente disponible",
    "not_available": "Sin disponibilidad actual",
}


def _opportunity_label(pattern_id: str, tactical_opportunity: str) -> str:
    return _OPPORTUNITY_LABELS.get(
        tactical_opportunity, _OPPORTUNITY_LABELS.get(pattern_id, tactical_opportunity)
    )


def _evidence_table(evidence: PublicEvidenceComponent) -> None:
    st.markdown(
        "| Métrica | Valor |\n"
        "|---|---|\n"
        f"| Activaciones etiquetadas | {evidence.labeled_activations} |\n"
        f"| Éxitos / fallos | {evidence.successes} / {evidence.failures} |\n"
        f"| Partidos distintos | {evidence.distinct_matches} |"
    )
    if evidence.success_rate is not None:
        lower = (
            f"{evidence.wilson_lower:.3f}"
            if evidence.wilson_lower is not None
            else "—"
        )
        upper = (
            f"{evidence.wilson_upper:.3f}"
            if evidence.wilson_upper is not None
            else "—"
        )
        st.markdown(
            f"Tasa de éxito: **{evidence.success_rate:.3f}** "
            f"(intervalo descriptivo {lower}–{upper}); estado de la "
            f"evidencia: {evidence.evidence_state}."
        )


def _render_card(card: PublicPatternCard) -> None:
    opportunity = _opportunity_label(card.pattern_id, card.tactical_opportunity)
    actor = _ACTOR_LABELS.get(card.actor, card.actor)
    st.markdown(
        f"### {card.pattern_id} · {opportunity} ({actor}) — "
        f"{_CARD_STATUS_LABELS.get(card.status, card.status)}"
    )
    if card.status == "not_available":
        st.write(
            "Sin opciones tácticas para este patrón en esta orientación "
            "(el sistema abstiene; no se inventan recomendaciones)."
        )
        return
    ranked = sorted(
        (option for option in card.options if option.status == "ranked"),
        key=lambda option: option.rank_position or 0,
    )
    if ranked:
        rows = [
            "| Puesto | Categoría | Estado | Puntuación | Intervalo |"
            "|---|---|---|---|---|"
        ]
        for option in ranked:
            score = (
                f"{option.score:.3f}" if option.score is not None else "—"
            )
            if option.descriptive_uncertainty_envelope is not None:
                envelope = (
                    f"{option.descriptive_uncertainty_envelope[0]:.3f}–"
                    f"{option.descriptive_uncertainty_envelope[1]:.3f}"
                )
            else:
                envelope = "—"
            rows.append(
                f"| {option.rank_position} | {option.category} | "
                f"{_OPTION_STATUS_LABELS.get(option.status, option.status)} | "
                f"{score} | {envelope} |"
            )
        st.markdown("\n".join(rows))
    else:
        st.write(
            "Sin opciones clasificadas para este patrón en esta "
            "orientación (evidencia insuficiente: abstención del sistema)."
        )
    if card.top_options:
        st.write(f"**Opciones destacadas (top_k):** {', '.join(card.top_options)}")
    abstained = [
        option for option in card.options
        if option.status == "abstained_insufficient_evidence"
    ]
    if abstained:
        with st.expander(
            f"Opciones con evidencia insuficiente ({len(abstained)})"
        ):
            for option in abstained:
                st.write(f"- {option.category}: abstención del sistema.")
    with st.expander("Evidencia y explicaciones (contrato público)"):
        for option in ranked:
            if option.canonical_explanation:
                st.write(f"**{option.category}** — {option.canonical_explanation}")
            st.caption("Perspectiva ejecutora:")
            _evidence_table(option.executor_evidence)
            st.caption("Perspectiva rival (permitida):")
            _evidence_table(option.opponent_allowed_evidence)


def render_public_recommendation(model: PublicRecommendation) -> None:
    """Render legible y cerrado de la ficha pública P11 (sin PII propio)."""
    st.success(
        f"Estado de la ficha: {_RESPONSE_STATUS_LABELS.get(model.status, model.status)}"
    )
    if model.status_reason_codes:
        st.caption(
            "Motivos del estado: " + ", ".join(model.status_reason_codes)
        )
    if model.status == "not_available":
        st.warning(
            "Esta orientación no tiene opciones tácticas disponibles "
            "ahora. El sistema abstiene: no se muestran recomendaciones "
            "inventadas."
        )
        return
    for card in model.cards:
        _render_card(card)
    if model.limitations:
        with st.expander("Limitaciones del método (público)"):
            for limitation in model.limitations:
                st.write(f"- {limitation}")
    st.info(
        "Las recomendaciones son evidencia histórica observacional, no "
        "causalidad ni garantía de éxito."
    )


def form_inputs() -> tuple[str, str, date, str]:
    """Formulario compartido jugador/rival/fecha (reutilizable por P19)."""
    st.subheader("Consultar una orientación")
    with st.form("recommendation_form", clear_on_submit=True):
        player = st.text_input(
            "Jugador",
            key="recommendation_player",
            max_chars=UI_MAX_IDENTIFIER_LENGTH,
            help="1-64 caracteres; sin espacios al final, sin '/','\\','://','..','~' ni caracteres de control.",
        )
        opponent = st.text_input(
            "Rival",
            key="recommendation_opponent",
            max_chars=UI_MAX_IDENTIFIER_LENGTH,
            help="Mismas reglas que jugador; debe ser diferente del jugador.",
        )
        chosen_date = st.date_input(
            "Fecha (as of)", value=date.today(), key="recommendation_as_of_date"
        )
        submitted = st.form_submit_button(
            "Solicitar recomendación", key="recommendation_submit"
        )
    return player, opponent, chosen_date, "" if not submitted else "sent"


def main() -> None:
    # HTTPX/httpcore registran la URL a nivel INFO/DEBUG. Esta UI local
    # desactiva ambos canales antes de construir el cliente para que el
    # puerto configurado no se propague al logging anfitrion.
    logging.getLogger("httpx").disabled = True
    logging.getLogger("httpcore").disabled = True
    st.set_page_config(
        page_title="Recomendador táctico (local)",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Recomendador táctico de tenis (evidencia histórica)")
    st.caption(
        "Interfaz local sobre la API P17. Las fichas son evidencia "
        "histórica observacional: no afirman causalidad ni garantizan "
        "éxito. Todo el tráfico es loopback local."
    )
    with st.expander("Configuración del servicio local"):
        raw_url = st.text_input(
            "URL del servicio local",
            value=UI_API_BASE_DEFAULT,
            key="recommendation_api_base_url",
            help="Solo http://127.0.0.1[puerto] (1-65535), sin credenciales, path, query ni fragmentos.",
        )
    try:
        base_url = validate_api_base_url(raw_url)
    except ValueError:
        base_url = None
    player_raw, opponent_raw, chosen_date, submitted = form_inputs()
    if not submitted:
        st.caption(
            "Introduce jugador, rival y fecha y pulsa "
            "«Solicitar recomendación»."
        )
        return
    if base_url is None:
        st.error("La URL del servicio no es una URL loopback segura válida.")
        return
    try:
        player = validate_local_identifier(player_raw, "player")
        opponent = validate_local_identifier(opponent_raw, "opponent")
    except _IdentifierContractError as error:
        st.error(str(error))
        return
    if player == opponent:
        st.error("Jugador y rival deben ser diferentes.")
        return
    if isinstance(chosen_date, datetime):
        as_of_raw = chosen_date.date().isoformat()
    elif isinstance(chosen_date, date):
        as_of_raw = chosen_date.isoformat()
    else:
        as_of_raw = ""
    try:
        as_of = validate_local_date(as_of_raw)
    except _IdentifierContractError as error:
        st.error(str(error))
        return
    client = httpx.Client(
        timeout=UI_REQUEST_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
    )
    try:
        with st.spinner("Consultando el servicio local…"):
            outcome = fetch_recommendation(
                client, base_url, player, opponent, as_of
            )
    finally:
        client.close()
    if outcome.kind == _UI_LOCAL_OUTCOMES["ok"] and outcome.model is not None:
        render_public_recommendation(outcome.model)
        return
    label = outcome_kind_label(outcome)
    st.error(f"{outcome.message} {label}".strip())


if __name__ == "__main__":
    main()


__all__ = (
    "UI_API_BASE_DEFAULT",
    "UI_API_HOST_ALLOWED",
    "UI_CONTAINER_API_BASE_URL",
    "UI_MAX_IDENTIFIER_LENGTH",
    "UI_MAX_PORT",
    "UI_MAX_RESPONSE_BYTES",
    "UI_MIN_PORT",
    "UI_RECOMMENDATIONS_PATH",
    "UI_REQUEST_TIMEOUT_SECONDS",
    "PublicContractError",
    "PublicEvidenceComponent",
    "PublicOption",
    "PublicPatternCard",
    "PublicRecommendation",
    "UIOutcome",
    "fetch_recommendation",
    "form_inputs",
    "main",
    "outcome_kind_label",
    "parse_public_recommendation",
    "render_public_recommendation",
    "validate_api_base_url",
    "validate_container_api_base_url",
    "validate_local_date",
    "validate_local_identifier",
)
