"""Servicio puro de recomendaciones tacticas publicas P12.

Capa de aplicacion que transforma una orientacion jugador-rival
(``TacticalRecommendationQuery``) en la respuesta publica P11
(``TacticalRecommendationResponse``) mediante un provider inyectable
(``TacticalRecommendationProvider``).

Garantias contractuales:

- No depende de FastAPI ni de ningun framework HTTP.
- No lee datos, no escribe disco, no usa red y no ejecuta el pipeline P10.
- No fabrica respuestas: solo devuelve el objeto P11 proyectado por
  ``build_public_tactical_recommendation`` para un resultado upstream
  reconstruido y validado.
- Los identificadores (player, opponent) y la fecha de corte existen solo
  en la query y en la frontera interna del provider; nunca salen en la
  respuesta publica ni en los errores.
- Sin estado global mutable: las constantes son inmutables y el servicio
  solo retiene el provider inyectado.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
import re
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from src.recommender.tactical_prioritization import (
    TacticalPrioritizationResult,
    validate_tactical_prioritization_result,
)
from src.recommender.tactical_recommendation_contract import (
    TacticalRecommendationResponse,
    build_public_tactical_recommendation,
    validate_public_tactical_recommendation,
)


SERVICE_CONTRACT_VERSION: Final = "1.0.0"
SERVICE_API_VERSION: Final = "v1"
SERVICE_NAME: Final = "tactical-recommendation-api"
MAX_IDENTIFIER_LENGTH: Final = 64

SERVICE_REASON_CODES: Final = frozenset(
    (
        "invalid_request",
        "recommendation_not_found",
        "recommendation_unavailable",
        "provider_timeout",
        "provider_unavailable",
        "upstream_contract_violation",
        "internal_error",
    )
)
SERVICE_STAGES: Final = frozenset(
    (
        "request_validation",
        "provider_lookup",
        "result_validation",
        "public_projection",
        "internal",
    )
)
SERVICE_LOG_EVENTS: Final = frozenset(
    (
        "healthz",
        "recommendation_requested",
        "recommendation_completed",
        "recommendation_rejected",
    )
)
SERVICE_LOG_METHODS: Final = frozenset(("GET", "POST"))
SERVICE_LOG_ENDPOINTS: Final = frozenset(
    ("/healthz", "/api/v1/recommendations")
)
SERVICE_LOG_STATUS_CODES: Final = frozenset(
    (200, 404, 422, 500, 502, 503, 504)
)
_REQUEST_ID_PATTERN: Final = re.compile(r"[0-9a-f]{8,64}")

_QUERY_CONTROL_CHARACTERS: Final = frozenset(
    tuple(range(0x00, 0x20)) + (0x7F,) + tuple(range(0x80, 0xA0))
)
_QUERY_FORBIDDEN_FRAGMENTS: Final = ("/", "\\", "://", "..", "~")


class TacticalRecommendationServiceError(Exception):
    """Fallo cerrado del servicio de recomendaciones tacticas P12.

    Cada instancia conserva exclusivamente un reason_code estable, un
    stage cerrado, un retryable boolean real y un mensaje publico
    sanitizado constante. No conserva query, IDs, fechas, rutas,
    secuencias, excepciones, stack traces, repr de objetos ni
    fingerprints upstream.
    """

    reason_code: Final[str]
    stage: Final[str]
    retryable: Final[bool]
    public_message: Final[str]

    def __init__(self) -> None:
        super().__init__()


class InvalidRequestError(TacticalRecommendationServiceError):
    """La query no cumple el contrato de entrada cerrado."""

    reason_code = "invalid_request"
    stage = "request_validation"
    retryable = False
    public_message = "La request no cumple el contrato de entrada cerrado del servicio."


class RecommendationNotFoundError(TacticalRecommendationServiceError):
    """El provider declara que la orientacion no tiene ficha publica."""

    reason_code = "recommendation_not_found"
    stage = "provider_lookup"
    retryable = False
    public_message = "La orientacion solicitada no tiene ficha publica disponible."


class RecommendationUnavailableError(TacticalRecommendationServiceError):
    """El provider declara que la ficha existe pero no esta disponible ahora."""

    reason_code = "recommendation_unavailable"
    stage = "provider_lookup"
    retryable = True
    public_message = "La ficha publica existe pero no esta disponible en este momento."


class ProviderTimeoutError(TacticalRecommendationServiceError):
    """El provider declara un timeout contractual de consulta.

    La imposicion y cancelacion fisica del timeout pertenecen al
    adaptador real futuro; aqui solo se traduce el contrato declarado.
    """

    reason_code = "provider_timeout"
    stage = "provider_lookup"
    retryable = True
    public_message = "La consulta al backend excedio el tiempo contractual."


class ProviderUnavailableError(TacticalRecommendationServiceError):
    """El provider declara que el backend no esta disponible."""

    reason_code = "provider_unavailable"
    stage = "provider_lookup"
    retryable = True
    public_message = "El backend de recomendaciones no esta disponible."


class UpstreamContractViolationError(TacticalRecommendationServiceError):
    """El resultado upstream viola el contrato P10/P11 esperado."""

    reason_code = "upstream_contract_violation"
    stage = "result_validation"
    retryable = False
    public_message = "El backend devolvio un resultado fuera de contrato."


class InternalServiceError(TacticalRecommendationServiceError):
    """Fallo inesperado sanitizado: no conserva el origen."""

    reason_code = "internal_error"
    stage = "internal"
    retryable = False
    public_message = "Fallo interno sin clasificar del servicio."


_SERVICE_ERROR_CATALOG: Final = MappingProxyType(
    {
        "invalid_request": InvalidRequestError,
        "recommendation_not_found": RecommendationNotFoundError,
        "recommendation_unavailable": RecommendationUnavailableError,
        "provider_timeout": ProviderTimeoutError,
        "provider_unavailable": ProviderUnavailableError,
        "upstream_contract_violation": UpstreamContractViolationError,
        "internal_error": InternalServiceError,
    }
)


def _service_integrity_check() -> None:
    expected_stages = {
        "invalid_request": "request_validation",
        "recommendation_not_found": "provider_lookup",
        "recommendation_unavailable": "provider_lookup",
        "provider_timeout": "provider_lookup",
        "provider_unavailable": "provider_lookup",
        "upstream_contract_violation": "result_validation",
        "internal_error": "internal",
    }
    if set(_SERVICE_ERROR_CATALOG) != SERVICE_REASON_CODES:
        raise RuntimeError("El catalogo de errores no coincide con los reason codes.")
    for reason_code, error_cls in _SERVICE_ERROR_CATALOG.items():
        if error_cls.stage != expected_stages[reason_code]:
            raise RuntimeError("Stage del error fuera del mapeo cerrado.")
        if type(error_cls.public_message) is not str or not error_cls.public_message:
            raise RuntimeError("Mensaje publico del error invalido.")
        if type(error_cls.retryable) is not bool:
            raise RuntimeError("Retryable del error debe ser bool real.")


_service_integrity_check()


@dataclass(frozen=True)
class TacticalRecommendationQuery:
    """Orientacion jugador-rival para una fecha de corte civil.

    El cliente no puede configurar pesos, thresholds, scopes, politicas,
    patrones ni top_k: la politica queda congelada upstream en P11.
    """

    player_id: str
    opponent_id: str
    as_of_date: date

    def __post_init__(self) -> None:
        validate_tactical_recommendation_query(self)


def _strict_identifier(value: object, field_name: str) -> None:
    if type(value) is not str:
        raise InvalidRequestError()
    if not value or value != value.strip() or len(value) > MAX_IDENTIFIER_LENGTH:
        raise InvalidRequestError()
    if any(ord(character) in _QUERY_CONTROL_CHARACTERS for character in value):
        raise InvalidRequestError()
    if any(fragment in value for fragment in _QUERY_FORBIDDEN_FRAGMENTS):
        raise InvalidRequestError()


def validate_tactical_recommendation_query(query: object) -> None:
    """Valida exhaustivamente una query del contrato cerrado de P12."""
    if type(query) is not TacticalRecommendationQuery:
        raise InvalidRequestError()
    for field_name in ("player_id", "opponent_id"):
        _strict_identifier(getattr(query, field_name), field_name)
    if query.player_id == query.opponent_id:
        raise InvalidRequestError()
    as_of = query.as_of_date
    if type(as_of) is not date:
        raise InvalidRequestError()
    if len(fields(TacticalRecommendationQuery)) != 3:
        raise RuntimeError("La query contractual gano campos configurables.")


def redacted_service_event(
    event: str,
    *,
    request_id: str | None = None,
    method: str | None = None,
    endpoint: str | None = None,
    status_code: int | None = None,
    stage: str | None = None,
    reason_code: str | None = None,
) -> MappingProxyType[str, object]:
    """Evento de log estructurado sin PII.

    Solo admite campos de dominios cerrados y constantes. ``request_id``
    debe ser una cadena hexadecimal opaca (generada por la capa HTTP) y
    no identifica a jugador, rival, partido ni fecha. Jamas acepta query,
    IDs, fechas, rutas de fichero, cuerpos de request ni cuerpos de
    response.
    """
    if type(event) is not str or event not in SERVICE_LOG_EVENTS:
        raise ValueError("Evento de log fuera del catalogo cerrado.")
    payload: dict[str, object] = {"event": event}
    if request_id is not None:
        if type(request_id) is not str or _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
            raise ValueError("Request id fuera del dominio opaco hexadecimal.")
        payload["request_id"] = request_id
    if method is not None:
        if type(method) is not str or method not in SERVICE_LOG_METHODS:
            raise ValueError("Metodo de log fuera del catalogo cerrado.")
        payload["method"] = method
    if endpoint is not None:
        if type(endpoint) is not str or endpoint not in SERVICE_LOG_ENDPOINTS:
            raise ValueError("Endpoint de log fuera del catalogo cerrado.")
        payload["endpoint"] = endpoint
    if status_code is not None:
        if type(status_code) is not int or status_code not in SERVICE_LOG_STATUS_CODES:
            raise ValueError("Status de log fuera del catalogo cerrado.")
        payload["status_code"] = status_code
    if stage is not None:
        if type(stage) is not str or stage not in SERVICE_STAGES:
            raise ValueError("Stage de log fuera del catalogo cerrado.")
        payload["stage"] = stage
    if reason_code is not None:
        if type(reason_code) is not str or reason_code not in SERVICE_REASON_CODES:
            raise ValueError("Reason code de log fuera del catalogo cerrado.")
        payload["reason_code"] = reason_code
    return MappingProxyType(payload)


@runtime_checkable
class TacticalRecommendationProvider(Protocol):
    """Backend inyectable de la consulta de recomendaciones.

    Debe aceptar una query validada y devolver exclusivamente un
    ``TacticalPrioritizationResult``; puede declarar los errores cerrados
    del servicio. No depende de FastAPI, no ejecuta P10 implicitamente y
    no debe mantener estado global mutable.
    """

    def fetch_tactical_prioritization(
        self, query: TacticalRecommendationQuery
    ) -> TacticalPrioritizationResult:
        """Devuelve el resultado priorizado para la orientacion de query."""


class TacticalRecommendationService:
    """Servicio de aplicacion puro: query -> provider -> P11 publico.

    Flujo:
    1. ``request_validation``: validacion cerrada de la query.
    2. ``provider_lookup``: una unica invocacion al provider.
    3. ``result_validation``: reconstruccion del resultado upstream y
       verificacion de la frontera de identidades.
    4. ``public_projection``: una unica llamada a
       ``build_public_tactical_recommendation`` y validacion publico
       reconstructiva con el source.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: object) -> None:
        fetch = getattr(provider, "fetch_tactical_prioritization", None)
        if not callable(fetch):
            raise TypeError(
                "provider debe implementar fetch_tactical_prioritization(query)."
            )
        object.__setattr__(self, "_provider", provider)

    @property
    def provider(self) -> TacticalRecommendationProvider:
        return self._provider

    def recommend(
        self, query: TacticalRecommendationQuery
    ) -> TacticalRecommendationResponse:
        """Produce la respuesta publica P11 para una query validada."""
        if type(query) is not TacticalRecommendationQuery:
            raise InvalidRequestError()
        validate_tactical_recommendation_query(query)

        try:
            result = self._provider.fetch_tactical_prioritization(query)
        except TacticalRecommendationServiceError:
            raise
        except Exception:
            raise InternalServiceError() from None

        if type(result) is not TacticalPrioritizationResult:
            raise UpstreamContractViolationError()
        try:
            validate_tactical_prioritization_result(result)
            summary = result.matchup_query
            if (
                summary.player != query.player_id
                or summary.opponent != query.opponent_id
                or summary.as_of_date != query.as_of_date.isoformat()
            ):
                raise UpstreamContractViolationError()
            response = build_public_tactical_recommendation(result)
            validate_public_tactical_recommendation(response, source=result)
        except TacticalRecommendationServiceError:
            raise
        except Exception:
            raise UpstreamContractViolationError() from None
        return response
