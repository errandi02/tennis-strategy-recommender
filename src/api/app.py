"""API HTTP publica P12 de recomendaciones tacticas sobre FastAPI.

Capa de presentacion sobre el servicio puro P12: mapea
``GET /healthz`` y ``POST /api/v1/recommendations`` a
``TacticalRecommendationService`` con provider inyectado. No lee datos,
no usa disco, no ejecuta el pipeline P10, no lleva estado global
mutable y no instancia proveedor real al importar el modulo.

Contrato documentado:

- ``POST /api/v1/recommendations`` acepta exactamente ``player_id``,
  ``opponent_id`` y ``as_of_date`` (fecha civil ISO estricta
  ``YYYY-MM-DD``); rechaza campos extra, JSON invalido e identidades
  fuera del dominio cerrado.
- El exito devuelve el cuerpo canonico P11 exacto
  (``canonical_tactical_recommendation_json``) con ``ETag`` fuerte
  ``"<FINGERPRINT PUBLICA P11>"`` (determinista) y ``X-Request-ID``
  hexadecimal opaco generado por request.
- Los errores del provider se traducen al mapa cerrado de status;
  ``ProviderTimeoutError`` se expresa como 504 (la imposicion fisica del
  timeout pertenece al adaptador real futuro).
- Todos los errores del contrato usan el mismo sobre cerrado
  ``{"error": {reason_code, stage, message, retryable}}``.
- Los logs usan solo ``redacted_service_event`` (dominios cerrados):
  jamas contienen player_id, opponent_id, as_of_date, cuerpo de request
  ni cuerpo de response.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import fields
from datetime import date
from types import MappingProxyType
from typing import Final, Literal

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.recommender.tactical_recommendation_contract import (
    CARD_RECONCILIATIONS,
    RESPONSE_RECONCILIATIONS,
    PublicEvidenceComponent,
    TacticalPatternCard,
    TacticalRecommendationOption,
    TacticalRecommendationResponse,
    canonical_tactical_recommendation_json,
    tactical_recommendation_fingerprint,
)
from src.recommender.tactical_recommendation_service import (
    MAX_IDENTIFIER_LENGTH,
    SERVICE_API_VERSION,
    SERVICE_LOG_ENDPOINTS,
    SERVICE_LOG_METHODS,
    SERVICE_NAME,
    InternalServiceError,
    InvalidRequestError,
    TacticalRecommendationQuery,
    TacticalRecommendationService,
    TacticalRecommendationServiceError,
    redacted_service_event,
)


HEALTH_PATH: Final = "/healthz"
RECOMMENDATIONS_PATH: Final = "/api/v1/recommendations"
HEALTH_OPERATION_ID: Final = "tactical_recommendation_health"
RECOMMENDATIONS_OPERATION_ID: Final = "tactical_recommendation_post"
API_LOGGER_NAME: Final = "tactical_recommendation_api"

_CIVIL_ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")

_ERROR_HTTP_STATUS: Final = MappingProxyType(
    {
        "invalid_request": 422,
        "recommendation_not_found": 404,
        "recommendation_unavailable": 503,
        "provider_timeout": 504,
        "provider_unavailable": 503,
        "upstream_contract_violation": 502,
        "internal_error": 500,
    }
)

_LOGGER = logging.getLogger(API_LOGGER_NAME)


ReasonCode = Literal[
    "invalid_request",
    "recommendation_not_found",
    "recommendation_unavailable",
    "provider_timeout",
    "provider_unavailable",
    "upstream_contract_violation",
    "internal_error",
]
Stage = Literal[
    "request_validation",
    "provider_lookup",
    "result_validation",
    "public_projection",
    "internal",
]


class PublicError(BaseModel):
    """Sobre de error cerrado; el mensaje es la constante del catalogo."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason_code: ReasonCode
    stage: Stage
    message: str
    retryable: bool


class PublicErrorEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error: PublicError


class TacticalRecommendationRequest(BaseModel):
    """Request estricta: solo las tres nociones del contrato P12."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    player_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    opponent_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    as_of_date: date

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _strict_civil_iso_date(cls, raw: object) -> date:
        if type(raw) is str:
            if raw != raw.strip() or _CIVIL_ISO_DATE.fullmatch(raw) is None:
                raise ValueError(
                    "as_of_date debe ser una fecha civil ISO estricta."
                )
            return date.fromisoformat(raw)
        raise ValueError("as_of_date debe ser una fecha civil ISO estricta.")


class PublicEvidenceComponentSchema(BaseModel):
    """Schema HTTP cerrado del componente de evidencia publico P11."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    perspective: Literal["executor", "opponent_allowed"]
    scope: Literal["global"]
    evidence_state: Literal[
        "available",
        "insufficient_labeled_attempts",
        "insufficient_matches",
        "no_observed_category",
        "not_applicable",
        "not_available",
    ]
    labeled_activations: int = Field(ge=0)
    successes: int = Field(ge=0)
    failures: int = Field(ge=0)
    distinct_matches: int = Field(ge=0)
    success_rate: float | None = Field(ge=0.0, le=1.0)
    wilson_lower: float | None = Field(ge=0.0, le=1.0)
    wilson_upper: float | None = Field(ge=0.0, le=1.0)


class TacticalRecommendationOptionSchema(BaseModel):
    """Schema HTTP cerrado de una opcion tactica publica P11."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern_id: Literal["P02", "P04", "P05", "P06"]
    category: str
    tactical_opportunity: Literal[
        "first_serve_direction",
        "initial_return_direction",
        "initial_return_depth",
        "initial_return_shot_type",
    ]
    actor: Literal["server", "returner"]
    status: Literal[
        "ranked",
        "abstained_insufficient_evidence",
        "not_applicable",
        "not_available",
    ]
    reason_codes: tuple[str, ...]
    executor_evidence: PublicEvidenceComponentSchema
    opponent_allowed_evidence: PublicEvidenceComponentSchema
    score_formula: Literal[
        "combined_rate = 0.5 * executor_success_rate + 0.5 * opponent_allowed_success_rate"
    ]
    score: float | None = Field(ge=0.0, le=1.0)
    descriptive_uncertainty_envelope: tuple[float, float] | None
    rank_position: int | None = Field(ge=1)
    tie_group: int | None = Field(ge=1)
    canonical_explanation: str


class CardReconciliationsSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    catalog_complete: Literal[True]
    options_in_catalog_order: Literal[True]
    ranked_and_abstained_partition: Literal[True]
    rank_positions_reconciled: Literal[True]
    tie_expansions_preserved: Literal[True]
    counts_reconciled: Literal[True]


class TacticalPatternCardSchema(BaseModel):
    """Schema HTTP cerrado de una tarjeta tactica publica P11."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern_id: Literal["P02", "P04", "P05", "P06"]
    actor: Literal["server", "returner"]
    tactical_opportunity: Literal[
        "first_serve_direction",
        "initial_return_direction",
        "initial_return_depth",
        "initial_return_shot_type",
    ]
    status: Literal["available", "partially_available", "not_available"]
    status_reason_codes: tuple[str, ...]
    categories: tuple[str, ...]
    options: tuple[TacticalRecommendationOptionSchema, ...]
    # El JSON canonico P11 compacta estas tres particiones a categorias;
    # los objetos completos permanecen una sola vez en ``options``.
    ranked_options: tuple[str, ...]
    top_options: tuple[str, ...]
    abstained_options: tuple[str, ...]
    requested_top_k: int = Field(ge=1)
    effective_top_k: int = Field(ge=0)
    tie_expanded: bool
    tie_group_count: int = Field(ge=0)
    total_options: int = Field(ge=0)
    scored_options: int = Field(ge=0)
    abstained_options_count: int = Field(ge=0)
    reconciliations: CardReconciliationsSchema


class ResponseReconciliationsSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_frozen: Literal[True]
    four_scoreable_patterns_ordered: Literal[True]
    no_cross_pattern_ranking: Literal[True]
    status_reconciled: Literal[True]
    identity_and_upstream_fingerprints_redacted: Literal[True]
    descriptive_not_causal: Literal[True]


class TacticalRecommendationResponseSchema(BaseModel):
    """Representacion OpenAPI fiel y cerrada de TacticalRecommendationResponse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: Literal["1.0.0"]
    status: Literal["available", "partially_available", "not_available"]
    status_reason_codes: tuple[str, ...]
    methodology: Literal["descriptive_observational"]
    combination: Literal["equal_weight_executor_opponent"]
    score_formula: Literal[
        "combined_rate = 0.5 * executor_success_rate + 0.5 * opponent_allowed_success_rate"
    ]
    uncertainty_method: Literal[
        "descriptive_uncertainty_envelope_from_two_wilson_intervals"
    ]
    executor_weight: Literal[0.5]
    opponent_weight: Literal[0.5]
    encoder_policy: Literal["component_only"]
    evidence_scope: Literal["global_only"]
    minimum_labeled_activations: Literal[50]
    minimum_distinct_matches: Literal[5]
    requested_top_k: Literal[3]
    ranking_scope: Literal["independent_within_pattern"]
    global_cross_pattern_ranking: Literal[False]
    cards: tuple[TacticalPatternCardSchema, ...]
    limitations: tuple[str, ...]
    reconciliations: ResponseReconciliationsSchema
    fingerprint: str = Field(pattern=r"^[0-9A-F]{64}$")


class HealthResponseSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_version: Literal["v1"]
    service: Literal["tactical-recommendation-api"]
    status: Literal["ok"]


def _schema_integrity_check() -> None:
    """Impide que los schemas documentales diverjan de las dataclasses P11."""
    pairs = (
        (PublicEvidenceComponentSchema, PublicEvidenceComponent),
        (TacticalRecommendationOptionSchema, TacticalRecommendationOption),
        (TacticalPatternCardSchema, TacticalPatternCard),
        (TacticalRecommendationResponseSchema, TacticalRecommendationResponse),
    )
    for schema_model, contract_type in pairs:
        if tuple(schema_model.model_fields) != tuple(
            item.name for item in fields(contract_type)
        ):
            raise RuntimeError("Schema HTTP divergente del contrato publico P11.")
    if tuple(CardReconciliationsSchema.model_fields) != tuple(CARD_RECONCILIATIONS):
        raise RuntimeError("Schema HTTP de reconciliaciones de tarjeta divergente.")
    if tuple(ResponseReconciliationsSchema.model_fields) != tuple(RESPONSE_RECONCILIATIONS):
        raise RuntimeError("Schema HTTP de reconciliaciones de respuesta divergente.")


_schema_integrity_check()


def _json_bytes(body: dict[str, object]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _request_id(request: Request) -> str:
    """Genera una sola ID opaca por request y la comparte con sus handlers."""
    value = getattr(request.state, "p12_request_id", None)
    if value is None:
        value = uuid.uuid4().hex
        request.state.p12_request_id = value
    return value


def _error_response(
    request_id: str,
    reason_code: str,
    stage: str,
    message: str,
    retryable: bool,
    status_code: int,
) -> Response:
    body: dict[str, object] = {
        "error": {
            "reason_code": reason_code,
            "stage": stage,
            "message": message,
            "retryable": retryable,
        }
    }
    return Response(
        content=_json_bytes(body),
        status_code=status_code,
        media_type="application/json",
        headers={"X-Request-ID": request_id},
    )


def _log_service_event(
    request: Request,
    status_code: int,
    *,
    event: str,
    request_id: str,
    stage: str | None = None,
    reason_code: str | None = None,
) -> None:
    values: dict[str, object] = {
        "request_id": request_id,
        "status_code": status_code,
    }
    if request.method in SERVICE_LOG_METHODS:
        values["method"] = request.method
    if request.url.path in SERVICE_LOG_ENDPOINTS:
        values["endpoint"] = request.url.path
    if stage is not None:
        values["stage"] = stage
    if reason_code is not None:
        values["reason_code"] = reason_code
    try:
        record = redacted_service_event(event, **values)
        _LOGGER.info("service_event %s", json.dumps(dict(record), sort_keys=True))
    except Exception:
        _LOGGER.info("service_event %s status_code=%d", event, status_code)


def create_app(provider: object) -> FastAPI:
    """Crea la instancia HTTP con provider inyectado (sin I/O al importar)."""
    service = TacticalRecommendationService(provider)
    app = FastAPI(
        title=f"{SERVICE_NAME} http",
        version=SERVICE_API_VERSION,
        openapi_version="3.1.0",
    )

    def _envelope(description: str) -> dict[str, object]:
        return {"model": PublicErrorEnvelope, "description": description}

    recommendation_responses: dict[int, dict[str, object]] = {
        200: {
            "description": "Ficha publica P11 en JSON canonico byte-determinista.",
            "model": TacticalRecommendationResponseSchema,
            "headers": {
                "ETag": {
                    "description": (
                        'Strong entity tag: "<FINGERPRINT PUBLICA P11>".'
                    ),
                    "schema": {"type": "string"},
                },
                "X-Request-ID": {
                    "description": (
                        "Identificador hexadecimal opaco generado por request."
                    ),
                    "schema": {"type": "string"},
                },
            },
        },
        404: _envelope(
            "Orientacion sin ficha publica (recommendation_not_found)."
        ),
        422: _envelope("Request fuera del contrato cerrado (invalid_request)."),
        500: _envelope("Fallo interno sanitizado (internal_error)."),
        502: _envelope(
            "Resultado upstream fuera de contrato (upstream_contract_violation)."
        ),
        503: _envelope(
            "Ficha no disponible (recommendation_unavailable | provider_unavailable)."
        ),
        504: _envelope("Timeout contractual del provider (provider_timeout)."),
    }

    @app.exception_handler(TacticalRecommendationServiceError)
    async def _handle_service_error(
        request: Request, exc: TacticalRecommendationServiceError
    ) -> Response:
        status_code = _ERROR_HTTP_STATUS[exc.reason_code]
        request_id = _request_id(request)
        _log_service_event(
            request,
            status_code,
            event="recommendation_rejected",
            request_id=request_id,
            stage=exc.stage,
            reason_code=exc.reason_code,
        )
        return _error_response(
            request_id,
            exc.reason_code,
            exc.stage,
            exc.public_message,
            exc.retryable,
            status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> Response:
        error = InvalidRequestError()
        request_id = _request_id(request)
        _log_service_event(
            request,
            422,
            event="recommendation_rejected",
            request_id=request_id,
            stage="request_validation",
            reason_code="invalid_request",
        )
        return _error_response(
            request_id,
            error.reason_code,
            error.stage,
            error.public_message,
            error.retryable,
            422,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> Response:
        error = InternalServiceError()
        request_id = _request_id(request)
        _log_service_event(
            request,
            500,
            event="recommendation_rejected",
            request_id=request_id,
            stage="internal",
            reason_code="internal_error",
        )
        return _error_response(
            request_id,
            error.reason_code,
            error.stage,
            error.public_message,
            error.retryable,
            500,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_route_outside_contract(
        request: Request, exc: StarletteHTTPException
    ) -> Response:
        # 404/405 de rutas no publicadas quedan fuera del catalogo semantico
        # del POST, pero tampoco exponen el sobre ``detail`` del framework.
        return Response(
            status_code=exc.status_code,
            headers={"X-Request-ID": _request_id(request)},
        )

    @app.get(
        HEALTH_PATH,
        operation_id=HEALTH_OPERATION_ID,
        summary="Vitalidad del proceso (no consulta al provider).",
        response_model=HealthResponseSchema,
    )
    async def healthz(request: Request) -> Response:
        request_id = _request_id(request)
        _log_service_event(request, 200, event="healthz", request_id=request_id)
        body = {
            "api_version": SERVICE_API_VERSION,
            "service": SERVICE_NAME,
            "status": "ok",
        }
        return Response(
            content=_json_bytes(body),
            media_type="application/json",
            headers={"X-Request-ID": request_id},
        )

    @app.post(
        RECOMMENDATIONS_PATH,
        operation_id=RECOMMENDATIONS_OPERATION_ID,
        summary="Ficha tactica publica P11 con politicas congeladas.",
        responses=recommendation_responses,
    )
    async def post_recommendations(
        request: Request, payload: TacticalRecommendationRequest
    ) -> Response:
        request_id = _request_id(request)
        query = TacticalRecommendationQuery(
            payload.player_id, payload.opponent_id, payload.as_of_date
        )
        response = service.recommend(query)
        fingerprint = tactical_recommendation_fingerprint(response)
        _log_service_event(
            request,
            200,
            event="recommendation_completed",
            request_id=request_id,
            stage="public_projection",
        )
        return Response(
            content=canonical_tactical_recommendation_json(response),
            media_type="application/json",
            headers={
                "ETag": f'"{fingerprint}"',
                "X-Request-ID": request_id,
            },
        )

    return app
