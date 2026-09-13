"""Pruebas de la capa HTTP P12 con proveedores sinteticos y TestClient.

Verifican el contrato publico: cuerpo canonico P11 byte-exacto, ETag
fuerte con el fingerprint P11, X-Request-ID opaco por request, sobre de
error cerrado, 422 de validacion sin tocar el provider, OpenAPI semantica
con operation IDs fijos y modulo sin estado global ni app al importar.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import fields
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.app import create_app
from src.recommender.tactical_recommendation_contract import (
    TacticalRecommendationResponse,
    canonical_tactical_recommendation_json,
)
from src.recommender.tactical_recommendation_service import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    RecommendationNotFoundError,
    RecommendationUnavailableError,
    TacticalRecommendationQuery,
    TacticalRecommendationService,
    UpstreamContractViolationError,
)
from test_tactical_recommendation_service import (
    AS_OF_DATE,
    SCORING_PATTERNS,
    _RecordingProvider,
    _prioritization,
    _sentinel_scan,
)


_PLAYER = "PLAYER_SECRET_123"
_OPPONENT = "OPPONENT_SECRET_456"
_HEALTH_PATH = "/healthz"
_POST_PATH = "/api/v1/recommendations"
_REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}")


def _valid_body() -> dict[str, str]:
    return {
        "player_id": _PLAYER,
        "opponent_id": _OPPONENT,
        "as_of_date": "2021-01-01",
    }


def _result_for(state: str, variant: int = 0) -> object:
    return _prioritization(
        state,
        variant,
        AS_OF_DATE,
        False,
        player=_PLAYER,
        opponent=_OPPONENT,
    )


def _provider_for(state: str, variant: int = 0) -> _RecordingProvider:
    return _RecordingProvider(_result_for(state, variant))


def _client(provider: _RecordingProvider) -> TestClient:
    return TestClient(create_app(provider))


def _logged_events(caplog) -> list[dict[str, object]]:
    events = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("service_event "):
            continue
        events.append(json.loads(message[len("service_event "):]))
    return events


def test_api_module_does_not_build_app_or_provider_at_import():
    tree = ast.parse(Path(api_module.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else None)
            )
            assert name not in {"FastAPI", "TestClient", "create_app"}
    assert not hasattr(api_module, "app")
    assert callable(api_module.create_app)
    for forbidden in ("_PROVIDER", "PROVIDER", "DEFAULT_PROVIDER", "APP"):
        assert forbidden not in dir(api_module)


def test_create_app_requires_a_valid_provider():
    with pytest.raises(TypeError):
        create_app(None)
    with pytest.raises(TypeError):
        create_app("sin hook")


# --------------------------------------------------------------------------
# /healthz
# --------------------------------------------------------------------------


def test_healthz_reports_process_liveness():
    client = _client(_provider_for("available"))
    response = client.get(_HEALTH_PATH)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "api_version": "v1",
        "service": "tactical-recommendation-api",
        "status": "ok",
    }
    assert _REQUEST_ID_RE.fullmatch(response.headers["x-request-id"])


def test_healthz_body_is_deterministic_but_request_ids_differ():
    client = _client(_provider_for("available"))
    first = client.get(_HEALTH_PATH)
    second = client.get(_HEALTH_PATH)
    assert first.content == second.content
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


# --------------------------------------------------------------------------
# POST /api/v1/recommendations: exito
# --------------------------------------------------------------------------


def test_post_returns_exact_canonical_p11_bytes():
    provider = _provider_for("available")
    response = _client(provider).post(_POST_PATH, json=_valid_body())
    assert response.status_code == 200
    service = TacticalRecommendationService(_provider_for("available"))
    expected = canonical_tactical_recommendation_json(
        service.recommend(TacticalRecommendationQuery(_PLAYER, _OPPONENT, AS_OF_DATE))
    )
    assert response.content == expected
    assert provider.calls == 1


def test_success_exposes_strong_etag_and_request_id():
    response = _client(_provider_for("available")).post(_POST_PATH, json=_valid_body())
    fingerprint = json.loads(response.content)["fingerprint"]
    assert re.fullmatch(r'"[0-9A-F]{64}"', response.headers["etag"])
    assert response.headers["etag"] == f'"{fingerprint}"'
    assert not response.headers["etag"].startswith("W/")
    assert _REQUEST_ID_RE.fullmatch(response.headers["x-request-id"])


def test_success_body_has_exactly_the_p11_top_level_shape():
    response = _client(_provider_for("available")).post(_POST_PATH, json=_valid_body())
    payload = response.json()
    assert sorted(payload) == sorted(
        item.name for item in fields(TacticalRecommendationResponse)
    )
    assert [card["pattern_id"] for card in payload["cards"]] == list(SCORING_PATTERNS)
    text = response.content.decode("utf-8")
    for forbidden_key in (
        "player_id",
        "opponent_id",
        "as_of_date",
        "as_of",
        "match",
        "point",
        "sequence",
        "request_id",
    ):
        assert f'"{forbidden_key}"' not in text


def test_success_body_leaks_no_identity_dates_or_request_data():
    response = _client(_provider_for("available")).post(_POST_PATH, json=_valid_body())
    _sentinel_scan(
        response.content,
        _PLAYER,
        _OPPONENT,
        "2021-01-01",
        "Alice",
        "Bob",
        "Neutral",
        "h-p02",
        "2020-01-02",
    )
    assert response.headers["x-request-id"] not in response.content.decode("utf-8")


def test_consecutive_posts_are_byte_deterministic_with_distinct_request_ids():
    client = _client(_provider_for("available"))
    first = client.post(_POST_PATH, json=_valid_body())
    second = client.post(_POST_PATH, json=_valid_body())
    assert first.content == second.content
    assert first.headers["etag"] == second.headers["etag"]
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        ("partial", "partially_available"),
        ("absent", "not_available"),
    ],
)
def test_non_available_states_still_return_the_public_contract(state, expected_status):
    response = _client(_provider_for(state)).post(_POST_PATH, json=_valid_body())
    assert response.status_code == 200
    assert response.json()["status"] == expected_status
    assert response.headers["etag"]


# --------------------------------------------------------------------------
# POST: errores del provider (mapa cerrado)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_cls", "status_code", "retryable"),
    [
        (RecommendationNotFoundError, 404, False),
        (RecommendationUnavailableError, 503, True),
        (ProviderTimeoutError, 504, True),
        (ProviderUnavailableError, 503, True),
        (UpstreamContractViolationError, 502, False),
    ],
)
def test_known_provider_errors_use_the_closed_envelope(error_cls, status_code, retryable):
    response = _client(_RecordingProvider(raise_error=error_cls())).post(
        _POST_PATH, json=_valid_body()
    )
    assert response.status_code == status_code
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == {"reason_code", "stage", "message", "retryable"}
    assert error["reason_code"] == error_cls.reason_code
    assert error["stage"] == error_cls.stage
    assert error["retryable"] is retryable
    assert error["message"] == error_cls.public_message
    assert _REQUEST_ID_RE.fullmatch(response.headers["x-request-id"])
    _sentinel_scan(response.content, _PLAYER, _OPPONENT, "2021-01-01")


@pytest.mark.parametrize(
    "unexpected",
    [
        RuntimeError("PLAYER_SECRET_123 /Users/x C:\\tmp 2021-05-01"),
        KeyError("PLAYER_SECRET_123"),
        ZeroDivisionError(),
    ],
)
def test_unexpected_provider_failures_are_sanitized_internal_500(unexpected):
    response = _client(_RecordingProvider(raise_error=unexpected)).post(
        _POST_PATH, json=_valid_body()
    )
    assert response.status_code == 500
    error = response.json()["error"]
    assert error == {
        "reason_code": "internal_error",
        "stage": "internal",
        "message": "Fallo interno sin clasificar del servicio.",
        "retryable": False,
    }
    _sentinel_scan(
        response.content,
        _PLAYER,
        "/Users/x",
        "C:\\tmp",
        "2021-05-01",
        type(unexpected).__name__,
    )


# --------------------------------------------------------------------------
# POST: validacion (422 sin llamar al provider)
# --------------------------------------------------------------------------


_INVALID_BODIES: tuple[dict[str, object], ...] = (
    {},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT},
    {"player_id": _PLAYER, "as_of_date": "2021-01-01"},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": "2021-01-01", "top_k": 7},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": "2021-01-01", "encoder_policy": "component_only"},
    {"player_id": "", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": " Bob", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": "A" * 65, "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": "a/b", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": "a\\b", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": "http://x", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": "a..b", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": "~root", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": "C:\\Users", "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": 1, "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": True, "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": None, "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {"player_id": ["x"], "opponent_id": _OPPONENT, "as_of_date": "2021-01-01"},
    {**_valid_body(), "player_id": _PLAYER, "opponent_id": _PLAYER},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": "2021-01-01T00:00:00"},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": "2021-02-30"},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": "2021/01/01"},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": " 2021-01-01"},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": "20210101"},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": 20210101},
    {"player_id": _PLAYER, "opponent_id": _OPPONENT, "as_of_date": None},
)


def _invalid_body_id(body: dict[str, object]) -> str:
    return json.dumps(body, sort_keys=True, default=str)


@pytest.mark.parametrize(
    "body", _INVALID_BODIES, ids=[_invalid_body_id(item) for item in _INVALID_BODIES]
)
def test_invalid_bodies_are_rejected_without_provider_call(body):
    provider = _provider_for("available")
    response = _client(provider).post(_POST_PATH, json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error == {
        "reason_code": "invalid_request",
        "stage": "request_validation",
        "message": "La request no cumple el contrato de entrada cerrado del servicio.",
        "retryable": False,
    }
    assert provider.calls == 0
    assert _REQUEST_ID_RE.fullmatch(response.headers["x-request-id"])


@pytest.mark.parametrize(
    ("raw", "body_type"),
    [
        (b"{not json", "application/json"),
        (b"\"solo text\"", "application/json"),
        (b"[1, 2]", "application/json"),
    ],
)
def test_malformed_json_payload_is_rejected(raw, body_type):
    provider = _provider_for("available")
    response = _client(provider).post(
        _POST_PATH, content=raw, headers={"Content-Type": body_type}
    )
    assert response.status_code == 422
    assert response.json()["error"]["reason_code"] == "invalid_request"
    assert provider.calls == 0


# --------------------------------------------------------------------------
# Rutas fuera del contrato
# --------------------------------------------------------------------------


def test_undocumented_paths_stay_outside_the_contract():
    client = _client(_provider_for("available"))
    assert client.get("/").status_code == 404
    assert client.get(_POST_PATH).status_code == 405
    assert client.post(_HEALTH_PATH).status_code == 405
    assert "error" not in client.get("/").json()


def test_apps_are_independent_per_injected_provider():
    broken = _client(_RecordingProvider(raise_error=ProviderUnavailableError()))
    working = _client(_provider_for("available"))
    assert broken.post(_POST_PATH, json=_valid_body()).status_code == 503
    assert working.post(_POST_PATH, json=_valid_body()).status_code == 200


# --------------------------------------------------------------------------
# OpenAPI
# --------------------------------------------------------------------------


def test_openapi_is_semantically_stable_across_calls():
    client = _client(_provider_for("available"))
    first = json.dumps(client.get("/openapi.json").json(), sort_keys=True)
    second = json.dumps(client.get("/openapi.json").json(), sort_keys=True)
    assert first == second
    spec = client.get("/openapi.json").json()
    assert spec["paths"][_POST_PATH]["post"]["operationId"] == "tactical_recommendation_post"
    assert spec["paths"][_HEALTH_PATH]["get"]["operationId"] == "tactical_recommendation_health"
    spec_text = json.dumps(spec)
    for hidden_pattern in ("P03", "P07", "P08", "P09"):
        assert f'"{hidden_pattern}"' not in spec_text


def test_openapi_documents_the_exact_p11_response_schema():
    spec = _client(_provider_for("available")).get("/openapi.json").json()
    post = spec["paths"][_POST_PATH]["post"]
    schema = post["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        item.name for item in fields(TacticalRecommendationResponse)
    }
    assert schema["properties"]["cards"]["type"] == "array"
    assert schema["properties"]["executor_weight"]["type"] == "number"
    assert schema["properties"]["minimum_labeled_activations"]["type"] == "integer"
    assert schema["properties"]["global_cross_pattern_ranking"]["type"] == "boolean"
    headers = post["responses"]["200"]["headers"]
    assert "ETag" in headers
    assert "X-Request-ID" in headers
    for status in ("404", "422", "500", "502", "503", "504"):
        assert status in post["responses"]
    components = spec["components"]["schemas"]
    assert "PublicErrorEnvelope" in components
    assert set(components["PublicError"]["required"]) == {
        "reason_code",
        "stage",
        "message",
        "retryable",
    }


def test_openapi_documents_the_closed_request_schema():
    spec = _client(_provider_for("available")).get("/openapi.json").json()
    post = spec["paths"][_POST_PATH]["post"]
    request_schema = post["requestBody"]["content"]["application/json"]["schema"]
    if "$ref" in request_schema:
        request_schema = spec["components"]["schemas"][
            request_schema["$ref"].rsplit("/", 1)[-1]
        ]
    assert set(request_schema["required"]) == {
        "player_id",
        "opponent_id",
        "as_of_date",
    }
    assert request_schema["properties"]["as_of_date"]["type"] == "string"


# --------------------------------------------------------------------------
# Logging estructurado sin PII
# --------------------------------------------------------------------------


def test_success_logs_a_completed_event_with_closed_domains(caplog):
    client = _client(_provider_for("available"))
    with caplog.at_level(logging.INFO, logger=api_module.API_LOGGER_NAME):
        response = client.post(_POST_PATH, json=_valid_body())
    events = _logged_events(caplog)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "recommendation_completed"
    assert event["status_code"] == 200
    assert event["method"] == "POST"
    assert event["endpoint"] == _POST_PATH
    assert event["stage"] == "public_projection"
    assert _REQUEST_ID_RE.fullmatch(event["request_id"])
    assert response.headers["x-request-id"] == event["request_id"]
    _sentinel_scan(json.dumps(event), _PLAYER, _OPPONENT, "2021-01-01")


def test_healthz_logs_only_the_healthz_event_without_bodies(caplog):
    client = _client(_provider_for("available"))
    with caplog.at_level(logging.INFO, logger=api_module.API_LOGGER_NAME):
        response = client.get(_HEALTH_PATH)
    events = _logged_events(caplog)
    assert [event["event"] for event in events] == ["healthz"]
    assert set(events[0]) == {"event", "request_id", "status_code", "method", "endpoint"}
    assert response.headers["x-request-id"] == events[0]["request_id"]


@pytest.mark.parametrize(
    ("state", "status_code", "reason_code"),
    [
        ("not_found", 404, "recommendation_not_found"),
        ("unavailable", 503, "recommendation_unavailable"),
        ("timeout", 504, "provider_timeout"),
    ],
)
def test_rejected_requests_log_redacted_rejection_events(
    caplog, state, status_code, reason_code
):
    error_map = {
        "not_found": RecommendationNotFoundError,
        "unavailable": RecommendationUnavailableError,
        "timeout": ProviderTimeoutError,
    }
    client = _client(_RecordingProvider(raise_error=error_map[state]()))
    with caplog.at_level(logging.INFO, logger=api_module.API_LOGGER_NAME):
        response = client.post(_POST_PATH, json=_valid_body())
    assert response.status_code == status_code
    events = _logged_events(caplog)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "recommendation_rejected"
    assert event["reason_code"] == reason_code
    assert event["status_code"] == status_code
    assert "stage" in event
    _sentinel_scan(
        json.dumps(event), _PLAYER, _OPPONENT, "2021-01-01", "fetch_tactical"
    )


def test_unexpected_failure_logs_internal_rejection_without_stack(caplog):
    client = _client(
        _RecordingProvider(raise_error=RuntimeError("PLAYER_SECRET_123 trace /Users/x"))
    )
    with caplog.at_level(logging.INFO, logger=api_module.API_LOGGER_NAME):
        response = client.post(_POST_PATH, json=_valid_body())
    assert response.status_code == 500
    events = _logged_events(caplog)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "recommendation_rejected"
    assert event["reason_code"] == "internal_error"
    assert event["stage"] == "internal"
    _sentinel_scan(json.dumps(event), _PLAYER, "trace", "/Users/x", "RuntimeError")
