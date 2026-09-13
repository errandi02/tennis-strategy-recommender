"""Pruebas de la capa HTTP P12 con proveedores sinteticos y TestClient.

Verifican el contrato publico: cuerpo canonico P11 byte-exacto, ETag
fuerte con el fingerprint P11, X-Request-ID opaco por request, sobre de
error cerrado, 422 de validacion sin tocar el provider, OpenAPI semantica
con operation IDs fijos y modulo sin estado global ni app al importar.
"""

from __future__ import annotations

import ast
from datetime import date
from hashlib import sha256
import json
import logging
import re
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import app as api_module
from src.api.app import (
    TacticalRecommendationRequest,
    TacticalRecommendationResponseSchema,
    create_app,
)
from src.recommender.tactical_recommendation_contract import (
    CARD_RECONCILIATIONS,
    PUBLIC_FINGERPRINT_DOMAIN,
    RESPONSE_RECONCILIATIONS,
    PublicEvidenceComponent,
    TacticalPatternCard,
    TacticalRecommendationOption,
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


def _client(
    provider: _RecordingProvider, *, raise_server_exceptions: bool = True
) -> TestClient:
    return TestClient(
        create_app(provider), raise_server_exceptions=raise_server_exceptions
    )


def _logged_events(caplog) -> list[dict[str, object]]:
    events = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("service_event "):
            continue
        events.append(json.loads(message[len("service_event "):]))
    return events


def _resolve_schema(spec: dict[str, object], schema: dict[str, object]) -> dict[str, object]:
    if "$ref" not in schema:
        return schema
    return spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]


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


def test_factory_does_not_expose_provider_as_mutable_app_state():
    provider = _provider_for("available")
    app = create_app(provider)
    assert app.state._state == {}
    assert not hasattr(app, "provider")
    assert not hasattr(app.state, "provider")


def test_create_app_requires_a_valid_provider():
    with pytest.raises(TypeError):
        create_app(None)
    with pytest.raises(TypeError):
        create_app("sin hook")


# --------------------------------------------------------------------------
# /healthz
# --------------------------------------------------------------------------


def test_healthz_reports_process_liveness():
    provider = _provider_for("available")
    client = _client(provider)
    response = client.get(_HEALTH_PATH)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "api_version": "v1",
        "service": "tactical-recommendation-api",
        "status": "ok",
    }
    assert _REQUEST_ID_RE.fullmatch(response.headers["x-request-id"])
    assert provider.calls == 0


def test_healthz_body_is_deterministic_but_request_ids_differ():
    client = _client(_provider_for("available"))
    first = client.get(_HEALTH_PATH)
    second = client.get(_HEALTH_PATH)
    assert first.content == second.content
    assert first.headers["x-request-id"] != second.headers["x-request-id"]


def test_each_handled_request_generates_exactly_one_server_request_id(monkeypatch):
    issued: list[str] = []

    def fake_uuid4():
        value = f"{len(issued) + 1:032x}"
        issued.append(value)
        return SimpleNamespace(hex=value)

    monkeypatch.setattr(api_module.uuid, "uuid4", fake_uuid4)
    success = _client(_provider_for("available")).post(
        _POST_PATH,
        json=_valid_body(),
        headers={"X-Request-ID": "client-supplied-id-must-be-ignored"},
    )
    rejected = _client(
        _RecordingProvider(raise_error=ProviderTimeoutError())
    ).post(_POST_PATH, json=_valid_body())
    invalid = _client(_provider_for("available")).post(_POST_PATH, json={})
    health = _client(_provider_for("available")).get(_HEALTH_PATH)
    assert issued == [f"{value:032x}" for value in range(1, 5)]
    assert [
        success.headers["x-request-id"],
        rejected.headers["x-request-id"],
        invalid.headers["x-request-id"],
        health.headers["x-request-id"],
    ] == issued
    assert success.headers["x-request-id"] != "client-supplied-id-must-be-ignored"


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
    documented = TacticalRecommendationResponseSchema.model_validate(response.json())
    assert documented.model_dump(mode="json") == response.json()
    assert provider.calls == 1


def test_success_exposes_independently_recomputed_strong_etag_and_request_id():
    response = _client(_provider_for("available")).post(_POST_PATH, json=_valid_body())
    payload = json.loads(response.content)
    published_fingerprint = payload.pop("fingerprint")
    core = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    independently_recomputed = sha256(PUBLIC_FINGERPRINT_DOMAIN + core).hexdigest().upper()
    assert re.fullmatch(r'"[0-9A-F]{64}"', response.headers["etag"])
    assert published_fingerprint == independently_recomputed
    assert response.headers["etag"] == f'"{independently_recomputed}"'
    assert not response.headers["etag"].startswith("W/")
    assert _REQUEST_ID_RE.fullmatch(response.headers["x-request-id"])


def test_success_obtains_etag_through_the_public_p11_verifier(monkeypatch):
    calls = []
    real_verifier = api_module.tactical_recommendation_fingerprint

    def verifying_spy(response):
        calls.append(response)
        return real_verifier(response)

    monkeypatch.setattr(
        api_module, "tactical_recommendation_fingerprint", verifying_spy
    )
    response = _client(_provider_for("available")).post(
        _POST_PATH, json=_valid_body()
    )
    assert response.status_code == 200
    assert len(calls) == 1
    assert response.headers["etag"] == f'"{calls[0].fingerprint}"'


def test_http_request_model_requires_a_string_date_before_parsing():
    with pytest.raises(ValidationError):
        TacticalRecommendationRequest(
            player_id=_PLAYER,
            opponent_id=_OPPONENT,
            as_of_date=date(2021, 1, 1),
        )
    request = TacticalRecommendationRequest(**_valid_body())
    assert type(request.as_of_date) is date
    assert request.as_of_date == date(2021, 1, 1)


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


@pytest.mark.parametrize("body_type", ["text/plain", "application/octet-stream"])
def test_incorrect_media_types_fail_with_the_same_closed_422(body_type):
    provider = _provider_for("available")
    response = _client(provider).post(
        _POST_PATH,
        content=json.dumps(_valid_body()).encode("utf-8"),
        headers={"Content-Type": body_type},
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "reason_code": "invalid_request",
            "stage": "request_validation",
            "message": "La request no cumple el contrato de entrada cerrado del servicio.",
            "retryable": False,
        }
    }
    assert provider.calls == 0


# --------------------------------------------------------------------------
# Rutas fuera del contrato
# --------------------------------------------------------------------------


def test_undocumented_paths_stay_outside_the_contract():
    client = _client(_provider_for("available"))
    responses = (
        client.get("/"),
        client.get(_POST_PATH),
        client.post(_HEALTH_PATH),
    )
    assert [response.status_code for response in responses] == [404, 405, 405]
    for response in responses:
        assert response.content == b""
        assert "detail" not in response.text
        assert _REQUEST_ID_RE.fullmatch(response.headers["x-request-id"])


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
    assert set(spec["paths"]) == {_HEALTH_PATH, _POST_PATH}
    assert set(spec["paths"][_HEALTH_PATH]) == {"get"}
    assert set(spec["paths"][_POST_PATH]) == {"post"}
    assert spec["paths"][_POST_PATH]["post"]["operationId"] == "tactical_recommendation_post"
    assert spec["paths"][_HEALTH_PATH]["get"]["operationId"] == "tactical_recommendation_health"
    spec_text = json.dumps(spec)
    for hidden_pattern in ("P03", "P07", "P08", "P09"):
        assert f'"{hidden_pattern}"' not in spec_text
    _sentinel_scan(
        spec_text,
        "PLAYER_SECRET_123",
        "OPPONENT_SECRET_456",
        "C:\\private\\secret",
        "/Users/private/secret",
        "\\\\server\\share\\secret",
        "~/secret",
        "../secret",
        "file:///secret",
        "2021-05-01",
        "6f27",
    )


def test_openapi_documents_the_exact_p11_response_schema():
    spec = _client(_provider_for("available")).get("/openapi.json").json()
    post = spec["paths"][_POST_PATH]["post"]
    schema = _resolve_schema(
        spec,
        post["responses"]["200"]["content"]["application/json"]["schema"],
    )
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        item.name for item in fields(TacticalRecommendationResponse)
    }
    assert schema["properties"]["cards"]["type"] == "array"
    assert schema["properties"]["cards"]["items"] == {
        "$ref": "#/components/schemas/TacticalPatternCardSchema"
    }
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
    expected_contract_fields = {
        "PublicEvidenceComponentSchema": PublicEvidenceComponent,
        "TacticalRecommendationOptionSchema": TacticalRecommendationOption,
        "TacticalPatternCardSchema": TacticalPatternCard,
        "TacticalRecommendationResponseSchema": TacticalRecommendationResponse,
    }
    for schema_name, contract_type in expected_contract_fields.items():
        component = components[schema_name]
        assert component["additionalProperties"] is False
        assert tuple(component["properties"]) == tuple(
            item.name for item in fields(contract_type)
        )
        assert set(component["required"]) == {
            item.name for item in fields(contract_type)
        }
    assert tuple(components["CardReconciliationsSchema"]["properties"]) == tuple(
        CARD_RECONCILIATIONS
    )
    assert tuple(components["ResponseReconciliationsSchema"]["properties"]) == tuple(
        RESPONSE_RECONCILIATIONS
    )
    option_schema = components["TacticalRecommendationOptionSchema"]
    assert option_schema["properties"]["executor_evidence"] == {
        "$ref": "#/components/schemas/PublicEvidenceComponentSchema"
    }
    assert option_schema["properties"]["opponent_allowed_evidence"] == {
        "$ref": "#/components/schemas/PublicEvidenceComponentSchema"
    }
    card_schema = components["TacticalPatternCardSchema"]
    assert card_schema["properties"]["options"]["items"] == {
        "$ref": "#/components/schemas/TacticalRecommendationOptionSchema"
    }
    for compact_partition in (
        "ranked_options",
        "top_options",
        "abstained_options",
    ):
        assert card_schema["properties"][compact_partition]["items"] == {
            "type": "string"
        }
    for component in components.values():
        if component.get("type") == "object":
            assert component.get("additionalProperties") is False


def test_openapi_documents_the_exact_health_contract():
    spec = _client(_provider_for("available")).get("/openapi.json").json()
    schema = _resolve_schema(
        spec,
        spec["paths"][_HEALTH_PATH]["get"]["responses"]["200"]["content"]
        ["application/json"]["schema"],
    )
    assert set(schema["properties"]) == {"api_version", "service", "status"}
    assert set(schema["required"]) == {"api_version", "service", "status"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["api_version"]["const"] == "v1"
    assert schema["properties"]["service"]["const"] == "tactical-recommendation-api"
    assert schema["properties"]["status"]["const"] == "ok"
    assert "timestamp" not in json.dumps(schema).lower()


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
    assert set(request_schema["properties"]) == {
        "player_id",
        "opponent_id",
        "as_of_date",
    }
    assert request_schema["additionalProperties"] is False
    assert request_schema["properties"]["as_of_date"]["type"] == "string"
    assert request_schema["properties"]["as_of_date"]["format"] == "date"
    request_text = json.dumps(request_schema, sort_keys=True)
    for knob in (
        "top_k",
        "weights",
        "thresholds",
        "scope",
        "policy",
        "patterns",
        "fallback",
        "smoothing",
    ):
        assert knob not in request_text


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


def test_generic_fastapi_handler_is_reached_and_redacts_every_privacy_sentinel(
    caplog, monkeypatch
):
    sentinels = (
        "PLAYER_SECRET_123",
        "OPPONENT_SECRET_456",
        "C:\\private\\secret",
        "/Users/private/secret",
        "\\\\server\\share\\secret",
        "~/secret",
        "../secret",
        "file:///secret",
        "2021-05-01",
        "6f27",
    )

    def unexpected(self, query):
        raise RuntimeError(" ".join(sentinels))

    generated_ids = []

    def fake_uuid4():
        generated_ids.append("f" * 32)
        return SimpleNamespace(hex="f" * 32)

    monkeypatch.setattr(TacticalRecommendationService, "recommend", unexpected)
    monkeypatch.setattr(api_module.uuid, "uuid4", fake_uuid4)
    app = create_app(_provider_for("available"))
    assert Exception in app.exception_handlers
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.INFO, logger=api_module.API_LOGGER_NAME):
        response = client.post(_POST_PATH, json=_valid_body())
    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "reason_code": "internal_error",
            "stage": "internal",
            "message": "Fallo interno sin clasificar del servicio.",
            "retryable": False,
        }
    }
    exposed = response.content.decode("utf-8") + json.dumps(dict(response.headers))
    exposed += " ".join(record.getMessage() for record in caplog.records)
    _sentinel_scan(exposed, *sentinels, "RuntimeError", "traceback")
    assert len(_logged_events(caplog)) == 1
    assert generated_ids == ["f" * 32]
    assert response.headers["x-request-id"] == "f" * 32
