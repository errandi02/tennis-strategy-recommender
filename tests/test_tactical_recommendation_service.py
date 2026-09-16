"""Pruebas sinteticas del servicio puro P12 de recomendaciones tacticas.

Cada fixture upstream se valida antes de usarse. La evidencia sintetica
usa el mismo mecanismo semantico que P11-B (attempts -> orquestador ->
encoder -> evidencia -> priorizacion), sin leer datos, sin P10 y sin
importar tests desde produccion.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
from datetime import date, datetime, timedelta
from functools import lru_cache
import json
from types import MappingProxyType, SimpleNamespace
import uuid

import pytest

import src.recommender.tactical_recommendation_service as service_module
from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
    encode_tactical_attempt,
    feature_schema_fingerprint,
)
from src.recommender.tactical_history_profiles import (
    HISTORY_CONTRACT_VERSION,
    OBSERVATION_PROVENANCE,
    TacticalHistoricalObservation,
    TacticalLabelAvailability,
    TacticalOutcomeLabel,
    TacticalPlayerRole,
    validate_tactical_historical_observation,
)
from src.recommender.tactical_matchup_evidence import (
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupQuery,
    build_tactical_matchup_evidence,
    validate_tactical_matchup_evidence,
)
from src.recommender.tactical_prioritization import (
    TacticalPrioritizationResult,
    prioritize_tactical_matchup,
    validate_tactical_prioritization_result,
)
from src.recommender.tactical_recommendation_contract import (
    TacticalRecommendationContractError,
    TacticalRecommendationResponse,
    build_public_tactical_recommendation,
    canonical_tactical_recommendation_json,
    validate_public_tactical_recommendation,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
)
from src.recommender.tactical_recommendation_service import (
    MAX_IDENTIFIER_LENGTH,
    CatalogNotFoundError,
    InternalServiceError,
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RecommendationNotFoundError,
    RecommendationUnavailableError,
    TacticalRecommendationQuery,
    TacticalRecommendationService,
    TacticalRecommendationServiceError,
    UpstreamContractViolationError,
    redacted_service_event,
    validate_tactical_recommendation_query,
)


SCORING_PATTERNS = ("P02", "P04", "P05", "P06")
SERVE_DIRECTIONS = (("4", "wide"), ("5", "body"), ("6", "T"))
SHOT_TYPES = "fbrsvzopuylmhijkt"
AS_OF_DATE = date(2021, 1, 1)


@dataclass(frozen=True)
class _Attempt:
    match_id: str
    point_number: int
    effective_date: date
    server: str
    returner: str
    sequence: str
    serve_number: int
    previous_attempt_was_fault: bool
    server_won_point: bool


def _identity(variant: int) -> tuple[str, str, str, str, str, date]:
    if variant == 0:
        return "Alice", "Bob", "Neutral", "h", "", date(2020, 1, 2)
    return "Irene", "Jules", "Synthetic", "z", "c", date(2019, 1, 2)


def _attempt(
    *,
    match_id: str,
    point_number: int,
    effective_date: date,
    server: str,
    returner: str,
    sequence: str,
    serve_number: int = 1,
    previous_fault: bool = False,
    success: bool = True,
) -> _Attempt:
    return _Attempt(
        match_id,
        point_number,
        effective_date,
        server,
        returner,
        sequence,
        serve_number,
        previous_fault,
        success,
    )


def _full_history(variant: int, player: str, opponent: str) -> tuple[_Attempt, ...]:
    neutral, namespace, let, base_date = _identity(variant)[2:]
    attempts: list[_Attempt] = []
    point = 1000 * variant + 1
    for perspective in ("executor", "opponent"):
        for serve_code, category in SERVE_DIRECTIONS:
            for index in range(50):
                server = player if perspective == "executor" else f"{neutral}P02{category}"
                returner = (
                    f"{neutral}P02{category}" if perspective == "executor" else opponent
                )
                attempts.append(
                    _attempt(
                        match_id=f"{namespace}-p02-{perspective}-{category}-{index // 10}",
                        point_number=point,
                        effective_date=base_date + timedelta(days=index % 20),
                        server=server,
                        returner=returner,
                        sequence=f"{let}{serve_code}#",
                        success=index % 2 == 0,
                    )
                )
                point += 1
    for perspective in ("executor", "opponent"):
        for shot_index, shot_type in enumerate(SHOT_TYPES):
            lateral = str(shot_index % 3 + 1)
            depth = str(shot_index % 3 + 7)
            for index in range(50):
                server = (
                    f"{neutral}P06{shot_type}" if perspective == "executor" else opponent
                )
                returner = (
                    player if perspective == "executor" else f"{neutral}P06{shot_type}"
                )
                second_serve = index % 2 == 1
                attempts.append(
                    _attempt(
                        match_id=f"{namespace}-p06-{perspective}-{shot_type}-{index // 10}",
                        point_number=point,
                        effective_date=base_date + timedelta(days=index % 20),
                        server=server,
                        returner=returner,
                        sequence=f"{let}4{shot_type}{lateral}{depth}",
                        serve_number=2 if second_serve else 1,
                        previous_fault=second_serve and index % 4 == 1,
                        success=index % 2 != 0,
                    )
                )
                point += 1
    return tuple(attempts)


def _partial_history(variant: int, player: str, opponent: str) -> tuple[_Attempt, ...]:
    _, _, _, namespace, let, base_date = _identity(variant)
    attempts: list[_Attempt] = []
    point = 1000 * variant + 1
    for perspective in ("executor", "opponent"):
        for serve_code, category in SERVE_DIRECTIONS:
            for index in range(50):
                server = player if perspective == "executor" else f"N{category}"
                returner = f"N{category}" if perspective == "executor" else opponent
                attempts.append(
                    _attempt(
                        match_id=f"{namespace}-partial-{perspective}-{category}-{index // 10}",
                        point_number=point,
                        effective_date=base_date + timedelta(days=index % 20),
                        server=server,
                        returner=returner,
                        sequence=f"{let}{serve_code}#",
                        success=index % 2 == 0,
                    )
                )
                point += 1
    return tuple(attempts)


def _absent_history() -> tuple[_Attempt, ...]:
    actor_pairs = (
        ("Alice", "N1"),
        ("N2", "Bob"),
        ("N3", "Alice"),
        ("Bob", "N4"),
    )
    sequences = ("0q00", "4", "4#", "4n")
    return tuple(
        _attempt(
            match_id=f"absent-{index}",
            point_number=index + 1,
            effective_date=date(2020, 5, index + 1),
            server=server,
            returner=returner,
            sequence=sequences[index],
            serve_number=2 if index == 3 else 1,
            previous_fault=index == 3,
            success=index % 2 == 0,
        )
        for index, (server, returner) in enumerate(actor_pairs)
    )


def _observations(
    attempts: tuple[_Attempt, ...], schema
) -> tuple[TacticalHistoricalObservation, ...]:
    observations = []
    for attempt in attempts:
        request = AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION,
            attempt.sequence,
            attempt.serve_number,
            attempt.previous_attempt_was_fault,
        )
        extraction = extract_tactical_signals_for_attempt(request)
        vector = encode_tactical_attempt(extraction, schema)
        observation = TacticalHistoricalObservation(
            HISTORY_CONTRACT_VERSION,
            attempt.match_id,
            attempt.point_number,
            attempt.serve_number,
            attempt.effective_date,
            attempt.server,
            attempt.returner,
            TacticalPlayerRole.SERVER,
            vector,
            (
                TacticalOutcomeLabel.SERVER_WON_POINT
                if attempt.server_won_point
                else TacticalOutcomeLabel.RETURNER_WON_POINT
            ),
            TacticalLabelAvailability.AVAILABLE,
            OBSERVATION_PROVENANCE,
        )
        validate_tactical_historical_observation(observation)
        observations.append(observation)
    return tuple(observations)


def _query_for(
    player: str,
    opponent: str,
    schema,
    *,
    as_of: date = AS_OF_DATE,
) -> TacticalMatchupQuery:
    return TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        player,
        opponent,
        as_of,
        TacticalEncodingPolicy.COMPONENT_ONLY,
        feature_schema_fingerprint(schema),
        (1, 2),
        None,
        50,
        5,
        SCORING_PATTERNS,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )


@lru_cache(maxsize=16)
def _prioritization(
    state: str,
    variant: int,
    as_of: date,
    reversed_input: bool,
    *,
    player: str | None = None,
    opponent: str | None = None,
) -> TacticalPrioritizationResult:
    identity = _identity(variant)
    resolved_player = identity[0] if player is None else player
    resolved_opponent = identity[1] if opponent is None else opponent
    if state == "available":
        attempts = _full_history(variant, resolved_player, resolved_opponent)
    elif state == "partial":
        attempts = _partial_history(variant, resolved_player, resolved_opponent)
    else:
        attempts = _absent_history()
    if reversed_input:
        attempts = tuple(reversed(attempts))
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    evidence = build_tactical_matchup_evidence(
        _observations(attempts, schema),
        _query_for(resolved_player, resolved_opponent, schema, as_of=as_of),
        schema,
    )
    validate_tactical_matchup_evidence(evidence)
    result = prioritize_tactical_matchup(evidence, top_k=3)
    validate_tactical_prioritization_result(result)
    return result


@pytest.fixture(scope="module")
def available_result() -> TacticalPrioritizationResult:
    return _prioritization("available", 0, AS_OF_DATE, False)


@pytest.fixture(scope="module")
def alternative_identity_result() -> TacticalPrioritizationResult:
    return _prioritization("available", 1, AS_OF_DATE, False)


@pytest.fixture(scope="module")
def partial_result() -> TacticalPrioritizationResult:
    return _prioritization("partial", 0, AS_OF_DATE, False)


@pytest.fixture(scope="module")
def absent_result() -> TacticalPrioritizationResult:
    return _prioritization("absent", 0, AS_OF_DATE, False)


class _RecordingProvider:
    """Doble de test P12 que ademas satisface el catalogo P25
    (``list_players``/``list_opponents``/``list_as_of_dates``): por
    defecto devuelve catalogos vacios (nunca invocados por los tests
    que solo ejercitan recomendaciones), y opcionalmente puede
    inyectarse contenido sintetico o un error para los tests de
    catalogo."""

    def __init__(
        self,
        result=None,
        *,
        raise_error: BaseException | None = None,
        players: tuple[str, ...] = (),
        opponents: tuple[object, ...] = (),
        as_of_dates: tuple[date, ...] = (),
        catalog_raise_error: BaseException | None = None,
    ):
        self.result = result
        self.raise_error = raise_error
        self.calls = 0
        self.last_query = None
        self._players = players
        self._opponents = opponents
        self._as_of_dates = as_of_dates
        self.catalog_raise_error = catalog_raise_error
        self.catalog_calls = 0
        self.last_catalog_player_id: str | None = None
        self.last_catalog_opponent_id: str | None = None

    def fetch_tactical_prioritization(self, query: TacticalRecommendationQuery):
        self.calls += 1
        self.last_query = query
        if self.raise_error is not None:
            raise self.raise_error
        return self.result

    def list_players(self) -> tuple[str, ...]:
        self.catalog_calls += 1
        if self.catalog_raise_error is not None:
            raise self.catalog_raise_error
        return self._players

    def list_opponents(self, player_id: str) -> tuple[object, ...]:
        self.catalog_calls += 1
        self.last_catalog_player_id = player_id
        if self.catalog_raise_error is not None:
            raise self.catalog_raise_error
        return self._opponents

    def list_as_of_dates(
        self, player_id: str, opponent_id: str
    ) -> tuple[date, ...]:
        self.catalog_calls += 1
        self.last_catalog_player_id = player_id
        self.last_catalog_opponent_id = opponent_id
        if self.catalog_raise_error is not None:
            raise self.catalog_raise_error
        return self._as_of_dates


def _sentinel_scan(payload: object, *sentinels: str) -> None:
    if isinstance(payload, (bytes, bytearray)):
        text = payload.decode("utf-8", errors="replace")
    elif isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, default=repr, ensure_ascii=False)
    for sentinel in sentinels:
        assert sentinel not in text, f"Sentinela leak: {sentinel!r}"


def _valid_query(
    player_id: str = "Alice",
    opponent_id: str = "Bob",
    as_of: date = AS_OF_DATE,
) -> TacticalRecommendationQuery:
    return TacticalRecommendationQuery(player_id, opponent_id, as_of)


# --------------------------------------------------------------------------
# Contrato de query
# --------------------------------------------------------------------------


def test_query_accepts_valid_orientation():
    query = _valid_query()
    assert query.player_id == "Alice"
    assert query.opponent_id == "Bob"
    assert query.as_of_date == AS_OF_DATE
    assert type(query.as_of_date) is date
    validate_tactical_recommendation_query(query)


def test_query_is_frozen_and_exact_type():
    query = _valid_query()
    assert type(query) is TacticalRecommendationQuery
    with pytest.raises(FrozenInstanceError):
        query.player_id = "Carol"
    moved = replace(query, player_id="Carol")
    assert moved.player_id == "Carol"


def test_query_has_exactly_three_conceptual_fields_and_no_policy_knobs():
    names = tuple(item.name for item in fields(TacticalRecommendationQuery))
    assert names == ("player_id", "opponent_id", "as_of_date")
    for knob in (
        "top_k",
        "weights",
        "thresholds",
        "scope",
        "policy",
        "patterns",
        "smoothing",
        "fallback",
        "encoder_policy",
        "minimums",
    ):
        assert knob not in names


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("player_id", ""),
        ("player_id", " "),
        ("player_id", " x"),
        ("player_id", "x "),
        ("player_id", "\t x\n"),
        ("player_id", "A" * (MAX_IDENTIFIER_LENGTH + 1)),
        ("player_id", "a/b"),
        ("player_id", "a\\b"),
        ("player_id", "http://x"),
        ("player_id", "a..b"),
        ("player_id", "~root"),
        ("player_id", "//server/share"),
        ("player_id", "a\x00b"),
        ("player_id", "a\x01b"),
        ("player_id", "a\x7fb"),
        ("player_id", "a\x85b"),
        ("player_id", 1),
        ("player_id", True),
        ("player_id", None),
        ("player_id", b"bytes"),
        ("player_id", ["x"]),
        ("player_id", {"x": 1}),
        ("opponent_id", ""),
        ("opponent_id", " x "),
        ("opponent_id", "C:\\Users"),
        ("opponent_id", 2.5),
        ("opponent_id", False),
    ],
)
def test_invalid_identifiers_are_rejected(field, value):
    kwargs = {"player_id": "Alice", "opponent_id": "Bob"}
    kwargs[field] = value
    with pytest.raises(InvalidRequestError):
        _valid_query(**kwargs)


@pytest.mark.parametrize("length", [1, MAX_IDENTIFIER_LENGTH])
def test_identifier_length_boundaries_are_accepted(length):
    query = _valid_query("A" * length, "B" * length)
    validate_tactical_recommendation_query(query)


def test_same_player_and_opponent_are_rejected():
    with pytest.raises(InvalidRequestError):
        _valid_query("SameOne", "SameOne")


def test_as_of_date_must_be_exact_civil_date():
    with pytest.raises(InvalidRequestError):
        _valid_query(as_of=datetime(2021, 1, 1))
    with pytest.raises(InvalidRequestError):
        _valid_query(as_of=datetime(2021, 1, 1, 12, 30))
    with pytest.raises(InvalidRequestError):
        _valid_query(as_of="2021-01-01")
    with pytest.raises(InvalidRequestError):
        _valid_query(as_of=20210101)
    with pytest.raises(InvalidRequestError):
        _valid_query(as_of=None)
    leap = _valid_query(as_of=date(2024, 2, 29))
    validate_tactical_recommendation_query(leap)


def test_non_exact_query_types_are_rejected_before_provider():
    provider = _RecordingProvider(_prioritization("available", 0, AS_OF_DATE, False))
    service = TacticalRecommendationService(provider)
    with pytest.raises(InvalidRequestError):
        service.recommend({"player_id": "Alice", "opponent_id": "Bob"})
    with pytest.raises(InvalidRequestError):
        service.recommend("Alice/Bob")

    class _Subclass(TacticalRecommendationQuery):
        pass

    with pytest.raises(InvalidRequestError):
        service.recommend(_Subclass("Alice", "Bob", AS_OF_DATE))
    assert provider.calls == 0


# --------------------------------------------------------------------------
# Construccion del servicio
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_provider", [None, 42, "x", object(), {"k": 1}])
def test_service_requires_callable_provider(bad_provider):
    with pytest.raises(TypeError):
        TacticalRecommendationService(bad_provider)


def test_provider_with_non_callable_hook_is_rejected():
    with pytest.raises(TypeError):
        TacticalRecommendationService(
            SimpleNamespace(fetch_tactical_prioritization="no-callback")
        )


def test_service_exposes_provider_and_keeps_no_extra_state():
    provider = _RecordingProvider()
    service = TacticalRecommendationService(provider)
    assert service.provider is provider
    assert TacticalRecommendationService.__slots__ == ("_provider",)
    assert not hasattr(service, "__dict__")


# --------------------------------------------------------------------------
# Camino feliz
# --------------------------------------------------------------------------


def test_recommends_available_public_response(available_result):
    service = TacticalRecommendationService(_RecordingProvider(available_result))
    response = service.recommend(_valid_query())
    assert type(response) is TacticalRecommendationResponse
    assert response.status == "available"
    validate_public_tactical_recommendation(response, source=available_result)
    assert tuple(card.pattern_id for card in response.cards) == SCORING_PATTERNS


def test_recommends_partially_available_public_response(partial_result):
    service = TacticalRecommendationService(_RecordingProvider(partial_result))
    response = service.recommend(_valid_query())
    assert response.status == "partially_available"
    validate_public_tactical_recommendation(response, source=partial_result)


def test_recommends_not_available_public_response(absent_result):
    service = TacticalRecommendationService(_RecordingProvider(absent_result))
    response = service.recommend(_valid_query())
    assert response.status == "not_available"
    assert all(card.scored_options == 0 for card in response.cards)
    validate_public_tactical_recommendation(response, source=absent_result)


def test_provider_is_called_exactly_once_with_the_same_query_object(available_result):
    provider = _RecordingProvider(available_result)
    service = TacticalRecommendationService(provider)
    query = _valid_query()
    service.recommend(query)
    service.recommend(query)
    assert provider.calls == 2
    assert provider.last_query is query


def test_public_builder_and_validator_are_called_exactly_once(
    available_result, monkeypatch
):
    counts = {"builder": 0, "validator": 0}
    real_builder = service_module.build_public_tactical_recommendation
    real_validator = service_module.validate_public_tactical_recommendation

    def counting_builder(result):
        counts["builder"] += 1
        return real_builder(result)

    def counting_validator(response, source=None):
        counts["validator"] += 1
        return real_validator(response, source=source)

    monkeypatch.setattr(
        service_module, "build_public_tactical_recommendation", counting_builder
    )
    monkeypatch.setattr(
        service_module, "validate_public_tactical_recommendation", counting_validator
    )
    service = TacticalRecommendationService(_RecordingProvider(available_result))
    service.recommend(_valid_query())
    assert counts == {"builder": 1, "validator": 1}


# --------------------------------------------------------------------------
# Errores conocidos del provider
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_cls",
    [
        RecommendationNotFoundError,
        RecommendationUnavailableError,
        ProviderTimeoutError,
        ProviderUnavailableError,
    ],
)
def test_known_provider_errors_are_normalized_without_exception_chains(error_cls):
    source = error_cls()
    source.__cause__ = RuntimeError("PLAYER_SECRET_123 C:\\private\\secret")
    provider = _RecordingProvider(raise_error=source)
    service = TacticalRecommendationService(provider)
    with pytest.raises(error_cls) as excinfo:
        service.recommend(_valid_query())
    error = excinfo.value
    assert error is not source
    assert error.reason_code == error_cls.reason_code
    assert error.stage == error_cls.stage
    assert error.retryable is error_cls.retryable
    assert error.public_message == error_cls.public_message
    assert error.args == ()
    assert error.__cause__ is None
    assert error.__context__ is None


def test_provider_error_subclasses_and_mutated_instances_are_not_trusted():
    class _ForgedTimeout(ProviderTimeoutError):
        reason_code = "provider_timeout"

    forged = _ForgedTimeout()
    forged.public_message = "PLAYER_SECRET_123 /Users/private/secret"
    service = TacticalRecommendationService(_RecordingProvider(raise_error=forged))
    with pytest.raises(InternalServiceError) as excinfo:
        service.recommend(_valid_query())
    assert type(excinfo.value) is InternalServiceError
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    _sentinel_scan(
        repr(excinfo.value), "PLAYER_SECRET_123", "/Users/private/secret"
    )


@pytest.mark.parametrize(
    "unexpected",
    [
        RuntimeError("Alice /Users/omar C:\\tmp 2021-05-01 seq=7"),
        ValueError("file:///private/secret"),
        KeyError("Alice"),
        Exception("~~"),
        ZeroDivisionError(),
    ],
)
def test_unexpected_provider_exceptions_become_sanitized_internal_error(unexpected):
    provider = _RecordingProvider(raise_error=unexpected)
    service = TacticalRecommendationService(provider)
    with pytest.raises(InternalServiceError) as excinfo:
        service.recommend(_valid_query())
    error = excinfo.value
    assert error.reason_code == "internal_error"
    assert error.stage == "internal"
    assert error.retryable is False
    assert error.args == ()
    assert error.__cause__ is None
    assert error.__context__ is None
    _sentinel_scan(
        repr(error), "Alice", "/Users/omar", "C:\\tmp", "2021-05-01", "file://"
    )


def test_service_never_returns_when_provider_declares_failure(available_result):
    provider = _RecordingProvider(
        available_result, raise_error=RecommendationUnavailableError()
    )
    service = TacticalRecommendationService(provider)
    with pytest.raises(RecommendationUnavailableError):
        service.recommend(_valid_query())


def test_provider_timeout_is_a_contract_translation_only():
    provider = _RecordingProvider(raise_error=ProviderTimeoutError())
    service = TacticalRecommendationService(provider)
    with pytest.raises(ProviderTimeoutError) as excinfo:
        service.recommend(_valid_query())
    assert excinfo.value.retryable is True
    assert excinfo.value.stage == "provider_lookup"


# --------------------------------------------------------------------------
# Frontera upstream: resultado
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrong_result",
    [
        {"state": "available"},
        "available",
        7,
        None,
        [],
        ("P02",),
        object(),
    ],
)
def test_provider_wrong_type_is_upstream_contract_violation(wrong_result):
    service = TacticalRecommendationService(_RecordingProvider(wrong_result))
    with pytest.raises(UpstreamContractViolationError):
        service.recommend(_valid_query())


def test_result_for_a_different_orientation_is_rejected(available_result):
    service = TacticalRecommendationService(_RecordingProvider(available_result))
    with pytest.raises(UpstreamContractViolationError):
        service.recommend(_valid_query("Carol", "Dave"))
    with pytest.raises(UpstreamContractViolationError):
        service.recommend(_valid_query("Bob", "Alice"))


def test_result_for_a_different_cut_date_is_rejected():
    other_cut = _prioritization("partial", 0, date(2020, 6, 1), False)
    validate_tactical_prioritization_result(other_cut)
    service = TacticalRecommendationService(_RecordingProvider(other_cut))
    with pytest.raises(UpstreamContractViolationError):
        service.recommend(_valid_query())


def test_corrupted_result_object_is_rejected_with_sanitized_upstream_error():
    corrupted = object.__new__(TacticalPrioritizationResult)

    class _Shell:
        pass

    spoofed = _Shell()
    spoofed.__class__ = TacticalPrioritizationResult
    for bad in (corrupted, spoofed):
        service = TacticalRecommendationService(_RecordingProvider(bad))
        with pytest.raises(UpstreamContractViolationError) as excinfo:
            service.recommend(_valid_query())
        assert excinfo.value.args == ()
        _sentinel_scan(repr(excinfo.value), "Alice", "TacticalPrioritizationResult")


def test_manipulated_public_response_is_rejected(
    available_result, partial_result, monkeypatch
):
    foreign_response = build_public_tactical_recommendation(partial_result)
    monkeypatch.setattr(
        service_module,
        "build_public_tactical_recommendation",
        lambda result: foreign_response,
    )
    service = TacticalRecommendationService(_RecordingProvider(available_result))
    with pytest.raises(UpstreamContractViolationError):
        service.recommend(_valid_query())
    assert foreign_response.status == "partially_available"


def test_builder_failure_is_upstream_contract_violation(
    available_result, monkeypatch
):
    service = TacticalRecommendationService(_RecordingProvider(available_result))

    def failing_builder(result):
        try:
            raise RuntimeError("PLAYER_SECRET_123 C:\\private\\secret")
        except RuntimeError as source:
            raise TacticalRecommendationContractError(
                "politica fuera del contrato congelado"
            ) from source

    monkeypatch.setattr(
        service_module, "build_public_tactical_recommendation", failing_builder
    )
    with pytest.raises(UpstreamContractViolationError) as excinfo:
        service.recommend(_valid_query())
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    _sentinel_scan(
        repr(excinfo.value), "PLAYER_SECRET_123", "C:\\private\\secret"
    )


# --------------------------------------------------------------------------
# Privacidad y determinismo
# --------------------------------------------------------------------------


def test_different_identities_with_equivalent_evidence_yield_identical_public_output(
    available_result, alternative_identity_result
):
    left = TacticalRecommendationService(
        _RecordingProvider(available_result)
    ).recommend(_valid_query())
    right = TacticalRecommendationService(
        _RecordingProvider(alternative_identity_result)
    ).recommend(_valid_query("Irene", "Jules"))
    left_json = canonical_tactical_recommendation_json(left)
    right_json = canonical_tactical_recommendation_json(right)
    assert left == right
    assert left_json == right_json
    assert left.fingerprint == right.fingerprint
    _sentinel_scan(
        left_json, "Alice", "Bob", "Neutral", "h-p02", "2020-01-02", "2021-01-01"
    )
    _sentinel_scan(
        right_json, "Irene", "Jules", "Synthetic", "z-p06", "2019-01-02", "2021-01-01"
    )


def test_public_response_exposes_no_identity_bearing_fields(available_result, absent_result):
    for result in (available_result, absent_result):
        response = TacticalRecommendationService(_RecordingProvider(result)).recommend(
            _valid_query()
        )
        text = canonical_tactical_recommendation_json(response).decode("utf-8")
        for key_fragment in (
            "player_id",
            "opponent_id",
            "player",
            "opponent",
            "as_of",
            "match",
            "point",
            "sequence",
        ):
            assert f'"{key_fragment}"' not in text
    # La identidad vive solo en la frontera interna (resumen upstream),
    # nunca en el payload publico ya verificado arriba.
    assert absent_result.matchup_query.player == "Alice"
    _sentinel_scan(canonical_tactical_recommendation_json(
        TacticalRecommendationService(_RecordingProvider(absent_result)).recommend(
            _valid_query()
        )
    ), "Alice", "Bob")


def test_repeated_recommendations_are_byte_deterministic(available_result):
    service = TacticalRecommendationService(_RecordingProvider(available_result))
    first = canonical_tactical_recommendation_json(service.recommend(_valid_query()))
    second = canonical_tactical_recommendation_json(service.recommend(_valid_query()))
    assert first == second


def test_reversed_input_history_yields_the_same_public_output():
    forward = _prioritization("available", 0, AS_OF_DATE, False)
    backward = _prioritization("available", 0, AS_OF_DATE, True)
    left = TacticalRecommendationService(_RecordingProvider(forward)).recommend(
        _valid_query()
    )
    right = TacticalRecommendationService(_RecordingProvider(backward)).recommend(
        _valid_query()
    )
    assert left == right
    assert left.fingerprint == right.fingerprint


# --------------------------------------------------------------------------
# Frontera de logging y catalogo de errores
# --------------------------------------------------------------------------


def test_redacted_service_event_uses_only_closed_domains():
    event = redacted_service_event(
        "recommendation_rejected",
        request_id="a" * 32,
        method="POST",
        endpoint="/api/v1/recommendations",
        status_code=504,
        stage="provider_lookup",
        reason_code="provider_timeout",
    )
    assert type(event) is MappingProxyType
    assert event == {
        "event": "recommendation_rejected",
        "request_id": "a" * 32,
        "method": "POST",
        "endpoint": "/api/v1/recommendations",
        "status_code": 504,
        "stage": "provider_lookup",
        "reason_code": "provider_timeout",
    }
    with pytest.raises(TypeError):
        event["event"] = "otro"
    minimal = redacted_service_event("healthz")
    assert minimal == {"event": "healthz"}
    random_id = uuid.uuid4().hex
    with_id = redacted_service_event("recommendation_completed", request_id=random_id)
    assert with_id["request_id"] == random_id


def test_redacted_service_event_rejects_open_domains():
    with pytest.raises(ValueError):
        redacted_service_event("player_alice")
    with pytest.raises(ValueError):
        redacted_service_event("healthz", request_id="Alice/2021-01-01")
    with pytest.raises(ValueError):
        redacted_service_event("healthz", request_id="G" * 32)
    with pytest.raises(ValueError):
        redacted_service_event("healthz", request_id="a" * 7)
    with pytest.raises(ValueError):
        redacted_service_event("healthz", request_id=123)
    with pytest.raises(ValueError):
        redacted_service_event("recommendation_completed", stage="C:\\x")
    with pytest.raises(ValueError):
        redacted_service_event("recommendation_completed", method="DELETE")
    with pytest.raises(ValueError):
        redacted_service_event("recommendation_completed", endpoint="/admin")
    with pytest.raises(ValueError):
        redacted_service_event("recommendation_completed", status_code=999)
    with pytest.raises(ValueError):
        redacted_service_event("recommendation_completed", status_code=True)
    with pytest.raises(ValueError):
        redacted_service_event("recommendation_completed", reason_code="user:alice")
    with pytest.raises(ValueError):
        redacted_service_event("recommendation_completed", stage=7)


def test_service_error_catalog_is_closed_stable_and_pii_free():
    catalog = {
        cls.reason_code: cls
        for cls in (
            InvalidRequestError,
            RecommendationNotFoundError,
            RecommendationUnavailableError,
            ProviderTimeoutError,
            ProviderUnavailableError,
            UpstreamContractViolationError,
            InternalServiceError,
            CatalogNotFoundError,
        )
    }
    assert set(catalog) == service_module.SERVICE_REASON_CODES
    assert len(catalog) == 8
    expected = {
        "invalid_request": (False, "request_validation"),
        "recommendation_not_found": (False, "provider_lookup"),
        "recommendation_unavailable": (True, "provider_lookup"),
        "provider_timeout": (True, "provider_lookup"),
        "provider_unavailable": (True, "provider_lookup"),
        "upstream_contract_violation": (False, "result_validation"),
        "internal_error": (False, "internal"),
        "catalog_not_found": (False, "catalog_lookup"),
    }
    for reason_code, (retryable, stage) in expected.items():
        cls = catalog[reason_code]
        assert cls.retryable is retryable
        assert cls.stage == stage
        assert cls.stage in service_module.SERVICE_STAGES
        assert type(cls.public_message) is str and cls.public_message
        instance = cls()
        assert instance.args == ()
        assert isinstance(instance, TacticalRecommendationServiceError)
    for error in (
        InvalidRequestError(),
        RecommendationNotFoundError(),
        RecommendationUnavailableError(),
        ProviderTimeoutError(),
        ProviderUnavailableError(),
        UpstreamContractViolationError(),
        InternalServiceError(),
        CatalogNotFoundError(),
    ):
        _sentinel_scan(repr(error), "Alice", "/Users", "C:\\", "2021")
