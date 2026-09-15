"""Tests sinteticos P18: UI Streamlit local sobre la API HTTP P17.

Verifican: import sin efectos (AST + subprocess), imports cerrados P18
(sin P10-P16, sin pandas/pyarrow/pickle/subprocess/uvicorn, sin
``src.*``), ausencia de referencia a la variable de entorno de ruta del
snapshot, sin session_state/cache/analiticas, URL loopback segura
(esquema http unico, host 127.0.0.1 exacto, puerto 1..65535 estricto,
sin credenciales/path/query/fragmento) con mensajes cerrados,
identificadores y fecha con el mismo contrato cerrado que la API P12,
parse estricto del contrato publico P11 (juegos de claves exactos,
vinculacion cerrada patron-oportunidad-actor, estados de evidencia
cerrados, reconciliaciones de conteo), ``fetch_recommendation`` con
peticion exacta (URL, cuerpo de 3 claves, timeout explicito 30 s),
UNICA peticion por llamada y CERO reintentos, mensajes cerrados en
espanol para 404/422/500/502/503/504 y fallos de red local, privacidad
de errores (sin URL, sin identificadores, sin fecha, sin tracebacks) y
render headless de fichas disponibles/no disponibles. Ningun test toca
el snapshot real, no arranca ningun servidor y usa solo datos
sinteticos.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

import src.ui.streamlit_app as ui


_ROOT = Path(__file__).resolve().parents[1]
_BASE_URL = "http://127.0.0.1:59999"
_PLAYER = "JugadorSecretoX123"
_OPPONENT = "RivalSecretoY456"
_AS_OF = "2031-06-30"
_SENSITIVE = (
    _PLAYER,
    _OPPONENT,
    _AS_OF,
    "59999",
    "player_id",
    "opponent_id",
    "as_of_date",
    "api/v1",
    "Traceback",
)


class _FakeResponse:
    def __init__(self, status_code: object, content: object) -> None:
        self.status_code = status_code
        self.content = content


class _FakeClient:
    def __init__(self, response: _FakeResponse | None = None,
                 exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[tuple[str, dict, object]] = []

    def post(self, url: str, *, json: object = None,
             timeout: object = None) -> _FakeResponse:
        self.calls.append((url, json, timeout))
        if self.exc is not None:
            raise self.exc
        assert self.response is not None
        return self.response


def _evidence(perspective: str = "executor",
              state: str = "available") -> dict:
    if state == "available":
        counts = (60, 40, 20, 6)
        rates: tuple[float | None, ...] = (
            0.6666666666666666, 0.5555, 0.7777
        )
    else:
        counts = (0, 0, 0, 0)
        rates = (None, None, None)
    labeled, successes, failures, distinct = counts
    return {
        "perspective": perspective,
        "scope": "global",
        "evidence_state": state,
        "labeled_activations": labeled,
        "successes": successes,
        "failures": failures,
        "distinct_matches": distinct,
        "success_rate": rates[0],
        "wilson_lower": rates[1],
        "wilson_upper": rates[2],
    }


def _option(pattern: str = "P02",
            opportunity: str = "first_serve_direction",
            actor: str = "server",
            category: str = "c4",
            rank: int | None = 1,
            status: str = "ranked") -> dict:
    return {
        "pattern_id": pattern,
        "category": category,
        "tactical_opportunity": opportunity,
        "actor": actor,
        "status": status,
        "reason_codes": ["ok"] if rank else ["insufficient"],
        "executor_evidence": _evidence(state="available" if rank
                                       else "insufficient_labeled_attempts"),
        "opponent_allowed_evidence": _evidence(
            "opponent_allowed",
            state="available" if rank else "insufficient_labeled_attempts",
        ),
        "score_formula": "weighted",
        "score": 0.66 if rank else None,
        "descriptive_uncertainty_envelope": (
            [0.5555, 0.7777] if rank else None
        ),
        "rank_position": rank,
        "tie_group": 1 if rank else None,
        "canonical_explanation": "Explicacion canonica sintetica.",
    }


def _card(pattern: str = "P02",
          opportunity: str = "first_serve_direction",
          actor: str = "server",
          status: str = "available",
          options: tuple[dict, ...] | None = None) -> dict:
    opts = options if options is not None else (_option(),)
    ranked = [item for item in opts if item["status"] == "ranked"]
    abstained = [
        item for item in opts
        if item["status"] == "abstained_insufficient_evidence"
    ]
    return {
        "pattern_id": pattern,
        "actor": actor,
        "tactical_opportunity": opportunity,
        "status": status,
        "status_reason_codes": [],
        "categories": [item["category"] for item in opts],
        "options": list(opts),
        "ranked_options": [item["category"] for item in ranked],
        "top_options": [item["category"] for item in ranked[:1]],
        "abstained_options": [item["category"] for item in abstained],
        "requested_top_k": 3,
        "effective_top_k": len(ranked),
        "tie_expanded": False,
        "tie_group_count": 1,
        "total_options": len(opts),
        "scored_options": len(ranked),
        "abstained_options_count": len(abstained),
        "reconciliations": {
            key: True for key in ui._CARD_RECONCILIATION_KEYS
        },
    }


def _payload(cards: tuple[dict, ...] | None = None,
             status: str = "available") -> dict:
    return {
        "contract_version": "p11",
        "status": status,
        "status_reason_codes": [],
        "methodology": "sintetica",
        "combination": "w",
        "score_formula": "weighted",
        "uncertainty_method": "wilson",
        "executor_weight": 0.6,
        "opponent_weight": 0.4,
        "encoder_policy": "cerrado",
        "evidence_scope": "global",
        "minimum_labeled_activations": 50,
        "minimum_distinct_matches": 5,
        "requested_top_k": 3,
        "ranking_scope": "within_pattern",
        "global_cross_pattern_ranking": False,
        "cards": list(cards) if cards is not None else [_card()],
        "limitations": ["Limitacion sintetica publica."],
        "reconciliations": {
            key: True for key in ui._RESPONSE_RECONCILIATION_KEYS
        },
        "fingerprint": "fp-sintetico",
    }


def _fetch_json(payload: dict, status_code: int = 200) -> ui.UIOutcome:
    client = _FakeClient(
        _FakeResponse(status_code, json.dumps(payload).encode("utf-8"))
    )
    return ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )


def _model(status: str = "available") -> ui.PublicRecommendation:
    return ui.parse_public_recommendation(
        json.loads(json.dumps(_payload(status=status)))
    )


# --------------------------------------------------------------------- #
# A. Import sin efectos e invariantes de arquitectura (AST + subprocess) #
# --------------------------------------------------------------------- #


def test_ui_import_subprocess_sin_efectos(tmp_path) -> None:
    sentinel = tmp_path / "sentinela.txt"
    sentinel.write_text("x", encoding="utf-8")
    before = {item.name for item in tmp_path.iterdir()}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.ui.streamlit_app as u; "
            "print(u.UI_REQUEST_TIMEOUT_SECONDS)",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "30.0"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_ast_ui_imports_cerrados() -> None:
    tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
                for alias in node.names:
                    imported.add(f"{node.module}.{alias.name}")
    allowed = {
        "__future__",
        "__future__.annotations",
        "json",
        "re",
        "dataclasses",
        "dataclasses.dataclass",
        "datetime",
        "datetime.date",
        "datetime.datetime",
        "typing",
        "typing.Final",
        "urllib.parse",
        "urllib.parse.urlsplit",
        "httpx",
        "streamlit",
    }
    assert imported <= allowed
    assert not any(name.startswith("src.") for name in imported), (
        "La UI P18 no puede importar P10-P16 ni ningun modulo src."
    )
    banned = {"pandas", "pyarrow", "pickle", "subprocess", "uvicorn"}
    overlap = {
        name for name in imported
        if any(name == bad or name.startswith(bad + ".") for bad in banned)
    }
    assert overlap == set()


def test_ast_ui_guard_main_y_cuerpo_modulo() -> None:
    tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
        ):
            continue
        assert isinstance(
            statement,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.If,
            ),
        ), f"Estado de modulo no permitido: {type(statement).__name__}"
    guards = [
        statement for statement in tree.body
        if isinstance(statement, ast.If)
    ]
    assert len(guards) == 1
    guard = guards[0]
    assert isinstance(guard.test, ast.Compare)
    assert isinstance(guard.test.left, ast.Name)
    assert guard.test.left.id == "__name__"
    assert len(guard.test.ops) == 1
    assert isinstance(guard.test.ops[0], ast.Eq)
    call = guard.body[0]
    assert isinstance(call, ast.Expr)
    assert isinstance(call.value, ast.Call)
    assert isinstance(call.value.func, ast.Name)
    assert call.value.func.id == "main"
    assert len(guard.body) == 1


def test_ast_ui_sin_print_ni_i_o_ni_environ() -> None:
    tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in {"print", "open", "input"}, (
                f"LLamada prohibida: {node.id}()"
            )
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "environ", "getenv", "stdout", "stderr",
            }, f"Acceso prohibido: {node.attr}"


def test_ui_fuente_sin_snapshot_ni_estado() -> None:
    source = Path(ui.__file__).read_text(encoding="utf-8")
    for banned in (
        "TENNIS_TACTICAL_SNAPSHOT_PATH",
        "session_state",
        "st.cache",
        "uvicorn",
        "pandas",
        "pyarrow",
        "pickle",
        "subprocess",
        "logging",
    ):
        assert banned not in source, f"Prohibido en la UI: {banned}"


# --------------------------------------------------------------------- #
# B. URL loopback cerrada                                               #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("http://127.0.0.1", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:1", "http://127.0.0.1:1"),
        ("http://127.0.0.1:65535", "http://127.0.0.1:65535"),
        ("http://127.0.0.1:59999", "http://127.0.0.1:59999"),
    ],
)
def test_url_loopback_validas(raw: str, expected: str) -> None:
    assert ui.validate_api_base_url(raw) == expected


def test_url_valor_por_defecto() -> None:
    assert ui.UI_API_BASE_DEFAULT == "http://127.0.0.1:8000"
    assert ui.validate_api_base_url(ui.UI_API_BASE_DEFAULT) == (
        ui.UI_API_BASE_DEFAULT
    )


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("https://127.0.0.1:8000", "scheme"),
        ("127.0.0.1:8000", "scheme"),
        ("http://localhost:8000", "host"),
        ("http://10.0.0.1:8000", "host"),
        ("http://192.168.1.5:8000", "host"),
        ("http://127.1:8000", "host"),
        ("http://127.0.0.1.:8000", "host"),
        ("http://[::1]:8000", "host"),
        ("http://example.com:8000", "host"),
        ("http://user@127.0.0.1:8000", "credentials"),
        ("http://user:pass@127.0.0.1:8000", "credentials"),
        ("http://127.0.0.1:8000/", "path"),
        ("http://127.0.0.1:8000/x", "path"),
        ("http://127.0.0.1:8000?x=1", "query"),
        ("http://127.0.0.1:8000#f", "fragment"),
        ("http://127.0.0.1:0", "port"),
        ("http://127.0.0.1:65536", "port"),
        ("http://127.0.0.1:0080", "port"),
        ("http://127.0.0.1:/", "port"),
        ("http://127.0.0.1:99999999999", "port"),
        ("http://127.0.0.1:abc", "port"),
        ("  http://127.0.0.1:8000", "format"),
        ("http://127.0.0.1:8000 ", "format"),
        ("", "format"),
    ],
)
def test_url_invalidas_mensajes_cerrados(raw: str, key: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        ui.validate_api_base_url(raw)
    assert str(excinfo.value) == ui._UI_URL_MESSAGES[key]


def test_url_tipos_no_string() -> None:
    for bad in (None, 8000, ["http://127.0.0.1:8000"], b"http://127.0.0.1"):
        with pytest.raises(ValueError) as excinfo:
            ui.validate_api_base_url(bad)
        assert str(excinfo.value) == ui._UI_URL_MESSAGES["format"]


# --------------------------------------------------------------------- #
# C. Identificadores y fecha (contrato cerrado P12)                     #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    ["j1", "O" * 64, "a-b_c.9", "jugador_uno_2024"],
)
def test_identificadores_validos(raw: str) -> None:
    assert ui.validate_local_identifier(raw, "player") == raw
    assert ui.validate_local_identifier(raw, "opponent") == raw


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " j1",
        "j1 ",
        "x" * 65,
        "a/b",
        "a\\b",
        "a://b",
        "a..b",
        "a~b",
        "\x00a",
        "\x1fa",
        "a\x7f",
        "\x80a",
        "\x9fa",
        123,
        None,
        ["j1"],
    ],
)
def test_identificadores_invalidos_mensajes_cerrados(raw: object) -> None:
    with pytest.raises(ui._IdentifierContractError) as excinfo:
        ui.validate_local_identifier(raw, "player")
    assert str(excinfo.value) == "Jugador: identificador fuera del contrato."
    with pytest.raises(ui._IdentifierContractError) as excinfo:
        ui.validate_local_identifier(" a/b ", "opponent")
    assert (
        str(excinfo.value) == "Rival: identificador fuera del contrato."
    )


@pytest.mark.parametrize("raw", ["2024-01-15", "2024-02-29", "2000-01-01"])
def test_fechas_validas(raw: str) -> None:
    assert ui.validate_local_date(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "2023-02-29",
        "2024-02-30",
        "2024-13-01",
        "2024-00-10",
        "24-01-15",
        "2024-1-15",
        "2024-01-15 ",
        "2024-01-15T00:00:00",
        "",
        None,
        5,
    ],
)
def test_fechas_invalidas_mensaje_cerrado(raw: object) -> None:
    with pytest.raises(ui._IdentifierContractError) as excinfo:
        ui.validate_local_date(raw)
    assert (
        str(excinfo.value)
        == "Fecha: usa el formato YYYY-MM-DD con una fecha civil real."
    )


# --------------------------------------------------------------------- #
# D. Parse estricto del contrato publico P11                            #
# --------------------------------------------------------------------- #


def test_parse_payload_valido() -> None:
    model = _model()
    assert model.status == "available"
    assert model.status_reason_codes == ()
    assert model.limitations == ("Limitacion sintetica publica.",)
    card = model.cards[0]
    assert card.pattern_id == "P02"
    assert card.actor == "server"
    option = card.options[0]
    assert option.rank_position == 1
    assert option.score == 0.66
    assert option.executor_evidence.labeled_activations == 60
    assert option.executor_evidence.successes == 40
    assert option.executor_evidence.failures == 20
    assert option.executor_evidence.distinct_matches == 6
    assert option.executor_evidence.perspective == "executor"
    assert option.opponent_allowed_evidence.perspective == "opponent_allowed"
    assert (
        option.descriptive_uncertainty_envelope == (0.5555, 0.7777)
    )


def test_parse_payload_no_disponible() -> None:
    model = _model(status="not_available")
    assert model.status == "not_available"
    assert model.cards[0].status == "available"


@pytest.mark.parametrize(
    ("level", "key"),
    [
        ("response", "fingerprint"),
        ("response", "cards"),
        ("card", "reconciliations"),
        ("card", "tactical_opportunity"),
        ("option", "canonical_explanation"),
        ("evidence", "wilson_upper"),
    ],
)
def test_parse_falta_clave(level: str, key: str) -> None:
    data = json.loads(json.dumps(_payload()))
    if level == "response":
        del data[key]
    elif level == "card":
        del data["cards"][0][key]
    elif level == "option":
        del data["cards"][0]["options"][0][key]
    else:
        del data["cards"][0]["options"][0]["executor_evidence"][key]
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


@pytest.mark.parametrize(
    "level",
    ["response", "card", "option", "evidence"],
)
def test_parse_clave_extra(level: str) -> None:
    data = json.loads(json.dumps(_payload()))
    if level == "response":
        data["extra"] = 1
    elif level == "card":
        data["cards"][0]["extra"] = 1
    elif level == "option":
        data["cards"][0]["options"][0]["extra"] = 1
    else:
        data["cards"][0]["options"][0]["executor_evidence"]["extra"] = 1
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_reconciliaciones_falsas() -> None:
    data = json.loads(json.dumps(_payload()))
    data["reconciliations"]["policy_frozen"] = False
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["reconciliations"]["counts_reconciled"] = 0
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


@pytest.mark.parametrize("status", ["maybe", "AVAILABLE", ""])
def test_parse_status_respuesta_fuera_dominio(status: str) -> None:
    data = json.loads(json.dumps(_payload()))
    data["status"] = status
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_status_tarjeta_fuera_dominio() -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["status"] = "done"
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


@pytest.mark.parametrize(
    "status", ["ranked_up", "abstain", ""]
)
def test_parse_status_opcion_fuera_dominio(status: str) -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["status"] = status
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


@pytest.mark.parametrize(
    ("mutation"),
    [
        "bad_pattern",
        "p02_actor_returner",
        "p05_opportunity_wrong",
        "option_actor_wrong",
    ],
)
def test_parse_vinculacion_patron_opcion_actors(mutation: str) -> None:
    data = json.loads(json.dumps(_payload()))
    card_data = data["cards"][0]
    if mutation == "bad_pattern":
        card_data["pattern_id"] = "P99"
    elif mutation == "p02_actor_returner":
        card_data["actor"] = "returner"
    elif mutation == "p05_opportunity_wrong":
        card_data["pattern_id"] = "P05"
        card_data["tactical_opportunity"] = "first_serve_direction"
    else:
        card_data["options"][0]["actor"] = "returner"
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scope", "match"),
        ("perspective", "opponent"),
        ("evidence_state", "maybe"),
        ("labeled_activations", 7),
        ("failures", 123),
        ("distinct_matches", 61),
        ("success_rate", 1.5),
        ("wilson_lower", -0.1),
        ("wilson_upper", None),
    ],
)
def test_parse_evidence_fuera_contrato(field: str, value: object) -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["executor_evidence"][field] = value
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_evidence_activaciones_cero_con_rates() -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["executor_evidence"] = {
        "perspective": "executor",
        "scope": "global",
        "evidence_state": "available",
        "labeled_activations": 0,
        "successes": 0,
        "failures": 0,
        "distinct_matches": 0,
        "success_rate": 0.5,
        "wilson_lower": 0.4,
        "wilson_upper": 0.6,
    }
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", 1.5),
        ("score", -0.1),
        ("score", 1),
        ("rank_position", 0),
        ("rank_position", True),
        ("tie_group", 0),
        ("rank_position", "1"),
    ],
)
def test_parse_score_rank_tipo(field: str, value: object) -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0][field] = value
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_ranked_score_nulo() -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["score"] = None
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_ranked_envolvente_no_contiene_score() -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["descriptive_uncertainty_envelope"] = (
        [0.7, 0.9]
    )
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_ranked_sin_tie_group() -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["tie_group"] = None
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_abstencion_conserva_ranking() -> None:
    data = json.loads(json.dumps(_payload()))
    option_data = data["cards"][0]["options"][0]
    option_data["status"] = "abstained_insufficient_evidence"
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)
    for field in ("score", "descriptive_uncertainty_envelope",
                 "rank_position", "tie_group"):
        data = json.loads(json.dumps(_payload()))
        option_data = data["cards"][0]["options"][0]
        option_data["status"] = "not_applicable"
        option_data["score"] = None
        option_data["descriptive_uncertainty_envelope"] = None
        option_data["rank_position"] = None
        option_data["tie_group"] = None
        option_data["reason_codes"] = ["no_cobra"]
        option_data["executor_evidence"] = _evidence(
            state="no_observed_category"
        )
        option_data["opponent_allowed_evidence"] = _evidence(
            "opponent_allowed", state="no_observed_category"
        )
        option_data[field] = {"score": 0.5,
                              "descriptive_uncertainty_envelope": [0.4, 0.6],
                              "rank_position": 1, "tie_group": 1}[field]
        with pytest.raises(ui.PublicContractError):
            ui.parse_public_recommendation(data)


@pytest.mark.parametrize(
    "envelope",
    [
        [0.9, 0.5],
        [0.5],
        [0.5, 0.6, 0.7],
        [True, 0.6],
        [0.5, "0.6"],
        (0.5, 0.6),
    ],
)
def test_parse_envelope_fuera_contrato(envelope: object) -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["descriptive_uncertainty_envelope"] = (
        envelope
    )
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_tuplas_vacias_o_tipos() -> None:
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["reason_codes"] = ["ok", ""]
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)
    data = json.loads(json.dumps(_payload()))
    data["cards"][0]["options"][0]["reason_codes"] = "ok"
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)
    data = json.loads(json.dumps(_payload()))
    data["cards"] = "no-lista"
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


# --------------------------------------------------------------------- #
# E. fetch_recommendation: peticion exacta, cerrado, privado            #
# --------------------------------------------------------------------- #


def test_fetch_peticion_exacta() -> None:
    client = _FakeClient(_FakeResponse(200, b"{}"))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    url, body, timeout = client.calls[0]
    assert url == _BASE_URL + "/api/v1/recommendations"
    assert body == {
        "player_id": _PLAYER,
        "opponent_id": _OPPONENT,
        "as_of_date": _AS_OF,
    }
    assert set(body) == {"player_id", "opponent_id", "as_of_date"}
    assert timeout == ui.UI_REQUEST_TIMEOUT_SECONDS == 30.0
    assert len(client.calls) == 1


def test_fetch_200_contrato_ok() -> None:
    outcome = _fetch_json(_payload())
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["ok"]
    assert outcome.status_code == 200
    assert outcome.model is not None
    assert outcome.model.status == "available"
    assert outcome.message == ""


@pytest.mark.parametrize(
    "status_code",
    [404, 422, 500, 502, 503, 504],
)
def test_fetch_status_cerrados(status_code: int) -> None:
    client = _FakeClient(_FakeResponse(status_code, b"{}"))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["status_error"]
    assert outcome.status_code == status_code
    assert outcome.model is None
    assert outcome.message == ui._UI_STATUS_MESSAGES[status_code]
    assert len(client.calls) == 1


def test_fetch_status_desconocido() -> None:
    client = _FakeClient(_FakeResponse(599, b"{}"))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["status_error"]
    assert outcome.status_code == 599
    assert (
        outcome.message
        == ui._UI_LOCAL_MESSAGES["unexpected"]
    )


def test_fetch_200_json_invalido() -> None:
    client = _FakeClient(_FakeResponse(200, b"<esto no es json>"))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert (
        outcome.message
        == ui._UI_LOCAL_MESSAGES["incompatible_response"]
    )


def test_fetch_200_contrato_violado() -> None:
    payload = _payload()
    del payload["fingerprint"]
    outcome = _fetch_json(payload)
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert outcome.status_code == 200
    assert outcome.model is None


def test_fetch_status_code_no_int() -> None:
    client = _FakeClient(_FakeResponse("200", b"{}"))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert outcome.status_code is None


def test_fetch_content_no_bytes() -> None:
    client = _FakeClient(_FakeResponse(200, None))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("remito"),
        httpx.ReadError("corto"),
        httpx.ConnectTimeout("lento"),
        httpx.ReadTimeout("tardio"),
        httpx.PoolTimeout("cola"),
    ],
    ids=["connect", "read", "connect_timeout", "read_timeout", "pool_timeout"],
)
def test_fetch_fallos_red(exc: httpx.HTTPError) -> None:
    client = _FakeClient(exc=exc)
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    if isinstance(exc, httpx.TimeoutException):
        expected_kind = ui._UI_LOCAL_OUTCOMES["timeout"]
    else:
        expected_kind = ui._UI_LOCAL_OUTCOMES["connection"]
    assert outcome.kind == expected_kind
    assert outcome.message == ui._UI_LOCAL_MESSAGES[expected_kind]
    assert len(client.calls) == 1


def test_fetch_excepcion_exotica_sin_propagar() -> None:
    client = _FakeClient(exc=RuntimeError("fallo-improbable-interno"))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert outcome.message == ui._UI_LOCAL_MESSAGES["unexpected"]
    assert "fallo-improbable-interno" not in outcome.message


def test_fetch_cero_reintentos() -> None:
    attempts = 0

    class _SiempreFallando:
        def post(self, url: str, *, json: object = None,
                 timeout: object = None) -> None:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("caida")

    client = _SiempreFallando()
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert attempts == 1
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["connection"]


def test_fetch_privacidad_en_todos_fallos() -> None:
    scenarios: list[ui.UIOutcome] = []
    good = json.dumps(_payload()).encode("utf-8")
    broken = json.loads(json.dumps(_payload()))
    del broken["fingerprint"]
    scenarios.append(
        ui.fetch_recommendation(
            _FakeClient(_FakeResponse(200, good)), _BASE_URL,
            _PLAYER, _OPPONENT, _AS_OF,
        )
    )
    scenarios.append(
        ui.fetch_recommendation(
            _FakeClient(_FakeResponse(200, json.dumps(broken).encode())),
            _BASE_URL, _PLAYER, _OPPONENT, _AS_OF,
        )
    )
    scenarios.append(
        ui.fetch_recommendation(
            _FakeClient(_FakeResponse(200, b"no-json")), _BASE_URL,
            _PLAYER, _OPPONENT, _AS_OF,
        )
    )
    for status_code in (404, 422, 500, 502, 503, 504, 599):
        scenarios.append(
            ui.fetch_recommendation(
                _FakeClient(_FakeResponse(status_code, b"{}")),
                _BASE_URL, _PLAYER, _OPPONENT, _AS_OF,
            )
        )
    for exc in (
        httpx.ConnectError("remito " + _BASE_URL + " " + _PLAYER),
        httpx.ReadTimeout("tardio " + _AS_OF),
        RuntimeError("interna " + _OPPONENT),
    ):
        scenarios.append(
            ui.fetch_recommendation(
                _FakeClient(exc=exc), _BASE_URL,
                _PLAYER, _OPPONENT, _AS_OF,
            )
        )

    assert len(scenarios) >= 12
    for outcome in scenarios:
        if outcome.kind == ui._UI_LOCAL_OUTCOMES["ok"]:
            continue
        for sensitive in _SENSITIVE:
            assert sensitive not in outcome.message
        assert "Traceback" not in outcome.message


# --------------------------------------------------------------------- #
# F. Etiquetas cerradas                                                 #
# --------------------------------------------------------------------- #


def test_outcome_kind_label() -> None:
    ok = ui.UIOutcome(kind=ui._UI_LOCAL_OUTCOMES["ok"], message="")
    assert ui.outcome_kind_label(ok) == ""
    status = ui.UIOutcome(
        kind=ui._UI_LOCAL_OUTCOMES["status_error"],
        message=ui._UI_STATUS_MESSAGES[404],
        status_code=404,
    )
    assert ui.outcome_kind_label(status) == "(status 404)"
    local = ui.UIOutcome(
        kind=ui._UI_LOCAL_OUTCOMES["timeout"],
        message=ui._UI_LOCAL_MESSAGES["timeout"],
    )
    assert ui.outcome_kind_label(local) == "(fallo local del cliente)"


# --------------------------------------------------------------------- #
# G. Render headless (bare mode Streamlit)                              #
# --------------------------------------------------------------------- #


@pytest.fixture()
def streamlit_silenced():
    logger = logging.getLogger("streamlit")
    original = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(original)


def test_render_disponible_sin_excepciones(streamlit_silenced) -> None:
    model = _model()
    ui.render_public_recommendation(model)


def test_render_no_disponible_sin_excepciones(streamlit_silenced) -> None:
    model = _model(status="not_available")
    ui.render_public_recommendation(model)


def test_render_parcial_sin_excepciones(streamlit_silenced) -> None:
    payload = _payload()
    payload["status"] = "partially_available"
    payload["cards"] = [
        _card(),
        _card(
            pattern="P04",
            opportunity="initial_return_direction",
            actor="returner",
            status="not_available",
            options=(),
        ),
        _card(
            pattern="P05",
            opportunity="initial_return_depth",
            actor="returner",
            options=(
                _option(
                    pattern="P05",
                    opportunity="initial_return_depth",
                    actor="returner",
                    category="c1",
                    status="abstained_insufficient_evidence",
                    rank=None,
                ),
            ),
        ),
        _card(
            pattern="P06",
            opportunity="initial_return_shot_type",
            actor="returner",
            options=(
                _option(
                    pattern="P06",
                    opportunity="initial_return_shot_type",
                    actor="returner",
                    category="c1",
                ),
                _option(
                    pattern="P06",
                    opportunity="initial_return_shot_type",
                    actor="returner",
                    category="c2",
                    rank=2,
                ),
            ),
        ),
    ]
    model = ui.parse_public_recommendation(payload)
    ui.render_public_recommendation(model)
