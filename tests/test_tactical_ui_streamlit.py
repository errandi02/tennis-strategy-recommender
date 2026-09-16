"""Tests sinteticos P18: UI Streamlit local sobre la API HTTP P17.

Verifican: import sin efectos (AST + subprocess), imports cerrados P18
(solo el contrato publico P11; sin P10-P16/P13 ni
pandas/pyarrow/pickle/subprocess/uvicorn), ausencia de referencia a la variable de entorno de ruta del
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
from functools import lru_cache
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

import src.ui.streamlit_app as ui
from src.recommender.tactical_recommendation_contract import (
    PUBLIC_LIMITATIONS,
    canonical_tactical_recommendation_json,
)
from tests.test_tactical_recommendation_contract import (
    _all_available_specs,
    _partial_specs,
    _response,
    _zero_specs,
)


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
    def __init__(self, status_code: object, content: object,
                 headers: dict[str, str] | None = None,
                 chunks: tuple[object, ...] | None = None) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {} if headers is None else headers
        self._chunks = chunks
        self.iterated = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def iter_bytes(self):
        self.iterated = True
        if self._chunks is not None:
            yield from self._chunks
            return
        if type(self.content) is not bytes:
            yield self.content
            return
        yield self.content


class _FakeClient:
    def __init__(self, response: _FakeResponse | None = None,
                 exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[tuple[str, str, dict, object, object]] = []

    def stream(self, method: str, url: str, *, json: object = None,
               timeout: object = None,
               follow_redirects: object = None) -> _FakeResponse:
        self.calls.append((method, url, json, timeout, follow_redirects))
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


@lru_cache(maxsize=3)
def _canonical_payload(status: str) -> bytes:
    specs = {
        "available": _all_available_specs,
        "partially_available": _partial_specs,
        "not_available": _zero_specs,
    }[status]()
    return canonical_tactical_recommendation_json(_response(specs))


def _payload(status: str = "available") -> dict:
    return json.loads(_canonical_payload(status))


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
    # Unica excepcion tolerada en stderr: el aviso benigno e inherente
    # de Streamlit al aplicar ``st.cache_data(...)`` fuera de un
    # runtime real (una vez por funcion cacheada a nivel de modulo,
    # ver P25); CUALQUIER otra linea (traceback, red, I/O real) sigue
    # fallando este test.
    _benign_cache_warning = "No runtime found, using MemoryCacheStorageManager"
    for line in completed.stderr.splitlines():
        assert _benign_cache_warning in line, f"stderr inesperado: {line!r}"
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
        "logging",
        "re",
        "dataclasses",
        "dataclasses.dataclass",
        "datetime",
        "datetime.date",
        "types",
        "types.MappingProxyType",
        "typing",
        "typing.Final",
        "unicodedata",
        "urllib.parse",
        "urllib.parse.quote",
        "urllib.parse.urlsplit",
        "httpx",
        "streamlit",
        "src.recommender",
        "src.recommender.tactical_recommendation_contract",
    }
    assert imported <= allowed
    assert {
        name for name in imported if name.startswith("src.")
    } <= {
        "src.recommender",
        "src.recommender.tactical_recommendation_contract",
    }
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
    """P25: el asistente de 3 pasos (jugador -> rival -> fecha
    dependientes, con invalidacion en cascada) es ESTRUCTURALMENTE
    imposible en el modelo de reruns de Streamlit sin
    ``st.session_state`` (para recordar la seleccion vigente entre
    reruns) ni un cache acotado (``st.cache_data``, para no repetir la
    misma peticion HTTP en cada interaccion). Por eso, a diferencia de
    P18-P19 originales, este modulo SI usa ambos -- nunca para datos
    sensibles ni errores (ver ``UI_CATALOG_CACHE_TTL_SECONDS`` y los
    tests de invalidacion). El resto de las prohibiciones originales
    (sin snapshot, sin P10-P16, sin uvicorn/pandas/pyarrow/pickle/
    subprocess) sigue vigente sin excepcion."""
    source = Path(ui.__file__).read_text(encoding="utf-8")
    for banned in (
        "TENNIS_TACTICAL_SNAPSHOT_PATH",
        "uvicorn",
        "pandas",
        "pyarrow",
        "pickle",
        "subprocess",
    ):
        assert banned not in source, f"Prohibido en la UI: {banned}"
    assert "trust_env=False" in source
    assert "max_keepalive_connections=0" in source
    assert "follow_redirects=False" in source
    assert 'logging.getLogger("httpx").disabled = True' in source
    assert 'logging.getLogger("httpcore").disabled = True' in source
    assert ui.UI_MAX_RESPONSE_BYTES == 1024 * 1024


def test_cache_de_catalogo_tiene_ttl_acotado_y_nunca_cachea_errores() -> None:
    """El cache de catalogos declara un TTL finito y explicito (nunca
    ``ttl=None``/indefinido); las funciones cacheadas lanzan en
    cualquier fallo -- Streamlit nunca cachea una llamada que termina
    en excepcion, luego un error nunca queda "pegado" en cache."""
    assert isinstance(ui.UI_CATALOG_CACHE_TTL_SECONDS, int)
    assert 0 < ui.UI_CATALOG_CACHE_TTL_SECONDS <= 300
    source = Path(ui.__file__).read_text(encoding="utf-8")
    assert source.count("@st.cache_data(ttl=UI_CATALOG_CACHE_TTL_SECONDS") == 3
    assert "ttl=None" not in source


def test_dependencia_y_comando_desactivan_telemetria() -> None:
    requirements = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert requirements.splitlines().count("streamlit==1.63.0") == 1
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "streamlit run src/ui/streamlit_app.py "
        "--browser.gatherUsageStats false"
    ) in readme


def test_wizard_state_keys_are_stable_string_constants() -> None:
    keys = (
        ui.STATE_PLAYER, ui.STATE_OPPONENT, ui.STATE_DATE,
        ui.STATE_PLAYER_SEARCH, ui.STATE_OPPONENT_SEARCH,
        ui.STATE_RESULT, ui.STATE_RESULT_KEY,
    )
    assert all(type(key) is str and key for key in keys)
    assert len(set(keys)) == len(keys)


def test_changing_player_clears_opponent_and_date() -> None:
    state: dict[str, object] = {
        ui.STATE_PLAYER: "Alice",
        ui.STATE_OPPONENT: "Bob",
        ui.STATE_DATE: "2031-06-30",
        ui.STATE_RESULT: object(),
    }
    ui.apply_player_selection_change(state, "Carol")
    assert state[ui.STATE_PLAYER] == "Carol"
    assert state[ui.STATE_OPPONENT] is None
    assert state[ui.STATE_DATE] is None
    assert state[ui.STATE_RESULT] is None


def test_selecting_same_player_again_does_not_clear_opponent_or_date() -> None:
    state: dict[str, object] = {
        ui.STATE_PLAYER: "Alice",
        ui.STATE_OPPONENT: "Bob",
        ui.STATE_DATE: "2031-06-30",
    }
    ui.apply_player_selection_change(state, "Alice")
    assert state[ui.STATE_OPPONENT] == "Bob"
    assert state[ui.STATE_DATE] == "2031-06-30"


def test_changing_opponent_clears_date_but_not_player() -> None:
    state: dict[str, object] = {
        ui.STATE_PLAYER: "Alice",
        ui.STATE_OPPONENT: "Bob",
        ui.STATE_DATE: "2031-06-30",
        ui.STATE_RESULT: object(),
    }
    ui.apply_opponent_selection_change(state, "Dave")
    assert state[ui.STATE_PLAYER] == "Alice"
    assert state[ui.STATE_OPPONENT] == "Dave"
    assert state[ui.STATE_DATE] is None
    assert state[ui.STATE_RESULT] is None


def test_changing_date_clears_only_the_previous_result() -> None:
    state: dict[str, object] = {
        ui.STATE_PLAYER: "Alice",
        ui.STATE_OPPONENT: "Bob",
        ui.STATE_DATE: "2031-06-30",
        ui.STATE_RESULT: object(),
    }
    ui.apply_date_selection_change(state, "2031-05-01")
    assert state[ui.STATE_PLAYER] == "Alice"
    assert state[ui.STATE_OPPONENT] == "Bob"
    assert state[ui.STATE_DATE] == "2031-05-01"
    assert state[ui.STATE_RESULT] is None


def test_apply_selection_change_works_on_a_mapping_like_state(monkeypatch) -> None:
    """La logica de invalidacion no depende de ``st.session_state``
    real: funciona sobre cualquier objeto tipo mapa mutable, para poder
    probarla sin un runtime de Streamlit."""

    class _MappingLike(dict):
        pass

    state = _MappingLike({ui.STATE_PLAYER: "Alice", ui.STATE_OPPONENT: "Bob"})
    ui.apply_player_selection_change(state, "Zoe")
    assert state[ui.STATE_OPPONENT] is None


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
    assert model.status_reason_codes == (
        "all_requested_patterns_have_scored_candidates",
    )
    assert model.limitations == PUBLIC_LIMITATIONS
    assert tuple(card.pattern_id for card in model.cards) == (
        "P02", "P04", "P05", "P06"
    )
    card = model.cards[0]
    assert card.pattern_id == "P02"
    assert card.actor == "server"
    option = card.options[0]
    assert option.rank_position == 3
    assert option.score == 0.325
    assert option.executor_evidence.labeled_activations == 60
    assert option.executor_evidence.successes == 15
    assert option.executor_evidence.failures == 45
    assert option.executor_evidence.distinct_matches == 10
    assert option.executor_evidence.perspective == "executor"
    assert option.opponent_allowed_evidence.perspective == "opponent_allowed"
    assert (
        option.descriptive_uncertainty_envelope
        == (0.22173006532469614, 0.4493300829250353)
    )


def test_parse_payload_no_disponible() -> None:
    model = _model(status="not_available")
    assert model.status == "not_available"
    assert all(card.status == "not_available" for card in model.cards)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", "p11"),
        ("methodology", "sintetica"),
        ("combination", "weighted"),
        ("score_formula", "weighted"),
        ("uncertainty_method", "wilson"),
        ("executor_weight", 0.6),
        ("opponent_weight", 0.4),
        ("encoder_policy", "profile_only"),
        ("evidence_scope", "surface"),
        ("minimum_labeled_activations", 49),
        ("minimum_distinct_matches", 4),
        ("requested_top_k", 2),
        ("ranking_scope", "within_pattern"),
        ("global_cross_pattern_ranking", True),
        ("limitations", ["texto no contractual"]),
        ("fingerprint", "A" * 64),
    ],
)
def test_parse_exige_constantes_y_fingerprint_p11_exactos(
    field: str, value: object
) -> None:
    data = _payload()
    data[field] = value
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(data)


def test_parse_exige_cuatro_tarjetas_en_orden_y_catalogos_exactos() -> None:
    missing = _payload()
    del missing["cards"][-1]
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(missing)

    reordered = _payload()
    reordered["cards"][0], reordered["cards"][1] = (
        reordered["cards"][1], reordered["cards"][0]
    )
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(reordered)

    catalog = _payload()
    catalog["cards"][0]["categories"] = ["4", "6", "5"]
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(catalog)


def test_parse_recalcula_wilson_score_y_particiones() -> None:
    wilson = _payload()
    wilson["cards"][0]["options"][0]["executor_evidence"][
        "wilson_lower"
    ] = 0.0
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(wilson)

    score = _payload()
    score["cards"][0]["options"][0]["score"] += 0.01
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(score)

    partition = _payload()
    partition["cards"][0]["ranked_options"] = partition["cards"][0][
        "ranked_options"
    ][:-1]
    with pytest.raises(ui.PublicContractError):
        ui.parse_public_recommendation(partition)


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
    method, url, body, timeout, follow_redirects = client.calls[0]
    assert method == "POST"
    assert url == _BASE_URL + "/api/v1/recommendations"
    assert body == {
        "player_id": _PLAYER,
        "opponent_id": _OPPONENT,
        "as_of_date": _AS_OF,
    }
    assert set(body) == {"player_id", "opponent_id", "as_of_date"}
    assert timeout == ui.UI_REQUEST_TIMEOUT_SECONDS == 30.0
    assert follow_redirects is False
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


def test_fetch_rechaza_content_length_excesivo_sin_leer_cuerpo() -> None:
    response = _FakeResponse(
        200,
        b"{}",
        headers={"content-length": str(ui.UI_MAX_RESPONSE_BYTES + 1)},
        chunks=(RuntimeError("no debe iterarse"),),
    )
    outcome = ui.fetch_recommendation(
        _FakeClient(response), _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert response.iterated is False


def test_fetch_rechaza_respuesta_chunked_al_superar_limite() -> None:
    response = _FakeResponse(
        200,
        b"",
        chunks=(b"x" * ui.UI_MAX_RESPONSE_BYTES, b"y"),
    )
    client = _FakeClient(response)
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("base_url", "player", "opponent", "as_of"),
    [
        ("https://127.0.0.1:59999", _PLAYER, _OPPONENT, _AS_OF),
        (_BASE_URL, True, _OPPONENT, _AS_OF),
        (_BASE_URL, _PLAYER, 2, _AS_OF),
        (_BASE_URL, _PLAYER, _PLAYER, _AS_OF),
        (_BASE_URL, _PLAYER, _OPPONENT, 20310630),
        (_BASE_URL, _PLAYER, _OPPONENT, "2031-6-30"),
    ],
)
def test_fetch_rechaza_request_fuera_de_contrato_antes_de_http(
    base_url: object, player: object, opponent: object, as_of: object
) -> None:
    client = _FakeClient()
    outcome = ui.fetch_recommendation(
        client, base_url, player, opponent, as_of  # type: ignore[arg-type]
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert outcome.message == ui._UI_LOCAL_MESSAGES["unexpected"]
    assert client.calls == []


def test_fetch_no_sigue_redirect_ni_lee_su_cuerpo() -> None:
    response = _FakeResponse(307, b"", chunks=(RuntimeError("no leer"),))
    client = _FakeClient(response)
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["status_error"]
    assert outcome.status_code == 307
    assert client.calls[0][-1] is False
    assert response.iterated is False


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


def test_fetch_200_json_con_clave_duplicada_falla_cerrado() -> None:
    client = _FakeClient(_FakeResponse(200, b'{"status":1,"status":2}'))
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.kind == ui._UI_LOCAL_OUTCOMES["incompatible_response"]
    assert outcome.message == ui._UI_LOCAL_MESSAGES["incompatible_response"]


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
        def stream(self, method: str, url: str, *, json: object = None,
                   timeout: object = None,
                   follow_redirects: object = None) -> None:
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
    model = _model(status="partially_available")
    ui.render_public_recommendation(model)


# --------------------------------------------------------------------- #
# H. Modo contenedor P19: endpoint fijo, sin entrada del usuario         #
# --------------------------------------------------------------------- #


def test_run_recommendation_experience_es_publico_y_reutilizable() -> None:
    """P25: sustituye a ``form_inputs`` (formulario unico, sin catalogo)
    por el asistente de 3 pasos, compartido entre P18 local y P19
    contenedor."""
    assert hasattr(ui, "run_recommendation_experience")
    assert not hasattr(ui, "form_inputs")
    assert not hasattr(ui, "_run_recommendation_experience")


def test_validate_container_api_base_url_acepta_solo_la_constante() -> None:
    assert (
        ui.validate_container_api_base_url(ui.UI_CONTAINER_API_BASE_URL)
        == ui.UI_CONTAINER_API_BASE_URL
    )
    assert ui.UI_CONTAINER_API_BASE_URL == "http://api:8000"


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:8000",
        "http://api:8000/",
        "http://api:8001",
        "https://api:8000",
        "http://attacker.example:8000",
        "http://api:8000?x=1",
        "",
        None,
        42,
    ],
)
def test_validate_container_api_base_url_rechaza_todo_lo_demas(bad_url) -> None:
    with pytest.raises(ValueError):
        ui.validate_container_api_base_url(bad_url)


def test_fetch_recommendation_modo_contenedor_usa_url_fija(monkeypatch) -> None:
    payload = _payload()
    client = _FakeClient(
        _FakeResponse(200, json.dumps(payload).encode("utf-8"))
    )
    outcome = ui.fetch_recommendation(
        client,
        ui.UI_CONTAINER_API_BASE_URL,
        _PLAYER,
        _OPPONENT,
        _AS_OF,
        container_mode=True,
    )
    assert outcome.model is not None
    assert client.calls[0][1] == (
        ui.UI_CONTAINER_API_BASE_URL + ui.UI_RECOMMENDATIONS_PATH
    )


def test_fetch_recommendation_modo_contenedor_rechaza_url_loopback() -> None:
    """En modo contenedor, una URL local valida en modo local se rechaza:
    el validador cerrado del modo contenedor no acepta ningun host
    distinto de la constante interna, ni siquiera loopback."""
    client = _FakeClient()
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF, container_mode=True
    )
    assert outcome.model is None
    assert client.calls == []


def test_fetch_recommendation_modo_local_por_defecto_no_cambia() -> None:
    """container_mode por defecto es False: comportamiento identico a
    antes de P19 para cualquier llamada existente sin el nuevo kwarg."""
    payload = _payload()
    client = _FakeClient(
        _FakeResponse(200, json.dumps(payload).encode("utf-8"))
    )
    outcome = ui.fetch_recommendation(
        client, _BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.model is not None
    assert client.calls[0][1] == _BASE_URL + ui.UI_RECOMMENDATIONS_PATH


def test_fetch_recommendation_modo_local_rechaza_url_contenedor() -> None:
    """Modo local (container_mode=False, el default) nunca acepta el
    endpoint interno del contenedor: sigue exigiendo loopback exacto."""
    client = _FakeClient()
    outcome = ui.fetch_recommendation(
        client, ui.UI_CONTAINER_API_BASE_URL, _PLAYER, _OPPONENT, _AS_OF
    )
    assert outcome.model is None
    assert client.calls == []


# --------------------------------------------------------------------- #
# I. P25 -- catalogo de descubrimiento: fetch, parseo, filtrado          #
# --------------------------------------------------------------------- #


def _json_response(body: object, status: int = 200) -> _FakeResponse:
    payload = json.dumps(body).encode("utf-8")
    return _FakeResponse(status, payload, headers={"content-length": str(len(payload))})


def test_fetch_players_catalog_parses_and_calls_expected_url() -> None:
    client = _FakeClient(_json_response({"players": ["Alice", "Bob"]}))
    players = ui.fetch_players_catalog(client, _BASE_URL)
    assert players == ("Alice", "Bob")
    assert client.calls[0][0] == "GET"
    assert client.calls[0][1] == _BASE_URL + ui.UI_CATALOG_PLAYERS_PATH


def test_fetch_players_catalog_raises_on_non_200() -> None:
    client = _FakeClient(_json_response({}, status=500))
    with pytest.raises(ui.CatalogUnavailableError):
        ui.fetch_players_catalog(client, _BASE_URL)


def test_fetch_players_catalog_raises_on_malformed_payload() -> None:
    client = _FakeClient(_json_response({"players": [1, 2]}))
    with pytest.raises(ui.CatalogUnavailableError):
        ui.fetch_players_catalog(client, _BASE_URL)


def test_fetch_players_catalog_raises_on_timeout() -> None:
    client = _FakeClient(exc=httpx.TimeoutException("slow"))
    with pytest.raises(ui.CatalogUnavailableError) as exc_info:
        ui.fetch_players_catalog(client, _BASE_URL)
    message = str(exc_info.value)
    for forbidden in ("slow", "Traceback", _BASE_URL):
        assert forbidden not in message


def test_fetch_players_catalog_enforces_max_response_bytes() -> None:
    huge = json.dumps({"players": ["A" * 10]}).encode("utf-8")
    response = _FakeResponse(
        200, huge, headers={"content-length": str(ui.UI_MAX_RESPONSE_BYTES + 1)}
    )
    client = _FakeClient(response)
    with pytest.raises(ui.CatalogUnavailableError):
        ui.fetch_players_catalog(client, _BASE_URL)
    assert not response.iterated


def test_fetch_opponents_catalog_parses_player_and_opponents() -> None:
    body = {
        "player_id": _PLAYER,
        "opponents": [
            {"opponent_id": "Zoe", "matchup_count": 3},
            {"opponent_id": "Amy", "matchup_count": 1},
        ],
    }
    client = _FakeClient(_json_response(body))
    opponents = ui.fetch_opponents_catalog(client, _BASE_URL, _PLAYER)
    assert opponents == (
        ui.CatalogOpponent(opponent_id="Zoe", matchup_count=3),
        ui.CatalogOpponent(opponent_id="Amy", matchup_count=1),
    )
    assert quote(_PLAYER, safe="") in client.calls[0][1]


def test_fetch_opponents_catalog_url_encodes_player_with_spaces() -> None:
    player_with_space = "Rafael Nadal"
    client = _FakeClient(_json_response({"player_id": player_with_space, "opponents": []}))
    ui.fetch_opponents_catalog(client, _BASE_URL, player_with_space)
    assert " " not in client.calls[0][1]
    assert "Rafael%20Nadal" in client.calls[0][1] or "Rafael+Nadal" in client.calls[0][1]


def test_fetch_opponents_catalog_rejects_malformed_player_id_before_request() -> None:
    client = _FakeClient()
    with pytest.raises(ui.CatalogUnavailableError):
        ui.fetch_opponents_catalog(client, _BASE_URL, "../etc/passwd")
    assert client.calls == []


def test_fetch_opponents_catalog_raises_on_404() -> None:
    client = _FakeClient(_json_response({}, status=404))
    with pytest.raises(ui.CatalogUnavailableError):
        ui.fetch_opponents_catalog(client, _BASE_URL, _PLAYER)


def test_fetch_dates_catalog_parses_iso_dates_in_given_order() -> None:
    body = {
        "player_id": _PLAYER,
        "opponent_id": _OPPONENT,
        "as_of_dates": ["2031-06-30", "2031-01-01"],
    }
    client = _FakeClient(_json_response(body))
    dates = ui.fetch_dates_catalog(client, _BASE_URL, _PLAYER, _OPPONENT)
    assert dates == ("2031-06-30", "2031-01-01")


def test_fetch_dates_catalog_rejects_non_iso_dates() -> None:
    body = {
        "player_id": _PLAYER, "opponent_id": _OPPONENT,
        "as_of_dates": ["30/06/2031"],
    }
    client = _FakeClient(_json_response(body))
    with pytest.raises(ui.CatalogUnavailableError):
        ui.fetch_dates_catalog(client, _BASE_URL, _PLAYER, _OPPONENT)


def test_catalog_fetchers_never_leak_sensitive_content_in_exception_message() -> None:
    client = _FakeClient(_json_response({}, status=500))
    for fetcher in (
        lambda: ui.fetch_players_catalog(client, _BASE_URL),
        lambda: ui.fetch_opponents_catalog(client, _BASE_URL, _PLAYER),
        lambda: ui.fetch_dates_catalog(client, _BASE_URL, _PLAYER, _OPPONENT),
    ):
        with pytest.raises(ui.CatalogUnavailableError) as exc_info:
            fetcher()
        message = str(exc_info.value)
        for sensitive in _SENSITIVE:
            assert sensitive not in message


# --------------------------------------------------------------------- #
# J. P25 -- filtrado de busqueda insensible a mayusculas y acentos       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("", ("Félix", "alice", "BOB")),
        ("fel", ("Félix",)),
        ("FÉL", ("Félix",)),
        ("felix", ("Félix",)),
        ("ALICE", ("alice",)),
        ("bo", ("BOB",)),
        ("zzz", ()),
    ],
)
def test_filter_catalog_by_search_case_and_accent_insensitive(
    query: str, expected: tuple[str, ...]
) -> None:
    options = ("Félix", "alice", "BOB")
    assert ui.filter_catalog_by_search(options, query) == expected


def test_filter_catalog_by_search_preserves_input_order() -> None:
    options = ("Zoe", "Amy", "Bob")
    assert ui.filter_catalog_by_search(options, "") == options
    assert ui.filter_catalog_by_search(options, "o") == ("Zoe", "Bob")


def test_filter_catalog_by_search_never_mutates_canonical_identifier() -> None:
    """El filtrado es SOLO para decidir que mostrar; el valor devuelto
    sigue siendo el identificador canonico original, nunca una version
    normalizada."""
    options = ("Félix Auger-Aliassime",)
    result = ui.filter_catalog_by_search(options, "felix")
    assert result == ("Félix Auger-Aliassime",)


# --------------------------------------------------------------------- #
# K. P25 -- invalidacion en cascada al enviar la consulta final          #
# --------------------------------------------------------------------- #


def test_run_recommendation_experience_calls_fetch_recommendation_at_most_once() -> None:
    """Garantia estructural (AST, no de comportamiento simulado): el
    cuerpo de ``run_recommendation_experience`` invoca
    ``fetch_recommendation`` (la UNICA peticion POST) como maximo una
    vez -- nunca dentro de un bucle, nunca mas de una llamada textual."""
    tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
    target = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_recommendation_experience"
    )
    for node in ast.walk(target):
        assert not isinstance(node, (ast.For, ast.While)), (
            "run_recommendation_experience no debe contener bucles "
            "(evita reintentos automaticos de POST)."
        )
    call_count = sum(
        1 for node in ast.walk(target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fetch_recommendation"
    )
    assert call_count == 1


def test_run_recommendation_experience_calls_catalog_fetchers_at_most_once_each() -> None:
    tree = ast.parse(Path(ui.__file__).read_text(encoding="utf-8"))
    target = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_recommendation_experience"
    )
    for cached_name in (
        "_cached_players_catalog", "_cached_opponents_catalog", "_cached_dates_catalog",
    ):
        call_count = sum(
            1 for node in ast.walk(target)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == cached_name
        )
        assert call_count == 1, f"{cached_name} debe invocarse exactamente una vez."
