"""Tests sinteticos P17: arranque local productivo P12 sobre snapshot P13.

Verifican: carga P13 exactamente una vez por arranque, cero carga al
importar, cero carga por request, reutilizacion del provider entre
requests, health y OpenAPI sin metadata privada, fallos cerrados antes
de ``uvicorn.run`` (snapshot ausente/malformado/incompatible, ruta
dentro del repositorio, ruta fuera de contrato, fallo del servidor),
errores y exception chaining sanitizados sin traceback ni rutas,
argumentos CLI invalidos que no cargan nada, host/puerto/worker/reload
cerrados, bytes P11 y ETag exactos, import sin efectos (AST +
subprocess) y concurrency sintetica segura. Ningun test arranca un
servidor real, carga el snapshot real o invoca P10/P14/P15.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.runtime as runtime
import src.recommender.persisted_tactical_recommendation_provider as p13
from src.recommender.tactical_recommendation_contract import (
    canonical_tactical_recommendation_json,
    tactical_recommendation_fingerprint,
)
from src.recommender.tactical_recommendation_service import (
    SERVICE_NAME,
    TacticalRecommendationQuery,
    TacticalRecommendationService,
)
from test_tactical_recommendation_service import (
    AS_OF_DATE,
    _prioritization,
)


_ROOT = Path(__file__).resolve().parents[1]
_PLAYER = "RTPlayerSecretA"
_OPPONENT = "RTRivalSecretB"
_AS_OF_BODY = AS_OF_DATE.isoformat()

_MESSAGES = runtime._RUNTIME_MESSAGES


def _body() -> dict[str, str]:
    return {
        "player_id": _PLAYER,
        "opponent_id": _OPPONENT,
        "as_of_date": _AS_OF_BODY,
    }


def _snapshot_file(base: Path, name: str = "runtime-snapshot.json") -> Path:
    result = _prioritization(
        "available", 0, AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    snapshot = p13.build_persisted_tactical_recommendation_snapshot([result])
    path = base / name
    p13.persist_persisted_tactical_recommendation_snapshot(snapshot, path)
    return path


class _CountingFactory:
    def __init__(self, real) -> None:
        self.real = real
        self.calls: list = []

    def __call__(self, path) -> object:
        self.calls.append(path)
        return self.real(path)


@pytest.fixture()
def snapshot_path(tmp_path) -> Path:
    return _snapshot_file(tmp_path)


@pytest.fixture()
def load_factory(monkeypatch, snapshot_path):
    """Espia la carga P13 sin cambiar su comportamiento real."""
    factory = _CountingFactory(
        p13.create_persisted_tactical_recommendation_provider
    )
    monkeypatch.setattr(
        p13, "create_persisted_tactical_recommendation_provider", factory
    )
    return factory


@pytest.fixture()
def uvicorn_spy(monkeypatch):
    """Espia uvicorn.run sin arrancar ningun servidor real."""
    calls: list = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(runtime.uvicorn, "run", _spy)
    return calls


# --------------------------------------------------------------------- #
# A. Import sin efectos e invariantes de arquitectura (AST + subprocess) #
# --------------------------------------------------------------------- #


def test_runtime_import_subprocess_sin_efectos(tmp_path):
    sentinel = tmp_path / "sentinela.txt"
    sentinel.write_text("x", encoding="utf-8")
    before = {item.name for item in tmp_path.iterdir()}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", "import src.api.runtime as r; print(r.RUNTIME_WORKERS)"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "1"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_ast_runtime_imports_cerrados():
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    imported = set()
    banned_modules = {
        "socket",
        "urllib",
        "http",
        "requests",
        "pandas",
        "pyarrow",
        "pickle",
        "subprocess",
        "os",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
                for alias in node.names:
                    imported.add(f"{node.module}.{alias.name}")
    banned = {
        name for name in imported
        if any(name == bad or name.startswith(bad + ".") for bad in banned_modules)
    }
    assert banned == set()
    assert not any(
        name.startswith("src.analysis") for name in imported
    ), "runtime P17 no puede tocar P10/P14/P15"
    allowed = {
        "__future__",
        "__future__.annotations",
        "argparse",
        "re",
        "collections.abc",
        "collections.abc.Sequence",
        "pathlib",
        "pathlib.Path",
        "typing",
        "typing.Final",
        "uvicorn",
        "fastapi",
        "fastapi.FastAPI",
        "src.api.app",
        "src.api.app.create_app",
        "src.recommender",
        "src.recommender.persisted_tactical_recommendation_provider",
    }
    assert imported <= allowed


def test_ast_runtime_solo_guard_main_y_sin_llamadas_modulo():
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ) and type(statement.value.value) is str:
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
    main_calls = []
    for statement in tree.body:
        if not isinstance(statement, ast.If):
            continue
        test = statement.test
        is_guard = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        )
        if not is_guard:
            continue
        for sub in ast.walk(statement):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "main"
            ):
                main_calls.append(sub)
    assert len(main_calls) == 1
    # Cero carga de snapshot/uvicorn a nivel de modulo.
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            pytest.fail("Llamada de modulo encontrada en runtime")
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Attribute
                ) and node.func.attr in {
                    "absolute",
                    "exists",
                    "is_dir",
                    "is_file",
                    "open",
                    "read_bytes",
                    "read_text",
                    "resolve",
                    "stat",
                }:
                    pytest.fail(
                        f"I/O de modulo encontrado: {node.func.attr}()"
                    )


def test_ast_runtime_exception_chaining_cerrado():
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    raised = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "SystemExit"
    ]
    assert raised, "El runtime debe salir con SystemExit cerrados."
    for raise_node in raised:
        assert isinstance(raise_node.cause, ast.Constant)
        assert raise_node.cause.value is None, (
            "Toda salida P17 usa 'raise SystemExit(...) from None'."
        )


def test_ast_runtime_sin_print_ni_stdout():
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "print":
            pytest.fail("print() en runtime: la ruta nunca se imprime.")
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ("stdout", "stderr")
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            # sys.stderr solo existe dentro del dict de logging cerrado.
            pytest.fail("Acceso a streams en runtime fuera del logging cerrado.")


# --------------------------------------------------------------------- #
# B. Construccion: una sola carga y provider inyectado                    #
# --------------------------------------------------------------------- #


def test_build_carga_p13_exactly_1_vez_y_sirve(snapshot_path, load_factory):
    app = runtime.build_app_from_snapshot(str(snapshot_path))
    assert load_factory.calls == [str(snapshot_path)]
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200


def test_cero_carga_por_request_y_reutilizacion_provider(
    snapshot_path, load_factory
):
    app = runtime.build_app_from_snapshot(str(snapshot_path))
    client = TestClient(app)
    first_bodies = []
    first_etags = []
    for _index in range(2):
        healthz = client.get("/healthz")
        assert healthz.status_code == 200
        post = client.post("/api/v1/recommendations", json=_body())
        assert post.status_code == 200
        first_bodies.append(post.content)
        first_etags.append(post.headers.get("ETag"))
    assert load_factory.calls == [str(snapshot_path)]
    assert first_bodies[0] == first_bodies[1]
    assert first_etags[0] == first_etags[1]


def test_health_y_openapi_no_expone_metadata_privada(snapshot_path):
    app = runtime.build_app_from_snapshot(str(snapshot_path))
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "api_version": "v1",
        "service": SERVICE_NAME,
        "status": "ok",
    }
    openapi_blob = json.dumps(app.openapi(), sort_keys=True, default=str)
    assert str(snapshot_path) not in openapi_blob
    assert _PLAYER not in openapi_blob
    assert _OPPONENT not in openapi_blob
    state_values = list(vars(app.state).values())
    for value in state_values:
        assert str(snapshot_path) not in repr(value)
    post = client.post("/api/v1/recommendations", json=_body())
    assert str(snapshot_path) not in post.text


# --------------------------------------------------------------------- #
# C. Fallos cerrados ANTES de uvicorn.run                                 #
# --------------------------------------------------------------------- #


def test_cli_snapshot_ausente_falla_antes_de_uvicorn(
    tmp_path, load_factory, uvicorn_spy
):
    missing = tmp_path / "no-existe.json"
    with pytest.raises(SystemExit) as exc_info:
        runtime.main(["--snapshot-path", str(missing)])
    assert exc_info.value.code == _MESSAGES["snapshot_load_failed"]
    # Una unica tentativa de carga (rechazada por P13); cero servidor.
    assert load_factory.calls == [str(missing)]
    assert uvicorn_spy == []
    assert str(missing) not in str(exc_info.value.code)
    assert "Traceback" not in str(exc_info.value.code)


def test_cli_snapshot_malformado_falla_antes_de_uvicorn(
    tmp_path, load_factory, uvicorn_spy
):
    bad = tmp_path / "malformado.json"
    bad.write_bytes(b"\xef\xbb\xbf\xff\xfe[no-json")
    with pytest.raises(SystemExit) as exc_info:
        runtime.main(["--snapshot-path", str(bad)])
    assert exc_info.value.code == _MESSAGES["snapshot_load_failed"]
    assert uvicorn_spy == []
    assert str(bad) not in str(exc_info.value.code)


def test_cli_snapshot_incompatible_falla_antes_de_uvicorn(
    tmp_path, load_factory, uvicorn_spy
):
    wrong = tmp_path / "incompatible.json"
    wrong.write_text('{"contract": "no-es-un-snapshot-p13"}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        runtime.main(["--snapshot-path", str(wrong)])
    assert exc_info.value.code == _MESSAGES["snapshot_load_failed"]
    assert uvicorn_spy == []


def test_build_ruta_dentro_del_repo_se_rechaza_sin_cargar(
    load_factory
):
    inner = str(_ROOT / "src" / "api" / "runtime-interior.json")
    with pytest.raises(runtime.RuntimePathContractError) as exc_info:
        runtime.build_app_from_snapshot(inner)
    assert str(exc_info.value) == _MESSAGES["route_in_repository"]
    assert load_factory.calls == []


def test_build_rutas_fuera_de_contrato_rechazadas(load_factory):
    bad_routes = [
        "runtime-relative.json",
        "file:///tmp/runtime-snapshot.json",
        "~/runtime-snapshot.json",
        "/Users/cualquiera/../../private/runtime-snapshot.json",
        "/tmp/runtime-snapshot\x00.json",
        "C:",
        42,
        None,
        (),
    ]
    for bad in bad_routes:
        with pytest.raises(runtime.RuntimePathContractError) as exc_info:
            runtime.build_app_from_snapshot(bad)
        assert str(exc_info.value) == _MESSAGES["route_rejected"]
    assert load_factory.calls == []


def test_cli_directorio_en_lugar_de_snapshot_falla_antes_de_uvicorn(
    tmp_path, load_factory, uvicorn_spy
):
    directory = tmp_path / "es-directorio"
    directory.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        runtime.main(["--snapshot-path", str(directory)])
    assert exc_info.value.code == _MESSAGES["snapshot_load_failed"]
    assert uvicorn_spy == []


def test_cli_fallo_uvicorn_produce_salida_cerrada(
    snapshot_path, load_factory, monkeypatch, uvicorn_spy
):
    def _uvicorn_explota(*args, **kwargs):
        raise RuntimeError("boom con detalle /privado/runtime.json")

    monkeypatch.setattr(runtime.uvicorn, "run", _uvicorn_explota)
    with pytest.raises(SystemExit) as exc_info:
        runtime.main(["--snapshot-path", str(snapshot_path)])
    assert exc_info.value.code == _MESSAGES["server_unavailable"]
    assert "/privado/runtime.json" not in str(exc_info.value.code)
    assert load_factory.calls == [str(snapshot_path)]


@pytest.mark.parametrize("stage", ["load", "serve"])
def test_keyboard_interrupt_devuelve_130_sin_salida(
    stage, snapshot_path, load_factory, monkeypatch, capsys
):
    if stage == "load":
        def _interrupt_load(_path):
            raise KeyboardInterrupt

        monkeypatch.setattr(
            p13,
            "create_persisted_tactical_recommendation_provider",
            _interrupt_load,
        )
    else:
        def _interrupt_serve(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(runtime.uvicorn, "run", _interrupt_serve)

    assert runtime.main(["--snapshot-path", str(snapshot_path)]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_argparse_no_repite_argumentos_privados(
    snapshot_path, load_factory, uvicorn_spy, capsys
):
    secret = str(snapshot_path.parent / "identidad-secreta")
    with pytest.raises(SystemExit) as exc_info:
        runtime.main(["--snapshot-path", str(snapshot_path), "--force", secret])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == _MESSAGES["cli_usage_error"] + "\n"
    assert secret not in captured.err
    assert str(snapshot_path) not in captured.err
    assert load_factory.calls == []
    assert uvicorn_spy == []


# --------------------------------------------------------------------- #
# D. CLI cerrado: uso, host, puerto, worker, reload                       #
# --------------------------------------------------------------------- #


def test_cli_argumentos_invalidos_no_cargan(snapshot_path, load_factory, uvicorn_spy):
    base = ["--snapshot-path", str(snapshot_path)]
    invalid_argv = [
        [],
        base + ["--port", "70000"],
        base + ["--port", "0"],
        base + ["--port", "0080"],
        base + ["--port", "80 0"],
        base + ["--port", "-1"],
        base + ["--host", "0.0.0.0"],
        base + ["--host", "localhost"],
        base + ["--workers", "4"],
        base + ["--reload"],
        base + ["--force"],
    ]
    for argv in invalid_argv:
        with pytest.raises(SystemExit) as exc_info:
            runtime.main(list(argv))
        assert exc_info.value.code == 2
    assert load_factory.calls == []
    assert uvicorn_spy == []


def test_cli_valido_entrega_a_uvicorn_cerrado(
    snapshot_path, load_factory, uvicorn_spy
):
    code = runtime.main(
        ["--snapshot-path", str(snapshot_path), "--port", "8123"]
    )
    assert code == 0
    assert load_factory.calls == [str(snapshot_path)]
    assert len(uvicorn_spy) == 1
    args, kwargs = uvicorn_spy[0]
    assert len(args) == 1
    assert args[0] is not None
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8123
    assert kwargs["workers"] == 1
    assert kwargs["reload"] is False
    assert kwargs["access_log"] is False
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        assert kwargs["log_config"]["loggers"][logger_name] == {
            "handlers": [],
            "level": "CRITICAL",
            "propagate": False,
        }
    assert str(snapshot_path) not in repr(kwargs.get("log_config", ""))
    assert str(snapshot_path) not in repr(args)


def test_logging_config_es_nueva_y_no_reactiva_access_log():
    first = runtime._closed_logging_config()
    second = runtime._closed_logging_config()
    assert first is not second
    first["loggers"]["uvicorn.access"]["handlers"].append("closed")
    assert second["loggers"]["uvicorn.access"]["handlers"] == []


# --------------------------------------------------------------------- #
# E. Contrato P11 byte-exacto atraves del bootstrap                       #
# --------------------------------------------------------------------- #


def test_bytes_p11_y_etag_permanecen_exactos(snapshot_path):
    app = runtime.build_app_from_snapshot(str(snapshot_path))
    client = TestClient(app)
    response = client.post("/api/v1/recommendations", json=_body())
    assert response.status_code == 200
    provider = p13.create_persisted_tactical_recommendation_provider(
        str(snapshot_path)
    )
    service = TacticalRecommendationService(provider)
    expected = service.recommend(
        TacticalRecommendationQuery(_PLAYER, _OPPONENT, AS_OF_DATE)
    )
    assert response.content == canonical_tactical_recommendation_json(expected)
    assert response.headers["ETag"] == (
        f'"{tactical_recommendation_fingerprint(expected)}"'
    )


# --------------------------------------------------------------------- #
# F. Concurrency sintetica segura                                         #
# --------------------------------------------------------------------- #


def test_concurrency_sintetica_segura(snapshot_path, load_factory):
    app = runtime.build_app_from_snapshot(str(snapshot_path))
    outcomes: list[tuple[int, bytes]] = []
    outcome_lock = threading.Lock()

    def _worker(_index: int) -> None:
        client = TestClient(app)
        healthz = client.get("/healthz")
        post = client.post("/api/v1/recommendations", json=_body())
        with outcome_lock:
            outcomes.append((healthz.status_code, b""))
            outcomes.append((post.status_code, post.content))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outcomes) == 16
    assert all(status == 200 for status, _content in outcomes)
    post_bodies = [content for _status, content in outcomes[1::2]]
    assert all(content == post_bodies[0] for content in post_bodies[1:])
    assert load_factory.calls == [str(snapshot_path)]
