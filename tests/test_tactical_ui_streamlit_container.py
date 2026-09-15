"""Tests sinteticos P19: entrypoint de contenedor sobre P18/P17.

Verifican: import sin efectos, cuerpo de modulo cerrado (solo un guard
__main__), invariantes AST (imports permitidos, sin print/open/input),
fallo cerrado cuando la variable de entorno dedicada esta ausente o
manipulada (sin renderizar el formulario ni permitir ninguna
peticion), flujo exitoso solo con el valor exacto activando el modo
contenedor (URL fija ``http://api:8000``, nunca configurable por el
usuario), y ausencia total de la variable, su valor o la URL interna
en cualquier mensaje mostrado. Ningun test arranca Streamlit real ni
toca red: todo I/O de Streamlit y httpx esta sustituido por dobles.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.ui.streamlit_container_app as cui


_ROOT = Path(__file__).resolve().parents[1]


class _StreamlitDouble:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.titles: list[str] = []
        self.captions: list[str] = []
        self.page_config_calls = 0
        self.spinner_calls = 0

    def set_page_config(self, **_kwargs) -> None:
        self.page_config_calls += 1

    def error(self, message: str) -> None:
        self.errors.append(message)

    def title(self, text: str) -> None:
        self.titles.append(text)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def spinner(self, _text: str):
        self.spinner_calls += 1

        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


@pytest.fixture()
def st_double(monkeypatch) -> _StreamlitDouble:
    double = _StreamlitDouble()
    monkeypatch.setattr(cui, "st", double)
    return double


# --------------------------------------------------------------------- #
# A. Import sin efectos e invariantes de arquitectura (AST + subprocess) #
# --------------------------------------------------------------------- #


def test_import_subprocess_sin_efectos(tmp_path) -> None:
    sentinel = tmp_path / "sentinela.txt"
    sentinel.write_text("x", encoding="utf-8")
    before = {item.name for item in tmp_path.iterdir()}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.ui.streamlit_container_app as c; "
            "print(c.CONTAINER_RUNTIME_MODE_VALUE)",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "container"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_ast_imports_cerrados() -> None:
    tree = ast.parse(Path(cui.__file__).read_text(encoding="utf-8"))
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
        "logging",
        "os",
        "datetime",
        "datetime.date",
        "datetime.datetime",
        "typing",
        "typing.Final",
        "httpx",
        "streamlit",
        "src.ui.streamlit_app",
        "src.ui.streamlit_app.UI_CONTAINER_API_BASE_URL",
        "src.ui.streamlit_app.UI_REQUEST_TIMEOUT_SECONDS",
        "src.ui.streamlit_app.fetch_recommendation",
        "src.ui.streamlit_app.form_inputs",
        "src.ui.streamlit_app.outcome_kind_label",
        "src.ui.streamlit_app.render_public_recommendation",
        "src.ui.streamlit_app.validate_local_date",
        "src.ui.streamlit_app.validate_local_identifier",
    }
    assert imported <= allowed
    banned = {"pandas", "pyarrow", "pickle", "subprocess", "uvicorn"}
    overlap = {
        name for name in imported
        if any(name == bad or name.startswith(bad + ".") for bad in banned)
    }
    assert overlap == set()


def test_ast_guard_main_y_cuerpo_modulo() -> None:
    tree = ast.parse(Path(cui.__file__).read_text(encoding="utf-8"))
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
    guards = [statement for statement in tree.body if isinstance(statement, ast.If)]
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


def test_ast_sin_print_ni_open_ni_input() -> None:
    tree = ast.parse(Path(cui.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in {"print", "open", "input"}, (
                f"Llamada prohibida: {node.id}()"
            )
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"stdout", "stderr"}, (
                f"Acceso prohibido: {node.attr}"
            )


def test_solo_este_modulo_lee_environ() -> None:
    """A diferencia de streamlit_app.py (P18), este modulo SI puede leer
    os.environ: es su unico proposito (resolver el modo contenedor)."""
    tree = ast.parse(Path(cui.__file__).read_text(encoding="utf-8"))
    found_environ_get = any(
        isinstance(node, ast.Attribute)
        and node.attr == "get"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environ"
        for node in ast.walk(tree)
    )
    assert found_environ_get


# --------------------------------------------------------------------- #
# B. Fallo cerrado: variable ausente o manipulada                        #
# --------------------------------------------------------------------- #


def test_modo_ausente_falla_cerrado_sin_formulario(
    monkeypatch, st_double
) -> None:
    monkeypatch.delenv(cui.CONTAINER_RUNTIME_MODE_ENV, raising=False)
    called = []
    monkeypatch.setattr(cui, "form_inputs", lambda: called.append(True))
    cui.main()
    assert st_double.errors == [cui._CONTAINER_MESSAGES["config_rejected"]]
    assert called == []
    assert st_double.titles == []


@pytest.mark.parametrize(
    "bad_value",
    ["", "true", "1", "CONTAINER", " container", "container ", "local"],
)
def test_modo_manipulado_falla_cerrado_sin_formulario(
    bad_value, monkeypatch, st_double
) -> None:
    monkeypatch.setenv(cui.CONTAINER_RUNTIME_MODE_ENV, bad_value)
    called = []
    monkeypatch.setattr(cui, "form_inputs", lambda: called.append(True))
    cui.main()
    assert st_double.errors == [cui._CONTAINER_MESSAGES["config_rejected"]]
    assert called == []


def test_mensaje_de_error_no_revela_variable_ni_valor_ni_url(
    monkeypatch, st_double
) -> None:
    monkeypatch.setenv(cui.CONTAINER_RUNTIME_MODE_ENV, "secreto-manipulado")
    cui.main()
    blob = " ".join(st_double.errors + st_double.captions + st_double.titles)
    assert cui.CONTAINER_RUNTIME_MODE_ENV not in blob
    assert "secreto-manipulado" not in blob
    assert "api:8000" not in blob


def test_mensaje_local_activo_no_revela_url_interna(monkeypatch, st_double) -> None:
    monkeypatch.setenv(
        cui.CONTAINER_RUNTIME_MODE_ENV, cui.CONTAINER_RUNTIME_MODE_VALUE
    )
    monkeypatch.setattr(cui, "form_inputs", lambda: ("", "", None, ""))
    cui.main()
    blob = " ".join(st_double.errors + st_double.captions + st_double.titles)
    assert "api:8000" not in blob
    assert cui.CONTAINER_RUNTIME_MODE_ENV not in blob


# --------------------------------------------------------------------- #
# C. Modo activo: URL fija, nunca configurable por el usuario            #
# --------------------------------------------------------------------- #


def test_modo_activo_sin_submit_no_llama_fetch(monkeypatch, st_double) -> None:
    monkeypatch.setenv(
        cui.CONTAINER_RUNTIME_MODE_ENV, cui.CONTAINER_RUNTIME_MODE_VALUE
    )
    monkeypatch.setattr(cui, "form_inputs", lambda: ("", "", None, ""))
    fetch_calls = []
    monkeypatch.setattr(
        cui, "fetch_recommendation", lambda *a, **k: fetch_calls.append((a, k))
    )
    cui.main()
    assert fetch_calls == []
    assert st_double.errors == []


def test_modo_activo_submit_usa_url_fija_del_contenedor(
    monkeypatch, st_double
) -> None:
    from datetime import date

    monkeypatch.setenv(
        cui.CONTAINER_RUNTIME_MODE_ENV, cui.CONTAINER_RUNTIME_MODE_VALUE
    )
    monkeypatch.setattr(
        cui,
        "form_inputs",
        lambda: ("JugadorX", "RivalY", date(2031, 6, 30), "sent"),
    )

    captured = {}

    class _FakeOutcome:
        model = None
        message = "cerrado"

    def _fake_fetch(client, base_url, player, opponent, as_of, *, container_mode):
        captured["base_url"] = base_url
        captured["player"] = player
        captured["opponent"] = opponent
        captured["as_of"] = as_of
        captured["container_mode"] = container_mode
        return _FakeOutcome()

    monkeypatch.setattr(cui, "fetch_recommendation", _fake_fetch)
    monkeypatch.setattr(cui, "outcome_kind_label", lambda _outcome: "(fallo)")

    class _FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(cui.httpx, "Client", _FakeClient)

    cui.main()

    assert captured["base_url"] == cui.UI_CONTAINER_API_BASE_URL
    assert captured["base_url"] == "http://api:8000"
    assert captured["player"] == "JugadorX"
    assert captured["opponent"] == "RivalY"
    assert captured["as_of"] == "2031-06-30"
    assert captured["container_mode"] is True


def test_modo_activo_jugador_igual_rival_se_rechaza(monkeypatch, st_double) -> None:
    from datetime import date

    monkeypatch.setenv(
        cui.CONTAINER_RUNTIME_MODE_ENV, cui.CONTAINER_RUNTIME_MODE_VALUE
    )
    monkeypatch.setattr(
        cui,
        "form_inputs",
        lambda: ("Igual", "Igual", date(2031, 6, 30), "sent"),
    )
    fetch_calls = []
    monkeypatch.setattr(
        cui, "fetch_recommendation", lambda *a, **k: fetch_calls.append((a, k))
    )
    cui.main()
    assert fetch_calls == []
    assert "Jugador y rival deben ser diferentes." in st_double.errors


def test_modo_activo_identificador_invalido_se_rechaza(
    monkeypatch, st_double
) -> None:
    from datetime import date

    monkeypatch.setenv(
        cui.CONTAINER_RUNTIME_MODE_ENV, cui.CONTAINER_RUNTIME_MODE_VALUE
    )
    monkeypatch.setattr(
        cui,
        "form_inputs",
        lambda: ("../etc/passwd", "RivalY", date(2031, 6, 30), "sent"),
    )
    fetch_calls = []
    monkeypatch.setattr(
        cui, "fetch_recommendation", lambda *a, **k: fetch_calls.append((a, k))
    )
    cui.main()
    assert fetch_calls == []
    assert len(st_double.errors) == 1
