"""P15: orquestador offline P10 -> records -> P14 -> P13 (P16).

Cubre: autorizacion de ejecucion real (habilitada en ``True`` para un
intento manual unico, sin reintento automatico y con metadata
contractual de razon/politica/cierre), contrato de rutas privadas,
flujo autorizado simulado solo con ejecuciones sinteticas inyectadas
(nunca la fuente real), proyeccion target -> record, sellado P10,
fallos e interrupciones, performance log cerrado sin PII, invariantes
de arquitectura por AST y equivalencia con la generacion P14 directa.
Cero ejecucion real: todos los tests inyectan runners/fixtures
sinteticos; el comportamiento bloqueado se verifica con monkeypatch
de ``REAL_EXECUTION_AUTHORIZED`` a ``False``. Ningun test invoca al
lector real de la fuente.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
from datetime import date
from pathlib import Path

import pytest

import src.analysis.tactical_recommendation_snapshot_generator as p14
import src.analysis.tactical_recommendation_snapshot_pipeline as p15
import src.analysis.tactical_recommender_pipeline as p10
import src.recommender.persisted_tactical_recommendation_provider as p13
import src.recommender.tactical_prioritization as prioritization
import test_tactical_recommender_pipeline as p10tests

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_SOURCE = Path(p15.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
# Infraestructura sintetica                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def p10_mini_result():
    """Ejecucion P10 real en memoria (sin I/O) con dos targets recíprocos."""
    target_a = p10tests._target(
        "Alice", "Bob", date(2021, 1, 1), "mm1", "player_1_vs_player_2"
    )
    target_b = p10tests._target(
        "Bob", "Alice", date(2021, 1, 1), "mm1", "player_2_vs_player_1"
    )
    return p10tests.run_tactical_recommender_pipeline(
        p10tests._points(), (target_a, target_b), p10tests._config()
    )


@pytest.fixture()
def mini_policy():
    """Politica P14 p10_offline para el universo sintetico 2/1."""
    policy = object.__new__(p14.SnapshotGenerationPolicy)
    values = dict(
        mode=p14.SNAPSHOT_GENERATION_MODE_P10_OFFLINE,
        scoring_policy=prioritization.default_tactical_scoring_policy(),
        expected_encoder_policy=p10.PRODUCTION_ENCODER_POLICY,
        expected_scope_strategy="global_only",
        expected_requested_patterns=("P02", "P04", "P05", "P06"),
        expected_serve_numbers=(1, 2),
        expected_minimum_labeled_attempts=1,
        expected_minimum_matches=1,
        expected_window_days=None,
        expected_top_k=3,
        required_entries=2,
        required_reciprocal_pairs=1,
        expected_fold_orientations=((2021, 2),),
    )
    for name, value in values.items():
        object.__setattr__(policy, name, value)
    return policy


@pytest.fixture()
def authorized(monkeypatch, mini_policy):
    """Habilita la ejecucion P15 y ancla P14 a la politica sintetica."""
    monkeypatch.setattr(p15, "REAL_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        p14, "validate_snapshot_generation_policy", lambda _p: None
    )
    monkeypatch.setattr(
        p14, "P10_OFFLINE_SNAPSHOT_GENERATION_POLICY", mini_policy
    )


@pytest.fixture()
def external_dir(tmp_path):
    """Directorio externo al repo, sin enlaces simbolicos en cadena."""
    return tmp_path.resolve()


def _raw_paths(base: Path):
    return (
        str(base / "private-snapshot.json"),
        str(base / "private-performance.json"),
    )


def _run(base: Path, **kwargs):
    snapshot_raw, log_raw = _raw_paths(base)
    return p15.run_tactical_recommendation_snapshot_pipeline(
        Path(snapshot_raw), Path(log_raw), **kwargs
    )


def _read_log(base: Path) -> dict:
    return json.loads((base / "private-performance.json").read_bytes())


class _Spy:
    def __init__(self, value=None, exc: Exception | None = None) -> None:
        self.value = value
        self.exc = exc
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.value


def _clone(source, **overrides) -> object:
    """Clon profundo de atributos sin pasar por __post_init__ P10."""
    clone = object.__new__(type(source))
    for name, value in vars(source).items():
        object.__setattr__(clone, name, overrides.get(name, value))
    return clone


# --------------------------------------------------------------------- #
# A. Autorizacion de ejecucion real                                      #
# --------------------------------------------------------------------- #


def test_autorizacion_p16_es_unica_constante_true():
    assert p15.REAL_EXECUTION_AUTHORIZED is True
    assert p15.REAL_EXECUTION_AUTHORIZATION_REASON == (
        "single_manual_private_snapshot_generation_authorized_after_preflight"
    )
    assert p15.AUTOMATIC_RETRY is False
    assert p15.SINGLE_MANUAL_EXECUTION_POLICY == (
        "single_manual_execution_without_automatic_retry"
    )
    assert p15.PREVIOUS_REAL_P15_ATTEMPTS == 0
    assert p15.COMPLETED_REAL_EXECUTIONS == 0
    assert p15.INTERRUPTED_REAL_EXECUTIONS == 0
    assert p15.AUTOMATIC_RETRIES_PERFORMED == 0
    assert isinstance(p15.REAL_SNAPSHOT_EXECUTION_BLOCK_REASON, str)
    assert p15.REAL_SNAPSHOT_EXECUTION_BLOCK_REASON
    assert isinstance(p15.POST_EXECUTION_CLOSURE_RULE, str)
    assert "REAL_EXECUTION_AUTHORIZED = False" in p15.POST_EXECUTION_CLOSURE_RULE


def test_p10_permanece_bloqueado_historicamente():
    assert p10.REAL_EXECUTION_AUTHORIZED is False
    assert p10.FURTHER_REAL_EXECUTION_AUTHORIZED is False


def test_runner_bloqueado_lanza_preflight_sin_efectos(
    external_dir, monkeypatch
):
    monkeypatch.setattr(p15, "REAL_EXECUTION_AUTHORIZED", False)
    with pytest.raises(
        p15.TacticalRecommendationSnapshotPipelineError
    ) as exc_info:
        _run(external_dir, p10_runner=lambda: None)
    assert exc_info.value.stage == "preflight"
    assert (
        exc_info.value.reason_code == "real_snapshot_generation_not_authorized"
    )
    assert list(external_dir.iterdir()) == []


def test_bloqueo_previo_a_lectores_y_generador(external_dir, monkeypatch):
    monkeypatch.setattr(p15, "REAL_EXECUTION_AUTHORIZED", False)
    p10_spy = _Spy(None)
    generator_spy = _Spy(None)
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError):
        _run(
            external_dir,
            p10_runner=p10_spy,
            snapshot_generator=generator_spy,
        )
    assert p10_spy.calls == []
    assert generator_spy.calls == []


def test_cli_bloqueada_sin_io_y_sin_bypass_por_ambiente(
    external_dir, monkeypatch
):
    monkeypatch.setattr(p15, "REAL_EXECUTION_AUTHORIZED", False)
    snapshot_raw, log_raw = _raw_paths(external_dir)
    for variable in ("P15_REAL_EXECUTION", "ALLOW_REAL", "TENNIS_REAL"):
        monkeypatch.setenv(variable, "true")
    with pytest.raises(SystemExit) as exc_info:
        p15.main(
            ["--snapshot-path", snapshot_raw, "--performance-log", log_raw]
        )
    assert exc_info.value.code == p15.REAL_SNAPSHOT_EXECUTION_BLOCK_REASON
    assert list(external_dir.iterdir()) == []


def test_ambiente_no_altera_la_autorizacion(
    authorized, external_dir, p10_mini_result, monkeypatch
):
    for variable in ("P15_REAL_EXECUTION", "ALLOW_REAL", "TENNIS_REAL"):
        monkeypatch.setenv(variable, "true")
    assert p15.REAL_EXECUTION_AUTHORIZED is True
    base = external_dir / "cli"
    base.mkdir()
    snapshot_raw, log_raw = _raw_paths(base)
    completed = _run(base, p10_runner=lambda: p10_mini_result)
    assert completed.execution_status == "completed"
    # Con la puerta forzada a False, el entorno no la re-habilita.
    monkeypatch.setattr(p15, "REAL_EXECUTION_AUTHORIZED", False)
    with pytest.raises(SystemExit) as exc_info:
        p15.main(
            ["--snapshot-path", snapshot_raw, "--performance-log", log_raw]
        )
    assert exc_info.value.code == p15.REAL_SNAPSHOT_EXECUTION_BLOCK_REASON


def test_cli_rechaza_fuente_alternativa(authorized, external_dir):
    snapshot_raw, log_raw = _raw_paths(external_dir)
    for extra in (
        ("--source", "/tmp/otra-fuente.parquet"),
        ("--points", str(p10.POINTS_PATH)),
    ):
        with pytest.raises(SystemExit) as exc_info:
            p15.main(
                [
                    "--snapshot-path",
                    snapshot_raw,
                    "--performance-log",
                    log_raw,
                    extra[0],
                    extra[1],
                ]
            )
        assert exc_info.value.code == 2


def test_cli_valida_rutas_antes_de_llegar_a_ejecutar(
    authorized, external_dir, monkeypatch
):
    called: list = []
    (external_dir / "ya-existente.json").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        p15,
        "run_tactical_recommendation_snapshot_pipeline",
        lambda _s, _l: called.append(1),
    )
    with pytest.raises(SystemExit) as exc_info:
        p15.main(
            [
                "--snapshot-path",
                str(external_dir / "ya-existente.json"),
                "--performance-log",
                str(external_dir / "private-performance.json"),
            ]
        )
    assert (
        exc_info.value.code
        == p15._ERROR_MESSAGES["path_contract_violation"]
    )
    assert called == []


_SNAPSHOT_FINAL = (
    "/Users/omar.errandi/Documents/tactical-recommendations-private-v1.json"
)
_LOG_FINAL = "/Users/omar.errandi/Documents/p16-snapshot-performance.json"


def test_rutas_definitivas_cuentan_con_el_contrato_p15():
    assert p15._is_posix_private_path(_SNAPSHOT_FINAL)
    assert p15._is_posix_private_path(_LOG_FINAL)
    documents = Path(_SNAPSHOT_FINAL).parent
    if not documents.is_dir():
        pytest.skip("el destino definitivo no existe en esta maquina")
    snapshot, log = p15.validate_snapshot_pipeline_paths_cli(
        _SNAPSHOT_FINAL, _LOG_FINAL
    )
    assert snapshot == Path(_SNAPSHOT_FINAL)
    assert log == Path(_LOG_FINAL)
    assert not Path(_SNAPSHOT_FINAL).exists()
    assert not Path(_LOG_FINAL).exists()
    assert os.access(documents, os.W_OK | os.X_OK)


def test_cli_no_ofrece_banderas_de_autorizacion(authorized, external_dir):
    snapshot_raw, log_raw = _raw_paths(external_dir)
    with pytest.raises(SystemExit) as exc_info:
        p15.main(
            [
                "--snapshot-path",
                snapshot_raw,
                "--performance-log",
                log_raw,
                "--authorize-real",
                "true",
            ]
        )
    assert exc_info.value.code == 2


def test_ast_autorizacion_unica_constante_true_sin_ambiente():
    tree = ast.parse(_MODULE_SOURCE)
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            if node.target.id == "REAL_EXECUTION_AUTHORIZED":
                assignments.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "REAL_EXECUTION_AUTHORIZED"
                ):
                    assignments.append(node)
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Constant) and value.value is True
    environment = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
        )
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getenv"
        )
    ]
    assert environment == []


# --------------------------------------------------------------------- #
# B. Contrato de rutas privadas                                          #
# --------------------------------------------------------------------- #


def test_rutas_validas_externas_devuelven_pare(external_dir):
    snapshot_raw, log_raw = _raw_paths(external_dir)
    snapshot, log = p15.validate_snapshot_pipeline_paths_cli(
        snapshot_raw, log_raw
    )
    assert snapshot == Path(snapshot_raw)
    assert log == Path(log_raw)


def test_no_existe_log_p10_derivado_en_el_contrato_p15(external_dir):
    # El log derivado P10 dejio de existir: un fichero con el antiguo
    # prenombre en el directorio externo no bloquea la validacion.
    (external_dir / "private-performance.p10-pipeline.json").write_text(
        "x", encoding="utf-8"
    )
    snapshot_raw, log_raw = _raw_paths(external_dir)
    snapshot, log = p15.validate_snapshot_pipeline_paths_cli(
        snapshot_raw, log_raw
    )
    assert snapshot.name == "private-snapshot.json"
    assert log.name == "private-performance.json"
    module_source = Path(p15.__file__).read_text(encoding="utf-8")
    assert "_derived_p10_log_path" not in module_source
    assert ".p10-pipeline" not in module_source


@pytest.mark.parametrize(
    "snapshot_raw, log_raw",
    [
        pytest.param(
            str(p10.ROOT / "private-snapshot.json"),
            str(Path("/tmp") / "private-performance.json"),
            id="dentro-del-repositorio",
        ),
        pytest.param(
            str(Path("/tmp") / "a.json"), str(Path("/tmp") / "a.json"),
            id="mismo-destino",
        ),
        pytest.param(
            str(Path("/tmp") / "file:///tmp/x.json"),
            str(Path("/tmp") / "y.json"),
            id="uri-esquema",
        ),
        pytest.param(
            "/tmp/../etc/private.json",
            str(Path("/tmp") / "y.json"),
            id="trayectoria",
        ),
        pytest.param(
            "C:\\tmp\\private.json",
            str(Path("/tmp") / "y.json"),
            id="ruta-windows",
        ),
        pytest.param(
            "/tmp/aco\\bdario.json",
            str(Path("/tmp") / "y.json"),
            id="backslash",
        ),
        pytest.param(
            "/tmp/aco:bulario.json",
            str(Path("/tmp") / "y.json"),
            id="dos-puntos",
        ),
        pytest.param(
            "/tmp/aco\x00mbulario.json",
            str(Path("/tmp") / "y.json"),
            id="nulo",
        ),
        pytest.param(
            "/tmp/aco\tmbulario.json",
            str(Path("/tmp") / "y.json"),
            id="control",
        ),
        pytest.param(
            "/~/home/user/x.json",
            str(Path("/tmp") / "y.json"),
            id="tilde",
        ),
        pytest.param(
            "tmp/privado.json", str(Path("/tmp") / "y.json"),
            id="relative",
        ),
        pytest.param(
            "", str(Path("/tmp") / "y.json"), id="vacia",
        ),
        pytest.param(
            "/a" * 300 + ".json",
            str(Path("/tmp") / "y.json"),
            id="demasiado-larga",
        ),
        pytest.param(
            "/tmp/./privado.json",
            str(Path("/tmp") / "y.json"),
            id="componente-punto",
        ),
        pytest.param(
            "/no-existe-p15-directorio/privado.json",
            str(Path("/tmp") / "y.json"),
            id="padre-inexistente",
        ),
        pytest.param(
            12, str(Path("/tmp") / "y.json"), id="no-string",
        ),
    ],
)
def test_rutas_fuera_de_contrato_se_rechazan(
    external_dir, snapshot_raw, log_raw
):
    base_snapshot = snapshot_raw
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError) as exc:
        p15.validate_snapshot_pipeline_paths_cli(
            snapshot_raw, log_raw
        )
    assert exc.value.stage == "preflight"
    assert exc.value.reason_code == "path_contract_violation"
    message = str(exc.value)
    if isinstance(base_snapshot, str) and base_snapshot:
        assert base_snapshot not in message
    assert str(external_dir) not in message


def test_destino_existente_o_directorio_se_rechaza(external_dir):
    (external_dir / "ya-existe.json").write_text("x", encoding="utf-8")
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError):
        p15.validate_snapshot_pipeline_paths_cli(
            str(external_dir / "ya-existe.json"),
            str(external_dir / "private-performance.json"),
        )
    (external_dir / "sub").mkdir()
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError):
        p15.validate_snapshot_pipeline_paths_cli(
            str(external_dir / "sub"),
            str(external_dir / "private-performance.json"),
        )


def test_enlace_simbolico_en_ancestros_se_rechaza(external_dir):
    real = external_dir / "real"
    real.mkdir()
    link = external_dir / "alias"
    link.symlink_to(real)
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError):
        p15.validate_snapshot_pipeline_paths_cli(
            str(link / "private.json"),
            str(external_dir / "private-performance.json"),
        )


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            "C:\\Users\\omar.errandi\\privado\\snapshot.json", id="unidad"
        ),
        pytest.param(
            "C:/Users/omar.errandi/privado/snapshot.json", id="unidad-barra"
        ),
        pytest.param(
            "c:\\tmp\\snapshot.json", id="unidad-minuscula"
        ),
        pytest.param(
            "\\\\srv01\\share\\compartida\\snapshot.json", id="unc"
        ),
    ],
)
def test_ruta_windows_textual_valida(raw):
    assert p15._is_windows_private_path(raw)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            "C:\\Users\\..\\privado\\snapshot.json", id="traversal"
        ),
        pytest.param(
            "C:\\Users\\.\\privado\\snapshot.json", id="punto"
        ),
        pytest.param(
            "relativo\\privado\\snapshot.json", id="relative"
        ),
        pytest.param("C:relative.json", id="drive-relative"),
        pytest.param("file:///C:/privado/snapshot.json", id="uri"),
        pytest.param("https://host/privado.json", id="https"),
        pytest.param("C:\\", id="raiz-sin-componente"),
        pytest.param("C:\\a b.json", id="espacio"),
        pytest.param("C:\\caf\xe9.json", id="no-ascii"),
        pytest.param("C:\\a\x00b.json", id="nulo"),
        pytest.param("~\\usuario\\snapshot.json", id="tilde"),
        pytest.param("", id="vacia"),
        pytest.param("C:\\" + "a" * 4096 + ".json", id="demasiado-larga"),
        pytest.param(12, id="no-string"),
    ],
)
def test_ruta_windows_textual_rechaza(raw):
    assert not p15._is_windows_private_path(raw)


def test_validador_posix_mantiene_el_contrato_original():
    assert p15._is_posix_private_path("/tmp/privado/snapshot.json")
    assert not p15._is_posix_private_path(
        "C:\\Users\\omar.errandi\\privado\\snapshot.json"
    )
    assert not p15._is_posix_private_path("/tmp/../privado.json")
    assert not p15._is_posix_private_path("file:///tmp/x.json")


def test_validacion_nativa_usa_semantica_del_sistema_actual(external_dir):
    import os

    snapshot_raw, log_raw = _raw_paths(external_dir)
    snapshot, log = p15.validate_snapshot_pipeline_paths_cli(
        snapshot_raw, log_raw
    )
    assert snapshot == Path(snapshot_raw)
    assert log == Path(log_raw)
    if os.name == "nt":
        assert p15._is_native_private_path("C:\\tmp\\x.json")
        assert not p15._is_native_private_path("/tmp/x.json")
    else:
        assert p15._is_native_private_path("/tmp/x.json")
        assert not p15._is_native_private_path("C:\\tmp\\x.json")


@pytest.mark.parametrize("bad_value", [None, 12, b"/tmp/a.json", object()])
def test_runner_reyecta_rutas_no_path(authorized, external_dir, bad_value):
    snapshot_raw, log_raw = _raw_paths(external_dir)
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError) as exc:
        p15.run_tactical_recommendation_snapshot_pipeline(
            bad_value, Path(log_raw), p10_runner=lambda: None
        )
    assert exc.value.reason_code == "path_contract_violation"
    assert list(external_dir.iterdir()) == []


# --------------------------------------------------------------------- #
# C. Flujo autorizado simulado (solo sintesis inyectada)                 #
# --------------------------------------------------------------------- #


def test_flujo_completo_con_p14_real(
    authorized, external_dir, p10_mini_result
):
    p10_spy = _Spy(p10_mini_result)
    result = _run(
        external_dir,
        p10_runner=p10_spy,
    )
    assert len(p10_spy.calls) == 1
    assert p10_spy.calls[0] == ((), {})
    assert result.execution_status == "completed"
    assert result.failed_stage is None
    assert result.reason_codes == ()
    assert result.completed_stages == p15.PIPELINE_STAGES
    assert tuple(
        stage for stage, _seconds in result.stage_seconds
    ) == p15.PIPELINE_STAGES
    assert all(
        type(seconds) is float and seconds >= 0.0
        for _stage, seconds in result.stage_seconds
    )
    assert result.targets_total == 2
    assert result.targets_processed == 2
    assert result.records_built == 2
    assert result.snapshot_entries == 2
    assert result.snapshot_bytes > 0
    assert len(result.snapshot_fingerprint) == 64
    assert all(c in "0123456789ABCDEF" for c in result.snapshot_fingerprint)
    assert dict(result.operation_counters) == {
        "p10_executions": 1,
        "targets_projected": 2,
        "generation_calls": 1,
        "persistence_calls": 1,
        "verification_calls": 1,
    }
    assert dict(result.reconciliations) == {
        "p10_result_validated": True,
        "targets_projected": True,
        "p14_universe_reconciled": True,
        "snapshot_persisted_and_verified": True,
    }
    assert type(result.generation_diagnostics) is p14.SnapshotGenerationDiagnostics
    snapshot = external_dir / "private-snapshot.json"
    assert snapshot.exists()
    payload = _read_log(external_dir)
    assert payload["execution_status"] == "completed"
    assert payload["completed_stages"] == list(p15.PIPELINE_STAGES)


def test_llamadas_exactas_y_identidad_de_records(
    authorized, external_dir, p10_mini_result, mini_policy
):
    captured: dict = {}

    def _generador(records, path, *, policy):
        captured["records"] = records
        captured["path"] = path
        captured["policy"] = policy
        return p14.generate_and_persist_tactical_recommendation_snapshot(
            records, path, policy=policy
        )

    result = _run(
        external_dir,
        p10_runner=lambda: p10_mini_result,
        snapshot_generator=_generador,
    )
    assert result.execution_status == "completed"
    assert captured["path"] == external_dir / "private-snapshot.json"
    assert captured["policy"] is mini_policy
    records = captured["records"]
    assert type(records) is tuple
    assert len(records) == 2
    for item, record in zip(p10_mini_result.target_results, records):
        assert type(record) is p14.P10OfflineSnapshotRecord
        assert record.target_match_id == item.target.target_match_id
        assert record.fold == item.target.fold
        assert record.orientation == item.target.orientation
        assert record.prioritization is item.prioritization


def test_solo_la_autorizacion_p15_gobierna_la_ruta_productiva(
    authorized, external_dir, p10_mini_result, monkeypatch
):
    compute_calls: list[tuple[tuple, dict]] = []
    publisher_calls: list = []

    def compute_spy(*args, **kwargs):
        compute_calls.append((args, kwargs))
        return p10_mini_result

    monkeypatch.setattr(
        p10, "compute_tactical_pipeline_result", compute_spy
    )
    monkeypatch.setattr(
        p10,
        "publish_tactical_pipeline_artifacts",
        lambda *_a, **_k: publisher_calls.append(1),
    )
    base = external_dir / "prod"
    base.mkdir()
    result = p15.run_tactical_recommendation_snapshot_pipeline(
        base / "private-snapshot.json",
        base / "private-performance.json",
    )
    assert result.execution_status == "completed"
    # Una llamada a la frontera compute-only P10, con la fuente
    # contractual y sin dependencias adicionales.
    assert len(compute_calls) == 1
    assert compute_calls[0] == ((p10.POINTS_PATH,), {})
    # Cero publisher P10, cero log P10 derivado, cero artefactos P10.
    assert publisher_calls == []
    files = [name for name in base.iterdir()]
    assert not any("p10-pipeline" in name.name for name in files)
    assert {name.name for name in files} == {
        "private-snapshot.json",
        "private-performance.json",
    }
    # Las constantes historicas P10 permanecen falsas e intactas.
    assert p10.REAL_EXECUTION_AUTHORIZED is False
    assert p10.FURTHER_REAL_EXECUTION_AUTHORIZED is False


def test_resultado_no_lleva_rutas_de_privacidad(
    authorized, external_dir, p10_mini_result
):
    result = _run(external_dir, p10_runner=lambda: p10_mini_result)
    blob = json.dumps(
        {
            field.name: getattr(result, field.name)
            for field in dataclasses.fields(result)
        },
        default=repr,
    )
    blob += json.dumps(
        result.generation_diagnostics.__dict__, default=repr
    )
    assert str(external_dir) not in blob
    assert "private-snapshot.json" not in blob
    assert "private-performance.json" not in blob


def test_orden_de_records_no_altera_snapshot(authorized, p10_mini_result):
    import tempfile

    records = p15.project_tactical_target_records(p10_mini_result)
    outputs = []
    for order in (records, records[::-1]):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve()
            direct = p14.generate_and_persist_tactical_recommendation_snapshot(
                order,
                base / "directo.json",
                policy=p14.P10_OFFLINE_SNAPSHOT_GENERATION_POLICY,
            )
            outputs.append(
                (
                    direct,
                    (base / "directo.json").read_bytes(),
                )
            )
    assert outputs[0][0].snapshot_fingerprint == outputs[1][0].snapshot_fingerprint
    assert outputs[0][0].serialized_bytes == outputs[1][0].serialized_bytes
    assert outputs[0][1] == outputs[1][1]


def test_reloj_inyectado_es_determinista(authorized, p10_mini_result):
    import tempfile

    def _reloj_paso():
        ticks = iter(range(200))
        return lambda: float(next(ticks))

    results = []
    for _index in range(2):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve()
            result = p15.run_tactical_recommendation_snapshot_pipeline(
                base / "private-snapshot.json",
                base / "private-performance.json",
                p10_runner=lambda: p10_mini_result,
                clock=_reloj_paso(),
            )
            results.append((result, _read_log(base)))
    first_result, first_payload = results[0]
    second_result, second_payload = results[1]
    assert first_result.execution_status == "completed"
    assert first_result.stage_seconds == second_result.stage_seconds
    assert all(
        type(seconds) is float
        and seconds == seconds
        and 0.0 <= seconds < float("inf")
        for _stage, seconds in first_result.stage_seconds
    )
    assert first_payload["elapsed_seconds"] == second_payload[
        "elapsed_seconds"
    ]
    assert first_payload["stage_seconds"] == second_payload[
        "stage_seconds"
    ]


def test_cli_autorizada_mapea_codigos_de_salida(
    authorized, external_dir, p10_mini_result, monkeypatch
):
    def _sub(name: str) -> Path:
        base = external_dir / name
        base.mkdir()
        return base

    interrupted = _run(
        _sub("interrupted"),
        p10_runner=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    failed = _run(
        _sub("failed"),
        p10_runner=lambda: (_ for _ in ()).throw(
            p10.TacticalPipelineExecutionError("source_validation", "x")
        ),
    )
    completed = _run(
        _sub("completed"), p10_runner=lambda: p10_mini_result
    )
    assert interrupted.execution_status == "interrupted"
    assert failed.execution_status == "failed"
    assert completed.execution_status == "completed"
    fresh_base = external_dir / "cli"
    fresh_base.mkdir()
    snapshot_raw, log_raw = _raw_paths(fresh_base)
    arguments = ["--snapshot-path", snapshot_raw, "--performance-log", log_raw]
    with monkeypatch.context() as ctx:
        ctx.setattr(
            p15,
            "run_tactical_recommendation_snapshot_pipeline",
            lambda _s, _l: interrupted,
        )
        with pytest.raises(SystemExit) as exc_info:
            p15.main(arguments)
        assert exc_info.value.code == 130
    with monkeypatch.context() as ctx:
        ctx.setattr(
            p15,
            "run_tactical_recommendation_snapshot_pipeline",
            lambda _s, _l: failed,
        )
        with pytest.raises(SystemExit) as exc_info:
            p15.main(arguments)
        assert exc_info.value.code == 1
    with monkeypatch.context() as ctx:
        ctx.setattr(
            p15,
            "run_tactical_recommendation_snapshot_pipeline",
            lambda _s, _l: completed,
        )
        assert p15.main(arguments) is None


# --------------------------------------------------------------------- #
# D. Proyeccion target -> record                                         #
# --------------------------------------------------------------------- #


def test_proyeccion_copia_campos_explicitos(p10_mini_result):
    records = p15.project_tactical_target_records(p10_mini_result)
    assert type(records) is tuple
    assert len(records) == len(p10_mini_result.target_results)
    for item, record in zip(p10_mini_result.target_results, records):
        assert type(record) is p14.P10OfflineSnapshotRecord
        assert record.target_match_id == item.target.target_match_id
        assert record.fold == item.target.fold
        assert record.orientation == item.target.orientation
        assert record.prioritization is item.prioritization


def test_proyeccion_reyecta_tipo_distinto():
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError) as exc:
        p15.project_tactical_target_records("no-resultado")
    assert exc.value.stage == "target_projection"
    assert exc.value.reason_code == "target_projection_failed"


def test_proyeccion_sanea_campo_bloqueado(p10_mini_result):
    item = p10_mini_result.target_results[0]
    broken = _clone(
        item,
        target=_clone(item.target, target_match_id=""),
    )
    tampered = _clone(p10_mini_result, target_results=(broken,))
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError) as exc:
        p15.project_tactical_target_records(tampered)
    assert exc.value.reason_code == "target_projection_failed"
    assert "mm1" not in str(exc.value)


def test_proyeccion_emite_progreso_por_paso(p10_mini_result, monkeypatch):
    monkeypatch.setattr(p15, "_PROGRESS_STEP", 1)
    progress: list[tuple[int, int, int]] = []
    records = p15.project_tactical_target_records(
        p10_mini_result,
        progress_callback=lambda total, done, built: progress.append(
            (total, done, built)
        ),
    )
    assert len(records) == 2
    assert progress == [(2, 1, 1), (2, 2, 2)]


# --------------------------------------------------------------------- #
# E. Sellado P10 (defensa en profundidad)                                #
# --------------------------------------------------------------------- #


def test_sello_valido_pasa_comprobacion(p10_mini_result):
    p15._check_p10_seal_metadata(p10_mini_result)


def test_sello_violado_en_frontera_p10(
    authorized, external_dir, p10_mini_result
):
    seal = p10_mini_result.test_seal
    tampered = _clone(
        p10_mini_result,
        test_seal=_clone(seal, used_for_method_selection=True),
    )
    result = _run(external_dir, p10_runner=lambda: tampered)
    assert result.execution_status == "failed"
    assert result.failed_stage == "p10_pipeline"
    assert result.reason_codes == ("p10_result_invalid",)
    assert not (external_dir / "private-snapshot.json").exists()


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda source: _clone(
                source,
                test_seal=_clone(source.test_seal, test_status="open"),
            ),
            id="estado-abierto",
        ),
        pytest.param(
            lambda source: _clone(
                source,
                test_seal=_clone(
                    source.test_seal, used_for_method_selection=True
                ),
            ),
            id="usado-en-metodo",
        ),
        pytest.param(
            lambda source: _clone(
                source,
                test_seal=_clone(
                    source.test_seal,
                    counters=_bump_counter(source.test_seal.counters),
                ),
            ),
            id="contador-no-cero",
        ),
        pytest.param(
            lambda source: _clone(
                source,
                leakage_audit=_bump_first_field(source.leakage_audit),
            ),
            id="fuga-no-cero",
        ),
        pytest.param(
            lambda source: _clone(
                source,
                config=_clone(
                    source.config,
                    development_cutoff=date(2020, 6, 30),
                ),
            ),
            id="corte-movido",
        ),
        pytest.param(
            lambda source: _replace_target(
                source, as_of_date=date(2024, 1, 1)
            ),
            id="target-sellado",
        ),
        pytest.param(
            lambda source: _replace_target(source, fold="validation_1999"),
            id="fold-desconocido",
        ),
        pytest.param(
            lambda source: _replace_target(
                source, orientation="player_1_vs_player_3"
            ),
            id="orientacion-invalida",
        ),
    ],
)
def test_comprobacion_de_sello_lanza_violacion(p10_mini_result, tamper):
    with pytest.raises(p15.TacticalRecommendationSnapshotPipelineError) as exc:
        p15._check_p10_seal_metadata(tamper(p10_mini_result))
    assert exc.value.stage == "p10_pipeline"
    assert exc.value.reason_code == "p10_seal_violation"


def _bump_counter(counters: tuple[tuple[str, int], ...]):
    items = list(counters)
    items[0] = (items[0][0], 1)
    return tuple(items)


def _bump_first_field(record):
    first = dataclasses.fields(record)[0].name
    return _clone(record, **{first: 1})


def _replace_target(source, **target_fields):
    item = source.target_results[0]
    replaced = _clone(
        item, target=_clone(item.target, **target_fields)
    )
    return _clone(source, target_results=(replaced,))


# --------------------------------------------------------------------- #
# F. Fallos, interrupciones y sin reintento                              #
# --------------------------------------------------------------------- #


def test_fallo_p10_ejecucion_clasificado(
    authorized, external_dir, p10_mini_result
):
    spy = _Spy(
        None,
        exc=p10.TacticalPipelineExecutionError(
            "source_validation", "synthetic"
        ),
    )
    result = _run(external_dir, p10_runner=spy)
    assert result.execution_status == "failed"
    assert result.failed_stage == "p10_pipeline"
    assert result.reason_codes == ("p10_pipeline_execution_failed",)
    assert dict(result.operation_counters)["p10_executions"] == 0
    assert dict(result.reconciliations) == {
        key: False for key in p15.PIPELINE_RECONCILIATION_KEYS
    }
    assert not (external_dir / "private-snapshot.json").exists()
    payload = _read_log(external_dir)
    assert payload["execution_status"] == "failed"
    assert payload["reason_code"] == "p10_pipeline_execution_failed"
    assert payload["completed_stages"] == ["preflight"]


def test_fallo_p10_contract_clasificado(
    authorized, external_dir
):
    spy = _Spy(None, exc=p10.TacticalPipelineContractError("x"))
    result = _run(external_dir, p10_runner=spy)
    assert (result.execution_status, result.failed_stage) == (
        "failed",
        "p10_pipeline",
    )
    assert result.reason_codes == ("p10_pipeline_execution_failed",)


def test_cero_reintento_despues_de_fallo(authorized, external_dir):
    spy = _Spy(
        None,
        exc=p10.TacticalPipelineExecutionError("source_validation", "x"),
    )
    result = _run(external_dir, p10_runner=spy)
    assert result.execution_status == "failed"
    assert len(spy.calls) == 1


def test_runner_devuelve_no_pipeline_result(
    authorized, external_dir
):
    for position, wrong in enumerate((None, "snapshot", 41, ())):
        base = external_dir / f"case-{position}"
        base.mkdir()
        result = _run(base, p10_runner=lambda w=wrong: w)
        assert result.execution_status == "failed"
        assert result.failed_stage == "p10_pipeline"
        assert result.reason_codes == ("p10_result_not_pipeline_result",)


def test_resultado_p10_invalido_por_contrato(
    authorized, external_dir, p10_mini_result
):
    tampered = dataclasses.replace(
        p10_mini_result, contract_version="9.9.9"
    )
    result = _run(external_dir, p10_runner=lambda: tampered)
    assert result.execution_status == "failed"
    assert result.failed_stage == "p10_pipeline"
    assert result.reason_codes == ("p10_result_invalid",)


def test_generador_devuelve_tipo_distinto(
    authorized, external_dir, p10_mini_result
):
    result = _run(
        external_dir,
        p10_runner=lambda: p10_mini_result,
        snapshot_generator=lambda *_a, **_k: "no-snapshot",
    )
    assert result.execution_status == "failed"
    assert result.failed_stage == "snapshot_generation"
    assert result.reason_codes == ("p15_execution_error",)
    assert not (external_dir / "private-snapshot.json").exists()


def test_generador_falla_sin_snapshot(
    authorized, external_dir, p10_mini_result
):
    calls: list = []

    def _generador_fallido(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("explota con detalle no publicable")

    result = _run(
        external_dir,
        p10_runner=lambda: p10_mini_result,
        snapshot_generator=_generador_fallido,
    )
    assert len(calls) == 1
    assert result.execution_status == "failed"
    assert result.failed_stage == "snapshot_generation"
    assert result.reason_codes == ("p15_execution_error",)
    assert "explota" not in str(result)
    assert not (external_dir / "private-snapshot.json").exists()


def test_error_p14_generacion_clasificado(
    authorized, external_dir, p10_mini_result
):
    result = _run(
        external_dir,
        p10_runner=lambda: p10_mini_result,
        snapshot_generator=lambda *_a, **_k: (
            _raise(p14.TacticalRecommendationSnapshotGenerationError(
                "boom", reason_code="synthetic_failure"
            ))
        ),
    )
    assert result.execution_status == "failed"
    assert result.failed_stage == "snapshot_generation"
    assert result.reason_codes == ("snapshot_generation_failed",)


def test_error_p14_verificacion_mismatch(
    authorized, external_dir, p10_mini_result
):
    result = _run(
        external_dir,
        p10_runner=lambda: p10_mini_result,
        snapshot_generator=lambda *_a, **_k: (
            _raise(
                p14.TacticalRecommendationSnapshotGenerationError(
                    "mismatch",
                    reason_code="p13_verification_mismatch",
                )
            )
        ),
    )
    assert result.execution_status == "failed"
    assert result.failed_stage == "verification"
    assert result.reason_codes == ("verification_failed",)
    assert "snapshot_generation" in result.completed_stages
    assert "persistence" in result.completed_stages


def test_error_p13_persistencia_clasificado(
    authorized, external_dir, p10_mini_result
):
    result = _run(
        external_dir,
        p10_runner=lambda: p10_mini_result,
        snapshot_generator=lambda *_a, **_k: (
            _raise(p13.SnapshotUnavailableError())
        ),
    )
    assert result.execution_status == "failed"
    assert result.failed_stage == "persistence"
    assert result.reason_codes == ("persistence_failed",)
    assert "snapshot_generation" in result.completed_stages
    assert not (external_dir / "private-snapshot.json").exists()


def _raise(exc):
    raise exc


@pytest.mark.parametrize(
    "where",
    ["p10_pipeline", "target_projection", "snapshot_generation"],
)
def test_interrupcion_manual_por_fase(
    authorized, external_dir, p10_mini_result, where, monkeypatch
):
    if where == "p10_pipeline":
        result = _run(
            external_dir,
            p10_runner=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    elif where == "target_projection":
        monkeypatch.setattr(
            p15,
            "project_tactical_target_records",
            lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        result = _run(external_dir, p10_runner=lambda: p10_mini_result)
    else:
        result = _run(
            external_dir,
            p10_runner=lambda: p10_mini_result,
            snapshot_generator=lambda *_a, **_k: (
                (_ for _ in ()).throw(KeyboardInterrupt())
            ),
        )
    assert result.execution_status == "interrupted"
    assert result.failed_stage == where
    assert result.reason_codes == ("manual_interrupt",)
    payload = _read_log(external_dir)
    assert payload["execution_status"] == "interrupted"
    assert payload["reason_code"] == "manual_interrupt"
    assert payload["completed_stages"] == [
        stage for stage in p15.PIPELINE_STAGES if stage != where
    ][: p15.PIPELINE_STAGES.index(where)]


def test_error_p15_mensajes_cerrados():
    error = p15.TacticalRecommendationSnapshotPipelineError(
        "snapshot_generation", "snapshot_generation_failed"
    )
    assert str(error) == "Generacion P14 fallida; sin snapshot."
    assert error.stage == "snapshot_generation"
    fallback = p15.TacticalRecommendationSnapshotPipelineError(
        "etapa-inexistente", "motivo-inexistente"
    )
    assert fallback.stage == "preflight"
    assert fallback.reason_code == "p15_execution_error"


def test_validador_de_resultado_rechaza_manipulaciones():
    diagnostics = p14.SnapshotGenerationDiagnostics(
        elements_inspected=2,
        results_validated=2,
        entries_built=2,
        builder_calls=2,
        serializations=2,
        persistences=1,
        verifications=1,
        canonical_bytes=128,
        estimated_universe_bytes=128,
    )

    def _completado(**overrides):
        values = dict(
            contract=p15.PIPELINE_CONTRACT_NAME,
            schema_version=p15.PIPELINE_SCHEMA_VERSION,
            execution_status="completed",
            failed_stage=None,
            reason_codes=(),
            completed_stages=p15.PIPELINE_STAGES,
            stage_seconds=tuple(
                (stage, 0.0) for stage in p15.PIPELINE_STAGES
            ),
            targets_total=2,
            targets_processed=2,
            records_built=2,
            snapshot_entries=2,
            snapshot_bytes=128,
            operation_counters=(
                ("p10_executions", 1),
                ("targets_projected", 2),
                ("generation_calls", 1),
                ("persistence_calls", 1),
                ("verification_calls", 1),
            ),
            reconciliations=tuple(
                (key, True) for key in p15.PIPELINE_RECONCILIATION_KEYS
            ),
            snapshot_fingerprint="A" * 64,
            generation_diagnostics=diagnostics,
        )
        values.update(overrides)
        return values

    result = p15.TacticalRecommendationSnapshotPipelineResult(
        **_completado()
    )
    p15.validate_tactical_recommendation_snapshot_pipeline_result(result)

    class _Subclass(p15.TacticalRecommendationSnapshotPipelineResult):
        pass

    with pytest.raises(TypeError):
        p15.validate_tactical_recommendation_snapshot_pipeline_result(
            _Subclass(**{f.name: getattr(result, f.name) for f in dataclasses.fields(result)})
        )
    with pytest.raises(ValueError):
        p15.TacticalRecommendationSnapshotPipelineResult(
            **_completado(snapshot_fingerprint="a" * 63)
        )
    with pytest.raises(ValueError):
        p15.TacticalRecommendationSnapshotPipelineResult(
            **_completado(snapshot_fingerprint=None)
        )
    with pytest.raises(ValueError):
        p15.TacticalRecommendationSnapshotPipelineResult(
            **_completado(
                execution_status="failed",
                failed_stage="persistence",
                reason_codes=("persistence_failed",),
                completed_stages=p15.PIPELINE_STAGES[:4],
                snapshot_fingerprint=None,
                generation_diagnostics=None,
                snapshot_entries=0,
                snapshot_bytes=0,
                operation_counters=(
                    ("p10_executions", 1),
                    ("targets_projected", 2),
                    ("generation_calls", 1),
                    ("persistence_calls", 1),
                    ("verification_calls", 0),
                ),
            )
        )


# --------------------------------------------------------------------- #
# G. Performance log agregado, atomico y sin PII                         #
# --------------------------------------------------------------------- #


def _valid_payload(**overrides) -> dict:
    payload: dict = {
        "analysis_name": p15.PIPELINE_ANALYSIS_NAME,
        "execution_status": "completed",
        "current_stage": "verification",
        "completed_stages": list(p15.PIPELINE_STAGES),
        "elapsed_seconds": 1.5,
        "stage_seconds": dict.fromkeys(p15.PIPELINE_STAGES, 0.1),
        "targets_total": 2,
        "targets_processed": 2,
        "records_built": 2,
        "snapshot_entries": 2,
        "snapshot_bytes": 10,
        "operation_counters": dict.fromkeys(
            p15.PIPELINE_OPERATION_COUNTER_FIELDS, 1
        ),
    }
    payload.update(overrides)
    return payload


def test_log_serializacion_canonica_y_idempotente():
    payload = _valid_payload()
    encoded = p15._performance_log_bytes(payload)
    assert isinstance(encoded, bytes)
    decoded = json.loads(encoded)
    assert decoded == payload
    redumped = json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert redumped == encoded


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"execution_status": "exploded"}, id="estado-abierto"
        ),
        pytest.param(
            {"analysis_name": "otro-analysis"}, id="analysis-abierto"
        ),
        pytest.param(
            {"elapsed_seconds": float("nan")}, id="nan"
        ),
        pytest.param(
            {"targets_processed": 3}, id="progreso-inconsistente"
        ),
        pytest.param(
            {"records_built": 3}, id="proyeccion-inconsistente"
        ),
        pytest.param(
            {"extra_clave": 1}, id="clave-extra"
        ),
        pytest.param(
            {
                "operation_counters": dict.fromkeys(
                    (
                        "generation_calls",
                        "p10_executions",
                        "persistence_calls",
                        "targets_projected",
                        "verification_calls",
                    ),
                    0,
                )
            },
            id="contadores-desordenados",
        ),
        pytest.param(
            {"completed_stages": ["persistence", "preflight"]},
            id="completado-desordenado",
        ),
        pytest.param(
            {"current_stage": "verificacion"}, id="fase-abierta"
        ),
        pytest.param(
            {"reason_code": "motivo-inexistente"},
            id="motivo-abierto",
        ),
    ],
)
def test_log_rechaza_payloads_fuera_de_cierre(overrides):
    payload = _valid_payload(**overrides)
    with pytest.raises(ValueError):
        p15._performance_log_bytes(payload)


def test_log_payload_de_ejecucion_real(
    authorized, external_dir, p10_mini_result
):
    _run(external_dir, p10_runner=lambda: p10_mini_result)
    raw = (external_dir / "private-performance.json").read_bytes()
    payload = json.loads(raw)
    assert set(payload) == {
        "analysis_name",
        "execution_status",
        "current_stage",
        "completed_stages",
        "elapsed_seconds",
        "stage_seconds",
        "targets_total",
        "targets_processed",
        "records_built",
        "snapshot_entries",
        "snapshot_bytes",
        "operation_counters",
    }
    assert payload["analysis_name"] == p15.PIPELINE_ANALYSIS_NAME
    assert payload["execution_status"] == "completed"
    assert payload["current_stage"] == "verification"
    assert payload["completed_stages"] == list(p15.PIPELINE_STAGES)
    assert set(payload["stage_seconds"]) == set(p15.PIPELINE_STAGES)
    assert all(
        type(value) is float
        and value == value
        and 0.0 <= value < float("inf")
        for value in payload["stage_seconds"].values()
    )
    assert payload["targets_total"] == 2
    assert payload["targets_processed"] == 2
    assert payload["records_built"] == 2
    assert payload["snapshot_entries"] == 2
    assert payload["snapshot_bytes"] > 0
    assert set(payload["operation_counters"]) == set(
        p15.PIPELINE_OPERATION_COUNTER_FIELDS
    )
    assert payload["operation_counters"] == {
        "p10_executions": 1,
        "targets_projected": 2,
        "generation_calls": 1,
        "persistence_calls": 1,
        "verification_calls": 1,
    }
    assert "reason_code" not in payload
    assert b": " not in raw and b", " not in raw
    assert b"Alice" not in raw
    assert b"Bob" not in raw
    assert b"mm1" not in raw
    assert str(external_dir).encode() not in raw


def test_log_atomico_sin_residuos(tmp_path):
    base = tmp_path.resolve()
    log_writer = p15._SnapshotPipelinePerformanceLog(
        base / "private-performance.json", lambda: 0.0
    )
    log_writer.begin()
    log_writer.begin_stage("p10_pipeline")
    log_writer.fail("p10_pipeline", "p10_pipeline_execution_failed")
    leftovers = [
        entry.name for entry in base.iterdir() if entry.name.endswith(".tmp")
    ]
    assert leftovers == []
    assert (base / "private-performance.json").exists()


def test_escritora_siguen_protocolo_de_llamadas(
    authorized, p10_mini_result
):
    calls: list[tuple] = []

    class _Escritura:
        def __init__(self, path, clock) -> None:
            calls.append(("init", None))

        def bind_counters(self, counters) -> None:
            calls.append(("bind_counters", None))

        def begin(self) -> None:
            calls.append(("begin", None))

        def begin_stage(self, stage) -> None:
            calls.append(("begin_stage", stage))

        def complete_stage(self, stage, seconds) -> None:
            calls.append(("complete_stage", stage))

        def progress(self, stage, **kwargs) -> None:
            calls.append(("progress", stage))
            calls.append(("progress_kwargs", tuple(sorted(kwargs.items()))))

        def set_snapshot_metrics(self, entries, size) -> None:
            calls.append(("set_snapshot_metrics", (entries, size)))

        def complete(self) -> None:
            calls.append(("complete", None))

        def fail(self, stage, reason_code) -> None:
            calls.append(("fail", stage))

        def interrupt(self) -> None:
            calls.append(("interrupt", None))

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        base = Path(td).resolve()
        result = p15.run_tactical_recommendation_snapshot_pipeline(
            base / "private-snapshot.json",
            base / "private-performance.json",
            p10_runner=lambda: p10_mini_result,
            log_writer_factory=_Escritura,
        )
    assert result.execution_status == "completed"
    assert [call[0] for call in calls if call[0] != "progress_kwargs"] == [
        "init",
        "bind_counters",
        "begin",
        "complete_stage",
        "begin_stage",
        "complete_stage",
        "begin_stage",
        "progress",
        "complete_stage",
        "begin_stage",
        "complete_stage",
        "complete_stage",
        "complete_stage",
        "set_snapshot_metrics",
        "complete",
    ]
    stages = [
        call[1]
        for call in calls
        if call[0] in ("begin_stage", "complete_stage", "progress")
    ]
    assert stages == [
        "preflight",
        "p10_pipeline",
        "p10_pipeline",
        "target_projection",
        "target_projection",
        "target_projection",
        "snapshot_generation",
        "snapshot_generation",
        "persistence",
        "verification",
    ]
    progress_calls = [
        call for call in calls if call[0] == "progress_kwargs"
    ]
    assert len(progress_calls) == 1
    progress_kwargs = progress_calls[0][1]
    assert progress_kwargs == (
        ("records_built", 2),
        ("targets_processed", 2),
        ("targets_total", 2),
    )
    metrics = [call[1] for call in calls if call[0] == "set_snapshot_metrics"][0]
    assert metrics[0] == 2
    assert type(metrics[1]) is int and metrics[1] > 0


def test_log_de_fallo_lleva_motivo_cerrado(
    authorized, external_dir
):
    result = _run(
        external_dir,
        p10_runner=lambda: (_ for _ in ()).throw(
            p10.TacticalPipelineExecutionError("source_validation", "x")
        ),
    )
    assert result.execution_status == "failed"
    payload = _read_log(external_dir)
    assert payload["execution_status"] == "failed"
    assert payload["current_stage"] == "p10_pipeline"
    assert payload["reason_code"] == "p10_pipeline_execution_failed"
    assert payload["completed_stages"] == ["preflight"]


# --------------------------------------------------------------------- #
# H. Invariantes de arquitectura (AST)                                   #
# --------------------------------------------------------------------- #


def test_ast_imports_cerrados_y_sin_io_en_entrada():
    tree = ast.parse(_MODULE_SOURCE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    assert imported == {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "json",
        "os",
        "pathlib",
        "re",
        "src",
        "tempfile",
        "time",
        "types",
        "typing",
    }
    dangerous_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {"eval", "exec", "open", "compile", "input", "breakpoint",
            "print", "__import__"}
    }
    assert dangerous_calls == set()
    os_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    }
    assert os_attributes <= {"path", "replace", "close", "name"}


def test_ast_sin_asserts_sin_pii_y_con_una_sola_banda():
    tree = ast.parse(_MODULE_SOURCE)
    asserts = [
        node for node in ast.walk(tree) if isinstance(node, ast.Assert)
    ]
    assert asserts == []
    for forbidden in (
        "reports/",
        "data/processed",
        "read_csv",
        "read_parquet",
        "to_parquet",
        "pickle",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "fastapi",
        "streamlit",
        ".parquet",
    ):
        assert forbidden not in _MODULE_SOURCE
    compute_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_tactical_pipeline_result"
    ]
    historic_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute_authorized_real_pipeline"
    ]
    # Una unica banda productiva: la frontera compute-only P10; la
    # frontera historica (con publica y logs) no se llama desde P15.
    assert len(compute_calls) == 1
    assert len(historic_calls) == 0


def test_ast_sin_estado_mutable_modulo():
    tree = ast.parse(_MODULE_SOURCE)
    allowed_value = (
        ast.Constant,
        ast.Tuple,
        ast.Name,
        ast.Attribute,
        ast.Compare,
    )

    def _is_immutable(value) -> bool:
        if isinstance(value, allowed_value):
            return True
        if isinstance(value, ast.Set):
            return True
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                return value.func.id in {"frozenset", "MappingProxyType"}
            if (
                isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "re"
                and value.func.attr == "compile"
            ):
                return True
        return False

    for statement in tree.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            assert statement.value is not None
            assert _is_immutable(statement.value), statement.lineno
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    assert _is_immutable(statement.value), statement.lineno


def test_ast_no_llamadas_a_nivel_de_modulo():
    tree = ast.parse(_MODULE_SOURCE)
    for statement in tree.body:
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
                ast.Expr,
            ),
        )


def test_ast_sin_ejecucion_automatica_ni_reintento():
    tree = ast.parse(_MODULE_SOURCE)
    execution_names = {
        "run_tactical_recommendation_snapshot_pipeline",
        "main",
        "compute_tactical_pipeline_result",
    }
    retry_sites = []
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        for sub in ast.walk(loop):
            if not isinstance(sub, ast.Call) or sub is loop:
                continue
            name = None
            if isinstance(sub.func, ast.Name):
                name = sub.func.id
            elif isinstance(sub.func, ast.Attribute):
                name = sub.func.attr
            if name in execution_names:
                retry_sites.append((loop.lineno, name))
    assert retry_sites == []
    assert [n for n in ast.walk(tree) if isinstance(n, ast.While)] == []


def test_main_no_se_invoca_al_importar():
    tree = ast.parse(_MODULE_SOURCE)
    main_calls = []
    for statement in tree.body:
        if isinstance(statement, ast.If):
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
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "main"
        ):
            pytest.fail("main() se invoca al importar")
    assert len(main_calls) == 1


def test_ningun_test_invoca_al_lector_real():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    reader_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"read_parquet", "read_csv"}
            )
            or (
                isinstance(node.func, ast.Name)
                and node.func.id in {"read_parquet", "read_csv"}
            )
        )
    ]
    assert reader_calls == []
    p10_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "p10"
    }
    assert not (p10_attributes & {"read_parquet", "_read_source_points_once"})


# --------------------------------------------------------------------- #
# I. Equivalencia con P14 directa y consumo P13                          #
# --------------------------------------------------------------------- #


def test_snapshot_equivalente_a_p14_directa(
    authorized, p10_mini_result, mini_policy
):
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        base = Path(td).resolve()
        pipeline_result = _run(
            base, p10_runner=lambda: p10_mini_result
        )
        assert pipeline_result.execution_status == "completed"
        records = p15.project_tactical_target_records(p10_mini_result)
        direct = p14.generate_and_persist_tactical_recommendation_snapshot(
            records, base / "directo.json", policy=mini_policy
        )
        pipeline_raw = (base / "private-snapshot.json").read_bytes()
        direct_raw = (base / "directo.json").read_bytes()
        assert pipeline_raw == direct_raw
        assert (
            pipeline_result.snapshot_fingerprint
            == direct.snapshot_fingerprint
        )
        assert pipeline_result.snapshot_bytes == direct.serialized_bytes
        loaded = p13.load_persisted_tactical_recommendation_snapshot(
            base / "private-snapshot.json"
        )
        assert loaded.entry_count == pipeline_result.snapshot_entries
        assert loaded.fingerprint == pipeline_result.snapshot_fingerprint
