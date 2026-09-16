"""Tests sinteticos P22: runner productivo (serializacion, publicacion
atomica del bundle y frontera cerrada). Ningun test lee Parquet/CSV
reales ni escribe en ``reports/`` real: toda publicacion ocurre en
``tmp_path``, y todo loader/evaluador es sintetico o inyectado.

Guard local (NO un conftest.py global -- ver revision de diseno): una
fixture ``autouse`` definida SOLO en este modulo comprueba, mediante
existencia/tamano/SHA-256 del bundle contractual real
(``FINAL_EVALUATION_BUNDLE_DIR``), que ningun test lo toco -- sea que
no exista (estado actual) o que exista legitimamente tras una futura
evaluacion real autorizada (en cuyo caso debe permanecer BYTE A BYTE
identico). No presupone ausencia permanente: los tests de este archivo
siguen siendo ejecutables despues de que exista un bundle real.

Cubre: puerta cerrada (parcheada a False donde se ejercita ese camino)
antes de cualquier I/O, fuente ausente/directorio/symlink/no-regular,
bundle preexistente bloqueado, serializacion determinista, JSON con
claves duplicadas rechazado en verificacion, fallo al escribir/
verificar CUALQUIERA de los cinco archivos deja cero bundle final y
cero temporales, fallo del UNICO replace final deja cero bundle y cero
temporales, ningun estado parcial visible antes del rename, exactamente
un replace de publicacion (y una unica llamada a cada una de fuente/
evaluacion/publicacion en el runner completo), verificacion posterior
del bundle completo, permisos privados, limites de tamano, cero
reintentos (incluida ausencia de bucle en ``main()``), SIGTERM/SIGINT
gestionados sin dejar temporales (simulado y con senal real de SO),
argv real del CLI correctamente reenviado a ``main()``, cero efectos al
importar, variables de entorno simuladas sin cambiar la puerta, y AST
con una unica asignacion de ``REAL_TEST_EVALUATION_AUTHORIZED`` (en
P20, nunca reasignada aqui).

IMPORTANTE (P23): el valor VIGENTE (no parcheado) de
``runner.REAL_TEST_EVALUATION_AUTHORIZED`` es ``True`` -- una unica
ejecucion manual autorizada tras el preflight completo, ver
``final_sealed_evaluation.py``. Todo test de este archivo que ejercite
el camino "puerta cerrada" parchea explicitamente a False; todo test
que ejercite el camino "autorizado" sigue inyectando un
``source_reader`` sintetico o sustituyendo ``execute_real_sealed_test_
evaluation``. Ningun test de este archivo invoca jamas el lector real
ni deja que la ruta por defecto (``pd.read_parquet`` sobre la fuente
contractual real) se ejecute.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import src.analysis.final_sealed_evaluation_adapter as adapter
import src.analysis.final_sealed_evaluation_orchestrator as orchestrator
import src.analysis.final_sealed_evaluation_runner as runner
from test_final_sealed_evaluation_adapter import (
    _small_points_dataframe,
    small_cardinalities,  # noqa: F401 (fixture reexportada)
)


_ROOT = Path(__file__).resolve().parents[1]
_IS_WINDOWS = os.name == "nt"


# --------------------------------------------------------------------- #
# Guard LOCAL (solo este modulo): el bundle contractual real nunca      #
# cambia, exista o no.                                                   #
# --------------------------------------------------------------------- #


def _bundle_metadata(bundle_dir: Path) -> tuple | None:
    """``None`` si el bundle no existe; si existe, una tupla ordenada de
    ``(ruta relativa, tamano, sha256)`` por archivo. Compararla antes/
    despues detecta CUALQUIER cambio sin presuponer que deba estar
    ausente."""
    if not bundle_dir.exists():
        return None
    if not bundle_dir.is_dir():
        return ("not_a_directory", bundle_dir.stat().st_size)
    entries = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((str(path.relative_to(bundle_dir)), path.stat().st_size, digest))
    return tuple(entries)


@pytest.fixture(autouse=True)
def _guard_real_bundle_untouched():
    before = _bundle_metadata(runner.FINAL_EVALUATION_BUNDLE_DIR)
    yield
    after = _bundle_metadata(runner.FINAL_EVALUATION_BUNDLE_DIR)
    assert before == after, (
        "El bundle contractual real (reports/final_evaluation/) cambio "
        "durante un test; todos los tests deben publicar exclusivamente "
        "bajo tmp_path."
    )


def test_guard_fixture_tolerates_a_preexisting_legitimate_bundle(
    small_outcome, tmp_path, monkeypatch
) -> None:
    """Prueba que el propio guard funciona igual de bien cuando el
    bundle YA existe legitimamente (simulando el estado posterior a una
    evaluacion real autorizada): publica en un bundle_dir de prueba,
    computa su metadata, y confirma que SI ese bundle fuese el
    contractual real, una segunda pasada sin tocarlo reportaria
    metadata identica (antes==despues)."""
    fake_bundle = tmp_path / "final_evaluation"
    runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=fake_bundle)
    before = _bundle_metadata(fake_bundle)
    after = _bundle_metadata(fake_bundle)
    assert before == after
    assert before is not None


# --------------------------------------------------------------------- #
# A. Import sin efectos y AST                                            #
# --------------------------------------------------------------------- #


def test_import_subprocess_sin_efectos(tmp_path) -> None:
    sentinel = tmp_path / "sentinela.txt"
    sentinel.write_text("x", encoding="utf-8")
    before = {item.name for item in tmp_path.iterdir()}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import src.analysis.final_sealed_evaluation_runner as r; "
            "print(r.FINAL_SEALED_EVALUATION_RUNNER_CONTRACT_VERSION)",
        ],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "1.0.0"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def _authorization_assignments(source: str) -> list[ast.AST]:
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REAL_TEST_EVALUATION_AUTHORIZED"
            for target in node.targets
        ):
            found.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REAL_TEST_EVALUATION_AUTHORIZED"
        ):
            found.append(node)
    return found


def test_ast_single_authorization_assignment_lives_only_in_p20() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert _authorization_assignments(source) == []


def test_ast_p20_has_exactly_one_authorization_assignment() -> None:
    import src.analysis.final_sealed_evaluation as p20

    source = Path(p20.__file__).read_text(encoding="utf-8")
    assert len(_authorization_assignments(source)) == 1
    assert p20.REAL_TEST_EVALUATION_AUTHORIZED is True


# --------------------------------------------------------------------- #
# B. Puerta False antes de cualquier I/O                                 #
# --------------------------------------------------------------------- #


def test_execute_real_evaluation_aborts_before_source_reader_called(monkeypatch) -> None:
    """Ejercita deliberadamente la rama con la puerta CERRADA (parcheada
    a False aqui): el valor vigente hoy es True (P23, una unica
    ejecucion manual autorizada); este test protege el camino
    POSTERIOR al cierre de esa ejecucion (tras completarse, fallar o
    interrumpirse, la puerta vuelve a False en un commit futuro)."""
    monkeypatch.setattr(runner, "REAL_TEST_EVALUATION_AUTHORIZED", False)

    def _forbidden(_path):
        raise AssertionError("source_reader invocado con la puerta cerrada.")

    with pytest.raises(SystemExit) as exc_info:
        runner.execute_real_sealed_test_evaluation(source_reader=_forbidden)
    assert str(exc_info.value) == runner._ERROR_MESSAGES["not_authorized"]


def test_execute_real_evaluation_aborts_before_touching_bundle_dir(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "REAL_TEST_EVALUATION_AUTHORIZED", False)
    pre_existing = tmp_path / "final_evaluation"
    pre_existing.mkdir()
    (pre_existing / "sentinel.txt").write_text("real-looking-but-untouched", encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.execute_real_sealed_test_evaluation(
            source_reader=lambda _p: (_ for _ in ()).throw(AssertionError()),
            bundle_dir=pre_existing,
        )
    assert (pre_existing / "sentinel.txt").read_text(encoding="utf-8") == (
        "real-looking-but-untouched"
    )


def test_main_returns_1_with_closed_gate_before_any_io(monkeypatch) -> None:
    monkeypatch.setattr(runner, "REAL_TEST_EVALUATION_AUTHORIZED", False)
    assert runner.main([]) == 1


def test_main_default_state_is_authorized_but_never_invoked_by_tests() -> None:
    """P23: el valor VIGENTE (no parcheado) de la puerta es True -- una
    unica ejecucion manual autorizada. Ningun test de este archivo
    debe invocar ``runner.main([])``/``execute_real_sealed_test_
    evaluation()`` bajo este valor real sin sustituir ``source_reader``
    (eso dispararia una lectura real de la fuente contractual); este
    test documenta y fija el valor esperado sin ejecutar nada mas."""
    assert runner.REAL_TEST_EVALUATION_AUTHORIZED is True


def test_no_bypass_env_var_opens_the_gate() -> None:
    """Envenena con "false" -- el OPUESTO del valor real (True tras
    P23) -- para demostrar que el entorno tampoco puede apagarla."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT)
    env["REAL_TEST_EVALUATION_AUTHORIZED"] = "false"
    env["TENNIS_FINAL_EVALUATION_FORCE"] = "0"
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import src.analysis.final_sealed_evaluation_runner as r; "
            "print(r.REAL_TEST_EVALUATION_AUTHORIZED)",
        ],
        cwd=str(_ROOT), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "True"


def test_cli_rejects_any_argv_without_touching_gate() -> None:
    assert runner.main(["--force"]) == 2
    assert runner.main(["--retry"]) == 2
    assert runner.main(["--authorized"]) == 2


# --------------------------------------------------------------------- #
# C. Fuente: ausente, directorio, symlink, no regular                    #
# --------------------------------------------------------------------- #


def test_validate_source_path_contract_rejects_wrong_path(tmp_path) -> None:
    other = tmp_path / "not-the-contractual-source.parquet"
    other.write_bytes(b"x")
    with pytest.raises(runner.FinalSealedEvaluationRunnerError) as exc_info:
        runner._validate_source_path_contract(other)
    assert exc_info.value.reason_code == "source_path_contract_violation"


def test_validate_source_path_contract_rejects_missing_file(monkeypatch) -> None:
    monkeypatch.setattr(
        runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", Path("does-not-exist.parquet")
    )
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner._validate_source_path_contract(Path("does-not-exist.parquet"))


def test_validate_source_path_contract_rejects_directory(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "as-directory"
    directory.mkdir()
    monkeypatch.setattr(runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", directory)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner._validate_source_path_contract(directory)


@pytest.mark.skipif(_IS_WINDOWS, reason="symlinks reales requieren privilegios en Windows")
def test_validate_source_path_contract_rejects_symlink(tmp_path, monkeypatch) -> None:
    target = tmp_path / "real.parquet"
    target.write_bytes(b"x")
    link = tmp_path / "link.parquet"
    link.symlink_to(target)
    monkeypatch.setattr(runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", link)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner._validate_source_path_contract(link)


# --------------------------------------------------------------------- #
# D. Bundle preexistente bloquea antes de leer la fuente                 #
# --------------------------------------------------------------------- #


def test_validate_bundle_destination_rejects_existing_directory(tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    bundle_dir.mkdir()
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner.validate_bundle_destination(bundle_dir)


def test_validate_bundle_destination_rejects_existing_file(tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    bundle_dir.write_text("not a directory", encoding="utf-8")
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner.validate_bundle_destination(bundle_dir)


def test_validate_bundle_destination_rejects_preflight_p20_path() -> None:
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner.validate_bundle_destination(
            runner._REPORTS_DIR / "final_evaluation_preflight.json"
        )


def test_validate_bundle_destination_passes_when_absent(tmp_path) -> None:
    runner.validate_bundle_destination(tmp_path / "final_evaluation")


@pytest.mark.skipif(_IS_WINDOWS, reason="symlinks reales requieren privilegios en Windows")
def test_validate_bundle_destination_rejects_symlinked_bundle(tmp_path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "final_evaluation"
    link.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner.validate_bundle_destination(link)


@pytest.mark.skipif(_IS_WINDOWS, reason="symlinks reales requieren privilegios en Windows")
def test_validate_bundle_destination_rejects_symlinked_parent(tmp_path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner.validate_bundle_destination(linked_parent / "final_evaluation")


# --------------------------------------------------------------------- #
# E. Serializacion determinista, tamano, claves duplicadas                #
# --------------------------------------------------------------------- #


@pytest.fixture()
def small_outcome(small_cardinalities):
    df = _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")
    adapted = adapter.adapt_real_points_dataframe(df)
    return orchestrator.evaluate_final_sealed_test(adapted)


def test_summary_is_deterministic(small_outcome) -> None:
    first = runner.build_final_evaluation_summary(small_outcome)
    second = runner.build_final_evaluation_summary(small_outcome)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_bundle_payloads_are_byte_deterministic(small_outcome) -> None:
    first = runner.build_final_evaluation_bundle_payloads(small_outcome)
    second = runner.build_final_evaluation_bundle_payloads(small_outcome)
    assert first == second


def test_summary_has_no_target_level_identities(small_outcome) -> None:
    blob = json.dumps(runner.build_final_evaluation_summary(small_outcome))
    assert "SyntheticPlayerA" not in blob
    assert "SyntheticPlayerB" not in blob
    assert str(_ROOT) not in blob


def test_summary_includes_baseline_numerator_denominator_rule(small_outcome) -> None:
    summary = runner.build_final_evaluation_summary(small_outcome)
    baseline = summary["p02_population_baseline"]
    assert baseline["rule"] == small_outcome.p02_population_baseline.rule
    assert baseline["numerator"] == small_outcome.p02_population_baseline.numerator
    assert baseline["denominator"] == small_outcome.p02_population_baseline.denominator
    assert baseline["rate"] == small_outcome.p02_population_baseline.rate
    assert (
        summary["fingerprints"]["p02_population_baseline_fingerprint"]
        == small_outcome.p02_population_baseline_fingerprint
    )


def test_summary_baseline_identical_between_rolling_and_frozen(small_outcome) -> None:
    summary = runner.build_final_evaluation_summary(small_outcome)
    rolling_baseline = summary["rolling"]["p02_population_baseline_comparison"]
    frozen_baseline = summary["frozen"]["p02_population_baseline_comparison"]
    for key in (
        "population_baseline_rule",
        "population_baseline_numerator",
        "population_baseline_denominator",
        "population_baseline_rate",
    ):
        assert rolling_baseline[key] == frozen_baseline[key]


def test_summary_flags_p02_as_only_predictive_pattern(small_outcome) -> None:
    summary = runner.build_final_evaluation_summary(small_outcome)
    assert summary["p02_only_predictive_metrics"] is True
    assert summary["p04_p05_p06_coverage_only_not_predictive"] is True


def test_payloads_reject_oversized_artifact(small_outcome, monkeypatch) -> None:
    monkeypatch.setattr(runner, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError) as exc_info:
        runner.build_final_evaluation_bundle_payloads(small_outcome)
    assert exc_info.value.reason_code == "artifact_too_large"


def test_verify_bundle_rejects_duplicate_json_keys(tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    bundle_dir.mkdir()
    (bundle_dir / "tables").mkdir()
    for relative in runner.BUNDLE_RELATIVE_PATHS:
        target = bundle_dir / relative
        if relative.endswith(".json"):
            target.write_text('{"a": 1, "a": 2}', encoding="utf-8")
        else:
            target.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner._verify_bundle_directory(bundle_dir)


def test_verify_bundle_rejects_missing_file(tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    bundle_dir.mkdir()
    (bundle_dir / "tables").mkdir()
    for relative in runner.BUNDLE_RELATIVE_PATHS[:-1]:
        target = bundle_dir / relative
        target.write_text("{}" if relative.endswith(".json") else "a\n1\n", encoding="utf-8")
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner._verify_bundle_directory(bundle_dir)


def test_verify_bundle_rejects_extra_file(tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    bundle_dir.mkdir()
    (bundle_dir / "tables").mkdir()
    for relative in runner.BUNDLE_RELATIVE_PATHS:
        target = bundle_dir / relative
        target.write_text("{}" if relative.endswith(".json") else "a\n1\n", encoding="utf-8")
    (bundle_dir / "tables" / "unexpected.csv").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner._verify_bundle_directory(bundle_dir)


# --------------------------------------------------------------------- #
# F. Publicacion atomica de bundle: un unico replace, todo o nada        #
# --------------------------------------------------------------------- #


def test_publish_creates_exactly_the_expected_bundle(small_outcome, tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    published = runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert published == bundle_dir
    found = {
        str(path.relative_to(bundle_dir)).replace(os.sep, "/")
        for path in bundle_dir.rglob("*")
        if path.is_file()
    }
    assert found == set(runner.BUNDLE_RELATIVE_PATHS)
    runner.verify_published_bundle(bundle_dir)


def test_publish_uses_exactly_one_os_replace_call(small_outcome, tmp_path, monkeypatch) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    calls = []
    real_replace = os.replace

    def _counting_replace(src, dst):
        calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(runner.os, "replace", _counting_replace)
    runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert len(calls) == 1
    assert Path(calls[0][1]) == bundle_dir


def test_publish_rejects_second_attempt_over_existing_bundle(small_outcome, tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)


def test_publish_leaves_no_temp_directory_on_success(small_outcome, tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    leftovers = [
        item for item in tmp_path.iterdir()
        if item != bundle_dir and item.name.startswith(".")
    ]
    assert leftovers == []


@pytest.mark.skipif(_IS_WINDOWS, reason="bits de permiso POSIX no aplican igual en NTFS")
def test_publish_sets_private_permissions(small_outcome, tmp_path) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    published = runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    for relative in runner.BUNDLE_RELATIVE_PATHS:
        mode = stat.S_IMODE((published / relative).stat().st_mode)
        assert mode == (stat.S_IRUSR | stat.S_IWUSR)


def test_publish_fsyncs_temp_and_parent_directories(
    small_outcome, tmp_path, monkeypatch
) -> None:
    calls = []
    real_fsync_dir = runner._fsync_directory_best_effort

    def _spy(directory):
        calls.append(directory)
        return real_fsync_dir(directory)

    monkeypatch.setattr(runner, "_fsync_directory_best_effort", _spy)
    bundle_dir = tmp_path / "final_evaluation"
    runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    # Se espera al menos: tables/ del temporal, el temporal en si, y el
    # padre (reports_like) tras el replace.
    assert len(calls) >= 3
    assert tmp_path in calls


@pytest.mark.parametrize("failing_relative", list(runner.BUNDLE_RELATIVE_PATHS))
def test_write_failure_on_any_of_the_five_files_leaves_zero_bundle_and_zero_temp(
    small_outcome, tmp_path, monkeypatch, failing_relative
) -> None:
    bundle_dir = tmp_path / "final_evaluation"
    real_open = open

    def _flaky_open(path, mode="r", *args, **kwargs):
        if "wb" in mode and str(path).replace(os.sep, "/").endswith(failing_relative):
            raise OSError("fallo simulado de escritura")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _flaky_open)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError) as exc_info:
        runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert exc_info.value.reason_code == "publish_failed"
    assert not bundle_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_verification_failure_leaves_zero_bundle_and_zero_temp(
    small_outcome, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        runner, "_verify_bundle_directory", lambda *_a, **_k: (_ for _ in ()).throw(
            runner._fail("verification_failed")
        )
    )
    bundle_dir = tmp_path / "final_evaluation"
    with pytest.raises(runner.FinalSealedEvaluationRunnerError):
        runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_replace_failure_leaves_zero_bundle_and_zero_temp(
    small_outcome, tmp_path, monkeypatch
) -> None:
    bundle_dir = tmp_path / "final_evaluation"

    def _flaky_replace(_src, _dst):
        raise OSError("fallo simulado de replace")

    monkeypatch.setattr(runner.os, "replace", _flaky_replace)
    with pytest.raises(runner.FinalSealedEvaluationRunnerError) as exc_info:
        runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert exc_info.value.reason_code == "publish_failed"
    assert not bundle_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_no_partial_bundle_visible_before_final_replace(
    small_outcome, tmp_path, monkeypatch
) -> None:
    """Antes del unico os.replace, el destino final NUNCA debe existir
    (ni parcial ni completo): se comprueba justo antes de invocarlo."""
    bundle_dir = tmp_path / "final_evaluation"
    real_replace = os.replace
    observed_before_replace = {}

    def _observing_replace(src, dst):
        observed_before_replace["bundle_exists"] = Path(dst).exists()
        return real_replace(src, dst)

    monkeypatch.setattr(runner.os, "replace", _observing_replace)
    runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert observed_before_replace["bundle_exists"] is False


# --------------------------------------------------------------------- #
# G. Fallo en cada fase del runner completo, sin publicacion parcial     #
# --------------------------------------------------------------------- #


def _authorized_runner_env(monkeypatch):
    monkeypatch.setattr(runner, "REAL_TEST_EVALUATION_AUTHORIZED", True)


def test_full_runner_synthetic_success_with_injected_loader(
    small_cardinalities, tmp_path, monkeypatch
) -> None:
    _authorized_runner_env(monkeypatch)
    bundle_dir = tmp_path / "final_evaluation"
    source = tmp_path / "synthetic-source.parquet"
    source.write_bytes(b"synthetic-not-real-parquet")
    monkeypatch.setattr(runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", source)

    def _synthetic_reader(_path):
        return _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")

    published = runner.execute_real_sealed_test_evaluation(
        source_reader=_synthetic_reader, bundle_dir=bundle_dir
    )
    assert published == bundle_dir
    runner.verify_published_bundle(bundle_dir)


def test_full_runner_adapter_failure_leaves_no_bundle(
    small_cardinalities, tmp_path, monkeypatch
) -> None:
    _authorized_runner_env(monkeypatch)
    bundle_dir = tmp_path / "final_evaluation"
    source = tmp_path / "synthetic-source.parquet"
    source.write_bytes(b"synthetic-not-real-parquet")
    monkeypatch.setattr(runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", source)

    def _broken_reader(_path):
        return _small_points_dataframe(train=999, validation=2, test=2)

    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        runner.execute_real_sealed_test_evaluation(
            source_reader=_broken_reader, bundle_dir=bundle_dir
        )
    assert not bundle_dir.exists()


def test_full_runner_keyboard_interrupt_via_cli_returns_130(monkeypatch) -> None:
    _authorized_runner_env(monkeypatch)
    monkeypatch.setattr(
        runner,
        "execute_real_sealed_test_evaluation",
        lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert runner.main() == 130


# --------------------------------------------------------------------- #
# H. P23: SIGTERM gestionado, argv real del CLI, llamadas unicas.        #
#                                                                         #
# ADVERTENCIA para quien anada tests a este archivo: tras P23, el valor  #
# VIGENTE (no parcheado) de ``runner.REAL_TEST_EVALUATION_AUTHORIZED``   #
# es ``True`` (una unica ejecucion manual autorizada, ver               #
# ``final_sealed_evaluation.py``). Invocar ``runner.main()`` sin         #
# argumentos o ``execute_real_sealed_test_evaluation()`` sin             #
# ``source_reader`` explicito, SIN antes parchear la puerta a False o    #
# sustituir ``execute_real_sealed_test_evaluation``/``source_reader``,   #
# dispara una LECTURA REAL del Parquet contractual. Cada test que        #
# ejercite el camino "puerta cerrada" debe parchear explicitamente a     #
# False; cada test que ejercite el camino "autorizado" debe seguir       #
# inyectando un ``source_reader`` sintetico como ya hace el resto de     #
# este archivo.                                                         #
# --------------------------------------------------------------------- #


def test_normalize_execution_signals_restores_original_handlers() -> None:
    import signal as signal_module

    original_term = signal_module.getsignal(signal_module.SIGTERM)
    original_int = signal_module.getsignal(signal_module.SIGINT)
    with runner._normalize_execution_signals():
        assert (
            signal_module.getsignal(signal_module.SIGTERM)
            is runner._raise_managed_termination
        )
        assert (
            signal_module.getsignal(signal_module.SIGINT)
            is runner._raise_managed_termination
        )
    assert signal_module.getsignal(signal_module.SIGTERM) is original_term
    assert signal_module.getsignal(signal_module.SIGINT) is original_int


def test_normalize_execution_signals_restores_handlers_even_on_exception() -> None:
    import signal as signal_module

    original_term = signal_module.getsignal(signal_module.SIGTERM)
    with pytest.raises(ValueError):
        with runner._normalize_execution_signals():
            raise ValueError("fallo simulado dentro de la ventana gestionada")
    assert signal_module.getsignal(signal_module.SIGTERM) is original_term


def test_managed_termination_signal_during_write_propagates_unconverted_and_cleans_up(
    small_outcome, tmp_path, monkeypatch
) -> None:
    """Simulado (sin senal real de SO): inyecta _ManagedTerminationSignal
    durante la escritura y confirma que se relanza SIN convertirse en
    publish_failed (para que main() pueda distinguir 130 de 1), y que
    el temporal se borra igual que ante cualquier otro fallo."""
    bundle_dir = tmp_path / "final_evaluation"
    real_open = open

    def _flaky_open(path, mode="r", *args, **kwargs):
        if "wb" in mode:
            raise runner._ManagedTerminationSignal
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _flaky_open)
    with pytest.raises(runner._ManagedTerminationSignal):
        runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_managed_termination_signal_during_replace_propagates_unconverted_and_cleans_up(
    small_outcome, tmp_path, monkeypatch
) -> None:
    bundle_dir = tmp_path / "final_evaluation"

    def _flaky_replace(_src, _dst):
        raise runner._ManagedTerminationSignal

    monkeypatch.setattr(runner.os, "replace", _flaky_replace)
    with pytest.raises(runner._ManagedTerminationSignal):
        runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_real_sigterm_during_publication_is_caught_and_cleans_up(
    small_outcome, tmp_path, monkeypatch
) -> None:
    """SIGTERM REAL de sistema operativo (no simulado), entregado
    durante la fase de escritura del bundle mientras los manejadores de
    ``_normalize_execution_signals`` estan instalados: debe capturarse
    como ``_ManagedTerminationSignal``, limpiar el temporal por
    completo y no dejar bundle final. Verificado previamente de forma
    manual en este mismo entorno (Windows) con
    ``signal.raise_signal(signal.SIGTERM)``."""
    import signal as signal_module

    bundle_dir = tmp_path / "final_evaluation"
    real_open = open
    raised = {"done": False}

    def _signal_then_open(path, mode="r", *args, **kwargs):
        if "wb" in mode and not raised["done"]:
            raised["done"] = True
            signal_module.raise_signal(signal_module.SIGTERM)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _signal_then_open)
    with runner._normalize_execution_signals():
        with pytest.raises(runner._ManagedTerminationSignal):
            runner.publish_final_evaluation_bundle(small_outcome, bundle_dir=bundle_dir)
    assert not bundle_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_full_runner_managed_termination_signal_via_cli_returns_130(monkeypatch) -> None:
    _authorized_runner_env(monkeypatch)
    monkeypatch.setattr(
        runner,
        "execute_real_sealed_test_evaluation",
        lambda **_kwargs: (_ for _ in ()).throw(runner._ManagedTerminationSignal()),
    )
    assert runner.main() == 130


def test_ast_main_has_no_retry_loop() -> None:
    """Sin bucle de reintento en ``main()``: un fallo o una
    interrupcion consumen la unica autorizacion (regla de cierre de
    P23); una segunda ejecucion exige una nueva invocacion manual, no
    un reintento automatico dentro del propio proceso."""
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_def = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    for node in ast.walk(main_def):
        assert not isinstance(node, (ast.For, ast.While)), (
            "main() no debe contener ningun bucle de reintento."
        )


def test_cli_subprocess_real_argv_rejects_unknown_flags_with_exit_2() -> None:
    """Prueba de regresion del bug corregido en P23: el guard
    ``if __name__ == \"__main__\":`` antes llamaba a ``main()`` SIN
    argumentos, dejando ``if argv: return 2`` como codigo muerto en
    ejecucion real (cualquier flag tecleado por un usuario se ignoraba
    en silencio). Solo se invoca aqui con argumentos que DEBEN
    rechazarse antes de tocar la puerta o cualquier I/O (salida 2);
    nunca sin argumentos, para no arriesgar una evaluacion real ahora
    que la puerta vale True (P23)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT)
    for bad_args in (["--force"], ["--retry"], ["some-positional-path"]):
        completed = subprocess.run(
            [
                sys.executable, "-m",
                "src.analysis.final_sealed_evaluation_runner", *bad_args,
            ],
            cwd=str(_ROOT), env=env, capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 2, (bad_args, completed.stdout, completed.stderr)


def test_full_runner_calls_source_reader_exactly_once(
    small_cardinalities, tmp_path, monkeypatch
) -> None:
    _authorized_runner_env(monkeypatch)
    bundle_dir = tmp_path / "final_evaluation"
    source = tmp_path / "synthetic-source.parquet"
    source.write_bytes(b"synthetic-not-real-parquet")
    monkeypatch.setattr(runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", source)

    calls = []

    def _counting_reader(_path):
        calls.append(_path)
        return _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")

    runner.execute_real_sealed_test_evaluation(
        source_reader=_counting_reader, bundle_dir=bundle_dir
    )
    assert len(calls) == 1


def test_full_runner_calls_evaluate_final_sealed_test_exactly_once(
    small_cardinalities, tmp_path, monkeypatch
) -> None:
    _authorized_runner_env(monkeypatch)
    bundle_dir = tmp_path / "final_evaluation"
    source = tmp_path / "synthetic-source.parquet"
    source.write_bytes(b"synthetic-not-real-parquet")
    monkeypatch.setattr(runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", source)

    calls = []
    real_evaluate = runner.evaluate_final_sealed_test

    def _counting_evaluate(*args, **kwargs):
        calls.append(1)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_final_sealed_test", _counting_evaluate)

    def _synthetic_reader(_path):
        return _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")

    runner.execute_real_sealed_test_evaluation(
        source_reader=_synthetic_reader, bundle_dir=bundle_dir
    )
    assert len(calls) == 1


def test_full_runner_calls_publish_exactly_once(
    small_cardinalities, tmp_path, monkeypatch
) -> None:
    _authorized_runner_env(monkeypatch)
    bundle_dir = tmp_path / "final_evaluation"
    source = tmp_path / "synthetic-source.parquet"
    source.write_bytes(b"synthetic-not-real-parquet")
    monkeypatch.setattr(runner, "FINAL_SEALED_EVALUATION_SOURCE_PATH", source)

    calls = []
    real_publish = runner.publish_final_evaluation_bundle

    def _counting_publish(*args, **kwargs):
        calls.append(1)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(runner, "publish_final_evaluation_bundle", _counting_publish)

    def _synthetic_reader(_path):
        return _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")

    runner.execute_real_sealed_test_evaluation(
        source_reader=_synthetic_reader, bundle_dir=bundle_dir
    )
    assert len(calls) == 1
