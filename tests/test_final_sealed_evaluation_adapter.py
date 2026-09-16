"""Tests sinteticos P22: adaptador productivo puro (DataFrame ya
cargado -> estructuras P21). Ningun test lee Parquet/CSV reales: todo
DataFrame es construido en memoria con datos sinteticos, con
identificadores y secuencias claramente etiquetados como ficticios.

Cubre: esquema incompleto/tipos incorrectos, fronteras exactas de
fechas y +/-1 dia, cardinalidades exactas train=4188/validation=1805/
test=1531/orientaciones=3062 (incluyendo la escala real completa una
sola vez), duplicados y solapamientos entre particiones, exactamente
dos orientaciones por partido, orden deterministico ante entrada
reordenada, no mutacion del DataFrame de entrada, labels P02 presente/
ausente, ausencia de labels P04/P05/P06, y AST sin I/O de import.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

import src.analysis.final_sealed_evaluation_adapter as adapter
import src.analysis.tactical_recommender_pipeline as p10


_ROOT = Path(__file__).resolve().parents[1]
_COLUMNS = list(p10.SOURCE_COLUMNS)


def _points_row(
    match_id: str,
    when: date,
    *,
    point_number: int = 1,
    surface: str = "Hard",
    server: int = 1,
    point_winner: int = 1,
    player_1: str = "SyntheticPlayerA",
    player_2: str = "SyntheticPlayerB",
    first_serve: str = "4",
    second_serve: object = None,
) -> dict:
    return dict(
        match_id=match_id, point_number=point_number, date=when, surface=surface,
        server=server, point_winner=point_winner, player_1=player_1, player_2=player_2,
        first_serve=first_serve, second_serve=second_serve,
    )


def _match_rows(n: int, start: date, end: date, prefix: str, **overrides) -> list[dict]:
    span = (end - start).days + 1
    return [
        _points_row(f"{prefix}{i}", start + timedelta(days=i % span), **overrides)
        for i in range(n)
    ]


def _small_points_dataframe(
    *, train: int = 2, validation: int = 2, test: int = 2, **overrides
) -> pd.DataFrame:
    data = (
        _match_rows(train, date(2015, 1, 1), date(2019, 12, 31), "train", **overrides)
        + _match_rows(validation, date(2020, 1, 1), date(2023, 12, 31), "val", **overrides)
        + _match_rows(test, date(2024, 1, 1), date(2026, 5, 21), "test", **overrides)
    )
    return pd.DataFrame(data, columns=_COLUMNS)


@pytest.fixture()
def small_cardinalities(monkeypatch):
    monkeypatch.setattr(adapter, "EXPECTED_TRAIN_MATCHES", 2)
    monkeypatch.setattr(adapter, "EXPECTED_VALIDATION_MATCHES", 2)
    monkeypatch.setattr(adapter, "EXPECTED_TEST_MATCHES", 2)
    # EXPECTED_TEST_ORIENTATIONS se importo con "from ... import" dentro
    # del adaptador: es un nombre propio de su namespace, parchear
    # p20.EXPECTED_TEST_ORIENTATIONS no lo afectaria.
    monkeypatch.setattr(adapter, "EXPECTED_TEST_ORIENTATIONS", 4)
    # build_test_targets (P21) re-importa EXPECTED_TEST_MATCHES/
    # EXPECTED_TEST_ORIENTATIONS de p20 EN CADA LLAMADA (import local
    # deliberado); el orquestador lo invoca con
    # enforce_frozen_cardinalities=True, asi que tambien hay que
    # parchear la fuente p20 directamente para las pruebas a pequena
    # escala que pasan por el orquestador.
    import src.analysis.final_sealed_evaluation as p20

    monkeypatch.setattr(p20, "EXPECTED_TEST_MATCHES", 2)
    monkeypatch.setattr(p20, "EXPECTED_TEST_ORIENTATIONS", 4)


@lru_cache(maxsize=1)
def _full_scale_points_dataframe() -> pd.DataFrame:
    """Escala real completa (4188/1805/1531): se construye UNA sola vez
    para todo el modulo de tests (~40s), reutilizada por varios tests."""
    return _small_points_dataframe(train=4188, validation=1805, test=1531)


# --------------------------------------------------------------------- #
# A. Import sin efectos e invariantes AST                                #
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
            "import src.analysis.final_sealed_evaluation_adapter as a; "
            "print(a.FINAL_SEALED_EVALUATION_ADAPTER_CONTRACT_VERSION)",
        ],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "1.0.0"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_ast_no_readers_no_environ() -> None:
    tree = ast.parse(Path(adapter.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "environ", "getenv", "argv", "read_parquet", "read_csv",
            }, f"Acceso prohibido: {node.attr}"


def test_ast_no_module_level_calls() -> None:
    tree = ast.parse(Path(adapter.__file__).read_text(encoding="utf-8"))
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
        ):
            continue
        assert isinstance(
            statement,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign),
        ), f"Estado de modulo no permitido: {type(statement).__name__}"


# --------------------------------------------------------------------- #
# B. Esquema incompleto / tipos incorrectos                              #
# --------------------------------------------------------------------- #


def test_adapt_rejects_non_dataframe() -> None:
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe({"match_id": ["m1"]})


def test_adapt_rejects_missing_column(small_cardinalities) -> None:
    df = _small_points_dataframe().drop(columns=["surface"])
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_rejects_wrong_column_order(small_cardinalities) -> None:
    df = _small_points_dataframe()
    reordered = df.loc[:, list(reversed(_COLUMNS))]
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(reordered)


def test_adapt_rejects_bad_server_type(small_cardinalities) -> None:
    df = _small_points_dataframe(server=3)
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_rejects_non_substantive_first_serve(small_cardinalities) -> None:
    df = _small_points_dataframe(first_serve="")
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


# --------------------------------------------------------------------- #
# C. Fronteras exactas de fechas (+/- 1 dia) y cardinalidades             #
# --------------------------------------------------------------------- #


def test_adapt_rejects_test_date_one_day_before_start(small_cardinalities) -> None:
    df = _small_points_dataframe()
    df = pd.concat(
        [df, pd.DataFrame([_points_row("edge0", date(2023, 12, 31))], columns=_COLUMNS)],
        ignore_index=True,
    )
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_rejects_date_one_day_after_test_end(small_cardinalities) -> None:
    df = _small_points_dataframe()
    df = pd.concat(
        [df, pd.DataFrame([_points_row("edge1", date(2026, 5, 22))], columns=_COLUMNS)],
        ignore_index=True,
    )
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_rejects_wrong_train_cardinality(small_cardinalities) -> None:
    df = _small_points_dataframe(train=3)
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_rejects_wrong_validation_cardinality(small_cardinalities) -> None:
    df = _small_points_dataframe(validation=1)
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_rejects_wrong_test_cardinality(small_cardinalities) -> None:
    df = _small_points_dataframe(test=3)
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_accepts_exact_real_cardinalities() -> None:
    """Escala real completa: 4188 train, 1805 validacion, 1531 test,
    3062 orientaciones -- SIN monkeypatch de las constantes congeladas."""
    result = adapter.adapt_real_points_dataframe(_full_scale_points_dataframe())
    assert result.train_matches == 4188
    assert result.validation_matches == 1805
    assert result.test_matches == 1531
    assert result.test_orientations == 3062
    assert len(result.test_match_rows) == 1531


# --------------------------------------------------------------------- #
# D. Duplicados, solapamientos, dos orientaciones, orden determinista    #
# --------------------------------------------------------------------- #


def test_adapt_rejects_duplicate_match_id_across_splits(small_cardinalities) -> None:
    df = _small_points_dataframe()
    duplicate = df.iloc[[0]].copy()
    duplicate.loc[:, "date"] = date(2024, 6, 1)
    df = pd.concat([df, duplicate], ignore_index=True)
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_adapt_rejects_match_spanning_the_temporal_cut(small_cardinalities) -> None:
    df = _small_points_dataframe()
    straddling = pd.DataFrame(
        [
            _points_row("straddle", date(2023, 12, 31), point_number=1),
            _points_row("straddle", date(2024, 1, 1), point_number=2),
        ],
        columns=_COLUMNS,
    )
    df = pd.concat([df, straddling], ignore_index=True)
    with pytest.raises(adapter.FinalSealedEvaluationAdapterError):
        adapter.adapt_real_points_dataframe(df)


def test_test_match_rows_have_exactly_two_orientations_worth(small_cardinalities) -> None:
    df = _small_points_dataframe()
    result = adapter.adapt_real_points_dataframe(df)
    assert result.test_orientations == 2 * len(result.test_match_rows)


def test_adapt_deterministic_regardless_of_row_order(small_cardinalities) -> None:
    df = _small_points_dataframe()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    result_a = adapter.adapt_real_points_dataframe(df)
    result_b = adapter.adapt_real_points_dataframe(shuffled)
    assert result_a.test_match_rows == result_b.test_match_rows
    assert result_a.development_observations == result_b.development_observations
    assert result_a.test_observations == result_b.test_observations


def test_adapt_does_not_mutate_input_dataframe(small_cardinalities) -> None:
    df = _small_points_dataframe()
    fingerprint_before = (df.shape, tuple(df.columns), df["match_id"].tolist())
    adapter.adapt_real_points_dataframe(df)
    fingerprint_after = (df.shape, tuple(df.columns), df["match_id"].tolist())
    assert fingerprint_before == fingerprint_after


# --------------------------------------------------------------------- #
# E. Labels P02 presente/ausente; sin labels P04/P05/P06                 #
# --------------------------------------------------------------------- #


def test_p02_observed_attempt_present_for_recognizable_direction(small_cardinalities) -> None:
    df = _small_points_dataframe(first_serve="4")
    result = adapter.adapt_real_points_dataframe(df)
    assert len(result.test_p02_observed_attempts) > 0
    assert all(item.category for item in result.test_p02_observed_attempts)
    assert len(result.development_p02_observed_attempts) > 0
    assert all(item.category for item in result.development_p02_observed_attempts)


def test_p02_observed_attempt_absent_for_unrecognizable_direction(small_cardinalities) -> None:
    df = _small_points_dataframe(first_serve="n4")  # UNKNOWN_FIRST_SERVE
    result = adapter.adapt_real_points_dataframe(df)
    assert result.test_p02_observed_attempts == ()
    assert result.development_p02_observed_attempts == ()


def test_development_p02_observed_attempts_never_include_test_dates(
    small_cardinalities,
) -> None:
    """La baseline poblacional (decision humana congelada) exige fecha
    maxima 2023-12-31: lo garantiza construct_tactical_attempt_records
    (P10) al rechazar cualquier fecha de test antes de que este
    adaptador pueda derivar ningun P02ObservedAttempt de desarrollo."""
    df = _small_points_dataframe(first_serve="4")
    result = adapter.adapt_real_points_dataframe(df)
    development_match_ids = {row.match_id for row in df.itertuples() if row.date.year < 2024}
    assert all(
        item.target_match_id in development_match_ids
        for item in result.development_p02_observed_attempts
    )
    test_match_ids = {item.match_id for item in result.test_match_rows}
    assert not test_match_ids & {
        item.target_match_id for item in result.development_p02_observed_attempts
    }


def test_adapter_does_not_duplicate_p10_point_construction_logic() -> None:
    """Bloqueo de revision: el adaptador ya NO define ninguna copia de
    _point_seeds / _sequence_presence / _documents_service_fault; usa
    exclusivamente construct_tactical_attempt_records_for_test_partition
    (P10, publica)."""
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "def _point_seeds",
        "def _sequence_presence",
        "def _documents_service_fault",
        "AttemptSignalRequest(",
        "extract_tactical_signals_for_attempt(",
        "encode_tactical_attempt(",
    ):
        assert forbidden not in source, f"Duplicacion detectada: {forbidden!r}"
    assert "construct_tactical_attempt_records_for_test_partition" in source


def test_adapter_never_fabricates_p04_p05_p06_labels() -> None:
    """El adaptador no expone ningun tipo/campo de label para P04/P05/
    P06: solo P02ObservedAttempt existe en su superficie publica."""
    for name in adapter.__all__:
        assert "P04" not in name and "P05" not in name and "P06" not in name
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "P04ObservedAttempt" not in source
    assert "P05ObservedAttempt" not in source
    assert "P06ObservedAttempt" not in source


# --------------------------------------------------------------------- #
# F. Errores cerrados sin identidades ni rutas                           #
# --------------------------------------------------------------------- #


def test_adapter_errors_never_leak_identities_or_paths(small_cardinalities) -> None:
    df = _small_points_dataframe(train=3)
    try:
        adapter.adapt_real_points_dataframe(df)
    except adapter.FinalSealedEvaluationAdapterError as error:
        message = str(error)
        assert "SyntheticPlayerA" not in message
        assert str(_ROOT) not in message
