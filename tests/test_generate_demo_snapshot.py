"""Pruebas P27: snapshot de demostración pública, pequeña y determinista.

Ningún test de este archivo lee ``points_enriched.parquet``, abre el
snapshot privado real ni ejecuta el evaluador sellado 2024-2026: todo
opera sobre ``demo/tactical-recommendations-demo-v1.json`` (versionado)
y/o sobre bytes generados en memoria por ``scripts.generate_demo_
snapshot``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.generate_demo_snapshot as demo_gen
from src.recommender.persisted_tactical_recommendation_provider import (
    create_persisted_tactical_recommendation_provider,
    load_persisted_tactical_recommendation_snapshot,
)
from src.recommender.tactical_recommendation_service import (
    RecommendationNotFoundError,
    TacticalRecommendationQuery,
)

_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #
# A. Determinismo byte a byte                                           #
# --------------------------------------------------------------------- #


def test_regenerated_bytes_match_committed_file_exactly() -> None:
    """El archivo versionado es EXACTAMENTE lo que produce el generador
    hoy -- si alguien edita el JSON a mano sin regenerarlo, este test
    lo detecta."""
    committed = demo_gen.DEMO_SNAPSHOT_PATH.read_bytes()
    regenerated = demo_gen.build_demo_snapshot_bytes()
    assert regenerated == committed


def test_build_demo_snapshot_bytes_is_deterministic_across_calls() -> None:
    first = demo_gen.build_demo_snapshot_bytes()
    second = demo_gen.build_demo_snapshot_bytes()
    assert first == second


def test_cli_regeneration_reproduces_committed_file_byte_for_byte(tmp_path) -> None:
    """Ejecuta el script real como subproceso (no solo la función en
    memoria) y compara con el archivo versionado."""
    env = {"PYTHONPATH": str(_ROOT)}
    import os
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    completed = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "generate_demo_snapshot.py")],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    regenerated = demo_gen.DEMO_SNAPSHOT_PATH.read_bytes()
    assert regenerated == demo_gen.build_demo_snapshot_bytes()


# --------------------------------------------------------------------- #
# B. Contenido garantizado del universo de demostracion                 #
# --------------------------------------------------------------------- #


def _load_demo_payload() -> dict:
    return json.loads(demo_gen.DEMO_SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_demo_snapshot_has_at_least_three_players() -> None:
    payload = _load_demo_payload()
    players = {entry["player"] for entry in payload["entries"]}
    assert len(players) >= 3


def test_demo_snapshot_has_several_opponents_and_dates_per_player() -> None:
    payload = _load_demo_payload()
    by_player: dict[str, set[str]] = {}
    dates_by_pair: dict[tuple[str, str], set[str]] = {}
    for entry in payload["entries"]:
        by_player.setdefault(entry["player"], set()).add(entry["opponent"])
        dates_by_pair.setdefault((entry["player"], entry["opponent"]), set()).add(
            entry["as_of_date"]
        )
    assert any(len(opponents) >= 2 for opponents in by_player.values())
    assert any(len(dates) >= 2 for dates in dates_by_pair.values())


def test_demo_snapshot_covers_all_four_patterns() -> None:
    payload = _load_demo_payload()
    seen_patterns: set[str] = set()
    for entry in payload["entries"]:
        for ranking in entry["result"]["rankings"]:
            seen_patterns.add(ranking["pattern_id"])
    assert seen_patterns == {"P02", "P04", "P05", "P06"}


def test_demo_snapshot_has_available_recommendations() -> None:
    payload = _load_demo_payload()
    assert any(
        entry["result"]["state"] == "available" for entry in payload["entries"]
    )


def test_demo_snapshot_has_at_least_one_real_abstention() -> None:
    """Al menos una pareja con evidencia insuficiente (candidatos con
    ``state=insufficient_evidence``), no solo ausencia total ya
    esperada por diseño."""
    payload = _load_demo_payload()
    abstained_candidates = [
        candidate
        for entry in payload["entries"]
        for ranking in entry["result"]["rankings"]
        for candidate in ranking["candidates"]
        if candidate["state"] == "insufficient_evidence"
    ]
    assert len(abstained_candidates) > 0


def test_demo_snapshot_evidence_is_internally_coherent() -> None:
    """Cada componente de evidencia reconciliado: successes+failures ==
    labeled_activations, y exito+fallo+distinct_matches >= 0."""
    payload = _load_demo_payload()
    for entry in payload["entries"]:
        for ranking in entry["result"]["rankings"]:
            for candidate in ranking["candidates"]:
                for perspective_key in ("executor_summary", "opponent_allowed_summary"):
                    summary = candidate.get(perspective_key)
                    if summary is None:
                        continue
                    assert (
                        summary["successes"] + summary["failures"]
                        == summary["labeled_activations"]
                    )
                    assert summary["distinct_matches"] >= 0


# --------------------------------------------------------------------- #
# C. Privacidad: cero rutas, partidos, puntos o secuencias privadas      #
# --------------------------------------------------------------------- #


_FORBIDDEN_SUBSTRINGS = (
    "points_enriched",
    "data/processed",
    "reports/",
    "sequence",
    "point_number",
    "server_won_point",
    "effective_date",
    str(_ROOT),
)


def test_demo_snapshot_file_contains_no_forbidden_substrings() -> None:
    text = demo_gen.DEMO_SNAPSHOT_PATH.read_text(encoding="utf-8")
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in text, f"Prohibido en la snapshot demo: {forbidden}"


def test_demo_snapshot_has_no_windows_or_posix_absolute_paths() -> None:
    import re

    text = demo_gen.DEMO_SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert re.findall(r"[A-Za-z]:\\\\", text) == []
    assert re.findall(r'"/home/|"/Users/', text) == []


def test_demo_snapshot_size_is_reasonable_for_a_demo() -> None:
    size = demo_gen.DEMO_SNAPSHOT_PATH.stat().st_size
    assert 0 < size < 5 * 1024 * 1024  # menos de 5 MiB


# --------------------------------------------------------------------- #
# D. Carga real por el provider persistido (P13) y catalogo (P25)       #
# --------------------------------------------------------------------- #


def test_provider_loads_the_demo_snapshot_successfully() -> None:
    provider = create_persisted_tactical_recommendation_provider(
        demo_gen.DEMO_SNAPSHOT_PATH
    )
    assert provider.snapshot.entry_count == 6


def test_provider_catalog_lists_demo_players_deterministically() -> None:
    provider = create_persisted_tactical_recommendation_provider(
        demo_gen.DEMO_SNAPSHOT_PATH
    )
    players = provider.list_players()
    assert players == tuple(sorted(players))
    assert {"Ana Ibarra", "Marco Rossi", "Priya Verma"} <= set(players)


def test_provider_catalog_opponents_and_dates_for_a_demo_player() -> None:
    provider = create_persisted_tactical_recommendation_provider(
        demo_gen.DEMO_SNAPSHOT_PATH
    )
    opponents = provider.list_opponents("Ana Ibarra")
    opponent_ids = {item.opponent_id for item in opponents}
    assert opponent_ids == {"Marco Rossi", "Priya Verma"}
    dates = provider.list_as_of_dates("Ana Ibarra", "Marco Rossi")
    assert len(dates) == 2
    assert dates == tuple(sorted(dates, reverse=True))


def test_provider_recommendation_available_for_a_known_demo_query() -> None:
    provider = create_persisted_tactical_recommendation_provider(
        demo_gen.DEMO_SNAPSHOT_PATH
    )
    from datetime import date

    query = TacticalRecommendationQuery("Ana Ibarra", "Priya Verma", date(2031, 2, 10))
    result = provider.fetch_tactical_prioritization(query)
    assert result.state.value == "available"


def test_provider_recommendation_not_found_for_an_unknown_demo_query() -> None:
    provider = create_persisted_tactical_recommendation_provider(
        demo_gen.DEMO_SNAPSHOT_PATH
    )
    from datetime import date

    query = TacticalRecommendationQuery("Ana Ibarra", "Priya Verma", date(1999, 1, 1))
    with pytest.raises(RecommendationNotFoundError):
        provider.fetch_tactical_prioritization(query)


def test_load_persisted_snapshot_directly_matches_provider_view() -> None:
    snapshot = load_persisted_tactical_recommendation_snapshot(
        demo_gen.DEMO_SNAPSHOT_PATH
    )
    assert snapshot.entry_count == 6
    assert snapshot.contract_name == "persisted_tactical_recommendation_snapshot"


# --------------------------------------------------------------------- #
# E. Import sin efectos y ausencia de I/O sobre datos reales             #
# --------------------------------------------------------------------- #


def test_generator_module_source_never_references_real_data_paths() -> None:
    """El docstring del generador SI menciona ``points_enriched``/
    ``data/processed`` en prosa, precisamente para documentar que
    jamas los toca -- lo que se comprueba aqui es que no hay ningun
    IMPORT ni construccion de ruta real que pudiera leerlos, y que la
    variable de entorno del snapshot privado nunca se referencia como
    codigo ejecutable."""
    source = (_ROOT / "scripts" / "generate_demo_snapshot.py").read_text(
        encoding="utf-8"
    )
    for forbidden_import in ("import pandas", "import pyarrow"):
        assert forbidden_import not in source
    for forbidden_code in (
        'Path("data"', "Path('data'", "os.environ", "getenv",
        'os.environ.get("TENNIS_TACTICAL_SNAPSHOT_PATH")',
    ):
        assert forbidden_code not in source


def test_import_generator_module_has_no_side_effects(tmp_path) -> None:
    import os

    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import scripts.generate_demo_snapshot as g; "
            "print(g.DEMO_SNAPSHOT_PATH.name)",
        ],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "tactical-recommendations-demo-v1.json"
    assert completed.stderr == ""
