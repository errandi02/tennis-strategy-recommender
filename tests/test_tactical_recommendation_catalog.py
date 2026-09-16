"""Pruebas sinteticas P25: catalogo de descubrimiento (jugadores,
rivales, fechas de corte) derivado exclusivamente de triples
``(player, opponent, as_of_date)`` en memoria. Ningun test lee
Parquet/CSV/snapshot real ni el test sellado 2024-2026: todas las
claves son literales sintéticos construidos a mano.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.recommender.tactical_recommendation_catalog import (
    MAX_CATALOG_DATES_PER_MATCHUP,
    MAX_CATALOG_IDENTIFIER_LENGTH,
    MAX_CATALOG_OPPONENTS_PER_PLAYER,
    MAX_CATALOG_PLAYERS,
    CatalogInvalidIdentifierError,
    CatalogTooLargeError,
    TacticalCatalogProvider,
    TacticalOpponentSummary,
    catalog_as_of_dates,
    catalog_opponents,
    catalog_players,
)


def _keys(*rows: tuple[str, str, str]) -> tuple[tuple[str, str, date], ...]:
    return tuple(
        (player, opponent, date.fromisoformat(as_of))
        for player, opponent, as_of in rows
    )


_SYNTHETIC_KEYS = _keys(
    ("Alice", "Bob", "2031-01-10"),
    ("Alice", "Bob", "2031-02-20"),
    ("Alice", "Carol", "2031-03-01"),
    ("Bob", "Alice", "2031-04-15"),
    ("Dave", "Alice", "2031-05-05"),
)


# --------------------------------------------------------------------- #
# A. catalog_players: distintos, ordenados, sin inventar                 #
# --------------------------------------------------------------------- #


def test_catalog_players_returns_sorted_distinct_players() -> None:
    assert catalog_players(_SYNTHETIC_KEYS) == ("Alice", "Bob", "Dave")


def test_catalog_players_empty_snapshot_returns_empty_tuple() -> None:
    assert catalog_players(()) == ()


def test_catalog_players_only_counts_the_player_role_not_opponent() -> None:
    """"Carol" solo aparece como rival de Alice, nunca como
    ``key.player`` propio: no debe aparecer como jugador seleccionable
    (no existe ninguna recomendacion real PARA Carol en este catalogo
    sintetico)."""
    players = catalog_players(_SYNTHETIC_KEYS)
    assert "Carol" not in players


def test_catalog_players_too_large_fails_closed() -> None:
    huge = _keys(*((f"Player{i}", "Opponent", "2031-01-01") for i in range(MAX_CATALOG_PLAYERS + 1)))
    with pytest.raises(CatalogTooLargeError):
        catalog_players(huge)


def test_catalog_players_at_exact_limit_succeeds() -> None:
    exact = _keys(*((f"Player{i}", "Opponent", "2031-01-01") for i in range(MAX_CATALOG_PLAYERS)))
    assert len(catalog_players(exact)) == MAX_CATALOG_PLAYERS


# --------------------------------------------------------------------- #
# B. catalog_opponents: reales, ordenados, con conteo honesto            #
# --------------------------------------------------------------------- #


def test_catalog_opponents_returns_sorted_real_opponents_with_counts() -> None:
    result = catalog_opponents(_SYNTHETIC_KEYS, "Alice")
    assert result == (
        TacticalOpponentSummary(opponent_id="Bob", matchup_count=2),
        TacticalOpponentSummary(opponent_id="Carol", matchup_count=1),
    )


def test_catalog_opponents_never_returns_the_player_itself() -> None:
    """Nunca debe poder seleccionarse el mismo jugador como rival."""
    result = catalog_opponents(_SYNTHETIC_KEYS, "Alice")
    assert all(item.opponent_id != "Alice" for item in result)


def test_catalog_opponents_unknown_player_returns_empty_tuple() -> None:
    assert catalog_opponents(_SYNTHETIC_KEYS, "Ghost") == ()


def test_catalog_opponents_matchup_count_is_never_invented() -> None:
    """El conteo es exactamente el numero de entradas reales de esa
    pareja, nunca un valor aproximado o inventado."""
    result = catalog_opponents(_SYNTHETIC_KEYS, "Alice")
    by_id = {item.opponent_id: item.matchup_count for item in result}
    assert by_id["Bob"] == 2
    assert by_id["Carol"] == 1


@pytest.mark.parametrize(
    "bad_id",
    ["", " Alice", "Alice ", "a/b", "a\\b", "a://b", "a..b", "a~b", "x" * (MAX_CATALOG_IDENTIFIER_LENGTH + 1), "\x00Alice"],
)
def test_catalog_opponents_rejects_malformed_identifier(bad_id: str) -> None:
    with pytest.raises(CatalogInvalidIdentifierError):
        catalog_opponents(_SYNTHETIC_KEYS, bad_id)


def test_catalog_opponents_too_large_fails_closed() -> None:
    huge = _keys(*(("Alice", f"Opponent{i}", "2031-01-01") for i in range(MAX_CATALOG_OPPONENTS_PER_PLAYER + 1)))
    with pytest.raises(CatalogTooLargeError):
        catalog_opponents(huge, "Alice")


# --------------------------------------------------------------------- #
# C. catalog_as_of_dates: fechas exactas de la pareja, mas reciente     #
#    primero                                                             #
# --------------------------------------------------------------------- #


def test_catalog_as_of_dates_returns_dates_most_recent_first() -> None:
    result = catalog_as_of_dates(_SYNTHETIC_KEYS, "Alice", "Bob")
    assert result == (date(2031, 2, 20), date(2031, 1, 10))


def test_catalog_as_of_dates_unknown_pair_returns_empty_tuple() -> None:
    assert catalog_as_of_dates(_SYNTHETIC_KEYS, "Alice", "Dave") == ()


def test_catalog_as_of_dates_is_specific_to_the_exact_pair() -> None:
    """Las fechas de (Alice, Bob) no deben mezclarse con las de
    (Alice, Carol) ni con las de (Bob, Alice) -- orientacion exacta."""
    alice_bob = catalog_as_of_dates(_SYNTHETIC_KEYS, "Alice", "Bob")
    alice_carol = catalog_as_of_dates(_SYNTHETIC_KEYS, "Alice", "Carol")
    bob_alice = catalog_as_of_dates(_SYNTHETIC_KEYS, "Bob", "Alice")
    assert set(alice_bob).isdisjoint(alice_carol)
    assert alice_bob != bob_alice


@pytest.mark.parametrize("bad_id", ["", "a/b", "x" * (MAX_CATALOG_IDENTIFIER_LENGTH + 1)])
def test_catalog_as_of_dates_rejects_malformed_identifiers(bad_id: str) -> None:
    with pytest.raises(CatalogInvalidIdentifierError):
        catalog_as_of_dates(_SYNTHETIC_KEYS, bad_id, "Bob")
    with pytest.raises(CatalogInvalidIdentifierError):
        catalog_as_of_dates(_SYNTHETIC_KEYS, "Alice", bad_id)


def test_catalog_as_of_dates_too_large_fails_closed() -> None:
    huge = tuple(
        ("Alice", "Bob", date.fromordinal(730000 + i))
        for i in range(MAX_CATALOG_DATES_PER_MATCHUP + 1)
    )
    with pytest.raises(CatalogTooLargeError):
        catalog_as_of_dates(huge, "Alice", "Bob")


# --------------------------------------------------------------------- #
# D. Protocolo de catalogo: estructura cerrada, sin FastAPI/P10          #
# --------------------------------------------------------------------- #


def test_protocol_is_runtime_checkable_and_requires_all_three_methods() -> None:
    class _Full:
        def list_players(self):
            return ()

        def list_opponents(self, player_id):
            return ()

        def list_as_of_dates(self, player_id, opponent_id):
            return ()

    class _MissingOne:
        def list_players(self):
            return ()

        def list_opponents(self, player_id):
            return ()

    assert isinstance(_Full(), TacticalCatalogProvider)
    assert not isinstance(_MissingOne(), TacticalCatalogProvider)
    assert not isinstance(object(), TacticalCatalogProvider)


def test_module_has_no_io_or_analytical_imports() -> None:
    import ast
    from pathlib import Path
    import src.recommender.tactical_recommendation_catalog as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = {"pandas", "pyarrow", "fastapi", "pickle", "subprocess", "uvicorn"}
    assert not (imported & banned)
    # Ninguna dependencia de P13 (evita el ciclo de import documentado).
    assert "src.recommender.persisted_tactical_recommendation_provider" not in imported


def test_import_has_no_side_effects(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path

    import os

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import src.recommender.tactical_recommendation_catalog as c; "
            "print(c.CATALOG_CONTRACT_VERSION)",
        ],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "1.0.0"
    assert completed.stderr == ""
