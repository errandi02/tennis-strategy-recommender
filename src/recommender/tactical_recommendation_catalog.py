"""Catalogo de descubrimiento P25: jugadores, rivales y fechas de corte
validas, derivados EXCLUSIVAMENTE del mismo snapshot persistido y
validado que ya usa P12/P17 para servir recomendaciones (P13). Este
modulo no abre Parquet/CSV, no ejecuta P10, no lee el test sellado
2024-2026 ni ninguna fuente alternativa, y no importa el modulo P13
(para evitar un ciclo de import: P13 importa este modulo para exponer
los tres metodos de catalogo sobre el provider persistido). Opera
exclusivamente sobre triples ``(player, opponent, as_of_date)`` que el
llamante ya tiene en memoria -- exactamente las mismas claves que
sirven cada recomendacion real.

No inventa informacion: cada jugador, rival o fecha que devuelve este
modulo corresponde a una entrada real y ya validada del snapshot, por
lo que una recomendacion posterior con esos tres valores nunca
devolvera "no encontrada" por causa del catalogo.

Limites de cardinalidad cerrados y deterministas (``MAX_CATALOG_*``):
si el snapshot legitimamente los superase, la funcion falla de forma
cerrada (``CatalogTooLargeError``) en vez de truncar en silencio -- un
catalogo truncado induciria a error a quien lo usa para construir una
consulta valida.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Final, Protocol, runtime_checkable


CATALOG_CONTRACT_VERSION: Final = "1.0.0"

# Duplicado deliberadamente del valor de
# ``tactical_recommendation_service.MAX_IDENTIFIER_LENGTH``: este
# modulo no puede importar ese modulo de servicio sin crear un ciclo
# (P13 importa este modulo, y el modulo de servicio es importado por
# P13 tambien). Mismo patron ya usado entre P13 y el servicio.
MAX_CATALOG_IDENTIFIER_LENGTH: Final = 64

# Cardinalidad razonable para un catalogo de descubrimiento: muy por
# encima del universo de diseno real (P16: 3.610 entradas, 1.805
# parejas jugador-rival) pero acotada para que un snapshot anomalo
# falle de forma cerrada en vez de producir una respuesta HTTP de
# tamano arbitrario.
MAX_CATALOG_PLAYERS: Final = 5_000
MAX_CATALOG_OPPONENTS_PER_PLAYER: Final = 2_000
MAX_CATALOG_DATES_PER_MATCHUP: Final = 2_000

_CATALOG_CONTROL_CHARACTERS: Final = frozenset(
    tuple(range(0x00, 0x20)) + (0x7F,) + tuple(range(0x80, 0xA0))
)
_CATALOG_FORBIDDEN_FRAGMENTS: Final = ("/", "\\", "://", "..", "~")


class CatalogError(RuntimeError):
    """Fallo cerrado del catalogo de descubrimiento P25."""

    __slots__ = ()


class CatalogInvalidIdentifierError(CatalogError):
    """El identificador de entrada no cumple el contrato cerrado."""

    __slots__ = ()


class CatalogTooLargeError(CatalogError):
    """El catalogo resultante supera el limite cerrado de cardinalidad."""

    __slots__ = ()


def _strict_catalog_identifier(value: object) -> str:
    if type(value) is not str:
        raise CatalogInvalidIdentifierError()
    if (
        not value
        or value != value.strip()
        or len(value) > MAX_CATALOG_IDENTIFIER_LENGTH
    ):
        raise CatalogInvalidIdentifierError()
    if any(ord(character) in _CATALOG_CONTROL_CHARACTERS for character in value):
        raise CatalogInvalidIdentifierError()
    if any(fragment in value for fragment in _CATALOG_FORBIDDEN_FRAGMENTS):
        raise CatalogInvalidIdentifierError()
    return value


@dataclass(frozen=True, slots=True)
class TacticalOpponentSummary:
    """Rival real de un jugador, con el numero de fechas de corte
    validas para esa pareja exacta -- nunca inventado: es la
    cardinalidad exacta de entradas del snapshot para esa pareja."""

    opponent_id: str
    matchup_count: int


def catalog_players(keys: Iterable[tuple[str, str, date]]) -> tuple[str, ...]:
    """Jugadores distintos para los que existe al menos una
    recomendacion real (``key.player`` de alguna entrada), en orden
    alfabetico determinista."""
    players = sorted({player for player, _opponent, _as_of_date in keys})
    if len(players) > MAX_CATALOG_PLAYERS:
        raise CatalogTooLargeError()
    return tuple(players)


def catalog_opponents(
    keys: Iterable[tuple[str, str, date]], player_id: str
) -> tuple[TacticalOpponentSummary, ...]:
    """Rivales reales de ``player_id``, ordenados alfabeticamente por
    identificador, con el numero de fechas de corte validas por
    pareja. Tupla vacia si ``player_id`` no tiene ninguna entrada."""
    identifier = _strict_catalog_identifier(player_id)
    counts: dict[str, int] = {}
    for player, opponent, _as_of_date in keys:
        if player == identifier:
            counts[opponent] = counts.get(opponent, 0) + 1
    if len(counts) > MAX_CATALOG_OPPONENTS_PER_PLAYER:
        raise CatalogTooLargeError()
    return tuple(
        TacticalOpponentSummary(opponent_id=opponent, matchup_count=counts[opponent])
        for opponent in sorted(counts)
    )


def catalog_as_of_dates(
    keys: Iterable[tuple[str, str, date]], player_id: str, opponent_id: str
) -> tuple[date, ...]:
    """Fechas de corte (``as_of_date``) validas para la pareja exacta
    ``(player_id, opponent_id)``, de mas reciente a mas antigua. Tupla
    vacia si la pareja no tiene ninguna entrada."""
    player = _strict_catalog_identifier(player_id)
    opponent = _strict_catalog_identifier(opponent_id)
    dates = sorted(
        (
            as_of_date
            for entry_player, entry_opponent, as_of_date in keys
            if entry_player == player and entry_opponent == opponent
        ),
        reverse=True,
    )
    if len(dates) > MAX_CATALOG_DATES_PER_MATCHUP:
        raise CatalogTooLargeError()
    return tuple(dates)


@runtime_checkable
class TacticalCatalogProvider(Protocol):
    """Backend inyectable del catalogo de descubrimiento P25.

    Misma disciplina que ``TacticalRecommendationProvider`` (P12): no
    depende de FastAPI, no ejecuta P10, no mantiene estado global
    mutable. Debe derivarse exclusivamente del mismo snapshot
    persistido y validado que ``fetch_tactical_prioritization``."""

    def list_players(self) -> tuple[str, ...]:
        """Jugadores distintos con al menos una recomendacion real."""

    def list_opponents(self, player_id: str) -> tuple[TacticalOpponentSummary, ...]:
        """Rivales reales de ``player_id``, con numero de fechas por pareja."""

    def list_as_of_dates(
        self, player_id: str, opponent_id: str
    ) -> tuple[date, ...]:
        """Fechas de corte validas para la pareja exacta, mas reciente primero."""


__all__ = (
    "CATALOG_CONTRACT_VERSION",
    "MAX_CATALOG_DATES_PER_MATCHUP",
    "MAX_CATALOG_IDENTIFIER_LENGTH",
    "MAX_CATALOG_OPPONENTS_PER_PLAYER",
    "MAX_CATALOG_PLAYERS",
    "CatalogError",
    "CatalogInvalidIdentifierError",
    "CatalogTooLargeError",
    "TacticalCatalogProvider",
    "TacticalOpponentSummary",
    "catalog_as_of_dates",
    "catalog_opponents",
    "catalog_players",
)
