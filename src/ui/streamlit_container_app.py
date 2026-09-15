"""UI Streamlit P19: entrypoint de contenedor sobre la API P17 interna.

Capa minima de arranque sobre P18 (``src.ui.streamlit_app``) pensada
EXCLUSIVAMENTE para la imagen Docker ``ui`` de P19. En lugar de una URL
editable por el usuario (como en modo local), este modulo resuelve el
servicio EXCLUSIVAMENTE a partir de la variable de entorno dedicada
``TENNIS_TACTICAL_RUNTIME_MODE``, con el unico valor exacto aceptado
``container``. La variable decide SOLO si el modulo activa el modo
contenedor; el host de destino (``http://api:8000``, el servicio
interno de Docker Compose) es una constante fija de
``src.ui.streamlit_app`` que nunca proviene del entorno ni de ninguna
entrada del usuario. Sin la variable, o con cualquier valor distinto de
``container`` (vacio, manipulado o desconocido), la interfaz falla
cerrada: no renderiza el formulario ni permite ninguna peticion, y no
revela la variable, su valor ni la URL interna en la UI, logs o
errores.

Solo este modulo lee variables de entorno; ``streamlit_app.py`` (P18)
permanece sin acceso a ``os.environ`` y sin cambios de comportamiento
local. Import sin efectos: no arranca Streamlit, no hace red y no lee
la variable de entorno hasta que ``main()`` se ejecuta explicitamente.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Final

import httpx

import streamlit as st

from src.ui.streamlit_app import (
    UI_CONTAINER_API_BASE_URL,
    UI_REQUEST_TIMEOUT_SECONDS,
    fetch_recommendation,
    form_inputs,
    outcome_kind_label,
    render_public_recommendation,
    validate_local_date,
    validate_local_identifier,
)


CONTAINER_RUNTIME_MODE_ENV: Final = "TENNIS_TACTICAL_RUNTIME_MODE"
CONTAINER_RUNTIME_MODE_VALUE: Final = "container"

_CONTAINER_MESSAGES: Final = {
    "config_rejected": (
        "No se pudo iniciar la interfaz (configuracion local rechazada)."
    ),
}


def is_container_mode_active() -> bool:
    """True solo con el valor exacto cerrado; cualquier otra cosa, falso."""
    return os.environ.get(CONTAINER_RUNTIME_MODE_ENV) == CONTAINER_RUNTIME_MODE_VALUE


def main() -> None:
    # HTTPX/httpcore registran la URL a nivel INFO/DEBUG; se desactivan
    # antes de construir cualquier cliente, igual que en P18 local.
    logging.getLogger("httpx").disabled = True
    logging.getLogger("httpcore").disabled = True
    st.set_page_config(
        page_title="Recomendador táctico (contenedor)",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if not is_container_mode_active():
        st.error(_CONTAINER_MESSAGES["config_rejected"])
        return
    st.title("Recomendador táctico de tenis (evidencia histórica)")
    st.caption(
        "Interfaz en contenedor sobre la API P17 interna. Las fichas son "
        "evidencia histórica observacional: no afirman causalidad ni "
        "garantizan éxito. El servicio se resuelve automáticamente; no "
        "es configurable desde la interfaz."
    )
    player_raw, opponent_raw, chosen_date, submitted = form_inputs()
    if not submitted:
        st.caption(
            "Introduce jugador, rival y fecha y pulsa "
            "«Solicitar recomendación»."
        )
        return
    try:
        player = validate_local_identifier(player_raw, "player")
        opponent = validate_local_identifier(opponent_raw, "opponent")
    except ValueError as error:
        st.error(str(error))
        return
    if player == opponent:
        st.error("Jugador y rival deben ser diferentes.")
        return
    if isinstance(chosen_date, datetime):
        as_of_raw = chosen_date.date().isoformat()
    elif isinstance(chosen_date, date):
        as_of_raw = chosen_date.isoformat()
    else:
        as_of_raw = ""
    try:
        as_of = validate_local_date(as_of_raw)
    except ValueError as error:
        st.error(str(error))
        return
    client = httpx.Client(
        timeout=UI_REQUEST_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
    )
    try:
        with st.spinner("Consultando el servicio…"):
            outcome = fetch_recommendation(
                client,
                UI_CONTAINER_API_BASE_URL,
                player,
                opponent,
                as_of,
                container_mode=True,
            )
    finally:
        client.close()
    if outcome.model is not None:
        render_public_recommendation(outcome.model)
        return
    label = outcome_kind_label(outcome)
    st.error(f"{outcome.message} {label}".strip())


if __name__ == "__main__":
    main()


__all__ = (
    "CONTAINER_RUNTIME_MODE_ENV",
    "CONTAINER_RUNTIME_MODE_VALUE",
    "is_container_mode_active",
    "main",
)
