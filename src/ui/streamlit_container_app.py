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
cerrada: no renderiza el asistente ni permite ninguna peticion, y no
revela la variable, su valor ni la URL interna en la UI, logs o
errores.

Solo este modulo lee variables de entorno; ``streamlit_app.py`` (P18)
permanece sin acceso a ``os.environ`` y sin cambios de comportamiento
local. Import sin efectos: no arranca Streamlit, no hace red y no lee
la variable de entorno hasta que ``main()`` se ejecuta explicitamente.

P25: el asistente de 3 pasos (jugador -> rival -> fecha) y el render
del resultado son EXACTAMENTE los mismos que en modo local
(``run_recommendation_experience``); solo cambia como se resuelve la
URL base (constante fija de contenedor, nunca editable)."""

from __future__ import annotations

import logging
import os
from typing import Final

import streamlit as st

from src.ui.streamlit_app import (
    UI_CONTAINER_API_BASE_URL,
    run_recommendation_experience,
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
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    if not is_container_mode_active():
        st.error(_CONTAINER_MESSAGES["config_rejected"])
        return
    run_recommendation_experience(UI_CONTAINER_API_BASE_URL, container_mode=True)


if __name__ == "__main__":
    main()


__all__ = (
    "CONTAINER_RUNTIME_MODE_ENV",
    "CONTAINER_RUNTIME_MODE_VALUE",
    "is_container_mode_active",
    "main",
)
