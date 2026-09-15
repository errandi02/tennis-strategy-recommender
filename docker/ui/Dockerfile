# Imagen ui (P18/P19): Streamlit sobre el servicio api interno.
# Base fijada por digest para reproducibilidad (python:3.12-slim-bookworm).
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Usuario no-root con UID/GID explicitos; sin shell de login.
RUN groupadd --gid 10002 tennisui \
    && useradd --uid 10002 --gid 10002 --no-create-home --shell /usr/sbin/nologin tennisui

WORKDIR /app

COPY docker/ui/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# COPY minimo: solo el entrypoint de contenedor P19, el modulo P18 del
# que depende y el cierre stdlib puro de src/recommender que su import
# de tactical_recommendation_contract requiere (sin datos, reports,
# tests ni notebooks; nunca el snapshot privado).
COPY src/__init__.py ./src/__init__.py
COPY src/ui ./src/ui
COPY src/recommender ./src/recommender
COPY src/analysis/__init__.py ./src/analysis/__init__.py
COPY src/analysis/first_serve_direction_classification.py ./src/analysis/first_serve_direction_classification.py
COPY src/analysis/return_depth_feasibility.py ./src/analysis/return_depth_feasibility.py
COPY src/analysis/return_direction_feasibility.py ./src/analysis/return_direction_feasibility.py
COPY src/analysis/return_profile_feasibility.py ./src/analysis/return_profile_feasibility.py
COPY src/analysis/return_shot_type_feasibility.py ./src/analysis/return_shot_type_feasibility.py
COPY src/parsing ./src/parsing

USER 10002:10002

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "src/ui/streamlit_container_app.py", \
    "--server.address=0.0.0.0", "--server.port=8501", \
    "--browser.gatherUsageStats=false", "--server.headless=true"]
