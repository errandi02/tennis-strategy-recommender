# Imagen api (P17/P19): FastAPI P12 sobre el snapshot privado P13.
# Base fijada por digest para reproducibilidad (python:3.12-slim-bookworm).
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

# Usuario no-root con UID/GID explicitos; sin shell de login.
RUN groupadd --gid 10001 tennisapi \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin tennisapi

WORKDIR /app

COPY docker/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# COPY minimo: solo los paquetes que el runtime P17/P12 importa
# transitivamente (src/analysis y src/parsing son cierre stdlib puro
# de src/recommender; no se copian datos, reports, tests ni notebooks).
COPY src/__init__.py ./src/__init__.py
COPY src/api ./src/api
COPY src/recommender ./src/recommender
COPY src/analysis/__init__.py ./src/analysis/__init__.py
COPY src/analysis/first_serve_direction_classification.py ./src/analysis/first_serve_direction_classification.py
COPY src/analysis/return_depth_feasibility.py ./src/analysis/return_depth_feasibility.py
COPY src/analysis/return_direction_feasibility.py ./src/analysis/return_direction_feasibility.py
COPY src/analysis/return_profile_feasibility.py ./src/analysis/return_profile_feasibility.py
COPY src/analysis/return_shot_type_feasibility.py ./src/analysis/return_shot_type_feasibility.py
COPY src/parsing ./src/parsing
COPY docker/api/healthcheck.py ./healthcheck.py

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "/app/healthcheck.py"]

ENTRYPOINT ["python", "-m", "src.api.runtime"]
