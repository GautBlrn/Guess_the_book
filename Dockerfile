# =============================================================================
# Dockerfile pour l'app NLP (identification de livre + résumés).
# =============================================================================
#
# Construit une image qui contient :
# - Python 3.12 slim
# - Les deps de requirements.txt (flask, streamlit, spacy, etc.)
# - Le modèle spaCy fr_core_news_sm pré-téléchargé
# - Le code du projet (api.py, app/, pipeline/, scripts/)
#
# Au démarrage du container, start.sh lance gunicorn (Flask) puis Streamlit.
# Le bundle de serving (artifacts/serving_bundle.pkl) est téléchargé depuis
# S3 à chaque démarrage : pas embarqué dans l'image, ce qui permet de
# régénérer les artéfacts sans rebuild Docker.

FROM python:3.12-slim

# --- Variables d'environnement Python ---
# PYTHONDONTWRITEBYTECODE = pas de fichiers .pyc sur disque
# PYTHONUNBUFFERED        = logs Python visibles en temps reel via `docker logs`
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Dependances systeme ---
# curl : utilise par start.sh pour le healthcheck d'attente
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- Dependances Python ---
# On copie d'abord requirements.txt seulement, pour profiter du cache
# Docker : tant que requirements.txt ne change pas, on ne reinstalle pas.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Modèle spaCy ---
# Téléchargement direct via la cmd spacy. ~12 Mo, ajoute à l'image au build
# pour que le container démarre vite (pas de téléchargement au runtime).
RUN python -m spacy download fr_core_news_sm

# --- Code du projet ---
# On copie en dernier pour que les changements de code n'invalident pas
# les couches précédentes (deps, spacy).
COPY api.py start.sh ./
COPY app/ ./app/
COPY pipeline/ ./pipeline/

# --- Permissions ---
RUN chmod +x start.sh

# --- Variables d'environnement de l'app ---
# API_URL : Streamlit appelle Flask sur localhost (même container).
# Les crédentials S3 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# AWS_S3_ENDPOINT_URL, AWS_DEFAULT_REGION) sont passés au runtime via
# `docker run --env-file .env`, PAS embarqués dans l'image.
ENV API_URL=http://localhost:5000

# --- Ports ---
# 5000 : API Flask (interne, mais exposé pour débug curl depuis l'hôte)
# 8501 : Streamlit (UI utilisateur)
EXPOSE 5000 8501

# --- Healthcheck ---
# Docker / Scaleway peuvent utiliser ca pour savoir si le container est sain.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:5000/health || exit 1

CMD ["./start.sh"]