#!/bin/bash
# =============================================================================
# Demarre le backend Flask (gunicorn) et le frontend Streamlit.
# =============================================================================
#
# Architecture : un seul container Docker, deux process.
# - gunicorn sert l'API Flask sur le port 5000 (acces interne)
# - streamlit sert l'UI sur le port 8501 (expose a l'utilisateur)
# - Streamlit appelle Flask via http://localhost:5000 (meme container)
#
# Le `--workers 1 --preload` est volontaire :
# - 1 worker : on a peu de trafic (app demo), 1 worker suffit largement
# - --preload : charge le bundle UNE seule fois avant de fork le worker.
#   Plus rapide au demarrage et moins gourmand en RAM.
#
# Si gunicorn plante, on tue Streamlit pour que Docker redemarre tout.

set -e

echo "[start.sh] Lancement de gunicorn (Flask)..."
gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 1 \
    --preload \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    api:app &

GUNICORN_PID=$!
echo "[start.sh] gunicorn PID = $GUNICORN_PID"

# On attend que /health reponde, sinon Streamlit va planter au premier
# appel API. ~10 secondes en pratique (telechargement du bundle + resumes).
echo "[start.sh] Attente que l'API soit prete..."
for i in {1..30}; do
    if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
        echo "[start.sh] API prete apres ${i}s."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "[start.sh] API toujours pas prete apres 30s -- abandon."
        kill $GUNICORN_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Trap pour propager le signal d'arret a gunicorn quand Streamlit s'arrete
trap "echo '[start.sh] Arret demande...'; kill $GUNICORN_PID 2>/dev/null || true; exit 0" SIGTERM SIGINT

echo "[start.sh] Lancement de Streamlit..."
exec streamlit run app/Home.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false