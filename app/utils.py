"""
Utilitaires pour l'app Streamlit déployée.

Cette version remplace l'ancien `utils.py` qui chargeait le corpus
directement depuis S3. Désormais, l'app Streamlit ne fait plus que
des appels HTTP vers le backend Flask (`api.py`). Tout le calcul
lourd (annotation spaCy, similarités, lecture S3) est côté serveur.

Bénéfices :
- L'app démarre instantanément (pas de chargement de 26 Parquet).
- Les paramètres TF-IDF / MMR sont figés côté serveur, donc
  l'utilisateur final ne peut pas se tirer une balle dans le pied
  en choisissant une mauvaise config.
- Le backend peut être mis à l'échelle indépendamment de Streamlit.

L'URL de l'API est lue depuis la variable d'environnement `API_URL`.
Par défaut `http://localhost:5000`, ce qui marche à la fois pour le
développement local (Flask + Streamlit lancés séparément) et pour
le conteneur Docker Scaleway (les deux services dans le même
container partagent localhost).
"""
import os

import requests
import streamlit as st


# ============================================================================
# Configuration
# ============================================================================

API_URL = os.environ.get("API_URL", "http://localhost:5000")
TIMEOUT_DEFAULT = 30  # secondes -- généreux pour le premier /api/identifier
                     # qui charge spaCy (~3s)


# ============================================================================
# Client HTTP
# ============================================================================

def _get(path, **kwargs):
    """GET sur l'API. Renvoie le JSON ou lève une exception claire."""
    url = f"{API_URL}{path}"
    try:
        r = requests.get(url, timeout=TIMEOUT_DEFAULT, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Impossible de joindre l'API à {API_URL}. "
            "Vérifie que le backend Flask est lancé."
        )
    except requests.exceptions.HTTPError as e:
        # On essaie de récupérer le message d'erreur structuré
        try:
            detail = r.json().get("error", str(e))
        except Exception:
            detail = str(e)
        raise RuntimeError(detail)


def _post(path, payload, **kwargs):
    """POST JSON sur l'API."""
    url = f"{API_URL}{path}"
    try:
        r = requests.post(
            url, json=payload, timeout=TIMEOUT_DEFAULT, **kwargs,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Impossible de joindre l'API à {API_URL}. "
            "Vérifie que le backend Flask est lancé."
        )
    except requests.exceptions.HTTPError as e:
        try:
            detail = r.json().get("error", str(e))
        except Exception:
            detail = str(e)
        raise RuntimeError(detail)


# ============================================================================
# Endpoints API (avec cache Streamlit)
# ============================================================================

@st.cache_data(ttl=10, show_spinner=False)
def api_health():
    """Healthcheck de l'API. Cache 10s pour ne pas spammer le serveur."""
    return _get("/health")


@st.cache_data(ttl=300, show_spinner="Chargement du catalogue...")
def api_get_livres():
    """Liste les livres du corpus. Cache 5 min (le corpus ne bouge pas)."""
    return _get("/api/livres")["livres"]


def api_identifier(extrait, top_k=5):
    """
    Identifie un livre depuis un extrait. Pas de cache : chaque extrait
    est unique et le calcul est rapide côté serveur (~200 ms après
    chargement de spaCy).
    """
    return _post(
        "/api/identifier",
        {"extrait": extrait, "top_k": top_k},
    )


@st.cache_data(ttl=3600, show_spinner=False)
def api_get_resume(livre_slug):
    """
    Récupère le résumé MMR pré-généré. Cache 1h (résumés figés sur S3
    tant qu'on ne relance pas rebuild_artifacts)."""
    return _get(f"/api/resume/{livre_slug}")["resume"]


def api_extrait_aleatoire(livre_slug, taille=80):
    """
    Tire un extrait aléatoire d'un livre. Pas de cache : on veut un
    extrait différent à chaque clic du bouton.
    """
    return _get(f"/api/extrait/{livre_slug}?taille={taille}")["extrait"]


def reset_caches():
    """Vide les caches Streamlit (utile pour forcer un refresh)."""
    st.cache_data.clear()
    st.cache_resource.clear()


# ============================================================================
# Helpers de présentation
# ============================================================================

def format_livre(livre):
    """
    Format compact pour affichage en liste : 'Auteur -- Livre (genre)'.

    `livre` est un dict tel que renvoyé par /api/livres
    (clés : auteur, livre, genre, livre_slug, ...).
    """
    return f"{livre['auteur']} -- {livre['livre']} ({livre['genre']})"