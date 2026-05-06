"""
Backend Flask pour l'application "Identifier un livre".

Au démarrage, le serveur :
1. Télécharge le bundle de serving depuis S3 (artifacts/serving_bundle.pkl).
   Ça ramène en mémoire le vectoriseur TF-IDF, la matrice du corpus
   L2-normalisée, et l'index des livres. ~5 secondes pour ~4 Mo.
2. Pré-charge en RAM tous les résumés MMR (26 fichiers .txt, ~150 Ko total).
3. Charge le pipeline d'annotation spaCy `fr_core_news_sm` (~12 Mo, lazy
   au premier appel à /api/identifier pour ne pas ralentir le démarrage
   du conteneur — le healthcheck répond plus vite).

Une fois prêt, le serveur ne touche plus à S3 sauf si on le redémarre.

Routes exposées :

  GET  /health                              -> {"status": "ok"} (et "ready": bool)
  GET  /api/livres                          -> liste l'index du corpus
  POST /api/identifier                      -> top-5 livres pour un extrait
  GET  /api/resume/<livre_slug>             -> résumé MMR pré-généré
  GET  /api/extrait/<livre_slug>?taille=N   -> extrait aléatoire de N mots

Les paramètres MMR / TF-IDF / métrique sont figés au build du bundle.
L'utilisateur final ne peut pas les changer : c'est volontaire pour
garder l'app simple et toujours utiliser la meilleure config connue.

Lancement local :

    python api.py

    # Ou via gunicorn pour la prod :
    gunicorn --bind 0.0.0.0:5000 --workers 2 api:app
"""
import io
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import storage
from pipeline.annotation import annoter, extraire_termes
from pipeline.config import PREFIXES
from pipeline.representations import similarites_cosinus
from pipeline.tokenization import tokeniser


# ============================================================================
# Configuration
# ============================================================================

BUNDLE_KEY = f"{PREFIXES['artifacts']}serving_bundle.pkl"
TOP_K_DEFAULT = 5
TOP_K_MAX = 10  # plafond pour éviter qu'un client demande tout le corpus

# Logger basique : utile pour `docker logs -f` côté Scaleway
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("api")


# ============================================================================
# État global (chargé au démarrage)
# ============================================================================

# Convention : tous les attributs en lowercase, accessibles via STATE.x
class State:
    bundle = None        # dict {version, config, vectoriseur, matrice, index}
    resumes = {}         # {livre_slug: contenu_texte}
    nlp_loaded = False   # spaCy est lazy-loaded au premier /api/identifier


STATE = State()


# ============================================================================
# Chargement initial
# ============================================================================

def charger_bundle():
    """Récupère le bundle depuis S3 et l'unpickle en mémoire."""
    log.info(f"Téléchargement du bundle depuis s3://{BUNDLE_KEY}...")
    t0 = time.time()
    octets = storage.get_bytes(BUNDLE_KEY)
    bundle = pickle.loads(octets)
    log.info(
        f"  bundle chargé en {time.time() - t0:.1f}s : "
        f"{len(bundle['index'])} livres, "
        f"vocab {len(bundle['vectoriseur'].vocabulaire_):,}, "
        f"matrice {bundle['matrice'].shape}"
    )
    return bundle


def precharger_resumes(index):
    """Lit tous les résumés MMR depuis S3 et les met en cache RAM."""
    log.info(f"Préchargement de {len(index)} résumés depuis S3...")
    t0 = time.time()
    resumes = {}
    n_ok, n_ko = 0, 0
    for entree in index:
        try:
            texte = storage.get_text(entree["cle_resume"])
            resumes[entree["livre_slug"]] = texte
            n_ok += 1
        except Exception as e:
            # Un résumé manquant n'empêche pas le serveur de démarrer.
            log.warning(f"  résumé manquant pour {entree['livre_slug']} : {e}")
            n_ko += 1
    log.info(
        f"  {n_ok} résumés chargés en {time.time() - t0:.1f}s "
        f"({n_ko} échec{'s' if n_ko > 1 else ''})"
    )
    return resumes


def initialiser():
    """Charge bundle + résumés. Appelé une fois au démarrage."""
    storage.ensure_bucket()
    STATE.bundle = charger_bundle()
    STATE.resumes = precharger_resumes(STATE.bundle["index"])
    log.info("Serveur prêt à servir des requêtes.")


# ============================================================================
# Logique métier
# ============================================================================

def vectoriser_extrait(extrait):
    """
    Annote l'extrait avec spaCy, extrait les lemmes, vectorise via le
    TF-IDF du bundle. Renvoie un vecteur (V,) L2-normalisé.

    Premier appel : charge spaCy `fr_core_news_sm` (~2 secondes).
    Appels suivants : annotation immédiate.
    """
    if not STATE.nlp_loaded:
        # `annoter()` charge le modèle spaCy en lazy via @lru_cache,
        # donc le premier appel paie le coût d'initialisation.
        log.info("Premier appel : chargement de spaCy...")
        STATE.nlp_loaded = True

    df = annoter(extrait)
    champ = STATE.bundle["config"]["champ"]
    # Le bundle a été construit avec champ="lemmes" par défaut. On extrait
    # la même chose ici pour rester cohérent.
    if champ == "lemmes":
        termes = extraire_termes(df, champ="lemma")
    else:  # tokens
        termes = extraire_termes(df, champ="text")

    if not termes:
        return None

    vec = STATE.bundle["vectoriseur"]
    # transform attend une liste de documents (chaque doc = liste de termes).
    matrice = vec.transform([termes])
    return matrice[0]  # vecteur (V,) L2-normalisé


def classer_extrait(extrait, top_k=TOP_K_DEFAULT):
    """
    Annote, vectorise, et compare au corpus. Renvoie une liste de dicts
    {livre, auteur, genre, score, livre_slug} triée par score décroissant.
    """
    requete = vectoriser_extrait(extrait)
    if requete is None:
        return []

    matrice = STATE.bundle["matrice"]
    scores = similarites_cosinus(matrice, requete)

    index = STATE.bundle["index"]
    # argsort décroissant, on borne à top_k
    rangs = np.argsort(scores)[::-1][:top_k]

    resultats = []
    for r in rangs:
        e = index[int(r)]
        resultats.append({
            "rang":        len(resultats) + 1,
            "livre":       e["livre"],
            "livre_slug":  e["livre_slug"],
            "auteur":      e["auteur"],
            "genre":       e["genre"],
            "score":       float(scores[int(r)]),
        })
    return resultats


# ============================================================================
# Application Flask
# ============================================================================

app = Flask(__name__)
CORS(app)  # utile si Streamlit est servi sur un autre port (8501 vs 5000)


@app.route("/health", methods=["GET"])
def route_health():
    """Healthcheck pour Scaleway / Docker / load balancer."""
    return jsonify({
        "status": "ok",
        "ready":  STATE.bundle is not None,
        "n_livres": len(STATE.bundle["index"]) if STATE.bundle else 0,
        "n_resumes": len(STATE.resumes),
    })


@app.route("/api/livres", methods=["GET"])
def route_livres():
    """Renvoie l'index du corpus (sans la matrice TF-IDF, trop lourde)."""
    if STATE.bundle is None:
        return jsonify({"error": "Server not ready"}), 503

    # On expose tout sauf l'index interne (i) qui ne concerne que le serveur.
    livres = [
        {k: v for k, v in entree.items() if k != "i"}
        for entree in STATE.bundle["index"]
    ]
    return jsonify({
        "n_livres": len(livres),
        "livres":   livres,
    })


@app.route("/api/identifier", methods=["POST"])
def route_identifier():
    """
    Reçoit {"extrait": "...", "top_k": 5 (optionnel)}.
    Renvoie {"resultats": [...]} avec le top-k livres.
    """
    if STATE.bundle is None:
        return jsonify({"error": "Server not ready"}), 503

    data = request.get_json(silent=True) or {}
    extrait = data.get("extrait", "").strip()
    if not extrait:
        return jsonify({"error": "Le champ 'extrait' est vide"}), 400

    # Borne et valide top_k
    top_k = int(data.get("top_k", TOP_K_DEFAULT))
    top_k = max(1, min(top_k, TOP_K_MAX))

    t0 = time.time()
    try:
        resultats = classer_extrait(extrait, top_k=top_k)
    except Exception as e:
        log.exception(f"Erreur dans classer_extrait : {e}")
        return jsonify({"error": f"Erreur interne : {e}"}), 500
    duree_ms = int((time.time() - t0) * 1000)

    if not resultats:
        return jsonify({
            "error": "L'extrait n'a produit aucun terme exploitable "
                     "(trop court, ou vocabulaire hors corpus).",
        }), 400

    return jsonify({
        "resultats":  resultats,
        "duree_ms":   duree_ms,
        "n_termes":   "calculé côté serveur",  # peu utile à exposer
        "config":     STATE.bundle["config"],
    })


@app.route("/api/resume/<livre_slug>", methods=["GET"])
def route_resume(livre_slug):
    """Renvoie le résumé MMR pré-généré pour le livre demandé."""
    if STATE.bundle is None:
        return jsonify({"error": "Server not ready"}), 503

    contenu = STATE.resumes.get(livre_slug)
    if contenu is None:
        return jsonify({
            "error": f"Aucun résumé trouvé pour '{livre_slug}'",
            "slugs_disponibles": list(STATE.resumes.keys())[:10],
        }), 404

    return jsonify({
        "livre_slug": livre_slug,
        "resume":     contenu,
    })


@app.route("/api/extrait/<livre_slug>", methods=["GET"])
def route_extrait_aleatoire(livre_slug):
    """
    Renvoie un extrait aléatoire d'environ `taille` mots du livre demandé.

    Utilisé par la page Streamlit "Identifier" pour le bouton "Extrait
    aléatoire", qui permet de tester l'app sans copier-coller un vrai
    extrait. Le clean est lu directement depuis S3 (pas pré-chargé en
    RAM pour ne pas alourdir le conteneur — ~50 ms par appel).

    Query params :
      taille : nombre approximatif de mots (défaut 80, max 500)
    """
    if STATE.bundle is None:
        return jsonify({"error": "Server not ready"}), 503

    # Trouve l'entrée correspondante dans l'index
    entree = next(
        (e for e in STATE.bundle["index"] if e["livre_slug"] == livre_slug),
        None,
    )
    if entree is None:
        return jsonify({
            "error": f"Livre inconnu : '{livre_slug}'",
        }), 404

    try:
        taille = int(request.args.get("taille", 80))
    except (TypeError, ValueError):
        taille = 80
    taille = max(20, min(taille, 500))

    # Reconstruit la clé S3 du clean depuis les champs de l'entrée.
    # On suit la même convention que `construire_cle()` mais sans charger
    # le module : c'est juste un chemin S3 connu.
    cle_clean = (
        f"{PREFIXES['clean']}{entree['genre']}/"
        f"{entree['auteur_slug']}/{entree['livre_slug']}.txt"
    )

    try:
        texte = storage.get_text(cle_clean)
    except Exception as e:
        log.warning(f"Lecture clean échouée pour {livre_slug} : {e}")
        return jsonify({
            "error": f"Impossible de lire le clean : {e}",
        }), 500

    # Échantillonnage : on coupe en mots, on tire un offset aléatoire,
    # on prend `taille` mots consécutifs. Pas besoin de découpage en
    # phrases ici — l'extrait est un échantillon brut, pas un résumé.
    mots = texte.split()
    if len(mots) <= taille:
        extrait = " ".join(mots)
    else:
        debut = int(np.random.randint(0, len(mots) - taille))
        extrait = " ".join(mots[debut:debut + taille])

    return jsonify({
        "livre_slug": livre_slug,
        "extrait":    extrait,
        "n_mots":     len(extrait.split()),
    })


# ============================================================================
# Démarrage
# ============================================================================

# Initialisation immédiate : on veut que le serveur soit prêt dès le premier
# appel /health. Si le téléchargement S3 échoue, on laisse l'exception
# remonter pour que Docker / Scaleway redémarre le conteneur.
initialiser()


if __name__ == "__main__":
    # Mode développement local : Flask dev server.
    # En prod, lancer avec gunicorn (voir start.sh).
    app.run(host="0.0.0.0", port=5000, debug=False)