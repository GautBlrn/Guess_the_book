"""
Page d'accueil : identifier un livre depuis un extrait.

L'utilisateur colle un passage (~50 à 200 mots) et l'app interroge le
backend Flask qui renvoie le top-5 des livres du corpus par similarité
TF-IDF cosinus.

Bonus : un bouton "extrait aléatoire d'un livre du corpus" pour tester
l'app sans copier-coller un vrai extrait. Utile pour la démo.

Tous les paramètres (config TF-IDF, métrique, mode d'annotation) sont
figés côté serveur. L'utilisateur final n'a aucun bouton à toucher
sauf l'extrait à coller et le bouton de soumission.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    api_extrait_aleatoire,
    api_get_livres,
    api_health,
    api_identifier,
    format_livre,
)


# --- Configuration de la page ---

st.set_page_config(
    page_title="Identifier un livre",
    layout="wide",
)

st.title("Identifier un livre depuis un extrait")
st.caption(
    "Colle un passage d'environ 50 à 200 mots. L'app le compare à chaque "
    "livre du corpus et renvoie les meilleurs candidats."
)


# --- Vérification de l'API ---

try:
    sante = api_health()
except RuntimeError as e:
    st.error(f"Backend indisponible : {e}")
    st.stop()

if not sante.get("ready"):
    st.warning("Le serveur n'est pas encore prêt. Réessaie dans quelques secondes.")
    st.stop()


# --- Chargement de l'index du corpus ---

try:
    livres = api_get_livres()
except RuntimeError as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

if not livres:
    st.error("Le corpus est vide.")
    st.stop()


# --- Zone de saisie : 2 colonnes ---

col_in, col_demo = st.columns([3, 1])

with col_in:
    extrait = st.text_area(
        "Extrait",
        height=200,
        placeholder=(
            "Colle ici un passage que tu veux identifier. "
            "Idéalement 50 à 200 mots."
        ),
        key="extrait_text",
    )

with col_demo:
    st.markdown("**Extrait aléatoire**")
    st.caption("Tirer un passage d'un livre du corpus pour tester.")
    livre_demo_idx = st.selectbox(
        "Source",
        options=range(len(livres)),
        format_func=lambda i: format_livre(livres[i]),
        key="demo_source",
        label_visibility="collapsed",
    )
    taille_demo = st.slider(
        "Taille (mots)", 30, 300, 80, step=10,
    )

    # Callback pour tirer un extrait : on l'utilise en `on_click` plutôt
    # qu'un if-button, sinon Streamlit interdit de modifier le state du
    # text_area déjà instancié plus haut dans le rerun courant.
    def _tirer_extrait(livre_idx, taille):
        slug = livres[livre_idx]["livre_slug"]
        try:
            extrait_genere = api_extrait_aleatoire(slug, taille=taille)
        except RuntimeError as e:
            st.session_state["erreur_demo"] = str(e)
            return
        st.session_state["extrait_text"] = extrait_genere
        # On lie l'attendu à l'extrait : si l'utilisateur édite le texte,
        # le lien est rompu via la comparaison `extrait_attendu == extrait`.
        st.session_state["livre_attendu_idx"] = livre_idx
        st.session_state["extrait_attendu"] = extrait_genere
        st.session_state.pop("erreur_demo", None)

    st.button(
        "Tirer un extrait",
        use_container_width=True,
        on_click=_tirer_extrait,
        args=(livre_demo_idx, taille_demo),
    )

if "erreur_demo" in st.session_state:
    st.error(st.session_state["erreur_demo"])


# --- Garde fou ---

if not extrait or not extrait.strip():
    st.info("Saisis ou tire un extrait pour lancer l'identification.")
    st.stop()


# Statistiques rapides sur l'extrait
n_mots_brut = len(extrait.split())
st.caption(
    f"Extrait : {len(extrait):,} caractères -- {n_mots_brut} mots (split simple)"
)


# --- Appel API ---

with st.spinner("Identification en cours..."):
    try:
        reponse = api_identifier(extrait, top_k=10)
    except RuntimeError as e:
        st.error(f"Erreur : {e}")
        st.stop()

resultats = reponse["resultats"]
duree_ms = reponse["duree_ms"]


# --- Affichage des résultats ---

# Mode démo : on connaît le bon livre si l'extrait n'a pas été modifié
livre_attendu_idx = None
if (st.session_state.get("livre_attendu_idx") is not None
        and st.session_state.get("extrait_attendu") == extrait):
    livre_attendu_idx = st.session_state["livre_attendu_idx"]

# Bandeau de résultat principal
top1 = resultats[0]
top1_label = f"{top1['auteur']} -- {top1['livre']} ({top1['genre']})"

if livre_attendu_idx is not None:
    livre_attendu = livres[livre_attendu_idx]
    rang_attendu = next(
        (r["rang"] for r in resultats
         if r["livre_slug"] == livre_attendu["livre_slug"]),
        None,
    )
    if rang_attendu == 1:
        st.success(
            f"**Identification correcte** -- top 1 : {top1_label} "
            f"(score = {top1['score']:.3f})"
        )
    elif rang_attendu is not None:
        score_attendu = next(
            r["score"] for r in resultats
            if r["livre_slug"] == livre_attendu["livre_slug"]
        )
        st.warning(
            f"**Top 1 :** {top1_label} (score = {top1['score']:.3f}) -- "
            f"le vrai livre ({format_livre(livre_attendu)}) est au "
            f"**rang {rang_attendu}** avec un score de {score_attendu:.3f}."
        )
    else:
        st.warning(
            f"**Top 1 :** {top1_label} (score = {top1['score']:.3f}) -- "
            f"le vrai livre ({format_livre(livre_attendu)}) n'est pas dans "
            "le top 10 retourné."
        )
else:
    st.success(
        f"**Top 1 :** {top1_label} (score = {top1['score']:.3f})"
    )

st.caption(f"Calcul effectué en {duree_ms} ms.")


# Tableau du top 10 avec barres de progression
st.markdown("### Classement complet")
df_resultats = pd.DataFrame([
    {
        "Rang":   r["rang"],
        "Auteur": r["auteur"],
        "Livre":  r["livre"],
        "Genre":  r["genre"],
        "Score":  r["score"],
    }
    for r in resultats
])

score_max = max(1.0, max(r["score"] for r in resultats))
st.dataframe(
    df_resultats,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Similarité cosinus",
            min_value=0.0,
            max_value=score_max,
            format="%.3f",
        ),
    },
)


# --- Diagnostics ---

with st.expander("Configuration utilisée"):
    config = reponse.get("config", {})
    st.markdown(
        f"- **Champ TF-IDF :** {config.get('champ', 'lemmes')}\n"
        f"- **n-grammes max :** {config.get('ngram_max', 1)}\n"
        f"- **Métrique :** {config.get('metrique', 'cosinus')}\n"
        f"- **min_df :** {config.get('min_df', 2)}\n"
        f"- **max_df_ratio :** {config.get('max_df_ratio', 0.85)}"
    )
    st.caption(
        "Ces paramètres sont figés côté serveur (issus du benchmark : "
        "lemmes_1g + cosinus = 99.2 % top-1 sur le corpus actuel)."
    )