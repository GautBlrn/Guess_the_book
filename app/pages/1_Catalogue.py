"""
Page Catalogue : affiche les livres du corpus.

Version déployée : un seul onglet "Corpus actuel". L'onglet "Ajouter un
livre" de la version locale a été retiré -- ajouter un livre déclenche
le pipeline complet (clean → tokens → annotations spaCy + écritures
S3), opération admin qui n'a pas sa place dans une app publique.

Pour ajouter un livre, on garde la version locale du fichier (avant
client HTTP) qui appelait `pipeline_un_livre()` directement.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import api_get_livres, api_health, reset_caches


st.set_page_config(page_title="Catalogue", layout="wide")
st.title("Catalogue du corpus")


# --- Vérification de l'API ---

try:
    sante = api_health()
except RuntimeError as e:
    st.error(f"Backend indisponible : {e}")
    st.stop()

if not sante.get("ready"):
    st.warning("Le serveur n'est pas encore prêt. Réessaie dans quelques secondes.")
    st.stop()


# --- Bouton de rafraîchissement ---

col_actions, _ = st.columns([1, 5])
with col_actions:
    if st.button("Rafraîchir", help="Recharge la liste depuis l'API"):
        reset_caches()
        st.rerun()


# --- Chargement et affichage ---

try:
    livres = api_get_livres()
except RuntimeError as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

if not livres:
    st.warning("Le corpus est vide.")
    st.stop()

df_corpus = pd.DataFrame([
    {
        "Auteur":   livre["auteur"],
        "Livre":    livre["livre"],
        "Genre":    livre["genre"],
        "Tokens":   livre["n_tokens"],
        "Phrases":  livre["n_phrases"],
    }
    for livre in livres
])

# Résumé haut de page
col1, col2, col3, col4 = st.columns(4)
col1.metric("Livres", len(livres))
col2.metric("Auteurs uniques", df_corpus["Auteur"].nunique())
col3.metric("Genres uniques", df_corpus["Genre"].nunique())
col4.metric("Tokens (total)", f"{df_corpus['Tokens'].sum():,}")

st.markdown("### Détail")
st.dataframe(
    df_corpus.sort_values("Auteur"),
    use_container_width=True,
    hide_index=True,
)

# Répartition par genre
st.markdown("### Répartition par genre")
st.bar_chart(
    df_corpus.groupby("Genre").size().rename("Nombre de livres"),
    use_container_width=True,
)