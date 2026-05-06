"""
Page Résumés : affiche le résumé MMR pré-généré pour le livre choisi.

Version déployée : aucun calcul à la volée. Les résumés sont produits
en amont par `scripts/rebuild_artifacts.py` qui les stocke sur S3, et
le serveur Flask les pré-charge tous en RAM au démarrage. Du coup
l'affichage est instantané.

Différences avec la version locale (avant le déploiement) :
- Plus de sliders MMR (k, lambda) ni de filtres -- la config est figée
  par EMB_PARAMS / MMR_PARAMS au moment du rebuild.
- Plus d'onglet "Effet de lambda" (calcul à la volée trop lourd pour
  une instance Scaleway sans GPU).
- Plus d'onglet "Abstractif (BARThez)" (660 Mo + 30-60s par livre,
  incompatible avec la promesse "l'utilisateur n'attend pas").
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import api_get_livres, api_get_resume, api_health, format_livre


st.set_page_config(page_title="Résumés", layout="wide")
st.title("Résumés extractifs")
st.caption(
    "Résumés générés en amont par MMR sur embeddings sentence-camembert "
    "avec score hybride (centre TF-IDF + TextRank + bonus longueur). "
    "Configuration figée pour cette app publique."
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


# --- Chargement de l'index ---

try:
    livres = api_get_livres()
except RuntimeError as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

if not livres:
    st.error("Le corpus est vide.")
    st.stop()


# --- Sélection du livre ---

idx = st.selectbox(
    "Livre",
    options=range(len(livres)),
    format_func=lambda i: format_livre(livres[i]),
)
livre = livres[idx]


# --- Affichage des métadonnées ---

col1, col2, col3 = st.columns(3)
col1.metric("Tokens", f"{livre['n_tokens']:,}")
col2.metric("Phrases", f"{livre['n_phrases']:,}")
col3.metric("Genre", livre["genre"])


# --- Récupération et affichage du résumé ---

with st.spinner("Chargement du résumé..."):
    try:
        resume_texte = api_get_resume(livre["livre_slug"])
    except RuntimeError as e:
        st.error(f"Erreur : {e}")
        st.stop()

st.markdown(f"### Résumé de _{livre['livre']}_")
# Le résumé contient déjà un en-tête (livre, auteur, genre, k, lambda).
# On l'affiche tel quel dans un bloc de texte préformaté pour respecter
# la mise en page d'origine.
st.text(resume_texte)


# --- Téléchargement ---

st.download_button(
    "Télécharger en .txt",
    data=resume_texte.encode("utf-8"),
    file_name=f"resume_{livre['livre_slug']}.txt",
    mime="text/plain",
)