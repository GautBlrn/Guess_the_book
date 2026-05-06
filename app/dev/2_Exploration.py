"""
Page Exploration : visualisations du corpus.

Trois sections :
1. Vue par livre : top lemmes, distribution POS, longueurs de phrases pour
   un livre selectionne.
2. Vue corpus : Zipf, richesse lexicale (Heaps).
3. Heatmap POS : profils grammaticaux compares entre auteurs.
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import format_livre, get_corpus


st.set_page_config(page_title="Exploration", layout="wide")
st.title("Exploration du corpus")


corpus = get_corpus()
if not corpus:
    st.error("Le corpus est vide. Va sur l'onglet Catalogue pour ajouter des livres.")
    st.stop()


# --- Choix du mode ---

mode = st.radio(
    "Vue",
    options=["Par livre", "Corpus entier", "Comparaison auteurs (POS)"],
    horizontal=True,
)


# ============================================================================
# Mode 1 : par livre
# ============================================================================

if mode == "Par livre":
    idx = st.selectbox(
        "Livre",
        options=range(len(corpus)),
        format_func=lambda i: format_livre(corpus[i]),
    )
    c = corpus[idx]
    df = c["df"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tokens", f"{len(df):,}")
    col2.metric("Phrases", f"{df['sent_id'].nunique():,}")
    col3.metric("Vocabulaire (lemmes)", f"{df['lemma'].nunique():,}")
    col4.metric(
        "Longueur moy. phrase",
        f"{len(df) / df['sent_id'].nunique():.1f} tokens",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Top 20 lemmes (hors mots-outils)")
        masque = df["is_alpha"] & ~df["is_stop"] & ~df["is_punct"]
        top = (
            df.loc[masque, "lemma"].str.lower().value_counts().head(20)
            .rename_axis("lemme").reset_index(name="frequence")
        )
        chart = alt.Chart(top).mark_bar().encode(
            x=alt.X("frequence:Q", title="Frequence"),
            y=alt.Y("lemme:N", sort="-x", title=""),
            tooltip=["lemme", "frequence"],
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.markdown("### Distribution des parties du discours")
        pos = (
            df["pos"].value_counts(normalize=True).mul(100)
            .rename_axis("POS").reset_index(name="pct")
        )
        chart = alt.Chart(pos).mark_bar().encode(
            x=alt.X("pct:Q", title="% des tokens"),
            y=alt.Y("POS:N", sort="-x", title=""),
            tooltip=["POS", alt.Tooltip("pct:Q", format=".1f")],
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Distribution des longueurs de phrases")
    longueurs = (
        df.groupby("sent_id").size()
        .rename("tokens").reset_index()
    )
    chart = alt.Chart(longueurs).mark_bar().encode(
        x=alt.X("tokens:Q", bin=alt.Bin(maxbins=50), title="Tokens par phrase"),
        y=alt.Y("count():Q", title="Nombre de phrases"),
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)


# ============================================================================
# Mode 2 : corpus entier
# ============================================================================

elif mode == "Corpus entier":
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Loi de Zipf")
        st.caption(
            "Loi empirique : la frequence du n-ieme lemme le plus frequent "
            "decroit comme 1/n. En log-log on attend une droite."
        )
        # Concatenation des lemmes filtres de tout le corpus
        tous = pd.concat([
            c["df"].loc[
                c["df"]["is_alpha"] & ~c["df"]["is_stop"] & ~c["df"]["is_punct"],
                "lemma"
            ].str.lower()
            for c in corpus
        ])
        freq = tous.value_counts()
        rangs = np.arange(1, len(freq) + 1)
        df_zipf = pd.DataFrame({
            "rang": rangs, "frequence": freq.values, "type": "observe",
        })
        df_ref = pd.DataFrame({
            "rang": rangs, "frequence": freq.iloc[0] / rangs, "type": "Zipf theorique",
        })
        df_zipf_full = pd.concat([df_zipf, df_ref])

        chart = alt.Chart(df_zipf_full).mark_line(opacity=0.7).encode(
            x=alt.X("rang:Q", scale=alt.Scale(type="log"), title="Rang du lemme"),
            y=alt.Y("frequence:Q", scale=alt.Scale(type="log"), title="Frequence"),
            color=alt.Color("type:N", title=""),
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.markdown("### Richesse lexicale")
        st.caption(
            "Loi de Heaps : la taille du vocabulaire croit comme une racine "
            "de la longueur du texte. Chaque point est un livre."
        )
        df_heaps = pd.DataFrame([{
            "auteur":   c["auteur"],
            "n_tokens": len(c["lemmes"]),
            "vocab":    len(set(c["lemmes"])),
            "genre":    c["genre"],
        } for c in corpus])

        chart = alt.Chart(df_heaps).mark_circle(size=120).encode(
            x=alt.X("n_tokens:Q",
                    scale=alt.Scale(type="log"),
                    title="Tokens (echelle log)"),
            y=alt.Y("vocab:Q",
                    scale=alt.Scale(type="log"),
                    title="Vocabulaire (echelle log)"),
            color=alt.Color("genre:N"),
            tooltip=["auteur", "genre", "n_tokens", "vocab"],
        ).properties(height=400)
        text = alt.Chart(df_heaps).mark_text(
            dx=8, dy=-8, fontSize=9, opacity=0.7,
        ).encode(
            x="n_tokens:Q", y="vocab:Q",
            text=alt.Text("auteur:N"),
        )
        st.altair_chart(chart + text, use_container_width=True)

    st.markdown("### Resume tabulaire")
    df_summary = pd.DataFrame([{
        "Auteur":      c["auteur"],
        "Livre":       c["livre"],
        "Genre":       c["genre"],
        "Tokens":      len(c["df"]),
        "Phrases":     c["df"]["sent_id"].nunique(),
        "Lemmes uniques": c["df"]["lemma"].nunique(),
        "Tokens/phrase": round(len(c["df"]) / c["df"]["sent_id"].nunique(), 1),
    } for c in corpus])
    st.dataframe(df_summary, use_container_width=True, hide_index=True)


# ============================================================================
# Mode 3 : comparaison POS auteurs
# ============================================================================

else:
    st.caption(
        "Heatmap des profils grammaticaux : pour chaque auteur, quelle "
        "fraction de ses tokens est-elle de chaque categorie ?"
    )

    POS_AFFICHES = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON",
                    "DET", "ADP", "CCONJ", "SCONJ", "PUNCT"]

    lignes = []
    for c in corpus:
        counts = c["df"]["pos"].value_counts(normalize=True) * 100
        for pos in POS_AFFICHES:
            lignes.append({
                "auteur": c["auteur"],
                "livre":  c["livre"],
                "POS":    pos,
                "pct":    float(counts.get(pos, 0.0)),
            })
    df_pos = pd.DataFrame(lignes)

    chart = alt.Chart(df_pos).mark_rect().encode(
        x=alt.X("POS:N", sort=POS_AFFICHES, title="Partie du discours"),
        y=alt.Y("auteur:N", title=""),
        color=alt.Color("pct:Q", scale=alt.Scale(scheme="yelloworangered"),
                        title="% des tokens"),
        tooltip=["auteur", "livre", "POS", alt.Tooltip("pct:Q", format=".1f")],
    ).properties(height=max(300, 30 * len(set(df_pos["auteur"]))))

    text = alt.Chart(df_pos).mark_text(fontSize=9).encode(
        x=alt.X("POS:N", sort=POS_AFFICHES),
        y="auteur:N",
        text=alt.Text("pct:Q", format=".1f"),
        color=alt.condition(
            alt.datum.pct > 18, alt.value("white"), alt.value("black"),
        ),
    )
    st.altair_chart(chart + text, use_container_width=True)
