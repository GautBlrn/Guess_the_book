"""
Acces a Project Gutenberg.

Deux entrees :
- `chercher(...)` : recherche dans le catalogue local pg_catalog.csv par
  titre et/ou auteur. Renvoie les candidats avec leur score de match.
- `telecharger(text_id)` : telecharge le texte brut d'un livre a partir
  de son identifiant Gutenberg (Text#).
"""
from functools import lru_cache

import pandas as pd
import requests

from pipeline.config import (
    CATALOG_PATH, DEFAULT_LANGUAGE, GUTENBERG_DOWNLOAD_URLS,
)
from pipeline.parsing import auteur_principal, inferer_genre


@lru_cache(maxsize=1)
def _charger_catalogue(path=str(CATALOG_PATH)):
    """Charge et enrichit le catalogue Gutenberg (cache memoire)."""
    df = pd.read_csv(path, dtype=str)
    df = df[df["Type"] == "Text"].copy()
    df["genre"] = df["Bookshelves"].apply(inferer_genre)
    df["auteur_clean"] = df["Authors"].apply(auteur_principal)
    return df


def chercher(titre=None, auteur=None, langue=DEFAULT_LANGUAGE, n_max=10):
    """
    Recherche un livre dans le catalogue Gutenberg.

    Au moins un des deux parmi `titre` et `auteur` doit etre fourni. La
    recherche est insensible a la casse et utilise un simple `contains`
    sur les colonnes Title et auteur_clean.

    Renvoie un DataFrame trie par pertinence (un titre + auteur qui
    matchent tous les deux passe avant un seul match).
    """
    if not titre and not auteur:
        raise ValueError("Donne au moins un titre ou un auteur.")

    df = _charger_catalogue()

    # Filtre langue si specifie
    if langue:
        df = df[df["Language"].fillna("").str.split("; ").apply(
            lambda l: langue in l
        )]

    # On ne garde que les livres avec metadonnees exploitables
    df = df[df["Title"].notna() & df["auteur_clean"].notna()].copy()

    score_titre = (
        df["Title"].str.contains(titre, case=False, na=False).astype(int)
        if titre else 0
    )
    score_auteur = (
        df["auteur_clean"].str.contains(auteur, case=False, na=False).astype(int)
        if auteur else 0
    )
    df["score"] = score_titre + score_auteur

    # On exige au moins un match
    df = df[df["score"] > 0].sort_values(
        ["score", "Title"], ascending=[False, True]
    )
    return df.head(n_max)[
        ["Text#", "Title", "auteur_clean", "genre", "Bookshelves", "score"]
    ].reset_index(drop=True)


def telecharger(text_id, timeout=30, taille_min=1000):
    """
    Telecharge le texte d'un livre Gutenberg a partir de son Text#.

    Essaie plusieurs URLs (cache, files-0, files-8) et renvoie le contenu
    decode (UTF-8, fallback latin-1) ou None si tout echoue.
    """
    text_id = str(text_id)
    for modele in GUTENBERG_DOWNLOAD_URLS:
        url = modele.format(tid=text_id)
        try:
            r = requests.get(url, timeout=timeout)
        except requests.RequestException:
            continue
        if r.status_code == 200 and len(r.content) > taille_min:
            try:
                return r.content.decode("utf-8")
            except UnicodeDecodeError:
                return r.content.decode("latin-1")
    return None
