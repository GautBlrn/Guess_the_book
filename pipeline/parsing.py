"""
Parsing et normalisation de chaines (auteurs, cles S3, chemins).

Tout ce qui transforme du texte brut metadonnee (auteur Gutenberg avec
dates, titre avec accents, etc.) vers la forme normalisee qu'on stocke
dans S3, et vice-versa.
"""
import re
import unicodedata

import pandas as pd

from pipeline.config import GENRE_MAPPING


# --- Slugification ---

# Ligatures que NFKD ne decompose PAS, et qu'il faut donc traiter avant.
#
# Unicode considere « œ » et « æ » comme des lettres a part entiere de
# l'orthographe francaise, pas comme des ligatures typographiques a la
# maniere de « ﬁ ». NFKD les laisse donc intactes, et le passage en ASCII
# qui suit les supprime purement et simplement : « Œuvres completes »
# devenait « uvres_completes ». Le cas n'a rien de marginal sur un fonds
# francais, ou les « Œuvres completes de X » se comptent par dizaines.
LIGATURES = str.maketrans({
    "œ": "oe", "Œ": "OE",
    "æ": "ae", "Æ": "AE",
})


def slugifier(txt):
    """Rend une chaine safe pour une cle S3 : pas d'accent, pas d'espace."""
    txt = txt.translate(LIGATURES)
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^\w\s-]", "", txt).strip().lower()
    return re.sub(r"[\s_-]+", "_", txt)


def deslug(slug):
    """Inverse approximatif : transforme un slug en libelle lisible.

    `zola_emile` -> `Zola Emile`, `pg8560_le_docteur_pascal` -> `Le Docteur Pascal`
    (apres retrait du prefixe `pgNNNN_`).
    """
    slug = re.sub(r"^pg\d+_", "", slug)
    return " ".join(p.capitalize() for p in slug.split("_"))


# --- Parsing du catalogue Gutenberg ---

def auteur_principal(authors):
    """Extrait le premier auteur, nettoie dates et roles entre crochets."""
    if pd.isna(authors):
        return None
    premier = authors.split(";")[0].strip()
    premier = re.sub(r",\s*\d{4}\??-\d{0,4}\??", "", premier)
    premier = re.sub(r"\s*\[.*?\]", "", premier)
    return premier.strip() or None


def inferer_genre(bookshelves):
    """Infere un genre interne a partir du champ Bookshelves de Gutenberg."""
    if pd.isna(bookshelves):
        return None
    texte = str(bookshelves).lower()
    for cle, genre in GENRE_MAPPING.items():
        if cle in texte:
            return genre
    return None


# --- Parsing de cles S3 ---

def parser_chemin(cle, prefix):
    """
    Decompose une cle S3 en (genre, auteur, livre).

    Exemple : 'annotations/novel/zola_emile/pg8560_le_docteur_pascal.parquet'
              -> {genre: 'novel', auteur_slug: 'zola_emile',
                  livre_slug: 'pg8560_le_docteur_pascal',
                  auteur: 'Zola Emile', livre: 'Le Docteur Pascal'}
    """
    if cle.startswith(prefix):
        cle_relative = cle[len(prefix):]
    else:
        cle_relative = cle
    parts = cle_relative.split("/")
    if len(parts) < 3:
        raise ValueError(f"Cle inattendue : {cle!r} (3 segments attendus)")
    genre = parts[0]
    auteur_slug = parts[1]
    # On retire l'extension du dernier segment
    livre_slug = re.sub(r"\.[^.]+$", "", parts[2])
    return {
        "genre":       genre,
        "auteur_slug": auteur_slug,
        "livre_slug":  livre_slug,
        "auteur":      deslug(auteur_slug),
        "livre":       deslug(livre_slug),
        "cle":         cle,
    }


def construire_cle(prefix, genre, auteur_slug, livre_slug, extension):
    """Construit une cle S3 coherente avec la convention prefix/genre/auteur/livre.ext"""
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{prefix}{genre}/{auteur_slug}/{livre_slug}{ext}"
