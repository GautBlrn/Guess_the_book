"""
Chargement du corpus annote depuis S3.

Une seule fonction `charger_corpus()` retourne la liste de dicts
attendue par les notebooks 04-07 : chaque element contient les
metadonnees du livre (auteur, genre, livre, slugs) et le DataFrame
annote, plus les sequences `lemmes` et `tokens` precalculees.
"""
from pipeline import storage
from pipeline.annotation import extraire_termes
from pipeline.config import GENRE_OVERRIDES, PREFIXES
from pipeline.parsing import parser_chemin


def charger_corpus(prefixe=None, with_termes=True, verbose=True):
    """
    Charge tous les Parquet annotes du bucket en memoire.

    Renvoie une liste ordonnee (par cle S3) de dicts :
        {auteur, auteur_slug, livre, livre_slug, genre, cle, df,
         lemmes (si with_termes), tokens (si with_termes)}

    `lemmes` et `tokens` sont les sequences filtrees standard :
        is_alpha & ~is_stop & ~is_punct, lowercased.

    Le genre derive de la structure de la cle S3 (chemin annotations/genre/auteur/livre)
    peut etre surcharge via `GENRE_OVERRIDES` dans la config (cle = livre_slug).
    Sert a corriger les cas ou Gutenberg classe deux tomes du meme roman
    sous des Bookshelves differents (ex: Monte Cristo I/II).
    """
    if prefixe is None:
        prefixe = PREFIXES["annotations"]

    objets = storage.list_objects(prefixe, suffix=".parquet")
    if not objets:
        raise RuntimeError(
            f"Aucun .parquet trouve sous {prefixe}. "
            "As-tu execute le notebook 03_annotate_book ?"
        )
    if verbose:
        print(f"Chargement de {len(objets)} livres annotes...")

    corpus = []
    n_overrides = 0
    for obj in objets:
        info = parser_chemin(obj["Key"], prefixe)

        # Override de genre eventuel pour cette oeuvre
        if info["livre_slug"] in GENRE_OVERRIDES:
            ancien = info["genre"]
            info["genre"] = GENRE_OVERRIDES[info["livre_slug"]]
            if verbose and ancien != info["genre"]:
                print(f"  override genre: {info['livre_slug']} "
                      f"{ancien} -> {info['genre']}")
                n_overrides += 1

        info["df"] = storage.get_parquet(obj["Key"])
        if with_termes:
            info["lemmes"] = extraire_termes(info["df"], champ="lemma")
            info["tokens"] = extraire_termes(info["df"], champ="text")
        corpus.append(info)

    if verbose:
        if n_overrides:
            print(f"  {n_overrides} override(s) de genre appliques.")
        print(f"  {len(corpus)} livres charges")
    return corpus


def trouver_livre(corpus, indice_ou_motif):
    """
    Recherche un livre dans le corpus.

    `indice_ou_motif` peut etre :
      - un int : index dans la liste corpus
      - une chaine : motif a matcher dans `auteur`, `livre` ou `genre`
                     (insensible casse, premier match gagne)

    Plus stable que `corpus[N]` qui change quand on ajoute un livre.
    """
    if isinstance(indice_ou_motif, int):
        return corpus[indice_ou_motif]
    motif = indice_ou_motif.lower()
    for c in corpus:
        if (motif in c["auteur"].lower()
                or motif in c["livre"].lower()
                or motif in c["genre"].lower()):
            return c
    raise KeyError(f"Aucun livre ne matche {indice_ou_motif!r}")