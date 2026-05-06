"""
Orchestrateur du pipeline complet pour UN livre.

Donne un livre via son contenu brut + metadonnees (auteur slug,
genre, nom de fichier), et il passe par les 4 etages :

    raw -> clean -> tokens -> annotations

en uploadant a chaque etage. C'est exactement ce que font les
notebooks 01-03 dans une boucle, factorise pour pouvoir l'appeler
ailleurs (ex. depuis `add_book.py`).

`pipeline_un_livre()` renvoie un dict avec les cles S3 produites et
quelques stats (nb tokens, nb phrases). En cas d'erreur a une etape
on logge et on arrete : pas de retry, pas de skip silencieux.
"""
import time

from pipeline import storage
from pipeline.annotation import annoter
from pipeline.cleaning import nettoyer
from pipeline.config import PREFIXES
from pipeline.parsing import construire_cle
from pipeline.tokenization import tokeniser


def pipeline_un_livre(contenu_raw, auteur_slug, genre, livre_slug,
                      verbose=True):
    """
    Passe un livre par toutes les etapes et upload vers S3.

    Parametres
    ----------
    contenu_raw : str
        Texte brut du livre (sortie de telecharger_gutenberg).
    auteur_slug : str
        Slug auteur, par exemple 'zola_emile'.
    genre : str
        Genre interne, par exemple 'novel'.
    livre_slug : str
        Slug du livre, par exemple 'pg8560_le_docteur_pascal' (sans extension).
    verbose : bool
        Si True, log la progression de chaque etape.

    Renvoie
    -------
    dict avec les cles S3 ecrites et des stats.
    """
    log = print if verbose else (lambda *a, **k: None)
    metadata_base = {"auteur": auteur_slug, "genre": genre, "source": "gutenberg"}
    resultats = {"auteur": auteur_slug, "genre": genre, "livre": livre_slug}

    # --- 1. raw ---
    cle_raw = construire_cle(PREFIXES["raw"], genre, auteur_slug, livre_slug, ".txt")
    storage.put_text(cle_raw, contenu_raw, metadata=metadata_base)
    resultats["cle_raw"] = cle_raw
    log(f"  raw         -> {cle_raw} ({len(contenu_raw):,} car.)")

    # --- 2. clean ---
    contenu_clean = nettoyer(contenu_raw)
    cle_clean = construire_cle(PREFIXES["clean"], genre, auteur_slug, livre_slug, ".txt")
    storage.put_text(
        cle_clean, contenu_clean,
        metadata={**metadata_base, "original_key": cle_raw},
    )
    resultats["cle_clean"] = cle_clean
    log(f"  clean       -> {cle_clean} ({len(contenu_clean):,} car.)")

    # --- 3. tokens ---
    tokens = tokeniser(contenu_clean)
    cle_tokens = construire_cle(PREFIXES["tokens"], genre, auteur_slug, livre_slug, ".json")
    storage.put_json(
        cle_tokens, tokens,
        metadata={**metadata_base, "original_key": cle_clean,
                  "n_tokens": str(len(tokens))},
    )
    resultats["cle_tokens"] = cle_tokens
    resultats["n_tokens"] = len(tokens)
    log(f"  tokens      -> {cle_tokens} ({len(tokens):,} tokens)")

    # --- 4. annotations ---
    t0 = time.time()
    df = annoter(contenu_clean)
    duree = time.time() - t0
    cle_annot = construire_cle(
        PREFIXES["annotations"], genre, auteur_slug, livre_slug, ".parquet"
    )
    storage.put_parquet(
        cle_annot, df,
        metadata={**metadata_base, "original_key": cle_clean,
                  "n_tokens": str(len(df)),
                  "n_phrases": str(df["sent_id"].nunique())},
    )
    resultats["cle_annotations"] = cle_annot
    resultats["n_tokens_annotes"] = len(df)
    resultats["n_phrases"] = int(df["sent_id"].nunique())
    log(f"  annotations -> {cle_annot} ({len(df):,} tokens, "
        f"{df['sent_id'].nunique():,} phrases, {duree:.1f}s)")

    return resultats
