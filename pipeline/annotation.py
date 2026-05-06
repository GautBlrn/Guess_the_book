"""
Annotation linguistique : POS, lemmes, frontieres de phrases.

Modele : `fr_core_news_sm` (12 Mo, rapide, suffisant pour le francais
standard). NER desactivee : entrainee sur du news moderne, peu fiable
sur du XIXe siecle. Tagger + parser + lemmatizer sont actifs.

`annoter(texte)` decoupe en chunks (frontieres sur double saut de
ligne pour ne pas couper en plein milieu d'une phrase), envoie via
`nlp.pipe()` pour le traitement en batch, et concatene les tokens
en un DataFrame unique. Les `sent_id` et `token_id` sont continus
d'un chunk au suivant.
"""
from functools import lru_cache

import pandas as pd
import spacy

from pipeline.config import ANNOTATE_PARAMS
from pipeline.tokenization import normaliser


@lru_cache(maxsize=1)
def _get_nlp():
    """Pipeline spaCy complet (lazy load, mis en cache)."""
    return spacy.load(
        ANNOTATE_PARAMS["model"],
        disable=ANNOTATE_PARAMS["disable"],
    )


def decouper_en_chunks(texte, taille_max=None):
    """
    Decoupe le texte en chunks d'au plus `taille_max` caracteres,
    en cherchant une frontiere sur double saut de ligne pour eviter
    de couper en plein milieu d'une phrase.
    """
    if taille_max is None:
        taille_max = ANNOTATE_PARAMS["chunk_size"]
    chunks = []
    i = 0
    n = len(texte)
    while i < n:
        if n - i <= taille_max:
            chunks.append(texte[i:])
            break
        cible = i + taille_max
        coupe = texte.rfind("\n\n", i + taille_max // 2, cible)
        if coupe == -1:
            coupe = cible
        chunks.append(texte[i:coupe])
        i = coupe
    return chunks


def annoter(texte, batch_size=None, chunk_size=None):
    """
    Annote un texte complet et renvoie un DataFrame avec une ligne
    par token (colonnes : sent_id, token_id, text, lemma, pos, tag,
    is_punct, is_stop, is_alpha).
    """
    if batch_size is None:
        batch_size = ANNOTATE_PARAMS["batch_size"]
    if chunk_size is None:
        chunk_size = ANNOTATE_PARAMS["chunk_size"]

    nlp = _get_nlp()
    texte = normaliser(texte)
    chunks = decouper_en_chunks(texte, taille_max=chunk_size)

    lignes = []
    sent_offset = 0
    token_offset = 0

    for doc in nlp.pipe(chunks, batch_size=batch_size):
        sent_id_local = {sent.start: i for i, sent in enumerate(doc.sents)}
        sent_id_courant = 0

        for tok in doc:
            if tok.is_space:
                continue
            if tok.i in sent_id_local:
                sent_id_courant = sent_id_local[tok.i]
            lignes.append({
                "sent_id":  sent_offset + sent_id_courant,
                "token_id": token_offset,
                "text":     tok.text,
                "lemma":    tok.lemma_ if tok.lemma_ else tok.text,
                "pos":      tok.pos_,
                "tag":      tok.tag_,
                "is_punct": tok.is_punct,
                "is_stop":  tok.is_stop,
                "is_alpha": tok.is_alpha,
            })
            token_offset += 1

        sent_offset += len(sent_id_local)

    df = pd.DataFrame(lignes)
    # Types compacts pour reduire la taille du Parquet
    df["sent_id"]  = df["sent_id"].astype("int32")
    df["token_id"] = df["token_id"].astype("int32")
    df["pos"]      = df["pos"].astype("category")
    df["tag"]      = df["tag"].astype("category")
    return df


def extraire_termes(df, champ="lemma", lower=True):
    """
    Extrait la sequence de termes filtres d'un livre annote.

    Filtre standard : `is_alpha & ~is_stop & ~is_punct`. Convertit
    en minuscules par defaut.
    """
    masque = df["is_alpha"] & ~df["is_stop"] & ~df["is_punct"]
    serie = df.loc[masque, champ]
    if lower:
        serie = serie.str.lower()
    return serie.tolist()
