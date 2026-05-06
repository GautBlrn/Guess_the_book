"""
Résumé abstractif (genere) avec BARThez.

Modèle : `moussaKam/barthez-orangesum-abstract`. C'est BARThez (BART pre-entrainé
sur du français) fine-tune sur OrangeSum -- un dataset de paires
(article presse, résumé court). Adapte au résumé abstractif français en CPU.

Usage typique :

    from pipeline.summarization  import resumer_livre
    from pipeline.abstractive    import resumer_avec_barthez

    phrases_mmr = resumer_livre(df, k=15, ordre_narratif=True)
    resume_fluide = resumer_avec_barthez(phrases_mmr)

Stratégie : on concatène les phrases MMR (deja filtrées, déjà diversifiées)
en un texte source de quelques centaines de mots, et BARThez en produit un
résumé fluide d'environ 100-200 tokens. Inutile de re-balancer le livre
entier : MMR à déjà fait le tri.

Performance : ~30 a 60 secondes par livre sur CPU moderne (modèle de 165M
paramètres, génération par beam search avec num_beams=4).
"""
from functools import lru_cache


MODELE_DEFAUT = "moussaKam/barthez-orangesum-abstract"


@lru_cache(maxsize=1)
def _get_pipeline(modele=MODELE_DEFAUT):
    """
    Charge BARThez (lazy + cache).

    Premier appel : télécharge le modèle depuis Hugging Face Hub
    (~660 Mo). Mis en cache dans ~/.cache/huggingface/, runs suivants
    instantanés.

    On force `use_fast=True` (BarthezTokenizerFast) parce que la version
    "slow" du tokenizer BARThez est cassée dans transformers >= 4.50
    (incompatibilité avec la nouvelle API de la classe Unigram du
    package tokenizers). Le Fast tokenizer fait exactement le même
    travail et est bien plus rapide.
    """
    # Imports a la volée : transformers est lourd à importer (1-2s),
    # on ne veut pas plomber le démarrage de l'app si l'utilisateur
    # ne fait que de l'extractif.
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(modele, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(modele)
    return hf_pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=-1,    # CPU. Mettre 0 si CUDA dispo.
    )


def _construire_source(phrases_mmr, max_chars=4000):
    """
    Concatène les phrases MMR en un texte source.

    On réordonne par sent_id si ce n'est pas déjà fait pour préserver
    l'ordre narratif -- BARThez génère de meilleurs résumés quand le
    texte source est chronologique. On tronque à `max_chars` pour rester
    dans la fenêtre de contexte de BARThez (max ~1024 tokens, soit
    ~4000-5000 caractères en français).
    """
    triees = sorted(phrases_mmr, key=lambda p: p["sent_id"])
    texte = " ".join(p["texte"] for p in triees)
    if len(texte) > max_chars:
        # On coupe sur la derniere frontière de phrase avant la limite,
        # pour éviter de finir au milieu d'un mot.
        coupe = texte.rfind(". ", 0, max_chars)
        texte = texte[: coupe + 1] if coupe > 0 else texte[:max_chars]
    return texte


def resumer_avec_barthez(
    phrases_mmr, modele=MODELE_DEFAUT,
    max_length=200, min_length=80, num_beams=4,
):
    """
    Produit un résumé fluide à partir des phrases MMR.

    Paramètres
    ----------
    phrases_mmr : list[dict]
        Sortie de `summarization.resumer_livre(...)`. Doit contenir au
        minimum la clé `texte` et `sent_id` pour chaque phrase.
    max_length, min_length : int
        Bornes de longueur du résumé généré (en tokens BARThez,
        approximativement 1.3 token / mot français).
    num_beams : int
        Largeur du beam search. 4 est un bon compromis qualité/vitesse
        en CPU. Mettre 1 pour génération gloutonne (plus rapide).

    Renvoie
    -------
    str : le résumé généré.
    """
    if not phrases_mmr:
        return ""

    pipe = _get_pipeline(modele)
    source = _construire_source(phrases_mmr)

    sortie = pipe(
        source,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,
        truncation=True,
    )
    return sortie[0]["summary_text"].strip()


def precharger_modele(modele=MODELE_DEFAUT):
    """
    Force le téléchargement et le chargement du modèle en mémoire.

    A appeler explicitement (par exemple depuis un bouton de l'app) si
    on veut payer le coût du chargement maintenant plutot qu'au premier
    appel à `resumer_avec_barthez`.
    """
    _get_pipeline(modele)