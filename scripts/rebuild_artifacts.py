"""
Régénère les artéfacts globaux du corpus après ajout/suppression d'un livre.

Etages produits :

  artifacts/
    tfidf_<config>.pkl          - matrices TF-IDF + vectoriseur (3 configs)
    w2v_corpus.kv               - embeddings Word2Vec entraines sur le corpus
    corpus_index.json           - index ordonne des livres (auteur, genre, livre, cle)

  summaries/
    <genre>/<auteur>/<livre>.txt  - résumés MMR (k phrases, lambda par defaut)

Usage :

    python -m scripts.rebuild_artifacts
    python -m scripts.rebuild_artifacts --skip-w2v       # plus rapide
    python -m scripts.rebuild_artifacts --skip-summaries # juste TF-IDF + Word2Vec
"""
import argparse
import io
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import storage
from pipeline.config import MMR_PARAMS, PREFIXES, TFIDF_PARAMS, W2V_PARAMS
from pipeline.corpus import charger_corpus
from pipeline.parsing import construire_cle
from pipeline.representations import TfIdfMaison
from pipeline.summarization import formatter_resume, resumer_livre


# --- CLI ---

def parser_args():
    p = argparse.ArgumentParser(description="Régenère les artefacts globaux.")
    p.add_argument("--skip-tfidf",     action="store_true")
    p.add_argument("--skip-w2v",       action="store_true")
    p.add_argument("--skip-summaries", action="store_true")
    return p.parse_args()


# --- TF-IDF ---

CONFIGS_TFIDF = {
    "lemmes_1g":  {"champ": "lemmes", "ngram_max": 1},
    "lemmes_12g": {"champ": "lemmes", "ngram_max": 2},
    "tokens_12g": {"champ": "tokens", "ngram_max": 2},
}


def construire_tfidf(corpus, params=None):
    """Construit les 3 matrices TF-IDF et les uploade dans artifacts/."""
    if params is None:
        params = TFIDF_PARAMS

    for nom, cfg in CONFIGS_TFIDF.items():
        documents = [c[cfg["champ"]] for c in corpus]
        vec = TfIdfMaison(
            ngram_max=cfg["ngram_max"],
            min_df=params["min_df"],
            max_df_ratio=params["max_df_ratio"],
        )
        X = vec.fit_transform(documents)
        print(f"  {nom:12s} : {X.shape}, vocabulaire {len(vec.vocabulaire_):,} termes")

        # Pickle (vectoriseur + matrice)
        artefact = {"vectoriseur": vec, "matrice": X, "config": cfg}
        cle = f"{PREFIXES['artifacts']}tfidf_{nom}.pkl"
        storage.put_bytes(
            cle, pickle.dumps(artefact),
            content_type="application/octet-stream",
            metadata={"config": nom, "vocab_size": str(len(vec.vocabulaire_))},
        )


# --- Word2Vec ---

def construire_phrases_w2v(corpus):
    """Découpe le corpus en phrases pour Word2Vec (frontières = sent_id)."""
    import pandas as pd
    phrases = []
    for c in corpus:
        df = c["df"]
        masque = df["is_alpha"] & ~df["is_stop"] & ~df["is_punct"]
        sub = df.loc[masque, ["sent_id", "lemma"]].copy()
        sub["lemma"] = sub["lemma"].str.lower()
        for _, groupe in sub.groupby("sent_id"):
            if len(groupe) >= 3:
                phrases.append(groupe["lemma"].tolist())
    return phrases


def construire_w2v(corpus, params=None):
    """Entraine Word2Vec sur le corpus et uploade les vecteurs."""
    if params is None:
        params = W2V_PARAMS
    from gensim.models import Word2Vec

    phrases = construire_phrases_w2v(corpus)
    print(f"  {len(phrases):,} phrases pour entrainement")

    t0 = time.time()
    w2v = Word2Vec(
        sentences=phrases,
        vector_size=params["vector_size"],
        window=params["window"],
        min_count=params["min_count"],
        sg=params["sg"],
        workers=params["workers"],
        epochs=params["epochs"],
        seed=params["seed"],
    )
    print(f"  entraine en {time.time() - t0:.1f}s "
          f"(vocab {len(w2v.wv):,}, dim {w2v.wv.vector_size})")

    # On sauve les vecteurs en .kv (KeyedVectors)
    buf = io.BytesIO()
    pickle.dump({"wv": w2v.wv, "params": params}, buf)
    cle = f"{PREFIXES['artifacts']}w2v_corpus.pkl"
    storage.put_bytes(
        cle, buf.getvalue(),
        metadata={"vocab_size": str(len(w2v.wv)),
                  "dim":        str(w2v.wv.vector_size)},
    )


# --- Index du corpus ---

def construire_index(corpus):
    """Index ordonne du corpus, stocke comme JSON. Sert de référence partagée."""
    index = [{
        "auteur":       c["auteur"],
        "auteur_slug":  c["auteur_slug"],
        "livre":        c["livre"],
        "livre_slug":   c["livre_slug"],
        "genre":        c["genre"],
        "cle":          c["cle"],
        "n_tokens":     int(len(c["df"])),
        "n_phrases":    int(c["df"]["sent_id"].nunique()),
    } for c in corpus]
    cle = f"{PREFIXES['artifacts']}corpus_index.json"
    storage.put_json(cle, index, metadata={"n_livres": str(len(index))})
    print(f"  index uploade : {cle} ({len(index)} livres)")


# --- Résumés MMR ---

def construire_resumes(corpus):
    """Génère et uploade les résumés MMR pour tous les livres."""
    for c in corpus:
        phrases = resumer_livre(
            c["df"],
            champ_termes="lemma",
            k=MMR_PARAMS["k_phrases"],
            lambda_=MMR_PARAMS["lambda_default"],
        )
        resume = formatter_resume(
            phrases,
            livre=c["livre"], auteur=c["auteur"], genre=c["genre"],
            k=MMR_PARAMS["k_phrases"], lambda_=MMR_PARAMS["lambda_default"],
        )
        cle = construire_cle(
            PREFIXES["summaries"], c["genre"], c["auteur_slug"], c["livre_slug"], ".txt"
        )
        storage.put_text(
            cle, resume,
            metadata={"source": "summary_mmr",
                      "k":       str(MMR_PARAMS["k_phrases"]),
                      "lambda":  str(MMR_PARAMS["lambda_default"])},
        )
        print(f"  {c['auteur']:35s} -> {cle}")


# --- Main ---

def main():
    args = parser_args()
    storage.ensure_bucket()
    corpus = charger_corpus()

    if not args.skip_tfidf:
        print("--- TF-IDF ---")
        construire_tfidf(corpus)
        print()

    if not args.skip_w2v:
        print("--- Word2Vec ---")
        construire_w2v(corpus)
        print()

    if not args.skip_summaries:
        print("--- Resumes MMR ---")
        construire_resumes(corpus)
        print()

    print("--- Index ---")
    construire_index(corpus)
    print("\nTermine.")


if __name__ == "__main__":
    main()
