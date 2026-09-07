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

Les resumes orphelins (restes d'un changement de genre : la cle S3 encode
le genre, donc un override deplace le resume sans effacer l'ancien) sont
listes a chaque run et supprimes seulement sur demande :

    python -m scripts.rebuild_artifacts --nettoyer-orphelins
"""
import argparse
import io
import pickle
import sys
import time
from pathlib import Path

# --- RACINE_DEPOT ---
# Ce fichier est livre a deux endroits, et un nombre fixe de `.parent` ne peut
# pas convenir aux deux :
#
#     dans le depot     projet_nlp_app/scripts/rebuild_artifacts.py
#     dans la remise    Projet_2_.../Modeles/A_Gautier_Blairon/code_entrainement/
#                       ou `parent.parent` vaut `Modeles/`, qui ne porte
#                       aucun paquet `pipeline/`
#
# La racine cherchee est celle qui porte `pipeline/config.py`. On la trouve par
# ce repere plutot que par un comptage de niveaux, et le repli est explicite.
# Le bloc est recopie dans `build_serving_bundle.py` plutot que mis en commun :
# il s'execute avant que `pipeline` soit importable, donc il ne peut pas en
# venir.
REPERE_RACINE = Path("pipeline") / "config.py"
DISPOSITIONS = ("", "Applications/A_Gautier_Blairon")


def _racine_depot() -> Path:
    """Racine d'ou le paquet `pipeline` et les artefacts se resolvent."""
    ici = Path(__file__).resolve()
    for ancetre in ici.parents:
        for disposition in DISPOSITIONS:
            candidat = ancetre / disposition if disposition else ancetre
            if (candidat / REPERE_RACINE).is_file():
                return candidat
    raise SystemExit(
        f"racine du depot introuvable depuis {ici}\n"
        f"  repere cherche : {REPERE_RACINE}\n"
        f"  dispositions essayees : {DISPOSITIONS}\n"
        "Lancer le script depuis le depot ou depuis l'archive decompressee."
    )


RACINE = _racine_depot()
sys.path.insert(0, str(RACINE))

# --- CLI ---

def parser_args():
    p = argparse.ArgumentParser(description="Régenère les artefacts globaux.")
    p.add_argument("--skip-tfidf",     action="store_true")
    p.add_argument("--skip-w2v",       action="store_true")
    p.add_argument("--skip-summaries", action="store_true")
    p.add_argument("--nettoyer-orphelins", action="store_true",
                   help="Supprime les resumes S3 qui ne correspondent a aucun "
                        "livre du corpus (restes d'un changement de genre). "
                        "Sans ce flag, ils sont seulement listes.")
    return p.parse_args()

# --- `--help` sans la pile lourde ---
# Les imports ci-dessous tirent boto3, botocore, pandas et gensim. Le
# LISEZ-MOI de la remise promet que chaque script de `Modeles/` se verifie en
# une commande qui « ne calcule rien et ne demande ni GPU ni reseau » : un
# evaluateur qui lance `--help` dans un Python nu ne doit donc pas tomber sur
# un ModuleNotFoundError avant d'avoir lu le mode d'emploi. Meme motif que
# l'import tardif d'`ultralytics` dans le `train.py` du projet 1.
if any(a in ("-h", "--help") for a in sys.argv[1:]):
    parser_args()          # argparse imprime l'aide puis leve SystemExit(0)
    raise SystemExit(0)    # filet, si jamais argparse rendait la main


from pipeline import storage
from pipeline.config import (
    MMR_PARAMS, PREFIXES, TFIDF_CONFIGS, TFIDF_PARAMS, W2V_PARAMS,
)
from pipeline.corpus import charger_corpus, iter_corpus
from pipeline.parsing import construire_cle
from pipeline.representations import TfIdfMaison
from pipeline.summarization import formatter_resume, resumer_livre


# --- TF-IDF ---

# Les configs vivent dans pipeline/config.py : pipeline/benchmark.py evalue
# la meme liste, et une seule source evite que le rapport decrive des
# representations qui ne sont pas celles qu'on sert.
CONFIGS_TFIDF = TFIDF_CONFIGS


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

def phrases_w2v_livre(df):
    """Découpe UN livre annoté en phrases pour Word2Vec (frontières = sent_id)."""
    masque = df["is_alpha"] & ~df["is_stop"] & ~df["is_punct"]
    sub = df.loc[masque, ["sent_id", "lemma"]].copy()
    sub["lemma"] = sub["lemma"].str.lower()
    return [
        groupe["lemma"].tolist()
        for _, groupe in sub.groupby("sent_id")
        if len(groupe) >= 3
    ]


def construire_phrases_w2v(corpus):
    """
    Découpe un corpus déjà chargé en phrases. Conservé pour les notebooks,
    qui travaillent sur un corpus tenu en mémoire ; `main()` passe par
    `phrases_w2v_livre` livre par livre pour ne pas tenir 300 DataFrames.
    """
    return [p for c in corpus for p in phrases_w2v_livre(c["df"])]


def construire_w2v(phrases, params=None):
    """
    Entraine Word2Vec sur les phrases fournies et uploade les vecteurs.

    Prend les phrases deja decoupees, et non le corpus : elles sont
    accumulees pendant la passe de streaming de `main()`, ce qui evite de
    recharger les DataFrames une seconde fois. La liste de phrases reste
    en memoire (gensim fait `epochs` passes dessus et la relire depuis S3
    a chaque epoque couterait bien plus cher que de la garder).
    """
    if params is None:
        params = W2V_PARAMS
    from gensim.models import Word2Vec

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
    # `n_tokens` et `n_phrases` viennent de `charger_corpus`, qui les calcule
    # pendant le chargement : l'index n'a donc pas besoin des DataFrames.
    index = [{
        "auteur":       c["auteur"],
        "auteur_slug":  c["auteur_slug"],
        "livre":        c["livre"],
        "livre_slug":   c["livre_slug"],
        "genre":        c["genre"],
        "cle":          c["cle"],
        "n_tokens":     c["n_tokens"],
        "n_phrases":    c["n_phrases"],
    } for c in corpus]
    cle = f"{PREFIXES['artifacts']}corpus_index.json"
    storage.put_json(cle, index, metadata={"n_livres": str(len(index))})
    print(f"  index uploade : {cle} ({len(index)} livres)")


# --- Résumés MMR ---

def resumer_et_uploader(c):
    """Génère et uploade le résumé MMR d'UN livre (`c` doit porter son `df`)."""
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


def construire_resumes(corpus):
    """Génère et uploade les résumés MMR pour tous les livres."""
    for c in corpus:
        resumer_et_uploader(c)


# --- Nettoyage des resumes orphelins ---

def cles_resumes_attendues(corpus):
    """Cles S3 des resumes du corpus courant. Meme construction que l'ecriture."""
    return {
        construire_cle(
            PREFIXES["summaries"], c["genre"], c["auteur_slug"], c["livre_slug"], ".txt"
        )
        for c in corpus
    }


def nettoyer_resumes_orphelins(corpus, appliquer=False):
    """
    Signale (et supprime si `appliquer`) les resumes sans livre correspondant.

    D'ou viennent les orphelins : la cle S3 d'un resume encode le GENRE
    (`summaries/<genre>/<auteur>/<livre>.txt`). Ajouter une entree dans
    `GENRE_OVERRIDES`, ou voir Gutenberg reclasser un livre, deplace donc
    le resume sans effacer l'ancien. Un livre retire du corpus laisse
    pareillement le sien derriere lui.

    TROIS GARDE-FOUS, parce que la fonction supprime des donnees a partir
    d'un raisonnement par difference, et qu'un raisonnement par difference
    devient destructeur des que l'un des deux ensembles est faux :

    1. Un corpus vide bloque tout. Sans ce test, un `charger_corpus` qui
       ne renvoie rien (bucket mal configure, prefixe vide) ferait passer
       TOUS les resumes pour des orphelins.
    2. Un resume ATTENDU mais absent de S3 bloque la suppression. C'est le
       cas dangereux et il n'a rien d'exotique : `--skip-summaries` apres
       un ajout dans `GENRE_OVERRIDES` laisse l'ancienne cle en place et
       la nouvelle jamais ecrite. Supprimer "l'orphelin" laisserait alors
       le livre SANS AUCUN resume. On n'efface l'ancien etat que si le
       nouveau est integralement en place.
    3. Plus d'orphelins que de resumes attendus bloque aussi. Ce ratio ne
       peut pas arriver en fonctionnement normal, ou les orphelins sont
       quelques restes de reclassement ; il signale un corpus charge
       partiellement.

    Les garde-fous 2 et 3 refusent la suppression sans lever : le rebuild
    doit finir son travail (index, matrices) meme si le nettoyage est
    juge trop risqué. Seul le corpus vide leve, parce que c'est une erreur
    d'appel et non une situation de donnees.

    Ne supprime jamais sans `appliquer=True`, meme quand tout est sain :
    le defaut est de lister, l'effacement se demande explicitement.
    """
    if not corpus:
        raise ValueError(
            "Corpus vide : refus de nettoyer. Tous les resumes seraient "
            "vus comme orphelins."
        )

    attendues = cles_resumes_attendues(corpus)
    presentes = {
        o["Key"] for o in storage.list_objects(PREFIXES["summaries"], suffix=".txt")
    }
    orphelins = sorted(presentes - attendues)

    if not orphelins:
        print(f"  aucun orphelin ({len(presentes)} resumes pour {len(corpus)} livres)")
        return []

    manquantes = attendues - presentes
    if manquantes:
        print(f"  {len(orphelins)} orphelin(s) detecte(s), mais "
              f"{len(manquantes)} resume(s) attendu(s) manquent sur S3 :")
        for cle in sorted(manquantes)[:5]:
            print(f"    absent : {cle}")
        print("  REFUS de supprimer : les orphelins sont peut-etre la seule "
              "copie restante. Relance sans --skip-summaries d'abord.")
        return orphelins

    if len(orphelins) > len(attendues):
        print(f"  REFUS de supprimer : {len(orphelins)} orphelins pour "
              f"{len(attendues)} resumes attendus. Ce ratio signale un corpus "
              "charge partiellement, pas un reclassement. Verifie le corpus.")
        return orphelins

    print(f"  {len(orphelins)} orphelin(s) sur {len(presentes)} resumes :")
    for cle in orphelins:
        print(f"    {cle}")

    if not appliquer:
        print("  (rien supprime : relance avec --nettoyer-orphelins)")
        return orphelins

    for cle in orphelins:
        storage.delete_object(cle)
        print(f"    supprime : {cle}")
    return orphelins


# --- Main ---

def main():
    args = parser_args()
    storage.ensure_bucket()

    # Chargement SANS les DataFrames : le TF-IDF ne lit que les sequences de
    # termes, l'index et le nettoyage des orphelins que les metadonnees. Les
    # DataFrames sont ~3,2 Mo par livre, soit 1 Go a 300 livres qu'on ne
    # garderait que pour deux comptages deja calcules au chargement.
    corpus = charger_corpus(with_df=False)

    if not args.skip_tfidf:
        print("--- TF-IDF ---")
        construire_tfidf(corpus)
        print()

    # Word2Vec et les resumes sont les deux seuls a vouloir les DataFrames,
    # et tous deux travaillent livre par livre. Une passe de streaming les
    # sert ensemble : chaque Parquet est relu une fois, et un seul est en
    # memoire a la fois.
    if not args.skip_w2v or not args.skip_summaries:
        quoi = " et ".join(
            n for n, actif in (("Word2Vec", not args.skip_w2v),
                               ("resumes MMR", not args.skip_summaries)) if actif
        )
        print(f"--- Passe par livre : {quoi} ---")
        phrases_w2v = []
        for c in iter_corpus(with_termes=False, with_df=True, verbose=True):
            if not args.skip_w2v:
                phrases_w2v.extend(phrases_w2v_livre(c["df"]))
            if not args.skip_summaries:
                resumer_et_uploader(c)
        print()

        if not args.skip_w2v:
            print("--- Word2Vec ---")
            construire_w2v(phrases_w2v)
            print()

    # Apres l'ecriture des resumes : les cles fraiches doivent etre en
    # place avant de calculer la difference, sinon celles du run courant
    # passeraient pour des orphelins.
    print("--- Resumes orphelins ---")
    nettoyer_resumes_orphelins(corpus, appliquer=args.nettoyer_orphelins)
    print()

    print("--- Index ---")
    construire_index(corpus)
    print("\nTermine.")


if __name__ == "__main__":
    main()
