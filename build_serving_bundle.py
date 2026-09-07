"""
Construit le bundle de serving compact pour le backend Flask.

Le serveur Flask charge UN seul fichier au demarrage et garde tout en
memoire. Ca remplace le `charger_corpus()` du runtime, qui ramene 291
Parquet et 19,9 M de tokens. Ici on ne garde que ce qui est strictement
necessaire pour la classification d'extraits :

- Le vectoriseur TF-IDF (la meilleure config selon le benchmark)
- La matrice TF-IDF du corpus, deja L2-normalisee
- L'index ordonne des livres (auteur, livre, genre, slug, n_tokens)
- Les chemins S3 vers les resumes pre-generes

Le bundle est uploade dans `artifacts/serving_bundle.pkl` sur S3.
Au demarrage, Flask le telecharge en quelques secondes (typiquement
5-15 Mo selon le vocabulaire), puis ne touche plus a S3 sauf pour
recuperer un resume (lecture texte rapide).

Choix de la config TF-IDF par defaut : `tokens_12g` + `cosinus`, retenue
par le bloc de mesure plus bas (93.95 % top-1 sur 1455 extraits de longueur
realiste et 291 livres, contre 92.65 % pour `lemmes_12g` et 70.31 % pour
`lemmes_1g`). L'utilisateur final ne pourra pas choisir d'autre config,
c'est volontaire pour l'app deployee, qui doit etre simple et toujours
utiliser le meilleur reglage connu.

Ce script est a la RACINE du projet et non dans `scripts/`, donc il se
lance en fichier :

    python build_serving_bundle.py

    # Forcer une autre config (rare, pour test) :
    python build_serving_bundle.py --champ lemmes --ngram 2
"""
import argparse
import io
import pickle
import sys
import time
from pathlib import Path

# --- RACINE_DEPOT ---
# Ce fichier vit a la RACINE du projet, donc son propre repertoire est celui
# qui contient `pipeline/`. Le `parent.parent` d'origine, herite de l'epoque ou
# il etait dans `scripts/`, pointait un cran trop haut : il ne fonctionnait que
# parce que le repertoire courant est deja dans sys.path quand on lance le
# script depuis la racine.
#
# Mais `Path(__file__).parent` ne suffit pas non plus, parce que le fichier est
# livre a deux endroits :
#
#     dans le depot     projet_nlp_app/build_serving_bundle.py
#     dans la remise    Projet_2_.../Modeles/A_Gautier_Blairon/code_entrainement/
#                       ou le repertoire du fichier ne porte aucun `pipeline/`
#
# La racine cherchee est celle qui porte `pipeline/config.py`. On la trouve par
# ce repere plutot que par un comptage de niveaux, et le repli est explicite.
# Le bloc est recopie depuis `scripts/rebuild_artifacts.py` plutot que mis en
# commun : il s'execute avant que `pipeline` soit importable, donc il ne peut
# pas en venir.
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

# --- Config par defaut du bundle ---
# `tokens_12g` + cosinus, avec le min_df=1 de TFIDF_PARAMS.
#
# Mesure sur 1455 extraits de LONGUEUR VARIABLE (plage 7-70 termes), 291
# livres, ce qui est le regime reel de l'app :
#
#     lemmes_1g   : 70.31 % top-1,  144 k termes
#     lemmes_12g  : 92.65 % top-1,  4,1 M termes
#     tokens_12g  : 93.95 % top-1,  5,0 M termes
#
# TROIS ARBITRAGES, TOUS TRANCHES PAR DES TESTS APPARIES et non par la
# comparaison de taux agreges. Le meme jeu d'extraits passe dans toutes les
# configs, et `tirer_extraits` fournit meme les deux champs du MEME passage :
# comparer les taux comme s'il s'agissait d'echantillons independants
# jetterait l'appariement, qui est l'information la plus utile.
#
# 1. BIGRAMMES contre unigrammes : +22,34 points, McNemar p = 2,8e-93, sur
#    331 paires discordantes reparties 328 contre 3. Ecrasant. Un extrait
#    court contient peu d'unigrammes discriminants, et un unigramme rare
#    dans 26 livres ne l'est plus dans 291, alors qu'un bigramme reste
#    presque unique quelle que soit la taille du corpus.
#
# 2. TOKENS contre lemmes : +1,31 point, McNemar p = 8,8e-4, Wilcoxon sur
#    les rangs reciproques p = 1,6e-4, sur 31 paires discordantes reparties
#    25 contre 6.
#
#    C'EST UN CHANGEMENT DE DECISION. A 26 livres, `tokens_12g` devancait
#    `lemmes_12g` d'un extrait sur 520 et l'ecart avait ete juge non
#    significatif, a juste titre. A 291 livres il ne l'est plus. La
#    lemmatisation normalise les formes flechies, ce qui aide a regrouper
#    deux occurrences d'un meme mot mais efface aussi des indices : sur une
#    tache d'IDENTIFICATION, la forme exacte employee par l'auteur est
#    elle-meme une signature. Le prix est un vocabulaire 22 % plus gros.
#
# 3. min_df = 1 contre 2 : +10,03 points, McNemar p = 2,0e-39. Le reglage
#    ne rapportait que 1,5 point a 26 livres ; sa justification s'est donc
#    RENFORCEE avec l'echelle, ce qui n'allait pas de soi. Le prix est un
#    facteur 5,4 sur le vocabulaire (4,1 M contre 773 k pour les lemmes).
#
# La contrainte memoire qui pesait sur ces choix est levee. `TfIdfMaison`
# renvoyait une matrice DENSE, dont le cout grimpe comme N^2. A 291 livres
# `tokens_12g` demanderait 291 x 5 048 963 x 8 octets, soit 11,8 Go : le
# bundle n'aurait pas pu etre construit. La matrice est creuse depuis (cf.
# le docstring de `pipeline/representations.py`), pour un resultat
# numeriquement identique.
#
# Le bundle servi grossit avec le corpus mais reste transportable : c'est
# le nnz qui croit, lineairement avec le nombre de livres, et non le
# produit N x V.
#
# Le choix du cosinus plutot que de l'euclidien est cosmetique : sur des
# vecteurs L2-normalises les deux donnent le MEME classement (cf. le
# commentaire de section dans pipeline/benchmark.py et la conclusion du
# notebook 05).
#
# Reproduire : python -m scripts.run_benchmark --skip-resumes
#              python -m scripts.tests_apparies
DEFAULT_CHAMP = "tokens"
DEFAULT_NGRAM_MAX = 2
DEFAULT_METRIQUE = "cosinus"


def parser_args():
    p = argparse.ArgumentParser(
        description="Construit le bundle de serving pour Flask."
    )
    p.add_argument(
        "--champ",
        choices=["lemmes", "tokens"],
        default=DEFAULT_CHAMP,
        help=f"Champ pour le TF-IDF (defaut: {DEFAULT_CHAMP})",
    )
    p.add_argument(
        "--ngram",
        type=int,
        choices=[1, 2],
        default=DEFAULT_NGRAM_MAX,
        help=f"ngram_max pour le TF-IDF (defaut: {DEFAULT_NGRAM_MAX})",
    )
    p.add_argument(
        "--sortie",
        default=None,
        help="Cle S3 de sortie (defaut: artifacts/serving_bundle.pkl)",
    )
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
from pipeline.config import PREFIXES, TFIDF_PARAMS
from pipeline.corpus import charger_corpus
from pipeline.parsing import construire_cle
from pipeline.representations import TfIdfMaison


def construire_bundle(champ, ngram_max):
    """Charge le corpus, fit le TF-IDF, prepare l'index, retourne le bundle."""
    print(f"\n--- Chargement du corpus ---")
    t0 = time.time()
    # Sans les DataFrames : le bundle ne retient que la matrice, le
    # vectoriseur et l'index. `n_tokens` et `n_phrases` sont calcules au
    # chargement, donc l'index ne perd rien.
    corpus = charger_corpus(with_df=False, verbose=True)
    print(f"  charge en {time.time() - t0:.1f}s")

    print(f"\n--- TF-IDF (champ={champ}, ngram_max={ngram_max}) ---")
    t0 = time.time()
    documents = [c[champ] for c in corpus]
    vec = TfIdfMaison(
        ngram_max=ngram_max,
        min_df=TFIDF_PARAMS["min_df"],
        max_df_ratio=TFIDF_PARAMS["max_df_ratio"],
    )
    matrice = vec.fit_transform(documents)
    print(f"  matrice {matrice.shape}, vocabulaire {len(vec.vocabulaire_):,} termes")
    print(f"  fit en {time.time() - t0:.1f}s")

    print(f"\n--- Construction de l'index ---")
    # Ordre identique a l'ordre des lignes de la matrice TF-IDF.
    # On garde uniquement les champs utiles cote serveur.
    index = []
    for i, c in enumerate(corpus):
        # Cle du resume MMR pre-genere (relation directe avec construire_cle).
        cle_resume = construire_cle(
            PREFIXES["summaries"],
            c["genre"],
            c["auteur_slug"],
            c["livre_slug"],
            ".txt",
        )
        index.append({
            "i":           i,                      # ligne dans la matrice
            "auteur":      c["auteur"],
            "auteur_slug": c["auteur_slug"],
            "livre":       c["livre"],
            "livre_slug":  c["livre_slug"],
            "genre":       c["genre"],
            "n_tokens":    c["n_tokens"],
            "n_phrases":   c["n_phrases"],
            "cle_resume":  cle_resume,
        })

    bundle = {
        "version":     1,
        "config": {
            "champ":      champ,
            "ngram_max":  ngram_max,
            "metrique":   DEFAULT_METRIQUE,
            "min_df":     TFIDF_PARAMS["min_df"],
            "max_df_ratio": TFIDF_PARAMS["max_df_ratio"],
        },
        "vectoriseur": vec,
        "matrice":     matrice,    # deja L2-normalisee par TfIdfMaison
        "index":       index,
    }
    return bundle


def uploader_bundle(bundle, cle_sortie):
    """Pickle + upload vers S3."""
    print(f"\n--- Serialisation et upload ---")
    t0 = time.time()
    buf = io.BytesIO()
    pickle.dump(bundle, buf, protocol=pickle.HIGHEST_PROTOCOL)
    octets = buf.getvalue()
    taille_mo = len(octets) / (1024 * 1024)
    print(f"  taille bundle : {taille_mo:.1f} Mo "
          f"({len(octets):,} octets)")

    storage.put_bytes(
        cle_sortie,
        octets,
        content_type="application/octet-stream",
        metadata={
            "version":     str(bundle["version"]),
            "champ":       bundle["config"]["champ"],
            "ngram_max":   str(bundle["config"]["ngram_max"]),
            "n_livres":    str(len(bundle["index"])),
            "vocab_size":  str(len(bundle["vectoriseur"].vocabulaire_)),
        },
    )
    print(f"  uploade vers s3://{cle_sortie} en {time.time() - t0:.1f}s")


def main():
    args = parser_args()
    storage.ensure_bucket()

    cle_sortie = args.sortie or f"{PREFIXES['artifacts']}serving_bundle.pkl"

    bundle = construire_bundle(args.champ, args.ngram)
    uploader_bundle(bundle, cle_sortie)

    # Recap
    print(f"\n{'=' * 60}")
    print("Bundle construit avec succes.")
    print(f"  Cle S3       : {cle_sortie}")
    print(f"  Livres       : {len(bundle['index'])}")
    print(f"  Vocabulaire  : {len(bundle['vectoriseur'].vocabulaire_):,}")
    print(f"  Matrice      : {bundle['matrice'].shape}")
    print(f"  Config       : {bundle['config']['champ']}_"
          f"{bundle['config']['ngram_max']}g")
    print(f"\nLe serveur Flask peut maintenant charger ce bundle au demarrage.")


if __name__ == "__main__":
    main()
