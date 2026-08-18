"""
Construit le bundle de serving compact pour le backend Flask.

Le serveur Flask charge UN seul fichier au demarrage et garde tout en
memoire. Ca remplace le `charger_corpus()` du runtime, qui ramene 26
Parquet et 600k tokens. Ici on ne garde que ce qui est strictement
necessaire pour la classification d'extraits :

- Le vectoriseur TF-IDF (la meilleure config selon le benchmark)
- La matrice TF-IDF du corpus, deja L2-normalisee
- L'index ordonne des livres (auteur, livre, genre, slug, n_tokens)
- Les chemins S3 vers les resumes pre-generes

Le bundle est uploade dans `artifacts/serving_bundle.pkl` sur S3.
Au demarrage, Flask le telecharge en quelques secondes (typiquement
5-15 Mo selon le vocabulaire), puis ne touche plus a S3 sauf pour
recuperer un resume (lecture texte rapide).

Choix de la config TF-IDF par defaut : `lemmes_1g` + `cosinus` ressort
a 99.2 % top-1 dans ton benchmark sur le corpus actuel. C'est cette
config que le bundle contient. L'utilisateur final ne pourra pas
choisir d'autre config -- c'est volontaire pour l'app deployee, qui
doit etre simple et toujours utiliser le meilleur reglage connu.

Usage :

    python -m scripts.build_serving_bundle

    # Forcer une autre config (rare, pour test) :
    python -m scripts.build_serving_bundle --champ lemmes --ngram 2
"""
import argparse
import io
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import storage
from pipeline.config import PREFIXES, TFIDF_PARAMS
from pipeline.corpus import charger_corpus
from pipeline.parsing import construire_cle
from pipeline.representations import TfIdfMaison


# --- Config par defaut du bundle ---
# `lemmes_12g` + cosinus, avec le min_df=1 de TFIDF_PARAMS.
#
# Mesure sur 520 extraits de LONGUEUR VARIABLE (plage 7-70 termes, mediane
# 38, soit ~108 mots colles), ce qui est le regime reel de l'app :
#
#     lemmes_1g   : 87.50 % top-1,  38 k termes,   7,9 Mo
#     lemmes_12g  : 97.88 % top-1, 734 k termes, 152,7 Mo
#
# Soit 10,4 points pour 145 Mo. L'arbitrage precedent retenait `lemmes_1g`,
# mais il reposait sur des extraits fixes de 200 termes, ou l'ecart n'etait
# que de 1,2 point (98.46 contre 99.62). Or 200 termes informatifs valent
# ~573 mots colles, presque le double du maximum de l'app : ce protocole ne
# mesurait jamais le regime de production. Un extrait court contient peu
# d'unigrammes, donc la preuve apportee par les bigrammes y pese
# proportionnellement bien plus -- d'ou l'ecart qui explose quand on mesure
# ce que les utilisateurs collent vraiment.
#
# ATTENTION : la matrice est DENSE (`TfIdfMaison.fit_transform` fait un
# `toarray()`) et le vocabulaire croit avec le corpus, donc la memoire
# grimpe en gros comme N^2. 153 Mo a 26 livres, plusieurs Go a 100. Passer
# `TfIdfMaison` en csr_matrix est le prerequis a toute croissance du corpus
# avec cette config.
#
# `tokens_12g` fait 98.08 %, soit un extrait de plus sur 520 : ecart non
# significatif, on reste sur les lemmes.
#
# Le choix du cosinus plutot que de l'euclidien est cosmetique : sur des
# vecteurs L2-normalises les deux donnent le MEME classement (cf. le
# commentaire de section dans pipeline/benchmark.py et la conclusion du
# notebook 05).
#
# Reproduire : python -m scripts.run_benchmark --skip-resumes
DEFAULT_CHAMP = "lemmes"
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


def construire_bundle(champ, ngram_max):
    """Charge le corpus, fit le TF-IDF, prepare l'index, retourne le bundle."""
    print(f"\n--- Chargement du corpus ---")
    t0 = time.time()
    corpus = charger_corpus(verbose=True)
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
            "n_tokens":    int(len(c["df"])),
            "n_phrases":   int(c["df"]["sent_id"].nunique()),
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
