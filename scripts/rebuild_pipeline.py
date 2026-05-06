"""
Régénère clean/, tokens/, annotations/ pour tous les livres du bucket.

A utiliser après une amélioration du cleaning, du tokenizer, ou de
l'annotation : on n'a pas besoin de re-télécharger Gutenberg, le `raw/`
est déjà là. On itère sur les fichiers `raw/<genre>/<auteur>/<livre>.txt`
et on relance `pipeline_un_livre()` pour chacun.

Le rebuild des artefacts globaux (TF-IDF, Word2Vec, MMR) se fait
ensuite avec `python -m scripts.rebuild_artifacts`. Il n'est pas
déclenché automatiquement parce qu'il dure plusieurs minutes ; on
veut pouvoir vérifier le clean/annotations avant de régénérer le reste.

Usage :

    python -m scripts.rebuild_pipeline
    python -m scripts.rebuild_pipeline --filtre "munchausen"   # un seul livre
    python -m scripts.rebuild_pipeline --filtre "zola"          # tous les livres d'un auteur
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import storage
from pipeline.config import PREFIXES
from pipeline.parsing import parser_chemin
from pipeline.pipeline_full import pipeline_un_livre


def parser_args():
    p = argparse.ArgumentParser(
        description="Régenère clean/, tokens/, annotations/ à partir de raw/."
    )
    p.add_argument(
        "--filtre", default=None,
        help="Ne traite que les livres dont la clé S3 contient ce motif "
             "(ex. --filtre 'munchausen' ou --filtre 'zola')",
    )
    return p.parse_args()


def main():
    args = parser_args()

    livres_raw = storage.list_objects(PREFIXES["raw"], suffix=".txt")
    if not livres_raw:
        print(f"Aucun .txt sous {PREFIXES['raw']}. Lance d'abord upload_corpus.")
        sys.exit(1)

    if args.filtre:
        avant = len(livres_raw)
        livres_raw = [
            o for o in livres_raw if args.filtre.lower() in o["Key"].lower()
        ]
        print(f"Filtre '{args.filtre}' : {len(livres_raw)}/{avant} livres retenus")
        if not livres_raw:
            sys.exit(0)

    print(f"\n{len(livres_raw)} livre(s) à traiter\n")

    reussis, echecs = 0, []
    debut_global = time.time()

    for i, obj in enumerate(livres_raw, 1):
        cle = obj["Key"]
        info = parser_chemin(cle, PREFIXES["raw"])
        print(f"[{i:2d}/{len(livres_raw)}] {info['auteur']} -- {info['livre']} ({info['genre']})")

        try:
            contenu = storage.get_text(cle)
            pipeline_un_livre(
                contenu_raw=contenu,
                auteur_slug=info["auteur_slug"],
                genre=info["genre"],
                livre_slug=info["livre_slug"],
                verbose=False,
            )
            reussis += 1
        except Exception as e:
            print(f"  Echec : {type(e).__name__} : {e}")
            echecs.append((cle, str(e)))

    duree = time.time() - debut_global
    print(f"\n--- Termine en {duree:.0f}s ---")
    print(f"  Reussis : {reussis}/{len(livres_raw)}")
    if echecs:
        print(f"  Echecs : {len(echecs)}")
        for cle, err in echecs:
            print(f"    {cle} : {err}")

    print("\nPour régénérer aussi les artéfacts globaux "
          "(TF-IDF, Word2Vec, resumes MMR) :")
    print("    python -m scripts.rebuild_artifacts")


if __name__ == "__main__":
    main()
