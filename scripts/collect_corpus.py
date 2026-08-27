"""
Monte un corpus de plusieurs centaines de livres, par quotas de genre.

Remplace `scripts/upload_corpus.py` des que la cible depasse la dizaine de
livres. Trois differences, toutes motivees par le passage a l'echelle :

  SELECTION      quota par genre au lieu d'un tirage uniforme, sinon le
                 corpus recopie le desequilibre du catalogue.
  VALIDATION     chaque texte telecharge passe les portes de
                 `pipeline/collecte.valider_texte` avant d'entrer.
  REPRISE        les livres deja presents sous `raw/` sont sautes, donc une
                 interruption ne coute que le livre en cours.

Le detail du POURQUOI de chaque porte est dans `pipeline/collecte`.

Usage :

    # Voir ce qui serait collecte, sans rien telecharger ni ecrire
    python -m scripts.collect_corpus --n-par-genre 20 --dry-run

    # Collecte reelle : brut uniquement, comme upload_corpus
    python -m scripts.collect_corpus --n-par-genre 20

    # Collecte + pipeline complet (raw -> clean -> tokens -> annotations)
    python -m scripts.collect_corpus --n-par-genre 20 --pipeline

    # Reprendre une collecte interrompue : meme commande, les livres deja
    # presents sont sautes.
    python -m scripts.collect_corpus --n-par-genre 20 --pipeline

    # Se limiter a quelques genres
    python -m scripts.collect_corpus --genres mystery scifi_fantasy --n-par-genre 40

APRES la collecte, il reste a regenerer les artefacts globaux :

    python -m scripts.rebuild_artifacts

QUELLE DUREE. Le telechargement est borne par la politesse envers Gutenberg
(`--delai`, 1 s par defaut) et l'annotation par spaCy. Sur le corpus temoin
l'annotation tourne autour de 20 s pour un livre de 700 k caracteres, donc
compter environ 2 h pour 320 livres avec `--pipeline`, et quelques minutes
sans. Le script est reprenable precisement parce que c'est long.
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import storage
from pipeline.collecte import selectionner_stratifie, valider_texte
from pipeline.config import COLLECTE_PARAMS, DEFAULT_LANGUAGE, PREFIXES
from pipeline.gutenberg import _charger_catalogue, telecharger
from pipeline.parsing import construire_cle, slugifier
from pipeline.pipeline_full import pipeline_un_livre


# --- CLI ---

def parser_args():
    p = argparse.ArgumentParser(
        description="Monte un corpus Gutenberg par quotas de genre.",
    )
    p.add_argument("--n-par-genre", type=int, default=20,
                   help="Plafond de livres retenus par genre (defaut: 20). "
                        "Les genres moins fournis donnent ce qu'ils ont.")
    p.add_argument("--genres", nargs="+", default=None,
                   help="Restreint a ces genres (defaut: tous ceux du catalogue)")
    p.add_argument("--langue", default=DEFAULT_LANGUAGE,
                   help=f"Code langue du catalogue (defaut: {DEFAULT_LANGUAGE})")
    p.add_argument("--seed", type=int, default=42,
                   help="Graine du tirage dans chaque genre (defaut: 42)")
    p.add_argument("--delai", type=float, default=1.0,
                   help="Pause entre deux telechargements, en secondes (defaut: 1.0)")
    p.add_argument("--pipeline", action="store_true",
                   help="Enchaine clean -> tokens -> annotations apres l'upload "
                        "du brut. Sans ce flag, seul raw/ est ecrit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Affiche la selection et s'arrete. Aucun telechargement, "
                        "aucune ecriture.")
    return p.parse_args()


# --- Selection ---

def catalogue_eligible(langue):
    """Catalogue filtre sur ce qui est exploitable par le pipeline."""
    df = _charger_catalogue()
    if langue:
        df = df[df["Language"].fillna("").str.split("; ").apply(
            lambda l: langue in l
        )]
    return df[df["Title"].notna()
              & df["auteur_clean"].notna()
              & df["genre"].notna()]


def afficher_selection(selection, deja_presents):
    """Recapitule ce qui sera tente, genre par genre."""
    print(f"\n{'genre':20s} {'retenus':>8s} {'deja la':>8s} {'a faire':>8s}")
    print("-" * 48)
    total_a_faire = 0
    for genre, groupe in selection.groupby("genre"):
        deja = sum(str(row["Text#"]) in deja_presents for _, row in groupe.iterrows())
        a_faire = len(groupe) - deja
        total_a_faire += a_faire
        print(f"{genre:20s} {len(groupe):8d} {deja:8d} {a_faire:8d}")
    print("-" * 48)
    print(f"{'TOTAL':20s} {len(selection):8d} "
          f"{len(selection) - total_a_faire:8d} {total_a_faire:8d}")
    return total_a_faire


# --- Reprise ---

def _slug_livre(row):
    return f"pg{row['Text#']}_{slugifier(row['Title'])}"


ID_DANS_SLUG = re.compile(r"^pg(\d+)_")


def ids_deja_presents(prefixe):
    """
    Identifiants Gutenberg des livres deja ecrits sous un prefixe S3.

    Sert la reprise : lister le prefixe suffit a savoir ce qui est fait,
    sans etat local a maintenir. On lit `raw/` et pas `annotations/` parce
    que c'est raw qui marque qu'un livre a ete accepte -- l'annotation, elle,
    peut manquer si la collecte precedente tournait sans `--pipeline`.

    ON COMPARE LES IDENTIFIANTS, PAS LES SLUGS COMPLETS. Le slug derive du
    titre du catalogue, et ce titre bouge : sur les 26 livres du corpus
    initial, deux slugs recalcules aujourd'hui ne retombent plus sur ceux
    stockes, Gutenberg ayant retouche les titres depuis (« Oeuvres » devenu
    « Œuvres », un titre rallonge de sa liste de dates).

    Comparer les slugs ferait donc passer ces livres pour absents, les
    reteleporterait sous une NOUVELLE cle, et le corpus se retrouverait avec
    deux exemplaires du meme ouvrage. Un doublon n'est pas anodin ici : il
    compte pour deux documents dans le calcul de l'IDF, ce qui abaisse le
    poids des termes propres a ce livre et le DESAVANTAGE a la recherche,
    en plus de rendre son identification ambigue entre ses deux copies.

    L'identifiant Gutenberg, lui, est la cle primaire de leur catalogue et
    ne bouge pas.
    """
    objets = storage.list_objects(prefixe, suffix=".txt")
    ids = set()
    for o in objets:
        m = ID_DANS_SLUG.match(Path(o["Key"]).stem)
        if m:
            ids.add(m.group(1))
    return ids


# --- Collecte ---

def collecter(selection, deja_presents, avec_pipeline, delai):
    """
    Telecharge, valide et ecrit chaque livre retenu.

    Renvoie le compteur des issues : `ok`, `deja_present`,
    `echec_telechargement`, plus une entree par raison de rejet
    (`pas_francais`, `trop_court`, ...).
    """
    bilan = Counter()
    rejets = []

    for n, (_, row) in enumerate(selection.iterrows(), start=1):
        tid = row["Text#"]
        titre = row["Title"]
        genre = row["genre"]
        auteur = slugifier(row["auteur_clean"])
        livre_slug = _slug_livre(row)

        if str(tid) in deja_presents:
            bilan["deja_present"] += 1
            continue

        print(f"[{n}/{len(selection)}] pg{tid} {titre[:50]} ({genre})")

        contenu = telecharger(tid)
        if contenu is None:
            print("    echec de telechargement")
            bilan["echec_telechargement"] += 1
            rejets.append((tid, titre, "echec_telechargement"))
            time.sleep(delai)
            continue

        valide, raison = valider_texte(contenu, genre=genre)
        if not valide:
            print(f"    ecarte : {raison} ({len(contenu):,} caracteres)")
            bilan[raison] += 1
            rejets.append((tid, titre, raison))
            time.sleep(delai)
            continue

        if avec_pipeline:
            pipeline_un_livre(
                contenu_raw=contenu,
                auteur_slug=auteur,
                genre=genre,
                livre_slug=livre_slug,
                verbose=True,
            )
        else:
            cle = construire_cle(PREFIXES["raw"], genre, auteur, livre_slug, ".txt")
            storage.put_text(
                cle, contenu,
                metadata={"auteur": auteur, "genre": genre,
                          "source": "gutenberg", "text_id": str(tid)},
            )
            print(f"    raw -> {cle} ({len(contenu):,} caracteres)")

        bilan["ok"] += 1
        time.sleep(delai)

    return bilan, rejets


def afficher_bilan(bilan, rejets, selection):
    """Bilan final, avec le detail des rejets pour pouvoir les reexaminer."""
    print("\n" + "=" * 58)
    print("BILAN")
    print("=" * 58)
    print(f"  selectionnes         {len(selection):5d}")
    print(f"  deja presents        {bilan['deja_present']:5d}")
    print(f"  ajoutes              {bilan['ok']:5d}")

    causes = {c: n for c, n in bilan.items() if c not in {"ok", "deja_present"}}
    if causes:
        print(f"  ecartes              {sum(causes.values()):5d}")
        for cause, n in sorted(causes.items(), key=lambda kv: -kv[1]):
            print(f"      {cause:24s} {n:5d}")

    if rejets:
        print(f"\n  Detail des {len(rejets)} rejets :")
        for tid, titre, raison in rejets:
            print(f"      pg{str(tid):<7s} {raison:22s} {titre[:44]}")

    if bilan["ok"]:
        print("\nProchaine etape : python -m scripts.rebuild_artifacts")


# --- Main ---

def main():
    args = parser_args()

    print(f"Catalogue : langue={args.langue!r}")
    df = catalogue_eligible(args.langue)
    print(f"  {len(df):,} livres eligibles (titre + auteur + genre detectable)")

    selection = selectionner_stratifie(
        df, n_par_genre=args.n_par_genre, seed=args.seed, genres=args.genres,
    )
    if selection.empty:
        print("Aucun livre ne correspond a ces criteres.")
        sys.exit(1)

    # La reprise a besoin de savoir ce qui est deja la. En dry-run on lit
    # aussi S3, mais uniquement pour lister : c'est une lecture.
    deja = ids_deja_presents(PREFIXES["raw"])
    a_faire = afficher_selection(selection, deja)

    if args.dry_run:
        print("\n[dry-run] Rien n'a ete telecharge ni ecrit.")
        seuils = COLLECTE_PARAMS
        print(f"\nPortes de validation qui s'appliqueraient :")
        print(f"  taille brute       {seuils['taille_min_defaut']:,} "
              f"a {seuils['taille_max']:,} caracteres")
        print(f"  part anglaise max  {seuils['part_anglaise_max']}")
        print(f"  ratio nettoyage    >= {seuils['ratio_nettoyage_min']}")
        return

    if not a_faire:
        print("\nTout est deja collecte. Rien a faire.")
        return

    storage.ensure_bucket()
    etapes = "raw -> clean -> tokens -> annotations" if args.pipeline else "raw"
    print(f"\nCollecte de {a_faire} livre(s), etapes : {etapes}\n")

    t0 = time.time()
    bilan, rejets = collecter(selection, deja, args.pipeline, args.delai)
    afficher_bilan(bilan, rejets, selection)
    print(f"\nDuree : {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
