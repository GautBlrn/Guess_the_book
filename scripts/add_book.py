"""
Ajoute un nouveau livre au corpus en le passant par TOUTES les etapes.

Usage interactif :

    python -m scripts.add_book --titre "Madame Bovary"
    python -m scripts.add_book --auteur "Flaubert"
    python -m scripts.add_book --titre "Bovary" --auteur "Flaubert"
    python -m scripts.add_book --auteur "Flaubert" --rebuild

L'utilisateur voit la liste des candidats trouves dans le catalogue
Gutenberg, choisit lequel ajouter, et le script :

  1. Telecharge depuis Gutenberg
  2. Lance le pipeline complet : raw -> clean -> tokens -> annotations
  3. Si --rebuild : regenere les artefacts globaux (matrices TF-IDF,
     Word2Vec, resume MMR du nouveau livre).

Les genres/auteurs sont automatiquement extraits du catalogue Gutenberg
(via Bookshelves -> mapping). Si le genre n'est pas detectable, on
demande a l'utilisateur de le saisir.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import storage
from pipeline.config import GENRE_MAPPING
from pipeline.gutenberg import chercher, telecharger
from pipeline.parsing import slugifier
from pipeline.pipeline_full import pipeline_un_livre


# --- CLI ---

def parser_args():
    p = argparse.ArgumentParser(
        description="Ajoute un livre Gutenberg au corpus."
    )
    p.add_argument("--titre",  help="Titre (recherche partielle, insensible casse)")
    p.add_argument("--auteur", help="Auteur (recherche partielle, insensible casse)")
    p.add_argument("--langue", default="fr", help="Code langue (defaut: fr)")
    p.add_argument("--n-max",  type=int, default=10,
                   help="Nombre max de candidats a afficher (defaut: 10)")
    p.add_argument("--rebuild", action="store_true",
                   help="Regenere les artefacts globaux (TF-IDF, Word2Vec, MMR) "
                        "apres l'ajout. Optionnel.")
    p.add_argument("--non-interactive", action="store_true",
                   help="Selectionne automatiquement le 1er candidat sans demander")
    return p.parse_args()


# --- Selection interactive ---

def afficher_candidats(df):
    print(f"\n{len(df)} candidat(s) trouve(s) :\n")
    for i, row in df.iterrows():
        genre = row["genre"] or "??"
        print(f"  [{i}] pg{row['Text#']:>5}  {row['Title'][:55]:55s}  "
              f"-- {row['auteur_clean'][:30]:30s} ({genre})")


def choisir_candidat(df, non_interactive=False):
    if df.empty:
        print("Aucun candidat trouve. Affine le titre ou l'auteur.")
        return None
    afficher_candidats(df)
    if non_interactive:
        print(f"\n[non-interactif] Selection automatique : index 0")
        return df.iloc[0]
    while True:
        rep = input("\nIndex du livre a ajouter (ou 'q' pour quitter) : ").strip()
        if rep.lower() in {"q", "quit", "exit"}:
            return None
        if rep.isdigit() and int(rep) in df.index:
            return df.loc[int(rep)]
        print("Index invalide.")


def demander_genre(non_interactive=False):
    """Genre non detectable depuis Bookshelves : on demande."""
    genres_dispo = sorted(set(GENRE_MAPPING.values()))
    print("\nGenre non detecte automatiquement.")
    print(f"Genres disponibles : {', '.join(genres_dispo)}")
    if non_interactive:
        print("[non-interactif] Genre par defaut : 'novel'")
        return "novel"
    while True:
        rep = input("Genre a utiliser : ").strip().lower()
        if rep in genres_dispo:
            return rep
        print(f"Inconnu. Choisis parmi : {', '.join(genres_dispo)}")


# --- Rebuild des artefacts globaux ---

def rebuild_artefacts():
    """Delegue a scripts.rebuild_artifacts.main()."""
    from scripts.rebuild_artifacts import main as rebuild_main
    print("\n--- Regeneration des artefacts globaux ---")
    rebuild_main()


# --- Main ---

def main():
    args = parser_args()
    if not args.titre and not args.auteur:
        print("Erreur : donne au moins --titre ou --auteur.")
        sys.exit(1)

    print(f"Recherche dans le catalogue (titre={args.titre!r}, auteur={args.auteur!r}, langue={args.langue!r})...")
    candidats = chercher(
        titre=args.titre, auteur=args.auteur,
        langue=args.langue, n_max=args.n_max,
    )

    livre = choisir_candidat(candidats, non_interactive=args.non_interactive)
    if livre is None:
        print("Annule.")
        sys.exit(0)

    # Genre : utilise celui detecte, sinon demande
    genre = livre["genre"]
    if not genre:
        genre = demander_genre(non_interactive=args.non_interactive)

    auteur_slug = slugifier(livre["auteur_clean"])
    livre_slug  = f"pg{livre['Text#']}_{slugifier(livre['Title'])}"

    print(f"\n--- Ajout : pg{livre['Text#']} ({livre['Title']}) ---")
    print(f"  auteur : {livre['auteur_clean']!r} -> {auteur_slug}")
    print(f"  genre  : {genre}")
    print(f"  slug   : {livre_slug}")

    print("\n[1/2] Telechargement Gutenberg...")
    contenu = telecharger(livre["Text#"])
    if contenu is None:
        print(f"  ECHEC : impossible de telecharger pg{livre['Text#']}")
        sys.exit(1)
    print(f"  OK ({len(contenu):,} caracteres)")

    print("\n[2/2] Pipeline complet (raw -> clean -> tokens -> annotations)...")
    storage.ensure_bucket()
    resultats = pipeline_un_livre(
        contenu_raw=contenu,
        auteur_slug=auteur_slug,
        genre=genre,
        livre_slug=livre_slug,
    )

    print("\n--- Resume ---")
    print(f"  raw         : {resultats['cle_raw']}")
    print(f"  clean       : {resultats['cle_clean']}")
    print(f"  tokens      : {resultats['cle_tokens']} ({resultats['n_tokens']:,} tokens)")
    print(f"  annotations : {resultats['cle_annotations']} "
          f"({resultats['n_tokens_annotes']:,} tokens annotes, "
          f"{resultats['n_phrases']:,} phrases)")

    if args.rebuild:
        rebuild_artefacts()
    else:
        print("\nNote : pour integrer ce livre au benchmark global "
              "(matrices TF-IDF, Word2Vec, resume MMR), relance avec --rebuild "
              "ou execute python -m scripts.rebuild_artifacts")


if __name__ == "__main__":
    main()
