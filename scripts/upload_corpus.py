"""
Upload initial du corpus Gutenberg vers S3 (etage `raw/`).

Tire N livres au hasard du catalogue (filtres par langue, genre détectable,
auteur présent), télécharge le contenu Gutenberg, et upload tel quel sous
`raw/<genre>/<auteur_slug>/pg<id>_<titre_slug>.txt`.

Le NETTOYAGE ne se fait PAS ici : c'est le notebook 01_clean_book qui
produit `clean/`. Cela permet de re-nettoyer sans retélécharger.

Usage : python -m scripts.upload_corpus
"""
import sys
import time
from pathlib import Path

# Ajout du parent au sys.path quand le script est lancé directement
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import storage
from pipeline.config import CATALOG_PATH, DEFAULT_LANGUAGE, PREFIXES
from pipeline.gutenberg import _charger_catalogue, telecharger
from pipeline.parsing import construire_cle, slugifier


CFG_LOCAL = {
    "n_books":     10,
    "delay_s":     1.0,
    "random_seed": 43,
}


def selectionner(n_books, langue=DEFAULT_LANGUAGE, seed=43):
    """Tire N livres FR au hasard avec genre detectable et auteur present."""
    df = _charger_catalogue()
    if langue:
        df = df[df["Language"].fillna("").str.split("; ").apply(
            lambda l: langue in l
        )]
    df = df[df["Title"].notna() & df["auteur_clean"].notna() & df["genre"].notna()]
    print(f"Livres eligibles ({langue}) : {len(df)}")
    print("Repartition par genre :")
    print(df["genre"].value_counts(), "\n")
    return df.sample(n_books, random_state=seed)


def main():
    storage.ensure_bucket()

    selection = selectionner(
        n_books=CFG_LOCAL["n_books"],
        seed=CFG_LOCAL["random_seed"],
    )
    print(f"Echantillon tire : {len(selection)} livres\n")

    reussis, echecs = 0, []
    for _, row in selection.iterrows():
        tid     = row["Text#"]
        auteur  = slugifier(row["auteur_clean"])
        genre   = row["genre"]
        titre   = row["Title"]
        livre_slug = f"pg{tid}_{slugifier(titre)}"

        print(f"[{tid}] {titre} -- {auteur} ({genre})")

        contenu = telecharger(tid)
        if contenu is None:
            print("  Telechargement echoue")
            echecs.append(tid)
            time.sleep(CFG_LOCAL["delay_s"])
            continue

        cle = construire_cle(PREFIXES["raw"], genre, auteur, livre_slug, ".txt")
        storage.put_text(
            cle, contenu,
            metadata={"auteur": auteur, "genre": genre, "source": "gutenberg",
                      "text_id": str(tid)},
        )
        print(f"  Uploade : {cle}")
        reussis += 1
        time.sleep(CFG_LOCAL["delay_s"])

    print(f"\n--- Bilan : {reussis}/{len(selection)} uploades ---")
    if echecs:
        print(f"Echecs (IDs Gutenberg) : {echecs}")


if __name__ == "__main__":
    main()
