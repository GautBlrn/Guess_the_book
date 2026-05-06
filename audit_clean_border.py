"""
Inspection des débuts et fins des textes nettoyés.

Pour chaque livre présent sous le prefixe `clean/`, dump dans un seul
fichier texte les `n_chars_debut` premiers et `n_chars_fin` derniers
caractères, avec un en-tête clair séparant chaque livre.

Sert à auditer visuellement ce qui passe au travers des regex de
`cleaning.py` : en-têtes éditoriaux non standards, pieds de page
ebooksgratuits, dedicaces, mentions de transcription, décorations,
notes de fin, etc.

Usage :
    python audit_clean_borders.py

Sortie :
    audit_clean_borders.txt -- un seul fichier prêt a etre partage.

Paramètres ajustables en haut de fichier :
    N_CHARS_DEBUT : nombre de caracteres a montrer au debut de chaque livre
    N_CHARS_FIN   : idem a la fin
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import storage
from pipeline.config import PREFIXES
from pipeline.parsing import parser_chemin


# --- Paramètres ---
N_CHARS_DEBUT = 2500     # large, pour voir 30-40 lignes
N_CHARS_FIN   = 2500
SORTIE        = "audit_clean_borders.txt"


def formater_borne(texte, n, label):
    """Tronque le texte a n caracteres en marquant ou on coupe."""
    if len(texte) <= n:
        return texte, False
    return texte[:n], True


def main():
    print(f"Listing des fichiers sous {PREFIXES['clean']}...")
    objets = storage.list_objects(PREFIXES["clean"], suffix=".txt")
    if not objets:
        print(f"Aucun .txt trouve sous {PREFIXES['clean']}.")
        return

    objets = sorted(objets, key=lambda o: o["Key"])
    print(f"  {len(objets)} fichiers trouves.\n")

    sortie_path = Path(SORTIE)
    t0 = time.time()
    n_lus = 0

    with sortie_path.open("w", encoding="utf-8") as f_out:
        # En-tête général
        f_out.write("=" * 100 + "\n")
        f_out.write(f"AUDIT DES BORDS DE TEXTES NETTOYES\n")
        f_out.write(f"Genere le : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_out.write(f"Nombre de fichiers : {len(objets)}\n")
        f_out.write(f"Caracteres affiches : "
                     f"{N_CHARS_DEBUT} debut + {N_CHARS_FIN} fin\n")
        f_out.write("=" * 100 + "\n\n\n")

        for i, obj in enumerate(objets, 1):
            cle = obj["Key"]
            try:
                info = parser_chemin(cle, PREFIXES["clean"])
            except Exception as e:
                f_out.write(f"### {cle} -- erreur de parsing : {e}\n\n")
                continue

            try:
                texte = storage.get_text(cle)
            except Exception as e:
                f_out.write(f"### {cle} -- erreur de lecture : {e}\n\n")
                continue

            taille = len(texte)
            debut, debut_tronque = formater_borne(texte, N_CHARS_DEBUT, "debut")
            fin_chars = N_CHARS_FIN if taille > N_CHARS_DEBUT + N_CHARS_FIN else 0
            fin = texte[-fin_chars:] if fin_chars else ""
            fin_tronque = fin_chars > 0

            # Bloc par livre
            f_out.write("#" * 100 + "\n")
            f_out.write(f"# [{i}/{len(objets)}] {info['auteur']} -- {info['livre']}\n")
            f_out.write(f"# genre = {info['genre']}\n")
            f_out.write(f"# cle   = {cle}\n")
            f_out.write(f"# taille = {taille:,} caracteres\n")
            f_out.write("#" * 100 + "\n\n")

            f_out.write("---------- DEBUT ----------\n")
            f_out.write(debut)
            if debut_tronque:
                f_out.write(f"\n[... {taille - N_CHARS_DEBUT:,} caracteres tronques ...]\n")
            f_out.write("\n---------- FIN DEBUT ----------\n\n")

            if fin_tronque:
                f_out.write("---------- FIN ----------\n")
                f_out.write(f"[... {taille - N_CHARS_DEBUT - N_CHARS_FIN:,} "
                            f"caracteres tronques ...]\n")
                f_out.write(fin)
                f_out.write("\n---------- FIN FIN ----------\n\n\n")
            else:
                f_out.write("[fichier trop court : debut couvre tout]\n\n\n")

            n_lus += 1
            if i % 5 == 0:
                print(f"  {i}/{len(objets)} traites...")

    duree = time.time() - t0
    taille_sortie = sortie_path.stat().st_size
    print(f"\nTermine en {duree:.1f}s.")
    print(f"  {n_lus} livres dumpes")
    print(f"  Sortie : {sortie_path.resolve()}")
    print(f"  Taille : {taille_sortie:,} octets "
          f"(~{taille_sortie / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()