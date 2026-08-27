"""
Tests apparies entre methodes de resume, livre par livre.

POURQUOI PAS LES MOYENNES. Sur 291 livres, MMR sort a 0,381 de redondance
contre 0,405 pour `tfidf_naif`, et 0,255 de couverture contre 0,248. Deux
ecarts dans le bon sens, mais des moyennes ne disent pas si MMR gagne
REGULIEREMENT ou s'il doit son avance a quelques livres atypiques. Les
ecarts-types valent ici 0,10 sur la redondance, soit quatre fois l'ecart
entre les deux methodes : lire ces moyennes comme un classement serait une
erreur.

L'appariement est naturel et fort : chaque livre est resume par toutes les
methodes a partir du MEME pool de phrases et du MEME encodage camembert.
La seule difference entre deux colonnes est la selection des k phrases.

TROIS TESTS, DU PLUS ROBUSTE AU PLUS PUISSANT.

`test des signes` (binomial exact sur le nombre de livres ou A gagne) ne
suppose rien de la distribution des ecarts et ne regarde que leur SIGNE.
C'est le test que la documentation existante utilise deja, donc celui qui
rend les nouveaux chiffres comparables aux anciens.

`Wilcoxon` sur les differences appariees tient compte de l'AMPLITUDE des
ecarts et non seulement de leur signe. Plus puissant, au prix d'une
hypothese de symetrie de la distribution des differences.

L'intervalle de confiance bootstrap sur la difference moyenne dit de
COMBIEN les methodes different, ce qu'aucun des deux tests ne donne. Un
ecart significatif mais minuscule reste un ecart minuscule.

SENS DE LECTURE. La redondance se MINIMISE, la couverture se MAXIMISE. Le
script normalise les deux pour que « A gagne » veuille toujours dire « A
est meilleur ».

Usage :

    python -m scripts.tests_resumes
    python -m scripts.tests_resumes --limite 20     # essai rapide
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from pipeline.benchmark import mesurer_resumes_par_livre
from pipeline.config import MMR_PARAMS
from pipeline.corpus import charger_corpus


ALPHA = 0.05

# (critere, sens) ou sens vaut +1 si plus grand est meilleur.
CRITERES = [
    ("redondance", -1),
    ("couverture", +1),
]

COMPARAISONS = [
    ("mmr", "tfidf_naif", "MMR bat-il le top-k naif ?"),
    ("mmr", "textrank",   "MMR bat-il TextRank seul ?"),
]


def parser_args():
    p = argparse.ArgumentParser(
        description="Tests apparies entre methodes de resume.",
    )
    p.add_argument("--limite", type=int, default=None)
    p.add_argument("--sortie", default="resultats")
    p.add_argument("--k", type=int, default=MMR_PARAMS["k_phrases"])
    p.add_argument("--lambda", dest="lambda_", type=float,
                   default=MMR_PARAMS["lambda_default"])
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    return p.parse_args()


def bootstrap_ic(differences, n=10_000, seed=42):
    """Intervalle de confiance a 95 % de la difference moyenne, par bootstrap."""
    rng = np.random.default_rng(seed)
    tirages = rng.choice(differences, size=(n, len(differences)), replace=True)
    moyennes = tirages.mean(axis=1)
    return float(np.percentile(moyennes, 2.5)), float(np.percentile(moyennes, 97.5))


def comparer(detail, nom_a, nom_b, critere, sens, n_bootstrap):
    """Compare deux methodes sur un critere, livre par livre."""
    a = detail[detail["methode"] == nom_a].set_index("livre_slug")[critere]
    b = detail[detail["methode"] == nom_b].set_index("livre_slug")[critere]

    # Intersection : un livre saute par une methode l'est par toutes, mais
    # on ne le suppose pas.
    communs = a.index.intersection(b.index)
    a, b = a.loc[communs], b.loc[communs]

    # `sens` ramene tout a « positif = A est meilleur ».
    differences = (a.values - b.values) * sens

    n_a = int(np.sum(differences > 0))
    n_b = int(np.sum(differences < 0))
    n_nuls = int(np.sum(differences == 0))

    # Test des signes : les ex aequo n'entrent pas, comme dans McNemar.
    if n_a + n_b > 0:
        p_signes = float(binomtest(n_a, n_a + n_b, 0.5).pvalue)
    else:
        p_signes = 1.0

    if np.any(differences != 0):
        p_wilcoxon = float(wilcoxon(differences, zero_method="wilcox").pvalue)
    else:
        p_wilcoxon = 1.0

    bas, haut = bootstrap_ic(differences, n=n_bootstrap)

    return {
        "critere":     critere,
        "A":           nom_a,
        "B":           nom_b,
        "n_livres":    len(communs),
        "moyenne_A":   float(a.mean()),
        "moyenne_B":   float(b.mean()),
        "A_gagne":     n_a,
        "B_gagne":     n_b,
        "ex_aequo":    n_nuls,
        "diff_moyenne": float(differences.mean()),
        "ic_bas":      bas,
        "ic_haut":     haut,
        "p_signes":    p_signes,
        "p_wilcoxon":  p_wilcoxon,
    }


def _verdict(p, seuil):
    if p < seuil:
        return f"SIGNIFICATIF (seuil ajuste {seuil:.4f})"
    if p < ALPHA:
        return f"significatif brut, PAS apres ajustement ({seuil:.4f})"
    return "non concluant"


def afficher(resultats, seuil):
    for r in resultats:
        sens = "plus bas est meilleur" if r["critere"] == "redondance" \
            else "plus haut est meilleur"
        print(f"\n{'=' * 76}")
        print(f"{r['critere'].upper()}  ({sens})   {r['A']} contre {r['B']}")
        print(f"{'=' * 76}")
        print(f"  moyennes   {r['moyenne_A']:.4f}  contre  {r['moyenne_B']:.4f}")
        print(f"  {r['A']} gagne sur {r['A_gagne']}/{r['n_livres']} livres, "
              f"{r['B']} sur {r['B_gagne']}, {r['ex_aequo']} ex aequo")
        print()
        print(f"  difference moyenne (positif = {r['A']} meilleur)")
        print(f"    {r['diff_moyenne']:+.4f}   "
              f"IC 95 % [{r['ic_bas']:+.4f}, {r['ic_haut']:+.4f}]")
        contient_zero = r["ic_bas"] <= 0 <= r["ic_haut"]
        print(f"    l'intervalle {'CONTIENT' if contient_zero else 'exclut'} zero")
        print()
        print(f"  test des signes  p = {r['p_signes']:.2e}   -> "
              f"{_verdict(r['p_signes'], seuil)}")
        print(f"  Wilcoxon         p = {r['p_wilcoxon']:.2e}   -> "
              f"{_verdict(r['p_wilcoxon'], seuil)}")


def main():
    args = parser_args()
    n_tests = len(COMPARAISONS) * len(CRITERES)
    seuil = ALPHA / n_tests

    corpus = charger_corpus(with_df=True, limite=args.limite)

    print(f"\n--- Mesures par livre (k={args.k}, lambda={args.lambda_}) ---",
          flush=True)
    t0 = time.time()
    detail = mesurer_resumes_par_livre(
        corpus, k=args.k, lambda_=args.lambda_, verbose=True,
    )
    print(f"  {len(detail)} mesures en {time.time() - t0:.0f}s")

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    detail.to_csv(sortie / "resumes_par_livre.csv", index=False)

    print(f"\nSeuil {ALPHA}, ajuste a {seuil:.4f} pour {n_tests} tests "
          f"(Bonferroni)")

    resultats = [
        comparer(detail, a, b, critere, sens, args.n_bootstrap)
        for a, b, _ in COMPARAISONS
        for critere, sens in CRITERES
    ]
    afficher(resultats, seuil)

    pd.DataFrame(resultats).to_csv(sortie / "tests_resumes.csv", index=False)
    print(f"\n-> {sortie / 'tests_resumes.csv'}")
    print(f"-> {sortie / 'resumes_par_livre.csv'}")


if __name__ == "__main__":
    main()
