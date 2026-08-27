"""
Tests apparies entre representations, sur le jeu d'extraits du benchmark.

POURQUOI APPARIE. Le benchmark rend des taux agreges : 93,95 % pour
`tokens_12g` contre 92,65 % pour `lemmes_12g`. Comparer ces deux nombres
avec un test de proportions supposerait deux echantillons independants, ce
qu'ils ne sont pas : c'est le MEME jeu de 1455 extraits qui passe dans les
deux representations, et `tirer_extraits` fournit meme les deux champs du
MEME passage. Ignorer l'appariement, c'est jeter l'information la plus
utile et perdre en puissance : deux representations qui echouent sur les
memes extraits difficiles ne different pas vraiment, meme si l'ecart
agrege parait grand.

DEUX TESTS, PARCE QU'ILS NE VOIENT PAS LA MEME CHOSE.

`McNemar` (exact, via un test binomial sur les paires discordantes) ne
regarde que le succes au rang 1. Il repond a « laquelle trouve le bon
livre en premier plus souvent », et ne compte que les extraits ou les deux
representations different. C'est le test qui correspond au chiffre annonce
dans le rapport.

`Wilcoxon` sur les rangs reciproques (1/rang, la quantite que le MRR
moyenne) utilise davantage : passer du rang 9 au rang 2 compte, alors que
McNemar l'ignore tant que ce n'est pas le rang 1. Plus puissant quand les
deux representations trouvent souvent, mais pas toujours en tete.

Un desaccord entre les deux se lit : McNemar non significatif et Wilcoxon
significatif signifie que l'ecart porte sur la qualite du classement et non
sur le taux de reussite en tete.

TROIS COMPARAISONS, DONC UN AJUSTEMENT. Trois tests sur les memes donnees
multiplient les chances d'un faux positif. Le seuil de Bonferroni est
rappele en tete de sortie ; il est conservateur, mais l'alternative serait
de choisir la comparaison la plus flatteuse apres coup.

Usage :

    python -m scripts.tests_apparies
    python -m scripts.tests_apparies --limite 30      # essai rapide
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from pipeline.benchmark import tirer_extraits
from pipeline.config import BENCH_PARAMS, TFIDF_PARAMS
from pipeline.corpus import charger_corpus
from pipeline.representations import TfIdfMaison, similarites_cosinus


ALPHA = 0.05

# Les configurations comparees. On garde le cosinus seul : sur des vecteurs
# L2-normalises l'euclidien produit le meme classement, donc le tester
# reviendrait a dupliquer la ligne.
CONFIGS = {
    "lemmes_1g":         {"champ": "lemmes", "ngram_max": 1, "min_df": 1},
    "lemmes_12g":        {"champ": "lemmes", "ngram_max": 2, "min_df": 1},
    "tokens_12g":        {"champ": "tokens", "ngram_max": 2, "min_df": 1},
    "lemmes_12g_mindf2": {"champ": "lemmes", "ngram_max": 2, "min_df": 2},
}

COMPARAISONS = [
    ("tokens_12g", "lemmes_12g",
     "La config servie est-elle la bonne ?"),
    ("lemmes_12g", "lemmes_1g",
     "Les bigrammes valent-ils leur cout en vocabulaire ?"),
    ("lemmes_12g", "lemmes_12g_mindf2",
     "min_df=1 se justifie-t-il encore a 291 livres ?"),
]


def parser_args():
    p = argparse.ArgumentParser(
        description="Tests apparies entre representations TF-IDF.",
    )
    p.add_argument("--limite", type=int, default=None,
                   help="N'utiliser que les N premiers livres (essai rapide)")
    p.add_argument("--sortie", default="resultats",
                   help="Repertoire des CSV (defaut: resultats)")
    p.add_argument("--seed", type=int, default=BENCH_PARAMS["seed"])
    return p.parse_args()


def rangs_par_config(corpus, extraits, nom, cfg):
    """
    Rang du bon livre pour chaque extrait, dans une configuration donnee.

    Renvoie un tableau d'entiers 1-indexes, aligne sur `extraits`.
    """
    champ = cfg["champ"]
    vec = TfIdfMaison(
        ngram_max=cfg["ngram_max"],
        min_df=cfg["min_df"],
        max_df_ratio=TFIDF_PARAMS["max_df_ratio"],
    )

    t0 = time.time()
    X = vec.fit_transform([c[champ] for c in corpus])
    Q = vec.transform([e[champ] for e in extraits])

    slugs = [c["livre_slug"] for c in corpus]
    cible = {s: i for i, s in enumerate(slugs)}

    rangs = np.empty(len(extraits), dtype=int)
    for i, extrait in enumerate(extraits):
        scores = similarites_cosinus(X, Q[i])
        # Tri stable : deux scores egaux sont departages par l'ordre du
        # corpus, comme dans `pipeline/benchmark.py`, pour que les rangs
        # soient comparables d'une config a l'autre.
        ordre = np.argsort(-scores, kind="stable")
        rangs[i] = int(np.where(ordre == cible[extrait["livre_slug"]])[0][0]) + 1

    print(f"  {nom:20s} vocab {len(vec.vocabulaire_):>9,d} termes, "
          f"top-1 {np.mean(rangs == 1) * 100:5.2f} %, "
          f"MRR {np.mean(1 / rangs):.4f}  ({time.time() - t0:.0f}s)",
          flush=True)
    return rangs


def mcnemar_exact(succes_a, succes_b):
    """
    McNemar exact : test binomial sur les paires discordantes.

    `b` compte les extraits ou A reussit et B echoue, `c` l'inverse. Les
    paires concordantes n'entrent pas dans le test, ce qui est le principe :
    un extrait que les deux trouvent, ou que les deux ratent, n'apporte
    aucune information sur laquelle est meilleure.
    """
    b = int(np.sum(succes_a & ~succes_b))
    c = int(np.sum(~succes_a & succes_b))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def comparer(nom_a, nom_b, rangs, question):
    """Compare deux configurations et renvoie une ligne de resultat."""
    ra, rb = rangs[nom_a], rangs[nom_b]
    succes_a, succes_b = ra == 1, rb == 1

    b, c, p_mcnemar = mcnemar_exact(succes_a, succes_b)

    # Wilcoxon sur les rangs reciproques. `zero_method="wilcox"` retire les
    # paires nulles, c'est-a-dire les extraits classes au meme rang par les
    # deux configs, qui sont l'immense majorite.
    inv_a, inv_b = 1 / ra, 1 / rb
    differences = inv_a - inv_b
    if np.any(differences != 0):
        p_wilcoxon = float(wilcoxon(inv_a, inv_b, zero_method="wilcox").pvalue)
    else:
        p_wilcoxon = 1.0

    return {
        "A": nom_a,
        "B": nom_b,
        "question": question,
        "top1_A": float(np.mean(succes_a)),
        "top1_B": float(np.mean(succes_b)),
        "mrr_A": float(np.mean(inv_a)),
        "mrr_B": float(np.mean(inv_b)),
        "A_gagne": b,
        "B_gagne": c,
        "discordantes": b + c,
        "p_mcnemar": p_mcnemar,
        "p_wilcoxon": p_wilcoxon,
        "n_differents": int(np.sum(differences != 0)),
    }


def afficher(resultats, alpha_ajuste):
    for r in resultats:
        print(f"\n{'=' * 74}")
        print(f"{r['A']}  contre  {r['B']}")
        print(f"  {r['question']}")
        print(f"{'=' * 74}")
        print(f"  top-1     {r['top1_A'] * 100:6.2f} %  contre "
              f"{r['top1_B'] * 100:6.2f} %   "
              f"(ecart {(r['top1_A'] - r['top1_B']) * 100:+.2f} pts)")
        print(f"  MRR       {r['mrr_A']:6.4f}    contre {r['mrr_B']:6.4f}")
        print()
        print(f"  McNemar exact, sur le succes au rang 1")
        print(f"    {r['A']} seul gagnant : {r['A_gagne']:4d} extraits")
        print(f"    {r['B']} seul gagnant : {r['B_gagne']:4d} extraits")
        print(f"    paires discordantes  : {r['discordantes']:4d} "
              f"(les concordantes n'informent pas)")
        print(f"    p = {r['p_mcnemar']:.2e}   -> "
              f"{_verdict(r['p_mcnemar'], alpha_ajuste)}")
        print()
        print(f"  Wilcoxon, sur les rangs reciproques")
        print(f"    extraits classes differemment : {r['n_differents']:4d}")
        print(f"    p = {r['p_wilcoxon']:.2e}   -> "
              f"{_verdict(r['p_wilcoxon'], alpha_ajuste)}")


def _verdict(p, alpha_ajuste):
    if p < alpha_ajuste:
        return f"SIGNIFICATIF (seuil ajuste {alpha_ajuste:.4f})"
    if p < ALPHA:
        return (f"significatif brut, mais PAS apres ajustement "
                f"({alpha_ajuste:.4f})")
    return "non concluant"


def main():
    args = parser_args()
    alpha_ajuste = ALPHA / len(COMPARAISONS)

    corpus = charger_corpus(with_df=False, limite=args.limite)

    extraits = tirer_extraits(
        corpus,
        n_par_livre=BENCH_PARAMS["n_extraits_par_livre"],
        taille=BENCH_PARAMS["taille_extrait"],
        seed=args.seed,
    )
    print(f"\n{len(extraits)} extraits sur {len(corpus)} livres "
          f"(seed {args.seed})")
    print(f"Seuil {ALPHA}, ajuste a {alpha_ajuste:.4f} pour "
          f"{len(COMPARAISONS)} comparaisons (Bonferroni)\n")

    print("--- Rangs par configuration ---")
    rangs = {
        nom: rangs_par_config(corpus, extraits, nom, cfg)
        for nom, cfg in CONFIGS.items()
    }

    resultats = [comparer(a, b, rangs, q) for a, b, q in COMPARAISONS]
    afficher(resultats, alpha_ajuste)

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    chemin = sortie / "tests_apparies.csv"
    pd.DataFrame(resultats).to_csv(chemin, index=False)
    print(f"\n-> {chemin}")


if __name__ == "__main__":
    main()
