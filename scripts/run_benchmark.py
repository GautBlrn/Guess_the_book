"""
Lance les experiences d'evaluation et exporte les tableaux en CSV.

Pilote `pipeline/benchmark.py` depuis la ligne de commande, pour produire
les tableaux du rapport sans ouvrir un notebook. Les CSV sortent dans
`resultats/` (ignore par git, c'est regenerable).

Usage :

    python -m scripts.run_benchmark                     # tout
    python -m scripts.run_benchmark --skip-resumes      # recherche seule
    python -m scripts.run_benchmark --balayer-lambda    # + effet de lambda
    python -m scripts.run_benchmark --limite 3          # essai rapide

DEUX REGIMES DE COUT tres differents :

- La partie RECHERCHE ne touche ni camembert ni torch. Quelques secondes
  sur un corpus de 20 livres. C'est `--skip-resumes`.
- La partie RESUME encode chaque phrase candidate avec camembert : au
  premier lancement elle telecharge ~440 Mo, puis compte quelques
  secondes par livre sur CPU. `--limite N` sert a verifier que la chaine
  tourne avant de lancer le corpus entier.

Fichiers produits :

    resultats/recherche.csv          - une ligne par (representation, metrique)
    resultats/recherche_courbes.csv  - format long, Rappel@k pour k = 1..10
    resultats/resumes.csv            - une ligne par methode de resume
    resultats/lambda.csv             - une ligne par valeur de lambda
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.benchmark import (
    balayer_lambda,
    courbes_rappel_recherche,
    evaluer_recherche,
    evaluer_resumes,
    tirer_extraits,
)
from pipeline.config import BENCH_PARAMS, MMR_PARAMS
from pipeline.corpus import charger_corpus


# --- CLI ---

def parser_args():
    p = argparse.ArgumentParser(
        description="Lance les experiences d'evaluation et exporte les CSV.",
    )
    p.add_argument("--sortie", default="resultats",
                   help="Repertoire des CSV (defaut: resultats)")
    p.add_argument("--limite", type=int,
                   help="N'evalue que les N premiers livres. Pour verifier "
                        "que la chaine tourne avant le corpus entier.")

    p.add_argument("--skip-recherche", action="store_true")
    p.add_argument("--skip-resumes",   action="store_true",
                   help="Saute la partie resume, donc aucun chargement de "
                        "camembert. C'est le mode rapide.")
    p.add_argument("--balayer-lambda", action="store_true",
                   help="Ajoute le balayage de lambda (couteux : un resume "
                        "complet du corpus par valeur testee).")

    p.add_argument("--n-extraits", type=int,
                   default=BENCH_PARAMS["n_extraits_par_livre"],
                   help=f"Extraits tires par livre "
                        f"(defaut: {BENCH_PARAMS['n_extraits_par_livre']})")
    p.add_argument("--taille-extrait", type=int,
                   default=BENCH_PARAMS["taille_extrait"],
                   help=f"Termes par extrait "
                        f"(defaut: {BENCH_PARAMS['taille_extrait']})")
    p.add_argument("--seed", type=int, default=BENCH_PARAMS["seed"],
                   help=f"Seed du tirage (defaut: {BENCH_PARAMS['seed']})")
    p.add_argument("--k-max", type=int, default=10,
                   help="k maximum des courbes de rappel (defaut: 10)")

    p.add_argument("--k", type=int, default=MMR_PARAMS["k_phrases"],
                   help=f"Phrases par resume (defaut: {MMR_PARAMS['k_phrases']})")
    p.add_argument("--lambda", dest="lambda_", type=float,
                   default=MMR_PARAMS["lambda_default"],
                   help=f"Lambda MMR (defaut: {MMR_PARAMS['lambda_default']})")
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.3, 0.5, 0.6, 0.8, 1.0],
                   help="Valeurs balayees par --balayer-lambda")
    p.add_argument("--avec-l", action="store_true",
                   help="Calcule aussi ROUGE-L. La LCS contre le texte "
                        "integral est en O(n x m) : compte plusieurs minutes "
                        "par livre, pour une metrique peu informative ici.")
    return p.parse_args()


# --- Export ---

def exporter(table, repertoire, nom):
    """Ecrit un DataFrame en CSV et l'affiche."""
    repertoire.mkdir(parents=True, exist_ok=True)
    chemin = repertoire / f"{nom}.csv"
    table.to_csv(chemin, index=False)
    print(table.to_string(index=False))
    print(f"  -> {chemin}\n")


# --- Main ---

def main():
    args = parser_args()
    sortie = Path(args.sortie)

    # Pas de `storage.ensure_bucket()` ici, contrairement a
    # rebuild_artifacts : ce script ne fait que LIRE le corpus et ecrire
    # des CSV en local. `ensure_bucket` appelle `create_bucket`, donc
    # exiger ce droit ferait echouer le benchmark sur une cle en lecture
    # seule alors qu'il n'ecrit rien sur S3.
    corpus = charger_corpus()
    if args.limite:
        corpus = corpus[:args.limite]
        print(f"Limite a {len(corpus)} livre(s).")
    if not corpus:
        print("Corpus vide, rien a evaluer.")
        return 1

    if not args.skip_recherche:
        print("\n--- Recherche ---")
        t0 = time.time()
        extraits = tirer_extraits(
            corpus,
            n_par_livre=args.n_extraits,
            taille=args.taille_extrait,
            seed=args.seed,
        )
        if not extraits:
            # Arrive si tous les livres font moins de `taille_extrait`
            # termes : mieux vaut le dire que sortir un tableau vide.
            print(f"  Aucun extrait tirable : aucun livre n'atteint "
                  f"{args.taille_extrait} termes. Baisse --taille-extrait.")
        else:
            print(f"  {len(extraits)} extraits de {args.taille_extrait} termes "
                  f"sur {len(corpus)} livres (seed {args.seed})\n")
            exporter(
                evaluer_recherche(corpus, extraits),
                sortie, "recherche",
            )
            exporter(
                courbes_rappel_recherche(corpus, extraits, k_max=args.k_max),
                sortie, "recherche_courbes",
            )
            print(f"  recherche terminee en {time.time() - t0:.1f}s\n")

    if not args.skip_resumes:
        print("\n--- Resumes ---")
        t0 = time.time()
        table = evaluer_resumes(
            corpus, k=args.k, lambda_=args.lambda_, avec_l=args.avec_l,
        )
        if table.empty:
            print(f"  Aucun livre n'a {args.k} phrases candidates. "
                  f"Baisse --k ou verifie les filtres MMR_PARAMS.")
        else:
            print()
            exporter(table, sortie, "resumes")
        print(f"  resumes termines en {time.time() - t0:.1f}s\n")

    if args.balayer_lambda:
        print("\n--- Balayage de lambda ---")
        t0 = time.time()
        table = balayer_lambda(
            corpus, lambdas=args.lambdas, k=args.k, avec_l=args.avec_l,
        )
        if table.empty:
            print(f"  Aucun livre n'a {args.k} phrases candidates.")
        else:
            exporter(table, sortie, "lambda")
        print(f"  balayage termine en {time.time() - t0:.1f}s\n")

    print("Termine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
