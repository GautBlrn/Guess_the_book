"""
Orchestration des experiences d'evaluation.

`pipeline/evaluation.py` contient les metriques (fonctions pures, aucun
acces reseau). Ce module-ci fait le sale boulot autour : construire le jeu
de test, boucler sur les configurations, empiler les resultats en
DataFrame. C'est lui qui produit les tableaux du rapport (5.5.2) et les
donnees des graphiques (5.5.1).

Il prend toujours le `corpus` en argument plutot que de le charger
lui-meme : c'est l'appelant (notebook ou script) qui fait
`charger_corpus()`. Du coup les boucles restent testables avec un corpus
synthetique, sans credentials S3.

Deux familles d'experiences, une par tache :

1. `evaluer_recherche` -- identifier un livre depuis un extrait.
   On tire des extraits contigus dans les livres du corpus, on les
   reclasse contre le corpus entier, et on regarde a quel rang le bon
   livre remonte.

2. `evaluer_resumes` -- comparer les methodes de resume extractif.
   TF-IDF naif / TextRank / MMR, compares en ROUGE contre le texte
   integral, plus les metriques de redondance qui sont les seules a
   vraiment separer MMR d'un simple top-k.

BIAIS A ASSUMER DANS LE RAPPORT : les extraits sont tires des memes
livres que ceux indexes, donc l'extrait est litteralement contenu dans le
document cible. Les scores sont mecaniquement hauts et ne se transposent
pas a un extrait d'un livre absent du corpus. C'est voulu -- c'est
exactement l'usage de l'app -- mais ca doit etre ecrit noir sur blanc,
sinon un 99 % top-1 se lit comme une performance alors que c'est en
partie une tautologie.
"""
from __future__ import annotations

import random
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipeline.config import (
    BENCH_PARAMS,
    MMR_PARAMS,
    TFIDF_CONFIGS,
    TFIDF_PARAMS,
)
from pipeline.evaluation import (
    agreger_scores,
    rang_du_bon_livre,
    resumer_metriques_recherche,
    rouge_complet,
)
from pipeline.representations import (
    TfIdfMaison,
    similarites_cosinus,
    similarites_euclidiennes,
    similarites_jaccard,
)
from pipeline.summarization import (
    centralite_textrank,
    extraire_phrases,
    resumer_livre,
    similarite_au_centre,
    vectoriser_phrases_tfidf,
)


# ============================================================================
# 1. Jeu de test pour la recherche
# ============================================================================


def tirer_extraits(
    corpus: Sequence[dict],
    n_par_livre: int | None = None,
    taille: int | tuple[int, int] | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Tire des extraits contigus dans chaque livre du corpus.

    Renvoie une liste de dicts `{livre_slug, livre, auteur, i_debut,
    taille, lemmes, tokens}`. Les deux champs de termes sont extraits a
    la meme position : `extraire_termes` applique le meme masque pour
    `lemma` et `text`, donc les deux sequences sont alignees index par
    index et un seul `i_debut` suffit pour les deux.

    L'extrait est un segment CONTIGU, pas un echantillon de mots au
    hasard : c'est ce que fait un vrai utilisateur qui copie un passage,
    et c'est nettement plus difficile qu'un sac de mots tire uniformement
    sur tout le livre, qui verrait beaucoup plus du vocabulaire global.

    `taille` accepte deux formes :

    - un `int` : tous les extraits font cette longueur. Utile pour isoler
      l'effet de la longueur (courbe de difficulte), pas pour estimer la
      performance vecue.
    - un couple `(min, max)` : la longueur est TIREE pour chaque extrait,
      uniformement dans la plage. C'est le defaut, et c'est le protocole
      realiste : un utilisateur colle ce qu'il a sous la main, il ne
      compte pas ses mots. Evaluer a longueur fixe repond a la question
      "quelle performance pour un extrait de N termes", qui n'est pas
      celle qu'on se pose.

    Pourquoi ca compte : a 200 termes filtres (~573 mots bruts, plus que
    le maximum de 300 que propose l'app), le top-1 est de 98.5 % et
    Rappel@3 sature a 1.0. A 28 termes (~80 mots, le defaut de l'app), il
    tombe a 85.4 %. Le meme systeme, mesure sur deux protocoles, donne
    treize points d'ecart.
    """
    if n_par_livre is None:
        n_par_livre = BENCH_PARAMS["n_extraits_par_livre"]
    if taille is None:
        taille = BENCH_PARAMS["taille_extrait"]
    if seed is None:
        seed = BENCH_PARAMS["seed"]

    if isinstance(taille, (tuple, list)):
        taille_min, taille_max = taille
    else:
        taille_min = taille_max = taille
    if taille_min < 1 or taille_max < taille_min:
        raise ValueError(f"plage de taille invalide : {taille!r}")

    rng = random.Random(seed)
    extraits: list[dict] = []
    for livre in corpus:
        lemmes, tokens = livre["lemmes"], livre["tokens"]
        if len(lemmes) < taille_min:
            continue
        for _ in range(n_par_livre):
            # Borne par la longueur du livre : un livre court fournit des
            # extraits plus courts plutot que d'etre exclu du jeu de test.
            t = rng.randint(taille_min, min(taille_max, len(lemmes)))
            i = rng.randint(0, len(lemmes) - t)
            extraits.append({
                "livre_slug": livre["livre_slug"],
                "livre":      livre["livre"],
                "auteur":     livre["auteur"],
                "i_debut":    i,
                "taille":     t,
                "lemmes":     lemmes[i:i + t],
                "tokens":     tokens[i:i + t],
            })
    return extraits


# ============================================================================
# 2. Experiences de recherche
# ============================================================================
#
# Trois metriques, deux familles. Cosinus et euclidien travaillent sur les
# vecteurs TF-IDF ; Jaccard travaille sur des ensembles et ignore donc
# completement les poids IDF.
#
# COSINUS ET EUCLIDIEN DONNENT LE MEME CLASSEMENT, toujours. Les vecteurs
# sont L2-normalises par `TfIdfMaison`, donc :
#
#     ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 2 - 2 cos(u, v)
#
# La distance est une fonction strictement decroissante du cosinus, et
# `similarites_euclidiennes` renvoie 1/(1+d), decroissante elle aussi :
# l'ordre est identique. Sur le corpus reel les deux lignes du tableau
# sont egales a la sixieme decimale, ce n'est pas une coincidence.
#
# On les calcule quand meme (le sujet demande les trois, et l'app les
# expose), mais le rapport doit dire qu'elles ne se "confirment" pas
# mutuellement : c'est la meme mesure ecrite deux fois. Seul Jaccard
# apporte une information differente, et il perd nettement (MRR 0.85
# contre 0.98), ce qui est attendu puisqu'il jette les poids IDF et les
# frequences.
#
# Seule exception theorique : une ligne entierement nulle reste de norme
# 0 apres normalisation (`_l2_normaliser` preserve les lignes nulles), et
# l'equivalence ne tient plus pour elle. En pratique un livre au vecteur
# nul n'a aucun terme du vocabulaire, ca ne se produit pas.
#
# Pour que la comparaison soit honnete, on construit les ensembles Jaccard
# a partir des colonnes non nulles des memes matrices TF-IDF, et pas des
# termes bruts : les trois metriques voient ainsi exactement le meme espace
# de features (meme vocabulaire, meme filtrage min_df / max_df, memes
# n-grammes). Sinon on comparerait autant les espaces que les metriques.

METRIQUES = ("cosinus", "euclidien", "jaccard")


def _ensembles_non_nuls(matrice) -> list[set[int]]:
    """
    Ensemble des indices de features actives, ligne par ligne.

    Sur une csr, les indices actifs d'une ligne sont deja stockes tels quels
    (`indices` entre deux bornes de `indptr`) : on les lit, au lieu de
    balayer les V colonnes de chaque ligne comme le ferait `flatnonzero`.
    A 300 livres en bigrammes ce balayage porterait sur 2,5 milliards de
    cellules dont 99,6 % de zeros.
    """
    if sp.issparse(matrice):
        csr = matrice.tocsr()
        return [
            set(csr.indices[csr.indptr[i]:csr.indptr[i + 1]].tolist())
            for i in range(csr.shape[0])
        ]
    return [set(np.flatnonzero(ligne).tolist()) for ligne in matrice]


def _classer(
    matrice_corpus,
    ensembles_corpus: list[set[int]],
    requete,
    ensemble_requete: set[int],
    metrique: str,
) -> np.ndarray:
    """Scores de similarite du corpus a une requete, selon la metrique."""
    if metrique == "cosinus":
        return similarites_cosinus(matrice_corpus, requete)
    if metrique == "euclidien":
        return similarites_euclidiennes(matrice_corpus, requete)
    if metrique == "jaccard":
        return similarites_jaccard(ensembles_corpus, ensemble_requete)
    raise ValueError(f"metrique inconnue : {metrique!r}")


def evaluer_recherche(
    corpus: Sequence[dict],
    extraits: Sequence[dict],
    configs: dict[str, dict] | None = None,
    metriques: Iterable[str] = METRIQUES,
    ks: Iterable[int] = (1, 3, 5, 10),
    params_tfidf: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Croise representations x metriques et renvoie une ligne par couple.

    Une ligne = une configuration complete, avec MRR, rang median et
    Rappel@k / Precision@k pour chaque k. C'est le tableau de suivi des
    experiences du 5.5.2, directement exportable.

    Le TF-IDF est ajuste UNE FOIS par representation puis reutilise pour
    toutes les metriques : refaire le fit par metrique serait du calcul
    jete, et surtout ferait varier le vocabulaire entre deux lignes du
    tableau qui sont censees ne differer que par la metrique.

    Les extraits sont vectorises avec `transform` et non `fit_transform` :
    ils ne doivent pas participer au calcul de l'IDF, sinon l'extrait
    influence la representation du corpus contre lequel on le compare.
    """
    if configs is None:
        configs = TFIDF_CONFIGS
    if params_tfidf is None:
        params_tfidf = TFIDF_PARAMS
    if not extraits:
        raise ValueError("Jeu de test vide : appelle `tirer_extraits` d'abord.")

    # Materialises : les deux sont reparcourus a chaque config, un
    # generateur serait vide des la deuxieme representation et le tableau
    # sortirait silencieusement incomplet.
    metriques = list(metriques)
    ks = list(ks)

    slugs_corpus = [c["livre_slug"] for c in corpus]
    lignes = []

    for nom_config, cfg in configs.items():
        t0 = time.time()
        champ = cfg["champ"]

        vec = TfIdfMaison(
            ngram_max=cfg["ngram_max"],
            min_df=params_tfidf["min_df"],
            max_df_ratio=params_tfidf["max_df_ratio"],
        )
        matrice = vec.fit_transform([c[champ] for c in corpus])
        requetes = vec.transform([e[champ] for e in extraits])

        ensembles_corpus = _ensembles_non_nuls(matrice)
        ensembles_requetes = _ensembles_non_nuls(requetes)

        for metrique in metriques:
            rangs: list[int | None] = []
            for i, extrait in enumerate(extraits):
                scores = _classer(
                    matrice, ensembles_corpus,
                    requetes[i], ensembles_requetes[i],
                    metrique,
                )
                # argsort decroissant, stable pour que deux scores egaux
                # soient departages par l'ordre du corpus et pas par un
                # detail d'implementation de numpy.
                ordre = np.argsort(-scores, kind="stable")
                classement = [slugs_corpus[j] for j in ordre]
                rangs.append(rang_du_bon_livre(classement, extrait["livre_slug"]))

            ligne = {
                "representation": nom_config,
                "champ":          champ,
                "ngram_max":      cfg["ngram_max"],
                "metrique":       metrique,
                **resumer_metriques_recherche(rangs, ks=ks),
            }
            lignes.append(ligne)

        if verbose:
            print(
                f"  {nom_config:12s} vocabulaire {len(vec.vocabulaire_):>7,} "
                f"termes, {len(extraits)} requetes x {len(metriques)} "
                f"metriques en {time.time() - t0:.1f}s"
            )

    return pd.DataFrame(lignes)


def courbes_rappel_recherche(
    corpus: Sequence[dict],
    extraits: Sequence[dict],
    configs: dict[str, dict] | None = None,
    metrique: str = "cosinus",
    k_max: int = 10,
    params_tfidf: dict | None = None,
) -> pd.DataFrame:
    """
    Rappel@k pour k = 1..k_max, une ligne par (representation, k).

    Format long, directement consommable par un `sns.lineplot(x="k",
    y="rappel", hue="representation")` : c'est le graphique n°1 du sujet.

    On reappelle `evaluer_recherche` avec `ks=range(1, k_max+1)` plutot
    que de refaire la boucle : une seule implementation du classement,
    donc pas de risque que la courbe et le tableau racontent deux
    histoires differentes.
    """
    table = evaluer_recherche(
        corpus, extraits,
        configs=configs, metriques=[metrique],
        ks=range(1, k_max + 1),
        params_tfidf=params_tfidf, verbose=False,
    )
    lignes = []
    for _, row in table.iterrows():
        for k in range(1, k_max + 1):
            lignes.append({
                "representation": row["representation"],
                "metrique":       metrique,
                "k":              k,
                "rappel":         row[f"rappel@{k}"],
                "precision":      row[f"precision@{k}"],
            })
    return pd.DataFrame(lignes)


# ============================================================================
# 3. Experiences de resume
# ============================================================================
#
# ATTENTION A CE QU'ON MESURE. Le corpus n'a pas de resumes de reference
# ecrits par des humains, donc il n'y a pas de "gold standard". On compare
# chaque resume au TEXTE INTEGRAL du livre, ce qui change l'interpretation :
#
#   - la PRECISION vaut ~1 pour TOUTE methode extractive, par
#     construction : chaque phrase du resume est copiee du livre, donc
#     chacun de ses n-grammes s'y trouve (seuls les n-grammes a cheval
#     sur la jointure de deux phrases selectionnees manquent) ;
#   - le RAPPEL est domine par la LONGUEUR du resume : le clipping compte
#     min(occurrences), donc a k fixe toutes les methodes obtiennent
#     quasiment la meme valeur ;
#   - donc le F1 aussi.
#
# Conclusion a ne pas contourner : ROUGE contre le texte integral NE
# CLASSE PAS les methodes extractives. Sur un corpus de test, les trois
# methodes sortent des ROUGE identiques a la troisieme decimale. C'est
# une propriete du montage, pas un bug, et un tableau qui ne montrerait
# que ROUGE ne dirait strictement rien.
#
# Ce qui discrimine reellement, et que `evaluer_resumes` sort donc aussi :
#
#   - REDONDANCE : cosinus moyen entre phrases retenues. C'est ce que MMR
#     optimise explicitement.
#   - COUVERTURE : part de la masse lexicale du livre touchee. A longueur
#     egale, un resume diversifie en couvre plus qu'un empilement de
#     phrases qui se ressemblent.
#
# ROUGE reste calcule parce que le sujet le demande (5.5.1) et parce
# qu'il sert de garde-fou (un effondrement signalerait un resume casse),
# mais le rapport doit dire pourquoi ce n'est pas lui qui tranche.

METHODES_RESUME = ("tfidf_naif", "textrank", "mmr")


def _selection_top_k(phrases: Sequence[dict], scores: np.ndarray, k: int) -> list[dict]:
    """Les k phrases de meilleur score, annotees comme `resumer_livre` le fait."""
    indices = np.argsort(-scores, kind="stable")[:k]
    selection = []
    for rang, i in enumerate(indices, 1):
        p = dict(phrases[int(i)])
        p["rang_mmr"] = rang
        p["score"] = float(scores[int(i)])
        selection.append(p)
    return selection


def preparer_pool(
    df: pd.DataFrame,
    champ_termes: str = "lemma",
    encodeur: Callable[[list[str]], np.ndarray] | None = None,
) -> tuple[list[dict], np.ndarray | None]:
    """
    Phrases candidates et leurs embeddings, calcules une seule fois.

    Les embeddings camembert sont de loin le poste dominant du benchmark
    (plusieurs secondes par livre contre quelques millisecondes pour tout
    le reste). Les calculer une fois et les faire circuler est ce qui rend
    le balayage de lambda praticable.

    `encodeur` : injectable pour les tests, evite de charger camembert
    (~440 Mo) et torch. Par defaut, l'encodeur de `summarization`.
    """
    phrases = extraire_phrases(df, champ_termes=champ_termes)
    if not phrases:
        return [], None
    if encodeur is None:
        from pipeline.summarization import _encoder_phrases as encodeur
    return phrases, encodeur([p["texte"] for p in phrases])


def resumes_par_methode(
    df: pd.DataFrame,
    methodes: Iterable[str] = METHODES_RESUME,
    k: int | None = None,
    lambda_: float | None = None,
    champ_termes: str = "lemma",
    encodeur: Callable[[list[str]], np.ndarray] | None = None,
    pool: tuple[list[dict], np.ndarray | None] | None = None,
    poids: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    """
    Produit le resume d'un livre par chaque methode, sur le MEME pool.

    Les trois methodes partent des memes phrases candidates et des memes
    embeddings : l'ecart mesure vient donc de la selection, pas d'un pool
    de depart different.

    L'arm `mmr` passe par `resumer_livre`, la vraie fonction de
    production, plutot que par une reimplementation locale : le benchmark
    doit mesurer ce qui est servi, pas une copie qui derivera. Le prix a
    payer est un `extraire_phrases` refait en interne par `resumer_livre`,
    ce qui est negligeable a cote des embeddings et vaut la garantie.

    `pool` : sortie de `preparer_pool`, pour partager les embeddings entre
    plusieurs appels (comparaison de methodes, balayage de lambda).

    `poids` : ponderation du score hybride, transmise a `resumer_livre`.
    N'affecte que l'arm `mmr` : `tfidf_naif` et `textrank` sont par
    definition des composantes isolees, les ponderer n'aurait pas de sens.
    """
    if k is None:
        k = MMR_PARAMS["k_phrases"]
    if lambda_ is None:
        lambda_ = MMR_PARAMS["lambda_default"]
    methodes = list(methodes)

    if pool is None:
        pool = preparer_pool(df, champ_termes=champ_termes, encodeur=encodeur)
    phrases, embeddings = pool

    if len(phrases) < k:
        # Livre trop court ou filtres trop stricts : toutes les methodes
        # renverraient la meme chose, la comparaison n'a pas d'objet.
        return {m: [] for m in methodes}

    sorties: dict[str, list[dict]] = {}
    for methode in methodes:
        if methode == "tfidf_naif":
            matrice = vectoriser_phrases_tfidf(
                phrases,
                ngram_max=MMR_PARAMS["ngram_max"],
                min_df=MMR_PARAMS["min_df"],
            )
            sorties[methode] = _selection_top_k(
                phrases, similarite_au_centre(matrice), k
            )
        elif methode == "textrank":
            sorties[methode] = _selection_top_k(
                phrases, centralite_textrank(embeddings), k
            )
        elif methode == "mmr":
            sorties[methode] = resumer_livre(
                df, champ_termes=champ_termes, k=k, lambda_=lambda_,
                embeddings_pre=embeddings, poids=poids,
            )
        else:
            raise ValueError(f"methode inconnue : {methode!r}")
    return sorties


def redondance(
    selection: Sequence[dict],
    phrases: Sequence[dict],
    embeddings: np.ndarray,
) -> float:
    """
    Cosinus moyen entre paires de phrases selectionnees.

    Plus c'est bas, moins le resume se repete. C'est la metrique que MMR
    optimise explicitement et que ROUGE ne voit pas : deux phrases qui
    disent la meme chose avec des mots differents comptent double en
    ROUGE et sont penalisees ici.

    Les embeddings sont L2-normalises par `_encoder_phrases`, donc le
    produit scalaire est directement le cosinus.

    Le raccordement selection -> embedding passe par `sent_id` et pas par
    la position dans la liste : `resumer_livre` reconstruit ses propres
    dicts de phrases (`dict(phrases[i])` apres son propre
    `extraire_phrases`), donc toute cle qu'on aurait ajoutee au pool ne
    survivrait pas, et un index positionnel ne serait pas comparable.
    `sent_id` vient du DataFrame annote et est stable.
    """
    position = {p["sent_id"]: i for i, p in enumerate(phrases)}
    manquants = [p["sent_id"] for p in selection if p["sent_id"] not in position]
    if manquants:
        # Signifie que la selection vient d'un autre pool que celui passe
        # en argument (filtres d'extraction differents). Mieux vaut lever
        # que renvoyer une redondance calculee sur une partie des phrases.
        raise KeyError(
            f"{len(manquants)} phrase(s) selectionnee(s) absentes du pool "
            f"(sent_id {manquants[:5]}...) : pool et selection incoherents."
        )
    if len(selection) < 2:
        return 0.0
    indices = [position[p["sent_id"]] for p in selection]
    sous_matrice = embeddings[indices]
    sims = sous_matrice @ sous_matrice.T
    iu = np.triu_indices_from(sims, k=1)
    return float(np.mean(sims[iu]))


def couverture(selection: Sequence[dict], reference: Sequence[str]) -> float:
    """
    Part de la masse lexicale du livre touchee par le resume.

    Proportion des tokens du livre dont le terme apparait au moins une
    fois dans le resume. Ponderee par frequence, donc couvrir un mot
    frequent compte plus que couvrir un hapax.

    C'est la metrique qui manque a ROUGE dans ce montage. Contre le texte
    integral, le rappel ROUGE est domine par la LONGUEUR du resume (le
    clipping compte min(occurrences), donc a k fixe toutes les methodes
    obtiennent quasiment le meme), alors que la couverture mesure la
    DIVERSITE du vocabulaire retenu : deux resumes de meme longueur
    peuvent couvrir des parts tres differentes du livre. C'est ce que MMR
    est cense ameliorer, et ce qu'un top-k naif degrade en empilant des
    phrases qui se ressemblent.
    """
    if not reference:
        return 0.0
    termes_resume = {t for p in selection for t in p["termes"]}
    if not termes_resume:
        return 0.0
    return sum(1 for t in reference if t in termes_resume) / len(reference)


def _mesurer(
    selection: Sequence[dict],
    phrases: Sequence[dict],
    embeddings: np.ndarray,
    reference: Sequence[str],
    avec_l: bool,
) -> tuple[dict[str, float], float, float]:
    """Scores ROUGE, redondance et couverture. Partage entre les deux boucles."""
    candidat = [t for p in selection for t in p["termes"]]
    return (
        rouge_complet(candidat, reference, avec_l=avec_l),
        redondance(selection, phrases, embeddings),
        couverture(selection, reference),
    )


def _agreger(
    scores: Sequence[dict],
    redondances: Sequence[float],
    couvertures: Sequence[float],
) -> dict[str, float]:
    """Moyennes ROUGE + intrinseques, mises en forme pour une ligne de DataFrame."""
    return {
        **agreger_scores(scores),
        "redondance":            float(np.mean(redondances)),
        "redondance_ecart_type": float(np.std(redondances)),
        "couverture":            float(np.mean(couvertures)),
        "couverture_ecart_type": float(np.std(couvertures)),
    }


def evaluer_resumes(
    corpus: Sequence[dict],
    methodes: Iterable[str] = METHODES_RESUME,
    k: int | None = None,
    lambda_: float | None = None,
    avec_l: bool = False,
    encodeur: Callable[[list[str]], np.ndarray] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare les methodes de resume sur tout le corpus.

    Une ligne par methode : moyennes et ecarts-types ROUGE sur les livres,
    plus la redondance moyenne. Lire le docstring de section ci-dessus
    avant d'interpreter les chiffres, la reference est le texte integral
    et pas un resume de reference.

    `avec_l=False` par defaut, et ce n'est pas de la frilosite : la LCS
    est en O(len(candidat) x len(reference)), soit ~300 x 80 000 = 24
    millions d'operations en Python pur PAR LIVRE. Sur un corpus de 20
    livres ca fait passer le benchmark de quelques secondes a plusieurs
    minutes, pour une metrique dont le rappel est de toute facon
    ininterpretable ici.
    """
    if k is None:
        k = MMR_PARAMS["k_phrases"]
    if lambda_ is None:
        lambda_ = MMR_PARAMS["lambda_default"]
    methodes = list(methodes)

    detail = mesurer_resumes_par_livre(
        corpus, methodes=methodes, k=k, lambda_=lambda_,
        avec_l=avec_l, encodeur=encodeur, verbose=verbose,
    )
    return agreger_resumes(detail, methodes, k, lambda_)


def mesurer_resumes_par_livre(
    corpus: Sequence[dict],
    methodes: Iterable[str] = METHODES_RESUME,
    k: int | None = None,
    lambda_: float | None = None,
    avec_l: bool = False,
    encodeur: Callable[[list[str]], np.ndarray] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Mesures BRUTES, une ligne par (livre, methode).

    Separee de l'agregation parce que les moyennes ne permettent pas de
    comparer deux methodes. Chaque livre est evalue par toutes les methodes
    sur le MEME pool de phrases et le MEME encodage : les mesures sont donc
    appariees, et un test des signes ou un Wilcoxon sur les differences
    livre par livre est bien plus informatif que l'ecart des moyennes.

    C'est exactement ce qui manquait pour trancher MMR contre `tfidf_naif` :
    les deux moyennes se touchaient, sans qu'on puisse dire si l'ecart tenait
    a une superiorite reguliere ou a quelques livres atypiques.
    """
    if k is None:
        k = MMR_PARAMS["k_phrases"]
    if lambda_ is None:
        lambda_ = MMR_PARAMS["lambda_default"]
    methodes = list(methodes)

    lignes: list[dict] = []
    n_sautes = 0

    for livre in corpus:
        t0 = time.time()
        reference = livre["lemmes"]

        # Un seul encodage par livre, partage entre les methodes et
        # reutilise pour la redondance.
        phrases, embeddings = preparer_pool(livre["df"], encodeur=encodeur)
        if len(phrases) < k:
            n_sautes += 1
            if verbose:
                print(f"  {livre['livre'][:40]:40s} saute "
                      f"({len(phrases)} phrases candidates < k={k})")
            continue

        resumes = resumes_par_methode(
            livre["df"], methodes=methodes, k=k, lambda_=lambda_,
            pool=(phrases, embeddings),
        )

        for methode, selection in resumes.items():
            scores, redond, couv = _mesurer(
                selection, phrases, embeddings, reference, avec_l
            )
            lignes.append({
                "livre_slug":  livre["livre_slug"],
                "livre":       livre["livre"],
                "genre":       livre["genre"],
                "methode":     methode,
                "redondance":  redond,
                "couverture":  couv,
                "n_phrases_candidates": len(phrases),
                **scores,
            })

        if verbose:
            print(f"  {livre['livre'][:40]:40s} {time.time() - t0:5.1f}s")

    if verbose and n_sautes:
        print(f"  {n_sautes} livre(s) saute(s), trop peu de phrases candidates.")

    return pd.DataFrame(lignes)


def agreger_resumes(detail, methodes, k, lambda_) -> pd.DataFrame:
    """Moyennes et ecarts-types par methode, depuis les mesures brutes."""
    if detail.empty:
        return pd.DataFrame([])

    colonnes_scores = [
        c for c in detail.columns
        if c.startswith(("rouge1", "rouge2", "rougeL"))
    ]

    lignes = []
    for methode in methodes:
        sous = detail[detail["methode"] == methode]
        if sous.empty:
            continue
        lignes.append({
            "methode": methode,
            "k":       k,
            "lambda":  lambda_ if methode == "mmr" else None,
            **_agreger(
                sous[colonnes_scores].to_dict("records"),
                sous["redondance"].tolist(),
                sous["couverture"].tolist(),
            ),
        })
    return pd.DataFrame(lignes)


def balayer_lambda(
    corpus: Sequence[dict],
    lambdas: Sequence[float] = (0.3, 0.5, 0.6, 0.8, 1.0),
    k: int | None = None,
    avec_l: bool = False,
    encodeur: Callable[[list[str]], np.ndarray] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Effet de lambda sur le compromis pertinence / diversite, en MMR seul.

    `lambda_=1.0` desactive le terme de diversite : MMR degenere alors en
    top-k sur le score hybride. C'est le point de comparaison qui rend le
    balayage lisible, il doit maximiser ROUGE et la redondance a la fois.
    Justifie le `lambda_default` de la config autrement qu'a l'oeil.

    La boucle est livre-par-livre et pas lambda-par-lambda : le pool
    d'embeddings d'un livre est calcule une fois et reutilise pour tous
    les lambdas. Dans l'autre sens on paierait le camembert autant de
    fois qu'il y a de valeurs a tester, soit l'essentiel du cout total
    pour un resultat identique.
    """
    if k is None:
        k = MMR_PARAMS["k_phrases"]
    lambdas = list(lambdas)

    scores_par_lambda: dict[float, list[dict]] = {lam: [] for lam in lambdas}
    redondances: dict[float, list[float]] = {lam: [] for lam in lambdas}
    couvertures: dict[float, list[float]] = {lam: [] for lam in lambdas}

    for livre in corpus:
        phrases, embeddings = preparer_pool(livre["df"], encodeur=encodeur)
        if len(phrases) < k:
            continue
        if verbose:
            print(f"  {livre['livre'][:40]:40s} {len(phrases)} phrases")

        for lam in lambdas:
            selection = resumes_par_methode(
                livre["df"], methodes=["mmr"], k=k, lambda_=lam,
                pool=(phrases, embeddings),
            )["mmr"]
            scores, redond, couv = _mesurer(
                selection, phrases, embeddings, livre["lemmes"], avec_l
            )
            scores_par_lambda[lam].append(scores)
            redondances[lam].append(redond)
            couvertures[lam].append(couv)

    lignes = [
        {
            "methode": "mmr",
            "k":       k,
            "lambda":  lam,
            **_agreger(scores_par_lambda[lam], redondances[lam], couvertures[lam]),
        }
        for lam in lambdas
        if scores_par_lambda[lam]
    ]
    return pd.DataFrame(lignes)
