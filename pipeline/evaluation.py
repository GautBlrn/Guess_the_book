"""
Metriques d'evaluation : recherche (rangs) et resume (ROUGE / BLEU).

Ce module ne contient QUE des fonctions pures : pas d'acces S3, pas de
chargement de modele. Il est donc testable sans credentials et sans
telechargement. L'orchestration (construction du jeu de test, boucles
d'experiences) vit dans `pipeline/benchmark.py`.

Deux familles de metriques, une par tache du projet :

1. **Recherche** (`identifier un livre depuis un extrait`).
   L'entree est une liste de rangs : pour chaque requete, le rang auquel
   le bon livre est apparu dans le classement (1 = premier, None = absent).
   On en derive Precision@k, Rappel@k et MRR.

2. **Resume** (`comparer TF-IDF naif / TextRank / MMR`).
   ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum et BLEU, implementes a la main
   pour rester coherent avec le reste du projet (TF-IDF est deja fait
   main) et pour eviter une dependance de plus. Verifies au bit pres
   contre `rouge-score` (implementation Google) et `nltk`.

Convention de tokenisation : ces metriques operent sur des **listes de
termes deja tokenises**, jamais sur des chaines. C'est a l'appelant de
decider s'il compare des tokens bruts ou des lemmes -- et le choix change
les scores, donc il doit etre documente dans le rapport.

Deux formes d'entree cohabitent cote resume, et les confondre fausse les
scores sans lever d'erreur :

- termes **a plat** (`["le", "chat", ...]`) pour `rouge_n`, `rouge_l`,
  `rouge_complet`, `bleu` ;
- resume **decoupe en phrases** (`[["le", "chat"], ["il", "dort"]]`) pour
  `rouge_lsum` et `rouge_complet_phrases`.

`termes_par_phrase` fait le pont depuis la sortie de
`summarization.resumer_livre`, et `agreger_scores` moyenne le tout sur un
corpus de livres.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np


# ============================================================================
# 1. Metriques de recherche
# ============================================================================
#
# Cas d'usage : UN seul document pertinent par requete (l'extrait vient
# d'un livre et d'un seul). Cette hypothese simplifie tout, mais elle a une
# consequence qu'il faut assumer explicitement dans le rapport :
#
#   - Rappel@k    = 1 si le bon livre est dans le top-k, 0 sinon.
#   - Precision@k = Rappel@k / k  (au plus 1 bonne reponse parmi k)
#
# Donc Precision@k et Rappel@k portent la MEME information a un facteur k
# pres. Les afficher tous les deux est demande par le sujet, mais dire
# qu'ils "se confirment mutuellement" serait faux : ils sont redondants
# par construction. La metrique qui apporte vraiment quelque chose ici est
# le MRR, qui distingue "bon livre en rang 2" de "bon livre en rang 9".


def rappel_at_k(rangs: Sequence[int | None], k: int) -> float:
    """
    Proportion de requetes dont le bon livre est dans le top-k.

    `rangs` : rang du bon livre pour chaque requete, 1-indexe.
              `None` (ou 0) si le bon livre n'a pas ete retourne du tout.

    Avec un unique document pertinent par requete, c'est aussi le
    "taux de bonne reponse au rang k" (accuracy@k).
    """
    if not rangs:
        return 0.0
    touches = sum(1 for r in rangs if r is not None and 0 < r <= k)
    return touches / len(rangs)


def precision_at_k(rangs: Sequence[int | None], k: int) -> float:
    """
    Precision moyenne sur les k premiers resultats.

    Un seul document pertinent existe par requete, donc au mieux 1 des k
    resultats est correct : Precision@k = Rappel@k / k. Elle decroit
    mecaniquement quand k augmente, ce qui est attendu et doit etre
    explique dans le rapport plutot que presente comme une degradation.
    """
    if k <= 0:
        raise ValueError("k doit etre >= 1")
    return rappel_at_k(rangs, k) / k


def f1_at_k(rangs: Sequence[int | None], k: int) -> float:
    """Moyenne harmonique de Precision@k et Rappel@k."""
    p = precision_at_k(rangs, k)
    r = rappel_at_k(rangs, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(rangs: Sequence[int | None]) -> float:
    """
    Mean Reciprocal Rank : moyenne de 1 / rang du bon livre.

    Une requete dont le bon livre est absent du classement compte 0.

    Interpretation metier : un MRR de 1.0 veut dire "toujours en tete",
    0.5 veut dire "en moyenne en 2e position". C'est la metrique la plus
    parlante pour le libraire, parce qu'elle dit combien de propositions
    il devra lire avant de tomber sur le bon livre.
    """
    if not rangs:
        return 0.0
    total = sum(1.0 / r for r in rangs if r is not None and r > 0)
    return total / len(rangs)


def courbe_rappel(rangs: Sequence[int | None], k_max: int = 10) -> np.ndarray:
    """
    Rappel@k pour k = 1..k_max. Alimente le graphique n°1 du sujet
    (comparaison TF-IDF / Word2Vec / CamemBERT, k de 1 a 10).
    """
    return np.array([rappel_at_k(rangs, k) for k in range(1, k_max + 1)])


def courbe_precision(rangs: Sequence[int | None], k_max: int = 10) -> np.ndarray:
    """Precision@k pour k = 1..k_max."""
    return np.array([precision_at_k(rangs, k) for k in range(1, k_max + 1)])


def resumer_metriques_recherche(
    rangs: Sequence[int | None], ks: Iterable[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """
    Tableau de bord complet pour une configuration de recherche.

    Renvoie un dict plat, directement empilable en DataFrame pour le
    tableau de suivi des experiences demande en 5.5.2.
    """
    sortie: dict[str, float] = {
        "n_requetes": float(len(rangs)),
        "mrr": mrr(rangs),
        "rang_median": float(np.median([r for r in rangs if r])) if any(rangs) else float("nan"),
    }
    for k in ks:
        sortie[f"rappel@{k}"] = rappel_at_k(rangs, k)
        sortie[f"precision@{k}"] = precision_at_k(rangs, k)
    return sortie


def rang_du_bon_livre(classement_slugs: Sequence[str], slug_attendu: str) -> int | None:
    """
    Position 1-indexee de `slug_attendu` dans un classement, ou None.

    Helper pour convertir la sortie de l'API (liste de livres triee) en
    l'entree attendue par les metriques ci-dessus.
    """
    for i, slug in enumerate(classement_slugs, 1):
        if slug == slug_attendu:
            return i
    return None


# ============================================================================
# 2. ROUGE
# ============================================================================
#
# ROUGE mesure le recouvrement entre un resume candidat et une ou plusieurs
# references. Contrairement a BLEU (concu pour la traduction, oriente
# precision), ROUGE est historiquement oriente rappel : "combien de ce que
# la reference dit se retrouve dans le candidat".
#
# On renvoie systematiquement precision, rappel ET F1, parce que sur du
# resume extractif la precision seule est trompeuse : un resume tres court
# fait de phrases copiees a une precision quasi parfaite.


def _ngrammes(termes: Sequence[str], n: int) -> Counter:
    """Compteur des n-grammes d'une sequence de termes."""
    if n <= 0:
        raise ValueError("n doit etre >= 1")
    if len(termes) < n:
        return Counter()
    return Counter(
        tuple(termes[i:i + n]) for i in range(len(termes) - n + 1)
    )


def _scores(recouvrement: int, n_candidat: int, n_reference: int) -> dict[str, float]:
    """Assemble precision / rappel / F1 a partir d'un recouvrement brut."""
    precision = recouvrement / n_candidat if n_candidat else 0.0
    rappel = recouvrement / n_reference if n_reference else 0.0
    if precision + rappel == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * rappel / (precision + rappel)
    return {"precision": precision, "rappel": rappel, "f1": f1}


def rouge_n(
    candidat: Sequence[str], reference: Sequence[str], n: int = 1
) -> dict[str, float]:
    """
    ROUGE-N entre un candidat et une reference (listes de termes).

    Le recouvrement est "clippe" : un n-gramme present 3 fois dans le
    candidat et 1 fois dans la reference ne compte que pour 1. Sans ce
    clipping, repeter le meme mot ferait monter le score artificiellement.
    """
    ng_cand = _ngrammes(candidat, n)
    ng_ref = _ngrammes(reference, n)
    recouvrement = sum((ng_cand & ng_ref).values())
    return _scores(recouvrement, sum(ng_cand.values()), sum(ng_ref.values()))


def longueur_lcs(a: Sequence[str], b: Sequence[str]) -> int:
    """
    Longueur de la plus longue sous-sequence commune (LCS).

    Programmation dynamique en O(len(a) * len(b)) en temps, mais en
    O(min(len(a), len(b))) en memoire : on ne garde que deux lignes de la
    table.

    ATTENTION AU COUT : comparer un resume de 300 termes au texte integral
    d'un roman (~80 000 termes) fait 24 millions d'operations en Python
    pur, soit plusieurs secondes par livre. Pour l'evaluation "candidat vs
    texte integral", passer par `rouge_n` (lineaire) ou echantillonner la
    reference.
    """
    if not a or not b:
        return 0
    # On itere sur la plus longue en externe pour que la ligne de DP
    # (de taille len(court) + 1) reste la plus petite possible.
    if len(a) < len(b):
        court, long_ = a, b
    else:
        court, long_ = b, a

    precedente = [0] * (len(court) + 1)
    for terme_long in long_:
        courante = [0] * (len(court) + 1)
        for j, terme_court in enumerate(court, 1):
            if terme_long == terme_court:
                courante[j] = precedente[j - 1] + 1
            else:
                courante[j] = max(precedente[j], courante[j - 1])
        precedente = courante
    return precedente[-1]


def rouge_l(candidat: Sequence[str], reference: Sequence[str]) -> dict[str, float]:
    """
    ROUGE-L : recouvrement base sur la plus longue sous-sequence commune.

    Interet par rapport a ROUGE-N : la LCS ne demande pas que les mots
    soient contigus, seulement qu'ils soient dans le bon ordre. Elle
    capture donc la structure de phrase la ou ROUGE-2 ne verrait rien
    des qu'un mot est insere au milieu d'un bigramme.
    """
    lcs = longueur_lcs(candidat, reference)
    return _scores(lcs, len(candidat), len(reference))


# ----------------------------------------------------------------------------
# ROUGE-Lsum (LCS niveau resume)
# ----------------------------------------------------------------------------
#
# `rouge_l` ci-dessus colle tout le resume en une seule sequence de termes.
# Sur un resume multi-phrases c'est trompeur : la LCS peut sauter d'une
# phrase du candidat a une autre, et compter comme "structure preservee"
# un alignement qui traverse trois phrases sans rapport.
#
# ROUGE-Lsum (Lin 2004, section 3.2) travaille phrase par phrase : pour
# chaque phrase de la REFERENCE, on prend l'union des positions couvertes
# par la LCS avec chacune des phrases du candidat. Un terme de la reference
# ne peut donc etre "gagne" qu'une fois, mais il peut l'etre par n'importe
# quelle phrase du candidat.
#
# Ce n'est pas moins couteux que `rouge_l` (le total des produits de
# longueurs est le meme), mais la table de DP reste de la taille d'une
# phrase et non du resume entier. Et surtout c'est la variante que
# `rouge-score` publie sous le nom `rougeLsum`, donc celle a laquelle nos
# chiffres seront compares.


def _table_lcs(reference: Sequence[str], candidat: Sequence[str]) -> list[list[int]]:
    """
    Table de programmation dynamique complete des longueurs de LCS.

    Contrairement a `longueur_lcs`, on garde toute la table : le backtracking
    en a besoin pour retrouver QUELLES positions sont alignees, pas seulement
    combien. Acceptable ici parce qu'on ne l'appelle que sur des phrases.
    """
    table = [[0] * (len(candidat) + 1) for _ in range(len(reference) + 1)]
    for i, terme_ref in enumerate(reference, 1):
        for j, terme_cand in enumerate(candidat, 1):
            if terme_ref == terme_cand:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table


def _indices_lcs(reference: Sequence[str], candidat: Sequence[str]) -> list[int]:
    """
    Positions (dans `reference`) des termes retenus par une LCS.

    Plusieurs LCS de meme longueur peuvent exister ; on remonte la table de
    facon deterministe (meme ordre de tie-break que `rouge-score`) pour que
    les scores soient reproductibles et comparables a la reference.
    """
    if not reference or not candidat:
        return []
    table = _table_lcs(reference, candidat)
    i, j = len(reference), len(candidat)
    indices: list[int] = []
    while i > 0 and j > 0:
        if reference[i - 1] == candidat[j - 1]:
            indices.append(i - 1)
            i -= 1
            j -= 1
        elif table[i][j - 1] > table[i - 1][j]:
            j -= 1
        else:
            i -= 1
    indices.reverse()
    return indices


def _union_lcs(phrase_ref: Sequence[str], phrases_cand: Sequence[Sequence[str]]) -> list[str]:
    """Termes d'une phrase de reference couverts par au moins une phrase candidate."""
    couvertes: set[int] = set()
    for phrase_cand in phrases_cand:
        couvertes.update(_indices_lcs(phrase_ref, phrase_cand))
    return [phrase_ref[i] for i in sorted(couvertes)]


def rouge_lsum(
    candidat: Sequence[Sequence[str]], reference: Sequence[Sequence[str]]
) -> dict[str, float]:
    """
    ROUGE-Lsum : LCS niveau resume, entre deux resumes decoupes en phrases.

    `candidat` et `reference` sont des listes de phrases, chaque phrase
    etant une liste de termes -- pas des listes de termes a plat. C'est
    exactement la forme que sort `summarization.resumer_livre` via le
    champ `termes` (voir `termes_par_phrase`).

    Le comptage des "hits" est borne par les frequences globales des deux
    cotes : sans ce garde-fou, un terme frequent dans le candidat pourrait
    etre credite une fois par phrase de reference et faire exploser le
    rappel.
    """
    n_ref = sum(len(p) for p in reference)
    n_cand = sum(len(p) for p in candidat)
    if n_ref == 0 or n_cand == 0:
        return _scores(0, n_cand, n_ref)

    restants_ref: Counter = Counter()
    for phrase in reference:
        restants_ref.update(phrase)
    restants_cand: Counter = Counter()
    for phrase in candidat:
        restants_cand.update(phrase)

    recouvrement = 0
    for phrase_ref in reference:
        for terme in _union_lcs(phrase_ref, candidat):
            if restants_ref[terme] > 0 and restants_cand[terme] > 0:
                recouvrement += 1
                restants_ref[terme] -= 1
                restants_cand[terme] -= 1

    return _scores(recouvrement, n_cand, n_ref)


def rouge_multi_references(
    candidat: Sequence[str],
    references: Sequence[Sequence[str]],
    n: int = 1,
    variante: str = "n",
) -> dict[str, float]:
    """
    ROUGE contre plusieurs references : on garde le meilleur F1.

    C'est la convention ROUGE officielle. Prendre le max plutot que la
    moyenne evite de penaliser un candidat correct qui ne ressemble qu'a
    une seule des references disponibles.

    `variante` : "n" pour ROUGE-N (utilise `n`), "l" pour ROUGE-L. Pas de
    "lsum" ici : `rouge_lsum` attend des phrases et pas des termes a plat,
    melanger les deux formes dans la meme signature serait un piege. Pour
    du multi-references en Lsum, boucler et prendre le max de "f1" a la main.
    """
    if not references:
        raise ValueError("Il faut au moins une reference.")
    if variante == "l":
        tous = [rouge_l(candidat, ref) for ref in references]
    elif variante == "n":
        tous = [rouge_n(candidat, ref, n=n) for ref in references]
    else:
        raise ValueError("variante doit valoir 'n' ou 'l'")
    return max(tous, key=lambda d: d["f1"])


def rouge_complet(
    candidat: Sequence[str], reference: Sequence[str], avec_l: bool = True
) -> dict[str, float]:
    """
    ROUGE-1, ROUGE-2 et ROUGE-L en un appel, aplati pour un DataFrame.

    `avec_l=False` saute la LCS, qui est de loin la partie la plus couteuse
    quand la reference est longue.
    """
    sortie: dict[str, float] = {}
    for n in (1, 2):
        for cle, valeur in rouge_n(candidat, reference, n=n).items():
            sortie[f"rouge{n}_{cle}"] = valeur
    if avec_l:
        for cle, valeur in rouge_l(candidat, reference).items():
            sortie[f"rougeL_{cle}"] = valeur
    return sortie


def rouge_complet_phrases(
    candidat: Sequence[Sequence[str]],
    reference: Sequence[Sequence[str]],
    avec_l: bool = False,
    ngrammes_par_phrase: bool = True,
) -> dict[str, float]:
    """
    Meme chose que `rouge_complet`, mais a partir de resumes en phrases.

    La partie LCS passe par `rouge_lsum`. `avec_l=True` ajoute en plus le
    ROUGE-L a plat, utile uniquement pour montrer dans le rapport l'ecart
    entre les deux variantes.

    `ngrammes_par_phrase` decide du sort des bigrammes a cheval sur deux
    phrases :

    - `True` (defaut) : les n-grammes sont extraits phrase par phrase. Un
      resume extractif colle des phrases prises a des centaines de pages
      d'ecart ; le bigramme forme par le dernier mot de l'une et le premier
      mot de l'autre est un artefact de concatenation, pas du contenu.
    - `False` : reproduit `rouge-score`, qui aplatit le resume avant de
      compter. A utiliser pour comparer nos chiffres a ceux de la
      litterature ou d'un autre outil.

    L'ecart est nul sur ROUGE-1 (un unigramme ne traverse rien) et vaut au
    plus (nb_phrases - 1) bigrammes sur ROUGE-2, soit quelques pourcents
    pour un resume de 10 phrases de 30 termes. Dire dans le rapport lequel
    des deux modes est rapporte.
    """
    plat_cand = [t for phrase in candidat for t in phrase]
    plat_ref = [t for phrase in reference for t in phrase]

    sortie: dict[str, float] = {}
    for n in (1, 2):
        if ngrammes_par_phrase:
            ng_cand: Counter = Counter()
            ng_ref: Counter = Counter()
            for phrase in candidat:
                ng_cand += _ngrammes(phrase, n)
            for phrase in reference:
                ng_ref += _ngrammes(phrase, n)
            recouvrement = sum((ng_cand & ng_ref).values())
            scores = _scores(
                recouvrement, sum(ng_cand.values()), sum(ng_ref.values())
            )
        else:
            scores = rouge_n(plat_cand, plat_ref, n=n)
        for cle, valeur in scores.items():
            sortie[f"rouge{n}_{cle}"] = valeur

    for cle, valeur in rouge_lsum(candidat, reference).items():
        sortie[f"rougeLsum_{cle}"] = valeur
    if avec_l:
        for cle, valeur in rouge_l(plat_cand, plat_ref).items():
            sortie[f"rougeL_{cle}"] = valeur
    return sortie


def termes_par_phrase(
    phrases: Sequence[dict], champ: str = "termes"
) -> list[list[str]]:
    """
    Sortie de `summarization.resumer_livre` -> entree de `rouge_lsum`.

    Passerelle volontairement minuscule : elle ne fait que projeter un
    champ, pour que le module d'evaluation n'ait jamais a importer
    `summarization` et reste testable a vide.
    """
    return [list(p[champ]) for p in phrases]


# ----------------------------------------------------------------------------
# Agregation sur un corpus
# ----------------------------------------------------------------------------


def agreger_scores(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """
    Moyenne et ecart-type, cle par cle, d'une liste de dicts de scores.

    Accepte n'importe quelle sortie plate du module (`rouge_complet`,
    `rouge_complet_phrases`, ...). L'ecart-type compte autant que la
    moyenne dans le rapport : sur 10 livres, un ROUGE-1 moyen de 0.30 a
    +/- 0.02 et le meme a +/- 0.15 ne racontent pas la meme histoire sur
    la robustesse de la methode.

    Les cles absentes d'un dict sont ignorees pour ce dict (moyenne
    calculee sur les seuls livres ou la metrique existe), ce qui permet de
    melanger des resultats calcules avec et sans `avec_l`.
    """
    if not scores:
        return {"n_resumes": 0.0}

    cles: list[str] = []
    for d in scores:
        for cle in d:
            if cle not in cles:
                cles.append(cle)

    sortie: dict[str, float] = {"n_resumes": float(len(scores))}
    for cle in cles:
        valeurs = [d[cle] for d in scores if cle in d]
        sortie[cle] = float(np.mean(valeurs))
        sortie[f"{cle}_ecart_type"] = float(np.std(valeurs))
    return sortie


def resumer_metriques_resume(
    candidats: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    avec_l: bool = True,
) -> dict[str, float]:
    """
    Tableau de bord ROUGE pour une configuration de resume.

    Pendant de `resumer_metriques_recherche` pour la tache 2 : une ligne de
    DataFrame par configuration (TF-IDF naif / TextRank / MMR), a empiler
    pour le tableau de suivi des experiences.

    `candidats` et `references` sont apparies positionnellement (un couple
    par livre) et doivent avoir la meme longueur.
    """
    if len(candidats) != len(references):
        raise ValueError(
            f"{len(candidats)} candidats pour {len(references)} references"
        )
    scores = [
        rouge_complet(cand, ref, avec_l=avec_l)
        for cand, ref in zip(candidats, references)
    ]
    return agreger_scores(scores)


# ============================================================================
# 3. BLEU
# ============================================================================


def bleu(
    candidat: Sequence[str],
    references: Sequence[Sequence[str]],
    n_max: int = 4,
    poids: Sequence[float] | None = None,
    lissage: bool = True,
) -> float:
    """
    BLEU avec penalite de brievete et lissage optionnel.

    Formule : BP * exp( somme_n poids_n * log(p_n) ), ou p_n est la
    precision n-gramme "modifiee" (clippee par le compte maximum observe
    parmi les references).

    `lissage=True` applique le lissage add-1 sur les ordres n >= 2
    (methode 2 de Chen & Cherry 2014, due a Lin & Och 2004 -- la methode 1
    est un epsilon, pas un add-1). Sans lui, un seul ordre a zero
    recouvrement suffit a annuler tout le score -- ce qui arrive tout le
    temps sur des resumes courts en 4-grammes. Equivalent a
    `nltk.translate.bleu_score.SmoothingFunction().method2`, verifie dans
    tests/test_evaluation.py.

    BLEU est de toute facon un choix discutable pour du resume extractif :
    concu pour la traduction, il est oriente precision et penalise la
    reformulation. Il est demande par le sujet (5.5.1), on le calcule, mais
    le rapport doit dire que ROUGE reste la metrique de reference ici.
    """
    if not references:
        raise ValueError("Il faut au moins une reference.")
    if poids is None:
        poids = [1.0 / n_max] * n_max
    if len(poids) != n_max:
        raise ValueError("poids doit avoir n_max elements")
    if not candidat:
        return 0.0

    log_precisions = []
    for n in range(1, n_max + 1):
        ng_cand = _ngrammes(candidat, n)
        total_cand = sum(ng_cand.values())
        if total_cand == 0:
            # Candidat plus court que n : cet ordre n'est pas evaluable.
            log_precisions.append(float("-inf"))
            continue

        # Clipping multi-references : pour chaque n-gramme, le compte
        # maximum observe sur l'ensemble des references.
        max_ref: Counter = Counter()
        for ref in references:
            max_ref |= _ngrammes(ref, n)
        recouvrement = sum((ng_cand & max_ref).values())

        if lissage and n > 1:
            recouvrement += 1
            total_cand += 1
        if recouvrement == 0:
            log_precisions.append(float("-inf"))
        else:
            log_precisions.append(np.log(recouvrement / total_cand))

    if any(lp == float("-inf") for lp in log_precisions):
        return 0.0

    # Penalite de brievete, calculee contre la reference de longueur la
    # plus proche (convention BLEU).
    len_cand = len(candidat)
    len_ref = min((len(r) for r in references), key=lambda l: (abs(l - len_cand), l))
    if len_cand > len_ref:
        bp = 1.0
    elif len_cand == 0:
        bp = 0.0
    else:
        bp = float(np.exp(1 - len_ref / len_cand))

    score = bp * float(np.exp(sum(w * lp for w, lp in zip(poids, log_precisions))))
    return score
