"""
Tests de `pipeline.representations`.

Trois niveaux, dans l'esprit de `test_evaluation.py` :

1. **Contrat de sortie**. `TfIdfMaison` renvoie desormais du creux, et
   plusieurs appelants indexent la matrice (`matrice[i]`) ou la passent aux
   fonctions de similarite. Ces tests fixent ce contrat pour qu'un futur
   retour au dense ne passe pas inapercu.

2. **Equivalence avec une reference dense recalculee a la main**. C'est le
   test qui compte : le passage au creux devait etre une optimisation
   memoire, pas un changement de resultat. La reference est recalculee ici
   depuis la definition (comptage, IDF lissee, L2), pas recopiee de
   l'implementation testee.

3. **Comparaison a sklearn**, saute si le paquet est absent, comme pour
   ROUGE et BLEU dans `test_evaluation.py`.

Le calcul des similarites euclidiennes est passe d'une difference explicite
(`X - requete`) a l'identite ||x-q||^2 = ||x||^2 + ||q||^2 - 2x.q, pour ne
pas materialiser une matrice dense. Les tests comparent donc a la formule
naive sur de petits cas, ou les deux sont calculables.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sp

from pipeline.representations import (
    TfIdfMaison,
    calculer_idf,
    construire_ngrammes,
    cosinus,
    similarites_cosinus,
    similarites_euclidiennes,
    similarites_jaccard,
)


# ============================================================================
# Corpus de test
# ============================================================================

DOCUMENTS = [
    "le chat dort sur le tapis rouge".split(),
    "le chien dort sur le tapis bleu".split(),
    "un oiseau chante dans le jardin".split(),
    "le jardin rouge accueille un chat".split(),
    "personne ne bouge".split(),
]


def tfidf_reference(documents, ngram_max=1, min_df=1, max_df_ratio=1.0):
    """
    TF-IDF dense recalcule depuis la definition, comme temoin.

    Conventions sklearn : idf(t) = log((1+N)/(1+df(t))) + 1, tf = comptage
    brut, normalisation L2 par ligne, lignes nulles laissees a zero.
    """
    N = len(documents)
    comptes = [Counter(construire_ngrammes(d, ngram_max)) for d in documents]

    df = Counter()
    for c in comptes:
        df.update(c.keys())

    vocab = sorted(t for t, n in df.items()
                   if min_df <= n <= max_df_ratio * N)
    index = {t: i for i, t in enumerate(vocab)}

    X = np.zeros((N, len(vocab)))
    for i, c in enumerate(comptes):
        for terme, n in c.items():
            if terme in index:
                X[i, index[terme]] = n * (math.log((1 + N) / (1 + df[terme])) + 1.0)

    normes = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(normes == 0, 1.0, normes), vocab


# ============================================================================
# 1. Contrat de sortie
# ============================================================================

def test_fit_transform_renvoie_du_creux():
    X = TfIdfMaison().fit_transform(DOCUMENTS)
    assert sp.issparse(X)
    assert X.shape[0] == len(DOCUMENTS)


def test_transform_renvoie_du_creux():
    vec = TfIdfMaison()
    vec.fit_transform(DOCUMENTS)
    Q = vec.transform([["chat", "tapis"]])
    assert sp.issparse(Q)
    assert Q.shape == (1, len(vec.vocabulaire_))


def test_indexation_donne_une_ligne_utilisable_comme_requete():
    """`api.vectoriser_extrait` renvoie `matrice[0]` : cette forme doit passer."""
    vec = TfIdfMaison()
    X = vec.fit_transform(DOCUMENTS)
    requete = vec.transform([["chat", "jardin"]])[0]

    scores = similarites_cosinus(X, requete)
    assert scores.shape == (len(DOCUMENTS),)
    assert np.all(np.isfinite(scores))


def test_transform_avant_fit_leve():
    with pytest.raises(RuntimeError):
        TfIdfMaison().transform([["chat"]])


# ============================================================================
# 2. Equivalence avec la reference dense
# ============================================================================

@pytest.mark.parametrize("ngram_max", [1, 2, 3])
@pytest.mark.parametrize("min_df,max_df_ratio", [(1, 1.0), (2, 1.0), (1, 0.6)])
def test_matrice_identique_a_la_reference(ngram_max, min_df, max_df_ratio):
    attendu, vocab = tfidf_reference(
        DOCUMENTS, ngram_max, min_df, max_df_ratio,
    )
    vec = TfIdfMaison(ngram_max, min_df, max_df_ratio)
    obtenu = vec.fit_transform(DOCUMENTS)

    assert vec.vocabulaire_ == vocab
    assert np.allclose(obtenu.toarray(), attendu, atol=1e-12)


def test_lignes_normalisees_l2():
    X = TfIdfMaison(ngram_max=2).fit_transform(DOCUMENTS).toarray()
    normes = np.linalg.norm(X, axis=1)
    assert np.allclose(normes, 1.0, atol=1e-12)


def test_ligne_hors_vocabulaire_reste_nulle():
    """
    Un extrait dont aucun terme n'est connu doit donner une ligne nulle,
    et non des NaN : la normalisation ne doit pas diviser par zero.
    """
    vec = TfIdfMaison()
    X = vec.fit_transform(DOCUMENTS)
    Q = vec.transform([["xyzzy", "plugh"]])

    assert Q.nnz == 0
    scores = similarites_cosinus(X, Q[0])
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores, 0.0)


def test_idf_penalise_les_termes_frequents():
    """`le` est dans 4 documents sur 5, `personne` dans un seul."""
    vec = TfIdfMaison()
    vec.fit_transform(DOCUMENTS)
    idf = dict(zip(vec.vocabulaire_, vec.idf_))
    assert idf["le"] < idf["personne"]


# ============================================================================
# 3. Similarites
# ============================================================================

def test_cosinus_accepte_creux_et_dense_indifferemment():
    vec = TfIdfMaison()
    X = vec.fit_transform(DOCUMENTS)
    requete = vec.transform([["chat", "tapis"]])

    creux = similarites_cosinus(X, requete[0])
    dense = similarites_cosinus(X.toarray(), requete.toarray()[0])
    assert np.allclose(creux, dense, atol=1e-12)


def test_cosinus_vaut_un_sur_le_document_identique():
    vec = TfIdfMaison()
    X = vec.fit_transform(DOCUMENTS)
    requete = vec.transform([DOCUMENTS[0]])[0]
    scores = similarites_cosinus(X, requete)
    assert scores[0] == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(scores) == 0


def test_euclidien_egale_la_formule_naive():
    """
    L'identite ||x-q||^2 = ||x||^2 + ||q||^2 - 2x.q doit redonner exactement
    la difference explicite, y compris sur la ligne nulle du corpus.
    """
    vec = TfIdfMaison()
    X = vec.fit_transform(DOCUMENTS)
    requete = vec.transform([["chat", "jardin", "rouge"]])[0]

    obtenu = similarites_euclidiennes(X, requete)

    dense_X = X.toarray()
    dense_q = requete.toarray().ravel()
    distances = np.linalg.norm(dense_X - dense_q[np.newaxis, :], axis=1)
    attendu = 1.0 / (1.0 + distances)

    assert np.allclose(obtenu, attendu, atol=1e-12)


def test_euclidien_et_cosinus_classent_pareil():
    """
    Sur des vecteurs L2-normalises les deux metriques donnent le meme
    ordre. C'est la conclusion du notebook 05, et elle doit survivre au
    changement de mode de calcul.

    On ecarte la ligne nulle du corpus : sa norme n'est pas 1, donc
    l'equivalence des deux metriques ne s'y applique pas.
    """
    vec = TfIdfMaison()
    X = vec.fit_transform(DOCUMENTS)
    non_nulles = np.asarray(X.multiply(X).sum(axis=1)).ravel() > 0
    X = X[non_nulles]

    for termes in (["chat"], ["tapis", "bleu"], ["jardin", "rouge", "oiseau"]):
        requete = vec.transform([termes])[0]
        cos = similarites_cosinus(X, requete)
        euc = similarites_euclidiennes(X, requete)
        assert (np.argsort(-cos, kind="stable")
                == np.argsort(-euc, kind="stable")).all()


def test_jaccard_sur_ensembles():
    a = [{1, 2, 3}, {1, 2}, set()]
    scores = similarites_jaccard(a, {1, 2})
    assert scores[0] == pytest.approx(2 / 3)
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)


# ============================================================================
# 4. Helpers
# ============================================================================

def test_construire_ngrammes():
    assert construire_ngrammes(["a", "b", "c"], 1) == ["a", "b", "c"]
    assert construire_ngrammes(["a", "b", "c"], 2) == [
        "a", "b", "c", "a b", "b c",
    ]


def test_calculer_idf_coherent_avec_la_classe():
    idf_fonction = calculer_idf(DOCUMENTS)
    vec = TfIdfMaison()
    vec.fit_transform(DOCUMENTS)
    idf_classe = dict(zip(vec.vocabulaire_, vec.idf_))
    for terme, valeur in idf_classe.items():
        assert idf_fonction[terme] == pytest.approx(valeur)


def test_cosinus_gere_le_vecteur_nul():
    assert cosinus(np.zeros(3), np.ones(3)) == 0.0


# ============================================================================
# 5. Reference externe : sklearn
# ============================================================================

def test_contre_sklearn():
    """
    Meme vocabulaire et memes valeurs que `TfidfVectorizer`.

    C'est ce qui justifie d'avoir reimplemente TF-IDF a la main. On passe
    `token_pattern=r"\\S+"` et `lowercase=False` pour que sklearn prenne nos
    termes deja tokenises tels quels, sans retokeniser.
    """
    sklearn_text = pytest.importorskip(
        "sklearn.feature_extraction.text",
        reason="scikit-learn absent : comparaison de reference sautee",
    )

    textes = [" ".join(d) for d in DOCUMENTS]
    sk = sklearn_text.TfidfVectorizer(
        analyzer="word", token_pattern=r"\S+", lowercase=False, min_df=1,
    )
    attendu = sk.fit_transform(textes).toarray()

    vec = TfIdfMaison()
    obtenu = vec.fit_transform(DOCUMENTS).toarray()

    assert list(sk.get_feature_names_out()) == vec.vocabulaire_
    assert np.allclose(obtenu, attendu, atol=1e-12)
