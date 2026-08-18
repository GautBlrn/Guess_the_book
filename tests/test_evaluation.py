"""
Tests de `pipeline.evaluation`.

Deux niveaux :

1. **Proprietes et cas limites**, verifiables sans dependance : listes vides,
   candidat identique a la reference, monotonies attendues (Rappel@k croit
   avec k), coherence entre metriques.

2. **Comparaison a une implementation de reference** : `rouge-score`
   (Google) pour ROUGE, `nltk` pour BLEU. C'est ce qui justifie d'avoir
   reecrit ces metriques a la main : on montre qu'on tombe sur les memes
   chiffres. Ces tests sont sautes si le paquet n'est pas installe, pour
   que la suite reste verte avec `requirements.txt` seul.

Le corpus de test est genere aleatoirement a partir d'un petit vocabulaire,
avec une seed fixe : assez de recouvrement partiel pour que les scores
soient interessants, et reproductible.
"""
from __future__ import annotations

import random

import pytest

from pipeline.evaluation import (
    _indices_lcs,
    _ngrammes,
    agreger_scores,
    bleu,
    courbe_precision,
    courbe_rappel,
    f1_at_k,
    longueur_lcs,
    mrr,
    precision_at_k,
    rang_du_bon_livre,
    rappel_at_k,
    resumer_metriques_recherche,
    resumer_metriques_resume,
    rouge_complet,
    rouge_complet_phrases,
    rouge_l,
    rouge_lsum,
    rouge_multi_references,
    rouge_n,
    termes_par_phrase,
)

VOCAB = "le chat noir dort sur un tapis rouge pendant que la pluie tombe".split()
CLES = ("precision", "rappel", "f1")


def _phrases(rng: random.Random, n_phrases: int, n_max: int) -> list[list[str]]:
    """Resume aleatoire : liste de phrases, chaque phrase liste de termes."""
    return [
        [rng.choice(VOCAB) for _ in range(rng.randint(1, n_max))]
        for _ in range(rng.randint(1, n_phrases))
    ]


def _plat(phrases):
    return [t for phrase in phrases for t in phrase]


def _triplet(scores: dict[str, float]) -> tuple[float, float, float]:
    return tuple(scores[c] for c in CLES)


# ============================================================================
# 1. Metriques de recherche
# ============================================================================


def test_rappel_at_k_compte_les_rangs_dans_le_top_k():
    rangs = [1, 3, 5, None, 12]
    assert rappel_at_k(rangs, 1) == pytest.approx(1 / 5)
    assert rappel_at_k(rangs, 3) == pytest.approx(2 / 5)
    assert rappel_at_k(rangs, 5) == pytest.approx(3 / 5)
    assert rappel_at_k(rangs, 20) == pytest.approx(4 / 5)


def test_rappel_at_k_traite_zero_comme_absent():
    # Une implementation d'API peut renvoyer 0 plutot que None pour "absent".
    assert rappel_at_k([0, 0], 10) == 0.0
    assert rappel_at_k([None, 0, 1], 1) == pytest.approx(1 / 3)


def test_rappel_est_croissant_en_k():
    rng = random.Random(0)
    rangs = [rng.choice([None, *range(1, 15)]) for _ in range(50)]
    courbe = courbe_rappel(rangs, k_max=10)
    assert all(courbe[i] <= courbe[i + 1] for i in range(len(courbe) - 1))


def test_precision_vaut_rappel_sur_k():
    # Consequence de l'hypothese "un seul document pertinent par requete",
    # assumee explicitement dans le module. Si ca casse, l'hypothese a change.
    rangs = [1, 2, None, 7]
    for k in (1, 3, 5, 10):
        assert precision_at_k(rangs, k) == pytest.approx(rappel_at_k(rangs, k) / k)
    assert courbe_precision(rangs, 4) == pytest.approx(courbe_rappel(rangs, 4) / [1, 2, 3, 4])


def test_precision_refuse_k_nul_ou_negatif():
    with pytest.raises(ValueError):
        precision_at_k([1], 0)
    with pytest.raises(ValueError):
        precision_at_k([1], -3)


def test_f1_at_k_est_nul_si_aucune_bonne_reponse():
    assert f1_at_k([None, None], 5) == 0.0
    assert f1_at_k([1], 1) == pytest.approx(1.0)


def test_mrr_moyenne_les_inverses_de_rang():
    assert mrr([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)
    assert mrr([None, 1]) == pytest.approx(0.5)
    assert mrr([]) == 0.0


def test_metriques_vides_ne_levent_pas():
    assert rappel_at_k([], 5) == 0.0
    assert precision_at_k([], 5) == 0.0
    assert f1_at_k([], 5) == 0.0
    assert mrr([]) == 0.0


def test_resumer_metriques_recherche_expose_les_bonnes_cles():
    sortie = resumer_metriques_recherche([1, 3, None], ks=(1, 3))
    assert sortie["n_requetes"] == 3.0
    assert sortie["rang_median"] == pytest.approx(2.0)
    assert set(sortie) == {
        "n_requetes", "mrr", "rang_median",
        "rappel@1", "precision@1", "rappel@3", "precision@3",
    }


def test_rang_du_bon_livre():
    classement = ["zola_nana", "hugo_miserables", "verne_lune"]
    assert rang_du_bon_livre(classement, "zola_nana") == 1
    assert rang_du_bon_livre(classement, "verne_lune") == 3
    assert rang_du_bon_livre(classement, "absent") is None
    assert rang_du_bon_livre([], "zola_nana") is None


# ============================================================================
# 2. Briques ROUGE : n-grammes et LCS
# ============================================================================


def test_ngrammes_compte_avec_repetitions():
    assert _ngrammes(["a", "b", "a", "b"], 2) == {("a", "b"): 2, ("b", "a"): 1}
    assert _ngrammes(["a"], 2) == {}
    with pytest.raises(ValueError):
        _ngrammes(["a"], 0)


def test_longueur_lcs_cas_connus():
    assert longueur_lcs(list("abcde"), list("ace")) == 3
    assert longueur_lcs(list("abc"), list("cba")) == 1
    assert longueur_lcs([], list("abc")) == 0
    assert longueur_lcs(list("abc"), list("abc")) == 3


def test_longueur_lcs_est_symetrique():
    rng = random.Random(1)
    for _ in range(50):
        a = [rng.choice(VOCAB[:5]) for _ in range(rng.randint(0, 8))]
        b = [rng.choice(VOCAB[:5]) for _ in range(rng.randint(0, 8))]
        assert longueur_lcs(a, b) == longueur_lcs(b, a)


def test_indices_lcs_designe_une_vraie_sous_sequence_maximale():
    rng = random.Random(2)
    for _ in range(200):
        a = [rng.choice(VOCAB[:6]) for _ in range(rng.randint(0, 9))]
        b = [rng.choice(VOCAB[:6]) for _ in range(rng.randint(0, 9))]
        indices = _indices_lcs(a, b)

        assert indices == sorted(set(indices)), "indices non strictement croissants"
        assert len(indices) == longueur_lcs(a, b), "LCS non maximale"

        # Les termes designes doivent former une sous-sequence de b aussi.
        j = 0
        for terme in (a[i] for i in indices):
            while j < len(b) and b[j] != terme:
                j += 1
            assert j < len(b), f"{terme} absent de {b} dans l'ordre"
            j += 1


# ============================================================================
# 3. ROUGE : proprietes
# ============================================================================


@pytest.mark.parametrize("n", [1, 2, 3])
def test_rouge_n_parfait_sur_candidat_identique(n):
    termes = ["le", "chat", "noir", "dort", "la"]
    assert _triplet(rouge_n(termes, termes, n=n)) == (1.0, 1.0, 1.0)


def test_rouge_n_nul_sur_vocabulaires_disjoints():
    assert _triplet(rouge_n(["a", "b"], ["c", "d"])) == (0.0, 0.0, 0.0)


def test_rouge_n_clippe_les_repetitions():
    # "chat" 3 fois cote candidat, 1 fois cote reference : recouvrement = 1.
    scores = rouge_n(["chat", "chat", "chat"], ["chat", "dort"])
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["rappel"] == pytest.approx(1 / 2)


def test_rouge_n_gere_les_sequences_vides():
    assert _triplet(rouge_n([], ["a"])) == (0.0, 0.0, 0.0)
    assert _triplet(rouge_n(["a"], [])) == (0.0, 0.0, 0.0)
    assert _triplet(rouge_n([], [])) == (0.0, 0.0, 0.0)


def test_rouge_l_ignore_les_insertions():
    # ROUGE-2 ne voit plus le bigramme, ROUGE-L retrouve l'ordre.
    cand = ["le", "chat", "dort"]
    ref = ["le", "gros", "chat", "dort"]
    assert rouge_n(cand, ref, n=2)["f1"] < rouge_l(cand, ref)["f1"]
    assert rouge_l(cand, ref)["precision"] == pytest.approx(1.0)


def test_rouge_lsum_insensible_a_l_ordre_des_phrases():
    # C'est la raison d'etre de Lsum : un resume extractif reordonne ne doit
    # pas etre penalise, alors que la LCS a plat casse sur la permutation.
    cand = [["la", "pluie", "tombe"], ["le", "chat", "dort"]]
    ref = [["le", "chat", "dort"], ["la", "pluie", "tombe"]]
    assert _triplet(rouge_lsum(cand, ref)) == (1.0, 1.0, 1.0)
    assert rouge_l(_plat(cand), _plat(ref))["f1"] == pytest.approx(0.5)


def test_rouge_lsum_ne_credite_pas_deux_fois_le_meme_terme():
    # "chat" une seule fois cote candidat, deux phrases de reference le
    # contiennent : sans le garde-fou de frequence, le rappel monterait a 1.
    cand = [["chat"]]
    ref = [["chat"], ["chat"]]
    assert rouge_lsum(cand, ref)["rappel"] == pytest.approx(0.5)


def test_rouge_lsum_gere_les_resumes_vides():
    assert _triplet(rouge_lsum([], [["a"]])) == (0.0, 0.0, 0.0)
    assert _triplet(rouge_lsum([["a"]], [])) == (0.0, 0.0, 0.0)
    assert _triplet(rouge_lsum([[]], [[]])) == (0.0, 0.0, 0.0)


def test_rouge_multi_references_prend_le_meilleur_f1():
    cand = ["le", "chat", "dort"]
    mauvaise = ["la", "pluie", "tombe"]
    bonne = ["le", "chat", "dort"]
    assert rouge_multi_references(cand, [mauvaise, bonne])["f1"] == pytest.approx(1.0)
    assert rouge_multi_references(cand, [bonne, mauvaise], variante="l")["f1"] == pytest.approx(1.0)


def test_rouge_multi_references_valide_ses_arguments():
    with pytest.raises(ValueError):
        rouge_multi_references(["a"], [])
    with pytest.raises(ValueError):
        rouge_multi_references(["a"], [["a"]], variante="lsum")


def test_rouge_complet_expose_les_cles_attendues():
    cand, ref = ["le", "chat", "dort"], ["le", "chat", "noir", "dort"]
    sortie = rouge_complet(cand, ref)
    assert set(sortie) == {
        f"rouge{v}_{c}" for v in ("1", "2", "L") for c in CLES
    }
    assert not any(cle.startswith("rougeL_") for cle in rouge_complet(cand, ref, avec_l=False))


def test_rouge_complet_phrases_expose_lsum():
    cand = [["le", "chat"], ["il", "dort"]]
    sortie = rouge_complet_phrases(cand, cand)
    assert set(sortie) == {
        f"rouge{v}_{c}" for v in ("1", "2", "Lsum") for c in CLES
    }
    assert all(valeur == pytest.approx(1.0) for valeur in sortie.values())
    assert "rougeL_f1" in rouge_complet_phrases(cand, cand, avec_l=True)


def test_bigrammes_de_jointure_ne_comptent_pas_par_defaut():
    # "chat" et "il" sont dans deux phrases differentes : le bigramme
    # ("chat", "il") est un artefact de concatenation, pas du contenu.
    cand = [["le", "chat"], ["il", "dort"]]
    ref = [["le", "chat", "il", "dort"]]
    par_phrase = rouge_complet_phrases(cand, ref)
    a_plat = rouge_complet_phrases(cand, ref, ngrammes_par_phrase=False)

    assert par_phrase["rouge2_rappel"] < a_plat["rouge2_rappel"]
    # Les unigrammes ne traversent aucune frontiere : identiques dans les deux modes.
    for cle in CLES:
        assert par_phrase[f"rouge1_{cle}"] == pytest.approx(a_plat[f"rouge1_{cle}"])


def test_termes_par_phrase_projette_la_sortie_de_resumer_livre():
    phrases = [
        {"sent_id": 3, "texte": "Le chat dort.", "termes": ["le", "chat", "dormir"]},
        {"sent_id": 9, "texte": "Il pleut.", "termes": ["il", "pleuvoir"]},
    ]
    assert termes_par_phrase(phrases) == [["le", "chat", "dormir"], ["il", "pleuvoir"]]
    assert termes_par_phrase(phrases, champ="texte") == [
        list("Le chat dort."), list("Il pleut.")
    ]
    assert termes_par_phrase([]) == []


# ============================================================================
# 4. Agregation sur un corpus
# ============================================================================


def test_agreger_scores_moyenne_et_ecart_type():
    sortie = agreger_scores([{"x": 0.0}, {"x": 1.0}])
    assert sortie["n_resumes"] == 2.0
    assert sortie["x"] == pytest.approx(0.5)
    assert sortie["x_ecart_type"] == pytest.approx(0.5)


def test_agreger_scores_tolere_les_cles_manquantes():
    # Permet de melanger des resultats calcules avec et sans `avec_l`.
    sortie = agreger_scores([{"x": 0.0, "y": 1.0}, {"x": 1.0}])
    assert sortie["y"] == pytest.approx(1.0)
    assert sortie["y_ecart_type"] == pytest.approx(0.0)


def test_agreger_scores_sur_liste_vide():
    assert agreger_scores([]) == {"n_resumes": 0.0}


def test_resumer_metriques_resume_apparie_positionnellement():
    sortie = resumer_metriques_resume([["le", "chat"]], [["le", "chat", "dort"]])
    assert sortie["n_resumes"] == 1.0
    assert sortie["rouge1_precision"] == pytest.approx(1.0)
    assert sortie["rouge1_rappel"] == pytest.approx(2 / 3)


def test_resumer_metriques_resume_refuse_un_appariement_incoherent():
    with pytest.raises(ValueError, match="2 candidats pour 1 references"):
        resumer_metriques_resume([["a"], ["b"]], [["a"]])


# ============================================================================
# 5. BLEU : proprietes
# ============================================================================


def test_bleu_parfait_sur_candidat_identique():
    termes = ["le", "chat", "noir", "dort", "sur", "un", "tapis"]
    assert bleu(termes, [termes], lissage=False) == pytest.approx(1.0)


def test_bleu_penalise_la_brievete():
    ref = ["le", "chat", "noir", "dort", "sur", "un", "tapis", "rouge"]
    complet = bleu(ref, [ref])
    tronque = bleu(ref[:4], [ref])
    assert tronque < complet


def test_bleu_lissage_evite_le_zero_absorbant():
    # Recouvrement unigramme mais aucun 4-gramme commun : sans lissage le
    # score s'annule, ce qui arrive tout le temps sur des resumes courts.
    cand = ["le", "chat", "tapis", "pluie", "rouge"]
    ref = ["le", "chien", "tapis", "tombe", "rouge"]
    assert bleu(cand, ref and [ref], lissage=False) == 0.0
    assert bleu(cand, [ref], lissage=True) > 0.0


def test_bleu_valide_ses_arguments():
    with pytest.raises(ValueError):
        bleu(["a"], [])
    with pytest.raises(ValueError):
        bleu(["a"], [["a"]], n_max=4, poids=[0.5, 0.5])


def test_bleu_candidat_vide():
    assert bleu([], [["a", "b"]]) == 0.0


# ============================================================================
# 6. Comparaison aux implementations de reference
# ============================================================================


def _scorer_rouge(cles):
    rouge_scorer = pytest.importorskip(
        "rouge_score.rouge_scorer",
        reason="rouge-score non installe (cf. requirements-dev.txt)",
    )
    return rouge_scorer.RougeScorer(cles, use_stemmer=False, split_summaries=False)


def _texte(phrases) -> str:
    """Resume en phrases -> texte tel que `rouge-score` l'attend (une phrase par ligne)."""
    return "\n".join(" ".join(phrase) for phrase in phrases)


@pytest.mark.parametrize("tirage", range(30))
def test_rouge_1_2_lsum_identiques_a_rouge_score(tirage):
    scorer = _scorer_rouge(["rouge1", "rouge2", "rougeLsum"])
    rng = random.Random(1000 + tirage)
    cand = _phrases(rng, n_phrases=4, n_max=8)
    ref = _phrases(rng, n_phrases=5, n_max=10)

    attendu = scorer.score(_texte(ref), _texte(cand))
    # Mode compatible : rouge-score aplatit avant d'extraire les n-grammes.
    obtenu = rouge_complet_phrases(cand, ref, ngrammes_par_phrase=False)

    for nom, prefixe in (("rouge1", "rouge1"), ("rouge2", "rouge2"), ("rougeLsum", "rougeLsum")):
        ref_score = attendu[nom]
        assert obtenu[f"{prefixe}_precision"] == pytest.approx(ref_score.precision)
        assert obtenu[f"{prefixe}_rappel"] == pytest.approx(ref_score.recall)
        assert obtenu[f"{prefixe}_f1"] == pytest.approx(ref_score.fmeasure)

    direct = rouge_lsum(cand, ref)
    assert direct["f1"] == pytest.approx(attendu["rougeLsum"].fmeasure)


@pytest.mark.parametrize("tirage", range(30))
def test_rouge_l_identique_a_rouge_score(tirage):
    scorer = _scorer_rouge(["rougeL"])
    rng = random.Random(2000 + tirage)
    cand = [rng.choice(VOCAB) for _ in range(rng.randint(1, 12))]
    ref = [rng.choice(VOCAB) for _ in range(rng.randint(1, 12))]

    attendu = scorer.score(" ".join(ref), " ".join(cand))["rougeL"]
    obtenu = rouge_l(cand, ref)
    assert obtenu["precision"] == pytest.approx(attendu.precision)
    assert obtenu["rappel"] == pytest.approx(attendu.recall)
    assert obtenu["f1"] == pytest.approx(attendu.fmeasure)


# nltk previent bruyamment quand un ordre n'a aucun recouvrement et annule
# le score. C'est precisement le cas que `lissage=False` doit reproduire, et
# l'assertion verifie qu'on tombe sur le meme zero : le warning est attendu.
# Le `\s*` n'est pas cosmetique : le message nltk commence par un retour a la
# ligne, et pytest strip() le motif avant de le compiler, donc un "\n" ecrit
# en tete serait supprime et le filtre ne matcherait jamais.
@pytest.mark.filterwarnings(r"ignore:\s*The hypothesis contains:UserWarning")
@pytest.mark.parametrize("lissage", [False, True])
@pytest.mark.parametrize("tirage", range(20))
def test_bleu_identique_a_nltk(tirage, lissage):
    nltk_bleu = pytest.importorskip(
        "nltk.translate.bleu_score",
        reason="nltk non installe (cf. requirements-dev.txt)",
    )
    rng = random.Random(3000 + tirage)
    # Candidat d'au moins n_max termes, sinon un ordre n'est pas evaluable
    # et les deux implementations divergent sur la convention de sortie.
    cand = [rng.choice(VOCAB) for _ in range(rng.randint(4, 14))]
    refs = [[rng.choice(VOCAB) for _ in range(rng.randint(4, 14))] for _ in range(2)]

    # `lissage=True` correspond au add-1 sur les ordres n >= 2, soit method2
    # chez nltk (Lin & Och 2004), pas method1 qui est un epsilon.
    smoothing = nltk_bleu.SmoothingFunction().method2 if lissage else None
    attendu = nltk_bleu.sentence_bleu(refs, cand, smoothing_function=smoothing)
    assert bleu(cand, refs, lissage=lissage) == pytest.approx(attendu, abs=1e-9)
