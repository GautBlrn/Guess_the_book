"""
Tests du suivi d'avancement de `pipeline.corpus`.

Une estimation de temps restant n'est utile que si elle est plausible. Deux
facons de la rendre trompeuse ont ete rencontrees pour de vrai sur le corpus
de 291 livres, et ce sont elles que ces tests verrouillent :

1. la moyenne depuis le depart, empoisonnee par la lenteur initiale des
   acces S3 : elle annoncait 8 616 s restantes la ou il en fallait 170 ;
2. la moyenne sur le seul dernier palier, prise en defaut par les rafales
   du chargement parallele : deux paliers dans le meme instant donnaient un
   debit de 129 055 livres/s.

Aucun acces reseau : l'horloge est remplacee.
"""
from __future__ import annotations

import pytest

from pipeline import corpus as mod


@pytest.fixture
def horloge(monkeypatch):
    """Horloge pilotee, pour que les debits soient deterministes."""
    etat = {"t": 1000.0}
    monkeypatch.setattr(mod.time, "time", lambda: etat["t"])

    def avancer(secondes):
        etat["t"] += secondes
    return avancer


def lignes(capsys):
    return [l for l in capsys.readouterr().out.splitlines() if l.strip()]


def restant_de(ligne):
    """Extrait le nombre de secondes restantes annonce."""
    return float(ligne.split("~")[1].split("s")[0])


def debit_de(ligne):
    return float(ligne.split(",")[1].strip().rstrip("/s"))


# ============================================================================
# Le demarrage lent ne doit pas empoisonner l'estimation
# ============================================================================

def test_le_demarrage_lent_est_oublie(horloge, capsys):
    """
    Profil reel : 10 livres en 307 s, puis 0,65 s par livre. Apres quelques
    paliers en regime de croisiere, l'estimation doit refleter ce regime et
    non la moyenne generale.
    """
    suivi = mod._Progression(291)

    horloge(306.6)
    suivi.point(10)

    for n in range(20, 60, 10):
        horloge(6.5)
        suivi.point(n)

    sorties = lignes(capsys)
    # Le tout premier point ne peut rien savoir de mieux que la moyenne.
    assert restant_de(sorties[0]) > 5000
    # Une fois la fenetre remplie de paliers rapides, l'estimation redescend.
    assert restant_de(sorties[-1]) < 400


def test_estimation_stable_en_regime_constant(horloge, capsys):
    suivi = mod._Progression(200)
    for n in range(10, 110, 10):
        horloge(10.0)     # 1 livre/s, constant
        suivi.point(n)

    sorties = lignes(capsys)
    for ligne in sorties[3:]:
        assert debit_de(ligne) == pytest.approx(1.0, abs=0.01)
    # A 100/200 livres et 1/s, il reste 100 s.
    assert restant_de(sorties[-1]) == pytest.approx(100.0, abs=1.0)


# ============================================================================
# Les rafales du chargement parallele
# ============================================================================

def test_rafale_ne_produit_pas_de_debit_absurde(horloge, capsys):
    """
    Le cas observe : deux paliers dans le meme instant. Sans lissage, le
    debit valait 129 055 livres/s et le temps restant tombait a 0.
    """
    suivi = mod._Progression(291)

    horloge(106.9)
    suivi.point(10)
    horloge(0.0)          # rafale : le palier suivant arrive sans delai
    suivi.point(20)
    horloge(28.6)
    suivi.point(30)

    for ligne in lignes(capsys):
        assert debit_de(ligne) < 100, f"debit absurde : {ligne}"


def test_fenetre_entierement_instantanee_ne_divise_pas_par_zero(horloge,
                                                                capsys):
    """Garde-fou : si toute la fenetre tombe dans le meme instant."""
    suivi = mod._Progression(100)
    for n in (10, 20, 30, 40):
        horloge(0.0)
        suivi.point(n)

    sorties = lignes(capsys)
    assert any("inconnu" in l for l in sorties)
    for ligne in sorties:
        assert "inf" not in ligne.lower()


# ============================================================================
# Forme de la sortie
# ============================================================================

def test_la_ligne_porte_l_avancement(horloge, capsys):
    suivi = mod._Progression(291)
    horloge(10.0)
    suivi.point(30)

    ligne = lignes(capsys)[0]
    assert "30/291" in ligne
    assert "livres" in ligne


# ============================================================================
# `limite` doit couper avant les telechargements
# ============================================================================

@pytest.fixture
def faux_s3(monkeypatch):
    """
    Remplace S3 par un double qui compte les Parquet reellement telecharges.

    C'est ce compteur qui porte le test : `limite` n'a d'interet que si elle
    evite le reseau. L'appliquer apres coup, sur la liste rendue, donnerait
    le meme resultat visible en ayant paye le corpus entier.
    """
    import pandas as pd

    etat = {"telecharges": []}
    cles = [f"annotations/novel/auteur_{i}/pg{i}_livre.parquet" for i in range(50)]

    monkeypatch.setattr(
        mod.storage, "list_objects",
        lambda prefix, suffix=None: [{"Key": k} for k in cles],
    )

    def faux_parquet(cle):
        etat["telecharges"].append(cle)
        return pd.DataFrame({
            "sent_id":  [0, 0, 1],
            "lemma":    ["chat", "dormir", "tapis"],
            "text":     ["chats", "dort", "tapis"],
            "is_alpha": [True, True, True],
            "is_stop":  [False, False, False],
            "is_punct": [False, False, False],
        })

    monkeypatch.setattr(mod.storage, "get_parquet", faux_parquet)
    monkeypatch.setattr(mod, "extraire_termes", lambda df, champ: ["x"])
    return etat


def test_limite_evite_les_telechargements(faux_s3):
    corpus = mod.charger_corpus(limite=5, verbose=False)
    assert len(corpus) == 5
    assert len(faux_s3["telecharges"]) == 5, \
        "les 45 autres livres ne doivent pas etre telecharges"


def test_sans_limite_tout_est_charge(faux_s3):
    corpus = mod.charger_corpus(verbose=False)
    assert len(corpus) == 50
    assert len(faux_s3["telecharges"]) == 50


def test_limite_superieure_au_corpus(faux_s3):
    corpus = mod.charger_corpus(limite=999, verbose=False)
    assert len(corpus) == 50


def test_limite_sur_iter_corpus(faux_s3):
    livres = list(mod.iter_corpus(limite=3, verbose=False))
    assert len(livres) == 3
    assert len(faux_s3["telecharges"]) == 3


def test_ordre_preserve_malgre_le_parallelisme(faux_s3):
    """
    L'ordre des cles S3 est celui des lignes de la matrice TF-IDF et de
    l'index servi par l'API. Le chargement parallele ne doit pas le
    melanger.
    """
    corpus = mod.charger_corpus(limite=20, verbose=False)
    attendus = [f"pg{i}_livre" for i in range(20)]
    assert [c["livre_slug"] for c in corpus] == attendus


def test_ecoule_compte_depuis_le_debut(horloge, capsys):
    """
    Le temps ecoule reste cumule, meme si le debit est sur une fenetre :
    c'est la duree reelle du chargement, et elle ne doit pas etre lissee.
    """
    suivi = mod._Progression(100)
    horloge(50.0)
    suivi.point(10)
    horloge(25.0)
    suivi.point(20)

    sorties = lignes(capsys)
    assert "75.0s ecoulees" in sorties[1]
