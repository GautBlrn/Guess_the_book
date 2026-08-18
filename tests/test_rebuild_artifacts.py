"""
Tests du nettoyage des resumes orphelins.

Cette fonction SUPPRIME des objets S3 a partir d'une difference
d'ensembles. Si l'un des deux ensembles est faux, elle efface des donnees
valides sans rien signaler. Les tests portent donc moins sur le cas
nominal que sur les situations ou elle doit refuser d'agir.

Aucun acces S3 : `storage.list_objects` et `storage.delete_object` sont
remplaces, et les suppressions sont enregistrees au lieu d'etre faites.
"""
from __future__ import annotations

import pytest

from scripts import rebuild_artifacts as ra


def _livre(slug, genre="roman", auteur="auteur_x"):
    return {"livre_slug": slug, "auteur_slug": auteur, "genre": genre,
            "livre": slug, "auteur": auteur}


def _cle(slug, genre="roman", auteur="auteur_x"):
    return f"summaries/{genre}/{auteur}/{slug}.txt"


@pytest.fixture
def s3(monkeypatch):
    """Faux S3 : `etat` contient les cles presentes, `supprimes` les appels."""
    etat = {"cles": [], "supprimes": []}

    monkeypatch.setattr(
        ra.storage, "list_objects",
        lambda prefix, suffix=None: [{"Key": k} for k in etat["cles"]],
    )
    monkeypatch.setattr(
        ra.storage, "delete_object", lambda cle: etat["supprimes"].append(cle)
    )
    return etat


def test_cle_attendue_suit_la_convention_d_ecriture(s3):
    corpus = [_livre("pg1_titre", genre="adventure", auteur="dumas")]
    assert ra.cles_resumes_attendues(corpus) == {
        "summaries/adventure/dumas/pg1_titre.txt"
    }


def test_aucun_orphelin_ne_supprime_rien(s3):
    corpus = [_livre("pg1"), _livre("pg2")]
    s3["cles"] = [_cle("pg1"), _cle("pg2")]
    assert ra.nettoyer_resumes_orphelins(corpus, appliquer=True) == []
    assert s3["supprimes"] == []


def test_orphelin_de_reclassement_est_supprime(s3):
    # Le cas nominal : GENRE_OVERRIDES a deplace pg1 vers `adventure`,
    # l'ancienne cle `historical_fiction` traine.
    corpus = [_livre("pg1", genre="adventure")]
    s3["cles"] = [_cle("pg1", genre="adventure"),
                  _cle("pg1", genre="historical_fiction")]
    orphelins = ra.nettoyer_resumes_orphelins(corpus, appliquer=True)
    assert orphelins == [_cle("pg1", genre="historical_fiction")]
    assert s3["supprimes"] == [_cle("pg1", genre="historical_fiction")]


def test_par_defaut_liste_sans_supprimer(s3):
    corpus = [_livre("pg1", genre="adventure")]
    s3["cles"] = [_cle("pg1", genre="adventure"),
                  _cle("pg1", genre="historical_fiction")]
    orphelins = ra.nettoyer_resumes_orphelins(corpus)
    assert len(orphelins) == 1
    assert s3["supprimes"] == [], "le defaut ne doit jamais supprimer"


def test_refuse_si_un_resume_attendu_manque(s3):
    # Le scenario dangereux : --skip-summaries apres un changement de
    # genre. La nouvelle cle n'a jamais ete ecrite, donc supprimer
    # l'ancienne laisserait le livre sans aucun resume.
    corpus = [_livre("pg1", genre="adventure")]
    s3["cles"] = [_cle("pg1", genre="historical_fiction")]
    orphelins = ra.nettoyer_resumes_orphelins(corpus, appliquer=True)
    assert orphelins == [_cle("pg1", genre="historical_fiction")]
    assert s3["supprimes"] == [], "la seule copie restante ne doit pas partir"


def test_refuse_si_trop_d_orphelins(s3):
    # Corpus charge partiellement : un seul livre vu, huit resumes en
    # place. Le ratio trahit le probleme.
    corpus = [_livre("pg1")]
    s3["cles"] = [_cle("pg1")] + [_cle(f"pg{i}") for i in range(2, 10)]
    ra.nettoyer_resumes_orphelins(corpus, appliquer=True)
    assert s3["supprimes"] == []


def test_corpus_vide_leve(s3):
    s3["cles"] = [_cle("pg1"), _cle("pg2")]
    with pytest.raises(ValueError, match="Corpus vide"):
        ra.nettoyer_resumes_orphelins([], appliquer=True)
    assert s3["supprimes"] == []
