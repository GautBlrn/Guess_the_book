"""
Tests de `pipeline.collecte`.

Ce module decide ce qui entre dans le corpus. Ses erreurs sont silencieuses
par nature : un texte anglais accepte ne fait rien planter, il degrade juste
l'IDF et les resultats de recherche sans que rien ne le signale. Les tests
portent donc surtout sur les REJETS, et sur le fait que chaque porte rejette
pour la bonne raison -- un rejet correct pour un mauvais motif rendrait le
bilan de collecte trompeur.

Aucun acces reseau ni S3 : les textes sont fabriques ici.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.collecte import (
    part_anglaise,
    selectionner_stratifie,
    texte_en_francais,
    valider_texte,
)
from pipeline.config import COLLECTE_PARAMS


# ============================================================================
# Textes de test
# ============================================================================

# Des phrases riches en mots-outils, comme n'importe quelle prose. Repetees
# pour depasser a la fois le plancher de taille et le minimum de mots-outils
# necessaire pour que le compte de langue soit juge fiable.
PHRASE_FR = (
    "Il etait une fois dans la ville que les gens du quartier appelaient "
    "encore le vieux port, un homme qui ne sortait jamais avant la nuit. "
    "On disait de lui beaucoup de choses, mais personne ne savait vraiment "
    "ce qu il faisait de ses journees, et cela suffisait a nourrir toutes "
    "les conversations du dimanche. "
)
PHRASE_EN = (
    "It was in the town that the people of the district still called the "
    "old harbour, a man who would not go out before the night had come. "
    "They said many things about him, but nobody really knew what he did "
    "with his days, and that was enough to feed all the conversations of "
    "the week. "
)


def texte(phrase, taille):
    """Repete `phrase` jusqu'a depasser `taille` caracteres."""
    return (phrase * (taille // len(phrase) + 2))[:taille]


TAILLE_OK = COLLECTE_PARAMS["taille_min_defaut"] + 10_000


# ============================================================================
# Detection de langue
# ============================================================================

def test_francais_reconnu():
    assert texte_en_francais(texte(PHRASE_FR, TAILLE_OK))


def test_anglais_rejete():
    assert not texte_en_francais(texte(PHRASE_EN, TAILLE_OK))


def test_les_deux_populations_sont_franchement_separees():
    """
    La porte n'a d'interet que si les deux populations ne se touchent pas.
    Sur le corpus reel, le francais plafonne a 0,004 et l'anglais depasse
    0,99 : le seuil est place dans un vide de deux ordres de grandeur.
    """
    fr = part_anglaise(texte(PHRASE_FR, TAILLE_OK))
    en = part_anglaise(texte(PHRASE_EN, TAILLE_OK))
    assert fr < 0.05
    assert en > 0.90
    assert fr < COLLECTE_PARAMS["part_anglaise_max"] < en


def test_texte_trop_court_pour_decider():
    """
    Sous le minimum de mots-outils, `part_anglaise` rend None et le texte
    est refuse : on ne devine pas sur un echantillon trop maigre.
    """
    assert part_anglaise("Bonjour.") is None
    assert not texte_en_francais("Bonjour.")


def test_seuls_les_60_pourcent_centraux_comptent():
    """
    Un en-tete et un pied anglais, comme en porte tout fichier Gutenberg,
    ne doivent pas faire basculer la decision sur un livre francais.
    """
    corps = texte(PHRASE_FR, TAILLE_OK)
    entoure = texte(PHRASE_EN, 3_000) + corps + texte(PHRASE_EN, 3_000)
    assert texte_en_francais(entoure)


# ============================================================================
# Portes de validation
# ============================================================================

def test_texte_valide_accepte():
    ok, raison = valider_texte(texte(PHRASE_FR, TAILLE_OK), genre="novel")
    assert ok
    assert raison is None


def test_texte_vide():
    assert valider_texte("", genre="novel") == (False, "vide")
    assert valider_texte(None, genre="novel") == (False, "vide")


def test_trop_court():
    petit = texte(PHRASE_FR, COLLECTE_PARAMS["taille_min_defaut"] - 1_000)
    ok, raison = valider_texte(petit, genre="novel")
    assert not ok
    assert raison == "trop_court"


def test_trop_long():
    enorme = texte(PHRASE_FR, COLLECTE_PARAMS["taille_max"] + 1_000)
    ok, raison = valider_texte(enorme, genre="novel")
    assert not ok
    assert raison == "trop_long"


def test_pas_francais():
    ok, raison = valider_texte(texte(PHRASE_EN, TAILLE_OK), genre="novel")
    assert not ok
    assert raison == "pas_francais"


def test_ordre_des_portes_la_taille_avant_la_langue():
    """
    Un texte anglais trop court doit etre rejete sur la TAILLE, pas sur la
    langue : les portes vont du controle le moins cher au plus cher, et le
    bilan de collecte doit refleter cet ordre pour rester lisible.
    """
    court_en = texte(PHRASE_EN, 5_000)
    ok, raison = valider_texte(court_en, genre="novel")
    assert not ok
    assert raison == "trop_court"


def test_seuil_abaisse_pour_les_genres_courts():
    """
    Un recueil de 20 000 caracteres est un fragment pour un roman et une
    oeuvre complete pour de la poesie.
    """
    court = texte(PHRASE_FR, 20_000)
    assert valider_texte(court, genre="novel") == (False, "trop_court")
    ok, raison = valider_texte(court, genre="poetry")
    assert ok
    assert raison is None


def test_genre_inconnu_retombe_sur_le_seuil_par_defaut():
    court = texte(PHRASE_FR, 20_000)
    assert valider_texte(court, genre="genre_inexistant") == (False, "trop_court")
    assert valider_texte(court, genre=None) == (False, "trop_court")


def test_nettoyage_qui_devore_le_texte():
    """
    Un « FIN » precoce fait couper `nettoyer()` au tout debut du fichier.
    Le texte restant est trop maigre pour etre le livre : la derniere porte
    doit l'attraper, meme si le texte brut passait toutes les precedentes.
    """
    piege = texte(PHRASE_FR, 5_000) + "\nFIN.\n" + texte(PHRASE_FR, TAILLE_OK)
    ok, raison = valider_texte(piege, genre="novel")
    assert not ok
    assert raison in {"vide_apres_nettoyage", "nettoyage_suspect"}


# ============================================================================
# Selection stratifiee
# ============================================================================

@pytest.fixture
def catalogue():
    """Catalogue synthetique deliberement desequilibre, comme le vrai."""
    lignes = []
    for genre, n in (("novel", 50), ("poetry", 8), ("mystery", 20)):
        for i in range(n):
            lignes.append({
                "Text#": f"{genre[:3]}{i}",
                "Title": f"{genre} numero {i:03d}",
                "auteur_clean": f"Auteur {i % 7}",
                "genre": genre,
            })
    return pd.DataFrame(lignes)


def test_quota_plafonne_les_genres_abondants(catalogue):
    selection = selectionner_stratifie(catalogue, n_par_genre=10)
    comptes = selection["genre"].value_counts()
    assert comptes["novel"] == 10
    assert comptes["mystery"] == 10


def test_genre_moins_fourni_donne_tout_ce_qu_il_a(catalogue):
    """8 poemes disponibles pour un quota de 10 : on prend les 8, sans
    completer avec un autre genre."""
    selection = selectionner_stratifie(catalogue, n_par_genre=10)
    assert selection["genre"].value_counts()["poetry"] == 8
    assert len(selection) == 10 + 8 + 10


def test_selection_reproductible(catalogue):
    a = selectionner_stratifie(catalogue, n_par_genre=5, seed=42)
    b = selectionner_stratifie(catalogue, n_par_genre=5, seed=42)
    assert list(a["Text#"]) == list(b["Text#"])


def test_seed_differente_change_le_tirage(catalogue):
    a = selectionner_stratifie(catalogue, n_par_genre=5, seed=1)
    b = selectionner_stratifie(catalogue, n_par_genre=5, seed=2)
    assert list(a["Text#"]) != list(b["Text#"])


def test_restriction_a_certains_genres(catalogue):
    selection = selectionner_stratifie(
        catalogue, n_par_genre=5, genres=["poetry", "mystery"],
    )
    assert set(selection["genre"]) == {"poetry", "mystery"}


def test_genre_absent_du_catalogue_est_ignore(catalogue):
    selection = selectionner_stratifie(
        catalogue, n_par_genre=5, genres=["poetry", "genre_inexistant"],
    )
    assert set(selection["genre"]) == {"poetry"}


def test_catalogue_vide(catalogue):
    vide = catalogue.iloc[0:0]
    assert selectionner_stratifie(vide, n_par_genre=5).empty


def test_pas_de_doublon_dans_la_selection(catalogue):
    """
    Un livre ne doit apparaitre qu'une fois. Un doublon compte pour deux
    documents dans l'IDF, ce qui abaisse le poids des termes propres au
    livre et le desavantage a la recherche.
    """
    selection = selectionner_stratifie(catalogue, n_par_genre=10)
    assert selection["Text#"].is_unique
