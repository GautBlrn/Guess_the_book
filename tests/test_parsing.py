"""
Tests de `pipeline.parsing`.

`slugifier` fabrique les cles S3 sous lesquelles les livres sont stockes.
Une cle mal formee n'est pas une erreur visible : le livre s'ecrit quand
meme, sous un nom different de celui attendu. C'est exactement le mecanisme
qui produit des doublons dans le corpus, et un doublon compte pour deux
documents dans l'IDF, ce qui desavantage le livre a la recherche.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.parsing import (
    auteur_principal,
    construire_cle,
    deslug,
    parser_chemin,
    slugifier,
)


# ============================================================================
# slugifier
# ============================================================================

def test_accents_et_espaces():
    assert slugifier("Mémoires du duc de Rovigo") == "memoires_du_duc_de_rovigo"


def test_ponctuation_retiree():
    assert slugifier("Le comte de Monte-Cristo, Tome II") == \
        "le_comte_de_monte_cristo_tome_ii"


@pytest.mark.parametrize("entree,attendu", [
    ("Œuvres complètes", "oeuvres_completes"),
    ("Les œuvres de Molière", "les_oeuvres_de_moliere"),
    ("Le nœud gordien", "le_noeud_gordien"),
    ("Cæsar", "caesar"),
    ("ÆGYPTUS", "aegyptus"),
])
def test_ligatures_developpees(entree, attendu):
    """
    NFKD ne decompose ni « œ » ni « æ » : Unicode les traite comme des
    lettres a part entiere du francais, pas comme des ligatures
    typographiques a la maniere de « ﬁ ». Sans traitement explicite, le
    passage en ASCII les supprimait, et « Œuvres completes » devenait
    « uvres_completes ».
    """
    assert slugifier(entree) == attendu


def test_ligature_typographique_toujours_geree_par_nfkd():
    """Celle-la, NFKD la decompose : le correctif ne doit pas la casser."""
    assert slugifier("ﬁgure") == "figure"


def test_pas_de_separateurs_consecutifs():
    assert slugifier("Un  titre -- avec   des trous") == "un_titre_avec_des_trous"


def test_resultat_toujours_sur_pour_une_cle_s3():
    """Seuls des caracteres de mot et des underscores doivent survivre."""
    for titre in ["Œuvres, Tome I : « suite »", "L'Été 1900 !", "A/B (test)"]:
        slug = slugifier(titre)
        assert slug == slug.lower()
        assert "/" not in slug
        assert all(c.isalnum() or c == "_" for c in slug), slug


# ============================================================================
# auteur_principal
# ============================================================================

def test_auteur_dates_retirees():
    assert auteur_principal("Zola, Émile, 1840-1902") == "Zola, Émile"


def test_auteur_premier_seulement():
    assert auteur_principal("Dumas, Alexandre; Maquet, Auguste") == \
        "Dumas, Alexandre"


def test_auteur_role_entre_crochets_retire():
    assert auteur_principal("Hugo, Victor [Illustrator]") == "Hugo, Victor"


def test_auteur_absent():
    assert auteur_principal(pd.NA) is None


# ============================================================================
# Cles S3
# ============================================================================

def test_construire_puis_parser_est_un_aller_retour():
    cle = construire_cle(
        "annotations/", "novel", "zola_emile", "pg8560_le_docteur_pascal",
        ".parquet",
    )
    assert cle == "annotations/novel/zola_emile/pg8560_le_docteur_pascal.parquet"

    info = parser_chemin(cle, "annotations/")
    assert info["genre"] == "novel"
    assert info["auteur_slug"] == "zola_emile"
    assert info["livre_slug"] == "pg8560_le_docteur_pascal"


def test_extension_avec_ou_sans_point():
    a = construire_cle("raw/", "novel", "a", "b", ".txt")
    b = construire_cle("raw/", "novel", "a", "b", "txt")
    assert a == b


def test_deslug_retire_le_prefixe_gutenberg():
    assert deslug("pg8560_le_docteur_pascal") == "Le Docteur Pascal"


def test_cle_trop_courte_leve():
    with pytest.raises(ValueError):
        parser_chemin("annotations/orphelin.parquet", "annotations/")
