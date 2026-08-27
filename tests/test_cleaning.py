"""
Tests de `pipeline.cleaning`.

Centres sur `couper_decorations`, dont le motif a fait bloquer une collecte.

Le bug etait un backtracking catastrophique : la classe de caracteres
contenait `\\s`, donc `\\n`, alors que le `$` multiligne et le `\\r?\\n?`
qui suivaient pretendaient consommer ce meme saut de ligne. Le moteur
devait essayer toutes les repartitions possibles, et le groupe etant
repete par `+`, leur nombre doublait tous les deux sauts de ligne.

Le cout ne se voyait que lorsque `\\Z` echouait, c'est-a-dire sur un texte
se terminant par de la prose plutot que par des decorations. Mesure sur
des lignes vides CRLF suivies de texte, avec l'ancien motif :

    12 lignes  0,001 s
    16 lignes  0,009 s
    20 lignes  0,145 s
    24 lignes  2,331 s
    28 lignes  > 10 s

Le test de terminaison ci-dessous utilise 2 000 lignes. L'ancien motif ne
l'aurait jamais fini ; le nouveau le traite en quelques millisecondes.
"""
from __future__ import annotations

import time

from pipeline.cleaning import couper_decorations, nettoyer


# Budget large devant le temps reel mesure (de l'ordre de la milliseconde),
# mais infiniment plus petit que ce que le motif fautif aurait demande.
BUDGET_SECONDES = 5.0


# ============================================================================
# Terminaison
# ============================================================================

def test_terminaison_sur_lignes_vides_crlf_suivies_de_prose():
    """Le cas exact qui bloquait : fin de texte non decorative."""
    piege = ("\r\n" * 2000) + "texte final sans decoration."

    debut = time.monotonic()
    resultat = couper_decorations(piege)
    duree = time.monotonic() - debut

    assert duree < BUDGET_SECONDES, f"{duree:.1f}s : backtracking de retour"
    assert resultat.endswith("texte final sans decoration.")


def test_terminaison_sur_bandes_decoratives_intercalees():
    """Variante avec de vraies decorations, toujours suivie de prose."""
    piege = ("   * * *   \r\n" * 1000) + "et le livre continue ici."

    debut = time.monotonic()
    resultat = couper_decorations(piege)
    duree = time.monotonic() - debut

    assert duree < BUDGET_SECONDES
    assert resultat.endswith("et le livre continue ici.")


def test_nettoyer_complet_termine():
    """`nettoyer()` enchaine tout : la chaine entiere doit rendre la main."""
    piege = "Un debut de livre.\r\n" + ("\r\n" * 1500) + "Une fin de livre."

    debut = time.monotonic()
    nettoyer(piege)
    assert time.monotonic() - debut < BUDGET_SECONDES


# ============================================================================
# Comportement
# ============================================================================

def test_bande_decorative_finale_retiree():
    texte = "Le corps du livre.\n\n***\n===\n___\n"
    assert couper_decorations(texte) == "Le corps du livre."


def test_bande_decorative_finale_retiree_en_crlf():
    """Les fichiers Gutenberg francais sont souvent en CRLF."""
    texte = "Le corps du livre.\r\n\r\n* * *\r\n"
    assert couper_decorations(texte) == "Le corps du livre."


def test_texte_sans_decoration_intact():
    texte = "Le corps du livre, sans rien apres."
    assert couper_decorations(texte) == texte


def test_decoration_au_milieu_conservee():
    """
    On ne coupe qu'en FIN de fichier. Un separateur au milieu appartient au
    corps du texte et doit survivre.
    """
    texte = "Premiere partie.\n\n* * *\n\nSeconde partie."
    assert couper_decorations(texte) == texte


def test_espaces_finaux_retires():
    assert couper_decorations("Le texte.\n\n   \n\t\n") == "Le texte."


def test_chaine_vide():
    assert couper_decorations("") == ""


def test_texte_entierement_decoratif():
    assert couper_decorations("***\n===\n") == ""
