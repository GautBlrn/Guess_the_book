"""
Selection et validation des livres candidats a l'entree du corpus.

Deux besoins apparaissent des que le corpus depasse la vingtaine de livres,
et qu'on ne peut plus regarder chaque ajout a l'oeil.

SELECTION. `scripts/upload_corpus.py` tirait N livres au hasard dans le
catalogue (`df.sample`). A 10 livres c'est sans consequence ; a 300 le
tirage uniforme recopie simplement la forme du catalogue, qui est tres
desequilibree :

    novel 710, biography 385, historical_fiction 302, essays 238,
    poetry 192, travel 185, humour 183, romance 153, adventure 152,
    philosophy 150, drama 140, short_stories 109, scifi_fantasy 84,
    mystery 60, children 42, mythology 42        (3127 livres FR eligibles)

Un tirage uniforme de 300 donnerait ~68 romans et ~4 livres de mythologie.
Or la tache est d'identifier un livre parmi tous les autres : ce qui la rend
difficile, ce sont les VOISINS proches, donc les paquets de livres du meme
genre et du meme registre. Un corpus qui empile les romans et neglige les
genres rares mesure une difficulte qui n'est pas celle du produit.
`selectionner_stratifie` impose donc un quota par genre.

VALIDATION. `telecharger` ne verifie que `len(contenu) > 1000`. Ca laisse
passer trois choses qui polluent silencieusement le corpus :

  - les textes qui ne sont pas en francais malgre le catalogue. Le champ
    `Language` de `pg_catalog.csv` decrit l'oeuvre, pas le fichier : editions
    bilingues, apparats critiques anglais autour d'un texte francais,
    entrees mal renseignees.
  - les textes trop courts pour porter la tache : sous quelques milliers de
    mots, un extrait de 70 termes represente une part enorme du livre, et le
    resume MMR n'a pas assez de phrases candidates pour choisir.
  - les textes que `nettoyer()` ne sait pas decouper, qui arrivent dans le
    corpus avec leur en-tete editorial et leur table des matieres.

Les trois portes sont dans `valider_texte`, qui renvoie la raison du rejet
plutot qu'un booleen : a 300 livres on veut savoir CE QUI a ete ecarte, pas
seulement combien.
"""
import re

from pipeline.cleaning import nettoyer
from pipeline.config import COLLECTE_PARAMS


# ============================================================================
# Detection de la langue reelle du texte
# ============================================================================
#
# On compte des mots-outils, pas du vocabulaire. Les mots pleins d'un roman
# francais traduit de l'anglais restent francais, mais un APPAREIL editorial
# anglais (preface, notes, licence) trahit sa langue par ses articles et ses
# auxiliaires, qui sont les mots les plus frequents de n'importe quel texte.
#
# Les deux listes ne contiennent que des formes SANS ambiguite entre les deux
# langues. Sont volontairement absents : "on", "or", "a", "no", "son", "sur",
# "pour", "en", "as", "the" mis a part rien de commun -- tout mot qui existe
# des deux cotes brouille le compte au lieu de l'affiner.

MOTS_FRANCAIS = frozenset("""
    le la les un une des du de et est sont etait etaient sera
    dans que qui quoi dont ou avec sans sous chez vers depuis pendant
    ce cet cette ces celui celle ceux
    il elle ils elles nous vous lui leur leurs
    mais donc car ainsi alors puis encore deja jamais toujours
    tres bien aussi tout tous toute toutes
    mon ma mes ton ta tes notre votre nos vos
    plus moins beaucoup peu assez trop
    etre avoir faire dire fait dit avait etaient furent
    quand comme parce afin pourquoi
""".split())

MOTS_ANGLAIS = frozenset("""
    the and of to in that is was with for his her he she it as at by this
    but not are from they you have had which were their there would been
    when will about an if then than them him its our who what how where why
    some any more most such only into over after before should could must
    being does did upon very much many every these those through while
    said says shall may might because both each other than once
""".split())

# Sous ce nombre de mots-outils reperes, le compte n'est pas fiable : on
# refuse plutot que de deviner. Un vrai livre en depasse largement le seuil.
MOTS_MINIMAUX = 200

MOT = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+")


def part_anglaise(texte):
    """
    Part de mots-outils anglais parmi les mots-outils reconnus, sur le corps.

    Renvoie None si le texte contient trop peu de mots-outils pour trancher.

    On echantillonne les 60 % centraux et pas le texte entier : l'en-tete et
    le pied Gutenberg sont en anglais meme sur un livre francais (licence,
    « Produced by », colophon). Les compter reviendrait a penaliser tous les
    livres de la meme quantite fixe, ce qui deplace le seuil sans rien
    apprendre. Le milieu, lui, est du texte d'auteur.
    """
    milieu = texte[int(len(texte) * 0.2):int(len(texte) * 0.8)].lower()

    francais = anglais = 0
    for mot in MOT.findall(milieu):
        if mot in MOTS_FRANCAIS:
            francais += 1
        elif mot in MOTS_ANGLAIS:
            anglais += 1

    total = francais + anglais
    if total < MOTS_MINIMAUX:
        return None
    return anglais / total


def texte_en_francais(texte, seuil=None):
    """Vrai si le corps du texte est majoritairement francais."""
    if seuil is None:
        seuil = COLLECTE_PARAMS["part_anglaise_max"]
    part = part_anglaise(texte)
    if part is None:
        return False
    return part < seuil


# ============================================================================
# Validation d'un candidat
# ============================================================================

def valider_texte(contenu, genre=None, params=None):
    """
    Decide si un texte brut telecharge merite d'entrer dans le corpus.

    Renvoie `(True, None)` si le texte passe, `(False, raison)` sinon, ou
    `raison` est une chaine courte utilisable comme cle de comptage dans le
    bilan de collecte.

    Quatre portes, dans l'ordre du moins cher au plus cher :

    1. taille brute -- rejette les notices, les fragments et les compendiums
       multi-volumes. Le plafond n'est pas cosmetique : l'annotation spaCy
       est lineaire en longueur et un fichier de 5 Mo monopolise le pipeline
       plusieurs minutes pour un seul livre du corpus.
    2. langue du corps -- voir `part_anglaise`.
    3. nettoyage -- on fait tourner `nettoyer()` pour de vrai. C'est la seule
       facon de savoir s'il sait traiter ce fichier, et ca coute quelques
       millisecondes contre plusieurs secondes d'annotation en aval.
    4. taille apres nettoyage, en absolu et en proportion du brut. Un texte
       qui perd l'essentiel de son volume au nettoyage signale un fichier
       dont la structure a piege les ancres (en-tete non reconnu suivi d'un
       « FIN » precoce, par exemple) : le peu qui reste n'est pas le livre.

    Le marqueur `*** START OF ... ***` n'est deliberement PAS exige, alors
    qu'il serait le test le plus evident. `nettoyer()` reconnait aussi les
    en-tetes « Ebooks libres et gratuits », Gallica et Internet Archive, qui
    sont surrepresentes dans le fonds francais et n'ont pas ce marqueur.
    L'exiger reviendrait a jeter une partie des livres francais les mieux
    traites par le pipeline. La porte 4 verifie ce qui compte vraiment, a
    savoir que le nettoyage a produit un texte plausible.
    """
    if params is None:
        params = COLLECTE_PARAMS

    if not contenu:
        return False, "vide"

    n_brut = len(contenu)
    taille_min = params["taille_min_par_genre"].get(
        genre, params["taille_min_defaut"]
    )
    if n_brut < taille_min:
        return False, "trop_court"
    if n_brut > params["taille_max"]:
        return False, "trop_long"

    if not texte_en_francais(contenu, params["part_anglaise_max"]):
        return False, "pas_francais"

    propre = nettoyer(contenu)
    n_propre = len(propre)
    if n_propre < taille_min:
        return False, "vide_apres_nettoyage"
    if n_propre / n_brut < params["ratio_nettoyage_min"]:
        return False, "nettoyage_suspect"

    return True, None


# ============================================================================
# Selection stratifiee
# ============================================================================

def selectionner_stratifie(df, n_par_genre, seed=42, genres=None):
    """
    Tire jusqu'a `n_par_genre` livres dans chaque genre.

    `df` est le catalogue deja filtre (langue, auteur present, genre
    detectable). Renvoie un DataFrame trie par genre puis par titre, avec un
    ordre reproductible a seed fixee.

    Les genres qui comptent moins de `n_par_genre` livres eligibles donnent
    tout ce qu'ils ont : on ne complete pas avec un autre genre. Le quota est
    un plafond destine a empecher les genres abondants d'ecraser les autres,
    pas un plancher qu'il faudrait atteindre en trichant.

    `groupby(...).sample()` ferait la meme chose en une ligne mais leve des
    que le quota depasse l'effectif d'un groupe, ce qui est precisement le
    cas normal ici.
    """
    if genres is None:
        genres = sorted(df["genre"].dropna().unique())

    morceaux = []
    for genre in genres:
        pool = df[df["genre"] == genre]
        if pool.empty:
            continue
        morceaux.append(pool.sample(min(n_par_genre, len(pool)), random_state=seed))

    if not morceaux:
        return df.iloc[0:0]

    import pandas as pd
    return pd.concat(morceaux).sort_values(["genre", "Title"])
