"""
Tokenisation : normalisation typographique + decoupage spaCy.

Le modele `spacy.blank("fr")` ne charge que le tokenizer (pas de POS,
parser, NER) -- suffisant pour cette etape, qui cherche juste a
decouper le texte en mots en gerant les contractions francaises
(`l'echo`, `qu'il`) et les traits d'union pronominaux (`dit-elle`).

La table NORMALISATION harmonise les caracteres typographiques
(apostrophes courbes, tirets cadratins, guillemets francais) pour
que le corpus parle le meme alphabet a chaque etape ulterieure.
La meme table est utilisee dans `annotation.py`.
"""
from functools import lru_cache

import spacy
import re


# Table de normalisation typographique.
# But : que tous les livres parlent le meme alphabet, pour ne pas
# fragmenter les frequences des etapes ulterieures.
NORMALISATION = str.maketrans({
    "\u2019": "'",   # apostrophe courbe droite
    "\u2018": "'",   # apostrophe courbe gauche
    "\u02BC": "'",   # lettre modificatrice apostrophe
    "\u2013": "-",   # tiret demi-cadratin
    "\u2014": "-",   # tiret cadratin
    "\u00AB": '"',   # guillemet ouvrant francais
    "\u00BB": '"',   # guillemet fermant francais
    "\u201C": '"',   # guillemet ouvrant anglais
    "\u201D": '"',   # guillemet fermant anglais
    "\u2026": "...", # points de suspension
    "\u00A0": " ",   # espace insecable
})

_TIRET_DIALOGUE = re.compile(r'--(?=\S)')

def normaliser(texte):
    """Applique la table de normalisation typographique."""
    texte = texte.translate(NORMALISATION)
    texte = _TIRET_DIALOGUE.sub('-- ', texte)
    return texte


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Tokenizer spaCy francais (lazy load, mis en cache)."""
    nlp = spacy.blank("fr")
    nlp.max_length = 10_000_000
    return nlp


def tokeniser(texte):
    """
    Decoupe un texte en tokens spaCy. Les espaces purs sont filtres.

    Retourne une liste de chaines (le `.text` de chaque token).
    """
    nlp = _get_tokenizer()
    texte = normaliser(texte)
    doc = nlp(texte)
    return [t.text for t in doc if not t.is_space]
