"""
Nettoyage des textes Gutenberg (v3.1).

Cinq étapes :
- `couper_debut` : retire l'en-tete (marqueurs *** START ***, lignes
  Produced by sous toutes leurs variantes, references gallica/gutenberg.org,
  notes de transcription, mentions Internet Archive, etc.).
- `couper_fin` : retire le pied de page Gutenberg explicite, puis coupe
  après le DERNIER 'FIN' du texte.
- `retirer_annexes_post_fin` : APRES un FIN explicite, retire les blocs
  annexes connus (TABLE DES MATIERES finale, ARBRE GENEALOGIQUE,
  ACHEVE D'IMPRIMER, IMPRIMERIE X). Ne s'active QUE si un FIN a été
  trouve, pour éviter de couper trop sur les livres sans FIN explicite.
- `retirer_marqueurs_inline` : retire [Illustration], [Note du traducteur],
  etc. dans le corps du texte.
- `retirer_typographie_gutenberg` : retire les italiques `_mot_` et
  les separateurs `* * *`.
- `isoler_titres` : entoure les titres de chapitre par des doubles
  sauts de ligne pour forcer la segmentation spaCy.
- `couper_decorations` : enlève les bandes de #, *, =, -, _, ~, +
  collées a la toute fin.

`nettoyer(texte)` enchaîne tout ça.
"""
import re

from pipeline.config import CLEAN_PARAMS


# ============================================================================
# DEBUT : ancres pour fin d'en-tete editorial
# ============================================================================

HEADER_END_ANCHORS = [
    # --- Marqueurs Gutenberg standards ---
    re.compile(
        r"\*{2,}\s*START OF TH(?:E|IS) PROJECT GUTENBERG.*?\*{2,}",
        re.IGNORECASE | re.DOTALL,
    ),

    # --- "Produced by ..." sous toutes ses formes ---
    re.compile(
        r"Produced by.*?Proofread(?:ing\s+Team|ers)\.?",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Produced by[\s\S]{0,300}?Distributed\s+Proofreading[\s\S]{0,400}?\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Produced by Ebooks libres et gratuits[\s\S]{0,400}?\.(?=\s*\n\s*\n)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?im)^Produced by\s+[A-Z][\w\.\-' ]{2,60}(?:\s+and\s+[\w\.\-' ]+)?\s*$"
    ),

    # --- Mentions diverses ---
    re.compile(
        r"(?im)^.*?this text is also available at\s+https?://\S+.*$"
    ),
    re.compile(r"(?im)^.*gallica\.bnf\.fr.*$"),
    re.compile(r"(?im)^.*www\.gutenberg\.\w+.*$"),
    re.compile(r"(?im)^.*ebooksgratuits\.com.*$"),
    re.compile(r"(?im)^.*archive\.org/details/.*$"),
    re.compile(r"(?im)^.*www\.pgdp\.net.*$"),
    re.compile(
        r"(?im)^.*Biblioth[eè]que nationale de France\s*\(?BnF\b.*$"
    ),
    re.compile(r"(?im)^This (?:file|e?Book) was produced.*$"),
    re.compile(
        r"Images of the original pages[^.]*\.\s*See\s+https?://\S+",
        re.IGNORECASE | re.DOTALL,
    ),

    # --- Notes de transcription multilignes ---
    re.compile(
        r"(?ims)^\s*Note sur la transcription\s*[:.][^\n]*"
        r"(?:\n(?!\s*$)[^\n]*)*"
    ),
    re.compile(
        r"(?ims)^\s*Cette version num[eé]ris[eé]e[^\n]*"
        r"(?:\n(?!\s*$)[^\n]*)*"
    ),
    # "Au lecteur." + mentions techniques jusqu'a un titre majuscules
    re.compile(
        r"(?ims)^\s*Au lecteur\.?\s*\n+"
        r"\s*(?:L['’‘ʼ`]orthographe d['’‘ʼ`]origine"
        r"|Cette version (?:[eé]lectronique|num[eé]ris[eé]e))"
        r"[\s\S]*?"
        r"(?=\n\s*\n\s*[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]{3,})"
    ),

    # --- Queue de "Produced by" multilignes : fragments residuels ---
    re.compile(
        r"(?im)^.*l['’‘ʼ`]autorisation de les utiliser pour pr[eé]parer ce texte\.?\s*$"
    ),
]


# ============================================================================
# FIN : ancres pour debut de pied de page
# ============================================================================

END_GUTENBERG = [
    re.compile(
        r"\*{2,}\s*END OF TH(?:E|IS) PROJECT GUTENBERG.*?\*{2,}",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(?m)^End of (?:the )?Project Gutenberg.*$", re.IGNORECASE),
]

# Un FIN explicite, eventuellement avec une mention de tome/livre.
FIN_PATTERN = re.compile(r"(?m)^\s*FIN(?:\s+[^.\n]{1,80})?\.?\s*$")

# Annexes a retirer APRES un FIN. On ne les utilise qu'en post-traitement
# d'un FIN trouve, jamais pour deviner ou couper en l'absence de FIN.
ANNEXES_POST_FIN = [
    # "Achevé d'imprimer le X chez Y" (Morel, Rouquette, Coppee)
    re.compile(r"(?im)^\s*ACHEV[EÉ]\s+D[''`]IMPRIMER\b.*$"),
    # "IMPRIMERIE NELSON, EDIMBOURG, ECOSSE" (Weyman)
    re.compile(r"(?im)^\s*IMPRIMERIE\s+[A-Z][A-Z\-' ]{2,}.*$"),
    # "Imprimé en France" (France)
    re.compile(r"(?im)^\s*Imprim[eé]\s+en\s+France\s*$"),
    # "TABLE" en ligne seule (Raspe, Sacher en post-FIN)
    re.compile(r"(?m)^\s*TABLE\s*$"),
    # "TABLE DES MATIERES" / "FIN DE LA TABLE..."
    re.compile(r"(?im)^\s*(?:FIN DE LA )?TABLE(?:\s+DES\s+MATI[EÈ]RES)?\.?\s*$"),
    # "ARBRE GENEALOGIQUE..." (Zola Pascal)
    re.compile(r"(?im)^\s*ARBRE\s+G[EÉ]N[EÉ]ALOGIQUE.*$"),
]

# Bandes decoratives en toute fin de fichier.
#
# LA CLASSE NE DOIT PAS CONTENIR `\n`. La version precedente utilisait
# `\s`, qui l'inclut, dans `(?:^[\s...]*$\r?\n?)+\Z` : trois elements du
# motif pretendaient alors consommer le meme saut de ligne (la classe, le
# `$` multiligne, et le `\n?`). Le moteur doit essayer toutes les facons de
# repartir les sauts de ligne entre eux, et comme le groupe est repete par
# `+`, le nombre de repartitions explose avec le nombre de lignes.
#
# Tant que `\Z` reussit, le premier essai suffit et personne ne voit rien.
# C'est quand `\Z` ECHOUE que le moteur explore tout l'espace avant de
# renoncer, et `sub()` tente le motif a chaque position de la chaine. Sur
# un fichier CRLF de 326 000 caracteres se terminant par de la prose
# (pg75717, « L'Etbaye »), le nettoyage ne rendait plus la main.
#
# `\r` peut rester dans la classe : en Python, `$` ne s'ancre que devant
# `\n`, jamais devant `\r`, donc il n'y a pas d'ambiguite le concernant.
# Chaque iteration consomme exactement une ligne, et le motif redevient
# lineaire.
TRAILING_DECORATION = re.compile(r"(?m)(?:^[ \t\r#*=\-_~+]*$\n?)+\Z")


# ============================================================================
# INLINE : marqueurs editoriaux a supprimer du corps du texte
# ============================================================================

MARQUEURS_INLINE = [
    re.compile(r"\[\s*Illustrations?[^\]]*\]"),
    re.compile(
        r"\[\s*Note\s+du\s+(?:traducteur|transcripteur|correcteur|relecteur)[^\]]*\]",
        re.IGNORECASE,
    ),
    re.compile(r"\[\s*N\.\s*d\.\s*[TtRrEe]\.\s*[^\]]*\]"),
    re.compile(r"\[\s*sic\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*Transcriber'?s?\s+note[^\]]*\]", re.IGNORECASE),
]


# ============================================================================
# TYPOGRAPHIE GUTENBERG
# ============================================================================

ITALIQUE_UNDERSCORE = re.compile(
    r"(?<![A-Za-z0-9])_([A-Za-z\u00C0-\u017F][^_\n]{0,200}?[A-Za-z\u00C0-\u017F.,!?;:'\"\u2019\u00BB\u201D])_(?![A-Za-z0-9])"
)
SEPARATEUR_ASTERISQUES = re.compile(r"(?m)^\s*(?:\*\s*){3,}\s*$")


# ============================================================================
# TITRES : patterns a isoler par \n\n
# ============================================================================

TITRES_PATTERNS = [
    re.compile(
        r"(?im)^[\t ]*"
        r"(?:CHAPITRE|CHAPTER|LIVRE|PARTIE|TOME|TITRE|SECTION|EPILOGUE|PROLOGUE)"
        r"\s+(?:[IVXLCDM]+|\d{1,3}|PREMIER[E]?|DEUXIEME|TROISIEME|QUATRIEME|"
        r"CINQUIEME|SIXIEME|SEPTIEME|HUITIEME|NEUVIEME|DIXIEME)"
        r"\s*\.?\s*$"
    ),
    re.compile(
        r"(?im)^[\t ]*"
        r"(?:PREMIERE|DEUXIEME|TROISIEME|QUATRIEME|CINQUIEME)"
        r"\s+(?:PARTIE|LIVRE|TOME)"
        r"\s*\.?\s*$"
    ),
]


# ============================================================================
# Etapes de nettoyage
# ============================================================================

def couper_debut(texte, fenetre=None):
    """
    Cherche tous les marqueurs d'en-tete dans les `fenetre` premiers
    caracteres et coupe au plus tardif.
    """
    if fenetre is None:
        fenetre = CLEAN_PARAMS["header_window"]
    debut = texte[:fenetre]
    fin_header = -1
    for pattern in HEADER_END_ANCHORS:
        for m in pattern.finditer(debut):
            if m.end() > fin_header:
                fin_header = m.end()
    if fin_header > 0:
        return texte[fin_header:].lstrip("\r\n")
    return texte


def couper_fin(texte):
    """
    Retire le pied Gutenberg, puis coupe apres le DERNIER 'FIN' du texte.

    Si aucun FIN n'est trouve, on laisse le texte tel quel (pas de
    devinette risquee). Le post-traitement `retirer_annexes_post_fin`
    nettoiera ensuite ce qui suit le FIN.
    """
    for pattern in END_GUTENBERG:
        m = pattern.search(texte)
        if m:
            texte = texte[:m.start()]
            break

    matches_fin = list(FIN_PATTERN.finditer(texte))
    if matches_fin:
        dernier = matches_fin[-1]
        texte = texte[: dernier.end()]

    return texte.strip()


def retirer_annexes_post_fin(texte):
    """
    Retire les annexes (TABLE DES MATIERES, ACHEVE D'IMPRIMER, etc.)
    qui peuvent suivre un FIN explicite.

    Ne s'active QUE si un FIN est present dans le texte. Sinon on laisse
    intact pour ne pas risquer de couper du contenu legitime sur des
    livres sans marqueur de fin (Bougainville, Hugo, Homer, Sandre,
    Zola Bonheur Des Dames, etc.).

    Cherche les annexes uniquement dans la portion APRES le dernier FIN.
    """
    matches_fin = list(FIN_PATTERN.finditer(texte))
    if not matches_fin:
        return texte

    pos_fin = matches_fin[-1].end()
    apres_fin = texte[pos_fin:]

    # On cherche le PREMIER pattern d'annexe dans la portion apres FIN
    pos_coupe_relative = -1
    for pattern in ANNEXES_POST_FIN:
        for m in pattern.finditer(apres_fin):
            if pos_coupe_relative == -1 or m.start() < pos_coupe_relative:
                pos_coupe_relative = m.start()

    if pos_coupe_relative > 0:
        return texte[: pos_fin + pos_coupe_relative].rstrip()
    return texte


def retirer_marqueurs_inline(texte):
    """Supprime les marqueurs editoriaux inline."""
    for pattern in MARQUEURS_INLINE:
        texte = pattern.sub("", texte)
    return texte


def retirer_typographie_gutenberg(texte):
    """Retire italiques `_mot_` et separateurs `* * *`."""
    texte = ITALIQUE_UNDERSCORE.sub(r"\1", texte)
    texte = SEPARATEUR_ASTERISQUES.sub("\n\n", texte)
    return texte


def isoler_titres(texte):
    """Entoure les titres de chapitre par des doubles sauts de ligne."""
    def _entourer(match):
        return f"\n\n{match.group(0).strip()}\n\n"

    for pattern in TITRES_PATTERNS:
        texte = pattern.sub(_entourer, texte)
    return texte


def couper_decorations(texte):
    """Retire les bandes decoratives en toute fin de fichier."""
    return TRAILING_DECORATION.sub("", texte).rstrip()


def nettoyer(texte, fenetre=None):
    """Enchaine toutes les etapes de nettoyage."""
    texte = couper_debut(texte, fenetre=fenetre)
    texte = couper_fin(texte)
    texte = retirer_annexes_post_fin(texte)
    texte = retirer_marqueurs_inline(texte)
    texte = retirer_typographie_gutenberg(texte)
    texte = isoler_titres(texte)
    texte = couper_decorations(texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte