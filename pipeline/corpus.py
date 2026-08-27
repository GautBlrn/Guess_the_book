"""
Chargement du corpus annote depuis S3.

`charger_corpus()` retourne la liste de dicts attendue par les notebooks
04-07 : chaque element contient les metadonnees du livre (auteur, genre,
livre, slugs) et le DataFrame annote, plus les sequences `lemmes` et
`tokens` precalculees.

`iter_corpus()` fait la meme chose en generateur, un livre a la fois.

QUEL COUT EN MEMOIRE. Mesure RSS sur un echantillon de 30 livres reparti
sur tout le corpus : 7,8 Mo par livre pour les DataFrames et les sequences
de termes reunies, soit environ 2,2 Go a 291 livres.

Or la plupart des appelants n'ouvrent jamais les DataFrames : le TF-IDF ne
lit que `lemmes`/`tokens`, l'index n'a besoin que de deux comptages, et les
resumes comme Word2Vec traitent les livres un par un.

D'ou les trois leviers ci-dessous. `with_df=False` rend la main sans les
DataFrames mais avec `n_tokens` et `n_phrases` deja calcules, `iter_corpus()`
sert a qui veut les DataFrames sans en tenir tout le corpus, et `limite`
coupe la liste avant tout telechargement.

QUEL COUT EN TEMPS. Le chargement est a 100 % du reseau : 26,8 s par livre
de telechargement mesures contre 0,01 s de lecture du Parquet et 0,01 s
d'extraction des termes. Inutile de chercher a optimiser le calcul, il n'y
en a pas. Voir `S3_PARAMS` pour le detail et pour le piege des mesures
prises a des moments differents.
"""
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from pipeline import storage
from pipeline.annotation import extraire_termes
from pipeline.config import GENRE_OVERRIDES, PREFIXES, S3_PARAMS
from pipeline.parsing import parser_chemin


# Un point d'avancement tous les N livres. Assez frequent pour qu'un
# chargement de 300 livres reste suivable, assez espace pour ne pas noyer
# les messages qui comptent (overrides de genre, reprises reseau).
PAS_PROGRESSION = 10


class _Progression:
    """
    Suivi d'avancement d'un chargement, avec estimation du temps restant.

    LE DEBIT EST MESURE SUR UNE FENETRE GLISSANTE, pas depuis le depart. Un
    chargement S3 commence par une phase lente sans rapport avec son regime
    de croisiere : sur un corpus de 291 livres, les 10 premiers ont demande
    307 s et les suivants 0,65 s chacun. Une moyenne cumulee reste marquee
    par ce demarrage pendant tout le reste du chargement, et annoncait
    8 616 s restantes la ou il en fallait 170.

    LA FENETRE COUVRE PLUSIEURS PALIERS, et pas seulement le dernier. Les
    telechargements etant paralleles, les livres arrivent par rafales : deux
    paliers peuvent se succeder dans le meme instant, ce qui donne un palier
    de duree quasi nulle et un debit absurde (129 055 livres/s observes).
    Moyenner sur les derniers paliers absorbe ces rafales, et le
    `duree < 1e-3` reste comme garde-fou pour le cas ou toute la fenetre
    tomberait dans le meme instant.
    """

    # Nombre de paliers sur lesquels lisser. Trois suffisent a absorber une
    # rafale sans rendre l'estimation insensible a un vrai changement de
    # regime, ce qui est justement ce qu'on veut voir.
    FENETRE = 3

    def __init__(self, total):
        self.total = total
        self.depart = time.time()
        self.historique = deque([(0, self.depart)], maxlen=self.FENETRE + 1)

    def point(self, n):
        maintenant = time.time()
        ecoule = maintenant - self.depart

        n_ancien, t_ancien = self.historique[0]
        duree = maintenant - t_ancien
        debit = (n - n_ancien) / duree if duree > 1e-3 else 0
        self.historique.append((n, maintenant))

        if debit > 0:
            restant = f"~{(self.total - n) / debit:.0f}s restantes"
        else:
            restant = "temps restant inconnu"
        print(f"  {n:4d}/{self.total} livres  ({ecoule:5.1f}s ecoulees, "
              f"{debit:4.1f}/s, {restant})", flush=True)


def _lister_annotations(prefixe, verbose, limite=None):
    """
    Liste les Parquet annotes, ou leve si le prefixe est vide.

    `limite` coupe la liste AVANT tout telechargement. C'est le seul endroit
    ou la coupe est utile : charger les 291 livres pour n'en garder cinq
    ferait payer plusieurs minutes de reseau a un essai qui existe
    precisement pour ne pas les payer.
    """
    objets = storage.list_objects(prefixe, suffix=".parquet")
    if not objets:
        raise RuntimeError(
            f"Aucun .parquet trouve sous {prefixe}. "
            "As-tu execute le notebook 03_annotate_book ?"
        )
    total = len(objets)
    if limite is not None:
        objets = objets[:limite]
    if verbose:
        if limite is not None and len(objets) < total:
            print(f"Chargement de {len(objets)} livres annotes "
                  f"(limite, sur {total} disponibles)...")
        else:
            print(f"Chargement de {len(objets)} livres annotes...")
    return objets


def _charger_un(obj, prefixe, with_termes, with_df, verbose):
    """Charge un livre annote et calcule ce qui depend de son DataFrame."""
    info = parser_chemin(obj["Key"], prefixe)

    # Override de genre eventuel pour cette oeuvre
    override = False
    if info["livre_slug"] in GENRE_OVERRIDES:
        ancien = info["genre"]
        info["genre"] = GENRE_OVERRIDES[info["livre_slug"]]
        if ancien != info["genre"]:
            override = True
            if verbose:
                print(f"  override genre: {info['livre_slug']} "
                      f"{ancien} -> {info['genre']}")

    df = storage.get_parquet(obj["Key"])

    # Calcules tant que le DataFrame est sous la main : ce sont les deux
    # seules choses que `construire_index` lui demande, et les retenir ici
    # evite de garder tout le DataFrame pour deux entiers.
    info["n_tokens"] = int(len(df))
    info["n_phrases"] = int(df["sent_id"].nunique())

    if with_termes:
        info["lemmes"] = extraire_termes(df, champ="lemma")
        info["tokens"] = extraire_termes(df, champ="text")
    if with_df:
        info["df"] = df

    return info, override


def iter_corpus(prefixe=None, with_termes=True, with_df=True, verbose=True,
                limite=None):
    """
    Meme chargement que `charger_corpus`, mais un livre a la fois.

    A utiliser des que le traitement est livre par livre (generation des
    resumes, decoupage en phrases pour Word2Vec) : le pic memoire reste
    borne au lieu de croitre avec le corpus.

    Les livres sont telecharges par tranches de `S3_PARAMS['parallelisme']`,
    pour la meme raison que dans `charger_corpus`. Le pic memoire est donc
    celui d'une TRANCHE, pas d'un seul livre -- huit livres au lieu d'un,
    soit quelques dizaines de Mo, ce qui reste sans commune mesure avec le
    corpus entier et evite d'attendre chaque latence l'une apres l'autre.
    """
    if prefixe is None:
        prefixe = PREFIXES["annotations"]

    objets = _lister_annotations(prefixe, verbose, limite)
    suivi = _Progression(len(objets))
    taille_tranche = min(S3_PARAMS["parallelisme"], max(1, len(objets)))

    n = 0
    with ThreadPoolExecutor(max_workers=taille_tranche) as pool:
        for debut in range(0, len(objets), taille_tranche):
            tranche = objets[debut:debut + taille_tranche]
            charges = list(pool.map(
                lambda o: _charger_un(o, prefixe, with_termes, with_df, verbose),
                tranche,
            ))
            for info, _ in charges:
                n += 1
                if verbose and (n % PAS_PROGRESSION == 0 or n == len(objets)):
                    suivi.point(n)
                yield info


def charger_corpus(prefixe=None, with_termes=True, with_df=True, verbose=True,
                   limite=None):
    """
    Charge tous les Parquet annotes du bucket en memoire.

    Renvoie une liste ordonnee (par cle S3) de dicts :
        {auteur, auteur_slug, livre, livre_slug, genre, cle,
         n_tokens, n_phrases,
         df (si with_df), lemmes (si with_termes), tokens (si with_termes)}

    `lemmes` et `tokens` sont les sequences filtrees standard :
        is_alpha & ~is_stop & ~is_punct, lowercased.

    `with_df=False` economise ~3,2 Mo par livre en ne retenant pas le
    DataFrame annote. `n_tokens` et `n_phrases` restent disponibles : ils
    sont calcules pendant le chargement, avant que le DataFrame soit
    relache.

    Le genre derive de la structure de la cle S3 (chemin annotations/genre/auteur/livre)
    peut etre surcharge via `GENRE_OVERRIDES` dans la config (cle = livre_slug).
    Sert a corriger les cas ou Gutenberg classe deux tomes du meme roman
    sous des Bookshelves differents (ex: Monte Cristo I/II).
    """
    if prefixe is None:
        prefixe = PREFIXES["annotations"]

    objets = _lister_annotations(prefixe, verbose, limite)
    suivi = _Progression(len(objets))

    # Chargement PARALLELE. La lecture d'un livre est presque entierement de
    # l'attente reseau, donc les threads se recouvrent bien malgre le GIL,
    # que boto3 comme pyarrow relachent pendant leurs entrees-sorties.
    #
    # L'ordre du corpus est celui des cles S3, et il compte : c'est l'ordre
    # des lignes de la matrice TF-IDF et celui de l'index servi par l'API.
    # `executor.map` le preserve, contrairement a `as_completed`.
    parallelisme = min(S3_PARAMS["parallelisme"], max(1, len(objets)))

    corpus = []
    n_overrides = 0
    with ThreadPoolExecutor(max_workers=parallelisme) as pool:
        resultats = pool.map(
            lambda o: _charger_un(o, prefixe, with_termes, with_df, verbose),
            objets,
        )
        for n, (info, override) in enumerate(resultats, start=1):
            n_overrides += override
            corpus.append(info)
            if verbose and (n % PAS_PROGRESSION == 0 or n == len(objets)):
                suivi.point(n)

    if verbose:
        if n_overrides:
            print(f"  {n_overrides} override(s) de genre appliques.")
        print(f"  {len(corpus)} livres charges en "
              f"{time.time() - suivi.depart:.1f}s "
              f"({parallelisme} telechargements simultanes)", flush=True)
    return corpus


def trouver_livre(corpus, indice_ou_motif):
    """
    Recherche un livre dans le corpus.

    `indice_ou_motif` peut etre :
      - un int : index dans la liste corpus
      - une chaine : motif a matcher dans `auteur`, `livre` ou `genre`
                     (insensible casse, premier match gagne)

    Plus stable que `corpus[N]` qui change quand on ajoute un livre.
    """
    if isinstance(indice_ou_motif, int):
        return corpus[indice_ou_motif]
    motif = indice_ou_motif.lower()
    for c in corpus:
        if (motif in c["auteur"].lower()
                or motif in c["livre"].lower()
                or motif in c["genre"].lower()):
            return c
    raise KeyError(f"Aucun livre ne matche {indice_ou_motif!r}")