"""
Configuration centrale du pipeline.

Tout ce qui est partage entre les notebooks et les scripts vit ici.
Les valeurs sont surchargeables localement via le dict CFG dans chaque
notebook si besoin (ex. taille_extrait dépendant de l'experience en cours).
"""
from pathlib import Path

# --- S3 / stockage ---
BUCKET = "gautier-blairon-bucket"
REGION = "fr-par"
ENDPOINT_URL = "https://s3.fr-par.scw.cloud"

# --- Robustesse des acces S3 ---
#
# Un chargement de corpus, c'est un GET par livre, en serie. A 26 livres une
# coupure passagere etait improbable et sans consequence ; a 300, chaque run
# enchaine 300 requetes et une seule qui echoue faisait tout perdre.
#
# `read_timeout` est releve parce que les Parquet annotes vont jusqu'a
# quelques Mo et que le defaut de 60 s se revele juste sur une liaison
# chargee. Les tentatives couvrent le cas ou la requete part quand meme en
# timeout : on la reemet, avec une attente qui double a chaque essai.
S3_PARAMS = {
    "connect_timeout":   15,
    "read_timeout":      120,
    "max_tentatives":    5,
    "attente_initiale":  1.0,   # secondes, doublee a chaque nouvel essai

    # Telechargements simultanes au chargement du corpus.
    #
    # LE CHARGEMENT EST A 100 % DU RESEAU. Profil mesure sur 20 livres :
    # telechargement 26,83 s par livre, lecture du Parquet 0,01 s,
    # `extraire_termes` 0,01 s. Autrement dit, une fois les octets arrives,
    # le corpus entier se met en forme en 3 secondes. Inutile de chercher a
    # optimiser le calcul, il n'y en a pas.
    #
    # Et ce n'est pas le debit mais la LATENCE : les objets font moins d'un
    # Mo, donc 27 s par objet est de l'attente, pas du transfert.
    #
    # ATTENTION EN COMPARANT DES MESURES. La latence de l'endpoint varie
    # d'un facteur 15 selon le moment : le meme chargement de 291 livres a
    # pris 1,7 s par livre un jour et 27 s le lendemain. Deux mesures prises
    # a des heures differentes ne se comparent pas, et c'est un piege dans
    # lequel il est facile de tomber.
    #
    # Le gain du parallelisme a donc ete mesure en A/B au meme instant, sur
    # deux lots d'objets entrelaces : 27,59 s par objet en serie contre
    # 10,12 s a 8 threads, soit x2,7. Et pas x8 : l'endpoint bride aussi la
    # concurrence, ce qui est la raison de ne pas monter beaucoup plus haut.
    # `max_pool_connections` du client est aligne sur cette valeur, sinon
    # les threads se mettent en file d'attente sur les connexions.
    "parallelisme":      8,
}

# Préfixes : un par étage du pipeline
PREFIXES = {
    "raw":          "raw/",
    "clean":        "clean/",
    "tokens":       "tokens/",
    "annotations":  "annotations/",
    "summaries":    "summaries/",
    "artifacts":    "artifacts/",   # matrices TF-IDF / Word2Vec serialisées
}

# --- Catalogue Gutenberg ---
CATALOG_PATH = Path("pg_catalog.csv")
DEFAULT_LANGUAGE = "fr"
GUTENBERG_DOWNLOAD_URLS = [
    "https://www.gutenberg.org/cache/epub/{tid}/pg{tid}.txt",
    "https://www.gutenberg.org/files/{tid}/{tid}-0.txt",
    "https://www.gutenberg.org/files/{tid}/{tid}-8.txt",
]

# --- Mapping shelves -> genre interne ---
# Ordre = priorite : premier match gagne.
GENRE_MAPPING = {
    "science-fiction & fantasy":      "scifi_fantasy",
    "crime, thrillers and mystery":   "mystery",
    "historical novels":              "historical_fiction",
    "adventure":                      "adventure",
    "romance":                        "romance",
    "humour":                         "humour",
    "short stories":                  "short_stories",
    "plays/films/dramas":             "drama",
    "poetry":                         "poetry",
    "children & young adult reading": "children",
    "biographies":                    "biography",
    "essays, letters & speeches":     "essays",
    "travel writing":                 "travel",
    "philosophy & ethics":            "philosophy",
    "mythology, legends & folklore":  "mythology",
    "novels":                         "novel",
}

# --- Override de genres au chargement du corpus ---
# Pour les oeuvres en plusieurs tomes que Gutenberg classe differemment
# d'un tome a l'autre. La cle est le `livre_slug` (apres deslug ce serait
# le meme nom de livre), la valeur est le genre force.
#
# Pourquoi : pour la tache de classification d'extraits, deux tomes
# du meme roman doivent avoir le meme genre. Pour le resume MMR, c'est
# moins critique mais ca evite des incoherences cote vis-a-vis utilisateur.
#
# Applique au runtime dans corpus.charger_corpus(), donc pas besoin de
# regenerer les Parquet annotes pour le prendre en compte.
GENRE_OVERRIDES = {
    # Le Comte de Monte Cristo : Tome I est classe "historical_fiction",
    # Tome II est classe "adventure" par Gutenberg. On force "adventure"
    # qui est le genre dominant des deux Bookshelves Gutenberg pour cette
    # oeuvre.
    "pg17989_le_comte_de_monte_cristo_tome_i":  "adventure",
    "pg17990_le_comte_de_monte_cristo_tome_ii": "adventure",
}

# --- Nettoyage Gutenberg ---
CLEAN_PARAMS = {
    "header_window": 15000
}

# --- Collecte du corpus ---
#
# Seuils des portes de `pipeline/collecte.valider_texte`. Calibres sur les
# 26 livres du corpus initial, qui servent de temoin : un reglage qui
# rejetterait un livre deja valide serait trop severe par construction.
# Les valeurs mesurees sur ce temoin sont reportees en face de chaque seuil.
COLLECTE_PARAMS = {
    # Taille du texte BRUT, en caracteres.
    # Temoin : de 50 480 (Promenades et interieurs) a 1 242 876 (L'homme qui
    # rit). Le plancher laisse donc une marge d'un facteur 1,7 sous le plus
    # court des livres deja retenus.
    "taille_min_defaut": 30_000,
    # Genres ou un texte court est la norme et non un fichier tronque. Un
    # recueil de poemes ou une piece en un acte sont des oeuvres completes a
    # 20 000 caracteres, la ou un roman a cette taille est un fragment.
    "taille_min_par_genre": {
        "poetry":        15_000,
        "drama":         15_000,
        "short_stories": 15_000,
        "children":      15_000,
    },
    # Plafond : l'annotation spaCy est lineaire en longueur, et les fichiers
    # au-dela sont en pratique des compendiums multi-volumes plutot que des
    # oeuvres uniques. Marge d'un facteur 2,4 au-dessus du plus long temoin.
    "taille_max": 3_000_000,

    # Part maximale de mots-outils anglais dans le corps du texte.
    # Temoin : maximum observe 0,004, sur 26 livres tous francais. Un texte
    # reellement anglais tourne autour de 0,9. Les deux populations sont
    # separees par deux ordres de grandeur, donc la valeur exacte du seuil
    # importe peu -- 0,25 est place la ou il n'y a rien, volontairement.
    "part_anglaise_max": 0.25,

    # Part minimale du texte brut qui doit survivre au nettoyage.
    # Temoin : minimum observe 0,85 (Oeuvres completes de Gustave Flaubert),
    # mediane 0,98. Le seuil a 0,50 ne se declenche donc que sur un
    # nettoyage franchement pathologique, pas sur un en-tete un peu gras.
    "ratio_nettoyage_min": 0.50,
}

# --- Annotation ---
ANNOTATE_PARAMS = {
    "model":      "fr_core_news_sm",
    "disable":    ["ner"],
    "chunk_size": 100_000,
    "batch_size": 8,
}

# --- TF-IDF ---
#
# min_df=1 et pas 2, contre l'habitude. Le reflexe "min_df=2 pour filtrer le
# bruit" vient de la classification de documents, ou un terme vu une seule
# fois est du bruit. Ici la tache est l'INVERSE : identifier un livre depuis
# un extrait. Un terme present dans un seul livre du corpus n'est pas du
# bruit, c'est l'indice parfait -- un nom de personnage, un lieu, un mot
# forge par l'auteur. min_df=2 les eliminait tous, soit 52 % du vocabulaire.
#
# Mesure sur 520 extraits de 200 termes, 26 livres, seed 42, comparaison
# appariee (meme jeu de test pour les deux reglages) :
#
#   lemmes_1g   min_df=2 : 95.77 % top-1 (22 erreurs)   vocab  17 980
#   lemmes_1g   min_df=1 : 98.46 % top-1 ( 8 erreurs)   vocab  38 197
#   lemmes_12g  min_df=2 : 98.08 % top-1 (10 erreurs)   vocab  86 941
#   lemmes_12g  min_df=1 : 99.62 % top-1 ( 2 erreurs)   vocab 734 223
#
# Les 8 echecs repares sont des paires de livres proches (deux memoires
# napoleoniens, deux souvenirs de theatre) que seul le vocabulaire rare
# separe. Les 2 qui resistent sont Monte-Cristo Tome I pris pour le Tome II,
# soit le meme roman en deux volumes : irreductible.
#
# ATTENTION AU COUT EN BIGRAMMES. Les bigrammes sont presque tous uniques,
# donc min_df=1 les garde quasiment tous : le vocabulaire de `lemmes_12g`
# passe de 87 k a 734 k termes, et comme `TfIdfMaison.fit_transform` fait un
# `toarray()`, la matrice DENSE passe de 18 Mo a 153 Mo pour 26 livres, en
# croissance lineaire avec le corpus. C'est pour ca que la config servie
# reste `lemmes_1g` (7,9 Mo) : elle capte l'essentiel du gain pour un
# cinquieme de la memoire de `lemmes_12g` en min_df=2.
#
# Reproduire : python -m scripts.run_benchmark --skip-resumes
TFIDF_PARAMS = {
    "min_df":       1,
    "max_df_ratio": 0.85,
}

# Les trois representations construites par rebuild_artifacts et evaluees
# par benchmark. Source unique : le benchmark doit mesurer exactement les
# configs qui sont servies, sinon les chiffres du rapport ne decrivent pas
# ce qui tourne en prod.
TFIDF_CONFIGS = {
    "lemmes_1g":  {"champ": "lemmes", "ngram_max": 1},
    "lemmes_12g": {"champ": "lemmes", "ngram_max": 2},
    "tokens_12g": {"champ": "tokens", "ngram_max": 2},
}

# --- Word2Vec ---
W2V_PARAMS = {
    "vector_size": 100,
    "window":      5,
    "min_count":   3,
    "epochs":      10,
    "sg":          1,        # skip-gram
    "workers":     4,
    "seed":        42,
}

# --- MMR ---
MMR_PARAMS = {
    "k_phrases":         10,
    "lambda_default":    0.6,
    # min_tokens_phrase a 12 (sur tokens informatifs) eliminait 70-90 %
    # des phrases sur la majorite du corpus -- tous les dialogues et
    # phrases d'action courtes etaient jetes. A 8, on garde environ
    # 40-50 % des phrases tout en filtrant les interjections.
    "min_tokens_phrase": 8,
    "max_tokens_phrase": 60,     # élimine les paragraphes mal segmentés
    "min_densite":       0.30,   # ratio informatifs / total minimum
    "max_propn_ratio":   0.40,   # ratio noms propres / informatifs maximum
    "min_df":            3,
    "ngram_max":         2,
}

# --- Embeddings de phrases (résume extractif) ---
EMB_PARAMS = {
    # 'base' : 110M paramètres, dim 768, ~440 Mo
    # 'large': 400M paramètres, dim 1024, ~1.3 Go (5x plus lent en CPU)
    "modele": "dangvantuan/sentence-camembert-base",

    # Pondérations du score hybride. La somme ne vaut pas 1 et n'a pas a
    # valoir 1 : seul le score relatif entre phrases sert (cf. le docstring
    # de `summarization.score_hybride`).
    # - centre   : similarité avec le theme global du livre
    # - textrank : centralité du graphe (PageRank sur similarités)
    # - longueur : bonus pour les phrases de taille naturelle
    #
    # textrank a 0.20 et non 0.30. Balayage sur les 26 livres du corpus,
    # comparaison appariee contre le top-k TF-IDF naif (`tfidf_naif`), test
    # des signes :
    #
    #   w      redondance  MMR gagne      p     couverture  MMR gagne      p
    #   0.30     0.4385       7/26     0.029      0.2673      17/26     0.169
    #   0.20     0.3953      16/26     0.327      0.2620      15/26     0.557
    #   0.10     0.3259      26/26    <0.0001     0.2426       7/26     0.029
    #   0.00     0.2651      26/26    <0.0001     0.2264       1/26    <0.0001
    #   (baseline tfidf_naif : redondance 0.4200, couverture 0.2575)
    #
    # A 0.30, MMR etait SIGNIFICATIVEMENT PLUS redondant que le simple
    # top-k, ce qu'il est cense minimiser : la composante TextRank est de
    # loin la plus redondante des trois, et le terme de diversite de MMR ne
    # rattrapait pas ce qu'elle introduisait dans la pertinence.
    #
    # Mais baisser w n'est pas gratuit : la couverture se degrade en
    # miroir. TextRank apporte la centralite thematique ; sans lui MMR
    # choisit des phrases mutuellement dissemblables mais individuellement
    # peu representatives. On echange un axe contre l'autre.
    #
    # 0.20 est le point de PARITE : les deux tests sont non significatifs
    # dans les deux sens, MMR cesse d'etre plus redondant que la baseline
    # sans payer en couverture. Ce n'est pas une victoire de MMR sur le
    # top-k naif -- aucune ponderation ne la produit sur ces deux criteres,
    # et le rapport doit le dire.
    #
    # Reserve : redondance et couverture sont des proxys intrinseques. Ils
    # ne mesurent ni la lisibilite ni la coherence narrative, qui etaient
    # les motivations qualitatives de la refonte v4. Un jugement humain sur
    # quelques livres trancherait ce que ces deux chiffres ne tranchent pas.
    "poids_score": {
        "centre":   0.70,
        "textrank": 0.20,
        "longueur": 0.05,
    },
}

# --- BARThez (resume abstratif) ---
BARTHEZ_PARAMS = {
    "model_name":      "moussaKam/barthez-orangesum-abstract",
    "max_input_chars": 4000,    # bart limit ~1024 tokens
    "max_output_len":  200,     # tokens de sortie
    "min_output_len":  80,
    "num_beams":       4,
    "no_repeat_ngram_size": 3,
}

# --- Benchmark ---
#
# `taille_extrait` est une PLAGE, pas une longueur fixe, et l'unite est le
# terme informatif (apres le filtre is_alpha & ~is_stop & ~is_punct), pas
# le mot brut. Le ratio mesure sur le corpus est de 0.349 terme informatif
# par token, donc :
#
#     7 termes  ~=  20 mots colles     (une phrase)
#    70 termes  ~= 200 mots colles     (deux ou trois paragraphes)
#
# Pourquoi une plage. L'utilisateur colle ce qu'il a sous la main, il ne
# compte pas ses mots : la longueur est une variable aleatoire du probleme,
# pas un parametre d'experience. Mesurer a longueur fixe repond a "quelle
# performance pour un extrait de N termes", qui n'est pas la question.
#
# Pourquoi PAS 200, l'ancienne valeur. 200 termes informatifs valent ~573
# mots bruts, soit presque le double du maximum de 300 que propose le
# slider de l'app. Le benchmark mesurait donc un scenario que le produit
# ne peut pas produire, et annoncait 98.5 % de top-1 la ou le defaut de
# l'app (80 mots, ~28 termes) donne 85.4 %. Treize points d'ecart, et
# Rappel@3 saturait a 1.0, ce qui rendait la courbe du rapport plate et
# sans information.
#
# Pour retrouver une courbe de difficulte en fonction de la longueur,
# passer un int a `tirer_extraits` plutot que de changer cette valeur.
BENCH_PARAMS = {
    "n_extraits_par_livre": 5,
    "taille_extrait":       (7, 70),
    "seed":                 42,
}