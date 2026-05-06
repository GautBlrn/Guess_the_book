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

# --- Annotation ---
ANNOTATE_PARAMS = {
    "model":      "fr_core_news_sm",
    "disable":    ["ner"],
    "chunk_size": 100_000,
    "batch_size": 8,
}

# --- TF-IDF ---
TFIDF_PARAMS = {
    "min_df":       2,
    "max_df_ratio": 0.85,
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

    # Pondérations du score hybride (somme = 1)
    # - centre   : similarité avec le theme global du livre
    # - textrank : centralité du graphe (PageRank sur similarités)
    # - longueur : bonus pour les phrases de taille naturelle
    "poids_score": {
        "centre":   0.70,
        "textrank": 0.30,
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
BENCH_PARAMS = {
    "n_extraits_par_livre": 5,
    "taille_extrait":       200,
    "seed":                 42,
}