# Guess the Book ! -- Identification de livres à partir d'extraits

Pipeline complet : téléchargement Gutenberg -> nettoyage -> tokénisation -> annotation linguistique -> représentations TF-IDF / Word2Vec / fastText -> résumés extractifs (MMR). Stockage Scaleway Object Storage (S3).

## Architecture

```
projet/
  pipeline/                # logique métier (importable)
    config.py              # parametres centraux (BUCKET, préfixes, hyperparams)
    storage.py             # client S3 + helpers I/O
    parsing.py             # slugifier, parser_chemin, auteur_principal
    gutenberg.py           # recherche dans le catalogue + téléchargement
    cleaning.py            # nettoyer() : enlève en-têtes/pieds Gutenberg
    tokenization.py        # tokeniser() : spaCy blank fr + normalisation typo
    annotation.py          # annoter() : POS, lemmes, frontières de phrases
    representations.py     # TfIdfMaison, similarités, agregation embeddings
    summarization.py       # extraire_phrases, mmr, resumer_livre
    pipeline_full.py       # raw -> clean -> tokens -> annotations en une fonction
    corpus.py              # charger_corpus, trouver_livre
    abstractive.py         # produit les résumés
    evaluation.py          # métriques : rangs (Precision@k, MRR), ROUGE, BLEU
    benchmark.py           # orchestration des expériences (jeu de test, boucles, tableaux)

  scripts/                 # entrées CLI
    upload_corpus.py       # upload initial de N livres tires au hasard
    add_book.py            # ajoute un livre Gutenberg (recherche par titre/auteur)
    rebuild_artifacts.py   # regenere TF-IDF + Word2Vec + resumes globaux
    run_benchmark.py       # lance les experiences d'evaluation, exporte les CSV

  notebooks/               # narration / exploration / validation
    01_clean_book.ipynb       # nettoyage
    02_tokenize_book.ipynb    # tokenisation
    03_annotate_book.ipynb    # annotation POS / lemmes
    04_explore_corpus.ipynb   # exploration visuelle
    05_tfidf_similarity.ipynb # representations TF-IDF + benchmark
    06_embeddings_pca.ipynb   # Word2Vec, fastText, PCA, t-SNE, UMAP
    07_summary_mmr.ipynb      # resume extractif MMR

  tests/                   # suite pytest
    test_evaluation.py     # metriques recherche + ROUGE / BLEU vs implementations de reference
    test_benchmark.py      # orchestration, sur corpus synthetique (ni S3 ni camembert)

  app/                     # application Streamlit interactive
    Home.py                # identification d'un livre depuis un extrait
    utils.py               # helpers partages (cache corpus, cache TF-IDF)
    pages/
      1_Catalogue.py       # voir le corpus + ajouter un livre via Gutenberg
      2_Exploration.py     # visualisations corpus (POS, lemmes, Zipf, Heaps)
      3_Resumes.py         # resumes MMR avec lambda interactif

  pg_catalog.csv          # catalogue Gutenberg
  .env                    # credentials
```

## Installation

**Python 3.12 obligatoire.** C'est la version du `Dockerfile` (`python:3.12-slim`), et surtout le plafond de la stack : `gensim` et `spacy` ne publient pas de wheel au-delà de cp313, `tokenizers` encore moins. Sur une version plus récente, pip retombe sur une compilation depuis les sources qui échoue (le C généré par Cython accède à `PyLongObject.ob_digit`, champ déplacé par CPython 3.12).

Si `python3.12` n'est pas dans les dépôts de ta distribution (cas d'Ubuntu 26.04) :

```bash
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update && sudo apt install python3.12 python3.12-venv
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

`tests/test_evaluation.py` couvre les métriques de `pipeline/evaluation.py`. Les métriques faites main y sont comparées à `rouge-score` (implémentation Google) pour ROUGE et à `nltk` pour BLEU. Ces deux paquets ne servent que de référence de test et ne sont jamais importés par `pipeline/` : sans eux la suite reste verte, les tests de comparaison sont simplement sautés.

Récupérer le catalogue Gutenberg (une fois) :

```bash
wget https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
```

Configurer les crédentials :

```bash
cp .env.example .env
# Editer .env avec tes vraies cles SCW_ACCESS_KEY / SCW_SECRET_KEY
```

## Première utilisation : monter le corpus

```bash
# 1. Tirer N livres au hasard du catalogue et les uploader sous raw/
python -m scripts.upload_corpus

# 2. Lancer notebooks 01 -> 03 dans l'ordre pour produire clean/, tokens/, annotations/

# 3. Générer les artefacts globaux (TF-IDF, Word2Vec, resumes MMR)
python -m scripts.rebuild_artifacts
```

## Ajouter un livre

```bash
# Recherche interactive dans le catalogue Gutenberg
python -m scripts.add_book --titre "Madame Bovary"
python -m scripts.add_book --auteur "Flaubert"
python -m scripts.add_book --titre "Bovary" --auteur "Flaubert"

# Avec regeneration immediate des artefacts globaux
python -m scripts.add_book --titre "Notre-Dame de Paris" --rebuild

# Mode non-interactif : prend le 1er candidat
python -m scripts.add_book --auteur "Maupassant" --non-interactive --rebuild
```

Le script :
1. cherche dans le catalogue local `pg_catalog.csv`
2. affiche les candidats trouves avec leur identifiant Gutenberg, titre, auteur, genre detecte
3. demande lequel ajouter (ou prend le premier en mode non-interactif)
4. telecharge depuis Gutenberg
5. passe le livre par le pipeline complet (raw -> clean -> tokens -> annotations)
6. avec `--rebuild` : regenere les matrices TF-IDF, Word2Vec et resumes MMR pour integrer le nouveau livre au benchmark

## Évaluation

```bash
# Recherche seule : quelques secondes, ne charge pas camembert
python -m scripts.run_benchmark --skip-resumes

# Essai rapide de la chaîne complète sur 3 livres avant le corpus entier
python -m scripts.run_benchmark --limite 3

# Tout, plus l'effet de lambda sur le compromis pertinence/diversité
python -m scripts.run_benchmark --balayer-lambda
```

Les CSV sortent dans `resultats/` (ignoré par git) : `recherche.csv` (une ligne par représentation x métrique), `recherche_courbes.csv` (Rappel@k en format long pour le graphique), `resumes.csv` et `lambda.csv`.

Deux régimes de coût très différents. La partie recherche ne touche ni torch ni CamemBERT. La partie résumé encode chaque phrase candidate, donc télécharge ~440 Mo au premier lancement puis compte quelques secondes par livre sur CPU.

Deux points à garder en tête en lisant les chiffres, tous les deux détaillés dans les commentaires de `pipeline/benchmark.py` :

- les extraits sont tirés des livres qui sont eux-mêmes indexés, donc l'extrait est contenu dans le document cible. Les scores de recherche sont mécaniquement hauts et ne se transposent pas à un livre absent du corpus.
- ROUGE est calculé contre le texte intégral faute de résumés de référence. Dans ce montage la précision vaut ~1 pour toute méthode extractive et le rappel ne dépend que de la longueur : ROUGE ne classe pas les méthodes. Ce sont les colonnes `redondance` et `couverture` qui les séparent.

## App interactive (Streamlit)

```bash
streamlit run app/Home.py
```

Quatre pages dans la sidebar :

- **Home** : identification d'un livre a partir d'un extrait. L'utilisateur colle un passage (ou en tire un aleatoirement depuis un livre du corpus pour tester), choisit la representation TF-IDF (lemmes 1g/12g, tokens 12g) et la metrique (cosinus, euclidien, jaccard). L'app classe les livres du corpus et affiche le top 10 avec leur score. En mode demo, l'app montre le rang du vrai livre dans le classement.

- **Catalogue** : deux onglets. *Corpus actuel* liste tous les livres indexes avec leurs stats. *Ajouter un livre* offre une recherche dans `pg_catalog.csv` par titre/auteur, selection d'un candidat, et lance le pipeline complet en arriere-plan. Apres ajout, l'app suggere un rebuild des artefacts globaux.

- **Exploration** : visualisations interactives. Mode *Par livre* (top lemmes, distribution POS, longueurs de phrases). Mode *Corpus entier* (Zipf, richesse lexicale a la Heaps). Mode *Comparaison auteurs* (heatmap des profils grammaticaux).

- **Resumes** : sliders pour `k` (nombre de phrases) et `lambda` (compromis pertinence/diversite). Le resume est recalcule en direct, avec metriques de pertinence et redondance, et un graphique de l'effet de lambda sur le livre courant. Bouton de telechargement en `.txt`.

## Régénérer les artéfacts seuls

```bash
python -m scripts.rebuild_artifacts                  # tout
python -m scripts.rebuild_artifacts --skip-w2v       # plus rapide
python -m scripts.rebuild_artifacts --skip-summaries # juste TF-IDF + Word2Vec
```

## Configuration

Toute la configuration vit dans `pipeline/config.py` :

- `BUCKET`, `REGION`, `ENDPOINT_URL` : connexion S3
- `PREFIXES` : nommage des etages (`raw/`, `clean/`, `tokens/`, ...)
- `GENRE_MAPPING` : correspondance Bookshelves Gutenberg -> genre interne
- `CLEAN_PARAMS`, `ANNOTATE_PARAMS`, `TFIDF_PARAMS`, `W2V_PARAMS`, `MMR_PARAMS`,
  `BENCH_PARAMS` : hyperparametres centralises

Pour tester un paramètre différent (ex. `lambda` MMR a 0.5 au lieu de 0.7),
modifier la valeur dans `config.py` ou la surcharger localement dans le
notebook avec un dict `CFG`.

## Choix de design

- **Le pipeline ne nettoie pas dans `upload_corpus`.** Le brut est conserve tel quel sous `raw/`. Le nettoyage est isole dans `clean_book.ipynb` -> `clean/`. On peut re-nettoyer sans retelecharger.
- **`get_client()` est mis en cache (`lru_cache`)** : un seul client S3 par processus.
- **`ensure_bucket()` est explicite** (pas d'effet de bord a l'import). A appeler une fois.
- **`TfIdfMaison` couvre livres et phrases** (anciens `TfIdfMaison` + `TfIdfPhrases` fusionnes). Conventions sklearn (smoothing IDF, normalisation L2). Validation contre `sklearn.TfidfVectorizer` dans le notebook 05.
- **Les notebooks importent du package** plutot que de redefinir les fonctions. Pour les modifier, on edite `pipeline/`, pas le notebook.