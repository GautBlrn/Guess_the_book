# Guide de test

Comment vérifier que le programme marche, du plus rapide au plus complet.
Chaque niveau est indépendant : tu peux t'arrêter après le 2 si tu veux
juste savoir si l'app tourne. Les temps et les sorties donnés ici ont été
mesurés sur une machine Ubuntu, CPU seul.

Pour comprendre ce que tu testes, voir [fonctionnement.md](fonctionnement.md).
Pour les chiffres d'évaluation, voir [evaluation.md](evaluation.md).

## Résumé

| niveau | ce que ça teste | réseau | durée |
|---|---|---|---|
| 0 | installation | pip | 3 à 5 min |
| 1 | métriques et orchestration (pytest) | non | 5 s |
| 2 | accès S3 | S3 | 5 s |
| 3 | API Flask, les 5 routes | S3 | 1 min |
| 4 | interface Streamlit | S3 | 5 min à la main |
| 5 | benchmark de recherche | S3 | 1 min |
| 6 | benchmark de résumé | S3 + HF | 10 min et 440 Mo au premier lancement |
| 7 | image Docker complète | S3 | 1 à 3 min de build |

## Niveau 0 : installation

**Python 3.12 obligatoire.** C'est le plafond de la stack : `gensim` et
`spacy` ne publient pas de wheel au-delà de cp313. Sur Ubuntu 26.04,
`python3.12` passe par `ppa:deadsnakes/ppa` (voir le README).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m spacy download fr_core_news_sm
```

`requirements-dev.txt` contient tout (pipeline, notebooks, tests, torch,
transformers). `requirements.txt` est le sous-ensemble runtime utilisé par
le `Dockerfile`, il n'a ni torch ni gensim.

**Ne saute pas le `spacy download`.** Sans lui, tout ce qui n'annote pas
marche (pytest, S3, démarrage de l'API), et seul `/api/identifier` casse,
avec une 500 et un `[E050] Can't find model 'fr_core_news_sm'` enfoui dans
les logs serveur. C'est la panne la plus facile à diagnostiquer trop tard.

Il faut aussi un `.env` à la racine :

```
SCW_ACCESS_KEY=...
SCW_SECRET_KEY=...
```

Ce sont les deux seules variables lues (par `pipeline/storage.py`, via
`python-dotenv`). Les autres clés du `.env` ne servent pas au code Python.

Le catalogue Gutenberg n'est nécessaire que pour ajouter un livre :

```bash
wget https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
```

Vérification que l'install tient :

```bash
python -c "import spacy; spacy.load('fr_core_news_sm'); print('spacy ok')"
python -c "import flask, streamlit, boto3, pandas; print('deps ok')"
```

## Niveau 1 : la suite pytest

Le seul niveau qui ne touche ni S3 ni le réseau. C'est le test à lancer en
premier et après chaque modification de `pipeline/`.

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Attendu :

```
181 passed in 4.17s
```

Ce que ça couvre :

- `tests/test_evaluation.py` (41 fonctions) : les métriques faites main,
  comparées à `rouge-score` (Google) pour ROUGE et à `nltk` pour BLEU, sur
  des tirages aléatoires seedés. Écart maximum attendu : `0.00e+00`.
- `tests/test_benchmark.py` (34 fonctions) : les boucles d'expériences, sur
  un corpus synthétique, donc sans S3 ni CamemBERT.
- `tests/test_rebuild_artifacts.py` (7 fonctions) : le nettoyage des
  résumés orphelins, qui **supprime des objets S3**. Les tests portent
  surtout sur les cas où la fonction doit refuser d'agir. S3 est simulé,
  rien n'est réellement supprimé.

Le compte de 181 vient du paramétrage : 82 fonctions de test au total.

Si `rouge-score` ou `nltk` manquent, la suite reste verte et les tests de
comparaison sont sautés. Pour t'assurer qu'ils tournent vraiment :

```bash
python -m pytest tests/test_evaluation.py -q -rs   # -rs liste les skips
```

## Niveau 2 : l'accès S3

```bash
python -c "
from pipeline import storage
storage.ensure_bucket()
c = storage.get_client()
for o in c.list_objects_v2(Bucket='gautier-blairon-bucket', Prefix='artifacts/')['Contents']:
    print(f\"{o['Key']:45s} {o['Size']/1e6:7.1f} Mo\")
"
```

Attendu :

```
artifacts/corpus_index.json                       0.0 Mo
artifacts/serving_bundle.pkl                    178.6 Mo
artifacts/tfidf_lemmes_12g.pkl                  178.6 Mo
artifacts/tfidf_lemmes_1g.pkl                     9.0 Mo
artifacts/tfidf_tokens_12g.pkl                  204.0 Mo
artifacts/w2v_corpus.pkl                          7.8 Mo
```

Si ça lève `RuntimeError: Credentials Scaleway manquants`, le `.env` est
absent ou mal nommé. Si ça lève une erreur boto3 403, les clés sont
mauvaises ou n'ont pas les droits sur le bucket.

Les préfixes `clean/` et `summaries/` doivent contenir 26 objets chacun.

## Niveau 3 : l'API Flask seule

```bash
python api.py
```

Le serveur télécharge le bundle et précharge les résumés avant d'écouter.
Logs attendus, une dizaine de secondes au total :

```
[INFO] Téléchargement du bundle depuis s3://artifacts/serving_bundle.pkl...
[INFO]   bundle chargé en 2.6s : 26 livres, vocab 734,223, matrice (26, 734223)
[INFO] Préchargement de 26 résumés depuis S3...
[INFO]   26 résumés chargés en 5.5s (0 échec)
[INFO] Serveur prêt à servir des requêtes.
```

Le `WARNING: This is a development server` est normal en local, la prod
passe par gunicorn (`start.sh`).

Dans un second terminal :

**3.1 Healthcheck**

```bash
curl -s http://localhost:5000/health
```

```json
{"n_livres":26,"n_resumes":26,"ready":true,"status":"ok"}
```

`ready:false` ou `n_resumes` inférieur à `n_livres` signale un problème S3.
Un résumé manquant n'empêche pas le démarrage, il apparaît en `WARNING`
dans les logs.

**3.2 Index du corpus**

```bash
curl -s http://localhost:5000/api/livres | python -m json.tool | head -20
```

26 entrées, chacune avec `auteur`, `livre`, `genre`, `livre_slug`,
`n_tokens`, `n_phrases`, `cle_resume`. Le champ interne `i` ne doit pas
être exposé.

**3.3 Le test qui compte : extrait aléatoire puis identification**

C'est le test de bout en bout de la fonction principale. Il tire un
passage d'un livre connu et vérifie que l'API retrouve ce livre en
première position.

```bash
python - <<'EOF'
import requests
S = "http://localhost:5000"
slug = "pg50398_aventures_de_baron_de_munchausen"
extrait = requests.get(f"{S}/api/extrait/{slug}?taille=80").json()["extrait"]
r = requests.post(f"{S}/api/identifier", json={"extrait": extrait, "top_k": 3}).json()
print("config servie :", r["config"])
print("latence serveur :", r["duree_ms"], "ms")
for x in r["resultats"]:
    marque = "  <-- attendu" if x["livre_slug"] == slug else ""
    print(f'  {x["rang"]}. {x["score"]:.4f}  {x["livre_slug"]}{marque}')
EOF
```

Sortie type :

```
config servie : {'champ': 'lemmes', 'max_df_ratio': 0.85, 'metrique': 'cosinus', 'min_df': 1, 'ngram_max': 2}
latence serveur : 1779 ms
  1. 0.0447  pg50398_aventures_de_baron_de_munchausen  <-- attendu
  2. 0.0086  pg64305_chair
  3. 0.0043  pg57373_oeuvres_completes_de_gustave_flaubert_tome_5...
```

Deux points à vérifier :

- **le bon livre est en rang 1**, avec un score environ 5 fois supérieur
  au second. Sur un extrait de 80 mots, c'est le cas dans 85 à 96 % des
  cas selon la représentation, donc un échec isolé n'est pas un bug :
  relance sur un autre livre avant de conclure.
- **la latence du premier appel est de l'ordre de 1,8 s**, elle inclut le
  chargement de spaCy. Le second appel tombe à environ 15 ms. Si tous les
  appels restent à 1,8 s, spaCy est rechargé à chaque fois.

**3.4 Cas limites**

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5000/api/identifier \
     -H 'Content-Type: application/json' -d '{"extrait":""}'          # attendu 400
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/api/resume/nexiste_pas  # attendu 404
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/api/extrait/nexiste_pas # attendu 404
```

Le plafonnement de `top_k` se vérifie ainsi : demander `top_k: 99` doit
renvoyer 10 résultats, pas 26.

**3.5 Résumé**

```bash
curl -s http://localhost:5000/api/resume/pg50398_aventures_de_baron_de_munchausen \
  | python -c "import json,sys; print(json.load(sys.stdin)['resume'][:300])"
```

Le texte commence par un en-tête généré (`titre`, `auteur`, `genre`,
`Resume extractif - 10 phrases - MMR lambda=0.6`) suivi de 10 phrases. Un
résumé complet fait dans les 600 mots.

**3.6 Tous les extraits, d'un coup**

La route `/api/extrait` a longtemps renvoyé 500 sur Monte-Cristo Tome I :
son genre est corrigé par `GENRE_OVERRIDES` alors que son objet `clean/`
est resté rangé sous le genre d'origine, et la clé S3 était reconstruite
depuis le genre de l'index. Elle est maintenant résolue par listing au
démarrage (`indexer_cleans`), le log doit afficher `26/26 textes nettoyés
localisés`. Pour vérifier qu'aucun livre ne régresse :

```bash
python - <<'EOF'
import requests
S = "http://localhost:5000"
livres = requests.get(f"{S}/api/livres").json()["livres"]
ko = [l["livre_slug"] for l in livres
      if requests.get(f'{S}/api/extrait/{l["livre_slug"]}?taille=60').status_code != 200]
print(f"extraits OK : {len(livres) - len(ko)}/{len(livres)}")
for slug in ko:
    print("  KO", slug)
EOF
```

Attendu : `extraits OK : 26/26`.

## Niveau 4 : l'interface Streamlit

L'API doit tourner (niveau 3), Streamlit ne fait que l'appeler.

```bash
# terminal 1
python api.py
# terminal 2
streamlit run app/Home.py
```

Ouvre `http://localhost:8501`. Si l'API n'est pas là, chaque page affiche
« Backend indisponible » et s'arrête proprement : c'est le comportement
attendu, pas un plantage.

Scénario de recette, dans l'ordre :

1. **Home, mode démo.** Choisis une source dans « Extrait aléatoire »,
   règle la taille à 80 mots, clique « Tirer un extrait ». Le texte
   apparaît dans la zone de saisie et le bandeau de résultat doit afficher
   « Identification correcte » en vert.
2. **Home, effet de la longueur.** Refais le tirage à 30 mots, plusieurs
   fois. Tu dois voir apparaître des bandeaux orange « le vrai livre est
   au rang 2 ». C'est attendu et c'est la démonstration la plus parlante
   du projet : la longueur de l'extrait est le paramètre qui gouverne la
   difficulté, bien plus que la taille du corpus. L'ordre de grandeur
   mesuré en `lemmes_1g` va de 64 % de top-1 à 30 mots à 96 % à 300 mots
   (tableau complet dans `evaluation.md`). Le bundle servi étant en
   `lemmes_12g`, tu observeras mieux que ça, la pente reste la même.
3. **Home, extrait collé à la main.** Colle un passage d'un livre du
   corpus trouvé ailleurs. Le bandeau passe en mode simple (pas de rang
   attendu) puisque l'app ne connaît plus la bonne réponse. Vérifie que le
   lien se rompt bien : édite un extrait tiré aléatoirement, le mode démo
   doit disparaître.
4. **Catalogue.** 26 livres, 23 auteurs, 9 genres, 2 719 190 tokens au
   total. Le bouton « Rafraîchir » vide le cache et rappelle l'API.
5. **Résumés.** Sélectionne un livre, le résumé s'affiche avec son
   en-tête. Le bouton de téléchargement produit un `.txt` non vide.

Note : `app/dev/2_Exploration.py` n'est pas chargé par Streamlit (il est
hors de `pages/`) et il est cassé de toute façon, il importe une fonction
`get_corpus` qui n'existe plus. Ne le compte pas dans la recette.

## Niveau 5 : le benchmark de recherche

Ne charge ni torch ni CamemBERT, donc rapide.

```bash
python -m scripts.run_benchmark --skip-resumes
```

Mesuré : 37 s pour la partie recherche, 52 s au total avec le chargement
du corpus. Sorties dans `resultats/` : `recherche.csv` et
`recherche_courbes.csv`.

Avec les valeurs par défaut (`n_extraits_par_livre=5`, longueur tirée dans
7 à 70 termes informatifs, seed 42), tu dois retrouver exactement :

| représentation | métrique | n_requetes | top-1 | MRR |
|---|---|---|---|---|
| lemmes_1g | cosinus | 130 | 86,15 % | 0,901 |
| lemmes_12g | cosinus | 130 | 96,15 % | 0,979 |
| tokens_12g | cosinus | 130 | 96,15 % | 0,979 |
| lemmes_1g | jaccard | 130 | 50,00 % | 0,681 |

Deux invariants qui valident le calcul plus sûrement que les valeurs
elles-mêmes :

- **les lignes `cosinus` et `euclidien` sont identiques**, à toutes les
  décimales. Sur des vecteurs L2-normalisés c'est une identité
  mathématique, pas une coïncidence. Si elles diffèrent, la normalisation
  est cassée quelque part.
- **`precision@k = rappel@k / k`** exactement, puisqu'il y a un seul
  document pertinent par requête.

Le tableau de `evaluation.md` est calculé sur 520 extraits et non 130. Pour
le reproduire :

```bash
python -m scripts.run_benchmark --skip-resumes --n-extraits 20
```

Les scores sont mécaniquement hauts : les extraits sont tirés des livres
qui sont eux-mêmes indexés, donc l'extrait est littéralement contenu dans
le document cible. Ne lis pas ces 96 % comme une performance de
généralisation.

## Niveau 6 : le benchmark de résumé

Coûteux : encode chaque phrase candidate avec `sentence-camembert-base`,
donc environ 440 Mo téléchargés au premier lancement, puis quelques
secondes par livre sur CPU.

```bash
# essai sur 3 livres avant de lancer les 26
python -m scripts.run_benchmark --limite 3

# tout, plus l'effet de lambda sur le compromis pertinence/diversité
python -m scripts.run_benchmark --balayer-lambda
```

Sorties : `resumes.csv`, `lambda.csv`.

Ce qu'il faut regarder dans `resumes.csv`, et surtout ce qu'il ne faut pas
y chercher :

- `rouge1_precision` doit valoir **1,0 avec un écart-type nul** pour les
  trois méthodes. C'est structurel, chaque phrase étant copiée du livre.
  Une valeur différente de 1,0 signale un résumé cassé, pas une méthode
  meilleure ou moins bonne. C'est le seul usage utile de ROUGE ici.
- `rouge2_precision` doit tourner autour de 0,97, les 3 % manquants étant
  les bigrammes à cheval sur la jointure entre deux phrases.
- les colonnes qui séparent réellement les méthodes sont `redondance` (mmr
  autour de 0,395, tfidf_naif 0,420, textrank 0,584) et `couverture` (mmr
  0,262, tfidf_naif 0,258).

## Niveau 7 : Docker

Le test le plus proche de la production : un conteneur, deux process.

```bash
docker build -t nlp-app .
docker run --rm --env-file .env -p 8501:8501 -p 5000:5000 nlp-app
```

Le build passe (une minute avec un cache pip chaud, quelques minutes à
froid) et produit une image de 1,18 Go. L'image installe
`requirements.txt` et non `requirements-dev.txt`, donc ni torch ni gensim.

Logs attendus au démarrage, environ 11 s :

```
[start.sh] Lancement de gunicorn (Flask)...
[start.sh] Attente que l'API soit prete...
[INFO]   bundle chargé en 2.7s : 26 livres, vocab 734,223, matrice (26, 734223)
[INFO]   26 résumés chargés en 5.2s (0 échec)
[INFO]   26/26 textes nettoyés localisés.
[INFO] Serveur prêt à servir des requêtes.
[start.sh] API prete apres 11s.
[start.sh] Lancement de Streamlit...
```

**Ne touche pas aux pins numpy et spacy de `requirements.txt` sans relire
ceci.** Le bundle est un pickle produit sous `requirements-dev.txt`, donc
sous numpy 2. Un runtime en numpy 1.26 ne sait pas le relire (numpy 2 a
renommé `numpy.core` en `numpy._core`, et le pickle référence ce chemin) :
gunicorn meurt à l'import et le conteneur ne démarre jamais. C'était le cas
avant le 19 août 2026. La contrainte est donc :

- version majeure de numpy identique entre `requirements.txt` et
  `requirements-dev.txt` ;
- `spacy>=3.8` au runtime, parce que spacy 3.7.5 résout thinc 8.2.4, dont
  la wheel est compilée contre numpy 1.x et casse à l'import sous numpy 2
  (`ValueError: numpy.dtype size changed`). Ça se voit dès le
  `spacy download` du build, pas au runtime.

Si tu changes quand même de version majeure de numpy d'un seul côté,
`charger_bundle()` te le dira maintenant explicitement au lieu de lâcher un
`ModuleNotFoundError` sur un chemin interne de numpy.

Puis les mêmes vérifications qu'aux niveaux 3 et 4, sur les ports publiés.
Le `HEALTHCHECK` du Dockerfile doit faire passer le conteneur en
`healthy` au bout d'une minute environ :

```bash
docker ps --format '{{.Names}}\t{{.Status}}'
```

À vérifier aussi, parce que c'est ce qui casse en déploiement :

- **sans `--env-file .env`, le conteneur doit mourir au démarrage**, pas
  démarrer à moitié. `initialiser()` laisse remonter l'exception pour que
  l'orchestrateur relance. C'est ce comportement qui a rendu la panne
  numpy visible immédiatement plutôt qu'au premier appel utilisateur.
- l'image ne contient ni le bundle ni les résumés, ils sont téléchargés à
  chaque démarrage. Un `docker run` après un `rebuild_artifacts` prend
  donc en compte les nouveaux artéfacts sans reconstruction d'image.

## Opérations qui modifient le bucket

Rien de ce qui précède n'écrit sur S3. Les commandes ci-dessous, si.
À ne pas lancer pour « tester ».

```bash
python -m scripts.add_book --titre "Madame Bovary"          # ajoute un livre
python -m scripts.rebuild_pipeline --filtre "munchausen"    # rejoue clean/tokens/annotations
python -m scripts.rebuild_artifacts                         # regénère TF-IDF, Word2Vec, résumés
python build_serving_bundle.py                              # écrase le bundle servi
```

`rebuild_artifacts --nettoyer-orphelins` **supprime des objets S3** à
partir d'une différence d'ensembles. C'est précisément la fonction que
`tests/test_rebuild_artifacts.py` encadre. Ne la lance pas sans avoir
vérifié le contenu du bucket avant.

Chaîne complète après ajout d'un livre :

```bash
python -m scripts.add_book --titre "Notre-Dame de Paris" --rebuild
python build_serving_bundle.py
# puis redémarrer l'API pour qu'elle recharge le bundle
```

## Dépannage

| symptôme | cause | correction |
|---|---|---|
| `RuntimeError: Credentials Scaleway manquants` | pas de `.env` ou clés absentes | créer `.env` avec `SCW_ACCESS_KEY` et `SCW_SECRET_KEY` |
| `/api/identifier` renvoie 500, logs `[E050] Can't find model` | modèle spaCy non installé | `python -m spacy download fr_core_news_sm` |
| Streamlit : « Backend indisponible » | API pas lancée, ou `API_URL` faux | lancer `python api.py`, vérifier `API_URL` |
| `/health` renvoie `ready:false` | bundle pas encore chargé | attendre 10 s, sinon lire les logs serveur |
| pip échoue en compilant gensim ou tokenizers | Python plus récent que 3.12 | recréer le venv avec `python3.12` |
| `/api/extrait/<slug>` renvoie 500 `NoSuchKey` | le listing de `clean/` n'a pas trouvé le livre, repli sur le chemin reconstruit depuis le genre | vérifier le log `n/26 textes nettoyés localisés` au démarrage |
| le benchmark de résumé tourne indéfiniment | premier téléchargement de CamemBERT (440 Mo) | laisser finir, ou tester d'abord avec `--limite 3` |
| conteneur : `RuntimeError: Le bundle n'est pas lisible avec numpy X` | version majeure de numpy différente entre `requirements.txt` et `requirements-dev.txt` | aligner les deux, voir le niveau 7 |
| build : `ValueError: numpy.dtype size changed` au `spacy download` | spacy 3.7.x résout thinc 8.2.4, compilé contre numpy 1.x | garder `spacy>=3.8` dans `requirements.txt` |
