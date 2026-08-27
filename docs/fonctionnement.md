# Comment le programme fonctionne

Document de référence sur l'architecture et les mécanismes du projet.
Pour lancer et vérifier le programme, voir le [guide de test](guide_test.md).
Pour les chiffres d'évaluation et les décisions qu'ils ont imposées, voir
[evaluation.md](evaluation.md).

## 1. Ce que fait le programme

Deux fonctions, sur un corpus de 26 livres français du projet Gutenberg :

1. **Identifier un livre à partir d'un extrait.** Tu colles un passage,
   l'app te renvoie les livres du corpus classés par similarité, avec un
   score.
2. **Afficher un résumé extractif** de chaque livre du corpus, produit
   hors ligne par MMR.

Tout ce qui est lourd (téléchargement Gutenberg, nettoyage, annotation
spaCy, entraînement TF-IDF, génération des résumés) se fait **hors
ligne**, en amont, et atterrit sur S3. L'application servie ne fait que
lire des artéfacts déjà calculés.

## 2. Vue d'ensemble

```
  Gutenberg (pg_catalog.csv + téléchargement HTTP)
        |
        |  scripts/upload_corpus.py, scripts/add_book.py
        v
   S3  raw/          texte brut, avec en-têtes Gutenberg
        |  cleaning.nettoyer
   S3  clean/        texte seul
        |  tokenization.tokeniser
   S3  tokens/       tokens normalisés (Parquet)
        |  annotation.annoter
   S3  annotations/  un token par ligne : lemme, POS, frontières de phrase
        |
        +--> scripts/rebuild_artifacts.py --> S3 artifacts/  (TF-IDF, Word2Vec)
        |                                     S3 summaries/  (résumés MMR)
        +--> build_serving_bundle.py     --> S3 artifacts/serving_bundle.pkl
                                                     |
                                                     v
                                        api.py (Flask)  <--HTTP--  app/ (Streamlit)
```

Le point de découpe est le **bundle de serving**. Le pipeline produit un
seul `.pkl` qui contient le vectoriseur TF-IDF entraîné, la matrice du
corpus et l'index des livres. L'API charge ce fichier au démarrage et ne
retouche plus à S3 ensuite, sauf pour lire un extrait aléatoire.

## 3. Le corpus et son stockage

Bucket Scaleway `gautier-blairon-bucket` (région `fr-par`), un préfixe par
étage du pipeline, déclarés dans `PREFIXES` de `pipeline/config.py`.
Convention de clé identique partout :

```
<etage>/<genre>/<auteur_slug>/<livre_slug>.<ext>
clean/adventure/dumas_alexandre/pg17990_le_comte_de_monte_cristo_tome_ii.txt
```

Le `livre_slug` porte l'identifiant Gutenberg en préfixe (`pg17990_`), ce
qui garantit l'unicité même si deux éditions portent le même titre.

État actuel du corpus :

| grandeur | valeur | corpus initial |
|---|---|---|
| livres | 291 | 26 |
| auteurs | 221 | 23 |
| genres | 16 | 9 |
| tokens | 19 925 564 | 2 719 190 |
| phrases | 997 851 | 137 244 |

Répartition par genre, une fois la collecte stratifiée passée :

```
novel 23, biography 22, historical_fiction 21, philosophy 20, mystery 19,
travel 19, adventure 18, mythology 18, romance 18, short_stories 18,
drama 17, scifi_fantasy 17, essays 16, humour 16, children 15, poetry 14
```

De 14 à 23 livres par genre, quand le catalogue Gutenberg dont ils sortent
compte 710 romans pour 42 livres de mythologie. C'est l'objet du quota par
genre de `scripts/collect_corpus.py` : la difficulté d'identification vient
des voisins proches, donc d'un corpus qui contient plusieurs livres du même
registre, pas d'un corpus qui en contient beaucoup au total.

Le corpus initial de 26 livres reste en colonne de droite parce que la
plupart des mesures d'arbitrage documentées dans `docs/evaluation.md` ont
été prises dessus et n'ont pas encore été rejouées à cette échelle.

Le genre vient des Bookshelves Gutenberg, mappés par `GENRE_MAPPING`.
`GENRE_OVERRIDES` corrige au runtime les oeuvres en plusieurs tomes que
Gutenberg range différemment d'un tome à l'autre (le cas Monte-Cristo).

### Les accès S3 reprennent sur incident réseau

Charger le corpus, c'est un GET par livre, en série. À 26 livres une
coupure passagère était improbable et sans conséquence. À 300, chaque
chargement enchaîne 300 requêtes, et une seule qui échoue faisait perdre
toutes les précédentes : c'est arrivé sur un `ReadTimeoutError` au milieu
d'un chargement de 291 livres.

`storage._lire_corps` réémet donc la requête, jusqu'à `max_tentatives`
fois, avec une attente qui double à chaque essai. Deux points méritent
d'être connus :

**Pourquoi la reprise de botocore ne suffit pas.** `ReadTimeoutError` est
levée pendant `Body.read()`, c'est-à-dire après que botocore a rendu la
main sur la requête. Sa couche de réessai n'a plus prise, et le flux est
mort : il faut refaire le GET entier, pas relire le corps. La configuration
`retries` du client reste utile, mais seulement pour ce qui échoue avant
que la réponse commence à arriver.

**Ce qui n'est pas rejoué.** Une clé absente, un refus d'authentification,
tout ce qui n'est pas un incident serveur remonte immédiatement. Les rejouer
cinq fois ne ferait que retarder un message d'erreur exact. Le tri se fait
dans `_est_transitoire` : erreurs réseau, plus les codes S3 d'incident
serveur et tout statut 5xx.

Les seuils vivent dans `S3_PARAMS` de `pipeline/config.py`, et
`tests/test_storage.py` couvre les deux comportements sans toucher au
réseau.

### Le chargement rend compte de son avancement

`charger_corpus` et `iter_corpus` affichent une ligne tous les
`PAS_PROGRESSION` livres, avec le débit et une estimation du temps
restant. Sur un corpus de quelques centaines de livres, un chargement
silencieux de plusieurs minutes est indistinguable d'un blocage.

## 4. Le pipeline de traitement

Trois étages, un module chacun, tous importables sans effet de bord.

**`cleaning.nettoyer`** enlève l'en-tête et le pied de page Gutenberg. La
détection se fait dans une fenêtre de 15 000 caractères en tête
(`CLEAN_PARAMS`), pas sur tout le fichier : les marqueurs Gutenberg
apparaissent aussi dans le corps de certains textes, et chercher partout
tronquait des livres.

**`tokenization.tokeniser`** utilise un spaCy `blank("fr")`, donc pas de
modèle statistique : à cet étage on veut une segmentation en tokens et
une normalisation typographique (apostrophes, tirets, guillemets), pas
une analyse.

**`annotation.annoter`** charge `fr_core_news_sm` avec la NER désactivée
et traite le texte par blocs de 100 000 caractères. Sortie : un DataFrame
avec un token par ligne, sa forme, son lemme, son POS, et les frontières
de phrase (`sent_id`). C'est ce `sent_id` qui sert de clé stable partout
ensuite, y compris dans les métriques de résumé.

Les trois étages sont enchaînés par `pipeline_full.pipeline_un_livre`,
utilisé par `scripts/add_book.py` et `scripts/rebuild_pipeline.py`.

## 5. L'identification d'un livre

### La représentation

`representations.TfIdfMaison` est une réimplémentation de TF-IDF, avec les
conventions sklearn (lissage de l'IDF, normalisation L2), validée contre
`sklearn.TfidfVectorizer` dans le notebook 05. Trois configurations sont
construites et évaluées, déclarées une seule fois dans `TFIDF_CONFIGS` :
`lemmes_1g`, `lemmes_12g`, `tokens_12g`.

Deux réglages méritent d'être compris :

- **`min_df=1`**, à rebours de l'habitude. En classification de documents,
  un terme vu une fois est du bruit. Ici la tâche est inverse : un terme
  présent dans un seul livre est l'indice parfait, un nom de personnage ou
  un lieu. `min_df=2` supprimait 52 % du vocabulaire, et avec lui la seule
  chose qui séparait deux mémoires napoléoniens.
- **`max_df_ratio=0.85`** élimine les termes présents dans plus de 85 %
  des livres, qui ne discriminent rien.

Le prix de `min_df=1` est en bigrammes : ils sont presque tous uniques,
donc quasiment tous conservés. Le vocabulaire de `lemmes_12g` monte à
734 223 termes et la matrice dense à 153 Mo pour 26 livres. C'est assumé,
la justification chiffrée est dans `evaluation.md`.

### Le classement

`similarites_cosinus(X, requete)` sur des vecteurs L2-normalisés, donc un
simple produit matriciel. Les distances euclidienne et cosinus produisent
le **même classement** sur des vecteurs normalisés (`||u-v||² = 2-2cos`),
ce sont deux écritures de la même mesure. Jaccard est la seule des trois
qui apporte une information différente, et elle perd nettement puisqu'elle
jette les poids IDF.

### Le bundle de serving

`build_serving_bundle.py` charge le corpus, entraîne le TF-IDF, et
sérialise un dict `{version, config, vectoriseur, matrice, index}` vers
`artifacts/serving_bundle.pkl`. L'`index` contient une entrée par livre :
métadonnées, `n_tokens`, `n_phrases`, et la clé S3 de son résumé.

Le bundle actuellement en place est en `lemmes_12g` (178,6 Mo, vocabulaire
de 734 223 termes). Le choix de `lemmes_12g` plutôt que `lemmes_1g` vient
de la mesure à longueur d'extrait réaliste : 10,4 points de top-1 d'écart.
Attention, le commentaire de `TFIDF_PARAMS` dans `config.py` affirme
encore que la config servie est `lemmes_1g`, il n'a pas suivi la décision.

## 6. Le résumé extractif

Chaîne en trois temps, dans `pipeline/summarization.py` :

1. **Filtrage des phrases candidates.** `MMR_PARAMS` impose entre 8 et 60
   tokens informatifs, une densité informative d'au moins 0,30 et au plus
   40 % de noms propres. Le seuil bas était à 12 initialement, ce qui
   jetait 70 à 90 % des phrases : tous les dialogues et toutes les phrases
   d'action. À 8, il en reste 40 à 50 %.
2. **Score de pertinence hybride**, pondéré par `EMB_PARAMS["poids_score"]` :
   0,70 similarité au centre thématique du livre, 0,20 centralité TextRank
   (PageRank sur le graphe de similarités, environ 10 % des arêtes les plus
   fortes conservées), 0,05 bonus de longueur naturelle. Les similarités
   sont calculées sur des embeddings `sentence-camembert-base`.
3. **Sélection MMR** : à chaque itération on prend la phrase qui maximise
   `lambda * pertinence - (1-lambda) * max_similarité_aux_déjà_choisies`.
   `k=10`, `lambda=0,6`.

Le poids TextRank est passé de 0,30 à 0,20 parce qu'à 0,30 le résumé MMR
était mesurablement **plus** redondant que le simple top-k TF-IDF, ce
qu'il est censé minimiser. Le détail du balayage et la conclusion (0,20
est un point de parité, pas une victoire) sont dans `evaluation.md`.

`pipeline/abstractive.py` contient une variante abstractive avec BARThez
(`barthez-orangesum-abstract`). Elle n'est pas servie par l'API : les
résumés exposés sont les fichiers `summaries/*.txt` générés par MMR.

## 7. L'architecture de service

Deux process dans un seul conteneur, orchestrés par `start.sh`.

### Backend Flask, `api.py`, port 5000

Au démarrage, `initialiser()` télécharge le bundle, précharge en RAM les
26 résumés, puis liste `clean/` une fois pour associer chaque `livre_slug`
à la clé S3 réelle de son texte nettoyé (`indexer_cleans`). Ce dernier
point évite de reconstruire le chemin depuis le genre de l'index, qui peut
avoir été corrigé au runtime par `GENRE_OVERRIDES` sans que l'objet S3 ait
bougé. Mesuré : 2,7 s pour le bundle, 5,2 s pour les résumés, 0,3 s pour
le listing, soit prêt en une dizaine de secondes. spaCy est chargé
paresseusement au premier `/api/identifier` pour que le healthcheck
réponde vite.

| route | méthode | rôle |
|---|---|---|
| `/health` | GET | statut, `ready`, nombre de livres et de résumés |
| `/api/livres` | GET | index du corpus, sans la matrice |
| `/api/identifier` | POST | `{extrait, top_k}` vers le top-k classé |
| `/api/resume/<slug>` | GET | résumé MMR préchargé |
| `/api/extrait/<slug>?taille=N` | GET | extrait aléatoire de N mots, lu depuis `clean/` |

`top_k` est borné à 10 côté serveur. Les paramètres TF-IDF sont figés au
build du bundle : l'utilisateur final ne peut pas les changer, c'est
volontaire.

Latence mesurée sur `/api/identifier` : 1,8 s au premier appel (chargement
spaCy inclus), puis environ 15 ms.

### Frontend Streamlit, `app/`, port 8501

L'app ne touche ni S3 ni spaCy, elle ne fait que des appels HTTP vers
`API_URL` (défaut `http://localhost:5000`). Conséquence : elle démarre
instantanément et peut être mise à l'échelle séparément du backend.

Trois pages :

- **Home** : zone de saisie, plus un bouton « tirer un extrait » qui
  appelle `/api/extrait` sur un livre choisi (30 à 300 mots, 80 par
  défaut). Quand l'extrait vient de ce bouton et n'a pas été édité, l'app
  connaît la bonne réponse et affiche le **rang du vrai livre**, ce qui en
  fait un mode démo auto-évalué.
- **Catalogue** : métriques du corpus, tableau détaillé, répartition par
  genre. Lecture seule.
- **Résumés** : sélection d'un livre, affichage du résumé MMR,
  téléchargement en `.txt`.

`app/dev/2_Exploration.py` est hors du dossier `pages/`, donc Streamlit ne
le charge pas. C'est volontaire, mais le fichier est aussi devenu
incompatible : il importe `get_corpus` de `utils`, qui n'existe plus
depuis que `utils.py` est passé en client HTTP pur.

### Docker

`start.sh` lance gunicorn (1 worker, `--preload`), attend que `/health`
réponde pendant au plus 30 s, puis passe la main à Streamlit avec `exec`.
Si gunicorn meurt, Streamlit est tué pour que Docker relance l'ensemble.

L'image ne contient ni le bundle ni les résumés : ils sont téléchargés
depuis S3 à chaque démarrage. On peut donc régénérer les artéfacts sans
reconstruire l'image. Elle ne contient pas non plus `scripts/`, l'app
servie n'en a pas besoin.

Les crédentials sont passés au runtime par `--env-file .env`. Le
commentaire du `Dockerfile` parle de variables `AWS_*` : c'est faux,
`pipeline/storage.py` lit `SCW_ACCESS_KEY` et `SCW_SECRET_KEY`.

Le bundle est un pickle, ce qui lie l'environnement qui le produit
(`requirements-dev.txt`) à celui qui le lit (`requirements.txt`). La
version majeure de numpy doit donc être la même des deux côtés : numpy 2 a
renommé `numpy.core` en `numpy._core` et le pickle référence ce chemin de
module. C'est ce qui empêchait le conteneur de démarrer jusqu'au
19 août 2026. Les pins des deux fichiers portent l'avertissement, et
`charger_bundle()` retraduit désormais l'échec en message explicite.

## 8. Ce qui est configurable, et où

Tout est dans `pipeline/config.py` : `BUCKET`, `PREFIXES`, `GENRE_MAPPING`,
`GENRE_OVERRIDES`, `CLEAN_PARAMS`, `ANNOTATE_PARAMS`, `TFIDF_PARAMS`,
`TFIDF_CONFIGS`, `W2V_PARAMS`, `MMR_PARAMS`, `EMB_PARAMS`,
`BARTHEZ_PARAMS`, `BENCH_PARAMS`.

`TFIDF_CONFIGS` est la source unique des représentations : le benchmark
mesure exactement les configurations que `rebuild_artifacts` construit,
pour que les chiffres du rapport décrivent ce qui tourne réellement.

Deux réglages ne se propagent pas tout seuls après modification :

- changer `TFIDF_PARAMS` ou `TFIDF_CONFIGS` impose de rejouer
  `rebuild_artifacts` puis `build_serving_bundle`, sinon l'API sert
  toujours l'ancien vectoriseur ;
- changer `MMR_PARAMS` ou `EMB_PARAMS` impose de régénérer les résumés,
  que l'API précharge tels quels.

## 9. Écarts connus entre la documentation et le code

Listés ici pour éviter de perdre du temps dessus :

- `README.md` décrit encore quatre pages Streamlit avec des sliders MMR
  interactifs et un onglet d'ajout de livre dans le Catalogue. Ces
  fonctions ont disparu avec le passage au backend Flask, la config étant
  désormais figée côté serveur.
- Le commentaire de `TFIDF_PARAMS` annonce `lemmes_1g` comme config
  servie, alors que le bundle en place est `lemmes_12g`.
- Le commentaire du `Dockerfile` cite des variables `AWS_*` inexistantes.
- L'objet `clean/` de Monte-Cristo Tome I est toujours rangé sous son
  genre d'origine (`historical_fiction`) alors que l'index porte le genre
  corrigé par `GENRE_OVERRIDES` (`adventure`). L'API ne s'y casse plus les
  dents (elle résout les clés par listing, cf. section 7), mais les deux
  tomes restent rangés dans deux dossiers différents sur S3. Un
  `rebuild_pipeline` les réalignerait, au prix d'un orphelin à nettoyer.
- Les notebooks 05 et 07 datent d'avant les décisions actées dans
  `evaluation.md` et leurs conclusions sont périmées.
