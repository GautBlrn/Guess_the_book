# Évaluation : ce qui est mesuré, et ce que les mesures ont changé

Document de référence sur la chaîne d'évaluation du projet. Il décrit ce
qui est calculé, comment, et surtout les décisions de configuration que
les mesures ont imposées. Les chiffres cités viennent tous d'exécutions
réelles sur le corpus de 26 livres, reproductibles avec les commandes du
[guide de test](guide_test.md).

## 1. Architecture

Deux modules, séparés par une frontière stricte.

**`pipeline/evaluation.py`** ne contient que des fonctions pures. Aucun
accès S3, aucun chargement de modèle. Il est donc testable sans
credentials et sans téléchargement.

- Recherche : `rappel_at_k`, `precision_at_k`, `f1_at_k`, `mrr`,
  `resumer_metriques_recherche`
- Résumé : `rouge_n`, `rouge_l`, `rouge_lsum`, `rouge_complet`, `bleu`,
  `agreger_scores`

**`pipeline/benchmark.py`** fait l'orchestration : construction du jeu de
test, boucles d'expériences, agrégation en DataFrame. Il prend le corpus
en argument plutôt que de le charger lui-même, ce qui permet de tester
les boucles avec un corpus synthétique.

**`scripts/run_benchmark.py`** expose le tout en ligne de commande et
exporte les tableaux en CSV.

### Pourquoi les métriques sont faites main

Cohérence avec le reste du projet (TF-IDF est déjà réimplémenté) et une
dépendance de moins au runtime. Le prix à payer est qu'il faut prouver
qu'elles sont correctes : `tests/test_evaluation.py` compare ROUGE-1,
ROUGE-2, ROUGE-L et ROUGE-Lsum à `rouge-score` (implémentation Google) et
BLEU à `nltk`, sur des tirages aléatoires seedés. Écart maximum mesuré :
`0.00e+00`.

Ces deux paquets ne servent que de référence de test. Ils ne sont jamais
importés par `pipeline/`, et sans eux la suite reste verte : les tests de
comparaison sont sautés.

## 2. Tâche de recherche

Identifier un livre à partir d'un extrait. Un seul document pertinent par
requête.

### Conséquence à assumer

Avec un unique document pertinent, `Precision@k = Rappel@k / k`. Les deux
portent donc la même information à un facteur près. Les afficher tous les
deux est demandé, mais écrire qu'ils « se confirment mutuellement » serait
faux : ils sont redondants par construction. La métrique qui apporte
réellement quelque chose est le MRR, qui distingue « bon livre en rang 2 »
de « bon livre en rang 9 ».

Même remarque pour les métriques de similarité. Sur des vecteurs
L2-normalisés :

```
||u - v||² = ||u||² + ||v||² - 2 u·v = 2 - 2 cos(u, v)
```

La distance euclidienne est une fonction strictement décroissante du
cosinus, donc **les deux produisent le même classement**. Sur le corpus
réel, les deux lignes du tableau sont égales à la sixième décimale. Ce
n'est pas une confirmation croisée, c'est la même mesure écrite deux fois.
Seul Jaccard apporte une information différente, et il perd nettement
(MRR 0,85 contre 0,98), ce qui est attendu puisqu'il jette les poids IDF
et les fréquences.

### Résultats

520 extraits, 26 livres, longueur tirée dans la plage 7-70 termes
informatifs (médiane 38, soit environ 108 mots collés), `min_df=1`,
cosinus.

| représentation | top-1 | @3 | @5 | @10 | MRR |
|---|---|---|---|---|---|
| lemmes_1g | 87,50 % | 95,38 % | 98,08 % | 99,04 % | 0,919 |
| lemmes_12g | 97,88 % | 100 % | 100 % | 100 % | 0,989 |
| tokens_12g | 98,08 % | 100 % | 100 % | 100 % | 0,990 |

### Le biais à écrire noir sur blanc

Les extraits sont tirés des mêmes livres que ceux indexés, donc l'extrait
est littéralement contenu dans le document cible. Les scores sont
mécaniquement hauts et ne se transposent pas à un extrait d'un livre
absent du corpus. C'est voulu, c'est exactement l'usage de l'app, mais un
lecteur qui l'ignore lira 98 % comme une performance alors que c'est en
partie une tautologie.

## 3. Tâche de résumé

Comparer trois méthodes extractives : `tfidf_naif` (top-k sur la
similarité au centre TF-IDF), `textrank` (top-k sur la centralité du
graphe), `mmr` (score hybride puis sélection MMR, la méthode de
production).

### ROUGE ne classe pas ces méthodes

Le corpus n'a pas de résumés de référence écrits par des humains. On
compare donc chaque résumé au **texte intégral**, ce qui change tout :

- la **précision** vaut 1 pour toute méthode extractive, par
  construction. Chaque phrase est copiée du livre, donc chacun de ses
  unigrammes s'y trouve.
- le **rappel** est dominé par la longueur du résumé. Le clipping compte
  `min(occurrences)`, donc à `k` fixé toutes les méthodes obtiennent
  quasiment la même valeur.
- donc le F1 aussi.

Ce n'est pas une hypothèse. Mesuré sur les 26 livres :
`rouge1_precision = 1,0` avec un **écart-type nul**, pour les trois
méthodes.

`rouge2_precision` vaut 0,97 et non 1,0. Les 3 % manquants sont les
bigrammes à cheval sur la jointure entre deux phrases sélectionnées, un
artefact de concaténation et non du contenu.

**Conclusion : ROUGE contre le texte intégral ne classe pas des méthodes
extractives.** Il est calculé parce que le sujet le demande et parce
qu'un effondrement signalerait un résumé cassé, mais ce n'est pas lui qui
tranche.

### Les métriques qui discriminent

`redondance` : cosinus moyen entre paires de phrases retenues. C'est ce
que MMR optimise explicitement et que ROUGE ne voit pas, puisque deux
phrases qui disent la même chose avec des mots différents comptent double
en ROUGE.

`couverture` : part de la masse lexicale du livre touchée par le résumé,
pondérée par fréquence. À longueur égale, un résumé diversifié en couvre
plus qu'un empilement de phrases semblables.

### Résultats

26 livres, `k=10`, `lambda=0.6`, poids TextRank à 0,20.

| méthode | redondance | couverture | rouge1_precision |
|---|---|---|---|
| tfidf_naif | 0,4200 | 0,2575 | 1,0 (σ = 0) |
| **mmr** | **0,3953** | **0,2620** | 1,0 (σ = 0) |
| textrank | 0,5840 | 0,2569 | 1,0 (σ = 0) |

### Le résultat inconfortable

Avec la pondération d'origine (TextRank à 0,30), la comparaison appariée
livre par livre contre `tfidf_naif`, test des signes bilatéral :

| comparaison | MMR gagne | diff. moyenne | p |
|---|---|---|---|
| redondance, MMR vs tfidf_naif | 7 / 26 | +0,0185 | **0,029** |
| redondance, MMR vs textrank | 26 / 26 | −0,1455 | < 0,0001 |
| couverture, MMR vs tfidf_naif | 17 / 26 | +0,0098 | 0,169 |

Le mécanisme MMR fonctionne : il bat TextRank sur 26 livres sur 26. Mais
le pipeline complet était **significativement plus redondant que le
simple top-k TF-IDF**, ce qu'il est censé minimiser. La pertinence valant
0,70 centre + 0,30 TextRank, et TextRank étant de loin le plus redondant
des trois, le terme de diversité passait son temps à réparer ce que la
composante TextRank introduisait, sans y parvenir complètement.

Le balayage du poids TextRank montre que c'est un arbitrage, pas un
optimum :

| poids TextRank | redondance | MMR gagne | p | couverture | MMR gagne | p |
|---|---|---|---|---|---|---|
| 0,30 | 0,4385 | 7/26 | 0,029 | 0,2673 | 17/26 | 0,169 |
| **0,20** | **0,3953** | 16/26 | 0,327 | **0,2620** | 15/26 | 0,557 |
| 0,10 | 0,3259 | 26/26 | < 0,0001 | 0,2426 | 7/26 | 0,029 |
| 0,00 | 0,2651 | 26/26 | < 0,0001 | 0,2264 | 1/26 | < 0,0001 |

Baisser le poids fait chuter la redondance, mais dégrade la couverture en
miroir. TextRank apporte la centralité thématique ; sans lui MMR choisit
des phrases mutuellement dissemblables mais individuellement peu
représentatives.

**Aucune pondération ne rend MMR clairement supérieur au top-k TF-IDF
naïf sur ces deux critères.** 0,20 est le point de parité : les deux
tests sont non significatifs dans les deux sens. C'est une parité, pas
une victoire, et le rapport doit le dire.

Réserves de méthode : dix tests de signes ont été lancés, donc les `p`
autour de 0,03 sont fragiles à toute correction pour comparaisons
multiples, ceux à moins de 0,0001 tiennent. Et redondance et couverture
restent des proxys intrinsèques : ils ne mesurent ni la lisibilité ni la
cohérence narrative, qui étaient les motivations qualitatives de la
refonte v4. Un jugement humain sur quelques livres trancherait ce que ces
deux chiffres ne tranchent pas.

## 4. Les trois décisions prises

### `min_df` passe de 2 à 1

Le réflexe « min_df=2 pour filtrer le bruit » vient de la classification
de documents, où un terme vu une seule fois est du bruit. Ici la tâche
est **inverse** : un terme présent dans un seul livre du corpus n'est pas
du bruit, c'est l'indice parfait. Un nom de personnage, un lieu, un mot
forgé par l'auteur. `min_df=2` les éliminait tous, soit 52 % du
vocabulaire.

520 extraits fixes de 200 termes, comparaison appariée :

| config | min_df=2 | min_df=1 |
|---|---|---|
| lemmes_1g | 95,77 % (22 erreurs) | 98,46 % (8) |
| lemmes_12g | 98,08 % (10 erreurs) | **99,62 % (2)** |
| tokens_12g | 98,08 % (10 erreurs) | 99,62 % (2) |

Les 8 échecs réparés sont des paires de livres réellement voisins : deux
mémoires napoléoniens, deux souvenirs de théâtre du XIXe, deux romans de
la Révolution. Le signal qui les sépare est exactement le vocabulaire
rare que `min_df=2` supprimait. Les 2 échecs qui résistent sont
Monte-Cristo Tome I confondu avec le Tome II, soit le même roman en deux
volumes : irréductible.

Le gain **grandit** avec la taille du corpus, il ne s'érode pas. Mesuré
par sous-échantillonnage, 5 tirages par taille :

| N livres | % vocab df=1 | gain de min_df=1 |
|---|---|---|
| 5 | 61,0 % | +0,60 point |
| 10 | 55,4 % | +0,60 |
| 15 | 54,2 % | +1,33 |
| 20 | 53,1 % | +1,35 |
| 26 | 51,9 % | **+2,69** |

La proportion de termes rares baisse, mais le nombre de distracteurs
monte, donc leur pouvoir discriminant compte davantage.

### Le bundle servi passe de `lemmes_1g` à `lemmes_12g`

C'est une conséquence du point suivant. À extraits fixes de 200 termes,
l'écart entre les deux configs est de 1,2 point et ne justifie pas 145 Mo
de mémoire supplémentaire. À longueur d'extrait réaliste, il est de
**10,4 points** (87,50 % contre 97,88 %). Un extrait court contient peu
d'unigrammes, donc la preuve apportée par les bigrammes y pèse
proportionnellement bien plus.

Coût : vocabulaire de 734 223 termes, matrice dense de 153 Mo, bundle
sérialisé de 178,6 Mo.

### Le poids TextRank passe de 0,30 à 0,20

Voir la section 3.

## 5. Le protocole de tirage des extraits

C'est le changement le plus important du lot, parce qu'il invalide
rétroactivement des conclusions.

**Avant** : 200 termes informatifs, longueur fixe. Le ratio mesuré sur le
corpus est de 0,349 terme informatif par token brut, donc 200 termes
valent environ **573 mots collés**. Or le slider de l'app plafonne à 300
mots et vaut 80 par défaut. Le benchmark mesurait un scénario que le
produit ne peut pas produire.

| mots dans l'app | ≈ termes | top-1 (lemmes_1g) |
|---|---|---|
| 30 (minimum) | 10 | 63,85 % |
| **80 (défaut)** | 28 | **85,38 %** |
| 150 | 52 | 91,73 % |
| 300 (maximum) | 105 | 96,54 % |

**Après** : la longueur est **tirée** dans une plage, pas fixée.
L'utilisateur colle ce qu'il a sous la main, il ne compte pas ses mots :
la longueur est une variable aléatoire du problème, pas un paramètre
d'expérience.

Effet secondaire bénéfique : à 200 termes, Rappel@3, @5 et @10 valaient
tous 1,0, donc le graphique « Rappel@k pour k de 1 à 10 » demandé par le
sujet était une droite plate sans information. Avec le protocole
réaliste, `lemmes_1g` donne 87,5 → 95,4 → 98,1 → 99,0.

Pour retrouver une courbe de difficulté en fonction de la longueur,
passer un `int` à `tirer_extraits` plutôt que de changer la plage.

### Grossir le corpus n'aiderait pas la recherche

La mesure de scaling montre qu'à `min_df=1` le top-1 reste entre 97,9 %
et 99,2 % de 5 à 26 livres, sans tendance. Le paramètre qui gouverne la
difficulté est la longueur d'extrait, pas le nombre de livres. Passer à
60 livres servirait uniquement les statistiques du résumé, où
`p = 0,169` sur la couverture reste non concluant à n = 26.

## 6. Limites connues

`TfIdfMaison.fit_transform` fait un `toarray()`, donc la matrice est
dense. Le vocabulaire croît avec le corpus, donc la mémoire grimpe en
gros comme N². 153 Mo à 26 livres avec `lemmes_12g`, plusieurs Go à 100.
Passer en `csr_matrix` est le prérequis à toute croissance du corpus avec
la config servie. `similarites_cosinus` fait `X @ requete`, qui marche
nativement en sparse ; `_l2_normaliser` et `_ensembles_non_nuls`
supposent du dense et devront être adaptés. `similarites_euclidiennes`
n'a pas besoin d'être portée, puisqu'elle est prouvée équivalente au
cosinus.

Les notebooks 05 et 07 ont été exécutés avant ces changements. Leurs
sorties utilisent `min_df=2` et l'ancienne pondération, et le 05 conclut
que `lemmes_1g` est la meilleure config, ce que le benchmark contredit.
Le 05 tire de plus `debut_l` et `debut_t` indépendamment, donc ses
extraits « lemmes » et « tokens » ne portent pas sur les mêmes passages
et sa comparaison entre représentations n'est pas contrôlée. Les rejouer
en appelant `pipeline.benchmark` supprimerait le problème.
