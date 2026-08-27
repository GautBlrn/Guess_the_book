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

1455 extraits, 291 livres, longueur tirée dans la plage 7-70 termes
informatifs, `min_df=1`, cosinus.

| représentation | vocabulaire | top-1 | @3 | @5 | @10 | MRR |
|---|---|---|---|---|---|---|
| lemmes_1g | 143 882 | 70,31 % | 84,26 % | 88,52 % | 92,92 % | 0,785 |
| lemmes_12g | 4 144 734 | 92,65 % | 98,97 % | 99,59 % | 100 % | 0,958 |
| tokens_12g | 5 048 963 | **93,95 %** | 99,24 % | 99,73 % | 100 % | **0,965** |

Jaccard reste très en retrait, et l'écart se creuse avec le corpus :
28,18 % de top-1 en `lemmes_1g` contre 70,31 % pour le cosinus, avec un
rang médian de 5. Jeter les poids IDF coûte d'autant plus cher qu'il y a
de livres entre lesquels trancher.

### Ce que le passage à 291 livres a changé

Pour mémoire, les mêmes mesures sur le corpus de 26 livres (520 extraits) :

| représentation | top-1 à 26 livres | top-1 à 291 livres | écart |
|---|---|---|---|
| lemmes_1g | 87,50 % | 70,31 % | -17,2 pts |
| lemmes_12g | 97,88 % | 92,65 % | -5,2 pts |
| tokens_12g | 98,08 % | 93,95 % | -4,1 pts |

**La baisse était l'issue attendue**, et c'est un résultat, pas une
régression : identifier un livre parmi 291 est plus difficile que parmi 26.
Un corpus plus grand ne rend pas le système meilleur, il rend la mesure
honnête.

**Le benchmark a cessé de saturer**, ce qui était l'objectif principal. À
26 livres, Rappel@3, @5 et @10 valaient tous 1,0 pour les deux configs
bigrammes : la courbe demandée par le sujet était plate et les configs
indistinguables au-delà du rang 1. `lemmes_1g` donne maintenant une vraie
courbe, de 0,703 à 0,929 sur dix rangs. Les configs bigrammes saturent
encore, mais seulement à partir du rang 8.

**L'écart entre unigrammes et bigrammes a explosé**, et c'est le résultat
le plus utile. Il valait 10,4 points à 26 livres, il en vaut **22,3** à
291. Le choix de servir une config bigrammes reposait sur un écart mesuré
dans un régime saturé ; il repose maintenant sur un écart franc, mesuré sur
1455 extraits. La raison est mécanique : un unigramme discriminant dans un
corpus de 26 livres ne l'est plus dans 291, alors qu'un bigramme reste
presque unique quelle que soit la taille du corpus.

**Le vocabulaire a grossi comme prévu**, de 734 k à 4,1 M de termes pour
`lemmes_12g`. Une matrice dense pèserait ici 291 × 4 144 734 × 8 octets,
soit **9,7 Go**, et 11,8 Go pour `tokens_12g`. Le calcul n'aurait pas
tourné. C'est la justification du passage au creux, vérifiée en vraie
grandeur plutôt qu'extrapolée.

À noter que l'extrapolation par loi de Heaps annonçait 8,3 M de termes à
300 livres, contre 4,1 M mesurés à 291 : elle surestimait d'un facteur 2,
l'exposant ayant été ajusté sur des corpus de 8 à 26 livres où le
vocabulaire croît plus vite qu'il ne le fait ensuite.

### Tests appariés entre représentations

Les taux agrégés ci-dessus ne suffisent pas à départager deux
représentations. Comparer 93,95 % et 92,65 % avec un test de proportions
supposerait deux échantillons indépendants, alors que c'est le **même** jeu
de 1455 extraits qui passe dans les deux, et que `tirer_extraits` fournit
les deux champs du **même passage**. Ignorer l'appariement jette
l'information la plus utile : deux représentations qui échouent sur les
mêmes extraits difficiles ne diffèrent pas vraiment, même si l'écart
agrégé paraît grand.

Deux tests, qui ne voient pas la même chose. **McNemar** exact (test
binomial sur les paires discordantes) ne regarde que le succès au rang 1,
et ne compte que les extraits où les deux configurations diffèrent.
**Wilcoxon** sur les rangs réciproques utilise davantage : passer du rang 9
au rang 2 compte, alors que McNemar l'ignore.

Seuil 0,05 ajusté à 0,0167 par Bonferroni pour trois comparaisons.

| comparaison | écart top-1 | A gagne | B gagne | p (McNemar) | p (Wilcoxon) |
|---|---|---|---|---|---|
| `tokens_12g` vs `lemmes_12g` | +1,31 pts | 25 | 6 | 8,8 × 10⁻⁴ | 1,6 × 10⁻⁴ |
| `lemmes_12g` vs `lemmes_1g` | +22,34 pts | 328 | 3 | 2,8 × 10⁻⁹³ | 3,5 × 10⁻⁶⁶ |
| `lemmes_12g` vs `min_df=2` | +10,03 pts | 150 | 4 | 2,0 × 10⁻³⁹ | 2,5 × 10⁻³² |

Les trois sont significatives, et les deux tests concordent à chaque fois.

**`tokens_12g` bat `lemmes_12g`, et ce n'est plus discutable.** À 26 livres
l'écart valait un extrait sur 520 et avait été jugé non concluant, à juste
titre. À 291 livres il porte sur 31 paires discordantes réparties 25 contre
6, ce qui ne s'explique pas par le hasard. Le bundle servi utilise
aujourd'hui `lemmes_12g` : **il devrait passer à `tokens_12g`**, au prix
d'un vocabulaire 22 % plus gros (5,0 M contre 4,1 M de termes).

L'interprétation est cohérente avec le reste : la lemmatisation normalise
les formes fléchies, ce qui aide à rapprocher deux occurrences d'un même
mot mais efface aussi des indices. Sur une tâche d'identification, la forme
exacte employée par l'auteur est elle-même une signature.

**`min_df = 1` se justifie mieux à 291 livres qu'à 26**, ce qui n'allait
pas de soi. Le réglage coûtait 1,5 point de top-1 à 26 livres ; il en
rapporte **10,03** à 291. Autrement dit, plus le corpus grandit, plus les
termes rares deviennent l'indice décisif, exactement l'argument qui avait
motivé le choix. Le prix reste le vocabulaire : 4,1 M de termes contre
773 k à `min_df = 2`, soit un facteur 5,4 en mémoire et en temps de calcul
pour ces 10 points.

Reproduction : `python -m scripts.tests_apparies`

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

**À 26 livres, aucune pondération ne rendait MMR clairement supérieur au
top-k TF-IDF naïf sur ces deux critères.** 0,20 était le point de parité :
les deux tests non significatifs dans les deux sens. Une parité, pas une
victoire.

### Ce que 291 livres ont changé : la parité devient une victoire

Les mêmes comparaisons appariées, rejouées sur 291 livres avec la
pondération retenue (TextRank à 0,20) et `lambda = 0,6`. Seuil 0,05 ajusté
à 0,0125 par Bonferroni pour quatre tests.

| comparaison | MMR gagne | diff. moyenne | IC 95 % | p (signes) | p (Wilcoxon) |
|---|---|---|---|---|---|
| redondance vs `tfidf_naif` | 196 / 291 | +0,0244 | [+0,0188, +0,0302] | 2,1 × 10⁻⁹ | 2,0 × 10⁻¹⁴ |
| couverture vs `tfidf_naif` | 179 / 291 | +0,0073 | [+0,0048, +0,0098] | 2,5 × 10⁻⁵ | 8,7 × 10⁻⁸ |
| redondance vs `textrank` | 290 / 291 | +0,1895 | [+0,1804, +0,1987] | 1,5 × 10⁻⁸⁵ | 3,0 × 10⁻⁴⁹ |
| couverture vs `textrank` | 142 / 291 | −0,0018 | [−0,0053, +0,0016] | 0,725 | 0,369 |

*(le signe est normalisé : positif signifie toujours que MMR est meilleur)*

**MMR bat `tfidf_naif` sur les deux critères à la fois, et les deux tests
concordent.** C'est le renversement de la conclusion précédente, et il ne
vient pas d'un changement de méthode mais du seul passage de 26 à 291
livres : le mécanisme faisait déjà mieux, l'échantillon ne permettait pas
de le voir. À n = 26, la couverture donnait 17/26 et `p = 0,169` ; à
n = 291, la même proportion donne 179/291 et `p = 2,5 × 10⁻⁵`.

**Contre TextRank, le partage est net et instructif.** MMR est moins
redondant sur 290 livres sur 291, ce qui est écrasant, mais les deux sont
équivalents en couverture (142 contre 149, intervalle de confiance
contenant zéro). Autrement dit MMR ne sacrifie rien en représentativité
pour gagner toute cette diversité, ce qui est exactement le comportement
attendu d'un terme de diversité bien réglé.

### Significatif ne veut pas dire important

Les intervalles de confiance sont là pour ça, et ils tempèrent. L'avantage
de couverture sur `tfidf_naif` vaut +0,0073 pour une couverture moyenne de
0,25, soit **3 % en relatif**. Celui de redondance vaut +0,0244 pour une
moyenne de 0,40, soit 6 %. Ces écarts sont réels et réguliers, ils ne sont
pas spectaculaires. Un `p` de 10⁻¹⁴ mesure la certitude que l'écart existe,
pas sa taille.

Réserves de méthode : redondance et couverture restent des proxys
intrinsèques. Ils ne mesurent ni la lisibilité ni la cohérence narrative,
qui étaient les motivations qualitatives de la refonte v4. Un jugement
humain sur quelques livres trancherait ce que ces deux chiffres ne
tranchent pas.

Reproduction : `python -m scripts.tests_resumes`

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

### Grossir le corpus n'améliore pas le taux de réussite

La mesure de scaling montre qu'à `min_df=1` le top-1 reste entre 97,9 %
et 99,2 % de 5 à 26 livres, sans tendance. Le paramètre qui gouverne la
difficulté est la longueur d'extrait, pas le nombre de livres.

Il faut donc être clair sur ce qu'on attend d'un corpus plus grand, parce
que ce n'est pas un meilleur score, et que le présenter comme tel serait
malhonnête. Trois bénéfices réels, aucun n'étant le taux de réussite :

**Un benchmark qui discrimine à nouveau.** À 26 livres, Rappel@3, @5 et
@10 valent 1,0 pour `lemmes_12g` et `tokens_12g` : les deux configs sont
indistinguables au-delà du rang 1, et la courbe Rappel@k demandée par le
sujet est plate. Une saturation n'est pas un bon résultat, c'est une
absence de mesure. Trois cents livres remettent des voisins proches dans
le classement et redonnent de l'information à la courbe.

**De la puissance statistique côté résumé.** C'est le manque le plus net :
`p = 0,169` sur la couverture n'est pas concluant à n = 26, et le
départage entre MMR et le top-k naïf reste en suspens. À n ≈ 320, le même
test des signes tranche.

**Une couverture de genres qui ressemble au problème.** Le corpus initial
tient à 26 livres tirés uniformément ; la collecte stratifiée en met une
vingtaine dans chacun des 16 genres. La difficulté d'identification vient
des voisins proches, donc d'un corpus qui contient plusieurs livres du
même registre, pas d'un corpus qui en contient beaucoup au total.

Ce que grossir le corpus ne fera pas : améliorer les 97,9 % à 99,2 %.
Il faut plutôt s'attendre à les voir **baisser**, puisque la tâche
devient plus difficile, et c'est le comportement attendu.

**Vérifié après coup.** Le passage à 291 livres a fait exactement cela :
le top-1 de `lemmes_12g` est passé de 97,88 % à 92,65 %, celui de
`lemmes_1g` de 87,50 % à 70,31 %. Les trois bénéfices annoncés sont au
rendez-vous, la désaturation de la courbe Rappel@k et le creusement de
l'écart entre unigrammes et bigrammes en tête. Le détail est en section 2.

## 6. Coût mémoire des représentations

Cette section était une limite connue : la matrice TF-IDF était dense, ce
qui plafonnait le corpus. Elle est désormais creuse (`csr_matrix`), et
c'est ce qui rend possible le passage à plusieurs centaines de livres.

### Ce que coûtait le dense

Le vocabulaire des configs bigrammes croît presque linéairement avec le
nombre de livres, donc une matrice dense N × V croît en gros comme N².

Vocabulaires **mesurés**, et coût qu'aurait la matrice dense correspondante
(N × V × 8 octets) :

| config | vocab à 26 livres | dense à 26 | vocab à 291 livres | dense à 291 |
|---|---|---|---|---|
| `lemmes_1g` | 38 197 | 7,9 Mo | 143 882 | 335 Mo |
| `lemmes_12g` | 734 223 | 153 Mo | 4 144 734 | **9,7 Go** |
| `tokens_12g` | 837 739 | 174 Mo | 5 048 963 | **11,8 Go** |

Le creux, lui, ne stocke que les cellules non nulles, dont le nombre croît
linéairement avec les livres et non avec le produit N × V : de l'ordre de
130 à 150 Mo à 291 livres pour les configs bigrammes.

Une extrapolation par loi de Heaps ajustée sur 8 à 26 livres annonçait
8,3 M de termes à 300 livres pour `lemmes_12g`, soit 20 Go en dense. La
mesure réelle donne 4,1 M et 9,7 Go : l'extrapolation surestimait d'un
facteur 2, l'exposant ayant été ajusté sur de petits corpus où le
vocabulaire croît plus vite qu'ensuite. La conclusion ne bouge pas pour
autant, 9,7 Go restant hors de portée.

Le remplissage tombe de 5,0 % à 0,4 % pour `lemmes_12g` entre 26 et 300
livres : le dense se paie de plus en plus cher à mesure qu'il sert de
moins en moins.

### Ce que le changement n'a pas changé

Le passage au creux devait être neutre sur les résultats, et il l'est.
Vérifié à trois niveaux :

1. **Matrice et vocabulaire** identiques à l'implémentation dense sur les
   trois configs du corpus réel, à 3 × 10⁻¹⁶ près sur les similarités.
2. **Classements** identiques : rang du vrai livre inchangé sur les 130
   requêtes, et ordre relatif conservé partout où deux scores sont séparés
   par plus que le bruit flottant. Les seules permutations observées
   portent sur des blocs de livres à score nul, que le tri départage
   arbitrairement dans les deux cas.
3. **Benchmark complet** : `evaluer_recherche` rejoué avec les deux
   implémentations sur le même corpus et la même graine rend des tableaux
   `pandas` **strictement égaux** (`DataFrame.equals` vrai).

La comparaison à `sklearn.TfidfVectorizer` reste valide (écart 5,5 × 10⁻¹⁴
sur la config servie), et `tests/test_representations.py` fige ces
propriétés.

Effet secondaire mesuré : le creux est aussi un peu plus rapide,
`lemmes_12g` passant de 8,1 s à 6,2 s pour 130 requêtes × 3 métriques.

### Deux détails d'implémentation qui comptent

`similarites_euclidiennes` **a** été portée, contrairement à ce qui était
prévu ici. L'argument « elle est équivalente au cosinus donc inutile de la
porter » vaut pour le choix de la métrique servie, pas pour le code : le
benchmark l'évalue quand même, et sa formulation d'origine calculait
`X - requete`, une opération dense qui aurait alloué les 20 Go que le
passage au creux vise justement à éviter. Elle passe donc par l'identité
`||x-q||² = ||x||² + ||q||² - 2x·q`, qui ne touche que les cellules non
nulles.

`fit_transform` fait maintenant **deux passes** sur les documents au lieu
d'une. L'ancienne version gardait un `Counter` par document pendant tout
le calcul ; à 300 livres en bigrammes cela ferait environ 21 millions
d'entrées de dictionnaire à clés chaînes vivantes simultanément, soit
plusieurs Go avant même d'allouer la matrice. On reconstruit donc les
n-grammes une seconde fois pour n'avoir jamais qu'un document en mémoire.

### Le reste du chargement

`charger_corpus()` garde par défaut le DataFrame annoté de chaque livre,
soit 3,2 Mo par livre mesurés, auxquels s'ajoutent 3,1 Mo de séquences de
termes : environ 1,9 Go à 300 livres. Or presque personne n'a besoin des
DataFrames en permanence. `charger_corpus(with_df=False)` les relâche en
conservant `n_tokens` et `n_phrases`, et `iter_corpus()` sert les
consommateurs livre par livre (résumés MMR, découpage Word2Vec).

## 7. Limites connues

Les notebooks 05 et 07 ont été exécutés avant ces changements. Leurs
sorties utilisent `min_df=2` et l'ancienne pondération, et le 05 conclut
que `lemmes_1g` est la meilleure config, ce que le benchmark contredit.
Le 05 tire de plus `debut_l` et `debut_t` indépendamment, donc ses
extraits « lemmes » et « tokens » ne portent pas sur les mêmes passages
et sa comparaison entre représentations n'est pas contrôlée. Les rejouer
en appelant `pipeline.benchmark` supprimerait le problème.

La question `min_df` est tranchée, et en faveur de 1 : voir les tests
appariés de la section 2. Le réglage rapporte 10,03 points de top-1 à 291
livres, contre 1,5 à 26. Il reste à décider si ces 10 points valent le
facteur 5,4 sur le vocabulaire, mais ce n'est plus une question de mesure,
c'est un arbitrage entre exactitude et ressources.

Reste ouverte la **bascule du bundle servi vers `tokens_12g`**, que les
mêmes tests recommandent et qui n'a pas encore été faite dans
`build_serving_bundle.py`.

Les seuils de `COLLECTE_PARAMS` sont calibrés sur les 26 livres du corpus
initial pris comme témoin : aucun d'eux ne serait rejeté par les portes.
C'est une garantie contre un réglage trop sévère, pas contre un réglage
trop permissif, qui ne se verra qu'en inspectant ce que la collecte a
effectivement laissé entrer.
