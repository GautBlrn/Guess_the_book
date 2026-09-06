# Luigi, pour information

Écrit le 7 septembre 2026 par Gautier. Ce dépôt est le bras A du projet 2, tu n'y
travailles pas. Un seul commit, et il te concerne quand même à deux titres.

## Ce qui a changé

`scripts/rebuild_artifacts.py` et `build_serving_bundle.py` ne se lançaient plus
depuis la remise assemblée. Le premier insérait `Path(__file__).parent.parent`
dans `sys.path`, le second son propre répertoire : tous deux visaient la racine
du dépôt depuis leur emplacement de travail. Copiés sous
`Modèles/A_Gautier_Blairon/code_entrainement/`, ils pointaient un dossier sans
paquet `pipeline/`, et `from pipeline import storage` levait
`ModuleNotFoundError`.

La racine est trouvée par un repère, `pipeline/config.py`. Les deux sortent
maintenant avec le code 0 depuis l'archive décompressée, ce qui rend vraie la
phrase du LISEZ-MOI qui annonce le bundle de 269,1 Mo comme régénérable.

## Le point qui te concerne pour le rapport 2

`pipeline/abstractive.py` implémente un résumé abstractif avec **BARThez**
(`moussaKam/barthez-orangesum-abstract`). Il n'apparaissait ni dans le rapport
consolidé, ni dans le support. Le rapport écrivait « l'implémentation B ajoute
une méthode générative », ce qui disait au jury que le bras A n'avait livré
aucune IA générative sur le projet évalué là-dessus, alors que le code existe.

Il est maintenant décrit en section 7.4 du rapport 2 : le modèle, le motif du
choix, la stratégie de générer à partir des phrases déjà sélectionnées par MMR
plutôt que du livre entier, et la raison écrite de son retrait de l'application,
660 Mo et 30 à 60 s par livre contre une contrainte de latence. Un arbitrage
produit documenté vaut mieux qu'un module silencieux.

L'absence de mesure sur BARThez est déclarée comme limite en section 10.2 :
aucune comparaison n'est tentée entre ta brique T5 et celle-ci, elles ne sont pas
au même stade.

## Où lire la suite

Le détail est dans `JOURNAL_CORRECTION_BC02.md`, à la racine du dépôt `livrables-bc02` sur GitHub. Les notes qui te concernent
vraiment sont les `POUR_LUIGI.md` d'`epi_detection`, de `le-limier` et de
`livrables`.
