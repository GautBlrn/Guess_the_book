# Procès-verbal de recette · Projet 2 · voie lexicale

**Objet.** Vérification fonctionnelle et vérification d'accessibilité de l'application
d'identification d'un livre à partir d'un extrait, bras A, plus le contrôle du parcours au
clavier de l'application déployée du bras B.

**Session.** Nuit du 9 au 10 septembre 2026, Europe/Paris.
**Version sous test.** Celle du présent dossier de remise, `Applications/A_Gautier_Blairon/`,
plus les correctifs consignés en section 7, appliqués et vérifiés pendant la session.

Ce document consigne ce qui a été observé. Les seuils annoncés sont mesurés, les points non
vérifiés sont nommés comme tels en section 8, et les écarts trouvés sont conservés même
lorsqu'ils ont été corrigés dans la foulée, parce que la correction fait partie de la preuve.

---

## 1. Environnement et méthode

Chrome sur poste Linux, viewport de 1920 × 905 pixels CSS. Backend Flask sur le port 5000,
interface Streamlit sur le port 8501, tous deux démarrés depuis l'environnement virtuel décrit
par le `README`, modèle spaCy `fr_core_news_sm` installé.

Les parcours au clavier sont exécutés par envoi de séquences de touches au navigateur, avec
relevé de `document.activeElement` après chaque tabulation : la méthode mesure l'ordre de focus
et l'atteignabilité réels du document, pas l'ergonomie ressentie.

Les temps sont relevés de deux façons, pour ne pas confondre le coût du modèle et celui de
l'interface :

- **temps applicatif**, celui que l'application mesure et affiche elle-même (« Calcul effectué
  en N ms »), et pour l'API celui qu'elle renvoie dans son champ `duree_ms` ;
- **temps perçu**, du déclenchement de l'action jusqu'à l'apparition du résultat dans la page.

Les ratios de contraste suivent la formule WCAG 2.1, en composant les couches semi-transparentes
sur leur fond réel avant comparaison. Seuils appliqués : 4,5 pour le texte courant, 3,0 pour le
texte large au sens de la norme. Une précaution s'est révélée nécessaire en cours de session :
un premier relevé du bouton secondaire donnait 3,30 alors que la mesure au repos donne 12,53.
La valeur basse correspondait à l'état de survol, que le premier relevé n'avait pas identifié.
Les états sont donc distingués explicitement dans les tableaux qui suivent.

## 2. Recette fonctionnelle de l'interface

Critère visé : §5.4.2 du sujet, « Fonctionnalité complète : saisie, recherche, affichage des
5 résultats avec score » et « Stabilité et temps de réponse : l'application ne plante pas,
réponse inférieure à 10 secondes ».

| # | Cas | Résultat obtenu | Applicatif | Perçu |
|---|---|---|---|---|
| CF1 | Extrait tiré du corpus, 80 mots, premier appel après démarrage | identification correcte, rang 1, score 0,075 | 1 935 ms | 2 414 ms |
| CF2 | Deuxième tirage | identification correcte, rang 1 | 88 ms | 1 605 ms |
| CF3 | Troisième tirage | **le vrai livre sort au rang 2** (0,057) derrière Ruskin (0,080), et l'application l'annonce d'elle-même | 126 ms | 1 600 ms |
| CF4 | Texte étranger au corpus, 11 mots | « Top 1 » à 0,013, assorti de l'avertissement « Extrait court (11 mots)… » | 76 ms | mesuré au clic |
| CF5 | Champ vidé | résultat retiré, message « Saisis ou tire un extrait, puis clique sur "Identifier le livre". » | sans objet | 598 ms |
| CF6 | Passage modifié après un résultat | résultat retiré, l'application redemande un clic au lieu de relancer une recherche non demandée | sans objet | sans objet |

**Lecture des temps.** Le premier appel porte le chargement des artefacts de service, les
suivants retombent sous 130 ms côté serveur. L'exigence des dix secondes est tenue avec deux
ordres de grandeur de marge à chaud et un facteur quatre à froid. Le classement est affiché sur
dix rangs là où le sujet en demande cinq, chacun avec son rang, son auteur, son titre, son genre
et sa similarité cosinus sous forme de barre accompagnée de sa valeur.

**Le cas CF3 mérite d'être gardé.** Un extrait authentique du corpus n'est pas toujours retrouvé
au rang 1, et l'interface le dit sans le maquiller : elle nomme le livre attendu, donne son rang
et son score. C'est cohérent avec le MRR mesuré sur 1 455 requêtes dans `evaluation.md` : le
taux de réussite au premier rang n'est pas de 100 %, et une démonstration qui ne tomberait
jamais sur ce cas donnerait une image fausse du système.

**Sur la valeur des scores.** Les extraits réellement issus du fonds obtiennent entre 0,036 et
0,126, un texte étranger au corpus 0,013. L'écart de rang est net, l'écart de valeur absolue est
faible : la similarité cosinus d'un TF-IDF à grand vocabulaire n'est pas une probabilité de
bonne réponse, c'est le rang qui porte l'information. Ce point est à énoncer d'entrée, avant
qu'on demande pourquoi une bonne réponse ne « vaut » que 0,04.

## 3. Recette de l'API

L'API a été interrogée directement sur son port, depuis sa propre origine, ce qui teste le
service indépendamment de l'interface.

| Appel | Réponse | Temps |
|---|---|---|
| `GET /health` | 200, `ready: true`, 291 livres et 291 résumés chargés | immédiat |
| `POST /api/identifier` avec `top_k: "cinq"` | **400**, « Le champ 'top_k' doit être un entier. » | 4 ms |
| `POST /api/identifier` avec un corps de 300 Ko | **413**, requête refusée par le plafond de 256 Ko | 6 ms |
| `POST /api/identifier`, extrait valide, `top_k: 5` | 200, cinq résultats, `duree_ms` à 91 | 96 ms côté client |

Les trois garde-fous du service se comportent comme leur code l'annonce : une faute de saisie du
client reçoit un 400 et non un 500, un corps démesuré est refusé avant d'atteindre spaCy, et
aucun détail d'exception n'est renvoyé au client.

## 4. Accessibilité : parcours au clavier

Huit arrêts de tabulation, relevés dans cet ordre : conteneur du contenu principal, zone de
saisie « Extrait », liste de sélection de la source, curseur « Taille (mots) », bouton « Tirer un
extrait », bouton « Identifier le livre », puis les deux premiers boutons de l'en-tête Streamlit,
avant les liens de navigation et le retour au début.

- Tous les éléments interactifs sont atteignables au clavier et aucun piège n'a été rencontré,
  critères 2.1.1 et 2.1.2 tenus.
- L'indicateur de focus est visible partout : le conteneur du champ et les boutons reçoivent une
  bordure colorée nette et l'intitulé du champ change de couleur simultanément, critère 2.4.7
  tenu.
- Deux réserves d'ordre, toutes deux mineures et toutes deux dues au gabarit de la plateforme
  ou à la disposition en colonnes. Le bouton « Identifier le livre » est atteint après le panneau
  de droite alors qu'il se lit avant lui, parce que la colonne de démonstration le précède dans
  le document. Et le contenu principal précède l'en-tête et la navigation latérale, soit
  l'inverse de l'ordre visuel. Le sens reste compréhensible dans les deux cas. Déplacer le bouton
  à l'intérieur de la colonne de saisie corrigerait le premier point.
- Deux boutons de l'en-tête Streamlit n'exposent aucun nom accessible : un lecteur d'écran
  annoncera « bouton » sans plus. Éléments de la plateforme, hors du code de l'application.

## 5. Accessibilité : contrastes mesurés

Le sujet exige des couleurs respectant WCAG AA. État initial, puis état après les correctifs de
la section 7 :

| Élément | Taille | Avant | Après | Seuil |
|---|---|---|---|---|
| Titre principal | 44 px gras | 12,53 | 12,53 | 3,0 |
| Intitulé du champ | 14 px | 12,53 | 12,53 | 4,5 |
| Lien de navigation actif | 16 px | 11,10 | 11,10 | 4,5 |
| Bandeau de résultat | 16 px | 5,51 | 5,51 | 4,5 |
| Texte d'aide sous le titre | 14 px | **3,70** | **6,65** | 4,5 |
| Texte indicatif du champ de saisie | 16 px | **3,54** | **5,93** | 4,5 |
| Bouton secondaire, au repos et au focus | 16 px | 12,53 | 12,53 | 4,5 |
| Bouton secondaire, au survol | 16 px | **3,30** | **6,54** | 4,5 |
| Bouton de soumission | 16 px | sans objet | **6,54** | 4,5 |

Tous les éléments mesurés satisfont désormais le seuil AA, dans les trois états testés.

## 6. Accessibilité : autres constats

- **Langue du document.** L'attribut vaut `en` alors que l'interface est intégralement en
  français, ce qui met en échec le critère 3.1.1. Streamlit écrit cette valeur en dur dans son
  gabarit HTML et ne l'expose à aucune option de configuration. La limite est donc réelle et
  assumée, et elle appartient à la plateforme, pas au code de l'application.
- **Équivalents textuels.** Aucune information n'est portée par une image. Les scores sont des
  barres de progression accompagnées de leur valeur chiffrée, donc lisibles sans la couleur, ce
  qui couvre l'exigence de texte alternatif du sujet.
- **Retour utilisateur.** Un indicateur « Identification en cours… » couvre l'attente, le temps
  de calcul est affiché après coup, les erreurs du service remontent dans un bandeau dédié.

## 7. Écarts trouvés pendant la recette, et ce qui a été fait

| # | Écart | Traitement |
|---|---|---|
| E1 | Aucun bouton de recherche : l'identification partait dès que le champ était non vide, alors que le §5.4.2 liste « Bouton de recherche » et que le docstring du fichier en annonçait déjà un | bouton « Identifier le livre » ajouté. Le texte soumis sert de jeton dans l'état de session, donc modifier le passage après un résultat redemande un clic au lieu de relancer une recherche non demandée |
| E2 | Trois contrastes sous le seuil AA, tous hérités du thème Streamlit par défaut | couleur primaire portée à #B3261E dans `.streamlit/config.toml`, mentions secondaires et texte indicatif ramenés à une couleur pleine par une règle de style |
| E3 | Un texte étranger au corpus recevait un « Top 1 » aussi affirmatif qu'une vraie correspondance | réserve de confiance ajoutée : sous 30 mots ou sous un score de 0,025, le résultat reste affiché mais explicitement requalifié en hypothèse. Le seuil est posé entre les 0,013 mesurés hors corpus et les 0,036 du plus faible vrai positif |
| E4 | Le thème n'était pas copié dans l'image Docker, donc les contrastes corrigés en local repassaient sous le seuil dans le conteneur, sans que rien ne le signale | `COPY .streamlit/` ajouté au `Dockerfile` |
| E5 | `start.sh` livré avec des fins de ligne CRLF : le shebang devient `/bin/bash\r`, et le conteneur construit depuis ce dossier échoue au démarrage avec « no such file or directory » | fichier reconverti en fins de ligne Unix |

## 8. Non vérifié, assumé comme tel

| Point | Raison |
|---|---|
| Lecteur d'écran | coût élevé, hors périmètre de cette session. Reste déclaré non vérifié |
| Agrandissement du navigateur à 200 % sur l'interface de la voie A | non pilotable par l'outillage employé. La mise en page est fluide et sans largeur fixe, mais ce n'est pas une mesure |
| Comportement du service sous charge simultanée | l'API tourne avec un seul processus de travail, choix documenté dans `start.sh`. Aucun test de charge n'a été mené |

## 9. Contrôle du parcours au clavier de l'application déployée du bras B

Vérification menée sur l'application en ligne, en complément de la capture d'agrandissement à
200 % déjà versée au dossier.

Cinq arrêts de tabulation : lien d'évitement « Aller au contenu », groupe de boutons radio du
mode de comparaison, zone de saisie, bouton « Retrouver le livre », bouton « Charger l'exemple ».

- Le lien d'évitement est masqué hors écran et devient visible au focus, sa cible existe,
  critère 2.4.1 tenu.
- Le focus est visible sur les cinq arrêts, critère 2.4.7 tenu.
- Le bouton de recherche s'active à `Entrée`, le chargement de l'exemple à `Espace`, et ce
  dernier rend le focus à la zone de saisie après l'avoir remplie.
- Le choix du mode de comparaison se fait aux flèches, comportement natif et conforme d'un
  groupe de boutons radio.
- Aucun piège au clavier.
- **Une réserve.** Après soumission, la page est reconstruite et le focus retombe sur le corps
  du document, en haut, alors que le résultat s'affiche plus bas. Rien ne l'annonce ni n'y
  conduit, et l'élément portant `role="status"` prévu à cet effet reste vide dans la page de
  résultats. Correctif possible : y écrire le résultat en une phrase et porter le focus sur le
  titre des résultats au retour de la requête.

Mesure de temps relevée au passage, application déjà chaude : 3 465 ms entre l'envoi et la
première ligne de réponse du serveur, pour un extrait de 705 caractères, avec cinq candidats et
leurs scores et l'ouvrage attendu au rang 1 à 0,838. L'exigence des dix secondes est tenue. Le
premier appel après une période creuse reste celui annoncé par le LISEZ-MOI, une trentaine de
secondes le temps que le conteneur se réveille, et il n'a pas été remesuré ici.

## 10. Captures jointes

| Fichier | Ce qu'elle montre | État de l'application |
|---|---|---|
| `preuves_recette/voieA_identification_correcte.jpg` | bras A, extrait de 80 mots tiré du corpus, identification correcte au rang 1 (0,036), classement complet | **avant les correctifs E1 et E2** : thème Streamlit par défaut (curseur #FF4B4B) et pas encore de bouton « Identifier le livre ». Elle prouve l'identification, pas l'état final de l'interface, que décrivent les sections 2, 4 et 5 |
| `preuves_recette/voieB_cinq_candidats.jpg` | bras B déployé, ouvrage identifié et ses voisins aux rangs 2 à 5 | application en ligne, telle que contrôlée en section 9 |
