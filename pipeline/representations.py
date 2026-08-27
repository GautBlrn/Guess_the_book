"""
Representations vectorielles : TF-IDF, similarites, agregation embeddings.

Une seule classe `TfIdfMaison` couvre les deux usages :
- corpus = livres   (notebook `tfidf_similarity`)
- corpus = phrases  (notebook `summary_mmr`, ou un livre = un corpus de phrases)

Conventions sklearn : smoothing IDF, normalisation L2 par document.
Validee contre `sklearn.TfidfVectorizer` dans le notebook 05.

LA MATRICE EST CREUSE (`csr_matrix`), et c'est ce qui rend le corpus
extensible. Elle etait dense jusqu'ici (`toarray()` en fin de
`fit_transform`), ce qui plafonnait le corpus a quelques dizaines de
livres : le vocabulaire des configs bigrammes croit presque lineairement
avec le nombre de livres (loi de Heaps ajustee, b = 0.97), donc une
matrice dense N x V croit en gros comme N^2.

Mesure sur le corpus reel, extrapolation Heaps a 300 livres :

    config        26 livres          300 livres (extrapole)
                  dense    sparse    dense     sparse
    lemmes_1g     7,9 Mo   1,6 Mo    0,4 Go     18 Mo
    lemmes_12g    153 Mo    12 Mo     20 Go    132 Mo
    tokens_12g    174 Mo    13 Mo     23 Go    147 Mo

Le remplissage mesure baisse quand le corpus grandit, parce que le nombre
de cellules non nulles croit lineairement (chaque livre apporte les siennes)
alors que N x V croit en N^1.97 :

    config        26 livres    300 livres (extrapole)
    lemmes_1g      13,3 %          2,8 %
    lemmes_12g      5,0 %          0,4 %
    tokens_12g      4,9 %          0,4 %

Autrement dit le dense se paie de plus en plus cher a mesure qu'il sert de
moins en moins.

Deux consequences pour qui touche a ce module :

1. `fit_transform` et `transform` renvoient une `csr_matrix`. L'indexation
   `matrice[i]` donne une ligne creuse (1, V) et non un vecteur (V,).
   Les fonctions de similarite ci-dessous acceptent les deux formes.
2. Le seul appelant qui veut du dense est `summarization.vectoriser_phrases_tfidf`,
   ou le corpus est les phrases d'UN livre (quelques milliers de lignes,
   matrice de quelques Mo) et ou tout le calcul MMR en aval est ecrit en
   dense. Il densifie explicitement, chez lui.

Reproduire la mesure : voir `docs/evaluation.md`, section « Coût mémoire
des représentations ».
"""
import math
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


def construire_ngrammes(termes, n_max):
    """Genere les n-grammes 1..n_max d'une sequence de termes."""
    if n_max <= 1:
        return list(termes)
    sortie = list(termes)
    for n in range(2, n_max + 1):
        for i in range(len(termes) - n + 1):
            sortie.append(" ".join(termes[i:i + n]))
    return sortie


class TfIdfMaison:
    """
    TF-IDF implementation maison, conventions sklearn :
      idf(t) = log((1 + N) / (1 + df(t))) + 1
      tf(t, d) = comptage brut
      normalisation L2 par document

    Couvre n-grammes (1..ngram_max), filtrage min_df / max_df_ratio.
    """

    def __init__(self, ngram_max=1, min_df=1, max_df_ratio=1.0):
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocabulaire_ = None
        self.index_ = None
        self.idf_ = None

    def fit_transform(self, documents):
        """
        documents : list[list[str]]   (chaque document = sequence de termes)
        Renvoie une matrice TF-IDF creuse (csr_matrix, shape = (N, V)).

        DEUX PASSES SUR LES DOCUMENTS, et pas une seule. La version
        precedente gardait un `Counter` par document pendant tout le calcul
        (`comptes_par_doc`), ce qui tient a 26 livres mais pas a 300 : en
        `lemmes_12g` un livre porte ~70 k n-grammes distincts, donc 300
        livres font ~21 M d'entrees de dict a cles chaines vivantes en meme
        temps, plusieurs Go avant meme d'allouer la matrice.

        On paie donc la construction des n-grammes deux fois (une fois pour
        le DF, une fois pour les lignes) pour ne jamais tenir qu'un document
        a la fois. Le surcout CPU est reel mais modeste : ~0,7 s a 26 livres,
        de l'ordre de 15 s a 300, contre un pic memoire qui rendait le calcul
        simplement impossible.
        """
        N = len(documents)

        # 1. Document frequency par terme (passe 1)
        df_compteur = Counter()
        for termes in documents:
            df_compteur.update(set(construire_ngrammes(termes, self.ngram_max)))

        # 2. Filtrage de vocabulaire
        max_df_abs = self.max_df_ratio * N
        vocabulaire = sorted(
            t for t, df in df_compteur.items()
            if self.min_df <= df <= max_df_abs
        )
        self.vocabulaire_ = vocabulaire
        self.index_ = {t: i for i, t in enumerate(vocabulaire)}
        V = len(vocabulaire)

        # 3. IDF (avec lissage)
        idf = np.zeros(V)
        for t in vocabulaire:
            idf[self.index_[t]] = math.log((1 + N) / (1 + df_compteur[t])) + 1.0
        self.idf_ = idf

        # On libere le compteur de DF avant d'allouer la matrice : sur les
        # configs bigrammes il pese plus lourd que la matrice elle-meme.
        del df_compteur

        # 4. Matrice TF -> TF-IDF -> L2 (passe 2)
        return self._vectoriser(documents, N, V, idf)

    def transform(self, documents):
        """Vectorise de nouveaux documents avec le vocabulaire deja appris."""
        if self.vocabulaire_ is None:
            raise RuntimeError("Appelle fit_transform avant transform.")
        return self._vectoriser(
            documents, len(documents), len(self.vocabulaire_), self.idf_,
        )

    def _vectoriser(self, documents, N, V, idf):
        """
        Construit la csr_matrix TF-IDF L2-normalisee, document par document.

        On remplit directement les trois tableaux CSR (`indptr`, `indices`,
        `data`) plutot que de passer par un COO (rows, cols, vals) : les
        documents arrivent dans l'ordre des lignes, ce qui EST la structure
        CSR. Ca evite le tri et la fusion de doublons que ferait le
        constructeur COO, et surtout les trois listes Python intermediaires
        de longueur nnz (~11 M d'elements a 300 livres en bigrammes).
        """
        indptr = np.zeros(N + 1, dtype=np.int64)
        indices_par_doc = []
        data_par_doc = []

        for i, termes in enumerate(documents):
            comptes = Counter(construire_ngrammes(termes, self.ngram_max))
            cols, vals = [], []
            for terme, n in comptes.items():
                j = self.index_.get(terme)
                if j is not None:
                    cols.append(j)
                    vals.append(n)
            # CSR ne l'exige pas, mais des indices tries par ligne rendent
            # les produits matriciels de scipy sensiblement plus rapides.
            ordre = np.argsort(cols, kind="stable")
            cols = np.asarray(cols, dtype=np.int32)[ordre]
            vals = np.asarray(vals, dtype=np.float64)[ordre]
            indices_par_doc.append(cols)
            data_par_doc.append(vals)
            indptr[i + 1] = indptr[i] + len(cols)

        indices = (np.concatenate(indices_par_doc) if indices_par_doc
                   else np.empty(0, dtype=np.int32))
        data = (np.concatenate(data_par_doc) if data_par_doc
                else np.empty(0, dtype=np.float64))

        X = csr_matrix((data, indices, indptr), shape=(N, V))

        # TF -> TF-IDF : mise a l'echelle par colonne, faite sur `data` en
        # place. `X.multiply(idf)` donnerait le meme resultat mais repasse
        # par un COO et double la memoire le temps du calcul.
        X.data *= idf[X.indices]

        return _l2_normaliser(X)


def _l2_normaliser(matrice):
    """
    Normalise chaque ligne pour avoir une norme L2 de 1.

    Les lignes nulles sont preservees telles quelles (norme 0, pas de
    division) : c'est le cas d'un extrait dont aucun terme n'est dans le
    vocabulaire appris, et le comportement attendu en aval est un score de
    similarite nul avec tout le corpus, pas un NaN.

    Accepte creux comme dense. Le chemin creux modifie `data` en place et
    renvoie la meme matrice, donc n'appeler que sur une matrice fraiche.
    """
    if sp.issparse(matrice):
        carres = np.asarray(matrice.multiply(matrice).sum(axis=1)).ravel()
        normes = np.sqrt(carres)
        normes[normes == 0] = 1.0
        # Indice de ligne de chaque cellule non nulle, deduit de indptr.
        lignes = np.repeat(np.arange(matrice.shape[0]), np.diff(matrice.indptr))
        matrice.data /= normes[lignes]
        return matrice

    normes = np.linalg.norm(matrice, axis=1, keepdims=True)
    normes = np.where(normes == 0, 1.0, normes)
    return matrice / normes


# --- Similarites ---

def _colonne(requete):
    """
    Ramene une requete a une colonne (V, 1), en gardant sa nature.

    Les appelants passent selon les cas un vecteur dense (V,), une ligne
    creuse (1, V) telle que la renvoie `matrice[i]` sur une csr, ou deja une
    colonne. On accepte les trois plutot que d'imposer une forme : cote
    `api.py` comme cote `benchmark.py`, la requete sort d'un `transform` et
    est donc creuse, et la densifier couterait 8 x V octets par appel (64 Mo
    a 300 livres en bigrammes) pour un vecteur qui porte quelques centaines
    de valeurs non nulles.
    """
    if sp.issparse(requete):
        if requete.ndim == 2 and requete.shape[0] == 1:
            return requete.T.tocsr()
        return requete.tocsr()
    return np.asarray(requete).reshape(-1, 1)


def similarites_cosinus(X, requete):
    """
    X : (N, V) L2-normalisee, creuse ou dense. requete : L2-normalisee.

    Renvoie un tableau 1-D de N scores. Sur des vecteurs L2-normalises le
    cosinus est le simple produit scalaire.
    """
    scores = X @ _colonne(requete)
    if sp.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores).ravel()


def similarites_euclidiennes(X, requete):
    """
    Score de proximite 1 / (1 + distance), donc plus haut = plus proche.

    Calcule par l'identite ||x - q||^2 = ||x||^2 + ||q||^2 - 2 x.q plutot
    que par la difference explicite. Meme resultat, mais `X - requete` est
    une operation DENSE : elle remplit les V colonnes de chacune des N
    lignes, soit 20 Go a 300 livres en `lemmes_12g` pour un calcul dont le
    resultat tient en N flottants. L'identite ne touche que les cellules non
    nulles.

    On ne suppose pas ||x|| = 1 malgre la normalisation en amont : une ligne
    vide (aucun terme du document dans le vocabulaire) a une norme nulle et
    doit rester traitee comme telle.
    """
    produit = similarites_cosinus(X, requete)

    if sp.issparse(X):
        normes_x2 = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    else:
        normes_x2 = np.einsum("ij,ij->i", X, X)

    q = _colonne(requete)
    if sp.issparse(q):
        norme_q2 = float(q.multiply(q).sum())
    else:
        plat = np.asarray(q).ravel()
        norme_q2 = float(plat @ plat)

    # Le clip a 0 absorbe les negatifs de l'ordre de 1e-16 que l'identite
    # produit quand x et q sont quasi identiques.
    d2 = np.maximum(normes_x2 + norme_q2 - 2.0 * produit, 0.0)
    return 1.0 / (1.0 + np.sqrt(d2))


def similarites_jaccard(ensembles_corpus, ensemble_requete):
    """J(corpus_i, requete) pour chaque livre i du corpus."""
    sims = np.zeros(len(ensembles_corpus))
    for i, ensemble in enumerate(ensembles_corpus):
        if not ensemble and not ensemble_requete:
            continue
        inter = len(ensemble & ensemble_requete)
        union = len(ensemble | ensemble_requete)
        sims[i] = inter / union if union else 0.0
    return sims


# --- Agregation pour embeddings ---

def calculer_idf(documents):
    """IDF lisse style sklearn, retourne un dict {terme: idf}."""
    N = len(documents)
    df = Counter()
    for termes in documents:
        df.update(set(termes))
    return {t: math.log((1 + N) / (1 + d)) + 1.0 for t, d in df.items()}


def vecteur_livre_embeddings(termes, vecteurs, idf=None):
    """
    Construit le vecteur agrege d'un livre dans un espace d'embeddings.

    - Si idf=None : moyenne simple des vecteurs des mots presents.
    - Sinon       : moyenne ponderee par TF-IDF.

    Les mots hors vocabulaire des `vecteurs` sont ignores. Renvoie un
    vecteur nul si aucun mot du livre n'est dans le vocabulaire.
    """
    if idf is None:
        vecs = [vecteurs[t] for t in termes if t in vecteurs]
        if not vecs:
            return np.zeros(vecteurs.vector_size)
        return np.mean(vecs, axis=0)

    tf = Counter(termes)
    n_termes = len(termes)
    somme_vec = np.zeros(vecteurs.vector_size)
    somme_poids = 0.0
    for t, n in tf.items():
        if t in vecteurs and t in idf:
            poids = (n / n_termes) * idf[t]
            somme_vec += poids * vecteurs[t]
            somme_poids += poids
    if somme_poids == 0:
        return np.zeros(vecteurs.vector_size)
    return somme_vec / somme_poids


def cosinus(u, v):
    """Cosinus entre deux vecteurs (gere les vecteurs nuls)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(u @ v / (nu * nv))
