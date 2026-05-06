"""
Representations vectorielles : TF-IDF, similarites, agregation embeddings.

Une seule classe `TfIdfMaison` couvre les deux usages :
- corpus = livres   (notebook `tfidf_similarity`)
- corpus = phrases  (notebook `summary_mmr`, ou un livre = un corpus de phrases)

Conventions sklearn : smoothing IDF, normalisation L2 par document.
Validee contre `sklearn.TfidfVectorizer` dans le notebook 05.
"""
import math
from collections import Counter

import numpy as np
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
        Renvoie une matrice TF-IDF (np.ndarray, shape = (N, V))
        """
        N = len(documents)

        # 1. Comptage brut par document avec n-grammes
        comptes_par_doc = [
            Counter(construire_ngrammes(termes, self.ngram_max))
            for termes in documents
        ]

        # 2. Document frequency par terme
        df_compteur = Counter()
        for c in comptes_par_doc:
            df_compteur.update(c.keys())

        # 3. Filtrage de vocabulaire
        max_df_abs = self.max_df_ratio * N
        vocabulaire = sorted(
            t for t, df in df_compteur.items()
            if self.min_df <= df <= max_df_abs
        )
        self.vocabulaire_ = vocabulaire
        self.index_ = {t: i for i, t in enumerate(vocabulaire)}
        V = len(vocabulaire)

        # 4. IDF (avec lissage)
        idf = np.zeros(V)
        for t in vocabulaire:
            idf[self.index_[t]] = math.log((1 + N) / (1 + df_compteur[t])) + 1.0
        self.idf_ = idf

        # 5. Matrice TF -> TF-IDF
        rows, cols, vals = [], [], []
        for i, comptes in enumerate(comptes_par_doc):
            for terme, n in comptes.items():
                if terme in self.index_:
                    rows.append(i)
                    cols.append(self.index_[terme])
                    vals.append(n)
        tf_sparse = csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=float)
        tfidf = tf_sparse.toarray() * idf[np.newaxis, :]

        # 6. Normalisation L2 par document
        return _l2_normaliser(tfidf)

    def transform(self, documents):
        """Vectorise de nouveaux documents avec le vocabulaire deja appris."""
        if self.vocabulaire_ is None:
            raise RuntimeError("Appelle fit_transform avant transform.")
        N = len(documents)
        V = len(self.vocabulaire_)
        rows, cols, vals = [], [], []
        for i, termes in enumerate(documents):
            comptes = Counter(construire_ngrammes(termes, self.ngram_max))
            for terme, n in comptes.items():
                if terme in self.index_:
                    rows.append(i)
                    cols.append(self.index_[terme])
                    vals.append(n)
        tf_sparse = csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=float)
        tfidf = tf_sparse.toarray() * self.idf_[np.newaxis, :]
        return _l2_normaliser(tfidf)


def _l2_normaliser(matrice):
    """Normalise chaque ligne pour avoir une norme L2 de 1 (lignes nulles preservees)."""
    normes = np.linalg.norm(matrice, axis=1, keepdims=True)
    normes = np.where(normes == 0, 1.0, normes)
    return matrice / normes


# --- Similarites ---

def similarites_cosinus(X, requete):
    """X : (N, V) L2-normalisee. requete : (V,) L2-normalisee."""
    return X @ requete


def similarites_euclidiennes(X, requete):
    """Score de proximite 1 / (1 + distance), donc plus haut = plus proche."""
    distances = np.linalg.norm(X - requete[np.newaxis, :], axis=1)
    return 1.0 / (1.0 + distances)


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
