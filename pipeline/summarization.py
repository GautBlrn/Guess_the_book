"""
Résumé extractif par MMR (Maximal Marginal Relevance) -- v4.

Refonte motivée par le diagnostic du résumé "Au Bonheur Des Dames" :
les phrases selectionnées étaient toutes des paragraphes descriptifs longs,
agglutinées loin du centre du livre dans la projection PCA.

Trois problèmes identifiés en v3 :

1. La similarité au centre via embeddings sentence-camembert correlé
   fortement avec la longueur (Spearman +0.51 sur Au Bonheur Des Dames).
   Les phrases longues sont mécaniquement plus proches du centre de
   gravité -- la composante "centre" agissait comme un bonus longueur
   deguisé.

2. TextRank avec seuil 0.3 fixe sur des cosinus camembert garde un
   graphe trop dense (44 % d'aretes), donc PageRank converge vers une
   distribution quasi-uniforme et n'est plus discriminant (scores tous
   a 0.001).

3. Centre + TextRank sur camembert selectionnent les mêmes phrases
   (Jaccard 0.82). Le score hybride pondère n'apporte aucune diversité.

Refonte v4 :

- **Pertinence** = similarité TF-IDF (1-2 grammes de lemmes) avec le
  centre. La creusité naturelle du TF-IDF capture la specificité
  lexicale (noms propres, vocabulaire de l'intrigue) qui manquait
  cruellement à camembert sur du Zola.

- **Diversite MMR** = embeddings camembert. Camembert reste utile
  pour detecter les paraphrases : deux phrases qui disent la meme
  chose avec des mots differents seront proches dans l'espace
  camembert mais loin dans l'espace TF-IDF.

- **TextRank** sur camembert avec seuil dynamique (quantile 90 de la
  distribution des cosinus du livre). Sans seuil dynamique, le seuil
  optimal varie d'un livre a l'autre selon sa coherence thematique.

- **Bonus longueur** garde un poids residuel (0.05) : la pertinence
  TF-IDF ne souffre plus du biais longueur, donc on n'a plus besoin
  de compenser, mais on garde un petit bonus pour eviter les phrases
  trop courtes qui auraient un score TF-IDF haut par accident.

Chargement lazy du modele camembert via `_get_embedder()`. L'import
du module ne declenche AUCUN telechargement.
"""
from functools import lru_cache

import numpy as np

from pipeline.config import EMB_PARAMS, MMR_PARAMS
from pipeline.representations import TfIdfMaison


# Ponctuation francaise pour reconstruction de phrases lisibles
PUNCT_COLLE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "..."}
OUVRANTS = {"(", "[", "{"}


# ============================================================================
# Modele d'embeddings (lazy load)
# ============================================================================

@lru_cache(maxsize=2)
def _get_embedder(modele=None):
    """
    Charge le modele sentence-transformers en lazy.

    Premier appel : telecharge ~440 Mo (base) ou ~1.3 Go (large).
    Mis en cache par nom de modele : on peut basculer entre base et
    large sans recharger le base.
    """
    if modele is None:
        modele = EMB_PARAMS["modele"]
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(modele)


def _encoder_phrases(textes, modele=None, batch_size=32):
    """Encode une liste de textes en embeddings L2-normalises."""
    embedder = _get_embedder(modele)
    return embedder.encode(
        textes,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )


# ============================================================================
# Reconstruction et extraction
# ============================================================================

def reconstruire_phrase(tokens):
    """Recolle des tokens en respectant la ponctuation francaise."""
    morceaux = []
    for t in tokens:
        if not morceaux:
            morceaux.append(t)
            continue
        if t in PUNCT_COLLE:
            morceaux.append(t)
        elif morceaux[-1].endswith("'") or morceaux[-1] in OUVRANTS:
            morceaux.append(t)
        else:
            morceaux.append(" " + t)
    return "".join(morceaux)


def extraire_phrases(
    df,
    champ_termes="lemma",
    min_tokens=None,
    max_tokens=None,
    min_densite=None,
    max_propn_ratio=None,
):
    """
    DataFrame annote -> liste de phrases candidates filtrees.

    Filtres :
    - longueur informative dans [min_tokens, max_tokens]
    - densite lexicale (informatifs / total) >= min_densite
    - ratio de noms propres (PROPN / informatifs) <= max_propn_ratio

    Retour : liste de dicts {sent_id, texte, termes, n_tokens_total,
    n_informatifs, densite, propn_ratio}.
    """
    if min_tokens is None:       min_tokens = MMR_PARAMS["min_tokens_phrase"]
    if max_tokens is None:       max_tokens = MMR_PARAMS["max_tokens_phrase"]
    if min_densite is None:      min_densite = MMR_PARAMS["min_densite"]
    if max_propn_ratio is None:  max_propn_ratio = MMR_PARAMS["max_propn_ratio"]

    masque_informatif = df["is_alpha"] & ~df["is_stop"] & ~df["is_punct"]

    phrases = []
    for sent_id, groupe in df.groupby("sent_id", sort=True):
        n_total = len(groupe)
        if n_total < min_tokens:
            continue

        sub = groupe[masque_informatif.loc[groupe.index]]
        n_informatifs = len(sub)
        if n_informatifs < min_tokens:
            continue
        if n_informatifs > max_tokens:
            continue

        densite = n_informatifs / n_total
        if densite < min_densite:
            continue

        propn_ratio = (sub["pos"] == "PROPN").sum() / n_informatifs
        if propn_ratio > max_propn_ratio:
            continue

        texte = reconstruire_phrase(groupe["text"].tolist())
        termes = sub[champ_termes].str.lower().tolist()

        phrases.append({
            "sent_id":         int(sent_id),
            "texte":           texte,
            "termes":          termes,
            "n_tokens_total":  int(n_total),
            "n_informatifs":   int(n_informatifs),
            "densite":         float(densite),
            "propn_ratio":     float(propn_ratio),
        })
    return phrases


# ============================================================================
# Representations vectorielles des phrases
# ============================================================================

def vectoriser_phrases_tfidf(phrases, ngram_max=2, min_df=2):
    """
    Vectorise les phrases en TF-IDF lemmes (1-2 grammes par defaut).

    Capture la specificite lexicale du livre : noms propres recurrents,
    vocabulaire metier (commerce, magasin), expressions repetees. C'est
    cette specificite qui manquait a camembert pour identifier les
    phrases narrativement importantes.

    Renvoie une matrice (N, V) L2-normalisee, donc cosinus = produit
    scalaire.
    """
    documents = [p["termes"] for p in phrases]
    vec = TfIdfMaison(ngram_max=ngram_max, min_df=min_df, max_df_ratio=1.0)
    return vec.fit_transform(documents)


# ============================================================================
# Composantes du score hybride
# ============================================================================

def similarite_au_centre(matrice):
    """
    Cosinus de chaque phrase avec le centre de gravite.

    Mesure : à quel point cette phrase représente le "thème moyen"
    du livre. Les vecteurs etant L2-normalisés, le centre est juste
    leur moyenne re-normalisee.

    En v4, cette fonction est appelee sur la matrice TF-IDF (et non
    plus sur les embeddings camembert) : la creusite du TF-IDF limite
    le biais longueur et fait ressortir les phrases qui contiennent
    le vocabulaire specifique du livre.
    """
    centre = matrice.mean(axis=0)
    n = np.linalg.norm(centre)
    if n > 0:
        centre = centre / n
    return matrice @ centre


def centralite_textrank(emb_phrases, n_iter=30, damping=0.85,
                         seuil=None, quantile_seuil=0.90):
    """
    Score TextRank (PageRank sur le graphe de similarite des phrases).

    Une phrase est centrale si elle est semantiquement proche d'autres
    phrases qui sont elles-memes centrales.

    `seuil` : si fourni, valeur fixe a partir de laquelle on garde une
    arete. Si None (defaut), on calcule dynamiquement le seuil comme le
    `quantile_seuil` de la distribution des cosinus du livre.

    Le seuil fixe a 0.3 de la v3 produisait un graphe trop dense sur
    camembert (44 % d'aretes restantes) et PageRank convergeait vers
    une distribution quasi-uniforme. Avec un seuil au quantile 90, on
    garde environ 10 % des aretes -- les plus fortes -- et le rank
    redevient discriminant.

    NOTE : cette fonction tourne sur les embeddings camembert. Le
    TextRank capture la coherence semantique (paraphrases proches),
    contrairement a la similarite au centre TF-IDF qui capture la
    coherence lexicale (vocabulaire specifique).
    """
    sim = emb_phrases @ emb_phrases.T
    np.fill_diagonal(sim, 0.0)

    if seuil is None:
        # Quantile sur la triangle superieure stricte
        iu = np.triu_indices_from(sim, k=1)
        seuil = float(np.quantile(sim[iu], quantile_seuil))

    sim = np.where(sim >= seuil, sim, 0.0)

    sommes = sim.sum(axis=1, keepdims=True)
    sommes = np.where(sommes == 0, 1.0, sommes)
    M = sim / sommes

    N = emb_phrases.shape[0]
    rank = np.full(N, 1.0 / N)
    for _ in range(n_iter):
        rank = (1 - damping) / N + damping * (M.T @ rank)
    return rank


def bonus_longueur(phrases, fenetre=(15, 40)):
    """
    Bonus en triangle pour les phrases de taille naturelle.

    En v4, le poids de cette composante est tres faible (0.05) puisque
    le biais longueur de la pertinence a ete corrige (TF-IDF au lieu
    de camembert). Le bonus sert juste a eviter qu'une phrase tres
    courte se retrouve en tete par accident sur un score TF-IDF eleve.
    """
    a, b = fenetre
    longueurs = np.array([p["n_informatifs"] for p in phrases])
    return np.where(
        (longueurs >= a) & (longueurs <= b),
        1.0,
        np.maximum(0.0, 1.0 - np.where(
            longueurs < a,
            (a - longueurs) / a,
            (longueurs - b) / b,
        )),
    )


def _normaliser(x):
    """Min-max sur [0, 1]. Renvoie 0.5 si x est constant."""
    rng = x.max() - x.min()
    if rng < 1e-9:
        return np.full_like(x, 0.5, dtype=float)
    return (x - x.min()) / rng


def score_hybride(phrases, matrice_tfidf, emb_camembert, poids=None):
    """
    Combinaison ponderee normalisee des composantes de pertinence.

    Composantes (v4) :
    - centre   : similarite TF-IDF avec le centre du livre
                 (specificite lexicale, capte les noms propres et le
                 vocabulaire de l'intrigue)
    - textrank : centralite du graphe de similarite camembert
                 (coherence thematique, lien semantique entre phrases)
    - longueur : bonus residuel pour les phrases de taille naturelle

    Pas de score de position : sur un roman, l'incipit et le denouement
    sont souvent autant signifiants que le milieu.

    `poids` : dict avec cles `centre`, `textrank`, `longueur`. Defaut
    depuis EMB_PARAMS.

    La somme des poids n'a pas a valoir 1, et le defaut livre vaut
    d'ailleurs 1.05 (0.70 + 0.30 + 0.05). Seul le score RELATIF entre
    phrases est utilise, par MMR puis par le tri : multiplier tous les
    poids par une constante ne change aucun classement. Ce sont les
    rapports entre composantes qui comptent.
    """
    if poids is None:
        poids = EMB_PARAMS["poids_score"]

    s_centre = _normaliser(similarite_au_centre(matrice_tfidf))
    s_text   = _normaliser(centralite_textrank(emb_camembert))
    s_long   = _normaliser(bonus_longueur(phrases))

    return (poids["centre"]   * s_centre
          + poids["textrank"] * s_text
          + poids["longueur"] * s_long)


# ============================================================================
# MMR sur embeddings camembert (diversite semantique)
# ============================================================================

def mmr(embeddings, pertinences, k, lambda_):
    """
    Selection gloutonne MMR.

    embeddings  : (N, d) embeddings camembert L2-normalises
    pertinences : (N,)   scores de pertinence pre-calcules (TF-IDF + TextRank)
    k           : nombre de phrases a selectionner
    lambda_     : compromis pertinence / diversite (1 = pertinence pure)

    On utilise camembert pour la diversite parce qu'il detecte les
    paraphrases : deux phrases qui disent la meme chose avec des mots
    differents seront proches en camembert. TF-IDF aurait juste detecte
    les phrases qui partagent du vocabulaire, pas les paraphrases.
    """
    N = embeddings.shape[0]
    selectionnes = []
    candidats = list(range(N))
    max_sim_a_S = np.full(N, -np.inf)

    while len(selectionnes) < k and candidats:
        if not selectionnes:
            scores = pertinences[candidats]
        else:
            scores = (
                lambda_ * pertinences[candidats]
                - (1 - lambda_) * max_sim_a_S[candidats]
            )
        meilleur = candidats[int(np.argmax(scores))]
        selectionnes.append(meilleur)
        candidats.remove(meilleur)
        sims_au_dernier = embeddings @ embeddings[meilleur]
        max_sim_a_S = np.maximum(max_sim_a_S, sims_au_dernier)

    return selectionnes


# ============================================================================
# API publique
# ============================================================================

def resumer_livre(df, champ_termes="lemma", k=None, lambda_=None,
                  min_tokens=None, max_tokens=None, min_df=None,
                  min_densite=None, max_propn_ratio=None,
                  ordre_narratif=False,
                  embeddings_pre=None, poids=None):
    """
    Bout-en-bout : DataFrame annote -> liste de phrases selectionnees.

    Pipeline v4 :
    1. Extraction des phrases candidates (filtres longueur/densite/propn).
    2. Vectorisation TF-IDF (1-2 grammes) -> pertinence "centre".
    3. Embeddings camembert -> TextRank et diversite MMR.
    4. Score hybride : 0.5 centre TF-IDF + 0.4 TextRank + 0.1 longueur
       (poids ajustables via EMB_PARAMS["poids_score"]).
    5. Selection MMR sur les embeddings camembert.

    Parametres
    ----------
    embeddings_pre : ndarray, optionnel
        Embeddings camembert deja calcules pour les phrases candidates.
        Permet de reutiliser le calcul cote app/script entre plusieurs
        appels (ex. quand l'utilisateur change lambda).
    poids : dict, optionnel
        Ponderation du score hybride (cles `centre`, `textrank`,
        `longueur`). Defaut : `EMB_PARAMS["poids_score"]`.

        Existe pour que `pipeline/benchmark.py` puisse balayer la
        ponderation sans monkeypatcher la config : le benchmark doit
        mesurer cette fonction-ci, pas une reimplementation locale. La
        mesure appariee sur 26 livres montre que le poids TextRank de
        0.30 rend MMR plus redondant que le simple top-k TF-IDF
        (7 livres sur 26, test des signes p = 0.029), donc ce parametre
        n'est pas theorique : c'est le levier a explorer.

    Renvoie : liste de dicts avec les memes cles que `extraire_phrases`,
    enrichies de `score` (score hybride final) et `rang_mmr` (rang de
    selection, 1 = premier choisi).

    `ordre_narratif=True` : reordonne par sent_id (utile pour BARThez
    en aval, et plus naturel a lire).
    """
    if k is None:           k = MMR_PARAMS["k_phrases"]
    if lambda_ is None:     lambda_ = MMR_PARAMS["lambda_default"]
    if min_df is None:      min_df = MMR_PARAMS.get("min_df", 2)

    phrases = extraire_phrases(
        df, champ_termes=champ_termes,
        min_tokens=min_tokens, max_tokens=max_tokens,
        min_densite=min_densite, max_propn_ratio=max_propn_ratio,
    )
    if len(phrases) < k:
        for i, p in enumerate(phrases, 1):
            p["rang_mmr"] = i
            p["score"] = 1.0
        return phrases

    # 1. TF-IDF des phrases (pertinence)
    matrice_tfidf = vectoriser_phrases_tfidf(
        phrases,
        ngram_max=MMR_PARAMS.get("ngram_max", 2),
        min_df=min_df,
    )

    # 2. Embeddings camembert (TextRank + diversite MMR)
    if embeddings_pre is not None:
        emb_cam = embeddings_pre
    else:
        emb_cam = _encoder_phrases([p["texte"] for p in phrases])

    # 3. Score hybride
    scores = score_hybride(phrases, matrice_tfidf, emb_cam, poids=poids)

    # 4. MMR sur camembert
    indices = mmr(emb_cam, scores, k=k, lambda_=lambda_)

    selection = []
    for rang, i in enumerate(indices, 1):
        p = dict(phrases[i])
        p["rang_mmr"] = rang
        p["score"] = float(scores[i])
        selection.append(p)

    if ordre_narratif:
        selection.sort(key=lambda x: x["sent_id"])

    return selection


def formatter_resume(phrases, livre, auteur, genre, k, lambda_):
    """Construit la chaine textuelle finale du resume."""
    en_tete = (
        f"{livre}\n"
        f"{auteur}  ({genre})\n"
        f"Resume extractif - {k} phrases - MMR lambda={lambda_}\n"
        + "=" * 70 + "\n\n"
    )
    corps = "\n\n".join(
        f"[{i + 1:2d}] {p['texte']}" for i, p in enumerate(phrases)
    )
    return en_tete + corps + "\n"