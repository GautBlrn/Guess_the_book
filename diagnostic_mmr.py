"""
Diagnostic du resume MMR : pourquoi les phrases selectionnees sont
agglutinees loin du centre du livre, et pourquoi le resume est domine
par des paragraphes descriptifs longs.

Mesures effectuees (camembert ET TF-IDF maison en parallele) :

1. Distribution des cosinus inter-phrases : quel pourcentage de paires
   passe le seuil 0.3 du TextRank ? Si c'est >90 %, le filtre est inutile
   et le PageRank converge vers une distribution quasi-uniforme.

2. Correlation longueur de phrase / similarite au centre. Si la
   correlation de Spearman est > 0.5, la composante "centre" du score
   hybride favorise mecaniquement les phrases longues.

3. Top 10 phrases par chacune des composantes isolees du score :
   - centre seul
   - textrank seul
   - longueur seule
   On compare la composition (longueur moyenne, dialogue vs description)
   pour identifier la composante coupable.

4. Top 10 phrases si on remplace les embeddings camembert par TF-IDF
   maison dans tout le pipeline. Permet de voir si le probleme vient
   de la representation ou du score.

Sortie : console + un PNG dans le dossier courant.

Usage :
    python diagnostic_mmr.py "Au Bonheur"

L'argument est un motif passe a `trouver_livre()`.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from pipeline.corpus import charger_corpus, trouver_livre
from pipeline.summarization import (
    _encoder_phrases,
    bonus_longueur,
    centralite_textrank,
    extraire_phrases,
    similarite_au_centre,
)
from pipeline.representations import TfIdfMaison


# Largeurs typographiques pour la console
LARGEUR = 90
SEP = "=" * LARGEUR
SOUS_SEP = "-" * LARGEUR


def titre(texte):
    print(f"\n{SEP}\n{texte}\n{SEP}")


def sous_titre(texte):
    print(f"\n{SOUS_SEP}\n{texte}\n{SOUS_SEP}")


def tronquer(texte, n=140):
    """Tronque proprement pour l'affichage console."""
    texte = " ".join(texte.split())
    return texte if len(texte) <= n else texte[: n - 3] + "..."


# ============================================================================
# Construction des representations
# ============================================================================

def vectoriser_phrases_tfidf(phrases, ngram_max=1, min_df=2):
    """
    Vectorise les phrases avec TfIdfMaison. Chaque phrase = un document.
    Renvoie une matrice (N, V) L2-normalisee, donc cosinus = produit scalaire.
    """
    documents = [p["termes"] for p in phrases]
    tfidf = TfIdfMaison(ngram_max=ngram_max, min_df=min_df, max_df_ratio=1.0)
    return tfidf.fit_transform(documents)


# ============================================================================
# Mesures de diagnostic
# ============================================================================

def distribution_cosinus(emb, nom):
    """Affiche les quantiles de la distribution des cosinus inter-phrases."""
    sim = emb @ emb.T
    # Seulement la triangle superieure stricte (paires distinctes)
    iu = np.triu_indices_from(sim, k=1)
    valeurs = sim[iu]

    print(f"\n[{nom}] Distribution des cosinus inter-phrases ({len(valeurs):,} paires):")
    for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
        print(f"  quantile {int(q*100):2d}% : {np.quantile(valeurs, q):.3f}")
    print(f"  moyenne     : {valeurs.mean():.3f}")
    print(f"  ecart-type  : {valeurs.std():.3f}")

    # Pourcentage qui passe le seuil 0.3 du TextRank
    pct = (valeurs >= 0.3).mean() * 100
    print(f"  paires >= 0.30 (seuil TextRank actuel) : {pct:.1f} %")
    if pct > 80:
        print(f"  --> ALERTE : le seuil 0.30 ne filtre presque rien.")
        print(f"      PageRank va converger vers une distribution quasi-uniforme.")

    return valeurs


def correlation_longueur_centre(phrases, scores_centre, nom):
    """Spearman entre n_informatifs et similarite au centre."""
    longueurs = np.array([p["n_informatifs"] for p in phrases])
    rho, pval = spearmanr(longueurs, scores_centre)
    print(f"\n[{nom}] Correlation Spearman longueur vs similarite_au_centre :")
    print(f"  rho = {rho:+.3f}  (p = {pval:.2e})")
    if rho > 0.5:
        print(f"  --> ALERTE : les phrases longues sont mecaniquement plus proches du centre.")
        print(f"      La composante 'centre' agit comme un bonus longueur deguise.")
    return rho


def top_k_par_composante(phrases, valeurs, k, nom_composante):
    """Affiche les k phrases qui maximisent une composante donnee."""
    indices = np.argsort(valeurs)[::-1][:k]
    print(f"\n  Top {k} phrases par '{nom_composante}' :")
    for rang, i in enumerate(indices, 1):
        p = phrases[i]
        print(f"  [{rang:2d}] (n_inf={p['n_informatifs']:3d}, "
              f"score={valeurs[i]:+.3f}) {tronquer(p['texte'])}")
    return indices


def stats_top_k(phrases, indices, nom):
    """Statistiques agregees sur un top-k."""
    longueurs = [phrases[i]["n_informatifs"] for i in indices]
    propn = [phrases[i]["propn_ratio"] for i in indices]
    print(f"\n  Stats top-{len(indices)} [{nom}] :")
    print(f"    longueur informative moyenne : {np.mean(longueurs):.1f} "
          f"(min={min(longueurs)}, max={max(longueurs)})")
    print(f"    ratio noms propres moyen      : {np.mean(propn):.2f}")


# ============================================================================
# Visualisation
# ============================================================================

def figure_diagnostic(emb_cam, emb_tfidf, phrases,
                       centre_cam, centre_tfidf, chemin_sortie):
    """4 sous-graphes : distributions de cosinus + nuage longueur/centre."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogramme cosinus camembert
    sim_cam = (emb_cam @ emb_cam.T)[np.triu_indices(len(phrases), k=1)]
    axes[0, 0].hist(sim_cam, bins=60, color="steelblue", edgecolor="white")
    axes[0, 0].axvline(0.3, color="red", linestyle="--",
                        label="seuil TextRank actuel = 0.3")
    axes[0, 0].set_xlabel("cosinus inter-phrases")
    axes[0, 0].set_ylabel("nombre de paires")
    axes[0, 0].set_title("Camembert -- distribution des cosinus")
    axes[0, 0].legend()

    # 2. Histogramme cosinus TF-IDF
    # TF-IDF est creuse : beaucoup de paires a 0
    sim_tf = (emb_tfidf @ emb_tfidf.T)[np.triu_indices(len(phrases), k=1)]
    axes[0, 1].hist(sim_tf, bins=60, color="darkorange", edgecolor="white")
    axes[0, 1].axvline(0.3, color="red", linestyle="--",
                        label="seuil 0.3 (a titre indicatif)")
    axes[0, 1].set_xlabel("cosinus inter-phrases")
    axes[0, 1].set_ylabel("nombre de paires")
    axes[0, 1].set_title("TF-IDF maison -- distribution des cosinus")
    axes[0, 1].legend()

    # 3. Longueur vs centre (camembert)
    longueurs = [p["n_informatifs"] for p in phrases]
    axes[1, 0].scatter(longueurs, centre_cam, s=10, alpha=0.4, color="steelblue")
    axes[1, 0].set_xlabel("nombre de tokens informatifs")
    axes[1, 0].set_ylabel("similarite au centre")
    axes[1, 0].set_title("Camembert -- biais longueur vers le centre")

    # 4. Longueur vs centre (TF-IDF)
    axes[1, 1].scatter(longueurs, centre_tfidf, s=10, alpha=0.4, color="darkorange")
    axes[1, 1].set_xlabel("nombre de tokens informatifs")
    axes[1, 1].set_ylabel("similarite au centre")
    axes[1, 1].set_title("TF-IDF -- biais longueur vers le centre")

    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=120, bbox_inches="tight")
    print(f"\nFigure sauvegardee : {chemin_sortie}")


# ============================================================================
# Main
# ============================================================================

def main(motif="Au Bonheur"):
    titre(f"Diagnostic MMR -- recherche : '{motif}'")

    # 1. Chargement
    corpus = charger_corpus(verbose=True)
    livre = trouver_livre(corpus, motif)
    print(f"\nLivre trouve : {livre['livre']} -- {livre['auteur']} ({livre['genre']})")
    print(f"  {len(livre['df']):,} tokens annotes")

    # 2. Extraction des phrases candidates
    phrases = extraire_phrases(livre["df"])
    print(f"  {len(phrases):,} phrases candidates apres filtrage")
    if len(phrases) < 20:
        print("Trop peu de phrases pour un diagnostic utile.")
        return

    # 3. Encodage camembert (lent au premier appel : telechargement ~440 Mo)
    sous_titre("Encodage camembert")
    print("Encodage des phrases avec sentence-camembert (peut etre long)...")
    emb_cam = _encoder_phrases([p["texte"] for p in phrases])
    print(f"  embeddings : shape={emb_cam.shape}, "
          f"normes L2 entre {np.linalg.norm(emb_cam, axis=1).min():.3f} "
          f"et {np.linalg.norm(emb_cam, axis=1).max():.3f}")

    # 4. Vectorisation TF-IDF maison
    sous_titre("Vectorisation TF-IDF maison")
    emb_tfidf = vectoriser_phrases_tfidf(phrases, ngram_max=1, min_df=2)
    print(f"  matrice TF-IDF : shape={emb_tfidf.shape}")
    nnz = (emb_tfidf > 0).sum() / emb_tfidf.size * 100
    print(f"  densite : {nnz:.2f} % (TF-IDF naturellement creux)")

    # 5. Distributions de cosinus
    titre("1. Distribution des cosinus inter-phrases")
    distribution_cosinus(emb_cam, "camembert")
    distribution_cosinus(emb_tfidf, "tf-idf")

    # 6. Composantes du score
    titre("2. Composantes du score hybride -- camembert")
    centre_cam = similarite_au_centre(emb_cam)
    textrank_cam = centralite_textrank(emb_cam, seuil=0.3)
    longueur_score = bonus_longueur(phrases)

    correlation_longueur_centre(phrases, centre_cam, "camembert")

    # 7. Composantes - TF-IDF
    titre("3. Composantes du score hybride -- TF-IDF")
    centre_tfidf = similarite_au_centre(emb_tfidf)
    textrank_tfidf = centralite_textrank(emb_tfidf, seuil=0.3)
    correlation_longueur_centre(phrases, centre_tfidf, "tf-idf")

    # 8. Top-10 par composante isolee -- camembert
    titre("4. Top-10 par composante isolee (camembert)")
    k = 10
    idx_centre = top_k_par_composante(phrases, centre_cam, k, "centre (camembert)")
    stats_top_k(phrases, idx_centre, "centre cam")

    idx_textrank = top_k_par_composante(phrases, textrank_cam, k, "textrank (camembert)")
    stats_top_k(phrases, idx_textrank, "textrank cam")

    idx_longueur = top_k_par_composante(phrases, longueur_score, k, "bonus longueur")
    stats_top_k(phrases, idx_longueur, "longueur")

    # 9. Top-10 -- TF-IDF
    titre("5. Top-10 par composante isolee (TF-IDF)")
    idx_centre_tf = top_k_par_composante(phrases, centre_tfidf, k, "centre (tf-idf)")
    stats_top_k(phrases, idx_centre_tf, "centre tf-idf")

    idx_textrank_tf = top_k_par_composante(phrases, textrank_tfidf, k, "textrank (tf-idf)")
    stats_top_k(phrases, idx_textrank_tf, "textrank tf-idf")

    # 10. Recouvrement entre les selections
    titre("6. Recouvrement entre les selections (Jaccard sur top-10)")

    def jaccard(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

    paires = [
        ("centre cam",   idx_centre,      "centre tfidf",   idx_centre_tf),
        ("textrank cam", idx_textrank,    "textrank tfidf", idx_textrank_tf),
        ("centre cam",   idx_centre,      "textrank cam",   idx_textrank),
        ("centre cam",   idx_centre,      "longueur",       idx_longueur),
    ]
    for n1, i1, n2, i2 in paires:
        print(f"  {n1:20s} vs {n2:20s} : Jaccard = {jaccard(i1, i2):.2f}")

    # 11. Figure de synthese
    titre("7. Sauvegarde de la figure de diagnostic")
    figure_diagnostic(emb_cam, emb_tfidf, phrases,
                       centre_cam, centre_tfidf,
                       "diagnostic_mmr.png")

    # 12. Resume final
    titre("8. Verdict")
    print("Lecture des resultats :")
    print()
    print("- Si 'paires >= 0.30' depasse 80 % en camembert : seuil TextRank a remonter.")
    print("  Recalibrer sur le quantile 75 ou 90 plutot qu'une valeur fixe.")
    print()
    print("- Si la correlation longueur/centre est > 0.5 : le score 'centre' est un")
    print("  bonus longueur deguise. Soit on retire le bonus longueur explicite,")
    print("  soit on normalise les embeddings par longueur, soit on passe en TF-IDF")
    print("  pour la pertinence (la creusite limite ce biais).")
    print()
    print("- Si les top-10 'centre cam' et 'centre tfidf' ont un Jaccard < 0.2 :")
    print("  les deux representations capturent des choses differentes. La selection")
    print("  finale depend lourdement du choix d'embedding -- verifier visuellement")
    print("  laquelle resume mieux le livre.")


if __name__ == "__main__":
    motif = sys.argv[1] if len(sys.argv) > 1 else "Au Bonheur"
    main(motif)