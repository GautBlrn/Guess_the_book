"""
Audit du corpus annoté : qualité globale + segmentation en phrases.

Deux volets :

VOLET 1 -- Qualité par livre
- Nombre de tokens, lemmes uniques (richesse lexicale)
- Distribution POS (alerte si une POS est anormalement haute/basse)
- Top 20 lemmes informatifs (alerte si artéfacts Gutenberg restent)
- Top 10 tokens bruts ponctuation incluse (détecte les `[`, `*`, etc.)
- Ratio is_alpha (alerte si trop de non-alphabétiques = pollution)

VOLET 2 -- Segmentation en phrases
- Distribution du nombre de tokens par phrase (par livre)
- Phrases anormalement longues (>200 tokens : titre/paragraphe collé)
- Phrases anormalement courtes (<5 tokens : titre isolé, marqueur)
- Pourcentage de phrases qui passeraient le filtre `extraire_phrases`

Sortie :
- Console : un bloc par livre, avec marqueurs ALERTE
- Fichier : audit_corpus.csv (ligne par livre, colonnes mesurées)
- Figure : audit_corpus.png (4 sous-graphes de synthèse)

Usage :
    python audit_corpus.py
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import MMR_PARAMS
from pipeline.corpus import charger_corpus


# Largeurs typographiques
LARGEUR = 90
SEP = "=" * LARGEUR


# Seuils d'alerte (heuristiques littéraires français XIXe)
SEUILS = {
    "propn_min":          0.025,   # < 2.5 % = anormalement bas pour roman/récit
    "propn_max":          0.10,    # > 10 % = liste de noms ou index
    "cconj_max":          0.05,    # > 5 % = traduction calquée ou énumeration
    "punct_max":          0.20,    # > 20 % = sur-ponctuation (dialogues hyper-decoupes)
    "alpha_min":          0.65,    # < 65 % = pollution ou texte tres ponctué
    "phrase_longue":      200,     # tokens
    "phrase_courte":      5,
    "ratio_phrases_min":  0.30,    # % phrases qui passent extraire_phrases
}


# ============================================================================
# Volet 1 : qualité par livre
# ============================================================================

def stats_livre(c):
    """Calcule un dict de stats pour un livre annote."""
    df = c["df"]
    n_tokens = len(df)

    pos_counts = df["pos"].value_counts()
    pos_ratios = (pos_counts / n_tokens).to_dict()

    masque_inf = df["is_alpha"] & ~df["is_stop"] & ~df["is_punct"]
    n_informatifs = int(masque_inf.sum())
    lemmes_inf = df.loc[masque_inf, "lemma"].str.lower()
    lemmes_uniques = lemmes_inf.nunique()

    # TTR (type-token ratio) sur les informatifs : indicateur de richesse
    ttr = lemmes_uniques / max(n_informatifs, 1)

    # Top lemmes informatifs et tokens bruts (pour traquer les artefacts)
    top_lemmes = lemmes_inf.value_counts().head(20)
    top_tokens_bruts = df["text"].value_counts().head(15)

    return {
        "auteur":          c["auteur"],
        "livre":           c["livre"],
        "genre":           c["genre"],
        "n_tokens":        n_tokens,
        "n_informatifs":   n_informatifs,
        "n_phrases":       int(df["sent_id"].nunique()),
        "lemmes_uniques":  int(lemmes_uniques),
        "ttr_informatifs": float(ttr),
        "ratio_alpha":     float(df["is_alpha"].mean()),
        "ratio_punct":     float(df["is_punct"].mean()),
        "ratio_stop":      float(df["is_stop"].mean()),
        "pos_ratios":      pos_ratios,
        "top_lemmes":      top_lemmes,
        "top_tokens":      top_tokens_bruts,
    }


def alertes_qualite(stats):
    """Retourne la liste des alertes pour un livre (chaines de caracteres)."""
    alertes = []
    pos = stats["pos_ratios"]

    propn = pos.get("PROPN", 0.0)
    if propn < SEUILS["propn_min"]:
        alertes.append(f"PROPN tres bas ({propn*100:.1f} %)"
                       " -- noms propres mal taggés ?")
    elif propn > SEUILS["propn_max"]:
        alertes.append(f"PROPN tres haut ({propn*100:.1f} %)"
                       " -- liste de noms ou index ?")

    cconj = pos.get("CCONJ", 0.0)
    if cconj > SEUILS["cconj_max"]:
        alertes.append(f"CCONJ tres haut ({cconj*100:.1f} %)"
                       " -- traduction calquee ?")

    punct = stats["ratio_punct"]
    if punct > SEUILS["punct_max"]:
        alertes.append(f"PUNCT tres haut ({punct*100:.1f} %)"
                       " -- dialogues sur-decoupes ?")

    if stats["ratio_alpha"] < SEUILS["alpha_min"]:
        alertes.append(f"ratio_alpha bas ({stats['ratio_alpha']*100:.1f} %)"
                       " -- pollution ?")

    # Detection d'artefacts dans le top tokens bruts
    suspects = {"[", "]", "*", "#", "_", "Illustration", "ILLUSTRATION",
                 "Gutenberg", "Project"}
    artefacts_trouves = [t for t in stats["top_tokens"].index
                         if any(s in str(t) for s in suspects)]
    if artefacts_trouves:
        alertes.append(f"artefacts dans top tokens : {artefacts_trouves[:3]}")

    return alertes


# ============================================================================
# Volet 2 : segmentation en phrases
# ============================================================================

def stats_segmentation(c):
    """Stats sur la distribution des longueurs de phrases."""
    df = c["df"]
    longueurs = df.groupby("sent_id").size()

    # Phrases qui passeraient extraire_phrases (filtre min_tokens / max_tokens)
    min_tok = MMR_PARAMS["min_tokens_phrase"]
    max_tok = MMR_PARAMS["max_tokens_phrase"]
    masque_inf = df["is_alpha"] & ~df["is_stop"] & ~df["is_punct"]
    longueurs_inf = df[masque_inf].groupby("sent_id").size()

    n_phrases = len(longueurs)
    n_passe_filtre = int(((longueurs_inf >= min_tok)
                          & (longueurs_inf <= max_tok)).sum())
    ratio_passe = n_passe_filtre / max(n_phrases, 1)

    return {
        "n_phrases":            n_phrases,
        "longueur_mediane":     float(longueurs.median()),
        "longueur_p95":         float(longueurs.quantile(0.95)),
        "longueur_max":         int(longueurs.max()),
        "n_phrases_longues":    int((longueurs > SEUILS["phrase_longue"]).sum()),
        "n_phrases_courtes":    int((longueurs < SEUILS["phrase_courte"]).sum()),
        "n_passe_filtre":       n_passe_filtre,
        "ratio_passe_filtre":   float(ratio_passe),
        "longueurs_brutes":     longueurs,  # pour la figure
    }


def alertes_segmentation(seg):
    alertes = []
    if seg["n_phrases_longues"] > 5:
        alertes.append(f"{seg['n_phrases_longues']} phrases > "
                       f"{SEUILS['phrase_longue']} tokens"
                       " -- titres colles ou paragraphes mal segmentes")
    if seg["longueur_max"] > 1000:
        alertes.append(f"phrase max = {seg['longueur_max']} tokens"
                       " -- bloc entier non segmente")
    if seg["ratio_passe_filtre"] < SEUILS["ratio_phrases_min"]:
        alertes.append(
            f"seulement {seg['ratio_passe_filtre']*100:.0f} % des phrases "
            "passent le filtre extraire_phrases"
        )
    return alertes


# ============================================================================
# Affichage console
# ============================================================================

def afficher_livre(stats, seg, alertes):
    """Bloc console pour un livre."""
    print(f"\n{stats['auteur']:30s} -- {stats['livre']}")
    print(f"  genre={stats['genre']:18s} "
          f"tokens={stats['n_tokens']:>7,}  "
          f"phrases={seg['n_phrases']:>5,}  "
          f"lemmes uniques={stats['lemmes_uniques']:>5,}  "
          f"TTR={stats['ttr_informatifs']:.3f}")

    pos = stats["pos_ratios"]
    print(f"  POS : NOUN={pos.get('NOUN', 0)*100:4.1f}  "
          f"VERB={pos.get('VERB', 0)*100:4.1f}  "
          f"PROPN={pos.get('PROPN', 0)*100:4.1f}  "
          f"ADJ={pos.get('ADJ', 0)*100:4.1f}  "
          f"ADV={pos.get('ADV', 0)*100:4.1f}  "
          f"CCONJ={pos.get('CCONJ', 0)*100:4.1f}  "
          f"PUNCT={stats['ratio_punct']*100:4.1f}")

    print(f"  Segmentation : phrase mediane={seg['longueur_mediane']:.0f} tok, "
          f"p95={seg['longueur_p95']:.0f}, max={seg['longueur_max']}, "
          f"longues(>{SEUILS['phrase_longue']})={seg['n_phrases_longues']}, "
          f"courtes(<{SEUILS['phrase_courte']})={seg['n_phrases_courtes']}, "
          f"passe filtre={seg['ratio_passe_filtre']*100:.0f}%")

    print(f"  Top lemmes informatifs : "
          f"{list(stats['top_lemmes'].head(8).index)}")

    if alertes:
        for a in alertes:
            print(f"  ALERTE : {a}")


# ============================================================================
# Figure de synthèse
# ============================================================================

def figure_synthese(rangees, chemin):
    """4 sous-graphes : POS, longueurs phrases, ratio passe filtre, TTR."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. PROPN par livre (ordonné par valeur)
    df = pd.DataFrame(rangees).sort_values("propn_pct")
    axes[0, 0].barh(df["auteur"], df["propn_pct"], color="steelblue")
    axes[0, 0].axvline(SEUILS["propn_min"] * 100, color="red", linestyle="--",
                        label=f"seuil bas {SEUILS['propn_min']*100:.1f}%")
    axes[0, 0].set_xlabel("% PROPN")
    axes[0, 0].set_title("PROPN par livre (alertes a gauche)")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="y", labelsize=7)

    # 2. Phrases longues par livre (échelle log si nécessaire)
    df2 = df.sort_values("n_phrases_longues", ascending=True)
    axes[0, 1].barh(df2["auteur"], df2["n_phrases_longues"], color="darkorange")
    axes[0, 1].set_xlabel(f"nombre de phrases > {SEUILS['phrase_longue']} tokens")
    axes[0, 1].set_title("Phrases anormalement longues par livre")
    axes[0, 1].tick_params(axis="y", labelsize=7)

    # 3. Ratio phrases qui passent le filtre
    df3 = df.sort_values("ratio_passe", ascending=True)
    couleurs = ["red" if r < SEUILS["ratio_phrases_min"] else "seagreen"
                for r in df3["ratio_passe"]]
    axes[1, 0].barh(df3["auteur"], df3["ratio_passe"] * 100, color=couleurs)
    axes[1, 0].axvline(SEUILS["ratio_phrases_min"] * 100, color="red",
                        linestyle="--",
                        label=f"seuil {SEUILS['ratio_phrases_min']*100:.0f}%")
    axes[1, 0].set_xlabel("% phrases passant le filtre extraire_phrases")
    axes[1, 0].set_title("Phrases utilisables pour MMR")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis="y", labelsize=7)

    # 4. TTR vs nb tokens (richesse lexicale relative a la taille)
    axes[1, 1].scatter(df["n_tokens"], df["ttr"], s=40, alpha=0.7)
    for _, r in df.iterrows():
        axes[1, 1].annotate(r["auteur"].split()[0], (r["n_tokens"], r["ttr"]),
                              fontsize=7, alpha=0.8)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("nombre de tokens (log)")
    axes[1, 1].set_ylabel("TTR sur lemmes informatifs")
    axes[1, 1].set_title("Richesse lexicale relative")

    plt.tight_layout()
    plt.savefig(chemin, dpi=110, bbox_inches="tight")
    print(f"\nFigure sauvegardee : {chemin}")


# ============================================================================
# Main
# ============================================================================

def main():
    print(SEP)
    print("AUDIT DU CORPUS ANNOTE")
    print(SEP)

    corpus = charger_corpus(verbose=True)
    print(f"\n{len(corpus)} livres charges. Audit en cours...\n")

    livres_avec_alertes = []
    rangees_csv = []
    rangees_fig = []

    for c in corpus:
        stats = stats_livre(c)
        seg = stats_segmentation(c)
        alertes = alertes_qualite(stats) + alertes_segmentation(seg)
        afficher_livre(stats, seg, alertes)

        if alertes:
            livres_avec_alertes.append((stats["auteur"], alertes))

        # Pour CSV et figure : version aplatie
        ligne = {
            "auteur":             stats["auteur"],
            "livre":              stats["livre"],
            "genre":              stats["genre"],
            "n_tokens":           stats["n_tokens"],
            "n_phrases":          seg["n_phrases"],
            "lemmes_uniques":     stats["lemmes_uniques"],
            "ttr":                stats["ttr_informatifs"],
            "propn_pct":          stats["pos_ratios"].get("PROPN", 0) * 100,
            "cconj_pct":          stats["pos_ratios"].get("CCONJ", 0) * 100,
            "punct_pct":          stats["ratio_punct"] * 100,
            "phrase_mediane":     seg["longueur_mediane"],
            "phrase_p95":         seg["longueur_p95"],
            "phrase_max":         seg["longueur_max"],
            "n_phrases_longues":  seg["n_phrases_longues"],
            "n_phrases_courtes":  seg["n_phrases_courtes"],
            "ratio_passe":        seg["ratio_passe_filtre"],
            "n_alertes":          len(alertes),
        }
        rangees_csv.append(ligne)
        rangees_fig.append(ligne)

    # Récap des alertes
    print(f"\n{SEP}")
    print("RECAPITULATIF DES ALERTES")
    print(SEP)
    if not livres_avec_alertes:
        print("Aucune alerte. Le corpus est sain.")
    else:
        print(f"{len(livres_avec_alertes)} livre(s) avec alerte(s) :\n")
        for auteur, alertes in livres_avec_alertes:
            print(f"  {auteur}")
            for a in alertes:
                print(f"    - {a}")

    # Sauvegarde CSV
    df_csv = pd.DataFrame(rangees_csv)
    df_csv.to_csv("audit_corpus.csv", index=False)
    print(f"\nCSV sauvegarde : audit_corpus.csv ({len(df_csv)} lignes)")

    # Figure
    figure_synthese(rangees_fig, "audit_corpus.png")


if __name__ == "__main__":
    main()