"""
Tests de `pipeline.benchmark`.

Le corpus est synthetique et l'encodeur de phrases est injecte : aucun
acces S3, aucun telechargement de camembert. Ce qui est teste ici, ce
n'est pas la qualite des resultats (elle depend du vrai corpus) mais que
l'orchestration ne se trompe pas : appariement extrait -> bon livre,
partage du pool d'embeddings, forme des tableaux, et les quelques pieges
ou une erreur passerait inapercue en produisant des chiffres plausibles.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from pipeline.benchmark import (
    METHODES_RESUME,
    balayer_lambda,
    couverture,
    evaluer_recherche,
    evaluer_resumes,
    preparer_pool,
    redondance,
    resumes_par_methode,
    tirer_extraits,
)

# min_df=1 : chaque livre synthetique a un vocabulaire qui lui est propre,
# donc avec le min_df=2 de production tous les termes discriminants
# seraient filtres et le corpus deviendrait inidentifiable.
PARAMS_TEST = {"min_df": 1, "max_df_ratio": 1.0}
K_TEST = 4


def _livre(slug: str, vocabulaire: list[str], n_termes: int, seed: int) -> dict:
    """Livre synthetique : termes tires dans un vocabulaire qui lui est propre."""
    rng = np.random.default_rng(seed)
    lemmes = [vocabulaire[i] for i in rng.integers(0, len(vocabulaire), n_termes)]
    return {
        "livre_slug": slug,
        "livre":      slug.replace("_", " ").title(),
        "auteur":     f"Auteur {slug[-1]}",
        "genre":      "roman",
        "lemmes":     lemmes,
        # `tokens` doit rester aligne index par index sur `lemmes` : c'est
        # l'hypothese sur laquelle `tirer_extraits` repose pour n'utiliser
        # qu'un seul `i_debut`.
        "tokens":     [t.upper() for t in lemmes],
    }


@pytest.fixture
def corpus():
    """Quatre livres a vocabulaires disjoints, donc parfaitement separables."""
    vocabulaires = [
        ["navire", "voile", "ocean", "tempete", "capitaine"],
        ["chateau", "duchesse", "bal", "carrosse", "valet"],
        ["moteur", "fusee", "orbite", "planete", "cosmonaute"],
        ["poison", "enquete", "inspecteur", "alibi", "cadavre"],
    ]
    return [
        _livre(f"livre_{i}", vocab, n_termes=400, seed=i)
        for i, vocab in enumerate(vocabulaires)
    ]


def _df_annote(phrases: list[list[str]]) -> pd.DataFrame:
    """DataFrame annote minimal, au format attendu par `extraire_phrases`."""
    lignes = []
    for sent_id, mots in enumerate(phrases):
        for mot in mots:
            lignes.append({
                "sent_id":  sent_id,
                "text":     mot,
                "lemma":    mot,
                "pos":      "NOUN",
                "is_alpha": True,
                "is_stop":  False,
                "is_punct": False,
            })
    return pd.DataFrame(lignes)


@pytest.fixture
def livre_annote():
    """Un livre annote de 25 phrases de 10 termes, vocabulaire recurrent."""
    rng = np.random.default_rng(7)
    vocab = ["navire", "voile", "ocean", "tempete", "capitaine",
             "port", "vague", "mat", "cale", "equipage"]
    phrases = [
        [vocab[i] for i in rng.integers(0, len(vocab), 10)]
        for _ in range(25)
    ]
    df = _df_annote(phrases)
    return {
        "livre_slug": "livre_mer",
        "livre":      "Livre Mer",
        "auteur":     "Auteur M",
        "genre":      "aventure",
        "df":         df,
        "lemmes":     df["lemma"].tolist(),
        "tokens":     df["text"].tolist(),
    }


def _encodeur_factice(dim: int = 8, seed: int = 3):
    """Encodeur deterministe renvoyant des vecteurs L2-normalises."""
    def encoder(textes):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(len(textes), dim))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return encoder


# ============================================================================
# Jeu de test
# ============================================================================


def test_tirer_extraits_taille_et_nombre(corpus):
    extraits = tirer_extraits(corpus, n_par_livre=3, taille=50, seed=0)
    assert len(extraits) == 3 * len(corpus)
    assert all(len(e["lemmes"]) == 50 for e in extraits)
    assert all(len(e["tokens"]) == 50 for e in extraits)


def test_tirer_extraits_plage_fait_varier_la_longueur(corpus):
    extraits = tirer_extraits(corpus, n_par_livre=10, taille=(7, 70), seed=0)
    tailles = [e["taille"] for e in extraits]
    assert all(7 <= t <= 70 for t in tailles)
    assert all(len(e["lemmes"]) == e["taille"] for e in extraits)
    # Une plage doit reellement produire des longueurs differentes : si
    # l'implementation retombait sur une valeur unique, le protocole
    # redeviendrait "longueur fixe" sans que rien ne le signale.
    assert len(set(tailles)) > 5


def test_tirer_extraits_int_reste_a_longueur_fixe(corpus):
    extraits = tirer_extraits(corpus, n_par_livre=5, taille=40, seed=0)
    assert {e["taille"] for e in extraits} == {40}


def test_tirer_extraits_refuse_une_plage_invalide(corpus):
    with pytest.raises(ValueError, match="plage de taille invalide"):
        tirer_extraits(corpus, taille=(70, 7))
    with pytest.raises(ValueError, match="plage de taille invalide"):
        tirer_extraits(corpus, taille=0)


def test_tirer_extraits_borne_par_la_longueur_du_livre(corpus):
    # Un livre plus court que le haut de la plage doit fournir des
    # extraits plus courts, pas etre exclu ni lever.
    corpus[0]["lemmes"] = corpus[0]["lemmes"][:25]
    corpus[0]["tokens"] = corpus[0]["tokens"][:25]
    extraits = tirer_extraits(corpus, n_par_livre=5, taille=(7, 70), seed=0)
    du_court = [e for e in extraits if e["livre_slug"] == corpus[0]["livre_slug"]]
    assert len(du_court) == 5
    assert all(e["taille"] <= 25 for e in du_court)


def test_tirer_extraits_lemmes_et_tokens_alignes(corpus):
    # Un seul `i_debut` sert pour les deux champs : si l'alignement casse,
    # les experiences "tokens" evalueraient des extraits d'un autre passage
    # que les experiences "lemmes", sans que rien ne le signale.
    extraits = tirer_extraits(corpus, n_par_livre=2, taille=30, seed=0)
    for e in extraits:
        assert [t.upper() for t in e["lemmes"]] == e["tokens"]


def test_tirer_extraits_est_contigu(corpus):
    extraits = tirer_extraits(corpus, n_par_livre=2, taille=40, seed=1)
    par_slug = {c["livre_slug"]: c for c in corpus}
    for e in extraits:
        source = par_slug[e["livre_slug"]]["lemmes"]
        i = e["i_debut"]
        assert e["lemmes"] == source[i:i + 40]


def test_tirer_extraits_est_reproductible(corpus):
    a = tirer_extraits(corpus, n_par_livre=2, taille=20, seed=42)
    b = tirer_extraits(corpus, n_par_livre=2, taille=20, seed=42)
    c = tirer_extraits(corpus, n_par_livre=2, taille=20, seed=43)
    assert [e["i_debut"] for e in a] == [e["i_debut"] for e in b]
    assert [e["i_debut"] for e in a] != [e["i_debut"] for e in c]


def test_tirer_extraits_saute_les_livres_trop_courts(corpus):
    corpus[0]["lemmes"] = corpus[0]["lemmes"][:10]
    corpus[0]["tokens"] = corpus[0]["tokens"][:10]
    extraits = tirer_extraits(corpus, n_par_livre=2, taille=50, seed=0)
    slugs = {e["livre_slug"] for e in extraits}
    assert corpus[0]["livre_slug"] not in slugs
    assert len(extraits) == 2 * 3


# ============================================================================
# Experiences de recherche
# ============================================================================


def test_evaluer_recherche_identifie_un_corpus_separable(corpus):
    extraits = tirer_extraits(corpus, n_par_livre=3, taille=60, seed=0)
    table = evaluer_recherche(
        corpus, extraits, metriques=["cosinus"],
        params_tfidf=PARAMS_TEST, verbose=False,
    )
    # Vocabulaires disjoints : le bon livre doit toujours sortir premier.
    assert (table["rappel@1"] == 1.0).all()
    assert (table["mrr"] == 1.0).all()


def test_evaluer_recherche_croise_configs_et_metriques(corpus):
    extraits = tirer_extraits(corpus, n_par_livre=1, taille=40, seed=0)
    configs = {
        "lemmes_1g": {"champ": "lemmes", "ngram_max": 1},
        "tokens_1g": {"champ": "tokens", "ngram_max": 1},
    }
    table = evaluer_recherche(
        corpus, extraits, configs=configs,
        metriques=["cosinus", "jaccard"],
        params_tfidf=PARAMS_TEST, verbose=False,
    )
    assert len(table) == 4
    assert set(table["representation"]) == {"lemmes_1g", "tokens_1g"}
    assert set(table["metrique"]) == {"cosinus", "jaccard"}


def test_evaluer_recherche_accepte_des_iterateurs(corpus):
    # Regression : `metriques` et `ks` sont reparcourus a chaque config.
    # Avec un generateur non materialise, la deuxieme representation
    # produisait zero ligne et le tableau sortait incomplet en silence.
    extraits = tirer_extraits(corpus, n_par_livre=1, taille=40, seed=0)
    configs = {
        "a": {"champ": "lemmes", "ngram_max": 1},
        "b": {"champ": "tokens", "ngram_max": 1},
    }
    table = evaluer_recherche(
        corpus, extraits, configs=configs,
        metriques=iter(["cosinus", "euclidien"]),
        ks=iter([1, 5]),
        params_tfidf=PARAMS_TEST, verbose=False,
    )
    assert len(table) == 4
    assert {"rappel@1", "rappel@5", "precision@1", "precision@5"} <= set(table.columns)


def test_evaluer_recherche_refuse_un_jeu_vide(corpus):
    with pytest.raises(ValueError, match="Jeu de test vide"):
        evaluer_recherche(corpus, [], verbose=False)


def test_evaluer_recherche_refuse_une_metrique_inconnue(corpus):
    extraits = tirer_extraits(corpus, n_par_livre=1, taille=40, seed=0)
    with pytest.raises(ValueError, match="metrique inconnue"):
        evaluer_recherche(
            corpus, extraits, metriques=["mahalanobis"],
            params_tfidf=PARAMS_TEST, verbose=False,
        )


def test_evaluer_recherche_n_utilise_pas_les_extraits_pour_l_idf(corpus):
    # Les extraits passent par `transform`, pas `fit_transform` : ajouter
    # des extraits ne doit pas changer la representation du corpus, donc
    # les scores des extraits communs restent identiques.
    peu = tirer_extraits(corpus, n_par_livre=1, taille=40, seed=0)
    beaucoup = tirer_extraits(corpus, n_par_livre=6, taille=40, seed=0)
    args = dict(metriques=["cosinus"], params_tfidf=PARAMS_TEST, verbose=False)
    assert (
        evaluer_recherche(corpus, peu, **args)["mrr"].iloc[0]
        == evaluer_recherche(corpus, beaucoup, **args)["mrr"].iloc[0]
    )


# ============================================================================
# Experiences de resume
# ============================================================================


def test_preparer_pool_encode_chaque_phrase(livre_annote):
    phrases, embeddings = preparer_pool(
        livre_annote["df"], encodeur=_encodeur_factice()
    )
    assert len(phrases) == 25
    assert embeddings.shape == (25, 8)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_resumes_par_methode_produit_les_trois_arms(livre_annote):
    pool = preparer_pool(livre_annote["df"], encodeur=_encodeur_factice())
    resumes = resumes_par_methode(livre_annote["df"], k=K_TEST, pool=pool)
    assert set(resumes) == set(METHODES_RESUME)
    for methode, selection in resumes.items():
        assert len(selection) == K_TEST, methode
        assert all("termes" in p and "sent_id" in p for p in selection)


def test_poids_est_transmis_jusqu_au_score_hybride(livre_annote, monkeypatch):
    # Le passthrough doit atteindre `score_hybride` : sans lui, un balayage
    # de ponderation renverrait des resultats identiques sans rien signaler.
    from pipeline import summarization

    vus = []
    vrai = summarization.score_hybride

    def espion(phrases, matrice_tfidf, emb, poids=None):
        vus.append(poids)
        return vrai(phrases, matrice_tfidf, emb, poids=poids)

    monkeypatch.setattr(summarization, "score_hybride", espion)

    pool = preparer_pool(livre_annote["df"], encodeur=_encodeur_factice())
    voulu = {"centre": 1.0, "textrank": 0.0, "longueur": 0.0}
    resumes_par_methode(
        livre_annote["df"], methodes=["mmr"], k=K_TEST, pool=pool, poids=voulu
    )
    assert vus == [voulu]


def test_poids_change_la_selection_mmr(livre_annote):
    # Deux ponderations opposees ne doivent pas donner le meme resume,
    # sinon le parametre serait accepte puis ignore quelque part.
    pool = preparer_pool(livre_annote["df"], encodeur=_encodeur_factice())
    args = dict(methodes=["mmr"], k=K_TEST, pool=pool)
    centre = resumes_par_methode(
        livre_annote["df"], poids={"centre": 1.0, "textrank": 0.0, "longueur": 0.0}, **args
    )["mmr"]
    textrank = resumes_par_methode(
        livre_annote["df"], poids={"centre": 0.0, "textrank": 1.0, "longueur": 0.0}, **args
    )["mmr"]
    assert [p["sent_id"] for p in centre] != [p["sent_id"] for p in textrank]


def test_resumes_par_methode_livre_trop_court(livre_annote):
    pool = preparer_pool(livre_annote["df"], encodeur=_encodeur_factice())
    resumes = resumes_par_methode(livre_annote["df"], k=999, pool=pool)
    assert all(selection == [] for selection in resumes.values())


def test_resumes_par_methode_refuse_une_methode_inconnue(livre_annote):
    pool = preparer_pool(livre_annote["df"], encodeur=_encodeur_factice())
    with pytest.raises(ValueError, match="methode inconnue"):
        resumes_par_methode(
            livre_annote["df"], methodes=["lead_k"], k=K_TEST, pool=pool
        )


def test_redondance_bornes():
    phrases = [{"sent_id": i} for i in range(3)]
    identiques = np.tile(np.array([[1.0, 0.0]]), (3, 1))
    assert redondance(phrases, phrases, identiques) == pytest.approx(1.0)
    assert redondance(phrases[:1], phrases, identiques) == 0.0

    orthogonaux = np.eye(3)
    assert redondance(phrases, phrases, orthogonaux) == pytest.approx(0.0)


def test_redondance_leve_si_le_pool_ne_correspond_pas():
    # Le piege que ce garde-fou protege : `resumer_livre` reconstruit ses
    # propres dicts de phrases, donc tout raccordement par identite ou par
    # position renverrait 0.0 en silence -- une redondance parfaite,
    # exactement sur la colonne censee departager MMR.
    pool = [{"sent_id": i} for i in range(3)]
    selection = [{"sent_id": 99}]
    with pytest.raises(KeyError, match="absentes du pool"):
        redondance(selection, pool, np.eye(3))


def test_redondance_relie_par_sent_id_pas_par_position():
    pool = [{"sent_id": 10}, {"sent_id": 20}, {"sent_id": 30}]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Les deux premieres phrases du pool sont identiques : une selection
    # qui les designe par sent_id doit voir une redondance de 1.
    assert redondance(
        [{"sent_id": 10}, {"sent_id": 20}], pool, embeddings
    ) == pytest.approx(1.0)
    assert redondance(
        [{"sent_id": 10}, {"sent_id": 30}], pool, embeddings
    ) == pytest.approx(0.0)


def test_couverture_borne_et_monotone():
    reference = ["a", "a", "b", "c"]
    assert couverture([], reference) == 0.0
    assert couverture([{"termes": ["z"]}], reference) == 0.0
    # "a" pese 2 tokens sur 4 : couvrir "a" seul vaut deja 0.5.
    assert couverture([{"termes": ["a"]}], reference) == pytest.approx(0.5)
    assert couverture([{"termes": ["a", "b", "c"]}], reference) == pytest.approx(1.0)
    assert couverture([{"termes": ["a"]}], []) == 0.0


def test_couverture_recompense_la_diversite():
    # Deux resumes de MEME longueur totale : celui qui repete un terme
    # couvre moins. C'est exactement ce que le rappel ROUGE ne voit pas.
    reference = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    repetitif = [{"termes": ["a", "a"]}, {"termes": ["a", "a"]}]
    diversifie = [{"termes": ["a", "b"]}, {"termes": ["c", "a"]}]
    assert couverture(diversifie, reference) > couverture(repetitif, reference)


def test_evaluer_resumes_une_ligne_par_methode(livre_annote):
    table = evaluer_resumes(
        [livre_annote], k=K_TEST,
        encodeur=_encodeur_factice(), verbose=False,
    )
    assert list(table["methode"]) == list(METHODES_RESUME)
    assert (table["n_resumes"] == 1).all()
    # `lambda` n'a de sens que pour MMR.
    assert table.loc[table["methode"] == "mmr", "lambda"].notna().all()
    assert table.loc[table["methode"] != "mmr", "lambda"].isna().all()
    for colonne in ("rouge1_precision", "rouge2_f1", "redondance", "couverture"):
        assert colonne in table.columns
    # `avec_l=False` par defaut : la LCS contre le texte integral est le
    # poste qui ferait exploser le temps de calcul.
    assert not any(c.startswith("rougeL_") for c in table.columns)


def test_evaluer_resumes_avec_l_ajoute_les_colonnes(livre_annote):
    table = evaluer_resumes(
        [livre_annote], k=K_TEST, avec_l=True,
        encodeur=_encodeur_factice(), verbose=False,
    )
    assert "rougeL_f1" in table.columns


def test_evaluer_resumes_saute_les_livres_trop_courts(livre_annote):
    table = evaluer_resumes(
        [livre_annote], k=999,
        encodeur=_encodeur_factice(), verbose=False,
    )
    assert table.empty


def test_balayer_lambda_une_ligne_par_valeur(livre_annote):
    lambdas = [0.3, 0.6, 1.0]
    table = balayer_lambda(
        [livre_annote], lambdas=lambdas, k=K_TEST,
        encodeur=_encodeur_factice(), verbose=False,
    )
    assert list(table["lambda"]) == lambdas
    assert (table["methode"] == "mmr").all()
    assert (table["n_resumes"] == 1).all()


# ============================================================================
# Script CLI
# ============================================================================


def test_run_benchmark_recherche_ecrit_les_csv(corpus, tmp_path, monkeypatch):
    # Cable l'entree du script sur un corpus synthetique : verifie argparse,
    # l'enchainement et l'export sans toucher S3. `--skip-resumes` evite
    # camembert, que le script ne permet pas d'injecter.
    from scripts import run_benchmark

    monkeypatch.setattr(run_benchmark, "charger_corpus", lambda: corpus)
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark",
        "--skip-resumes",
        "--sortie", str(tmp_path),
        "--n-extraits", "2",
        "--taille-extrait", "40",
        "--k-max", "3",
    ])

    assert run_benchmark.main() == 0
    recherche = pd.read_csv(tmp_path / "recherche.csv")
    courbes = pd.read_csv(tmp_path / "recherche_courbes.csv")
    assert len(recherche) == 3 * 3          # 3 representations x 3 metriques
    assert set(courbes["k"]) == {1, 2, 3}
    assert not (tmp_path / "resumes.csv").exists()


def test_run_benchmark_signale_un_corpus_vide(tmp_path, monkeypatch):
    from scripts import run_benchmark

    monkeypatch.setattr(run_benchmark, "charger_corpus", lambda: [])
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark", "--sortie", str(tmp_path),
    ])
    assert run_benchmark.main() == 1


def test_run_benchmark_signale_des_livres_trop_courts(corpus, tmp_path, monkeypatch):
    # Aucun livre n'atteint la taille demandee : le script doit le dire
    # plutot que d'exporter un tableau vide ou de lever dans `evaluer_recherche`.
    from scripts import run_benchmark

    monkeypatch.setattr(run_benchmark, "charger_corpus", lambda: corpus)
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark", "--skip-resumes",
        "--sortie", str(tmp_path),
        "--taille-extrait", "99999",
    ])
    assert run_benchmark.main() == 0
    assert not (tmp_path / "recherche.csv").exists()


def test_balayer_lambda_partage_le_pool(livre_annote, monkeypatch):
    # Le pool d'embeddings doit etre calcule une fois par livre, pas une
    # fois par lambda : c'est tout l'interet de la boucle livre-par-livre.
    appels = []
    encodeur = _encodeur_factice()

    def encodeur_compte(textes):
        appels.append(len(textes))
        return encodeur(textes)

    balayer_lambda(
        [livre_annote], lambdas=[0.3, 0.6, 1.0], k=K_TEST,
        encodeur=encodeur_compte, verbose=False,
    )
    assert len(appels) == 1, f"{len(appels)} encodages pour 1 livre x 3 lambdas"
