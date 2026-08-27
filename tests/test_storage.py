"""
Tests de la reprise reseau de `pipeline.storage`.

Un chargement de corpus enchaine un GET par livre. A 26 livres, une coupure
passagere etait improbable ; a 291, un seul `ReadTimeoutError` faisait
perdre les 290 lectures precedentes et le run entier.

Le piege est que `ReadTimeoutError` survient pendant `Body.read()`, donc
APRES que botocore a rendu la main sur la requete : sa couche de reessai ne
peut plus rien, le flux est mort. Il faut reemettre le GET soi-meme, et
c'est ce que ces tests verifient.

Aucun acces reseau : le client S3 est remplace par un double.
"""
from __future__ import annotations

import botocore.exceptions as bex
import pytest

from pipeline import storage


class FauxCorps:
    """Corps de reponse S3 qui echoue les `n` premieres lectures."""

    def __init__(self, contenu, echecs, erreur):
        self.contenu = contenu
        self.echecs = echecs
        self.erreur = erreur

    def read(self):
        if self.echecs > 0:
            self.echecs -= 1
            raise self.erreur
        return self.contenu


class FauxClient:
    """Client S3 minimal qui compte les GET et scenarise les echecs."""

    def __init__(self, contenu=b"donnees", echecs=0, erreur=None):
        self.contenu = contenu
        self.echecs_restants = echecs
        self.erreur = erreur or bex.ReadTimeoutError(endpoint_url="s3://test")
        self.appels = 0

    def get_object(self, Bucket, Key):
        self.appels += 1
        # Un echec consomme une tentative : le corps echoue une seule fois,
        # ce qui oblige l'appelant a refaire un GET complet.
        if self.echecs_restants > 0:
            self.echecs_restants -= 1
            return {"Body": FauxCorps(self.contenu, 1, self.erreur)}
        return {"Body": FauxCorps(self.contenu, 0, self.erreur)}


@pytest.fixture
def sans_attente(monkeypatch):
    """Neutralise les pauses entre tentatives pour garder les tests rapides."""
    monkeypatch.setattr(storage.time, "sleep", lambda _: None)


@pytest.fixture
def client(monkeypatch):
    """Installe un faux client et renvoie une fabrique."""
    def installer(**kwargs):
        faux = FauxClient(**kwargs)
        monkeypatch.setattr(storage, "get_client", lambda: faux)
        return faux
    return installer


# ============================================================================
# Reprise
# ============================================================================

def test_lecture_sans_incident(client, sans_attente):
    faux = client(contenu=b"bonjour")
    assert storage._lire_corps("cle") == b"bonjour"
    assert faux.appels == 1


def test_reprise_apres_read_timeout(client, sans_attente):
    """Le cas reel : timeout pendant la lecture du corps."""
    faux = client(contenu=b"bonjour", echecs=2)
    assert storage._lire_corps("cle", verbose=False) == b"bonjour"
    assert faux.appels == 3, "chaque reprise doit REEMETTRE le GET"


@pytest.mark.parametrize("erreur", [
    bex.ReadTimeoutError(endpoint_url="s3://test"),
    bex.ConnectTimeoutError(endpoint_url="s3://test"),
    bex.EndpointConnectionError(endpoint_url="s3://test"),
    bex.ResponseStreamingError(error="coupure"),
])
def test_erreurs_reseau_rejouees(client, sans_attente, erreur):
    faux = client(contenu=b"ok", echecs=1, erreur=erreur)
    assert storage._lire_corps("cle", verbose=False) == b"ok"
    assert faux.appels == 2


def test_abandon_apres_le_nombre_max_de_tentatives(client, sans_attente):
    maxi = storage.S3_PARAMS["max_tentatives"]
    faux = client(echecs=maxi + 5)
    with pytest.raises(RuntimeError, match="abandonnee apres"):
        storage._lire_corps("cle", verbose=False)
    assert faux.appels == maxi


def test_attente_doublee_entre_tentatives(client, monkeypatch):
    """Le backoff doit croitre, pour ne pas marteler un serveur qui peine."""
    attentes = []
    monkeypatch.setattr(storage.time, "sleep", attentes.append)
    client(contenu=b"ok", echecs=3)

    storage._lire_corps("cle", verbose=False)

    assert attentes == [
        storage.S3_PARAMS["attente_initiale"] * 2 ** i
        for i in range(len(attentes))
    ]


# ============================================================================
# Ce qui ne doit PAS etre rejoue
# ============================================================================

def _erreur_client(code, statut):
    return bex.ClientError(
        {"Error": {"Code": code},
         "ResponseMetadata": {"HTTPStatusCode": statut}},
        "GetObject",
    )


def test_cle_absente_remonte_immediatement(client, sans_attente):
    """
    Rejouer un 404 cinq fois ne ferait que retarder un message exact.
    """
    faux = client(erreur=_erreur_client("NoSuchKey", 404), echecs=1)
    with pytest.raises(bex.ClientError):
        storage._lire_corps("cle", verbose=False)
    assert faux.appels == 1


def test_refus_d_authentification_remonte_immediatement(client, sans_attente):
    faux = client(erreur=_erreur_client("AccessDenied", 403), echecs=1)
    with pytest.raises(bex.ClientError):
        storage._lire_corps("cle", verbose=False)
    assert faux.appels == 1


@pytest.mark.parametrize("code,statut", [
    ("InternalError", 500),
    ("ServiceUnavailable", 503),
    ("SlowDown", 503),
    ("UnCodeInconnu", 500),   # tout 5xx est un incident serveur
])
def test_incidents_serveur_rejoues(client, sans_attente, code, statut):
    faux = client(contenu=b"ok", erreur=_erreur_client(code, statut), echecs=1)
    assert storage._lire_corps("cle", verbose=False) == b"ok"
    assert faux.appels == 2


# ============================================================================
# Les helpers passent bien par la reprise
# ============================================================================

def test_get_text_reprend(client, sans_attente):
    client(contenu="été".encode("utf-8"), echecs=1)
    assert storage.get_text("cle") == "été"


def test_get_json_reprend(client, sans_attente):
    client(contenu=b'{"a": 1}', echecs=1)
    assert storage.get_json("cle") == {"a": 1}


def test_get_parquet_reprend(client, sans_attente):
    import io

    import pandas as pd

    tampon = io.BytesIO()
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(tampon)
    client(contenu=tampon.getvalue(), echecs=1)

    df = storage.get_parquet("cle")
    assert list(df["x"]) == [1, 2, 3]


def test_get_bytes_reprend(client, sans_attente):
    """
    Le chemin du bundle de serving, 269 Mo, donc le plus expose aux
    coupures. Un demarrage de l'API a echoue ici sur un `IncompleteRead`
    a 227 Mo sur 282, cette fonction ayant ete oubliee lors du cablage de
    la reprise.
    """
    faux = client(contenu=b"bundle", echecs=2)
    assert storage.get_bytes("cle") == b"bundle"
    assert faux.appels == 3


# ============================================================================
# Garde-fou : aucun lecteur ne doit court-circuiter la reprise
# ============================================================================

def test_aucun_lecteur_public_n_appelle_get_object_en_direct():
    """
    `get_bytes` avait ete oublie lors du cablage de la reprise, et le defaut
    ne s'est vu qu'en production, sur le plus gros objet du bucket. Ce test
    verifie la propriete au niveau du MODULE plutot que fonction par
    fonction : un futur `get_image` ou `get_npz` ecrit sur le meme moule
    serait attrape ici sans que personne pense a ajouter son test.

    `_lire_corps` est la seule fonction autorisee a appeler `get_object`.
    """
    import inspect

    source = inspect.getsource(storage)
    lignes = source.splitlines()

    # Bornes de `_lire_corps`, seule exception legitime.
    debut = next(i for i, l in enumerate(lignes) if l.startswith("def _lire_corps"))
    fin = next(
        (i for i in range(debut + 1, len(lignes))
         if lignes[i].startswith("def ") or lignes[i].startswith("@")),
        len(lignes),
    )

    fautives = [
        (i + 1, l.strip())
        for i, l in enumerate(lignes)
        if "get_object(" in l and not (debut <= i < fin)
    ]
    assert not fautives, (
        "ces lignes appellent get_object sans passer par _lire_corps, "
        f"donc sans reprise reseau : {fautives}"
    )
