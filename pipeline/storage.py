"""
Acces S3 (Scaleway Object Storage).

Une seule connexion S3 partagee, expose via la fonction `get_client()`.
Les helpers `get_text/json/parquet` et `put_text/json/parquet` factorisent
les patterns repetes dans tous les notebooks (encode UTF-8, dump JSON,
serialisation Parquet en memoire, etc.).

L'import du module ne fait AUCUN appel reseau. La creation du bucket
est explicite via `ensure_bucket()` (a appeler une fois, ou jamais si
le bucket existe deja).
"""
import io
import json
import os
import time
from functools import lru_cache

import boto3
import botocore.exceptions as bex
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

from pipeline.config import BUCKET, ENDPOINT_URL, REGION, S3_PARAMS

load_dotenv()


# Erreurs reseau passageres : la requete est rejouable telle quelle.
#
# `ReadTimeoutError` merite une mention. Elle n'herite PAS de
# `botocore.exceptions.ConnectionError` mais de `HTTPClientError`, donc les
# listes d'erreurs reseau ecrites d'instinct la manquent. Surtout, elle est
# levee pendant `Body.read()`, c'est-a-dire APRES que botocore a rendu la
# main sur la requete : sa propre couche de reessai ne peut plus rien, le
# flux est mort et il faut reemettre le GET. C'est exactement ce qui a fait
# echouer un chargement de 291 livres sur une seule lecture.
ERREURS_TRANSITOIRES = (
    bex.ConnectionError,        # couvre ConnectTimeoutError, EndpointConnectionError
    bex.ReadTimeoutError,
    bex.ResponseStreamingError,
    bex.IncompleteReadError,
)

# Codes S3 qui signalent un incident cote serveur, donc rejouables.
CODES_TRANSITOIRES = {
    "InternalError", "ServiceUnavailable", "SlowDown",
    "RequestTimeout", "RequestTimeTooSkewed", "503", "500",
}


@lru_cache(maxsize=1)
def get_client():
    """Retourne le client S3 (mis en cache : un seul par processus)."""
    key_id = os.getenv("SCW_ACCESS_KEY")
    secret = os.getenv("SCW_SECRET_KEY")
    if not key_id or not secret:
        raise RuntimeError(
            "Credentials Scaleway manquants. Verifie que le fichier .env "
            "existe et contient SCW_ACCESS_KEY et SCW_SECRET_KEY."
        )
    return boto3.client(
        service_name="s3",
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(
            signature_version="s3v4",
            connect_timeout=S3_PARAMS["connect_timeout"],
            read_timeout=S3_PARAMS["read_timeout"],
            # Reessais de botocore, qui traitent ce qui echoue AVANT que la
            # reponse commence a arriver. Ce qui casse pendant la lecture du
            # corps est repris par `_lire_corps` ci-dessous.
            retries={"max_attempts": S3_PARAMS["max_tentatives"],
                     "mode": "standard"},
            # Le pool doit pouvoir servir tous les telechargements
            # simultanes, sinon les threads se mettent en file d'attente
            # sur les connexions et la parallelisation ne sert a rien.
            max_pool_connections=max(10, S3_PARAMS["parallelisme"]),
        ),
    )


def _est_transitoire(erreur):
    """Vrai si l'erreur vaut la peine d'etre rejouee."""
    if isinstance(erreur, ERREURS_TRANSITOIRES):
        return True
    if isinstance(erreur, bex.ClientError):
        reponse = erreur.response or {}
        code = str(reponse.get("Error", {}).get("Code", ""))
        statut = reponse.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return code in CODES_TRANSITOIRES or int(statut or 0) >= 500
    return False


def _lire_corps(key, bucket=BUCKET, verbose=True):
    """
    GET + lecture complete du corps, avec reprise sur erreur passagere.

    Reemet la requete entiere a chaque tentative, et pas seulement la
    lecture : une fois le flux tombe en timeout il n'est plus lisible, la
    reponse doit etre redemandee.

    L'attente double a chaque essai (1 s, 2 s, 4 s...). Une erreur qui n'est
    pas passagere, typiquement une cle absente ou un refus d'authentification,
    remonte immediatement : la rejouer cinq fois ne ferait que retarder un
    message d'erreur exact.
    """
    attente = S3_PARAMS["attente_initiale"]
    derniere = None

    for tentative in range(1, S3_PARAMS["max_tentatives"] + 1):
        try:
            reponse = get_client().get_object(Bucket=bucket, Key=key)
            return reponse["Body"].read()
        except Exception as erreur:
            if not _est_transitoire(erreur):
                raise
            derniere = erreur
            if tentative == S3_PARAMS["max_tentatives"]:
                break
            if verbose:
                print(f"    {type(erreur).__name__} sur {key}, "
                      f"nouvel essai dans {attente:.0f}s "
                      f"({tentative}/{S3_PARAMS['max_tentatives'] - 1})")
            time.sleep(attente)
            attente *= 2

    raise RuntimeError(
        f"Lecture de {key!r} abandonnee apres "
        f"{S3_PARAMS['max_tentatives']} tentatives : {derniere}"
    ) from derniere


def ensure_bucket(bucket=BUCKET):
    """Cree le bucket s'il n'existe pas. A appeler explicitement."""
    s3 = get_client()
    try:
        s3.create_bucket(Bucket=bucket)
        print(f"Bucket cree : {bucket}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass


# --- Lecture ---

def get_text(key, bucket=BUCKET, encoding="utf-8"):
    """Telecharge un fichier texte depuis S3."""
    return _lire_corps(key, bucket=bucket).decode(encoding)


def get_json(key, bucket=BUCKET):
    """Telecharge un JSON depuis S3 et le parse."""
    return json.loads(get_text(key, bucket=bucket))


def get_parquet(key, bucket=BUCKET):
    """Telecharge un Parquet depuis S3 et renvoie un DataFrame."""
    return pd.read_parquet(io.BytesIO(_lire_corps(key, bucket=bucket)))


def list_objects(prefix, bucket=BUCKET, suffix=None):
    """Liste les objets sous un prefixe, paginee. Filtre optionnel par suffixe."""
    s3 = get_client()
    paginator = s3.get_paginator("list_objects_v2")
    objets = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if suffix is None or obj["Key"].endswith(suffix):
                objets.append(obj)
    return objets


# --- Suppression ---

def delete_object(key, bucket=BUCKET):
    """
    Supprime un objet S3. IRREVERSIBLE si le versioning est desactive.

    Volontairement unitaire : pas de suppression par prefixe ni par lot.
    Un appelant qui veut effacer plusieurs cles doit les enumerer, donc
    les avoir construites explicitement. Une API `delete_prefix` rendrait
    trop facile d'effacer tout `summaries/` sur une faute de frappe.
    """
    get_client().delete_object(Bucket=bucket, Key=key)


# --- Ecriture ---

def put_text(key, contenu, bucket=BUCKET, content_type="text/plain; charset=utf-8",
             metadata=None):
    """Upload une chaine UTF-8 vers S3."""
    get_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=contenu.encode("utf-8"),
        ContentType=content_type,
        Metadata=metadata or {},
    )


def put_json(key, data, bucket=BUCKET, metadata=None):
    """Serialise un objet en JSON et upload."""
    contenu = json.dumps(data, ensure_ascii=False)
    put_text(
        key, contenu, bucket=bucket,
        content_type="application/json; charset=utf-8",
        metadata=metadata,
    )


def put_parquet(key, df, bucket=BUCKET, metadata=None, compression="snappy"):
    """Serialise un DataFrame en Parquet (memoire) puis upload."""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression=compression, index=False)
    buf.seek(0)
    get_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/x-parquet",
        Metadata=metadata or {},
    )


def put_bytes(key, data, bucket=BUCKET, content_type="application/octet-stream",
              metadata=None):
    """Upload bytes bruts (pour pickle ou autres binaires)."""
    get_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        Metadata=metadata or {},
    )


def get_bytes(key, bucket=BUCKET):
    """
    Telecharge bytes bruts, avec reprise sur erreur passagere.

    C'est par ici que passe le bundle de serving, et c'est l'objet le plus
    gros du bucket : 269 Mo, contre quelques Mo pour un Parquet annote. Plus
    la lecture est longue, plus elle a de chances d'etre coupee en route, et
    une lecture coupee ne peut pas etre reprise la ou elle en etait -- il
    faut refaire le GET.

    Un demarrage de l'API a echoue sur ce chemin, sur un `IncompleteRead`
    apres 227 Mo lus sur 282 attendus, parce que cette fonction faisait
    encore son `get_object` en direct alors que les autres lecteurs etaient
    deja passes par `_lire_corps`.
    """
    return _lire_corps(key, bucket=bucket)