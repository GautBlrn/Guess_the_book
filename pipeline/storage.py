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
from functools import lru_cache

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

from pipeline.config import BUCKET, ENDPOINT_URL, REGION

load_dotenv()


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
        config=Config(signature_version="s3v4"),
    )


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
    r = get_client().get_object(Bucket=bucket, Key=key)
    return r["Body"].read().decode(encoding)


def get_json(key, bucket=BUCKET):
    """Telecharge un JSON depuis S3 et le parse."""
    return json.loads(get_text(key, bucket=bucket))


def get_parquet(key, bucket=BUCKET):
    """Telecharge un Parquet depuis S3 et renvoie un DataFrame."""
    r = get_client().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(r["Body"].read()))


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
    """Telecharge bytes bruts."""
    r = get_client().get_object(Bucket=bucket, Key=key)
    return r["Body"].read()