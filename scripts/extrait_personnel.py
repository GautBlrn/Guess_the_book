"""Identifie l'extrait personnel et ecrit la mesure brute sur disque.

Repond au paragraphe 6.1 du sujet : titre identifie, score de confiance,
modele retenu. Ne calcule rien lui-meme, il interroge l'API et enregistre
telle quelle la reponse, pour que la synthese redigee soit tracable.

Prerequis : l'API tourne (python api.py), et mon_extrait.txt existe.

Usage :
    python scripts/extrait_personnel.py
    python scripts/extrait_personnel.py --extrait chemin/vers/extrait.txt --top-k 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import requests

SERVEUR = "http://localhost:5000"
DELAI = 120  # le premier appel charge spaCy, environ 2 s, on garde de la marge


def lire_extrait(chemin: pathlib.Path) -> str:
    if not chemin.exists():
        sys.exit(f"Extrait introuvable : {chemin}")
    texte = chemin.read_text(encoding="utf-8").strip()
    if not texte:
        sys.exit(f"Extrait vide : {chemin}")
    return texte


def index_des_livres(serveur: str) -> dict[str, dict[str, Any]]:
    reponse = requests.get(f"{serveur}/api/livres", timeout=DELAI)
    reponse.raise_for_status()
    return {livre["livre_slug"]: livre for livre in reponse.json()["livres"]}


def identifier(serveur: str, extrait: str, top_k: int) -> dict[str, Any]:
    reponse = requests.post(
        f"{serveur}/api/identifier",
        json={"extrait": extrait, "top_k": top_k},
        timeout=DELAI,
    )
    reponse.raise_for_status()
    return reponse.json()


def resume(serveur: str, slug: str) -> str | None:
    reponse = requests.get(f"{serveur}/api/resume/{slug}", timeout=DELAI)
    if reponse.status_code != 200:
        return None
    return reponse.json().get("resume")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--extrait", type=pathlib.Path, default=pathlib.Path("mon_extrait.txt"))
    parseur.add_argument("--serveur", default=SERVEUR)
    parseur.add_argument("--top-k", type=int, default=5)
    parseur.add_argument("--sortie", type=pathlib.Path,
                         default=pathlib.Path("resultats/extrait_personnel.json"))
    args = parseur.parse_args()

    extrait = lire_extrait(args.extrait)
    mots = len(extrait.split())

    try:
        sante = requests.get(f"{args.serveur}/health", timeout=DELAI).json()
    except requests.RequestException as erreur:
        sys.exit(f"API injoignable sur {args.serveur} : {erreur}\nLance 'python api.py' d'abord.")
    if not sante.get("ready"):
        sys.exit(f"API pas prete : {sante}")

    livres = index_des_livres(args.serveur)
    reponse = identifier(args.serveur, extrait, args.top_k)
    resultats = reponse["resultats"]

    for ligne in resultats:
        meta = livres.get(ligne["livre_slug"], {})
        ligne["titre"] = meta.get("livre")
        ligne["auteur"] = meta.get("auteur")
        ligne["genre"] = meta.get("genre")

    premier = resultats[0]
    second = resultats[1] if len(resultats) > 1 else None
    ecart = (premier["score"] / second["score"]) if second and second["score"] else None

    mesure = {
        "extrait": {
            "fichier": str(args.extrait),
            "n_mots": mots,
            "n_caracteres": len(extrait),
            "texte": extrait,
        },
        "corpus": {"n_livres": sante.get("n_livres"), "n_resumes": sante.get("n_resumes")},
        "modele_retenu": reponse["config"],
        "duree_ms": reponse["duree_ms"],
        "resultats": resultats,
        "ecart_rang1_rang2": ecart,
    }

    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(mesure, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Extrait      : {mots} mots, {len(extrait)} caracteres")
    print(f"Corpus servi : {sante.get('n_livres')} livres")
    print(f"Modele retenu: {reponse['config']}")
    print(f"Latence      : {reponse['duree_ms']} ms")
    print()
    for ligne in resultats:
        print(f"  {ligne['rang']}. {ligne['score']:.4f}  {ligne.get('titre')} "
              f"({ligne.get('auteur')})  [{ligne['livre_slug']}]")
    if ecart:
        print(f"\nEcart rang 1 / rang 2 : x{ecart:.1f}")

    texte_resume = resume(args.serveur, premier["livre_slug"])
    if texte_resume:
        chemin = args.sortie.with_name("extrait_personnel_resume.txt")
        chemin.write_text(texte_resume, encoding="utf-8")
        print(f"\nResume du livre identifie ecrit dans {chemin}")

    print(f"Mesure brute ecrite dans {args.sortie}")


if __name__ == "__main__":
    main()
