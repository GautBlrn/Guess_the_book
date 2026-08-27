"""
Configuration pytest : rend `pipeline` importable depuis les tests.

Le projet n'est pas installe en package (pas de setup.py / pyproject), donc
sans ca `from pipeline.evaluation import ...` echoue des que pytest est
lance depuis un autre repertoire que la racine.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))
