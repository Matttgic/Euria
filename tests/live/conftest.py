"""Tests réels : ils appellent les vraies API. Lancement : pytest -m live
Les sources à clé sont sautées (pas en échec) si la clé est absente de l'environnement."""

import pytest

from euria import config


def require_key(value, name):
    if not value:
        pytest.skip(f"{name} absente : test sauté")


def current_season():
    return config.season_for()
