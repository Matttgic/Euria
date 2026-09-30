import json
from pathlib import Path

import pytest

from euria import config

SAMPLES = Path(__file__).parent / "samples"


def load_sample(name: str):
    path = SAMPLES / name
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if name.endswith(".json") else text


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Chaque test a son propre dossier de données : le cache et le CSV réels ne sont jamais touchés."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "BETS_FILE", tmp_path / "suivi_paris.csv")
    monkeypatch.setattr(config, "PREDICTIONS_FILE", tmp_path / "predictions.csv")
    return tmp_path
