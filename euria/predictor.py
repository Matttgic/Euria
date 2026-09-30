"""Modèle unique du projet (models/model_sklearn.pkl, XGBoost à 12 variables) et calcul de la value."""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache

import numpy as np

from . import config

log = logging.getLogger(__name__)
OUTCOMES = ("Home", "Draw", "Away")


@lru_cache(maxsize=1)
def load_model():
    import joblib

    with warnings.catch_warnings():  # avertissement de version XGBoost sans conséquence sur les prédictions
        warnings.simplefilter("ignore")
        model = joblib.load(config.MODEL_FILE)
    expected = getattr(model, "n_features_in_", 12)
    if expected != 12:
        raise RuntimeError(f"Le modèle attend {expected} variables, 12 prévues")
    return model


def predict(vector: list[float]) -> dict[str, float]:
    """Probabilités {"Home", "Draw", "Away"} dans l'ordre de classes du modèle (config.CLASS_LABELS)."""
    model = load_model()
    probs = model.predict_proba(np.array([vector], dtype=float))[0]
    by_label = {config.CLASS_LABELS[int(cls)]: float(p) for cls, p in zip(model.classes_, probs)}
    return {outcome: round(by_label[outcome], 4) for outcome in OUTCOMES}


def choose_quote(quotes: list[dict]) -> dict | None:
    """Cote retenue pour la value : le premier bookmaker disponible dans config.ODDS_BOOKMAKERS."""
    by_book = {q["bookmaker"]: q for q in quotes}
    for book in config.ODDS_BOOKMAKERS:
        if book in by_book:
            return by_book[book]
    return quotes[0] if quotes else None


def value_bets(probabilities: dict[str, float], quote: dict, threshold: float | None = None) -> list[dict]:
    threshold = config.VALUE_THRESHOLD if threshold is None else threshold
    prices = {"Home": quote["home_odds"], "Draw": quote["draw_odds"], "Away": quote["away_odds"]}
    out = []
    for outcome in OUTCOMES:
        value = probabilities[outcome] * prices[outcome]
        if value > threshold:
            out.append({"outcome": outcome, "probability": probabilities[outcome], "odds": prices[outcome], "value": round(value, 3)})
    return sorted(out, key=lambda b: -b["value"])
