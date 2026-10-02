"""Probabilités du marché : retrait de la marge du bookmaker et CLV (closing line value).

Méthode de Shin (1993) : suppose une part z de parieurs « initiés » ; elle retire plus de marge sur
les outsiders que sur les favoris, ce qui corrige le biais favori-outsider. Référence : paquet
Python `shin` (github.com/mberk/shin), même équation résolue ici par dichotomie sur z."""

from __future__ import annotations

from math import sqrt


def _valid(odds: list[float]) -> bool:
    return len(odds) >= 2 and all(o and o > 1.0 for o in odds)


def implied_multiplicative(odds: list[float]) -> list[float] | None:
    """Probabilités proportionnelles aux 1/cote (marge répartie au prorata)."""
    if not _valid(odds):
        return None
    inverse = [1.0 / o for o in odds]
    total = sum(inverse)
    return [p / total for p in inverse]


def _shin_probs(inverse: list[float], total: float, z: float) -> list[float]:
    return [(sqrt(z * z + 4.0 * (1.0 - z) * q * q / total) - z) / (2.0 * (1.0 - z)) for q in inverse]


def implied_shin(odds: list[float]) -> list[float] | None:
    """Probabilités sans marge selon Shin. None si les cotes sont inexploitables."""
    if not _valid(odds):
        return None
    inverse = [1.0 / o for o in odds]
    total = sum(inverse)
    if total <= 1.0:  # pas de marge (ou cotes « meilleur prix » combinées) : rien à retirer
        return [q / total for q in inverse]
    # La somme des probabilités décroît avec z : on cherche z tel qu'elle vaille 1.
    low, high = 0.0, 0.5
    for _ in range(100):
        z = (low + high) / 2.0
        if sum(_shin_probs(inverse, total, z)) > 1.0:
            low = z
        else:
            high = z
    probs = _shin_probs(inverse, total, (low + high) / 2.0)
    norm = sum(probs)
    return [p / norm for p in probs]


def clv(taken_odds: float, closing_probability: float) -> float:
    """Espérance du pari au prix de clôture : cote prise × probabilité juste de clôture − 1.

    > 0 : la cote prise battait la clôture (signe d'un vrai avantage sur un grand nombre de paris)."""
    return taken_odds * closing_probability - 1.0
