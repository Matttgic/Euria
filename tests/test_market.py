"""Retrait de la marge (Shin, proportionnel) et CLV."""

import pytest

from euria import market


def test_shin_matches_reference_package():
    # Valeurs du paquet Python `shin` (mberk/shin) sur les mêmes cotes.
    assert market.implied_shin([1.3, 6, 8.5]) == pytest.approx([0.74594, 0.15112, 0.10294], abs=1e-5)
    assert market.implied_shin([1.05, 15, 30]) == pytest.approx([0.92726, 0.05200, 0.02074], abs=1e-5)


def test_shin_takes_more_margin_from_longshots_than_proportional():
    odds = [1.3, 6, 8.5]
    shin = market.implied_shin(odds)
    prop = market.implied_multiplicative(odds)
    assert sum(shin) == pytest.approx(1.0) and sum(prop) == pytest.approx(1.0)
    assert shin[0] > prop[0] and shin[2] < prop[2]


def test_symmetric_market_and_invalid_odds():
    assert market.implied_shin([1.9, 1.9]) == pytest.approx([0.5, 0.5])
    assert market.implied_shin([1.0, 3.0, 4.0]) is None
    assert market.implied_multiplicative([2.0]) is None


def test_clv():
    # Pris à 2,20 ; clôture juste à 50 % (cote 2,00) : on a battu la clôture de 10 %.
    assert market.clv(2.2, 0.5) == pytest.approx(0.10)
    assert market.clv(1.8, 0.5) == pytest.approx(-0.10)
