"""Rapprochement des noms d'équipes entre la source principale et le secours des matchs.

Sur une saison complète, chaque équipe de football-data.co.uk doit correspondre à exactement
une équipe de football-data.org (5 appels football-data.org)."""

import pytest

from euria import config
from euria.config import LEAGUES
from euria.sources import betbetter, football_charts, football_data_couk, football_data_org
from euria.teams import best_match
from tests.live.conftest import current_season, require_key

pytestmark = pytest.mark.live


@pytest.mark.parametrize("code", list(LEAGUES))
def test_names_match_between_sources(code):
    require_key(config.FOOTBALL_DATA_TOKEN, "FOOTBALL_DATA_TOKEN")
    season = current_season() - 1  # saison terminée : 20 équipes présentes des deux côtés
    league = LEAGUES[code]
    fd_names = {m[k] for m in football_data_org.fetch_matches(league, season) for k in ("home", "away")}
    couk_names = {m[k] for m in football_data_couk.fetch_matches(league, season) for k in ("home", "away")}
    mapping = {name: best_match(name, fd_names) for name in sorted(couk_names)}
    unmatched = [n for n, m in mapping.items() if m is None]
    matched = [m for m in mapping.values() if m]
    duplicates = {m for m in matched if matched.count(m) > 1}
    assert not unmatched and not duplicates, (
        f"{code} — sans correspondance : {unmatched} ; doublons : {sorted(duplicates)} ; "
        f"noms football-data.org : {sorted(fd_names)}"
    )


def _assert_all_match(code, names, reference):
    mapping = {name: best_match(name, reference) for name in sorted(names)}
    unmatched = [n for n, m in mapping.items() if m is None]
    matched = [m for m in mapping.values() if m]
    duplicates = {m for m in matched if matched.count(m) > 1}
    assert not unmatched and not duplicates, (
        f"{code} — sans correspondance : {unmatched} ; doublons : {sorted(duplicates)} ; référence : {sorted(reference)}"
    )


@pytest.mark.parametrize("code", list(LEAGUES))
def test_enrichment_names_match_results(code):
    """Les équipes de Football Charts et de Bet Better doivent se rapprocher des résultats (sans clé)."""
    league = LEAGUES[code]
    reference = {m[k] for m in football_data_couk.fetch_matches(league, current_season()) for k in ("home", "away")}
    _assert_all_match(f"{code} Football Charts", {r["team"] for r in football_charts.fetch_table(league)}, reference)
    _assert_all_match(f"{code} Bet Better", {o[k] for o in betbetter.fetch_picks(league) for k in ("home", "away")}, reference)
