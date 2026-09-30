import pytest

from euria.config import LEAGUES
from euria.http import get_json
from euria.sources import betbetter

pytestmark = pytest.mark.live


def test_picks_fields_still_exist():
    payload = get_json("Bet Better", f"{betbetter.BASE_URL}/epl/picks", params={"format": "json"})
    head_to_head = [p for p in payload["picks"] if p.get("market") == "Head to Head"]
    if not head_to_head:
        pytest.skip("aucun pronostic 1N2 publié en ce moment (trêve ou intersaison)")
    pick = head_to_head[0]
    for field in ("game", "gameTimeUtc", "selection", "winProbabilityPct"):
        assert field in pick, field
    assert " @ " in pick["game"]


@pytest.mark.parametrize("code", list(LEAGUES))
def test_every_league_answers(code):
    assert isinstance(betbetter.fetch_picks(LEAGUES[code]), list)
