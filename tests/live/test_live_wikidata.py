import pytest

from euria.sources import wikidata
from euria.teams import best_match

pytestmark = pytest.mark.live


def test_venues_still_available():
    venues = wikidata.fetch_all_venues()
    assert len(venues) >= 60  # 100 clubs trouvés le 30/09/2026
    names = [v["team"] for v in venues]
    for club in ("Paris SG", "Man United", "Bayern Munich", "Inter", "Real Madrid"):
        assert best_match(club, names), club
