import pytest

from euria.sources import nominatim

pytestmark = pytest.mark.live


def test_geocode_stadium():
    lat, lon = nominatim.geocode("Stamford Bridge, London")
    assert 51.4 < lat < 51.6 and -0.3 < lon < 0.0
