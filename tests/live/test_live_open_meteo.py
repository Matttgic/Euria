from datetime import datetime, timedelta, timezone

import pytest

from euria.sources import open_meteo

pytestmark = pytest.mark.live


def test_forecast_fields_still_exist():
    kickoff = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(days=5)
    weather = open_meteo.fetch_weather(48.8414, 2.2530, kickoff)
    assert weather["temperature_c"] is not None and weather["wind_speed_ms"] is not None
