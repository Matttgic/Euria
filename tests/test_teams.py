import pytest

from euria.teams import best_match, same_team

# Paires réellement observées entre les sources (football-data.co.uk, Parlay, OpenLigaDB, Wikidata).
PAIRS = [
    ("Man United", "Manchester United F.C."),
    ("Man City", "Manchester City F.C."),
    ("Nott'm Forest", "Nottingham Forest"),
    ("Leeds", "Leeds United"),
    ("Paris SG", "Paris Saint-Germain FC"),
    ("Paris FC", "Paris FC"),
    ("Ein Frankfurt", "Eintracht Frankfurt"),
    ("M'gladbach", "Borussia Mönchengladbach"),
    ("FC Koln", "1. FC Köln"),
    ("Bayern Munich", "FC Bayern München"),
    ("Inter", "Inter Milan"),
    ("Milan", "AC Milan"),
    ("Ath Madrid", "Atlético Madrid"),
    ("Ath Bilbao", "Athletic Club"),
    ("Sociedad", "Real Sociedad"),
    ("Lens", "R.C. Lens"),
    ("Brighton", "Brighton & Hove Albion F.C."),
    # Noms courts réels de football-data.org face à football-data.co.uk et à l'ancien suivi des paris
    ("Barcelona", "Barça"),
    ("Ath Madrid", "Atleti"),
    ("Atletico Madrid", "Atleti"),
    ("Oviedo", "Real Oviedo"),
    ("Wolves", "Wolverhampton"),
    ("Hamburg", "HSV"),
    ("Hamburger SV", "HSV"),
    ("Werder Bremen", "Bremen"),
    ("Ein Frankfurt", "Frankfurt"),
    ("Eintracht Frankfurt", "Frankfurt"),
    ("Lyon", "Olympique Lyon"),
    ("Borussia Mönchengladbach", "M'gladbach"),
    ("Paris SG", "PSG"),
    ("Nott'm Forest", "Nottingham"),
    # Football Charts et Bet Better
    ("Atl. Madrid", "Ath Madrid"),
    ("Dep. A Coruna", "La Coruna"),
    ("CA Osasuna", "Osasuna"),
]


@pytest.mark.parametrize("a,b", PAIRS)
def test_known_variants_match(a, b):
    assert same_team(a, b)


@pytest.mark.parametrize("a,b", [
    ("Man United", "Manchester City F.C."),
    ("Paris FC", "Paris Saint-Germain FC"),
    ("Milan", "Inter Milan"),
    ("Real Madrid", "Atlético Madrid"),
    ("Barcelona", "RCD Espanyol de Barcelona"),
    ("Oviedo", "Real Madrid"),
    ("Frankfurt", "Union Berlin"),
    ("Atl. Madrid", "Real Madrid"),
])
def test_different_clubs_do_not_match(a, b):
    assert not same_team(a, b)


def test_best_match_refuses_to_guess():
    assert best_match("Chelsea", ["Arsenal F.C.", "Fulham F.C."]) is None
    assert best_match("Milan", ["AC Milan", "Inter Milan"]) == "AC Milan"
