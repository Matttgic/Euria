from euria import cache
from euria.fallback import Provider, fetch
from euria.http import SourceError


def ok(value):
    return lambda: value


def broken(source="X"):
    def _fail():
        raise SourceError(source, "en panne")

    return _fail


def test_principal_used_and_cached():
    calls = []

    def principal():
        calls.append(1)
        return [1, 2]

    first = fetch("k", 3600, [Provider("A", principal, "Mention A"), Provider("B", ok([9]))])
    assert (first.data, first.source, first.stale, first.attribution) == ([1, 2], "A", False, "Mention A")
    second = fetch("k", 3600, [Provider("A", principal), Provider("B", ok([9]))])
    assert second.data == [1, 2] and len(calls) == 1  # servi par le cache


def test_secours_when_principal_fails():
    result = fetch("k", 3600, [Provider("A", broken("A")), Provider("B", ok([9]), "Mention B")])
    assert (result.data, result.source, result.attribution) == ([9], "B", "Mention B")
    assert result.errors == ["A : en panne"]


def test_last_known_value_when_everything_fails():
    cache.write("k", [7], "A")
    result = fetch("k", 0, [Provider("A", broken("A")), Provider("B", broken("B"))])
    assert result.data == [7] and result.stale is True and result.updated_at
    assert "dernière valeur connue" in result.message


def test_clear_message_when_nothing_known():
    result = fetch("k", 3600, [Provider("A", broken("A"))])
    assert result.data is None and result.stale is True and "indisponible" in result.message


def test_unexpected_parser_bug_does_not_crash():
    def buggy():
        raise KeyError("champ disparu")

    result = fetch("k", 3600, [Provider("A", buggy), Provider("B", ok([1]))])
    assert result.source == "B" and "erreur inattendue" in result.errors[0]


def test_empty_answer_tries_next_source():
    result = fetch("k", 3600, [Provider("A", ok([])), Provider("B", ok([3]))])
    assert result.source == "B"
    only_empty = fetch("k2", 3600, [Provider("A", ok([]))])
    assert only_empty.data == [] and only_empty.stale is False


def test_empty_answer_is_cached():
    calls = []

    def empty():
        calls.append(1)
        return []

    fetch("k", 3600, [Provider("A", empty)])
    again = fetch("k", 3600, [Provider("A", empty)])
    assert again.data == [] and len(calls) == 1


def test_prune_removes_old_entries(monkeypatch):
    cache.write("odds:PL", [1], "A")
    cache.write("matches:PL", [1], "A")
    assert cache.prune("odds:", -1) == 1
    assert cache.read("odds:PL") is None and cache.read("matches:PL") is not None
