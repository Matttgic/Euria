"""Principale -> secours -> dernière valeur connue. Ne lève jamais d'exception."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from . import cache
from .http import SourceError

log = logging.getLogger(__name__)


@dataclass
class Provider:
    name: str  # nom affiché (« football-data.org »)
    fetch: Callable[[], Any]  # lève SourceError si la donnée est indisponible
    attribution: str | None = None  # mention exigée par la licence


@dataclass
class Result:
    data: Any
    source: str | None
    updated_at: str | None  # heure de récupération (ISO 8601, UTC)
    stale: bool = False  # True = dernière valeur connue, sources en panne
    message: str | None = None
    attribution: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.data is not None

    def meta(self) -> dict:
        return {
            "source": self.source,
            "updated_at": self.updated_at,
            "stale": self.stale,
            "message": self.message,
            "attribution": self.attribution,
            "errors": self.errors,  # pourquoi chaque source écartée a échoué (clé absente, délai, quota…)
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_empty(data: Any) -> bool:
    return data is None or (isinstance(data, (list, dict)) and len(data) == 0)


def fetch(key: str, ttl_s: float, providers: Sequence[Provider]) -> Result:
    attributions = {p.name: p.attribution for p in providers}

    cached = cache.read(key)
    if cached and cached.age_s < ttl_s:
        return Result(cached.data, cached.source, cached.saved_at_iso, attribution=attributions.get(cached.source))

    errors: list[str] = []
    empty_success: Result | None = None
    for provider in providers:
        try:
            data = provider.fetch()
        except SourceError as exc:
            errors.append(str(exc))
            log.warning("Source en échec pour %s : %s", key, exc)
            continue
        except Exception as exc:  # un bug de parsing ne doit jamais casser le bot ni l'API
            errors.append(f"{provider.name} : erreur inattendue ({exc.__class__.__name__}: {exc})")
            log.exception("Erreur inattendue pour %s via %s", key, provider.name)
            continue
        if _is_empty(data):
            # Réponse valide mais vide (ex. pas de match à venir) : on tente quand même la suivante.
            empty_success = empty_success or Result(data, provider.name, _now_iso(), attribution=provider.attribution)
            continue
        entry = cache.write(key, data, provider.name)
        return Result(data, provider.name, entry.saved_at_iso, attribution=provider.attribution, errors=errors)

    if empty_success is not None:
        # Une réponse vide valide (aucune actu, pas de match à venir) est mise en cache elle aussi :
        # sinon la source serait rappelée à chaque requête et son quota vite épuisé.
        entry = cache.write(key, empty_success.data, empty_success.source)
        empty_success.updated_at = entry.saved_at_iso
        empty_success.errors = errors
        return empty_success

    if cached is not None:
        return Result(
            cached.data,
            cached.source,
            cached.saved_at_iso,
            stale=True,
            message=f"Sources indisponibles : dernière valeur connue ({cached.source}, {cached.saved_at_iso}).",
            attribution=attributions.get(cached.source),
            errors=errors,
        )
    return Result(
        None,
        None,
        None,
        stale=True,
        message="Donnée indisponible : toutes les sources sont en panne et aucune valeur n'a encore été enregistrée.",
        errors=errors,
    )
