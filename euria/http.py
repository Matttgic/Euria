"""Appels HTTP communs à toutes les sources : délai maximum, User-Agent, erreurs lisibles."""

from __future__ import annotations

from typing import Any

import requests

from . import config


class SourceError(Exception):
    """Une source n'a pas pu fournir la donnée (réseau, clé, quota, format inattendu)."""

    def __init__(self, source: str, message: str):
        super().__init__(f"{source} : {message}")
        self.source = source
        self.message = message


_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT})

_STATUS_MESSAGES = {
    401: "clé absente ou refusée",
    403: "accès refusé (clé, offre ou protection du site)",
    404: "ressource introuvable",
    429: "limite d'appels atteinte",
}


def get(source: str, url: str, *, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    """GET avec délai maximum de config.HTTP_TIMEOUT_S secondes. Lève SourceError en cas de problème."""
    try:
        response = _session.get(url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT_S)
    except requests.Timeout:
        raise SourceError(source, f"pas de réponse en {config.HTTP_TIMEOUT_S} s") from None
    except requests.RequestException as exc:
        raise SourceError(source, f"erreur réseau ({exc.__class__.__name__})") from None
    if response.status_code >= 400:
        reason = _STATUS_MESSAGES.get(response.status_code, "erreur du serveur")
        raise SourceError(source, f"HTTP {response.status_code}, {reason}")
    return response


def get_json(source: str, url: str, **kwargs: Any) -> Any:
    response = get(source, url, **kwargs)
    try:
        return response.json()
    except ValueError:
        raise SourceError(source, "réponse qui n'est pas du JSON") from None


def get_text(source: str, url: str, **kwargs: Any) -> str:
    response = get(source, url, **kwargs)
    response.encoding = response.encoding or "utf-8"
    return response.text


def require(source: str, condition: bool, message: str) -> None:
    """Vérifie un point du format de réponse ; sinon la source est considérée comme en panne."""
    if not condition:
        raise SourceError(source, f"format de réponse inattendu ({message})")
