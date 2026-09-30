"""Envoi des messages Telegram (jeton côté serveur uniquement, jamais dans le dépôt)."""

from __future__ import annotations

import logging

import requests

from . import config

log = logging.getLogger(__name__)
_MARKDOWN_SPECIAL = ("\\", "_", "*", "`", "[")


def escape(text: str) -> str:
    """Échappe un texte extérieur (titre d'actu, nom d'équipe) pour le Markdown de Telegram :
    un « _ » ou un « * » non fermé ferait refuser tout le message."""
    for char in _MARKDOWN_SPECIAL:
        text = text.replace(char, "\\" + char)
    return text


def send(message: str) -> bool:
    if not (config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID):
        log.info("Telegram non configuré, message non envoyé :\n%s", message)
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=config.HTTP_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        log.warning("Telegram injoignable : %s", exc.__class__.__name__)
        return False
    if response.status_code != 200:
        log.warning("Telegram a refusé le message : HTTP %s", response.status_code)
        return False
    return True
