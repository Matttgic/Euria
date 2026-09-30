"""Cache sur disque : chaque entrée garde la donnée, sa source et son heure de récupération.

Une entrée périmée n'est jamais effacée par la lecture : elle sert de « dernière valeur connue »
quand toutes les sources sont en panne."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


@dataclass
class Entry:
    data: Any
    source: str
    saved_at: float  # horodatage Unix

    @property
    def age_s(self) -> float:
        return time.time() - self.saved_at

    @property
    def saved_at_iso(self) -> str:
        return datetime.fromtimestamp(self.saved_at, timezone.utc).isoformat(timespec="seconds")


def _path(key: str, cache_dir: Path | None = None) -> Path:
    base = cache_dir or config.CACHE_DIR
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:60]
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return base / f"{readable}-{digest}.json"


def read(key: str) -> Entry | None:
    path = _path(key)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Entry(raw["data"], raw["source"], float(raw["saved_at"]))
    except (OSError, ValueError, KeyError):
        return None


def write(key: str, data: Any, source: str) -> Entry:
    entry = Entry(data, source, time.time())
    path = _path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"key": key, "source": source, "saved_at": entry.saved_at, "data": data}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)  # écriture atomique : jamais de fichier à moitié écrit
    return entry


def prune(prefix: str, max_age_s: float) -> int:
    """Supprime les entrées d'un type plus vieilles que max_age_s (ex. cotes > 90 jours)."""
    removed = 0
    if not config.CACHE_DIR.exists():
        return 0
    for path in config.CACHE_DIR.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(raw.get("key", "")).startswith(prefix) and time.time() - float(raw.get("saved_at", 0)) > max_age_s:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
