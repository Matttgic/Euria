"""Point d'entrée de l'API : uvicorn main:app --host 0.0.0.0 --port 8000"""

from euria.api import app

__all__ = ["app"]
