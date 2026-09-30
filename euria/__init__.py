"""Euria : détection de value bets sur les 5 grands championnats européens.

Organisation du paquet :
- config      : réglages (variables d'environnement), championnats, durées de cache
- http        : appels HTTP communs (délai max 8 s, User-Agent, erreurs propres)
- cache       : cache fichier avec durée de vie + dernière valeur connue
- fallback    : principale -> secours -> dernière valeur connue
- sources/    : un module par API externe, qui renvoie toujours le même format
- services    : données prêtes à l'emploi (matchs, cotes, météo, stades)
- stats       : classement et variables du modèle, calculés à partir des scores
- predictor   : modèle unique (models/model_sklearn.pkl)
- bot / api   : les deux points d'entrée (GitHub Actions et FastAPI)
"""

__version__ = "2.0.0"
