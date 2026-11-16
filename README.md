# Promos et milieux

Page de calculs pour les paris sportifs, **sans aucun pronostic** : uniquement des maths sur des cotes que tu saisis.

1. **Freebets et bonus couverts** : convertir un freebet ou un « premier pari remboursé si perdant » en gain garanti, en
   couvrant les autres issues chez un autre bookmaker (mises de couverture, gain, taux de conversion).
2. **Milieux** : « Plus de » une ligne chez un bookmaker, « Moins de » une ligne plus haute chez un autre. Résultat de
   chaque cas, fréquence de milieu à partir de laquelle c'est rentable, et probabilité du milieu d'après Pinnacle
   (deux lignes, ou une ligne et une loi normale).
3. **Promos « remboursé si… » et « payé dès 2 buts d'avance »** (football) : valeur exacte de la promo avec les
   probabilités justes du 1N2 de Pinnacle ; défaite d'un but, 0-0 et avance de deux buts par un modèle de Poisson ajusté
   sur ces cotes (la valeur « 2 buts d'avance » est un minimum : le modèle sous-estime les remontées).

Les surebets et les cotes boostées sont suivis en direct par Cote Value (onglet Boosts).

## Fichiers

- `site/` : la page publiée (`index.html`, `style.css`, `page.js`), calculs purs dans `site/calculs.js`.
- `tests/calculs.test.mjs` : `node --test tests/*.test.mjs` (Node 20 ou plus, aucune dépendance).
- `.github/workflows/pages.yml` : tests puis publication sur GitHub Pages à chaque poussée sur `main`
  (une fois : Settings → Pages → Source : « GitHub Actions »).

Tout est calculé dans le navigateur ; rien n'est envoyé. Jeu responsable : Joueurs Info Service, 09 74 75 13 13.
