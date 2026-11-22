# Euria

Paris sportifs **sans aucun pronostic**, à l'adresse https://matttgic.github.io/Euria/.

## En direct

Lu à chaque ouverture puis toutes les 3 minutes dans `euria.json`, publié par Cote Value à chaque relevé des cotes
françaises (toutes les 5 à 10 minutes, `cotes/euria.py` du dépôt cotes-value) :

- **Milieux en value** : « Plus de » une ligne chez un bookmaker et « Moins de » une ligne plus haute chez un autre (ou
  deux handicaps). Seuls ceux dont l'espérance est positive d'après les cotes justes de Pinnacle aux deux lignes sont
  listés (chance estimée par un modèle, et signalée, quand Pinnacle n'a qu'une des deux lignes), avec les mises pour la somme saisie (réduites si la perte hors du milieu dépasserait la « perte max » saisie), le résultat dans et hors du milieu, la chance du milieu. Alerte Pushover :
  case « Milieux en value » dans Cote Value › Notifications.
- **Convertir un freebet maintenant** : pour le bookmaker choisi, les paris qui transforment un freebet ou un « pari
  remboursé si perdant » en gain garanti avec les cotes du moment, couverts aux meilleures cotes des autres.

Les surebets, les cotes boostées et les boosts couverts sont dans Cote Value (onglet Boosts) ; la page en donne le
nombre du moment.

## Calculs à la main

1. **Freebets et bonus couverts** : convertir un freebet ou un « premier pari remboursé si perdant » en gain garanti, en
   couvrant les autres issues chez un autre bookmaker (mises de couverture, gain, taux de conversion).
2. **Milieux** : « Plus de » une ligne chez un bookmaker, « Moins de » une ligne plus haute chez un autre. Résultat de
   chaque cas, fréquence de milieu à partir de laquelle c'est rentable, et probabilité du milieu d'après Pinnacle
   (deux lignes, ou une ligne et une loi normale).
3. **Promos « remboursé si… » et « payé dès 2 buts d'avance »** (football) : valeur exacte de la promo avec les
   probabilités justes du 1N2 de Pinnacle ; défaite d'un but, 0-0 et avance de deux buts par un modèle de Poisson ajusté
   sur ces cotes (la valeur « 2 buts d'avance » est un minimum : le modèle sous-estime les remontées). Pas en direct :
   les promos sont annoncées en texte libre sur chaque site.
4. **Combiné boosté** (« +X % sur tes combinés ») : probabilité juste de chaque sélection (cotes Pinnacle du marché ou
   cote juste), espérance avec le bonus sur le gain net ou sur le retour total, en argent ou en freebet, et bonus
   minimal pour être rentable. Sélections de matchs différents (indépendantes).

## Fichiers

- `site/` : la page publiée (`index.html`, `style.css`, `page.js` pour les calculs à la main, `direct.js` pour la partie
  en direct, `affichage.js` pour la mise en forme commune), calculs purs dans `site/calculs.js`, police dans
  `site/polices/`. `?donnees=<adresse>` remplace la source d'`euria.json` (essais).
- `tests/` : `node --test tests/*.test.mjs` (Node 20 ou plus, aucune dépendance).
- `.github/workflows/pages.yml` : tests puis publication sur GitHub Pages à chaque poussée sur `main`.

## Présentation

Une seule police, Archivo (chasse variable, licence OFL, servie avec la page), de l'encre, du papier et un jaune
(`#ffd100`). Des filets plutôt que des cadres, les chiffres tabulaires dans les colonnes, le chiffre qui décide surligné.
Mode sombre selon le téléphone. Tout le HTML produit passe par `remplir()` (`affichage.js`), qui met les espaces
insécables de la typographie française (« », avant : ; ? ! et %).

Rien n'est envoyé : la page lit les cotes publiées et calcule dans le navigateur. Jeu responsable : Joueurs Info
Service, 09 74 75 13 13.
