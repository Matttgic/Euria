// Calculs purs de l'onglet Stats : milieux réglés (euria-stats.json, publié par Cote Value), filtres et découpages
// par sport, type de pari, largeur du milieu, espérance, chance, bookmakers… Aucun accès à la page ici.

const JOUR_MS = 24 * 3600 * 1000;

/** Lignes de euria-stats.json ({colonnes, lignes}) en objets ; rien si le fichier est mal formé. */
export function lignesStats(doc) {
  const cols = doc && Array.isArray(doc.colonnes) ? doc.colonnes : null;
  if (!cols || !Array.isArray(doc.lignes)) return [];
  return doc.lignes.filter(Array.isArray).map(l => Object.fromEntries(cols.map((c, i) => [c, l[i] ?? null])))
    .filter(x => Number.isFinite(x.resultat_100) && Number.isFinite(x.esperance));
}

/** Clé d'un match : nom et heure de début (anciens fichiers sans nom : sport, ligue, début). */
export const cleMatch = x => [x.match ?? `${x.sport ?? ''}|${x.ligue ?? ''}`, x.debut ?? ''].join('|');

/**
 * Un seul milieu par match, celui de meilleure espérance au repérage (celui qu'on aurait joué) : les milieux d'un même
 * match se règlent sur le même score, ils ne sont pas indépendants.
 */
export function unParMatch(lignes) {
  const garde = new Map();
  for (const x of lignes) {
    const k = cleMatch(x), y = garde.get(k);
    if (!y || x.esperance > y.esperance) garde.set(k, x);
  }
  return lignes.filter(x => garde.get(cleMatch(x)) === x);
}

const dans = (x, min, max) => (!Number.isFinite(min) || (Number.isFinite(x) && x >= min))
  && (!Number.isFinite(max) || (Number.isFinite(x) && x <= max));

/**
 * Filtre : `jours` (match dans les N derniers jours avant `maintenant`), `sport`, `marche`, `periode`, `bookmaker`
 * (un des deux paris chez lui), `chance` ('pinnacle' : deux lignes de la référence, 'estimee' : modèle), bornes
 * `largeurMin/Max` (résultats dans le milieu), `esperanceMin/Max` (€ pour 100 €), `pMin/pMax` (chance du milieu, 0 à 1),
 * `avanceMin/Max` (heures entre le repérage et le match), `value` ('value' : en value au repérage, 'hors' : presque en
 * value, de −2 à 0 € pour 100 €, suivis pour choisir les critères ; les lignes sans `en_value` sont en value),
 * `unParMatch` (le meilleur milieu de chaque match, après les autres critères). Un critère vide ou absent ne filtre pas.
 */
export function filtrer(lignes, f = {}, maintenant = Date.now()) {
  const choisis = lignes.filter(x => {
    if (Number.isFinite(f.jours) && !(maintenant - Date.parse(x.debut) <= f.jours * JOUR_MS)) return false;
    for (const k of ['sport', 'marche', 'periode', 'ligue', 'reference']) if (f[k] && x[k] !== f[k]) return false;
    if (f.bookmaker && !(x.bookmakers || []).includes(f.bookmaker)) return false;
    if (f.chance === 'pinnacle' && x.estime) return false;
    if (f.chance === 'estimee' && !x.estime) return false;
    if (f.value === 'value' && x.en_value === false) return false;
    if (f.value === 'hors' && x.en_value !== false) return false;
    return dans(x.largeur, f.largeurMin, f.largeurMax) && dans(x.esperance, f.esperanceMin, f.esperanceMax)
      && dans(x.p_milieu, f.pMin, f.pMax) && dans(x.avance_h, f.avanceMin, f.avanceMax);
  });
  return f.unParMatch ? unParMatch(choisis) : choisis;
}

/**
 * Bilan d'un groupe pour 100 € par milieu : résultat, espérance annoncée, rendement (résultat / mises), milieux tombés
 * contre attendus, et écart résultat − espérance en écarts-types (z) : chaque milieu vaut `milieu_100` avec la chance
 * p, `hors_100` sinon, variance p(1 − p)(milieu − hors)². |z| < 2 : écart compatible avec le hasard. Le z suppose des
 * milieux indépendants : avec plusieurs milieux par match, il exagère (unParMatch). `matchs` : matchs différents.
 */
export function agreger(lignes) {
  let resultat = 0, esperance = 0, tombes = 0, attendus = 0, variance = 0, avecP = 0;
  const matchs = new Set();
  for (const x of lignes) {
    matchs.add(cleMatch(x));
    resultat += x.resultat_100;
    esperance += x.esperance;
    if (x.tombe) tombes++;
    if (Number.isFinite(x.p_milieu)) {
      attendus += x.p_milieu;
      avecP++;
      if (Number.isFinite(x.milieu_100) && Number.isFinite(x.hors_100))
        variance += x.p_milieu * (1 - x.p_milieu) * (x.milieu_100 - x.hors_100) ** 2;
    }
  }
  const n = lignes.length;
  return {n, matchs: matchs.size, resultat, esperance, rendement: n ? resultat / (100 * n) : null,
    rendement_attendu: n ? esperance / (100 * n) : null, tombes, attendus: avecP ? attendus : null,
    z: variance > 0 ? (resultat - esperance) / Math.sqrt(variance) : null};
}

/** Tranches [{nom, min, max}] (min ≤ x < max) : nom de la tranche d'un nombre, null s'il manque. */
const T = (...l) => l.map(([min, max, nom]) => ({min, max, nom}));
const dansTranche = tranches => x => Number.isFinite(x) ? (tranches.find(t => x >= t.min && x < t.max) || {}).nom ?? null : null;

const MARCHES = {TOTAL: 'Total', HANDICAP: 'Handicap (écart)', TOTAL_DOM: 'Total domicile', TOTAL_EXT: 'Total extérieur',
  JEUX_TOTAL: 'Total de jeux', JEUX_HANDICAP: 'Handicap de jeux', SETS_TOTAL: 'Total de sets', SETS_HANDICAP: 'Handicap de sets'};
const SPORTS = {football: 'Football', basket: 'Basket', tennis: 'Tennis', hockey: 'Hockey', handball: 'Handball',
  rugby: 'Rugby', volley: 'Volley', football_americain: 'Football américain', baseball: 'Baseball'};
const libre = s => s ? String(s)[0].toUpperCase() + String(s).slice(1).toLowerCase().replace(/_/g, ' ') : '';
export const nomSport = s => SPORTS[s] || libre(s);
export const nomMarche = m => MARCHES[m] || libre(m);
const quand = iso => isNaN(Date.parse(iso)) ? '' : new Date(iso).toLocaleString('fr-FR', {day: '2-digit', month: '2-digit',
  hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris'});
export const nomPeriode = p => p === 'MATCH' ? 'Match entier' : libre(p);

/**
 * Découpages proposés : `cle` donne le groupe d'un milieu (null : hors découpage), ou `valeur` et `tranches` (groupes
 * dans l'ordre des tranches ; sinon du plus fourni au moins fourni). `filtre` ou `champs` : critères qu'un clic sur le
 * groupe applique (critereGroupe).
 */
export const DECOUPAGES = [
  {id: 'sport', titre: 'Par sport', cle: x => x.sport, nom: nomSport, filtre: 'sport'},
  {id: 'marche', titre: 'Par type de pari', cle: x => x.marche, nom: nomMarche, filtre: 'marche'},
  {id: 'largeur', titre: 'Par largeur du milieu', sous: 'résultats gagnants entre les deux lignes', valeur: x => x.largeur,
    champs: ['largeurMin', 'largeurMax'], entier: true,
    tranches: T([-Infinity, 2, '1'], [2, 3, '2'], [3, 5, '3 à 4'], [5, 10, '5 à 9'], [10, 20, '10 à 19'], [20, Infinity, '20 et plus'])},
  {id: 'esperance', titre: 'Par espérance annoncée', sous: 'pour 100 € misés', valeur: x => x.esperance,
    champs: ['esperanceMin', 'esperanceMax'],
    tranches: T([-Infinity, 1, 'moins de 1 €'], [1, 2, '1 à 2 €'], [2, 5, '2 à 5 €'], [5, 10, '5 à 10 €'], [10, Infinity, '10 € et plus'])},
  {id: 'chance', titre: 'Par chance du milieu', valeur: x => x.p_milieu, champs: ['pMin', 'pMax'], echelle: 100,
    tranches: T([-Infinity, 0.05, 'moins de 5 %'], [0.05, 0.1, '5 à 10 %'], [0.1, 0.2, '10 à 20 %'], [0.2, 0.4, '20 à 40 %'],
      [0.4, Infinity, '40 % et plus'])},
  {id: 'value', titre: 'En value ou presque', cle: x => x.en_value === false ? 'hors' : 'value', filtre: 'value',
    nom: v => v === 'hors' ? 'Presque en value (−2 à 0 €)' : 'En value'},
  {id: 'estime', titre: 'Par origine de la chance', cle: x => x.estime ? 'estimee' : 'pinnacle', filtre: 'chance',
    nom: v => v === 'estimee' ? 'Estimée (une ligne de référence)' : 'Deux lignes de référence'},
  {id: 'bookmakers', titre: 'Par bookmakers', sous: 'les deux paris', cle: x => (x.bookmakers || []).join(' + ') || null,
    filtre: 'bookmaker', filtrable: v => !v.includes(' + ')},
  {id: 'periode', titre: 'Par période du match', cle: x => x.periode, nom: nomPeriode, filtre: 'periode'},
  {id: 'avance', titre: 'Par avance sur le match', sous: 'repéré combien d’heures avant le début', valeur: x => x.avance_h,
    champs: ['avanceMin', 'avanceMax'],
    tranches: T([-Infinity, 1, 'moins d’1 h'], [1, 6, '1 à 6 h'], [6, 24, '6 à 24 h'], [24, 72, '1 à 3 jours'], [72, Infinity, '3 jours et plus'])},
  {id: 'duree', titre: 'Par durée d’affichage', sous: 'temps resté disponible', valeur: x => x.duree_min,
    tranches: T([-Infinity, 10, 'moins de 10 min'], [10, 60, '10 à 60 min'], [60, 360, '1 à 6 h'], [360, Infinity, '6 h et plus'])},
  {id: 'reference', titre: 'Par référence', cle: x => x.reference, filtre: 'reference'},
  {id: 'ligue', titre: 'Par ligue', sous: '15 plus fournies', cle: x => x.ligue, filtre: 'ligue', max: 15},
  {id: 'match', titre: 'Par match', sous: '30 plus récents', cle: cleMatch, max: 30, recent: x => x.debut || '',
    libelle: x => [x.match || [nomSport(x.sport), x.ligue].filter(Boolean).join(', '), quand(x.debut)].filter(Boolean).join(' · ')},
];

/**
 * Critères qu'un clic sur un groupe applique : sa valeur pour un champ choisi dans une liste, ses bornes pour une
 * tranche (borne haute incluse, moins 1 pour un nombre entier ; `echelle` : saisie en %). null : pas de filtre.
 */
export function critereGroupe(d, cle) {
  if (d.filtre) return !d.filtrable || d.filtrable(cle) ? {[d.filtre]: cle} : null;
  const t = d.champs && (d.tranches || []).find(x => x.nom === cle);
  if (!t) return null;
  const e = d.echelle || 1, borne = v => Number.isFinite(v) ? Math.round(v * e * 100) / 100 : '';
  return {[d.champs[0]]: borne(t.min), [d.champs[1]]: borne(Number.isFinite(t.max) && d.entier ? t.max - 1 : t.max)};
}

/** Groupes d'un découpage : [{cle, nom, filtre, ...agreger}], groupes vides exclus. */
export function decouper(lignes, d) {
  const cle = d.tranches ? x => dansTranche(d.tranches)(d.valeur(x)) : d.cle;
  const groupes = new Map();
  for (const x of lignes) {
    const k = cle(x);
    if (k == null || k === '') continue;
    if (!groupes.has(k)) groupes.set(k, []);
    groupes.get(k).push(x);
  }
  const rang = k => d.tranches ? d.tranches.findIndex(t => t.nom === k) : 0;
  let out = [...groupes].map(([k, g]) => ({cle: k, nom: d.libelle ? d.libelle(g[0]) : d.nom ? d.nom(k) : String(k), debut: d.recent ? d.recent(g[0]) : '',
    ...agreger(g)}));
  out.sort(d.tranches ? (a, b) => rang(a.cle) - rang(b.cle)
    : d.recent ? (a, b) => b.debut.localeCompare(a.debut) || b.n - a.n
      : (a, b) => b.n - a.n || String(a.nom).localeCompare(b.nom));
  if (d.max) out = out.slice(0, d.max);
  return out;
}

/** Valeurs distinctes d'un champ (listes des filtres), les plus fréquentes d'abord. */
export function valeurs(lignes, champ) {
  const c = new Map();
  for (const x of lignes) for (const v of champ === 'bookmakers' ? x.bookmakers || [] : [x[champ]])
    if (v != null && v !== '') c.set(v, (c.get(v) || 0) + 1);
  return [...c].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(b[0])).map(([v]) => v);
}
