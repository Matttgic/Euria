import test from 'node:test';
import assert from 'node:assert/strict';
import {lignesStats, filtrer, agreger, decouper, valeurs, critereGroupe, unParMatch, cleMatch, DECOUPAGES} from '../site/stats.js';

const COLS = ['debut', 'match', 'sport', 'ligue', 'marche', 'periode', 'largeur', 'reference', 'bookmakers', 'cotes', 'esperance',
  'p_milieu', 'estime', 'milieu_100', 'hors_100', 'avance_h', 'duree_min', 'esperance_derniere', 'resultat_100', 'tombe'];
const MAINTENANT = Date.parse('2026-11-23T12:00:00Z');
const l = (o) => COLS.map(c => o[c] ?? null);
const DOC = {colonnes: COLS, lignes: [
  l({debut: '2026-11-22T20:00:00Z', match: 'Lakers – Celtics', sport: 'basket', ligue: 'NBA', marche: 'TOTAL', periode: 'MATCH', largeur: 3,
    bookmakers: ['Betclic', 'Winamax'], esperance: 6.63, p_milieu: 0.08, estime: false, milieu_100: 97.47, hors_100: -1.27,
    avance_h: 5, duree_min: 30, resultat_100: 97.47, tombe: true}),
  l({debut: '2026-11-20T20:00:00Z', match: 'Heat – Knicks', sport: 'basket', ligue: 'NBA', marche: 'HANDICAP', periode: 'MATCH', largeur: 1,
    bookmakers: ['Betclic'], esperance: 1.5, p_milieu: 0.03, estime: true, milieu_100: 90, hors_100: -2,
    avance_h: 30, duree_min: 5, resultat_100: -2, tombe: false}),
  l({debut: '2026-10-01T20:00:00Z', sport: 'football_americain', ligue: 'NFL', marche: 'HANDICAP', periode: 'MATCH',
    largeur: 22, bookmakers: ['Betclic'], esperance: 4, p_milieu: 0.74, estime: true, milieu_100: 20, hors_100: -40,
    avance_h: 0.5, duree_min: 400, resultat_100: -40, tombe: false}),
  l({debut: '2026-11-22T20:00:00Z', match: 'Lakers – Celtics', sport: 'basket', ligue: 'NBA', marche: 'HANDICAP',
    periode: 'MATCH', largeur: 5, bookmakers: ['Winamax'], esperance: 2, p_milieu: 0.1, estime: true, milieu_100: 80,
    hors_100: -5, avance_h: 5, duree_min: 30, resultat_100: -5, tombe: false}),
  ['mal formé'], 'pas une ligne',
]};

test('lignes de euria-stats.json : colonnes nommées, lignes incomplètes écartées', () => {
  const L = lignesStats(DOC);
  assert.equal(L.length, 4);
  assert.equal(L[0].sport, 'basket');
  assert.deepEqual(L[0].bookmakers, ['Betclic', 'Winamax']);
  assert.deepEqual(lignesStats(null), []);
  assert.deepEqual(lignesStats({colonnes: COLS}), []);
});

test('filtres : sport, type de pari, bookmaker, chance, bornes, jours ; un critère vide ne filtre pas', () => {
  const L = lignesStats(DOC);
  const n = f => filtrer(L, f, MAINTENANT).length;
  assert.equal(n({}), 4);
  assert.equal(n({sport: 'basket'}), 3);
  assert.equal(n({marche: 'HANDICAP'}), 3);
  assert.equal(n({bookmaker: 'Winamax'}), 2);
  assert.equal(n({chance: 'pinnacle'}), 1);
  assert.equal(n({chance: 'estimee'}), 3);
  assert.equal(n({largeurMin: 2, largeurMax: 5}), 2);
  assert.equal(n({esperanceMin: 2}), 3);
  assert.equal(n({pMax: 0.1}), 3);
  assert.equal(n({avanceMin: 1}), 3);
  assert.equal(n({jours: 7}), 3);
  assert.equal(n({largeurMin: NaN, sport: ''}), 4);
  assert.equal(n({unParMatch: true}), 3);
  assert.equal(n({unParMatch: true, bookmaker: 'Winamax'}), 1);   // critères d'abord, puis le meilleur du match
});

test('agrégat : résultat, rendement, tombés contre attendus, écart en écarts-types', () => {
  const L = lignesStats(DOC).slice(0, 2);
  const g = agreger(L);
  assert.equal(g.n, 2);
  assert.ok(Math.abs(g.resultat - 95.47) < 1e-9);
  assert.ok(Math.abs(g.rendement - 0.4774) < 1e-3);
  assert.equal(g.tombes, 1);
  assert.ok(Math.abs(g.attendus - 0.11) < 1e-9);
  const variance = 0.08 * 0.92 * 98.74 ** 2 + 0.03 * 0.97 * 92 ** 2;
  assert.ok(Math.abs(g.z - (95.47 - 8.13) / Math.sqrt(variance)) < 1e-9);
  assert.deepEqual(agreger([]), {n: 0, matchs: 0, resultat: 0, esperance: 0, rendement: null, rendement_attendu: null, tombes: 0,
    attendus: null, z: null});
});

test('découpages : sport le plus fourni d’abord, tranches de largeur dans l’ordre', () => {
  const L = lignesStats(DOC), d = id => DECOUPAGES.find(x => x.id === id);
  assert.deepEqual(decouper(L, d('sport')).map(g => [g.nom, g.n, g.matchs]), [['Basket', 3, 2], ['Football américain', 1, 1]]);
  assert.deepEqual(decouper(L, d('largeur')).map(g => g.nom), ['1', '3 à 4', '5 à 9', '20 et plus']);
  assert.deepEqual(decouper(L, d('chance')).map(g => g.nom), ['moins de 5 %', '5 à 10 %', '10 à 20 %', '40 % et plus']);
  assert.deepEqual(decouper(L, d('marche')).map(g => g.nom), ['Handicap (écart)', 'Total']);
  assert.deepEqual(decouper(L, d('reference')), []);                     // pas de référence dans ces lignes
  for (const x of DECOUPAGES) assert.ok(Array.isArray(decouper(L, x)), x.id);
});

test('clic sur un groupe : critère du champ, bornes de la tranche (entières, en %)', () => {
  const d = id => DECOUPAGES.find(x => x.id === id);
  assert.deepEqual(critereGroupe(d('sport'), 'basket'), {sport: 'basket'});
  assert.deepEqual(critereGroupe(d('estime'), 'estimee'), {chance: 'estimee'});
  assert.deepEqual(critereGroupe(d('largeur'), '3 à 4'), {largeurMin: 3, largeurMax: 4});
  assert.deepEqual(critereGroupe(d('largeur'), '20 et plus'), {largeurMin: 20, largeurMax: ''});
  assert.deepEqual(critereGroupe(d('largeur'), '1'), {largeurMin: '', largeurMax: 1});
  assert.deepEqual(critereGroupe(d('chance'), '5 à 10 %'), {pMin: 5, pMax: 10});
  assert.equal(critereGroupe(d('bookmakers'), 'Betclic + Winamax'), null);
  assert.deepEqual(critereGroupe(d('bookmakers'), 'Betclic'), {bookmaker: 'Betclic'});
  assert.equal(critereGroupe(d('duree'), '10 à 60 min'), null);
});

test('valeurs des listes : les plus fréquentes d’abord, bookmakers un par un', () => {
  const L = lignesStats(DOC);
  assert.deepEqual(valeurs(L, 'sport'), ['basket', 'football_americain']);
  assert.deepEqual(valeurs(L, 'bookmakers'), ['Betclic', 'Winamax']);
});

test('un milieu par match : le meilleur (espérance au repérage), les matchs comptés à part', () => {
  const L = lignesStats(DOC);
  const garde = unParMatch(L);
  assert.equal(garde.length, 3);
  assert.equal(garde.filter(x => x.match === 'Lakers – Celtics')[0].esperance, 6.63);
  assert.equal(agreger(L).matchs, 3);
  assert.equal(cleMatch({sport: 'nfl', ligue: null, debut: 'x'}), 'nfl||x');       // ancien fichier sans nom de match
  const m = decouper(L, DECOUPAGES.find(x => x.id === 'match'));
  assert.equal(m[0].n, 2);                                                         // le plus récent d'abord
  assert.match(m[0].nom, /^Lakers – Celtics · 22\/11/);
  assert.equal(m.at(-1).nom.startsWith('Football américain, NFL'), true);
});

test('milieux en value ou presque : filtre et découpage ; une ligne sans en_value est en value', () => {
  const cols = [...COLS, 'en_value'];
  const ligne = (o) => cols.map(c => o[c] ?? null);
  const base = {debut: '2026-11-22T20:00:00Z', sport: 'football', marche: 'TOTAL', resultat_100: -2, tombe: false};
  const L = lignesStats({colonnes: cols, lignes: [ligne({...base, match: 'A – B', esperance: 3, en_value: true}),
    ligne({...base, match: 'C – D', esperance: -1.2, en_value: false}), ligne({...base, match: 'E – F', esperance: 2})]});
  assert.equal(filtrer(L, {value: 'value'}, MAINTENANT).length, 2);
  assert.deepEqual(filtrer(L, {value: 'hors'}, MAINTENANT).map(x => x.match), ['C – D']);
  assert.equal(filtrer(L, {}, MAINTENANT).length, 3);
  const d = DECOUPAGES.find(x => x.id === 'value');
  const g = decouper(L, d);
  assert.deepEqual(g.map(x => [x.cle, x.n]), [['value', 2], ['hors', 1]]);
  assert.deepEqual(critereGroupe(d, 'hors'), {value: 'hors'});
});
