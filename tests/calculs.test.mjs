// Calculs de la page (site/calculs.js) : node --test tests/
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {
  lireNombre, probasJustes, couvrirFreebet, couvrirRembourse, milieu, probasMilieuDeuxLignes, probasMilieuNormale,
  phi, phiInverse, scores, issues, ajusterPoisson, probaMenerDe, valeurPromo, probaSelection, combineBooste,
  classerConversions, milieuRecalcule,
} from '../site/calculs.js';

const proche = (a, b, eps = 1e-2) => assert.ok(Math.abs(a - b) <= eps, `${a} ≠ ${b} (± ${eps})`);

test('nombres saisis à la française', () => {
  assert.equal(lireNombre('2,10'), 2.1);
  assert.equal(lireNombre(' 100 € '), 100);
  assert.ok(Number.isNaN(lireNombre('')));
  assert.ok(Number.isNaN(lireNombre('abc')));
});

test('probabilités justes : marge retirée en proportion', () => {
  const p = probasJustes([1.9, 1.9]);
  proche(p[0], 0.5, 1e-12);
  assert.equal(probasJustes([1.9, 0.9]), null);
});

test('freebet de 10 € à 4,00 couvert à 1,33 : environ 7,44 € garantis', () => {
  const r = couvrirFreebet({montant: 10, cote: 4, couvertures: [1.33]});
  proche(r.mises[0], 22.56);
  proche(r.gain, 7.44);
  proche(r.conversion, 0.744, 1e-3);
  // les deux issues rapportent pareil
  proche(r.retourSiA - r.mises[0], r.mises[0] * 1.33 - r.mises[0]);
});

test('freebet à 3 issues : deux couvertures, gain égal partout', () => {
  const r = couvrirFreebet({montant: 10, cote: 5, couvertures: [3.5, 1.9]});
  const total = r.mises[0] + r.mises[1];
  proche(r.gain, 40 - total, 1e-9);
  proche(r.mises[0] * 3.5 - total, r.gain, 1e-9);
  proche(r.mises[1] * 1.9 - total, r.gain, 1e-9);
  assert.equal(couvrirFreebet({montant: 10, cote: 5, couvertures: []}), null);
});

test('freebet mise rendue : la mise compte dans le retour', () => {
  const r = couvrirFreebet({montant: 10, cote: 4, couvertures: [1.33], miseRendue: true});
  proche(r.retourSiA, 40, 1e-9);
});

test('premier pari remboursé en freebet (valeur 0,75) : même gain quelle que soit l’issue', () => {
  const r = couvrirRembourse({mise: 100, cote: 3, couvertures: [1.45], valeur: 0.75});
  proche(r.mises[0], 155.17);
  proche(r.gain, 44.83);
  proche(r.mises[0] * 1.45 - r.mises[0] - 100 + r.rendu, r.gain, 1e-9);
  const plafonne = couvrirRembourse({mise: 100, cote: 3, couvertures: [1.45], valeur: 1, plafond: 50});
  assert.equal(plafonne.rendu, 50);
});

test('milieu 210,5 / 213,5 à 1,90, 100 € chacun : +180 € dedans, −10 € sinon, rentable au-delà d’une fois sur 19', () => {
  const r = milieu({ligne1: 210.5, cote1: 1.9, ligne2: 213.5, cote2: 1.9, total: 200, mode: 'egal'});
  proche(r.dedans, 180, 1e-9);
  proche(r.dessous, -10, 1e-9);
  proche(r.dessus, -10, 1e-9);
  proche(r.seuil, 1 / 19, 1e-9);
});

test('milieu équilibré : même retour des deux côtés', () => {
  const r = milieu({ligne1: 2.5, cote1: 2.1, ligne2: 3.5, cote2: 1.8, total: 100});
  proche(r.mises[0] * 2.1, r.mises[1] * 1.8, 1e-9);
  proche(r.dessous, r.dessus, 1e-9);
});

test('milieu : lignes entières et lignes inversées refusées', () => {
  assert.ok(milieu({ligne1: 210, cote1: 1.9, ligne2: 213.5, cote2: 1.9, total: 200}).erreur);
  assert.ok(milieu({ligne1: 213.5, cote1: 1.9, ligne2: 210.5, cote2: 1.9, total: 200}).erreur);
});

test('probabilité du milieu : deux lignes Pinnacle', () => {
  const p = probasMilieuDeuxLignes({plus1: 1.8, moins1: 2.0, plus2: 2.1, moins2: 1.72});
  proche(p.dessous + p.milieu + p.dessus, 1, 1e-12);
  assert.ok(p.milieu > 0);
  assert.equal(probasMilieuDeuxLignes({plus1: 2.1, moins1: 1.72, plus2: 1.8, moins2: 2.0}), null);
  const r = milieu({ligne1: 210.5, cote1: 1.9, ligne2: 213.5, cote2: 1.9, total: 200, mode: 'egal', pMilieu: p});
  proche(r.esperance, p.milieu * 180 - (1 - p.milieu) * 10, 1e-9);
});

test('loi normale', () => {
  proche(phi(0), 0.5, 1e-7);
  proche(phi(1.96), 0.975, 1e-4);
  proche(phiInverse(0.975), 1.96, 1e-3);
  const p = probasMilieuNormale({ligne: 211.5, plus: 1.9, moins: 1.9, sigma: 13, ligne1: 210.5, ligne2: 213.5});
  proche(p.moyenne, 211.5, 1e-6);
  proche(p.milieu, phi(2 / 13) - phi(-1 / 13), 1e-9);
});

test('Poisson : matrice complète et ajustement qui retrouve les buts attendus', () => {
  const m = scores(1.6, 1.0);
  proche(m.flat().reduce((a, b) => a + b, 0), 1, 1e-5);
  const r = issues(m);
  proche(r.dom + r.nul + r.ext, 1, 1e-5);
  const fit = ajusterPoisson({dom: r.dom, nul: r.nul, ext: r.ext, plus25: r.plus25});
  proche(fit.lDom, 1.6, 0.01);
  proche(fit.lExt, 1.0, 0.01);
});

test('mener de deux buts : au moins aussi probable que gagner de deux buts ou plus', () => {
  const m = scores(1.3, 1.3);
  let gagneDeDeux = 0;
  for (let i = 0; i < m.length; i++) for (let j = 0; j < m.length; j++) if (i - j >= 2) gagneDeDeux += m[i][j];
  const r = probaMenerDe(1.3, 1.3, 2);
  assert.ok(r.mene >= gagneDeDeux - 1e-3, `${r.mene} < ${gagneDeDeux}`);
  // gagné = mené de 2 et gagné, ou gagné d'un but sans avoir mené de 2 : la somme dépasse la victoire seule
  assert.ok(r.mene + r.gagneSansMener >= issues(m).dom - 1e-3);
  assert.ok(r.mene > 0.15 && r.mene < 0.45);
});

const PIN = {dom: 2.0, nul: 3.6, ext: 4.5};              // 1/2 + 1/3,6 + 1/4,5 = 1 : probabilités exactes

test('remboursé si nul, en argent : la promo vaut la probabilité du nul', () => {
  const r = valeurPromo({type: 'nul', camp: 'dom', cote: 2.0, mise: 10, pinnacle: PIN});
  proche(r.evSansPromo, 0, 1e-9);
  proche(r.ev, 10 / 3.6, 1e-9);
  const freebet = valeurPromo({type: 'nul', camp: 'dom', cote: 2.0, mise: 10, pinnacle: PIN, remboursement: 0.75});
  proche(freebet.ev, 0.75 * 10 / 3.6, 1e-9);
});

test('remboursé si le pari perd, plafonné', () => {
  const r = valeurPromo({type: 'perdant', camp: 'ext', cote: 4.5, mise: 20, pinnacle: PIN, plafond: 10});
  proche(r.ev, 20 * (4.5 / 4.5 - 1) + (1 - 1 / 4.5) * 10, 1e-9);
});

test('payé dès 2 buts d’avance : vaut plus que le pari seul', () => {
  const r = valeurPromo({type: 'deuxButs', camp: 'dom', cote: 2.0, mise: 10, pinnacle: {...PIN, plus25: 1.9, moins25: 1.9}});
  assert.ok(r.pPaye > r.pGagne && r.ev > r.evSansPromo);
  assert.ok(r.pPaye - r.pGagne < 0.15);
});

test('défaite par un but et 0-0 : probabilités du modèle', () => {
  const a = valeurPromo({type: 'defaiteUn', camp: 'dom', cote: 2.0, mise: 10, pinnacle: PIN});
  const b = valeurPromo({type: 'zeroZero', camp: 'dom', cote: 2.0, mise: 10, pinnacle: PIN});
  assert.ok(a.pRembourse > 0.1 && a.pRembourse < 0.3);
  assert.ok(b.pRembourse > 0.03 && b.pRembourse < 0.15);
});

test('autre événement : probabilité saisie, jamais plus que la défaite', () => {
  const r = valeurPromo({type: 'autre', camp: 'dom', cote: 2.0, mise: 10, pinnacle: PIN, pAutre: 0.1});
  proche(r.ev, 1, 1e-9);
  assert.equal(valeurPromo({type: 'autre', camp: 'dom', cote: 2.0, mise: 10, pinnacle: PIN, pAutre: 0.6}), null);
});

test('probabilité d’une sélection : marché Pinnacle complet ou cote juste', () => {
  proche(probaSelection({pinnacle: [1.95, 1.95]}), 0.5, 1e-12);
  proche(probaSelection({pinnacle: [2.0, 3.6, 4.5]}), 0.5, 1e-12);
  proche(probaSelection({juste: 2.5}), 0.4, 1e-12);
  assert.equal(probaSelection({pinnacle: [1.95]}), null);          // une seule cote Pinnacle : marge inconnue
  assert.equal(probaSelection({pinnacle: [1.95, NaN]}), null);
  assert.equal(probaSelection({juste: 0.9}), null);
});

// Trois sélections à 5 % de marge chacune (cote = cote juste / 1,05) : 1,05³ − 1 = 15,8 % de marges cumulées.
const marge5 = juste => ({cote: juste / 1.05, proba: 1 / juste});

test('combiné boosté sur le retour total : +20 % bat 15,8 % de marges, quelles que soient les cotes', () => {
  for (const juste of [1.3, 2.0, 3.0]) {
    const r = combineBooste({selections: [marge5(juste), marge5(juste), marge5(juste)], mise: 10, bonus: 0.2, base: 'total'});
    proche(r.rendement, 1.2 / 1.05 ** 3 - 1, 1e-9);              // +3,7 %
    proche(r.bonusMinimal, 1.05 ** 3 - 1, 1e-9);
  }
});

test('combiné boosté sur le gain net : tout dépend des cotes', () => {
  const haut = combineBooste({selections: [marge5(2), marge5(2), marge5(2)], mise: 10, bonus: 0.2});
  const bas = combineBooste({selections: [marge5(1.3), marge5(1.3), marge5(1.3)], mise: 10, bonus: 0.2});
  proche(haut.rendement, 0.0117, 1e-3);                           // cotes à 1,90 : encore rentable
  proche(bas.rendement, -0.0543, 1e-3);                           // cotes à 1,24 : perdant
  proche(haut.sansBonus, 1 / 1.05 ** 3 - 1, 1e-9);
  proche(haut.cote * (1 + haut.bonusMinimal) - haut.bonusMinimal, 8, 1e-9);   // à ce bonus, retour × P = mise
  proche(haut.retour, 10 * (haut.cote + (haut.cote - 1) * 0.2), 1e-9);
});

test('combiné boosté : bonus en freebet compté à sa valeur, saisies impossibles refusées', () => {
  const sel = [marge5(2), marge5(2), marge5(2)];
  const argent = combineBooste({selections: sel, mise: 10, bonus: 0.2});
  const freebet = combineBooste({selections: sel, mise: 10, bonus: 0.2, valeur: 0.75});
  proche(argent.esperance - freebet.esperance, 0.125 * 10 * (argent.cote - 1) * 0.2 * 0.25, 1e-9);
  proche(combineBooste({selections: [{cote: 2.2, proba: 0.5}, {cote: 2.2, proba: 0.5}], mise: 10, bonus: 0}).bonusMinimal, 0, 1e-12);
  assert.equal(combineBooste({selections: [marge5(2)], mise: 10, bonus: 0.2}), null);
  assert.equal(combineBooste({selections: sel, mise: 0, bonus: 0.2}), null);
  assert.equal(combineBooste({selections: sel, mise: 10, bonus: 0.2, valeur: 1.5}), null);
  assert.equal(combineBooste({selections: sel, mise: 10, bonus: 0.2, base: 'autre'}), null);
});

// Données au format d'euria.json (Cote Value, cotes/euria.py).
const CANDIDATS = [
  {match: 'A – B', jambes: [{cote: 3.4}, {cote: 1.98}, {cote: 4.5}]},     // nul, couvert par les deux équipes
  {match: 'C – D', jambes: [{cote: 1.95}, {cote: 1.8}]},
  {match: 'E – F', jambes: [{cote: 2.0}]},                                  // sans couverture : ignoré
];

test('en direct : conversions recalculées pour le montant saisi, la meilleure d’abord', () => {
  const fb = classerConversions(CANDIDATS, {type: 'freebet', montant: 10});
  assert.deepEqual(fb.map(c => c.match), ['A – B', 'C – D']);
  proche(fb[0].calcul.gain, 24 * (1 - 1 / 1.98 - 1 / 4.5), 1e-9);
  proche(fb[0].calcul.mises[0], 24 / 1.98, 1e-9);
  const rb = classerConversions(CANDIDATS, {type: 'rembourse', montant: 50, valeur: 0.75, plafond: 20});
  proche(rb[0].calcul.rendu, 15, 1e-9);                               // 75 % de 20 € remboursés
  assert.equal(classerConversions(CANDIDATS, {type: 'freebet', montant: 0}).length, 0);
  assert.deepEqual(classerConversions(null, {montant: 10}), []);
});

test('en direct : milieu recalculé pour la somme saisie', () => {
  const m = {bornes: [210.5, 213.5], jambes: [{cote: 1.95}, {cote: 2.0}], p_dessous: 0.45, p_milieu: 0.08, p_dessus: 0.47};
  const r = milieuRecalcule(m, 200);
  proche(r.mises[0], 200 * 2 / 3.95, 1e-9);
  proche(r.esperance, r.mises[0] * (1.95 * 0.55 - 1) + r.mises[1] * (2 * 0.53 - 1), 1e-9);
  assert.equal(milieuRecalcule({jambes: []}, 100), null);
});
