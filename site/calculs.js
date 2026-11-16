// Calculs de la page : fonctions pures, sans DOM (testées par tests/calculs.test.mjs).
// Toutes les cotes sont décimales (2,10 = 2.1). Les probabilités justes viennent des cotes Pinnacle saisies,
// marge retirée en proportion (chaque 1/cote divisé par la somme des 1/cote du marché).

/** « 2,10 », « 2.1 », « 10 € » -> nombre ; vide ou illisible -> NaN. */
export function lireNombre(texte) {
  const s = String(texte ?? '').replace(/\s|€|%/g, '').replace(',', '.');
  return s === '' ? NaN : Number(s);
}

const valide = c => Number.isFinite(c) && c > 1;

/** Probabilités justes d'un marché complet (cotes Pinnacle) : 1/cote normalisés. */
export function probasJustes(cotes) {
  if (!cotes.length || !cotes.every(valide)) return null;
  const inv = cotes.map(c => 1 / c), s = inv.reduce((a, b) => a + b, 0);
  return inv.map(x => x / s);
}

// --------------------------------------------------------------------------- 1. freebets et bonus

/**
 * Freebet joué sur A (cote `cote`), couvert en argent réel sur toutes les autres issues (`couvertures`, une cote par
 * issue : « pas A » à 2 issues, ou N et 2 séparément à 3 issues). `miseRendue` : freebet dont la mise revient avec le
 * gain (rare) ; sinon seul le gain net est payé.
 * Mises égalisées : chaque issue rapporte la même somme. Rend {mises, gain, conversion, retourSiA}.
 */
export function couvrirFreebet({montant, cote, couvertures, miseRendue = false}) {
  if (!(montant > 0) || !valide(cote) || !couvertures.length || !couvertures.every(valide)) return null;
  const retour = miseRendue ? montant * cote : montant * (cote - 1);
  const mises = couvertures.map(c => retour / c);
  const gain = retour - mises.reduce((a, b) => a + b, 0);
  return {mises, gain, conversion: gain / montant, retourSiA: retour};
}

/**
 * « Premier pari remboursé si perdant » : mise réelle `mise` sur A à `cote`, couverte sur les autres issues.
 * Remboursement : `valeur` € par € remboursé (1 en argent, ~0,75 en freebet), plafonné à `plafond` €.
 * Mises égalisées : gain identique quelle que soit l'issue (le remboursement compté à sa valeur).
 */
export function couvrirRembourse({mise, cote, couvertures, valeur = 1, plafond = Infinity}) {
  if (!(mise > 0) || !valide(cote) || !couvertures.length || !couvertures.every(valide)) return null;
  if (!(valeur >= 0 && valeur <= 1)) return null;
  const rendu = valeur * Math.min(mise, plafond > 0 ? plafond : Infinity);
  const mises = couvertures.map(c => (mise * cote - rendu) / c);
  const total = mises.reduce((a, b) => a + b, 0);
  return {mises, gain: mise * (cote - 1) - total, rendu, miseTotale: mise + total};
}

// --------------------------------------------------------------------------- 4. milieux

const demiLigne = x => Number.isFinite(x) && Math.abs(2 * x - Math.round(2 * x)) < 1e-9 && Math.abs(Math.round(2 * x)) % 2 === 1;

/** Fonction de répartition de la loi normale centrée réduite (approximation d'Abramowitz-Stegun 7.1.26). */
export function phi(z) {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    * Math.exp(-z * z / 2);
  return z >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}

/** Quantile de la loi normale (inverse de phi, par dichotomie : assez précis pour ce calcul). */
export function phiInverse(p) {
  let bas = -8, haut = 8;
  for (let i = 0; i < 80; i++) { const m = (bas + haut) / 2; if (phi(m) < p) bas = m; else haut = m; }
  return (bas + haut) / 2;
}

/**
 * Milieu : « Plus de L1 » à `cote1` chez A et « Moins de L2 » à `cote2` chez B, avec L1 < L2 (demi-lignes).
 * Fonctionne aussi pour un écart de points (handicaps) : X = l'écart, côté 1 gagne si X > L1, côté 2 si X < L2.
 * `mode` : « equilibre » (même retour hors du milieu) ou « egal » (même mise). `total` : somme des deux mises.
 * Probabilité du milieu (facultative) : `pMilieu` directement. Rend les trois résultats, la fréquence de milieu à
 * partir de laquelle c'est rentable et l'espérance si `pMilieu` est donnée.
 */
export function milieu({ligne1, cote1, ligne2, cote2, total, mode = 'equilibre', pMilieu = null}) {
  if (!valide(cote1) || !valide(cote2) || !(total > 0)) return null;
  if (!demiLigne(ligne1) || !demiLigne(ligne2)) return {erreur: 'Utilise des lignes en ,5 (sans remboursement possible).'};
  if (!(ligne1 < ligne2)) return {erreur: 'Il faut « Plus de » sous « Moins de » : sinon il n’y a pas de milieu.'};
  const m1 = mode === 'egal' ? total / 2 : total * cote2 / (cote1 + cote2);
  const m2 = total - m1;
  const dessous = m2 * cote2 - total, dessus = m1 * cote1 - total, dedans = m1 * cote1 + m2 * cote2 - total;
  // Fréquence de milieu qui annule l'espérance, en supposant le reste partagé comme les deux côtés hors milieu
  // (mode équilibré : les deux côtés rapportent pareil).
  const pire = Math.min(dessous, dessus);
  const seuil = pire >= 0 ? 0 : -pire / (dedans - pire);
  const out = {mises: [m1, m2], dessous, dessus, dedans, seuil, largeur: ligne2 - ligne1};
  if (pMilieu != null && Number.isFinite(pMilieu.milieu)) {
    out.esperance = pMilieu.milieu * dedans + pMilieu.dessous * dessous + pMilieu.dessus * dessus;
  }
  return out;
}

/** Probabilités dessous / milieu / dessus d'après deux lignes Pinnacle (Plus et Moins à L1, Plus et Moins à L2). */
export function probasMilieuDeuxLignes({plus1, moins1, plus2, moins2}) {
  const a = probasJustes([plus1, moins1]), b = probasJustes([plus2, moins2]);
  if (!a || !b) return null;
  const pPlus1 = a[0], pPlus2 = b[0];
  if (pPlus2 > pPlus1) return null;                 // incohérent : plus probable de dépasser la ligne la plus haute
  return {dessous: 1 - pPlus1, milieu: pPlus1 - pPlus2, dessus: pPlus2};
}

/**
 * Probabilités dessous / milieu / dessus avec une seule ligne Pinnacle (Plus / Moins à `ligne`) et une loi normale
 * d'écart-type `sigma` (repère : NBA total ~ 13 points, écart ~ 12 ; NFL total ~ 10, écart ~ 13,5).
 */
export function probasMilieuNormale({ligne, plus, moins, sigma, ligne1, ligne2}) {
  const p = probasJustes([plus, moins]);
  if (!p || !(sigma > 0)) return null;
  const mu = ligne + sigma * phiInverse(p[0]);      // P(X > ligne) = p[0]
  const sous1 = phi((ligne1 - mu) / sigma), sous2 = phi((ligne2 - mu) / sigma);
  return {dessous: sous1, milieu: Math.max(0, sous2 - sous1), dessus: 1 - sous2, moyenne: mu};
}

// --------------------------------------------------------------------------- 5. promos (football, loi de Poisson)

const MAX_BUTS = 10;

function poissonLoi(lambda) {
  const out = [Math.exp(-lambda)];
  for (let k = 1; k <= MAX_BUTS; k++) out.push(out[k - 1] * lambda / k);
  return out;
}

/** Matrice des scores [buts domicile][buts extérieur] avec deux lois de Poisson indépendantes. */
export function scores(lDom, lExt) {
  const a = poissonLoi(lDom), b = poissonLoi(lExt);
  return a.map(x => b.map(y => x * y));
}

/** {dom, nul, ext, plus25, zeroZero, perdDomParUn, perdExtParUn} d'une matrice de scores. */
export function issues(m) {
  const r = {dom: 0, nul: 0, ext: 0, plus25: 0, zeroZero: m[0][0], perdDomParUn: 0, perdExtParUn: 0};
  for (let i = 0; i <= MAX_BUTS; i++) for (let j = 0; j <= MAX_BUTS; j++) {
    const p = m[i][j];
    if (i > j) r.dom += p; else if (i === j) r.nul += p; else r.ext += p;
    if (i + j >= 3) r.plus25 += p;
    if (j - i === 1) r.perdDomParUn += p;
    if (i - j === 1) r.perdExtParUn += p;
  }
  return r;
}

/**
 * Buts attendus (domicile, extérieur) qui reproduisent au mieux les probabilités justes 1N2 (et Plus de 2,5 buts si
 * donnée) : moindres carrés, grille puis affinage.
 */
export function ajusterPoisson({dom, nul, ext, plus25 = null}) {
  const ecart = (a, b) => {
    const r = issues(scores(a, b));
    let e = (r.dom - dom) ** 2 + (r.nul - nul) ** 2 + (r.ext - ext) ** 2;
    if (plus25 != null) e += (r.plus25 - plus25) ** 2;
    return e;
  };
  let best = {a: 1.3, b: 1.1, e: Infinity};
  for (let a = 0.1; a <= 4.5; a += 0.05) for (let b = 0.1; b <= 4.5; b += 0.05) {
    const e = ecart(a, b);
    if (e < best.e) best = {a, b, e};
  }
  for (let pas = 0.01; pas >= 0.0005; pas /= 4) {
    let mieux = true;
    while (mieux) {
      mieux = false;
      for (const [da, db] of [[pas, 0], [-pas, 0], [0, pas], [0, -pas]]) {
        const a = best.a + da, b = best.b + db;
        if (a <= 0.02 || b <= 0.02) continue;
        const e = ecart(a, b);
        if (e < best.e - 1e-15) { best = {a, b, e}; mieux = true; }
      }
    }
  }
  return {lDom: best.a, lExt: best.b, erreur: Math.sqrt(best.e / (plus25 == null ? 3 : 4))};
}

/**
 * Probabilité qu'une équipe (taux de buts `lPour`, adversaire `lContre`, sur tout le match) mène de `ecart` buts à un
 * moment du match, et probabilité qu'elle gagne sans jamais avoir mené ainsi. Taux constants pendant le match
 * (simplification), découpage en `pas` instants.
 */
export function probaMenerDe(lPour, lContre, ecart = 2, pas = 3000) {
  const bas = -MAX_BUTS * 2, n = ecart - bas;         // états : écart de bas à ecart - 1, absorbant à ecart
  let dist = new Float64Array(n);
  dist[-bas] = 1;
  let atteint = 0;
  const pp = lPour / pas, pc = lContre / pas, reste = 1 - pp - pc;
  for (let t = 0; t < pas; t++) {
    const suiv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = dist[i];
      if (!p) continue;
      suiv[i] += p * reste;
      if (i + 1 === n) atteint += p * pp; else suiv[i + 1] += p * pp;
      if (i > 0) suiv[i - 1] += p * pc; else suiv[i] += p * pc;
    }
    dist = suiv;
  }
  let gagneSans = 0;
  for (let i = -bas + 1; i < n; i++) gagneSans += dist[i];
  return {mene: atteint, gagneSansMener: gagneSans};
}

export const PROMOS = {
  nul: 'Remboursé si match nul',
  defaiteUn: 'Remboursé si défaite par un but d’écart',
  zeroZero: 'Remboursé si 0-0',
  perdant: 'Remboursé si le pari perd',
  deuxButs: 'Payé dès 2 buts d’avance',
  autre: 'Remboursé si un autre événement (probabilité à saisir)',
};

/**
 * Valeur d'une promo sur un pari « victoire de `camp` » (« dom » ou « ext ») à `cote`, mise `mise`.
 * `pinnacle` : cotes 1, N, 2 (et Plus / Moins de 2,5 facultatifs). `remboursement` : valeur d'1 € remboursé (1 en
 * argent, ~0,75 en freebet), `plafond` en €. `pAutre` : probabilité de l'événement remboursé (type « autre »),
 * supposé arriver quand le pari perd.
 * Rend {ev, evSansPromo, pGagne, pRembourse, lDom, lExt}, ev en € pour la mise.
 */
export function valeurPromo({type, camp, cote, mise, pinnacle, remboursement = 1, plafond = Infinity, pAutre = null}) {
  if (!valide(cote) || !(mise > 0) || !['dom', 'ext'].includes(camp) || !(type in PROMOS)) return null;
  const p1n2 = probasJustes([pinnacle.dom, pinnacle.nul, pinnacle.ext]);
  if (!p1n2) return null;
  const pOU = Number.isFinite(pinnacle.plus25) && Number.isFinite(pinnacle.moins25)
    ? probasJustes([pinnacle.plus25, pinnacle.moins25]) : null;
  const fit = ajusterPoisson({dom: p1n2[0], nul: p1n2[1], ext: p1n2[2], plus25: pOU ? pOU[0] : null});
  const r = issues(scores(fit.lDom, fit.lExt));
  // Probabilités justes de Pinnacle pour le pari lui-même ; le modèle sert aux événements que Pinnacle ne cote pas.
  const pGagne = camp === 'dom' ? p1n2[0] : p1n2[2];
  const evSansPromo = mise * (cote * pGagne - 1);
  const rendu = remboursement * Math.min(mise, plafond > 0 ? plafond : Infinity);
  let pRembourse = 0, pPaye = pGagne;
  if (type === 'nul') pRembourse = p1n2[1];
  else if (type === 'defaiteUn') pRembourse = camp === 'dom' ? r.perdDomParUn : r.perdExtParUn;
  else if (type === 'zeroZero') pRembourse = r.zeroZero;
  else if (type === 'perdant') pRembourse = 1 - pGagne;
  else if (type === 'autre') {
    if (!(pAutre >= 0 && pAutre <= 1 - pGagne)) return null;
    pRembourse = pAutre;
  } else if (type === 'deuxButs') {
    const [pour, contre] = camp === 'dom' ? [fit.lDom, fit.lExt] : [fit.lExt, fit.lDom];
    const m = probaMenerDe(pour, contre, 2);
    pPaye = m.mene + m.gagneSansMener;               // modèle : mené de 2, ou gagné d'un but sans jamais mener de 2
    // le modèle donne sa propre victoire ; on garde l'écart entre « payé » et « gagné » du modèle, appliqué à Pinnacle
    const gagneModele = camp === 'dom' ? r.dom : r.ext;
    pPaye = Math.min(1, pGagne + (pPaye - gagneModele));
  }
  const ev = type === 'deuxButs' ? mise * (cote * pPaye - 1) : evSansPromo + pRembourse * rendu;
  return {ev, evSansPromo, pGagne, pRembourse, pPaye, lDom: fit.lDom, lExt: fit.lExt, erreur: fit.erreur};
}

// --------------------------------------------------------------------------- 6. combiné boosté

/**
 * Probabilité juste d'une sélection : cotes Pinnacle de tout le marché (`pinnacle`, ta sélection d'abord, au moins deux
 * issues, marge retirée en proportion), ou cote juste déjà calculée (`juste`, par exemple celle de Cote Value).
 */
export function probaSelection({pinnacle = null, juste = null}) {
  if (pinnacle) {
    if (pinnacle.length < 2) return null;
    const p = probasJustes(pinnacle);
    return p ? p[0] : null;
  }
  return valide(juste) ? 1 / juste : null;
}

/**
 * Combiné boosté : `selections` de matchs différents (indépendantes), chacune {cote, proba} (cote du bookmaker,
 * probabilité juste). Bonus `bonus` (0,25 = +25 %) sur le gain net (`base` « net ») ou sur le retour total
 * (« total »), payé en argent ou en freebet compté à `valeur` (1 = argent), s'il passe.
 * Rend {cote, proba, retour, esperance, rendement, sansBonus, bonusMinimal}, esperance en € pour la mise.
 */
export function combineBooste({selections, mise, bonus, base = 'net', valeur = 1}) {
  if (!(mise > 0) || !(bonus >= 0) || !(valeur >= 0 && valeur <= 1) || !['net', 'total'].includes(base)) return null;
  if (!selections || selections.length < 2 || !selections.every(s => valide(s.cote) && s.proba > 0 && s.proba < 1)) return null;
  const cote = selections.reduce((a, s) => a * s.cote, 1);
  const proba = selections.reduce((a, s) => a * s.proba, 1);
  const assiette = base === 'total' ? cote : cote - 1;          // ce que le bonus multiplie, par € misé
  const retour = mise * (cote + assiette * bonus * valeur);     // s'il passe
  const esperance = proba * retour - mise;
  // bonus qui annule l'espérance : proba × (cote + assiette × b × valeur) = 1
  const bonusMinimal = assiette * valeur > 0 ? Math.max(0, (1 / proba - cote) / (assiette * valeur)) : Infinity;
  return {cote, proba, retour, esperance, rendement: esperance / mise, sansBonus: proba * cote - 1, bonusMinimal};
}

// --------------------------------------------------------------------------- en direct (euria.json de Cote Value)

/**
 * Paris relevés par Cote Value pour convertir une offre chez un bookmaker (`candidats` : euria.json › conversions),
 * recalculés pour l'offre saisie : « freebet » (montant du freebet) ou « rembourse » (mise, remboursement compté
 * `valeur`, plafonné à `plafond`). Le premier pari de chaque candidat est celui de l'offre, les autres le couvrent.
 * Rend les candidats avec leur calcul (`calcul`), du plus gros gain garanti au plus petit.
 */
export function classerConversions(candidats, {type = 'freebet', montant, valeur = 0.75, plafond = Infinity}) {
  const out = [];
  for (const c of candidats || []) {
    const [offre, ...couv] = c.jambes || [];
    if (!offre || !couv.length) continue;
    const cotes = couv.map(j => j.cote);
    const calcul = type === 'freebet' ? couvrirFreebet({montant, cote: offre.cote, couvertures: cotes})
      : couvrirRembourse({mise: montant, cote: offre.cote, couvertures: cotes, valeur, plafond});
    if (calcul) out.push({...c, calcul});
  }
  return out.sort((a, b) => b.calcul.gain - a.calcul.gain);
}

/** Un milieu relevé par Cote Value (euria.json › milieux), recalculé pour `total` euros (mises équilibrées). */
export function milieuRecalcule(m, total) {
  const [a, b] = m.jambes || [];
  if (!a || !b || !Array.isArray(m.bornes)) return null;
  return milieu({ligne1: m.bornes[0], cote1: a.cote, ligne2: m.bornes[1], cote2: b.cote, total, mode: 'equilibre',
                 pMilieu: {dessous: m.p_dessous, milieu: m.p_milieu, dessus: m.p_dessus}});
}
