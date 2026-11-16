// Liaison des formulaires aux calculs (calculs.js) : tout est recalculé à chaque saisie, rien n'est envoyé.
import {
  lireNombre, couvrirFreebet, couvrirRembourse, milieu, probasMilieuDeuxLignes, probasMilieuNormale, valeurPromo,
  PROMOS, probaSelection, combineBooste,
} from './calculs.js';
import {nb, euros, signe, pct, cote, fois, classe, esc, ligne, manque, total, cle, barre, remplir} from './affichage.js';

const $ = id => document.getElementById(id);
const n = id => lireNombre($(id).value);
const montrer = (selecteur, oui) => document.querySelectorAll(selecteur).forEach(el => { el.hidden = !oui; });
const virgule = x => String(x).replace('.', ',');

// --------------------------------------------------------------------------- 1. freebets

function freebets() {
  const type = document.querySelector('input[name="fb-type"]:checked').value;
  const deux = $('fb-issues').value === '2';
  montrer('.si-2', deux);
  montrer('.si-freebet', type === 'freebet');
  montrer('.si-rembourse', type === 'rembourse');
  montrer('.si-fbrembourse', type === 'rembourse' && $('fb-forme').value === 'freebet');
  $('fb-montant-nom').textContent = type === 'freebet' ? 'Montant du freebet' : 'Ta mise';
  const couvertures = deux ? [n('fb-c1'), n('fb-c2')] : [n('fb-c1')];
  const cote1 = n('fb-cote'), montant = n('fb-montant');
  let r, html = '';
  if (type === 'freebet') {
    r = couvrirFreebet({montant, cote: cote1, couvertures, miseRendue: $('fb-rendue').checked});
    if (!r) { remplir($('fb-res'), manque('Saisis le montant et des cotes supérieures à 1.')); return; }
    html += ligne('Retour du freebet si A gagne', euros(r.retourSiA));
  } else {
    const valeur = $('fb-forme').value === 'argent' ? 1 : n('fb-valeur') / 100;
    const plafond = Number.isFinite(n('fb-plafond')) ? n('fb-plafond') : Infinity;
    r = couvrirRembourse({mise: montant, cote: cote1, couvertures, valeur, plafond});
    if (!r) { remplir($('fb-res'), manque('Saisis la mise, des cotes supérieures à 1 et une valeur de freebet entre 0 et 100 %.')); return; }
    html += ligne('Remboursement compté si A perd', euros(r.rendu));
  }
  r.mises.forEach((m, i) => { html += ligne(`Couverture ${i + 1} à ${cote(couvertures[i])}`, euros(m)); });
  if (type === 'rembourse') html += ligne('Argent engagé', euros(r.miseTotale));
  html += total('Gain garanti', signe(r.gain), r.gain,
    type === 'freebet' ? `quoi qu’il arrive, soit ${pct(r.conversion, 0)} du freebet` : 'quoi qu’il arrive');
  remplir($('fb-res'), html);
}

// --------------------------------------------------------------------------- 2. milieux

function milieux() {
  const methode = $('mi-proba').value;
  montrer('.si-deux', methode === 'deux');
  montrer('.si-normale', methode === 'normale');
  montrer('.si-sigma', methode === 'normale' && $('mi-sigma').value === 'autre');
  const l1 = n('mi-l1'), l2 = n('mi-l2');
  let p = null, note = '';
  if (methode === 'deux') {
    p = probasMilieuDeuxLignes({plus1: n('mi-p1p'), moins1: n('mi-p1m'), plus2: n('mi-p2p'), moins2: n('mi-p2m')});
    if (!p) note = manque('Saisis les quatre cotes Pinnacle (Plus et Moins aux deux lignes).');
  } else if (methode === 'normale') {
    const sigma = $('mi-sigma').value === 'autre' ? n('mi-sigma-autre') : Number($('mi-sigma').value);
    p = probasMilieuNormale({ligne: n('mi-nl'), plus: n('mi-np'), moins: n('mi-nm'), sigma, ligne1: l1, ligne2: l2});
    if (!p) note = manque('Saisis la ligne Pinnacle, ses cotes Plus et Moins, et un écart-type.');
  }
  const r = milieu({ligne1: l1, cote1: n('mi-c1'), ligne2: l2, cote2: n('mi-c2'), total: n('mi-total'),
                    mode: $('mi-mode').value, pMilieu: p});
  if (!r) { remplir($('mi-res'), manque('Saisis les deux lignes, leurs cotes et la somme à miser.')); return; }
  if (r.erreur) { remplir($('mi-res'), manque(r.erreur)); return; }
  // Avec une probabilité : barre des trois issues, chaque ligne marquée de sa couleur et de sa chance.
  const chance = x => p ? ` <i class="p">${pct(x)}</i>` : '';
  const marque = m => p ? cle(m) : '';
  let html = ligne(`Mise « Plus de ${virgule(l1)} »`, euros(r.mises[0]))
    + ligne(`Mise « Moins de ${virgule(l2)} »`, euros(r.mises[1]));
  if (p) {
    html += barre([p.dessous, p.milieu, p.dessus], `Sous ${virgule(l1)} : ${pct(p.dessous)}. Entre les deux : `
      + `${pct(p.milieu)}. Au-dessus de ${virgule(l2)} : ${pct(p.dessus)}.`);
  }
  html += ligne(`${marque(false)}Sous ${virgule(l1)}${chance(p?.dessous)}`, signe(r.dessous), classe(r.dessous))
    + ligne(`${marque(true)}Entre les deux (le milieu)${chance(p?.milieu)}`, signe(r.dedans), classe(r.dedans))
    + ligne(`${marque(false)}Au-dessus de ${virgule(l2)}${chance(p?.dessus)}`, signe(r.dessus), classe(r.dessus));
  const seuil = r.seuil > 0 ? fois(1 / r.seuil) : null;
  if (p && r.esperance != null) {
    const mise = r.mises[0] + r.mises[1];
    html += seuil
      ? ligne('Rentable si le milieu tombe plus de', `1 fois sur ${seuil}`)
      : ligne('Gagnant dans tous les cas', 'c’est aussi un surebet', 'pos');
    html += total('Espérance', signe(r.esperance), r.esperance, `${pct(r.esperance / mise)} de la mise`, true);
  } else {
    html += seuil
      ? total('Rentable si le milieu tombe plus de', `1 fois sur ${seuil}`, null, `soit ${pct(r.seuil)} des matchs`)
      : total('Gagnant dans tous les cas', 'surebet', 1, 'le milieu n’est qu’un bonus');
  }
  remplir($('mi-res'), html + note);
}

// --------------------------------------------------------------------------- 3. promos

function promos() {
  const type = $('pr-type').value;
  montrer('.si-autre', type === 'autre');
  montrer('.si-rembourse-promo', type !== 'deuxButs');
  montrer('.si-fb-promo', type !== 'deuxButs' && $('pr-forme').value === 'freebet');
  const remboursement = $('pr-forme').value === 'argent' ? 1 : n('pr-valeur') / 100;
  const plafond = Number.isFinite(n('pr-plafond')) ? n('pr-plafond') : Infinity;
  const pinnacle = {dom: n('pr-p1'), nul: n('pr-pn'), ext: n('pr-p2'), plus25: n('pr-po'), moins25: n('pr-pu')};
  const r = valeurPromo({type, camp: $('pr-camp').value, cote: n('pr-cote'), mise: n('pr-mise'), pinnacle,
                         remboursement, plafond, pAutre: n('pr-autre') / 100});
  if (!r) {
    remplir($('pr-res'), manque(type === 'autre'
      ? 'Saisis les cotes, la mise et une probabilité d’événement possible (au plus la probabilité que le pari perde).'
      : 'Saisis les trois cotes Pinnacle, la cote de ton bookmaker et la mise.'));
    return;
  }
  const mise = n('pr-mise');
  let html = ligne('Probabilité juste de victoire (Pinnacle)', pct(r.pGagne))
    + ligne('Sans la promo', `${signe(r.evSansPromo)} (${pct(r.evSansPromo / mise)})`, classe(r.evSansPromo));
  if (type === 'deuxButs') html += ligne('Payé (avance de 2 buts ou victoire)', `au moins ${pct(r.pPaye)}`);
  else html += ligne('Probabilité d’être remboursé', pct(r.pRembourse));
  html += total('Avec la promo', signe(r.ev), r.ev, `${pct(r.ev / mise)} de la mise`, true);
  html += `<p class="note">Buts attendus du modèle : ${nb(r.lDom)} – ${nb(r.lExt)}${
    r.erreur > 0.01 ? ' (ajustement imparfait : vérifie les cotes saisies)' : ''}.</p>`;
  remplir($('pr-res'), html);
}

// --------------------------------------------------------------------------- 4. combiné boosté

// Saisies gardées quand on change le nombre de sélections ou le mode :
// [cote du bookmaker, Pinnacle sélection, Pinnacle autre issue, Pinnacle 3e issue, cote juste].
const cbSaisies = [['1,85', '1,95', '1,95', '', ''], ['2,10', '2,20', '3,50', '3,60', ''], ['1,60', '1,66', '2,35', '', '']];
const CB_CHAMPS = {
  pinnacle: [[0, 'Ton bookmaker'], [1, 'Pinnacle : ta sélection'], [2, 'Pinnacle : autre issue'], [3, 'Pinnacle : 3e issue']],
  juste: [[0, 'Cote chez ton bookmaker'], [4, 'Cote juste']],
};

function lignesCombine() {
  const zone = $('cb-lignes'), mode = $('cb-mode').value, nombre = Number($('cb-nombre').value);
  const cleZone = `${mode}|${nombre}`;
  if (zone.dataset.cle === cleZone) return;
  const ancien = CB_CHAMPS[zone.dataset.mode];
  if (ancien) zone.querySelectorAll('.cb-ligne').forEach((l, i) => {
    l.querySelectorAll('input').forEach((x, k) => { cbSaisies[i][ancien[k][0]] = x.value; });
  });
  while (cbSaisies.length < nombre) cbSaisies.push(['', '', '', '', '']);
  const champs = CB_CHAMPS[mode];
  zone.dataset.cle = cleZone;
  zone.dataset.mode = mode;
  zone.classList.toggle('cb-mode-juste', mode === 'juste');
  remplir(zone, `<div class="cb-tete" aria-hidden="true"><span></span>${champs.map(([, t]) => `<span>${t}</span>`).join('')}</div>`
    + Array.from({length: nombre}, (_, i) => `<div class="cb-ligne"><span class="cb-no" aria-hidden="true">${i + 1}</span>${
      champs.map(([k, t]) => `<input inputmode="decimal" aria-label="Sélection ${i + 1}, ${t}" value="${esc(cbSaisies[i][k])}">`)
        .join('')}</div>`).join(''));
  montrer('.si-cb-pinnacle', mode === 'pinnacle');
}

function combine() {
  lignesCombine();
  const juste = $('cb-mode').value === 'juste';
  montrer('.si-cb-fb', $('cb-forme').value === 'freebet');
  const selections = [], manquantes = [];
  $('cb-lignes').querySelectorAll('.cb-ligne').forEach((l, i) => {
    const v = [...l.querySelectorAll('input')].map(x => x.value.trim());
    const coteBook = lireNombre(v[0]);
    const proba = juste ? probaSelection({juste: lireNombre(v[1])})
      : probaSelection({pinnacle: v.slice(1).filter(x => x !== '').map(lireNombre)});
    if (coteBook > 1 && proba != null) selections.push({cote: coteBook, proba}); else manquantes.push(i + 1);
  });
  if (manquantes.length) {
    remplir($('cb-res'), manque(`Sélection${manquantes.length > 1 ? 's' : ''} ${manquantes.join(', ')} : saisis la cote de ton bookmaker et ${
      juste ? 'la cote juste' : 'au moins deux cotes Pinnacle du marché, ta sélection d’abord'}.`));
    return;
  }
  const valeur = $('cb-forme').value === 'argent' ? 1 : n('cb-valeur') / 100;
  const r = combineBooste({selections, mise: n('cb-mise'), bonus: n('cb-bonus') / 100, base: $('cb-base').value, valeur});
  if (!r) { remplir($('cb-res'), manque('Saisis le bonus, la mise et une valeur de freebet entre 0 et 100 %.')); return; }
  let html = selections.map((s, i) => {
    const v = s.cote * s.proba - 1;
    return ligne(`Sélection ${i + 1} : ${cote(s.cote)}, juste ${cote(1 / s.proba)}`, (v > 0 ? '+' : '') + pct(v), classe(v));
  }).join('');
  html += ligne('Cote du combiné', cote(r.cote))
    + ligne('Chance qu’il passe', `${pct(r.proba)}, 1 fois sur ${fois(1 / r.proba)}`)
    + ligne('Sans le bonus', pct(r.sansBonus), classe(r.sansBonus))
    + ligne('Bonus minimal pour être rentable', r.bonusMinimal === 0 ? 'aucun : déjà rentable'
      : Number.isFinite(r.bonusMinimal) ? '+' + pct(r.bonusMinimal) : '—')
    + ligne('Retour s’il passe', euros(r.retour));
  html += total('Avec le bonus', signe(r.esperance), r.esperance, `${pct(r.rendement)} de la mise`, true);
  html += `<p class="note">Rentable en moyenne seulement : ce combiné perd ${pct(1 - r.proba, 0)} du temps.</p>`;
  remplir($('cb-res'), html);
}

// --------------------------------------------------------------------------- liaison

$('pr-type').innerHTML = Object.entries(PROMOS).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
for (const [zone, calcul] of [['freebets', freebets], ['milieux', milieux], ['promos', promos], ['combine', combine]]) {
  const section = $(zone);
  section.addEventListener('input', calcul);
  section.addEventListener('change', calcul);
  calcul();
}
