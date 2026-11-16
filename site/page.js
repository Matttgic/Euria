// Liaison des formulaires aux calculs (calculs.js) : tout est recalculé à chaque saisie, rien n'est envoyé.
import {
  lireNombre, couvrirFreebet, couvrirRembourse, milieu, probasMilieuDeuxLignes, probasMilieuNormale, valeurPromo,
  PROMOS, probaSelection, combineBooste,
} from './calculs.js';

const $ = id => document.getElementById(id);
const n = id => lireNombre($(id).value);
const euros = x => (x < 0 ? '−' : '') + Math.abs(x).toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' €';
const signe = x => (x > 0 ? '+' : '') + euros(x);
const pct = (x, d = 1) => (x < 0 ? '−' : '') + Math.abs(x * 100).toLocaleString('fr-FR', {minimumFractionDigits: d, maximumFractionDigits: d}) + ' %';
const classe = x => x > 0.005 ? 'pos' : x < -0.005 ? 'neg' : '';
const cote = x => x.toLocaleString('fr-FR', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const montrer = (selecteur, oui) => document.querySelectorAll(selecteur).forEach(el => { el.hidden = !oui; });
const ligne = (titre, valeur, cls = '') => `<div class="ligne"><span>${titre}</span><b class="${cls}">${valeur}</b></div>`;
const manque = texte => `<p class="manque">${texte}</p>`;

// --------------------------------------------------------------------------- 1. freebets

function freebets() {
  const type = document.querySelector('input[name="fb-type"]:checked').value;
  const deux = $('fb-issues').value === '2';
  montrer('.si-2', deux);
  montrer('.si-freebet', type === 'freebet');
  montrer('.si-rembourse', type === 'rembourse');
  montrer('.si-fbrembourse', type === 'rembourse' && $('fb-forme').value === 'freebet');
  $('fb-montant-nom').textContent = type === 'freebet' ? 'du freebet' : 'de ta mise';
  const couvertures = deux ? [n('fb-c1'), n('fb-c2')] : [n('fb-c1')];
  const cote1 = n('fb-cote'), montant = n('fb-montant');
  let r, html = '';
  if (type === 'freebet') {
    r = couvrirFreebet({montant, cote: cote1, couvertures, miseRendue: $('fb-rendue').checked});
    if (!r) { $('fb-res').innerHTML = manque('Saisis le montant et des cotes supérieures à 1.'); return; }
    html += ligne(`Freebet de ${euros(montant)} sur A à ${cote(cote1)}`, `retour ${euros(r.retourSiA)} s’il gagne`);
  } else {
    const valeur = $('fb-forme').value === 'argent' ? 1 : n('fb-valeur') / 100;
    const plafond = Number.isFinite(n('fb-plafond')) ? n('fb-plafond') : Infinity;
    r = couvrirRembourse({mise: montant, cote: cote1, couvertures, valeur, plafond});
    if (!r) { $('fb-res').innerHTML = manque('Saisis la mise, des cotes supérieures à 1 et une valeur de freebet entre 0 et 100 %.'); return; }
    html += ligne(`Mise ${euros(montant)} sur A à ${cote(cote1)}`, `remboursement compté ${euros(r.rendu)}`);
  }
  r.mises.forEach((m, i) => { html += ligne(`Couverture ${i + 1} à ${cote(couvertures[i])}`, euros(m)); });
  html += `<div class="verdict ${classe(r.gain)}">Gain garanti : ${signe(r.gain)} quoi qu’il arrive</div>`;
  if (type === 'freebet') html += ligne('Conversion du freebet', pct(r.conversion, 0), classe(r.gain));
  else html += ligne('Argent engagé', euros(r.miseTotale));
  $('fb-res').innerHTML = html;
}

// --------------------------------------------------------------------------- 2. milieux

function milieux() {
  const methode = $('mi-proba').value;
  montrer('.si-deux', methode === 'deux');
  montrer('.si-normale', methode === 'normale');
  montrer('.si-sigma', methode === 'normale' && $('mi-sigma').value === 'autre');
  const l1 = n('mi-l1'), l2 = n('mi-l2');
  let pMilieu = null, note = '';
  if (methode === 'deux') {
    pMilieu = probasMilieuDeuxLignes({plus1: n('mi-p1p'), moins1: n('mi-p1m'), plus2: n('mi-p2p'), moins2: n('mi-p2m')});
    if (!pMilieu) note = manque('Saisis les quatre cotes Pinnacle (Plus et Moins aux deux lignes).');
  } else if (methode === 'normale') {
    const sigma = $('mi-sigma').value === 'autre' ? n('mi-sigma-autre') : Number($('mi-sigma').value);
    pMilieu = probasMilieuNormale({ligne: n('mi-nl'), plus: n('mi-np'), moins: n('mi-nm'), sigma, ligne1: l1, ligne2: l2});
    if (!pMilieu) note = manque('Saisis la ligne Pinnacle, ses cotes Plus et Moins, et un écart-type.');
  }
  const r = milieu({ligne1: l1, cote1: n('mi-c1'), ligne2: l2, cote2: n('mi-c2'), total: n('mi-total'),
                    mode: $('mi-mode').value, pMilieu});
  if (!r) { $('mi-res').innerHTML = manque('Saisis les deux lignes, leurs cotes et la somme à miser.'); return; }
  if (r.erreur) { $('mi-res').innerHTML = manque(r.erreur); return; }
  let html = ligne(`Mise « Plus de ${String(l1).replace('.', ',')} »`, euros(r.mises[0]))
    + ligne(`Mise « Moins de ${String(l2).replace('.', ',')} »`, euros(r.mises[1]))
    + ligne(`Sous ${String(l1).replace('.', ',')}`, signe(r.dessous), classe(r.dessous))
    + ligne(`Entre les deux (le milieu)`, signe(r.dedans), classe(r.dedans))
    + ligne(`Au-dessus de ${String(l2).replace('.', ',')}`, signe(r.dessus), classe(r.dessus));
  html += r.seuil > 0
    ? `<div class="verdict">Rentable si le milieu tombe plus d’une fois sur ${(1 / r.seuil).toLocaleString('fr-FR', {maximumFractionDigits: 1})} (${pct(r.seuil)})</div>`
    : `<div class="verdict pos">Gagnant dans tous les cas : c’est aussi un surebet</div>`;
  if (pMilieu) {
    html += ligne('Probabilité du milieu', pct(pMilieu.milieu));
    if (r.esperance != null) {
      const mise = r.mises[0] + r.mises[1];
      html += `<div class="verdict ${classe(r.esperance)}">Espérance : ${signe(r.esperance)} (${pct(r.esperance / mise)} de la mise) · ${r.esperance > 0 ? 'à jouer' : 'à laisser'}</div>`;
    }
  }
  $('mi-res').innerHTML = html + note;
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
    $('pr-res').innerHTML = manque(type === 'autre'
      ? 'Saisis les cotes, la mise et une probabilité d’événement possible (au plus la probabilité que le pari perde).'
      : 'Saisis les trois cotes Pinnacle, la cote de ton bookmaker et la mise.');
    return;
  }
  const mise = n('pr-mise');
  let html = ligne('Probabilité juste de victoire (Pinnacle)', pct(r.pGagne))
    + ligne('Sans la promo', `${signe(r.evSansPromo)} (${pct(r.evSansPromo / mise)})`, classe(r.evSansPromo));
  if (type === 'deuxButs') html += ligne('Payé (avance de 2 buts ou victoire)', `au moins ${pct(r.pPaye)}`);
  else html += ligne('Probabilité d’être remboursé', pct(r.pRembourse));
  html += `<div class="verdict ${classe(r.ev)}">Avec la promo : ${signe(r.ev)} (${pct(r.ev / mise)}) · ${r.ev > 0 ? 'à jouer' : 'à laisser'}</div>`;
  html += `<p class="note">Buts attendus du modèle : ${r.lDom.toLocaleString('fr-FR', {maximumFractionDigits: 2})} – ${r.lExt.toLocaleString('fr-FR', {maximumFractionDigits: 2})}${r.erreur > 0.01 ? ' (ajustement imparfait : vérifie les cotes saisies)' : ''}.</p>`;
  $('pr-res').innerHTML = html;
}

// --------------------------------------------------------------------------- 4. combiné boosté

// Saisies gardées quand on change le nombre de sélections ou le mode :
// [cote du bookmaker, Pinnacle sélection, Pinnacle autre issue, Pinnacle 3e issue, cote juste].
const cbSaisies = [['1,85', '1,95', '1,95', '', ''], ['2,10', '2,20', '3,50', '3,60', ''], ['1,60', '1,66', '2,35', '', '']];
const CB_CHAMPS = {
  pinnacle: [[0, 'Ton bookmaker'], [1, 'Pinnacle : ta sélection'], [2, 'Pinnacle : autre issue'], [3, 'Pinnacle : 3e issue']],
  juste: [[0, 'Cote chez ton bookmaker'], [4, 'Cote juste']],
};
const attr = v => String(v).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

function lignesCombine() {
  const zone = $('cb-lignes'), mode = $('cb-mode').value, nombre = Number($('cb-nombre').value);
  const cle = `${mode}|${nombre}`;
  if (zone.dataset.cle === cle) return;
  const ancien = CB_CHAMPS[zone.dataset.mode];
  if (ancien) zone.querySelectorAll('.cb-ligne').forEach((l, i) => {
    l.querySelectorAll('input').forEach((x, k) => { cbSaisies[i][ancien[k][0]] = x.value; });
  });
  while (cbSaisies.length < nombre) cbSaisies.push(['', '', '', '', '']);
  const champs = CB_CHAMPS[mode];
  zone.dataset.cle = cle;
  zone.dataset.mode = mode;
  zone.classList.toggle('cb-mode-juste', mode === 'juste');
  zone.innerHTML = `<div class="cb-tete" aria-hidden="true">${champs.map(([, t]) => `<span>${t}</span>`).join('')}</div>`
    + Array.from({length: nombre}, (_, i) => `<div class="cb-ligne">${champs.map(([k, t]) =>
      `<input inputmode="decimal" aria-label="Sélection ${i + 1}, ${t}" value="${attr(cbSaisies[i][k])}">`).join('')}</div>`).join('');
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
    $('cb-res').innerHTML = manque(`Sélection${manquantes.length > 1 ? 's' : ''} ${manquantes.join(', ')} : saisis la cote de ton bookmaker et ${
      juste ? 'la cote juste' : 'au moins deux cotes Pinnacle du marché, ta sélection d’abord'}.`);
    return;
  }
  const valeur = $('cb-forme').value === 'argent' ? 1 : n('cb-valeur') / 100;
  const r = combineBooste({selections, mise: n('cb-mise'), bonus: n('cb-bonus') / 100, base: $('cb-base').value, valeur});
  if (!r) { $('cb-res').innerHTML = manque('Saisis le bonus, la mise et une valeur de freebet entre 0 et 100 %.'); return; }
  let html = selections.map((s, i) => {
    const v = s.cote * s.proba - 1;
    return ligne(`Sélection ${i + 1} : ${cote(s.cote)} pour une cote juste de ${cote(1 / s.proba)}`, (v > 0 ? '+' : '') + pct(v), classe(v));
  }).join('');
  html += ligne('Cote du combiné', cote(r.cote))
    + ligne('Chance qu’il passe', `${pct(r.proba)}, 1 fois sur ${(1 / r.proba).toLocaleString('fr-FR', {maximumFractionDigits: 1})}`)
    + ligne('Sans le bonus', pct(r.sansBonus), classe(r.sansBonus))
    + ligne('Bonus minimal pour être rentable', r.bonusMinimal === 0 ? 'aucun : déjà rentable'
      : Number.isFinite(r.bonusMinimal) ? '+' + pct(r.bonusMinimal) : '—')
    + ligne('S’il passe', `retour ${euros(r.retour)}`);
  html += `<div class="verdict ${classe(r.esperance)}">Avec le bonus : ${signe(r.esperance)} (${pct(r.rendement)}) · ${r.esperance > 0 ? 'à jouer' : 'à laisser'}</div>`;
  html += `<p class="note">Rentable en moyenne seulement : ce combiné perd ${pct(1 - r.proba, 0)} du temps.</p>`;
  $('cb-res').innerHTML = html;
}

// --------------------------------------------------------------------------- liaison

$('pr-type').innerHTML = Object.entries(PROMOS).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
for (const [zone, calcul] of [['freebets', freebets], ['milieux', milieux], ['promos', promos], ['combine', combine]]) {
  const section = $(zone);
  section.addEventListener('input', calcul);
  section.addEventListener('change', calcul);
  calcul();
}
