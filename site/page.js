// Liaison des formulaires aux calculs (calculs.js) : tout est recalculé à chaque saisie, rien n'est envoyé.
import {
  lireNombre, couvrirFreebet, couvrirRembourse, milieu, probasMilieuDeuxLignes, probasMilieuNormale, valeurPromo,
  PROMOS,
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

// --------------------------------------------------------------------------- liaison

$('pr-type').innerHTML = Object.entries(PROMOS).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
for (const [zone, calcul] of [['freebets', freebets], ['milieux', milieux], ['promos', promos]]) {
  const section = $(zone);
  section.addEventListener('input', calcul);
  section.addEventListener('change', calcul);
  calcul();
}
