// Partie « en direct » : lit euria.json, publié par Cote Value à chaque relevé (même origine que cette page,
// https://matttgic.github.io), et affiche les milieux en value et les meilleures conversions de freebets.
// Les mises sont recalculées ici pour les montants saisis (calculs.js). ?donnees=<adresse> remplace la source (essais).
import {lireNombre, classerConversions, milieuRecalcule} from './calculs.js';

const SOURCE = new URLSearchParams(location.search).get('donnees') || 'https://matttgic.github.io/cotes-value/euria.json';
const RAFRAICHIR_MS = 3 * 60 * 1000;
const ANCIEN_MIN = 20;
const COTE_VALUE = 'https://matttgic.github.io/cotes-value/';
const SPORTS = {football: 'Football', basket: 'Basket', tennis: 'Tennis', hockey: 'Hockey', handball: 'Handball',
  rugby: 'Rugby', volley: 'Volley', football_americain: 'Football américain', baseball: 'Baseball'};

const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const nb = (x, d = 2) => Number(x).toLocaleString('fr-FR', {minimumFractionDigits: d, maximumFractionDigits: d});
const euros = x => (x < 0 ? '−' : '') + nb(Math.abs(x)) + ' €';
const signe = x => (x > 0 ? '+' : '') + euros(x);
const pct = (x, d = 1) => (x < 0 ? '−' : '') + nb(Math.abs(100 * x), d) + ' %';
const classe = x => x > 0.005 ? 'pos' : x < -0.005 ? 'neg' : '';
const date = iso => new Date(iso);
const heure = iso => isNaN(date(iso)) ? '?' : date(iso).toLocaleTimeString('fr-FR', {hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris'});
const quand = iso => isNaN(date(iso)) ? '' : date(iso).toLocaleString('fr-FR', {weekday: 'short', day: '2-digit', month: '2-digit',
  hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris'});
const sport = s => SPORTS[s] || (s ? s[0].toUpperCase() + s.slice(1) : '');
const lire = k => { try { return localStorage.getItem(k); } catch { return null; } };
const ecrire = (k, v) => { try { localStorage.setItem(k, v); } catch { /* stockage indisponible */ } };
const lien = j => typeof j.lien === 'string' && j.lien.startsWith('https://') && !j.lien_repli
  ? `<a class="ouvrir" href="${esc(j.lien)}" target="_blank" rel="noopener">Ouvrir chez ${esc(j.bookmaker)}</a>` : '';

let D = null;

function jambe(j, titre, mise) {
  return `<div class="jambe"><span class="book">${esc(titre)}</span><b>${esc(j.pari)}</b>`
    + `<span class="cote">${nb(j.cote)}${j.cote_juste ? `<small>juste ${nb(j.cote_juste)}</small>` : ''}</span>`
    + `<span class="mise">${euros(mise)}</span>${lien(j)}</div>`;
}

function entete(x) {
  return `<p class="meta">${esc(sport(x.sport))}${x.ligue ? ' · ' + esc(x.ligue) : ''} · ${esc(quand(x.debut))}</p>`
    + `<h3>${esc(x.match)}</h3>`;
}

/** Ce qui doit arriver pour que les deux paris gagnent. */
function zone(m) {
  const u = m.unite ? ' ' + m.unite : '', plage = (a, b) => a === b ? `${a}` : `${a} à ${b}`;
  if (!m.ecart) return `le total fait ${plage(m.de, m.a)}${u}`;
  if (m.de >= 1) return `${m.domicile} gagne de ${plage(m.de, m.a)}${u} d’écart`;
  if (m.a <= -1) return `${m.exterieur} gagne de ${plage(-m.a, -m.de)}${u} d’écart`;
  return `l’écart ${m.domicile} − ${m.exterieur} fait ${plage(m.de, m.a)}${u}`;
}

function etat() {
  const age = (Date.now() - date(D.maj)) / 60000;
  $('etat').innerHTML = D.maj ? `Cotes relevées à ${heure(D.maj)}${age > ANCIEN_MIN
    ? ' · <b class="neg">relevé ancien : les cotes ont pu bouger</b>' : ''}. Vérifie chaque cote avant de miser.`
    : 'Pas encore de relevé publié par Cote Value.';
  const o = D.autres || {}, s = o.surebets || 0, b = o.boosts_couverts || 0;
  $('autres').innerHTML = s || b ? `En ce moment dans Cote Value, gain garanti aussi : ${s} surebet${s > 1 ? 's' : ''} et `
    + `${b} boost${b > 1 ? 's' : ''} couvert${b > 1 ? 's' : ''}. <a href="${COTE_VALUE}#boosts">Voir l’onglet Boosts</a>` : '';
}

function milieux() {
  if (!D) return;
  const total = lireNombre($('dm-total').value), liste = D.milieux || [];
  if (!(total > 0)) { $('dm-liste').innerHTML = '<p class="manque">Saisis la somme à miser.</p>'; return; }
  const cartes = liste.map(m => {
    const r = milieuRecalcule(m, total);
    if (!r || r.erreur) return '';
    const [a, b] = m.jambes, hors = Math.min(r.dessous, r.dessus);
    return `<article class="carte">${entete(m)}
<p class="zone">Les deux paris gagnent si ${esc(zone(m))} : ${pct(m.p_milieu)} de chances d’après Pinnacle.</p>
${jambe(a, a.bookmaker, r.mises[0])}${jambe(b, b.bookmaker, r.mises[1])}
<div class="ligne"><span>Dans le milieu</span><b class="${classe(r.dedans)}">${signe(r.dedans)}</b></div>
<div class="ligne"><span>Sinon, au pire</span><b class="${classe(hors)}">${signe(hors)}</b></div>
<div class="verdict ${classe(r.esperance)}">Espérance : ${signe(r.esperance)} (${pct(r.esperance / total)} de la mise)</div></article>`;
  }).filter(Boolean);
  $('dm-liste').innerHTML = cartes.length ? cartes.join('')
    : '<p class="vide">Aucun milieu en value au dernier relevé. Ils apparaissent surtout sur les totaux de points quand deux bookmakers n’ont pas la même ligne.</p>';
  const b = (D.bilan_milieux || {})['7'] || {};
  $('dm-bilan').textContent = b.n ? `Sur 7 jours : ${b.n} milieu${b.n > 1 ? 'x' : ''} en value repéré${b.n > 1 ? 's' : ''}, `
    + `espérance médiane ${nb(b.esperance_mediane)} € pour 100 €, chance médiane du milieu ${pct(b.p_milieu_mediane)}. `
    + 'Pas encore réglés : c’est une espérance, pas un résultat.' : '';
}

function freebets() {
  if (!D) return;
  const conv = D.conversions || {}, books = Object.keys(conv), sel = $('df-book');
  if (sel.dataset.liste !== books.join('|')) {
    const avant = sel.value || lire('euria-book');
    sel.innerHTML = books.map(b => `<option>${esc(b)}</option>`).join('');
    if (books.includes(avant)) sel.value = avant;
    sel.dataset.liste = books.join('|');
  }
  const type = $('df-type').value;
  document.querySelectorAll('.si-df-rembourse').forEach(el => { el.hidden = type !== 'rembourse'; });
  $('df-montant-nom').firstChild.textContent = type === 'freebet' ? 'Montant du freebet ' : 'Ta mise ';
  if (!books.length) { $('df-liste').innerHTML = '<p class="vide">Aucune cote exploitable au dernier relevé.</p>'; return; }
  const montant = lireNombre($('df-montant').value), plafond = lireNombre($('df-plafond').value);
  const options = classerConversions(conv[sel.value] || [], {type, montant, valeur: lireNombre($('df-valeur').value) / 100,
    plafond: Number.isFinite(plafond) && plafond > 0 ? plafond : Infinity}).slice(0, 5);
  if (!options.length) {
    $('df-liste').innerHTML = '<p class="manque">Saisis un montant et une valeur de remboursement entre 0 et 100 %.</p>';
    return;
  }
  $('df-liste').innerHTML = options.map(c => {
    const [offre, ...couv] = c.jambes, r = c.calcul;
    return `<article class="carte">${entete(c)}<p class="zone">${esc(c.pari)}</p>
${jambe(offre, `${offre.bookmaker} · ${type === 'freebet' ? 'ton freebet' : 'ta mise'}`, montant)}
${couv.map((j, i) => jambe(j, `${j.bookmaker} · couverture`, r.mises[i])).join('')}
<div class="verdict ${classe(r.gain)}">Gain garanti : ${signe(r.gain)} quoi qu’il arrive${
      type === 'freebet' ? ` (${pct(r.conversion, 0)} du freebet)` : ''}</div></article>`;
  }).join('');
}

async function charger() {
  try {
    const r = await fetch(SOURCE, {cache: 'no-store'});
    if (!r.ok) throw new Error(String(r.status));
    D = await r.json();
  } catch {
    if (!D) $('etat').textContent = 'Cotes indisponibles pour l’instant (Cote Value injoignable). Les calculs à la main restent utilisables.';
    return;
  }
  etat();
  milieux();
  freebets();
}

$('direct-milieux').addEventListener('input', milieux);
for (const ev of ['input', 'change']) {
  $('direct-freebets').addEventListener(ev, () => { ecrire('euria-book', $('df-book').value); freebets(); });
}
charger();
setInterval(() => { if (document.visibilityState === 'visible') charger(); }, RAFRAICHIR_MS);
