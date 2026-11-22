// Partie « en direct » : lit euria.json, publié par Cote Value à chaque relevé (même origine que cette page,
// https://matttgic.github.io), et affiche les milieux en value et les meilleures conversions de freebets.
// Les mises sont recalculées ici pour les montants saisis (calculs.js). ?donnees=<adresse> remplace la source (essais).
import {lireNombre, classerConversions, milieuPlafonne} from './calculs.js';
import {nb, euros, signe, pct, classe, esc, ligne, cle, barre, remplir} from './affichage.js';

const SOURCE = new URLSearchParams(location.search).get('donnees') || 'https://matttgic.github.io/cotes-value/euria.json';
const RAFRAICHIR_MS = 3 * 60 * 1000;
const ANCIEN_MIN = 20;
const COTE_VALUE = 'https://matttgic.github.io/cotes-value/';
const SPORTS = {football: 'Football', basket: 'Basket', tennis: 'Tennis', hockey: 'Hockey', handball: 'Handball',
  rugby: 'Rugby', volley: 'Volley', football_americain: 'Football américain', baseball: 'Baseball'};

const $ = id => document.getElementById(id);
const date = iso => new Date(iso);
const valide = iso => Boolean(iso) && !isNaN(date(iso));
const heure = iso => date(iso).toLocaleTimeString('fr-FR', {hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris'});
const jour = iso => date(iso).toLocaleDateString('fr-FR', {day: '2-digit', month: '2-digit', timeZone: 'Europe/Paris'});
const quand = iso => valide(iso) ? date(iso).toLocaleString('fr-FR', {weekday: 'short', day: '2-digit', month: '2-digit',
  hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris'}) : '';
const sport = s => SPORTS[s] || (s ? s[0].toUpperCase() + s.slice(1) : '');
const numero = i => String(i + 1).padStart(2, '0');
const pluriel = (k, mot, s = 's') => `${k} ${mot}${k > 1 ? s : ''}`;
const lire = k => { try { return localStorage.getItem(k); } catch { return null; } };
const ecrire = (k, v) => { try { localStorage.setItem(k, v); } catch { /* stockage indisponible */ } };

let D = null;

/** Un pari à placer : bookmaker (lien vers le match quand Cote Value en a un), pari, cote, mise. */
function jambe(j, role, mise) {
  const nom = esc(j.bookmaker);
  const book = typeof j.lien === 'string' && j.lien.startsWith('https://') && !j.lien_repli
    ? `<a href="${esc(j.lien)}" target="_blank" rel="noopener" title="Ouvrir le match chez ${nom}">${nom}</a>` : nom;
  return `<tr><td class="qui"><span class="book">${book}${role ? ` <em>${role}</em>` : ''}</span>`
    + `<span class="pari">${esc(j.pari)}</span></td>`
    + `<td class="cote">${nb(j.cote)}${j.cote_juste ? `<small>${j.estimee ? 'estimée' : 'juste'} ${nb(j.cote_juste)}</small>` : ''}</td>`
    + `<td class="mise">${euros(mise)}</td></tr>`;
}

const paris = lignes => '<table class="paris"><thead class="vh"><tr><th>Bookmaker et pari</th><th>Cote</th>'
  + `<th>Mise</th></tr></thead><tbody>${lignes}</tbody></table>`;

function entete(x, i) {
  return `<p class="op-meta"><span class="op-no">${numero(i)}</span>`
    + `<span><b>${esc(sport(x.sport))}</b>${x.ligue ? ', ' + esc(x.ligue) : ''}</span>`
    + `<time${valide(x.debut) ? ` datetime="${esc(x.debut)}"` : ''}>${esc(quand(x.debut))}</time></p>`
    + `<h3>${esc(x.match)}</h3>`;
}

const chiffre = (titre, x, precision) =>
  `<p class="op-fig"><span>${titre}</span><b class="${classe(x)}">${signe(x)}</b><small>${precision}</small></p>`;

/** Ce qui doit arriver pour que les deux paris gagnent. */
function zone(m) {
  const u = m.unite ? '\u00a0' + m.unite : '', plage = (a, b) => a === b ? `${a}` : `${a} à ${b}`;
  if (!m.ecart) return `le total fait ${plage(m.de, m.a)}${u}`;
  if (m.de >= 1) return `${m.domicile} gagne de ${plage(m.de, m.a)}${u} d’écart`;
  if (m.a <= -1) return `${m.exterieur} gagne de ${plage(-m.a, -m.de)}${u} d’écart`;
  return `l’écart ${m.domicile} − ${m.exterieur} fait ${plage(m.de, m.a)}${u}`;
}

function etat() {
  const o = D.autres || {}, s = o.surebets || 0, b = o.boosts_couverts || 0;
  $('autres').hidden = !(s || b);
  remplir($('autres'), s || b ? `<b>Gain garanti aussi dans Cote Value :</b> ${pluriel(s, 'surebet')} et `
    + `${pluriel(b, 'boost')} couvert${b > 1 ? 's' : ''} en ce moment. <a href="${COTE_VALUE}#boosts">Voir l’onglet Boosts</a>` : '');
  if (!valide(D.maj)) {
    $('etat').textContent = 'Pas encore de relevé publié par Cote Value.';
    $('nav-heure').textContent = '';
    return;
  }
  const min = Math.max(0, Math.floor((Date.now() - date(D.maj)) / 60000));
  const age = min < 1 ? 'à l’instant' : min < 60 ? `il y a ${min}\u00a0min`
    : min < 1440 ? `il y a ${Math.floor(min / 60)}\u00a0h\u00a0${String(min % 60).padStart(2, '0')}` : `le ${jour(D.maj)}`;
  const vieux = min > ANCIEN_MIN;
  remplir($('etat'), `Cotes françaises relevées par Cote Value à <b>${heure(D.maj)}</b>, ${age}. `
    + (vieux ? '<b class="vieux">Relevé ancien : les cotes ont pu bouger.</b>' : 'Vérifie chaque cote avant de miser.'));
  $('nav-heure').innerHTML = `<span class="nav-lg">Cotes de </span>${heure(D.maj)}`;
  $('nav-heure').classList.toggle('vieux', vieux);
}

function milieux() {
  if (!D) return;
  const total = lireNombre($('dm-total').value), perteMax = lireNombre($('dm-perte').value);
  const lignes = (D.milieux || []).map(m => ({m, r: milieuPlafonne(m, total > 0 ? total : 100, perteMax)}))
    .filter(x => x.r && !x.r.erreur)
    .sort((x, y) => y.r.esperance / y.r.total - x.r.esperance / x.r.total);   // meilleure espérance en % d'abord
  $('dm-compte').textContent = lignes.length ? `${lignes.length} en value` : 'aucun en ce moment';
  $('nb-milieux').textContent = lignes.length;
  $('nb-milieux').hidden = !lignes.length;
  const bilan = (D.bilan_milieux || {})['7'] || {};
  remplir($('dm-bilan'), bilan.n ? `Sur 7 jours : <b>${pluriel(bilan.n, 'milieu', 'x')}</b> en value `
    + `repéré${bilan.n > 1 ? 's' : ''}, espérance médiane <b>${euros(bilan.esperance_mediane)}</b> pour 100 €, `
    + `chance médiane du milieu <b>${pct(bilan.p_milieu_mediane)}</b>. Pas encore réglés : c’est une espérance, `
    + 'pas un résultat.' : '');
  if (!(total > 0)) { $('dm-liste').innerHTML = '<p class="manque">Saisis la somme à miser.</p>'; return; }
  remplir($('dm-liste'), lignes.length ? lignes.map(({m, r}, i) => {
    const [a, b] = m.jambes, hors = Math.min(r.dessous, r.dessus);
    const parts = [m.p_dessous, m.p_milieu, m.p_dessus];
    return `<article class="op">${entete(m, i)}${chiffre('Espérance', r.esperance, `${pct(r.esperance / r.total)} de la mise`)}
<p class="op-zone">Les deux paris gagnent si ${esc(zone(m))}.${m.estime ? ' Chance estimée par un modèle : Pinnacle n’a pas l’une des deux lignes.' : ''}</p>
${r.reduit ? `<p class="op-zone">Mises réduites à ${euros(r.total)} au total pour ne pas perdre plus de ${euros(perteMax)} hors du milieu.</p>` : ''}
${paris(jambe(a, '', r.mises[0]) + jambe(b, '', r.mises[1]))}
<div class="issues">${barre(parts, `Milieu : ${pct(m.p_milieu)} de chances, ${signe(r.dedans)}. `
      + `Sinon : ${pct(1 - m.p_milieu)}, au pire ${signe(hors)}.`)}
${ligne(`${cle(true)}Milieu <i class="p">${pct(m.p_milieu)}</i>`, signe(r.dedans), classe(r.dedans))}
${ligne(`${cle(false)}Sinon <i class="p">${pct(1 - m.p_milieu)}</i>`, `${signe(hors)} au pire`, classe(hors))}</div></article>`;
  }).join('')
    : '<p class="vide">Aucun milieu en value au dernier relevé. Il en faut deux bookmakers qui n’ont pas la même ligne. Au basket, ils affichent presque toujours les mêmes lignes avec environ 12 % de marge sur les totaux&nbsp;: un milieu y perd en moyenne.</p>');
}

const CONCLURE = 30;   // milieux réglés avant de tirer une conclusion

function bilan() {
  if (!D) return;
  const B = D.bilan_regle || {}, liste = D.regles || [], tout = B.tout || {};
  const attente = tout.en_attente ? ` ${pluriel(tout.en_attente, 'milieu')} en attente de résultat.` : '';
  if (!tout.n) {
    remplir($('bi-resume'), `<p class="vide">Aucun milieu réglé pour l’instant. Le premier bilan arrive après les premiers matchs.${attente}</p>`);
    $('bi-liste').innerHTML = '';
    return;
  }
  const bloc = (titre, b) => !b || !b.n ? '' : `<div class="bi-per"><h3>${titre}</h3>`
    + ligne(`Milieux réglés`, String(b.n))
    + ligne(`Résultat (100\u00a0€ par milieu)`, signe(b.resultat), classe(b.resultat))
    + ligne(`Espérance annoncée`, signe(b.esperance), classe(b.esperance))
    + ligne(`Milieux tombés`, `${b.tombes} (attendu ≈\u00a0${nb(b.tombes_attendus, 1)})`) + '</div>';
  const n = tout.n, verdict = n < CONCLURE
    ? `<b>Trop tôt pour conclure</b>&nbsp;: ${n}/${CONCLURE} milieux réglés. Un milieu tombe rarement, le résultat varie beaucoup au début.`
    : tout.resultat > 0
      ? `<b>Ça marche pour l’instant</b>&nbsp;: ${signe(tout.resultat)} sur ${n} milieux (espérance ${signe(tout.esperance)}).`
      : `<b>Pas concluant</b>&nbsp;: ${signe(tout.resultat)} sur ${n} milieux alors que l’espérance était ${signe(tout.esperance)}. Attends encore avant de miser.`;
  remplir($('bi-resume'), `<p class="bilan">${verdict}${attente}</p><div class="bi-grille">`
    + bloc('7 derniers jours', B['7']) + bloc('30 derniers jours', B['30']) + bloc('Depuis le début', tout) + '</div>');
  remplir($('bi-liste'), '<table class="bi-table"><thead><tr><th>Match</th><th>Espérance</th><th>Résultat</th></tr></thead><tbody>'
    + liste.map(e => `<tr><td><b>${esc(e.match)}</b><small>${esc(sport(e.sport))} · ${esc(quand(e.debut))} · `
      + `${esc(zone({...e, ecart: String(e.marche || '').endsWith('HANDICAP')}))}`
      + `${Array.isArray(e.score) ? ` · score ${e.score.map(Number).join('\u2011')}` : ''}</small></td>`
      + `<td class="${classe(e.esperance_initiale)}">${signe(e.esperance_initiale)}</td>`
      + `<td class="${classe(e.resultat_100)}">${signe(e.resultat_100)}${e.tombe ? '<small>milieu tombé</small>' : ''}</td></tr>`).join('')
    + '</tbody></table>');
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
  $('df-montant-nom').textContent = type === 'freebet' ? 'Montant du freebet' : 'Ta mise';
  if (!books.length) { $('df-liste').innerHTML = '<p class="vide">Aucune cote exploitable au dernier relevé.</p>'; return; }
  const montant = lireNombre($('df-montant').value), plafond = lireNombre($('df-plafond').value);
  const options = classerConversions(conv[sel.value] || [], {type, montant, valeur: lireNombre($('df-valeur').value) / 100,
    plafond: Number.isFinite(plafond) && plafond > 0 ? plafond : Infinity}).slice(0, 5);
  if (!options.length) {
    $('df-liste').innerHTML = '<p class="manque">Saisis un montant et une valeur de remboursement entre 0 et 100 %.</p>';
    return;
  }
  remplir($('df-liste'), options.map((c, i) => {
    const [offre, ...couv] = c.jambes, r = c.calcul;
    return `<article class="op">${entete(c, i)}${chiffre('Gain garanti', r.gain,
      type === 'freebet' ? `${pct(r.conversion, 0)} du freebet` : 'quoi qu’il arrive')}
<p class="op-zone">${esc(c.pari)}</p>
${paris(jambe(offre, type === 'freebet' ? 'ton freebet' : 'ta mise', montant)
      + couv.map((j, k) => jambe(j, 'couverture', r.mises[k])).join(''))}</article>`;
  }).join(''));
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
  bilan();
  freebets();
}

$('direct-milieux').addEventListener('input', milieux);
for (const ev of ['input', 'change']) {
  $('direct-freebets').addEventListener(ev, () => { ecrire('euria-book', $('df-book').value); freebets(); });
}
charger();
setInterval(() => { if (document.visibilityState === 'visible') charger(); }, RAFRAICHIR_MS);
setInterval(() => { if (D) etat(); }, 60 * 1000);   // l'âge du relevé avance même sans nouveau relevé
