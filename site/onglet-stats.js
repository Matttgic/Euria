// Onglet Stats : lit euria-stats.json (tous les milieux réglés, publié par Cote Value à côté d'euria.json) à la
// première ouverture de l'onglet, puis au plus toutes les 10 minutes. Filtres et découpages dans stats.js.
// ?stats=<adresse> remplace la source (essais).
import {lireNombre} from './calculs.js';
import {nb, signe, pct, classe, esc, ligne, remplir} from './affichage.js';
import {lignesStats, filtrer, agreger, decouper, valeurs, critereGroupe, DECOUPAGES, nomSport, nomMarche, nomPeriode}
  from './stats.js';

const P = new URLSearchParams(location.search);
const SOURCE = P.get('stats') || 'https://matttgic.github.io/cotes-value/euria-stats.json';
const RELIRE_MS = 10 * 60 * 1000;
const CONCLURE = 30;
const LISTES = {sport: nomSport, marche: nomMarche, periode: nomPeriode, bookmaker: null, ligue: null, reference: null};
const CHAMPS = {bookmaker: 'bookmakers'};
const NOMBRES = ['jours', 'largeurMin', 'largeurMax', 'esperanceMin', 'esperanceMax', 'pMin', 'pMax', 'avanceMin', 'avanceMax'];

const $ = id => document.getElementById(id);
const form = $('st-filtres');
let lignes = null, lu = 0, enCours = false;

function listes() {
  for (const [nom, libelle] of Object.entries(LISTES)) {
    const sel = form.elements[nom], vals = valeurs(lignes, CHAMPS[nom] || nom), avant = sel.value;
    const cle = vals.join('|');
    if (sel.dataset.liste === cle) continue;
    remplir(sel, '<option value="">tous</option>'
      + vals.map(v => `<option value="${esc(v)}">${esc(libelle ? libelle(v) : v)}</option>`).join(''));
    sel.value = vals.includes(avant) ? avant : '';
    sel.dataset.liste = cle;
  }
}

function criteres() {
  const f = {};
  for (const el of form.elements) {
    if (!el.name || el.value === '') continue;
    f[el.name] = NOMBRES.includes(el.name) ? lireNombre(el.value) : el.value;
  }
  for (const k of ['pMin', 'pMax']) if (Number.isFinite(f[k])) f[k] /= 100;
  return f;
}

const ecart = z => z == null ? '' : `${z > 0 ? '+' : z < 0 ? '−' : ''}${nb(Math.abs(z), 1)} σ`;
const tombes = g => g.attendus == null ? String(g.tombes) : `${g.tombes} / ${nb(g.attendus, 1)}`;

function resume(g) {
  if (!g.n) return '<p class="vide">Aucun milieu réglé avec ces critères.</p>';
  const verdict = g.n < CONCLURE
    ? `<b>Trop peu de milieux</b>&nbsp;: ${g.n}/${CONCLURE}, ne conclus rien.`
    : g.z != null && Math.abs(g.z) >= 2
      ? `<b>${g.z > 0 ? 'Mieux' : 'Moins bien'} que l’espérance</b>&nbsp;: écart de ${ecart(g.z)}, peu probable par hasard.`
      : g.z != null ? `<b>Conforme à l’espérance</b>&nbsp;: l’écart (${ecart(g.z)}) peut venir du hasard.`
        : `<b>Écart à l’espérance non mesurable</b>&nbsp;: gains des milieux inconnus.`;
  return `<p class="bilan">${verdict}</p><div class="st-chiffres">`
    + ligne('Milieux réglés', String(g.n))
    + ligne('Résultat (100 € par milieu)', signe(g.resultat), classe(g.resultat))
    + ligne('Rendement', pct(g.rendement), classe(g.rendement))
    + ligne('Espérance annoncée', `${signe(g.esperance)} (${pct(g.rendement_attendu)})`, classe(g.esperance))
    + ligne('Milieux tombés / attendus', tombes(g))
    + (g.z == null ? '' : ligne('Écart à l’espérance', ecart(g.z))) + '</div>';
}

function tableau(d, groupes) {
  if (!groupes.length) return '';
  const tete = '<thead><tr><th>Groupe</th><th>Milieux</th><th>Résultat</th><th>Rendement</th><th>Attendu</th>'
    + '<th>Tombés</th><th>Écart</th></tr></thead>';
  const corps = groupes.map(g => {
    const c = critereGroupe(d, g.cle), nom = esc(g.nom);
    const cellule = c ? `<button type="button" class="st-lien" data-critere="${esc(JSON.stringify(c))}">${nom}</button>` : nom;
    return `<tr${g.n < CONCLURE ? ' class="peu"' : ''}><td>${cellule}</td><td>${g.n}</td>`
      + `<td class="${classe(g.resultat)}">${signe(g.resultat)}</td><td class="${classe(g.rendement)}">${pct(g.rendement)}</td>`
      + `<td>${pct(g.rendement_attendu)}</td><td>${tombes(g)}</td>`
      + `<td class="${g.z != null && Math.abs(g.z) >= 2 ? classe(g.z) : ''}">${ecart(g.z)}</td></tr>`;
  }).join('');
  return `<div class="st-bloc"><h3>${esc(d.titre)}${d.sous ? ` <small>${esc(d.sous)}</small>` : ''}</h3>`
    + `<div class="st-defile"><table class="bi-table st-table">${tete}<tbody>${corps}</tbody></table></div></div>`;
}

function afficher() {
  if (!lignes) return;
  const choisis = filtrer(lignes, criteres());
  $('st-compte').textContent = `${choisis.length} / ${lignes.length} milieux`;
  remplir($('st-resume'), resume(agreger(choisis)));
  remplir($('st-groupes'), choisis.length
    ? DECOUPAGES.map(d => tableau(d, decouper(choisis, d))).join('')
      + '<p class="note">En gris&nbsp;: moins de 30 milieux, chiffres peu fiables.</p>'
    : '');
}

async function charger() {
  if (enCours || (lignes && Date.now() - lu < RELIRE_MS)) return;
  enCours = true;
  try {
    const r = await fetch(SOURCE, {cache: 'no-store'});
    if (!r.ok) throw new Error(String(r.status));
    lignes = lignesStats(await r.json());
    lu = Date.now();
    listes();
    afficher();
  } catch {
    if (!lignes) remplir($('st-resume'), '<p class="vide">Statistiques indisponibles pour l’instant (pas encore publiées par Cote Value, ou injoignable).</p>');
  } finally {
    enCours = false;
  }
}

const visible = () => !$('stats').classList.contains('cache');
function siVisible() { if (visible()) charger(); }

form.addEventListener('input', afficher);
form.addEventListener('change', afficher);
form.addEventListener('submit', e => e.preventDefault());
form.addEventListener('reset', () => setTimeout(afficher));
$('st-groupes').addEventListener('click', e => {
  const b = e.target.closest('[data-critere]');
  if (!b) return;
  for (const [k, v] of Object.entries(JSON.parse(b.dataset.critere))) if (form.elements[k]) form.elements[k].value = v;
  afficher();
  form.scrollIntoView({block: 'start', behavior: 'smooth'});
});
addEventListener('hashchange', () => setTimeout(siVisible));
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') siVisible(); });
setTimeout(siVisible);
