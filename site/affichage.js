// Mise en forme commune aux calculs à la main (page.js) et à la partie en direct (direct.js) : nombres à la
// française, lignes de résultat, total, barre des issues d'un milieu. Aucun calcul ici (voir calculs.js).

const INS = '\u00a0';                                  // espace insécable avant € et %
const arrondi = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

export const nb = (x, d = 2) => Number(x).toLocaleString('fr-FR', {minimumFractionDigits: d, maximumFractionDigits: d});
export const cote = x => nb(x);
export const euros = x => (arrondi(x, 2) < 0 ? '−' : '') + nb(Math.abs(x)) + INS + '€';
export const signe = x => (arrondi(x, 2) > 0 ? '+' : '') + euros(x);
export const pct = (x, d = 1) => (arrondi(100 * x, d) < 0 ? '−' : '') + nb(Math.abs(100 * x), d) + INS + '%';
export const fois = x => x.toLocaleString('fr-FR', {maximumFractionDigits: 1});
export const classe = x => x > 0.005 ? 'pos' : x < -0.005 ? 'neg' : '';
export const esc = v => String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

/** Ligne de résultat : intitulé, points de conduite, valeur. */
export const ligne = (titre, valeur, cls = '') =>
  `<div class="ligne"><span>${titre}</span><b class="${cls}">${valeur}</b></div>`;

/** Saisie manquante ou incohérente. */
export const manque = texte => `<p class="manque">${texte}</p>`;

/**
 * Le chiffre qui décide (gain garanti, espérance), souligné deux fois comme un total de compte. `x` donne le signe ;
 * `verdict` ajoute « à jouer » ou « à laisser ».
 */
export function total(titre, valeur, x = null, precision = '', verdict = false) {
  const cls = x == null ? '' : classe(x);
  const tampon = verdict ? `<span class="tampon ${cls}">${x > 0 ? 'à jouer' : 'à laisser'}</span>` : '';
  const bas = precision || tampon ? `<small>${precision}${tampon}</small>` : '';
  return `<div class="total"><span>${titre}</span><b class="${cls}">${valeur}</b>${bas}</div>`;
}

/** Carré de couleur qui relie une ligne à sa part de la barre (milieu en vert, le reste en gris). */
export const cle = milieu => `<i class="cle${milieu ? ' m' : ''}" aria-hidden="true"></i>`;

/**
 * Barre des issues d'un milieu : sous la 1re ligne, dans le milieu, au-dessus de la 2e, largeurs proportionnelles
 * aux probabilités. Les chiffres sont dans les lignes qui suivent (et dans le libellé lu par les lecteurs d'écran).
 */
export function barre([dessous, milieu, dessus], libelle) {
  if (![dessous, milieu, dessus].every(p => Number.isFinite(p) && p >= 0) || !(dessous + milieu + dessus > 0)) return '';
  const part = (p, m) => `<i${m ? ' class="m"' : ''} style="flex-grow:${p.toFixed(4)}"></i>`;
  return `<div class="barre" role="img" aria-label="${esc(libelle)}">${part(dessous)}${part(milieu, true)}${part(dessus)}</div>`;
}

/** Espaces insécables de la typographie française : « x », avant : ; ! ? et %. */
export const typo = s => String(s).replace(/« /g, '«\u00a0').replace(/ »/g, '\u00a0»')
  .replace(/ ([:;!?%])/g, '\u00a0$1');

/** Remplit un élément avec du HTML produit par la page, typographie française appliquée. */
export const remplir = (el, html) => { el.innerHTML = typo(html); };
