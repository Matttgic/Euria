// Une page par onglet : seul le groupe [data-page] de l'onglet choisi est affiché. L'adresse (#bilan…) garde l'onglet
// au rechargement ; une ancre interne à une page (#promos dans Calculs) ouvre la page puis descend jusqu'à elle.
const groupes = [...document.querySelectorAll('[data-page]')];
const onglets = [...document.querySelectorAll('.nav a[href^="#"]')];
const DEFAUT = 'milieux';

function afficher() {
  const id = decodeURIComponent(location.hash.slice(1));
  const cible = id ? document.getElementById(id) : null;
  const page = cible?.closest('[data-page]')?.dataset.page || DEFAUT;
  // #autres reste masqué par direct.js tant qu'il est vide : on ne touche qu'à l'onglet
  for (const g of groupes) g.classList.toggle('cache', g.dataset.page !== page);
  for (const a of onglets) {
    const p = document.getElementById(a.hash.slice(1))?.dataset.page;
    if (p === page) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
  const racine = cible && cible.dataset.page;   // l'onglet lui-même : haut de page
  if (cible && !racine) cible.scrollIntoView({block: 'start'});
  else window.scrollTo({top: 0, behavior: 'instant'});
}

addEventListener('hashchange', afficher);
afficher();
