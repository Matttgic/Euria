import test from 'node:test';
import assert from 'node:assert/strict';
import {euros, signe, pct, typo, barre, total, esc} from '../site/affichage.js';

const INS = ' ';

test('montants : signe moins typographique, pas de « −0,00 » ni de « +0,00 »', () => {
  assert.equal(euros(-12.5), `−12,50${INS}€`);
  assert.equal(signe(12), `+12,00${INS}€`);
  assert.equal(signe(0.004), `0,00${INS}€`);
  assert.equal(euros(-0.004), `0,00${INS}€`);
  assert.equal(pct(-0.0004), `0,0${INS}%`);
  assert.equal(pct(0.0525), `5,3${INS}%`);
});

test('typographie française : espaces insécables', () => {
  assert.equal(typo('« Plus de » : 100 % ; ok ?'), `«${INS}Plus de${INS}»${INS}: 100${INS}%${INS}; ok${INS}?`);
  assert.equal(typo('https://a.fr/x?y=1'), 'https://a.fr/x?y=1');
});

test('barre des issues : parts proportionnelles, rien si une probabilité manque', () => {
  const html = barre([0.45, 0.08, 0.47], 'Milieu : 8 %');
  assert.deepEqual([...html.matchAll(/flex-grow:([\d.]+)/g)].map(m => Number(m[1])), [0.45, 0.08, 0.47]);
  assert.match(html, /class="m"/);
  assert.equal(barre([0.45, undefined, 0.47], ''), '');
  assert.equal(barre([0, 0, 0], ''), '');
});

test('total : verdict selon le signe, texte échappé ailleurs', () => {
  assert.match(total('Espérance', '+1,00 €', 1, '', true), /tampon pos">à jouer/);
  assert.match(total('Espérance', '−1,00 €', -1, '', true), /tampon neg">à laisser/);
  assert.equal(esc('<a href="x">&</a>'), '&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;');
});
