// #method: the stage block in mid-screen lights its region of the pinned architecture figure; the rest is veiled.

// Stage regions on assets/img/method.webp (paper Fig. 2, 8:3), percent of the image [x0, y0, x1, y1].
// Measured on a 3200x1200 render of the figure; the phone crops use the same boxes.
const STAGES = {
  1: [[0.4, 25.4, 44.0, 74.6]],                                   // both photos, shared ViT encoders, image-token grids
  2: [[0.3, 0.0, 26.5, 24.4], [0.3, 75.6, 26.5, 100.0]],          // user clicks + tokenizer, view 1 (top) and view 2
  3: [[26.5, 0.0, 45.6, 24.4], [45.6, 10.0, 50.4, 41.5],          // adapter 1, its arrow and the (+) into view 1
      [26.5, 75.6, 45.6, 100.0], [45.6, 58.5, 50.4, 90.0]],       // adapter 2, its arrow and the (+) into view 2
  4: [[50.4, 0.0, 100.0, 100.0]],                                 // both decoders, heads, 3D reconstruction
};
const FULL = [[0, 0, 100, 100]];
const SLOTS = 4;  // every clip polygon has the same vertex count so clip-path can transition between stages

// One polygon for up to SLOTS rectangles: from an anchor, trace each rectangle clockwise and return along the same
// line (zero area), so the nonzero fill rule fills exactly the union of the rectangles.
function polygon(rects) {
  const rs = Array.from({ length: SLOTS }, (_, i) => rects[Math.min(i, rects.length - 1)]);
  const pts = ['0% 0%'];
  for (const [x0, y0, x1, y1] of rs) {
    pts.push(`${x0}% ${y0}%`, `${x1}% ${y0}%`, `${x1}% ${y1}%`, `${x0}% ${y1}%`, `${x0}% ${y0}%`, '0% 0%');
  }
  return `polygon(${pts.join(', ')})`;
}

function init(root) {
  const img = root.querySelector('.method-fig__img');
  const lit = root.querySelector('.method-fig__lit');
  const stages = [...root.querySelectorAll('.method-stage')];
  const pills = [...root.querySelectorAll('.method-steps [data-stage]')];
  if (!img || !lit || stages.length === 0) return;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const wide = matchMedia('(min-width: 960px)');

  lit.style.clipPath = polygon(FULL);
  for (const [n, rects] of Object.entries(STAGES)) {
    const title = root.querySelector(`#method-stage-${n} .method-stage__title`);
    for (const [x0, y0, x1, y1] of rects) {
      const hit = document.createElement('button');
      hit.type = 'button';
      hit.className = 'method-fig__hit';
      hit.dataset.stage = n;
      hit.tabIndex = -1;  // mouse shortcut only; the stage pills are the keyboard and screen-reader controls
      hit.setAttribute('aria-hidden', 'true');
      hit.title = `Stage ${n}: ${title ? title.textContent : ''}`;
      Object.assign(hit.style, { left: `${x0}%`, top: `${y0}%`, width: `${x1 - x0}%`, height: `${y1 - y0}%` });
      img.appendChild(hit);
    }
  }

  let active = -1;
  function set(n) {
    if (n === active) return;
    active = n;
    if (n) root.dataset.active = String(n);
    else delete root.dataset.active;
    lit.style.clipPath = polygon(n ? STAGES[n] : FULL);
    for (const s of stages) s.classList.toggle('is-active', Number(s.dataset.stage) === n);
    for (const p of pills) p.setAttribute('aria-pressed', String(Number(p.dataset.stage) === n));
  }

  // The stage overlapping the mid-screen band (48-52% of the viewport height, same as the observer's rootMargin) is
  // active; above the first stage nothing is (whole figure lit); below the last stage the last one stays.
  function pick() {
    const lo = innerHeight * 0.48;
    const hi = innerHeight * 0.52;
    let best = null;
    let most = 0;
    for (const s of stages) {
      const r = s.getBoundingClientRect();
      const overlap = Math.min(r.bottom, hi) - Math.max(r.top, lo);
      if (overlap > most) { best = s; most = overlap; }
    }
    if (best) set(Number(best.dataset.stage));
    else if (stages[0].getBoundingClientRect().top >= hi) set(0);
  }
  const band = new IntersectionObserver(pick, { rootMargin: '-48% 0px -48% 0px' });
  for (const s of stages) band.observe(s);
  addEventListener('resize', pick, { passive: true });
  addEventListener('scrollend', pick, { passive: true });
  set(0);
  pick();

  function go(n) {
    const s = root.querySelector(`#method-stage-${n}`);
    if (s && wide.matches) s.scrollIntoView({ block: 'center', behavior: reduce.matches ? 'auto' : 'smooth' });
  }
  root.addEventListener('click', (e) => {
    const t = e.target.closest('[data-stage]');
    if (!t || !root.contains(t)) return;
    if (t.classList.contains('method-stage') && Number(t.dataset.stage) === active) return;
    go(Number(t.dataset.stage));
  });
}

const section = document.getElementById('method');
if (section) init(section);
