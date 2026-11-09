// #clicks: pose accuracy vs number of clicks K. Four small multiples (RRA@15, mAA@30 x in-distribution,
// out-of-distribution). Each panel draws on once when >= 55% visible (no-click baselines first, Click3R last),
// with a hover / tap / keyboard tooltip and a table view. Reduced motion shows the final state.
// Values: the paper's prompt-count ablation on Hard3DBench (2-view), rounded half-up to one decimal.

const KS = [0, 1, 2, 4, 8, 12];
const SERIES = {
  ours: { label: 'Click3R (ours)', short: 'Click3R', tiny: 'Click3R', cls: 'ours' },
  retrained: { label: 'DUSt3R (retrained)', short: 'DUSt3R (retrained)', tiny: 'Retrained', cls: 'retrained' },
  pi3: { label: 'Pi3 (224)', short: 'Pi3 (224)', tiny: 'Pi3', cls: 'ref' },
  vggt: { label: 'VGGT (224)', short: 'VGGT (224)', tiny: 'VGGT', cls: 'ref' },
  pretrained: { label: 'DUSt3R', short: 'DUSt3R', tiny: 'DUSt3R', cls: 'pretrained' },
};
const PANELS = [
  { metric: 'RRA@15', split: 'In-distribution',
    ours: [70.8, 73.5, 76.3, 77.0, 79.5, 78.8], base: { retrained: 70.5, pi3: 75.5, vggt: 59.9, pretrained: 47.8 } },
  { metric: 'mAA@30', split: 'In-distribution',
    ours: [55.3, 57.5, 60.4, 61.3, 61.8, 61.5], base: { retrained: 56.1, pi3: 60.7, vggt: 40.5, pretrained: 39.2 } },
  { metric: 'RRA@15', split: 'Out-of-distribution',
    ours: [77.0, 82.0, 85.7, 86.0, 88.7, 90.0], base: { retrained: 78.0, pi3: 93.7, vggt: 81.3, pretrained: 75.3 } },
  { metric: 'mAA@30', split: 'Out-of-distribution',
    ours: [50.8, 56.6, 60.1, 60.4, 62.4, 62.8], base: { retrained: 52.6, pi3: 71.1, vggt: 47.4, pretrained: 45.3 } },
];
// Draw-on timeline (ms): [start, duration] per series; labels fade in after their line is done.
const TL = { pretrained: [0, 650], vggt: [120, 650], pi3: [120, 650], retrained: [260, 700], ours: [950, 1300] };
const FADE = 280;
const TOTAL = TL.ours[0] + TL.ours[1] + FADE;
const BASE_KEYS = ['pretrained', 'vggt', 'pi3', 'retrained'];  // paint order, Click3R on top
const NS = 'http://www.w3.org/2000/svg';

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const fmt = (v) => v.toFixed(1);

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function swatch(key) {
  const s = SERIES[key];
  const dot = key === 'ours' ? '<circle class="clicks-m" cx="14" cy="6" r="4"/>' : '';
  return `<svg class="clicks-sw" viewBox="0 0 28 12" aria-hidden="true"><line class="clicks-l clicks-l--${s.cls}" x1="2" y1="6" x2="26" y2="6"/>${dot}</svg>`;
}
function niceStep(range) {
  const raw = range / 4;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
}

// Place direct labels at their line's height, pushed apart by at least `gap`, inside [top, bottom].
function spread(items, gap, top, bottom) {
  items.sort((a, b) => a.y0 - b.y0);
  items.forEach((it, i) => { it.y = Math.max(it.y0, i ? items[i - 1].y + gap : top); });
  for (let i = items.length - 1; i >= 0; i--) {
    const limit = i === items.length - 1 ? bottom : items[i + 1].y - gap;
    items[i].y = Math.min(items[i].y, limit);
  }
  return items;
}

function makePanel(P, idx, grid) {
  const fig = document.createElement('figure');
  fig.className = 'clicks-panel';
  fig.innerHTML = `<figcaption class="clicks-ptitle"><span class="clicks-metric">${P.metric}<span class="clicks-up" aria-hidden="true">↑</span></span>` +
    `<span class="clicks-split">${P.split}</span></figcaption>` +
    '<div class="clicks-plot"><div class="clicks-tip" hidden></div></div>';
  grid.appendChild(fig);
  const st = { P, idx, fig, box: fig.querySelector('.clicks-plot'), tip: fig.querySelector('.clicks-tip'),
    w: 0, svg: null, phase: 'idle', raf: 0, t0: 0, active: -1 };
  return st;
}

function build(st) {
  const { P, idx, box } = st;
  const w = Math.round(box.clientWidth);
  if (!w || w === st.w) return false;
  st.w = w;
  const narrow = w < 470;
  const fs = narrow ? 11 : 12;
  const M = { l: narrow ? 28 : 34, r: narrow ? 92 : 152, t: 12, b: narrow ? 38 : 42 };
  const H = Math.round(Math.min(Math.max(w * 0.6, 240), 330));
  const pw = w - M.l - M.r;
  const ph = H - M.t - M.b;
  const vals = P.ours.concat(Object.values(P.base));
  const step = niceStep(Math.max(...vals) - Math.min(...vals));
  const lo = Math.floor((Math.min(...vals) - step * 0.3) / step) * step;
  const hi = Math.ceil((Math.max(...vals) + step * 0.3) / step) * step;
  const x = (k) => M.l + (k / KS[KS.length - 1]) * pw;
  const y = (v) => M.t + ((hi - v) / (hi - lo)) * ph;
  st.x = x;
  st.M = M;

  if (st.svg) st.svg.remove();
  const desc = `${P.metric}, ${P.split.toLowerCase()}: Click3R ${P.ours.map((v, i) => `${fmt(v)} at K = ${KS[i]}`).join(', ')}; ` +
    BASE_KEYS.slice().reverse().map((k) => `${SERIES[k].label} ${fmt(P.base[k])}`).join(', ') + ' (no clicks).';
  const svg = el('svg', { class: 'clicks-svg', width: w, height: H, viewBox: `0 0 ${w} ${H}`, role: 'img', tabindex: 0,
    'aria-label': desc, style: `--fs:${fs}px` });
  box.insertBefore(svg, st.tip);
  st.svg = svg;
  const defs = el('defs', {}, svg);

  // Axes and grid (static)
  const gAx = el('g', { class: 'clicks-axes' }, svg);
  for (let v = lo; v <= hi + 1e-9; v += step) {
    el('line', { class: 'clicks-gl', x1: M.l, x2: M.l + pw, y1: y(v), y2: y(v) }, gAx);
    el('text', { class: 'clicks-tick', x: M.l - 7, y: y(v), dy: '0.35em', 'text-anchor': 'end' }, gAx).textContent = +v.toFixed(1);
  }
  el('line', { class: 'clicks-axis', x1: M.l, x2: M.l + pw, y1: M.t + ph, y2: M.t + ph }, gAx);
  KS.forEach((k) => {
    el('line', { class: 'clicks-axis', x1: x(k), x2: x(k), y1: M.t + ph, y2: M.t + ph + 4 }, gAx);
    el('text', { class: 'clicks-tick', x: x(k), y: M.t + ph + 17, 'text-anchor': 'middle' }, gAx).textContent = k;
  });
  el('text', { class: 'clicks-axis-title', x: M.l + pw / 2, y: H - 5, 'text-anchor': 'middle' }, gAx).textContent = 'Number of clicks K';

  // No-click baselines: horizontal lines revealed left to right by a growing clip rect (keeps their dash pattern)
  const gLines = el('g', {}, svg);
  st.base = BASE_KEYS.map((key) => {
    const id = `clicks-clip-${idx}-${key}`;
    const cp = el('clipPath', { id }, defs);
    const rect = el('rect', { x: M.l - 4, y: 0, width: 0, height: H }, cp);
    el('line', { class: `clicks-l clicks-l--${SERIES[key].cls}`, x1: M.l, x2: M.l + pw, y1: y(P.base[key]), y2: y(P.base[key]),
      'clip-path': `url(#${id})` }, gLines);
    return { key, rect, t0: TL[key][0], dur: TL[key][1], full: pw + 8 };
  });

  // Click3R: polyline drawn with stroke-dasharray, markers pop in as the line reaches them
  const pts = KS.map((k, i) => [x(k), y(P.ours[i])]);
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const len = cum[cum.length - 1];
  const poly = el('polyline', { class: 'clicks-l clicks-l--ours', points: pts.map((p) => p.join(',')).join(' '),
    style: `stroke-dasharray:${len + 2} ${len + 40}` }, svg);
  const R = narrow ? 4 : 4.5;
  const markers = pts.map((p) => el('circle', { class: 'clicks-m', cx: p[0], cy: p[1], r: 0 }, svg));
  st.ours = { poly, len, cum, markers, R, t0: TL.ours[0], dur: TL.ours[1] };

  // Direct labels in the right gutter, spread apart, with a short leader from the line end
  const items = spread(['ours', ...BASE_KEYS].map((key) => ({ key, y0: y(key === 'ours' ? P.ours[P.ours.length - 1] : P.base[key]) })),
    fs + 3, M.t + 2, M.t + ph + 6);
  const xr = M.l + pw;
  st.labels = {};
  items.forEach((it) => {
    const s = SERIES[it.key];
    const g = el('g', { class: `clicks-lab clicks-lab--${s.cls}`, style: 'opacity:0' }, svg);
    el('path', { class: 'clicks-leader', d: `M${xr + (it.key === 'ours' ? 7 : 3)},${it.y0} L${xr + 10},${it.y}` }, g);
    const t = el('text', { x: xr + 13, y: it.y, dy: '0.35em' }, g);
    const v = it.key === 'ours' ? P.ours[P.ours.length - 1] : P.base[it.key];
    el('tspan', { class: 'clicks-val' }, t).textContent = fmt(v);
    el('tspan', { class: 'clicks-name', dx: 5 }, t).textContent = narrow ? s.tiny : s.short;
    st.labels[it.key] = g;
  });

  // Hover layer: crosshair snapped to the nearest K
  st.cross = el('line', { class: 'clicks-cross', x1: 0, x2: 0, y1: M.t, y2: M.t + ph, visibility: 'hidden' }, svg);
  st.hit = el('rect', { class: 'clicks-hit', x: M.l - 10, y: 0, width: pw + 20, height: M.t + ph + 22 }, svg);
  bindHover(st);
  return true;
}

function frame(st, t) {
  st.base.forEach((b) => {
    b.rect.setAttribute('width', (easeOut(clamp01((t - b.t0) / b.dur)) * b.full).toFixed(1));
    st.labels[b.key].style.opacity = clamp01((t - b.t0 - b.dur) / FADE).toFixed(3);
  });
  const o = st.ours;
  const f = easeInOut(clamp01((t - o.t0) / o.dur));
  const d = f * o.len;
  o.poly.style.strokeDashoffset = (o.len + 2 - d).toFixed(1);
  o.poly.style.opacity = f > 0 ? 1 : 0;
  o.markers.forEach((m, j) => {
    // a marker grows over the last 14 px before the line reaches it
    const a = j === st.active ? 1 : f > 0 ? clamp01((d - o.cum[j] + 14) / 14) : 0;
    m.setAttribute('r', ((j === st.active ? 1.4 : 1) * o.R * easeOut(a)).toFixed(2));
  });
  st.labels.ours.style.opacity = clamp01((t - o.t0 - o.dur) / FADE).toFixed(3);
  st.t = t;
}

function play(st) {
  cancelAnimationFrame(st.raf);
  st.phase = 'playing';
  st.t0 = performance.now();
  const tick = (now) => {
    const t = now - st.t0;
    frame(st, Math.min(t, TOTAL));
    if (t < TOTAL) st.raf = requestAnimationFrame(tick);
    else st.phase = 'done';
  };
  st.raf = requestAnimationFrame(tick);
}

function bindHover(st) {
  const { P, svg, hit, tip } = st;
  const order = ['ours', ...BASE_KEYS.slice().reverse()];
  const show = (i) => {
    st.active = i;
    const xk = st.x(KS[i]);
    st.cross.setAttribute('x1', xk);
    st.cross.setAttribute('x2', xk);
    st.cross.setAttribute('visibility', 'visible');
    const rows = order.map((key) => ({ key, v: key === 'ours' ? P.ours[i] : P.base[key] })).sort((a, b) => b.v - a.v);
    tip.innerHTML = `<div class="clicks-tip-h"><span>${P.metric} ↑</span><span>K = ${KS[i]}</span></div>` +
      rows.map((r) => `<div class="clicks-tip-r${r.key === 'ours' ? ' is-ours' : ''}">${swatch(r.key)}<span>${SERIES[r.key].label}</span><b>${fmt(r.v)}</b></div>`).join('');
    tip.hidden = false;
    const tw = tip.offsetWidth;
    const bw = st.box.clientWidth;
    let left = xk + 14;
    if (left + tw > bw) left = xk - 14 - tw;
    tip.style.left = `${Math.max(0, Math.min(left, bw - tw))}px`;
    tip.style.top = `${st.M.t}px`;
    frame(st, st.phase === 'idle' ? -1 : st.t);
  };
  const hide = () => {
    st.active = -1;
    tip.hidden = true;
    st.cross.setAttribute('visibility', 'hidden');
    frame(st, st.phase === 'idle' ? -1 : st.t);
  };
  const nearest = (evt) => {
    const px = evt.clientX - svg.getBoundingClientRect().left;
    let i = 0;
    KS.forEach((k, j) => { if (Math.abs(st.x(k) - px) < Math.abs(st.x(KS[i]) - px)) i = j; });
    return i;
  };
  hit.addEventListener('pointermove', (e) => show(nearest(e)));
  hit.addEventListener('pointerdown', (e) => show(nearest(e)));
  hit.addEventListener('pointerleave', (e) => { if (e.pointerType !== 'touch') hide(); });
  svg.addEventListener('focus', () => show(st.active < 0 ? KS.length - 1 : st.active));
  svg.addEventListener('blur', hide);
  svg.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      show(Math.min(KS.length - 1, Math.max(0, (st.active < 0 ? KS.length - 1 : st.active) + (e.key === 'ArrowRight' ? 1 : -1))));
    } else if (e.key === 'Escape') hide();
  });
  st.hide = hide;
}

function init(root) {
  const grid = root.querySelector('[data-clicks-grid]');
  const replay = root.querySelector('[data-clicks-replay]');
  if (!grid) return;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const panels = PANELS.map((P, i) => makePanel(P, i, grid));

  const refresh = (st) => {
    if (!build(st)) return;
    if (st.phase === 'playing') { cancelAnimationFrame(st.raf); st.phase = 'done'; }
    if (reduced) st.phase = 'done';
    frame(st, st.phase === 'done' ? TOTAL : -1);
  };
  panels.forEach(refresh);
  const ro = new ResizeObserver((entries) => entries.forEach((en) => {
    const st = panels.find((p) => p.box === en.target);
    if (st) refresh(st);
  }));
  panels.forEach((st) => ro.observe(st.box));

  // Tap outside a chart hides a touch tooltip
  document.addEventListener('pointerdown', (e) => panels.forEach((st) => {
    if (st.active >= 0 && !st.box.contains(e.target)) st.hide();
  }));

  if (reduced) return;
  const io = new IntersectionObserver((entries) => entries.forEach((en) => {
    const st = panels.find((p) => p.fig === en.target);
    if (st && en.intersectionRatio >= 0.55 && st.phase === 'idle') {
      play(st);
      io.unobserve(en.target);
    }
  }), { threshold: [0, 0.55, 1] });
  panels.forEach((st) => io.observe(st.fig));
  if (replay) {
    replay.hidden = false;
    replay.addEventListener('click', () => panels.forEach((st) => { io.unobserve(st.fig); play(st); }));
  }
}

const section = document.getElementById('clicks');
if (section) init(section);
