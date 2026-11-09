// Click3R project page, Explore section (#try). Shows real per-pixel predictions from assets/data/tryit.
// three.js and the data load only when the section comes within one viewport; the render loop runs only on screen.

const PALETTE = ['#ff0000', '#0000ff', '#008000', '#ffa500', '#800080', '#a52a2a', '#ffc0cb', '#808080', '#808000',
  '#00ffff', '#ff00ff', '#000080'];  // the demo's PAIR_COLORS (export_demo_pairs.COLORS): click 1 red, 2 blue, 3 green ...
const DARK_TEXT = new Set(['#ffa500', '#ffc0cb', '#00ffff']);
const MODES = { click3r: 'Click3R', dust3r: 'DUSt3R', pi3: 'Pi3 (224)', gt: 'Ground truth' };
const MORPH_MS = 900;
const ALREADY_RIGHT = 10;  // deg
const FOV = 40;
const PIX_OVERLAP = 1.2;  // point diameter in image pixels (slightly above 1 so surfaces close up)
const SVGNS = 'http://www.w3.org/2000/svg';
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
// Slow connection (Save-Data, 2G / 3G; Chromium reports it) or a low-end device: gallery tiles show posters and play
// only while hovered or focused, instead of up to 16 looping videos downloading and decoding at once.
const conn = navigator.connection;
const lite = Boolean(conn && (conn.saveData || /2g|3g/.test(conn.effectiveType || '')))
  || (navigator.deviceMemory || 8) <= 2 || (navigator.hardwareConcurrency || 8) <= 2;

const section = document.getElementById('try');
if (section && section.querySelector('.tryit-stage')) {
  const io = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    io.disconnect();
    start(section);
  }, { rootMargin: '100% 0px 100% 0px' });
  io.observe(section);
}

// Live demo card: the poster thumbnail opens the screen recording in a modal. The video gets its poster and src on
// the first open (preload none; nothing is fetched before) and pauses on close; Esc (native), the close button and a click outside the video close it.
const modal = section && section.querySelector('.tryit-modal');
if (modal) {
  const video = modal.querySelector('video');
  section.querySelector('.tryit-demo-thumb').addEventListener('click', () => {
    if (!video.getAttribute('src')) { video.poster = video.dataset.poster; video.src = video.dataset.src; }
    modal.showModal();
    video.play().then(null, () => {});  // a refused play leaves the controls
  });
  modal.querySelector('.tryit-modal-close').addEventListener('click', () => modal.close());
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.close(); });
  modal.addEventListener('close', () => video.pause());
}

const clicksLabel = (k) => (k === 0 ? 'no clicks' : k === 1 ? '1 click' : `${k} clicks`);
// half-up on the 2-decimal manifest values (174.85 -> 174.9, as the paper tables and the old page)
const fmtDeg = (v) => (v === null || v === undefined ? '–' : `${(Math.round(v * 10 + 1e-6) / 10).toFixed(1)}°`);
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

// A network or parse error resolves to null, so the callers' "could not be loaded" paths show.
function fetchJSON(url) {
  return fetch(url).then((r) => (r.ok ? r.json() : null)).then((v) => v, () => null);
}

// A gzipped file (.gz), decompressed here: static hosts such as GitHub Pages do not compress binary responses.
function fetchGz(url) {
  return fetch(url).then((r) => (r.ok ? new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer() : null))
    .then((v) => v, () => null);
}

async function start(sec) {
  const $ = (s) => sec.querySelector(s);
  const ui = {
    grid: $('.tryit-grid'), stage: $('.tryit-stage'), photos: $('.tryit-photos'), imgs: [...sec.querySelectorAll('.tryit-photo img')],
    svg: $('.tryit-overlay'), add: $('.tryit-add'), remove: $('.tryit-remove'), count: $('.tryit-count'),
    hint: $('.tryit-hint'), viewer: $('.tryit-viewer'), canvas: $('.tryit-canvas'), bar: $('.tryit-progress i'),
    state: $('.tryit-state'), resets: [...sec.querySelectorAll('.tryit-reset')], errValue: $('.tryit-err-value'), spark: $('.tryit-spark'),
    modes: [...sec.querySelectorAll('.tryit-mode')], cams: [...sec.querySelectorAll('.tryit-cam')], marks: $('.tryit-marks'),
  };
  // Hide clicks: hides the click dots, their lines and the next-click ring on the two photos (kept across pairs;
  // adding a click shows them again)
  const setMarksHidden = (hide) => {
    ui.photos.classList.toggle('hide-marks', hide);
    ui.marks.setAttribute('aria-pressed', String(hide));
    ui.marks.textContent = hide ? 'Show clicks' : 'Hide clicks';
  };
  ui.marks.addEventListener('click', () => setMarksHidden(!ui.photos.classList.contains('hide-marks')));
  const root = new URL(sec.dataset.tryitRoot || 'assets/data/tryit/', document.baseURI);
  // one WebGL2 context, on the viewer's own canvas (three.js reuses it)
  ui.gl = ui.canvas.getContext('webgl2', { antialias: true, alpha: true, powerPreference: 'high-performance' });
  const hasGL = ui.gl !== null;
  if (!hasGL) ui.viewer.classList.add('no-webgl');

  // ---------- loading progress (thin line on top of the viewer) ----------
  let progTotal = 0;
  let progDone = 0;
  const progress = (done, total) => {
    progDone = done;
    progTotal = total;
    ui.bar.style.transform = `scaleX(${total ? Math.max(0.04, done / total) : 0})`;
    ui.viewer.classList.toggle('is-loading', total > 0 && done < total);
  };
  const tracked = (p) => {
    progress(progDone, progTotal + 1);
    return p.then((v) => { progress(progDone + 1, progTotal); return v; });
  };
  const settle = () => { if (progDone >= progTotal) progress(0, 0); };

  // manifest order = page order (neighbouring tiles come from different datasets)
  const galURL = new URL($('.tryit-gallery').dataset.gallery, document.baseURI);
  const [THREE, manifest, gallery] = await Promise.all([
    hasGL ? tracked(import('../../lib/three.module.min.js')) : null,
    tracked(fetchJSON(new URL('manifest.json', root))),
    fetchJSON(galURL),
  ]);
  if (!manifest || !manifest.length) {
    progress(0, 0);
    ui.hint.textContent = 'The example data could not be loaded.';
    return;
  }

  const S = { idx: -1, entry: null, pd: null, k: 0, mode: 'click3r', hot: -1, sel: 0, req: 0 };

  // ---------- gallery: one animated tile per pair (Click3R without clicks -> with them), dataset filter ----------
  // Tile videos (assets/media/gallery, 320 px loops): src attached within 400 px of the viewport, play at >= 30 %
  // visible, pause otherwise; reduced motion: posters only; lite (slow connection / low-end device): posters, a tile
  // plays while hovered or focused.
  const media = new Map((gallery || []).map((g) => [g.id, g]));
  const asset = (p) => { const u = new URL(p, document.baseURI); u.search = galURL.search; return u.href; };
  // No tile video is fetched before the first pair's points are on screen (or 8 s have passed): on a slow connection
  // the 16 loops (~3 MB) would otherwise compete with the viewer's first state.
  let galleryGo = null;
  const galleryReady = new Promise((r) => { galleryGo = r; });
  const attach = (v) => { if (!v.getAttribute('src')) v.src = v.dataset.src; };
  const playTile = (v) => galleryReady.then(() => {
    if (v.dataset.want !== '1') return;
    attach(v);
    if (v.paused) v.play().then(null, () => {});  // a refused play leaves the poster
  });
  const nearIO = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (en.isIntersecting) { const v = en.target; galleryReady.then(() => attach(v)); nearIO.unobserve(v); }
    }
  }, { rootMargin: '400px 0px 400px 0px' });
  const playIO = new IntersectionObserver((entries) => {
    for (const en of entries) {
      const v = en.target;
      v.dataset.want = en.isIntersecting && en.intersectionRatio >= 0.3 ? '1' : '0';
      if (v.dataset.want === '1') playTile(v);
      else if (!v.paused) v.pause();
    }
  }, { threshold: [0, 0.3, 0.6, 1] });
  const tiles = manifest.map((e, i) => {
    const g = media.get(e.id);
    const clicks = g ? g.clicks : e.default_k;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tryit-tile';
    b.dataset.id = e.id;
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', `Pair ${i + 1}, ${clicksLabel(clicks)}`);
    const box = document.createElement('span');
    box.className = 'tryit-tile-media';
    const v = document.createElement('video');
    v.muted = true;
    v.defaultMuted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = 'none';
    v.setAttribute('aria-hidden', 'true');
    v.poster = g ? asset(g.poster) : new URL(`${e.id}/view1.jpg`, root).href;
    if (g && !reduced) {
      v.dataset.src = asset(g.file);
      if (lite) {
        const on = () => { v.dataset.want = '1'; playTile(v); };
        const off = () => { v.dataset.want = '0'; if (!v.paused) v.pause(); };
        b.addEventListener('pointerenter', on);
        b.addEventListener('pointerleave', off);
        b.addEventListener('focus', on);
        b.addEventListener('blur', off);
      } else {
        nearIO.observe(v);
        playIO.observe(v);
      }
    }
    box.appendChild(v);
    const cap = document.createElement('span');
    cap.className = 'tryit-tile-cap';
    const ck = document.createElement('span');
    ck.textContent = clicksLabel(clicks);
    cap.append(ck);
    b.append(box, cap);
    b.addEventListener('click', () => { selectPair(i); bringViewer(); });
    ui.grid.appendChild(b);
    return b;
  });

  // after a tile click: bring the stage (photos + 3D view) into view unless the viewer is already fully visible
  function bringViewer() {
    const r = ui.viewer.getBoundingClientRect();
    const top = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
    if (r.top >= top && r.bottom <= window.innerHeight) return;
    ui.stage.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
  }

  // ---------- 3D viewer ----------
  const V = hasGL ? makeViewer(THREE, ui, () => S, (i) => setHot(i)) : null;

  // ---------- data ----------
  // Point colours = the two input photos (view<v>.jpg, 2x upscales of the model crops, already loaded for the photo
  // column) drawn at the model resolution: within 1-4 of 255 of colors.bin, which the page therefore no longer fetches.
  function viewColors(base, meta) {
    if (!meta) return Promise.resolve(null);
    const { H, W } = meta;
    const cv = document.createElement('canvas');
    cv.width = W;
    cv.height = H;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    const out = new Uint8Array(2 * H * W * 3);
    const one = (v) => {
      const img = new Image();
      img.src = new URL(`view${v + 1}.jpg`, base).href;  // same URL as the photo column, so the browser reuses it
      return img.decode().then(() => {
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, W, H);
        const d = ctx.getImageData(0, 0, W, H).data;
        for (let i = 0, o = v * H * W * 3; i < H * W; i++, o += 3) {
          out[o] = d[4 * i];
          out[o + 1] = d[4 * i + 1];
          out[o + 2] = d[4 * i + 2];
        }
        return true;
      }, () => false);
    };
    return one(0).then((a) => one(1).then((b) => (a && b ? out : null)));
  }

  function pairData(entry) {
    const base = new URL(`${entry.id}/`, root);
    const pd = { entry, base, meta: null, colors: null, states: new Map(), done: new Set() };
    // All predicted states of a pair share one mask, so k0_mask serves k*, dust3r and pi3.
    const meta = fetchJSON(new URL('meta.json', base));
    pd.ready = Promise.all([meta, meta.then((m) => viewColors(base, m)),
      fetchGz(new URL('gt_mask.bin.gz', base)), fetchGz(new URL('k0_mask.bin.gz', base))]).then(([meta, colors, gtMask, predMask]) => {
      pd.meta = meta;
      pd.colors = colors;
      pd.gtMask = gtMask && new Uint8Array(gtMask);
      pd.predMask = predMask && new Uint8Array(predMask);
      return meta !== null && colors !== null && pd.gtMask !== null && pd.predMask !== null;
    });
    return pd;
  }

  function loadState(pd, name) {
    if (!pd.states.has(name)) {
      const p = Promise.all([fetchGz(new URL(`${name}.p16.gz`, pd.base)), pd.ready])
        .then(([b, ok]) => { pd.done.add(name); return ok && b && V ? V.decodeState(pd, name, b, name === 'gt' ? pd.gtMask : pd.predMask) : null; });
      pd.states.set(name, p);
    }
    return pd.states.get(name);
  }

  // Fetch states before they are needed: the next click on every change, a comparison when its pill is hovered.
  // (A state is ~0.2 MB as .p16.gz, so nothing else is fetched up front.)
  function warm(pd, names) {
    for (const n of names) {
      const k = /^k(\d+)$/.exec(n);
      if (!k || Number(k[1]) <= pd.entry.k_max) loadState(pd, n);
    }
  }

  // ---------- pair selection ----------
  async function selectPair(i) {
    if (i === S.idx) return;
    const sel = ++S.sel;
    const switching = S.idx !== -1;
    S.idx = i;
    S.entry = manifest[i];
    S.k = 0;
    S.mode = 'click3r';
    S.hot = -1;
    tiles.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
    ui.imgs.forEach((img, v) => { img.src = new URL(`${S.entry.id}/view${v + 1}.jpg`, root).href; });
    ui.svg.replaceChildren();
    groups = [];
    next = null;
    if (switching) {  // the previous pair's cloud fades while the new one loads; the readout waits for it
      ui.viewer.classList.add('is-switching');
      ui.errValue.textContent = '–';
      ui.state.textContent = 'Loading';
    }
    const pd = pairData(S.entry);
    S.pd = pd;
    updateControls();
    const ok = await tracked(pd.ready);
    if (sel !== S.sel) return;
    if (!ok) {
      settle();
      ui.hint.textContent = 'This pair could not be loaded.';
      return;
    }
    buildOverlay();
    updateControls();
    drawSpark();
    if (!V) { settle(); return; }
    const st0 = await tracked(loadState(pd, 'k0'));
    if (sel !== S.sel) return;
    settle();
    if (!st0) return;
    V.showPair(pd, st0);
    ui.viewer.classList.remove('is-switching');
    setReadout();
    warm(pd, ['k1']);
  }

  // ---------- state changes ----------
  async function show(name, delay) {
    if (!V || !S.pd) { setReadout(); return; }
    const req = ++S.req;
    const pd = S.pd;
    const p = loadState(pd, name);
    const st = await (pd.done.has(name) ? p : tracked(p));
    settle();
    if (req !== S.req || pd !== S.pd || !st) return;
    V.goTo(st, delay, setReadout);
  }

  function setK(k) {
    if (!S.pd || !S.pd.meta || k < 0 || k > S.entry.k_max || k === S.k) return;
    const add = k > S.k;
    S.k = k;
    S.mode = 'click3r';
    updateControls();
    updateOverlay(add ? S.k - 1 : -1);
    if (V) V.setActive(S.k, S.k);
    show(`k${S.k}`, add && !reduced ? 320 : 0);
    if (S.pd) warm(S.pd, [`k${S.k + 1}`]);
  }

  function setMode(m) {
    if (!S.pd || !S.pd.meta || m === S.mode) return;
    S.mode = m;
    updateControls();
    if (V) V.setActive(S.k, m === 'click3r' ? S.k : -1);
    show(m === 'click3r' ? `k${S.k}` : m, 0);
  }

  ui.add.addEventListener('click', () => { setMarksHidden(false); setK(S.k + 1); });
  ui.remove.addEventListener('click', () => setK(S.k - 1));
  ui.modes.forEach((b) => {
    b.addEventListener('click', () => setMode(b.dataset.mode));
    const pre = () => { if (S.pd && S.pd.meta && b.dataset.mode !== 'click3r') warm(S.pd, [b.dataset.mode]); };
    b.addEventListener('pointerenter', pre);
    b.addEventListener('focus', pre);
  });
  ui.resets.forEach((b) => b.addEventListener('click', () => V && V.resetView(b.dataset.view)));

  function updateControls() {
    const K = S.entry ? S.entry.k_max : 0;
    ui.count.innerHTML = `<b>${S.k}</b> of <b>${K}</b>`;
    ui.count.setAttribute('aria-label', `${S.k} of ${K} clicks`);
    const ready = !!(S.pd && S.pd.meta);
    ui.add.disabled = !ready || S.k >= K;
    ui.remove.disabled = !ready || S.k <= 0;
    ui.modes.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === S.mode)));
    // pairs where Click3R is already right without clicks (k0 < ALREADY_RIGHT deg) tell the comparison story instead
    const right = !!S.entry && S.entry.rre.click3r[0] < ALREADY_RIGHT;
    ui.hint.innerHTML = S.k >= K ? 'All clicks added; compare with DUSt3R, Pi3 and the ground truth.'
      : right && S.k === 0 ? 'Click3R is already right here without clicks; DUSt3R and Pi3 are not. Compare them under the 3D view.'
        : 'Add a click with the button or the pulsing ring.';
    sec.classList.toggle('tryit-is-compare', S.mode !== 'click3r');
  }

  // Readout follows the state on screen: called when a morph ends.
  function setReadout() {
    const e = S.entry;
    if (!e) return;
    const v = S.mode === 'click3r' ? e.rre.click3r[S.k] : S.mode === 'gt' ? null : e.rre[S.mode];
    ui.state.textContent = S.mode === 'click3r' ? `Click3R · ${clicksLabel(S.k)}`
      : S.mode === 'gt' ? 'Ground truth' : `${MODES[S.mode]} · no clicks`;
    ui.errValue.textContent = S.mode === 'gt' ? 'reference' : fmtDeg(v);
    ui.errValue.classList.toggle('is-small', S.mode === 'gt');
    drawSpark();
  }

  // ---------- sparkline: Click3R rotation error vs number of clicks ----------
  function drawSpark() {
    const e = S.entry;
    const svg = ui.spark;
    svg.replaceChildren();
    if (!e) return;
    const ys = e.rre.click3r;
    const W = 132;
    const top = 4;
    const bot = 32;
    const all = ys.concat([e.rre.dust3r, e.rre.pi3]).filter((v) => v !== null);
    const ymax = Math.max(10, ...all);
    const X = (k) => (ys.length === 1 ? W / 2 : 4 + (k * (W - 8)) / (ys.length - 1));
    const Y = (v) => bot - ((bot - top) * v) / ymax;
    svgEl('line', { x1: 0, x2: W, y1: bot, y2: bot, class: 'tryit-spark-axis' }, svg);
    if (S.mode === 'dust3r' || S.mode === 'pi3') {
      const v = e.rre[S.mode];
      if (v !== null) svgEl('line', { x1: 0, x2: W, y1: Y(v), y2: Y(v), class: 'tryit-spark-ref' }, svg);
    }
    svgEl('polyline', { points: ys.map((v, k) => `${X(k)},${Y(v)}`).join(' '), class: 'tryit-spark-line' }, svg);
    ys.forEach((v, k) => {
      const cur = S.mode === 'click3r' && k === S.k;
      svgEl('circle', { cx: X(k), cy: Y(v), r: cur ? 3.6 : 2, class: cur ? 'tryit-spark-cur' : 'tryit-spark-dot' }, svg);
    });
    svgEl('text', { x: X(0), y: 43, class: 'tryit-spark-tick', 'text-anchor': 'middle' }, svg).textContent = '0';
    svgEl('text', { x: X(ys.length - 1), y: 43, class: 'tryit-spark-tick', 'text-anchor': 'middle' }, svg).textContent = String(ys.length - 1);
    svgEl('text', { x: W / 2, y: 43, class: 'tryit-spark-tick', 'text-anchor': 'middle' }, svg).textContent = 'clicks';
  }

  // ---------- 2D clicks on the photos ----------
  let groups = [];
  let next = null;
  function buildOverlay() {
    const svg = ui.svg;
    svg.replaceChildren();
    groups = [];
    const clicks = S.pd.meta.clicks;
    const links = svgEl('g', { class: 'tryit-links' }, svg);
    const dots = svgEl('g', {}, svg);
    clicks.forEach((c, i) => {
      const col = PALETTE[i % PALETTE.length];
      const under = svgEl('line', { class: 'tryit-link-under' }, links);
      const line = svgEl('line', { class: 'tryit-link', stroke: col }, links);
      const g = svgEl('g', { class: 'tryit-click', 'data-i': i }, dots);
      const ends = [0, 1].map(() => {
        const d = svgEl('g', { class: 'tryit-dot' }, g);
        svgEl('circle', { r: 9, fill: col }, d);
        svgEl('text', { 'text-anchor': 'middle', dy: '0.35em', fill: DARK_TEXT.has(col) ? '#15171a' : '#fff' }, d).textContent = String(i + 1);
        d.addEventListener('pointerenter', () => setHot(i));
        d.addEventListener('pointerleave', () => setHot(-1));
        return d;
      });
      groups.push({ c, under, line, g, ends });
    });
    next = svgEl('g', { class: 'tryit-next' }, svg);
    next.ghost = svgEl('line', { class: 'tryit-ghost' }, next);
    next.rings = [0, 1].map(() => {
      const r = svgEl('g', { class: 'tryit-ring' }, next);
      svgEl('circle', { r: 16, class: 'tryit-ring-hit' }, r);
      svgEl('circle', { r: 9, class: 'tryit-ring-core' }, r);
      svgEl('circle', { r: 9, class: 'tryit-ring-pulse' }, r);
      r.addEventListener('click', () => setK(S.k + 1));
      r.addEventListener('pointerenter', () => setHot(S.k));
      r.addEventListener('pointerleave', () => setHot(-1));
      return r;
    });
    layoutOverlay();
    updateOverlay(-1);
  }

  function layoutOverlay() {
    if (!groups.length && !next) return;
    const box = ui.photos.getBoundingClientRect();
    const rects = ui.imgs.map((im) => im.getBoundingClientRect());
    const at = (v, x, y) => [rects[v].left - box.left + x * rects[v].width, rects[v].top - box.top + y * rects[v].height];
    const small = box.width < 420;
    for (const gr of groups) {
      const a = at(0, gr.c[0], gr.c[1]);
      const b = at(1, gr.c[2], gr.c[3]);
      for (const ln of [gr.under, gr.line]) {
        ln.setAttribute('x1', a[0]); ln.setAttribute('y1', a[1]); ln.setAttribute('x2', b[0]); ln.setAttribute('y2', b[1]);
      }
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      gr.line.style.strokeDasharray = `${len} ${len}`;
      gr.line.dataset.len = len;
      gr.ends[0].setAttribute('transform', `translate(${a[0]} ${a[1]})${small ? ' scale(0.85)' : ''}`);
      gr.ends[1].setAttribute('transform', `translate(${b[0]} ${b[1]})${small ? ' scale(0.85)' : ''}`);
      gr.ends.forEach((d) => d.classList.toggle('is-small', small));
    }
    if (next) next.small = small;
    placeNext();
  }

  function placeNext() {
    if (!next || !S.pd || !S.pd.meta) return;
    const clicks = S.pd.meta.clicks;
    const on = S.k < clicks.length;
    next.classList.toggle('is-on', on);
    if (!on) return;
    const c = clicks[S.k];
    const box = ui.photos.getBoundingClientRect();
    const rects = ui.imgs.map((im) => im.getBoundingClientRect());
    const pts = [[0, c[0], c[1]], [1, c[2], c[3]]].map(([v, x, y]) => [rects[v].left - box.left + x * rects[v].width, rects[v].top - box.top + y * rects[v].height]);
    next.rings.forEach((r, v) => r.setAttribute('transform', `translate(${pts[v][0]} ${pts[v][1]})`));
    next.ghost.setAttribute('x1', pts[0][0]); next.ghost.setAttribute('y1', pts[0][1]);
    next.ghost.setAttribute('x2', pts[1][0]); next.ghost.setAttribute('y2', pts[1][1]);
    next.style.setProperty('--tryit-next', PALETTE[S.k % PALETTE.length]);
  }

  function updateOverlay(added) {
    groups.forEach((gr, i) => {
      const on = i < S.k;
      gr.g.classList.toggle('is-on', on);
      gr.under.classList.toggle('is-on', on);
      gr.line.classList.toggle('is-on', on);
      const len = Number(gr.line.dataset.len || 0);
      if (i === added && !reduced) {  // draw the new link from photo 1 to photo 2
        gr.line.style.transition = 'none';
        gr.line.style.strokeDashoffset = String(len);
        gr.line.getBoundingClientRect();
        gr.line.style.transition = '';
      }
      gr.line.style.strokeDashoffset = on ? '0' : String(len);
    });
    placeNext();
    paintHot();
  }

  function setHot(i) {
    if (i === S.hot) return;
    S.hot = i;
    paintHot();
    if (V) V.setHot(i);
  }

  function paintHot() {
    groups.forEach((gr, i) => {
      gr.g.classList.toggle('is-hot', i === S.hot);
      gr.line.classList.toggle('is-hot', i === S.hot);
      gr.g.classList.toggle('is-dim', S.hot >= 0 && i !== S.hot);
      gr.line.classList.toggle('is-dim', S.hot >= 0 && i !== S.hot);
    });
  }

  new ResizeObserver(() => layoutOverlay()).observe(ui.photos);
  ui.imgs.forEach((im) => im.addEventListener('load', () => layoutOverlay()));

  selectPair(0).then(galleryGo, galleryGo);
  setTimeout(galleryGo, 8000);
}

// =====================================================================================================
// three.js viewer: one Points object whose vertex shader mixes two per-pixel position buffers (A -> B).
// =====================================================================================================
function makeViewer(THREE, ui, getS, onHover) {
  const LUT = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) LUT[i] = THREE.DataUtils.fromHalfFloat(i);

  const renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, context: ui.gl, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);
  camera.up.set(0, -1, 0);  // data are in camera-1 OpenCV axes: x right, y down, z forward

  const material = new THREE.ShaderMaterial({
    uniforms: { uT: { value: 1 }, uAppear: { value: 1 }, uProj: { value: 500 }, uPixA: { value: 0.005 },
      uPixB: { value: 0.005 }, uC2: { value: new THREE.Vector3() }, uMin: { value: 1 }, uMax: { value: 12 } },
    vertexShader: `
      attribute vec3 posB;
      attribute float visA;
      attribute float visB;
      attribute float side;
      attribute vec3 rgb;
      uniform float uT, uAppear, uProj, uPixA, uPixB, uMin, uMax;
      uniform vec3 uC2;
      varying vec3 vRgb;
      varying float vSize;
      void main() {
        vec3 p = mix(position, posB, uT);
        float vis = mix(visA, visB, uT) * uAppear;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        // one image pixel covers (distance to its own camera) / focal in the world; project that footprint
        float own = side < 0.5 ? length(p) : length(p - uC2);
        float s = clamp(own * mix(uPixA, uPixB, uT) * uProj / max(-mv.z, 1e-3), uMin, uMax) * vis;
        if (vis < 0.03) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = s;
        vSize = s;
        vRgb = rgb;
      }`,
    fragmentShader: `
      varying vec3 vRgb;
      varying float vSize;
      void main() {
        if (vSize > 2.5) { vec2 c = gl_PointCoord - 0.5; if (dot(c, c) > 0.25) discard; }
        gl_FragColor = vec4(vRgb, 1.0);
      }`,
  });

  let points = null;
  let attrs = null;
  let pd = null;
  let cur = null;  // state the morph is heading to
  const morph = { t0: 0, delay: 0, dur: 0, done: null };
  let appear = { t0: -1e9, dur: 700 };
  let fromMarks = [];
  let toMarks = [];
  let fromCam = null;
  let toCam = null;

  // ---------- click markers: one sphere per view, joined by a bar, in the click colour ----------
  const markerRoot = new THREE.Group();
  scene.add(markerRoot);
  const sphereGeo = new THREE.SphereGeometry(1, 20, 14);
  const barGeo = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
  const outlineMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.BackSide, depthTest: false, transparent: true });
  let markers = [];
  const active = { k: 0, ghost: -1 };  // ghost = the next click, drawn faint so the visitor sees what it will join

  function buildMarkers(n) {
    for (const m of markers) markerRoot.remove(m.group);
    markers = [];
    for (let i = 0; i < n; i++) {
      const col = new THREE.Color(PALETTE[i % PALETTE.length]);
      const fill = new THREE.MeshBasicMaterial({ color: col, depthTest: false, transparent: true });
      const barMat = new THREE.MeshBasicMaterial({ color: col, depthTest: false, transparent: true, opacity: 0.9 });
      const group = new THREE.Group();
      const ends = [0, 1].map(() => {
        const o = new THREE.Mesh(sphereGeo, outlineMat);
        o.renderOrder = 2;
        const s = new THREE.Mesh(sphereGeo, fill);
        s.renderOrder = 3;
        group.add(o, s);
        return { o, s };
      });
      const bar = new THREE.Mesh(barGeo, barMat);
      bar.renderOrder = 1;
      group.add(bar);
      markerRoot.add(group);
      markers.push({ group, ends, bar, fill, barMat, vis: 0, hot: 0, g: 0, a: new THREE.Vector3(), b: new THREE.Vector3(), okA: false, okB: false });
    }
  }

  // ---------- cameras: frustum of camera 1 (fixed) and camera 2 (moves with the prediction) ----------
  const frustumGeo = new THREE.BufferGeometry();
  {
    const a = 0.42;
    const c = [[-a, -a, 1], [a, -a, 1], [a, a, 1], [-a, a, 1]];
    const v = [];
    for (const p of c) v.push(0, 0, 0, ...p);
    for (let i = 0; i < 4; i++) v.push(...c[i], ...c[(i + 1) % 4]);
    v.push(-a * 0.35, -a, 1, 0, -a * 1.45, 1, 0, -a * 1.45, 1, a * 0.35, -a, 1);  // small "up" tick
    frustumGeo.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  }
  const cam1 = new THREE.LineSegments(frustumGeo, new THREE.LineBasicMaterial({ color: 0x9aa1ab, transparent: true, opacity: 0.55 }));
  const cam2 = new THREE.LineSegments(frustumGeo, new THREE.LineBasicMaterial({ color: 0xf2f3f5, transparent: true, opacity: 0.8 }));
  cam1.matrixAutoUpdate = false;
  cam2.matrixAutoUpdate = false;
  scene.add(cam1, cam2);

  function camPose(m) {
    if (!m) return null;
    const M = new THREE.Matrix4().set(...m.flat());
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    M.decompose(p, q, s);
    return { p, q };
  }

  // ---------- data decoding ----------
  // buf: a .p16 state, after gunzip: float16 bit patterns stored planar (x of every pixel, then y, then z),
  // delta-coded along each plane (mod 2^16) and byte-split (all high bytes, then all low bytes), for gzip.
  function decodeState(pdata, name, buf, mask) {
    const { H, W } = pdata.meta;
    const n = 2 * H * W;
    const u8 = new Uint8Array(buf);
    if (u8.length !== n * 6 || mask.length !== n) return null;
    const pos = new Float32Array(n * 3);
    for (let c = 0; c < 3; c++) {
      let v = 0;
      for (let i = 0, k = c * n; i < n; i++, k++) {
        v = (v + ((u8[k] << 8) | u8[3 * n + k])) & 0xffff;
        pos[3 * i + c] = LUT[v];
      }
    }
    const pix = PIX_OVERLAP / focal(pos, mask, H, W);
    const marks = pdata.meta.clicks.map((c) => ({ a: pick(pos, mask, 0, c[0], c[1], H, W), b: pick(pos, mask, 1, c[2], c[3], H, W) }));
    return { name, pos, mask, marks, pix, cam: camPose(pdata.meta.cam2[name]) };
  }

  // focal length (crop pixels) implied by this state's view-1 points: u = f * x / z, v = f * y / z (centred principal
  // point); sets the point size so one point covers one image pixel
  function focal(pos, mask, H, W) {
    const fs = [];
    for (let y = 1; y < H; y += 3) {
      for (let x = 1; x < W; x += 3) {
        const i = y * W + x;
        const z = pos[3 * i + 2];
        if (!mask[i] || !(z > 1e-3)) continue;
        const u = x + 0.5 - W / 2;
        const v = y + 0.5 - H / 2;
        if (Math.abs(u) > W / 6 && Math.abs(pos[3 * i]) > 1e-6) fs.push((u * z) / pos[3 * i]);
        if (Math.abs(v) > H / 6 && Math.abs(pos[3 * i + 1]) > 1e-6) fs.push((v * z) / pos[3 * i + 1]);
      }
    }
    const pf = fs.filter((f) => f > 0).sort((a, b) => a - b);
    return pf.length > 50 ? clamp(pf[pf.length >> 1], 80, 2000) : 250;
  }

  // 3D point of a clicked pixel; nearest kept pixel within 4 px if the pixel itself is masked
  function pick(pos, mask, v, xn, yn, H, W) {
    const cx = clamp(Math.floor(xn * W), 0, W - 1);
    const cy = clamp(Math.floor(yn * H), 0, H - 1);
    for (let r = 0; r <= 4; r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          const x = cx + dx;
          const y = cy + dy;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          const i = v * H * W + y * W + x;
          if (mask[i]) return [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
        }
      }
    }
    return null;
  }

  // ---------- orbit camera (own controls: drag rotates; wheel, two-finger scroll and pinch zoom) ----------
  const orbit = { target: new THREE.Vector3(0, 0, 1), az: 0, el: 0, dist: 2, vAz: 0, vEl: 0, user: false, sway: 0,
    home: null, back: null, minD: 0.3, maxD: 8, up: new THREE.Vector3(0, -1, 0) };
  const UP1 = new THREE.Vector3(0, -1, 0);  // camera 1's image up (OpenCV y down)

  // Home view per pair: look from behind camera 1, turned toward camera 2's side and slightly from above. The distance
  // is the smallest at which the reference points (CO3D 97th, other datasets 80th percentile), the opening click pair
  // and (within a limit) both cameras fit inside the canvas, leaving room for the HUD at top and bottom.
  function fitView(stA) {
    const { H, W } = pd.meta;
    const n = H * W;
    const gt = pd.gtMask;
    let useGt = 0;
    for (let i = 0; i < n; i++) useGt += gt[i] && stA.mask[i] ? 1 : 0;
    const pts = [];
    for (let i = 0; i < n; i += 3) {
      if (!stA.mask[i] || (useGt >= 1000 && !gt[i])) continue;
      pts.push(new THREE.Vector3(stA.pos[3 * i], stA.pos[3 * i + 1], stA.pos[3 * i + 2]));
    }
    const med = (a) => { const q = Float32Array.from(a).sort(); return q.length ? q[q.length >> 1] : 0; };
    const t = new THREE.Vector3(med(pts.map((p) => p.x)), med(pts.map((p) => p.y)), med(pts.map((p) => p.z)) || 1);
    // the first click (shown as the faint pair at the start) pulls the target 30% toward itself
    const m0 = stA.marks[0];
    const ends = m0 ? [m0.a, m0.b].filter(Boolean).map((p) => new THREE.Vector3(...p)) : [];
    if (ends.length) t.lerp(ends.reduce((acc, p) => acc.add(p), new THREE.Vector3()).multiplyScalar(1 / ends.length), 0.3);
    const c2 = camPose(pd.meta.cam2.gt) || camPose(pd.meta.cam2[stA.name]);
    const side = c2 && c2.p.x > 0 ? 1 : -1;
    const az = 0.42 * side;
    const el = 0.24;
    const dir = new THREE.Vector3(Math.sin(az) * Math.cos(el), -Math.sin(el), -Math.cos(az) * Math.cos(el));
    const probe = new THREE.PerspectiveCamera();
    probe.up.set(0, -1, 0);
    probe.position.copy(t).add(dir);
    probe.lookAt(t);
    probe.updateMatrixWorld();
    const right = new THREE.Vector3().setFromMatrixColumn(probe.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(probe.matrixWorld, 1);
    const aspect = (cv.clientWidth || 4) / (cv.clientHeight || 3);
    const tanV = Math.tan(((FOV / 2) * Math.PI) / 180) * 0.74;  // HUD bands at top and bottom
    const tanH = Math.tan(((FOV / 2) * Math.PI) / 180) * aspect * 0.86;
    const q = new THREE.Vector3();
    const need = (p) => {
      q.subVectors(p, t);
      return q.dot(dir) + Math.max(Math.abs(q.dot(right)) / tanH, Math.abs(q.dot(up)) / tanV);
    };
    const nd = pts.map(need).sort((a, b) => a - b);
    // CO3D GT covers only the object: show all of it; scenes have long tails (far rooms), so fit 80% of them
    const pct = pd.entry.dataset === 'co3d' ? 0.97 : 0.8;
    const dObj = Math.max(0.35, nd.length ? nd[Math.floor(nd.length * pct)] : 2);
    const dMark = ends.reduce((d, p) => Math.max(d, need(p) * 1.08), 0);
    const dCam = Math.max(need(new THREE.Vector3()), c2 ? need(c2.p) : 0);
    const dist = clamp(Math.max(dObj, dMark, Math.min(dCam, 1.3 * dObj)) * 1.06, 0.35, 8);
    orbit.home = { target: t, az, el, dist };
    orbit.minD = dist * 0.25;
    orbit.maxD = dist * 3;
  }

  function placeCamera() {
    const ce = Math.cos(orbit.el);
    const dir = new THREE.Vector3(Math.sin(orbit.az) * ce, -Math.sin(orbit.el), -Math.cos(orbit.az) * ce);
    camera.position.copy(orbit.target).addScaledVector(dir, orbit.dist);
    camera.up.copy(orbit.up);
    camera.lookAt(orbit.target);
  }

  // Target pose of a reset: the home view, or the view from camera 1 / camera 2 of the shown state: looking along
  // the camera's optical axis at the home target's depth d, from slightly behind the camera (1.3 d) for a wider view.
  const BEHIND = 1.3;
  function viewPose(which) {
    const h = orbit.home;
    if (which !== 'cam1' && which !== 'cam2') return h;
    const d = h.target.length() || 1;
    if (which === 'cam1') return { target: new THREE.Vector3(0, 0, d), az: 0, el: 0, dist: BEHIND * d };
    const c = camAt(eNow(performance.now()));
    if (!c) return h;
    const axis = new THREE.Vector3(0, 0, 1).applyQuaternion(c.q);
    const back = axis.clone().negate();  // from the target towards the camera
    const up = new THREE.Vector3(0, -1, 0).applyQuaternion(c.q);  // camera 2's image up, so its roll matches image 2
    return { target: c.p.clone().addScaledVector(axis, d), az: Math.atan2(back.x, -back.z), el: Math.asin(clamp(-back.y, -1, 1)), dist: BEHIND * d, up };
  }

  function resetView(instant, which) {
    if (!orbit.home) return;
    const h = viewPose(which);
    orbit.user = false;
    orbit.base = h;  // the slow sway continues around the chosen view
    orbit.sway = 0;
    orbit.vAz = 0;
    orbit.vEl = 0;
    if (instant || reduced) {
      orbit.target.copy(h.target); orbit.az = h.az; orbit.el = h.el; orbit.dist = h.dist;
      orbit.up.copy(h.up || UP1);
      orbit.back = null;
      return;
    }
    orbit.back = { t0: performance.now(), to: h, from: { target: orbit.target.clone(), az: orbit.az, el: orbit.el, dist: orbit.dist, up: orbit.up.clone() } };
  }

  // ---------- pointer input ----------
  const ptrs = new Map();
  let drag = 'rotate';
  let pinch = 0;
  const cv = ui.canvas;
  const interact = () => { orbit.user = true; orbit.back = null; };
  cv.addEventListener('contextmenu', (e) => e.preventDefault());
  cv.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0 && e.button !== 2) return;
    cv.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
    drag = e.button === 2 || e.shiftKey ? 'pan' : 'rotate';
    if (ptrs.size === 2) {
      const [p, q] = [...ptrs.values()];
      pinch = Math.hypot(p.x - q.x, p.y - q.y);
    }
    interact();
    cv.classList.add('is-dragging');
    kick();
  });
  cv.addEventListener('pointermove', (e) => {
    if (!ptrs.has(e.pointerId)) {
      if (e.pointerType === 'mouse') hoverAt(e);
      return;
    }
    kick();
    const p = ptrs.get(e.pointerId);
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch > 0 && d > 0) orbit.dist = clamp(orbit.dist * (pinch / d), orbit.minD, orbit.maxD);
      pinch = d;
      return;
    }
    if (drag === 'pan') {
      const h = cv.clientHeight || 1;
      const s = (2 * orbit.dist * Math.tan(((FOV / 2) * Math.PI) / 180)) / h;
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
      orbit.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
      return;
    }
    orbit.vAz = -dx * 0.0065;
    orbit.vEl = dy * 0.0065;
    orbit.az += orbit.vAz;
    orbit.el = clamp(orbit.el + orbit.vEl, -1.3, 1.3);
  });
  const up = (e) => {
    ptrs.delete(e.pointerId);
    pinch = 0;
    if (!ptrs.size) cv.classList.remove('is-dragging');
  };
  cv.addEventListener('pointerup', up);
  cv.addEventListener('pointercancel', up);
  cv.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse' && !ptrs.size) onHover(-1); });
  // Vertical scroll (mouse wheel, two-finger trackpad scroll) and pinch (a wheel event with ctrlKey) zoom. A page scroll
  // that is still going on (its momentum included) when the viewer slides under the cursor keeps scrolling the page;
  // sideways swipes are left to the browser (back / forward on a trackpad).
  let lastPageScroll = -1e9;
  window.addEventListener('scroll', () => { lastPageScroll = performance.now(); }, { passive: true });
  cv.addEventListener('wheel', (e) => {
    const pinch = e.ctrlKey || e.metaKey;
    if (!pinch && (performance.now() - lastPageScroll < 250 || Math.abs(e.deltaX) > Math.abs(e.deltaY))) return;
    e.preventDefault();
    interact();
    const dy = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 600 : 1);  // lines / pages -> pixels
    orbit.dist = clamp(orbit.dist * Math.exp(dy * 0.0015), orbit.minD, orbit.maxD);
    kick();
  }, { passive: false });

  // hover a 3D marker -> highlight its click pair (2D and 3D)
  const tmp = new THREE.Vector3();
  function hoverAt(e) {
    const r = cv.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    let best = -1;
    let bd = 14;
    markers.forEach((m, i) => {
      if (i >= active.k || m.vis < 0.5) return;
      for (const [ok, v] of [[m.okA, m.a], [m.okB, m.b]]) {
        if (!ok) continue;
        tmp.copy(v).project(camera);
        const d = Math.hypot((tmp.x * 0.5 + 0.5) * r.width - mx, (0.5 - tmp.y * 0.5) * r.height - my);
        if (d < bd) { bd = d; best = i; }
      }
    });
    onHover(best);
  }

  // ---------- pair / state transitions ----------
  function showPair(pdata, st) {
    pd = pdata;
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
    }
    const n = 2 * pd.meta.H * pd.meta.W;
    const geo = new THREE.BufferGeometry();
    attrs = {
      a: new THREE.BufferAttribute(new Float32Array(st.pos), 3),
      b: new THREE.BufferAttribute(new Float32Array(st.pos), 3),
      va: new THREE.BufferAttribute(new Float32Array(n), 1),
      vb: new THREE.BufferAttribute(new Float32Array(n), 1),
    };
    for (let i = 0; i < n; i++) { attrs.va.array[i] = st.mask[i]; attrs.vb.array[i] = st.mask[i]; }
    const side = new Float32Array(n);
    side.fill(1, n / 2);
    geo.setAttribute('position', attrs.a);
    geo.setAttribute('posB', attrs.b);
    geo.setAttribute('visA', attrs.va);
    geo.setAttribute('visB', attrs.vb);
    geo.setAttribute('side', new THREE.BufferAttribute(side, 1));
    geo.setAttribute('rgb', new THREE.BufferAttribute(pd.colors, 3, true));
    points = new THREE.Points(geo, material);
    points.frustumCulled = false;
    scene.add(points);
    cur = st;
    morph.dur = 0;
    morph.done = null;
    fromMarks = st.marks;
    toMarks = st.marks;
    fromCam = st.cam;
    toCam = st.cam;
    material.uniforms.uPixA.value = st.pix;
    material.uniforms.uPixB.value = st.pix;
    buildMarkers(pd.meta.clicks.length);
    active.k = 0;
    active.ghost = 0;
    fitView(st);
    resetView(true);
    appear = { t0: performance.now(), dur: reduced ? 0 : 700 };
    kick();
  }

  function eNow(now) {
    if (morph.dur <= 0) return 1;
    return ease(clamp((now - morph.t0 - morph.delay) / morph.dur, 0, 1));
  }

  const lerp3 = (p, q, e) => (p && q ? [p[0] + (q[0] - p[0]) * e, p[1] + (q[1] - p[1]) * e, p[2] + (q[2] - p[2]) * e] : p || q);
  function marksAt(e) {
    return toMarks.map((m, i) => ({ a: lerp3(fromMarks[i] && fromMarks[i].a, m.a, e), b: lerp3(fromMarks[i] && fromMarks[i].b, m.b, e) }));
  }
  function camAt(e) {
    if (!fromCam || !toCam) return e < 0.5 ? fromCam : toCam;
    return { p: fromCam.p.clone().lerp(toCam.p, e), q: fromCam.q.clone().slerp(toCam.q, e) };
  }

  function goTo(st, delay, done) {
    if (!points || !st) return;
    const now = performance.now();
    const e = eNow(now);
    const A = attrs.a.array;
    const B = attrs.b.array;
    const vA = attrs.va.array;
    const vB = attrs.vb.array;
    const n = vA.length;
    fromMarks = marksAt(e);
    fromCam = camAt(e);
    const u = material.uniforms;
    u.uPixA.value += (u.uPixB.value - u.uPixA.value) * e;
    u.uPixB.value = st.pix;
    if (e >= 1) {
      A.set(B);
      vA.set(vB);
    } else if (e > 0) {
      for (let i = 0; i < A.length; i++) A[i] += (B[i] - A[i]) * e;
      for (let i = 0; i < n; i++) vA[i] += (vB[i] - vA[i]) * e;
    }
    B.set(st.pos);
    for (let i = 0; i < n; i++) {
      const m = st.mask[i];
      vB[i] = m;
      if (!m && vA[i] > 0) { B[3 * i] = A[3 * i]; B[3 * i + 1] = A[3 * i + 1]; B[3 * i + 2] = A[3 * i + 2]; }
      if (m && vA[i] === 0) { A[3 * i] = B[3 * i]; A[3 * i + 1] = B[3 * i + 1]; A[3 * i + 2] = B[3 * i + 2]; }
    }
    for (const at of Object.values(attrs)) at.needsUpdate = true;
    toMarks = st.marks;
    toCam = st.cam;
    cur = st;
    morph.t0 = now;
    morph.delay = delay || 0;
    morph.dur = reduced ? 0 : MORPH_MS;
    morph.done = done || null;
    if (morph.dur === 0 && morph.done) { const d = morph.done; morph.done = null; d(); }
    kick();
  }

  function setActive(k, ghost) { active.k = k; active.ghost = ghost; kick(); }
  let hotIdx = -1;
  function setHot(i) { hotIdx = i; kick(); }

  // ---------- render loop (runs only while the viewer is on screen) ----------
  let raf = 0;
  let visible = false;
  let last = performance.now();
  const q0 = new THREE.Vector3(0, 1, 0);
  const dirV = new THREE.Vector3();
  function kick() { if (!raf && visible && !document.hidden) { last = performance.now(); raf = requestAnimationFrame(tick); } }

  // tick() draws one frame and schedules the next only while something moves (morph, fade-in, camera sway / inertia /
  // reset, markers easing, the pulsing ghost); input and state changes call kick() to restart it.
  function tick(now) {
    raf = 0;
    if (!visible || document.hidden) return;
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    resize();
    if (!pd) { renderer.render(scene, camera); return; }
    let moving = !!orbit.back || (!orbit.user && !reduced)
      || (orbit.user && (ptrs.size > 0 || Math.abs(orbit.vAz) > 1e-5 || Math.abs(orbit.vEl) > 1e-5));

    // camera: inertia, auto sway until the user interacts, animated reset
    if (orbit.back) {
      const h = orbit.back.to || orbit.home;
      const f = orbit.back.from;
      const u = ease(clamp((now - orbit.back.t0) / 650, 0, 1));
      orbit.target.copy(f.target).lerp(h.target, u);
      orbit.az = f.az + (h.az - f.az) * u;
      orbit.el = f.el + (h.el - f.el) * u;
      orbit.dist = f.dist + (h.dist - f.dist) * u;
      orbit.up.copy(f.up).lerp(h.up || UP1, u).normalize();
      if (u >= 1) orbit.back = null;
    } else if (!orbit.user) {
      if (!reduced) orbit.sway += dt;
      orbit.az = (orbit.base || orbit.home).az + 0.32 * Math.sin((orbit.sway * 2 * Math.PI) / 18);
    } else if (!ptrs.size) {
      const decay = 0.9 ** (dt * 60);
      orbit.vAz *= decay;
      orbit.vEl *= decay;
      if (Math.abs(orbit.vAz) > 1e-5 || Math.abs(orbit.vEl) > 1e-5) {
        orbit.az += orbit.vAz;
        orbit.el = clamp(orbit.el + orbit.vEl, -1.3, 1.3);
      }
    }
    placeCamera();

    // geometry morph
    const e = eNow(now);
    material.uniforms.uT.value = e;
    material.uniforms.uAppear.value = appear.dur ? ease(clamp((now - appear.t0) / appear.dur, 0, 1)) : 1;
    if (e >= 1 && morph.done) { const d = morph.done; morph.done = null; d(); }
    if (e < 1 || (appear.dur && now - appear.t0 < appear.dur)) moving = true;

    // camera 2 frustum
    const fr = 0.045 * orbit.home.dist;
    cam1.matrix.makeScale(fr, fr, fr);
    const c = camAt(e);
    if (c) material.uniforms.uC2.value.copy(c.p);
    cam2.visible = !!c;
    if (c) cam2.matrix.compose(c.p, c.q, new THREE.Vector3(fr, fr, fr));
    camLabel(0, tmp.set(0, -fr * 0.6, 0));
    camLabel(1, c ? tmp.set(0, -fr * 0.6, 0).applyQuaternion(c.q).add(c.p) : null);

    // click markers
    const mk = marksAt(e);
    const k = dt * 9;
    markers.forEach((m, i) => {
      const ghost = i >= active.k && i === active.ghost;
      const target = i < active.k ? 1 : ghost ? 0.72 : 0;
      m.vis += (target - m.vis) * Math.min(1, reduced ? 1 : k * 1.4);
      m.g += ((ghost ? 1 : 0) - m.g) * Math.min(1, reduced ? 1 : k * 1.4);
      m.hot += ((hotIdx === i ? 1 : 0) - m.hot) * Math.min(1, k * 1.6);
      if (Math.abs(target - m.vis) > 0.002 || Math.abs((ghost ? 1 : 0) - m.g) > 0.002
        || Math.abs((hotIdx === i ? 1 : 0) - m.hot) > 0.002 || (ghost && !reduced)) moving = true;
      const dim = (hotIdx >= 0 && hotIdx !== i ? 0.3 : 1) * (1 - 0.42 * m.g);
      const p = mk[i];
      m.okA = !!(p && p.a);
      m.okB = !!(p && p.b);
      if (m.okA) m.a.fromArray(p.a);
      if (m.okB) m.b.fromArray(p.b);
      m.group.visible = m.vis > 0.01 && (m.okA || m.okB);
      if (!m.group.visible) return;
      // the ghost breathes with the pulsing ring in the photos (1.6 s)
      const pulse = ghost && !reduced ? 1 + 0.14 * Math.sin((now / 1600) * 2 * Math.PI) : 1;
      const sc = m.vis * (1 + 0.45 * m.hot) * pulse;
      m.ends.forEach((end, j) => {
        const ok = j === 0 ? m.okA : m.okB;
        const v = j === 0 ? m.a : m.b;
        end.s.visible = ok;
        end.o.visible = ok && m.g < 0.5;
        if (!ok) return;
        const r = camera.position.distanceTo(v) * 0.011 * sc;
        end.s.position.copy(v);
        end.s.scale.setScalar(r);
        end.o.position.copy(v);
        end.o.scale.setScalar(r * 1.32);
      });
      m.fill.opacity = dim;
      m.barMat.opacity = 0.85 * dim;
      m.bar.visible = m.okA && m.okB;
      if (m.bar.visible) {
        dirV.subVectors(m.b, m.a);
        const len = dirV.length();
        m.bar.visible = len > 1e-4;
        if (m.bar.visible) {
          const mid = tmp.addVectors(m.a, m.b).multiplyScalar(0.5);
          const w = camera.position.distanceTo(mid) * 0.0032 * sc;
          m.bar.position.copy(mid);
          m.bar.quaternion.setFromUnitVectors(q0, dirV.multiplyScalar(1 / len));
          m.bar.scale.set(w, len, w);
        }
      }
    });
    outlineMat.opacity = hotIdx >= 0 ? 0.6 : 1;
    renderer.render(scene, camera);
    if (moving) raf = requestAnimationFrame(tick);
  }

  // HTML label next to a camera centre (hidden when behind the view camera or off the canvas)
  function camLabel(i, p) {
    const el = ui.cams[i];
    if (!el) return;
    let on = !!p;
    if (on) {
      p.project(camera);
      const x = (p.x * 0.5 + 0.5) * cv.clientWidth;
      const y = (0.5 - p.y * 0.5) * cv.clientHeight;
      on = p.z < 1 && x > 8 && x < cv.clientWidth - 70 && y > 44 && y < cv.clientHeight - 70;
      if (on) el.style.transform = `translate(${Math.round(x + 6)}px, ${Math.round(y - 16)}px)`;
    }
    el.classList.toggle('is-on', on);
  }

  function resize() {
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    const pr = renderer.getPixelRatio();
    if (cv.width !== Math.round(w * pr) || cv.height !== Math.round(h * pr)) {
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(1, h);
      camera.updateProjectionMatrix();
    }
    material.uniforms.uProj.value = (h * pr) / (2 * Math.tan(((FOV / 2) * Math.PI) / 180));
    material.uniforms.uMin.value = pr;
    material.uniforms.uMax.value = 7 * pr;
  }

  new IntersectionObserver((entries) => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible) kick();
  }).observe(ui.viewer);
  document.addEventListener('visibilitychange', kick);
  new ResizeObserver(kick).observe(cv);

  return { decodeState, showPair, goTo, setActive, setHot, resetView: (which) => { resetView(false, which); kick(); } };
}
