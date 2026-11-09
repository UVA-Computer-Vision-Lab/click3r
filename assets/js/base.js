// Click3R project page: shared behaviour. ES module, no globals.
// - nav: transparent over the dark hero, white with a hairline after it; phone menu; 2 px reading progress; scrollspy
// - smooth in-page anchors with the nav offset
// - [data-reveal]: fade up once when it enters the viewport
// - video[data-lazy][data-src]: src attached within 400 px of the viewport, plays at >= 30 % visible, pauses otherwise;
//   data-poster (instead of poster) is set within 1200 px, so posters far down the page do not load with it
// - [data-zoom]: click opens the image (data-zoom URL or the inner <img>) in a lightbox
// - [data-copy="#selector"]: copies that element's text
// Elements added later by section scripts are picked up too (MutationObserver).

// The inline head script removes .js (shows every [data-reveal] block) if this module has not run within 3 s.
document.documentElement.classList.add('base-ready');

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const reduced = () => reducedMotion.matches;
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const navHeight = () => { const nav = $('[data-nav]'); return nav ? nav.getBoundingClientRect().height : 0; };

// Starts playback; a refused play (autoplay policy, missing file) leaves the poster and shows the controls.
function play(video) {
  const p = video.play();
  if (p) p.then(null, () => { video.dataset.playBlocked = '1'; });
}

/* ------------------------------------------------------------------ nav */
function initNav() {
  const nav = $('[data-nav]');
  if (!nav) return;
  const hero = $('.hero');
  const toggle = $('[data-nav-toggle]', nav);

  if (hero) {
    const io = new IntersectionObserver(([e]) => {
      nav.classList.toggle('nav--solid', !e.isIntersecting);
    }, { rootMargin: `-${Math.round(navHeight())}px 0px 0px 0px`, threshold: 0 });
    io.observe(hero);
  } else {
    nav.classList.add('nav--solid');
  }

  const setOpen = (open) => {
    nav.classList.toggle('nav--open', open);
    if (toggle) toggle.setAttribute('aria-expanded', String(open));
  };
  if (toggle) toggle.addEventListener('click', () => setOpen(!nav.classList.contains('nav--open')));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });
  document.addEventListener('click', (e) => { if (!nav.contains(e.target)) setOpen(false); });
  window.matchMedia('(min-width: 1100px)').addEventListener('change', () => setOpen(false));
  nav.addEventListener('click', (e) => { if (e.target.closest('a')) setOpen(false); });
}

/* ------------------------------------------------------------------ reading progress */
function initProgress() {
  const bar = $('[data-progress]');
  if (!bar) return;
  let queued = false;
  const draw = () => {
    queued = false;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
    bar.style.transform = `scaleX(${p.toFixed(4)})`;
  };
  const queue = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };
  window.addEventListener('scroll', queue, { passive: true });
  window.addEventListener('resize', queue);
  new ResizeObserver(queue).observe(document.body);
  draw();
}

/* ------------------------------------------------------------------ scrollspy */
function initSpy() {
  const links = $$('.nav__link[href^="#"]');
  if (!links.length) return;
  // a link is active for its own section and for any id listed in data-spy
  const owner = new Map();
  for (const a of links) {
    const ids = (a.dataset.spy || a.getAttribute('href').slice(1)).split(/\s+/);
    for (const id of ids) owner.set(id, a);
  }
  const visible = new Set();
  const update = () => {
    // the visible section that comes last in the page wins (the band is thin, normally one)
    let current = null;
    for (const s of $$('main > section[id]')) if (visible.has(s.id)) current = s.id;
    const active = current ? owner.get(current) : null;
    for (const a of links) {
      if (a === active) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    }
  };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) visible.add(e.target.id);
      else visible.delete(e.target.id);
    }
    update();
  }, { rootMargin: '-42% 0px -52% 0px', threshold: 0 });
  for (const s of $$('main > section[id]')) io.observe(s);
}

/* ------------------------------------------------------------------ in-page anchors */
// Sections sit flush under the nav; anything else gets 16 px of air. Media above the target can change height while
// the smooth scroll runs (lazy images, section scripts), so after it settles the scroll is re-aimed up to 3 times.
function scrollToTarget(target, toTop) {
  const behavior = reduced() ? 'auto' : 'smooth';
  const offset = () => (target.matches('main > section') ? navHeight() : navHeight() + 16);
  const goal = () => (toTop ? 0 : Math.max(0, target.getBoundingClientRect().top + window.scrollY - offset()));
  window.scrollTo({ top: goal(), behavior });
  let tries = 0;
  const started = performance.now();
  let userMoved = false;
  const stop = () => { userMoved = true; };
  for (const ev of ['wheel', 'touchstart', 'keydown']) window.addEventListener(ev, stop, { once: true, passive: true });
  const settle = () => {
    if (userMoved || performance.now() - started > 5000) return;
    const maxY = document.documentElement.scrollHeight - window.innerHeight;
    const off = Math.abs(goal() - window.scrollY);
    if (off <= 2 || tries >= 3 || (window.scrollY >= maxY - 1 && goal() >= maxY)) return;
    tries += 1;
    window.scrollTo({ top: goal(), behavior: off > 400 ? behavior : 'auto' });
    wait();
  };
  const wait = () => {
    if ('onscrollend' in window) window.addEventListener('scrollend', settle, { once: true });
    else setTimeout(settle, 900);
  };
  if (behavior === 'auto') setTimeout(settle, 120);
  else wait();
}

function initAnchors() {
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const id = decodeURIComponent(a.getAttribute('href').slice(1));
    const target = id ? document.getElementById(id) : null;
    if (!target) return;
    e.preventDefault();
    scrollToTarget(target, id === 'top');
    if (location.hash !== `#${id}`) history.pushState(null, '', `#${id}`);
    if (!target.hasAttribute('tabindex') && !target.matches('a, button, input, select, textarea')) {
      target.setAttribute('tabindex', '-1');
    }
    target.focus({ preventScroll: true });
  });
}

/* ------------------------------------------------------------------ reveal */
const revealIO = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add('is-revealed');
    revealIO.unobserve(e.target);
  }
}, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });

function setupReveal(el) {
  if (el.dataset.revealReady) return;
  el.dataset.revealReady = '1';
  if (el.dataset.revealDelay) el.style.transitionDelay = `${Number(el.dataset.revealDelay)}ms`;
  if (reduced()) { el.classList.add('is-revealed'); return; }
  revealIO.observe(el);
}

/* ------------------------------------------------------------------ lazy videos */
// Posters are lazy too: write data-poster (not poster) and it is set 1200 px before the video reaches the viewport.
function attachPoster(video) {
  if (video.dataset.poster && !video.getAttribute('poster')) video.poster = video.dataset.poster;
}

function attachSource(video) {
  if (video.dataset.attached) return;
  video.dataset.attached = '1';
  attachPoster(video);
  const sources = $$('source[data-src]', video);
  for (const s of sources) s.src = s.dataset.src;
  if (video.dataset.src) video.src = video.dataset.src;
  else if (sources.length) video.load();
}

const nearIO = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    attachSource(e.target);
    nearIO.unobserve(e.target);
  }
}, { rootMargin: '400px 0px 400px 0px', threshold: 0 });

const posterIO = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    attachPoster(e.target);
    posterIO.unobserve(e.target);
  }
}, { rootMargin: '1200px 0px 1200px 0px', threshold: 0 });

const playIO = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting && e.intersectionRatio >= 0.3) {
      attachSource(v);
      if (v.paused) play(v);
    } else if (!v.paused) {
      v.pause();
    }
  }
}, { threshold: [0, 0.3, 0.6, 1] });

function setupLazyVideo(v) {
  if (v.dataset.lazyReady) return;
  v.dataset.lazyReady = '1';
  v.muted = true;
  v.playsInline = true;
  if (v.dataset.poster) posterIO.observe(v);
  if (reduced()) {
    // poster only; the reader can start it with the controls
    v.removeAttribute('autoplay');
    v.preload = 'none';
    v.controls = true;
    nearIO.observe(v);
    return;
  }
  v.removeAttribute('autoplay');
  nearIO.observe(v);
  playIO.observe(v);
}

/* ------------------------------------------------------------------ hero video */
function initHero() {
  const hero = $('.hero');
  const v = $('[data-hero-video]');
  if (!hero || !v) return;
  const toggle = $('[data-hero-toggle]');
  const phone = window.matchMedia('(max-width: 699px)');
  let userPaused = reduced();

  const setPressed = () => {
    if (!toggle) return;
    toggle.setAttribute('aria-pressed', String(v.paused));
    toggle.setAttribute('aria-label', v.paused ? 'Play video' : 'Pause video');
  };
  const pick = () => {
    const src = phone.matches ? v.dataset.srcSquare : v.dataset.srcWide;
    if (!src || v.dataset.current === src) return;
    v.dataset.current = src;
    if (!v.dataset.posterWide) v.dataset.posterWide = v.getAttribute('poster') || '';
    const poster = phone.matches ? (v.dataset.posterSquare || v.dataset.posterWide) : v.dataset.posterWide;
    if (poster) v.poster = poster;
    hero.classList.remove('hero--novideo');
    v.preload = userPaused ? 'none' : 'auto';
    v.src = src;
    if (!userPaused) play(v);
  };
  v.muted = true;
  if (userPaused) v.removeAttribute('autoplay');
  v.addEventListener('error', () => hero.classList.add('hero--novideo'));
  v.addEventListener('play', setPressed);
  v.addEventListener('pause', setPressed);
  phone.addEventListener('change', pick);
  pick();
  setPressed();

  if (toggle) {
    toggle.addEventListener('click', () => {
      if (v.paused) { userPaused = false; play(v); } else { userPaused = true; v.pause(); }
    });
  }
  // stop decoding while the hero is off screen
  new IntersectionObserver(([e]) => {
    if (!e.isIntersecting && !v.paused) v.pause();
    else if (e.isIntersecting && v.paused && !userPaused) play(v);
  }, { threshold: 0 }).observe(v);
}

/* ------------------------------------------------------------------ zoom lightbox */
function initZoom() {
  let dialog = null;
  let img = null;
  let stage = null;

  const build = () => {
    dialog = document.createElement('dialog');
    dialog.className = 'zoom';
    dialog.setAttribute('aria-label', 'Enlarged image');
    dialog.innerHTML = '<span class="zoom__hint">Click the image to switch between fit and full size</span>'
      + '<button class="zoom__close" type="button">Close <span aria-hidden="true">✕</span></button>'
      + '<div class="zoom__stage"><img class="zoom__img" alt=""></div>';
    document.body.append(dialog);
    img = $('.zoom__img', dialog);
    stage = $('.zoom__stage', dialog);
    $('.zoom__close', dialog).addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => {
      document.documentElement.style.overflow = '';
      dialog.classList.remove('zoom--full');
    });
    stage.addEventListener('click', (e) => {
      if (e.target !== img) { dialog.close(); return; }
      // toggle fit / natural size, keeping the clicked point under the pointer
      const r = img.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width;
      const fy = (e.clientY - r.top) / r.height;
      const full = dialog.classList.toggle('zoom--full');
      if (full) {
        stage.scrollLeft = fx * img.naturalWidth - e.clientX;
        stage.scrollTop = fy * img.naturalHeight - e.clientY + 56;
      }
    });
  };

  document.addEventListener('click', (e) => {
    const trigger = e.target.closest('[data-zoom]');
    if (!trigger || e.button !== 0 || e.metaKey || e.ctrlKey) return;
    const inner = trigger.matches('img') ? trigger : $('img', trigger);
    const src = trigger.dataset.zoom || (inner && (inner.currentSrc || inner.src));
    if (!src) return;
    e.preventDefault();
    if (!dialog) build();
    img.src = src;
    img.alt = inner ? inner.alt : '';
    dialog.classList.remove('zoom--full');
    document.documentElement.style.overflow = 'hidden';
    dialog.showModal();
    stage.scrollTo(0, 0);
  });
}

/* ------------------------------------------------------------------ copy */
function initCopy() {
  const flash = (btn, text) => {
    const label = $('[data-copy-label]', btn) || btn;
    const before = label.dataset.label || label.textContent;
    label.dataset.label = before;
    label.textContent = text;
    btn.classList.add('is-done');
    clearTimeout(Number(btn.dataset.copyTimer));
    btn.dataset.copyTimer = String(setTimeout(() => { label.textContent = before; btn.classList.remove('is-done'); }, 1800));
  };
  const selectAndCopy = (el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    return document.execCommand('copy');
  };
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-copy]');
    if (!btn) return;
    const src = $(btn.dataset.copy);
    if (!src) return;
    const text = src.textContent.trim() + '\n';
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(
        () => flash(btn, 'Copied'),
        () => flash(btn, selectAndCopy(src) ? 'Copied' : 'Press Ctrl+C'),
      );
    } else {
      flash(btn, selectAndCopy(src) ? 'Copied' : 'Press Ctrl+C');
    }
  });
}

/* ------------------------------------------------------------------ scan (now and for nodes added later) */
function scan(root) {
  if (root.matches && root.matches('[data-reveal]')) setupReveal(root);
  if (root.matches && root.matches('video[data-lazy]')) setupLazyVideo(root);
  for (const el of $$('[data-reveal]', root)) setupReveal(el);
  for (const v of $$('video[data-lazy]', root)) setupLazyVideo(v);
}

initNav();
initProgress();
initSpy();
initAnchors();
initHero();
initZoom();
initCopy();
scan(document);
new MutationObserver((records) => {
  for (const r of records) for (const n of r.addedNodes) if (n.nodeType === 1) scan(n);
}).observe(document.body, { childList: true, subtree: true });
