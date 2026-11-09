// Click3R hero player. ES module, no globals.
// Plays the short clips of assets/media/hero/manifest.json as a playlist:
// - random order without repeats (reshuffled after each round, never the same clip twice in a row);
// - crossfade (XF s) between two stacked <video> elements: the incoming clip fades in on top, the outgoing one
//   stays opaque underneath until the fade ends; only the next clip is preloaded, and only once the current one is
//   fully buffered, so on a slow connection the two downloads do not compete; only the shown clip gets a poster;
// - square clips (file_sq) under 700 px (the hero's phone media query), wide clips otherwise;
// - pauses while the hero is off screen or the tab is hidden; the toggle button pauses / resumes;
// - prefers-reduced-motion: the first manifest clip's poster, no autoplay (the toggle starts the playlist). Save-Data and
//   slow connections still autoplay.
// base.js initHero() only handles [data-hero-video], so it leaves this player alone.

const XF = 0.4;
const root = document.querySelector('[data-hero-player]');

if (root) {
  const hero = root.closest('.hero') || root;
  const vids = Array.from(root.querySelectorAll('video'));
  const toggle = root.querySelector('[data-hero-toggle]');
  const narrow = window.matchMedia('(max-width: 699px)');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const stamp = root.dataset.v ? `?v=${root.dataset.v}` : '';
  const st = { clips: [], order: [], pos: 0, cur: 0, userPaused: reduced, inView: true, started: false, fading: false,
    fails: 0 };

  const media = (clip, wide, square) => `${narrow.matches && clip[square] ? clip[square] : clip[wide]}${stamp}`;
  const shuffle = (n, avoid) => {
    const a = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    if (n > 1 && a[0] === avoid) [a[0], a[1]] = [a[1], a[0]];
    return a;
  };
  // the clip at playlist position p; a new shuffled round is appended when needed
  const clipAt = (p) => {
    while (p >= st.order.length) st.order = st.order.concat(shuffle(st.clips.length, st.order[st.order.length - 1]));
    return st.clips[st.order[p]];
  };
  const other = () => vids[1 - st.cur];
  const shouldPlay = () => st.started && !st.userPaused && st.inView && !document.hidden;

  // shown: the clip is on screen now and needs its poster until the first frame; a preloaded clip gets none
  const load = (el, clip, shown) => {
    el.dataset.clip = clip.id;
    if (shown) el.poster = media(clip, 'poster', 'poster_sq');
    else el.removeAttribute('poster');
    el.preload = 'auto';
    el.src = media(clip, 'file', 'file_sq');
  };
  const buffered = (el) => el.duration > 0 && el.buffered.length > 0
    && el.buffered.end(el.buffered.length - 1) >= el.duration - 0.25;
  const preloadNext = () => {  // only while the hero is on screen: off screen the bandwidth goes to the rest of the page
    if (!st.inView) return;
    const b = other();
    if (!st.fading && b.dataset.clip !== clipAt(st.pos + 1).id) load(b, clipAt(st.pos + 1), false);
  };
  const play = (el) => {
    const p = el.play();
    if (p) {
      p.then(null, (e) => {
        if (e && e.name === 'NotAllowedError') { st.userPaused = true; setPressed(); }  // autoplay refused
      });
    }
  };
  const setPressed = () => {
    if (!toggle) return;
    toggle.setAttribute('aria-pressed', String(st.userPaused));
    toggle.setAttribute('aria-label', st.userPaused ? 'Play video' : 'Pause video');
  };
  const sync = () => {
    const a = vids[st.cur];
    if (shouldPlay()) {
      if (a.paused) play(a);
    } else {
      for (const el of vids) if (!el.paused) el.pause();
    }
    setPressed();
  };

  const advance = () => {
    if (st.fading) return;
    st.fading = true;
    const a = vids[st.cur], b = other();
    st.pos += 1;
    const next = clipAt(st.pos);
    if (b.dataset.clip !== next.id) load(b, next, true);
    else if (b.readyState < 2) b.poster = media(next, 'poster', 'poster_sq');  // no frame yet: show its poster
    if (b.readyState > 0) b.currentTime = 0;
    b.classList.add('is-on', 'is-top');
    a.classList.remove('is-top');
    st.cur = 1 - st.cur;
    if (shouldPlay()) play(b);
    setTimeout(() => {
      a.pause();
      a.classList.remove('is-on');
      st.fading = false;
      if (buffered(vids[st.cur])) preloadNext();  // else tick() / 'progress' preload it once the new clip is buffered
    }, XF * 1000 + 60);
  };

  // start the crossfade XF before the end of the current clip
  const tick = () => {
    const a = vids[st.cur];
    if (!st.fading && a.duration && a.currentTime >= a.duration - XF) advance();
    else if (!st.fading && buffered(a)) preloadNext();
    if (!vids[st.cur].paused) requestAnimationFrame(tick);
  };

  const start = () => {
    st.started = true;
    load(vids[st.cur], clipAt(st.pos), true);
    sync();
  };

  for (const el of vids) {
    el.muted = true;
    el.playsInline = true;
    el.addEventListener('playing', () => {
      if (el !== vids[st.cur]) return;
      st.fails = 0;
      if (buffered(el)) preloadNext();
      requestAnimationFrame(tick);
    });
    el.addEventListener('progress', () => { if (el === vids[st.cur] && buffered(el)) preloadNext(); });
    el.addEventListener('ended', () => { if (el === vids[st.cur]) advance(); });
    el.addEventListener('error', () => {
      if (!el.getAttribute('src')) return;
      st.fails += 1;
      if (st.fails > 2 * st.clips.length) { hero.classList.add('hero--novideo'); return; }
      if (el === vids[st.cur]) {
        advance();
      } else if (!st.fading) {
        st.order.splice(st.pos + 1, 1);  // skip the broken clip
        load(el, clipAt(st.pos + 1), false);
      }
    });
  }

  if (toggle) {
    toggle.addEventListener('click', () => {
      st.userPaused = !st.userPaused;
      if (!st.userPaused && !st.started) start();
      else sync();
    });
  }
  new IntersectionObserver(([e]) => { st.inView = e.isIntersecting; sync(); }, { threshold: 0 }).observe(root);
  document.addEventListener('visibilitychange', sync);

  // crossing 700 px: reload the current clip in the other format at the same time, and the preloaded one
  narrow.addEventListener('change', () => {
    if (!st.clips.length) return;
    const a = vids[st.cur];
    if (!st.started) { a.poster = media(clipAt(st.pos), 'poster', 'poster_sq'); return; }
    const t = a.currentTime;
    load(a, clipAt(st.pos), true);
    a.addEventListener('loadedmetadata', () => { a.currentTime = Math.min(t, Math.max(0, a.duration - XF - 0.2)); },
      { once: true });
    sync();
    if (!st.fading) load(other(), clipAt(st.pos + 1), false);
  });

  setPressed();
  fetch(root.dataset.manifest).then((r) => (r.ok ? r.json() : null)).then((list) => {
    if (!list || !list.length) { hero.classList.add('hero--novideo'); return; }
    st.clips = list;
    if (reduced) {
      // poster of the first clip only; the toggle starts the playlist from it
      st.order = [0].concat(shuffle(list.length, -1).filter((i) => i !== 0));
      vids[st.cur].poster = media(list[0], 'poster', 'poster_sq');
      return;
    }
    st.order = shuffle(list.length, -1);
    start();
  }, () => hero.classList.add('hero--novideo'));
}
