// Modo presentación para los visores de slides: una slide por vez, a pantalla completa.
// Teclas: → / espacio / PgDn siguiente · ← / PgUp anterior · Inicio / Fin · N notas · G todas · F pantalla completa.
// La slide actual queda en la URL (#3), así se puede compartir un link a una slide.
(function () {
  const frames = [...document.querySelectorAll('.frame')];
  if (!frames.length) return;
  const notes = frames.map(f => f.nextElementSibling && f.nextElementSibling.classList.contains('note') ? f.nextElementSibling : null);
  const W = 1920, H = 1080, BAR = 56;
  let cur = 0;

  const css = document.createElement('style');
  css.textContent = `
body.present{overflow:hidden;background:#000}
body.present header,body.present .frame:not(.cur),body.present .note{display:none}
body.present .frame.cur{position:fixed;left:0;right:0;margin:0 auto;border:0;border-radius:0;aspect-ratio:auto}
body.present.notes .note.cur{display:block;position:fixed;left:0;right:0;bottom:0;width:auto;margin:0;padding:20px max(16px,calc((100vw - 1100px)/2)) ${BAR + 16}px;overflow:auto;background:#03181A;border-top:1px solid #0B2E33;color:#DEE2E6;font-size:17px;font-family:Rubik,sans-serif;font-weight:300}
.pbar{position:fixed;left:50%;bottom:12px;transform:translateX(-50%);display:flex;align-items:center;gap:4px;padding:6px;background:rgba(1,1,1,.9);border:1px solid #0B2E33;border-radius:10px;font:500 14px 'IBM Plex Mono',monospace;color:#ADB5BD;z-index:10;transition:opacity .3s;user-select:none}
body:not(.present) .pbar{position:fixed}
body.present.idle .pbar{opacity:0}
.pbar button{background:none;border:0;color:inherit;font:inherit;padding:8px 12px;border-radius:6px;cursor:pointer;min-width:40px}
.pbar button:hover{background:#046B7D;color:#FFFFFF}
.pbar button.on{color:#25F6FE}
.pbar .count{padding:0 8px;min-width:64px;text-align:center}
.pbar a{color:inherit;text-decoration:none;padding:8px 12px}
body:not(.present) .frame{cursor:pointer}
@media (max-width:600px){.pbar .opt{display:none}}
`;
  document.head.appendChild(css);

  const bar = document.createElement('div');
  bar.className = 'pbar';
  bar.innerHTML = `<a href="../../" title="Inicio">⌂</a>
<button data-a="prev" title="Anterior (←)">‹</button><span class="count"></span><button data-a="next" title="Siguiente (→)">›</button>
<button data-a="notes" title="Notas (N)">notas</button><button data-a="grid" title="Todas las slides (G)">todas</button>
<button data-a="full" class="opt" title="Pantalla completa (F)">⛶</button>`;
  document.body.appendChild(bar);
  const count = bar.querySelector('.count');
  const btn = a => bar.querySelector(`[data-a="${a}"]`);

  // El visor trae su propio checkbox de notas en el header; lo mantenemos sincronizado.
  const box = document.querySelector('header input[type=checkbox]');

  function scale(f) { f.firstElementChild.style.transform = 'scale(' + f.clientWidth / W + ')'; }

  function layout() {
    frames.forEach(f => { f.style.width = f.style.height = f.style.top = ''; });
    if (document.body.classList.contains('present')) {
      const withNotes = document.body.classList.contains('notes') && notes[cur];
      const availH = withNotes ? innerHeight * 0.62 : innerHeight;
      const w = Math.min(innerWidth, availH * W / H), h = w * H / W;
      const f = frames[cur];
      f.style.width = w + 'px';
      f.style.height = h + 'px';
      f.style.top = (withNotes ? 0 : (innerHeight - h) / 2) + 'px';
      if (withNotes) notes[cur].style.top = h + 'px';
    }
    frames.forEach(scale);
  }

  function show(i, push) {
    cur = Math.max(0, Math.min(frames.length - 1, i));
    frames.forEach((f, j) => f.classList.toggle('cur', j === cur));
    notes.forEach((n, j) => n && n.classList.toggle('cur', j === cur));
    count.textContent = (cur + 1) + ' / ' + frames.length;
    btn('prev').disabled = cur === 0;
    btn('next').disabled = cur === frames.length - 1;
    const hash = '#' + (cur + 1);
    if (location.hash !== hash) history[push ? 'pushState' : 'replaceState'](null, '', hash);
    layout();
  }

  function setNotes(on) {
    document.body.classList.toggle('notes', on);
    if (box) box.checked = on;
    btn('notes').classList.toggle('on', on);
    layout();
  }

  function setPresent(on) {
    document.body.classList.toggle('present', on);
    btn('grid').classList.toggle('on', !on);
    layout();
    if (!on) frames[cur].scrollIntoView({ block: 'center' });
  }

  function fromHash() {
    const n = parseInt(location.hash.slice(1), 10);
    return n >= 1 ? n - 1 : 0;
  }

  bar.addEventListener('click', e => {
    const a = e.target.closest('button') && e.target.closest('button').dataset.a;
    if (a === 'prev') show(cur - 1, true);
    else if (a === 'next') show(cur + 1, true);
    else if (a === 'notes') setNotes(!document.body.classList.contains('notes'));
    else if (a === 'grid') setPresent(!document.body.classList.contains('present'));
    else if (a === 'full') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });

  if (box) box.addEventListener('change', () => setNotes(box.checked));

  // En la vista "todas", tocar una slide la abre en modo presentación.
  frames.forEach((f, i) => f.addEventListener('click', () => {
    if (!document.body.classList.contains('present')) { show(i, true); setPresent(true); }
  }));

  addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || (k === ' ' && !e.shiftKey)) show(cur + 1, true);
    else if (k === 'ArrowLeft' || k === 'PageUp' || (k === ' ' && e.shiftKey)) show(cur - 1, true);
    else if (k === 'Home') show(0, true);
    else if (k === 'End') show(frames.length - 1, true);
    else if (k === 'n' || k === 'N') setNotes(!document.body.classList.contains('notes'));
    else if (k === 'g' || k === 'G' || (k === 'Escape' && !document.fullscreenElement)) setPresent(!document.body.classList.contains('present'));
    else if (k === 'f' || k === 'F') btn('full').click();
    else return;
    if (document.body.classList.contains('present') || k === 'g' || k === 'G') e.preventDefault();
  });

  let x0 = null;
  addEventListener('touchstart', e => { x0 = e.touches.length === 1 ? e.touches[0].clientX : null; }, { passive: true });
  addEventListener('touchend', e => {
    if (x0 === null || !document.body.classList.contains('present')) return;
    const dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 50) show(cur + (dx < 0 ? 1 : -1), true);
    x0 = null;
  });

  let idle;
  addEventListener('mousemove', () => {
    document.body.classList.remove('idle');
    clearTimeout(idle);
    idle = setTimeout(() => document.body.classList.add('idle'), 2500);
  });

  addEventListener('popstate', () => show(fromHash()));
  addEventListener('resize', layout);
  document.fonts && document.fonts.ready.then(layout);

  document.body.classList.add('present');
  setNotes(document.body.classList.contains('notes'));
  show(fromHash());
  setPresent(true);
})();
