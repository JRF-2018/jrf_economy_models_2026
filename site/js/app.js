// app.js: 画面のつなぎ。模型(Worker)、時間のつまみ、情景、つまみ、カード、新聞、セーブとロード、言葉。
import { t, setLang, getLang, applyI18n, onLang, fmt, pct } from './i18n.js';
import { Levers, Cards, BigChart, headlines, INDICATORS } from './ui/panels.js';
import { Analysis } from './ui/analysis.js';
import { Life } from './ui/life.js';
import { Sound } from './ui/sound.js';

const IM = window.IM;
const $ = s => document.querySelector(s);
const SPEEDS = { 1: 2.8, 4: 1.4, 16: 0.35 };          // 一年の秒数(×1 で昼と夜がゆっくり一巡)
const CUR_LABEL = { common: 'cur_common', float: 'cur_float', fixed: 'cur_fixed', anchor: 'cur_float' };

// ---------------------------------------------------------------- 模型の窓口(Worker。使えなければ画面の中で)
function makeBackend(onMsg) {
  const localEngine = () => new IM.Engine(m => setTimeout(() => onMsg(m), 0));
  let w = null;
  try { w = new Worker('js/worker.js'); } catch (e) { w = null; }       // file:// では Worker が作れない
  if (!w) {
    const E = localEngine();
    return { post(m) { setTimeout(() => IM.handle(E, m, r => onMsg(r)), 0); }, mode: 'local' };
  }
  let ok = false, mode = 'worker', E = null;
  const pending = [];
  w.onmessage = ev => { ok = true; pending.length = 0; onMsg(ev.data); };
  w.onerror = ev => {
    if (ok) { console.error(ev); return; }
    console.warn('Worker が使えないので、画面の中で回す');
    mode = 'local';
    E = localEngine();
    for (const m of pending.splice(0)) setTimeout(() => IM.handle(E, m, r => onMsg(r)), 0);
  };
  return {
    post(m) {
      if (mode === 'worker') { if (!ok) pending.push(m); w.postMessage(m); }
      else setTimeout(() => IM.handle(E, m, r => onMsg(r)), 0);
    },
    get mode() { return mode; },
  };
}

// ---------------------------------------------------------------- 状態
const S = {
  setup: null, branches: new Map(), cur: null, ghost: null, p: 0, playing: false, speed: 1, phase: 0, undo: [], cmdN: 0,
  sel: 0, waiting: false, quality: 'high', news: [], reqN: 0, resolvers: new Map(), loading: null, tries: 0,
};
let backend, scene = null, levers, cards, big, analysis, life;
const sound = new Sound();

function send(m) {
  return new Promise(res => {
    const req = ++S.reqN;
    S.resolvers.set(req, res);
    backend.post(Object.assign({ req }, m));
  });
}
function onMsg(m) {
  if (m.type === 'frame') {
    const b = S.branches.get(m.branch);
    if (b) {
      b.frames[m.frame.t] = m.frame;
      m.frame.news = headlines(m.frame, b.frames[m.frame.t - 1], names());
      b.end = Math.max(b.end, m.frame.t + 1);
      if (S.loading && m.branch === S.cur) showToast(t('computing') + ` ${m.frame.t + 1} / ${S.loading}`, 400);
      drawBar();
    }
  } else if (m.type === 'broken') {
    const b = S.branches.get(m.branch);
    if (b) { b.broken = m.t; drawBar(); }
  } else if (m.type === 'error') {
    console.error(m.msg);
    showToast('error: ' + m.msg.split('\n')[0], 6000);
  }
  if (m.req && S.resolvers.has(m.req)) { S.resolvers.get(m.req)(m); S.resolvers.delete(m.req); }
}

const names = () => (S.setup ? S.setup.regions : ['A', 'B', 'C']).map(r => t('country_' + r));
const isSingle = () => S.setup && S.setup.regions.length === 1;
const branch = () => S.branches.get(S.cur);
function newBranch(id, parent, fork, frames, commands) {
  const b = { id, parent, fork, frames, commands: commands || [], end: frames.length, broken: null, seen: frames.length, n: ++S.tries };
  S.branches.set(id, b);
  return b;
}

// その年のはじめのパラメータ(元帳の命令を、年 p まで当てる)
function paramsAt(i, p) {
  const r = S.setup.regions[i];
  const P = Object.assign(IM.defaultParams(), S.setup.common, (S.setup.region_kw || {})[r] || {});
  for (const c of branch().commands) {
    if (c.t > p) continue;
    if (c.type === 'param' && (c.region === i || c.region === null || c.region === undefined)) Object.assign(P, c.data);
  }
  return P;
}
function curLabel(i) {
  const w = S.setup;
  if (!w) return '';
  const cur = typeof w.currency === 'string' ? w.currency : w.currency[w.regions[i]];
  if (w.currency === 'common') return t('cur_common');
  if (typeof w.currency === 'string') return t(CUR_LABEL[w.currency]);
  const shared = w.regions.filter(r => w.currency[r] === cur).length > 1;
  if (shared) return t('cur_common') + ' (' + w.regions.filter(r => w.currency[r] === cur).join('+') + ')';
  return t((w.regimes || {})[cur] === 'float' ? 'cur_float' : 'cur_fixed');
}

// ---------------------------------------------------------------- 一年を見せる
function render() {
  const b = branch();
  if (!b) return;
  const fr = b.frames[S.p - 1], prev = b.frames[S.p - 2];
  b.seen = Math.max(b.seen, S.p);
  $('#year-num').textContent = String(S.p);
  $('#year-num2').textContent = String(S.p);
  if (scene && fr) {
    scene.setFrame(fr, prev, {
      names: names(), curLabel, dur: SPEEDS[S.speed],
      stats: i => {
        const r = fr.regions[i];
        return `<span>${t('ind_E')} ${fmt(r.E, 0)}</span><span>${t('ind_real_wage')} ${fmt(r.real_wage, 2)}</span>`
          + (S.setup.currency !== 'common' ? `<span>${t('ind_e')} ${fmt(r.e_cur, 3)}</span>` : '')
          + (r.rev_G !== undefined ? `<span class="il-rev">${t('il_grievance')} ${fmt(r.rev_G, 2)}${r.regime ? ' · ' + t('il_new_regime') : ''}</span>` : '')
          + (r.up_s !== undefined && r.up_s > 0 ? `<span class="il-up">${t('il_workshop')} ${pct(r.up_s, 0)}</span>` : '');
      },
    });
  }
  const ghost = S.ghost !== null && S.branches.get(S.ghost) ? S.branches.get(S.ghost).frames : null;
  cards.update(b.frames, S.p, S.sel, ghost);
  if (S.view === 'analysis' && analysis && analysis.tab === 'compare') { analysis.fill(); analysis.drawCompare(); }
  if (S.view === 'life' && life) life.update();
  if (big.I) big.update(b.frames, S.p, ghost, names());
  levers.show(paramsAt(S.sel, S.p), { single: S.setup.regions.length === 1, PN: fr ? fr.regions[S.sel].PN : 1 });
  // 帳簿の灯
  let worst = 0;
  if (fr) for (const r of fr.regions) worst = Math.max(worst, r.books / r.books_tol);
  const led = $('#ledger');
  led.classList.toggle('bad', worst >= 1);
  led.querySelector('.ledger-text').textContent = t(worst >= 1 ? 'ledger_bad' : 'ledger_ok');
  led.title = t('ledger_tip', { v: worst.toExponential(1) });
  // 新聞(その年までの見出し、新しい順)
  const items = [];
  for (let k = S.p; k >= 0 && items.length < 7; k--) {
    for (const c of b.commands) if (c.t === k && c.label && !c.hidden && items.length < 7) items.push({ y: k, h: { i: c.region, s: c.label }, cmd: true });
    const f = k < S.p ? b.frames[k] : null;
    if (f && f.news) for (const h of f.news) if (items.length < 7) items.push({ y: k + 1, h });
  }
  const news = $('#news-list');
  news.innerHTML = items.map(o => `<li class="${o.cmd ? 'cmd' : ''}" style="--accent:${o.h.i >= 0 ? accent(o.h.i) : '#9fb3d9'}"><b>${t('yearN', { y: o.y })}</b> ${o.h.s}</li>`).join('');
  const brk = b.broken !== null && S.p >= b.broken;
  $('#broken').classList.toggle('show', brk);
  if (brk) $('#broken').textContent = t('broken', { y: b.broken });
  // 一年だけ進んだとき(再生、一年進む)は、その年の出来事を音と大きな札で
  const ls = S.lastShown;
  if (fr && ls && ls.b === b.id && S.p === ls.p + 1) yearEvents(fr, prev);
  if (fr) sound.storm(fr.regions.some(r => r.dis_size > 0) ? 1 : 0);
  S.lastShown = { b: b.id, p: S.p };
  drawBar();
}

// ---------------------------------------------------------------- その年の出来事の音と札
function banner(big, sub, kind) {
  const el = $('#banner');
  el.className = 'banner ' + (kind || '');
  el.querySelector('.banner-big').textContent = big;
  el.querySelector('.banner-sub').textContent = sub || '';
  void el.offsetWidth;
  el.classList.add('show');
  clearTimeout(banner.tm);
  banner.tm = setTimeout(() => el.classList.remove('show'), 3200);
}
function yearEvents(fr, prev) {
  if (!prev) return;
  const nm = names();
  const sounds = [];
  let shown = false;
  fr.regions.forEach((r, i) => {
    const q = prev.regions[i];
    if (r.revolution > 0) {
      sounds.push(r.regime ? 'revolution' : 'restoration');
      if (!shown) { banner(t(r.regime ? 'bn_revolution' : 'bn_restoration'), nm[i], r.regime ? 'rev' : 'res'); shown = true; }
    }
    if (r.upgraded > 0) { sounds.push('upgrade'); if (!shown) { banner(t('bn_upgrade'), nm[i], 'up'); shown = true; } }
    if (r.dis_size > 0) {
      const h = fr.houses[i] && fr.houses[i].hit;
      const local = h && Array.from(h).reduce((a, v) => a + v, 0) < h.length * 0.5;
      sounds.push(local ? 'local' : 'disaster');
    }
    if (Math.max(1 - r.KN / q.KN, 1 - r.KL / q.KL, 1 - r.KI / q.KI) > 0.05) sounds.push('collapse');
    if (r.defaulted > 0 || (r.crisis > 0 && !(q.crisis > 0))) sounds.push('alarm');
    if (r.vote_hi !== undefined && (r.vote_hi !== q.vote_hi || r.vote_lo !== q.vote_lo)) sounds.push('election');
    if (r.up_s > 0 && !(q.up_s > 0)) sounds.push('workshop');
    if (q.up_s > 0 && !(r.up_s > 0)) sounds.push('gone');
    if ((r.capped > 0 && !(q.capped > 0)) || (r.short > 0.02 && !(q.short > 0.02))) sounds.push('shortage');
    if (r.breakups >= 5 && r.breakups > 1.5 * (q.breakups || 0)) sounds.push('breakups');
    // あなたの世帯: 代替わりと、子が生まれる
    if (fr.pl && fr.pl[i] !== null && fr.pl[i] !== undefined && q.pl_gen !== undefined) {
      if (r.pl_gen > q.pl_gen) { sounds.unshift('generation'); banner(t('bn_generation', { n: r.pl_gen }), t('bn_generation_sub', { c: nm[i] }), 'gen'); shown = true; }
      else if (r.pl_kids > q.pl_kids) { sounds.push('child'); if (S.view === 'life' && !shown) { banner(t('bn_child'), t('bn_child_sub'), 'gen'); shown = true; } }
    }
  });
  // 同じ年に重なったら、三つまで、少しずつずらして
  Array.from(new Set(sounds)).slice(0, 3).forEach((s, k) => setTimeout(() => sound.play(s), k * 450));
}
const accent = i => ['#2dd4bf', '#f59e0b', '#a78bfa'][i] || '#9fb3d9';

// ---------------------------------------------------------------- 時間のつまみ
const svg = d => `<svg viewBox="0 0 24 24">${d}</svg>`;
const FLAG_ICON = {
  param: svg('<circle cx="12" cy="12" r="5" fill="none" stroke="currentColor" stroke-width="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4" stroke="currentColor" stroke-width="3"/>'),
  disaster: svg('<path d="M13 2 4 14h7l-1 8 9-12h-7z" fill="currentColor"/>'),
  capital_loss: svg('<path d="M3 21V9l6 4V9l6 4V5h6v16z" fill="currentColor"/>'),
  dice: svg('<rect x="3" y="3" width="18" height="18" rx="4" fill="currentColor"/><circle cx="8" cy="8" r="2" fill="#111"/><circle cx="16" cy="16" r="2" fill="#111"/><circle cx="12" cy="12" r="2" fill="#111"/>'),
  fork: svg('<path d="M6 3v6a6 6 0 0 0 6 6h0a6 6 0 0 1 6 6M6 3v18" fill="none" stroke="currentColor" stroke-width="3"/>'),
  broken: svg('<path d="M5 5l14 14M19 5 5 19" stroke="currentColor" stroke-width="4"/>'),
  other: svg('<path d="M12 3l9 9-9 9-9-9z" fill="currentColor"/>'),
};
function barScale() { const b = branch(); return Math.max(100, Math.ceil(((b ? b.end : 0) + 25) / 50) * 50); }
function drawBar() {
  const b = branch();
  if (!b) return;
  const M = barScale();
  $('#bar-done').style.width = (100 * b.end / M) + '%';
  $('#bar-knob').style.left = (100 * S.p / M) + '%';
  $('#bar-played').style.width = (100 * S.p / M) + '%';
  $('#year-end').textContent = String(b.end);
  const flags = $('#bar-flags');
  const seen = new Set();
  let html = '';
  for (const c of b.commands) {
    const key = c.t + ':' + c.type;
    if (seen.has(key) || c.hidden) continue;
    seen.add(key);
    const icon = FLAG_ICON[c.type] || FLAG_ICON.other;
    html += `<span class="flag flag-${c.type}" style="left:${100 * c.t / M}%;--accent:${c.region >= 0 ? accent(c.region) : '#fff'}" title="${t('yearN', { y: c.t })}: ${(c.label || '').replace(/"/g, '')}">${icon}</span>`;
  }
  if (b.fork > 0) html += `<span class="flag fork" style="left:${100 * b.fork / M}%" title="${t('h_branch', { y: b.fork })}">${FLAG_ICON.fork}</span>`;
  if (b.broken !== null) html += `<span class="flag broken" style="left:${100 * b.broken / M}%">${FLAG_ICON.broken}</span>`;
  flags.innerHTML = html;
  const ticks = $('#bar-ticks');
  if (ticks.dataset.m !== String(M)) {
    ticks.dataset.m = String(M);
    let th = '';
    for (let y = 0; y <= M; y += M <= 200 ? 25 : 50) th += `<span style="left:${100 * y / M}%">${y}</span>`;
    ticks.innerHTML = th;
  }
  // 試しの一覧
  const list = $('#tries');
  const triesHtml = Array.from(S.branches.values()).map(o =>
    `<button class="try ${o.id === S.cur ? 'cur' : ''} ${o.id === S.ghost ? 'ghost' : ''}" data-id="${o.id}" title="${t('h_branch', { y: o.fork })}">${t('branch_n', { n: o.n })}</button>`).join('');
  if (list.dataset.html !== triesHtml) { list.dataset.html = triesHtml; list.innerHTML = triesHtml; }
  // 押している最中に中身を書き換えると、押したことが消える(×16 で止められないことがあった)。変わったときだけ書き換える
  const pb = $('#play');
  if (pb.dataset.on !== String(S.playing)) {
    pb.dataset.on = String(S.playing);
    pb.innerHTML = S.playing ? '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>' : '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
  }
}
function seek(p) {
  const b = branch();
  if (S.iron && S.view === 'life') p = Math.max(p, b.seen);       // やり直しなし: 見た年より前へは戻れない
  S.p = Math.max(1, Math.min(Math.round(p), b.end));
  S.phase = 0;
  render();
}
async function ensure(until) {
  const b = branch();
  if (b.end >= until || b.broken !== null || S.waiting) return;
  S.waiting = true;
  await send({ type: 'compute', branch: b.id, until });
  S.waiting = false;
}
function setPlaying(on) {
  S.playing = on;
  drawBar();
}

// 命令を出す: 今の年 p のはじめに当てる。p より先に済んだ年があれば、見た年なら枝を分け、見ていなければ捨てて足す
async function command(cmds, label) {
  const b = branch();
  const year = S.p;
  for (const c of cmds) { if (label && !c.label) c.label = label; c.id = 'c' + (++S.cmdN) + '-' + Date.now().toString(36); }
  const seenAhead = b.end > year && b.seen > year;
  const r = await send({ type: 'command', branch: b.id, year, cmds, truncate: !seenAhead });
  if (r.branch !== b.id) {
    const nb = newBranch(r.branch, b.id, year, b.frames.slice(0, year), r.commands);
    S.undo.push({ kind: 'fork', from: b.id, to: nb.id, ghost: S.ghost, p: year });
    S.ghost = b.id;
    S.cur = nb.id;
    showToast(t('h_branch', { y: year }), 1800);
    sound.play('fork');
  } else {
    S.undo.push({ kind: 'append', branch: b.id, ids: cmds.map(c => c.id), year, p: year });
    b.commands = r.commands;
    if (r.end < b.end) { b.frames.length = r.end; b.end = r.end; }
    b.end = r.end;
    if (b.broken !== null && b.broken >= year) b.broken = null;
  }
  render();
  await ensure(S.p + 1);
}

// 取り消し: 最後に出した命令を、出す前の様子に戻す(枝を分けたなら元の枝へ、足したなら除いて計算し直す)
async function undo() {
  if (S.iron && S.view === 'life') return;
  const u = S.undo.pop();
  if (!u) { showToast(t('undo_none'), 1200); return; }
  setPlaying(false);
  if (u.kind === 'fork') {
    if (S.cur === u.to) { S.cur = u.from; S.ghost = u.ghost; }
    S.branches.delete(u.to);
    await send({ type: 'drop', branch: u.to });
    if (S.ghost === u.to) S.ghost = null;
    seek(Math.min(u.p, branch().end));
  } else {
    const b = S.branches.get(u.branch);
    if (!b) return;
    const r = await send({ type: 'uncommand', branch: b.id, ids: u.ids, year: u.year });
    b.commands = r.commands;
    if (r.end < b.end) b.frames.length = r.end;
    b.end = r.end;
    b.broken = null;
    S.cur = b.id;
    S.p = Math.min(u.p, b.end);
    await ensure(S.p + 1);
    seek(u.p);
  }
  showToast(t('undone'), 1200);
  sound.play('undo');
}

// ---------------------------------------------------------------- 毎こま(再生)
let last = performance.now();
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  const b = branch();
  if (!b) return;
  if (S.playing) {
    const dur = SPEEDS[S.speed];
    S.phase += dt / dur;
    if (b.end < S.p + 1) ensure(S.p + 1);
    if (S.phase >= 1) {
      if (b.broken !== null && S.p >= b.broken) { setPlaying(false); S.phase = 0; }
      else if (b.frames[S.p]) { S.phase -= 1; S.p += 1; render(); ensure(S.p + 1); }
      else S.phase = 0.999;
    }
  }
  // 一年 = 昼と夜の一巡(×16 では四年で一巡)。止めているときは、昼の光で見せる
  if (scene) {
    if (S.playing) S.shown = S.speed === 16 ? ((S.p + S.phase) / 4) % 1 : S.phase % 1;
    else S.shown = (S.shown === undefined ? 0.22 : S.shown) + (0.22 - (S.shown === undefined ? 0.22 : S.shown)) * Math.min(dt * 2, 1);
    scene.setPhase(S.shown);
  }
}

// ---------------------------------------------------------------- セーブとロード
function recipe() {
  const b = branch();
  return { version: IM.ENGINE_VERSION, setup: S.setup, commands: b.commands, end: b.end, p: S.p, sel: S.sel };
}
function store(k, v) {
  try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); return true; } catch (e) { return null; }
}
async function loadRecipe(rc) {
  setPlaying(false);
  S.loading = rc.end || 1;
  const r = await send({ type: 'load', recipe: rc });
  S.branches.clear();
  S.tries = 0;
  S.setup = r.setup;
  newBranch(r.branch, null, 0, [], r.commands);
  S.cur = r.branch;
  S.ghost = null;
  showToast(t('loaded', { y: rc.end }), 2000);
  await send({ type: 'compute', branch: r.branch, until: Math.max(rc.end || 1, 1) });
  S.loading = null;
  S.undo = [];
  S.sel = rc.sel || 0;
  cards.hide = I => (I.key === 'e' && (S.setup.currency === 'common' || S.setup.regions.length === 1)) || (I.key === 'tb_Y' && S.setup.regions.length === 1)
    || (I.only === 'single' && S.setup.regions.length !== 1);
  if (scene) scene.setCount(S.setup.regions.length);
  cards.build();
  if (scene) scene.select(S.sel);
  buildCountryPicker();
  seek(rc.p || rc.end || 1);
}
function b64e(s) { return btoa(unescape(encodeURIComponent(s))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); }
function b64d(s) { s = s.replace(/-/g, '+').replace(/_/g, '/'); while (s.length % 4) s += '='; return decodeURIComponent(escape(atob(s))); }

function setupMenus() {
  const slots = $('#slots');
  const draw = () => {
    let html = '';
    for (let n = 1; n <= 3; n++) {
      const raw = store('im-sandbox-slot-' + n);
      let info = t('empty');
      if (raw) { try { const o = JSON.parse(raw); info = `${t('yearN', { y: o.p })} · ${new Date(o.savedAt).toLocaleString(getLang() === 'ja' ? 'ja-JP' : 'en-US')}`; } catch (e) { /* 壊れた枠 */ } }
      html += `<div class="slot"><span>${t('slot', { n })}</span><small>${info}</small>
        <button data-save="${n}">${t('save')}</button><button data-load="${n}" ${raw ? '' : 'disabled'}>${t('load')}</button></div>`;
    }
    slots.innerHTML = html;
  };
  $('#btn-save').addEventListener('click', () => { draw(); $('#savebox').classList.toggle('open'); });
  slots.addEventListener('click', async e => {
    const s = e.target.dataset.save, l = e.target.dataset.load;
    if (s) {
      const rc = Object.assign(recipe(), { savedAt: Date.now() });
      if (store('im-sandbox-slot-' + s, JSON.stringify(rc)) === null) showToast(t('nostore'), 3000);
      else showToast(t('saved', { n: s }), 1500);
      draw();
    }
    if (l) {
      const raw = store('im-sandbox-slot-' + l);
      if (raw) { $('#savebox').classList.remove('open'); await loadRecipe(JSON.parse(raw)); }
    }
  });
  $('#btn-export').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(recipe(), null, 1)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `im_sandbox_${S.p}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
  $('#btn-import').addEventListener('click', () => $('#file').click());
  $('#file').addEventListener('change', async e => {
    const f = e.target.files[0];
    if (!f) return;
    const rc = JSON.parse(await f.text());
    $('#savebox').classList.remove('open');
    await loadRecipe(rc);
  });
  $('#btn-share').addEventListener('click', async () => {
    const url = location.href.split('#')[0] + '#r=' + b64e(JSON.stringify(recipe()));
    try { await navigator.clipboard.writeText(url); showToast(t('copied'), 1500); } catch (e) { prompt('URL', url); }
  });
}

// ---------------------------------------------------------------- 出来事とサイコロ
function setupEvents() {
  const ev = {
    disaster: () => [{ type: 'disaster', region: S.sel, data: { size: 0.3, scope: 'common' } }],
    local: () => [{ type: 'disaster', region: S.sel, data: { size: 0.4, scope: 'idio' } }],
    halve: () => [{ type: 'capital_loss', region: S.sel, data: { share: 0.5 } }],
    hike: () => [{ type: 'param', region: S.sel, data: { r_cb: Math.min(paramsAt(S.sel, S.p).r_cb + 0.03, 0.2) } }],
    euphoria: () => [{ type: 'param', region: S.sel, data: { theta_shock: 0.12 } }, { type: 'param', region: S.sel, dt: 8, hidden: true, data: { theta_shock: 0.0 } }],
    // 段 B7: 新しい財の工房を開く(開き直す)。育つかどうかは、工房への補助と、作った量の累計(ライトの法則)しだい
    workshop: () => [{ type: 'param', region: S.sel, data: { up_gen: (paramsAt(S.sel, S.p).up_gen || 0) + 1 } }],
  };
  document.querySelectorAll('[data-ev]').forEach(btn => btn.addEventListener('click', () => {
    const k = btn.dataset.ev;
    const cmds = ev[k]();
    command(cmds, `${names()[S.sel]}: ${t('ev_' + k)}`);
    btn.classList.remove('pulse'); void btn.offsetWidth; btn.classList.add('pulse');
    sound.play({ hike: 'hike', euphoria: 'euphoria', workshop: 'workshop' }[k] || 'click');
  }));
  const dice = $('#dice');
  // 長押し: 平行世界(分析の画面の扇の図)
  let pressT = null, longFired = false;
  dice.addEventListener('pointerdown', () => { longFired = false; pressT = setTimeout(() => { longFired = true; setView('analysis'); analysis.showTab('fan'); analysis.runFan(); }, 600); });
  dice.addEventListener('pointerup', () => clearTimeout(pressT));
  dice.addEventListener('pointerleave', () => clearTimeout(pressT));
  dice.addEventListener('click', () => {
    if (longFired || (S.iron && S.view === 'life')) return;
    const seed = (Math.random() * 4294967296) >>> 0;
    dice.classList.remove('roll'); void dice.offsetWidth; dice.classList.add('roll');
    sound.play('dice');
    command([{ type: 'dice', region: -1, data: { seed } }], t('h_dice', { n: S.tries + 1 }));
  });
}

// ---------------------------------------------------------------- 画面の切り替え(箱庭 / 分析 / 一人の暮らし)
function setView(v) {
  S.view = v;
  document.querySelectorAll('[data-view]').forEach(x => x.classList.toggle('on', x.dataset.view === v));
  document.body.classList.toggle('view-analysis', v === 'analysis');
  document.body.classList.toggle('view-life', v === 'life');
  $('#soon').classList.toggle('show', v === 'life' && !window.__life);
  if (v === 'analysis') analysis.open();
  if (v === 'life' && window.__life) window.__life.open();
  render();
}

// ---------------------------------------------------------------- 国を選ぶ
function buildCountryPicker() {
  const box = $('#countries');
  box.innerHTML = S.setup.regions.map((r, i) => `<button class="ctry ${i === S.sel ? 'sel' : ''}" data-i="${i}" style="--accent:${accent(i)}">
     <span class="ctry-flag"></span><span class="ctry-name">${t('country_' + r)}</span><small>${curLabel(i)}</small></button>`).join('');
}
function selectCountry(i, focus = true) {
  S.sel = i;
  buildCountryPicker();
  if (scene) { scene.select(i); if (focus) scene.focus(i); }
  render();
}

// ---------------------------------------------------------------- 小さな知らせ
let toastTimer = null;
function showToast(msg, ms = 1500) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), ms);
}

// ---------------------------------------------------------------- 始める
async function start() {
  document.documentElement.lang = getLang();
  applyI18n(document);
  backend = makeBackend(onMsg);
  levers = new Levers($('#levers'), (key, v) => {
    const label = `${names()[S.sel]}: ${t('p_' + key)} → ${typeof v === 'number' ? (key === 'tau_k' && v < 0 ? t('p_tau_k_same') : fmt(v, 3)) : typeof v === 'boolean' ? t('p_bool_' + v) : t('p_' + key + '_' + v)}`;
    const data = { [key]: v };
    if (key === 'pension' && v !== 'none' && !(paramsAt(S.sel, S.p).ret_age > 0)) data.ret_age = 45;   // 年金には引退の年齢が要る
    command([{ type: 'param', region: S.sel, data }], label);
  });
  levers.build();
  big = new BigChart($('#modal'));
  cards = new Cards($('#cards'), I => {
    const ghost = S.ghost !== null && S.branches.get(S.ghost) ? S.branches.get(S.ghost).frames : null;
    big.open(I, branch().frames, S.p, ghost, names());
  });
  cards.build();
  // 情景(three.js が読めないときは、グラフだけで動く)
  try {
    const mod = await import('./ui/scene.js');
    scene = new mod.Scene($('#scene'), 3, 400);
    scene.onPick(i => selectCountry(i));
  } catch (e) {
    console.warn('three.js の情景なし', e);
    $('#scene').classList.add('noscene');
  }
  // 時間のつまみ
  $('#play').addEventListener('click', () => setPlaying(!S.playing));
  $('#step-back').addEventListener('click', () => { setPlaying(false); seek(S.p - 1); });
  $('#step-fwd').addEventListener('click', async () => { setPlaying(false); await ensure(S.p + 1); seek(S.p + 1); });
  $('#to-start').addEventListener('click', () => { setPlaying(false); seek(1); });
  $('#to-end').addEventListener('click', () => { setPlaying(false); seek(branch().end); });
  document.querySelectorAll('[data-speed]').forEach(b => b.addEventListener('click', () => {
    S.speed = +b.dataset.speed;
    document.querySelectorAll('[data-speed]').forEach(x => x.classList.toggle('on', x === b));
  }));
  const bar = $('#bar');
  let drag = false;
  const at = e => { const r = bar.getBoundingClientRect(); return (e.clientX - r.left) / r.width * barScale(); };
  bar.addEventListener('pointerdown', e => { drag = true; bar.setPointerCapture(e.pointerId); setPlaying(false); seek(at(e)); });
  bar.addEventListener('pointermove', e => { if (drag) seek(at(e)); });
  bar.addEventListener('pointerup', () => { drag = false; });
  $('#tries').addEventListener('click', e => {
    const id = e.target.dataset.id;
    if (id === undefined) return;
    const nid = +id;
    if (nid === S.cur) return;
    S.ghost = S.cur;
    S.cur = nid;
    seek(Math.min(S.p, branch().end));
  });
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if ((e.ctrlKey || e.metaKey) && e.code === 'KeyZ') { e.preventDefault(); undo(); return; }
    if (e.code === 'Escape') { document.querySelectorAll('.popover.open').forEach(p => p.classList.remove('open')); big.close(); return; }
    if (e.code === 'Space') { e.preventDefault(); setPlaying(!S.playing); }
    if (e.code === 'ArrowLeft') { setPlaying(false); seek(S.p - 1); }
    if (e.code === 'ArrowRight') { setPlaying(false); ensure(S.p + 1).then(() => seek(S.p + 1)); }
  });
  $('#countries').addEventListener('click', e => { const b = e.target.closest('.ctry'); if (b) selectCountry(+b.dataset.i); });
  $('#btn-overview').addEventListener('click', () => { if (scene) scene.focus(null); });
  // 音: ブラウザは、人が一度押すまで音を出させない。初めの押し(どこでも)で鳴らせるようにする
  const unlock = () => { sound.unlock(); drawSound(); };
  document.addEventListener('pointerdown', unlock, true);
  document.addEventListener('keydown', unlock, true);
  const drawSound = () => {
    const b = $('#btn-sound');
    b.classList.toggle('off', !sound.on);
    b.classList.toggle('wait', sound.on && !sound.ready);
    b.title = t(!sound.on ? 'snd_off' : sound.ready ? 'snd_on' : 'snd_wait');
    b.innerHTML = sound.on ? '<svg viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>'
                           : '<svg viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 9l5 6M21 9l-5 6" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>';
  };
  sound.listeners.push(drawSound);
  $('#btn-sound').addEventListener('click', () => { sound.unlock(); sound.setOn(!sound.on); drawSound(); if (sound.on) sound.play('click'); });
  $('#set-vol').value = sound.vol;
  $('#set-vol').addEventListener('input', e => { sound.setVol(+e.target.value); });
  $('#set-vol').addEventListener('change', () => sound.play('election'));
  onLang(drawSound);
  drawSound();
  // 小さな画面: 左右の板と出来事の列は引き出し(上の帯のボタンで開け閉め。左右は同時に一つ)
  document.querySelectorAll('.drawer-btn').forEach(b => b.addEventListener('click', () => {
    const d = b.dataset.drawer, cls = 'drawer-' + d, on = !document.body.classList.contains(cls);
    if (d !== 'events') document.body.classList.remove('drawer-left', 'drawer-right');
    document.body.classList.toggle(cls, on);
    setTimeout(() => { cards.cards.forEach(o => o.chart && o.chart.resize()); if (life && life.chart) life.chart.resize(); }, 50);
  }));
  $('#btn-undo').addEventListener('click', () => undo());
  // 浮かぶ板は、✕・外を押す・Esc で閉じる
  document.querySelectorAll('.popover').forEach(p => {
    const x = document.createElement('button');
    x.className = 'pop-close'; x.textContent = '✕';
    x.addEventListener('click', () => p.classList.remove('open'));
    p.prepend(x);
  });
  document.addEventListener('pointerdown', e => {
    document.querySelectorAll('.popover.open').forEach(p => {
      const opener = { savebox: '#btn-save', settings: '#btn-settings', helpbox: '#btn-help' }[p.id];
      if (!p.contains(e.target) && !(opener && e.target.closest(opener))) p.classList.remove('open');
    });
  });
  // 言葉
  document.querySelectorAll('[data-lang]').forEach(b => b.addEventListener('click', () => setLang(b.dataset.lang)));
  onLang(() => {
    if (analysis) analysis.fill();
    if (life) { life.build(); }
    document.querySelectorAll('[data-lang]').forEach(b => b.classList.toggle('on', b.dataset.lang === getLang()));
    levers.build(); cards.build(); buildCountryPicker();
    for (const b of S.branches.values()) b.frames.forEach((f, k) => { if (f) f.news = headlines(f, b.frames[k - 1], names()); });
    render();
  });
  document.querySelectorAll('[data-lang]').forEach(b => b.classList.toggle('on', b.dataset.lang === getLang()));
  // 画面の切り替え(分析と一人の暮らしは次の段)
  analysis = new Analysis($('#analysis'), {
    send, branches: () => S.branches, cur: () => S.cur, ghost: () => S.ghost, p: () => S.p, sel: () => S.sel,
    setup: () => S.setup, names,
  });
  life = new Life($('#life-left'), $('#life-right'), {
    command, branch, p: () => S.p, setup: () => S.setup, names, scene: () => scene, params: paramsAt,
    act: (i, p) => {
      // 選び方は、その年までの命令を重ねたもの(子を持つは一度きりなので、出したその年だけ)
      const a = {};
      for (const c of branch().commands) if (c.type === 'act' && c.region === i && c.t <= p) Object.assign(a, c.data, c.data.child && c.t < p ? { child: false } : {});
      return a;
    },
    iron: () => S.iron, setIron: on => { S.iron = on; document.body.classList.toggle('iron', on); },
    // 暮らす国を選んで始め直す: プレイヤーの世帯と選挙を、その国へ移した世界を新しく作る
    moveTo: async i => {
      const setup = IM.deepCopy(S.setup);
      setup.region_kw = setup.region_kw || {};
      const PL = { player: true, vote: true, vote_noise: true, rho_w: 0.01, rho_k: 0.02 };
      setup.regions.forEach((nm, k) => {
        const kw = Object.assign({}, setup.region_kw[nm] || {});
        for (const key of Object.keys(PL)) delete kw[key];
        if (k === i) Object.assign(kw, PL);
        setup.region_kw[nm] = kw;
      });
      await loadRecipe({ setup, commands: [], end: 1, p: 1, sel: i });
      setView('life');
    },
  });
  window.__life = life;
  document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
  // 設定
  $('#btn-settings').addEventListener('click', () => $('#settings').classList.toggle('open'));
  $('#btn-help').addEventListener('click', () => $('#helpbox').classList.toggle('open'));
  document.querySelectorAll('[data-q]').forEach(b => b.addEventListener('click', () => {
    S.quality = b.dataset.q;
    document.querySelectorAll('[data-q]').forEach(x => x.classList.toggle('on', x === b));
    if (scene) scene.setQuality(S.quality);
  }));
  $('#btn-reset').addEventListener('click', async () => {
    const setup = IM.defaultSetup();
    const v = $('#set-cur').value;
    if (v === 'single') {
      setup.regions = ['A']; setup.seeds = { A: 0 }; setup.currency = 'common'; setup.capital = false;
      setup.region_kw = { A: setup.region_kw.A };
    } else if (v === 'ABx_Cy') { setup.currency = { A: 'x', B: 'x', C: 'y' }; setup.regimes = { y: 'float' }; }
    else if (v === 'common') { setup.currency = 'common'; setup.regimes = null; }
    else if (v === 'float' || v === 'fixed') { setup.currency = v; setup.regimes = null; }
    if (v !== 'single') setup.capital = $('#set-cap').value === 'on';
    setup.Ns = 400;
    $('#settings').classList.remove('open');
    await loadRecipe({ setup, commands: [], end: 1, p: 1, sel: S.sel });
  });
  setupMenus();
  setupEvents();
  // 始めの世界(URL に元帳があれば、それを読む)
  let rc = null;
  const m = location.hash.match(/#r=([A-Za-z0-9_-]+)/);
  if (m) { try { rc = JSON.parse(b64d(m[1])); } catch (e) { rc = null; } }
  await loadRecipe(rc || { setup: IM.defaultSetup(), commands: [], end: 1, p: 1, sel: 0 });
  requestAnimationFrame(tick);
  $('#boot').classList.add('done');
  // 試し(Playwright)のための窓口
  window.__im = {
    state: () => ({ p: S.p, end: branch().end, cur: S.cur, ghost: S.ghost, branches: S.branches.size, playing: S.playing,
                    backend: backend.mode, books: branch().frames.slice(0, S.p).reduce((a, f) => Math.max(a, ...f.regions.map(r => r.books / r.books_tol)), 0) }),
    frames: () => branch().frames,
    seek, ensure: async u => { while (branch().end < u && branch().broken === null) await ensure(Math.min(u, branch().end + 25)); },
    command, select: selectCountry, play: on => setPlaying(on), speed: s => { S.speed = s; }, view: setView,
    analysis: () => analysis, life: () => life, scene: () => scene, sound: () => sound,
    verify: async () => (await send({ type: 'verify', branch: S.cur, frames: branch().frames })).result,
    recipe, loadRecipe,
  };
}
start();
