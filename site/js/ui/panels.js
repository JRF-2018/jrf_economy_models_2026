// panels.js: 政策のつまみ、指標のカードと時系列(ECharts)、新聞の見出し、大きなグラフ。
import { t, fmt, pct, getLang } from '../i18n.js';
import { ACCENT } from './scene.js';

const x = v => fmt(v, 2) + '×';
const p1 = v => pct(v, 1);
const p2 = v => pct(v, 2);
// 政策のつまみ(操作する国のパラメータ)。ch はノート 4 の章
export const LEVERS = [
  { g: 'g_fiscal', key: 'fiscal', type: 'select', opts: ['burden', 'deficit', 'debt', 'anchor', 'structural'], ch: 6 },
  { g: 'g_fiscal', key: 'burden_target', min: 0, max: 0.05, step: 0.002, f: p1, ch: 6, show: P => P.fiscal === 'burden' },
  { g: 'g_fiscal', key: 'benefit', min: 0, max: 2, step: 0.1, f: x, ch: 3 },
  { g: 'g_fiscal', key: 'g_tr', min: 0, max: 0.1, step: 0.005, f: p1, ch: 7 },
  { g: 'g_tax', key: 'tau_k', min: 0, max: 0.6, step: 0.01, f: p1, ch: 9, same: true },
  { g: 'g_tax', key: 'redistrib', min: 0, max: 1, step: 0.05, f: p1, ch: 9 },
  { g: 'g_tax', key: 'wealth_tax', min: 0, max: 0.05, step: 0.0025, f: p2, ch: 12 },
  { g: 'g_tax', key: 'enf_F', min: 0, max: 1, step: 0.05, f: p1, ch: 5 },
  { g: 'g_money', key: 'r_cb', min: 0, max: 0.12, step: 0.0025, f: p2, ch: 3 },
  { g: 'g_money', key: 'phi_pi', min: 0, max: 1.2, step: 0.05, f: v => fmt(v, 2), ch: 3 },
  { g: 'g_money', key: 'lam_T', min: 0, max: 0.95, step: 0.05, f: p1, ch: 4 },
  { g: 'g_money', key: 'mu_u', min: 0, max: 2, step: 0.1, f: v => fmt(v, 1), ch: 4 },
  { g: 'g_resolve', key: 'resolve', type: 'select', opts: ['none', 'levy', 'austerity', 'default', 'repression'], ch: 6 },
  { g: 'g_resolve', key: 'theta_resolve', min: 0.2, max: 3, step: 0.1, f: x, ch: 6 },
  { g: 'g_open', key: 'chi_F', min: 0, max: 0.3, step: 0.01, f: v => fmt(v, 2), ch: 5 },
  { g: 'g_open', key: 'theta_F0', min: 0, max: 0.5, step: 0.01, f: p1, ch: 5 },
  { g: 'g_safety', key: 'protect', min: 0, max: 0.05, step: 0.0025, f: p2, ch: 6 },
  { g: 'g_safety', key: 'relief', type: 'select', opts: ['none', 'state', 'mutual'], ch: 6 },
  { g: 'g_safety', key: 'learn', type: 'bool', ch: 9 },
  { g: 'g_safety', key: 'dis_rate', min: 0, max: 0.1, step: 0.005, f: p1, ch: 6 },
  // 段 B4: 部品
  { g: 'g_vote', key: 'vote', type: 'bool', ch: 9 },
  { g: 'g_vote', key: 'tau_hi', min: 0.2, max: 0.8, step: 0.05, f: p1, ch: 9, show: P => P.vote },
  { g: 'g_vote', key: 'tau_lo', min: 0, max: 0.3, step: 0.05, f: p1, ch: 9, show: P => P.vote },
  { g: 'g_age', key: 'pension', type: 'select', opts: ['none', 'payg', 'points', 'funded'], ch: 12 },
  { g: 'g_age', key: 'pen_rate', min: 0, max: 0.8, step: 0.05, f: p1, ch: 12, show: P => P.pension === 'payg' || P.pension === 'points' },
  { g: 'g_age', key: 'child_allow', min: 0, max: 1.5, step: 0.1, f: x, ch: 12 },
  { g: 'g_research', key: 'research', type: 'bool', ch: 10 },
  { g: 'g_research', key: 's_R', min: 0, max: 0.1, step: 0.005, f: p1, ch: 10, show: P => P.research },
  { g: 'g_research', key: 'psi_v', min: 0, max: 1, step: 0.05, f: v => fmt(v, 2), ch: 11 },
  // 段 B7: 遊びの出来事
  { g: 'g_events', key: 'revolt', type: 'bool', ch: 15 },
  { g: 'g_events', key: 'rev_thr', min: 0.4, max: 2.0, step: 0.1, f: v => fmt(v, 1), ch: 15, show: P => P.revolt },
  { g: 'g_events', key: 'up_sub', min: 0, max: 0.8, step: 0.05, f: p1, ch: 15 },
  { g: 'g_crisis', key: 'cap_N', min: 0, max: 1.5, step: 0.05, f: v => (v > 0 ? fmt(v, 2) + '×' : '—'), ch: 7, rel: 'PN' },   // 今の価格に対する倍
  { g: 'g_crisis', key: 'crisis_rule', type: 'select', opts: ['none', 'last', 'share'], ch: 7 },
  { g: 'g_crisis', key: 'inventory', type: 'bool', ch: 7 },
  { g: 'g_crisis', key: 'g_stock', min: 0, max: 0.3, step: 0.01, f: p1, ch: 7 },
  { g: 'g_crisis', key: 'g_pw', min: 0, max: 0.1, step: 0.005, f: p1, ch: 7 },
  { g: 'g_crisis', key: 'f_K', min: 0, max: 0.1, step: 0.01, f: v => fmt(v, 2), ch: 7 },
];

// 指標のカード
export const INDICATORS = [
  { key: 'E', label: 'ind_E', get: r => r.E, f: v => fmt(v, 0) },
  { key: 'real_wage', label: 'ind_real_wage', get: r => r.real_wage, f: v => fmt(v, 2) },
  { key: 'infl', label: 'ind_infl', get: r => r.infl, f: p1, signed: true },
  { key: 'top10', label: 'ind_top10', get: r => r.top10_inc, f: p1 },
  { key: 'gini_nw', label: 'ind_gini_nw', get: r => r.gini_nw, f: v => fmt(v, 3) },
  { key: 'b', label: 'ind_b', get: r => r.b, f: x },
  { key: 'tau_k', label: 'ind_tau_k', get: r => (r.tau_k < 0 ? r.tau : r.tau_k), f: p1 },
  { key: 'hungry', label: 'ind_hungry', get: r => r.hungry, f: p1 },
  { key: 'tau', label: 'ind_tau', get: r => r.tau, f: p1 },
  { key: 'breakups', label: 'ind_breakups', get: r => r.breakups, f: v => fmt(v, 1) },
  { key: 'pT', label: 'ind_pT', get: r => r.pT, f: v => fmt(v, 0) },
  { key: 'tb_Y', label: 'ind_tb_Y', get: r => r.tb_Y, f: p2, signed: true },
  { key: 'e', label: 'ind_e', get: r => r.e_cur, f: v => fmt(v, 3) },
];

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

export class Levers {
  constructor(root, onChange) {
    this.root = root;
    this.onChange = onChange;
    this.inputs = {};
  }
  build() {
    this.root.innerHTML = '';
    let grp = null, gname = null;
    for (const L of LEVERS) {
      if (L.g !== gname) {
        gname = L.g;
        grp = el('details', 'lever-group');
        grp.open = ['g_fiscal', 'g_tax', 'g_money'].includes(L.g);
        grp.appendChild(el('summary', '', `<span>${t(L.g)}</span>`));
        this.root.appendChild(grp);
      }
      const row = el('div', 'lever');
      row.dataset.key = L.key;
      const head = el('div', 'lever-head');
      head.appendChild(el('label', '', t('p_' + L.key)));
      const val = el('span', 'lever-val');
      head.appendChild(val);
      row.appendChild(head);
      let input;
      if (L.type === 'bool') {
        input = el('select', 'lever-select');
        for (const o of ['false', 'true']) {
          const op = el('option', '', t('p_bool_' + o));
          op.value = o;
          input.appendChild(op);
        }
        input.addEventListener('change', () => this.onChange(L.key, input.value === 'true'));
      } else if (L.type === 'select') {
        input = el('select', 'lever-select');
        for (const o of L.opts) {
          const op = el('option', '', t('p_' + L.key + '_' + o));
          op.value = o;
          input.appendChild(op);
        }
        input.addEventListener('change', () => this.onChange(L.key, input.value));
      } else {
        input = el('input', 'lever-range');
        input.type = 'range';
        input.min = L.min; input.max = L.max; input.step = L.step;
        input.addEventListener('input', () => { val.textContent = L.f(+input.value); });
        input.addEventListener('change', () => this.onChange(L.key, L.rel ? (+input.value) * (this.ctx[L.rel] || 1) : +input.value));
      }
      row.appendChild(input);
      if (L.same) {
        const lab = el('label', 'lever-same');
        const cb = el('input');
        cb.type = 'checkbox';
        lab.appendChild(cb);
        lab.appendChild(document.createTextNode(' ' + t('p_tau_k_same')));
        cb.addEventListener('change', () => this.onChange(L.key, cb.checked ? -1.0 : +input.value));
        row.appendChild(lab);
        this.inputs[L.key + '_same'] = cb;
      }
      const nb = el('div', 'lever-nb', `${t('nb_link')}: ${getLang() === 'ja' ? '第 ' + L.ch + ' 章' : 'Ch. ' + L.ch}`);
      row.appendChild(nb);
      grp.appendChild(row);
      this.inputs[L.key] = { input, val, L, row };
    }
  }
  // 今の年のパラメータの値を見せる。ctx.single: 一国だけの世界か(危機時の生産のつまみを出す)
  show(P, ctx = {}) {
    this.ctx = ctx;
    for (const k in this.inputs) {
      const o = this.inputs[k];
      if (!o || !o.L) continue;
      const v = P[k];
      if (o.L.type === 'bool') o.input.value = v ? 'true' : 'false';
      else if (o.L.type === 'select') o.input.value = v;
      else {
        const same = o.L.same && v < 0;
        const shown = o.L.rel ? v / (ctx[o.L.rel] || 1) : v;
        o.input.value = same ? 0 : shown;
        o.input.disabled = same;
        o.val.textContent = same ? t('p_tau_k_same') : o.L.f(shown);
        if (o.L.same) this.inputs[k + '_same'].checked = same;
      }
      o.row.style.display = (o.L.show && !o.L.show(P)) || (o.L.only === 'single' && !ctx.single) ? 'none' : '';
    }
    // 中のつまみがすべて隠れた組は、組ごと隠す
    this.root.querySelectorAll('.lever-group').forEach(g => {
      const vis = Array.from(g.querySelectorAll('.lever')).some(r => r.style.display !== 'none');
      g.style.display = vis ? '' : 'none';
    });
  }
}

// ---------------------------------------------------------------- 指標のカード
export class Cards {
  constructor(root, onOpen) {
    this.root = root;
    this.onOpen = onOpen;
    this.cards = [];
    this.ec = window.echarts || null;
  }
  build() {
    this.root.innerHTML = '';
    this.cards = [];
    for (const I of INDICATORS) {
      if (this.hide && this.hide(I)) continue;
      const c = el('div', 'card');
      c.innerHTML = `<div class="card-head"><span class="card-label">${t(I.label)}</span><span class="card-delta"></span></div>
        <div class="card-val"></div><div class="card-spark"></div>`;
      c.addEventListener('click', () => this.onOpen(I));
      this.root.appendChild(c);
      const spark = c.querySelector('.card-spark');
      let chart = null;
      if (this.ec) {
        chart = this.ec.init(spark, null, { renderer: 'canvas' });
        chart.setOption({
          animation: false, grid: { left: 0, right: 0, top: 2, bottom: 2 },
          xAxis: { type: 'value', show: false, min: 'dataMin', max: 'dataMax' }, yAxis: { type: 'value', show: false, scale: true },
          series: [{ type: 'line', showSymbol: false, data: [], lineStyle: { width: 1.6 }, areaStyle: { opacity: 0.12 } },
                   { type: 'line', showSymbol: false, data: [], lineStyle: { width: 1, type: 'dashed', opacity: 0.5 }, silent: true }],
        });
      }
      this.cards.push({ I, c, chart });
    }
    window.addEventListener('resize', () => this.cards.forEach(o => o.chart && o.chart.resize()));
  }
  // frames: 今の枝の年の列(0..p−1)。ghost: 前の枝の列。i: 操作する国
  update(frames, p, i, ghost) {
    const lo = Math.max(0, p - 120);
    const color = ACCENT[i];
    for (const o of this.cards) {
      const I = o.I;
      const cur = frames[p - 1], prev = frames[p - 2];
      const v = cur ? I.get(cur.regions[i]) : NaN;
      o.c.querySelector('.card-val').textContent = I.f(v);
      const d = cur && prev ? I.get(cur.regions[i]) - I.get(prev.regions[i]) : NaN;
      const dEl = o.c.querySelector('.card-delta');
      dEl.textContent = Number.isFinite(d) && d !== 0 ? (d > 0 ? '▲' : '▼') : '';
      dEl.className = 'card-delta ' + (d > 0 ? 'up' : d < 0 ? 'down' : '');
      if (o.chart) {
        const data = [];
        for (let k = lo; k < p; k++) if (frames[k]) data.push([k, I.get(frames[k].regions[i])]);
        const gd = [];
        if (ghost) for (let k = lo; k < Math.min(ghost.length, p + 30); k++) if (ghost[k]) gd.push([k, I.get(ghost[k].regions[i])]);
        o.chart.setOption({ series: [{ data, lineStyle: { color }, itemStyle: { color }, areaStyle: { color } },
                                     { data: gd, lineStyle: { color: '#c8d3e6' } }] });
      }
    }
  }
}

// ---------------------------------------------------------------- 大きなグラフ(三国と、前の試し)
export class BigChart {
  constructor(modal) {
    this.modal = modal;
    this.box = modal.querySelector('.modal-chart');
    this.title = modal.querySelector('.modal-title');
    this.chart = window.echarts ? window.echarts.init(this.box) : null;
    modal.querySelector('.modal-close').addEventListener('click', () => this.close());
    modal.addEventListener('click', e => { if (e.target === modal) this.close(); });
    window.addEventListener('resize', () => this.chart && this.chart.resize());
    this.I = null;
  }
  open(I, frames, p, ghost, names) {
    this.I = I;
    this.modal.classList.add('open');
    this.title.textContent = t(I.label);
    if (this.chart) { this.chart.resize(); this.update(frames, p, ghost, names); }
  }
  close() { this.modal.classList.remove('open'); this.I = null; }
  update(frames, p, ghost, names) {
    if (!this.I || !this.chart) return;
    const I = this.I;
    const series = [];
    for (let i = 0; i < names.length; i++) {
      const d = [];
      for (let k = 0; k < frames.length; k++) if (frames[k]) d.push([k, I.get(frames[k].regions[i])]);
      series.push({ name: names[i], type: 'line', showSymbol: false, data: d, lineStyle: { width: 2, color: ACCENT[i] }, itemStyle: { color: ACCENT[i] },
                    markLine: i === 0 ? { symbol: 'none', silent: true, data: [{ xAxis: p - 1 }], lineStyle: { color: '#fff', type: 'solid', opacity: 0.5 }, label: { show: false } } : undefined });
      if (ghost) {
        const g = [];
        for (let k = 0; k < ghost.length; k++) if (ghost[k]) g.push([k, I.get(ghost[k].regions[i])]);
        series.push({ name: names[i] + ' · ' + t('ghost_short'), type: 'line', showSymbol: false, data: g, lineStyle: { width: 1.2, type: 'dashed', color: ACCENT[i], opacity: 0.45 }, itemStyle: { color: ACCENT[i] } });
      }
    }
    this.chart.setOption({
      backgroundColor: 'transparent', animation: false,
      textStyle: { color: '#c9d6ee', fontFamily: 'Inter, "Noto Sans JP", sans-serif' },
      tooltip: { trigger: 'axis', valueFormatter: v => I.f(v) },
      legend: { top: 0, textStyle: { color: '#c9d6ee' } },
      grid: { left: 60, right: 24, top: 40, bottom: 40 },
      xAxis: { type: 'value', name: t('year'), min: 0, max: 'dataMax', axisLine: { lineStyle: { color: '#55627a' } }, splitLine: { show: false } },
      yAxis: { type: 'value', scale: true, axisLabel: { formatter: v => I.f(v) }, splitLine: { lineStyle: { color: 'rgba(120,140,180,0.15)' } } },
      series,
    }, true);
  }
}

// ---------------------------------------------------------------- 新聞の見出し
export function headlines(fr, prev, names) {
  const out = [];
  if (!fr) return out;
  fr.regions.forEach((r, i) => {
    const c = names[i];
    if (prev && r.vote_hi !== undefined && (r.vote_hi !== prev.regions[i].vote_hi || r.vote_lo !== prev.regions[i].vote_lo)) {
      const hi = r.vote_hi > r.vote_lo;
      out.push({ k: 'vote', i, s: t('h_vote', { c, tk: pct(hi ? r.tau_k : r.tau_k, 0), side: t(hi ? 'h_vote_hi' : 'h_vote_lo'),
                                                tw: pct(r.turnout_w, 0), tk2: pct(r.turnout_k, 0) }) });
    }
    if (prev && fr.pl && fr.pl[i] !== null && fr.pl[i] !== undefined && prev.regions[i].pl_gen !== undefined) {
      const q = prev.regions[i];
      if (r.pl_gen > q.pl_gen) out.push({ k: 'pl', i, s: t('h_pl_gen', { n: r.pl_gen }) });
      else if (r.pl_kids > q.pl_kids) out.push({ k: 'pl', i, s: t('h_pl_child') });
    }
    if (r.revolution > 0) out.push({ k: 'rev', i, s: t(r.regime ? 'h_revolution' : 'h_restoration', { c, tk: pct(r.tau_k < 0 ? r.tau : r.tau_k, 0) }) });
    if (prev && r.up_s !== undefined) {
      const q0 = prev.regions[i].up_s;
      if (r.up_s > 0 && !(q0 > 0)) out.push({ k: 'up', i, s: t('h_workshop_open', { c }) });
      if (q0 > 0 && !(r.up_s > 0)) out.push({ k: 'up', i, s: t('h_workshop_gone', { c }) });
      if (r.upgraded > 0) out.push({ k: 'up', i, s: t('h_upgraded', { c }) });
      else if (r.up_s >= 0.1 && q0 < 0.1) out.push({ k: 'up', i, s: t('h_workshop_grow', { c, p: pct(r.up_s, 0) }) });
    }
    if (prev) {
      const q = prev.regions[i];
      if (r.capped > 0 && !(q.capped > 0)) out.push({ k: 'cap', i, s: t('h_capped', { c }) });
      if (r.subsidy > 0 && !(q.subsidy > 0)) out.push({ k: 'last', i, s: t('h_last', { c }) });
      if (q.S_G > 0 && r.S_G < 0.8 * q.S_G && (r.short > 0 || r.capped > 0)) out.push({ k: 'stock', i, s: t('h_stock', { c, p: pct(1 - r.S_G / q.S_G, 0) }) });
    }
    if (r.short > 0.01) out.push({ k: 'short', i, s: t('h_short', { c, p: pct(r.short, 0) }) });
    if (r.hungry > 0.05 && !(prev && prev.regions[i].hungry > 0.05)) out.push({ k: 'hungry', i, s: t('h_hungry', { c, p: pct(r.hungry, 0) }) });
    if (r.dis_size > 0) out.push({ k: 'disaster', i, s: t('h_disaster', { c, p: pct(r.dis_size, 0) }) });
    if (r.defaulted > 0) out.push({ k: 'default', i, s: t('h_default', { c, p: pct(0.3, 0) }) });
    if (prev) {
      const q = prev.regions[i];
      if (r.crisis > 0 && !(q.crisis > 0)) out.push({ k: 'crisis', i, s: t('h_crisis', { c, b: fmt(r.b, 2) }) });
      if (r.breakups >= 5 && r.breakups > 1.5 * (q.breakups || 0)) out.push({ k: 'breakups', i, s: t('h_breakups', { c, n: fmt(r.breakups, 0) }) });
      const de = Math.log(r.e_cur / q.e_cur);
      if (de < -0.08) out.push({ k: 'fx', i, s: t('h_fx_down', { c, p: pct(1 - Math.exp(de), 0) }) });
      if (de > 0.08) out.push({ k: 'fx', i, s: t('h_fx_up', { c, p: pct(Math.exp(de) - 1, 0) }) });
      if (r.E < 0.92 * q.E) out.push({ k: 'unemp', i, s: t('h_unemp', { c, p: pct(1 - r.E / q.E, 0) }) });
      const g = Math.log(r.pT / q.pT);
      if (g > 0.3) out.push({ k: 'land', i, s: t('h_land_boom', { c, p: pct(Math.exp(g) - 1, 0) }) });
      if (g < -0.3) out.push({ k: 'land', i, s: t('h_land_bust', { c, p: pct(1 - Math.exp(g), 0) }) });
    }
  });
  return out;
}
