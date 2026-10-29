// life.js: 一人の暮らしの画面(段 B6)。プレイヤーの世帯(標本の中の一つの記入、人数 0.01)として、
// 暮らしの支出・何を削るか・働く度合い・子・投票と呼びかけを選び、代替わりしながら暮らす(ワールドネバーランドの代替わり)。
// 右には、世帯の様子、正味資産の年表(代の境に印)、家系図、選挙(群れの相関と、決め手になったか)。
import { t, fmt, pct } from '../i18n.js';
import { ACCENT } from './scene.js';

const CHOICES = [
  { key: 'spend', min: 0.4, max: 1.6, step: 0.05, def: 1.0, f: v => fmt(v, 2) + '×' },
  { key: 'lux', min: 0, max: 1, step: 0.05, def: 0.7, f: v => pct(v, 0) },
  { key: 'work', min: 0, max: 1, step: 0.05, def: 1.0, f: v => pct(v, 0) },
  { key: 'campaign', min: 0, max: 1, step: 0.05, def: 0.0, f: v => pct(v, 0) },
];

export class Life {
  constructor(left, right, ctx) {
    this.left = left;
    this.right = right;
    this.ctx = ctx;         // { command, branch(), p(), setup(), names(), scene(), act(i, p) }
    this.ec = window.echarts || null;
    this.chart = null;
    this.build();
  }
  plRegion() {
    const b = this.ctx.branch();
    const f = b && b.frames[Math.max(this.ctx.p() - 1, 0)];
    if (!f || !f.pl) return -1;
    return f.pl.findIndex(v => v !== null && v !== undefined);
  }
  build() {
    const L = this.left;
    L.innerHTML = `<div class="panel-title"><span>${t('life_choices')}</span></div>
      <div class="life-who"></div>
      ${CHOICES.map(c => `<div class="lever" data-act="${c.key}"><div class="lever-head"><label>${t('act_' + c.key)}</label><span class="lever-val"></span></div>
        <input type="range" min="${c.min}" max="${c.max}" step="${c.step}"><div class="lever-nb">${t('act_' + c.key + '_help')}</div></div>`).join('')}
      <div class="life-mix"><div class="mix-bar"><span class="mx-n"></span><span class="mx-l"></span></div><div class="mix-leg"></div></div>
      <div class="lever"><div class="lever-head"><label>${t('act_vote')}</label></div>
        <select class="lever-select" data-act="vote"><option value="">${t('act_vote_none')}</option><option value="hi">${t('act_vote_hi')}</option><option value="lo">${t('act_vote_lo')}</option></select>
        <div class="lever-nb">${t('act_vote_help')}</div></div>
      <button class="primary life-child">${t('act_child')}</button>
      <div class="lever-nb life-child-note"></div>
      <label class="life-iron"><input type="checkbox"> ${t('life_ironman')}</label>
      <div class="lever life-move"><div class="lever-head"><label>${t('life_move')}</label></div>
        <select class="lever-select"></select><button class="life-move-btn">${t('life_move_btn')}</button>
        <div class="lever-nb">${t('life_move_help')}</div></div>`;
    L.querySelectorAll('[data-act]').forEach(row => {
      const key = row.dataset.act;
      const inp = row.tagName === 'SELECT' ? row : row.querySelector('input');
      if (!inp) return;
      const c = CHOICES.find(x => x.key === key);
      if (c) inp.addEventListener('input', () => { row.querySelector('.lever-val').textContent = c.f(+inp.value); });
      inp.addEventListener('change', () => {
        const v = inp.tagName === 'SELECT' ? (inp.value || null) : +inp.value;
        const shown = c ? c.f(v) : (v ? t('act_vote_' + v) : t('act_vote_none'));
        this.ctx.command([{ type: 'act', region: this.plRegion(), data: { [key]: v } }], `${t('life_you')}: ${t('act_' + key)} → ${shown}`);
      });
    });
    L.querySelector('.life-child').addEventListener('click', () => {
      this.ctx.command([{ type: 'act', region: this.plRegion(), data: { child: true } }], `${t('life_you')}: ${t('act_child')}`);
    });
    L.querySelector('.life-iron input').addEventListener('change', e => this.ctx.setIron(e.target.checked));
    L.querySelector('.life-move-btn').addEventListener('click', () => {
      const i = +L.querySelector('.life-move select').value;
      if (window.confirm(t('life_move_confirm', { c: this.ctx.names()[i] }))) this.ctx.moveTo(i);
    });
    const R = this.right;
    R.innerHTML = `<div class="life-status"></div>
      <div class="life-sub">${t('life_wealth')}</div><div class="life-chart"></div>
      <div class="life-sub">${t('life_tree')}</div><div class="life-tree"></div>
      <div class="life-sub">${t('life_election')}</div><div class="life-vote"></div>`;
    this.chart = null;
  }
  open() {
    const i = this.plRegion();
    const b = this.ctx.branch(), f = b.frames[this.ctx.p() - 1];
    const sc = this.ctx.scene();
    if (sc && i >= 0 && f) sc.focusHouse(i, f.pl[i]);
    this.update();
  }
  update() {
    const C = this.ctx, b = C.branch(), p = C.p(), i = this.plRegion();
    if (!b || i < 0) {
      this.right.querySelector('.life-status').innerHTML = `<p>${t('life_noplayer')}</p>`;
      return;
    }
    const f = b.frames[p - 1], r = f.regions[i];
    const act = C.act(i, p);
    // 選び方の今の値
    this.left.querySelector('.life-who').innerHTML = t('life_who', { c: C.names()[i] });
    for (const c of CHOICES) {
      const row = this.left.querySelector(`[data-act=${c.key}]`);
      const v = act[c.key] === undefined || act[c.key] === null ? c.def : act[c.key];
      row.querySelector('input').value = v;
      row.querySelector('.lever-val').textContent = c.f(v);
    }
    this.left.querySelector('select[data-act=vote]').value = act.vote || '';
    const lux = act.lux === undefined || act.lux === null ? 0.7 : act.lux;
    this.left.querySelector('.mx-n').style.width = pct(1 - lux, 0);
    this.left.querySelector('.mx-l').style.width = pct(lux, 0);
    this.left.querySelector('.mix-leg').textContent = t('life_mix', { n: pct(1 - lux, 0), l: pct(lux, 0) });
    const kids = r.pl_kids || 0, P = C.params(i, p);
    const btn = this.left.querySelector('.life-child');
    btn.disabled = kids > 0 || !(P.child_cost > 0) || !!act.child;
    this.left.querySelector('.life-child-note').textContent = kids > 0 ? t('life_kids', { y: fmt(kids, 0) }) : (act.child ? t('life_child_next') : t('act_child_help'));
    this.left.querySelector('.life-iron input').checked = !!C.iron();
    const ms = this.left.querySelector('.life-move select');
    const mh = C.names().map((nm, k) => `<option value="${k}">${nm}</option>`).join('');
    if (ms.dataset.h !== mh) { ms.innerHTML = mh; ms.dataset.h = mh; }
    if (document.activeElement !== ms) ms.value = String(i);
    // 世帯の様子
    const rent = r.pl_rentier;
    const items = [
      [t('life_gen'), t('life_gen_n', { n: fmt(r.pl_gen, 0) })], [t('life_age'), t('life_years', { y: fmt(r.pl_age, 0) })],
      [t('life_class'), rent ? t('life_rentier') : t('life_worker')], [t('life_nw'), fmt(r.pl_nw_rel, 2) + '× ' + t('life_avg')],
      [t('life_inc'), fmt(r.pl_Yd, 1)], [t('life_dep'), fmt(r.pl_M, 1)], [t('life_asset'), pct(r.pl_asset, 0)],
      [t('life_pivots'), fmt(r.pl_pivots, 0)],
    ];
    this.right.querySelector('.life-status').innerHTML = `<div class="life-head" style="--accent:${ACCENT[i]}">${t('life_you')} · ${C.names()[i]}${r.pl_hungry > 0 ? ` <span class="life-hungry">${t('life_hungry')}</span>` : ''}</div>`
      + `<div class="life-grid">${items.map(([k, v]) => `<div><small>${k}</small><b>${v}</b></div>`).join('')}</div>`;
    // 家系図: 代ごとの年と、終わりの財産
    const gens = [];
    let cur = null;
    for (let k = 0; k < p; k++) {
      const q = b.frames[k] && b.frames[k].regions[i];
      if (!q) continue;
      if (!cur || q.pl_gen !== cur.gen) {
        if (cur) gens.push(cur);
        cur = { gen: q.pl_gen, y0: k + 1, y1: k + 1, peak: q.pl_nw_rel, end: q.pl_nw_rel, rent: q.pl_rentier, kids: 0, piv0: q.pl_pivots, piv: 0, hungry: 0 };
      }
      cur.y1 = k + 1; cur.end = q.pl_nw_rel; cur.peak = Math.max(cur.peak, q.pl_nw_rel); cur.rent = q.pl_rentier;
      cur.piv = q.pl_pivots - cur.piv0; cur.hungry += q.pl_hungry > 0 ? 1 : 0;
      const pk = k > 0 && b.frames[k - 1] ? b.frames[k - 1].regions[i].pl_kids : 0;
      if (q.pl_kids > pk) cur.kids += 1;                // 子を育てる年が増えた年 = 子を持った年
    }
    if (cur) gens.push(cur);
    this.right.querySelector('.life-tree').innerHTML = gens.slice(-6).map((g, k, a) => `<div class="gen ${k === a.length - 1 ? 'now' : ''}" style="--accent:${ACCENT[i]}">
      <div class="gen-n">${t('life_gen_n', { n: g.gen })}</div><div class="gen-y">${t('yearN', { y: g.y0 })}〜${g.y1}</div>
      <div class="gen-w">${fmt(g.end, 2)}×</div><div class="gen-c">${g.rent ? t('life_rentier') : t('life_worker')}${g.kids ? ' · ' + t('life_children', { n: g.kids }) : ''}${g.piv ? ' · ★' + g.piv : ''}${g.hungry ? ' · ' + t('life_hungry_y', { n: g.hungry }) : ''}</div></div>`).join('<div class="gen-link"></div>');
    // 正味資産の年表(代の境に印)
    const box = this.right.querySelector('.life-chart');
    if (this.ec) {
      if (!this.chart || this.chart.getDom() !== box) this.chart = this.ec.init(box);
      const d = [];
      for (let k = Math.max(0, p - 150); k < p; k++) { const q = b.frames[k] && b.frames[k].regions[i]; if (q) d.push([k + 1, Math.max(q.pl_nw_rel, 1e-3)]); }
      this.chart.setOption({
        backgroundColor: 'transparent', animation: false, textStyle: { color: '#c9d6ee', fontFamily: 'Inter, "Noto Sans JP", sans-serif' },
        grid: { left: 44, right: 10, top: 10, bottom: 22 }, tooltip: { trigger: 'axis', valueFormatter: v => fmt(v, 2) + '×' },
        xAxis: { type: 'value', min: 'dataMin', max: 'dataMax', splitLine: { show: false } },
        yAxis: { type: 'log', axisLabel: { formatter: v => fmt(v, v < 1 ? 2 : 0) + '×' }, splitLine: { lineStyle: { color: 'rgba(120,140,180,0.15)' } } },
        series: [{ type: 'line', showSymbol: false, data: d, lineStyle: { width: 2, color: '#ffe08a' }, areaStyle: { color: '#ffe08a', opacity: 0.08 },
                   markLine: { symbol: 'none', silent: true, lineStyle: { color: ACCENT[i], opacity: 0.6 }, label: { show: false },
                               data: gens.slice(1).map(g => ({ xAxis: g.y0 })).concat([{ yAxis: 1 }]) } }],
      }, true);
    }
    // 選挙
    const V = this.right.querySelector('.life-vote');
    let ev = null;
    for (let k = p - 1; k >= 1; k--) {
      const q = b.frames[k] && b.frames[k].regions[i], q0 = b.frames[k - 1] && b.frames[k - 1].regions[i];
      if (q && q0 && q.vote_hi !== undefined && (q.vote_hi !== q0.vote_hi || q.vote_lo !== q0.vote_lo)) { ev = { y: k + 1, q }; break; }
    }
    if (!P.vote) { V.innerHTML = `<p class="lever-nb">${t('life_novote')}</p>`; return; }
    if (!ev) { V.innerHTML = `<p class="lever-nb">${t('life_vote_wait', { n: P.vote_every })}</p>`; return; }
    const q = ev.q, tot = Math.max(q.vote_hi + q.vote_lo, 1e-9);
    const margin = Math.abs(q.vote_hi - q.vote_lo);
    const sway = (P.k_sway || 1e-3) * (act.campaign || 0);
    const g = act.vote === 'hi' ? 'w' : act.vote === 'lo' ? 'k' : (r.pl_rentier ? 'k' : 'w');   // 呼びかけが効く群れ(im1.py の hold_vote と同じ)
    V.innerHTML = `<div class="vote-bar"><span class="vb-hi" style="width:${100 * q.vote_hi / tot}%">${t('h_vote_hi_short')} ${fmt(q.vote_hi, 1)}</span>`
      + `<span class="vb-lo" style="width:${100 * q.vote_lo / tot}%">${t('h_vote_lo_short')} ${fmt(q.vote_lo, 1)}</span></div>`
      + `<div class="vote-info">${t('life_vote_info', { y: ev.y, tw: pct(q.turnout_w, 0), tk: pct(q.turnout_k, 0), tau: pct(q.tau_k < 0 ? q.tau : q.tau_k, 0) })}</div>`
      + `<div class="vote-info">${t('life_vote_you', { v: q.pl_vote ? t('act_vote_' + q.pl_vote) : t('act_vote_none'), piv: q.pl_pivot > 0 ? t('life_pivot_yes') : t('life_pivot_no'),
                                                    m: fmt(margin, 2), w: fmt(0.01, 2) })}</div>`
      + `<div class="vote-info">${t('life_rho', { rw: fmt(P.rho_w + (g === 'w' ? sway : 0), 4), rk: fmt(P.rho_k + (g === 'k' ? sway : 0), 4), s: fmt(sway, 4) })}</div>`;
  }
}
