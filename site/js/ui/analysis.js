// analysis.js: 分析の画面。(1) 二つの試しの比べ、(2) 平行世界の扇の図(今の年から運だけを引き直した N 本の未来)、
// (3) 小さな政策の地図(二つのつまみの 5 × 5 の格子を、今の年から H 年回し、指標の平均を色で)。
// 平行世界と地図は、分析の Worker(最大 3 本。模型の Worker と合わせて 4 本まで)で回す。
import { t, fmt, pct } from '../i18n.js';
import { INDICATORS, LEVERS } from './panels.js';
import { ACCENT } from './scene.js';

const KEYS = ['E', 'real_wage', 'infl', 'top10_inc', 'gini_nw', 'b', 'tau', 'tau_k', 'breakups', 'pT', 'tb_Y', 'e_cur', 'hungry', 'Y'];
const MAP_LEVERS = LEVERS.filter(L => !L.type && !L.rel && !L.same && !L.only);

// 分析の仕事の Worker の溜まり(Worker が使えないときは、画面の中で一つずつ)
class Pool {
  constructor(n) {
    this.workers = [];
    this.queue = [];
    this.busy = new Map();
    this.nextId = 0;
    try {
      for (let k = 0; k < n; k++) {
        const w = new Worker('js/analysis_worker.js');
        w.onmessage = ev => this._done(w, ev.data);
        w.onerror = () => { this.workers = this.workers.filter(x => x !== w); };
        this.workers.push(w);
      }
    } catch (e) { this.workers = []; }
    this.free = this.workers.slice();
  }
  run(job) {
    return new Promise(res => {
      job.id = ++this.nextId;
      job.res = res;
      this.queue.push(job);
      this._pump();
    });
  }
  _pump() {
    if (!this.workers.length) {
      // 画面の中で(一つずつ、画面を止めないように少し待って)
      const job = this.queue.shift();
      if (!job) return;
      setTimeout(() => { job.res(window.IM.runJob(job.setup, job.state, job.cmds, job.years, job.keys)); this._pump(); }, 0);
      return;
    }
    while (this.free.length && this.queue.length) {
      const w = this.free.pop(), job = this.queue.shift();
      this.busy.set(w, job);
      w.postMessage({ id: job.id, setup: job.setup, state: job.state, cmds: job.cmds, years: job.years, keys: job.keys });
    }
  }
  _done(w, m) {
    const job = this.busy.get(w);
    this.busy.delete(w);
    this.free.push(w);
    if (job) job.res(m.error ? { error: m.error } : m.result);
    this._pump();
  }
}

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
// 系列(keys → 配列)から、指標の値の配列を作る
function indSeries(I, ser) {
  const n = ser.E.length, out = [];
  for (let y = 0; y < n; y++) {
    const r = {};
    for (const k of KEYS) r[k] = ser[k][y];
    out.push(I.get(r));
  }
  return out;
}

export class Analysis {
  constructor(root, ctx) {
    this.root = root;
    this.ctx = ctx;           // { send, branches(), cur(), p(), sel(), setup(), names() }
    const hw = (navigator.hardwareConcurrency || 4);
    this.pool = new Pool(Math.max(1, Math.min(3, hw - 1)));
    this.ec = window.echarts || null;
    this.charts = {};
    this.fan = null;
    this.map = null;
    this.tab = 'compare';
    root.querySelectorAll('[data-atab]').forEach(b => b.addEventListener('click', () => this.showTab(b.dataset.atab)));
    root.querySelector('#fan-run').addEventListener('click', () => this.runFan());
    root.querySelector('#map-run').addEventListener('click', () => this.runMap());
    for (const id of ['#cmp-a', '#cmp-b', '#cmp-ind', '#cmp-ctry']) root.querySelector(id).addEventListener('change', () => this.drawCompare());
    for (const id of ['#fan-ind', '#fan-ctry']) root.querySelector(id).addEventListener('change', () => this.drawFan());
    for (const id of ['#map-ind']) root.querySelector(id).addEventListener('change', () => this.drawMap());
    window.addEventListener('resize', () => Object.values(this.charts).forEach(c => c.resize()));
  }
  chart(id) {
    if (!this.ec) return null;
    if (!this.charts[id]) this.charts[id] = this.ec.init(this.root.querySelector('#' + id));
    return this.charts[id];
  }
  open() { this.fill(); this.showTab(this.tab); }
  showTab(k) {
    this.tab = k;
    this.root.querySelectorAll('[data-atab]').forEach(b => b.classList.toggle('on', b.dataset.atab === k));
    this.root.querySelectorAll('.apane').forEach(p => p.classList.toggle('on', p.id === 'apane-' + k));
    this.fill();
    if (k === 'compare') this.drawCompare();
    if (k === 'fan') this.drawFan();
    if (k === 'map') this.drawMap();
    setTimeout(() => Object.values(this.charts).forEach(c => c.resize()), 30);
  }
  // 選ぶものを埋める(言葉が替わったとき、試しが増えたときにも)
  fill() {
    const C = this.ctx;
    const tries = Array.from(C.branches().values());
    const opt = (sel, items, keep) => {
      const old = sel.value;
      sel.innerHTML = items.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
      if (keep && items.some(([v]) => String(v) === old)) sel.value = old;
    };
    const tr = tries.map(b => [b.id, t('branch_n', { n: b.n }) + (b.fork ? ` (${t('yearN', { y: b.fork })}〜)` : '')]);
    const A = this.root.querySelector('#cmp-a'), B = this.root.querySelector('#cmp-b');
    opt(A, tr, true); opt(B, tr, true);
    if (!A.dataset.touched) { const g = C.ghost(); A.value = String(g !== null && g !== undefined ? g : tries[0].id); B.value = String(C.cur()); }
    const inds = INDICATORS.filter(I => !(I.only === 'single' && C.setup().regions.length !== 1)).map(I => [I.key, t(I.label)]);
    for (const id of ['#cmp-ind', '#fan-ind', '#map-ind']) opt(this.root.querySelector(id), inds, true);
    const ct = C.names().map((nm, i) => [i, nm]);
    for (const id of ['#cmp-ctry', '#fan-ctry']) { const s = this.root.querySelector(id); opt(s, ct, true); if (!s.dataset.touched) s.value = String(C.sel()); }
    const lv = MAP_LEVERS.map(L => [L.key, t('p_' + L.key)]);
    opt(this.root.querySelector('#map-x'), lv, true);
    opt(this.root.querySelector('#map-y'), lv, true);
    const mx = this.root.querySelector('#map-x'), my = this.root.querySelector('#map-y');
    if (!mx.dataset.touched) { mx.value = 'r_cb'; my.value = 'wealth_tax'; mx.dataset.touched = my.dataset.touched = '1'; }
    A.onchange = B.onchange = () => { A.dataset.touched = '1'; this.drawCompare(); };
  }
  ind(id) { const k = this.root.querySelector(id).value; return INDICATORS.find(I => I.key === k) || INDICATORS[0]; }

  // ------------------------------------------------ 二つの試しの比べ
  drawCompare() {
    const C = this.ctx, ch = this.chart('cmp-chart');
    const a = C.branches().get(+this.root.querySelector('#cmp-a').value), b = C.branches().get(+this.root.querySelector('#cmp-b').value);
    const I = this.ind('#cmp-ind'), i = +this.root.querySelector('#cmp-ctry').value || 0;
    if (!a || !b) return;
    const sa = [], sb = [], d = [];
    const n = Math.max(a.end, b.end);
    for (let k = 0; k < n; k++) {
      const fa = a.frames[k], fb = b.frames[k];
      const va = fa ? I.get(fa.regions[i]) : null, vb = fb ? I.get(fb.regions[i]) : null;
      if (va !== null) sa.push([k + 1, va]);
      if (vb !== null) sb.push([k + 1, vb]);
      if (va !== null && vb !== null) d.push([k + 1, vb - va]);
    }
    // 分かれた年からあとの平均の差
    const fork = Math.max(a.fork || 0, b.fork || 0);
    const after = d.filter(([y]) => y > fork);
    const mean = v => v.reduce((x, y) => x + y, 0) / Math.max(v.length, 1);
    const ma = mean(sa.filter(([y]) => y > fork && y <= Math.min(a.end, b.end)).map(v => v[1]));
    const mb = mean(sb.filter(([y]) => y > fork && y <= Math.min(a.end, b.end)).map(v => v[1]));
    this.root.querySelector('#cmp-sum').innerHTML = t('cmp_sum', {
      y0: fork, y1: Math.min(a.end, b.end), a: t('branch_n', { n: a.n }), b: t('branch_n', { n: b.n }),
      va: I.f(ma), vb: I.f(mb), d: I.f(mb - ma), n: after.length,
    });
    if (!ch) return;
    const col = ACCENT[i];
    ch.setOption({
      backgroundColor: 'transparent', animation: false, textStyle: { color: '#c9d6ee', fontFamily: 'Inter, "Noto Sans JP", sans-serif' },
      tooltip: { trigger: 'axis', valueFormatter: v => I.f(v) },
      legend: { top: 0, textStyle: { color: '#c9d6ee' } },
      grid: [{ left: 64, right: 20, top: 34, height: '50%' }, { left: 64, right: 20, top: '70%', bottom: 30 }],
      xAxis: [{ type: 'value', min: 1, max: 'dataMax', gridIndex: 0, axisLabel: { show: false }, splitLine: { show: false } },
              { type: 'value', min: 1, max: 'dataMax', gridIndex: 1, name: t('year'), splitLine: { show: false } }],
      yAxis: [{ type: 'value', scale: true, gridIndex: 0, axisLabel: { formatter: v => I.f(v) }, splitLine: { lineStyle: { color: 'rgba(120,140,180,0.15)' } } },
              { type: 'value', scale: true, gridIndex: 1, name: t('cmp_diff'), axisLabel: { formatter: v => I.f(v) }, splitLine: { lineStyle: { color: 'rgba(120,140,180,0.15)' } } }],
      series: [
        { name: t('branch_n', { n: a.n }), type: 'line', showSymbol: false, data: sa, lineStyle: { width: 1.6, type: 'dashed', color: '#c8d3e6' }, itemStyle: { color: '#c8d3e6' } },
        { name: t('branch_n', { n: b.n }), type: 'line', showSymbol: false, data: sb, lineStyle: { width: 2, color: col }, itemStyle: { color: col },
          markLine: fork ? { symbol: 'none', silent: true, data: [{ xAxis: fork }], lineStyle: { color: '#fff', opacity: 0.4 }, label: { formatter: t('cmp_fork') } } : undefined },
        { name: t('cmp_diff'), type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: d, itemStyle: { color: p => (p.value[1] >= 0 ? '#34d399' : '#ff4d6d') } },
      ],
    }, true);
  }

  // ------------------------------------------------ 平行世界の扇の図
  async runFan(opts = {}) {
    const C = this.ctx;
    const N = opts.n || +this.root.querySelector('#fan-n').value, H = opts.h || +this.root.querySelector('#fan-h').value;
    const p = C.p(), bid = C.cur();
    const info = this.root.querySelector('#fan-info');
    info.textContent = t('computing');
    const st = await C.send({ type: 'state', branch: bid, year: p });
    const base = opts.seed !== undefined ? opts.seed : (Math.random() * 4294967296) >>> 0;
    const seeds = Array.from({ length: N }, (_, k) => window.IM.mixSeed(base, p, k, 99));
    const t0 = performance.now();
    let done = 0;
    const results = await Promise.all(seeds.map(seed => this.pool.run({ setup: C.setup(), state: st.state, cmds: [{ type: 'dice', data: { seed } }], years: H, keys: KEYS })
      .then(r => { done++; info.textContent = `${t('computing')} ${done} / ${N}`; return r; })));
    this.fan = { p, H, seeds, base, results, bid, ms: performance.now() - t0 };
    info.textContent = t('fan_info', { n: N, h: H, y: p, s: ((performance.now() - t0) / 1000).toFixed(1), w: this.pool.workers.length || 1 });
    this.drawFan();
    return this.fan;
  }
  drawFan() {
    const ch = this.chart('fan-chart');
    if (!ch) return;
    const C = this.ctx, I = this.ind('#fan-ind'), i = +this.root.querySelector('#fan-ctry').value || 0;
    const b = C.branches().get(C.cur());
    const hist = [];
    const p0 = this.fan ? this.fan.p : C.p();
    for (let k = Math.max(0, p0 - 60); k < p0; k++) if (b.frames[k]) hist.push([k + 1, I.get(b.frames[k].regions[i])]);
    const series = [{ name: t('fan_hist'), type: 'line', showSymbol: false, data: hist, lineStyle: { width: 2, color: '#e6eeff' }, itemStyle: { color: '#e6eeff' } }];
    if (this.fan) {
      const F = this.fan;
      const paths = F.results.filter(r => !r.error).map(r => indSeries(I, r.series[i]));
      const q = [[], [], [], [], []];
      for (let y = 0; y < F.H; y++) {
        const v = paths.map(s => s[y]).filter(Number.isFinite).sort((x, z) => x - z);
        [0.1, 0.25, 0.5, 0.75, 0.9].forEach((qq, k) => q[k].push([F.p + y + 1, quantile(v, qq)]));
      }
      const col = ACCENT[i];
      // 幅は、上の道と下の道を一つの多角形にして描く(数の軸どうしでは、積み上げが横の軸で積まれるので)
      const band = (lo, hi, name, op) => [{
        name, type: 'custom', silent: true, data: [[lo[0][0], lo[0][1]]], z: 1, itemStyle: { color: col, opacity: op + 0.2 },
        renderItem: (params, api) => {
          const pts = hi.map(v => api.coord(v)).concat(lo.slice().reverse().map(v => api.coord(v)));
          return { type: 'polygon', shape: { points: pts }, style: { fill: col, opacity: op } };
        },
        encode: { x: 0, y: 1 },
      }];
      series.push(...band(q[0], q[4], t('fan_80'), 0.18), ...band(q[1], q[3], t('fan_50'), 0.3));
      series.push({ name: t('fan_median'), type: 'line', showSymbol: false, data: q[2], lineStyle: { width: 2, color: col }, itemStyle: { color: col } });
      paths.slice(0, 8).forEach((s, k) => series.push({ name: 'w' + k, type: 'line', showSymbol: false, silent: true, tooltip: { show: false },
        data: s.map((v, y) => [F.p + y + 1, v]), lineStyle: { width: 0.7, color: col, opacity: 0.35 } }));
      // 今の試しが、その先も済んでいれば(実際に通った道)
      const real = [];
      for (let k = F.p; k < Math.min(b.end, F.p + F.H); k++) if (b.frames[k]) real.push([k + 1, I.get(b.frames[k].regions[i])]);
      if (real.length) series.push({ name: t('fan_real'), type: 'line', showSymbol: false, data: real, lineStyle: { width: 1.5, type: 'dashed', color: '#ffc94d' }, itemStyle: { color: '#ffc94d' } });
      const broken = F.results.filter(r => r.broken !== null && r.broken !== undefined).length;
      if (broken) this.root.querySelector('#fan-info').textContent += ' ' + t('fan_broken', { n: broken });
    }
    ch.setOption({
      backgroundColor: 'transparent', animation: false, textStyle: { color: '#c9d6ee', fontFamily: 'Inter, "Noto Sans JP", sans-serif' },
      tooltip: { trigger: 'axis', valueFormatter: v => I.f(v) },
      legend: { top: 0, textStyle: { color: '#c9d6ee' }, data: [t('fan_hist'), t('fan_median'), t('fan_50'), t('fan_80'), t('fan_real')] },
      grid: { left: 64, right: 20, top: 40, bottom: 34 },
      xAxis: { type: 'value', name: t('year'), min: 'dataMin', max: 'dataMax', splitLine: { show: false } },
      yAxis: { type: 'value', scale: true, axisLabel: { formatter: v => I.f(v) }, splitLine: { lineStyle: { color: 'rgba(120,140,180,0.15)' } } },
      series,
    }, true);
  }

  // ------------------------------------------------ 小さな政策の地図
  async runMap(opts = {}) {
    const C = this.ctx;
    const kx = opts.x || this.root.querySelector('#map-x').value, ky = opts.y || this.root.querySelector('#map-y').value;
    const H = opts.h || +this.root.querySelector('#map-h').value;
    const Lx = MAP_LEVERS.find(L => L.key === kx), Ly = MAP_LEVERS.find(L => L.key === ky);
    const i = C.sel(), p = C.p(), bid = C.cur();
    const info = this.root.querySelector('#map-info');
    info.textContent = t('computing');
    const st = await C.send({ type: 'state', branch: bid, year: p });
    const lv = L => Array.from({ length: 5 }, (_, k) => +(L.min + (L.max - L.min) * k / 4).toFixed(6));
    const xs = lv(Lx), ys = lv(Ly);
    const jobs = [];
    const t0 = performance.now();
    let done = 0;
    for (let a = 0; a < 5; a++) for (let c = 0; c < 5; c++) {
      const data = { [kx]: xs[a], [ky]: ys[c] };
      jobs.push(this.pool.run({ setup: C.setup(), state: st.state, cmds: [{ type: 'param', region: i, data }], years: H, keys: KEYS })
        .then(r => { done++; info.textContent = `${t('computing')} ${done} / 25`; return { a, c, r }; }));
    }
    const res = await Promise.all(jobs);
    this.map = { kx, ky, xs, ys, H, p, i, res, cur: st.state.R[i].p };
    info.textContent = t('map_info', { h: H, y: p, c: C.names()[i], s: ((performance.now() - t0) / 1000).toFixed(1) });
    this.drawMap();
    return this.map;
  }
  drawMap() {
    const ch = this.chart('map-chart');
    if (!ch || !this.map) return;
    const M = this.map, I = this.ind('#map-ind');
    const Lx = MAP_LEVERS.find(L => L.key === M.kx), Ly = MAP_LEVERS.find(L => L.key === M.ky);
    const data = [];
    let lo = Infinity, hi = -Infinity;
    for (const { a, c, r } of M.res) {
      let v = NaN;
      if (!r.error && r.broken === null) {
        const s = indSeries(I, r.series[M.i]);
        const tail = s.slice(-10).filter(Number.isFinite);
        v = tail.reduce((x, y) => x + y, 0) / Math.max(tail.length, 1);
      }
      if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      data.push([a, c, Number.isFinite(v) ? v : '-']);
    }
    // 今の値に一番近い格子
    const near = (vals, v) => vals.reduce((b, x, k) => (Math.abs(x - v) < Math.abs(vals[b] - v) ? k : b), 0);
    const cx = near(M.xs, M.cur[M.kx]), cy = near(M.ys, M.cur[M.ky]);
    ch.setOption({
      backgroundColor: 'transparent', animation: false, textStyle: { color: '#c9d6ee', fontFamily: 'Inter, "Noto Sans JP", sans-serif' },
      tooltip: { formatter: p => `${t('p_' + M.kx)} ${Lx.f(M.xs[p.value[0]])}<br>${t('p_' + M.ky)} ${Ly.f(M.ys[p.value[1]])}<br><b>${t(I.label)} ${p.value[2] === '-' ? t('map_broken') : I.f(p.value[2])}</b>` },
      grid: { left: 110, right: 90, top: 20, bottom: 60 },
      xAxis: { type: 'category', data: M.xs.map(v => Lx.f(v)), name: t('p_' + M.kx), nameLocation: 'middle', nameGap: 34, splitArea: { show: true } },
      yAxis: { type: 'category', data: M.ys.map(v => Ly.f(v)), name: t('p_' + M.ky), nameLocation: 'middle', nameGap: 80, splitArea: { show: true } },
      visualMap: { min: Number.isFinite(lo) ? lo : 0, max: Number.isFinite(hi) ? hi : 1, calculable: true, orient: 'vertical', right: 10, top: 'center',
                   textStyle: { color: '#c9d6ee' }, formatter: v => I.f(v), inRange: { color: ['#1d2b64', '#3d7bff', '#a06bff', '#ffc94d'] } },
      series: [{ type: 'heatmap', data, label: { show: true, color: '#fff', fontSize: 11, formatter: p => (p.value[2] === '-' ? '✕' : I.f(p.value[2])) },
                 itemStyle: { borderColor: 'rgba(5,9,20,0.6)', borderWidth: 2 },
                 markPoint: { symbol: 'rect', symbolSize: [10, 10], data: [{ coord: [cx, cy] }], itemStyle: { color: '#fff' }, label: { show: false } } }],
    }, true);
  }
}
