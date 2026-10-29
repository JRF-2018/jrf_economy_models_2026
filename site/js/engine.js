// engine.js: 時間のつまみの心臓。試しの元帳(レシピ)、鍵の年、枝、計算し直し。DOM を使わない(Worker と quickjs で動く)。
// 決定論の約束: 年 t の状態 = 年 t のはじめ(命令を当てる前)。命令は年 t のはじめに当て、その年を計算する。
// 鍵の年(KEY 年ごと)に模型の状態の全部を持ち、戻った年から枝を始めるときは、手前の鍵の年から計算し直す。
(function (G) {
  'use strict';
  const IM = G.IM || (G.IM = {});
  const KEY = 10;
  const VERSION = 'im-sandbox-1';

  // 既定の世界: 三つの国が共通の通貨を使い、家計は正味資産の 3% を外の資産に置く(資本が動く)。子を育てる費用(段 9)があり、
  // A 国にはプレイヤーの世帯(段 19)が住み、資本所得の税を選挙で決める(段 5。三国とも選挙にすると、税 0.5 と全額の再分配で一家離散が増えすぎた)。
  // B 国は少し生産性が低い。C 国は贅沢品が得意で原料が苦手、安全網が薄く、土地を好む。
  // 始めの移り変わり(warmup 年)を黙って回し、そのあとを 0 年とする
  function defaultSetup() {
    return {
      version: VERSION,
      regions: ['A', 'B', 'C'], Ns: 400, seeds: { A: 0, B: 1, C: 2 }, warmup: 60,
      currency: 'common', regimes: null, capital: true,
      common: { g_gamma: 0.0, credit: true, land: true, c_h: 0.005, disaster: true, breakup_eps: 1e-9, theta_F0: 0.03, child_cost: 0.3, age_track: true },
      region_kw: { A: { player: true, vote: true, vote_noise: true, rho_w: 0.01, rho_k: 0.02 },
                   B: { gamma0: [0.12, 0.33, 0.3] }, C: { gamma0: [0.1, 0.26, 0.31], benefit: 0.6, theta0: 0.2 } },
    };
  }
  // 一国だけの世界(危機時の生産は im3 の世界になく、一国の模型 Im1 でだけ動くので)。World と同じ形の窓口を持つ
  class Single {
    constructor(setup) {
      const r = setup.regions[0];
      const kw = Object.assign({}, setup.common, (setup.region_kw || {})[r] || {});
      this.R = [new IM.Im1({ households: 'continuous', Ns: setup.Ns, seed: (setup.seeds || {})[r] || 0, region: '', kw })];
      this.names = [r];
      this.e = Float64Array.from([1.0]); this.cur = [0]; this.dest = [0]; this.KF = Float64Array.from([0.0]);
      this.lastFlows = null;
      this.t = 0;
      this.log = [];
    }
    get ledgers() { return [this.R[0].L]; }
    step() {
      const rec = this.R[0].step();
      Object.assign(rec, { exports: 0, imports: 0, tb: 0, tb_Y: 0, ib: 0, fxpos: 0, interbank: 0, e_cur: 1.0 });
      this.t += 1;
      return [rec];
    }
    getState() { return { t: this.t, R: [this.R[0].getState(true)] }; }
    setState(s) { this.t = s.t; this.R[0].setState(s.R[0]); }
  }
  function newWorld(setup) { return setup.regions.length === 1 ? new Single(setup) : new IM.World(worldOpts(setup)); }

  // 設定から世界を作る(始めの移り変わりを回し、時計を 0 に戻す。時計は災害の予定と 'anchor' のルールにだけ効く)
  function makeWorld(setup) {
    const w = newWorld(setup);
    for (let k = 0; k < (setup.warmup || 0); k++) w.step();
    w.t = 0;
    w.log = [];
    for (const e of w.R) { e.t = 0; e.log = []; }
    for (const L of w.ledgers) L.t = 0;
    return w;
  }
  function worldOpts(setup) {
    if (setup.regions.length === 1) return setup;
    return { regions: setup.regions, Ns: setup.Ns, seeds: setup.seeds, currency: setup.currency, regimes: setup.regimes,
             capital: setup.capital, union: setup.union || null, common: setup.common, region_kw: setup.region_kw,
             households: 'continuous' };
  }
  // 種の混ぜ合わせ(サイコロ: 年 t から先の乱数を、種 s と年と地域から作る)
  function mixSeed(s, t, i, k) {
    let h = (s >>> 0) ^ Math.imul(t + 1, 0x9E3779B1) ^ Math.imul(i + 1, 0x85EBCA77) ^ Math.imul(k + 1, 0xC2B2AE3D);
    h = Math.imul(h ^ (h >>> 16), 0x7FEB352D) >>> 0;
    h = Math.imul(h ^ (h >>> 15), 0x846CA68B) >>> 0;
    return (h ^ (h >>> 16)) >>> 0;
  }

  // 一年の記録のうち、画面に出すもの
  const REC_KEYS = ['t', 'E', 'unemp', 'real_wage', 'W', 'PN', 'PL', 'PI', 'bN', 'infl', 'Y', 'tau', 'b', 'G', 'deficit', 'r_policy',
                    'r_loan', 'r_bond', 'top10_inc', 'gini_inc', 'gini_nw', 'top10_nw', 'pT', 'land_value', 'breakups', 'wo_h',
                    'hh_debt', 'n_borrowers', 'uN', 'uL', 'uI', 'KN', 'KL', 'KI', 'xN', 'xL', 'xI', 'dis_size', 'relief', 'K_lost',
                    'prot', 'crisis', 'defaulted', 'levy', 'exports', 'imports', 'tb', 'tb_Y', 'ib', 'fxpos', 'interbank', 'e_cur',
                    'F', 'F_Y', 'KF', 'KF_Y', 'union_in', 'books', 'books_tol', 'theta', 'tau_k', 'wage_share', 'loans', 'div',
                    // 段 B4: 危機時の生産、学習、投票、研究、文化、年金
                    'hungry', 'short', 'capped', 'S_G', 'InvN', 'subsidy', 'gov_buy', 'hungry_kids', 'hungry_ret', 'ret_share', 'pensions',
                    'child_allow', 'belief', 'protect', 'vote_hi', 'vote_lo', 'turnout_w', 'turnout_k', 'n_rentier', 'n_var', 'gain',
                    'gamma_front', 'sR', 'L_R', 'redistrib', 'acct_use', 'acct_lux', 'acct_save', 'acct_explore', 'transfer', 'tax_c', 'deficit',
                    // 段 B6: プレイヤーの世帯
                    'pl_M', 'pl_Yd', 'pl_nw', 'pl_nw_rel', 'pl_age', 'pl_kids', 'pl_asset', 'pl_gen', 'pl_pivots', 'pl_rentier',
                    'pl_hungry', 'pl_vote', 'pl_campaign', 'pl_pivot', 'pl_rentier_vote',
                    // 段 B7: 格上げの工房と革命
                    'up_s', 'up_gX', 'up_adv', 'upgraded', 'up_pay', 'rev_G', 'revolution', 'regime'];

  function makeFrame(w, recs) {
    const n = w.R.length;
    const regions = recs.map(r => {
      const o = {};
      for (const k of REC_KEYS) if (k in r) o[k] = r[k];
      return o;
    });
    const houses = w.R.map(e => {
      const H = e.H, m = H.M.length, P = e.p;
      const h = new Float32Array(m), c = new Float32Array(m), r = new Float32Array(m);
      const v_eq = e.v_eq_last;
      let tot = 0.0, hs = 0.0;
      const nw = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        nw[i] = H.M[i] + e.pT * H.land[i] + v_eq * H.eq[i] + H.F[i];
        tot += Math.max(nw[i], 0); hs += H.heads[i];
      }
      const mean = Math.max(tot / hs, 1e-12);
      for (let i = 0; i < m; i++) {
        h[i] = nw[i] / H.heads[i] / mean;           // 一人あたりの正味資産(平均との比)
        c[i] = H.asset_share[i];                     // 所得のうち資産所得の割合
        r[i] = e.risk ? e.risk[i] : 0.0;             // 借金 / 借入の上限
      }
      // 局地の災害(その年に災害があったとき): 打たれた世帯
      let hit = null;
      if (e.dis_rec && e.dis_rec.dis_size > 0 && e.last_hit) { hit = new Uint8Array(m); for (let i = 0; i < m; i++) hit[i] = e.last_hit[i] ? 1 : 0; }
      return { h, c, r, hit };
    });
    // 財の流れ [買い手 s][産地 r](基準の通貨で)と、資本の流れ
    const flows = [];
    const f = w.lastFlows;
    for (let s = 0; s < n; s++) {
      const row = [];
      for (let r = 0; r < n; r++) row.push(f ? (f[s][r][0] + f[s][r][1] + f[s][r][2]) * w.e[w.cur[s]] : 0.0);
      flows.push(row);
    }
    const pl = w.R.map(e => e.pl);          // プレイヤーの記入(なければ null)
    return { t: recs[0].t, regions, houses, flows, pl, KF: Array.from(w.KF).map((v, o) => v * w.e[w.cur[o]]), dest: w.dest.slice(),
             e: Array.from(w.e), cur: w.cur.slice() };
  }

  class Engine {
    constructor(post) {
      this.post = post || (() => {});
      this.branches = new Map();
      this.nextId = 0;
      this.world = null;
      this.live = null;            // 今の世界が乗っている枝の id
    }
    reset(setup) {
      this.setup = setup || defaultSetup();
      this.world = makeWorld(this.setup);
      this.branches.clear();
      this.nextId = 0;
      const b = this._newBranch(null, 0, []);
      b.keys.set(0, this.world.getState());
      this.live = b.id;
      return b.id;
    }
    _newBranch(parent, fork, commands) {
      const id = this.nextId++;
      const b = { id, parent, fork, commands, keys: new Map(), end: fork, broken: null };
      this.branches.set(id, b);
      return b;
    }
    applyCommands(w, cmds, t) {
      for (const c of cmds) {
        if (c.t !== t) continue;
        const i = c.region === undefined || c.region === null ? -1 : c.region;
        const targets = i < 0 ? w.R.map((_, k) => k) : [i];
        if (c.type === 'param') for (const k of targets) Object.assign(w.R[k].p, c.data);
        else if (c.type === 'act') for (const k of targets) Object.assign(w.R[k].act, c.data);   // プレイヤーの選び方(段 19)
        else if (c.type === 'disaster') for (const k of targets) w.R[k].shock_queue.push([t, c.data.size, c.data.scope || 'common']);
        else if (c.type === 'capital_loss') {
          // 能力の半減: 一国の生産能力の一部が失われる(災害の怪我なしの形)
          for (const k of targets) {
            const e = w.R[k];
            for (let j = 0; j < 3; j++) { const lost = c.data.share * e.K[j]; e.K[j] -= lost; e.L.revalue(e.real[j], e.firms[j], -lost); }
          }
        } else if (c.type === 'dice') {
          w.R.forEach((e, k) => { IM.RNGS.forEach((name, q) => { e[name] = new IM.PortableRNG(mixSeed(c.data.seed, t, k, q + 1)); }); });
        }
      }
    }
    _step(b, emit) {
      const w = this.world, t = w.t;
      this.applyCommands(w, b.commands, t);
      let recs;
      try {
        recs = w.step();
        for (const r of recs) for (const k of ['E', 'Y', 'PN', 'W']) if (!Number.isFinite(r[k])) throw new Error('数でない: ' + k);
      } catch (err) {
        b.broken = t;
        if (emit) this.post({ type: 'broken', branch: b.id, t, msg: String(err && err.message || err) });
        return null;
      }
      if ((t + 1) % KEY === 0 && !b.keys.has(t + 1)) b.keys.set(t + 1, w.getState());
      if (t + 1 > b.end) b.end = t + 1;
      const fr = makeFrame(w, recs);
      if (emit) this.post({ type: 'frame', branch: b.id, frame: fr });
      return fr;
    }
    // 枝 b の年 year のはじめの状態に、今の世界を置く(手前の鍵の年から計算し直す)
    seekWorld(bid, year) {
      const b = this.branches.get(bid);
      if (this.live === bid && this.world.t === year) return;
      let k = -1;
      for (const t of b.keys.keys()) if (t <= year && t > k) k = t;
      this.world.setState(b.keys.get(k));
      this.live = bid;
      while (this.world.t < year) if (!this._step(b, false)) break;
    }
    // 枝 bid を until 年まで計算する(済んでいない年だけ。frame を送る)
    compute(bid, until) {
      const b = this.branches.get(bid);
      if (b.broken !== null) return b.end;
      this.seekWorld(bid, Math.min(b.end, until));
      while (this.world.t < until && b.broken === null) this._step(b, true);
      return b.end;
    }
    // 命令: 年 year に cmds を当てる。year が枝の済んだ終わりなら、その枝に足す。手前なら、新しい枝を分ける
    // truncate: 年 year から先の済んだ年が、まだ画面に出ていない(先に計算しただけ)なら、枝を分けずに捨てて足す
    command(bid, year, cmds, truncate = false) {
      const b = this.branches.get(bid);
      if (truncate && year < b.end) {
        b.end = year;
        for (const t of Array.from(b.keys.keys())) if (t > year) b.keys.delete(t);
        if (b.broken !== null && b.broken >= year) b.broken = null;
        if (this.live === bid && this.world.t > year) this.live = null;
      }
      cmds = cmds.map(c => Object.assign({}, c, { t: year + (c.dt || 0) }));   // dt: 先の年に置く命令(熱狂の終わりなど)
      if (year >= b.end && b.broken === null) {
        // 終わりで足すとき、先の年の命令は年の順に並べる(当てるのは年ごとなので、順は同じ年の中でだけ効く)
        b.commands = b.commands.concat(cmds);
        return bid;
      }
      this.seekWorld(bid, year);
      const nb = this._newBranch(bid, year, b.commands.filter(c => c.t < year).concat(cmds));
      for (const [t, s] of b.keys) if (t <= year) nb.keys.set(t, s);
      if (!nb.keys.has(year) && year % KEY === 0) nb.keys.set(year, this.world.getState());
      this.live = nb.id;
      return nb.id;
    }
    // 取り消し: 命令(id で指す)を元帳から除き、その年から先の済んだ年を捨てる(計算し直す)
    uncommand(bid, ids, year) {
      const b = this.branches.get(bid);
      const drop = new Set(ids);
      b.commands = b.commands.filter(c => !drop.has(c.id));
      if (year < b.end) {
        b.end = year;
        for (const t of Array.from(b.keys.keys())) if (t > year) b.keys.delete(t);
        if (this.live === bid && this.world.t > year) this.live = null;
      }
      if (b.broken !== null && b.broken >= year) b.broken = null;
      return b;
    }
    dropBranch(bid) { this.branches.delete(bid); if (this.live === bid) this.live = null; }
    recipe(bid) {
      const b = this.branches.get(bid);
      return { version: VERSION, setup: this.setup, commands: b.commands, end: b.end };
    }
    // ロード: 元帳から枝を作り直す(初めから計算する)
    load(recipe) {
      const id = this.reset(recipe.setup);
      const b = this.branches.get(id);
      b.commands = recipe.commands.slice();
      this.compute(id, recipe.end || 0);
      return id;
    }
    // 関門: 枝の数を、初めから通しで計算した数と比べる(完全に同じかどうか)
    verify(bid, frames) {
      const b = this.branches.get(bid);
      const w = makeWorld(this.setup);
      let maxd = 0, n = 0;
      for (let t = 0; t < b.end; t++) {
        this.applyCommands(w, b.commands, t);
        const recs = w.step();
        const fr = makeFrame(w, recs);
        const g = frames[t];
        fr.regions.forEach((r, i) => {
          for (const k in r) {
            const a = r[k], c = g.regions[i][k];
            if (a !== c && !(Number.isNaN(a) && Number.isNaN(c))) { maxd = Math.max(maxd, Math.abs(a - c)); n++; }
          }
        });
        fr.houses.forEach((h, i) => { for (const q of ['h', 'c', 'r']) for (let j = 0; j < h[q].length; j++) if (h[q][j] !== g.houses[i][q][j] && !(Number.isNaN(h[q][j]) && Number.isNaN(g.houses[i][q][j]))) { maxd = Math.max(maxd, Math.abs(h[q][j] - g.houses[i][q][j])); n++; } });
      }
      // 今の世界は動かしていない(別の世界で確かめた)
      return { years: b.end, mismatches: n, maxAbsDiff: maxd };
    }
  }

  // 画面からの命令を受ける(Worker でも、Worker が使えないときの画面の中の道でも、同じ形)
  function handle(E, m, post) {
    try {
      if (m.type === 'reset') {
        const id = E.reset(m.setup);
        post({ type: 'reset', req: m.req, branch: id, setup: E.setup, commands: [] });
      } else if (m.type === 'load') {
        const id = E.reset(m.recipe.setup);
        E.branches.get(id).commands = m.recipe.commands.slice();
        post({ type: 'reset', req: m.req, branch: id, setup: E.setup, commands: m.recipe.commands.slice(), loading: m.recipe.end || 0 });
      } else if (m.type === 'compute') {
        const end = E.compute(m.branch, m.until);
        const b = E.branches.get(m.branch);
        post({ type: 'computed', req: m.req, branch: m.branch, end, broken: b.broken });
      } else if (m.type === 'command') {
        const id = E.command(m.branch, m.year, m.cmds, !!m.truncate);
        const b = E.branches.get(id);
        post({ type: 'commanded', req: m.req, branch: id, parent: m.branch, year: m.year, commands: b.commands, end: b.end });
      } else if (m.type === 'uncommand') {
        const b = E.uncommand(m.branch, m.ids, m.year);
        post({ type: 'uncommanded', req: m.req, branch: b.id, commands: b.commands, end: b.end });
      } else if (m.type === 'drop') {
        E.dropBranch(m.branch);
        post({ type: 'dropped', req: m.req });
      } else if (m.type === 'state') {
        // 分析の画面: 枝 m.branch の年 m.year のはじめの状態(平行世界・政策の地図の元)
        E.seekWorld(m.branch, m.year);
        post({ type: 'state', req: m.req, state: E.world.getState(), t: m.year });
      } else if (m.type === 'recipe') {
        post({ type: 'recipe', req: m.req, recipe: E.recipe(m.branch) });
      } else if (m.type === 'verify') {
        post({ type: 'verified', req: m.req, result: E.verify(m.branch, m.frames) });
      }
    } catch (err) {
      post({ type: 'error', req: m.req, msg: String(err && err.stack || err) });
    }
  }

    // 分析の仕事: 状態から始めて、命令を当て、years 年回し、keys の系列を返す(analysis_worker.js が使う)
  function runJob(setup, state, cmds, years, keys) {
    const w = newWorld(setup);
    w.setState(IM.deepCopy(state));
    const E = new Engine();
    const t0 = w.t;
    const out = w.R.map(() => { const o = {}; for (const k of keys) o[k] = []; return o; });
    let broken = null;
    for (let y = 0; y < years; y++) {
      E.applyCommands(w, cmds.map(c => Object.assign({}, c, { t: t0 })), w.t);
      let recs;
      try { recs = w.step(); } catch (err) { broken = t0 + y; break; }
      if (recs.some(r => !Number.isFinite(r.E) || !Number.isFinite(r.Y))) { broken = t0 + y; break; }
      recs.forEach((r, i) => { for (const k of keys) out[i][k].push(r[k] === undefined ? null : r[k]); });
    }
    return { series: out, broken, t0 };
  }

  Object.assign(IM, { Engine, handle, makeWorld, Single, newWorld, runJob, defaultSetup, worldOpts, mixSeed, makeFrame, KEY, REC_KEYS, ENGINE_VERSION: VERSION });
})(typeof globalThis !== 'undefined' ? globalThis : this);
