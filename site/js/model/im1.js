// im1.js: 統合模型 im の一国の模型(im1.py の写し。rng='portable'、solver='portable' の道)。
// 写すのは、ブラウザで使う部品: 価格と生産、信用、標本の家計、土地と知財(知財の市場はなし)、財政のルール、金利のルール、
// 資本所得の税と再分配、富裕税と行き詰まりの道、災害(守りと救済)、外の資産(im3 から)、
// 段 B4: 危機時の生産(在庫、価格の上限、最後の生産者・損失の分担、備蓄、公共事業、固定費、廃棄)、学習('neutral')、
// 投票(票の揺れ)、研究(企業の集まりなし)、文化(種類の数と波及)、年齢と年金(賦課・貢献比例・積立)、養育の費用と児童手当、
// 家計ごとの運と腕、段 B6: プレイヤーの記入(段 19)と「何を削るか」。
// 写していない部品を入れると、投げる(checkSupported)。名前と順は im1.py と同じにする。
(function (G) {
  'use strict';
  const IM = G.IM || (G.IM = {});

  const SECT = ['N', 'L', 'I'];
  const FIRMS = ['fN', 'fL', 'fI'];
  const SECTORS = ['hh'].concat(FIRMS, ['bank', 'gov']);
  const INSTRUMENTS = ['deposits', 'loans', 'bonds', 'foreign'];
  const REAL = ['capital_N', 'capital_L', 'capital_I', 'stock_N', 'stock_G'];

  // numpy の logaddexp(0, y) と同じ形
  function logaddexp0(y) {
    if (y === 0) return Math.LN2;
    return y > 0 ? y + Math.log1p(Math.exp(-y)) : Math.log1p(Math.exp(y));
  }
  function softplus(x, k = 1.0) { return logaddexp0(k * x) / k; }
  function softmin(a, b, k = 50.0) { return a - softplus(a - b, k / Math.max(Math.abs(b), 1e-9)); }

  const zeros = n => new Float64Array(n);
  const sum = a => { let s = 0.0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
  const dot = (a, b) => { let s = 0.0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const F64 = a => Float64Array.from(a);
  const at = (v, i) => (typeof v === 'number' ? v : v[i]);      // 数か、記入ごとの列か

  function defaultParams() {
    // im1.py の default_params() のうち、ここで使うもの(値は同じ)。写していない部品のスイッチも、既定の値で持つ
    return {
      pop_w: 1000.0, n_cap: 5.0, q0: 1.0, gamma0: [0.1, 0.3, 0.3], alpha: [0.3, 0.5, 0.3], kappa: [1.0, 1.0, 1.0],
      delta: 0.1, g_gamma: 0.01,
      m0: [0.3, 0.3, 0.3], zeta: 0.5, u_star: 0.85, omega: 1.0, eta: 0.05, phi_r: 0.5, phi_u: 0.3, phi_w: 0.1, chi: 0.2,
      c_y: 0.9, c_m: 0.2, c_yc: 0.5, c_mc: 0.05, b_N: 0.3, beta_s: 3.0, w_res: 2.0, tactic: 1e-4,
      rng: 'portable', solver: 'portable',
      r_bar: 0.06, tau0: 0.2, deficit_target: 0.0, k_tau: 0.2, fiscal: 'burden', anchor_start: 100, b_target: 0.3, k_debt: 0.2,
      benefit: 1.0,
      credit: false, r_cb: 0.02, phi_pi: 0.0, pi_target: 0.0, spread: 0.02, r_eq: 0.02, f_R: 0.3, lam: 0.6, amort: 0.2,
      mu_prem: 0.0, theta_prem: 0.3, phi_fixed: -1.0, tau_k: -1.0, redistrib: 0.0,
      land: false, T_land: 1000.0, rho_T: 0.1, phi_R: 0.1, sigma_R: 0.1, eta_IP: 1.0, delta_IP: 0.1, theta0: 0.15, theta1: 3.0,
      chi_T: 0.2, h_y: 10.0, nu_T: 0.2, lam_T: 0.6, mu_u: 0.5, c_h: 0.0, breakup_eps: 0.0, haircut_max: 0.3, turnover: 0.025, inherit: 'lottery',
      theta_shock: 0.0,
      ip_market: false, lam_IP: 0.0,
      disaster: false, dis_rate: 0.0, dis_q: 0.0, dis_min: 0.05, dis_alpha: 1.5, dis_max: 0.8, dis_scope: 'common', hit_share: 0.1,
      inj_coef: 0.5, rec_inj: 0.3, protect: 0.0, prot_scale: 200.0, prot_decay: 0.05, relief: 'none', relief_rate: 0.5,
      mu_B: 0.0, theta_B: 1.0, burden_target: 0.016, resolve: 'none', theta_resolve: 1.0, levy_rate: 0.02, wealth_tax: 0.0,
      enf_F: 1.0, theta_F0: 0.0, phi_K: 0.0, phi_L: 0.0, chi_F: 0.05, theta_F_max: 0.9, levy_exempt: 3.0, austerity: 0.05,
      haircut: 0.3, rbar_link: 1.0,
      learn: false, learn_omega: 1.0, prior_a: 0.5, prior_b: 10.0, k_learn: 0.2, protect_max: 0.05, learn_attitude: 'neutral',
      vote: false, vote_every: 10, tau_lo: 0.1, tau_hi: 0.5, rho_w: 1e-3, rho_k: 1e-3, c_vote: 0.01, vote_noise: false, rentier_cut: 0.5,
      research: false, s_R: 0.02, trial_cost: 1.0, p_hit: 0.05, g0: 0.002, alpha_R: 1.5, g_cap: 0.3, cap_R: 0.5, adaptive_R: false,
      omega_R: 0.9, h_R: 5.0, s_R_min: 0.0, s_R_max: 0.1,
      hoard: false, attention: 1.0, auto_share: 0.0,
      age_track: false, ret_age: 0, pension: 'none', pen_rate: 0.4, fund_rate: 0.1, fund_draw: 0.1, fund_eq: 0.0, phi_q: 0.0, child_allow: 0.0, vat_bi: false,
      luck_sigma: 0.0, luck_nu: 0.0, luck_rho: 0.0, luck_skill: 0.0, eq_init: 'pareto', child_cost: 0.0, child_years: 20,
      psi_v: 0.0, culture_spill: 0.0, spill_cap: 3.0, ip_ref: 400.0, sigma_v: 3.0, eps_v: 1.0,
      crisis_track: false, inventory: false, inv_target: 0.1, chi_inv: 0.5, zeta_inv: 0.3, cap_N: 0.0, crisis_rule: 'none',
      g_stock: 0.0, chi_S: 0.2, delta_S: 0.05, g_pw: 0.0, g_tr: 0.0, f_K: 0.0, delta_scrap: 0.0,
      firms: 0, lending: 'none', k_mem: 0.0, s_pub: 0.0, eq_market: false,
      player: false, player_src: 0, player_heads: 0.01, player_heir: 'self', k_sway: 1e-3,
      is_split: 0,
      // 段 B7: 遊びの出来事(革命、格上げの工房)
      revolt: false, rev_wE: 3.0, rev_ww: 3.0, rev_wh: 2.0, rev_wb: 0.02, rev_mem: 0.8, rev_thr: 1.0, rev_cool: 15,
      rev_tau_k: 0.5, rev_redistrib: 1.0, rev_wealth_tax: 0.01,
      up_gen: 0, up_s0: 0.02, up_g0: 0.6, up_floor: 0.1, up_wright: 0.3, up_cum0: 50.0, up_speed: 0.5, up_sub: 0.0,
    };
  }

  // 写していない部品が入っていないか
  function checkSupported(P) {
    const bad = [];
    for (const k of ['hoard', 'vat_bi', 'ip_market', 'eq_market']) if (P[k]) bad.push(k);
    for (const k of ['firms', 'auto_share', 'k_mem', 'phi_q', 'is_split', 'dis_q', 'fund_eq']) if (P[k]) bad.push(k);
    if (P.attention !== 1.0) bad.push('attention');
    if (P.learn && P.learn_attitude !== 'neutral') bad.push('learn_attitude');
    if (['none', 'payg', 'points', 'funded'].indexOf(P.pension) < 0) bad.push('pension');
    if (['none', 'last', 'share'].indexOf(P.crisis_rule) < 0) bad.push('crisis_rule');
    if (P.lending !== 'none') bad.push('lending');
    if (P.rng !== 'portable' || P.solver !== 'portable') bad.push('rng/solver');
    if (bad.length) throw new Error('im1.js に写していない部品: ' + bad.join(', '));
  }

  // ------------------------------------------------ 標本の家計
  class Households {
    constructor(o) {
      this.M = F64(o.M); this.Yd = F64(o.Yd); this.heads = F64(o.heads); this.lab = F64(o.lab); this.eq = F64(o.eq);
      this.c_y = F64(o.c_y); this.c_m = F64(o.c_m);
      this.propensity = o.propensity || 'fixed';
      this.cw = o.cw || [0.9, 0.2]; this.cc = o.cc || [0.5, 0.05];
      const n = this.M.length;
      this.asset_share = zeros(n);
      if (this.propensity === 'fixed') for (let i = 0; i < n; i++) this.asset_share[i] = this.eq[i] > 0 ? 1.0 : 0.0;
      for (const k of ['land', 'ip', 'inj', 'extra', 'F', 'kids', 'skill', 'age', 'pf', 'pts', 'hd']) this[k] = zeros(n);
    }
    budgets() {
      const n = this.M.length, B = zeros(n);
      if (this.propensity === 'continuous') {
        for (let i = 0; i < n; i++) {
          const a = clip(this.asset_share[i], 0.0, 1.0);
          this.c_y[i] = this.cw[0] + (this.cc[0] - this.cw[0]) * a;
          this.c_m[i] = this.cw[1] + (this.cc[1] - this.cw[1]) * a;
        }
      }
      for (let i = 0; i < n; i++) {
        const b = this.c_y[i] * Math.max(this.Yd[i], 0.0) + this.c_m[i] * Math.max(this.M[i] + this.F[i], 0.0) + this.extra[i];
        B[i] = Math.max(b, 0.0);
      }
      return B;
    }
  }
  const H_FIELDS = ['M', 'Yd', 'heads', 'lab', 'eq', 'c_y', 'c_m', 'asset_share', 'land', 'ip', 'inj', 'extra', 'F', 'kids', 'skill',
                    'age', 'pf', 'pts', 'hd'];
  const EXT = ['M', 'Yd', 'heads', 'lab', 'eq', 'land', 'ip', 'extra', 'F', 'pf', 'pts', 'hd'];   // 記入の合計の量(足せる)
  const INT = ['c_y', 'c_m', 'asset_share', 'inj', 'kids', 'skill', 'age'];                      // 記入の中の割合

  function makeHouseholds(P, kind = 'two', Ns = 500, seed = 0, eqTail = 1.5) {
    if (kind === 'two') {
      return new Households({ M: [3000.0, 1000.0], Yd: [4500.0, 500.0], heads: [P.pop_w, P.n_cap], lab: [1.0, 0.0], eq: [0.0, 1.0],
                              c_y: [P.c_y, P.c_yc], c_m: [P.c_m, P.c_mc] });
    }
    const rng = new IM.PortableRNG(seed);
    const heads = new Float64Array(Ns).fill((P.pop_w + P.n_cap) / Ns);
    let eq = rng.pareto(eqTail, Ns);
    for (let i = 0; i < Ns; i++) eq[i] += 1e-3;
    if (P.eq_init === 'equal') eq = F64(heads);
    let s = sum(eq);
    for (let i = 0; i < Ns; i++) eq[i] /= s;
    const lab = zeros(Ns);
    for (let i = 0; i < Ns; i++) lab[i] = heads[i] * Math.exp(-30.0 * eq[i]);
    s = sum(lab);
    for (let i = 0; i < Ns; i++) lab[i] /= s;
    const M = zeros(Ns), Yd = zeros(Ns);
    for (let i = 0; i < Ns; i++) { M[i] = 3000.0 * lab[i] + 1000.0 * eq[i]; Yd[i] = 4500.0 * lab[i] + 500.0 * eq[i]; }
    const H = new Households({ M, Yd, heads, lab, eq, c_y: new Float64Array(Ns).fill(P.c_y), c_m: new Float64Array(Ns).fill(P.c_m),
                               propensity: 'continuous', cw: [P.c_y, P.c_m], cc: [P.c_yc, P.c_mc] });
    for (let i = 0; i < Ns; i++) H.asset_share[i] = 500.0 * eq[i] / Yd[i];
    return H;
  }

  // 重みつきの分布の量(im1.py の gini_w、top_share_w)
  function order(x) {
    const o = Array.from(x.keys());
    o.sort((a, b) => x[a] - x[b]);
    return o;
  }
  function giniW(x, w) {
    const o = order(x);
    let cw = 0.0, cx = 0.0, prev = 0.0, acc = 0.0;
    for (const i of o) {
      cw += w[i];
      cx += x[i] * w[i];
      acc += w[i] * (cx + prev);
      prev = cx;
    }
    if (cx <= 0) return 0.0;
    return 1 - acc / (cw * cx);
  }
  function topShareW(x, w, q = 0.1) {
    const o = order(x).reverse();
    let tot = 0.0;
    for (const i of o) tot += w[i];
    let cw = 0.0, num = 0.0, den = 0.0;
    for (const i of o) {
      cw += w[i];
      const take = clip((q * tot - (cw - w[i])) / w[i], 0.0, 1.0);
      num += x[i] * w[i] * take;
    }
    for (let i = 0; i < x.length; i++) den += x[i] * w[i];
    return num / Math.max(den, 1e-12);
  }

  // ------------------------------------------------ 一国の模型
  class Im1 {
    constructor(opts = {}) {
      const P = Object.assign(defaultParams(), opts.p || {}, opts.kw || {});
      this.p = P;
      checkSupported(P);
      const households = opts.households || 'two', Ns = opts.Ns || 500, seed = opts.seed || 0;
      const region = opts.region || '';
      const extra = opts.extraSectors || [];
      const sfx = region ? '@' + region : '';
      this.sfx = sfx;
      this.region = region;
      this.sec = {};
      for (const k of SECTORS.concat(extra)) this.sec[k] = k + sfx;
      this.firms = FIRMS.map(f => f + sfx);
      this.real = REAL.map(r => r + sfx);
      const standalone = !opts.ledger;
      this.L = standalone ? new IM.Ledger(Object.values(this.sec), INSTRUMENTS, this.real) : opts.ledger;
      this.K = F64([1250.0, 400.0, 1100.0]);
      this.gamma = F64(P.gamma0);
      this.gamma_new = F64(P.gamma0);
      this.m0 = F64(P.m0);
      this.W = 10.0;
      this.prices = F64([3.0, 8.0, 6.0]);
      this.PN_prev = this.prices[0];
      this.tau = P.tau0;
      this.loans = zeros(3);
      this.top10_prev = 0.0;
      this.infl_prev = 0.0;
      this.b_last = 0.0;
      this.r_policy = null;
      this.r_bond = null;
      this.premB = 0.0;
      this.setRates();
      this.rho_prev = new Float64Array(3).fill(this.rbar);
      this.u_prev = new Float64Array(3).fill(P.u_star);
      this.profit_prev = 0.0;
      this.H = makeHouseholds(P, households, Ns, seed);
      const L = this.L, hh = this.sec.hh, bank = this.sec.bank, gov = this.sec.gov;
      L.revalue('deposits', hh, 4000.0); L.revalue('deposits', bank, -4000.0);
      L.revalue('bonds', bank, 4000.0); L.revalue('bonds', gov, -4000.0);
      for (let j = 0; j < 3; j++) L.revalue(this.real[j], this.firms[j], this.K[j]);
      if (P.credit) {
        const L0 = zeros(3);
        for (let j = 0; j < 3; j++) L0[j] = (1 - P.f_R) * P.delta / P.amort * this.prices[2] * P.kappa[j] * this.K[j];
        for (let j = 0; j < 3; j++) { L.revalue('loans', bank, L0[j]); L.revalue('loans', this.firms[j], -L0[j]); }
        const L0s = sum(L0);
        L.revalue('deposits', hh, L0s); L.revalue('deposits', bank, -L0s);
        const H = this.H, n = H.M.length, share = zeros(n);
        for (let i = 0; i < n; i++) share[i] = 0.5 * H.lab[i] + 0.5 * H.eq[i];
        const ss = sum(share);
        for (let i = 0; i < n; i++) H.M[i] = H.M[i] + L0s * share[i] / ss;
        this.loans = F64(L0);
      }
      if (standalone) { L.close(); L.t = 0; }
      this.rng = new IM.PortableRNG(seed + 1);
      this.pT = 1.0; this.gT_e = 0.0; this.wo_h = 0.0;
      this.qIP = null;
      this.prot = 0.0; this.Y_last = 0.0; this.shock_queue = []; this.dis_rec = {};
      this.v_eq_last = 0.0;
      if (P.ret_age > 0 || P.age_track) this.H.age = new IM.PortableRNG(seed + 11).exponential(1.0 / Math.max(P.turnover, 1e-3), this.H.M.length);
      this.kap_new = F64(P.kappa);
      this.kap_avg = F64(P.kappa);
      this.res_rng = new IM.PortableRNG(seed + 3);
      this.luck_rng = new IM.PortableRNG(seed + 7);
      this.vote_rng = new IM.PortableRNG(seed + 13);
      this.xf = {};
      this.prof_R_last = null;
      this.Y_prev = null;
      this.InvN = null; this.S_G = 0.0; this.N_last = null; this._cx = null;
      this.sR = new Float64Array(3).fill(P.s_R);
      this.g_hat = null;
      this.events_this_year = 0;
      if (P.vote) this.vote_rec = { vote_hi: 0.0, vote_lo: 0.0, turnout_w: 0.0, turnout_k: 0.0, n_rentier: 0.0 };
      this.dis_rng = new IM.PortableRNG(seed + 2);
      if (P.land) {
        const H = this.H, n = H.M.length, share = zeros(n), hs = sum(H.heads);
        for (let i = 0; i < n; i++) share[i] = 0.5 * H.eq[i] + 0.5 * H.heads[i] / hs;
        const ss = sum(share);
        for (let i = 0; i < n; i++) H.land[i] = P.T_land * share[i] / ss;
        this.pT = Math.max(P.rho_T * this.prices[0] * 1200.0 / (P.T_land * 0.08), 1.0);
      }
      this.log = [];
      this.t = 0;
      this.pl = null;
      this.act = {};
      this.cf = 1.0;
      this.lab_inc_last = null;
      this.b_ref = 0.0;
      this.in_default = false;
      this.n_eff = 0.0; this.t_eff = 0.0;
      this.vote_pl = {};
      this.up_s = 0.0; this.up_gX = P.up_g0; this.up_cum = 0.0; this.up_gen_seen = 0; this.up_peak = 0.0;   // 段 B7: 格上げの工房
      this.rev_G = 0.0; this.regime = 0; this.rev_last = -1e6; this.rev_prev = null; this.rev_alt = null;   // 段 B7: 革命
      if (P.player) this.carvePlayer();
    }

    // ------------------------------------------------ 個人として生きる箱庭(段 19)
    carvePlayer() {
      const P = this.p, H = this.H;
      const i = P.player_src;
      const f = Math.min(P.player_heads / H.heads[i], 0.5);
      for (const name of EXT) {
        const arr = H[name], n = arr.length, out = new Float64Array(n + 1);
        out.set(arr);
        out[n] = arr[i] * f;
        out[i] = arr[i] * (1 - f);
        H[name] = out;
      }
      for (const name of INT) {
        const arr = H[name], n = arr.length, out = new Float64Array(n + 1);
        out.set(arr);
        out[n] = arr[i];
        H[name] = out;
      }
      this.pl = H.M.length - 1;
      this.pl_gen = 1; this.pl_pivots = 0;
    }
    playerState() {
      const H = this.H, k = this.pl;
      if (k === null) return {};
      const v_eq = this.v_eq_last;
      const nw = H.M[k] + this.pT * H.land[k] + (this.qIP || 0.0) * H.ip[k] + v_eq * H.eq[k] + H.F[k] + H.pf[k];
      const hd = H.heads[k];
      let tot = 0.0;
      for (let i = 0; i < H.M.length; i++) tot += H.M[i] + this.pT * H.land[i] + v_eq * H.eq[i] + H.F[i] + H.pf[i];
      const mean_nw = tot / sum(H.heads);
      return { pl_M: H.M[k] / hd, pl_Yd: H.Yd[k] / hd, pl_nw: nw / hd, pl_nw_rel: nw / hd / Math.max(mean_nw, 1e-12),
               pl_age: H.age[k], pl_kids: H.kids[k], pl_asset: H.asset_share[k], pl_gen: this.pl_gen, pl_pivots: this.pl_pivots,
               pl_rentier: H.asset_share[k] > this.p.rentier_cut };
    }

    pfDeposits() { return sum(this.H.pf); }

    setRates() {
      const P = this.p;
      if (!P.credit) { this.prem = 0.0; this.r_loan = 0.0; this.rbar = P.r_bar; return; }
      const phi = P.phi_fixed < 0 ? this.top10_prev : P.phi_fixed;
      this.prem = P.mu_prem * softplus(phi - P.theta_prem, 50.0);
      const infl = this.infl_prev;
      this.r_policy = Math.max(0.0, P.r_cb + P.phi_pi * (infl - P.pi_target));
      this.r_loan = this.r_policy + P.spread + this.prem;
      const r_ref = P.r_cb + P.spread;
      this.rbar = P.r_eq + P.rbar_link * this.r_loan + (1 - P.rbar_link) * r_ref;
      const b = this.b_last;
      this.premB = P.mu_B * softplus(b - P.theta_B, 50.0);
      if (P.resolve === 'repression' && b > P.theta_resolve) this.premB = 0.0;
      this.r_bond = this.r_policy + this.premB;
    }

    // ------------------------------------------------ 年の中(速い層)
    planInvestment() {
      const P = this.p, H = this.H;
      const inv = zeros(3);
      for (let j = 0; j < 3; j++) {
        const g = P.delta + P.phi_r * (this.rho_prev[j] - this.rbar) + P.phi_u * (this.u_prev[j] - P.u_star);
        inv[j] = this.K[j] * softplus(g, 50.0);
      }
      const cost = this.prices[2] * dot(this.kap_new, inv);
      let M_c = 0.0;
      if (H.propensity === 'continuous') { for (let i = 0; i < H.M.length; i++) M_c += H.asset_share[i] * Math.max(H.M[i], 0.0); }
      else M_c = dot(H.eq, H.M);
      let avail = Math.max(this.profit_prev, 0.0) + Math.max(M_c, 0.0) * 0.5;
      if (P.credit) {
        const cb = 1.0 * this.cf;
        let s = 0.0;
        for (let j = 0; j < 3; j++) s += Math.max(P.lam * this.prices[2] * this.kap_avg[j] * this.K[j] - this.loans[j], 0.0) * cb;
        avail += s;
      }
      this.inv_limited = cost > avail && cost > 0;
      if (this.inv_limited) for (let j = 0; j < 3; j++) inv[j] = inv[j] * avail / cost;
      return inv;
    }

    // 段 B7: 格上げの工房があれば、贅沢品の一製品あたり労働は、古い企業と工房の割合で混ぜたもの
    gammaEff() {
      if (this.p.up_gen > 0 && this.up_s > 0) {
        const g = F64(this.gamma);
        g[1] = (1 - this.up_s) * this.gamma[1] + this.up_s * this.up_gX;
        return g;
      }
      return this.gamma;
    }
    // 段 9: 生存の量を数える人数
    subsHeads() {
      const H = this.H, P = this.p;
      if (P.child_cost === 0) return H.heads;
      const out = zeros(H.heads.length);
      for (let i = 0; i < out.length; i++) out[i] = H.heads[i] * (1 + P.child_cost * (H.kids[i] > 0 ? 1 : 0));
      return out;
    }
    // 段 8: 贅沢品の種類の数
    variety() {
      const P = this.p;
      if (P.psi_v === 0 || !P.land) return 1.0;
      return Math.pow(Math.max(sum(this.H.ip) / P.ip_ref, 1e-6), P.psi_v);
    }
    bShare() {
      const P = this.p;
      if (P.psi_v === 0 || P.eps_v === 1.0) return P.b_N;
      const n = this.variety();
      const c = (1 - P.b_N) * Math.pow(n, (P.eps_v - 1) / (P.sigma_v - 1));
      return P.b_N / (P.b_N + c);
    }
    // 段 B6: プレイヤーの「何を削るか」(超えた支出のうち贅沢品に回す割合)
    bVec() {
      const b = this.bShare();
      if (this.pl === null || this.act.lux === undefined || this.act.lux === null) return b;
      const v = new Float64Array(this.H.M.length).fill(b);
      v[this.pl] = 1.0 - clip(this.act.lux, 0.0, 1.0);
      return v;
    }

    demand(price, inv, B) {
      const P = this.p;
      const PN = price[0], PL = price[1];
      const sh = this.subsHeads();
      const bN_ = this.bVec();
      let xN = 0.0, xL = 0.0;
      for (let i = 0; i < B.length; i++) {
        const base = PN * sh[i] * P.q0;
        const disc = softplus(B[i] - base, 10.0 / Math.max(base, 1e-9));
        const b = at(bN_, i);
        xN += (B[i] - disc) / PN + b * disc / PN;
        xL += (1 - b) * disc / PL;
      }
      const xI = (P.alpha[0] * xN + P.alpha[1] * xL + dot(this.kap_new, inv)) / (1 - P.alpha[2]);
      return [xN, xL, xI];
    }

    crisisOn() {
      const P = this.p;
      return !!(P.crisis_track || P.inventory || P.cap_N > 0 || P.g_stock > 0 || P.g_pw > 0 || P.f_K > 0 || this.S_G > 0
                || P.crisis_rule !== 'none' || P.delta_scrap > 0);
    }
    supplyAtCap(cap, PI) {
      const P = this.p;
      const var0 = P.alpha[0] * PI + (P.land ? P.rho_T * cap : 0.0);
      const y = ((cap - var0) / (this.W * this.gamma[0]) - 1.0) / P.omega;
      if (y <= 1e-12) return 0.0;
      const u = 1.0 + Math.log(Math.expm1(Math.min(y * 20.0, 700.0))) / 20.0;
      return this.K[0] * Math.max(u, 0.0);
    }
    evaluateCrisis(price, inv, B) {
      const P = this.p;
      const [PN, PL, PI] = price;
      const sh = this.subsHeads();
      const bN_ = this.bVec();
      let dN = 0.0, xL = 0.0;
      for (let i = 0; i < B.length; i++) {
        const base = PN * sh[i] * P.q0;
        const disc = softplus(B[i] - base, 10.0 / Math.max(base, 1e-9));
        const b = at(bN_, i);
        dN += (B[i] - disc) / PN + b * disc / PN;
        xL += (1 - b) * disc / PL;
      }
      const capped = P.cap_N > 0 && PN >= P.cap_N * (1 - 1e-9);
      const gbuy = capped ? 0.0 : this._gbuy;
      const D = dN + gbuy;
      const Inv = P.inventory ? (this.InvN === null ? P.inv_target * dN : this.InvN) : 0.0;
      const plan = P.inventory ? Math.max(D + P.chi_inv * (P.inv_target * D - Inv), 0.0) : D;
      let rel = 0.0, xN, firm_sales;
      if (capped) {
        const xs = this.supplyAtCap(P.cap_N, PI);
        xN = Math.min(xs, plan);
        if (P.crisis_rule === 'last' || P.crisis_rule === 'share') {
          const need = P.q0 * sum(sh);
          xN = Math.max(xN, Math.min(plan, Math.max(need - Inv - this.S_G, 0.0)));
        }
        firm_sales = Math.min(D, xN + Inv);
        rel = Math.min(D - firm_sales, this.S_G);
      } else {
        xN = plan;
        firm_sales = Math.min(D, xN + Inv);
        if (firm_sales < D) rel = Math.min(D - firm_sales, this.S_G);
      }
      const sales = firm_sales + rel;
      const hhN = sales - gbuy;
      const GI = this._GI_nom / PI;
      const xI = (P.alpha[0] * xN + P.alpha[1] * xL + dot(this.kap_new, inv) + GI) / (1 - P.alpha[2]);
      const x = [xN, xL, xI];
      const u = [0, 0, 0], geff = [0, 0, 0], cost = [0, 0, 0], m = [0, 0, 0];
      const ge = this.gammaEff();
      for (let j = 0; j < 3; j++) {
        u[j] = x[j] / this.K[j];
        geff[j] = ge[j] * (1 + P.omega * softplus(u[j] - 1.0, 20.0));
        cost[j] = this.W * geff[j] + PI * P.alpha[j];
        m[j] = this.m0[j] + P.zeta * Math.tanh(u[j] - P.u_star);
      }
      if (P.land) cost[0] = cost[0] + P.rho_T * PN;
      if (P.inventory) m[0] = m[0] + P.zeta_inv * Math.tanh(1.0 - (Inv + xN - firm_sales) / Math.max(P.inv_target * D, 1e-12));
      this._cx = { dN, D, gbuy, xN, firm_sales, rel, hhN, short: Math.max(dN - hhN, 0.0), capped, GI, Inv };
      return { x, u, geff, cost, m };
    }

    evaluate(price, inv, B) {
      const P = this.p;
      if (this.crisisOn()) return this.evaluateCrisis(price, inv, B);
      this._cx = null;
      const x = this.demand(price, inv, B);
      const u = [0, 0, 0], geff = [0, 0, 0], cost = [0, 0, 0], m = [0, 0, 0];
      const ge = this.gammaEff();
      for (let j = 0; j < 3; j++) {
        u[j] = x[j] / this.K[j];
        geff[j] = ge[j] * (1 + P.omega * softplus(u[j] - 1.0, 20.0));
        cost[j] = this.W * geff[j] + price[2] * P.alpha[j];
        m[j] = this.m0[j] + P.zeta * Math.tanh(u[j] - P.u_star);
      }
      if (P.land) cost[0] = cost[0] + P.rho_T * price[0];
      return { x, u, geff, cost, m };
    }

    residual(z, inv, B) {
      const P = this.p;
      const price = z.map(Math.exp);
      const ev = this.evaluate(price, inv, B);
      const res = [];
      for (let j = 0; j < 3; j++) {
        let target = (1 + ev.m[j]) * ev.cost[j];
        if (j === 0 && P.cap_N > 0) target = Math.min(target, P.cap_N);
        res.push(z[j] - Math.log(target));
      }
      const st = Math.sqrt(P.tactic);
      for (let j = 0; j < 3; j++) res.push(st * (z[j] - Math.log(this.prices[j])));
      return res;
    }

    solvePrices(inv, B) {
      let sol = IM.lmSolve(z => this.residual(z, inv, B), Array.from(this.prices, Math.log), 1e-13, 0.0);
      const P = this.p;
      if (P.cap_N > 0 && Math.exp(sol.x[0]) >= P.cap_N * (1 - 1e-6)) {
        // 価格の上限が効くとき: 必需品の価格を上限にそろえ、残りの二つを解き直す(im1.py と同じ。上限の角を解から外す)
        const z0 = Math.log(P.cap_N);
        const full = y => this.residual([z0, y[0], y[1]], inv, B);
        const sub = IM.lmSolve(y => { const r = full(y); return [r[1], r[2], r[4], r[5]]; }, sol.x.slice(1), 1e-13, 0.0);
        sol = { x: [z0, sub.x[0], sub.x[1]], fun: full(sub.x), nfev: sol.nfev + sub.nfev, nit: sol.nit + sub.nit };
      }
      return { price: sol.x.map(Math.exp), sol };
    }

    // ------------------------------------------------ 土地の市場、貸倒れ、世代の入れ替わり(段 2)
    landMarket(rent, roy, rL, W, wage_bill, E, Yd, lab_inc, div, bN, bL) {
      const P = this.p, H = this.H, n = H.M.length;
      const T = P.T_land;
      const wbar = wage_bill / Math.max(E, 1e-9);
      const U_lim = zeros(n);
      for (let i = 0; i < n; i++) U_lim[i] = P.mu_u * wbar * H.heads[i] * this.cf;
      const ip_tot = sum(H.ip);
      const v_IP = ip_tot > 0 ? roy / Math.max(ip_tot * (rL + P.delta_IP), 1e-9) : 0.0;
      const v_eq = Math.max(div, 0.0) / Math.max(this.rbar, 1e-3);
      const y_T = rent / (T * this.pT);
      const x_T = 0.2 * Math.tanh((y_T + this.gT_e - rL) / 0.2);
      const theta = Math.min(P.theta0 * Math.exp(P.theta1 * x_T) + P.theta_shock, 0.9);
      const other = zeros(n), human = zeros(n);
      for (let i = 0; i < n; i++) {
        other[i] = H.M[i] + v_IP * H.ip[i] + v_eq * H.eq[i];
        human[i] = P.h_y * Math.max(lab_inc[i] * (1 - this.tau), 0.0);
      }
      const land = H.land, chi = P.chi_T * P.attention, lT = P.lam_T;
      const holdings = (pr, out) => {
        let s = 0.0;
        for (let i = 0; i < n; i++) {
          const nw = other[i] + pr * land[i];
          const d = Math.max(0.0, theta * (nw + human[i]) / pr);
          const cap = (U_lim[i] + H.M[i] + pr * land[i]) / ((1 - lT) * pr);
          const v = Math.min(Math.max(land[i] + chi * (d - land[i]), 0.0), Math.max(cap, 0.0));
          if (out) out[i] = v;
          s += v;
        }
        return s;
      };
      let lo = Math.log(this.pT) - 8, hi = Math.log(this.pT) + 8;
      for (let it = 0; it < 100; it++) {
        const mid = 0.5 * (lo + hi);
        if (holdings(Math.exp(mid)) > T) lo = mid; else hi = mid;
      }
      const pT = Math.exp(0.5 * (lo + hi));
      const nw_ = zeros(n);
      const s0 = holdings(pT, nw_);
      const land_err = s0 / T - 1;
      for (let i = 0; i < n; i++) nw_[i] = nw_[i] * (T / s0);
      for (let i = 0; i < n; i++) H.M[i] = H.M[i] - pT * (nw_[i] - land[i]);
      H.land = nw_;
      this.gT_e += P.nu_T * (Math.log(pT / this.pT) - this.gT_e);
      this.pT = pT;
      // 貸倒れ: 借入の上限を超えた借金を棒引きにする。損は預金者が預金に応じて負う
      const risk = zeros(n), over = zeros(n);
      let wo = 0.0, breakups = 0.0;
      for (let i = 0; i < n; i++) {
        const limit = lT * pT * H.land[i] + U_lim[i] + 0.0;
        over[i] = Math.max(0.0, -H.M[i] - limit);
        risk[i] = Math.max(-H.M[i], 0.0) / Math.max(limit, 1e-12);
        wo += over[i];
        if (over[i] > P.breakup_eps * Math.max(limit, 1e-12)) breakups += H.heads[i];
      }
      this.risk = risk;
      this.over = over;
      for (let i = 0; i < n; i++) H.M[i] = H.M[i] + over[i];
      let bailout = 0.0;
      if (wo > 0) {
        const dep = zeros(n);
        for (let i = 0; i < n; i++) dep[i] = Math.max(H.M[i], 0.0);
        const ds = sum(dep);
        const cut = Math.min(wo, P.haircut_max * ds);
        for (let i = 0; i < n; i++) H.M[i] = H.M[i] - cut * dep[i] / Math.max(ds, 1e-12);
        bailout = wo - cut;
        if (bailout > 0) {
          this.L.pay('bank bailout', this.sec.gov, this.sec.hh, bailout);
          this.L.trade('bonds', this.sec.bank, this.sec.gov, bailout);
        }
      }
      // 世代の入れ替わり: 去る世帯の財産を相続させる。新しい世帯は財産なしで始まる
      let uniform = true;
      for (let i = 0; i < n; i++) if (!(Math.abs(H.heads[i] - H.heads[0]) <= 1e-8 + 1e-5 * Math.abs(H.heads[0]))) { uniform = false; break; }
      this.uniform_w = uniform;
      this.pl_heir_now = false;
      if (P.turnover > 0) {
        const draw = this.rng.random(n);
        for (let i = 0; i < n; i++) {
          if (!(draw[i] < P.turnover)) continue;
          if (i === this.pl && P.player_heir === 'self') {
            // 段 19: プレイヤーの次の代が、財産を継いで遊びを続ける
            if (H.pf[i] !== 0.0) { H.M[i] += H.pf[i]; H.pf[i] = 0.0; }
            if (P.child_cost > 0) H.kids[i] = P.child_years;
            H.skill[i] = 0.0; H.age[i] = 0.0; H.pts[i] = 0.0;
            this.pl_gen += 1;
            this.pl_heir_now = true;
            continue;
          }
          let h;
          if (uniform) { h = this.rng.integers(n - 1); h = h + (h >= i ? 1 : 0); }
          else {
            const pw = F64(H.heads); pw[i] = 0.0;
            const ps = sum(pw);
            for (let k = 0; k < n; k++) pw[k] = pw[k] / ps;
            h = this.rng.choice(n, pw);
          }
          const w_other = F64(H.heads); w_other[i] = 0.0;
          const ws = sum(w_other);
          for (let k = 0; k < n; k++) w_other[k] /= ws;
          if (H.pf[i] !== 0.0) { H.M[i] += H.pf[i]; H.pf[i] = 0.0; }
          for (const name of ['M', 'land', 'ip', 'eq', 'F']) {
            const arr = H[name];
            const val = arr[i];
            arr[i] = 0.0;
            if (P.inherit === 'lottery') arr[h] += val;
            else for (let k = 0; k < n; k++) arr[k] += val * w_other[k];
          }
          if (P.child_cost > 0) H.kids[i] = P.child_years;
          H.skill[i] = 0.0; H.age[i] = 0.0; H.pts[i] = 0.0;
        }
      }
      let hl = 0.0, nb = 0.0;
      const nwpc = zeros(n), nwpos = zeros(n);
      for (let i = 0; i < n; i++) {
        hl += Math.max(-H.M[i], 0.0);
        if (H.M[i] < 0) nb += H.heads[i];
        const nw = H.M[i] + pT * H.land[i] + v_IP * H.ip[i] + v_eq * H.eq[i] + H.F[i];
        nwpc[i] = nw / H.heads[i];
        nwpos[i] = Math.max(nw, 0) / H.heads[i];
      }
      return { pT, theta, land_err, wo_h: wo, bailout, breakups, hh_debt: hl, n_borrowers: nb,
               gini_nw: giniW(nwpc, H.heads), top10_nw: topShareW(nwpos, H.heads), land_value: pT * T, ip_tot: sum(H.ip),
               land_sum_err: Math.abs(sum(H.land) - T) };
    }

    planForeign(r_out, fx = 1.0) {
      const P = this.p, H = this.H, n = H.M.length;
      const Fv = zeros(n), nw = zeros(n);
      let nws = 0.0;
      for (let i = 0; i < n; i++) {
        Fv[i] = H.F[i] * fx;
        nw[i] = H.M[i] + this.pT * H.land[i] + (this.qIP || 0.0) * H.ip[i] + this.v_eq_last * H.eq[i] + Fv[i];
        nws += nw[i];
      }
      const lrate = P.wealth_tax + ((P.resolve === 'levy' && this.b_last > P.theta_resolve) ? P.levy_rate : 0.0);
      const ex = P.levy_exempt * Math.max(nws / sum(H.heads), 1e-12);
      const rD = P.credit ? this.r_policy : 0.0;
      const dF = zeros(n);
      for (let i = 0; i < n; i++) {
        const pc = nw[i] / H.heads[i];
        const eff = lrate * (1 - P.enf_F) * Math.max(pc - ex, 0.0) / Math.max(pc, 1e-12);
        const th = clip(P.theta_F0 + P.phi_K * (r_out - rD) + P.phi_L * eff, 0.0, P.theta_F_max);
        const d = P.chi_F * P.attention * (th * Math.max(nw[i], 0.0) - Fv[i]);
        dF[i] = Math.min(Math.max(d, -Fv[i]), Math.max(H.M[i], 0.0));
      }
      return dF;
    }

    // ------------------------------------------------ 災害(段 4)
    applyDisaster(size, scope = 'common') {
      const P = this.p, H = this.H, L = this.L, n = H.M.length;
      const d = size * Math.exp(-this.prot / P.prot_scale);
      const hit = new Array(n).fill(false), add = zeros(n);
      let dK;
      if (scope === 'common') {
        hit.fill(true);
        add.fill(Math.min(P.inj_coef * d, 1.0));
        dK = d;
      } else {
        const ord = this.dis_rng.permutation(n);
        const hs = sum(H.heads);
        let c = 0.0, any = false;
        for (const i of ord) {
          c += H.heads[i];
          if (c / hs <= P.hit_share + 1e-12) { hit[i] = true; any = true; }
        }
        if (!any) hit[ord[0]] = true;
        let hsum = 0.0;
        for (let i = 0; i < n; i++) if (hit[i]) hsum += H.heads[i];
        const share = hsum / hs;
        for (let i = 0; i < n; i++) add[i] = hit[i] ? Math.min(P.inj_coef * d / share, 1.0) : 0.0;
        dK = d * share;
      }
      for (let i = 0; i < n; i++) H.inj[i] = 1 - (1 - H.inj[i]) * (1 - add[i]);
      const lost = zeros(3);
      for (let j = 0; j < 3; j++) { lost[j] = dK * this.K[j]; this.K[j] = this.K[j] - lost[j]; }
      for (let j = 0; j < 3; j++) L.revalue(this.real[j], this.firms[j], -lost[j]);
      const need = zeros(n);
      for (let i = 0; i < n; i++) {
        const li = this.lab_inc_last ? Math.max(this.lab_inc_last[i], 0.0) : 0.0;
        need[i] = P.relief_rate * li * add[i] / P.rec_inj;
      }
      let relief = sum(need);
      if (P.relief === 'state' && relief > 0) {
        L.pay('disaster relief', this.sec.gov, this.sec.hh, relief);
        L.trade('bonds', this.sec.bank, this.sec.gov, relief);
        for (let i = 0; i < n; i++) H.M[i] = H.M[i] + need[i];
      } else if (P.relief === 'mutual' && relief > 0) {
        const dep = zeros(n);
        for (let i = 0; i < n; i++) dep[i] = hit[i] ? 0.0 : Math.max(H.M[i], 0.0);
        const ds = sum(dep);
        const pay = Math.min(relief, ds);
        for (let i = 0; i < n; i++) H.M[i] = H.M[i] - pay * dep[i] / Math.max(ds, 1e-12) + need[i] * (pay / relief);
        relief = pay;
      } else if (P.relief === 'none') relief = 0.0;
      this.last_hit = hit;
      let kl = 0.0;
      for (let j = 0; j < 3; j++) kl += lost[j] * this.prices[2] * this.kap_avg[j];
      this.dis_rec = { dis_size: size, dis_d: d, relief, K_lost: kl };
    }

    // ------------------------------------------------ 学習と政治(段 5)
    updateBelief() {
      const P = this.p;
      const n = this.events_this_year || 0;
      this.n_eff = P.learn_omega * this.n_eff + n;
      this.t_eff = P.learn_omega * this.t_eff + 1.0;
      const a = P.prior_a + this.n_eff, b = P.prior_b + this.t_eff;
      const lam = a / b;
      this.belief = lam;
      P.protect = Math.min(P.protect_max, P.k_learn * lam);
    }

    holdVote() {
      const P = this.p, H = this.H, n = H.M.length;
      const cap = zeros(n);
      for (let i = 0; i < n; i++) cap[i] = Math.max(this.cap_inc_last[i], 0.0);
      const rent = new Array(n);
      for (let i = 0; i < n; i++) rent[i] = H.asset_share[i] > P.rentier_cut;
      const heads = H.heads;
      const hs = sum(heads);
      let yp = 0.0;
      for (let i = 0; i < n; i++) yp += Math.max(H.Yd[i], 0.0);
      const inc_pc = Math.max(yp / hs, 1e-12);
      const dtau = P.tau_hi - P.tau_lo;
      const transfer_pc = dtau * sum(cap) / hs;
      const stake = zeros(n);
      for (let i = 0; i < n; i++) stake[i] = (transfer_pc * heads[i] - dtau * cap[i]) / heads[i] / inc_pc;
      const k = this.pl;
      let pl_g = null;
      const mask_pl = new Array(n).fill(true);
      if (k !== null) {
        const side0 = this.act.vote;
        pl_g = side0 === 'hi' ? 'w' : side0 === 'lo' ? 'k' : (rent[k] ? 'k' : 'w');
        mask_pl[k] = false;
      }
      let z = {};
      const tally = sway => {
        const out = {};
        for (const [g, isRent, rho0, sign] of [['w', false, P.rho_w, +1], ['k', true, P.rho_k, -1]]) {
          let hm = 0.0, sw = 0.0, any = false;
          for (let i = 0; i < n; i++) if (rent[i] === isRent && mask_pl[i]) { any = true; hm += heads[i]; sw += stake[i] * heads[i]; }
          if (!any) { out[g] = [0.0, 0.0]; continue; }
          const rho = rho0 + (g === pl_g ? sway : 0.0);
          let A = Math.max(sign * (sw / hm), 0.0);
          if (A < 1e-12) A = 0.0;                    // im1.py の portable の道と同じ(資産家がいない年の丸め)
          const t = -Math.expm1(-A * Math.sqrt(rho / Math.PI) / P.c_vote);   // im1.py の portable の道と同じ(桁落ちを避ける)
          let v = hm * t;
          if (P.vote_noise) v = Math.max(v + hm * Math.sqrt(rho * t * (1 - t)) * z[g], 0.0);
          out[g] = [v, t];
        }
        return out;
      };
      if (P.vote_noise) z = { w: this.vote_rng.normal(), k: this.vote_rng.normal() };
      let out = tally(0.0);
      let nr = 0.0;
      for (let i = 0; i < n; i++) if (rent[i]) nr += heads[i];
      if (P.vote_noise && out.w[0] === 0.0 && out.k[0] === 0.0) {
        P.redistrib = 1.0;
        this.vote_rec = { vote_hi: 0.0, vote_lo: 0.0, turnout_w: 0.0, turnout_k: 0.0, n_rentier: nr };
        return;
      }
      if (k !== null) {
        const camp = clip(this.act.campaign || 0.0, 0.0, 1.0);
        const side = this.act.vote;
        const base_hi = out.w[0] > out.k[0];
        if (camp > 0) out = tally(P.k_sway * camp);
        let hi_v = out.w[0], lo_v = out.k[0];
        if (side === 'hi') hi_v += heads[k];
        else if (side === 'lo') lo_v += heads[k];
        const win_hi = hi_v > lo_v;
        const pivot = win_hi !== base_hi;
        this.pl_pivots += pivot ? 1 : 0;
        this.vote_pl = { pl_vote: side || '', pl_campaign: camp, pl_pivot: pivot ? 1.0 : 0.0, pl_rentier_vote: rent[k] };
        out = { w: [hi_v, out.w[1]], k: [lo_v, out.k[1]] };
      }
      P.tau_k = out.w[0] > out.k[0] ? P.tau_hi : P.tau_lo;
      P.redistrib = 1.0;
      this.vote_rec = { vote_hi: out.w[0], vote_lo: out.k[0], turnout_w: out.w[1], turnout_k: out.k[1], n_rentier: nr };
    }

    // ------------------------------------------------ 一年(基本の層)
    beginYear() {
      const P = this.p, H = this.H, n = H.M.length;
      this.dis_rec = {};
      if (P.up_gen !== this.up_gen_seen) {
        // 段 B7: 工房を開く(開き直す)
        this.up_gen_seen = P.up_gen;
        if (P.up_gen > 0) { this.up_s = P.up_s0; this.up_gX = P.up_g0; this.up_cum = P.up_cum0; this.up_peak = P.up_s0; }
        else { this.up_s = 0.0; this.up_gX = P.up_g0; this.up_cum = 0.0; this.up_peak = 0.0; }
      }
      if (P.disaster) {
        for (const q of this.shock_queue.filter(q => q[0] === this.t)) this.applyDisaster(q[1], q[2]);
        const q = P.dis_rate;
        if (P.dis_rate > 0 && this.dis_rng.random() < q) {
          const size = Math.min(P.dis_min * Math.pow(1 - this.dis_rng.random(), -1 / P.dis_alpha), P.dis_max);
          this.applyDisaster(size, P.dis_scope);
        }
        this.events_this_year = Object.keys(this.dis_rec).length ? 1 : 0;
      }
      if (P.learn) this.updateBelief();
      if (P.vote && this.t % P.vote_every === 0 && this.t > 0) this.holdVote();
      this._inv = this.planInvestment();
      if (this.crisisOn()) {
        const hhN0 = this.N_last;
        this._gbuy = (P.g_stock > 0 && hhN0) ? P.chi_S * Math.max(P.g_stock * hhN0 - this.S_G, 0.0) : 0.0;
        this._GI_nom = (P.g_pw > 0 && this.Y_prev) ? P.g_pw * this.Y_prev : 0.0;
      }
      if (P.land) {
        for (let i = 0; i < n; i++) H.extra[i] = P.c_h * Math.max(this.pT * H.land[i] - Math.max(-H.M[i], 0.0), 0.0);
      }
      this._B = H.budgets();
      if (this.pl !== null) {
        // 段 19: プレイヤーの支出(規則の予算の倍率)と、子を持つ選び方
        this._B[this.pl] = Math.max(this._B[this.pl] * (this.act.spend === undefined ? 1.0 : this.act.spend), 0.0);
        if (this.act.child && P.child_cost > 0 && H.kids[this.pl] === 0) {
          H.kids[this.pl] = P.child_years;
          this.act.child = false;
        }
      }
      this._vat_i = null;
      this._Bsum_last = sum(this._B);
      return [this._inv, this._B];
    }

    step() {
      const [inv, B] = this.beginYear();
      const { price, sol } = this.solvePrices(inv, B);
      const ev = this.evaluate(price, inv, B);
      const qc = this._cx === null ? [ev.x[0], ev.x[1]] : [this._cx.hhN, ev.x[1]];
      const r3 = sol.fun.slice(0, 3);
      const rec = this.finishYear(price, price, ev.x, ev.u, ev.geff, ev.cost, qc, Math.sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]));
      rec.books = this.L.check().max;
      rec.books_tol = 1e-6 * Math.max(1.0, Math.abs(this.L.st(this.sec.gov, 'bonds')));
      this.L.close();
      this.log.push(rec);
      return rec;
    }

    route(flow, payer, good, amount) {
      const to = this.firms[good];
      if (to !== payer) this.L.pay(flow, payer, to, amount);
    }

    finishYear(price, pbuy, x, u, geff, cost, qc, resid) {
      const P = this.p, L = this.L, H = this.H, n = H.M.length;
      const inv = this._inv, B = this._B;
      const hh = this.sec.hh, bank = this.sec.bank, gov = this.sec.gov;
      const F = this.firms;
      const [PN, PL, PI] = price;
      const [bN, bL, bI] = pbuy;
      const W = this.W;
      let S = P.pop_w / (1.0 + Math.exp(-P.beta_s * (Math.log(W / bN) - Math.log(P.w_res))));
      const ret = new Array(n).fill(false);
      let anyRet = false;
      if (P.ret_age > 0) for (let i = 0; i < n; i++) { ret[i] = H.age[i] >= P.ret_age; if (ret[i]) anyRet = true; }
      let pop_eff = P.pop_w;
      if (P.ret_age > 0) {
        let a = 0.0;
        for (let i = 0; i < n; i++) if (!ret[i]) a += H.lab[i];
        const ws = a / sum(H.lab);
        S = S * ws;
        pop_eff = P.pop_w * ws;
      }
      let Dl = geff[0] * x[0] + geff[1] * x[1] + geff[2] * x[2];
      let L_R = [0.0, 0.0, 0.0];
      if (P.research) {
        let budget = [0, 1, 2].map(j => this.sR[j] * price[j] * x[j]);
        if (this.prof_R_last !== null) budget = budget.map((v, j) => Math.min(v, P.cap_R * Math.max(this.prof_R_last[j], 0.0)));
        L_R = budget.map(v => v / W);
        Dl = Dl + (L_R[0] + L_R[1] + L_R[2]);
      }
      let prot_L = 0.0;
      if (P.disaster) {
        const inj_share = dot(H.lab, H.inj) / sum(H.lab);
        S = S * (1 - inj_share);
        prot_L = P.protect * this.Y_last / W;
        Dl = Dl + prot_L;
      }
      const pub_L = (P.s_pub > 0 && this.Y_prev) ? P.s_pub * this.Y_prev / W : 0.0;
      Dl = Dl + pub_L;
      let E = softmin(Dl, S);
      let unemp = pop_eff - E;
      const wages_j = [0, 1, 2].map(j => W * geff[j] * x[j]);
      const cx = this._cx;
      if (cx !== null && P.f_K > 0) {
        for (let j = 0; j < 3; j++) wages_j[j] = wages_j[j] + W * P.f_K * this.K[j];
        const Dl_fix = P.f_K * (this.K[0] + this.K[1] + this.K[2]);
        Dl = Dl + Dl_fix;
        E = softmin(Dl, S);
        unemp = pop_eff - E;
      }
      const ben = P.benefit * P.q0 * bN * unemp;
      const profit = [0, 1, 2].map(j => (price[j] - cost[j]) * x[j]);
      let subsidy = 0.0;
      const share_pay = [0.0, 0.0, 0.0];
      if (cx !== null) {
        profit[0] = PN * cx.firm_sales - cost[0] * x[0];
        if (P.f_K > 0) for (let j = 0; j < 3; j++) profit[j] = profit[j] - W * P.f_K * this.K[j];
        if ((P.crisis_rule === 'last' || P.crisis_rule === 'share') && profit[0] < 0) {
          if (P.crisis_rule === 'last') subsidy = -profit[0];
          else {
            const rev1 = price[1] * x[1], rev2 = price[2] * x[2];
            share_pay[1] = -profit[0] * rev1 / (rev1 + rev2);
            share_pay[2] = -profit[0] * rev2 / (rev1 + rev2);
          }
          profit[0] = 0.0;
        }
      }
      const inv_cost_j = [0, 1, 2].map(j => bI * this.kap_new[j] * inv[j]);
      // ---- 信用: 借入、利子、銀行の利益
      const rL = this.r_loan, rD = P.credit ? this.r_policy : 0.0;
      const loan_loss = [0.0, 0.0, 0.0];
      const Lf = this.loans;
      const B_bank = L.st(bank, 'bonds');
      const dL = [0.0, 0.0, 0.0];
      if (P.credit) {
        for (let j = 0; j < 3; j++) {
          const cap = P.lam * bI * this.kap_avg[j] * this.K[j];
          const v = (1 - P.f_R) * inv_cost_j[j] - P.amort * Lf[j];
          dL[j] = Math.min(Math.max(v, -Lf[j]), Math.max(cap - Lf[j], 0.0));
          if (this.cf < 1.0 && dL[j] > 0) dL[j] = dL[j] * this.cf;
        }
      }
      let up_pay = 0.0;
      if (P.up_gen > 0 && this.up_s > 0 && P.up_sub > 0) {
        up_pay = P.up_sub * this.up_s * W * this.up_gX * x[1];       // 段 B7: 工房の労働の費用の一部を政府が埋める
        profit[1] = profit[1] + up_pay;
      }
      const int_f = [0, 1, 2].map(j => rL * Lf[j]);
      this.prof_R_last = [0, 1, 2].map(j => profit[j] - int_f[j]);
      const rent = P.land ? P.rho_T * PN * x[0] : 0.0;
      const roy = P.land ? P.phi_R * Math.max(profit[1], 0.0) : 0.0;
      const res_w = [0, 1, 2].map(j => W * L_R[j]);
      const royv = [0.0, roy, 0.0];
      const divj = [0, 1, 2].map(j => profit[j] - int_f[j] - inv_cost_j[j] + dL[j] - royv[j] - res_w[j] - share_pay[j]);
      const div = divj[0] + divj[1] + divj[2];
      const depo = zeros(n), debt = zeros(n);
      for (let i = 0; i < n; i++) { depo[i] = Math.max(H.M[i], 0.0); debt[i] = Math.max(-H.M[i], 0.0); }
      const rB = P.credit ? (this.r_bond !== null ? this.r_bond : rD) : rD;
      const int_D = rD * (sum(depo) + this.pfDeposits()), int_HL = rL * sum(debt), int_B = rB * B_bank;
      const bank_profit = (int_f[0] + int_f[1] + int_f[2]) + int_HL + int_B - int_D - (this.xf.int_nr || 0.0) - (loan_loss[0] + loan_loss[1] + loan_loss[2]);
      const cap_pre = div + bank_profit;
      const tau_k = P.tau_k < 0 ? this.tau : P.tau_k;
      const wsum = wages_j[0] + wages_j[1] + wages_j[2], rwsum = res_w[0] + res_w[1] + res_w[2];
      let tax_w = this.tau * (wsum + rwsum);
      let tax_c = tau_k * Math.max(cap_pre, 0.0);
      const tax_c_eq = tax_c;
      tax_c += tau_k * (rent + roy);
      let tr = P.redistrib * tax_c;
      if (P.g_tr > 0 && this.Y_prev) tr = tr + P.g_tr * this.Y_prev;
      const b_last = this.b_last;
      const crisis = b_last > P.theta_resolve;
      const levy_i = zeros(n);
      const lrate = ((crisis && P.resolve === 'levy') ? P.levy_rate : 0.0) + P.wealth_tax;
      if (lrate > 0) {
        const nw = zeros(n);
        let nws = 0.0;
        for (let i = 0; i < n; i++) {
          nw[i] = H.M[i] + this.pT * H.land[i] + (this.qIP || 0.0) * H.ip[i] + this.v_eq_last * H.eq[i] + P.enf_F * H.F[i];
          nws += nw[i];
        }
        const ex = P.levy_exempt * Math.max(nws / sum(H.heads), 1e-12);
        for (let i = 0; i < n; i++) levy_i[i] = lrate * Math.max(nw[i] / H.heads[i] - ex, 0.0) * H.heads[i];
      }
      if (crisis && P.resolve === 'austerity') tax_w = tax_w + P.austerity * wsum;
      const levy = sum(levy_i);
      // ---- 帳簿への書き込み
      const cons = [bN * qc[0], bL * qc[1]];
      for (let j = 0; j < 3; j++) {
        L.pay('wages', F[j], hh, wages_j[j]);
        L.pay('research wages', F[j], hh, res_w[j]);
      }
      let gov_buy = 0.0;
      if (cx !== null) {
        const rel_hh = Math.min(cx.rel, qc[0]);
        this.route('consumption', hh, 0, bN * (qc[0] - rel_hh));
        L.pay('stockpile release', hh, gov, bN * rel_hh);
        this.route('stockpile purchase', gov, 0, bN * cx.gbuy);      // 買い手の価格(一国では売る価格と同じ)
        this.route('public works', gov, 2, bI * cx.GI);
        L.pay('production subsidy', gov, F[0], subsidy);
        for (const j of [1, 2]) L.pay('loss sharing', F[j], F[0], share_pay[j]);
        gov_buy = bN * cx.gbuy + bI * cx.GI + subsidy - bN * rel_hh;
        const dInv = P.inventory ? x[0] - cx.firm_sales : 0.0;
        this.InvN = P.inventory ? (cx.Inv + dInv) : null;
        L.produceReal('stock_N' + this.sfx, F[0], dInv);
        const dS = cx.gbuy - cx.rel - P.delta_S * (this.S_G + cx.gbuy - cx.rel);
        L.produceReal('stock_G' + this.sfx, gov, dS);
        this.S_G = this.S_G + dS;
      } else {
        this.route('consumption', hh, 0, cons[0]);
      }
      this.route('consumption', hh, 1, cons[1]);
      for (let j = 0; j < 3; j++) {
        this.route('intermediate', F[j], 2, bI * P.alpha[j] * x[j]);
        this.route('investment goods', F[j], 2, inv_cost_j[j]);
        L.produceReal(this.real[j], F[j], -P.delta * this.K[j] + inv[j]);
        L.trade('loans', bank, F[j], dL[j]);
        L.pay('loan interest', F[j], bank, int_f[j]);
        L.pay('dividends', F[j], hh, divj[j]);
      }
      L.pay('land rent', F[0], hh, rent);
      L.pay('royalties', F[1], hh, roy);
      L.pay('deposit interest', bank, hh, int_D);
      L.pay('loan interest', hh, bank, int_HL);
      L.pay('bond interest', gov, bank, int_B);
      L.pay('dividends', bank, hh, bank_profit);
      L.pay('benefits', gov, hh, ben);
      L.pay('transfers', gov, hh, tr);
      // 段 6: 外の資産の利子、外の資産の売り買い、財政同盟の給付
      const xf = this.xf;
      const incF = xf.incF || 0.0, dF = xf.dF || null;
      const ben_U = xf.union_ben || 0.0, union_in = xf.union_in || 0.0;
      const Fs = sum(H.F);
      const incF_i = zeros(n);
      if (Fs > 0) for (let i = 0; i < n; i++) incF_i[i] = incF * H.F[i] / Fs;
      const tax_F = tau_k * P.enf_F * incF;
      if ('ext' in this.sec) {
        L.pay('foreign income', this.sec.ext, hh, incF);
        if (dF !== null) L.trade('foreign', hh, this.sec.ext, sum(dF));
      }
      L.pay('union benefits', gov, hh, ben_U);
      tax_c = tax_c + tax_F;
      let prot_w = W * prot_L;
      L.pay('protection', gov, hh, prot_w);
      const pub_w = W * pub_L;
      L.pay('public research', gov, hh, pub_w);
      prot_w = prot_w + pub_w;
      tax_w = tax_w + this.tau * prot_w;
      // 段 10: 年金(賦課方式・貢献に比例)、児童手当
      const pen_i = zeros(n);
      if ((P.pension === 'payg' || P.pension === 'points') && anyRet) {
        let wh = 0.0, rh = 0.0;
        for (let i = 0; i < n; i++) { if (ret[i]) rh += H.heads[i]; else wh += H.heads[i]; }
        const wbar = wsum / Math.max(wh, 1e-12);
        const pen_tot = P.pen_rate * wbar * rh;
        let wp = 0.0;
        if (P.pension === 'points') for (let i = 0; i < n; i++) if (ret[i]) wp += H.pts[i];
        if (P.pension === 'points' && wp > 0) for (let i = 0; i < n; i++) pen_i[i] = ret[i] ? pen_tot * H.pts[i] / wp : 0.0;
        else for (let i = 0; i < n; i++) pen_i[i] = ret[i] ? pen_tot * H.heads[i] / rh : 0.0;
      }
      const ca_i = zeros(n);
      if (P.child_allow > 0) for (let i = 0; i < n; i++) ca_i[i] = P.child_allow * P.q0 * bN * H.heads[i] * (H.kids[i] > 0 ? 1 : 0);
      const pen_s = sum(pen_i), ca_s = sum(ca_i);
      L.pay('pensions', gov, hh, pen_s);
      L.pay('child allowance', gov, hh, ca_s);
      gov_buy = gov_buy + (pen_s + ca_s - 0.0);
      L.pay('workshop subsidy', gov, F[1], up_pay);
      gov_buy = gov_buy + up_pay;
      L.pay('taxes', hh, gov, tax_w + tax_c);
      L.pay('wealth tax', hh, gov, levy);
      const deficit = ben + tr + int_B + prot_w - tax_w - tax_c - levy + ben_U - union_in + gov_buy;
      L.trade('bonds', bank, gov, deficit);
      this.loans = F64([Lf[0] + dL[0], Lf[1] + dL[1], Lf[2] + dL[2]]);
      // ---- 標本の所得と預金
      let labw = H.lab;
      if (P.disaster) {
        labw = zeros(n);
        for (let i = 0; i < n; i++) labw[i] = H.lab[i] * (1 - H.inj[i]);
        const s = sum(labw);
        for (let i = 0; i < n; i++) labw[i] = labw[i] / s;
      }
      if (P.ret_age > 0) {
        const lw = zeros(n);
        for (let i = 0; i < n; i++) lw[i] = ret[i] ? 0.0 : labw[i];
        const s = sum(lw);
        for (let i = 0; i < n; i++) lw[i] = lw[i] / s;
        labw = lw;
      }
      if (this.pl !== null && (this.act.work === undefined ? 1.0 : this.act.work) !== 1.0) {
        const lw = F64(labw);
        lw[this.pl] *= clip(this.act.work, 0.0, 1.0);
        const s = sum(lw);
        for (let i = 0; i < n; i++) lw[i] = lw[i] / s;
        labw = lw;
      }
      const hs = sum(H.heads);
      const lab_tot = wsum + rwsum + prot_w + ben + ben_U;
      const lab_inc = zeros(n), cap_inc = zeros(n), tax_i = zeros(n);
      let Yd = zeros(n);
      for (let i = 0; i < n; i++) {
        lab_inc[i] = lab_tot * labw[i] + tr * H.heads[i] / hs + pen_i[i] + ca_i[i];
        cap_inc[i] = cap_pre * H.eq[i] + rD * depo[i] - rL * debt[i];
        tax_i[i] = tax_c_eq * H.eq[i];
        cap_inc[i] = cap_inc[i] + incF_i[i];
        tax_i[i] = tax_i[i] + tau_k * P.enf_F * incF_i[i];
      }
      if (P.land) {
        const ips = sum(H.ip);
        for (let i = 0; i < n; i++) {
          const land_inc = rent * H.land[i] / P.T_land + (ips > 0 ? roy * H.ip[i] / ips : roy * H.eq[i]);
          cap_inc[i] = cap_inc[i] + land_inc;
          tax_i[i] = tax_i[i] + tau_k * land_inc;
        }
      }
      this.cap_inc_last = cap_inc;
      for (let i = 0; i < n; i++) {
        Yd[i] = lab_inc[i] - this.tau * (wsum + rwsum + prot_w) * labw[i] + cap_inc[i] - tax_i[i] - levy_i[i];
        if (crisis && P.resolve === 'austerity') Yd[i] = Yd[i] - P.austerity * wsum * labw[i];
      }
      let spend = B;
      let hungry = 0.0, sub_gap = 0.0, hungry_kids = 0.0, hungry_ret = 0.0, got = null;
      const shq0 = this.subsHeads();
      if (cx !== null) {
        // 割り当て: 品不足なら、まず生存の量を配り、残りを足りない分に比例して配る。買えなかった分は預金に残る
        const bv = this.bVec();
        const dN_i = zeros(n);
        for (let i = 0; i < n; i++) {
          const base = bN * shq0[i] * P.q0;
          const disc = softplus(B[i] - base, 10.0 / Math.max(base, 1e-9));
          dN_i[i] = (B[i] - disc) / bN + at(bv, i) * disc / bN;
        }
        const avail = qc[0];
        got = dN_i;
        if (avail < sum(dN_i) * (1 - 1e-12)) {
          const sub = zeros(n);
          for (let i = 0; i < n; i++) sub[i] = Math.min(dN_i[i], shq0[i] * P.q0);
          const ss = sum(sub);
          got = zeros(n);
          if (avail <= ss) for (let i = 0; i < n; i++) got[i] = sub[i] * avail / ss;
          else {
            let rest = 0.0;
            for (let i = 0; i < n; i++) rest += dN_i[i] - sub[i];
            for (let i = 0; i < n; i++) got[i] = sub[i] + (dN_i[i] - sub[i]) * (avail - ss) / rest;
          }
          spend = zeros(n);
          for (let i = 0; i < n; i++) spend[i] = B[i] - (dN_i[i] - got[i]) * bN;
        }
        let hh_ = 0.0, gap = 0.0, kh = 0.0, kt = 0.0, rhh = 0.0, rt = 0.0, sht = 0.0;
        for (let i = 0; i < n; i++) {
          const short = got[i] < shq0[i] * P.q0 * (1 - 1e-6);
          if (short) hh_ += H.heads[i];
          gap += Math.max(shq0[i] * P.q0 - got[i], 0.0);
          sht += shq0[i];
          if (H.kids[i] > 0) { kt += H.heads[i]; if (short) kh += H.heads[i]; }
          if (ret[i]) { rt += H.heads[i]; if (short) rhh += H.heads[i]; }
        }
        hungry = hh_ / hs;
        sub_gap = gap / (sht * P.q0);
        hungry_kids = kt > 0 ? kh / Math.max(kt, 1e-12) : 0.0;
        hungry_ret = rt > 0 ? rhh / Math.max(rt, 1e-12) : 0.0;
        this.N_last = qc[0];
      }
      if (P.pension === 'funded' && P.ret_age > 0) {
        for (let i = 0; i < n; i++) {
          const contrib = ret[i] ? 0.0 : P.fund_rate * Math.max(lab_inc[i], 0.0);
          const draw = ret[i] ? P.fund_draw * H.pf[i] : 0.0;
          H.pf[i] = H.pf[i] * (1 + rD) + contrib - draw;
          Yd[i] = Yd[i] - contrib + draw;
        }
      } else if (sum(H.pf) > 0) for (let i = 0; i < n; i++) H.pf[i] = H.pf[i] * (1 + rD);
      if (P.pension === 'points' && P.ret_age > 0) {
        for (let i = 0; i < n; i++) if (!ret[i]) H.pts[i] = H.pts[i] + Math.max(lab_inc[i] - pen_i[i] - ca_i[i], 0.0);
      }
      for (let i = 0; i < n; i++) H.M[i] = H.M[i] + Yd[i] - spend[i] - 0.0;
      if (dF !== null) for (let i = 0; i < n; i++) { H.M[i] = H.M[i] - dF[i]; H.F[i] = H.F[i] + dF[i]; }
      H.Yd = Yd;
      for (let i = 0; i < n; i++) {
        H.asset_share[i] = (Math.abs(lab_inc[i]) + Math.abs(cap_inc[i]) > 0)
          ? Math.max(cap_inc[i], 0.0) / Math.max(lab_inc[i] + Math.max(cap_inc[i], 0.0), 1e-12) : 0.0;
      }
      if (P.luck_sigma > 0 || P.luck_skill > 0) {
        // 段 9: 家計ごとの資産の運と腕
        const z = P.luck_nu > 2 ? this.luck_rng.standard_t(P.luck_nu, n) : this.luck_rng.standard_normal(n);
        if (P.luck_nu > 2) for (let i = 0; i < n; i++) z[i] = z[i] * Math.sqrt((P.luck_nu - 2) / P.luck_nu);
        const hm = sum(H.heads) / n;
        const sc = zeros(n);
        for (let i = 0; i < n; i++) sc[i] = Math.sqrt(hm / H.heads[i]);
        if (this.pl !== null) sc[this.pl] = 1.0;
        if (P.luck_skill > 0) {
          const zz = this.luck_rng.standard_normal(n);
          for (let i = 0; i < n; i++) H.skill[i] = P.luck_rho * H.skill[i] + P.luck_skill * Math.sqrt(1 - P.luck_rho * P.luck_rho) * zz[i];
        }
        const tot = sum(H.eq);
        for (let i = 0; i < n; i++) H.eq[i] = H.eq[i] * Math.exp(clip(P.luck_sigma * sc[i] * z[i] + H.skill[i], -5.0, 5.0));
        const s2 = sum(H.eq);
        for (let i = 0; i < n; i++) H.eq[i] = H.eq[i] * tot / s2;
      }
      if (P.child_cost > 0) for (let i = 0; i < n; i++) H.kids[i] = Math.max(H.kids[i] - 1, 0.0);
      if (P.ret_age > 0 || P.age_track) for (let i = 0; i < n; i++) H.age[i] = H.age[i] + 1.0;
      let land_rec = {};
      if (P.land) {
        const bv = this.bVec();
        const shl = this.subsHeads();                       // 子を育てる年を減らしたあとの人数(im1.py と同じ順)
        let ipn = 0.0;
        for (let i = 0; i < n; i++) {
          const shq = shl[i] * P.q0;
          const lux_i = (1 - at(bv, i)) * softplus(B[i] - bN * shq, 10.0 / Math.max(bN * shq, 1e-9));
          const ip_new = P.eta_IP * P.sigma_R * lux_i / bL;
          H.ip[i] = H.ip[i] * (1 - P.delta_IP) + ip_new;
          ipn += ip_new;
        }
        land_rec = this.landMarket(rent, roy, rL, W, wsum, E, Yd, lab_inc, div, bN, bL);
        land_rec.ip_new = ipn;
      }
      // ---- 能力、技術、競争、賃金、財政
      const rho = [0, 1, 2].map(j => profit[j] / (bI * this.kap_avg[j] * this.K[j]) - P.delta);
      const K_old = F64(this.K);
      for (let j = 0; j < 3; j++) this.K[j] = (1 - P.delta) * this.K[j] + inv[j];
      const scrap = [0.0, 0.0, 0.0];
      if (P.delta_scrap > 0) {
        for (let j = 0; j < 3; j++) scrap[j] = P.delta_scrap * clip(-(rho[j] + P.delta) / 0.1, 0.0, 1.0) * this.K[j];
        if (P.crisis_rule === 'last') scrap[0] = 0.0;
        for (let j = 0; j < 3; j++) this.K[j] = this.K[j] - scrap[j];
        for (let j = 0; j < 3; j++) L.revalue(this.real[j], F[j], -scrap[j]);
      }
      for (let j = 0; j < 3; j++) this.gamma_new[j] *= (1 - P.g_gamma);
      const gains = [0.0, 0.0, 0.0];
      if (P.research) {
        for (let j = 0; j < 3; j++) {
          const lam_j = L_R[j] / P.trial_cost * P.p_hit * (P.culture_spill ? Math.min(Math.pow(this.variety(), P.culture_spill), P.spill_cap) : 1.0);
          if (!Number.isFinite(lam_j)) throw new Error('研究の試行の数が数でない');
          const hits = this.res_rng.poisson(Math.max(lam_j, 0.0));
          if (hits > 1000000) {
            const a = P.alpha_R;
            gains[j] = hits * (P.g0 + Math.pow(P.g0, a) * (Math.pow(P.g_cap, 1 - a) - Math.pow(P.g0, 1 - a)) / (1 - a));
          } else if (hits > 0) {
            const r = this.res_rng.random(hits);
            let s = 0.0;
            for (let k = 0; k < hits; k++) s += Math.min(P.g0 * Math.pow(1 - r[k], -1 / P.alpha_R), P.g_cap);
            gains[j] = s;
          }
        }
        for (let j = 0; j < 3; j++) this.gamma_new[j] *= Math.exp(-gains[j]);
        if (P.adaptive_R) {
          const Lp = [0, 1, 2].map(j => Math.max(geff[j] * x[j], 1e-9));
          if (this.g_hat === null) {
            const b0 = P.s_R / Math.max(P.s_R_max - P.s_R, 1e-9);
            this.g_hat = Lp.map(v => b0 / (v * P.h_R));
          }
          for (let j = 0; j < 3; j++) {
            const per = gains[j] / Math.max(L_R[j], 1e-12);
            this.g_hat[j] = P.omega_R * this.g_hat[j] + (1 - P.omega_R) * (L_R[j] > 0 ? per : this.g_hat[j]);
            const b = this.g_hat[j] * Lp[j] * P.h_R;
            this.sR[j] = Math.max(P.s_R_min, P.s_R_max * b / (1 + b));
          }
        }
      }
      for (let j = 0; j < 3; j++) this.gamma[j] = ((1 - P.delta) * K_old[j] * this.gamma[j] + inv[j] * this.gamma_new[j]) / this.K[j];
      const excess = 0.0;
      const m0_old = F64(this.m0);
      for (let j = 0; j < 3; j++) this.m0[j] = clip(this.m0[j] - P.eta * (rho[j] - this.rbar - excess), 0.01, 2.0);
      if (cx !== null && cx.capped) this.m0[0] = m0_old[0];
      const infl = Math.log(bN / this.PN_prev);
      this.infl_prev = infl;
      this.W = W * Math.exp(P.phi_w * (Dl / S - 1.0) + P.chi * infl);
      let Y = cons[0] + cons[1] + bI * dot(this.kap_new, inv);
      if (cx !== null) {
        Y = bN * qc[0] + cons[1] + bI * dot(this.kap_new, inv) + bN * cx.gbuy + bI * cx.GI - bN * Math.min(cx.rel, qc[0]);
        if (P.inventory) Y = Y + PN * (x[0] - cx.firm_sales);
      }
      this.Y_prev = Y;
      if (P.disaster) {
        this.Y_last = Y;
        this.lab_inc_last = lab_inc;
        this.prot = this.prot * (1 - P.prot_decay) + prot_L;
        for (let i = 0; i < n; i++) H.inj[i] = H.inj[i] * (1 - P.rec_inj);
      }
      const gdebt0 = -L.st(gov, 'bonds');
      if (P.fiscal === 'debt') {
        this.tau = clip(this.tau + P.k_debt * (gdebt0 / Y - P.b_target), 0.0, 0.6);
      } else if (P.fiscal === 'anchor') {
        const b = gdebt0 / Y;
        if (this.t < P.anchor_start) {
          this.b_ref = b;
          this.tau = clip(this.tau + P.k_tau * (deficit / Y - P.deficit_target), 0.0, 0.6);
        } else {
          this.tau = clip(this.tau + P.k_debt * (b - this.b_ref), 0.0, 0.6);
        }
      } else if (P.fiscal === 'burden') {
        const allowed = P.burden_target - rB * gdebt0 / Y;
        this.tau = clip(this.tau + P.k_tau * (deficit / Y - allowed), 0.0, 0.6);
      } else if (P.fiscal === 'real') {
        const d_rule = deficit - (Math.exp(infl) - 1) * (gdebt0 - deficit);
        this.tau = clip(this.tau + P.k_tau * (d_rule / Y - P.deficit_target), 0.0, 0.6);
      } else {
        const d_rule = P.fiscal === 'deficit' ? deficit : deficit - ben;
        this.tau = clip(this.tau + P.k_tau * (d_rule / Y - P.deficit_target), 0.0, 0.6);
      }
      this.rho_prev = F64(rho);
      this.u_prev = F64(u);
      this.profit_prev = profit[0] + profit[1] + profit[2];
      this.PN_prev = bN;
      this.prices = F64(price);
      this.pbuy = F64(pbuy);
      const dist = this.distStats(Yd);
      const rbar_used = this.rbar, prem_used = this.prem, rL_used = this.r_loan;
      this.top10_prev = dist.top10_inc;
      let gdebt = -L.st(gov, 'bonds');
      let defaulted = 0.0;
      if (P.resolve === 'default' && gdebt / Y > P.theta_resolve && !this.in_default) {
        const cut = P.haircut * L.st(bank, 'bonds');
        L.revalue('bonds', bank, -cut); L.revalue('bonds', gov, +cut);
        const dep = zeros(n);
        for (let i = 0; i < n; i++) dep[i] = Math.max(H.M[i], 0.0);
        const ds = sum(dep);
        const take = Math.min(cut, ds);
        for (let i = 0; i < n; i++) H.M[i] = H.M[i] - take * dep[i] / Math.max(ds, 1e-12);
        L.revalue('deposits', hh, -take); L.revalue('deposits', bank, +take);
        if (cut > take) {
          for (let i = 0; i < n; i++) H.M[i] = H.M[i] - (cut - take) * H.eq[i];
          L.revalue('deposits', hh, -(cut - take)); L.revalue('deposits', bank, +(cut - take));
        }
        defaulted = cut;
        this.in_default = true;
        gdebt = -L.st(gov, 'bonds');
      }
      if (gdebt / Y < 0.8 * P.theta_resolve) this.in_default = false;
      this.b_last = gdebt / Y;
      this.v_eq_last = Math.max(div, 0.0) / Math.max(this.rbar, 1e-3);
      this.setRates();
      const sample_gap = Math.max(Math.abs(sum(H.M) + this.pfDeposits() - L.st(hh, 'deposits')), Math.abs(sum(H.F) - L.st(hh, 'foreign')));
      let rh = 0.0;
      for (let i = 0; i < n; i++) if (ret[i]) rh += H.heads[i];
      // 段 B7: 格上げの工房(ライトの法則で費用が下がり、古い企業より安ければ割合が広がる)
      let up_adv = 0.0, upgraded = 0.0;
      if (P.up_gen > 0 && this.up_s > 0) {
        this.up_cum += this.up_s * x[1];
        const gX = Math.max(P.up_g0 * Math.pow(this.up_cum / P.up_cum0, -P.up_wright), P.up_floor);
        up_adv = Math.log(this.gamma[1] / (gX * (1 - P.up_sub)));
        const s_new = Math.min(Math.max(this.up_s + P.up_speed * this.up_s * (1 - this.up_s) * up_adv, 0.0), 0.99);
        if (this.up_peak < 0.5 && s_new >= 0.5) upgraded = 1.0;
        this.up_peak = Math.max(this.up_peak, s_new);
        this.up_s = s_new >= 1e-4 ? s_new : 0.0;
        this.up_gX = gX;
      }
      // 段 B7: 革命(移り変わりの痛みが積もった不満が閾値を超えると、政策の組が入れ替わる)
      let revolution = 0.0;
      if (P.revolt) {
        const brk = land_rec.breakups || 0.0;
        if (this.rev_prev !== null) {
          const [E0, w0, h0] = this.rev_prev;
          const pain = P.rev_wE * Math.max(0.0, 1 - E / E0) + P.rev_ww * Math.max(0.0, -Math.log((W / bN) / w0))
            + P.rev_wh * Math.max(0.0, hungry - h0) + P.rev_wb * brk;
          this.rev_G = P.rev_mem * this.rev_G + pain;
        }
        this.rev_prev = [E, W / bN, hungry];
        if (this.rev_G > P.rev_thr && this.t - this.rev_last >= P.rev_cool) {
          const alt = this.rev_alt !== null ? this.rev_alt : { tau_k: P.rev_tau_k, redistrib: P.rev_redistrib, wealth_tax: P.rev_wealth_tax };
          this.rev_alt = { tau_k: P.tau_k, redistrib: P.redistrib, wealth_tax: P.wealth_tax };
          Object.assign(P, alt);
          this.regime = 1 - this.regime;
          this.rev_last = this.t;
          this.rev_G = 0.0;
          revolution = 1.0;
        }
      }
      const rec = Object.assign({
        t: this.t, W, PN, PL, PI, bN, bL, bI, real_wage: W / bN, S, D: Dl, E, unemp,
        xN: x[0], xL: x[1], xI: x[2], uN: u[0], uL: u[1], uI: u[2], KN: K_old[0], KL: K_old[1], KI: K_old[2],
        rhoN: rho[0], rhoL: rho[1], rhoI: rho[2], tau: this.tau, M_w: H.M[0], M_c: H.M[1], deficit,
        G: -L.st(gov, 'bonds'), Y, resid, sample_gap, rbar: rbar_used, infl, r_policy: rD, prem: prem_used, r_loan: rL_used,
        loans: Lf[0] + Lf[1] + Lf[2], dL: dL[0] + dL[1] + dL[2], bank_profit, div, tax_c, transfer: tr,
        inv: inv_cost_j[0] + inv_cost_j[1] + inv_cost_j[2], rent, roy,
      }, dist, land_rec, {
        prot: this.prot, prot_L, inj_share: dot(H.lab, H.inj) / sum(H.lab),
        dis_size: this.dis_rec.dis_size || 0.0, relief: this.dis_rec.relief || 0.0, K_lost: this.dis_rec.K_lost || 0.0,
        b: this.b_last, premB: this.premB, r_bond: rB, levy, crisis: crisis ? 1.0 : 0.0, defaulted,
        prem_income: (rB - rD) * B_bank, belief: this.belief || 0.0, protect: P.protect, tau_k: P.tau_k,
        L_R: L_R[0] + L_R[1] + L_R[2], res_w: rwsum, gain: gains[0] + gains[1] + gains[2], sR: (this.sR[0] + this.sR[1] + this.sR[2]) / 3,
        gamma_front: (this.gamma_new[0] + this.gamma_new[1] + this.gamma_new[2]) / 3, pub_L,
        wage_share: Y ? (wsum + rwsum) / Y : NaN, kap_avg: (this.kap_avg[0] + this.kap_avg[1] + this.kap_avg[2]) / 3,
        acct_use: bN * qc[0], acct_lux: cons[1], acct_save: sum(Yd) - sum(spend),
        acct_explore: P.land ? P.sigma_R * cons[1] + rwsum : rwsum,
        cap_Y: Y ? bI * dot(this.kap_avg, this.K) / Y : NaN,
      });
      if (!(P.ret_age === 0 && P.child_allow === 0)) Object.assign(rec, { ret_share: rh / hs, pensions: pen_s, child_allow: ca_s, vat: 0.0, pf: sum(H.pf) });
      if (P.psi_v !== 0) Object.assign(rec, { n_var: this.variety(), b_share: this.bShare(), lux_eff: qc[1] * Math.pow(this.variety(), 1 / (P.sigma_v - 1)) });
      if (cx !== null) Object.assign(rec, { short: cx.short / Math.max(cx.dN, 1e-12), hungry, sub_gap, hungry_kids, hungry_ret, capped: cx.capped ? 1.0 : 0.0,
                                            InvN: this.InvN || 0.0, S_G: this.S_G, subsidy, share_pay: share_pay[0] + share_pay[1] + share_pay[2],
                                            gov_buy, scrap: scrap[0] + scrap[1] + scrap[2], hhN: qc[0], xN_prod: x[0] });
      if (this.vote_rec) Object.assign(rec, this.vote_rec);
      if (P.up_gen > 0) Object.assign(rec, { up_s: this.up_s, up_gX: this.up_gX, up_adv, upgraded, up_pay });
      if (P.revolt) Object.assign(rec, { rev_G: this.rev_G, revolution, regime: this.regime });
      if (this.pl !== null) {
        const k = this.pl;
        Object.assign(rec, this.playerState(), { pl_hungry: cx === null ? (H.Yd[k] < 0 ? 1.0 : 0.0) : (got[k] < shq0[k] * P.q0 * (1 - 1e-6) ? 1.0 : 0.0) }, this.vote_pl);
      }
      this.vote_pl = {};
      this.t += 1;
      return rec;
    }

    distStats(Yd) {
      const H = this.H, n = H.M.length;
      const pc = zeros(n), nw = zeros(n);
      for (let i = 0; i < n; i++) { pc[i] = Yd[i] / H.heads[i]; nw[i] = H.M[i] / H.heads[i]; }
      return { gini_inc: giniW(pc, H.heads), top10_inc: topShareW(pc, H.heads), gini_dep: giniW(nw, H.heads) };
    }

    // ------------------------------------------------ 状態の写し(時間のつまみ)
    // 年の境(step のあと)の状態を、構造化複製できる素の値にする。帳簿は、模型が一国のときだけ含める(世界では World が持つ)
    getState(withLedger = true) {
      const s = {};
      for (const k of Object.keys(this)) {
        if (k === 'L' || k === 'H' || k === 'log' || k === 'route' || k === 'xf' || RNGS.indexOf(k) >= 0) continue;
        s[k] = deepCopy(this[k]);
      }
      s.H = {};
      for (const k of H_FIELDS.concat(['propensity', 'cw', 'cc'])) s.H[k] = deepCopy(this.H[k]);
      s.rngs = {};
      for (const k of RNGS) s.rngs[k] = this[k].getState();
      if (withLedger) s.L = this.L.snapshot();
      return s;
    }
    setState(s, ledger) {
      for (const k of Object.keys(s)) {
        if (k === 'H' || k === 'L' || k === 'rngs') continue;
        this[k] = deepCopy(s[k]);
      }
      for (const k of Object.keys(s.H)) this.H[k] = deepCopy(s.H[k]);
      for (const k of RNGS) this[k].setState(s.rngs[k]);
      this.L = ledger || IM.Ledger.restore(s.L);
      this.xf = {};
      this.log = [];
    }
  }
  const RNGS = ['rng', 'dis_rng', 'res_rng', 'luck_rng', 'vote_rng'];

  function deepCopy(v) {
    if (v === null || typeof v !== 'object') return v;
    if (ArrayBuffer.isView(v)) return v.slice();
    if (Array.isArray(v)) return v.map(deepCopy);
    const o = {};
    for (const k of Object.keys(v)) o[k] = deepCopy(v[k]);
    return o;
  }

  function simulate(T = 200, shocks = [], opts = {}) {
    const e = new Im1(opts);
    for (let t = 0; t < T; t++) {
      for (const [ts, ch] of shocks) if (ts === t) Object.assign(e.p, ch);
      e.step();
    }
    return e;
  }

  Object.assign(IM, { SECT, FIRMS, SECTORS, INSTRUMENTS, REAL, softplus, softmin, logaddexp0, defaultParams, checkSupported,
                      Households, makeHouseholds, giniW, topShareW, Im1, simulate1: simulate, deepCopy, H_FIELDS, RNGS });
})(typeof globalThis !== 'undefined' ? globalThis : this);
