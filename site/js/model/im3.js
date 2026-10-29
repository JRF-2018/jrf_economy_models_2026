// im3.js: 統合模型 im の地域と輸出入、通貨と為替(im3.py の写し。solver='portable' の道)。
// 地域は、それぞれ一国の模型(Im1)を部品として持つ。帳簿は通貨ごとに一つ。財の貿易はアーミントンの型。
// 共通の通貨(銀行の間の貸し = TARGET2)、別々の通貨と為替(固定と変動)、資本の移動、財政同盟。
(function (G) {
  'use strict';
  const IM = G.IM || (G.IM = {});
  const { softplus } = IM;

  class World {
    constructor(o = {}) {
      const regions = o.regions || ['A', 'B'];
      const n = regions.length;
      this.names = regions.slice();
      let cmap, regimes;
      const currency = o.currency || 'common';
      if (currency === 'common') {
        cmap = {}; for (const r of regions) cmap[r] = 'c0';
        regimes = { c0: 'fixed' };
      } else if (currency === 'fixed' || currency === 'float') {
        cmap = {}; regimes = {};
        for (const r of regions) { cmap[r] = r; regimes[r] = currency; }
      } else {
        cmap = Object.assign({}, currency);
        regimes = Object.assign({}, o.regimes || {});
      }
      this.cmap = cmap;
      this.curs = [];
      for (const r of regions) if (this.curs.indexOf(cmap[r]) < 0) this.curs.push(cmap[r]);
      this.cur = regions.map(r => this.curs.indexOf(cmap[r]));
      this.regime = this.curs.map((c, k) => k > 0 ? (regimes[c] || 'fixed') : 'anchor');
      this.floating = [];
      this.regime.forEach((g, k) => { if (g === 'float') this.floating.push(k); });
      this.e = o.e0 ? Float64Array.from(o.e0) : new Float64Array(this.curs.length).fill(1.0);
      this.fx_tactic = o.fx_tactic !== undefined ? o.fx_tactic : 1e-4;
      const common = Object.assign({}, o.common || {});
      const region_kw = o.region_kw || {};
      const seeds = o.seeds || Object.fromEntries(regions.map(r => [r, 0]));
      const multi = this.curs.length > 1;
      let extra = multi ? ['fx'] : [];
      this.capital = !!o.capital;
      let dest = o.dest;
      if (!dest) { dest = {}; regions.forEach((r, i) => { dest[r] = i === 0 ? regions[1] : regions[0]; }); }
      this.dest = regions.map(r => this.names.indexOf(dest[r]));
      if (this.capital) extra = extra.concat(multi ? ['ext', 'nr'] : ['ext']);
      this.union = o.union || null;
      this.kappa_q = o.kappa_q !== undefined ? o.kappa_q : 0.5;
      // 通貨ごとの帳簿
      this.ledgers = [];
      const ledOf = {};
      for (let k = 0; k < this.curs.length; k++) {
        const group = regions.filter((r, i) => this.cur[i] === k);
        const sectors = [], bankOf = {};
        for (const r of group) {
          for (const x of IM.SECTORS.concat(extra)) { sectors.push(`${x}@${r}`); bankOf[`${x}@${r}`] = `bank@${r}`; }
          if (this.capital) {
            const d = regions[this.dest[regions.indexOf(r)]];
            if (cmap[d] === cmap[r]) bankOf[`ext@${r}`] = `bank@${d}`;
          }
        }
        const real = [];
        for (const r of group) for (const x of IM.REAL) real.push(`${x}@${r}`);
        const L = new IM.Ledger(sectors, IM.INSTRUMENTS, real, group.length > 1 ? bankOf : null);
        this.ledgers.push(L);
        for (const r of group) ledOf[r] = L;
      }
      this.extra = extra;
      this.R = [];
      regions.forEach((r, i) => {
        const kw = Object.assign({}, common, region_kw[r] || {});
        this.R.push(new IM.Im1({ households: o.households || 'continuous', Ns: o.Ns || 400, seed: seeds[r], ledger: ledOf[r],
                                 region: r, extraSectors: extra, kw }));
      });
      for (const L of this.ledgers) { L.close(); L.t = 0; }
      this.sigma = Float64Array.from(o.sigma || [8.0, 2.0, 4.0]);
      const tc = o.tc !== undefined ? o.tc : 0.1, home = o.home !== undefined ? o.home : 0.5;
      this.tau_trade = []; this.w = [];
      for (let s = 0; s < n; s++) {
        this.tau_trade.push(regions.map((_, r) => r === s ? 1.0 : 1.0 + tc));
        this.w.push(regions.map((_, r) => r === s ? home : (1 - home) / Math.max(n - 1, 1)));
      }
      this.prices = this.R.map(e => Float64Array.from(e.prices));
      this.R.forEach((e, s) => { e.route = this._router(s); });
      this.log = [];
      this.t = 0;
      this.hold = new Float64Array(n);
      this.KF = new Float64Array(n);
      this.INT = new Float64Array(n);
      this.dF = new Array(n).fill(null);
      this.sh_del = null;
      this.crisis_now = null;
    }

    planCapital(X) {
      this.R.forEach((e, o) => {
        const d = this.dest[o];
        const fx = X[o][d] / this.X_last[o][d];
        this.dF[o] = e.planForeign(this.r_out[o], fx);
        let s = 0.0; for (const v of this.dF[o]) s += v;
        this.KF[o] = s;
      });
    }
    sameCur(o, d) { return this.cur[o] === this.cur[d]; }
    holding(o) {
      const d = this.dest[o];
      if (this.sameCur(o, d)) return this.R[o].L.st(`ext@${this.names[o]}`, 'deposits');
      return this.hold[o];
    }

    // ------------------------------------------------ 価格の指数と支出の割合
    xrate(e) {
      e = e || this.e;
      const v = this.cur.map(k => e[k]);
      return v.map(vs => v.map(vr => vr / vs));          // X[s][r] = v[r] / v[s]
    }
    buyPrices(p, e) {
      const n = this.R.length;
      const X = this.xrate(e);
      const Pb = [], sh = [];
      for (let s = 0; s < n; s++) { Pb.push([0, 0, 0]); sh.push([]); for (let r = 0; r < n; r++) sh[s].push([0, 0, 0]); }
      for (let j = 0; j < 3; j++) {
        const sg = this.sigma[j];
        for (let s = 0; s < n; s++) {
          const c = [];
          let tot = 0.0;
          for (let r = 0; r < n; r++) {
            let v = this.w[s][r] * Math.pow(p[r][j] * X[s][r] * this.tau_trade[s][r], 1 - sg);
            if (!(this.w[s][r] > 0)) v = 0.0;
            c.push(v);
            tot += v;
          }
          Pb[s][j] = Math.pow(tot, 1 / (1 - sg));
          for (let r = 0; r < n; r++) sh[s][r][j] = c[r] / tot;
        }
      }
      return { Pb, sh };
    }
    _router(s) {
      return (flow, payer, good, amount) => {
        this.R.forEach((e, r) => {
          // 危機時の生産の年: 必需品は、実際に届いた割合で払う
          const a = amount * ((good === 0 && this.sh_del) ? this.sh_del[s][r] : this.sh[s][r][good]);
          const to = e.firms[good];
          if (to === payer || a === 0) return;
          if (this.cur[r] === this.cur[s]) {
            this.R[s].L.pay(r === s ? flow : flow + ' (import)', payer, to, a);
          } else {
            this.R[s].L.pay('imports', payer, this.R[s].sec.fx, a);
            this.pending[r][good] += a / this.X[s][r];
          }
        });
      };
    }

    // ------------------------------------------------ 年の中の釣り合い
    expI(Pb, x) {
      return this.R.map((e, s) => {
        const P = e.p;
        let a = 0.0;
        for (let j = 0; j < 3; j++) a += P.alpha[j] * x[s][j];
        let b = 0.0;
        for (let j = 0; j < 3; j++) b += e.kap_new[j] * e._inv[j];
        return Pb[s][2] * (a + b);
      });
    }
    crisisOn() { return this.R.some(e => e.crisisOn()); }
    market(p, e) {
      if (this.crisisOn()) return this.marketCrisis(p, e);
      this.crisis_now = null;
      const n = this.R.length;
      const { Pb, sh } = this.buyPrices(p, e);
      const X = this.xrate(e);
      const qc = this.R.map((e_, s) => e_.demand(Pb[s], e_._inv, e_._B).slice(0, 2));
      const exp_c = qc.map((q, s) => [q[0] * Pb[s][0], q[1] * Pb[s][1]]);
      let x = this.R.map((e_, r) => e_.demand(p[r], e_._inv, e_._B));
      for (let it = 0; it < 500; it++) {
        const eI = this.expI(Pb, x);
        const exp = exp_c.map((c, s) => [c[0], c[1], eI[s]]);
        const xn = [];
        let dmax = 0.0;
        for (let r = 0; r < n; r++) {
          const row = [0, 0, 0];
          for (let j = 0; j < 3; j++) {
            let acc = 0.0;
            for (let s = 0; s < n; s++) acc += sh[s][r][j] * exp[s][j] * (1.0 / X[s][r]);
            row[j] = acc / p[r][j];
            dmax = Math.max(dmax, Math.abs(row[j] - x[r][j]) / (Math.abs(x[r][j]) + 1e-12));
          }
          xn.push(row);
        }
        x = xn;
        if (dmax < 1e-15) break;
      }
      const out = this.R.map((e_, r) => {
        const Pr = e_.p;
        const u = [0, 0, 0], geff = [0, 0, 0], cost = [0, 0, 0], m = [0, 0, 0];
        const ge = e_.gammaEff();
        for (let j = 0; j < 3; j++) {
          u[j] = x[r][j] / e_.K[j];
          geff[j] = ge[j] * (1 + Pr.omega * softplus(u[j] - 1.0, 20.0));
          cost[j] = e_.W * geff[j] + Pb[r][2] * Pr.alpha[j];
          m[j] = e_.m0[j] + Pr.zeta * Math.tanh(u[j] - Pr.u_star);
        }
        if (Pr.land) cost[0] = cost[0] + Pr.rho_T * p[r][0];
        return { u, geff, cost, m };
      });
      return { Pb, sh, qc, x, out };
    }
    // 危機時の生産の年の中の釣り合い(im3.py の market_crisis の写し)
    marketCrisis(p, e) {
      const n = this.R.length, R = this.R;
      const { Pb, sh } = this.buyPrices(p, e);
      const X = this.xrate(e);
      const P = R.map(e_ => e_.p);
      const qc = R.map((e_, s) => e_.demand(Pb[s], e_._inv, e_._B).slice(0, 2));
      const capped = R.map((e_, r) => P[r].cap_N > 0 && p[r][0] >= P[r].cap_N * (1 - 1e-9));
      const gb = R.map((e_, s) => (capped[s] ? 0.0 : (e_._gbuy || 0.0)));
      const GIn = R.map(e_ => e_._GI_nom || 0.0);
      const dN = qc.map(q => q[0]);
      const D = dN.map((v, s) => v + gb[s]);
      let w = [];
      for (let s = 0; s < n; s++) { w.push([]); for (let r = 0; r < n; r++) w[s].push(sh[s][r][0]); }
      const xN = new Array(n).fill(0), fsales = new Array(n).fill(0), Inv = new Array(n).fill(0), phi = new Array(n).fill(1);
      let DR = new Array(n).fill(0);
      for (let it = 0; it < 60; it++) {
        DR = []; const DRh = [];
        for (let r = 0; r < n; r++) {
          let a = 0.0, b = 0.0;
          for (let s = 0; s < n; s++) { a += w[s][r] * Pb[s][0] * D[s] / X[s][r]; b += w[s][r] * Pb[s][0] * dN[s] / X[s][r]; }
          DR.push(a / p[r][0]); DRh.push(b / p[r][0]);
        }
        R.forEach((e_, r) => {
          const Pr = P[r];
          Inv[r] = Pr.inventory ? (e_.InvN === null ? Pr.inv_target * DRh[r] : e_.InvN) : 0.0;
          const plan = Pr.inventory ? Math.max(DR[r] + Pr.chi_inv * (Pr.inv_target * DR[r] - Inv[r]), 0.0) : DR[r];
          if (capped[r]) {
            const xs = e_.supplyAtCap(Pr.cap_N, Pb[r][2]);
            xN[r] = Math.min(xs, plan);
            if (Pr.crisis_rule === 'last' || Pr.crisis_rule === 'share') {
              let hs = 0.0; for (const v of e_.subsHeads()) hs += v;
              const need = Pr.q0 * hs;
              xN[r] = Math.max(xN[r], Math.min(plan, Math.max(need - Inv[r] - e_.S_G, 0.0)));
            }
          } else xN[r] = plan;
          fsales[r] = Math.min(DR[r], xN[r] + Inv[r]);
          phi[r] = DR[r] > 0 ? fsales[r] / DR[r] : 1.0;
        });
        const open = phi.map(v => v >= 1 - 1e-12);
        let moved = 0.0;
        const wn = w.map(row => row.slice());
        for (let s = 0; s < n; s++) {
          let unmet = 0.0, base = 0.0;
          for (let r = 0; r < n; r++) unmet += w[s][r] * (1 - phi[r]);
          for (let r = 0; r < n; r++) if (open[r]) base += sh[s][r][0];
          if (unmet <= 1e-15 || base <= 0) continue;
          for (let r = 0; r < n; r++) wn[s][r] = w[s][r] * phi[r] + (open[r] ? unmet * sh[s][r][0] / base : 0.0);
          moved = Math.max(moved, unmet);
        }
        if (moved <= 1e-15) break;
        w = wn;
      }
      const f = [];
      for (let s = 0; s < n; s++) { let a = 0.0; for (let r = 0; r < n; r++) a += w[s][r] * phi[r]; f.push(a); }
      const deliv = f.map((v, s) => v * D[s]);
      const rel = R.map((e_, s) => (deliv[s] < D[s] ? Math.min(D[s] - deliv[s], e_.S_G) : 0.0));
      const hhN = deliv.map((v, s) => v + rel[s] - gb[s]);
      this.sh_del_now = [];
      for (let s = 0; s < n; s++) { const row = []; for (let r = 0; r < n; r++) row.push(f[s] > 0 ? w[s][r] * phi[r] / f[s] : sh[s][r][0]); this.sh_del_now.push(row); }
      const exp_c = qc.map((q, s) => [Pb[s][0] * deliv[s], q[1] * Pb[s][1]]);
      let x = R.map((e_, r) => e_.demand(p[r], e_._inv, e_._B));
      for (let r = 0; r < n; r++) x[r][0] = xN[r];
      for (let it = 0; it < 500; it++) {
        const eI = this.expI(Pb, x).map((v, s) => v + GIn[s]);
        const xn = [];
        let dmax = 0.0;
        for (let r = 0; r < n; r++) {
          const row = [xN[r], 0, 0];
          for (let j = 1; j < 3; j++) {
            let acc = 0.0;
            for (let s = 0; s < n; s++) acc += sh[s][r][j] * (j === 1 ? exp_c[s][1] : eI[s]) * (1.0 / X[s][r]);
            row[j] = acc / p[r][j];
          }
          for (let j = 0; j < 3; j++) dmax = Math.max(dmax, Math.abs(row[j] - x[r][j]) / (Math.abs(x[r][j]) + 1e-12));
          xn.push(row);
        }
        x = xn;
        if (dmax < 1e-15) break;
      }
      const cxs = [];
      const out = R.map((e_, r) => {
        const Pr = e_.p;
        const u = [0, 0, 0], geff = [0, 0, 0], cost = [0, 0, 0], m = [0, 0, 0];
        const ge = e_.gammaEff();
        for (let j = 0; j < 3; j++) {
          u[j] = x[r][j] / e_.K[j];
          geff[j] = ge[j] * (1 + Pr.omega * softplus(u[j] - 1.0, 20.0));
          cost[j] = e_.W * geff[j] + Pb[r][2] * Pr.alpha[j];
          m[j] = e_.m0[j] + Pr.zeta * Math.tanh(u[j] - Pr.u_star);
        }
        if (Pr.land) cost[0] = cost[0] + Pr.rho_T * p[r][0];
        if (Pr.inventory) m[0] = m[0] + Pr.zeta_inv * Math.tanh(1.0 - (Inv[r] + xN[r] - fsales[r]) / Math.max(Pr.inv_target * DR[r], 1e-12));
        cxs.push({ dN: dN[r], D: D[r], gbuy: gb[r], xN: xN[r], firm_sales: fsales[r], rel: rel[r], hhN: hhN[r],
                   short: Math.max(dN[r] - hhN[r], 0.0), capped: capped[r], GI: GIn[r] / Pb[r][2], Inv: Inv[r], phi: phi[r], fill: f[r] });
        return { u, geff, cost, m };
      });
      this.crisis_now = { cx: cxs, deliv, gb, GIn };
      const qc2 = qc.map((q, s) => [hhN[s], q[1]]);
      return { Pb, sh, qc: qc2, x, out };
    }
    flows(Pb, sh, qc, x) {
      const n = this.R.length;
      if (this.crisis_now) {
        const C = this.crisis_now;
        const eI = this.expI(Pb, x).map((v, s) => v + C.GIn[s]);
        const f = [];
        for (let s = 0; s < n; s++) {
          f.push([]);
          for (let r = 0; r < n; r++) f[s].push([this.sh_del_now[s][r] * Pb[s][0] * C.deliv[s], sh[s][r][1] * qc[s][1] * Pb[s][1], sh[s][r][2] * eI[s]]);
        }
        return f;
      }
      const eI = this.expI(Pb, x);
      const f = [];
      for (let s = 0; s < n; s++) {
        const ex = [qc[s][0] * Pb[s][0], qc[s][1] * Pb[s][1], eI[s]];
        f.push([]);
        for (let r = 0; r < n; r++) f[s].push([0, 1, 2].map(j => sh[s][r][j] * ex[j]));
      }
      return f;
    }
    areaBalance(k, f, X) {
      const n = this.R.length;
      const inn = this.cur.map(c => c === k);
      const fs = (s, r) => f[s][r][0] + f[s][r][1] + f[s][r][2];
      let ex = 0.0, im = 0.0, spend = 0.0;
      for (let r = 0; r < n; r++) if (inn[r]) for (let s = 0; s < n; s++) if (!inn[s]) ex += fs(s, r) / X[s][r];
      for (let s = 0; s < n; s++) if (inn[s]) for (let r = 0; r < n; r++) if (!inn[r]) im += fs(s, r);
      for (let s = 0; s < n; s++) if (inn[s]) for (let r = 0; r < n; r++) spend += fs(s, r);
      let cap = 0.0;
      if (this.capital) {
        for (let o = 0; o < n; o++) {
          const d = this.dest[o];
          if (inn[o] && !inn[d]) cap += -this.KF[o] + this.INT[o] * X[o][d];
          else if (inn[d] && !inn[o]) cap += this.KF[o] * X[d][o] - this.INT[o];
        }
      }
      return (ex - im + cap) / Math.max(spend, 1e-9);
    }
    _unpack(z) {
      const n = this.R.length;
      const p = [];
      for (let r = 0; r < n; r++) p.push([Math.exp(z[3 * r]), Math.exp(z[3 * r + 1]), Math.exp(z[3 * r + 2])]);
      const e = Float64Array.from(this.e);
      this.floating.forEach((k, i) => { e[k] = Math.exp(z[3 * n + i]); });
      return { p, e };
    }
    residual(z) {
      const n = this.R.length;
      const { p, e } = this._unpack(z);
      const mk = this.market(p, e);
      const res = [];
      for (let r = 0; r < n; r++) {
        const { cost, m } = mk.out[r];
        const cap = this.R[r].p.cap_N;
        for (let j = 0; j < 3; j++) {
          let target = (1 + m[j]) * cost[j];
          if (j === 0 && cap > 0) target = Math.min(target, cap);   // 価格の上限(段 7)
          res.push(Math.log(p[r][j]) - Math.log(target));
        }
      }
      if (this.floating.length) {
        const X = this.xrate(e);
        if (this.capital) this.planCapital(X);
        const f = this.flows(mk.Pb, mk.sh, mk.qc, mk.x);
        for (const k of this.floating) res.push(10.0 * this.areaBalance(k, f, X));
      }
      const tac = Math.sqrt(this.R[0].p.tactic);
      for (let r = 0; r < n; r++) for (let j = 0; j < 3; j++) res.push(tac * (z[3 * r + j] - Math.log(this.prices[r][j])));
      const ft = Math.sqrt(this.fx_tactic);
      this.floating.forEach((k, i) => { res.push(ft * (z[3 * n + i] - Math.log(this.e[k]))); });
      return res;
    }

    // ------------------------------------------------ 一年
    step() {
      const n = this.R.length;
      for (const e of this.R) e.beginYear();
      if (this.capital) {
        const X0 = this.xrate();
        this.X_last = X0;
        this.r_out = new Float64Array(n);
        this.R.forEach((e, o) => {
          const d = this.dest[o];
          const r_d = this.R[d].p.credit ? this.R[d].r_policy : 0.0;
          const q = X0[o][d] * this.R[d].PN_prev / e.PN_prev;
          this.r_out[o] = r_d - this.kappa_q * Math.log(Math.max(q, 1e-12));
          this.INT[o] = r_d * Math.max(this.holding(o), 0.0);
        });
        this.planCapital(X0);
      }
      const ub = new Float64Array(n), uin = new Float64Array(n);
      if (this.union) {
        const mem = this.union.members.map(m => this.names.indexOf(m));
        if (new Set(mem.map(m => this.cur[m])).size !== 1) throw new Error('財政同盟は同じ通貨の地域だけ');
        if (this.lastRecs) {
          let w_ref = 0.0; for (const m of mem) w_ref += this.R[m].W; w_ref /= mem.length;
          const U = mem.map(m => this.lastRecs[m].unemp);
          let Um = 0.0; for (const u of U) Um += u; Um /= U.length;
          mem.forEach((m, i) => {
            const net = this.union.psi * w_ref * (U[i] - Um);
            uin[m] = net;
            ub[m] = Math.max(net, 0.0);
          });
          let possum = 0.0;
          for (let i = 0; i < n; i++) if (uin[i] > 0) possum += uin[i];
          const L = this.R[mem[0]].L;
          for (const a of mem) if (uin[a] < 0) for (const b of mem) if (uin[b] > 0) {
            L.pay('union transfer', `gov@${this.names[a]}`, `gov@${this.names[b]}`, -uin[a] * uin[b] / possum);
          }
        }
      }
      const z0 = [];
      for (let r = 0; r < n; r++) for (let j = 0; j < 3; j++) z0.push(Math.log(this.prices[r][j]));
      for (const k of this.floating) z0.push(Math.log(this.e[k]));
      let sol = IM.lmSolve(z => this.residual(z), z0, 1e-13, 0.0);
      // 価格の上限が効く地域: 必需品の価格を上限にそろえ、残りを解き直す(im3.py と同じ)
      const fix = [];
      this.R.forEach((e, r) => { if (e.p.cap_N > 0 && Math.exp(sol.x[3 * r]) >= e.p.cap_N * (1 - 1e-6)) fix.push(3 * r); });
      if (fix.length) {
        const nz = sol.x.length, nf = this.floating.length;
        const free = [];
        for (let k = 0; k < nz; k++) if (fix.indexOf(k) < 0) free.push(k);
        const drop = new Set(fix.concat(fix.map(k => 3 * n + nf + k)));
        const full = y => { const z = sol.x.slice(); for (const k of fix) z[k] = Math.log(this.R[Math.floor(k / 3)].p.cap_N); free.forEach((k, q) => { z[k] = y[q]; }); return z; };
        const subRes = y => this.residual(full(y)).filter((v, k) => !drop.has(k));
        const sub = IM.lmSolve(subRes, free.map(k => sol.x[k]), 1e-13, 0.0);
        const zf = full(sub.x);
        sol = { x: zf, fun: this.residual(zf), nfev: sol.nfev + sub.nfev, nit: sol.nit + sub.nit };
      }
      const up = this._unpack(sol.x);
      const p = up.p;
      this.e = up.e;
      const mk = this.market(p);
      const { Pb, sh, qc, x, out } = mk;
      this.sh = sh;
      const C = this.crisis_now;
      this.sh_del = C ? this.sh_del_now : null;
      this.R.forEach((e, r) => { e._cx = C ? C.cx[r] : null; });       // 危機時の生産の年: 地域ごとの結果を finishYear に
      this.X = this.xrate();
      this.pending = this.R.map(() => [0.0, 0.0, 0.0]);
      const X = this.X;
      if (this.capital) {
        this.planCapital(X);
        this.R.forEach((e, o) => {
          const d = this.dest[o];
          if (this.sameCur(o, d)) return;
          const no = this.names[o];
          const newV = this.hold[o] * X[o][d];
          const g = newV - e.L.st(`hh@${no}`, 'foreign');
          e.L.revalue('foreign', `hh@${no}`, g);
          e.L.revalue('foreign', `ext@${no}`, -g);
          let Fs = 0.0; for (const v of e.H.F) Fs += v;
          if (Fs > 0) for (let i = 0; i < e.H.F.length; i++) e.H.F[i] = e.H.F[i] * (newV / Fs);
        });
      }
      this.R.forEach((e, r) => {
        e.xf = { union_ben: ub[r], union_in: uin[r] };
        if (this.capital) {
          const d = this.dest[r];
          e.xf.dF = this.dF[r];
          e.xf.incF = this.INT[r] * X[r][d];
        }
      });
      if (this.capital) {
        for (let d = 0; d < n; d++) {
          let s = 0.0;
          for (let o = 0; o < n; o++) if (this.dest[o] === d) s += this.INT[o];
          this.R[d].xf.int_nr = s;
        }
      }
      const recs = this.R.map((e, r) => {
        const { u, geff, cost } = out[r];
        const f3 = sol.fun.slice(3 * r, 3 * r + 3);
        const resid = Math.sqrt(f3[0] * f3[0] + f3[1] * f3[1] + f3[2] * f3[2]);
        return e.finishYear(p[r], Pb[r], x[r], u, geff, cost, qc[r], resid);
      });
      this.prices = p.map(v => Float64Array.from(v));
      if (this.curs.length > 1) {
        this.R.forEach((e, r) => { for (let j = 0; j < 3; j++) e.L.pay('exports', e.sec.fx, e.firms[j], this.pending[r][j]); });
      }
      if (this.capital) {
        this.R.forEach((e, o) => {
          const d = this.dest[o];
          const no = this.names[o], nd = this.names[d];
          const Ld = this.R[d].L;
          if (this.sameCur(o, d)) {
            Ld.pay('deposit interest (non-residents)', `bank@${nd}`, `ext@${no}`, this.INT[o]);
            return;
          }
          const Lo = e.L;
          Lo.pay('capital outflow', `ext@${no}`, `fx@${no}`, this.KF[o]);
          Ld.pay('capital inflow', `fx@${nd}`, `nr@${nd}`, this.KF[o] / X[o][d]);
          this.hold[o] += this.KF[o] / X[o][d];
          Ld.pay('deposit interest (non-residents)', `bank@${nd}`, `nr@${nd}`, this.INT[o]);
          Ld.pay('income outflow', `nr@${nd}`, `fx@${nd}`, this.INT[o]);
          Lo.pay('income inflow', `fx@${no}`, `ext@${no}`, this.INT[o] * X[o][d]);
          const hh = `hh@${no}`;
          const newV = this.hold[o] * X[o][d];
          const g = newV - Lo.st(hh, 'foreign');
          Lo.revalue('foreign', hh, g);
          Lo.revalue('foreign', `ext@${no}`, -g);
          const H = e.H;
          let Fs = 0.0; for (const v of H.F) Fs += v;
          if (Fs > 0) for (let i = 0; i < H.F.length; i++) H.F[i] = H.F[i] * (newV / Fs);
        });
      }
      let books = 0.0, tolb = 0.0;
      for (const L of this.ledgers) {
        let gb = 0.0;
        for (const s of L.sectors) if (s.startsWith('gov@')) gb = Math.max(gb, Math.abs(L.st(s, 'bonds')));
        const tol = 1e-6 * Math.max(1.0, gb);
        const err = L.check().max;
        if (err / tol > books / Math.max(tolb, 1e-300) || tolb === 0) { books = err; tolb = tol; }
      }
      // 貿易の収支(産地の通貨)と、二地域の間の収支
      const f = this.flows(Pb, sh, qc, x);
      this.lastFlows = f;                   // 画面の流れの弧(買い手 s、産地 r、財 j。買い手の通貨)
      const e2 = this.curs.length > 1 ? this.e[1] / this.e[0] : 1.0;
      const fsum = (s, r) => f[s][r][0] + f[s][r][1] + f[s][r][2];
      recs.forEach((rec, r) => {
        const exp_r = [0, 1, 2].map(j => { let a = 0.0; for (let s = 0; s < n; s++) if (s !== r) a += f[s][r][j] / X[s][r]; return a; });
        let imports = 0.0;
        for (let q = 0; q < n; q++) imports += fsum(r, q);
        imports -= fsum(r, r);
        const nm = this.names[r];
        const L = this.R[r].L;
        const ib = L.instruments.indexOf('interbank') >= 0 ? L.st(`bank@${nm}`, 'interbank') : 0.0;
        const fxpos = this.curs.length > 1 ? -L.st(`fx@${nm}`, 'deposits') : 0.0;
        for (let q = 0; q < n; q++) if (q !== r) rec[`bil_${this.names[q]}`] = fsum(q, r) / X[q][r] - fsum(r, q);
        if (this.capital) {
          let Fs = 0.0; for (const v of this.R[r].H.F) Fs += v;
          Object.assign(rec, { F: Fs, F_Y: Fs / rec.Y, KF: this.KF[r], KF_Y: this.KF[r] / rec.Y,
                               incF_Y: this.INT[r] * X[r][this.dest[r]] / rec.Y, F_gap: Math.abs(Fs - L.st(`hh@${nm}`, 'foreign')) });
        }
        const ex = exp_r[0] + exp_r[1] + exp_r[2];
        Object.assign(rec, { union_in: uin[r], union_ben: ub[r], exports: ex, imports, tb: ex - imports, tb_Y: (ex - imports) / rec.Y,
                             expN: exp_r[0], expL: exp_r[1], expI: exp_r[2], ib, fxpos, interbank: ib + fxpos, books, books_tol: tolb, e: e2,
                             e_cur: this.e[this.cur[r]] });
      });
      for (const L of this.ledgers) L.close();
      this.lastRecs = recs;
      this.log.push(recs);
      this.t += 1;
      return recs;
    }

    // ------------------------------------------------ 状態の写し(時間のつまみの鍵の年)
    getState() {
      return {
        t: this.t, e: Array.from(this.e), prices: this.prices.map(v => Array.from(v)), hold: Array.from(this.hold),
        KF: Array.from(this.KF), INT: Array.from(this.INT), lastRecs: this.lastRecs ? IM.deepCopy(this.lastRecs) : null,
        ledgers: this.ledgers.map(L => L.snapshot()), R: this.R.map(e => e.getState(false)),
      };
    }
    setState(s) {
      this.t = s.t;
      this.e = Float64Array.from(s.e);
      this.prices = s.prices.map(v => Float64Array.from(v));
      this.hold = Float64Array.from(s.hold);
      this.KF = Float64Array.from(s.KF);
      this.INT = Float64Array.from(s.INT);
      this.lastRecs = s.lastRecs ? IM.deepCopy(s.lastRecs) : null;
      this.ledgers = s.ledgers.map(o => IM.Ledger.restore(o));
      const ledOf = {};
      this.ledgers.forEach((L, k) => { this.names.forEach((r, i) => { if (this.cur[i] === k) ledOf[r] = L; }); });
      this.R.forEach((e, i) => { e.setState(s.R[i], ledOf[this.names[i]]); });
      this.dF = new Array(this.R.length).fill(null);
      this.log = [];
    }
  }

  function simulateWorld(T = 300, events = [], opts = {}) {
    const w = new World(opts);
    for (let t = 0; t < T; t++) {
      for (const [ts, reg, ch] of events) if (ts === t) w.R.forEach((e, i) => { if (reg === null || reg === w.names[i]) Object.assign(e.p, ch); });
      w.step();
    }
    return w;
  }
  Object.assign(IM, { World, simulateWorld });
})(typeof globalThis !== 'undefined' ? globalThis : this);
