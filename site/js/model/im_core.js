// im_core.js: 統合模型 im の骨組み(im_core.py の写し): 帳簿(部門の貸借の表と取引の流れの表)。
// お金は銀行の預金に統一する。すべての支払いは、払う側の預金を減らし、受け取る側の預金を増やす。
(function (G) {
  'use strict';
  const IM = G.IM || (G.IM = {});

  // 二重の表(名前 → 部門 → 額)。入れた順を保つ(Python の defaultdict と同じ順で足す)
  class Table2 {
    constructor() { this.m = new Map(); }
    row(name) {
      let r = this.m.get(name);
      if (!r) { r = new Map(); this.m.set(name, r); }
      return r;
    }
    add(name, sector, v) {
      const r = this.row(name);
      r.set(sector, (r.get(sector) || 0.0) + v);
    }
    get(name, sector) {
      const r = this.m.get(name);
      return r ? (r.get(sector) || 0.0) : 0.0;
    }
  }

  class Ledger {
    // 部門の貸借の表(ストック)と、取引の流れの表(一期のフロー)。ストックは持ち主から見た符号(資産は正、負債は負)。
    // bankOf を与えると(部門 → その部門が預金を持つ銀行)、銀行をまたぐ支払いを銀行の間の貸し('interbank')で決済する。
    constructor(sectors, instruments, real = [], bankOf = null) {
      this.sectors = Array.from(sectors);
      this.bankOf = bankOf ? Object.assign({}, bankOf) : null;
      let ins = Array.from(instruments);
      if (this.bankOf && ins.indexOf('interbank') < 0) ins.push('interbank');
      for (const r of real) if (ins.indexOf(r) < 0) ins.push(r);
      this.instruments = ins;
      this.real = new Set(real);
      this.stock = new Map();
      for (const s of this.sectors) for (const i of this.instruments) this.stock.set(s + '|' + i, 0.0);
      this.t = 0;
      this._begin();
    }
    _begin() {
      this.flows = new Table2();
      this.fin = new Table2();
      this.reval = new Table2();
      this.open = new Map(this.stock);
    }
    st(s, i) { return this.stock.get(s + '|' + i); }
    _add(s, i, v) { const k = s + '|' + i; this.stock.set(k, this.stock.get(k) + v); }
    pay(name, frm, to, amount) {
      if (amount === 0) return;
      this.flows.add(name, frm, -amount);
      this.flows.add(name, to, +amount);
      this._move('deposits', frm, -amount);
      this._move('deposits', to, +amount);
      this._settle(frm, to, amount);
    }
    trade(instrument, buyer, seller, amount) {
      if (amount === 0) return;
      this._move(instrument, buyer, +amount);
      this._move(instrument, seller, -amount);
      this._move('deposits', buyer, -amount);
      this._move('deposits', seller, +amount);
      this._settle(buyer, seller, amount);
    }
    _settle(frm, to, amount) {
      if (!this.bankOf) return;
      const bf = this.bankOf[frm], bt = this.bankOf[to];
      if (bf === bt) return;
      this._move('deposits', bf, +amount);
      this._move('deposits', bt, -amount);
      this._move('interbank', bf, -amount);
      this._move('interbank', bt, +amount);
    }
    produceReal(instrument, sector, amount) {
      if (!this.real.has(instrument)) throw new Error('実物の資産でない: ' + instrument);
      this._add(sector, instrument, amount);
      this.fin.add(instrument, sector, amount);
    }
    revalue(instrument, sector, amount) {
      this._add(sector, instrument, amount);
      this.reval.add(instrument, sector, amount);
    }
    _move(instrument, sector, amount) {
      this._add(sector, instrument, amount);
      this.fin.add(instrument, sector, amount);
    }
    // 帳簿の恒等式を確かめ、最大のずれを返す(投げない。画面の「帳簿の灯」に出す)
    check() {
      const err = {};
      let fr = 0.0;
      for (const row of this.flows.m.values()) {
        let s = 0.0;
        for (const v of row.values()) s += v;
        fr = Math.max(fr, Math.abs(s));
      }
      err.flow_rows = fr;
      const finInstr = this.instruments.filter(i => !this.real.has(i));
      let sc = 0.0;
      for (const i of finInstr) {
        let s = 0.0;
        for (const sec of this.sectors) s += this.st(sec, i);
        sc = Math.max(sc, Math.abs(s));
      }
      err.stock_cols = sc;
      let sf = 0.0;
      for (const s of this.sectors) for (const i of this.instruments) {
        const k = s + '|' + i;
        sf = Math.max(sf, Math.abs(this.stock.get(k) - this.open.get(k) - this.fin.get(i, s) - this.reval.get(i, s)));
      }
      err.stock_flow = sf;
      let bud = 0.0;
      for (const s of this.sectors) {
        let income = 0.0;
        for (const row of this.flows.m.values()) income += (row.get(s) || 0.0);
        let nafa = 0.0;
        for (const i of finInstr) nafa += this.fin.get(i, s);
        bud = Math.max(bud, Math.abs(income - nafa));
      }
      err.budget = bud;
      if (this.bankOf) {
        let gap = 0.0;
        const banks = Array.from(new Set(Object.values(this.bankOf)));
        for (const b of banks) {
          let held = 0.0;
          for (const s of this.sectors) if (this.bankOf[s] === b && s !== b) held += this.st(s, 'deposits');
          gap = Math.max(gap, Math.abs(held + this.st(b, 'deposits')));
        }
        err.per_bank = gap;
      }
      err.max = Math.max(...Object.values(err));
      return err;
    }
    close() { this.t += 1; this._begin(); }
    income(sector) {
      let s = 0.0;
      for (const row of this.flows.m.values()) s += (row.get(sector) || 0.0);
      return s;
    }
    // 状態の写し(時間のつまみの鍵の年)。期の途中では使わない(期首に呼ぶ)
    snapshot() {
      return { sectors: this.sectors, instruments: this.instruments, real: Array.from(this.real), bankOf: this.bankOf,
               stock: Array.from(this.stock.entries()), t: this.t };
    }
    static restore(o) {
      const L = new Ledger(o.sectors, o.instruments, o.real, o.bankOf);
      L.stock = new Map(o.stock);
      L.t = o.t;
      L._begin();
      return L;
    }
  }
  IM.Ledger = Ledger;
})(typeof globalThis !== 'undefined' ? globalThis : this);
