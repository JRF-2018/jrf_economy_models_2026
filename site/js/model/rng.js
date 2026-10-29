// rng.js: 持ち運べる乱数(im の portable.py の PortableRNG の写し)。xoshiro128** と、逆関数法・Box–Muller の引き方。
// 手順を変えるときは、portable.py も同じに直す。
(function (G) {
  'use strict';
  const IM = G.IM || (G.IM = {});

  function splitmix32(x) {
    x = (x + 0x9E3779B9) >>> 0;
    let z = x;
    z = Math.imul(z ^ (z >>> 16), 0x85EBCA6B) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xC2B2AE35) >>> 0;
    z = (z ^ (z >>> 16)) >>> 0;
    return [x, z];
  }
  const rotl = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;

  class PortableRNG {
    constructor(seed = 0) {
      let x = seed >>> 0;
      const s = [];
      for (let i = 0; i < 4; i++) {
        const o = splitmix32(x);
        x = o[0];
        s.push(o[1]);
      }
      if (!(s[0] | s[1] | s[2] | s[3])) s[0] = 1;
      this.s = s;
    }
    getState() { return this.s.slice(); }
    setState(s) { this.s = s.slice(); }
    next32() {
      let [s0, s1, s2, s3] = this.s;
      const result = Math.imul(rotl(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
      const t = (s1 << 9) >>> 0;
      s2 = (s2 ^ s0) >>> 0;
      s3 = (s3 ^ s1) >>> 0;
      s1 = (s1 ^ s2) >>> 0;
      s0 = (s0 ^ s3) >>> 0;
      s2 = (s2 ^ t) >>> 0;
      s3 = rotl(s3, 11);
      this.s = [s0, s1, s2, s3];
      return result;
    }
    u() {
      const a = this.next32() >>> 5, b = this.next32() >>> 6;
      return (a * 67108864.0 + b) / 9007199254740992.0;
    }
    many(f, n) {
      if (n === undefined || n === null) return f();
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = f();
      return out;
    }
    random(n) { return this.many(() => this.u(), n); }
    integers(high) { return Math.floor(this.u() * high); }
    normal() {
      const u1 = this.u(), u2 = this.u();
      return Math.sqrt(-2.0 * Math.log(1.0 - u1)) * Math.cos(2.0 * Math.PI * u2);
    }
    standard_normal(n) { return this.many(() => this.normal(), n); }
    exponential(scale, n) { return this.many(() => -Math.log(1.0 - this.u()) * scale, n); }
    pareto(a, n) { return this.many(() => Math.pow(1.0 - this.u(), -1.0 / a) - 1.0, n); }
    standard_t(nu, n) {
      const k = Math.round(nu);
      if (Math.abs(nu - k) > 1e-12 || k < 1) throw new Error('持ち運べる t 分布は、整数の自由度だけ');
      return this.many(() => {
        const z = this.normal();
        let c = 0.0;
        for (let i = 0; i < k; i++) { const v = this.normal(); c += v * v; }
        return z / Math.sqrt(c / k);
      }, n);
    }
    poisson(lam) {
      if (lam <= 0) return 0;
      if (lam < 30) {
        const L = Math.exp(-lam);
        let k = 0, p = 1.0;
        for (;;) {
          p *= this.u();
          if (p <= L) return k;
          k += 1;
        }
      }
      return Math.max(Math.floor(lam + Math.sqrt(lam) * this.normal() + 0.5), 0);
    }
    permutation(n) {
      const a = [];
      for (let i = 0; i < n; i++) a.push(i);
      for (let i = n - 1; i > 0; i--) {
        const j = this.integers(i + 1);
        const t = a[i]; a[i] = a[j]; a[j] = t;
      }
      return a;
    }
    choice(n, p) {
      if (!p) return this.integers(n);
      const u = this.u();
      let c = 0.0;
      for (let i = 0; i < n; i++) {
        c += p[i];
        if (u < c) return i;
      }
      return n - 1;
    }
  }
  IM.PortableRNG = PortableRNG;
})(typeof globalThis !== 'undefined' ? globalThis : this);
