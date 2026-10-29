// solver.js: Levenberg–Marquardt(im の portable.py の lm_solve の写し。手順と演算の順を同じに書く)。
(function (G) {
  'use strict';
  const IM = G.IM || (G.IM = {});

  function solveLinear(A, b) {
    const n = b.length;
    for (let k = 0; k < n; k++) {
      let piv = k, best = Math.abs(A[k][k]);
      for (let i = k + 1; i < n; i++) {
        if (Math.abs(A[i][k]) > best) { piv = i; best = Math.abs(A[i][k]); }
      }
      if (piv !== k) {
        let t = A[k]; A[k] = A[piv]; A[piv] = t;
        t = b[k]; b[k] = b[piv]; b[piv] = t;
      }
      let d = A[k][k];
      if (d === 0.0) d = 1e-300;
      for (let i = k + 1; i < n; i++) {
        const f = A[i][k] / d;
        if (f !== 0.0) {
          const Ai = A[i], Ak = A[k];
          for (let j = k; j < n; j++) Ai[j] -= f * Ak[j];
          b[i] -= f * b[k];
        }
      }
    }
    const x = new Array(n).fill(0.0);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i];
      const Ai = A[i];
      for (let j = i + 1; j < n; j++) s -= Ai[j] * x[j];
      const d = Ai[i] !== 0.0 ? Ai[i] : 1e-300;
      x[i] = s / d;
    }
    return x;
  }

  function sumsq(r) {
    let s = 0.0;
    for (let i = 0; i < r.length; i++) s += r[i] * r[i];
    return s;
  }

  function lmSolve(fun, z0, xtol = 1e-13, ftol = 1e-13, maxIter = 200) {
    let z = Array.from(z0);
    const n = z.length;
    let r = Array.from(fun(z.slice()));
    const m = r.length;
    let cost = sumsq(r);
    let nfev = 1, lam = -1.0, nit = 0;
    const eps = 1.4901161193847656e-08;
    for (let it = 1; it <= maxIter; it++) {
      nit = it;
      const J = [];
      for (let k = 0; k < m; k++) J.push(new Array(n).fill(0.0));
      for (let i = 0; i < n; i++) {
        const h = eps * Math.max(1.0, Math.abs(z[i]));
        const zz = z.slice();
        zz[i] = z[i] + h;
        const hh = zz[i] - z[i];
        const ri = fun(zz);
        nfev += 1;
        for (let k = 0; k < m; k++) J[k][i] = (ri[k] - r[k]) / hh;
      }
      const A = [];
      for (let i = 0; i < n; i++) A.push(new Array(n).fill(0.0));
      const g = new Array(n).fill(0.0);
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          let s = 0.0;
          for (let k = 0; k < m; k++) s += J[k][i] * J[k][j];
          A[i][j] = s;
          A[j][i] = s;
        }
        let s = 0.0;
        for (let k = 0; k < m; k++) s += J[k][i] * r[k];
        g[i] = s;
      }
      if (lam < 0) lam = 1e-3;
      let accepted = false, done = false;
      for (let tries = 0; tries < 60; tries++) {
        const M = A.map(row => row.slice());
        for (let i = 0; i < n; i++) M[i][i] = A[i][i] + lam * Math.max(A[i][i], 1e-12);
        const dz = solveLinear(M, g.map(v => -v));
        const zn = z.map((v, i) => v + dz[i]);
        const rn = Array.from(fun(zn.slice()));
        nfev += 1;
        const cn = sumsq(rn);
        if (cn === cn && cn <= cost) {
          let step = 0.0, zmax = 0.0;
          for (let i = 0; i < n; i++) {
            if (Math.abs(dz[i]) > step) step = Math.abs(dz[i]);
            if (Math.abs(z[i]) > zmax) zmax = Math.abs(z[i]);
          }
          const smallF = (cost - cn) <= ftol * cost;
          z = zn; r = rn;
          cost = cn;
          lam = lam / 3.0;
          accepted = true;
          done = step <= xtol * (1.0 + zmax) || smallF || cost === 0.0;
          break;
        }
        lam = lam * 4.0;
      }
      if (!accepted || done) break;
    }
    return { x: z, fun: r, nfev, nit };
  }
  IM.lmSolve = lmSolve;
  IM.solveLinear = solveLinear;
})(typeof globalThis !== 'undefined' ? globalThis : this);
