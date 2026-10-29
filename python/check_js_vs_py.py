# site/js/model の JavaScript の模型が、Python の参照実装(im、rng='portable'・solver='portable')と同じ数を出すことを確かめる。
# JavaScript は quickjs(Python から使える小さな JS エンジン)で動かす。
#   python python/check_js_vs_py.py [rng solver im1 im3 ...]
import json
import os
import sys
import time

import numpy as np
import quickjs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for cand in (os.path.join(ROOT, 'im'), os.path.join(ROOT, '..', 'work_4')):
    if os.path.exists(os.path.join(cand, 'im1.py')):
        sys.path.insert(0, cand)
        break
import portable  # noqa: E402

# 手元では 1e-9 以内で合う。機械ごとに numpy の exp・log の最後の桁が違うので、合否は 1e-8 で決める(2026-10-29)
TOL = 1e-8
JS_FILES = ['rng.js', 'solver.js', 'im_core.js', 'im1.js', 'im3.js']


def context():
    ctx = quickjs.Context()
    ctx.set_memory_limit(1 << 30)
    ctx.set_time_limit(-1)
    for f in JS_FILES:
        p = os.path.join(ROOT, 'site', 'js', 'model', f)
        if os.path.exists(p):
            ctx.eval(open(p, encoding='utf-8').read())
    return ctx


def js(ctx, expr):
    return json.loads(ctx.eval(f'JSON.stringify((function(){{ {expr} }})())'))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    d[both_nan] = 0.0
    return float(np.nanmax(np.where(np.isnan(d), np.inf, d)))


# ---------------------------------------------------------------- 乱数
def check_rng(ctx):
    worst = 0.0
    for seed in (0, 1, 7, 12345, 4294967295):
        r = portable.PortableRNG(seed)
        py = dict(u32=[r.next32() for _ in range(50)], u=list(r.random(50)), n=list(r.standard_normal(30)),
                  e=list(r.exponential(2.0, 10)), p=list(r.pareto(1.5, 20)), t=list(r.standard_t(5, 5)),
                  po=[r.poisson(l) for l in (0.0, 0.3, 4.0, 29.0, 31.0, 500.0)], perm=list(map(int, r.permutation(17))),
                  ch=[r.choice(5, [0.1, 0.2, 0.3, 0.25, 0.15]) for _ in range(10)], it=[r.integers(9) for _ in range(10)],
                  st=r.get_state())
        o = js(ctx, f"""const r = new IM.PortableRNG({seed});
            const u32 = []; for (let i = 0; i < 50; i++) u32.push(r.next32());
            return {{u32, u: Array.from(r.random(50)), n: Array.from(r.standard_normal(30)), e: Array.from(r.exponential(2.0, 10)),
                    p: Array.from(r.pareto(1.5, 20)), t: Array.from(r.standard_t(5, 5)),
                    po: [0.0, 0.3, 4.0, 29.0, 31.0, 500.0].map(l => r.poisson(l)), perm: r.permutation(17),
                    ch: Array.from({{length: 10}}, () => r.choice(5, [0.1, 0.2, 0.3, 0.25, 0.15])),
                    it: Array.from({{length: 10}}, () => r.integers(9)), st: r.getState()}};""")
        for k in py:
            if k in ('u32', 'po', 'perm', 'ch', 'it', 'st'):
                assert list(map(int, py[k])) == list(map(int, o[k])), (seed, k, py[k][:5], o[k][:5])
            else:
                worst = max(worst, rel(py[k], o[k]))
    print(f'rng     5 seeds, integer streams identical, float max rel. diff {worst:.2e}')
    return worst


# ---------------------------------------------------------------- 解き方
SOLVER_CASES = {
    'rosenbrock': ('[10 * (z[1] - z[0] * z[0]), 1 - z[0]]', lambda z: np.array([10 * (z[1] - z[0] ** 2), 1 - z[0]]), [-1.2, 1.0]),
    'exp fit + tactic': ('[Math.exp(z[0]) - 3, Math.exp(z[1]) * Math.exp(z[0]) - 8, Math.tanh(z[2]) + z[0] - 0.5, 0.01 * (z[0] - 1), 0.01 * (z[1] - 2), 0.01 * (z[2])]',
                         lambda z: np.array([np.exp(z[0]) - 3, np.exp(z[1]) * np.exp(z[0]) - 8, np.tanh(z[2]) + z[0] - 0.5,
                                             0.01 * (z[0] - 1), 0.01 * (z[1] - 2), 0.01 * z[2]]), [0.0, 0.0, 0.0]),
    'powell singular': ('[z[0] + 10 * z[1], Math.sqrt(5) * (z[2] - z[3]), (z[1] - 2 * z[2]) ** 2, Math.sqrt(10) * (z[0] - z[3]) ** 2]',
                        lambda z: np.array([z[0] + 10 * z[1], np.sqrt(5) * (z[2] - z[3]), (z[1] - 2 * z[2]) ** 2, np.sqrt(10) * (z[0] - z[3]) ** 2]),
                        [3.0, -1.0, 0.0, 1.0]),
}


def check_solver(ctx):
    worst = 0.0
    for name, (jsf, pyf, z0) in SOLVER_CASES.items():
        s = portable.lm_solve(pyf, np.array(z0))
        o = js(ctx, f'const s = IM.lmSolve(z => {jsf}, {json.dumps(z0)}); return s;')
        d = rel(s.x, o['x'])
        worst = max(worst, d)
        print(f'solver  {name:18s} nit py {s.nit} js {o["nit"]}  nfev py {s.nfev} js {o["nfev"]}  x rel. diff {d:.2e}  |r|^2 {np.sum(s.fun ** 2):.2e}')
    return worst


def main(which):
    ctx = context()
    t0 = time.time()
    worst = {}
    if 'rng' in which:
        worst['rng'] = check_rng(ctx)
    if 'solver' in which:
        worst['solver'] = check_solver(ctx)
    for k in ('im1', 'im3'):
        if k in which and k in CHECKS:
            worst[k] = CHECKS[k](ctx)
    print(f'worst {max(worst.values()):.2e} (tol {TOL:g})  {time.time() - t0:.1f}s')
    ok = max(worst.values()) < TOL
    print('OK' if ok else 'FAIL')
    return ok


# ---------------------------------------------------------------- 一国の模型
PORT = dict(rng='portable', solver='portable', breakup_eps=1e-9)
BASE1 = dict(households='continuous', Ns=200, g_gamma=0.0, credit=True, land=True, c_h=0.005)
SCEN1 = {
    'two types, no credit': (dict(households='two'), 100, [], []),
    'base (credit, land)': (dict(BASE1, seed=1), 200, [], []),
    'disaster idio, relief, rate hike': (dict(BASE1, seed=2, disaster=True, dis_rate=0.05, dis_scope='idio', relief='state', protect=0.01),
                                         150, [(50, dict(r_cb=0.05))], []),
    'taylor, levy, wealth tax, equal': (dict(BASE1, seed=3, phi_pi=0.5, resolve='levy', theta_resolve=0.3, wealth_tax=0.01,
                                             inherit='equal', tau_k=0.3, redistrib=0.5), 150, [], []),
    'default path, mu_B, deficit rule': (dict(BASE1, seed=4, fiscal='deficit', mu_B=0.05, theta_B=0.2, resolve='default',
                                              theta_resolve=0.3, g_tr=0.02), 150, [], []),
    'euphoria, mutual relief, queue': (dict(BASE1, seed=5, disaster=True, relief='mutual'), 120,
                                       [(30, dict(theta_shock=0.1)), (45, dict(theta_shock=0.0))], [(40, 0.3, 'common'), (70, 0.2, 'idio')]),
    # 段 B4: 部品ごとの場面
    'B4 crisis: cap, last producer, stock': (dict(BASE1, seed=6, disaster=True, crisis_track=True, inventory=True, crisis_rule='last',
                                                  g_stock=0.05, g_pw=0.02, f_K=0.05, delta_scrap=0.1), 120,
                                             [(40, dict(cap_N=2.2)), (80, dict(cap_N=0.0))], [(40, 0.4, 'common')]),
    'B4 crisis: loss sharing, cap': (dict(BASE1, seed=7, crisis_rule='share', cap_N=2.4, delta_scrap=0.1), 100, [], []),
    'B4 learn, vote with noise': (dict(BASE1, seed=8, disaster=True, dis_rate=0.08, learn=True, learn_omega=0.95, vote=True, vote_noise=True,
                                       rho_w=0.01, rho_k=0.02), 150, [], []),
    'B4 research, culture': (dict(BASE1, seed=9, research=True, adaptive_R=True, psi_v=0.5, eps_v=2.0, culture_spill=0.5, g_gamma=0.0), 150, [], []),
    'B4 pension payg, child, luck': (dict(BASE1, seed=10, ret_age=45, pension='payg', child_cost=0.3, child_allow=0.5,
                                          luck_sigma=0.1, luck_skill=0.05, luck_rho=0.9), 150, [], []),
    'B4 pension points, luck t': (dict(BASE1, seed=11, ret_age=45, pension='points', luck_sigma=0.1, luck_nu=5), 120, [], []),
    'B4 pension funded': (dict(BASE1, seed=12, ret_age=45, pension='funded'), 120, [], []),
    'B4 age tracked, pension switched on later': (dict(BASE1, seed=15, age_track=True, child_cost=0.3), 100, [(30, dict(ret_age=45, pension='payg'))], []),
    'B6 player: spend, lux, work, child, vote': (dict(BASE1, seed=13, player=True, child_cost=0.3, ret_age=45, pension='payg', vote=True,
                                                      vote_noise=True, rho_w=0.01, rho_k=0.02, luck_sigma=0.1), 150, [], [],
                                                 [(5, dict(spend=0.8, lux=0.2, work=0.7)), (15, dict(child=True)), (20, dict(vote='hi', campaign=1.0)),
                                                  (60, dict(spend=1.2, lux=0.9, work=1.0, vote='lo', campaign=0.0)), (90, dict(child=True))]),
    'B7 revolution (low threshold), disasters, crisis tracked': (dict(BASE1, seed=16, revolt=True, rev_thr=0.5, disaster=True, dis_rate=0.08, crisis_track=True,
                                                                       vote=True, vote_noise=True), 150, [], [(60, 0.4, 'common')]),
    'B7 workshop: subsidy, later reseed': (dict(BASE1, seed=17), 150, [(20, dict(up_gen=1, up_sub=0.5)), (60, dict(up_sub=0.0)), (100, dict(up_gen=2, up_s0=0.05))], []),
    'B6 player in crisis world': (dict(BASE1, seed=14, player=True, crisis_track=True, cap_N=2.3, crisis_rule='last', child_cost=0.5), 100, [], [],
                                  [(10, dict(lux=0.0, spend=0.6))]),
}


def py_run1(kw, T, shocks, queue, acts=()):
    import im1
    kw = dict(kw)
    hh, Ns, seed = kw.pop('households', 'two'), kw.pop('Ns', 500), kw.pop('seed', 0)
    e = im1.Im1(households=hh, Ns=Ns, seed=seed, **PORT, **kw)
    e.shock_queue = [tuple(q) for q in queue]
    for t in range(T):
        for ts, ch in shocks:
            if ts == t:
                e.p.set(**ch)
        for ta, a in acts:
            if ta == t:
                e.act.update(a)
        e.step()
    return e.log


def js_run1(ctx, kw, T, shocks, queue, acts=()):
    kw = dict(kw)
    hh, Ns, seed = kw.pop('households', 'two'), kw.pop('Ns', 500), kw.pop('seed', 0)
    return js(ctx, f'''const e = new IM.Im1({{households: {json.dumps(hh)}, Ns: {Ns}, seed: {seed}, kw: {json.dumps(dict(kw, breakup_eps=1e-9))}}});
        e.shock_queue = {json.dumps(queue)};
        const sh = {json.dumps(shocks)}, ac = {json.dumps(list(acts))};
        for (let t = 0; t < {T}; t++) {{ for (const [ts, ch] of sh) if (ts === t) Object.assign(e.p, ch);
          for (const [ta, a] of ac) if (ta === t) Object.assign(e.act, a); e.step(); }}
        return e.log;''')


# 水準の量(0 の近くを通らない)。これは各年の相対のずれで測る
LEVELS = ('W', 'PN', 'PL', 'PI', 'bN', 'bL', 'bI', 'real_wage', 'S', 'D', 'E', 'xN', 'xL', 'xI', 'KN', 'KL', 'KI', 'Y', 'tau',
          'pT', 'land_value', 'gini_nw', 'gini_inc', 'top10_inc', 'top10_nw', 'loans', 'ip_tot', 'e_cur')


# 収支の量(大きな二つの流れの差。変動の為替では 0 の近くに解かれる)は、流れの大きさ(輸出)に対する相対で測る
def ref_scale(k, py):
    base = k.split('.')[-1]
    pre = k[:-len(base)]
    if (base in ('tb', 'fxpos', 'interbank', 'ib') or base.startswith('bil_')) and pre + 'exports' in py[0]:
        return float(np.max(np.abs([float(r[pre + 'exports']) for r in py])))
    if base in ('wo_h', 'bailout', 'relief', 'K_lost', 'defaulted', 'levy', 'subsidy', 'share_pay', 'scrap', 'gov_buy', 'S_G', 'InvN',
                'hh_debt', 'prot', 'prot_L', 'pensions', 'child_allow', 'pf', 'L_R', 'res_w', 'acct_explore'):
        # 額の量(0 から始まったり、丸めの大きさの 1e-14 が出たりする)は、所得の大きさに対する相対で測る
        return float(np.max(np.abs([float(r[pre + 'Y']) for r in py])))
    if base in ('short', 'hungry', 'sub_gap', 'hungry_kids', 'hungry_ret'):
        return 1.0                                  # 割合の量(0 の近くでは丸めの 1e-13 が出る)
    if base in ('vote_hi', 'vote_lo', 'turnout_w', 'turnout_k'):
        # 票の数は、賭け金(一人あたりの配りものと資本所得の差)から来る近い数の差の量。人口(人数の単位)に対して測る
        return 1000.0 if base.startswith('vote') else 1.0
    if base == 'tb_Y' and pre + 'exports' in py[0]:
        return float(np.max(np.abs([float(r[pre + 'exports']) / float(r[pre + 'Y']) for r in py])))
    return 0.0


def series_diff(a, b, pointwise=False, ref=0.0):
    """pointwise なら各年の相対のずれ |a − b| / max(|a|, |b|)。そうでなければ、系列の大きさに対する相対のずれ
    |a − b| / max(系列の |a| の最大)(0 の近くを通る量: 物価上昇率、利潤率、赤字、国債など。近い数の差の量は、
    各年の相対のずれが意味を持たない)。"""
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    fin = np.isfinite(a)
    scale = max(float(np.max(np.abs(a[fin]))) if fin.any() else 0.0, ref, 1e-300)
    den = np.maximum(np.abs(a), np.abs(b)) if pointwise else scale
    d = np.abs(a - b) / np.maximum(den, 1e-300)
    d[np.isnan(a) & np.isnan(b)] = 0.0
    return float(np.max(np.where(np.isnan(d), np.inf, d)))


# 分母が 0 の近くを通りうる比の量: 預金のジニ(分母は預金の和。再分配や年金で預金の総額が 0 の近くを通ると、桁が落ちる)。
# どの場面でも、許容 TOL_ILL_KEY で数える(GitHub Actions の機械では numpy の exp・log の最後の桁が手元と違い、
# 投票と年金の三国の場面で 1.7e-9 になった。2026-10-29)
ILL_KEYS = ('gini_dep',)
TOL_ILL_KEY = 1e-6


def compare_logs(py, jl, label):
    if isinstance(py[0], list):          # 世界の模型: 地域ごとの記録を、名前に地域をつけて並べる
        py = [{f'{i}.{k}': v for i, r in enumerate(rs) for k, v in r.items()} for rs in py]
        jl = [{f'{i}.{k}': v for i, r in enumerate(rs) for k, v in r.items()} for rs in jl]
    skip = ('books', 'sample_gap', 'land_err', 'land_sum_err', 'resid', 'books_tol', 'F_gap')
    keys = [k for k in jl[0] if k in py[0] and isinstance(py[0][k], (int, float, np.floating, np.integer, bool, np.bool_))
            and k.split('.')[-1] not in skip]
    worst, wk, worst_p, wkp, worst_i, wki = 0.0, None, 0.0, None, 0.0, None
    for k in keys:
        a = np.array([float(r[k]) for r in py]); b = np.array([np.nan if r[k] is None else float(r[k]) for r in jl])
        d = series_diff(a, b, ref=ref_scale(k, py))
        if k.split('.')[-1] in ILL_KEYS:
            if d > worst_i:
                worst_i, wki = d, k
            continue
        if d > worst:
            worst, wk = d, k
        if k.split('.')[-1] in LEVELS:
            d = series_diff(a, b, pointwise=True)
            if d > worst_p:
                worst_p, wkp = d, k
    bk = [k for k in jl[0] if k.split('.')[-1] == 'books']
    tk = [k for k in jl[0] if k.split('.')[-1] == 'books_tol']
    books = max(float(r[bk[0]]) / float(r[tk[0]]) for r in jl) if tk else max(float(r[bk[0]]) for r in jl)
    missing = [k for k in py[0] if k not in jl[0]]
    print(f'{label:40s} {len(keys):3d} keys  scale-rel. {worst:.1e} ({wk})  levels pointwise {worst_p:.1e} ({wkp})  JS books/tol {books:.1e}'
          + (f'  ill-conditioned {worst_i:.1e} ({wki}, tol {TOL_ILL_KEY:g})' if wki else ''))
    return max(worst, worst_p, worst_i * TOL / TOL_ILL_KEY), missing


def check_im1(ctx):
    worst = 0.0
    for name, (kw, T, shocks, queue, *rest) in SCEN1.items():
        acts = rest[0] if rest else ()
        t0 = time.time(); py = py_run1(kw, T, shocks, queue, acts); tp = time.time() - t0
        t0 = time.time(); jl = js_run1(ctx, kw, T, shocks, queue, acts); tj = time.time() - t0
        w, missing = compare_logs(py, jl, f'im1  {name}')
        print(f'      py {tp:.1f}s  quickjs {tj:.1f}s' + (f'  (ill-conditioned: tolerance {TOL_UNSTABLE[name]:g})' if name in TOL_UNSTABLE else ''))
        worst = max(worst, w * TOL / TOL_UNSTABLE[name] if name in TOL_UNSTABLE else w)
    print('      keys only in Python:', ' '.join(missing))
    return worst


# ---------------------------------------------------------------- 世界の模型(三つの国)
BASEW = dict(g_gamma=0.0, credit=True, land=True, c_h=0.005, disaster=True)
SCEN3 = {
    'three, common currency': (dict(regions=['A', 'B', 'C'], Ns=120, seeds={'A': 0, 'B': 1, 'C': 2}, common=BASEW), 80,
                               [(20, 'A', dict(r_cb=0.05))], {'B': [(30, 0.3, 'common')]}),
    'three, float, capital': (dict(regions=['A', 'B', 'C'], Ns=120, seeds={'A': 0, 'B': 1, 'C': 2}, currency='float', capital=True,
                                   common=BASEW, region_kw={'B': dict(theta_F0=0.05)}), 80, [(25, 'C', dict(wealth_tax=0.01))], {}),
    'A+B union, C float, capital': (dict(regions=['A', 'B', 'C'], Ns=120, seeds={'A': 3, 'B': 4, 'C': 5}, currency={'A': 'x', 'B': 'x', 'C': 'y'},
                                         regimes={'y': 'float'}, capital=True, union=dict(members=['A', 'B'], psi=0.3),
                                         common=BASEW, region_kw={'C': dict(theta_F0=0.05, phi_K=1.0)}), 80, [],
                                    {'A': [(20, 0.4, 'idio')]}),
    'two, fixed': (dict(regions=['A', 'B'], Ns=120, seeds={'A': 0, 'B': 1}, currency='fixed', common=BASEW), 60, [], {}),
    'B4 three: vote, pension, learn, player in A': (dict(regions=['A', 'B', 'C'], Ns=120, seeds={'A': 0, 'B': 1, 'C': 2}, capital=True,
                                                        common=dict(BASEW, theta_F0=0.03, vote=True, vote_noise=True, rho_w=0.01, rho_k=0.02,
                                                                    ret_age=45, pension='payg', child_cost=0.3, learn=True, dis_rate=0.05,
                                                                    research=True, psi_v=0.5, eps_v=2.0),
                                                        region_kw={'A': dict(player=True), 'C': dict(pension='points', child_allow=0.5)}),
                                                   100, [(30, 'B', dict(r_cb=0.05))], {},
                                                   [(5, 'A', dict(spend=0.8, lux=0.2, work=0.7)), (12, 'A', dict(child=True)),
                                                    (20, 'A', dict(vote='hi', campaign=1.0))]),
    'B7 three: revolution in A, workshop in B': (dict(regions=['A', 'B', 'C'], Ns=120, seeds={'A': 0, 'B': 1, 'C': 2}, capital=True,
                                                     common=dict(BASEW, theta_F0=0.03),
                                                     region_kw={'A': dict(revolt=True, rev_thr=0.5, dis_rate=0.05)}),
                                                120, [(30, 'B', dict(up_gen=1, up_sub=0.5))], {}),
    # 三国の危機時の生産(_cap_rel: その年の必需品の価格に対する倍で、価格の上限を置く)
    'crisis three, common: cap in A, last, stock, works': (dict(regions=['A', 'B', 'C'], Ns=120, seeds={'A': 0, 'B': 1, 'C': 2}, capital=True,
                                                               common=dict(BASEW, theta_F0=0.03),
                                                               region_kw={'A': dict(crisis_rule='last', g_stock=0.05, g_pw=0.02, inventory=True, f_K=0.03),
                                                                          'B': dict(crisis_track=True)}),
                                                          80, [(40, 'A', dict(_cap_rel=0.9)), (65, 'A', dict(cap_N=0.0))], {'B': [(50, 0.3, 'common')]}),
    'crisis three, float: cap in C, share, inventory': (dict(regions=['A', 'B', 'C'], Ns=120, seeds={'A': 0, 'B': 1, 'C': 2}, currency='float',
                                                             common=BASEW, region_kw={'C': dict(crisis_rule='share', inventory=True, delta_scrap=0.1)}),
                                                        80, [(40, 'C', dict(_cap_rel=0.9))], {}),
}


def py_run3(kw, T, events, queues, acts=()):
    import im3
    kw = dict(kw)
    kw['common'] = dict(kw.get('common', {}), **PORT)
    w = im3.World(**kw)
    for nm, q in queues.items():
        w.R[w.names.index(nm)].shock_queue = [tuple(x) for x in q]
    for t in range(T):
        for ts, reg, ch in events:
            if ts == t:
                for nm, e in zip(w.names, w.R):
                    if reg is None or reg == nm:
                        ch2 = dict(ch)
                        if '_cap_rel' in ch2:
                            ch2['cap_N'] = ch2.pop('_cap_rel') * e.PN_prev
                        e.p.set(**ch2)
        for ta, reg, a in acts:
            if ta == t:
                w.R[w.names.index(reg)].act.update(a)
        w.step()
    return w.log


def js_run3(ctx, kw, T, events, queues, acts=()):
    kw = dict(kw)
    kw['common'] = dict(kw.get('common', {}), breakup_eps=1e-9)
    return js(ctx, f'''const w = new IM.World({json.dumps(kw)});
        const qs = {json.dumps(queues)};
        for (const nm in qs) w.R[w.names.indexOf(nm)].shock_queue = qs[nm];
        const ev = {json.dumps(events)};
        for (let t = 0; t < {T}; t++) {{
          for (const [ts, reg, ch] of ev) if (ts === t) w.R.forEach((e, i) => {{ if (reg === null || reg === w.names[i]) {{
            const c2 = Object.assign({{}}, ch); if ('_cap_rel' in c2) {{ c2.cap_N = c2._cap_rel * e.PN_prev; delete c2._cap_rel; }} Object.assign(e.p, c2); }} }});
          for (const [ta, reg, a] of {json.dumps(list(acts))}) if (ta === t) Object.assign(w.R[w.names.indexOf(reg)].act, a);
          w.step();
        }}
        return w.log;''')


# 不安定な道の場面: 丸めのずれが年を追って育つので、許容を広げる(JRF 2026-10-29: Python と JS のずれはそこまで厳密でなくてよい。
# README とノートに注を書く)。C の通貨が崩れる 60 年目から育ち、80 年目で 6e-8 ほど
TOL_UNSTABLE = {'crisis three, float: cap in C, share, inventory': 1e-6,
                # 革命が 15 年ごとに資本の税と再分配を入れ替えると、預金の総額が 0 の近くを通り、預金のジニ(分母が預金の和)が 82 などの
                # 意味のない値になって桁が落ちる。ほかの量は 1e-9 で合う
                'B7 revolution (low threshold), disasters, crisis tracked': 1e-6}


def check_im3(ctx):
    worst = 0.0
    for name, (kw, T, events, queues, *rest) in SCEN3.items():
        acts = rest[0] if rest else ()
        t0 = time.time(); py = py_run3(kw, T, events, queues, acts); tp = time.time() - t0
        t0 = time.time(); jl = js_run3(ctx, kw, T, events, queues, acts); tj = time.time() - t0
        w, missing = compare_logs(py, jl, f'im3  {name}')
        print(f'      py {tp:.1f}s  quickjs {tj:.1f}s' + (f'  (unstable path: tolerance {TOL_UNSTABLE[name]:g})' if name in TOL_UNSTABLE else ''))
        worst = max(worst, w * TOL / TOL_UNSTABLE[name] if name in TOL_UNSTABLE else w)   # 広げた許容に対する比で数える
    return worst


CHECKS = {'im1': check_im1, 'im3': check_im3}

if __name__ == '__main__':
    which = sys.argv[1:] or ['rng', 'solver', 'im1', 'im3']
    sys.exit(0 if main(which) else 1)
