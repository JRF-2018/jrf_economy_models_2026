# results_stage8.py: 段 8 の実験(文化と多様性: 贅沢品の種類の数 n = (知財 / 400)^ψ)。python -W ignore results_stage8.py > results_stage8.txt
# 一国、標本 2000 記入、400 年の後半 300 年、乱数の種 12 個。財政は既定、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor
from results_stage5b import tail

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005)
SEEDS = range(12)
T, BURN = 400, 100


def one(args):
    kw, seed, shock = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    rec = []
    try:
        for t in range(T):
            if shock and t == 300:
                e.K[0] *= 0.5
            rec.append(e.step())
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(BURN, T) if not shock else slice(290, 330)
    sv = kw.get('sigma_v', 3.0)
    nv = g('n_var') if 'n_var' in rec[0] else np.ones(T)
    xL = g('lux_eff') / nv ** (1 / (sv - 1)) if 'lux_eff' in rec[0] else None
    # 実質の消費(効用の物差し): 必需品 + 贅沢品の実効の量 × 基準の相対価格。種類の効き目を入れたものと、入れないもの
    lux_q = g('xL')
    out = dict(lux=lux_q[n].mean(), lux_eff=(lux_q * nv ** (1 / (sv - 1)))[n].mean(), E=g('E')[n].mean(), rw=g('real_wage')[n].mean(), top10i=g('top10_inc')[n].mean(), books=np.nanmax(g('books') / g('Y')),
               ipt=g('ip_tot')[n].mean(), n_var=np.nanmean(nv[n]), bsh=np.nanmean(g('b_share')[n]) if 'b_share' in rec[0] else 0.3,
               prod=np.log(g('gamma_front')[BURN] / g('gamma_front')[T - 1]) / (T - 1 - BURN),
               Rip=np.nanmean(np.log1p(g('R_ip')[n].clip(-0.99)) - np.log1p(g('infl')[n])) if 'R_ip' in rec[0] else np.nan)
    if shock:
        out.update(n_min=np.nanmin(nv[300:330]) / np.nanmean(nv[280:300]), n_end=nv[329] / np.nanmean(nv[280:300]),
                   E_min=g('E')[300:330].min() / g('E')[280:300].mean())
    return out


def table(title, cases, shock=False, cols=None):
    print(title)
    jobs = [(kw, s, shock) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    cols = cols or [('雇用', 'E', '.0f'), ('実質賃金', 'rw', '.3f'), ('種類の数', 'n_var', '.2f'), ('必需品の割合', 'bsh', '.3f'),
                    ('知財の総量', 'ipt', '.0f'), ('贅沢品の量', 'lux', '.0f'), ('贅沢品の実効の量', 'lux_eff', '.0f'), ('所得の上位10%', 'top10i', '.3f')]
    print('  場合                          | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:28s} | ' + ' | '.join(f"{m[k]:{f}}±{se[k]:{f}}" for _, k, f in cols)
              + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")


def v1_grid():
    cases = [('ψ = 0(一種類)', dict())]
    for psi in (0.5, 1.0):
        for eps in (1.0, 2.0, 3.0):
            cases.append((f'ψ={psi} ε={eps}', dict(psi_v=psi, eps_v=eps)))
    table('V1 文化の効果 ψ と、必需品と贅沢品の間の代わりの利きやすさ ε。ε = 1 なら支出の割合は変わらず、種類は効用だけに効く', cases)


def v2_ip():
    cols = [('雇用', 'E', '.0f'), ('種類の数', 'n_var', '.2f'), ('知財の総量', 'ipt', '.0f'), ('知財の持ち主の実質の利回り', 'Rip', '+.3f')]
    cases = (('知財の市場、ψ = 0', dict(ip_market=True)), ('知財の市場、ψ=1 ε=2', dict(ip_market=True, psi_v=1.0, eps_v=2.0)),
             ('知財の市場、ψ=1 ε=3', dict(ip_market=True, psi_v=1.0, eps_v=3.0)))
    table('V2 知財の市場(段 2b、需要で値段)と文化: 種類が増えると、知財の持ち主の損は和らぐか', cases, cols=cols)


def v3_tech():
    R = dict(research=True, s_R=0.02, **tail(1.5))
    cols = [('雇用', 'E', '.0f'), ('実質賃金(必需品で測る)', 'rw', '.3f'), ('技術の進み', 'prod', '.4f'), ('種類の数', 'n_var', '.2f'), ('必需品の割合', 'bsh', '.3f'),
            ('贅沢品の量', 'lux', '.0f'), ('贅沢品の実効の量(種類の効き目を入れる)', 'lux_eff', '.0f'), ('所得の上位10%', 'top10i', '.3f')]
    cases = (('技術なし、文化なし', dict()), ('技術(研究 2%)', R), ('文化(ψ=1 ε=2)', dict(psi_v=1.0, eps_v=2.0)),
             ('技術 + 文化', dict(R, psi_v=1.0, eps_v=2.0)), ('技術 + 文化(ε=3)', dict(R, psi_v=1.0, eps_v=3.0)))
    table('V3 文化と技術(gj_6 の「文化と技術の対立」): 研究(労働を節約)と、種類(労働を吸う)', cases, cols=cols)


def v4_crisis():
    cols = [('種類の数の底(打撃の前に対する比)', 'n_min', '.3f'), ('30 年後の種類の数', 'n_end', '.3f'), ('雇用の底', 'E_min', '.3f')]
    cases = (('ψ=1 ε=1', dict(psi_v=1.0)), ('ψ=1 ε=2', dict(psi_v=1.0, eps_v=2.0)), ('ψ=1 ε=3', dict(psi_v=1.0, eps_v=3.0)))
    table('V4 危機と文化: 300 年目に必需品の能力が半分。贅沢品の支出が削られ、創作(知財)が減って、種類の数が落ちるか', cases, shock=True, cols=cols)


if __name__ == '__main__':
    v1_grid()
    v2_ip()
    v3_tech()
    v4_crisis()
