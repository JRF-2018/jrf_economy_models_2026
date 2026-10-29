# results_stage9.py: 段 9 の実験(家計の大バクチと養育の費用)。python -W ignore results_stage9.py > results_stage9.txt
# 一国、標本 2000 記入、400 年の後半 300 年(分布は最後の年)、乱数の種 12 個。財政は既定、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, crisis_track=True)
SEEDS = range(12)
T, BURN = 400, 100


def hill(x, w, q=0.05):
    """上位 q の人数の、一人あたりの正味資産のパレートの指数(ヒルの推定)。"""
    o = np.argsort(-x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    top = cw <= q
    if top.sum() < 5:
        return np.nan
    xt, wt = x[top], w[top]
    xm = xt[-1]
    return float(wt.sum() / np.sum(wt * np.log(xt / xm)))


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(BURN, T)
    H = e.H
    nw = (H.M + e.pT * H.land + (e.qIP or 0.0) * H.ip + e.v_eq_last * H.eq) / H.heads
    o = np.argsort(-nw); cw = np.cumsum(H.heads[o]) / H.heads.sum()
    top1 = float((nw[o] * H.heads[o])[cw <= 0.01].sum() / max((nw * H.heads).sum(), 1e-12))
    eqpc = H.eq / H.heads
    return dict(top10=g('top10_nw')[n].mean(), gini=g('gini_nw')[n].mean(), top1=top1, alpha=hill(np.maximum(nw, 1e-9), H.heads),
                alpha_eq=hill(np.maximum(eqpc, 1e-12), H.heads), top10i=g('top10_inc')[n].mean(), E=g('E')[n].mean(),
                hungry=g('hungry')[n].mean(), hungry_kids=g('hungry_kids')[n].mean(), brk=g('breakups')[n].sum() / (T - BURN) * 100,
                rw=g('real_wage')[n].mean(), books=np.nanmax(g('books') / g('Y')), gap=np.nanmax(g('sample_gap') / g('Y')))


def table(title, cases, cols):
    print(title)
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    print('  場合                              | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:32s} | ' + ' | '.join(f"{m[k]:{f}}±{se[k]:{f}}" for _, k, f in cols) + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿と標本の差/所得の最大 {max(max(r['books'], r['gap']) for r in res if r is not None):.0e})")


DIST = [('富の上位10%', 'top10', '.3f'), ('富の上位1%', 'top1', '.3f'), ('富のジニ', 'gini', '.3f'), ('富の裾の指数(上位5%、ヒル)', 'alpha', '.2f'),
        ('持分の裾の指数', 'alpha_eq', '.2f'), ('所得の上位10%', 'top10i', '.3f'), ('雇用', 'E', '.0f')]


def l1_tail():
    cases = [('初めにパレート、運なし(基準)', dict()), ('初めに平等、運なし', dict(eq_init='equal'))]
    for sg in (0.1, 0.2, 0.4):
        cases.append((f'初めに平等、運 σ={sg}(正規)', dict(eq_init='equal', luck_sigma=sg)))
    cases.append(('初めに平等、運 σ=0.2(t 分布 ν=3)', dict(eq_init='equal', luck_sigma=0.2, luck_nu=3.0)))
    cases.append(('初めに平等、運 σ=0.2、入れ替わり 5%', dict(eq_init='equal', luck_sigma=0.2, turnover=0.05)))
    cases.append(('初めに平等、運 σ=0.2、相続は等分', dict(eq_init='equal', luck_sigma=0.2, inherit='equal')))
    cases.append(('初めにパレート、運 σ=0.2', dict(luck_sigma=0.2)))
    table('L1 富の裾は内から出るか: 初めの持分を平等にして、家計ごとの運(持分の取り分の対数の揺れ)を入れる。分布は 400 年目(裾の指数)と後半 300 年の平均', cases, DIST)


def l2_skill():
    cols = DIST + [('一家離散/100年', 'brk', '.0f')]
    S = dict(eq_init='equal', luck_sigma=0.1)
    cases = (('運だけ(σ=0.1)', S), ('運 + 腕(標準偏差 0.03、持続 0.95)', dict(S, luck_skill=0.03, luck_rho=0.95)),
             ('運だけ + 富裕税 1%', dict(S, wealth_tax=0.01)), ('運 + 腕 + 富裕税 1%', dict(S, luck_skill=0.03, luck_rho=0.95, wealth_tax=0.01)),
             ('運だけ + 資本所得の税 0.5', dict(S, tau_k=0.5)), ('運 + 腕 + 資本所得の税 0.5', dict(S, luck_skill=0.03, luck_rho=0.95, tau_k=0.5)))
    table('L2 運用の腕(持続する利回りの差)と、富裕税・資本所得の税(Guvenen ほか 2023 の論点)', cases, cols)


def l3_child():
    cols = [('生存の量に届かない人', 'hungry', '.3f'), ('子を育てる世帯のうち届かない人', 'hungry_kids', '.3f'), ('一家離散/100年', 'brk', '.0f'),
            ('富のジニ', 'gini', '.3f'), ('実質賃金', 'rw', '.3f'), ('雇用', 'E', '.0f')]
    cases = []
    for cc in (0.0, 0.5, 1.0):
        cases.append((f'養育の費用 {cc}', dict(child_cost=cc)))
    cases.append(('養育の費用 1.0 + 給付 2%', dict(child_cost=1.0, g_tr=0.02)))
    cases.append(('養育の費用 1.0、入れ替わり 5%', dict(child_cost=1.0, turnover=0.05)))
    cases.append(('養育の費用 1.0、相続は等分', dict(child_cost=1.0, inherit='equal')))
    table('L3 養育の費用: 新しく始まる世帯は 20 年、生存の量が一人 q0 × (1 + 費用)。生存の量に届かない人と一家離散', cases, cols)


if __name__ == '__main__':
    l1_tail()
    l2_skill()
    l3_child()
