# results_stage10b.py: 段 10b 積立の年金で株を買う。python -W ignore results_stage10b.py > results_stage10b.txt
# 段 10 の A1 と同じ経済(引退 45 年、寿命 40 年)。標本 2000 記入、300 年の後半 200 年、種 12 個。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor
from results_stage10 import BASE, SEEDS, T, BURN

L = dict(ret_age=45, turnover=0.025)


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **L, **kw))
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(BURN, T)
    Y = g('Y')[n]
    return dict(E=g('E')[n].mean(), sdE=g('E')[n].std(), hungry=g('hungry')[n].mean(), h_ret=g('hungry_ret')[n].mean(),
                pf=np.nanmean(g('pf')[n] / Y), b=g('b')[n].mean(), tau=g('tau')[n].mean(), inv=np.mean(g('inv')[n] / Y),
                gini=g('gini_nw')[n].mean(), fe=np.nanmean(g('fund_e')[n]) if 'fund_e' in rec[-1] else 0.0,
                qg=np.nanmean(g('q_gap')[n]) if 'q_gap' in rec[-1] else 0.0,
                books=np.nanmax(np.maximum(g('books'), g('sample_gap')) / g('Y')))


if __name__ == '__main__':
    cols = [('雇用', 'E', '.0f'), ('雇用の揺れ', 'sdE', '.0f'), ('届かない人', 'hungry', '.3f'), ('引退した人のうち', 'h_ret', '.3f'),
            ('積立/所得', 'pf', '.2f'), ('基金の持分', 'fe', '.2f'), ('株の開き(対数)', 'qg', '+.2f'), ('投資/所得', 'inv', '.3f'),
            ('国債/所得', 'b', '.2f'), ('賃金への税率', 'tau', '.2f'), ('富のジニ', 'gini', '.3f')]
    D = dict(eq_market=True, eq_pricing='demand')
    cases = [('賦課方式(比べる元)', dict(pension='payg')),
             ('積立、預金だけ(段 10)', dict(pension='funded')),
             ('積立、預金だけ、株の市場あり', dict(pension='funded', **D)),
             ('積立、株 3 割', dict(pension='funded', fund_eq=0.3, **D)),
             ('積立、株 6 割', dict(pension='funded', fund_eq=0.6, **D)),
             ('積立、株 6 割、投資が q に反応 0.02', dict(pension='funded', fund_eq=0.6, phi_q=0.02, **D)),
             ('積立、株 6 割、投資が q に反応 0.05', dict(pension='funded', fund_eq=0.6, phi_q=0.05, **D)),
             ('積立、株 6 割、基本の値打ちで値段', dict(pension='funded', fund_eq=0.6, eq_market=True)),
             ('賦課方式、株の市場あり', dict(pension='payg', **D)),
             ('賦課方式、株の市場あり、q に反応 0.05', dict(pension='payg', phi_q=0.05, **D))]
    print('B1 積立の年金の基金が株を買う(需要で値段のつく株式の市場に、基金も買い手として加わる)と、株の値段の開きを通じて投資が増える道(phi_q)')
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    print('  場合                                 | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not rs:
            print(f'  {lab:36s} | すべて崩れた'); continue
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:36s} | ' + ' | '.join(f"{m[k]:{f}}±{se[k]:{f}}" for _, k, f in cols) + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿と標本の差/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")
