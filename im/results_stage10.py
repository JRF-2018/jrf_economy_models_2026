# results_stage10.py: 段 10 の実験(年齢と年金と BI、児童手当)。python -W ignore results_stage10.py > results_stage10.txt
# 一国、標本 2000 記入、300 年の後半 200 年、乱数の種 12 個。財政は既定(利払いの重さ)、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, crisis_track=True)
SEEDS = range(12)
T, BURN = 300, 100


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(BURN, T)
    Y = g('Y')[n]
    tr = (np.nan_to_num(g('pensions')[n]) + np.nan_to_num(g('child_allow')[n]) + g('transfer')[n]) / Y
    return dict(E=g('E')[n].mean(), rw=g('real_wage')[n].mean(), hungry=g('hungry')[n].mean(), h_ret=g('hungry_ret')[n].mean(),
                h_kids=g('hungry_kids')[n].mean(), ret=np.nanmean(g('ret_share')[n]) if 'ret_share' in rec[0] else 0.0,
                tau=g('tau')[n].mean(), b=g('b')[n].mean(), gini=g('gini_nw')[n].mean(), top10i=g('top10_inc')[n].mean(),
                pen=np.nanmean(g('pensions')[n] / Y) if 'pensions' in rec[0] else 0.0, pf=np.nanmean(g('pf')[n] / Y) if 'pf' in rec[0] else 0.0,
                trY=np.mean(tr), vat=np.nanmean(g('vat')[n] / Y) if 'vat' in rec[0] else 0.0, brk=g('breakups')[n].sum() / (T - BURN) * 100,
                books=np.nanmax(np.maximum(g('books'), g('sample_gap')) / g('Y')))


def table(title, cases, cols):
    print(title)
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    print('  場合                              | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not rs:
            print(f'  {lab:32s} | すべて崩れた'); continue
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:32s} | ' + ' | '.join(f"{m[k]:{f}}±{se[k]:{f}}" for _, k, f in cols) + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿と標本の差/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")


def a1_pension():
    cols = [('引退した人の割合', 'ret', '.2f'), ('雇用', 'E', '.0f'), ('実質賃金', 'rw', '.3f'), ('生存の量に届かない人', 'hungry', '.3f'),
            ('引退した人のうち', 'h_ret', '.3f'), ('年金/所得', 'pen', '.3f'), ('積立/所得', 'pf', '.2f'), ('賃金への税率', 'tau', '.2f'),
            ('国債/所得', 'b', '.2f'), ('富のジニ', 'gini', '.3f')]
    cases = [('引退なし(基準)', dict())]
    for to, lab in ((0.025, '寿命 40 年'), (0.02, '寿命 50 年(高齢化)')):
        for pen in ('none', 'payg', 'points', 'funded'):
            cases.append((f'{lab}、{pen}', dict(ret_age=45, pension=pen, turnover=to)))
    table('A1 引退(世帯が始まって 45 年で引退)と年金の方式。賦課方式は働く人一人あたりの賃金の 4 割、貢献比例は同じ総額を労働所得の累計に比例して、'
          '積立は労働所得の 1 割を積み、引退してから毎年残高の 1 割を取り崩す', cases, cols)


def a2_bi():
    cols = [('雇用', 'E', '.0f'), ('生存の量に届かない人', 'hungry', '.3f'), ('子を育てる世帯のうち', 'h_kids', '.3f'), ('一家離散/100年', 'brk', '.0f'),
            ('移転/所得', 'trY', '.3f'), ('消費税/所得', 'vat', '.3f'), ('賃金への税率', 'tau', '.2f'), ('国債/所得', 'b', '.2f'), ('富のジニ', 'gini', '.3f'),
            ('所得の上位10%', 'top10i', '.3f')]
    C = dict(child_cost=1.0)
    cases = (('養育の費用 1.0、手当てなし', C),
             ('BI 2%(国債と財政のルール)', dict(C, g_tr=0.02)),
             ('BI 2%(消費税で)', dict(C, g_tr=0.02, vat_bi=True)),
             ('BI 6%(消費税で)', dict(C, g_tr=0.06, vat_bi=True)),
             ('児童手当 0.5', dict(C, child_allow=0.5)),
             ('児童手当 1.0', dict(C, child_allow=1.0)),
             ('失業の給付をやめ、BI 2%(消費税で)', dict(C, benefit=0.0, g_tr=0.02, vat_bi=True)),
             ('失業の給付をやめ、児童手当 1.0', dict(C, benefit=0.0, child_allow=1.0)))
    table('A2 BI(一人あたり同額の給付)と、児童手当(子を育てる世帯に、一人あたり生存の量の値段の何倍か)。養育の費用 1.0(段 9 の L9-5・L9-6 の続き)', cases, cols)


if __name__ == '__main__':
    a1_pension()
    a2_bi()
