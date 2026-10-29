# results_stage5c.py: 段 5 の世界樹の案(部門の中に、規則的に研究する企業と、真似る企業を置く)。
# python -W ignore results_stage5c.py > results_stage5c.txt
# 標本 2000 記入、400 年の後半 300 年、乱数の種 12 個。研究は売上の 2%(規則的な企業)、当たりの裾 α = 1.5(平均 0.006)。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor
from results_stage5b import tail

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, s_R=0.02)
F20 = dict(firms=20)
SEEDS = range(12)
T, BURN = 400, 100


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **tail(1.5), **kw)) if 'alpha_R' not in kw else im1.Im1(seed=seed, **dict(BASE, **kw))
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(BURN, T)
    yrs = T - 1 - BURN
    out = dict(prod=np.log(g('gamma_front')[BURN] / g('gamma_front')[T - 1]) / yrs,
               rw=np.log(g('real_wage')[T - 1] / g('real_wage')[BURN]) / yrs, E=g('E')[n].mean(), top10i=g('top10_inc')[n].mean(),
               books=np.nanmax(g('books') / g('Y')))
    if 'rule_share' in rec[0]:
        gr, gl = g('gain_rule')[n].sum(), g('gain_learn')[n].sum()
        out.update(rule=g('rule_share')[-100:].mean(), hhi=g('hhi')[n].mean(), rr=np.nanmean(g('rate_rule')[n]),
                   rl=np.nanmean(g('rate_learn')[n]), exits=(g('exits_rule')[T - 1] + g('exits_learn')[T - 1] - g('exits_rule')[BURN] - g('exits_learn')[BURN]) / yrs * 100,
                   hits_rule=gr / max(gr + gl, 1e-12), sd_rule=np.std(g('rule_share')[n]))
    return out


def table(title, cases):
    print(title)
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    print('  場合                          | 技術の進み | 実質賃金の伸び | 雇用 | 所得の上位10% | 規則的な企業の占有率(最後の100年) | その揺れ | 集中(HHI) | 利益の率 規則的/真似る | 当たりのうち規則的な企業の分 | 退出/100年')
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        m = {k: np.nanmean([r.get(k, np.nan) for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r.get(k, np.nan) for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        fr = (f" | {m['rule']:.2f}±{se['rule']:.2f} | {m['sd_rule']:.2f} | {m['hhi']:.2f} | {m['rr']:.3f}/{m['rl']:.3f} | {m['hits_rule']:.2f} | {m['exits']:.0f}"
              if 'rule' in m else ' |  —  |  —  |  —  |  —  |  —  |  — ')
        print(f"  {lab:28s} | {m['prod']:.4f}±{se['prod']:.4f} | {m['rw']:+.4f} | {m['E']:.0f} | {m['top10i']:.3f}{fr}"
              + (f'  ({len(rs)}/{len(SEEDS)} の種だけ最後まで)' if len(rs) < len(SEEDS) else ''))


def w1_basic():
    table('W1 部門に企業が一つの場合と、20 社の場合。型の混ぜ方と参入の仕方。「選ぶ」= 参入の型を、型ごとの占有率あたりの利益で選ぶ(突然変異 5%)。'
          '「固定」= 参入はいつも初めの割合',
          (('企業は一つ(段 5 の探索)', dict()),
           ('20 社、すべて規則的', dict(F20, rule_share0=1.0, entry='fixed')),
           ('20 社、すべて真似る', dict(F20, rule_share0=0.0, entry='fixed')),
           ('20 社、半々、参入を選ぶ', dict(F20)),
           ('20 社、半々、参入は固定', dict(F20, entry='fixed')),
           ('20 社、半々、真似が遅い ι=0.1', dict(F20, iota_L=0.1)),
           ('20 社、半々、真似が速い ι=0.6', dict(F20, iota_L=0.6))))


def w2_support():
    table('W2 規則的な企業を支えるもの(20 社、半々、参入を選ぶ)。知財の保護 = 他社の知識は ip_lag 年前のものしか真似られない。'
          '公の研究所 = 政府が所得の s_pub の研究の人を雇い、当たりは誰でも遅れなく真似られる。長い目の株主 = 退出の閾値を −0.05 → −0.3',
          (('支えなし', dict(F20)),
           ('知財の保護 5 年', dict(F20, ip_lag=5)),
           ('知財の保護 10 年', dict(F20, ip_lag=10)),
           ('知財の保護 20 年', dict(F20, ip_lag=20)),
           ('公の研究所 1%', dict(F20, s_pub=0.01)),
           ('公の研究所 3%', dict(F20, s_pub=0.03)),
           ('長い目の株主', dict(F20, r_exit=-0.3)),
           ('真似る企業も少し研究 0.5%', dict(F20, s_L=0.005)),
           ('知財の保護 10 年 + 公の研究所 1%', dict(F20, ip_lag=10, s_pub=0.01))))


def w4_monopoly():
    table('W4 集中の値上げ(集中した部門は、利益の率の目標が r̄ + 0.2 × (HHI − 1/20) になる)を入れて、支えを比べる',
          (('支えなし', dict(F20, mu_hhi=0.2)),
           ('知財の保護 10 年', dict(F20, ip_lag=10, mu_hhi=0.2)),
           ('知財の保護 5 年', dict(F20, ip_lag=5, mu_hhi=0.2)),
           ('公の研究所 3%', dict(F20, s_pub=0.03, mu_hhi=0.2)),
           ('20 社、すべて規則的', dict(F20, rule_share0=1.0, entry='fixed', mu_hhi=0.2))))


def w3_tail():
    cases = []
    for a in (3.0, 1.5, 1.1):
        cases.append((f'α={a} 支えなし', dict(F20, **tail(a))))
        cases.append((f'α={a} 知財の保護 10 年', dict(F20, ip_lag=10, **tail(a))))
    table('W3 当たりの裾(平均を揃えて裾だけを変える)と、規則的な企業の生き残り', cases)


if __name__ == '__main__':
    w1_basic()
    w2_support()
    w3_tail()
    w4_monopoly()
