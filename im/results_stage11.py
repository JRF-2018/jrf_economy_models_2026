# results_stage11.py: 段 11 の実験(自動化と再分配)。python -W ignore results_stage11.py > results_stage11.txt
# 一国、標本 2000 記入、400 年の後半 300 年、乱数の種 12 個。研究は売上の 2%、当たりの裾 α = 1.5(段 5 の探索と同じ)。財政は既定、g_gamma = 0。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor
from results_stage5b import tail

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, research=True, s_R=0.02, **tail(1.5),
            phi_u=0.1, phi_r=0.25)   # 投資の反応を既定(0.3、0.5)より穏やかにする(既定のままでは自動化で資本の係数が上がると発散する。D0)
SEEDS = range(12)
T, BURN = 400, 100


def one(args):
    kw, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    n = slice(BURN, T)
    last = slice(T - 50, T)
    out = dict(prod=np.log(g('gamma_front')[BURN] / g('gamma_front')[T - 1]) / (T - 1 - BURN), E=g('E')[last].mean(), E_all=g('E')[n].mean(),
               rw_g=np.log(g('real_wage')[T - 1] / g('real_wage')[BURN]) / (T - 1 - BURN), ws=g('wage_share')[last].mean(),
               capY=g('cap_Y')[last].mean(), top10i=g('top10_inc')[last].mean(), top10=g('top10_nw')[last].mean(), kap=g('kap_avg')[T - 1],
               tau_k=g('tau_k')[last].mean(), books=np.nanmax(np.maximum(g('books'), g('sample_gap')) / g('Y')))
    if 'tax_high' in rec[0]:
        out['high'] = np.nanmean(g('tax_high')[n])
    return out


def table(title, cases):
    print(title)
    jobs = [(kw, s) for lab, kw in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(one, jobs))
    cols = [('技術の進み', 'prod', '.4f'), ('雇用(最後の50年)', 'E', '.0f'), ('雇用(後半300年)', 'E_all', '.0f'), ('実質賃金の伸び', 'rw_g', '+.4f'),
            ('労働の取り分(最後の50年)', 'ws', '.3f'), ('資本/所得', 'capY', '.2f'), ('資本の係数(400年目)', 'kap', '.2f'),
            ('所得の上位10%', 'top10i', '.3f'), ('富の上位10%', 'top10', '.3f'), ('資本所得への税率', 'tau_k', '.2f')]
    print('  場合                              | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not rs:
            print(f'  {lab:32s} | すべて崩れた'); continue
        m = {k: np.nanmean([r[k] for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r[k] for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:32s} | ' + ' | '.join(f"{m[k]:{f}}±{se[k]:{f.lstrip('+')}}" for _, k, f in cols) + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿と標本の差/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")


def d0_harrod():
    D = dict(phi_u=0.3, phi_r=0.5)
    cases = (('既定の投資の反応、自動化なし', D), ('既定の投資の反応、自動化 0.5', dict(D, auto_share=0.5)), ('既定の投資の反応、自動化 1.0', dict(D, auto_share=1.0)),
             ('φ_u = 0.15、自動化 1.0', dict(phi_u=0.15, phi_r=0.5, auto_share=1.0)))
    table('D0 投資の反応(稼働率 φ_u、利潤率 φ_r)が既定のままだと、自動化で資本の係数が上がるにつれ、投資の波が大きくなって発散する(ハロッドの不安定)', cases)


def d1_auto():
    cases = [('研究なし', dict(research=False))]
    for a in (0.0, 0.5, 1.0):
        cases.append((f'自動化の割合 {a}', dict(auto_share=a)))
    cases.append(('自動化 1.0、資本の係数の上がり方 0.5', dict(auto_share=1.0, auto_kappa=0.5)))
    cases.append(('自動化 1.0、資本の係数の上がり方 2', dict(auto_share=1.0, auto_kappa=2.0)))
    table('D1(投資の反応 φ_u = 0.1、φ_r = 0.25)研究の当たりのうち、仕事を置き換える(自動化の)割合。自動化の当たりは、労働の係数を下げ、新しい能力の資本の係数を上げる(費用が下がるときだけ使う)', cases)


def d2_vote():
    V = dict(vote=True, rho_w=1e-3, rho_k=1e-4)
    cases = (('自動化なし、投票なし', dict()), ('自動化 1.0、投票なし', dict(auto_share=1.0)),
             ('自動化なし、投票(資本所得の税 0.1 か 0.5)', V), ('自動化 1.0、投票', dict(V, auto_share=1.0)))
    table('D2 自動化と政治: 10 年ごとに資本所得への税(0.1 か 0.5、税収は一人あたり同額で配る)を投票で決める(段 5 の V1〜V3 と同じ形)', cases)


def d3_culture():
    Cu = dict(psi_v=1.0, eps_v=2.0)
    cases = (('自動化なし', dict()), ('自動化 1.0', dict(auto_share=1.0)), ('自動化なし + 文化', Cu), ('自動化 1.0 + 文化', dict(Cu, auto_share=1.0)))
    table('D3 自動化と、新しい種類(段 8 の文化。新しい仕事の受け皿になるか。Acemoglu and Restrepo 2018 の「復帰の効果」の代わり)', cases)


if __name__ == '__main__':
    d0_harrod()
    d1_auto()
    d2_vote()
    d3_culture()
