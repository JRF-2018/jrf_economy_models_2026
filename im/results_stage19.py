# results_stage19.py: 段 19 個人として生きる箱庭の準備(プレイヤーの記入)。python -W ignore results_stage19.py > results_stage19.txt
# 一国、標本 2000 記入、300 年、種 8 個。プレイヤーの記入の人数は 0.01(全体の約 1000 の中の一人の世帯)。
import os
import numpy as np
import im1
from concurrent.futures import ProcessPoolExecutor

BASE = dict(households='continuous', Ns=2000, g_gamma=0.0, credit=True, land=True, c_h=0.005, crisis_track=True)
SEEDS = range(8)
T = 300


def run(args):
    kw, act, seed = args
    e = im1.Im1(seed=seed, **dict(BASE, **kw))
    e.act = dict(act)
    try:
        rec = [e.step() for _ in range(T)]
    except (ValueError, FloatingPointError, AssertionError, np.linalg.LinAlgError):
        return None
    g = lambda k: np.array([r.get(k, np.nan) for r in rec], float)
    votes = [r for r in rec if 'pl_pivot' in r]
    hi = [r['tau_k'] == e.p.tau_hi for r in rec[1:]]
    out = dict(E=g('E')[100:].mean(), hi=np.mean([r['tau_k'] == e.p.tau_hi for r in rec[100:]]),
               pivot=np.mean([r['pl_pivot'] for r in votes]) if votes else np.nan,
               books=np.nanmax(np.maximum(g('books'), g('sample_gap')) / g('Y')))
    if e.pl is not None:
        out.update(nw_rel=g('pl_nw_rel')[-1], nw_rel_mean=g('pl_nw_rel')[100:].mean(), hungry=g('pl_hungry')[100:].mean(),
                   gen=rec[-1]['pl_gen'], rentier=np.mean([r['pl_rentier'] for r in rec[100:]]))
    return out


def table(title, cases, cols):
    print(title)
    jobs = [(kw, act, s) for lab, kw, act in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('IM_WORKERS', 4))) as ex:
        res = list(ex.map(run, jobs))
    print('  場合                                   | ' + ' | '.join(c[0] for c in cols))
    for i, (lab, kw, act) in enumerate(cases):
        rs = [r for r in res[i * len(SEEDS):(i + 1) * len(SEEDS)] if r is not None]
        if not rs:
            print(f'  {lab:36s} | すべて崩れた'); continue
        m = {k: np.nanmean([r.get(k, np.nan) for r in rs]) for k in rs[0]}
        se = {k: np.nanstd([r.get(k, np.nan) for r in rs]) / np.sqrt(len(rs)) for k in rs[0]}
        print(f'  {lab:36s} | ' + ' | '.join(f"{m.get(k, np.nan):{f}}±{se.get(k, np.nan):{f}}" for _, k, f in cols)
              + (f'  ({len(rs)}/{len(SEEDS)})' if len(rs) < len(SEEDS) else ''))
    print(f"  (帳簿と標本の差/所得の最大 {max(r['books'] for r in res if r is not None):.0e})")


def p1_downs():
    cols = [('高い税の年の割合', 'hi', '.2f'), ('プレイヤーが決め手になった投票の割合', 'pivot', '.3f'), ('雇用', 'E', '.0f')]
    cases = []
    for rw in (1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3):
        V = dict(vote=True, rho_w=rw, rho_k=1e-2, vote_noise=True)
        cases += [(f'ρ_w = {rw:g}、プレイヤーなし', V, {}),
                  (f'ρ_w = {rw:g}、一票だけ(高い税)', dict(V, player=True), dict(vote='hi')),
                  (f'ρ_w = {rw:g}、一票 + 呼びかけ(k 1e-4)', dict(V, player=True, k_sway=1e-4), dict(vote='hi', campaign=1.0)),
                  (f'ρ_w = {rw:g}、一票 + 呼びかけ(k 1e-3)', dict(V, player=True, k_sway=1e-3), dict(vote='hi', campaign=1.0))]
    table('P1 ダウンズの投票(段 5 の V1 の相転移、資産家 ρ_k = 1e-2): プレイヤーの一票と、呼びかけ(自分の群れの相関 ρ に k × 呼びかけ を足す見本の効き目)。'
          '決め手 = プレイヤーがいなければ結果が逆だった投票。票の揺れあり(vote_noise)', cases, cols)


def p2_life():
    cols = [('300 年目の正味資産(平均との比)', 'nw_rel', '.2f'), ('100 年目からの平均', 'nw_rel_mean', '.2f'), ('生存の量に届かない年の割合', 'hungry', '.3f'),
            ('代の数', 'gen', '.1f'), ('資産家だった年の割合', 'rentier', '.2f'), ('雇用', 'E', '.0f')]
    L = dict(player=True, ret_age=45, pension='payg', child_cost=0.5)
    cases = [('規則どおり(支出 1、働く)', L, dict()),
             ('倹約(支出 0.7)', L, dict(spend=0.7)),
             ('浪費(支出 1.3)', L, dict(spend=1.3)),
             ('半分だけ働く', L, dict(work=0.5)),
             ('倹約、代替わりで財産を手放す(くじ)', dict(L, player_heir='rule'), dict(spend=0.7)),
             ('倹約、運あり(σ 0.2)', dict(L, luck_sigma=0.2), dict(spend=0.7))]
    table('P2 一つの暮らし(引退 45 年、賦課方式の年金、養育の費用 0.5): プレイヤーの選び方と、代替わりで財産を継ぐか', cases, cols)


if __name__ == '__main__':
    p1_downs()
    p2_life()
